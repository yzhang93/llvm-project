//===- EmulateNarrowInt.cpp - Narrow integer operation emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowIntEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns the offset of the value in `targetBits` representation.
///
/// `srcIdx` is an index into a 1-D array with each element having `sourceBits`.
/// It's assumed to be non-negative.
///
/// When accessing an element in the array treating as having elements of
/// `targetBits`, multiple values are loaded in the same time. The method
/// returns the offset where the `srcIdx` locates in the value. For example, if
/// `sourceBits` equals to 4 and `targetBits` equals to 8, the x-th element is
/// located at (x % 2) * 4. Because there are two elements in one i8, and one
/// element has 4 bits.
static Value getOffsetForBitwidth(Location loc, Value srcIdx, int sourceBits,
                                  int targetBits, OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  IntegerType targetType = builder.getIntegerType(targetBits);
  IntegerAttr idxAttr =
      builder.getIntegerAttr(targetType, targetBits / sourceBits);
  auto idx = builder.create<arith::ConstantOp>(loc, targetType, idxAttr);
  IntegerAttr srcBitsAttr = builder.getIntegerAttr(targetType, sourceBits);
  auto srcBitsValue =
      builder.create<arith::ConstantOp>(loc, targetType, srcBitsAttr);
  auto m = builder.create<arith::RemUIOp>(loc, srcIdx, idx);
  return builder.create<arith::MulIOp>(loc, targetType, m, srcBitsValue);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertMemRefAlloc
//===----------------------------------------------------------------------===//

struct ConvertMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, newTy, adaptor.getDynamicSizes(), adaptor.getSymbolOperands(),
        adaptor.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefLoad
//===----------------------------------------------------------------------===//

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));
    if (op.getMemRefType() == cast<MemRefType>(newTy))
      return success();

    auto loc = op.getLoc();
    Value source = adaptor.getMemref();
    auto sourceType = cast<MemRefType>(source.getType());
    auto srcElementType = sourceType.getElementType();
    unsigned sourceRank = sourceType.getRank();

    auto oldElementType =
        cast<MemRefType>(op.getMemref().getType()).getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = srcElementType.getIntOrFloatBitWidth();
    assert(dstBits % srcBits == 0);

    // The emulation only works on 1D memref types. To make this work on N-D
    // memref, we need to linearize the offset.
    // Specifically, %0 = memref.load %0[%v0][%v1] :
    // memref<?x?xi4, strided<[?, ?], offset = ?>> can be replaced with
    // %b, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %0
    // %linearized_offset = %v0 * %stride#0 + %scaled_v1 * %stride#1
    // where %scaled_v1 = v1 / targetBits * sourceBits
    // %linearized_size = %size0 * %size1
    // %linearized = memref.reinterpret_cast %b, offset = [%offset], sizes =
    // [%linearized_size], strides = [%stride#1] %load = memref.load
    // %linearized[%linearized_offset] : memref<?xi4, strided<?, offset = ?>>
    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, source);
    auto baseBuffer = stridedMetadata.getBaseBuffer();
    auto baseSizes = stridedMetadata.getSizes();
    auto baseStrides = stridedMetadata.getStrides();
    auto baseOffset = stridedMetadata.getOffset();

    SmallVector<Value> indices = adaptor.getIndices();
    assert(indices.size() == baseStrides.size());
    assert(indices.size() == sourceRank);

    // Only the last index is modified to load the bits needed.
    IndexType targetType = rewriter.getIndexType();
    IntegerAttr attr = rewriter.getIndexAttr(dstBits / srcBits);
    auto scaler = rewriter.create<arith::ConstantOp>(loc, targetType, attr);
    indices.back() =
        rewriter.create<arith::DivUIOp>(loc, indices.back(), scaler);

    SmallVector<Value> adjustOffsets;
    for (unsigned i = 0; i < sourceRank; ++i) {
      adjustOffsets.push_back(
          rewriter.create<arith::MulIOp>(loc, indices[i], baseStrides[i]));
    }

    // Linearize offset and sizes.
    Value linearizedOffset = adjustOffsets[0];
    Value linearizedSize = baseSizes[0];
    for (unsigned i = 1; i < sourceRank; ++i) {
      linearizedOffset = rewriter.create<arith::AddIOp>(loc, linearizedOffset,
                                                        adjustOffsets[i]);
      linearizedSize =
          rewriter.create<arith::MulIOp>(loc, linearizedSize, baseSizes[i]);
    }

    // Flatten n-D MemRef to 1-D MemRef.
    auto layoutAttr = StridedLayoutAttr::get(
        sourceType.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
    int64_t staticShape = sourceType.hasStaticShape()
                              ? sourceType.getNumElements()
                              : ShapedType::kDynamic;
    auto flattenMemrefType = MemRefType::get(
        staticShape, srcElementType, layoutAttr, sourceType.getMemorySpace());

    auto reinterpret = rewriter.create<memref::ReinterpretCastOp>(
        loc, flattenMemrefType, baseBuffer, baseOffset, linearizedSize,
        baseStrides.back());

    auto newLoad = rewriter.create<memref::LoadOp>(
        loc, srcElementType, reinterpret.getResult(), linearizedOffset,
        op.getNontemporal());

    // Get the offset and shift the bits to the rightmost.
    auto lastIdx = rewriter.create<arith::IndexCastUIOp>(
        loc, srcElementType, adaptor.getIndices().back());
    Value BitwidthOffset =
        getOffsetForBitwidth(loc, lastIdx, srcBits, dstBits, rewriter);
    auto bitsLoad =
        rewriter.create<arith::ShRSIOp>(loc, newLoad, BitwidthOffset);

    // Get the low bits by truncating the result.
    auto result =
        rewriter.create<arith::TruncIOp>(loc, oldElementType, bitsLoad);
    rewriter.replaceOp(op, result.getResult());

    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void memref::populateMemRefNarrowIntEmulationPatterns(
    arith::NarrowIntEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `memref.*` conversion patterns.
  patterns.add<ConvertMemRefAlloc, ConvertMemRefLoad>(typeConverter,
                                                      patterns.getContext());
}

void memref::populateMemRefNarrowIntEmulationConversions(
    arith::NarrowIntEmulationConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](MemRefType ty) -> std::optional<Type> {
        auto intTy = dyn_cast<IntegerType>(ty.getElementType());
        if (!intTy)
          return ty;

        unsigned width = intTy.getWidth();
        if (width >= typeConverter.targetBitwidth)
          return ty;
        else {
          Type newElemTy =
              IntegerType::get(ty.getContext(), typeConverter.targetBitwidth,
                               intTy.getSignedness());
          if (!newElemTy)
            return std::nullopt;
          return ty.cloneWith(std::nullopt, newElemTy);
        }
      });
}
