//===- EmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ConvertVectorLoad
//===----------------------------------------------------------------------===//

struct ConvertVectorLoad final : OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    Type newTy = getTypeConverter()->convertType(op.getType());
//    if (!newTy) {
//      return rewriter.notifyMatchFailure(
//          op->getLoc(),
//          llvm::formatv("failed to convert vector type: {0}", op.getType()));
//    }

    auto loc = op.getLoc();
    Type oldElementType = op.getType().getElementType();
    Type newElementType = getTypeConverter()->convertType<VectorType>(op.getType()).getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    auto newType =
        VectorType::get(op.getVectorType().getNumElements() / scale,
            newElementType);

    auto newLoad = rewriter.create<vector::LoadOp>(
        loc, newType, adaptor.getBase(), adaptor.getIndices());

    auto packedType =
        VectorType::get(newLoad.getVectorType().getNumElements() * scale,
            oldElementType);

    auto bitCast = rewriter.create<vector::BitCastOp>(
        loc, packedType, newLoad);

    rewriter.replaceOp(op, bitCast->getResult(0));

    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void vector::populateVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `vector.*` conversion patterns.
  patterns
      .add<ConvertVectorLoad>(typeConverter, patterns.getContext());
}

void vector::populateVectorNarrowTypeEmulationConversions(
    arith::NarrowTypeEmulationConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](VectorType ty) -> std::optional<Type> {
        auto intTy = dyn_cast<IntegerType>(ty.getElementType());
        if (!intTy)
          return ty;

        unsigned width = intTy.getWidth();
        unsigned loadStoreWidth = typeConverter.getLoadStoreBitwidth();
        if (width >= loadStoreWidth)
          return ty;

        auto newElemTy = IntegerType::get(ty.getContext(), loadStoreWidth,
                                          intTy.getSignedness());
        if (!newElemTy)
          return std::nullopt;

        return ty.cloneWith(std::nullopt, newElemTy);
      });
}
