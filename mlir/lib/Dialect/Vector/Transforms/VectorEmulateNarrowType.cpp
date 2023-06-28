//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----*- C++
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

    auto loc = op.getLoc();
    Type oldElementType = op.getType().getElementType();
    Type newElementType =
        cast<MemRefType>(adaptor.getBase().getType()).getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // For example, to emulate i4 to i8, the following op:
    //
    // %1 = vector.load %0[%c0, %c0] : memref<3x4xi4>, vector<4xi4>
    //
    // can be replaced with
    //
    // %1 = vector.load %0[%c0, %c0] : memref<3x4xi8>, vector<2xi8>
    // %2 = vector.bitcast %1 : vector<2xi8> to vector<4xi4>
    //
    ArrayRef<int64_t> origShape = op.getVectorType().getShape();
    auto numElements = llvm::to_vector(origShape);
    numElements.back() = int(std::ceil(double(numElements.back()) / scale));

    auto newType = VectorType::get(numElements, newElementType);
    auto newLoad = rewriter.create<vector::LoadOp>(
        loc, newType, adaptor.getBase(), adaptor.getIndices());

    numElements.back() *= scale;
    auto castType = VectorType::get(numElements, oldElementType);
    auto bitCast = rewriter.create<vector::BitCastOp>(loc, castType, newLoad);

    // To deal with the odd number of elements at the last dimension.
    if (llvm::to_vector(origShape).back() % scale != 0) {
      SmallVector<int64_t> offsets(origShape.size(), 0);
      SmallVector<int64_t> strides(origShape.size(), 1);
      auto extractOp = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, bitCast, offsets, origShape, strides);
      rewriter.replaceOp(op, extractOp->getResult(0));
    } else {
      rewriter.replaceOp(op, bitCast->getResult(0));
    }

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