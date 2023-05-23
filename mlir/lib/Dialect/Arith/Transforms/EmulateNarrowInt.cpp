//===- EmulateNarrowInt.cpp - Narrow integer operation emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowIntEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

arith::NarrowIntEmulationConverter::NarrowIntEmulationConverter(
    unsigned targetWideInt)
    : targetBitwidth(targetWideInt) {
  assert(llvm::isPowerOf2_32(targetWideInt) &&
         "Only power-of-two integers are supported");
  assert(targetWideInt >= 8 && "Target integer type too narrow");

  // Allow unknown types.
  addConversion([](Type ty) -> std::optional<Type> { return ty; });

  // Scalar case.
  addConversion([this](IntegerType ty) -> std::optional<Type> {
    unsigned width = ty.getWidth();
    if (width >= targetBitwidth)
      return ty;
    else
      return IntegerType::get(ty.getContext(), targetBitwidth);

    return std::nullopt;
  });

  // Vector case.
  addConversion([this](VectorType ty) -> std::optional<Type> {
    auto intTy = dyn_cast<IntegerType>(ty.getElementType());
    if (!intTy)
      return ty;

    unsigned width = intTy.getWidth();
    if (width >= targetBitwidth)
      return ty;
    else
      return VectorType::get(to_vector(ty.getShape()),
                             IntegerType::get(ty.getContext(), targetBitwidth));

    return std::nullopt;
  });

  // Function case.
  addConversion([this](FunctionType ty) -> std::optional<Type> {
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return std::nullopt;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return std::nullopt;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateArithNarrowIntEmulationPatterns(
    NarrowIntEmulationConverter &typeConverter, RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}
