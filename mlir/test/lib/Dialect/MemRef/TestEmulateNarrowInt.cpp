//===- TestEmulateNarrowInt.cpp - Test Narrow Int Emulation  ------*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowIntEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TestEmulateNarrowIntPass
    : public PassWrapper<TestEmulateNarrowIntPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEmulateNarrowIntPass)

  TestEmulateNarrowIntPass() = default;
  TestEmulateNarrowIntPass(const TestEmulateNarrowIntPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect>();
  }
  StringRef getArgument() const final { return "test-emulate-narrow-int"; }
  StringRef getDescription() const final {
    return "Function pass to test Narrow Integer Emulation";
  }

  void runOnOperation() override {
    if (!llvm::isPowerOf2_32(targetWideInt) || targetWideInt < 8) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    arith::NarrowIntEmulationConverter typeConverter(targetWideInt);
    memref::populateMemRefNarrowIntEmulationConversions(typeConverter);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](Operation *op) {
      return typeConverter.isLegal(cast<func::FuncOp>(op).getFunctionType());
    });
    auto opLegalCallback = [&typeConverter](Operation *op) {
      return typeConverter.isLegal(op);
    };
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(opLegalCallback);
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, vector::VectorDialect, memref::MemRefDialect>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(ctx);

    // Add common pattenrs to support contants, functions, etc.
    arith::populateArithNarrowIntEmulationPatterns(typeConverter, patterns);
    memref::populateMemRefNarrowIntEmulationPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

  Option<unsigned> targetWideInt{*this, "target-wide-int",
                                 llvm::cl::desc("Target integer bit width"),
                                 llvm::cl::init(8)};
};
} // namespace

namespace mlir::test {
void registerTestEmulateNarrowIntPass() {
  PassRegistration<TestEmulateNarrowIntPass>();
}
} // namespace mlir::test
