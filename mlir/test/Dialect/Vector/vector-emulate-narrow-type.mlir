// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=4 memref-load-bitwidth=8" %s | FileCheck %s

// Expect no conversions, i8 is supported.
// CHECK-LABEL: func @vector_load_i8
// CHECK-SAME:  (%[[ARG:.*]]: memref<3x4xi8>, %[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
// CHECK-NEXT:  [[L:%.+]] = vector.load %[[ARG]][%[[IDX0]], %[[IDX1]]] : memref<3x4xi8>, vector<4xi8>
// CHECK-NEXT:  return
func.func @vector_load_i8(%arg0: memref<3x4xi8>, %arg1: index, %arg2: index) {
    %0 = vector.load %arg0[%arg1, %arg2] : memref<3x4xi8>, vector<4xi8>
    return
}

// -----

// CHECK-LABEL: func @vector_load_i4
// CHECK-SAME:  (%[[ARG:.*]]: memref<3x4xi8>, %[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
// CHECK-NEXT:  %[[CST:.*]] = arith.constant dense<0> : vector<3x4xi4>
// CHECK-NEXT:  %[[LOAD:.*]] = vector.load %[[ARG]][%[[IDX0]], %[[IDX1]]] : memref<3x4xi8>, vector<2xi8>
// CHECK-NEXT:  %[[BITCAST:.*]] = vector.bitcast %[[LOAD]] : vector<2xi8> to vector<4xi4>
// CHECK-NEXT:  %[[INSERT:.*]] = vector.insert %[[BITCAST]], %[[CST]] [0] : vector<4xi4> into vector<3x4xi4>
// CHECK-NEXT:  return
func.func @vector_load_i4(%arg0: memref<3x4xi4>, %arg1: index, %arg2: index) {
    %cst = arith.constant dense<0> : vector<3x4xi4>
    %0 = vector.load %arg0[%arg1, %arg2] : memref<3x4xi4>, vector<4xi4>
    %1 = vector.insert %0, %cst [0] : vector<4xi4> into vector<3x4xi4>
    return
}

// -----

// CHECK-LABEL: func @vector_load_i4_odd_number
// CHECK-SAME:  (%[[ARG:.*]]: memref<3x5xi8>, %[[IDX0:.*]]: index, %[[IDX1:.*]]: index)
// CHECK-NEXT:  %[[LOAD:.*]] = vector.load %[[ARG]][%[[IDX0]], %[[IDX1]]] : memref<3x5xi8>, vector<3xi8>
// CHECK-NEXT:  %[[BITCAST:.*]] = vector.bitcast %[[LOAD]] : vector<3xi8> to vector<6xi4>
// CHECK-NEXT:  %[[EXTRACT:.*]] = vector.extract_strided_slice %[[BITCAST]] {offsets = [0], sizes = [5], strides = [1]} : vector<6xi4> to vector<5xi4>
// CHECK-NEXT:  return
func.func @vector_load_i4_odd_number(%arg0: memref<3x5xi4>, %arg1: index, %arg2: index) {
    %0 = vector.load %arg0[%arg1, %arg2] : memref<3x5xi4>, vector<5xi4>
    return
}
