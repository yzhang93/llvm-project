// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=8 memref-load-bitwidth=8" %s | FileCheck %s

// Expect no conversions.
// CHECK-LABEL: func @memref_i8
// CHECK:         [[M:%.+]] = memref.alloc() : memref<4xi8, 1>
// CHECK-NEXT:    [[V:%.+]] = memref.load [[M]][{{%.+}}] : memref<4xi8, 1>
// CHECK-NEXT:    memref.store {{%.+}}, [[M]][{{%.+}}] : memref<4xi8, 1>
// CHECK-NEXT:    return
func.func @memref_i8() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i8
    %m = memref.alloc() : memref<4xi8, 1>
    %v = memref.load %m[%c0] : memref<4xi8, 1>
    memref.store %c1, %m[%c0] : memref<4xi8, 1>
    return
}

// CHECK-LABEL: func @memref_load_i4
// CHECK-SAME:    (%[[ARG:.*]]: index)
// CHECK-NEXT:    %[[M:.*]]  = memref.alloc() : memref<4xi8>
// CHECK-NEXT:    %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]], %[[STRIDES:.*]] = memref.extract_strided_metadata %[[M]] : memref<4xi8> -> memref<i8>, index, index, index
// CHECK-NEXT:    %[[CI:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[LINEAR:.*]] = arith.muli %[[ARG]], %[[STRIDES]] : index
// CHECK-NEXT:    %[[INDEX:.*]] = arith.divui %[[LINEAR]], %[[CI]] : index
// CHECK-NEXT:    %[[AOFF:.*]] = arith.divui %[[OFFSET]], %[[CI]] : index
// CHECK-NEXT:    %[[CAST:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[AOFF]]], sizes: [%[[SIZES]]], strides: [%[[STRIDES]]] : memref<i8> to memref<4xi8, strided<[?], offset: ?>>
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %[[CAST]][%[[INDEX]]] : memref<4xi8, strided<[?], offset: ?>>
// CHECK-NEXT:    %[[I:.*]] = arith.index_castui %[[ARG]] : index to i8
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i8
// CHECK-NEXT:    %[[REM:.*]] = arith.remui %[[I]], %[[C2]] : i8
// CHECK-NEXT:    %[[STEP:.*]] = arith.muli %[[REM]], %[[C4]] : i8
// CHECK-NEXT:    %[[SHIFT:.*]] = arith.shrsi %[[LOAD]], %[[STEP]] : i8
// CHECK-NEXT:    %[[MASK:.*]] = arith.constant 15 : i8
// CHECK-NEXT:    %[[RES:.*]] = arith.andi %[[SHIFT]], %[[MASK]] : i8
// CHECK-NEXT:    return
func.func @memref_load_i4(%arg0: index) {
    %0 = memref.alloc() : memref<4xi4>
    %1 = memref.load %0[%arg0] : memref<4xi4>
    return
}


// CHECK-LABEL: func @memref_load_i4_rank2
// CHECK-SAME:    (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK-NEXT:    %[[M:.*]]  = memref.alloc() : memref<4x4xi8>
// CHECK-NEXT:    %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[M]] : memref<4x4xi8> -> memref<i8>, index, index, index, index, index
// CHECK-NEXT:    %[[CI:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[IDX1:.*]] = arith.muli %[[ARG0]], %[[STRIDES]]#0 : index
// CHECK-NEXT:    %[[IDX2:.*]] = arith.muli %[[ARG1]], %[[STRIDES]]#1 : index
// CHECK-NEXT:    %[[LINEAR:.*]] = arith.addi %[[IDX1]], %[[IDX2]] : index
// CHECK-NEXT:    %[[LSIZE:.*]] = arith.muli %[[SIZES]]#0, %[[SIZES]]#1 : index
// CHECK-NEXT:    %[[INDEX:.*]] = arith.divui %[[LINEAR]], %[[CI]] : index
// CHECK-NEXT:    %[[AOFF:.*]] = arith.divui %[[OFFSET]], %[[CI]] : index
// CHECK-NEXT:    %[[CAST:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[AOFF]]], sizes: [%[[LSIZE]]], strides: [%[[STRIDES]]#1] : memref<i8> to memref<16xi8, strided<[?], offset: ?>>
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %[[CAST]][%[[INDEX]]] : memref<16xi8, strided<[?], offset: ?>>
// CHECK-NEXT:    %[[I:.*]] = arith.index_castui %[[ARG1]] : index to i8
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i8
// CHECK-NEXT:    %[[REM:.*]] = arith.remui %[[I]], %[[C2]] : i8
// CHECK-NEXT:    %[[STEP:.*]] = arith.muli %[[REM]], %[[C4]] : i8
// CHECK-NEXT:    %[[SHIFT:.*]] = arith.shrsi %[[LOAD]], %[[STEP]] : i8
// CHECK-NEXT:    %[[MASK:.*]] = arith.constant 15 : i8
// CHECK-NEXT:    %[[RES:.*]] = arith.andi %[[SHIFT]], %[[MASK]] : i8
// CHECK-NEXT:    return
func.func @memref_load_i4_rank2(%arg0: index, %arg1: index) {
    %0 = memref.alloc() : memref<4x4xi4>
    %1 = memref.load %0[%arg0,%arg1] : memref<4x4xi4>
    return
}
