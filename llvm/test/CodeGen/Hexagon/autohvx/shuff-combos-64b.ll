; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Generator: vdeal(0x1f), vshuff(0x32), vshuff(0x24), vshuff(0x26), vshuff(0x08), vdeal(0x3a), vshuff(0x0c), vdeal(0x0e), vdeal(0x30), vdeal(0x22), vdeal(0x14), vdeal(0x36), vdeal(0x18), vdeal(0x0a), vdeal(0x3c)
; CHECK-LABEL: test_0000:
; CHECK-DAG: [[R00:r[0-9]+]] = #49
; CHECK-DAG: [[R01:r[0-9]+]] = #3
; CHECK: v[[H00:[0-9]+]]:[[L00:[0-9]+]] = vshuff(v1,v0,[[R00]])
; CHECK: v[[H01:[0-9]+]]:[[L01:[0-9]+]] = vdeal(v[[H00]],v[[L00]],[[R01]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0000(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 2, i32 32, i32 34, i32 4, i32 6, i32 36, i32 38, i32 8, i32 10, i32 40, i32 42, i32 12, i32 14, i32 44, i32 46, i32 1, i32 3, i32 33, i32 35, i32 5, i32 7, i32 37, i32 39, i32 9, i32 11, i32 41, i32 43, i32 13, i32 15, i32 45, i32 47, i32 16, i32 18, i32 48, i32 50, i32 20, i32 22, i32 52, i32 54, i32 24, i32 26, i32 56, i32 58, i32 28, i32 30, i32 60, i32 62, i32 17, i32 19, i32 49, i32 51, i32 21, i32 23, i32 53, i32 55, i32 25, i32 27, i32 57, i32 59, i32 29, i32 31, i32 61, i32 63, i32 64, i32 66, i32 96, i32 98, i32 68, i32 70, i32 100, i32 102, i32 72, i32 74, i32 104, i32 106, i32 76, i32 78, i32 108, i32 110, i32 65, i32 67, i32 97, i32 99, i32 69, i32 71, i32 101, i32 103, i32 73, i32 75, i32 105, i32 107, i32 77, i32 79, i32 109, i32 111, i32 80, i32 82, i32 112, i32 114, i32 84, i32 86, i32 116, i32 118, i32 88, i32 90, i32 120, i32 122, i32 92, i32 94, i32 124, i32 126, i32 81, i32 83, i32 113, i32 115, i32 85, i32 87, i32 117, i32 119, i32 89, i32 91, i32 121, i32 123, i32 93, i32 95, i32 125, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x1e), vshuff(0x00), vdeal(0x12), vshuff(0x34), vshuff(0x0b), vshuff(0x2b), vdeal(0x16), vshuff(0x2e), vshuff(0x1a), vdeal(0x28), vshuff(0x2d), vdeal(0x15), vdeal(0x1d), vshuff(0x25), vshuff(0x0d)
; CHECK-LABEL: test_0001:
; CHECK-DAG: [[R10:r[0-9]+]] = #10
; CHECK-DAG: [[R11:r[0-9]+]] = #34
; CHECK-DAG: [[R12:r[0-9]+]] = #16
; CHECK: v[[H10:[0-9]+]]:[[L10:[0-9]+]] = vshuff(v1,v0,[[R10]])
; CHECK: v[[H11:[0-9]+]]:[[L11:[0-9]+]] = vshuff(v[[H10]],v[[L10]],[[R11]])
; CHECK: v[[H12:[0-9]+]]:[[L12:[0-9]+]] = vshuff(v[[H11]],v[[L11]],[[R12]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0001(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 12, i32 13, i32 2, i32 3, i32 10, i32 11, i32 6, i32 7, i32 14, i32 15, i32 32, i32 33, i32 40, i32 41, i32 36, i32 37, i32 44, i32 45, i32 34, i32 35, i32 42, i32 43, i32 38, i32 39, i32 46, i32 47, i32 64, i32 65, i32 72, i32 73, i32 68, i32 69, i32 76, i32 77, i32 66, i32 67, i32 74, i32 75, i32 70, i32 71, i32 78, i32 79, i32 96, i32 97, i32 104, i32 105, i32 100, i32 101, i32 108, i32 109, i32 98, i32 99, i32 106, i32 107, i32 102, i32 103, i32 110, i32 111, i32 16, i32 17, i32 24, i32 25, i32 20, i32 21, i32 28, i32 29, i32 18, i32 19, i32 26, i32 27, i32 22, i32 23, i32 30, i32 31, i32 48, i32 49, i32 56, i32 57, i32 52, i32 53, i32 60, i32 61, i32 50, i32 51, i32 58, i32 59, i32 54, i32 55, i32 62, i32 63, i32 80, i32 81, i32 88, i32 89, i32 84, i32 85, i32 92, i32 93, i32 82, i32 83, i32 90, i32 91, i32 86, i32 87, i32 94, i32 95, i32 112, i32 113, i32 120, i32 121, i32 116, i32 117, i32 124, i32 125, i32 114, i32 115, i32 122, i32 123, i32 118, i32 119, i32 126, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x2c), vshuff(0x27), vshuff(0x07), vshuff(0x12), vdeal(0x04), vshuff(0x03), vshuff(0x23), vshuff(0x26), vdeal(0x06), vdeal(0x08), vdeal(0x01), vshuff(0x09), vdeal(0x11), vdeal(0x19), vshuff(0x21)
; CHECK-LABEL: test_0002:
; CHECK-DAG: [[R20:r[0-9]+]] = #5
; CHECK-DAG: [[R21:r[0-9]+]] = #18
; CHECK: v[[H20:[0-9]+]]:[[L20:[0-9]+]] = vdeal(v1,v0,[[R20]])
; CHECK: v[[H21:[0-9]+]]:[[L21:[0-9]+]] = vshuff(v[[H20]],v[[L20]],[[R21]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0002(<128 x i8> %v0) #0 {
; CHECK-NOT: v{{[0-9:]+}} =
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 4, i32 1, i32 5, i32 64, i32 68, i32 65, i32 69, i32 8, i32 12, i32 9, i32 13, i32 72, i32 76, i32 73, i32 77, i32 2, i32 6, i32 3, i32 7, i32 66, i32 70, i32 67, i32 71, i32 10, i32 14, i32 11, i32 15, i32 74, i32 78, i32 75, i32 79, i32 32, i32 36, i32 33, i32 37, i32 96, i32 100, i32 97, i32 101, i32 40, i32 44, i32 41, i32 45, i32 104, i32 108, i32 105, i32 109, i32 34, i32 38, i32 35, i32 39, i32 98, i32 102, i32 99, i32 103, i32 42, i32 46, i32 43, i32 47, i32 106, i32 110, i32 107, i32 111, i32 16, i32 20, i32 17, i32 21, i32 80, i32 84, i32 81, i32 85, i32 24, i32 28, i32 25, i32 29, i32 88, i32 92, i32 89, i32 93, i32 18, i32 22, i32 19, i32 23, i32 82, i32 86, i32 83, i32 87, i32 26, i32 30, i32 27, i32 31, i32 90, i32 94, i32 91, i32 95, i32 48, i32 52, i32 49, i32 53, i32 112, i32 116, i32 113, i32 117, i32 56, i32 60, i32 57, i32 61, i32 120, i32 124, i32 121, i32 125, i32 50, i32 54, i32 51, i32 55, i32 114, i32 118, i32 115, i32 119, i32 58, i32 62, i32 59, i32 63, i32 122, i32 126, i32 123, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x11), vshuff(0x2b), vdeal(0x3d), vdeal(0x3e), vshuff(0x02), vdeal(0x1c), vdeal(0x2f), vdeal(0x0f), vshuff(0x36), vshuff(0x38), vdeal(0x35), vshuff(0x1b), vshuff(0x3b), vdeal(0x21), vdeal(0x15)
; CHECK-LABEL: test_0003:
; CHECK-DAG: [[R30:r[0-9]+]] = #34
; CHECK-DAG: [[R31:r[0-9]+]] = #10
; CHECK-DAG: [[R32:r[0-9]+]] = #5
; CHECK: v[[H30:[0-9]+]]:[[L30:[0-9]+]] = vshuff(v1,v0,[[R30]])
; CHECK: v[[H31:[0-9]+]]:[[L31:[0-9]+]] = vdeal(v[[H30]],v[[L30]],[[R31]])
; CHECK: v[[H32:[0-9]+]]:[[L32:[0-9]+]] = vdeal(v[[H31]],v[[L31]],[[R32]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0003(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 4, i32 8, i32 12, i32 64, i32 68, i32 72, i32 76, i32 32, i32 36, i32 40, i32 44, i32 96, i32 100, i32 104, i32 108, i32 16, i32 20, i32 24, i32 28, i32 80, i32 84, i32 88, i32 92, i32 48, i32 52, i32 56, i32 60, i32 112, i32 116, i32 120, i32 124, i32 2, i32 6, i32 10, i32 14, i32 66, i32 70, i32 74, i32 78, i32 34, i32 38, i32 42, i32 46, i32 98, i32 102, i32 106, i32 110, i32 18, i32 22, i32 26, i32 30, i32 82, i32 86, i32 90, i32 94, i32 50, i32 54, i32 58, i32 62, i32 114, i32 118, i32 122, i32 126, i32 1, i32 5, i32 9, i32 13, i32 65, i32 69, i32 73, i32 77, i32 33, i32 37, i32 41, i32 45, i32 97, i32 101, i32 105, i32 109, i32 17, i32 21, i32 25, i32 29, i32 81, i32 85, i32 89, i32 93, i32 49, i32 53, i32 57, i32 61, i32 113, i32 117, i32 121, i32 125, i32 3, i32 7, i32 11, i32 15, i32 67, i32 71, i32 75, i32 79, i32 35, i32 39, i32 43, i32 47, i32 99, i32 103, i32 107, i32 111, i32 19, i32 23, i32 27, i32 31, i32 83, i32 87, i32 91, i32 95, i32 51, i32 55, i32 59, i32 63, i32 115, i32 119, i32 123, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x0a), vdeal(0x10), vdeal(0x31), vshuff(0x30), vdeal(0x00), vdeal(0x39), vdeal(0x0e), vshuff(0x37), vshuff(0x17), vshuff(0x06), vshuff(0x07), vshuff(0x09), vshuff(0x3c), vshuff(0x33), vshuff(0x33)
; CHECK-LABEL: test_0004:
; CHECK-DAG: [[R40:r[0-9]+]] = #57
; CHECK-DAG: [[R41:r[0-9]+]] = #6
; CHECK-DAG: [[R42:r[0-9]+]] = #1
; CHECK: v[[H40:[0-9]+]]:[[L40:[0-9]+]] = vshuff(v1,v0,[[R40]])
; CHECK: v[[H41:[0-9]+]]:[[L41:[0-9]+]] = vshuff(v[[H40]],v[[L40]],[[R41]])
; CHECK: v[[H42:[0-9]+]]:[[L42:[0-9]+]] = vshuff(v[[H41]],v[[L41]],[[R42]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0004(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 4, i32 32, i32 36, i32 2, i32 6, i32 34, i32 38, i32 1, i32 5, i32 33, i32 37, i32 3, i32 7, i32 35, i32 39, i32 8, i32 12, i32 40, i32 44, i32 10, i32 14, i32 42, i32 46, i32 9, i32 13, i32 41, i32 45, i32 11, i32 15, i32 43, i32 47, i32 16, i32 20, i32 48, i32 52, i32 18, i32 22, i32 50, i32 54, i32 17, i32 21, i32 49, i32 53, i32 19, i32 23, i32 51, i32 55, i32 24, i32 28, i32 56, i32 60, i32 26, i32 30, i32 58, i32 62, i32 25, i32 29, i32 57, i32 61, i32 27, i32 31, i32 59, i32 63, i32 64, i32 68, i32 96, i32 100, i32 66, i32 70, i32 98, i32 102, i32 65, i32 69, i32 97, i32 101, i32 67, i32 71, i32 99, i32 103, i32 72, i32 76, i32 104, i32 108, i32 74, i32 78, i32 106, i32 110, i32 73, i32 77, i32 105, i32 109, i32 75, i32 79, i32 107, i32 111, i32 80, i32 84, i32 112, i32 116, i32 82, i32 86, i32 114, i32 118, i32 81, i32 85, i32 113, i32 117, i32 83, i32 87, i32 115, i32 119, i32 88, i32 92, i32 120, i32 124, i32 90, i32 94, i32 122, i32 126, i32 89, i32 93, i32 121, i32 125, i32 91, i32 95, i32 123, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x1c), vshuff(0x31), vdeal(0x1f), vshuff(0x29), vdeal(0x1a), vshuff(0x2a), vshuff(0x25), vshuff(0x05), vshuff(0x04), vshuff(0x23), vdeal(0x0d), vdeal(0x20), vshuff(0x29), vdeal(0x2f), vshuff(0x1d)
; CHECK-LABEL: test_0005:
; CHECK-DAG: [[R50:r[0-9]+]] = #33
; CHECK-DAG: [[R51:r[0-9]+]] = #12
; CHECK-DAG: [[R52:r[0-9]+]] = #1{{$}}
; CHECK: v[[H50:[0-9]+]]:[[L50:[0-9]+]] = vshuff(v1,v0,[[R50]])
; CHECK: v[[H51:[0-9]+]]:[[L51:[0-9]+]] = vshuff(v[[H50]],v[[L50]],[[R51]])
; CHECK: v[[H52:[0-9]+]]:[[L52:[0-9]+]] = vshuff(v[[H51]],v[[L51]],[[R52]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0005(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 8, i32 2, i32 10, i32 32, i32 40, i32 34, i32 42, i32 4, i32 12, i32 6, i32 14, i32 36, i32 44, i32 38, i32 46, i32 16, i32 24, i32 18, i32 26, i32 48, i32 56, i32 50, i32 58, i32 20, i32 28, i32 22, i32 30, i32 52, i32 60, i32 54, i32 62, i32 1, i32 9, i32 3, i32 11, i32 33, i32 41, i32 35, i32 43, i32 5, i32 13, i32 7, i32 15, i32 37, i32 45, i32 39, i32 47, i32 17, i32 25, i32 19, i32 27, i32 49, i32 57, i32 51, i32 59, i32 21, i32 29, i32 23, i32 31, i32 53, i32 61, i32 55, i32 63, i32 64, i32 72, i32 66, i32 74, i32 96, i32 104, i32 98, i32 106, i32 68, i32 76, i32 70, i32 78, i32 100, i32 108, i32 102, i32 110, i32 80, i32 88, i32 82, i32 90, i32 112, i32 120, i32 114, i32 122, i32 84, i32 92, i32 86, i32 94, i32 116, i32 124, i32 118, i32 126, i32 65, i32 73, i32 67, i32 75, i32 97, i32 105, i32 99, i32 107, i32 69, i32 77, i32 71, i32 79, i32 101, i32 109, i32 103, i32 111, i32 81, i32 89, i32 83, i32 91, i32 113, i32 121, i32 115, i32 123, i32 85, i32 93, i32 87, i32 95, i32 117, i32 125, i32 119, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x22), vshuff(0x24), vdeal(0x16), vdeal(0x18), vshuff(0x17), vdeal(0x2d), vshuff(0x38), vshuff(0x20), vshuff(0x37), vdeal(0x3f), vdeal(0x10), vdeal(0x32), vshuff(0x14), vshuff(0x13), vdeal(0x0b)
; CHECK-LABEL: test_0006:
; CHECK-DAG: [[R60:r[0-9]+]] = #3{{$}}
; CHECK-DAG: [[R61:r[0-9]+]] = #36
; CHECK: v[[H60:[0-9]+]]:[[L60:[0-9]+]] = vdeal(v1,v0,[[R60]])
; CHECK: v[[H61:[0-9]+]]:[[L61:[0-9]+]] = vshuff(v[[H60]],v[[L60]],[[R61]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0006(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 2, i32 64, i32 66, i32 1, i32 3, i32 65, i32 67, i32 8, i32 10, i32 72, i32 74, i32 9, i32 11, i32 73, i32 75, i32 16, i32 18, i32 80, i32 82, i32 17, i32 19, i32 81, i32 83, i32 24, i32 26, i32 88, i32 90, i32 25, i32 27, i32 89, i32 91, i32 4, i32 6, i32 68, i32 70, i32 5, i32 7, i32 69, i32 71, i32 12, i32 14, i32 76, i32 78, i32 13, i32 15, i32 77, i32 79, i32 20, i32 22, i32 84, i32 86, i32 21, i32 23, i32 85, i32 87, i32 28, i32 30, i32 92, i32 94, i32 29, i32 31, i32 93, i32 95, i32 32, i32 34, i32 96, i32 98, i32 33, i32 35, i32 97, i32 99, i32 40, i32 42, i32 104, i32 106, i32 41, i32 43, i32 105, i32 107, i32 48, i32 50, i32 112, i32 114, i32 49, i32 51, i32 113, i32 115, i32 56, i32 58, i32 120, i32 122, i32 57, i32 59, i32 121, i32 123, i32 36, i32 38, i32 100, i32 102, i32 37, i32 39, i32 101, i32 103, i32 44, i32 46, i32 108, i32 110, i32 45, i32 47, i32 109, i32 111, i32 52, i32 54, i32 116, i32 118, i32 53, i32 55, i32 117, i32 119, i32 60, i32 62, i32 124, i32 126, i32 61, i32 63, i32 125, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x0f), vdeal(0x01), vshuff(0x3b), vdeal(0x0c), vdeal(0x3f), vdeal(0x26), vshuff(0x28), vdeal(0x3a), vdeal(0x02), vdeal(0x1b), vshuff(0x0e), vdeal(0x03), vshuff(0x3d), vshuff(0x2c), vshuff(0x15)
; CHECK-LABEL: test_0007:
; CHECK-DAG: [[R70:r[0-9]+]] = #50
; CHECK-DAG: [[R71:r[0-9]+]] = #5{{$}}
; CHECK-DAG: [[R72:r[0-9]+]] = #8
; CHECK: v[[H70:[0-9]+]]:[[L70:[0-9]+]] = vshuff(v1,v0,[[R70]])
; CHECK: v[[H71:[0-9]+]]:[[L71:[0-9]+]] = vdeal(v[[H70]],v[[L70]],[[R71]])
; CHECK: v[[H72:[0-9]+]]:[[L72:[0-9]+]] = vshuff(v[[H71]],v[[L71]],[[R72]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0007(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 4, i32 64, i32 68, i32 32, i32 36, i32 96, i32 100, i32 1, i32 5, i32 65, i32 69, i32 33, i32 37, i32 97, i32 101, i32 2, i32 6, i32 66, i32 70, i32 34, i32 38, i32 98, i32 102, i32 3, i32 7, i32 67, i32 71, i32 35, i32 39, i32 99, i32 103, i32 16, i32 20, i32 80, i32 84, i32 48, i32 52, i32 112, i32 116, i32 17, i32 21, i32 81, i32 85, i32 49, i32 53, i32 113, i32 117, i32 18, i32 22, i32 82, i32 86, i32 50, i32 54, i32 114, i32 118, i32 19, i32 23, i32 83, i32 87, i32 51, i32 55, i32 115, i32 119, i32 8, i32 12, i32 72, i32 76, i32 40, i32 44, i32 104, i32 108, i32 9, i32 13, i32 73, i32 77, i32 41, i32 45, i32 105, i32 109, i32 10, i32 14, i32 74, i32 78, i32 42, i32 46, i32 106, i32 110, i32 11, i32 15, i32 75, i32 79, i32 43, i32 47, i32 107, i32 111, i32 24, i32 28, i32 88, i32 92, i32 56, i32 60, i32 120, i32 124, i32 25, i32 29, i32 89, i32 93, i32 57, i32 61, i32 121, i32 125, i32 26, i32 30, i32 90, i32 94, i32 58, i32 62, i32 122, i32 126, i32 27, i32 31, i32 91, i32 95, i32 59, i32 63, i32 123, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x1e), vdeal(0x19), vdeal(0x34), vdeal(0x07), vshuff(0x27), vdeal(0x1e), vdeal(0x21), vdeal(0x2b), vshuff(0x11), vdeal(0x35), vshuff(0x0a), vdeal(0x39), vdeal(0x0c), vdeal(0x17), vdeal(0x23)
; CHECK-LABEL: test_0008:
; CHECK-DAG: [[R80:r[0-9]+]] = #5
; CHECK-DAG: [[R81:r[0-9]+]] = #18
; CHECK-DAG: [[R82:r[0-9]+]] = #9
; CHECK: v[[H80:[0-9]+]]:[[L80:[0-9]+]] = vshuff(v1,v0,[[R80]])
; CHECK: v[[H81:[0-9]+]]:[[L81:[0-9]+]] = vshuff(v[[H80]],v[[L80]],[[R81]])
; CHECK: v[[H82:[0-9]+]]:[[L82:[0-9]+]] = vshuff(v[[H81]],v[[L81]],[[R82]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0008(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 16, i32 4, i32 20, i32 1, i32 17, i32 5, i32 21, i32 64, i32 80, i32 68, i32 84, i32 65, i32 81, i32 69, i32 85, i32 2, i32 18, i32 6, i32 22, i32 3, i32 19, i32 7, i32 23, i32 66, i32 82, i32 70, i32 86, i32 67, i32 83, i32 71, i32 87, i32 32, i32 48, i32 36, i32 52, i32 33, i32 49, i32 37, i32 53, i32 96, i32 112, i32 100, i32 116, i32 97, i32 113, i32 101, i32 117, i32 34, i32 50, i32 38, i32 54, i32 35, i32 51, i32 39, i32 55, i32 98, i32 114, i32 102, i32 118, i32 99, i32 115, i32 103, i32 119, i32 8, i32 24, i32 12, i32 28, i32 9, i32 25, i32 13, i32 29, i32 72, i32 88, i32 76, i32 92, i32 73, i32 89, i32 77, i32 93, i32 10, i32 26, i32 14, i32 30, i32 11, i32 27, i32 15, i32 31, i32 74, i32 90, i32 78, i32 94, i32 75, i32 91, i32 79, i32 95, i32 40, i32 56, i32 44, i32 60, i32 41, i32 57, i32 45, i32 61, i32 104, i32 120, i32 108, i32 124, i32 105, i32 121, i32 109, i32 125, i32 42, i32 58, i32 46, i32 62, i32 43, i32 59, i32 47, i32 63, i32 106, i32 122, i32 110, i32 126, i32 107, i32 123, i32 111, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x1d), vshuff(0x18), vdeal(0x09), vshuff(0x2a), vdeal(0x03), vdeal(0x27), vdeal(0x25), vdeal(0x13), vshuff(0x3a), vshuff(0x19), vshuff(0x06), vshuff(0x0f), vshuff(0x3c), vshuff(0x2e), vshuff(0x36)
; CHECK-LABEL: test_0009:
; CHECK-DAG: [[R90:r[0-9]+]] = #17
; CHECK-DAG: [[R91:r[0-9]+]] = #40
; CHECK-DAG: [[R92:r[0-9]+]] = #6
; CHECK: v[[H90:[0-9]+]]:[[L90:[0-9]+]] = vdeal(v1,v0,[[R90]])
; CHECK: v[[H91:[0-9]+]]:[[L91:[0-9]+]] = vshuff(v[[H90]],v[[L90]],[[R91]])
; CHECK: v[[H92:[0-9]+]]:[[L92:[0-9]+]] = vdeal(v[[H91]],v[[L91]],[[R92]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_0009(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 16, i32 4, i32 20, i32 32, i32 48, i32 36, i32 52, i32 1, i32 17, i32 5, i32 21, i32 33, i32 49, i32 37, i32 53, i32 64, i32 80, i32 68, i32 84, i32 96, i32 112, i32 100, i32 116, i32 65, i32 81, i32 69, i32 85, i32 97, i32 113, i32 101, i32 117, i32 8, i32 24, i32 12, i32 28, i32 40, i32 56, i32 44, i32 60, i32 9, i32 25, i32 13, i32 29, i32 41, i32 57, i32 45, i32 61, i32 72, i32 88, i32 76, i32 92, i32 104, i32 120, i32 108, i32 124, i32 73, i32 89, i32 77, i32 93, i32 105, i32 121, i32 109, i32 125, i32 2, i32 18, i32 6, i32 22, i32 34, i32 50, i32 38, i32 54, i32 3, i32 19, i32 7, i32 23, i32 35, i32 51, i32 39, i32 55, i32 66, i32 82, i32 70, i32 86, i32 98, i32 114, i32 102, i32 118, i32 67, i32 83, i32 71, i32 87, i32 99, i32 115, i32 103, i32 119, i32 10, i32 26, i32 14, i32 30, i32 42, i32 58, i32 46, i32 62, i32 11, i32 27, i32 15, i32 31, i32 43, i32 59, i32 47, i32 63, i32 74, i32 90, i32 78, i32 94, i32 106, i32 122, i32 110, i32 126, i32 75, i32 91, i32 79, i32 95, i32 107, i32 123, i32 111, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x05), vshuff(0x10), vdeal(0x0d), vshuff(0x12), vdeal(0x08), vshuff(0x22), vdeal(0x24), vshuff(0x3e), vdeal(0x00), vshuff(0x14), vdeal(0x3b), vdeal(0x33), vshuff(0x2f), vdeal(0x13), vdeal(0x14)
; CHECK-LABEL: test_000a:
; CHECK-DAG: [[Ra0:r[0-9]+]] = #56
; CHECK-DAG: [[Ra1:r[0-9]+]] = #13
; CHECK-DAG: [[Ra2:r[0-9]+]] = #2
; CHECK: v[[Ha0:[0-9]+]]:[[La0:[0-9]+]] = vshuff(v1,v0,[[Ra0]])
; CHECK: v[[Ha1:[0-9]+]]:[[La1:[0-9]+]] = vdeal(v[[Ha0]],v[[La0]],[[Ra1]])
; CHECK: v[[Ha2:[0-9]+]]:[[La2:[0-9]+]] = vshuff(v[[Ha1]],v[[La1]],[[Ra2]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000a(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 4, i32 1, i32 5, i32 64, i32 68, i32 65, i32 69, i32 32, i32 36, i32 33, i32 37, i32 96, i32 100, i32 97, i32 101, i32 8, i32 12, i32 9, i32 13, i32 72, i32 76, i32 73, i32 77, i32 40, i32 44, i32 41, i32 45, i32 104, i32 108, i32 105, i32 109, i32 16, i32 20, i32 17, i32 21, i32 80, i32 84, i32 81, i32 85, i32 48, i32 52, i32 49, i32 53, i32 112, i32 116, i32 113, i32 117, i32 24, i32 28, i32 25, i32 29, i32 88, i32 92, i32 89, i32 93, i32 56, i32 60, i32 57, i32 61, i32 120, i32 124, i32 121, i32 125, i32 2, i32 6, i32 3, i32 7, i32 66, i32 70, i32 67, i32 71, i32 34, i32 38, i32 35, i32 39, i32 98, i32 102, i32 99, i32 103, i32 10, i32 14, i32 11, i32 15, i32 74, i32 78, i32 75, i32 79, i32 42, i32 46, i32 43, i32 47, i32 106, i32 110, i32 107, i32 111, i32 18, i32 22, i32 19, i32 23, i32 82, i32 86, i32 83, i32 87, i32 50, i32 54, i32 51, i32 55, i32 114, i32 118, i32 115, i32 119, i32 26, i32 30, i32 27, i32 31, i32 90, i32 94, i32 91, i32 95, i32 58, i32 62, i32 59, i32 63, i32 122, i32 126, i32 123, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x12), vshuff(0x2c), vdeal(0x2d), vshuff(0x01), vshuff(0x1f), vshuff(0x30), vdeal(0x2a), vdeal(0x0b), vdeal(0x32), vshuff(0x08), vdeal(0x1b), vdeal(0x09), vshuff(0x1c), vshuff(0x16), vdeal(0x38)
; CHECK-LABEL: test_000b:
; CHECK-DAG: [[Rb0:r[0-9]+]] = #12
; CHECK-DAG: [[Rb1:r[0-9]+]] = #33
; CHECK-DAG: [[Rb2:r[0-9]+]] = #18
; CHECK: v[[Hb0:[0-9]+]]:[[Lb0:[0-9]+]] = vdeal(v1,v0,[[Rb0]])
; CHECK: v[[Hb1:[0-9]+]]:[[Lb1:[0-9]+]] = vdeal(v[[Hb0]],v[[Lb0]],[[Rb1]])
; CHECK: v[[Hb2:[0-9]+]]:[[Lb2:[0-9]+]] = vshuff(v[[Hb1]],v[[Lb1]],[[Rb2]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000b(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 32, i32 1, i32 33, i32 8, i32 40, i32 9, i32 41, i32 64, i32 96, i32 65, i32 97, i32 72, i32 104, i32 73, i32 105, i32 2, i32 34, i32 3, i32 35, i32 10, i32 42, i32 11, i32 43, i32 66, i32 98, i32 67, i32 99, i32 74, i32 106, i32 75, i32 107, i32 4, i32 36, i32 5, i32 37, i32 12, i32 44, i32 13, i32 45, i32 68, i32 100, i32 69, i32 101, i32 76, i32 108, i32 77, i32 109, i32 6, i32 38, i32 7, i32 39, i32 14, i32 46, i32 15, i32 47, i32 70, i32 102, i32 71, i32 103, i32 78, i32 110, i32 79, i32 111, i32 16, i32 48, i32 17, i32 49, i32 24, i32 56, i32 25, i32 57, i32 80, i32 112, i32 81, i32 113, i32 88, i32 120, i32 89, i32 121, i32 18, i32 50, i32 19, i32 51, i32 26, i32 58, i32 27, i32 59, i32 82, i32 114, i32 83, i32 115, i32 90, i32 122, i32 91, i32 123, i32 20, i32 52, i32 21, i32 53, i32 28, i32 60, i32 29, i32 61, i32 84, i32 116, i32 85, i32 117, i32 92, i32 124, i32 93, i32 125, i32 22, i32 54, i32 23, i32 55, i32 30, i32 62, i32 31, i32 63, i32 86, i32 118, i32 87, i32 119, i32 94, i32 126, i32 95, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x31), vdeal(0x29), vshuff(0x19), vshuff(0x39), vdeal(0x17), vshuff(0x28), vshuff(0x0f), vdeal(0x23), vdeal(0x2e), vshuff(0x3d), vdeal(0x1a), vdeal(0x02), vshuff(0x3e), vshuff(0x20), vshuff(0x3f)
; CHECK-LABEL: test_000c:
; CHECK-DAG: [[Rc0:r[0-9]+]] = #12
; CHECK-DAG: [[Rc1:r[0-9]+]] = #6
; CHECK-DAG: [[Rc2:r[0-9]+]] = #17
; CHECK-DAG: [[Rc3:r[0-9]+]] = #32
; CHECK: v[[Hc0:[0-9]+]]:[[Lc0:[0-9]+]] = vshuff(v1,v0,[[Rc0]])
; CHECK: v[[Hc1:[0-9]+]]:[[Lc1:[0-9]+]] = vdeal(v[[Hc0]],v[[Lc0]],[[Rc1]])
; CHECK: v[[Hc2:[0-9]+]]:[[Lc2:[0-9]+]] = vdeal(v[[Hc1]],v[[Lc1]],[[Rc2]])
; CHECK: v[[Hc3:[0-9]+]]:[[Lc3:[0-9]+]] = vshuff(v[[Hc2]],v[[Lc2]],[[Rc3]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000c(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 16, i32 64, i32 80, i32 8, i32 24, i32 72, i32 88, i32 4, i32 20, i32 68, i32 84, i32 12, i32 28, i32 76, i32 92, i32 2, i32 18, i32 66, i32 82, i32 10, i32 26, i32 74, i32 90, i32 6, i32 22, i32 70, i32 86, i32 14, i32 30, i32 78, i32 94, i32 1, i32 17, i32 65, i32 81, i32 9, i32 25, i32 73, i32 89, i32 5, i32 21, i32 69, i32 85, i32 13, i32 29, i32 77, i32 93, i32 3, i32 19, i32 67, i32 83, i32 11, i32 27, i32 75, i32 91, i32 7, i32 23, i32 71, i32 87, i32 15, i32 31, i32 79, i32 95, i32 32, i32 48, i32 96, i32 112, i32 40, i32 56, i32 104, i32 120, i32 36, i32 52, i32 100, i32 116, i32 44, i32 60, i32 108, i32 124, i32 34, i32 50, i32 98, i32 114, i32 42, i32 58, i32 106, i32 122, i32 38, i32 54, i32 102, i32 118, i32 46, i32 62, i32 110, i32 126, i32 33, i32 49, i32 97, i32 113, i32 41, i32 57, i32 105, i32 121, i32 37, i32 53, i32 101, i32 117, i32 45, i32 61, i32 109, i32 125, i32 35, i32 51, i32 99, i32 115, i32 43, i32 59, i32 107, i32 123, i32 39, i32 55, i32 103, i32 119, i32 47, i32 63, i32 111, i32 127>
  ret <128 x i8> %p
}

; Generator: vdeal(0x3c), vdeal(0x24), vdeal(0x05), vdeal(0x37), vshuff(0x21), vdeal(0x11), vdeal(0x1d), vshuff(0x00), vshuff(0x34), vshuff(0x0d), vshuff(0x3a), vshuff(0x1f), vshuff(0x03), vshuff(0x1e), vdeal(0x29)
; CHECK-LABEL: test_000d:
; CHECK-DAG: [[Rd0:r[0-9]+]] = #40
; CHECK-DAG: [[Rd1:r[0-9]+]] = #28
; CHECK: v[[Hd0:[0-9]+]]:[[Ld0:[0-9]+]] = vshuff(v1,v0,[[Rd0]])
; CHECK: v[[Hd1:[0-9]+]]:[[Ld1:[0-9]+]] = vdeal(v[[Hd0]],v[[Ld0]],[[Rd1]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000d(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 1, i32 2, i32 3, i32 64, i32 65, i32 66, i32 67, i32 16, i32 17, i32 18, i32 19, i32 80, i32 81, i32 82, i32 83, i32 32, i32 33, i32 34, i32 35, i32 96, i32 97, i32 98, i32 99, i32 48, i32 49, i32 50, i32 51, i32 112, i32 113, i32 114, i32 115, i32 8, i32 9, i32 10, i32 11, i32 72, i32 73, i32 74, i32 75, i32 24, i32 25, i32 26, i32 27, i32 88, i32 89, i32 90, i32 91, i32 40, i32 41, i32 42, i32 43, i32 104, i32 105, i32 106, i32 107, i32 56, i32 57, i32 58, i32 59, i32 120, i32 121, i32 122, i32 123, i32 4, i32 5, i32 6, i32 7, i32 68, i32 69, i32 70, i32 71, i32 20, i32 21, i32 22, i32 23, i32 84, i32 85, i32 86, i32 87, i32 36, i32 37, i32 38, i32 39, i32 100, i32 101, i32 102, i32 103, i32 52, i32 53, i32 54, i32 55, i32 116, i32 117, i32 118, i32 119, i32 12, i32 13, i32 14, i32 15, i32 76, i32 77, i32 78, i32 79, i32 28, i32 29, i32 30, i32 31, i32 92, i32 93, i32 94, i32 95, i32 44, i32 45, i32 46, i32 47, i32 108, i32 109, i32 110, i32 111, i32 60, i32 61, i32 62, i32 63, i32 124, i32 125, i32 126, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x18), vdeal(0x36), vdeal(0x33), vdeal(0x26), vshuff(0x04), vshuff(0x2d), vshuff(0x35), vdeal(0x34), vdeal(0x2e), vdeal(0x25), vdeal(0x28), vshuff(0x0c), vdeal(0x07), vshuff(0x35), vshuff(0x01)
; CHECK-LABEL: test_000e:
; CHECK-DAG: [[Re0:r[0-9]+]] = #58
; CHECK: v[[He0:[0-9]+]]:[[Le0:[0-9]+]] = vshuff(v1,v0,[[Re0]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000e(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 1, i32 64, i32 65, i32 4, i32 5, i32 68, i32 69, i32 2, i32 3, i32 66, i32 67, i32 6, i32 7, i32 70, i32 71, i32 8, i32 9, i32 72, i32 73, i32 12, i32 13, i32 76, i32 77, i32 10, i32 11, i32 74, i32 75, i32 14, i32 15, i32 78, i32 79, i32 16, i32 17, i32 80, i32 81, i32 20, i32 21, i32 84, i32 85, i32 18, i32 19, i32 82, i32 83, i32 22, i32 23, i32 86, i32 87, i32 24, i32 25, i32 88, i32 89, i32 28, i32 29, i32 92, i32 93, i32 26, i32 27, i32 90, i32 91, i32 30, i32 31, i32 94, i32 95, i32 32, i32 33, i32 96, i32 97, i32 36, i32 37, i32 100, i32 101, i32 34, i32 35, i32 98, i32 99, i32 38, i32 39, i32 102, i32 103, i32 40, i32 41, i32 104, i32 105, i32 44, i32 45, i32 108, i32 109, i32 42, i32 43, i32 106, i32 107, i32 46, i32 47, i32 110, i32 111, i32 48, i32 49, i32 112, i32 113, i32 52, i32 53, i32 116, i32 117, i32 50, i32 51, i32 114, i32 115, i32 54, i32 55, i32 118, i32 119, i32 56, i32 57, i32 120, i32 121, i32 60, i32 61, i32 124, i32 125, i32 58, i32 59, i32 122, i32 123, i32 62, i32 63, i32 126, i32 127>
  ret <128 x i8> %p
}

; Generator: vshuff(0x1a), vshuff(0x10), vdeal(0x2b), vshuff(0x15), vdeal(0x12), vdeal(0x30), vshuff(0x23), vshuff(0x02), vshuff(0x32), vshuff(0x08), vshuff(0x05), vdeal(0x3e), vshuff(0x39), vshuff(0x0a), vshuff(0x0e)
; CHECK-LABEL: test_000f:
; CHECK-DAG: [[Rf0:r[0-9]+]] = #44
; CHECK-DAG: [[Rf1:r[0-9]+]] = #18
; CHECK: v[[Hf0:[0-9]+]]:[[Lf0:[0-9]+]] = vshuff(v1,v0,[[Rf0]])
; CHECK: v[[Hf1:[0-9]+]]:[[Lf1:[0-9]+]] = vshuff(v[[Hf0]],v[[Lf0]],[[Rf1]])
; CHECK-NOT: v{{[0-9:]+}} =
define <128 x i8> @test_000f(<128 x i8> %v0) #0 {
  %p = shufflevector <128 x i8> %v0, <128 x i8> undef, <128 x i32><i32 0, i32 1, i32 32, i32 33, i32 64, i32 65, i32 96, i32 97, i32 4, i32 5, i32 36, i32 37, i32 68, i32 69, i32 100, i32 101, i32 2, i32 3, i32 34, i32 35, i32 66, i32 67, i32 98, i32 99, i32 6, i32 7, i32 38, i32 39, i32 70, i32 71, i32 102, i32 103, i32 8, i32 9, i32 40, i32 41, i32 72, i32 73, i32 104, i32 105, i32 12, i32 13, i32 44, i32 45, i32 76, i32 77, i32 108, i32 109, i32 10, i32 11, i32 42, i32 43, i32 74, i32 75, i32 106, i32 107, i32 14, i32 15, i32 46, i32 47, i32 78, i32 79, i32 110, i32 111, i32 16, i32 17, i32 48, i32 49, i32 80, i32 81, i32 112, i32 113, i32 20, i32 21, i32 52, i32 53, i32 84, i32 85, i32 116, i32 117, i32 18, i32 19, i32 50, i32 51, i32 82, i32 83, i32 114, i32 115, i32 22, i32 23, i32 54, i32 55, i32 86, i32 87, i32 118, i32 119, i32 24, i32 25, i32 56, i32 57, i32 88, i32 89, i32 120, i32 121, i32 28, i32 29, i32 60, i32 61, i32 92, i32 93, i32 124, i32 125, i32 26, i32 27, i32 58, i32 59, i32 90, i32 91, i32 122, i32 123, i32 30, i32 31, i32 62, i32 63, i32 94, i32 95, i32 126, i32 127>
  ret <128 x i8> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

