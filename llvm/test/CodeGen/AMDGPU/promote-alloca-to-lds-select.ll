; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck %s

; CHECK-LABEL: @lds_promoted_alloca_select_invalid_pointer_operand(
; CHECK: %alloca = alloca i32
; CHECK: select i1 undef, ptr addrspace(5) poison, ptr addrspace(5) %alloca
define amdgpu_kernel void @lds_promoted_alloca_select_invalid_pointer_operand() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  %select = select i1 undef, ptr addrspace(5) poison, ptr addrspace(5) %alloca
  store i32 0, ptr addrspace(5) %select, align 4
  ret void
}

; CHECK-LABEL: @lds_promote_alloca_select_two_derived_pointers(
; CHECK: [[ARRAYGEP:%[0-9]+]] = getelementptr inbounds [256 x [16 x i32]], ptr addrspace(3) @lds_promote_alloca_select_two_derived_pointers.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(3) [[ARRAYGEP]], i32 0, i32 %a
; CHECK: %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(3) [[ARRAYGEP]], i32 0, i32 %b
; CHECK: %select = select i1 undef, ptr addrspace(3) %ptr0, ptr addrspace(3) %ptr1
; CHECK: store i32 0, ptr addrspace(3) %select, align 4
define amdgpu_kernel void @lds_promote_alloca_select_two_derived_pointers(i32 %a, i32 %b) #0 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %a
  %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %b
  %select = select i1 undef, ptr addrspace(5) %ptr0, ptr addrspace(5) %ptr1
  store i32 0, ptr addrspace(5) %select, align 4
  ret void
}

; FIXME: This should be promotable but requires knowing that both will be promoted first.

; CHECK-LABEL: @lds_promote_alloca_select_two_allocas(
; CHECK: %alloca0 = alloca i32, i32 16, align 4
; CHECK: %alloca1 = alloca i32, i32 16, align 4
; CHECK: %ptr0 = getelementptr inbounds i32, ptr addrspace(5) %alloca0, i32 %a
; CHECK: %ptr1 = getelementptr inbounds i32, ptr addrspace(5) %alloca1, i32 %b
; CHECK: %select = select i1 undef, ptr addrspace(5) %ptr0, ptr addrspace(5) %ptr1
define amdgpu_kernel void @lds_promote_alloca_select_two_allocas(i32 %a, i32 %b) #0 {
  %alloca0 = alloca i32, i32 16, align 4, addrspace(5)
  %alloca1 = alloca i32, i32 16, align 4, addrspace(5)
  %ptr0 = getelementptr inbounds i32, ptr addrspace(5) %alloca0, i32 %a
  %ptr1 = getelementptr inbounds i32, ptr addrspace(5) %alloca1, i32 %b
  %select = select i1 undef, ptr addrspace(5) %ptr0, ptr addrspace(5) %ptr1
  store i32 0, ptr addrspace(5) %select, align 4
  ret void
}

; TODO: Maybe this should be canonicalized to select on the constant and GEP after.
; CHECK-LABEL: @lds_promote_alloca_select_two_derived_constant_pointers(
; CHECK: [[ARRAYGEP:%[0-9]+]] = getelementptr inbounds [256 x [16 x i32]], ptr addrspace(3) @lds_promote_alloca_select_two_derived_constant_pointers.alloca, i32 0, i32 %{{[0-9]+}}
; CHECK: %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(3) [[ARRAYGEP]], i32 0, i32 1
; CHECK: %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(3) [[ARRAYGEP]], i32 0, i32 3
; CHECK: %select = select i1 undef, ptr addrspace(3) %ptr0, ptr addrspace(3) %ptr1
; CHECK: store i32 0, ptr addrspace(3) %select, align 4
define amdgpu_kernel void @lds_promote_alloca_select_two_derived_constant_pointers() #0 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 1
  %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 3
  %select = select i1 undef, ptr addrspace(5) %ptr0, ptr addrspace(5) %ptr1
  store i32 0, ptr addrspace(5) %select, align 4
  ret void
}

; FIXME: Can be promoted, but we'd have to recursively show that the select
; operands all point to the same alloca.

; CHECK-LABEL: @lds_promoted_alloca_select_input_select(
; CHECK: alloca
define amdgpu_kernel void @lds_promoted_alloca_select_input_select(i32 %a, i32 %b, i32 %c, i1 %c1, i1 %c2) #0 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %a
  %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %b
  %ptr2 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %c
  %select0 = select i1 %c1, ptr addrspace(5) %ptr0, ptr addrspace(5) %ptr1
  %select1 = select i1 %c2, ptr addrspace(5) %select0, ptr addrspace(5) %ptr2
  store i32 0, ptr addrspace(5) %select1, align 4
  ret void
}

define amdgpu_kernel void @lds_promoted_alloca_select_input_phi(i32 %a, i32 %b, i32 %c, i1 %c0) #0 {
entry:
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  %ptr0 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %a
  %ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %b
  store i32 0, ptr addrspace(5) %ptr0
  br i1 %c0, label %bb1, label %bb2

bb1:
  %ptr2 = getelementptr inbounds [16 x i32], ptr addrspace(5) %alloca, i32 0, i32 %c
  %select0 = select i1 undef, ptr addrspace(5) poison, ptr addrspace(5) %ptr2
  store i32 0, ptr addrspace(5) %ptr1
  br label %bb2

bb2:
  %phi.ptr = phi ptr addrspace(5) [ %ptr0, %entry ], [ %select0, %bb1 ]
  %select1 = select i1 undef, ptr addrspace(5) %phi.ptr, ptr addrspace(5) %ptr1
  store i32 0, ptr addrspace(5) %select1, align 4
  ret void
}

; CHECK-LABEL: @select_null_rhs(
; CHECK-NOT: alloca
; CHECK: select i1 %tmp2, ptr addrspace(3) %{{[0-9]+}}, ptr addrspace(3) null
define amdgpu_kernel void @select_null_rhs(ptr addrspace(1) nocapture %arg, i32 %arg1) #1 {
bb:
  %tmp = alloca double, align 8, addrspace(5)
  store double 0.000000e+00, ptr addrspace(5) %tmp, align 8
  %tmp2 = icmp eq i32 %arg1, 0
  %tmp3 = select i1 %tmp2, ptr addrspace(5) %tmp, ptr addrspace(5) null
  store double 1.000000e+00, ptr addrspace(5) %tmp3, align 8
  %tmp4 = load double, ptr addrspace(5) %tmp, align 8
  store double %tmp4, ptr addrspace(1) %arg
  ret void
}

; CHECK-LABEL: @select_null_lhs(
; CHECK-NOT: alloca
; CHECK: select i1 %tmp2, ptr addrspace(3) null, ptr addrspace(3) %{{[0-9]+}}
define amdgpu_kernel void @select_null_lhs(ptr addrspace(1) nocapture %arg, i32 %arg1) #1 {
bb:
  %tmp = alloca double, align 8, addrspace(5)
  store double 0.000000e+00, ptr addrspace(5) %tmp, align 8
  %tmp2 = icmp eq i32 %arg1, 0
  %tmp3 = select i1 %tmp2, ptr addrspace(5) null, ptr addrspace(5) %tmp
  store double 1.000000e+00, ptr addrspace(5) %tmp3, align 8
  %tmp4 = load double, ptr addrspace(5) %tmp, align 8
  store double %tmp4, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { norecurse nounwind "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { norecurse nounwind }
