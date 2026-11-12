; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-vector-pseudos -verify-machineinstrs -stop-after=machine-scheduler < %s | FileCheck --check-prefix=PSEUDO %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs -stop-after=machine-scheduler < %s | FileCheck --check-prefix=DEFAULT %s

; PlumbCompiler: with -amdgpu-vector-pseudos, element-wise vector ops and
; reductions on 2/4/8/16 x 32-bit lanes stay SI_VECTOR_* pseudos through the
; pre-RA machine scheduler. vselect(setcc) on integer lanes is one pseudo.

; DEFAULT-NOT: SI_VECTOR_

; PSEUDO-LABEL: name: f32_ops
; PSEUDO-DAG: SI_VECTOR_FMUL_V16F32
; PSEUDO-DAG: SI_VECTOR_FSUB_V16F32
; PSEUDO-DAG: SI_VECTOR_EXP2_V16F32
; PSEUDO-DAG: SI_VECTOR_FPTRUNC_V16F32_BF16
; PSEUDO-DAG: SI_VECTOR_REDUCE_FMAX_V16F32
; PSEUDO-DAG: SI_VECTOR_REDUCE_FADD_V16F32
define amdgpu_kernel void @f32_ops(ptr addrspace(1) %p, ptr addrspace(1) %q, ptr addrspace(1) %r, float %m) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <16 x float>, ptr addrspace(1) %p, i32 %tid
  %a = load <16 x float>, ptr addrspace(1) %gp
  %ins = insertelement <16 x float> poison, float %m, i64 0
  %sm = shufflevector <16 x float> %ins, <16 x float> poison, <16 x i32> zeroinitializer
  %s = fmul <16 x float> %a, %sm
  %mx = call float @llvm.vector.reduce.fmax.v16f32(<16 x float> %s)
  %i2 = insertelement <16 x float> poison, float %mx, i64 0
  %smx = shufflevector <16 x float> %i2, <16 x float> poison, <16 x i32> zeroinitializer
  %d = fsub <16 x float> %s, %smx
  %e = call afn <16 x float> @llvm.exp2.v16f32(<16 x float> %d)
  %sum = call reassoc float @llvm.vector.reduce.fadd.v16f32(float -0.0, <16 x float> %e)
  %t = fptrunc <16 x float> %e to <16 x bfloat>
  %gq = getelementptr <16 x bfloat>, ptr addrspace(1) %q, i32 %tid
  store <16 x bfloat> %t, ptr addrspace(1) %gq
  %gr = getelementptr float, ptr addrspace(1) %r, i32 %tid
  store float %sum, ptr addrspace(1) %gr
  ret void
}

; PSEUDO-LABEL: name: i32_ops
; PSEUDO-DAG: SI_VECTOR_MUL_V8I32
; PSEUDO-DAG: SI_VECTOR_ADD_V8I32
; PSEUDO-DAG: SI_VECTOR_SHL_V8I32
; PSEUDO-DAG: SI_VECTOR_OR_V8I32
; PSEUDO-DAG: SI_VECTOR_ICMP_SELECT_V8B32 {{.*}}, 20,
define amdgpu_kernel void @i32_ops(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %n) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <8 x i32>, ptr addrspace(1) %p, i32 %tid
  %a = load <8 x i32>, ptr addrspace(1) %gp
  %ins = insertelement <8 x i32> poison, i32 %n, i64 0
  %sn = shufflevector <8 x i32> %ins, <8 x i32> poison, <8 x i32> zeroinitializer
  %m = mul <8 x i32> %a, %sn
  %b = add <8 x i32> %m, %a
  %c = shl <8 x i32> %b, splat (i32 1)
  %o = or <8 x i32> %c, splat (i32 3)
  %k = icmp slt <8 x i32> %a, %sn
  %sel = select <8 x i1> %k, <8 x i32> %o, <8 x i32> splat (i32 -1)
  %gq = getelementptr <8 x i32>, ptr addrspace(1) %q, i32 %tid
  store <8 x i32> %sel, ptr addrspace(1) %gq
  ret void
}
