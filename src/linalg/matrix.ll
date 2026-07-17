; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; universe_linalg_* — dense linear-algebra foundation (BLAS core). Row-major,
; contiguous f64 matrices; dimensions passed explicitly; caller owns all
; storage (ops write into a caller-provided out buffer, never allocate). Method
; names follow Julia's LinearAlgebra stdlib (functional inspiration only — the
; layout, kernels, and IR are designed here from first principles).
;
; Matrix convention: A is m rows x n cols, row-major, so element (i,j) lives at
; A[i*n + j] (leading dimension lda = n). Errors i32: 0 OK, 1 NULL_PTR,
; 3 SIZE_OVERFLOW, 8 INVALID_ARG (shape mismatch / non-normalizable). Value
; returns (dot/norm/tr) return the number directly; a 0-length input yields 0.0
; and the caller owns pointer validity (matching the ml-kernel convention).
;
; DESIGN — the centrepiece is a REGISTER-BLOCKED SIMD GEMM (matmul):
;   * TILE: a 4x4 block of C is accumulated entirely in vector registers — 8
;     independent <2 x double> accumulators (4 rows x 2 half-rows). The K-loop
;     loads the 4 contiguous C-columns of B row p as two <2 x double> vectors
;     (b0,b1), broadcasts each of the 4 A elements A[(i+r),p], and issues an
;     outer-product update `c[r] += a_r * b` via `llvm.fma.v2f64`. The two B
;     vectors are REUSED across all 4 rows (register reuse), and the 8
;     accumulator chains are INDEPENDENT so the FMAs pipeline without a
;     loop-carried stall (the classic dependency-breaking multi-accumulator
;     trick — cf. the ml-kernel 4-accumulator reductions). This lowers to
;     packed `fmla.2d` (AArch64) / `vfmadd*pd` (x86) with a spill-free inner
;     loop — verified at the hot-path gate.
;   * SCALAR EDGE + ORACLE: rows/cols outside the largest 4-multiple tiling
;     (m&~3, n&~3) are finished by a strided scalar dot per output element
;     (li_gemm_elem). This is the SIMD-first "scalar tail". A separately
;     exported `universe_linalg_matmul_scalar` triple-loop is the cross-check
;     ORACLE the vector path is validated against (< 1e-9 rel on random input).
;   * K==0 is the empty sum: C is zeroed. m==0 or n==0 write nothing.
;   * FP flags: the GEMM uses the plain `llvm.fma.v2f64` (no reassoc) so the 8
;     accumulator chains stay as written and every product is a fused multiply-
;     add (more accurate than the separate-rounding scalar oracle, so the two
;     agree well within 1e-9). The reduction helpers (dot/asum/amax) use `fast`
;     for the 4-accumulator tree + fmla contraction; their `_scalar` twins use
;     unflagged ops as the stable reference.
;
; Vector reductions (dot / norm / norm_fro) mirror the ml-kernel shape: a
; <2 x double> primary path (4 accumulators for dot, 8 doubles/iter) with a
; scalar tail for the sub-vector remainder and as the oracle.
;
; API (f64, row-major; element (i,j) at A[i*n+j]):
;   i32 universe_linalg_matmul(A,m,k, B,k2,n, C_out)   ; C = A*B  (gemm)
;   i32 universe_linalg_matmul_scalar(A,m,k, B,k2,n, C_out)  ; triple-loop oracle
;   i32 universe_linalg_matvec(A,m,n, x, y_out)        ; y = A*x  (gemv)
;   i32 universe_linalg_matvec_scalar(A,m,n, x, y_out)
;   i32 universe_linalg_mul_scalar(A,n, s, out)        ; out = s*A (n elements)
;   i32 universe_linalg_add(A, B, n, out)              ; out = A+B (n elements)
;   i32 universe_linalg_sub(A, B, n, out)              ; out = A-B (n elements)
;   i32 universe_linalg_transpose(A,m,n, out)          ; out(n x m) = A^T
;   double universe_linalg_dot(x, y, n)                ; sum x[i]*y[i]
;   double universe_linalg_dot_scalar(x, y, n)
;   double universe_linalg_norm(x, n, p)               ; p: 2, 1, 0=Inf
;   double universe_linalg_norm_scalar(x, n, p)
;   double universe_linalg_norm_fro(A, n)              ; Frobenius = sqrt(dot(A,A))
;   double universe_linalg_norm_fro_scalar(A, n)
;   i32 universe_linalg_normalize(x, n, out)           ; out = x/norm2 (8 if norm=0)
;   i32 universe_linalg_cross(a, b, out)               ; 3-vectors
;   double universe_linalg_tr(A, n)                    ; trace of n x n
;   i32 universe_linalg_diag(A, n, out)                ; out[i]=A[i,i] (n x n)
;   i32 universe_linalg_diagm(v, n, out)               ; out = diag(v) (n x n)
;   i32 universe_linalg_identity(out, n)               ; out = I_n
;   i32 universe_linalg_kron(A,am,an, B,bm,bn, out)    ; Kronecker product

declare double @llvm.fma.f64(double, double, double)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare double @llvm.sqrt.f64(double)
declare double @llvm.fabs.f64(double)
declare <2 x double> @llvm.fabs.v2f64(<2 x double>)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>)
declare double @llvm.vector.reduce.fmax.v2f64(<2 x double>)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare double @llvm.maxnum.f64(double, double)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ==================================================================== helpers

; li_dot_impl — 4-accumulator <2 x double> reduction of sum a[i]*b[i]
; (8 doubles/iteration). alwaysinline: folds into dot/norm/matvec at zero cost.
define internal double @li_dot_impl(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
entry:
  %has8 = icmp uge i64 %n, 8
  br i1 %has8, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %pa0 = getelementptr inbounds nuw double, ptr %a, i64 %i
  %va0 = load <2 x double>, ptr %pa0, align 8
  %pb0 = getelementptr inbounds nuw double, ptr %b, i64 %i
  %vb0 = load <2 x double>, ptr %pb0, align 8
  %m0 = fmul fast <2 x double> %va0, %vb0
  %acc0n = fadd fast <2 x double> %acc0, %m0
  %i1 = add nuw i64 %i, 2
  %pa1 = getelementptr inbounds nuw double, ptr %a, i64 %i1
  %va1 = load <2 x double>, ptr %pa1, align 8
  %pb1 = getelementptr inbounds nuw double, ptr %b, i64 %i1
  %vb1 = load <2 x double>, ptr %pb1, align 8
  %m1 = fmul fast <2 x double> %va1, %vb1
  %acc1n = fadd fast <2 x double> %acc1, %m1
  %i2 = add nuw i64 %i, 4
  %pa2 = getelementptr inbounds nuw double, ptr %a, i64 %i2
  %va2 = load <2 x double>, ptr %pa2, align 8
  %pb2 = getelementptr inbounds nuw double, ptr %b, i64 %i2
  %vb2 = load <2 x double>, ptr %pb2, align 8
  %m2 = fmul fast <2 x double> %va2, %vb2
  %acc2n = fadd fast <2 x double> %acc2, %m2
  %i3 = add nuw i64 %i, 6
  %pa3 = getelementptr inbounds nuw double, ptr %a, i64 %i3
  %va3 = load <2 x double>, ptr %pa3, align 8
  %pb3 = getelementptr inbounds nuw double, ptr %b, i64 %i3
  %vb3 = load <2 x double>, ptr %pb3, align 8
  %m3 = fmul fast <2 x double> %va3, %vb3
  %acc3n = fadd fast <2 x double> %acc3, %m3
  %inext = add nuw i64 %i, 8
  %lim = sub nuw i64 %n, 8
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <2 x double> %acc0n, %acc1n
  %c1 = fadd fast <2 x double> %acc2n, %acc3n
  %csum = fadd fast <2 x double> %c0, %c1
  %hs = call fast double @llvm.vector.reduce.fadd.v2f64(double -0.0, <2 x double> %csum)
  br label %red

red:
  %base = phi double [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi double [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw double, ptr %a, i64 %j
  %fa = load double, ptr %ta, align 8
  %tb = getelementptr inbounds nuw double, ptr %b, i64 %j
  %fb = load double, ptr %tb, align 8
  %p = fmul fast double %fa, %fb
  %saccn = fadd fast double %sacc, %p
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret double %sacc
}

; li_asum_impl — sum of |x[i]| ( vector <2 x double> fabs + scalar tail ).
define internal double @li_asum_impl(ptr readonly %x, i64 %n) #2 {
entry:
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %red

vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %acc = phi <2 x double> [ zeroinitializer, %entry ], [ %accn, %vloop ]
  %p = getelementptr inbounds nuw double, ptr %x, i64 %i
  %v = load <2 x double>, ptr %p, align 8
  %av = call <2 x double> @llvm.fabs.v2f64(<2 x double> %v)
  %accn = fadd fast <2 x double> %acc, %av
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hs = call fast double @llvm.vector.reduce.fadd.v2f64(double -0.0, <2 x double> %accn)
  br label %red

red:
  %base = phi double [ 0.0, %entry ], [ %hs, %vdone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %vdone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tbody ]
  %sacc = phi double [ %base, %red ], [ %saccn, %tbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tbody

tbody:
  %tp = getelementptr inbounds nuw double, ptr %x, i64 %j
  %fx = load double, ptr %tp, align 8
  %afx = call double @llvm.fabs.f64(double %fx)
  %saccn = fadd fast double %sacc, %afx
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret double %sacc
}

; li_amax_impl — max of |x[i]| (Inf-norm). vector fmax + scalar tail; init 0.
define internal double @li_amax_impl(ptr readonly %x, i64 %n) #2 {
entry:
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %red

vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %acc = phi <2 x double> [ zeroinitializer, %entry ], [ %accn, %vloop ]
  %p = getelementptr inbounds nuw double, ptr %x, i64 %i
  %v = load <2 x double>, ptr %p, align 8
  %av = call <2 x double> @llvm.fabs.v2f64(<2 x double> %v)
  %accn = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %acc, <2 x double> %av)
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hs = call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %accn)
  br label %red

red:
  %base = phi double [ 0.0, %entry ], [ %hs, %vdone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %vdone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tbody ]
  %smax = phi double [ %base, %red ], [ %smaxn, %tbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tbody

tbody:
  %tp = getelementptr inbounds nuw double, ptr %x, i64 %j
  %fx = load double, ptr %tp, align 8
  %afx = call double @llvm.fabs.f64(double %fx)
  %smaxn = call double @llvm.maxnum.f64(double %smax, double %afx)
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret double %smax
}

; li_gemm_elem — strided scalar dot for one C element: sum_p arow[p]*B[p*n+j].
; Used for the scalar edge tiles of the register-blocked GEMM.
define internal double @li_gemm_elem(ptr readonly %arow, ptr readonly %B, i64 %k, i64 %n, i64 %j) #2 {
entry:
  %bj = getelementptr inbounds double, ptr %B, i64 %j
  br label %loop

loop:
  %p = phi i64 [ 0, %entry ], [ %pn, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %ap = getelementptr inbounds double, ptr %arow, i64 %p
  %av = load double, ptr %ap, align 8
  %boff = mul nuw i64 %p, %n
  %bp = getelementptr inbounds double, ptr %bj, i64 %boff
  %bv = load double, ptr %bp, align 8
  %accn = call double @llvm.fma.f64(double %av, double %bv, double %acc)
  %pn = add nuw i64 %p, 1
  %more = icmp ult i64 %pn, %k
  br i1 %more, label %loop, label %done

done:
  ret double %accn
}

; gemm_micro_4x4 — register-blocked 4x4 micro-kernel. Accumulates
; C[i0..i0+3][j0..j0+3] over the full K dimension in 8 <2 x double> registers.
; A: m x k (lda=k), B: k x n (ldb=n), C: m x n (ldc=n). Requires k>=1 and the
; whole 4x4 tile in-bounds (caller guarantees).
define internal void @gemm_micro_4x4(ptr readonly %A, ptr readonly %B, ptr %C, i64 %k, i64 %n, i64 %i0, i64 %j0) #3 {
entry:
  %r0 = mul nuw i64 %i0, %k
  %ar0 = getelementptr inbounds double, ptr %A, i64 %r0
  %ii1 = add nuw i64 %i0, 1
  %r1 = mul nuw i64 %ii1, %k
  %ar1 = getelementptr inbounds double, ptr %A, i64 %r1
  %ii2 = add nuw i64 %i0, 2
  %r2 = mul nuw i64 %ii2, %k
  %ar2 = getelementptr inbounds double, ptr %A, i64 %r2
  %ii3 = add nuw i64 %i0, 3
  %r3 = mul nuw i64 %ii3, %k
  %ar3 = getelementptr inbounds double, ptr %A, i64 %r3
  %bj = getelementptr inbounds double, ptr %B, i64 %j0
  br label %kloop

kloop:
  %p = phi i64 [ 0, %entry ], [ %pn, %kloop ]
  %c00 = phi <2 x double> [ zeroinitializer, %entry ], [ %c00n, %kloop ]
  %c01 = phi <2 x double> [ zeroinitializer, %entry ], [ %c01n, %kloop ]
  %c10 = phi <2 x double> [ zeroinitializer, %entry ], [ %c10n, %kloop ]
  %c11 = phi <2 x double> [ zeroinitializer, %entry ], [ %c11n, %kloop ]
  %c20 = phi <2 x double> [ zeroinitializer, %entry ], [ %c20n, %kloop ]
  %c21 = phi <2 x double> [ zeroinitializer, %entry ], [ %c21n, %kloop ]
  %c30 = phi <2 x double> [ zeroinitializer, %entry ], [ %c30n, %kloop ]
  %c31 = phi <2 x double> [ zeroinitializer, %entry ], [ %c31n, %kloop ]
  ; B row p, 4 contiguous columns starting at j0 -> two <2 x double>
  %boff = mul nuw i64 %p, %n
  %bp0 = getelementptr inbounds double, ptr %bj, i64 %boff
  %b0 = load <2 x double>, ptr %bp0, align 8
  %bp1 = getelementptr inbounds double, ptr %bp0, i64 2
  %b1 = load <2 x double>, ptr %bp1, align 8
  ; broadcast A[(i0+r),p] and fma into the two half-rows
  %a0p = getelementptr inbounds double, ptr %ar0, i64 %p
  %a0 = load double, ptr %a0p, align 8
  %a0i = insertelement <2 x double> poison, double %a0, i64 0
  %a0b = shufflevector <2 x double> %a0i, <2 x double> poison, <2 x i32> zeroinitializer
  %c00n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a0b, <2 x double> %b0, <2 x double> %c00)
  %c01n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a0b, <2 x double> %b1, <2 x double> %c01)
  %a1p = getelementptr inbounds double, ptr %ar1, i64 %p
  %a1 = load double, ptr %a1p, align 8
  %a1i = insertelement <2 x double> poison, double %a1, i64 0
  %a1b = shufflevector <2 x double> %a1i, <2 x double> poison, <2 x i32> zeroinitializer
  %c10n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a1b, <2 x double> %b0, <2 x double> %c10)
  %c11n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a1b, <2 x double> %b1, <2 x double> %c11)
  %a2p = getelementptr inbounds double, ptr %ar2, i64 %p
  %a2 = load double, ptr %a2p, align 8
  %a2i = insertelement <2 x double> poison, double %a2, i64 0
  %a2b = shufflevector <2 x double> %a2i, <2 x double> poison, <2 x i32> zeroinitializer
  %c20n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a2b, <2 x double> %b0, <2 x double> %c20)
  %c21n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a2b, <2 x double> %b1, <2 x double> %c21)
  %a3p = getelementptr inbounds double, ptr %ar3, i64 %p
  %a3 = load double, ptr %a3p, align 8
  %a3i = insertelement <2 x double> poison, double %a3, i64 0
  %a3b = shufflevector <2 x double> %a3i, <2 x double> poison, <2 x i32> zeroinitializer
  %c30n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a3b, <2 x double> %b0, <2 x double> %c30)
  %c31n = call <2 x double> @llvm.fma.v2f64(<2 x double> %a3b, <2 x double> %b1, <2 x double> %c31)
  %pn = add nuw i64 %p, 1
  %more = icmp ult i64 %pn, %k
  br i1 %more, label %kloop, label %store

store:
  %cr0 = mul nuw i64 %i0, %n
  %c0p = getelementptr inbounds double, ptr %C, i64 %cr0
  %c0j = getelementptr inbounds double, ptr %c0p, i64 %j0
  store <2 x double> %c00n, ptr %c0j, align 8
  %c0j2 = getelementptr inbounds double, ptr %c0j, i64 2
  store <2 x double> %c01n, ptr %c0j2, align 8
  %cr1 = mul nuw i64 %ii1, %n
  %c1p = getelementptr inbounds double, ptr %C, i64 %cr1
  %c1j = getelementptr inbounds double, ptr %c1p, i64 %j0
  store <2 x double> %c10n, ptr %c1j, align 8
  %c1j2 = getelementptr inbounds double, ptr %c1j, i64 2
  store <2 x double> %c11n, ptr %c1j2, align 8
  %cr2 = mul nuw i64 %ii2, %n
  %c2p = getelementptr inbounds double, ptr %C, i64 %cr2
  %c2j = getelementptr inbounds double, ptr %c2p, i64 %j0
  store <2 x double> %c20n, ptr %c2j, align 8
  %c2j2 = getelementptr inbounds double, ptr %c2j, i64 2
  store <2 x double> %c21n, ptr %c2j2, align 8
  %cr3 = mul nuw i64 %ii3, %n
  %c3p = getelementptr inbounds double, ptr %C, i64 %cr3
  %c3j = getelementptr inbounds double, ptr %c3p, i64 %j0
  store <2 x double> %c30n, ptr %c3j, align 8
  %c3j2 = getelementptr inbounds double, ptr %c3j, i64 2
  store <2 x double> %c31n, ptr %c3j2, align 8
  ret void
}

; ==================================================================== matmul

define i32 @universe_linalg_matmul(ptr readonly %A, i64 %m, i64 %k, ptr readonly %B, i64 %k2, i64 %n, ptr %C) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nB = icmp eq ptr %B, null
  %nC = icmp eq ptr %C, null
  %n0 = or i1 %nA, %nB
  %nz = or i1 %n0, %nC
  br i1 %nz, label %enull, label %shape

shape:
  %badk = icmp ne i64 %k, %k2
  br i1 %badk, label %einval, label %ovf

ovf:
  ; overflow-check the element-count products (validation; no allocation).
  %mn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %o0 = extractvalue { i64, i1 } %mn, 1
  %mk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %k)
  %o1 = extractvalue { i64, i1 } %mk, 1
  %kn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %k, i64 %n)
  %o2 = extractvalue { i64, i1 } %kn, 1
  %oa = or i1 %o0, %o1
  %ob = or i1 %oa, %o2
  br i1 %ob, label %eovf, label %dims

dims:
  ; empty output: nothing to write.
  %me = icmp eq i64 %m, 0
  %nez = icmp eq i64 %n, 0
  %empty = or i1 %me, %nez
  br i1 %empty, label %ok, label %kcheck

kcheck:
  %k0 = icmp eq i64 %k, 0
  br i1 %k0, label %zeroC, label %tiled

zeroC:
  %cnt = mul nuw i64 %m, %n
  %bytes = shl nuw i64 %cnt, 3
  call void @llvm.memset.p0.i64(ptr %C, i8 0, i64 %bytes, i1 false)
  br label %ok

tiled:
  %mfull = and i64 %m, -4
  %nfull = and i64 %n, -4
  br label %iloop

iloop:
  %i = phi i64 [ 0, %tiled ], [ %inext, %icont ]
  %ilt = icmp ult i64 %i, %mfull
  br i1 %ilt, label %jloop, label %rowrem

jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnext, %jloop ]
  call void @gemm_micro_4x4(ptr %A, ptr %B, ptr %C, i64 %k, i64 %n, i64 %i, i64 %j)
  %jnext = add nuw i64 %j, 4
  %jlt = icmp ult i64 %jnext, %nfull
  br i1 %jlt, label %jloop, label %coltail

; column remainder [nfull, n) for the 4-row block at %i
coltail:
  %ct = icmp ult i64 %nfull, %n
  br i1 %ct, label %ctloop, label %icont

ctloop:
  %cj = phi i64 [ %nfull, %coltail ], [ %cjn, %ctrow_done ]
  br label %ctr

ctr:
  %rr = phi i64 [ 0, %ctloop ], [ %rrn, %ctr ]
  %ri = add nuw i64 %i, %rr
  %aoff = mul nuw i64 %ri, %k
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %val = call double @li_gemm_elem(ptr %arow, ptr %B, i64 %k, i64 %n, i64 %cj)
  %coff = mul nuw i64 %ri, %n
  %cp0 = getelementptr inbounds double, ptr %C, i64 %coff
  %cp = getelementptr inbounds double, ptr %cp0, i64 %cj
  store double %val, ptr %cp, align 8
  %rrn = add nuw i64 %rr, 1
  %rrmore = icmp ult i64 %rrn, 4
  br i1 %rrmore, label %ctr, label %ctrow_done

ctrow_done:
  %cjn = add nuw i64 %cj, 1
  %cjmore = icmp ult i64 %cjn, %n
  br i1 %cjmore, label %ctloop, label %icont

icont:
  %inext = add nuw i64 %i, 4
  br label %iloop

; row remainder [mfull, m): full scalar rows
rowrem:
  %rlt = icmp ult i64 %mfull, %m
  br i1 %rlt, label %rrloop, label %ok

rrloop:
  %ir = phi i64 [ %mfull, %rowrem ], [ %irn, %rrdone ]
  %araoff = mul nuw i64 %ir, %k
  %arowr = getelementptr inbounds double, ptr %A, i64 %araoff
  %crbase = mul nuw i64 %ir, %n
  br label %rrj

rrj:
  %jr = phi i64 [ 0, %rrloop ], [ %jrn, %rrj ]
  %rval = call double @li_gemm_elem(ptr %arowr, ptr %B, i64 %k, i64 %n, i64 %jr)
  %rcp0 = getelementptr inbounds double, ptr %C, i64 %crbase
  %rcp = getelementptr inbounds double, ptr %rcp0, i64 %jr
  store double %rval, ptr %rcp, align 8
  %jrn = add nuw i64 %jr, 1
  %jrmore = icmp ult i64 %jrn, %n
  br i1 %jrmore, label %rrj, label %rrdone

rrdone:
  %irn = add nuw i64 %ir, 1
  %irmore = icmp ult i64 %irn, %m
  br i1 %irmore, label %rrloop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
}

; matmul_scalar — triple-loop oracle. C[i,j] = sum_p A[i,p]*B[p,j].
define i32 @universe_linalg_matmul_scalar(ptr readonly %A, i64 %m, i64 %k, ptr readonly %B, i64 %k2, i64 %n, ptr %C) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nB = icmp eq ptr %B, null
  %nC = icmp eq ptr %C, null
  %n0 = or i1 %nA, %nB
  %nz = or i1 %n0, %nC
  br i1 %nz, label %enull, label %shape

shape:
  %badk = icmp ne i64 %k, %k2
  br i1 %badk, label %einval, label %dims

dims:
  %me = icmp eq i64 %m, 0
  %nez = icmp eq i64 %n, 0
  %empty = or i1 %me, %nez
  br i1 %empty, label %ok, label %iloop

iloop:
  %i = phi i64 [ 0, %dims ], [ %inext, %idone ]
  %aoff = mul nuw i64 %i, %k
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %coff = mul nuw i64 %i, %n
  br label %jloop

jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnext, %pstore ]
  br label %ploop

ploop:
  %p = phi i64 [ 0, %jloop ], [ %pn, %ploop ]
  %acc = phi double [ 0.0, %jloop ], [ %accn, %ploop ]
  %ap = getelementptr inbounds double, ptr %arow, i64 %p
  %av = load double, ptr %ap, align 8
  %bo = mul nuw i64 %p, %n
  %bo2 = add nuw i64 %bo, %j
  %bp = getelementptr inbounds double, ptr %B, i64 %bo2
  %bv = load double, ptr %bp, align 8
  %mm = fmul double %av, %bv
  %accn = fadd double %acc, %mm
  %pn = add nuw i64 %p, 1
  %pmore = icmp ult i64 %pn, %k
  br i1 %pmore, label %ploop, label %pstore

pstore:
  %cidx = add nuw i64 %coff, %j
  %cp = getelementptr inbounds double, ptr %C, i64 %cidx
  store double %accn, ptr %cp, align 8
  %jnext = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jnext, %n
  br i1 %jmore, label %jloop, label %idone

idone:
  %inext = add nuw i64 %i, 1
  %imore = icmp ult i64 %inext, %m
  br i1 %imore, label %iloop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
}

; ==================================================================== matvec

define i32 @universe_linalg_matvec(ptr readonly %A, i64 %m, i64 %n, ptr readonly %x, ptr %y) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nx = icmp eq ptr %x, null
  %ny = icmp eq ptr %y, null
  %a0 = or i1 %nA, %nx
  %az = or i1 %a0, %ny
  br i1 %az, label %enull, label %ovf

ovf:
  %mn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %o0 = extractvalue { i64, i1 } %mn, 1
  br i1 %o0, label %eovf, label %dims

dims:
  %me = icmp eq i64 %m, 0
  br i1 %me, label %ok, label %loop

loop:
  %i = phi i64 [ 0, %dims ], [ %inext, %loop ]
  %off = mul nuw i64 %i, %n
  %row = getelementptr inbounds double, ptr %A, i64 %off
  %dp = call double @li_dot_impl(ptr %row, ptr %x, i64 %n)
  %yp = getelementptr inbounds double, ptr %y, i64 %i
  store double %dp, ptr %yp, align 8
  %inext = add nuw i64 %i, 1
  %more = icmp ult i64 %inext, %m
  br i1 %more, label %loop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
eovf:
  ret i32 3
}

define i32 @universe_linalg_matvec_scalar(ptr readonly %A, i64 %m, i64 %n, ptr readonly %x, ptr %y) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nx = icmp eq ptr %x, null
  %ny = icmp eq ptr %y, null
  %a0 = or i1 %nA, %nx
  %az = or i1 %a0, %ny
  br i1 %az, label %enull, label %dims

dims:
  %me = icmp eq i64 %m, 0
  br i1 %me, label %ok, label %loop

loop:
  %i = phi i64 [ 0, %dims ], [ %inext, %idone ]
  %off = mul nuw i64 %i, %n
  %row = getelementptr inbounds double, ptr %A, i64 %off
  %nz = icmp eq i64 %n, 0
  br i1 %nz, label %store, label %dploop

dploop:
  %p = phi i64 [ 0, %loop ], [ %pn, %dploop ]
  %acc = phi double [ 0.0, %loop ], [ %accn, %dploop ]
  %ap = getelementptr inbounds double, ptr %row, i64 %p
  %av = load double, ptr %ap, align 8
  %xp = getelementptr inbounds double, ptr %x, i64 %p
  %xv = load double, ptr %xp, align 8
  %mm = fmul double %av, %xv
  %accn = fadd double %acc, %mm
  %pn = add nuw i64 %p, 1
  %pmore = icmp ult i64 %pn, %n
  br i1 %pmore, label %dploop, label %store

store:
  %val = phi double [ 0.0, %loop ], [ %accn, %dploop ]
  %yp = getelementptr inbounds double, ptr %y, i64 %i
  store double %val, ptr %yp, align 8
  br label %idone

idone:
  %inext = add nuw i64 %i, 1
  %more = icmp ult i64 %inext, %m
  br i1 %more, label %loop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
}

; ============================================================ element-wise ops

; mul_scalar: out = s*A over n elements (vector <2 x double> + scalar tail).
define i32 @universe_linalg_mul_scalar(ptr readonly %A, i64 %n, double %s, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %no = icmp eq ptr %out, null
  %nz = or i1 %nA, %no
  br i1 %nz, label %enull, label %init

init:
  %si = insertelement <2 x double> poison, double %s, i64 0
  %sb = shufflevector <2 x double> %si, <2 x double> poison, <2 x i32> zeroinitializer
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %tail

vloop:
  %i = phi i64 [ 0, %init ], [ %inext, %vloop ]
  %p = getelementptr inbounds nuw double, ptr %A, i64 %i
  %v = load <2 x double>, ptr %p, align 8
  %r = fmul <2 x double> %v, %sb
  %op = getelementptr inbounds nuw double, ptr %out, i64 %i
  store <2 x double> %r, ptr %op, align 8
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %tail

tail:
  %ti = phi i64 [ 0, %init ], [ %inext, %vloop ], [ %tin, %tbody ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %ok, label %tbody

tbody:
  %tp = getelementptr inbounds nuw double, ptr %A, i64 %ti
  %tv = load double, ptr %tp, align 8
  %tr = fmul double %tv, %s
  %top = getelementptr inbounds nuw double, ptr %out, i64 %ti
  store double %tr, ptr %top, align 8
  %tin = add nuw i64 %ti, 1
  br label %tail

ok:
  ret i32 0
enull:
  ret i32 1
}

define i32 @universe_linalg_add(ptr readonly %A, ptr readonly %B, i64 %n, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nB = icmp eq ptr %B, null
  %no = icmp eq ptr %out, null
  %n0 = or i1 %nA, %nB
  %nz = or i1 %n0, %no
  br i1 %nz, label %enull, label %init

init:
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %tail

vloop:
  %i = phi i64 [ 0, %init ], [ %inext, %vloop ]
  %pa = getelementptr inbounds nuw double, ptr %A, i64 %i
  %va = load <2 x double>, ptr %pa, align 8
  %pb = getelementptr inbounds nuw double, ptr %B, i64 %i
  %vb = load <2 x double>, ptr %pb, align 8
  %r = fadd <2 x double> %va, %vb
  %op = getelementptr inbounds nuw double, ptr %out, i64 %i
  store <2 x double> %r, ptr %op, align 8
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %tail

tail:
  %ti = phi i64 [ 0, %init ], [ %inext, %vloop ], [ %tin, %tbody ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %ok, label %tbody

tbody:
  %tpa = getelementptr inbounds nuw double, ptr %A, i64 %ti
  %fa = load double, ptr %tpa, align 8
  %tpb = getelementptr inbounds nuw double, ptr %B, i64 %ti
  %fb = load double, ptr %tpb, align 8
  %tr = fadd double %fa, %fb
  %top = getelementptr inbounds nuw double, ptr %out, i64 %ti
  store double %tr, ptr %top, align 8
  %tin = add nuw i64 %ti, 1
  br label %tail

ok:
  ret i32 0
enull:
  ret i32 1
}

define i32 @universe_linalg_sub(ptr readonly %A, ptr readonly %B, i64 %n, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nB = icmp eq ptr %B, null
  %no = icmp eq ptr %out, null
  %n0 = or i1 %nA, %nB
  %nz = or i1 %n0, %no
  br i1 %nz, label %enull, label %init

init:
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %tail

vloop:
  %i = phi i64 [ 0, %init ], [ %inext, %vloop ]
  %pa = getelementptr inbounds nuw double, ptr %A, i64 %i
  %va = load <2 x double>, ptr %pa, align 8
  %pb = getelementptr inbounds nuw double, ptr %B, i64 %i
  %vb = load <2 x double>, ptr %pb, align 8
  %r = fsub <2 x double> %va, %vb
  %op = getelementptr inbounds nuw double, ptr %out, i64 %i
  store <2 x double> %r, ptr %op, align 8
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %tail

tail:
  %ti = phi i64 [ 0, %init ], [ %inext, %vloop ], [ %tin, %tbody ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %ok, label %tbody

tbody:
  %tpa = getelementptr inbounds nuw double, ptr %A, i64 %ti
  %fa = load double, ptr %tpa, align 8
  %tpb = getelementptr inbounds nuw double, ptr %B, i64 %ti
  %fb = load double, ptr %tpb, align 8
  %tr = fsub double %fa, %fb
  %top = getelementptr inbounds nuw double, ptr %out, i64 %ti
  store double %tr, ptr %top, align 8
  %tin = add nuw i64 %ti, 1
  br label %tail

ok:
  ret i32 0
enull:
  ret i32 1
}

; ==================================================================== transpose
; out is n x m: out[j*m + i] = A[i*n + j].
define i32 @universe_linalg_transpose(ptr readonly %A, i64 %m, i64 %n, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %no = icmp eq ptr %out, null
  %nz = or i1 %nA, %no
  br i1 %nz, label %enull, label %ovf

ovf:
  %mn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %o0 = extractvalue { i64, i1 } %mn, 1
  br i1 %o0, label %eovf, label %dims

dims:
  %me = icmp eq i64 %m, 0
  %ne = icmp eq i64 %n, 0
  %empty = or i1 %me, %ne
  br i1 %empty, label %ok, label %iloop

iloop:
  %i = phi i64 [ 0, %dims ], [ %inext, %idone ]
  %arow = mul nuw i64 %i, %n
  br label %jloop

jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnext, %jloop ]
  %aidx = add nuw i64 %arow, %j
  %ap = getelementptr inbounds double, ptr %A, i64 %aidx
  %v = load double, ptr %ap, align 8
  %ocol = mul nuw i64 %j, %m
  %oidx = add nuw i64 %ocol, %i
  %op = getelementptr inbounds double, ptr %out, i64 %oidx
  store double %v, ptr %op, align 8
  %jnext = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jnext, %n
  br i1 %jmore, label %jloop, label %idone

idone:
  %inext = add nuw i64 %i, 1
  %imore = icmp ult i64 %inext, %m
  br i1 %imore, label %iloop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
eovf:
  ret i32 3
}

; ==================================================================== dot / norm

define double @universe_linalg_dot(ptr readonly %x, ptr readonly %y, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %go
go:
  %r = call double @li_dot_impl(ptr %x, ptr %y, i64 %n)
  ret double %r
zero:
  ret double 0.0
}

define double @universe_linalg_dot_scalar(ptr readonly %x, ptr readonly %y, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %px = getelementptr inbounds nuw double, ptr %x, i64 %i
  %fx = load double, ptr %px, align 8
  %py = getelementptr inbounds nuw double, ptr %y, i64 %i
  %fy = load double, ptr %py, align 8
  %m = fmul double %fx, %fy
  %accn = fadd double %acc, %m
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi double [ 0.0, %entry ], [ %accn, %loop ]
  ret double %r
}

; norm: p==2 -> sqrt(dot(x,x)); p==1 -> sum|x|; p==0 -> max|x| (Inf); else l2.
define double @universe_linalg_norm(ptr readonly %x, i64 %n, i32 %p) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %sw
sw:
  switch i32 %p, label %l2 [ i32 1, label %l1
                             i32 0, label %linf ]
l2:
  %d = call double @li_dot_impl(ptr %x, ptr %x, i64 %n)
  %r2 = call double @llvm.sqrt.f64(double %d)
  ret double %r2
l1:
  %r1 = call double @li_asum_impl(ptr %x, i64 %n)
  ret double %r1
linf:
  %ri = call double @li_amax_impl(ptr %x, i64 %n)
  ret double %ri
zero:
  ret double 0.0
}

define double @universe_linalg_norm_scalar(ptr readonly %x, i64 %n, i32 %p) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %sa = phi double [ 0.0, %entry ], [ %san, %loop ]
  %s1 = phi double [ 0.0, %entry ], [ %s1n, %loop ]
  %mx = phi double [ 0.0, %entry ], [ %mxn, %loop ]
  %px = getelementptr inbounds nuw double, ptr %x, i64 %i
  %v = load double, ptr %px, align 8
  %sq = fmul double %v, %v
  %san = fadd double %sa, %sq
  %av = call double @llvm.fabs.f64(double %v)
  %s1n = fadd double %s1, %av
  %gt = fcmp ogt double %av, %mx
  %mxn = select i1 %gt, double %av, double %mx
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %sel
sel:
  switch i32 %p, label %l2 [ i32 1, label %l1
                             i32 0, label %linf ]
l2:
  %r2 = call double @llvm.sqrt.f64(double %san)
  ret double %r2
l1:
  ret double %s1n
linf:
  ret double %mxn
zero:
  ret double 0.0
}

define double @universe_linalg_norm_fro(ptr readonly %A, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %go
go:
  %d = call double @li_dot_impl(ptr %A, ptr %A, i64 %n)
  %r = call double @llvm.sqrt.f64(double %d)
  ret double %r
zero:
  ret double 0.0
}

define double @universe_linalg_norm_fro_scalar(ptr readonly %A, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %px = getelementptr inbounds nuw double, ptr %A, i64 %i
  %v = load double, ptr %px, align 8
  %sq = fmul double %v, %v
  %accn = fadd double %acc, %sq
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = call double @llvm.sqrt.f64(double %accn)
  ret double %r
zero:
  ret double 0.0
}

; normalize: out = x / norm2(x). Returns 8 (INVALID_ARG) if norm2 == 0.
define i32 @universe_linalg_normalize(ptr readonly %x, i64 %n, ptr %out) #1 {
entry:
  %nx = icmp eq ptr %x, null
  %no = icmp eq ptr %out, null
  %nz = or i1 %nx, %no
  br i1 %nz, label %enull, label %chkn
chkn:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %einval, label %norm
norm:
  %d = call double @li_dot_impl(ptr %x, ptr %x, i64 %n)
  %nrm = call double @llvm.sqrt.f64(double %d)
  %isz = fcmp oeq double %nrm, 0.0
  br i1 %isz, label %einval, label %go
go:
  %inv = fdiv double 1.0, %nrm
  %rc = call i32 @universe_linalg_mul_scalar(ptr %x, i64 %n, double %inv, ptr %out)
  ret i32 %rc
enull:
  ret i32 1
einval:
  ret i32 8
}

; ==================================================================== cross
; 3-vectors: out = a x b.
define i32 @universe_linalg_cross(ptr readonly %a, ptr readonly %b, ptr %out) #1 {
entry:
  %na = icmp eq ptr %a, null
  %nb = icmp eq ptr %b, null
  %no = icmp eq ptr %out, null
  %n0 = or i1 %na, %nb
  %nz = or i1 %n0, %no
  br i1 %nz, label %enull, label %go
go:
  %a0p = getelementptr inbounds double, ptr %a, i64 0
  %a0 = load double, ptr %a0p, align 8
  %a1p = getelementptr inbounds double, ptr %a, i64 1
  %a1 = load double, ptr %a1p, align 8
  %a2p = getelementptr inbounds double, ptr %a, i64 2
  %a2 = load double, ptr %a2p, align 8
  %b0p = getelementptr inbounds double, ptr %b, i64 0
  %b0 = load double, ptr %b0p, align 8
  %b1p = getelementptr inbounds double, ptr %b, i64 1
  %b1 = load double, ptr %b1p, align 8
  %b2p = getelementptr inbounds double, ptr %b, i64 2
  %b2 = load double, ptr %b2p, align 8
  ; out0 = a1*b2 - a2*b1
  %m12 = fmul double %a1, %b2
  %m21 = fmul double %a2, %b1
  %o0 = fsub double %m12, %m21
  ; out1 = a2*b0 - a0*b2
  %m20 = fmul double %a2, %b0
  %m02 = fmul double %a0, %b2
  %o1 = fsub double %m20, %m02
  ; out2 = a0*b1 - a1*b0
  %m01 = fmul double %a0, %b1
  %m10 = fmul double %a1, %b0
  %o2 = fsub double %m01, %m10
  %o0p = getelementptr inbounds double, ptr %out, i64 0
  store double %o0, ptr %o0p, align 8
  %o1p = getelementptr inbounds double, ptr %out, i64 1
  store double %o1, ptr %o1p, align 8
  %o2p = getelementptr inbounds double, ptr %out, i64 2
  store double %o2, ptr %o2p, align 8
  ret i32 0
enull:
  ret i32 1
}

; ==================================================================== trace / diag

define double @universe_linalg_tr(ptr readonly %A, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %accn, %loop ]
  %stride = add nuw i64 %n, 1
  %idx = mul nuw i64 %i, %stride
  %p = getelementptr inbounds double, ptr %A, i64 %idx
  %v = load double, ptr %p, align 8
  %accn = fadd double %acc, %v
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret double %accn
zero:
  ret double 0.0
}

; diag: out[i] = A[i,i] for i in 0..n (A is n x n).
define i32 @universe_linalg_diag(ptr readonly %A, i64 %n, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %no = icmp eq ptr %out, null
  %nz = or i1 %nA, %no
  br i1 %nz, label %enull, label %dims
dims:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %ok, label %loop
loop:
  %i = phi i64 [ 0, %dims ], [ %in, %loop ]
  %stride = add nuw i64 %n, 1
  %idx = mul nuw i64 %i, %stride
  %p = getelementptr inbounds double, ptr %A, i64 %idx
  %v = load double, ptr %p, align 8
  %op = getelementptr inbounds double, ptr %out, i64 %i
  store double %v, ptr %op, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %ok
ok:
  ret i32 0
enull:
  ret i32 1
}

; diagm: out = diag(v), an n x n matrix (zeroed, then v on the diagonal).
define i32 @universe_linalg_diagm(ptr readonly %v, i64 %n, ptr %out) #1 {
entry:
  %nv = icmp eq ptr %v, null
  %no = icmp eq ptr %out, null
  %nz = or i1 %nv, %no
  br i1 %nz, label %enull, label %ovf
ovf:
  %nn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %cnt = extractvalue { i64, i1 } %nn, 0
  %o0 = extractvalue { i64, i1 } %nn, 1
  br i1 %o0, label %eovf, label %dims
dims:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %ok, label %zeroit
zeroit:
  %bytes = shl nuw i64 %cnt, 3
  call void @llvm.memset.p0.i64(ptr %out, i8 0, i64 %bytes, i1 false)
  br label %loop
loop:
  %i = phi i64 [ 0, %zeroit ], [ %in, %loop ]
  %vp = getelementptr inbounds double, ptr %v, i64 %i
  %val = load double, ptr %vp, align 8
  %stride = add nuw i64 %n, 1
  %idx = mul nuw i64 %i, %stride
  %op = getelementptr inbounds double, ptr %out, i64 %idx
  store double %val, ptr %op, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %ok
ok:
  ret i32 0
enull:
  ret i32 1
eovf:
  ret i32 3
}

; identity: out = I_n (n x n).
define i32 @universe_linalg_identity(ptr %out, i64 %n) #1 {
entry:
  %no = icmp eq ptr %out, null
  br i1 %no, label %enull, label %ovf
ovf:
  %nn = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %cnt = extractvalue { i64, i1 } %nn, 0
  %o0 = extractvalue { i64, i1 } %nn, 1
  br i1 %o0, label %eovf, label %dims
dims:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %ok, label %zeroit
zeroit:
  %bytes = shl nuw i64 %cnt, 3
  call void @llvm.memset.p0.i64(ptr %out, i8 0, i64 %bytes, i1 false)
  br label %loop
loop:
  %i = phi i64 [ 0, %zeroit ], [ %in, %loop ]
  %stride = add nuw i64 %n, 1
  %idx = mul nuw i64 %i, %stride
  %op = getelementptr inbounds double, ptr %out, i64 %idx
  store double 1.0, ptr %op, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %ok
ok:
  ret i32 0
enull:
  ret i32 1
eovf:
  ret i32 3
}

; ==================================================================== kron
; Kronecker product: A (am x an) (x) B (bm x bn) -> out ((am*bm) x (an*bn)).
; out[(i*bm+p), (j*bn+q)] = A[i,j] * B[p,q].
define i32 @universe_linalg_kron(ptr readonly %A, i64 %am, i64 %an, ptr readonly %B, i64 %bm, i64 %bn, ptr %out) #1 {
entry:
  %nA = icmp eq ptr %A, null
  %nB = icmp eq ptr %B, null
  %no = icmp eq ptr %out, null
  %x0 = or i1 %nA, %nB
  %nz = or i1 %x0, %no
  br i1 %nz, label %enull, label %ovf
ovf:
  ; out columns = an*bn (leading dim); rows = am*bm; guard both products.
  %cc = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %an, i64 %bn)
  %ocols = extractvalue { i64, i1 } %cc, 0
  %oc0 = extractvalue { i64, i1 } %cc, 1
  %rr = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %am, i64 %bm)
  %or0 = extractvalue { i64, i1 } %rr, 1
  %tt = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ocols, i64 %am)
  %ot0 = extractvalue { i64, i1 } %tt, 1
  %oa = or i1 %oc0, %or0
  %ob = or i1 %oa, %ot0
  br i1 %ob, label %eovf, label %dims
dims:
  %e0 = icmp eq i64 %am, 0
  %e1 = icmp eq i64 %an, 0
  %e2 = icmp eq i64 %bm, 0
  %e3 = icmp eq i64 %bn, 0
  %ea = or i1 %e0, %e1
  %eb = or i1 %e2, %e3
  %empty = or i1 %ea, %eb
  br i1 %empty, label %ok, label %iloop

iloop:
  %i = phi i64 [ 0, %dims ], [ %in, %idone ]
  %airow = mul nuw i64 %i, %an
  br label %jloop
jloop:
  %j = phi i64 [ 0, %iloop ], [ %jn, %jdone ]
  %aidx = add nuw i64 %airow, %j
  %ap = getelementptr inbounds double, ptr %A, i64 %aidx
  %aval = load double, ptr %ap, align 8
  ; out row block base row = i*bm ; out col block base col = j*bn
  %obr = mul nuw i64 %i, %bm
  %obc = mul nuw i64 %j, %bn
  br label %ploop
ploop:
  %p = phi i64 [ 0, %jloop ], [ %pn, %pdone ]
  %bprow = mul nuw i64 %p, %bn
  %orow = add nuw i64 %obr, %p
  %orowoff = mul nuw i64 %orow, %ocols
  %orowbase = add nuw i64 %orowoff, %obc
  br label %qloop
qloop:
  %q = phi i64 [ 0, %ploop ], [ %qn, %qloop ]
  %bidx = add nuw i64 %bprow, %q
  %bp = getelementptr inbounds double, ptr %B, i64 %bidx
  %bval = load double, ptr %bp, align 8
  %prod = fmul double %aval, %bval
  %oidx = add nuw i64 %orowbase, %q
  %op = getelementptr inbounds double, ptr %out, i64 %oidx
  store double %prod, ptr %op, align 8
  %qn = add nuw i64 %q, 1
  %qmore = icmp ult i64 %qn, %bn
  br i1 %qmore, label %qloop, label %pdone
pdone:
  %pn = add nuw i64 %p, 1
  %pmore = icmp ult i64 %pn, %bm
  br i1 %pmore, label %ploop, label %jdone
jdone:
  %jn = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jn, %an
  br i1 %jmore, label %jloop, label %idone
idone:
  %in = add nuw i64 %i, 1
  %imore = icmp ult i64 %in, %am
  br i1 %imore, label %iloop, label %ok

ok:
  ret i32 0
enull:
  ret i32 1
eovf:
  ret i32 3
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
