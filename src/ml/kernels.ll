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

; universe_ml_* — SIMD numeric kernels (BLAS-1/BLAS-2 substrate for the ML
; engines: k-means, kNN, linear/logistic models). f32, row-major, unit stride.
; Pure compute over caller-owned memory; never allocates.
;
; DESIGN — SIMD-first with scalar fallback/oracle (canonical embodiment):
;   * PRIMARY PATH is a portable <4 x float> vector loop. It lowers to SSE
;     (x86) and NEON (AArch64) — both baseline everywhere we ship — so no
;     runtime CPU check. Reductions (dot/sum/l2/dist) run FOUR independent
;     <4 x float> accumulators (16 floats/iter) to break the loop-carried FP
;     dependency chain: the four fmul/fadd (fmla) issue in parallel and the
;     pipeline stays full. The four partials fold once, then a horizontal
;     vector reduce, then a scalar tail (<16 elems) finishes the remainder.
;   * SCALAR PATH is the fallback AND the oracle. Every reduction/kernel ships
;     a `*_scalar` twin: a straight sequential loop used to (a) handle the sub-4
;     remainder and (b) cross-check the vector result in tests. A kernel is not
;     "done" until |vector - scalar| < 1e-4 * |scalar| on random inputs.
;   * FP REORDERING is DELIBERATE and documented. The reduction hot paths use
;     `fast` (reassoc + contract + nsz + arcp) because summation is
;     associative to within tolerance and we WANT: (a) the four-accumulator
;     tree reorder, (b) fmul+fadd contracted to a single fmla, (c) -0.0 as the
;     additive identity in the horizontal reduce. The `*_scalar` oracles use
;     UNFLAGGED fadd/fmul (strict left-to-right) so they are a stable
;     reference; the two agree within relative tolerance, never bit-for-bit.
;     argmin/argmax use NO fast flags on the compares — the located index must
;     be exact and deterministic (first occurrence, lowest index).
;   * argmin/argmax PRIMARY: a vector min/max-value reduction (llvm.minnum /
;     maxnum over <4 x float>, horizontal reduce) then a scalar `oeq` scan for
;     the FIRST index holding that exact value. Because minnum/maxnum select
;     an actual element (no rounding), the `oeq` always hits and the result is
;     bit-identical to the scalar single-pass strict-`<`/`>` twin.
;
; API (f32; n/d/m are element counts; row-major matrices):
;   float universe_ml_dot(a, b, n)                 ; sum a[i]*b[i]
;   float universe_ml_dot_scalar(a, b, n)
;   void  universe_ml_axpy(y, alpha, x, n)         ; y[i] += alpha*x[i]
;   void  universe_ml_axpy_scalar(y, alpha, x, n)
;   void  universe_ml_scale(x, alpha, n)           ; x[i] *= alpha (in place)
;   void  universe_ml_scale_scalar(x, alpha, n)
;   float universe_ml_sum(x, n)
;   float universe_ml_sum_scalar(x, n)
;   float universe_ml_mean(x, n)                   ; 0 if n==0
;   float universe_ml_mean_scalar(x, n)
;   float universe_ml_l2_norm(x, n)                ; sqrt(dot(x,x))
;   float universe_ml_l2_norm_scalar(x, n)
;   float universe_ml_l2_dist2(a, b, n)            ; sum (a[i]-b[i])^2
;   float universe_ml_l2_dist2_scalar(a, b, n)
;   float universe_ml_l1_dist(a, b, n)             ; sum |a[i]-b[i]|
;   float universe_ml_l1_dist_scalar(a, b, n)
;   void  universe_ml_gemv(y, A, x, m, n)          ; y = A*x, A is m x n
;   void  universe_ml_gemv_scalar(y, A, x, m, n)
;   i64   universe_ml_argmin(x, n)                 ; first index of min, -1 if n==0
;   i64   universe_ml_argmin_scalar(x, n)
;   i64   universe_ml_argmax(x, n)                 ; first index of max, -1 if n==0
;   i64   universe_ml_argmax_scalar(x, n)

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare float @llvm.vector.reduce.fmin.v4f32(<4 x float>)
declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare float @llvm.fabs.f32(float)
declare float @llvm.sqrt.f32(float)

; ============================================================ dot (internal)
; 4-accumulator <4 x float> reduction of sum a[i]*b[i]. alwaysinline: folds
; into dot / l2_norm / gemv at zero cost within this module.
define internal float @ml_dot_impl(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %pa0 = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va0 = load <4 x float>, ptr %pa0, align 4
  %pb0 = getelementptr inbounds nuw float, ptr %b, i64 %i
  %vb0 = load <4 x float>, ptr %pb0, align 4
  %m0 = fmul fast <4 x float> %va0, %vb0
  %acc0n = fadd fast <4 x float> %acc0, %m0
  %i1 = add nuw i64 %i, 4
  %pa1 = getelementptr inbounds nuw float, ptr %a, i64 %i1
  %va1 = load <4 x float>, ptr %pa1, align 4
  %pb1 = getelementptr inbounds nuw float, ptr %b, i64 %i1
  %vb1 = load <4 x float>, ptr %pb1, align 4
  %m1 = fmul fast <4 x float> %va1, %vb1
  %acc1n = fadd fast <4 x float> %acc1, %m1
  %i2 = add nuw i64 %i, 8
  %pa2 = getelementptr inbounds nuw float, ptr %a, i64 %i2
  %va2 = load <4 x float>, ptr %pa2, align 4
  %pb2 = getelementptr inbounds nuw float, ptr %b, i64 %i2
  %vb2 = load <4 x float>, ptr %pb2, align 4
  %m2 = fmul fast <4 x float> %va2, %vb2
  %acc2n = fadd fast <4 x float> %acc2, %m2
  %i3 = add nuw i64 %i, 12
  %pa3 = getelementptr inbounds nuw float, ptr %a, i64 %i3
  %va3 = load <4 x float>, ptr %pa3, align 4
  %pb3 = getelementptr inbounds nuw float, ptr %b, i64 %i3
  %vb3 = load <4 x float>, ptr %pb3, align 4
  %m3 = fmul fast <4 x float> %va3, %vb3
  %acc3n = fadd fast <4 x float> %acc3, %m3
  %inext = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <4 x float> %acc0n, %acc1n
  %c1 = fadd fast <4 x float> %acc2n, %acc3n
  %csum = fadd fast <4 x float> %c0, %c1
  %hs = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %csum)
  br label %red

red:
  %base = phi float [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi float [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw float, ptr %a, i64 %j
  %fa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds nuw float, ptr %b, i64 %j
  %fb = load float, ptr %tb, align 4
  %p = fmul fast float %fa, %fb
  %saccn = fadd fast float %sacc, %p
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret float %sacc
}

; ============================================================ sum (internal)
define internal float @ml_sum_impl(ptr readonly %x, i64 %n) #2 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %p0 = getelementptr inbounds nuw float, ptr %x, i64 %i
  %v0 = load <4 x float>, ptr %p0, align 4
  %acc0n = fadd fast <4 x float> %acc0, %v0
  %i1 = add nuw i64 %i, 4
  %p1 = getelementptr inbounds nuw float, ptr %x, i64 %i1
  %v1 = load <4 x float>, ptr %p1, align 4
  %acc1n = fadd fast <4 x float> %acc1, %v1
  %i2 = add nuw i64 %i, 8
  %p2 = getelementptr inbounds nuw float, ptr %x, i64 %i2
  %v2 = load <4 x float>, ptr %p2, align 4
  %acc2n = fadd fast <4 x float> %acc2, %v2
  %i3 = add nuw i64 %i, 12
  %p3 = getelementptr inbounds nuw float, ptr %x, i64 %i3
  %v3 = load <4 x float>, ptr %p3, align 4
  %acc3n = fadd fast <4 x float> %acc3, %v3
  %inext = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <4 x float> %acc0n, %acc1n
  %c1 = fadd fast <4 x float> %acc2n, %acc3n
  %csum = fadd fast <4 x float> %c0, %c1
  %hs = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %csum)
  br label %red

red:
  %base = phi float [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi float [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %tp = getelementptr inbounds nuw float, ptr %x, i64 %j
  %fx = load float, ptr %tp, align 4
  %saccn = fadd fast float %sacc, %fx
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret float %sacc
}

; ========================================================== dist2 (internal)
define internal float @ml_dist2_impl(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %pa0 = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va0 = load <4 x float>, ptr %pa0, align 4
  %pb0 = getelementptr inbounds nuw float, ptr %b, i64 %i
  %vb0 = load <4 x float>, ptr %pb0, align 4
  %d0 = fsub fast <4 x float> %va0, %vb0
  %sq0 = fmul fast <4 x float> %d0, %d0
  %acc0n = fadd fast <4 x float> %acc0, %sq0
  %i1 = add nuw i64 %i, 4
  %pa1 = getelementptr inbounds nuw float, ptr %a, i64 %i1
  %va1 = load <4 x float>, ptr %pa1, align 4
  %pb1 = getelementptr inbounds nuw float, ptr %b, i64 %i1
  %vb1 = load <4 x float>, ptr %pb1, align 4
  %d1 = fsub fast <4 x float> %va1, %vb1
  %sq1 = fmul fast <4 x float> %d1, %d1
  %acc1n = fadd fast <4 x float> %acc1, %sq1
  %i2 = add nuw i64 %i, 8
  %pa2 = getelementptr inbounds nuw float, ptr %a, i64 %i2
  %va2 = load <4 x float>, ptr %pa2, align 4
  %pb2 = getelementptr inbounds nuw float, ptr %b, i64 %i2
  %vb2 = load <4 x float>, ptr %pb2, align 4
  %d2 = fsub fast <4 x float> %va2, %vb2
  %sq2 = fmul fast <4 x float> %d2, %d2
  %acc2n = fadd fast <4 x float> %acc2, %sq2
  %i3 = add nuw i64 %i, 12
  %pa3 = getelementptr inbounds nuw float, ptr %a, i64 %i3
  %va3 = load <4 x float>, ptr %pa3, align 4
  %pb3 = getelementptr inbounds nuw float, ptr %b, i64 %i3
  %vb3 = load <4 x float>, ptr %pb3, align 4
  %d3 = fsub fast <4 x float> %va3, %vb3
  %sq3 = fmul fast <4 x float> %d3, %d3
  %acc3n = fadd fast <4 x float> %acc3, %sq3
  %inext = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <4 x float> %acc0n, %acc1n
  %c1 = fadd fast <4 x float> %acc2n, %acc3n
  %csum = fadd fast <4 x float> %c0, %c1
  %hs = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %csum)
  br label %red

red:
  %base = phi float [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi float [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw float, ptr %a, i64 %j
  %fa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds nuw float, ptr %b, i64 %j
  %fb = load float, ptr %tb, align 4
  %d = fsub fast float %fa, %fb
  %sq = fmul fast float %d, %d
  %saccn = fadd fast float %sacc, %sq
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret float %sacc
}

; ============================================================= l1 (internal)
define internal float @ml_l1_impl(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %pa0 = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va0 = load <4 x float>, ptr %pa0, align 4
  %pb0 = getelementptr inbounds nuw float, ptr %b, i64 %i
  %vb0 = load <4 x float>, ptr %pb0, align 4
  %d0 = fsub fast <4 x float> %va0, %vb0
  %ab0 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %d0)
  %acc0n = fadd fast <4 x float> %acc0, %ab0
  %i1 = add nuw i64 %i, 4
  %pa1 = getelementptr inbounds nuw float, ptr %a, i64 %i1
  %va1 = load <4 x float>, ptr %pa1, align 4
  %pb1 = getelementptr inbounds nuw float, ptr %b, i64 %i1
  %vb1 = load <4 x float>, ptr %pb1, align 4
  %d1 = fsub fast <4 x float> %va1, %vb1
  %ab1 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %d1)
  %acc1n = fadd fast <4 x float> %acc1, %ab1
  %i2 = add nuw i64 %i, 8
  %pa2 = getelementptr inbounds nuw float, ptr %a, i64 %i2
  %va2 = load <4 x float>, ptr %pa2, align 4
  %pb2 = getelementptr inbounds nuw float, ptr %b, i64 %i2
  %vb2 = load <4 x float>, ptr %pb2, align 4
  %d2 = fsub fast <4 x float> %va2, %vb2
  %ab2 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %d2)
  %acc2n = fadd fast <4 x float> %acc2, %ab2
  %i3 = add nuw i64 %i, 12
  %pa3 = getelementptr inbounds nuw float, ptr %a, i64 %i3
  %va3 = load <4 x float>, ptr %pa3, align 4
  %pb3 = getelementptr inbounds nuw float, ptr %b, i64 %i3
  %vb3 = load <4 x float>, ptr %pb3, align 4
  %d3 = fsub fast <4 x float> %va3, %vb3
  %ab3 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %d3)
  %acc3n = fadd fast <4 x float> %acc3, %ab3
  %inext = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <4 x float> %acc0n, %acc1n
  %c1 = fadd fast <4 x float> %acc2n, %acc3n
  %csum = fadd fast <4 x float> %c0, %c1
  %hs = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %csum)
  br label %red

red:
  %base = phi float [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi float [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw float, ptr %a, i64 %j
  %fa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds nuw float, ptr %b, i64 %j
  %fb = load float, ptr %tb, align 4
  %d = fsub fast float %fa, %fb
  %ab = call float @llvm.fabs.f32(float %d)
  %saccn = fadd fast float %sacc, %ab
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret float %sacc
}

; =================================================================== dot API
define float @universe_ml_dot(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %r = call float @ml_dot_impl(ptr %a, ptr %b, i64 %n)
  ret float %r
}

define float @universe_ml_dot_scalar(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %loop ]
  %pa = getelementptr inbounds nuw float, ptr %a, i64 %i
  %fa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds nuw float, ptr %b, i64 %i
  %fb = load float, ptr %pb, align 4
  %m = fmul float %fa, %fb
  %accn = fadd float %acc, %m
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi float [ 0.0, %entry ], [ %accn, %loop ]
  ret float %r
}

; =================================================================== sum API
define float @universe_ml_sum(ptr readonly %x, i64 %n) #0 {
entry:
  %r = call float @ml_sum_impl(ptr %x, i64 %n)
  ret float %r
}

define float @universe_ml_sum_scalar(ptr readonly %x, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %loop ]
  %p = getelementptr inbounds nuw float, ptr %x, i64 %i
  %fx = load float, ptr %p, align 4
  %accn = fadd float %acc, %fx
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi float [ 0.0, %entry ], [ %accn, %loop ]
  ret float %r
}

; ================================================================== mean API
define float @universe_ml_mean(ptr readonly %x, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %go

go:
  %s = call float @ml_sum_impl(ptr %x, i64 %n)
  %nf = uitofp i64 %n to float
  %m = fdiv fast float %s, %nf
  ret float %m

zero:
  ret float 0.0
}

define float @universe_ml_mean_scalar(ptr readonly %x, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %go

go:
  %s = call float @universe_ml_sum_scalar(ptr %x, i64 %n)
  %nf = uitofp i64 %n to float
  %m = fdiv float %s, %nf
  ret float %m

zero:
  ret float 0.0
}

; =============================================================== l2_norm API
define float @universe_ml_l2_norm(ptr readonly %x, i64 %n) #0 {
entry:
  %d = call float @ml_dot_impl(ptr %x, ptr %x, i64 %n)
  %r = call float @llvm.sqrt.f32(float %d)
  ret float %r
}

define float @universe_ml_l2_norm_scalar(ptr readonly %x, i64 %n) #0 {
entry:
  %d = call float @universe_ml_dot_scalar(ptr %x, ptr %x, i64 %n)
  %r = call float @llvm.sqrt.f32(float %d)
  ret float %r
}

; ============================================================== l2_dist2 API
define float @universe_ml_l2_dist2(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %r = call float @ml_dist2_impl(ptr %a, ptr %b, i64 %n)
  ret float %r
}

define float @universe_ml_l2_dist2_scalar(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %loop ]
  %pa = getelementptr inbounds nuw float, ptr %a, i64 %i
  %fa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds nuw float, ptr %b, i64 %i
  %fb = load float, ptr %pb, align 4
  %d = fsub float %fa, %fb
  %sq = fmul float %d, %d
  %accn = fadd float %acc, %sq
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi float [ 0.0, %entry ], [ %accn, %loop ]
  ret float %r
}

; =============================================================== l1_dist API
define float @universe_ml_l1_dist(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %r = call float @ml_l1_impl(ptr %a, ptr %b, i64 %n)
  ret float %r
}

define float @universe_ml_l1_dist_scalar(ptr readonly %a, ptr readonly %b, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %loop ]
  %pa = getelementptr inbounds nuw float, ptr %a, i64 %i
  %fa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds nuw float, ptr %b, i64 %i
  %fb = load float, ptr %pb, align 4
  %d = fsub float %fa, %fb
  %ab = call float @llvm.fabs.f32(float %d)
  %accn = fadd float %acc, %ab
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi float [ 0.0, %entry ], [ %accn, %loop ]
  ret float %r
}

; ================================================================== axpy API
; y[i] += alpha * x[i]. contract lets fadd(y, fmul(alpha,x)) fuse to fmla.
define void @universe_ml_axpy(ptr %y, float %alpha, ptr readonly %x, i64 %n) #1 {
entry:
  %ains = insertelement <4 x float> poison, float %alpha, i64 0
  %aspl = shufflevector <4 x float> %ains, <4 x float> poison, <4 x i32> zeroinitializer
  %has4 = icmp uge i64 %n, 4
  br i1 %has4, label %vloop, label %tail

vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %px = getelementptr inbounds nuw float, ptr %x, i64 %i
  %vx = load <4 x float>, ptr %px, align 4
  %py = getelementptr inbounds nuw float, ptr %y, i64 %i
  %vy = load <4 x float>, ptr %py, align 4
  %mul = fmul contract <4 x float> %aspl, %vx
  %res = fadd contract <4 x float> %vy, %mul
  store <4 x float> %res, ptr %py, align 4
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %n, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %tail

tail:
  %ti = phi i64 [ 0, %entry ], [ %inext, %vloop ], [ %tin, %tailbody ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %done, label %tailbody

tailbody:
  %tx = getelementptr inbounds nuw float, ptr %x, i64 %ti
  %fx = load float, ptr %tx, align 4
  %ty = getelementptr inbounds nuw float, ptr %y, i64 %ti
  %fy = load float, ptr %ty, align 4
  %tm = fmul contract float %alpha, %fx
  %tr = fadd contract float %fy, %tm
  store float %tr, ptr %ty, align 4
  %tin = add nuw i64 %ti, 1
  br label %tail

done:
  ret void
}

define void @universe_ml_axpy_scalar(ptr %y, float %alpha, ptr readonly %x, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %px = getelementptr inbounds nuw float, ptr %x, i64 %i
  %fx = load float, ptr %px, align 4
  %py = getelementptr inbounds nuw float, ptr %y, i64 %i
  %fy = load float, ptr %py, align 4
  %m = fmul float %alpha, %fx
  %r = fadd float %fy, %m
  store float %r, ptr %py, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ================================================================= scale API
define void @universe_ml_scale(ptr %x, float %alpha, i64 %n) #1 {
entry:
  %ains = insertelement <4 x float> poison, float %alpha, i64 0
  %aspl = shufflevector <4 x float> %ains, <4 x float> poison, <4 x i32> zeroinitializer
  %has4 = icmp uge i64 %n, 4
  br i1 %has4, label %vloop, label %tail

vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %px = getelementptr inbounds nuw float, ptr %x, i64 %i
  %vx = load <4 x float>, ptr %px, align 4
  %res = fmul <4 x float> %vx, %aspl
  store <4 x float> %res, ptr %px, align 4
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %n, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %tail

tail:
  %ti = phi i64 [ 0, %entry ], [ %inext, %vloop ], [ %tin, %tailbody ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %done, label %tailbody

tailbody:
  %tx = getelementptr inbounds nuw float, ptr %x, i64 %ti
  %fx = load float, ptr %tx, align 4
  %tr = fmul float %fx, %alpha
  store float %tr, ptr %tx, align 4
  %tin = add nuw i64 %ti, 1
  br label %tail

done:
  ret void
}

define void @universe_ml_scale_scalar(ptr %x, float %alpha, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %px = getelementptr inbounds nuw float, ptr %x, i64 %i
  %fx = load float, ptr %px, align 4
  %r = fmul float %fx, %alpha
  store float %r, ptr %px, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ================================================================== gemv API
; y = A*x, A row-major m x n; y[i] = dot(row i, x).
define void @universe_ml_gemv(ptr %y, ptr readonly %A, ptr readonly %x, i64 %m, i64 %n) #1 {
entry:
  %z = icmp eq i64 %m, 0
  br i1 %z, label %done, label %loop

loop:
  %r = phi i64 [ 0, %entry ], [ %rn, %loop ]
  %off = mul nuw i64 %r, %n
  %row = getelementptr inbounds nuw float, ptr %A, i64 %off
  %dp = call float @ml_dot_impl(ptr %row, ptr %x, i64 %n)
  %yp = getelementptr inbounds nuw float, ptr %y, i64 %r
  store float %dp, ptr %yp, align 4
  %rn = add nuw i64 %r, 1
  %more = icmp ult i64 %rn, %m
  br i1 %more, label %loop, label %done

done:
  ret void
}

define void @universe_ml_gemv_scalar(ptr %y, ptr readonly %A, ptr readonly %x, i64 %m, i64 %n) #1 {
entry:
  %z = icmp eq i64 %m, 0
  br i1 %z, label %done, label %loop

loop:
  %r = phi i64 [ 0, %entry ], [ %rn, %loop ]
  %off = mul nuw i64 %r, %n
  %row = getelementptr inbounds nuw float, ptr %A, i64 %off
  %dp = call float @universe_ml_dot_scalar(ptr %row, ptr %x, i64 %n)
  %yp = getelementptr inbounds nuw float, ptr %y, i64 %r
  store float %dp, ptr %yp, align 4
  %rn = add nuw i64 %r, 1
  %more = icmp ult i64 %rn, %m
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ================================================================ argmin API
; Vector min-value reduction, then first-index oeq scan (exact, no fast flags).
define i64 @universe_ml_argmin(ptr readonly %x, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %none, label %init

init:
  %x0 = load float, ptr %x, align 4
  %v0ins = insertelement <4 x float> poison, float %x0, i64 0
  %vinit = shufflevector <4 x float> %v0ins, <4 x float> poison, <4 x i32> zeroinitializer
  %has4 = icmp uge i64 %n, 4
  br i1 %has4, label %vloop, label %svred

vloop:
  %i = phi i64 [ 0, %init ], [ %inext, %vloop ]
  %vmin = phi <4 x float> [ %vinit, %init ], [ %vminn, %vloop ]
  %p = getelementptr inbounds nuw float, ptr %x, i64 %i
  %v = load <4 x float>, ptr %p, align 4
  %vminn = call <4 x float> @llvm.minnum.v4f32(<4 x float> %vmin, <4 x float> %v)
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %n, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hmin = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> %vminn)
  br label %svred

svred:
  %rstart = phi i64 [ 0, %init ], [ %inext, %vdone ]
  %rmin0 = phi float [ %x0, %init ], [ %hmin, %vdone ]
  br label %sloop

sloop:
  %sj = phi i64 [ %rstart, %svred ], [ %sjn, %sbody ]
  %smin = phi float [ %rmin0, %svred ], [ %sminn, %sbody ]
  %sdone = icmp uge i64 %sj, %n
  br i1 %sdone, label %locate, label %sbody

sbody:
  %sp = getelementptr inbounds nuw float, ptr %x, i64 %sj
  %sv = load float, ptr %sp, align 4
  %sminn = call float @llvm.minnum.f32(float %smin, float %sv)
  %sjn = add nuw i64 %sj, 1
  br label %sloop

locate:
  br label %lloop

lloop:
  %lj = phi i64 [ 0, %locate ], [ %ljn, %lcont ]
  %lp = getelementptr inbounds nuw float, ptr %x, i64 %lj
  %lv = load float, ptr %lp, align 4
  %hit = fcmp oeq float %lv, %smin
  br i1 %hit, label %found, label %lcont

lcont:
  %ljn = add nuw i64 %lj, 1
  br label %lloop

found:
  ret i64 %lj

none:
  ret i64 -1
}

define i64 @universe_ml_argmin_scalar(ptr readonly %x, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %none, label %init

init:
  %x0 = load float, ptr %x, align 4
  br label %loop

loop:
  %i = phi i64 [ 1, %init ], [ %in, %cont ]
  %best = phi float [ %x0, %init ], [ %bestn, %cont ]
  %bi = phi i64 [ 0, %init ], [ %bin, %cont ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %ret, label %cont

cont:
  %p = getelementptr inbounds nuw float, ptr %x, i64 %i
  %v = load float, ptr %p, align 4
  %lt = fcmp olt float %v, %best
  %bestn = select i1 %lt, float %v, float %best
  %bin = select i1 %lt, i64 %i, i64 %bi
  %in = add nuw i64 %i, 1
  br label %loop

ret:
  ret i64 %bi

none:
  ret i64 -1
}

; ================================================================ argmax API
define i64 @universe_ml_argmax(ptr readonly %x, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %none, label %init

init:
  %x0 = load float, ptr %x, align 4
  %v0ins = insertelement <4 x float> poison, float %x0, i64 0
  %vinit = shufflevector <4 x float> %v0ins, <4 x float> poison, <4 x i32> zeroinitializer
  %has4 = icmp uge i64 %n, 4
  br i1 %has4, label %vloop, label %svred

vloop:
  %i = phi i64 [ 0, %init ], [ %inext, %vloop ]
  %vmax = phi <4 x float> [ %vinit, %init ], [ %vmaxn, %vloop ]
  %p = getelementptr inbounds nuw float, ptr %x, i64 %i
  %v = load <4 x float>, ptr %p, align 4
  %vmaxn = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %vmax, <4 x float> %v)
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %n, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hmax = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %vmaxn)
  br label %svred

svred:
  %rstart = phi i64 [ 0, %init ], [ %inext, %vdone ]
  %rmax0 = phi float [ %x0, %init ], [ %hmax, %vdone ]
  br label %sloop

sloop:
  %sj = phi i64 [ %rstart, %svred ], [ %sjn, %sbody ]
  %smax = phi float [ %rmax0, %svred ], [ %smaxn, %sbody ]
  %sdone = icmp uge i64 %sj, %n
  br i1 %sdone, label %locate, label %sbody

sbody:
  %sp = getelementptr inbounds nuw float, ptr %x, i64 %sj
  %sv = load float, ptr %sp, align 4
  %smaxn = call float @llvm.maxnum.f32(float %smax, float %sv)
  %sjn = add nuw i64 %sj, 1
  br label %sloop

locate:
  br label %lloop

lloop:
  %lj = phi i64 [ 0, %locate ], [ %ljn, %lcont ]
  %lp = getelementptr inbounds nuw float, ptr %x, i64 %lj
  %lv = load float, ptr %lp, align 4
  %hit = fcmp oeq float %lv, %smax
  br i1 %hit, label %found, label %lcont

lcont:
  %ljn = add nuw i64 %lj, 1
  br label %lloop

found:
  ret i64 %lj

none:
  ret i64 -1
}

define i64 @universe_ml_argmax_scalar(ptr readonly %x, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %none, label %init

init:
  %x0 = load float, ptr %x, align 4
  br label %loop

loop:
  %i = phi i64 [ 1, %init ], [ %in, %cont ]
  %best = phi float [ %x0, %init ], [ %bestn, %cont ]
  %bi = phi i64 [ 0, %init ], [ %bin, %cont ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %ret, label %cont

cont:
  %p = getelementptr inbounds nuw float, ptr %x, i64 %i
  %v = load float, ptr %p, align 4
  %gt = fcmp ogt float %v, %best
  %bestn = select i1 %gt, float %v, float %best
  %bin = select i1 %gt, i64 %i, i64 %bi
  %in = add nuw i64 %i, 1
  br label %loop

ret:
  ret i64 %bi

none:
  ret i64 -1
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
