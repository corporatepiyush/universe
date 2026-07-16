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

; universe_ml_linreg_* / universe_ml_logreg_* — linear and logistic regression
; by batch gradient descent over row-major f32 samples.
;
; DESIGN:
;   * ALGORITHM CLASS: full-batch gradient descent on MSE (linear) / logistic
;     log-loss (binary). Both share one weight-update kernel: prediction
;     p_i = f(w·x_i + b) with f = identity (linear) or sigmoid (logistic);
;     residual r_i = p_i - y_i; gradients grad_w = (1/n) Σ r_i x_i and
;     grad_b = (1/n) Σ r_i; step w -= lr·grad_w, b -= lr·grad_b. The single
;     internal `lr_fit_impl` carries an `i1 logistic` flag (a loop-invariant
;     branch the optimizer hoists), so the two models are one implementation.
;   * HOT PATH is the w·x dot in the per-sample prediction; it uses `lr_dot`,
;     an `alwaysinline` copy of the kernels' 4-accumulator <4 x float> dot, so
;     prediction vectorizes to fmla with no cross-module call.
;   * sigmoid uses libc `expf`; ±inf inputs collapse to the correct 0/1 limits,
;     so no explicit clamp is needed for correctness.
;   * FP: `fast` on the associative gradient reductions; the SGD step itself is
;     order-independent per coordinate.
;   * MEMORY: one scratch malloc for the gradient accumulator (d f32), reused
;     across iterations, freed on exit.
;
; API (f32; X is n*d row-major; w is d):
;   i32   universe_ml_linreg_fit(X, y /*n f32*/, n, d, iters, lr,
;                                w /*out d*/, b_out /*out float*/)
;   i32   universe_ml_linreg_predict(X, n, d, w, b, out /*n f32*/)
;   i32   universe_ml_logreg_fit(X, y /*n f32 in {0,1}*/, n, d, iters, lr,
;                                w /*out d*/, b_out /*out float*/)
;   i32   universe_ml_logreg_predict_proba(X, n, d, w, b, out /*n f32*/)
;     returns 0 OK, 2 OOM, 8 INVALID_ARG.

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare float @expf(float)
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; ================================================== lr_dot (inlined kernel)
define internal float @lr_dot(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
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

; ============================================================ shared fit impl
define internal i32 @lr_fit_impl(ptr readonly %X, ptr readonly %y, i64 %n, i64 %d,
                                 i64 %iters, float %lr, ptr %w, ptr %b_out,
                                 i1 %logistic) #0 {
entry:
  %bad1 = icmp eq i64 %n, 0
  %bad2 = icmp eq i64 %d, 0
  %bad = or i1 %bad1, %bad2
  br i1 %bad, label %einval, label %ok

ok:
  %dbytes = shl i64 %d, 2
  call void @llvm.memset.p0.i64(ptr %w, i8 0, i64 %dbytes, i1 false)
  %gradw = call ptr @malloc(i64 %dbytes)
  %isnull = icmp eq ptr %gradw, null
  br i1 %isnull, label %eoom, label %ready

ready:
  %nf = uitofp i64 %n to float
  %invn = fdiv fast float 1.0, %nf
  %lrn = fmul fast float %lr, %invn
  br label %ihead

ihead:
  %it = phi i64 [ 0, %ready ], [ %itn, %icont ]
  %b = phi float [ 0.0, %ready ], [ %bn, %icont ]
  %itgo = icmp ult i64 %it, %iters
  br i1 %itgo, label %izero, label %fin

izero:
  call void @llvm.memset.p0.i64(ptr %gradw, i8 0, i64 %dbytes, i1 false)
  br label %shead

shead:
  %si = phi i64 [ 0, %izero ], [ %sin, %supdate ]
  %gradb = phi float [ 0.0, %izero ], [ %gradbn, %supdate ]
  %sgo = icmp ult i64 %si, %n
  br i1 %sgo, label %sbody, label %uhead

sbody:
  %xoff = mul nuw i64 %si, %d
  %xi = getelementptr inbounds nuw float, ptr %X, i64 %xoff
  %z0 = call float @lr_dot(ptr %xi, ptr %w, i64 %d)
  %z = fadd float %z0, %b
  br i1 %logistic, label %sig, label %lin

sig:
  %nz = fneg float %z
  %e = call float @expf(float %nz)
  %den = fadd float 1.0, %e
  %psig = fdiv float 1.0, %den
  br label %predmerge

lin:
  br label %predmerge

predmerge:
  %pred = phi float [ %psig, %sig ], [ %z, %lin ]
  %yp = getelementptr inbounds nuw float, ptr %y, i64 %si
  %yi = load float, ptr %yp, align 4
  %r = fsub float %pred, %yi
  %gradbn = fadd fast float %gradb, %r
  br label %ghead

ghead:
  %gj = phi i64 [ 0, %predmerge ], [ %gjn, %gbody ]
  %ggo = icmp ult i64 %gj, %d
  br i1 %ggo, label %gbody, label %supdate

gbody:
  %xjp = getelementptr inbounds nuw float, ptr %xi, i64 %gj
  %xj = load float, ptr %xjp, align 4
  %gwp = getelementptr inbounds nuw float, ptr %gradw, i64 %gj
  %gw = load float, ptr %gwp, align 4
  %rx = fmul fast float %r, %xj
  %gwn = fadd fast float %gw, %rx
  store float %gwn, ptr %gwp, align 4
  %gjn = add nuw i64 %gj, 1
  br label %ghead

supdate:
  %sin = add nuw i64 %si, 1
  br label %shead

uhead:
  br label %whead

whead:
  %wj = phi i64 [ 0, %uhead ], [ %wjn, %wbody ]
  %wgo = icmp ult i64 %wj, %d
  br i1 %wgo, label %wbody, label %bupd

wbody:
  %ugwp = getelementptr inbounds nuw float, ptr %gradw, i64 %wj
  %ugw = load float, ptr %ugwp, align 4
  %uwp = getelementptr inbounds nuw float, ptr %w, i64 %wj
  %uwv = load float, ptr %uwp, align 4
  %wstep = fmul fast float %lrn, %ugw
  %wnew = fsub fast float %uwv, %wstep
  store float %wnew, ptr %uwp, align 4
  %wjn = add nuw i64 %wj, 1
  br label %whead

bupd:
  %bstep = fmul fast float %lrn, %gradb
  %bn = fsub fast float %b, %bstep
  br label %icont

icont:
  %itn = add nuw i64 %it, 1
  br label %ihead

fin:
  store float %b, ptr %b_out, align 4
  call void @free(ptr %gradw)
  ret i32 0

einval:
  ret i32 8

eoom:
  ret i32 2
}

; ============================================================ predict (shared)
define internal i32 @lr_predict_impl(ptr readonly %X, i64 %n, i64 %d,
                                     ptr readonly %w, float %b, ptr %out,
                                     i1 %logistic) #0 {
entry:
  %bad2 = icmp eq i64 %d, 0
  br i1 %bad2, label %einval, label %ok

ok:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %ok ], [ %in, %cont ]
  %xoff = mul nuw i64 %i, %d
  %xi = getelementptr inbounds nuw float, ptr %X, i64 %xoff
  %z0 = call float @lr_dot(ptr %xi, ptr %w, i64 %d)
  %z = fadd float %z0, %b
  br i1 %logistic, label %sig, label %store

sig:
  %nz = fneg float %z
  %e = call float @expf(float %nz)
  %den = fadd float 1.0, %e
  %psig = fdiv float 1.0, %den
  br label %store

store:
  %pred = phi float [ %psig, %sig ], [ %z, %loop ]
  %op = getelementptr inbounds nuw float, ptr %out, i64 %i
  store float %pred, ptr %op, align 4
  br label %cont

cont:
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret i32 0

einval:
  ret i32 8
}

; ================================================================== public API
define i32 @universe_ml_linreg_fit(ptr readonly %X, ptr readonly %y, i64 %n, i64 %d,
                                   i64 %iters, float %lr, ptr %w, ptr %b_out) #0 {
entry:
  %r = call i32 @lr_fit_impl(ptr %X, ptr %y, i64 %n, i64 %d, i64 %iters, float %lr, ptr %w, ptr %b_out, i1 false)
  ret i32 %r
}

define i32 @universe_ml_linreg_predict(ptr readonly %X, i64 %n, i64 %d, ptr readonly %w, float %b, ptr %out) #0 {
entry:
  %r = call i32 @lr_predict_impl(ptr %X, i64 %n, i64 %d, ptr %w, float %b, ptr %out, i1 false)
  ret i32 %r
}

define i32 @universe_ml_logreg_fit(ptr readonly %X, ptr readonly %y, i64 %n, i64 %d,
                                   i64 %iters, float %lr, ptr %w, ptr %b_out) #0 {
entry:
  %r = call i32 @lr_fit_impl(ptr %X, ptr %y, i64 %n, i64 %d, i64 %iters, float %lr, ptr %w, ptr %b_out, i1 true)
  ret i32 %r
}

define i32 @universe_ml_logreg_predict_proba(ptr readonly %X, i64 %n, i64 %d, ptr readonly %w, float %b, ptr %out) #0 {
entry:
  %r = call i32 @lr_predict_impl(ptr %X, i64 %n, i64 %d, ptr %w, float %b, ptr %out, i1 true)
  ret i32 %r
}

attributes #0 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
