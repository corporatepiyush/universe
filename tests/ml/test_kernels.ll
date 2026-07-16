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

; Tests for universe_ml_* SIMD kernels: vector == scalar (within f32
; tolerance) over fixed-seed random arrays at many lengths, plus known-answer
; checks and a --bench GFLOP/s comparison.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
declare float @llvm.fabs.f32(float)

; kernel API under test
declare float @universe_ml_dot(ptr, ptr, i64)
declare float @universe_ml_dot_scalar(ptr, ptr, i64)
declare void @universe_ml_axpy(ptr, float, ptr, i64)
declare void @universe_ml_axpy_scalar(ptr, float, ptr, i64)
declare void @universe_ml_scale(ptr, float, i64)
declare void @universe_ml_scale_scalar(ptr, float, i64)
declare float @universe_ml_sum(ptr, i64)
declare float @universe_ml_sum_scalar(ptr, i64)
declare float @universe_ml_mean(ptr, i64)
declare float @universe_ml_mean_scalar(ptr, i64)
declare float @universe_ml_l2_norm(ptr, i64)
declare float @universe_ml_l2_norm_scalar(ptr, i64)
declare float @universe_ml_l2_dist2(ptr, ptr, i64)
declare float @universe_ml_l2_dist2_scalar(ptr, ptr, i64)
declare float @universe_ml_l1_dist(ptr, ptr, i64)
declare float @universe_ml_l1_dist_scalar(ptr, ptr, i64)
declare void @universe_ml_gemv(ptr, ptr, ptr, i64, i64)
declare void @universe_ml_gemv_scalar(ptr, ptr, ptr, i64, i64)
declare i64 @universe_ml_argmin(ptr, i64)
declare i64 @universe_ml_argmin_scalar(ptr, i64)
declare i64 @universe_ml_argmax(ptr, i64)
declare i64 @universe_ml_argmax_scalar(ptr, i64)

@msg.dot   = private unnamed_addr constant [16 x i8] c"dot vec==scalar\00", align 1
@msg.sum   = private unnamed_addr constant [16 x i8] c"sum vec==scalar\00", align 1
@msg.mean  = private unnamed_addr constant [17 x i8] c"mean vec==scalar\00", align 1
@msg.norm  = private unnamed_addr constant [20 x i8] c"l2norm vec==scalar\0A\00", align 1
@msg.d2    = private unnamed_addr constant [19 x i8] c"dist2 vec==scalar\0A\00", align 1
@msg.l1    = private unnamed_addr constant [16 x i8] c"l1 vec==scalar\0A\00", align 1
@msg.axpy  = private unnamed_addr constant [18 x i8] c"axpy vec==scalar\0A\00", align 1
@msg.scale = private unnamed_addr constant [19 x i8] c"scale vec==scalar\0A\00", align 1
@msg.gemv  = private unnamed_addr constant [18 x i8] c"gemv vec==scalar\0A\00", align 1
@msg.amin  = private unnamed_addr constant [17 x i8] c"argmin vec==scal\00", align 1
@msg.amax  = private unnamed_addr constant [17 x i8] c"argmax vec==scal\00", align 1
@msg.kdot  = private unnamed_addr constant [12 x i8] c"known dot10\00", align 1
@msg.ksum  = private unnamed_addr constant [12 x i8] c"known sum10\00", align 1
@msg.kmean = private unnamed_addr constant [14 x i8] c"known mean2.5\00", align 1
@msg.kd2   = private unnamed_addr constant [13 x i8] c"known dist14\00", align 1
@msg.kl1   = private unnamed_addr constant [11 x i8] c"known l1 6\00", align 1
@msg.knorm = private unnamed_addr constant [15 x i8] c"known normsq30\00", align 1
@msg.kamax = private unnamed_addr constant [12 x i8] c"known amax3\00", align 1
@msg.kamin = private unnamed_addr constant [12 x i8] c"known amin0\00", align 1
@msg.a0    = private unnamed_addr constant [16 x i8] c"argmin empty -1\00", align 1
@msg.b0    = private unnamed_addr constant [16 x i8] c"argmax empty -1\00", align 1

@dv.samp = internal global [16 x double] zeroinitializer, align 8
@ds.samp = internal global [16 x double] zeroinitializer, align 8
@av.samp = internal global [16 x double] zeroinitializer, align 8
@as.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.dv = private unnamed_addr constant [12 x i8] c"dot vec 64K\00"
@lbl.ds = private unnamed_addr constant [15 x i8] c"dot scalar 64K\00"
@lbl.av = private unnamed_addr constant [13 x i8] c"axpy vec 64K\00"
@lbl.as = private unnamed_addr constant [16 x i8] c"axpy scalar 64K\00"

; length list (fixed): 0,1,2,3,4,5,7,8,15,16,17,31,63,64,100,255,256,1000
@lengths = private unnamed_addr constant [18 x i64]
  [i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 7, i64 8, i64 15,
   i64 16, i64 17, i64 31, i64 63, i64 64, i64 100, i64 255, i64 256, i64 1000], align 8

; ============================================================ fill random arr
; value in [-1, 1): (rand & 0xFFFF)/32768 - 1
define internal void @fill(ptr %a, i64 %n, ptr %st) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %low = and i64 %r, 65535
  %rf = uitofp i64 %low to float
  %sc = fmul float %rf, 0x3F00000000000000
  %v = fsub float %sc, 1.0
  %p = getelementptr inbounds nuw float, ptr %a, i64 %i
  store float %v, ptr %p, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ============================================================ close (v ~= s)
; bound = (|s| + 1) * 2^-12  (relative ~2.4e-4 with an absolute floor)
define internal void @ck(float %v, float %s, ptr %msg) {
entry:
  %d = fsub float %v, %s
  %ad = call float @llvm.fabs.f32(float %d)
  %as = call float @llvm.fabs.f32(float %s)
  %b0 = fadd float %as, 1.0
  %b = fmul float %b0, 0x3F30000000000000
  %ok = fcmp ole float %ad, %b
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; ===================================================== count array mismatches
define internal i64 @arrdiff(ptr %p, ptr %q, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %violn, %loop ]
  %pp = getelementptr inbounds nuw float, ptr %p, i64 %i
  %pv = load float, ptr %pp, align 4
  %qp = getelementptr inbounds nuw float, ptr %q, i64 %i
  %qv = load float, ptr %qp, align 4
  %d = fsub float %pv, %qv
  %ad = call float @llvm.fabs.f32(float %d)
  %aq = call float @llvm.fabs.f32(float %qv)
  %b0 = fadd float %aq, 1.0
  %b = fmul float %b0, 0x3F30000000000000
  %bad = fcmp ogt float %ad, %b
  %inc = zext i1 %bad to i64
  %violn = add nuw i64 %viol, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi i64 [ 0, %entry ], [ %violn, %loop ]
  ret i64 %r
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8
  ; scratch buffers: up to 1000 elems each (a,b,y1,y2) + gemv A (16*64=1024)
  %a = call ptr @malloc(i64 4096)
  %b = call ptr @malloc(i64 4096)
  %y1 = call ptr @malloc(i64 4096)
  %y2 = call ptr @malloc(i64 4096)
  %A = call ptr @malloc(i64 8192)
  br label %lenloop

lenloop:
  %li = phi i64 [ 0, %entry ], [ %lin, %lentail ]
  %lp = getelementptr inbounds nuw [18 x i64], ptr @lengths, i64 0, i64 %li
  %n = load i64, ptr %lp, align 8
  call void @fill(ptr %a, i64 %n, ptr %st)
  call void @fill(ptr %b, i64 %n, ptr %st)

  ; --- reductions: vector vs scalar ---
  %dv = call float @universe_ml_dot(ptr %a, ptr %b, i64 %n)
  %ds = call float @universe_ml_dot_scalar(ptr %a, ptr %b, i64 %n)
  call void @ck(float %dv, float %ds, ptr @msg.dot)

  %sv = call float @universe_ml_sum(ptr %a, i64 %n)
  %ss = call float @universe_ml_sum_scalar(ptr %a, i64 %n)
  call void @ck(float %sv, float %ss, ptr @msg.sum)

  %mv = call float @universe_ml_mean(ptr %a, i64 %n)
  %ms = call float @universe_ml_mean_scalar(ptr %a, i64 %n)
  call void @ck(float %mv, float %ms, ptr @msg.mean)

  %nv = call float @universe_ml_l2_norm(ptr %a, i64 %n)
  %ns = call float @universe_ml_l2_norm_scalar(ptr %a, i64 %n)
  call void @ck(float %nv, float %ns, ptr @msg.norm)

  %d2v = call float @universe_ml_l2_dist2(ptr %a, ptr %b, i64 %n)
  %d2s = call float @universe_ml_l2_dist2_scalar(ptr %a, ptr %b, i64 %n)
  call void @ck(float %d2v, float %d2s, ptr @msg.d2)

  %l1v = call float @universe_ml_l1_dist(ptr %a, ptr %b, i64 %n)
  %l1s = call float @universe_ml_l1_dist_scalar(ptr %a, ptr %b, i64 %n)
  call void @ck(float %l1v, float %l1s, ptr @msg.l1)

  ; --- argmin / argmax: exact vec == scalar ---
  %aiv = call i64 @universe_ml_argmin(ptr %a, i64 %n)
  %ais = call i64 @universe_ml_argmin_scalar(ptr %a, i64 %n)
  call void @ut_check_eq(i64 %aiv, i64 %ais, ptr @msg.amin)
  %axv = call i64 @universe_ml_argmax(ptr %a, i64 %n)
  %axs = call i64 @universe_ml_argmax_scalar(ptr %a, i64 %n)
  call void @ut_check_eq(i64 %axv, i64 %axs, ptr @msg.amax)

  ; --- axpy: y1 = y2 = a; run vector on y1, scalar on y2, compare ---
  %nbytes = mul nuw i64 %n, 4
  call void @cpy(ptr %y1, ptr %a, i64 %n)
  call void @cpy(ptr %y2, ptr %a, i64 %n)
  call void @universe_ml_axpy(ptr %y1, float 2.5, ptr %b, i64 %n)
  call void @universe_ml_axpy_scalar(ptr %y2, float 2.5, ptr %b, i64 %n)
  %ad = call i64 @arrdiff(ptr %y1, ptr %y2, i64 %n)
  call void @ut_check_eq(i64 %ad, i64 0, ptr @msg.axpy)

  ; --- scale: same pattern ---
  call void @cpy(ptr %y1, ptr %a, i64 %n)
  call void @cpy(ptr %y2, ptr %a, i64 %n)
  call void @universe_ml_scale(ptr %y1, float 1.5, i64 %n)
  call void @universe_ml_scale_scalar(ptr %y2, float 1.5, i64 %n)
  %sd = call i64 @arrdiff(ptr %y1, ptr %y2, i64 %n)
  call void @ut_check_eq(i64 %sd, i64 0, ptr @msg.scale)

  br label %lentail

lentail:
  %lin = add nuw i64 %li, 1
  %lmore = icmp ult i64 %lin, 18
  br i1 %lmore, label %lenloop, label %gemv

; ---- gemv over a few (m,n) shapes ----
gemv:
  ; shape 1: m=8, n=8
  call void @fill(ptr %A, i64 64, ptr %st)
  call void @fill(ptr %b, i64 8, ptr %st)
  call void @universe_ml_gemv(ptr %y1, ptr %A, ptr %b, i64 8, i64 8)
  call void @universe_ml_gemv_scalar(ptr %y2, ptr %A, ptr %b, i64 8, i64 8)
  %g1 = call i64 @arrdiff(ptr %y1, ptr %y2, i64 8)
  call void @ut_check_eq(i64 %g1, i64 0, ptr @msg.gemv)
  ; shape 2: m=5, n=17
  call void @fill(ptr %A, i64 85, ptr %st)
  call void @fill(ptr %b, i64 17, ptr %st)
  call void @universe_ml_gemv(ptr %y1, ptr %A, ptr %b, i64 5, i64 17)
  call void @universe_ml_gemv_scalar(ptr %y2, ptr %A, ptr %b, i64 5, i64 17)
  %g2 = call i64 @arrdiff(ptr %y1, ptr %y2, i64 5)
  call void @ut_check_eq(i64 %g2, i64 0, ptr @msg.gemv)
  ; shape 3: m=16, n=64
  call void @fill(ptr %A, i64 1024, ptr %st)
  call void @fill(ptr %b, i64 64, ptr %st)
  call void @universe_ml_gemv(ptr %y1, ptr %A, ptr %b, i64 16, i64 64)
  call void @universe_ml_gemv_scalar(ptr %y2, ptr %A, ptr %b, i64 16, i64 64)
  %g3 = call i64 @arrdiff(ptr %y1, ptr %y2, i64 16)
  call void @ut_check_eq(i64 %g3, i64 0, ptr @msg.gemv)

  ; ---- known-answer: a=[1,2,3,4], b=[1,1,1,1] ----
  store float 1.0, ptr %a, align 4
  %a1 = getelementptr inbounds nuw float, ptr %a, i64 1
  store float 2.0, ptr %a1, align 4
  %a2 = getelementptr inbounds nuw float, ptr %a, i64 2
  store float 3.0, ptr %a2, align 4
  %a3 = getelementptr inbounds nuw float, ptr %a, i64 3
  store float 4.0, ptr %a3, align 4
  store float 1.0, ptr %b, align 4
  %bb1 = getelementptr inbounds nuw float, ptr %b, i64 1
  store float 1.0, ptr %bb1, align 4
  %bb2 = getelementptr inbounds nuw float, ptr %b, i64 2
  store float 1.0, ptr %bb2, align 4
  %bb3 = getelementptr inbounds nuw float, ptr %b, i64 3
  store float 1.0, ptr %bb3, align 4

  %kdot = call float @universe_ml_dot(ptr %a, ptr %b, i64 4)
  call void @ck(float %kdot, float 10.0, ptr @msg.kdot)
  %ksum = call float @universe_ml_sum(ptr %a, i64 4)
  call void @ck(float %ksum, float 10.0, ptr @msg.ksum)
  %kmean = call float @universe_ml_mean(ptr %a, i64 4)
  call void @ck(float %kmean, float 2.5, ptr @msg.kmean)
  %kd2 = call float @universe_ml_l2_dist2(ptr %a, ptr %b, i64 4)
  call void @ck(float %kd2, float 14.0, ptr @msg.kd2)
  %kl1 = call float @universe_ml_l1_dist(ptr %a, ptr %b, i64 4)
  call void @ck(float %kl1, float 6.0, ptr @msg.kl1)
  %knsq = call float @universe_ml_dot(ptr %a, ptr %a, i64 4)
  call void @ck(float %knsq, float 30.0, ptr @msg.knorm)
  %kamax = call i64 @universe_ml_argmax(ptr %a, i64 4)
  call void @ut_check_eq(i64 %kamax, i64 3, ptr @msg.kamax)
  %kamin = call i64 @universe_ml_argmin(ptr %a, i64 4)
  call void @ut_check_eq(i64 %kamin, i64 0, ptr @msg.kamin)

  ; empty-array argmin/argmax -> -1
  %e0 = call i64 @universe_ml_argmin(ptr %a, i64 0)
  call void @ut_check_eq(i64 %e0, i64 -1, ptr @msg.a0)
  %e1 = call i64 @universe_ml_argmax(ptr %a, i64 0)
  call void @ut_check_eq(i64 %e1, i64 -1, ptr @msg.b0)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  call void @run_bench(ptr %A)
  br label %fin

fin:
  call void @free(ptr %a)
  call void @free(ptr %b)
  call void @free(ptr %y1)
  call void @free(ptr %y2)
  call void @free(ptr %A)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; float copy helper
define internal void @cpy(ptr %dst, ptr %src, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %sp = getelementptr inbounds nuw float, ptr %src, i64 %i
  %v = load float, ptr %sp, align 4
  %dp = getelementptr inbounds nuw float, ptr %dst, i64 %i
  store float %v, ptr %dp, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ============================================================ bench (64K dot)
define internal void @run_bench(ptr %scratch) {
entry:
  %N = add i64 0, 65536
  %bytes = mul nuw i64 %N, 4
  %a = call ptr @malloc(i64 %bytes)
  %b = call ptr @malloc(i64 %bytes)
  %st = alloca i64, align 8
  store i64 12345678, ptr %st, align 8
  call void @fill(ptr %a, i64 %N, ptr %st)
  call void @fill(ptr %b, i64 %N, ptr %st)

  ; iterations
  %iters = add i64 0, 2000

  ; ops_per_rep = 2 flops/elem * N * iters = 2*65536*2000 = 262144000
  ; --- dot vector (distribution) ---
  br label %dv.rep.head
dv.rep.head:
  %dv.rep = phi i64 [ 0, %entry ], [ %dv.rep.n, %dv.rep.cont ]
  %dv.t0 = call double @ut_now_sec()
  br label %dv.loop
dv.loop:
  %dvi = phi i64 [ 0, %dv.rep.head ], [ %dvin, %dv.loop ]
  %dvacc = phi float [ 0.0, %dv.rep.head ], [ %dvacn, %dv.loop ]
  %dvr = call float @universe_ml_dot(ptr %a, ptr %b, i64 %N)
  %dvacn = fadd float %dvacc, %dvr
  %dvin = add nuw i64 %dvi, 1
  %dvm = icmp ult i64 %dvin, %iters
  br i1 %dvm, label %dv.loop, label %dv.done
dv.done:
  %dv.t1 = call double @ut_now_sec()
  store volatile float %dvacn, ptr %scratch, align 4
  %dv.dt = fsub double %dv.t1, %dv.t0
  %dv.warm = icmp eq i64 %dv.rep, 0
  br i1 %dv.warm, label %dv.rep.cont, label %dv.rep.store
dv.rep.store:
  %dv.idx = sub i64 %dv.rep, 1
  %dv.sp = getelementptr inbounds [16 x double], ptr @dv.samp, i64 0, i64 %dv.idx
  store double %dv.dt, ptr %dv.sp, align 8
  br label %dv.rep.cont
dv.rep.cont:
  %dv.rep.n = add i64 %dv.rep, 1
  %dv.more = icmp ult i64 %dv.rep.n, 17
  br i1 %dv.more, label %dv.rep.head, label %dv.rep.end
dv.rep.end:
  call void @ut_report_dist(ptr @dv.samp, i64 16, i64 262144000, ptr @lbl.dv)

  ; --- dot scalar (distribution) ---
  br label %ds.rep.head
ds.rep.head:
  %ds.rep = phi i64 [ 0, %dv.rep.end ], [ %ds.rep.n, %ds.rep.cont ]
  %ds.t0 = call double @ut_now_sec()
  br label %ds.loop
ds.loop:
  %dsi = phi i64 [ 0, %ds.rep.head ], [ %dsin, %ds.loop ]
  %dsacc = phi float [ 0.0, %ds.rep.head ], [ %dsacn, %ds.loop ]
  %dsr = call float @universe_ml_dot_scalar(ptr %a, ptr %b, i64 %N)
  %dsacn = fadd float %dsacc, %dsr
  %dsin = add nuw i64 %dsi, 1
  %dsm = icmp ult i64 %dsin, %iters
  br i1 %dsm, label %ds.loop, label %ds.done
ds.done:
  %ds.t1 = call double @ut_now_sec()
  store volatile float %dsacn, ptr %scratch, align 4
  %ds.dt = fsub double %ds.t1, %ds.t0
  %ds.warm = icmp eq i64 %ds.rep, 0
  br i1 %ds.warm, label %ds.rep.cont, label %ds.rep.store
ds.rep.store:
  %ds.idx = sub i64 %ds.rep, 1
  %ds.sp = getelementptr inbounds [16 x double], ptr @ds.samp, i64 0, i64 %ds.idx
  store double %ds.dt, ptr %ds.sp, align 8
  br label %ds.rep.cont
ds.rep.cont:
  %ds.rep.n = add i64 %ds.rep, 1
  %ds.more = icmp ult i64 %ds.rep.n, 17
  br i1 %ds.more, label %ds.rep.head, label %ds.rep.end
ds.rep.end:
  call void @ut_report_dist(ptr @ds.samp, i64 16, i64 262144000, ptr @lbl.ds)

  ; --- axpy vector (distribution; 2 flops/elem) ---
  br label %av.rep.head
av.rep.head:
  %av.rep = phi i64 [ 0, %ds.rep.end ], [ %av.rep.n, %av.rep.cont ]
  %av.t0 = call double @ut_now_sec()
  br label %av.loop
av.loop:
  %avi = phi i64 [ 0, %av.rep.head ], [ %avin, %av.loop ]
  call void @universe_ml_axpy(ptr %a, float 1.0, ptr %b, i64 %N)
  %avin = add nuw i64 %avi, 1
  %avm = icmp ult i64 %avin, %iters
  br i1 %avm, label %av.loop, label %av.done
av.done:
  %av.t1 = call double @ut_now_sec()
  store volatile float 0.0, ptr %a, align 4
  %av.dt = fsub double %av.t1, %av.t0
  %av.warm = icmp eq i64 %av.rep, 0
  br i1 %av.warm, label %av.rep.cont, label %av.rep.store
av.rep.store:
  %av.idx = sub i64 %av.rep, 1
  %av.sp = getelementptr inbounds [16 x double], ptr @av.samp, i64 0, i64 %av.idx
  store double %av.dt, ptr %av.sp, align 8
  br label %av.rep.cont
av.rep.cont:
  %av.rep.n = add i64 %av.rep, 1
  %av.more = icmp ult i64 %av.rep.n, 17
  br i1 %av.more, label %av.rep.head, label %av.rep.end
av.rep.end:
  call void @ut_report_dist(ptr @av.samp, i64 16, i64 262144000, ptr @lbl.av)

  ; --- axpy scalar (distribution; 2 flops/elem) ---
  br label %as.rep.head
as.rep.head:
  %as.rep = phi i64 [ 0, %av.rep.end ], [ %as.rep.n, %as.rep.cont ]
  %as.t0 = call double @ut_now_sec()
  br label %as.loop
as.loop:
  %asi = phi i64 [ 0, %as.rep.head ], [ %asin, %as.loop ]
  call void @universe_ml_axpy_scalar(ptr %a, float 1.0, ptr %b, i64 %N)
  %asin = add nuw i64 %asi, 1
  %asm = icmp ult i64 %asin, %iters
  br i1 %asm, label %as.loop, label %as.done
as.done:
  %as.t1 = call double @ut_now_sec()
  store volatile float 0.0, ptr %a, align 4
  %as.dt = fsub double %as.t1, %as.t0
  %as.warm = icmp eq i64 %as.rep, 0
  br i1 %as.warm, label %as.rep.cont, label %as.rep.store
as.rep.store:
  %as.idx = sub i64 %as.rep, 1
  %as.sp = getelementptr inbounds [16 x double], ptr @as.samp, i64 0, i64 %as.idx
  store double %as.dt, ptr %as.sp, align 8
  br label %as.rep.cont
as.rep.cont:
  %as.rep.n = add i64 %as.rep, 1
  %as.more = icmp ult i64 %as.rep.n, 17
  br i1 %as.more, label %as.rep.head, label %as.rep.end
as.rep.end:
  call void @ut_report_dist(ptr @as.samp, i64 16, i64 262144000, ptr @lbl.as)

  call void @free(ptr %a)
  call void @free(ptr %b)
  ret void
}
