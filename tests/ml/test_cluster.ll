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

; Tests for universe_ml_kmeans_*: a 3-cluster well-separated synthetic set
; (fixed seed) must converge so within-cluster labels are consistent, the
; three clusters land in distinct labels, inertia is small and decreases with
; more iterations, predict agrees with fit, and error codes fire.

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

declare i32 @universe_ml_kmeans_fit(ptr, i64, i64, i64, i64, float, ptr, ptr, ptr)
declare i32 @universe_ml_kmeans_predict(ptr, i64, i64, i64, ptr, ptr)

@centers = private unnamed_addr constant [6 x float]
  [float 0.0, float 0.0, float 10.0, float 10.0, float 20.0, float 0.0], align 4

@m.fit    = private unnamed_addr constant [12 x i8] c"fit ok rc=0\00", align 1
@m.within = private unnamed_addr constant [21 x i8] c"within-cluster label\00", align 1
@m.d01    = private unnamed_addr constant [16 x i8] c"labels 0!=1 dst\00", align 1
@m.d02    = private unnamed_addr constant [16 x i8] c"labels 0!=2 dst\00", align 1
@m.d12    = private unnamed_addr constant [16 x i8] c"labels 1!=2 dst\00", align 1
@m.inpos  = private unnamed_addr constant [14 x i8] c"inertia > 0.0\00", align 1
@m.insmall= private unnamed_addr constant [15 x i8] c"inertia < 30.0\00", align 1
@m.indec  = private unnamed_addr constant [19 x i8] c"inertia decreases\0A\00", align 1
@m.pred   = private unnamed_addr constant [17 x i8] c"predict==fit lbl\00", align 1
@m.ek0    = private unnamed_addr constant [13 x i8] c"k=0 -> INVAL\00", align 1
@m.ed0    = private unnamed_addr constant [13 x i8] c"d=0 -> INVAL\00", align 1
@m.ekn    = private unnamed_addr constant [13 x i8] c"k>n -> INVAL\00", align 1
@cluster.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.cluster = private unnamed_addr constant [29 x i8] c"kmeans n=10000 d=8 k=8 x10it\00"

; n=90, d=2, k=3, 30 points per cluster.

; fill X[90*2] with cluster i/30 center + uniform noise in [-0.5, 0.5)
define internal void @gen(ptr %X, ptr %st) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %ci = udiv i64 %i, 30
  %cx.off = mul nuw i64 %ci, 2
  %cx.p = getelementptr inbounds nuw [6 x float], ptr @centers, i64 0, i64 %cx.off
  %cx = load float, ptr %cx.p, align 4
  %cy.off = add nuw i64 %cx.off, 1
  %cy.p = getelementptr inbounds nuw [6 x float], ptr @centers, i64 0, i64 %cy.off
  %cy = load float, ptr %cy.p, align 4
  ; noise x
  %rx = call i64 @ut_rand(ptr %st)
  %rxl = and i64 %rx, 65535
  %rxf = uitofp i64 %rxl to float
  %rxs = fmul float %rxf, 0x3EF0000000000000
  %nx = fsub float %rxs, 0.5
  ; noise y
  %ry = call i64 @ut_rand(ptr %st)
  %ryl = and i64 %ry, 65535
  %ryf = uitofp i64 %ryl to float
  %rys = fmul float %ryf, 0x3EF0000000000000
  %ny = fsub float %rys, 0.5
  %px = fadd float %cx, %nx
  %py = fadd float %cy, %ny
  %row = mul nuw i64 %i, 2
  %xp = getelementptr inbounds nuw float, ptr %X, i64 %row
  store float %px, ptr %xp, align 4
  %row1 = add nuw i64 %row, 1
  %yp = getelementptr inbounds nuw float, ptr %X, i64 %row1
  store float %py, ptr %yp, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, 90
  br i1 %more, label %loop, label %done

done:
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 424242, ptr %st, align 8
  %X = call ptr @malloc(i64 720)       ; 90*2*4
  %C = call ptr @malloc(i64 24)        ; 3*2*4
  %labels = call ptr @malloc(i64 360)  ; 90*4
  %labels2 = call ptr @malloc(i64 360)
  %inertia = alloca float, align 4
  %inertia1 = alloca float, align 4
  call void @gen(ptr %X, ptr %st)

  ; --- inertia after only 1 iteration (for the decrease check) ---
  %rc1 = call i32 @universe_ml_kmeans_fit(ptr %X, i64 90, i64 2, i64 3, i64 1, float 0.0, ptr %C, ptr %labels2, ptr %inertia1)
  %in1 = load float, ptr %inertia1, align 4

  ; --- full fit ---
  %rc = call i32 @universe_ml_kmeans_fit(ptr %X, i64 90, i64 2, i64 3, i64 100, float 0x3F20000000000000, ptr %C, ptr %labels, ptr %inertia)
  %rcok = icmp eq i32 %rc, 0
  call void @ut_check(i1 %rcok, ptr @m.fit)
  %inv = load float, ptr %inertia, align 4

  ; within-cluster label consistency: labels[i] == labels[(i/30)*30]
  br label %wl

wl:
  %wi = phi i64 [ 0, %entry ], [ %win, %wl ]
  %wviol = phi i64 [ 0, %entry ], [ %wvioln, %wl ]
  %wci = udiv i64 %wi, 30
  %wref.idx = mul nuw i64 %wci, 30
  %wref.p = getelementptr inbounds nuw i32, ptr %labels, i64 %wref.idx
  %wref = load i32, ptr %wref.p, align 4
  %wp = getelementptr inbounds nuw i32, ptr %labels, i64 %wi
  %wl.v = load i32, ptr %wp, align 4
  %wne = icmp ne i32 %wl.v, %wref
  %winc = zext i1 %wne to i64
  %wvioln = add nuw i64 %wviol, %winc
  %win = add nuw i64 %wi, 1
  %wmore = icmp ult i64 %win, 90
  br i1 %wmore, label %wl, label %wdone

wdone:
  call void @ut_check_eq(i64 %wvioln, i64 0, ptr @m.within)

  ; distinct labels across clusters
  %l0 = load i32, ptr %labels, align 4
  %l30.p = getelementptr inbounds nuw i32, ptr %labels, i64 30
  %l30 = load i32, ptr %l30.p, align 4
  %l60.p = getelementptr inbounds nuw i32, ptr %labels, i64 60
  %l60 = load i32, ptr %l60.p, align 4
  %d01 = icmp ne i32 %l0, %l30
  call void @ut_check(i1 %d01, ptr @m.d01)
  %d02 = icmp ne i32 %l0, %l60
  call void @ut_check(i1 %d02, ptr @m.d02)
  %d12 = icmp ne i32 %l30, %l60
  call void @ut_check(i1 %d12, ptr @m.d12)

  ; inertia sanity + decrease
  %ipos = fcmp ogt float %inv, 0.0
  call void @ut_check(i1 %ipos, ptr @m.inpos)
  %ismall = fcmp olt float %inv, 30.0
  call void @ut_check(i1 %ismall, ptr @m.insmall)
  %idec = fcmp ole float %inv, %in1
  call void @ut_check(i1 %idec, ptr @m.indec)

  ; predict on same data must reproduce the fit labels
  %prc = call i32 @universe_ml_kmeans_predict(ptr %X, i64 90, i64 2, i64 3, ptr %C, ptr %labels2)
  br label %pl

pl:
  %pi = phi i64 [ 0, %wdone ], [ %pin, %pl ]
  %pviol = phi i64 [ 0, %wdone ], [ %pvioln, %pl ]
  %pa = getelementptr inbounds nuw i32, ptr %labels, i64 %pi
  %pav = load i32, ptr %pa, align 4
  %pb = getelementptr inbounds nuw i32, ptr %labels2, i64 %pi
  %pbv = load i32, ptr %pb, align 4
  %pne = icmp ne i32 %pav, %pbv
  %pinc = zext i1 %pne to i64
  %pvioln = add nuw i64 %pviol, %pinc
  %pin = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pin, 90
  br i1 %pmore, label %pl, label %pdone

pdone:
  call void @ut_check_eq(i64 %pvioln, i64 0, ptr @m.pred)

  ; --- error codes ---
  %ek0 = call i32 @universe_ml_kmeans_fit(ptr %X, i64 90, i64 2, i64 0, i64 10, float 0.0, ptr %C, ptr %labels, ptr %inertia)
  %ek0i = sext i32 %ek0 to i64
  call void @ut_check_eq(i64 %ek0i, i64 8, ptr @m.ek0)
  %ed0 = call i32 @universe_ml_kmeans_fit(ptr %X, i64 90, i64 0, i64 3, i64 10, float 0.0, ptr %C, ptr %labels, ptr %inertia)
  %ed0i = sext i32 %ed0 to i64
  call void @ut_check_eq(i64 %ed0i, i64 8, ptr @m.ed0)
  %ekn = call i32 @universe_ml_kmeans_fit(ptr %X, i64 3, i64 2, i64 5, i64 10, float 0.0, ptr %C, ptr %labels, ptr %inertia)
  %ekni = sext i32 %ekn to i64
  call void @ut_check_eq(i64 %ekni, i64 8, ptr @m.ekn)

  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  call void @run_bench()
  br label %fin

fin:
  call void @free(ptr %X)
  call void @free(ptr %C)
  call void @free(ptr %labels)
  call void @free(ptr %labels2)
  %rcs = call i32 @ut_summary()
  ret i32 %rcs
}

; bench: n=10000, d=8, k=8, 10 Lloyd iterations; report ms.
define internal void @run_bench() {
entry:
  %n = add i64 0, 10000
  %d = add i64 0, 8
  %k = add i64 0, 8
  %nd = mul nuw i64 %n, %d
  %xb = shl i64 %nd, 2
  %X = call ptr @malloc(i64 %xb)
  %kd = mul nuw i64 %k, %d
  %cb = shl i64 %kd, 2
  %C = call ptr @malloc(i64 %cb)
  %lb = shl i64 %n, 2
  %labels = call ptr @malloc(i64 %lb)
  %inertia = alloca float, align 4
  %st = alloca i64, align 8
  store i64 987654321, ptr %st, align 8
  br label %fillloop
fillloop:
  %i = phi i64 [ 0, %entry ], [ %in, %fillloop ]
  %r = call i64 @ut_rand(ptr %st)
  %rl = and i64 %r, 65535
  %rf = uitofp i64 %rl to float
  %p = getelementptr inbounds nuw float, ptr %X, i64 %i
  store float %rf, ptr %p, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nd
  br i1 %more, label %fillloop, label %timeit
timeit:
  br label %b.rep.head
b.rep.head:
  %b.rep = phi i64 [ 0, %timeit ], [ %b.rep.n, %b.rep.cont ]
  %b.t0 = call double @ut_now_sec()
  %b.rc = call i32 @universe_ml_kmeans_fit(ptr %X, i64 %n, i64 %d, i64 %k, i64 10, float 0.0, ptr %C, ptr %labels, ptr %inertia)
  %b.t1 = call double @ut_now_sec()
  %b.dt = fsub double %b.t1, %b.t0
  %b.warm = icmp eq i64 %b.rep, 0
  br i1 %b.warm, label %b.rep.cont, label %b.rep.store
b.rep.store:
  %b.idx = sub i64 %b.rep, 1
  %b.sp = getelementptr inbounds [16 x double], ptr @cluster.samp, i64 0, i64 %b.idx
  store double %b.dt, ptr %b.sp, align 8
  br label %b.rep.cont
b.rep.cont:
  %b.rep.n = add i64 %b.rep, 1
  %b.more = icmp ult i64 %b.rep.n, 17
  br i1 %b.more, label %b.rep.head, label %b.rep.done
b.rep.done:
  ; ops_per_rep = n * 10 iterations = 100000 point-iterations
  call void @ut_report_dist(ptr @cluster.samp, i64 16, i64 100000, ptr @lbl.cluster)
  call void @free(ptr %X)
  call void @free(ptr %C)
  call void @free(ptr %labels)
  ret void
}
