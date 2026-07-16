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

; Tests for universe_observ_* (async logging, striped metrics, HDR histogram).
;   * logging: level gate suppression; 4-thread MPSC push, exactly-once/no-loss
;     conservation (direct pop tally); drain formats + dedups into a pipe.
;   * metrics: 4 threads x 1M inc => sum == 4M exactly; gauge set/get.
;   * histogram: known distribution -> count/min/max/mean + p50/p95/p99 within
;     precision; merge == recording the union.
;   * --bench: log push throughput, hist record throughput, per-rep latency
;     percentiles reported via the observ histogram itself.

declare ptr    @universe_observ_logger_create(i64, i32)
declare void   @universe_observ_logger_destroy(ptr)
declare void   @universe_observ_log_set_level(ptr, i32)
declare i1     @universe_observ_log_enabled(ptr, i32)
declare i32    @universe_observ_log(ptr, i32, i32, ptr, i64, i64, i64, i64, i64)
declare i64    @universe_observ_log_drain(ptr, ptr)
declare i64    @universe_observ_log_pending(ptr)
declare i64    @universe_observ_log_dropped(ptr)

declare ptr    @universe_observ_metrics_create(i64, i64, i64)
declare void   @universe_observ_metrics_destroy(ptr)
declare void   @universe_observ_counter_inc(ptr, i64, i64, i64)
declare i64    @universe_observ_counter_sum(ptr, i64)
declare void   @universe_observ_gauge_set(ptr, i64, i64)
declare i64    @universe_observ_gauge_get(ptr, i64)

declare ptr    @universe_observ_hist_create(i64, i64, i64)
declare void   @universe_observ_hist_destroy(ptr)
declare void   @universe_observ_hist_record(ptr, i64)
declare i64    @universe_observ_hist_percentile(ptr, double)
declare i64    @universe_observ_hist_count(ptr)
declare i64    @universe_observ_hist_min(ptr)
declare i64    @universe_observ_hist_max(ptr)
declare double @universe_observ_hist_mean(ptr)
declare i32    @universe_observ_hist_merge(ptr, ptr)

declare i32    @universe_conc_mpsc_ring_pop(ptr, ptr)
declare ptr    @universe_io_writer_create(i32, i64)
declare void   @universe_io_writer_destroy(ptr)

declare i32    @pthread_create(ptr, ptr, ptr, ptr)
declare i32    @pthread_join(i64, ptr)
declare i32    @pipe(ptr)
declare i64    @read(i32, ptr, i64)
declare i32    @close(i32)
declare i32    @printf(ptr, ...)
declare void   @ut_check(i1, ptr)
declare void   @ut_check_eq(i64, i64, ptr)
declare i64    @ut_rand(ptr)
declare i32    @ut_summary()
declare i1     @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void   @ut_report_dist(ptr, i64, i64, ptr)

@g.lg      = internal global ptr null, align 8
@g.n       = internal global i64 0, align 8
@g.tids    = internal global [8 x i64] zeroinitializer, align 8
@g.metrics = internal global ptr null, align 8
@g.mn      = internal global i64 0, align 8

@msg.work  = private unnamed_addr constant [12 x i8] c"worker.tick\00"
@msg.a     = private unnamed_addr constant [6 x i8] c"alpha\00"
@msg.b     = private unnamed_addr constant [5 x i8] c"beta\00"
@msg.c     = private unnamed_addr constant [6 x i8] c"gamma\00"

@m.suppress = private unnamed_addr constant [26 x i8] c"log gate suppresses below\00"
@m.pending1 = private unnamed_addr constant [24 x i8] c"log enabled pushes rec\0A\00"
@m.consv    = private unnamed_addr constant [30 x i8] c"log 4-thread exactly-once ok\0A\00"
@m.tally    = private unnamed_addr constant [24 x i8] c"log per-thread tally ok\00"
@m.drainn   = private unnamed_addr constant [22 x i8] c"drain count == pushed\00"
@m.lines    = private unnamed_addr constant [23 x i8] c"drain dedup collapsed\0A\00"
@m.mconsv   = private unnamed_addr constant [27 x i8] c"metrics 4x1M conservation\0A\00"
@m.gauge    = private unnamed_addr constant [15 x i8] c"gauge set/get\0A\00"
@m.hcount   = private unnamed_addr constant [15 x i8] c"hist count ok\0A\00"
@m.hminmax  = private unnamed_addr constant [14 x i8] c"hist min/max\0A\00"
@m.hmean    = private unnamed_addr constant [11 x i8] c"hist mean\0A\00"
@m.hp50     = private unnamed_addr constant [9 x i8] c"hist p50\00"
@m.hp95     = private unnamed_addr constant [9 x i8] c"hist p95\00"
@m.hp99     = private unnamed_addr constant [9 x i8] c"hist p99\00"
@m.hmerge   = private unnamed_addr constant [21 x i8] c"hist merge == union\0A\00"
@m.null     = private unnamed_addr constant [20 x i8] c"null guards return\0A\00"

@obslog.samp  = internal global [16 x double] zeroinitializer, align 8
@obshist.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.obslog  = private unnamed_addr constant [18 x i8] c"log push 200k rec\00"
@lbl.obshist = private unnamed_addr constant [17 x i8] c"hist rec 1M recs\00"

; ===========================================================================
; workers
; ===========================================================================
define internal ptr @log_worker(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %tid32 = trunc i64 %tid to i32
  %lg = load ptr, ptr @g.lg, align 8
  %n = load i64, ptr @g.n, align 8
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %r = call i32 @universe_observ_log(ptr %lg, i32 1, i32 %tid32, ptr @msg.work, i64 2, i64 %tid, i64 %i, i64 0, i64 0)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret ptr null
}

define internal ptr @m_worker(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %m = load ptr, ptr @g.metrics, align 8
  %n = load i64, ptr @g.mn, align 8
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  call void @universe_observ_counter_inc(ptr %m, i64 0, i64 %tid, i64 1)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret ptr null
}

define internal void @spawn_join(i64 %nt, ptr %fn) {
entry:
  br label %spawn
spawn:
  %i = phi i64 [ 0, %entry ], [ %i.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %i
  %arg = inttoptr i64 %i to ptr
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr %fn, ptr %arg)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nt
  br i1 %more, label %spawn, label %join
join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, %nt
  br i1 %jmore, label %join, label %done
done:
  ret void
}

; ===========================================================================
; logging: level gate + pending
; ===========================================================================
define internal void @test_log_suppress() {
entry:
  ; threshold 2: level 0 and 1 suppressed, level 2/3 logged
  %lg = call ptr @universe_observ_logger_create(i64 64, i32 2)
  %en0 = call i1 @universe_observ_log_enabled(ptr %lg, i32 0)
  %en3 = call i1 @universe_observ_log_enabled(ptr %lg, i32 3)
  %g0 = xor i1 %en0, true
  %gate.ok = and i1 %g0, %en3
  ; a suppressed call records nothing
  %r0 = call i32 @universe_observ_log(ptr %lg, i32 0, i32 0, ptr @msg.a, i64 0, i64 0, i64 0, i64 0)
  %p0 = call i64 @universe_observ_log_pending(ptr %lg)
  %empty = icmp eq i64 %p0, 0
  %ok1 = and i1 %gate.ok, %empty
  call void @ut_check(i1 %ok1, ptr @m.suppress)
  ; an enabled call enqueues exactly one
  %r1 = call i32 @universe_observ_log(ptr %lg, i32 3, i32 0, ptr @msg.a, i64 0, i64 0, i64 0, i64 0)
  %p1 = call i64 @universe_observ_log_pending(ptr %lg)
  %one = icmp eq i64 %p1, 1
  call void @ut_check(i1 %one, ptr @m.pending1)
  call void @universe_observ_logger_destroy(ptr %lg)
  ret void
}

; ===========================================================================
; logging: 4-thread conservation via direct pop tally
; ===========================================================================
define internal void @test_log_conservation() {
entry:
  %lg = call ptr @universe_observ_logger_create(i64 524288, i32 0)
  store ptr %lg, ptr @g.lg, align 8
  store i64 100000, ptr @g.n, align 8
  call void @spawn_join(i64 4, ptr @log_worker)
  ; drop-free: pending == 4*100000
  %pend = call i64 @universe_observ_log_pending(ptr %lg)
  %drop = call i64 @universe_observ_log_dropped(ptr %lg)
  %nodrop = icmp eq i64 %drop, 0
  ; direct-pop tally
  %q = load ptr, ptr %lg, align 8
  %buf = alloca [64 x i8], align 8
  %tally = alloca [4 x i64], align 8
  %t0 = getelementptr inbounds [4 x i64], ptr %tally, i64 0, i64 0
  store i64 0, ptr %t0, align 8
  %t1 = getelementptr inbounds [4 x i64], ptr %tally, i64 0, i64 1
  store i64 0, ptr %t1, align 8
  %t2 = getelementptr inbounds [4 x i64], ptr %tally, i64 0, i64 2
  store i64 0, ptr %t2, align 8
  %t3 = getelementptr inbounds [4 x i64], ptr %tally, i64 0, i64 3
  store i64 0, ptr %t3, align 8
  br label %poploop
poploop:
  %total = phi i64 [ 0, %entry ], [ %total.n, %pcont ]
  %rc = call i32 @universe_conc_mpsc_ring_pop(ptr %q, ptr %buf)
  %done = icmp ne i32 %rc, 0
  br i1 %done, label %check, label %pcont
pcont:
  %tidp = getelementptr inbounds nuw i8, ptr %buf, i64 20
  %tid = load i32, ptr %tidp, align 4
  %tid64 = zext i32 %tid to i64
  %tslot = getelementptr inbounds [4 x i64], ptr %tally, i64 0, i64 %tid64
  %tv = load i64, ptr %tslot, align 8
  %tv.n = add i64 %tv, 1
  store i64 %tv.n, ptr %tslot, align 8
  %total.n = add i64 %total, 1
  br label %poploop
check:
  %consv = icmp eq i64 %total, 400000
  %c0 = and i1 %consv, %nodrop
  %pendok = icmp eq i64 %pend, 400000
  %c1 = and i1 %c0, %pendok
  call void @ut_check(i1 %c1, ptr @m.consv)
  ; per-thread tally each == 100000
  %v0 = load i64, ptr %t0, align 8
  %v1 = load i64, ptr %t1, align 8
  %v2 = load i64, ptr %t2, align 8
  %v3 = load i64, ptr %t3, align 8
  %e0 = icmp eq i64 %v0, 100000
  %e1 = icmp eq i64 %v1, 100000
  %e2 = icmp eq i64 %v2, 100000
  %e3 = icmp eq i64 %v3, 100000
  %te0 = and i1 %e0, %e1
  %te1 = and i1 %e2, %e3
  %tallyok = and i1 %te0, %te1
  call void @ut_check(i1 %tallyok, ptr @m.tally)
  call void @universe_observ_logger_destroy(ptr %lg)
  ret void
}

; ===========================================================================
; logging: drain formats + collapses consecutive identical records
;   pattern (10 records): a b c d d d d d e f  -> 6 output lines
; ===========================================================================
define internal void @test_log_drain() {
entry:
  %fds = alloca [2 x i32], align 4
  %pr = call i32 @pipe(ptr %fds)
  %rfdp = getelementptr inbounds [2 x i32], ptr %fds, i64 0, i64 0
  %wfdp = getelementptr inbounds [2 x i32], ptr %fds, i64 0, i64 1
  %rfd = load i32, ptr %rfdp, align 4
  %wfd = load i32, ptr %wfdp, align 4
  %w = call ptr @universe_io_writer_create(i32 %wfd, i64 4096)
  %lg = call ptr @universe_observ_logger_create(i64 64, i32 0)
  ; 3 distinct
  %r1 = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.a, i64 1, i64 1, i64 0, i64 0, i64 0)
  %r2 = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.b, i64 1, i64 2, i64 0, i64 0, i64 0)
  %r3 = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.c, i64 1, i64 3, i64 0, i64 0, i64 0)
  br label %dups
dups:
  %k = phi i64 [ 0, %entry ], [ %k.n, %dups ]
  %rd = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.a, i64 1, i64 9, i64 0, i64 0, i64 0)
  %k.n = add nuw i64 %k, 1
  %kmore = icmp ult i64 %k.n, 5
  br i1 %kmore, label %dups, label %tail
tail:
  %r4 = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.b, i64 1, i64 7, i64 0, i64 0, i64 0)
  %r5 = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.c, i64 1, i64 8, i64 0, i64 0, i64 0)
  ; drain -> writes to pipe + flush
  %n = call i64 @universe_observ_log_drain(ptr %lg, ptr %w)
  call void @universe_io_writer_destroy(ptr %w)
  %cr = call i32 @close(i32 %wfd)
  %nok = icmp eq i64 %n, 10
  call void @ut_check(i1 %nok, ptr @m.drainn)
  ; read pipe, count newlines
  %rbuf = alloca [4096 x i8], align 1
  %got = call i64 @read(i32 %rfd, ptr %rbuf, i64 4096)
  %cr2 = call i32 @close(i32 %rfd)
  br label %scan
scan:
  %si = phi i64 [ 0, %tail ], [ %si.n, %scont ]
  %lines = phi i64 [ 0, %tail ], [ %lines.n, %scont ]
  %sdone = icmp sge i64 %si, %got
  br i1 %sdone, label %sdoneb, label %sbody
sbody:
  %cp = getelementptr inbounds nuw i8, ptr %rbuf, i64 %si
  %ch = load i8, ptr %cp, align 1
  %isnl = icmp eq i8 %ch, 10
  %inc = zext i1 %isnl to i64
  %lines.n = add i64 %lines, %inc
  %si.n = add nuw i64 %si, 1
  br label %scont
scont:
  br label %scan
sdoneb:
  %linesok = icmp eq i64 %lines, 6
  call void @ut_check(i1 %linesok, ptr @m.lines)
  call void @universe_observ_logger_destroy(ptr %lg)
  ret void
}

; ===========================================================================
; metrics
; ===========================================================================
define internal void @test_metrics() {
entry:
  %m = call ptr @universe_observ_metrics_create(i64 4, i64 4, i64 8)
  store ptr %m, ptr @g.metrics, align 8
  store i64 1000000, ptr @g.mn, align 8
  call void @spawn_join(i64 4, ptr @m_worker)
  %sum = call i64 @universe_observ_counter_sum(ptr %m, i64 0)
  %ok = icmp eq i64 %sum, 4000000
  call void @ut_check(i1 %ok, ptr @m.mconsv)
  ; gauge set/get
  call void @universe_observ_gauge_set(ptr %m, i64 2, i64 12345)
  %g = call i64 @universe_observ_gauge_get(ptr %m, i64 2)
  %gok = icmp eq i64 %g, 12345
  call void @ut_check(i1 %gok, ptr @m.gauge)
  call void @universe_observ_metrics_destroy(ptr %m)
  ret void
}

; ===========================================================================
; histogram: fill [lo, hi] inclusive
; ===========================================================================
define internal void @hist_fill(ptr %h, i64 %lo, i64 %hi) {
entry:
  br label %loop
loop:
  %v = phi i64 [ %lo, %entry ], [ %v.n, %loop ]
  call void @universe_observ_hist_record(ptr %h, i64 %v)
  %v.n = add i64 %v, 1
  %more = icmp ule i64 %v.n, %hi
  br i1 %more, label %loop, label %done
done:
  ret void
}

define internal void @test_hist() {
entry:
  %h = call ptr @universe_observ_hist_create(i64 1, i64 100000, i64 3)
  call void @hist_fill(ptr %h, i64 1, i64 10000)
  %cnt = call i64 @universe_observ_hist_count(ptr %h)
  %cok = icmp eq i64 %cnt, 10000
  call void @ut_check(i1 %cok, ptr @m.hcount)
  %mn = call i64 @universe_observ_hist_min(ptr %h)
  %mx = call i64 @universe_observ_hist_max(ptr %h)
  %mnok = icmp eq i64 %mn, 1
  %mxok = icmp eq i64 %mx, 10000
  %mmok = and i1 %mnok, %mxok
  call void @ut_check(i1 %mmok, ptr @m.hminmax)
  ; mean ~ 5000.5
  %mean = call double @universe_observ_hist_mean(ptr %h)
  %md = fsub double %mean, 5.000500e+03
  %mabs = call double @llvm.fabs.f64(double %md)
  %meanok = fcmp olt double %mabs, 1.000000e+00
  call void @ut_check(i1 %meanok, ptr @m.hmean)
  ; percentiles within ~2%
  %p50 = call i64 @universe_observ_hist_percentile(ptr %h, double 5.000000e+01)
  call void @check_near(i64 %p50, i64 5000, i64 120, ptr @m.hp50)
  %p95 = call i64 @universe_observ_hist_percentile(ptr %h, double 9.500000e+01)
  call void @check_near(i64 %p95, i64 9500, i64 200, ptr @m.hp95)
  %p99 = call i64 @universe_observ_hist_percentile(ptr %h, double 9.900000e+01)
  call void @check_near(i64 %p99, i64 9900, i64 200, ptr @m.hp99)
  call void @universe_observ_hist_destroy(ptr %h)

  ; merge == union
  %a = call ptr @universe_observ_hist_create(i64 1, i64 100000, i64 3)
  %b = call ptr @universe_observ_hist_create(i64 1, i64 100000, i64 3)
  %u = call ptr @universe_observ_hist_create(i64 1, i64 100000, i64 3)
  call void @hist_fill(ptr %a, i64 1, i64 10000)
  call void @hist_fill(ptr %b, i64 10001, i64 20000)
  call void @hist_fill(ptr %u, i64 1, i64 20000)
  %mrc = call i32 @universe_observ_hist_merge(ptr %a, ptr %b)
  %cu = call i64 @universe_observ_hist_count(ptr %u)
  %ca = call i64 @universe_observ_hist_count(ptr %a)
  %cnteq = icmp eq i64 %ca, %cu
  %pa50 = call i64 @universe_observ_hist_percentile(ptr %a, double 5.000000e+01)
  %pu50 = call i64 @universe_observ_hist_percentile(ptr %u, double 5.000000e+01)
  %pa99 = call i64 @universe_observ_hist_percentile(ptr %a, double 9.900000e+01)
  %pu99 = call i64 @universe_observ_hist_percentile(ptr %u, double 9.900000e+01)
  %pe50 = icmp eq i64 %pa50, %pu50
  %pe99 = icmp eq i64 %pa99, %pu99
  %me0 = icmp eq i32 %mrc, 0
  %me1 = and i1 %cnteq, %pe50
  %me2 = and i1 %me1, %pe99
  %mergeok = and i1 %me2, %me0
  call void @ut_check(i1 %mergeok, ptr @m.hmerge)
  call void @universe_observ_hist_destroy(ptr %a)
  call void @universe_observ_hist_destroy(ptr %b)
  call void @universe_observ_hist_destroy(ptr %u)
  ret void
}

declare double @llvm.fabs.f64(double)

define internal void @check_near(i64 %got, i64 %want, i64 %tol, ptr %msg) {
entry:
  %d = sub i64 %got, %want
  %neg = icmp slt i64 %d, 0
  %nd = sub i64 0, %d
  %ad = select i1 %neg, i64 %nd, i64 %d
  %ok = icmp ule i64 %ad, %tol
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; ===========================================================================
; null guards
; ===========================================================================
define internal void @test_null() {
entry:
  %e = call i1 @universe_observ_log_enabled(ptr null, i32 0)
  %r = call i32 @universe_observ_log(ptr null, i32 1, i32 0, ptr @msg.a, i64 0, i64 0, i64 0, i64 0)
  %d = call i64 @universe_observ_log_drain(ptr null, ptr null)
  %cs = call i64 @universe_observ_counter_sum(ptr null, i64 0)
  %gg = call i64 @universe_observ_gauge_get(ptr null, i64 0)
  %hc = call i64 @universe_observ_hist_count(ptr null)
  %hp = call i64 @universe_observ_hist_percentile(ptr null, double 5.000000e+01)
  %hm = call i32 @universe_observ_hist_merge(ptr null, ptr null)
  call void @universe_observ_hist_record(ptr null, i64 5)
  call void @universe_observ_counter_inc(ptr null, i64 0, i64 0, i64 1)
  %en.ok = xor i1 %e, true
  %r.ok = icmp eq i32 %r, 1
  %d.ok = icmp eq i64 %d, 0
  %cs.ok = icmp eq i64 %cs, 0
  %hp.ok = icmp eq i64 %hp, 0
  %hm.ok = icmp eq i32 %hm, 1
  %a0 = and i1 %en.ok, %r.ok
  %a1 = and i1 %a0, %d.ok
  %a2 = and i1 %a1, %cs.ok
  %a3 = and i1 %a2, %hp.ok
  %a4 = and i1 %a3, %hm.ok
  call void @ut_check(i1 %a4, ptr @m.null)
  ret void
}

; ===========================================================================
; --bench
; ===========================================================================
define internal void @bench() {
entry:
  %seed = alloca i64, align 8
  ; log push throughput distribution (ops_per_rep = 200000 records)
  ; warmup rep (untimed discard)
  %wlg = call ptr @universe_observ_logger_create(i64 2097152, i32 0)
  call void @push_n(ptr %wlg, i64 200000)
  call void @universe_observ_logger_destroy(ptr %wlg)
  br label %lrep
lrep:
  %ri = phi i64 [ 0, %entry ], [ %ri.n, %lrep ]
  %lg = call ptr @universe_observ_logger_create(i64 2097152, i32 0)
  %t0 = call double @ut_now_sec()
  call void @push_n(ptr %lg, i64 200000)
  %t1 = call double @ut_now_sec()
  call void @universe_observ_logger_destroy(ptr %lg)
  %dt = fsub double %t1, %t0
  %lsp = getelementptr inbounds [16 x double], ptr @obslog.samp, i64 0, i64 %ri
  store double %dt, ptr %lsp, align 8
  %ri.n = add nuw i64 %ri, 1
  %lmore = icmp ult i64 %ri.n, 16
  br i1 %lmore, label %lrep, label %lreport
lreport:
  call void @ut_report_dist(ptr @obslog.samp, i64 16, i64 200000, ptr @lbl.obslog)

  ; hist record throughput distribution (ops_per_rep = 1000000 records)
  store i64 88172645463325252, ptr %seed, align 8
  br label %hrep
hrep:
  ; rep 0 is warm-up (discarded); reps 1..16 stored
  %hi = phi i64 [ 0, %lreport ], [ %hi.n, %hrep.cont ]
  %h = call ptr @universe_observ_hist_create(i64 1, i64 1000000, i64 3)
  %h0 = call double @ut_now_sec()
  call void @hrec_n(ptr %h, i64 1000000, ptr %seed)
  %h1 = call double @ut_now_sec()
  call void @universe_observ_hist_destroy(ptr %h)
  %hdt = fsub double %h1, %h0
  %hwarm = icmp eq i64 %hi, 0
  br i1 %hwarm, label %hrep.cont, label %hrep.store
hrep.store:
  %hidx = sub i64 %hi, 1
  %hsp = getelementptr inbounds [16 x double], ptr @obshist.samp, i64 0, i64 %hidx
  store double %hdt, ptr %hsp, align 8
  br label %hrep.cont
hrep.cont:
  %hi.n = add nuw i64 %hi, 1
  %hmore = icmp ult i64 %hi.n, 17
  br i1 %hmore, label %hrep, label %hreport
hreport:
  call void @ut_report_dist(ptr @obshist.samp, i64 16, i64 1000000, ptr @lbl.obshist)
  ret void
}

define internal void @push_n(ptr %lg, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %r = call i32 @universe_observ_log(ptr %lg, i32 1, i32 0, ptr @msg.work, i64 2, i64 %i, i64 42, i64 0, i64 0)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

define internal void @hrec_n(ptr %h, i64 %n, ptr %seed) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %rnd = call i64 @ut_rand(ptr %seed)
  %v = urem i64 %rnd, 1000000
  %v1 = add i64 %v, 1
  call void @universe_observ_hist_record(ptr %h, i64 %v1)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

declare double @llvm.minnum.f64(double, double)

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_log_suppress()
  call void @test_log_drain()
  call void @test_hist()
  call void @test_metrics()
  call void @test_log_conservation()
  call void @test_null()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %dobench, label %fin
dobench:
  call void @bench()
  br label %fin
fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
