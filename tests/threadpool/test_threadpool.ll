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

; Tests for universe_threadpool: exact completion counts, full-ring
; backpressure (queue < tasks), pool reuse after wait, concurrent submitters,
; error codes, --bench task round-trip.

declare ptr @universe_threadpool_create(i64, i64)
declare i32 @universe_threadpool_submit(ptr, ptr, ptr)
declare i32 @universe_threadpool_wait(ptr)
declare i32 @universe_threadpool_destroy(ptr)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@tp.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.tp  = private unnamed_addr constant [23 x i8] c"threadpool 8w submit  \00"

@g.count = internal global i64 0, align 8
@g.tp = internal global ptr null, align 8
@g.subtids = internal global [4 x i64] zeroinitializer, align 8
@g.subfails = internal global i64 0, align 8

@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.count100 = private unnamed_addr constant [21 x i8] c"100k tasks completed\00"
@m.count150 = private unnamed_addr constant [21 x i8] c"pool reusable +50k  \00"
@m.destroy  = private unnamed_addr constant [11 x i8] c"destroy ok\00"
@m.backpr   = private unnamed_addr constant [26 x i8] c"backpressure 1k via q=16 \00"
@m.errs     = private unnamed_addr constant [12 x i8] c"error codes\00"
@m.multi    = private unnamed_addr constant [26 x i8] c"4 submitters x25k = 100k \00"
@m.subok    = private unnamed_addr constant [21 x i8] c"submitters no errors\00"

define internal void @task_inc(ptr %arg) {
entry:
  %old = atomicrmw add ptr %arg, i64 1 monotonic, align 8
  ret void
}

define internal void @test_basic() {
entry:
  store i64 0, ptr @g.count, align 8
  %tp = call ptr @universe_threadpool_create(i64 4, i64 256)
  %ok = icmp ne ptr %tp, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %submit, label %done

submit:
  %i = phi i64 [ 0, %entry ], [ %i.n, %submit ]
  %rc = call i32 @universe_threadpool_submit(ptr %tp, ptr @task_inc, ptr @g.count)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %submit, label %wait1

wait1:
  %w1 = call i32 @universe_threadpool_wait(ptr %tp)
  %c1 = load i64, ptr @g.count, align 8
  call void @ut_check_eq(i64 %c1, i64 100000, ptr @m.count100)
  br label %submit2

submit2:                                    ; pool stays usable after wait
  %j = phi i64 [ 0, %wait1 ], [ %j.n, %submit2 ]
  %rc2 = call i32 @universe_threadpool_submit(ptr %tp, ptr @task_inc, ptr @g.count)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 50000
  br i1 %more2, label %submit2, label %wait2

wait2:
  %w2 = call i32 @universe_threadpool_wait(ptr %tp)
  %c2 = load i64, ptr @g.count, align 8
  call void @ut_check_eq(i64 %c2, i64 150000, ptr @m.count150)
  %d = call i32 @universe_threadpool_destroy(ptr %tp)
  %d.w = zext i32 %d to i64
  call void @ut_check_eq(i64 %d.w, i64 0, ptr @m.destroy)
  br label %done

done:
  ret void
}

define internal void @test_backpressure() {
entry:                                      ; 1 worker, tiny ring: submit must
  store i64 0, ptr @g.count, align 8        ; block instead of dropping tasks
  %tp = call ptr @universe_threadpool_create(i64 1, i64 16)
  br label %submit

submit:
  %i = phi i64 [ 0, %entry ], [ %i.n, %submit ]
  %rc = call i32 @universe_threadpool_submit(ptr %tp, ptr @task_inc, ptr @g.count)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %submit, label %wait

wait:
  %w = call i32 @universe_threadpool_wait(ptr %tp)
  %c = load i64, ptr @g.count, align 8
  call void @ut_check_eq(i64 %c, i64 1000, ptr @m.backpr)
  %d = call i32 @universe_threadpool_destroy(ptr %tp)
  ret void
}

define internal void @test_errors() {
entry:
  %tp = call ptr @universe_threadpool_create(i64 2, i64 64)
  %e1 = call i32 @universe_threadpool_submit(ptr null, ptr @task_inc, ptr @g.count)
  %e2 = call i32 @universe_threadpool_submit(ptr %tp, ptr null, ptr null)
  %e3 = call i32 @universe_threadpool_wait(ptr null)
  %e4 = call i32 @universe_threadpool_destroy(ptr null)
  %s1 = add nuw i32 %e1, %e2
  %s2 = add nuw i32 %e3, %e4
  %s = add nuw i32 %s1, %s2                 ; four NULL_PTR errors = 4
  %s.w = zext i32 %s to i64
  call void @ut_check_eq(i64 %s.w, i64 4, ptr @m.errs)
  %d = call i32 @universe_threadpool_destroy(ptr %tp)
  ret void
}

; concurrent submitter: 25k tasks each, from 4 threads at once
define internal ptr @submitter(ptr %arg) {
entry:
  %tp = load ptr, ptr @g.tp, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %errs = phi i64 [ 0, %entry ], [ %errs.n, %loop ]
  %rc = call i32 @universe_threadpool_submit(ptr %tp, ptr @task_inc, ptr @g.count)
  %bad = icmp ne i32 %rc, 0
  %e.inc = zext i1 %bad to i64
  %errs.n = add nuw i64 %errs, %e.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 25000
  br i1 %more, label %loop, label %tally

tally:
  %old = atomicrmw add ptr @g.subfails, i64 %errs.n monotonic, align 8
  ret ptr null
}

define internal void @test_multi_submitter() {
entry:
  store i64 0, ptr @g.count, align 8
  store i64 0, ptr @g.subfails, align 8
  %tp = call ptr @universe_threadpool_create(i64 4, i64 128)
  store ptr %tp, ptr @g.tp, align 8
  br label %spawn

spawn:
  %t = phi i64 [ 0, %entry ], [ %t.n, %spawn ]
  %slot = getelementptr inbounds nuw [4 x i64], ptr @g.subtids, i64 0, i64 %t
  %rc = call i32 @pthread_create(ptr %slot, ptr null, ptr @submitter, ptr null)
  %t.n = add nuw nsw i64 %t, 1
  %more = icmp ult i64 %t.n, 4
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %slot2 = getelementptr inbounds nuw [4 x i64], ptr @g.subtids, i64 0, i64 %j
  %tid = load i64, ptr %slot2, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 4
  br i1 %more2, label %join, label %verify

verify:
  %w = call i32 @universe_threadpool_wait(ptr %tp)
  %fails = load i64, ptr @g.subfails, align 8
  call void @ut_check_eq(i64 %fails, i64 0, ptr @m.subok)
  %c = load i64, ptr @g.count, align 8
  call void @ut_check_eq(i64 %c, i64 100000, ptr @m.multi)
  %d = call i32 @universe_threadpool_destroy(ptr %tp)
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) of the
; submit-100k-tasks-and-drain batch on an 8-worker pool. 17 reps: rep 0 warm-up
; (discarded), reps 1..16 recorded. Each rep recreates the pool and resets the
; counter. ops_per_rep = 100k tasks.
define internal void @bench() {
entry:
  br label %rep

rep:
  %r = phi i64 [ 0, %entry ], [ %r.n, %rep.next ]
  store i64 0, ptr @g.count, align 8
  %tp = call ptr @universe_threadpool_create(i64 8, i64 1024)
  %t0 = call double @ut_now_sec()
  br label %submit

submit:
  %i = phi i64 [ 0, %rep ], [ %i.n, %submit ]
  %rc = call i32 @universe_threadpool_submit(ptr %tp, ptr @task_inc, ptr @g.count)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %submit, label %wait

wait:
  %w = call i32 @universe_threadpool_wait(ptr %tp)
  %t1 = call double @ut_now_sec()
  %d = call i32 @universe_threadpool_destroy(ptr %tp)
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %r, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %idx = sub i64 %r, 1
  %sp = getelementptr inbounds [16 x double], ptr @tp.samp, i64 0, i64 %idx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %r.n = add nuw nsw i64 %r, 1
  %rmore = icmp ult i64 %r.n, 17
  br i1 %rmore, label %rep, label %done

done:
  call void @ut_report_dist(ptr @tp.samp, i64 16, i64 100000, ptr @lbl.tp)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_backpressure()
  call void @test_errors()
  call void @test_multi_submitter()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
