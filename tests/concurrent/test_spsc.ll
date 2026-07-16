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

; Tests for universe_conc_spsc (bounded wait-free ring) and
; universe_conc_spsc_seg (unbounded segment queue).
;
; The real gate is the 1-producer / 1-consumer stress: the producer pushes the
; strictly increasing sequence 0..N-1; the consumer must observe EXACTLY that
; sequence (any lost / duplicated / torn / reordered item trips a violation
; counter) AND the running sum must match N*(N-1)/2. The whole stress loops
; REPS times for both structures. This is the weak-memory proof on ARM64.

declare ptr @universe_conc_spsc_create(i64, i64)
declare i32 @universe_conc_spsc_enqueue(ptr, ptr)
declare i32 @universe_conc_spsc_dequeue(ptr, ptr)
declare i64 @universe_conc_spsc_count(ptr)
declare i32 @universe_conc_spsc_is_empty(ptr)
declare i32 @universe_conc_spsc_is_full(ptr)
declare i64 @universe_conc_spsc_capacity(ptr)
declare void @universe_conc_spsc_destroy(ptr)

declare ptr @universe_conc_spsc_seg_create(i64)
declare i32 @universe_conc_spsc_seg_enqueue(ptr, ptr)
declare i32 @universe_conc_spsc_seg_dequeue(ptr, ptr)
declare i64 @universe_conc_spsc_seg_count(ptr)
declare i32 @universe_conc_spsc_seg_is_empty(ptr)
declare void @universe_conc_spsc_seg_destroy(ptr)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @sched_yield()
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@g.q = internal global ptr null, align 8

@spsc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.spsc  = private unnamed_addr constant [22 x i8] c"bounded spsc push+pop\00"

@m.b.single = private unnamed_addr constant [27 x i8] c"bounded single-thread FIFO\00"
@m.b.full   = private unnamed_addr constant [19 x i8] c"bounded full/empty\00"
@m.s.single = private unnamed_addr constant [32 x i8] c"segment single-thread FIFO+span\00"
@m.s.empty  = private unnamed_addr constant [20 x i8] c"segment empty state\00"
@m.stress   = private unnamed_addr constant [30 x i8] c"1P/1C stress: strict FIFO+sum\00"

; ---------------------------------------------------------------------------
; Single-threaded correctness
; ---------------------------------------------------------------------------

define internal void @test_single_bounded() {
entry:
  ; capacity rounds up to a minimum power-of-two of 8.
  %rb = call ptr @universe_conc_spsc_create(i64 8, i64 8)
  %v = alloca i64, align 8
  br label %fill

fill:                                 ; push 10..17 (8 items => full)
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %fviol = phi i64 [ 0, %entry ], [ %fviol.n, %fill ]
  %pv = add nuw nsw i64 %i, 10
  store i64 %pv, ptr %v, align 8
  %pe = call i32 @universe_conc_spsc_enqueue(ptr %rb, ptr %v)
  %fbad = icmp ne i32 %pe, 0
  %finc = zext i1 %fbad to i64
  %fviol.n = add i64 %fviol, %finc
  %i.n = add nuw nsw i64 %i, 1
  %fmore = icmp ult i64 %i.n, 8
  br i1 %fmore, label %fill, label %afterfill

afterfill:
  ; 9th push must be FULL
  store i64 99, ptr %v, align 8
  %pfull = call i32 @universe_conc_spsc_enqueue(ptr %rb, ptr %v)
  %isfull = call i32 @universe_conc_spsc_is_full(ptr %rb)
  %cnt8 = call i64 @universe_conc_spsc_count(ptr %rb)
  %cap = call i64 @universe_conc_spsc_capacity(ptr %rb)
  ; drain 10,11
  %d0 = call i32 @universe_conc_spsc_dequeue(ptr %rb, ptr %v)
  %g0 = load i64, ptr %v, align 8
  %d1 = call i32 @universe_conc_spsc_dequeue(ptr %rb, ptr %v)
  %g1 = load i64, ptr %v, align 8
  ; wrap: push 18,19
  store i64 18, ptr %v, align 8
  %p5 = call i32 @universe_conc_spsc_enqueue(ptr %rb, ptr %v)
  store i64 19, ptr %v, align 8
  %p6 = call i32 @universe_conc_spsc_enqueue(ptr %rb, ptr %v)
  br label %drain

drain:                                ; expect 12..19 in order
  %j = phi i64 [ 0, %afterfill ], [ %j.n, %drain ]
  %dviol = phi i64 [ 0, %afterfill ], [ %dviol.n, %drain ]
  %dr = call i32 @universe_conc_spsc_dequeue(ptr %rb, ptr %v)
  %gv = load i64, ptr %v, align 8
  %exp = add nuw nsw i64 %j, 12
  %dbad.v = icmp ne i64 %gv, %exp
  %dbad.rc = icmp ne i32 %dr, 0
  %dbad = or i1 %dbad.v, %dbad.rc
  %dinc = zext i1 %dbad to i64
  %dviol.n = add i64 %dviol, %dinc
  %j.n = add nuw nsw i64 %j, 1
  %dmore = icmp ult i64 %j.n, 8
  br i1 %dmore, label %drain, label %tail

tail:
  ; now empty
  %d6 = call i32 @universe_conc_spsc_dequeue(ptr %rb, ptr %v)
  %isempty = call i32 @universe_conc_spsc_is_empty(ptr %rb)

  ; assemble FIFO correctness (fill + drain had zero violations)
  %fifo.ok = icmp eq i64 %dviol.n, 0
  call void @ut_check(i1 %fifo.ok, ptr @m.b.single)

  ; assemble full/empty/error-code correctness
  %e0 = icmp eq i64 %fviol.n, 0
  %e1 = icmp eq i32 %pfull, 6         ; FULL
  %e2 = icmp eq i32 %isfull, 1
  %e3 = icmp eq i64 %cnt8, 8
  %e4 = icmp eq i64 %cap, 8
  %e5 = icmp eq i32 %d6, 4            ; EMPTY
  %e6 = icmp eq i32 %isempty, 1
  %b0 = and i1 %e0, %e1
  %b1 = and i1 %b0, %e2
  %b2 = and i1 %b1, %e3
  %b3 = and i1 %b2, %e4
  %b4 = and i1 %b3, %e5
  %b5 = and i1 %b4, %e6
  call void @ut_check(i1 %b5, ptr @m.b.full)
  call void @universe_conc_spsc_destroy(ptr %rb)
  ret void
}

define internal void @test_single_seg() {
entry:
  %q = call ptr @universe_conc_spsc_seg_create(i64 8)
  %isempty0 = call i32 @universe_conc_spsc_seg_is_empty(ptr %q)
  %cnt0 = call i64 @universe_conc_spsc_seg_count(ptr %q)
  %v = alloca i64, align 8
  br label %fill

fill:                                 ; enqueue 0..2499 (spans 3 chunks)
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %pe = call i32 @universe_conc_spsc_seg_enqueue(ptr %q, ptr %v)
  %i.n = add nuw nsw i64 %i, 1
  %fmore = icmp ult i64 %i.n, 2500
  br i1 %fmore, label %fill, label %midcheck

midcheck:
  %cntmid = call i64 @universe_conc_spsc_seg_count(ptr %q)
  br label %drain

drain:
  %j = phi i64 [ 0, %midcheck ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %midcheck ], [ %viol.n, %drain ]
  %pd = call i32 @universe_conc_spsc_seg_dequeue(ptr %q, ptr %v)
  %val = load i64, ptr %v, align 8
  %bad.v = icmp ne i64 %val, %j
  %bad.rc = icmp ne i32 %pd, 0
  %bad = or i1 %bad.v, %bad.rc
  %inc = zext i1 %bad to i64
  %viol.n = add i64 %viol, %inc
  %j.n = add nuw nsw i64 %j, 1
  %dmore = icmp ult i64 %j.n, 2500
  br i1 %dmore, label %drain, label %finish

finish:
  %de = call i32 @universe_conc_spsc_seg_dequeue(ptr %q, ptr %v)   ; EMPTY now
  %cntf = call i64 @universe_conc_spsc_seg_count(ptr %q)
  %isempty1 = call i32 @universe_conc_spsc_seg_is_empty(ptr %q)

  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.s.single)

  %s0 = icmp eq i32 %isempty0, 1
  %s1 = icmp eq i64 %cnt0, 0
  %s2 = icmp eq i64 %cntmid, 2500
  %s3 = icmp eq i32 %de, 4
  %s4 = icmp eq i64 %cntf, 0
  %s5 = icmp eq i32 %isempty1, 1
  %c0 = and i1 %s0, %s1
  %c1 = and i1 %c0, %s2
  %c2 = and i1 %c1, %s3
  %c3 = and i1 %c2, %s4
  %c4 = and i1 %c3, %s5
  call void @ut_check(i1 %c4, ptr @m.s.empty)
  call void @universe_conc_spsc_seg_destroy(ptr %q)
  ret void
}

; ---------------------------------------------------------------------------
; Concurrent stress (1 producer thread + main consumer)
; ---------------------------------------------------------------------------

define internal ptr @producer_bounded(ptr %arg) {
entry:
  %q = load ptr, ptr @g.q, align 8
  %v = alloca i64, align 8
  br label %next

next:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  store i64 %i, ptr %v, align 8
  br label %try

try:
  %rc = call i32 @universe_conc_spsc_enqueue(ptr %q, ptr %v)
  %full = icmp ne i32 %rc, 0
  br i1 %full, label %yield, label %pushed

yield:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %next, label %done

done:
  ret ptr null
}

define internal ptr @producer_seg(ptr %arg) {
entry:
  %q = load ptr, ptr @g.q, align 8
  %v = alloca i64, align 8
  br label %next

next:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  store i64 %i, ptr %v, align 8
  br label %try

try:
  %rc = call i32 @universe_conc_spsc_seg_enqueue(ptr %q, ptr %v)
  %full = icmp ne i32 %rc, 0
  br i1 %full, label %yield, label %pushed

yield:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %next, label %done

done:
  ret ptr null
}

define internal i64 @stress_bounded() {
entry:
  %q = call ptr @universe_conc_spsc_create(i64 1024, i64 8)
  store ptr %q, ptr @g.q, align 8
  %tid.slot = alloca i64, align 8
  %rc = call i32 @pthread_create(ptr nonnull %tid.slot, ptr null, ptr @producer_bounded, ptr null)
  %spawn.bad = icmp ne i32 %rc, 0
  br i1 %spawn.bad, label %spawn.fail, label %setup

spawn.fail:
  call void @universe_conc_spsc_destroy(ptr %q)
  ret i64 1

setup:
  %v = alloca i64, align 8
  br label %consume

consume:
  %i = phi i64 [ 0, %setup ], [ %i.n, %got ]
  %viol = phi i64 [ 0, %setup ], [ %viol.n, %got ]
  %sum = phi i64 [ 0, %setup ], [ %sum.n, %got ]
  br label %try

try:
  %prc = call i32 @universe_conc_spsc_dequeue(ptr %q, ptr %v)
  %empty = icmp ne i32 %prc, 0
  br i1 %empty, label %yield, label %got

yield:
  %y = call i32 @sched_yield()
  br label %try

got:
  %val = load i64, ptr %v, align 8
  %bad = icmp ne i64 %val, %i
  %inc = zext i1 %bad to i64
  %viol.n = add i64 %viol, %inc
  %sum.n = add i64 %sum, %val
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %consume, label %finish

finish:
  %tid = load i64, ptr %tid.slot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %left = call i64 @universe_conc_spsc_count(ptr %q)
  %badsum = icmp ne i64 %sum.n, 499999500000
  %sinc = zext i1 %badsum to i64
  %badleft = icmp ne i64 %left, 0
  %linc = zext i1 %badleft to i64
  %t1 = add i64 %viol.n, %sinc
  %tot = add i64 %t1, %linc
  call void @universe_conc_spsc_destroy(ptr %q)
  ret i64 %tot
}

define internal i64 @stress_seg() {
entry:
  %q = call ptr @universe_conc_spsc_seg_create(i64 8)
  store ptr %q, ptr @g.q, align 8
  %tid.slot = alloca i64, align 8
  %rc = call i32 @pthread_create(ptr nonnull %tid.slot, ptr null, ptr @producer_seg, ptr null)
  %spawn.bad = icmp ne i32 %rc, 0
  br i1 %spawn.bad, label %spawn.fail, label %setup

spawn.fail:
  call void @universe_conc_spsc_seg_destroy(ptr %q)
  ret i64 1

setup:
  %v = alloca i64, align 8
  br label %consume

consume:
  %i = phi i64 [ 0, %setup ], [ %i.n, %got ]
  %viol = phi i64 [ 0, %setup ], [ %viol.n, %got ]
  %sum = phi i64 [ 0, %setup ], [ %sum.n, %got ]
  br label %try

try:
  %prc = call i32 @universe_conc_spsc_seg_dequeue(ptr %q, ptr %v)
  %empty = icmp ne i32 %prc, 0
  br i1 %empty, label %yield, label %got

yield:
  %y = call i32 @sched_yield()
  br label %try

got:
  %val = load i64, ptr %v, align 8
  %bad = icmp ne i64 %val, %i
  %inc = zext i1 %bad to i64
  %viol.n = add i64 %viol, %inc
  %sum.n = add i64 %sum, %val
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %consume, label %finish

finish:
  %tid = load i64, ptr %tid.slot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %left = call i64 @universe_conc_spsc_seg_count(ptr %q)
  %badsum = icmp ne i64 %sum.n, 499999500000
  %sinc = zext i1 %badsum to i64
  %badleft = icmp ne i64 %left, 0
  %linc = zext i1 %badleft to i64
  %t1 = add i64 %viol.n, %sinc
  %tot = add i64 %t1, %linc
  call void @universe_conc_spsc_seg_destroy(ptr %q)
  ret i64 %tot
}

define internal void @run_stress() {
entry:
  br label %loop

loop:
  %r = phi i64 [ 0, %entry ], [ %r.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %vb = call i64 @stress_bounded()
  %vs = call i64 @stress_seg()
  %s = add i64 %vb, %vs
  %acc.n = add i64 %acc, %s
  %r.n = add nuw nsw i64 %r, 1
  %more = icmp ult i64 %r.n, 10
  br i1 %more, label %loop, label %fin

fin:
  call void @ut_check_eq(i64 %acc.n, i64 0, ptr @m.stress)
  ret void
}

; ---------------------------------------------------------------------------
; Bench: steady-state push+pop throughput on the bounded ring
; ---------------------------------------------------------------------------

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op).
; 17 reps per op; rep 0 is warm-up (discarded), reps 1..16 recorded.
; batch M = 20M push+pop iterations; ops_per_rep = 40M (enqueue+dequeue).
define internal void @bench() {
entry:
  %q = call ptr @universe_conc_spsc_create(i64 1024, i64 8)
  %v = alloca i64, align 8
  store i64 42, ptr %v, align 8
  br label %rep

rep:
  %r = phi i64 [ 0, %entry ], [ %r.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  %e = call i32 @universe_conc_spsc_enqueue(ptr %q, ptr %v)
  %d = call i32 @universe_conc_spsc_dequeue(ptr %q, ptr %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 20000000
  br i1 %more, label %loop, label %rep.mid

rep.mid:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %r, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %idx = sub i64 %r, 1
  %sp = getelementptr inbounds [16 x double], ptr @spsc.samp, i64 0, i64 %idx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %r.n = add nuw nsw i64 %r, 1
  %rmore = icmp ult i64 %r.n, 17
  br i1 %rmore, label %rep, label %done

done:
  call void @ut_report_dist(ptr @spsc.samp, i64 16, i64 40000000, ptr @lbl.spsc)
  call void @universe_conc_spsc_destroy(ptr %q)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single_bounded()
  call void @test_single_seg()
  %want = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %want, label %do.bench, label %do.stress

do.bench:
  call void @bench()
  br label %end

do.stress:
  call void @run_stress()
  br label %end

end:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
