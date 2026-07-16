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

; Tests for universe_conc_mpsc_ring (bounded) and _seg (unbounded).
; Stress: 4 producer threads each enqueue 100000 distinct items encoded as
; (producer_id << 32 | seq); the single consumer (main) drains all 400000 and
; verifies exactly-once delivery (per-producer expected-count) AND per-producer
; FIFO (a producer's items arrive in its own seq order). Any loss, duplication,
; tear or reorder trips the aggregated violation counter. Weak-memory proof on
; ARM64: publish is release, consume is acquire.

declare ptr @universe_conc_mpsc_ring_create(i64, i64)
declare i32 @universe_conc_mpsc_ring_push(ptr, ptr)
declare i32 @universe_conc_mpsc_ring_pop(ptr, ptr)
declare i64 @universe_conc_mpsc_ring_count(ptr)
declare i64 @universe_conc_mpsc_ring_capacity(ptr)
declare void @universe_conc_mpsc_ring_destroy(ptr)

declare ptr @universe_conc_mpsc_seg_create(i64)
declare i32 @universe_conc_mpsc_seg_push(ptr, ptr)
declare i32 @universe_conc_mpsc_seg_pop(ptr, ptr)
declare i64 @universe_conc_mpsc_seg_count(ptr)
declare void @universe_conc_mpsc_seg_destroy(ptr)

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

@mpsc.samp = internal global [16 x double] zeroinitializer, align 8

@g.q = internal global ptr null, align 8
@g.ids = internal global [4 x i64] [i64 0, i64 1, i64 2, i64 3], align 8
@g.mode = internal global i32 0, align 4       ; 0 = ring, 1 = seg
@g.per = internal global i64 100000, align 8   ; items per producer

@m.rsingle = private unnamed_addr constant [22 x i8] c"ring single-thread   \00"
@m.rcap    = private unnamed_addr constant [14 x i8] c"ring capacity\00"
@m.ssingle = private unnamed_addr constant [22 x i8] c"seg single-thread    \00"
@m.spawn   = private unnamed_addr constant [15 x i8] c"producer spawn\00"
@m.rviol   = private unnamed_addr constant [24 x i8] c"ring exactly-once/FIFO \00"
@m.rtot    = private unnamed_addr constant [24 x i8] c"ring total conserved   \00"
@m.rdrain  = private unnamed_addr constant [24 x i8] c"ring empty at finish   \00"
@m.sviol   = private unnamed_addr constant [24 x i8] c"seg  exactly-once/FIFO \00"
@m.stot    = private unnamed_addr constant [24 x i8] c"seg  total conserved   \00"
@m.sdrain  = private unnamed_addr constant [24 x i8] c"seg  empty at finish   \00"
@m.lring   = private unnamed_addr constant [5 x i8] c"ring\00"
@m.lseg    = private unnamed_addr constant [4 x i8] c"seg\00"

; ---- single-thread: bounded ring -------------------------------------------
define internal void @test_ring_single() {
entry:
  ; requested 4 rounds up to the module minimum capacity (8).
  %rb = call ptr @universe_conc_mpsc_ring_create(i64 4, i64 8)
  %cap = call i64 @universe_conc_mpsc_ring_capacity(ptr %rb)
  %capok = icmp eq i64 %cap, 8
  call void @ut_check(i1 %capok, ptr @m.rcap)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %viol.f = phi i64 [ 0, %entry ], [ %viol.f2, %fill ]
  %fv = mul i64 %i, 10
  store i64 %fv, ptr %v, align 8
  %pr = call i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %v)
  %prbad = icmp ne i32 %pr, 0
  %fz = zext i1 %prbad to i64
  %viol.f2 = add i64 %viol.f, %fz
  %i.n = add i64 %i, 1
  %fmore = icmp ult i64 %i.n, %cap
  br i1 %fmore, label %fill, label %overfull

overfull:
  ; one more push must be FULL (6)
  store i64 9999, ptr %v, align 8
  %ef = call i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %v)
  %ef.ok = icmp eq i32 %ef, 6
  %ef.z = zext i1 %ef.ok to i64        ; 1 when correct
  br label %drain

drain:
  %j = phi i64 [ 0, %overfull ], [ %j.n, %drain ]
  %viol.d = phi i64 [ 0, %overfull ], [ %viol.d2, %drain ]
  %pc = call i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %v)
  %got = load i64, ptr %v, align 8
  %want = mul i64 %j, 10
  %pcbad = icmp ne i32 %pc, 0
  %valbad = icmp ne i64 %got, %want
  %dbad = or i1 %pcbad, %valbad
  %dz = zext i1 %dbad to i64
  %viol.d2 = add i64 %viol.d, %dz
  %j.n = add i64 %j, 1
  %dmore = icmp ult i64 %j.n, %cap
  br i1 %dmore, label %drain, label %empty

empty:
  %pe = call i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %v)
  %pe.ok = icmp eq i32 %pe, 4
  %pe.z = zext i1 %pe.ok to i64
  ; wrap: push 3, pop 3 FIFO
  store i64 111, ptr %v, align 8
  %w1 = call i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %v)
  store i64 222, ptr %v, align 8
  %w2 = call i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %v)
  store i64 333, ptr %v, align 8
  %w3 = call i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %v)
  %q1 = call i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %v)
  %wg1 = load i64, ptr %v, align 8
  %q2 = call i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %v)
  %wg2 = load i64, ptr %v, align 8
  %q3 = call i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %v)
  %wg3 = load i64, ptr %v, align 8
  %wok1 = icmp eq i64 %wg1, 111
  %wok2 = icmp eq i64 %wg2, 222
  %wok3 = icmp eq i64 %wg3, 333
  ; aggregate all
  %agg0 = icmp eq i64 %viol.f2, 0
  %agg1 = icmp eq i64 %viol.d2, 0
  %agg2 = icmp eq i64 %ef.z, 1
  %agg3 = icmp eq i64 %pe.z, 1
  %x0 = and i1 %agg0, %agg1
  %x1 = and i1 %x0, %agg2
  %x2 = and i1 %x1, %agg3
  %x3 = and i1 %x2, %wok1
  %x4 = and i1 %x3, %wok2
  %x5 = and i1 %x4, %wok3
  call void @ut_check(i1 %x5, ptr @m.rsingle)
  call void @universe_conc_mpsc_ring_destroy(ptr %rb)
  ret void
}

; ---- single-thread: unbounded seg (crosses chunks + pool reuse) ------------
define internal void @test_seg_single() {
entry:
  %q = call ptr @universe_conc_mpsc_seg_create(i64 8)
  %v = alloca i64, align 8
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  ; round 1: push 2048 (chunks 0,1), pop 2048 -> recycles chunk 0 and 1
  call void @seg_pushpop_check(ptr %q, i64 0, i64 2048, ptr %viol)
  ; round 2: push 2048 more (chunks 2,3 reuse pooled), pop 2048
  call void @seg_pushpop_check(ptr %q, i64 2048, i64 2048, ptr %viol)
  ; now EMPTY
  %e = call i32 @universe_conc_mpsc_seg_pop(ptr %q, ptr %v)
  %eok = icmp eq i32 %e, 4
  %vfin = load i64, ptr %viol, align 8
  %vok = icmp eq i64 %vfin, 0
  %both = and i1 %eok, %vok
  call void @ut_check(i1 %both, ptr @m.ssingle)
  call void @universe_conc_mpsc_seg_destroy(ptr %q)
  ret void
}

; push %n items valued %start.. then pop %n, verifying FIFO. Adds to *viol.
define internal void @seg_pushpop_check(ptr %q, i64 %start, i64 %n, ptr %viol) {
entry:
  %v = alloca i64, align 8
  %end = add i64 %start, %n
  br label %push

push:
  %i = phi i64 [ %start, %entry ], [ %i.n, %push ]
  store i64 %i, ptr %v, align 8
  %pr = call i32 @universe_conc_mpsc_seg_push(ptr %q, ptr %v)
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %end
  br i1 %more, label %push, label %pop

pop:
  %j = phi i64 [ %start, %push ], [ %j.n, %pop ]
  %vc = phi i64 [ 0, %push ], [ %vc.n, %pop ]
  %rc = call i32 @universe_conc_mpsc_seg_pop(ptr %q, ptr %v)
  %got = load i64, ptr %v, align 8
  %badrc = icmp ne i32 %rc, 0
  %badval = icmp ne i64 %got, %j
  %bad = or i1 %badrc, %badval
  %binc = zext i1 %bad to i64
  %vc.n = add i64 %vc, %binc
  %j.n = add i64 %j, 1
  %more2 = icmp ult i64 %j.n, %end
  br i1 %more2, label %pop, label %fin

fin:
  %old = load i64, ptr %viol, align 8
  %new = add i64 %old, %vc.n
  store i64 %new, ptr %viol, align 8
  ret void
}

; ---- producer thread (both variants dispatch on @g.mode) -------------------
define internal ptr @producer(ptr %arg) {
entry:
  %pid = load i64, ptr %arg, align 8
  %q = load ptr, ptr @g.q, align 8
  %mode = load i32, ptr @g.mode, align 4
  %per = load i64, ptr @g.per, align 8
  %hi = shl i64 %pid, 32
  %v = alloca i64, align 8
  %isseg = icmp eq i32 %mode, 1
  br label %next

next:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  %val = or i64 %hi, %i
  store i64 %val, ptr %v, align 8
  br label %try

try:
  br i1 %isseg, label %try.seg, label %try.ring

try.ring:
  %rcr = call i32 @universe_conc_mpsc_ring_push(ptr %q, ptr %v)
  br label %chk

try.seg:
  %rcs = call i32 @universe_conc_mpsc_seg_push(ptr %q, ptr %v)
  br label %chk

chk:
  %rc = phi i32 [ %rcr, %try.ring ], [ %rcs, %try.seg ]
  %fail = icmp ne i32 %rc, 0
  br i1 %fail, label %yield, label %pushed

yield:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %per
  br i1 %more, label %next, label %done

done:
  ret ptr null
}

; ---- shared consumer: drain %total items, verify exactly-once + FIFO -------
; returns violation count; writes final per-nothing. Consumes from @g.q.
define internal i64 @consume_verify(i64 %total) {
entry:
  %mode = load i32, ptr @g.mode, align 4
  %q = load ptr, ptr @g.q, align 8
  %isseg = icmp eq i32 %mode, 1
  %exp = alloca [4 x i64], align 8
  store [4 x i64] zeroinitializer, ptr %exp, align 8
  %v = alloca i64, align 8
  br label %loop

loop:
  %done = phi i64 [ 0, %entry ], [ %done.n, %join ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %join ]
  br label %try

try:
  br i1 %isseg, label %try.seg, label %try.ring

try.ring:
  %rcr = call i32 @universe_conc_mpsc_ring_pop(ptr %q, ptr %v)
  br label %chk

try.seg:
  %rcs = call i32 @universe_conc_mpsc_seg_pop(ptr %q, ptr %v)
  br label %chk

chk:
  %rc = phi i32 [ %rcr, %try.ring ], [ %rcs, %try.seg ]
  %empty = icmp ne i32 %rc, 0
  br i1 %empty, label %yield, label %got

yield:
  %y = call i32 @sched_yield()
  br label %try

got:
  %val = load i64, ptr %v, align 8
  %pid = lshr i64 %val, 32
  %seq = and i64 %val, 4294967295
  %pid.ok = icmp ult i64 %pid, 4
  br i1 %pid.ok, label %decode, label %badpid

badpid:
  %viol.bp = add i64 %viol, 1
  br label %join

decode:
  %ep = getelementptr inbounds [4 x i64], ptr %exp, i64 0, i64 %pid
  %want = load i64, ptr %ep, align 8
  %fifo.bad = icmp ne i64 %seq, %want
  %vinc = zext i1 %fifo.bad to i64
  %viol.d = add i64 %viol, %vinc
  %want.n = add i64 %want, 1
  store i64 %want.n, ptr %ep, align 8
  br label %join

join:
  %viol.n = phi i64 [ %viol.bp, %badpid ], [ %viol.d, %decode ]
  %done.n = add i64 %done, 1
  %more = icmp ult i64 %done.n, %total
  br i1 %more, label %loop, label %fin

fin:
  ; verify each producer delivered exactly %per items
  %per = load i64, ptr @g.per, align 8
  %e0 = getelementptr inbounds [4 x i64], ptr %exp, i64 0, i64 0
  %c0 = load i64, ptr %e0, align 8
  %e1 = getelementptr inbounds [4 x i64], ptr %exp, i64 0, i64 1
  %c1 = load i64, ptr %e1, align 8
  %e2 = getelementptr inbounds [4 x i64], ptr %exp, i64 0, i64 2
  %c2 = load i64, ptr %e2, align 8
  %e3 = getelementptr inbounds [4 x i64], ptr %exp, i64 0, i64 3
  %c3 = load i64, ptr %e3, align 8
  %b0 = icmp ne i64 %c0, %per
  %b1 = icmp ne i64 %c1, %per
  %b2 = icmp ne i64 %c2, %per
  %b3 = icmp ne i64 %c3, %per
  %z0 = zext i1 %b0 to i64
  %z1 = zext i1 %b1 to i64
  %z2 = zext i1 %b2 to i64
  %z3 = zext i1 %b3 to i64
  %s0 = add i64 %viol.n, %z0
  %s1 = add i64 %s0, %z1
  %s2 = add i64 %s1, %z2
  %s3 = add i64 %s2, %z3
  ret i64 %s3
}

; ---- stress driver for one variant -----------------------------------------
; %mode 0=ring 1=seg. spawns 4 producers, consumes all, joins, checks.
define internal void @run_stress(i32 %mode, ptr %mviol, ptr %mtot, ptr %mdrain) {
entry:
  store i32 %mode, ptr @g.mode, align 4
  %per = load i64, ptr @g.per, align 8
  %total = mul i64 %per, 4
  ; create queue
  %isseg = icmp eq i32 %mode, 1
  br i1 %isseg, label %mk.seg, label %mk.ring

mk.ring:
  %rq = call ptr @universe_conc_mpsc_ring_create(i64 1024, i64 8)
  br label %store.q

mk.seg:
  %sq = call ptr @universe_conc_mpsc_seg_create(i64 8)
  br label %store.q

store.q:
  %q = phi ptr [ %rq, %mk.ring ], [ %sq, %mk.seg ]
  store ptr %q, ptr @g.q, align 8
  %tids = alloca [4 x i64], align 8
  br label %spawn

spawn:
  %k = phi i64 [ 0, %store.q ], [ %k.n, %spawn ]
  %tidp = getelementptr inbounds [4 x i64], ptr %tids, i64 0, i64 %k
  %idp = getelementptr inbounds [4 x i64], ptr @g.ids, i64 0, i64 %k
  %cr = call i32 @pthread_create(ptr %tidp, ptr null, ptr @producer, ptr %idp)
  %crw = zext i32 %cr to i64
  call void @ut_check_eq(i64 %crw, i64 0, ptr @m.spawn)
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 4
  br i1 %more, label %spawn, label %consume

consume:
  %viol = call i64 @consume_verify(i64 %total)
  call void @ut_check_eq(i64 %viol, i64 0, ptr %mviol)
  br label %joins

joins:
  %j = phi i64 [ 0, %consume ], [ %j.n, %joins ]
  %tjp = getelementptr inbounds [4 x i64], ptr %tids, i64 0, i64 %j
  %tid = load i64, ptr %tjp, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 4
  br i1 %jmore, label %joins, label %check

check:
  br i1 %isseg, label %cnt.seg, label %cnt.ring

cnt.ring:
  %lr = call i64 @universe_conc_mpsc_ring_count(ptr %q)
  br label %fin

cnt.seg:
  %ls = call i64 @universe_conc_mpsc_seg_count(ptr %q)
  br label %fin

fin:
  %left = phi i64 [ %lr, %cnt.ring ], [ %ls, %cnt.seg ]
  call void @ut_check_eq(i64 %left, i64 0, ptr %mdrain)
  ; total conserved: all producers reported per each => consume_verify checked
  call void @ut_check_eq(i64 %viol, i64 0, ptr %mtot)
  br i1 %isseg, label %d.seg, label %d.ring

d.ring:
  call void @universe_conc_mpsc_ring_destroy(ptr %q)
  br label %ret

d.seg:
  call void @universe_conc_mpsc_seg_destroy(ptr %q)
  br label %ret

ret:
  ret void
}

; ---- bench: aggregate producer throughput under contention -----------------
; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) of the
; 4-producer parallel batch. 17 reps: rep 0 warm-up (discarded), reps 1..16
; recorded. Each rep recreates the queue. ops_per_rep = 4*per items.
define internal void @run_bench(i32 %mode, ptr %label) {
entry:
  store i32 %mode, ptr @g.mode, align 4
  %per = load i64, ptr @g.per, align 8
  %total = mul i64 %per, 4
  %isseg = icmp eq i32 %mode, 1
  %tids = alloca [4 x i64], align 8
  br label %rep

rep:
  %r = phi i64 [ 0, %entry ], [ %r.n, %rep.next ]
  br i1 %isseg, label %mk.seg, label %mk.ring

mk.ring:
  %rq = call ptr @universe_conc_mpsc_ring_create(i64 1024, i64 8)
  br label %store.q

mk.seg:
  %sq = call ptr @universe_conc_mpsc_seg_create(i64 8)
  br label %store.q

store.q:
  %q = phi ptr [ %rq, %mk.ring ], [ %sq, %mk.seg ]
  store ptr %q, ptr @g.q, align 8
  %t0 = call double @ut_now_sec()
  br label %spawn

spawn:
  %k = phi i64 [ 0, %store.q ], [ %k.n, %spawn ]
  %tidp = getelementptr inbounds [4 x i64], ptr %tids, i64 0, i64 %k
  %idp = getelementptr inbounds [4 x i64], ptr @g.ids, i64 0, i64 %k
  %cr = call i32 @pthread_create(ptr %tidp, ptr null, ptr @producer, ptr %idp)
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 4
  br i1 %more, label %spawn, label %consume

consume:
  %viol = call i64 @consume_verify(i64 %total)
  br label %joins

joins:
  %j = phi i64 [ 0, %consume ], [ %j.n, %joins ]
  %tjp = getelementptr inbounds [4 x i64], ptr %tids, i64 0, i64 %j
  %tid = load i64, ptr %tjp, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 4
  br i1 %jmore, label %joins, label %rep.mid

rep.mid:
  %t1 = call double @ut_now_sec()
  br i1 %isseg, label %d.seg, label %d.ring

d.ring:
  call void @universe_conc_mpsc_ring_destroy(ptr %q)
  br label %rec

d.seg:
  call void @universe_conc_mpsc_seg_destroy(ptr %q)
  br label %rec

rec:
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %r, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %idx = sub i64 %r, 1
  %sp = getelementptr inbounds [16 x double], ptr @mpsc.samp, i64 0, i64 %idx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %r.n = add nuw nsw i64 %r, 1
  %rmore = icmp ult i64 %r.n, 17
  br i1 %rmore, label %rep, label %done

done:
  call void @ut_report_dist(ptr @mpsc.samp, i64 16, i64 %total, ptr %label)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %do.test

do.bench:
  call void @run_bench(i32 0, ptr @m.lring)
  call void @run_bench(i32 1, ptr @m.lseg)
  br label %fin

do.test:
  call void @test_ring_single()
  call void @test_seg_single()
  call void @run_stress(i32 0, ptr @m.rviol, ptr @m.rtot, ptr @m.rdrain)
  call void @run_stress(i32 1, ptr @m.sviol, ptr @m.stot, ptr @m.sdrain)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
