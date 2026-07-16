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

; Tests for universe_conc_scounter (striped counter).
;   * single-thread: rounding, inc/sum/reset/shards, null guards.
;   * CONSERVATION stress: 4 threads x 1,000,000 incs each => sum == 4,000,000
;     EXACTLY (no lost updates). Looped 10x at -O0 and -O3 on the weak memory
;     model. A single lost atomicrmw would drop the sum below 4,000,000.
;   * --bench: striped counter vs a single shared atomic under 8-thread
;     contention — shows the sharding win.

declare ptr  @universe_conc_scounter_create(i64)
declare void @universe_conc_scounter_inc(ptr, i64, i64)
declare i64  @universe_conc_scounter_sum(ptr)
declare void @universe_conc_scounter_reset(ptr)
declare i64  @universe_conc_scounter_shards(ptr)
declare void @universe_conc_scounter_destroy(ptr)

declare i32    @pthread_create(ptr, ptr, ptr, ptr)
declare i32    @pthread_join(i64, ptr)
declare i32    @printf(ptr, ...)
declare void   @ut_check(i1, ptr)
declare void   @ut_check_eq(i64, i64, ptr)
declare i32    @ut_summary()
declare i1     @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void   @ut_report_dist(ptr, i64, i64, ptr)

@striped.samp = internal global [16 x double] zeroinitializer, align 8
@single.samp  = internal global [16 x double] zeroinitializer, align 8
@lbl.striped  = private unnamed_addr constant [21 x i8] c"striped 8x scounter \00"
@lbl.single   = private unnamed_addr constant [21 x i8] c"single 8x atomic add\00"

@gc.c      = internal global ptr null, align 8
@gc.n      = internal global i64 0, align 8
@gc.single = internal global i64 0, align 8
@gc.tids   = internal global [8 x i64] zeroinitializer, align 8

@m.single  = private unnamed_addr constant [24 x i8] c"scounter single-thread\0A\00"
@m.round   = private unnamed_addr constant [26 x i8] c"scounter round-to-pow2 ok\00"
@m.null    = private unnamed_addr constant [21 x i8] c"scounter null guards\00"
@m.stress  = private unnamed_addr constant [30 x i8] c"scounter 4x1M conservation ok\00"

; ===========================================================================
; single-thread correctness
; ===========================================================================
define internal void @test_single() {
entry:
  ; rounding: create(3) -> 4 shards; create(0) -> 64
  %c3 = call ptr @universe_conc_scounter_create(i64 3)
  %s3 = call i64 @universe_conc_scounter_shards(ptr %c3)
  %r3 = icmp eq i64 %s3, 4
  %c0 = call ptr @universe_conc_scounter_create(i64 0)
  %s0 = call i64 @universe_conc_scounter_shards(ptr %c0)
  %r0 = icmp eq i64 %s0, 64
  %rr = and i1 %r3, %r0
  call void @ut_check(i1 %rr, ptr @m.round)
  call void @universe_conc_scounter_destroy(ptr %c3)
  call void @universe_conc_scounter_destroy(ptr %c0)

  ; inc across shards, sum, reset
  %c = call ptr @universe_conc_scounter_create(i64 16)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  ; put i into shard (i & 15); delta = i+1  => total = sum_{i=0..99}(i+1) = 5050
  %d = add nuw i64 %i, 1
  call void @universe_conc_scounter_inc(ptr %c, i64 %i, i64 %d)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 100
  br i1 %more, label %loop, label %check

check:
  %sum = call i64 @universe_conc_scounter_sum(ptr %c)
  %sum.ok = icmp eq i64 %sum, 5050
  call void @universe_conc_scounter_reset(ptr %c)
  %sum2 = call i64 @universe_conc_scounter_sum(ptr %c)
  %z.ok = icmp eq i64 %sum2, 0
  %both = and i1 %sum.ok, %z.ok
  call void @ut_check(i1 %both, ptr @m.single)
  call void @universe_conc_scounter_destroy(ptr %c)
  ret void
}

define internal void @test_null() {
entry:
  ; all must be safe no-ops / zero returns
  call void @universe_conc_scounter_inc(ptr null, i64 0, i64 1)
  %s = call i64 @universe_conc_scounter_sum(ptr null)
  %sh = call i64 @universe_conc_scounter_shards(ptr null)
  call void @universe_conc_scounter_reset(ptr null)
  call void @universe_conc_scounter_destroy(ptr null)
  %a = icmp eq i64 %s, 0
  %b = icmp eq i64 %sh, 0
  %ok = and i1 %a, %b
  call void @ut_check(i1 %ok, ptr @m.null)
  ret void
}

; ===========================================================================
; stress: worker increments its own shard n times, delta 1
; ===========================================================================
define internal ptr @sc_worker(ptr %arg) {
entry:
  %id = ptrtoint ptr %arg to i64
  %c = load ptr, ptr @gc.c, align 8
  %n = load i64, ptr @gc.n, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  call void @universe_conc_scounter_inc(ptr %c, i64 %id, i64 1)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret ptr null
}

; reference worker: all threads hammer ONE shared atomic (single-atomic bench)
define internal ptr @sc_worker_single(ptr %arg) {
entry:
  %n = load i64, ptr @gc.n, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %o = atomicrmw add ptr @gc.single, i64 1 monotonic, align 8
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
  %slot = getelementptr inbounds [8 x i64], ptr @gc.tids, i64 0, i64 %i
  %arg = inttoptr i64 %i to ptr
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr %fn, ptr %arg)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nt
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @gc.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, %nt
  br i1 %jmore, label %join, label %done

done:
  ret void
}

; run 10 rounds of 4x1,000,000; return total violations (sum != 4,000,000)
define internal i64 @run_stress() {
entry:
  %c = call ptr @universe_conc_scounter_create(i64 64)
  store ptr %c, ptr @gc.c, align 8
  store i64 1000000, ptr @gc.n, align 8
  br label %round

round:
  %r = phi i64 [ 0, %entry ], [ %r.n, %round ]
  %v = phi i64 [ 0, %entry ], [ %v.n, %round ]
  call void @universe_conc_scounter_reset(ptr %c)
  call void @spawn_join(i64 4, ptr @sc_worker)
  %sum = call i64 @universe_conc_scounter_sum(ptr %c)
  %bad = icmp ne i64 %sum, 4000000
  %inc = zext i1 %bad to i64
  %v.n = add i64 %v, %inc
  %r.n = add nuw i64 %r, 1
  %more = icmp ult i64 %r.n, 10
  br i1 %more, label %round, label %done

done:
  call void @universe_conc_scounter_destroy(ptr %c)
  ret i64 %v.n
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) for the
; striped (disjoint, no contention) vs single-atomic (contended) 8-thread
; batches. 17 reps: rep 0 warm-up (discarded), reps 1..16 recorded. Each rep
; resets the counter. ops_per_rep = 8 * 2M = 16M increments.
define internal void @bench() {
entry:
  store i64 2000000, ptr @gc.n, align 8
  %c = call ptr @universe_conc_scounter_create(i64 8)
  store ptr %c, ptr @gc.c, align 8
  br label %st.rep

; striped: 8 shards, 8 threads, disjoint => no contention
st.rep:
  %sr = phi i64 [ 0, %entry ], [ %sr.n, %st.next ]
  call void @universe_conc_scounter_reset(ptr %c)
  %st0 = call double @ut_now_sec()
  call void @spawn_join(i64 8, ptr @sc_worker)
  %st1 = call double @ut_now_sec()
  %sdt = fsub double %st1, %st0
  %swarm = icmp eq i64 %sr, 0
  br i1 %swarm, label %st.next, label %st.store

st.store:
  %sidx = sub i64 %sr, 1
  %ssp = getelementptr inbounds [16 x double], ptr @striped.samp, i64 0, i64 %sidx
  store double %sdt, ptr %ssp, align 8
  br label %st.next

st.next:
  %sr.n = add nuw nsw i64 %sr, 1
  %srmore = icmp ult i64 %sr.n, 17
  br i1 %srmore, label %st.rep, label %st.done

st.done:
  call void @ut_report_dist(ptr @striped.samp, i64 16, i64 16000000, ptr @lbl.striped)
  call void @universe_conc_scounter_destroy(ptr %c)
  br label %sg.rep

; single shared atomic, 8 threads contending
sg.rep:
  %gr = phi i64 [ 0, %st.done ], [ %gr.n, %sg.next ]
  store i64 0, ptr @gc.single, align 8
  %gt0 = call double @ut_now_sec()
  call void @spawn_join(i64 8, ptr @sc_worker_single)
  %gt1 = call double @ut_now_sec()
  %gdt = fsub double %gt1, %gt0
  %gwarm = icmp eq i64 %gr, 0
  br i1 %gwarm, label %sg.next, label %sg.store

sg.store:
  %gidx = sub i64 %gr, 1
  %gsp = getelementptr inbounds [16 x double], ptr @single.samp, i64 0, i64 %gidx
  store double %gdt, ptr %gsp, align 8
  br label %sg.next

sg.next:
  %gr.n = add nuw nsw i64 %gr, 1
  %grmore = icmp ult i64 %gr.n, 17
  br i1 %grmore, label %sg.rep, label %sg.done

sg.done:
  call void @ut_report_dist(ptr @single.samp, i64 16, i64 16000000, ptr @lbl.single)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single()
  call void @test_null()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %dobench, label %dostress

dobench:
  call void @bench()
  br label %fin

dostress:
  %v = call i64 @run_stress()
  call void @ut_check_eq(i64 %v, i64 0, ptr @m.stress)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
