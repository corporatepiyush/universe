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

; Tests for universe_conc_shardmap (striped-lock concurrent hash map).
;   * single-thread: put/get/overwrite/delete/len/missing, many shards + in-
;     shard collisions.
;   * null-arg guards.
;   * pthread stress (the gate): 4 threads, each on a DISJOINT key range (put,
;     get-verify, delete-odd) AND a shared OVERLAPPING range (all write same
;     key/value). After join: every surviving disjoint key present with the
;     right value; deleted keys absent; every shared key present; len exact.
;     No lost updates on disjoint keys. Looped 10x at -O0 and -O3.
;   * --bench: 8-thread throughput on a 256-shard map vs a 1-shard map (a
;     single global lock — same code path, one shard) on a contended key set.

declare ptr  @universe_conc_shardmap_create(i64, i64)
declare i32  @universe_conc_shardmap_put(ptr, ptr, i64, i64)
declare i32  @universe_conc_shardmap_get(ptr, ptr, i64, ptr)
declare i32  @universe_conc_shardmap_delete(ptr, ptr, i64)
declare i64  @universe_conc_shardmap_len(ptr)
declare i64  @universe_conc_shardmap_shards(ptr)
declare void @universe_conc_shardmap_destroy(ptr)

declare i32    @pthread_create(ptr, ptr, ptr, ptr)
declare i32    @pthread_join(i64, ptr)
declare i32    @printf(ptr, ...)
declare void   @ut_check(i1, ptr)
declare void   @ut_check_eq(i64, i64, ptr)
declare i32    @ut_summary()
declare i1     @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void   @ut_report_dist(ptr, i64, i64, ptr)

@shard256.samp = internal global [16 x double] zeroinitializer, align 8
@shard1.samp   = internal global [16 x double] zeroinitializer, align 8
@lbl.shard256  = private unnamed_addr constant [22 x i8] c"256-shard 8x put+get \00"
@lbl.shard1    = private unnamed_addr constant [22 x i8] c"1-shard 8x put+get   \00"

@gm.m    = internal global ptr null, align 8
@gm.err  = internal global i64 0, align 8
@gm.tids = internal global [8 x i64] zeroinitializer, align 8

; stress config
;   PER    = 50000 disjoint keys per thread
;   SHARED = 5000  overlapping keys
;   SHARED_BASE = 900000000 (disjoint from thread ranges t*1000000+j)
;   surviving disjoint = 4 * (PER - PER/2) = 100000 ; + SHARED => len 105000

@m.single = private unnamed_addr constant [24 x i8] c"shardmap single-thread\0A\00"
@m.null   = private unnamed_addr constant [21 x i8] c"shardmap null guards\00"
@m.stress = private unnamed_addr constant [28 x i8] c"shardmap 4-thread stress ok\00"

; ===========================================================================
; single-thread correctness
; ===========================================================================
define internal void @test_single() {
entry:
  %m = call ptr @universe_conc_shardmap_create(i64 64, i64 16)
  %kb = alloca i64, align 8
  %out = alloca i64, align 8
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  br label %put

; put 2000 keys: key bytes = i (8 bytes), val = i*7+1
put:
  %i = phi i64 [ 0, %entry ], [ %i.n, %put ]
  store i64 %i, ptr %kb, align 8
  %v = mul nuw i64 %i, 7
  %val = add nuw i64 %v, 1
  %rc = call i32 @universe_conc_shardmap_put(ptr %m, ptr %kb, i64 8, i64 %val)
  %bad = icmp ne i32 %rc, 0
  %vl0 = load i64, ptr %viol, align 8
  %vi0 = zext i1 %bad to i64
  %vn0 = add i64 %vl0, %vi0
  store i64 %vn0, ptr %viol, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 2000
  br i1 %more, label %put, label %getp

; get all back, verify value
getp:
  %j = phi i64 [ 0, %put ], [ %j.n, %getp ]
  store i64 %j, ptr %kb, align 8
  %grc = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %gv = load i64, ptr %out, align 8
  %ev = mul nuw i64 %j, 7
  %ev1 = add nuw i64 %ev, 1
  %grc.bad = icmp ne i32 %grc, 0
  %gv.bad = icmp ne i64 %gv, %ev1
  %g.bad = or i1 %grc.bad, %gv.bad
  %vl1 = load i64, ptr %viol, align 8
  %vi1 = zext i1 %g.bad to i64
  %vn1 = add i64 %vl1, %vi1
  store i64 %vn1, ptr %viol, align 8
  %j.n = add nuw i64 %j, 1
  %gmore = icmp ult i64 %j.n, 2000
  br i1 %gmore, label %getp, label %lenchk

lenchk:
  %len1 = call i64 @universe_conc_shardmap_len(ptr %m)
  %len1.bad = icmp ne i64 %len1, 2000
  %vl2 = load i64, ptr %viol, align 8
  %vi2 = zext i1 %len1.bad to i64
  %vn2 = add i64 %vl2, %vi2
  store i64 %vn2, ptr %viol, align 8
  ; overwrite key 5 with new value 424242
  store i64 5, ptr %kb, align 8
  %orc = call i32 @universe_conc_shardmap_put(ptr %m, ptr %kb, i64 8, i64 424242)
  %orc2 = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %ov = load i64, ptr %out, align 8
  %ov.ok = icmp eq i64 %ov, 424242
  %len2 = call i64 @universe_conc_shardmap_len(ptr %m)
  %len2.ok = icmp eq i64 %len2, 2000
  %ow.ok = and i1 %ov.ok, %len2.ok
  %ow.bad = xor i1 %ow.ok, true
  %vl3 = load i64, ptr %viol, align 8
  %vi3 = zext i1 %ow.bad to i64
  %vn3 = add i64 %vl3, %vi3
  store i64 %vn3, ptr %viol, align 8
  br label %del

; delete keys 0..999
del:
  %d = phi i64 [ 0, %lenchk ], [ %d.n, %del ]
  store i64 %d, ptr %kb, align 8
  %drc = call i32 @universe_conc_shardmap_delete(ptr %m, ptr %kb, i64 8)
  %drc.bad = icmp ne i32 %drc, 0
  %vl4 = load i64, ptr %viol, align 8
  %vi4 = zext i1 %drc.bad to i64
  %vn4 = add i64 %vl4, %vi4
  store i64 %vn4, ptr %viol, align 8
  %d.n = add nuw i64 %d, 1
  %dmore = icmp ult i64 %d.n, 1000
  br i1 %dmore, label %del, label %postdel

postdel:
  %len3 = call i64 @universe_conc_shardmap_len(ptr %m)
  %len3.bad = icmp ne i64 %len3, 1000
  %vl5 = load i64, ptr %viol, align 8
  %vi5 = zext i1 %len3.bad to i64
  %vn5 = add i64 %vl5, %vi5
  store i64 %vn5, ptr %viol, align 8
  ; deleted key 0 must now be NOT_FOUND (5)
  store i64 0, ptr %kb, align 8
  %mrc = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %mrc.bad = icmp ne i32 %mrc, 5
  ; a never-inserted key must be NOT_FOUND
  store i64 987654321, ptr %kb, align 8
  %nrc = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %nrc.bad = icmp ne i32 %nrc, 5
  %miss.bad = or i1 %mrc.bad, %nrc.bad
  %vl6 = load i64, ptr %viol, align 8
  %vi6 = zext i1 %miss.bad to i64
  %vn6 = add i64 %vl6, %vi6
  store i64 %vn6, ptr %viol, align 8
  ; surviving key 1500 still present, correct value
  store i64 1500, ptr %kb, align 8
  %src = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %sv = load i64, ptr %out, align 8
  %sv.bad = icmp ne i64 %sv, 10501         ; 1500*7+1
  %src.bad = icmp ne i32 %src, 0
  %surv.bad = or i1 %src.bad, %sv.bad
  %vl7 = load i64, ptr %viol, align 8
  %vi7 = zext i1 %surv.bad to i64
  %vn7 = add i64 %vl7, %vi7
  store i64 %vn7, ptr %viol, align 8
  %vfin = load i64, ptr %viol, align 8
  %pass = icmp eq i64 %vfin, 0
  call void @ut_check(i1 %pass, ptr @m.single)
  call void @universe_conc_shardmap_destroy(ptr %m)
  ret void
}

define internal void @test_null() {
entry:
  %m = call ptr @universe_conc_shardmap_create(i64 8, i64 16)
  %kb = alloca i64, align 8
  %out = alloca i64, align 8
  store i64 1, ptr %kb, align 8
  %e1 = call i32 @universe_conc_shardmap_put(ptr null, ptr %kb, i64 8, i64 1)
  %e2 = call i32 @universe_conc_shardmap_put(ptr %m, ptr null, i64 8, i64 1)
  %e3 = call i32 @universe_conc_shardmap_get(ptr null, ptr %kb, i64 8, ptr %out)
  %e4 = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr null)
  %e5 = call i32 @universe_conc_shardmap_delete(ptr null, ptr %kb, i64 8)
  %o1 = icmp eq i32 %e1, 1
  %o2 = icmp eq i32 %e2, 1
  %o3 = icmp eq i32 %e3, 1
  %o4 = icmp eq i32 %e4, 1
  %o5 = icmp eq i32 %e5, 1
  %a1 = and i1 %o1, %o2
  %a2 = and i1 %a1, %o3
  %a3 = and i1 %a2, %o4
  %a4 = and i1 %a3, %o5
  call void @ut_check(i1 %a4, ptr @m.null)
  call void @universe_conc_shardmap_destroy(ptr %m)
  ret void
}

; ===========================================================================
; stress worker: disjoint range [t*1M, t*1M+50000) + shared [900M, 900M+5000)
; ===========================================================================
define internal ptr @map_worker(ptr %arg) {
entry:
  %t = ptrtoint ptr %arg to i64
  %m = load ptr, ptr @gm.m, align 8
  %kb = alloca i64, align 8
  %out = alloca i64, align 8
  %base = mul nuw i64 %t, 1000000
  br label %p1

; phase 1: put disjoint keys, val = key*2+1
p1:
  %j = phi i64 [ 0, %entry ], [ %j.n, %p1 ]
  %k1 = add nuw i64 %base, %j
  store i64 %k1, ptr %kb, align 8
  %v1a = mul nuw i64 %k1, 2
  %v1 = add nuw i64 %v1a, 1
  %r1 = call i32 @universe_conc_shardmap_put(ptr %m, ptr %kb, i64 8, i64 %v1)
  %b1 = icmp ne i32 %r1, 0
  %e1 = zext i1 %b1 to i64
  %o1 = atomicrmw add ptr @gm.err, i64 %e1 monotonic, align 8
  %j.n = add nuw i64 %j, 1
  %m1 = icmp ult i64 %j.n, 50000
  br i1 %m1, label %p1, label %p2

; phase 2: get-verify own disjoint keys (stable; no other thread touches them)
p2:
  %j2 = phi i64 [ 0, %p1 ], [ %j2.n, %p2 ]
  %k2 = add nuw i64 %base, %j2
  store i64 %k2, ptr %kb, align 8
  %r2 = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %gv = load i64, ptr %out, align 8
  %ev2a = mul nuw i64 %k2, 2
  %ev2 = add nuw i64 %ev2a, 1
  %rb = icmp ne i32 %r2, 0
  %vb = icmp ne i64 %gv, %ev2
  %b2 = or i1 %rb, %vb
  %e2 = zext i1 %b2 to i64
  %o2 = atomicrmw add ptr @gm.err, i64 %e2 monotonic, align 8
  %j2.n = add nuw i64 %j2, 1
  %m2 = icmp ult i64 %j2.n, 50000
  br i1 %m2, label %p2, label %p3

; phase 3: delete odd-index disjoint keys
p3:
  %j3 = phi i64 [ 0, %p2 ], [ %j3.n, %p3.cont ]
  %odd = and i64 %j3, 1
  %isodd = icmp eq i64 %odd, 1
  br i1 %isodd, label %p3.del, label %p3.cont

p3.del:
  %k3 = add nuw i64 %base, %j3
  store i64 %k3, ptr %kb, align 8
  %r3 = call i32 @universe_conc_shardmap_delete(ptr %m, ptr %kb, i64 8)
  %b3 = icmp ne i32 %r3, 0
  %e3 = zext i1 %b3 to i64
  %o3 = atomicrmw add ptr @gm.err, i64 %e3 monotonic, align 8
  br label %p3.cont

p3.cont:
  %j3.n = add nuw i64 %j3, 1
  %m3 = icmp ult i64 %j3.n, 50000
  br i1 %m3, label %p3, label %p4

; phase 4: put shared overlapping keys (all threads, same key/value)
p4:
  %j4 = phi i64 [ 0, %p3.cont ], [ %j4.n, %p4 ]
  %k4 = add nuw i64 900000000, %j4
  store i64 %k4, ptr %kb, align 8
  %v4a = mul nuw i64 %k4, 3
  %v4 = add nuw i64 %v4a, 1
  %r4 = call i32 @universe_conc_shardmap_put(ptr %m, ptr %kb, i64 8, i64 %v4)
  %b4 = icmp ne i32 %r4, 0
  %e4 = zext i1 %b4 to i64
  %o4 = atomicrmw add ptr @gm.err, i64 %e4 monotonic, align 8
  %j4.n = add nuw i64 %j4, 1
  %m4 = icmp ult i64 %j4.n, 5000
  br i1 %m4, label %p4, label %done

done:
  ret ptr null
}

define internal void @spawn_join_map() {
entry:
  br label %spawn

spawn:
  %i = phi i64 [ 0, %entry ], [ %i.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr @gm.tids, i64 0, i64 %i
  %arg = inttoptr i64 %i to ptr
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr @map_worker, ptr %arg)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 4
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @gm.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 4
  br i1 %jmore, label %join, label %done

done:
  ret void
}

; verify final map state single-threaded; returns violation count
define internal i64 @verify_map(ptr %m) {
entry:
  %kb = alloca i64, align 8
  %out = alloca i64, align 8
  br label %t.loop

t.loop:
  %t = phi i64 [ 0, %entry ], [ %t.n, %t.done ]
  %viol.t = phi i64 [ 0, %entry ], [ %viol.td, %t.done ]
  %base = mul nuw i64 %t, 1000000
  br label %d.loop

d.loop:
  %j = phi i64 [ 0, %t.loop ], [ %j.n, %d.cont ]
  %viol.j = phi i64 [ %viol.t, %t.loop ], [ %viol.jn, %d.cont ]
  %k = add nuw i64 %base, %j
  store i64 %k, ptr %kb, align 8
  %rc = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %odd = and i64 %j, 1
  %isodd = icmp eq i64 %odd, 1
  br i1 %isodd, label %expect.absent, label %expect.present

expect.absent:
  %ra = icmp ne i32 %rc, 5
  %via = zext i1 %ra to i64
  br label %d.cont

expect.present:
  %gv = load i64, ptr %out, align 8
  %eva = mul nuw i64 %k, 2
  %ev = add nuw i64 %eva, 1
  %rp = icmp ne i32 %rc, 0
  %vp = icmp ne i64 %gv, %ev
  %bp = or i1 %rp, %vp
  %vip = zext i1 %bp to i64
  br label %d.cont

d.cont:
  %vinc = phi i64 [ %via, %expect.absent ], [ %vip, %expect.present ]
  %viol.jn = add i64 %viol.j, %vinc
  %j.n = add nuw i64 %j, 1
  %dmore = icmp ult i64 %j.n, 50000
  br i1 %dmore, label %d.loop, label %t.done

t.done:
  %viol.td = phi i64 [ %viol.jn, %d.cont ]
  %t.n = add nuw i64 %t, 1
  %tmore = icmp ult i64 %t.n, 4
  br i1 %tmore, label %t.loop, label %shared

shared:
  br label %s.loop

s.loop:
  %s = phi i64 [ 0, %shared ], [ %s.n, %s.loop ]
  %viol.s = phi i64 [ %viol.td, %shared ], [ %viol.sn, %s.loop ]
  %sk = add nuw i64 900000000, %s
  store i64 %sk, ptr %kb, align 8
  %src = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %sgv = load i64, ptr %out, align 8
  %seva = mul nuw i64 %sk, 3
  %sev = add nuw i64 %seva, 1
  %srb = icmp ne i32 %src, 0
  %svb = icmp ne i64 %sgv, %sev
  %sb = or i1 %srb, %svb
  %svi = zext i1 %sb to i64
  %viol.sn = add i64 %viol.s, %svi
  %s.n = add nuw i64 %s, 1
  %smore = icmp ult i64 %s.n, 5000
  br i1 %smore, label %s.loop, label %lenchk

lenchk:
  %len = call i64 @universe_conc_shardmap_len(ptr %m)
  %len.bad = icmp ne i64 %len, 105000
  %vlen = zext i1 %len.bad to i64
  %vtot = add i64 %viol.sn, %vlen
  ret i64 %vtot
}

; run 10 rounds; return total violations (map errors + verify mismatches)
define internal i64 @run_stress() {
entry:
  br label %round

round:
  %r = phi i64 [ 0, %entry ], [ %r.n, %round ]
  %tv = phi i64 [ 0, %entry ], [ %tv.n, %round ]
  %m = call ptr @universe_conc_shardmap_create(i64 256, i64 64)
  store ptr %m, ptr @gm.m, align 8
  store i64 0, ptr @gm.err, align 8
  call void @spawn_join_map()
  %verr = call i64 @verify_map(ptr %m)
  %merr = load i64, ptr @gm.err, align 8
  %rv = add i64 %verr, %merr
  %tv.n = add i64 %tv, %rv
  call void @universe_conc_shardmap_destroy(ptr %m)
  %r.n = add nuw i64 %r, 1
  %more = icmp ult i64 %r.n, 10
  br i1 %more, label %round, label %done

done:
  ret i64 %tv.n
}

; ---- bench worker: contended mixed put/get on a small shared key range -----
@gm.benchn = internal global i64 0, align 8

define internal ptr @bench_worker(ptr %arg) {
entry:
  %m = load ptr, ptr @gm.m, align 8
  %n = load i64, ptr @gm.benchn, align 8
  %kb = alloca i64, align 8
  %out = alloca i64, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %key = and i64 %i, 4095
  store i64 %key, ptr %kb, align 8
  %rp = call i32 @universe_conc_shardmap_put(ptr %m, ptr %kb, i64 8, i64 %key)
  %rg = call i32 @universe_conc_shardmap_get(ptr %m, ptr %kb, i64 8, ptr %out)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret ptr null
}

define internal void @bench_spawn_join() {
entry:
  br label %spawn

spawn:
  %i = phi i64 [ 0, %entry ], [ %i.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr @gm.tids, i64 0, i64 %i
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr @bench_worker, ptr null)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 8
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @gm.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 8
  br i1 %jmore, label %join, label %done

done:
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) for the
; 256-shard (low contention) vs 1-shard (single global lock, same code path)
; 8-thread batches. 17 reps: rep 0 warm-up (discarded), reps 1..16 recorded.
; Each rep recreates the map. ops_per_rep = 8 * 500k * (put+get) = 8M.
define internal void @bench() {
entry:
  store i64 500000, ptr @gm.benchn, align 8
  br label %s256.rep

; 256-shard map
s256.rep:
  %ar = phi i64 [ 0, %entry ], [ %ar.n, %s256.next ]
  %m1 = call ptr @universe_conc_shardmap_create(i64 256, i64 8192)
  store ptr %m1, ptr @gm.m, align 8
  %at0 = call double @ut_now_sec()
  call void @bench_spawn_join()
  %at1 = call double @ut_now_sec()
  call void @universe_conc_shardmap_destroy(ptr %m1)
  %adt = fsub double %at1, %at0
  %awarm = icmp eq i64 %ar, 0
  br i1 %awarm, label %s256.next, label %s256.store

s256.store:
  %aidx = sub i64 %ar, 1
  %asp = getelementptr inbounds [16 x double], ptr @shard256.samp, i64 0, i64 %aidx
  store double %adt, ptr %asp, align 8
  br label %s256.next

s256.next:
  %ar.n = add nuw nsw i64 %ar, 1
  %armore = icmp ult i64 %ar.n, 17
  br i1 %armore, label %s256.rep, label %s256.done

s256.done:
  call void @ut_report_dist(ptr @shard256.samp, i64 16, i64 8000000, ptr @lbl.shard256)
  br label %s1.rep

; 1-shard map (single global lock, same code path)
s1.rep:
  %br = phi i64 [ 0, %s256.done ], [ %br.n, %s1.next ]
  %m2 = call ptr @universe_conc_shardmap_create(i64 1, i64 8192)
  store ptr %m2, ptr @gm.m, align 8
  %bt0 = call double @ut_now_sec()
  call void @bench_spawn_join()
  %bt1 = call double @ut_now_sec()
  call void @universe_conc_shardmap_destroy(ptr %m2)
  %bdt = fsub double %bt1, %bt0
  %bwarm = icmp eq i64 %br, 0
  br i1 %bwarm, label %s1.next, label %s1.store

s1.store:
  %bidx = sub i64 %br, 1
  %bsp = getelementptr inbounds [16 x double], ptr @shard1.samp, i64 0, i64 %bidx
  store double %bdt, ptr %bsp, align 8
  br label %s1.next

s1.next:
  %br.n = add nuw nsw i64 %br, 1
  %brmore = icmp ult i64 %br.n, 17
  br i1 %brmore, label %s1.rep, label %s1.done

s1.done:
  call void @ut_report_dist(ptr @shard1.samp, i64 16, i64 8000000, ptr @lbl.shard1)
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
