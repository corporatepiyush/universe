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

; Tests for universe_alloc_cpool (lock-free concurrent pool).
; Stress: 4 threads x 100k churn on a 64-block pool. Each holder stamps its
; tid into the block and re-reads it after busywork — a stamp change means
; the same block was handed to two threads (ABA / lost-update bug).

declare ptr @universe_alloc_cpool_create(i64, i64)
declare ptr @universe_alloc_cpool_alloc(ptr)
declare void @universe_alloc_cpool_free(ptr, ptr)
declare i64 @universe_alloc_cpool_live(ptr)
declare i64 @universe_alloc_cpool_capacity(ptr)
declare void @universe_alloc_cpool_destroy(ptr)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)

@g.pool = internal global ptr null, align 8
@g.tids = internal global [4 x i64] zeroinitializer, align 8
@g.viol = internal global i64 0, align 8
@g.starved = internal global i64 0, align 8

@m.drain    = private unnamed_addr constant [19 x i8] c"drain all 8 blocks\00"
@m.exhaust  = private unnamed_addr constant [16 x i8] c"9th alloc null \00"
@m.refill   = private unnamed_addr constant [22 x i8] c"redrain via free list\00"
@m.live0    = private unnamed_addr constant [7 x i8] c"live 0\00"
@m.spawn    = private unnamed_addr constant [16 x i8] c"threads spawned\00"
@m.noviol   = private unnamed_addr constant [25 x i8] c"no double-alloc detected\00"
@m.nostarve = private unnamed_addr constant [23 x i8] c"no starvation (64>4x2)\00"
@m.live0s   = private unnamed_addr constant [22 x i8] c"live 0 after stress  \00"

@g.blocks = internal global [8 x ptr] zeroinitializer, align 16

define internal void @test_single() {
entry:
  %p = call ptr @universe_alloc_cpool_create(i64 48, i64 8)
  br label %drain

drain:
  %i = phi i64 [ 0, %entry ], [ %i.n, %drain ]
  %nulls = phi i64 [ 0, %entry ], [ %nulls.n, %drain ]
  %b = call ptr @universe_alloc_cpool_alloc(ptr %p)
  %slot = getelementptr inbounds nuw [8 x ptr], ptr @g.blocks, i64 0, i64 %i
  store ptr %b, ptr %slot, align 8
  %b.null = icmp eq ptr %b, null
  %n.inc = zext i1 %b.null to i64
  %nulls.n = add nuw i64 %nulls, %n.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 8
  br i1 %more, label %drain, label %full

full:
  call void @ut_check_eq(i64 %nulls.n, i64 0, ptr @m.drain)
  %extra = call ptr @universe_alloc_cpool_alloc(ptr %p)
  %extra.null = icmp eq ptr %extra, null
  call void @ut_check(i1 %extra.null, ptr @m.exhaust)
  br label %freeall

freeall:
  %j = phi i64 [ 0, %full ], [ %j.n, %freeall ]
  %slot2 = getelementptr inbounds nuw [8 x ptr], ptr @g.blocks, i64 0, i64 %j
  %q = load ptr, ptr %slot2, align 8
  call void @universe_alloc_cpool_free(ptr %p, ptr %q)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 8
  br i1 %more2, label %freeall, label %redrain.pre

redrain.pre:
  br label %redrain

redrain:                                     ; comes back through free list now
  %k = phi i64 [ 0, %redrain.pre ], [ %k.n, %redrain ]
  %renulls = phi i64 [ 0, %redrain.pre ], [ %renulls.n, %redrain ]
  %rb = call ptr @universe_alloc_cpool_alloc(ptr %p)
  %rb.null = icmp eq ptr %rb, null
  %re.inc = zext i1 %rb.null to i64
  %renulls.n = add nuw i64 %renulls, %re.inc
  %slot3 = getelementptr inbounds nuw [8 x ptr], ptr @g.blocks, i64 0, i64 %k
  store ptr %rb, ptr %slot3, align 8
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 8
  br i1 %more3, label %redrain, label %refree

refree:
  call void @ut_check_eq(i64 %renulls.n, i64 0, ptr @m.refill)
  br label %refree.loop

refree.loop:
  %m = phi i64 [ 0, %refree ], [ %m.n, %refree.loop ]
  %slot4 = getelementptr inbounds nuw [8 x ptr], ptr @g.blocks, i64 0, i64 %m
  %fq = load ptr, ptr %slot4, align 8
  call void @universe_alloc_cpool_free(ptr %p, ptr %fq)
  %m.n = add nuw nsw i64 %m, 1
  %more4 = icmp ult i64 %m.n, 8
  br i1 %more4, label %refree.loop, label %fin

fin:
  %live = call i64 @universe_alloc_cpool_live(ptr %p)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  call void @universe_alloc_cpool_destroy(ptr %p)
  ret void
}

; Stress worker: churn alloc/stamp/verify/free 100k times, hold 2 at a time.
define internal ptr @worker(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %pool = load ptr, ptr @g.pool, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop.end ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %loop.end ]
  %starv = phi i64 [ 0, %entry ], [ %starv.n, %loop.end ]
  %p1 = call ptr @universe_alloc_cpool_alloc(ptr %pool)
  %p2 = call ptr @universe_alloc_cpool_alloc(ptr %pool)
  %p1.null = icmp eq ptr %p1, null
  %p2.null = icmp eq ptr %p2, null
  %any.null = or i1 %p1.null, %p2.null
  br i1 %any.null, label %starved, label %stamp

starved:                                     ; 64 blocks, max 8 held: never
  %s.inc = zext i1 %any.null to i64
  %starv.inc = add nuw i64 %starv, %s.inc
  br i1 %p1.null, label %starve.skip1, label %starve.free1

starve.free1:
  call void @universe_alloc_cpool_free(ptr %pool, ptr %p1)
  br label %starve.skip1

starve.skip1:
  br i1 %p2.null, label %loop.end.starved, label %starve.free2

starve.free2:
  call void @universe_alloc_cpool_free(ptr %pool, ptr %p2)
  br label %loop.end.starved

loop.end.starved:
  br label %loop.end

stamp:                                       ; stamp at +8 (byte 0 is the
  %s1 = getelementptr inbounds nuw i8, ptr %p1, i64 8  ; free-list link)
  %s2 = getelementptr inbounds nuw i8, ptr %p2, i64 8
  store i64 %tid, ptr %s1, align 8
  store i64 %tid, ptr %s2, align 8
  ; busywork: forces a window for a racing double-alloc to trample the stamp
  %r1 = load volatile i64, ptr %s1, align 8
  %r2 = load volatile i64, ptr %s2, align 8
  %back1 = load i64, ptr %s1, align 8
  %back2 = load i64, ptr %s2, align 8
  %ok1 = icmp eq i64 %back1, %tid
  %ok2 = icmp eq i64 %back2, %tid
  %both = and i1 %ok1, %ok2
  %v.b = xor i1 %both, true
  %v.inc = zext i1 %v.b to i64
  call void @universe_alloc_cpool_free(ptr %pool, ptr %p2)
  call void @universe_alloc_cpool_free(ptr %pool, ptr %p1)
  br label %loop.tally

loop.tally:
  br label %loop.end

loop.end:
  %v.add = phi i64 [ %v.inc, %loop.tally ], [ 0, %loop.end.starved ]
  %s.add = phi i64 [ 0, %loop.tally ], [ 1, %loop.end.starved ]
  %viol.n = add nuw i64 %viol, %v.add
  %starv.n = add nuw i64 %starv, %s.add
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %loop, label %tally

tally:
  %v.old = atomicrmw add ptr @g.viol, i64 %viol.n monotonic, align 8
  %s.old = atomicrmw add ptr @g.starved, i64 %starv.n monotonic, align 8
  ret ptr null
}

define internal void @test_stress() {
entry:
  %p = call ptr @universe_alloc_cpool_create(i64 64, i64 64)
  store ptr %p, ptr @g.pool, align 8
  br label %spawn

spawn:
  %t = phi i64 [ 0, %entry ], [ %t.n, %spawn ]
  %fails = phi i64 [ 0, %entry ], [ %fails.n, %spawn ]
  %tid.slot = getelementptr inbounds nuw [4 x i64], ptr @g.tids, i64 0, i64 %t
  %arg = inttoptr i64 %t to ptr
  %rc = call i32 @pthread_create(ptr %tid.slot, ptr null, ptr @worker, ptr %arg)
  %rc.bad = icmp ne i32 %rc, 0
  %f.inc = zext i1 %rc.bad to i64
  %fails.n = add nuw i64 %fails, %f.inc
  %t.n = add nuw nsw i64 %t, 1
  %more = icmp ult i64 %t.n, 4
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %tid.slot2 = getelementptr inbounds nuw [4 x i64], ptr @g.tids, i64 0, i64 %j
  %tid = load i64, ptr %tid.slot2, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 4
  br i1 %more2, label %join, label %verify

verify:
  call void @ut_check_eq(i64 %fails.n, i64 0, ptr @m.spawn)
  %viol = load i64, ptr @g.viol, align 8
  call void @ut_check_eq(i64 %viol, i64 0, ptr @m.noviol)
  %starved = load i64, ptr @g.starved, align 8
  call void @ut_check_eq(i64 %starved, i64 0, ptr @m.nostarve)
  %live = call i64 @universe_alloc_cpool_live(ptr %p)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0s)
  call void @universe_alloc_cpool_destroy(ptr %p)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single()
  call void @test_stress()
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
