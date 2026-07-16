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

; Tests for universe_alloc_carena (wait-free concurrent arena).
; Stress: 4 threads x 50k allocs; each thread fills its blocks with its own
; tag; afterwards every block must carry exactly its allocator's tag — any
; overlap between threads would tear a tag. Runs on every make test.

declare ptr @universe_alloc_carena_create(i64)
declare ptr @universe_alloc_carena_alloc(ptr, i64)
declare ptr @universe_alloc_carena_alloc_aligned(ptr, i64, i64)
declare void @universe_alloc_carena_reset(ptr)
declare i64 @universe_alloc_carena_used(ptr)
declare i64 @universe_alloc_carena_capacity(ptr)
declare void @universe_alloc_carena_destroy(ptr)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)

@g.arena = internal global ptr null, align 8
@g.ptrs = internal global [200000 x ptr] zeroinitializer, align 16
@g.tids = internal global [4 x i64] zeroinitializer, align 8

@m.basic   = private unnamed_addr constant [20 x i8] c"single-thread bump \00"
@m.aligned = private unnamed_addr constant [20 x i8] c"aligned alloc 4096 \00"
@m.reset   = private unnamed_addr constant [15 x i8] c"reset to zero \00"
@m.spawn   = private unnamed_addr constant [16 x i8] c"threads spawned\00"
@m.nonull  = private unnamed_addr constant [22 x i8] c"no null under stress \00"
@m.used    = private unnamed_addr constant [21 x i8] c"used == 4x50kx32 sum\00"
@m.tags    = private unnamed_addr constant [25 x i8] c"no cross-thread overlap \00"

define internal void @test_single() {
entry:
  %a = call ptr @universe_alloc_carena_create(i64 65536)
  %p1 = call ptr @universe_alloc_carena_alloc(ptr %a, i64 100)
  %p2 = call ptr @universe_alloc_carena_alloc(ptr %a, i64 100)
  %p1.i = ptrtoint ptr %p1 to i64
  %p2.i = ptrtoint ptr %p2 to i64
  %delta = sub i64 %p2.i, %p1.i
  ; 100 rounds to 112
  call void @ut_check_eq(i64 %delta, i64 112, ptr @m.basic)
  %pa = call ptr @universe_alloc_carena_alloc_aligned(ptr %a, i64 64, i64 4096)
  %pa.i = ptrtoint ptr %pa to i64
  %pa.lo = and i64 %pa.i, 4095
  %pa.null = icmp ne ptr %pa, null
  %pa.al = icmp eq i64 %pa.lo, 0
  %pa.ok = and i1 %pa.null, %pa.al
  call void @ut_check(i1 %pa.ok, ptr @m.aligned)
  call void @universe_alloc_carena_reset(ptr %a)
  %used = call i64 @universe_alloc_carena_used(ptr %a)
  call void @ut_check_eq(i64 %used, i64 0, ptr @m.reset)
  call void @universe_alloc_carena_destroy(ptr %a)
  ret void
}

; worker: arg = tid (0..3). 50k allocs of 32B, tag with 'A'+tid, record ptrs.
define internal ptr @worker(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %tag64 = add nuw i64 %tid, 65
  %tag = trunc i64 %tag64 to i8
  %base = mul nuw i64 %tid, 50000
  %arena = load ptr, ptr @g.arena, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %p = call ptr @universe_alloc_carena_alloc(ptr %arena, i64 32)
  %idx = add nuw i64 %base, %i
  %slot = getelementptr inbounds nuw [200000 x ptr], ptr @g.ptrs, i64 0, i64 %idx
  store ptr %p, ptr %slot, align 8
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 32, i1 false)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 50000
  br i1 %more, label %loop, label %done

done:
  ret ptr null
}

define internal void @test_stress() {
entry:
  ; exactly 4*50000*32 = 6,400,000 bytes
  %a = call ptr @universe_alloc_carena_create(i64 6400000)
  store ptr %a, ptr @g.arena, align 8
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
  br i1 %more2, label %join, label %verify.pre

verify.pre:
  call void @ut_check_eq(i64 %fails.n, i64 0, ptr @m.spawn)
  %used = call i64 @universe_alloc_carena_used(ptr %a)
  call void @ut_check_eq(i64 %used, i64 6400000, ptr @m.used)
  br label %verify

verify:                                      ; each recorded block: nonnull +
  %k = phi i64 [ 0, %verify.pre ], [ %k.n, %verify ] ; still wears its thread tag
  %nulls = phi i64 [ 0, %verify.pre ], [ %nulls.n, %verify ]
  %bad = phi i64 [ 0, %verify.pre ], [ %bad.n, %verify ]
  %slot3 = getelementptr inbounds nuw [200000 x ptr], ptr @g.ptrs, i64 0, i64 %k
  %p = load ptr, ptr %slot3, align 8
  %p.null = icmp eq ptr %p, null
  %n.inc = zext i1 %p.null to i64
  %nulls.n = add nuw i64 %nulls, %n.inc
  %owner = udiv i64 %k, 50000
  %want64 = add nuw i64 %owner, 65
  %want = trunc i64 %want64 to i8
  %b0 = load i8, ptr %p, align 1
  %p31 = getelementptr inbounds nuw i8, ptr %p, i64 31
  %b31 = load i8, ptr %p31, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok31 = icmp eq i8 %b31, %want
  %okA = and i1 %ok0, %ok31
  %bad.b = xor i1 %okA, true
  %bad.inc = zext i1 %bad.b to i64
  %bad.n = add nuw i64 %bad, %bad.inc
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 200000
  br i1 %more3, label %verify, label %report

report:
  call void @ut_check_eq(i64 %nulls.n, i64 0, ptr @m.nonull)
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.tags)
  call void @universe_alloc_carena_destroy(ptr %a)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single()
  call void @test_stress()
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
