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

; Tests for universe_alloc_arena: alignment, accounting, non-overlap,
; exhaustion, overflow guards, reset reuse, --bench vs malloc.

declare ptr @universe_alloc_arena_create(i64)
declare ptr @universe_alloc_arena_alloc(ptr, i64)
declare ptr @universe_alloc_arena_alloc_aligned(ptr, i64, i64)
declare void @universe_alloc_arena_reset(ptr)
declare i64 @universe_alloc_arena_used(ptr)
declare i64 @universe_alloc_arena_capacity(ptr)
declare void @universe_alloc_arena_destroy(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@g.blocks = internal global [128 x ptr] zeroinitializer, align 16

@arena.samp  = internal global [16 x double] zeroinitializer, align 8
@malloc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.arena   = private unnamed_addr constant [16 x i8] c"arena alloc 64B\00"
@lbl.malloc  = private unnamed_addr constant [16 x i8] c"malloc/free 64B\00"

@m.create    = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.align16   = private unnamed_addr constant [22 x i8] c"blocks all 16-aligned\00"
@m.align256  = private unnamed_addr constant [21 x i8] c"aligned alloc honors\00"
@m.overlap   = private unnamed_addr constant [19 x i8] c"blocks not overlap\00"
@m.used      = private unnamed_addr constant [15 x i8] c"used() correct\00"
@m.cap       = private unnamed_addr constant [19 x i8] c"capacity() correct\00"
@m.exhaust   = private unnamed_addr constant [20 x i8] c"exhaustion get null\00"
@m.ovfsize   = private unnamed_addr constant [20 x i8] c"huge size get null \00"
@m.ovfcreate = private unnamed_addr constant [21 x i8] c"create overflow null\00"
@m.zerocap   = private unnamed_addr constant [20 x i8] c"zero-cap arena full\00"
@m.reset     = private unnamed_addr constant [19 x i8] c"reset reuses start\00"

define internal void @test_basic() {
entry:
  %arena = call ptr @universe_alloc_arena_create(i64 1048576)
  %ok = icmp ne ptr %arena, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %fill, label %done

fill:                                       ; 128 blocks of 48B, tagged
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %mis = phi i64 [ 0, %entry ], [ %mis.n, %fill ]
  %p = call ptr @universe_alloc_arena_alloc(ptr %arena, i64 48)
  %slot = getelementptr inbounds nuw [128 x ptr], ptr @g.blocks, i64 0, i64 %i
  store ptr %p, ptr %slot, align 8
  %tag = trunc i64 %i to i8
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 48, i1 false)
  %p.i = ptrtoint ptr %p to i64
  %lo = and i64 %p.i, 15
  %misaligned = icmp ne i64 %lo, 0
  %mis.inc = zext i1 %misaligned to i64
  %mis.n = add nuw i64 %mis, %mis.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 128
  br i1 %more, label %fill, label %verify

verify:                                     ; every block still holds its tag
  call void @ut_check_eq(i64 %mis.n, i64 0, ptr @m.align16)
  br label %vloop

vloop:
  %j = phi i64 [ 0, %verify ], [ %j.n, %vloop ]
  %bad = phi i64 [ 0, %verify ], [ %bad.n, %vloop ]
  %slot2 = getelementptr inbounds nuw [128 x ptr], ptr @g.blocks, i64 0, i64 %j
  %q = load ptr, ptr %slot2, align 8
  %want = trunc i64 %j to i8
  %b0 = load i8, ptr %q, align 1
  %q24 = getelementptr inbounds nuw i8, ptr %q, i64 24
  %b24 = load i8, ptr %q24, align 1
  %q47 = getelementptr inbounds nuw i8, ptr %q, i64 47
  %b47 = load i8, ptr %q47, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok24 = icmp eq i8 %b24, %want
  %ok47 = icmp eq i8 %b47, %want
  %okA = and i1 %ok0, %ok24
  %okB = and i1 %okA, %ok47
  %bad.b = xor i1 %okB, true
  %bad.inc = zext i1 %bad.b to i64
  %bad.n = add nuw i64 %bad, %bad.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 128
  br i1 %more2, label %vloop, label %after

after:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.overlap)
  ; 48 is already 16-aligned: 128 x 48 = 6144, no rounding waste
  %used = call i64 @universe_alloc_arena_used(ptr %arena)
  call void @ut_check_eq(i64 %used, i64 6144, ptr @m.used)
  %cap = call i64 @universe_alloc_arena_capacity(ptr %arena)
  call void @ut_check_eq(i64 %cap, i64 1048576, ptr @m.cap)

  ; big alignment
  %pa = call ptr @universe_alloc_arena_alloc_aligned(ptr %arena, i64 100, i64 256)
  %pa.i = ptrtoint ptr %pa to i64
  %pa.lo = and i64 %pa.i, 255
  %pa.null = icmp ne ptr %pa, null
  %pa.al = icmp eq i64 %pa.lo, 0
  %pa.ok = and i1 %pa.null, %pa.al
  call void @ut_check(i1 %pa.ok, ptr @m.align256)

  ; exhaustion + overflow guards
  %big = call ptr @universe_alloc_arena_alloc(ptr %arena, i64 2097152)
  %big.null = icmp eq ptr %big, null
  call void @ut_check(i1 %big.null, ptr @m.exhaust)
  %huge = call ptr @universe_alloc_arena_alloc(ptr %arena, i64 -8)
  %huge.null = icmp eq ptr %huge, null
  call void @ut_check(i1 %huge.null, ptr @m.ovfsize)

  ; reset reuse: first alloc after reset lands at payload start again
  %first = load ptr, ptr @g.blocks, align 8
  call void @universe_alloc_arena_reset(ptr %arena)
  %again = call ptr @universe_alloc_arena_alloc(ptr %arena, i64 16)
  %same = icmp eq ptr %again, %first
  call void @ut_check(i1 %same, ptr @m.reset)

  call void @universe_alloc_arena_destroy(ptr %arena)
  call void @universe_alloc_arena_destroy(ptr null)
  br label %done

done:
  ret void
}

define internal void @test_edge_create() {
entry:
  %bad = call ptr @universe_alloc_arena_create(i64 -1)
  %bad.null = icmp eq ptr %bad, null
  call void @ut_check(i1 %bad.null, ptr @m.ovfcreate)

  %zero = call ptr @universe_alloc_arena_create(i64 0)
  %zero.ok = icmp ne ptr %zero, null
  br i1 %zero.ok, label %probe, label %skip

probe:
  %p = call ptr @universe_alloc_arena_alloc(ptr %zero, i64 1)
  %p.null = icmp eq ptr %p, null
  call void @ut_check(i1 %p.null, ptr @m.zerocap)
  call void @universe_alloc_arena_destroy(ptr %zero)
  br label %skip

skip:
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op).
; 17 reps per op; rep 0 is warm-up (discarded), reps 1..16 recorded.
define internal void @bench() {
entry:
  br label %ar.rep

ar.rep:                                     ; arena: 100k x 64B, reset every 1024
  %ar.r = phi i64 [ 0, %entry ], [ %ar.r.n, %ar.next ]
  %ar.arena = call ptr @universe_alloc_arena_create(i64 67108864)
  %ar.t0 = call double @ut_now_sec()
  br label %aloop

aloop:
  %i = phi i64 [ 0, %ar.rep ], [ %i.n, %acont ]
  %p = call ptr @universe_alloc_arena_alloc(ptr %ar.arena, i64 64)
  store i64 %i, ptr %p, align 8
  %i.n = add nuw nsw i64 %i, 1
  %chunk = and i64 %i.n, 1023
  %atend = icmp eq i64 %chunk, 0
  br i1 %atend, label %do.reset, label %acont

do.reset:
  call void @universe_alloc_arena_reset(ptr %ar.arena)
  br label %acont

acont:
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %aloop, label %ar.mid

ar.mid:
  %ar.t1 = call double @ut_now_sec()
  call void @universe_alloc_arena_destroy(ptr %ar.arena)
  %ar.el = fsub double %ar.t1, %ar.t0
  %ar.warm = icmp eq i64 %ar.r, 0
  br i1 %ar.warm, label %ar.next, label %ar.store

ar.store:
  %ar.idx = sub i64 %ar.r, 1
  %ar.sp = getelementptr inbounds [16 x double], ptr @arena.samp, i64 0, i64 %ar.idx
  store double %ar.el, ptr %ar.sp, align 8
  br label %ar.next

ar.next:
  %ar.r.n = add nuw nsw i64 %ar.r, 1
  %ar.rmore = icmp ult i64 %ar.r.n, 17
  br i1 %ar.rmore, label %ar.rep, label %ar.done

ar.done:
  call void @ut_report_dist(ptr @arena.samp, i64 16, i64 100000, ptr @lbl.arena)
  br label %ml.rep

ml.rep:                                     ; malloc/free churn reference
  %ml.r = phi i64 [ 0, %ar.done ], [ %ml.r.n, %ml.next ]
  %ml.t0 = call double @ut_now_sec()
  br label %mloop

mloop:
  ; volatile store defeats malloc/free pair elision so the reference is honest
  %j = phi i64 [ 0, %ml.rep ], [ %j.n, %mloop ]
  %q = call ptr @malloc(i64 64)
  store volatile i64 %j, ptr %q, align 8
  call void @free(ptr %q)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 100000
  br i1 %more2, label %mloop, label %ml.mid

ml.mid:
  %ml.t1 = call double @ut_now_sec()
  %ml.el = fsub double %ml.t1, %ml.t0
  %ml.warm = icmp eq i64 %ml.r, 0
  br i1 %ml.warm, label %ml.next, label %ml.store

ml.store:
  %ml.idx = sub i64 %ml.r, 1
  %ml.sp = getelementptr inbounds [16 x double], ptr @malloc.samp, i64 0, i64 %ml.idx
  store double %ml.el, ptr %ml.sp, align 8
  br label %ml.next

ml.next:
  %ml.r.n = add nuw nsw i64 %ml.r, 1
  %ml.rmore = icmp ult i64 %ml.r.n, 17
  br i1 %ml.rmore, label %ml.rep, label %ml.done

ml.done:
  call void @ut_report_dist(ptr @malloc.samp, i64 16, i64 100000, ptr @lbl.malloc)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_edge_create()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
