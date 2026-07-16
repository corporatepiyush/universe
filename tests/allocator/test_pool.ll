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

; Tests for universe_alloc_pool: full drain, exhaustion, free/reuse cycling,
; pattern integrity across frees, accounting, overflow guards, --bench.

declare ptr @universe_alloc_pool_create(i64, i64)
declare ptr @universe_alloc_pool_alloc(ptr)
declare void @universe_alloc_pool_free(ptr, ptr)
declare i64 @universe_alloc_pool_live(ptr)
declare i64 @universe_alloc_pool_capacity(ptr)
declare void @universe_alloc_pool_destroy(ptr)

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

@g.ptrs = internal global [1000 x ptr] zeroinitializer, align 16

@pool.samp   = internal global [16 x double] zeroinitializer, align 8
@malloc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.pool    = private unnamed_addr constant [25 x i8] c"pool alloc/free pair 64B\00"
@lbl.malloc  = private unnamed_addr constant [27 x i8] c"malloc alloc/free pair 64B\00"

@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.drain    = private unnamed_addr constant [21 x i8] c"all blocks allocable\00"
@m.align    = private unnamed_addr constant [19 x i8] c"blocks 16-aligned \00"
@m.exhaust  = private unnamed_addr constant [21 x i8] c"exhausted pool null \00"
@m.live     = private unnamed_addr constant [13 x i8] c"live tracked\00"
@m.cap      = private unnamed_addr constant [12 x i8] c"capacity ok\00"
@m.reuse    = private unnamed_addr constant [21 x i8] c"freed blocks reused \00"
@m.integrity = private unnamed_addr constant [25 x i8] c"survivors keep patterns \00"
@m.livezero = private unnamed_addr constant [20 x i8] c"live 0 after frees \00"
@m.ovf      = private unnamed_addr constant [21 x i8] c"create overflow null\00"

define internal void @test_lifecycle() {
entry:
  %pool = call ptr @universe_alloc_pool_create(i64 40, i64 1000)
  %ok = icmp ne ptr %pool, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %drain, label %done

drain:                                       ; take all 1000; tag each
  %i = phi i64 [ 0, %entry ], [ %i.n, %drain ]
  %nulls = phi i64 [ 0, %entry ], [ %nulls.n, %drain ]
  %mis = phi i64 [ 0, %entry ], [ %mis.n, %drain ]
  %p = call ptr @universe_alloc_pool_alloc(ptr %pool)
  %slot = getelementptr inbounds nuw [1000 x ptr], ptr @g.ptrs, i64 0, i64 %i
  store ptr %p, ptr %slot, align 8
  %isnull = icmp eq ptr %p, null
  %null.inc = zext i1 %isnull to i64
  %nulls.n = add nuw i64 %nulls, %null.inc
  %tag = trunc i64 %i to i8
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 40, i1 false)
  %p.i = ptrtoint ptr %p to i64
  %lo = and i64 %p.i, 15
  %misal = icmp ne i64 %lo, 0
  %mis.inc = zext i1 %misal to i64
  %mis.n = add nuw i64 %mis, %mis.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %drain, label %full

full:
  call void @ut_check_eq(i64 %nulls.n, i64 0, ptr @m.drain)
  call void @ut_check_eq(i64 %mis.n, i64 0, ptr @m.align)
  %extra = call ptr @universe_alloc_pool_alloc(ptr %pool)
  %extra.null = icmp eq ptr %extra, null
  call void @ut_check(i1 %extra.null, ptr @m.exhaust)
  %live = call i64 @universe_alloc_pool_live(ptr %pool)
  call void @ut_check_eq(i64 %live, i64 1000, ptr @m.live)
  %cap = call i64 @universe_alloc_pool_capacity(ptr %pool)
  call void @ut_check_eq(i64 %cap, i64 1000, ptr @m.cap)
  br label %free.evens

free.evens:                                  ; free every even-index block
  %j = phi i64 [ 0, %full ], [ %j.n, %free.evens ]
  %slot2 = getelementptr inbounds nuw [1000 x ptr], ptr @g.ptrs, i64 0, i64 %j
  %q = load ptr, ptr %slot2, align 8
  call void @universe_alloc_pool_free(ptr %pool, ptr %q)
  %j.n = add nuw nsw i64 %j, 2
  %more2 = icmp ult i64 %j.n, 1000
  br i1 %more2, label %free.evens, label %realloc

realloc:                                     ; re-take 500; all must succeed
  %k = phi i64 [ 0, %free.evens ], [ %k.n, %realloc ]
  %renulls = phi i64 [ 0, %free.evens ], [ %renulls.n, %realloc ]
  %r = call ptr @universe_alloc_pool_alloc(ptr %pool)
  %r.null = icmp eq ptr %r, null
  %re.inc = zext i1 %r.null to i64
  %renulls.n = add nuw i64 %renulls, %re.inc
  call void @llvm.memset.p0.i64(ptr %r, i8 -1, i64 40, i1 false)
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 500
  br i1 %more3, label %realloc, label %verify

verify:                                      ; odd-index survivors keep tags
  call void @ut_check_eq(i64 %renulls.n, i64 0, ptr @m.reuse)
  br label %vloop

vloop:
  %m = phi i64 [ 1, %verify ], [ %m.n, %vloop ]
  %bad = phi i64 [ 0, %verify ], [ %bad.n, %vloop ]
  %slot3 = getelementptr inbounds nuw [1000 x ptr], ptr @g.ptrs, i64 0, i64 %m
  %s = load ptr, ptr %slot3, align 8
  %want = trunc i64 %m to i8
  %b0 = load i8, ptr %s, align 1
  %s39 = getelementptr inbounds nuw i8, ptr %s, i64 39
  %b39 = load i8, ptr %s39, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok39 = icmp eq i8 %b39, %want
  %okA = and i1 %ok0, %ok39
  %bad.b = xor i1 %okA, true
  %bad.inc = zext i1 %bad.b to i64
  %bad.n = add nuw i64 %bad, %bad.inc
  %m.n = add nuw nsw i64 %m, 2
  %more4 = icmp ult i64 %m.n, 1000
  br i1 %more4, label %vloop, label %after

after:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.integrity)
  call void @universe_alloc_pool_destroy(ptr %pool)
  call void @universe_alloc_pool_destroy(ptr null)
  br label %done

done:
  ret void
}

define internal void @test_accounting() {
entry:
  %pool = call ptr @universe_alloc_pool_create(i64 8, i64 4)
  %a = call ptr @universe_alloc_pool_alloc(ptr %pool)
  %b = call ptr @universe_alloc_pool_alloc(ptr %pool)
  call void @universe_alloc_pool_free(ptr %pool, ptr %a)
  call void @universe_alloc_pool_free(ptr %pool, ptr %b)
  %live = call i64 @universe_alloc_pool_live(ptr %pool)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.livezero)
  call void @universe_alloc_pool_destroy(ptr %pool)

  ; overflow guards: stride*count overflow and size round-up overflow
  %bad1 = call ptr @universe_alloc_pool_create(i64 -1, i64 8)
  %bad1.null = icmp eq ptr %bad1, null
  call void @ut_check(i1 %bad1.null, ptr @m.ovf)
  %bad2 = call ptr @universe_alloc_pool_create(i64 1024, i64 4611686018427387904)
  %bad2.null = icmp eq ptr %bad2, null
  call void @ut_check(i1 %bad2.null, ptr @m.ovf)
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op).
; 17 reps per op; rep 0 is warm-up (discarded), reps 1..16 recorded.
; ops_per_rep = 100k alloc/free pairs.
define internal void @bench() {
entry:
  %pool = call ptr @universe_alloc_pool_create(i64 64, i64 16)
  br label %pl.rep

pl.rep:                                      ; alloc+free churn through slots
  %pl.r = phi i64 [ 0, %entry ], [ %pl.r.n, %pl.next ]
  %pl.t0 = call double @ut_now_sec()
  br label %ploop

ploop:
  %i = phi i64 [ 0, %pl.rep ], [ %i.n, %ploop ]
  %p1 = call ptr @universe_alloc_pool_alloc(ptr %pool)
  %p2 = call ptr @universe_alloc_pool_alloc(ptr %pool)
  store i64 %i, ptr %p1, align 8
  store i64 %i, ptr %p2, align 8
  call void @universe_alloc_pool_free(ptr %pool, ptr %p2)
  call void @universe_alloc_pool_free(ptr %pool, ptr %p1)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %ploop, label %pl.mid

pl.mid:
  %pl.t1 = call double @ut_now_sec()
  %pl.el = fsub double %pl.t1, %pl.t0
  %pl.warm = icmp eq i64 %pl.r, 0
  br i1 %pl.warm, label %pl.next, label %pl.store

pl.store:
  %pl.idx = sub i64 %pl.r, 1
  %pl.sp = getelementptr inbounds [16 x double], ptr @pool.samp, i64 0, i64 %pl.idx
  store double %pl.el, ptr %pl.sp, align 8
  br label %pl.next

pl.next:
  %pl.r.n = add nuw nsw i64 %pl.r, 1
  %pl.rmore = icmp ult i64 %pl.r.n, 17
  br i1 %pl.rmore, label %pl.rep, label %pl.done

pl.done:
  call void @universe_alloc_pool_destroy(ptr %pool)
  call void @ut_report_dist(ptr @pool.samp, i64 16, i64 100000, ptr @lbl.pool)
  br label %ml.rep

ml.rep:
  %ml.r = phi i64 [ 0, %pl.done ], [ %ml.r.n, %ml.next ]
  %ml.t0 = call double @ut_now_sec()
  br label %mloop

mloop:
  ; volatile stores defeat malloc/free pair elision (honest reference)
  %j = phi i64 [ 0, %ml.rep ], [ %j.n, %mloop ]
  %q1 = call ptr @malloc(i64 64)
  %q2 = call ptr @malloc(i64 64)
  store volatile i64 %j, ptr %q1, align 8
  store volatile i64 %j, ptr %q2, align 8
  call void @free(ptr %q2)
  call void @free(ptr %q1)
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
  call void @test_lifecycle()
  call void @test_accounting()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
