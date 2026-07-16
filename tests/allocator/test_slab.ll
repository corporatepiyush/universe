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

; Tests for universe_alloc_slab: multi-slab growth, pattern integrity across
; free/realloc, empty-slab release + regrowth, overflow guards, --bench.

declare ptr @universe_alloc_slab_create(i64, i64)
declare ptr @universe_alloc_slab_alloc(ptr)
declare void @universe_alloc_slab_free(ptr, ptr)
declare i64 @universe_alloc_slab_live(ptr)
declare void @universe_alloc_slab_destroy(ptr)

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

@g.objs = internal global [25 x ptr] zeroinitializer, align 16

@slab.samp   = internal global [16 x double] zeroinitializer, align 8
@malloc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.slab    = private unnamed_addr constant [25 x i8] c"slab alloc/free pair 64B\00"
@lbl.malloc  = private unnamed_addr constant [27 x i8] c"malloc alloc/free pair 64B\00"

@m.create  = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.grow    = private unnamed_addr constant [25 x i8] c"grows across 3+ slabs ok\00"
@m.align   = private unnamed_addr constant [18 x i8] c"objs 16-aligned  \00"
@m.live25  = private unnamed_addr constant [8 x i8] c"live 25\00"
@m.live12  = private unnamed_addr constant [8 x i8] c"live 12\00"
@m.realloc = private unnamed_addr constant [17 x i8] c"realloc after 13\00"
@m.tags    = private unnamed_addr constant [21 x i8] c"survivor tags intact\00"
@m.empty   = private unnamed_addr constant [21 x i8] c"live 0 all released \00"
@m.regrow  = private unnamed_addr constant [22 x i8] c"alloc works after all\00"
@m.ovf     = private unnamed_addr constant [21 x i8] c"create overflow null\00"

define internal void @test_lifecycle() {
entry:
  %s = call ptr @universe_alloc_slab_create(i64 24, i64 10)
  %ok = icmp ne ptr %s, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %fill, label %done

fill:                                        ; 25 objects across >= 3 slabs
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %nulls = phi i64 [ 0, %entry ], [ %nulls.n, %fill ]
  %mis = phi i64 [ 0, %entry ], [ %mis.n, %fill ]
  %p = call ptr @universe_alloc_slab_alloc(ptr %s)
  %slot = getelementptr inbounds nuw [25 x ptr], ptr @g.objs, i64 0, i64 %i
  store ptr %p, ptr %slot, align 8
  %isnull = icmp eq ptr %p, null
  %n.inc = zext i1 %isnull to i64
  %nulls.n = add nuw i64 %nulls, %n.inc
  %tag = trunc i64 %i to i8
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 24, i1 false)
  %p.i = ptrtoint ptr %p to i64
  %lo = and i64 %p.i, 15
  %misal = icmp ne i64 %lo, 0
  %m.inc = zext i1 %misal to i64
  %mis.n = add nuw i64 %mis, %m.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 25
  br i1 %more, label %fill, label %filled

filled:
  call void @ut_check_eq(i64 %nulls.n, i64 0, ptr @m.grow)
  call void @ut_check_eq(i64 %mis.n, i64 0, ptr @m.align)
  %live = call i64 @universe_alloc_slab_live(ptr %s)
  call void @ut_check_eq(i64 %live, i64 25, ptr @m.live25)
  br label %free.evens

free.evens:                                  ; free indices 0,2,...,24 (13 objs)
  %j = phi i64 [ 0, %filled ], [ %j.n, %free.evens ]
  %slot2 = getelementptr inbounds nuw [25 x ptr], ptr @g.objs, i64 0, i64 %j
  %q = load ptr, ptr %slot2, align 8
  call void @universe_alloc_slab_free(ptr %s, ptr %q)
  %j.n = add nuw nsw i64 %j, 2
  %more2 = icmp ult i64 %j.n, 25
  br i1 %more2, label %free.evens, label %check.12

check.12:
  %live2 = call i64 @universe_alloc_slab_live(ptr %s)
  call void @ut_check_eq(i64 %live2, i64 12, ptr @m.live12)
  br label %realloc

realloc:                                     ; take 13 back
  %k = phi i64 [ 0, %check.12 ], [ %k.n, %realloc ]
  %renulls = phi i64 [ 0, %check.12 ], [ %renulls.n, %realloc ]
  %r = call ptr @universe_alloc_slab_alloc(ptr %s)
  %r.null = icmp eq ptr %r, null
  %re.inc = zext i1 %r.null to i64
  %renulls.n = add nuw i64 %renulls, %re.inc
  call void @llvm.memset.p0.i64(ptr %r, i8 -1, i64 24, i1 false)
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 13
  br i1 %more3, label %realloc, label %verify

verify:                                      ; odd survivors keep tags
  call void @ut_check_eq(i64 %renulls.n, i64 0, ptr @m.realloc)
  br label %vloop

vloop:
  %v = phi i64 [ 1, %verify ], [ %v.n, %vloop ]
  %bad = phi i64 [ 0, %verify ], [ %bad.n, %vloop ]
  %slot3 = getelementptr inbounds nuw [25 x ptr], ptr @g.objs, i64 0, i64 %v
  %sp = load ptr, ptr %slot3, align 8
  %want = trunc i64 %v to i8
  %b0 = load i8, ptr %sp, align 1
  %sp23 = getelementptr inbounds nuw i8, ptr %sp, i64 23
  %b23 = load i8, ptr %sp23, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok23 = icmp eq i8 %b23, %want
  %okA = and i1 %ok0, %ok23
  %bad.b = xor i1 %okA, true
  %bad.inc = zext i1 %bad.b to i64
  %bad.n = add nuw i64 %bad, %bad.inc
  %v.n = add nuw nsw i64 %v, 2
  %more4 = icmp ult i64 %v.n, 25
  br i1 %more4, label %vloop, label %destroy.all

destroy.all:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.tags)
  call void @universe_alloc_slab_destroy(ptr %s)
  br label %done

done:
  ret void
}

define internal void @test_release_regrow() {
entry:
  %s = call ptr @universe_alloc_slab_create(i64 64, i64 8)
  %a = call ptr @universe_alloc_slab_alloc(ptr %s)
  %b = call ptr @universe_alloc_slab_alloc(ptr %s)
  call void @universe_alloc_slab_free(ptr %s, ptr %b)
  call void @universe_alloc_slab_free(ptr %s, ptr %a)
  %live = call i64 @universe_alloc_slab_live(ptr %s)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.empty)
  ; slab was released; allocator must transparently grow again
  %c = call ptr @universe_alloc_slab_alloc(ptr %s)
  %c.ok = icmp ne ptr %c, null
  call void @ut_check(i1 %c.ok, ptr @m.regrow)
  store i64 42, ptr %c, align 8
  call void @universe_alloc_slab_free(ptr %s, ptr %c)
  call void @universe_alloc_slab_destroy(ptr %s)
  call void @universe_alloc_slab_destroy(ptr null)

  %bad = call ptr @universe_alloc_slab_create(i64 -1, i64 8)
  %bad.null = icmp eq ptr %bad, null
  call void @ut_check(i1 %bad.null, ptr @m.ovf)
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op).
; 17 reps per op; rep 0 is warm-up (discarded), reps 1..16 recorded.
; ops_per_rep = 100k alloc/free pairs.
define internal void @bench() {
entry:
  %s = call ptr @universe_alloc_slab_create(i64 64, i64 256)
  br label %sl.rep

sl.rep:
  %sl.r = phi i64 [ 0, %entry ], [ %sl.r.n, %sl.next ]
  %sl.t0 = call double @ut_now_sec()
  br label %sloop

sloop:
  %i = phi i64 [ 0, %sl.rep ], [ %i.n, %sloop ]
  %p1 = call ptr @universe_alloc_slab_alloc(ptr %s)
  %p2 = call ptr @universe_alloc_slab_alloc(ptr %s)
  store i64 %i, ptr %p1, align 8
  store i64 %i, ptr %p2, align 8
  call void @universe_alloc_slab_free(ptr %s, ptr %p2)
  call void @universe_alloc_slab_free(ptr %s, ptr %p1)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %sloop, label %sl.mid

sl.mid:
  %sl.t1 = call double @ut_now_sec()
  %sl.el = fsub double %sl.t1, %sl.t0
  %sl.warm = icmp eq i64 %sl.r, 0
  br i1 %sl.warm, label %sl.next, label %sl.store

sl.store:
  %sl.idx = sub i64 %sl.r, 1
  %sl.sp = getelementptr inbounds [16 x double], ptr @slab.samp, i64 0, i64 %sl.idx
  store double %sl.el, ptr %sl.sp, align 8
  br label %sl.next

sl.next:
  %sl.r.n = add nuw nsw i64 %sl.r, 1
  %sl.rmore = icmp ult i64 %sl.r.n, 17
  br i1 %sl.rmore, label %sl.rep, label %sl.done

sl.done:
  call void @universe_alloc_slab_destroy(ptr %s)
  call void @ut_report_dist(ptr @slab.samp, i64 16, i64 100000, ptr @lbl.slab)
  br label %ml.rep

ml.rep:
  %ml.r = phi i64 [ 0, %sl.done ], [ %ml.r.n, %ml.next ]
  %ml.t0 = call double @ut_now_sec()
  br label %mloop

mloop:
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
  call void @test_release_regrow()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
