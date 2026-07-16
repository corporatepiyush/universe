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

; Tests for universe_alloc_buddy. Crown invariant: drain the whole arena as
; min blocks, free them all, then a single whole-arena alloc must succeed —
; proving coalescing merged everything back. Plus overlap stamps, exhaustion,
; random sized churn, --bench.

declare ptr @universe_alloc_buddy_create(i64, i64)
declare ptr @universe_alloc_buddy_alloc(ptr, i64)
declare void @universe_alloc_buddy_free(ptr, ptr, i64)
declare i64 @universe_alloc_buddy_live(ptr)
declare void @universe_alloc_buddy_destroy(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@g.blocks = internal global [4096 x ptr] zeroinitializer, align 16

@buddy.samp  = internal global [16 x double] zeroinitializer, align 8
@malloc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.buddy   = private unnamed_addr constant [25 x i8] c"buddy alloc/free 64+512B\00"
@lbl.malloc  = private unnamed_addr constant [26 x i8] c"malloc alloc/free 64+512B\00"

@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.drain    = private unnamed_addr constant [22 x i8] c"drain 4096 min blocks\00"
@m.distinct = private unnamed_addr constant [21 x i8] c"blocks carry stamps \00"
@m.exhaust  = private unnamed_addr constant [17 x i8] c"exhausted null  \00"
@m.coalesce = private unnamed_addr constant [29 x i8] c"full coalesce after free-all\00"
@m.live0    = private unnamed_addr constant [7 x i8] c"live 0\00"
@m.mixed    = private unnamed_addr constant [26 x i8] c"mixed sizes stay in range\00"
@m.churn    = private unnamed_addr constant [23 x i8] c"random churn ends live\00"

define internal void @test_drain_coalesce() {
entry:                                       ; 256 KiB arena of 64B mins
  %b = call ptr @universe_alloc_buddy_create(i64 262144, i64 64)
  %ok = icmp ne ptr %b, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %drain, label %done

drain:                                       ; all 4096 blocks
  %i = phi i64 [ 0, %entry ], [ %i.n, %drain ]
  %nulls = phi i64 [ 0, %entry ], [ %nulls.n, %drain ]
  %p = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 64)
  %slot = getelementptr inbounds nuw [4096 x ptr], ptr @g.blocks, i64 0, i64 %i
  store ptr %p, ptr %slot, align 8
  %isnull = icmp eq ptr %p, null
  %n.inc = zext i1 %isnull to i64
  %nulls.n = add nuw i64 %nulls, %n.inc
  %tag = trunc i64 %i to i8
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 64, i1 false)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 4096
  br i1 %more, label %drain, label %full

full:
  call void @ut_check_eq(i64 %nulls.n, i64 0, ptr @m.drain)
  %extra = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 64)
  %extra.null = icmp eq ptr %extra, null
  call void @ut_check(i1 %extra.null, ptr @m.exhaust)
  br label %verify

verify:                                      ; stamps intact = no overlap
  %j = phi i64 [ 0, %full ], [ %j.n, %verify ]
  %bad = phi i64 [ 0, %full ], [ %bad.n, %verify ]
  %slot2 = getelementptr inbounds nuw [4096 x ptr], ptr @g.blocks, i64 0, i64 %j
  %q = load ptr, ptr %slot2, align 8
  %want = trunc i64 %j to i8
  %b0 = load i8, ptr %q, align 1
  %q63 = getelementptr inbounds nuw i8, ptr %q, i64 63
  %b63 = load i8, ptr %q63, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok63 = icmp eq i8 %b63, %want
  %okA = and i1 %ok0, %ok63
  %bad.b = xor i1 %okA, true
  %bad.inc = zext i1 %bad.b to i64
  %bad.n = add nuw i64 %bad, %bad.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 4096
  br i1 %more2, label %verify, label %freeall

freeall:
  %k = phi i64 [ 0, %verify ], [ %k.n, %freeall ]
  %slot3 = getelementptr inbounds nuw [4096 x ptr], ptr @g.blocks, i64 0, i64 %k
  %fq = load ptr, ptr %slot3, align 8
  call void @universe_alloc_buddy_free(ptr %b, ptr %fq, i64 64)
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 4096
  br i1 %more3, label %freeall, label %grand

grand:                                       ; the coalescing proof
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.distinct)
  %live = call i64 @universe_alloc_buddy_live(ptr %b)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  %whole = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 262144)
  %whole.ok = icmp ne ptr %whole, null
  call void @ut_check(i1 %whole.ok, ptr @m.coalesce)
  call void @universe_alloc_buddy_free(ptr %b, ptr %whole, i64 262144)
  call void @universe_alloc_buddy_destroy(ptr %b)
  br label %done

done:
  ret void
}

define internal void @test_mixed() {
entry:                                       ; mixed sizes land inside arena
  %b = call ptr @universe_alloc_buddy_create(i64 1048576, i64 64)
  %p1 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 100)     ; ->128
  %p2 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 4000)    ; ->4096
  %p3 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 65536)
  %p4 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 1)       ; ->64
  %n1 = icmp ne ptr %p1, null
  %n2 = icmp ne ptr %p2, null
  %n3 = icmp ne ptr %p3, null
  %n4 = icmp ne ptr %p4, null
  %a1 = and i1 %n1, %n2
  %a2 = and i1 %a1, %n3
  %a3 = and i1 %a2, %n4
  ; write through them fully (would crash/corrupt if overlapping badly)
  call void @llvm.memset.p0.i64(ptr %p1, i8 1, i64 100, i1 false)
  call void @llvm.memset.p0.i64(ptr %p2, i8 2, i64 4000, i1 false)
  call void @llvm.memset.p0.i64(ptr %p3, i8 3, i64 65536, i1 false)
  call void @llvm.memset.p0.i64(ptr %p4, i8 4, i64 1, i1 false)
  %c1 = load i8, ptr %p1, align 1
  %c2 = load i8, ptr %p2, align 1
  %c3 = load i8, ptr %p3, align 1
  %c4 = load i8, ptr %p4, align 1
  %k1 = icmp eq i8 %c1, 1
  %k2 = icmp eq i8 %c2, 2
  %k3 = icmp eq i8 %c3, 3
  %k4 = icmp eq i8 %c4, 4
  %x1 = and i1 %k1, %k2
  %x2 = and i1 %x1, %k3
  %x3 = and i1 %x2, %k4
  %all = and i1 %a3, %x3
  call void @ut_check(i1 %all, ptr @m.mixed)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p2, i64 4000)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p4, i64 1)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p1, i64 100)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p3, i64 65536)
  %live = call i64 @universe_alloc_buddy_live(ptr %b)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  ; whole-arena alloc proves coalescing again
  %whole = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 1048576)
  %whole.ok = icmp ne ptr %whole, null
  call void @ut_check(i1 %whole.ok, ptr @m.coalesce)
  call void @universe_alloc_buddy_destroy(ptr %b)
  ret void
}

define internal void @test_churn() {
entry:                                       ; random size alloc/free pairs
  %b = call ptr @universe_alloc_buddy_create(i64 1048576, i64 64)
  %seed = alloca i64, align 8
  store i64 123, ptr %seed, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %szr = and i64 %r, 8191
  %sz = or i64 %szr, 1
  %p = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 %sz)
  store i64 %i, ptr %p, align 8
  call void @universe_alloc_buddy_free(ptr %b, ptr %p, i64 %sz)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 50000
  br i1 %more, label %loop, label %done

done:
  %live = call i64 @universe_alloc_buddy_live(ptr %b)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.churn)
  %whole = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 1048576)
  %whole.ok = icmp ne ptr %whole, null
  call void @ut_check(i1 %whole.ok, ptr @m.coalesce)
  call void @universe_alloc_buddy_destroy(ptr %b)
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op).
; 17 reps per op; rep 0 is warm-up (discarded), reps 1..16 recorded.
; ops_per_rep = 100k alloc/free pairs (each 64B + 512B).
define internal void @bench() {
entry:
  %b = call ptr @universe_alloc_buddy_create(i64 16777216, i64 64)
  br label %bd.rep

bd.rep:
  %bd.r = phi i64 [ 0, %entry ], [ %bd.r.n, %bd.next ]
  %bd.t0 = call double @ut_now_sec()
  br label %bloop

bloop:
  %i = phi i64 [ 0, %bd.rep ], [ %i.n, %bloop ]
  %p1 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 64)
  %p2 = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 512)
  store i64 %i, ptr %p1, align 8
  store i64 %i, ptr %p2, align 8
  call void @universe_alloc_buddy_free(ptr %b, ptr %p2, i64 512)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p1, i64 64)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %bloop, label %bd.mid

bd.mid:
  %bd.t1 = call double @ut_now_sec()
  %bd.el = fsub double %bd.t1, %bd.t0
  %bd.warm = icmp eq i64 %bd.r, 0
  br i1 %bd.warm, label %bd.next, label %bd.store

bd.store:
  %bd.idx = sub i64 %bd.r, 1
  %bd.sp = getelementptr inbounds [16 x double], ptr @buddy.samp, i64 0, i64 %bd.idx
  store double %bd.el, ptr %bd.sp, align 8
  br label %bd.next

bd.next:
  %bd.r.n = add nuw nsw i64 %bd.r, 1
  %bd.rmore = icmp ult i64 %bd.r.n, 17
  br i1 %bd.rmore, label %bd.rep, label %bd.done

bd.done:
  call void @universe_alloc_buddy_destroy(ptr %b)
  call void @ut_report_dist(ptr @buddy.samp, i64 16, i64 100000, ptr @lbl.buddy)
  br label %ml.rep

ml.rep:
  %ml.r = phi i64 [ 0, %bd.done ], [ %ml.r.n, %ml.next ]
  %ml.t0 = call double @ut_now_sec()
  br label %mloop

mloop:
  %j = phi i64 [ 0, %ml.rep ], [ %j.n, %mloop ]
  %q1 = call ptr @malloc(i64 64)
  %q2 = call ptr @malloc(i64 512)
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
  call void @test_drain_coalesce()
  call void @test_mixed()
  call void @test_churn()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
