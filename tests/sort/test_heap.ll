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

; Tests for universe_sort_heap: qsort cross-check, edges, >1KiB elements,
; errors, --bench vs libc qsort.

declare i32 @universe_sort_heap(ptr, i64, i64, ptr)

declare void @qsort(ptr, i64, i64, ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@g.a      = internal global [8192 x i32] zeroinitializer, align 16
@g.ref    = internal global [8192 x i32] zeroinitializer, align 16
@g.big    = internal global [96256 x i8] zeroinitializer, align 16 ; 64 x 1504
@g.master = internal global [100000 x i32] zeroinitializer, align 16
@g.work   = internal global [100000 x i32] zeroinitializer, align 16

@m.null    = private unnamed_addr constant [19 x i8] c"null args rejected\00"
@m.trivial = private unnamed_addr constant [21 x i8] c"trivial sizes no-ops\00"
@m.random  = private unnamed_addr constant [24 x i8] c"random ints match qsort\00"
@m.sorted  = private unnamed_addr constant [22 x i8] c"presorted stays sorte\00"
@m.reverse = private unnamed_addr constant [18 x i8] c"reverse sorted ok\00"
@m.big.ord = private unnamed_addr constant [18 x i8] c"big elems ordered\00"
@m.big.pay = private unnamed_addr constant [25 x i8] c"big elems payload intact\00"
@heap.samp = internal global [16 x double] zeroinitializer, align 8
@heapq.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.heap = private unnamed_addr constant [22 x i8] c"heapsort i32 n=100000\00"
@lbl.heapq = private unnamed_addr constant [19 x i8] c"qsort i32 n=100000\00"

define i32 @cmp_i32(ptr %a, ptr %b) {
entry:
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %gt = icmp sgt i32 %x, %y
  %lt = icmp slt i32 %x, %y
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

define internal void @test_errors() {
entry:
  %v = alloca [2 x i32], align 8
  store i32 2, ptr %v, align 8
  %rc1 = call i32 @universe_sort_heap(ptr null, i64 2, i64 4, ptr @cmp_i32)
  %rc2 = call i32 @universe_sort_heap(ptr nonnull %v, i64 2, i64 4, ptr null)
  %s = add nuw i32 %rc1, %rc2
  %s.w = zext i32 %s to i64
  call void @ut_check_eq(i64 %s.w, i64 2, ptr @m.null)
  %rc3 = call i32 @universe_sort_heap(ptr nonnull %v, i64 0, i64 4, ptr @cmp_i32)
  %rc4 = call i32 @universe_sort_heap(ptr nonnull %v, i64 2, i64 0, ptr @cmp_i32)
  %t = add nuw i32 %rc3, %rc4
  %t.w = zext i32 %t to i64
  call void @ut_check_eq(i64 %t.w, i64 0, ptr @m.trivial)
  ret void
}

define internal void @test_random() {
entry:
  %seed = alloca i64, align 8
  store i64 42, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %v = trunc i64 %r to i32
  %a.p = getelementptr inbounds nuw [8192 x i32], ptr @g.a, i64 0, i64 %i
  %ref.p = getelementptr inbounds nuw [8192 x i32], ptr @g.ref, i64 0, i64 %i
  store i32 %v, ptr %a.p, align 4
  store i32 %v, ptr %ref.p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 8192
  br i1 %more, label %fill, label %run

run:
  %rc = call i32 @universe_sort_heap(ptr @g.a, i64 8192, i64 4, ptr @cmp_i32)
  call void @qsort(ptr @g.ref, i64 8192, i64 4, ptr @cmp_i32)
  %mc = call i32 @memcmp(ptr @g.a, ptr @g.ref, i64 32768)
  %mc.w = sext i32 %mc to i64
  call void @ut_check_eq(i64 %mc.w, i64 0, ptr @m.random)
  ret void
}

define internal void @test_edges() {
entry:
  br label %fill.asc

fill.asc:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill.asc ]
  %v = trunc i64 %i to i32
  %p = getelementptr inbounds nuw [8192 x i32], ptr @g.a, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 1000
  br i1 %c, label %fill.asc, label %sort.asc

sort.asc:
  %rc0 = call i32 @universe_sort_heap(ptr @g.a, i64 1000, i64 4, ptr @cmp_i32)
  br label %chk.asc

chk.asc:
  %j = phi i64 [ 0, %sort.asc ], [ %j.n, %chk.asc ]
  %bad = phi i64 [ 0, %sort.asc ], [ %bad.n, %chk.asc ]
  %p2 = getelementptr inbounds nuw [8192 x i32], ptr @g.a, i64 0, i64 %j
  %got = load i32, ptr %p2, align 4
  %want = trunc i64 %j to i32
  %ne = icmp ne i32 %got, %want
  %inc = zext i1 %ne to i64
  %bad.n = add nuw i64 %bad, %inc
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 1000
  br i1 %c2, label %chk.asc, label %rep.asc

rep.asc:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.sorted)
  br label %fill.desc

fill.desc:
  %k = phi i64 [ 0, %rep.asc ], [ %k.n, %fill.desc ]
  %dv64 = sub nsw i64 999, %k
  %dv = trunc i64 %dv64 to i32
  %p3 = getelementptr inbounds nuw [8192 x i32], ptr @g.a, i64 0, i64 %k
  store i32 %dv, ptr %p3, align 4
  %k.n = add nuw nsw i64 %k, 1
  %c3 = icmp ult i64 %k.n, 1000
  br i1 %c3, label %fill.desc, label %sort.desc

sort.desc:
  %rc1 = call i32 @universe_sort_heap(ptr @g.a, i64 1000, i64 4, ptr @cmp_i32)
  br label %chk.desc

chk.desc:
  %m = phi i64 [ 0, %sort.desc ], [ %m.n, %chk.desc ]
  %bad2 = phi i64 [ 0, %sort.desc ], [ %bad2.n, %chk.desc ]
  %p4 = getelementptr inbounds nuw [8192 x i32], ptr @g.a, i64 0, i64 %m
  %got2 = load i32, ptr %p4, align 4
  %want2 = trunc i64 %m to i32
  %ne2 = icmp ne i32 %got2, %want2
  %inc2 = zext i1 %ne2 to i64
  %bad2.n = add nuw i64 %bad2, %inc2
  %m.n = add nuw nsw i64 %m, 1
  %c4 = icmp ult i64 %m.n, 1000
  br i1 %c4, label %chk.desc, label %rep.desc

rep.desc:
  call void @ut_check_eq(i64 %bad2.n, i64 0, ptr @m.reverse)
  ret void
}

define internal void @test_big() {
entry:
  %seed = alloca i64, align 8
  store i64 99, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %key = urem i64 %r, 32
  %key32 = trunc nuw i64 %key to i32
  %off = mul nuw nsw i64 %i, 1504
  %p = getelementptr inbounds nuw i8, ptr @g.big, i64 %off
  store i32 %key32, ptr %p, align 4
  %pad.p = getelementptr inbounds nuw i8, ptr %p, i64 4
  %fv64 = add nuw i64 %key, 1
  %fv = trunc i64 %fv64 to i8
  call void @llvm.memset.p0.i64(ptr %pad.p, i8 %fv, i64 1500, i1 false)
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 64
  br i1 %c, label %fill, label %run

run:
  %rc = call i32 @universe_sort_heap(ptr @g.big, i64 64, i64 1504, ptr @cmp_i32)
  br label %chk

chk:
  %j = phi i64 [ 0, %run ], [ %j.n, %chk.latch ]
  %bad.ord = phi i64 [ 0, %run ], [ %bad.ord.n, %chk.latch ]
  %bad.pay = phi i64 [ 0, %run ], [ %bad.pay.n, %chk.latch ]
  %off.j = mul nuw nsw i64 %j, 1504
  %pj = getelementptr inbounds nuw i8, ptr @g.big, i64 %off.j
  %kj = load i32, ptr %pj, align 4
  %first = icmp eq i64 %j, 0
  br i1 %first, label %pay, label %ord

ord:
  %off.prev = sub nuw i64 %off.j, 1504
  %pprev = getelementptr inbounds nuw i8, ptr @g.big, i64 %off.prev
  %kprev = load i32, ptr %pprev, align 4
  %viol = icmp sgt i32 %kprev, %kj
  br label %pay

pay:
  %ord.viol = phi i1 [ false, %chk ], [ %viol, %ord ]
  %ord.inc = zext i1 %ord.viol to i64
  %bad.ord.n = add nuw i64 %bad.ord, %ord.inc
  %want.v = add i32 %kj, 1
  %want8 = trunc i32 %want.v to i8
  %pad0.p = getelementptr inbounds nuw i8, ptr %pj, i64 4
  %padN.p = getelementptr inbounds nuw i8, ptr %pj, i64 1503
  %pad0 = load i8, ptr %pad0.p, align 1
  %padN = load i8, ptr %padN.p, align 1
  %ok0 = icmp eq i8 %pad0, %want8
  %okN = icmp eq i8 %padN, %want8
  %okpay = and i1 %ok0, %okN
  %pay.b = xor i1 %okpay, true
  %pay.inc = zext i1 %pay.b to i64
  %bad.pay.n = add nuw i64 %bad.pay, %pay.inc
  br label %chk.latch

chk.latch:
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 64
  br i1 %c2, label %chk, label %rep

rep:
  call void @ut_check_eq(i64 %bad.ord.n, i64 0, ptr @m.big.ord)
  call void @ut_check_eq(i64 %bad.pay.n, i64 0, ptr @m.big.pay)
  ret void
}

define internal void @bench() {
entry:
  %seed = alloca i64, align 8
  store i64 1, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %v = trunc i64 %r to i32
  %p = getelementptr inbounds nuw [100000 x i32], ptr @g.master, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 100000
  br i1 %c, label %fill, label %run.u

run.u:
  ; heapsort: 17 reps of (refill 100000 keys, time one sort); discard rep 0
  ; (warm-up), report over the remaining 16. The sort is destructive, so each
  ; rep re-copies the master. ops/rep = 100000 (ns per element).
  br label %hu.rep
hu.rep:
  %hrep = phi i64 [ 0, %run.u ], [ %hrep.n, %hu.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 400000, i1 false)
  %t0 = call double @ut_now_sec()
  %rc = call i32 @universe_sort_heap(ptr @g.work, i64 100000, i64 4, ptr @cmp_i32)
  %t1 = call double @ut_now_sec()
  %hel = fsub double %t1, %t0
  %hkeep = icmp ugt i64 %hrep, 0
  br i1 %hkeep, label %hu.store, label %hu.next
hu.store:
  %hidx = sub i64 %hrep, 1
  %hsp = getelementptr inbounds [16 x double], ptr @heap.samp, i64 0, i64 %hidx
  store double %hel, ptr %hsp, align 8
  br label %hu.next
hu.next:
  %hrep.n = add nuw i64 %hrep, 1
  %hmore = icmp ult i64 %hrep.n, 17
  br i1 %hmore, label %hu.rep, label %hu.report
hu.report:
  call void @ut_report_dist(ptr @heap.samp, i64 16, i64 100000, ptr @lbl.heap)
  br label %hq.rep
hq.rep:
  %qrep = phi i64 [ 0, %hu.report ], [ %qrep.n, %hq.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 400000, i1 false)
  %t2 = call double @ut_now_sec()
  call void @qsort(ptr @g.work, i64 100000, i64 4, ptr @cmp_i32)
  %t3 = call double @ut_now_sec()
  %qel = fsub double %t3, %t2
  %qkeep = icmp ugt i64 %qrep, 0
  br i1 %qkeep, label %hq.store, label %hq.next
hq.store:
  %qidx = sub i64 %qrep, 1
  %qsp = getelementptr inbounds [16 x double], ptr @heapq.samp, i64 0, i64 %qidx
  store double %qel, ptr %qsp, align 8
  br label %hq.next
hq.next:
  %qrep.n = add nuw i64 %qrep, 1
  %qmore = icmp ult i64 %qrep.n, 17
  br i1 %qmore, label %hq.rep, label %hq.report
hq.report:
  call void @ut_report_dist(ptr @heapq.samp, i64 16, i64 100000, ptr @lbl.heapq)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_errors()
  call void @test_random()
  call void @test_edges()
  call void @test_big()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
