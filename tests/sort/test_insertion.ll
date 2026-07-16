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

; Tests for universe_sort_insertion. Pure-IR test driver: errors, qsort
; cross-check on fixed-seed random data, edges (sorted/reverse/all-equal),
; stability, >1KiB elements (chunked-swap path), --bench mode.

; ---- external contracts ---------------------------------------------------

declare i32 @universe_sort_insertion(ptr, i64, i64, ptr)

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

; ---- test data ------------------------------------------------------------

@g.a      = internal global [4096 x i32] zeroinitializer, align 16
@g.ref    = internal global [4096 x i32] zeroinitializer, align 16
@g.b      = internal global [513 x i64] zeroinitializer, align 16
@g.pairs  = internal global [2048 x i64] zeroinitializer, align 16
@g.big    = internal global [96256 x i8] zeroinitializer, align 16 ; 64 x 1504
@g.master = internal global [20000 x i32] zeroinitializer, align 16
@g.work   = internal global [20000 x i32] zeroinitializer, align 16

@m.null      = private unnamed_addr constant [19 x i8] c"null args rejected\00"
@m.trivial   = private unnamed_addr constant [25 x i8] c"trivial sizes are no-ops\00"
@m.untouched = private unnamed_addr constant [28 x i8] c"count<2 must not touch data\00"
@m.sortrc    = private unnamed_addr constant [16 x i8] c"sort returns OK\00"
@m.random    = private unnamed_addr constant [24 x i8] c"random ints match qsort\00"
@m.sorted    = private unnamed_addr constant [23 x i8] c"presorted stays sorted\00"
@m.reverse   = private unnamed_addr constant [18 x i8] c"reverse sorted ok\00"
@m.equal     = private unnamed_addr constant [20 x i8] c"all-equal untouched\00"
@m.stab.ord  = private unnamed_addr constant [24 x i8] c"stability: keys ordered\00"
@m.stab.seq  = private unnamed_addr constant [25 x i8] c"stability: seq preserved\00"
@m.big.rc    = private unnamed_addr constant [13 x i8] c"big elems rc\00"
@m.big.ord   = private unnamed_addr constant [18 x i8] c"big elems ordered\00"
@m.big.pay   = private unnamed_addr constant [25 x i8] c"big elems payload intact\00"
@ins.samp = internal global [16 x double] zeroinitializer, align 8
@insq.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.ins = private unnamed_addr constant [22 x i8] c"insertion i32 n=20000\00"
@lbl.insq = private unnamed_addr constant [18 x i8] c"qsort i32 n=20000\00"

; ---- comparators ----------------------------------------------------------

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

define i32 @cmp_u64(ptr %a, ptr %b) {
entry:
  %x = load i64, ptr %a, align 8
  %y = load i64, ptr %b, align 8
  %gt = icmp ugt i64 %x, %y
  %lt = icmp ult i64 %x, %y
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

; pair packed in i64: key = low 32 bits, seq = high 32. Compare key ONLY.
define i32 @cmp_pair_key(ptr %a, ptr %b) {
entry:
  %x = load i64, ptr %a, align 8
  %y = load i64, ptr %b, align 8
  %kx = and i64 %x, 4294967295
  %ky = and i64 %y, 4294967295
  %gt = icmp ugt i64 %kx, %ky
  %lt = icmp ult i64 %kx, %ky
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

; ---- error / trivial cases ------------------------------------------------

define internal void @test_errors() {
entry:
  %v = alloca [2 x i32], align 8
  store i32 2, ptr %v, align 8
  %v1 = getelementptr inbounds nuw i8, ptr %v, i64 4
  store i32 1, ptr %v1, align 4

  %rc1 = call i32 @universe_sort_insertion(ptr null, i64 2, i64 4, ptr @cmp_i32)
  %rc1.w = zext i32 %rc1 to i64
  call void @ut_check_eq(i64 %rc1.w, i64 1, ptr @m.null)

  %rc2 = call i32 @universe_sort_insertion(ptr nonnull %v, i64 2, i64 4, ptr null)
  %rc2.w = zext i32 %rc2 to i64
  call void @ut_check_eq(i64 %rc2.w, i64 1, ptr @m.null)

  %rc3 = call i32 @universe_sort_insertion(ptr nonnull %v, i64 0, i64 4, ptr @cmp_i32)
  %rc4 = call i32 @universe_sort_insertion(ptr nonnull %v, i64 1, i64 4, ptr @cmp_i32)
  %rc5 = call i32 @universe_sort_insertion(ptr nonnull %v, i64 2, i64 0, ptr @cmp_i32)
  %s34 = or i32 %rc3, %rc4
  %s = or i32 %s34, %rc5
  %s.w = zext i32 %s to i64
  call void @ut_check_eq(i64 %s.w, i64 0, ptr @m.trivial)

  %a0 = load i32, ptr %v, align 8
  %a1 = load i32, ptr %v1, align 4
  %k0 = icmp eq i32 %a0, 2
  %k1 = icmp eq i32 %a1, 1
  %keep = and i1 %k0, %k1
  call void @ut_check(i1 %keep, ptr @m.untouched)
  ret void
}

; ---- random ints vs libc qsort --------------------------------------------

define internal void @test_random() {
entry:
  %seed = alloca i64, align 8
  store i64 42, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.next, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %m = urem i64 %r, 1000
  %v = trunc i64 %m to i32
  %a.p = getelementptr inbounds nuw [4096 x i32], ptr @g.a, i64 0, i64 %i
  %ref.p = getelementptr inbounds nuw [4096 x i32], ptr @g.ref, i64 0, i64 %i
  store i32 %v, ptr %a.p, align 4
  store i32 %v, ptr %ref.p, align 4
  %i.next = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.next, 4096
  br i1 %more, label %fill, label %run

run:
  %rc = call i32 @universe_sort_insertion(ptr @g.a, i64 4096, i64 4, ptr @cmp_i32)
  %rc.w = zext i32 %rc to i64
  call void @ut_check_eq(i64 %rc.w, i64 0, ptr @m.sortrc)
  call void @qsort(ptr @g.ref, i64 4096, i64 4, ptr @cmp_i32)
  %mc = call i32 @memcmp(ptr @g.a, ptr @g.ref, i64 16384)
  %mc.w = sext i32 %mc to i64
  call void @ut_check_eq(i64 %mc.w, i64 0, ptr @m.random)
  ret void
}

; ---- edges: presorted / reverse / all-equal (513 x u64) --------------------

define internal void @test_edges() {
entry:
  br label %fill.asc

fill.asc:
  %i0 = phi i64 [ 0, %entry ], [ %i0.n, %fill.asc ]
  %p0 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i0
  store i64 %i0, ptr %p0, align 8
  %i0.n = add nuw nsw i64 %i0, 1
  %c0 = icmp ult i64 %i0.n, 513
  br i1 %c0, label %fill.asc, label %sort.asc

sort.asc:
  %rc0 = call i32 @universe_sort_insertion(ptr @g.b, i64 513, i64 8, ptr @cmp_u64)
  br label %chk.asc

chk.asc:
  %i1 = phi i64 [ 0, %sort.asc ], [ %i1.n, %chk.asc ]
  %bad1 = phi i64 [ 0, %sort.asc ], [ %bad1.n, %chk.asc ]
  %p1 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i1
  %v1 = load i64, ptr %p1, align 8
  %ne1 = icmp ne i64 %v1, %i1
  %inc1 = zext i1 %ne1 to i64
  %bad1.n = add nuw i64 %bad1, %inc1
  %i1.n = add nuw nsw i64 %i1, 1
  %c1 = icmp ult i64 %i1.n, 513
  br i1 %c1, label %chk.asc, label %rep.asc

rep.asc:
  call void @ut_check_eq(i64 %bad1.n, i64 0, ptr @m.sorted)
  br label %fill.desc

fill.desc:
  %i2 = phi i64 [ 0, %rep.asc ], [ %i2.n, %fill.desc ]
  %p2 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i2
  %dv = sub nuw i64 513, %i2
  store i64 %dv, ptr %p2, align 8
  %i2.n = add nuw nsw i64 %i2, 1
  %c2 = icmp ult i64 %i2.n, 513
  br i1 %c2, label %fill.desc, label %sort.desc

sort.desc:
  %rc1 = call i32 @universe_sort_insertion(ptr @g.b, i64 513, i64 8, ptr @cmp_u64)
  br label %chk.desc

chk.desc:
  %i3 = phi i64 [ 0, %sort.desc ], [ %i3.n, %chk.desc ]
  %bad3 = phi i64 [ 0, %sort.desc ], [ %bad3.n, %chk.desc ]
  %p3 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i3
  %v3 = load i64, ptr %p3, align 8
  %want3 = add nuw i64 %i3, 1
  %ne3 = icmp ne i64 %v3, %want3
  %inc3 = zext i1 %ne3 to i64
  %bad3.n = add nuw i64 %bad3, %inc3
  %i3.n = add nuw nsw i64 %i3, 1
  %c3 = icmp ult i64 %i3.n, 513
  br i1 %c3, label %chk.desc, label %rep.desc

rep.desc:
  call void @ut_check_eq(i64 %bad3.n, i64 0, ptr @m.reverse)
  br label %fill.eq

fill.eq:
  %i4 = phi i64 [ 0, %rep.desc ], [ %i4.n, %fill.eq ]
  %p4 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i4
  store i64 7, ptr %p4, align 8
  %i4.n = add nuw nsw i64 %i4, 1
  %c4 = icmp ult i64 %i4.n, 513
  br i1 %c4, label %fill.eq, label %sort.eq

sort.eq:
  %rc2 = call i32 @universe_sort_insertion(ptr @g.b, i64 513, i64 8, ptr @cmp_u64)
  br label %chk.eq

chk.eq:
  %i5 = phi i64 [ 0, %sort.eq ], [ %i5.n, %chk.eq ]
  %bad5 = phi i64 [ 0, %sort.eq ], [ %bad5.n, %chk.eq ]
  %p5 = getelementptr inbounds nuw [513 x i64], ptr @g.b, i64 0, i64 %i5
  %v5 = load i64, ptr %p5, align 8
  %ne5 = icmp ne i64 %v5, 7
  %inc5 = zext i1 %ne5 to i64
  %bad5.n = add nuw i64 %bad5, %inc5
  %i5.n = add nuw nsw i64 %i5, 1
  %c5 = icmp ult i64 %i5.n, 513
  br i1 %c5, label %chk.eq, label %rep.eq

rep.eq:
  call void @ut_check_eq(i64 %bad5.n, i64 0, ptr @m.equal)
  ret void
}

; ---- stability: many duplicate keys, seq must stay in order ----------------

define internal void @test_stability() {
entry:
  %seed = alloca i64, align 8
  store i64 7, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %key = urem i64 %r, 16
  %seq = shl nuw i64 %i, 32
  %pair = or disjoint i64 %key, %seq
  %p = getelementptr inbounds nuw [2048 x i64], ptr @g.pairs, i64 0, i64 %i
  store i64 %pair, ptr %p, align 8
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 2048
  br i1 %c, label %fill, label %run

run:
  %rc = call i32 @universe_sort_insertion(ptr @g.pairs, i64 2048, i64 8, ptr @cmp_pair_key)
  br label %chk

chk:
  %j = phi i64 [ 1, %run ], [ %j.n, %chk ]
  %bad.ord = phi i64 [ 0, %run ], [ %bad.ord.n, %chk ]
  %bad.seq = phi i64 [ 0, %run ], [ %bad.seq.n, %chk ]
  %j.prev = add nsw i64 %j, -1
  %pp = getelementptr inbounds nuw [2048 x i64], ptr @g.pairs, i64 0, i64 %j.prev
  %cp = getelementptr inbounds nuw [2048 x i64], ptr @g.pairs, i64 0, i64 %j
  %prev = load i64, ptr %pp, align 8
  %cur = load i64, ptr %cp, align 8
  %pk = and i64 %prev, 4294967295
  %ck = and i64 %cur, 4294967295
  %ord.viol = icmp ugt i64 %pk, %ck
  %ord.inc = zext i1 %ord.viol to i64
  %bad.ord.n = add nuw i64 %bad.ord, %ord.inc
  %same = icmp eq i64 %pk, %ck
  %ps = lshr i64 %prev, 32
  %cs = lshr i64 %cur, 32
  %seq.rev = icmp uge i64 %ps, %cs
  %seq.viol = and i1 %same, %seq.rev
  %seq.inc = zext i1 %seq.viol to i64
  %bad.seq.n = add nuw i64 %bad.seq, %seq.inc
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 2048
  br i1 %c2, label %chk, label %rep

rep:
  call void @ut_check_eq(i64 %bad.ord.n, i64 0, ptr @m.stab.ord)
  call void @ut_check_eq(i64 %bad.seq.n, i64 0, ptr @m.stab.seq)
  ret void
}

; ---- >1KiB elements: exercises the chunked-swap path -----------------------
; element = <{ i32 key, [1500 x i8] pad }>, size 1504; pad filled with key+1.

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
  %fill.v64 = add nuw i64 %key, 1
  %fill.v = trunc i64 %fill.v64 to i8
  call void @llvm.memset.p0.i64(ptr %pad.p, i8 %fill.v, i64 1500, i1 false)
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 64
  br i1 %c, label %fill, label %run

run:
  %rc = call i32 @universe_sort_insertion(ptr @g.big, i64 64, i64 1504, ptr @cmp_i32)
  %rc.w = zext i32 %rc to i64
  call void @ut_check_eq(i64 %rc.w, i64 0, ptr @m.big.rc)
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
  %pay.inc.b = xor i1 %okpay, true
  %pay.inc = zext i1 %pay.inc.b to i64
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

; ---- bench: universe insertion vs libc qsort (scale reference) -------------

define internal void @bench() {
entry:
  %seed = alloca i64, align 8
  store i64 1, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %v = trunc i64 %r to i32
  %p = getelementptr inbounds nuw [20000 x i32], ptr @g.master, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 20000
  br i1 %c, label %fill, label %iu.rep

  ; insertion sort: 17 reps of (refill 20000 keys, time one sort); discard rep 0
  ; (warm-up), report over the remaining 16. Destructive, so each rep re-copies
  ; the master. ops/rep = 20000 (ns per element).
iu.rep:
  %irep = phi i64 [ 0, %fill ], [ %irep.n, %iu.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 80000, i1 false)
  %t0 = call double @ut_now_sec()
  %rc0 = call i32 @universe_sort_insertion(ptr @g.work, i64 20000, i64 4, ptr @cmp_i32)
  %t1 = call double @ut_now_sec()
  %iel = fsub double %t1, %t0
  %ikeep = icmp ugt i64 %irep, 0
  br i1 %ikeep, label %iu.store, label %iu.next
iu.store:
  %iidx = sub i64 %irep, 1
  %isp = getelementptr inbounds [16 x double], ptr @ins.samp, i64 0, i64 %iidx
  store double %iel, ptr %isp, align 8
  br label %iu.next
iu.next:
  %irep.n = add nuw i64 %irep, 1
  %imore = icmp ult i64 %irep.n, 17
  br i1 %imore, label %iu.rep, label %iu.report
iu.report:
  call void @ut_report_dist(ptr @ins.samp, i64 16, i64 20000, ptr @lbl.ins)
  br label %iq.rep
iq.rep:
  %qrep = phi i64 [ 0, %iu.report ], [ %qrep.n, %iq.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 80000, i1 false)
  %t2 = call double @ut_now_sec()
  call void @qsort(ptr @g.work, i64 20000, i64 4, ptr @cmp_i32)
  %t3 = call double @ut_now_sec()
  %qel = fsub double %t3, %t2
  %qkeep = icmp ugt i64 %qrep, 0
  br i1 %qkeep, label %iq.store, label %iq.next
iq.store:
  %qidx = sub i64 %qrep, 1
  %qsp = getelementptr inbounds [16 x double], ptr @insq.samp, i64 0, i64 %qidx
  store double %qel, ptr %qsp, align 8
  br label %iq.next
iq.next:
  %qrep.n = add nuw i64 %qrep, 1
  %qmore = icmp ult i64 %qrep.n, 17
  br i1 %qmore, label %iq.rep, label %iq.report
iq.report:
  call void @ut_report_dist(ptr @insq.samp, i64 16, i64 20000, ptr @lbl.insq)
  ret void
}

; ---------------------------------------------------------------------------

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_errors()
  call void @test_random()
  call void @test_edges()
  call void @test_stability()
  call void @test_big()
  %want.bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %want.bench, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
