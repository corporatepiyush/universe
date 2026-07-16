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

; Tests for the array-backed B-tree (universe_ds_btree_*).

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare void @qsort(ptr, i64, i64, ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare i32 @printf(ptr, ...)

declare i64 @ut_rand(ptr)
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@btf.bench.samp = internal global [16 x double] zeroinitializer, align 8
@bsr.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.btf.bench = private unnamed_addr constant [16 x i8] c"btree find 500k\00"
@lbl.bsr.bench = private unnamed_addr constant [13 x i8] c"bsearch 500k\00"

declare ptr @universe_ds_btree_create()
declare void @universe_ds_btree_destroy(ptr)
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_delete(ptr, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)
declare i32 @universe_ds_btree_contains(ptr, i64)
declare i32 @universe_ds_btree_min(ptr, ptr, ptr)
declare i32 @universe_ds_btree_max(ptr, ptr, ptr)
declare i32 @universe_ds_btree_floor(ptr, i64, ptr, ptr)
declare i32 @universe_ds_btree_ceiling(ptr, i64, ptr, ptr)
declare i64 @universe_ds_btree_range(ptr, i64, i64, ptr, ptr, i64)
declare i64 @universe_ds_btree_count(ptr)
declare i64 @universe_ds_btree_size(ptr)

@msg.count0    = private unnamed_addr constant [17 x i8] c"empty count == 0\00"
@msg.minempty  = private unnamed_addr constant [13 x i8] c"min on empty\00"
@msg.findempty = private unnamed_addr constant [14 x i8] c"find on empty\00"
@msg.cont0     = private unnamed_addr constant [14 x i8] c"contains == 0\00"
@msg.ins1      = private unnamed_addr constant [10 x i8] c"insert 42\00"
@msg.count1    = private unnamed_addr constant [11 x i8] c"count == 1\00"
@msg.find42    = private unnamed_addr constant [15 x i8] c"find 42 val 84\00"
@msg.cont1     = private unnamed_addr constant [14 x i8] c"contains == 1\00"
@msg.ov        = private unnamed_addr constant [17 x i8] c"overwrite val 99\00"
@msg.floor     = private unnamed_addr constant [11 x i8] c"floor case\00"
@msg.ceil      = private unnamed_addr constant [13 x i8] c"ceiling case\00"
@msg.minmax    = private unnamed_addr constant [12 x i8] c"min/max key\00"
@msg.del       = private unnamed_addr constant [12 x i8] c"delete case\00"
@msg.range     = private unnamed_addr constant [11 x i8] c"range case\00"

@msg.big.ins   = private unnamed_addr constant [20 x i8] c"big: insert rc == 0\00"
@msg.big.cnt   = private unnamed_addr constant [16 x i8] c"big: count == N\00"
@msg.big.find  = private unnamed_addr constant [20 x i8] c"big: find all right\00"
@msg.big.abs   = private unnamed_addr constant [18 x i8] c"big: absent == 5 \00"
@msg.big.ord   = private unnamed_addr constant [22 x i8] c"big: ordered ascend  \00"
@msg.big.ordn  = private unnamed_addr constant [19 x i8] c"big: ordered count\00"
@msg.big.qs    = private unnamed_addr constant [23 x i8] c"big: qsort cross-chk  \00"
@msg.big.subn  = private unnamed_addr constant [19 x i8] c"big: subrange cnt \00"
@msg.big.sub   = private unnamed_addr constant [19 x i8] c"big: subrange keys\00"
@msg.big.delc  = private unnamed_addr constant [21 x i8] c"big: del even rc==0 \00"
@msg.big.delcnt = private unnamed_addr constant [20 x i8] c"big: count post-del\00"
@msg.big.deven = private unnamed_addr constant [20 x i8] c"big: even gone == 5\00"
@msg.big.dodd  = private unnamed_addr constant [19 x i8] c"big: odd survives \00"
@msg.big.empty = private unnamed_addr constant [19 x i8] c"big: drained empty\00"

; signed i64 comparator for qsort
define i32 @cmp_i64(ptr %a, ptr %b) {
entry:
  %av = load i64, ptr %a, align 8
  %bv = load i64, ptr %b, align 8
  %gt = icmp sgt i64 %av, %bv
  %lt = icmp slt i64 %av, %bv
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub i32 %g, %l
  ret i32 %r
}

; ---------------------------------------------------------------------------
; edge cases: empty / single / overwrite / delete-to-empty
; ---------------------------------------------------------------------------
define void @test_edges() {
entry:
  %kslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %t = call ptr @universe_ds_btree_create()

  %c0 = call i64 @universe_ds_btree_count(ptr %t)
  call void @ut_check_eq(i64 %c0, i64 0, ptr @msg.count0)
  %m0 = call i32 @universe_ds_btree_min(ptr %t, ptr %kslot, ptr %vslot)
  %m0ok = icmp eq i32 %m0, 4
  call void @ut_check(i1 %m0ok, ptr @msg.minempty)
  %f0 = call i32 @universe_ds_btree_find(ptr %t, i64 5, ptr %vslot)
  %f0ok = icmp eq i32 %f0, 5
  call void @ut_check(i1 %f0ok, ptr @msg.findempty)
  %ct0 = call i32 @universe_ds_btree_contains(ptr %t, i64 5)
  %ct0ok = icmp eq i32 %ct0, 0
  call void @ut_check(i1 %ct0ok, ptr @msg.cont0)

  %i1 = call i32 @universe_ds_btree_insert(ptr %t, i64 42, i64 84)
  %i1ok = icmp eq i32 %i1, 0
  call void @ut_check(i1 %i1ok, ptr @msg.ins1)
  %c1 = call i64 @universe_ds_btree_count(ptr %t)
  call void @ut_check_eq(i64 %c1, i64 1, ptr @msg.count1)
  %f1 = call i32 @universe_ds_btree_find(ptr %t, i64 42, ptr %vslot)
  %v1 = load i64, ptr %vslot, align 8
  %f1rc = icmp eq i32 %f1, 0
  %f1v = icmp eq i64 %v1, 84
  %f1ok = and i1 %f1rc, %f1v
  call void @ut_check(i1 %f1ok, ptr @msg.find42)
  %ct1 = call i32 @universe_ds_btree_contains(ptr %t, i64 42)
  %ct1ok = icmp eq i32 %ct1, 1
  call void @ut_check(i1 %ct1ok, ptr @msg.cont1)

  ; overwrite
  %ow = call i32 @universe_ds_btree_insert(ptr %t, i64 42, i64 99)
  %f2 = call i32 @universe_ds_btree_find(ptr %t, i64 42, ptr %vslot)
  %v2 = load i64, ptr %vslot, align 8
  %v2ok = icmp eq i64 %v2, 99
  %c1b = call i64 @universe_ds_btree_count(ptr %t)
  %c1bok = icmp eq i64 %c1b, 1
  %owok = and i1 %v2ok, %c1bok
  call void @ut_check(i1 %owok, ptr @msg.ov)

  ; floor / ceiling on {42}
  %fl1 = call i32 @universe_ds_btree_floor(ptr %t, i64 100, ptr %kslot, ptr %vslot)
  %flk1 = load i64, ptr %kslot, align 8
  %fl1ok0 = icmp eq i32 %fl1, 0
  %fl1ok1 = icmp eq i64 %flk1, 42
  %fl1ok = and i1 %fl1ok0, %fl1ok1
  %fl2 = call i32 @universe_ds_btree_floor(ptr %t, i64 10, ptr %kslot, ptr %vslot)
  %fl2ok = icmp eq i32 %fl2, 5
  %flok = and i1 %fl1ok, %fl2ok
  call void @ut_check(i1 %flok, ptr @msg.floor)

  %ce1 = call i32 @universe_ds_btree_ceiling(ptr %t, i64 10, ptr %kslot, ptr %vslot)
  %cek1 = load i64, ptr %kslot, align 8
  %ce1ok0 = icmp eq i32 %ce1, 0
  %ce1ok1 = icmp eq i64 %cek1, 42
  %ce1ok = and i1 %ce1ok0, %ce1ok1
  %ce2 = call i32 @universe_ds_btree_ceiling(ptr %t, i64 100, ptr %kslot, ptr %vslot)
  %ce2ok = icmp eq i32 %ce2, 5
  %ceok = and i1 %ce1ok, %ce2ok
  call void @ut_check(i1 %ceok, ptr @msg.ceil)

  ; second key
  %i2 = call i32 @universe_ds_btree_insert(ptr %t, i64 10, i64 20)
  %mn = call i32 @universe_ds_btree_min(ptr %t, ptr %kslot, ptr %vslot)
  %mnk = load i64, ptr %kslot, align 8
  %mx = call i32 @universe_ds_btree_max(ptr %t, ptr %kslot, ptr %vslot)
  %mxk = load i64, ptr %kslot, align 8
  %mnok = icmp eq i64 %mnk, 10
  %mxok = icmp eq i64 %mxk, 42
  %mmok = and i1 %mnok, %mxok
  call void @ut_check(i1 %mmok, ptr @msg.minmax)

  ; deletes
  %d0 = call i32 @universe_ds_btree_delete(ptr %t, i64 999)
  %d0ok = icmp eq i32 %d0, 5
  %d1 = call i32 @universe_ds_btree_delete(ptr %t, i64 10)
  %d1ok = icmp eq i32 %d1, 0
  %f3 = call i32 @universe_ds_btree_find(ptr %t, i64 10, ptr %vslot)
  %f3ok = icmp eq i32 %f3, 5
  %d2 = call i32 @universe_ds_btree_delete(ptr %t, i64 42)
  %d2ok = icmp eq i32 %d2, 0
  %cend = call i64 @universe_ds_btree_count(ptr %t)
  %cendok = icmp eq i64 %cend, 0
  %da = and i1 %d0ok, %d1ok
  %db = and i1 %f3ok, %d2ok
  %dc = and i1 %da, %db
  %dd = and i1 %dc, %cendok
  call void @ut_check(i1 %dd, ptr @msg.del)

  call void @universe_ds_btree_destroy(ptr %t)
  ret void
}

; ---------------------------------------------------------------------------
; small deterministic: floor / ceiling / range on {10,20,...,100}
; ---------------------------------------------------------------------------
define void @test_small() {
entry:
  %kslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %ok = alloca [16 x i64], align 8
  %ov = alloca [16 x i64], align 8
  %t = call ptr @universe_ds_btree_create()
  br label %fill

fill:
  %i = phi i64 [ 1, %entry ], [ %in, %fill ]
  %key = mul i64 %i, 10
  %val = mul i64 %key, 3
  %rc = call i32 @universe_ds_btree_insert(ptr %t, i64 %key, i64 %val)
  %in = add i64 %i, 1
  %more = icmp ule i64 %in, 10
  br i1 %more, label %fill, label %checks

checks:
  ; range(25,65) -> {30,40,50,60} count 4
  %r1 = call i64 @universe_ds_btree_range(ptr %t, i64 25, i64 65, ptr %ok, ptr %ov, i64 16)
  %r1ok = icmp eq i64 %r1, 4
  %ok1p = getelementptr inbounds nuw i64, ptr %ok, i64 0
  %ok1 = load i64, ptr %ok1p, align 8
  %ok1v = icmp eq i64 %ok1, 30
  %ok4p = getelementptr inbounds nuw i64, ptr %ok, i64 3
  %ok4 = load i64, ptr %ok4p, align 8
  %ok4v = icmp eq i64 %ok4, 60
  %ra = and i1 %r1ok, %ok1v
  %rb = and i1 %ra, %ok4v
  ; range(10,100) -> 10
  %r2 = call i64 @universe_ds_btree_range(ptr %t, i64 10, i64 100, ptr %ok, ptr %ov, i64 16)
  %r2ok = icmp eq i64 %r2, 10
  ; range(0,5) -> 0
  %r3 = call i64 @universe_ds_btree_range(ptr %t, i64 0, i64 5, ptr %ok, ptr %ov, i64 16)
  %r3ok = icmp eq i64 %r3, 0
  ; range(100,200) -> {100} 1
  %r4 = call i64 @universe_ds_btree_range(ptr %t, i64 100, i64 200, ptr %ok, ptr %ov, i64 16)
  %r4ok = icmp eq i64 %r4, 1
  %rc1 = and i1 %rb, %r2ok
  %rc2 = and i1 %rc1, %r3ok
  %rc3 = and i1 %rc2, %r4ok
  call void @ut_check(i1 %rc3, ptr @msg.range)

  ; floor(35)=30, floor(10)=10, floor(5)=none
  %fl1 = call i32 @universe_ds_btree_floor(ptr %t, i64 35, ptr %kslot, ptr %vslot)
  %flk1 = load i64, ptr %kslot, align 8
  %flc1 = icmp eq i64 %flk1, 30
  %fl2 = call i32 @universe_ds_btree_floor(ptr %t, i64 10, ptr %kslot, ptr %vslot)
  %flk2 = load i64, ptr %kslot, align 8
  %flc2 = icmp eq i64 %flk2, 10
  %fl3 = call i32 @universe_ds_btree_floor(ptr %t, i64 5, ptr %kslot, ptr %vslot)
  %flc3 = icmp eq i32 %fl3, 5
  %fa = and i1 %flc1, %flc2
  %fb = and i1 %fa, %flc3
  call void @ut_check(i1 %fb, ptr @msg.floor)

  ; ceiling(35)=40, ceiling(100)=100, ceiling(105)=none
  %ce1 = call i32 @universe_ds_btree_ceiling(ptr %t, i64 35, ptr %kslot, ptr %vslot)
  %cek1 = load i64, ptr %kslot, align 8
  %cec1 = icmp eq i64 %cek1, 40
  %ce2 = call i32 @universe_ds_btree_ceiling(ptr %t, i64 100, ptr %kslot, ptr %vslot)
  %cek2 = load i64, ptr %kslot, align 8
  %cec2 = icmp eq i64 %cek2, 100
  %ce3 = call i32 @universe_ds_btree_ceiling(ptr %t, i64 105, ptr %kslot, ptr %vslot)
  %cec3 = icmp eq i32 %ce3, 5
  %ca = and i1 %cec1, %cec2
  %cb = and i1 %ca, %cec3
  call void @ut_check(i1 %cb, ptr @msg.ceil)

  call void @universe_ds_btree_destroy(ptr %t)
  ret void
}

; ---------------------------------------------------------------------------
; big randomized: N keys, insert shuffled, find, ordered, subrange, deletes
; ---------------------------------------------------------------------------
define void @test_big() {
entry:
  %vslot = alloca i64, align 8
  %seed = alloca i64, align 8
  store i64 -6534278822746914715, ptr %seed, align 8   ; 0x9E3779B97F4A7C15
  %N = add i64 0, 50000
  %bytes = shl i64 %N, 3
  %keys = call ptr @malloc(i64 %bytes)
  %okb = call ptr @malloc(i64 %bytes)
  %ovb = call ptr @malloc(i64 %bytes)
  %kc  = call ptr @malloc(i64 %bytes)
  ; keys[i] = i
  br label %initk

initk:
  %ii = phi i64 [ 0, %entry ], [ %iin, %initk ]
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %ii
  store i64 %ii, ptr %kp, align 8
  %iin = add i64 %ii, 1
  %imore = icmp ult i64 %iin, %N
  br i1 %imore, label %initk, label %shuf.pre

shuf.pre:
  %Nm1 = sub i64 %N, 1
  br label %shuf

shuf:
  ; Fisher-Yates from high downto 1
  %si = phi i64 [ %Nm1, %shuf.pre ], [ %sin, %shuf ]
  %r = call i64 @ut_rand(ptr %seed)
  %sip1 = add i64 %si, 1
  %j = urem i64 %r, %sip1
  %sip = getelementptr inbounds nuw i64, ptr %keys, i64 %si
  %jp = getelementptr inbounds nuw i64, ptr %keys, i64 %j
  %sv = load i64, ptr %sip, align 8
  %jv = load i64, ptr %jp, align 8
  store i64 %jv, ptr %sip, align 8
  store i64 %sv, ptr %jp, align 8
  %sin = sub i64 %si, 1
  %smore = icmp ugt i64 %sin, 0
  br i1 %smore, label %shuf, label %build

build:
  %t = call ptr @universe_ds_btree_create()
  br label %ins

ins:
  %vi.ins = phi i64 [ 0, %build ], [ %vi.ins.n, %ins ]
  %ins.i = phi i64 [ 0, %build ], [ %ins.in, %ins ]
  %ins.kp = getelementptr inbounds nuw i64, ptr %keys, i64 %ins.i
  %ins.k = load i64, ptr %ins.kp, align 8
  %ins.v = mul i64 %ins.k, 2
  %ins.v2 = add i64 %ins.v, 1
  %ins.rc = call i32 @universe_ds_btree_insert(ptr %t, i64 %ins.k, i64 %ins.v2)
  %ins.bad = icmp ne i32 %ins.rc, 0
  %ins.badz = zext i1 %ins.bad to i64
  %vi.ins.n = add i64 %vi.ins, %ins.badz
  %ins.in = add i64 %ins.i, 1
  %ins.more = icmp ult i64 %ins.in, %N
  br i1 %ins.more, label %ins, label %ins.done

ins.done:
  call void @ut_check_eq(i64 %vi.ins, i64 0, ptr @msg.big.ins)
  %cnt = call i64 @universe_ds_btree_count(ptr %t)
  call void @ut_check_eq(i64 %cnt, i64 %N, ptr @msg.big.cnt)
  br label %findall

findall:
  %vi.find = phi i64 [ 0, %ins.done ], [ %vi.find.n, %findall ]
  %f.i = phi i64 [ 0, %ins.done ], [ %f.in, %findall ]
  %f.rc = call i32 @universe_ds_btree_find(ptr %t, i64 %f.i, ptr %vslot)
  %f.v = load i64, ptr %vslot, align 8
  %f.exp = mul i64 %f.i, 2
  %f.exp2 = add i64 %f.exp, 1
  %f.rcok = icmp eq i32 %f.rc, 0
  %f.vok = icmp eq i64 %f.v, %f.exp2
  %f.ok = and i1 %f.rcok, %f.vok
  %f.bad = xor i1 %f.ok, true
  %f.badz = zext i1 %f.bad to i64
  %vi.find.n = add i64 %vi.find, %f.badz
  %f.in = add i64 %f.i, 1
  %f.more = icmp ult i64 %f.in, %N
  br i1 %f.more, label %findall, label %findall.done

findall.done:
  call void @ut_check_eq(i64 %vi.find, i64 0, ptr @msg.big.find)
  ; absent keys
  %ab1 = call i32 @universe_ds_btree_find(ptr %t, i64 %N, ptr %vslot)
  %ab1ok = icmp eq i32 %ab1, 5
  %ab2 = call i32 @universe_ds_btree_find(ptr %t, i64 -1, ptr %vslot)
  %ab2ok = icmp eq i32 %ab2, 5
  %abok = and i1 %ab1ok, %ab2ok
  call void @ut_check(i1 %abok, ptr @msg.big.abs)

  ; ordered full range
  %ordn = call i64 @universe_ds_btree_range(ptr %t, i64 -9223372036854775808, i64 9223372036854775807, ptr %okb, ptr %ovb, i64 %N)
  call void @ut_check_eq(i64 %ordn, i64 %N, ptr @msg.big.ordn)
  br label %ordchk

ordchk:
  %vi.ord = phi i64 [ 0, %findall.done ], [ %vi.ord.n, %ordchk ]
  %o.i = phi i64 [ 0, %findall.done ], [ %o.in, %ordchk ]
  %o.kp = getelementptr inbounds nuw i64, ptr %okb, i64 %o.i
  %o.k = load i64, ptr %o.kp, align 8
  %o.vp = getelementptr inbounds nuw i64, ptr %ovb, i64 %o.i
  %o.v = load i64, ptr %o.vp, align 8
  %o.expv = mul i64 %o.i, 2
  %o.expv2 = add i64 %o.expv, 1
  %o.kok = icmp eq i64 %o.k, %o.i
  %o.vok = icmp eq i64 %o.v, %o.expv2
  %o.ok = and i1 %o.kok, %o.vok
  %o.bad = xor i1 %o.ok, true
  %o.badz = zext i1 %o.bad to i64
  %vi.ord.n = add i64 %vi.ord, %o.badz
  %o.in = add i64 %o.i, 1
  %o.more = icmp ult i64 %o.in, %N
  br i1 %o.more, label %ordchk, label %ordchk.done

ordchk.done:
  call void @ut_check_eq(i64 %vi.ord, i64 0, ptr @msg.big.ord)
  ; qsort cross-check: sort a copy of the shuffled keys, compare to ordered out
  call void @llvm.memcpy.p0.p0.i64(ptr %kc, ptr %keys, i64 %bytes, i1 false)
  call void @qsort(ptr %kc, i64 %N, i64 8, ptr @cmp_i64)
  br label %qschk

qschk:
  %vi.qs = phi i64 [ 0, %ordchk.done ], [ %vi.qs.n, %qschk ]
  %q.i = phi i64 [ 0, %ordchk.done ], [ %q.in, %qschk ]
  %q.ap = getelementptr inbounds nuw i64, ptr %kc, i64 %q.i
  %q.a = load i64, ptr %q.ap, align 8
  %q.bp = getelementptr inbounds nuw i64, ptr %okb, i64 %q.i
  %q.b = load i64, ptr %q.bp, align 8
  %q.bad = icmp ne i64 %q.a, %q.b
  %q.badz = zext i1 %q.bad to i64
  %vi.qs.n = add i64 %vi.qs, %q.badz
  %q.in = add i64 %q.i, 1
  %q.more = icmp ult i64 %q.in, %N
  br i1 %q.more, label %qschk, label %qschk.done

qschk.done:
  call void @ut_check_eq(i64 %vi.qs, i64 0, ptr @msg.big.qs)
  ; subrange [1234, 5678] -> 4445 contiguous keys
  %subn = call i64 @universe_ds_btree_range(ptr %t, i64 1234, i64 5678, ptr %okb, ptr %ovb, i64 %N)
  call void @ut_check_eq(i64 %subn, i64 4445, ptr @msg.big.subn)
  br label %subchk

subchk:
  %vi.sub = phi i64 [ 0, %qschk.done ], [ %vi.sub.n, %subchk ]
  %s.i = phi i64 [ 0, %qschk.done ], [ %s.in, %subchk ]
  %s.kp = getelementptr inbounds nuw i64, ptr %okb, i64 %s.i
  %s.k = load i64, ptr %s.kp, align 8
  %s.exp = add i64 %s.i, 1234
  %s.bad = icmp ne i64 %s.k, %s.exp
  %s.badz = zext i1 %s.bad to i64
  %vi.sub.n = add i64 %vi.sub, %s.badz
  %s.in = add i64 %s.i, 1
  %s.more = icmp ult i64 %s.in, 4445
  br i1 %s.more, label %subchk, label %subchk.done

subchk.done:
  call void @ut_check_eq(i64 %vi.sub, i64 0, ptr @msg.big.sub)
  ; delete all even keys
  br label %delev

delev:
  %vi.del = phi i64 [ 0, %subchk.done ], [ %vi.del.n, %delev ]
  %d.k = phi i64 [ 0, %subchk.done ], [ %d.kn, %delev ]
  %d.rc = call i32 @universe_ds_btree_delete(ptr %t, i64 %d.k)
  %d.bad = icmp ne i32 %d.rc, 0
  %d.badz = zext i1 %d.bad to i64
  %vi.del.n = add i64 %vi.del, %d.badz
  %d.kn = add i64 %d.k, 2
  %d.more = icmp ult i64 %d.kn, %N
  br i1 %d.more, label %delev, label %delev.done

delev.done:
  call void @ut_check_eq(i64 %vi.del, i64 0, ptr @msg.big.delc)
  %halfN = lshr i64 %N, 1
  %cnt2 = call i64 @universe_ds_btree_count(ptr %t)
  call void @ut_check_eq(i64 %cnt2, i64 %halfN, ptr @msg.big.delcnt)
  br label %verify

verify:
  ; evens gone (find==5), odds present (find==0, right val)
  %vi.ev = phi i64 [ 0, %delev.done ], [ %vi.ev.n, %vcont ]
  %vi.od = phi i64 [ 0, %delev.done ], [ %vi.od.n, %vcont ]
  %ve.i = phi i64 [ 0, %delev.done ], [ %ve.in, %vcont ]
  %ekey = shl i64 %ve.i, 1
  %erc = call i32 @universe_ds_btree_find(ptr %t, i64 %ekey, ptr %vslot)
  %erc.bad = icmp ne i32 %erc, 5
  %erc.badz = zext i1 %erc.bad to i64
  %vi.ev.n = add i64 %vi.ev, %erc.badz
  %okey = add i64 %ekey, 1
  %ltN = icmp ult i64 %okey, %N
  br i1 %ltN, label %chkodd, label %vcont

chkodd:
  %orc = call i32 @universe_ds_btree_find(ptr %t, i64 %okey, ptr %vslot)
  %ov.v = load i64, ptr %vslot, align 8
  %oexp = mul i64 %okey, 2
  %oexp2 = add i64 %oexp, 1
  %orc.ok0 = icmp eq i32 %orc, 0
  %orc.ok1 = icmp eq i64 %ov.v, %oexp2
  %orc.ok = and i1 %orc.ok0, %orc.ok1
  %orc.bad = xor i1 %orc.ok, true
  %orc.badz = zext i1 %orc.bad to i64
  %vi.od.chk = add i64 %vi.od, %orc.badz
  br label %vcont

vcont:
  %vi.od.n = phi i64 [ %vi.od, %verify ], [ %vi.od.chk, %chkodd ]
  %ve.in = add i64 %ve.i, 1
  %halfmore = icmp ult i64 %ve.in, %halfN
  br i1 %halfmore, label %verify, label %verify.done

verify.done:
  call void @ut_check_eq(i64 %vi.ev, i64 0, ptr @msg.big.deven)
  call void @ut_check_eq(i64 %vi.od, i64 0, ptr @msg.big.dodd)
  ; drain the odds
  br label %delod

delod:
  %vi.do2 = phi i64 [ 0, %verify.done ], [ %vi.do2.n, %delod ]
  %do.k = phi i64 [ 1, %verify.done ], [ %do.kn, %delod ]
  %do.rc = call i32 @universe_ds_btree_delete(ptr %t, i64 %do.k)
  %do.bad = icmp ne i32 %do.rc, 0
  %do.badz = zext i1 %do.bad to i64
  %vi.do2.n = add i64 %vi.do2, %do.badz
  %do.kn = add i64 %do.k, 2
  %do.more = icmp ult i64 %do.kn, %N
  br i1 %do.more, label %delod, label %delod.done

delod.done:
  %cnt3 = call i64 @universe_ds_btree_count(ptr %t)
  %cnt3ok = icmp eq i64 %cnt3, 0
  call void @ut_check(i1 %cnt3ok, ptr @msg.big.empty)

  call void @universe_ds_btree_destroy(ptr %t)
  call void @free(ptr %keys)
  call void @free(ptr %okb)
  call void @free(ptr %ovb)
  call void @free(ptr %kc)
  ret void
}

; iterative binary search over a sorted i64 array; returns index or -1
define i64 @bsearch_ref(ptr %arr, i64 %n, i64 %key) {
entry:
  br label %loop
loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %n, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done
step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %mp = getelementptr inbounds nuw i64, ptr %arr, i64 %mid
  %mv = load i64, ptr %mp, align 8
  %lt = icmp slt i64 %mv, %key
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %lt, i64 %mid1, i64 %lo
  %hi.n = select i1 %lt, i64 %hi, i64 %mid
  br label %loop
done:
  ret i64 %lo
}

; ---------------------------------------------------------------------------
; bench: btree find vs sorted-array binary search
; ---------------------------------------------------------------------------
define void @do_bench(i32 %argc, ptr %argv) {
entry:
  %vslot = alloca i64, align 8
  %seed = alloca i64, align 8
  %sink = alloca i64, align 8
  %want = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %want, label %go, label %skip
skip:
  ret void
go:
  store i64 88172645463325252, ptr %seed, align 8
  store i64 0, ptr %sink, align 8
  %N = add i64 0, 100000
  %bytes = shl i64 %N, 3
  %arr = call ptr @malloc(i64 %bytes)
  %t = call ptr @universe_ds_btree_create()
  br label %fill
fill:
  %fi = phi i64 [ 0, %go ], [ %fin, %fill ]
  %ap = getelementptr inbounds nuw i64, ptr %arr, i64 %fi
  store i64 %fi, ptr %ap, align 8
  %irc = call i32 @universe_ds_btree_insert(ptr %t, i64 %fi, i64 %fi)
  %fin = add i64 %fi, 1
  %fmore = icmp ult i64 %fin, %N
  br i1 %fmore, label %fill, label %warm
warm:
  %M = add i64 0, 500000
  br label %bt.rep
; --- btree find distribution (warm-up rep discarded) ---
bt.rep:
  %bt.r = phi i64 [ 0, %warm ], [ %bt.rn, %bt.rep.cont ]
  %bt.t0 = call double @ut_now_sec()
  br label %bt.iter
bt.iter:
  %bt.i = phi i64 [ 0, %bt.rep ], [ %bt.in, %bt.iter ]
  %bt.rnd = call i64 @ut_rand(ptr %seed)
  %bt.key = urem i64 %bt.rnd, %N
  %bt.rc = call i32 @universe_ds_btree_find(ptr %t, i64 %bt.key, ptr %vslot)
  %bt.v = load i64, ptr %vslot, align 8
  %bt.acc = load i64, ptr %sink, align 8
  %bt.acc2 = add i64 %bt.acc, %bt.v
  store volatile i64 %bt.acc2, ptr %sink, align 8
  %bt.in = add i64 %bt.i, 1
  %bt.imore = icmp ult i64 %bt.in, %M
  br i1 %bt.imore, label %bt.iter, label %bt.iter.done
bt.iter.done:
  %bt.t1 = call double @ut_now_sec()
  %bt.dt = fsub double %bt.t1, %bt.t0
  %bt.warm = icmp eq i64 %bt.r, 0
  br i1 %bt.warm, label %bt.rep.cont, label %bt.rep.store
bt.rep.store:
  %bt.si = sub i64 %bt.r, 1
  %bt.sp = getelementptr inbounds double, ptr @btf.bench.samp, i64 %bt.si
  store double %bt.dt, ptr %bt.sp, align 8
  br label %bt.rep.cont
bt.rep.cont:
  %bt.rn = add i64 %bt.r, 1
  %bt.repmore = icmp ult i64 %bt.rn, 17
  br i1 %bt.repmore, label %bt.rep, label %bt.report
bt.report:
  call void @ut_report_dist(ptr @btf.bench.samp, i64 16, i64 %M, ptr @lbl.btf.bench)
  br label %bs.rep
; --- sorted-array binary search distribution ---
bs.rep:
  %bs.r = phi i64 [ 0, %bt.report ], [ %bs.rn, %bs.rep.cont ]
  %bs.t0 = call double @ut_now_sec()
  br label %bs.iter
bs.iter:
  %bs.i = phi i64 [ 0, %bs.rep ], [ %bs.in, %bs.iter ]
  %bs.rnd = call i64 @ut_rand(ptr %seed)
  %bs.key = urem i64 %bs.rnd, %N
  %bs.idx = call i64 @bsearch_ref(ptr %arr, i64 %N, i64 %bs.key)
  %bs.acc = load i64, ptr %sink, align 8
  %bs.acc2 = add i64 %bs.acc, %bs.idx
  store volatile i64 %bs.acc2, ptr %sink, align 8
  %bs.in = add i64 %bs.i, 1
  %bs.imore = icmp ult i64 %bs.in, %M
  br i1 %bs.imore, label %bs.iter, label %bs.iter.done
bs.iter.done:
  %bs.t1 = call double @ut_now_sec()
  %bs.dt = fsub double %bs.t1, %bs.t0
  %bs.warm = icmp eq i64 %bs.r, 0
  br i1 %bs.warm, label %bs.rep.cont, label %bs.rep.store
bs.rep.store:
  %bs.si = sub i64 %bs.r, 1
  %bs.sp = getelementptr inbounds double, ptr @bsr.bench.samp, i64 %bs.si
  store double %bs.dt, ptr %bs.sp, align 8
  br label %bs.rep.cont
bs.rep.cont:
  %bs.rn = add i64 %bs.r, 1
  %bs.repmore = icmp ult i64 %bs.rn, 17
  br i1 %bs.repmore, label %bs.rep, label %report
report:
  call void @ut_report_dist(ptr @bsr.bench.samp, i64 16, i64 %M, ptr @lbl.bsr.bench)
  call void @universe_ds_btree_destroy(ptr %t)
  call void @free(ptr %arr)
  ret void
}

; ---------------------------------------------------------------------------
; randomized INTERLEAVED put/delete vs a bounded-domain shadow. Deletes here are
; data-dependent (random key, random op) — unlike test_big's fixed delete-all-
; evens schedule — so they exercise arbitrary borrow/merge/underflow orderings.
; Invariant after EVERY op: contains(k) == shadow[k]; plus a final full-domain
; scan and a count == popcount(shadow) check.
; ---------------------------------------------------------------------------
@msg.il.state = private unnamed_addr constant [25 x i8] c"interleaved contains==sh\00"
@msg.il.scan  = private unnamed_addr constant [22 x i8] c"interleaved full scan\00"
@msg.il.count = private unnamed_addr constant [21 x i8] c"interleaved count ok\00"

define void @test_interleaved() {
entry:
  %seed = alloca i64, align 8
  store i64 -4265267296055464877, ptr %seed, align 8    ; 0xC4CEB9FE1A85EC53
  %vslot = alloca i64, align 8
  %present = call ptr @calloc(i64 4096, i64 1)
  %t = call ptr @universe_ds_btree_create()
  br label %op.head

op.head:
  %oi = phi i64 [ 0, %entry ], [ %oi.n, %op.tail ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %op.tail ]
  %cnt = phi i64 [ 0, %entry ], [ %cnt.n, %op.tail ]
  %r = call i64 @ut_rand(ptr %seed)
  %k = urem i64 %r, 4096
  %r2 = call i64 @ut_rand(ptr %seed)
  %op = and i64 %r2, 1
  %pp = getelementptr inbounds nuw i8, ptr %present, i64 %k
  %was = load i8, ptr %pp, align 1
  %km = mul i64 %k, 3
  %val = add i64 %km, 7
  %isins = icmp eq i64 %op, 1
  br i1 %isins, label %do.ins, label %do.del

do.ins:
  %irc = call i32 @universe_ds_btree_insert(ptr %t, i64 %k, i64 %val)
  %was0 = icmp eq i8 %was, 0
  %cinc = zext i1 %was0 to i64
  %cnt.ins = add i64 %cnt, %cinc
  store i8 1, ptr %pp, align 1
  br label %chk

do.del:
  %drc = call i32 @universe_ds_btree_delete(ptr %t, i64 %k)
  %was1 = icmp eq i8 %was, 1
  %cdec = zext i1 %was1 to i64
  %cnt.del = sub i64 %cnt, %cdec
  store i8 0, ptr %pp, align 1
  br label %chk

chk:
  %cnt.n = phi i64 [ %cnt.ins, %do.ins ], [ %cnt.del, %do.del ]
  %exp = phi i8 [ 1, %do.ins ], [ 0, %do.del ]
  %c = call i32 @universe_ds_btree_contains(ptr %t, i64 %k)
  %expi = zext i8 %exp to i32
  %cbad = icmp ne i32 %c, %expi
  %cbadz = zext i1 %cbad to i64
  %viol.n1 = add i64 %viol, %cbadz
  %ispres = icmp eq i8 %exp, 1
  br i1 %ispres, label %vchk, label %op.tail

vchk:
  %frc = call i32 @universe_ds_btree_find(ptr %t, i64 %k, ptr %vslot)
  %fv = load i64, ptr %vslot, align 8
  %frcok = icmp eq i32 %frc, 0
  %fvok = icmp eq i64 %fv, %val
  %fok = and i1 %frcok, %fvok
  %fbad = xor i1 %fok, true
  %fbadz = zext i1 %fbad to i64
  br label %op.tail

op.tail:
  %vadd = phi i64 [ 0, %chk ], [ %fbadz, %vchk ]
  %viol.n = add i64 %viol.n1, %vadd
  %oi.n = add nuw i64 %oi, 1
  %omore = icmp ult i64 %oi.n, 300000
  br i1 %omore, label %op.head, label %op.done

op.done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @msg.il.state)
  br label %scan

scan:
  %si = phi i64 [ 0, %op.done ], [ %si.n, %scan ]
  %sviol = phi i64 [ 0, %op.done ], [ %sviol.n, %scan ]
  %spp = getelementptr inbounds nuw i8, ptr %present, i64 %si
  %sp = load i8, ptr %spp, align 1
  %sc = call i32 @universe_ds_btree_contains(ptr %t, i64 %si)
  %sexp = zext i8 %sp to i32
  %sbad = icmp ne i32 %sc, %sexp
  %sbadz = zext i1 %sbad to i64
  %sviol.n = add i64 %sviol, %sbadz
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 4096
  br i1 %smore, label %scan, label %scan.done

scan.done:
  call void @ut_check_eq(i64 %sviol.n, i64 0, ptr @msg.il.scan)
  %fcnt = call i64 @universe_ds_btree_count(ptr %t)
  call void @ut_check_eq(i64 %fcnt, i64 %cnt.n, ptr @msg.il.count)
  call void @universe_ds_btree_destroy(ptr %t)
  call void @free(ptr %present)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_edges()
  call void @test_small()
  call void @test_big()
  call void @test_interleaved()
  call void @do_bench(i32 %argc, ptr %argv)
  %r = call i32 @ut_summary()
  ret i32 %r
}
