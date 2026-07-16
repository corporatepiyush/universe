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

; Tests for the array-backed ordered map (treemap).

declare ptr  @universe_ds_treemap_create(i64)
declare void @universe_ds_treemap_destroy(ptr)
declare i32  @universe_ds_treemap_put(ptr, i64, i64)
declare i32  @universe_ds_treemap_get(ptr, i64, ptr)
declare i32  @universe_ds_treemap_contains(ptr, i64)
declare i32  @universe_ds_treemap_delete(ptr, i64)
declare i64  @universe_ds_treemap_size(ptr)
declare i32  @universe_ds_treemap_floor(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_ceiling(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_higher(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_lower(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_min(ptr, ptr, ptr)
declare i32  @universe_ds_treemap_max(ptr, ptr, ptr)
declare i32  @universe_ds_treemap_first(ptr, ptr, ptr)
declare i32  @universe_ds_treemap_last(ptr, ptr, ptr)
declare i64  @universe_ds_treemap_range(ptr, i64, i64, ptr, ptr, i64)
declare void @universe_ds_treemap_foreach(ptr, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64  @ut_rand(ptr)
declare double @ut_now_sec()
declare i1   @ut_want_bench(i32, ptr)
declare i32  @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@tm.bench.samp   = internal global [16 x double] zeroinitializer, align 8
@tmbs.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.tm.bench   = private unnamed_addr constant [15 x i8] c"treemap-get 4M\00"
@lbl.tmbs.bench = private unnamed_addr constant [13 x i8] c"binsearch 4M\00"

declare ptr  @malloc(i64)
declare void @free(ptr)
declare ptr @calloc(i64, i64)
declare i32  @printf(ptr, ...)

@msg.b.grc  = private constant [16 x i8] c"basic get rc ok\00"
@msg.b.gv   = private constant [15 x i8] c"basic get val \00"
@msg.b.ow   = private constant [16 x i8] c"overwrite value\00"
@msg.b.owl  = private constant [16 x i8] c"overwrite count\00"
@msg.b.ag   = private constant [15 x i8] c"absent get is5\00"
@msg.b.ac   = private constant [16 x i8] c"absent cont is5\00"
@msg.b.ad   = private constant [16 x i8] c"absent del  is5\00"
@msg.b.pc   = private constant [17 x i8] c"present cont is0\00"
@msg.b.drc  = private constant [14 x i8] c"delete rc  ok\00"
@msg.b.dg   = private constant [17 x i8] c"del then get is5\00"
@msg.b.dl   = private constant [15 x i8] c"del then size \00"
@msg.b.rp   = private constant [13 x i8] c"re-put rc ok\00"
@msg.b.nullput = private constant [14 x i8] c"null put is 1\00"
@msg.b.nullget = private constant [14 x i8] c"null get is 1\00"

@msg.e.min0 = private constant [16 x i8] c"empty min  is 5\00"
@msg.e.max0 = private constant [16 x i8] c"empty max  is 5\00"
@msg.e.fl0  = private constant [16 x i8] c"empty floor is5\00"
@msg.e.s0   = private constant [13 x i8] c"empty size 0\00"
@msg.e.s1   = private constant [11 x i8] c"one size 1\00"
@msg.e.s2   = private constant [11 x i8] c"two size 2\00"
@msg.e.two  = private constant [16 x i8] c"two sorted iter\00"

@msg.p.grc  = private constant [15 x i8] c"perm get viols\00"
@msg.p.abs  = private constant [16 x i8] c"perm absent got\00"
@msg.p.sz   = private constant [15 x i8] c"perm size ==N\00\00"
@msg.p.ord  = private constant [17 x i8] c"perm iter sorted\00"
@msg.p.cnt  = private constant [16 x i8] c"perm iter count\00"
@msg.p.del  = private constant [15 x i8] c"perm del viols\00"
@msg.p.dsz  = private constant [16 x i8] c"perm del  size \00"

@msg.s.fl   = private constant [12 x i8] c"floor viols\00"
@msg.s.ce   = private constant [14 x i8] c"ceiling viols\00"
@msg.s.hi   = private constant [13 x i8] c"higher viols\00"
@msg.s.lo   = private constant [12 x i8] c"lower viols\00"
@msg.s.mm   = private constant [15 x i8] c"minmax  viols \00"

@msg.r.cnt  = private constant [15 x i8] c"range count vs\00"
@msg.r.val  = private constant [16 x i8] c"range value vs \00"
@msg.r.cap  = private constant [16 x i8] c"range cap-limit\00"
@msg.r.emp  = private constant [15 x i8] c"range empty  0\00"
@msg.r.big  = private constant [16 x i8] c"unsigned order \00"


; ---- foreach context: {prev, first_flag, violations, count} --------------
; monotone-ascending check callback
define internal void @cb_ordered(ptr %ctx, i64 %k, i64 %v) {
entry:
  %prev.p = getelementptr inbounds nuw i8, ptr %ctx, i64 0
  %first.p = getelementptr inbounds nuw i8, ptr %ctx, i64 8
  %vio.p = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  %cnt.p = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %first = load i64, ptr %first.p, align 8
  %isfirst = icmp eq i64 %first, 0
  br i1 %isfirst, label %skip, label %chk

chk:
  %prev = load i64, ptr %prev.p, align 8
  ; violation if k <= prev (must strictly ascend, unsigned)
  %le = icmp ule i64 %k, %prev
  br i1 %le, label %bump, label %skip

bump:
  %vio = load i64, ptr %vio.p, align 8
  %vio.n = add i64 %vio, 1
  store i64 %vio.n, ptr %vio.p, align 8
  br label %skip

skip:
  store i64 %k, ptr %prev.p, align 8
  store i64 1, ptr %first.p, align 8
  %cnt = load i64, ptr %cnt.p, align 8
  %cnt.n = add i64 %cnt, 1
  store i64 %cnt.n, ptr %cnt.p, align 8
  ret void
}

; ---- Test 1: basic correctness + error codes ----------------------------
define internal void @t_basic() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 8)
  %ob = alloca i64, align 8

  ; null-guard on put/get
  %np = call i32 @universe_ds_treemap_put(ptr null, i64 1, i64 1)
  %npok = icmp eq i32 %np, 1
  call void @ut_check(i1 %npok, ptr @msg.b.nullput)
  %ng = call i32 @universe_ds_treemap_get(ptr null, i64 1, ptr %ob)
  %ngok = icmp eq i32 %ng, 1
  call void @ut_check(i1 %ngok, ptr @msg.b.nullget)

  ; put(5)=500, get -> 500
  %p1 = call i32 @universe_ds_treemap_put(ptr %m, i64 5, i64 500)
  %g1 = call i32 @universe_ds_treemap_get(ptr %m, i64 5, ptr %ob)
  %g1ok = icmp eq i32 %g1, 0
  call void @ut_check(i1 %g1ok, ptr @msg.b.grc)
  %v1 = load i64, ptr %ob, align 8
  %v1ok = icmp eq i64 %v1, 500
  call void @ut_check(i1 %v1ok, ptr @msg.b.gv)

  ; overwrite put(5)=999; size stays 1
  %p2 = call i32 @universe_ds_treemap_put(ptr %m, i64 5, i64 999)
  %g2 = call i32 @universe_ds_treemap_get(ptr %m, i64 5, ptr %ob)
  %v2 = load i64, ptr %ob, align 8
  %v2ok = icmp eq i64 %v2, 999
  call void @ut_check(i1 %v2ok, ptr @msg.b.ow)
  %sz2 = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz2, i64 1, ptr @msg.b.owl)

  ; absent get/contains/delete -> 5 ; present contains -> 0
  %ga = call i32 @universe_ds_treemap_get(ptr %m, i64 42, ptr %ob)
  %gaok = icmp eq i32 %ga, 5
  call void @ut_check(i1 %gaok, ptr @msg.b.ag)
  %ca = call i32 @universe_ds_treemap_contains(ptr %m, i64 42)
  %caok = icmp eq i32 %ca, 5
  call void @ut_check(i1 %caok, ptr @msg.b.ac)
  %da = call i32 @universe_ds_treemap_delete(ptr %m, i64 42)
  %daok = icmp eq i32 %da, 5
  call void @ut_check(i1 %daok, ptr @msg.b.ad)
  %cp = call i32 @universe_ds_treemap_contains(ptr %m, i64 5)
  %cpok = icmp eq i32 %cp, 0
  call void @ut_check(i1 %cpok, ptr @msg.b.pc)

  ; delete(5) then get -> 5, size 0
  %d1 = call i32 @universe_ds_treemap_delete(ptr %m, i64 5)
  %d1ok = icmp eq i32 %d1, 0
  call void @ut_check(i1 %d1ok, ptr @msg.b.drc)
  %g3 = call i32 @universe_ds_treemap_get(ptr %m, i64 5, ptr %ob)
  %g3ok = icmp eq i32 %g3, 5
  call void @ut_check(i1 %g3ok, ptr @msg.b.dg)
  %sz3 = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz3, i64 0, ptr @msg.b.dl)

  ; re-put works
  %p3 = call i32 @universe_ds_treemap_put(ptr %m, i64 5, i64 77)
  %p3ok = icmp eq i32 %p3, 0
  call void @ut_check(i1 %p3ok, ptr @msg.b.rp)

  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; ---- Test 2: edge sizes 0/1/2 + empty ordered queries -------------------
define internal void @t_edges() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 2)
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8

  %sz0 = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz0, i64 0, ptr @msg.e.s0)
  %mn = call i32 @universe_ds_treemap_min(ptr %m, ptr %ok, ptr %ov)
  %mnok = icmp eq i32 %mn, 5
  call void @ut_check(i1 %mnok, ptr @msg.e.min0)
  %mx = call i32 @universe_ds_treemap_max(ptr %m, ptr %ok, ptr %ov)
  %mxok = icmp eq i32 %mx, 5
  call void @ut_check(i1 %mxok, ptr @msg.e.max0)
  %fl = call i32 @universe_ds_treemap_floor(ptr %m, i64 10, ptr %ok, ptr %ov)
  %flok = icmp eq i32 %fl, 5
  call void @ut_check(i1 %flok, ptr @msg.e.fl0)

  ; one element
  %p1 = call i32 @universe_ds_treemap_put(ptr %m, i64 100, i64 1)
  %sz1 = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz1, i64 1, ptr @msg.e.s1)

  ; two elements inserted out of order (30 then 100 already, add 30)
  %p2 = call i32 @universe_ds_treemap_put(ptr %m, i64 30, i64 2)
  %sz2 = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz2, i64 2, ptr @msg.e.s2)
  ; min should be 30, max 100 (insertion order was 100,30)
  %mn2 = call i32 @universe_ds_treemap_min(ptr %m, ptr %ok, ptr %ov)
  %mnk = load i64, ptr %ok, align 8
  %mx2 = call i32 @universe_ds_treemap_max(ptr %m, ptr %ok, ptr %ov)
  %mxk = load i64, ptr %ok, align 8
  %twoc = icmp eq i64 %mnk, 30
  %twod = icmp eq i64 %mxk, 100
  %twook = and i1 %twoc, %twod
  call void @ut_check(i1 %twook, ptr @msg.e.two)

  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; ---- Test 3: random-insert-order permutation of 0..N-1 ------------------
; key(i) = (i * 32749) mod 50000 is a permutation (gcd(32749,50000)=1).
define internal void @t_perm() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 8)
  %ob = alloca i64, align 8
  br label %ins

ins:
  %i = phi i64 [ 0, %entry ], [ %i.n, %ins ]
  %prod = mul i64 %i, 32749
  %key = urem i64 %prod, 50000
  %val = add i64 %key, 7
  %p = call i32 @universe_ds_treemap_put(ptr %m, i64 %key, i64 %val)
  %i.n = add nuw i64 %i, 1
  %ins.done = icmp uge i64 %i.n, 50000
  br i1 %ins.done, label %verify, label %ins

verify:
  %sz = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %sz, i64 50000, ptr @msg.p.sz)
  br label %vloop

vloop:
  %j = phi i64 [ 0, %verify ], [ %j.n, %vcont ]
  %vio = phi i64 [ 0, %verify ], [ %vio.n, %vcont ]
  %g = call i32 @universe_ds_treemap_get(ptr %m, i64 %j, ptr %ob)
  %got = load i64, ptr %ob, align 8
  %exp = add i64 %j, 7
  %rcbad = icmp ne i32 %g, 0
  %vbad = icmp ne i64 %got, %exp
  %bad = or i1 %rcbad, %vbad
  %badi = zext i1 %bad to i64
  %vio.n = add i64 %vio, %badi
  %j.n = add nuw i64 %j, 1
  %vdone = icmp uge i64 %j.n, 50000
  br i1 %vdone, label %vfin, label %vcont

vcont:
  br label %vloop

vfin:
  call void @ut_check_eq(i64 %vio.n, i64 0, ptr @msg.p.grc)
  ; absent keys 50000..50009 -> 5
  %ab = call i32 @universe_ds_treemap_get(ptr %m, i64 50005, ptr %ob)
  %abok = icmp eq i32 %ab, 5
  call void @ut_check(i1 %abok, ptr @msg.p.abs)

  ; ordered iterate check
  %ctx = alloca [4 x i64], align 8
  store i64 0, ptr %ctx, align 8
  %f.p = getelementptr inbounds nuw i8, ptr %ctx, i64 8
  store i64 0, ptr %f.p, align 8
  %v.p = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  store i64 0, ptr %v.p, align 8
  %c.p = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  store i64 0, ptr %c.p, align 8
  call void @universe_ds_treemap_foreach(ptr %m, ptr @cb_ordered, ptr %ctx)
  %iter.vio = load i64, ptr %v.p, align 8
  call void @ut_check_eq(i64 %iter.vio, i64 0, ptr @msg.p.ord)
  %iter.cnt = load i64, ptr %c.p, align 8
  call void @ut_check_eq(i64 %iter.cnt, i64 50000, ptr @msg.p.cnt)

  ; delete every even key, re-verify
  br label %dloop

dloop:
  %dk = phi i64 [ 0, %vfin ], [ %dk.n, %dcont ]
  %dr = call i32 @universe_ds_treemap_delete(ptr %m, i64 %dk)
  %dk.n = add nuw i64 %dk, 2
  %ddone = icmp uge i64 %dk.n, 50000
  br i1 %ddone, label %dverify, label %dcont

dcont:
  br label %dloop

dverify:
  %dsz = call i64 @universe_ds_treemap_size(ptr %m)
  call void @ut_check_eq(i64 %dsz, i64 25000, ptr @msg.p.dsz)
  br label %dvloop

dvloop:
  %e = phi i64 [ 0, %dverify ], [ %e.n, %dvcont ]
  %dvio = phi i64 [ 0, %dverify ], [ %dvio.n, %dvcont ]
  %ce = call i32 @universe_ds_treemap_contains(ptr %m, i64 %e)
  ; even -> should be absent (5); odd -> present (0)
  %isodd = and i64 %e, 1
  %oddb = icmp ne i64 %isodd, 0
  %want = select i1 %oddb, i32 0, i32 5
  %mism = icmp ne i32 %ce, %want
  %mi = zext i1 %mism to i64
  %dvio.n = add i64 %dvio, %mi
  %e.n = add nuw i64 %e, 1
  %edone = icmp uge i64 %e.n, 50000
  br i1 %edone, label %dvfin, label %dvcont

dvcont:
  br label %dvloop

dvfin:
  call void @ut_check_eq(i64 %dvio.n, i64 0, ptr @msg.p.del)
  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; ---- Test 4: floor/ceiling/higher/lower/min/max on known inputs ---------
; keys = {10,20,30,40,50} value = key
define internal void @t_search() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 8)
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  %p0 = call i32 @universe_ds_treemap_put(ptr %m, i64 30, i64 30)
  %p1 = call i32 @universe_ds_treemap_put(ptr %m, i64 10, i64 10)
  %p2 = call i32 @universe_ds_treemap_put(ptr %m, i64 50, i64 50)
  %p3 = call i32 @universe_ds_treemap_put(ptr %m, i64 20, i64 20)
  %p4 = call i32 @universe_ds_treemap_put(ptr %m, i64 40, i64 40)

  ; floor: floor(25)=20, floor(30)=30, floor(5)=none, floor(100)=50, floor(10)=10
  %fvio = alloca i64, align 8
  store i64 0, ptr %fvio, align 8
  call void @chk_query(ptr %m, i32 0, i64 25, i32 0, i64 20, ptr %fvio, ptr %ok)
  call void @chk_query(ptr %m, i32 0, i64 30, i32 0, i64 30, ptr %fvio, ptr %ok)
  call void @chk_query(ptr %m, i32 0, i64 5,  i32 5, i64 0,  ptr %fvio, ptr %ok)
  call void @chk_query(ptr %m, i32 0, i64 100,i32 0, i64 50, ptr %fvio, ptr %ok)
  call void @chk_query(ptr %m, i32 0, i64 10, i32 0, i64 10, ptr %fvio, ptr %ok)
  %fv = load i64, ptr %fvio, align 8
  call void @ut_check_eq(i64 %fv, i64 0, ptr @msg.s.fl)

  ; ceiling: ceiling(25)=30, ceiling(30)=30, ceiling(50)=50, ceiling(51)=none, ceiling(5)=10
  %cvio = alloca i64, align 8
  store i64 0, ptr %cvio, align 8
  call void @chk_query(ptr %m, i32 1, i64 25, i32 0, i64 30, ptr %cvio, ptr %ok)
  call void @chk_query(ptr %m, i32 1, i64 30, i32 0, i64 30, ptr %cvio, ptr %ok)
  call void @chk_query(ptr %m, i32 1, i64 50, i32 0, i64 50, ptr %cvio, ptr %ok)
  call void @chk_query(ptr %m, i32 1, i64 51, i32 5, i64 0,  ptr %cvio, ptr %ok)
  call void @chk_query(ptr %m, i32 1, i64 5,  i32 0, i64 10, ptr %cvio, ptr %ok)
  %cv = load i64, ptr %cvio, align 8
  call void @ut_check_eq(i64 %cv, i64 0, ptr @msg.s.ce)

  ; higher: higher(30)=40, higher(50)=none, higher(5)=10, higher(49)=50
  %hvio = alloca i64, align 8
  store i64 0, ptr %hvio, align 8
  call void @chk_query(ptr %m, i32 2, i64 30, i32 0, i64 40, ptr %hvio, ptr %ok)
  call void @chk_query(ptr %m, i32 2, i64 50, i32 5, i64 0,  ptr %hvio, ptr %ok)
  call void @chk_query(ptr %m, i32 2, i64 5,  i32 0, i64 10, ptr %hvio, ptr %ok)
  call void @chk_query(ptr %m, i32 2, i64 49, i32 0, i64 50, ptr %hvio, ptr %ok)
  %hv = load i64, ptr %hvio, align 8
  call void @ut_check_eq(i64 %hv, i64 0, ptr @msg.s.hi)

  ; lower: lower(30)=20, lower(10)=none, lower(100)=50, lower(11)=10
  %lvio = alloca i64, align 8
  store i64 0, ptr %lvio, align 8
  call void @chk_query(ptr %m, i32 3, i64 30, i32 0, i64 20, ptr %lvio, ptr %ok)
  call void @chk_query(ptr %m, i32 3, i64 10, i32 5, i64 0,  ptr %lvio, ptr %ok)
  call void @chk_query(ptr %m, i32 3, i64 100,i32 0, i64 50, ptr %lvio, ptr %ok)
  call void @chk_query(ptr %m, i32 3, i64 11, i32 0, i64 10, ptr %lvio, ptr %ok)
  %lv = load i64, ptr %lvio, align 8
  call void @ut_check_eq(i64 %lv, i64 0, ptr @msg.s.lo)

  ; min=10, max=50, first=10, last=50
  %mnr = call i32 @universe_ds_treemap_min(ptr %m, ptr %ok, ptr %ov)
  %mnk = load i64, ptr %ok, align 8
  %mxr = call i32 @universe_ds_treemap_max(ptr %m, ptr %ok, ptr %ov)
  %mxk = load i64, ptr %ok, align 8
  %frr = call i32 @universe_ds_treemap_first(ptr %m, ptr %ok, ptr %ov)
  %frk = load i64, ptr %ok, align 8
  %lrr = call i32 @universe_ds_treemap_last(ptr %m, ptr %ok, ptr %ov)
  %lrk = load i64, ptr %ok, align 8
  %a = icmp eq i64 %mnk, 10
  %b = icmp eq i64 %mxk, 50
  %c = icmp eq i64 %frk, 10
  %d = icmp eq i64 %lrk, 50
  %ab = and i1 %a, %b
  %cd = and i1 %c, %d
  %mmok = and i1 %ab, %cd
  %mmv = zext i1 %mmok to i64
  ; store violation count (0 if ok)
  %mmvio = xor i64 %mmv, 1
  call void @ut_check_eq(i64 %mmvio, i64 0, ptr @msg.s.mm)

  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; helper: run query kind (0 floor,1 ceiling,2 higher,3 lower), compare rc and,
; when rc==0, key; bump *vio on mismatch. %scratch is a caller i64 out slot.
define internal void @chk_query(ptr %m, i32 %kind, i64 %arg, i32 %exp.rc, i64 %exp.k, ptr %vio, ptr %scratch) {
entry:
  switch i32 %kind, label %do.floor [ i32 0, label %do.floor
                                       i32 1, label %do.ceil
                                       i32 2, label %do.high
                                       i32 3, label %do.low ]

do.floor:
  %r0 = call i32 @universe_ds_treemap_floor(ptr %m, i64 %arg, ptr %scratch, ptr null)
  br label %cmp

do.ceil:
  %r1 = call i32 @universe_ds_treemap_ceiling(ptr %m, i64 %arg, ptr %scratch, ptr null)
  br label %cmp

do.high:
  %r2 = call i32 @universe_ds_treemap_higher(ptr %m, i64 %arg, ptr %scratch, ptr null)
  br label %cmp

do.low:
  %r3 = call i32 @universe_ds_treemap_lower(ptr %m, i64 %arg, ptr %scratch, ptr null)
  br label %cmp

cmp:
  %rc = phi i32 [ %r0, %do.floor ], [ %r1, %do.ceil ], [ %r2, %do.high ], [ %r3, %do.low ]
  %rc.bad = icmp ne i32 %rc, %exp.rc
  ; when expected found, also compare key
  %want.found = icmp eq i32 %exp.rc, 0
  br i1 %want.found, label %chk.key, label %fin

chk.key:
  %gk = load i64, ptr %scratch, align 8
  %kbad = icmp ne i64 %gk, %exp.k
  br label %fin

fin:
  %kbad.f = phi i1 [ false, %cmp ], [ %kbad, %chk.key ]
  %anybad = or i1 %rc.bad, %kbad.f
  br i1 %anybad, label %bump, label %ret

bump:
  %v = load i64, ptr %vio, align 8
  %v.n = add i64 %v, 1
  store i64 %v.n, ptr %vio, align 8
  br label %ret

ret:
  ret void
}

; ---- Test 5: range vs linear reference over sparse keys -----------------
; keys = 0,10,20,...,9990  (M=1000)   value = key+3
define internal void @t_range() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 8)
  %refk = call ptr @malloc(i64 8000)   ; 1000 * 8
  %ok = call ptr @malloc(i64 8000)
  %ov = call ptr @malloc(i64 8000)
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %key = mul i64 %i, 10
  %val = add i64 %key, 3
  %p = call i32 @universe_ds_treemap_put(ptr %m, i64 %key, i64 %val)
  %rp = getelementptr inbounds nuw i64, ptr %refk, i64 %i
  store i64 %key, ptr %rp, align 8
  %i.n = add nuw i64 %i, 1
  %fdone = icmp uge i64 %i.n, 1000
  br i1 %fdone, label %tests, label %fill

tests:
  ; several (lo,hi) inclusive pairs; reference count via linear scan of refk
  ; pairs stored inline; iterate them
  %cvio = alloca i64, align 8
  store i64 0, ptr %cvio, align 8
  %vvio = alloca i64, align 8
  store i64 0, ptr %vvio, align 8
  br label %ploop

ploop:
  %pi = phi i64 [ 0, %tests ], [ %pi.n, %pcont ]
  ; derive (lo,hi) from pi: use a small set
  %lo = call i64 @range_lo(i64 %pi)
  %hi = call i64 @range_hi(i64 %pi)
  ; reference count
  %refc = call i64 @ref_count(ptr %refk, i64 1000, i64 %lo, i64 %hi)
  ; treemap range
  %cnt = call i64 @universe_ds_treemap_range(ptr %m, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 1000)
  %cbad = icmp ne i64 %cnt, %refc
  %cbi = zext i1 %cbad to i64
  %cv0 = load i64, ptr %cvio, align 8
  %cv1 = add i64 %cv0, %cbi
  store i64 %cv1, ptr %cvio, align 8
  ; verify out entries: keys ascending in [lo,hi], value=key+3
  %nchk = call i64 @verify_out(ptr %ok, ptr %ov, i64 %cnt, i64 %lo, i64 %hi)
  %vv0 = load i64, ptr %vvio, align 8
  %vv1 = add i64 %vv0, %nchk
  store i64 %vv1, ptr %vvio, align 8
  %pi.n = add nuw i64 %pi, 1
  %pdone = icmp uge i64 %pi.n, 8
  br i1 %pdone, label %pfin, label %pcont

pcont:
  br label %ploop

pfin:
  %cvf = load i64, ptr %cvio, align 8
  call void @ut_check_eq(i64 %cvf, i64 0, ptr @msg.r.cnt)
  %vvf = load i64, ptr %vvio, align 8
  call void @ut_check_eq(i64 %vvf, i64 0, ptr @msg.r.val)

  ; empty range: lo > hi -> 0
  %er = call i64 @universe_ds_treemap_range(ptr %m, i64 500, i64 100, ptr %ok, ptr %ov, i64 1000)
  %erok = icmp eq i64 %er, 0
  call void @ut_check(i1 %erok, ptr @msg.r.emp)

  ; cap-limited: full range but out_cap=5 -> returns 1000, writes 5 sorted
  %cr = call i64 @universe_ds_treemap_range(ptr %m, i64 0, i64 100000, ptr %ok, ptr %ov, i64 5)
  %crc = icmp eq i64 %cr, 1000
  ; check first 5 keys are 0,10,20,30,40
  %k0 = load i64, ptr %ok, align 8
  %ok4.p = getelementptr inbounds nuw i64, ptr %ok, i64 4
  %k4 = load i64, ptr %ok4.p, align 8
  %c0 = icmp eq i64 %k0, 0
  %c4 = icmp eq i64 %k4, 40
  %cc = and i1 %c0, %c4
  %capok = and i1 %crc, %cc
  call void @ut_check(i1 %capok, ptr @msg.r.cap)

  call void @free(ptr %refk)
  call void @free(ptr %ok)
  call void @free(ptr %ov)
  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

define internal i64 @range_lo(i64 %i) {
entry:
  ; los: 0, 55, 100, 9990, 3333, 10000, 0, 5000
  switch i64 %i, label %d [ i64 0, label %a0
                            i64 1, label %a1
                            i64 2, label %a2
                            i64 3, label %a3
                            i64 4, label %a4
                            i64 5, label %a5
                            i64 6, label %a6
                            i64 7, label %a7 ]
a0: ret i64 0
a1: ret i64 55
a2: ret i64 100
a3: ret i64 9990
a4: ret i64 3333
a5: ret i64 10000
a6: ret i64 0
a7: ret i64 5000
d:  ret i64 0
}

define internal i64 @range_hi(i64 %i) {
entry:
  ; his: 0, 205, 100, 9990, 6666, 20000, 100000, 5000
  switch i64 %i, label %d [ i64 0, label %a0
                            i64 1, label %a1
                            i64 2, label %a2
                            i64 3, label %a3
                            i64 4, label %a4
                            i64 5, label %a5
                            i64 6, label %a6
                            i64 7, label %a7 ]
a0: ret i64 0
a1: ret i64 205
a2: ret i64 100
a3: ret i64 9990
a4: ret i64 6666
a5: ret i64 20000
a6: ret i64 100000
a7: ret i64 5000
d:  ret i64 0
}

; reference: count of refk entries in [lo,hi] inclusive (unsigned)
define internal i64 @ref_count(ptr %refk, i64 %n, i64 %lo, i64 %hi) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %kp = getelementptr inbounds nuw i64, ptr %refk, i64 %i
  %k = load i64, ptr %kp, align 8
  %ge = icmp uge i64 %k, %lo
  %le = icmp ule i64 %k, %hi
  %in = and i1 %ge, %le
  %ini = zext i1 %in to i64
  %acc.n = add i64 %acc, %ini
  %i.n = add nuw i64 %i, 1
  %done = icmp uge i64 %i.n, %n
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %acc.n
}

; verify out keys strictly ascending in [lo,hi] and value==key+3; return #viols
define internal i64 @verify_out(ptr %ok, ptr %ov, i64 %cnt, i64 %lo, i64 %hi) {
entry:
  %z = icmp eq i64 %cnt, 0
  br i1 %z, label %fin0, label %loop

fin0:
  ret i64 0

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %vio = phi i64 [ 0, %entry ], [ %vio.n, %loop ]
  %prev = phi i64 [ 0, %entry ], [ %k, %loop ]
  %kp = getelementptr inbounds nuw i64, ptr %ok, i64 %i
  %k = load i64, ptr %kp, align 8
  %vp = getelementptr inbounds nuw i64, ptr %ov, i64 %i
  %v = load i64, ptr %vp, align 8
  ; range
  %ge = icmp uge i64 %k, %lo
  %le = icmp ule i64 %k, %hi
  %inrange = and i1 %ge, %le
  ; value
  %expv = add i64 %k, 3
  %vok = icmp eq i64 %v, %expv
  ; ascending: ok if (i==0) or (k>prev)
  %notfirst = icmp ne i64 %i, 0
  %asc = icmp ugt i64 %k, %prev
  %ordok = select i1 %notfirst, i1 %asc, i1 true
  %good0 = and i1 %inrange, %vok
  %good = and i1 %good0, %ordok
  %bad = xor i1 %good, true
  %badi = zext i1 %bad to i64
  %vio.n = add i64 %vio, %badi
  %i.n = add nuw i64 %i, 1
  %done = icmp uge i64 %i.n, %cnt
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %vio.n
}

; ---- Test 6: unsigned ordering across the full 64-bit range -------------
define internal void @t_unsigned() {
entry:
  %m = call ptr @universe_ds_treemap_create(i64 8)
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  ; keys: 1, 2^63 (min signed), 2^63+5, 2^64-1 (max unsigned)
  %big1 = add i64 0, -9223372036854775808   ; 0x8000000000000000
  %big2 = add i64 %big1, 5
  %maxu = add i64 0, -1                       ; 0xFFFFFFFFFFFFFFFF
  %pa = call i32 @universe_ds_treemap_put(ptr %m, i64 %maxu, i64 400)
  %pb = call i32 @universe_ds_treemap_put(ptr %m, i64 1, i64 100)
  %pc = call i32 @universe_ds_treemap_put(ptr %m, i64 %big2, i64 300)
  %pd = call i32 @universe_ds_treemap_put(ptr %m, i64 %big1, i64 200)
  ; unsigned order should be: 1 < big1 < big2 < maxu
  %mn = call i32 @universe_ds_treemap_min(ptr %m, ptr %ok, ptr %ov)
  %mnk = load i64, ptr %ok, align 8
  %mx = call i32 @universe_ds_treemap_max(ptr %m, ptr %ok, ptr %ov)
  %mxk = load i64, ptr %ok, align 8
  ; floor(big2) should be big2; higher(big1)=big2; lower(maxu)=big2
  %fr = call i32 @universe_ds_treemap_floor(ptr %m, i64 %big2, ptr %ok, ptr null)
  %frk = load i64, ptr %ok, align 8
  %a = icmp eq i64 %mnk, 1
  %b = icmp eq i64 %mxk, %maxu
  %c = icmp eq i64 %frk, %big2
  %ab = and i1 %a, %b
  %ok3 = and i1 %ab, %c
  call void @ut_check(i1 %ok3, ptr @msg.r.big)
  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; ---- Bench: treemap get vs sorted-array binary search --------------------
define internal void @run_bench() {
entry:
  %N = add i64 0, 65536
  %m = call ptr @universe_ds_treemap_create(i64 %N)
  %ob = alloca i64, align 8
  %karr = call ptr @malloc(i64 524288)   ; 65536*8 sorted keys
  %varr = call ptr @malloc(i64 524288)
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %key = add i64 %i, 1
  %val = mul i64 %key, 2654435761
  %p = call i32 @universe_ds_treemap_put(ptr %m, i64 %key, i64 %val)
  %kp = getelementptr inbounds nuw i64, ptr %karr, i64 %i
  store i64 %key, ptr %kp, align 8
  %vp = getelementptr inbounds nuw i64, ptr %varr, i64 %i
  store i64 %val, ptr %vp, align 8
  %i.n = add nuw i64 %i, 1
  %fdone = icmp uge i64 %i.n, 65536
  br i1 %fdone, label %hot.rep.head, label %fill

; --- treemap get distribution (read-only, warm-up rep discarded) ---
hot.rep.head:
  %hrep = phi i64 [ 0, %fill ], [ %hrep.n, %hot.rep.cont ]
  %t0 = call double @ut_now_sec()
  br label %hot

hot:
  %q = phi i64 [ 0, %hot.rep.head ], [ %q.n, %hot ]
  %acc = phi i64 [ 0, %hot.rep.head ], [ %acc.n, %hot ]
  %qk.m = and i64 %q, 65535
  %qk = add i64 %qk.m, 1
  %gr = call i32 @universe_ds_treemap_get(ptr %m, i64 %qk, ptr %ob)
  %got = load i64, ptr %ob, align 8
  %acc.n = add i64 %acc, %got
  %q.n = add nuw i64 %q, 1
  %hdone = icmp uge i64 %q.n, 4194304
  br i1 %hdone, label %hot.rep.done, label %hot

hot.rep.done:
  store volatile i64 %acc.n, ptr %ob, align 8
  %t1 = call double @ut_now_sec()
  %hdt = fsub double %t1, %t0
  %hot.warm = icmp eq i64 %hrep, 0
  br i1 %hot.warm, label %hot.rep.cont, label %hot.rep.store
hot.rep.store:
  %hot.si = sub i64 %hrep, 1
  %hot.sp = getelementptr inbounds double, ptr @tm.bench.samp, i64 %hot.si
  store double %hdt, ptr %hot.sp, align 8
  br label %hot.rep.cont
hot.rep.cont:
  %hrep.n = add nuw i64 %hrep, 1
  %hot.more = icmp ult i64 %hrep.n, 17
  br i1 %hot.more, label %hot.rep.head, label %hot.report
hot.report:
  call void @ut_report_dist(ptr @tm.bench.samp, i64 16, i64 4194304, ptr @lbl.tm.bench)
  br label %bs.rep.head

; --- reference: manual binary search over sorted karr ---
bs.rep.head:
  %brep = phi i64 [ 0, %hot.report ], [ %brep.n, %bs.rep.cont ]
  %t2 = call double @ut_now_sec()
  br label %bs

bs:
  %bq = phi i64 [ 0, %bs.rep.head ], [ %bq.n, %bs ]
  %bacc = phi i64 [ 0, %bs.rep.head ], [ %bacc.n, %bs ]
  %bqk.m = and i64 %bq, 65535
  %bqk = add i64 %bqk.m, 1
  %idx = call i64 @bsearch_ref(ptr %karr, i64 65536, i64 %bqk)
  %bvp = getelementptr inbounds nuw i64, ptr %varr, i64 %idx
  %bv = load i64, ptr %bvp, align 8
  %bacc.n = add i64 %bacc, %bv
  %bq.n = add nuw i64 %bq, 1
  %bdone = icmp uge i64 %bq.n, 4194304
  br i1 %bdone, label %bs.rep.done, label %bs

bs.rep.done:
  store volatile i64 %bacc.n, ptr %ob, align 8
  %t3 = call double @ut_now_sec()
  %bdt = fsub double %t3, %t2
  %bs.warm = icmp eq i64 %brep, 0
  br i1 %bs.warm, label %bs.rep.cont, label %bs.rep.store
bs.rep.store:
  %bs.si = sub i64 %brep, 1
  %bs.sp = getelementptr inbounds double, ptr @tmbs.bench.samp, i64 %bs.si
  store double %bdt, ptr %bs.sp, align 8
  br label %bs.rep.cont
bs.rep.cont:
  %brep.n = add nuw i64 %brep, 1
  %bs.more = icmp ult i64 %brep.n, 17
  br i1 %bs.more, label %bs.rep.head, label %bs.report
bs.report:
  call void @ut_report_dist(ptr @tmbs.bench.samp, i64 16, i64 4194304, ptr @lbl.tmbs.bench)
  call void @free(ptr %karr)
  call void @free(ptr %varr)
  call void @universe_ds_treemap_destroy(ptr %m)
  ret void
}

; plain lower_bound binary search over sorted keys, returns exact index (key present)
define internal i64 @bsearch_ref(ptr %keys, i64 %n, i64 %k) {
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
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %less = icmp ult i64 %kv, %k
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %less, i64 %mid1, i64 %lo
  %hi.n = select i1 %less, i64 %hi, i64 %mid
  br label %loop

done:
  ret i64 %lo
}

; randomized INTERLEAVED put/delete vs a bounded-domain shadow: data-dependent
; deletes exercise arbitrary gap-close memmove orderings. Invariant after every
; op: contains(k) == shadow[k]; plus final full scan + size.
@msg.il.state = private unnamed_addr constant [25 x i8] c"interleaved contains==sh\00"
@msg.il.scan  = private unnamed_addr constant [22 x i8] c"interleaved full scan\00"
@msg.il.count = private unnamed_addr constant [20 x i8] c"interleaved size ok\00"

define void @test_interleaved() {
entry:
  %seed = alloca i64, align 8
  store i64 -4265267296055464877, ptr %seed, align 8
  %vslot = alloca i64, align 8
  %present = call ptr @calloc(i64 4096, i64 1)
  %t = call ptr @universe_ds_treemap_create(i64 16)
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
  %irc = call i32 @universe_ds_treemap_put(ptr %t, i64 %k, i64 %val)
  %was0 = icmp eq i8 %was, 0
  %cinc = zext i1 %was0 to i64
  %cnt.ins = add i64 %cnt, %cinc
  store i8 1, ptr %pp, align 1
  br label %chk

do.del:
  %drc = call i32 @universe_ds_treemap_delete(ptr %t, i64 %k)
  %was1 = icmp eq i8 %was, 1
  %cdec = zext i1 %was1 to i64
  %cnt.del = sub i64 %cnt, %cdec
  store i8 0, ptr %pp, align 1
  br label %chk

chk:
  %cnt.n = phi i64 [ %cnt.ins, %do.ins ], [ %cnt.del, %do.del ]
  %exp = phi i8 [ 1, %do.ins ], [ 0, %do.del ]
  %c = call i32 @universe_ds_treemap_contains(ptr %t, i64 %k)
  %pres = icmp eq i8 %exp, 1                 ; treemap contains: 0=present, 5=absent
  %wantc = select i1 %pres, i32 0, i32 5
  %cbad = icmp ne i32 %c, %wantc
  %cbadz = zext i1 %cbad to i64
  %viol.n1 = add i64 %viol, %cbadz
  %ispres = icmp eq i8 %exp, 1
  br i1 %ispres, label %vchk, label %op.tail

vchk:
  %frc = call i32 @universe_ds_treemap_get(ptr %t, i64 %k, ptr %vslot)
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
  %sc = call i32 @universe_ds_treemap_contains(ptr %t, i64 %si)
  %spres = icmp eq i8 %sp, 1
  %swant = select i1 %spres, i32 0, i32 5
  %sbad = icmp ne i32 %sc, %swant
  %sbadz = zext i1 %sbad to i64
  %sviol.n = add i64 %sviol, %sbadz
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 4096
  br i1 %smore, label %scan, label %scan.done

scan.done:
  call void @ut_check_eq(i64 %sviol.n, i64 0, ptr @msg.il.scan)
  %fcnt = call i64 @universe_ds_treemap_size(ptr %t)
  call void @ut_check_eq(i64 %fcnt, i64 %cnt.n, ptr @msg.il.count)
  call void @universe_ds_treemap_destroy(ptr %t)
  call void @free(ptr %present)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @t_basic()
  call void @t_edges()
  call void @t_perm()
  call void @t_search()
  call void @t_range()
  call void @t_unsigned()
  call void @test_interleaved()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary

do.bench:
  call void @run_bench()
  br label %summary

summary:
  %r = call i32 @ut_summary()
  ret i32 %r
}
