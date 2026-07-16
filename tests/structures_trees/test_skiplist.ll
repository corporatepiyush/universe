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

; Tests for universe_ds_skiplist: happy path, every reachable error code, edge
; sizes (0,1,2,~64K), ordered iteration / rank / select cross-checked against a
; qsort'd reference (rank(select(i))==i), floor/ceiling/min/max, delete+re-get,
; determinism across two identical builds, and a --bench vs a sorted-array
; binary search and a B-tree.

declare ptr @universe_ds_skiplist_create()
declare void @universe_ds_skiplist_destroy(ptr)
declare i32 @universe_ds_skiplist_put(ptr, i64, i64)
declare i32 @universe_ds_skiplist_get(ptr, i64, ptr)
declare i32 @universe_ds_skiplist_contains(ptr, i64)
declare i32 @universe_ds_skiplist_delete(ptr, i64)
declare i32 @universe_ds_skiplist_min(ptr, ptr, ptr)
declare i32 @universe_ds_skiplist_max(ptr, ptr, ptr)
declare i32 @universe_ds_skiplist_floor(ptr, i64, ptr, ptr)
declare i32 @universe_ds_skiplist_ceiling(ptr, i64, ptr, ptr)
declare i32 @universe_ds_skiplist_rank(ptr, i64, ptr)
declare i32 @universe_ds_skiplist_select(ptr, i64, ptr, ptr)
declare i64 @universe_ds_skiplist_range(ptr, i64, i64, ptr, ptr, i64)
declare i64 @universe_ds_skiplist_size(ptr)

; B-tree (bench comparison)
declare ptr @universe_ds_btree_create()
declare void @universe_ds_btree_destroy(ptr)
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare void @qsort(ptr, i64, i64, ptr)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@sl.bench.samp   = internal global [16 x double] zeroinitializer, align 8
@slbt.bench.samp = internal global [16 x double] zeroinitializer, align 8
@slar.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.sl.bench   = private unnamed_addr constant [16 x i8] c"skiplist get 1M\00"
@lbl.slbt.bench = private unnamed_addr constant [13 x i8] c"btree get 1M\00"
@lbl.slar.bench = private unnamed_addr constant [22 x i8] c"sorted-arr bsearch 1M\00"

@m.basic  = private unnamed_addr constant [18 x i8] c"skiplist basic op\00"
@m.ord    = private unnamed_addr constant [26 x i8] c"skiplist ordered/rank/sel\00"
@m.range  = private unnamed_addr constant [15 x i8] c"skiplist range\00"
@m.fc     = private unnamed_addr constant [27 x i8] c"skiplist floor/ceil/minmax\00"
@m.del    = private unnamed_addr constant [16 x i8] c"skiplist delete\00"
@m.err    = private unnamed_addr constant [20 x i8] c"skiplist err/edge n\00"
@m.large  = private unnamed_addr constant [18 x i8] c"skiplist 64K keys\00"
@m.det    = private unnamed_addr constant [21 x i8] c"skiplist determinism\00"

define internal i32 @cmp_i64(ptr %a, ptr %b) {
entry:
  %av = load i64, ptr %a, align 8
  %bv = load i64, ptr %b, align 8
  %lt = icmp slt i64 %av, %bv
  %gt = icmp sgt i64 %av, %bv
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub i32 %g, %l
  ret i32 %r
}

define internal void @bump_if(i1 %bad, ptr %viol) {
entry:
  br i1 %bad, label %inc, label %done
inc:
  %v = load i64, ptr %viol, align 8
  %v.n = add nuw i64 %v, 1
  store i64 %v.n, ptr %viol, align 8
  br label %done
done:
  ret void
}

; ===========================================================================
; basic: put/get/contains/overwrite
; ===========================================================================
define internal void @test_basic() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %g = alloca i64, align 8
  %s = call ptr @universe_ds_skiplist_create()
  call i32 @universe_ds_skiplist_put(ptr %s, i64 10, i64 100)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 20, i64 200)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 30, i64 300)
  ; get present
  %r10 = call i32 @universe_ds_skiplist_get(ptr %s, i64 10, ptr %g)
  %v10 = load i64, ptr %g, align 8
  %b1 = icmp ne i32 %r10, 0
  %b2 = icmp ne i64 %v10, 100
  call void @bump_if(i1 %b1, ptr %viol)
  call void @bump_if(i1 %b2, ptr %viol)
  ; get absent
  %r15 = call i32 @universe_ds_skiplist_get(ptr %s, i64 15, ptr %g)
  %b3 = icmp ne i32 %r15, 5
  call void @bump_if(i1 %b3, ptr %viol)
  ; contains
  %c20 = call i32 @universe_ds_skiplist_contains(ptr %s, i64 20)
  %c99 = call i32 @universe_ds_skiplist_contains(ptr %s, i64 99)
  %b4 = icmp ne i32 %c20, 1
  %b5 = icmp ne i32 %c99, 0
  call void @bump_if(i1 %b4, ptr %viol)
  call void @bump_if(i1 %b5, ptr %viol)
  ; overwrite
  call i32 @universe_ds_skiplist_put(ptr %s, i64 20, i64 999)
  %r20 = call i32 @universe_ds_skiplist_get(ptr %s, i64 20, ptr %g)
  %v20 = load i64, ptr %g, align 8
  %b6 = icmp ne i64 %v20, 999
  call void @bump_if(i1 %b6, ptr %viol)
  ; size stays 3 after overwrite
  %sz = call i64 @universe_ds_skiplist_size(ptr %s)
  %b7 = icmp ne i64 %sz, 3
  call void @bump_if(i1 %b7, ptr %viol)
  call void @universe_ds_skiplist_destroy(ptr %s)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.basic)
  ret void
}

; ===========================================================================
; ordered iteration + rank/select cross-checked vs qsort'd distinct reference
; ===========================================================================
define internal void @test_ordered() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %state = alloca i64, align 8
  store i64 76543210987654321, ptr %state, align 8
  %n = add i64 0, 3000
  %shadow = call ptr @malloc(i64 24000)
  %ref = call ptr @malloc(i64 24000)
  %outk = call ptr @malloc(i64 24000)
  %outv = call ptr @malloc(i64 24000)
  %s = call ptr @universe_ds_skiplist_create()
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr %state)
  %k = urem i64 %r, 5000                ; range < n -> some duplicates
  %val = add i64 %k, 1000000007
  %sp = getelementptr inbounds i64, ptr %shadow, i64 %i
  store i64 %k, ptr %sp, align 8
  call i32 @universe_ds_skiplist_put(ptr %s, i64 %k, i64 %val)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %sort

sort:
  call void @qsort(ptr %shadow, i64 %n, i64 8, ptr @cmp_i64)
  ; dedup shadow -> ref[m]
  br label %dd.head

dd.head:
  %di = phi i64 [ 0, %sort ], [ %di.n, %dd.next ]
  %m = phi i64 [ 0, %sort ], [ %m.n, %dd.next ]
  %dcond = icmp ult i64 %di, %n
  br i1 %dcond, label %dd.body, label %dd.done

dd.body:
  %dsp = getelementptr inbounds i64, ptr %shadow, i64 %di
  %dk = load i64, ptr %dsp, align 8
  %isfirst = icmp eq i64 %di, 0
  br i1 %isfirst, label %dd.emit, label %dd.cmp

dd.cmp:
  %dim1 = add i64 %di, -1
  %dpp = getelementptr inbounds i64, ptr %shadow, i64 %dim1
  %dpk = load i64, ptr %dpp, align 8
  %distinct = icmp ne i64 %dk, %dpk
  br i1 %distinct, label %dd.emit, label %dd.skip

dd.emit:
  %rmp = getelementptr inbounds i64, ptr %ref, i64 %m
  store i64 %dk, ptr %rmp, align 8
  %m.e = add i64 %m, 1
  br label %dd.next

dd.skip:
  br label %dd.next

dd.next:
  %m.n = phi i64 [ %m.e, %dd.emit ], [ %m, %dd.skip ]
  %di.n = add nuw i64 %di, 1
  br label %dd.head

dd.done:
  ; size == m
  %sz = call i64 @universe_ds_skiplist_size(ptr %s)
  %bsz = icmp ne i64 %sz, %m
  call void @bump_if(i1 %bsz, ptr %viol)
  ; per-element: select(i)==ref[i], its val, rank(ref[i])==i, get==val
  br label %chk.head

chk.head:
  %ci = phi i64 [ 0, %dd.done ], [ %ci.n, %chk.body ]
  %ccond = icmp ult i64 %ci, %m
  br i1 %ccond, label %chk.body, label %chk.range

chk.body:
  %sk = alloca i64, align 8
  %sv = alloca i64, align 8
  %rk = alloca i64, align 8
  %gv = alloca i64, align 8
  %refi = getelementptr inbounds i64, ptr %ref, i64 %ci
  %refk = load i64, ptr %refi, align 8
  %expv = add i64 %refk, 1000000007
  ; select
  %rs = call i32 @universe_ds_skiplist_select(ptr %s, i64 %ci, ptr %sk, ptr %sv)
  %skv = load i64, ptr %sk, align 8
  %svv = load i64, ptr %sv, align 8
  %e1 = icmp ne i64 %skv, %refk
  %e2 = icmp ne i64 %svv, %expv
  %e0 = icmp ne i32 %rs, 0
  call void @bump_if(i1 %e0, ptr %viol)
  call void @bump_if(i1 %e1, ptr %viol)
  call void @bump_if(i1 %e2, ptr %viol)
  ; rank(select(i)) == i
  %rr = call i32 @universe_ds_skiplist_rank(ptr %s, i64 %refk, ptr %rk)
  %rkv = load i64, ptr %rk, align 8
  %e3 = icmp ne i64 %rkv, %ci
  %e3b = icmp ne i32 %rr, 0
  call void @bump_if(i1 %e3, ptr %viol)
  call void @bump_if(i1 %e3b, ptr %viol)
  ; get
  %rg = call i32 @universe_ds_skiplist_get(ptr %s, i64 %refk, ptr %gv)
  %gvv = load i64, ptr %gv, align 8
  %e4 = icmp ne i64 %gvv, %expv
  call void @bump_if(i1 %e4, ptr %viol)
  %ci.n = add nuw i64 %ci, 1
  br label %chk.head

chk.range:
  ; full-domain range returns all m keys in order
  %cnt = call i64 @universe_ds_skiplist_range(ptr %s, i64 -9223372036854775808, i64 9223372036854775807, ptr %outk, ptr %outv, i64 3000)
  %rbad = icmp ne i64 %cnt, %m
  call void @bump_if(i1 %rbad, ptr %viol)
  br label %rg.head

rg.head:
  %gi = phi i64 [ 0, %chk.range ], [ %gi.n, %rg.body ]
  %rgcond = icmp ult i64 %gi, %m
  br i1 %rgcond, label %rg.body, label %fin

rg.body:
  %okp = getelementptr inbounds i64, ptr %outk, i64 %gi
  %okv = load i64, ptr %okp, align 8
  %rfp = getelementptr inbounds i64, ptr %ref, i64 %gi
  %rfv = load i64, ptr %rfp, align 8
  %rbad2 = icmp ne i64 %okv, %rfv
  call void @bump_if(i1 %rbad2, ptr %viol)
  %gi.n = add nuw i64 %gi, 1
  br label %rg.head

fin:
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.ord)
  call void @universe_ds_skiplist_destroy(ptr %s)
  call void @free(ptr %shadow)
  call void @free(ptr %ref)
  call void @free(ptr %outk)
  call void @free(ptr %outv)
  ret void
}

; ===========================================================================
; floor / ceiling / min / max
; ===========================================================================
define internal void @test_floor_ceiling() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  %s = call ptr @universe_ds_skiplist_create()
  ; keys 10,20,30,40,50
  call i32 @universe_ds_skiplist_put(ptr %s, i64 10, i64 11)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 20, i64 21)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 30, i64 31)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 40, i64 41)
  call i32 @universe_ds_skiplist_put(ptr %s, i64 50, i64 51)
  ; floor(25)=20 ; floor(30)=30 ; floor(5)=NOT_FOUND
  call i32 @universe_ds_skiplist_floor(ptr %s, i64 25, ptr %ok, ptr %ov)
  %f25 = load i64, ptr %ok, align 8
  %b1 = icmp ne i64 %f25, 20
  call void @bump_if(i1 %b1, ptr %viol)
  call i32 @universe_ds_skiplist_floor(ptr %s, i64 30, ptr %ok, ptr %ov)
  %f30 = load i64, ptr %ok, align 8
  %b2 = icmp ne i64 %f30, 30
  call void @bump_if(i1 %b2, ptr %viol)
  %rf5 = call i32 @universe_ds_skiplist_floor(ptr %s, i64 5, ptr %ok, ptr %ov)
  %b3 = icmp ne i32 %rf5, 5
  call void @bump_if(i1 %b3, ptr %viol)
  ; ceiling(25)=30 ; ceiling(30)=30 ; ceiling(55)=NOT_FOUND
  call i32 @universe_ds_skiplist_ceiling(ptr %s, i64 25, ptr %ok, ptr %ov)
  %c25 = load i64, ptr %ok, align 8
  %b4 = icmp ne i64 %c25, 30
  call void @bump_if(i1 %b4, ptr %viol)
  call i32 @universe_ds_skiplist_ceiling(ptr %s, i64 30, ptr %ok, ptr %ov)
  %c30 = load i64, ptr %ok, align 8
  %b5 = icmp ne i64 %c30, 30
  call void @bump_if(i1 %b5, ptr %viol)
  %rc55 = call i32 @universe_ds_skiplist_ceiling(ptr %s, i64 55, ptr %ok, ptr %ov)
  %b6 = icmp ne i32 %rc55, 5
  call void @bump_if(i1 %b6, ptr %viol)
  ; min=10 (val 11), max=50 (val 51)
  call i32 @universe_ds_skiplist_min(ptr %s, ptr %ok, ptr %ov)
  %mnk = load i64, ptr %ok, align 8
  %mnv = load i64, ptr %ov, align 8
  %b7 = icmp ne i64 %mnk, 10
  %b8 = icmp ne i64 %mnv, 11
  call void @bump_if(i1 %b7, ptr %viol)
  call void @bump_if(i1 %b8, ptr %viol)
  call i32 @universe_ds_skiplist_max(ptr %s, ptr %ok, ptr %ov)
  %mxk = load i64, ptr %ok, align 8
  %mxv = load i64, ptr %ov, align 8
  %b9 = icmp ne i64 %mxk, 50
  %b10 = icmp ne i64 %mxv, 51
  call void @bump_if(i1 %b9, ptr %viol)
  call void @bump_if(i1 %b10, ptr %viol)
  call void @universe_ds_skiplist_destroy(ptr %s)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.fc)
  ret void
}

; ===========================================================================
; delete then re-get; span integrity after deletes checked via rank/select
; ===========================================================================
define internal void @test_delete() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %g = alloca i64, align 8
  %s = call ptr @universe_ds_skiplist_create()
  %n = add i64 0, 1000
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %val = add i64 %i, 7
  call i32 @universe_ds_skiplist_put(ptr %s, i64 %i, i64 %val)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %del
del:
  ; delete all even keys
  %di = phi i64 [ 0, %fill ], [ %di.n, %del ]
  call i32 @universe_ds_skiplist_delete(ptr %s, i64 %di)
  %di.n = add nuw i64 %di, 2
  %dmore = icmp ult i64 %di.n, %n
  br i1 %dmore, label %del, label %delmiss
delmiss:
  ; deleting an absent key returns 5
  %rmiss = call i32 @universe_ds_skiplist_delete(ptr %s, i64 4)
  %bmiss = icmp ne i32 %rmiss, 5
  call void @bump_if(i1 %bmiss, ptr %viol)
  ; size == 500
  %sz = call i64 @universe_ds_skiplist_size(ptr %s)
  %bsz = icmp ne i64 %sz, 500
  call void @bump_if(i1 %bsz, ptr %viol)
  br label %chk
chk:
  %ci = phi i64 [ 0, %delmiss ], [ %ci.n, %chk ]
  %viol2 = phi i64 [ 0, %delmiss ], [ %viol2.f, %chk ]
  ; even -> absent (5); odd -> present with val
  %isodd = and i64 %ci, 1
  %odd = icmp eq i64 %isodd, 1
  %rget = call i32 @universe_ds_skiplist_get(ptr %s, i64 %ci, ptr %g)
  %gv = load i64, ptr %g, align 8
  %expv = add i64 %ci, 7
  %present = icmp eq i32 %rget, 0
  %valok = icmp eq i64 %gv, %expv
  ; expected: odd => present&&valok ; even => !present (rget==5)
  %oddgood = and i1 %present, %valok
  %evengood = icmp eq i32 %rget, 5
  %good = select i1 %odd, i1 %oddgood, i1 %evengood
  %bad = xor i1 %good, true
  %inc = zext i1 %bad to i64
  %viol2.f = add i64 %viol2, %inc
  %ci.n = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, %n
  br i1 %cmore, label %chk, label %chkrank
chkrank:
  ; remaining keys are 1,3,5,...,999 -> rank(2k+1)==k, select(k)==2k+1
  %rk = alloca i64, align 8
  %sk = alloca i64, align 8
  %sv = alloca i64, align 8
  br label %rankloop
rankloop:
  %ri = phi i64 [ 0, %chkrank ], [ %ri.n, %rankloop ]
  %viol3 = phi i64 [ %viol2.f, %chkrank ], [ %viol3.f, %rankloop ]
  %key = add i64 %ri, %ri
  %key1 = add i64 %key, 1            ; 2*ri + 1
  %rr = call i32 @universe_ds_skiplist_rank(ptr %s, i64 %key1, ptr %rk)
  %rkv = load i64, ptr %rk, align 8
  %rbad = icmp ne i64 %rkv, %ri
  %rs = call i32 @universe_ds_skiplist_select(ptr %s, i64 %ri, ptr %sk, ptr %sv)
  %skv = load i64, ptr %sk, align 8
  %sbad = icmp ne i64 %skv, %key1
  %anybad = or i1 %rbad, %sbad
  %inc3 = zext i1 %anybad to i64
  %viol3.f = add i64 %viol3, %inc3
  %ri.n = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, 500
  br i1 %rmore, label %rankloop, label %fin
fin:
  %tot = load i64, ptr %viol, align 8
  %all = add i64 %tot, %viol3.f
  call void @ut_check_eq(i64 %all, i64 0, ptr @m.del)
  call void @universe_ds_skiplist_destroy(ptr %s)
  ret void
}

; ===========================================================================
; errors + edge sizes 0,1,2
; ===========================================================================
define internal void @test_errors() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  ; null handle
  %e.put = call i32 @universe_ds_skiplist_put(ptr null, i64 1, i64 1)
  %e.get = call i32 @universe_ds_skiplist_get(ptr null, i64 1, ptr %ok)
  %e.del = call i32 @universe_ds_skiplist_delete(ptr null, i64 1)
  %b1 = icmp ne i32 %e.put, 1
  %b2 = icmp ne i32 %e.get, 1
  %b3 = icmp ne i32 %e.del, 1
  call void @bump_if(i1 %b1, ptr %viol)
  call void @bump_if(i1 %b2, ptr %viol)
  call void @bump_if(i1 %b3, ptr %viol)
  ; n=0: min/max empty(4), select oob(7), rank not-found(5)
  %s = call ptr @universe_ds_skiplist_create()
  %e.min = call i32 @universe_ds_skiplist_min(ptr %s, ptr %ok, ptr %ov)
  %e.max = call i32 @universe_ds_skiplist_max(ptr %s, ptr %ok, ptr %ov)
  %e.sel = call i32 @universe_ds_skiplist_select(ptr %s, i64 0, ptr %ok, ptr %ov)
  %e.rnk = call i32 @universe_ds_skiplist_rank(ptr %s, i64 5, ptr %ok)
  %b4 = icmp ne i32 %e.min, 4
  %b5 = icmp ne i32 %e.max, 4
  %b6 = icmp ne i32 %e.sel, 7
  %b7 = icmp ne i32 %e.rnk, 5
  call void @bump_if(i1 %b4, ptr %viol)
  call void @bump_if(i1 %b5, ptr %viol)
  call void @bump_if(i1 %b6, ptr %viol)
  call void @bump_if(i1 %b7, ptr %viol)
  ; n=1
  call i32 @universe_ds_skiplist_put(ptr %s, i64 42, i64 4242)
  call i32 @universe_ds_skiplist_min(ptr %s, ptr %ok, ptr %ov)
  %n1mn = load i64, ptr %ok, align 8
  call i32 @universe_ds_skiplist_max(ptr %s, ptr %ok, ptr %ov)
  %n1mx = load i64, ptr %ok, align 8
  %e.sel1 = call i32 @universe_ds_skiplist_select(ptr %s, i64 0, ptr %ok, ptr %ov)
  %n1s = load i64, ptr %ok, align 8
  %e.selo = call i32 @universe_ds_skiplist_select(ptr %s, i64 1, ptr %ok, ptr %ov)
  %b8 = icmp ne i64 %n1mn, 42
  %b9 = icmp ne i64 %n1mx, 42
  %b10 = icmp ne i64 %n1s, 42
  %b11 = icmp ne i32 %e.selo, 7
  call void @bump_if(i1 %b8, ptr %viol)
  call void @bump_if(i1 %b9, ptr %viol)
  call void @bump_if(i1 %b10, ptr %viol)
  call void @bump_if(i1 %b11, ptr %viol)
  ; n=2
  call i32 @universe_ds_skiplist_put(ptr %s, i64 7, i64 77)
  call i32 @universe_ds_skiplist_select(ptr %s, i64 0, ptr %ok, ptr %ov)
  %n2s0 = load i64, ptr %ok, align 8
  call i32 @universe_ds_skiplist_select(ptr %s, i64 1, ptr %ok, ptr %ov)
  %n2s1 = load i64, ptr %ok, align 8
  %b12 = icmp ne i64 %n2s0, 7
  %b13 = icmp ne i64 %n2s1, 42
  call void @bump_if(i1 %b12, ptr %viol)
  call void @bump_if(i1 %b13, ptr %viol)
  ; delete down to empty
  call i32 @universe_ds_skiplist_delete(ptr %s, i64 7)
  call i32 @universe_ds_skiplist_delete(ptr %s, i64 42)
  %sz0 = call i64 @universe_ds_skiplist_size(ptr %s)
  %b14 = icmp ne i64 %sz0, 0
  call void @bump_if(i1 %b14, ptr %viol)
  call void @universe_ds_skiplist_destroy(ptr %s)
  call void @universe_ds_skiplist_destroy(ptr null)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.err)
  ret void
}

; ===========================================================================
; large: 65536 keys 0..65535 inserted in a shuffled order
; ===========================================================================
define internal void @test_large() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %state = alloca i64, align 8
  store i64 13579246801234567, ptr %state, align 8
  %n = add i64 0, 65536
  %perm = call ptr @malloc(i64 524288)   ; 65536 i64
  %s = call ptr @universe_ds_skiplist_create()
  ; perm[i]=i
  br label %init
init:
  %i = phi i64 [ 0, %entry ], [ %i.n, %init ]
  %pp = getelementptr inbounds i64, ptr %perm, i64 %i
  store i64 %i, ptr %pp, align 8
  %i.n = add nuw i64 %i, 1
  %m0 = icmp ult i64 %i.n, %n
  br i1 %m0, label %init, label %shuf
shuf:
  ; Fisher-Yates from the top
  %j = phi i64 [ 65535, %init ], [ %j.n, %shuf ]
  %rr = call i64 @ut_rand(ptr %state)
  %jp1 = add i64 %j, 1
  %sw = urem i64 %rr, %jp1
  %jp = getelementptr inbounds i64, ptr %perm, i64 %j
  %swp = getelementptr inbounds i64, ptr %perm, i64 %sw
  %jv = load i64, ptr %jp, align 8
  %swv = load i64, ptr %swp, align 8
  store i64 %swv, ptr %jp, align 8
  store i64 %jv, ptr %swp, align 8
  %j.z = icmp eq i64 %j, 0
  %j.n = add i64 %j, -1
  br i1 %j.z, label %ins, label %shuf
ins:
  %ii = phi i64 [ 0, %shuf ], [ %ii.n, %ins ]
  %ipp = getelementptr inbounds i64, ptr %perm, i64 %ii
  %key = load i64, ptr %ipp, align 8
  %val = add i64 %key, 3
  call i32 @universe_ds_skiplist_put(ptr %s, i64 %key, i64 %val)
  %ii.n = add nuw i64 %ii, 1
  %m1 = icmp ult i64 %ii.n, %n
  br i1 %m1, label %ins, label %verify
verify:
  %sz = call i64 @universe_ds_skiplist_size(ptr %s)
  %bsz = icmp ne i64 %sz, %n
  call void @bump_if(i1 %bsz, ptr %viol)
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  call i32 @universe_ds_skiplist_min(ptr %s, ptr %ok, ptr %ov)
  %mn = load i64, ptr %ok, align 8
  %bmn = icmp ne i64 %mn, 0
  call void @bump_if(i1 %bmn, ptr %viol)
  call i32 @universe_ds_skiplist_max(ptr %s, ptr %ok, ptr %ov)
  %mx = load i64, ptr %ok, align 8
  %bmx = icmp ne i64 %mx, 65535
  call void @bump_if(i1 %bmx, ptr %viol)
  ; spot-check select(i)==i and rank(i)==i and value
  br label %vchk
vchk:
  %vi = phi i64 [ 0, %verify ], [ %vi.n, %vchk ]
  %viol2 = phi i64 [ 0, %verify ], [ %viol2.f, %vchk ]
  %sk = alloca i64, align 8
  %sv = alloca i64, align 8
  %rk = alloca i64, align 8
  call i32 @universe_ds_skiplist_select(ptr %s, i64 %vi, ptr %sk, ptr %sv)
  %skv = load i64, ptr %sk, align 8
  %svv = load i64, ptr %sv, align 8
  %ev = add i64 %vi, 3
  %e1 = icmp ne i64 %skv, %vi
  %e2 = icmp ne i64 %svv, %ev
  call i32 @universe_ds_skiplist_rank(ptr %s, i64 %vi, ptr %rk)
  %rkv = load i64, ptr %rk, align 8
  %e3 = icmp ne i64 %rkv, %vi
  %o1 = or i1 %e1, %e2
  %o2 = or i1 %o1, %e3
  %inc = zext i1 %o2 to i64
  %viol2.f = add i64 %viol2, %inc
  ; step by 137 to sample across the whole range
  %vi.n = add nuw i64 %vi, 137
  %vmore = icmp ult i64 %vi.n, %n
  br i1 %vmore, label %vchk, label %fin
fin:
  %t1 = load i64, ptr %viol, align 8
  %all = add i64 %t1, %viol2.f
  call void @ut_check_eq(i64 %all, i64 0, ptr @m.large)
  call void @universe_ds_skiplist_destroy(ptr %s)
  call void @free(ptr %perm)
  ret void
}

; ===========================================================================
; determinism: two identical build sequences agree on select/rank/size
; ===========================================================================
define internal void @test_determinism() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %a = call ptr @universe_ds_skiplist_create()
  %b = call ptr @universe_ds_skiplist_create()
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  ; a fixed permutation key = (i*137+11) mod 1000
  %m = mul i64 %i, 137
  %m2 = add i64 %m, 11
  %key = urem i64 %m2, 1000
  %val = add i64 %key, 500
  call i32 @universe_ds_skiplist_put(ptr %a, i64 %key, i64 %val)
  call i32 @universe_ds_skiplist_put(ptr %b, i64 %key, i64 %val)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %fill, label %cmp
cmp:
  %sza = call i64 @universe_ds_skiplist_size(ptr %a)
  %szb = call i64 @universe_ds_skiplist_size(ptr %b)
  %bsz = icmp ne i64 %sza, %szb
  call void @bump_if(i1 %bsz, ptr %viol)
  br label %loop
loop:
  %ci = phi i64 [ 0, %cmp ], [ %ci.n, %loop ]
  %viol2 = phi i64 [ 0, %cmp ], [ %viol2.f, %loop ]
  %ka = alloca i64, align 8
  %kb = alloca i64, align 8
  %va = alloca i64, align 8
  %vb = alloca i64, align 8
  %ra = call i32 @universe_ds_skiplist_select(ptr %a, i64 %ci, ptr %ka, ptr %va)
  %rb = call i32 @universe_ds_skiplist_select(ptr %b, i64 %ci, ptr %kb, ptr %vb)
  %kav = load i64, ptr %ka, align 8
  %kbv = load i64, ptr %kb, align 8
  %vav = load i64, ptr %va, align 8
  %vbv = load i64, ptr %vb, align 8
  %d1 = icmp ne i64 %kav, %kbv
  %d2 = icmp ne i64 %vav, %vbv
  %dd = or i1 %d1, %d2
  %inc = zext i1 %dd to i64
  %viol2.f = add i64 %viol2, %inc
  %ci.n = add nuw i64 %ci, 1
  %lmore = icmp ult i64 %ci.n, %sza
  br i1 %lmore, label %loop, label %fin
fin:
  %t = load i64, ptr %viol, align 8
  %all = add i64 %t, %viol2.f
  call void @ut_check_eq(i64 %all, i64 0, ptr @m.det)
  call void @universe_ds_skiplist_destroy(ptr %a)
  call void @universe_ds_skiplist_destroy(ptr %b)
  ret void
}

; ===========================================================================
; bench: skiplist get vs btree get vs sorted-array binary search
; ===========================================================================
define internal void @bench() {
entry:
  %state = alloca i64, align 8
  %g = alloca i64, align 8
  %vsink = alloca i64, align 8
  %n = add i64 0, 100000
  %queries = add i64 0, 1000000
  %arr = call ptr @malloc(i64 800000)    ; 100k sorted keys
  %s = call ptr @universe_ds_skiplist_create()
  %bt = call ptr @universe_ds_btree_create()
  br label %build
build:
  %i = phi i64 [ 0, %entry ], [ %i.n, %build ]
  %key = mul i64 %i, 2                    ; even keys 0,2,4,...
  %ap = getelementptr inbounds i64, ptr %arr, i64 %i
  store i64 %key, ptr %ap, align 8
  %val = add i64 %key, 9
  call i32 @universe_ds_skiplist_put(ptr %s, i64 %key, i64 %val)
  call i32 @universe_ds_btree_insert(ptr %bt, i64 %key, i64 %val)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %build, label %sl.rep.head
; --- skiplist get (read-only, warm-up rep discarded) ---
sl.rep.head:
  %slrep = phi i64 [ 0, %build ], [ %slrep.n, %sl.rep.cont ]
  store i64 555777999111333555, ptr %state, align 8
  %t0 = call double @ut_now_sec()
  br label %sl.loop
sl.loop:
  %qi = phi i64 [ 0, %sl.rep.head ], [ %qi.n, %sl.loop ]
  %acc = phi i64 [ 0, %sl.rep.head ], [ %acc.n, %sl.loop ]
  %r = call i64 @ut_rand(ptr %state)
  %kk = urem i64 %r, 200000
  %rc = call i32 @universe_ds_skiplist_get(ptr %s, i64 %kk, ptr %g)
  %hit = icmp eq i32 %rc, 0
  %gv = load i64, ptr %g, align 8
  %add = select i1 %hit, i64 %gv, i64 0
  %acc.n = add i64 %acc, %add
  %qi.n = add nuw i64 %qi, 1
  %qm = icmp ult i64 %qi.n, %queries
  br i1 %qm, label %sl.loop, label %sl.rep.done
sl.rep.done:
  %t1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr %vsink, align 8
  %slelapsed = fsub double %t1, %t0
  %slwarm = icmp eq i64 %slrep, 0
  br i1 %slwarm, label %sl.rep.cont, label %sl.rep.store
sl.rep.store:
  %slsi = sub i64 %slrep, 1
  %slsp = getelementptr inbounds double, ptr @sl.bench.samp, i64 %slsi
  store double %slelapsed, ptr %slsp, align 8
  br label %sl.rep.cont
sl.rep.cont:
  %slrep.n = add nuw i64 %slrep, 1
  %slmore = icmp ult i64 %slrep.n, 17
  br i1 %slmore, label %sl.rep.head, label %sl.rep.report
sl.rep.report:
  call void @ut_report_dist(ptr @sl.bench.samp, i64 16, i64 1000000, ptr @lbl.sl.bench)
  br label %bt.rep.head
; --- btree get (read-only) ---
bt.rep.head:
  %btrep = phi i64 [ 0, %sl.rep.report ], [ %btrep.n, %bt.rep.cont ]
  store i64 555777999111333555, ptr %state, align 8
  %bt0 = call double @ut_now_sec()
  br label %bt.loop
bt.loop:
  %bqi = phi i64 [ 0, %bt.rep.head ], [ %bqi.n, %bt.loop ]
  %bacc = phi i64 [ 0, %bt.rep.head ], [ %bacc.n, %bt.loop ]
  %br = call i64 @ut_rand(ptr %state)
  %bkk = urem i64 %br, 200000
  %brc = call i32 @universe_ds_btree_find(ptr %bt, i64 %bkk, ptr %g)
  %bhit = icmp eq i32 %brc, 0
  %bgv = load i64, ptr %g, align 8
  %badd = select i1 %bhit, i64 %bgv, i64 0
  %bacc.n = add i64 %bacc, %badd
  %bqi.n = add nuw i64 %bqi, 1
  %bqm = icmp ult i64 %bqi.n, %queries
  br i1 %bqm, label %bt.loop, label %bt.rep.done
bt.rep.done:
  %bt1 = call double @ut_now_sec()
  store volatile i64 %bacc.n, ptr %vsink, align 8
  %btelapsed = fsub double %bt1, %bt0
  %btwarm = icmp eq i64 %btrep, 0
  br i1 %btwarm, label %bt.rep.cont, label %bt.rep.store
bt.rep.store:
  %btsi = sub i64 %btrep, 1
  %btsp = getelementptr inbounds double, ptr @slbt.bench.samp, i64 %btsi
  store double %btelapsed, ptr %btsp, align 8
  br label %bt.rep.cont
bt.rep.cont:
  %btrep.n = add nuw i64 %btrep, 1
  %btmore = icmp ult i64 %btrep.n, 17
  br i1 %btmore, label %bt.rep.head, label %bt.rep.report
bt.rep.report:
  call void @ut_report_dist(ptr @slbt.bench.samp, i64 16, i64 1000000, ptr @lbl.slbt.bench)
  br label %ar.rep.head
; --- sorted-array binary search (read-only) ---
ar.rep.head:
  %arrep = phi i64 [ 0, %bt.rep.report ], [ %arrep.n, %ar.rep.cont ]
  store i64 555777999111333555, ptr %state, align 8
  %at0 = call double @ut_now_sec()
  br label %ar.loop
ar.loop:
  %aqi = phi i64 [ 0, %ar.rep.head ], [ %aqi.n, %ar.next ]
  %aacc = phi i64 [ 0, %ar.rep.head ], [ %aacc.n, %ar.next ]
  %ar = call i64 @ut_rand(ptr %state)
  %akk = urem i64 %ar, 200000
  br label %bs
bs:
  %lo = phi i64 [ 0, %ar.loop ], [ %lo.n, %bs.step ]
  %hi = phi i64 [ %n, %ar.loop ], [ %hi.n, %bs.step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %bs.step, label %bs.done
bs.step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %mp = getelementptr inbounds i64, ptr %arr, i64 %mid
  %mv = load i64, ptr %mp, align 8
  %ltk = icmp slt i64 %mv, %akk
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %ltk, i64 %mid1, i64 %lo
  %hi.n = select i1 %ltk, i64 %hi, i64 %mid
  br label %bs
bs.done:
  %inb = icmp ult i64 %lo, %n
  br i1 %inb, label %bs.chk, label %ar.next
bs.chk:
  %fp = getelementptr inbounds i64, ptr %arr, i64 %lo
  %fv = load i64, ptr %fp, align 8
  %found = icmp eq i64 %fv, %akk
  %fadd = select i1 %found, i64 %fv, i64 0
  %accplus = add i64 %aacc, %fadd
  br label %ar.next
ar.next:
  %aacc.n = phi i64 [ %aacc, %bs.done ], [ %accplus, %bs.chk ]
  %aqi.n = add nuw i64 %aqi, 1
  %aqm = icmp ult i64 %aqi.n, %queries
  br i1 %aqm, label %ar.loop, label %ar.rep.done
ar.rep.done:
  %at1 = call double @ut_now_sec()
  store volatile i64 %aacc.n, ptr %vsink, align 8
  %arelapsed = fsub double %at1, %at0
  %arwarm = icmp eq i64 %arrep, 0
  br i1 %arwarm, label %ar.rep.cont, label %ar.rep.store
ar.rep.store:
  %arsi = sub i64 %arrep, 1
  %arsp = getelementptr inbounds double, ptr @slar.bench.samp, i64 %arsi
  store double %arelapsed, ptr %arsp, align 8
  br label %ar.rep.cont
ar.rep.cont:
  %arrep.n = add nuw i64 %arrep, 1
  %armore = icmp ult i64 %arrep.n, 17
  br i1 %armore, label %ar.rep.head, label %ar.rep.report
ar.rep.report:
  call void @ut_report_dist(ptr @slar.bench.samp, i64 16, i64 1000000, ptr @lbl.slar.bench)
  call void @universe_ds_skiplist_destroy(ptr %s)
  call void @universe_ds_btree_destroy(ptr %bt)
  call void @free(ptr %arr)
  ret void
}

; randomized INTERLEAVED put/delete vs a bounded-domain shadow (data-dependent
; deletes exercise arbitrary span-repair orderings, unlike the fixed schedules
; in test_delete). Invariant after every op: contains(k) == shadow[k].
@msg.il.state = private unnamed_addr constant [25 x i8] c"interleaved contains==sh\00"
@msg.il.scan  = private unnamed_addr constant [22 x i8] c"interleaved full scan\00"
@msg.il.count = private unnamed_addr constant [20 x i8] c"interleaved size ok\00"

define void @test_interleaved() {
entry:
  %seed = alloca i64, align 8
  store i64 -4265267296055464877, ptr %seed, align 8
  %vslot = alloca i64, align 8
  %present = call ptr @calloc(i64 4096, i64 1)
  %t = call ptr @universe_ds_skiplist_create()
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
  %irc = call i32 @universe_ds_skiplist_put(ptr %t, i64 %k, i64 %val)
  %was0 = icmp eq i8 %was, 0
  %cinc = zext i1 %was0 to i64
  %cnt.ins = add i64 %cnt, %cinc
  store i8 1, ptr %pp, align 1
  br label %chk

do.del:
  %drc = call i32 @universe_ds_skiplist_delete(ptr %t, i64 %k)
  %was1 = icmp eq i8 %was, 1
  %cdec = zext i1 %was1 to i64
  %cnt.del = sub i64 %cnt, %cdec
  store i8 0, ptr %pp, align 1
  br label %chk

chk:
  %cnt.n = phi i64 [ %cnt.ins, %do.ins ], [ %cnt.del, %do.del ]
  %exp = phi i8 [ 1, %do.ins ], [ 0, %do.del ]
  %c = call i32 @universe_ds_skiplist_contains(ptr %t, i64 %k)
  %expi = zext i8 %exp to i32
  %cbad = icmp ne i32 %c, %expi
  %cbadz = zext i1 %cbad to i64
  %viol.n1 = add i64 %viol, %cbadz
  %ispres = icmp eq i8 %exp, 1
  br i1 %ispres, label %vchk, label %op.tail

vchk:
  %frc = call i32 @universe_ds_skiplist_get(ptr %t, i64 %k, ptr %vslot)
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
  %sc = call i32 @universe_ds_skiplist_contains(ptr %t, i64 %si)
  %sexp = zext i8 %sp to i32
  %sbad = icmp ne i32 %sc, %sexp
  %sbadz = zext i1 %sbad to i64
  %sviol.n = add i64 %sviol, %sbadz
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 4096
  br i1 %smore, label %scan, label %scan.done

scan.done:
  call void @ut_check_eq(i64 %sviol.n, i64 0, ptr @msg.il.scan)
  %fcnt = call i64 @universe_ds_skiplist_size(ptr %t)
  call void @ut_check_eq(i64 %fcnt, i64 %cnt.n, ptr @msg.il.count)
  call void @universe_ds_skiplist_destroy(ptr %t)
  call void @free(ptr %present)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_ordered()
  call void @test_floor_ceiling()
  call void @test_delete()
  call void @test_errors()
  call void @test_large()
  call void @test_determinism()
  call void @test_interleaved()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish
do.bench:
  call void @bench()
  br label %finish
finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
