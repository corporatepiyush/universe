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

; Tests for probabilistic membership filters: cuckoo, XOR, blocked bloom.
; Core contract: NO false negatives + a bounded false-positive rate.
; Inserted keys are 0..N-1 ; absent probe keys are N..2N-1 (disjoint).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)

declare ptr @universe_ds_cuckoo_create(i64)
declare void @universe_ds_cuckoo_destroy(ptr)
declare i32 @universe_ds_cuckoo_add(ptr, i64)
declare i32 @universe_ds_cuckoo_contains(ptr, i64)
declare i32 @universe_ds_cuckoo_delete(ptr, i64)
declare i64 @universe_ds_cuckoo_count(ptr)
declare i64 @universe_ds_cuckoo_capacity(ptr)
declare ptr @universe_ds_xor_build(ptr, i64)
declare void @universe_ds_xor_destroy(ptr)
declare i32 @universe_ds_xor_contains(ptr, i64)
declare ptr @universe_ds_bbloom_create(i64, i64)
declare void @universe_ds_bbloom_destroy(ptr)
declare i32 @universe_ds_bbloom_add(ptr, i64)
declare i32 @universe_ds_bbloom_contains(ptr, i64)

@s.cuk.create   = private constant [17 x i8] c"cuckoo create ok\00"
@s.cuk.noadd    = private constant [23 x i8] c"cuckoo all adds fit ok\00"
@s.cuk.nofn     = private constant [26 x i8] c"cuckoo no false negatives\00"
@s.cuk.fpr      = private constant [21 x i8] c"cuckoo fpr < ceiling\00"
@s.cuk.empty    = private constant [21 x i8] c"cuckoo empty->absent\00"
@s.cuk.single   = private constant [18 x i8] c"cuckoo single key\00"
@s.cuk.full     = private constant [22 x i8] c"cuckoo FULL reachable\00"
@s.cuk.survive  = private constant [27 x i8] c"cuckoo delete survivors ok\00"
@s.cuk.delgone  = private constant [27 x i8] c"cuckoo deleted mostly gone\00"
@s.cuk.count    = private constant [23 x i8] c"cuckoo count invariant\00"
@s.xor.build    = private constant [13 x i8] c"xor build ok\00"
@s.xor.nofn     = private constant [23 x i8] c"xor no false negatives\00"
@s.xor.fpr      = private constant [18 x i8] c"xor fpr < ceiling\00"
@s.xor.det      = private constant [18 x i8] c"xor deterministic\00"
@s.xor.one      = private constant [23 x i8] c"xor single key present\00"
@s.xor.empty0   = private constant [16 x i8] c"xor empty build\00"
@s.bb.create    = private constant [17 x i8] c"bbloom create ok\00"
@s.bb.nofn      = private constant [26 x i8] c"bbloom no false negatives\00"
@s.bb.fpr       = private constant [21 x i8] c"bbloom fpr < ceiling\00"
@s.bb.empty     = private constant [21 x i8] c"bbloom empty->absent\00"
@s.bb.single    = private constant [18 x i8] c"bbloom single key\00"

@f.cuk.m  = private constant [36 x i8] c"  cuckoo:  FPR %lld/10000 (permil)\0A\00"
@f.xor.m  = private constant [36 x i8] c"  xor:     FPR %lld/10000 (permil)\0A\00"
@f.bb.m   = private constant [36 x i8] c"  bbloom:  FPR %lld/10000 (permil)\0A\00"
@f.bench  = private constant [40 x i8] c"  %s contains: %10lld ops/s (acc=%lld)\0A\00"
@f.bhdr   = private constant [28 x i8] c"-- bench contains ops/s --\0A\00"
@n.cuk    = private constant [7 x i8] c"cuckoo\00"
@n.xor    = private constant [7 x i8] c"xor   \00"
@n.bb     = private constant [7 x i8] c"bbloom\00"
@n.naive  = private constant [7 x i8] c"naive \00"
@lbl.f_cuckoo = private unnamed_addr constant [16 x i8] c"cuckoo contains\00"
@lbl.f_xor    = private unnamed_addr constant [13 x i8] c"xor contains\00"
@lbl.f_bbloom = private unnamed_addr constant [16 x i8] c"bbloom contains\00"
@lbl.f_naive  = private unnamed_addr constant [15 x i8] c"naive contains\00"
@filters.samp = internal global [16 x double] zeroinitializer, align 8

; ======================================================================
;  CUCKOO tests
; ======================================================================
define void @test_cuckoo() {
entry:
  %N = add i64 0, 20000
  %f = call ptr @universe_ds_cuckoo_create(i64 20000)
  %nn = icmp ne ptr %f, null
  call void @ut_check(i1 %nn, ptr @s.cuk.create)

  ; empty -> everything absent (sample 1000 of the absent range)
  br label %el

el:
  %ei = phi i64 [ 0, %entry ], [ %ei.n, %el ]
  %eacc = phi i64 [ 0, %entry ], [ %eacc.n, %el ]
  %ek = add i64 %ei, 100000
  %er = call i32 @universe_ds_cuckoo_contains(ptr %f, i64 %ek)
  %er64 = zext i32 %er to i64
  %eacc.n = add i64 %eacc, %er64
  %ei.n = add i64 %ei, 1
  %em = icmp ult i64 %ei.n, 1000
  br i1 %em, label %el, label %edone

edone:
  %ez = icmp eq i64 %eacc.n, 0
  call void @ut_check(i1 %ez, ptr @s.cuk.empty)

  ; insert 0..N-1
  br label %il

il:
  %ii = phi i64 [ 0, %edone ], [ %ii.n, %il ]
  %af = phi i64 [ 0, %edone ], [ %af.n, %il ]
  %ar = call i32 @universe_ds_cuckoo_add(ptr %f, i64 %ii)
  %isf = icmp ne i32 %ar, 0
  %isf64 = zext i1 %isf to i64
  %af.n = add i64 %af, %isf64
  %ii.n = add i64 %ii, 1
  %im = icmp ult i64 %ii.n, %N
  br i1 %im, label %il, label %idone

idone:
  call void @ut_check_eq(i64 %af, i64 0, ptr @s.cuk.noadd)

  ; no false negatives
  br label %cl

cl:
  %ci = phi i64 [ 0, %idone ], [ %ci.n, %cl ]
  %miss = phi i64 [ 0, %idone ], [ %miss.n, %cl ]
  %cr = call i32 @universe_ds_cuckoo_contains(ptr %f, i64 %ci)
  %cno = icmp eq i32 %cr, 0
  %cno64 = zext i1 %cno to i64
  %miss.n = add i64 %miss, %cno64
  %ci.n = add i64 %ci, 1
  %cm = icmp ult i64 %ci.n, %N
  br i1 %cm, label %cl, label %cdone

cdone:
  call void @ut_check_eq(i64 %miss, i64 0, ptr @s.cuk.nofn)

  ; FPR: query N absent keys (N..2N-1)
  br label %pl

pl:
  %pi = phi i64 [ 0, %cdone ], [ %pi.n, %pl ]
  %fp = phi i64 [ 0, %cdone ], [ %fp.n, %pl ]
  %pk = add i64 %pi, %N
  %pr = call i32 @universe_ds_cuckoo_contains(ptr %f, i64 %pk)
  %pr64 = zext i32 %pr to i64
  %fp.n = add i64 %fp, %pr64
  %pi.n = add i64 %pi, 1
  %pm = icmp ult i64 %pi.n, %N
  br i1 %pm, label %pl, label %pdone

pdone:
  ; FPR in permille-of-ten (x/10000): fp*10000/N
  %fpx = mul i64 %fp, 10000
  %fppm = udiv i64 %fpx, %N
  %r0 = call i32 (ptr, ...) @printf(ptr @f.cuk.m, i64 %fppm)
  ; ceiling: FPR < 3%  => fp*100 < 3*N
  %fp100 = mul i64 %fp, 100
  %lim3 = mul i64 %N, 3
  %okfpr = icmp ult i64 %fp100, %lim3
  call void @ut_check(i1 %okfpr, ptr @s.cuk.fpr)
  call void @universe_ds_cuckoo_destroy(ptr %f)

  ; single key
  %sf = call ptr @universe_ds_cuckoo_create(i64 32)
  %sa = call i32 @universe_ds_cuckoo_add(ptr %sf, i64 424242)
  %sc = call i32 @universe_ds_cuckoo_contains(ptr %sf, i64 424242)
  %sok = icmp eq i32 %sc, 1
  call void @ut_check(i1 %sok, ptr @s.cuk.single)
  call void @universe_ds_cuckoo_destroy(ptr %sf)

  ; FULL reachable: tiny filter, spam inserts until add returns 6 or cap hit
  %bf = call ptr @universe_ds_cuckoo_create(i64 32)
  br label %fl

fl:
  %fi = phi i64 [ 0, %pdone ], [ %fi.n, %flcont ]
  %fr = call i32 @universe_ds_cuckoo_add(ptr %bf, i64 %fi)
  %isfull = icmp eq i32 %fr, 6
  br i1 %isfull, label %fdone, label %flcont

flcont:
  %fi.n = add i64 %fi, 1
  %flm = icmp ult i64 %fi.n, 4000
  br i1 %flm, label %fl, label %fdone

fdone:
  %sawfull = phi i1 [ true, %fl ], [ false, %flcont ]
  call void @ut_check(i1 %sawfull, ptr @s.cuk.full)
  call void @universe_ds_cuckoo_destroy(ptr %bf)

  ; delete test on a low-load filter: D keys, delete first half
  %D = add i64 0, 8000
  %df = call ptr @universe_ds_cuckoo_create(i64 32000)
  br label %dil

dil:
  %di = phi i64 [ 0, %fdone ], [ %di.n, %dil ]
  %dfa = phi i64 [ 0, %fdone ], [ %dfa.n, %dil ]
  %dar = call i32 @universe_ds_cuckoo_add(ptr %df, i64 %di)
  %darf = icmp ne i32 %dar, 0
  %darf64 = zext i1 %darf to i64
  %dfa.n = add i64 %dfa, %darf64
  %di.n = add i64 %di, 1
  %dim = icmp ult i64 %di.n, %D
  br i1 %dim, label %dil, label %ddel

ddel:
  ; delete first D/2 keys
  %half = udiv i64 %D, 2
  br label %dl

dl:
  %dli = phi i64 [ 0, %ddel ], [ %dli.n, %dl ]
  %dr = call i32 @universe_ds_cuckoo_delete(ptr %df, i64 %dli)
  %dli.n = add i64 %dli, 1
  %dlm = icmp ult i64 %dli.n, %half
  br i1 %dlm, label %dl, label %dchk

dchk:
  ; survivors [half, D) must all be present
  br label %svl

svl:
  %svi = phi i64 [ %half, %dchk ], [ %svi.n, %svl ]
  %svmiss = phi i64 [ 0, %dchk ], [ %svmiss.n, %svl ]
  %svr = call i32 @universe_ds_cuckoo_contains(ptr %df, i64 %svi)
  %svno = icmp eq i32 %svr, 0
  %svno64 = zext i1 %svno to i64
  %svmiss.n = add i64 %svmiss, %svno64
  %svi.n = add i64 %svi, 1
  %svm = icmp ult i64 %svi.n, %D
  br i1 %svm, label %svl, label %sdone

sdone:
  ; survivor loss must be tiny (shared-fingerprint caveat): < 0.5%
  %svlim = udiv i64 %half, 200
  %svok = icmp ule i64 %svmiss.n, %svlim
  call void @ut_check(i1 %svok, ptr @s.cuk.survive)

  ; deleted [0,half) mostly absent
  br label %gl

gl:
  %gi = phi i64 [ 0, %sdone ], [ %gi.n, %gl ]
  %gpres = phi i64 [ 0, %sdone ], [ %gpres.n, %gl ]
  %gr = call i32 @universe_ds_cuckoo_contains(ptr %df, i64 %gi)
  %gr64 = zext i32 %gr to i64
  %gpres.n = add i64 %gpres, %gr64
  %gi.n = add i64 %gi, 1
  %gm = icmp ult i64 %gi.n, %half
  br i1 %gm, label %gl, label %gdone

gdone:
  ; at most ~5% of deleted still register (fp collisions with survivors)
  %glim = mul i64 %half, 5
  %glim2 = udiv i64 %glim, 100
  %gok = icmp ule i64 %gpres.n, %glim2
  call void @ut_check(i1 %gok, ptr @s.cuk.delgone)

  ; count invariant: D adds - half deletes == D-half
  %cnt = call i64 @universe_ds_cuckoo_count(ptr %df)
  %expc = sub i64 %D, %half
  call void @ut_check_eq(i64 %cnt, i64 %expc, ptr @s.cuk.count)
  call void @universe_ds_cuckoo_destroy(ptr %df)
  ret void
}

; ======================================================================
;  XOR tests
; ======================================================================
define void @test_xor() {
entry:
  %N = add i64 0, 20000
  %bytes = shl i64 %N, 3
  %keys = call ptr @malloc(i64 %bytes)
  br label %kl

kl:
  %ki = phi i64 [ 0, %entry ], [ %ki.n, %kl ]
  %kp = getelementptr inbounds i64, ptr %keys, i64 %ki
  store i64 %ki, ptr %kp, align 8
  %ki.n = add i64 %ki, 1
  %km = icmp ult i64 %ki.n, %N
  br i1 %km, label %kl, label %build

build:
  %f = call ptr @universe_ds_xor_build(ptr %keys, i64 %N)
  %nn = icmp ne ptr %f, null
  call void @ut_check(i1 %nn, ptr @s.xor.build)

  ; no false negatives
  br label %cl

cl:
  %ci = phi i64 [ 0, %build ], [ %ci.n, %cl ]
  %miss = phi i64 [ 0, %build ], [ %miss.n, %cl ]
  %cr = call i32 @universe_ds_xor_contains(ptr %f, i64 %ci)
  %cno = icmp eq i32 %cr, 0
  %cno64 = zext i1 %cno to i64
  %miss.n = add i64 %miss, %cno64
  %ci.n = add i64 %ci, 1
  %cm = icmp ult i64 %ci.n, %N
  br i1 %cm, label %cl, label %cdone

cdone:
  call void @ut_check_eq(i64 %miss, i64 0, ptr @s.xor.nofn)

  ; FPR on absent keys
  br label %pl

pl:
  %pi = phi i64 [ 0, %cdone ], [ %pi.n, %pl ]
  %fp = phi i64 [ 0, %cdone ], [ %fp.n, %pl ]
  %pk = add i64 %pi, %N
  %pr = call i32 @universe_ds_xor_contains(ptr %f, i64 %pk)
  %pr64 = zext i32 %pr to i64
  %fp.n = add i64 %fp, %pr64
  %pi.n = add i64 %pi, 1
  %pm = icmp ult i64 %pi.n, %N
  br i1 %pm, label %pl, label %pdone

pdone:
  %fpx = mul i64 %fp, 10000
  %fppm = udiv i64 %fpx, %N
  %r0 = call i32 (ptr, ...) @printf(ptr @f.xor.m, i64 %fppm)
  ; XOR 8-bit fp -> ~1/256 ; ceiling < 2%
  %fp100 = mul i64 %fp, 100
  %lim2 = mul i64 %N, 2
  %okfpr = icmp ult i64 %fp100, %lim2
  call void @ut_check(i1 %okfpr, ptr @s.xor.fpr)

  ; determinism: rebuild, compare contains over a sample of 4096
  %f2 = call ptr @universe_ds_xor_build(ptr %keys, i64 %N)
  br label %dl

dl:
  %di = phi i64 [ 0, %pdone ], [ %di.n, %dl ]
  %ddiff = phi i64 [ 0, %pdone ], [ %ddiff.n, %dl ]
  %da = call i32 @universe_ds_xor_contains(ptr %f, i64 %di)
  %db = call i32 @universe_ds_xor_contains(ptr %f2, i64 %di)
  %dne = icmp ne i32 %da, %db
  %dne64 = zext i1 %dne to i64
  %ddiff.n = add i64 %ddiff, %dne64
  %di.n = add i64 %di, 1
  %dm = icmp ult i64 %di.n, 4096
  br i1 %dm, label %dl, label %ddone

ddone:
  call void @ut_check_eq(i64 %ddiff, i64 0, ptr @s.xor.det)
  call void @universe_ds_xor_destroy(ptr %f)
  call void @universe_ds_xor_destroy(ptr %f2)
  call void @free(ptr %keys)

  ; single key
  %k1 = call ptr @malloc(i64 8)
  store i64 777777, ptr %k1, align 8
  %sf = call ptr @universe_ds_xor_build(ptr %k1, i64 1)
  %scr = call i32 @universe_ds_xor_contains(ptr %sf, i64 777777)
  %sok = icmp eq i32 %scr, 1
  call void @ut_check(i1 %sok, ptr @s.xor.one)
  call void @universe_ds_xor_destroy(ptr %sf)
  call void @free(ptr %k1)

  ; empty build (n=0, keys=null allowed)
  %ef = call ptr @universe_ds_xor_build(ptr null, i64 0)
  %enn = icmp ne ptr %ef, null
  call void @ut_check(i1 %enn, ptr @s.xor.empty0)
  call void @universe_ds_xor_destroy(ptr %ef)
  ret void
}

; ======================================================================
;  BLOCKED BLOOM tests
; ======================================================================
define void @test_bbloom() {
entry:
  %N = add i64 0, 20000
  %f = call ptr @universe_ds_bbloom_create(i64 20000, i64 16)
  %nn = icmp ne ptr %f, null
  call void @ut_check(i1 %nn, ptr @s.bb.create)

  ; empty -> absent (exact: fresh bits all zero)
  br label %el

el:
  %ei = phi i64 [ 0, %entry ], [ %ei.n, %el ]
  %eacc = phi i64 [ 0, %entry ], [ %eacc.n, %el ]
  %ek = add i64 %ei, 100000
  %er = call i32 @universe_ds_bbloom_contains(ptr %f, i64 %ek)
  %er64 = zext i32 %er to i64
  %eacc.n = add i64 %eacc, %er64
  %ei.n = add i64 %ei, 1
  %em = icmp ult i64 %ei.n, 1000
  br i1 %em, label %el, label %edone

edone:
  %ez = icmp eq i64 %eacc.n, 0
  call void @ut_check(i1 %ez, ptr @s.bb.empty)

  ; insert 0..N-1
  br label %il

il:
  %ii = phi i64 [ 0, %edone ], [ %ii.n, %il ]
  %ar = call i32 @universe_ds_bbloom_add(ptr %f, i64 %ii)
  %ii.n = add i64 %ii, 1
  %im = icmp ult i64 %ii.n, %N
  br i1 %im, label %il, label %cl

cl:
  %ci = phi i64 [ 0, %il ], [ %ci.n, %cl ]
  %miss = phi i64 [ 0, %il ], [ %miss.n, %cl ]
  %cr = call i32 @universe_ds_bbloom_contains(ptr %f, i64 %ci)
  %cno = icmp eq i32 %cr, 0
  %cno64 = zext i1 %cno to i64
  %miss.n = add i64 %miss, %cno64
  %ci.n = add i64 %ci, 1
  %cm = icmp ult i64 %ci.n, %N
  br i1 %cm, label %cl, label %cdone

cdone:
  call void @ut_check_eq(i64 %miss, i64 0, ptr @s.bb.nofn)

  ; FPR on absent keys
  br label %pl

pl:
  %pi = phi i64 [ 0, %cdone ], [ %pi.n, %pl ]
  %fp = phi i64 [ 0, %cdone ], [ %fp.n, %pl ]
  %pk = add i64 %pi, %N
  %pr = call i32 @universe_ds_bbloom_contains(ptr %f, i64 %pk)
  %pr64 = zext i32 %pr to i64
  %fp.n = add i64 %fp, %pr64
  %pi.n = add i64 %pi, 1
  %pm = icmp ult i64 %pi.n, %N
  br i1 %pm, label %pl, label %pdone

pdone:
  %fpx = mul i64 %fp, 10000
  %fppm = udiv i64 %fpx, %N
  %r0 = call i32 (ptr, ...) @printf(ptr @f.bb.m, i64 %fppm)
  ; ceiling < 3%
  %fp100 = mul i64 %fp, 100
  %lim3 = mul i64 %N, 3
  %okfpr = icmp ult i64 %fp100, %lim3
  call void @ut_check(i1 %okfpr, ptr @s.bb.fpr)

  ; single key
  %sf = call ptr @universe_ds_bbloom_create(i64 8, i64 16)
  %sa = call i32 @universe_ds_bbloom_add(ptr %sf, i64 999999)
  %sc = call i32 @universe_ds_bbloom_contains(ptr %sf, i64 999999)
  %sok = icmp eq i32 %sc, 1
  call void @ut_check(i1 %sok, ptr @s.bb.single)
  call void @universe_ds_bbloom_destroy(ptr %sf)
  call void @universe_ds_bbloom_destroy(ptr %f)
  ret void
}

; ======================================================================
;  Naive multi-cache-line bloom (bench reference): header{ i64 mask@0 },
;  words at +64. k=8 bits spread across the WHOLE array.
; ======================================================================
define ptr @naive_create(i64 %nwords_pow2) {
entry:
  %bytes = shl i64 %nwords_pow2, 3
  %tot = add i64 %bytes, 64
  %m = call ptr @malloc(i64 %tot)
  %wp = getelementptr inbounds i8, ptr %m, i64 64
  call void @llvm.memset.p0.i64(ptr %wp, i8 0, i64 %bytes, i1 false)
  %nbits = shl i64 %nwords_pow2, 6
  %mask = sub i64 %nbits, 1
  store i64 %mask, ptr %m, align 8
  ret ptr %m
}
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define i32 @naive_add(ptr %f, i64 %key) {
entry:
  %mask = load i64, ptr %f, align 8
  %wp = getelementptr inbounds i8, ptr %f, i64 64
  %h1 = call i64 @nmix(i64 %key)
  %h2 = call i64 @nmix(i64 %h1)
  %bodd = or i64 %h2, 1
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ib = mul i64 %i, %bodd
  %raw = add i64 %h1, %ib
  %pos = and i64 %raw, %mask
  %word = lshr i64 %pos, 6
  %bit = and i64 %pos, 63
  %wep = getelementptr inbounds i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %m1 = shl nuw i64 1, %bit
  %new = or i64 %cur, %m1
  store i64 %new, ptr %wep, align 8
  %i.n = add i64 %i, 1
  %mo = icmp ult i64 %i.n, 8
  br i1 %mo, label %loop, label %done
done:
  ret i32 0
}

define i32 @naive_contains(ptr %f, i64 %key) {
entry:
  %mask = load i64, ptr %f, align 8
  %wp = getelementptr inbounds i8, ptr %f, i64 64
  %h1 = call i64 @nmix(i64 %key)
  %h2 = call i64 @nmix(i64 %h1)
  %bodd = or i64 %h2, 1
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 1, %entry ], [ %acc.n, %loop ]
  %ib = mul i64 %i, %bodd
  %raw = add i64 %h1, %ib
  %pos = and i64 %raw, %mask
  %word = lshr i64 %pos, 6
  %bit = and i64 %pos, 63
  %wep = getelementptr inbounds i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %sh = lshr i64 %cur, %bit
  %is = and i64 %sh, 1
  %acc.n = and i64 %acc, %is
  %i.n = add i64 %i, 1
  %mo = icmp ult i64 %i.n, 8
  br i1 %mo, label %loop, label %done
done:
  %r = trunc i64 %acc.n to i32
  ret i32 %r
}

; local copy of the splitmix64 mixer for the naive reference
define i64 @nmix(i64 %x) {
entry:
  %z = add i64 %x, 11400714819323198485
  %s1 = lshr i64 %z, 30
  %x1 = xor i64 %z, %s1
  %m1 = mul i64 %x1, 13787848793156543929
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %m1, %s2
  %m2 = mul i64 %x2, 10723151780598845931
  %s3 = lshr i64 %m2, 31
  %r = xor i64 %m2, %s3
  ret i64 %r
}

; ======================================================================
;  Bench harness: 17 reps (rep 0 warm-up, discarded) of `passes` full
;  sweeps of contains; each measured rep's elapsed seconds is stored into
;  the caller's [16 x double] buffer for ut_report_dist. ops_per_rep = n*passes.
; ======================================================================
define void @bench_contains(ptr %fn, ptr %f, i64 %n, i64 %passes, ptr %samples) {
entry:
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %pl
pl:
  %p = phi i64 [ 0, %rep ], [ %p.n, %pouter ]
  %pacc = phi i64 [ 0, %rep ], [ %pacc2, %pouter ]
  br label %inner
inner:
  %j = phi i64 [ 0, %pl ], [ %j.n, %inner ]
  %iacc = phi i64 [ %pacc, %pl ], [ %iacc.n, %inner ]
  %ir = call i32 %fn(ptr %f, i64 %j)
  %ir64 = zext i32 %ir to i64
  %iacc.n = add i64 %iacc, %ir64
  %j.n = add i64 %j, 1
  %jm = icmp ult i64 %j.n, %n
  br i1 %jm, label %inner, label %pouter
pouter:
  %pacc2 = phi i64 [ %iacc.n, %inner ]
  %p.n = add i64 %p, 1
  %pm = icmp ult i64 %p.n, %passes
  br i1 %pm, label %pl, label %rep.done
rep.done:
  %t1 = call double @ut_now_sec()
  ; publish acc so the sweeps are not dead-code-eliminated
  store volatile i64 %pacc2, ptr @bench.sink, align 8
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %rep.i, 0
  br i1 %warm, label %rep.next, label %rep.store
rep.store:
  %sidx = sub i64 %rep.i, 1
  %sp = getelementptr inbounds double, ptr %samples, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next
rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %fin
fin:
  ret void
}
@bench.sink = internal global i64 0, align 8

define void @run_bench(i64 %N) {
entry:
  ; large working set (filters exceed fast cache) so the blocked-bloom
  ; single-cache-line advantage over the scattered naive bloom is visible.
  %passes = add i64 0, 10
  %r0 = call i32 (ptr, ...) @printf(ptr @f.bhdr)
  %ops = mul i64 %N, %passes

  ; build populated filters
  %chint = shl i64 %N, 1
  %cf = call ptr @universe_ds_cuckoo_create(i64 %chint)
  %kbytes = shl i64 %N, 3
  %kbuf = call ptr @malloc(i64 %kbytes)
  br label %bl
bl:
  %bi = phi i64 [ 0, %entry ], [ %bi.n, %bl ]
  %z0 = call i32 @universe_ds_cuckoo_add(ptr %cf, i64 %bi)
  %kp = getelementptr inbounds i64, ptr %kbuf, i64 %bi
  store i64 %bi, ptr %kp, align 8
  %bi.n = add i64 %bi, 1
  %bm = icmp ult i64 %bi.n, %N
  br i1 %bm, label %bl, label %bdone
bdone:
  %xf = call ptr @universe_ds_xor_build(ptr %kbuf, i64 %N)
  %bbf = call ptr @universe_ds_bbloom_create(i64 %N, i64 16)
  ; match naive total bits to the blocked filter (nblocks*8 words)
  %nblk = load i64, ptr %bbf, align 8
  %nwords = shl i64 %nblk, 3
  %nvf = call ptr @naive_create(i64 %nwords)
  br label %bl2
bl2:
  %bi2 = phi i64 [ 0, %bdone ], [ %bi2.n, %bl2 ]
  %z1 = call i32 @universe_ds_bbloom_add(ptr %bbf, i64 %bi2)
  %z2 = call i32 @naive_add(ptr %nvf, i64 %bi2)
  %bi2.n = add i64 %bi2, 1
  %bm2 = icmp ult i64 %bi2.n, %N
  br i1 %bm2, label %bl2, label %timeit

timeit:
  call void @bench_contains(ptr @universe_ds_cuckoo_contains, ptr %cf, i64 %N, i64 %passes, ptr @filters.samp)
  call void @ut_report_dist(ptr @filters.samp, i64 16, i64 %ops, ptr @lbl.f_cuckoo)

  call void @bench_contains(ptr @universe_ds_xor_contains, ptr %xf, i64 %N, i64 %passes, ptr @filters.samp)
  call void @ut_report_dist(ptr @filters.samp, i64 16, i64 %ops, ptr @lbl.f_xor)

  call void @bench_contains(ptr @universe_ds_bbloom_contains, ptr %bbf, i64 %N, i64 %passes, ptr @filters.samp)
  call void @ut_report_dist(ptr @filters.samp, i64 16, i64 %ops, ptr @lbl.f_bbloom)

  call void @bench_contains(ptr @naive_contains, ptr %nvf, i64 %N, i64 %passes, ptr @filters.samp)
  call void @ut_report_dist(ptr @filters.samp, i64 16, i64 %ops, ptr @lbl.f_naive)

  call void @universe_ds_cuckoo_destroy(ptr %cf)
  call void @universe_ds_xor_destroy(ptr %xf)
  call void @universe_ds_bbloom_destroy(ptr %bbf)
  call void @free(ptr %nvf)
  call void @free(ptr %kbuf)
  ret void
}

define i64 @ops_per_sec(i64 %ops, double %t) {
entry:
  %opsf = uitofp i64 %ops to double
  %r = fdiv double %opsf, %t
  %ri = fptoui double %r to i64
  ret i64 %ri
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_cuckoo()
  call void @test_xor()
  call void @test_bbloom()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %dobench, label %fin
dobench:
  call void @run_bench(i64 2000000)
  br label %fin
fin:
  %r = call i32 @ut_summary()
  ret i32 %r
}
