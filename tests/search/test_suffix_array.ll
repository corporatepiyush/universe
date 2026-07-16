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

; Tests for universe_search_sa: known "banana" SA/LCP, adjacent-suffix order
; + permutation + Kasai-LCP invariants over a random string, find/count vs a
; naive substring scan, and edges (len 0/1, all-same, periodic). --bench:
; SA build time + find throughput vs naive scan.

declare ptr @universe_search_sa_build(ptr, i64)
declare i64 @universe_search_sa_find(ptr, ptr, ptr, i64)
declare i64 @universe_search_sa_count(ptr, ptr, ptr, i64)
declare i64 @universe_search_sa_at(ptr, i64)
declare i64 @universe_search_sa_lcp_at(ptr, i64)
declare i64 @universe_search_sa_len(ptr)
declare void @universe_search_sa_destroy(ptr)

declare i32 @printf(ptr, ...)
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare void @free(ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@t.banana = private unnamed_addr constant [6 x i8] c"banana"
@exp.sa   = private unnamed_addr constant [6 x i32] [i32 5, i32 3, i32 1, i32 0, i32 4, i32 2]
@exp.lcp  = private unnamed_addr constant [6 x i32] [i32 0, i32 1, i32 3, i32 0, i32 0, i32 2]
@p.ana    = private unnamed_addr constant [3 x i8] c"ana"
@p.ban    = private unnamed_addr constant [3 x i8] c"ban"
@p.zzz    = private unnamed_addr constant [3 x i8] c"zzz"
@t.same   = private unnamed_addr constant [8 x i8] c"aaaaaaaa"
@p.aaa    = private unnamed_addr constant [3 x i8] c"aaa"
@t.period = private unnamed_addr constant [8 x i8] c"abababab"
@p.abab   = private unnamed_addr constant [4 x i8] c"abab"
@t.one    = private unnamed_addr constant [1 x i8] c"q"
@p.q      = private unnamed_addr constant [1 x i8] c"q"

@m.saExact = private unnamed_addr constant [16 x i8] c"banana SA exact\00"
@m.lcpExact = private unnamed_addr constant [17 x i8] c"banana LCP exact\00"
@m.order   = private unnamed_addr constant [20 x i8] c"SA sorted order    \00"
@m.perm    = private unnamed_addr constant [20 x i8] c"SA is permutation  \00"
@m.lcpinv  = private unnamed_addr constant [20 x i8] c"LCP matches Kasai  \00"
@m.rcount  = private unnamed_addr constant [20 x i8] c"count vs naive     \00"
@m.rfind   = private unnamed_addr constant [20 x i8] c"find valid vs naive\00"
@m.anaC    = private unnamed_addr constant [16 x i8] c"ana count == 2 \00"
@m.banF    = private unnamed_addr constant [16 x i8] c"ban find == 0  \00"
@m.absF    = private unnamed_addr constant [16 x i8] c"zzz find == -1 \00"
@m.absC    = private unnamed_addr constant [16 x i8] c"zzz count == 0 \00"
@m.same    = private unnamed_addr constant [18 x i8] c"aaa in aaaaaaaa 6\00"
@m.period  = private unnamed_addr constant [16 x i8] c"abab periodic 3\00"
@m.len0    = private unnamed_addr constant [16 x i8] c"empty len 0 ok \00"
@m.len1    = private unnamed_addr constant [16 x i8] c"len1 find == 0 \00"
@sa.buildsamp = internal global [16 x double] zeroinitializer, align 8
@sa.findsamp  = internal global [16 x double] zeroinitializer, align 8
@sa.naivesamp = internal global [16 x double] zeroinitializer, align 8
@sa.sink = internal global i64 0, align 8
@lbl.sabuild = private unnamed_addr constant [23 x i8] c"suffix array build200K\00"
@lbl.safind  = private unnamed_addr constant [21 x i8] c"suffix array find L8\00"
@lbl.sanaive = private unnamed_addr constant [22 x i8] c"suffix array naive L8\00"

; full suffix compare: -1 a<b, 0 equal, 1 a>b
define internal i32 @suffix_cmp(ptr %text, i64 %N, i64 %a, i64 %b) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %eqc ]
  %ai = add i64 %a, %i
  %bi = add i64 %b, %i
  %ae = icmp uge i64 %ai, %N
  %be = icmp uge i64 %bi, %N
  br i1 %ae, label %a.end, label %a.ok

a.end:
  br i1 %be, label %ret0, label %retlt

a.ok:
  br i1 %be, label %retgt, label %cmp

cmp:
  %ap = getelementptr inbounds i8, ptr %text, i64 %ai
  %ca = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %text, i64 %bi
  %cb = load i8, ptr %bp, align 1
  %lt = icmp ult i8 %ca, %cb
  br i1 %lt, label %retlt, label %gtc

gtc:
  %gt = icmp ugt i8 %ca, %cb
  br i1 %gt, label %retgt, label %eqc

eqc:
  %i.n = add i64 %i, 1
  br label %loop

ret0:
  ret i32 0
retlt:
  ret i32 -1
retgt:
  ret i32 1
}

define internal i64 @suffix_lcp(ptr %text, i64 %N, i64 %a, i64 %b) {
entry:
  br label %loop

loop:
  %h = phi i64 [ 0, %entry ], [ %h.n, %cont ]
  %ai = add i64 %a, %h
  %bi = add i64 %b, %h
  %ae = icmp uge i64 %ai, %N
  %be = icmp uge i64 %bi, %N
  %end = or i1 %ae, %be
  br i1 %end, label %done, label %cmp

cmp:
  %ap = getelementptr inbounds i8, ptr %text, i64 %ai
  %ca = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %text, i64 %bi
  %cb = load i8, ptr %bp, align 1
  %eq = icmp eq i8 %ca, %cb
  br i1 %eq, label %cont, label %done

cont:
  %h.n = add i64 %h, 1
  br label %loop

done:
  ret i64 %h
}

; naive occurrence count of pattern in text
define internal i64 @naive_count(ptr %text, i64 %N, ptr %pat, i64 %plen) {
entry:
  %fits = icmp ule i64 %plen, %N
  br i1 %fits, label %loop, label %zero

zero:
  ret i64 0

loop:
  %s = phi i64 [ 0, %entry ], [ %s.n, %cont ]
  %cnt = phi i64 [ 0, %entry ], [ %cnt.n, %cont ]
  %last = sub i64 %N, %plen
  %ok = icmp ule i64 %s, %last
  br i1 %ok, label %body, label %done

body:
  %tp = getelementptr inbounds i8, ptr %text, i64 %s
  %c = call i32 @memcmp(ptr %tp, ptr %pat, i64 %plen)
  %eq = icmp eq i32 %c, 0
  %inc = zext i1 %eq to i64
  br label %cont

cont:
  %cnt.n = add i64 %cnt, %inc
  %s.n = add i64 %s, 1
  br label %loop

done:
  ret i64 %cnt
}

; ---- known "banana": SA and LCP exact -------------------------------------
define internal void @test_banana() {
entry:
  %sa = call ptr @universe_search_sa_build(ptr @t.banana, i64 6)
  br label %chk

chk:
  %i = phi i64 [ 0, %entry ], [ %i.n, %chk ]
  %sabad = phi i64 [ 0, %entry ], [ %sabad.n, %chk ]
  %lcpbad = phi i64 [ 0, %entry ], [ %lcpbad.n, %chk ]
  %got.sa = call i64 @universe_search_sa_at(ptr %sa, i64 %i)
  %esp = getelementptr inbounds i32, ptr @exp.sa, i64 %i
  %esv = load i32, ptr %esp, align 4
  %esz = zext i32 %esv to i64
  %sane = icmp ne i64 %got.sa, %esz
  %sinc = zext i1 %sane to i64
  %sabad.n = add i64 %sabad, %sinc
  %got.lcp = call i64 @universe_search_sa_lcp_at(ptr %sa, i64 %i)
  %elp = getelementptr inbounds i32, ptr @exp.lcp, i64 %i
  %elv = load i32, ptr %elp, align 4
  %elz = zext i32 %elv to i64
  %lne = icmp ne i64 %got.lcp, %elz
  %linc = zext i1 %lne to i64
  %lcpbad.n = add i64 %lcpbad, %linc
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, 6
  br i1 %more, label %chk, label %report

report:
  call void @ut_check_eq(i64 %sabad, i64 0, ptr @m.saExact)
  call void @ut_check_eq(i64 %lcpbad, i64 0, ptr @m.lcpExact)
  ; find/count small asserts
  %anaC = call i64 @universe_search_sa_count(ptr %sa, ptr @t.banana, ptr @p.ana, i64 3)
  call void @ut_check_eq(i64 %anaC, i64 2, ptr @m.anaC)
  %banF = call i64 @universe_search_sa_find(ptr %sa, ptr @t.banana, ptr @p.ban, i64 3)
  call void @ut_check_eq(i64 %banF, i64 0, ptr @m.banF)
  %absF = call i64 @universe_search_sa_find(ptr %sa, ptr @t.banana, ptr @p.zzz, i64 3)
  %absF.ok = icmp eq i64 %absF, -1
  call void @ut_check(i1 %absF.ok, ptr @m.absF)
  %absC = call i64 @universe_search_sa_count(ptr %sa, ptr @t.banana, ptr @p.zzz, i64 3)
  call void @ut_check_eq(i64 %absC, i64 0, ptr @m.absC)
  call void @universe_search_sa_destroy(ptr %sa)
  ret void
}

; ---- edges: all-same, periodic, len0, len1 --------------------------------
define internal void @test_edges() {
entry:
  ; "aaaaaaaa" (8), "aaa" occurs at 0..5 -> 6 times
  %sa1 = call ptr @universe_search_sa_build(ptr @t.same, i64 8)
  %c1 = call i64 @universe_search_sa_count(ptr %sa1, ptr @t.same, ptr @p.aaa, i64 3)
  call void @ut_check_eq(i64 %c1, i64 6, ptr @m.same)
  call void @universe_search_sa_destroy(ptr %sa1)
  ; "abababab" (8), "abab" occurs at 0,2,4 -> 3 times
  %sa2 = call ptr @universe_search_sa_build(ptr @t.period, i64 8)
  %c2 = call i64 @universe_search_sa_count(ptr %sa2, ptr @t.period, ptr @p.abab, i64 4)
  call void @ut_check_eq(i64 %c2, i64 3, ptr @m.period)
  call void @universe_search_sa_destroy(ptr %sa2)
  ; len 0
  %sa0 = call ptr @universe_search_sa_build(ptr @t.banana, i64 0)
  %n0 = call i64 @universe_search_sa_len(ptr %sa0)
  %f0 = call i64 @universe_search_sa_find(ptr %sa0, ptr @t.banana, ptr @p.q, i64 1)
  %z0 = icmp eq i64 %n0, 0
  %m0 = icmp eq i64 %f0, -1
  %ok0 = and i1 %z0, %m0
  call void @ut_check(i1 %ok0, ptr @m.len0)
  call void @universe_search_sa_destroy(ptr %sa0)
  ; len 1
  %sa1b = call ptr @universe_search_sa_build(ptr @t.one, i64 1)
  %f1 = call i64 @universe_search_sa_find(ptr %sa1b, ptr @t.one, ptr @p.q, i64 1)
  call void @ut_check_eq(i64 %f1, i64 0, ptr @m.len1)
  call void @universe_search_sa_destroy(ptr %sa1b)
  ret void
}

; ---- random string: order + permutation + LCP + find/count ----------------
define internal void @test_random() {
entry:
  %seed = alloca i64, align 8
  store i64 13572468, ptr %seed, align 8
  %N = add i64 0, 1500
  %text = call ptr @malloc(i64 %N)
  br label %tfill

tfill:
  %ti = phi i64 [ 0, %entry ], [ %ti.n, %tfill ]
  %r = call i64 @ut_rand(ptr %seed)
  %m = urem i64 %r, 3
  %b0 = add i64 %m, 97
  %b = trunc i64 %b0 to i8
  %tp = getelementptr inbounds i8, ptr %text, i64 %ti
  store i8 %b, ptr %tp, align 1
  %ti.n = add i64 %ti, 1
  %more = icmp ult i64 %ti.n, %N
  br i1 %more, label %tfill, label %build

build:
  %sa = call ptr @universe_search_sa_build(ptr %text, i64 %N)
  %seen = call ptr @calloc(i64 %N, i64 1)
  br label %vloop

vloop:                                              ; verify order + perm + LCP
  %vi = phi i64 [ 0, %build ], [ %vi.n, %vcont ]
  %ordbad = phi i64 [ 0, %build ], [ %ordbad.n, %vcont ]
  %permbad = phi i64 [ 0, %build ], [ %permbad.n, %vcont ]
  %lcpbad = phi i64 [ 0, %build ], [ %lcpbad.n, %vcont ]
  %cur = call i64 @universe_search_sa_at(ptr %sa, i64 %vi)
  ; mark seen[cur]
  %seenp = getelementptr inbounds i8, ptr %seen, i64 %cur
  %sv = load i8, ptr %seenp, align 1
  %dup = icmp ne i8 %sv, 0
  %pinc = zext i1 %dup to i64
  %permbad.n = add i64 %permbad, %pinc
  store i8 1, ptr %seenp, align 1
  %first = icmp eq i64 %vi, 0
  br i1 %first, label %vcont, label %vcmp

vcmp:
  %vim1 = sub i64 %vi, 1
  %prev = call i64 @universe_search_sa_at(ptr %sa, i64 %vim1)
  %c = call i32 @suffix_cmp(ptr %text, i64 %N, i64 %prev, i64 %cur)
  %notlt = icmp sge i32 %c, 0                        ; prev should be < cur
  %oinc = zext i1 %notlt to i64
  %o.acc = add i64 %ordbad, %oinc
  %expl = call i64 @suffix_lcp(ptr %text, i64 %N, i64 %prev, i64 %cur)
  %gotl = call i64 @universe_search_sa_lcp_at(ptr %sa, i64 %vi)
  %lne = icmp ne i64 %expl, %gotl
  %linc = zext i1 %lne to i64
  %l.acc = add i64 %lcpbad, %linc
  br label %vcont

vcont:
  %ordbad.n = phi i64 [ %ordbad, %vloop ], [ %o.acc, %vcmp ]
  %lcpbad.n = phi i64 [ %lcpbad, %vloop ], [ %l.acc, %vcmp ]
  %vi.n = add i64 %vi, 1
  %vmore = icmp ult i64 %vi.n, %N
  br i1 %vmore, label %vloop, label %vdone

vdone:
  call void @ut_check_eq(i64 %ordbad, i64 0, ptr @m.order)
  call void @ut_check_eq(i64 %permbad, i64 0, ptr @m.perm)
  call void @ut_check_eq(i64 %lcpbad, i64 0, ptr @m.lcpinv)
  ; find/count cross-check with random patterns
  br label %qloop

qloop:
  %qi = phi i64 [ 0, %vdone ], [ %qi.n, %qcont ]
  %cbad = phi i64 [ 0, %vdone ], [ %cbad.n, %qcont ]
  %fbad = phi i64 [ 0, %vdone ], [ %fbad.n, %qcont ]
  ; random pattern length 1..6
  %rl = call i64 @ut_rand(ptr %seed)
  %plm = urem i64 %rl, 6
  %plen = add i64 %plm, 1
  %pat = call ptr @malloc(i64 %plen)
  ; 50/50 present (copy from random text pos) or maybe absent (inject 'z')
  %rp = call i64 @ut_rand(ptr %seed)
  %absent = urem i64 %rp, 2
  %rs = call i64 @ut_rand(ptr %seed)
  %maxstart = sub i64 %N, %plen
  %spos = urem i64 %rs, %maxstart
  br label %pcopy

pcopy:
  %pj = phi i64 [ 0, %qloop ], [ %pj.n, %pcopy ]
  %srcp = getelementptr inbounds i8, ptr %text, i64 %spos
  %srcj = getelementptr inbounds i8, ptr %srcp, i64 %pj
  %sc = load i8, ptr %srcj, align 1
  %dstj = getelementptr inbounds i8, ptr %pat, i64 %pj
  store i8 %sc, ptr %dstj, align 1
  %pj.n = add i64 %pj, 1
  %pjmore = icmp ult i64 %pj.n, %plen
  br i1 %pjmore, label %pcopy, label %pmaybe

pmaybe:
  %mkabs = icmp ne i64 %absent, 0
  br i1 %mkabs, label %inject, label %qrun

inject:                                             ; force absence: set last='z'
  %lastj = sub i64 %plen, 1
  %injp = getelementptr inbounds i8, ptr %pat, i64 %lastj
  store i8 122, ptr %injp, align 1
  br label %qrun

qrun:
  %nc = call i64 @naive_count(ptr %text, i64 %N, ptr %pat, i64 %plen)
  %sc.cnt = call i64 @universe_search_sa_count(ptr %sa, ptr %text, ptr %pat, i64 %plen)
  %cne = icmp ne i64 %nc, %sc.cnt
  %cinc = zext i1 %cne to i64
  %fnd = call i64 @universe_search_sa_find(ptr %sa, ptr %text, ptr %pat, i64 %plen)
  %hasocc = icmp ne i64 %nc, 0
  br i1 %hasocc, label %chkpos, label %chkneg

chkpos:                                             ; find must be a real occurrence
  %fp = getelementptr inbounds i8, ptr %text, i64 %fnd
  %mc = call i32 @memcmp(ptr %fp, ptr %pat, i64 %plen)
  %mbad = icmp ne i32 %mc, 0
  %frange = icmp ugt i64 %fnd, %maxstart
  %fb1 = or i1 %mbad, %frange
  br label %qcont

chkneg:                                             ; find must be -1
  %fneg = icmp ne i64 %fnd, -1
  br label %qcont

qcont:
  %finc.b = phi i1 [ %fb1, %chkpos ], [ %fneg, %chkneg ]
  %finc = zext i1 %finc.b to i64
  %cbad.n = add i64 %cbad, %cinc
  %fbad.n = add i64 %fbad, %finc
  call void @free(ptr %pat)
  %qi.n = add i64 %qi, 1
  %qmore = icmp ult i64 %qi.n, 4000
  br i1 %qmore, label %qloop, label %qdone

qdone:
  call void @ut_check_eq(i64 %cbad, i64 0, ptr @m.rcount)
  call void @ut_check_eq(i64 %fbad, i64 0, ptr @m.rfind)
  call void @universe_search_sa_destroy(ptr %sa)
  call void @free(ptr %seen)
  call void @free(ptr %text)
  ret void
}

; ---- bench -----------------------------------------------------------------
define internal void @bench() {
entry:
  %seed = alloca i64, align 8
  store i64 999, ptr %seed, align 8
  %N = add i64 0, 200000
  %text = call ptr @malloc(i64 %N)
  br label %tfill

tfill:
  %ti = phi i64 [ 0, %entry ], [ %ti.n, %tfill ]
  %r = call i64 @ut_rand(ptr %seed)
  %m = urem i64 %r, 4
  %b0 = add i64 %m, 97
  %b = trunc i64 %b0 to i8
  %tp = getelementptr inbounds i8, ptr %text, i64 %ti
  store i8 %b, ptr %tp, align 1
  %ti.n = add i64 %ti, 1
  %more = icmp ult i64 %ti.n, %N
  br i1 %more, label %tfill, label %warm

warm:
  ; pattern from text
  %pp = getelementptr inbounds i8, ptr %text, i64 12345
  ; ---- BUILD bench: 17 reps of one SA build+destroy over N bytes; discard rep 0
  ;      (warm-up), report over the remaining 16. ops/rep = N = 200000 (ns/byte). ----
  br label %bd.rep

bd.rep:
  %brep = phi i64 [ 0, %warm ], [ %brep.n, %bd.next ]
  %bt0 = call double @ut_now_sec()
  %bsa = call ptr @universe_search_sa_build(ptr %text, i64 %N)
  %bt1 = call double @ut_now_sec()
  call void @universe_search_sa_destroy(ptr %bsa)
  %bel = fsub double %bt1, %bt0
  %bkeep = icmp ugt i64 %brep, 0
  br i1 %bkeep, label %bd.store, label %bd.next
bd.store:
  %bidx = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @sa.buildsamp, i64 0, i64 %bidx
  store double %bel, ptr %bsp, align 8
  br label %bd.next
bd.next:
  %brep.n = add nuw i64 %brep, 1
  %bmore = icmp ult i64 %brep.n, 17
  br i1 %bmore, label %bd.rep, label %bd.report
bd.report:
  call void @ut_report_dist(ptr @sa.buildsamp, i64 16, i64 200000, ptr @lbl.sabuild)
  ; one surviving SA for the find bench
  %sa = call ptr @universe_search_sa_build(ptr %text, i64 %N)
  br label %fd.rep

; ---- FIND bench: 17 reps of a 100000-find batch over the built SA; discard
;      rep 0 (warm-up), report over the remaining 16. ops/rep = 100000 (ns/find). ----
fd.rep:
  %frep = phi i64 [ 0, %bd.report ], [ %frep.n, %fd.next ]
  %ft0 = call double @ut_now_sec()
  br label %fq
fq:
  %fi = phi i64 [ 0, %fd.rep ], [ %fi.n, %fq ]
  %acc = phi i64 [ 0, %fd.rep ], [ %acc.n, %fq ]
  %fr = call i64 @universe_search_sa_find(ptr %sa, ptr %text, ptr %pp, i64 8)
  %acc.n = add i64 %acc, %fr
  %fi.n = add i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 100000
  br i1 %fmore, label %fq, label %fd.rep.done
fd.rep.done:
  %ft1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr @sa.sink, align 8
  %fel = fsub double %ft1, %ft0
  %fkeep = icmp ugt i64 %frep, 0
  br i1 %fkeep, label %fd.store, label %fd.next
fd.store:
  %fidx = sub i64 %frep, 1
  %fsp = getelementptr inbounds [16 x double], ptr @sa.findsamp, i64 0, i64 %fidx
  store double %fel, ptr %fsp, align 8
  br label %fd.next
fd.next:
  %frep.n = add nuw i64 %frep, 1
  %fmore2 = icmp ult i64 %frep.n, 17
  br i1 %fmore2, label %fd.rep, label %fd.report
fd.report:
  call void @ut_report_dist(ptr @sa.findsamp, i64 16, i64 100000, ptr @lbl.safind)
  br label %nv.rep

; ---- NAIVE baseline: same 100000-query batch via linear scan; distribution
;      over 16 reps for comparison. ops/rep = 100000 (ns per naive query). ----
nv.rep:
  %nrep = phi i64 [ 0, %fd.report ], [ %nrep.n, %nv.next ]
  %nt0 = call double @ut_now_sec()
  br label %nq
nq:
  %ni = phi i64 [ 0, %nv.rep ], [ %ni.n, %nq ]
  %nacc = phi i64 [ 0, %nv.rep ], [ %nacc.n, %nq ]
  %nr = call i64 @naive_count(ptr %text, i64 %N, ptr %pp, i64 8)
  %nacc.n = add i64 %nacc, %nr
  %ni.n = add i64 %ni, 1
  %nmore = icmp ult i64 %ni.n, 100000
  br i1 %nmore, label %nq, label %nv.rep.done
nv.rep.done:
  %nt1 = call double @ut_now_sec()
  store volatile i64 %nacc.n, ptr @sa.sink, align 8
  %nel = fsub double %nt1, %nt0
  %nkeep = icmp ugt i64 %nrep, 0
  br i1 %nkeep, label %nv.store, label %nv.next
nv.store:
  %nidx = sub i64 %nrep, 1
  %nsp = getelementptr inbounds [16 x double], ptr @sa.naivesamp, i64 0, i64 %nidx
  store double %nel, ptr %nsp, align 8
  br label %nv.next
nv.next:
  %nrep.n = add nuw i64 %nrep, 1
  %nmore2 = icmp ult i64 %nrep.n, 17
  br i1 %nmore2, label %nv.rep, label %nv.report
nv.report:
  call void @ut_report_dist(ptr @sa.naivesamp, i64 16, i64 100000, ptr @lbl.sanaive)
  call void @universe_search_sa_destroy(ptr %sa)
  call void @free(ptr %text)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_banana()
  call void @test_edges()
  call void @test_random()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary

do.bench:
  call void @bench()
  br label %summary

summary:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
