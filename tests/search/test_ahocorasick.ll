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

; Tests for universe_search_ac: hand-verified pattern sets (overlaps, suffix
; relations, start/end matches), error/edge cases, and a fixed-seed random
; text/pattern cross-check against a naive O(n*m*k) brute-force reference via
; a (pattern_id, end_offset) occupancy grid. --bench: AC scan MB/s.

declare ptr @universe_search_ac_build(ptr, ptr, i64)
declare i64 @universe_search_ac_search(ptr, ptr, i64, ptr, ptr, i64)
declare void @universe_search_ac_destroy(ptr)

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

@p.he   = private unnamed_addr constant [2 x i8] c"he"
@p.she  = private unnamed_addr constant [3 x i8] c"she"
@p.his  = private unnamed_addr constant [3 x i8] c"his"
@p.hers = private unnamed_addr constant [4 x i8] c"hers"
@t.ushers = private unnamed_addr constant [6 x i8] c"ushers"

@p.a  = private unnamed_addr constant [1 x i8] c"a"
@p.ab = private unnamed_addr constant [2 x i8] c"ab"
@p.b  = private unnamed_addr constant [1 x i8] c"b"
@t.ab = private unnamed_addr constant [2 x i8] c"ab"

@t.empty = private unnamed_addr constant [1 x i8] c"x"

@m.classic = private unnamed_addr constant [22 x i8] c"classic he/she/hers  \00"
@m.classicN = private unnamed_addr constant [20 x i8] c"classic count == 3 \00"
@m.startend = private unnamed_addr constant [22 x i8] c"a/ab/b start-end-sfx \00"
@m.startendN = private unnamed_addr constant [17 x i8] c"a/ab/b count 3  \00"
@m.nomatch = private unnamed_addr constant [17 x i8] c"no-match count 0\00"
@m.single  = private unnamed_addr constant [16 x i8] c"single pattern \00"
@m.emptytext = private unnamed_addr constant [16 x i8] c"empty text -> 0\00"
@m.nullac  = private unnamed_addr constant [16 x i8] c"null ac -> -1  \00"
@m.zeropat = private unnamed_addr constant [18 x i8] c"empty pat -> null\00"
@m.nbuild  = private unnamed_addr constant [18 x i8] c"n=0 builds valid \00"
@m.rand    = private unnamed_addr constant [19 x i8] c"random vs brute   \00"
@ac.samp = internal global [16 x double] zeroinitializer, align 8
@ac.sink = internal global i64 0, align 8
@lbl.ac = private unnamed_addr constant [22 x i8] c"aho-corasick 4MB text\00"

; ---- brute force: add +1 into grid[pid*K + end] for every occurrence -------
; returns match count.
define internal i64 @brute(ptr %patterns, ptr %lens, i64 %n, ptr %text, i64 %len, ptr %grid, i64 %K) {
entry:
  br label %oi.head

oi.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %oi.cont ]
  %cnt = phi i64 [ 0, %entry ], [ %cnt.o, %oi.cont ]
  %more = icmp ult i64 %i, %n
  br i1 %more, label %oi.body, label %done

oi.body:
  %lp = getelementptr inbounds i64, ptr %lens, i64 %i
  %L = load i64, ptr %lp, align 8
  %pp = getelementptr inbounds ptr, ptr %patterns, i64 %i
  %p = load ptr, ptr %pp, align 8
  %fits = icmp ule i64 %L, %len
  br i1 %fits, label %s.head, label %oi.cont

s.head:
  %s = phi i64 [ 0, %oi.body ], [ %s.n, %s.cont ]
  %cnt.s = phi i64 [ %cnt, %oi.body ], [ %cnt.s2, %s.cont ]
  %last = sub i64 %len, %L
  %s.ok = icmp ule i64 %s, %last
  br i1 %s.ok, label %s.body, label %oi.cont.j

s.body:
  %tp = getelementptr inbounds i8, ptr %text, i64 %s
  %cmp = call i32 @memcmp(ptr %tp, ptr %p, i64 %L)
  %eq = icmp eq i32 %cmp, 0
  br i1 %eq, label %hit, label %s.cont

hit:
  %end = add i64 %s, %L
  %end.i = sub i64 %end, 1
  %ik = mul i64 %i, %K
  %cell = add i64 %ik, %end.i
  %gp = getelementptr inbounds i8, ptr %grid, i64 %cell
  %gv = load i8, ptr %gp, align 1
  %gv.n = add i8 %gv, 1
  store i8 %gv.n, ptr %gp, align 1
  %cnt.hit = add i64 %cnt.s, 1
  br label %s.cont

s.cont:
  %cnt.s2 = phi i64 [ %cnt.hit, %hit ], [ %cnt.s, %s.body ]
  %s.n = add i64 %s, 1
  br label %s.head

oi.cont.j:
  br label %oi.cont

oi.cont:
  %cnt.o = phi i64 [ %cnt, %oi.body ], [ %cnt.s, %oi.cont.j ]
  %i.n = add i64 %i, 1
  br label %oi.head

done:
  ret i64 %cnt
}

; ---- run AC + brute over a case, return #(grid mismatches) + count-mismatch
define internal i64 @run_check(ptr %patterns, ptr %lens, i64 %n, ptr %text, i64 %len) {
entry:
  %K = add i64 %len, 1
  %gsz = mul i64 %n, %K
  %gsz1 = call i64 @llvm.umax.i64(i64 %gsz, i64 1)
  %grid = call ptr @calloc(i64 %gsz1, i64 1)
  %ac = call ptr @universe_search_ac_build(ptr %patterns, ptr %lens, i64 %n)
  %ac.null = icmp eq ptr %ac, null
  br i1 %ac.null, label %badbuild, label %run

badbuild:
  call void @free(ptr %grid)
  ret i64 1000000

run:
  %cap = add i64 %gsz1, 1
  %ids.b = mul i64 %cap, 4
  %ends.b = mul i64 %cap, 8
  %out.ids = call ptr @malloc(i64 %ids.b)
  %out.ends = call ptr @malloc(i64 %ends.b)
  %ac.count = call i64 @universe_search_ac_search(ptr %ac, ptr %text, i64 %len, ptr %out.ids, ptr %out.ends, i64 %cap)
  br label %sub.head

sub.head:                                          ; grid[cell] -= 1 for AC hits
  %k = phi i64 [ 0, %run ], [ %k.n, %sub.body ]
  %k.more = icmp ult i64 %k, %ac.count
  br i1 %k.more, label %sub.body, label %after.ac

sub.body:
  %ip = getelementptr inbounds i32, ptr %out.ids, i64 %k
  %pid32 = load i32, ptr %ip, align 4
  %pid = sext i32 %pid32 to i64
  %ep = getelementptr inbounds i64, ptr %out.ends, i64 %k
  %end = load i64, ptr %ep, align 8
  %ik = mul i64 %pid, %K
  %cell = add i64 %ik, %end
  %gp = getelementptr inbounds i8, ptr %grid, i64 %cell
  %gv = load i8, ptr %gp, align 1
  %gv.n = sub i8 %gv, 1
  store i8 %gv.n, ptr %gp, align 1
  %k.n = add i64 %k, 1
  br label %sub.head

after.ac:
  %bf.count = call i64 @brute(ptr %patterns, ptr %lens, i64 %n, ptr %text, i64 %len, ptr %grid, i64 %K)
  br label %scan.head

scan.head:                                         ; count nonzero cells
  %c = phi i64 [ 0, %after.ac ], [ %c.n, %scan.cont ]
  %mm = phi i64 [ 0, %after.ac ], [ %mm.n, %scan.cont ]
  %c.more = icmp ult i64 %c, %gsz
  br i1 %c.more, label %scan.body, label %fin

scan.body:
  %cgp = getelementptr inbounds i8, ptr %grid, i64 %c
  %cgv = load i8, ptr %cgp, align 1
  %nz = icmp ne i8 %cgv, 0
  %inc = zext i1 %nz to i64
  %mm.b = add i64 %mm, %inc
  br label %scan.cont

scan.cont:
  %mm.n = phi i64 [ %mm.b, %scan.body ]
  %c.n = add i64 %c, 1
  br label %scan.head

fin:
  %cnt.bad = icmp ne i64 %ac.count, %bf.count
  %cnt.inc = zext i1 %cnt.bad to i64
  %mm.total = add i64 %mm, %cnt.inc
  call void @universe_search_ac_destroy(ptr %ac)
  call void @free(ptr %out.ids)
  call void @free(ptr %out.ends)
  call void @free(ptr %grid)
  ret i64 %mm.total
}

declare i64 @llvm.umax.i64(i64, i64)

; ---- classic he/she/his/hers over "ushers" --------------------------------
define internal void @test_classic() {
entry:
  %pats = alloca [4 x ptr], align 8
  %lens = alloca [4 x i64], align 8
  %p0 = getelementptr inbounds [4 x ptr], ptr %pats, i64 0, i64 0
  store ptr @p.he, ptr %p0, align 8
  %p1 = getelementptr inbounds [4 x ptr], ptr %pats, i64 0, i64 1
  store ptr @p.she, ptr %p1, align 8
  %p2 = getelementptr inbounds [4 x ptr], ptr %pats, i64 0, i64 2
  store ptr @p.his, ptr %p2, align 8
  %p3 = getelementptr inbounds [4 x ptr], ptr %pats, i64 0, i64 3
  store ptr @p.hers, ptr %p3, align 8
  %l0 = getelementptr inbounds [4 x i64], ptr %lens, i64 0, i64 0
  store i64 2, ptr %l0, align 8
  %l1 = getelementptr inbounds [4 x i64], ptr %lens, i64 0, i64 1
  store i64 3, ptr %l1, align 8
  %l2 = getelementptr inbounds [4 x i64], ptr %lens, i64 0, i64 2
  store i64 3, ptr %l2, align 8
  %l3 = getelementptr inbounds [4 x i64], ptr %lens, i64 0, i64 3
  store i64 4, ptr %l3, align 8
  %mm = call i64 @run_check(ptr %pats, ptr %lens, i64 4, ptr @t.ushers, i64 6)
  %ok = icmp eq i64 %mm, 0
  call void @ut_check(i1 %ok, ptr @m.classic)
  ; explicit count assert
  %ac = call ptr @universe_search_ac_build(ptr %pats, ptr %lens, i64 4)
  %cnt = call i64 @universe_search_ac_search(ptr %ac, ptr @t.ushers, i64 6, ptr null, ptr null, i64 0)
  call void @ut_check_eq(i64 %cnt, i64 3, ptr @m.classicN)
  call void @universe_search_ac_destroy(ptr %ac)
  ret void
}

; ---- a/ab/b over "ab": start, end, suffix relation ------------------------
define internal void @test_startend() {
entry:
  %pats = alloca [3 x ptr], align 8
  %lens = alloca [3 x i64], align 8
  %p0 = getelementptr inbounds [3 x ptr], ptr %pats, i64 0, i64 0
  store ptr @p.a, ptr %p0, align 8
  %p1 = getelementptr inbounds [3 x ptr], ptr %pats, i64 0, i64 1
  store ptr @p.ab, ptr %p1, align 8
  %p2 = getelementptr inbounds [3 x ptr], ptr %pats, i64 0, i64 2
  store ptr @p.b, ptr %p2, align 8
  %l0 = getelementptr inbounds [3 x i64], ptr %lens, i64 0, i64 0
  store i64 1, ptr %l0, align 8
  %l1 = getelementptr inbounds [3 x i64], ptr %lens, i64 0, i64 1
  store i64 2, ptr %l1, align 8
  %l2 = getelementptr inbounds [3 x i64], ptr %lens, i64 0, i64 2
  store i64 1, ptr %l2, align 8
  %mm = call i64 @run_check(ptr %pats, ptr %lens, i64 3, ptr @t.ab, i64 2)
  %ok = icmp eq i64 %mm, 0
  call void @ut_check(i1 %ok, ptr @m.startend)
  %ac = call ptr @universe_search_ac_build(ptr %pats, ptr %lens, i64 3)
  %cnt = call i64 @universe_search_ac_search(ptr %ac, ptr @t.ab, i64 2, ptr null, ptr null, i64 0)
  call void @ut_check_eq(i64 %cnt, i64 3, ptr @m.startendN)
  call void @universe_search_ac_destroy(ptr %ac)
  ret void
}

; ---- no-match + single pattern + errors -----------------------------------
define internal void @test_edges() {
entry:
  ; no match: pattern "his" over "ushers"
  %pats1 = alloca [1 x ptr], align 8
  %lens1 = alloca [1 x i64], align 8
  store ptr @p.his, ptr %pats1, align 8
  store i64 3, ptr %lens1, align 8
  %ac1 = call ptr @universe_search_ac_build(ptr %pats1, ptr %lens1, i64 1)
  %cnt1 = call i64 @universe_search_ac_search(ptr %ac1, ptr @t.ushers, i64 6, ptr null, ptr null, i64 0)
  %nm = icmp eq i64 %cnt1, 0
  call void @ut_check(i1 %nm, ptr @m.nomatch)
  ; single pattern "she" over "ushers": one match
  store ptr @p.she, ptr %pats1, align 8
  store i64 3, ptr %lens1, align 8
  %ac2 = call ptr @universe_search_ac_build(ptr %pats1, ptr %lens1, i64 1)
  %cnt2 = call i64 @universe_search_ac_search(ptr %ac2, ptr @t.ushers, i64 6, ptr null, ptr null, i64 0)
  %sg = icmp eq i64 %cnt2, 1
  call void @ut_check(i1 %sg, ptr @m.single)
  ; empty text
  %cnt3 = call i64 @universe_search_ac_search(ptr %ac2, ptr @t.empty, i64 0, ptr null, ptr null, i64 0)
  %et = icmp eq i64 %cnt3, 0
  call void @ut_check(i1 %et, ptr @m.emptytext)
  ; null ac -> -1
  %cnt4 = call i64 @universe_search_ac_search(ptr null, ptr @t.ushers, i64 6, ptr null, ptr null, i64 0)
  %na = icmp eq i64 %cnt4, -1
  call void @ut_check(i1 %na, ptr @m.nullac)
  ; empty pattern (len 0) -> build null
  store ptr @p.she, ptr %pats1, align 8
  store i64 0, ptr %lens1, align 8
  %acbad = call ptr @universe_search_ac_build(ptr %pats1, ptr %lens1, i64 1)
  %zp = icmp eq ptr %acbad, null
  call void @ut_check(i1 %zp, ptr @m.zeropat)
  ; n=0 -> valid empty automaton, search 0
  %acn = call ptr @universe_search_ac_build(ptr null, ptr null, i64 0)
  %acn.ok = icmp ne ptr %acn, null
  %cnt5 = call i64 @universe_search_ac_search(ptr %acn, ptr @t.ushers, i64 6, ptr null, ptr null, i64 0)
  %cnt5.z = icmp eq i64 %cnt5, 0
  %nb.ok = and i1 %acn.ok, %cnt5.z
  call void @ut_check(i1 %nb.ok, ptr @m.nbuild)
  call void @universe_search_ac_destroy(ptr %ac1)
  call void @universe_search_ac_destroy(ptr %ac2)
  call void @universe_search_ac_destroy(ptr %acn)
  ret void
}

; ---- random text + random short patterns vs brute -------------------------
define internal void @test_random() {
entry:
  %seed = alloca i64, align 8
  store i64 987654321, ptr %seed, align 8
  %NP = add i64 0, 12                               ; patterns
  %TL = add i64 0, 2000                             ; text length
  %text = call ptr @malloc(i64 %TL)
  %pats = alloca [12 x ptr], align 8
  %lens = alloca [12 x i64], align 8
  ; fill text over alphabet 'a'..'d'
  br label %tfill

tfill:
  %ti = phi i64 [ 0, %entry ], [ %ti.n, %tfill ]
  %tr = call i64 @ut_rand(ptr %seed)
  %tm = urem i64 %tr, 4
  %tb0 = add i64 %tm, 97
  %tb = trunc i64 %tb0 to i8
  %tcp = getelementptr inbounds i8, ptr %text, i64 %ti
  store i8 %tb, ptr %tcp, align 1
  %ti.n = add i64 %ti, 1
  %tmore = icmp ult i64 %ti.n, %TL
  br i1 %tmore, label %tfill, label %pgen

pgen:                                               ; each pattern len 2..5
  %pi = phi i64 [ 0, %tfill ], [ %pi.n, %pgen.cont ]
  %pr = call i64 @ut_rand(ptr %seed)
  %pmod = urem i64 %pr, 4
  %plen = add i64 %pmod, 2
  %pbuf = call ptr @malloc(i64 %plen)
  br label %pchar

pchar:
  %ci = phi i64 [ 0, %pgen ], [ %ci.n, %pchar ]
  %cr = call i64 @ut_rand(ptr %seed)
  %cm = urem i64 %cr, 4
  %cb0 = add i64 %cm, 97
  %cb = trunc i64 %cb0 to i8
  %ccp = getelementptr inbounds i8, ptr %pbuf, i64 %ci
  store i8 %cb, ptr %ccp, align 1
  %ci.n = add i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, %plen
  br i1 %cmore, label %pchar, label %pgen.cont

pgen.cont:
  %pslot = getelementptr inbounds [12 x ptr], ptr %pats, i64 0, i64 %pi
  store ptr %pbuf, ptr %pslot, align 8
  %lslot = getelementptr inbounds [12 x i64], ptr %lens, i64 0, i64 %pi
  store i64 %plen, ptr %lslot, align 8
  %pi.n = add i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, %NP
  br i1 %pmore, label %pgen, label %check

check:
  %pats.p = getelementptr inbounds [12 x ptr], ptr %pats, i64 0, i64 0
  %lens.p = getelementptr inbounds [12 x i64], ptr %lens, i64 0, i64 0
  %mm = call i64 @run_check(ptr %pats.p, ptr %lens.p, i64 %NP, ptr %text, i64 %TL)
  call void @ut_check_eq(i64 %mm, i64 0, ptr @m.rand)
  ; free patterns
  br label %pfree

pfree:
  %fi = phi i64 [ 0, %check ], [ %fi.n, %pfree ]
  %fslot = getelementptr inbounds [12 x ptr], ptr %pats, i64 0, i64 %fi
  %fp = load ptr, ptr %fslot, align 8
  call void @free(ptr %fp)
  %fi.n = add i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, %NP
  br i1 %fmore, label %pfree, label %tfree

tfree:
  call void @free(ptr %text)
  ret void
}

; ---- bench: scan throughput over a large text -----------------------------
define internal void @bench() {
entry:
  %seed = alloca i64, align 8
  store i64 424242, ptr %seed, align 8
  %TL = add i64 0, 4000000
  %NP = add i64 0, 100
  %text = call ptr @malloc(i64 %TL)
  %pats = call ptr @malloc(i64 800)                 ; 100 * 8
  %lens = call ptr @malloc(i64 800)
  br label %tfill

tfill:
  %ti = phi i64 [ 0, %entry ], [ %ti.n, %tfill ]
  %tr = call i64 @ut_rand(ptr %seed)
  %tm = urem i64 %tr, 6
  %tb0 = add i64 %tm, 97
  %tb = trunc i64 %tb0 to i8
  %tcp = getelementptr inbounds i8, ptr %text, i64 %ti
  store i8 %tb, ptr %tcp, align 1
  %ti.n = add i64 %ti, 1
  %tmore = icmp ult i64 %ti.n, %TL
  br i1 %tmore, label %tfill, label %pgen

pgen:
  %pi = phi i64 [ 0, %tfill ], [ %pi.n, %pgen.cont ]
  %pr = call i64 @ut_rand(ptr %seed)
  %pmod = urem i64 %pr, 4
  %plen = add i64 %pmod, 3
  %pbuf = call ptr @malloc(i64 %plen)
  br label %pchar

pchar:
  %ci = phi i64 [ 0, %pgen ], [ %ci.n, %pchar ]
  %cr = call i64 @ut_rand(ptr %seed)
  %cm = urem i64 %cr, 6
  %cb0 = add i64 %cm, 97
  %cb = trunc i64 %cb0 to i8
  %ccp = getelementptr inbounds i8, ptr %pbuf, i64 %ci
  store i8 %cb, ptr %ccp, align 1
  %ci.n = add i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, %plen
  br i1 %cmore, label %pchar, label %pgen.cont

pgen.cont:
  %pslot = getelementptr inbounds ptr, ptr %pats, i64 %pi
  store ptr %pbuf, ptr %pslot, align 8
  %lslot = getelementptr inbounds i64, ptr %lens, i64 %pi
  store i64 %plen, ptr %lslot, align 8
  %pi.n = add i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, %NP
  br i1 %pmore, label %pgen, label %build

build:
  %ac = call ptr @universe_search_ac_build(ptr %pats, ptr %lens, i64 %NP)
  ; 17 reps of a single 4 MB search; discard rep 0 (warm-up), report over the
  ; remaining 16. ops/rep = TL = 4000000 text bytes (ns per byte).
  br label %ac.rep
ac.rep:
  %arep = phi i64 [ 0, %build ], [ %arep.n, %ac.next ]
  %t0 = call double @ut_now_sec()
  %cnt = call i64 @universe_search_ac_search(ptr %ac, ptr %text, i64 %TL, ptr null, ptr null, i64 0)
  %t1 = call double @ut_now_sec()
  store volatile i64 %cnt, ptr @ac.sink, align 8
  %ael = fsub double %t1, %t0
  %akeep = icmp ugt i64 %arep, 0
  br i1 %akeep, label %ac.store, label %ac.next
ac.store:
  %aidx = sub i64 %arep, 1
  %asp = getelementptr inbounds [16 x double], ptr @ac.samp, i64 0, i64 %aidx
  store double %ael, ptr %asp, align 8
  br label %ac.next
ac.next:
  %arep.n = add nuw i64 %arep, 1
  %amore = icmp ult i64 %arep.n, 17
  br i1 %amore, label %ac.rep, label %ac.report
ac.report:
  call void @ut_report_dist(ptr @ac.samp, i64 16, i64 4000000, ptr @lbl.ac)
  call void @universe_search_ac_destroy(ptr %ac)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_classic()
  call void @test_startend()
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
