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

; Tests for universe_ml_hnsw_*. CORRECTNESS gate is RECALL vs exact brute
; force: build an index over a fixed-seed random corpus (n=2000, d=32) and for
; a batch of random queries assert recall@10 exceeds 0.90 at ef_search=64.
; Also: k>n, single element, duplicate vectors, empty-index search out_n=0,
; create with bad args returns null. --bench reports query throughput.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i64 @ut_rand(ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)

declare ptr @universe_ml_hnsw_create(i64, i32, i64, i64)
declare i32 @universe_ml_hnsw_insert(ptr, ptr, i64)
declare i32 @universe_ml_hnsw_search(ptr, ptr, i64, i64, ptr, ptr, ptr)
declare i64 @universe_ml_hnsw_len(ptr)
declare void @universe_ml_hnsw_destroy(ptr)

@m.create   = private unnamed_addr constant [16 x i8] c"create non-null\00"
@m.baddim   = private unnamed_addr constant [17 x i8] c"bad dims => null\00"
@m.badmet   = private unnamed_addr constant [19 x i8] c"bad metric => null\00"
@m.insrc    = private unnamed_addr constant [11 x i8] c"insert ok \00"
@m.len      = private unnamed_addr constant [10 x i8] c"len == n \00"
@m.srcrc    = private unnamed_addr constant [11 x i8] c"search ok \00"
@m.outn     = private unnamed_addr constant [13 x i8] c"out_n == 10 \00"
@m.recall   = private unnamed_addr constant [16 x i8] c"recall@10 > .90\00"
@m.kgtn     = private unnamed_addr constant [14 x i8] c"k>n: out_n==n\00"
@m.single   = private unnamed_addr constant [16 x i8] c"single elem hit\00"
@m.dup      = private unnamed_addr constant [15 x i8] c"dup vecs out_n\00"
@m.emptyn   = private unnamed_addr constant [16 x i8] c"empty out_n==0 \00"
@m.emptyrc  = private unnamed_addr constant [14 x i8] c"empty rc == 0\00"

@f.recall   = private unnamed_addr constant [30 x i8] c"recall@10 = %lld/1000 (n=%d)\0A\00"
@f.ins      = private unnamed_addr constant [28 x i8] c"BENCH insert: %.1f vec/sec\0A\00"
@l.query    = private unnamed_addr constant [6 x i8] c"query\00"

; ---- scalar reference squared-L2 distance over d floats ----
define internal float @refdist2(ptr %a, ptr %b, i64 %d) {
entry:
  br label %head
head:
  %i = phi i64 [ 0, %entry ], [ %in, %body ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %body ]
  %go = icmp ult i64 %i, %d
  br i1 %go, label %body, label %ret
body:
  %ap = getelementptr inbounds float, ptr %a, i64 %i
  %av = load float, ptr %ap, align 4
  %bp = getelementptr inbounds float, ptr %b, i64 %i
  %bv = load float, ptr %bp, align 4
  %df = fsub float %av, %bv
  %sq = fmul float %df, %df
  %accn = fadd float %acc, %sq
  %in = add i64 %i, 1
  br label %head
ret:
  ret float %acc
}

; ---- random float in [-1,1) from the ut LCG ----
define internal float @rndf(ptr %state) {
entry:
  %r = call i64 @ut_rand(ptr %state)
  %m24 = and i64 %r, 16777215
  %f = uitofp i64 %m24 to float
  %u = fmul float %f, 0x3E70000000000000 ; 1/16777216
  %s = fmul float %u, 2.0
  %c = fsub float %s, 1.0
  ret float %c
}

define internal void @genvec(ptr %dst, i64 %d, ptr %state) {
entry:
  br label %head
head:
  %i = phi i64 [ 0, %entry ], [ %in, %body ]
  %go = icmp ult i64 %i, %d
  br i1 %go, label %body, label %ret
body:
  %v = call float @rndf(ptr %state)
  %p = getelementptr inbounds float, ptr %dst, i64 %i
  store float %v, ptr %p, align 4
  %in = add i64 %i, 1
  br label %head
ret:
  ret void
}

; ---- exact top-10 (replace-worst) filling %ei[0..10) with nearest indices ----
define internal void @brute10(ptr %corpus, i64 %n, i64 %d, ptr %q, ptr %ed, ptr %ei) {
entry:
  br label %ihead
ihead:
  %i = phi i64 [ 0, %entry ], [ %in, %ibody ]
  %iok = icmp ult i64 %i, 10
  br i1 %iok, label %ibody, label %scanhead
ibody:
  %edp = getelementptr inbounds float, ptr %ed, i64 %i
  store float 0x7FF0000000000000, ptr %edp, align 4
  %eip = getelementptr inbounds i64, ptr %ei, i64 %i
  store i64 -1, ptr %eip, align 8
  %in = add i64 %i, 1
  br label %ihead
scanhead:
  %j = phi i64 [ 0, %ihead ], [ %jn, %scancont ]
  %jok = icmp ult i64 %j, %n
  br i1 %jok, label %scanbody, label %ret
scanbody:
  %roff = mul i64 %j, %d
  %row = getelementptr inbounds float, ptr %corpus, i64 %roff
  %dist = call float @refdist2(ptr %q, ptr %row, i64 %d)
  %d0 = load float, ptr %ed, align 4
  br label %mhead
mhead:
  %p = phi i64 [ 1, %scanbody ], [ %pn, %mbody ]
  %maxpos = phi i64 [ 0, %scanbody ], [ %maxposn, %mbody ]
  %maxval = phi float [ %d0, %scanbody ], [ %maxvaln, %mbody ]
  %pgo = icmp ult i64 %p, 10
  br i1 %pgo, label %mbody, label %mdone
mbody:
  %pp = getelementptr inbounds float, ptr %ed, i64 %p
  %pv = load float, ptr %pp, align 4
  %gt = fcmp ogt float %pv, %maxval
  %maxposn = select i1 %gt, i64 %p, i64 %maxpos
  %maxvaln = select i1 %gt, float %pv, float %maxval
  %pn = add i64 %p, 1
  br label %mhead
mdone:
  %better = fcmp olt float %dist, %maxval
  br i1 %better, label %replace, label %scancont
replace:
  %rdp = getelementptr inbounds float, ptr %ed, i64 %maxpos
  store float %dist, ptr %rdp, align 4
  %rip = getelementptr inbounds i64, ptr %ei, i64 %maxpos
  store i64 %j, ptr %rip, align 8
  br label %scancont
scancont:
  %jn = add i64 %j, 1
  br label %scanhead
ret:
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  %outl = alloca [10 x i64], align 8
  %outd = alloca [10 x float], align 4
  %outn = alloca i64, align 8
  %exd = alloca [10 x float], align 4
  %exi = alloca [10 x i64], align 8
  %qvec = alloca [32 x float], align 16

  ; ---- corpus ----
  %corpus = call ptr @malloc(i64 256000) ; 2000*32*4
  br label %fill.head
fill.head:
  %fi = phi i64 [ 0, %entry ], [ %fin, %fill.body ]
  %fgo = icmp ult i64 %fi, 2000
  br i1 %fgo, label %fill.body, label %build
fill.body:
  %foff = mul i64 %fi, 32
  %frow = getelementptr inbounds float, ptr %corpus, i64 %foff
  call void @genvec(ptr %frow, i64 32, ptr %seed)
  %fin = add i64 %fi, 1
  br label %fill.head

build:
  %h = call ptr @universe_ml_hnsw_create(i64 32, i32 1, i64 16, i64 100)
  %hnn = icmp ne ptr %h, null
  call void @ut_check(i1 %hnn, ptr @m.create)
  ; insert all
  br label %ins.head
ins.head:
  %ii = phi i64 [ 0, %build ], [ %iin, %ins.body ]
  %insviol = phi i64 [ 0, %build ], [ %insviol.n, %ins.body ]
  %igo = icmp ult i64 %ii, 2000
  br i1 %igo, label %ins.body, label %ins.done
ins.body:
  %ioff = mul i64 %ii, 32
  %irow = getelementptr inbounds float, ptr %corpus, i64 %ioff
  %irc = call i32 @universe_ml_hnsw_insert(ptr %h, ptr %irow, i64 %ii)
  %ibad = icmp ne i32 %irc, 0
  %ibad64 = zext i1 %ibad to i64
  %insviol.n = add i64 %insviol, %ibad64
  %iin = add i64 %ii, 1
  br label %ins.head
ins.done:
  call void @ut_check_eq(i64 %insviol, i64 0, ptr @m.insrc)
  %ln = call i64 @universe_ml_hnsw_len(ptr %h)
  call void @ut_check_eq(i64 %ln, i64 2000, ptr @m.len)

  ; ---- recall over 50 queries ----
  br label %q.head
q.head:
  %qi = phi i64 [ 0, %ins.done ], [ %qin, %q.cont ]
  %found = phi i64 [ 0, %ins.done ], [ %found.n, %q.cont ]
  %srcviol = phi i64 [ 0, %ins.done ], [ %srcviol.n, %q.cont ]
  %outnviol = phi i64 [ 0, %ins.done ], [ %outnviol.n, %q.cont ]
  %qgo = icmp ult i64 %qi, 50
  br i1 %qgo, label %q.body, label %q.report
q.body:
  call void @genvec(ptr %qvec, i64 32, ptr %seed)
  %src = call i32 @universe_ml_hnsw_search(ptr %h, ptr %qvec, i64 10, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %sbad = icmp ne i32 %src, 0
  %sbad64 = zext i1 %sbad to i64
  %srcviol.n = add i64 %srcviol, %sbad64
  %onv = load i64, ptr %outn, align 8
  %onbad = icmp ne i64 %onv, 10
  %onbad64 = zext i1 %onbad to i64
  %outnviol.n = add i64 %outnviol, %onbad64
  ; exact
  call void @brute10(ptr %corpus, i64 2000, i64 32, ptr %qvec,
             ptr %exd, ptr %exi)
  ; count overlap of approx labels vs exact indices
  br label %cmp.head
cmp.head:
  %ci = phi i64 [ 0, %q.body ], [ %cin, %cmp.cont ]
  %cf = phi i64 [ %found, %q.body ], [ %cf.n, %cmp.cont ]
  %cgo = icmp ult i64 %ci, 10
  br i1 %cgo, label %cmp.body, label %q.cont
cmp.body:
  %lp = getelementptr inbounds [10 x i64], ptr %outl, i64 0, i64 %ci
  %lbl = load i64, ptr %lp, align 8
  br label %in.head
in.head:
  %bi = phi i64 [ 0, %cmp.body ], [ %bin, %in.body ]
  %hit = phi i64 [ 0, %cmp.body ], [ %hit.n, %in.body ]
  %bgo = icmp ult i64 %bi, 10
  br i1 %bgo, label %in.body, label %cmp.acc
in.body:
  %xp = getelementptr inbounds [10 x i64], ptr %exi, i64 0, i64 %bi
  %xv = load i64, ptr %xp, align 8
  %eqm = icmp eq i64 %xv, %lbl
  %eqm64 = zext i1 %eqm to i64
  %hit.n = add i64 %hit, %eqm64
  %bin = add i64 %bi, 1
  br label %in.head
cmp.acc:
  %cf.n = add i64 %cf, %hit
  br label %cmp.cont
cmp.cont:
  %cin = add i64 %ci, 1
  br label %cmp.head
q.cont:
  %found.n = phi i64 [ %cf, %cmp.head ]
  %qin = add i64 %qi, 1
  br label %q.head
q.report:
  ; recall permil = found*1000/500
  %fx = mul i64 %found, 1000
  %permil = udiv i64 %fx, 500
  %pr = call i32 (ptr, ...) @printf(ptr @f.recall, i64 %permil, i32 2000)
  call void @ut_check_eq(i64 %srcviol, i64 0, ptr @m.srcrc)
  call void @ut_check_eq(i64 %outnviol, i64 0, ptr @m.outn)
  %recok = icmp ugt i64 %permil, 900
  call void @ut_check(i1 %recok, ptr @m.recall)

  ; ---- k>n on the same index? use small index for clarity ----
  %hsmall = call ptr @universe_ml_hnsw_create(i64 32, i32 1, i64 16, i64 100)
  %s0 = getelementptr inbounds float, ptr %corpus, i64 0
  %s1 = getelementptr inbounds float, ptr %corpus, i64 32
  %s2 = getelementptr inbounds float, ptr %corpus, i64 64
  %ig1 = call i32 @universe_ml_hnsw_insert(ptr %hsmall, ptr %s0, i64 100)
  %ig2 = call i32 @universe_ml_hnsw_insert(ptr %hsmall, ptr %s1, i64 101)
  %ig3 = call i32 @universe_ml_hnsw_insert(ptr %hsmall, ptr %s2, i64 102)
  %ig4 = call i32 @universe_ml_hnsw_search(ptr %hsmall, ptr %s0, i64 10, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %kn = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %kn, i64 3, ptr @m.kgtn)
  call void @universe_ml_hnsw_destroy(ptr %hsmall)

  ; ---- single element: nearest of itself is itself ----
  %h1 = call ptr @universe_ml_hnsw_create(i64 32, i32 1, i64 16, i64 100)
  %ig5 = call i32 @universe_ml_hnsw_insert(ptr %h1, ptr %s0, i64 777)
  %ig6 = call i32 @universe_ml_hnsw_search(ptr %h1, ptr %s0, i64 1, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %o1n = load i64, ptr %outn, align 8
  %o1l = load i64, ptr %outl, align 8
  %o1ok1 = icmp eq i64 %o1n, 1
  %o1ok2 = icmp eq i64 %o1l, 777
  %o1ok = and i1 %o1ok1, %o1ok2
  call void @ut_check(i1 %o1ok, ptr @m.single)
  call void @universe_ml_hnsw_destroy(ptr %h1)

  ; ---- duplicate vectors: 5 identical, search returns 5 ----
  %hd = call ptr @universe_ml_hnsw_create(i64 32, i32 1, i64 16, i64 100)
  br label %dup.head
dup.head:
  %di = phi i64 [ 0, %q.report ], [ %din, %dup.body ]
  %dgo = icmp ult i64 %di, 5
  br i1 %dgo, label %dup.body, label %dup.done
dup.body:
  %ig7 = call i32 @universe_ml_hnsw_insert(ptr %hd, ptr %s0, i64 %di)
  %din = add i64 %di, 1
  br label %dup.head
dup.done:
  call i32 @universe_ml_hnsw_search(ptr %hd, ptr %s0, i64 3, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %dn = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %dn, i64 3, ptr @m.dup)
  call void @universe_ml_hnsw_destroy(ptr %hd)

  ; ---- empty index search returns out_n=0 ----
  %he = call ptr @universe_ml_hnsw_create(i64 32, i32 1, i64 16, i64 100)
  %erc = call i32 @universe_ml_hnsw_search(ptr %he, ptr %s0, i64 5, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %en = load i64, ptr %outn, align 8
  %erc64 = sext i32 %erc to i64
  call void @ut_check_eq(i64 %en, i64 0, ptr @m.emptyn)
  call void @ut_check_eq(i64 %erc64, i64 0, ptr @m.emptyrc)
  call void @universe_ml_hnsw_destroy(ptr %he)

  ; ---- create with bad args ----
  %hbad1 = call ptr @universe_ml_hnsw_create(i64 0, i32 1, i64 16, i64 100)
  %bad1 = icmp eq ptr %hbad1, null
  call void @ut_check(i1 %bad1, ptr @m.baddim)
  %hbad2 = call ptr @universe_ml_hnsw_create(i64 32, i32 5, i64 16, i64 100)
  %bad2 = icmp eq ptr %hbad2, null
  call void @ut_check(i1 %bad2, ptr @m.badmet)

  ; ---- optional bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %cleanup
bench:
  call void @bench_queries(ptr %h, ptr %corpus, ptr %seed, ptr %outl, ptr %outd, ptr %outn)
  br label %cleanup
cleanup:
  call void @universe_ml_hnsw_destroy(ptr %h)
  call void @free(ptr %corpus)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; ---- bench: time 16 reps of 500 queries each; report ns/query distribution ----
define internal void @bench_queries(ptr %h, ptr %corpus, ptr %seed,
                                    ptr %outl, ptr %outd, ptr %outn) {
entry:
  %samples = alloca [16 x double], align 8
  %qv = alloca [32 x float], align 16
  ; warm-up rep (discarded)
  br label %warm.head
warm.head:
  %wi = phi i64 [ 0, %entry ], [ %win, %warm.body ]
  %wgo = icmp ult i64 %wi, 500
  br i1 %wgo, label %warm.body, label %reps
warm.body:
  call void @genvec(ptr %qv, i64 32, ptr %seed)
  %ig8 = call i32 @universe_ml_hnsw_search(ptr %h, ptr %qv, i64 10, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %win = add i64 %wi, 1
  br label %warm.head
reps:
  br label %rep.head
rep.head:
  %ri = phi i64 [ 0, %reps ], [ %rin, %rep.store ]
  %rgo = icmp ult i64 %ri, 16
  br i1 %rgo, label %rep.body, label %done
rep.body:
  %t0 = call double @ut_now_sec()
  br label %qb.head
qb.head:
  %qj = phi i64 [ 0, %rep.body ], [ %qjn, %qb.body ]
  %qgo = icmp ult i64 %qj, 500
  br i1 %qgo, label %qb.body, label %rep.time
qb.body:
  call void @genvec(ptr %qv, i64 32, ptr %seed)
  %ig9 = call i32 @universe_ml_hnsw_search(ptr %h, ptr %qv, i64 10, i64 64,
             ptr %outl, ptr %outd, ptr %outn)
  %qjn = add i64 %qj, 1
  br label %qb.head
rep.time:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  br label %rep.store
rep.store:
  %sp = getelementptr inbounds [16 x double], ptr %samples, i64 0, i64 %ri
  store double %dt, ptr %sp, align 8
  %rin = add i64 %ri, 1
  br label %rep.head
done:
  %sbase = getelementptr inbounds [16 x double], ptr %samples, i64 0, i64 0
  call void @ut_report_dist(ptr %sbase, i64 16, i64 500, ptr @l.query)
  ret void
}
