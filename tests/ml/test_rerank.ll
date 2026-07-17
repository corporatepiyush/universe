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

; Tests for universe_ml_rrf / universe_ml_mmr.
;   RRF: hand-computed 2-3 list case (exact scores + order), single-list case,
;        tie-break-by-lowest-id, empty lists, out_cap < distinct, default k,
;        null args.
;   MMR: lambda=1 => pure top-k by relevance; lambda=0 => hand-checked diversity
;        order; fixed-seed random trials cross-checked BIT-CONSISTENTLY against a
;        scalar brute-force oracle (integer-valued vectors keep dot/norm sums
;        exact so the vector `fast` reduction and the scalar oracle agree
;        bit-for-bit -> the greedy argmax picks the same indices); edges k>n,
;        k=0, n=0, null args.
;   --bench times both kernels (warm-up + 16 reps, ns/op distribution).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare float @llvm.fabs.f32(float)
declare float @llvm.sqrt.f32(float)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

declare i32 @universe_ml_rrf(ptr, ptr, i64, float, ptr, ptr, i64, ptr)
declare i32 @universe_ml_mmr(ptr, i64, i64, ptr, float, i64, ptr, ptr)

; ------------------------------------------------------------- RRF fixtures
@l0 = internal global [3 x i64] [i64 10, i64 20, i64 30]
@l1 = internal global [2 x i64] [i64 20, i64 40]
@l2 = internal global [1 x i64] [i64 10]
@ids3 = internal global [3 x ptr] [ptr @l0, ptr @l1, ptr @l2]
@len3 = internal global [3 x i64] [i64 3, i64 2, i64 1]

@sl0 = internal global [3 x i64] [i64 7, i64 8, i64 9]
@ids1 = internal global [1 x ptr] [ptr @sl0]
@len1 = internal global [1 x i64] [i64 3]

@t0 = internal global [1 x i64] [i64 200]
@t1 = internal global [1 x i64] [i64 100]
@idst = internal global [2 x ptr] [ptr @t0, ptr @t1]
@lent = internal global [2 x i64] [i64 1, i64 1]

@idse = internal global [2 x ptr] [ptr null, ptr null]
@lene = internal global [2 x i64] [i64 0, i64 0]

@rout_ids = internal global [8 x i64] zeroinitializer, align 8
@rout_sc  = internal global [8 x float] zeroinitializer, align 4
@rout_n   = internal global i64 0, align 8

; ------------------------------------------------------------- MMR fixtures
; lambda=1 pure-relevance case: distinct integer relevances.
@mrel = internal global [5 x float] [float 1.0, float 5.0, float 3.0, float 9.0, float 2.0], align 4
@mvec = internal global [20 x float] zeroinitializer, align 4   ; 5 x 4 (unused at lambda 1)
@mout = internal global [8 x i64] zeroinitializer, align 8
@moutn = internal global i64 0, align 8

; lambda=0 diversity case: v0==v1, v2 orthogonal (3 x 4)
@dvec = internal global [12 x float]
  [float 1.0, float 0.0, float 0.0, float 0.0,
   float 1.0, float 0.0, float 0.0, float 0.0,
   float 0.0, float 1.0, float 0.0, float 0.0], align 4
@drel = internal global [3 x float] [float 1.0, float 1.0, float 1.0], align 4
@dout = internal global [8 x i64] zeroinitializer, align 8
@doutn = internal global i64 0, align 8

; random-trial scratch (max n=64, max d=16)
@vbuf   = internal global [1024 x float] zeroinitializer, align 4
@relbuf = internal global [64 x float] zeroinitializer, align 4
@modout = internal global [64 x i64] zeroinitializer, align 8
@refout = internal global [64 x i64] zeroinitializer, align 8
@modn   = internal global i64 0, align 8
@refchosen = internal global [64 x i8] zeroinitializer, align 1
@refmaxsim = internal global [64 x float] zeroinitializer, align 4

; bench scratch
@bl     = internal global [1024 x i64] zeroinitializer, align 8
@bids   = internal global [4 x ptr] zeroinitializer, align 8
@blen   = internal global [4 x i64] [i64 256, i64 256, i64 256, i64 256], align 8
@boids  = internal global [64 x i64] zeroinitializer, align 8
@bosc   = internal global [64 x float] zeroinitializer, align 4
@bon    = internal global i64 0, align 8
@bvecs  = internal global [4096 x float] zeroinitializer, align 4
@brel   = internal global [64 x float] zeroinitializer, align 4
@bmout  = internal global [64 x i64] zeroinitializer, align 8
@bmn    = internal global i64 0, align 8
@samples = internal global [16 x double] zeroinitializer, align 8

; ------------------------------------------------------------- messages
@m.rc  = private unnamed_addr constant [12 x i8] c"rrf 3-list \00"
@m.rn  = private unnamed_addr constant [12 x i8] c"rrf 3 n==4 \00"
@m.ro0 = private unnamed_addr constant [12 x i8] c"rrf ord[0] \00"
@m.ro1 = private unnamed_addr constant [12 x i8] c"rrf ord[1] \00"
@m.ro2 = private unnamed_addr constant [12 x i8] c"rrf ord[2] \00"
@m.ro3 = private unnamed_addr constant [12 x i8] c"rrf ord[3] \00"
@m.rs0 = private unnamed_addr constant [12 x i8] c"rrf scr[0] \00"
@m.rs1 = private unnamed_addr constant [12 x i8] c"rrf scr[1] \00"
@m.rs3 = private unnamed_addr constant [12 x i8] c"rrf scr[3] \00"
@m.s1  = private unnamed_addr constant [12 x i8] c"rrf 1-list \00"
@m.s1s = private unnamed_addr constant [12 x i8] c"rrf 1 scr0 \00"
@m.tie = private unnamed_addr constant [12 x i8] c"rrf tie id \00"
@m.tie1= private unnamed_addr constant [12 x i8] c"rrf tie id1\00"
@m.emp = private unnamed_addr constant [12 x i8] c"rrf empty  \00"
@m.cap = private unnamed_addr constant [12 x i8] c"rrf cap<n  \00"
@m.capn= private unnamed_addr constant [12 x i8] c"rrf cap n=2\00"
@m.dfl = private unnamed_addr constant [12 x i8] c"rrf k=0 dfl\00"
@m.rnull = private unnamed_addr constant [12 x i8] c"rrf null=1 \00"

@m.l1n = private unnamed_addr constant [12 x i8] c"mmr L1 n==5\00"
@m.l1o = private unnamed_addr constant [12 x i8] c"mmr L1 rank\00"
@m.l0o = private unnamed_addr constant [12 x i8] c"mmr L0 div \00"
@m.l0n = private unnamed_addr constant [12 x i8] c"mmr L0 n==3\00"
@m.trial = private unnamed_addr constant [15 x i8] c"mmr trial==ref\00"
@m.tn  = private unnamed_addr constant [12 x i8] c"mmr trial n\00"
@m.kgtn= private unnamed_addr constant [12 x i8] c"mmr k>n n=3\00"
@m.k0  = private unnamed_addr constant [12 x i8] c"mmr k=0 n=0\00"
@m.n0  = private unnamed_addr constant [12 x i8] c"mmr n=0 n=0\00"
@m.mnull = private unnamed_addr constant [12 x i8] c"mmr null=1 \00"

@lbl.rrf = private unnamed_addr constant [9 x i8] c"rrf fuse\00"
@lbl.mmr = private unnamed_addr constant [9 x i8] c"mmr pick\00"

; ============================================================ approx float
define internal void @approx(float %v, float %ex, ptr %msg) {
entry:
  %d = fsub float %v, %ex
  %ad = call float @llvm.fabs.f32(float %d)
  %tol = fdiv float 1.0, 1.0e4                    ; ~1e-4, no hex-literal guesswork
  %ok = fcmp olt float %ad, %tol
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; ============================================================ scalar cosine
define internal float @cos_ref(ptr %a, ptr %b, i64 %d) {
entry:
  %z = icmp eq i64 %d, 0
  br i1 %z, label %fin, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %dot = phi float [ 0.0, %entry ], [ %dotn, %loop ]
  %na = phi float [ 0.0, %entry ], [ %nan, %loop ]
  %nb = phi float [ 0.0, %entry ], [ %nbn, %loop ]
  %pa = getelementptr inbounds float, ptr %a, i64 %i
  %fa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %b, i64 %i
  %fb = load float, ptr %pb, align 4
  %pd = fmul float %fa, %fb
  %dotn = fadd float %dot, %pd
  %pa2 = fmul float %fa, %fa
  %nan = fadd float %na, %pa2
  %pb2 = fmul float %fb, %fb
  %nbn = fadd float %nb, %pb2
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %d
  br i1 %more, label %loop, label %fin

fin:
  %fdot = phi float [ 0.0, %entry ], [ %dotn, %loop ]
  %fna  = phi float [ 0.0, %entry ], [ %nan, %loop ]
  %fnb  = phi float [ 0.0, %entry ], [ %nbn, %loop ]
  %sqa = call float @llvm.sqrt.f32(float %fna)
  %sqb = call float @llvm.sqrt.f32(float %fnb)
  %den = fmul float %sqa, %sqb
  %isz = fcmp oeq float %den, 0.0
  %sim = fdiv float %fdot, %den
  %res = select i1 %isz, float 0.0, float %sim
  ret float %res
}

; ============================================================ scalar MMR oracle
; Mirrors universe_ml_mmr exactly (clamp, greedy argmax w/ lowest-index tie,
; incremental maxsim). Writes selection into %out, returns want=min(k,n).
define internal i64 @mmr_ref(ptr %vecs, i64 %n, i64 %d, ptr %rel, float %lambda, i64 %k, ptr %out) {
entry:
  %kltn = icmp ult i64 %k, %n
  %want = select i1 %kltn, i64 %k, i64 %n
  ; reset scratch
  call void @llvm.memset.p0.i64(ptr @refchosen, i8 0, i64 %n, i1 false)
  %msbytes = shl i64 %n, 2
  call void @llvm.memset.p0.i64(ptr @refmaxsim, i8 0, i64 %msbytes, i1 false)
  %wz = icmp eq i64 %want, 0
  br i1 %wz, label %done, label %lam

lam:
  %llt0 = fcmp olt float %lambda, 0.0
  %lam0 = select i1 %llt0, float 0.0, float %lambda
  %lgt1 = fcmp ogt float %lam0, 1.0
  %lm = select i1 %lgt1, float 1.0, float %lam0
  %om = fsub float 1.0, %lm
  br label %round

round:
  %rc = phi i64 [ 0, %lam ], [ %rc2, %upd.done ]
  br label %scan

scan:
  %c = phi i64 [ 0, %round ], [ %cn, %scan ]
  %bi = phi i64 [ -1, %round ], [ %nbi, %scan ]
  %bs = phi float [ 0.0, %round ], [ %nbs, %scan ]
  %chp = getelementptr inbounds i8, ptr @refchosen, i64 %c
  %ch = load i8, ptr %chp, align 1
  %notch = icmp eq i8 %ch, 0
  %rp = getelementptr inbounds float, ptr %rel, i64 %c
  %rv = load float, ptr %rp, align 4
  %mp = getelementptr inbounds float, ptr @refmaxsim, i64 %c
  %mv = load float, ptr %mp, align 4
  %t1 = fmul float %lm, %rv
  %t2 = fmul float %om, %mv
  %sc = fsub float %t1, %t2
  %first = icmp eq i64 %bi, -1
  %gt = fcmp ogt float %sc, %bs
  %bet = or i1 %first, %gt
  %take = and i1 %notch, %bet
  %nbi = select i1 %take, i64 %c, i64 %bi
  %nbs = select i1 %take, float %sc, float %bs
  %cn = add nuw i64 %c, 1
  %sm = icmp ult i64 %cn, %n
  br i1 %sm, label %scan, label %scandone

scandone:
  %chpp = getelementptr inbounds i8, ptr @refchosen, i64 %nbi
  store i8 1, ptr %chpp, align 1
  %outp = getelementptr inbounds i64, ptr %out, i64 %rc
  store i64 %nbi, ptr %outp, align 8
  %poff = mul nuw i64 %nbi, %d
  %prow = getelementptr inbounds float, ptr %vecs, i64 %poff
  br label %upd

upd:
  %uc = phi i64 [ 0, %scandone ], [ %ucn, %upd.cont ]
  %uchp = getelementptr inbounds i8, ptr @refchosen, i64 %uc
  %uch = load i8, ptr %uchp, align 1
  %un = icmp eq i8 %uch, 0
  br i1 %un, label %upd.do, label %upd.cont

upd.do:
  %uoff = mul nuw i64 %uc, %d
  %urow = getelementptr inbounds float, ptr %vecs, i64 %uoff
  %s = call float @cos_ref(ptr %urow, ptr %prow, i64 %d)
  %ump = getelementptr inbounds float, ptr @refmaxsim, i64 %uc
  %umv = load float, ptr %ump, align 4
  %big = fcmp ogt float %s, %umv
  %nm = select i1 %big, float %s, float %umv
  store float %nm, ptr %ump, align 4
  br label %upd.cont

upd.cont:
  %ucn = add nuw i64 %uc, 1
  %um = icmp ult i64 %ucn, %n
  br i1 %um, label %upd, label %upd.done

upd.done:
  %rc2 = add nuw i64 %rc, 1
  %rm = icmp ult i64 %rc2, %want
  br i1 %rm, label %round, label %done

done:
  ret i64 %want
}

; ============================================================ fill int vec
; %buf[0..n) = (rand & 15) - 8  (integer-valued -> exact float sums)
define internal void @fill_int(ptr %buf, i64 %n, ptr %st) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %lo = and i64 %r, 15
  %s = sub i64 %lo, 8
  %f = sitofp i64 %s to float
  %p = getelementptr inbounds float, ptr %buf, i64 %i
  store float %f, ptr %p, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ============================================================ main
define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  ; ---------------------------------------------------------- RRF 3-list
  %r1 = call i32 @universe_ml_rrf(ptr @ids3, ptr @len3, i64 3, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr @rout_n)
  %r1ok = icmp eq i32 %r1, 0
  call void @ut_check(i1 %r1ok, ptr @m.rc)
  %rn = load i64, ptr @rout_n, align 8
  call void @ut_check_eq(i64 %rn, i64 4, ptr @m.rn)
  ; order: 10, 20, 40, 30
  %o0 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 0), align 8
  call void @ut_check_eq(i64 %o0, i64 10, ptr @m.ro0)
  %o1 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 1), align 8
  call void @ut_check_eq(i64 %o1, i64 20, ptr @m.ro1)
  %o2 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 2), align 8
  call void @ut_check_eq(i64 %o2, i64 40, ptr @m.ro2)
  %o3 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 3), align 8
  call void @ut_check_eq(i64 %o3, i64 30, ptr @m.ro3)
  ; scores (expected computed at runtime from 1/(k+rank) to avoid hex literals)
  %inv60 = fdiv float 1.0, 6.0e1
  %inv61 = fdiv float 1.0, 6.1e1
  %inv62 = fdiv float 1.0, 6.2e1
  %ex.10 = fadd float %inv60, %inv60   ; id10: 1/60 + 1/60
  %ex.20 = fadd float %inv61, %inv60   ; id20: 1/61 + 1/60
  %s0 = load float, ptr getelementptr inbounds ([8 x float], ptr @rout_sc, i64 0, i64 0), align 4
  call void @approx(float %s0, float %ex.10, ptr @m.rs0)
  %s1 = load float, ptr getelementptr inbounds ([8 x float], ptr @rout_sc, i64 0, i64 1), align 4
  call void @approx(float %s1, float %ex.20, ptr @m.rs1)
  %s3 = load float, ptr getelementptr inbounds ([8 x float], ptr @rout_sc, i64 0, i64 3), align 4
  call void @approx(float %s3, float %inv62, ptr @m.rs3)   ; id30: 1/62

  ; ---------------------------------------------------------- RRF single list
  %r2 = call i32 @universe_ml_rrf(ptr @ids1, ptr @len1, i64 1, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr @rout_n)
  %rn2 = load i64, ptr @rout_n, align 8
  call void @ut_check_eq(i64 %rn2, i64 3, ptr @m.s1)
  %ss0 = load float, ptr getelementptr inbounds ([8 x float], ptr @rout_sc, i64 0, i64 0), align 4
  %ex.s = fdiv float 1.0, 6.0e1
  call void @approx(float %ss0, float %ex.s, ptr @m.s1s)  ; 1/60

  ; ---------------------------------------------------------- RRF tie -> lowest id
  %r3 = call i32 @universe_ml_rrf(ptr @idst, ptr @lent, i64 2, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr @rout_n)
  %te0 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 0), align 8
  call void @ut_check_eq(i64 %te0, i64 100, ptr @m.tie)
  %te1 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 1), align 8
  call void @ut_check_eq(i64 %te1, i64 200, ptr @m.tie1)

  ; ---------------------------------------------------------- RRF empty lists
  %r4 = call i32 @universe_ml_rrf(ptr @idse, ptr @lene, i64 2, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr @rout_n)
  %rn4 = load i64, ptr @rout_n, align 8
  %r4ok = icmp eq i32 %r4, 0
  call void @ut_check(i1 %r4ok, ptr @m.emp)
  call void @ut_check_eq(i64 %rn4, i64 0, ptr @m.emp)

  ; ---------------------------------------------------------- RRF out_cap < distinct
  %r5 = call i32 @universe_ml_rrf(ptr @ids3, ptr @len3, i64 3, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 2, ptr @rout_n)
  %rn5 = load i64, ptr @rout_n, align 8
  call void @ut_check_eq(i64 %rn5, i64 2, ptr @m.capn)
  %c0 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 0), align 8
  %c1 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @rout_ids, i64 0, i64 1), align 8
  %cok0 = icmp eq i64 %c0, 10
  %cok1 = icmp eq i64 %c1, 20
  %cok = and i1 %cok0, %cok1
  call void @ut_check(i1 %cok, ptr @m.cap)

  ; ---------------------------------------------------------- RRF default k (0 -> 60)
  %r6 = call i32 @universe_ml_rrf(ptr @ids1, ptr @len1, i64 1, float 0.0,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr @rout_n)
  %d0 = load float, ptr getelementptr inbounds ([8 x float], ptr @rout_sc, i64 0, i64 0), align 4
  %ex.d = fdiv float 1.0, 6.0e1
  call void @approx(float %d0, float %ex.d, ptr @m.dfl)   ; 1/60

  ; ---------------------------------------------------------- RRF null arg -> 1
  %r7 = call i32 @universe_ml_rrf(ptr @ids1, ptr @len1, i64 1, float 6.0e1,
                                  ptr @rout_ids, ptr @rout_sc, i64 8, ptr null)
  %r7bad = icmp eq i32 %r7, 1
  call void @ut_check(i1 %r7bad, ptr @m.rnull)

  ; ---------------------------------------------------------- MMR lambda=1
  %mm1 = call i32 @universe_ml_mmr(ptr @mvec, i64 5, i64 4, ptr @mrel, float 1.0,
                                   i64 5, ptr @mout, ptr @moutn)
  %mn1 = load i64, ptr @moutn, align 8
  call void @ut_check_eq(i64 %mn1, i64 5, ptr @m.l1n)
  ; expected descending-rel order: 3,1,2,4,0
  %e3 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @mout, i64 0, i64 0), align 8
  %e1 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @mout, i64 0, i64 1), align 8
  %e2 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @mout, i64 0, i64 2), align 8
  %e4 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @mout, i64 0, i64 3), align 8
  %e0 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @mout, i64 0, i64 4), align 8
  %q3 = icmp eq i64 %e3, 3
  %q1 = icmp eq i64 %e1, 1
  %q2 = icmp eq i64 %e2, 2
  %q4 = icmp eq i64 %e4, 4
  %q0 = icmp eq i64 %e0, 0
  %qa = and i1 %q3, %q1
  %qb = and i1 %q2, %q4
  %qc = and i1 %qa, %qb
  %qd = and i1 %qc, %q0
  call void @ut_check(i1 %qd, ptr @m.l1o)

  ; ---------------------------------------------------------- MMR lambda=0 diversity
  %mm0 = call i32 @universe_ml_mmr(ptr @dvec, i64 3, i64 4, ptr @drel, float 0.0,
                                   i64 3, ptr @dout, ptr @doutn)
  %dn = load i64, ptr @doutn, align 8
  call void @ut_check_eq(i64 %dn, i64 3, ptr @m.l0n)
  ; expected pick order: 0, 2, 1
  %g0 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @dout, i64 0, i64 0), align 8
  %g1 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @dout, i64 0, i64 1), align 8
  %g2 = load i64, ptr getelementptr inbounds ([8 x i64], ptr @dout, i64 0, i64 2), align 8
  %d0k = icmp eq i64 %g0, 0
  %d2k = icmp eq i64 %g1, 2
  %d1k = icmp eq i64 %g2, 1
  %dka = and i1 %d0k, %d2k
  %dkb = and i1 %dka, %d1k
  call void @ut_check(i1 %dkb, ptr @m.l0o)

  ; ---------------------------------------------------------- MMR random trials
  br label %trial.pre

trial.pre:
  store i64 88172645463325252, ptr %st, align 8
  br label %trial

; trial parameters cycle over a small fixed table via index
trial:
  %ti = phi i64 [ 0, %trial.pre ], [ %tin, %trial.cont ]
  %viol = phi i64 [ 0, %trial.pre ], [ %viol2, %trial.cont ]
  ; n = 4 + (ti*3 % 9)   -> 4..12 ; d = 4 + (ti%3)*4 -> 4,8,12 ; k = n/2+1
  %m1 = mul i64 %ti, 3
  %m2 = urem i64 %m1, 9
  %n = add i64 4, %m2
  %dm = urem i64 %ti, 3
  %dmm = mul i64 %dm, 4
  %d = add i64 4, %dmm
  %khalf = lshr i64 %n, 1
  %k = add i64 %khalf, 1
  ; lambda cycles 0.0, 0.5, 1.0
  %lmi = urem i64 %ti, 3
  %isl0 = icmp eq i64 %lmi, 0
  %isl1 = icmp eq i64 %lmi, 1
  %lam.a = select i1 %isl1, float 5.0e-1, float 1.0
  %lambda = select i1 %isl0, float 0.0, float %lam.a
  ; fill vectors (n*d ints) and relevances (n ints in [-8,7])
  %nd = mul i64 %n, %d
  call void @fill_int(ptr @vbuf, i64 %nd, ptr %st)
  call void @fill_int(ptr @relbuf, i64 %n, ptr %st)
  ; module + oracle
  %rc.m = call i32 @universe_ml_mmr(ptr @vbuf, i64 %n, i64 %d, ptr @relbuf, float %lambda,
                                    i64 %k, ptr @modout, ptr @modn)
  %want.r = call i64 @mmr_ref(ptr @vbuf, i64 %n, i64 %d, ptr @relbuf, float %lambda,
                              i64 %k, ptr @refout)
  %mnn = load i64, ptr @modn, align 8
  %neqn = icmp ne i64 %mnn, %want.r
  %vinc.n = zext i1 %neqn to i64
  %viol.n = add i64 %viol, %vinc.n
  br label %cmp

cmp:
  %ci = phi i64 [ 0, %trial ], [ %cin, %cmp ]
  %vc = phi i64 [ %viol.n, %trial ], [ %vc2, %cmp ]
  %mop = getelementptr inbounds i64, ptr @modout, i64 %ci
  %mo = load i64, ptr %mop, align 8
  %rop = getelementptr inbounds i64, ptr @refout, i64 %ci
  %ro = load i64, ptr %rop, align 8
  %ne = icmp ne i64 %mo, %ro
  %vinc = zext i1 %ne to i64
  %vc2 = add i64 %vc, %vinc
  %cin = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %cin, %want.r
  br i1 %cmore, label %cmp, label %trial.cont

trial.cont:
  %viol2 = phi i64 [ %vc2, %cmp ]
  %tin = add nuw i64 %ti, 1
  %tmore = icmp ult i64 %tin, 30
  br i1 %tmore, label %trial, label %trial.after

trial.after:
  call void @ut_check_eq(i64 %viol2, i64 0, ptr @m.trial)

  ; ---------------------------------------------------------- MMR edges
  ; k>n
  %ek = call i32 @universe_ml_mmr(ptr @dvec, i64 3, i64 4, ptr @drel, float 0.5,
                                  i64 99, ptr @dout, ptr @doutn)
  %ekn = load i64, ptr @doutn, align 8
  call void @ut_check_eq(i64 %ekn, i64 3, ptr @m.kgtn)
  ; k=0
  %e0k = call i32 @universe_ml_mmr(ptr @dvec, i64 3, i64 4, ptr @drel, float 0.5,
                                   i64 0, ptr @dout, ptr @doutn)
  %e0kn = load i64, ptr @doutn, align 8
  call void @ut_check_eq(i64 %e0kn, i64 0, ptr @m.k0)
  ; n=0
  %en0 = call i32 @universe_ml_mmr(ptr @dvec, i64 0, i64 4, ptr @drel, float 0.5,
                                   i64 5, ptr @dout, ptr @doutn)
  %en0n = load i64, ptr @doutn, align 8
  call void @ut_check_eq(i64 %en0n, i64 0, ptr @m.n0)
  ; null arg
  %emn = call i32 @universe_ml_mmr(ptr @dvec, i64 3, i64 4, ptr @drel, float 0.5,
                                   i64 3, ptr @dout, ptr null)
  %emnbad = icmp eq i32 %emn, 1
  call void @ut_check(i1 %emnbad, ptr @m.mnull)

  ; ---------------------------------------------------------- bench
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %fin

do.bench:
  call void @run_bench()
  br label %fin

fin:
  %rc0 = call i32 @ut_summary()
  ret i32 %rc0
}

; ============================================================ bench
define internal void @run_bench() {
entry:
  %st = alloca i64, align 8
  store i64 1234567891011, ptr %st, align 8
  ; fill 4 lists of 256 ids each in [0,4096)
  br label %fillids

fillids:
  %i = phi i64 [ 0, %entry ], [ %in, %fillids ]
  %r = call i64 @ut_rand(ptr %st)
  %id = and i64 %r, 4095
  %p = getelementptr inbounds i64, ptr @bl, i64 %i
  store i64 %id, ptr %p, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, 1024
  br i1 %more, label %fillids, label %setptrs

setptrs:
  ; bids[j] = &bl[j*256]
  %b0 = getelementptr inbounds i64, ptr @bl, i64 0
  store ptr %b0, ptr getelementptr inbounds ([4 x ptr], ptr @bids, i64 0, i64 0), align 8
  %b1 = getelementptr inbounds i64, ptr @bl, i64 256
  store ptr %b1, ptr getelementptr inbounds ([4 x ptr], ptr @bids, i64 0, i64 1), align 8
  %b2 = getelementptr inbounds i64, ptr @bl, i64 512
  store ptr %b2, ptr getelementptr inbounds ([4 x ptr], ptr @bids, i64 0, i64 2), align 8
  %b3 = getelementptr inbounds i64, ptr @bl, i64 768
  store ptr %b3, ptr getelementptr inbounds ([4 x ptr], ptr @bids, i64 0, i64 3), align 8
  ; warm up
  %w = call i32 @universe_ml_rrf(ptr @bids, ptr @blen, i64 4, float 6.0e1,
                                 ptr @boids, ptr @bosc, i64 64, ptr @bon)
  br label %rrf.reps

rrf.reps:
  %ri = phi i64 [ 0, %setptrs ], [ %rin, %rrf.reps ]
  %t0 = call double @ut_now_sec()
  %rr = call i32 @universe_ml_rrf(ptr @bids, ptr @blen, i64 4, float 6.0e1,
                                  ptr @boids, ptr @bosc, i64 64, ptr @bon)
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %sp = getelementptr inbounds double, ptr @samples, i64 %ri
  store double %dt, ptr %sp, align 8
  %rin = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %rin, 16
  br i1 %rmore, label %rrf.reps, label %rrf.report

rrf.report:
  call void @ut_report_dist(ptr @samples, i64 16, i64 1024, ptr @lbl.rrf)
  ; MMR bench: n=64, d=64, k=16
  call void @fill_int(ptr @bvecs, i64 4096, ptr %st)
  call void @fill_int(ptr @brel, i64 64, ptr %st)
  %mw = call i32 @universe_ml_mmr(ptr @bvecs, i64 64, i64 64, ptr @brel, float 5.0e-1,
                                  i64 16, ptr @bmout, ptr @bmn)
  br label %mmr.reps

mmr.reps:
  %mi = phi i64 [ 0, %rrf.report ], [ %min, %mmr.reps ]
  %mt0 = call double @ut_now_sec()
  %mr = call i32 @universe_ml_mmr(ptr @bvecs, i64 64, i64 64, ptr @brel, float 5.0e-1,
                                  i64 16, ptr @bmout, ptr @bmn)
  %mt1 = call double @ut_now_sec()
  %mdt = fsub double %mt1, %mt0
  %msp = getelementptr inbounds double, ptr @samples, i64 %mi
  store double %mdt, ptr %msp, align 8
  %min = add nuw i64 %mi, 1
  %mmore = icmp ult i64 %min, 16
  br i1 %mmore, label %mmr.reps, label %mmr.report

mmr.report:
  ; ops ~ k*n*d = 16*64*64 = 65536 cosine-dim ops
  call void @ut_report_dist(ptr @samples, i64 16, i64 65536, ptr @lbl.mmr)
  ret void
}
