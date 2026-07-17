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

; Tests for universe_ml_ivf_* — the correctness gate is RECALL@10 vs an exact
; brute-force oracle over a fixed-seed clustered corpus (n=4000, d=32, nlist=64):
;   * nprobe=nlist scans every cell => exhaustive => recall ~= 1.0 (>= 0.99,
;     allowing fp-reassociation tie boundaries),
;   * nprobe=8 => a lower but bounded recall (>= 0.80 on clustered data).
; Plus edge cases: search-before-train => 8, train-empty => 4, k>total clamps,
; single-cell == exact flat, nprobe>nlist clamps, bad create args => null,
; a cosine smoke path. A --bench mode times the probe search.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)

declare ptr @universe_ml_ivf_create(i64, i32, i64)
declare i32 @universe_ml_ivf_add(ptr, ptr, i64)
declare i32 @universe_ml_ivf_train(ptr)
declare i32 @universe_ml_ivf_search(ptr, ptr, i64, i64, ptr, ptr, ptr)
declare i64 @universe_ml_ivf_len(ptr)
declare void @universe_ml_ivf_destroy(ptr)

@m.badc1 = private unnamed_addr constant [16 x i8] c"create dims=0 0\00", align 1
@m.badc2 = private unnamed_addr constant [16 x i8] c"create metric=2\00", align 1
@m.badc3 = private unnamed_addr constant [15 x i8] c"create nlist=0\00", align 1
@m.empty = private unnamed_addr constant [16 x i8] c"train empty ->4\00", align 1
@m.len   = private unnamed_addr constant [10 x i8] c"len==4000\00", align 1
@m.pre   = private unnamed_addr constant [17 x i8] c"search pre-train\00", align 1
@m.tr    = private unnamed_addr constant [11 x i8] c"train ok 0\00", align 1
@m.rcf   = private unnamed_addr constant [15 x i8] c"search full rc\00", align 1
@m.nf    = private unnamed_addr constant [14 x i8] c"search full=k\00", align 1
@m.rc8   = private unnamed_addr constant [14 x i8] c"search np8 rc\00", align 1
@m.rfull = private unnamed_addr constant [18 x i8] c"recall full>=0.99\00", align 1
@m.r8    = private unnamed_addr constant [17 x i8] c"recall np8>=0.80\00", align 1
@m.ktot  = private unnamed_addr constant [17 x i8] c"k>total out_n==3\00", align 1
@m.ktrc  = private unnamed_addr constant [13 x i8] c"k>total rc 0\00", align 1
@m.clmp  = private unnamed_addr constant [18 x i8] c"nprobe clamp n==k\00", align 1
@m.sc    = private unnamed_addr constant [18 x i8] c"single-cell exact\00", align 1
@m.cos   = private unnamed_addr constant [14 x i8] c"cosine smoke\0A\00", align 1
@m.cosrc = private unnamed_addr constant [13 x i8] c"cosine rc==0\00", align 1
@m.fp    = private unnamed_addr constant [32 x i8] c"recall full=%lld/1000 np8=%lld\0A\00", align 1
@m.bh    = private unnamed_addr constant [41 x i8] c"bench ivf np8: min=%.3f ms mean=%.3f ms\0A\00", align 1

; ---------------------------------------------------------------- frand [0,1)
define internal float @frand(ptr %st) {
entry:
  %r = call i64 @ut_rand(ptr %st)
  %low = and i64 %r, 16777215
  %f = uitofp i64 %low to float
  %s = fmul float %f, 0x3E70000000000000
  ret float %s
}

; ------------------------------------------------ dist2s (scalar oracle L2^2)
define internal float @dist2s(ptr %a, ptr %b, i64 %d) {
entry:
  br label %head
head:
  %j = phi i64 [ 0, %entry ], [ %jn, %body ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %body ]
  %go = icmp ult i64 %j, %d
  br i1 %go, label %body, label %ret
body:
  %pa = getelementptr inbounds float, ptr %a, i64 %j
  %xa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %b, i64 %j
  %xb = load float, ptr %pb, align 4
  %df = fsub float %xa, %xb
  %sq = fmul float %df, %df
  %accn = fadd float %acc, %sq
  %jn = add nuw i64 %j, 1
  br label %head
ret:
  ret float %acc
}

; --------------------------------------- brute_topk10: 10 nearest indices
define internal void @brute_topk10(ptr %corpus, i64 %n, i64 %d, ptr %q, ptr %outidx) {
entry:
  %bd = alloca [10 x float], align 4
  %bi = alloca [10 x i64], align 8
  br label %ih
ih:
  %ii = phi i64 [ 0, %entry ], [ %iin, %ih ]
  %bdp = getelementptr inbounds [10 x float], ptr %bd, i64 0, i64 %ii
  store float 0x7FF0000000000000, ptr %bdp, align 4
  %bip = getelementptr inbounds [10 x i64], ptr %bi, i64 0, i64 %ii
  store i64 -1, ptr %bip, align 8
  %iin = add nuw i64 %ii, 1
  %imore = icmp ult i64 %iin, 10
  br i1 %imore, label %ih, label %sh
sh:
  %s = phi i64 [ 0, %ih ], [ %sn, %sc ]
  %sgo = icmp ult i64 %s, %n
  br i1 %sgo, label %sb, label %wr
sb:
  %roff = mul nuw i64 %s, %d
  %row = getelementptr inbounds float, ptr %corpus, i64 %roff
  %dist = call float @dist2s(ptr %q, ptr %row, i64 %d)
  %d0 = load float, ptr %bd, align 4
  br label %mh
mh:
  %p = phi i64 [ 1, %sb ], [ %pn, %mb ]
  %mpos = phi i64 [ 0, %sb ], [ %mposn, %mb ]
  %mval = phi float [ %d0, %sb ], [ %mvaln, %mb ]
  %pgo = icmp ult i64 %p, 10
  br i1 %pgo, label %mb, label %md
mb:
  %pp = getelementptr inbounds [10 x float], ptr %bd, i64 0, i64 %p
  %pv = load float, ptr %pp, align 4
  %gt = fcmp ogt float %pv, %mval
  %mposn = select i1 %gt, i64 %p, i64 %mpos
  %mvaln = select i1 %gt, float %pv, float %mval
  %pn = add nuw i64 %p, 1
  br label %mh
md:
  %better = fcmp olt float %dist, %mval
  br i1 %better, label %rep, label %sc
rep:
  %rdp = getelementptr inbounds [10 x float], ptr %bd, i64 0, i64 %mpos
  store float %dist, ptr %rdp, align 4
  %rip = getelementptr inbounds [10 x i64], ptr %bi, i64 0, i64 %mpos
  store i64 %s, ptr %rip, align 8
  br label %sc
sc:
  %sn = add nuw i64 %s, 1
  br label %sh
wr:
  br label %wh
wh:
  %w = phi i64 [ 0, %wr ], [ %wn, %wh ]
  %sbi = getelementptr inbounds [10 x i64], ptr %bi, i64 0, i64 %w
  %sv = load i64, ptr %sbi, align 8
  %obi = getelementptr inbounds i64, ptr %outidx, i64 %w
  store i64 %sv, ptr %obi, align 8
  %wn = add nuw i64 %w, 1
  %wmore = icmp ult i64 %wn, 10
  br i1 %wmore, label %wh, label %done
done:
  ret void
}

; ---------------------------- count_matches: overlap of labels[0..k) with idx[0..k)
define internal i64 @count_matches(ptr %labels, ptr %idx, i64 %k) {
entry:
  br label %oh
oh:
  %i = phi i64 [ 0, %entry ], [ %in, %oc ]
  %m = phi i64 [ 0, %entry ], [ %mn, %oc ]
  %ogo = icmp ult i64 %i, %k
  br i1 %ogo, label %ob, label %ret
ob:
  %lp = getelementptr inbounds i64, ptr %labels, i64 %i
  %lv = load i64, ptr %lp, align 8
  br label %jh
jh:
  %j = phi i64 [ 0, %ob ], [ %jn, %jc ]
  %found = phi i64 [ 0, %ob ], [ %foundn, %jc ]
  %jgo = icmp ult i64 %j, %k
  br i1 %jgo, label %jb, label %jd
jb:
  %ip = getelementptr inbounds i64, ptr %idx, i64 %j
  %iv = load i64, ptr %ip, align 8
  %eq = icmp eq i64 %lv, %iv
  %inc = zext i1 %eq to i64
  br label %jc
jc:
  %foundn = add i64 %found, %inc
  %jn = add nuw i64 %j, 1
  br label %jh
jd:
  %hit = icmp ugt i64 %found, 0
  %hinc = zext i1 %hit to i64
  br label %oc
oc:
  %mn = add i64 %m, %hinc
  %in = add nuw i64 %i, 1
  br label %oh
ret:
  ret i64 %m
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8
  %outl = alloca [10 x i64], align 8
  %outd = alloca [10 x float], align 4
  %outn = alloca i64, align 8
  %idx  = alloca [10 x i64], align 8

  ; ---- bad create args -> null ----
  %bc1 = call ptr @universe_ml_ivf_create(i64 0, i32 1, i64 64)
  %bc1n = icmp eq ptr %bc1, null
  call void @ut_check(i1 %bc1n, ptr @m.badc1)
  %bc2 = call ptr @universe_ml_ivf_create(i64 32, i32 2, i64 64)
  %bc2n = icmp eq ptr %bc2, null
  call void @ut_check(i1 %bc2n, ptr @m.badc2)
  %bc3 = call ptr @universe_ml_ivf_create(i64 32, i32 1, i64 0)
  %bc3n = icmp eq ptr %bc3, null
  call void @ut_check(i1 %bc3n, ptr @m.badc3)

  ; ---- train empty -> 4 ----
  %h0 = call ptr @universe_ml_ivf_create(i64 32, i32 1, i64 4)
  %e0 = call i32 @universe_ml_ivf_train(ptr %h0)
  %e0i = sext i32 %e0 to i64
  call void @ut_check_eq(i64 %e0i, i64 4, ptr @m.empty)
  call void @universe_ml_ivf_destroy(ptr %h0)

  ; ---- build clustered corpus: 64 centers in [0,16), points = center + [-.5,.5) ----
  %centers = call ptr @malloc(i64 8192)   ; 64*32*4
  br label %ch
ch:
  %cc = phi i64 [ 0, %entry ], [ %ccn, %chnext ]
  %cdd = mul nuw i64 %cc, 32
  br label %chd
chd:
  %cj = phi i64 [ 0, %ch ], [ %cjn, %chd ]
  %coff = add nuw i64 %cdd, %cj
  %cf = call float @frand(ptr %st)
  %cf16 = fmul float %cf, 16.0
  %cp = getelementptr inbounds float, ptr %centers, i64 %coff
  store float %cf16, ptr %cp, align 4
  %cjn = add nuw i64 %cj, 1
  %cjm = icmp ult i64 %cjn, 32
  br i1 %cjm, label %chd, label %chnext
chnext:
  %ccn = add nuw i64 %cc, 1
  %ccm = icmp ult i64 %ccn, 64
  br i1 %ccm, label %ch, label %corp

corp:
  %corpus = call ptr @malloc(i64 512000)  ; 4000*32*4
  br label %ph
ph:
  %pi = phi i64 [ 0, %corp ], [ %pin, %phnext ]
  %clr = call i64 @ut_rand(ptr %st)
  %cl = and i64 %clr, 63
  %clbase = mul nuw i64 %cl, 32
  %pbase = mul nuw i64 %pi, 32
  br label %phd
phd:
  %pj = phi i64 [ 0, %ph ], [ %pjn, %phd ]
  %cco = add nuw i64 %clbase, %pj
  %ccp = getelementptr inbounds float, ptr %centers, i64 %cco
  %ccv = load float, ptr %ccp, align 4
  %nz = call float @frand(ptr %st)
  %nzc = fsub float %nz, 0.5
  %pv = fadd float %ccv, %nzc
  %poo = add nuw i64 %pbase, %pj
  %pp = getelementptr inbounds float, ptr %corpus, i64 %poo
  store float %pv, ptr %pp, align 4
  %pjn = add nuw i64 %pj, 1
  %pjm = icmp ult i64 %pjn, 32
  br i1 %pjm, label %phd, label %phnext
phnext:
  %pin = add nuw i64 %pi, 1
  %pim = icmp ult i64 %pin, 4000
  br i1 %pim, label %ph, label %build

build:
  %h = call ptr @universe_ml_ivf_create(i64 32, i32 1, i64 64)
  br label %ah
ah:
  %ai = phi i64 [ 0, %build ], [ %ain, %ah ]
  %aoff = mul nuw i64 %ai, 32
  %ap = getelementptr inbounds float, ptr %corpus, i64 %aoff
  %arc = call i32 @universe_ml_ivf_add(ptr %h, ptr %ap, i64 %ai)
  %ain = add nuw i64 %ai, 1
  %am = icmp ult i64 %ain, 4000
  br i1 %am, label %ah, label %postadd

postadd:
  %ln = call i64 @universe_ml_ivf_len(ptr %h)
  call void @ut_check_eq(i64 %ln, i64 4000, ptr @m.len)

  ; ---- search before train -> 8 ----
  %pre = call i32 @universe_ml_ivf_search(ptr %h, ptr %corpus, i64 10, i64 8,
                                          ptr %outl, ptr %outd, ptr %outn)
  %prei = sext i32 %pre to i64
  call void @ut_check_eq(i64 %prei, i64 8, ptr @m.pre)

  ; ---- train ----
  %trc = call i32 @universe_ml_ivf_train(ptr %h)
  %trci = sext i32 %trc to i64
  call void @ut_check_eq(i64 %trci, i64 0, ptr @m.tr)

  ; ---- build 100 queries (clustered) ----
  %queries = call ptr @malloc(i64 12800)  ; 100*32*4
  br label %qh
qh:
  %qi = phi i64 [ 0, %postadd ], [ %qin, %qhnext ]
  %qclr = call i64 @ut_rand(ptr %st)
  %qcl = and i64 %qclr, 63
  %qclbase = mul nuw i64 %qcl, 32
  %qbase = mul nuw i64 %qi, 32
  br label %qhd
qhd:
  %qj = phi i64 [ 0, %qh ], [ %qjn, %qhd ]
  %qco = add nuw i64 %qclbase, %qj
  %qcp = getelementptr inbounds float, ptr %centers, i64 %qco
  %qcv = load float, ptr %qcp, align 4
  %qnz = call float @frand(ptr %st)
  %qnzc = fsub float %qnz, 0.5
  %qv = fadd float %qcv, %qnzc
  %qoo = add nuw i64 %qbase, %qj
  %qp = getelementptr inbounds float, ptr %queries, i64 %qoo
  store float %qv, ptr %qp, align 4
  %qjn = add nuw i64 %qj, 1
  %qjm = icmp ult i64 %qjn, 32
  br i1 %qjm, label %qhd, label %qhnext
qhnext:
  %qin = add nuw i64 %qi, 1
  %qim = icmp ult i64 %qin, 100
  br i1 %qim, label %qh, label %recall

  ; ---- recall loop over queries ----
recall:
  br label %rh
rh:
  %ri = phi i64 [ 0, %recall ], [ %rin, %rc ]
  %mfull = phi i64 [ 0, %recall ], [ %mfulln, %rc ]
  %m8 = phi i64 [ 0, %recall ], [ %m8n, %rc ]
  %rgo = icmp ult i64 %ri, 100
  br i1 %rgo, label %rb, label %rdone
rb:
  %roff = mul nuw i64 %ri, 32
  %qptr = getelementptr inbounds float, ptr %queries, i64 %roff
  call void @brute_topk10(ptr %corpus, i64 4000, i64 32, ptr %qptr, ptr %idx)
  ; full: nprobe=64
  %rcf = call i32 @universe_ml_ivf_search(ptr %h, ptr %qptr, i64 10, i64 64,
                                          ptr %outl, ptr %outd, ptr %outn)
  %rcfi = sext i32 %rcf to i64
  call void @ut_check_eq(i64 %rcfi, i64 0, ptr @m.rcf)
  %onv = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %onv, i64 10, ptr @m.nf)
  %mf = call i64 @count_matches(ptr %outl, ptr %idx, i64 10)
  ; np8
  %rc8 = call i32 @universe_ml_ivf_search(ptr %h, ptr %qptr, i64 10, i64 8,
                                          ptr %outl, ptr %outd, ptr %outn)
  %rc8i = sext i32 %rc8 to i64
  call void @ut_check_eq(i64 %rc8i, i64 0, ptr @m.rc8)
  %m8c = call i64 @count_matches(ptr %outl, ptr %idx, i64 10)
  br label %rc
rc:
  %mfulln = add i64 %mfull, %mf
  %m8n = add i64 %m8, %m8c
  %rin = add nuw i64 %ri, 1
  br label %rh
rdone:
  ; total = 100*10 = 1000
  %fp = call i32 (ptr, ...) @printf(ptr @m.fp, i64 %mfull, i64 %m8)
  ; recall_full >= 0.99  => mfull*100 >= 99*1000
  %f100 = mul i64 %mfull, 100
  %fok = icmp uge i64 %f100, 99000
  call void @ut_check(i1 %fok, ptr @m.rfull)
  ; recall_np8 >= 0.80 => m8*100 >= 80*1000
  %e100 = mul i64 %m8, 100
  %eok = icmp uge i64 %e100, 80000
  call void @ut_check(i1 %eok, ptr @m.r8)

  ; ---- nprobe > nlist clamps (huge nprobe behaves like full) ----
  %clq = getelementptr inbounds float, ptr %queries, i64 0
  %clrc = call i32 @universe_ml_ivf_search(ptr %h, ptr %clq, i64 10, i64 1000000,
                                           ptr %outl, ptr %outd, ptr %outn)
  %clon = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %clon, i64 10, ptr @m.clmp)

  ; ---- k > total clamps out_n ----
  %tbuf = call ptr @malloc(i64 384)  ; 3*32*4
  br label %th
th:
  %ti = phi i64 [ 0, %rdone ], [ %tin, %thn ]
  %tb = mul nuw i64 %ti, 32
  br label %thd
thd:
  %tj = phi i64 [ 0, %th ], [ %tjn, %thd ]
  %too = add nuw i64 %tb, %tj
  %tf = call float @frand(ptr %st)
  %tp = getelementptr inbounds float, ptr %tbuf, i64 %too
  store float %tf, ptr %tp, align 4
  %tjn = add nuw i64 %tj, 1
  %tjm = icmp ult i64 %tjn, 32
  br i1 %tjm, label %thd, label %thn
thn:
  %tin = add nuw i64 %ti, 1
  %tim = icmp ult i64 %tin, 3
  br i1 %tim, label %th, label %tbuild
tbuild:
  %ht = call ptr @universe_ml_ivf_create(i64 32, i32 1, i64 1)
  %ta0 = getelementptr inbounds float, ptr %tbuf, i64 0
  %tar0 = call i32 @universe_ml_ivf_add(ptr %ht, ptr %ta0, i64 100)
  %ta1 = getelementptr inbounds float, ptr %tbuf, i64 32
  %tar1 = call i32 @universe_ml_ivf_add(ptr %ht, ptr %ta1, i64 101)
  %ta2 = getelementptr inbounds float, ptr %tbuf, i64 64
  %tar2 = call i32 @universe_ml_ivf_add(ptr %ht, ptr %ta2, i64 102)
  %ttr = call i32 @universe_ml_ivf_train(ptr %ht)
  %tsrc = call i32 @universe_ml_ivf_search(ptr %ht, ptr %ta0, i64 10, i64 1,
                                           ptr %outl, ptr %outd, ptr %outn)
  %tsrci = sext i32 %tsrc to i64
  call void @ut_check_eq(i64 %tsrci, i64 0, ptr @m.ktrc)
  %ton = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %ton, i64 3, ptr @m.ktot)
  call void @universe_ml_ivf_destroy(ptr %ht)
  call void @free(ptr %tbuf)

  ; ---- single cell (nlist=1) == exact flat: top-10 of a 50-point set ----
  %sbuf = call ptr @malloc(i64 6400)  ; 50*32*4
  br label %ssh
ssh:
  %ssi = phi i64 [ 0, %tbuild ], [ %ssin, %ssn ]
  %ssb = mul nuw i64 %ssi, 32
  br label %sshd
sshd:
  %ssj = phi i64 [ 0, %ssh ], [ %ssjn, %sshd ]
  %ssoo = add nuw i64 %ssb, %ssj
  %ssf = call float @frand(ptr %st)
  %ssf10 = fmul float %ssf, 10.0
  %ssp = getelementptr inbounds float, ptr %sbuf, i64 %ssoo
  store float %ssf10, ptr %ssp, align 4
  %ssjn = add nuw i64 %ssj, 1
  %ssjm = icmp ult i64 %ssjn, 32
  br i1 %ssjm, label %sshd, label %ssn
ssn:
  %ssin = add nuw i64 %ssi, 1
  %ssim = icmp ult i64 %ssin, 50
  br i1 %ssim, label %ssh, label %sbuild
sbuild:
  %hs = call ptr @universe_ml_ivf_create(i64 32, i32 1, i64 1)
  br label %sah
sah:
  %sai = phi i64 [ 0, %sbuild ], [ %sain, %sah ]
  %saoff = mul nuw i64 %sai, 32
  %sap = getelementptr inbounds float, ptr %sbuf, i64 %saoff
  %sarc = call i32 @universe_ml_ivf_add(ptr %hs, ptr %sap, i64 %sai)
  %sain = add nuw i64 %sai, 1
  %sam = icmp ult i64 %sain, 50
  br i1 %sam, label %sah, label %strain
strain:
  %sttr = call i32 @universe_ml_ivf_train(ptr %hs)
  %sq = getelementptr inbounds float, ptr %sbuf, i64 0   ; query = point 0
  %ssrc = call i32 @universe_ml_ivf_search(ptr %hs, ptr %sq, i64 10, i64 1,
                                           ptr %outl, ptr %outd, ptr %outn)
  call void @brute_topk10(ptr %sbuf, i64 50, i64 32, ptr %sq, ptr %idx)
  %sm = call i64 @count_matches(ptr %outl, ptr %idx, i64 10)
  ; single cell scans all 50 -> exact; require full overlap
  %smok = icmp uge i64 %sm, 10
  call void @ut_check(i1 %smok, ptr @m.sc)
  call void @universe_ml_ivf_destroy(ptr %hs)
  call void @free(ptr %sbuf)

  ; ---- cosine smoke: normalize-on-add + query-normalize paths ----
  %hc = call ptr @universe_ml_ivf_create(i64 32, i32 0, i64 4)
  br label %cah
cah:
  %cai = phi i64 [ 0, %strain ], [ %cain, %cah ]
  %caoff = mul nuw i64 %cai, 32
  %cap2 = getelementptr inbounds float, ptr %corpus, i64 %caoff
  %carc = call i32 @universe_ml_ivf_add(ptr %hc, ptr %cap2, i64 %cai)
  %cain = add nuw i64 %cai, 1
  %cam = icmp ult i64 %cain, 200
  br i1 %cam, label %cah, label %ctrain
ctrain:
  %cttr = call i32 @universe_ml_ivf_train(ptr %hc)
  %csrc = call i32 @universe_ml_ivf_search(ptr %hc, ptr %corpus, i64 10, i64 2,
                                           ptr %outl, ptr %outd, ptr %outn)
  %csrci = sext i32 %csrc to i64
  call void @ut_check_eq(i64 %csrci, i64 0, ptr @m.cosrc)
  %cson = load i64, ptr %outn, align 8
  %csonok = icmp eq i64 %cson, 10
  call void @ut_check(i1 %csonok, ptr @m.cos)
  call void @universe_ml_ivf_destroy(ptr %hc)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %cleanup

bench:
  ; warm-up sweep
  br label %bw
bw:
  %bwi = phi i64 [ 0, %bench ], [ %bwin, %bw ]
  %bwoff = mul nuw i64 %bwi, 32
  %bwq = getelementptr inbounds float, ptr %queries, i64 %bwoff
  %bwr = call i32 @universe_ml_ivf_search(ptr %h, ptr %bwq, i64 10, i64 8,
                                          ptr %outl, ptr %outd, ptr %outn)
  %bwin = add nuw i64 %bwi, 1
  %bwm = icmp ult i64 %bwin, 100
  br i1 %bwm, label %bw, label %breps
breps:
  br label %brh
brh:
  %rep = phi i64 [ 0, %breps ], [ %repn, %brc ]
  %bmin = phi double [ 1.0e30, %breps ], [ %bminn, %brc ]
  %bsum = phi double [ 0.0, %breps ], [ %bsumn, %brc ]
  %repgo = icmp ult i64 %rep, 16
  br i1 %repgo, label %brb, label %brdone
brb:
  %t0 = call double @ut_now_sec()
  br label %bsh
bsh:
  %bsi = phi i64 [ 0, %brb ], [ %bsin, %bsh ]
  %bsoff = mul nuw i64 %bsi, 32
  %bsq = getelementptr inbounds float, ptr %queries, i64 %bsoff
  %bsr = call i32 @universe_ml_ivf_search(ptr %h, ptr %bsq, i64 10, i64 8,
                                          ptr %outl, ptr %outd, ptr %outn)
  %bsin = add nuw i64 %bsi, 1
  %bsm = icmp ult i64 %bsin, 100
  br i1 %bsm, label %bsh, label %brdt
brdt:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %dtms = fmul double %dt, 1000.0
  %lt = fcmp olt double %dtms, %bmin
  %bminn = select i1 %lt, double %dtms, double %bmin
  %bsumn = fadd double %bsum, %dtms
  br label %brc
brc:
  %repn = add nuw i64 %rep, 1
  br label %brh
brdone:
  %mean = fdiv double %bsum, 16.0
  %bpr = call i32 (ptr, ...) @printf(ptr @m.bh, double %bmin, double %mean)
  br label %cleanup

cleanup:
  call void @universe_ml_ivf_destroy(ptr %h)
  call void @free(ptr %centers)
  call void @free(ptr %corpus)
  call void @free(ptr %queries)
  %rcs = call i32 @ut_summary()
  ret i32 %rcs
}
