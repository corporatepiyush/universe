; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/groupby.ll — hash group_by + aggregation + partition_by.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()

declare ptr @malloc(i64)
declare void @free(ptr)

; frame API
declare ptr @universe_dataframe_new()
declare void @universe_dataframe_free(ptr)
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare ptr @universe_dataframe_series_values(ptr)
declare i32 @universe_dataframe_series_dtype(ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)

; groupby API under test
declare ptr @universe_dataframe_group_by(ptr, ptr, i64)
declare ptr @universe_dataframe_group_by_stable(ptr, ptr, i64)
declare i32 @universe_dataframe_group_by_agg(ptr, ptr, i64, ptr, ptr)
declare i32 @universe_dataframe_partition_by(ptr, ptr, i64, ptr, i64, ptr)
declare void @universe_dataframe_group_by_free(ptr)

@.k  = private constant [1 x i8] c"k"
@.v  = private constant [1 x i8] c"v"
@.k1 = private constant [2 x i8] c"k1"
@.k2 = private constant [2 x i8] c"k2"

; known frame: k=[1,2,1,3,2,1] v=[10,20,30,40,50,60]
@.kv = private constant [6 x i64] [i64 1, i64 2, i64 1, i64 3, i64 2, i64 1]
@.vv = private constant [6 x i64] [i64 10, i64 20, i64 30, i64 40, i64 50, i64 60]

; multi-key frame
@.mk1 = private constant [5 x i64] [i64 1, i64 1, i64 2, i64 2, i64 1]
@.mk2 = private constant [5 x i64] [i64 7, i64 7, i64 8, i64 7, i64 7]
@.mvv = private constant [5 x i64] [i64 1, i64 1, i64 1, i64 1, i64 1]

@.m.gbnull  = private constant [16 x i8] c"group_by null? \00"
@.m.gcount  = private constant [13 x i8] c"group count \00"
@.m.aggrc   = private constant [13 x i8] c"agg rc bad  \00"
@.m.kord    = private constant [16 x i8] c"key order wrong\00"
@.m.sum     = private constant [13 x i8] c"sum wrong   \00"
@.m.mean    = private constant [13 x i8] c"mean wrong  \00"
@.m.min     = private constant [13 x i8] c"min wrong   \00"
@.m.max     = private constant [13 x i8] c"max wrong   \00"
@.m.cnt     = private constant [13 x i8] c"count wrong \00"
@.m.nuq     = private constant [13 x i8] c"nuniq wrong \00"
@.m.owidth  = private constant [16 x i8] c"out width wrong\00"
@.m.mkg     = private constant [16 x i8] c"multikey groups\00"
@.m.pn      = private constant [13 x i8] c"part out_n  \00"
@.m.prc     = private constant [13 x i8] c"part rc bad \00"
@.m.ph      = private constant [16 x i8] c"part sub height\00"
@.m.pw      = private constant [16 x i8] c"part sub width \00"
@.m.pfull   = private constant [16 x i8] c"part full rc   \00"
@.m.eh      = private constant [16 x i8] c"empty out hgt  \00"
@.m.epn     = private constant [16 x i8] c"empty part n   \00"
@.m.xsum    = private constant [16 x i8] c"xcheck sum bad \00"
@.m.xcnt    = private constant [16 x i8] c"xcheck cnt bad \00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %kn = alloca [16 x i8], align 8
  %kn2 = alloca [32 x i8], align 8
  %aggcols = alloca [6 x i64], align 8
  %aggops = alloca [6 x i32], align 8
  %outp = alloca ptr, align 8
  %frames = alloca [8 x ptr], align 8
  %outn = alloca i64, align 8
  %rng = alloca i64, align 8

  ; ---- build known frame ----
  %kser = call ptr @universe_dataframe_series_from(i32 1, ptr @.kv, i64 6)
  %vser = call ptr @universe_dataframe_series_from(i32 1, ptr @.vv, i64 6)
  %df = call ptr @universe_dataframe_new()
  %r1 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.k, i64 1, ptr %kser)
  %r2 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.v, i64 1, ptr %vser)

  ; key_names = [{@.k, 1}]
  store ptr @.k, ptr %kn, align 8
  %knl = getelementptr inbounds i8, ptr %kn, i64 8
  store i64 1, ptr %knl, align 8

  ; ---- group_by ----
  %gb = call ptr @universe_dataframe_group_by(ptr %df, ptr %kn, i64 1)
  %gbnn = icmp ne ptr %gb, null
  call void @ut_check(i1 %gbnn, ptr @.m.gbnull)

  ; agg: sum,mean,min,max,count,n_unique on col v (index 1)
  %ac0 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 0
  store i64 1, ptr %ac0, align 8
  %ac1 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 1
  store i64 1, ptr %ac1, align 8
  %ac2 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 2
  store i64 1, ptr %ac2, align 8
  %ac3 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 3
  store i64 1, ptr %ac3, align 8
  %ac4 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 4
  store i64 1, ptr %ac4, align 8
  %ac5 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 5
  store i64 1, ptr %ac5, align 8
  %op0 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 0
  store i32 0, ptr %op0, align 4
  %op1 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 1
  store i32 1, ptr %op1, align 4
  %op2 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 2
  store i32 2, ptr %op2, align 4
  %op3 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 3
  store i32 3, ptr %op3, align 4
  %op4 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 4
  store i32 4, ptr %op4, align 4
  %op5 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 5
  store i32 5, ptr %op5, align 4

  %argc64 = zext i32 %argc to i64
  %arc = call i32 @universe_dataframe_group_by_agg(ptr %gb, ptr %aggcols, i64 6, ptr %aggops, ptr %outp)
  %arc0 = icmp eq i32 %arc, 0
  call void @ut_check(i1 %arc0, ptr @.m.aggrc)
  %out = load ptr, ptr %outp, align 8

  ; group count = out height = 3
  %oh = call i64 @universe_dataframe_height(ptr %out)
  call void @ut_check_eq(i64 %oh, i64 3, ptr @.m.gcount)
  ; out width = 1 key + 6 aggs = 7
  %ow = call i64 @universe_dataframe_width(ptr %out)
  call void @ut_check_eq(i64 %ow, i64 7, ptr @.m.owidth)

  ; column pointers
  %ck = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 0)
  %ckv = call ptr @universe_dataframe_series_values(ptr %ck)
  %csum = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 1)
  %csumv = call ptr @universe_dataframe_series_values(ptr %csum)
  %cmean = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 2)
  %cmeanv = call ptr @universe_dataframe_series_values(ptr %cmean)
  %cmin = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 3)
  %cminv = call ptr @universe_dataframe_series_values(ptr %cmin)
  %cmax = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 4)
  %cmaxv = call ptr @universe_dataframe_series_values(ptr %cmax)
  %ccnt = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 5)
  %ccntv = call ptr @universe_dataframe_series_values(ptr %ccnt)
  %cnuq = call ptr @universe_dataframe_select_at_idx(ptr %out, i64 6)
  %cnuqv = call ptr @universe_dataframe_series_values(ptr %cnuq)

  ; key order: 1,2,3
  %k0 = load i64, ptr %ckv, align 8
  %k0ok = icmp eq i64 %k0, 1
  call void @ut_check(i1 %k0ok, ptr @.m.kord)
  %k1p = getelementptr inbounds i64, ptr %ckv, i64 1
  %k1 = load i64, ptr %k1p, align 8
  %k1ok = icmp eq i64 %k1, 2
  call void @ut_check(i1 %k1ok, ptr @.m.kord)
  %k2p = getelementptr inbounds i64, ptr %ckv, i64 2
  %k2 = load i64, ptr %k2p, align 8
  %k2ok = icmp eq i64 %k2, 3
  call void @ut_check(i1 %k2ok, ptr @.m.kord)

  ; group0 (k=1): sum100 mean33.333 min10 max60 cnt3 nuq3
  %s0 = load double, ptr %csumv, align 8
  %s0ok = fcmp oeq double %s0, 1.000000e+02
  call void @ut_check(i1 %s0ok, ptr @.m.sum)
  %m0 = load double, ptr %cmeanv, align 8
  %m0exp = fdiv double 1.000000e+02, 3.000000e+00
  %m0d = fsub double %m0, %m0exp
  %m0lo = fcmp olt double %m0d, 1.000000e-06
  %m0hi = fcmp ogt double %m0d, -1.000000e-06
  %m0ok = and i1 %m0lo, %m0hi
  call void @ut_check(i1 %m0ok, ptr @.m.mean)
  %mn0 = load double, ptr %cminv, align 8
  %mn0ok = fcmp oeq double %mn0, 1.000000e+01
  call void @ut_check(i1 %mn0ok, ptr @.m.min)
  %mx0 = load double, ptr %cmaxv, align 8
  %mx0ok = fcmp oeq double %mx0, 6.000000e+01
  call void @ut_check(i1 %mx0ok, ptr @.m.max)
  %c0 = load i64, ptr %ccntv, align 8
  call void @ut_check_eq(i64 %c0, i64 3, ptr @.m.cnt)
  %u0 = load i64, ptr %cnuqv, align 8
  call void @ut_check_eq(i64 %u0, i64 3, ptr @.m.nuq)

  ; group1 (k=2): sum70 min20 max50 cnt2 nuq2
  %s1p = getelementptr inbounds double, ptr %csumv, i64 1
  %s1 = load double, ptr %s1p, align 8
  %s1ok = fcmp oeq double %s1, 7.000000e+01
  call void @ut_check(i1 %s1ok, ptr @.m.sum)
  %c1p = getelementptr inbounds i64, ptr %ccntv, i64 1
  %c1v = load i64, ptr %c1p, align 8
  call void @ut_check_eq(i64 %c1v, i64 2, ptr @.m.cnt)
  %mn1p = getelementptr inbounds double, ptr %cminv, i64 1
  %mn1 = load double, ptr %mn1p, align 8
  %mn1ok = fcmp oeq double %mn1, 2.000000e+01
  call void @ut_check(i1 %mn1ok, ptr @.m.min)
  %mx1p = getelementptr inbounds double, ptr %cmaxv, i64 1
  %mx1 = load double, ptr %mx1p, align 8
  %mx1ok = fcmp oeq double %mx1, 5.000000e+01
  call void @ut_check(i1 %mx1ok, ptr @.m.max)

  ; group2 (k=3): sum40 cnt1 nuq1
  %s2p = getelementptr inbounds double, ptr %csumv, i64 2
  %s2 = load double, ptr %s2p, align 8
  %s2ok = fcmp oeq double %s2, 4.000000e+01
  call void @ut_check(i1 %s2ok, ptr @.m.sum)
  %c2p = getelementptr inbounds i64, ptr %ccntv, i64 2
  %c2v = load i64, ptr %c2p, align 8
  call void @ut_check_eq(i64 %c2v, i64 1, ptr @.m.cnt)
  %u2p = getelementptr inbounds i64, ptr %cnuqv, i64 2
  %u2 = load i64, ptr %u2p, align 8
  call void @ut_check_eq(i64 %u2, i64 1, ptr @.m.nuq)

  call void @universe_dataframe_free(ptr %out)
  call void @universe_dataframe_group_by_free(ptr %gb)

  ; ---- stable ordering (same core) ----
  %gbs = call ptr @universe_dataframe_group_by_stable(ptr %df, ptr %kn, i64 1)
  %sarc = call i32 @universe_dataframe_group_by_agg(ptr %gbs, ptr %aggcols, i64 1, ptr %aggops, ptr %outp)
  %souts = load ptr, ptr %outp, align 8
  %sck = call ptr @universe_dataframe_select_at_idx(ptr %souts, i64 0)
  %sckv = call ptr @universe_dataframe_series_values(ptr %sck)
  %sk0 = load i64, ptr %sckv, align 8
  %sk0ok = icmp eq i64 %sk0, 1
  call void @ut_check(i1 %sk0ok, ptr @.m.kord)
  call void @universe_dataframe_free(ptr %souts)
  call void @universe_dataframe_group_by_free(ptr %gbs)

  ; ---- partition_by ----
  %parc = call i32 @universe_dataframe_partition_by(ptr %df, ptr %kn, i64 1, ptr %frames, i64 8, ptr %outn)
  %parc0 = icmp eq i32 %parc, 0
  call void @ut_check(i1 %parc0, ptr @.m.prc)
  %pn = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %pn, i64 3, ptr @.m.pn)
  ; group0 has 3 rows, group1 2, group2 1; each width 2
  %pf0p = getelementptr inbounds [8 x ptr], ptr %frames, i64 0, i64 0
  %pf0 = load ptr, ptr %pf0p, align 8
  %pf0h = call i64 @universe_dataframe_height(ptr %pf0)
  call void @ut_check_eq(i64 %pf0h, i64 3, ptr @.m.ph)
  %pf0w = call i64 @universe_dataframe_width(ptr %pf0)
  call void @ut_check_eq(i64 %pf0w, i64 2, ptr @.m.pw)
  %pf1p = getelementptr inbounds [8 x ptr], ptr %frames, i64 0, i64 1
  %pf1 = load ptr, ptr %pf1p, align 8
  %pf1h = call i64 @universe_dataframe_height(ptr %pf1)
  call void @ut_check_eq(i64 %pf1h, i64 2, ptr @.m.ph)
  %pf2p = getelementptr inbounds [8 x ptr], ptr %frames, i64 0, i64 2
  %pf2 = load ptr, ptr %pf2p, align 8
  %pf2h = call i64 @universe_dataframe_height(ptr %pf2)
  call void @ut_check_eq(i64 %pf2h, i64 1, ptr @.m.ph)
  call void @universe_dataframe_free(ptr %pf0)
  call void @universe_dataframe_free(ptr %pf1)
  call void @universe_dataframe_free(ptr %pf2)

  ; partition full: cap too small -> rc 6, out_n set
  %pfull = call i32 @universe_dataframe_partition_by(ptr %df, ptr %kn, i64 1, ptr %frames, i64 1, ptr %outn)
  %pfull6 = icmp eq i32 %pfull, 6
  call void @ut_check(i1 %pfull6, ptr @.m.pfull)

  call void @universe_dataframe_free(ptr %df)

  ; ---- multi-key grouping ----
  %mk1s = call ptr @universe_dataframe_series_from(i32 1, ptr @.mk1, i64 5)
  %mk2s = call ptr @universe_dataframe_series_from(i32 1, ptr @.mk2, i64 5)
  %mvs = call ptr @universe_dataframe_series_from(i32 1, ptr @.mvv, i64 5)
  %mdf = call ptr @universe_dataframe_new()
  %mr1 = call i32 @universe_dataframe_with_column(ptr %mdf, ptr @.k1, i64 2, ptr %mk1s)
  %mr2 = call i32 @universe_dataframe_with_column(ptr %mdf, ptr @.k2, i64 2, ptr %mk2s)
  %mr3 = call i32 @universe_dataframe_with_column(ptr %mdf, ptr @.v, i64 1, ptr %mvs)
  store ptr @.k1, ptr %kn2, align 8
  %kn2l = getelementptr inbounds i8, ptr %kn2, i64 8
  store i64 2, ptr %kn2l, align 8
  %kn2b = getelementptr inbounds i8, ptr %kn2, i64 16
  store ptr @.k2, ptr %kn2b, align 8
  %kn2bl = getelementptr inbounds i8, ptr %kn2, i64 24
  store i64 2, ptr %kn2bl, align 8
  %mgb = call ptr @universe_dataframe_group_by(ptr %mdf, ptr %kn2, i64 2)
  ; agg count on v (col 2)
  %mac0 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 0
  store i64 2, ptr %mac0, align 8
  %mop0 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 0
  store i32 4, ptr %mop0, align 4
  %marc = call i32 @universe_dataframe_group_by_agg(ptr %mgb, ptr %aggcols, i64 1, ptr %aggops, ptr %outp)
  %mout = load ptr, ptr %outp, align 8
  %mgc = call i64 @universe_dataframe_height(ptr %mout)
  call void @ut_check_eq(i64 %mgc, i64 3, ptr @.m.mkg)
  ; width = 2 keys + 1 agg = 3
  %mow = call i64 @universe_dataframe_width(ptr %mout)
  call void @ut_check_eq(i64 %mow, i64 3, ptr @.m.owidth)
  call void @universe_dataframe_free(ptr %mout)
  call void @universe_dataframe_group_by_free(ptr %mgb)
  call void @universe_dataframe_free(ptr %mdf)

  ; ---- empty df ----
  %eser = call ptr @universe_dataframe_series_new(i32 1, i64 0)
  %edf = call ptr @universe_dataframe_new()
  %er1 = call i32 @universe_dataframe_with_column(ptr %edf, ptr @.k, i64 1, ptr %eser)
  %egb = call ptr @universe_dataframe_group_by(ptr %edf, ptr %kn, i64 1)
  %eac0 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 0
  store i64 0, ptr %eac0, align 8
  %eop0 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 0
  store i32 4, ptr %eop0, align 4
  %earc = call i32 @universe_dataframe_group_by_agg(ptr %egb, ptr %aggcols, i64 1, ptr %aggops, ptr %outp)
  %eout = load ptr, ptr %outp, align 8
  %eoh = call i64 @universe_dataframe_height(ptr %eout)
  call void @ut_check_eq(i64 %eoh, i64 0, ptr @.m.eh)
  call void @universe_dataframe_free(ptr %eout)
  call void @universe_dataframe_group_by_free(ptr %egb)
  %eparc = call i32 @universe_dataframe_partition_by(ptr %edf, ptr %kn, i64 1, ptr %frames, i64 8, ptr %outn)
  %epn = load i64, ptr %outn, align 8
  call void @ut_check_eq(i64 %epn, i64 0, ptr @.m.epn)
  call void @universe_dataframe_free(ptr %edf)

  ; ---- randomized cross-check vs brute force ----
  ; N=500 rows, keys in [0,8), values in [0,1024)
  store i64 88172645463325252, ptr %rng, align 8
  %kbuf = call ptr @malloc(i64 4000)
  %vbuf = call ptr @malloc(i64 4000)
  %bsum = call ptr @malloc(i64 64)
  %bcnt = call ptr @malloc(i64 64)
  br label %bz.head

bz.head:
  %bzi = phi i64 [ 0, %entry ], [ %bzin, %bz.body ]
  %bzc = icmp ult i64 %bzi, 8
  br i1 %bzc, label %bz.body, label %fill.head

bz.body:
  %bsp = getelementptr inbounds i64, ptr %bsum, i64 %bzi
  store i64 0, ptr %bsp, align 8
  %bcp = getelementptr inbounds i64, ptr %bcnt, i64 %bzi
  store i64 0, ptr %bcp, align 8
  %bzin = add i64 %bzi, 1
  br label %bz.head

fill.head:
  %fi = phi i64 [ 0, %bz.head ], [ %fin, %fill.body ]
  %fc = icmp ult i64 %fi, 500
  br i1 %fc, label %fill.body, label %rbuild

fill.body:
  %rr = call i64 @ut_rand(ptr %rng)
  %kval = and i64 %rr, 7
  %rr2 = lshr i64 %rr, 8
  %vval = and i64 %rr2, 1023
  %kbp = getelementptr inbounds i64, ptr %kbuf, i64 %fi
  store i64 %kval, ptr %kbp, align 8
  %vbp = getelementptr inbounds i64, ptr %vbuf, i64 %fi
  store i64 %vval, ptr %vbp, align 8
  ; brute accumulate
  %absp = getelementptr inbounds i64, ptr %bsum, i64 %kval
  %abs = load i64, ptr %absp, align 8
  %abs2 = add i64 %abs, %vval
  store i64 %abs2, ptr %absp, align 8
  %abcp = getelementptr inbounds i64, ptr %bcnt, i64 %kval
  %abc = load i64, ptr %abcp, align 8
  %abc2 = add i64 %abc, 1
  store i64 %abc2, ptr %abcp, align 8
  %fin = add i64 %fi, 1
  br label %fill.head

rbuild:
  %rkser = call ptr @universe_dataframe_series_from(i32 1, ptr %kbuf, i64 500)
  %rvser = call ptr @universe_dataframe_series_from(i32 1, ptr %vbuf, i64 500)
  %rdf = call ptr @universe_dataframe_new()
  %rr1 = call i32 @universe_dataframe_with_column(ptr %rdf, ptr @.k, i64 1, ptr %rkser)
  %rr2c = call i32 @universe_dataframe_with_column(ptr %rdf, ptr @.v, i64 1, ptr %rvser)
  %rgb = call ptr @universe_dataframe_group_by(ptr %rdf, ptr %kn, i64 1)
  ; agg sum(v col1), count(v col1)
  %rac0 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 0
  store i64 1, ptr %rac0, align 8
  %rac1 = getelementptr inbounds [6 x i64], ptr %aggcols, i64 0, i64 1
  store i64 1, ptr %rac1, align 8
  %rop0 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 0
  store i32 0, ptr %rop0, align 4
  %rop1 = getelementptr inbounds [6 x i32], ptr %aggops, i64 0, i64 1
  store i32 4, ptr %rop1, align 4
  %rarc = call i32 @universe_dataframe_group_by_agg(ptr %rgb, ptr %aggcols, i64 2, ptr %aggops, ptr %outp)
  %rout = load ptr, ptr %outp, align 8
  %rg = call i64 @universe_dataframe_height(ptr %rout)
  %rkey = call ptr @universe_dataframe_select_at_idx(ptr %rout, i64 0)
  %rkeyv = call ptr @universe_dataframe_series_values(ptr %rkey)
  %rsumc = call ptr @universe_dataframe_select_at_idx(ptr %rout, i64 1)
  %rsumv = call ptr @universe_dataframe_series_values(ptr %rsumc)
  %rcntc = call ptr @universe_dataframe_select_at_idx(ptr %rout, i64 2)
  %rcntv = call ptr @universe_dataframe_series_values(ptr %rcntc)
  br label %xchk.head

xchk.head:
  %xi = phi i64 [ 0, %rbuild ], [ %xin, %xchk.body ]
  %xsv = phi i64 [ 0, %rbuild ], [ %xsv2, %xchk.body ]
  %xcv = phi i64 [ 0, %rbuild ], [ %xcv2, %xchk.body ]
  %xc = icmp ult i64 %xi, %rg
  br i1 %xc, label %xchk.body, label %xchk.done

xchk.body:
  %xkp = getelementptr inbounds i64, ptr %rkeyv, i64 %xi
  %xk = load i64, ptr %xkp, align 8
  %xsp = getelementptr inbounds double, ptr %rsumv, i64 %xi
  %xs = load double, ptr %xsp, align 8
  %xcp = getelementptr inbounds i64, ptr %rcntv, i64 %xi
  %xcnt = load i64, ptr %xcp, align 8
  ; brute lookup
  %bxsp = getelementptr inbounds i64, ptr %bsum, i64 %xk
  %bxs = load i64, ptr %bxsp, align 8
  %bxsf = sitofp i64 %bxs to double
  %sumbad = fcmp one double %xs, %bxsf
  %sumbadz = zext i1 %sumbad to i64
  %xsv2 = add i64 %xsv, %sumbadz
  %bxcp = getelementptr inbounds i64, ptr %bcnt, i64 %xk
  %bxc = load i64, ptr %bxcp, align 8
  %cntbad = icmp ne i64 %xcnt, %bxc
  %cntbadz = zext i1 %cntbad to i64
  %xcv2 = add i64 %xcv, %cntbadz
  %xin = add i64 %xi, 1
  br label %xchk.head

xchk.done:
  call void @ut_check_eq(i64 %xsv, i64 0, ptr @.m.xsum)
  call void @ut_check_eq(i64 %xcv, i64 0, ptr @.m.xcnt)
  call void @universe_dataframe_free(ptr %rout)
  call void @universe_dataframe_group_by_free(ptr %rgb)
  call void @universe_dataframe_free(ptr %rdf)
  call void @free(ptr %kbuf)
  call void @free(ptr %vbuf)
  call void @free(ptr %bsum)
  call void @free(ptr %bcnt)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
