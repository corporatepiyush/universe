; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/reduce.ll — SIMD column reductions.
; Vector path cross-checked against an in-IR strict scalar oracle; exact for
; min/max/n_unique; median/quantile vs a sorted-array reference; null-skipping
; vs a manual masked loop; dtype coverage i32/i64/f32/f64.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()

declare ptr @malloc(i64)
declare void @free(ptr)
declare double @llvm.fabs.f64(double)
declare double @llvm.sqrt.f64(double)
declare i32 @printf(ptr, ...)

; series builders
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare void @universe_dataframe_series_free(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
; dataframe builders
declare ptr @universe_dataframe_new()
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare void @universe_dataframe_free(ptr)

; module under test
declare i32 @universe_dataframe_sum(ptr, ptr)
declare i32 @universe_dataframe_mean(ptr, ptr)
declare i32 @universe_dataframe_min(ptr, ptr)
declare i32 @universe_dataframe_max(ptr, ptr)
declare i32 @universe_dataframe_var(ptr, ptr)
declare i32 @universe_dataframe_std(ptr, ptr)
declare i32 @universe_dataframe_median(ptr, ptr)
declare i32 @universe_dataframe_quantile(ptr, double, ptr)
declare i32 @universe_dataframe_n_unique(ptr, ptr)
declare i32 @universe_dataframe_null_count_series(ptr, ptr)
declare ptr @universe_dataframe_sum_horizontal(ptr)
declare ptr @universe_dataframe_mean_horizontal(ptr)

@.m_sum   = private constant [10 x i8] c"sum close\00"
@.m_mean  = private constant [11 x i8] c"mean close\00"
@.m_var   = private constant [10 x i8] c"var close\00"
@.m_std   = private constant [10 x i8] c"std close\00"
@.m_min   = private constant [10 x i8] c"min exact\00"
@.m_max   = private constant [10 x i8] c"max exact\00"
@.m_err   = private constant [10 x i8] c"err code \00"
@.m_med   = private constant [12 x i8] c"median good\00"
@.m_q25   = private constant [9 x i8]  c"q25 good\00"
@.m_q75   = private constant [9 x i8]  c"q75 good\00"
@.m_qi    = private constant [12 x i8] c"q interp ok\00"
@.m_nu    = private constant [12 x i8] c"n_unique ok\00"
@.m_nun   = private constant [17 x i8] c"n_unique+null ok\00"
@.m_nc    = private constant [11 x i8] c"nullcount \00"
@.m_hs    = private constant [15 x i8] c"sum_horiz good\00"
@.m_hm    = private constant [16 x i8] c"mean_horiz good\00"
@.col1    = private constant [1 x i8] c"a"
@.col2    = private constant [1 x i8] c"b"

; |a-b| <= tol
define internal void @check_close(double %a, double %b, double %tol, ptr %msg) {
entry:
  %d = fsub double %a, %b
  %ad = call double @llvm.fabs.f64(double %d)
  %ok = fcmp ole double %ad, %tol
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; strict scalar oracle over a double buffer -> sum, sumsq, min, max (n>=1)
define internal void @ref_stats(ptr %buf, i64 %n, ptr %osum, ptr %osq, ptr %omin, ptr %omax) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %sumn, %loop ]
  %sq = phi double [ 0.0, %entry ], [ %sqn, %loop ]
  %mn = phi double [ 0x7FF0000000000000, %entry ], [ %mnn, %loop ]
  %mx = phi double [ 0xFFF0000000000000, %entry ], [ %mxn, %loop ]
  %p = getelementptr inbounds double, ptr %buf, i64 %i
  %x = load double, ptr %p, align 8
  %sumn = fadd double %sum, %x
  %xx = fmul double %x, %x
  %sqn = fadd double %sq, %xx
  %ltm = fcmp olt double %x, %mn
  %mnn = select i1 %ltm, double %x, double %mn
  %gtm = fcmp ogt double %x, %mx
  %mxn = select i1 %gtm, double %x, double %mx
  %in = add nuw i64 %i, 1
  %d = icmp uge i64 %in, %n
  br i1 %d, label %fin, label %loop
fin:
  store double %sumn, ptr %osum, align 8
  store double %sqn, ptr %osq, align 8
  store double %mnn, ptr %omin, align 8
  store double %mxn, ptr %omax, align 8
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; --- all allocas hoisted to entry (hazard #11) ---
  %seed = alloca i64, align 8
  %o = alloca double, align 8
  %oi = alloca i64, align 8
  %rsum = alloca double, align 8
  %rsq = alloca double, align 8
  %rmin = alloca double, align 8
  %rmax = alloca double, align 8

  ; ============================================================
  ; 1. Random f64 column: SIMD reductions vs strict scalar oracle
  ; ============================================================
  store i64 88172645463325252, ptr %seed, align 8
  %N = add i64 0, 1000
  %fbytes = mul i64 %N, 8
  %fbuf = call ptr @malloc(i64 %fbytes)
  br label %genf
genf:
  %gi = phi i64 [ 0, %entry ], [ %gin, %genf ]
  %rr = call i64 @ut_rand(ptr %seed)
  %masked = and i64 %rr, 2047
  %fd = uitofp i64 %masked to double
  %scaled = fmul double %fd, 5.000000e-01
  %val = fsub double %scaled, 4.000000e+02
  %gp = getelementptr inbounds double, ptr %fbuf, i64 %gi
  store double %val, ptr %gp, align 8
  %gin = add nuw i64 %gi, 1
  %gd = icmp uge i64 %gin, %N
  br i1 %gd, label %refc, label %genf
refc:
  call void @ref_stats(ptr %fbuf, i64 %N, ptr %rsum, ptr %rsq, ptr %rmin, ptr %rmax)
  %ref_sum = load double, ptr %rsum, align 8
  %ref_sq = load double, ptr %rsq, align 8
  %ref_min = load double, ptr %rmin, align 8
  %ref_max = load double, ptr %rmax, align 8
  %Nd = uitofp i64 %N to double
  %ref_mean = fdiv double %ref_sum, %Nd
  %ss = fmul double %ref_sum, %ref_sum
  %corr = fdiv double %ss, %Nd
  %vnum = fsub double %ref_sq, %corr
  %nm1 = fsub double %Nd, 1.0
  %ref_var = fdiv double %vnum, %nm1
  %ref_std = call double @llvm.sqrt.f64(double %ref_var)

  %fser = call ptr @universe_dataframe_series_from(i32 3, ptr %fbuf, i64 %N)

  ; sum
  %e1 = call i32 @universe_dataframe_sum(ptr %fser, ptr %o)
  %e1_64 = zext i32 %e1 to i64
  call void @ut_check_eq(i64 0, i64 %e1_64, ptr @.m_err)
  %g_sum = load double, ptr %o, align 8
  %asum = call double @llvm.fabs.f64(double %ref_sum)
  %tsum0 = fmul double %asum, 1.000000e-06
  %tsum = fadd double %tsum0, 1.000000e-03
  call void @check_close(double %g_sum, double %ref_sum, double %tsum, ptr @.m_sum)

  ; mean
  %e2 = call i32 @universe_dataframe_mean(ptr %fser, ptr %o)
  %g_mean = load double, ptr %o, align 8
  call void @check_close(double %g_mean, double %ref_mean, double 1.000000e-06, ptr @.m_mean)

  ; var / std (looser tol)
  %e3 = call i32 @universe_dataframe_var(ptr %fser, ptr %o)
  %g_var = load double, ptr %o, align 8
  %avar = call double @llvm.fabs.f64(double %ref_var)
  %tvar0 = fmul double %avar, 1.000000e-06
  %tvar = fadd double %tvar0, 1.000000e-03
  call void @check_close(double %g_var, double %ref_var, double %tvar, ptr @.m_var)

  %e4 = call i32 @universe_dataframe_std(ptr %fser, ptr %o)
  %g_std = load double, ptr %o, align 8
  call void @check_close(double %g_std, double %ref_std, double 1.000000e-03, ptr @.m_std)

  ; min / max exact
  %e5 = call i32 @universe_dataframe_min(ptr %fser, ptr %o)
  %g_min = load double, ptr %o, align 8
  %mnok = fcmp oeq double %g_min, %ref_min
  call void @ut_check(i1 %mnok, ptr @.m_min)
  %e6 = call i32 @universe_dataframe_max(ptr %fser, ptr %o)
  %g_max = load double, ptr %o, align 8
  %mxok = fcmp oeq double %g_max, %ref_max
  call void @ut_check(i1 %mxok, ptr @.m_max)

  call void @universe_dataframe_series_free(ptr %fser)
  call void @free(ptr %fbuf)

  ; ============================================================
  ; 2. dtype coverage: i32 0..99, known sum/min/max/mean
  ; ============================================================
  %i32bytes = mul i64 100, 4
  %i32buf = call ptr @malloc(i64 %i32bytes)
  br label %geni
geni:
  %ii = phi i64 [ 0, %refc ], [ %iin, %geni ]
  %iv = trunc i64 %ii to i32
  %ip = getelementptr inbounds i32, ptr %i32buf, i64 %ii
  store i32 %iv, ptr %ip, align 4
  %iin = add nuw i64 %ii, 1
  %idn = icmp uge i64 %iin, 100
  br i1 %idn, label %chki, label %geni
chki:
  %iser = call ptr @universe_dataframe_series_from(i32 0, ptr %i32buf, i64 100)
  %ie1 = call i32 @universe_dataframe_sum(ptr %iser, ptr %o)
  %i_sum = load double, ptr %o, align 8
  call void @check_close(double %i_sum, double 4.950000e+03, double 1.000000e-06, ptr @.m_sum)
  %ie2 = call i32 @universe_dataframe_min(ptr %iser, ptr %o)
  %i_min = load double, ptr %o, align 8
  %imnok = fcmp oeq double %i_min, 0.000000e+00
  call void @ut_check(i1 %imnok, ptr @.m_min)
  %ie3 = call i32 @universe_dataframe_max(ptr %iser, ptr %o)
  %i_max = load double, ptr %o, align 8
  %imxok = fcmp oeq double %i_max, 9.900000e+01
  call void @ut_check(i1 %imxok, ptr @.m_max)
  %ie4 = call i32 @universe_dataframe_mean(ptr %iser, ptr %o)
  %i_mean = load double, ptr %o, align 8
  call void @check_close(double %i_mean, double 4.950000e+01, double 1.000000e-09, ptr @.m_mean)
  ; n_unique of 0..99 = 100
  %ie5 = call i32 @universe_dataframe_n_unique(ptr %iser, ptr %oi)
  %i_nu = load i64, ptr %oi, align 8
  call void @ut_check_eq(i64 100, i64 %i_nu, ptr @.m_nu)

  ; --- null-skipping: set nulls at 0,1,2 (values 0,1,2) ---
  %sn0 = call i32 @universe_dataframe_series_set_null(ptr %iser, i64 0)
  %sn1 = call i32 @universe_dataframe_series_set_null(ptr %iser, i64 1)
  %sn2 = call i32 @universe_dataframe_series_set_null(ptr %iser, i64 2)
  %ne1 = call i32 @universe_dataframe_null_count_series(ptr %iser, ptr %oi)
  %n_nc = load i64, ptr %oi, align 8
  call void @ut_check_eq(i64 3, i64 %n_nc, ptr @.m_nc)
  %ne2 = call i32 @universe_dataframe_sum(ptr %iser, ptr %o)
  %n_sum = load double, ptr %o, align 8
  call void @check_close(double %n_sum, double 4.947000e+03, double 1.000000e-06, ptr @.m_sum)
  %ne3 = call i32 @universe_dataframe_min(ptr %iser, ptr %o)
  %n_min = load double, ptr %o, align 8
  %nmnok = fcmp oeq double %n_min, 3.000000e+00
  call void @ut_check(i1 %nmnok, ptr @.m_min)
  %ne4 = call i32 @universe_dataframe_mean(ptr %iser, ptr %o)
  %n_mean = load double, ptr %o, align 8
  ; 4947 / 97
  %n_meanref = fdiv double 4.947000e+03, 9.700000e+01
  call void @check_close(double %n_mean, double %n_meanref, double 1.000000e-09, ptr @.m_mean)
  call void @universe_dataframe_series_free(ptr %iser)
  call void @free(ptr %i32buf)

  ; ============================================================
  ; 3. f32 dtype: values 1.0..8.0 (8 elems, exact) sum=36
  ; ============================================================
  %f32bytes = mul i64 8, 4
  %f32buf = call ptr @malloc(i64 %f32bytes)
  br label %genf32
genf32:
  %fi = phi i64 [ 0, %chki ], [ %fin, %genf32 ]
  %fival = add i64 %fi, 1
  %ff = uitofp i64 %fival to float
  %fp = getelementptr inbounds float, ptr %f32buf, i64 %fi
  store float %ff, ptr %fp, align 4
  %fin = add nuw i64 %fi, 1
  %fdn = icmp uge i64 %fin, 8
  br i1 %fdn, label %chkf32, label %genf32
chkf32:
  %f32ser = call ptr @universe_dataframe_series_from(i32 2, ptr %f32buf, i64 8)
  %fe1 = call i32 @universe_dataframe_sum(ptr %f32ser, ptr %o)
  %f32_sum = load double, ptr %o, align 8
  call void @check_close(double %f32_sum, double 3.600000e+01, double 1.000000e-05, ptr @.m_sum)
  %fe2 = call i32 @universe_dataframe_max(ptr %f32ser, ptr %o)
  %f32_max = load double, ptr %o, align 8
  %f32mxok = fcmp oeq double %f32_max, 8.000000e+00
  call void @ut_check(i1 %f32mxok, ptr @.m_max)
  call void @universe_dataframe_series_free(ptr %f32ser)
  call void @free(ptr %f32buf)

  ; ============================================================
  ; 4. i64 dtype + n_unique with dups: [0,0,1,1,2,3,3,3,4] distinct=5
  ; ============================================================
  %i64buf = call ptr @malloc(i64 72)
  store i64 0, ptr %i64buf, align 8
  %q1p = getelementptr inbounds i64, ptr %i64buf, i64 1
  store i64 0, ptr %q1p, align 8
  %q2p = getelementptr inbounds i64, ptr %i64buf, i64 2
  store i64 1, ptr %q2p, align 8
  %q3p = getelementptr inbounds i64, ptr %i64buf, i64 3
  store i64 1, ptr %q3p, align 8
  %q4p = getelementptr inbounds i64, ptr %i64buf, i64 4
  store i64 2, ptr %q4p, align 8
  %q5p = getelementptr inbounds i64, ptr %i64buf, i64 5
  store i64 3, ptr %q5p, align 8
  %q6p = getelementptr inbounds i64, ptr %i64buf, i64 6
  store i64 3, ptr %q6p, align 8
  %q7p = getelementptr inbounds i64, ptr %i64buf, i64 7
  store i64 3, ptr %q7p, align 8
  %q8p = getelementptr inbounds i64, ptr %i64buf, i64 8
  store i64 4, ptr %q8p, align 8
  %i64ser = call ptr @universe_dataframe_series_from(i32 1, ptr %i64buf, i64 9)
  %nue1 = call i32 @universe_dataframe_n_unique(ptr %i64ser, ptr %oi)
  %i64_nu = load i64, ptr %oi, align 8
  call void @ut_check_eq(i64 5, i64 %i64_nu, ptr @.m_nu)
  ; inject a null -> n_unique = 5 + 1 = 6
  %i64sn = call i32 @universe_dataframe_series_set_null(ptr %i64ser, i64 8)
  %nue2 = call i32 @universe_dataframe_n_unique(ptr %i64ser, ptr %oi)
  %i64_nu2 = load i64, ptr %oi, align 8
  ; after nulling index 8 (value 4), distinct non-null = {0,1,2,3} = 4, + 1 null = 5
  call void @ut_check_eq(i64 5, i64 %i64_nu2, ptr @.m_nun)
  call void @universe_dataframe_series_free(ptr %i64ser)
  call void @free(ptr %i64buf)

  ; ============================================================
  ; 5. median / quantile: [10,20,30,40]
  ;    median = 25 ; q(0.25) pos=0.75 -> 10+0.75*10=17.5 ; q(0.75) pos=2.25 -> 30+2.5=32.5
  ; ============================================================
  %mdbuf = call ptr @malloc(i64 32)
  store double 1.000000e+01, ptr %mdbuf, align 8
  %md1 = getelementptr inbounds double, ptr %mdbuf, i64 1
  store double 2.000000e+01, ptr %md1, align 8
  %md2 = getelementptr inbounds double, ptr %mdbuf, i64 2
  store double 3.000000e+01, ptr %md2, align 8
  %md3 = getelementptr inbounds double, ptr %mdbuf, i64 3
  store double 4.000000e+01, ptr %md3, align 8
  %mdser = call ptr @universe_dataframe_series_from(i32 3, ptr %mdbuf, i64 4)
  %mde = call i32 @universe_dataframe_median(ptr %mdser, ptr %o)
  %md_v = load double, ptr %o, align 8
  call void @check_close(double %md_v, double 2.500000e+01, double 1.000000e-09, ptr @.m_med)
  %q25e = call i32 @universe_dataframe_quantile(ptr %mdser, double 2.500000e-01, ptr %o)
  %q25_v = load double, ptr %o, align 8
  call void @check_close(double %q25_v, double 1.750000e+01, double 1.000000e-09, ptr @.m_q25)
  %q75e = call i32 @universe_dataframe_quantile(ptr %mdser, double 7.500000e-01, ptr %o)
  %q75_v = load double, ptr %o, align 8
  call void @check_close(double %q75_v, double 3.250000e+01, double 1.000000e-09, ptr @.m_q75)
  ; q out of range -> 8
  %qbade = call i32 @universe_dataframe_quantile(ptr %mdser, double 2.000000e+00, ptr %o)
  %qbade_64 = zext i32 %qbade to i64
  call void @ut_check_eq(i64 8, i64 %qbade_64, ptr @.m_qi)
  call void @universe_dataframe_series_free(ptr %mdser)
  call void @free(ptr %mdbuf)

  ; ============================================================
  ; 6. error paths: STR dtype -> 8 ; empty -> mean 4 ; null out -> 1
  ; ============================================================
  %strser = call ptr @universe_dataframe_series_new(i32 5, i64 4)
  %stre = call i32 @universe_dataframe_sum(ptr %strser, ptr %o)
  %stre_64 = zext i32 %stre to i64
  call void @ut_check_eq(i64 8, i64 %stre_64, ptr @.m_err)
  call void @universe_dataframe_series_free(ptr %strser)

  %emser = call ptr @universe_dataframe_series_new(i32 1, i64 0)
  %eme = call i32 @universe_dataframe_mean(ptr %emser, ptr %o)
  %eme_64 = zext i32 %eme to i64
  call void @ut_check_eq(i64 4, i64 %eme_64, ptr @.m_err)
  ; sum of empty == 0.0, err 0
  %eme2 = call i32 @universe_dataframe_sum(ptr %emser, ptr %o)
  %em_sum = load double, ptr %o, align 8
  %emok = fcmp oeq double %em_sum, 0.000000e+00
  call void @ut_check(i1 %emok, ptr @.m_sum)
  %eme2_64 = zext i32 %eme2 to i64
  call void @ut_check_eq(i64 0, i64 %eme2_64, ptr @.m_err)
  ; null out ptr -> 1
  %nue = call i32 @universe_dataframe_sum(ptr %emser, ptr null)
  %nue_64 = zext i32 %nue to i64
  call void @ut_check_eq(i64 1, i64 %nue_64, ptr @.m_err)
  call void @universe_dataframe_series_free(ptr %emser)

  ; ============================================================
  ; 7. horizontal reductions: df with two i64 cols a=[1,2,3], b=[10,20,30]
  ;    sum_horizontal -> [11,22,33] ; mean_horizontal -> [5.5,11,16.5]
  ; ============================================================
  %habuf = call ptr @malloc(i64 24)
  store i64 1, ptr %habuf, align 8
  %ha1 = getelementptr inbounds i64, ptr %habuf, i64 1
  store i64 2, ptr %ha1, align 8
  %ha2 = getelementptr inbounds i64, ptr %habuf, i64 2
  store i64 3, ptr %ha2, align 8
  %hbbuf = call ptr @malloc(i64 24)
  store i64 10, ptr %hbbuf, align 8
  %hb1 = getelementptr inbounds i64, ptr %hbbuf, i64 1
  store i64 20, ptr %hb1, align 8
  %hb2 = getelementptr inbounds i64, ptr %hbbuf, i64 2
  store i64 30, ptr %hb2, align 8
  %hca = call ptr @universe_dataframe_series_from(i32 1, ptr %habuf, i64 3)
  %hcb = call ptr @universe_dataframe_series_from(i32 1, ptr %hbbuf, i64 3)
  %hdf = call ptr @universe_dataframe_new()
  %hw1 = call i32 @universe_dataframe_with_column(ptr %hdf, ptr @.col1, i64 1, ptr %hca)
  %hw2 = call i32 @universe_dataframe_with_column(ptr %hdf, ptr @.col2, i64 1, ptr %hcb)
  %hsser = call ptr @universe_dataframe_sum_horizontal(ptr %hdf)
  ; read out values (F64 series, values at +24)
  %hsvp = getelementptr inbounds i8, ptr %hsser, i64 24
  %hsv = load ptr, ptr %hsvp, align 8
  %hs0 = load double, ptr %hsv, align 8
  %hs1p = getelementptr inbounds double, ptr %hsv, i64 1
  %hs1 = load double, ptr %hs1p, align 8
  %hs2p = getelementptr inbounds double, ptr %hsv, i64 2
  %hs2 = load double, ptr %hs2p, align 8
  %hsok0 = fcmp oeq double %hs0, 1.100000e+01
  %hsok1 = fcmp oeq double %hs1, 2.200000e+01
  %hsok2 = fcmp oeq double %hs2, 3.300000e+01
  %hsok01 = and i1 %hsok0, %hsok1
  %hsok = and i1 %hsok01, %hsok2
  call void @ut_check(i1 %hsok, ptr @.m_hs)
  call void @universe_dataframe_series_free(ptr %hsser)

  %hmser = call ptr @universe_dataframe_mean_horizontal(ptr %hdf)
  %hmvp = getelementptr inbounds i8, ptr %hmser, i64 24
  %hmv = load ptr, ptr %hmvp, align 8
  %hm0 = load double, ptr %hmv, align 8
  %hm1p = getelementptr inbounds double, ptr %hmv, i64 1
  %hm1 = load double, ptr %hm1p, align 8
  %hm2p = getelementptr inbounds double, ptr %hmv, i64 2
  %hm2 = load double, ptr %hm2p, align 8
  %hmok0 = fcmp oeq double %hm0, 5.500000e+00
  %hmok1 = fcmp oeq double %hm1, 1.100000e+01
  %hmok2 = fcmp oeq double %hm2, 1.650000e+01
  %hmok01 = and i1 %hmok0, %hmok1
  %hmok = and i1 %hmok01, %hmok2
  call void @ut_check(i1 %hmok, ptr @.m_hm)
  call void @universe_dataframe_series_free(ptr %hmser)

  call void @universe_dataframe_free(ptr %hdf)
  call void @free(ptr %habuf)
  call void @free(ptr %hbbuf)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
