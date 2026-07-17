; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/select.ll — filter / take / sample / drop_nulls /
; fill_null / unique / is_unique / is_duplicated.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)

declare void @free(ptr)

; frame primitives used to build inputs / inspect outputs
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare void @universe_dataframe_series_free(ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare ptr @universe_dataframe_series_values(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)
declare ptr @universe_dataframe_new()
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_column(ptr, ptr, i64)
declare void @universe_dataframe_free(ptr)

; select.ll under test
declare ptr @universe_dataframe_filter(ptr, ptr)
declare ptr @universe_dataframe_take(ptr, ptr, i64)
declare ptr @universe_dataframe_sample_n(ptr, i64, i32, i64)
declare ptr @universe_dataframe_drop_nulls(ptr, ptr, i64)
declare i32 @universe_dataframe_fill_null(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_unique(ptr, ptr, i64, i32)
declare ptr @universe_dataframe_unique_stable(ptr, ptr, i64)
declare i32 @universe_dataframe_is_unique(ptr, ptr)
declare i32 @universe_dataframe_is_duplicated(ptr, ptr)

@.col_a = private constant [1 x i8] c"a"
@.col_b = private constant [1 x i8] c"b"

@.m.fh    = private constant [16 x i8] c"filter height  \00"
@.m.fw    = private constant [16 x i8] c"filter width   \00"
@.m.fv    = private constant [16 x i8] c"filter value   \00"
@.m.fe    = private constant [16 x i8] c"filter empty h \00"
@.m.fvld  = private constant [16 x i8] c"filter validity\00"
@.m.th    = private constant [16 x i8] c"take height    \00"
@.m.tv    = private constant [16 x i8] c"take value     \00"
@.m.toob  = private constant [16 x i8] c"take OOB !null \00"
@.m.uh    = private constant [16 x i8] c"unique height  \00"
@.m.uv    = private constant [16 x i8] c"unique value   \00"
@.m.ush   = private constant [16 x i8] c"unique_stable h\00"
@.m.iuh   = private constant [16 x i8] c"is_unique rc   \00"
@.m.iuv   = private constant [16 x i8] c"is_unique mask \00"
@.m.idv   = private constant [16 x i8] c"is_dup mask    \00"
@.m.snh   = private constant [16 x i8] c"sample_n height\00"
@.m.snd   = private constant [16 x i8] c"sample determ  \00"
@.m.snp   = private constant [16 x i8] c"sample perm sum\00"
@.m.dnh   = private constant [16 x i8] c"drop_nulls hgt \00"
@.m.dnv   = private constant [16 x i8] c"drop_nulls val \00"
@.m.fnr   = private constant [16 x i8] c"fill_null rc   \00"
@.m.fnv   = private constant [16 x i8] c"fill_null value\00"
@.m.fnc   = private constant [16 x i8] c"fill_null count\00"
@.m.fnn   = private constant [16 x i8] c"fill_null unnul\00"
@.m.fuzz  = private constant [16 x i8] c"fuzz violations\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %av6 = alloca [6 x i64], align 8       ; col "a"
  %bv6 = alloca [6 x i64], align 8       ; col "b"
  %idx3 = alloca [3 x i64], align 8
  %idxoob = alloca [1 x i64], align 8
  %sub1 = alloca [2 x i64], align 8      ; {ptr,len} subset for "a"
  %fillv = alloca i64, align 8
  %isn = alloca i8, align 1
  %rngst = alloca i64, align 8
  %av5 = alloca [5 x i64], align 8
  %av5b = alloca [5 x i64], align 8

  ; a = [10,20,30,20,10,40]
  %a0 = getelementptr [6 x i64], ptr %av6, i64 0, i64 0
  store i64 10, ptr %a0, align 8
  %a1 = getelementptr [6 x i64], ptr %av6, i64 0, i64 1
  store i64 20, ptr %a1, align 8
  %a2 = getelementptr [6 x i64], ptr %av6, i64 0, i64 2
  store i64 30, ptr %a2, align 8
  %a3 = getelementptr [6 x i64], ptr %av6, i64 0, i64 3
  store i64 20, ptr %a3, align 8
  %a4 = getelementptr [6 x i64], ptr %av6, i64 0, i64 4
  store i64 10, ptr %a4, align 8
  %a5 = getelementptr [6 x i64], ptr %av6, i64 0, i64 5
  store i64 40, ptr %a5, align 8
  ; b = [1,2,3,2,1,4]
  %b0 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 0
  store i64 1, ptr %b0, align 8
  %b1 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 1
  store i64 2, ptr %b1, align 8
  %b2 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 2
  store i64 3, ptr %b2, align 8
  %b3 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 3
  store i64 2, ptr %b3, align 8
  %b4 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 4
  store i64 1, ptr %b4, align 8
  %b5 = getelementptr [6 x i64], ptr %bv6, i64 0, i64 5
  store i64 4, ptr %b5, align 8

  %sa = call ptr @universe_dataframe_series_from(i32 1, ptr %av6, i64 6)
  %sb = call ptr @universe_dataframe_series_from(i32 1, ptr %bv6, i64 6)
  %df = call ptr @universe_dataframe_new()
  %w0 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_a, i64 1, ptr %sa)
  %w1 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_b, i64 1, ptr %sb)

  ; ================= filter: mask [1,0,1,0,1,1] -> rows 0,2,4,5 =============
  %mask = call ptr @universe_dataframe_series_new(i32 4, i64 6)
  %mv = call ptr @universe_dataframe_series_values(ptr %mask)
  %mv0 = getelementptr i8, ptr %mv, i64 0
  store i8 1, ptr %mv0, align 1
  %mv1 = getelementptr i8, ptr %mv, i64 1
  store i8 0, ptr %mv1, align 1
  %mv2 = getelementptr i8, ptr %mv, i64 2
  store i8 1, ptr %mv2, align 1
  %mv3 = getelementptr i8, ptr %mv, i64 3
  store i8 0, ptr %mv3, align 1
  %mv4 = getelementptr i8, ptr %mv, i64 4
  store i8 1, ptr %mv4, align 1
  %mv5 = getelementptr i8, ptr %mv, i64 5
  store i8 1, ptr %mv5, align 1

  %fout = call ptr @universe_dataframe_filter(ptr %df, ptr %mask)
  %fh = call i64 @universe_dataframe_height(ptr %fout)
  call void @ut_check_eq(i64 %fh, i64 4, ptr @.m.fh)
  %fw = call i64 @universe_dataframe_width(ptr %fout)
  call void @ut_check_eq(i64 %fw, i64 2, ptr @.m.fw)
  ; expected a = [10,30,10,40]
  %fca = call ptr @universe_dataframe_column(ptr %fout, ptr @.col_a, i64 1)
  %fcav = call ptr @universe_dataframe_series_values(ptr %fca)
  %fca0p = getelementptr i64, ptr %fcav, i64 0
  %fca0 = load i64, ptr %fca0p, align 8
  call void @ut_check_eq(i64 %fca0, i64 10, ptr @.m.fv)
  %fca1p = getelementptr i64, ptr %fcav, i64 1
  %fca1 = load i64, ptr %fca1p, align 8
  call void @ut_check_eq(i64 %fca1, i64 30, ptr @.m.fv)
  %fca2p = getelementptr i64, ptr %fcav, i64 2
  %fca2 = load i64, ptr %fca2p, align 8
  call void @ut_check_eq(i64 %fca2, i64 10, ptr @.m.fv)
  %fca3p = getelementptr i64, ptr %fcav, i64 3
  %fca3 = load i64, ptr %fca3p, align 8
  call void @ut_check_eq(i64 %fca3, i64 40, ptr @.m.fv)
  call void @universe_dataframe_free(ptr %fout)

  ; filter empty: mask all zero -> height 0
  store i8 0, ptr %mv0, align 1
  store i8 0, ptr %mv2, align 1
  store i8 0, ptr %mv4, align 1
  store i8 0, ptr %mv5, align 1
  %feout = call ptr @universe_dataframe_filter(ptr %df, ptr %mask)
  %feh = call i64 @universe_dataframe_height(ptr %feout)
  call void @ut_check_eq(i64 %feh, i64 0, ptr @.m.fe)
  call void @universe_dataframe_free(ptr %feout)
  call void @universe_dataframe_series_free(ptr %mask)

  ; ================= take: idx [5,0,2] -> a [40,10,30] =====================
  %i0 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 0
  store i64 5, ptr %i0, align 8
  %i1 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 1
  store i64 0, ptr %i1, align 8
  %i2 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 2
  store i64 2, ptr %i2, align 8
  %tout = call ptr @universe_dataframe_take(ptr %df, ptr %idx3, i64 3)
  %th = call i64 @universe_dataframe_height(ptr %tout)
  call void @ut_check_eq(i64 %th, i64 3, ptr @.m.th)
  %tca = call ptr @universe_dataframe_column(ptr %tout, ptr @.col_a, i64 1)
  %tcav = call ptr @universe_dataframe_series_values(ptr %tca)
  %t0p = getelementptr i64, ptr %tcav, i64 0
  %t0 = load i64, ptr %t0p, align 8
  call void @ut_check_eq(i64 %t0, i64 40, ptr @.m.tv)
  %t1p = getelementptr i64, ptr %tcav, i64 1
  %t1 = load i64, ptr %t1p, align 8
  call void @ut_check_eq(i64 %t1, i64 10, ptr @.m.tv)
  %t2p = getelementptr i64, ptr %tcav, i64 2
  %t2 = load i64, ptr %t2p, align 8
  call void @ut_check_eq(i64 %t2, i64 30, ptr @.m.tv)
  call void @universe_dataframe_free(ptr %tout)

  ; take OOB: idx [6] -> null
  %io0 = getelementptr [1 x i64], ptr %idxoob, i64 0, i64 0
  store i64 6, ptr %io0, align 8
  %toob = call ptr @universe_dataframe_take(ptr %df, ptr %idxoob, i64 1)
  %toobnull = icmp eq ptr %toob, null
  call void @ut_check(i1 %toobnull, ptr @.m.toob)

  ; ================= unique on subset "a" -> [10,20,30,40] ================
  %subp = getelementptr [2 x i64], ptr %sub1, i64 0, i64 0
  store ptr @.col_a, ptr %subp, align 8
  %sublp = getelementptr [2 x i64], ptr %sub1, i64 0, i64 1
  store i64 1, ptr %sublp, align 8
  %uout = call ptr @universe_dataframe_unique(ptr %df, ptr %sub1, i64 1, i32 1)
  %uh = call i64 @universe_dataframe_height(ptr %uout)
  call void @ut_check_eq(i64 %uh, i64 4, ptr @.m.uh)
  %uca = call ptr @universe_dataframe_column(ptr %uout, ptr @.col_a, i64 1)
  %ucav = call ptr @universe_dataframe_series_values(ptr %uca)
  %u0p = getelementptr i64, ptr %ucav, i64 0
  %u0 = load i64, ptr %u0p, align 8
  call void @ut_check_eq(i64 %u0, i64 10, ptr @.m.uv)
  %u1p = getelementptr i64, ptr %ucav, i64 1
  %u1 = load i64, ptr %u1p, align 8
  call void @ut_check_eq(i64 %u1, i64 20, ptr @.m.uv)
  %u2p = getelementptr i64, ptr %ucav, i64 2
  %u2 = load i64, ptr %u2p, align 8
  call void @ut_check_eq(i64 %u2, i64 30, ptr @.m.uv)
  %u3p = getelementptr i64, ptr %ucav, i64 3
  %u3 = load i64, ptr %u3p, align 8
  call void @ut_check_eq(i64 %u3, i64 40, ptr @.m.uv)
  call void @universe_dataframe_free(ptr %uout)

  ; unique over ALL columns (subset null) -> distinct (a,b) pairs = 4 rows
  %uaout = call ptr @universe_dataframe_unique(ptr %df, ptr null, i64 0, i32 1)
  %uah = call i64 @universe_dataframe_height(ptr %uaout)
  call void @ut_check_eq(i64 %uah, i64 4, ptr @.m.uh)
  call void @universe_dataframe_free(ptr %uaout)

  ; unique_stable subset "a"
  %usout = call ptr @universe_dataframe_unique_stable(ptr %df, ptr %sub1, i64 1)
  %ush = call i64 @universe_dataframe_height(ptr %usout)
  call void @ut_check_eq(i64 %ush, i64 4, ptr @.m.ush)
  call void @universe_dataframe_free(ptr %usout)

  ; ================= is_unique / is_duplicated (all cols) =================
  ; pairs (a,b): (10,1)x2 (20,2)x2 (30,3)x1 (40,4)x1
  ; is_unique  = [0,0,1,0,0,1] ; is_duplicated = [1,1,0,1,1,0]
  %umask = call ptr @universe_dataframe_series_new(i32 4, i64 6)
  %iurc = call i32 @universe_dataframe_is_unique(ptr %df, ptr %umask)
  %iurc64 = sext i32 %iurc to i64
  call void @ut_check_eq(i64 %iurc64, i64 0, ptr @.m.iuh)
  %umv = call ptr @universe_dataframe_series_values(ptr %umask)
  %um0p = getelementptr i8, ptr %umv, i64 0
  %um0 = load i8, ptr %um0p, align 1
  %um0z = zext i8 %um0 to i64
  call void @ut_check_eq(i64 %um0z, i64 0, ptr @.m.iuv)
  %um2p = getelementptr i8, ptr %umv, i64 2
  %um2 = load i8, ptr %um2p, align 1
  %um2z = zext i8 %um2 to i64
  call void @ut_check_eq(i64 %um2z, i64 1, ptr @.m.iuv)
  %um5p = getelementptr i8, ptr %umv, i64 5
  %um5 = load i8, ptr %um5p, align 1
  %um5z = zext i8 %um5 to i64
  call void @ut_check_eq(i64 %um5z, i64 1, ptr @.m.iuv)
  call void @universe_dataframe_series_free(ptr %umask)

  %dmask = call ptr @universe_dataframe_series_new(i32 4, i64 6)
  %idrc = call i32 @universe_dataframe_is_duplicated(ptr %df, ptr %dmask)
  %dmv = call ptr @universe_dataframe_series_values(ptr %dmask)
  %dm0p = getelementptr i8, ptr %dmv, i64 0
  %dm0 = load i8, ptr %dm0p, align 1
  %dm0z = zext i8 %dm0 to i64
  call void @ut_check_eq(i64 %dm0z, i64 1, ptr @.m.idv)
  %dm2p = getelementptr i8, ptr %dmv, i64 2
  %dm2 = load i8, ptr %dm2p, align 1
  %dm2z = zext i8 %dm2 to i64
  call void @ut_check_eq(i64 %dm2z, i64 0, ptr @.m.idv)
  call void @universe_dataframe_series_free(ptr %dmask)

  ; ================= sample_n =============================================
  ; with replacement, n=4, fixed seed -> deterministic
  %s1 = call ptr @universe_dataframe_sample_n(ptr %df, i64 4, i32 1, i64 987654321)
  %s1h = call i64 @universe_dataframe_height(ptr %s1)
  call void @ut_check_eq(i64 %s1h, i64 4, ptr @.m.snh)
  %s2 = call ptr @universe_dataframe_sample_n(ptr %df, i64 4, i32 1, i64 987654321)
  %s1ca = call ptr @universe_dataframe_column(ptr %s1, ptr @.col_a, i64 1)
  %s1cav = call ptr @universe_dataframe_series_values(ptr %s1ca)
  %s2ca = call ptr @universe_dataframe_column(ptr %s2, ptr @.col_a, i64 1)
  %s2cav = call ptr @universe_dataframe_series_values(ptr %s2ca)
  br label %det.loop
det.loop:
  %dk = phi i64 [ 0, %entry ], [ %dkn, %det.cont ]
  %dviol = phi i64 [ 0, %entry ], [ %dviol.n, %det.cont ]
  %dcmp = icmp ult i64 %dk, 4
  br i1 %dcmp, label %det.body, label %det.done
det.body:
  %d1p = getelementptr i64, ptr %s1cav, i64 %dk
  %d1 = load i64, ptr %d1p, align 8
  %d2p = getelementptr i64, ptr %s2cav, i64 %dk
  %d2 = load i64, ptr %d2p, align 8
  %dne = icmp ne i64 %d1, %d2
  %dinc = zext i1 %dne to i64
  %dviol.b = add i64 %dviol, %dinc
  br label %det.cont
det.cont:
  %dviol.n = phi i64 [ %dviol.b, %det.body ]
  %dkn = add i64 %dk, 1
  br label %det.loop
det.done:
  call void @ut_check_eq(i64 %dviol, i64 0, ptr @.m.snd)
  call void @universe_dataframe_free(ptr %s1)
  call void @universe_dataframe_free(ptr %s2)

  ; without replacement n=6 -> permutation: sum of "a" preserved (150)
  %sp = call ptr @universe_dataframe_sample_n(ptr %df, i64 6, i32 0, i64 42)
  %sph = call i64 @universe_dataframe_height(ptr %sp)
  call void @ut_check_eq(i64 %sph, i64 6, ptr @.m.snh)
  %spca = call ptr @universe_dataframe_column(ptr %sp, ptr @.col_a, i64 1)
  %spcav = call ptr @universe_dataframe_series_values(ptr %spca)
  br label %sum.loop
sum.loop:
  %sk = phi i64 [ 0, %det.done ], [ %skn, %sum.body ]
  %ssum = phi i64 [ 0, %det.done ], [ %ssum.n, %sum.body ]
  %scmp = icmp ult i64 %sk, 6
  br i1 %scmp, label %sum.body, label %sum.done
sum.body:
  %svp = getelementptr i64, ptr %spcav, i64 %sk
  %sv = load i64, ptr %svp, align 8
  %ssum.n = add i64 %ssum, %sv
  %skn = add i64 %sk, 1
  br label %sum.loop
sum.done:
  call void @ut_check_eq(i64 %ssum, i64 130, ptr @.m.snp)
  call void @universe_dataframe_free(ptr %sp)

  ; ================= drop_nulls ===========================================
  ; df2: single col "a" = [10,20,30,40,50] with null at idx 2
  %d2a0 = getelementptr [5 x i64], ptr %av5, i64 0, i64 0
  store i64 10, ptr %d2a0, align 8
  %d2a1 = getelementptr [5 x i64], ptr %av5, i64 0, i64 1
  store i64 20, ptr %d2a1, align 8
  %d2a2 = getelementptr [5 x i64], ptr %av5, i64 0, i64 2
  store i64 30, ptr %d2a2, align 8
  %d2a3 = getelementptr [5 x i64], ptr %av5, i64 0, i64 3
  store i64 40, ptr %d2a3, align 8
  %d2a4 = getelementptr [5 x i64], ptr %av5, i64 0, i64 4
  store i64 50, ptr %d2a4, align 8
  %s2a = call ptr @universe_dataframe_series_from(i32 1, ptr %av5, i64 5)
  %sn2 = call i32 @universe_dataframe_series_set_null(ptr %s2a, i64 2)
  %df2 = call ptr @universe_dataframe_new()
  %w2 = call i32 @universe_dataframe_with_column(ptr %df2, ptr @.col_a, i64 1, ptr %s2a)

  %dnout = call ptr @universe_dataframe_drop_nulls(ptr %df2, ptr null, i64 0)
  %dnh = call i64 @universe_dataframe_height(ptr %dnout)
  call void @ut_check_eq(i64 %dnh, i64 4, ptr @.m.dnh)
  %dnca = call ptr @universe_dataframe_column(ptr %dnout, ptr @.col_a, i64 1)
  %dncav = call ptr @universe_dataframe_series_values(ptr %dnca)
  %dn0p = getelementptr i64, ptr %dncav, i64 0
  %dn0 = load i64, ptr %dn0p, align 8
  call void @ut_check_eq(i64 %dn0, i64 10, ptr @.m.dnv)
  %dn2p = getelementptr i64, ptr %dncav, i64 2
  %dn2 = load i64, ptr %dn2p, align 8
  call void @ut_check_eq(i64 %dn2, i64 40, ptr @.m.dnv)
  %dnnc = call i64 @universe_dataframe_series_null_count(ptr %dnca)
  call void @ut_check_eq(i64 %dnnc, i64 0, ptr @.m.dnv)
  call void @universe_dataframe_free(ptr %dnout)

  ; filter preserving a null row: keep rows 1,2 (row2 is null) via mask
  %m2 = call ptr @universe_dataframe_series_new(i32 4, i64 5)
  %m2v = call ptr @universe_dataframe_series_values(ptr %m2)
  %m2v1 = getelementptr i8, ptr %m2v, i64 1
  store i8 1, ptr %m2v1, align 1
  %m2v2 = getelementptr i8, ptr %m2v, i64 2
  store i8 1, ptr %m2v2, align 1
  %f2out = call ptr @universe_dataframe_filter(ptr %df2, ptr %m2)
  %f2ca = call ptr @universe_dataframe_column(ptr %f2out, ptr @.col_a, i64 1)
  %f2nc = call i64 @universe_dataframe_series_null_count(ptr %f2ca)
  call void @ut_check_eq(i64 %f2nc, i64 1, ptr @.m.fvld)
  ; output row 1 corresponds to source row 2 (null)
  %f2isn = call i32 @universe_dataframe_series_is_null(ptr %f2ca, i64 1, ptr %isn)
  %f2isnv = load i8, ptr %isn, align 1
  %f2isnz = zext i8 %f2isnv to i64
  call void @ut_check_eq(i64 %f2isnz, i64 1, ptr @.m.fvld)
  call void @universe_dataframe_free(ptr %f2out)
  call void @universe_dataframe_series_free(ptr %m2)
  call void @universe_dataframe_free(ptr %df2)

  ; ================= fill_null ============================================
  ; df3: col "a" = [10,20,30,40,50], null at idx 2, fill with 99
  %d3a0 = getelementptr [5 x i64], ptr %av5b, i64 0, i64 0
  store i64 10, ptr %d3a0, align 8
  %d3a1 = getelementptr [5 x i64], ptr %av5b, i64 0, i64 1
  store i64 20, ptr %d3a1, align 8
  %d3a2 = getelementptr [5 x i64], ptr %av5b, i64 0, i64 2
  store i64 30, ptr %d3a2, align 8
  %d3a3 = getelementptr [5 x i64], ptr %av5b, i64 0, i64 3
  store i64 40, ptr %d3a3, align 8
  %d3a4 = getelementptr [5 x i64], ptr %av5b, i64 0, i64 4
  store i64 50, ptr %d3a4, align 8
  %s3a = call ptr @universe_dataframe_series_from(i32 1, ptr %av5b, i64 5)
  %sn3 = call i32 @universe_dataframe_series_set_null(ptr %s3a, i64 2)
  %df3 = call ptr @universe_dataframe_new()
  %w3 = call i32 @universe_dataframe_with_column(ptr %df3, ptr @.col_a, i64 1, ptr %s3a)

  store i64 99, ptr %fillv, align 8
  %fnrc = call i32 @universe_dataframe_fill_null(ptr %df3, ptr @.col_a, i64 1, ptr %fillv)
  %fnrc64 = sext i32 %fnrc to i64
  call void @ut_check_eq(i64 %fnrc64, i64 0, ptr @.m.fnr)
  %f3ca = call ptr @universe_dataframe_column(ptr %df3, ptr @.col_a, i64 1)
  %f3cav = call ptr @universe_dataframe_series_values(ptr %f3ca)
  %f3v2p = getelementptr i64, ptr %f3cav, i64 2
  %f3v2 = load i64, ptr %f3v2p, align 8
  call void @ut_check_eq(i64 %f3v2, i64 99, ptr @.m.fnv)
  %f3nc = call i64 @universe_dataframe_series_null_count(ptr %f3ca)
  call void @ut_check_eq(i64 %f3nc, i64 0, ptr @.m.fnc)
  %f3isn = call i32 @universe_dataframe_series_is_null(ptr %f3ca, i64 2, ptr %isn)
  %f3isnv = load i8, ptr %isn, align 1
  %f3isnz = zext i8 %f3isnv to i64
  call void @ut_check_eq(i64 %f3isnz, i64 0, ptr @.m.fnn)
  ; fill_null not-found -> 5
  %fnnf = call i32 @universe_dataframe_fill_null(ptr %df3, ptr @.col_b, i64 1, ptr %fillv)
  %fnnf64 = sext i32 %fnnf to i64
  call void @ut_check_eq(i64 %fnnf64, i64 5, ptr @.m.fnr)
  call void @universe_dataframe_free(ptr %df3)

  ; ================= fuzz: random masks -> height == popcount =============
  store i64 20260717, ptr %rngst, align 8
  br label %fz.loop
fz.loop:
  %fzk = phi i64 [ 0, %sum.done ], [ %fzkn, %fz.cont ]
  %fzviol = phi i64 [ 0, %sum.done ], [ %fzviol.o, %fz.cont ]
  %fzcmp = icmp ult i64 %fzk, 300
  br i1 %fzcmp, label %fz.body, label %fz.done
fz.body:
  %fmask = call ptr @universe_dataframe_series_new(i32 4, i64 6)
  %fmv = call ptr @universe_dataframe_series_values(ptr %fmask)
  br label %fill.loop
fill.loop:
  %fj = phi i64 [ 0, %fz.body ], [ %fjn, %fill.body ]
  %fexp = phi i64 [ 0, %fz.body ], [ %fexp.n, %fill.body ]
  %fjcmp = icmp ult i64 %fj, 6
  br i1 %fjcmp, label %fill.body, label %fill.done
fill.body:
  %rr = call i64 @ut_rand(ptr %rngst)
  %bit = and i64 %rr, 1
  %bit8 = trunc i64 %bit to i8
  %fmvp = getelementptr i8, ptr %fmv, i64 %fj
  store i8 %bit8, ptr %fmvp, align 1
  %fexp.n = add i64 %fexp, %bit
  %fjn = add i64 %fj, 1
  br label %fill.loop
fill.done:
  %fzout = call ptr @universe_dataframe_filter(ptr %df, ptr %fmask)
  %fzh = call i64 @universe_dataframe_height(ptr %fzout)
  %fzbad = icmp ne i64 %fzh, %fexp
  %fzinc = zext i1 %fzbad to i64
  %fzviol.n = add i64 %fzviol, %fzinc
  call void @universe_dataframe_free(ptr %fzout)
  call void @universe_dataframe_series_free(ptr %fmask)
  ; random in-range take of 3 indices
  %ti0 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 0
  %tr0 = call i64 @ut_rand(ptr %rngst)
  %tm0 = urem i64 %tr0, 6
  store i64 %tm0, ptr %ti0, align 8
  %ti1 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 1
  %tr1 = call i64 @ut_rand(ptr %rngst)
  %tm1 = urem i64 %tr1, 6
  store i64 %tm1, ptr %ti1, align 8
  %ti2 = getelementptr [3 x i64], ptr %idx3, i64 0, i64 2
  %tr2 = call i64 @ut_rand(ptr %rngst)
  %tm2 = urem i64 %tr2, 6
  store i64 %tm2, ptr %ti2, align 8
  %tzout = call ptr @universe_dataframe_take(ptr %df, ptr %idx3, i64 3)
  %tzh = call i64 @universe_dataframe_height(ptr %tzout)
  %tzbad = icmp ne i64 %tzh, 3
  %tzinc = zext i1 %tzbad to i64
  %fzviol.o = add i64 %fzviol.n, %tzinc
  call void @universe_dataframe_free(ptr %tzout)
  br label %fz.cont
fz.cont:
  %fzkn = add i64 %fzk, 1
  br label %fz.loop
fz.done:
  call void @ut_check_eq(i64 %fzviol, i64 0, ptr @.m.fuzz)

  call void @universe_dataframe_free(ptr %df)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
