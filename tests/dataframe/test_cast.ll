; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/cast.ll — SIMD dtype conversion + null-aware fill/
; clip/is_null. Every (from,to) numeric cast is cross-checked against a scalar
; oracle that uses the SAME LLVM convert instruction (per the SIMD contract the
; scalar tail IS the oracle), so the vector path must match bit-for-bit
; (memcmp==0). Saturation on out-of-range floats is checked against hardcoded
; expected values (a real KAT, not just self-consistency). fill/clip/mask are
; checked against manual reference loops. Edge sizes 0/1/17/64/large.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @llvm.fptosi.sat.i32.f64(double)

; frame.ll
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare void @universe_dataframe_series_free(ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare i32 @universe_dataframe_series_dtype(ptr)
declare ptr @universe_dataframe_series_values(ptr)
declare ptr @universe_dataframe_series_validity(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)

; module under test
declare ptr @universe_dataframe_cast(ptr, i32)
declare ptr @universe_dataframe_fill_null_value(ptr, i32, i64, double)
declare ptr @universe_dataframe_clip(ptr, double, double)
declare ptr @universe_dataframe_is_null_mask(ptr)
declare ptr @universe_dataframe_is_not_null_mask(ptr)

@.m_i32i64 = private constant [16 x i8] c"cast i32->i64 =\00"
@.m_i32f32 = private constant [16 x i8] c"cast i32->f32 =\00"
@.m_i32f64 = private constant [16 x i8] c"cast i32->f64 =\00"
@.m_i64i32 = private constant [16 x i8] c"cast i64->i32 =\00"
@.m_i64f32 = private constant [16 x i8] c"cast i64->f32 =\00"
@.m_i64f64 = private constant [16 x i8] c"cast i64->f64 =\00"
@.m_f32i32 = private constant [16 x i8] c"cast f32->i32 =\00"
@.m_f32i64 = private constant [16 x i8] c"cast f32->i64 =\00"
@.m_f32f64 = private constant [16 x i8] c"cast f32->f64 =\00"
@.m_f64i32 = private constant [16 x i8] c"cast f64->i32 =\00"
@.m_f64i64 = private constant [16 x i8] c"cast f64->i64 =\00"
@.m_f64f32 = private constant [16 x i8] c"cast f64->f32 =\00"
@.m_sat    = private constant [18 x i8] c"f64->i32 saturate\00"
@.m_nullpr = private constant [18 x i8] c"cast keeps nulls \00"
@.m_ncpr   = private constant [16 x i8] c"cast null_count\00"
@.m_fill   = private constant [15 x i8] c"fill_null vals\00"
@.m_fillnc = private constant [16 x i8] c"fill no nulls  \00"
@.m_fillvd = private constant [16 x i8] c"fill validity 0\00"
@.m_filli  = private constant [16 x i8] c"fill_null i32  \00"
@.m_isnull = private constant [14 x i8] c"is_null_mask \00"
@.m_isnot  = private constant [18 x i8] c"is_not_null_mask \00"
@.m_clipi  = private constant [12 x i8] c"clip i32   \00"
@.m_clipf  = private constant [12 x i8] c"clip f64   \00"

; ---------------------------------------------------------------------------
; source builders (random values; oracle mirrors the exact convert op)
; ---------------------------------------------------------------------------
define internal ptr @mk_i32(i64 %n, ptr %seed) {
entry:
  %s = call ptr @universe_dataframe_series_new(i32 0, i64 %n)
  %v = call ptr @universe_dataframe_series_values(ptr %s)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %r = call i64 @ut_rand(ptr %seed)
  %r32 = trunc i64 %r to i32
  %p = getelementptr inbounds i32, ptr %v, i64 %i
  store i32 %r32, ptr %p, align 4
  %in = add i64 %i, 1
  br label %lp
done:
  ret ptr %s
}

define internal ptr @mk_i64(i64 %n, ptr %seed) {
entry:
  %s = call ptr @universe_dataframe_series_new(i32 1, i64 %n)
  %v = call ptr @universe_dataframe_series_values(ptr %s)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %r = call i64 @ut_rand(ptr %seed)
  %p = getelementptr inbounds i64, ptr %v, i64 %i
  store i64 %r, ptr %p, align 8
  %in = add i64 %i, 1
  br label %lp
done:
  ret ptr %s
}

; f64 values in i32 range (so f64->i32/i64 stay in-range), with a fractional part
define internal ptr @mk_f64(i64 %n, ptr %seed) {
entry:
  %s = call ptr @universe_dataframe_series_new(i32 3, i64 %n)
  %v = call ptr @universe_dataframe_series_values(ptr %s)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %r = call i64 @ut_rand(ptr %seed)
  %r16 = and i64 %r, 65535
  %rc = sub i64 %r16, 32768
  %f = sitofp i64 %rc to double
  %fh = fadd double %f, 5.000000e-01
  %p = getelementptr inbounds double, ptr %v, i64 %i
  store double %fh, ptr %p, align 8
  %in = add i64 %i, 1
  br label %lp
done:
  ret ptr %s
}

define internal ptr @mk_f32(i64 %n, ptr %seed) {
entry:
  %s = call ptr @universe_dataframe_series_new(i32 2, i64 %n)
  %v = call ptr @universe_dataframe_series_values(ptr %s)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %r = call i64 @ut_rand(ptr %seed)
  %r16 = and i64 %r, 65535
  %rc = sub i64 %r16, 32768
  %f = sitofp i64 %rc to float
  %fh = fadd float %f, 2.500000e-01
  %p = getelementptr inbounds float, ptr %v, i64 %i
  store float %fh, ptr %p, align 4
  %in = add i64 %i, 1
  br label %lp
done:
  ret ptr %s
}

; ---------------------------------------------------------------------------
; per-pair helpers: oracle buffer (scalar convert) vs cast result (memcmp==0)
; ---------------------------------------------------------------------------
define internal void @p_i32_i64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i32, ptr %sv, i64 %i
  %x = load i32, ptr %sp, align 4
  %e = sext i32 %x to i64
  %ep = getelementptr inbounds i64, ptr %exp, i64 %i
  store i64 %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 1)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i32i64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_i32_f32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i32, ptr %sv, i64 %i
  %x = load i32, ptr %sp, align 4
  %e = sitofp i32 %x to float
  %ep = getelementptr inbounds float, ptr %exp, i64 %i
  store float %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 2)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i32f32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_i32_f64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i32, ptr %sv, i64 %i
  %x = load i32, ptr %sp, align 4
  %e = sitofp i32 %x to double
  %ep = getelementptr inbounds double, ptr %exp, i64 %i
  store double %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 3)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i32f64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_i64_i32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i64, ptr %sv, i64 %i
  %x = load i64, ptr %sp, align 8
  %e = trunc i64 %x to i32
  %ep = getelementptr inbounds i32, ptr %exp, i64 %i
  store i32 %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 0)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i64i32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_i64_f32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i64, ptr %sv, i64 %i
  %x = load i64, ptr %sp, align 8
  %e = sitofp i64 %x to float
  %ep = getelementptr inbounds float, ptr %exp, i64 %i
  store float %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 2)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i64f32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_i64_f64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i64, ptr %sv, i64 %i
  %x = load i64, ptr %sp, align 8
  %e = sitofp i64 %x to double
  %ep = getelementptr inbounds double, ptr %exp, i64 %i
  store double %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 3)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_i64f64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f32_f64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds float, ptr %sv, i64 %i
  %x = load float, ptr %sp, align 4
  %e = fpext float %x to double
  %ep = getelementptr inbounds double, ptr %exp, i64 %i
  store double %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 3)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f32f64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f32_i32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds float, ptr %sv, i64 %i
  %x = load float, ptr %sp, align 4
  %e = fptosi float %x to i32
  %ep = getelementptr inbounds i32, ptr %exp, i64 %i
  store i32 %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 0)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f32i32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f32_i64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds float, ptr %sv, i64 %i
  %x = load float, ptr %sp, align 4
  %e = fptosi float %x to i64
  %ep = getelementptr inbounds i64, ptr %exp, i64 %i
  store i64 %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 1)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f32i64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f64_i32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds double, ptr %sv, i64 %i
  %x = load double, ptr %sp, align 8
  %e = fptosi double %x to i32
  %ep = getelementptr inbounds i32, ptr %exp, i64 %i
  store i32 %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 0)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f64i32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f64_i64(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds double, ptr %sv, i64 %i
  %x = load double, ptr %sp, align 8
  %e = fptosi double %x to i64
  %ep = getelementptr inbounds i64, ptr %exp, i64 %i
  store i64 %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 1)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f64i64)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

define internal void @p_f64_f32(ptr %src, i64 %n) {
entry:
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds double, ptr %sv, i64 %i
  %x = load double, ptr %sp, align 8
  %e = fptrunc double %x to float
  %ep = getelementptr inbounds float, ptr %exp, i64 %i
  store float %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_cast(ptr %src, i32 2)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_f64f32)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  ret void
}

; run all 12 pairs at length n
define internal void @all_pairs(i64 %n, ptr %seed) {
entry:
  %si32 = call ptr @mk_i32(i64 %n, ptr %seed)
  call void @p_i32_i64(ptr %si32, i64 %n)
  call void @p_i32_f32(ptr %si32, i64 %n)
  call void @p_i32_f64(ptr %si32, i64 %n)
  call void @universe_dataframe_series_free(ptr %si32)
  %si64 = call ptr @mk_i64(i64 %n, ptr %seed)
  call void @p_i64_i32(ptr %si64, i64 %n)
  call void @p_i64_f32(ptr %si64, i64 %n)
  call void @p_i64_f64(ptr %si64, i64 %n)
  call void @universe_dataframe_series_free(ptr %si64)
  %sf64 = call ptr @mk_f64(i64 %n, ptr %seed)
  call void @p_f64_i32(ptr %sf64, i64 %n)
  call void @p_f64_i64(ptr %sf64, i64 %n)
  call void @p_f64_f32(ptr %sf64, i64 %n)
  call void @universe_dataframe_series_free(ptr %sf64)
  %sf32 = call ptr @mk_f32(i64 %n, ptr %seed)
  call void @p_f32_i32(ptr %sf32, i64 %n)
  call void @p_f32_i64(ptr %sf32, i64 %n)
  call void @p_f32_f64(ptr %sf32, i64 %n)
  call void @universe_dataframe_series_free(ptr %sf32)
  ret void
}

; ---------------------------------------------------------------------------
; saturation KAT: f64 -> i32 with out-of-range / NaN inputs
; ---------------------------------------------------------------------------
define internal void @test_saturate() {
entry:
  %s = call ptr @universe_dataframe_series_new(i32 3, i64 6)
  %v = call ptr @universe_dataframe_series_values(ptr %s)
  %p0 = getelementptr inbounds double, ptr %v, i64 0
  store double 1.000000e+30, ptr %p0, align 8
  %p1 = getelementptr inbounds double, ptr %v, i64 1
  store double -1.000000e+30, ptr %p1, align 8
  %p2 = getelementptr inbounds double, ptr %v, i64 2
  store double 4.290000e+01, ptr %p2, align 8
  %p3 = getelementptr inbounds double, ptr %v, i64 3
  store double -4.290000e+01, ptr %p3, align 8
  %p4 = getelementptr inbounds double, ptr %v, i64 4
  store double 0x7FF8000000000000, ptr %p4, align 8
  %p5 = getelementptr inbounds double, ptr %v, i64 5
  store double 2.500000e+00, ptr %p5, align 8
  %out = call ptr @universe_dataframe_cast(ptr %s, i32 0)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %o0 = load i32, ptr %ov, align 4
  %e0 = icmp eq i32 %o0, 2147483647
  call void @ut_check(i1 %e0, ptr @.m_sat)
  %q1 = getelementptr inbounds i32, ptr %ov, i64 1
  %o1 = load i32, ptr %q1, align 4
  %e1 = icmp eq i32 %o1, -2147483648
  call void @ut_check(i1 %e1, ptr @.m_sat)
  %q2 = getelementptr inbounds i32, ptr %ov, i64 2
  %o2 = load i32, ptr %q2, align 4
  %e2 = icmp eq i32 %o2, 42
  call void @ut_check(i1 %e2, ptr @.m_sat)
  %q3 = getelementptr inbounds i32, ptr %ov, i64 3
  %o3 = load i32, ptr %q3, align 4
  %e3 = icmp eq i32 %o3, -42
  call void @ut_check(i1 %e3, ptr @.m_sat)
  %q4 = getelementptr inbounds i32, ptr %ov, i64 4
  %o4 = load i32, ptr %q4, align 4
  %e4 = icmp eq i32 %o4, 0
  call void @ut_check(i1 %e4, ptr @.m_sat)
  %q5 = getelementptr inbounds i32, ptr %ov, i64 5
  %o5 = load i32, ptr %q5, align 4
  %e5 = icmp eq i32 %o5, 2
  call void @ut_check(i1 %e5, ptr @.m_sat)
  call void @universe_dataframe_series_free(ptr %out)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; read validity bit directly: true if element i is NULL
define internal i1 @ref_isnull(ptr %s, i64 %i) {
entry:
  %vld = call ptr @universe_dataframe_series_validity(ptr %s)
  %n = icmp eq ptr %vld, null
  br i1 %n, label %valid, label %chk
valid:
  ret i1 false
chk:
  %bi = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %i, 7
  %sh8 = trunc i64 %sh to i8
  %bit = lshr i8 %b, %sh8
  %lo = and i8 %bit, 1
  %isv = icmp ne i8 %lo, 0
  %isn = xor i1 %isv, true
  ret i1 %isn
}

; ---------------------------------------------------------------------------
; cast preserves nulls
; ---------------------------------------------------------------------------
define internal void @test_null_preserve(ptr %seed) {
entry:
  %n = add i64 0, 40
  %s = call ptr @mk_i32(i64 %n, ptr %seed)
  call void @universe_dataframe_series_set_null(ptr %s, i64 0)
  call void @universe_dataframe_series_set_null(ptr %s, i64 7)
  call void @universe_dataframe_series_set_null(ptr %s, i64 8)
  call void @universe_dataframe_series_set_null(ptr %s, i64 39)
  %out = call ptr @universe_dataframe_cast(ptr %s, i32 3)
  ; null_count equal
  %sc = call i64 @universe_dataframe_series_null_count(ptr %s)
  %oc = call i64 @universe_dataframe_series_null_count(ptr %out)
  call void @ut_check_eq(i64 %sc, i64 %oc, ptr @.m_ncpr)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %viol = phi i64 [ 0, %entry ], [ %vn, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %a = call i1 @ref_isnull(ptr %s, i64 %i)
  %b = call i1 @ref_isnull(ptr %out, i64 %i)
  %ne = xor i1 %a, %b
  %inc = zext i1 %ne to i64
  %vn = add i64 %viol, %inc
  %in = add i64 %i, 1
  br label %lp
done:
  %ok = icmp eq i64 %viol, 0
  call void @ut_check(i1 %ok, ptr @.m_nullpr)
  call void @universe_dataframe_series_free(ptr %out)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; ---------------------------------------------------------------------------
; fill_null on f64 with nulls
; ---------------------------------------------------------------------------
define internal void @test_fill_f64(ptr %seed) {
entry:
  %n = add i64 0, 37
  %s = call ptr @mk_f64(i64 %n, ptr %seed)
  call void @universe_dataframe_series_set_null(ptr %s, i64 0)
  call void @universe_dataframe_series_set_null(ptr %s, i64 5)
  call void @universe_dataframe_series_set_null(ptr %s, i64 8)
  call void @universe_dataframe_series_set_null(ptr %s, i64 36)
  %sv = call ptr @universe_dataframe_series_values(ptr %s)
  ; expected buffer
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %isn = call i1 @ref_isnull(ptr %s, i64 %i)
  %sp = getelementptr inbounds double, ptr %sv, i64 %i
  %orig = load double, ptr %sp, align 8
  %e = select i1 %isn, double 7.500000e+00, double %orig
  %ep = getelementptr inbounds double, ptr %exp, i64 %i
  store double %e, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_fill_null_value(ptr %s, i32 3, i64 0, double 7.500000e+00)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_fill)
  ; result has no nulls
  %oc = call i64 @universe_dataframe_series_null_count(ptr %out)
  %okc = icmp eq i64 %oc, 0
  call void @ut_check(i1 %okc, ptr @.m_fillnc)
  %ovld = call ptr @universe_dataframe_series_validity(ptr %out)
  %vnull = icmp eq ptr %ovld, null
  call void @ut_check(i1 %vnull, ptr @.m_fillvd)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; fill_null on i32 with ival
define internal void @test_fill_i32(ptr %seed) {
entry:
  %n = add i64 0, 20
  %s = call ptr @mk_i32(i64 %n, ptr %seed)
  call void @universe_dataframe_series_set_null(ptr %s, i64 3)
  call void @universe_dataframe_series_set_null(ptr %s, i64 19)
  %sv = call ptr @universe_dataframe_series_values(ptr %s)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %isn = call i1 @ref_isnull(ptr %s, i64 %i)
  %sp = getelementptr inbounds i32, ptr %sv, i64 %i
  %orig = load i32, ptr %sp, align 4
  %e = select i1 %isn, i32 -777, i32 %orig
  %ep = getelementptr inbounds i32, ptr %exp, i64 %i
  store i32 %e, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_fill_null_value(ptr %s, i32 0, i64 -777, double 0.000000e+00)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_filli)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; ---------------------------------------------------------------------------
; is_null_mask / is_not_null_mask
; ---------------------------------------------------------------------------
define internal void @test_null_mask(ptr %seed) {
entry:
  %n = add i64 0, 45
  %s = call ptr @mk_i32(i64 %n, ptr %seed)
  call void @universe_dataframe_series_set_null(ptr %s, i64 1)
  call void @universe_dataframe_series_set_null(ptr %s, i64 2)
  call void @universe_dataframe_series_set_null(ptr %s, i64 15)
  call void @universe_dataframe_series_set_null(ptr %s, i64 44)
  %mn = call ptr @universe_dataframe_is_null_mask(ptr %s)
  %mnv = call ptr @universe_dataframe_series_values(ptr %mn)
  %mp = call ptr @universe_dataframe_is_not_null_mask(ptr %s)
  %mpv = call ptr @universe_dataframe_series_values(ptr %mp)
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %bd ]
  %viol = phi i64 [ 0, %entry ], [ %vn3, %bd ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %bd, label %done
bd:
  %isn = call i1 @ref_isnull(ptr %s, i64 %i)
  %en = zext i1 %isn to i8
  %np = getelementptr inbounds i8, ptr %mnv, i64 %i
  %gn = load i8, ptr %np, align 1
  %bad1 = icmp ne i8 %gn, %en
  %ep2 = xor i1 %isn, true
  %enn = zext i1 %ep2 to i8
  %pp = getelementptr inbounds i8, ptr %mpv, i64 %i
  %gp = load i8, ptr %pp, align 1
  %bad2 = icmp ne i8 %gp, %enn
  %inc1 = zext i1 %bad1 to i64
  %inc2 = zext i1 %bad2 to i64
  %vn1 = add i64 %viol, %inc1
  %vn3 = add i64 %vn1, %inc2
  %in = add i64 %i, 1
  br label %lp
done:
  %ok = icmp eq i64 %viol, 0
  call void @ut_check(i1 %ok, ptr @.m_isnull)
  ; a series with NO nulls -> is_null all 0, is_not_null all 1
  %s2 = call ptr @mk_i32(i64 %n, ptr %seed)
  %m2 = call ptr @universe_dataframe_is_not_null_mask(ptr %s2)
  %m2v = call ptr @universe_dataframe_series_values(ptr %m2)
  br label %lp2
lp2:
  %j = phi i64 [ 0, %done ], [ %jn, %bd2 ]
  %viol2 = phi i64 [ 0, %done ], [ %vv2, %bd2 ]
  %c2 = icmp ult i64 %j, %n
  br i1 %c2, label %bd2, label %done2
bd2:
  %gp2 = getelementptr inbounds i8, ptr %m2v, i64 %j
  %g2 = load i8, ptr %gp2, align 1
  %bad = icmp ne i8 %g2, 1
  %incc = zext i1 %bad to i64
  %vv2 = add i64 %viol2, %incc
  %jn = add i64 %j, 1
  br label %lp2
done2:
  %ok2 = icmp eq i64 %viol2, 0
  call void @ut_check(i1 %ok2, ptr @.m_isnot)
  call void @universe_dataframe_series_free(ptr %m2)
  call void @universe_dataframe_series_free(ptr %s2)
  call void @universe_dataframe_series_free(ptr %mn)
  call void @universe_dataframe_series_free(ptr %mp)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; ---------------------------------------------------------------------------
; clip
; ---------------------------------------------------------------------------
define internal void @test_clip_i32(ptr %seed) {
entry:
  %n = add i64 0, 33
  %s = call ptr @mk_i32(i64 %n, ptr %seed)
  %sv = call ptr @universe_dataframe_series_values(ptr %s)
  %bytes = mul i64 %n, 4
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds i32, ptr %sv, i64 %i
  %x = load i32, ptr %sp, align 4
  %lo = call i32 @smax32(i32 %x, i32 -100)
  %clamped = call i32 @smin32(i32 %lo, i32 100)
  %ep = getelementptr inbounds i32, ptr %exp, i64 %i
  store i32 %clamped, ptr %ep, align 4
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_clip(ptr %s, double -1.000000e+02, double 1.000000e+02)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_clipi)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

define internal i32 @smax32(i32 %a, i32 %b) {
entry:
  %c = icmp sgt i32 %a, %b
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r
}
define internal i32 @smin32(i32 %a, i32 %b) {
entry:
  %c = icmp slt i32 %a, %b
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r
}

define internal void @test_clip_f64(ptr %seed) {
entry:
  %n = add i64 0, 33
  %s = call ptr @mk_f64(i64 %n, ptr %seed)
  %sv = call ptr @universe_dataframe_series_values(ptr %s)
  %bytes = mul i64 %n, 8
  %exp = call ptr @malloc(i64 %bytes)
  br label %ol
ol:
  %i = phi i64 [ 0, %entry ], [ %in, %ob ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %ob, label %run
ob:
  %sp = getelementptr inbounds double, ptr %sv, i64 %i
  %x = load double, ptr %sp, align 8
  %lohi = fcmp olt double %x, -5.000000e+00
  %t1 = select i1 %lohi, double -5.000000e+00, double %x
  %hic = fcmp ogt double %t1, 5.000000e+00
  %t2 = select i1 %hic, double 5.000000e+00, double %t1
  %ep = getelementptr inbounds double, ptr %exp, i64 %i
  store double %t2, ptr %ep, align 8
  %in = add i64 %i, 1
  br label %ol
run:
  %out = call ptr @universe_dataframe_clip(ptr %s, double -5.000000e+00, double 5.000000e+00)
  %ov = call ptr @universe_dataframe_series_values(ptr %out)
  %r = call i32 @memcmp(ptr %ov, ptr %exp, i64 %bytes)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @.m_clipf)
  call void @universe_dataframe_series_free(ptr %out)
  call void @free(ptr %exp)
  call void @universe_dataframe_series_free(ptr %s)
  ret void
}

; ---------------------------------------------------------------------------
define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  ; edge + representative sizes for all 12 pairs
  call void @all_pairs(i64 0, ptr %seed)
  call void @all_pairs(i64 1, ptr %seed)
  call void @all_pairs(i64 2, ptr %seed)
  call void @all_pairs(i64 3, ptr %seed)
  call void @all_pairs(i64 4, ptr %seed)
  call void @all_pairs(i64 16, ptr %seed)
  call void @all_pairs(i64 17, ptr %seed)
  call void @all_pairs(i64 64, ptr %seed)
  call void @all_pairs(i64 100, ptr %seed)
  call void @all_pairs(i64 1000, ptr %seed)
  call void @test_saturate()
  call void @test_null_preserve(ptr %seed)
  call void @test_fill_f64(ptr %seed)
  call void @test_fill_i32(ptr %seed)
  call void @test_null_mask(ptr %seed)
  call void @test_clip_i32(ptr %seed)
  call void @test_clip_f64(ptr %seed)
  %r = call i32 @ut_summary()
  ret i32 %r
}
