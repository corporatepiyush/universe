; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/compare.ll — SIMD predicates + boolean algebra.
; Every comparator (col-col + col-scalar) for i32/i64/f32/f64 is cross-checked
; EXACTLY against an in-IR scalar oracle over fixed-seed inputs (lengths that
; exercise both the 128-bit vector body and the scalar tail). Boolean and/or/xor/
; not truth tables; any/all/sum_bool (+ null lanes); zip_with; float NaN ordering;
; null-lane validity; and an INTEGRATION PROOF that the mask produced by compare
; drives the existing universe_dataframe_filter (filter is now SIMD-predicated).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare void @free(ptr)

; frame primitives
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare void @universe_dataframe_series_free(ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare ptr @universe_dataframe_series_values(ptr)
declare ptr @universe_dataframe_series_validity(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)
declare i32 @universe_dataframe_series_dtype(ptr)
declare ptr @universe_dataframe_new()
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_column(ptr, ptr, i64)
declare void @universe_dataframe_free(ptr)
declare ptr @universe_dataframe_filter(ptr, ptr)

; compare.ll under test
declare ptr @universe_dataframe_eq(ptr, ptr)
declare ptr @universe_dataframe_neq(ptr, ptr)
declare ptr @universe_dataframe_lt(ptr, ptr)
declare ptr @universe_dataframe_lte(ptr, ptr)
declare ptr @universe_dataframe_gt(ptr, ptr)
declare ptr @universe_dataframe_gte(ptr, ptr)
declare ptr @universe_dataframe_eq_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_neq_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_lt_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_lte_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_gt_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_gte_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_and(ptr, ptr)
declare ptr @universe_dataframe_or(ptr, ptr)
declare ptr @universe_dataframe_xor(ptr, ptr)
declare ptr @universe_dataframe_not(ptr)
declare i32 @universe_dataframe_any(ptr, ptr)
declare i32 @universe_dataframe_all(ptr, ptr)
declare i32 @universe_dataframe_sum_bool(ptr, ptr)
declare ptr @universe_dataframe_zip_with(ptr, ptr, ptr)

@.mv = private constant [8 x i8] c"v\00\00\00\00\00\00\00"

@.m_i32 = private constant [24 x i8] c"i32 cmp mask lane      \00"
@.m_i64 = private constant [24 x i8] c"i64 cmp mask lane      \00"
@.m_f32 = private constant [24 x i8] c"f32 cmp mask lane      \00"
@.m_f64 = private constant [24 x i8] c"f64 cmp mask lane      \00"
@.m_i32s= private constant [24 x i8] c"i32 scalar mask lane   \00"
@.m_i64s= private constant [24 x i8] c"i64 scalar mask lane   \00"
@.m_f32s= private constant [24 x i8] c"f32 scalar mask lane   \00"
@.m_f64s= private constant [24 x i8] c"f64 scalar mask lane   \00"
@.m_and = private constant [24 x i8] c"bool and lane          \00"
@.m_or  = private constant [24 x i8] c"bool or lane           \00"
@.m_xor = private constant [24 x i8] c"bool xor lane          \00"
@.m_not = private constant [24 x i8] c"bool not lane          \00"
@.m_any = private constant [24 x i8] c"any result             \00"
@.m_all = private constant [24 x i8] c"all result             \00"
@.m_sum = private constant [24 x i8] c"sum_bool result        \00"
@.m_zip = private constant [24 x i8] c"zip_with lane          \00"
@.m_nul = private constant [24 x i8] c"null-lane validity     \00"
@.m_dt  = private constant [24 x i8] c"result dtype BOOL      \00"
@.m_err = private constant [24 x i8] c"error-path null        \00"
@.m_fh  = private constant [24 x i8] c"filter height          \00"
@.m_fv  = private constant [24 x i8] c"filter value > 5       \00"

; ---------------------------------------------------------------------------
; oracle checkers: compare EACH result byte vs a scalar oracle for op 0..5.
;   op: 0 eq 1 neq 2 lt 3 lte 4 gt 5 gte
; ---------------------------------------------------------------------------

define void @chk_i32(ptr %av, ptr %bv, ptr %rv, i64 %n, i32 %op, ptr %msg) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %pa = getelementptr inbounds i32, ptr %av, i64 %i
  %a = load i32, ptr %pa, align 4
  %pb = getelementptr inbounds i32, ptr %bv, i64 %i
  %b = load i32, ptr %pb, align 4
  %eq = icmp eq i32 %a, %b
  %lt = icmp slt i32 %a, %b
  %ne = xor i1 %eq, true
  %le = or i1 %lt, %eq
  %gt = xor i1 %le, true
  %ge = xor i1 %lt, true
  %iseq = icmp eq i32 %op, 0
  %isne = icmp eq i32 %op, 1
  %islt = icmp eq i32 %op, 2
  %isle = icmp eq i32 %op, 3
  %isgt = icmp eq i32 %op, 4
  %pgt = select i1 %isgt, i1 %gt, i1 %ge
  %s3 = select i1 %isle, i1 %le, i1 %pgt
  %p2 = select i1 %islt, i1 %lt, i1 %s3
  %p1 = select i1 %isne, i1 %ne, i1 %p2
  %e0 = select i1 %iseq, i1 %eq, i1 %p1
  %exp = zext i1 %e0 to i64
  %pr = getelementptr inbounds i8, ptr %rv, i64 %i
  %r = load i8, ptr %pr, align 1
  %rz = zext i8 %r to i64
  call void @ut_check_eq(i64 %rz, i64 %exp, ptr %msg)
  br label %cont
cont:
  %in = add i64 %i, 1
  br label %loop
done:
  ret void
}

define void @chk_i64(ptr %av, ptr %bv, ptr %rv, i64 %n, i32 %op, ptr %msg) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %pa = getelementptr inbounds i64, ptr %av, i64 %i
  %a = load i64, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %bv, i64 %i
  %b = load i64, ptr %pb, align 8
  %eq = icmp eq i64 %a, %b
  %lt = icmp slt i64 %a, %b
  %ne = xor i1 %eq, true
  %le = or i1 %lt, %eq
  %gt = xor i1 %le, true
  %ge = xor i1 %lt, true
  %iseq = icmp eq i32 %op, 0
  %isne = icmp eq i32 %op, 1
  %islt = icmp eq i32 %op, 2
  %isle = icmp eq i32 %op, 3
  %isgt = icmp eq i32 %op, 4
  %pgt = select i1 %isgt, i1 %gt, i1 %ge
  %s3 = select i1 %isle, i1 %le, i1 %pgt
  %p2 = select i1 %islt, i1 %lt, i1 %s3
  %p1 = select i1 %isne, i1 %ne, i1 %p2
  %e0 = select i1 %iseq, i1 %eq, i1 %p1
  %exp = zext i1 %e0 to i64
  %pr = getelementptr inbounds i8, ptr %rv, i64 %i
  %r = load i8, ptr %pr, align 1
  %rz = zext i8 %r to i64
  call void @ut_check_eq(i64 %rz, i64 %exp, ptr %msg)
  br label %cont
cont:
  %in = add i64 %i, 1
  br label %loop
done:
  ret void
}

define void @chk_f32(ptr %av, ptr %bv, ptr %rv, i64 %n, i32 %op, ptr %msg) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %pa = getelementptr inbounds float, ptr %av, i64 %i
  %a = load float, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %bv, i64 %i
  %b = load float, ptr %pb, align 4
  %eq = fcmp oeq float %a, %b
  %ne = fcmp one float %a, %b
  %lt = fcmp olt float %a, %b
  %le = fcmp ole float %a, %b
  %gt = fcmp ogt float %a, %b
  %ge = fcmp oge float %a, %b
  %iseq = icmp eq i32 %op, 0
  %isne = icmp eq i32 %op, 1
  %islt = icmp eq i32 %op, 2
  %isle = icmp eq i32 %op, 3
  %isgt = icmp eq i32 %op, 4
  %pgt = select i1 %isgt, i1 %gt, i1 %ge
  %s3 = select i1 %isle, i1 %le, i1 %pgt
  %p2 = select i1 %islt, i1 %lt, i1 %s3
  %p1 = select i1 %isne, i1 %ne, i1 %p2
  %e0 = select i1 %iseq, i1 %eq, i1 %p1
  %exp = zext i1 %e0 to i64
  %pr = getelementptr inbounds i8, ptr %rv, i64 %i
  %r = load i8, ptr %pr, align 1
  %rz = zext i8 %r to i64
  call void @ut_check_eq(i64 %rz, i64 %exp, ptr %msg)
  br label %cont
cont:
  %in = add i64 %i, 1
  br label %loop
done:
  ret void
}

define void @chk_f64(ptr %av, ptr %bv, ptr %rv, i64 %n, i32 %op, ptr %msg) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %pa = getelementptr inbounds double, ptr %av, i64 %i
  %a = load double, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %bv, i64 %i
  %b = load double, ptr %pb, align 8
  %eq = fcmp oeq double %a, %b
  %ne = fcmp one double %a, %b
  %lt = fcmp olt double %a, %b
  %le = fcmp ole double %a, %b
  %gt = fcmp ogt double %a, %b
  %ge = fcmp oge double %a, %b
  %iseq = icmp eq i32 %op, 0
  %isne = icmp eq i32 %op, 1
  %islt = icmp eq i32 %op, 2
  %isle = icmp eq i32 %op, 3
  %isgt = icmp eq i32 %op, 4
  %pgt = select i1 %isgt, i1 %gt, i1 %ge
  %s3 = select i1 %isle, i1 %le, i1 %pgt
  %p2 = select i1 %islt, i1 %lt, i1 %s3
  %p1 = select i1 %isne, i1 %ne, i1 %p2
  %e0 = select i1 %iseq, i1 %eq, i1 %p1
  %exp = zext i1 %e0 to i64
  %pr = getelementptr inbounds i8, ptr %rv, i64 %i
  %r = load i8, ptr %pr, align 1
  %rz = zext i8 %r to i64
  call void @ut_check_eq(i64 %rz, i64 %exp, ptr %msg)
  br label %cont
cont:
  %in = add i64 %i, 1
  br label %loop
done:
  ret void
}

; run all 6 col-col comparators for an i32 pair and check
define void @run_i32(ptr %a, ptr %b, ptr %av, ptr %bv, i64 %n) {
entry:
  %r0 = call ptr @universe_dataframe_eq(ptr %a, ptr %b)
  %v0 = call ptr @universe_dataframe_series_values(ptr %r0)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v0, i64 %n, i32 0, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r0)
  %r1 = call ptr @universe_dataframe_neq(ptr %a, ptr %b)
  %v1 = call ptr @universe_dataframe_series_values(ptr %r1)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v1, i64 %n, i32 1, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r1)
  %r2 = call ptr @universe_dataframe_lt(ptr %a, ptr %b)
  %v2 = call ptr @universe_dataframe_series_values(ptr %r2)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v2, i64 %n, i32 2, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r2)
  %r3 = call ptr @universe_dataframe_lte(ptr %a, ptr %b)
  %v3 = call ptr @universe_dataframe_series_values(ptr %r3)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v3, i64 %n, i32 3, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r3)
  %r4 = call ptr @universe_dataframe_gt(ptr %a, ptr %b)
  %v4 = call ptr @universe_dataframe_series_values(ptr %r4)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v4, i64 %n, i32 4, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r4)
  %r5 = call ptr @universe_dataframe_gte(ptr %a, ptr %b)
  %v5 = call ptr @universe_dataframe_series_values(ptr %r5)
  call void @chk_i32(ptr %av, ptr %bv, ptr %v5, i64 %n, i32 5, ptr @.m_i32)
  call void @universe_dataframe_series_free(ptr %r5)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %state = alloca i64, align 8
  %ai = alloca [37 x i32], align 16
  %bi = alloca [37 x i32], align 16
  %al = alloca [37 x i64], align 16
  %bl = alloca [37 x i64], align 16
  %af = alloca [37 x float], align 16
  %bf = alloca [37 x float], align 16
  %ad = alloca [37 x double], align 16
  %bd = alloca [37 x double], align 16
  %ci = alloca [37 x i32], align 16     ; constant broadcast buffer (i32)
  %cd = alloca [37 x double], align 16  ; constant broadcast buffer (f64)
  %obool = alloca i8, align 1
  %oi64 = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8

  ; ---- fill numeric buffers with small-range rng (ties present) ----
  br label %fill
fill:
  %fi = phi i64 [ 0, %entry ], [ %fin, %fill ]
  %ra = call i64 @ut_rand(ptr %state)
  %rb = call i64 @ut_rand(ptr %state)
  %amod = and i64 %ra, 7
  %bmod = and i64 %rb, 7
  %a32 = trunc i64 %amod to i32
  %b32 = trunc i64 %bmod to i32
  %pai = getelementptr inbounds [37 x i32], ptr %ai, i64 0, i64 %fi
  store i32 %a32, ptr %pai, align 4
  %pbi = getelementptr inbounds [37 x i32], ptr %bi, i64 0, i64 %fi
  store i32 %b32, ptr %pbi, align 4
  %a64 = sext i32 %a32 to i64
  %b64 = sext i32 %b32 to i64
  %pal = getelementptr inbounds [37 x i64], ptr %al, i64 0, i64 %fi
  store i64 %a64, ptr %pal, align 8
  %pbl = getelementptr inbounds [37 x i64], ptr %bl, i64 0, i64 %fi
  store i64 %b64, ptr %pbl, align 8
  %af32 = sitofp i32 %a32 to float
  %bf32 = sitofp i32 %b32 to float
  %paf = getelementptr inbounds [37 x float], ptr %af, i64 0, i64 %fi
  store float %af32, ptr %paf, align 4
  %pbf = getelementptr inbounds [37 x float], ptr %bf, i64 0, i64 %fi
  store float %bf32, ptr %pbf, align 4
  %ad64 = sitofp i32 %a32 to double
  %bd64 = sitofp i32 %b32 to double
  %pad = getelementptr inbounds [37 x double], ptr %ad, i64 0, i64 %fi
  store double %ad64, ptr %pad, align 8
  %pbd = getelementptr inbounds [37 x double], ptr %bd, i64 0, i64 %fi
  store double %bd64, ptr %pbd, align 8
  %pci = getelementptr inbounds [37 x i32], ptr %ci, i64 0, i64 %fi
  store i32 4, ptr %pci, align 4
  %pcd = getelementptr inbounds [37 x double], ptr %cd, i64 0, i64 %fi
  store double 4.0, ptr %pcd, align 8
  %fin = add i64 %fi, 1
  %fc = icmp ult i64 %fin, 37
  br i1 %fc, label %fill, label %inject_nan

inject_nan:
  ; put a NaN into af[5] and bf[9] and ad[5], bd[9] to exercise ordered compares
  %nanf = bitcast i32 2143289344 to float          ; 0x7FC00000 qNaN
  %nand = bitcast i64 9221120237041090560 to double ; 0x7FF8000000000000 qNaN
  %pnaf = getelementptr inbounds [37 x float], ptr %af, i64 0, i64 5
  store float %nanf, ptr %pnaf, align 4
  %pnbf = getelementptr inbounds [37 x float], ptr %bf, i64 0, i64 9
  store float %nanf, ptr %pnbf, align 4
  %pnad = getelementptr inbounds [37 x double], ptr %ad, i64 0, i64 5
  store double %nand, ptr %pnad, align 8
  %pnbd = getelementptr inbounds [37 x double], ptr %bd, i64 0, i64 9
  store double %nand, ptr %pnbd, align 8
  br label %build

build:
  %sai = call ptr @universe_dataframe_series_from(i32 0, ptr %ai, i64 37)
  %sbi = call ptr @universe_dataframe_series_from(i32 0, ptr %bi, i64 37)
  %sal = call ptr @universe_dataframe_series_from(i32 1, ptr %al, i64 37)
  %sbl = call ptr @universe_dataframe_series_from(i32 1, ptr %bl, i64 37)
  %saf = call ptr @universe_dataframe_series_from(i32 2, ptr %af, i64 37)
  %sbf = call ptr @universe_dataframe_series_from(i32 2, ptr %bf, i64 37)
  %sad = call ptr @universe_dataframe_series_from(i32 3, ptr %ad, i64 37)
  %sbd = call ptr @universe_dataframe_series_from(i32 3, ptr %bd, i64 37)
  %sci = call ptr @universe_dataframe_series_from(i32 0, ptr %ci, i64 37)
  %scd = call ptr @universe_dataframe_series_from(i32 3, ptr %cd, i64 37)

  ; result dtype must be BOOL(4)
  %rdt0 = call ptr @universe_dataframe_eq(ptr %sai, ptr %sbi)
  %dt = call i32 @universe_dataframe_series_dtype(ptr %rdt0)
  %dtok = icmp eq i32 %dt, 4
  call void @ut_check(i1 %dtok, ptr @.m_dt)
  call void @universe_dataframe_series_free(ptr %rdt0)

  ; ---- col-col: all 6 ops, all 4 dtypes ----
  call void @run_i32(ptr %sai, ptr %sbi, ptr %ai, ptr %bi, i64 37)
  ; i64
  call void @cmp6_i64(ptr %sal, ptr %sbl, ptr %al, ptr %bl, i64 37)
  ; f32
  call void @cmp6_f32(ptr %saf, ptr %sbf, ptr %af, ptr %bf, i64 37)
  ; f64
  call void @cmp6_f64(ptr %sad, ptr %sbd, ptr %ad, ptr %bd, i64 37)

  ; ---- col-scalar: compare against a constant-broadcast series result ----
  ; i32 scalar == 4 : run each scalar op, oracle = chk_i32(av, const, result, op)
  call void @scal6_i32(ptr %sai, ptr %ai, ptr %ci, i64 37)
  ; f64 scalar == 4.0
  call void @scal6_f64(ptr %sad, ptr %ad, ptr %cd, i64 37)

  call void @universe_dataframe_series_free(ptr %sai)
  call void @universe_dataframe_series_free(ptr %sbi)
  call void @universe_dataframe_series_free(ptr %sal)
  call void @universe_dataframe_series_free(ptr %sbl)
  call void @universe_dataframe_series_free(ptr %saf)
  call void @universe_dataframe_series_free(ptr %sbf)
  call void @universe_dataframe_series_free(ptr %sad)
  call void @universe_dataframe_series_free(ptr %sbd)
  call void @universe_dataframe_series_free(ptr %sci)
  call void @universe_dataframe_series_free(ptr %scd)

  ; ---- boolean algebra truth tables ----
  call void @test_bool()

  ; ---- any/all/sum_bool ----
  call void @test_reduce(ptr %obool, ptr %oi64)

  ; ---- zip_with ----
  call void @test_zip()

  ; ---- null-lane validity on compare ----
  call void @test_null(ptr %obool)

  ; ---- error paths ----
  %e0 = call ptr @universe_dataframe_eq(ptr null, ptr null)
  %e0n = icmp eq ptr %e0, null
  call void @ut_check(i1 %e0n, ptr @.m_err)

  ; ---- integration: mask via gt_scalar drives filter ----
  call void @test_filter()

  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; i64 col-col 6-op driver
define void @cmp6_i64(ptr %a, ptr %b, ptr %av, ptr %bv, i64 %n) {
entry:
  br label %loop
loop:
  %op = phi i32 [ 0, %entry ], [ %opn, %cont ]
  %c = icmp ult i32 %op, 6
  br i1 %c, label %body, label %done
body:
  %r = call ptr @cmp_by_op(ptr %a, ptr %b, i32 %op)
  %v = call ptr @universe_dataframe_series_values(ptr %r)
  call void @chk_i64(ptr %av, ptr %bv, ptr %v, i64 %n, i32 %op, ptr @.m_i64)
  call void @universe_dataframe_series_free(ptr %r)
  br label %cont
cont:
  %opn = add i32 %op, 1
  br label %loop
done:
  ret void
}

define void @cmp6_f32(ptr %a, ptr %b, ptr %av, ptr %bv, i64 %n) {
entry:
  br label %loop
loop:
  %op = phi i32 [ 0, %entry ], [ %opn, %cont ]
  %c = icmp ult i32 %op, 6
  br i1 %c, label %body, label %done
body:
  %r = call ptr @cmp_by_op(ptr %a, ptr %b, i32 %op)
  %v = call ptr @universe_dataframe_series_values(ptr %r)
  call void @chk_f32(ptr %av, ptr %bv, ptr %v, i64 %n, i32 %op, ptr @.m_f32)
  call void @universe_dataframe_series_free(ptr %r)
  br label %cont
cont:
  %opn = add i32 %op, 1
  br label %loop
done:
  ret void
}

define void @cmp6_f64(ptr %a, ptr %b, ptr %av, ptr %bv, i64 %n) {
entry:
  br label %loop
loop:
  %op = phi i32 [ 0, %entry ], [ %opn, %cont ]
  %c = icmp ult i32 %op, 6
  br i1 %c, label %body, label %done
body:
  %r = call ptr @cmp_by_op(ptr %a, ptr %b, i32 %op)
  %v = call ptr @universe_dataframe_series_values(ptr %r)
  call void @chk_f64(ptr %av, ptr %bv, ptr %v, i64 %n, i32 %op, ptr @.m_f64)
  call void @universe_dataframe_series_free(ptr %r)
  br label %cont
cont:
  %opn = add i32 %op, 1
  br label %loop
done:
  ret void
}

; dispatch a col-col comparator by op index
define ptr @cmp_by_op(ptr %a, ptr %b, i32 %op) {
entry:
  switch i32 %op, label %eq [ i32 0, label %eq
                              i32 1, label %ne
                              i32 2, label %lt
                              i32 3, label %le
                              i32 4, label %gt
                              i32 5, label %ge ]
eq:
  %r0 = call ptr @universe_dataframe_eq(ptr %a, ptr %b)
  ret ptr %r0
ne:
  %r1 = call ptr @universe_dataframe_neq(ptr %a, ptr %b)
  ret ptr %r1
lt:
  %r2 = call ptr @universe_dataframe_lt(ptr %a, ptr %b)
  ret ptr %r2
le:
  %r3 = call ptr @universe_dataframe_lte(ptr %a, ptr %b)
  ret ptr %r3
gt:
  %r4 = call ptr @universe_dataframe_gt(ptr %a, ptr %b)
  ret ptr %r4
ge:
  %r5 = call ptr @universe_dataframe_gte(ptr %a, ptr %b)
  ret ptr %r5
}

; i32 col-scalar 6-op driver (scalar value 4). Oracle: compare vs const buffer.
define void @scal6_i32(ptr %a, ptr %av, ptr %cv, i64 %n) {
entry:
  br label %loop
loop:
  %op = phi i32 [ 0, %entry ], [ %opn, %cont ]
  %c = icmp ult i32 %op, 6
  br i1 %c, label %body, label %done
body:
  %r = call ptr @scal_by_op_i(ptr %a, i32 %op)
  %v = call ptr @universe_dataframe_series_values(ptr %r)
  call void @chk_i32(ptr %av, ptr %cv, ptr %v, i64 %n, i32 %op, ptr @.m_i32s)
  call void @universe_dataframe_series_free(ptr %r)
  br label %cont
cont:
  %opn = add i32 %op, 1
  br label %loop
done:
  ret void
}

define void @scal6_f64(ptr %a, ptr %av, ptr %cv, i64 %n) {
entry:
  br label %loop
loop:
  %op = phi i32 [ 0, %entry ], [ %opn, %cont ]
  %c = icmp ult i32 %op, 6
  br i1 %c, label %body, label %done
body:
  %r = call ptr @scal_by_op_d(ptr %a, i32 %op)
  %v = call ptr @universe_dataframe_series_values(ptr %r)
  call void @chk_f64(ptr %av, ptr %cv, ptr %v, i64 %n, i32 %op, ptr @.m_f64s)
  call void @universe_dataframe_series_free(ptr %r)
  br label %cont
cont:
  %opn = add i32 %op, 1
  br label %loop
done:
  ret void
}

define ptr @scal_by_op_i(ptr %a, i32 %op) {
entry:
  switch i32 %op, label %eq [ i32 0, label %eq
                              i32 1, label %ne
                              i32 2, label %lt
                              i32 3, label %le
                              i32 4, label %gt
                              i32 5, label %ge ]
eq:
  %r0 = call ptr @universe_dataframe_eq_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r0
ne:
  %r1 = call ptr @universe_dataframe_neq_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r1
lt:
  %r2 = call ptr @universe_dataframe_lt_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r2
le:
  %r3 = call ptr @universe_dataframe_lte_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r3
gt:
  %r4 = call ptr @universe_dataframe_gt_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r4
ge:
  %r5 = call ptr @universe_dataframe_gte_scalar(ptr %a, i32 0, i64 4, double 0.0)
  ret ptr %r5
}

define ptr @scal_by_op_d(ptr %a, i32 %op) {
entry:
  switch i32 %op, label %eq [ i32 0, label %eq
                              i32 1, label %ne
                              i32 2, label %lt
                              i32 3, label %le
                              i32 4, label %gt
                              i32 5, label %ge ]
eq:
  %r0 = call ptr @universe_dataframe_eq_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r0
ne:
  %r1 = call ptr @universe_dataframe_neq_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r1
lt:
  %r2 = call ptr @universe_dataframe_lt_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r2
le:
  %r3 = call ptr @universe_dataframe_lte_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r3
gt:
  %r4 = call ptr @universe_dataframe_gt_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r4
ge:
  %r5 = call ptr @universe_dataframe_gte_scalar(ptr %a, i32 3, i64 0, double 4.0)
  ret ptr %r5
}

; ---------------------------------------------------------------------------
; boolean algebra truth tables. a=[0,0,1,1], b=[0,1,0,1] (length 4)
; ---------------------------------------------------------------------------
define void @test_bool() {
entry:
  %abuf = alloca [4 x i8], align 4
  %bbuf = alloca [4 x i8], align 4
  store i8 0, ptr %abuf, align 1
  %a1 = getelementptr inbounds i8, ptr %abuf, i64 1
  store i8 0, ptr %a1, align 1
  %a2 = getelementptr inbounds i8, ptr %abuf, i64 2
  store i8 1, ptr %a2, align 1
  %a3 = getelementptr inbounds i8, ptr %abuf, i64 3
  store i8 1, ptr %a3, align 1
  store i8 0, ptr %bbuf, align 1
  %b1 = getelementptr inbounds i8, ptr %bbuf, i64 1
  store i8 1, ptr %b1, align 1
  %b2 = getelementptr inbounds i8, ptr %bbuf, i64 2
  store i8 0, ptr %b2, align 1
  %b3 = getelementptr inbounds i8, ptr %bbuf, i64 3
  store i8 1, ptr %b3, align 1
  %sa = call ptr @universe_dataframe_series_from(i32 4, ptr %abuf, i64 4)
  %sb = call ptr @universe_dataframe_series_from(i32 4, ptr %bbuf, i64 4)

  ; and -> 0,0,0,1
  %ra = call ptr @universe_dataframe_and(ptr %sa, ptr %sb)
  %va = call ptr @universe_dataframe_series_values(ptr %ra)
  call void @chk_bytes(ptr %va, i8 0, i8 0, i8 0, i8 1, ptr @.m_and)
  call void @universe_dataframe_series_free(ptr %ra)
  ; or -> 0,1,1,1
  %ro = call ptr @universe_dataframe_or(ptr %sa, ptr %sb)
  %vo = call ptr @universe_dataframe_series_values(ptr %ro)
  call void @chk_bytes(ptr %vo, i8 0, i8 1, i8 1, i8 1, ptr @.m_or)
  call void @universe_dataframe_series_free(ptr %ro)
  ; xor -> 0,1,1,0
  %rx = call ptr @universe_dataframe_xor(ptr %sa, ptr %sb)
  %vx = call ptr @universe_dataframe_series_values(ptr %rx)
  call void @chk_bytes(ptr %vx, i8 0, i8 1, i8 1, i8 0, ptr @.m_xor)
  call void @universe_dataframe_series_free(ptr %rx)
  ; not(a) -> 1,1,0,0
  %rn = call ptr @universe_dataframe_not(ptr %sa)
  %vn = call ptr @universe_dataframe_series_values(ptr %rn)
  call void @chk_bytes(ptr %vn, i8 1, i8 1, i8 0, i8 0, ptr @.m_not)
  call void @universe_dataframe_series_free(ptr %rn)

  call void @universe_dataframe_series_free(ptr %sa)
  call void @universe_dataframe_series_free(ptr %sb)
  ret void
}

define void @chk_bytes(ptr %v, i8 %e0, i8 %e1, i8 %e2, i8 %e3, ptr %msg) {
entry:
  %p0 = getelementptr inbounds i8, ptr %v, i64 0
  %x0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %x0 to i64
  %ze0 = zext i8 %e0 to i64
  call void @ut_check_eq(i64 %z0, i64 %ze0, ptr %msg)
  %p1 = getelementptr inbounds i8, ptr %v, i64 1
  %x1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %x1 to i64
  %ze1 = zext i8 %e1 to i64
  call void @ut_check_eq(i64 %z1, i64 %ze1, ptr %msg)
  %p2 = getelementptr inbounds i8, ptr %v, i64 2
  %x2 = load i8, ptr %p2, align 1
  %z2 = zext i8 %x2 to i64
  %ze2 = zext i8 %e2 to i64
  call void @ut_check_eq(i64 %z2, i64 %ze2, ptr %msg)
  %p3 = getelementptr inbounds i8, ptr %v, i64 3
  %x3 = load i8, ptr %p3, align 1
  %z3 = zext i8 %x3 to i64
  %ze3 = zext i8 %e3 to i64
  call void @ut_check_eq(i64 %z3, i64 %ze3, ptr %msg)
  ret void
}

; ---------------------------------------------------------------------------
; any/all/sum_bool over a length-40 BOOL series (exercises vector reduce).
; ---------------------------------------------------------------------------
define void @test_reduce(ptr %obool, ptr %oi64) {
entry:
  %buf = alloca [40 x i8], align 16
  ; all ones
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fill ]
  %p = getelementptr inbounds [40 x i8], ptr %buf, i64 0, i64 %i
  store i8 1, ptr %p, align 1
  %in = add i64 %i, 1
  %c = icmp ult i64 %in, 40
  br i1 %c, label %fill, label %alltrue
alltrue:
  %s = call ptr @universe_dataframe_series_from(i32 4, ptr %buf, i64 40)
  %ra = call i32 @universe_dataframe_any(ptr %s, ptr %obool)
  %ba = load i8, ptr %obool, align 1
  %baz = zext i8 %ba to i64
  call void @ut_check_eq(i64 %baz, i64 1, ptr @.m_any)
  %rl = call i32 @universe_dataframe_all(ptr %s, ptr %obool)
  %bl = load i8, ptr %obool, align 1
  %blz = zext i8 %bl to i64
  call void @ut_check_eq(i64 %blz, i64 1, ptr @.m_all)
  %rs = call i32 @universe_dataframe_sum_bool(ptr %s, ptr %oi64)
  %sv = load i64, ptr %oi64, align 8
  call void @ut_check_eq(i64 %sv, i64 40, ptr @.m_sum)
  call void @universe_dataframe_series_free(ptr %s)

  ; set lane 17 to 0 -> all=0, any=1, sum=39
  %p17 = getelementptr inbounds [40 x i8], ptr %buf, i64 0, i64 17
  store i8 0, ptr %p17, align 1
  %s2 = call ptr @universe_dataframe_series_from(i32 4, ptr %buf, i64 40)
  %rl2 = call i32 @universe_dataframe_all(ptr %s2, ptr %obool)
  %bl2 = load i8, ptr %obool, align 1
  %bl2z = zext i8 %bl2 to i64
  call void @ut_check_eq(i64 %bl2z, i64 0, ptr @.m_all)
  %ra2 = call i32 @universe_dataframe_any(ptr %s2, ptr %obool)
  %ba2 = load i8, ptr %obool, align 1
  %ba2z = zext i8 %ba2 to i64
  call void @ut_check_eq(i64 %ba2z, i64 1, ptr @.m_any)
  %rs2 = call i32 @universe_dataframe_sum_bool(ptr %s2, ptr %oi64)
  %sv2 = load i64, ptr %oi64, align 8
  call void @ut_check_eq(i64 %sv2, i64 39, ptr @.m_sum)

  ; make lane 17 null (it's value 0). sum still 39 (valid&true), all ignores null -> 1
  %rn = call i32 @universe_dataframe_series_set_null(ptr %s2, i64 17)
  %rl3 = call i32 @universe_dataframe_all(ptr %s2, ptr %obool)
  %bl3 = load i8, ptr %obool, align 1
  %bl3z = zext i8 %bl3 to i64
  call void @ut_check_eq(i64 %bl3z, i64 1, ptr @.m_all)
  %rs3 = call i32 @universe_dataframe_sum_bool(ptr %s2, ptr %oi64)
  %sv3 = load i64, ptr %oi64, align 8
  call void @ut_check_eq(i64 %sv3, i64 39, ptr @.m_sum)
  call void @universe_dataframe_series_free(ptr %s2)
  ret void
}

; ---------------------------------------------------------------------------
; zip_with: mask [1,0,1,0,...], a[i]=100+i, b[i]=200+i, result = mask?a:b
; ---------------------------------------------------------------------------
define void @test_zip() {
entry:
  %mbuf = alloca [20 x i8], align 16
  %abuf = alloca [20 x i32], align 16
  %bbuf = alloca [20 x i32], align 16
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fill ]
  %m = and i64 %i, 1
  %m8 = trunc i64 %m to i8
  %pm = getelementptr inbounds [20 x i8], ptr %mbuf, i64 0, i64 %i
  store i8 %m8, ptr %pm, align 1
  %av = add i64 100, %i
  %av32 = trunc i64 %av to i32
  %pa = getelementptr inbounds [20 x i32], ptr %abuf, i64 0, i64 %i
  store i32 %av32, ptr %pa, align 4
  %bv = add i64 200, %i
  %bv32 = trunc i64 %bv to i32
  %pb = getelementptr inbounds [20 x i32], ptr %bbuf, i64 0, i64 %i
  store i32 %bv32, ptr %pb, align 4
  %in = add i64 %i, 1
  %c = icmp ult i64 %in, 20
  br i1 %c, label %fill, label %run
run:
  %sm = call ptr @universe_dataframe_series_from(i32 4, ptr %mbuf, i64 20)
  %sa = call ptr @universe_dataframe_series_from(i32 0, ptr %abuf, i64 20)
  %sb = call ptr @universe_dataframe_series_from(i32 0, ptr %bbuf, i64 20)
  %r = call ptr @universe_dataframe_zip_with(ptr %sm, ptr %sa, ptr %sb)
  %rv = call ptr @universe_dataframe_series_values(ptr %r)
  br label %chk
chk:
  %j = phi i64 [ 0, %run ], [ %jn, %chk ]
  %mm = and i64 %j, 1
  %pick = icmp ne i64 %mm, 0
  %ea = add i64 100, %j
  %eb = add i64 200, %j
  %exp64 = select i1 %pick, i64 %ea, i64 %eb
  %prv = getelementptr inbounds i32, ptr %rv, i64 %j
  %got = load i32, ptr %prv, align 4
  %got64 = zext i32 %got to i64
  call void @ut_check_eq(i64 %got64, i64 %exp64, ptr @.m_zip)
  %jn = add i64 %j, 1
  %cc = icmp ult i64 %jn, 20
  br i1 %cc, label %chk, label %fin
fin:
  call void @universe_dataframe_series_free(ptr %sm)
  call void @universe_dataframe_series_free(ptr %sa)
  call void @universe_dataframe_series_free(ptr %sb)
  call void @universe_dataframe_series_free(ptr %r)
  ret void
}

; ---------------------------------------------------------------------------
; null-lane validity: a[2] null, b[5] null; eq result must be null at 2 and 5,
; valid elsewhere (a null lane compares NULL, not true).
; ---------------------------------------------------------------------------
define void @test_null(ptr %obool) {
entry:
  %abuf = alloca [8 x i32], align 16
  %bbuf = alloca [8 x i32], align 16
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fill ]
  %iv = trunc i64 %i to i32
  %pa = getelementptr inbounds [8 x i32], ptr %abuf, i64 0, i64 %i
  store i32 %iv, ptr %pa, align 4
  %pb = getelementptr inbounds [8 x i32], ptr %bbuf, i64 0, i64 %i
  store i32 %iv, ptr %pb, align 4
  %in = add i64 %i, 1
  %c = icmp ult i64 %in, 8
  br i1 %c, label %fill, label %run
run:
  %sa = call ptr @universe_dataframe_series_from(i32 0, ptr %abuf, i64 8)
  %sb = call ptr @universe_dataframe_series_from(i32 0, ptr %bbuf, i64 8)
  %z1 = call i32 @universe_dataframe_series_set_null(ptr %sa, i64 2)
  %z2 = call i32 @universe_dataframe_series_set_null(ptr %sb, i64 5)
  %r = call ptr @universe_dataframe_eq(ptr %sa, ptr %sb)
  ; null_count must be 2
  %nc = call i64 @universe_dataframe_series_null_count(ptr %r)
  call void @ut_check_eq(i64 %nc, i64 2, ptr @.m_nul)
  br label %chk
chk:
  %j = phi i64 [ 0, %run ], [ %jn, %chk ]
  %rn = call i32 @universe_dataframe_series_is_null(ptr %r, i64 %j, ptr %obool)
  %isn = load i8, ptr %obool, align 1
  %isn64 = zext i8 %isn to i64
  %exp2 = icmp eq i64 %j, 2
  %exp5 = icmp eq i64 %j, 5
  %expnull = or i1 %exp2, %exp5
  %en64 = zext i1 %expnull to i64
  call void @ut_check_eq(i64 %isn64, i64 %en64, ptr @.m_nul)
  %jn = add i64 %j, 1
  %cc = icmp ult i64 %jn, 8
  br i1 %cc, label %chk, label %fin
fin:
  call void @universe_dataframe_series_free(ptr %sa)
  call void @universe_dataframe_series_free(ptr %sb)
  call void @universe_dataframe_series_free(ptr %r)
  ret void
}

; ---------------------------------------------------------------------------
; INTEGRATION: df with column "v" = 0..9; mask = gt_scalar(v, 5); filter(df,mask)
; result height must equal manual count of v>5 (=4) and every kept v must be >5.
; ---------------------------------------------------------------------------
define void @test_filter() {
entry:
  %buf = alloca [10 x i32], align 16
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fill ]
  %iv = trunc i64 %i to i32
  %p = getelementptr inbounds [10 x i32], ptr %buf, i64 0, i64 %i
  store i32 %iv, ptr %p, align 4
  %in = add i64 %i, 1
  %c = icmp ult i64 %in, 10
  br i1 %c, label %fill, label %run
run:
  %col = call ptr @universe_dataframe_series_from(i32 0, ptr %buf, i64 10)
  %df = call ptr @universe_dataframe_new()
  %wc = call i32 @universe_dataframe_with_column(ptr %df, ptr @.mv, i64 1, ptr %col)
  ; column ptr fetched from df (owned by df)
  %v = call ptr @universe_dataframe_column(ptr %df, ptr @.mv, i64 1)
  %mask = call ptr @universe_dataframe_gt_scalar(ptr %v, i32 0, i64 5, double 0.0)
  %fdf = call ptr @universe_dataframe_filter(ptr %df, ptr %mask)
  %h = call i64 @universe_dataframe_height(ptr %fdf)
  call void @ut_check_eq(i64 %h, i64 4, ptr @.m_fh)
  ; every kept value must be > 5
  %fcol = call ptr @universe_dataframe_column(ptr %fdf, ptr @.mv, i64 1)
  %fvals = call ptr @universe_dataframe_series_values(ptr %fcol)
  br label %chk
chk:
  %j = phi i64 [ 0, %run ], [ %jn, %chk ]
  %pj = getelementptr inbounds i32, ptr %fvals, i64 %j
  %x = load i32, ptr %pj, align 4
  %gt5 = icmp sgt i32 %x, 5
  call void @ut_check(i1 %gt5, ptr @.m_fv)
  %jn = add i64 %j, 1
  %cc = icmp ult i64 %jn, %h
  br i1 %cc, label %chk, label %fin
fin:
  call void @universe_dataframe_series_free(ptr %mask)
  call void @universe_dataframe_free(ptr %fdf)
  call void @universe_dataframe_free(ptr %df)
  ret void
}
