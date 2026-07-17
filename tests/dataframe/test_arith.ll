; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/arith.ll — SIMD element-wise arithmetic.
; The SCALAR path IS the oracle: for every dtype the vector kernel result is
; cross-checked element-by-element against an in-IR scalar computation of the
; SAME op (exact for int/logical, 1e-5 rel for float). Also: null propagation
; (out null = union of input nulls), int div-by-zero => null, scalar broadcast,
; unary neg/abs, horizontal min/max, edge sizes 0/1/odd/16/17/large.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @printf(ptr, ...)
declare double @llvm.fabs.f64(double)

; frame.ll
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_values(ptr)
declare void @universe_dataframe_series_free(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare ptr @universe_dataframe_new()
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare void @universe_dataframe_free(ptr)

; module under test
declare ptr @universe_dataframe_add(ptr, ptr)
declare ptr @universe_dataframe_sub(ptr, ptr)
declare ptr @universe_dataframe_mul(ptr, ptr)
declare ptr @universe_dataframe_div(ptr, ptr)
declare ptr @universe_dataframe_rem(ptr, ptr)
declare ptr @universe_dataframe_neg(ptr)
declare ptr @universe_dataframe_abs(ptr)
declare ptr @universe_dataframe_add_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_sub_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_mul_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_div_scalar(ptr, i32, i64, double)
declare ptr @universe_dataframe_min_horizontal(ptr)
declare ptr @universe_dataframe_max_horizontal(ptr)

@.m_iadd = private constant [9 x i8]  c"i add ok\00"
@.m_isub = private constant [9 x i8]  c"i sub ok\00"
@.m_imul = private constant [9 x i8]  c"i mul ok\00"
@.m_idiv = private constant [9 x i8]  c"i div ok\00"
@.m_irem = private constant [9 x i8]  c"i rem ok\00"
@.m_fadd = private constant [9 x i8]  c"f add ok\00"
@.m_fsub = private constant [9 x i8]  c"f sub ok\00"
@.m_fmul = private constant [9 x i8]  c"f mul ok\00"
@.m_fdiv = private constant [9 x i8]  c"f div ok\00"
@.m_frem = private constant [9 x i8]  c"f rem ok\00"
@.m_sadd = private constant [12 x i8] c"scalar add \00"
@.m_smul = private constant [12 x i8] c"scalar mul \00"
@.m_neg  = private constant [7 x i8]  c"neg ok\00"
@.m_abs  = private constant [7 x i8]  c"abs ok\00"
@.m_np   = private constant [13 x i8] c"null propag \00"
@.m_nv   = private constant [13 x i8] c"non-null pos\00"
@.m_dz   = private constant [14 x i8] c"divzero null \00"
@.m_dznc = private constant [12 x i8] c"divzero cnt\00"
@.m_hmin = private constant [9 x i8]  c"hmin val\00"
@.m_hmax = private constant [9 x i8]  c"hmax val\00"
@.m_hnul = private constant [10 x i8] c"hrow null\00"
@.m_len  = private constant [9 x i8]  c"out len \00"
@.c0     = private constant [2 x i8]  c"x\00"
@.c1     = private constant [2 x i8]  c"y\00"

; |a-b| <= tol*(|b|+1)
define internal void @check_close(double %a, double %b, ptr %msg) {
entry:
  %d = fsub double %a, %b
  %ad = call double @llvm.fabs.f64(double %d)
  %ab = call double @llvm.fabs.f64(double %b)
  %scale = fadd double %ab, 1.0
  %tol = fmul double %scale, 1.000000e-05
  %ok = fcmp ole double %ad, %tol
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; ============================================================ i32
define internal void @test_i32(i64 %len, ptr %seed) {
entry:
  %a = call ptr @universe_dataframe_series_new(i32 0, i64 %len)
  %b = call ptr @universe_dataframe_series_new(i32 0, i64 %len)
  %av = call ptr @universe_dataframe_series_values(ptr %a)
  %bv = call ptr @universe_dataframe_series_values(ptr %b)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fbody ]
  %fd = icmp uge i64 %i, %len
  br i1 %fd, label %run, label %fbody
fbody:
  %r1 = call i64 @ut_rand(ptr %seed)
  %r2 = call i64 @ut_rand(ptr %seed)
  %m1 = urem i64 %r1, 1001
  %xa64 = sub i64 %m1, 500
  %xa = trunc i64 %xa64 to i32
  %m2 = urem i64 %r2, 500
  %xb64 = add i64 %m2, 1
  %xb = trunc i64 %xb64 to i32
  %pa = getelementptr inbounds i32, ptr %av, i64 %i
  store i32 %xa, ptr %pa, align 4
  %pb = getelementptr inbounds i32, ptr %bv, i64 %i
  store i32 %xb, ptr %pb, align 4
  %in = add nuw i64 %i, 1
  br label %fill
run:
  ; add
  %oadd = call ptr @universe_dataframe_add(ptr %a, ptr %b)
  %oaddv = call ptr @universe_dataframe_series_values(ptr %oadd)
  br label %ca
ca:
  %cai = phi i64 [ 0, %run ], [ %cain, %cab ]
  %cad = icmp uge i64 %cai, %len
  br i1 %cad, label %ca.d, label %cab
cab:
  %ca.pa = getelementptr inbounds i32, ptr %av, i64 %cai
  %ca.xa = load i32, ptr %ca.pa, align 4
  %ca.pb = getelementptr inbounds i32, ptr %bv, i64 %cai
  %ca.xb = load i32, ptr %ca.pb, align 4
  %ca.e = add i32 %ca.xa, %ca.xb
  %ca.po = getelementptr inbounds i32, ptr %oaddv, i64 %cai
  %ca.g = load i32, ptr %ca.po, align 4
  %ca.es = sext i32 %ca.e to i64
  %ca.gs = sext i32 %ca.g to i64
  call void @ut_check_eq(i64 %ca.gs, i64 %ca.es, ptr @.m_iadd)
  %cain = add nuw i64 %cai, 1
  br label %ca
ca.d:
  call void @universe_dataframe_series_free(ptr %oadd)
  ; sub
  %osub = call ptr @universe_dataframe_sub(ptr %a, ptr %b)
  %osubv = call ptr @universe_dataframe_series_values(ptr %osub)
  br label %cs
cs:
  %csi = phi i64 [ 0, %ca.d ], [ %csin, %csb ]
  %csd = icmp uge i64 %csi, %len
  br i1 %csd, label %cs.d, label %csb
csb:
  %cs.pa = getelementptr inbounds i32, ptr %av, i64 %csi
  %cs.xa = load i32, ptr %cs.pa, align 4
  %cs.pb = getelementptr inbounds i32, ptr %bv, i64 %csi
  %cs.xb = load i32, ptr %cs.pb, align 4
  %cs.e = sub i32 %cs.xa, %cs.xb
  %cs.po = getelementptr inbounds i32, ptr %osubv, i64 %csi
  %cs.g = load i32, ptr %cs.po, align 4
  %cs.es = sext i32 %cs.e to i64
  %cs.gs = sext i32 %cs.g to i64
  call void @ut_check_eq(i64 %cs.gs, i64 %cs.es, ptr @.m_isub)
  %csin = add nuw i64 %csi, 1
  br label %cs
cs.d:
  call void @universe_dataframe_series_free(ptr %osub)
  ; mul
  %omul = call ptr @universe_dataframe_mul(ptr %a, ptr %b)
  %omulv = call ptr @universe_dataframe_series_values(ptr %omul)
  br label %cm
cm:
  %cmi = phi i64 [ 0, %cs.d ], [ %cmin, %cmb ]
  %cmd = icmp uge i64 %cmi, %len
  br i1 %cmd, label %cm.d, label %cmb
cmb:
  %cm.pa = getelementptr inbounds i32, ptr %av, i64 %cmi
  %cm.xa = load i32, ptr %cm.pa, align 4
  %cm.pb = getelementptr inbounds i32, ptr %bv, i64 %cmi
  %cm.xb = load i32, ptr %cm.pb, align 4
  %cm.e = mul i32 %cm.xa, %cm.xb
  %cm.po = getelementptr inbounds i32, ptr %omulv, i64 %cmi
  %cm.g = load i32, ptr %cm.po, align 4
  %cm.es = sext i32 %cm.e to i64
  %cm.gs = sext i32 %cm.g to i64
  call void @ut_check_eq(i64 %cm.gs, i64 %cm.es, ptr @.m_imul)
  %cmin = add nuw i64 %cmi, 1
  br label %cm
cm.d:
  call void @universe_dataframe_series_free(ptr %omul)
  ; div
  %odiv = call ptr @universe_dataframe_div(ptr %a, ptr %b)
  %odivv = call ptr @universe_dataframe_series_values(ptr %odiv)
  br label %cd
cd:
  %cdi = phi i64 [ 0, %cm.d ], [ %cdin, %cdb ]
  %cdd = icmp uge i64 %cdi, %len
  br i1 %cdd, label %cd.d, label %cdb
cdb:
  %cd.pa = getelementptr inbounds i32, ptr %av, i64 %cdi
  %cd.xa = load i32, ptr %cd.pa, align 4
  %cd.pb = getelementptr inbounds i32, ptr %bv, i64 %cdi
  %cd.xb = load i32, ptr %cd.pb, align 4
  %cd.e = sdiv i32 %cd.xa, %cd.xb
  %cd.po = getelementptr inbounds i32, ptr %odivv, i64 %cdi
  %cd.g = load i32, ptr %cd.po, align 4
  %cd.es = sext i32 %cd.e to i64
  %cd.gs = sext i32 %cd.g to i64
  call void @ut_check_eq(i64 %cd.gs, i64 %cd.es, ptr @.m_idiv)
  %cdin = add nuw i64 %cdi, 1
  br label %cd
cd.d:
  call void @universe_dataframe_series_free(ptr %odiv)
  ; rem
  %orem = call ptr @universe_dataframe_rem(ptr %a, ptr %b)
  %oremv = call ptr @universe_dataframe_series_values(ptr %orem)
  br label %cr
cr:
  %cri = phi i64 [ 0, %cd.d ], [ %crin, %crb ]
  %crd = icmp uge i64 %cri, %len
  br i1 %crd, label %cr.d, label %crb
crb:
  %cr.pa = getelementptr inbounds i32, ptr %av, i64 %cri
  %cr.xa = load i32, ptr %cr.pa, align 4
  %cr.pb = getelementptr inbounds i32, ptr %bv, i64 %cri
  %cr.xb = load i32, ptr %cr.pb, align 4
  %cr.e = srem i32 %cr.xa, %cr.xb
  %cr.po = getelementptr inbounds i32, ptr %oremv, i64 %cri
  %cr.g = load i32, ptr %cr.po, align 4
  %cr.es = sext i32 %cr.e to i64
  %cr.gs = sext i32 %cr.g to i64
  call void @ut_check_eq(i64 %cr.gs, i64 %cr.es, ptr @.m_irem)
  %crin = add nuw i64 %cri, 1
  br label %cr
cr.d:
  call void @universe_dataframe_series_free(ptr %orem)
  ; scalar add 7, mul 3
  %osa = call ptr @universe_dataframe_add_scalar(ptr %a, i32 0, i64 7, double 0.0)
  %osav = call ptr @universe_dataframe_series_values(ptr %osa)
  %osm = call ptr @universe_dataframe_mul_scalar(ptr %a, i32 0, i64 3, double 0.0)
  %osmv = call ptr @universe_dataframe_series_values(ptr %osm)
  ; neg, abs
  %oneg = call ptr @universe_dataframe_neg(ptr %a)
  %onegv = call ptr @universe_dataframe_series_values(ptr %oneg)
  %oabs = call ptr @universe_dataframe_abs(ptr %a)
  %oabsv = call ptr @universe_dataframe_series_values(ptr %oabs)
  br label %cu
cu:
  %cui = phi i64 [ 0, %cr.d ], [ %cuin, %cub ]
  %cud = icmp uge i64 %cui, %len
  br i1 %cud, label %cu.d, label %cub
cub:
  %cu.pa = getelementptr inbounds i32, ptr %av, i64 %cui
  %cu.xa = load i32, ptr %cu.pa, align 4
  ; scalar add
  %cu.sae = add i32 %cu.xa, 7
  %cu.sap = getelementptr inbounds i32, ptr %osav, i64 %cui
  %cu.sag = load i32, ptr %cu.sap, align 4
  %cu.sae64 = sext i32 %cu.sae to i64
  %cu.sag64 = sext i32 %cu.sag to i64
  call void @ut_check_eq(i64 %cu.sag64, i64 %cu.sae64, ptr @.m_sadd)
  ; scalar mul
  %cu.sme = mul i32 %cu.xa, 3
  %cu.smp = getelementptr inbounds i32, ptr %osmv, i64 %cui
  %cu.smg = load i32, ptr %cu.smp, align 4
  %cu.sme64 = sext i32 %cu.sme to i64
  %cu.smg64 = sext i32 %cu.smg to i64
  call void @ut_check_eq(i64 %cu.smg64, i64 %cu.sme64, ptr @.m_smul)
  ; neg
  %cu.ne = sub i32 0, %cu.xa
  %cu.np = getelementptr inbounds i32, ptr %onegv, i64 %cui
  %cu.ng = load i32, ptr %cu.np, align 4
  %cu.ne64 = sext i32 %cu.ne to i64
  %cu.ng64 = sext i32 %cu.ng to i64
  call void @ut_check_eq(i64 %cu.ng64, i64 %cu.ne64, ptr @.m_neg)
  ; abs
  %cu.neg = icmp slt i32 %cu.xa, 0
  %cu.negd = sub i32 0, %cu.xa
  %cu.abe = select i1 %cu.neg, i32 %cu.negd, i32 %cu.xa
  %cu.abp = getelementptr inbounds i32, ptr %oabsv, i64 %cui
  %cu.abg = load i32, ptr %cu.abp, align 4
  %cu.abe64 = sext i32 %cu.abe to i64
  %cu.abg64 = sext i32 %cu.abg to i64
  call void @ut_check_eq(i64 %cu.abg64, i64 %cu.abe64, ptr @.m_abs)
  %cuin = add nuw i64 %cui, 1
  br label %cu
cu.d:
  call void @universe_dataframe_series_free(ptr %osa)
  call void @universe_dataframe_series_free(ptr %osm)
  call void @universe_dataframe_series_free(ptr %oneg)
  call void @universe_dataframe_series_free(ptr %oabs)
  call void @universe_dataframe_series_free(ptr %a)
  call void @universe_dataframe_series_free(ptr %b)
  ret void
}

; ============================================================ i64
define internal void @test_i64(i64 %len, ptr %seed) {
entry:
  %a = call ptr @universe_dataframe_series_new(i32 1, i64 %len)
  %b = call ptr @universe_dataframe_series_new(i32 1, i64 %len)
  %av = call ptr @universe_dataframe_series_values(ptr %a)
  %bv = call ptr @universe_dataframe_series_values(ptr %b)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fbody ]
  %fd = icmp uge i64 %i, %len
  br i1 %fd, label %run, label %fbody
fbody:
  %r1 = call i64 @ut_rand(ptr %seed)
  %r2 = call i64 @ut_rand(ptr %seed)
  %m1 = urem i64 %r1, 200001
  %xa = sub i64 %m1, 100000
  %m2 = urem i64 %r2, 1000
  %xb = add i64 %m2, 1
  %pa = getelementptr inbounds i64, ptr %av, i64 %i
  store i64 %xa, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %bv, i64 %i
  store i64 %xb, ptr %pb, align 8
  %in = add nuw i64 %i, 1
  br label %fill
run:
  %oadd = call ptr @universe_dataframe_add(ptr %a, ptr %b)
  %oaddv = call ptr @universe_dataframe_series_values(ptr %oadd)
  %omul = call ptr @universe_dataframe_mul(ptr %a, ptr %b)
  %omulv = call ptr @universe_dataframe_series_values(ptr %omul)
  %odiv = call ptr @universe_dataframe_div(ptr %a, ptr %b)
  %odivv = call ptr @universe_dataframe_series_values(ptr %odiv)
  %orem = call ptr @universe_dataframe_rem(ptr %a, ptr %b)
  %oremv = call ptr @universe_dataframe_series_values(ptr %orem)
  br label %chk
chk:
  %i2 = phi i64 [ 0, %run ], [ %i2n, %cb ]
  %d2 = icmp uge i64 %i2, %len
  br i1 %d2, label %fin, label %cb
cb:
  %pa2 = getelementptr inbounds i64, ptr %av, i64 %i2
  %xa2 = load i64, ptr %pa2, align 8
  %pb2 = getelementptr inbounds i64, ptr %bv, i64 %i2
  %xb2 = load i64, ptr %pb2, align 8
  %eadd = add i64 %xa2, %xb2
  %gaddp = getelementptr inbounds i64, ptr %oaddv, i64 %i2
  %gadd = load i64, ptr %gaddp, align 8
  call void @ut_check_eq(i64 %gadd, i64 %eadd, ptr @.m_iadd)
  %emul = mul i64 %xa2, %xb2
  %gmulp = getelementptr inbounds i64, ptr %omulv, i64 %i2
  %gmul = load i64, ptr %gmulp, align 8
  call void @ut_check_eq(i64 %gmul, i64 %emul, ptr @.m_imul)
  %ediv = sdiv i64 %xa2, %xb2
  %gdivp = getelementptr inbounds i64, ptr %odivv, i64 %i2
  %gdiv = load i64, ptr %gdivp, align 8
  call void @ut_check_eq(i64 %gdiv, i64 %ediv, ptr @.m_idiv)
  %erem = srem i64 %xa2, %xb2
  %gremp = getelementptr inbounds i64, ptr %oremv, i64 %i2
  %grem = load i64, ptr %gremp, align 8
  call void @ut_check_eq(i64 %grem, i64 %erem, ptr @.m_irem)
  %i2n = add nuw i64 %i2, 1
  br label %chk
fin:
  call void @universe_dataframe_series_free(ptr %oadd)
  call void @universe_dataframe_series_free(ptr %omul)
  call void @universe_dataframe_series_free(ptr %odiv)
  call void @universe_dataframe_series_free(ptr %orem)
  call void @universe_dataframe_series_free(ptr %a)
  call void @universe_dataframe_series_free(ptr %b)
  ret void
}

; ============================================================ f32
define internal void @test_f32(i64 %len, ptr %seed) {
entry:
  %a = call ptr @universe_dataframe_series_new(i32 2, i64 %len)
  %b = call ptr @universe_dataframe_series_new(i32 2, i64 %len)
  %av = call ptr @universe_dataframe_series_values(ptr %a)
  %bv = call ptr @universe_dataframe_series_values(ptr %b)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fbody ]
  %fd = icmp uge i64 %i, %len
  br i1 %fd, label %run, label %fbody
fbody:
  %r1 = call i64 @ut_rand(ptr %seed)
  %r2 = call i64 @ut_rand(ptr %seed)
  %m1 = urem i64 %r1, 20000
  %m1s = sub i64 %m1, 10000
  %xa64 = sitofp i64 %m1s to double
  %xa = fptrunc double %xa64 to float
  %m2 = urem i64 %r2, 1000
  %m2a = add i64 %m2, 1
  %xb64 = sitofp i64 %m2a to double
  %xb = fptrunc double %xb64 to float
  %pa = getelementptr inbounds float, ptr %av, i64 %i
  store float %xa, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %bv, i64 %i
  store float %xb, ptr %pb, align 4
  %in = add nuw i64 %i, 1
  br label %fill
run:
  %oadd = call ptr @universe_dataframe_add(ptr %a, ptr %b)
  %oaddv = call ptr @universe_dataframe_series_values(ptr %oadd)
  %osub = call ptr @universe_dataframe_sub(ptr %a, ptr %b)
  %osubv = call ptr @universe_dataframe_series_values(ptr %osub)
  %omul = call ptr @universe_dataframe_mul(ptr %a, ptr %b)
  %omulv = call ptr @universe_dataframe_series_values(ptr %omul)
  %odiv = call ptr @universe_dataframe_div(ptr %a, ptr %b)
  %odivv = call ptr @universe_dataframe_series_values(ptr %odiv)
  br label %chk
chk:
  %i2 = phi i64 [ 0, %run ], [ %i2n, %cb ]
  %d2 = icmp uge i64 %i2, %len
  br i1 %d2, label %fin, label %cb
cb:
  %pa2 = getelementptr inbounds float, ptr %av, i64 %i2
  %xa2 = load float, ptr %pa2, align 4
  %pb2 = getelementptr inbounds float, ptr %bv, i64 %i2
  %xb2 = load float, ptr %pb2, align 4
  %eaddf = fadd float %xa2, %xb2
  %eadd = fpext float %eaddf to double
  %gaddp = getelementptr inbounds float, ptr %oaddv, i64 %i2
  %gaddf = load float, ptr %gaddp, align 4
  %gadd = fpext float %gaddf to double
  call void @check_close(double %gadd, double %eadd, ptr @.m_fadd)
  %esubf = fsub float %xa2, %xb2
  %esub = fpext float %esubf to double
  %gsubp = getelementptr inbounds float, ptr %osubv, i64 %i2
  %gsubf = load float, ptr %gsubp, align 4
  %gsub = fpext float %gsubf to double
  call void @check_close(double %gsub, double %esub, ptr @.m_fsub)
  %emulf = fmul float %xa2, %xb2
  %emul = fpext float %emulf to double
  %gmulp = getelementptr inbounds float, ptr %omulv, i64 %i2
  %gmulf = load float, ptr %gmulp, align 4
  %gmul = fpext float %gmulf to double
  call void @check_close(double %gmul, double %emul, ptr @.m_fmul)
  %edivf = fdiv float %xa2, %xb2
  %ediv = fpext float %edivf to double
  %gdivp = getelementptr inbounds float, ptr %odivv, i64 %i2
  %gdivf = load float, ptr %gdivp, align 4
  %gdiv = fpext float %gdivf to double
  call void @check_close(double %gdiv, double %ediv, ptr @.m_fdiv)
  %i2n = add nuw i64 %i2, 1
  br label %chk
fin:
  call void @universe_dataframe_series_free(ptr %oadd)
  call void @universe_dataframe_series_free(ptr %osub)
  call void @universe_dataframe_series_free(ptr %omul)
  call void @universe_dataframe_series_free(ptr %odiv)
  call void @universe_dataframe_series_free(ptr %a)
  call void @universe_dataframe_series_free(ptr %b)
  ret void
}

; ============================================================ f64
define internal void @test_f64(i64 %len, ptr %seed) {
entry:
  %a = call ptr @universe_dataframe_series_new(i32 3, i64 %len)
  %b = call ptr @universe_dataframe_series_new(i32 3, i64 %len)
  %av = call ptr @universe_dataframe_series_values(ptr %a)
  %bv = call ptr @universe_dataframe_series_values(ptr %b)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fbody ]
  %fd = icmp uge i64 %i, %len
  br i1 %fd, label %run, label %fbody
fbody:
  %r1 = call i64 @ut_rand(ptr %seed)
  %r2 = call i64 @ut_rand(ptr %seed)
  %m1 = urem i64 %r1, 2000000
  %m1s = sub i64 %m1, 1000000
  %xa = sitofp i64 %m1s to double
  %m2 = urem i64 %r2, 10000
  %m2a = add i64 %m2, 1
  %xb = sitofp i64 %m2a to double
  %pa = getelementptr inbounds double, ptr %av, i64 %i
  store double %xa, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %bv, i64 %i
  store double %xb, ptr %pb, align 8
  %in = add nuw i64 %i, 1
  br label %fill
run:
  %oadd = call ptr @universe_dataframe_add(ptr %a, ptr %b)
  %oaddv = call ptr @universe_dataframe_series_values(ptr %oadd)
  %osub = call ptr @universe_dataframe_sub(ptr %a, ptr %b)
  %osubv = call ptr @universe_dataframe_series_values(ptr %osub)
  %omul = call ptr @universe_dataframe_mul(ptr %a, ptr %b)
  %omulv = call ptr @universe_dataframe_series_values(ptr %omul)
  %odiv = call ptr @universe_dataframe_div(ptr %a, ptr %b)
  %odivv = call ptr @universe_dataframe_series_values(ptr %odiv)
  %orem = call ptr @universe_dataframe_rem(ptr %a, ptr %b)
  %oremv = call ptr @universe_dataframe_series_values(ptr %orem)
  ; scalar add 2.5, mul -1.5
  %osa = call ptr @universe_dataframe_add_scalar(ptr %a, i32 3, i64 0, double 2.500000e+00)
  %osav = call ptr @universe_dataframe_series_values(ptr %osa)
  %osm = call ptr @universe_dataframe_mul_scalar(ptr %a, i32 3, i64 0, double -1.500000e+00)
  %osmv = call ptr @universe_dataframe_series_values(ptr %osm)
  br label %chk
chk:
  %i2 = phi i64 [ 0, %run ], [ %i2n, %cb ]
  %d2 = icmp uge i64 %i2, %len
  br i1 %d2, label %fin, label %cb
cb:
  %pa2 = getelementptr inbounds double, ptr %av, i64 %i2
  %xa2 = load double, ptr %pa2, align 8
  %pb2 = getelementptr inbounds double, ptr %bv, i64 %i2
  %xb2 = load double, ptr %pb2, align 8
  %eadd = fadd double %xa2, %xb2
  %gaddp = getelementptr inbounds double, ptr %oaddv, i64 %i2
  %gadd = load double, ptr %gaddp, align 8
  call void @check_close(double %gadd, double %eadd, ptr @.m_fadd)
  %esub = fsub double %xa2, %xb2
  %gsubp = getelementptr inbounds double, ptr %osubv, i64 %i2
  %gsub = load double, ptr %gsubp, align 8
  call void @check_close(double %gsub, double %esub, ptr @.m_fsub)
  %emul = fmul double %xa2, %xb2
  %gmulp = getelementptr inbounds double, ptr %omulv, i64 %i2
  %gmul = load double, ptr %gmulp, align 8
  call void @check_close(double %gmul, double %emul, ptr @.m_fmul)
  %ediv = fdiv double %xa2, %xb2
  %gdivp = getelementptr inbounds double, ptr %odivv, i64 %i2
  %gdiv = load double, ptr %gdivp, align 8
  call void @check_close(double %gdiv, double %ediv, ptr @.m_fdiv)
  %erem = frem double %xa2, %xb2
  %gremp = getelementptr inbounds double, ptr %oremv, i64 %i2
  %grem = load double, ptr %gremp, align 8
  call void @check_close(double %grem, double %erem, ptr @.m_frem)
  %esa = fadd double %xa2, 2.500000e+00
  %gsap = getelementptr inbounds double, ptr %osav, i64 %i2
  %gsa = load double, ptr %gsap, align 8
  call void @check_close(double %gsa, double %esa, ptr @.m_sadd)
  %esm = fmul double %xa2, -1.500000e+00
  %gsmp = getelementptr inbounds double, ptr %osmv, i64 %i2
  %gsm = load double, ptr %gsmp, align 8
  call void @check_close(double %gsm, double %esm, ptr @.m_smul)
  %i2n = add nuw i64 %i2, 1
  br label %chk
fin:
  call void @universe_dataframe_series_free(ptr %oadd)
  call void @universe_dataframe_series_free(ptr %osub)
  call void @universe_dataframe_series_free(ptr %omul)
  call void @universe_dataframe_series_free(ptr %odiv)
  call void @universe_dataframe_series_free(ptr %orem)
  call void @universe_dataframe_series_free(ptr %osa)
  call void @universe_dataframe_series_free(ptr %osm)
  call void @universe_dataframe_series_free(ptr %a)
  call void @universe_dataframe_series_free(ptr %b)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8

  ; ---- per-dtype col-col + col-scalar + unary across edge sizes ----
  call void @test_i32(i64 0, ptr %seed)
  call void @test_i32(i64 1, ptr %seed)
  call void @test_i32(i64 7, ptr %seed)
  call void @test_i32(i64 16, ptr %seed)
  call void @test_i32(i64 17, ptr %seed)
  call void @test_i32(i64 1000, ptr %seed)
  call void @test_i64(i64 1, ptr %seed)
  call void @test_i64(i64 15, ptr %seed)
  call void @test_i64(i64 256, ptr %seed)
  call void @test_f32(i64 1, ptr %seed)
  call void @test_f32(i64 7, ptr %seed)
  call void @test_f32(i64 17, ptr %seed)
  call void @test_f32(i64 512, ptr %seed)
  call void @test_f64(i64 1, ptr %seed)
  call void @test_f64(i64 9, ptr %seed)
  call void @test_f64(i64 300, ptr %seed)

  ; ---- null propagation (out null = union of input nulls) ----
  %na = call ptr @universe_dataframe_series_new(i32 3, i64 8)
  %nb = call ptr @universe_dataframe_series_new(i32 3, i64 8)
  %nav = call ptr @universe_dataframe_series_values(ptr %na)
  %nbv = call ptr @universe_dataframe_series_values(ptr %nb)
  br label %nfill
nfill:
  %ni = phi i64 [ 0, %entry ], [ %nin, %nfb ]
  %nfd = icmp uge i64 %ni, 8
  br i1 %nfd, label %nset, label %nfb
nfb:
  %nap = getelementptr inbounds double, ptr %nav, i64 %ni
  store double 1.000000e+00, ptr %nap, align 8
  %nbp = getelementptr inbounds double, ptr %nbv, i64 %ni
  store double 2.000000e+00, ptr %nbp, align 8
  %nin = add nuw i64 %ni, 1
  br label %nfill
nset:
  ; a null at 1,3 ; b null at 3,5
  %ig0 = call i32 @universe_dataframe_series_set_null(ptr %na, i64 1)
  %ig1 = call i32 @universe_dataframe_series_set_null(ptr %na, i64 3)
  %ig2 = call i32 @universe_dataframe_series_set_null(ptr %nb, i64 3)
  %ig3 = call i32 @universe_dataframe_series_set_null(ptr %nb, i64 5)
  %nout = call ptr @universe_dataframe_add(ptr %na, ptr %nb)
  ; expected nulls at {1,3,5}
  call void @chknull(ptr %nout, i64 0, i1 false)
  call void @chknull(ptr %nout, i64 1, i1 true)
  call void @chknull(ptr %nout, i64 2, i1 false)
  call void @chknull(ptr %nout, i64 3, i1 true)
  call void @chknull(ptr %nout, i64 4, i1 false)
  call void @chknull(ptr %nout, i64 5, i1 true)
  call void @chknull(ptr %nout, i64 6, i1 false)
  %nc = call i64 @universe_dataframe_series_null_count(ptr %nout)
  call void @ut_check_eq(i64 %nc, i64 3, ptr @.m_np)
  call void @universe_dataframe_series_free(ptr %nout)
  call void @universe_dataframe_series_free(ptr %na)
  call void @universe_dataframe_series_free(ptr %nb)

  ; ---- integer div-by-zero => null (i32) ----
  %za = call ptr @universe_dataframe_series_new(i32 0, i64 6)
  %zb = call ptr @universe_dataframe_series_new(i32 0, i64 6)
  %zav = call ptr @universe_dataframe_series_values(ptr %za)
  %zbv = call ptr @universe_dataframe_series_values(ptr %zb)
  br label %zfill
zfill:
  %zi = phi i64 [ 0, %nset ], [ %zin, %zfb ]
  %zfd = icmp uge i64 %zi, 6
  br i1 %zfd, label %zset, label %zfb
zfb:
  %zap = getelementptr inbounds i32, ptr %zav, i64 %zi
  store i32 42, ptr %zap, align 4
  %zbp = getelementptr inbounds i32, ptr %zbv, i64 %zi
  store i32 2, ptr %zbp, align 4
  %zin = add nuw i64 %zi, 1
  br label %zfill
zset:
  ; zero out divisor at 2 and 4
  %zb2 = getelementptr inbounds i32, ptr %zbv, i64 2
  store i32 0, ptr %zb2, align 4
  %zb4 = getelementptr inbounds i32, ptr %zbv, i64 4
  store i32 0, ptr %zb4, align 4
  %zout = call ptr @universe_dataframe_div(ptr %za, ptr %zb)
  call void @chknull(ptr %zout, i64 0, i1 false)
  call void @chknull(ptr %zout, i64 2, i1 true)
  call void @chknull(ptr %zout, i64 3, i1 false)
  call void @chknull(ptr %zout, i64 4, i1 true)
  %znc = call i64 @universe_dataframe_series_null_count(ptr %zout)
  call void @ut_check_eq(i64 %znc, i64 2, ptr @.m_dznc)
  ; non-null lanes computed correctly (42/2 = 21)
  %zov = call ptr @universe_dataframe_series_values(ptr %zout)
  %zo0p = getelementptr inbounds i32, ptr %zov, i64 0
  %zo0 = load i32, ptr %zo0p, align 4
  %zo0e = sext i32 %zo0 to i64
  call void @ut_check_eq(i64 %zo0e, i64 21, ptr @.m_dz)
  call void @universe_dataframe_series_free(ptr %zout)
  call void @universe_dataframe_series_free(ptr %za)
  call void @universe_dataframe_series_free(ptr %zb)

  ; ---- horizontal min/max across two i32 columns ----
  %hc0 = call ptr @universe_dataframe_series_new(i32 0, i64 5)
  %hc1 = call ptr @universe_dataframe_series_new(i32 0, i64 5)
  %hv0 = call ptr @universe_dataframe_series_values(ptr %hc0)
  %hv1 = call ptr @universe_dataframe_series_values(ptr %hc1)
  ; col0 = [10, 3, 7, 8, 100]  col1 = [1, 30, 7, 9, 50]
  store i32 10, ptr %hv0, align 4
  %hv0a = getelementptr inbounds i32, ptr %hv0, i64 1
  store i32 3, ptr %hv0a, align 4
  %hv0b = getelementptr inbounds i32, ptr %hv0, i64 2
  store i32 7, ptr %hv0b, align 4
  %hv0c = getelementptr inbounds i32, ptr %hv0, i64 3
  store i32 8, ptr %hv0c, align 4
  %hv0d = getelementptr inbounds i32, ptr %hv0, i64 4
  store i32 100, ptr %hv0d, align 4
  store i32 1, ptr %hv1, align 4
  %hv1a = getelementptr inbounds i32, ptr %hv1, i64 1
  store i32 30, ptr %hv1a, align 4
  %hv1b = getelementptr inbounds i32, ptr %hv1, i64 2
  store i32 7, ptr %hv1b, align 4
  %hv1c = getelementptr inbounds i32, ptr %hv1, i64 3
  store i32 9, ptr %hv1c, align 4
  %hv1d = getelementptr inbounds i32, ptr %hv1, i64 4
  store i32 50, ptr %hv1d, align 4
  ; row 4: make both null -> output null
  %ig4 = call i32 @universe_dataframe_series_set_null(ptr %hc0, i64 4)
  %ig5 = call i32 @universe_dataframe_series_set_null(ptr %hc1, i64 4)
  %df = call ptr @universe_dataframe_new()
  %ig6 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.c0, i64 1, ptr %hc0)
  %ig7 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.c1, i64 1, ptr %hc1)
  %hmin = call ptr @universe_dataframe_min_horizontal(ptr %df)
  %hmax = call ptr @universe_dataframe_max_horizontal(ptr %df)
  %hminv = call ptr @universe_dataframe_series_values(ptr %hmin)
  %hmaxv = call ptr @universe_dataframe_series_values(ptr %hmax)
  ; row0 min=1 max=10 ; row1 min=3 max=30 ; row2 min=7 max=7 ; row3 min=8 max=9
  %hmin0 = load double, ptr %hminv, align 8
  call void @check_close(double %hmin0, double 1.000000e+00, ptr @.m_hmin)
  %hmax0 = load double, ptr %hmaxv, align 8
  call void @check_close(double %hmax0, double 1.000000e+01, ptr @.m_hmax)
  %hmin1p = getelementptr inbounds double, ptr %hminv, i64 1
  %hmin1 = load double, ptr %hmin1p, align 8
  call void @check_close(double %hmin1, double 3.000000e+00, ptr @.m_hmin)
  %hmax1p = getelementptr inbounds double, ptr %hmaxv, i64 1
  %hmax1 = load double, ptr %hmax1p, align 8
  call void @check_close(double %hmax1, double 3.000000e+01, ptr @.m_hmax)
  %hmin3p = getelementptr inbounds double, ptr %hminv, i64 3
  %hmin3 = load double, ptr %hmin3p, align 8
  call void @check_close(double %hmin3, double 8.000000e+00, ptr @.m_hmin)
  %hmax3p = getelementptr inbounds double, ptr %hmaxv, i64 3
  %hmax3 = load double, ptr %hmax3p, align 8
  call void @check_close(double %hmax3, double 9.000000e+00, ptr @.m_hmax)
  ; row4 both null -> null
  call void @chknull(ptr %hmin, i64 4, i1 true)
  call void @chknull(ptr %hmax, i64 4, i1 true)
  call void @universe_dataframe_series_free(ptr %hmin)
  call void @universe_dataframe_series_free(ptr %hmax)
  call void @universe_dataframe_free(ptr %df)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; check series element i null-ness against expectation (uses is_null).
define internal void @chknull(ptr %s, i64 %i, i1 %exp) {
entry:
  %ob = alloca i8, align 1
  %rc = call i32 @universe_dataframe_series_is_null(ptr %s, i64 %i, ptr %ob)
  %v = load i8, ptr %ob, align 1
  %isnull = icmp ne i8 %v, 0
  %ok = icmp eq i1 %isnull, %exp
  call void @ut_check(i1 %ok, ptr @.m_nv)
  ret void
}
