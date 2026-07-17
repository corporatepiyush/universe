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

; Tests for universe_linalg_eigen_sym / universe_linalg_svd. Eigen KATs
; (diagonal, 2x2 analytic, known 3x3) verify A*V == V*diag(vals) (tiny residual),
; V orthonormal (V^T V == I), and eigenvalue invariants (sum == trace, product ==
; det for the KAT matrices). SVD KATs (diagonal, known rectangular) reconstruct
; U*diag(S)*V^T == A, check singular values descending, and cross-check S^2
; against eigen_sym(A^T A). Fixed-seed random symmetric / rectangular matrices
; exercise the general path. Error codes 1 (NULL) and 8 (INVALID_ARG) are
; checked. --bench times eigen_sym on a random symmetric matrix.

declare void @ut_check(i1, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare double @llvm.fabs.f64(double)

declare i32 @universe_linalg_eigen_sym(ptr, i64, ptr, ptr)
declare i32 @universe_linalg_svd(ptr, i64, i64, ptr, ptr, ptr)
declare i32 @universe_linalg_matmul(ptr, i64, i64, ptr, i64, i64, ptr)
declare i32 @universe_linalg_transpose(ptr, i64, i64, ptr)
declare i32 @universe_linalg_identity(ptr, i64)
declare double @universe_linalg_tr(ptr, i64)

; ---- KAT constant matrices (row-major) ----
@k_diag4 = private constant [16 x double]
  [ double 1.0, double 0.0, double 0.0, double 0.0,
    double 0.0, double 2.0, double 0.0, double 0.0,
    double 0.0, double 0.0, double 5.0, double 0.0,
    double 0.0, double 0.0, double 0.0, double 8.0 ], align 8
@k_2x2 = private constant [4 x double]
  [ double 2.0, double 1.0, double 1.0, double 2.0 ], align 8
@k_3x3 = private constant [9 x double]
  [ double 2.0, double -1.0, double 0.0,
    double -1.0, double 2.0, double -1.0,
    double 0.0, double -1.0, double 2.0 ], align 8
@k_svd_diag = private constant [9 x double]
  [ double 3.0, double 0.0, double 0.0,
    double 0.0, double 1.0, double 0.0,
    double 0.0, double 0.0, double 2.0 ], align 8
@k_svd_rect = private constant [6 x double]
  [ double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0 ], align 8

; ---- scratch globals (max n=8 -> 64 elems) ----
@g_A   = private global [64 x double] zeroinitializer, align 8
@g_V   = private global [64 x double] zeroinitializer, align 8
@g_val = private global [64 x double] zeroinitializer, align 8
@g_t1  = private global [64 x double] zeroinitializer, align 8
@g_t2  = private global [64 x double] zeroinitializer, align 8
@g_I   = private global [64 x double] zeroinitializer, align 8
@g_U   = private global [64 x double] zeroinitializer, align 8
@g_S   = private global [64 x double] zeroinitializer, align 8
@g_Vt  = private global [64 x double] zeroinitializer, align 8
@g_AtA = private global [64 x double] zeroinitializer, align 8
@g_ev  = private global [64 x double] zeroinitializer, align 8
@g_evc = private global [64 x double] zeroinitializer, align 8

@m.eig_res  = private constant [25 x i8] c"eigen A*V==V*diag residl\00"
@m.eig_orth = private constant [25 x i8] c"eigen V^T V == I (ortho)\00"
@m.eig_tr   = private constant [25 x i8] c"eigen sum(vals)==trace  \00"
@m.d4_0     = private constant [25 x i8] c"eigen diag4 val[0]==1   \00"
@m.d4_1     = private constant [25 x i8] c"eigen diag4 val[1]==2   \00"
@m.d4_2     = private constant [25 x i8] c"eigen diag4 val[2]==5   \00"
@m.d4_3     = private constant [25 x i8] c"eigen diag4 val[3]==8   \00"
@m.a2_0     = private constant [25 x i8] c"eigen 2x2 val[0]==1     \00"
@m.a2_1     = private constant [25 x i8] c"eigen 2x2 val[1]==3     \00"
@m.t3_1     = private constant [25 x i8] c"eigen 3x3 val[1]==2     \00"
@m.t3_sum   = private constant [25 x i8] c"eigen 3x3 v0+v2==4      \00"
@m.t3_prd   = private constant [25 x i8] c"eigen 3x3 v0*v2==2      \00"
@m.t3_det   = private constant [25 x i8] c"eigen 3x3 prod(vals)==4 \00"
@m.svd_rec  = private constant [25 x i8] c"svd U diag(S) V^T == A  \00"
@m.svd_desc = private constant [25 x i8] c"svd singular vals desc  \00"
@m.svd_ata  = private constant [25 x i8] c"svd S^2 == eig(A^T A)   \00"
@m.svd_d0   = private constant [25 x i8] c"svd diag S[0]==3        \00"
@m.svd_d1   = private constant [25 x i8] c"svd diag S[1]==2        \00"
@m.svd_d2   = private constant [25 x i8] c"svd diag S[2]==1        \00"
@m.e_null1  = private constant [25 x i8] c"eigen NULL A -> rc 1    \00"
@m.e_arg1   = private constant [25 x i8] c"eigen neg n  -> rc 8    \00"
@m.e_null2  = private constant [25 x i8] c"svd NULL A   -> rc 1    \00"
@m.e_arg2   = private constant [25 x i8] c"svd neg m    -> rc 8    \00"
@m.bench    = private constant [35 x i8] c"eigen_sym 6x6 x50000: %.3f Mops/s\0A\00"

; ======================================================================= helpers

; approx |a-b| <= tol
define i1 @approx(double %a, double %b, double %tol) {
  %d = fsub double %a, %b
  %ad = call double @llvm.fabs.f64(double %d)
  %r = fcmp ole double %ad, %tol
  ret i1 %r
}

; max |x[i]-y[i]| over count elems (count >= 1)
define double @maxabsdiff(ptr %x, ptr %y, i64 %count) {
entry:
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %lp ]
  %mx = phi double [ 0.0, %entry ], [ %mxn, %lp ]
  %px = getelementptr inbounds double, ptr %x, i64 %i
  %a = load double, ptr %px, align 8
  %py = getelementptr inbounds double, ptr %y, i64 %i
  %b = load double, ptr %py, align 8
  %d = fsub double %a, %b
  %ad = call double @llvm.fabs.f64(double %d)
  %gt = fcmp ogt double %ad, %mx
  %mxn = select i1 %gt, double %ad, double %mx
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %count
  br i1 %more, label %lp, label %done
done:
  %res = phi double [ %mxn, %lp ]
  ret double %res
}

; out[i*n+j] = M[i*n+j] * d[j]   (scale columns by vector d)
define void @colscale(ptr %M, ptr %d, ptr %out, i64 %n) {
entry:
  br label %il
il:
  %i = phi i64 [ 0, %entry ], [ %in, %iend ]
  %irow = mul i64 %i, %n
  br label %jl
jl:
  %j = phi i64 [ 0, %il ], [ %jn, %jl ]
  %idx = add i64 %irow, %j
  %mp = getelementptr inbounds double, ptr %M, i64 %idx
  %mv = load double, ptr %mp, align 8
  %dp = getelementptr inbounds double, ptr %d, i64 %j
  %dv = load double, ptr %dp, align 8
  %pv = fmul double %mv, %dv
  %op = getelementptr inbounds double, ptr %out, i64 %idx
  store double %pv, ptr %op, align 8
  %jn = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jn, %n
  br i1 %jmore, label %jl, label %iend
iend:
  %in = add nuw i64 %i, 1
  %imore = icmp ult i64 %in, %n
  br i1 %imore, label %il, label %ret
ret:
  ret void
}

; random symmetric n x n into A (values ~[-10,10])
define void @fill_rand_sym(ptr %A, i64 %n, ptr %st) {
entry:
  br label %il
il:
  %i = phi i64 [ 0, %entry ], [ %in, %iend ]
  br label %jl
jl:
  %j = phi i64 [ %i, %il ], [ %jn, %jl ]
  %r = call i64 @ut_rand(ptr %st)
  %m = urem i64 %r, 2001
  %ms = sub i64 %m, 1000
  %f = sitofp i64 %ms to double
  %v = fdiv double %f, 100.0
  %ij = mul i64 %i, %n
  %ijx = add i64 %ij, %j
  %pij = getelementptr inbounds double, ptr %A, i64 %ijx
  store double %v, ptr %pij, align 8
  %ji = mul i64 %j, %n
  %jix = add i64 %ji, %i
  %pji = getelementptr inbounds double, ptr %A, i64 %jix
  store double %v, ptr %pji, align 8
  %jn = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jn, %n
  br i1 %jmore, label %jl, label %iend
iend:
  %in = add nuw i64 %i, 1
  %imore = icmp ult i64 %in, %n
  br i1 %imore, label %il, label %ret
ret:
  ret void
}

; random dense m x n into A
define void @fill_rand(ptr %A, i64 %m, i64 %n, ptr %st) {
entry:
  %cnt = mul i64 %m, %n
  br label %lp
lp:
  %i = phi i64 [ 0, %entry ], [ %in, %lp ]
  %r = call i64 @ut_rand(ptr %st)
  %mm = urem i64 %r, 2001
  %ms = sub i64 %mm, 1000
  %f = sitofp i64 %ms to double
  %v = fdiv double %f, 100.0
  %p = getelementptr inbounds double, ptr %A, i64 %i
  store double %v, ptr %p, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %cnt
  br i1 %more, label %lp, label %ret
ret:
  ret void
}

; run the standard eigen checks on symmetric A (n x n): returns void, records
; residual + orthonormality + trace checks. Uses g_V,g_val,g_t1,g_t2,g_I.
define void @check_eigen(ptr %A, i64 %n, double %tol) {
entry:
  %nn = mul i64 %n, %n
  %rc = call i32 @universe_linalg_eigen_sym(ptr %A, i64 %n, ptr @g_val, ptr @g_V)
  ; AV = A * V  -> g_t1
  %r1 = call i32 @universe_linalg_matmul(ptr %A, i64 %n, i64 %n, ptr @g_V, i64 %n, i64 %n, ptr @g_t1)
  ; VD = V scaled by vals columns -> g_t2
  call void @colscale(ptr @g_V, ptr @g_val, ptr @g_t2, i64 %n)
  %res = call double @maxabsdiff(ptr @g_t1, ptr @g_t2, i64 %nn)
  %okres = fcmp ole double %res, %tol
  call void @ut_check(i1 %okres, ptr @m.eig_res)
  ; orthonormal: Vt = V^T -> g_t1 ; G = Vt * V -> g_t2 ; I -> g_I
  %r2 = call i32 @universe_linalg_transpose(ptr @g_V, i64 %n, i64 %n, ptr @g_t1)
  %r3 = call i32 @universe_linalg_matmul(ptr @g_t1, i64 %n, i64 %n, ptr @g_V, i64 %n, i64 %n, ptr @g_t2)
  %r4 = call i32 @universe_linalg_identity(ptr @g_I, i64 %n)
  %orr = call double @maxabsdiff(ptr @g_t2, ptr @g_I, i64 %nn)
  %okorr = fcmp ole double %orr, %tol
  call void @ut_check(i1 %okorr, ptr @m.eig_orth)
  ; trace: sum(vals) == tr(A)
  %tr = call double @universe_linalg_tr(ptr %A, i64 %n)
  br label %sl
sl:
  %i = phi i64 [ 0, %entry ], [ %in, %sl ]
  %sacc = phi double [ 0.0, %entry ], [ %sn, %sl ]
  %vp = getelementptr inbounds double, ptr @g_val, i64 %i
  %vv = load double, ptr %vp, align 8
  %sn = fadd double %sacc, %vv
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %sl, label %trdone
trdone:
  %oktr = call i1 @approx(double %sn, double %tr, double %tol)
  call void @ut_check(i1 %oktr, ptr @m.eig_tr)
  ret void
}

; run SVD reconstruction + descending + S^2==eig(A^T A) on A (m x n).
define void @check_svd(ptr %A, i64 %m, i64 %n, double %tol) {
entry:
  %mn = mul i64 %m, %n
  %nn = mul i64 %n, %n
  %rc = call i32 @universe_linalg_svd(ptr %A, i64 %m, i64 %n, ptr @g_U, ptr @g_S, ptr @g_Vt)
  ; US = U (m x n) scaled by S columns -> g_t1 (explicit, U is not square)
  br label %uscale
uscale:
  br label %usi
usi:
  %i = phi i64 [ 0, %uscale ], [ %in, %usiend ]
  %irow = mul i64 %i, %n
  br label %usj
usj:
  %j = phi i64 [ 0, %usi ], [ %jn, %usj ]
  %idx = add i64 %irow, %j
  %up = getelementptr inbounds double, ptr @g_U, i64 %idx
  %uv = load double, ptr %up, align 8
  %sp = getelementptr inbounds double, ptr @g_S, i64 %j
  %sv = load double, ptr %sp, align 8
  %pv = fmul double %uv, %sv
  %tp = getelementptr inbounds double, ptr @g_t1, i64 %idx
  store double %pv, ptr %tp, align 8
  %jn = add nuw i64 %j, 1
  %jmore = icmp ult i64 %jn, %n
  br i1 %jmore, label %usj, label %usiend
usiend:
  %in = add nuw i64 %i, 1
  %imore = icmp ult i64 %in, %m
  br i1 %imore, label %usi, label %recon
recon:
  ; recon = US (m x n) * Vt (n x n) -> g_t2 (m x n)
  %r1 = call i32 @universe_linalg_matmul(ptr @g_t1, i64 %m, i64 %n, ptr @g_Vt, i64 %n, i64 %n, ptr @g_t2)
  %rr = call double @maxabsdiff(ptr @g_t2, ptr %A, i64 %mn)
  %okrr = fcmp ole double %rr, %tol
  call void @ut_check(i1 %okrr, ptr @m.svd_rec)
  ; descending: S[k] >= S[k+1] for k in 0..n-2 (never reads S[n])
  %nm1d = sub i64 %n, 1
  br label %dl
dl:
  %k = phi i64 [ 0, %recon ], [ %kn, %dlbody ]
  %okd = phi i1 [ true, %recon ], [ %okd2, %dlbody ]
  %atend = icmp uge i64 %k, %nm1d
  br i1 %atend, label %ddone, label %dlbody
dlbody:
  %kn = add nuw i64 %k, 1
  %kp = getelementptr inbounds double, ptr @g_S, i64 %k
  %sk = load double, ptr %kp, align 8
  %knp = getelementptr inbounds double, ptr @g_S, i64 %kn
  %skn = load double, ptr %knp, align 8
  %ge = fcmp oge double %sk, %skn
  %okd2 = and i1 %okd, %ge
  br label %dl
ddone:
  call void @ut_check(i1 %okd, ptr @m.svd_desc)
  ; AtA = A^T (n x m) * A (m x n) -> g_AtA (n x n); via transpose then matmul
  %r2 = call i32 @universe_linalg_transpose(ptr %A, i64 %m, i64 %n, ptr @g_t1)
  %r3 = call i32 @universe_linalg_matmul(ptr @g_t1, i64 %n, i64 %m, ptr %A, i64 %m, i64 %n, ptr @g_AtA)
  %r4 = call i32 @universe_linalg_eigen_sym(ptr @g_AtA, i64 %n, ptr @g_ev, ptr @g_evc)
  ; S descending, ev ascending -> compare S[k]^2 to ev[n-1-k]
  br label %cl
cl:
  %c = phi i64 [ 0, %ddone ], [ %cnx, %cl ]
  %oka = phi i1 [ true, %ddone ], [ %oka2, %cl ]
  %scp = getelementptr inbounds double, ptr @g_S, i64 %c
  %scv = load double, ptr %scp, align 8
  %sq = fmul double %scv, %scv
  %nm1 = sub i64 %n, 1
  %rev = sub i64 %nm1, %c
  %evp = getelementptr inbounds double, ptr @g_ev, i64 %rev
  %evv = load double, ptr %evp, align 8
  %near = call i1 @approx(double %sq, double %evv, double %tol)
  %oka2 = and i1 %oka, %near
  %cnx = add nuw i64 %c, 1
  %cmore = icmp ult i64 %cnx, %n
  br i1 %cmore, label %cl, label %cdone
cdone:
  call void @ut_check(i1 %oka2, ptr @m.svd_ata)
  ret void
}

; ======================================================================= main

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8

  ; ---- eigen diagonal 4x4 (ascending distinct -> vals == diagonal) ----
  %rd = call i32 @universe_linalg_eigen_sym(ptr @k_diag4, i64 4, ptr @g_val, ptr @g_V)
  %d0p = getelementptr inbounds double, ptr @g_val, i64 0
  %d0 = load double, ptr %d0p, align 8
  %od0 = call i1 @approx(double %d0, double 1.0, double 1.000000e-09)
  call void @ut_check(i1 %od0, ptr @m.d4_0)
  %d1p = getelementptr inbounds double, ptr @g_val, i64 1
  %d1 = load double, ptr %d1p, align 8
  %od1 = call i1 @approx(double %d1, double 2.0, double 1.000000e-09)
  call void @ut_check(i1 %od1, ptr @m.d4_1)
  %d2p = getelementptr inbounds double, ptr @g_val, i64 2
  %d2 = load double, ptr %d2p, align 8
  %od2 = call i1 @approx(double %d2, double 5.0, double 1.000000e-09)
  call void @ut_check(i1 %od2, ptr @m.d4_2)
  %d3p = getelementptr inbounds double, ptr @g_val, i64 3
  %d3 = load double, ptr %d3p, align 8
  %od3 = call i1 @approx(double %d3, double 8.0, double 1.000000e-09)
  call void @ut_check(i1 %od3, ptr @m.d4_3)
  call void @check_eigen(ptr @k_diag4, i64 4, double 1.000000e-09)

  ; ---- eigen 2x2 analytic (eigenvalues 1, 3) ----
  %r2 = call i32 @universe_linalg_eigen_sym(ptr @k_2x2, i64 2, ptr @g_val, ptr @g_V)
  %a0p = getelementptr inbounds double, ptr @g_val, i64 0
  %a0 = load double, ptr %a0p, align 8
  %oa0 = call i1 @approx(double %a0, double 1.0, double 1.000000e-09)
  call void @ut_check(i1 %oa0, ptr @m.a2_0)
  %a1p = getelementptr inbounds double, ptr @g_val, i64 1
  %a1 = load double, ptr %a1p, align 8
  %oa1 = call i1 @approx(double %a1, double 3.0, double 1.000000e-09)
  call void @ut_check(i1 %oa1, ptr @m.a2_1)
  call void @check_eigen(ptr @k_2x2, i64 2, double 1.000000e-09)

  ; ---- eigen known 3x3 tridiagonal: spectrum {2-sqrt2, 2, 2+sqrt2} ----
  ; pinned exactly by rational identities: mid==2, v0+v2==4, v0*v2==2, det==4.
  %r3 = call i32 @universe_linalg_eigen_sym(ptr @k_3x3, i64 3, ptr @g_val, ptr @g_V)
  %t0p = getelementptr inbounds double, ptr @g_val, i64 0
  %t0 = load double, ptr %t0p, align 8
  %t1p = getelementptr inbounds double, ptr @g_val, i64 1
  %t1 = load double, ptr %t1p, align 8
  %ot1 = call i1 @approx(double %t1, double 2.0, double 1.000000e-07)
  call void @ut_check(i1 %ot1, ptr @m.t3_1)
  %t2p = getelementptr inbounds double, ptr @g_val, i64 2
  %t2 = load double, ptr %t2p, align 8
  %sum02 = fadd double %t0, %t2
  %osum = call i1 @approx(double %sum02, double 4.0, double 1.000000e-07)
  call void @ut_check(i1 %osum, ptr @m.t3_sum)
  %prd02 = fmul double %t0, %t2
  %oprd = call i1 @approx(double %prd02, double 2.0, double 1.000000e-07)
  call void @ut_check(i1 %oprd, ptr @m.t3_prd)
  ; product of all == det == 4
  %pr01 = fmul double %t0, %t1
  %pr = fmul double %pr01, %t2
  %odet = call i1 @approx(double %pr, double 4.0, double 1.000000e-07)
  call void @ut_check(i1 %odet, ptr @m.t3_det)
  call void @check_eigen(ptr @k_3x3, i64 3, double 1.000000e-07)

  ; ---- eigen random symmetric n=5 and n=6 ----
  call void @fill_rand_sym(ptr @g_A, i64 5, ptr %st)
  call void @check_eigen(ptr @g_A, i64 5, double 1.000000e-06)
  call void @fill_rand_sym(ptr @g_A, i64 6, ptr %st)
  call void @check_eigen(ptr @g_A, i64 6, double 1.000000e-06)

  ; ---- svd diagonal 3x3 (S = 3,2,1) ----
  %sd = call i32 @universe_linalg_svd(ptr @k_svd_diag, i64 3, i64 3, ptr @g_U, ptr @g_S, ptr @g_Vt)
  %sd0p = getelementptr inbounds double, ptr @g_S, i64 0
  %sd0 = load double, ptr %sd0p, align 8
  %osd0 = call i1 @approx(double %sd0, double 3.0, double 1.000000e-09)
  call void @ut_check(i1 %osd0, ptr @m.svd_d0)
  %sd1p = getelementptr inbounds double, ptr @g_S, i64 1
  %sd1 = load double, ptr %sd1p, align 8
  %osd1 = call i1 @approx(double %sd1, double 2.0, double 1.000000e-09)
  call void @ut_check(i1 %osd1, ptr @m.svd_d1)
  %sd2p = getelementptr inbounds double, ptr @g_S, i64 2
  %sd2 = load double, ptr %sd2p, align 8
  %osd2 = call i1 @approx(double %sd2, double 1.0, double 1.000000e-09)
  call void @ut_check(i1 %osd2, ptr @m.svd_d2)
  call void @check_svd(ptr @k_svd_diag, i64 3, i64 3, double 1.000000e-07)

  ; ---- svd known rectangular 3x2 ----
  call void @check_svd(ptr @k_svd_rect, i64 3, i64 2, double 1.000000e-07)

  ; ---- svd random 4x3 and 5x5 ----
  call void @fill_rand(ptr @g_A, i64 4, i64 3, ptr %st)
  call void @check_svd(ptr @g_A, i64 4, i64 3, double 1.000000e-06)
  call void @fill_rand(ptr @g_A, i64 5, i64 5, ptr %st)
  call void @check_svd(ptr @g_A, i64 5, i64 5, double 1.000000e-06)

  ; ---- error codes ----
  %en1 = call i32 @universe_linalg_eigen_sym(ptr null, i64 3, ptr @g_val, ptr @g_V)
  %ok_en1 = icmp eq i32 %en1, 1
  call void @ut_check(i1 %ok_en1, ptr @m.e_null1)
  %ea1 = call i32 @universe_linalg_eigen_sym(ptr @k_3x3, i64 -1, ptr @g_val, ptr @g_V)
  %ok_ea1 = icmp eq i32 %ea1, 8
  call void @ut_check(i1 %ok_ea1, ptr @m.e_arg1)
  %en2 = call i32 @universe_linalg_svd(ptr null, i64 3, i64 2, ptr @g_U, ptr @g_S, ptr @g_Vt)
  %ok_en2 = icmp eq i32 %en2, 1
  call void @ut_check(i1 %ok_en2, ptr @m.e_null2)
  %ea2 = call i32 @universe_linalg_svd(ptr @k_svd_rect, i64 -1, i64 2, ptr @g_U, ptr @g_S, ptr @g_Vt)
  %ok_ea2 = icmp eq i32 %ea2, 8
  call void @ut_check(i1 %ok_ea2, ptr @m.e_arg2)

  ; ---- optional bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin
bench:
  call void @fill_rand_sym(ptr @g_A, i64 6, ptr %st)
  %t_start = call double @ut_now_sec()
  br label %bl
bl:
  %bi = phi i64 [ 0, %bench ], [ %bin, %bl ]
  %bx = call i32 @universe_linalg_eigen_sym(ptr @g_A, i64 6, ptr @g_val, ptr @g_V)
  %bin = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bin, 50000
  br i1 %bmore, label %bl, label %bdone
bdone:
  %t_end = call double @ut_now_sec()
  %dt = fsub double %t_end, %t_start
  %ops = fdiv double 5.000000e+04, %dt
  %mops = fdiv double %ops, 1.000000e+06
  %pc = call i32 (ptr, ...) @printf(ptr @m.bench, double %mops)
  br label %fin
fin:
  %s = call i32 @ut_summary()
  ret i32 %s
}
