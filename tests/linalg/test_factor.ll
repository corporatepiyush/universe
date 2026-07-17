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

; Tests for universe_linalg_* direct methods (factor.ll). Strong reconstruction
; KATs cross-checked against the matrix.ll BLAS core: L*U == P*A, Q*R == A with
; Q'Q == I, L*L' == A (Cholesky); solve residual ||Ax-b|| tiny; det of known
; matrices; inv*A == I; tri_solve exact; lstsq == normal-equations reference.
; Fixed-seed random well-conditioned (diagonally dominant / SPD) matrices are
; reconstructed to < 1e-7 rel. Error paths: singular->11, non-SPD->13,
; shape/dim->8, null->1. --bench times a random solve.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare double @llvm.fabs.f64(double)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; factor.ll (under test)
declare i32 @universe_linalg_lu(ptr, i64, ptr, ptr, ptr)
declare i32 @universe_linalg_solve(ptr, i64, ptr, i64, ptr)
declare i32 @universe_linalg_inv(ptr, i64, ptr)
declare double @universe_linalg_det(ptr, i64)
declare i32 @universe_linalg_cholesky(ptr, i64, ptr)
declare i32 @universe_linalg_qr(ptr, i64, i64, ptr, ptr)
declare i32 @universe_linalg_tri_solve(ptr, i64, ptr, i32, ptr)
declare i32 @universe_linalg_lstsq(ptr, i64, i64, ptr, ptr)

; matrix.ll (BLAS core, same domain — used to verify reconstructions)
declare i32 @universe_linalg_matmul(ptr, i64, i64, ptr, i64, i64, ptr)
declare i32 @universe_linalg_matvec(ptr, i64, i64, ptr, ptr)
declare i32 @universe_linalg_transpose(ptr, i64, i64, ptr)
declare i32 @universe_linalg_identity(ptr, i64)

@m.lu    = private unnamed_addr constant [24 x i8] c"LU  L*U==P*A recon     \00", align 1
@m.solve = private unnamed_addr constant [24 x i8] c"solve residual tiny    \00", align 1
@m.solvk = private unnamed_addr constant [24 x i8] c"KAT solve 3x3          \00", align 1
@m.inv   = private unnamed_addr constant [24 x i8] c"inv*A==I               \00", align 1
@m.chol  = private unnamed_addr constant [24 x i8] c"Cholesky L*L'==A recon \00", align 1
@m.qr    = private unnamed_addr constant [24 x i8] c"QR  Q*R==A, Q'Q==I     \00", align 1
@m.detd  = private unnamed_addr constant [24 x i8] c"KAT det diagonal       \00", align 1
@m.dett  = private unnamed_addr constant [24 x i8] c"KAT det upper-tri      \00", align 1
@m.det3  = private unnamed_addr constant [24 x i8] c"KAT det 3x3            \00", align 1
@m.dets  = private unnamed_addr constant [24 x i8] c"KAT det singular==0    \00", align 1
@m.triu  = private unnamed_addr constant [24 x i8] c"tri_solve upper exact  \00", align 1
@m.tril  = private unnamed_addr constant [24 x i8] c"tri_solve lower exact  \00", align 1
@m.lsq   = private unnamed_addr constant [24 x i8] c"lstsq == normal-eq     \00", align 1
@m.lsqk  = private unnamed_addr constant [24 x i8] c"KAT lstsq line fit     \00", align 1
@m.enul  = private unnamed_addr constant [24 x i8] c"err null->1            \00", align 1
@m.edim  = private unnamed_addr constant [24 x i8] c"err dim->8             \00", align 1
@m.esh   = private unnamed_addr constant [24 x i8] c"err lstsq m<n->8       \00", align 1
@m.esrhs = private unnamed_addr constant [24 x i8] c"err solve nrhs<1->8    \00", align 1
@m.esing = private unnamed_addr constant [24 x i8] c"err LU singular->11    \00", align 1
@m.espd  = private unnamed_addr constant [24 x i8] c"err Cholesky nSPD->13  \00", align 1
@b.fmt   = private unnamed_addr constant [30 x i8] c"BENCH solve %ldx%ld: %.1f ns\0A\00"

; close(v,s): |v-s| <= (|s|+1)*1e-7
define internal i1 @close(double %v, double %s) {
entry:
  %d = fsub double %v, %s
  %ad = call double @llvm.fabs.f64(double %d)
  %as = call double @llvm.fabs.f64(double %s)
  %b0 = fadd double %as, 1.0
  %b = fmul double %b0, 1.000000e-07
  %ok = fcmp ole double %ad, %b
  ret i1 %ok
}

; arrdiff(p,q,n): count elementwise mismatches (close)
define internal i64 @arrdiff(ptr %p, ptr %q, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %violn, %loop ]
  %pp = getelementptr inbounds double, ptr %p, i64 %i
  %pv = load double, ptr %pp, align 8
  %qp = getelementptr inbounds double, ptr %q, i64 %i
  %qv = load double, ptr %qp, align 8
  %ok = call i1 @close(double %pv, double %qv)
  %bad = xor i1 %ok, true
  %inc = zext i1 %bad to i64
  %violn = add nuw i64 %viol, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %violn, %loop ]
  ret i64 %r
}

; rndmat(st,a,n): fill n doubles in [-1,1)
define internal void @rndmat(ptr %st, ptr %a, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %low = and i64 %r, 1048575
  %rf = uitofp i64 %low to double
  %sc = fdiv double %rf, 5.242880e+05
  %v = fsub double %sc, 1.0
  %p = getelementptr inbounds double, ptr %a, i64 %i
  store double %v, ptr %p, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; mk_dd(st,A,n): random n*n then add n to the diagonal -> diagonally dominant,
; guaranteed nonsingular (and used as a stable well-conditioned test system).
define internal void @mk_dd(ptr %st, ptr %A, i64 %n) {
entry:
  %nn = mul i64 %n, %n
  call void @rndmat(ptr %st, ptr %A, i64 %nn)
  %nf = uitofp i64 %n to double
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %di = mul i64 %i, %n
  %didx = add i64 %di, %i
  %p = getelementptr inbounds double, ptr %A, i64 %didx
  %v = load double, ptr %p, align 8
  %nv = fadd double %v, %nf
  store double %nv, ptr %p, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; recon_lu_viol(A,n): LU factor, rebuild L,U, matmul, compare to P*A. Returns
; violation count (0 = perfect reconstruction).
define internal i64 @recon_lu_viol(ptr %A, i64 %n) {
entry:
  %LU = alloca [64 x double], align 8
  %Lm = alloca [64 x double], align 8
  %Um = alloca [64 x double], align 8
  %prod = alloca [64 x double], align 8
  %PA = alloca [64 x double], align 8
  %piv = alloca [16 x i32], align 4
  %sgn = alloca double, align 8
  %rc = call i32 @universe_linalg_lu(ptr %A, i64 %n, ptr %LU, ptr %piv, ptr %sgn)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %fail, label %clr
fail:
  ret i64 1
clr:
  call void @llvm.memset.p0.i64(ptr %Lm, i8 0, i64 512, i1 false)
  call void @llvm.memset.p0.i64(ptr %Um, i8 0, i64 512, i1 false)
  br label %irow
irow:
  %i = phi i64 [ 0, %clr ], [ %inx, %irowend ]
  br label %jcol
jcol:
  %j = phi i64 [ 0, %irow ], [ %jnx, %jnext ]
  %idx = mul i64 %i, %n
  %eidx = add i64 %idx, %j
  %pv = getelementptr inbounds double, ptr %LU, i64 %eidx
  %val = load double, ptr %pv, align 8
  %islow = icmp ugt i64 %i, %j
  %pl = getelementptr inbounds double, ptr %Lm, i64 %eidx
  %pu = getelementptr inbounds double, ptr %Um, i64 %eidx
  br i1 %islow, label %putL, label %putU
putL:
  store double %val, ptr %pl, align 8
  br label %jnext
putU:
  store double %val, ptr %pu, align 8
  br label %jnext
jnext:
  %jnx = add nuw i64 %j, 1
  %jm = icmp ult i64 %jnx, %n
  br i1 %jm, label %jcol, label %setdiag
setdiag:
  %di = mul i64 %i, %n
  %didx = add i64 %di, %i
  %pld = getelementptr inbounds double, ptr %Lm, i64 %didx
  store double 1.0, ptr %pld, align 8
  br label %irowend
irowend:
  %inx = add nuw i64 %i, 1
  %im = icmp ult i64 %inx, %n
  br i1 %im, label %irow, label %mkPA
mkPA:
  ; PA = A, then apply row swaps piv[k]
  %nn = mul i64 %n, %n
  br label %cploop
cploop:
  %ci = phi i64 [ 0, %mkPA ], [ %cin, %cploop ]
  %pca = getelementptr inbounds double, ptr %A, i64 %ci
  %cav = load double, ptr %pca, align 8
  %ppa = getelementptr inbounds double, ptr %PA, i64 %ci
  store double %cav, ptr %ppa, align 8
  %cin = add nuw i64 %ci, 1
  %cm = icmp ult i64 %cin, %nn
  br i1 %cm, label %cploop, label %swaploop
swaploop:
  %k = phi i64 [ 0, %cploop ], [ %kn, %swapend ]
  %ppk = getelementptr inbounds i32, ptr %piv, i64 %k
  %pk = load i32, ptr %ppk, align 4
  %pki = zext i32 %pk to i64
  br label %scol
scol:
  %sc = phi i64 [ 0, %swaploop ], [ %scn, %scol ]
  %krow = mul i64 %k, %n
  %kidx = add i64 %krow, %sc
  %prow = mul i64 %pki, %n
  %pidx = add i64 %prow, %sc
  %pk1 = getelementptr inbounds double, ptr %PA, i64 %kidx
  %pp1 = getelementptr inbounds double, ptr %PA, i64 %pidx
  %t1 = load double, ptr %pk1, align 8
  %t2 = load double, ptr %pp1, align 8
  store double %t2, ptr %pk1, align 8
  store double %t1, ptr %pp1, align 8
  %scn = add nuw i64 %sc, 1
  %scm = icmp ult i64 %scn, %n
  br i1 %scm, label %scol, label %swapend
swapend:
  %kn = add nuw i64 %k, 1
  %km = icmp ult i64 %kn, %n
  br i1 %km, label %swaploop, label %domul
domul:
  %mrc = call i32 @universe_linalg_matmul(ptr %Lm, i64 %n, i64 %n, ptr %Um, i64 %n, i64 %n, ptr %prod)
  %v = call i64 @arrdiff(ptr %prod, ptr %PA, i64 %nn)
  ret i64 %v
}

; solve_viol(A,n,st): pick a random x_true, form b=A*x_true, solve, compare.
define internal i64 @solve_viol(ptr %A, i64 %n, ptr %st) {
entry:
  %xt = alloca [16 x double], align 8
  %b = alloca [16 x double], align 8
  %xc = alloca [16 x double], align 8
  call void @rndmat(ptr %st, ptr %xt, i64 %n)
  %mv = call i32 @universe_linalg_matvec(ptr %A, i64 %n, i64 %n, ptr %xt, ptr %b)
  %rc = call i32 @universe_linalg_solve(ptr %A, i64 %n, ptr %b, i64 1, ptr %xc)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %fail, label %cmp
fail:
  ret i64 1
cmp:
  %v = call i64 @arrdiff(ptr %xc, ptr %xt, i64 %n)
  ret i64 %v
}

; inv_viol(A,n): inv then inv*A compared to I.
define internal i64 @inv_viol(ptr %A, i64 %n) {
entry:
  %iv = alloca [64 x double], align 8
  %prod = alloca [64 x double], align 8
  %I = alloca [64 x double], align 8
  %rc = call i32 @universe_linalg_inv(ptr %A, i64 %n, ptr %iv)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %fail, label %cmp
fail:
  ret i64 1
cmp:
  %mrc = call i32 @universe_linalg_matmul(ptr %iv, i64 %n, i64 %n, ptr %A, i64 %n, i64 %n, ptr %prod)
  %irc = call i32 @universe_linalg_identity(ptr %I, i64 %n)
  %nn = mul i64 %n, %n
  %v = call i64 @arrdiff(ptr %prod, ptr %I, i64 %nn)
  ret i64 %v
}

; qr_viol(A,m,n): Q*R==A + Q'Q==I + R strict-lower zero.
define internal i64 @qr_viol(ptr %A, i64 %m, i64 %n) {
entry:
  %Q = alloca [64 x double], align 8
  %R = alloca [64 x double], align 8
  %prod = alloca [64 x double], align 8
  %Qt = alloca [64 x double], align 8
  %QtQ = alloca [64 x double], align 8
  %I = alloca [64 x double], align 8
  %rc = call i32 @universe_linalg_qr(ptr %A, i64 %m, i64 %n, ptr %Q, ptr %R)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %fail, label %rec
fail:
  ret i64 1
rec:
  %mn = mul i64 %m, %n
  %mrc = call i32 @universe_linalg_matmul(ptr %Q, i64 %m, i64 %m, ptr %R, i64 %m, i64 %n, ptr %prod)
  %v1 = call i64 @arrdiff(ptr %prod, ptr %A, i64 %mn)
  %trc = call i32 @universe_linalg_transpose(ptr %Q, i64 %m, i64 %m, ptr %Qt)
  %qrc = call i32 @universe_linalg_matmul(ptr %Qt, i64 %m, i64 %m, ptr %Q, i64 %m, i64 %m, ptr %QtQ)
  %irc = call i32 @universe_linalg_identity(ptr %I, i64 %m)
  %mm = mul i64 %m, %m
  %v2 = call i64 @arrdiff(ptr %QtQ, ptr %I, i64 %mm)
  %v12 = add i64 %v1, %v2
  ; R strict-lower must be ~0
  br label %rlrow
rlrow:
  %i = phi i64 [ 0, %rec ], [ %inx, %rlrowend ]
  %vlo = phi i64 [ %v12, %rec ], [ %vlon, %rlrowend ]
  br label %rlcol
rlcol:
  %j = phi i64 [ 0, %rlrow ], [ %jnx, %rlcol ]
  %vc = phi i64 [ %vlo, %rlrow ], [ %vcn, %rlcol ]
  %islo = icmp ugt i64 %i, %j
  %ridx0 = mul i64 %i, %n
  %ridx = add i64 %ridx0, %j
  %pr = getelementptr inbounds double, ptr %R, i64 %ridx
  %rv = load double, ptr %pr, align 8
  %isz = call i1 @close(double %rv, double 0.0)
  %notz = xor i1 %isz, true
  %hit = and i1 %islo, %notz
  %inc = zext i1 %hit to i64
  %vcn = add i64 %vc, %inc
  %jnx = add nuw i64 %j, 1
  %jm = icmp ult i64 %jnx, %n
  br i1 %jm, label %rlcol, label %rlrowend
rlrowend:
  %vlon = phi i64 [ %vcn, %rlcol ]
  %inx = add nuw i64 %i, 1
  %im = icmp ult i64 %inx, %m
  br i1 %im, label %rlrow, label %rdone
rdone:
  ret i64 %vlon
}

; chol_viol(st,n): build SPD S = M*M' + n*I, factor, reconstruct L*L' == S.
define internal i64 @chol_viol(ptr %st, i64 %n) {
entry:
  %M = alloca [64 x double], align 8
  %Mt = alloca [64 x double], align 8
  %S = alloca [64 x double], align 8
  %L = alloca [64 x double], align 8
  %Lt = alloca [64 x double], align 8
  %rec = alloca [64 x double], align 8
  %nn = mul i64 %n, %n
  call void @rndmat(ptr %st, ptr %M, i64 %nn)
  %trc = call i32 @universe_linalg_transpose(ptr %M, i64 %n, i64 %n, ptr %Mt)
  %mrc = call i32 @universe_linalg_matmul(ptr %M, i64 %n, i64 %n, ptr %Mt, i64 %n, i64 %n, ptr %S)
  %nf = uitofp i64 %n to double
  br label %diag
diag:
  %i = phi i64 [ 0, %entry ], [ %inx, %diag ]
  %di = mul i64 %i, %n
  %didx = add i64 %di, %i
  %ps = getelementptr inbounds double, ptr %S, i64 %didx
  %sv = load double, ptr %ps, align 8
  %nv = fadd double %sv, %nf
  store double %nv, ptr %ps, align 8
  %inx = add nuw i64 %i, 1
  %im = icmp ult i64 %inx, %n
  br i1 %im, label %diag, label %fact
fact:
  %rc = call i32 @universe_linalg_cholesky(ptr %S, i64 %n, ptr %L)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %fail, label %cmp
fail:
  ret i64 1
cmp:
  %ltrc = call i32 @universe_linalg_transpose(ptr %L, i64 %n, i64 %n, ptr %Lt)
  %rrc = call i32 @universe_linalg_matmul(ptr %L, i64 %n, i64 %n, ptr %Lt, i64 %n, i64 %n, ptr %rec)
  %v = call i64 @arrdiff(ptr %rec, ptr %S, i64 %nn)
  ret i64 %v
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8

  ; ---- fixed-shape reconstruction over random well-conditioned matrices ----
  %A = alloca [64 x double], align 8
  br label %trial
trial:
  %ti = phi i64 [ 0, %entry ], [ %tin, %trial ]
  %vlu = phi i64 [ 0, %entry ], [ %vlun, %trial ]
  %vsol = phi i64 [ 0, %entry ], [ %vsoln, %trial ]
  %vinv = phi i64 [ 0, %entry ], [ %vinvn, %trial ]
  %vchol = phi i64 [ 0, %entry ], [ %vcholn, %trial ]
  ; size cycles 3,4,5,6
  %sz0 = urem i64 %ti, 4
  %sz = add i64 %sz0, 3
  call void @mk_dd(ptr %st, ptr %A, i64 %sz)
  %rlu = call i64 @recon_lu_viol(ptr %A, i64 %sz)
  %vlun = add i64 %vlu, %rlu
  %rsol = call i64 @solve_viol(ptr %A, i64 %sz, ptr %st)
  %vsoln = add i64 %vsol, %rsol
  %rinv = call i64 @inv_viol(ptr %A, i64 %sz)
  %vinvn = add i64 %vinv, %rinv
  %rchol = call i64 @chol_viol(ptr %st, i64 %sz)
  %vcholn = add i64 %vchol, %rchol
  %tin = add nuw i64 %ti, 1
  %tm = icmp ult i64 %tin, 24
  br i1 %tm, label %trial, label %qrtrial

qrtrial:
  ; QR over random tall matrices (m x n, m>=n)
  %B = alloca [64 x double], align 8
  br label %qloop
qloop:
  %qi = phi i64 [ 0, %qrtrial ], [ %qin, %qloop ]
  %vqr = phi i64 [ 0, %qrtrial ], [ %vqrn, %qloop ]
  ; (m,n) in {(4,3),(5,3),(6,4),(5,5)}
  %qc = urem i64 %qi, 4
  %isc0 = icmp eq i64 %qc, 0
  %isc1 = icmp eq i64 %qc, 1
  %isc2 = icmp eq i64 %qc, 2
  %m01 = select i1 %isc0, i64 4, i64 5
  %m012 = select i1 %isc2, i64 6, i64 %m01
  %mm = select i1 %isc1, i64 5, i64 %m012
  %n01 = select i1 %isc2, i64 4, i64 3
  %nn = select i1 %isc0, i64 3, i64 %n01
  %nnf = select i1 %isc1, i64 3, i64 %nn
  ; last case 5x5
  %islast = icmp eq i64 %qc, 3
  %mfin = select i1 %islast, i64 5, i64 %mm
  %nfin = select i1 %islast, i64 5, i64 %nnf
  %mn = mul i64 %mfin, %nfin
  call void @rndmat(ptr %st, ptr %B, i64 %mn)
  %rqr = call i64 @qr_viol(ptr %B, i64 %mfin, i64 %nfin)
  %vqrn = add i64 %vqr, %rqr
  %qin = add nuw i64 %qi, 1
  %qm = icmp ult i64 %qin, 16
  br i1 %qm, label %qloop, label %report

report:
  %oklu = icmp eq i64 %vlun, 0
  call void @ut_check(i1 %oklu, ptr @m.lu)
  %oksol = icmp eq i64 %vsoln, 0
  call void @ut_check(i1 %oksol, ptr @m.solve)
  %okinv = icmp eq i64 %vinvn, 0
  call void @ut_check(i1 %okinv, ptr @m.inv)
  %okchol = icmp eq i64 %vcholn, 0
  call void @ut_check(i1 %okchol, ptr @m.chol)
  %okqr = icmp eq i64 %vqrn, 0
  call void @ut_check(i1 %okqr, ptr @m.qr)

  ; ---- KAT det ----
  %Dd = alloca [9 x double], align 8
  store double 2.0, ptr %Dd, align 8
  %dd1 = getelementptr inbounds double, ptr %Dd, i64 1
  store double 0.0, ptr %dd1
  %dd2 = getelementptr inbounds double, ptr %Dd, i64 2
  store double 0.0, ptr %dd2
  %dd3 = getelementptr inbounds double, ptr %Dd, i64 3
  store double 0.0, ptr %dd3
  %dd4 = getelementptr inbounds double, ptr %Dd, i64 4
  store double 3.0, ptr %dd4
  %dd5 = getelementptr inbounds double, ptr %Dd, i64 5
  store double 0.0, ptr %dd5
  %dd6 = getelementptr inbounds double, ptr %Dd, i64 6
  store double 0.0, ptr %dd6
  %dd7 = getelementptr inbounds double, ptr %Dd, i64 7
  store double 0.0, ptr %dd7
  %dd8 = getelementptr inbounds double, ptr %Dd, i64 8
  store double 4.0, ptr %dd8
  %detd = call double @universe_linalg_det(ptr %Dd, i64 3)
  %okdetd = call i1 @close(double %detd, double 2.400000e+01)
  call void @ut_check(i1 %okdetd, ptr @m.detd)

  ; upper-tri [1 2 3; 0 4 5; 0 0 6] det = 24
  %Ut = alloca [9 x double], align 8
  store double 1.0, ptr %Ut, align 8
  %ut1 = getelementptr inbounds double, ptr %Ut, i64 1
  store double 2.0, ptr %ut1
  %ut2 = getelementptr inbounds double, ptr %Ut, i64 2
  store double 3.0, ptr %ut2
  %ut3 = getelementptr inbounds double, ptr %Ut, i64 3
  store double 0.0, ptr %ut3
  %ut4 = getelementptr inbounds double, ptr %Ut, i64 4
  store double 4.0, ptr %ut4
  %ut5 = getelementptr inbounds double, ptr %Ut, i64 5
  store double 5.0, ptr %ut5
  %ut6 = getelementptr inbounds double, ptr %Ut, i64 6
  store double 0.0, ptr %ut6
  %ut7 = getelementptr inbounds double, ptr %Ut, i64 7
  store double 0.0, ptr %ut7
  %ut8 = getelementptr inbounds double, ptr %Ut, i64 8
  store double 6.0, ptr %ut8
  %dett = call double @universe_linalg_det(ptr %Ut, i64 3)
  %okdett = call i1 @close(double %dett, double 2.400000e+01)
  call void @ut_check(i1 %okdett, ptr @m.dett)

  ; H = [6 1 1; 4 -2 5; 2 8 7] det = -306
  %Hm = alloca [9 x double], align 8
  store double 6.0, ptr %Hm, align 8
  %h1 = getelementptr inbounds double, ptr %Hm, i64 1
  store double 1.0, ptr %h1
  %h2 = getelementptr inbounds double, ptr %Hm, i64 2
  store double 1.0, ptr %h2
  %h3 = getelementptr inbounds double, ptr %Hm, i64 3
  store double 4.0, ptr %h3
  %h4 = getelementptr inbounds double, ptr %Hm, i64 4
  store double -2.0, ptr %h4
  %h5 = getelementptr inbounds double, ptr %Hm, i64 5
  store double 5.0, ptr %h5
  %h6 = getelementptr inbounds double, ptr %Hm, i64 6
  store double 2.0, ptr %h6
  %h7 = getelementptr inbounds double, ptr %Hm, i64 7
  store double 8.0, ptr %h7
  %h8 = getelementptr inbounds double, ptr %Hm, i64 8
  store double 7.0, ptr %h8
  %det3 = call double @universe_linalg_det(ptr %Hm, i64 3)
  %okdet3 = call i1 @close(double %det3, double -3.060000e+02)
  call void @ut_check(i1 %okdet3, ptr @m.det3)

  ; singular [2 4; 4 8] det = 0
  %Sg = alloca [4 x double], align 8
  store double 2.0, ptr %Sg, align 8
  %sg1 = getelementptr inbounds double, ptr %Sg, i64 1
  store double 4.0, ptr %sg1
  %sg2 = getelementptr inbounds double, ptr %Sg, i64 2
  store double 4.0, ptr %sg2
  %sg3 = getelementptr inbounds double, ptr %Sg, i64 3
  store double 8.0, ptr %sg3
  %dets = call double @universe_linalg_det(ptr %Sg, i64 2)
  %okdets = fcmp oeq double %dets, 0.0
  call void @ut_check(i1 %okdets, ptr @m.dets)

  ; ---- KAT solve 3x3: [3 2 -1; 2 -2 4; -1 0.5 -1] x = [1 -2 0] -> [1 -2 -2]
  %As = alloca [9 x double], align 8
  store double 3.0, ptr %As, align 8
  %as1 = getelementptr inbounds double, ptr %As, i64 1
  store double 2.0, ptr %as1
  %as2 = getelementptr inbounds double, ptr %As, i64 2
  store double -1.0, ptr %as2
  %as3 = getelementptr inbounds double, ptr %As, i64 3
  store double 2.0, ptr %as3
  %as4 = getelementptr inbounds double, ptr %As, i64 4
  store double -2.0, ptr %as4
  %as5 = getelementptr inbounds double, ptr %As, i64 5
  store double 4.0, ptr %as5
  %as6 = getelementptr inbounds double, ptr %As, i64 6
  store double -1.0, ptr %as6
  %as7 = getelementptr inbounds double, ptr %As, i64 7
  store double 5.000000e-01, ptr %as7
  %as8 = getelementptr inbounds double, ptr %As, i64 8
  store double -1.0, ptr %as8
  %bs = alloca [3 x double], align 8
  store double 1.0, ptr %bs, align 8
  %bs1 = getelementptr inbounds double, ptr %bs, i64 1
  store double -2.0, ptr %bs1
  %bs2 = getelementptr inbounds double, ptr %bs, i64 2
  store double 0.0, ptr %bs2
  %xs = alloca [3 x double], align 8
  %rcs = call i32 @universe_linalg_solve(ptr %As, i64 3, ptr %bs, i64 1, ptr %xs)
  %xs0 = load double, ptr %xs, align 8
  %pxs1 = getelementptr inbounds double, ptr %xs, i64 1
  %xs1 = load double, ptr %pxs1
  %pxs2 = getelementptr inbounds double, ptr %xs, i64 2
  %xs2 = load double, ptr %pxs2
  %oks0 = call i1 @close(double %xs0, double 1.0)
  %oks1 = call i1 @close(double %xs1, double -2.0)
  %oks2 = call i1 @close(double %xs2, double -2.0)
  %oks01 = and i1 %oks0, %oks1
  %oksk = and i1 %oks01, %oks2
  call void @ut_check(i1 %oksk, ptr @m.solvk)

  ; ---- tri_solve upper: [2 1 1; 0 3 1; 0 0 4] x = [5 4 8] ----
  %Tu = alloca [9 x double], align 8
  store double 2.0, ptr %Tu, align 8
  %tu1 = getelementptr inbounds double, ptr %Tu, i64 1
  store double 1.0, ptr %tu1
  %tu2 = getelementptr inbounds double, ptr %Tu, i64 2
  store double 1.0, ptr %tu2
  %tu3 = getelementptr inbounds double, ptr %Tu, i64 3
  store double 0.0, ptr %tu3
  %tu4 = getelementptr inbounds double, ptr %Tu, i64 4
  store double 3.0, ptr %tu4
  %tu5 = getelementptr inbounds double, ptr %Tu, i64 5
  store double 1.0, ptr %tu5
  %tu6 = getelementptr inbounds double, ptr %Tu, i64 6
  store double 0.0, ptr %tu6
  %tu7 = getelementptr inbounds double, ptr %Tu, i64 7
  store double 0.0, ptr %tu7
  %tu8 = getelementptr inbounds double, ptr %Tu, i64 8
  store double 4.0, ptr %tu8
  %bt = alloca [3 x double], align 8
  store double 5.0, ptr %bt, align 8
  %bt1 = getelementptr inbounds double, ptr %bt, i64 1
  store double 4.0, ptr %bt1
  %bt2 = getelementptr inbounds double, ptr %bt, i64 2
  store double 8.0, ptr %bt2
  %xtu = alloca [3 x double], align 8
  %rtu = call i32 @universe_linalg_tri_solve(ptr %Tu, i64 3, ptr %bt, i32 1, ptr %xtu)
  ; verify T*x == b (residual via matvec)
  %btchk = alloca [3 x double], align 8
  %mvu = call i32 @universe_linalg_matvec(ptr %Tu, i64 3, i64 3, ptr %xtu, ptr %btchk)
  %vtu = call i64 @arrdiff(ptr %btchk, ptr %bt, i64 3)
  %oktu = icmp eq i64 %vtu, 0
  call void @ut_check(i1 %oktu, ptr @m.triu)

  ; ---- tri_solve lower: [2 0 0; 1 3 0; 1 1 4] x = [5 4 8] ----
  %Tl = alloca [9 x double], align 8
  store double 2.0, ptr %Tl, align 8
  %tl1 = getelementptr inbounds double, ptr %Tl, i64 1
  store double 0.0, ptr %tl1
  %tl2 = getelementptr inbounds double, ptr %Tl, i64 2
  store double 0.0, ptr %tl2
  %tl3 = getelementptr inbounds double, ptr %Tl, i64 3
  store double 1.0, ptr %tl3
  %tl4 = getelementptr inbounds double, ptr %Tl, i64 4
  store double 3.0, ptr %tl4
  %tl5 = getelementptr inbounds double, ptr %Tl, i64 5
  store double 0.0, ptr %tl5
  %tl6 = getelementptr inbounds double, ptr %Tl, i64 6
  store double 1.0, ptr %tl6
  %tl7 = getelementptr inbounds double, ptr %Tl, i64 7
  store double 1.0, ptr %tl7
  %tl8 = getelementptr inbounds double, ptr %Tl, i64 8
  store double 4.0, ptr %tl8
  %xtl = alloca [3 x double], align 8
  %rtl = call i32 @universe_linalg_tri_solve(ptr %Tl, i64 3, ptr %bt, i32 0, ptr %xtl)
  %btchk2 = alloca [3 x double], align 8
  %mvl = call i32 @universe_linalg_matvec(ptr %Tl, i64 3, i64 3, ptr %xtl, ptr %btchk2)
  %vtl = call i64 @arrdiff(ptr %btchk2, ptr %bt, i64 3)
  %oktl = icmp eq i64 %vtl, 0
  call void @ut_check(i1 %oktl, ptr @m.tril)

  ; ---- KAT lstsq line fit: A=[1 1;1 2;1 3;1 4], b=[6 5 7 10] -> [3.5 1.4] ----
  %Al = alloca [8 x double], align 8
  store double 1.0, ptr %Al, align 8
  %al1 = getelementptr inbounds double, ptr %Al, i64 1
  store double 1.0, ptr %al1
  %al2 = getelementptr inbounds double, ptr %Al, i64 2
  store double 1.0, ptr %al2
  %al3 = getelementptr inbounds double, ptr %Al, i64 3
  store double 2.0, ptr %al3
  %al4 = getelementptr inbounds double, ptr %Al, i64 4
  store double 1.0, ptr %al4
  %al5 = getelementptr inbounds double, ptr %Al, i64 5
  store double 3.0, ptr %al5
  %al6 = getelementptr inbounds double, ptr %Al, i64 6
  store double 1.0, ptr %al6
  %al7 = getelementptr inbounds double, ptr %Al, i64 7
  store double 4.0, ptr %al7
  %bl = alloca [4 x double], align 8
  store double 6.0, ptr %bl, align 8
  %bl1 = getelementptr inbounds double, ptr %bl, i64 1
  store double 5.0, ptr %bl1
  %bl2 = getelementptr inbounds double, ptr %bl, i64 2
  store double 7.0, ptr %bl2
  %bl3 = getelementptr inbounds double, ptr %bl, i64 3
  store double 10.0, ptr %bl3
  %xl = alloca [2 x double], align 8
  %rls = call i32 @universe_linalg_lstsq(ptr %Al, i64 4, i64 2, ptr %bl, ptr %xl)
  %xl0 = load double, ptr %xl, align 8
  %pxl1 = getelementptr inbounds double, ptr %xl, i64 1
  %xl1 = load double, ptr %pxl1
  %okl0 = call i1 @close(double %xl0, double 3.500000e+00)
  %okl1 = call i1 @close(double %xl1, double 1.400000e+00)
  %oklk = and i1 %okl0, %okl1
  call void @ut_check(i1 %oklk, ptr @m.lsqk)

  ; lstsq == normal equations on a random 6x3 overdetermined system
  %Ao = alloca [64 x double], align 8
  %bo = alloca [16 x double], align 8
  call void @rndmat(ptr %st, ptr %Ao, i64 18)
  call void @rndmat(ptr %st, ptr %bo, i64 6)
  %xls = alloca [4 x double], align 8
  %rlo = call i32 @universe_linalg_lstsq(ptr %Ao, i64 6, i64 3, ptr %bo, ptr %xls)
  ; normal eq: (A'A) x = A'b ; form via transpose+matmul+matvec then solve
  %Aot = alloca [64 x double], align 8
  %ATA = alloca [16 x double], align 8
  %ATb = alloca [4 x double], align 8
  %xne = alloca [4 x double], align 8
  %tr2 = call i32 @universe_linalg_transpose(ptr %Ao, i64 6, i64 3, ptr %Aot)
  %mm2 = call i32 @universe_linalg_matmul(ptr %Aot, i64 3, i64 6, ptr %Ao, i64 6, i64 3, ptr %ATA)
  %mv2 = call i32 @universe_linalg_matvec(ptr %Aot, i64 3, i64 6, ptr %bo, ptr %ATb)
  %sne = call i32 @universe_linalg_solve(ptr %ATA, i64 3, ptr %ATb, i64 1, ptr %xne)
  %vlsq = call i64 @arrdiff(ptr %xls, ptr %xne, i64 3)
  %oklsq = icmp eq i64 %vlsq, 0
  call void @ut_check(i1 %oklsq, ptr @m.lsq)

  ; ---- error paths ----
  %en = call i32 @universe_linalg_det(ptr null, i64 3)
  %enul1 = call i32 @universe_linalg_inv(ptr null, i64 3, ptr %Ao)
  %oknul = icmp eq i32 %enul1, 1
  call void @ut_check(i1 %oknul, ptr @m.enul)

  %edim = call i32 @universe_linalg_lu(ptr %Ao, i64 0, ptr %Ao, ptr %st, ptr %st)
  %okdim = icmp eq i32 %edim, 8
  call void @ut_check(i1 %okdim, ptr @m.edim)

  %esh = call i32 @universe_linalg_lstsq(ptr %Ao, i64 2, i64 3, ptr %bo, ptr %xls)
  %oksh = icmp eq i32 %esh, 8
  call void @ut_check(i1 %oksh, ptr @m.esh)

  %esrhs = call i32 @universe_linalg_solve(ptr %As, i64 3, ptr %bs, i64 0, ptr %xs)
  %oksrhs = icmp eq i32 %esrhs, 8
  call void @ut_check(i1 %oksrhs, ptr @m.esrhs)

  ; singular LU: all-zero matrix (zero pivot) -> 11
  %Sz = alloca [9 x double], align 8
  call void @llvm.memset.p0.i64(ptr %Sz, i8 0, i64 72, i1 false)
  %LUz = alloca [9 x double], align 8
  %pvz = alloca [3 x i32], align 4
  %sgz = alloca double, align 8
  %esing = call i32 @universe_linalg_lu(ptr %Sz, i64 3, ptr %LUz, ptr %pvz, ptr %sgz)
  %oksing = icmp eq i32 %esing, 11
  call void @ut_check(i1 %oksing, ptr @m.esing)

  ; non-SPD Cholesky: [1 2; 2 1] -> 13
  %Np = alloca [4 x double], align 8
  store double 1.0, ptr %Np, align 8
  %np1 = getelementptr inbounds double, ptr %Np, i64 1
  store double 2.0, ptr %np1
  %np2 = getelementptr inbounds double, ptr %Np, i64 2
  store double 2.0, ptr %np2
  %np3 = getelementptr inbounds double, ptr %Np, i64 3
  store double 1.0, ptr %np3
  %Lnp = alloca [4 x double], align 8
  %espd = call i32 @universe_linalg_cholesky(ptr %Np, i64 2, ptr %Lnp)
  %okspd = icmp eq i32 %espd, 13
  call void @ut_check(i1 %okspd, ptr @m.espd)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin
bench:
  %Ab = alloca [64 x double], align 8
  %bb = alloca [16 x double], align 8
  %xb = alloca [16 x double], align 8
  call void @mk_dd(ptr %st, ptr %Ab, i64 8)
  call void @rndmat(ptr %st, ptr %bb, i64 8)
  %t0 = call double @ut_now_sec()
  br label %bloop
bloop:
  %bi = phi i64 [ 0, %bench ], [ %bin, %bloop ]
  %rcb = call i32 @universe_linalg_solve(ptr %Ab, i64 8, ptr %bb, i64 1, ptr %xb)
  %bin = add nuw i64 %bi, 1
  %bm = icmp ult i64 %bin, 200000
  br i1 %bm, label %bloop, label %bdone
bdone:
  %t1b = call double @ut_now_sec()
  %dt = fsub double %t1b, %t0
  %ns = fmul double %dt, 5.000000e-06
  %nsper = fdiv double %ns, 1.0
  %perop0 = fdiv double %dt, 2.000000e+05
  %perop = fmul double %perop0, 1.000000e+09
  call i32 (ptr, ...) @printf(ptr @b.fmt, i64 8, i64 8, double %perop)
  br label %fin
fin:
  %r = call i32 @ut_summary()
  ret i32 %r
}
