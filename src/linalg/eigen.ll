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

; universe_linalg_{eigen_sym,svd} — ITERATIVE dense eigen/singular decomposition
; (Julia LinearAlgebra names: `eigen`, `svd`). Row-major, contiguous f64; the
; caller owns all storage. A single contiguous scratch block is malloc'd once
; per call (never a per-iteration/per-element allocation) and freed on exit.
;
; SCOPE — REAL SYMMETRIC ONLY (stated deliberately):
;   * universe_linalg_eigen_sym assumes A is a REAL SYMMETRIC n x n matrix. It
;     reads the full A but treats it as symmetric (only the symmetric part is
;     meaningful). General NON-SYMMETRIC eigen (complex/defective spectra) is
;     EXPLICITLY OUT OF SCOPE — it needs the shifted-QR / Hessenberg algorithm
;     and is left for a future module. Do not feed a non-symmetric matrix and
;     expect its (complex) eigenpairs.
;   * universe_linalg_svd is a REAL SVD for any m x n via one-sided Jacobi on
;     the columns; it makes no symmetry assumption on A.
;
; DESIGN — eigen_sym: CYCLIC (two-sided) JACOBI ROTATIONS.
;   Algorithm class: Jacobi is chosen over shifted-QR because it is (a) trivial
;   to make numerically robust (each rotation is an exact orthogonal similarity,
;   so V stays orthonormal to rounding and every eigenpair residual is tiny),
;   (b) branch-light and data-parallel in its inner apply loop, and (c) always
;   convergent for symmetric input (off-diagonal Frobenius mass decreases every
;   sweep, quadratically near the end). We cap sweeps at 100 and return 12
;   (NOT_CONVERGED, per the plan's numeric-failure convention for this domain)
;   if that cap is hit — unreachable for well-formed symmetric input.
;
;   State: a working copy W = A (n x n, kept FULLY SYMMETRIC every step) and R
;   (n x n) that accumulates the LEFT product of rotations (R = G_k^T ... G_1^T),
;   so R's ROWS are the eigenvectors (R = Q^T where A = Q diag(vals) Q^T). Both
;   W-row-update and R-row-update are then CONTIGUOUS (row-major), which is the
;   whole point: the rotation-apply kernel streams two contiguous rows.
;
;   One rotation zeroing W[p,q] (p<q): compute c,s that diagonalise the 2x2
;   [[app,apq],[apq,aqq]] (theta=(aqq-app)/(2 apq); t=sign(theta)/(|theta|+
;   sqrt(theta^2+1)); c=1/sqrt(1+t^2); s=t c). B = G^T A G touches rows AND
;   columns p,q; using symmetry (W[i,p]=W[p,i]) we do it as: (1) a SIMD row
;   update of rows p,q over all columns j (B[p,j]=c W[p,j]-s W[q,j], B[q,j]=
;   s W[p,j]+c W[q,j]) — this is the hot `eig_rot_pair` kernel; (2) overwrite
;   the 2x2 block (B[p,p]=app-t*apq, B[q,q]=aqq+t*apq, B[p,q]=B[q,p]=0); (3)
;   MIRROR the two updated rows back into columns p,q to restore symmetry
;   (strided scalar, O(n), cold relative to the SIMD apply). R is updated by the
;   same `eig_rot_pair` on rows p,q. Final eigenvalues are diag(W); eigenvectors
;   are R's rows transposed into the caller's column-major `vecs` output; both
;   are sorted ASCENDING by eigenvalue (Julia convention).
;
; DESIGN — svd: ONE-SIDED JACOBI on columns.
;   We store a COLUMN-MAJOR working copy Wc of A (so each column is CONTIGUOUS
;   and the column-pair dot products / rotation applies are SIMD-friendly) and a
;   column-major accumulator Vc = I (n x n). Repeatedly, for each column pair
;   (i,j) we form the 2x2 gram [[a,g],[g,b]] (a=coli.coli, b=colj.colj,
;   g=coli.colj — three contiguous SIMD dots) and, if |g|>2^-50 sqrt(a b),
;   rotate columns i,j of Wc and Vc by the same c,s that diagonalise the gram
;   (identical rotation math to eigen). A sweep that performs zero rotations =
;   converged (cap 100 -> 12). Then sigma_i = ||Wc col i||, U col i = Wc col
;   i/sigma_i, V = Vc; outputs sorted DESCENDING by sigma. Because Wc_final =
;   A V and Vc = V exactly (product of exact rotations), U diag(S) V^T = A to
;   rounding regardless of how tight convergence is. Works for any m,n: when
;   rank < n the surplus singular values are 0 (their U columns are 0), and the
;   factorisation stays exact. Output shapes: U is m x n, S is length n, Vt is
;   n x n (V^T). For m<n this is a valid (non-minimal) SVD; the zero singular
;   values carry the rank deficiency.
;
;   AtA link: the n singular values squared are the eigenvalues of A^T A, so
;   svd cross-checks against eigen_sym(A^T A) in the test.
;
; SIMD: the rotation-apply (`eig_rot_pair`) and the column/row dot (`eig_dotc`)
; are the hot data-parallel kernels — 128-bit <2 x double> primary path with a
; scalar tail for the odd remainder (the tail is also the scalar fallback for a
; hypothetical no-vector target). Correctness is validated END-TO-END by the
; mathematical KATs (A v = lambda v, V^T V = I, U diag(S) V^T = A), which are a
; STRONGER oracle than a scalar twin of the kernel.
;
; API (f64, row-major; element (i,j) at A[i*n+j]):
;   i32 universe_linalg_eigen_sym(A, n, vals_out, vecs_out)  ; symmetric eigen
;   i32 universe_linalg_svd(A, m, n, U_out, S_out, Vt_out)   ; real SVD
; Errors i32: 0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW,
;   8 INVALID_ARG (negative dimension), 12 NOT_CONVERGED (sweep cap hit).

declare ptr @malloc(i64)
declare void @free(ptr)
declare double @llvm.sqrt.f64(double)
declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; ==================================================================== helpers

; eig_cs — rotation cosine/sine that diagonalises the symmetric 2x2
; [[app,apq],[apq,aqq]] (apq must be != 0, guaranteed by the caller). Returns
; <c, s>. Pure.
define internal <2 x double> @eig_cs(double %app, double %aqq, double %apq) #1 {
entry:
  %d = fsub double %aqq, %app
  %tapq = fmul double 2.0, %apq
  %theta = fdiv double %d, %tapq
  %th2 = fmul double %theta, %theta
  %th2p1 = fadd double %th2, 1.0
  %sq = call double @llvm.sqrt.f64(double %th2p1)
  %ath = call double @llvm.fabs.f64(double %theta)
  %den = fadd double %ath, %sq
  %sgn = call double @llvm.copysign.f64(double 1.0, double %theta)
  %t = fdiv double %sgn, %den
  %t2 = fmul double %t, %t
  %t2p1 = fadd double %t2, 1.0
  %sqc = call double @llvm.sqrt.f64(double %t2p1)
  %c = fdiv double 1.0, %sqc
  %s = fmul double %t, %c
  %v0 = insertelement <2 x double> undef, double %c, i64 0
  %v1 = insertelement <2 x double> %v0, double %s, i64 1
  ret <2 x double> %v1
}

; eig_dotc — SIMD dot product of two contiguous f64 arrays. <2 x double>
; primary path + scalar tail (also the scalar fallback/oracle).
define internal double @eig_dotc(ptr readonly %x, ptr readonly %y, i64 %n) #2 {
entry:
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %sinit
vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %acc = phi <2 x double> [ zeroinitializer, %entry ], [ %accn, %vloop ]
  %px = getelementptr inbounds nuw double, ptr %x, i64 %i
  %vx = load <2 x double>, ptr %px, align 8
  %py = getelementptr inbounds nuw double, ptr %y, i64 %i
  %vy = load <2 x double>, ptr %py, align 8
  %mm = fmul <2 x double> %vx, %vy
  %accn = fadd <2 x double> %acc, %mm
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone
vdone:
  %hs = call double @llvm.vector.reduce.fadd.v2f64(double -0.0, <2 x double> %accn)
  br label %tail
sinit:
  br label %tail
tail:
  %base0 = phi double [ 0.0, %sinit ], [ %hs, %vdone ]
  %start = phi i64 [ 0, %sinit ], [ %inext, %vdone ]
  br label %tloop
tloop:
  %j = phi i64 [ %start, %tail ], [ %jn, %tbody ]
  %sac = phi double [ %base0, %tail ], [ %sacn, %tbody ]
  %tdone = icmp uge i64 %j, %n
  br i1 %tdone, label %tret, label %tbody
tbody:
  %tx = getelementptr inbounds nuw double, ptr %x, i64 %j
  %fx = load double, ptr %tx, align 8
  %ty = getelementptr inbounds nuw double, ptr %y, i64 %j
  %fy = load double, ptr %ty, align 8
  %pp = fmul double %fx, %fy
  %sacn = fadd double %sac, %pp
  %jn = add nuw i64 %j, 1
  br label %tloop
tret:
  ret double %sac
}

; eig_rot_pair — in-place Givens rotation of two contiguous f64 rows/columns:
;   x' = c*x - s*y ,  y' = s*x + c*y   (elementwise over n).
; This is the hot rotation-apply kernel: <2 x double> primary + scalar tail.
; x and y never alias (distinct rows/columns), so the in-place vector stores are
; safe.
define internal void @eig_rot_pair(ptr %x, ptr %y, double %c, double %s, i64 %n) #3 {
entry:
  %c0 = insertelement <2 x double> undef, double %c, i64 0
  %vc = shufflevector <2 x double> %c0, <2 x double> undef, <2 x i32> zeroinitializer
  %s0 = insertelement <2 x double> undef, double %s, i64 0
  %vs = shufflevector <2 x double> %s0, <2 x double> undef, <2 x i32> zeroinitializer
  %has2 = icmp uge i64 %n, 2
  br i1 %has2, label %vloop, label %tailchk
vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %px = getelementptr inbounds nuw double, ptr %x, i64 %i
  %a = load <2 x double>, ptr %px, align 8
  %py = getelementptr inbounds nuw double, ptr %y, i64 %i
  %b = load <2 x double>, ptr %py, align 8
  %ca = fmul <2 x double> %vc, %a
  %sb = fmul <2 x double> %vs, %b
  %np = fsub <2 x double> %ca, %sb
  %sa = fmul <2 x double> %vs, %a
  %cb = fmul <2 x double> %vc, %b
  %nq = fadd <2 x double> %sa, %cb
  store <2 x double> %np, ptr %px, align 8
  store <2 x double> %nq, ptr %py, align 8
  %inext = add nuw i64 %i, 2
  %lim = sub nuw i64 %n, 2
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone
vdone:
  br label %tailchk
tailchk:
  %start = phi i64 [ 0, %entry ], [ %inext, %vdone ]
  %rem = icmp ult i64 %start, %n
  br i1 %rem, label %tbody, label %ret
tbody:
  %tx = getelementptr inbounds nuw double, ptr %x, i64 %start
  %ax = load double, ptr %tx, align 8
  %ty = getelementptr inbounds nuw double, ptr %y, i64 %start
  %by = load double, ptr %ty, align 8
  %cax = fmul double %c, %ax
  %sbx = fmul double %s, %by
  %npx = fsub double %cax, %sbx
  %sax = fmul double %s, %ax
  %cbx = fmul double %c, %by
  %nqx = fadd double %sax, %cbx
  store double %npx, ptr %tx, align 8
  store double %nqx, ptr %ty, align 8
  br label %ret
ret:
  ret void
}

; eig_offss — sum of squares of the strict-upper off-diagonal of W (n x n).
define internal double @eig_offss(ptr readonly %W, i64 %n) #4 {
entry:
  br label %ploop
ploop:
  %p = phi i64 [ 0, %entry ], [ %pn, %pinc ]
  %acc = phi double [ 0.0, %entry ], [ %accp, %pinc ]
  %prow = mul i64 %p, %n
  %pn = add nuw i64 %p, 1
  br label %qloop
qloop:
  %q = phi i64 [ %pn, %ploop ], [ %qn, %qbody ]
  %qacc = phi double [ %acc, %ploop ], [ %qaccn, %qbody ]
  %qdone = icmp uge i64 %q, %n
  br i1 %qdone, label %pinc, label %qbody
qbody:
  %idx = add i64 %prow, %q
  %pe = getelementptr inbounds nuw double, ptr %W, i64 %idx
  %v = load double, ptr %pe, align 8
  %sqv = fmul double %v, %v
  %qaccn = fadd double %qacc, %sqv
  %qn = add nuw i64 %q, 1
  br label %qloop
pinc:
  %accp = phi double [ %qacc, %qloop ]
  %pmore = icmp ult i64 %pn, %n
  br i1 %pmore, label %ploop, label %done
done:
  %final = phi double [ %accp, %pinc ]
  ret double %final
}

; eig_argsort — selection-sort an index permutation of [0,n) by keys[idx[.]].
; desc != 0 -> descending, else ascending.
define internal void @eig_argsort(ptr %idx, ptr readonly %keys, i64 %n, i32 %desc) #5 {
entry:
  %isdesc = icmp ne i32 %desc, 0
  br label %initl
initl:
  %k = phi i64 [ 0, %entry ], [ %kn, %initl ]
  %ip = getelementptr inbounds nuw i64, ptr %idx, i64 %k
  store i64 %k, ptr %ip, align 8
  %kn = add nuw i64 %k, 1
  %kmore = icmp ult i64 %kn, %n
  br i1 %kmore, label %initl, label %sortchk
sortchk:
  %lt2 = icmp ult i64 %n, 2
  br i1 %lt2, label %ret, label %aloop
aloop:
  %a = phi i64 [ 0, %sortchk ], [ %an, %aend ]
  %an = add nuw i64 %a, 1
  br label %bloop
bloop:
  %b = phi i64 [ %an, %aloop ], [ %bn, %bbody ]
  %pos = phi i64 [ %a, %aloop ], [ %posn, %bbody ]
  %bdone = icmp uge i64 %b, %n
  br i1 %bdone, label %swap, label %bbody
bbody:
  %ibp = getelementptr inbounds nuw i64, ptr %idx, i64 %b
  %ib = load i64, ptr %ibp, align 8
  %ipp = getelementptr inbounds nuw i64, ptr %idx, i64 %pos
  %ipos = load i64, ptr %ipp, align 8
  %kbp = getelementptr inbounds nuw double, ptr %keys, i64 %ib
  %kb = load double, ptr %kbp, align 8
  %kpp = getelementptr inbounds nuw double, ptr %keys, i64 %ipos
  %kp = load double, ptr %kpp, align 8
  %clt = fcmp olt double %kb, %kp
  %cgt = fcmp ogt double %kb, %kp
  %bet = select i1 %isdesc, i1 %cgt, i1 %clt
  %posn = select i1 %bet, i64 %b, i64 %pos
  %bn = add nuw i64 %b, 1
  br label %bloop
swap:
  %iap = getelementptr inbounds nuw i64, ptr %idx, i64 %a
  %va = load i64, ptr %iap, align 8
  %spp = getelementptr inbounds nuw i64, ptr %idx, i64 %pos
  %vp = load i64, ptr %spp, align 8
  store i64 %vp, ptr %iap, align 8
  store i64 %va, ptr %spp, align 8
  %amore = icmp ult i64 %an, %n
  br i1 %amore, label %aend, label %ret
aend:
  br label %aloop
ret:
  ret void
}

; eig_sweep — one full cyclic-Jacobi sweep over all pairs (p<q) of symmetric W,
; accumulating rotations into R's rows.
define internal void @eig_sweep(ptr %W, ptr %R, i64 %n) #5 {
entry:
  %lt2 = icmp ult i64 %n, 2
  br i1 %lt2, label %ret, label %ploop
ploop:
  %p = phi i64 [ 0, %entry ], [ %pn, %pinc ]
  %prow = mul i64 %p, %n
  %pn = add nuw i64 %p, 1
  %ppidx = add i64 %prow, %p
  br label %qloop
qloop:
  %q = phi i64 [ %pn, %ploop ], [ %qn, %qcont ]
  %qdone = icmp uge i64 %q, %n
  br i1 %qdone, label %pinc, label %qbody
qbody:
  %pqidx = add i64 %prow, %q
  %pqp = getelementptr inbounds nuw double, ptr %W, i64 %pqidx
  %apq = load double, ptr %pqp, align 8
  %isz = fcmp oeq double %apq, 0.0
  br i1 %isz, label %qcont, label %dorot
dorot:
  %appp = getelementptr inbounds nuw double, ptr %W, i64 %ppidx
  %app = load double, ptr %appp, align 8
  %qrow = mul i64 %q, %n
  %qqidx = add i64 %qrow, %q
  %qqp = getelementptr inbounds nuw double, ptr %W, i64 %qqidx
  %aqq = load double, ptr %qqp, align 8
  %cs = call <2 x double> @eig_cs(double %app, double %aqq, double %apq)
  %c = extractelement <2 x double> %cs, i64 0
  %s = extractelement <2 x double> %cs, i64 1
  %rowpp = getelementptr inbounds nuw double, ptr %W, i64 %prow
  %rowqp = getelementptr inbounds nuw double, ptr %W, i64 %qrow
  call void @eig_rot_pair(ptr %rowpp, ptr %rowqp, double %c, double %s, i64 %n)
  %t = fdiv double %s, %c
  %h = fmul double %t, %apq
  %newapp = fsub double %app, %h
  %newaqq = fadd double %aqq, %h
  store double %newapp, ptr %appp, align 8
  store double %newaqq, ptr %qqp, align 8
  store double 0.0, ptr %pqp, align 8
  %qpidx = add i64 %qrow, %p
  %qpp = getelementptr inbounds nuw double, ptr %W, i64 %qpidx
  store double 0.0, ptr %qpp, align 8
  br label %mloop
mloop:
  %i = phi i64 [ 0, %dorot ], [ %in, %mloop ]
  %irow = mul i64 %i, %n
  %piIdx = add i64 %prow, %i
  %ppi = getelementptr inbounds nuw double, ptr %W, i64 %piIdx
  %vpi = load double, ptr %ppi, align 8
  %ipIdx = add i64 %irow, %p
  %pip = getelementptr inbounds nuw double, ptr %W, i64 %ipIdx
  store double %vpi, ptr %pip, align 8
  %qiIdx = add i64 %qrow, %i
  %pqi = getelementptr inbounds nuw double, ptr %W, i64 %qiIdx
  %vqi = load double, ptr %pqi, align 8
  %iqIdx = add i64 %irow, %q
  %piq = getelementptr inbounds nuw double, ptr %W, i64 %iqIdx
  store double %vqi, ptr %piq, align 8
  %in = add nuw i64 %i, 1
  %imore = icmp ult i64 %in, %n
  br i1 %imore, label %mloop, label %rrot
rrot:
  %rrowp = getelementptr inbounds nuw double, ptr %R, i64 %prow
  %rrowq = getelementptr inbounds nuw double, ptr %R, i64 %qrow
  call void @eig_rot_pair(ptr %rrowp, ptr %rrowq, double %c, double %s, i64 %n)
  br label %qcont
qcont:
  %qn = add nuw i64 %q, 1
  br label %qloop
pinc:
  %pmore = icmp ult i64 %pn, %n
  br i1 %pmore, label %ploop, label %ret
ret:
  ret void
}

; svd_sweep — one one-sided-Jacobi sweep over column pairs (i<j) of Wc (column-
; major, m rows) with accumulator Vc (column-major, n rows). Returns the number
; of rotations performed (0 => converged).
define internal i64 @svd_sweep(ptr %Wc, ptr %Vc, i64 %m, i64 %n) #5 {
entry:
  %lt2 = icmp ult i64 %n, 2
  br i1 %lt2, label %ret0, label %iloop
iloop:
  %i = phi i64 [ 0, %entry ], [ %in, %iinc ]
  %cnt = phi i64 [ 0, %entry ], [ %cntI, %iinc ]
  %in = add nuw i64 %i, 1
  %icol = mul i64 %i, %m
  %ivec = mul i64 %i, %n
  %coli = getelementptr inbounds nuw double, ptr %Wc, i64 %icol
  %vci = getelementptr inbounds nuw double, ptr %Vc, i64 %ivec
  br label %jloop
jloop:
  %j = phi i64 [ %in, %iloop ], [ %jn, %jcont ]
  %cntj = phi i64 [ %cnt, %iloop ], [ %cntjn, %jcont ]
  %jdone = icmp uge i64 %j, %n
  br i1 %jdone, label %iinc, label %jbody
jbody:
  %jcol = mul i64 %j, %m
  %jvec = mul i64 %j, %n
  %colj = getelementptr inbounds nuw double, ptr %Wc, i64 %jcol
  %vcj = getelementptr inbounds nuw double, ptr %Vc, i64 %jvec
  %a = call double @eig_dotc(ptr %coli, ptr %coli, i64 %m)
  %b = call double @eig_dotc(ptr %colj, ptr %colj, i64 %m)
  %g = call double @eig_dotc(ptr %coli, ptr %colj, i64 %m)
  %absg = call double @llvm.fabs.f64(double %g)
  %ab = fmul double %a, %b
  %sab = call double @llvm.sqrt.f64(double %ab)
  %thr = fmul double %sab, 0x3CD0000000000000
  %skip = fcmp ole double %absg, %thr
  br i1 %skip, label %jcont0, label %dorot
dorot:
  %cs = call <2 x double> @eig_cs(double %a, double %b, double %g)
  %c = extractelement <2 x double> %cs, i64 0
  %s = extractelement <2 x double> %cs, i64 1
  call void @eig_rot_pair(ptr %coli, ptr %colj, double %c, double %s, i64 %m)
  call void @eig_rot_pair(ptr %vci, ptr %vcj, double %c, double %s, i64 %n)
  %cntup = add nuw i64 %cntj, 1
  br label %jcont
jcont0:
  br label %jcont
jcont:
  %cntjn = phi i64 [ %cntj, %jcont0 ], [ %cntup, %dorot ]
  %jn = add nuw i64 %j, 1
  br label %jloop
iinc:
  %cntI = phi i64 [ %cntj, %jloop ]
  %imore = icmp ult i64 %in, %n
  br i1 %imore, label %iloop, label %retn
retn:
  ret i64 %cntI
ret0:
  ret i64 0
}

; ==================================================================== eigen_sym

define i32 @universe_linalg_eigen_sym(ptr %A, i64 %n, ptr %vals, ptr %vecs) #0 {
entry:
  %an = icmp eq ptr %A, null
  %vn = icmp eq ptr %vals, null
  %en = icmp eq ptr %vecs, null
  %nb0 = or i1 %an, %vn
  %nb = or i1 %nb0, %en
  br i1 %nb, label %enull, label %chkn
chkn:
  %neg = icmp slt i64 %n, 0
  br i1 %neg, label %earg, label %chk0
chk0:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %retok0, label %sizes
sizes:
  %np1 = add i64 %n, 1
  %o1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %o1, 0
  %ov1 = extractvalue { i64, i1 } %o1, 1
  br i1 %ov1, label %eovf, label %sz2
sz2:
  %o2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %nn, i64 %nn)
  %twonn = extractvalue { i64, i1 } %o2, 0
  %ov2 = extractvalue { i64, i1 } %o2, 1
  %o3 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %twonn, i64 %n)
  %tn1 = extractvalue { i64, i1 } %o3, 0
  %ov3 = extractvalue { i64, i1 } %o3, 1
  %o4 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %tn1, i64 %n)
  %totd = extractvalue { i64, i1 } %o4, 0
  %ov4 = extractvalue { i64, i1 } %o4, 1
  %o5 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %totd, i64 8)
  %bytes = extractvalue { i64, i1 } %o5, 0
  %ov5 = extractvalue { i64, i1 } %o5, 1
  %ovA = or i1 %ov2, %ov3
  %ovB = or i1 %ov4, %ov5
  %ovC = or i1 %ovA, %ovB
  br i1 %ovC, label %eovf, label %domalloc
domalloc:
  %base = call ptr @malloc(i64 %bytes)
  %mnull = icmp eq ptr %base, null
  br i1 %mnull, label %eoom, label %haveM
haveM:
  %R = getelementptr inbounds nuw double, ptr %base, i64 %nn
  %idx = getelementptr inbounds nuw double, ptr %base, i64 %twonn
  %keys = getelementptr inbounds nuw double, ptr %base, i64 %tn1
  %cbytes = shl i64 %nn, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %base, ptr %A, i64 %cbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %R, i8 0, i64 %cbytes, i1 false)
  br label %idl
idl:
  %di = phi i64 [ 0, %haveM ], [ %din, %idl ]
  %didx = mul i64 %di, %np1
  %drp = getelementptr inbounds nuw double, ptr %R, i64 %didx
  store double 1.0, ptr %drp, align 8
  %din = add nuw i64 %di, 1
  %dmore = icmp ult i64 %din, %n
  br i1 %dmore, label %idl, label %afterfro
afterfro:
  %fro = call double @eig_dotc(ptr %base, ptr %base, i64 %nn)
  %tol2 = fmul double %fro, 0x39B0000000000000
  br label %sweeptop
sweeptop:
  %sweep = phi i64 [ 0, %afterfro ], [ %sweepn, %dosweep ]
  %offss = call double @eig_offss(ptr %base, i64 %n)
  %conv = fcmp ole double %offss, %tol2
  br i1 %conv, label %converged, label %chkcap
chkcap:
  %capped = icmp uge i64 %sweep, 100
  br i1 %capped, label %enotconv, label %dosweep
dosweep:
  call void @eig_sweep(ptr %base, ptr %R, i64 %n)
  %sweepn = add nuw i64 %sweep, 1
  br label %sweeptop
converged:
  br label %dkl
dkl:
  %e = phi i64 [ 0, %converged ], [ %een, %dkl ]
  %ediag = mul i64 %e, %np1
  %wep = getelementptr inbounds nuw double, ptr %base, i64 %ediag
  %wev = load double, ptr %wep, align 8
  %kep = getelementptr inbounds nuw double, ptr %keys, i64 %e
  store double %wev, ptr %kep, align 8
  %een = add nuw i64 %e, 1
  %emore = icmp ult i64 %een, %n
  br i1 %emore, label %dkl, label %dosort
dosort:
  call void @eig_argsort(ptr %idx, ptr %keys, i64 %n, i32 0)
  br label %okl
okl:
  %k = phi i64 [ 0, %dosort ], [ %kn, %kend ]
  %kip = getelementptr inbounds nuw i64, ptr %idx, i64 %k
  %ek = load i64, ptr %kip, align 8
  %kkp = getelementptr inbounds nuw double, ptr %keys, i64 %ek
  %lam = load double, ptr %kkp, align 8
  %vkp = getelementptr inbounds nuw double, ptr %vals, i64 %k
  store double %lam, ptr %vkp, align 8
  %ekrow = mul i64 %ek, %n
  br label %vil
vil:
  %vi = phi i64 [ 0, %okl ], [ %vin, %vil ]
  %rsrc = add i64 %ekrow, %vi
  %rp = getelementptr inbounds nuw double, ptr %R, i64 %rsrc
  %rv = load double, ptr %rp, align 8
  %virow = mul i64 %vi, %n
  %dst = add i64 %virow, %k
  %dp = getelementptr inbounds nuw double, ptr %vecs, i64 %dst
  store double %rv, ptr %dp, align 8
  %vin = add nuw i64 %vi, 1
  %vmore = icmp ult i64 %vin, %n
  br i1 %vmore, label %vil, label %kend
kend:
  %kn = add nuw i64 %k, 1
  %kmore = icmp ult i64 %kn, %n
  br i1 %kmore, label %okl, label %freeret
freeret:
  call void @free(ptr %base)
  ret i32 0
enotconv:
  call void @free(ptr %base)
  ret i32 12
retok0:
  ret i32 0
enull:
  ret i32 1
earg:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

; ==================================================================== svd

define i32 @universe_linalg_svd(ptr %A, i64 %m, i64 %n, ptr %U, ptr %S, ptr %Vt) #0 {
entry:
  %an = icmp eq ptr %A, null
  %un = icmp eq ptr %U, null
  %sn = icmp eq ptr %S, null
  %tn = icmp eq ptr %Vt, null
  %nb0 = or i1 %an, %un
  %nb1 = or i1 %sn, %tn
  %nb = or i1 %nb0, %nb1
  br i1 %nb, label %enull, label %chkn
chkn:
  %mneg = icmp slt i64 %m, 0
  %nneg = icmp slt i64 %n, 0
  %negx = or i1 %mneg, %nneg
  br i1 %negx, label %earg, label %chk0
chk0:
  %mz = icmp eq i64 %m, 0
  %nz = icmp eq i64 %n, 0
  %ez = or i1 %mz, %nz
  br i1 %ez, label %retok0, label %sizes
sizes:
  %o1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %mn = extractvalue { i64, i1 } %o1, 0
  %ov1 = extractvalue { i64, i1 } %o1, 1
  %o2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %o2, 0
  %ov2 = extractvalue { i64, i1 } %o2, 1
  %o3 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %mn, i64 %nn)
  %s1 = extractvalue { i64, i1 } %o3, 0
  %ov3 = extractvalue { i64, i1 } %o3, 1
  %o4 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s1, i64 %n)
  %s2 = extractvalue { i64, i1 } %o4, 0
  %ov4 = extractvalue { i64, i1 } %o4, 1
  %o5 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s2, i64 %n)
  %totd = extractvalue { i64, i1 } %o5, 0
  %ov5 = extractvalue { i64, i1 } %o5, 1
  %o6 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %totd, i64 8)
  %bytes = extractvalue { i64, i1 } %o6, 0
  %ov6 = extractvalue { i64, i1 } %o6, 1
  %ovA = or i1 %ov1, %ov2
  %ovB = or i1 %ov3, %ov4
  %ovC = or i1 %ov5, %ov6
  %ovD = or i1 %ovA, %ovB
  %ovE = or i1 %ovC, %ovD
  br i1 %ovE, label %eovf, label %domalloc
domalloc:
  %base = call ptr @malloc(i64 %bytes)
  %mnull = icmp eq ptr %base, null
  br i1 %mnull, label %eoom, label %haveM
haveM:
  %Vc = getelementptr inbounds nuw double, ptr %base, i64 %mn
  %idx = getelementptr inbounds nuw double, ptr %base, i64 %s1
  %nrm = getelementptr inbounds nuw double, ptr %base, i64 %s2
  br label %rfill
rfill:
  %r = phi i64 [ 0, %haveM ], [ %rnx, %rend ]
  %rrow = mul i64 %r, %n
  br label %ifill
ifill:
  %ii = phi i64 [ 0, %rfill ], [ %iinx, %ifill ]
  %aidx = add i64 %rrow, %ii
  %ap = getelementptr inbounds nuw double, ptr %A, i64 %aidx
  %av = load double, ptr %ap, align 8
  %wc = mul i64 %ii, %m
  %widx = add i64 %wc, %r
  %wp = getelementptr inbounds nuw double, ptr %base, i64 %widx
  store double %av, ptr %wp, align 8
  %iinx = add nuw i64 %ii, 1
  %imore = icmp ult i64 %iinx, %n
  br i1 %imore, label %ifill, label %rend
rend:
  %rnx = add nuw i64 %r, 1
  %rmore = icmp ult i64 %rnx, %m
  br i1 %rmore, label %rfill, label %vinit
vinit:
  %nnbytes = shl i64 %nn, 3
  call void @llvm.memset.p0.i64(ptr %Vc, i8 0, i64 %nnbytes, i1 false)
  %np1 = add i64 %n, 1
  br label %vidl
vidl:
  %vi = phi i64 [ 0, %vinit ], [ %vin, %vidl ]
  %vdidx = mul i64 %vi, %np1
  %vdp = getelementptr inbounds nuw double, ptr %Vc, i64 %vdidx
  store double 1.0, ptr %vdp, align 8
  %vin = add nuw i64 %vi, 1
  %vmore = icmp ult i64 %vin, %n
  br i1 %vmore, label %vidl, label %svtop
svtop:
  %sweep = phi i64 [ 0, %vidl ], [ %swn, %svcont ]
  %cnt = call i64 @svd_sweep(ptr %base, ptr %Vc, i64 %m, i64 %n)
  %cvd = icmp eq i64 %cnt, 0
  br i1 %cvd, label %svconv, label %svchkcap
svchkcap:
  %capped = icmp uge i64 %sweep, 100
  br i1 %capped, label %svnotconv, label %svcont
svcont:
  %swn = add nuw i64 %sweep, 1
  br label %svtop
svconv:
  br label %nrml
nrml:
  %ni = phi i64 [ 0, %svconv ], [ %nin, %nrml ]
  %nicol = mul i64 %ni, %m
  %ncolp = getelementptr inbounds nuw double, ptr %base, i64 %nicol
  %sig2 = call double @eig_dotc(ptr %ncolp, ptr %ncolp, i64 %m)
  %nrp = getelementptr inbounds nuw double, ptr %nrm, i64 %ni
  store double %sig2, ptr %nrp, align 8
  %nin = add nuw i64 %ni, 1
  %nmore = icmp ult i64 %nin, %n
  br i1 %nmore, label %nrml, label %svsort
svsort:
  call void @eig_argsort(ptr %idx, ptr %nrm, i64 %n, i32 1)
  br label %okl
okl:
  %k = phi i64 [ 0, %svsort ], [ %kn, %kend ]
  %kip = getelementptr inbounds nuw i64, ptr %idx, i64 %k
  %ci = load i64, ptr %kip, align 8
  %cnrp = getelementptr inbounds nuw double, ptr %nrm, i64 %ci
  %csig2 = load double, ptr %cnrp, align 8
  %sig = call double @llvm.sqrt.f64(double %csig2)
  %skp = getelementptr inbounds nuw double, ptr %S, i64 %k
  store double %sig, ptr %skp, align 8
  %pos = fcmp ogt double %sig, 0.0
  %recraw = fdiv double 1.0, %sig
  %rec = select i1 %pos, double %recraw, double 0.0
  %cicol = mul i64 %ci, %m
  br label %uloop
uloop:
  %ur = phi i64 [ 0, %okl ], [ %urn, %uloop ]
  %usrc = add i64 %cicol, %ur
  %usp = getelementptr inbounds nuw double, ptr %base, i64 %usrc
  %wv = load double, ptr %usp, align 8
  %uv = fmul double %wv, %rec
  %udrow = mul i64 %ur, %n
  %udst = add i64 %udrow, %k
  %udp = getelementptr inbounds nuw double, ptr %U, i64 %udst
  store double %uv, ptr %udp, align 8
  %urn = add nuw i64 %ur, 1
  %umore = icmp ult i64 %urn, %m
  br i1 %umore, label %uloop, label %vtstart
vtstart:
  %civec = mul i64 %ci, %n
  %krow = mul i64 %k, %n
  br label %vtloop
vtloop:
  %vr = phi i64 [ 0, %vtstart ], [ %vrn, %vtloop ]
  %vsrc = add i64 %civec, %vr
  %vsp = getelementptr inbounds nuw double, ptr %Vc, i64 %vsrc
  %vv = load double, ptr %vsp, align 8
  %vdst = add i64 %krow, %vr
  %vtp = getelementptr inbounds nuw double, ptr %Vt, i64 %vdst
  store double %vv, ptr %vtp, align 8
  %vrn = add nuw i64 %vr, 1
  %vtmore = icmp ult i64 %vrn, %n
  br i1 %vtmore, label %vtloop, label %kend
kend:
  %kn = add nuw i64 %k, 1
  %kmore = icmp ult i64 %kn, %n
  br i1 %kmore, label %okl, label %freeret
freeret:
  call void @free(ptr %base)
  ret i32 0
svnotconv:
  call void @free(ptr %base)
  ret i32 12
retok0:
  ret i32 0
enull:
  ret i32 1
earg:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

attributes #0 = { nounwind willreturn norecurse }
attributes #1 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(none) }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #5 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
