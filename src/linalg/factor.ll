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

; universe_linalg_* — DIRECT (non-iterative) dense linear algebra: LU with
; partial pivoting, triangular/general solve, matrix inverse, determinant,
; Cholesky, Householder QR, and least squares. Row-major, contiguous f64;
; element (i,j) of an m x n matrix lives at A[i*n + j] (leading dim = n).
; Method names follow Julia's LinearAlgebra stdlib (functional inspiration
; only — the layout, kernels, and IR are designed here from first principles).
; Builds on the src/linalg/matrix.ll BLAS core (same domain, same ABI); the
; hot inner axpy/dot are re-inlined here (per the hot-leaf duplication rule)
; so the elimination and Householder loops codegen to packed fma without a
; cross-module call.
;
; Errors i32: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 8 INVALID_ARG
; (shape / dimension), 11 SINGULAR (zero pivot / zero triangular diagonal),
; 13 NOT_SPD (Cholesky on a non-positive-definite matrix). Value-returning
; det yields 0.0 on any degenerate/singular input.
;
; DESIGN
; ------
; ALGORITHM CLASS — direct factorizations, the correct class for exact dense
; solves (an iterative method would be wrong here): O(n^3) elimination with a
; SIMD inner kernel is both fastest and most accurate for well-conditioned
; dense systems.
;   * LU: Doolittle elimination with PARTIAL PIVOTING (largest-magnitude pivot
;     per column → numerically stable, and the zero-pivot test is the singular
;     detector). Factor is IN PLACE on a caller/scratch copy: the strict lower
;     triangle holds L (unit diagonal implied), the upper triangle holds U, a
;     separate i32 pivot vector records the row swaps, and a sign accumulates
;     the permutation parity for det. The elimination inner loop is a contiguous
;     axpy  row_i[k+1..n] -= factor * row_k[k+1..n]  — SIMD-first (li_axpy).
;   * solve / inv: LU once, then per right-hand-side a permute + unit-lower
;     forward substitution + upper back substitution; each substitution's
;     dot(row_prefix, x_prefix) is the contiguous SIMD reduction (li_dot2).
;     inv = solve against the columns of I.
;   * det: product of U's diagonal times the pivot sign (0 if LU hit a zero
;     pivot).
;   * cholesky: Cholesky-Banachiewicz  L[i,j] = (A[i,j] - dot(L_i,L_j)) / L[j,j]
;     with the diagonal  L[i,i] = sqrt(A[i,i] - dot(L_i,L_i)); a non-positive
;     radicand proves the matrix is not SPD → 13. The prefix dot is li_dot2.
;   * qr: HOUSEHOLDER reflectors. The reflector H = I - beta*v*v^T is applied
;     to R and to Q via the RANK-1 ROW form (not the strided column form):
;     w = v^T * subblock (an accumulate axpy over CONTIGUOUS rows), then
;     subblock -= beta * v * w (another contiguous axpy). Row-major rows are
;     contiguous, so both phases are SIMD-first (li_axpy) — the only strided
;     touch is the one-shot column read that builds v (a cheap scalar norm).
;     Q is accumulated by replaying the stored reflectors onto I in reverse.
;   * lstsq: QR of the m x n (m>=n) system, apply Q^T to b by replaying the
;     reflectors on the rhs vector (contiguous), then upper back-substitute the
;     leading n x n block of R. No normal equations (A^T A squares the
;     condition number) and no explicit Q (cheaper).
;   * tri_solve: a bare triangular forward/back substitution (upper flag picks
;     the direction), diagonal-zero → singular.
;
; SIMD-FIRST — the two hot leaves are li_axpy (dst += coef*src, <2 x double>
; body + scalar tail) and li_dot2 (2-accumulator <2 x double> reduction +
; scalar tail). Both are alwaysinline so the elimination / substitution /
; Householder loops fold them in and lower to packed fma (fmla.2d / vfmadd*pd);
; the scalar tail is the sub-vector remainder AND the reference the tests
; cross-check the reconstruction against. Reflectors are applied by CONTIGUOUS
; row operations precisely so the vector path fires.
;
; ALLOCATION — one malloc per call for ALL scratch (an arena, not scattered
; allocs): a single block is carved into LU / column / pivot (solve, inv, det)
; or R / reflector-store / beta / work / column (qr, lstsq). Every size is
; overflow-checked (umul/uadd.with.overflow). tri_solve and cholesky need no
; scratch. Caller owns the input and output buffers.
;
; API (f64, row-major; n x n unless noted):
;   i32 universe_linalg_lu(A, n, LU_out, piv_out /*i32[n]*/, sign_out /*f64*/)
;   i32 universe_linalg_solve(A, n, b, nrhs, x_out)      ; A X = B (b,x n x nrhs)
;   i32 universe_linalg_inv(A, n, out)                   ; out = A^-1
;   double universe_linalg_det(A, n)                     ; det(A) (0 if singular)
;   i32 universe_linalg_cholesky(A, n, L_out)            ; SPD -> lower L (13 else)
;   i32 universe_linalg_qr(A, m, n, Q_out, R_out)        ; Q m x m, R m x n
;   i32 universe_linalg_tri_solve(T, n, b, upper, x_out) ; T x = b (upper!=0=>U)
;   i32 universe_linalg_lstsq(A, m, n, b, x_out)         ; min||Ax-b||, m>=n

declare double @llvm.fma.f64(double, double, double)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare double @llvm.sqrt.f64(double)
declare double @llvm.fabs.f64(double)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare ptr @malloc(i64)
declare void @free(ptr)

; ==================================================================== helpers

; li_dot2 — 2-accumulator <2 x double> reduction of sum a[i]*b[i], scalar tail.
; alwaysinline: folds into the substitution / Cholesky / reflector loops.
define internal double @li_dot2(ptr readonly %a, ptr readonly %b, i64 %n) #1 {
entry:
  %has4 = icmp uge i64 %n, 4
  br i1 %has4, label %vmain, label %red
vmain:
  %i = phi i64 [ 0, %entry ], [ %inext, %vmain ]
  %acc0 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc0n, %vmain ]
  %acc1 = phi <2 x double> [ zeroinitializer, %entry ], [ %acc1n, %vmain ]
  %pa0 = getelementptr inbounds double, ptr %a, i64 %i
  %va0 = load <2 x double>, ptr %pa0, align 8
  %pb0 = getelementptr inbounds double, ptr %b, i64 %i
  %vb0 = load <2 x double>, ptr %pb0, align 8
  %acc0n = call fast <2 x double> @llvm.fma.v2f64(<2 x double> %va0, <2 x double> %vb0, <2 x double> %acc0)
  %i2 = add nuw i64 %i, 2
  %pa1 = getelementptr inbounds double, ptr %a, i64 %i2
  %va1 = load <2 x double>, ptr %pa1, align 8
  %pb1 = getelementptr inbounds double, ptr %b, i64 %i2
  %vb1 = load <2 x double>, ptr %pb1, align 8
  %acc1n = call fast <2 x double> @llvm.fma.v2f64(<2 x double> %va1, <2 x double> %vb1, <2 x double> %acc1)
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %n, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vmain, label %vdone
vdone:
  %sum = fadd fast <2 x double> %acc0n, %acc1n
  %hs = call fast double @llvm.vector.reduce.fadd.v2f64(double -0.0, <2 x double> %sum)
  br label %red
red:
  %base = phi double [ 0.0, %entry ], [ %hs, %vdone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %vdone ]
  br label %tail
tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %body ]
  %sacc = phi double [ %base, %red ], [ %saccn, %body ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %body
body:
  %ta = getelementptr inbounds double, ptr %a, i64 %j
  %fa = load double, ptr %ta, align 8
  %tb = getelementptr inbounds double, ptr %b, i64 %j
  %fb = load double, ptr %tb, align 8
  %saccn = call fast double @llvm.fma.f64(double %fa, double %fb, double %sacc)
  %jnext = add nuw i64 %j, 1
  br label %tail
ret:
  ret double %sacc
}

; li_axpy — dst[j] += coef*src[j], j in 0..count. <2 x double> body + scalar
; tail. Plain (non-fast) fma: keeps the elimination/Householder update accurate
; and well-ordered vs the reference. alwaysinline hot leaf.
define internal void @li_axpy(ptr %dst, ptr readonly %src, double %coef, i64 %count) #2 {
entry:
  %vc0 = insertelement <2 x double> poison, double %coef, i64 0
  %vc = shufflevector <2 x double> %vc0, <2 x double> poison, <2 x i32> zeroinitializer
  %even = and i64 %count, -2
  %hasv = icmp ne i64 %even, 0
  br i1 %hasv, label %vmain, label %tail
vmain:
  %i = phi i64 [ 0, %entry ], [ %inext, %vmain ]
  %pd = getelementptr inbounds double, ptr %dst, i64 %i
  %vd = load <2 x double>, ptr %pd, align 8
  %ps = getelementptr inbounds double, ptr %src, i64 %i
  %vs = load <2 x double>, ptr %ps, align 8
  %r = call <2 x double> @llvm.fma.v2f64(<2 x double> %vc, <2 x double> %vs, <2 x double> %vd)
  store <2 x double> %r, ptr %pd, align 8
  %inext = add nuw i64 %i, 2
  %more = icmp ult i64 %inext, %even
  br i1 %more, label %vmain, label %tail
tail:
  %j = phi i64 [ 0, %entry ], [ %even, %vmain ]
  br label %tloop
tloop:
  %k = phi i64 [ %j, %tail ], [ %knext, %tbody ]
  %done = icmp uge i64 %k, %count
  br i1 %done, label %ret, label %tbody
tbody:
  %pd2 = getelementptr inbounds double, ptr %dst, i64 %k
  %vd2 = load double, ptr %pd2, align 8
  %ps2 = getelementptr inbounds double, ptr %src, i64 %k
  %vs2 = load double, ptr %ps2, align 8
  %r2 = call double @llvm.fma.f64(double %coef, double %vs2, double %vd2)
  store double %r2, ptr %pd2, align 8
  %knext = add nuw i64 %k, 1
  br label %tloop
ret:
  ret void
}

; li_lu_factor — in-place LU with partial pivoting on %lu (n x n). Writes L in
; the strict lower triangle (unit diag implied), U in the upper triangle, the
; row-swap indices to piv (i32[n]), and the permutation sign (+/-1.0) to sgn.
; Returns 0 OK, 11 SINGULAR on a zero pivot.
define internal i32 @li_lu_factor(ptr %lu, i64 %n, ptr %piv, ptr %sgn) #3 {
entry:
  store double 1.0, ptr %sgn, align 8
  br label %kloop
kloop:
  %k = phi i64 [ 0, %entry ], [ %kn, %kdone ]
  %krow = mul i64 %k, %n
  %kkidx = add i64 %krow, %k
  %pkk = getelementptr inbounds double, ptr %lu, i64 %kkidx
  %dkk = load double, ptr %pkk, align 8
  %akk = call double @llvm.fabs.f64(double %dkk)
  %kp1 = add nuw i64 %k, 1
  br label %psearch
psearch:
  %i = phi i64 [ %kp1, %kloop ], [ %in, %pcont ]
  %mr = phi i64 [ %k, %kloop ], [ %mrn, %pcont ]
  %mv = phi double [ %akk, %kloop ], [ %mvn, %pcont ]
  %pdone = icmp uge i64 %i, %n
  br i1 %pdone, label %pivsel, label %pbody
pbody:
  %irow = mul i64 %i, %n
  %iidx = add i64 %irow, %k
  %pik = getelementptr inbounds double, ptr %lu, i64 %iidx
  %dik = load double, ptr %pik, align 8
  %aik = call double @llvm.fabs.f64(double %dik)
  %gt = fcmp ogt double %aik, %mv
  %mrn = select i1 %gt, i64 %i, i64 %mr
  %mvn = select i1 %gt, double %aik, double %mv
  br label %pcont
pcont:
  %in = add nuw i64 %i, 1
  br label %psearch
pivsel:
  %sing = fcmp oeq double %mv, 0.0
  br i1 %sing, label %singular, label %pivok
pivok:
  %needswap = icmp ne i64 %mr, %k
  br i1 %needswap, label %doswap, label %afterswap
doswap:
  %mrow = mul i64 %mr, %n
  br label %swaploop
swaploop:
  %c = phi i64 [ 0, %doswap ], [ %cn, %swaploop ]
  %ka = add i64 %krow, %c
  %ma = add i64 %mrow, %c
  %pka = getelementptr inbounds double, ptr %lu, i64 %ka
  %pma = getelementptr inbounds double, ptr %lu, i64 %ma
  %t1 = load double, ptr %pka, align 8
  %t2 = load double, ptr %pma, align 8
  store double %t2, ptr %pka, align 8
  store double %t1, ptr %pma, align 8
  %cn = add nuw i64 %c, 1
  %cmore = icmp ult i64 %cn, %n
  br i1 %cmore, label %swaploop, label %swapdone
swapdone:
  %sv = load double, ptr %sgn, align 8
  %sneg = fneg double %sv
  store double %sneg, ptr %sgn, align 8
  br label %afterswap
afterswap:
  %ppk = getelementptr inbounds i32, ptr %piv, i64 %k
  %mr32 = trunc i64 %mr to i32
  store i32 %mr32, ptr %ppk, align 4
  %kkidx2 = add i64 %krow, %k
  %pkk2 = getelementptr inbounds double, ptr %lu, i64 %kkidx2
  %pivv = load double, ptr %pkk2, align 8
  %cnt = sub i64 %n, %kp1
  %ksrc = add i64 %krow, %kp1
  %psrc = getelementptr inbounds double, ptr %lu, i64 %ksrc
  br label %eloop
eloop:
  %ei = phi i64 [ %kp1, %afterswap ], [ %ein, %eloopc ]
  %edone = icmp uge i64 %ei, %n
  br i1 %edone, label %kdone, label %ebody
ebody:
  %eirow = mul i64 %ei, %n
  %eik = add i64 %eirow, %k
  %peik = getelementptr inbounds double, ptr %lu, i64 %eik
  %dval = load double, ptr %peik, align 8
  %factor = fdiv double %dval, %pivv
  store double %factor, ptr %peik, align 8
  %negf = fneg double %factor
  %edst = add i64 %eirow, %kp1
  %pedst = getelementptr inbounds double, ptr %lu, i64 %edst
  call void @li_axpy(ptr %pedst, ptr %psrc, double %negf, i64 %cnt)
  br label %eloopc
eloopc:
  %ein = add nuw i64 %ei, 1
  br label %eloop
kdone:
  %kn = add nuw i64 %k, 1
  %kmore = icmp ult i64 %kn, %n
  br i1 %kmore, label %kloop, label %ok
ok:
  ret i32 0
singular:
  ret i32 11
}

; li_lusolve — solve L U x = P b in place on %x (contiguous length n): apply the
; row permutation, unit-lower forward substitution, then upper back
; substitution. %lu is a completed li_lu_factor result.
define internal void @li_lusolve(ptr %lu, i64 %n, ptr %piv, ptr %x) #3 {
entry:
  br label %permloop
permloop:
  %k = phi i64 [ 0, %entry ], [ %kn, %permloop ]
  %ppk = getelementptr inbounds i32, ptr %piv, i64 %k
  %pk = load i32, ptr %ppk, align 4
  %pki = zext i32 %pk to i64
  %pxk = getelementptr inbounds double, ptr %x, i64 %k
  %pxp = getelementptr inbounds double, ptr %x, i64 %pki
  %vk = load double, ptr %pxk, align 8
  %vp = load double, ptr %pxp, align 8
  store double %vp, ptr %pxk, align 8
  store double %vk, ptr %pxp, align 8
  %kn = add nuw i64 %k, 1
  %pmore = icmp ult i64 %kn, %n
  br i1 %pmore, label %permloop, label %fwd
fwd:
  br label %fwdloop
fwdloop:
  %fi = phi i64 [ 0, %fwd ], [ %fin, %fwdloop ]
  %frow = mul i64 %fi, %n
  %prow = getelementptr inbounds double, ptr %lu, i64 %frow
  %fd = call double @li_dot2(ptr %prow, ptr %x, i64 %fi)
  %pfx = getelementptr inbounds double, ptr %x, i64 %fi
  %fxi = load double, ptr %pfx, align 8
  %fs = fsub double %fxi, %fd
  store double %fs, ptr %pfx, align 8
  %fin = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fin, %n
  br i1 %fmore, label %fwdloop, label %back
back:
  %nm1 = sub i64 %n, 1
  br label %backloop
backloop:
  %bi = phi i64 [ %nm1, %back ], [ %bin, %backcont ]
  %brow = mul i64 %bi, %n
  %bip1 = add nuw i64 %bi, 1
  %bseg = add i64 %brow, %bip1
  %pbseg = getelementptr inbounds double, ptr %lu, i64 %bseg
  %pxseg = getelementptr inbounds double, ptr %x, i64 %bip1
  %bcnt = sub i64 %n, %bip1
  %bd = call double @li_dot2(ptr %pbseg, ptr %pxseg, i64 %bcnt)
  %pbx = getelementptr inbounds double, ptr %x, i64 %bi
  %bxi = load double, ptr %pbx, align 8
  %bnum = fsub double %bxi, %bd
  %bdiagi = add i64 %brow, %bi
  %pbdiag = getelementptr inbounds double, ptr %lu, i64 %bdiagi
  %bdg = load double, ptr %pbdiag, align 8
  %br = fdiv double %bnum, %bdg
  store double %br, ptr %pbx, align 8
  %biz = icmp eq i64 %bi, 0
  br i1 %biz, label %done, label %backcont
backcont:
  %bin = sub i64 %bi, 1
  br label %backloop
done:
  ret void
}

; li_qr_factor — in-place Householder QR on %R (m x n). Stores reflector k's
; vector v (length m-k) at %V + k*m and its beta at %bta[k]; leaves R with the
; upper factor in its upper trapezoid (the strict-lower entries hold reflector
; residue — callers that need a clean R zero them). %w is a length-m scratch.
define internal i32 @li_qr_factor(ptr %R, i64 %m, i64 %n, ptr %V, ptr %bta, ptr %w) #3 {
entry:
  %mlt = icmp ult i64 %m, %n
  %t = select i1 %mlt, i64 %m, i64 %n
  br label %kloop
kloop:
  %k = phi i64 [ 0, %entry ], [ %kn, %kend ]
  %kdone = icmp uge i64 %k, %t
  br i1 %kdone, label %ret, label %kbody
kbody:
  %L = sub i64 %m, %k
  %kvm = mul i64 %k, %m
  %Vk = getelementptr inbounds double, ptr %V, i64 %kvm
  br label %colloop
colloop:
  %ci = phi i64 [ 0, %kbody ], [ %cin, %colloop ]
  %s2 = phi double [ 0.0, %kbody ], [ %s2n, %colloop ]
  %rr = add i64 %k, %ci
  %rrow = mul i64 %rr, %n
  %ridx = add i64 %rrow, %k
  %pxi = getelementptr inbounds double, ptr %R, i64 %ridx
  %xi = load double, ptr %pxi, align 8
  %pvi = getelementptr inbounds double, ptr %Vk, i64 %ci
  store double %xi, ptr %pvi, align 8
  %sq = fmul double %xi, %xi
  %s2n = fadd double %s2, %sq
  %cin = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %cin, %L
  br i1 %cmore, label %colloop, label %coldone
coldone:
  %x0 = load double, ptr %Vk, align 8
  %sigma = call double @llvm.sqrt.f64(double %s2n)
  %iszero = fcmp oeq double %sigma, 0.0
  br i1 %iszero, label %skip, label %reflect
skip:
  %pbk0 = getelementptr inbounds double, ptr %bta, i64 %k
  store double 0.0, ptr %pbk0, align 8
  br label %kend
reflect:
  %xge = fcmp oge double %x0, 0.0
  %nsig = fneg double %sigma
  %alpha = select i1 %xge, double %nsig, double %sigma
  %v0 = fsub double %x0, %alpha
  store double %v0, ptr %Vk, align 8
  %v0sq = fmul double %v0, %v0
  %x0sq = fmul double %x0, %x0
  %rest = fsub double %s2n, %x0sq
  %vn2 = fadd double %v0sq, %rest
  %beta = fdiv double 2.0, %vn2
  %pbk = getelementptr inbounds double, ptr %bta, i64 %k
  store double %beta, ptr %pbk, align 8
  %kkrow = mul i64 %k, %n
  %kkidx = add i64 %kkrow, %k
  %pkk = getelementptr inbounds double, ptr %R, i64 %kkidx
  store double %alpha, ptr %pkk, align 8
  br label %zloop
zloop:
  %zi = phi i64 [ 1, %reflect ], [ %zin, %zcont ]
  %zdone = icmp uge i64 %zi, %L
  br i1 %zdone, label %applypre, label %zbody
zbody:
  %zr = add i64 %k, %zi
  %zrow = mul i64 %zr, %n
  %zidx = add i64 %zrow, %k
  %pz = getelementptr inbounds double, ptr %R, i64 %zidx
  store double 0.0, ptr %pz, align 8
  br label %zcont
zcont:
  %zin = add nuw i64 %zi, 1
  br label %zloop
applypre:
  %c0 = add nuw i64 %k, 1
  %ncols = sub i64 %n, %c0
  %hascol = icmp ugt i64 %ncols, 0
  br i1 %hascol, label %wzero, label %kend
wzero:
  %wbytes = shl i64 %ncols, 3
  call void @llvm.memset.p0.i64(ptr %w, i8 0, i64 %wbytes, i1 false)
  br label %waloop
waloop:
  %wi = phi i64 [ 0, %wzero ], [ %win, %waloop ]
  %wvp = getelementptr inbounds double, ptr %Vk, i64 %wi
  %wv = load double, ptr %wvp, align 8
  %wrr = add i64 %k, %wi
  %wrow = mul i64 %wrr, %n
  %wridx = add i64 %wrow, %c0
  %prow2 = getelementptr inbounds double, ptr %R, i64 %wridx
  call void @li_axpy(ptr %w, ptr %prow2, double %wv, i64 %ncols)
  %win = add nuw i64 %wi, 1
  %wmore = icmp ult i64 %win, %L
  br i1 %wmore, label %waloop, label %uploop
uploop:
  %ui = phi i64 [ 0, %waloop ], [ %uin, %uploop ]
  %uvp = getelementptr inbounds double, ptr %Vk, i64 %ui
  %uv = load double, ptr %uvp, align 8
  %bv = fmul double %beta, %uv
  %ncoef = fneg double %bv
  %urr = add i64 %k, %ui
  %urow = mul i64 %urr, %n
  %uridx = add i64 %urow, %c0
  %uprow = getelementptr inbounds double, ptr %R, i64 %uridx
  call void @li_axpy(ptr %uprow, ptr %w, double %ncoef, i64 %ncols)
  %uin = add nuw i64 %ui, 1
  %umore = icmp ult i64 %uin, %L
  br i1 %umore, label %uploop, label %kend
kend:
  %kn = add nuw i64 %k, 1
  br label %kloop
ret:
  ret i32 0
}

; ==================================================================== exports

define i32 @universe_linalg_lu(ptr %A, i64 %n, ptr %LU, ptr %piv, ptr %sign) #0 {
entry:
  %an = icmp eq ptr %A, null
  %lun = icmp eq ptr %LU, null
  %pn = icmp eq ptr %piv, null
  %sn = icmp eq ptr %sign, null
  %o0 = or i1 %an, %lun
  %o1 = or i1 %pn, %sn
  %anynull = or i1 %o0, %o1
  br i1 %anynull, label %enull, label %chkn
chkn:
  %bad = icmp slt i64 %n, 1
  br i1 %bad, label %einval, label %sizes
sizes:
  %m1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %m1, 0
  %ov1 = extractvalue { i64, i1 } %m1, 1
  br i1 %ov1, label %eovf, label %sizes2
sizes2:
  %m2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nn, i64 8)
  %bytes = extractvalue { i64, i1 } %m2, 0
  %ov2 = extractvalue { i64, i1 } %m2, 1
  br i1 %ov2, label %eovf, label %docopy
docopy:
  call void @llvm.memmove.p0.p0.i64(ptr %LU, ptr %A, i64 %bytes, i1 false)
  %rc = call i32 @li_lu_factor(ptr %LU, i64 %n, ptr %piv, ptr %sign)
  ret i32 %rc
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
}

define i32 @universe_linalg_solve(ptr %A, i64 %n, ptr %b, i64 %nrhs, ptr %x) #0 {
entry:
  %sgn = alloca double, align 8
  %an = icmp eq ptr %A, null
  %bn = icmp eq ptr %b, null
  %xn = icmp eq ptr %x, null
  %o0 = or i1 %an, %bn
  %anynull = or i1 %o0, %xn
  br i1 %anynull, label %enull, label %chkn
chkn:
  %badn = icmp slt i64 %n, 1
  %badr = icmp slt i64 %nrhs, 1
  %bad = or i1 %badn, %badr
  br i1 %bad, label %einval, label %sizes
sizes:
  %m1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %m1, 0
  %ov1 = extractvalue { i64, i1 } %m1, 1
  br i1 %ov1, label %eovf, label %sizes2
sizes2:
  %a1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %nn, i64 %n)
  %t1 = extractvalue { i64, i1 } %a1, 0
  %ov2 = extractvalue { i64, i1 } %a1, 1
  br i1 %ov2, label %eovf, label %sizes3
sizes3:
  %m3 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %t1, i64 8)
  %dbytes = extractvalue { i64, i1 } %m3, 0
  %ov3 = extractvalue { i64, i1 } %m3, 1
  br i1 %ov3, label %eovf, label %sizes4
sizes4:
  %pbytes = shl i64 %n, 2
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %dbytes, i64 %pbytes)
  %total = extractvalue { i64, i1 } %a2, 0
  %ov4 = extractvalue { i64, i1 } %a2, 1
  br i1 %ov4, label %eovf, label %alloc
alloc:
  %base = call ptr @malloc(i64 %total)
  %isnull = icmp eq ptr %base, null
  br i1 %isnull, label %eoom, label %setup
setup:
  %lubytes = shl i64 %nn, 3
  %colbuf = getelementptr inbounds i8, ptr %base, i64 %lubytes
  %pivoff = shl i64 %t1, 3
  %pivp = getelementptr inbounds i8, ptr %base, i64 %pivoff
  call void @llvm.memmove.p0.p0.i64(ptr %base, ptr %A, i64 %lubytes, i1 false)
  %rc = call i32 @li_lu_factor(ptr %base, i64 %n, ptr %pivp, ptr %sgn)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %rhsloop, label %failfree
failfree:
  call void @free(ptr %base)
  ret i32 %rc
rhsloop:
  %c = phi i64 [ 0, %setup ], [ %cnx, %rhsnext ]
  br label %gather
gather:
  %gi = phi i64 [ 0, %rhsloop ], [ %gin, %gather ]
  %grow = mul i64 %gi, %nrhs
  %gidx = add i64 %grow, %c
  %pgb = getelementptr inbounds double, ptr %b, i64 %gidx
  %gv = load double, ptr %pgb, align 8
  %pgc = getelementptr inbounds double, ptr %colbuf, i64 %gi
  store double %gv, ptr %pgc, align 8
  %gin = add nuw i64 %gi, 1
  %gmore = icmp ult i64 %gin, %n
  br i1 %gmore, label %gather, label %dosolve
dosolve:
  call void @li_lusolve(ptr %base, i64 %n, ptr %pivp, ptr %colbuf)
  br label %scatter
scatter:
  %si = phi i64 [ 0, %dosolve ], [ %sin, %scatter ]
  %psc = getelementptr inbounds double, ptr %colbuf, i64 %si
  %sv = load double, ptr %psc, align 8
  %srow = mul i64 %si, %nrhs
  %sidx = add i64 %srow, %c
  %psx = getelementptr inbounds double, ptr %x, i64 %sidx
  store double %sv, ptr %psx, align 8
  %sin = add nuw i64 %si, 1
  %smore = icmp ult i64 %sin, %n
  br i1 %smore, label %scatter, label %rhsnext
rhsnext:
  %cnx = add nuw i64 %c, 1
  %cmore = icmp ult i64 %cnx, %nrhs
  br i1 %cmore, label %rhsloop, label %okfree
okfree:
  call void @free(ptr %base)
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

define i32 @universe_linalg_inv(ptr %A, i64 %n, ptr %out) #0 {
entry:
  %sgn = alloca double, align 8
  %an = icmp eq ptr %A, null
  %on = icmp eq ptr %out, null
  %anynull = or i1 %an, %on
  br i1 %anynull, label %enull, label %chkn
chkn:
  %bad = icmp slt i64 %n, 1
  br i1 %bad, label %einval, label %sizes
sizes:
  %m1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %m1, 0
  %ov1 = extractvalue { i64, i1 } %m1, 1
  br i1 %ov1, label %eovf, label %sizes2
sizes2:
  %a1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %nn, i64 %n)
  %t1 = extractvalue { i64, i1 } %a1, 0
  %ov2 = extractvalue { i64, i1 } %a1, 1
  br i1 %ov2, label %eovf, label %sizes3
sizes3:
  %m3 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %t1, i64 8)
  %dbytes = extractvalue { i64, i1 } %m3, 0
  %ov3 = extractvalue { i64, i1 } %m3, 1
  br i1 %ov3, label %eovf, label %sizes4
sizes4:
  %pbytes = shl i64 %n, 2
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %dbytes, i64 %pbytes)
  %total = extractvalue { i64, i1 } %a2, 0
  %ov4 = extractvalue { i64, i1 } %a2, 1
  br i1 %ov4, label %eovf, label %alloc
alloc:
  %base = call ptr @malloc(i64 %total)
  %isnull = icmp eq ptr %base, null
  br i1 %isnull, label %eoom, label %setup
setup:
  %lubytes = shl i64 %nn, 3
  %colbuf = getelementptr inbounds i8, ptr %base, i64 %lubytes
  %pivoff = shl i64 %t1, 3
  %pivp = getelementptr inbounds i8, ptr %base, i64 %pivoff
  call void @llvm.memmove.p0.p0.i64(ptr %base, ptr %A, i64 %lubytes, i1 false)
  %rc = call i32 @li_lu_factor(ptr %base, i64 %n, ptr %pivp, ptr %sgn)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %colloop, label %failfree
failfree:
  call void @free(ptr %base)
  ret i32 %rc
colloop:
  %c = phi i64 [ 0, %setup ], [ %cnx, %colnext ]
  br label %ecol
ecol:
  %ei = phi i64 [ 0, %colloop ], [ %ein, %ecol ]
  %iseq = icmp eq i64 %ei, %c
  %ev = select i1 %iseq, double 1.0, double 0.0
  %pec = getelementptr inbounds double, ptr %colbuf, i64 %ei
  store double %ev, ptr %pec, align 8
  %ein = add nuw i64 %ei, 1
  %emore = icmp ult i64 %ein, %n
  br i1 %emore, label %ecol, label %dosolve
dosolve:
  call void @li_lusolve(ptr %base, i64 %n, ptr %pivp, ptr %colbuf)
  br label %scatter
scatter:
  %si = phi i64 [ 0, %dosolve ], [ %sin, %scatter ]
  %psc = getelementptr inbounds double, ptr %colbuf, i64 %si
  %sv = load double, ptr %psc, align 8
  %srow = mul i64 %si, %n
  %sidx = add i64 %srow, %c
  %pso = getelementptr inbounds double, ptr %out, i64 %sidx
  store double %sv, ptr %pso, align 8
  %sin = add nuw i64 %si, 1
  %smore = icmp ult i64 %sin, %n
  br i1 %smore, label %scatter, label %colnext
colnext:
  %cnx = add nuw i64 %c, 1
  %cmore = icmp ult i64 %cnx, %n
  br i1 %cmore, label %colloop, label %okfree
okfree:
  call void @free(ptr %base)
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

define double @universe_linalg_det(ptr %A, i64 %n) #0 {
entry:
  %sgn = alloca double, align 8
  %an = icmp eq ptr %A, null
  br i1 %an, label %zero, label %chkn
chkn:
  %bad = icmp slt i64 %n, 1
  br i1 %bad, label %zero, label %sizes
sizes:
  %m1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %n)
  %nn = extractvalue { i64, i1 } %m1, 0
  %ov1 = extractvalue { i64, i1 } %m1, 1
  br i1 %ov1, label %zero, label %sizes2
sizes2:
  %m2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nn, i64 8)
  %dbytes = extractvalue { i64, i1 } %m2, 0
  %ov2 = extractvalue { i64, i1 } %m2, 1
  br i1 %ov2, label %zero, label %sizes3
sizes3:
  %pbytes = shl i64 %n, 2
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %dbytes, i64 %pbytes)
  %total = extractvalue { i64, i1 } %a2, 0
  %ov3 = extractvalue { i64, i1 } %a2, 1
  br i1 %ov3, label %zero, label %alloc
alloc:
  %base = call ptr @malloc(i64 %total)
  %isnull = icmp eq ptr %base, null
  br i1 %isnull, label %zero, label %setup
setup:
  %pivp = getelementptr inbounds i8, ptr %base, i64 %dbytes
  call void @llvm.memmove.p0.p0.i64(ptr %base, ptr %A, i64 %dbytes, i1 false)
  %rc = call i32 @li_lu_factor(ptr %base, i64 %n, ptr %pivp, ptr %sgn)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %prod, label %singfree
singfree:
  call void @free(ptr %base)
  ret double 0.0
prod:
  %s0 = load double, ptr %sgn, align 8
  br label %ploop
ploop:
  %pi = phi i64 [ 0, %prod ], [ %pin, %ploop ]
  %acc = phi double [ %s0, %prod ], [ %accn, %ploop ]
  %prow = mul i64 %pi, %n
  %pidx = add i64 %prow, %pi
  %pp = getelementptr inbounds double, ptr %base, i64 %pidx
  %dv = load double, ptr %pp, align 8
  %accn = fmul double %acc, %dv
  %pin = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pin, %n
  br i1 %pmore, label %ploop, label %pdone
pdone:
  call void @free(ptr %base)
  ret double %accn
zero:
  ret double 0.0
}

define i32 @universe_linalg_cholesky(ptr %A, i64 %n, ptr %L) #0 {
entry:
  %an = icmp eq ptr %A, null
  %ln = icmp eq ptr %L, null
  %anynull = or i1 %an, %ln
  br i1 %anynull, label %enull, label %chkn
chkn:
  %bad = icmp slt i64 %n, 1
  br i1 %bad, label %einval, label %iloop
iloop:
  %i = phi i64 [ 0, %chkn ], [ %inx, %zdone ]
  %irow = mul i64 %i, %n
  %pLi = getelementptr inbounds double, ptr %L, i64 %irow
  br label %jloop
jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnx, %jcont ]
  %jrow = mul i64 %j, %n
  %pLj = getelementptr inbounds double, ptr %L, i64 %jrow
  %d = call double @li_dot2(ptr %pLi, ptr %pLj, i64 %j)
  %aidx = add i64 %irow, %j
  %pA = getelementptr inbounds double, ptr %A, i64 %aidx
  %aij = load double, ptr %pA, align 8
  %s = fsub double %aij, %d
  %isdiag = icmp eq i64 %j, %i
  br i1 %isdiag, label %diag, label %offdiag
offdiag:
  %ljjidx = add i64 %jrow, %j
  %pLjj = getelementptr inbounds double, ptr %L, i64 %ljjidx
  %ljj = load double, ptr %pLjj, align 8
  %lij = fdiv double %s, %ljj
  %oidx = add i64 %irow, %j
  %pOut = getelementptr inbounds double, ptr %L, i64 %oidx
  store double %lij, ptr %pOut, align 8
  br label %jcont
diag:
  %notpd = fcmp ole double %s, 0.0
  br i1 %notpd, label %enotspd, label %diagok
diagok:
  %lii = call double @llvm.sqrt.f64(double %s)
  %didx = add i64 %irow, %i
  %pDiag = getelementptr inbounds double, ptr %L, i64 %didx
  store double %lii, ptr %pDiag, align 8
  br label %zupper
jcont:
  %jnx = add nuw i64 %j, 1
  br label %jloop
zupper:
  %zj = phi i64 [ %i, %diagok ], [ %zj0, %zbody ]
  %zj0 = add nuw i64 %zj, 1
  %zdonec = icmp uge i64 %zj0, %n
  br i1 %zdonec, label %zdone, label %zbody
zbody:
  %zidx = add i64 %irow, %zj0
  %pZ = getelementptr inbounds double, ptr %L, i64 %zidx
  store double 0.0, ptr %pZ, align 8
  br label %zupper
zdone:
  %inx = add nuw i64 %i, 1
  %imore = icmp ult i64 %inx, %n
  br i1 %imore, label %iloop, label %ok
ok:
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
enotspd:
  ret i32 13
}

define i32 @universe_linalg_qr(ptr %A, i64 %m, i64 %n, ptr %Q, ptr %R) #0 {
entry:
  %an = icmp eq ptr %A, null
  %qn = icmp eq ptr %Q, null
  %rn = icmp eq ptr %R, null
  %o0 = or i1 %an, %qn
  %anynull = or i1 %o0, %rn
  br i1 %anynull, label %enull, label %chkd
chkd:
  %badm = icmp slt i64 %m, 1
  %badn = icmp slt i64 %n, 1
  %bad = or i1 %badm, %badn
  br i1 %bad, label %einval, label %sizes
sizes:
  %mm1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %mn = extractvalue { i64, i1 } %mm1, 0
  %ov1 = extractvalue { i64, i1 } %mm1, 1
  br i1 %ov1, label %eovf, label %sizes2
sizes2:
  %a1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %mn, i64 %n)
  %s1 = extractvalue { i64, i1 } %a1, 0
  %ov2 = extractvalue { i64, i1 } %a1, 1
  br i1 %ov2, label %eovf, label %sizes3
sizes3:
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s1, i64 %m)
  %ndbl = extractvalue { i64, i1 } %a2, 0
  %ov3 = extractvalue { i64, i1 } %a2, 1
  br i1 %ov3, label %eovf, label %sizes4
sizes4:
  %m2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ndbl, i64 8)
  %total = extractvalue { i64, i1 } %m2, 0
  %ov4 = extractvalue { i64, i1 } %m2, 1
  br i1 %ov4, label %eovf, label %chkq
chkq:
  %mm2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %m)
  %qelt = extractvalue { i64, i1 } %mm2, 0
  %ov5 = extractvalue { i64, i1 } %mm2, 1
  br i1 %ov5, label %eovf, label %alloc
alloc:
  %base = call ptr @malloc(i64 %total)
  %isnull = icmp eq ptr %base, null
  br i1 %isnull, label %eoom, label %setup
setup:
  %betaoff = shl i64 %mn, 3
  %betap = getelementptr inbounds i8, ptr %base, i64 %betaoff
  %woff = shl i64 %s1, 3
  %wp = getelementptr inbounds i8, ptr %base, i64 %woff
  %rbytes = shl i64 %mn, 3
  call void @llvm.memmove.p0.p0.i64(ptr %R, ptr %A, i64 %rbytes, i1 false)
  %qbytes = shl i64 %qelt, 3
  call void @llvm.memset.p0.i64(ptr %Q, i8 0, i64 %qbytes, i1 false)
  br label %idloop
idloop:
  %di = phi i64 [ 0, %setup ], [ %din, %idloop ]
  %didx = mul i64 %di, %m
  %ddidx = add i64 %didx, %di
  %pqd = getelementptr inbounds double, ptr %Q, i64 %ddidx
  store double 1.0, ptr %pqd, align 8
  %din = add nuw i64 %di, 1
  %dmore = icmp ult i64 %din, %m
  br i1 %dmore, label %idloop, label %dofactor
dofactor:
  %fr = call i32 @li_qr_factor(ptr %R, i64 %m, i64 %n, ptr %base, ptr %betap, ptr %wp)
  %mlt = icmp ult i64 %m, %n
  %t = select i1 %mlt, i64 %m, i64 %n
  %tm1 = sub i64 %t, 1
  br label %rloop
rloop:
  %k = phi i64 [ %tm1, %dofactor ], [ %kdn, %rnext ]
  %L = sub i64 %m, %k
  %kvm = mul i64 %k, %m
  %Vk = getelementptr inbounds double, ptr %base, i64 %kvm
  %pbk = getelementptr inbounds double, ptr %betap, i64 %k
  %beta = load double, ptr %pbk, align 8
  %wbytes = shl i64 %m, 3
  call void @llvm.memset.p0.i64(ptr %wp, i8 0, i64 %wbytes, i1 false)
  br label %waloop
waloop:
  %wi = phi i64 [ 0, %rloop ], [ %win, %waloop ]
  %pwv = getelementptr inbounds double, ptr %Vk, i64 %wi
  %wv = load double, ptr %pwv, align 8
  %wrr = add i64 %k, %wi
  %wrow = mul i64 %wrr, %m
  %pqrow = getelementptr inbounds double, ptr %Q, i64 %wrow
  call void @li_axpy(ptr %wp, ptr %pqrow, double %wv, i64 %m)
  %win = add nuw i64 %wi, 1
  %wmore = icmp ult i64 %win, %L
  br i1 %wmore, label %waloop, label %uploop
uploop:
  %ui = phi i64 [ 0, %waloop ], [ %uin, %uploop ]
  %puv = getelementptr inbounds double, ptr %Vk, i64 %ui
  %uv = load double, ptr %puv, align 8
  %bv = fmul double %beta, %uv
  %ncoef = fneg double %bv
  %urr = add i64 %k, %ui
  %urow = mul i64 %urr, %m
  %puqrow = getelementptr inbounds double, ptr %Q, i64 %urow
  call void @li_axpy(ptr %puqrow, ptr %wp, double %ncoef, i64 %m)
  %uin = add nuw i64 %ui, 1
  %umore = icmp ult i64 %uin, %L
  br i1 %umore, label %uploop, label %rnext
rnext:
  %kdn = sub i64 %k, 1
  %kz = icmp eq i64 %k, 0
  br i1 %kz, label %zerolow, label %rloop
zerolow:
  br label %zrow
zrow:
  %zi = phi i64 [ 0, %zerolow ], [ %zin, %zrowend ]
  %zirow = mul i64 %zi, %n
  %jlim0 = icmp ult i64 %zi, %n
  %jlim = select i1 %jlim0, i64 %zi, i64 %n
  %hasj = icmp ugt i64 %jlim, 0
  br i1 %hasj, label %zcol, label %zrowend
zcol:
  %zj = phi i64 [ 0, %zrow ], [ %zjn, %zcol ]
  %zidx = add i64 %zirow, %zj
  %pzr = getelementptr inbounds double, ptr %R, i64 %zidx
  store double 0.0, ptr %pzr, align 8
  %zjn = add nuw i64 %zj, 1
  %zjmore = icmp ult i64 %zjn, %jlim
  br i1 %zjmore, label %zcol, label %zrowend
zrowend:
  %zin = add nuw i64 %zi, 1
  %zimore = icmp ult i64 %zin, %m
  br i1 %zimore, label %zrow, label %okfree
okfree:
  call void @free(ptr %base)
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

define i32 @universe_linalg_tri_solve(ptr %T, i64 %n, ptr %b, i32 %upper, ptr %x) #0 {
entry:
  %tn = icmp eq ptr %T, null
  %bn = icmp eq ptr %b, null
  %xn = icmp eq ptr %x, null
  %o0 = or i1 %tn, %bn
  %anynull = or i1 %o0, %xn
  br i1 %anynull, label %enull, label %chkn
chkn:
  %bad = icmp slt i64 %n, 1
  br i1 %bad, label %einval, label %pick
pick:
  %isup = icmp ne i32 %upper, 0
  br i1 %isup, label %backpre, label %fwdpre
fwdpre:
  br label %fwdloop
fwdloop:
  %fi = phi i64 [ 0, %fwdpre ], [ %fin, %fcont ]
  %frow = mul i64 %fi, %n
  %pTi = getelementptr inbounds double, ptr %T, i64 %frow
  %fd = call double @li_dot2(ptr %pTi, ptr %x, i64 %fi)
  %pfb = getelementptr inbounds double, ptr %b, i64 %fi
  %fbi = load double, ptr %pfb, align 8
  %fnum = fsub double %fbi, %fd
  %fdidx = add i64 %frow, %fi
  %pfd = getelementptr inbounds double, ptr %T, i64 %fdidx
  %fdg = load double, ptr %pfd, align 8
  %fzero = fcmp oeq double %fdg, 0.0
  br i1 %fzero, label %esing, label %fstore
fstore:
  %frr = fdiv double %fnum, %fdg
  %pfx = getelementptr inbounds double, ptr %x, i64 %fi
  store double %frr, ptr %pfx, align 8
  %fin = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fin, %n
  br i1 %fmore, label %fcont, label %ok
fcont:
  br label %fwdloop
backpre:
  %nm1 = sub i64 %n, 1
  br label %backloop
backloop:
  %bi = phi i64 [ %nm1, %backpre ], [ %bdn, %bcont ]
  %brow = mul i64 %bi, %n
  %bip1 = add nuw i64 %bi, 1
  %bseg = add i64 %brow, %bip1
  %pTseg = getelementptr inbounds double, ptr %T, i64 %bseg
  %pxseg = getelementptr inbounds double, ptr %x, i64 %bip1
  %bcnt = sub i64 %n, %bip1
  %bd = call double @li_dot2(ptr %pTseg, ptr %pxseg, i64 %bcnt)
  %pbb = getelementptr inbounds double, ptr %b, i64 %bi
  %bbi = load double, ptr %pbb, align 8
  %bnum = fsub double %bbi, %bd
  %bdidx = add i64 %brow, %bi
  %pbd = getelementptr inbounds double, ptr %T, i64 %bdidx
  %bdg = load double, ptr %pbd, align 8
  %bzero = fcmp oeq double %bdg, 0.0
  br i1 %bzero, label %esing, label %bstore
bstore:
  %brr = fdiv double %bnum, %bdg
  %pbx = getelementptr inbounds double, ptr %x, i64 %bi
  store double %brr, ptr %pbx, align 8
  %biz = icmp eq i64 %bi, 0
  br i1 %biz, label %ok, label %bcont
bcont:
  %bdn = sub i64 %bi, 1
  br label %backloop
ok:
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
esing:
  ret i32 11
}

define i32 @universe_linalg_lstsq(ptr %A, i64 %m, i64 %n, ptr %b, ptr %x) #0 {
entry:
  %an = icmp eq ptr %A, null
  %bn = icmp eq ptr %b, null
  %xn = icmp eq ptr %x, null
  %o0 = or i1 %an, %bn
  %anynull = or i1 %o0, %xn
  br i1 %anynull, label %enull, label %chkd
chkd:
  %badn = icmp slt i64 %n, 1
  %mltn = icmp slt i64 %m, %n
  %bad = or i1 %badn, %mltn
  br i1 %bad, label %einval, label %sizes
sizes:
  %mm1 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %m, i64 %n)
  %mn = extractvalue { i64, i1 } %mm1, 0
  %ov1 = extractvalue { i64, i1 } %mm1, 1
  br i1 %ov1, label %eovf, label %sizes2
sizes2:
  %mn2 = shl i64 %mn, 1
  %a1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %mn2, i64 %n)
  %s1 = extractvalue { i64, i1 } %a1, 0
  %ov2 = extractvalue { i64, i1 } %a1, 1
  br i1 %ov2, label %eovf, label %sizes3
sizes3:
  %m2x = shl i64 %m, 1
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s1, i64 %m2x)
  %ndbl = extractvalue { i64, i1 } %a2, 0
  %ov3 = extractvalue { i64, i1 } %a2, 1
  br i1 %ov3, label %eovf, label %sizes4
sizes4:
  %mb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ndbl, i64 8)
  %total = extractvalue { i64, i1 } %mb, 0
  %ov4 = extractvalue { i64, i1 } %mb, 1
  br i1 %ov4, label %eovf, label %alloc
alloc:
  %base = call ptr @malloc(i64 %total)
  %isnull = icmp eq ptr %base, null
  br i1 %isnull, label %eoom, label %setup
setup:
  %rbytes = shl i64 %mn, 3
  %voff = shl i64 %mn, 3
  %Vp = getelementptr inbounds i8, ptr %base, i64 %voff
  %betaoff = shl i64 %mn2, 3
  %betap = getelementptr inbounds i8, ptr %base, i64 %betaoff
  %cboff = shl i64 %s1, 3
  %cbuf = getelementptr inbounds i8, ptr %base, i64 %cboff
  %s1m = add i64 %s1, %m
  %wboff = shl i64 %s1m, 3
  %wbuf = getelementptr inbounds i8, ptr %base, i64 %wboff
  %mbytes = shl i64 %m, 3
  call void @llvm.memmove.p0.p0.i64(ptr %base, ptr %A, i64 %rbytes, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %cbuf, ptr %b, i64 %mbytes, i1 false)
  %fr = call i32 @li_qr_factor(ptr %base, i64 %m, i64 %n, ptr %Vp, ptr %betap, ptr %wbuf)
  br label %qloop
qloop:
  %k = phi i64 [ 0, %setup ], [ %kn, %qloop ]
  %L = sub i64 %m, %k
  %kvm = mul i64 %k, %m
  %Vk = getelementptr inbounds double, ptr %Vp, i64 %kvm
  %pbk = getelementptr inbounds double, ptr %betap, i64 %k
  %beta = load double, ptr %pbk, align 8
  %pck = getelementptr inbounds double, ptr %cbuf, i64 %k
  %w = call double @li_dot2(ptr %Vk, ptr %pck, i64 %L)
  %bw = fmul double %beta, %w
  %ncoef = fneg double %bw
  call void @li_axpy(ptr %pck, ptr %Vk, double %ncoef, i64 %L)
  %kn = add nuw i64 %k, 1
  %kmore = icmp ult i64 %kn, %n
  br i1 %kmore, label %qloop, label %backpre
backpre:
  %nm1 = sub i64 %n, 1
  br label %backloop
backloop:
  %bi = phi i64 [ %nm1, %backpre ], [ %bdn, %bcont ]
  %brow = mul i64 %bi, %n
  %bip1 = add nuw i64 %bi, 1
  %bsegi = add i64 %brow, %bip1
  %pRseg = getelementptr inbounds double, ptr %base, i64 %bsegi
  %pxseg = getelementptr inbounds double, ptr %x, i64 %bip1
  %bcnt = sub i64 %n, %bip1
  %bd = call double @li_dot2(ptr %pRseg, ptr %pxseg, i64 %bcnt)
  %pcbi = getelementptr inbounds double, ptr %cbuf, i64 %bi
  %cbi = load double, ptr %pcbi, align 8
  %bnum = fsub double %cbi, %bd
  %bdidx = add i64 %brow, %bi
  %pRd = getelementptr inbounds double, ptr %base, i64 %bdidx
  %bdg = load double, ptr %pRd, align 8
  %bzero = fcmp oeq double %bdg, 0.0
  br i1 %bzero, label %singfree, label %bstore
bstore:
  %bxr = fdiv double %bnum, %bdg
  %pbx = getelementptr inbounds double, ptr %x, i64 %bi
  store double %bxr, ptr %pbx, align 8
  %biz = icmp eq i64 %bi, 0
  br i1 %biz, label %okfree, label %bcont
bcont:
  %bdn = sub i64 %bi, 1
  br label %backloop
singfree:
  call void @free(ptr %base)
  ret i32 11
okfree:
  call void @free(ptr %base)
  ret i32 0
enull:
  ret i32 1
einval:
  ret i32 8
eovf:
  ret i32 3
eoom:
  ret i32 2
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind norecurse memory(argmem: read) }
attributes #2 = { alwaysinline nounwind norecurse memory(argmem: readwrite) }
attributes #3 = { nounwind norecurse }
