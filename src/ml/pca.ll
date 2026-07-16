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

; universe_ml_pca_* — Principal Component Analysis over row-major f32 samples.
;
; DESIGN:
;   * ALGORITHM CLASS: covariance power-iteration with Hotelling deflation.
;     For n samples x d features we (1) subtract the per-feature mean, (2) form
;     the d x d sample covariance  Cov = Xc^T Xc / (n-1)  as a sum of rank-1
;     outer products (each is a length-d axpy on a Cov row: unit stride, feeds
;     the SIMD kernel directly), then (3) extract the top-k eigenpairs one at a
;     time: power-iterate v <- normalize(Cov v) to the dominant eigenvector,
;     read the eigenvalue as the Rayleigh quotient  lambda = v^T Cov v, then
;     deflate  Cov <- Cov - lambda v v^T  so the next iteration finds the next
;     component. This is the right class here: PCA needs only the leading k of
;     d directions, and power+deflation costs O(iters*k*d^2) — far cheaper than
;     a full symmetric eigensolve when k << d, and every inner op (gemv, dot,
;     axpy, l2_norm) is a stride-1 kernel call.
;   * KERNEL REUSE: this layer only orchestrates. gemv/dot/axpy/scale/l2_norm
;     are the exported universe_ml_* kernels (SIMD <4 x float> with scalar
;     tail). They stay real cross-module calls (no LTO) — correct per the house
;     rule; the algorithm's arithmetic density lives inside those leaves.
;   * ORTHONORMALITY: every candidate eigenvector is re-orthogonalized (modified
;     Gram-Schmidt) against the already-found components each iteration. In
;     exact arithmetic deflation already yields orthogonal eigenvectors, but for
;     repeated/near-zero eigenvalues the leftover subspace is arbitrary; the
;     explicit re-orthogonalization guarantees the reported basis is orthonormal
;     and pins degenerate directions to a valid orthogonal complement.
;   * INIT: v[j] = (j mod 7) + 1, normalized — deterministic, allocation-free,
;     reproducible, and generically non-orthogonal to any eigenvector.
;   * MEMORY: one scratch malloc for fit holds Xc (n*d), mean (d), Cov (d*d),
;     and the two work vectors v,w (d each); transform uses mean+one row buffer.
;     All size math is overflow-checked; freed on every exit.
;   * FP: mean/cover/reduction accumulation inherits the kernels' `fast` flags
;     (associative sums, tolerance ok). The orchestration scalars here
;     (Rayleigh quotient, normalization reciprocal, deflation scale,
;     convergence delta) use UNFLAGGED fadd/fmul/fdiv: they are not reduction
;     hot loops, and unflagged keeps the eigenvalue/convergence test a stable,
;     deterministic reference. The zero-norm guard is an exact `oeq 0.0` (a
;     unit-normalized vector never blows up; only a true-zero norm must break).
;
; API (f32; X is n*d row-major; components is k*d row-major; n>=2):
;   i32  universe_ml_pca_fit(X, n, d, k,
;                            components /*out k*d*/, eigenvalues /*out k*/,
;                            evr /*out k, nullable — explained variance ratio*/,
;                            iters, tol)
;   i32  universe_ml_pca_transform(X, n, d, components, k, out /*out n*k*/)
;   void universe_ml_pca_explained_variance_ratio(eigenvalues, k, total_var, out)
;     fit returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW, 8 INVALID_ARG.

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64)
declare float @llvm.fabs.f32(float)
declare ptr @malloc(i64)
declare void @free(ptr)

; reused SIMD kernels (exported from src/ml/kernels.ll)
declare float @universe_ml_dot(ptr, ptr, i64)
declare void @universe_ml_axpy(ptr, float, ptr, i64)
declare void @universe_ml_scale(ptr, float, i64)
declare float @universe_ml_l2_norm(ptr, i64)
declare void @universe_ml_gemv(ptr, ptr, ptr, i64, i64)

; ================================================= per-feature mean (via axpy)
; mean[j] = (1/n) * sum_i X[i*d + j].  Callers guarantee n >= 1.
define internal void @pca_colmean(ptr readonly %X, i64 %n, i64 %d, ptr %mean) #0 {
entry:
  %mbytes = shl i64 %d, 2
  call void @llvm.memset.p0.i64(ptr %mean, i8 0, i64 %mbytes, i1 false)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = mul nuw i64 %i, %d
  %row = getelementptr inbounds nuw float, ptr %X, i64 %off
  call void @universe_ml_axpy(ptr %mean, float 1.0, ptr %row, i64 %d)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %nf = uitofp i64 %n to float
  %inv = fdiv float 1.0, %nf
  call void @universe_ml_scale(ptr %mean, float %inv, i64 %d)
  ret void
}

; ============================================================= mean-centering
; Xc[i] = X[i] - mean, row by row.
define internal void @pca_center(ptr readonly %X, i64 %n, i64 %d,
                                 ptr readonly %mean, ptr %Xc) #0 {
entry:
  %rb = shl i64 %d, 2
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = mul nuw i64 %i, %d
  %src = getelementptr inbounds nuw float, ptr %X, i64 %off
  %dst = getelementptr inbounds nuw float, ptr %Xc, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %rb, i1 false)
  call void @universe_ml_axpy(ptr %dst, float -1.0, ptr %mean, i64 %d)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ================================================= covariance = Xc^T Xc /(n-1)
; Accumulated as n rank-1 outer products: for each sample, each Cov row a gets
; an axpy of scale Xc[a] with the whole centered row. Callers guarantee n >= 2.
define internal void @pca_cov(ptr readonly %Xc, i64 %n, i64 %d, ptr %cov) #0 {
entry:
  %dd = mul nuw i64 %d, %d
  %cbytes = shl i64 %dd, 2
  call void @llvm.memset.p0.i64(ptr %cov, i8 0, i64 %cbytes, i1 false)
  br label %ih

ih:
  %i = phi i64 [ 0, %entry ], [ %in, %idone ]
  %igo = icmp ult i64 %i, %n
  br i1 %igo, label %ibody, label %scaleblk

ibody:
  %ioff = mul nuw i64 %i, %d
  %xrow = getelementptr inbounds nuw float, ptr %Xc, i64 %ioff
  br label %ah

ah:
  %a = phi i64 [ 0, %ibody ], [ %an, %abody ]
  %ago = icmp ult i64 %a, %d
  br i1 %ago, label %abody, label %idone

abody:
  %ap = getelementptr inbounds nuw float, ptr %xrow, i64 %a
  %alpha = load float, ptr %ap, align 4
  %aoff = mul nuw i64 %a, %d
  %crow = getelementptr inbounds nuw float, ptr %cov, i64 %aoff
  call void @universe_ml_axpy(ptr %crow, float %alpha, ptr %xrow, i64 %d)
  %an = add nuw i64 %a, 1
  br label %ah

idone:
  %in = add nuw i64 %i, 1
  br label %ih

scaleblk:
  %nm1 = sub i64 %n, 1
  %nm1f = uitofp i64 %nm1 to float
  %inv = fdiv float 1.0, %nm1f
  call void @universe_ml_scale(ptr %cov, float %inv, i64 %dd)
  ret void
}

; ===================================== orthogonalize target vs first m comps
; Modified Gram-Schmidt: t -= (t . comp_j) comp_j for j in [0,m). comps unit.
define internal void @pca_ortho(ptr %t, ptr readonly %comp, i64 %m, i64 %d) #0 {
entry:
  %go = icmp ugt i64 %m, 0
  br i1 %go, label %loop, label %done

loop:
  %j = phi i64 [ 0, %entry ], [ %jn, %loop ]
  %off = mul nuw i64 %j, %d
  %cj = getelementptr inbounds nuw float, ptr %comp, i64 %off
  %proj = call float @universe_ml_dot(ptr %t, ptr %cj, i64 %d)
  %nproj = fneg float %proj
  call void @universe_ml_axpy(ptr %t, float %nproj, ptr %cj, i64 %d)
  %jn = add nuw i64 %j, 1
  %more = icmp ult i64 %jn, %m
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ======================================== power iteration for one component
; Returns the dominant eigenvalue of `cov` in the complement of comp[0,m);
; leaves the corresponding unit eigenvector in v. w is caller-owned scratch.
define internal float @pca_power(ptr %cov, i64 %d, i64 %iters, float %tol,
                                 ptr %v, ptr %w, ptr readonly %comp, i64 %m) #0 {
entry:
  br label %ih

ih:
  %ij = phi i64 [ 0, %entry ], [ %ijn, %ib ]
  %igo = icmp ult i64 %ij, %d
  br i1 %igo, label %ib, label %norm

ib:
  %mod = urem i64 %ij, 7
  %modf = uitofp i64 %mod to float
  %val = fadd float %modf, 1.0
  %vp = getelementptr inbounds nuw float, ptr %v, i64 %ij
  store float %val, ptr %vp, align 4
  %ijn = add nuw i64 %ij, 1
  br label %ih

norm:
  call void @pca_ortho(ptr %v, ptr %comp, i64 %m, i64 %d)
  %nrm = call float @universe_ml_l2_norm(ptr %v, i64 %d)
  %ninv = fdiv float 1.0, %nrm
  call void @universe_ml_scale(ptr %v, float %ninv, i64 %d)
  br label %ph

ph:
  %it = phi i64 [ 0, %norm ], [ %itn, %pcont ]
  %prev = phi float [ 0.0, %norm ], [ %lam, %pcont ]
  %pgo = icmp ult i64 %it, %iters
  br i1 %pgo, label %pbody, label %final

pbody:
  call void @universe_ml_gemv(ptr %w, ptr %cov, ptr %v, i64 %d, i64 %d)
  %lam = call float @universe_ml_dot(ptr %v, ptr %w, i64 %d)
  call void @pca_ortho(ptr %w, ptr %comp, i64 %m, i64 %d)
  %wn = call float @universe_ml_l2_norm(ptr %w, i64 %d)
  %wz = fcmp oeq float %wn, 0.0
  br i1 %wz, label %final, label %pcont2

pcont2:
  %winv = fdiv float 1.0, %wn
  call void @universe_ml_scale(ptr %w, float %winv, i64 %d)
  %wbytes = shl i64 %d, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %v, ptr %w, i64 %wbytes, i1 false)
  %diff = fsub float %lam, %prev
  %adiff = call float @llvm.fabs.f32(float %diff)
  %alam = call float @llvm.fabs.f32(float %lam)
  %th = fmul float %tol, %alam
  %conv = fcmp ole float %adiff, %th
  br i1 %conv, label %final, label %pcont

pcont:
  %itn = add nuw i64 %it, 1
  br label %ph

final:
  call void @universe_ml_gemv(ptr %w, ptr %cov, ptr %v, i64 %d, i64 %d)
  %lamf = call float @universe_ml_dot(ptr %v, ptr %w, i64 %d)
  ret float %lamf
}

; =================================================================== fit API
define i32 @universe_ml_pca_fit(ptr readonly %X, i64 %n, i64 %d, i64 %k,
                                ptr %comp, ptr %eig, ptr %evr,
                                i64 %iters, float %tol) #1 {
entry:
  %bn = icmp ult i64 %n, 2
  %bd = icmp eq i64 %d, 0
  %bk = icmp eq i64 %k, 0
  %bkd = icmp ugt i64 %k, %d
  %b1 = or i1 %bn, %bd
  %b2 = or i1 %bk, %bkd
  %bad = or i1 %b1, %b2
  br i1 %bad, label %einval, label %ck

ck:
  %ndo = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %n, i64 %d)
  %nd = extractvalue {i64, i1} %ndo, 0
  %ndv = extractvalue {i64, i1} %ndo, 1
  %ddo = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %d, i64 %d)
  %dd = extractvalue {i64, i1} %ddo, 0
  %ddv = extractvalue {i64, i1} %ddo, 1
  %ov1 = or i1 %ndv, %ddv
  br i1 %ov1, label %eoverflow, label %ck2

ck2:
  %sum1o = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %nd, i64 %dd)
  %sum1 = extractvalue {i64, i1} %sum1o, 0
  %sum1v = extractvalue {i64, i1} %sum1o, 1
  %d3 = mul i64 %d, 3
  %sum2o = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %sum1, i64 %d3)
  %total = extractvalue {i64, i1} %sum2o, 0
  %sum2v = extractvalue {i64, i1} %sum2o, 1
  %ov2 = or i1 %sum1v, %sum2v
  br i1 %ov2, label %eoverflow, label %alloc

alloc:
  %bytes = shl i64 %total, 2
  %scratch = call ptr @malloc(i64 %bytes)
  %isnull = icmp eq ptr %scratch, null
  br i1 %isnull, label %eoom, label %run

run:
  %mean = getelementptr inbounds nuw float, ptr %scratch, i64 %nd
  %cov = getelementptr inbounds nuw float, ptr %mean, i64 %d
  %vv = getelementptr inbounds nuw float, ptr %cov, i64 %dd
  %ww = getelementptr inbounds nuw float, ptr %vv, i64 %d
  call void @pca_colmean(ptr %X, i64 %n, i64 %d, ptr %mean)
  call void @pca_center(ptr %X, i64 %n, i64 %d, ptr %mean, ptr %scratch)
  call void @pca_cov(ptr %scratch, i64 %n, i64 %d, ptr %cov)
  br label %th

; total variance = trace(Cov) = sum of all d eigenvalues (captured pre-deflation)
th:
  %ta = phi i64 [ 0, %run ], [ %tan, %tb ]
  %tv = phi float [ 0.0, %run ], [ %tvn, %tb ]
  %tgo = icmp ult i64 %ta, %d
  br i1 %tgo, label %tb, label %comps

tb:
  %tdiag = mul nuw i64 %ta, %d
  %tidx = add nuw i64 %tdiag, %ta
  %tcp = getelementptr inbounds nuw float, ptr %cov, i64 %tidx
  %tcv = load float, ptr %tcp, align 4
  %tvn = fadd float %tv, %tcv
  %tan = add nuw i64 %ta, 1
  br label %th

comps:
  br label %mh

mh:
  %m = phi i64 [ 0, %comps ], [ %mn, %mdone ]
  %mgo = icmp ult i64 %m, %k
  br i1 %mgo, label %mbody, label %evrchk

mbody:
  %lam = call float @pca_power(ptr %cov, i64 %d, i64 %iters, float %tol,
                               ptr %vv, ptr %ww, ptr %comp, i64 %m)
  %coff = mul nuw i64 %m, %d
  %cdst = getelementptr inbounds nuw float, ptr %comp, i64 %coff
  %cpbytes = shl i64 %d, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %cdst, ptr %vv, i64 %cpbytes, i1 false)
  %ep = getelementptr inbounds nuw float, ptr %eig, i64 %m
  store float %lam, ptr %ep, align 4
  br label %dh

; deflate: Cov -= lam * v v^T  (row a gets axpy of scale -lam*v[a] with v)
dh:
  %da = phi i64 [ 0, %mbody ], [ %dan, %dbody ]
  %dgo = icmp ult i64 %da, %d
  br i1 %dgo, label %dbody, label %mdone

dbody:
  %dvp = getelementptr inbounds nuw float, ptr %vv, i64 %da
  %dvv = load float, ptr %dvp, align 4
  %lamva = fmul float %lam, %dvv
  %dalpha = fneg float %lamva
  %droff = mul nuw i64 %da, %d
  %drow = getelementptr inbounds nuw float, ptr %cov, i64 %droff
  call void @universe_ml_axpy(ptr %drow, float %dalpha, ptr %vv, i64 %d)
  %dan = add nuw i64 %da, 1
  br label %dh

mdone:
  %mn = add nuw i64 %m, 1
  br label %mh

evrchk:
  %hasevr = icmp ne ptr %evr, null
  br i1 %hasevr, label %evrpre, label %freeblk

evrpre:
  %tvz = fcmp oeq float %tv, 0.0
  br label %eh

eh:
  %em = phi i64 [ 0, %evrpre ], [ %emn, %ebody ]
  %ego = icmp ult i64 %em, %k
  br i1 %ego, label %ebody, label %freeblk

ebody:
  %evp = getelementptr inbounds nuw float, ptr %eig, i64 %em
  %evv = load float, ptr %evp, align 4
  %rat = fdiv float %evv, %tv
  %ratz = select i1 %tvz, float 0.0, float %rat
  %eop = getelementptr inbounds nuw float, ptr %evr, i64 %em
  store float %ratz, ptr %eop, align 4
  %emn = add nuw i64 %em, 1
  br label %eh

freeblk:
  call void @free(ptr %scratch)
  ret i32 0

einval:
  ret i32 8

eoverflow:
  ret i32 3

eoom:
  ret i32 2
}

; ============================================================= transform API
; Center each sample by the per-feature mean of X, then project onto the k
; components: out[i*k + m] = (X[i] - mean) . comp[m].
define i32 @universe_ml_pca_transform(ptr readonly %X, i64 %n, i64 %d,
                                      ptr readonly %comp, i64 %k, ptr %out) #1 {
entry:
  %bd = icmp eq i64 %d, 0
  %bk = icmp eq i64 %k, 0
  %bad = or i1 %bd, %bk
  br i1 %bad, label %einval, label %ok

ok:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done0, label %go

go:
  %d2 = shl i64 %d, 1
  %bytes = shl i64 %d2, 2
  %scratch = call ptr @malloc(i64 %bytes)
  %isnull = icmp eq ptr %scratch, null
  br i1 %isnull, label %eoom, label %run

run:
  %buf = getelementptr inbounds nuw float, ptr %scratch, i64 %d
  call void @pca_colmean(ptr %X, i64 %n, i64 %d, ptr %scratch)
  %rb = shl i64 %d, 2
  br label %ih

ih:
  %i = phi i64 [ 0, %run ], [ %in, %idone ]
  %igo = icmp ult i64 %i, %n
  br i1 %igo, label %ibody, label %freeok

ibody:
  %ioff = mul nuw i64 %i, %d
  %src = getelementptr inbounds nuw float, ptr %X, i64 %ioff
  call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr %src, i64 %rb, i1 false)
  call void @universe_ml_axpy(ptr %buf, float -1.0, ptr %scratch, i64 %d)
  %obase = mul nuw i64 %i, %k
  br label %mh

mh:
  %m = phi i64 [ 0, %ibody ], [ %mnn, %mbody ]
  %mgo = icmp ult i64 %m, %k
  br i1 %mgo, label %mbody, label %idone

mbody:
  %moff = mul nuw i64 %m, %d
  %cm = getelementptr inbounds nuw float, ptr %comp, i64 %moff
  %proj = call float @universe_ml_dot(ptr %buf, ptr %cm, i64 %d)
  %oidx = add nuw i64 %obase, %m
  %op = getelementptr inbounds nuw float, ptr %out, i64 %oidx
  store float %proj, ptr %op, align 4
  %mnn = add nuw i64 %m, 1
  br label %mh

idone:
  %in = add nuw i64 %i, 1
  br label %ih

freeok:
  call void @free(ptr %scratch)
  br label %done0

done0:
  ret i32 0

einval:
  ret i32 8

eoom:
  ret i32 2
}

; ============================================== explained variance ratio API
; out[m] = eigenvalues[m] / total_var  (0 when total_var == 0).
define void @universe_ml_pca_explained_variance_ratio(ptr readonly %eig, i64 %k,
                                                      float %total, ptr %out) #0 {
entry:
  %z = fcmp oeq float %total, 0.0
  %empty = icmp eq i64 %k, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %ep = getelementptr inbounds nuw float, ptr %eig, i64 %i
  %ev = load float, ptr %ep, align 4
  %r = fdiv float %ev, %total
  %rr = select i1 %z, float 0.0, float %r
  %op = getelementptr inbounds nuw float, ptr %out, i64 %i
  store float %rr, ptr %op, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %k
  br i1 %more, label %loop, label %done

done:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
