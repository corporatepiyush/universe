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

; universe_ml_kmeans_* — Lloyd's k-means over row-major f32 samples.
;
; DESIGN:
;   * ALGORITHM CLASS: Lloyd iteration. Each pass = ASSIGN (every sample to its
;     nearest centroid by squared-L2) then UPDATE (each centroid to the mean of
;     its members). Squared-L2 avoids a per-comparison sqrt and preserves the
;     argmin. Convergence when no label changes OR total centroid movement
;     (sum of per-centroid squared shift) <= tol.
;   * HOT PATH is the assign step (n*k distance evaluations); it calls the
;     internal `km_dist2`, an `alwaysinline` copy of the kernels' 4-accumulator
;     <4 x float> squared-distance reduction, so the whole n*k*d inner cost
;     vectorizes to fmla with no cross-module call (the module edge would
;     otherwise force a `bl`; we duplicate the inline shape per the house rule).
;   * INIT: first-k rows as initial centroids — deterministic, allocation-free,
;     reproducible for tests. (k-means++ is a later add.)
;   * MEMORY: one scratch malloc holds the accumulation sums (k*d f32) followed
;     by the per-cluster counts (k i64); freed on every exit. Sizes are
;     overflow-checked. Empty clusters keep their previous centroid.
;   * FP: distance/accumulation use `fast` (associative reductions, tolerance
;     ok). The min-selection uses an unflagged `olt` compare so ties break to
;     the lowest centroid index deterministically.
;
; API (f32; X is n*d row-major; C is k*d row-major):
;   i32  universe_ml_kmeans_fit(X, n, d, k, max_iter, tol,
;                               C /*out k*d*/, labels /*out n i32*/,
;                               inertia /*out float, nullable*/)
;   i32  universe_ml_kmeans_predict(X, n, d, k, C, labels /*out n i32*/)
;     returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW, 8 INVALID_ARG.

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare ptr @malloc(i64)
declare void @free(ptr)

; ================================================= km_dist2 (inlined kernel)
define internal float @km_dist2(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %acc0 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc0n, %main ]
  %acc1 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc1n, %main ]
  %acc2 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc2n, %main ]
  %acc3 = phi <4 x float> [ zeroinitializer, %entry ], [ %acc3n, %main ]
  %pa0 = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va0 = load <4 x float>, ptr %pa0, align 4
  %pb0 = getelementptr inbounds nuw float, ptr %b, i64 %i
  %vb0 = load <4 x float>, ptr %pb0, align 4
  %d0 = fsub fast <4 x float> %va0, %vb0
  %sq0 = fmul fast <4 x float> %d0, %d0
  %acc0n = fadd fast <4 x float> %acc0, %sq0
  %i1 = add nuw i64 %i, 4
  %pa1 = getelementptr inbounds nuw float, ptr %a, i64 %i1
  %va1 = load <4 x float>, ptr %pa1, align 4
  %pb1 = getelementptr inbounds nuw float, ptr %b, i64 %i1
  %vb1 = load <4 x float>, ptr %pb1, align 4
  %d1 = fsub fast <4 x float> %va1, %vb1
  %sq1 = fmul fast <4 x float> %d1, %d1
  %acc1n = fadd fast <4 x float> %acc1, %sq1
  %i2 = add nuw i64 %i, 8
  %pa2 = getelementptr inbounds nuw float, ptr %a, i64 %i2
  %va2 = load <4 x float>, ptr %pa2, align 4
  %pb2 = getelementptr inbounds nuw float, ptr %b, i64 %i2
  %vb2 = load <4 x float>, ptr %pb2, align 4
  %d2 = fsub fast <4 x float> %va2, %vb2
  %sq2 = fmul fast <4 x float> %d2, %d2
  %acc2n = fadd fast <4 x float> %acc2, %sq2
  %i3 = add nuw i64 %i, 12
  %pa3 = getelementptr inbounds nuw float, ptr %a, i64 %i3
  %va3 = load <4 x float>, ptr %pa3, align 4
  %pb3 = getelementptr inbounds nuw float, ptr %b, i64 %i3
  %vb3 = load <4 x float>, ptr %pb3, align 4
  %d3 = fsub fast <4 x float> %va3, %vb3
  %sq3 = fmul fast <4 x float> %d3, %d3
  %acc3n = fadd fast <4 x float> %acc3, %sq3
  %inext = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %main, label %maindone

maindone:
  %c0 = fadd fast <4 x float> %acc0n, %acc1n
  %c1 = fadd fast <4 x float> %acc2n, %acc3n
  %csum = fadd fast <4 x float> %c0, %c1
  %hs = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %csum)
  br label %red

red:
  %base = phi float [ 0.0, %entry ], [ %hs, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %inext, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jnext, %tailbody ]
  %sacc = phi float [ %base, %red ], [ %saccn, %tailbody ]
  %done = icmp uge i64 %j, %n
  br i1 %done, label %ret, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw float, ptr %a, i64 %j
  %fa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds nuw float, ptr %b, i64 %j
  %fb = load float, ptr %tb, align 4
  %dd = fsub fast float %fa, %fb
  %sq = fmul fast float %dd, %dd
  %saccn = fadd fast float %sacc, %sq
  %jnext = add nuw i64 %j, 1
  br label %tail

ret:
  ret float %sacc
}

; ============================================================= assign phase
; For each sample pick nearest centroid, write label, sum inertia; return the
; number of samples whose label changed from the incoming labels[].
define internal i64 @km_assign(ptr readonly %X, i64 %n, i64 %d, i64 %k,
                               ptr readonly %C, ptr %labels, ptr %inertia_out) #1 {
entry:
  br label %shead

shead:
  %i = phi i64 [ 0, %entry ], [ %inext, %sdone ]
  %chg = phi i64 [ 0, %entry ], [ %chgn, %sdone ]
  %inr = phi float [ 0.0, %entry ], [ %inrn, %sdone ]
  %go = icmp ult i64 %i, %n
  br i1 %go, label %sbody, label %fin

sbody:
  %xoff = mul nuw i64 %i, %d
  %xi = getelementptr inbounds nuw float, ptr %X, i64 %xoff
  %d0 = call float @km_dist2(ptr %xi, ptr %C, i64 %d)
  br label %chead

chead:
  %c = phi i64 [ 1, %sbody ], [ %cnext, %cbody ]
  %best = phi i64 [ 0, %sbody ], [ %bestn, %cbody ]
  %bestd = phi float [ %d0, %sbody ], [ %bestdn, %cbody ]
  %cgo = icmp ult i64 %c, %k
  br i1 %cgo, label %cbody, label %sdone

cbody:
  %coff = mul nuw i64 %c, %d
  %cc = getelementptr inbounds nuw float, ptr %C, i64 %coff
  %dc = call float @km_dist2(ptr %xi, ptr %cc, i64 %d)
  %lt = fcmp olt float %dc, %bestd
  %bestn = select i1 %lt, i64 %c, i64 %best
  %bestdn = select i1 %lt, float %dc, float %bestd
  %cnext = add nuw i64 %c, 1
  br label %chead

sdone:
  %lp = getelementptr inbounds nuw i32, ptr %labels, i64 %i
  %old = load i32, ptr %lp, align 4
  %newl = trunc i64 %best to i32
  store i32 %newl, ptr %lp, align 4
  %diff = icmp ne i32 %old, %newl
  %dinc = zext i1 %diff to i64
  %chgn = add nuw i64 %chg, %dinc
  %inrn = fadd fast float %inr, %bestd
  %inext = add nuw i64 %i, 1
  br label %shead

fin:
  store float %inr, ptr %inertia_out, align 4
  ret i64 %chg
}

; ============================================================= update phase
; Recompute centroids as the mean of assigned samples; return total squared
; centroid movement (sum over centroids of ||new-old||^2).
define internal float @km_update(ptr readonly %X, i64 %n, i64 %d, i64 %k,
                                 ptr readonly %labels, ptr %C,
                                 ptr %sums, ptr %counts) #1 {
entry:
  %kd = mul nuw i64 %k, %d
  %sumbytes = shl i64 %kd, 2
  call void @llvm.memset.p0.i64(ptr %sums, i8 0, i64 %sumbytes, i1 false)
  %cntbytes = shl i64 %k, 3
  call void @llvm.memset.p0.i64(ptr %counts, i8 0, i64 %cntbytes, i1 false)
  br label %ahead

; --- accumulate sums[label] += X[i], counts[label]++ ---
ahead:
  %i = phi i64 [ 0, %entry ], [ %inext, %adone ]
  %ago = icmp ult i64 %i, %n
  br i1 %ago, label %abody, label %uhead

abody:
  %lp = getelementptr inbounds nuw i32, ptr %labels, i64 %i
  %li32 = load i32, ptr %lp, align 4
  %li = zext i32 %li32 to i64
  %cp = getelementptr inbounds nuw i64, ptr %counts, i64 %li
  %cval = load i64, ptr %cp, align 8
  %cval1 = add nuw i64 %cval, 1
  store i64 %cval1, ptr %cp, align 8
  %xoff = mul nuw i64 %i, %d
  %xi = getelementptr inbounds nuw float, ptr %X, i64 %xoff
  %soff = mul nuw i64 %li, %d
  %srow = getelementptr inbounds nuw float, ptr %sums, i64 %soff
  br label %achead

achead:
  %j = phi i64 [ 0, %abody ], [ %jnext, %acbody ]
  %jgo = icmp ult i64 %j, %d
  br i1 %jgo, label %acbody, label %adone

acbody:
  %xjp = getelementptr inbounds nuw float, ptr %xi, i64 %j
  %xj = load float, ptr %xjp, align 4
  %sjp = getelementptr inbounds nuw float, ptr %srow, i64 %j
  %sj = load float, ptr %sjp, align 4
  %sjn = fadd fast float %sj, %xj
  store float %sjn, ptr %sjp, align 4
  %jnext = add nuw i64 %j, 1
  br label %achead

adone:
  %inext = add nuw i64 %i, 1
  br label %ahead

; --- divide and measure shift ---
uhead:
  %uc = phi i64 [ 0, %ahead ], [ %ucnext, %ucont ]
  %shift = phi float [ 0.0, %ahead ], [ %shiftn, %ucont ]
  %ugo = icmp ult i64 %uc, %k
  br i1 %ugo, label %ubody, label %ufin

ubody:
  %ccp = getelementptr inbounds nuw i64, ptr %counts, i64 %uc
  %cnt = load i64, ptr %ccp, align 8
  %empty = icmp eq i64 %cnt, 0
  %coff = mul nuw i64 %uc, %d
  %crow = getelementptr inbounds nuw float, ptr %C, i64 %coff
  %srow2 = getelementptr inbounds nuw float, ptr %sums, i64 %coff
  br i1 %empty, label %ucont, label %divhead

divhead:
  %cntf = uitofp i64 %cnt to float
  %inv = fdiv fast float 1.0, %cntf
  br label %dhead

dhead:
  %dj = phi i64 [ 0, %divhead ], [ %djnext, %dbody ]
  %dsh = phi float [ 0.0, %divhead ], [ %dshn, %dbody ]
  %dgo = icmp ult i64 %dj, %d
  br i1 %dgo, label %dbody, label %ddone

dbody:
  %sjp2 = getelementptr inbounds nuw float, ptr %srow2, i64 %dj
  %sj2 = load float, ptr %sjp2, align 4
  %mean = fmul fast float %sj2, %inv
  %cjp = getelementptr inbounds nuw float, ptr %crow, i64 %dj
  %oldc = load float, ptr %cjp, align 4
  store float %mean, ptr %cjp, align 4
  %delta = fsub fast float %mean, %oldc
  %dsq = fmul fast float %delta, %delta
  %dshn = fadd fast float %dsh, %dsq
  %djnext = add nuw i64 %dj, 1
  br label %dhead

ddone:
  %shift2 = fadd fast float %shift, %dsh
  br label %ucont

ucont:
  %shiftn = phi float [ %shift, %ubody ], [ %shift2, %ddone ]
  %ucnext = add nuw i64 %uc, 1
  br label %uhead

ufin:
  ret float %shift
}

; =================================================================== fit API
define i32 @universe_ml_kmeans_fit(ptr readonly %X, i64 %n, i64 %d, i64 %k,
                                   i64 %maxit, float %tol,
                                   ptr %C, ptr %labels, ptr %inertia) #0 {
entry:
  %inloc = alloca float, align 4
  %bad1 = icmp eq i64 %n, 0
  %bad2 = icmp eq i64 %d, 0
  %bad3 = icmp eq i64 %k, 0
  %bad4 = icmp ugt i64 %k, %n
  %b12 = or i1 %bad1, %bad2
  %b34 = or i1 %bad3, %bad4
  %bad = or i1 %b12, %b34
  br i1 %bad, label %einval, label %ok

ok:
  %kdo = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %k, i64 %d)
  %kd = extractvalue {i64, i1} %kdo, 0
  %kdov = extractvalue {i64, i1} %kdo, 1
  br i1 %kdov, label %eoverflow, label %ok2

ok2:
  %kdbytes = shl i64 %kd, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %C, ptr %X, i64 %kdbytes, i1 false)
  %cntbytes = shl i64 %k, 3
  %scrbytes = add i64 %kdbytes, %cntbytes
  %sums = call ptr @malloc(i64 %scrbytes)
  %isnull = icmp eq ptr %sums, null
  br i1 %isnull, label %eoom, label %ok3

ok3:
  %counts = getelementptr inbounds nuw i8, ptr %sums, i64 %kdbytes
  %nbytes = shl i64 %n, 2
  call void @llvm.memset.p0.i64(ptr %labels, i8 -1, i64 %nbytes, i1 false)
  %runiter = icmp ugt i64 %maxit, 0
  br i1 %runiter, label %iterloop, label %fin

iterloop:
  %iter = phi i64 [ 0, %ok3 ], [ %itn, %itcont ]
  %chg = call i64 @km_assign(ptr %X, i64 %n, i64 %d, i64 %k, ptr %C, ptr %labels, ptr %inloc)
  %conv = icmp eq i64 %chg, 0
  br i1 %conv, label %fin, label %upd

upd:
  %shift = call float @km_update(ptr %X, i64 %n, i64 %d, i64 %k, ptr %labels, ptr %C, ptr %sums, ptr %counts)
  %small = fcmp ole float %shift, %tol
  br i1 %small, label %fin, label %itcont

itcont:
  %itn = add nuw i64 %iter, 1
  %more = icmp ult i64 %itn, %maxit
  br i1 %more, label %iterloop, label %fin

fin:
  %inval = load float, ptr %inloc, align 4
  %innn = icmp ne ptr %inertia, null
  br i1 %innn, label %storein, label %freeblk

storein:
  store float %inval, ptr %inertia, align 4
  br label %freeblk

freeblk:
  call void @free(ptr %sums)
  ret i32 0

einval:
  ret i32 8

eoverflow:
  ret i32 3

eoom:
  ret i32 2
}

; =============================================================== predict API
define i32 @universe_ml_kmeans_predict(ptr readonly %X, i64 %n, i64 %d, i64 %k,
                                       ptr readonly %C, ptr %labels) #0 {
entry:
  %inloc = alloca float, align 4
  %bad2 = icmp eq i64 %d, 0
  %bad3 = icmp eq i64 %k, 0
  %bad = or i1 %bad2, %bad3
  br i1 %bad, label %einval, label %ok

ok:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %go

go:
  %chg = call i64 @km_assign(ptr %X, i64 %n, i64 %d, i64 %k, ptr %C, ptr %labels, ptr %inloc)
  br label %done

done:
  ret i32 0

einval:
  ret i32 8
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
