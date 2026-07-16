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

; universe_ml_knn_* — brute-force k-nearest-neighbors classify / regress over
; row-major f32 training data.
;
; DESIGN:
;   * ALGORITHM CLASS: exact brute force. For each query we evaluate the
;     squared-L2 distance to every training row (squared-L2 preserves the
;     nearest-neighbor ordering and skips a per-point sqrt) and keep the k
;     smallest via a "replace-worst" top-k: a k-slot array of best distances +
;     indices, initialised to +inf; each new distance smaller than the current
;     worst slot evicts it. That is O(n*k) selection with an O(k) worst-scan —
;     ideal for the small k of a kNN model and branch-predictable.
;   * HOT PATH is the n*d distance sweep; it uses `knn_dist2`, an
;     `alwaysinline` copy of the kernels' 4-accumulator <4 x float> squared
;     distance, so the sweep vectorizes to fmla with no cross-module call.
;   * classify: uniform majority vote over the k neighbour labels (counts
;     buffer of `nclasses`, argmax with lowest-index tie-break). regress:
;     uniform mean of the k neighbour targets.
;   * MEMORY: one scratch malloc per call holds the k best distances (f32), the
;     k best indices (i64), and — for classify — the class counts (i64);
;     reused across all queries, freed on exit. No per-query allocation.
;
; API (f32 features; X* are row-major *_rows x d):
;   i32 universe_ml_knn_classify(Xtrain, ytrain /*n i32*/, n, d,
;                                Xq, nq, k, nclasses, out /*nq i32*/)
;   i32 universe_ml_knn_regress (Xtrain, ytrain /*n f32*/, n, d,
;                                Xq, nq, k, out /*nq f32*/)
;     returns 0 OK, 2 OOM, 8 INVALID_ARG.

declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; ================================================= knn_dist2 (inlined kernel)
define internal float @knn_dist2(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
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

; ======================================================= top-k (replace-worst)
; Fills bestd[0..k) / besti[0..k) with the k nearest training rows to xq.
; Requires k <= n so every slot ends real.
define internal void @knn_topk(ptr readonly %Xtrain, i64 %n, i64 %d,
                               ptr readonly %xq, i64 %k,
                               ptr %bestd, ptr %besti) #1 {
entry:
  br label %inithead

inithead:
  %ii = phi i64 [ 0, %entry ], [ %iin, %inithead ]
  %bdp = getelementptr inbounds nuw float, ptr %bestd, i64 %ii
  store float 0x7FF0000000000000, ptr %bdp, align 4
  %bip = getelementptr inbounds nuw i64, ptr %besti, i64 %ii
  store i64 -1, ptr %bip, align 8
  %iin = add nuw i64 %ii, 1
  %imore = icmp ult i64 %iin, %k
  br i1 %imore, label %inithead, label %scanhead

scanhead:
  %j = phi i64 [ 0, %inithead ], [ %jn, %scancont ]
  %jgo = icmp ult i64 %j, %n
  br i1 %jgo, label %scanbody, label %fin

scanbody:
  %roff = mul nuw i64 %j, %d
  %row = getelementptr inbounds nuw float, ptr %Xtrain, i64 %roff
  %dist = call float @knn_dist2(ptr %xq, ptr %row, i64 %d)
  ; find current worst slot; initial worst = bestd[0]
  %d0slot = load float, ptr %bestd, align 4
  br label %mhead

mhead:
  %p = phi i64 [ 1, %scanbody ], [ %pn, %mbody ]
  %maxpos = phi i64 [ 0, %scanbody ], [ %maxposn, %mbody ]
  %maxval = phi float [ %d0slot, %scanbody ], [ %maxvaln, %mbody ]
  %pgo = icmp ult i64 %p, %k
  br i1 %pgo, label %mbody, label %mdone

mbody:
  %pp = getelementptr inbounds nuw float, ptr %bestd, i64 %p
  %pv = load float, ptr %pp, align 4
  %gt = fcmp ogt float %pv, %maxval
  %maxposn = select i1 %gt, i64 %p, i64 %maxpos
  %maxvaln = select i1 %gt, float %pv, float %maxval
  %pn = add nuw i64 %p, 1
  br label %mhead

mdone:
  %better = fcmp olt float %dist, %maxval
  br i1 %better, label %replace, label %scancont

replace:
  %rdp = getelementptr inbounds nuw float, ptr %bestd, i64 %maxpos
  store float %dist, ptr %rdp, align 4
  %rip = getelementptr inbounds nuw i64, ptr %besti, i64 %maxpos
  store i64 %j, ptr %rip, align 8
  br label %scancont

scancont:
  %jn = add nuw i64 %j, 1
  br label %scanhead

fin:
  ret void
}

; ============================================================== classify API
define i32 @universe_ml_knn_classify(ptr readonly %Xtrain, ptr readonly %ytrain,
                                     i64 %n, i64 %d, ptr readonly %Xq, i64 %nq,
                                     i64 %k, i64 %nclasses, ptr %out) #0 {
entry:
  %bad1 = icmp eq i64 %n, 0
  %bad2 = icmp eq i64 %d, 0
  %bad3 = icmp eq i64 %k, 0
  %bad4 = icmp ugt i64 %k, %n
  %bad5 = icmp eq i64 %nclasses, 0
  %b12 = or i1 %bad1, %bad2
  %b34 = or i1 %bad3, %bad4
  %b125 = or i1 %b12, %bad5
  %bad = or i1 %b125, %b34
  br i1 %bad, label %einval, label %ok

ok:
  %qempty = icmp eq i64 %nq, 0
  br i1 %qempty, label %retok, label %alloc

alloc:
  ; scratch: bestd (k f32) + besti (k i64) + counts (nclasses i64)
  %kf = shl i64 %k, 2
  %ki = shl i64 %k, 3
  %cc = shl i64 %nclasses, 3
  %t0 = add i64 %kf, %ki
  %tot = add i64 %t0, %cc
  %scr = call ptr @malloc(i64 %tot)
  %isnull = icmp eq ptr %scr, null
  br i1 %isnull, label %eoom, label %run

run:
  %bestd = getelementptr inbounds nuw i8, ptr %scr, i64 0
  %besti = getelementptr inbounds nuw i8, ptr %scr, i64 %kf
  %off2 = add i64 %kf, %ki
  %counts = getelementptr inbounds nuw i8, ptr %scr, i64 %off2
  br label %qhead

qhead:
  %q = phi i64 [ 0, %run ], [ %qn, %qcont ]
  %qgo = icmp ult i64 %q, %nq
  br i1 %qgo, label %qbody, label %freeok

qbody:
  %qoff = mul nuw i64 %q, %d
  %xq = getelementptr inbounds nuw float, ptr %Xq, i64 %qoff
  call void @knn_topk(ptr %Xtrain, i64 %n, i64 %d, ptr %xq, i64 %k, ptr %bestd, ptr %besti)
  ; zero counts
  call void @llvm.memset.p0.i64(ptr %counts, i8 0, i64 %cc, i1 false)
  ; vote
  br label %vhead

vhead:
  %vp = phi i64 [ 0, %qbody ], [ %vpn, %vbody ]
  %vgo = icmp ult i64 %vp, %k
  br i1 %vgo, label %vbody, label %argmax

vbody:
  %bip = getelementptr inbounds nuw i64, ptr %besti, i64 %vp
  %idx = load i64, ptr %bip, align 8
  %lp = getelementptr inbounds nuw i32, ptr %ytrain, i64 %idx
  %lbl32 = load i32, ptr %lp, align 4
  %lbl = zext i32 %lbl32 to i64
  %cp = getelementptr inbounds nuw i64, ptr %counts, i64 %lbl
  %cv = load i64, ptr %cp, align 8
  %cv1 = add nuw i64 %cv, 1
  store i64 %cv1, ptr %cp, align 8
  %vpn = add nuw i64 %vp, 1
  br label %vhead

argmax:
  %c0p = getelementptr inbounds nuw i64, ptr %counts, i64 0
  %c0 = load i64, ptr %c0p, align 8
  br label %ahead

ahead:
  %ac = phi i64 [ 1, %argmax ], [ %acn, %abody ]
  %abest = phi i64 [ 0, %argmax ], [ %abestn, %abody ]
  %abestc = phi i64 [ %c0, %argmax ], [ %abestcn, %abody ]
  %ago = icmp ult i64 %ac, %nclasses
  br i1 %ago, label %abody, label %adone

abody:
  %acp = getelementptr inbounds nuw i64, ptr %counts, i64 %ac
  %acv = load i64, ptr %acp, align 8
  %agt = icmp ugt i64 %acv, %abestc
  %abestn = select i1 %agt, i64 %ac, i64 %abest
  %abestcn = select i1 %agt, i64 %acv, i64 %abestc
  %acn = add nuw i64 %ac, 1
  br label %ahead

adone:
  %outp = getelementptr inbounds nuw i32, ptr %out, i64 %q
  %pred = trunc i64 %abest to i32
  store i32 %pred, ptr %outp, align 4
  br label %qcont

qcont:
  %qn = add nuw i64 %q, 1
  br label %qhead

freeok:
  call void @free(ptr %scr)
  ret i32 0

retok:
  ret i32 0

einval:
  ret i32 8

eoom:
  ret i32 2
}

; =============================================================== regress API
define i32 @universe_ml_knn_regress(ptr readonly %Xtrain, ptr readonly %ytrain,
                                    i64 %n, i64 %d, ptr readonly %Xq, i64 %nq,
                                    i64 %k, ptr %out) #0 {
entry:
  %bad1 = icmp eq i64 %n, 0
  %bad2 = icmp eq i64 %d, 0
  %bad3 = icmp eq i64 %k, 0
  %bad4 = icmp ugt i64 %k, %n
  %b12 = or i1 %bad1, %bad2
  %b34 = or i1 %bad3, %bad4
  %bad = or i1 %b12, %b34
  br i1 %bad, label %einval, label %ok

ok:
  %qempty = icmp eq i64 %nq, 0
  br i1 %qempty, label %retok, label %alloc

alloc:
  %kf = shl i64 %k, 2
  %ki = shl i64 %k, 3
  %tot = add i64 %kf, %ki
  %scr = call ptr @malloc(i64 %tot)
  %isnull = icmp eq ptr %scr, null
  br i1 %isnull, label %eoom, label %run

run:
  %bestd = getelementptr inbounds nuw i8, ptr %scr, i64 0
  %besti = getelementptr inbounds nuw i8, ptr %scr, i64 %kf
  %kf32 = uitofp i64 %k to float
  br label %qhead

qhead:
  %q = phi i64 [ 0, %run ], [ %qn, %qcont ]
  %qgo = icmp ult i64 %q, %nq
  br i1 %qgo, label %qbody, label %freeok

qbody:
  %qoff = mul nuw i64 %q, %d
  %xq = getelementptr inbounds nuw float, ptr %Xq, i64 %qoff
  call void @knn_topk(ptr %Xtrain, i64 %n, i64 %d, ptr %xq, i64 %k, ptr %bestd, ptr %besti)
  br label %shead

shead:
  %sp = phi i64 [ 0, %qbody ], [ %spn, %sbody ]
  %sacc = phi float [ 0.0, %qbody ], [ %saccn, %sbody ]
  %sgo = icmp ult i64 %sp, %k
  br i1 %sgo, label %sbody, label %sdone

sbody:
  %bip = getelementptr inbounds nuw i64, ptr %besti, i64 %sp
  %idx = load i64, ptr %bip, align 8
  %yp = getelementptr inbounds nuw float, ptr %ytrain, i64 %idx
  %yv = load float, ptr %yp, align 4
  %saccn = fadd fast float %sacc, %yv
  %spn = add nuw i64 %sp, 1
  br label %shead

sdone:
  %mean = fdiv fast float %sacc, %kf32
  %outp = getelementptr inbounds nuw float, ptr %out, i64 %q
  store float %mean, ptr %outp, align 4
  br label %qcont

qcont:
  %qn = add nuw i64 %q, 1
  br label %qhead

freeok:
  call void @free(ptr %scr)
  ret i32 0

retok:
  ret i32 0

einval:
  ret i32 8

eoom:
  ret i32 2
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
