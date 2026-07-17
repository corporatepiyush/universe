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

; universe_ml_ivf_* — self-contained in-memory IVF-FLAT vector index.
;
; DESIGN:
;   * ALGORITHM CLASS: inverted-file index with an exact (flat) in-cell scan —
;     the middle tier between an O(n*d) brute-force flat scan and a graph index.
;     Training partitions the buffered corpus into `nlist` Voronoi cells by
;     k-means (delegated to universe_ml_kmeans_fit — a same-domain, cold,
;     once-per-build call, so a normal `bl` is correct here). A query scores the
;     `nprobe` nearest centroids, then EXHAUSTIVELY scans the members of those
;     cells into a bounded replace-worst top-k. Cost ~= O((nlist + nprobe*n/nlist)*d):
;     sub-linear when nlist ~= sqrt(n) and nprobe << nlist, exact WITHIN the
;     probed cells (nprobe == nlist degenerates to an exact flat scan).
;
;   * METRIC & DISTANCE: internally EVERYTHING is squared-L2 via ivf_dist2 (the
;     kernels' 4-accumulator <4 x float> reduction, duplicated inline per the
;     house no-cross-module-hot-call rule). Cosine (metric 0) is handled by
;     L2-NORMALIZING each vector on add and each query on search, plus spherical
;     centroids (centroids re-normalized after k-means): on the unit sphere
;     squared-L2 = 2 - 2*cos, monotonic with cosine distance, so the same kernel,
;     the same argmin, and the same top-k ordering serve both metrics. Reported
;     out_dists are that squared-L2 value.
;
;   * INVERTED LISTS: index arrays, never pointer-chased. Two-pass build after
;     training: pass 1 assigns every vector to its nearest (final, normalized)
;     centroid and counts per cell; a prefix sum yields list_off (i64[nlist+1],
;     CSR-style offsets); pass 2 scatters vector indices into a single dense
;     list_members (i32[n]) using a per-cell write cursor. Cells are sized to
;     their ACTUAL counts (no over-allocation); an empty cell is a zero-length
;     CSR range and is simply skipped.
;
;   * OWNERSHIP / LAYOUT: one opaque handle (malloc, 96 B header) owns the
;     buffered corpus (contiguous n*d f32, grown geometrically via realloc), the
;     parallel labels (i64[cap]), the trained centroids (nlist*d f32), and the
;     two CSR arrays. Everything is freed by destroy. nlist is clamped to n at
;     train time so k-means always has k <= n.
;
;   * CONCURRENCY: NONE — single-thread by contract. This is an owned, mutable
;     index (add mutates buffers/realloc; train rebuilds the model; search is
;     read-only but shares no state with a concurrent writer). There is no shared
;     mutable state across threads, hence NO atomics and no ordering to justify;
;     external synchronization is the caller's job if shared. Keeping it free of
;     any atomics keeps the hot search path a plain register/cache workload.
;
; API (f32 vectors, row-major; labels are caller i64 keys):
;   ptr  universe_ml_ivf_create(i64 dims, i32 metric /*0=cosine,1=l2*/, i64 nlist)
;   i32  universe_ml_ivf_add(ptr h, ptr vec, i64 label)
;   i32  universe_ml_ivf_train(ptr h)
;   i32  universe_ml_ivf_search(ptr h, ptr q, i64 k, i64 nprobe,
;                               ptr out_labels /*i64*/, ptr out_dists /*f32*/,
;                               ptr out_n /*i64*/)
;   i64  universe_ml_ivf_len(ptr h)
;   void universe_ml_ivf_destroy(ptr h)
;     error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY, 8 INVALID_ARG.

declare ptr @malloc(i64)
declare ptr @realloc(ptr, i64)
declare void @free(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)
declare float @llvm.sqrt.f32(float)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)

declare i32 @universe_ml_kmeans_fit(ptr, i64, i64, i64, i64, float, ptr, ptr, ptr)

; ---- handle field byte offsets ------------------------------------------------
;  +0  i64 dims      +8  i32 metric   +12 i32 trained   +16 i64 nlist_req
; +24  i64 n         +32 i64 cap      +40 ptr vectors    +48 ptr labels
; +56  ptr centroids +64 ptr list_off +72 ptr members    +80 i64 nlist_actual

; ================================================= ivf_dist2 (inlined kernel)
; Squared-L2 distance over %n f32 lanes — 4-accumulator <4 x float> reduction
; with a scalar tail; a byte-identical copy of the shared kernel shape so the
; sweep vectorizes to fmla with no cross-module call.
define internal float @ivf_dist2(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
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
  %dd1 = fsub fast <4 x float> %va1, %vb1
  %sq1 = fmul fast <4 x float> %dd1, %dd1
  %acc1n = fadd fast <4 x float> %acc1, %sq1
  %i2 = add nuw i64 %i, 8
  %pa2 = getelementptr inbounds nuw float, ptr %a, i64 %i2
  %va2 = load <4 x float>, ptr %pa2, align 4
  %pb2 = getelementptr inbounds nuw float, ptr %b, i64 %i2
  %vb2 = load <4 x float>, ptr %pb2, align 4
  %dd2 = fsub fast <4 x float> %va2, %vb2
  %sq2 = fmul fast <4 x float> %dd2, %dd2
  %acc2n = fadd fast <4 x float> %acc2, %sq2
  %i3 = add nuw i64 %i, 12
  %pa3 = getelementptr inbounds nuw float, ptr %a, i64 %i3
  %va3 = load <4 x float>, ptr %pa3, align 4
  %pb3 = getelementptr inbounds nuw float, ptr %b, i64 %i3
  %vb3 = load <4 x float>, ptr %pb3, align 4
  %dd3 = fsub fast <4 x float> %va3, %vb3
  %sq3 = fmul fast <4 x float> %dd3, %dd3
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

; ================================================= ivf_normalize (in place)
; L2-normalise %d floats in place; a zero-norm vector is left unchanged.
define internal void @ivf_normalize(ptr %v, i64 %d) #3 {
entry:
  br label %shead

shead:
  %j = phi i64 [ 0, %entry ], [ %jn, %sbody ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %sbody ]
  %go = icmp ult i64 %j, %d
  br i1 %go, label %sbody, label %div

sbody:
  %p = getelementptr inbounds nuw float, ptr %v, i64 %j
  %x = load float, ptr %p, align 4
  %sq = fmul fast float %x, %x
  %accn = fadd fast float %acc, %sq
  %jn = add nuw i64 %j, 1
  br label %shead

div:
  %zero = fcmp oeq float %acc, 0.0
  br i1 %zero, label %ret, label %scale

scale:
  %r = call float @llvm.sqrt.f32(float %acc)
  %inv = fdiv fast float 1.0, %r
  br label %mhead

mhead:
  %j2 = phi i64 [ 0, %scale ], [ %j2n, %mbody ]
  %go2 = icmp ult i64 %j2, %d
  br i1 %go2, label %mbody, label %ret

mbody:
  %p2 = getelementptr inbounds nuw float, ptr %v, i64 %j2
  %x2 = load float, ptr %p2, align 4
  %xn = fmul fast float %x2, %inv
  store float %xn, ptr %p2, align 4
  %j2n = add nuw i64 %j2, 1
  br label %mhead

ret:
  ret void
}

; ================================================= ivf_nearest (argmin cell)
; Index of the centroid nearest %x by squared-L2; lowest index breaks ties.
define internal i64 @ivf_nearest(ptr readonly %cent, i64 %nlist, i64 %d, ptr readonly %x) #1 {
entry:
  %d0 = call float @ivf_dist2(ptr %x, ptr %cent, i64 %d)
  br label %chead

chead:
  %c = phi i64 [ 1, %entry ], [ %cn, %cbody ]
  %best = phi i64 [ 0, %entry ], [ %bn, %cbody ]
  %bd = phi float [ %d0, %entry ], [ %bdn, %cbody ]
  %go = icmp ult i64 %c, %nlist
  br i1 %go, label %cbody, label %ret

cbody:
  %off = mul nuw i64 %c, %d
  %cc = getelementptr inbounds nuw float, ptr %cent, i64 %off
  %dc = call float @ivf_dist2(ptr %x, ptr %cc, i64 %d)
  %lt = fcmp olt float %dc, %bd
  %bn = select i1 %lt, i64 %c, i64 %best
  %bdn = select i1 %lt, float %dc, float %bd
  %cn = add nuw i64 %c, 1
  br label %chead

ret:
  ret i64 %best
}

; =================================================================== create
define ptr @universe_ml_ivf_create(i64 %dims, i32 %metric, i64 %nlist) #0 {
entry:
  %baddim = icmp eq i64 %dims, 0
  %badm = icmp ugt i32 %metric, 1
  %badnl = icmp eq i64 %nlist, 0
  %b1 = or i1 %baddim, %badm
  %bad = or i1 %b1, %badnl
  br i1 %bad, label %badret, label %ok

ok:
  %h = call ptr @malloc(i64 96)
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %badret, label %init

init:
  store i64 %dims, ptr %h, align 8
  %pm = getelementptr inbounds nuw i8, ptr %h, i64 8
  store i32 %metric, ptr %pm, align 8
  %pt = getelementptr inbounds nuw i8, ptr %h, i64 12
  store i32 0, ptr %pt, align 4
  %pnl = getelementptr inbounds nuw i8, ptr %h, i64 16
  store i64 %nlist, ptr %pnl, align 8
  %pn = getelementptr inbounds nuw i8, ptr %h, i64 24
  store i64 0, ptr %pn, align 8
  %pcap = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 0, ptr %pcap, align 8
  %pv = getelementptr inbounds nuw i8, ptr %h, i64 40
  store ptr null, ptr %pv, align 8
  %plb = getelementptr inbounds nuw i8, ptr %h, i64 48
  store ptr null, ptr %plb, align 8
  %pc = getelementptr inbounds nuw i8, ptr %h, i64 56
  store ptr null, ptr %pc, align 8
  %plo = getelementptr inbounds nuw i8, ptr %h, i64 64
  store ptr null, ptr %plo, align 8
  %pmb = getelementptr inbounds nuw i8, ptr %h, i64 72
  store ptr null, ptr %pmb, align 8
  %pna = getelementptr inbounds nuw i8, ptr %h, i64 80
  store i64 0, ptr %pna, align 8
  ret ptr %h

badret:
  ret ptr null
}

; ====================================================================== add
define i32 @universe_ml_ivf_add(ptr %h, ptr %vec, i64 %label) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  %vnull = icmp eq ptr %vec, null
  %anynull = or i1 %hnull, %vnull
  br i1 %anynull, label %enull, label %load

load:
  %dims = load i64, ptr %h, align 8
  %pn = getelementptr inbounds nuw i8, ptr %h, i64 24
  %n = load i64, ptr %pn, align 8
  %pcap = getelementptr inbounds nuw i8, ptr %h, i64 32
  %cap = load i64, ptr %pcap, align 8
  %pv = getelementptr inbounds nuw i8, ptr %h, i64 40
  %plb = getelementptr inbounds nuw i8, ptr %h, i64 48
  %need = icmp ult i64 %n, %cap
  br i1 %need, label %store, label %grow

grow:
  %capzero = icmp eq i64 %cap, 0
  %cap2 = shl i64 %cap, 1
  %newcap = select i1 %capzero, i64 16, i64 %cap2
  %vmo = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %newcap, i64 %dims)
  %velem = extractvalue {i64, i1} %vmo, 0
  %vov = extractvalue {i64, i1} %vmo, 1
  br i1 %vov, label %eover, label %grow2

grow2:
  %vbytes = shl i64 %velem, 2
  %oldv = load ptr, ptr %pv, align 8
  %nv = call ptr @realloc(ptr %oldv, i64 %vbytes)
  %nvnull = icmp eq ptr %nv, null
  br i1 %nvnull, label %eoom, label %grow3

grow3:
  %lbytes = shl i64 %newcap, 3
  %oldl = load ptr, ptr %plb, align 8
  %nl = call ptr @realloc(ptr %oldl, i64 %lbytes)
  %nlnull = icmp eq ptr %nl, null
  br i1 %nlnull, label %eoom, label %grow4

grow4:
  store ptr %nv, ptr %pv, align 8
  store ptr %nl, ptr %plb, align 8
  store i64 %newcap, ptr %pcap, align 8
  br label %store

store:
  %vecs = load ptr, ptr %pv, align 8
  %lbls = load ptr, ptr %plb, align 8
  %off = mul nuw i64 %n, %dims
  %dst = getelementptr inbounds nuw float, ptr %vecs, i64 %off
  %copybytes = shl i64 %dims, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %vec, i64 %copybytes, i1 false)
  %pm = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %pm, align 8
  %iscos = icmp eq i32 %metric, 0
  br i1 %iscos, label %norm, label %after

norm:
  call void @ivf_normalize(ptr %dst, i64 %dims)
  br label %after

after:
  %lp = getelementptr inbounds nuw i64, ptr %lbls, i64 %n
  store i64 %label, ptr %lp, align 8
  %n1 = add nuw i64 %n, 1
  store i64 %n1, ptr %pn, align 8
  %pt = getelementptr inbounds nuw i8, ptr %h, i64 12
  store i32 0, ptr %pt, align 4
  ret i32 0

enull:
  ret i32 1

eover:
  ret i32 3

eoom:
  ret i32 2
}

; ==================================================================== train
define i32 @universe_ml_ivf_train(ptr %h) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %enull, label %load

load:
  %pn = getelementptr inbounds nuw i8, ptr %h, i64 24
  %n = load i64, ptr %pn, align 8
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %eempty, label %clean

clean:
  %dims = load i64, ptr %h, align 8
  %pnl = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nlreq = load i64, ptr %pnl, align 8
  %pm = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %pm, align 8
  %pv = getelementptr inbounds nuw i8, ptr %h, i64 40
  %vecs = load ptr, ptr %pv, align 8
  %pc = getelementptr inbounds nuw i8, ptr %h, i64 56
  %plo = getelementptr inbounds nuw i8, ptr %h, i64 64
  %pmb = getelementptr inbounds nuw i8, ptr %h, i64 72
  %pna = getelementptr inbounds nuw i8, ptr %h, i64 80
  %pt = getelementptr inbounds nuw i8, ptr %h, i64 12
  ; free any prior model and reset to a clean untrained state
  %oc = load ptr, ptr %pc, align 8
  call void @free(ptr %oc)
  %olo = load ptr, ptr %plo, align 8
  call void @free(ptr %olo)
  %omb = load ptr, ptr %pmb, align 8
  call void @free(ptr %omb)
  store ptr null, ptr %pc, align 8
  store ptr null, ptr %plo, align 8
  store ptr null, ptr %pmb, align 8
  store i32 0, ptr %pt, align 4
  store i64 0, ptr %pna, align 8
  ; nlist = min(nlreq, n)  (>= 1 since n >= 1, nlreq >= 1)
  %ltn = icmp ult i64 %nlreq, %n
  %nlist = select i1 %ltn, i64 %nlreq, i64 %n
  ; centroids: nlist*dims f32
  %cdo = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %nlist, i64 %dims)
  %cd = extractvalue {i64, i1} %cdo, 0
  %cdov = extractvalue {i64, i1} %cdo, 1
  br i1 %cdov, label %eover, label %alloc

alloc:
  %cbytes = shl i64 %cd, 2
  %cent = call ptr @malloc(i64 %cbytes)
  %centnull = icmp eq ptr %cent, null
  br i1 %centnull, label %eoom, label %allocL

allocL:
  ; k-means scratch labels (i32[n]) — reused as the per-vector cell assignment
  %lblbytes = shl i64 %n, 2
  %kmlbl = call ptr @malloc(i64 %lblbytes)
  %kmnull = icmp eq ptr %kmlbl, null
  br i1 %kmnull, label %eoomC, label %fit

fit:
  %rc = call i32 @universe_ml_kmeans_fit(ptr %vecs, i64 %n, i64 %dims, i64 %nlist,
                                         i64 25, float 0x3F1A36E2E0000000,
                                         ptr %cent, ptr %kmlbl, ptr null)
  ; on cosine, re-normalise centroids so the assignment is spherical
  %iscos = icmp eq i32 %metric, 0
  br i1 %iscos, label %cnhead, label %listalloc

cnhead:
  %cc = phi i64 [ 0, %fit ], [ %ccn, %cnbody ]
  %cngo = icmp ult i64 %cc, %nlist
  br i1 %cngo, label %cnbody, label %listalloc

cnbody:
  %coff = mul nuw i64 %cc, %dims
  %crow = getelementptr inbounds nuw float, ptr %cent, i64 %coff
  call void @ivf_normalize(ptr %crow, i64 %dims)
  %ccn = add nuw i64 %cc, 1
  br label %cnhead

listalloc:
  ; list_off: i64[nlist+1]
  %nlp1 = add nuw i64 %nlist, 1
  %lobytes = shl i64 %nlp1, 3
  %listoff = call ptr @malloc(i64 %lobytes)
  %lonull = icmp eq ptr %listoff, null
  br i1 %lonull, label %eoomL, label %listalloc2

listalloc2:
  ; counts / write-cursor: i64[nlist]
  %cntbytes = shl i64 %nlist, 3
  %counts = call ptr @malloc(i64 %cntbytes)
  %cntnull = icmp eq ptr %counts, null
  br i1 %cntnull, label %eoomLO, label %listalloc3

listalloc3:
  call void @llvm.memset.p0.i64(ptr %counts, i8 0, i64 %cntbytes, i1 false)
  %membbytes = shl i64 %n, 2
  %memb = call ptr @malloc(i64 %membbytes)
  %membnull = icmp eq ptr %memb, null
  br i1 %membnull, label %eoomLOC, label %p1head

; --- pass 1: assign + count ---
p1head:
  %i1 = phi i64 [ 0, %listalloc3 ], [ %i1n, %p1body ]
  %p1go = icmp ult i64 %i1, %n
  br i1 %p1go, label %p1body, label %pfx

p1body:
  %xoff = mul nuw i64 %i1, %dims
  %xi = getelementptr inbounds nuw float, ptr %vecs, i64 %xoff
  %cell = call i64 @ivf_nearest(ptr %cent, i64 %nlist, i64 %dims, ptr %xi)
  %alp = getelementptr inbounds nuw i32, ptr %kmlbl, i64 %i1
  %cell32 = trunc i64 %cell to i32
  store i32 %cell32, ptr %alp, align 4
  %cp = getelementptr inbounds nuw i64, ptr %counts, i64 %cell
  %cv = load i64, ptr %cp, align 8
  %cv1 = add nuw i64 %cv, 1
  store i64 %cv1, ptr %cp, align 8
  %i1n = add nuw i64 %i1, 1
  br label %p1head

; --- prefix sum: list_off[c]=acc; counts[c]=acc (cursor start); acc+=cnt ---
pfx:
  %pc2 = phi i64 [ 0, %p1head ], [ %pc2n, %pfxbody ]
  %acc = phi i64 [ 0, %p1head ], [ %accn, %pfxbody ]
  %pfxgo = icmp ult i64 %pc2, %nlist
  br i1 %pfxgo, label %pfxbody, label %pfxfin

pfxbody:
  %lop = getelementptr inbounds nuw i64, ptr %listoff, i64 %pc2
  store i64 %acc, ptr %lop, align 8
  %cur = getelementptr inbounds nuw i64, ptr %counts, i64 %pc2
  %cnt = load i64, ptr %cur, align 8
  store i64 %acc, ptr %cur, align 8
  %accn = add nuw i64 %acc, %cnt
  %pc2n = add nuw i64 %pc2, 1
  br label %pfx

pfxfin:
  %lopN = getelementptr inbounds nuw i64, ptr %listoff, i64 %nlist
  store i64 %acc, ptr %lopN, align 8
  br label %p2head

; --- pass 2: scatter member indices ---
p2head:
  %i2 = phi i64 [ 0, %pfxfin ], [ %i2n, %p2body ]
  %p2go = icmp ult i64 %i2, %n
  br i1 %p2go, label %p2body, label %finish

p2body:
  %alp2 = getelementptr inbounds nuw i32, ptr %kmlbl, i64 %i2
  %cell2i = load i32, ptr %alp2, align 4
  %cell2 = zext i32 %cell2i to i64
  %cur2 = getelementptr inbounds nuw i64, ptr %counts, i64 %cell2
  %pos = load i64, ptr %cur2, align 8
  %pos1 = add nuw i64 %pos, 1
  store i64 %pos1, ptr %cur2, align 8
  %mp = getelementptr inbounds nuw i32, ptr %memb, i64 %pos
  %i2_32 = trunc i64 %i2 to i32
  store i32 %i2_32, ptr %mp, align 4
  %i2n = add nuw i64 %i2, 1
  br label %p2head

finish:
  store ptr %cent, ptr %pc, align 8
  store ptr %listoff, ptr %plo, align 8
  store ptr %memb, ptr %pmb, align 8
  store i64 %nlist, ptr %pna, align 8
  store i32 1, ptr %pt, align 4
  call void @free(ptr %counts)
  call void @free(ptr %kmlbl)
  ret i32 0

enull:
  ret i32 1

eempty:
  ret i32 4

eover:
  ret i32 3

eoom:
  ret i32 2

eoomC:
  call void @free(ptr %cent)
  ret i32 2

eoomL:
  call void @free(ptr %kmlbl)
  call void @free(ptr %cent)
  ret i32 2

eoomLO:
  call void @free(ptr %listoff)
  call void @free(ptr %kmlbl)
  call void @free(ptr %cent)
  ret i32 2

eoomLOC:
  call void @free(ptr %counts)
  call void @free(ptr %listoff)
  call void @free(ptr %kmlbl)
  call void @free(ptr %cent)
  ret i32 2
}

; =================================================================== search
define i32 @universe_ml_ivf_search(ptr %h, ptr %q, i64 %k, i64 %nprobe,
                                   ptr %out_labels, ptr %out_dists, ptr %out_n) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  %qnull = icmp eq ptr %q, null
  %oln = icmp eq ptr %out_labels, null
  %odn = icmp eq ptr %out_dists, null
  %onn = icmp eq ptr %out_n, null
  %z1 = or i1 %hnull, %qnull
  %z2 = or i1 %oln, %odn
  %z3 = or i1 %z2, %onn
  %anynull = or i1 %z1, %z3
  br i1 %anynull, label %enull, label %chk

chk:
  %pt = getelementptr inbounds nuw i8, ptr %h, i64 12
  %trained = load i32, ptr %pt, align 4
  %untrained = icmp eq i32 %trained, 0
  %kzero = icmp eq i64 %k, 0
  %binval = or i1 %untrained, %kzero
  br i1 %binval, label %einval, label %load

load:
  %dims = load i64, ptr %h, align 8
  %pm = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %pm, align 8
  %pn = getelementptr inbounds nuw i8, ptr %h, i64 24
  %n = load i64, ptr %pn, align 8
  %pv = getelementptr inbounds nuw i8, ptr %h, i64 40
  %vecs = load ptr, ptr %pv, align 8
  %plb = getelementptr inbounds nuw i8, ptr %h, i64 48
  %lbls = load ptr, ptr %plb, align 8
  %pc = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cent = load ptr, ptr %pc, align 8
  %plo = getelementptr inbounds nuw i8, ptr %h, i64 64
  %listoff = load ptr, ptr %plo, align 8
  %pmb = getelementptr inbounds nuw i8, ptr %h, i64 72
  %memb = load ptr, ptr %pmb, align 8
  %pna = getelementptr inbounds nuw i8, ptr %h, i64 80
  %nlist = load i64, ptr %pna, align 8
  ; keff = min(k, n)
  %kltn = icmp ult i64 %k, %n
  %keff = select i1 %kltn, i64 %k, i64 %n
  ; np = clamp(nprobe, 1, nlist)
  %npz = icmp eq i64 %nprobe, 0
  %np0 = select i1 %npz, i64 1, i64 %nprobe
  %npgt = icmp ugt i64 %np0, %nlist
  %np = select i1 %npgt, i64 %nlist, i64 %np0
  ; scratch layout (single malloc): probe_i(np*8) besti(keff*8) probe_d(np*4)
  ;   bestd(keff*4) qn(cosine? dims*4 : 0)
  %iscos = icmp eq i32 %metric, 0
  %o_besti = shl i64 %np, 3
  %npke = add nuw i64 %np, %keff
  %o_probed = shl i64 %npke, 3
  %npb = shl i64 %np, 2
  %o_bestd = add nuw i64 %o_probed, %npb
  %keb = shl i64 %keff, 2
  %o_qn = add nuw i64 %o_bestd, %keb
  %qnbytes = shl i64 %dims, 2
  %qnsel = select i1 %iscos, i64 %qnbytes, i64 0
  %total = add nuw i64 %o_qn, %qnsel
  %scr = call ptr @malloc(i64 %total)
  %scrnull = icmp eq ptr %scr, null
  br i1 %scrnull, label %eoom, label %carve

carve:
  %probe_i = getelementptr inbounds nuw i8, ptr %scr, i64 0
  %besti = getelementptr inbounds nuw i8, ptr %scr, i64 %o_besti
  %probe_d = getelementptr inbounds nuw i8, ptr %scr, i64 %o_probed
  %bestd = getelementptr inbounds nuw i8, ptr %scr, i64 %o_bestd
  %qn = getelementptr inbounds nuw i8, ptr %scr, i64 %o_qn
  ; query pointer: normalized copy for cosine, else the query as-is
  br i1 %iscos, label %mkqn, label %probeinit

mkqn:
  call void @llvm.memcpy.p0.p0.i64(ptr %qn, ptr %q, i64 %qnbytes, i1 false)
  call void @ivf_normalize(ptr %qn, i64 %dims)
  br label %probeinit

probeinit:
  %qptr = phi ptr [ %qn, %mkqn ], [ %q, %carve ]
  br label %pihead

; --- init probe arrays to +inf / -1 ---
pihead:
  %pi = phi i64 [ 0, %probeinit ], [ %pin, %pibody ]
  %pigo = icmp ult i64 %pi, %np
  br i1 %pigo, label %pibody, label %probehead

pibody:
  %pdp = getelementptr inbounds nuw float, ptr %probe_d, i64 %pi
  store float 0x7FF0000000000000, ptr %pdp, align 4
  %pip = getelementptr inbounds nuw i64, ptr %probe_i, i64 %pi
  store i64 -1, ptr %pip, align 8
  %pin = add nuw i64 %pi, 1
  br label %pihead

; --- select np nearest centroids (replace-worst) ---
probehead:
  %ci = phi i64 [ 0, %pihead ], [ %cin, %probecont ]
  %cigo = icmp ult i64 %ci, %nlist
  br i1 %cigo, label %probebody, label %resinit

probebody:
  %coff = mul nuw i64 %ci, %dims
  %crow = getelementptr inbounds nuw float, ptr %cent, i64 %coff
  %cdist = call float @ivf_dist2(ptr %qptr, ptr %crow, i64 %dims)
  %pd0 = load float, ptr %probe_d, align 4
  br label %pwhead

pwhead:
  %pw = phi i64 [ 1, %probebody ], [ %pwn, %pwbody ]
  %pwpos = phi i64 [ 0, %probebody ], [ %pwposn, %pwbody ]
  %pwval = phi float [ %pd0, %probebody ], [ %pwvaln, %pwbody ]
  %pwgo = icmp ult i64 %pw, %np
  br i1 %pwgo, label %pwbody, label %pwdone

pwbody:
  %pwp = getelementptr inbounds nuw float, ptr %probe_d, i64 %pw
  %pwv = load float, ptr %pwp, align 4
  %pwgt = fcmp ogt float %pwv, %pwval
  %pwposn = select i1 %pwgt, i64 %pw, i64 %pwpos
  %pwvaln = select i1 %pwgt, float %pwv, float %pwval
  %pwn = add nuw i64 %pw, 1
  br label %pwhead

pwdone:
  %pbetter = fcmp olt float %cdist, %pwval
  br i1 %pbetter, label %preplace, label %probecont

preplace:
  %rpd = getelementptr inbounds nuw float, ptr %probe_d, i64 %pwpos
  store float %cdist, ptr %rpd, align 4
  %rpi = getelementptr inbounds nuw i64, ptr %probe_i, i64 %pwpos
  store i64 %ci, ptr %rpi, align 8
  br label %probecont

probecont:
  %cin = add nuw i64 %ci, 1
  br label %probehead

; --- init result top-k to +inf / -1 ---
resinit:
  br label %rihead

rihead:
  %ri = phi i64 [ 0, %resinit ], [ %rin, %ribody ]
  %rigo = icmp ult i64 %ri, %keff
  br i1 %rigo, label %ribody, label %scanhead

ribody:
  %rdp = getelementptr inbounds nuw float, ptr %bestd, i64 %ri
  store float 0x7FF0000000000000, ptr %rdp, align 4
  %rip = getelementptr inbounds nuw i64, ptr %besti, i64 %ri
  store i64 -1, ptr %rip, align 8
  %rin = add nuw i64 %ri, 1
  br label %rihead

; --- scan members of the probed cells into the result top-k ---
scanhead:
  %sp = phi i64 [ 0, %rihead ], [ %spn, %scancont ]
  %spgo = icmp ult i64 %sp, %np
  br i1 %spgo, label %scanbody, label %sortinit

scanbody:
  %spip = getelementptr inbounds nuw i64, ptr %probe_i, i64 %sp
  %cellid = load i64, ptr %spip, align 8
  %lo_a = getelementptr inbounds nuw i64, ptr %listoff, i64 %cellid
  %mstart = load i64, ptr %lo_a, align 8
  %cellid1 = add nuw i64 %cellid, 1
  %lo_b = getelementptr inbounds nuw i64, ptr %listoff, i64 %cellid1
  %mend = load i64, ptr %lo_b, align 8
  br label %mhead

mhead:
  %mi = phi i64 [ %mstart, %scanbody ], [ %min, %mcont ]
  %mgo = icmp ult i64 %mi, %mend
  br i1 %mgo, label %mbody, label %scancont

mbody:
  %mmp = getelementptr inbounds nuw i32, ptr %memb, i64 %mi
  %vi32 = load i32, ptr %mmp, align 4
  %vi = zext i32 %vi32 to i64
  %voff = mul nuw i64 %vi, %dims
  %vrow = getelementptr inbounds nuw float, ptr %vecs, i64 %voff
  %vdist = call float @ivf_dist2(ptr %qptr, ptr %vrow, i64 %dims)
  %bd0 = load float, ptr %bestd, align 4
  br label %rwhead

rwhead:
  %rw = phi i64 [ 1, %mbody ], [ %rwn, %rwbody ]
  %rwpos = phi i64 [ 0, %mbody ], [ %rwposn, %rwbody ]
  %rwval = phi float [ %bd0, %mbody ], [ %rwvaln, %rwbody ]
  %rwgo = icmp ult i64 %rw, %keff
  br i1 %rwgo, label %rwbody, label %rwdone

rwbody:
  %rwp = getelementptr inbounds nuw float, ptr %bestd, i64 %rw
  %rwv = load float, ptr %rwp, align 4
  %rwgt = fcmp ogt float %rwv, %rwval
  %rwposn = select i1 %rwgt, i64 %rw, i64 %rwpos
  %rwvaln = select i1 %rwgt, float %rwv, float %rwval
  %rwn = add nuw i64 %rw, 1
  br label %rwhead

rwdone:
  %rbetter = fcmp olt float %vdist, %rwval
  br i1 %rbetter, label %rreplace, label %mcont

rreplace:
  %rrd = getelementptr inbounds nuw float, ptr %bestd, i64 %rwpos
  store float %vdist, ptr %rrd, align 4
  %rri = getelementptr inbounds nuw i64, ptr %besti, i64 %rwpos
  store i64 %vi, ptr %rri, align 8
  br label %mcont

mcont:
  %min = add nuw i64 %mi, 1
  br label %mhead

scancont:
  %spn = add nuw i64 %sp, 1
  br label %scanhead

; --- selection sort the keff result slots ascending by distance ---
sortinit:
  br label %sohead

sohead:
  %sa = phi i64 [ 0, %sortinit ], [ %san, %soafter ]
  %sago = icmp ult i64 %sa, %keff
  br i1 %sago, label %sobody, label %emit

sobody:
  %sap = getelementptr inbounds nuw float, ptr %bestd, i64 %sa
  %saval = load float, ptr %sap, align 4
  %sb0 = add nuw i64 %sa, 1
  br label %sihead

sihead:
  %sb = phi i64 [ %sb0, %sobody ], [ %sbn, %sibody ]
  %minpos = phi i64 [ %sa, %sobody ], [ %minposn, %sibody ]
  %minval = phi float [ %saval, %sobody ], [ %minvaln, %sibody ]
  %sigo = icmp ult i64 %sb, %keff
  br i1 %sigo, label %sibody, label %sidone

sibody:
  %sbp = getelementptr inbounds nuw float, ptr %bestd, i64 %sb
  %sbv = load float, ptr %sbp, align 4
  %silt = fcmp olt float %sbv, %minval
  %minposn = select i1 %silt, i64 %sb, i64 %minpos
  %minvaln = select i1 %silt, float %sbv, float %minval
  %sbn = add nuw i64 %sb, 1
  br label %sihead

sidone:
  %needswap = icmp ne i64 %minpos, %sa
  br i1 %needswap, label %soswap, label %soafter

soswap:
  ; swap distances
  %mpd = getelementptr inbounds nuw float, ptr %bestd, i64 %minpos
  %mpdv = load float, ptr %mpd, align 4
  store float %mpdv, ptr %sap, align 4
  store float %saval, ptr %mpd, align 4
  ; swap indices
  %sai = getelementptr inbounds nuw i64, ptr %besti, i64 %sa
  %saiv = load i64, ptr %sai, align 8
  %mpi = getelementptr inbounds nuw i64, ptr %besti, i64 %minpos
  %mpiv = load i64, ptr %mpi, align 8
  store i64 %mpiv, ptr %sai, align 8
  store i64 %saiv, ptr %mpi, align 8
  br label %soafter

soafter:
  %san = add nuw i64 %sa, 1
  br label %sohead

; --- emit sorted reals (besti != -1); +inf/-1 slots sort to the tail ---
emit:
  br label %emhead

emhead:
  %ej = phi i64 [ 0, %emit ], [ %ejn, %embody ]
  %ejlt = icmp ult i64 %ej, %keff
  br i1 %ejlt, label %emcheck, label %emfin

emcheck:
  %eip = getelementptr inbounds nuw i64, ptr %besti, i64 %ej
  %eiv = load i64, ptr %eip, align 8
  %ereal = icmp sge i64 %eiv, 0
  br i1 %ereal, label %embody, label %emfin

embody:
  %elp = getelementptr inbounds nuw i64, ptr %lbls, i64 %eiv
  %elbl = load i64, ptr %elp, align 8
  %olp = getelementptr inbounds nuw i64, ptr %out_labels, i64 %ej
  store i64 %elbl, ptr %olp, align 8
  %edp = getelementptr inbounds nuw float, ptr %bestd, i64 %ej
  %edv = load float, ptr %edp, align 4
  %odp = getelementptr inbounds nuw float, ptr %out_dists, i64 %ej
  store float %edv, ptr %odp, align 4
  %ejn = add nuw i64 %ej, 1
  br label %emhead

emfin:
  store i64 %ej, ptr %out_n, align 8
  call void @free(ptr %scr)
  ret i32 0

enull:
  ret i32 1

einval:
  ret i32 8

eoom:
  ret i32 2
}

; ====================================================================== len
define i64 @universe_ml_ivf_len(ptr %h) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %zero, label %load

load:
  %pn = getelementptr inbounds nuw i8, ptr %h, i64 24
  %n = load i64, ptr %pn, align 8
  ret i64 %n

zero:
  ret i64 0
}

; ================================================================== destroy
define void @universe_ml_ivf_destroy(ptr %h) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %ret, label %free

free:
  %pv = getelementptr inbounds nuw i8, ptr %h, i64 40
  %vecs = load ptr, ptr %pv, align 8
  call void @free(ptr %vecs)
  %plb = getelementptr inbounds nuw i8, ptr %h, i64 48
  %lbls = load ptr, ptr %plb, align 8
  call void @free(ptr %lbls)
  %pc = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cent = load ptr, ptr %pc, align 8
  call void @free(ptr %cent)
  %plo = getelementptr inbounds nuw i8, ptr %h, i64 64
  %listoff = load ptr, ptr %plo, align 8
  call void @free(ptr %listoff)
  %pmb = getelementptr inbounds nuw i8, ptr %h, i64 72
  %memb = load ptr, ptr %pmb, align 8
  call void @free(ptr %memb)
  call void @free(ptr %h)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
