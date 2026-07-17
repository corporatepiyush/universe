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

; universe_ml_hnsw_* — in-memory HNSW approximate nearest-neighbor index
; (Hierarchical Navigable Small World, multi-layer proximity graph).
;
; ============================================================================
; DESIGN
; ----------------------------------------------------------------------------
; ALGORITHM CLASS: a hierarchical navigable small-world graph gives O(log N)
; expected query cost vs the O(N) flat scan of universe_ml_knn. A node's level
; is drawn from a geometric distribution; the sparse upper layers are a
; "highway" descended greedily (ef=1) to land near the query, then layer 0 is
; explored best-first with breadth `ef` collecting the candidate neighbourhood.
; Graph quality comes from the neighbour-selection heuristic (Algorithm 4):
; keep a candidate only if it is closer to the new node than to every
; already-kept neighbour — this spreads links out and sharply raises recall
; over plain nearest-M. Over-budget nodes are re-pruned by the same heuristic.
;
; LAYOUT (SoA, flat index-linked graph — NO pointer chasing on the hot path):
;   * vectors  : capacity*dims f32, row-major. cosine => normalized on store so
;     squared-L2 is monotone with cosine distance (||a-b||^2 = 2-2cos for unit
;     vectors); ONE internal distance kernel (squared-L2, 4-acc <4 x float>)
;     serves both metrics.
;   * labels   : capacity i64 (caller key per node)
;   * levels   : capacity i32 (node top level)
;   * deg0     : capacity i32 (layer-0 degree)
;   * links0   : capacity*(m0+1) i32 — layer-0 adjacency, the HOT layer, one
;     flat contiguous array indexed by node*(m0+1). +1 slot absorbs the
;     transient over-append before a prune. Layer 0 is dense (every node) so it
;     is a flat array, never a per-node malloc.
;   * upper    : capacity ptr — per-node block for layers 1..level (RARE: ~1/m
;     of nodes reach level>=1), or null. Block i32 layout for a level-L node:
;     [deg_1..deg_L] then L neighbour runs of (m+1) i32 each.
;   * visited  : capacity i64, generation-stamped — a query stamps visited[v]
;     with a per-search generation counter, so a query NEVER pays an O(N) reset.
;   * scratch heaps (frontier min / result max, both capacity i64 packed
;     entries), cand/eps/sel/kept working buffers, qnorm — all owned by the
;     handle and grown with it, so search/insert allocate NOTHING per call.
; A packed heap entry is i64: low 32 bits = f32 distance (bit pattern), high 32
; bits = node id. One i64 load/store moves an entry; compare via the low float.
;
; CONCURRENCY: single-threaded v1 (no atomics — every field is a plain load/
; store). The generation-stamped visited set and handle-owned scratch make a
; query read-only w.r.t. the graph, so a future concurrent variant can add a
; shared read lock + per-thread scratch without restructuring. Justified: no
; atomic is used because there is no cross-thread sharing in this version.
;
; API (metric: 0=cosine, 1=l2; error codes 0 OK,1 NULL,2 OOM,3 OVERFLOW,8 INVAL):
;   ptr  universe_ml_hnsw_create(i64 dims, i32 metric, i64 m, i64 ef_construction)
;   i32  universe_ml_hnsw_insert(ptr h, ptr vec_f32, i64 label)
;   i32  universe_ml_hnsw_search(ptr h, ptr qvec, i64 k, i64 ef_search,
;                                ptr out_labels /*i64*/, ptr out_dists /*f32*/,
;                                ptr out_n /*i64*/)
;   i64  universe_ml_hnsw_len(ptr h)
;   void universe_ml_hnsw_destroy(ptr h)

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare ptr @realloc(ptr, i64)
declare void @free(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare double @llvm.log.f64(double)
declare double @llvm.floor.f64(double)
declare float @llvm.sqrt.f32(float)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ============================================================ packed entry ops
define internal float @hnsw_edist(i64 %e) #2 {
  %lo = trunc i64 %e to i32
  %f = bitcast i32 %lo to float
  ret float %f
}
define internal i32 @hnsw_eid(i64 %e) #2 {
  %hi = lshr i64 %e, 32
  %id = trunc i64 %hi to i32
  ret i32 %id
}
define internal i64 @hnsw_pack(float %d, i32 %id) #2 {
  %db = bitcast float %d to i32
  %dz = zext i32 %db to i64
  %iz = zext i32 %id to i64
  %ih = shl i64 %iz, 32
  %e = or i64 %ih, %dz
  ret i64 %e
}

; ============================================================= distance kernel
; squared-L2, 4-accumulator <4 x float> (mirrors src/ml/neighbors.ll knn_dist2).
define internal float @hnsw_dist2(ptr readonly %a, ptr readonly %b, i64 %n) #2 {
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

; ================================================================ field access
define internal ptr @hnsw_vec(ptr %h, i64 %node) #2 {
  %vp = getelementptr inbounds nuw i8, ptr %h, i64 96
  %vec = load ptr, ptr %vp, align 8
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %off = mul i64 %node, %dims
  %r = getelementptr inbounds float, ptr %vec, i64 %off
  ret ptr %r
}

; Returns {neigh_ptr, deg_ptr, max_conn} for (node, layer).
define internal { ptr, ptr, i64 } @hnsw_layer(ptr %h, i64 %node, i64 %layer) #2 {
entry:
  %mp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %m = load i64, ptr %mp, align 8
  %m0p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %m0 = load i64, ptr %m0p, align 8
  %is0 = icmp eq i64 %layer, 0
  br i1 %is0, label %l0, label %lu
l0:
  %deg0p = getelementptr inbounds nuw i8, ptr %h, i64 120
  %deg0 = load ptr, ptr %deg0p, align 8
  %degp0 = getelementptr inbounds nuw i32, ptr %deg0, i64 %node
  %links0p = getelementptr inbounds nuw i8, ptr %h, i64 128
  %links0 = load ptr, ptr %links0p, align 8
  %cap0 = add i64 %m0, 1
  %noff0 = mul i64 %node, %cap0
  %neigh0 = getelementptr inbounds i32, ptr %links0, i64 %noff0
  %r0a = insertvalue { ptr, ptr, i64 } undef, ptr %neigh0, 0
  %r0b = insertvalue { ptr, ptr, i64 } %r0a, ptr %degp0, 1
  %r0c = insertvalue { ptr, ptr, i64 } %r0b, i64 %m0, 2
  ret { ptr, ptr, i64 } %r0c
lu:
  %upperp = getelementptr inbounds nuw i8, ptr %h, i64 136
  %upper = load ptr, ptr %upperp, align 8
  %blkpp = getelementptr inbounds nuw ptr, ptr %upper, i64 %node
  %blk = load ptr, ptr %blkpp, align 8
  %levelsp = getelementptr inbounds nuw i8, ptr %h, i64 112
  %levels = load ptr, ptr %levelsp, align 8
  %lvp = getelementptr inbounds nuw i32, ptr %levels, i64 %node
  %lv32 = load i32, ptr %lvp, align 4
  %L = sext i32 %lv32 to i64
  %lm1 = sub i64 %layer, 1
  %degpu = getelementptr inbounds i32, ptr %blk, i64 %lm1
  %capu = add i64 %m, 1
  %run = mul i64 %lm1, %capu
  %noffu = add i64 %L, %run
  %neighu = getelementptr inbounds i32, ptr %blk, i64 %noffu
  %rua = insertvalue { ptr, ptr, i64 } undef, ptr %neighu, 0
  %rub = insertvalue { ptr, ptr, i64 } %rua, ptr %degpu, 1
  %ruc = insertvalue { ptr, ptr, i64 } %rub, i64 %m, 2
  ret { ptr, ptr, i64 } %ruc
}

; ==================================================================== heap sift
define internal void @hnsw_sift_up(ptr %base, i64 %i, i1 %ismax) #2 {
entry:
  br label %head
head:
  %ii = phi i64 [ %i, %entry ], [ %par, %swap ]
  %z = icmp eq i64 %ii, 0
  br i1 %z, label %ret, label %chk
chk:
  %im1 = sub i64 %ii, 1
  %par = udiv i64 %im1, 2
  %eip = getelementptr inbounds i64, ptr %base, i64 %ii
  %ei = load i64, ptr %eip, align 8
  %epp = getelementptr inbounds i64, ptr %base, i64 %par
  %ep = load i64, ptr %epp, align 8
  %di = call float @hnsw_edist(i64 %ei)
  %dp = call float @hnsw_edist(i64 %ep)
  %ltmin = fcmp olt float %di, %dp
  %ltmax = fcmp olt float %dp, %di
  %cond = select i1 %ismax, i1 %ltmax, i1 %ltmin
  br i1 %cond, label %swap, label %ret
swap:
  store i64 %ep, ptr %eip, align 8
  store i64 %ei, ptr %epp, align 8
  br label %head
ret:
  ret void
}

define internal void @hnsw_sift_down(ptr %base, i64 %size, i64 %i, i1 %ismax) #2 {
entry:
  br label %head
head:
  %ii = phi i64 [ %i, %entry ], [ %bestf, %swap ]
  %l = add i64 %ii, %ii
  %l1 = add i64 %l, 1
  %r1 = add i64 %l, 2
  %eip = getelementptr inbounds i64, ptr %base, i64 %ii
  %ei = load i64, ptr %eip, align 8
  %di = call float @hnsw_edist(i64 %ei)
  %lok = icmp ult i64 %l1, %size
  br i1 %lok, label %lload, label %lskip
lload:
  %elp = getelementptr inbounds i64, ptr %base, i64 %l1
  %el = load i64, ptr %elp, align 8
  %dl = call float @hnsw_edist(i64 %el)
  %prlmin = fcmp olt float %dl, %di
  %prlmax = fcmp olt float %di, %dl
  %prl = select i1 %ismax, i1 %prlmax, i1 %prlmin
  %bl = select i1 %prl, i64 %l1, i64 %ii
  %bdl = select i1 %prl, float %dl, float %di
  br label %rcheck
lskip:
  br label %rcheck
rcheck:
  %best1 = phi i64 [ %bl, %lload ], [ %ii, %lskip ]
  %bd1 = phi float [ %bdl, %lload ], [ %di, %lskip ]
  %rok = icmp ult i64 %r1, %size
  br i1 %rok, label %rload, label %rskip
rload:
  %erp = getelementptr inbounds i64, ptr %base, i64 %r1
  %er = load i64, ptr %erp, align 8
  %dr = call float @hnsw_edist(i64 %er)
  %prrmin = fcmp olt float %dr, %bd1
  %prrmax = fcmp olt float %bd1, %dr
  %prr = select i1 %ismax, i1 %prrmax, i1 %prrmin
  %br2 = select i1 %prr, i64 %r1, i64 %best1
  br label %decide
rskip:
  br label %decide
decide:
  %bestf = phi i64 [ %br2, %rload ], [ %best1, %rskip ]
  %same = icmp eq i64 %bestf, %ii
  br i1 %same, label %ret, label %swap
swap:
  %ebp = getelementptr inbounds i64, ptr %base, i64 %bestf
  %eb = load i64, ptr %ebp, align 8
  store i64 %eb, ptr %eip, align 8
  store i64 %ei, ptr %ebp, align 8
  br label %head
ret:
  ret void
}

; ======================================================== search one layer
; Populates the handle's result max-heap with the <=ef nearest nodes to %q
; reachable from the %epsn entry points in %eps (i32 ids). Returns result count.
define internal i64 @hnsw_search_layer(ptr %h, ptr %q, ptr %eps, i64 %epsn,
                                       i64 %ef, i64 %layer) #1 {
entry:
  %fsz = alloca i64, align 8
  %rsz = alloca i64, align 8
  store i64 0, ptr %fsz, align 8
  store i64 0, ptr %rsz, align 8
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %frp = getelementptr inbounds nuw i8, ptr %h, i64 152
  %frontier = load ptr, ptr %frp, align 8
  %rep = getelementptr inbounds nuw i8, ptr %h, i64 160
  %result = load ptr, ptr %rep, align 8
  %vip = getelementptr inbounds nuw i8, ptr %h, i64 144
  %visited = load ptr, ptr %vip, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 112
  %levels = load ptr, ptr %levp, align 8
  %vgp = getelementptr inbounds nuw i8, ptr %h, i64 88
  %g = load i64, ptr %vgp, align 8
  %gen = add i64 %g, 1
  store i64 %gen, ptr %vgp, align 8
  br label %seed.head
seed.head:
  %sj = phi i64 [ 0, %entry ], [ %sjn, %seed.cont ]
  %sgo = icmp ult i64 %sj, %epsn
  br i1 %sgo, label %seed.body, label %main.head
seed.body:
  %sep = getelementptr inbounds i32, ptr %eps, i64 %sj
  %sid32 = load i32, ptr %sep, align 4
  %sid = sext i32 %sid32 to i64
  %svec = call ptr @hnsw_vec(ptr %h, i64 %sid)
  %sd = call float @hnsw_dist2(ptr %q, ptr %svec, i64 %dims)
  %svp = getelementptr inbounds i64, ptr %visited, i64 %sid
  store i64 %gen, ptr %svp, align 8
  %spk = call i64 @hnsw_pack(float %sd, i32 %sid32)
  %fs0 = load i64, ptr %fsz, align 8
  %fslot = getelementptr inbounds i64, ptr %frontier, i64 %fs0
  store i64 %spk, ptr %fslot, align 8
  call void @hnsw_sift_up(ptr %frontier, i64 %fs0, i1 false)
  %fs1 = add i64 %fs0, 1
  store i64 %fs1, ptr %fsz, align 8
  %rs0 = load i64, ptr %rsz, align 8
  %rslot = getelementptr inbounds i64, ptr %result, i64 %rs0
  store i64 %spk, ptr %rslot, align 8
  call void @hnsw_sift_up(ptr %result, i64 %rs0, i1 true)
  %rs1 = add i64 %rs0, 1
  store i64 %rs1, ptr %rsz, align 8
  %rovf = icmp ugt i64 %rs1, %ef
  br i1 %rovf, label %seed.trim, label %seed.cont
seed.trim:
  %rns = sub i64 %rs1, 1
  %rlast = getelementptr inbounds i64, ptr %result, i64 %rns
  %rlv = load i64, ptr %rlast, align 8
  store i64 %rlv, ptr %result, align 8
  %rnspos = icmp ugt i64 %rns, 0
  br i1 %rnspos, label %seed.trim.sd, label %seed.trim.set
seed.trim.sd:
  call void @hnsw_sift_down(ptr %result, i64 %rns, i64 0, i1 true)
  br label %seed.trim.set
seed.trim.set:
  store i64 %rns, ptr %rsz, align 8
  br label %seed.cont
seed.cont:
  %sjn = add i64 %sj, 1
  br label %seed.head
main.head:
  %fscur = load i64, ptr %fsz, align 8
  %fempty = icmp eq i64 %fscur, 0
  br i1 %fempty, label %done, label %main.pop
main.pop:
  %croot = load i64, ptr %frontier, align 8
  %cd = call float @hnsw_edist(i64 %croot)
  %cid32 = call i32 @hnsw_eid(i64 %croot)
  %cid = sext i32 %cid32 to i64
  %fns = sub i64 %fscur, 1
  %flastp = getelementptr inbounds i64, ptr %frontier, i64 %fns
  %flv = load i64, ptr %flastp, align 8
  store i64 %flv, ptr %frontier, align 8
  store i64 %fns, ptr %fsz, align 8
  %fnspos = icmp ugt i64 %fns, 0
  br i1 %fnspos, label %main.pop.sd, label %main.worst
main.pop.sd:
  call void @hnsw_sift_down(ptr %frontier, i64 %fns, i64 0, i1 false)
  br label %main.worst
main.worst:
  %rroot = load i64, ptr %result, align 8
  %wd = call float @hnsw_edist(i64 %rroot)
  %stop = fcmp ogt float %cd, %wd
  br i1 %stop, label %done, label %expand.pre
expand.pre:
  %trip = call { ptr, ptr, i64 } @hnsw_layer(ptr %h, i64 %cid, i64 %layer)
  %neigh = extractvalue { ptr, ptr, i64 } %trip, 0
  %degp = extractvalue { ptr, ptr, i64 } %trip, 1
  %lvcp = getelementptr inbounds i32, ptr %levels, i64 %cid
  %lvc32 = load i32, ptr %lvcp, align 4
  %lvc = sext i32 %lvc32 to i64
  %haslayer = icmp uge i64 %lvc, %layer
  br i1 %haslayer, label %expand.deg, label %main.head
expand.deg:
  %deg32 = load i32, ptr %degp, align 4
  %deg = sext i32 %deg32 to i64
  br label %exp.head
exp.head:
  %t = phi i64 [ 0, %expand.deg ], [ %tn, %exp.cont ]
  %tgo = icmp ult i64 %t, %deg
  br i1 %tgo, label %exp.body, label %main.head
exp.body:
  %nbp = getelementptr inbounds i32, ptr %neigh, i64 %t
  %nb32 = load i32, ptr %nbp, align 4
  %nb = sext i32 %nb32 to i64
  %nvp = getelementptr inbounds i64, ptr %visited, i64 %nb
  %nvis = load i64, ptr %nvp, align 8
  %seen = icmp eq i64 %nvis, %gen
  br i1 %seen, label %exp.cont, label %exp.new
exp.new:
  store i64 %gen, ptr %nvp, align 8
  %nvec = call ptr @hnsw_vec(ptr %h, i64 %nb)
  %nd = call float @hnsw_dist2(ptr %q, ptr %nvec, i64 %dims)
  %rscur = load i64, ptr %rsz, align 8
  %rroot2 = load i64, ptr %result, align 8
  %wd2 = call float @hnsw_edist(i64 %rroot2)
  %notfull = icmp ult i64 %rscur, %ef
  %closer = fcmp olt float %nd, %wd2
  %keep = or i1 %notfull, %closer
  br i1 %keep, label %exp.push, label %exp.cont
exp.push:
  %npk = call i64 @hnsw_pack(float %nd, i32 %nb32)
  %efs = load i64, ptr %fsz, align 8
  %efslot = getelementptr inbounds i64, ptr %frontier, i64 %efs
  store i64 %npk, ptr %efslot, align 8
  call void @hnsw_sift_up(ptr %frontier, i64 %efs, i1 false)
  %efs1 = add i64 %efs, 1
  store i64 %efs1, ptr %fsz, align 8
  %ers = load i64, ptr %rsz, align 8
  %erslot = getelementptr inbounds i64, ptr %result, i64 %ers
  store i64 %npk, ptr %erslot, align 8
  call void @hnsw_sift_up(ptr %result, i64 %ers, i1 true)
  %ers1 = add i64 %ers, 1
  store i64 %ers1, ptr %rsz, align 8
  %erovf = icmp ugt i64 %ers1, %ef
  br i1 %erovf, label %exp.trim, label %exp.cont
exp.trim:
  %erns = sub i64 %ers1, 1
  %erlast = getelementptr inbounds i64, ptr %result, i64 %erns
  %erlv = load i64, ptr %erlast, align 8
  store i64 %erlv, ptr %result, align 8
  %ernspos = icmp ugt i64 %erns, 0
  br i1 %ernspos, label %exp.trim.sd, label %exp.trim.set
exp.trim.sd:
  call void @hnsw_sift_down(ptr %result, i64 %erns, i64 0, i1 true)
  br label %exp.trim.set
exp.trim.set:
  store i64 %erns, ptr %rsz, align 8
  br label %exp.cont
exp.cont:
  %tn = add i64 %t, 1
  br label %exp.head
done:
  %rfin = load i64, ptr %rsz, align 8
  ret i64 %rfin
}

; ================================================== nearest id in result heap
define internal i32 @hnsw_nearest(ptr %base, i64 %n) #1 {
entry:
  %e0 = load i64, ptr %base, align 8
  %d0 = call float @hnsw_edist(i64 %e0)
  %id0 = call i32 @hnsw_eid(i64 %e0)
  br label %head
head:
  %i = phi i64 [ 1, %entry ], [ %in, %cont ]
  %bd = phi float [ %d0, %entry ], [ %bdn, %cont ]
  %bid = phi i32 [ %id0, %entry ], [ %bidn, %cont ]
  %go = icmp ult i64 %i, %n
  br i1 %go, label %body, label %ret
body:
  %ep = getelementptr inbounds i64, ptr %base, i64 %i
  %e = load i64, ptr %ep, align 8
  %d = call float @hnsw_edist(i64 %e)
  %id = call i32 @hnsw_eid(i64 %e)
  %lt = fcmp olt float %d, %bd
  %bdn = select i1 %lt, float %d, float %bd
  %bidn = select i1 %lt, i32 %id, i32 %bid
  br label %cont
cont:
  %in = add i64 %i, 1
  br label %head
ret:
  ret i32 %bid
}

; ============================================== insertion sort packed asc
define internal void @hnsw_sort_asc(ptr %base, i64 %n) #1 {
entry:
  %small = icmp ult i64 %n, 2
  br i1 %small, label %ret, label %outer
outer:
  %i = phi i64 [ 1, %entry ], [ %in, %place ]
  %keyp = getelementptr inbounds i64, ptr %base, i64 %i
  %key = load i64, ptr %keyp, align 8
  %kd = call float @hnsw_edist(i64 %key)
  %jm1 = sub i64 %i, 1
  br label %inner
inner:
  %j = phi i64 [ %jm1, %outer ], [ %jn, %shift ]
  %js = add i64 %j, 1
  %jok = icmp sge i64 %j, 0
  br i1 %jok, label %cmp, label %place
cmp:
  %ajp = getelementptr inbounds i64, ptr %base, i64 %j
  %aj = load i64, ptr %ajp, align 8
  %ad = call float @hnsw_edist(i64 %aj)
  %gt = fcmp ogt float %ad, %kd
  br i1 %gt, label %shift, label %place
shift:
  %ajp1 = getelementptr inbounds i64, ptr %base, i64 %js
  store i64 %aj, ptr %ajp1, align 8
  %jn = sub i64 %j, 1
  br label %inner
place:
  %pp = getelementptr inbounds i64, ptr %base, i64 %js
  store i64 %key, ptr %pp, align 8
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %outer, label %ret
ret:
  ret void
}

; ===================================== neighbour-selection heuristic (Alg. 4)
; cand: packed candidates sorted ascending by distance; writes kept ids into
; handle kept buffer; returns kept count (<= M).
define internal i64 @hnsw_select(ptr %h, ptr %cand, i64 %wc, i64 %M) #1 {
entry:
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %kp = getelementptr inbounds nuw i8, ptr %h, i64 192
  %kept = load ptr, ptr %kp, align 8
  %kc = alloca i64, align 8
  store i64 0, ptr %kc, align 8
  br label %ohead
ohead:
  %i = phi i64 [ 0, %entry ], [ %in, %ocont ]
  %kcur = load i64, ptr %kc, align 8
  %full = icmp uge i64 %kcur, %M
  %igo = icmp ult i64 %i, %wc
  br i1 %full, label %ret, label %ochk
ochk:
  br i1 %igo, label %obody, label %ret
obody:
  %cp = getelementptr inbounds i64, ptr %cand, i64 %i
  %ce = load i64, ptr %cp, align 8
  %cd = call float @hnsw_edist(i64 %ce)
  %cid32 = call i32 @hnsw_eid(i64 %ce)
  %cid = sext i32 %cid32 to i64
  %cvec = call ptr @hnsw_vec(ptr %h, i64 %cid)
  br label %ihead
ihead:
  %j = phi i64 [ 0, %obody ], [ %jn, %icont ]
  %jgo = icmp ult i64 %j, %kcur
  br i1 %jgo, label %ibody, label %good
ibody:
  %rp = getelementptr inbounds i32, ptr %kept, i64 %j
  %rid32 = load i32, ptr %rp, align 4
  %rid = sext i32 %rid32 to i64
  %rvec = call ptr @hnsw_vec(ptr %h, i64 %rid)
  %dd = call float @hnsw_dist2(ptr %cvec, ptr %rvec, i64 %dims)
  %bad = fcmp olt float %dd, %cd
  br i1 %bad, label %ocont, label %icont
icont:
  %jn = add i64 %j, 1
  br label %ihead
good:
  %slot = getelementptr inbounds i32, ptr %kept, i64 %kcur
  store i32 %cid32, ptr %slot, align 4
  %kc1 = add i64 %kcur, 1
  store i64 %kc1, ptr %kc, align 8
  br label %ocont
ocont:
  %in = add i64 %i, 1
  br label %ohead
ret:
  %kfin = load i64, ptr %kc, align 8
  ret i64 %kfin
}

; ===================================== prune node's layer down to max_conn
define internal void @hnsw_prune(ptr %h, i64 %node, i64 %layer, i64 %maxc) #1 {
entry:
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %candp = getelementptr inbounds nuw i8, ptr %h, i64 168
  %cand = load ptr, ptr %candp, align 8
  %trip = call { ptr, ptr, i64 } @hnsw_layer(ptr %h, i64 %node, i64 %layer)
  %neigh = extractvalue { ptr, ptr, i64 } %trip, 0
  %degp = extractvalue { ptr, ptr, i64 } %trip, 1
  %deg32 = load i32, ptr %degp, align 4
  %deg = sext i32 %deg32 to i64
  %nvec = call ptr @hnsw_vec(ptr %h, i64 %node)
  br label %bhead
bhead:
  %i = phi i64 [ 0, %entry ], [ %in, %bbody ]
  %bgo = icmp ult i64 %i, %deg
  br i1 %bgo, label %bbody, label %bdone
bbody:
  %nbp = getelementptr inbounds i32, ptr %neigh, i64 %i
  %nb32 = load i32, ptr %nbp, align 4
  %nb = sext i32 %nb32 to i64
  %nbvec = call ptr @hnsw_vec(ptr %h, i64 %nb)
  %d = call float @hnsw_dist2(ptr %nvec, ptr %nbvec, i64 %dims)
  %pk = call i64 @hnsw_pack(float %d, i32 %nb32)
  %cslot = getelementptr inbounds i64, ptr %cand, i64 %i
  store i64 %pk, ptr %cslot, align 8
  %in = add i64 %i, 1
  br label %bhead
bdone:
  call void @hnsw_sort_asc(ptr %cand, i64 %deg)
  %kc = call i64 @hnsw_select(ptr %h, ptr %cand, i64 %deg, i64 %maxc)
  %kp = getelementptr inbounds nuw i8, ptr %h, i64 192
  %kept = load ptr, ptr %kp, align 8
  br label %whead
whead:
  %w = phi i64 [ 0, %bdone ], [ %wn, %wbody ]
  %wgo = icmp ult i64 %w, %kc
  br i1 %wgo, label %wbody, label %wdone
wbody:
  %ksp = getelementptr inbounds i32, ptr %kept, i64 %w
  %kv = load i32, ptr %ksp, align 4
  %ndst = getelementptr inbounds i32, ptr %neigh, i64 %w
  store i32 %kv, ptr %ndst, align 4
  %wn = add i64 %w, 1
  br label %whead
wdone:
  %kc32 = trunc i64 %kc to i32
  store i32 %kc32, ptr %degp, align 4
  ret void
}

; ===================================== append edge a->b at layer, prune if over
define internal void @hnsw_add_edge(ptr %h, i64 %a, i64 %b, i64 %layer, i64 %maxc) #1 {
entry:
  %trip = call { ptr, ptr, i64 } @hnsw_layer(ptr %h, i64 %a, i64 %layer)
  %neigh = extractvalue { ptr, ptr, i64 } %trip, 0
  %degp = extractvalue { ptr, ptr, i64 } %trip, 1
  %deg32 = load i32, ptr %degp, align 4
  %deg = sext i32 %deg32 to i64
  %slot = getelementptr inbounds i32, ptr %neigh, i64 %deg
  %b32 = trunc i64 %b to i32
  store i32 %b32, ptr %slot, align 4
  %deg1 = add i64 %deg, 1
  %deg1_32 = trunc i64 %deg1 to i32
  store i32 %deg1_32, ptr %degp, align 4
  %over = icmp ugt i64 %deg1, %maxc
  br i1 %over, label %prune, label %ret
prune:
  call void @hnsw_prune(ptr %h, i64 %a, i64 %layer, i64 %maxc)
  br label %ret
ret:
  ret void
}

; ===================================================== rng + random level
define internal i64 @hnsw_xorshift(ptr %h) #1 {
entry:
  %sp = getelementptr inbounds nuw i8, ptr %h, i64 48
  %x0 = load i64, ptr %sp, align 8
  %s13 = shl i64 %x0, 13
  %x1 = xor i64 %x0, %s13
  %s7 = lshr i64 %x1, 7
  %x2 = xor i64 %x1, %s7
  %s17 = shl i64 %x2, 17
  %x3 = xor i64 %x2, %s17
  store i64 %x3, ptr %sp, align 8
  ret i64 %x3
}

define internal i64 @hnsw_rand_level(ptr %h) #1 {
entry:
  %x = call i64 @hnsw_xorshift(ptr %h)
  %u53 = lshr i64 %x, 11
  %uf = uitofp i64 %u53 to double
  %r0 = fmul double %uf, 0x3CA0000000000000
  %iszero = fcmp ole double %r0, 0.0
  %r = select i1 %iszero, double 0x3CA0000000000000, double %r0
  %ln = call double @llvm.log.f64(double %r)
  %nln = fneg double %ln
  %mlp = getelementptr inbounds nuw i8, ptr %h, i64 40
  %ml = load double, ptr %mlp, align 8
  %scaled = fmul double %nln, %ml
  %fl = call double @llvm.floor.f64(double %scaled)
  %lvl = fptosi double %fl to i64
  %neg = icmp slt i64 %lvl, 0
  %lvl0 = select i1 %neg, i64 0, i64 %lvl
  %cap = icmp ugt i64 %lvl0, 31
  %lvlc = select i1 %cap, i64 31, i64 %lvl0
  ret i64 %lvlc
}

; ===================================================== store / normalize vector
define internal void @hnsw_store_vec(ptr %h, i64 %nid, ptr %src) #1 {
entry:
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %mp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %mp, align 4
  %dest = call ptr @hnsw_vec(ptr %h, i64 %nid)
  %isl2 = icmp eq i32 %metric, 1
  br i1 %isl2, label %copy, label %norm
copy:
  %bytes = shl i64 %dims, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %src, i64 %bytes, i1 false)
  ret void
norm:
  br label %shead
shead:
  %i = phi i64 [ 0, %norm ], [ %in, %sbody ]
  %acc = phi float [ 0.0, %norm ], [ %accn, %sbody ]
  %go = icmp ult i64 %i, %dims
  br i1 %go, label %sbody, label %sdone
sbody:
  %xp = getelementptr inbounds float, ptr %src, i64 %i
  %x = load float, ptr %xp, align 4
  %sq = fmul float %x, %x
  %accn = fadd float %acc, %sq
  %in = add i64 %i, 1
  br label %shead
sdone:
  %pos = fcmp ogt float %acc, 0.0
  %root = call float @llvm.sqrt.f32(float %acc)
  %invr = fdiv float 1.0, %root
  %inv = select i1 %pos, float %invr, float 0.0
  br label %whead
whead:
  %j = phi i64 [ 0, %sdone ], [ %jn, %wbody ]
  %wgo = icmp ult i64 %j, %dims
  br i1 %wgo, label %wbody, label %ret
wbody:
  %sxp = getelementptr inbounds float, ptr %src, i64 %j
  %sx = load float, ptr %sxp, align 4
  %nx = fmul float %sx, %inv
  %dxp = getelementptr inbounds float, ptr %dest, i64 %j
  store float %nx, ptr %dxp, align 4
  %jn = add i64 %j, 1
  br label %whead
ret:
  ret void
}

; Normalize (or copy) query into handle qnorm buffer; returns qnorm ptr.
define internal ptr @hnsw_prep_query(ptr %h, ptr %src) #1 {
entry:
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %mp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %mp, align 4
  %qp = getelementptr inbounds nuw i8, ptr %h, i64 200
  %qn = load ptr, ptr %qp, align 8
  %isl2 = icmp eq i32 %metric, 1
  br i1 %isl2, label %copy, label %norm
copy:
  %bytes = shl i64 %dims, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %qn, ptr %src, i64 %bytes, i1 false)
  ret ptr %qn
norm:
  br label %shead
shead:
  %i = phi i64 [ 0, %norm ], [ %in, %sbody ]
  %acc = phi float [ 0.0, %norm ], [ %accn, %sbody ]
  %go = icmp ult i64 %i, %dims
  br i1 %go, label %sbody, label %sdone
sbody:
  %xp = getelementptr inbounds float, ptr %src, i64 %i
  %x = load float, ptr %xp, align 4
  %sq = fmul float %x, %x
  %accn = fadd float %acc, %sq
  %in = add i64 %i, 1
  br label %shead
sdone:
  %pos = fcmp ogt float %acc, 0.0
  %root = call float @llvm.sqrt.f32(float %acc)
  %invr = fdiv float 1.0, %root
  %inv = select i1 %pos, float %invr, float 0.0
  br label %whead
whead:
  %j = phi i64 [ 0, %sdone ], [ %jn, %wbody ]
  %wgo = icmp ult i64 %j, %dims
  br i1 %wgo, label %wbody, label %ret
wbody:
  %sxp = getelementptr inbounds float, ptr %src, i64 %j
  %sx = load float, ptr %sxp, align 4
  %nx = fmul float %sx, %inv
  %dxp = getelementptr inbounds float, ptr %qn, i64 %j
  store float %nx, ptr %dxp, align 4
  %jn = add i64 %j, 1
  br label %whead
ret:
  ret ptr %qn
}

; ===================================================== grow capacity by 2x
; returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW.
define internal i32 @hnsw_grow(ptr %h) #1 {
entry:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cnt = load i64, ptr %cntp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %h, i64 64
  %cap = load i64, ptr %capp, align 8
  %need = icmp uge i64 %cnt, %cap
  br i1 %need, label %grow, label %okret
grow:
  %newcap = shl i64 %cap, 1
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  %dims = load i64, ptr %dp, align 8
  %m0p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %m0 = load i64, ptr %m0p, align 8
  %cap0 = add i64 %m0, 1
  %vm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %newcap, i64 %dims)
  %vprod = extractvalue { i64, i1 } %vm, 0
  %vovf = extractvalue { i64, i1 } %vm, 1
  %vb2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %vprod, i64 4)
  %vbytes = extractvalue { i64, i1 } %vb2, 0
  %vovf2 = extractvalue { i64, i1 } %vb2, 1
  %lm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %newcap, i64 %cap0)
  %lprod = extractvalue { i64, i1 } %lm, 0
  %lovf = extractvalue { i64, i1 } %lm, 1
  %lb2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %lprod, i64 4)
  %lbytes = extractvalue { i64, i1 } %lb2, 0
  %lovf2 = extractvalue { i64, i1 } %lb2, 1
  %o1 = or i1 %vovf, %vovf2
  %o2 = or i1 %lovf, %lovf2
  %ovf = or i1 %o1, %o2
  br i1 %ovf, label %ovfret, label %realloc
realloc:
  %n8 = shl i64 %newcap, 3
  %n4 = shl i64 %newcap, 2
  %vpp = getelementptr inbounds nuw i8, ptr %h, i64 96
  %vold = load ptr, ptr %vpp, align 8
  %vnew = call ptr @realloc(ptr %vold, i64 %vbytes)
  %vnull = icmp eq ptr %vnew, null
  br i1 %vnull, label %oomret, label %r1
r1:
  store ptr %vnew, ptr %vpp, align 8
  %lbp = getelementptr inbounds nuw i8, ptr %h, i64 104
  %lbold = load ptr, ptr %lbp, align 8
  %lbnew = call ptr @realloc(ptr %lbold, i64 %n8)
  %lbnull = icmp eq ptr %lbnew, null
  br i1 %lbnull, label %oomret, label %r2
r2:
  store ptr %lbnew, ptr %lbp, align 8
  %lvp = getelementptr inbounds nuw i8, ptr %h, i64 112
  %lvold = load ptr, ptr %lvp, align 8
  %lvnew = call ptr @realloc(ptr %lvold, i64 %n4)
  %lvnull = icmp eq ptr %lvnew, null
  br i1 %lvnull, label %oomret, label %r3
r3:
  store ptr %lvnew, ptr %lvp, align 8
  %dgp = getelementptr inbounds nuw i8, ptr %h, i64 120
  %dgold = load ptr, ptr %dgp, align 8
  %dgnew = call ptr @realloc(ptr %dgold, i64 %n4)
  %dgnull = icmp eq ptr %dgnew, null
  br i1 %dgnull, label %oomret, label %r4
r4:
  store ptr %dgnew, ptr %dgp, align 8
  %l0p = getelementptr inbounds nuw i8, ptr %h, i64 128
  %l0old = load ptr, ptr %l0p, align 8
  %l0new = call ptr @realloc(ptr %l0old, i64 %lbytes)
  %l0null = icmp eq ptr %l0new, null
  br i1 %l0null, label %oomret, label %r5
r5:
  store ptr %l0new, ptr %l0p, align 8
  %upp = getelementptr inbounds nuw i8, ptr %h, i64 136
  %upold = load ptr, ptr %upp, align 8
  %upnew = call ptr @realloc(ptr %upold, i64 %n8)
  %upnull = icmp eq ptr %upnew, null
  br i1 %upnull, label %oomret, label %r6
r6:
  store ptr %upnew, ptr %upp, align 8
  %vsp = getelementptr inbounds nuw i8, ptr %h, i64 144
  %vsold = load ptr, ptr %vsp, align 8
  %vsnew = call ptr @realloc(ptr %vsold, i64 %n8)
  %vsnull = icmp eq ptr %vsnew, null
  br i1 %vsnull, label %oomret, label %r7
r7:
  store ptr %vsnew, ptr %vsp, align 8
  %frp = getelementptr inbounds nuw i8, ptr %h, i64 152
  %frold = load ptr, ptr %frp, align 8
  %frnew = call ptr @realloc(ptr %frold, i64 %n8)
  %frnull = icmp eq ptr %frnew, null
  br i1 %frnull, label %oomret, label %r8
r8:
  store ptr %frnew, ptr %frp, align 8
  %rep = getelementptr inbounds nuw i8, ptr %h, i64 160
  %reold = load ptr, ptr %rep, align 8
  %renew = call ptr @realloc(ptr %reold, i64 %n8)
  %renull = icmp eq ptr %renew, null
  br i1 %renull, label %oomret, label %r9
r9:
  store ptr %renew, ptr %rep, align 8
  %cndp = getelementptr inbounds nuw i8, ptr %h, i64 168
  %cndold = load ptr, ptr %cndp, align 8
  %cndnew = call ptr @realloc(ptr %cndold, i64 %n8)
  %cndnull = icmp eq ptr %cndnew, null
  br i1 %cndnull, label %oomret, label %r10
r10:
  store ptr %cndnew, ptr %cndp, align 8
  %epp = getelementptr inbounds nuw i8, ptr %h, i64 176
  %epold = load ptr, ptr %epp, align 8
  %epnew = call ptr @realloc(ptr %epold, i64 %n4)
  %epnull = icmp eq ptr %epnew, null
  br i1 %epnull, label %oomret, label %r11
r11:
  store ptr %epnew, ptr %epp, align 8
  %zcnt = sub i64 %newcap, %cap
  %zb8 = shl i64 %zcnt, 3
  %upnewbase = getelementptr inbounds ptr, ptr %upnew, i64 %cap
  call void @llvm.memset.p0.i64(ptr %upnewbase, i8 0, i64 %zb8, i1 false)
  %vsnewbase = getelementptr inbounds i64, ptr %vsnew, i64 %cap
  call void @llvm.memset.p0.i64(ptr %vsnewbase, i8 0, i64 %zb8, i1 false)
  store i64 %newcap, ptr %capp, align 8
  br label %okret
okret:
  ret i32 0
oomret:
  ret i32 2
ovfret:
  ret i32 3
}

; ============================================================= create
define ptr @universe_ml_hnsw_create(i64 %dims, i32 %metric, i64 %m, i64 %efc) #0 {
entry:
  %baddim = icmp eq i64 %dims, 0
  %badmet0 = icmp slt i32 %metric, 0
  %badmet1 = icmp sgt i32 %metric, 1
  %badmet = or i1 %badmet0, %badmet1
  %bad = or i1 %baddim, %badmet
  br i1 %bad, label %fail0, label %clamp
clamp:
  %mlt = icmp ult i64 %m, 2
  %mc = select i1 %mlt, i64 2, i64 %m
  %m0 = shl i64 %mc, 1
  %eflt = icmp ult i64 %efc, %mc
  %efcv = select i1 %eflt, i64 %mc, i64 %efc
  %cap0 = add i64 %m0, 1
  %vm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 16, i64 %dims)
  %vprod = extractvalue { i64, i1 } %vm, 0
  %vovf = extractvalue { i64, i1 } %vm, 1
  %vbytes = shl i64 %vprod, 2
  %lm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 16, i64 %cap0)
  %lprod = extractvalue { i64, i1 } %lm, 0
  %lovf = extractvalue { i64, i1 } %lm, 1
  %lbytes = shl i64 %lprod, 2
  %ovf = or i1 %vovf, %lovf
  br i1 %ovf, label %fail0, label %alloc
alloc:
  %h = call ptr @calloc(i64 1, i64 208)
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail0, label %fields
fields:
  %dp = getelementptr inbounds nuw i8, ptr %h, i64 0
  store i64 %dims, ptr %dp, align 8
  %mep = getelementptr inbounds nuw i8, ptr %h, i64 8
  store i32 %metric, ptr %mep, align 4
  %mp = getelementptr inbounds nuw i8, ptr %h, i64 16
  store i64 %mc, ptr %mp, align 8
  %m0p = getelementptr inbounds nuw i8, ptr %h, i64 24
  store i64 %m0, ptr %m0p, align 8
  %efp = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 %efcv, ptr %efp, align 8
  %mf = uitofp i64 %mc to double
  %lnm = call double @llvm.log.f64(double %mf)
  %ml = fdiv double 1.0, %lnm
  %mlp = getelementptr inbounds nuw i8, ptr %h, i64 40
  store double %ml, ptr %mlp, align 8
  %rp = getelementptr inbounds nuw i8, ptr %h, i64 48
  store i64 -7046029254386353131, ptr %rp, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  store i64 0, ptr %cntp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %h, i64 64
  store i64 16, ptr %capp, align 8
  %enp = getelementptr inbounds nuw i8, ptr %h, i64 72
  store i64 -1, ptr %enp, align 8
  %mlvp = getelementptr inbounds nuw i8, ptr %h, i64 80
  store i64 0, ptr %mlvp, align 8
  %vgp = getelementptr inbounds nuw i8, ptr %h, i64 88
  store i64 0, ptr %vgp, align 8
  %vec = call ptr @malloc(i64 %vbytes)
  %vecnull = icmp eq ptr %vec, null
  br i1 %vecnull, label %fail1, label %a1
a1:
  %vpp = getelementptr inbounds nuw i8, ptr %h, i64 96
  store ptr %vec, ptr %vpp, align 8
  %lab = call ptr @malloc(i64 128)
  %labnull = icmp eq ptr %lab, null
  br i1 %labnull, label %fail1, label %a2
a2:
  %labp = getelementptr inbounds nuw i8, ptr %h, i64 104
  store ptr %lab, ptr %labp, align 8
  %lev = call ptr @malloc(i64 64)
  %levnull = icmp eq ptr %lev, null
  br i1 %levnull, label %fail1, label %a3
a3:
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 112
  store ptr %lev, ptr %levp, align 8
  %deg = call ptr @malloc(i64 64)
  %degnull = icmp eq ptr %deg, null
  br i1 %degnull, label %fail1, label %a4
a4:
  %dgp = getelementptr inbounds nuw i8, ptr %h, i64 120
  store ptr %deg, ptr %dgp, align 8
  %l0 = call ptr @malloc(i64 %lbytes)
  %l0null = icmp eq ptr %l0, null
  br i1 %l0null, label %fail1, label %a5
a5:
  %l0p = getelementptr inbounds nuw i8, ptr %h, i64 128
  store ptr %l0, ptr %l0p, align 8
  %up = call ptr @calloc(i64 16, i64 8)
  %upnull = icmp eq ptr %up, null
  br i1 %upnull, label %fail1, label %a6
a6:
  %upp = getelementptr inbounds nuw i8, ptr %h, i64 136
  store ptr %up, ptr %upp, align 8
  %vis = call ptr @calloc(i64 16, i64 8)
  %visnull = icmp eq ptr %vis, null
  br i1 %visnull, label %fail1, label %a7
a7:
  %vsp = getelementptr inbounds nuw i8, ptr %h, i64 144
  store ptr %vis, ptr %vsp, align 8
  %fr = call ptr @malloc(i64 128)
  %frnull = icmp eq ptr %fr, null
  br i1 %frnull, label %fail1, label %a8
a8:
  %frp = getelementptr inbounds nuw i8, ptr %h, i64 152
  store ptr %fr, ptr %frp, align 8
  %re = call ptr @malloc(i64 128)
  %renull = icmp eq ptr %re, null
  br i1 %renull, label %fail1, label %a9
a9:
  %rep = getelementptr inbounds nuw i8, ptr %h, i64 160
  store ptr %re, ptr %rep, align 8
  %cnd = call ptr @malloc(i64 128)
  %cndnull = icmp eq ptr %cnd, null
  br i1 %cndnull, label %fail1, label %a10
a10:
  %cndp = getelementptr inbounds nuw i8, ptr %h, i64 168
  store ptr %cnd, ptr %cndp, align 8
  %eps = call ptr @malloc(i64 64)
  %epsnull = icmp eq ptr %eps, null
  br i1 %epsnull, label %fail1, label %a11
a11:
  %epp = getelementptr inbounds nuw i8, ptr %h, i64 176
  store ptr %eps, ptr %epp, align 8
  %selb = shl i64 %cap0, 2
  %sel = call ptr @malloc(i64 %selb)
  %selnull = icmp eq ptr %sel, null
  br i1 %selnull, label %fail1, label %a12
a12:
  %slp = getelementptr inbounds nuw i8, ptr %h, i64 184
  store ptr %sel, ptr %slp, align 8
  %kept = call ptr @malloc(i64 %selb)
  %keptnull = icmp eq ptr %kept, null
  br i1 %keptnull, label %fail1, label %a13
a13:
  %kpp = getelementptr inbounds nuw i8, ptr %h, i64 192
  store ptr %kept, ptr %kpp, align 8
  %qnb = shl i64 %dims, 2
  %qn = call ptr @malloc(i64 %qnb)
  %qnnull = icmp eq ptr %qn, null
  br i1 %qnnull, label %fail1, label %a14
a14:
  %qnp = getelementptr inbounds nuw i8, ptr %h, i64 200
  store ptr %qn, ptr %qnp, align 8
  ret ptr %h
fail1:
  call void @universe_ml_hnsw_destroy(ptr %h)
  ret ptr null
fail0:
  ret ptr null
}

; ============================================================= insert
define i32 @universe_ml_hnsw_insert(ptr %h, ptr %vec, i64 %label) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %enull, label %ck2
ck2:
  %vnull = icmp eq ptr %vec, null
  br i1 %vnull, label %enull, label %grow
grow:
  %g = call i32 @hnsw_grow(ptr %h)
  %gok = icmp eq i32 %g, 0
  br i1 %gok, label %setup, label %greterr
greterr:
  ret i32 %g
setup:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %nid = load i64, ptr %cntp, align 8
  call void @hnsw_store_vec(ptr %h, i64 %nid, ptr %vec)
  %labp = getelementptr inbounds nuw i8, ptr %h, i64 104
  %labels = load ptr, ptr %labp, align 8
  %labslot = getelementptr inbounds i64, ptr %labels, i64 %nid
  store i64 %label, ptr %labslot, align 8
  %level = call i64 @hnsw_rand_level(ptr %h)
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 112
  %levels = load ptr, ptr %levp, align 8
  %levslot = getelementptr inbounds i32, ptr %levels, i64 %nid
  %level32 = trunc i64 %level to i32
  store i32 %level32, ptr %levslot, align 4
  %dgp = getelementptr inbounds nuw i8, ptr %h, i64 120
  %deg0 = load ptr, ptr %dgp, align 8
  %deg0slot = getelementptr inbounds i32, ptr %deg0, i64 %nid
  store i32 0, ptr %deg0slot, align 4
  %hasupper = icmp uge i64 %level, 1
  %mp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %m = load i64, ptr %mp, align 8
  %upp = getelementptr inbounds nuw i8, ptr %h, i64 136
  %upper = load ptr, ptr %upp, align 8
  %upslot = getelementptr inbounds ptr, ptr %upper, i64 %nid
  br i1 %hasupper, label %mkupper, label %noupper
mkupper:
  %mp2 = add i64 %m, 2
  %blkcnt = mul i64 %level, %mp2
  %blkbytes = shl i64 %blkcnt, 2
  %blk = call ptr @malloc(i64 %blkbytes)
  %blknull = icmp eq ptr %blk, null
  br i1 %blknull, label %eoom, label %mkupper2
mkupper2:
  %degbytes = shl i64 %level, 2
  call void @llvm.memset.p0.i64(ptr %blk, i8 0, i64 %degbytes, i1 false)
  store ptr %blk, ptr %upslot, align 8
  br label %bump
noupper:
  store ptr null, ptr %upslot, align 8
  br label %bump
bump:
  %nid1 = add i64 %nid, 1
  store i64 %nid1, ptr %cntp, align 8
  %enp = getelementptr inbounds nuw i8, ptr %h, i64 72
  %entry_id = load i64, ptr %enp, align 8
  %isfirst = icmp eq i64 %entry_id, -1
  br i1 %isfirst, label %first, label %connect.init
first:
  store i64 %nid, ptr %enp, align 8
  %mlvp = getelementptr inbounds nuw i8, ptr %h, i64 80
  store i64 %level, ptr %mlvp, align 8
  ret i32 0
connect.init:
  %vecnid = call ptr @hnsw_vec(ptr %h, i64 %nid)
  %epsp = getelementptr inbounds nuw i8, ptr %h, i64 176
  %eps = load ptr, ptr %epsp, align 8
  %entry32 = trunc i64 %entry_id to i32
  store i32 %entry32, ptr %eps, align 4
  %m0p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %m0 = load i64, ptr %m0p, align 8
  %efp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %efc = load i64, ptr %efp, align 8
  %mlvp2 = getelementptr inbounds nuw i8, ptr %h, i64 80
  %maxlevel = load i64, ptr %mlvp2, align 8
  %lvl1 = add i64 %level, 1
  br label %desc.head
desc.head:
  %lc = phi i64 [ %maxlevel, %connect.init ], [ %lcn, %desc.body ]
  %descgo = icmp uge i64 %lc, %lvl1
  br i1 %descgo, label %desc.body, label %conn.pre
desc.body:
  %dwc = call i64 @hnsw_search_layer(ptr %h, ptr %vecnid, ptr %eps, i64 1, i64 1, i64 %lc)
  %rep = getelementptr inbounds nuw i8, ptr %h, i64 160
  %result = load ptr, ptr %rep, align 8
  %near = call i32 @hnsw_nearest(ptr %result, i64 %dwc)
  store i32 %near, ptr %eps, align 4
  %lcn = sub i64 %lc, 1
  br label %desc.head
conn.pre:
  %lelt = icmp ult i64 %level, %maxlevel
  %lstart = select i1 %lelt, i64 %level, i64 %maxlevel
  br label %conn.head
conn.head:
  %epsn = phi i64 [ 1, %conn.pre ], [ %epsn.next, %conn.after ]
  %l = phi i64 [ %lstart, %conn.pre ], [ %ln, %conn.after ]
  %lge0 = icmp sge i64 %l, 0
  br i1 %lge0, label %conn.body, label %finish
conn.body:
  %wc = call i64 @hnsw_search_layer(ptr %h, ptr %vecnid, ptr %eps, i64 %epsn, i64 %efc, i64 %l)
  %rep2 = getelementptr inbounds nuw i8, ptr %h, i64 160
  %result2 = load ptr, ptr %rep2, align 8
  %cndp = getelementptr inbounds nuw i8, ptr %h, i64 168
  %cand = load ptr, ptr %cndp, align 8
  br label %cp.head
cp.head:
  %ci = phi i64 [ 0, %conn.body ], [ %cin, %cp.body ]
  %cgo = icmp ult i64 %ci, %wc
  br i1 %cgo, label %cp.body, label %cp.done
cp.body:
  %rsrc = getelementptr inbounds i64, ptr %result2, i64 %ci
  %rval = load i64, ptr %rsrc, align 8
  %cdst = getelementptr inbounds i64, ptr %cand, i64 %ci
  store i64 %rval, ptr %cdst, align 8
  %cin = add i64 %ci, 1
  br label %cp.head
cp.done:
  call void @hnsw_sort_asc(ptr %cand, i64 %wc)
  br label %ep.head
ep.head:
  %ei = phi i64 [ 0, %cp.done ], [ %ein, %ep.body ]
  %ego = icmp ult i64 %ei, %wc
  br i1 %ego, label %ep.body, label %ep.done
ep.body:
  %cesrc = getelementptr inbounds i64, ptr %cand, i64 %ei
  %ceval = load i64, ptr %cesrc, align 8
  %ceid = call i32 @hnsw_eid(i64 %ceval)
  %edst = getelementptr inbounds i32, ptr %eps, i64 %ei
  store i32 %ceid, ptr %edst, align 4
  %ein = add i64 %ei, 1
  br label %ep.head
ep.done:
  %isl0 = icmp eq i64 %l, 0
  %maxc = select i1 %isl0, i64 %m0, i64 %m
  %kc = call i64 @hnsw_select(ptr %h, ptr %cand, i64 %wc, i64 %m)
  %kpp = getelementptr inbounds nuw i8, ptr %h, i64 192
  %kept = load ptr, ptr %kpp, align 8
  %slp = getelementptr inbounds nuw i8, ptr %h, i64 184
  %sel = load ptr, ptr %slp, align 8
  br label %sc.head
sc.head:
  %si = phi i64 [ 0, %ep.done ], [ %sin, %sc.body ]
  %sgo = icmp ult i64 %si, %kc
  br i1 %sgo, label %sc.body, label %sc.done
sc.body:
  %ksrc = getelementptr inbounds i32, ptr %kept, i64 %si
  %kval = load i32, ptr %ksrc, align 4
  %sdst = getelementptr inbounds i32, ptr %sel, i64 %si
  store i32 %kval, ptr %sdst, align 4
  %sin = add i64 %si, 1
  br label %sc.head
sc.done:
  br label %ed.head
ed.head:
  %di = phi i64 [ 0, %sc.done ], [ %din, %ed.body ]
  %dgo = icmp ult i64 %di, %kc
  br i1 %dgo, label %ed.body, label %conn.after
ed.body:
  %esrc = getelementptr inbounds i32, ptr %sel, i64 %di
  %e32 = load i32, ptr %esrc, align 4
  %e = sext i32 %e32 to i64
  call void @hnsw_add_edge(ptr %h, i64 %nid, i64 %e, i64 %l, i64 %maxc)
  call void @hnsw_add_edge(ptr %h, i64 %e, i64 %nid, i64 %l, i64 %maxc)
  %din = add i64 %di, 1
  br label %ed.head
conn.after:
  %epsn.next = phi i64 [ %wc, %ed.head ]
  %ln = sub i64 %l, 1
  br label %conn.head
finish:
  %flvp = getelementptr inbounds nuw i8, ptr %h, i64 80
  %curmax = load i64, ptr %flvp, align 8
  %higher = icmp ugt i64 %level, %curmax
  br i1 %higher, label %newentry, label %doneins
newentry:
  store i64 %nid, ptr %enp, align 8
  store i64 %level, ptr %flvp, align 8
  br label %doneins
doneins:
  ret i32 0
enull:
  ret i32 1
eoom:
  ret i32 2
}

; ============================================================= search
define i32 @universe_ml_hnsw_search(ptr %h, ptr %qvec, i64 %k, i64 %efs,
                                    ptr %out_labels, ptr %out_dists, ptr %out_n) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  %qnull = icmp eq ptr %qvec, null
  %onnull = icmp eq ptr %out_n, null
  %b1 = or i1 %hnull, %qnull
  %bad = or i1 %b1, %onnull
  br i1 %bad, label %enull, label %ck2
ck2:
  %olnull = icmp eq ptr %out_labels, null
  %odnull = icmp eq ptr %out_dists, null
  %b2 = or i1 %olnull, %odnull
  %kpos = icmp ugt i64 %k, 0
  %needbuf = and i1 %kpos, %b2
  br i1 %needbuf, label %enull, label %begin
begin:
  store i64 0, ptr %out_n, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cnt = load i64, ptr %cntp, align 8
  %empty = icmp eq i64 %cnt, 0
  %kzero = icmp eq i64 %k, 0
  %stop = or i1 %empty, %kzero
  br i1 %stop, label %retok, label %run
run:
  %q = call ptr @hnsw_prep_query(ptr %h, ptr %qvec)
  %epsp = getelementptr inbounds nuw i8, ptr %h, i64 176
  %eps = load ptr, ptr %epsp, align 8
  %enp = getelementptr inbounds nuw i8, ptr %h, i64 72
  %entry_id = load i64, ptr %enp, align 8
  %entry32 = trunc i64 %entry_id to i32
  store i32 %entry32, ptr %eps, align 4
  %mlvp = getelementptr inbounds nuw i8, ptr %h, i64 80
  %maxlevel = load i64, ptr %mlvp, align 8
  %eflt = icmp ult i64 %efs, %k
  %ef = select i1 %eflt, i64 %k, i64 %efs
  br label %desc.head
desc.head:
  %lc = phi i64 [ %maxlevel, %run ], [ %lcn, %desc.body ]
  %go = icmp ugt i64 %lc, 0
  br i1 %go, label %desc.body, label %layer0
desc.body:
  %dwc = call i64 @hnsw_search_layer(ptr %h, ptr %q, ptr %eps, i64 1, i64 1, i64 %lc)
  %rep = getelementptr inbounds nuw i8, ptr %h, i64 160
  %result = load ptr, ptr %rep, align 8
  %near = call i32 @hnsw_nearest(ptr %result, i64 %dwc)
  store i32 %near, ptr %eps, align 4
  %lcn = sub i64 %lc, 1
  br label %desc.head
layer0:
  %wc = call i64 @hnsw_search_layer(ptr %h, ptr %q, ptr %eps, i64 1, i64 %ef, i64 0)
  %rep2 = getelementptr inbounds nuw i8, ptr %h, i64 160
  %result2 = load ptr, ptr %rep2, align 8
  %cndp = getelementptr inbounds nuw i8, ptr %h, i64 168
  %cand = load ptr, ptr %cndp, align 8
  br label %cp.head
cp.head:
  %ci = phi i64 [ 0, %layer0 ], [ %cin, %cp.body ]
  %cgo = icmp ult i64 %ci, %wc
  br i1 %cgo, label %cp.body, label %cp.done
cp.body:
  %rsrc = getelementptr inbounds i64, ptr %result2, i64 %ci
  %rval = load i64, ptr %rsrc, align 8
  %cdst = getelementptr inbounds i64, ptr %cand, i64 %ci
  store i64 %rval, ptr %cdst, align 8
  %cin = add i64 %ci, 1
  br label %cp.head
cp.done:
  %klt = icmp ult i64 %k, %wc
  %kk = select i1 %klt, i64 %k, i64 %wc
  %labp = getelementptr inbounds nuw i8, ptr %h, i64 104
  %labels = load ptr, ptr %labp, align 8
  %metp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %metric = load i32, ptr %metp, align 4
  %iscos = icmp eq i32 %metric, 0
  br label %sel.head
sel.head:
  %i = phi i64 [ 0, %cp.done ], [ %in, %sel.emit ]
  %selgo = icmp ult i64 %i, %kk
  br i1 %selgo, label %sel.find0, label %retok
sel.find0:
  %ei0 = getelementptr inbounds i64, ptr %cand, i64 %i
  %ev0 = load i64, ptr %ei0, align 8
  %ed0 = call float @hnsw_edist(i64 %ev0)
  %i1 = add i64 %i, 1
  br label %sel.fhead
sel.fhead:
  %j = phi i64 [ %i1, %sel.find0 ], [ %jn, %sel.fcont ]
  %bmin = phi float [ %ed0, %sel.find0 ], [ %bminn, %sel.fcont ]
  %bidx = phi i64 [ %i, %sel.find0 ], [ %bidxn, %sel.fcont ]
  %fgo = icmp ult i64 %j, %wc
  br i1 %fgo, label %sel.fbody, label %sel.swap
sel.fbody:
  %ejp = getelementptr inbounds i64, ptr %cand, i64 %j
  %ejv = load i64, ptr %ejp, align 8
  %ejd = call float @hnsw_edist(i64 %ejv)
  %lt = fcmp olt float %ejd, %bmin
  %bminn = select i1 %lt, float %ejd, float %bmin
  %bidxn = select i1 %lt, i64 %j, i64 %bidx
  br label %sel.fcont
sel.fcont:
  %jn = add i64 %j, 1
  br label %sel.fhead
sel.swap:
  %cip = getelementptr inbounds i64, ptr %cand, i64 %i
  %civ = load i64, ptr %cip, align 8
  %cbp = getelementptr inbounds i64, ptr %cand, i64 %bidx
  %cbv = load i64, ptr %cbp, align 8
  store i64 %cbv, ptr %cip, align 8
  store i64 %civ, ptr %cbp, align 8
  br label %sel.emit
sel.emit:
  %ev = load i64, ptr %cip, align 8
  %eid32 = call i32 @hnsw_eid(i64 %ev)
  %eid = sext i32 %eid32 to i64
  %edist = call float @hnsw_edist(i64 %ev)
  %lsrc = getelementptr inbounds i64, ptr %labels, i64 %eid
  %lval = load i64, ptr %lsrc, align 8
  %ldst = getelementptr inbounds i64, ptr %out_labels, i64 %i
  store i64 %lval, ptr %ldst, align 8
  %half = fmul float %edist, 5.000000e-01
  %odist = select i1 %iscos, float %half, float %edist
  %ddst = getelementptr inbounds float, ptr %out_dists, i64 %i
  store float %odist, ptr %ddst, align 4
  %in = add i64 %i, 1
  br label %sel.head
retok:
  %kkfinal = phi i64 [ 0, %begin ], [ %kk, %sel.head ]
  store i64 %kkfinal, ptr %out_n, align 8
  ret i32 0
enull:
  ret i32 1
}

; ============================================================= len / destroy
define i64 @universe_ml_hnsw_len(ptr %h) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %zero, label %get
get:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cnt = load i64, ptr %cntp, align 8
  ret i64 %cnt
zero:
  ret i64 0
}

define void @universe_ml_hnsw_destroy(ptr %h) #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %ret, label %freeupper
freeupper:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %cnt = load i64, ptr %cntp, align 8
  %upp = getelementptr inbounds nuw i8, ptr %h, i64 136
  %upper = load ptr, ptr %upp, align 8
  %upnull = icmp eq ptr %upper, null
  br i1 %upnull, label %freearrays, label %uphead
uphead:
  %i = phi i64 [ 0, %freeupper ], [ %in, %upcont ]
  %go = icmp ult i64 %i, %cnt
  br i1 %go, label %upbody, label %freearrays
upbody:
  %bp = getelementptr inbounds ptr, ptr %upper, i64 %i
  %blk = load ptr, ptr %bp, align 8
  call void @free(ptr %blk)
  br label %upcont
upcont:
  %in = add i64 %i, 1
  br label %uphead
freearrays:
  call void @hnsw_free_field(ptr %h, i64 96)
  call void @hnsw_free_field(ptr %h, i64 104)
  call void @hnsw_free_field(ptr %h, i64 112)
  call void @hnsw_free_field(ptr %h, i64 120)
  call void @hnsw_free_field(ptr %h, i64 128)
  call void @hnsw_free_field(ptr %h, i64 136)
  call void @hnsw_free_field(ptr %h, i64 144)
  call void @hnsw_free_field(ptr %h, i64 152)
  call void @hnsw_free_field(ptr %h, i64 160)
  call void @hnsw_free_field(ptr %h, i64 168)
  call void @hnsw_free_field(ptr %h, i64 176)
  call void @hnsw_free_field(ptr %h, i64 184)
  call void @hnsw_free_field(ptr %h, i64 192)
  call void @hnsw_free_field(ptr %h, i64 200)
  call void @free(ptr %h)
  br label %ret
ret:
  ret void
}

define internal void @hnsw_free_field(ptr %h, i64 %off) #1 {
entry:
  %fp = getelementptr inbounds i8, ptr %h, i64 %off
  %p = load ptr, ptr %fp, align 8
  call void @free(ptr %p)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind }
