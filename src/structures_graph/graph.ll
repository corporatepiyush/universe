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

; Array-backed graph (directed/undirected, optional i64 edge weights) with
; BFS / DFS / Dijkstra / connected-components / shortest-path.
;
; ============================================================================
; DESIGN — flat index-linked adjacency, NO pointer webs, NO per-node malloc
; ----------------------------------------------------------------------------
;   A graph is three flat i32 vertex arrays plus ONE flat edge-record array,
;   all addressed by i32 index (the house "pointer elimination + flat indexing"
;   idiom). Nothing per-vertex or per-edge is malloc'd individually; the two
;   payloads grow by doubling realloc exactly like the sibling array/queue.
;
;   Vertex u's out-adjacency is a singly-linked list threaded THROUGH the edge
;   array by i32 "next" indices (sentinel -1), not by pointers:
;       vhead[u] = index of first incident edge record, or -1
;       vtail[u] = index of last  incident edge record, or -1   (O(1) append)
;       vdeg[u]  = out-degree                                   (O(1) degree)
;   Appending at the tail keeps neighbours in INSERTION order (deterministic
;   BFS/DFS), and is O(1) because we cache the tail index.
;
;   Edge record (16 B, AoS, prefetch-friendly):
;       dst@0 (i32 global vertex id)  next@4 (i32 list link, -1 = end)
;       weight@8 (i64)
;   i32 dst/next => an edge index or vertex id must fit i32 (< 2^31); create
;   caps the vertex count at 2^30 so all index math stays in i32 and every
;   `dst`/`next`/`vhead` byte is a real slot. NOTE the sentinel trap: -1 as
;   i32 is 0xFFFFFFFF; it is compared AS i32 before any zext (a zext of the
;   sentinel would be 4294967295, not -1) — see IR hazard on narrow sentinels.
;
;   Directed: add_edge stores ONE record u->v. Undirected: TWO records
;   (u->v and v->u), so out-adjacency == full adjacency and the flood/BFS
;   routines need only follow `next`.
;
;   LAYOUT TRADEOFF vs CSR (catalog lists "CSR / bi-directional CSR"): this is
;   the MUTABLE, insertion-order adjacency variant — O(1) add_edge/degree and
;   incremental edits, but a neighbour scan chases `next` links through the
;   shared edge array (scattered, prefetcher-unfriendly). A contiguous CSR
;   (row-offset + sorted column-index arrays, built once) is the locality-
;   optimal READ-ONLY sibling for large static analytics and is a planned
;   separate module; choose THIS one when the graph is edited after build,
;   and the future CSR variant when it is built once and traversed many times.
;
;   Header (72 B, one malloc, stable handle):
;     vhead@0 vtail@8 vdeg@16 edges@24 nverts@32 vcap@40 ecount@48 ecap@56
;     directed@64
;
;   Traversal engines keep COMPUTE (index math) in registers and MEMORY
;   (visited/queue/dist scratch) in flat arrays sized once per call:
;     * BFS  : visited i8[nverts] + FIFO queue i32[nverts]; mark-on-enqueue,
;              record dequeue order. Each vertex enqueued once => O(V+E).
;     * DFS  : visited i8[nverts] + LIFO stack i32[ecount+1]; mark-on-pop
;              (true pre-order). Push bound = 1 + sum(deg) = 1 + ecount.
;     * Dijkstra: INLINE binary min-heap of (dist,vid) 16 B records with the
;              classic "hole" sift (one settle-store per op). Lazy deletion:
;              a popped stale entry (d > dist[u]) is skipped before scanning,
;              so every vertex is scanned exactly once and total pushes
;              <= ecount+1 => the heap fits in ecount+1 records with no growth.
;              Nonneg weights only (documented); unsigned distance compares.
;     * connected_components: union-find (path-halving) over every stored edge
;              treated as undirected, then a dense relabel pass. Robust whether
;              the graph was built directed or undirected.
;     * shortest_path: unweighted BFS hop distance u->v (-1 if unreachable).
;   Loop state lives in entry-block allocas (mem2reg promotes to registers at
;   -O3) rather than hand-woven phi chains — correctness first, same codegen.
;
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX):
;   ptr universe_ds_graph_create(i64 nverts, i32 directed)
;   void universe_ds_graph_destroy(ptr g)
;   i64 universe_ds_graph_add_vertex(ptr g)                 ; new id, -1 = OOM
;   i32 universe_ds_graph_add_edge(ptr g, i64 u, i64 v, i64 w)
;   i64 universe_ds_graph_degree(ptr g, i64 u)              ; -1 = bad index
;   i32 universe_ds_graph_has_edge(ptr g, i64 u, i64 v)     ; 1 yes / 0 no
;   i64 universe_ds_graph_neighbors(ptr g, i64 u, ptr out, i64 max) ; count
;   i64 universe_ds_graph_vcount(ptr g)
;   i64 universe_ds_graph_ecount(ptr g)                     ; edge records
;   i64 universe_ds_graph_bfs(ptr g, i64 src, ptr order_out)      ; count, -1 err
;   i64 universe_ds_graph_dfs(ptr g, i64 src, ptr order_out)      ; count, -1 err
;   i32 universe_ds_graph_dijkstra(ptr g, i64 src, ptr dist_out)  ; i64[nverts]
;   i64 universe_ds_graph_connected_components(ptr g, ptr labels_out) ; k, -1 err
;   i64 universe_ds_graph_shortest_path(ptr g, i64 u, i64 v)      ; hops, -1

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; internal helpers
; ===========================================================================

; Append one directed edge record u->v (weight w). Caller ensured edge room
; and validated indices. Threads the record onto u's tail-linked adjacency.
define internal void @g_link(ptr %g, i64 %u, i64 %v, i64 %w) #5 {
entry:
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges.r = load ptr, ptr %edges.p, align 8
  %ec.p = getelementptr inbounds nuw i8, ptr %g, i64 48
  %e = load i64, ptr %ec.p, align 8
  %e32 = trunc i64 %e to i32
  %v32 = trunc i64 %v to i32
  %recoff = shl nuw i64 %e, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges.r, i64 %recoff
  store i32 %v32, ptr %rec, align 4
  %nextp = getelementptr inbounds nuw i8, ptr %rec, i64 4
  store i32 -1, ptr %nextp, align 4
  %wp = getelementptr inbounds nuw i8, ptr %rec, i64 8
  store i64 %w, ptr %wp, align 8
  ; link into u's list
  %vhead = load ptr, ptr %g, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %g, i64 8
  %vtail = load ptr, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %g, i64 16
  %vdeg = load ptr, ptr %vd.p, align 8
  %uoff = shl nuw i64 %u, 2
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %uoff
  %tp = getelementptr inbounds nuw i8, ptr %vtail, i64 %uoff
  %h = load i32, ptr %hp, align 4
  %empty = icmp eq i32 %h, -1
  br i1 %empty, label %first, label %append

first:
  store i32 %e32, ptr %hp, align 4
  store i32 %e32, ptr %tp, align 4
  br label %bump

append:
  %t = load i32, ptr %tp, align 4
  %ti = zext i32 %t to i64
  %toff = shl nuw i64 %ti, 4
  %trec = getelementptr inbounds nuw i8, ptr %edges.r, i64 %toff
  %tnextp = getelementptr inbounds nuw i8, ptr %trec, i64 4
  store i32 %e32, ptr %tnextp, align 4
  store i32 %e32, ptr %tp, align 4
  br label %bump

bump:
  %dp = getelementptr inbounds nuw i8, ptr %vdeg, i64 %uoff
  %d = load i32, ptr %dp, align 4
  %d1 = add i32 %d, 1
  store i32 %d1, ptr %dp, align 4
  %e1 = add nuw i64 %e, 1
  store i64 %e1, ptr %ec.p, align 8
  ret void
}

; Grow the three vertex arrays 2x, initialising the new slots (-1/-1/0).
; Returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW.
define internal i32 @g_grow_verts(ptr %g) #3 {
entry:
  %vcap.p = getelementptr inbounds nuw i8, ptr %g, i64 40
  %vcap = load i64, ptr %vcap.p, align 8
  %vcap2 = shl i64 %vcap, 1
  %too.big = icmp ugt i64 %vcap2, 2147483648    ; keep vertex ids in i32
  br i1 %too.big, label %ovf, label %do, !prof !0

do:
  %bytes = shl nuw i64 %vcap2, 2
  %vhead0 = load ptr, ptr %g, align 8
  %vhead = call ptr @realloc(ptr %vhead0, i64 %bytes)
  %vh.null = icmp eq ptr %vhead, null
  br i1 %vh.null, label %oom, label %vt, !prof !0

vt:
  store ptr %vhead, ptr %g, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %g, i64 8
  %vtail0 = load ptr, ptr %vt.p, align 8
  %vtail = call ptr @realloc(ptr %vtail0, i64 %bytes)
  %vt.null = icmp eq ptr %vtail, null
  br i1 %vt.null, label %oom, label %vd, !prof !0

vd:
  store ptr %vtail, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %g, i64 16
  %vdeg0 = load ptr, ptr %vd.p, align 8
  %vdeg = call ptr @realloc(ptr %vdeg0, i64 %bytes)
  %vd.null = icmp eq ptr %vdeg, null
  br i1 %vd.null, label %oom, label %fill, !prof !0

fill:
  store ptr %vdeg, ptr %vd.p, align 8
  %newoff = shl nuw i64 %vcap, 2                ; old cap * 4
  %newlen = shl nuw i64 %vcap, 2                ; (2c - c) * 4 = c*4
  %hn = getelementptr inbounds nuw i8, ptr %vhead, i64 %newoff
  call void @llvm.memset.p0.i64(ptr %hn, i8 -1, i64 %newlen, i1 false)
  %tn = getelementptr inbounds nuw i8, ptr %vtail, i64 %newoff
  call void @llvm.memset.p0.i64(ptr %tn, i8 -1, i64 %newlen, i1 false)
  %dn = getelementptr inbounds nuw i8, ptr %vdeg, i64 %newoff
  call void @llvm.memset.p0.i64(ptr %dn, i8 0, i64 %newlen, i1 false)
  store i64 %vcap2, ptr %vcap.p, align 8
  ret i32 0

oom:
  ret i32 2
ovf:
  ret i32 3
}

; Grow the edge-record array 2x. Returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW.
define internal i32 @g_grow_edges(ptr %g) #3 {
entry:
  %ecap.p = getelementptr inbounds nuw i8, ptr %g, i64 56
  %ecap = load i64, ptr %ecap.p, align 8
  %ecap2 = shl i64 %ecap, 1
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ecap2, i64 16)
  %bytes = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %wrap = icmp eq i64 %ecap2, 0
  %bad = or i1 %o, %wrap
  br i1 %bad, label %ovf, label %do, !prof !0

do:
  %e.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges0 = load ptr, ptr %e.p, align 8
  %edges = call ptr @realloc(ptr %edges0, i64 %bytes)
  %e.null = icmp eq ptr %edges, null
  br i1 %e.null, label %oom, label %ok, !prof !0

ok:
  store ptr %edges, ptr %e.p, align 8
  store i64 %ecap2, ptr %ecap.p, align 8
  ret i32 0

oom:
  ret i32 2
ovf:
  ret i32 3
}

; Ensure at least one free edge slot. Returns 0/2/3.
define internal i32 @g_ensure_edge(ptr %g) #3 {
entry:
  %ec.p = getelementptr inbounds nuw i8, ptr %g, i64 48
  %ec = load i64, ptr %ec.p, align 8
  %ecap.p = getelementptr inbounds nuw i8, ptr %g, i64 56
  %ecap = load i64, ptr %ecap.p, align 8
  %full = icmp uge i64 %ec, %ecap
  br i1 %full, label %grow, label %ok, !prof !0

grow:
  %rc = call i32 @g_grow_edges(ptr %g)
  ret i32 %rc
ok:
  ret i32 0
}

; union-find find with path halving. parent[] is i32; all entries valid ids.
define internal i64 @g_find(ptr %parent, i64 %x0) #4 {
entry:
  br label %loop

loop:
  %x = phi i64 [ %x0, %entry ], [ %gp, %half ]
  %xoff = shl nuw i64 %x, 2
  %xp = getelementptr inbounds nuw i8, ptr %parent, i64 %xoff
  %px32 = load i32, ptr %xp, align 4
  %px = zext i32 %px32 to i64
  %isroot = icmp eq i64 %px, %x
  br i1 %isroot, label %done, label %half

half:
  %poff = shl nuw i64 %px, 2
  %pp = getelementptr inbounds nuw i8, ptr %parent, i64 %poff
  %gp32 = load i32, ptr %pp, align 4
  %gp = zext i32 %gp32 to i64
  store i32 %gp32, ptr %xp, align 4             ; parent[x] = parent[parent[x]]
  br label %loop

done:
  ret i64 %x
}

; Inline (dist,vid) binary min-heap (16 B records: dist@0 vid@8).
; push: hole sift-up. Returns new length.
define internal i64 @g_heap_push(ptr %heap, i64 %len, i64 %dist, i64 %vid) #4 {
entry:
  %hole = alloca i64, align 8
  store i64 %len, ptr %hole, align 8
  br label %loop

loop:
  %h = load i64, ptr %hole, align 8
  %atroot = icmp eq i64 %h, 0
  br i1 %atroot, label %place, label %body

body:
  %hm1 = sub i64 %h, 1
  %parent = lshr i64 %hm1, 1
  %poff = shl nuw i64 %parent, 4
  %prec = getelementptr inbounds nuw i8, ptr %heap, i64 %poff
  %pd = load i64, ptr %prec, align 8
  %gt = icmp ugt i64 %pd, %dist
  br i1 %gt, label %move, label %place

move:
  %pvp = getelementptr inbounds nuw i8, ptr %prec, i64 8
  %pv = load i64, ptr %pvp, align 8
  %hoff = shl nuw i64 %h, 4
  %hrec = getelementptr inbounds nuw i8, ptr %heap, i64 %hoff
  store i64 %pd, ptr %hrec, align 8
  %hvp = getelementptr inbounds nuw i8, ptr %hrec, i64 8
  store i64 %pv, ptr %hvp, align 8
  store i64 %parent, ptr %hole, align 8
  br label %loop

place:
  %hf = load i64, ptr %hole, align 8
  %foff = shl nuw i64 %hf, 4
  %frec = getelementptr inbounds nuw i8, ptr %heap, i64 %foff
  store i64 %dist, ptr %frec, align 8
  %fvp = getelementptr inbounds nuw i8, ptr %frec, i64 8
  store i64 %vid, ptr %fvp, align 8
  %newlen = add nuw i64 %len, 1
  ret i64 %newlen
}

; pop: extract min (into *od,*ov), sift the last element down. Returns new len.
; Precondition len >= 1.
define internal i64 @g_heap_pop(ptr %heap, i64 %len, ptr %od, ptr %ov) #4 {
entry:
  %d0 = load i64, ptr %heap, align 8
  %v0p = getelementptr inbounds nuw i8, ptr %heap, i64 8
  %v0 = load i64, ptr %v0p, align 8
  store i64 %d0, ptr %od, align 8
  store i64 %v0, ptr %ov, align 8
  %newlen = sub i64 %len, 1
  %empty = icmp eq i64 %newlen, 0
  br i1 %empty, label %retz, label %sift

retz:
  ret i64 0

sift:
  %loff = shl nuw i64 %newlen, 4
  %lrec = getelementptr inbounds nuw i8, ptr %heap, i64 %loff
  %ld = load i64, ptr %lrec, align 8
  %lvp = getelementptr inbounds nuw i8, ptr %lrec, i64 8
  %lv = load i64, ptr %lvp, align 8
  %hole = alloca i64, align 8
  store i64 0, ptr %hole, align 8
  br label %loop

loop:
  %h = load i64, ptr %hole, align 8
  %h2 = shl nuw i64 %h, 1
  %c0 = add nuw i64 %h2, 1
  %noc = icmp uge i64 %c0, %newlen
  br i1 %noc, label %place, label %pick

pick:
  %c0off = shl nuw i64 %c0, 4
  %c0rec = getelementptr inbounds nuw i8, ptr %heap, i64 %c0off
  %cd0 = load i64, ptr %c0rec, align 8
  %right = add nuw i64 %c0, 1
  %hasr = icmp ult i64 %right, %newlen
  %ridx = select i1 %hasr, i64 %right, i64 %c0   ; safe load even w/o right
  %roff = shl nuw i64 %ridx, 4
  %rrec = getelementptr inbounds nuw i8, ptr %heap, i64 %roff
  %rd = load i64, ptr %rrec, align 8
  %rlt = icmp ult i64 %rd, %cd0
  %useright = and i1 %hasr, %rlt
  %child = select i1 %useright, i64 %right, i64 %c0
  %cd = select i1 %useright, i64 %rd, i64 %cd0
  %settle = icmp uge i64 %cd, %ld
  br i1 %settle, label %place, label %down

down:
  %coff = shl nuw i64 %child, 4
  %crec = getelementptr inbounds nuw i8, ptr %heap, i64 %coff
  %cvp = getelementptr inbounds nuw i8, ptr %crec, i64 8
  %cv = load i64, ptr %cvp, align 8
  %hoff = shl nuw i64 %h, 4
  %hrec = getelementptr inbounds nuw i8, ptr %heap, i64 %hoff
  store i64 %cd, ptr %hrec, align 8
  %hvp = getelementptr inbounds nuw i8, ptr %hrec, i64 8
  store i64 %cv, ptr %hvp, align 8
  store i64 %child, ptr %hole, align 8
  br label %loop

place:
  %hf = load i64, ptr %hole, align 8
  %foff = shl nuw i64 %hf, 4
  %frec = getelementptr inbounds nuw i8, ptr %heap, i64 %foff
  store i64 %ld, ptr %frec, align 8
  %fvp = getelementptr inbounds nuw i8, ptr %frec, i64 8
  store i64 %lv, ptr %fvp, align 8
  ret i64 %newlen
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_graph_create(i64 %nverts, i32 %directed) local_unnamed_addr #1 {
entry:
  %vc.min = call i64 @llvm.umax.i64(i64 %nverts, i64 8)
  %too.big = icmp ugt i64 %vc.min, 1073741824       ; 2^30 vertex ceiling
  br i1 %too.big, label %fail0, label %vpow2, !prof !0

vpow2:
  %vcm1 = add i64 %vc.min, -1
  %vlz = call i64 @llvm.ctlz.i64(i64 %vcm1, i1 true)
  %vshift = sub nuw nsw i64 64, %vlz
  %vcap = shl nuw i64 1, %vshift
  %vbytes = shl nuw i64 %vcap, 2
  br label %alloc.hdr

alloc.hdr:
  %hdr = call ptr @malloc(i64 72)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail0, label %alloc.vh, !prof !0

alloc.vh:
  %vhead = call ptr @malloc(i64 %vbytes)
  %vhead.null = icmp eq ptr %vhead, null
  br i1 %vhead.null, label %free.hdr, label %alloc.vt, !prof !0

alloc.vt:
  %vtail = call ptr @malloc(i64 %vbytes)
  %vtail.null = icmp eq ptr %vtail, null
  br i1 %vtail.null, label %free.vh, label %alloc.vd, !prof !0

alloc.vd:
  %vdeg = call ptr @malloc(i64 %vbytes)
  %vdeg.null = icmp eq ptr %vdeg, null
  br i1 %vdeg.null, label %free.vt, label %alloc.e, !prof !0

alloc.e:
  %edges = call ptr @malloc(i64 256)                ; 16 records * 16 B
  %edges.null = icmp eq ptr %edges, null
  br i1 %edges.null, label %free.vd, label %init, !prof !0

init:
  call void @llvm.memset.p0.i64(ptr %vhead, i8 -1, i64 %vbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %vtail, i8 -1, i64 %vbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %vdeg, i8 0, i64 %vbytes, i1 false)
  store ptr %vhead, ptr %hdr, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store ptr %vtail, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store ptr %vdeg, ptr %vd.p, align 8
  %e.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store ptr %edges, ptr %e.p, align 8
  %nv.p = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 %nverts, ptr %nv.p, align 8
  %vcap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 40
  store i64 %vcap, ptr %vcap.p, align 8
  %ec.p = getelementptr inbounds nuw i8, ptr %hdr, i64 48
  store i64 0, ptr %ec.p, align 8
  %ecap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 56
  store i64 16, ptr %ecap.p, align 8
  %dir64 = zext i32 %directed to i64
  %dir.p = getelementptr inbounds nuw i8, ptr %hdr, i64 64
  store i64 %dir64, ptr %dir.p, align 8
  ret ptr %hdr

free.vd:
  call void @free(ptr %vdeg)
  br label %free.vt
free.vt:
  call void @free(ptr %vtail)
  br label %free.vh
free.vh:
  call void @free(ptr %vhead)
  br label %free.hdr
free.hdr:
  call void @free(ptr %hdr)
  br label %fail0
fail0:
  ret ptr null
}

define void @universe_ds_graph_destroy(ptr %g) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %done, label %do, !prof !0

do:
  %vhead = load ptr, ptr %g, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %g, i64 8
  %vtail = load ptr, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %g, i64 16
  %vdeg = load ptr, ptr %vd.p, align 8
  %e.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %e.p, align 8
  call void @free(ptr %vhead)
  call void @free(ptr %vtail)
  call void @free(ptr %vdeg)
  call void @free(ptr %edges)
  call void @free(ptr nonnull %g)
  br label %done
done:
  ret void
}

; ===========================================================================
; add_vertex / add_edge
; ===========================================================================
define i64 @universe_ds_graph_add_vertex(ptr %g) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %err, label %check, !prof !0

err:
  ret i64 -1

check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %vcap.p = getelementptr inbounds nuw i8, ptr %g, i64 40
  %vcap = load i64, ptr %vcap.p, align 8
  %full = icmp uge i64 %nv, %vcap
  br i1 %full, label %grow, label %cont, !prof !0

grow:
  %rc = call i32 @g_grow_verts(ptr %g)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %cont, label %err, !prof !2

cont:
  %vhead = load ptr, ptr %g, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %g, i64 8
  %vtail = load ptr, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %g, i64 16
  %vdeg = load ptr, ptr %vd.p, align 8
  %off = shl nuw i64 %nv, 2
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %off
  store i32 -1, ptr %hp, align 4
  %tp = getelementptr inbounds nuw i8, ptr %vtail, i64 %off
  store i32 -1, ptr %tp, align 4
  %dp = getelementptr inbounds nuw i8, ptr %vdeg, i64 %off
  store i32 0, ptr %dp, align 4
  %nv1 = add nuw i64 %nv, 1
  store i64 %nv1, ptr %nv.p, align 8
  ret i64 %nv
}

define i32 @universe_ds_graph_add_edge(ptr %g, i64 %u, i64 %v, i64 %w) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %ubad = icmp uge i64 %u, %nv
  %vbad = icmp uge i64 %v, %nv
  %bad = or i1 %ubad, %vbad
  br i1 %bad, label %err.idx, label %e1, !prof !0

err.idx:
  ret i32 7

e1:
  %rc1 = call i32 @g_ensure_edge(ptr %g)
  %ok1 = icmp eq i32 %rc1, 0
  br i1 %ok1, label %link1, label %err.grow, !prof !2

err.grow:
  ret i32 %rc1

link1:
  call void @g_link(ptr %g, i64 %u, i64 %v, i64 %w)
  %dir.p = getelementptr inbounds nuw i8, ptr %g, i64 64
  %dir = load i64, ptr %dir.p, align 8
  %isdir = icmp ne i64 %dir, 0
  br i1 %isdir, label %done, label %e2

e2:
  %rc2 = call i32 @g_ensure_edge(ptr %g)
  %ok2 = icmp eq i32 %rc2, 0
  br i1 %ok2, label %link2, label %err.grow2, !prof !2

err.grow2:
  ret i32 %rc2

link2:
  call void @g_link(ptr %g, i64 %v, i64 %u, i64 %w)
  br label %done

done:
  ret i32 0
}

; ===========================================================================
; degree / has_edge / neighbors / vcount / ecount
; ===========================================================================
define i64 @universe_ds_graph_degree(ptr %g, i64 %u) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %bad = icmp uge i64 %u, %nv
  br i1 %bad, label %err, label %ok, !prof !0
ok:
  %vd.p = getelementptr inbounds nuw i8, ptr %g, i64 16
  %vdeg = load ptr, ptr %vd.p, align 8
  %off = shl nuw i64 %u, 2
  %dp = getelementptr inbounds nuw i8, ptr %vdeg, i64 %off
  %d = load i32, ptr %dp, align 4
  %d64 = zext i32 %d to i64
  ret i64 %d64
}

define i32 @universe_ds_graph_has_edge(ptr %g, i64 %u, i64 %v) local_unnamed_addr #1 {
entry:
  %e = alloca i32, align 4
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %no, label %check, !prof !0
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %ubad = icmp uge i64 %u, %nv
  %vbad = icmp uge i64 %v, %nv
  %bad = or i1 %ubad, %vbad
  br i1 %bad, label %no, label %start, !prof !0
start:
  %vhead = load ptr, ptr %g, align 8
  %off = shl nuw i64 %u, 2
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %off
  %h = load i32, ptr %hp, align 4
  store i32 %h, ptr %e, align 4
  %v32 = trunc i64 %v to i32
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %loop
loop:
  %ei = load i32, ptr %e, align 4
  %end = icmp eq i32 %ei, -1
  br i1 %end, label %no, label %body
body:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst = load i32, ptr %rec, align 4
  %hit = icmp eq i32 %dst, %v32
  br i1 %hit, label %yes, label %next
next:
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  br label %loop
yes:
  ret i32 1
no:
  ret i32 0
}

define i64 @universe_ds_graph_neighbors(ptr %g, i64 %u, ptr %out, i64 %max) local_unnamed_addr #1 {
entry:
  %e = alloca i32, align 4
  %k = alloca i64, align 8
  %gnull = icmp eq ptr %g, null
  %onull = icmp eq ptr %out, null
  %bad0 = or i1 %gnull, %onull
  br i1 %bad0, label %zero, label %check, !prof !0
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %bad = icmp uge i64 %u, %nv
  br i1 %bad, label %zero, label %start, !prof !0
start:
  %vhead = load ptr, ptr %g, align 8
  %off = shl nuw i64 %u, 2
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %off
  %h = load i32, ptr %hp, align 4
  store i32 %h, ptr %e, align 4
  store i64 0, ptr %k, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %loop
loop:
  %ei = load i32, ptr %e, align 4
  %end = icmp eq i32 %ei, -1
  br i1 %end, label %done, label %chkmax
chkmax:
  %kk = load i64, ptr %k, align 8
  %atmax = icmp uge i64 %kk, %max
  br i1 %atmax, label %done, label %body
body:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst = load i32, ptr %rec, align 4
  %ooff = shl nuw i64 %kk, 2
  %op = getelementptr inbounds nuw i8, ptr %out, i64 %ooff
  store i32 %dst, ptr %op, align 4
  %kk1 = add nuw i64 %kk, 1
  store i64 %kk1, ptr %k, align 8
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  br label %loop
done:
  %kf = load i64, ptr %k, align 8
  ret i64 %kf
zero:
  ret i64 0
}

define i64 @universe_ds_graph_vcount(ptr %g) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %z, label %ok, !prof !0
z:
  ret i64 0
ok:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  ret i64 %nv
}

define i64 @universe_ds_graph_ecount(ptr %g) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %g, null
  br i1 %isnull, label %z, label %ok, !prof !0
z:
  ret i64 0
ok:
  %ec.p = getelementptr inbounds nuw i8, ptr %g, i64 48
  %ec = load i64, ptr %ec.p, align 8
  ret i64 %ec
}

; ===========================================================================
; BFS
; ===========================================================================
define i64 @universe_ds_graph_bfs(ptr %g, i64 %src, ptr %out) local_unnamed_addr #1 {
entry:
  %qh = alloca i64, align 8
  %qt = alloca i64, align 8
  %oidx = alloca i64, align 8
  %e = alloca i32, align 4
  %gnull = icmp eq ptr %g, null
  %onull = icmp eq ptr %out, null
  %bad0 = or i1 %gnull, %onull
  br i1 %bad0, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %srcbad = icmp uge i64 %src, %nv
  br i1 %srcbad, label %err, label %alloc, !prof !0
alloc:
  %visited = call ptr @malloc(i64 %nv)
  %vis.null = icmp eq ptr %visited, null
  br i1 %vis.null, label %err, label %allocq, !prof !0
allocq:
  %qbytes = shl nuw i64 %nv, 2
  %queue = call ptr @malloc(i64 %qbytes)
  %q.null = icmp eq ptr %queue, null
  br i1 %q.null, label %freev, label %setup, !prof !0
freev:
  call void @free(ptr %visited)
  br label %err
setup:
  call void @llvm.memset.p0.i64(ptr %visited, i8 0, i64 %nv, i1 false)
  %vsp = getelementptr inbounds nuw i8, ptr %visited, i64 %src
  store i8 1, ptr %vsp, align 1
  %src32 = trunc i64 %src to i32
  store i32 %src32, ptr %queue, align 4
  store i64 0, ptr %qh, align 8
  store i64 1, ptr %qt, align 8
  store i64 0, ptr %oidx, align 8
  %vhead = load ptr, ptr %g, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %loop
loop:
  %h = load i64, ptr %qh, align 8
  %t = load i64, ptr %qt, align 8
  %more = icmp ult i64 %h, %t
  br i1 %more, label %node, label %finish
node:
  %hoff = shl nuw i64 %h, 2
  %qslot = getelementptr inbounds nuw i8, ptr %queue, i64 %hoff
  %u32 = load i32, ptr %qslot, align 4
  %h1 = add nuw i64 %h, 1
  store i64 %h1, ptr %qh, align 8
  %oi = load i64, ptr %oidx, align 8
  %ooff = shl nuw i64 %oi, 2
  %op = getelementptr inbounds nuw i8, ptr %out, i64 %ooff
  store i32 %u32, ptr %op, align 4
  %oi1 = add nuw i64 %oi, 1
  store i64 %oi1, ptr %oidx, align 8
  %u = zext i32 %u32 to i64
  %uoff = shl nuw i64 %u, 2
  %uhp = getelementptr inbounds nuw i8, ptr %vhead, i64 %uoff
  %e0 = load i32, ptr %uhp, align 4
  store i32 %e0, ptr %e, align 4
  br label %inner
inner:
  %ei = load i32, ptr %e, align 4
  %iend = icmp eq i32 %ei, -1
  br i1 %iend, label %loop, label %ibody
ibody:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst32 = load i32, ptr %rec, align 4
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  %dst = zext i32 %dst32 to i64
  %vp = getelementptr inbounds nuw i8, ptr %visited, i64 %dst
  %vv = load i8, ptr %vp, align 1
  %seen = icmp ne i8 %vv, 0
  br i1 %seen, label %inner, label %add
add:
  store i8 1, ptr %vp, align 1
  %tt = load i64, ptr %qt, align 8
  %toff = shl nuw i64 %tt, 2
  %tslot = getelementptr inbounds nuw i8, ptr %queue, i64 %toff
  store i32 %dst32, ptr %tslot, align 4
  %tt1 = add nuw i64 %tt, 1
  store i64 %tt1, ptr %qt, align 8
  br label %inner
finish:
  call void @free(ptr %visited)
  call void @free(ptr %queue)
  %cnt = load i64, ptr %oidx, align 8
  ret i64 %cnt
}

; ===========================================================================
; DFS (iterative, mark-on-pop => true pre-order)
; ===========================================================================
define i64 @universe_ds_graph_dfs(ptr %g, i64 %src, ptr %out) local_unnamed_addr #1 {
entry:
  %sp = alloca i64, align 8
  %oidx = alloca i64, align 8
  %e = alloca i32, align 4
  %gnull = icmp eq ptr %g, null
  %onull = icmp eq ptr %out, null
  %bad0 = or i1 %gnull, %onull
  br i1 %bad0, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %srcbad = icmp uge i64 %src, %nv
  br i1 %srcbad, label %err, label %alloc, !prof !0
alloc:
  %visited = call ptr @malloc(i64 %nv)
  %vis.null = icmp eq ptr %visited, null
  br i1 %vis.null, label %err, label %allocs, !prof !0
allocs:
  %ec.p = getelementptr inbounds nuw i8, ptr %g, i64 48
  %ec = load i64, ptr %ec.p, align 8
  %scap = add nuw i64 %ec, 1
  %sbytes = shl nuw i64 %scap, 2
  %stack = call ptr @malloc(i64 %sbytes)
  %s.null = icmp eq ptr %stack, null
  br i1 %s.null, label %freev, label %setup, !prof !0
freev:
  call void @free(ptr %visited)
  br label %err
setup:
  call void @llvm.memset.p0.i64(ptr %visited, i8 0, i64 %nv, i1 false)
  %src32 = trunc i64 %src to i32
  store i32 %src32, ptr %stack, align 4
  store i64 1, ptr %sp, align 8
  store i64 0, ptr %oidx, align 8
  %vhead = load ptr, ptr %g, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %loop
loop:
  %s = load i64, ptr %sp, align 8
  %empty = icmp eq i64 %s, 0
  br i1 %empty, label %finish, label %pop
pop:
  %s1 = sub i64 %s, 1
  store i64 %s1, ptr %sp, align 8
  %soff = shl nuw i64 %s1, 2
  %sslot = getelementptr inbounds nuw i8, ptr %stack, i64 %soff
  %u32 = load i32, ptr %sslot, align 4
  %u = zext i32 %u32 to i64
  %vp = getelementptr inbounds nuw i8, ptr %visited, i64 %u
  %vv = load i8, ptr %vp, align 1
  %seen = icmp ne i8 %vv, 0
  br i1 %seen, label %loop, label %visit
visit:
  store i8 1, ptr %vp, align 1
  %oi = load i64, ptr %oidx, align 8
  %ooff = shl nuw i64 %oi, 2
  %op = getelementptr inbounds nuw i8, ptr %out, i64 %ooff
  store i32 %u32, ptr %op, align 4
  %oi1 = add nuw i64 %oi, 1
  store i64 %oi1, ptr %oidx, align 8
  %uoff = shl nuw i64 %u, 2
  %uhp = getelementptr inbounds nuw i8, ptr %vhead, i64 %uoff
  %e0 = load i32, ptr %uhp, align 4
  store i32 %e0, ptr %e, align 4
  br label %inner
inner:
  %ei = load i32, ptr %e, align 4
  %iend = icmp eq i32 %ei, -1
  br i1 %iend, label %loop, label %ibody
ibody:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst32 = load i32, ptr %rec, align 4
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  %dst = zext i32 %dst32 to i64
  %dvp = getelementptr inbounds nuw i8, ptr %visited, i64 %dst
  %dvv = load i8, ptr %dvp, align 1
  %dseen = icmp ne i8 %dvv, 0
  br i1 %dseen, label %inner, label %push
push:
  %ps = load i64, ptr %sp, align 8
  %psoff = shl nuw i64 %ps, 2
  %psslot = getelementptr inbounds nuw i8, ptr %stack, i64 %psoff
  store i32 %dst32, ptr %psslot, align 4
  %ps1 = add nuw i64 %ps, 1
  store i64 %ps1, ptr %sp, align 8
  br label %inner
finish:
  call void @free(ptr %visited)
  call void @free(ptr %stack)
  %cnt = load i64, ptr %oidx, align 8
  ret i64 %cnt
}

; ===========================================================================
; Dijkstra (nonneg weights)
; ===========================================================================
define i32 @universe_ds_graph_dijkstra(ptr %g, i64 %src, ptr %dist) local_unnamed_addr #1 {
entry:
  %od = alloca i64, align 8
  %ov = alloca i64, align 8
  %len = alloca i64, align 8
  %i = alloca i64, align 8
  %e = alloca i32, align 4
  %gnull = icmp eq ptr %g, null
  %dnull = icmp eq ptr %dist, null
  %bad0 = or i1 %gnull, %dnull
  br i1 %bad0, label %err.null, label %check, !prof !0
err.null:
  ret i32 1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %srcbad = icmp uge i64 %src, %nv
  br i1 %srcbad, label %err.idx, label %initloop, !prof !0
err.idx:
  ret i32 7
initloop:
  store i64 0, ptr %i, align 8
  br label %init.head
init.head:
  %ii = load i64, ptr %i, align 8
  %idone = icmp uge i64 %ii, %nv
  br i1 %idone, label %init.done, label %init.body
init.body:
  %ioff = shl nuw i64 %ii, 3
  %ip = getelementptr inbounds nuw i8, ptr %dist, i64 %ioff
  store i64 9223372036854775807, ptr %ip, align 8
  %ii1 = add nuw i64 %ii, 1
  store i64 %ii1, ptr %i, align 8
  br label %init.head
init.done:
  %soff = shl nuw i64 %src, 3
  %sp = getelementptr inbounds nuw i8, ptr %dist, i64 %soff
  store i64 0, ptr %sp, align 8
  %ec.p = getelementptr inbounds nuw i8, ptr %g, i64 48
  %ec = load i64, ptr %ec.p, align 8
  %hcap = add nuw i64 %ec, 1
  %hbytes = shl nuw i64 %hcap, 4
  %heap = call ptr @malloc(i64 %hbytes)
  %h.null = icmp eq ptr %heap, null
  br i1 %h.null, label %err.oom, label %pushroot, !prof !0
err.oom:
  ret i32 2
pushroot:
  %l0 = call i64 @g_heap_push(ptr %heap, i64 0, i64 0, i64 %src)
  store i64 %l0, ptr %len, align 8
  %vhead = load ptr, ptr %g, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %main
main:
  %L = load i64, ptr %len, align 8
  %ne = icmp ugt i64 %L, 0
  br i1 %ne, label %popn, label %main.done
popn:
  %L2 = call i64 @g_heap_pop(ptr %heap, i64 %L, ptr %od, ptr %ov)
  store i64 %L2, ptr %len, align 8
  %d = load i64, ptr %od, align 8
  %u = load i64, ptr %ov, align 8
  %duoff = shl nuw i64 %u, 3
  %dup = getelementptr inbounds nuw i8, ptr %dist, i64 %duoff
  %du = load i64, ptr %dup, align 8
  %stale = icmp ugt i64 %d, %du
  br i1 %stale, label %main, label %scan
scan:
  %uoff = shl nuw i64 %u, 2
  %uhp = getelementptr inbounds nuw i8, ptr %vhead, i64 %uoff
  %e0 = load i32, ptr %uhp, align 4
  store i32 %e0, ptr %e, align 4
  br label %sh
sh:
  %ei = load i32, ptr %e, align 4
  %iend = icmp eq i32 %ei, -1
  br i1 %iend, label %main, label %sbody
sbody:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst32 = load i32, ptr %rec, align 4
  %wp = getelementptr inbounds nuw i8, ptr %rec, i64 8
  %w = load i64, ptr %wp, align 8
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  %dst = zext i32 %dst32 to i64
  %nd = add i64 %d, %w
  %dstoff = shl nuw i64 %dst, 3
  %dvp = getelementptr inbounds nuw i8, ptr %dist, i64 %dstoff
  %dv = load i64, ptr %dvp, align 8
  %better = icmp ult i64 %nd, %dv
  br i1 %better, label %relax, label %sh
relax:
  store i64 %nd, ptr %dvp, align 8
  %Lc = load i64, ptr %len, align 8
  %Ln = call i64 @g_heap_push(ptr %heap, i64 %Lc, i64 %nd, i64 %dst)
  store i64 %Ln, ptr %len, align 8
  br label %sh
main.done:
  call void @free(ptr %heap)
  ret i32 0
}

; ===========================================================================
; connected_components (undirected view via union-find)
; ===========================================================================
define i64 @universe_ds_graph_connected_components(ptr %g, ptr %labels) local_unnamed_addr #1 {
entry:
  %i = alloca i64, align 8
  %k = alloca i64, align 8
  %e = alloca i32, align 4
  %gnull = icmp eq ptr %g, null
  %lnull = icmp eq ptr %labels, null
  %bad0 = or i1 %gnull, %lnull
  br i1 %bad0, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %z = icmp eq i64 %nv, 0
  br i1 %z, label %retz, label %alloc, !prof !0
retz:
  ret i64 0
alloc:
  %pbytes = shl nuw i64 %nv, 2
  %parent = call ptr @malloc(i64 %pbytes)
  %p.null = icmp eq ptr %parent, null
  br i1 %p.null, label %err, label %allocr, !prof !0
allocr:
  %rootid = call ptr @malloc(i64 %pbytes)
  %r.null = icmp eq ptr %rootid, null
  br i1 %r.null, label %freep, label %pinit, !prof !0
freep:
  call void @free(ptr %parent)
  br label %err
pinit:
  store i64 0, ptr %i, align 8
  br label %pi.head
pi.head:
  %pii = load i64, ptr %i, align 8
  %pidone = icmp uge i64 %pii, %nv
  br i1 %pidone, label %uinit, label %pi.body
pi.body:
  %pioff = shl nuw i64 %pii, 2
  %pip = getelementptr inbounds nuw i8, ptr %parent, i64 %pioff
  %pii32 = trunc i64 %pii to i32
  store i32 %pii32, ptr %pip, align 4
  %pii1 = add nuw i64 %pii, 1
  store i64 %pii1, ptr %i, align 8
  br label %pi.head
uinit:
  %vhead = load ptr, ptr %g, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  store i64 0, ptr %i, align 8
  br label %u.head
u.head:
  %ui = load i64, ptr %i, align 8
  %udone = icmp uge i64 %ui, %nv
  br i1 %udone, label %relabel, label %u.start
u.start:
  %uoff = shl nuw i64 %ui, 2
  %uhp = getelementptr inbounds nuw i8, ptr %vhead, i64 %uoff
  %ue0 = load i32, ptr %uhp, align 4
  store i32 %ue0, ptr %e, align 4
  br label %u.inner
u.inner:
  %ei = load i32, ptr %e, align 4
  %iend = icmp eq i32 %ei, -1
  br i1 %iend, label %u.next, label %u.body
u.body:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst32 = load i32, ptr %rec, align 4
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  %dst = zext i32 %dst32 to i64
  %ru = call i64 @g_find(ptr %parent, i64 %ui)
  %rv = call i64 @g_find(ptr %parent, i64 %dst)
  %same = icmp eq i64 %ru, %rv
  br i1 %same, label %u.inner, label %dolink
dolink:
  %ruoff = shl nuw i64 %ru, 2
  %rup = getelementptr inbounds nuw i8, ptr %parent, i64 %ruoff
  %rv32 = trunc i64 %rv to i32
  store i32 %rv32, ptr %rup, align 4
  br label %u.inner
u.next:
  %ui1 = add nuw i64 %ui, 1
  store i64 %ui1, ptr %i, align 8
  br label %u.head
relabel:
  call void @llvm.memset.p0.i64(ptr %rootid, i8 -1, i64 %pbytes, i1 false)
  store i64 0, ptr %k, align 8
  store i64 0, ptr %i, align 8
  br label %l.head
l.head:
  %li = load i64, ptr %i, align 8
  %ldone = icmp uge i64 %li, %nv
  br i1 %ldone, label %l.done, label %l.body
l.body:
  %r = call i64 @g_find(ptr %parent, i64 %li)
  %rroff = shl nuw i64 %r, 2
  %ridp = getelementptr inbounds nuw i8, ptr %rootid, i64 %rroff
  %rid = load i32, ptr %ridp, align 4
  %isnew = icmp eq i32 %rid, -1
  %loff = shl nuw i64 %li, 2
  %lp = getelementptr inbounds nuw i8, ptr %labels, i64 %loff
  br i1 %isnew, label %newc, label %setl
newc:
  %kk = load i64, ptr %k, align 8
  %kk32 = trunc i64 %kk to i32
  store i32 %kk32, ptr %ridp, align 4
  store i32 %kk32, ptr %lp, align 4
  %kk1 = add nuw i64 %kk, 1
  store i64 %kk1, ptr %k, align 8
  br label %l.next
setl:
  store i32 %rid, ptr %lp, align 4
  br label %l.next
l.next:
  %li1 = add nuw i64 %li, 1
  store i64 %li1, ptr %i, align 8
  br label %l.head
l.done:
  call void @free(ptr %parent)
  call void @free(ptr %rootid)
  %kf = load i64, ptr %k, align 8
  ret i64 %kf
}

; ===========================================================================
; shortest_path (unweighted BFS hop distance u->v, -1 if unreachable)
; ===========================================================================
define i64 @universe_ds_graph_shortest_path(ptr %g, i64 %u, i64 %v) local_unnamed_addr #1 {
entry:
  %qh = alloca i64, align 8
  %qt = alloca i64, align 8
  %e = alloca i32, align 4
  %gnull = icmp eq ptr %g, null
  br i1 %gnull, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %g, i64 32
  %nv = load i64, ptr %nv.p, align 8
  %ubad = icmp uge i64 %u, %nv
  %vbad = icmp uge i64 %v, %nv
  %bad = or i1 %ubad, %vbad
  br i1 %bad, label %err, label %chkeq, !prof !0
chkeq:
  %eq = icmp eq i64 %u, %v
  br i1 %eq, label %zero, label %alloc
zero:
  ret i64 0
alloc:
  %dbytes = shl nuw i64 %nv, 2
  %dist = call ptr @malloc(i64 %dbytes)
  %d.null = icmp eq ptr %dist, null
  br i1 %d.null, label %err, label %allocq, !prof !0
allocq:
  %queue = call ptr @malloc(i64 %dbytes)
  %q.null = icmp eq ptr %queue, null
  br i1 %q.null, label %freed, label %setup, !prof !0
freed:
  call void @free(ptr %dist)
  br label %err
setup:
  call void @llvm.memset.p0.i64(ptr %dist, i8 -1, i64 %dbytes, i1 false)
  %uoff = shl nuw i64 %u, 2
  %udp = getelementptr inbounds nuw i8, ptr %dist, i64 %uoff
  store i32 0, ptr %udp, align 4
  %u32 = trunc i64 %u to i32
  store i32 %u32, ptr %queue, align 4
  store i64 0, ptr %qh, align 8
  store i64 1, ptr %qt, align 8
  %vhead = load ptr, ptr %g, align 8
  %edges.p = getelementptr inbounds nuw i8, ptr %g, i64 24
  %edges = load ptr, ptr %edges.p, align 8
  br label %loop
loop:
  %h = load i64, ptr %qh, align 8
  %t = load i64, ptr %qt, align 8
  %more = icmp ult i64 %h, %t
  br i1 %more, label %node, label %done
node:
  %hoff = shl nuw i64 %h, 2
  %qslot = getelementptr inbounds nuw i8, ptr %queue, i64 %hoff
  %x32 = load i32, ptr %qslot, align 4
  %h1 = add nuw i64 %h, 1
  store i64 %h1, ptr %qh, align 8
  %x = zext i32 %x32 to i64
  %xoff = shl nuw i64 %x, 2
  %xdp = getelementptr inbounds nuw i8, ptr %dist, i64 %xoff
  %dx = load i32, ptr %xdp, align 4
  %xhp = getelementptr inbounds nuw i8, ptr %vhead, i64 %xoff
  %e0 = load i32, ptr %xhp, align 4
  store i32 %e0, ptr %e, align 4
  br label %inner
inner:
  %ei = load i32, ptr %e, align 4
  %iend = icmp eq i32 %ei, -1
  br i1 %iend, label %loop, label %ibody
ibody:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %y32 = load i32, ptr %rec, align 4
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  %y = zext i32 %y32 to i64
  %yoff = shl nuw i64 %y, 2
  %ydp = getelementptr inbounds nuw i8, ptr %dist, i64 %yoff
  %dy = load i32, ptr %ydp, align 4
  %unseen = icmp eq i32 %dy, -1
  br i1 %unseen, label %setd, label %inner
setd:
  %nd = add nsw i32 %dx, 1
  store i32 %nd, ptr %ydp, align 4
  %tt = load i64, ptr %qt, align 8
  %toff = shl nuw i64 %tt, 2
  %tslot = getelementptr inbounds nuw i8, ptr %queue, i64 %toff
  store i32 %y32, ptr %tslot, align 4
  %tt1 = add nuw i64 %tt, 1
  store i64 %tt1, ptr %qt, align 8
  br label %inner
done:
  %voff = shl nuw i64 %v, 2
  %vdp = getelementptr inbounds nuw i8, ptr %dist, i64 %voff
  %dv = load i32, ptr %vdp, align 4
  call void @free(ptr %dist)
  call void @free(ptr %queue)
  %dv64 = sext i32 %dv to i64
  ret i64 %dv64
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { cold nounwind willreturn }
attributes #4 = { alwaysinline nounwind willreturn norecurse }
attributes #5 = { alwaysinline nounwind willreturn norecurse memory(argmem: readwrite) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
