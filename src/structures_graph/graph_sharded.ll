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

; universe_ds_graph_sharded — CONCURRENT sharded-array graph.
;
; ============================================================================
; DESIGN — vertices striped across N shards; each shard owns its adjacency
; ----------------------------------------------------------------------------
;   A single lock over one shared adjacency store serialises every writer and
;   scales negatively with cores (the lock line ping-pongs). We STRIPE instead:
;   a POWER-OF-TWO number of shards, each a self-contained flat index-linked
;   adjacency store (the same array-backed layout as the sibling graph.ll)
;   guarded by its OWN spinlock on its OWN 128 B cache line.
;
;   OWNERSHIP: vertex `vid` (and its OUT-adjacency) belongs to shard
;   `vid & (N-1)` (mask, never modulo). Its slot inside that shard is the dense
;   LOCAL index `vid >> log2(N)` (vids of a shard are s, s+N, s+2N, ...). So a
;   shard stores exactly ceil((nverts-s)/N) vertices with zero gaps.
;
;   Edge record (16 B): dst@0 (i32 GLOBAL vid)  next@4 (i32)  weight@8 (i64).
;   The `dst` is the global id so traversal/has_edge need no local<->global
;   translation on the neighbour side.
;
;   ADD_EDGE and cross-shard locking:
;     * DIRECTED u->v : only shard(u) stores the record => lock ONE shard.
;     * UNDIRECTED    : store u->v in shard(u) AND v->u in shard(v). If
;       shard(u)==shard(v) we lock that ONE shard and add both records (never
;       lock twice — the spinlock is non-reentrant). Otherwise it is a
;       CROSS-SHARD edge: we lock the two shards in a FIXED GLOBAL ORDER
;       (LOWER shard index first, then higher) and release in reverse. A fixed
;       total order over all lockers is the classic deadlock-free discipline:
;       no cycle of "waits-for" can form. Threads touching disjoint vertex
;       sets never contend; only genuine same-shard traffic serialises, and
;       only within that shard.
;   degree / neighbors / has_edge lock ONLY the owning shard(u).
;
;   Spinlock: test-and-test-and-set on one i32 word at shard+0.
;     * acquire : cmpxchg 0->1 ACQUIRE (observes the prior holder's release =>
;                 that shard's arrays/counters are visible to us)
;     * spin    : plain monotonic loads until 0 (no cmpxchg storm)
;     * release : store 0 RELEASE (publish our mutations to the next holder)
;   All shard fields are plain memory, ordered solely by this lock. Chosen over
;   a mutex because critical sections are a handful of stores; a futex syscall
;   would dwarf the work.
;
;   ecount() takes EVERY shard lock in ASCENDING index order (one fixed global
;   order => deadlock-free with add_edge's lower-first rule), sums, releases.
;
;   Shard header (128 B, one line):  lock@0(i32)  vhead@8  vtail@16  vdeg@24
;     edges@32  ecount@40  ecap@48  nlocal@56  [pad -> 128]
;   Map header  (128 B):  nshards@0  shardmask@8  shardbits@16  nverts@24
;     directed@32  [pad -> 128] ; shards[] at +128, stride 128 (posix_memalign
;     128 => every shard lock sits alone on its line => no false sharing).
;
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX):
;   ptr universe_ds_graph_sharded_create(i64 nverts, i64 nshards, i32 directed)
;   void universe_ds_graph_sharded_destroy(ptr m)
;   i32 universe_ds_graph_sharded_add_edge(ptr m, i64 u, i64 v, i64 w)
;   i64 universe_ds_graph_sharded_degree(ptr m, i64 u)             ; -1 bad idx
;   i32 universe_ds_graph_sharded_has_edge(ptr m, i64 u, i64 v)    ; 1/0
;   i64 universe_ds_graph_sharded_neighbors(ptr m, i64 u, ptr out, i64 max)
;   i64 universe_ds_graph_sharded_ecount(ptr m)     ; total edge records
;   i64 universe_ds_graph_sharded_vcount(ptr m)
;   i64 universe_ds_graph_sharded_shards(ptr m)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; internal helpers
; ===========================================================================

; test-and-test-and-set spinlock acquire (lock word @ shard+0).
define internal void @gs_lock(ptr %sh) #3 {
entry:
  br label %spin
spin:
  %v = load atomic i32, ptr %sh monotonic, align 4
  %free = icmp eq i32 %v, 0
  br i1 %free, label %try, label %spin
try:
  %cx = cmpxchg weak ptr %sh, i32 0, i32 1 acquire monotonic
  %ok = extractvalue { i32, i1 } %cx, 1
  br i1 %ok, label %got, label %spin
got:
  ret void
}

define internal void @gs_unlock(ptr %sh) #3 {
entry:
  store atomic i32 0, ptr %sh release, align 4
  ret void
}

; Ensure the shard has >=1 free edge slot (grow 2x). 0 OK / 2 OOM / 3 OVERFLOW.
define internal i32 @gs_ensure(ptr %sh) #4 {
entry:
  %ec.p = getelementptr inbounds nuw i8, ptr %sh, i64 40
  %ec = load i64, ptr %ec.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 48
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %ec, %cap
  br i1 %full, label %grow, label %ok, !prof !0
grow:
  %cap2 = shl i64 %cap, 1
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 16)
  %bytes = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %wrap = icmp eq i64 %cap2, 0
  %bad = or i1 %o, %wrap
  br i1 %bad, label %ovf, label %do, !prof !0
do:
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %edges0 = load ptr, ptr %e.p, align 8
  %edges = call ptr @realloc(ptr %edges0, i64 %bytes)
  %e.null = icmp eq ptr %edges, null
  br i1 %e.null, label %oom, label %store, !prof !0
store:
  store ptr %edges, ptr %e.p, align 8
  store i64 %cap2, ptr %cap.p, align 8
  ret i32 0
ok:
  ret i32 0
oom:
  ret i32 2
ovf:
  ret i32 3
}

; Append one directed record u->v (weight w) into %sh. Caller holds the lock
; and ensured edge room. Local index of u is u >> bits.
define internal void @gs_link(ptr %sh, i64 %bits, i64 %u, i64 %v, i64 %w) #4 {
entry:
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %edges = load ptr, ptr %e.p, align 8
  %ec.p = getelementptr inbounds nuw i8, ptr %sh, i64 40
  %e = load i64, ptr %ec.p, align 8
  %e32 = trunc i64 %e to i32
  %v32 = trunc i64 %v to i32
  %recoff = shl nuw i64 %e, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %recoff
  store i32 %v32, ptr %rec, align 4
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  store i32 -1, ptr %np, align 4
  %wp = getelementptr inbounds nuw i8, ptr %rec, i64 8
  store i64 %w, ptr %wp, align 8
  %lu = lshr i64 %u, %bits
  %luoff = shl nuw i64 %lu, 2
  %vh.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %vhead = load ptr, ptr %vh.p, align 8
  %vt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  %vtail = load ptr, ptr %vt.p, align 8
  %vd.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %vdeg = load ptr, ptr %vd.p, align 8
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %luoff
  %tp = getelementptr inbounds nuw i8, ptr %vtail, i64 %luoff
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
  %trec = getelementptr inbounds nuw i8, ptr %edges, i64 %toff
  %tnp = getelementptr inbounds nuw i8, ptr %trec, i64 4
  store i32 %e32, ptr %tnp, align 4
  store i32 %e32, ptr %tp, align 4
  br label %bump
bump:
  %dp = getelementptr inbounds nuw i8, ptr %vdeg, i64 %luoff
  %d = load i32, ptr %dp, align 4
  %d1 = add i32 %d, 1
  store i32 %d1, ptr %dp, align 4
  %e1 = add nuw i64 %e, 1
  store i64 %e1, ptr %ec.p, align 8
  ret void
}

; ===========================================================================
; create
; ===========================================================================
define noalias ptr @universe_ds_graph_sharded_create(i64 %nverts, i64 %nshards, i32 %directed) local_unnamed_addr #1 {
entry:
  ; nshards: 0 => default 16; round up to pow2; clamp [1, 65536]
  %ns.zero = icmp eq i64 %nshards, 0
  %ns.req = select i1 %ns.zero, i64 16, i64 %nshards
  %ns.cl = call i64 @llvm.umin.i64(i64 %ns.req, i64 65536)
  %ns0 = call i64 @llvm.umax.i64(i64 %ns.cl, i64 1)
  %nsm1 = add i64 %ns0, -1
  %nslz = call i64 @llvm.ctlz.i64(i64 %nsm1, i1 false)
  %nsbits = sub nuw nsw i64 64, %nslz
  %nsh = shl nuw i64 1, %nsbits
  %nsmask = add i64 %nsh, -1
  ; top block bytes = 128 + nsh*128
  %bb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nsh, i64 128)
  %bb.v = extractvalue { i64, i1 } %bb, 0
  %bb.o = extractvalue { i64, i1 } %bb, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bb.v, i64 128)
  %tt.v = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %ovf = or i1 %bb.o, %tt.o
  br i1 %ovf, label %fail0, label %alloc, !prof !0
alloc:
  %slot = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot, i64 128, i64 %tt.v)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %fail0, label %chk, !prof !0
chk:
  %mem = load ptr, ptr %slot, align 8
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail0, label %init, !prof !0
init:
  call void @llvm.memset.p0.i64(ptr %mem, i8 0, i64 %tt.v, i1 false)
  store i64 %nsh, ptr %mem, align 8
  %smask.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %nsmask, ptr %smask.p, align 8
  %sbits.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %nsbits, ptr %sbits.p, align 8
  %nv.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %nverts, ptr %nv.p, align 8
  %dir64 = zext i32 %directed to i64
  %dir.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i64 %dir64, ptr %dir.p, align 8
  ; base = nverts >> bits ; rem = nverts & mask
  %base = lshr i64 %nverts, %nsbits
  %rem = and i64 %nverts, %nsmask
  br label %shloop
shloop:
  %s = phi i64 [ 0, %init ], [ %s.n, %shcont ]
  %hasextra = icmp ult i64 %s, %rem
  %extra = zext i1 %hasextra to i64
  %nlocal = add i64 %base, %extra
  %metacount = call i64 @llvm.umax.i64(i64 %nlocal, i64 1)
  %mbytes = shl nuw i64 %metacount, 2
  %shoff = shl nuw i64 %s, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %mem, i64 %shoff2
  %vhead = call ptr @malloc(i64 %mbytes)
  %vh.null = icmp eq ptr %vhead, null
  br i1 %vh.null, label %cleanup, label %s.vt, !prof !0
s.vt:
  %vh.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  store ptr %vhead, ptr %vh.p, align 8
  %vtail = call ptr @malloc(i64 %mbytes)
  %vt.null = icmp eq ptr %vtail, null
  br i1 %vt.null, label %cleanup, label %s.vd, !prof !0
s.vd:
  %vt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  store ptr %vtail, ptr %vt.p, align 8
  %vdeg = call ptr @malloc(i64 %mbytes)
  %vd.null = icmp eq ptr %vdeg, null
  br i1 %vd.null, label %cleanup, label %s.e, !prof !0
s.e:
  %vd.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  store ptr %vdeg, ptr %vd.p, align 8
  %edges = call ptr @malloc(i64 256)
  %e.null = icmp eq ptr %edges, null
  br i1 %e.null, label %cleanup, label %s.fill, !prof !0
s.fill:
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  store ptr %edges, ptr %e.p, align 8
  call void @llvm.memset.p0.i64(ptr %vhead, i8 -1, i64 %mbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %vtail, i8 -1, i64 %mbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %vdeg, i8 0, i64 %mbytes, i1 false)
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 48
  store i64 16, ptr %cap.p, align 8
  %nl.p = getelementptr inbounds nuw i8, ptr %sh, i64 56
  store i64 %nlocal, ptr %nl.p, align 8
  br label %shcont
shcont:
  %s.n = add nuw i64 %s, 1
  %more = icmp ult i64 %s.n, %nsh
  br i1 %more, label %shloop, label %done
done:
  ret ptr %mem
cleanup:
  br label %cl.loop
cl.loop:
  %ci = phi i64 [ 0, %cleanup ], [ %ci.n, %cl.body ]
  %cdone = icmp ugt i64 %ci, %s              ; free shards 0..s inclusive
  br i1 %cdone, label %cl.free, label %cl.body
cl.body:
  %coff = shl nuw i64 %ci, 7
  %coff2 = add nuw i64 %coff, 128
  %csh = getelementptr inbounds nuw i8, ptr %mem, i64 %coff2
  %cvh.p = getelementptr inbounds nuw i8, ptr %csh, i64 8
  %cvh = load ptr, ptr %cvh.p, align 8
  call void @free(ptr %cvh)
  %cvt.p = getelementptr inbounds nuw i8, ptr %csh, i64 16
  %cvt = load ptr, ptr %cvt.p, align 8
  call void @free(ptr %cvt)
  %cvd.p = getelementptr inbounds nuw i8, ptr %csh, i64 24
  %cvd = load ptr, ptr %cvd.p, align 8
  call void @free(ptr %cvd)
  %ce.p = getelementptr inbounds nuw i8, ptr %csh, i64 32
  %ce = load ptr, ptr %ce.p, align 8
  call void @free(ptr %ce)
  %ci.n = add nuw i64 %ci, 1
  br label %cl.loop
cl.free:
  call void @free(ptr nonnull %mem)
  br label %fail0
fail0:
  ret ptr null
}

; ===========================================================================
; add_edge
; ===========================================================================
define i32 @universe_ds_graph_sharded_add_edge(ptr %m, i64 %u, i64 %v, i64 %w) local_unnamed_addr #0 {
entry:
  %rcout = alloca i32, align 4
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %check, !prof !0
err.null:
  ret i32 1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %nv = load i64, ptr %nv.p, align 8
  %ubad = icmp uge i64 %u, %nv
  %vbad = icmp uge i64 %v, %nv
  %bad = or i1 %ubad, %vbad
  br i1 %bad, label %err.idx, label %route, !prof !0
err.idx:
  ret i32 7
route:
  %mask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %mask = load i64, ptr %mask.p, align 8
  %bits.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %bits = load i64, ptr %bits.p, align 8
  %dir.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %dir = load i64, ptr %dir.p, align 8
  %su = and i64 %u, %mask
  %sv = and i64 %v, %mask
  %suoff = shl nuw i64 %su, 7
  %suoff2 = add nuw i64 %suoff, 128
  %shu = getelementptr inbounds nuw i8, ptr %m, i64 %suoff2
  %svoff = shl nuw i64 %sv, 7
  %svoff2 = add nuw i64 %svoff, 128
  %shv = getelementptr inbounds nuw i8, ptr %m, i64 %svoff2
  %isdir = icmp ne i64 %dir, 0
  br i1 %isdir, label %directed, label %undirected

directed:
  call void @gs_lock(ptr %shu)
  %drc = call i32 @gs_ensure(ptr %shu)
  %dok = icmp eq i32 %drc, 0
  br i1 %dok, label %d.link, label %d.fail
d.link:
  call void @gs_link(ptr %shu, i64 %bits, i64 %u, i64 %v, i64 %w)
  call void @gs_unlock(ptr %shu)
  ret i32 0
d.fail:
  call void @gs_unlock(ptr %shu)
  ret i32 %drc

undirected:
  %lo = call i64 @llvm.umin.i64(i64 %su, i64 %sv)
  %hi = call i64 @llvm.umax.i64(i64 %su, i64 %sv)
  %looff = shl nuw i64 %lo, 7
  %looff2 = add nuw i64 %looff, 128
  %shlo = getelementptr inbounds nuw i8, ptr %m, i64 %looff2
  %hioff = shl nuw i64 %hi, 7
  %hioff2 = add nuw i64 %hioff, 128
  %shhi = getelementptr inbounds nuw i8, ptr %m, i64 %hioff2
  %diff = icmp ne i64 %lo, %hi
  call void @gs_lock(ptr %shlo)
  br i1 %diff, label %lockhi, label %body
lockhi:
  call void @gs_lock(ptr %shhi)
  br label %body
body:
  %rc1 = call i32 @gs_ensure(ptr %shu)
  %ok1 = icmp eq i32 %rc1, 0
  br i1 %ok1, label %link1, label %ufail1
link1:
  call void @gs_link(ptr %shu, i64 %bits, i64 %u, i64 %v, i64 %w)
  %rc2 = call i32 @gs_ensure(ptr %shv)
  %ok2 = icmp eq i32 %rc2, 0
  br i1 %ok2, label %link2, label %ufail2
link2:
  call void @gs_link(ptr %shv, i64 %bits, i64 %v, i64 %u, i64 %w)
  store i32 0, ptr %rcout, align 4
  br label %unlock
ufail1:
  store i32 %rc1, ptr %rcout, align 4
  br label %unlock
ufail2:
  store i32 %rc2, ptr %rcout, align 4
  br label %unlock
unlock:
  br i1 %diff, label %unlockhi, label %unlocklo
unlockhi:
  call void @gs_unlock(ptr %shhi)
  br label %unlocklo
unlocklo:
  call void @gs_unlock(ptr %shlo)
  %rcf = load i32, ptr %rcout, align 4
  ret i32 %rcf
}

; ===========================================================================
; degree / has_edge / neighbors
; ===========================================================================
define i64 @universe_ds_graph_sharded_degree(ptr %m, i64 %u) local_unnamed_addr #0 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err, label %check, !prof !0
err:
  ret i64 -1
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %nv = load i64, ptr %nv.p, align 8
  %bad = icmp uge i64 %u, %nv
  br i1 %bad, label %err, label %go, !prof !0
go:
  %mask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %mask = load i64, ptr %mask.p, align 8
  %bits.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %bits = load i64, ptr %bits.p, align 8
  %su = and i64 %u, %mask
  %shoff = shl nuw i64 %su, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  call void @gs_lock(ptr %sh)
  %lu = lshr i64 %u, %bits
  %luoff = shl nuw i64 %lu, 2
  %vd.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %vdeg = load ptr, ptr %vd.p, align 8
  %dp = getelementptr inbounds nuw i8, ptr %vdeg, i64 %luoff
  %d = load i32, ptr %dp, align 4
  call void @gs_unlock(ptr %sh)
  %d64 = zext i32 %d to i64
  ret i64 %d64
}

define i32 @universe_ds_graph_sharded_has_edge(ptr %m, i64 %u, i64 %v) local_unnamed_addr #0 {
entry:
  %e = alloca i32, align 4
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %no.early, label %check, !prof !0
no.early:
  ret i32 0
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %nv = load i64, ptr %nv.p, align 8
  %ubad = icmp uge i64 %u, %nv
  %vbad = icmp uge i64 %v, %nv
  %bad = or i1 %ubad, %vbad
  br i1 %bad, label %no.early, label %go, !prof !0
go:
  %mask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %mask = load i64, ptr %mask.p, align 8
  %bits.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %bits = load i64, ptr %bits.p, align 8
  %su = and i64 %u, %mask
  %shoff = shl nuw i64 %su, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  call void @gs_lock(ptr %sh)
  %lu = lshr i64 %u, %bits
  %luoff = shl nuw i64 %lu, 2
  %vh.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %vhead = load ptr, ptr %vh.p, align 8
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %edges = load ptr, ptr %e.p, align 8
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %luoff
  %h = load i32, ptr %hp, align 4
  store i32 %h, ptr %e, align 4
  %v32 = trunc i64 %v to i32
  br label %loop
loop:
  %ei = load i32, ptr %e, align 4
  %end = icmp eq i32 %ei, -1
  br i1 %end, label %miss, label %bodyb
bodyb:
  %ei64 = zext i32 %ei to i64
  %roff = shl nuw i64 %ei64, 4
  %rec = getelementptr inbounds nuw i8, ptr %edges, i64 %roff
  %dst = load i32, ptr %rec, align 4
  %hit = icmp eq i32 %dst, %v32
  br i1 %hit, label %found, label %next
next:
  %np = getelementptr inbounds nuw i8, ptr %rec, i64 4
  %nx = load i32, ptr %np, align 4
  store i32 %nx, ptr %e, align 4
  br label %loop
found:
  call void @gs_unlock(ptr %sh)
  ret i32 1
miss:
  call void @gs_unlock(ptr %sh)
  ret i32 0
}

define i64 @universe_ds_graph_sharded_neighbors(ptr %m, i64 %u, ptr %out, i64 %max) local_unnamed_addr #0 {
entry:
  %e = alloca i32, align 4
  %k = alloca i64, align 8
  %m.null = icmp eq ptr %m, null
  %o.null = icmp eq ptr %out, null
  %bad0 = or i1 %m.null, %o.null
  br i1 %bad0, label %zero.early, label %check, !prof !0
zero.early:
  ret i64 0
check:
  %nv.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %nv = load i64, ptr %nv.p, align 8
  %bad = icmp uge i64 %u, %nv
  br i1 %bad, label %zero.early, label %go, !prof !0
go:
  %mask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %mask = load i64, ptr %mask.p, align 8
  %bits.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %bits = load i64, ptr %bits.p, align 8
  %su = and i64 %u, %mask
  %shoff = shl nuw i64 %su, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  call void @gs_lock(ptr %sh)
  %lu = lshr i64 %u, %bits
  %luoff = shl nuw i64 %lu, 2
  %vh.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %vhead = load ptr, ptr %vh.p, align 8
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %edges = load ptr, ptr %e.p, align 8
  %hp = getelementptr inbounds nuw i8, ptr %vhead, i64 %luoff
  %h = load i32, ptr %hp, align 4
  store i32 %h, ptr %e, align 4
  store i64 0, ptr %k, align 8
  br label %loop
loop:
  %ei = load i32, ptr %e, align 4
  %end = icmp eq i32 %ei, -1
  br i1 %end, label %done, label %chkmax
chkmax:
  %kk = load i64, ptr %k, align 8
  %atmax = icmp uge i64 %kk, %max
  br i1 %atmax, label %done, label %bodyb
bodyb:
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
  call void @gs_unlock(ptr %sh)
  %kf = load i64, ptr %k, align 8
  ret i64 %kf
}

; ===========================================================================
; ecount — lock ALL shards ascending, sum records, release.
; ===========================================================================
define i64 @universe_ds_graph_sharded_ecount(ptr %m) local_unnamed_addr #0 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %null, label %setup, !prof !0
null:
  ret i64 0
setup:
  %nsh = load i64, ptr %m, align 8
  br label %lock.loop
lock.loop:
  %li = phi i64 [ 0, %setup ], [ %li.n, %lock.loop ]
  %loff = shl nuw i64 %li, 7
  %loff2 = add nuw i64 %loff, 128
  %lsh = getelementptr inbounds nuw i8, ptr %m, i64 %loff2
  call void @gs_lock(ptr %lsh)
  %li.n = add nuw i64 %li, 1
  %lmore = icmp ult i64 %li.n, %nsh
  br i1 %lmore, label %lock.loop, label %sum.loop
sum.loop:
  %si = phi i64 [ 0, %lock.loop ], [ %si.n, %sum.loop ]
  %acc = phi i64 [ 0, %lock.loop ], [ %acc.n, %sum.loop ]
  %soff = shl nuw i64 %si, 7
  %soff2 = add nuw i64 %soff, 128
  %ssh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %sc.p = getelementptr inbounds nuw i8, ptr %ssh, i64 40
  %sc = load i64, ptr %sc.p, align 8
  %acc.n = add i64 %acc, %sc
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, %nsh
  br i1 %smore, label %sum.loop, label %unlock.loop
unlock.loop:
  %ui = phi i64 [ 0, %sum.loop ], [ %ui.n, %unlock.loop ]
  %uoff = shl nuw i64 %ui, 7
  %uoff2 = add nuw i64 %uoff, 128
  %ush = getelementptr inbounds nuw i8, ptr %m, i64 %uoff2
  call void @gs_unlock(ptr %ush)
  %ui.n = add nuw i64 %ui, 1
  %umore = icmp ult i64 %ui.n, %nsh
  br i1 %umore, label %unlock.loop, label %fin
fin:
  ret i64 %acc.n
}

define i64 @universe_ds_graph_sharded_vcount(ptr %m) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %z, label %ok, !prof !0
z:
  ret i64 0
ok:
  %nv.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %nv = load i64, ptr %nv.p, align 8
  ret i64 %nv
}

define i64 @universe_ds_graph_sharded_shards(ptr %m) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %z, label %ok, !prof !0
z:
  ret i64 0
ok:
  %nsh = load i64, ptr %m, align 8
  ret i64 %nsh
}

; ===========================================================================
; destroy
; ===========================================================================
define void @universe_ds_graph_sharded_destroy(ptr %m) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %ret, label %setup, !prof !0
setup:
  %nsh = load i64, ptr %m, align 8
  br label %loop
loop:
  %si = phi i64 [ 0, %setup ], [ %si.n, %loop ]
  %soff = shl nuw i64 %si, 7
  %soff2 = add nuw i64 %soff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %vh.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %vh = load ptr, ptr %vh.p, align 8
  call void @free(ptr %vh)
  %vt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  %vt = load ptr, ptr %vt.p, align 8
  call void @free(ptr %vt)
  %vd.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %vd = load ptr, ptr %vd.p, align 8
  call void @free(ptr %vd)
  %e.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %ed = load ptr, ptr %e.p, align 8
  call void @free(ptr %ed)
  %si.n = add nuw i64 %si, 1
  %more = icmp ult i64 %si.n, %nsh
  br i1 %more, label %loop, label %freeblk
freeblk:
  call void @free(ptr nonnull %m)
  br label %ret
ret:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #3 = { alwaysinline nounwind norecurse }
attributes #4 = { alwaysinline nounwind norecurse }

!0 = !{!"branch_weights", i32 1, i32 2000}
