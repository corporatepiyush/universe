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

; universe_ds_art_sharded — HASH-sharded concurrent adaptive radix tree.
;
; ============================================================================
; DESIGN
; ----------------------------------------------------------------------------
;   A single lock over one ART serializes every writer and its lock line
;   ping-pongs across cores. We STRIPE: N (power-of-two) independent shards,
;   each a full array-backed ART (REUSED from art.ll — same domain, linked
;   directly; the hot leaf is the monomorphic universe_ds_art_* engine, no
;   vtable), guarded by its OWN test-and-test-and-set spinlock on its OWN 128 B
;   cache-line pair. Point ops lock exactly ONE shard, so threads touching keys
;   in different shards proceed fully in parallel.
;
;   SHARD MAPPING — by a HASH of the WHOLE key (FNV-1a/64), so identical keys
;   always route to the same shard:  shard = hash(key) & (N-1). Hashing the full
;   key (not a byte-range) keeps routing correct even when many keys share a
;   prefix; it only affects load balance, never correctness.
;
;   ORDERED GLOBAL OPS ARE NOT SUPPORTED. Hash-sharding deliberately DESTROYS
;   global key order: two keys adjacent in lexicographic order land in unrelated
;   shards. Therefore there is NO global min/max/ordered-iterate/floor/ceiling.
;   (Use the range-sharded btree for ordered global queries.) What we DO offer:
;     * point ops (put/get/delete/contains) — one shard lock, scalable;
;     * len — a consistent snapshot (lock all shards ascending, sum, unlock);
;     * prefix_scan — locks all shards, scans EACH shard's subtree for the
;       prefix, concatenates results. Each shard's emission is locally ordered
;       but the CONCATENATION across shards is NOT globally sorted. It is
;       COMPLETE (every matching key is emitted exactly once) — correct for a
;       membership/collection scan, just not for an ordered walk.
;
;   Locking: per-shard TTAS spinlock (one i32 word) — critical sections are a
;   single ART op, non-blocking; a futex/mutex syscall would dwarf them.
;     acquire: spin on monotonic loads, then cmpxchg 0->1 ACQUIRE (observes the
;              prior holder's release => the shard's tree is visible);
;     release: store 0 RELEASE.
;   Global ops take EVERY shard lock in ASCENDING index order (one fixed order
;   => deadlock-free) for a consistent snapshot, then release.
;
;   Shard record (128 B, isolated line pair): lock@0(i32) tree@8(ptr) [pad->128]
;   Map header: nshards@0(i64) shardmask@8(i64) ; shards[] at +128, stride 128.
;   posix_memalign(128) => shard 0 is 128-aligned; consecutive locks are exactly
;   128 B apart => never false-share.
;
; API (0 OK, 1 NULL_PTR, 2 OOM, 5 NOT_FOUND):
;   ptr  universe_ds_art_sharded_create(i64 nshards)
;   void universe_ds_art_sharded_destroy(ptr)
;   i32  universe_ds_art_sharded_put(ptr, ptr key, i64 klen, i64 val)
;   i32  universe_ds_art_sharded_get(ptr, ptr key, i64 klen, ptr outval)
;   i32  universe_ds_art_sharded_delete(ptr, ptr key, i64 klen)
;   i32  universe_ds_art_sharded_contains(ptr, ptr key, i64 klen)  ; 1/0
;   i64  universe_ds_art_sharded_len(ptr)                          ; snapshot
;   i64  universe_ds_art_sharded_shards(ptr)
;   i64  universe_ds_art_sharded_prefix_scan(ptr, ptr pfx, i64 plen, ptr cb, ptr ctx)
; ============================================================================

declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; reused array-ART engine (same domain, linked)
declare ptr @universe_ds_art_create()
declare void @universe_ds_art_destroy(ptr)
declare i32 @universe_ds_art_insert(ptr, ptr, i64, i64)
declare i32 @universe_ds_art_get(ptr, ptr, i64, ptr)
declare i32 @universe_ds_art_delete(ptr, ptr, i64)
declare i64 @universe_ds_art_count(ptr)
declare i64 @universe_ds_art_prefix_scan(ptr, ptr, i64, ptr, ptr)

; ===========================================================================
; spinlock + routing
; ===========================================================================
define internal void @arts_lock(ptr %sh) #2 {
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

define internal void @arts_unlock(ptr %sh) #3 {
entry:
  store atomic i32 0, ptr %sh release, align 4
  ret void
}

; FNV-1a 64-bit hash of the whole key -> shard index (& mask)
define internal i64 @arts_route(ptr %m, ptr %key, i64 %klen) #4 {
entry:
  %maskp = getelementptr inbounds nuw i8, ptr %m, i64 8
  %mask = load i64, ptr %maskp, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %step ]
  %h = phi i64 [ -3750763034362895579, %entry ], [ %h2, %step ]
  %done = icmp uge i64 %i, %klen
  br i1 %done, label %fin, label %step

step:
  %bp = getelementptr inbounds nuw i8, ptr %key, i64 %i
  %b = load i8, ptr %bp, align 1
  %b64 = zext i8 %b to i64
  %x = xor i64 %h, %b64
  %h2 = mul i64 %x, 1099511628211
  %in = add nuw i64 %i, 1
  br label %loop

fin:
  %shard = and i64 %h, %mask
  ret i64 %shard
}

; lock/unlock all shards in ascending index order
define internal void @arts_lock_all(ptr %m) #5 {
entry:
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  call void @arts_lock(ptr %sh)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %loop, label %done

done:
  ret void
}

define internal void @arts_unlock_all(ptr %m) #5 {
entry:
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  call void @arts_unlock(ptr %sh)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_art_sharded_create(i64 %nshards) local_unnamed_addr #1 {
entry:
  %ns.zero = icmp eq i64 %nshards, 0
  %ns.req = select i1 %ns.zero, i64 16, i64 %nshards
  %ns.cl = call i64 @llvm.umin.i64(i64 %ns.req, i64 4096)
  %ns0 = call i64 @llvm.umax.i64(i64 %ns.cl, i64 1)
  %nsm1 = add i64 %ns0, -1
  %nslz = call i64 @llvm.ctlz.i64(i64 %nsm1, i1 false)
  %shardbits = sub nuw nsw i64 64, %nslz
  %nsh = shl nuw i64 1, %shardbits
  %mask = add i64 %nsh, -1
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
  %maskp = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %mask, ptr %maskp, align 8
  br label %shloop

shloop:
  %si = phi i64 [ 0, %init ], [ %si.n, %shcont ]
  %off = shl nuw i64 %si, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %mem, i64 %off2
  %tree = call ptr @universe_ds_art_create()
  %tree.null = icmp eq ptr %tree, null
  br i1 %tree.null, label %cleanup, label %shinit, !prof !0

shinit:
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  store ptr %tree, ptr %tp, align 8
  br label %shcont

shcont:
  %si.n = add nuw i64 %si, 1
  %more = icmp ult i64 %si.n, %nsh
  br i1 %more, label %shloop, label %done

done:
  ret ptr %mem

cleanup:
  br label %cl.loop

cl.loop:
  %ci = phi i64 [ 0, %cleanup ], [ %ci.n, %cl.body ]
  %cdone = icmp uge i64 %ci, %si
  br i1 %cdone, label %cl.free, label %cl.body

cl.body:
  %coff = shl nuw i64 %ci, 7
  %coff2 = add nuw i64 %coff, 128
  %csh = getelementptr inbounds nuw i8, ptr %mem, i64 %coff2
  %ctp = getelementptr inbounds nuw i8, ptr %csh, i64 8
  %ctree = load ptr, ptr %ctp, align 8
  call void @universe_ds_art_destroy(ptr %ctree)
  %ci.n = add nuw i64 %ci, 1
  br label %cl.loop

cl.free:
  call void @free(ptr nonnull %mem)
  br label %fail0

fail0:
  ret ptr null
}

define void @universe_ds_art_sharded_destroy(ptr %m) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %setup, !prof !0

setup:
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %si = phi i64 [ 0, %setup ], [ %si.n, %loop ]
  %off = shl nuw i64 %si, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  call void @universe_ds_art_destroy(ptr %tree)
  %si.n = add nuw i64 %si, 1
  %more = icmp ult i64 %si.n, %nsh
  br i1 %more, label %loop, label %freeblock

freeblock:
  call void @free(ptr nonnull %m)
  br label %ret

ret:
  ret void
}

; ===========================================================================
; point ops — one shard lock each
; ===========================================================================
define i32 @universe_ds_art_sharded_put(ptr %m, ptr %key, i64 %klen, i64 %val) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %mn, %kn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @arts_route(ptr %m, ptr %key, i64 %klen)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @arts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_art_insert(ptr %tree, ptr %key, i64 %klen, i64 %val)
  call void @arts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_art_sharded_get(ptr %m, ptr %key, i64 %klen, ptr %out) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %mn, %kn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @arts_route(ptr %m, ptr %key, i64 %klen)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @arts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_art_get(ptr %tree, ptr %key, i64 %klen, ptr %out)
  call void @arts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_art_sharded_delete(ptr %m, ptr %key, i64 %klen) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %mn, %kn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @arts_route(ptr %m, ptr %key, i64 %klen)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @arts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_art_delete(ptr %tree, ptr %key, i64 %klen)
  call void @arts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_art_sharded_contains(ptr %m, ptr %key, i64 %klen) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %mn, %kn
  br i1 %bad, label %no, label %go, !prof !0

no:
  ret i32 0

go:
  %s = call i64 @arts_route(ptr %m, ptr %key, i64 %klen)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @arts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_art_get(ptr %tree, ptr %key, i64 %klen, ptr null)
  call void @arts_unlock(ptr %sh)
  %f = icmp eq i32 %rc, 0
  %z = zext i1 %f to i32
  ret i32 %z
}

; ===========================================================================
; len — lock all ascending, sum counts, unlock (consistent snapshot)
; ===========================================================================
define i64 @universe_ds_art_sharded_len(ptr %m) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  call void @arts_lock_all(ptr %m)
  %nsh = load i64, ptr %m, align 8
  br label %sum

sum:
  %i = phi i64 [ 0, %go ], [ %in, %sum ]
  %acc = phi i64 [ 0, %go ], [ %acc.n, %sum ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %c = call i64 @universe_ds_art_count(ptr %tree)
  %acc.n = add i64 %acc, %c
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %sum, label %fin

fin:
  call void @arts_unlock_all(ptr %m)
  ret i64 %acc.n
}

define i64 @universe_ds_art_sharded_shards(ptr %m) local_unnamed_addr #6 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  %nsh = load i64, ptr %m, align 8
  ret i64 %nsh
}

; ===========================================================================
; prefix_scan — lock all, scan each shard, concatenate (NOT globally ordered).
; returns total matches across shards.
; ===========================================================================
define i64 @universe_ds_art_sharded_prefix_scan(ptr %m, ptr %pfx, i64 %plen, ptr %cb, ptr %ctx) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %cbn = icmp eq ptr %cb, null
  %bad = or i1 %mn, %cbn
  br i1 %bad, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  call void @arts_lock_all(ptr %m)
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %go ], [ %in, %loop ]
  %acc = phi i64 [ 0, %go ], [ %acc.n, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %c = call i64 @universe_ds_art_prefix_scan(ptr %tree, ptr %pfx, i64 %plen, ptr %cb, ptr %ctx)
  %acc.n = add i64 %acc, %c
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %loop, label %fin

fin:
  call void @arts_unlock_all(ptr %m)
  ret i64 %acc.n
}

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn }
attributes #2 = { alwaysinline nounwind norecurse }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #5 = { nounwind norecurse }
attributes #6 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
