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

; universe_ds_btree_sharded — RANGE-sharded concurrent B-tree.
;
; ============================================================================
; DESIGN
; ----------------------------------------------------------------------------
;   A single lock over one B-tree serializes every writer and its lock line
;   ping-pongs across cores. We STRIPE: N (power-of-two) independent shards,
;   each a full array-backed B-tree (REUSED from btree.ll — same domain, linked
;   directly; the hot leaf is the monomorphic universe_ds_btree_* engine, no
;   vtable), guarded by its OWN test-and-test-and-set spinlock on its OWN 128 B
;   cache-line pair. Point ops lock exactly one shard, so threads touching keys
;   in different shards proceed fully in parallel.
;
;   RANGE-SHARD MAPPING (why range/floor/ceiling/min/max stay globally ordered):
;   the key space is partitioned by the HIGH bits of the key, NOT by a hash.
;   Map the SIGNED key to an order-preserving unsigned key by flipping the sign
;   bit: ukey = key XOR 0x8000000000000000 (this makes INT64_MIN -> 0 and
;   INT64_MAX -> UINT64_MAX, monotonic). Then
;        shard = ukey >> (64 - shardbits)        (shardbits = log2(N))
;   so shard indices increase monotonically with the key: EVERY key in shard i
;   is strictly less than every key in shard i+1. Consequences:
;     * a global in-order walk = shards 0,1,...,N-1, each B-tree walked in-order
;       (already sorted) => concatenation is globally sorted;
;     * min = smallest non-empty shard's min; max = largest non-empty shard's
;       max; floor/ceiling that miss in the home shard step to the adjacent
;       lower/higher shard;
;     * range(lo,hi) touches only shards route(lo)..route(hi), each queried with
;       the SAME [lo,hi] and its results concatenated in shard order.
;   (A hash-shard would destroy all of this; range-sharding is the whole point.)
;
;   Ordered / global ops (len, min, max, floor, ceiling, range) take EVERY shard
;   lock in ASCENDING index order (one fixed global order => deadlock-free) for a
;   consistent snapshot, then release. They are cold; the striped point-op fast
;   path is what scales.
;
;   Locking: per-shard TTAS spinlock (one i32 word) — critical sections are a
;   single B-tree op, non-blocking; a futex/mutex syscall would dwarf them.
;     acquire: cmpxchg 0->1 ACQUIRE (observes prior holder's release => the
;              shard's tree is visible); spin on monotonic loads (no cmpxchg
;              storm); release: store 0 RELEASE.
;
;   Shard record (128 B, isolated line pair): lock@0(i32) tree@8(ptr) [pad->128]
;   Map header: nshards@0(i64) shardbits@8(i64) ; shards[] at +128, stride 128.
;   posix_memalign(128) => shard 0 is 128-aligned; consecutive locks are exactly
;   128 B apart => never false-share.
;
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY, 5 NOT_FOUND):
;   ptr  universe_ds_btree_sharded_create(i64 nshards)
;   void universe_ds_btree_sharded_destroy(ptr)
;   i32  universe_ds_btree_sharded_put(ptr, i64 key, i64 val)
;   i32  universe_ds_btree_sharded_get(ptr, i64 key, ptr outval)
;   i32  universe_ds_btree_sharded_delete(ptr, i64 key)
;   i32  universe_ds_btree_sharded_contains(ptr, i64 key)
;   i64  universe_ds_btree_sharded_len(ptr)            ; consistent snapshot
;   i32  universe_ds_btree_sharded_min(ptr, ptr ok, ptr ov)
;   i32  universe_ds_btree_sharded_max(ptr, ptr ok, ptr ov)
;   i32  universe_ds_btree_sharded_floor(ptr, i64 key, ptr ok, ptr ov)
;   i32  universe_ds_btree_sharded_ceiling(ptr, i64 key, ptr ok, ptr ov)
;   i64  universe_ds_btree_sharded_range(ptr, i64 lo, i64 hi, ptr ok, ptr ov, i64 max)
;   i64  universe_ds_btree_sharded_shards(ptr)
; ============================================================================

declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; reused array-btree engine (same domain, linked)
declare ptr @universe_ds_btree_create()
declare void @universe_ds_btree_destroy(ptr)
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_delete(ptr, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)
declare i32 @universe_ds_btree_min(ptr, ptr, ptr)
declare i32 @universe_ds_btree_max(ptr, ptr, ptr)
declare i32 @universe_ds_btree_floor(ptr, i64, ptr, ptr)
declare i32 @universe_ds_btree_ceiling(ptr, i64, ptr, ptr)
declare i64 @universe_ds_btree_range(ptr, i64, i64, ptr, ptr, i64)
declare i64 @universe_ds_btree_count(ptr)

; ===========================================================================
; spinlock + routing
; ===========================================================================
define internal void @bts_lock(ptr %sh) #2 {
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

define internal void @bts_unlock(ptr %sh) #2 {
entry:
  store atomic i32 0, ptr %sh release, align 4
  ret void
}

; Map a signed key to its shard index via order-preserving high-bit split.
define internal i64 @bts_route(ptr %m, i64 %key) #3 {
entry:
  %sbp = getelementptr inbounds nuw i8, ptr %m, i64 8
  %shardbits = load i64, ptr %sbp, align 8
  %ukey = xor i64 %key, -9223372036854775808
  %samt = sub i64 64, %shardbits
  %samt.g = call i64 @llvm.umin.i64(i64 %samt, i64 63)
  %sh0 = lshr i64 %ukey, %samt.g
  %z = icmp eq i64 %shardbits, 0
  %shard = select i1 %z, i64 0, i64 %sh0
  ret i64 %shard
}

; Lock every shard in ascending index order (fixed global order => no deadlock).
define internal void @bts_lock_all(ptr %m) #4 {
entry:
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  call void @bts_lock(ptr %sh)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %loop, label %done

done:
  ret void
}

define internal void @bts_unlock_all(ptr %m) #4 {
entry:
  %nsh = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  call void @bts_unlock(ptr %sh)
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_btree_sharded_create(i64 %nshards) local_unnamed_addr #1 {
entry:
  %ns.zero = icmp eq i64 %nshards, 0
  %ns.req = select i1 %ns.zero, i64 16, i64 %nshards
  %ns.cl = call i64 @llvm.umin.i64(i64 %ns.req, i64 4096)
  %ns0 = call i64 @llvm.umax.i64(i64 %ns.cl, i64 1)
  %nsm1 = add i64 %ns0, -1
  %nslz = call i64 @llvm.ctlz.i64(i64 %nsm1, i1 false)
  %shardbits = sub nuw nsw i64 64, %nslz
  %nsh = shl nuw i64 1, %shardbits
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
  %sbp = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %shardbits, ptr %sbp, align 8
  br label %shloop

shloop:
  %si = phi i64 [ 0, %init ], [ %si.n, %shcont ]
  %off = shl nuw i64 %si, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %mem, i64 %off2
  %tree = call ptr @universe_ds_btree_create()
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
  call void @universe_ds_btree_destroy(ptr %ctree)
  %ci.n = add nuw i64 %ci, 1
  br label %cl.loop

cl.free:
  call void @free(ptr nonnull %mem)
  br label %fail0

fail0:
  ret ptr null
}

define void @universe_ds_btree_sharded_destroy(ptr %m) local_unnamed_addr #1 {
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
  call void @universe_ds_btree_destroy(ptr %tree)
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
define i32 @universe_ds_btree_sharded_put(ptr %m, i64 %key, i64 %val) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @bts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_insert(ptr %tree, i64 %key, i64 %val)
  call void @bts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_btree_sharded_get(ptr %m, i64 %key, ptr %out) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %on = icmp eq ptr %out, null
  %bad = or i1 %mn, %on
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @bts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_find(ptr %tree, i64 %key, ptr %out)
  call void @bts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_btree_sharded_delete(ptr %m, i64 %key) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @bts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_delete(ptr %tree, i64 %key)
  call void @bts_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_btree_sharded_contains(ptr %m, i64 %key) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %no, label %go, !prof !0

no:
  ret i32 0

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  %off = shl nuw i64 %s, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  call void @bts_lock(ptr %sh)
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_find(ptr %tree, i64 %key, ptr null)
  call void @bts_unlock(ptr %sh)
  %f = icmp eq i32 %rc, 0
  %z = zext i1 %f to i32
  ret i32 %z
}

; ===========================================================================
; len — lock all ascending, sum counts, unlock
; ===========================================================================
define i64 @universe_ds_btree_sharded_len(ptr %m) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  call void @bts_lock_all(ptr %m)
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
  %c = call i64 @universe_ds_btree_count(ptr %tree)
  %acc.n = add i64 %acc, %c
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %sum, label %fin

fin:
  call void @bts_unlock_all(ptr %m)
  ret i64 %acc.n
}

define i64 @universe_ds_btree_sharded_shards(ptr %m) local_unnamed_addr #5 {
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
; min — smallest non-empty shard's min (ascending scan)
; ===========================================================================
define i32 @universe_ds_btree_sharded_min(ptr %m, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %mn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  call void @bts_lock_all(ptr %m)
  %nsh = load i64, ptr %m, align 8
  br label %scan

scan:
  %i = phi i64 [ 0, %go ], [ %in, %next ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_min(ptr %tree, ptr %ok, ptr %ov)
  %hit = icmp eq i32 %rc, 0
  br i1 %hit, label %found, label %next

next:
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %scan, label %empty

found:
  call void @bts_unlock_all(ptr %m)
  ret i32 0

empty:
  call void @bts_unlock_all(ptr %m)
  ret i32 4
}

; ===========================================================================
; max — largest non-empty shard's max (descending scan)
; ===========================================================================
define i32 @universe_ds_btree_sharded_max(ptr %m, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %mn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  call void @bts_lock_all(ptr %m)
  %nsh = load i64, ptr %m, align 8
  %start = sub i64 %nsh, 1
  br label %scan

scan:
  %i = phi i64 [ %start, %go ], [ %in, %dec ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_max(ptr %tree, ptr %ok, ptr %ov)
  %hit = icmp eq i32 %rc, 0
  br i1 %hit, label %found, label %next

next:
  %atzero = icmp eq i64 %i, 0
  br i1 %atzero, label %empty, label %dec

dec:
  %in = sub i64 %i, 1
  br label %scan

found:
  call void @bts_unlock_all(ptr %m)
  ret i32 0

empty:
  call void @bts_unlock_all(ptr %m)
  ret i32 4
}

; ===========================================================================
; floor — home shard, else max of the highest non-empty shard below it
; ===========================================================================
define i32 @universe_ds_btree_sharded_floor(ptr %m, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %mn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  call void @bts_lock_all(ptr %m)
  %soff = shl nuw i64 %s, 7
  %soff2 = add nuw i64 %soff, 128
  %ssh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %stp = getelementptr inbounds nuw i8, ptr %ssh, i64 8
  %stree = load ptr, ptr %stp, align 8
  %frc = call i32 @universe_ds_btree_floor(ptr %stree, i64 %key, ptr %ok, ptr %ov)
  %fhit = icmp eq i32 %frc, 0
  br i1 %fhit, label %found, label %below

below:
  %atzero = icmp eq i64 %s, 0
  br i1 %atzero, label %none, label %scan.pre

scan.pre:
  %start = sub i64 %s, 1
  br label %scan

scan:
  %i = phi i64 [ %start, %scan.pre ], [ %in, %dec ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_max(ptr %tree, ptr %ok, ptr %ov)
  %hit = icmp eq i32 %rc, 0
  br i1 %hit, label %found, label %next

next:
  %atz = icmp eq i64 %i, 0
  br i1 %atz, label %none, label %dec

dec:
  %in = sub i64 %i, 1
  br label %scan

found:
  call void @bts_unlock_all(ptr %m)
  ret i32 0

none:
  call void @bts_unlock_all(ptr %m)
  ret i32 5
}

; ===========================================================================
; ceiling — home shard, else min of the lowest non-empty shard above it
; ===========================================================================
define i32 @universe_ds_btree_sharded_ceiling(ptr %m, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %mn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %s = call i64 @bts_route(ptr %m, i64 %key)
  call void @bts_lock_all(ptr %m)
  %nsh = load i64, ptr %m, align 8
  %soff = shl nuw i64 %s, 7
  %soff2 = add nuw i64 %soff, 128
  %ssh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %stp = getelementptr inbounds nuw i8, ptr %ssh, i64 8
  %stree = load ptr, ptr %stp, align 8
  %crc = call i32 @universe_ds_btree_ceiling(ptr %stree, i64 %key, ptr %ok, ptr %ov)
  %chit = icmp eq i32 %crc, 0
  br i1 %chit, label %found, label %above

above:
  %start = add nuw i64 %s, 1
  %hasabove = icmp ult i64 %start, %nsh
  br i1 %hasabove, label %scan, label %none

scan:
  %i = phi i64 [ %start, %above ], [ %in, %next ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %rc = call i32 @universe_ds_btree_min(ptr %tree, ptr %ok, ptr %ov)
  %hit = icmp eq i32 %rc, 0
  br i1 %hit, label %found, label %next

next:
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %nsh
  br i1 %more, label %scan, label %none

found:
  call void @bts_unlock_all(ptr %m)
  ret i32 0

none:
  call void @bts_unlock_all(ptr %m)
  ret i32 5
}

; ===========================================================================
; range — query shards route(lo)..route(hi), concatenate in shard order
; ===========================================================================
define i64 @universe_ds_btree_sharded_range(ptr %m, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max) local_unnamed_addr #0 {
entry:
  %mn = icmp eq ptr %m, null
  br i1 %mn, label %z, label %chkorder, !prof !0

z:
  ret i64 0

chkorder:
  %bad = icmp sgt i64 %lo, %hi
  br i1 %bad, label %z, label %go, !prof !0

go:
  %slo = call i64 @bts_route(ptr %m, i64 %lo)
  %shi = call i64 @bts_route(ptr %m, i64 %hi)
  call void @bts_lock_all(ptr %m)
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  br label %loop

loop:
  %i = phi i64 [ %slo, %go ], [ %in, %loop ]
  %total = phi i64 [ 0, %go ], [ %total2, %loop ]
  %wrote = phi i64 [ 0, %go ], [ %wrote2, %loop ]
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  %tp = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tree = load ptr, ptr %tp, align 8
  %okg = getelementptr inbounds nuw i64, ptr %ok, i64 %wrote
  %okoff = select i1 %okn, ptr null, ptr %okg
  %ovg = getelementptr inbounds nuw i64, ptr %ov, i64 %wrote
  %ovoff = select i1 %ovn, ptr null, ptr %ovg
  %remain = sub i64 %max, %wrote
  %cnt = call i64 @universe_ds_btree_range(ptr %tree, i64 %lo, i64 %hi, ptr %okoff, ptr %ovoff, i64 %remain)
  %total2 = add i64 %total, %cnt
  %wrote2 = call i64 @llvm.umin.i64(i64 %total2, i64 %max)
  %in = add nuw i64 %i, 1
  %more = icmp ule i64 %in, %shi
  br i1 %more, label %loop, label %fin

fin:
  call void @bts_unlock_all(ptr %m)
  ret i64 %total2
}

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn }
attributes #2 = { alwaysinline nounwind norecurse }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #4 = { nounwind norecurse }
attributes #5 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
