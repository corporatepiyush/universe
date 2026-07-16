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

; Sharded ordered map — the CONCURRENT flavor of the treemap. N power-of-two
; shards, each an INDEPENDENT array-backed treemap (src/structures_assoc/
; treemap.ll, reused by direct linked call) guarded by its own 128 B-padded
; TTAS spinlock. Unlike a hash-sharded map, shards here are RANGE-partitioned
; so global KEY ORDER is preserved across shards — the whole point of an
; ordered map.
;
; WHEN TO CHOOSE THIS VARIANT:
;   * Concurrent ordered map: many threads doing point put/get/delete on
;     DISJOINT key ranges, plus occasional ordered queries (range/min/max/
;     floor/ceiling). A single-lock treemap serializes every writer and its one
;     lock line ping-pongs across cores; range-striping drops contention ~N×
;     while keeping the array treemap's cache-dense ordered scans intact.
;
; ============================================================================
; RANGE-SHARD MAPPING (monotone => global order preserved)
; ----------------------------------------------------------------------------
;   Keys are ordered UNSIGNED (same as treemap.ll). With N = 2^s shards, split
;   the 64-bit unsigned key space into N contiguous equal blocks of width
;   W = 2^(64-s). The owning shard is the key's TOP s bits:
;
;       shard(key) = key >> (64 - s)          (== floor(key / W))
;
;   This is MONOTONE NON-DECREASING in the unsigned key: key1 <= key2  =>
;   shard(key1) <= shard(key2). Therefore every key in shard i is strictly
;   less (unsigned) than every key in shard i+1, and the global ascending
;   order is exactly shard 0's keys, then shard 1's, ... then shard N-1's.
;   Ordered global ops just visit shards in ascending index order and each
;   shard's own treemap yields its slice already sorted — no merge/heap needed.
;
;   Implementation detail: the shift amount is materialized as (64 - s) but
;   masked with 63 before use (a shift-by-64 is poison), and the result is
;   AND-ed with (N-1). For s>=1 the mask is a no-op; for s==0 (N==1) the mask
;   forces shard 0, so a single-shard map is handled without a branch.
;
; ============================================================================
; LOCKING / CONCURRENCY
; ----------------------------------------------------------------------------
;   Per-shard test-and-test-and-set spinlock (one i32 word on the shard's own
;   128 B line): acquire = cmpxchg 0->1 ACQUIRE (observes the prior holder's
;   release, so the shard's treemap is fully visible); spin = plain monotonic
;   loads (no cmpxchg storm on the contended line); release = store 0 RELEASE.
;   Critical sections are a treemap op (probe + a few memmoves) — far cheaper
;   than a futex round-trip, so a spinlock wins.
;
;   * POINT ops (put/get/delete/contains) route to ONE shard, lock it, delegate
;     to the shard's treemap, unlock. Threads on disjoint ranges never touch a
;     common lock line.
;   * ORDERED ops:
;       - size(): consistent snapshot — lock EVERY shard in ASCENDING index
;         order (one fixed global order => deadlock-free), sum, release all.
;       - min/max/floor/ceiling/higher/lower/foreach/range: hold AT MOST ONE
;         shard lock at a time (lock, read/emit, unlock; step to the next
;         shard) so they can never participate in a lock cycle. Correct because
;         range-sharding makes shard order == key order: e.g. floor(k) is
;         k's-shard floor, or (if that shard has nothing <= k) the MAX of the
;         nearest lower non-empty shard. Ordered ops are cold vs the point
;         fast path.
;   * foreach CALLBACK CONTRACT (a hard precondition, not a nicety): the user
;     `fn` runs WHILE this thread holds the current shard's TTAS spinlock. It
;     therefore MUST (a) NOT re-enter the map on any key that could route to the
;     SAME shard (put/get/delete on it self-deadlocks — the spinlock is non-
;     reentrant), and (b) be O(1)/short — a slow callback holds the lock across
;     its whole run, busy-spinning every contending thread. Callers needing a
;     heavy or re-entrant body must snapshot the entries out first. (A snapshot-
;     then-callback variant that releases the lock before `fn` is a planned
;     addition; until then the contract above is enforced by documentation.)
;
; Shard (128 B, one line):  lock@0(i32)  treemap@8(ptr)   [pad -> 128]
; Map header (128 B):  nshards@0  shift@8(=64-s)  mask@16(=N-1)  cap@24
;                      shards[] at +128, stride 128
; (posix_memalign 128 => shard 0 is 128-aligned; consecutive shard locks are
;  exactly 128 B apart => never false-share.)
;
; API (error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 5 NOT_FOUND):
;   ptr  universe_ds_treemap_sharded_create(i64 nshards, i64 cap_per_shard)
;   void universe_ds_treemap_sharded_destroy(ptr m)
;   i32  universe_ds_treemap_sharded_put(ptr m, i64 key, i64 val)
;   i32  universe_ds_treemap_sharded_get(ptr m, i64 key, ptr out_v)
;   i32  universe_ds_treemap_sharded_contains(ptr m, i64 key)
;   i32  universe_ds_treemap_sharded_delete(ptr m, i64 key)
;   i64  universe_ds_treemap_sharded_size(ptr m)         ; consistent snapshot
;   i64  universe_ds_treemap_sharded_shards(ptr m)
;   i32  universe_ds_treemap_sharded_min(ptr m, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_sharded_max(ptr m, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_sharded_floor(ptr m, i64 k, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_sharded_ceiling(ptr m, i64 k, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_sharded_higher(ptr m, i64 k, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_sharded_lower(ptr m, i64 k, ptr out_k, ptr out_v)
;   i64  universe_ds_treemap_sharded_range(ptr m, i64 lo, i64 hi, ptr out_k,
;                                          ptr out_v, i64 out_cap)  ; inclusive
;   void universe_ds_treemap_sharded_foreach(ptr m, ptr fn, ptr ctx)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; reused array-backed treemap (linked from src/structures_assoc/treemap.ll)
declare ptr @universe_ds_treemap_create(i64)
declare void @universe_ds_treemap_destroy(ptr)
declare i32 @universe_ds_treemap_put(ptr, i64, i64)
declare i32 @universe_ds_treemap_get(ptr, i64, ptr)
declare i32 @universe_ds_treemap_contains(ptr, i64)
declare i32 @universe_ds_treemap_delete(ptr, i64)
declare i64 @universe_ds_treemap_size(ptr)
declare i32 @universe_ds_treemap_floor(ptr, i64, ptr, ptr)
declare i32 @universe_ds_treemap_ceiling(ptr, i64, ptr, ptr)
declare i32 @universe_ds_treemap_higher(ptr, i64, ptr, ptr)
declare i32 @universe_ds_treemap_lower(ptr, i64, ptr, ptr)
declare i32 @universe_ds_treemap_min(ptr, ptr, ptr)
declare i32 @universe_ds_treemap_max(ptr, ptr, ptr)
declare i64 @universe_ds_treemap_range(ptr, i64, i64, ptr, ptr, i64)
declare void @universe_ds_treemap_foreach(ptr, ptr, ptr)

; ===========================================================================
; internal helpers
; ===========================================================================

; TTAS spinlock acquire (lock word @ shard+0). NOT willreturn (may spin).
define internal void @tms_lock(ptr %sh) #3 {
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

define internal void @tms_unlock(ptr %sh) #3 {
entry:
  store atomic i32 0, ptr %sh release, align 4
  ret void
}

; shard index for a key = (key >> (shift & 63)) & mask
define internal i64 @tms_idx(ptr %m, i64 %key) #4 {
entry:
  %shift.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %shift = load i64, ptr %shift.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %mask = load i64, ptr %mask.p, align 8
  %amt = and i64 %shift, 63
  %top = lshr i64 %key, %amt
  %idx = and i64 %top, %mask
  ret i64 %idx
}

; shard pointer for index i = m + 128 + i*128
define internal ptr @tms_shard(ptr %m, i64 %i) #4 {
entry:
  %off = shl nuw i64 %i, 7
  %off2 = add nuw i64 %off, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %off2
  ret ptr %sh
}

; ascending scan from %start: first non-empty shard's min -> emit, ret 0; else 5.
define internal i32 @tms_up_min(ptr %m, i64 %start, ptr %out_k, ptr %out_v) #1 {
entry:
  %n = load i64, ptr %m, align 8
  %past = icmp uge i64 %start, %n
  br i1 %past, label %miss, label %loop

loop:
  %i = phi i64 [ %start, %entry ], [ %i.n, %cont ]
  %sh = call ptr @tms_shard(ptr %m, i64 %i)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %sz = call i64 @universe_ds_treemap_size(ptr %tm)
  %ne = icmp ne i64 %sz, 0
  br i1 %ne, label %hit, label %cont

hit:
  %r = call i32 @universe_ds_treemap_min(ptr %tm, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  ret i32 0

cont:
  call void @tms_unlock(ptr %sh)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %miss

miss:
  ret i32 5
}

; descending scan from %start (signed; <0 => none): first non-empty max, ret 0.
define internal i32 @tms_down_max(ptr %m, i64 %start, ptr %out_k, ptr %out_v) #1 {
entry:
  %neg = icmp slt i64 %start, 0
  br i1 %neg, label %miss, label %loop

loop:
  %i = phi i64 [ %start, %entry ], [ %i.n, %dec ]
  %sh = call ptr @tms_shard(ptr %m, i64 %i)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %sz = call i64 @universe_ds_treemap_size(ptr %tm)
  %ne = icmp ne i64 %sz, 0
  br i1 %ne, label %hit, label %cont

hit:
  %r = call i32 @universe_ds_treemap_max(ptr %tm, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  ret i32 0

cont:
  call void @tms_unlock(ptr %sh)
  %atzero = icmp eq i64 %i, 0
  br i1 %atzero, label %miss, label %dec

dec:
  %i.n = sub i64 %i, 1
  br label %loop

miss:
  ret i32 5
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_treemap_sharded_create(i64 %nshards, i64 %cap_per_shard) local_unnamed_addr #2 {
entry:
  ; nshards: 0 => 16; clamp [1,65536]; round up to pow2
  %ns.zero = icmp eq i64 %nshards, 0
  %ns.req = select i1 %ns.zero, i64 16, i64 %nshards
  %ns.cl = call i64 @llvm.umin.i64(i64 %ns.req, i64 65536)
  %ns0 = call i64 @llvm.umax.i64(i64 %ns.cl, i64 1)
  %nsm1 = add i64 %ns0, -1
  %nslz = call i64 @llvm.ctlz.i64(i64 %nsm1, i1 false)
  %nssh = sub nuw nsw i64 64, %nslz
  %nsh = shl nuw i64 1, %nssh
  ; s = log2(nsh) = nssh ; shift = 64 - s ; mask = nsh - 1
  %shift = sub nuw nsw i64 64, %nssh
  %mask = add i64 %nsh, -1
  ; cap_per_shard default 8
  %cap0 = call i64 @llvm.umax.i64(i64 %cap_per_shard, i64 8)
  ; block bytes = 128 + nsh*128
  %bb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nsh, i64 128)
  %bb.v = extractvalue { i64, i1 } %bb, 0
  %bb.o = extractvalue { i64, i1 } %bb, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bb.v, i64 128)
  %tt.v = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %ovf = or i1 %bb.o, %tt.o
  br i1 %ovf, label %fail0, label %alloc, !prof !0

alloc:
  %slot.pp = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot.pp, i64 128, i64 %tt.v)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %fail0, label %chk, !prof !0

chk:
  %mem = load ptr, ptr %slot.pp, align 8
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail0, label %init, !prof !0

init:
  call void @llvm.memset.p0.i64(ptr %mem, i8 0, i64 %tt.v, i1 false)
  store i64 %nsh, ptr %mem, align 8
  %shift.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %shift, ptr %shift.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %mask, ptr %mask.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %cap0, ptr %cap.p, align 8
  br label %shloop

shloop:
  %si = phi i64 [ 0, %init ], [ %si.n, %shcont ]
  %sh = call ptr @tms_shard(ptr %mem, i64 %si)
  %tm = call ptr @universe_ds_treemap_create(i64 %cap0)
  %tm.null = icmp eq ptr %tm, null
  br i1 %tm.null, label %cleanup, label %shinit, !prof !0

shinit:
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  store ptr %tm, ptr %tm.p, align 8
  br label %shcont

shcont:
  %si.n = add nuw i64 %si, 1
  %more = icmp ult i64 %si.n, %nsh
  br i1 %more, label %shloop, label %done

done:
  ret ptr %mem

cleanup:
  ; %si shards [0,%si) were created; destroy them, free block, return null
  %has = icmp eq i64 %si, 0
  br i1 %has, label %cl.free, label %cl.loop

cl.loop:
  %ci = phi i64 [ 0, %cleanup ], [ %ci.n, %cl.body ]
  %csh = call ptr @tms_shard(ptr %mem, i64 %ci)
  %ctm.p = getelementptr inbounds nuw i8, ptr %csh, i64 8
  %ctm = load ptr, ptr %ctm.p, align 8
  call void @universe_ds_treemap_destroy(ptr %ctm)
  br label %cl.body

cl.body:
  %ci.n = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, %si
  br i1 %cmore, label %cl.loop, label %cl.free

cl.free:
  call void @free(ptr nonnull %mem)
  br label %fail0

fail0:
  ret ptr null
}

define void @universe_ds_treemap_sharded_destroy(ptr %m) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %setup, !prof !0

setup:
  %n = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %loop ]
  %sh = call ptr @tms_shard(ptr %m, i64 %i)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  call void @universe_ds_treemap_destroy(ptr %tm)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %free.block

free.block:
  call void @free(ptr nonnull %m)
  br label %ret

ret:
  ret void
}

; ===========================================================================
; point ops
; ===========================================================================
define i32 @universe_ds_treemap_sharded_put(ptr %m, i64 %key, i64 %val) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_put(ptr %tm, i64 %key, i64 %val)
  call void @tms_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %key, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  %o.null = icmp eq ptr %out_v, null
  %bad = or i1 %m.null, %o.null
  br i1 %bad, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_get(ptr %tm, i64 %key, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_treemap_sharded_contains(ptr %m, i64 %key) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_contains(ptr %tm, i64 %key)
  call void @tms_unlock(ptr %sh)
  ret i32 %rc
}

define i32 @universe_ds_treemap_sharded_delete(ptr %m, i64 %key) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_delete(ptr %tm, i64 %key)
  call void @tms_unlock(ptr %sh)
  ret i32 %rc
}

define i64 @universe_ds_treemap_sharded_shards(ptr %m) local_unnamed_addr #5 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  %n = load i64, ptr %m, align 8
  ret i64 %n
}

; ===========================================================================
; size — consistent snapshot: lock ALL shards ascending, sum, release all.
; ===========================================================================
define i64 @universe_ds_treemap_sharded_size(ptr %m) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %z, label %setup, !prof !0

z:
  ret i64 0

setup:
  %n = load i64, ptr %m, align 8
  br label %lock.loop

lock.loop:
  %li = phi i64 [ 0, %setup ], [ %li.n, %lock.loop ]
  %lsh = call ptr @tms_shard(ptr %m, i64 %li)
  call void @tms_lock(ptr %lsh)
  %li.n = add nuw i64 %li, 1
  %lmore = icmp ult i64 %li.n, %n
  br i1 %lmore, label %lock.loop, label %sum.loop

sum.loop:
  %si = phi i64 [ 0, %lock.loop ], [ %si.n, %sum.loop ]
  %acc = phi i64 [ 0, %lock.loop ], [ %acc.n, %sum.loop ]
  %ssh = call ptr @tms_shard(ptr %m, i64 %si)
  %stm.p = getelementptr inbounds nuw i8, ptr %ssh, i64 8
  %stm = load ptr, ptr %stm.p, align 8
  %sz = call i64 @universe_ds_treemap_size(ptr %stm)
  %acc.n = add i64 %acc, %sz
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, %n
  br i1 %smore, label %sum.loop, label %unlock.loop

unlock.loop:
  %ui = phi i64 [ 0, %sum.loop ], [ %ui.n, %unlock.loop ]
  %ush = call ptr @tms_shard(ptr %m, i64 %ui)
  call void @tms_unlock(ptr %ush)
  %ui.n = add nuw i64 %ui, 1
  %umore = icmp ult i64 %ui.n, %n
  br i1 %umore, label %unlock.loop, label %done

done:
  ret i64 %acc.n
}

; ===========================================================================
; ordered nearest-key queries (one shard lock at a time)
; ===========================================================================
define i32 @universe_ds_treemap_sharded_min(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %r = call i32 @tms_up_min(ptr %m, i64 0, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_sharded_max(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %n = load i64, ptr %m, align 8
  %start = sub i64 %n, 1
  %r = call i32 @tms_down_max(ptr %m, i64 %start, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_sharded_floor(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_floor(ptr %tm, i64 %key, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %hit, label %lower.shards

hit:
  ret i32 0

lower.shards:
  %start = sub i64 %idx, 1
  %r = call i32 @tms_down_max(ptr %m, i64 %start, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_sharded_lower(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_lower(ptr %tm, i64 %key, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %hit, label %lower.shards

hit:
  ret i32 0

lower.shards:
  %start = sub i64 %idx, 1
  %r = call i32 @tms_down_max(ptr %m, i64 %start, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_sharded_ceiling(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_ceiling(ptr %tm, i64 %key, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %hit, label %upper.shards

hit:
  ret i32 0

upper.shards:
  %start = add nuw i64 %idx, 1
  %r = call i32 @tms_up_min(ptr %m, i64 %start, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_sharded_higher(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %go, !prof !0

err.null:
  ret i32 1

go:
  %idx = call i64 @tms_idx(ptr %m, i64 %key)
  %sh = call ptr @tms_shard(ptr %m, i64 %idx)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  %rc = call i32 @universe_ds_treemap_higher(ptr %tm, i64 %key, ptr %out_k, ptr %out_v)
  call void @tms_unlock(ptr %sh)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %hit, label %upper.shards

hit:
  ret i32 0

upper.shards:
  %start = add nuw i64 %idx, 1
  %r = call i32 @tms_up_min(ptr %m, i64 %start, ptr %out_k, ptr %out_v)
  ret i32 %r
}

; ===========================================================================
; range [lo,hi] inclusive — visit shards route(lo)..route(hi) ascending,
; one lock at a time, concatenating each shard's already-sorted slice.
; ===========================================================================
define i64 @universe_ds_treemap_sharded_range(ptr %m, i64 %lo, i64 %hi, ptr %out_k, ptr %out_v, i64 %out_cap) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %z, label %chkord, !prof !0

z:
  ret i64 0

chkord:
  %bad = icmp ugt i64 %lo, %hi
  br i1 %bad, label %z, label %setup

setup:
  %rlo = call i64 @tms_idx(ptr %m, i64 %lo)
  %rhi = call i64 @tms_idx(ptr %m, i64 %hi)
  %have.out = icmp ne ptr %out_k, null
  br label %loop

loop:
  %i = phi i64 [ %rlo, %setup ], [ %i.n, %cont ]
  %total = phi i64 [ 0, %setup ], [ %total.n, %cont ]
  %written = phi i64 [ 0, %setup ], [ %written.n, %cont ]
  %sh = call ptr @tms_shard(ptr %m, i64 %i)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  ; compute this shard's out pointers/cap (only when caller supplied buffers)
  br i1 %have.out, label %prep.out, label %call.noout

prep.out:
  %rem0 = sub i64 %out_cap, %written
  %fit = icmp ult i64 %written, %out_cap
  %rem = select i1 %fit, i64 %rem0, i64 0
  %ok.i = getelementptr inbounds nuw i64, ptr %out_k, i64 %written
  %v.is.null = icmp eq ptr %out_v, null
  %ov.base = select i1 %v.is.null, ptr null, ptr %out_v
  %ov.i = getelementptr inbounds nuw i64, ptr %ov.base, i64 %written
  %ov.f = select i1 %v.is.null, ptr null, ptr %ov.i
  br label %call.out

call.out:
  %c1 = call i64 @universe_ds_treemap_range(ptr %tm, i64 %lo, i64 %hi, ptr %ok.i, ptr %ov.f, i64 %rem)
  ; wrote min(c1, rem)
  %wr.fit = icmp ult i64 %c1, %rem
  %wrote = select i1 %wr.fit, i64 %c1, i64 %rem
  br label %after

call.noout:
  %c2 = call i64 @universe_ds_treemap_range(ptr %tm, i64 %lo, i64 %hi, ptr null, ptr null, i64 0)
  br label %after

after:
  %c = phi i64 [ %c1, %call.out ], [ %c2, %call.noout ]
  %wr = phi i64 [ %wrote, %call.out ], [ 0, %call.noout ]
  call void @tms_unlock(ptr %sh)
  %total.n = add i64 %total, %c
  %written.n = add i64 %written, %wr
  %done = icmp uge i64 %i, %rhi
  br i1 %done, label %fin, label %cont

cont:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  ret i64 %total.n
}

; ===========================================================================
; foreach — ascending global order; one shard lock at a time.
; CALLBACK CONTRACT: fn runs UNDER the shard spinlock — it must not re-enter
; the map (same-shard re-entry self-deadlocks) and must be O(1). See DESIGN.
; ===========================================================================
define void @universe_ds_treemap_sharded_foreach(ptr %m, ptr %fn, ptr %ctx) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  %fn.null = icmp eq ptr %fn, null
  %bad = or i1 %m.null, %fn.null
  br i1 %bad, label %ret, label %setup, !prof !0

setup:
  %n = load i64, ptr %m, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %loop ]
  %sh = call ptr @tms_shard(ptr %m, i64 %i)
  call void @tms_lock(ptr %sh)
  %tm.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tm = load ptr, ptr %tm.p, align 8
  call void @universe_ds_treemap_foreach(ptr %tm, ptr %fn, ptr %ctx)
  call void @tms_unlock(ptr %sh)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %ret

ret:
  ret void
}

attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind }
attributes #3 = { alwaysinline nounwind norecurse }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #5 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
