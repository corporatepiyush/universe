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

; universe_conc_scounter — STRIPED (sharded) 64-bit counter.
;
; ============================================================================
; DESIGN — per-shard counters, each on its OWN 128 B cache line
; ----------------------------------------------------------------------------
;   A single shared atomic counter scales NEGATIVELY under contention: every
;   `atomicrmw add` bounces the one cache line between cores (coherency
;   ping-pong), so 8 cores are slower than 1. The fix is STATE SHARDING:
;   partition the counter into a POWER-OF-TWO number of independent shards,
;   each on its OWN 128 B cache line. A thread increments the shard chosen by
;   `shard_id & (N-1)` — with distinct per-thread ids the threads hit distinct
;   lines and their atomics never contend. `sum()` merges the shards (a cold,
;   infrequent read). Contention drops ~N×; conservation is exact because each
;   increment is a single `atomicrmw add` (no lost updates) and the merge sees
;   every shard.
;
;   Orderings (all MONOTONIC — a statistics counter carries NO happens-before
;   for other data; we only need atomicity/conservation of the count itself,
;   not ordering against unrelated memory):
;     - inc   : atomicrmw add   monotonic  (single LSE `ldadd` on ARM)
;     - sum   : atomic load     monotonic  (per shard)
;     - reset : atomic store 0  monotonic
;
;   Why per-thread/core sharding (id-based) rather than key-hash sharding:
;   a counter has no key, and per-id sharding is SKEW-PROOF — hot ids still
;   spread across shards. Choose N >= the number of writer threads (2x-4x
;   cores is the sweet spot; past the knee it is memory for nothing).
;
; Layout (posix_memalign 128, so shard 0 is 128-aligned and every shard line
; is a distinct 128 B cache line — locking/incrementing one never touches a
; neighbour's line):
;   +0    nshards (i64, power of two)
;   +8    mask (= nshards-1)
;   +128  shard[0] counter (atomic i64)          ; own 128 B line
;   +256  shard[1] counter (atomic i64)          ; own 128 B line
;   ...   shard[i] at 128 + i*128
;
; API:
;   ptr  universe_conc_scounter_create(i64 nshards)  ; nshards rounded up to
;                                                      ; pow2, clamped [1,1<<20];
;                                                      ; 0 => default 64
;   void universe_conc_scounter_inc(ptr c, i64 shard_id, i64 delta)
;   i64  universe_conc_scounter_sum(ptr c)           ; merge all shards
;   void universe_conc_scounter_reset(ptr c)
;   i64  universe_conc_scounter_shards(ptr c)
;   void universe_conc_scounter_destroy(ptr c)

declare i32  @posix_memalign(ptr, i64, i64)
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; create — nshards rounded up to a power of two, clamped to [1, 2^20].
; ===========================================================================
define noalias ptr @universe_conc_scounter_create(i64 %nshards) local_unnamed_addr #1 {
entry:
  ; 0 => default 64
  %is.zero = icmp eq i64 %nshards, 0
  %req = select i1 %is.zero, i64 64, i64 %nshards
  %clamped = call i64 @llvm.umin.i64(i64 %req, i64 1048576)
  %n0 = call i64 @llvm.umax.i64(i64 %clamped, i64 1)
  ; round up to power of two: 1 << (64 - ctlz(n0-1))
  %nm1 = add i64 %n0, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %nm1, i1 false)
  %shift = sub nuw nsw i64 64, %lz
  %nsh = shl nuw i64 1, %shift
  ; total = 128 + nsh*128  (nsh <= 2^20 so no realistic overflow, but check)
  %b = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nsh, i64 128)
  %b.v = extractvalue { i64, i1 } %b, 0
  %b.o = extractvalue { i64, i1 } %b, 1
  %t = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %b.v, i64 128)
  %t.v = extractvalue { i64, i1 } %t, 0
  %t.o = extractvalue { i64, i1 } %t, 1
  %ovf = or i1 %b.o, %t.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %slot.pp = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot.pp, i64 128, i64 %t.v)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %fail, label %chk, !prof !0

chk:
  %mem = load ptr, ptr %slot.pp, align 8
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  ; zero the whole block (metadata + all shard lines)
  call void @llvm.memset.p0.i64(ptr %mem, i8 0, i64 %t.v, i1 false)
  store i64 %nsh, ptr %mem, align 8
  %mask = add i64 %nsh, -1
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %mask, ptr %mask.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

; ===========================================================================
; inc — add delta to shard (shard_id & mask). Single atomicrmw, monotonic.
; ===========================================================================
define void @universe_conc_scounter_inc(ptr %c, i64 %shard_id, i64 %delta) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %ret, label %go, !prof !0

go:
  %mask.p = getelementptr inbounds nuw i8, ptr %c, i64 8
  %mask = load i64, ptr %mask.p, align 8
  %idx = and i64 %shard_id, %mask
  ; byte offset = 128 + idx*128 = (idx+1) << 7
  %idx1 = add nuw i64 %idx, 1
  %off = shl nuw i64 %idx1, 7
  %cnt.p = getelementptr inbounds nuw i8, ptr %c, i64 %off
  %old = atomicrmw add ptr %cnt.p, i64 %delta monotonic, align 8
  br label %ret

ret:
  ret void
}

; ===========================================================================
; sum — merge all shards (cold, infrequent). Monotonic loads.
; ===========================================================================
define i64 @universe_conc_scounter_sum(ptr %c) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %null, label %setup, !prof !0

null:
  ret i64 0

setup:
  %nsh = load i64, ptr %c, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %setup ], [ %acc.n, %loop ]
  %i1 = add nuw i64 %i, 1
  %off = shl nuw i64 %i1, 7
  %cnt.p = getelementptr inbounds nuw i8, ptr %c, i64 %off
  %v = load atomic i64, ptr %cnt.p monotonic, align 8
  %acc.n = add i64 %acc, %v
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nsh
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n
}

; ===========================================================================
; reset — zero every shard. Monotonic stores.
; ===========================================================================
define void @universe_conc_scounter_reset(ptr %c) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %ret, label %setup, !prof !0

setup:
  %nsh = load i64, ptr %c, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %loop ]
  %i1 = add nuw i64 %i, 1
  %off = shl nuw i64 %i1, 7
  %cnt.p = getelementptr inbounds nuw i8, ptr %c, i64 %off
  store atomic i64 0, ptr %cnt.p monotonic, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nsh
  br i1 %more, label %loop, label %ret

ret:
  ret void
}

define i64 @universe_conc_scounter_shards(ptr %c) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %null, label %go, !prof !0

null:
  ret i64 0

go:
  %nsh = load i64, ptr %c, align 8
  ret i64 %nsh
}

define void @universe_conc_scounter_destroy(ptr %c) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %ret, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %c)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
