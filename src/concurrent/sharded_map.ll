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

; universe_conc_shardmap — SHARDED (striped-lock) concurrent hash map.
; Byte-string keys (ptr + len) -> i64 values.
;
; ============================================================================
; DESIGN — lock striping: N independent shards, each its own lock + table
; ----------------------------------------------------------------------------
;   A single lock over one shared map SERIALIZES every writer and scales
;   NEGATIVELY with cores (the one lock line ping-pongs). Lock STRIPING fixes
;   this: partition the key space into a POWER-OF-TWO number of shards, each a
;   self-contained open-addressing map guarded by its OWN lock on its OWN
;   128 B cache line. A key routes to `shard = hash(key) & (N-1)` (mask, never
;   modulo); operations on keys in different shards proceed in parallel and
;   never touch each other's lock line, so contention drops ~N×. Threads
;   touching disjoint keys almost never collide on a shard; only genuine
;   same-key / same-shard contention serializes, and only within that shard.
;
;   Two decorrelated hashes from ONE FNV-1a pass:
;     * primary   h  = fnv1a(key)          ; shard = h & (nshards-1)
;     * in-shard  ih = splitmix64(h)       ; probe start = ih & (cap-1)
;   Using the SAME h for both would correlate shard bits with probe bits and
;   cluster a shard's keys; mixing h once decorrelates them for free.
;
;   Per-shard map: linear-probe open addressing, 40 B slots (AoS), load factor
;   bound 7/8. On overflow: rehash into 2× (grow) or same-size (reclaim
;   tombstones when live fits in half) — identical policy to the single-thread
;   swiss map, done under the shard lock so it is invisible to other shards.
;   Keys are copied into a per-entry malloc so callers may free their buffers;
;   delete frees the copy and tombstones the slot. (A per-shard arena would
;   avoid per-entry malloc when delete is rare — a later refinement.)
;
;   Locking: a per-shard test-and-test-and-set SPINLOCK (one i32 word). Chosen
;   over a pthread_mutex because critical sections are tiny (a probe + a few
;   stores) and non-blocking; a futex/mutex's syscall cost would dwarf the work.
;     * acquire : cmpxchg 0->1 ACQUIRE  (publishes nothing, observes the prior
;                 holder's release => the shard's table/count are visible)
;     * spin    : plain monotonic loads until the word reads 0 (test-and-test-
;                 and-set: no cmpxchg storm on the contended line)
;     * release : store 0 RELEASE (make our table mutations visible to the next
;                 holder). All shard fields (table/count/cap/mask/tomb) are
;                 plain memory ordered solely by this lock — no other atomics.
;
;   Global ops: len() takes EVERY shard lock in ASCENDING index order (one
;   fixed global order => deadlock-free) for a consistent snapshot, sums the
;   per-shard counts, then releases. Rare and cold; a striped map hides its
;   cost behind the per-shard fast path.
;
; Slot (40 B AoS):  state@0(i64: 0 EMPTY,1 FULL,2 TOMB) hash@8(ih) keyptr@16
;                   klen@24  val@32
; Shard header (128 B, one line): lock@0(i32) table@8 count@16 capacity@24
;                                 mask@32(=cap-1) tombstones@40  [pad -> 128]
; Map header: nshards@0 shard_mask@8 init_cap@16 ; shards[] at +128, stride 128
; (posix_memalign 128 => shard 0 is 128-aligned; consecutive shard locks are
;  exactly 128 B apart => never false-share.)
;
; API (error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 5 NOT_FOUND):
;   ptr  universe_conc_shardmap_create(i64 nshards, i64 cap_per_shard)
;   i32  universe_conc_shardmap_put(ptr m, ptr key, i64 klen, i64 val)
;   i32  universe_conc_shardmap_get(ptr m, ptr key, i64 klen, ptr out)
;   i32  universe_conc_shardmap_delete(ptr m, ptr key, i64 klen)
;   i64  universe_conc_shardmap_len(ptr m)         ; consistent snapshot
;   i64  universe_conc_shardmap_shards(ptr m)
;   void universe_conc_shardmap_destroy(ptr m)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; internal helpers (all alwaysinline => zero cost in the exported ops)
; ===========================================================================

; FNV-1a over key bytes. offset basis 0xcbf29ce484222325, prime 0x100000001b3.
define internal i64 @sm_fnv(ptr %key, i64 %klen) #4 {
entry:
  %z = icmp eq i64 %klen, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %h = phi i64 [ -3750763034362895579, %entry ], [ %h.n, %loop ]
  %bp = getelementptr inbounds nuw i8, ptr %key, i64 %i
  %b = load i8, ptr %bp, align 1
  %b64 = zext i8 %b to i64
  %x = xor i64 %h, %b64
  %h.n = mul i64 %x, 1099511628211
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %klen
  br i1 %more, label %loop, label %done

done:
  %hf = phi i64 [ -3750763034362895579, %entry ], [ %h.n, %loop ]
  ret i64 %hf
}

; splitmix64 finalizer — decorrelate the in-shard hash from the shard index.
define internal i64 @sm_mix(i64 %x) #5 {
entry:
  %a1 = lshr i64 %x, 30
  %a2 = xor i64 %a1, %x
  %a3 = mul i64 %a2, -4658895280553007687
  %a4 = lshr i64 %a3, 27
  %a5 = xor i64 %a4, %a3
  %a6 = mul i64 %a5, -7723592293110705685
  %a7 = lshr i64 %a6, 31
  %a8 = xor i64 %a7, %a6
  ret i64 %a8
}

; byte-wise key equality (len bytes). Only called on hash+len match => cold.
define internal i1 @sm_keq(ptr %a, ptr %b, i64 %len) #4 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %eq, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %av = load i8, ptr %ap, align 1
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %neq, label %cont

cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %len
  br i1 %more, label %loop, label %eq

eq:
  ret i1 true

neq:
  ret i1 false
}

; test-and-test-and-set spinlock acquire (lock word @ shard+0).
define internal void @sm_lock(ptr %sh) #3 {
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

define internal void @sm_unlock(ptr %sh) #3 {
entry:
  store atomic i32 0, ptr %sh release, align 4
  ret void
}

; Find slot holding key (state FULL, hash==ih, klen match, bytes match), or -1.
; A table always has >=1 EMPTY (load factor < 1), so the probe terminates.
define internal i64 @sm_find(ptr %table, i64 %mask, i64 %ih, ptr %key, i64 %klen) #6 {
entry:
  %start = and i64 %ih, %mask
  br label %probe

probe:
  %idx = phi i64 [ %start, %entry ], [ %idx.n, %next ]
  %off = mul nuw i64 %idx, 40
  %slot = getelementptr inbounds nuw i8, ptr %table, i64 %off
  %state = load i64, ptr %slot, align 8
  %empty = icmp eq i64 %state, 0
  br i1 %empty, label %notfound, label %chkfull

chkfull:
  %isfull = icmp eq i64 %state, 1
  br i1 %isfull, label %maybe, label %next

maybe:
  %hp = getelementptr inbounds nuw i8, ptr %slot, i64 8
  %sh2 = load i64, ptr %hp, align 8
  %heq = icmp eq i64 %sh2, %ih
  br i1 %heq, label %chklen, label %next

chklen:
  %klp = getelementptr inbounds nuw i8, ptr %slot, i64 24
  %sklen = load i64, ptr %klp, align 8
  %leq = icmp eq i64 %sklen, %klen
  br i1 %leq, label %chkbytes, label %next

chkbytes:
  %kpp = getelementptr inbounds nuw i8, ptr %slot, i64 16
  %skp = load ptr, ptr %kpp, align 8
  %same = call i1 @sm_keq(ptr %skp, ptr %key, i64 %klen)
  br i1 %same, label %found, label %next

next:
  %idx.n0 = add nuw i64 %idx, 1
  %idx.n = and i64 %idx.n0, %mask
  br label %probe

found:
  ret i64 %idx

notfound:
  ret i64 -1
}

; First EMPTY-or-TOMBSTONE slot in probe order (guaranteed to exist).
define internal i64 @sm_islot(ptr %table, i64 %mask, i64 %ih) #6 {
entry:
  %start = and i64 %ih, %mask
  br label %probe

probe:
  %idx = phi i64 [ %start, %entry ], [ %idx.n, %next ]
  %off = mul nuw i64 %idx, 40
  %slot = getelementptr inbounds nuw i8, ptr %table, i64 %off
  %state = load i64, ptr %slot, align 8
  %usable = icmp ne i64 %state, 1
  br i1 %usable, label %take, label %next

take:
  ret i64 %idx

next:
  %idx.n0 = add nuw i64 %idx, 1
  %idx.n = and i64 %idx.n0, %mask
  br label %probe
}

; Grow (2x) or reclaim (same size, drop tombstones) a shard's table in place.
; Caller holds the shard lock. Returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW.
define internal i32 @sm_resize(ptr %sh) #1 {
entry:
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %oldtable = load ptr, ptr %tbl.p, align 8
  %cnt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  %count = load i64, ptr %cnt.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %oldcap = load i64, ptr %cap.p, align 8
  %half = lshr i64 %oldcap, 1
  %cnt1 = add nuw i64 %count, 1
  %reclaim = icmp ule i64 %cnt1, %half
  %grown = shl nuw i64 %oldcap, 1
  %newcap = select i1 %reclaim, i64 %oldcap, i64 %grown
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %newcap, i64 40)
  %total = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  br i1 %tb.o, label %err.ovf, label %alloc, !prof !0

alloc:
  %newtable = call ptr @malloc(i64 %total)
  %nt.null = icmp eq ptr %newtable, null
  br i1 %nt.null, label %err.oom, label %setup, !prof !0

setup:
  call void @llvm.memset.p0.i64(ptr %newtable, i8 0, i64 %total, i1 false)
  %newmask = add i64 %newcap, -1
  br label %scan

scan:
  %i = phi i64 [ 0, %setup ], [ %i.n, %scan.cont ]
  %done = icmp eq i64 %i, %oldcap
  br i1 %done, label %finish, label %scan.body

scan.body:
  %ooff = mul nuw i64 %i, 40
  %oslot = getelementptr inbounds nuw i8, ptr %oldtable, i64 %ooff
  %ostate = load i64, ptr %oslot, align 8
  %full = icmp eq i64 %ostate, 1
  br i1 %full, label %move, label %scan.cont

move:
  %ohp = getelementptr inbounds nuw i8, ptr %oslot, i64 8
  %oh = load i64, ptr %ohp, align 8
  %ns = call i64 @sm_islot(ptr %newtable, i64 %newmask, i64 %oh)
  %nsoff = mul nuw i64 %ns, 40
  %nslot = getelementptr inbounds nuw i8, ptr %newtable, i64 %nsoff
  call void @llvm.memcpy.p0.p0.i64(ptr %nslot, ptr %oslot, i64 40, i1 false)
  br label %scan.cont

scan.cont:
  %i.n = add nuw i64 %i, 1
  br label %scan

finish:
  call void @free(ptr %oldtable)
  store ptr %newtable, ptr %tbl.p, align 8
  store i64 %newcap, ptr %cap.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  store i64 %newmask, ptr %mask.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %sh, i64 40
  store i64 0, ptr %tomb.p, align 8
  ret i32 0

err.oom:
  ret i32 2

err.ovf:
  ret i32 3
}

; ===========================================================================
; create
; ===========================================================================
define noalias ptr @universe_conc_shardmap_create(i64 %nshards, i64 %cap_per_shard) local_unnamed_addr #1 {
entry:
  ; nshards: 0 => default 64; round up to pow2; clamp [1, 65536]
  %ns.zero = icmp eq i64 %nshards, 0
  %ns.req = select i1 %ns.zero, i64 64, i64 %nshards
  %ns.cl = call i64 @llvm.umin.i64(i64 %ns.req, i64 65536)
  %ns0 = call i64 @llvm.umax.i64(i64 %ns.cl, i64 1)
  %nsm1 = add i64 %ns0, -1
  %nslz = call i64 @llvm.ctlz.i64(i64 %nsm1, i1 false)
  %nssh = sub nuw nsw i64 64, %nslz
  %nsh = shl nuw i64 1, %nssh
  ; cap_per_shard: 0 => default 16; round up to pow2; min 16
  %cp.zero = icmp eq i64 %cap_per_shard, 0
  %cp.req = select i1 %cp.zero, i64 16, i64 %cap_per_shard
  %cp0 = call i64 @llvm.umax.i64(i64 %cp.req, i64 16)
  %cpm1 = add i64 %cp0, -1
  %cplz = call i64 @llvm.ctlz.i64(i64 %cpm1, i1 false)
  %cpsh = sub nuw nsw i64 64, %cplz
  %cap = shl nuw i64 1, %cpsh
  ; per-shard table bytes = cap * 40
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 40)
  %tbytes = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  ; top block bytes = 128 + nsh*128
  %bb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nsh, i64 128)
  %bb.v = extractvalue { i64, i1 } %bb, 0
  %bb.o = extractvalue { i64, i1 } %bb, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bb.v, i64 128)
  %tt.v = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %o0 = or i1 %tb.o, %bb.o
  %ovf = or i1 %o0, %tt.o
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
  %smask = add i64 %nsh, -1
  %smask.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %smask, ptr %smask.p, align 8
  %icap.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %cap, ptr %icap.p, align 8
  %cmask = add i64 %cap, -1
  br label %shloop

shloop:
  %si = phi i64 [ 0, %init ], [ %si.n, %shcont ]
  %shoff = shl nuw i64 %si, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %mem, i64 %shoff2
  %tbl = call ptr @malloc(i64 %tbytes)
  %tbl.null = icmp eq ptr %tbl, null
  br i1 %tbl.null, label %cleanup, label %shinit, !prof !0

shinit:
  call void @llvm.memset.p0.i64(ptr %tbl, i8 0, i64 %tbytes, i1 false)
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  store ptr %tbl, ptr %tbl.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  store i64 %cap, ptr %cap.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  store i64 %cmask, ptr %mask.p, align 8
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
  %ctbl.p = getelementptr inbounds nuw i8, ptr %csh, i64 8
  %ctbl = load ptr, ptr %ctbl.p, align 8
  call void @free(ptr %ctbl)
  %ci.n = add nuw i64 %ci, 1
  br label %cl.loop

cl.free:
  call void @free(ptr nonnull %mem)
  br label %fail0

fail0:
  ret ptr null
}

; ===========================================================================
; put — insert or overwrite
; ===========================================================================
define i32 @universe_conc_shardmap_put(ptr %m, ptr %key, i64 %klen, i64 %val) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  %k.null = icmp eq ptr %key, null
  %bad = or i1 %m.null, %k.null
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %h = call i64 @sm_fnv(ptr %key, i64 %klen)
  %smask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %smask = load i64, ptr %smask.p, align 8
  %shidx = and i64 %h, %smask
  %shoff = shl nuw i64 %shidx, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  %ih = call i64 @sm_mix(i64 %h)
  call void @sm_lock(ptr %sh)
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tbl = load ptr, ptr %tbl.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %mask = load i64, ptr %mask.p, align 8
  %slotidx = call i64 @sm_find(ptr %tbl, i64 %mask, i64 %ih, ptr %key, i64 %klen)
  %found = icmp sge i64 %slotidx, 0
  br i1 %found, label %overwrite, label %absent

overwrite:
  %ooff = mul nuw i64 %slotidx, 40
  %oslot = getelementptr inbounds nuw i8, ptr %tbl, i64 %ooff
  %ovp = getelementptr inbounds nuw i8, ptr %oslot, i64 32
  store i64 %val, ptr %ovp, align 8
  call void @sm_unlock(ptr %sh)
  ret i32 0

absent:
  %cnt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  %count = load i64, ptr %cnt.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %sh, i64 40
  %tomb = load i64, ptr %tomb.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %cap8 = lshr i64 %cap, 3
  %thresh = sub i64 %cap, %cap8
  %used = add nuw i64 %count, %tomb
  %used1 = add nuw i64 %used, 1
  %needgrow = icmp ugt i64 %used1, %thresh
  br i1 %needgrow, label %grow, label %alloc.key, !prof !0

grow:
  %rc.rz = call i32 @sm_resize(ptr %sh)
  br label %reload

reload:
  %tbl2 = load ptr, ptr %tbl.p, align 8
  %mask2 = load i64, ptr %mask.p, align 8
  br label %alloc.key

alloc.key:
  %tbl.f = phi ptr [ %tbl, %absent ], [ %tbl2, %reload ]
  %mask.f = phi i64 [ %mask, %absent ], [ %mask2, %reload ]
  %kalloc = call i64 @llvm.umax.i64(i64 %klen, i64 1)
  %kcopy = call ptr @malloc(i64 %kalloc)
  %kc.null = icmp eq ptr %kcopy, null
  br i1 %kc.null, label %err.oom, label %place, !prof !0

err.oom:
  call void @sm_unlock(ptr %sh)
  ret i32 2

place:
  call void @llvm.memcpy.p0.p0.i64(ptr %kcopy, ptr %key, i64 %klen, i1 false)
  %islot = call i64 @sm_islot(ptr %tbl.f, i64 %mask.f, i64 %ih)
  %isoff = mul nuw i64 %islot, 40
  %islotp = getelementptr inbounds nuw i8, ptr %tbl.f, i64 %isoff
  %pstate = load i64, ptr %islotp, align 8
  %was.tomb = icmp eq i64 %pstate, 2
  store i64 1, ptr %islotp, align 8
  %ph = getelementptr inbounds nuw i8, ptr %islotp, i64 8
  store i64 %ih, ptr %ph, align 8
  %pk = getelementptr inbounds nuw i8, ptr %islotp, i64 16
  store ptr %kcopy, ptr %pk, align 8
  %pkl = getelementptr inbounds nuw i8, ptr %islotp, i64 24
  store i64 %klen, ptr %pkl, align 8
  %pv = getelementptr inbounds nuw i8, ptr %islotp, i64 32
  store i64 %val, ptr %pv, align 8
  %cnt.now = load i64, ptr %cnt.p, align 8
  %cnt.new = add nuw i64 %cnt.now, 1
  store i64 %cnt.new, ptr %cnt.p, align 8
  br i1 %was.tomb, label %dec.tomb, label %fin

dec.tomb:
  %tomb.now = load i64, ptr %tomb.p, align 8
  %tomb.new = sub i64 %tomb.now, 1
  store i64 %tomb.new, ptr %tomb.p, align 8
  br label %fin

fin:
  call void @sm_unlock(ptr %sh)
  ret i32 0
}

; ===========================================================================
; get
; ===========================================================================
define i32 @universe_conc_shardmap_get(ptr %m, ptr %key, i64 %klen, ptr %out) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  %k.null = icmp eq ptr %key, null
  %o.null = icmp eq ptr %out, null
  %b0 = or i1 %m.null, %k.null
  %bad = or i1 %b0, %o.null
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %h = call i64 @sm_fnv(ptr %key, i64 %klen)
  %smask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %smask = load i64, ptr %smask.p, align 8
  %shidx = and i64 %h, %smask
  %shoff = shl nuw i64 %shidx, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  %ih = call i64 @sm_mix(i64 %h)
  call void @sm_lock(ptr %sh)
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tbl = load ptr, ptr %tbl.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %mask = load i64, ptr %mask.p, align 8
  %slotidx = call i64 @sm_find(ptr %tbl, i64 %mask, i64 %ih, ptr %key, i64 %klen)
  %found = icmp sge i64 %slotidx, 0
  br i1 %found, label %hit, label %miss

hit:
  %ooff = mul nuw i64 %slotidx, 40
  %oslot = getelementptr inbounds nuw i8, ptr %tbl, i64 %ooff
  %vp = getelementptr inbounds nuw i8, ptr %oslot, i64 32
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %out, align 8
  call void @sm_unlock(ptr %sh)
  ret i32 0

miss:
  call void @sm_unlock(ptr %sh)
  ret i32 5
}

; ===========================================================================
; delete
; ===========================================================================
define i32 @universe_conc_shardmap_delete(ptr %m, ptr %key, i64 %klen) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  %k.null = icmp eq ptr %key, null
  %bad = or i1 %m.null, %k.null
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %h = call i64 @sm_fnv(ptr %key, i64 %klen)
  %smask.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %smask = load i64, ptr %smask.p, align 8
  %shidx = and i64 %h, %smask
  %shoff = shl nuw i64 %shidx, 7
  %shoff2 = add nuw i64 %shoff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %shoff2
  %ih = call i64 @sm_mix(i64 %h)
  call void @sm_lock(ptr %sh)
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tbl = load ptr, ptr %tbl.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %sh, i64 32
  %mask = load i64, ptr %mask.p, align 8
  %slotidx = call i64 @sm_find(ptr %tbl, i64 %mask, i64 %ih, ptr %key, i64 %klen)
  %found = icmp sge i64 %slotidx, 0
  br i1 %found, label %erase, label %miss

erase:
  %ooff = mul nuw i64 %slotidx, 40
  %oslot = getelementptr inbounds nuw i8, ptr %tbl, i64 %ooff
  %kpp = getelementptr inbounds nuw i8, ptr %oslot, i64 16
  %kp = load ptr, ptr %kpp, align 8
  call void @free(ptr %kp)
  store ptr null, ptr %kpp, align 8
  store i64 2, ptr %oslot, align 8
  %cnt.p = getelementptr inbounds nuw i8, ptr %sh, i64 16
  %count = load i64, ptr %cnt.p, align 8
  %count.n = sub i64 %count, 1
  store i64 %count.n, ptr %cnt.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %sh, i64 40
  %tomb = load i64, ptr %tomb.p, align 8
  %tomb.n = add nuw i64 %tomb, 1
  store i64 %tomb.n, ptr %tomb.p, align 8
  call void @sm_unlock(ptr %sh)
  ret i32 0

miss:
  call void @sm_unlock(ptr %sh)
  ret i32 5
}

; ===========================================================================
; len — consistent snapshot: lock ALL shards in ascending order, sum, release.
; ===========================================================================
define i64 @universe_conc_shardmap_len(ptr %m) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %null, label %setup, !prof !0

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
  call void @sm_lock(ptr %lsh)
  %li.n = add nuw i64 %li, 1
  %lmore = icmp ult i64 %li.n, %nsh
  br i1 %lmore, label %lock.loop, label %sum.loop

sum.loop:
  %si = phi i64 [ 0, %lock.loop ], [ %si.n, %sum.loop ]
  %acc = phi i64 [ 0, %lock.loop ], [ %acc.n, %sum.loop ]
  %soff = shl nuw i64 %si, 7
  %soff2 = add nuw i64 %soff, 128
  %ssh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %scnt.p = getelementptr inbounds nuw i8, ptr %ssh, i64 16
  %scnt = load i64, ptr %scnt.p, align 8
  %acc.n = add i64 %acc, %scnt
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, %nsh
  br i1 %smore, label %sum.loop, label %unlock.loop

unlock.loop:
  %ui = phi i64 [ 0, %sum.loop ], [ %ui.n, %unlock.loop ]
  %uoff = shl nuw i64 %ui, 7
  %uoff2 = add nuw i64 %uoff, 128
  %ush = getelementptr inbounds nuw i8, ptr %m, i64 %uoff2
  call void @sm_unlock(ptr %ush)
  %ui.n = add nuw i64 %ui, 1
  %umore = icmp ult i64 %ui.n, %nsh
  br i1 %umore, label %unlock.loop, label %done

done:
  ret i64 %acc.n
}

define i64 @universe_conc_shardmap_shards(ptr %m) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %null, label %go, !prof !0

null:
  ret i64 0

go:
  %nsh = load i64, ptr %m, align 8
  ret i64 %nsh
}

; ===========================================================================
; destroy — free every key copy, each shard table, then the block.
; ===========================================================================
define void @universe_conc_shardmap_destroy(ptr %m) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %setup, !prof !0

setup:
  %nsh = load i64, ptr %m, align 8
  br label %sh.loop

sh.loop:
  %si = phi i64 [ 0, %setup ], [ %si.n, %sh.next ]
  %soff = shl nuw i64 %si, 7
  %soff2 = add nuw i64 %soff, 128
  %sh = getelementptr inbounds nuw i8, ptr %m, i64 %soff2
  %tbl.p = getelementptr inbounds nuw i8, ptr %sh, i64 8
  %tbl = load ptr, ptr %tbl.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %sh, i64 24
  %cap = load i64, ptr %cap.p, align 8
  br label %slot.loop

slot.loop:
  %ki = phi i64 [ 0, %sh.loop ], [ %ki.n, %slot.cont ]
  %koff = mul nuw i64 %ki, 40
  %kslot = getelementptr inbounds nuw i8, ptr %tbl, i64 %koff
  %kstate = load i64, ptr %kslot, align 8
  %kfull = icmp eq i64 %kstate, 1
  br i1 %kfull, label %free.key, label %slot.cont

free.key:
  %kpp = getelementptr inbounds nuw i8, ptr %kslot, i64 16
  %kp = load ptr, ptr %kpp, align 8
  call void @free(ptr %kp)
  br label %slot.cont

slot.cont:
  %ki.n = add nuw i64 %ki, 1
  %kmore = icmp ult i64 %ki.n, %cap
  br i1 %kmore, label %slot.loop, label %free.tbl

free.tbl:
  call void @free(ptr %tbl)
  br label %sh.next

sh.next:
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, %nsh
  br i1 %smore, label %sh.loop, label %free.block

free.block:
  call void @free(ptr nonnull %m)
  br label %ret

ret:
  ret void
}

attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse memory(argmem: read) }
attributes #3 = { alwaysinline nounwind norecurse }
attributes #4 = { alwaysinline nounwind willreturn norecurse memory(argmem: read) }
attributes #5 = { alwaysinline nounwind willreturn norecurse memory(none) }
attributes #6 = { alwaysinline nounwind willreturn norecurse }

!0 = !{!"branch_weights", i32 1, i32 2000}
