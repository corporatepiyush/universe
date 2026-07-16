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

; MPSC (multi-producer / single-consumer) lock-free queues.
; Two variants:
;   * ring  — bounded, per-slot turn/sequence, XADD claim.
;   * seg   — unbounded, paged chunks addressed by a directory, XADD claim.
;
; DESIGN — why XADD and not a CAS retry loop:
;   The multi-producer contention point is the producer position counter. A
;   CAS(pos, pos+1) loop makes every producer re-read and re-try under
;   contention (O(producers) failed CASes). Instead each producer claims its
;   ticket with ONE `atomicrmw add pos, 1` (lowers to ARM `ldadd`/LSE or x86
;   `lock xadd`) — wait-free on the claim, no retry. Correctness of the DATA
;   is carried by a per-slot release/acquire handshake, NOT by the counter:
;   a monotonic counter load can never gate reading non-atomic slot bytes
;   (that would be a data race); the per-slot ready/sequence flag does.
;
; ---- ring (bounded) ---------------------------------------------------------
;   Slot i initialised with sequence = i.  Producer:
;     1. pre-check (pos - head >= cap) -> FULL, WITHOUT claiming (lossless:
;        a returned FULL never consumes a ticket, so producers may retry).
;     2. myPos = XADD(enqueue_pos, 1).                 ; the wait-free claim
;     3. wait (bounded) until seq[myPos&mask] == myPos ; slot free this lap
;        (only reachable when producers race past the pre-check by <=N-1;
;        the single consumer always drains, so the wait is short and cannot
;        deadlock — the lowest outstanding ticket always has a free slot).
;     4. memcpy the element, then store-release seq = myPos+1 (publish).
;   Consumer at pos: acquire seq; ready iff seq == pos+1; copy out; then
;   store-release seq = pos+cap (frees the slot for the next lap); advance.
;   Orderings: enqueue_pos XADD monotonic (data ordered by seq, not counter);
;   seq load acquire / store release (the real publish/consume gate);
;   dequeue_pos monotonic (a full-detection HINT only; not a data gate).
;
; ---- seg (unbounded) --------------------------------------------------------
;   Global positions are dense and used exactly once, so no lap reuse: a
;   per-slot `ready` flag (0/1) suffices.  A directory (flat array of chunk
;   pointers, DIR_SIZE entries) maps chunkIdx = pos>>LOG_CHUNK -> chunk in
;   O(1), so producers NEVER traverse a linked list (traversal + front
;   reclamation is the classic use-after-free; a directory sidesteps it).
;   Producer: pos = XADD; c = dir[chunkIdx] (acquire); if null install a fresh
;   chunk with ONE cmpxchg on that directory slot (cold, 1 per CHUNK items);
;   write payload; store-release ready=1.  Consumer drains a chunk in order,
;   and when it finishes a chunk's last slot recycles it to a freelist.
;   Reclamation is race-free because {chunks a producer may touch} =
;   [cons_chunk .. tail] and {chunks the consumer recycles} = strictly-older
;   fully-drained chunks; the two sets are disjoint (a chunk is fully drained
;   only after every producer that targeted it has published).  chunkIdx is
;   monotone and never recurs, so a recycled chunk's stale directory slot is
;   never read again.  Freelist is guarded by a tiny spinlock — this is the
;   COLD path (1 per CHUNK items); the hot per-item claim stays lock-free.
;   Capacity: DIR_SIZE*CHUNK positions before the directory is exhausted
;   (returns FULL); with DIR_SIZE=65536, CHUNK=1024 that is 64M live-lifetime
;   items — effectively unbounded for real workloads.

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i32 @sched_yield()

; ============================================================================
;  ring — bounded MPSC
;    layout (single allocation):
;      @0    enqueue_pos : atomic i64     (producers XADD)   [own 128B line]
;      @128  dequeue_pos : atomic i64     (consumer own; producers read hint)
;      @256  mask : i64 (=cap-1)
;      @264  cap  : i64
;      @272  esz  : i64
;      @280  payoff : i64 (=320 + cap*8)
;      @320  seq[cap] : i64
;      @payoff payload[cap] : esz
; ============================================================================

define noalias ptr @universe_conc_mpsc_ring_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
entry:
  %cap.bad = icmp eq i64 %capacity, 0
  %elem.bad = icmp eq i64 %elem_size, 0
  %too.big = icmp ugt i64 %capacity, 4611686018427387904
  %bad0 = or i1 %cap.bad, %elem.bad
  %bad = or i1 %bad0, %too.big
  br i1 %bad, label %fail, label %shape, !prof !0

shape:
  %c.min = call i64 @llvm.umax.i64(i64 %capacity, i64 8)
  %cm1 = add i64 %c.min, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %mask = add i64 %cap, -1
  %sb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 8)
  %sb.v = extractvalue { i64, i1 } %sb, 0
  %sb.o = extractvalue { i64, i1 } %sb, 1
  %pb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 %elem_size)
  %pb.v = extractvalue { i64, i1 } %pb, 0
  %pb.o = extractvalue { i64, i1 } %pb, 1
  %t1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %sb.v, i64 %pb.v)
  %t1.v = extractvalue { i64, i1 } %t1, 0
  %t1.o = extractvalue { i64, i1 } %t1, 1
  %t2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %t1.v, i64 320)
  %total = extractvalue { i64, i1 } %t2, 0
  %t2.o = extractvalue { i64, i1 } %t2, 1
  %o0 = or i1 %sb.o, %pb.o
  %o1 = or i1 %o0, %t1.o
  %ovf = or i1 %o1, %t2.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store atomic i64 0, ptr %mem monotonic, align 8            ; enqueue_pos
  %dq.p = getelementptr inbounds nuw i8, ptr %mem, i64 128
  store atomic i64 0, ptr %dq.p monotonic, align 8           ; dequeue_pos
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 256
  store i64 %mask, ptr %mask.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 264
  store i64 %cap, ptr %cap.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %mem, i64 272
  store i64 %elem_size, ptr %esz.p, align 8
  %payoff = add i64 %sb.v, 320
  %payoff.p = getelementptr inbounds nuw i8, ptr %mem, i64 280
  store i64 %payoff, ptr %payoff.p, align 8
  %seqs = getelementptr inbounds nuw i8, ptr %mem, i64 320
  br label %seqinit

seqinit:
  %i = phi i64 [ 0, %init ], [ %i.n, %seqinit ]
  %sp = getelementptr inbounds nuw i64, ptr %seqs, i64 %i
  store i64 %i, ptr %sp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %cap
  br i1 %more, label %seqinit, label %done

done:
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_conc_mpsc_ring_push(ptr %rb, ptr %elem) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %rb.null, %elem.null
  br i1 %any.null, label %err.null, label %precheck, !prof !0

err.null:
  ret i32 1

precheck:
  %dq.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %head = load atomic i64, ptr %dq.p monotonic, align 8
  %pos0 = load atomic i64, ptr %rb monotonic, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %used = sub i64 %pos0, %head
  %full = icmp ugt i64 %used, %mask
  br i1 %full, label %err.full, label %claim, !prof !0

err.full:
  ret i32 6

claim:
  %myPos = atomicrmw add ptr %rb, i64 1 monotonic, align 8   ; XADD ticket
  %slot = and i64 %myPos, %mask
  %seqs = getelementptr inbounds nuw i8, ptr %rb, i64 320
  %seqp = getelementptr inbounds nuw i64, ptr %seqs, i64 %slot
  br label %spin

spin:
  %s = load atomic i64, ptr %seqp acquire, align 8
  %isfree = icmp eq i64 %s, %myPos
  br i1 %isfree, label %write, label %spin.wait, !prof !1

spin.wait:
  %y = call i32 @sched_yield()
  br label %spin

write:
  %esz.p = getelementptr inbounds nuw i8, ptr %rb, i64 272
  %esz = load i64, ptr %esz.p, align 8
  %payoff.p = getelementptr inbounds nuw i8, ptr %rb, i64 280
  %payoff = load i64, ptr %payoff.p, align 8
  %paybase = getelementptr inbounds nuw i8, ptr %rb, i64 %payoff
  %off = mul nuw i64 %slot, %esz
  %dst = getelementptr inbounds nuw i8, ptr %paybase, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %pub = add i64 %myPos, 1
  store atomic i64 %pub, ptr %seqp release, align 8          ; publish element
  ret i32 0
}

define i32 @universe_conc_mpsc_ring_pop(ptr %rb, ptr %out) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %rb.null, %out.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %dq.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %head = load atomic i64, ptr %dq.p monotonic, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %slot = and i64 %head, %mask
  %seqs = getelementptr inbounds nuw i8, ptr %rb, i64 320
  %seqp = getelementptr inbounds nuw i64, ptr %seqs, i64 %slot
  %s = load atomic i64, ptr %seqp acquire, align 8
  %want = add i64 %head, 1
  %ready = icmp eq i64 %s, %want
  br i1 %ready, label %read, label %err.empty, !prof !1

err.empty:
  ret i32 4

read:
  %esz.p = getelementptr inbounds nuw i8, ptr %rb, i64 272
  %esz = load i64, ptr %esz.p, align 8
  %payoff.p = getelementptr inbounds nuw i8, ptr %rb, i64 280
  %payoff = load i64, ptr %payoff.p, align 8
  %paybase = getelementptr inbounds nuw i8, ptr %rb, i64 %payoff
  %off = mul nuw i64 %slot, %esz
  %src = getelementptr inbounds nuw i8, ptr %paybase, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  %cap.p = getelementptr inbounds nuw i8, ptr %rb, i64 264
  %cap = load i64, ptr %cap.p, align 8
  %freeval = add i64 %head, %cap
  store atomic i64 %freeval, ptr %seqp release, align 8      ; slot reusable
  %newhead = add i64 %head, 1
  store atomic i64 %newhead, ptr %dq.p monotonic, align 8    ; publish progress
  ret i32 0
}

define i64 @universe_conc_mpsc_ring_count(ptr %rb) local_unnamed_addr #2 {
entry:
  %pos = load atomic i64, ptr %rb monotonic, align 8
  %dq.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %head = load atomic i64, ptr %dq.p monotonic, align 8
  %count = sub i64 %pos, %head
  ret i64 %count
}

define i64 @universe_conc_mpsc_ring_capacity(ptr %rb) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_conc_mpsc_ring_destroy(ptr %rb) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %rb, null
  br i1 %is.null, label %ret, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %rb)
  br label %ret

ret:
  ret void
}

; ============================================================================
;  seg — unbounded MPSC (paged, directory-addressed)
;    chunk layout (one allocation, chunk_bytes each):
;      @0    pool_next : ptr   (freelist link)
;      @8    base_pos  : i64
;      @64   ready[1024] : i64  (0 empty / 1 ready)
;      @8256 payload[1024] : esz
;    queue header (calloc 512):
;      @0    enqueue_pos : atomic i64   [own line]
;      @128  dequeue_pos : atomic i64   (consumer own)
;      @136  cons_chunk  : ptr
;      @144  cons_base   : i64
;      @256  dir : ptr (DIR_SIZE=65536 atomic ptr entries)
;      @264  esz : i64
;      @272  chunk_bytes : i64
;      @384  pool_head : ptr
;      @392  pool_lock : i32
; ============================================================================

; internal: acquire the pool spinlock (cold path).
define internal void @seg_lock(ptr %q) #3 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %q, i64 392
  br label %spin
spin:
  %prev = atomicrmw xchg ptr %lp, i32 1 acquire, align 4
  %held = icmp ne i32 %prev, 0
  br i1 %held, label %wait, label %got, !prof !0
wait:
  %y = call i32 @sched_yield()
  br label %spin
got:
  ret void
}

define internal void @seg_unlock(ptr %q) #3 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %q, i64 392
  store atomic i32 0, ptr %lp release, align 4
  ret void
}

; internal: obtain a chunk for %base — from the freelist (reset flags) or a
; fresh zeroed allocation. Returns null on OOM.
define internal ptr @seg_alloc_chunk(ptr %q, i64 %base) #3 {
entry:
  call void @seg_lock(ptr %q)
  %ph.p = getelementptr inbounds nuw i8, ptr %q, i64 384
  %ph = load ptr, ptr %ph.p, align 8
  %empty = icmp eq ptr %ph, null
  br i1 %empty, label %fresh, label %reuse

reuse:
  %nxt = load ptr, ptr %ph, align 8                 ; pool_next
  store ptr %nxt, ptr %ph.p, align 8
  call void @seg_unlock(ptr %q)
  %rd = getelementptr inbounds nuw i8, ptr %ph, i64 64
  call void @llvm.memset.p0.i64(ptr %rd, i8 0, i64 8192, i1 false)
  %bp1 = getelementptr inbounds nuw i8, ptr %ph, i64 8
  store i64 %base, ptr %bp1, align 8
  ret ptr %ph

fresh:
  call void @seg_unlock(ptr %q)
  %cb.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  %cb = load i64, ptr %cb.p, align 8
  %c = call ptr @calloc(i64 1, i64 %cb)
  %cnull = icmp eq ptr %c, null
  br i1 %cnull, label %oom, label %setbase

setbase:
  %bp2 = getelementptr inbounds nuw i8, ptr %c, i64 8
  store i64 %base, ptr %bp2, align 8
  ret ptr %c

oom:
  ret ptr null
}

; internal: return a fully-drained chunk to the freelist (consumer, or a
; producer that lost the install race).
define internal void @seg_recycle_chunk(ptr %q, ptr %c) #3 {
entry:
  call void @seg_lock(ptr %q)
  %ph.p = getelementptr inbounds nuw i8, ptr %q, i64 384
  %ph = load ptr, ptr %ph.p, align 8
  store ptr %ph, ptr %c, align 8                     ; c->pool_next = head
  store ptr %c, ptr %ph.p, align 8
  call void @seg_unlock(ptr %q)
  ret void
}

define noalias ptr @universe_conc_mpsc_seg_create(i64 %elem_size) local_unnamed_addr #1 {
entry:
  %bad = icmp eq i64 %elem_size, 0
  br i1 %bad, label %fail, label %shape, !prof !0

shape:
  %pb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 1024, i64 %elem_size)
  %pb.v = extractvalue { i64, i1 } %pb, 0
  %pb.o = extractvalue { i64, i1 } %pb, 1
  %cbt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pb.v, i64 8256)
  %cb = extractvalue { i64, i1 } %cbt, 0
  %cb.o = extractvalue { i64, i1 } %cbt, 1
  %ovf = or i1 %pb.o, %cb.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %q = call ptr @calloc(i64 1, i64 512)
  %q.null = icmp eq ptr %q, null
  br i1 %q.null, label %fail, label %alloc.dir, !prof !0

alloc.dir:
  %dir = call ptr @calloc(i64 65536, i64 8)
  %dir.null = icmp eq ptr %dir, null
  br i1 %dir.null, label %free.q, label %init, !prof !0

free.q:
  call void @free(ptr %q)
  br label %fail

init:
  %dir.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  store ptr %dir, ptr %dir.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  store i64 %elem_size, ptr %esz.p, align 8
  %cb.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  store i64 %cb, ptr %cb.p, align 8
  ret ptr %q

fail:
  ret ptr null
}

define i32 @universe_conc_mpsc_seg_push(ptr %q, ptr %elem) local_unnamed_addr #4 {
entry:
  %q.null = icmp eq ptr %q, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %q.null, %elem.null
  br i1 %any.null, label %err.null, label %claim, !prof !0

err.null:
  ret i32 1

claim:
  %pos = atomicrmw add ptr %q, i64 1 monotonic, align 8      ; XADD ticket
  %chunkIdx = lshr i64 %pos, 10
  %slotIdx = and i64 %pos, 1023
  %oob = icmp uge i64 %chunkIdx, 65536
  br i1 %oob, label %err.full, label %lookup, !prof !0

err.full:
  ret i32 6

lookup:
  %dir.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %dir = load ptr, ptr %dir.p, align 8
  %slotp = getelementptr inbounds nuw ptr, ptr %dir, i64 %chunkIdx
  %c0 = load atomic ptr, ptr %slotp acquire, align 8
  %isnull = icmp eq ptr %c0, null
  br i1 %isnull, label %install, label %have, !prof !0

install:
  %base = shl i64 %chunkIdx, 10
  %nc = call ptr @seg_alloc_chunk(ptr %q, i64 %base)
  %nc.null = icmp eq ptr %nc, null
  br i1 %nc.null, label %err.oom, label %try.link, !prof !0

err.oom:
  ret i32 2

try.link:
  %cx = cmpxchg ptr %slotp, ptr null, ptr %nc acq_rel acquire
  %old = extractvalue { ptr, i1 } %cx, 0
  %won = extractvalue { ptr, i1 } %cx, 1
  br i1 %won, label %have, label %lost

lost:
  call void @seg_recycle_chunk(ptr %q, ptr %nc)
  br label %have

have:
  %c = phi ptr [ %c0, %lookup ], [ %nc, %try.link ], [ %old, %lost ]
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %rdbase = getelementptr inbounds nuw i8, ptr %c, i64 64
  %readyp = getelementptr inbounds nuw i64, ptr %rdbase, i64 %slotIdx
  %paybase = getelementptr inbounds nuw i8, ptr %c, i64 8256
  %off = mul nuw i64 %slotIdx, %esz
  %dst = getelementptr inbounds nuw i8, ptr %paybase, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  store atomic i64 1, ptr %readyp release, align 8           ; publish element
  ret i32 0
}

define i32 @universe_conc_mpsc_seg_pop(ptr %q, ptr %out) local_unnamed_addr #4 {
entry:
  %q.null = icmp eq ptr %q, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %q.null, %out.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %pos = load atomic i64, ptr %dq.p monotonic, align 8
  %chunkIdx = lshr i64 %pos, 10
  %slotIdx = and i64 %pos, 1023
  %wantbase = shl i64 %chunkIdx, 10
  %cc.p = getelementptr inbounds nuw i8, ptr %q, i64 136
  %cons = load ptr, ptr %cc.p, align 8
  %cb.p = getelementptr inbounds nuw i8, ptr %q, i64 144
  %consbase = load i64, ptr %cb.p, align 8
  %cons.null = icmp eq ptr %cons, null
  %base.mismatch = icmp ne i64 %consbase, %wantbase
  %need = or i1 %cons.null, %base.mismatch
  br i1 %need, label %lookup, label %have, !prof !1

lookup:
  %dir.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %dir = load ptr, ptr %dir.p, align 8
  %slotp = getelementptr inbounds nuw ptr, ptr %dir, i64 %chunkIdx
  %c2 = load atomic ptr, ptr %slotp acquire, align 8
  %c2.null = icmp eq ptr %c2, null
  br i1 %c2.null, label %err.empty, label %setcons, !prof !0

err.empty:
  ret i32 4

setcons:
  store ptr %c2, ptr %cc.p, align 8
  store i64 %wantbase, ptr %cb.p, align 8
  br label %have

have:
  %c = phi ptr [ %cons, %load.own ], [ %c2, %setcons ]
  %rdbase = getelementptr inbounds nuw i8, ptr %c, i64 64
  %readyp = getelementptr inbounds nuw i64, ptr %rdbase, i64 %slotIdx
  %r = load atomic i64, ptr %readyp acquire, align 8
  %notready = icmp eq i64 %r, 0
  br i1 %notready, label %err.empty2, label %read, !prof !0

err.empty2:
  ret i32 4

read:
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %paybase = getelementptr inbounds nuw i8, ptr %c, i64 8256
  %off = mul nuw i64 %slotIdx, %esz
  %src = getelementptr inbounds nuw i8, ptr %paybase, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  %newpos = add i64 %pos, 1
  store atomic i64 %newpos, ptr %dq.p monotonic, align 8
  %islast = icmp eq i64 %slotIdx, 1023
  br i1 %islast, label %recycle, label %done, !prof !0

recycle:
  call void @seg_recycle_chunk(ptr %q, ptr %c)
  store ptr null, ptr %cc.p, align 8                          ; force re-lookup
  ret i32 0

done:
  ret i32 0
}

define i64 @universe_conc_mpsc_seg_count(ptr %q) local_unnamed_addr #2 {
entry:
  %pos = load atomic i64, ptr %q monotonic, align 8
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %head = load atomic i64, ptr %dq.p monotonic, align 8
  %count = sub i64 %pos, %head
  ret i64 %count
}

define void @universe_conc_mpsc_seg_destroy(ptr %q) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %q, null
  br i1 %is.null, label %ret, label %free.pool, !prof !0

free.pool:
  %ph.p = getelementptr inbounds nuw i8, ptr %q, i64 384
  %ph0 = load ptr, ptr %ph.p, align 8
  br label %ploop

ploop:
  %pc = phi ptr [ %ph0, %free.pool ], [ %pnxt, %pfree ]
  %pend = icmp eq ptr %pc, null
  br i1 %pend, label %free.cons, label %pfree

pfree:
  %pnxt = load ptr, ptr %pc, align 8
  call void @free(ptr %pc)
  br label %ploop

free.cons:
  %cc.p = getelementptr inbounds nuw i8, ptr %q, i64 136
  %cons = load ptr, ptr %cc.p, align 8
  %cons.null = icmp eq ptr %cons, null
  br i1 %cons.null, label %free.dir, label %do.cons

do.cons:
  call void @free(ptr %cons)
  br label %free.dir

free.dir:
  %dir.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %dir = load ptr, ptr %dir.p, align 8
  call void @free(ptr %dir)
  call void @free(ptr nonnull %q)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind memory(argmem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { nounwind willreturn norecurse memory(argmem: read) }
attributes #3 = { nounwind }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
