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

; universe_conc_spmc / universe_conc_mpmc — lock-free BOUNDED queues with
; PER-SLOT SEQUENCE numbers. Completes the SPSC/MPSC/SPMC/MPMC family.
;
; ============================================================================
; DESIGN — per-slot-sequence bounded ring (the standard lock-free MPMC
;          discipline; implemented from first principles)
; ----------------------------------------------------------------------------
;   A monotone free-running index loaded from a shared counter does NOT by
;   itself make it safe to touch a slot's NON-atomic payload — that is a data
;   race. The fix is a PER-SLOT SEQUENCE number (an atomic i64 embedded in each
;   cell) that GATES access and carries the happens-before edge:
;
;     * A producer may write slot p only when its seq == p (the slot is empty
;       and belongs to this lap). It claims position p by CAS-advancing the
;       shared enqueue index, memcpys the payload, then RELEASE-stores
;       seq = p+1 to PUBLISH the item.
;     * A consumer may read slot p only when its seq == p+1 (an item is
;       present). It claims p by CAS-advancing the shared dequeue index,
;       ACQUIRE-loads the seq (synchronizes-with the producer's release, so the
;       payload bytes are visible), memcpys out, then RELEASE-stores
;       seq = p+capacity to hand the slot to the producer of the NEXT lap.
;
;   Why this is ABA- and loss-free: the seq is strictly monotone across laps
;   (p, p+1, p+cap, p+cap+1, p+2*cap, ...). A stale claimant's CAS on the index
;   fails (index already advanced) and retries; it never touches a slot out of
;   turn because the seq test is re-evaluated against the freshly-loaded index.
;   FULL and EMPTY fall straight out of the seq comparison — no separate flags.
;
;   Cell layout (stride = round_up(8 + elem_size, 8), so seq stays 8-aligned):
;     cell+0 : seq (atomic i64)
;     cell+8 : payload (elem_size bytes)
;
;   Header (producer / consumer indices on SEPARATE 128 B lines => no
;   false-sharing ping-pong between the two ends):
;     producer line : enqueue_pos/tail (atomic i64) @0
;     consumer line : dequeue_pos/head (atomic i64) @128
;     shared r/o    : mask @256, elem_size @264, stride @272
;     cells         : @320 .. @(320 + cap*stride)
;
;   Orderings (each justified; NO seq_cst anywhere):
;     - index load (own start / reload)    : monotonic (just a claim counter;
;         the payload edge rides on the seq, not the index)
;     - index claim CAS                    : monotonic/monotonic (relaxed) —
;         it publishes nothing; synchronization is on the seq
;     - seq load (both ends)               : ACQUIRE (observe the peer's slot
;         write / free before we touch the payload)
;     - seq publish (producer p+1)         : RELEASE (make payload visible)
;     - seq free    (consumer p+capacity)  : RELEASE (make "slot reusable"
;         visible; and order our payload READ before the producer overwrites)
;
;   SPMC specialization: exactly ONE producer, so the producer OWNS the tail
;   index — no CAS on the write side (load own tail monotonic, seq-gate, write,
;   RELEASE seq, publish tail monotonic). Consumers are the full multi-consumer
;   CAS+seq path. Consumers contend only on `head`; the producer is wait-free.
;
;   MPMC: both ends CAS their shared index; the seq discipline is identical.
;
; API (bounded; power-of-two capacity, index & (cap-1)):
;   ptr  universe_conc_{spmc,mpmc}_create(i64 capacity, i64 elem_size)
;   i32  universe_conc_{spmc,mpmc}_enqueue(ptr q, ptr elem)   ; 6 FULL, 1 NULL
;   i32  universe_conc_{spmc,mpmc}_dequeue(ptr q, ptr out)    ; 4 EMPTY,1 NULL
;   i64  universe_conc_{spmc,mpmc}_count(ptr q)               ; approximate
;   i64  universe_conc_{spmc,mpmc}_capacity(ptr q)
;   void universe_conc_{spmc,mpmc}_destroy(ptr q)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; internal: allocate + initialize a per-slot-sequence ring (shared by both
; SPMC and MPMC — the layout is identical). alwaysinline => zero cost at the
; two thin exported constructors.
; ===========================================================================
define internal ptr @conc_ring_new(i64 %capacity, i64 %elem_size) #1 {
entry:
  %cap.bad  = icmp eq i64 %capacity, 0
  %elem.bad = icmp eq i64 %elem_size, 0
  %too.big  = icmp ugt i64 %capacity, 4611686018427387904
  %bad0 = or i1 %cap.bad, %elem.bad
  %bad  = or i1 %bad0, %too.big
  br i1 %bad, label %fail, label %shape, !prof !0

shape:
  ; round capacity up to a power of two (min 8)
  %c.min = call i64 @llvm.umax.i64(i64 %capacity, i64 8)
  %cm1   = add i64 %c.min, -1
  %lz    = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap   = shl nuw i64 1, %shift
  ; stride = round_up(8 + elem_size, 8)
  %s0 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %elem_size, i64 8)
  %s0.v = extractvalue { i64, i1 } %s0, 0
  %s0.o = extractvalue { i64, i1 } %s0, 1
  %s1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s0.v, i64 7)
  %s1.v = extractvalue { i64, i1 } %s1, 0
  %s1.o = extractvalue { i64, i1 } %s1, 1
  %stride = and i64 %s1.v, -8
  ; bytes = cap * stride ; total = 320 + bytes
  %b = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 %stride)
  %b.v = extractvalue { i64, i1 } %b, 0
  %b.o = extractvalue { i64, i1 } %b, 1
  %t = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %b.v, i64 320)
  %t.v = extractvalue { i64, i1 } %t, 0
  %t.o = extractvalue { i64, i1 } %t, 1
  %o0 = or i1 %s0.o, %s1.o
  %o1 = or i1 %o0, %b.o
  %ovf = or i1 %o1, %t.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %t.v)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store atomic i64 0, ptr %mem monotonic, align 8          ; enqueue_pos/tail @0
  %dq.p = getelementptr inbounds nuw i8, ptr %mem, i64 128
  store atomic i64 0, ptr %dq.p monotonic, align 8         ; dequeue_pos/head @128
  %mask = add i64 %cap, -1
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 256
  store i64 %mask, ptr %mask.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %mem, i64 264
  store i64 %elem_size, ptr %esz.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %mem, i64 272
  store i64 %stride, ptr %stride.p, align 8
  %cellbase = getelementptr inbounds nuw i8, ptr %mem, i64 320
  br label %seqinit

seqinit:                                    ; slot i starts with seq = i (empty)
  %i = phi i64 [ 0, %init ], [ %i.n, %seqinit ]
  %off = mul nuw i64 %i, %stride
  %cell = getelementptr inbounds nuw i8, ptr %cellbase, i64 %off
  store atomic i64 %i, ptr %cell monotonic, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %cap
  br i1 %more, label %seqinit, label %done

done:
  ret ptr %mem

fail:
  ret ptr null
}

; ===========================================================================
; MPMC — multi-producer / multi-consumer
; ===========================================================================
define noalias ptr @universe_conc_mpmc_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
entry:
  %r = tail call ptr @conc_ring_new(i64 %capacity, i64 %elem_size)
  ret ptr %r
}

define i32 @universe_conc_mpmc_enqueue(ptr %q, ptr %elem) local_unnamed_addr #0 {
entry:
  %q.null = icmp eq ptr %q, null
  %e.null = icmp eq ptr %elem, null
  %anynull = or i1 %q.null, %e.null
  br i1 %anynull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  %stride = load i64, ptr %stride.p, align 8
  %cellbase = getelementptr inbounds nuw i8, ptr %q, i64 320
  %pos0 = load atomic i64, ptr %q monotonic, align 8       ; enqueue_pos
  br label %loop

loop:
  %pos = phi i64 [ %pos0, %setup ], [ %pos.re, %reload ], [ %old, %cas.fail ]
  %slot = and i64 %pos, %mask
  %celloff = mul nuw i64 %slot, %stride
  %cell = getelementptr inbounds nuw i8, ptr %cellbase, i64 %celloff
  %seq = load atomic i64, ptr %cell acquire, align 8
  %diff = sub i64 %seq, %pos
  %free = icmp eq i64 %diff, 0
  br i1 %free, label %claim, label %notfree

notfree:
  %full = icmp slt i64 %diff, 0
  br i1 %full, label %err.full, label %reload, !prof !0

err.full:
  ret i32 6

reload:
  %pos.re = load atomic i64, ptr %q monotonic, align 8
  br label %loop

claim:
  %posn = add i64 %pos, 1
  %cx = cmpxchg weak ptr %q, i64 %pos, i64 %posn monotonic monotonic
  %old = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %write, label %cas.fail

cas.fail:
  br label %loop

write:
  %data = getelementptr inbounds nuw i8, ptr %cell, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %data, ptr %elem, i64 %esz, i1 false)
  store atomic i64 %posn, ptr %cell release, align 8       ; PUBLISH item
  ret i32 0
}

define i32 @universe_conc_mpmc_dequeue(ptr %q, ptr %out) local_unnamed_addr #0 {
entry:
  %q.null = icmp eq ptr %q, null
  %o.null = icmp eq ptr %out, null
  %anynull = or i1 %q.null, %o.null
  br i1 %anynull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  %stride = load i64, ptr %stride.p, align 8
  %cellbase = getelementptr inbounds nuw i8, ptr %q, i64 320
  %cap = add i64 %mask, 1
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %pos0 = load atomic i64, ptr %dq.p monotonic, align 8    ; dequeue_pos
  br label %loop

loop:
  %pos = phi i64 [ %pos0, %setup ], [ %pos.re, %reload ], [ %old, %cas.fail ]
  %slot = and i64 %pos, %mask
  %celloff = mul nuw i64 %slot, %stride
  %cell = getelementptr inbounds nuw i8, ptr %cellbase, i64 %celloff
  %seq = load atomic i64, ptr %cell acquire, align 8
  %posp1 = add i64 %pos, 1
  %diff = sub i64 %seq, %posp1
  %ready = icmp eq i64 %diff, 0
  br i1 %ready, label %claim, label %notready

notready:
  %empty = icmp slt i64 %diff, 0
  br i1 %empty, label %err.empty, label %reload, !prof !0

err.empty:
  ret i32 4

reload:
  %pos.re = load atomic i64, ptr %dq.p monotonic, align 8
  br label %loop

claim:
  %cx = cmpxchg weak ptr %dq.p, i64 %pos, i64 %posp1 monotonic monotonic
  %old = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %read, label %cas.fail

cas.fail:
  br label %loop

read:
  %data = getelementptr inbounds nuw i8, ptr %cell, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %data, i64 %esz, i1 false)
  %freeseq = add i64 %pos, %cap
  store atomic i64 %freeseq, ptr %cell release, align 8    ; slot reusable next lap
  ret i32 0
}

define i64 @universe_conc_mpmc_count(ptr %q) local_unnamed_addr #2 {
entry:
  %eq = load atomic i64, ptr %q acquire, align 8
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %dq = load atomic i64, ptr %dq.p acquire, align 8
  %c = sub i64 %eq, %dq
  ret i64 %c
}

define i64 @universe_conc_mpmc_capacity(ptr %q) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_conc_mpmc_destroy(ptr %q) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %q, null
  br i1 %is.null, label %ret, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %q)
  br label %ret

ret:
  ret void
}

; ===========================================================================
; SPMC — single-producer / multi-consumer
;   Consumers share the MPMC dequeue path (CAS head + seq gate). The producer
;   OWNS the tail: no CAS on the write side => the producer is wait-free.
; ===========================================================================
define noalias ptr @universe_conc_spmc_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
entry:
  %r = tail call ptr @conc_ring_new(i64 %capacity, i64 %elem_size)
  ret ptr %r
}

define i32 @universe_conc_spmc_enqueue(ptr %q, ptr %elem) local_unnamed_addr #0 {
entry:
  %q.null = icmp eq ptr %q, null
  %e.null = icmp eq ptr %elem, null
  %anynull = or i1 %q.null, %e.null
  br i1 %anynull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  %stride = load i64, ptr %stride.p, align 8
  %cellbase = getelementptr inbounds nuw i8, ptr %q, i64 320
  %pos = load atomic i64, ptr %q monotonic, align 8        ; own tail (single producer)
  %slot = and i64 %pos, %mask
  %celloff = mul nuw i64 %slot, %stride
  %cell = getelementptr inbounds nuw i8, ptr %cellbase, i64 %celloff
  %seq = load atomic i64, ptr %cell acquire, align 8       ; observe consumer freed it
  %free = icmp eq i64 %seq, %pos
  br i1 %free, label %write, label %err.full, !prof !0

err.full:
  ret i32 6

write:
  %data = getelementptr inbounds nuw i8, ptr %cell, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %data, ptr %elem, i64 %esz, i1 false)
  %posn = add i64 %pos, 1
  store atomic i64 %posn, ptr %cell release, align 8       ; PUBLISH item
  store atomic i64 %posn, ptr %q monotonic, align 8        ; advance own tail
  ret i32 0
}

define i32 @universe_conc_spmc_dequeue(ptr %q, ptr %out) local_unnamed_addr #0 {
entry:
  %q.null = icmp eq ptr %q, null
  %o.null = icmp eq ptr %out, null
  %anynull = or i1 %q.null, %o.null
  br i1 %anynull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 264
  %esz = load i64, ptr %esz.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %q, i64 272
  %stride = load i64, ptr %stride.p, align 8
  %cellbase = getelementptr inbounds nuw i8, ptr %q, i64 320
  %cap = add i64 %mask, 1
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %pos0 = load atomic i64, ptr %dq.p monotonic, align 8    ; dequeue_pos
  br label %loop

loop:
  %pos = phi i64 [ %pos0, %setup ], [ %pos.re, %reload ], [ %old, %cas.fail ]
  %slot = and i64 %pos, %mask
  %celloff = mul nuw i64 %slot, %stride
  %cell = getelementptr inbounds nuw i8, ptr %cellbase, i64 %celloff
  %seq = load atomic i64, ptr %cell acquire, align 8
  %posp1 = add i64 %pos, 1
  %diff = sub i64 %seq, %posp1
  %ready = icmp eq i64 %diff, 0
  br i1 %ready, label %claim, label %notready

notready:
  %empty = icmp slt i64 %diff, 0
  br i1 %empty, label %err.empty, label %reload, !prof !0

err.empty:
  ret i32 4

reload:
  %pos.re = load atomic i64, ptr %dq.p monotonic, align 8
  br label %loop

claim:
  %cx = cmpxchg weak ptr %dq.p, i64 %pos, i64 %posp1 monotonic monotonic
  %old = extractvalue { i64, i1 } %cx, 0
  %ok = extractvalue { i64, i1 } %cx, 1
  br i1 %ok, label %read, label %cas.fail

cas.fail:
  br label %loop

read:
  %data = getelementptr inbounds nuw i8, ptr %cell, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %data, i64 %esz, i1 false)
  %freeseq = add i64 %pos, %cap
  store atomic i64 %freeseq, ptr %cell release, align 8    ; slot reusable next lap
  ret i32 0
}

define i64 @universe_conc_spmc_count(ptr %q) local_unnamed_addr #2 {
entry:
  %eq = load atomic i64, ptr %q acquire, align 8
  %dq.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %dq = load atomic i64, ptr %dq.p acquire, align 8
  %c = sub i64 %eq, %dq
  ret i64 %c
}

define i64 @universe_conc_spmc_capacity(ptr %q) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_conc_spmc_destroy(ptr %q) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %q, null
  br i1 %is.null, label %ret, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %q)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
