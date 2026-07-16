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

; universe_conc_spsc — single-producer / single-consumer wait-free queues.
; First member of the concurrent-queue family (MPSC/SPMC/MPMC follow).
;
; Two structures:
;   1) BOUNDED wait-free ring  (universe_conc_spsc_*)
;   2) UNBOUNDED linked-array / segment queue (universe_conc_spsc_seg_*)
;
; ============================================================================
; DESIGN — BOUNDED RING (power-of-two capacity, single allocation)
; ----------------------------------------------------------------------------
;   The producer OWNS `tail`; the consumer OWNS `head`. Each side reads only
;   its own index in steady state and consults a private CACHED copy of the
;   other side, refreshing that cache (an ACQUIRE load of the peer index) only
;   when the cached view claims full/empty. So a hot enqueue is: one monotonic
;   load of tail, one cached compare, one memcpy, one RELEASE store — no
;   shared cache line is touched, no CAS, no retry loop => WAIT-FREE.
;
;   Free-running u64 indices, mask wrap: slot = idx & mask;
;   used = tail - head is wrap-safe (unsigned) as long as capacity <= 2^63.
;   full  <=> used  > mask   (used == mask+1 == capacity)
;   empty <=> head == tail
;
;   Orderings (each justified):
;     - own-index load  : monotonic  (single writer per index; no publish)
;     - publish tail     : RELEASE    (makes the memcpy'd slot visible)
;     - refresh tail (pop): ACQUIRE   (observe that slot's bytes)
;     - publish head     : RELEASE    (slot is now reusable)
;     - refresh head(push): ACQUIRE   (the overwrite must not float above the
;                                      proof the consumer finished the slot)
;
;   Layout (128 B line separation so producer/consumer never false-share):
;     consumer line: head(atomic)@0,  cached_tail@8
;     producer line: tail(atomic)@128, cached_head@136
;     shared r/o   : mask@256, elem_size@264
;     payload      : @320
;
; ============================================================================
; DESIGN — SEGMENT QUEUE (unbounded; bounded-ring speed inside a chunk)
; ----------------------------------------------------------------------------
;   Fixed 1024-slot chunks linked by a `next` pointer. The producer appends to
;   the tail chunk; on crossing a chunk boundary it obtains a fresh chunk,
;   writes the element, links `prev.next = new` (RELEASE) then publishes the
;   free-running prod index (RELEASE). The consumer drains from the head chunk;
;   on crossing a boundary it follows `head.next` (ACQUIRE) and retires the old
;   chunk. Inside a chunk the mechanics are the bounded ring's: own-index
;   monotonic, peer-index acquire, publish release. No CAS anywhere => the data
;   path is WAIT-FREE.
;
;   Ordering proof for the boundary hand-off: the consumer observes prod>cons
;   via an ACQUIRE load of prod_idx; that synchronizes-with the producer's
;   RELEASE store of prod_idx, which was preceded (program order) by the memcpy
;   into the new chunk and the RELEASE store of prev.next=new. Hence the new
;   chunk pointer and its slot-0 bytes are both visible. The consumer's ACQUIRE
;   load of head.next is belt-and-suspenders on the same edge.
;
;   POOLING is SPSC-safe: retired chunks are handed producer<-consumer through
;   an embedded fixed wait-free ring of chunk pointers (consumer OWNS
;   recycle_tail, producer OWNS recycle_head; publish release / observe
;   acquire, same shape as the data ring). Ring full on return => free(); ring
;   empty on demand => malloc(). No shared freelist, no CAS.
;
;   Header layout:
;     consumer line: cons_idx(atomic)@0, head_chunk@8, recycle_tail(atomic)@16
;     producer line: prod_idx(atomic)@128, tail_chunk@136, recycle_head(atomic)@144
;     shared r/o   : elem_size@256
;     recycle slots: @320 .. @1343   (128 ptr entries * 8 B)
;   Chunk layout: next(atomic ptr)@0, payload@64 (1024 * elem_size bytes)
;
;   Concurrency is single-threaded-correct AND weak-memory correct (verified by
;   a 1P/1C >=1M-item strict-FIFO stress on ARM64 at -O0 and -O3).

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ===========================================================================
; BOUNDED RING
; ===========================================================================

define noalias ptr @universe_conc_spsc_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
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
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 %elem_size)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bytes.v, i64 320)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  %ovf = or i1 %bytes.o, %tot.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store atomic i64 0, ptr %mem monotonic, align 8            ; head
  %ct.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %ct.p, align 8                            ; cached_tail
  %tail.p = getelementptr inbounds nuw i8, ptr %mem, i64 128
  store atomic i64 0, ptr %tail.p monotonic, align 8         ; tail
  %ch.p = getelementptr inbounds nuw i8, ptr %mem, i64 136
  store i64 0, ptr %ch.p, align 8                            ; cached_head
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 256
  %mask = add i64 %cap, -1
  store i64 %mask, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %mem, i64 264
  store i64 %elem_size, ptr %elem.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_conc_spsc_enqueue(ptr %rb, ptr %elem) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %rb.null, %elem.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p monotonic, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %ch.p = getelementptr inbounds nuw i8, ptr %rb, i64 136
  %cached.head = load i64, ptr %ch.p, align 8
  %used.c = sub i64 %tail, %cached.head
  %maybe.full = icmp ugt i64 %used.c, %mask
  br i1 %maybe.full, label %refresh, label %copy, !prof !0

refresh:
  %head = load atomic i64, ptr %rb acquire, align 8
  store i64 %head, ptr %ch.p, align 8
  %used.r = sub i64 %tail, %head
  %really.full = icmp ugt i64 %used.r, %mask
  br i1 %really.full, label %err.full, label %copy

err.full:
  ret i32 6

copy:
  %elem.szp = getelementptr inbounds nuw i8, ptr %rb, i64 264
  %esz = load i64, ptr %elem.szp, align 8
  %slot = and i64 %tail, %mask
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %rb, i64 320
  %dst = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %tail.n = add i64 %tail, 1
  store atomic i64 %tail.n, ptr %tail.p release, align 8     ; publish element
  ret i32 0
}

define i32 @universe_conc_spsc_dequeue(ptr %rb, ptr %out) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %rb.null, %out.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %head = load atomic i64, ptr %rb monotonic, align 8
  %ct.p = getelementptr inbounds nuw i8, ptr %rb, i64 8
  %cached.tail = load i64, ptr %ct.p, align 8
  %maybe.empty = icmp eq i64 %head, %cached.tail
  br i1 %maybe.empty, label %refresh, label %copy, !prof !0

refresh:
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p acquire, align 8      ; see the element
  store i64 %tail, ptr %ct.p, align 8
  %really.empty = icmp eq i64 %head, %tail
  br i1 %really.empty, label %err.empty, label %copy

err.empty:
  ret i32 4

copy:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %elem.szp = getelementptr inbounds nuw i8, ptr %rb, i64 264
  %esz = load i64, ptr %elem.szp, align 8
  %slot = and i64 %head, %mask
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %rb, i64 320
  %src = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  %head.n = add i64 %head, 1
  store atomic i64 %head.n, ptr %rb release, align 8         ; slot reusable
  ret i32 0
}

define i64 @universe_conc_spsc_count(ptr %rb) local_unnamed_addr #2 {
entry:
  %head = load atomic i64, ptr %rb acquire, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p acquire, align 8
  %count = sub i64 %tail, %head
  ret i64 %count
}

define i32 @universe_conc_spsc_is_empty(ptr %rb) local_unnamed_addr #2 {
entry:
  %head = load atomic i64, ptr %rb acquire, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p acquire, align 8
  %eq = icmp eq i64 %head, %tail
  %r = zext i1 %eq to i32
  ret i32 %r
}

define i32 @universe_conc_spsc_is_full(ptr %rb) local_unnamed_addr #2 {
entry:
  %head = load atomic i64, ptr %rb acquire, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p acquire, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %used = sub i64 %tail, %head
  %full = icmp ugt i64 %used, %mask
  %r = zext i1 %full to i32
  ret i32 %r
}

define i64 @universe_conc_spsc_capacity(ptr %rb) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_conc_spsc_destroy(ptr %rb) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %rb, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %rb)
  br label %done

done:
  ret void
}

; ===========================================================================
; SEGMENT QUEUE (unbounded)
; ===========================================================================

define noalias ptr @universe_conc_spsc_seg_create(i64 %elem_size) local_unnamed_addr #1 {
entry:
  %elem.bad = icmp eq i64 %elem_size, 0
  br i1 %elem.bad, label %fail.null, label %shape, !prof !0

shape:
  %pbytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 1024, i64 %elem_size)
  %pbytes.v = extractvalue { i64, i1 } %pbytes, 0
  %pbytes.o = extractvalue { i64, i1 } %pbytes, 1
  %ctot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pbytes.v, i64 64)
  %ctot.v = extractvalue { i64, i1 } %ctot, 0
  %ctot.o = extractvalue { i64, i1 } %ctot, 1
  %ovf = or i1 %pbytes.o, %ctot.o
  br i1 %ovf, label %fail.null, label %alloc.hdr, !prof !0

alloc.hdr:
  %hdr = call ptr @malloc(i64 1344)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail.null, label %alloc.chunk, !prof !0

alloc.chunk:
  %chunk = call ptr @malloc(i64 %ctot.v)
  %chunk.null = icmp eq ptr %chunk, null
  br i1 %chunk.null, label %fail.hdr, label %init, !prof !0

init:
  store atomic i64 0, ptr %hdr monotonic, align 8              ; cons_idx
  %hc.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store ptr %chunk, ptr %hc.p, align 8                         ; head_chunk
  %rt.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store atomic i64 0, ptr %rt.p monotonic, align 8            ; recycle_tail
  %prod.p = getelementptr inbounds nuw i8, ptr %hdr, i64 128
  store atomic i64 0, ptr %prod.p monotonic, align 8          ; prod_idx
  %tc.p = getelementptr inbounds nuw i8, ptr %hdr, i64 136
  store ptr %chunk, ptr %tc.p, align 8                         ; tail_chunk
  %rh.p = getelementptr inbounds nuw i8, ptr %hdr, i64 144
  store atomic i64 0, ptr %rh.p monotonic, align 8            ; recycle_head
  %esz.p = getelementptr inbounds nuw i8, ptr %hdr, i64 256
  store i64 %elem_size, ptr %esz.p, align 8                    ; elem_size
  store atomic ptr null, ptr %chunk monotonic, align 8        ; chunk.next
  ret ptr %hdr

fail.hdr:
  call void @free(ptr nonnull %hdr)
  ret ptr null

fail.null:
  ret ptr null
}

define i32 @universe_conc_spsc_seg_enqueue(ptr %q, ptr %elem) local_unnamed_addr #3 {
entry:
  %q.null = icmp eq ptr %q, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %q.null, %elem.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %prod.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %prod = load atomic i64, ptr %prod.p monotonic, align 8
  %slot = and i64 %prod, 1023
  %slot0 = icmp eq i64 %slot, 0
  %nz = icmp ne i64 %prod, 0
  %at.edge = and i1 %slot0, %nz
  br i1 %at.edge, label %edge, label %inplace, !prof !0

inplace:
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %esz = load i64, ptr %esz.p, align 8
  %tc.p = getelementptr inbounds nuw i8, ptr %q, i64 136
  %chunk = load ptr, ptr %tc.p, align 8
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %chunk, i64 64
  %dst = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %prod.n = add i64 %prod, 1
  store atomic i64 %prod.n, ptr %prod.p release, align 8       ; publish element
  ret i32 0

edge:                                    ; slot 0 of a new chunk
  %rh.p = getelementptr inbounds nuw i8, ptr %q, i64 144
  %rh = load atomic i64, ptr %rh.p monotonic, align 8
  %rt.p = getelementptr inbounds nuw i8, ptr %q, i64 16
  %rt = load atomic i64, ptr %rt.p acquire, align 8
  %pool.empty = icmp eq i64 %rh, %rt
  br i1 %pool.empty, label %do.malloc, label %do.recycle

do.recycle:                              ; reuse a retired chunk
  %slots = getelementptr inbounds nuw i8, ptr %q, i64 320
  %ridx = and i64 %rh, 127
  %roff = mul nuw i64 %ridx, 8
  %sp = getelementptr inbounds nuw i8, ptr %slots, i64 %roff
  %newc.r = load ptr, ptr %sp, align 8
  %rh.n = add i64 %rh, 1
  store atomic i64 %rh.n, ptr %rh.p release, align 8
  br label %have.chunk

do.malloc:
  %esz.m = getelementptr inbounds nuw i8, ptr %q, i64 256
  %esz.mv = load i64, ptr %esz.m, align 8
  %pb = mul i64 1024, %esz.mv
  %ct = add i64 %pb, 64
  %newc.m = call ptr @malloc(i64 %ct)
  %m.null = icmp eq ptr %newc.m, null
  br i1 %m.null, label %err.oom, label %have.chunk, !prof !0

err.oom:
  ret i32 2

have.chunk:
  %newc = phi ptr [ %newc.r, %do.recycle ], [ %newc.m, %do.malloc ]
  %esz2.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %esz2 = load i64, ptr %esz2.p, align 8
  %payload.n = getelementptr inbounds nuw i8, ptr %newc, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %payload.n, ptr %elem, i64 %esz2, i1 false)
  store atomic ptr null, ptr %newc monotonic, align 8          ; newc.next = null
  %tc.p2 = getelementptr inbounds nuw i8, ptr %q, i64 136
  %oldtail = load ptr, ptr %tc.p2, align 8
  store atomic ptr %newc, ptr %oldtail release, align 8        ; link prev.next
  store ptr %newc, ptr %tc.p2, align 8                         ; tail_chunk = newc
  %prod.n2 = add i64 %prod, 1
  store atomic i64 %prod.n2, ptr %prod.p release, align 8      ; publish
  ret i32 0
}

define i32 @universe_conc_spsc_seg_dequeue(ptr %q, ptr %out) local_unnamed_addr #3 {
entry:
  %q.null = icmp eq ptr %q, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %q.null, %out.null
  br i1 %any.null, label %err.null, label %load.own, !prof !0

err.null:
  ret i32 1

load.own:
  %cons = load atomic i64, ptr %q monotonic, align 8
  %prod.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %prod = load atomic i64, ptr %prod.p acquire, align 8
  %empty = icmp eq i64 %cons, %prod
  br i1 %empty, label %err.empty, label %proceed, !prof !0

err.empty:
  ret i32 4

proceed:
  %slot = and i64 %cons, 1023
  %slot0 = icmp eq i64 %slot, 0
  %nz = icmp ne i64 %cons, 0
  %at.edge = and i1 %slot0, %nz
  br i1 %at.edge, label %edge, label %inplace, !prof !0

inplace:
  %hc.p = getelementptr inbounds nuw i8, ptr %q, i64 8
  %chunk.i = load ptr, ptr %hc.p, align 8
  br label %do.copy

edge:                                    ; first slot lives in the next chunk
  %hc.p2 = getelementptr inbounds nuw i8, ptr %q, i64 8
  %old = load ptr, ptr %hc.p2, align 8
  %next = load atomic ptr, ptr %old acquire, align 8          ; old.next
  store ptr %next, ptr %hc.p2, align 8                         ; head_chunk = next
  %rt.p = getelementptr inbounds nuw i8, ptr %q, i64 16
  %rt = load atomic i64, ptr %rt.p monotonic, align 8
  %rh.p = getelementptr inbounds nuw i8, ptr %q, i64 144
  %rh = load atomic i64, ptr %rh.p acquire, align 8
  %used = sub i64 %rt, %rh
  %pool.full = icmp ugt i64 %used, 127
  br i1 %pool.full, label %do.free, label %do.push

do.free:
  call void @free(ptr %old)
  br label %after.recycle

do.push:
  %slots = getelementptr inbounds nuw i8, ptr %q, i64 320
  %ridx = and i64 %rt, 127
  %roff = mul nuw i64 %ridx, 8
  %sp = getelementptr inbounds nuw i8, ptr %slots, i64 %roff
  store ptr %old, ptr %sp, align 8
  %rt.n = add i64 %rt, 1
  store atomic i64 %rt.n, ptr %rt.p release, align 8
  br label %after.recycle

after.recycle:
  br label %do.copy

do.copy:
  %chunk = phi ptr [ %chunk.i, %inplace ], [ %next, %after.recycle ]
  %esz.p = getelementptr inbounds nuw i8, ptr %q, i64 256
  %esz = load i64, ptr %esz.p, align 8
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %chunk, i64 64
  %src = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  %cons.n = add i64 %cons, 1
  store atomic i64 %cons.n, ptr %q release, align 8            ; slot consumed
  ret i32 0
}

define i64 @universe_conc_spsc_seg_count(ptr %q) local_unnamed_addr #2 {
entry:
  %cons = load atomic i64, ptr %q acquire, align 8
  %prod.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %prod = load atomic i64, ptr %prod.p acquire, align 8
  %count = sub i64 %prod, %cons
  ret i64 %count
}

define i32 @universe_conc_spsc_seg_is_empty(ptr %q) local_unnamed_addr #2 {
entry:
  %cons = load atomic i64, ptr %q acquire, align 8
  %prod.p = getelementptr inbounds nuw i8, ptr %q, i64 128
  %prod = load atomic i64, ptr %prod.p acquire, align 8
  %eq = icmp eq i64 %cons, %prod
  %r = zext i1 %eq to i32
  ret i32 %r
}

define void @universe_conc_spsc_seg_destroy(ptr %q) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %q, null
  br i1 %is.null, label %ret.block, label %walk.init, !prof !0

walk.init:
  %hc.p = getelementptr inbounds nuw i8, ptr %q, i64 8
  %hc = load ptr, ptr %hc.p, align 8
  br label %walk

walk:
  %cur = phi ptr [ %hc, %walk.init ], [ %nxt, %walk.body ]
  %cur.null = icmp eq ptr %cur, null
  br i1 %cur.null, label %list.done, label %walk.body

walk.body:
  %nxt = load atomic ptr, ptr %cur monotonic, align 8
  call void @free(ptr %cur)
  br label %walk

list.done:
  %rh.p = getelementptr inbounds nuw i8, ptr %q, i64 144
  %rh = load i64, ptr %rh.p, align 8
  %rt.p = getelementptr inbounds nuw i8, ptr %q, i64 16
  %rt = load i64, ptr %rt.p, align 8
  br label %rwalk

rwalk:
  %ri = phi i64 [ %rh, %list.done ], [ %ri.n, %rwalk.body ]
  %rdone = icmp uge i64 %ri, %rt
  br i1 %rdone, label %free.hdr, label %rwalk.body

rwalk.body:
  %slots = getelementptr inbounds nuw i8, ptr %q, i64 320
  %ridx = and i64 %ri, 127
  %roff = mul nuw i64 %ridx, 8
  %sp = getelementptr inbounds nuw i8, ptr %slots, i64 %roff
  %rc = load ptr, ptr %sp, align 8
  call void @free(ptr %rc)
  %ri.n = add i64 %ri, 1
  br label %rwalk

free.hdr:
  call void @free(ptr nonnull %q)
  br label %ret.block

ret.block:
  ret void
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse }

!0 = !{!"branch_weights", i32 1, i32 2000}
