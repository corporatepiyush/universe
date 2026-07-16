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

; SPSC (single-producer / single-consumer) lock-free ring buffer.
;
; DESIGN (vs the typical C implementation):
;   * No mutex. Producer owns `tail`, consumer owns `head`; each side reads
;     the other's index with ACQUIRE only when its CACHED copy says the ring
;     might be full/empty. In steady state a push is: one monotonic load of
;     your own index, one cached compare, one memcpy, one RELEASE store —
;     no shared-line ping-pong at all.
;   * Orderings, justified:
;       - store release tail  (push): publishes the memcpy'd element.
;       - load  acquire  tail (pop refresh): observes that element.
;       - store release head  (pop): publishes "slot is reusable".
;       - load  acquire  head (push refresh): slot contents are dead, safe
;         to overwrite. (acq needed: the overwrite must not be reordered
;         before we learn the consumer is done.)
;       - own-index loads are monotonic: single writer per index.
;   * Field padding (128B apart = distinct cache lines even on M-series):
;       consumer line: head(atomic)@0,  cached_tail@8
;       producer line: tail(atomic)@128, cached_head@136
;       shared r/o:    mask@256, elem@264
;       payload@320
;
; API (SPSC contract: ONE producer thread calls push, ONE consumer calls pop):
;   ptr universe_ds_cringbuf_create(i64 capacity, i64 elem_size)
;   i32 universe_ds_cringbuf_push(ptr rb, ptr elem)   ; 6 = FULL
;   i32 universe_ds_cringbuf_pop(ptr rb, ptr out)     ; 4 = EMPTY
;   i64 universe_ds_cringbuf_count(ptr rb)            ; approximate live
;   i64 universe_ds_cringbuf_capacity(ptr rb)
;   void universe_ds_cringbuf_destroy(ptr rb)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_cringbuf_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
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

define i32 @universe_ds_cringbuf_push(ptr %rb, ptr %elem) local_unnamed_addr #0 {
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

refresh:                                     ; cached view says full: recheck
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

define i32 @universe_ds_cringbuf_pop(ptr %rb, ptr %out) local_unnamed_addr #0 {
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

refresh:                                     ; cached view says empty: recheck
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

define i64 @universe_ds_cringbuf_count(ptr %rb) local_unnamed_addr #2 {
entry:
  %head = load atomic i64, ptr %rb acquire, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 128
  %tail = load atomic i64, ptr %tail.p acquire, align 8
  %count = sub i64 %tail, %head
  ret i64 %count
}

define i64 @universe_ds_cringbuf_capacity(ptr %rb) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 256
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_ds_cringbuf_destroy(ptr %rb) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %rb, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %rb)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
