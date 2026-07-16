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

; Fixed-capacity ring buffer (single-threaded), arbitrary element size.
;
; DESIGN (vs the typical C implementation):
;   * Capacity is rounded UP to a power of two: slot = index & mask — no
;     modulo, no compare-and-wrap branch. capacity() reports the actual
;     (rounded) capacity.
;   * Indices are free-running u64 monotonic counters; count = tail - head
;     works across wrap without any extra state (and full/empty are one
;     subtraction each).
;   * Header + slots in ONE allocation, payload at +64 (headers never share
;     a line with data). Elements are packed at elem_size stride; memcpy
;     in/out lowers to vectorized copies for common sizes.
;   * Layout: { head@0, tail@8, mask@16, elem@24 }, payload@64.
;
; API (i32 error codes: 0 OK, 1 NULL_PTR, 4 EMPTY, 6 FULL, 7 INVALID_INDEX):
;   ptr universe_ds_ringbuf_create(i64 capacity, i64 elem_size)
;   i32 universe_ds_ringbuf_push(ptr rb, ptr elem)
;   i32 universe_ds_ringbuf_pop(ptr rb, ptr out)
;   i32 universe_ds_ringbuf_peek(ptr rb, i64 index, ptr out)  ; 0 = oldest
;   i64 universe_ds_ringbuf_count(ptr rb)
;   i64 universe_ds_ringbuf_capacity(ptr rb)
;   void universe_ds_ringbuf_destroy(ptr rb)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_ringbuf_create(i64 %capacity, i64 %elem_size) local_unnamed_addr #1 {
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
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bytes.v, i64 64)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  %ovf = or i1 %bytes.o, %tot.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 0, ptr %mem, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  %mask = add i64 %cap, -1
  store i64 %mask, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %elem_size, ptr %elem.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_ds_ringbuf_push(ptr %rb, ptr %elem) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %rb.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head = load i64, ptr %rb, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 8
  %tail = load i64, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 16
  %mask = load i64, ptr %mask.p, align 8
  %used = sub i64 %tail, %head
  %full = icmp ugt i64 %used, %mask          ; used > cap-1  <=>  used == cap
  br i1 %full, label %err.full, label %copy, !prof !0

err.full:
  ret i32 6

copy:
  %elem.p = getelementptr inbounds nuw i8, ptr %rb, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %slot = and i64 %tail, %mask
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %rb, i64 64
  %dst = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %tail.n = add i64 %tail, 1
  store i64 %tail.n, ptr %tail.p, align 8
  ret i32 0
}

define i32 @universe_ds_ringbuf_pop(ptr %rb, ptr %out) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %rb.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head = load i64, ptr %rb, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 8
  %tail = load i64, ptr %tail.p, align 8
  %empty = icmp eq i64 %head, %tail
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 16
  %mask = load i64, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %rb, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %slot = and i64 %head, %mask
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %rb, i64 64
  %src = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  %head.n = add i64 %head, 1
  store i64 %head.n, ptr %rb, align 8
  ret i32 0
}

define i32 @universe_ds_ringbuf_peek(ptr %rb, i64 %index, ptr %out) local_unnamed_addr #0 {
entry:
  %rb.null = icmp eq ptr %rb, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %rb.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head = load i64, ptr %rb, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 8
  %tail = load i64, ptr %tail.p, align 8
  %count = sub i64 %tail, %head
  %oob = icmp uge i64 %index, %count
  br i1 %oob, label %err.index, label %copy, !prof !0

err.index:
  ret i32 7

copy:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 16
  %mask = load i64, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %rb, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %pos = add i64 %head, %index
  %slot = and i64 %pos, %mask
  %off = mul nuw i64 %slot, %esz
  %payload = getelementptr inbounds nuw i8, ptr %rb, i64 64
  %src = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  ret i32 0
}

define i64 @universe_ds_ringbuf_count(ptr %rb) local_unnamed_addr #2 {
entry:
  %head = load i64, ptr %rb, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %rb, i64 8
  %tail = load i64, ptr %tail.p, align 8
  %count = sub i64 %tail, %head
  ret i64 %count
}

define i64 @universe_ds_ringbuf_capacity(ptr %rb) local_unnamed_addr #2 {
entry:
  %mask.p = getelementptr inbounds nuw i8, ptr %rb, i64 16
  %mask = load i64, ptr %mask.p, align 8
  %cap = add nuw i64 %mask, 1
  ret i64 %cap
}

define void @universe_ds_ringbuf_destroy(ptr %rb) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %rb, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %rb)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
