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

; Growable double-ended queue over a power-of-two ring.
;
; DESIGN (vs the typical C implementation):
;   * Same free-running-index ring as the queue, but head moves DOWN for
;     push_front (indices are unsigned and wrap coherently through the
;     mask); all four end-ops are O(1) with no modulo and no branchy wrap.
;   * Growth doubles and linearizes with at most two memcpys.
;   * Layout: { ptr data@0, i64 head@8, i64 tail@16, i64 mask@24, i64 elem@32 }.
;     Occupied slots are [head, tail) in free-running index space.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 4 EMPTY):
;   ptr universe_ds_deque_create(i64 elem_size, i64 initial_cap)
;   i32 universe_ds_deque_push_front(ptr d, ptr elem)
;   i32 universe_ds_deque_push_back(ptr d, ptr elem)
;   i32 universe_ds_deque_pop_front(ptr d, ptr out)
;   i32 universe_ds_deque_pop_back(ptr d, ptr out)
;   i32 universe_ds_deque_peek_front(ptr d, ptr out)
;   i32 universe_ds_deque_peek_back(ptr d, ptr out)
;   i64 universe_ds_deque_count(ptr d)
;   void universe_ds_deque_destroy(ptr d)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_deque_create(i64 %elem_size, i64 %initial_cap) local_unnamed_addr #1 {
entry:
  %elem.bad = icmp eq i64 %elem_size, 0
  br i1 %elem.bad, label %fail, label %shape, !prof !0

shape:
  %c.min = call i64 @llvm.umax.i64(i64 %initial_cap, i64 8)
  %too.big = icmp ugt i64 %c.min, 4611686018427387904
  br i1 %too.big, label %fail, label %pow2, !prof !0

pow2:
  %cm1 = add i64 %c.min, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 %elem_size)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  br i1 %bytes.o, label %fail, label %alloc, !prof !0

alloc:
  %hdr = call ptr @malloc(i64 40)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %alloc.data, !prof !0

alloc.data:
  %data = call ptr @malloc(i64 %bytes.v)
  %data.null = icmp eq ptr %data, null
  br i1 %data.null, label %free.hdr, label %init, !prof !0

free.hdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %data, ptr %hdr, align 8
  %head.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 0, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  %mask = add i64 %cap, -1
  store i64 %mask, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 %elem_size, ptr %elem.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

; grow 2x and linearize [head, tail) to [0, used); returns 0/2. cold.
define internal i32 @dq_grow(ptr %d) #3 {
entry:
  %head.p = getelementptr inbounds nuw i8, ptr %d, i64 8
  %head = load i64, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %d, i64 16
  %tail = load i64, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %d, i64 24
  %mask = load i64, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %d, i64 32
  %esz = load i64, ptr %elem.p, align 8
  %used = sub i64 %tail, %head
  %cap = add i64 %mask, 1
  %cap2 = shl i64 %cap, 1
  %wrapped = icmp eq i64 %cap2, 0
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 %esz)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  %ovf = or i1 %wrapped, %bytes.o
  br i1 %ovf, label %oom, label %alloc

alloc:
  %new = call ptr @malloc(i64 %bytes.v)
  %new.null = icmp eq ptr %new, null
  br i1 %new.null, label %oom, label %migrate

migrate:
  %old = load ptr, ptr %d, align 8
  %hslot = and i64 %head, %mask
  %hoff = mul i64 %hslot, %esz
  %first.src = getelementptr inbounds nuw i8, ptr %old, i64 %hoff
  %first.slots = sub i64 %cap, %hslot          ; grow => used == cap
  %first.bytes = mul i64 %first.slots, %esz
  call void @llvm.memcpy.p0.p0.i64(ptr %new, ptr %first.src, i64 %first.bytes, i1 false)
  %second.slots = sub i64 %used, %first.slots
  %second.bytes = mul i64 %second.slots, %esz
  %second.dst = getelementptr inbounds nuw i8, ptr %new, i64 %first.bytes
  call void @llvm.memcpy.p0.p0.i64(ptr %second.dst, ptr %old, i64 %second.bytes, i1 false)
  call void @free(ptr %old)
  store ptr %new, ptr %d, align 8
  store i64 0, ptr %head.p, align 8
  store i64 %used, ptr %tail.p, align 8
  %mask2 = add i64 %cap2, -1
  store i64 %mask2, ptr %mask.p, align 8
  ret i32 0

oom:
  ret i32 2
}

define i32 @universe_ds_deque_push_back(ptr %d, ptr %elem) local_unnamed_addr #1 {
entry:
  %d.null = icmp eq ptr %d, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %d.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head.p = getelementptr inbounds nuw i8, ptr %d, i64 8
  %head = load i64, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %d, i64 16
  %tail = load i64, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %d, i64 24
  %mask = load i64, ptr %mask.p, align 8
  %used = sub i64 %tail, %head
  %full = icmp ugt i64 %used, %mask
  br i1 %full, label %grow, label %store.pre, !prof !0

grow:
  %grc = call i32 @dq_grow(ptr nonnull %d)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %reload, label %oom, !prof !2

oom:
  ret i32 2

reload:
  %tail.r = load i64, ptr %tail.p, align 8
  %mask.r = load i64, ptr %mask.p, align 8
  br label %store.elem

store.pre:
  br label %store.elem

store.elem:
  %tail.f = phi i64 [ %tail, %store.pre ], [ %tail.r, %reload ]
  %mask.f = phi i64 [ %mask, %store.pre ], [ %mask.r, %reload ]
  %data = load ptr, ptr %d, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %d, i64 32
  %esz = load i64, ptr %elem.p, align 8
  %slot = and i64 %tail.f, %mask.f
  %off = mul i64 %slot, %esz
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %tail.n = add i64 %tail.f, 1
  store i64 %tail.n, ptr %tail.p, align 8
  ret i32 0
}

define i32 @universe_ds_deque_push_front(ptr %d, ptr %elem) local_unnamed_addr #1 {
entry:
  %d.null = icmp eq ptr %d, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %d.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head.p = getelementptr inbounds nuw i8, ptr %d, i64 8
  %head = load i64, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %d, i64 16
  %tail = load i64, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %d, i64 24
  %mask = load i64, ptr %mask.p, align 8
  %used = sub i64 %tail, %head
  %full = icmp ugt i64 %used, %mask
  br i1 %full, label %grow, label %store.pre, !prof !0

grow:
  %grc = call i32 @dq_grow(ptr nonnull %d)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %reload, label %oom, !prof !2

oom:
  ret i32 2

reload:
  %head.r = load i64, ptr %head.p, align 8
  %mask.r = load i64, ptr %mask.p, align 8
  br label %store.elem

store.pre:
  br label %store.elem

store.elem:                                   ; head moves DOWN; wraps fine
  %head.f = phi i64 [ %head, %store.pre ], [ %head.r, %reload ]
  %mask.f = phi i64 [ %mask, %store.pre ], [ %mask.r, %reload ]
  %head.n = add i64 %head.f, -1
  %data = load ptr, ptr %d, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %d, i64 32
  %esz = load i64, ptr %elem.p, align 8
  %slot = and i64 %head.n, %mask.f
  %off = mul i64 %slot, %esz
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  store i64 %head.n, ptr %head.p, align 8
  ret i32 0
}

; shared pop/peek core: which=0 front, 1 back; remove=1 pops
define internal i32 @dq_take(ptr %d, ptr %out, i1 %back, i1 %remove) #0 {
entry:
  %d.null = icmp eq ptr %d, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %d.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head.p = getelementptr inbounds nuw i8, ptr %d, i64 8
  %head = load i64, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %d, i64 16
  %tail = load i64, ptr %tail.p, align 8
  %empty = icmp eq i64 %head, %tail
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %mask.p = getelementptr inbounds nuw i8, ptr %d, i64 24
  %mask = load i64, ptr %mask.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %d, i64 32
  %esz = load i64, ptr %elem.p, align 8
  %tail.m1 = add i64 %tail, -1
  %idx = select i1 %back, i64 %tail.m1, i64 %head
  %slot = and i64 %idx, %mask
  %off = mul i64 %slot, %esz
  %data = load ptr, ptr %d, align 8
  %src = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  br i1 %remove, label %shrink, label %done

shrink:
  %head.n = add i64 %head, 1
  br i1 %back, label %pop.back, label %pop.front

pop.back:
  store i64 %tail.m1, ptr %tail.p, align 8
  br label %done

pop.front:
  store i64 %head.n, ptr %head.p, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_ds_deque_pop_front(ptr %d, ptr %out) local_unnamed_addr #0 {
entry:
  %r = call i32 @dq_take(ptr %d, ptr %out, i1 false, i1 true)
  ret i32 %r
}

define i32 @universe_ds_deque_pop_back(ptr %d, ptr %out) local_unnamed_addr #0 {
entry:
  %r = call i32 @dq_take(ptr %d, ptr %out, i1 true, i1 true)
  ret i32 %r
}

define i32 @universe_ds_deque_peek_front(ptr %d, ptr %out) local_unnamed_addr #0 {
entry:
  %r = call i32 @dq_take(ptr %d, ptr %out, i1 false, i1 false)
  ret i32 %r
}

define i32 @universe_ds_deque_peek_back(ptr %d, ptr %out) local_unnamed_addr #0 {
entry:
  %r = call i32 @dq_take(ptr %d, ptr %out, i1 true, i1 false)
  ret i32 %r
}

define i64 @universe_ds_deque_count(ptr %d) local_unnamed_addr #2 {
entry:
  %head.p = getelementptr inbounds nuw i8, ptr %d, i64 8
  %head = load i64, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %d, i64 16
  %tail = load i64, ptr %tail.p, align 8
  %count = sub i64 %tail, %head
  ret i64 %count
}

define void @universe_ds_deque_destroy(ptr %d) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %d, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %d, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %d)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { cold nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
