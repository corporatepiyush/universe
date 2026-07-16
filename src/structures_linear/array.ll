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

; Dynamic array (vector), arbitrary element size.
;
; DESIGN (vs the typical C implementation):
;   * insert/remove shift with ONE llvm.memmove (vectorized block move)
;     instead of an element-at-a-time loop.
;   * Doubling growth through realloc (often extends in place). Stable
;     32-byte handle; only the data pointer moves.
;   * All size math overflow-checked; error paths cold.
;   * Layout: { ptr data@0, i64 count@8, i64 cap@16, i64 elem@24 }.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 7 INVALID_INDEX):
;   ptr universe_ds_array_create(i64 elem_size, i64 initial_cap)
;   i32 universe_ds_array_push(ptr a, ptr elem)
;   i32 universe_ds_array_get(ptr a, i64 index, ptr out)
;   i32 universe_ds_array_set(ptr a, i64 index, ptr elem)
;   i32 universe_ds_array_insert(ptr a, i64 index, ptr elem) ; index <= count
;   i32 universe_ds_array_remove(ptr a, i64 index, ptr out)  ; out may be null
;   i64 universe_ds_array_count(ptr a)
;   i64 universe_ds_array_capacity(ptr a)
;   void universe_ds_array_destroy(ptr a)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_array_create(i64 %elem_size, i64 %initial_cap) local_unnamed_addr #1 {
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
  %hdr = call ptr @malloc(i64 32)
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
  %count.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 %cap, ptr %cap.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 %elem_size, ptr %elem.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

; grow to 2x cap; returns 0 ok / 2 oom. (internal, cold)
define internal i32 @arr_grow(ptr %a) #3 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %cap2 = shl i64 %cap, 1
  %wrapped = icmp eq i64 %cap2, 0
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 %esz)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  %ovf = or i1 %wrapped, %bytes.o
  br i1 %ovf, label %oom, label %do.realloc

do.realloc:
  %old = load ptr, ptr %a, align 8
  %new = call ptr @realloc(ptr %old, i64 %bytes.v)
  %new.null = icmp eq ptr %new, null
  br i1 %new.null, label %oom, label %ok

ok:
  store ptr %new, ptr %a, align 8
  store i64 %cap2, ptr %cap.p, align 8
  ret i32 0

oom:
  ret i32 2
}

define i32 @universe_ds_array_push(ptr %a, ptr %elem) local_unnamed_addr #1 {
entry:
  %a.null = icmp eq ptr %a, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %a.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %count, %cap
  br i1 %full, label %grow, label %store.elem, !prof !0

grow:
  %grc = call i32 @arr_grow(ptr nonnull %a)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %store.elem, label %oom, !prof !2

oom:
  ret i32 2

store.elem:
  %data = load ptr, ptr %a, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %count, %esz
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

define i32 @universe_ds_array_get(ptr %a, i64 %index, ptr %out) local_unnamed_addr #0 {
entry:
  %a.null = icmp eq ptr %a, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %a.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  %oob = icmp uge i64 %index, %count
  br i1 %oob, label %err.index, label %copy, !prof !0

err.index:
  ret i32 7

copy:
  %data = load ptr, ptr %a, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %index, %esz
  %src = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  ret i32 0
}

define i32 @universe_ds_array_set(ptr %a, i64 %index, ptr %elem) local_unnamed_addr #0 {
entry:
  %a.null = icmp eq ptr %a, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %a.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  %oob = icmp uge i64 %index, %count
  br i1 %oob, label %err.index, label %copy, !prof !0

err.index:
  ret i32 7

copy:
  %data = load ptr, ptr %a, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %index, %esz
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  ret i32 0
}

define i32 @universe_ds_array_insert(ptr %a, i64 %index, ptr %elem) local_unnamed_addr #1 {
entry:
  %a.null = icmp eq ptr %a, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %a.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  %oob = icmp ugt i64 %index, %count           ; index == count is append
  br i1 %oob, label %err.index, label %room, !prof !0

err.index:
  ret i32 7

room:
  %cap.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %count, %cap
  br i1 %full, label %grow, label %shift, !prof !0

grow:
  %grc = call i32 @arr_grow(ptr nonnull %a)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %shift, label %oom, !prof !2

oom:
  ret i32 2

shift:                                        ; ONE memmove opens the gap
  %data = load ptr, ptr %a, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %index, %esz
  %src = getelementptr inbounds nuw i8, ptr %data, i64 %off
  %dst = getelementptr inbounds nuw i8, ptr %src, i64 %esz
  %tail.elems = sub nuw i64 %count, %index
  %tail.bytes = mul nuw i64 %tail.elems, %esz
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %tail.bytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %src, ptr %elem, i64 %esz, i1 false)
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

define i32 @universe_ds_array_remove(ptr %a, i64 %index, ptr %out) local_unnamed_addr #0 {
entry:
  %a.null = icmp eq ptr %a, null
  br i1 %a.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  %oob = icmp uge i64 %index, %count
  br i1 %oob, label %err.index, label %fetch, !prof !0

err.index:
  ret i32 7

fetch:
  %data = load ptr, ptr %a, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %a, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %index, %esz
  %victim = getelementptr inbounds nuw i8, ptr %data, i64 %off
  %want.out = icmp ne ptr %out, null
  br i1 %want.out, label %copy.out, label %shift

copy.out:
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull %out, ptr %victim, i64 %esz, i1 false)
  br label %shift

shift:                                        ; ONE memmove closes the gap
  %next = getelementptr inbounds nuw i8, ptr %victim, i64 %esz
  %after = add nuw i64 %index, 1
  %tail.elems = sub nuw i64 %count, %after
  %tail.bytes = mul nuw i64 %tail.elems, %esz
  call void @llvm.memmove.p0.p0.i64(ptr %victim, ptr %next, i64 %tail.bytes, i1 false)
  %count.n = add i64 %count, -1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

define i64 @universe_ds_array_count(ptr %a) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define i64 @universe_ds_array_capacity(ptr %a) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define void @universe_ds_array_destroy(ptr %a) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %a, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %a, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %a)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { cold nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
