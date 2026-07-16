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

; Growable array stack, arbitrary element size.
;
; DESIGN (vs the typical C implementation):
;   * Contiguous element array (top of stack = hottest line, always cached)
;     instead of any node-based layout; doubling growth via realloc — the
;     allocator can often extend in place, avoiding the copy entirely.
;   * Handle is a stable 32-byte header; only the data pointer moves on
;     growth. All size math overflow-checked.
;   * Layout: { ptr data@0, i64 count@8, i64 cap@16, i64 elem@24 }.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 4 EMPTY):
;   ptr universe_ds_stack_create(i64 elem_size, i64 initial_cap)
;   i32 universe_ds_stack_push(ptr st, ptr elem)
;   i32 universe_ds_stack_pop(ptr st, ptr out)
;   i32 universe_ds_stack_peek(ptr st, ptr out)
;   i64 universe_ds_stack_count(ptr st)
;   i64 universe_ds_stack_capacity(ptr st)
;   void universe_ds_stack_destroy(ptr st)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_stack_create(i64 %elem_size, i64 %initial_cap) local_unnamed_addr #1 {
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

define i32 @universe_ds_stack_push(ptr %st, ptr %elem) local_unnamed_addr #1 {
entry:
  %st.null = icmp eq ptr %st, null
  %elem.null = icmp eq ptr %elem, null
  %any.null = or i1 %st.null, %elem.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %st, i64 8
  %count = load i64, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %st, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %count, %cap
  br i1 %full, label %grow, label %store.elem, !prof !0

grow:
  %elem.p0 = getelementptr inbounds nuw i8, ptr %st, i64 24
  %esz0 = load i64, ptr %elem.p0, align 8
  %cap2 = shl i64 %cap, 1
  %wrapped = icmp eq i64 %cap2, 0
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 %esz0)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  %ovf = or i1 %wrapped, %bytes.o
  br i1 %ovf, label %err.oom, label %do.realloc, !prof !0

do.realloc:
  %old.data = load ptr, ptr %st, align 8
  %new.data = call ptr @realloc(ptr %old.data, i64 %bytes.v)
  %new.null = icmp eq ptr %new.data, null
  br i1 %new.null, label %err.oom, label %grown, !prof !0

grown:
  store ptr %new.data, ptr %st, align 8
  store i64 %cap2, ptr %cap.p, align 8
  br label %store.elem

err.oom:
  ret i32 2

store.elem:
  %data = load ptr, ptr %st, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %st, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %count, %esz
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 %esz, i1 false)
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

define i32 @universe_ds_stack_pop(ptr %st, ptr %out) local_unnamed_addr #0 {
entry:
  %st.null = icmp eq ptr %st, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %st.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %st, i64 8
  %count = load i64, ptr %count.p, align 8
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %count.n = add nsw i64 %count, -1
  %data = load ptr, ptr %st, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %st, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %count.n, %esz
  %src = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

define i32 @universe_ds_stack_peek(ptr %st, ptr %out) local_unnamed_addr #0 {
entry:
  %st.null = icmp eq ptr %st, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %st.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %count.p = getelementptr inbounds nuw i8, ptr %st, i64 8
  %count = load i64, ptr %count.p, align 8
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %top = add nsw i64 %count, -1
  %data = load ptr, ptr %st, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %st, i64 24
  %esz = load i64, ptr %elem.p, align 8
  %off = mul nuw i64 %top, %esz
  %src = getelementptr inbounds nuw i8, ptr %data, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %src, i64 %esz, i1 false)
  ret i32 0
}

define i64 @universe_ds_stack_count(ptr %st) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %st, i64 8
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define i64 @universe_ds_stack_capacity(ptr %st) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %st, i64 16
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define void @universe_ds_stack_destroy(ptr %st) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %st, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %st, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %st)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
