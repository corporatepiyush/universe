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

; Binary-heap priority queue over i64 keys (min-heap).
;
; DESIGN (functional spec: retrieve the minimum key in O(log n)):
;   * Implicit binary heap in a flat i64 array: children of i are 2i+1, 2i+2;
;     parent of i is (i-1)/2. No node pointers, no per-node allocation — the
;     whole heap is one contiguous, prefetch-friendly run.
;   * Stable 32-byte handle { ptr data@0, i64 len@8, i64 cap@16 } with the key
;     run in a SEPARATE malloc, exactly like the sibling array/stack: growth is
;     a realloc of the payload only, so the caller's handle never moves (a true
;     single-block header+payload could not grow without invalidating the
;     handle). cap is a power of two >= 8; doubling growth, all size math
;     overflow-checked.
;   * sift-up (push) and sift-down (pop) are tight index-math loops with the
;     hole carried in a register and a single store when it settles — the
;     classic "hole" optimization: we do not swap on every level, we shift the
;     smaller child up into the hole and drop the key in once. Fewer stores,
;     shorter dependency chain.
;   * MIN-heap over SIGNED i64 order (icmp slt), so negative keys sort below
;     positives and the negation trick below is exact. For a MAX-heap, negate
;     keys on the way in and out
;     (push(-k) / -pop()); an i64 key of INT64_MIN is the single value that
;     cannot be negated, so callers wanting a max-heap must avoid it. A
;     comparator flag was rejected: it would put an unpredictable branch (or an
;     indirect call) in the hottest inner compare of sift-down.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW, 4 EMPTY):
;   ptr universe_ds_binheap_create(i64 initial_cap)
;   void universe_ds_binheap_destroy(ptr h)
;   i32 universe_ds_binheap_push(ptr h, i64 key)
;   i32 universe_ds_binheap_pop(ptr h, ptr out_min)
;   i32 universe_ds_binheap_peek(ptr h, ptr out_min)
;   i64 universe_ds_binheap_len(ptr h)
;   i64 universe_ds_binheap_capacity(ptr h)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define noalias ptr @universe_ds_binheap_create(i64 %initial_cap) local_unnamed_addr #1 {
entry:
  %c.min = call i64 @llvm.umax.i64(i64 %initial_cap, i64 8)
  %too.big = icmp ugt i64 %c.min, 1152921504606846976   ; 2^60 keys ceiling
  br i1 %too.big, label %fail, label %pow2, !prof !0

pow2:
  %cm1 = add i64 %c.min, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %bytes = shl nuw i64 %cap, 3
  br label %alloc

alloc:
  %hdr = call ptr @malloc(i64 32)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %alloc.data, !prof !0

alloc.data:
  %data = call ptr @malloc(i64 %bytes)
  %data.null = icmp eq ptr %data, null
  br i1 %data.null, label %free.hdr, label %init, !prof !0

free.hdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %data, ptr %hdr, align 8
  %len.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %len.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 %cap, ptr %cap.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_binheap_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %h, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %h)
  br label %done

done:
  ret void
}

; grow the key run to 2x cap; 0 ok / 2 oom / 3 overflow. (internal, cold)
define internal i32 @binheap_grow(ptr %h) #3 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %cap2 = shl i64 %cap, 1
  %wrapped = icmp eq i64 %cap2, 0
  br i1 %wrapped, label %ovf, label %size

size:
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 8)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  br i1 %bytes.o, label %ovf, label %do.realloc

do.realloc:
  %old = load ptr, ptr %h, align 8
  %new = call ptr @realloc(ptr %old, i64 %bytes.v)
  %new.null = icmp eq ptr %new, null
  br i1 %new.null, label %oom, label %ok

ok:
  store ptr %new, ptr %h, align 8
  store i64 %cap2, ptr %cap.p, align 8
  ret i32 0

oom:
  ret i32 2

ovf:
  ret i32 3
}

define i32 @universe_ds_binheap_push(ptr %h, i64 %key) local_unnamed_addr #1 {
entry:
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %len, %cap
  br i1 %full, label %grow, label %siftup, !prof !0

grow:
  %grc = call i32 @binheap_grow(ptr nonnull %h)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %siftup, label %grow.fail, !prof !2

grow.fail:
  ret i32 %grc

siftup:
  ; carry a hole at index %len upward while parent > key.
  %data = load ptr, ptr %h, align 8
  br label %loop

loop:
  %hole = phi i64 [ %len, %siftup ], [ %parent, %shift ]
  %at.root = icmp eq i64 %hole, 0
  br i1 %at.root, label %settle, label %probe

probe:
  %hm1 = sub nuw i64 %hole, 1
  %parent = lshr i64 %hm1, 1
  %pp = getelementptr inbounds nuw i64, ptr %data, i64 %parent
  %pv = load i64, ptr %pp, align 8
  %need = icmp slt i64 %key, %pv
  br i1 %need, label %shift, label %settle

shift:
  %hp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %pv, ptr %hp, align 8
  br label %loop

settle:
  %sp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  %len.n = add nuw i64 %len, 1
  store i64 %len.n, ptr %len.p, align 8
  ret i32 0
}

define i32 @universe_ds_binheap_pop(ptr %h, ptr %out) local_unnamed_addr #0 {
entry:
  %h.null = icmp eq ptr %h, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %h.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %err.empty, label %pop, !prof !0

err.empty:
  ret i32 4

pop:
  %data = load ptr, ptr %h, align 8
  %min = load i64, ptr %data, align 8
  store i64 %min, ptr %out, align 8
  %len.n = add i64 %len, -1
  store i64 %len.n, ptr %len.p, align 8
  ; the last key fills the hole at the root and sifts down.
  %lastp = getelementptr inbounds nuw i64, ptr %data, i64 %len.n
  %key = load i64, ptr %lastp, align 8
  %was.last = icmp eq i64 %len.n, 0
  br i1 %was.last, label %done, label %loop

loop:
  %hole = phi i64 [ 0, %pop ], [ %smallest, %descend ]
  %left = shl i64 %hole, 1
  %left1 = or i64 %left, 1                       ; 2*hole + 1
  %has.left = icmp ult i64 %left1, %len.n
  br i1 %has.left, label %pick, label %settle

pick:
  %lp = getelementptr inbounds nuw i64, ptr %data, i64 %left1
  %lv = load i64, ptr %lp, align 8
  %right = add nuw i64 %left1, 1                 ; 2*hole + 2
  %has.right = icmp ult i64 %right, %len.n
  br i1 %has.right, label %pick.right, label %chose.left

pick.right:
  %rp = getelementptr inbounds nuw i64, ptr %data, i64 %right
  %rv = load i64, ptr %rp, align 8
  %r.smaller = icmp slt i64 %rv, %lv
  %child = select i1 %r.smaller, i64 %right, i64 %left1
  %cv = select i1 %r.smaller, i64 %rv, i64 %lv
  br label %compare

chose.left:
  br label %compare

compare:
  %smallest = phi i64 [ %child, %pick.right ], [ %left1, %chose.left ]
  %childv = phi i64 [ %cv, %pick.right ], [ %lv, %chose.left ]
  %need = icmp slt i64 %childv, %key
  br i1 %need, label %descend, label %settle

descend:
  %hp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %childv, ptr %hp, align 8
  br label %loop

settle:
  %sp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_ds_binheap_peek(ptr %h, ptr %out) local_unnamed_addr #0 {
entry:
  %h.null = icmp eq ptr %h, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %h.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %data = load ptr, ptr %h, align 8
  %min = load i64, ptr %data, align 8
  store i64 %min, ptr %out, align 8
  ret i32 0
}

define i64 @universe_ds_binheap_len(ptr %h) local_unnamed_addr #2 {
entry:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  ret i64 %len
}

define i64 @universe_ds_binheap_capacity(ptr %h) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { cold nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
