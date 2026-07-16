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

; Heapsort. O(N log N) worst case, in-place, not stable.
;
; DESIGN (vs the typical C implementation):
;   * Fully iterative siftdown (no recursion, no stack risk).
;   * Swaps move whole elements through a 1 KiB stack bounce buffer with
;     llvm.memcpy (vectorized), chunked for oversized elements — never a
;     byte-at-a-time loop.
;   * Child selection is compare-once: pick the larger child, then a single
;     comparison against the root decides swap-or-stop.
;   * Contract: (base, count, elem_size, cmp) -> i32 error.

define i32 @universe_sort_heap(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
entry:
  %base.null = icmp eq ptr %base, null
  %cmp.null = icmp eq ptr %cmp, null
  %any.null = or i1 %base.null, %cmp.null
  br i1 %any.null, label %err.null, label %check.trivial, !prof !0

err.null:
  ret i32 1

check.trivial:
  %count.small = icmp ult i64 %count, 2
  %size.zero = icmp eq i64 %elem_size, 0
  %trivial = or i1 %count.small, %size.zero
  br i1 %trivial, label %done, label %setup, !prof !0

setup:
  %tmp = alloca [1024 x i8], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %tmp)
  ; build phase: siftdown i = count/2-1 .. 0
  %half = lshr i64 %count, 1
  br label %build

build:
  %bi = phi i64 [ %half, %setup ], [ %bi.n, %build.body ]
  %bi.zero = icmp eq i64 %bi, 0
  br i1 %bi.zero, label %drain.pre, label %build.body

build.body:
  %bi.n = add i64 %bi, -1
  call void @siftdown(ptr %base, i64 %bi.n, i64 %count, i64 %elem_size, ptr %cmp, ptr nonnull %tmp)
  br label %build

drain.pre:
  br label %drain

drain:                                       ; pop max to the end, shrink heap
  %end = phi i64 [ %count, %drain.pre ], [ %end.n, %drain.body ]
  %one.left = icmp ult i64 %end, 2
  br i1 %one.left, label %fin, label %drain.body

drain.body:
  %end.n = add i64 %end, -1
  %end.off = mul i64 %end.n, %elem_size
  %end.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %end.off
  call void @swap_chunked(ptr %base, ptr %end.ptr, i64 %elem_size, ptr nonnull %tmp)
  call void @siftdown(ptr %base, i64 0, i64 %end.n, i64 %elem_size, ptr %cmp, ptr nonnull %tmp)
  br label %drain

fin:
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  br label %done

done:
  ret i32 0
}

; classic iterative siftdown of %root within heap [0, %end)
define internal void @siftdown(ptr %base, i64 %root, i64 %end, i64 %esz, ptr %cmp, ptr %tmp) #2 {
entry:
  br label %loop

loop:
  %r = phi i64 [ %root, %entry ], [ %child.sel, %swap ]
  %c1 = shl i64 %r, 1
  %child = add i64 %c1, 1
  %no.child = icmp uge i64 %child, %end
  br i1 %no.child, label %exit, label %pick

pick:                                        ; larger of the two children
  %child.r = add i64 %child, 1
  %have.right = icmp ult i64 %child.r, %end
  br i1 %have.right, label %cmp.kids, label %chosen

cmp.kids:
  %cl.off = mul i64 %child, %esz
  %cl.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %cl.off
  %cr.off = mul i64 %child.r, %esz
  %cr.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %cr.off
  %kids = call i32 %cmp(ptr %cr.ptr, ptr %cl.ptr) #4
  %right.bigger = icmp sgt i32 %kids, 0
  %bigger = select i1 %right.bigger, i64 %child.r, i64 %child
  br label %chosen

chosen:
  %child.sel = phi i64 [ %child, %pick ], [ %bigger, %cmp.kids ]
  %cs.off = mul i64 %child.sel, %esz
  %cs.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %cs.off
  %r.off = mul i64 %r, %esz
  %r.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %r.off
  %vs = call i32 %cmp(ptr %cs.ptr, ptr %r.ptr) #4
  %child.wins = icmp sgt i32 %vs, 0
  br i1 %child.wins, label %swap, label %exit

swap:
  call void @swap_chunked(ptr %r.ptr, ptr %cs.ptr, i64 %esz, ptr %tmp)
  br label %loop

exit:
  ret void
}

; swap two equal-size regions through a 1 KiB bounce buffer, block at a time
define internal void @swap_chunked(ptr noalias captures(none) %a, ptr noalias captures(none) %b, i64 %size, ptr noalias captures(none) %tmp) #0 {
entry:
  br label %loop

loop:
  %off = phi i64 [ 0, %entry ], [ %off.next, %loop ]
  %left = sub nuw i64 %size, %off
  %n = call i64 @llvm.umin.i64(i64 %left, i64 1024)
  %a.p = getelementptr inbounds nuw i8, ptr %a, i64 %off
  %b.p = getelementptr inbounds nuw i8, ptr %b, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %tmp, ptr %a.p, i64 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %a.p, ptr %b.p, i64 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %b.p, ptr align 16 %tmp, i64 %n, i1 false)
  %off.next = add nuw i64 %off, %n
  %more = icmp ult i64 %off.next, %size
  br i1 %more, label %loop, label %exit

exit:
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)
declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.lifetime.end.p0(ptr captures(none))

attributes #0 = { alwaysinline nounwind willreturn nosync nofree norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
