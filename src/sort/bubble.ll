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

; Bubble sort. O(N^2), stable, in-place.
;
; DESIGN (vs the typical C implementation):
;   * Last-swap optimization: the next pass only runs to where the previous
;     pass last swapped (strictly better than a naive boolean
;     early-exit — sorted tails are skipped entirely).
;   * Swaps are whole-element memcpys through a 1 KiB bounce buffer
;     (chunked for oversized elements), not byte loops.

define i32 @universe_sort_bubble(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
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
  br label %pass

pass:                                        ; scan [0, limit); track last swap
  %limit = phi i64 [ %count, %setup ], [ %last.swap, %pass.end ]
  %limit.small = icmp ult i64 %limit, 2
  br i1 %limit.small, label %fin, label %scan.pre

scan.pre:
  br label %scan

scan:
  %j = phi i64 [ 1, %scan.pre ], [ %j.n, %scan.latch ]
  %last = phi i64 [ 0, %scan.pre ], [ %last.n, %scan.latch ]
  %j.off = mul i64 %j, %elem_size
  %j.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %j.off
  %p.off = sub i64 %j.off, %elem_size
  %p.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %p.off
  %c = call i32 %cmp(ptr %j.ptr, ptr %p.ptr) #4
  %swap.needed = icmp slt i32 %c, 0
  br i1 %swap.needed, label %do.swap, label %scan.latch

do.swap:
  call void @swap_chunked_bb(ptr %p.ptr, ptr %j.ptr, i64 %elem_size, ptr nonnull %tmp)
  br label %scan.latch

scan.latch:
  %last.n = phi i64 [ %j, %do.swap ], [ %last, %scan ]
  %j.n = add nuw i64 %j, 1
  %more = icmp ult i64 %j.n, %limit
  br i1 %more, label %scan, label %pass.end

pass.end:
  %last.swap = phi i64 [ %last.n, %scan.latch ]
  br label %pass

fin:
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  br label %done

done:
  ret i32 0
}

define internal void @swap_chunked_bb(ptr noalias captures(none) %a, ptr noalias captures(none) %b, i64 %size, ptr noalias captures(none) %tmp) #0 {
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
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
