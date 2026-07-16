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

; Selection sort. O(N^2) compares but only O(N) element moves — the right
; N^2 sort when moves are expensive (huge elements).
;
; DESIGN (vs the typical C implementation):
;   * Min-index scan touches no data beyond the comparator; exactly one
;     swap per position (a naive swap was byte-at-a-time; ours is
;     block memcpy through a bounce buffer).
;   * Skips the swap entirely when the position already holds the minimum.

define i32 @universe_sort_selection(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
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
  %count.m1 = add i64 %count, -1
  br label %outer

outer:
  %i = phi i64 [ 0, %setup ], [ %i.n, %outer.latch ]
  %i.done = icmp uge i64 %i, %count.m1
  br i1 %i.done, label %fin, label %scan.pre

scan.pre:
  %i.off = mul i64 %i, %elem_size
  %i.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %i.off
  %i.next = add nuw i64 %i, 1
  br label %scan

scan:                                        ; find index of minimum in [i, count)
  %j = phi i64 [ %i.next, %scan.pre ], [ %j.n, %scan.latch ]
  %min = phi i64 [ %i, %scan.pre ], [ %min.n, %scan.latch ]
  %j.done = icmp uge i64 %j, %count
  br i1 %j.done, label %place, label %scan.body

scan.body:
  %j.off = mul i64 %j, %elem_size
  %j.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %j.off
  %min.off = mul i64 %min, %elem_size
  %min.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %min.off
  %c = call i32 %cmp(ptr %j.ptr, ptr %min.ptr) #4
  %smaller = icmp slt i32 %c, 0
  %min.n = select i1 %smaller, i64 %j, i64 %min
  br label %scan.latch

scan.latch:
  %j.n = add nuw i64 %j, 1
  br label %scan

place:
  %min.same = icmp eq i64 %min, %i
  br i1 %min.same, label %outer.latch, label %do.swap

do.swap:
  %m.off = mul i64 %min, %elem_size
  %m.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %m.off
  call void @swap_chunked_sel(ptr %i.ptr, ptr %m.ptr, i64 %elem_size, ptr nonnull %tmp)
  br label %outer.latch

outer.latch:
  %i.n = add nuw i64 %i, 1
  br label %outer

fin:
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  br label %done

done:
  ret i32 0
}

define internal void @swap_chunked_sel(ptr noalias captures(none) %a, ptr noalias captures(none) %b, i64 %size, ptr noalias captures(none) %tmp) #0 {
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
