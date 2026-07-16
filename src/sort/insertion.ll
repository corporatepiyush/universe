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

; Insertion sort. O(N^2) worst, O(N) nearly-sorted. Stable, in-place.
;
; DESIGN (vs the typical C implementation):
;   * A naive insertion sort swaps byte-by-byte per displaced position: shift*size
;     load/store pairs. Here the displaced element is
;     saved once to a stack buffer, the insertion point is found with
;     comparator calls only (no data movement), then the whole run is moved
;     with ONE llvm.memmove and the element placed with ONE llvm.memcpy —
;     both lower to vectorized block copies.
;   * Nearly-sorted fast path: one comparison per element, zero copies.
;   * elem_size > 1024 falls back to in-place chunked swaps (1 KiB blocks
;     through the same stack buffer) — still vectorized, never allocates.
;   * Error paths are cold (!prof); index math carries nuw/inbounds.

define i32 @universe_sort_insertion(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
entry:
  %base.null = icmp eq ptr %base, null
  %cmp.null = icmp eq ptr %cmp, null
  %any.null = or i1 %base.null, %cmp.null
  br i1 %any.null, label %err.null, label %check.trivial, !prof !0

err.null:                                         ; cold
  ret i32 1                                       ; UNIVERSE_ERR_NULL_PTR

check.trivial:
  %count.small = icmp ult i64 %count, 2
  %size.zero = icmp eq i64 %elem_size, 0
  %trivial = or i1 %count.small, %size.zero
  br i1 %trivial, label %done, label %setup, !prof !0

setup:
  %tmp = alloca [1024 x i8], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %tmp)
  %small.elem = icmp ule i64 %elem_size, 1024
  br label %outer.head

outer.head:                                       ; one iteration per element 1..count-1
  %i = phi i64 [ 1, %setup ], [ %i.next, %outer.latch ]
  %cur.off = mul nuw i64 %i, %elem_size
  %cur.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %cur.off
  %prev.idx = sub nuw i64 %i, 1
  %prev.off = sub nuw i64 %cur.off, %elem_size
  %prev.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %prev.off
  %c0 = call i32 %cmp(ptr %cur.ptr, ptr %prev.ptr) #4
  %displaced = icmp slt i32 %c0, 0
  br i1 %displaced, label %displace, label %outer.latch, !prof !1

displace:
  br i1 %small.elem, label %small.save, label %big.head

; ---- small elements: save once, scan, one memmove + one memcpy ----------

small.save:
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 16 %tmp, ptr %cur.ptr, i64 %elem_size, i1 false)
  br label %scan.head

scan.head:                                        ; invariant: tmp < elem[j] for all j in [j.cur, i)
  %j = phi i64 [ %prev.idx, %small.save ], [ %j.dec, %scan.step ]
  %j.zero = icmp eq i64 %j, 0
  br i1 %j.zero, label %scan.done, label %scan.cmp

scan.cmp:
  %j.dec = sub nuw i64 %j, 1
  %probe.off = mul nuw i64 %j.dec, %elem_size
  %probe.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %probe.off
  %c1 = call i32 %cmp(ptr nonnull %tmp, ptr %probe.ptr) #4
  %still.lt = icmp slt i32 %c1, 0
  br i1 %still.lt, label %scan.step, label %scan.done

scan.step:
  br label %scan.head

scan.done:                                        ; insert at slot j: shift [j, i) up one, place tmp
  %slot = phi i64 [ %j, %scan.head ], [ %j, %scan.cmp ]
  %slot.off = mul nuw i64 %slot, %elem_size
  %slot.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %slot.off
  %shift.dst = getelementptr inbounds nuw i8, ptr %slot.ptr, i64 %elem_size
  %run.len = sub nuw i64 %cur.off, %slot.off
  call void @llvm.memmove.p0.p0.i64(ptr %shift.dst, ptr %slot.ptr, i64 %run.len, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %slot.ptr, ptr nonnull align 16 %tmp, i64 %elem_size, i1 false)
  br label %outer.latch

; ---- oversized elements (>1 KiB): in-place chunked swap walk ------------

big.head:
  %bj = phi i64 [ %i, %displace ], [ %bj.dec, %big.step ]
  %bj.off = mul nuw i64 %bj, %elem_size
  %a.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %bj.off
  %bj.dec = sub nuw i64 %bj, 1
  %b.off = mul nuw i64 %bj.dec, %elem_size
  %b.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %b.off
  call void @swap_chunked(ptr %a.ptr, ptr %b.ptr, i64 %elem_size, ptr %tmp)
  %bj.more = icmp ugt i64 %bj.dec, 0
  br i1 %bj.more, label %big.cmp, label %outer.latch

big.cmp:
  %bprev.off = sub nuw i64 %b.off, %elem_size
  %bprev.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %bprev.off
  %c2 = call i32 %cmp(ptr %b.ptr, ptr %bprev.ptr) #4
  %big.lt = icmp slt i32 %c2, 0
  br i1 %big.lt, label %big.step, label %outer.latch

big.step:
  br label %big.head

; --------------------------------------------------------------------------

outer.latch:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %count
  br i1 %more, label %outer.head, label %exit.loop

exit.loop:
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  br label %done

done:
  ret i32 0                                       ; UNIVERSE_OK
}

; Swap two equal-size regions through a 1 KiB bounce buffer, block at a time.
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
declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)
declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.lifetime.end.p0(ptr captures(none))

attributes #0 = { alwaysinline nounwind willreturn nosync nofree norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 3}
