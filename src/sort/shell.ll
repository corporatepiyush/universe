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

; Shellsort with Knuth gaps (h = 3h+1). In-place, not stable.
;
; DESIGN (vs the typical C implementation):
;   * Gapped insertion HOLDS the moving element once (stack buffer) and
;     shifts run slots with whole-element memcpys — a naive implementation swapped
;     byte-by-byte per stride.
;   * Elements > 1 KiB use chunked swaps through the same buffer (in-place,
;     no allocation).

define i32 @universe_sort_shell(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
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
  %small.elem = icmp ule i64 %elem_size, 1024
  ; largest Knuth gap < count
  br label %gap.grow

gap.grow:
  %h = phi i64 [ 1, %setup ], [ %h3, %gap.grow ]
  %h3a = mul i64 %h, 3
  %h3 = add i64 %h3a, 1
  %fits = icmp ult i64 %h3, %count
  br i1 %fits, label %gap.grow, label %gaps

gaps:                                        ; h, then (h-1)/3, ... , 1
  %g = phi i64 [ %h, %gap.grow ], [ %g.next, %gap.done ]
  br label %outer

outer:                                       ; gapped insertion for gap g
  %i = phi i64 [ %g, %gaps ], [ %i.n, %outer.latch ]
  %i.done = icmp uge i64 %i, %count
  br i1 %i.done, label %gap.done, label %probe

probe:
  %cur.off = mul i64 %i, %elem_size
  %cur.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %cur.off
  %prev.idx = sub i64 %i, %g
  %prev.off = mul i64 %prev.idx, %elem_size
  %prev.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %prev.off
  %c0 = call i32 %cmp(ptr %cur.ptr, ptr %prev.ptr) #4
  %displaced = icmp slt i32 %c0, 0
  br i1 %displaced, label %displace, label %outer.latch, !prof !1

displace:
  br i1 %small.elem, label %hold, label %big.head

; ---- small path: hold once, shift strided slots with memcpy --------------

hold:
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 16 %tmp, ptr %cur.ptr, i64 %elem_size, i1 false)
  br label %scan

scan:                                        ; j walks down in gap strides
  %j = phi i64 [ %i, %hold ], [ %j.prev, %shift ]
  %j.prev = sub i64 %j, %g
  %jp.off = mul i64 %j.prev, %elem_size
  %jp.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %jp.off
  %j.off = mul i64 %j, %elem_size
  %j.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %j.off
  ; slot above gets predecessor element
  call void @llvm.memcpy.p0.p0.i64(ptr %j.ptr, ptr %jp.ptr, i64 %elem_size, i1 false)
  %has.more = icmp uge i64 %j.prev, %g
  br i1 %has.more, label %scan.cmp, label %place.low

scan.cmp:
  %pp.idx = sub i64 %j.prev, %g
  %pp.off = mul i64 %pp.idx, %elem_size
  %pp.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %pp.off
  %c1 = call i32 %cmp(ptr nonnull %tmp, ptr %pp.ptr) #4
  %still = icmp slt i32 %c1, 0
  br i1 %still, label %shift, label %place.low

shift:
  br label %scan

place.low:
  %slot.ptr = phi ptr [ %jp.ptr, %scan ], [ %jp.ptr, %scan.cmp ]
  call void @llvm.memcpy.p0.p0.i64(ptr %slot.ptr, ptr nonnull align 16 %tmp, i64 %elem_size, i1 false)
  br label %outer.latch

; ---- big path (>1KiB): strided chunked swaps ------------------------------

big.head:
  %bj = phi i64 [ %i, %displace ], [ %bj.prev, %big.step ]
  %bj.prev = sub i64 %bj, %g
  %a.off = mul i64 %bj, %elem_size
  %a.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %a.off
  %b.off = mul i64 %bj.prev, %elem_size
  %b.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %b.off
  call void @swap_chunked_sh(ptr %a.ptr, ptr %b.ptr, i64 %elem_size, ptr nonnull %tmp)
  %more.strides = icmp uge i64 %bj.prev, %g
  br i1 %more.strides, label %big.cmp, label %outer.latch

big.cmp:
  %bpp.idx = sub i64 %bj.prev, %g
  %bpp.off = mul i64 %bpp.idx, %elem_size
  %bpp.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %bpp.off
  %c2 = call i32 %cmp(ptr %b.ptr, ptr %bpp.ptr) #4
  %big.still = icmp slt i32 %c2, 0
  br i1 %big.still, label %big.step, label %outer.latch

big.step:
  br label %big.head

; ---------------------------------------------------------------------------

outer.latch:
  %i.n = add i64 %i, 1
  br label %outer

gap.done:
  %g.m1 = add i64 %g, -1
  %g.next = udiv i64 %g.m1, 3
  %more.gaps = icmp ugt i64 %g.next, 0
  br i1 %more.gaps, label %gaps, label %fin

fin:
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  br label %done

done:
  ret i32 0
}

define internal void @swap_chunked_sh(ptr noalias captures(none) %a, ptr noalias captures(none) %b, i64 %size, ptr noalias captures(none) %tmp) #0 {
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
!1 = !{!"branch_weights", i32 1, i32 3}
