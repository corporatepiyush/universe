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

; Quicksort. O(N log N) expected, in-place, not stable.
;
; DESIGN (vs the typical C implementation):
;   * Median-of-3 pivot swapped to the hi slot + Lomuto partition. Lomuto
;     costs a few more swaps than Hoare but PROVABLY terminates and always
;     excludes the pivot slot from both subranges (guaranteed progress even
;     with all-equal inputs — the classic Hoare value-pivot livelock can't
;     happen).
;   * NO recursion: explicit 64-entry range stack (pushes the LARGER side,
;     loops on the smaller → depth <= log2 N always fits).
;   * Runs <= 16 are left unsorted, then ONE final insertion pass over the
;     whole array finishes the job (nearly-sorted input = its O(N) case,
;     single sequential sweep instead of per-run calls).
;   * Elements > 1 KiB delegate to universe_sort_heap (in-place, no pivot
;     buffer needed) — same contract, no allocation either way.

declare i32 @universe_sort_insertion(ptr, i64, i64, ptr)
declare i32 @universe_sort_heap(ptr, i64, i64, ptr)

define i32 @universe_sort_quick(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
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
  br i1 %trivial, label %done, label %route, !prof !0

route:
  %big.elem = icmp ugt i64 %elem_size, 1024
  br i1 %big.elem, label %delegate, label %setup, !prof !0

delegate:
  %hrc = call i32 @universe_sort_heap(ptr nonnull %base, i64 %count, i64 %elem_size, ptr nonnull %cmp)
  ret i32 %hrc

setup:
  %tmp = alloca [1024 x i8], align 16       ; swap bounce
  %pivot = alloca [1024 x i8], align 16     ; pivot value
  %stack = alloca [128 x i64], align 16     ; 64 x {lo, hi}
  call void @llvm.lifetime.start.p0(ptr nonnull %tmp)
  call void @llvm.lifetime.start.p0(ptr nonnull %pivot)
  call void @llvm.lifetime.start.p0(ptr nonnull %stack)
  %count.m1 = add i64 %count, -1
  br label %loop

; range stack: sp counts PAIRS. current range in (lo, hi) regs.
loop:
  %lo = phi i64 [ 0, %setup ], [ %lo.next, %continue ]
  %hi = phi i64 [ %count.m1, %setup ], [ %hi.next, %continue ]
  %sp = phi i64 [ 0, %setup ], [ %sp.next, %continue ]
  ; SIGNED compare: an empty/inverted range (pivot landed at an end) has
  ; len < 0 and must fall through to pop, not partition.
  %len = sub i64 %hi, %lo
  %small.run = icmp slt i64 %len, 16
  br i1 %small.run, label %pop, label %partition

pop:                                        ; take next range off the stack
  %sp.zero = icmp eq i64 %sp, 0
  br i1 %sp.zero, label %polish, label %pop.take

pop.take:
  %sp.m1 = add i64 %sp, -1
  %idx2 = shl i64 %sp.m1, 1
  %lo.slot = getelementptr inbounds nuw [128 x i64], ptr %stack, i64 0, i64 %idx2
  %plo = load i64, ptr %lo.slot, align 8
  %idx2b = or disjoint i64 %idx2, 1
  %hi.slot = getelementptr inbounds nuw [128 x i64], ptr %stack, i64 0, i64 %idx2b
  %phi.v = load i64, ptr %hi.slot, align 8
  br label %continue.from.pop

continue.from.pop:
  br label %continue

partition:
  ; median of 3 (lo, mid, hi); move the median into the hi slot
  %mid = add i64 %lo, %hi
  %mid.i = lshr i64 %mid, 1
  %lo.off = mul i64 %lo, %elem_size
  %lo.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %lo.off
  %mid.off = mul i64 %mid.i, %elem_size
  %mid.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %mid.off
  %hi.off = mul i64 %hi, %elem_size
  %hi.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %hi.off
  %c.lm = call i32 %cmp(ptr %lo.ptr, ptr %mid.ptr) #4
  %lm = icmp sgt i32 %c.lm, 0
  %min.lm = select i1 %lm, ptr %mid.ptr, ptr %lo.ptr
  %max.lm = select i1 %lm, ptr %lo.ptr, ptr %mid.ptr
  %c.mh = call i32 %cmp(ptr %max.lm, ptr %hi.ptr) #4
  %mh = icmp sgt i32 %c.mh, 0
  %upper.med = select i1 %mh, ptr %hi.ptr, ptr %max.lm
  %c.lm2 = call i32 %cmp(ptr %min.lm, ptr %upper.med) #4
  %lm2 = icmp sgt i32 %c.lm2, 0
  %median = select i1 %lm2, ptr %min.lm, ptr %upper.med
  %median.is.hi = icmp eq ptr %median, %hi.ptr
  br i1 %median.is.hi, label %lomuto.pre, label %park.pivot

park.pivot:
  call void @swap_chunked_qs(ptr %median, ptr %hi.ptr, i64 %elem_size, ptr nonnull %tmp)
  br label %lomuto.pre

lomuto.pre:
  br label %lomuto

lomuto:                                     ; store index walks; pivot at hi
  %store.i = phi i64 [ %lo, %lomuto.pre ], [ %store.next, %lomuto.latch ]
  %scan.j2 = phi i64 [ %lo, %lomuto.pre ], [ %scan.next, %lomuto.latch ]
  %scan.done = icmp uge i64 %scan.j2, %hi
  br i1 %scan.done, label %place.pivot, label %lomuto.body

lomuto.body:
  %sj.off = mul i64 %scan.j2, %elem_size
  %sj.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %sj.off
  %cl = call i32 %cmp(ptr %sj.ptr, ptr %hi.ptr) #4
  %belongs.left = icmp slt i32 %cl, 0
  br i1 %belongs.left, label %lomuto.swap, label %lomuto.latch

lomuto.swap:
  %si.off = mul i64 %store.i, %elem_size
  %si.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %si.off
  %same.slot = icmp eq i64 %store.i, %scan.j2
  br i1 %same.slot, label %lomuto.bump, label %lomuto.doswap

lomuto.doswap:
  call void @swap_chunked_qs(ptr %si.ptr, ptr %sj.ptr, i64 %elem_size, ptr nonnull %tmp)
  br label %lomuto.bump

lomuto.bump:
  %store.i.up = add i64 %store.i, 1
  br label %lomuto.latch

lomuto.latch:
  %store.next = phi i64 [ %store.i, %lomuto.body ], [ %store.i.up, %lomuto.bump ]
  %scan.next = add i64 %scan.j2, 1
  br label %lomuto

place.pivot:                                ; swap pivot into its final slot
  %fin.off = mul i64 %store.i, %elem_size
  %fin.ptr = getelementptr inbounds nuw i8, ptr %base, i64 %fin.off
  %pivot.home = icmp eq i64 %store.i, %hi
  br i1 %pivot.home, label %split, label %place.doswap

place.doswap:
  call void @swap_chunked_qs(ptr %fin.ptr, ptr %hi.ptr, i64 %elem_size, ptr nonnull %tmp)
  br label %split

split:                                      ; ranges exclude the pivot slot:
  %p.m1 = add i64 %store.i, -1              ; [lo, p-1] and [p+1, hi]
  %left.len = sub i64 %p.m1, %lo
  %right.lo = add i64 %store.i, 1
  %right.len = sub i64 %hi, %right.lo
  ; signed lens: empty/inverted sides (pivot at an end) go negative and the
  ; loop's signed small-run check pops them harmlessly. Push larger side,
  ; iterate the smaller.
  %left.smaller = icmp slt i64 %left.len, %right.len
  %push.lo = select i1 %left.smaller, i64 %right.lo, i64 %lo
  %push.hi = select i1 %left.smaller, i64 %hi, i64 %p.m1
  %next.lo = select i1 %left.smaller, i64 %lo, i64 %right.lo
  %next.hi = select i1 %left.smaller, i64 %p.m1, i64 %hi
  %idx2c = shl i64 %sp, 1
  %lo.slot2 = getelementptr inbounds nuw [128 x i64], ptr %stack, i64 0, i64 %idx2c
  store i64 %push.lo, ptr %lo.slot2, align 8
  %idx2d = or disjoint i64 %idx2c, 1
  %hi.slot2 = getelementptr inbounds nuw [128 x i64], ptr %stack, i64 0, i64 %idx2d
  store i64 %push.hi, ptr %hi.slot2, align 8
  %sp.up = add i64 %sp, 1
  br label %continue.from.split

continue.from.split:
  br label %continue

continue:
  %lo.next = phi i64 [ %plo, %continue.from.pop ], [ %next.lo, %continue.from.split ]
  %hi.next = phi i64 [ %phi.v, %continue.from.pop ], [ %next.hi, %continue.from.split ]
  %sp.next = phi i64 [ %sp.m1, %continue.from.pop ], [ %sp.up, %continue.from.split ]
  br label %loop

polish:                                     ; one insertion pass finishes runs
  call void @llvm.lifetime.end.p0(ptr nonnull %tmp)
  call void @llvm.lifetime.end.p0(ptr nonnull %pivot)
  call void @llvm.lifetime.end.p0(ptr nonnull %stack)
  %irc = call i32 @universe_sort_insertion(ptr nonnull %base, i64 %count, i64 %elem_size, ptr nonnull %cmp)
  ret i32 %irc

done:
  ret i32 0
}

define internal void @swap_chunked_qs(ptr noalias captures(none) %a, ptr noalias captures(none) %b, i64 %size, ptr noalias captures(none) %tmp) #0 {
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
