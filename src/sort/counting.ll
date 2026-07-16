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

; Counting sort for i32 arrays. O(N + range).
;
; DESIGN (vs the typical C implementation):
;   * Min/max in ONE fused pass (auto-vectorizes: smin/smax reductions).
;   * Output written as value runs directly from the histogram — no second
;     "positions" prefix array, no scatter: purely sequential stores.
;   * Range guarded (<= 2^26 buckets) so hostile inputs can't OOM; wider
;     ranges return INVALID_ARG(8) — callers use radix/quick for those.
;
; API: i32 universe_sort_counting(ptr base /*i32*/, i64 count)

define i32 @universe_sort_counting(ptr %base, i64 %count) local_unnamed_addr #1 {
entry:
  %base.null = icmp eq ptr %base, null
  br i1 %base.null, label %err.null, label %check.trivial, !prof !0

err.null:
  ret i32 1

check.trivial:
  %small = icmp ult i64 %count, 2
  br i1 %small, label %done, label %minmax.pre, !prof !0

minmax.pre:
  %first = load i32, ptr %base, align 4
  br label %minmax

minmax:                                      ; fused min+max sweep
  %i = phi i64 [ 1, %minmax.pre ], [ %i.n, %minmax ]
  %mn = phi i32 [ %first, %minmax.pre ], [ %mn.n, %minmax ]
  %mx = phi i32 [ %first, %minmax.pre ], [ %mx.n, %minmax ]
  %p = getelementptr inbounds nuw i32, ptr %base, i64 %i
  %v = load i32, ptr %p, align 4
  %mn.n = call i32 @llvm.smin.i32(i32 %mn, i32 %v)
  %mx.n = call i32 @llvm.smax.i32(i32 %mx, i32 %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %minmax, label %shape, !llvm.loop !2

shape:
  %mn.w = sext i32 %mn.n to i64
  %mx.w = sext i32 %mx.n to i64
  %span = sub nsw i64 %mx.w, %mn.w
  %range = add nuw nsw i64 %span, 1
  %too.wide = icmp ugt i64 %range, 67108864
  br i1 %too.wide, label %err.range, label %alloc, !prof !0

err.range:
  ret i32 8

alloc:
  %bytes = shl nuw i64 %range, 3
  %counts = call ptr @calloc(i64 %range, i64 8)
  %counts.null = icmp eq ptr %counts, null
  br i1 %counts.null, label %err.oom, label %tally, !prof !0

err.oom:
  ret i32 2

tally:                                       ; histogram
  %j = phi i64 [ 0, %alloc ], [ %j.n, %tally ]
  %tp = getelementptr inbounds nuw i32, ptr %base, i64 %j
  %tv = load i32, ptr %tp, align 4
  %tv.w = sext i32 %tv to i64
  %bucket = sub nsw i64 %tv.w, %mn.w
  %cp = getelementptr inbounds nuw i64, ptr %counts, i64 %bucket
  %c = load i64, ptr %cp, align 8
  %c.n = add nuw i64 %c, 1
  store i64 %c.n, ptr %cp, align 8
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %count
  br i1 %more2, label %tally, label %emit.pre

emit.pre:
  br label %emit

emit:                                        ; write runs; sequential stores
  %b = phi i64 [ 0, %emit.pre ], [ %b.n, %emit.next ]
  %out = phi i64 [ 0, %emit.pre ], [ %out.after, %emit.next ]
  %b.done = icmp uge i64 %b, %range
  br i1 %b.done, label %cleanup, label %emit.bucket

emit.bucket:
  %bc.p = getelementptr inbounds nuw i64, ptr %counts, i64 %b
  %bc = load i64, ptr %bc.p, align 8
  %val.w = add nsw i64 %b, %mn.w
  %val = trunc i64 %val.w to i32
  br label %run

run:
  %k = phi i64 [ 0, %emit.bucket ], [ %k.n, %run.body ]
  %run.done = icmp uge i64 %k, %bc
  br i1 %run.done, label %emit.next, label %run.body

run.body:
  %oi = add nuw i64 %out, %k
  %op = getelementptr inbounds nuw i32, ptr %base, i64 %oi
  store i32 %val, ptr %op, align 4
  %k.n = add nuw i64 %k, 1
  br label %run

emit.next:
  %out.after = add nuw i64 %out, %bc
  %b.n = add nuw i64 %b, 1
  br label %emit

cleanup:
  call void @free(ptr nonnull %counts)
  br label %done

done:
  ret i32 0
}

declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)

attributes #1 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
