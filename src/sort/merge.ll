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

; Merge sort. O(N log N), STABLE. One aux allocation.
;
; DESIGN (vs the typical C implementation):
;   * BOTTOM-UP (iterative) with ping-pong buffers: zero recursion, exactly
;     ceil(log2 N) full passes, each a linear sweep — the access pattern the
;     prefetcher loves. No per-merge copy-back: roles swap each pass; one
;     final memcpy only if the sorted result landed in aux.
;   * When one run exhausts, the other side's remainder moves with a single
;     bulk memcpy instead of element-by-element.
;   * Aux buffer allocated once (overflow-checked); OOM -> error 2.

define i32 @universe_sort_merge(ptr %base, i64 %count, i64 %elem_size, ptr %cmp) local_unnamed_addr #1 {
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
  br i1 %trivial, label %done, label %alloc, !prof !0

alloc:
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %count, i64 %elem_size)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  br i1 %bytes.o, label %err.oom, label %do.alloc, !prof !0

do.alloc:
  %aux = call ptr @malloc(i64 %bytes.v)
  %aux.null = icmp eq ptr %aux, null
  br i1 %aux.null, label %err.oom, label %passes, !prof !0

err.oom:
  ret i32 2

passes:                                      ; ping-pong: src -> dst each pass
  %width = phi i64 [ 1, %do.alloc ], [ %width.n, %pass.end ]
  %src = phi ptr [ %base, %do.alloc ], [ %dst, %pass.end ]
  %dst = phi ptr [ %aux, %do.alloc ], [ %src, %pass.end ]
  %width.done = icmp uge i64 %width, %count
  br i1 %width.done, label %settle, label %pairs

pairs:                                       ; merge [lo,mid) with [mid,hi)
  %lo = phi i64 [ 0, %passes ], [ %hi, %pair.end ]
  %lo.done = icmp uge i64 %lo, %count
  br i1 %lo.done, label %pass.end, label %bounds

bounds:
  %mid.raw = add i64 %lo, %width
  %mid = call i64 @llvm.umin.i64(i64 %mid.raw, i64 %count)
  %hi.raw = add i64 %mid.raw, %width
  %hi = call i64 @llvm.umin.i64(i64 %hi.raw, i64 %count)
  br label %merge

merge:                                       ; two-pointer merge into dst[lo..)
  %a = phi i64 [ %lo, %bounds ], [ %a.n, %step ]
  %b = phi i64 [ %mid, %bounds ], [ %b.n, %step ]
  %o = phi i64 [ %lo, %bounds ], [ %o.n, %step ]
  %a.live = icmp ult i64 %a, %mid
  %b.live = icmp ult i64 %b, %hi
  %both = and i1 %a.live, %b.live
  br i1 %both, label %pick, label %tail

pick:
  %a.off = mul i64 %a, %elem_size
  %a.ptr = getelementptr inbounds nuw i8, ptr %src, i64 %a.off
  %b.off = mul i64 %b, %elem_size
  %b.ptr = getelementptr inbounds nuw i8, ptr %src, i64 %b.off
  %c = call i32 %cmp(ptr %b.ptr, ptr %a.ptr) #4
  ; take from A unless B is strictly smaller (stability)
  %b.wins = icmp slt i32 %c, 0
  %take.ptr = select i1 %b.wins, ptr %b.ptr, ptr %a.ptr
  %a.n.inc = add i64 %a, 1
  %b.n.inc = add i64 %b, 1
  %a.n = select i1 %b.wins, i64 %a, i64 %a.n.inc
  %b.n = select i1 %b.wins, i64 %b.n.inc, i64 %b
  %o.off = mul i64 %o, %elem_size
  %o.ptr = getelementptr inbounds nuw i8, ptr %dst, i64 %o.off
  call void @llvm.memcpy.p0.p0.i64(ptr %o.ptr, ptr %take.ptr, i64 %elem_size, i1 false)
  br label %step

step:
  %o.n = add i64 %o, 1
  br label %merge

tail:                                        ; bulk-copy whichever side remains
  %rem.from = select i1 %a.live, i64 %a, i64 %b
  %rem.end = select i1 %a.live, i64 %mid, i64 %hi
  %rem.count = sub i64 %rem.end, %rem.from
  %rem.any = icmp eq i64 %rem.count, 0
  br i1 %rem.any, label %pair.end, label %bulk

bulk:
  %rf.off = mul i64 %rem.from, %elem_size
  %rf.ptr = getelementptr inbounds nuw i8, ptr %src, i64 %rf.off
  %od.off = mul i64 %o, %elem_size
  %od.ptr = getelementptr inbounds nuw i8, ptr %dst, i64 %od.off
  %rem.bytes = mul i64 %rem.count, %elem_size
  call void @llvm.memcpy.p0.p0.i64(ptr %od.ptr, ptr %rf.ptr, i64 %rem.bytes, i1 false)
  br label %pair.end

pair.end:
  br label %pairs

pass.end:
  %width.n = shl i64 %width, 1
  br label %passes

settle:                                      ; result must end in base
  %in.aux = icmp eq ptr %src, %aux
  br i1 %in.aux, label %copy.back, label %cleanup

copy.back:
  call void @llvm.memcpy.p0.p0.i64(ptr %base, ptr %aux, i64 %bytes.v, i1 false)
  br label %cleanup

cleanup:
  call void @free(ptr nonnull %aux)
  br label %done

done:
  ret i32 0
}

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)

attributes #1 = { nounwind }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
