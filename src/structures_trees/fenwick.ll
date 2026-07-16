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

; Fenwick tree (binary indexed tree) of i64 partial sums. O(log n) point
; update and prefix query, O(1) size, contiguous cache-friendly storage.
;
; DESIGN (from first principles):
;   * ONE allocation: 64B header + a flat i64[n+1] tree. The tree is 1-based
;     internally (slot 0 unused); a user index k maps to tree slot k+1. calloc
;     gives a zeroed tree for free, so create is O(1) touch-free.
;   * The classic low-bit stride `i & -i` is a single neg+and — the loop body
;     carries no data-dependent branch other than the loop-back test, so both
;     backends emit a tight straight-line body. Update walks UP (i += lowbit
;     while i <= n); query walks DOWN (i -= lowbit while i != 0).
;   * Value queries (prefix/range/point) return i64 and have no error channel,
;     so they are defensive: a null handle yields 0, and counts/indices are
;     clamped into [0, n] (umin) instead of trapping. update() is the mutating
;     entry and DOES report errors (1 null, 7 out-of-range index).
;   * All create size math is overflow-checked (uadd/umul.with.overflow → 3).
;   * Layout: { i64 n@0, 56B pad } , tree i64[n+1] @64.
;
; API:
;   ptr universe_ds_fenwick_create(i64 n)                 ; null on OOM(2)/overflow(3)
;   i32 universe_ds_fenwick_update(ptr, i64 index, i64 delta) ; 0 / 1 null / 7 oob
;   i64 universe_ds_fenwick_prefix_sum(ptr, i64 count)    ; sum of [0,count)
;   i64 universe_ds_fenwick_range_sum(ptr, i64 lo, i64 hi); sum of [lo,hi)
;   i64 universe_ds_fenwick_point_get(ptr, i64 index)     ; element at index
;   i64 universe_ds_fenwick_size(ptr)                     ; n
;   void universe_ds_fenwick_destroy(ptr)

declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)

define noalias ptr @universe_ds_fenwick_create(i64 %n) local_unnamed_addr #1 {
entry:
  ; slots = n + 1
  %s = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %n, i64 1)
  %slots = extractvalue { i64, i1 } %s, 0
  %s.o = extractvalue { i64, i1 } %s, 1
  ; treebytes = slots * 8
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %slots, i64 8)
  %treebytes = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  ; total = treebytes + 64
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %treebytes, i64 64)
  %total = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %o1 = or i1 %s.o, %tb.o
  %ovf = or i1 %o1, %tt.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @calloc(i64 %total, i64 1)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %n, ptr %mem, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_ds_fenwick_update(ptr %f, i64 %index, i64 %delta) local_unnamed_addr #0 {
entry:
  %f.null = icmp eq ptr %f, null
  br i1 %f.null, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %n = load i64, ptr %f, align 8
  %oob = icmp uge i64 %index, %n
  br i1 %oob, label %err.idx, label %pre, !prof !0

err.idx:
  ret i32 7

pre:
  %tree = getelementptr inbounds nuw i8, ptr %f, i64 64
  %i0 = add i64 %index, 1
  br label %loop

loop:
  %i = phi i64 [ %i0, %pre ], [ %inext, %loop ]
  %ep = getelementptr inbounds i64, ptr %tree, i64 %i
  %cur = load i64, ptr %ep, align 8
  %new = add i64 %cur, %delta
  store i64 %new, ptr %ep, align 8
  %neg = sub i64 0, %i
  %low = and i64 %i, %neg
  %inext = add i64 %i, %low
  %cont = icmp ule i64 %inext, %n
  br i1 %cont, label %loop, label %done, !prof !1

done:
  ret i32 0
}

; internal: sum of tree slots walking DOWN from `count` (1-based). Assumes the
; caller already clamped count into [0, n]. count==0 -> 0.
define internal i64 @fen_query(ptr noalias readonly captures(none) %tree, i64 %count) #3 {
entry:
  %is0 = icmp eq i64 %count, 0
  br i1 %is0, label %ret0, label %loop

ret0:
  ret i64 0

loop:
  %i = phi i64 [ %count, %entry ], [ %inext, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %snew, %loop ]
  %ep = getelementptr inbounds i64, ptr %tree, i64 %i
  %v = load i64, ptr %ep, align 8
  %snew = add i64 %sum, %v
  %neg = sub i64 0, %i
  %low = and i64 %i, %neg
  %inext = sub i64 %i, %low
  %cont = icmp ne i64 %inext, 0
  br i1 %cont, label %loop, label %fin

fin:
  ret i64 %snew
}

define i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 %count) local_unnamed_addr #2 {
entry:
  %f.null = icmp eq ptr %f, null
  br i1 %f.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %f, align 8
  %c = call i64 @llvm.umin.i64(i64 %count, i64 %n)
  %tree = getelementptr inbounds nuw i8, ptr %f, i64 64
  %r = call i64 @fen_query(ptr %tree, i64 %c)
  ret i64 %r
}

define i64 @universe_ds_fenwick_range_sum(ptr %f, i64 %lo, i64 %hi) local_unnamed_addr #2 {
entry:
  %f.null = icmp eq ptr %f, null
  br i1 %f.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %f, align 8
  %hc = call i64 @llvm.umin.i64(i64 %hi, i64 %n)
  %lc = call i64 @llvm.umin.i64(i64 %lo, i64 %n)
  %empty = icmp uge i64 %lc, %hc
  br i1 %empty, label %ret0, label %compute, !prof !0

compute:
  %tree = getelementptr inbounds nuw i8, ptr %f, i64 64
  %a = call i64 @fen_query(ptr %tree, i64 %hc)
  %b = call i64 @fen_query(ptr %tree, i64 %lc)
  %d = sub i64 %a, %b
  ret i64 %d
}

define i64 @universe_ds_fenwick_point_get(ptr %f, i64 %index) local_unnamed_addr #2 {
entry:
  %f.null = icmp eq ptr %f, null
  br i1 %f.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %f, align 8
  %oob = icmp uge i64 %index, %n
  br i1 %oob, label %ret0, label %compute, !prof !0

compute:
  %tree = getelementptr inbounds nuw i8, ptr %f, i64 64
  %i1 = add i64 %index, 1
  %a = call i64 @fen_query(ptr %tree, i64 %i1)
  %b = call i64 @fen_query(ptr %tree, i64 %index)
  %d = sub i64 %a, %b
  ret i64 %d
}

define i64 @universe_ds_fenwick_size(ptr %f) local_unnamed_addr #2 {
entry:
  %f.null = icmp eq ptr %f, null
  br i1 %f.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %f, align 8
  ret i64 %n
}

define void @universe_ds_fenwick_destroy(ptr %f) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %f, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %f)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
