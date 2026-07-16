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

; Segment tree for range-sum with point-SET update over i64. Iterative,
; bottom-up, no recursion — every op is a tight while loop that climbs or
; walks the flat tree by >>1.
;
; DESIGN (from first principles):
;   * ONE allocation: 64B header + a flat i64 tree of EXACTLY 2*n slots. The
;     leaves live at tree[n .. 2n), internal node i covers the union of its
;     children tree[2i], tree[2i+1]. This is the arbitrary-n iterative layout:
;     it needs NO rounding up to a power of two, so it uses half the memory of
;     a 2*npow2 tree and still keeps the same branch-free >>1 climb. For a
;     commutative/associative monoid (integer sum) the inward lo/hi walk is
;     order-independent, so correctness holds for every n, not just powers of 2.
;   * point_update SETS (overwrites) the leaf, then re-sums ancestors climbing
;     i>>=1 until the root — O(log n), no recursion, no stack.
;   * range_sum([lo,hi)) seeds l=lo+n, r=hi+n and walks inward: whenever l is a
;     right child it contributes and steps in (l++), whenever r is a right
;     child r steps in (r--) and contributes; both shift right each round until
;     l>=r. The two boundary loads sit behind predictable branches so we never
;     read tree[2n] (one past the array) when hi==n.
;   * create() zeros via calloc (all leaves 0). create_from() copies the input
;     leaves with ONE memcpy then builds parents in a single downward sweep.
;   * Value queries return i64 with no error channel: null -> 0, and lo/hi are
;     clamped into [0,n]. point_update is the mutating entry and reports errors
;     (1 null, 7 out-of-range index).
;   * Layout: { i64 n@0, 56B pad }, tree i64[2n] @64.
;
; API:
;   ptr universe_ds_segtree_create(i64 n)                    ; zeroed; null on 2/3
;   ptr universe_ds_segtree_create_from(ptr vals, i64 n)     ; null on 1/2/3
;   i32 universe_ds_segtree_point_update(ptr, i64 index, i64 value) ; SET; 0/1/7
;   i64 universe_ds_segtree_range_sum(ptr, i64 lo, i64 hi)   ; sum of [lo,hi)
;   i64 universe_ds_segtree_size(ptr)                        ; n
;   void universe_ds_segtree_destroy(ptr)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)

; internal: overflow-checked total byte size for n leaves (64 + 2n*8 = 64 + n*16).
; returns { total, overflow }
define internal { i64, i1 } @seg_bytes(i64 %n) #4 {
entry:
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 16)
  %treebytes = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %treebytes, i64 64)
  %total = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %ovf = or i1 %tb.o, %tt.o
  %r0 = insertvalue { i64, i1 } poison, i64 %total, 0
  %r1 = insertvalue { i64, i1 } %r0, i1 %ovf, 1
  ret { i64, i1 } %r1
}

define noalias ptr @universe_ds_segtree_create(i64 %n) local_unnamed_addr #1 {
entry:
  %sz = call { i64, i1 } @seg_bytes(i64 %n)
  %total = extractvalue { i64, i1 } %sz, 0
  %ovf = extractvalue { i64, i1 } %sz, 1
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

define noalias ptr @universe_ds_segtree_create_from(ptr %vals, i64 %n) local_unnamed_addr #1 {
entry:
  %v.null = icmp eq ptr %vals, null
  br i1 %v.null, label %fail, label %sizes, !prof !0

sizes:
  %sz = call { i64, i1 } @seg_bytes(i64 %n)
  %total = extractvalue { i64, i1 } %sz, 0
  %ovf = extractvalue { i64, i1 } %sz, 1
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %n, ptr %mem, align 8
  %tree = getelementptr inbounds nuw i8, ptr %mem, i64 64
  ; copy leaves: tree[n .. 2n) <- vals[0 .. n)
  %leaves = getelementptr inbounds i64, ptr %tree, i64 %n
  %nbytes = shl i64 %n, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %leaves, ptr %vals, i64 %nbytes, i1 false)
  ; build parents: for i = n-1 downto 1: tree[i] = tree[2i] + tree[2i+1]
  %n.small = icmp ule i64 %n, 1
  br i1 %n.small, label %ret, label %build.pre

build.pre:
  %start = sub i64 %n, 1
  br label %build

build:
  %i = phi i64 [ %start, %build.pre ], [ %inext, %build ]
  %i2 = shl i64 %i, 1
  %i2p1 = or disjoint i64 %i2, 1
  %lc = getelementptr inbounds i64, ptr %tree, i64 %i2
  %rc = getelementptr inbounds i64, ptr %tree, i64 %i2p1
  %lv = load i64, ptr %lc, align 8
  %rv = load i64, ptr %rc, align 8
  %sum = add i64 %lv, %rv
  %ti = getelementptr inbounds i64, ptr %tree, i64 %i
  store i64 %sum, ptr %ti, align 8
  %inext = sub i64 %i, 1
  %cont = icmp ne i64 %inext, 0
  br i1 %cont, label %build, label %ret, !prof !1

ret:
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_ds_segtree_point_update(ptr %s, i64 %index, i64 %value) local_unnamed_addr #0 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %n = load i64, ptr %s, align 8
  %oob = icmp uge i64 %index, %n
  br i1 %oob, label %err.idx, label %set, !prof !0

err.idx:
  ret i32 7

set:
  %tree = getelementptr inbounds nuw i8, ptr %s, i64 64
  %pos = add i64 %index, %n
  %lp = getelementptr inbounds i64, ptr %tree, i64 %pos
  store i64 %value, ptr %lp, align 8
  %p0 = lshr i64 %pos, 1
  %z = icmp eq i64 %p0, 0
  br i1 %z, label %done, label %climb

climb:
  %i = phi i64 [ %p0, %set ], [ %inext, %climb ]
  %i2 = shl i64 %i, 1
  %i2p1 = or disjoint i64 %i2, 1
  %lc = getelementptr inbounds i64, ptr %tree, i64 %i2
  %rc = getelementptr inbounds i64, ptr %tree, i64 %i2p1
  %lv = load i64, ptr %lc, align 8
  %rv = load i64, ptr %rc, align 8
  %sum = add i64 %lv, %rv
  %ti = getelementptr inbounds i64, ptr %tree, i64 %i
  store i64 %sum, ptr %ti, align 8
  %inext = lshr i64 %i, 1
  %cont = icmp ne i64 %inext, 0
  br i1 %cont, label %climb, label %done, !prof !1

done:
  ret i32 0
}

define i64 @universe_ds_segtree_range_sum(ptr %s, i64 %lo, i64 %hi) local_unnamed_addr #2 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %s, align 8
  %hc = call i64 @llvm.umin.i64(i64 %hi, i64 %n)
  %lc = call i64 @llvm.umin.i64(i64 %lo, i64 %n)
  %empty = icmp uge i64 %lc, %hc
  br i1 %empty, label %ret0, label %walk.pre, !prof !0

walk.pre:
  %tree = getelementptr inbounds nuw i8, ptr %s, i64 64
  %l0 = add i64 %lc, %n
  %r0 = add i64 %hc, %n
  br label %loop

loop:
  %l = phi i64 [ %l0, %walk.pre ], [ %lnext, %cont ]
  %r = phi i64 [ %r0, %walk.pre ], [ %rnext, %cont ]
  %res = phi i64 [ 0, %walk.pre ], [ %res3, %cont ]
  %go2 = icmp ult i64 %l, %r
  br i1 %go2, label %lside, label %fin

lside:
  %lodd = and i64 %l, 1
  %ltake = icmp ne i64 %lodd, 0
  br i1 %ltake, label %ladd, label %rside

ladd:
  %lp = getelementptr inbounds i64, ptr %tree, i64 %l
  %lv = load i64, ptr %lp, align 8
  %res.l = add i64 %res, %lv
  %l.inc = add i64 %l, 1
  br label %rside

rside:
  %res.a = phi i64 [ %res.l, %ladd ], [ %res, %lside ]
  %l.a = phi i64 [ %l.inc, %ladd ], [ %l, %lside ]
  %rodd = and i64 %r, 1
  %rtake = icmp ne i64 %rodd, 0
  br i1 %rtake, label %radd, label %cont

radd:
  %r.dec = sub i64 %r, 1
  %rp = getelementptr inbounds i64, ptr %tree, i64 %r.dec
  %rv = load i64, ptr %rp, align 8
  %res.b = add i64 %res.a, %rv
  br label %cont

cont:
  %res3 = phi i64 [ %res.b, %radd ], [ %res.a, %rside ]
  %r.b = phi i64 [ %r.dec, %radd ], [ %r, %rside ]
  %lnext = lshr i64 %l.a, 1
  %rnext = lshr i64 %r.b, 1
  br label %loop

fin:
  ret i64 %res
}

define i64 @universe_ds_segtree_size(ptr %s) local_unnamed_addr #2 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %n = load i64, ptr %s, align 8
  ret i64 %n
}

define void @universe_ds_segtree_destroy(ptr %s) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %s, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %s)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
