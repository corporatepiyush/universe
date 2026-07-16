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

; Sparse set: O(1) membership / add / remove / clear over a bounded integer
; universe [0, capacity), backed by the classic dense/sparse pair.
;
; DESIGN (functional spec: a set of small integers with O(1) everything and
; perfect-locality iteration):
;   * TWO u32 arrays in ONE allocation:
;       dense[]  — the packed members, in insertion order (contiguous 0..count)
;       sparse[] — universe-indexed: for a present member x, sparse[x] is x's
;                  slot in dense, and dense[sparse[x]] == x. That mutual back-
;                  pointer IS the validity test, which is why sparse[] NEVER
;                  needs initialization: a stale sparse[x] either points past
;                  count (idx >= count) or lands on a slot whose value != x.
;   * Layout: 64-byte header { i64 count@0, i64 capacity@8 } then dense at +64
;     (capacity u32) then sparse at +64+capacity*4 (capacity u32). One malloc,
;     one free; dense is a single cache-line-aligned run the caller iterates
;     with unit stride (get_dense hands back its base + count).
;   * contains(x) is branch-lean and SAFE without initializing sparse: after
;     the cold bounds guard it is idx=sparse[x]; in=(idx<count);
;     clamped=select(in,idx,0); hit=(dense[clamped]==x); present = in & hit.
;     The select keeps the dense load IN BOUNDS (dense[0] always exists once
;     x<capacity proves capacity>=1) so a garbage idx can never read OOB, and
;     ANDing with `in` discards a coincidental dense[0]==x. Two loads, one
;     compare, no branch in the body — the select lowers to csel/cmov.
;   * add is idempotent; remove is swap-with-last in dense (move dense[count-1]
;     into the hole, fix its sparse back-pointer) — O(1), no shifting.
;   * clear() is a SINGLE store of count=0: because membership is defined by
;     the dense/sparse invariant relative to count, dropping count to 0 makes
;     every prior member test false at once. No memset — that is the whole win
;     over a bitset, which must zero its word array.
;   * Values are stored as u32 (universe <= 2^32-1); the i64 API truncates on
;     store and the stored width never exceeds capacity-1.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW, 7 INVALID_INDEX):
;   ptr  universe_ds_sparseset_create(i64 capacity)     ; capacity 0 allowed
;   void universe_ds_sparseset_destroy(ptr s)
;   i32  universe_ds_sparseset_add(ptr s, i64 x)        ; idempotent
;   i32  universe_ds_sparseset_remove(ptr s, i64 x)     ; idempotent, O(1)
;   i32  universe_ds_sparseset_contains(ptr s, i64 x)   ; 1 present, 0 not/null/oob
;   i64  universe_ds_sparseset_size(ptr s)              ; count, 0 if null
;   i64  universe_ds_sparseset_capacity(ptr s)          ; capacity, 0 if null
;   i32  universe_ds_sparseset_clear(ptr s)             ; O(1)
;   ptr  universe_ds_sparseset_get_dense(ptr s, ptr out_len) ; dense base, *out_len=count

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; ---- construction --------------------------------------------------------

define noalias ptr @universe_ds_sparseset_create(i64 %capacity) local_unnamed_addr #1 {
entry:
  ; universe must fit u32 so both member values and dense indices are storable.
  %too.big = icmp ugt i64 %capacity, 4294967295
  br i1 %too.big, label %fail, label %sizes, !prof !0

sizes:
  ; payload = capacity*4 (dense) + capacity*4 (sparse) = capacity*8 bytes.
  %pay = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %capacity, i64 8)
  %pay.v = extractvalue { i64, i1 } %pay, 0
  %pay.o = extractvalue { i64, i1 } %pay, 1
  br i1 %pay.o, label %fail, label %total, !prof !0

total:
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pay.v, i64 64)
  %tot.v = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  br i1 %tot.o, label %fail, label %alloc, !prof !0

alloc:
  %s = call ptr @malloc(i64 %tot.v)
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %fail, label %init, !prof !0

init:
  ; count = 0; capacity = capacity. dense/sparse are LEFT UNINITIALIZED — the
  ; dense/sparse invariant makes stale sparse entries harmless (see DESIGN).
  store i64 0, ptr %s, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  store i64 %capacity, ptr %cap.p, align 8
  ret ptr %s

fail:
  ret ptr null
}

define void @universe_ds_sparseset_destroy(ptr %s) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %s, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %s)
  br label %done

done:
  ret void
}

; ---- mutation ------------------------------------------------------------

define i32 @universe_ds_sparseset_add(ptr %s, i64 %x) local_unnamed_addr #0 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %cap.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %oob = icmp uge i64 %x, %cap
  br i1 %oob, label %err.idx, label %probe, !prof !0

err.idx:
  ret i32 7

probe:
  %count = load i64, ptr %s, align 8
  %dense = getelementptr inbounds nuw i8, ptr %s, i64 64
  %cap4 = shl nuw i64 %cap, 2
  %sparse = getelementptr inbounds nuw i8, ptr %dense, i64 %cap4
  %sp = getelementptr inbounds nuw i32, ptr %sparse, i64 %x
  %idx32 = load i32, ptr %sp, align 4
  %idx = zext i32 %idx32 to i64
  %in = icmp ult i64 %idx, %count
  %clamped = select i1 %in, i64 %idx, i64 0
  %dp = getelementptr inbounds nuw i32, ptr %dense, i64 %clamped
  %dv = load i32, ptr %dp, align 4
  %xt = trunc i64 %x to i32
  %hit = icmp eq i32 %dv, %xt
  %present = and i1 %in, %hit
  br i1 %present, label %already, label %insert

insert:
  ; dense[count] = x; sparse[x] = count; count++.
  %dp.new = getelementptr inbounds nuw i32, ptr %dense, i64 %count
  store i32 %xt, ptr %dp.new, align 4
  %count32 = trunc i64 %count to i32
  store i32 %count32, ptr %sp, align 4
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %s, align 8
  ret i32 0

already:
  ret i32 0
}

define i32 @universe_ds_sparseset_remove(ptr %s, i64 %x) local_unnamed_addr #0 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %cap.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %oob = icmp uge i64 %x, %cap
  br i1 %oob, label %err.idx, label %probe, !prof !0

err.idx:
  ret i32 7

probe:
  %count = load i64, ptr %s, align 8
  %dense = getelementptr inbounds nuw i8, ptr %s, i64 64
  %cap4 = shl nuw i64 %cap, 2
  %sparse = getelementptr inbounds nuw i8, ptr %dense, i64 %cap4
  %sp = getelementptr inbounds nuw i32, ptr %sparse, i64 %x
  %idx32 = load i32, ptr %sp, align 4
  %idx = zext i32 %idx32 to i64
  %in = icmp ult i64 %idx, %count
  %clamped = select i1 %in, i64 %idx, i64 0
  %dp = getelementptr inbounds nuw i32, ptr %dense, i64 %clamped
  %dv = load i32, ptr %dp, align 4
  %xt = trunc i64 %x to i32
  %hit = icmp eq i32 %dv, %xt
  %present = and i1 %in, %hit
  br i1 %present, label %do.remove, label %absent

do.remove:
  ; swap-with-last: move dense[count-1] into the removed slot (idx), then fix
  ; that moved member's sparse back-pointer; count--.  (present => count>=1)
  %last = sub nuw i64 %count, 1
  %last.p = getelementptr inbounds nuw i32, ptr %dense, i64 %last
  %lastval = load i32, ptr %last.p, align 4
  %hole.p = getelementptr inbounds nuw i32, ptr %dense, i64 %idx
  store i32 %lastval, ptr %hole.p, align 4
  %lastval64 = zext i32 %lastval to i64
  %sp.moved = getelementptr inbounds nuw i32, ptr %sparse, i64 %lastval64
  store i32 %idx32, ptr %sp.moved, align 4
  store i64 %last, ptr %s, align 8
  ret i32 0

absent:
  ret i32 0
}

define i32 @universe_ds_sparseset_clear(ptr %s) local_unnamed_addr #0 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  ; O(1) clear: a single store. Membership is relative to count, so count=0
  ; empties the set without touching dense or sparse.
  store i64 0, ptr %s, align 8
  ret i32 0
}

; ---- queries -------------------------------------------------------------

define i32 @universe_ds_sparseset_contains(ptr %s, i64 %x) local_unnamed_addr #2 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %absent, label %check, !prof !0

absent:
  ret i32 0

check:
  %cap.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %oob = icmp uge i64 %x, %cap
  br i1 %oob, label %absent, label %do, !prof !0

do:
  %count = load i64, ptr %s, align 8
  %dense = getelementptr inbounds nuw i8, ptr %s, i64 64
  %cap4 = shl nuw i64 %cap, 2
  %sparse = getelementptr inbounds nuw i8, ptr %dense, i64 %cap4
  %sp = getelementptr inbounds nuw i32, ptr %sparse, i64 %x
  %idx32 = load i32, ptr %sp, align 4
  %idx = zext i32 %idx32 to i64
  %in = icmp ult i64 %idx, %count
  %clamped = select i1 %in, i64 %idx, i64 0
  %dp = getelementptr inbounds nuw i32, ptr %dense, i64 %clamped
  %dv = load i32, ptr %dp, align 4
  %xt = trunc i64 %x to i32
  %hit = icmp eq i32 %dv, %xt
  %present = and i1 %in, %hit
  %r = zext i1 %present to i32
  ret i32 %r
}

define i64 @universe_ds_sparseset_size(ptr %s) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %s, null
  br i1 %is.null, label %z, label %do, !prof !0

z:
  ret i64 0

do:
  %count = load i64, ptr %s, align 8
  ret i64 %count
}

define i64 @universe_ds_sparseset_capacity(ptr %s) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %s, null
  br i1 %is.null, label %z, label %do, !prof !0

z:
  ret i64 0

do:
  %cap.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

; Hand back the dense array base (zero-copy iteration over [0,count)) and, when
; out_len is non-null, the current member count.
define ptr @universe_ds_sparseset_get_dense(ptr %s, ptr %out_len) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %s, null
  br i1 %is.null, label %null, label %do, !prof !0

null:
  %want.len0 = icmp ne ptr %out_len, null
  br i1 %want.len0, label %store.zero, label %ret.null

store.zero:
  store i64 0, ptr %out_len, align 8
  br label %ret.null

ret.null:
  ret ptr null

do:
  %want.len = icmp ne ptr %out_len, null
  br i1 %want.len, label %store.len, label %ret.dense

store.len:
  %count = load i64, ptr %s, align 8
  store i64 %count, ptr %out_len, align 8
  br label %ret.dense

ret.dense:
  %dense = getelementptr inbounds nuw i8, ptr %s, i64 64
  ret ptr %dense
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
