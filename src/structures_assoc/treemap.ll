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

; Ordered map (i64 key -> i64 value) — the ARRAY-BACKED flavor. Keys are kept
; in a single flat, contiguous, ascending run so every query is a branch-lean
; binary search and every ORDERED query (floor/ceiling/higher/lower/range) is
; pure index arithmetic on that run.
;
; WHEN TO CHOOSE THIS VARIANT:
;   * You need an ORDERED map: nearest-key lookups (floor/ceiling/higher/lower),
;     min/max, and ordered range scans — not just point get/put.
;   * The workload is search-heavy / iteration-heavy and mutation is moderate.
;     get/floor/ceiling/range all run at binary-search speed over ONE cache-
;     dense array (keys packed with no pointers, no node headers), so a scan of
;     an in-order range is a flat sequential walk — the friendliest possible
;     access pattern for the prefetcher. A pointer-linked balanced BST would
;     cost a cache miss per node on every descent AND a successor-walk per
;     range step; the sorted run pays neither. put/delete are O(n) memmoves,
;     so prefer a hash map when you need no ordering, or this when ordered
;     queries dominate and inserts are not the bottleneck.
;
; DESIGN (first principles; no external layout copied):
;   * TOTAL ORDER over keys is UNSIGNED i64 (icmp ult). This gives a single
;     total order across every 64-bit pattern and — critically — makes the
;     sharded sibling's top-bit range-partition MONOTONE (shard index rises
;     with the key), so global order is preserved across shards for free. A
;     caller wanting signed order biases keys by +2^63 on the way in/out.
;   * ONE allocation for storage: a data block holding keys[cap] i64 IMMEDIATELY
;     followed by values[cap] i64 (SoA). Search touches only the keys half, so
;     it streams 2x as many keys per cache line as an interleaved {k,v} layout.
;     A STABLE 64 B header holds {data ptr, count, cap} so growth may move the
;     block without invalidating the caller's handle.
;   * Growth: when count == cap, allocate a 2x block, memcpy the live keys and
;     (to their new offset) the live values, free the old block. Overflow-check
;     cap*16 with umul.with.overflow.
;   * put(k): lb = lower_bound(k) = first index with keys[i] >= k. If keys[lb]
;     == k overwrite in place; else grow-if-full then ONE memmove opens a slot
;     at lb and the pair is written — the array stays sorted by construction.
;   * delete(k): locate exact index; ONE memmove closes the gap.
;   * ORDERED queries via two boundary searches (all O(log n), zero allocation):
;       lower_bound(k) = first i, keys[i] >= k ; upper_bound(k) = first i, > k
;       ceiling(k) = keys[lb]                 (smallest >= k)
;       higher(k)  = keys[ub]                 (smallest >  k)
;       floor(k)   = keys[ub-1]               (largest  <= k)
;       lower(k)   = keys[lb-1]               (largest  <  k)
;       min/first  = keys[0] ; max/last = keys[count-1]
;       range[lo,hi] = keys[lower_bound(lo) .. upper_bound(hi))  (inclusive)
;   * Binary search reads keys[mid] ONLY for mid<count (lo<hi guards it), so no
;     load ever runs past the array; empty maps take the zero-trip fast exit.
;
; Header (64 B, one cache line):  data@0(ptr)  count@8  cap@16   [pad -> 64]
; Data block: keys[cap]*i64 | values[cap]*i64      (keys=data, values=data+cap*8)
;
; API (error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 5 NOT_FOUND):
;   ptr  universe_ds_treemap_create(i64 initial_cap)
;   void universe_ds_treemap_destroy(ptr m)
;   i32  universe_ds_treemap_put(ptr m, i64 key, i64 val)      ; insert/overwrite
;   i32  universe_ds_treemap_get(ptr m, i64 key, ptr out_v)
;   i32  universe_ds_treemap_contains(ptr m, i64 key)
;   i32  universe_ds_treemap_delete(ptr m, i64 key)
;   i64  universe_ds_treemap_size(ptr m)
;   i32  universe_ds_treemap_floor(ptr m, i64 key, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_ceiling(ptr m, i64 key, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_higher(ptr m, i64 key, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_lower(ptr m, i64 key, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_min(ptr m, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_max(ptr m, ptr out_k, ptr out_v)
;   i32  universe_ds_treemap_first(ptr m, ptr out_k, ptr out_v)  ; = min
;   i32  universe_ds_treemap_last(ptr m, ptr out_k, ptr out_v)   ; = max
;   i64  universe_ds_treemap_range(ptr m, i64 lo, i64 hi, ptr out_k, ptr out_v,
;                                  i64 out_cap)  ; inclusive [lo,hi]; writes up
;                                  ; to out_cap pairs (out ptrs may be null to
;                                  ; count only); returns total in-range count
;   void universe_ds_treemap_foreach(ptr m, ptr fn, ptr ctx)  ; ascending;
;                                  ; fn = void(ptr ctx, i64 key, i64 val)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

; ===========================================================================
; internal helpers (alwaysinline => zero cost inside the exported ops)
; ===========================================================================

; lower_bound: first index in [0,count] whose key >= %k (unsigned). Reads
; keys[mid] only while lo<hi, so mid<count always — never past the array.
define internal i64 @tm_lb(ptr %keys, i64 %count, i64 %k) #0 {
entry:
  br label %loop

loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %count, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done

step:
  %sum = add nuw i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %less = icmp ult i64 %kv, %k
  %mid1 = add nuw i64 %mid, 1
  %lo.n = select i1 %less, i64 %mid1, i64 %lo
  %hi.n = select i1 %less, i64 %hi, i64 %mid
  br label %loop

done:
  ret i64 %lo
}

; upper_bound: first index in [0,count] whose key > %k (unsigned).
define internal i64 @tm_ub(ptr %keys, i64 %count, i64 %k) #0 {
entry:
  br label %loop

loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %count, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done

step:
  %sum = add nuw i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %le = icmp ule i64 %kv, %k
  %mid1 = add nuw i64 %mid, 1
  %lo.n = select i1 %le, i64 %mid1, i64 %lo
  %hi.n = select i1 %le, i64 %hi, i64 %mid
  br label %loop

done:
  ret i64 %lo
}

; values base = data + cap*8
define internal ptr @tm_vals(ptr %m) #1 {
entry:
  %data = load ptr, ptr %m, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %cap8 = shl nuw i64 %cap, 3
  %vals = getelementptr inbounds nuw i8, ptr %data, i64 %cap8
  ret ptr %vals
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_treemap_create(i64 %initial_cap) local_unnamed_addr #2 {
entry:
  %cap = call i64 @llvm.umax.i64(i64 %initial_cap, i64 8)
  ; bytes = cap * 16
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 16)
  %bytes = extractvalue { i64, i1 } %tb, 0
  %ovf = extractvalue { i64, i1 } %tb, 1
  br i1 %ovf, label %fail, label %alloc.hdr, !prof !0

alloc.hdr:
  %hdr = call ptr @malloc(i64 64)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %alloc.data, !prof !0

alloc.data:
  %data = call ptr @malloc(i64 %bytes)
  %data.null = icmp eq ptr %data, null
  br i1 %data.null, label %free.hdr, label %init, !prof !0

free.hdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %data, ptr %hdr, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 %cap, ptr %cap.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_treemap_destroy(ptr %m) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %m, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %m)
  br label %done

done:
  ret void
}

; Ensure cap > count (room for one insert). Reallocates the SoA block 2x and
; copies both halves. Returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW. Caller reloads.
define internal i32 @tm_grow(ptr %m) #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %oldcap = load i64, ptr %cap.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %newcap = shl nuw i64 %oldcap, 1
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %newcap, i64 16)
  %bytes = extractvalue { i64, i1 } %tb, 0
  %ovf = extractvalue { i64, i1 } %tb, 1
  br i1 %ovf, label %err.ovf, label %alloc, !prof !0

alloc:
  %newdata = call ptr @malloc(i64 %bytes)
  %nd.null = icmp eq ptr %newdata, null
  br i1 %nd.null, label %err.oom, label %copy, !prof !0

copy:
  %olddata = load ptr, ptr %m, align 8
  %live8 = shl nuw i64 %count, 3
  ; copy keys[0..count)
  call void @llvm.memcpy.p0.p0.i64(ptr %newdata, ptr %olddata, i64 %live8, i1 false)
  ; copy values[0..count) : old vals at olddata+oldcap*8, new vals at newdata+newcap*8
  %oldcap8 = shl nuw i64 %oldcap, 3
  %oldvals = getelementptr inbounds nuw i8, ptr %olddata, i64 %oldcap8
  %newcap8 = shl nuw i64 %newcap, 3
  %newvals = getelementptr inbounds nuw i8, ptr %newdata, i64 %newcap8
  call void @llvm.memcpy.p0.p0.i64(ptr %newvals, ptr %oldvals, i64 %live8, i1 false)
  call void @free(ptr %olddata)
  store ptr %newdata, ptr %m, align 8
  store i64 %newcap, ptr %cap.p, align 8
  ret i32 0

err.oom:
  ret i32 2

err.ovf:
  ret i32 3
}

; ===========================================================================
; put — insert or overwrite
; ===========================================================================
define i32 @universe_ds_treemap_put(ptr %m, i64 %key, i64 %val) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %in.range = icmp ult i64 %lb, %count
  br i1 %in.range, label %chk.dup, label %insert

chk.dup:
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %lb
  %kv = load i64, ptr %kp, align 8
  %dup = icmp eq i64 %kv, %key
  br i1 %dup, label %overwrite, label %insert

overwrite:
  %vals.o = call ptr @tm_vals(ptr nonnull %m)
  %vp.o = getelementptr inbounds nuw i64, ptr %vals.o, i64 %lb
  store i64 %val, ptr %vp.o, align 8
  ret i32 0

insert:
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp eq i64 %count, %cap
  br i1 %full, label %grow, label %place, !prof !0

grow:
  %rc = call i32 @tm_grow(ptr nonnull %m)
  %rc.ok = icmp eq i32 %rc, 0
  br i1 %rc.ok, label %place, label %err.grow, !prof !1

err.grow:
  ret i32 %rc

place:
  ; reload data/vals (grow may have moved the block)
  %data2 = load ptr, ptr %m, align 8
  %vals2 = call ptr @tm_vals(ptr nonnull %m)
  %tail = sub i64 %count, %lb
  %tail8 = shl nuw i64 %tail, 3
  ; shift keys[lb..count) -> keys[lb+1..count+1)
  %ksrc = getelementptr inbounds nuw i64, ptr %data2, i64 %lb
  %lb1 = add nuw i64 %lb, 1
  %kdst = getelementptr inbounds nuw i64, ptr %data2, i64 %lb1
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %tail8, i1 false)
  ; shift values similarly
  %vsrc = getelementptr inbounds nuw i64, ptr %vals2, i64 %lb
  %vdst = getelementptr inbounds nuw i64, ptr %vals2, i64 %lb1
  call void @llvm.memmove.p0.p0.i64(ptr %vdst, ptr %vsrc, i64 %tail8, i1 false)
  ; write the new pair
  store i64 %key, ptr %ksrc, align 8
  store i64 %val, ptr %vsrc, align 8
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0
}

; ===========================================================================
; get / contains
; ===========================================================================
define i32 @universe_ds_treemap_get(ptr %m, i64 %key, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  %o.null = icmp eq ptr %out_v, null
  %bad = or i1 %m.null, %o.null
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %in.range = icmp ult i64 %lb, %count
  br i1 %in.range, label %chk, label %miss

chk:
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %lb
  %kv = load i64, ptr %kp, align 8
  %hit = icmp eq i64 %kv, %key
  br i1 %hit, label %found, label %miss

found:
  %vals = call ptr @tm_vals(ptr nonnull %m)
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %lb
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %out_v, align 8
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_contains(ptr %m, i64 %key) local_unnamed_addr #3 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %in.range = icmp ult i64 %lb, %count
  br i1 %in.range, label %chk, label %miss

chk:
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %lb
  %kv = load i64, ptr %kp, align 8
  %hit = icmp eq i64 %kv, %key
  %r = select i1 %hit, i32 0, i32 5
  ret i32 %r

miss:
  ret i32 5
}

; ===========================================================================
; delete
; ===========================================================================
define i32 @universe_ds_treemap_delete(ptr %m, i64 %key) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %in.range = icmp ult i64 %lb, %count
  br i1 %in.range, label %chk, label %miss

chk:
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %lb
  %kv = load i64, ptr %kp, align 8
  %hit = icmp eq i64 %kv, %key
  br i1 %hit, label %erase, label %miss

erase:
  %vals = call ptr @tm_vals(ptr nonnull %m)
  %lb1 = add nuw i64 %lb, 1
  %tail = sub i64 %count, %lb1
  %tail8 = shl nuw i64 %tail, 3
  ; close gap: keys[lb+1..count) -> keys[lb..count-1)
  %ksrc = getelementptr inbounds nuw i64, ptr %data, i64 %lb1
  %kdst = getelementptr inbounds nuw i64, ptr %data, i64 %lb
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %tail8, i1 false)
  %vsrc = getelementptr inbounds nuw i64, ptr %vals, i64 %lb1
  %vdst = getelementptr inbounds nuw i64, ptr %vals, i64 %lb
  call void @llvm.memmove.p0.p0.i64(ptr %vdst, ptr %vsrc, i64 %tail8, i1 false)
  %count.n = sub i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret i32 0

miss:
  ret i32 5
}

define i64 @universe_ds_treemap_size(ptr %m) local_unnamed_addr #3 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

; ===========================================================================
; ordered nearest-key queries.  Each writes out_k/out_v (when non-null) and
; returns 0 OK / 5 NOT_FOUND / 1 NULL_PTR.  Shared tail @tm_emit stores idx.
; ===========================================================================

; store keys[idx]/values[idx] into out_k/out_v when non-null, return 0.
define internal i32 @tm_emit(ptr %m, i64 %idx, ptr %out_k, ptr %out_v) #2 {
entry:
  %data = load ptr, ptr %m, align 8
  %ko.null = icmp eq ptr %out_k, null
  br i1 %ko.null, label %do.v, label %wk

wk:
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %idx
  %kv = load i64, ptr %kp, align 8
  store i64 %kv, ptr %out_k, align 8
  br label %do.v

do.v:
  %vo.null = icmp eq ptr %out_v, null
  br i1 %vo.null, label %done, label %wv

wv:
  %vals = call ptr @tm_vals(ptr nonnull %m)
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %idx
  %vv = load i64, ptr %vp, align 8
  store i64 %vv, ptr %out_v, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_ds_treemap_ceiling(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %ok = icmp ult i64 %lb, %count
  br i1 %ok, label %emit, label %miss

emit:
  %r = call i32 @tm_emit(ptr nonnull %m, i64 %lb, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_higher(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %ub = call i64 @tm_ub(ptr %data, i64 %count, i64 %key)
  %ok = icmp ult i64 %ub, %count
  br i1 %ok, label %emit, label %miss

emit:
  %r = call i32 @tm_emit(ptr nonnull %m, i64 %ub, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_floor(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %ub = call i64 @tm_ub(ptr %data, i64 %count, i64 %key)
  %none = icmp eq i64 %ub, 0
  br i1 %none, label %miss, label %emit

emit:
  %idx = sub i64 %ub, 1
  %r = call i32 @tm_emit(ptr nonnull %m, i64 %idx, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_lower(ptr %m, i64 %key, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %lb = call i64 @tm_lb(ptr %data, i64 %count, i64 %key)
  %none = icmp eq i64 %lb, 0
  br i1 %none, label %miss, label %emit

emit:
  %idx = sub i64 %lb, 1
  %r = call i32 @tm_emit(ptr nonnull %m, i64 %idx, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_min(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %miss, label %emit

emit:
  %r = call i32 @tm_emit(ptr nonnull %m, i64 0, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_max(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %miss, label %emit

emit:
  %idx = sub i64 %count, 1
  %r = call i32 @tm_emit(ptr nonnull %m, i64 %idx, ptr %out_k, ptr %out_v)
  ret i32 0

miss:
  ret i32 5
}

define i32 @universe_ds_treemap_first(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %r = call i32 @universe_ds_treemap_min(ptr %m, ptr %out_k, ptr %out_v)
  ret i32 %r
}

define i32 @universe_ds_treemap_last(ptr %m, ptr %out_k, ptr %out_v) local_unnamed_addr #2 {
entry:
  %r = call i32 @universe_ds_treemap_max(ptr %m, ptr %out_k, ptr %out_v)
  ret i32 %r
}

; ===========================================================================
; range [lo,hi] inclusive — writes up to out_cap pairs; returns total count.
; ===========================================================================
define i64 @universe_ds_treemap_range(ptr %m, i64 %lo, i64 %hi, ptr %out_k, ptr %out_v, i64 %out_cap) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %z, label %chkord, !prof !0

z:
  ret i64 0

chkord:
  ; empty range if lo > hi (unsigned)
  %bad = icmp ugt i64 %lo, %hi
  br i1 %bad, label %z, label %setup

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %start = call i64 @tm_lb(ptr %data, i64 %count, i64 %lo)   ; first >= lo
  %end = call i64 @tm_ub(ptr %data, i64 %count, i64 %hi)     ; first > hi
  %total = sub i64 %end, %start
  %have.out = icmp ne ptr %out_k, null
  %pos.total = icmp ugt i64 %total, 0
  %do.write = and i1 %have.out, %pos.total
  br i1 %do.write, label %write.pre, label %done

write.pre:
  %vals = call ptr @tm_vals(ptr nonnull %m)
  %v.null = icmp eq ptr %out_v, null
  ; number to write = min(total, out_cap)
  %fits = icmp ule i64 %total, %out_cap
  %nwrite = select i1 %fits, i64 %total, i64 %out_cap
  %nz = icmp eq i64 %nwrite, 0
  br i1 %nz, label %done, label %wloop

wloop:
  %i = phi i64 [ 0, %write.pre ], [ %i.n, %wcont ]
  %src.i = add nuw i64 %start, %i
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %src.i
  %kv = load i64, ptr %kp, align 8
  %okp = getelementptr inbounds nuw i64, ptr %out_k, i64 %i
  store i64 %kv, ptr %okp, align 8
  br i1 %v.null, label %wcont, label %wv

wv:
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %src.i
  %vv = load i64, ptr %vp, align 8
  %ovp = getelementptr inbounds nuw i64, ptr %out_v, i64 %i
  store i64 %vv, ptr %ovp, align 8
  br label %wcont

wcont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nwrite
  br i1 %more, label %wloop, label %done

done:
  ret i64 %total
}

; ===========================================================================
; foreach — ascending order; fn = void(ptr ctx, i64 key, i64 val)
; ===========================================================================
define void @universe_ds_treemap_foreach(ptr %m, ptr %fn, ptr %ctx) local_unnamed_addr #4 {
entry:
  %m.null = icmp eq ptr %m, null
  %fn.null = icmp eq ptr %fn, null
  %bad = or i1 %m.null, %fn.null
  br i1 %bad, label %done, label %setup, !prof !0

setup:
  %data = load ptr, ptr %m, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %done, label %loop.pre

loop.pre:
  %vals = call ptr @tm_vals(ptr nonnull %m)
  br label %loop

loop:
  %i = phi i64 [ 0, %loop.pre ], [ %i.n, %loop ]
  %kp = getelementptr inbounds nuw i64, ptr %data, i64 %i
  %kv = load i64, ptr %kp, align 8
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %vv = load i64, ptr %vp, align 8
  call void %fn(ptr %ctx, i64 %kv, i64 %vv)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #2 = { nounwind willreturn }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
