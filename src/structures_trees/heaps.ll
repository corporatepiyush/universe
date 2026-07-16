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

; universe_ds heap family — three priority-queue variants over i64 keys, each a
; better fit for a distinct workload than the binary heap. Single thread.
;
; ============================================================================
; DESIGN (choose the class by the workload; see docs/performance-principles.md)
; ----------------------------------------------------------------------------
; (1) d-ary heap  (universe_ds_dheap_*)  — GENERAL priority queue, D=4 default.
;     Implicit heap in a flat i64 array (like the binary heap) but with D
;     children per node: children of i are D*i+1 .. D*i+D, parent is (i-1)/D.
;     A larger D makes the tree SHALLOWER (log_D n) so pop's sift-down does
;     fewer levels; the D children of a node sit CONTIGUOUSLY so one sift step
;     scans a single cache line (D=4 => 32 B, D=8 => 64 B). Prefer over the
;     binary heap for large heaps / Dijkstra with unbounded weights. Choose D=4
;     (default) for balanced push/pop, D=8+ when pops dominate. MIN-heap over
;     SIGNED i64 (max-heap: negate keys in and out, avoiding INT64_MIN).
;     Same 32-byte detachable handle + separate payload malloc as the binary
;     heap so growth (realloc) never moves the caller's handle.
;
; (2) radix / monotone heap  (universe_ds_radixheap_*)  — Dijkstra with BOUNDED
;     INTEGER weights. PRECONDITION: keys are non-negative and successive pops
;     are NON-DECREASING (monotone), i.e. no key smaller than the last extracted
;     minimum is ever inserted (a push that violates this returns INVALID_ARG).
;     Under that contract it beats a comparison heap: O(1) amortized ops, no
;     log factor. Buckets 0..64: bucket 0 holds keys equal to the current
;     boundary `last`; bucket i (i>=1) holds keys whose highest bit DIFFERING
;     from `last` is bit (i-1) — i.e. b = 64 - clz(key XOR last), or 0 when
;     equal. pop empties bucket 0 first; when it is empty it finds the smallest
;     non-empty bucket, sets `last` to that bucket's min, and REDISTRIBUTES that
;     bucket's items into lower buckets under the new boundary — each item moves
;     to a strictly lower bucket, giving the amortized O(1). Items are
;     index-linked nodes (key,val) from a chunked pool.
;
; (3) pairing heap  (universe_ds_pairing_*)  — DENSE-graph Dijkstra / any PQ
;     needing DECREASE-KEY. A multiway tree of index-linked nodes; the min is
;     the root. insert/meld are O(1) (one link); decrease-key is O(1) amortized
;     (cut the node, meld its subtree back at the root); delete-min combines the
;     root's children with the classic TWO-PASS pairing (left-to-right pair,
;     then right-to-left fold), which is what makes pairing heaps fast in
;     practice. Nodes carry child / next-sibling / prev (parent-or-left-sibling)
;     i32 links so a cut is O(1). push returns the node index as a stable HANDLE
;     for a later decrease-key.
;
; ----------------------------------------------------------------------------
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY, 8 INVALID_ARG):
;   ; --- d-ary heap ---
;   ptr universe_ds_dheap_create(i64 initial_cap, i64 d)   ; d<2 -> 4
;   void universe_ds_dheap_destroy(ptr)
;   i32 universe_ds_dheap_push(ptr, i64 key)
;   i32 universe_ds_dheap_pop(ptr, ptr out_min)
;   i32 universe_ds_dheap_peek(ptr, ptr out_min)
;   i64 universe_ds_dheap_len(ptr)
;   ; --- radix / monotone heap ---
;   ptr universe_ds_radixheap_create(i64 initial_cap)
;   void universe_ds_radixheap_destroy(ptr)
;   i32 universe_ds_radixheap_push(ptr, i64 key, i64 val)  ; key<last -> 8
;   i32 universe_ds_radixheap_pop(ptr, ptr out_key, ptr out_val)
;   i32 universe_ds_radixheap_peek(ptr, ptr out_key)
;   i64 universe_ds_radixheap_len(ptr)
;   ; --- pairing heap ---
;   ptr universe_ds_pairing_create()
;   void universe_ds_pairing_destroy(ptr)
;   i64 universe_ds_pairing_push(ptr, i64 key, i64 val)    ; node handle, -1 fail
;   i32 universe_ds_pairing_pop(ptr, ptr out_key, ptr out_val)
;   i32 universe_ds_pairing_peek(ptr, ptr out_key, ptr out_val)
;   i32 universe_ds_pairing_decrease_key(ptr, i64 node, i64 newkey) ; > cur -> 8
;   i32 universe_ds_pairing_meld(ptr dst, ptr src)         ; absorb src into dst
;   i64 universe_ds_pairing_len(ptr)
; ============================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ###########################################################################
; d-ary heap
; ###########################################################################
; Header { ptr data@0, i64 len@8, i64 cap@16, i64 d@24 }

define noalias ptr @universe_ds_dheap_create(i64 %initial_cap, i64 %d) local_unnamed_addr #1 {
entry:
  %dbad = icmp ult i64 %d, 2
  %dd = select i1 %dbad, i64 4, i64 %d
  %c.min = call i64 @llvm.umax.i64(i64 %initial_cap, i64 8)
  %too.big = icmp ugt i64 %c.min, 1152921504606846976
  br i1 %too.big, label %fail, label %pow2, !prof !0

pow2:
  %cm1 = add i64 %c.min, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %bytes = shl nuw i64 %cap, 3
  %hdr = call ptr @malloc(i64 32)
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
  %len.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %len.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 %cap, ptr %cap.p, align 8
  %d.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 %dd, ptr %d.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_dheap_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %data = load ptr, ptr %h, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %h)
  br label %done

done:
  ret void
}

; grow the key run to 2x cap; 0 ok / 2 oom / 3 overflow.
define internal i32 @dheap_grow(ptr %h) #3 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %cap2 = shl i64 %cap, 1
  %wrapped = icmp eq i64 %cap2, 0
  br i1 %wrapped, label %ovf, label %size

size:
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 8)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  br i1 %bytes.o, label %ovf, label %do.realloc

do.realloc:
  %old = load ptr, ptr %h, align 8
  %new = call ptr @realloc(ptr %old, i64 %bytes.v)
  %new.null = icmp eq ptr %new, null
  br i1 %new.null, label %oom, label %ok

ok:
  store ptr %new, ptr %h, align 8
  store i64 %cap2, ptr %cap.p, align 8
  ret i32 0

oom:
  ret i32 2

ovf:
  ret i32 3
}

define i32 @universe_ds_dheap_push(ptr %h, i64 %key) local_unnamed_addr #1 {
entry:
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %len, %cap
  br i1 %full, label %grow, label %siftup, !prof !0

grow:
  %grc = call i32 @dheap_grow(ptr nonnull %h)
  %grew = icmp eq i32 %grc, 0
  br i1 %grew, label %siftup, label %grow.fail, !prof !2

grow.fail:
  ret i32 %grc

siftup:
  %d.p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %d = load i64, ptr %d.p, align 8
  %data = load ptr, ptr %h, align 8
  br label %loop

loop:
  %hole = phi i64 [ %len, %siftup ], [ %parent, %shift ]
  %at.root = icmp eq i64 %hole, 0
  br i1 %at.root, label %settle, label %probe

probe:
  %hm1 = sub nuw i64 %hole, 1
  %parent = udiv i64 %hm1, %d
  %pp = getelementptr inbounds nuw i64, ptr %data, i64 %parent
  %pv = load i64, ptr %pp, align 8
  %need = icmp slt i64 %key, %pv
  br i1 %need, label %shift, label %settle

shift:
  %hp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %pv, ptr %hp, align 8
  br label %loop

settle:
  %sp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  %len.n = add nuw i64 %len, 1
  store i64 %len.n, ptr %len.p, align 8
  ret i32 0
}

define i32 @universe_ds_dheap_pop(ptr %h, ptr %out) local_unnamed_addr #0 {
entry:
  %h.null = icmp eq ptr %h, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %h.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %err.empty, label %pop, !prof !0

err.empty:
  ret i32 4

pop:
  %d.p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %d = load i64, ptr %d.p, align 8
  %data = load ptr, ptr %h, align 8
  %min = load i64, ptr %data, align 8
  store i64 %min, ptr %out, align 8
  %len.n = add i64 %len, -1
  store i64 %len.n, ptr %len.p, align 8
  %lastp = getelementptr inbounds nuw i64, ptr %data, i64 %len.n
  %key = load i64, ptr %lastp, align 8
  %was.last = icmp eq i64 %len.n, 0
  br i1 %was.last, label %done, label %loop

loop:
  %hole = phi i64 [ 0, %pop ], [ %minc, %descend ]
  %base = mul i64 %hole, %d
  %first = add i64 %base, 1
  %has.child = icmp ult i64 %first, %len.n
  br i1 %has.child, label %scan.init, label %settle

scan.init:
  ; last child index (exclusive) = min(first + d, len.n)
  %firstd = add i64 %first, %d
  %stop = call i64 @llvm.umin.i64(i64 %firstd, i64 %len.n)
  %fp = getelementptr inbounds nuw i64, ptr %data, i64 %first
  %fv = load i64, ptr %fp, align 8
  %first1 = add i64 %first, 1
  br label %scan

scan:
  %c = phi i64 [ %first1, %scan.init ], [ %c.n, %scan.body ]
  %bestc = phi i64 [ %first, %scan.init ], [ %bestc.n, %scan.body ]
  %bestv = phi i64 [ %fv, %scan.init ], [ %bestv.n, %scan.body ]
  %more = icmp ult i64 %c, %stop
  br i1 %more, label %scan.body, label %scan.done

scan.body:
  %cp = getelementptr inbounds nuw i64, ptr %data, i64 %c
  %cv = load i64, ptr %cp, align 8
  %smaller = icmp slt i64 %cv, %bestv
  %bestc.n = select i1 %smaller, i64 %c, i64 %bestc
  %bestv.n = select i1 %smaller, i64 %cv, i64 %bestv
  %c.n = add nuw i64 %c, 1
  br label %scan

scan.done:
  %need = icmp slt i64 %bestv, %key
  br i1 %need, label %descend, label %settle

descend:
  %minc = phi i64 [ %bestc, %scan.done ]
  %minv = phi i64 [ %bestv, %scan.done ]
  %hp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %minv, ptr %hp, align 8
  br label %loop

settle:
  %sp = getelementptr inbounds nuw i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_ds_dheap_peek(ptr %h, ptr %out) local_unnamed_addr #0 {
entry:
  %h.null = icmp eq ptr %h, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %h.null, %out.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %err.empty, label %copy, !prof !0

err.empty:
  ret i32 4

copy:
  %data = load ptr, ptr %h, align 8
  %min = load i64, ptr %data, align 8
  store i64 %min, ptr %out, align 8
  ret i32 0
}

define i64 @universe_ds_dheap_len(ptr %h) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %h, null
  br i1 %n, label %z, label %l, !prof !0
z:
  ret i64 0
l:
  %len.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %len = load i64, ptr %len.p, align 8
  ret i64 %len
}

; ###########################################################################
; radix / monotone heap
; ###########################################################################
; Item node stride 24: key@0(i64) val@8(i64) next@16(i32) pad@20
; Header 320: pool@0 cap@8 nnodes@16 count@24 last@32 free@40(i32) pad@44
;             buckets@48 (i32[65], -1 = empty)

; bucket index for a key given the current boundary `last`.
define internal i64 @rh_bucket(i64 %key, i64 %last) #4 {
entry:
  %diff = xor i64 %key, %last
  %z = icmp eq i64 %diff, 0
  br i1 %z, label %ret0, label %calc

ret0:
  ret i64 0

calc:
  %clz = call i64 @llvm.ctlz.i64(i64 %diff, i1 true)
  %b = sub i64 64, %clz
  ret i64 %b
}

define internal i64 @rh_alloc(ptr %h) #5 {
entry:
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 40
  %fh = load i32, ptr %fhp, align 4
  %hasfree = icmp sge i32 %fh, 0
  br i1 %hasfree, label %pop, label %bump

pop:
  %base = load ptr, ptr %h, align 8
  %fhi = zext i32 %fh to i64
  %poff = mul nuw i64 %fhi, 24
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  %pnextp = getelementptr inbounds nuw i8, ptr %pnode, i64 16
  %next = load i32, ptr %pnextp, align 4
  store i32 %next, ptr %fhp, align 4
  ret i64 %fhi

bump:
  %nnp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nn = load i64, ptr %nnp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %cap = load i64, ptr %capp, align 8
  %atcap = icmp uge i64 %nn, %cap
  br i1 %atcap, label %grow, label %place, !prof !0

grow:
  %cap2 = shl i64 %cap, 1
  %wrap = icmp eq i64 %cap2, 0
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 24)
  %mv = extractvalue { i64, i1 } %m, 0
  %mo = extractvalue { i64, i1 } %m, 1
  %bad = or i1 %wrap, %mo
  br i1 %bad, label %fail, label %dorealloc, !prof !0

dorealloc:
  %old = load ptr, ptr %h, align 8
  %new = call ptr @realloc(ptr %old, i64 %mv)
  %newnull = icmp eq ptr %new, null
  br i1 %newnull, label %fail, label %okgrow, !prof !0

okgrow:
  store ptr %new, ptr %h, align 8
  store i64 %cap2, ptr %capp, align 8
  br label %place

place:
  %idx = load i64, ptr %nnp, align 8
  %idx1 = add nuw i64 %idx, 1
  store i64 %idx1, ptr %nnp, align 8
  ret i64 %idx

fail:
  ret i64 -1
}

define internal void @rh_free(ptr %h, i64 %idx) #4 {
entry:
  %base = load ptr, ptr %h, align 8
  %off = mul nuw i64 %idx, 24
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %nextp = getelementptr inbounds nuw i8, ptr %node, i64 16
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 40
  %fh = load i32, ptr %fhp, align 4
  store i32 %fh, ptr %nextp, align 4
  %idx32 = trunc i64 %idx to i32
  store i32 %idx32, ptr %fhp, align 4
  ret void
}

define noalias ptr @universe_ds_radixheap_create(i64 %initial_cap) local_unnamed_addr #1 {
entry:
  %c.min = call i64 @llvm.umax.i64(i64 %initial_cap, i64 16)
  %hdr = call ptr @malloc(i64 320)
  %hnull = icmp eq ptr %hdr, null
  br i1 %hnull, label %fail, label %adata, !prof !0

adata:
  %bytes = mul nuw i64 %c.min, 24
  %pool = call ptr @malloc(i64 %bytes)
  %pnull = icmp eq ptr %pool, null
  br i1 %pnull, label %freehdr, label %init, !prof !0

freehdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %pool, ptr %hdr, align 8
  %capp = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 %c.min, ptr %capp, align 8
  %nnp = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 0, ptr %nnp, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 0, ptr %cntp, align 8
  %lastp = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 0, ptr %lastp, align 8
  %fhp = getelementptr inbounds nuw i8, ptr %hdr, i64 40
  store i32 -1, ptr %fhp, align 4
  %bkts = getelementptr inbounds nuw i8, ptr %hdr, i64 48
  call void @llvm.memset.p0.i64(ptr %bkts, i8 -1, i64 260, i1 false)
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_radixheap_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %h, null
  br i1 %isnull, label %done, label %dofree, !prof !0

dofree:
  %pool = load ptr, ptr %h, align 8
  call void @free(ptr %pool)
  call void @free(ptr nonnull %h)
  br label %done

done:
  ret void
}

define i32 @universe_ds_radixheap_push(ptr %h, i64 %key, i64 %val) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %lastp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %last = load i64, ptr %lastp, align 8
  %bad = icmp slt i64 %key, %last
  br i1 %bad, label %err.arg, label %alloc, !prof !0

err.arg:
  ret i32 8

alloc:
  %z = call i64 @rh_alloc(ptr %h)
  %zbad = icmp slt i64 %z, 0
  br i1 %zbad, label %err.oom, label %fill, !prof !0

err.oom:
  ret i32 2

fill:
  %base = load ptr, ptr %h, align 8
  %zoff = mul nuw i64 %z, 24
  %znode = getelementptr inbounds nuw i8, ptr %base, i64 %zoff
  store i64 %key, ptr %znode, align 8
  %zvp = getelementptr inbounds nuw i8, ptr %znode, i64 8
  store i64 %val, ptr %zvp, align 8
  %b = call i64 @rh_bucket(i64 %key, i64 %last)
  %bkts = getelementptr inbounds nuw i8, ptr %h, i64 48
  %bp = getelementptr inbounds nuw i32, ptr %bkts, i64 %b
  %oldhead = load i32, ptr %bp, align 4
  %znextp = getelementptr inbounds nuw i8, ptr %znode, i64 16
  store i32 %oldhead, ptr %znextp, align 4
  %z32 = trunc i64 %z to i32
  store i32 %z32, ptr %bp, align 4
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  ret i32 0
}

define i32 @universe_ds_radixheap_pop(ptr %h, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  %oknull = icmp eq ptr %ok, null
  %ovnull = icmp eq ptr %ov, null
  %n1 = or i1 %hnull, %oknull
  %anynull = or i1 %n1, %ovnull
  br i1 %anynull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %err.empty, label %setup, !prof !0

err.empty:
  ret i32 4

setup:
  %base = load ptr, ptr %h, align 8
  %bkts = getelementptr inbounds nuw i8, ptr %h, i64 48
  %b0 = load i32, ptr %bkts, align 4
  %b0empty = icmp eq i32 %b0, -1
  br i1 %b0empty, label %refill, label %take0

refill:
  ; find smallest non-empty bucket bi in [1,64]
  br label %find

find:
  %bi = phi i64 [ 1, %refill ], [ %bi.n, %find.next ]
  %bip = getelementptr inbounds nuw i32, ptr %bkts, i64 %bi
  %bihead = load i32, ptr %bip, align 4
  %found = icmp ne i32 %bihead, -1
  br i1 %found, label %minscan.init, label %find.next

find.next:
  %bi.n = add nuw nsw i64 %bi, 1
  br label %find

minscan.init:
  ; scan bucket bi list for the minimum key -> new last
  %bihead64 = zext i32 %bihead to i64
  %h0off = mul nuw i64 %bihead64, 24
  %h0node = getelementptr inbounds nuw i8, ptr %base, i64 %h0off
  %h0key = load i64, ptr %h0node, align 8
  br label %minscan

minscan:
  %mcur = phi i32 [ %bihead, %minscan.init ], [ %mnext, %minscan.step ]
  %mmin = phi i64 [ %h0key, %minscan.init ], [ %mmin.n, %minscan.step ]
  %mnil = icmp eq i32 %mcur, -1
  br i1 %mnil, label %redist.init, label %minscan.step

minscan.step:
  %mcur64 = zext i32 %mcur to i64
  %mcoff = mul nuw i64 %mcur64, 24
  %mcnode = getelementptr inbounds nuw i8, ptr %base, i64 %mcoff
  %mck = load i64, ptr %mcnode, align 8
  %mless = icmp slt i64 %mck, %mmin
  %mmin.n = select i1 %mless, i64 %mck, i64 %mmin
  %mnextp = getelementptr inbounds nuw i8, ptr %mcnode, i64 16
  %mnext = load i32, ptr %mnextp, align 4
  br label %minscan

redist.init:
  ; set new boundary; detach bucket bi
  %lastp = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 %mmin, ptr %lastp, align 8
  store i32 -1, ptr %bip, align 4
  br label %redist

redist:
  %rcur = phi i32 [ %bihead, %redist.init ], [ %rnext, %redist.step ]
  %rnil = icmp eq i32 %rcur, -1
  br i1 %rnil, label %take0, label %redist.step

redist.step:
  %rcur64 = zext i32 %rcur to i64
  %rcoff = mul nuw i64 %rcur64, 24
  %rcnode = getelementptr inbounds nuw i8, ptr %base, i64 %rcoff
  %rck = load i64, ptr %rcnode, align 8
  %rnextp = getelementptr inbounds nuw i8, ptr %rcnode, i64 16
  %rnext = load i32, ptr %rnextp, align 4
  ; new bucket for this node under new last (%mmin)
  %nb = call i64 @rh_bucket(i64 %rck, i64 %mmin)
  %nbp = getelementptr inbounds nuw i32, ptr %bkts, i64 %nb
  %nbhead = load i32, ptr %nbp, align 4
  store i32 %nbhead, ptr %rnextp, align 4
  store i32 %rcur, ptr %nbp, align 4
  br label %redist

take0:
  %head = load i32, ptr %bkts, align 4
  %head64 = zext i32 %head to i64
  %hoff = mul nuw i64 %head64, 24
  %hnode = getelementptr inbounds nuw i8, ptr %base, i64 %hoff
  %hk = load i64, ptr %hnode, align 8
  store i64 %hk, ptr %ok, align 8
  %hvp = getelementptr inbounds nuw i8, ptr %hnode, i64 8
  %hv = load i64, ptr %hvp, align 8
  store i64 %hv, ptr %ov, align 8
  %hnextp = getelementptr inbounds nuw i8, ptr %hnode, i64 16
  %hnext = load i32, ptr %hnextp, align 4
  store i32 %hnext, ptr %bkts, align 4
  call void @rh_free(ptr %h, i64 %head64)
  %cnt2 = load i64, ptr %cntp, align 8
  %cnt2d = add i64 %cnt2, -1
  store i64 %cnt2d, ptr %cntp, align 8
  ret i32 0
}

define i32 @universe_ds_radixheap_peek(ptr %h, ptr %ok) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  %oknull = icmp eq ptr %ok, null
  %anynull = or i1 %hnull, %oknull
  br i1 %anynull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %err.empty, label %setup, !prof !0

err.empty:
  ret i32 4

setup:
  %base = load ptr, ptr %h, align 8
  %bkts = getelementptr inbounds nuw i8, ptr %h, i64 48
  %b0 = load i32, ptr %bkts, align 4
  %b0ok = icmp ne i32 %b0, -1
  br i1 %b0ok, label %uselast, label %refill

uselast:
  %lastp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %last = load i64, ptr %lastp, align 8
  store i64 %last, ptr %ok, align 8
  ret i32 0

refill:
  br label %find

find:
  %bi = phi i64 [ 1, %refill ], [ %bi.n, %find.next ]
  %bip = getelementptr inbounds nuw i32, ptr %bkts, i64 %bi
  %bihead = load i32, ptr %bip, align 4
  %found = icmp ne i32 %bihead, -1
  br i1 %found, label %minscan.init, label %find.next

find.next:
  %bi.n = add nuw nsw i64 %bi, 1
  br label %find

minscan.init:
  %bihead64 = zext i32 %bihead to i64
  %h0off = mul nuw i64 %bihead64, 24
  %h0node = getelementptr inbounds nuw i8, ptr %base, i64 %h0off
  %h0key = load i64, ptr %h0node, align 8
  br label %minscan

minscan:
  %mcur = phi i32 [ %bihead, %minscan.init ], [ %mnext, %minscan.step ]
  %mmin = phi i64 [ %h0key, %minscan.init ], [ %mmin.n, %minscan.step ]
  %mnil = icmp eq i32 %mcur, -1
  br i1 %mnil, label %emit, label %minscan.step

minscan.step:
  %mcur64 = zext i32 %mcur to i64
  %mcoff = mul nuw i64 %mcur64, 24
  %mcnode = getelementptr inbounds nuw i8, ptr %base, i64 %mcoff
  %mck = load i64, ptr %mcnode, align 8
  %mless = icmp slt i64 %mck, %mmin
  %mmin.n = select i1 %mless, i64 %mck, i64 %mmin
  %mnextp = getelementptr inbounds nuw i8, ptr %mcnode, i64 16
  %mnext = load i32, ptr %mnextp, align 4
  br label %minscan

emit:
  store i64 %mmin, ptr %ok, align 8
  ret i32 0
}

define i64 @universe_ds_radixheap_len(ptr %h) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %h, null
  br i1 %n, label %z, label %l, !prof !0
z:
  ret i64 0
l:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  ret i64 %cnt
}

; ###########################################################################
; pairing heap
; ###########################################################################
; Node stride 32: key@0(i64) val@8(i64) child@16(i32) next@20(i32) prev@24(i32)
;                 pad@28
; Header 40: pool@0 cap@8 nnodes@16 count@24 root@32(i32) free@36(i32)

; Link two subtree roots; the smaller-keyed becomes parent, the other its new
; leftmost child. Returns the winner (new subtree root) index. Does NOT set the
; winner's own next/prev (caller owns those).
define internal i64 @pr_link(ptr %base, i64 %a, i64 %b) #4 {
entry:
  %aoff = mul nuw i64 %a, 32
  %anode = getelementptr inbounds nuw i8, ptr %base, i64 %aoff
  %ak = load i64, ptr %anode, align 8
  %boff = mul nuw i64 %b, 32
  %bnode = getelementptr inbounds nuw i8, ptr %base, i64 %boff
  %bk = load i64, ptr %bnode, align 8
  %ale = icmp sle i64 %ak, %bk
  %parent = select i1 %ale, i64 %a, i64 %b
  %chld = select i1 %ale, i64 %b, i64 %a
  %poff = mul nuw i64 %parent, 32
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  %coff = mul nuw i64 %chld, 32
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %pchildp = getelementptr inbounds nuw i8, ptr %pnode, i64 16
  %oldc = load i32, ptr %pchildp, align 4
  %cnextp = getelementptr inbounds nuw i8, ptr %cnode, i64 20
  store i32 %oldc, ptr %cnextp, align 4
  %hasold = icmp ne i32 %oldc, -1
  br i1 %hasold, label %fixold, label %setchild

fixold:
  %oldc64 = zext i32 %oldc to i64
  %ocoff = mul nuw i64 %oldc64, 32
  %ocnode = getelementptr inbounds nuw i8, ptr %base, i64 %ocoff
  %ocprevp = getelementptr inbounds nuw i8, ptr %ocnode, i64 24
  %chld32.a = trunc i64 %chld to i32
  store i32 %chld32.a, ptr %ocprevp, align 4
  br label %setchild

setchild:
  %cprevp = getelementptr inbounds nuw i8, ptr %cnode, i64 24
  %parent32 = trunc i64 %parent to i32
  store i32 %parent32, ptr %cprevp, align 4
  %chld32 = trunc i64 %chld to i32
  store i32 %chld32, ptr %pchildp, align 4
  ret i64 %parent
}

define internal i64 @pr_alloc(ptr %h) #5 {
entry:
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 36
  %fh = load i32, ptr %fhp, align 4
  %hasfree = icmp sge i32 %fh, 0
  br i1 %hasfree, label %pop, label %bump

pop:
  %base = load ptr, ptr %h, align 8
  %fhi = zext i32 %fh to i64
  %poff = mul nuw i64 %fhi, 32
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  ; next-free stored in child slot @16 while free
  %pnextp = getelementptr inbounds nuw i8, ptr %pnode, i64 16
  %next = load i32, ptr %pnextp, align 4
  store i32 %next, ptr %fhp, align 4
  ret i64 %fhi

bump:
  %nnp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nn = load i64, ptr %nnp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %cap = load i64, ptr %capp, align 8
  %atcap = icmp uge i64 %nn, %cap
  br i1 %atcap, label %grow, label %place, !prof !0

grow:
  %cap2 = shl i64 %cap, 1
  %wrap = icmp eq i64 %cap2, 0
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 32)
  %mv = extractvalue { i64, i1 } %m, 0
  %mo = extractvalue { i64, i1 } %m, 1
  %bad = or i1 %wrap, %mo
  br i1 %bad, label %fail, label %dorealloc, !prof !0

dorealloc:
  %old = load ptr, ptr %h, align 8
  %new = call ptr @realloc(ptr %old, i64 %mv)
  %newnull = icmp eq ptr %new, null
  br i1 %newnull, label %fail, label %okgrow, !prof !0

okgrow:
  store ptr %new, ptr %h, align 8
  store i64 %cap2, ptr %capp, align 8
  br label %place

place:
  %idx = load i64, ptr %nnp, align 8
  %idx1 = add nuw i64 %idx, 1
  store i64 %idx1, ptr %nnp, align 8
  ret i64 %idx

fail:
  ret i64 -1
}

define internal void @pr_free(ptr %h, i64 %idx) #4 {
entry:
  %base = load ptr, ptr %h, align 8
  %off = mul nuw i64 %idx, 32
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %nextp = getelementptr inbounds nuw i8, ptr %node, i64 16
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 36
  %fh = load i32, ptr %fhp, align 4
  store i32 %fh, ptr %nextp, align 4
  %idx32 = trunc i64 %idx to i32
  store i32 %idx32, ptr %fhp, align 4
  ret void
}

define noalias ptr @universe_ds_pairing_create() local_unnamed_addr #1 {
entry:
  %hdr = call ptr @malloc(i64 40)
  %hnull = icmp eq ptr %hdr, null
  br i1 %hnull, label %fail, label %adata, !prof !0

adata:
  ; 16 initial node slots * 32 = 512 bytes
  %pool = call ptr @malloc(i64 512)
  %pnull = icmp eq ptr %pool, null
  br i1 %pnull, label %freehdr, label %init, !prof !0

freehdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %pool, ptr %hdr, align 8
  %capp = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 16, ptr %capp, align 8
  %nnp = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 0, ptr %nnp, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 0, ptr %cntp, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i32 -1, ptr %rootp, align 4
  %fhp = getelementptr inbounds nuw i8, ptr %hdr, i64 36
  store i32 -1, ptr %fhp, align 4
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_pairing_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %h, null
  br i1 %isnull, label %done, label %dofree, !prof !0

dofree:
  %pool = load ptr, ptr %h, align 8
  call void @free(ptr %pool)
  call void @free(ptr nonnull %h)
  br label %done

done:
  ret void
}

define i64 @universe_ds_pairing_push(ptr %h, i64 %key, i64 %val) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail, label %alloc, !prof !0

fail:
  ret i64 -1

alloc:
  %z = call i64 @pr_alloc(ptr %h)
  %zbad = icmp slt i64 %z, 0
  br i1 %zbad, label %fail, label %fill, !prof !0

fill:
  %base = load ptr, ptr %h, align 8
  %zoff = mul nuw i64 %z, 32
  %znode = getelementptr inbounds nuw i8, ptr %base, i64 %zoff
  store i64 %key, ptr %znode, align 8
  %zvp = getelementptr inbounds nuw i8, ptr %znode, i64 8
  store i64 %val, ptr %zvp, align 8
  %zchildp = getelementptr inbounds nuw i8, ptr %znode, i64 16
  store i32 -1, ptr %zchildp, align 4
  %znextp = getelementptr inbounds nuw i8, ptr %znode, i64 20
  store i32 -1, ptr %znextp, align 4
  %zprevp = getelementptr inbounds nuw i8, ptr %znode, i64 24
  store i32 -1, ptr %zprevp, align 4
  %rootp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %root = load i32, ptr %rootp, align 4
  %empty = icmp eq i32 %root, -1
  br i1 %empty, label %asroot, label %meld

asroot:
  %z32 = trunc i64 %z to i32
  store i32 %z32, ptr %rootp, align 4
  br label %fin

meld:
  %root64 = zext i32 %root to i64
  %w = call i64 @pr_link(ptr %base, i64 %root64, i64 %z)
  %w32 = trunc i64 %w to i32
  store i32 %w32, ptr %rootp, align 4
  %woff = mul nuw i64 %w, 32
  %wnode = getelementptr inbounds nuw i8, ptr %base, i64 %woff
  %wnextp = getelementptr inbounds nuw i8, ptr %wnode, i64 20
  store i32 -1, ptr %wnextp, align 4
  %wprevp = getelementptr inbounds nuw i8, ptr %wnode, i64 24
  store i32 -1, ptr %wprevp, align 4
  br label %fin

fin:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  ret i64 %z
}

define i32 @universe_ds_pairing_peek(ptr %h, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %rootp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %root = load i32, ptr %rootp, align 4
  %empty = icmp eq i32 %root, -1
  br i1 %empty, label %err.empty, label %emit, !prof !0

err.empty:
  ret i32 4

emit:
  %base = load ptr, ptr %h, align 8
  %root64 = zext i32 %root to i64
  %off = mul nuw i64 %root64, 32
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %k = load i64, ptr %node, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %doval, label %storek

storek:
  store i64 %k, ptr %ok, align 8
  br label %doval

doval:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %ret0, label %storeval

storeval:
  %vp = getelementptr inbounds nuw i8, ptr %node, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  br label %ret0

ret0:
  ret i32 0
}

define i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %rootp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %root = load i32, ptr %rootp, align 4
  %empty = icmp eq i32 %root, -1
  br i1 %empty, label %err.empty, label %extract, !prof !0

err.empty:
  ret i32 4

extract:
  %base = load ptr, ptr %h, align 8
  %root64 = zext i32 %root to i64
  %roff = mul nuw i64 %root64, 32
  %rnode = getelementptr inbounds nuw i8, ptr %base, i64 %roff
  %rk = load i64, ptr %rnode, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %dov, label %sk

sk:
  store i64 %rk, ptr %ok, align 8
  br label %dov

dov:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %children, label %sv

sv:
  %rvp = getelementptr inbounds nuw i8, ptr %rnode, i64 8
  %rv = load i64, ptr %rvp, align 8
  store i64 %rv, ptr %ov, align 8
  br label %children

children:
  %rchildp = getelementptr inbounds nuw i8, ptr %rnode, i64 16
  %c0 = load i32, ptr %rchildp, align 4
  call void @pr_free(ptr %h, i64 %root64)
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cntd = add i64 %cnt, -1
  store i64 %cntd, ptr %cntp, align 8
  %nochild = icmp eq i32 %c0, -1
  br i1 %nochild, label %clear, label %pass1.head

clear:
  store i32 -1, ptr %rootp, align 4
  ret i32 0

; ---- pass 1: pair adjacent siblings L->R, pushing results onto a stack
;      threaded through the prev field ----
pass1.head:
  %c = phi i32 [ %c0, %children ], [ %cnext, %pair ]
  %stack = phi i32 [ -1, %children ], [ %m32, %pair ]
  %cnil = icmp eq i32 %c, -1
  br i1 %cnil, label %pass2.init, label %pass1.body

pass1.body:
  %a = zext i32 %c to i64
  %aoff = mul nuw i64 %a, 32
  %anode = getelementptr inbounds nuw i8, ptr %base, i64 %aoff
  %anextp = getelementptr inbounds nuw i8, ptr %anode, i64 20
  %bsib = load i32, ptr %anextp, align 4
  %bnil = icmp eq i32 %bsib, -1
  br i1 %bnil, label %single, label %pair

single:
  ; leftover single: push onto stack
  store i32 -1, ptr %anextp, align 4
  %aprevp = getelementptr inbounds nuw i8, ptr %anode, i64 24
  store i32 %stack, ptr %aprevp, align 4
  br label %pass2.init

pair:
  %b = zext i32 %bsib to i64
  %boff = mul nuw i64 %b, 32
  %bnode = getelementptr inbounds nuw i8, ptr %base, i64 %boff
  %bnextp = getelementptr inbounds nuw i8, ptr %bnode, i64 20
  %cnext = load i32, ptr %bnextp, align 4
  %m = call i64 @pr_link(ptr %base, i64 %a, i64 %b)
  %m32 = trunc i64 %m to i32
  %moff = mul nuw i64 %m, 32
  %mnode = getelementptr inbounds nuw i8, ptr %base, i64 %moff
  %mnextp = getelementptr inbounds nuw i8, ptr %mnode, i64 20
  store i32 -1, ptr %mnextp, align 4
  %mprevp = getelementptr inbounds nuw i8, ptr %mnode, i64 24
  store i32 %stack, ptr %mprevp, align 4
  br label %pass1.head

; ---- pass 2: fold the stack (R->L order) into one tree ----
pass2.init:
  %fstack = phi i32 [ %stack, %pass1.head ], [ %c, %single ]
  br label %pass2.head

pass2.head:
  %s = phi i32 [ %fstack, %pass2.init ], [ %snext, %pass2.next ]
  %acc = phi i32 [ -1, %pass2.init ], [ %newacc, %pass2.next ]
  %snil = icmp eq i32 %s, -1
  br i1 %snil, label %finish, label %pass2.body

pass2.body:
  %t64 = zext i32 %s to i64
  %toff = mul nuw i64 %t64, 32
  %tnode = getelementptr inbounds nuw i8, ptr %base, i64 %toff
  %tprevp = getelementptr inbounds nuw i8, ptr %tnode, i64 24
  %snext = load i32, ptr %tprevp, align 4
  %accnil = icmp eq i32 %acc, -1
  br i1 %accnil, label %setfirst, label %meldacc

setfirst:
  br label %pass2.next

meldacc:
  %acc64 = zext i32 %acc to i64
  %w = call i64 @pr_link(ptr %base, i64 %acc64, i64 %t64)
  %w32 = trunc i64 %w to i32
  br label %pass2.next

pass2.next:
  %newacc = phi i32 [ %s, %setfirst ], [ %w32, %meldacc ]
  br label %pass2.head

finish:
  ; acc is the new root
  store i32 %acc, ptr %rootp, align 4
  %acc64f = zext i32 %acc to i64
  %aoff2 = mul nuw i64 %acc64f, 32
  %anode2 = getelementptr inbounds nuw i8, ptr %base, i64 %aoff2
  %anextp2 = getelementptr inbounds nuw i8, ptr %anode2, i64 20
  store i32 -1, ptr %anextp2, align 4
  %aprevp2 = getelementptr inbounds nuw i8, ptr %anode2, i64 24
  store i32 -1, ptr %aprevp2, align 4
  ret i32 0
}

define i32 @universe_ds_pairing_decrease_key(ptr %h, i64 %node, i64 %newkey) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %base = load ptr, ptr %h, align 8
  %noff = mul nuw i64 %node, 32
  %nnode = getelementptr inbounds nuw i8, ptr %base, i64 %noff
  %curk = load i64, ptr %nnode, align 8
  %bigger = icmp sgt i64 %newkey, %curk
  br i1 %bigger, label %err.arg, label %apply, !prof !0

err.arg:
  ret i32 8

apply:
  store i64 %newkey, ptr %nnode, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %root = load i32, ptr %rootp, align 4
  %node32 = trunc i64 %node to i32
  %isroot = icmp eq i32 %root, %node32
  br i1 %isroot, label %ret0, label %cut

cut:
  %prevp = getelementptr inbounds nuw i8, ptr %nnode, i64 24
  %p = load i32, ptr %prevp, align 4
  %nextp = getelementptr inbounds nuw i8, ptr %nnode, i64 20
  %nx = load i32, ptr %nextp, align 4
  %p64 = zext i32 %p to i64
  %poff = mul nuw i64 %p64, 32
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  %pchildp = getelementptr inbounds nuw i8, ptr %pnode, i64 16
  %pchild = load i32, ptr %pchildp, align 4
  %isleftmost = icmp eq i32 %pchild, %node32
  br i1 %isleftmost, label %fixparent, label %fixsib

fixparent:
  store i32 %nx, ptr %pchildp, align 4
  br label %fixnx

fixsib:
  %psibnextp = getelementptr inbounds nuw i8, ptr %pnode, i64 20
  store i32 %nx, ptr %psibnextp, align 4
  br label %fixnx

fixnx:
  %hasnx = icmp ne i32 %nx, -1
  br i1 %hasnx, label %setnxprev, label %detach

setnxprev:
  %nx64 = zext i32 %nx to i64
  %nxoff = mul nuw i64 %nx64, 32
  %nxnode = getelementptr inbounds nuw i8, ptr %base, i64 %nxoff
  %nxprevp = getelementptr inbounds nuw i8, ptr %nxnode, i64 24
  store i32 %p, ptr %nxprevp, align 4
  br label %detach

detach:
  store i32 -1, ptr %nextp, align 4
  store i32 -1, ptr %prevp, align 4
  %root64 = zext i32 %root to i64
  %w = call i64 @pr_link(ptr %base, i64 %root64, i64 %node)
  %w32 = trunc i64 %w to i32
  store i32 %w32, ptr %rootp, align 4
  %woff = mul nuw i64 %w, 32
  %wnode = getelementptr inbounds nuw i8, ptr %base, i64 %woff
  %wnextp = getelementptr inbounds nuw i8, ptr %wnode, i64 20
  store i32 -1, ptr %wnextp, align 4
  %wprevp = getelementptr inbounds nuw i8, ptr %wnode, i64 24
  store i32 -1, ptr %wprevp, align 4
  br label %ret0

ret0:
  ret i32 0
}

; Absorb src into dst by draining src (pop) and pushing into dst. O(n_src);
; a true O(1) meld is impossible across two independent index pools, so this
; combines the two heaps correctly while leaving the O(1) link for internal use.
define i32 @universe_ds_pairing_meld(ptr %dst, ptr %src) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %kslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  br label %loop

loop:
  %r = call i32 @universe_ds_pairing_pop(ptr %src, ptr %kslot, ptr %vslot)
  %done = icmp ne i32 %r, 0
  br i1 %done, label %fin, label %push

push:
  %k = load i64, ptr %kslot, align 8
  %v = load i64, ptr %vslot, align 8
  %hnd = call i64 @universe_ds_pairing_push(ptr %dst, i64 %k, i64 %v)
  br label %loop

fin:
  ret i32 0
}

define i64 @universe_ds_pairing_len(ptr %h) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %h, null
  br i1 %n, label %z, label %l, !prof !0
z:
  ret i64 0
l:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  ret i64 %cnt
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #3 = { cold nounwind willreturn }
attributes #4 = { nounwind willreturn norecurse nosync }
attributes #5 = { nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
