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

; universe_ds_btree — array-backed B-tree, i64 key -> i64 value, single thread.
;
; ============================================================================
; DESIGN (from first principles)
; ----------------------------------------------------------------------------
;   * ORDER t = 8: each node holds up to 2t-1 = 15 keys and 2t = 16 children,
;     with a minimum of t-1 = 7 keys (root exempt). A high fan-out keeps the
;     tree shallow (log_8 n): 64K keys is height <= 6, so every op touches at
;     most ~6 nodes. Keys of a node sit CONTIGUOUSLY so an in-node lookup is a
;     branch-lean binary search over a cache-resident run.
;
;   * CONTIGUOUS-TREE / INDEX-LINKED idiom: ALL nodes live in ONE flat, growable
;     array; a node references its children by i32 INDEX, never by pointer. This
;     halves child-link footprint vs 8-byte pointers, keeps the whole tree in
;     one allocation (great locality, one free), and — crucially — lets the
;     backing array double via realloc without rewriting a single interior link
;     (indices are position-independent). The header holds the array pointer so
;     a realloc that moves the block only updates one word.
;
;   * FREELIST + BUMP CURSOR (wilderness): fresh nodes come from a bump cursor
;     (nnodes); nodes freed by delete-merges are pushed onto an intrusive i32
;     freelist threaded through the node's first word (its nkeys slot, reused
;     while free). alloc pops the freelist first, else bumps, else grows 2x.
;     Create is O(1) and never pre-links anything.
;
;   * PROACTIVE (top-down) INSERT: split any full child BEFORE descending into
;     it, so a split never propagates back up — one downward pass, no parent
;     stack. A full root is split by growing a new root above it first.
;
;   * TOP-DOWN DELETE: before descending into a child that has only t-1 keys,
;     refill it to >= t (borrow from a sibling, else merge with one), so the
;     node we recurse into can always afford to lose a key — again one pass, no
;     rebalance backtracking. Internal-key deletion swaps in the in-order
;     predecessor/successor (whichever side can spare a key) then deletes that
;     leaf key; if neither can, the two children plus the separator merge.
;
;   * All node moves are ONE llvm.memmove / llvm.memcpy over the contiguous
;     key / value / child runs — never element loops. Size math on growth is
;     overflow-checked. Error/cold paths carry !prof weights.
;
;   * Comparisons are SIGNED (icmp slt/sgt) so the full i64 range orders
;     naturally for floor/ceiling/range/min/max.
;
; ----------------------------------------------------------------------------
; Node record (320 B = 5 cache lines, fixed stride):
;   nkeys  i32       @0     (while on freelist: next-free index)
;   leaf   i32       @4     (1 = leaf, 0 = internal)
;   keys   i64[15]   @8     [8,128)
;   vals   i64[15]   @128   [128,248)
;   child  i32[16]   @248   [248,312)
;   pad              @312   [312,320)
; Header (64 B):
;   nodes  ptr       @0     flat node array base
;   cap    i64       @8     node capacity (slots)
;   nnodes i64       @16    bump high-water (slots ever handed out)
;   count  i64       @24    live key count
;   root   i32       @32    root node index
;   free   i32       @36    freelist head (-1 = empty)
; ----------------------------------------------------------------------------
; API (0 OK, 1 NULL_PTR, 2 OOM, 4 EMPTY, 5 NOT_FOUND):
;   ptr  universe_ds_btree_create()
;   void universe_ds_btree_destroy(ptr)
;   i32  universe_ds_btree_insert(ptr, i64 key, i64 val)   ; overwrites on dup
;   i32  universe_ds_btree_delete(ptr, i64 key)
;   i32  universe_ds_btree_find(ptr, i64 key, ptr outval)  ; outval may be null
;   i32  universe_ds_btree_contains(ptr, i64 key)          ; 1 present / 0 absent
;   i32  universe_ds_btree_min(ptr, ptr outkey, ptr outval)
;   i32  universe_ds_btree_max(ptr, ptr outkey, ptr outval)
;   i32  universe_ds_btree_floor(ptr, i64 key, ptr ok, ptr ov)   ; largest <= key
;   i32  universe_ds_btree_ceiling(ptr, i64 key, ptr ok, ptr ov) ; smallest >= key
;   i64  universe_ds_btree_range(ptr, i64 lo, i64 hi, ptr ok, ptr ov, i64 max)
;   i64  universe_ds_btree_count(ptr)                      ; total keys
;   i64  universe_ds_btree_size(ptr)                       ; alias of count
; ============================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ===========================================================================
; internal helpers
; ===========================================================================

; Binary search within a node: first index with keys[idx] >= key (lower bound),
; in [0, nkeys]. SIGNED compare.
define internal i64 @bt_lb(ptr %node, i64 %key) #3 {
entry:
  %nk32 = load i32, ptr %node, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %node, i64 8
  br label %loop

loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %nk, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done

step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %lt = icmp slt i64 %kv, %key
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %lt, i64 %mid1, i64 %lo
  %hi.n = select i1 %lt, i64 %hi, i64 %mid
  br label %loop

done:
  ret i64 %lo
}

; Push node %idx onto the intrusive freelist (next stored at node word 0).
define internal void @bt_free_node(ptr %t, i64 %idx) #4 {
entry:
  %base = load ptr, ptr %t, align 8
  %off = mul nuw i64 %idx, 320
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %fhp = getelementptr inbounds nuw i8, ptr %t, i64 36
  %fh = load i32, ptr %fhp, align 4
  store i32 %fh, ptr %node, align 4
  %idx32 = trunc i64 %idx to i32
  store i32 %idx32, ptr %fhp, align 4
  ret void
}

; Allocate a node slot, growing the array 2x if needed. Returns index or -1.
define internal i64 @bt_alloc_node(ptr %t) #2 {
entry:
  %fhp = getelementptr inbounds nuw i8, ptr %t, i64 36
  %fh = load i32, ptr %fhp, align 4
  %hasfree = icmp sge i32 %fh, 0
  br i1 %hasfree, label %pop, label %bump

pop:
  %base = load ptr, ptr %t, align 8
  %fhi = zext i32 %fh to i64
  %poff = mul nuw i64 %fhi, 320
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  %next = load i32, ptr %pnode, align 4
  store i32 %next, ptr %fhp, align 4
  ret i64 %fhi

bump:
  %nnp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %nn = load i64, ptr %nnp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %cap = load i64, ptr %capp, align 8
  %atcap = icmp uge i64 %nn, %cap
  br i1 %atcap, label %grow, label %place, !prof !0

grow:
  %cap2 = shl i64 %cap, 1
  %wrap = icmp eq i64 %cap2, 0
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 320)
  %mv = extractvalue { i64, i1 } %m, 0
  %mo = extractvalue { i64, i1 } %m, 1
  %bad = or i1 %wrap, %mo
  br i1 %bad, label %fail, label %dorealloc, !prof !0

dorealloc:
  %old = load ptr, ptr %t, align 8
  %new = call ptr @realloc(ptr %old, i64 %mv)
  %newnull = icmp eq ptr %new, null
  br i1 %newnull, label %fail, label %okgrow, !prof !0

okgrow:
  store ptr %new, ptr %t, align 8
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

; Split full child %i of node %xidx. Median rises into x; upper half -> new z.
; Returns 0 OK / 2 OOM. May realloc the node array (caller must reload base).
define internal i32 @bt_split_child(ptr %t, i64 %xidx, i64 %i) #2 {
entry:
  %z = call i64 @bt_alloc_node(ptr %t)
  %zbad = icmp slt i64 %z, 0
  br i1 %zbad, label %oom, label %go, !prof !0

oom:
  ret i32 2

go:
  %base = load ptr, ptr %t, align 8
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %xchild = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cip = getelementptr inbounds nuw i32, ptr %xchild, i64 %i
  %yi32 = load i32, ptr %cip, align 4
  %yi = zext i32 %yi32 to i64
  %yoff = mul nuw i64 %yi, 320
  %y = getelementptr inbounds nuw i8, ptr %base, i64 %yoff
  %zoff = mul nuw i64 %z, 320
  %zn = getelementptr inbounds nuw i8, ptr %base, i64 %zoff
  %ylp = getelementptr inbounds nuw i8, ptr %y, i64 4
  %yleaf = load i32, ptr %ylp, align 4
  ; z.nkeys = 7, z.leaf = y.leaf
  store i32 7, ptr %zn, align 4
  %zlp = getelementptr inbounds nuw i8, ptr %zn, i64 4
  store i32 %yleaf, ptr %zlp, align 4
  ; copy y.keys[8..14] -> z.keys[0..6]  (7 keys = 56 bytes)
  %ykeys = getelementptr inbounds nuw i8, ptr %y, i64 8
  %zkeys = getelementptr inbounds nuw i8, ptr %zn, i64 8
  %yksrc = getelementptr inbounds nuw i64, ptr %ykeys, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %zkeys, ptr %yksrc, i64 56, i1 false)
  %yvals = getelementptr inbounds nuw i8, ptr %y, i64 128
  %zvals = getelementptr inbounds nuw i8, ptr %zn, i64 128
  %yvsrc = getelementptr inbounds nuw i64, ptr %yvals, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %zvals, ptr %yvsrc, i64 56, i1 false)
  %isleaf = icmp ne i32 %yleaf, 0
  br i1 %isleaf, label %afterchild, label %copychild

copychild:
  ; copy y.child[8..15] -> z.child[0..7]  (8 children = 32 bytes)
  %ychild = getelementptr inbounds nuw i8, ptr %y, i64 248
  %zchild = getelementptr inbounds nuw i8, ptr %zn, i64 248
  %ycsrc = getelementptr inbounds nuw i32, ptr %ychild, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %zchild, ptr %ycsrc, i64 32, i1 false)
  br label %afterchild

afterchild:
  ; y.nkeys = 7
  store i32 7, ptr %y, align 4
  %medkp = getelementptr inbounds nuw i64, ptr %ykeys, i64 7
  %medk = load i64, ptr %medkp, align 8
  %medvp = getelementptr inbounds nuw i64, ptr %yvals, i64 7
  %medv = load i64, ptr %medvp, align 8
  %xnk32 = load i32, ptr %x, align 4
  %xnk = zext i32 %xnk32 to i64
  ; shift x.child[i+1..xnk] -> [i+2..xnk+1]  ((xnk-i) children)
  %i1 = add i64 %i, 1
  %i2 = add i64 %i, 2
  %chi1 = getelementptr inbounds nuw i32, ptr %xchild, i64 %i1
  %chi2 = getelementptr inbounds nuw i32, ptr %xchild, i64 %i2
  %ncsh = sub i64 %xnk, %i
  %ncshb = shl i64 %ncsh, 2
  call void @llvm.memmove.p0.p0.i64(ptr %chi2, ptr %chi1, i64 %ncshb, i1 false)
  %z32 = trunc i64 %z to i32
  store i32 %z32, ptr %chi1, align 4
  ; shift x.keys[i..xnk-1] -> [i+1..xnk]  ((xnk-i) keys)
  %xkeys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %xvals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %nkshb = shl i64 %ncsh, 3
  %ki = getelementptr inbounds nuw i64, ptr %xkeys, i64 %i
  %ki1 = getelementptr inbounds nuw i64, ptr %xkeys, i64 %i1
  call void @llvm.memmove.p0.p0.i64(ptr %ki1, ptr %ki, i64 %nkshb, i1 false)
  %vi = getelementptr inbounds nuw i64, ptr %xvals, i64 %i
  %vi1 = getelementptr inbounds nuw i64, ptr %xvals, i64 %i1
  call void @llvm.memmove.p0.p0.i64(ptr %vi1, ptr %vi, i64 %nkshb, i1 false)
  store i64 %medk, ptr %ki, align 8
  store i64 %medv, ptr %vi, align 8
  %xnk1 = add i64 %xnk, 1
  %xnk1.32 = trunc i64 %xnk1 to i32
  store i32 %xnk1.32, ptr %x, align 4
  ret i32 0
}

; Rightmost (max) key of subtree %idx; returns key, writes val to %vout.
define internal i64 @bt_submax(ptr %base, i64 %idx, ptr %vout) #4 {
entry:
  br label %loop

loop:
  %ix = phi i64 [ %idx, %entry ], [ %nx, %down ]
  %off = mul nuw i64 %ix, 320
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %lp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %leaf = load i32, ptr %lp, align 4
  %nk32 = load i32, ptr %node, align 4
  %nk = zext i32 %nk32 to i64
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %ret, label %down

down:
  %chb = getelementptr inbounds nuw i8, ptr %node, i64 248
  %chp = getelementptr inbounds nuw i32, ptr %chb, i64 %nk
  %ch = load i32, ptr %chp, align 4
  %nx = zext i32 %ch to i64
  br label %loop

ret:
  %last = sub i64 %nk, 1
  %kb = getelementptr inbounds nuw i8, ptr %node, i64 8
  %kp = getelementptr inbounds nuw i64, ptr %kb, i64 %last
  %k = load i64, ptr %kp, align 8
  %vb = getelementptr inbounds nuw i8, ptr %node, i64 128
  %vp = getelementptr inbounds nuw i64, ptr %vb, i64 %last
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %vout, align 8
  ret i64 %k
}

; Leftmost (min) key of subtree %idx; returns key, writes val to %vout.
define internal i64 @bt_submin(ptr %base, i64 %idx, ptr %vout) #4 {
entry:
  br label %loop

loop:
  %ix = phi i64 [ %idx, %entry ], [ %nx, %down ]
  %off = mul nuw i64 %ix, 320
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %lp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %ret, label %down

down:
  %chb = getelementptr inbounds nuw i8, ptr %node, i64 248
  %ch = load i32, ptr %chb, align 4
  %nx = zext i32 %ch to i64
  br label %loop

ret:
  %kb = getelementptr inbounds nuw i8, ptr %node, i64 8
  %k = load i64, ptr %kb, align 8
  %vb = getelementptr inbounds nuw i8, ptr %node, i64 128
  %v = load i64, ptr %vb, align 8
  store i64 %v, ptr %vout, align 8
  ret i64 %k
}

; Borrow one key from left sibling (child[i-1]) into child[i]. Caller ensured
; the left sibling has >= t keys.
define internal void @bt_borrow_left(ptr %base, ptr %x, i64 %i) #2 {
entry:
  %xchild = getelementptr inbounds nuw i8, ptr %x, i64 248
  %xkeys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %xvals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %im1 = sub i64 %i, 1
  %cip = getelementptr inbounds nuw i32, ptr %xchild, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %ci = zext i32 %ci32 to i64
  %coff = mul nuw i64 %ci, 320
  %c = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %lip = getelementptr inbounds nuw i32, ptr %xchild, i64 %im1
  %li32 = load i32, ptr %lip, align 4
  %li = zext i32 %li32 to i64
  %loff = mul nuw i64 %li, 320
  %l = getelementptr inbounds nuw i8, ptr %base, i64 %loff
  %cnk32 = load i32, ptr %c, align 4
  %cnk = zext i32 %cnk32 to i64
  %lnk32 = load i32, ptr %l, align 4
  %lnk = zext i32 %lnk32 to i64
  %clp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %cleaf = load i32, ptr %clp, align 4
  %isleaf = icmp ne i32 %cleaf, 0
  %ckeys = getelementptr inbounds nuw i8, ptr %c, i64 8
  %cvals = getelementptr inbounds nuw i8, ptr %c, i64 128
  %cchild = getelementptr inbounds nuw i8, ptr %c, i64 248
  ; shift c keys/vals right by 1 (cnk elements)
  %ck1 = getelementptr inbounds nuw i64, ptr %ckeys, i64 1
  %cnkb = shl i64 %cnk, 3
  call void @llvm.memmove.p0.p0.i64(ptr %ck1, ptr %ckeys, i64 %cnkb, i1 false)
  %cv1 = getelementptr inbounds nuw i64, ptr %cvals, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr %cv1, ptr %cvals, i64 %cnkb, i1 false)
  br i1 %isleaf, label %afterc, label %shiftc

shiftc:
  ; shift c children right by 1 (cnk+1 elements)
  %cch1 = getelementptr inbounds nuw i32, ptr %cchild, i64 1
  %ncp1 = add i64 %cnk, 1
  %ncp1b = shl i64 %ncp1, 2
  call void @llvm.memmove.p0.p0.i64(ptr %cch1, ptr %cchild, i64 %ncp1b, i1 false)
  br label %afterc

afterc:
  ; c.keys[0] = x.keys[i-1]; c.vals[0] = x.vals[i-1]
  %xkm1 = getelementptr inbounds nuw i64, ptr %xkeys, i64 %im1
  %xk = load i64, ptr %xkm1, align 8
  store i64 %xk, ptr %ckeys, align 8
  %xvm1 = getelementptr inbounds nuw i64, ptr %xvals, i64 %im1
  %xv = load i64, ptr %xvm1, align 8
  store i64 %xv, ptr %cvals, align 8
  br i1 %isleaf, label %afterlc, label %setlc

setlc:
  ; c.child[0] = l.child[lnk]
  %lchb = getelementptr inbounds nuw i8, ptr %l, i64 248
  %lclast = getelementptr inbounds nuw i32, ptr %lchb, i64 %lnk
  %lc = load i32, ptr %lclast, align 4
  store i32 %lc, ptr %cchild, align 4
  br label %afterlc

afterlc:
  ; x.keys[i-1] = l.keys[lnk-1]; x.vals[i-1] = l.vals[lnk-1]
  %lnk1 = sub i64 %lnk, 1
  %lkeys = getelementptr inbounds nuw i8, ptr %l, i64 8
  %lkp = getelementptr inbounds nuw i64, ptr %lkeys, i64 %lnk1
  %lk = load i64, ptr %lkp, align 8
  store i64 %lk, ptr %xkm1, align 8
  %lvals = getelementptr inbounds nuw i8, ptr %l, i64 128
  %lvp = getelementptr inbounds nuw i64, ptr %lvals, i64 %lnk1
  %lv = load i64, ptr %lvp, align 8
  store i64 %lv, ptr %xvm1, align 8
  %lnk1.32 = trunc i64 %lnk1 to i32
  store i32 %lnk1.32, ptr %l, align 4
  %cnk1 = add i64 %cnk, 1
  %cnk1.32 = trunc i64 %cnk1 to i32
  store i32 %cnk1.32, ptr %c, align 4
  ret void
}

; Borrow one key from right sibling (child[i+1]) into child[i].
define internal void @bt_borrow_right(ptr %base, ptr %x, i64 %i) #2 {
entry:
  %xchild = getelementptr inbounds nuw i8, ptr %x, i64 248
  %xkeys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %xvals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %ip1 = add i64 %i, 1
  %cip = getelementptr inbounds nuw i32, ptr %xchild, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %ci = zext i32 %ci32 to i64
  %coff = mul nuw i64 %ci, 320
  %c = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %rip = getelementptr inbounds nuw i32, ptr %xchild, i64 %ip1
  %ri32 = load i32, ptr %rip, align 4
  %ri = zext i32 %ri32 to i64
  %roff = mul nuw i64 %ri, 320
  %r = getelementptr inbounds nuw i8, ptr %base, i64 %roff
  %cnk32 = load i32, ptr %c, align 4
  %cnk = zext i32 %cnk32 to i64
  %rnk32 = load i32, ptr %r, align 4
  %rnk = zext i32 %rnk32 to i64
  %clp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %cleaf = load i32, ptr %clp, align 4
  %isleaf = icmp ne i32 %cleaf, 0
  %ckeys = getelementptr inbounds nuw i8, ptr %c, i64 8
  %cvals = getelementptr inbounds nuw i8, ptr %c, i64 128
  %cchild = getelementptr inbounds nuw i8, ptr %c, i64 248
  %rkeys = getelementptr inbounds nuw i8, ptr %r, i64 8
  %rvals = getelementptr inbounds nuw i8, ptr %r, i64 128
  %rchild = getelementptr inbounds nuw i8, ptr %r, i64 248
  ; c.keys[cnk] = x.keys[i]; c.vals[cnk] = x.vals[i]
  %xki = getelementptr inbounds nuw i64, ptr %xkeys, i64 %i
  %xk = load i64, ptr %xki, align 8
  %ckn = getelementptr inbounds nuw i64, ptr %ckeys, i64 %cnk
  store i64 %xk, ptr %ckn, align 8
  %xvi = getelementptr inbounds nuw i64, ptr %xvals, i64 %i
  %xv = load i64, ptr %xvi, align 8
  %cvn = getelementptr inbounds nuw i64, ptr %cvals, i64 %cnk
  store i64 %xv, ptr %cvn, align 8
  br i1 %isleaf, label %afterc, label %setrc

setrc:
  ; c.child[cnk+1] = r.child[0]
  %cnp1 = add i64 %cnk, 1
  %cchn = getelementptr inbounds nuw i32, ptr %cchild, i64 %cnp1
  %rc0 = load i32, ptr %rchild, align 4
  store i32 %rc0, ptr %cchn, align 4
  br label %afterc

afterc:
  ; x.keys[i] = r.keys[0]; x.vals[i] = r.vals[0]
  %rk0 = load i64, ptr %rkeys, align 8
  store i64 %rk0, ptr %xki, align 8
  %rv0 = load i64, ptr %rvals, align 8
  store i64 %rv0, ptr %xvi, align 8
  ; shift r keys/vals left by 1 (rnk-1 elements)
  %rnk1 = sub i64 %rnk, 1
  %rnk1b = shl i64 %rnk1, 3
  %rk1 = getelementptr inbounds nuw i64, ptr %rkeys, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr %rkeys, ptr %rk1, i64 %rnk1b, i1 false)
  %rv1 = getelementptr inbounds nuw i64, ptr %rvals, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr %rvals, ptr %rv1, i64 %rnk1b, i1 false)
  br i1 %isleaf, label %afterrc, label %shiftrc

shiftrc:
  ; shift r children left by 1 (rnk elements)
  %rnkb = shl i64 %rnk, 2
  %rch1 = getelementptr inbounds nuw i32, ptr %rchild, i64 1
  call void @llvm.memmove.p0.p0.i64(ptr %rchild, ptr %rch1, i64 %rnkb, i1 false)
  br label %afterrc

afterrc:
  %rnk1.32 = trunc i64 %rnk1 to i32
  store i32 %rnk1.32, ptr %r, align 4
  %cnk1 = add i64 %cnk, 1
  %cnk1.32 = trunc i64 %cnk1 to i32
  store i32 %cnk1.32, ptr %c, align 4
  ret void
}

; Merge child[i] + separator keys[i] + child[i+1] into child[i]; drop the
; separator and child[i+1] from x; free the right node.
define internal void @bt_merge(ptr %t, ptr %base, ptr %x, i64 %i) #2 {
entry:
  %xchild = getelementptr inbounds nuw i8, ptr %x, i64 248
  %xkeys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %xvals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %ip1 = add i64 %i, 1
  %ip2 = add i64 %i, 2
  %lip = getelementptr inbounds nuw i32, ptr %xchild, i64 %i
  %li32 = load i32, ptr %lip, align 4
  %li = zext i32 %li32 to i64
  %loff = mul nuw i64 %li, 320
  %l = getelementptr inbounds nuw i8, ptr %base, i64 %loff
  %rip = getelementptr inbounds nuw i32, ptr %xchild, i64 %ip1
  %ri32 = load i32, ptr %rip, align 4
  %ri = zext i32 %ri32 to i64
  %roff = mul nuw i64 %ri, 320
  %r = getelementptr inbounds nuw i8, ptr %base, i64 %roff
  %lnk32 = load i32, ptr %l, align 4
  %lnk = zext i32 %lnk32 to i64
  %rnk32 = load i32, ptr %r, align 4
  %rnk = zext i32 %rnk32 to i64
  %xnk32 = load i32, ptr %x, align 4
  %xnk = zext i32 %xnk32 to i64
  %llp = getelementptr inbounds nuw i8, ptr %l, i64 4
  %cleaf = load i32, ptr %llp, align 4
  %isleaf = icmp ne i32 %cleaf, 0
  %lkeys = getelementptr inbounds nuw i8, ptr %l, i64 8
  %lvals = getelementptr inbounds nuw i8, ptr %l, i64 128
  %lchild = getelementptr inbounds nuw i8, ptr %l, i64 248
  %rkeys = getelementptr inbounds nuw i8, ptr %r, i64 8
  %rvals = getelementptr inbounds nuw i8, ptr %r, i64 128
  %rchild = getelementptr inbounds nuw i8, ptr %r, i64 248
  ; l.keys[lnk] = x.keys[i]; l.vals[lnk] = x.vals[i]
  %xki = getelementptr inbounds nuw i64, ptr %xkeys, i64 %i
  %xk = load i64, ptr %xki, align 8
  %lkn = getelementptr inbounds nuw i64, ptr %lkeys, i64 %lnk
  store i64 %xk, ptr %lkn, align 8
  %xvi = getelementptr inbounds nuw i64, ptr %xvals, i64 %i
  %xv = load i64, ptr %xvi, align 8
  %lvn = getelementptr inbounds nuw i64, ptr %lvals, i64 %lnk
  store i64 %xv, ptr %lvn, align 8
  ; copy r keys/vals -> l.keys[lnk+1..]  (rnk elements)
  %lnk1 = add i64 %lnk, 1
  %rnkb = shl i64 %rnk, 3
  %ldstk = getelementptr inbounds nuw i64, ptr %lkeys, i64 %lnk1
  call void @llvm.memcpy.p0.p0.i64(ptr %ldstk, ptr %rkeys, i64 %rnkb, i1 false)
  %ldstv = getelementptr inbounds nuw i64, ptr %lvals, i64 %lnk1
  call void @llvm.memcpy.p0.p0.i64(ptr %ldstv, ptr %rvals, i64 %rnkb, i1 false)
  br i1 %isleaf, label %afterch, label %copych

copych:
  ; copy r.child[0..rnk] -> l.child[lnk+1..]  (rnk+1 elements)
  %ldstc = getelementptr inbounds nuw i32, ptr %lchild, i64 %lnk1
  %rnp1 = add i64 %rnk, 1
  %rnp1b = shl i64 %rnp1, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %ldstc, ptr %rchild, i64 %rnp1b, i1 false)
  br label %afterch

afterch:
  ; l.nkeys = lnk + 1 + rnk
  %newlnk = add i64 %lnk1, %rnk
  %newlnk.32 = trunc i64 %newlnk to i32
  store i32 %newlnk.32, ptr %l, align 4
  ; remove keys[i]/vals[i] from x: shift [i+1..xnk-1] left  ((xnk-1-i) elements)
  %xnk1 = sub i64 %xnk, 1
  %mvk = sub i64 %xnk1, %i
  %mvkb = shl i64 %mvk, 3
  %kdst = getelementptr inbounds nuw i64, ptr %xkeys, i64 %i
  %ksrc = getelementptr inbounds nuw i64, ptr %xkeys, i64 %ip1
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %mvkb, i1 false)
  %vdst = getelementptr inbounds nuw i64, ptr %xvals, i64 %i
  %vsrc = getelementptr inbounds nuw i64, ptr %xvals, i64 %ip1
  call void @llvm.memmove.p0.p0.i64(ptr %vdst, ptr %vsrc, i64 %mvkb, i1 false)
  ; remove child[i+1] from x: shift [i+2..xnk] left  ((xnk-1-i) elements)
  %cdst = getelementptr inbounds nuw i32, ptr %xchild, i64 %ip1
  %csrc = getelementptr inbounds nuw i32, ptr %xchild, i64 %ip2
  %mvcb = shl i64 %mvk, 2
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %mvcb, i1 false)
  %xnk1.32 = trunc i64 %xnk1 to i32
  store i32 %xnk1.32, ptr %x, align 4
  call void @bt_free_node(ptr %t, i64 %ri)
  ret void
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_btree_create() local_unnamed_addr #0 {
entry:
  %hdr = call ptr @malloc(i64 64)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %anodes, !prof !0

anodes:
  %nodes = call ptr @malloc(i64 2560)
  %nodes.null = icmp eq ptr %nodes, null
  br i1 %nodes.null, label %freehdr, label %init, !prof !0

freehdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %nodes, ptr %hdr, align 8
  %capp = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 8, ptr %capp, align 8
  %nnp = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 1, ptr %nnp, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 0, ptr %cntp, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i32 0, ptr %rootp, align 4
  %fhp = getelementptr inbounds nuw i8, ptr %hdr, i64 36
  store i32 -1, ptr %fhp, align 4
  ; root node 0: nkeys=0, leaf=1
  store i32 0, ptr %nodes, align 4
  %r0lp = getelementptr inbounds nuw i8, ptr %nodes, i64 4
  store i32 1, ptr %r0lp, align 4
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_btree_destroy(ptr %t) local_unnamed_addr #0 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %done, label %dofree, !prof !0

dofree:
  %nodes = load ptr, ptr %t, align 8
  call void @free(ptr %nodes)
  call void @free(ptr nonnull %t)
  br label %done

done:
  ret void
}

; ===========================================================================
; insert (proactive top-down split)
; ===========================================================================
define i32 @universe_ds_btree_insert(ptr %t, i64 %k0, i64 %val) local_unnamed_addr #0 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %err.null, label %rootchk, !prof !0

err.null:
  ret i32 1

rootchk:
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %base0 = load ptr, ptr %t, align 8
  %rootz = zext i32 %root32 to i64
  %roff = mul nuw i64 %rootz, 320
  %rootn = getelementptr inbounds nuw i8, ptr %base0, i64 %roff
  %rnk = load i32, ptr %rootn, align 4
  %full = icmp eq i32 %rnk, 15
  br i1 %full, label %newroot, label %descend0

newroot:
  %s = call i64 @bt_alloc_node(ptr %t)
  %sbad = icmp slt i64 %s, 0
  br i1 %sbad, label %err.oom, label %newroot2, !prof !0

err.oom:
  ret i32 2

newroot2:
  %base1 = load ptr, ptr %t, align 8
  %soff = mul nuw i64 %s, 320
  %sn = getelementptr inbounds nuw i8, ptr %base1, i64 %soff
  store i32 0, ptr %sn, align 4
  %slp = getelementptr inbounds nuw i8, ptr %sn, i64 4
  store i32 0, ptr %slp, align 4
  %scp = getelementptr inbounds nuw i8, ptr %sn, i64 248
  store i32 %root32, ptr %scp, align 4
  %s32 = trunc i64 %s to i32
  store i32 %s32, ptr %rootp, align 4
  %rc = call i32 @bt_split_child(ptr %t, i64 %s, i64 0)
  %rcbad = icmp ne i32 %rc, 0
  br i1 %rcbad, label %err.oom, label %descend_s, !prof !0

descend_s:
  br label %loop

descend0:
  br label %loop

loop:
  %xidx = phi i64 [ %s, %descend_s ], [ %rootz, %descend0 ], [ %nextx, %cont ]
  %base = load ptr, ptr %t, align 8
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %i = call i64 @bt_lb(ptr %x, i64 %k0)
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %vals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %ilt = icmp ult i64 %i, %nk
  br i1 %ilt, label %chkeq, label %notpresent

chkeq:
  %kip = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  %kiv = load i64, ptr %kip, align 8
  %eq = icmp eq i64 %kiv, %k0
  br i1 %eq, label %overwrite, label %notpresent

overwrite:
  %vip = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  store i64 %val, ptr %vip, align 8
  ret i32 0

notpresent:
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %insleaf, label %internal

insleaf:
  %tail = sub i64 %nk, %i
  %tailb = shl i64 %tail, 3
  %ip1 = add i64 %i, 1
  %dstk = getelementptr inbounds nuw i64, ptr %keys, i64 %ip1
  %srck = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  call void @llvm.memmove.p0.p0.i64(ptr %dstk, ptr %srck, i64 %tailb, i1 false)
  %dstv = getelementptr inbounds nuw i64, ptr %vals, i64 %ip1
  %srcv = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  call void @llvm.memmove.p0.p0.i64(ptr %dstv, ptr %srcv, i64 %tailb, i1 false)
  store i64 %k0, ptr %srck, align 8
  store i64 %val, ptr %srcv, align 8
  %nk1 = add i64 %nk, 1
  %nk1.32 = trunc i64 %nk1 to i32
  store i32 %nk1.32, ptr %x, align 4
  %cntp = getelementptr inbounds nuw i8, ptr %t, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  ret i32 0

internal:
  %childs = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cip = getelementptr inbounds nuw i32, ptr %childs, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %ci = zext i32 %ci32 to i64
  %coff = mul nuw i64 %ci, 320
  %c = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %cnk = load i32, ptr %c, align 4
  %cfull = icmp eq i32 %cnk, 15
  br i1 %cfull, label %splitc, label %godown

splitc:
  %rc2 = call i32 @bt_split_child(ptr %t, i64 %xidx, i64 %i)
  %rc2bad = icmp ne i32 %rc2, 0
  br i1 %rc2bad, label %err.oom, label %postsplit, !prof !0

postsplit:
  %base2 = load ptr, ptr %t, align 8
  %x2off = mul nuw i64 %xidx, 320
  %x2 = getelementptr inbounds nuw i8, ptr %base2, i64 %x2off
  %keys2 = getelementptr inbounds nuw i8, ptr %x2, i64 8
  %mkp = getelementptr inbounds nuw i64, ptr %keys2, i64 %i
  %mk = load i64, ptr %mkp, align 8
  %keq2 = icmp eq i64 %k0, %mk
  br i1 %keq2, label %ow2, label %choose

ow2:
  %vals2 = getelementptr inbounds nuw i8, ptr %x2, i64 128
  %vip2 = getelementptr inbounds nuw i64, ptr %vals2, i64 %i
  store i64 %val, ptr %vip2, align 8
  ret i32 0

choose:
  %kgt = icmp sgt i64 %k0, %mk
  %ipg = add i64 %i, 1
  %i2 = select i1 %kgt, i64 %ipg, i64 %i
  %childs2 = getelementptr inbounds nuw i8, ptr %x2, i64 248
  %cip2 = getelementptr inbounds nuw i32, ptr %childs2, i64 %i2
  %cc32 = load i32, ptr %cip2, align 4
  %cc = zext i32 %cc32 to i64
  br label %cont

godown:
  br label %cont

cont:
  %nextx = phi i64 [ %cc, %choose ], [ %ci, %godown ]
  br label %loop
}

; ===========================================================================
; delete (top-down refill)
; ===========================================================================
define i32 @universe_ds_btree_delete(ptr %t, i64 %karg) local_unnamed_addr #0 {
entry:
  %pvslot = alloca i64, align 8
  %svslot = alloca i64, align 8
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  br label %loop

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %nx_s, %cont_same ], [ %nx_k, %cont_key ]
  %key = phi i64 [ %karg, %setup ], [ %key, %cont_same ], [ %nkey_k, %cont_key ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %i = call i64 @bt_lb(ptr %x, i64 %key)
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %vals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %child = getelementptr inbounds nuw i8, ptr %x, i64 248
  %ilt = icmp ult i64 %i, %nk
  br i1 %ilt, label %chkeq, label %notfound_here

chkeq:
  %kip = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  %kiv = load i64, ptr %kip, align 8
  %eq = icmp eq i64 %kiv, %key
  br i1 %eq, label %found, label %notfound_here

found:
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %case1, label %case2

case1:
  %nk1 = sub i64 %nk, 1
  %mv = sub i64 %nk1, %i
  %mvb = shl i64 %mv, 3
  %ip1 = add i64 %i, 1
  %ksrc = getelementptr inbounds nuw i64, ptr %keys, i64 %ip1
  call void @llvm.memmove.p0.p0.i64(ptr %kip, ptr %ksrc, i64 %mvb, i1 false)
  %vdst = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %vsrc = getelementptr inbounds nuw i64, ptr %vals, i64 %ip1
  call void @llvm.memmove.p0.p0.i64(ptr %vdst, ptr %vsrc, i64 %mvb, i1 false)
  %nk1.32 = trunc i64 %nk1 to i32
  store i32 %nk1.32, ptr %x, align 4
  %cntp = getelementptr inbounds nuw i8, ptr %t, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cntd = sub i64 %cnt, 1
  store i64 %cntd, ptr %cntp, align 8
  ret i32 0

case2:
  %cip = getelementptr inbounds nuw i32, ptr %child, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %ci = zext i32 %ci32 to i64
  %yLoff = mul nuw i64 %ci, 320
  %yL = getelementptr inbounds nuw i8, ptr %base, i64 %yLoff
  %ynk32 = load i32, ptr %yL, align 4
  %yok = icmp uge i32 %ynk32, 8
  br i1 %yok, label %usepred, label %trysucc

usepred:
  %pk = call i64 @bt_submax(ptr %base, i64 %ci, ptr %pvslot)
  %pv = load i64, ptr %pvslot, align 8
  store i64 %pk, ptr %kip, align 8
  %vip = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  store i64 %pv, ptr %vip, align 8
  br label %cont_key_pred

trysucc:
  %i1s = add i64 %i, 1
  %rips = getelementptr inbounds nuw i32, ptr %child, i64 %i1s
  %ri32s = load i32, ptr %rips, align 4
  %ris = zext i32 %ri32s to i64
  %yRoff = mul nuw i64 %ris, 320
  %yR = getelementptr inbounds nuw i8, ptr %base, i64 %yRoff
  %rnk32 = load i32, ptr %yR, align 4
  %rok = icmp uge i32 %rnk32, 8
  br i1 %rok, label %usesucc, label %domerge2

usesucc:
  %sk = call i64 @bt_submin(ptr %base, i64 %ris, ptr %svslot)
  %sv = load i64, ptr %svslot, align 8
  store i64 %sk, ptr %kip, align 8
  %vip2 = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  store i64 %sv, ptr %vip2, align 8
  br label %cont_key_succ

domerge2:
  call void @bt_merge(ptr %t, ptr %base, ptr %x, i64 %i)
  %xnk_a2 = load i32, ptr %x, align 4
  %curroot2 = load i32, ptr %rootp, align 4
  %isroot2 = icmp eq i64 %xidx, %root0
  %rootmatch2 = icmp eq i32 %curroot2, %root32
  %isr2 = and i1 %isroot2, %rootmatch2
  %empty2 = icmp eq i32 %xnk_a2, 0
  %shrink2 = and i1 %isr2, %empty2
  br i1 %shrink2, label %do_shrink2, label %cont_same

do_shrink2:
  %ci2.32 = trunc i64 %ci to i32
  store i32 %ci2.32, ptr %rootp, align 4
  call void @bt_free_node(ptr %t, i64 %xidx)
  br label %cont_same

notfound_here:
  %lp2 = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf2 = load i32, ptr %lp2, align 4
  %isleaf2 = icmp ne i32 %leaf2, 0
  br i1 %isleaf2, label %ret5, label %descend

ret5:
  ret i32 5

descend:
  %cidp = getelementptr inbounds nuw i32, ptr %child, i64 %i
  %cid32 = load i32, ptr %cidp, align 4
  %cid = zext i32 %cid32 to i64
  %cdoff = mul nuw i64 %cid, 320
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %cdoff
  %cnk3 = load i32, ptr %cnode, align 4
  %cbig = icmp uge i32 %cnk3, 8
  br i1 %cbig, label %cont_same, label %fixchild

fixchild:
  %hasleft = icmp ugt i64 %i, 0
  br i1 %hasleft, label %chkleft, label %try_right

chkleft:
  %im1 = sub i64 %i, 1
  %lidp = getelementptr inbounds nuw i32, ptr %child, i64 %im1
  %lid32 = load i32, ptr %lidp, align 4
  %lid = zext i32 %lid32 to i64
  %lnoff = mul nuw i64 %lid, 320
  %lnode = getelementptr inbounds nuw i8, ptr %base, i64 %lnoff
  %lnk3 = load i32, ptr %lnode, align 4
  %lspare = icmp uge i32 %lnk3, 8
  br i1 %lspare, label %do_bl, label %try_right

do_bl:
  call void @bt_borrow_left(ptr %base, ptr %x, i64 %i)
  br label %cont_same

try_right:
  %hasright = icmp ult i64 %i, %nk
  br i1 %hasright, label %chkright, label %do_merge3

chkright:
  %ip1r = add i64 %i, 1
  %ridp = getelementptr inbounds nuw i32, ptr %child, i64 %ip1r
  %rid32 = load i32, ptr %ridp, align 4
  %rid = zext i32 %rid32 to i64
  %rnoff = mul nuw i64 %rid, 320
  %rnode = getelementptr inbounds nuw i8, ptr %base, i64 %rnoff
  %rnk3 = load i32, ptr %rnode, align 4
  %rspare = icmp uge i32 %rnk3, 8
  br i1 %rspare, label %do_br, label %do_merge3

do_br:
  call void @bt_borrow_right(ptr %base, ptr %x, i64 %i)
  br label %cont_same

do_merge3:
  %imerge = icmp ult i64 %i, %nk
  br i1 %imerge, label %merge_right3, label %merge_left3

merge_right3:
  call void @bt_merge(ptr %t, ptr %base, ptr %x, i64 %i)
  br label %shrink3

merge_left3:
  %im1b = sub i64 %i, 1
  %lidp2 = getelementptr inbounds nuw i32, ptr %child, i64 %im1b
  %lid2.32 = load i32, ptr %lidp2, align 4
  %lid2 = zext i32 %lid2.32 to i64
  call void @bt_merge(ptr %t, ptr %base, ptr %x, i64 %im1b)
  br label %shrink3

shrink3:
  %mergedidx = phi i64 [ %cid, %merge_right3 ], [ %lid2, %merge_left3 ]
  %xnk_a3 = load i32, ptr %x, align 4
  %curroot3 = load i32, ptr %rootp, align 4
  %isroot3 = icmp eq i64 %xidx, %root0
  %rootmatch3 = icmp eq i32 %curroot3, %root32
  %isr3 = and i1 %isroot3, %rootmatch3
  %empty3 = icmp eq i32 %xnk_a3, 0
  %shrink = and i1 %isr3, %empty3
  br i1 %shrink, label %do_shrink3, label %cont_same

do_shrink3:
  %mi32 = trunc i64 %mergedidx to i32
  store i32 %mi32, ptr %rootp, align 4
  call void @bt_free_node(ptr %t, i64 %xidx)
  br label %cont_same

cont_key_pred:
  br label %cont_key

cont_key_succ:
  br label %cont_key

cont_key:
  %nx_k = phi i64 [ %ci, %cont_key_pred ], [ %ris, %cont_key_succ ]
  %nkey_k = phi i64 [ %pk, %cont_key_pred ], [ %sk, %cont_key_succ ]
  br label %loop

cont_same:
  %nx_s = phi i64 [ %cid, %descend ], [ %cid, %do_bl ], [ %cid, %do_br ], [ %ci, %domerge2 ], [ %ci, %do_shrink2 ], [ %mergedidx, %shrink3 ], [ %mergedidx, %do_shrink3 ]
  br label %loop
}

; ===========================================================================
; find / contains
; ===========================================================================
define i32 @universe_ds_btree_find(ptr %t, i64 %key, ptr %out) local_unnamed_addr #1 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  br label %loop

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %nx, %godown ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %i = call i64 @bt_lb(ptr %x, i64 %key)
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %ilt = icmp ult i64 %i, %nk
  br i1 %ilt, label %chkeq, label %notpresent

chkeq:
  %kip = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  %kiv = load i64, ptr %kip, align 8
  %eq = icmp eq i64 %kiv, %key
  br i1 %eq, label %hit, label %notpresent

hit:
  %outn = icmp ne ptr %out, null
  br i1 %outn, label %store, label %ret0

store:
  %vals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %vip = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %v = load i64, ptr %vip, align 8
  store i64 %v, ptr %out, align 8
  br label %ret0

ret0:
  ret i32 0

notpresent:
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %miss, label %godown

miss:
  ret i32 5

godown:
  %childs = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cip = getelementptr inbounds nuw i32, ptr %childs, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %nx = zext i32 %ci32 to i64
  br label %loop
}

define i32 @universe_ds_btree_contains(ptr %t, i64 %key) local_unnamed_addr #1 {
entry:
  %r = call i32 @universe_ds_btree_find(ptr %t, i64 %key, ptr null)
  %f = icmp eq i32 %r, 0
  %z = zext i1 %f to i32
  ret i32 %z
}

; ===========================================================================
; min / max
; ===========================================================================
define i32 @universe_ds_btree_min(ptr %t, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %tn = icmp eq ptr %t, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %tn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  %roff = mul nuw i64 %root0, 320
  %rootn = getelementptr inbounds nuw i8, ptr %base, i64 %roff
  %rnk = load i32, ptr %rootn, align 4
  %empty = icmp eq i32 %rnk, 0
  br i1 %empty, label %isempty, label %loop, !prof !0

isempty:
  ret i32 4

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %nx, %godown ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %atleaf, label %godown

atleaf:
  %kp = getelementptr inbounds nuw i8, ptr %x, i64 8
  %k = load i64, ptr %kp, align 8
  store i64 %k, ptr %ok, align 8
  %vp = getelementptr inbounds nuw i8, ptr %x, i64 128
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  ret i32 0

godown:
  %cp = getelementptr inbounds nuw i8, ptr %x, i64 248
  %c = load i32, ptr %cp, align 4
  %nx = zext i32 %c to i64
  br label %loop
}

define i32 @universe_ds_btree_max(ptr %t, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %tn = icmp eq ptr %t, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %tn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  %roff = mul nuw i64 %root0, 320
  %rootn = getelementptr inbounds nuw i8, ptr %base, i64 %roff
  %rnk = load i32, ptr %rootn, align 4
  %empty = icmp eq i32 %rnk, 0
  br i1 %empty, label %isempty, label %loop, !prof !0

isempty:
  ret i32 4

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %nx, %godown ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %atleaf, label %godown

atleaf:
  %last = sub i64 %nk, 1
  %kb = getelementptr inbounds nuw i8, ptr %x, i64 8
  %kp = getelementptr inbounds nuw i64, ptr %kb, i64 %last
  %k = load i64, ptr %kp, align 8
  store i64 %k, ptr %ok, align 8
  %vb = getelementptr inbounds nuw i8, ptr %x, i64 128
  %vp = getelementptr inbounds nuw i64, ptr %vb, i64 %last
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  ret i32 0

godown:
  %cb = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cp = getelementptr inbounds nuw i32, ptr %cb, i64 %nk
  %c = load i32, ptr %cp, align 4
  %nx = zext i32 %c to i64
  br label %loop
}

; ===========================================================================
; floor (largest key <= query) / ceiling (smallest key >= query)
; ===========================================================================
define i32 @universe_ds_btree_floor(ptr %t, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %tn = icmp eq ptr %t, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %tn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  br label %loop

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %idx2, %godown ]
  %bf = phi i1 [ false, %setup ], [ %bf2, %godown ]
  %bk = phi i64 [ 0, %setup ], [ %bk2, %godown ]
  %bv = phi i64 [ 0, %setup ], [ %bv2, %godown ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %vals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %i = call i64 @bt_lb(ptr %x, i64 %key)
  %ilt = icmp ult i64 %i, %nk
  br i1 %ilt, label %chkeq, label %noexact

chkeq:
  %kip = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  %kiv = load i64, ptr %kip, align 8
  %eq = icmp eq i64 %kiv, %key
  br i1 %eq, label %exact, label %noexact

exact:
  store i64 %kiv, ptr %ok, align 8
  %vip = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %ve = load i64, ptr %vip, align 8
  store i64 %ve, ptr %ov, align 8
  ret i32 0

noexact:
  %hascand = icmp ugt i64 %i, 0
  br i1 %hascand, label %setcand, label %aftercand

setcand:
  %im1 = sub i64 %i, 1
  %ckp = getelementptr inbounds nuw i64, ptr %keys, i64 %im1
  %ck = load i64, ptr %ckp, align 8
  %cvp = getelementptr inbounds nuw i64, ptr %vals, i64 %im1
  %cv = load i64, ptr %cvp, align 8
  br label %aftercand

aftercand:
  %bf2 = phi i1 [ true, %setcand ], [ %bf, %noexact ]
  %bk2 = phi i64 [ %ck, %setcand ], [ %bk, %noexact ]
  %bv2 = phi i64 [ %cv, %setcand ], [ %bv, %noexact ]
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %finish, label %godown

godown:
  %childs = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cip = getelementptr inbounds nuw i32, ptr %childs, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %idx2 = zext i32 %ci32 to i64
  br label %loop

finish:
  br i1 %bf2, label %haveans, label %noans

haveans:
  store i64 %bk2, ptr %ok, align 8
  store i64 %bv2, ptr %ov, align 8
  ret i32 0

noans:
  ret i32 5
}

define i32 @universe_ds_btree_ceiling(ptr %t, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #1 {
entry:
  %tn = icmp eq ptr %t, null
  %okn = icmp eq ptr %ok, null
  %ovn = icmp eq ptr %ov, null
  %b0 = or i1 %tn, %okn
  %bad = or i1 %b0, %ovn
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  br label %loop

loop:
  %xidx = phi i64 [ %root0, %setup ], [ %idx2, %godown ]
  %bf = phi i1 [ false, %setup ], [ %bf2, %godown ]
  %bk = phi i64 [ 0, %setup ], [ %bk2, %godown ]
  %bv = phi i64 [ 0, %setup ], [ %bv2, %godown ]
  %xoff = mul nuw i64 %xidx, 320
  %x = getelementptr inbounds nuw i8, ptr %base, i64 %xoff
  %nk32 = load i32, ptr %x, align 4
  %nk = zext i32 %nk32 to i64
  %keys = getelementptr inbounds nuw i8, ptr %x, i64 8
  %vals = getelementptr inbounds nuw i8, ptr %x, i64 128
  %i = call i64 @bt_lb(ptr %x, i64 %key)
  %ilt = icmp ult i64 %i, %nk
  br i1 %ilt, label %cand, label %aftercand

cand:
  %kip = getelementptr inbounds nuw i64, ptr %keys, i64 %i
  %kiv = load i64, ptr %kip, align 8
  %eq = icmp eq i64 %kiv, %key
  br i1 %eq, label %exact, label %setbest

exact:
  store i64 %kiv, ptr %ok, align 8
  %vip = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %ve = load i64, ptr %vip, align 8
  store i64 %ve, ptr %ov, align 8
  ret i32 0

setbest:
  %cvp = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %cv = load i64, ptr %cvp, align 8
  br label %aftercand

aftercand:
  %bf2 = phi i1 [ true, %setbest ], [ %bf, %loop ]
  %bk2 = phi i64 [ %kiv, %setbest ], [ %bk, %loop ]
  %bv2 = phi i64 [ %cv, %setbest ], [ %bv, %loop ]
  %lp = getelementptr inbounds nuw i8, ptr %x, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  br i1 %isleaf, label %finish, label %godown

godown:
  %childs = getelementptr inbounds nuw i8, ptr %x, i64 248
  %cip = getelementptr inbounds nuw i32, ptr %childs, i64 %i
  %ci32 = load i32, ptr %cip, align 4
  %idx2 = zext i32 %ci32 to i64
  br label %loop

finish:
  br i1 %bf2, label %haveans, label %noans

haveans:
  store i64 %bk2, ptr %ok, align 8
  store i64 %bv2, ptr %ov, align 8
  ret i32 0

noans:
  ret i32 5
}

; ===========================================================================
; range [lo, hi] inclusive: in-order collect; returns count of in-range keys,
; writes up to %max into ok/ov (either may be null for count-only).
; ===========================================================================
define internal void @bt_range_rec(ptr %base, i64 %idx, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max, ptr %cntp) #5 {
entry:
  %off = mul nuw i64 %idx, 320
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %nk32 = load i32, ptr %node, align 4
  %nk = zext i32 %nk32 to i64
  %lp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %leaf = load i32, ptr %lp, align 4
  %isleaf = icmp ne i32 %leaf, 0
  %keys = getelementptr inbounds nuw i8, ptr %node, i64 8
  %vals = getelementptr inbounds nuw i8, ptr %node, i64 128
  %child = getelementptr inbounds nuw i8, ptr %node, i64 248
  %i0 = call i64 @bt_lb(ptr %node, i64 %lo)
  br label %loop

loop:
  %j = phi i64 [ %i0, %entry ], [ %jn, %aftv ]
  br i1 %isleaf, label %keycheck, label %descend

descend:
  %cjp = getelementptr inbounds nuw i32, ptr %child, i64 %j
  %cj32 = load i32, ptr %cjp, align 4
  %cj = zext i32 %cj32 to i64
  call void @bt_range_rec(ptr %base, i64 %cj, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max, ptr %cntp)
  br label %keycheck

keycheck:
  %atend = icmp uge i64 %j, %nk
  br i1 %atend, label %ret, label %havekey

havekey:
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %j
  %kv = load i64, ptr %kp, align 8
  %over = icmp sgt i64 %kv, %hi
  br i1 %over, label %ret, label %inrange

inrange:
  %cnt = load i64, ptr %cntp, align 8
  %room = icmp ult i64 %cnt, %max
  %okok = icmp ne ptr %ok, null
  %wk = and i1 %okok, %room
  br i1 %wk, label %wkb, label %aftk

wkb:
  %okp = getelementptr inbounds nuw i64, ptr %ok, i64 %cnt
  store i64 %kv, ptr %okp, align 8
  br label %aftk

aftk:
  %ovok = icmp ne ptr %ov, null
  %wv = and i1 %ovok, %room
  br i1 %wv, label %wvb, label %aftv

wvb:
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %j
  %vv = load i64, ptr %vp, align 8
  %ovp = getelementptr inbounds nuw i64, ptr %ov, i64 %cnt
  store i64 %vv, ptr %ovp, align 8
  br label %aftv

aftv:
  %cntn = add i64 %cnt, 1
  store i64 %cntn, ptr %cntp, align 8
  %jn = add i64 %j, 1
  br label %loop

ret:
  ret void
}

define i64 @universe_ds_btree_range(ptr %t, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max) local_unnamed_addr #5 {
entry:
  %cntslot = alloca i64, align 8
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %ret0, label %chkorder, !prof !0

ret0:
  ret i64 0

chkorder:
  %bad = icmp sgt i64 %lo, %hi
  br i1 %bad, label %ret0, label %go, !prof !0

go:
  store i64 0, ptr %cntslot, align 8
  %base = load ptr, ptr %t, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %root32 = load i32, ptr %rootp, align 4
  %root0 = zext i32 %root32 to i64
  call void @bt_range_rec(ptr %base, i64 %root0, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max, ptr %cntslot)
  %r = load i64, ptr %cntslot, align 8
  ret i64 %r
}

; ===========================================================================
; count / size
; ===========================================================================
define i64 @universe_ds_btree_count(ptr %t) local_unnamed_addr #6 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %z, label %go, !prof !0

z:
  ret i64 0

go:
  %cntp = getelementptr inbounds nuw i8, ptr %t, i64 24
  %cnt = load i64, ptr %cntp, align 8
  ret i64 %cnt
}

define i64 @universe_ds_btree_size(ptr %t) local_unnamed_addr #6 {
entry:
  %r = call i64 @universe_ds_btree_count(ptr %t)
  ret i64 %r
}

attributes #0 = { nounwind willreturn }
attributes #1 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #5 = { nounwind }
attributes #6 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
