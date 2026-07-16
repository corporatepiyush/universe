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

; universe_ds_skiplist — INDEXABLE probabilistic ordered map, i64 key -> i64
; value, single thread.
;
; ============================================================================
; DESIGN (from first principles)
; ----------------------------------------------------------------------------
;   * A skip list is the ordered-map class chosen here because it delivers
;     O(log n) search/insert/delete with NO rotations or rebalancing passes
;     (unlike a balanced BST): every mutation touches only the O(log n) nodes on
;     the search path plus their forward links. That makes it cheaper to author
;     correctly in IR than a red-black/AVL tree and gives the same asymptotics.
;
;   * INDEX-LINKED, CHUNKED POOL (house idiom): ALL nodes live in ONE flat,
;     growable array; a node references its forward neighbours by i32 INDEX, not
;     pointer. Half the link footprint of 8-byte pointers, one allocation, great
;     locality, and the backing array can double via realloc without rewriting a
;     single interior link (indices are position-independent). Node 0 is the
;     head sentinel (level = MAX_LEVEL, key = -inf conceptually).
;
;   * FREELIST + BUMP CURSOR (wilderness): fresh nodes come from a bump cursor
;     (nnodes); deleted nodes are pushed on an intrusive i32 freelist threaded
;     through the node's `level` slot (reused while free). alloc pops the
;     freelist first, else bumps, else grows 2x. Create is O(1).
;
;   * INDEXABLE via per-link SPANS: every forward link carries the number of
;     level-0 nodes it skips (the destination inclusive). rank(key) sums the
;     spans traversed during the descent; select(i) descends while the running
;     position + span <= target. Both O(log n). Spans are i32 (≤ ~4e9 elems).
;
;   * LEVEL by coin flip, p = 1/4 (2 zero bits continue): shallower towers than
;     p=1/2, fewer forward slots touched per search step. MAX_LEVEL = 16 covers
;     4^16 ≈ 4.3e9 elements. The RNG is an internal, self-contained MMIX LCG
;     seeded to a FIXED constant in create() so the level sequence — and thus
;     the whole structure — is DETERMINISTIC across runs (the library must not
;     depend on the test harness; the constants match ut_rand for parity).
;
;   * SIGNED key order (icmp slt) so the full i64 range orders naturally for
;     min/max/floor/ceiling/range.
;
;   * Fixed-width forward/span arrays (MAX_LEVEL wide) per node keep a constant
;     node stride so the flat pool stays a simple indexed array; the unused
;     upper slots of a low tower are never read (searches only follow links at
;     levels below a node's own level). Memory for those slots is the price of
;     a pointer-free, realloc-safe layout.
;
; ----------------------------------------------------------------------------
; Node record (stride 152 B):
;   key    i64        @0     (head sentinel: unused / -inf)
;   value  i64        @8
;   level  i32        @16    tower height (while on freelist: next-free index)
;   pad    i32        @20
;   forward i32[16]   @24    [24,88)   forward[l] = next node at level l (-1 nil)
;   span    i32[16]   @88    [88,152)  span[l]   = level-0 nodes link l skips
; Header (48 B):
;   nodes  ptr        @0     flat node array base
;   cap    i64        @8     node capacity (slots)
;   nnodes i64        @16    bump high-water
;   count  i64        @24    live element count (excludes head sentinel)
;   level  i32        @32    current max tower level in use (1..16)
;   free   i32        @36    freelist head (-1 = empty)
;   rng    i64        @40    LCG state for level coin flips
; ----------------------------------------------------------------------------
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY, 5 NOT_FOUND,
;      7 INVALID_INDEX):
;   ptr universe_ds_skiplist_create()
;   void universe_ds_skiplist_destroy(ptr)
;   i32 universe_ds_skiplist_put(ptr, i64 key, i64 val)     ; overwrites on dup
;   i32 universe_ds_skiplist_get(ptr, i64 key, ptr outval)  ; outval may be null
;   i32 universe_ds_skiplist_contains(ptr, i64 key)         ; 1 present / 0 absent
;   i32 universe_ds_skiplist_delete(ptr, i64 key)
;   i32 universe_ds_skiplist_min(ptr, ptr outkey, ptr outval)
;   i32 universe_ds_skiplist_max(ptr, ptr outkey, ptr outval)
;   i32 universe_ds_skiplist_floor(ptr, i64 key, ptr ok, ptr ov)   ; largest <= key
;   i32 universe_ds_skiplist_ceiling(ptr, i64 key, ptr ok, ptr ov) ; smallest >= key
;   i32 universe_ds_skiplist_rank(ptr, i64 key, ptr out_rank)      ; 0-based index
;   i32 universe_ds_skiplist_select(ptr, i64 idx, ptr ok, ptr ov)  ; key/val at idx
;   i64 universe_ds_skiplist_range(ptr, i64 lo, i64 hi, ptr ok, ptr ov, i64 max)
;   i64 universe_ds_skiplist_size(ptr)
; ============================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

; ===========================================================================
; internal helpers
; ===========================================================================

; Advance the header RNG and pick a tower level in [1,16], p=1/4 per extra
; level. Mutates rng state at h+40.
define internal i64 @sl_randlevel(ptr %h) #4 {
entry:
  %rngp = getelementptr inbounds nuw i8, ptr %h, i64 40
  %s0 = load i64, ptr %rngp, align 8
  br label %loop

loop:
  %lvl = phi i64 [ 1, %entry ], [ %lvl1, %cont ]
  %s = phi i64 [ %s0, %entry ], [ %n, %cont ]
  %m = mul i64 %s, 6364136223846793005
  %n = add i64 %m, 1442695040888963407
  %hi = lshr i64 %n, 8
  %bits = and i64 %hi, 3
  %zero = icmp eq i64 %bits, 0
  %atmax = icmp uge i64 %lvl, 16
  %notmax = xor i1 %atmax, true
  %docont = and i1 %zero, %notmax
  br i1 %docont, label %cont, label %done

cont:
  %lvl1 = add nuw nsw i64 %lvl, 1
  br label %loop

done:
  store i64 %n, ptr %rngp, align 8
  ret i64 %lvl
}

; Push node %idx onto the intrusive freelist (next stored at node word @16).
define internal void @sl_free_node(ptr %h, i64 %idx) #4 {
entry:
  %base = load ptr, ptr %h, align 8
  %off = mul nuw i64 %idx, 152
  %node = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %lvlp = getelementptr inbounds nuw i8, ptr %node, i64 16
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 36
  %fh = load i32, ptr %fhp, align 4
  store i32 %fh, ptr %lvlp, align 4
  %idx32 = trunc i64 %idx to i32
  store i32 %idx32, ptr %fhp, align 4
  ret void
}

; Allocate a node slot, growing the array 2x if needed. Returns index or -1.
define internal i64 @sl_alloc_node(ptr %h) #5 {
entry:
  %fhp = getelementptr inbounds nuw i8, ptr %h, i64 36
  %fh = load i32, ptr %fhp, align 4
  %hasfree = icmp sge i32 %fh, 0
  br i1 %hasfree, label %pop, label %bump

pop:
  %base = load ptr, ptr %h, align 8
  %fhi = zext i32 %fh to i64
  %poff = mul nuw i64 %fhi, 152
  %pnode = getelementptr inbounds nuw i8, ptr %base, i64 %poff
  %plvlp = getelementptr inbounds nuw i8, ptr %pnode, i64 16
  %next = load i32, ptr %plvlp, align 4
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
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap2, i64 152)
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

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_skiplist_create() local_unnamed_addr #1 {
entry:
  %hdr = call ptr @malloc(i64 48)
  %hnull = icmp eq ptr %hdr, null
  br i1 %hnull, label %fail, label %anodes, !prof !0

anodes:
  ; 16 initial node slots * 152 = 2432 bytes
  %nodes = call ptr @malloc(i64 2432)
  %nnull = icmp eq ptr %nodes, null
  br i1 %nnull, label %freehdr, label %init, !prof !0

freehdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  store ptr %nodes, ptr %hdr, align 8
  %capp = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 16, ptr %capp, align 8
  %nnp = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 1, ptr %nnp, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 0, ptr %cntp, align 8
  %levp = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i32 1, ptr %levp, align 4
  %fhp = getelementptr inbounds nuw i8, ptr %hdr, i64 36
  store i32 -1, ptr %fhp, align 4
  %rngp = getelementptr inbounds nuw i8, ptr %hdr, i64 40
  store i64 88172645463325252, ptr %rngp, align 8
  ; head sentinel node 0: key=0,val=0, level=16, forward[*]=-1, span[*]=0
  store i64 0, ptr %nodes, align 8
  %hvp = getelementptr inbounds nuw i8, ptr %nodes, i64 8
  store i64 0, ptr %hvp, align 8
  %hlp = getelementptr inbounds nuw i8, ptr %nodes, i64 16
  store i32 16, ptr %hlp, align 4
  %hfwd = getelementptr inbounds nuw i8, ptr %nodes, i64 24
  call void @llvm.memset.p0.i64(ptr %hfwd, i8 -1, i64 64, i1 false)
  %hspan = getelementptr inbounds nuw i8, ptr %nodes, i64 88
  call void @llvm.memset.p0.i64(ptr %hspan, i8 0, i64 64, i1 false)
  ret ptr %hdr

fail:
  ret ptr null
}

define void @universe_ds_skiplist_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %h, null
  br i1 %isnull, label %done, label %dofree, !prof !0

dofree:
  %nodes = load ptr, ptr %h, align 8
  call void @free(ptr %nodes)
  call void @free(ptr nonnull %h)
  br label %done

done:
  ret void
}

; ===========================================================================
; get / contains
; ===========================================================================
define i32 @universe_ds_skiplist_get(ptr %h, i64 %key, ptr %out) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %key
  br i1 %lt, label %adv, label %stop

adv:
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %examine, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

examine:
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %cand = load i32, ptr %f0, align 4
  %cnil = icmp eq i32 %cand, -1
  br i1 %cnil, label %notfound, label %chkeq

chkeq:
  %cand64 = zext i32 %cand to i64
  %coff = mul nuw i64 %cand64, 152
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %ckey = load i64, ptr %cnode, align 8
  %eq = icmp eq i64 %ckey, %key
  br i1 %eq, label %found, label %notfound

found:
  %outnull = icmp eq ptr %out, null
  br i1 %outnull, label %ret0, label %storev

storev:
  %vp = getelementptr inbounds nuw i8, ptr %cnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %out, align 8
  br label %ret0

ret0:
  ret i32 0

notfound:
  ret i32 5
}

define i32 @universe_ds_skiplist_contains(ptr %h, i64 %key) local_unnamed_addr #1 {
entry:
  %r = call i32 @universe_ds_skiplist_get(ptr %h, i64 %key, ptr null)
  %f = icmp eq i32 %r, 0
  %z = zext i1 %f to i32
  ret i32 %z
}

; ===========================================================================
; put (insert / overwrite) with span maintenance
; ===========================================================================
define i32 @universe_ds_skiplist_put(ptr %h, i64 %key, i64 %val) local_unnamed_addr #1 {
entry:
  %update = alloca [16 x i64], align 8
  %rankv = alloca [16 x i64], align 8
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %oldlev = zext i32 %lev32 to i64
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  %rank = phi i64 [ 0, %setup ], [ %rc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %rc = phi i64 [ %rank, %lvl.head ], [ %radv, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %key
  br i1 %lt, label %adv, label %stop

adv:
  %sbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 88
  %sp = getelementptr inbounds nuw i32, ptr %sbase, i64 %i
  %s32 = load i32, ptr %sp, align 4
  %s64 = zext i32 %s32 to i64
  %radv = add i64 %rc, %s64
  br label %inner

stop:
  %up = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %i
  store i64 %xc, ptr %up, align 8
  %rp = getelementptr inbounds nuw [16 x i64], ptr %rankv, i64 0, i64 %i
  store i64 %rc, ptr %rp, align 8
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %afterdescend, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

afterdescend:
  ; %xc = update[0] (predecessor at level 0), %rc = rank0 (its position)
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %cand = load i32, ptr %f0, align 4
  %cnil = icmp eq i32 %cand, -1
  br i1 %cnil, label %doinsert, label %chkeq

chkeq:
  %cand64 = zext i32 %cand to i64
  %coff = mul nuw i64 %cand64, 152
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %ckey = load i64, ptr %cnode, align 8
  %eq = icmp eq i64 %ckey, %key
  br i1 %eq, label %overwrite, label %doinsert

overwrite:
  %ovp = getelementptr inbounds nuw i8, ptr %cnode, i64 8
  store i64 %val, ptr %ovp, align 8
  ret i32 0

doinsert:
  %nl = call i64 @sl_randlevel(ptr %h)
  %z = call i64 @sl_alloc_node(ptr %h)
  %zbad = icmp slt i64 %z, 0
  br i1 %zbad, label %err.oom, label %fill, !prof !0

err.oom:
  ret i32 2

fill:
  %base2 = load ptr, ptr %h, align 8
  %zoff = mul nuw i64 %z, 152
  %znode = getelementptr inbounds nuw i8, ptr %base2, i64 %zoff
  store i64 %key, ptr %znode, align 8
  %zvp = getelementptr inbounds nuw i8, ptr %znode, i64 8
  store i64 %val, ptr %zvp, align 8
  %zlp = getelementptr inbounds nuw i8, ptr %znode, i64 16
  %nl32 = trunc i64 %nl to i32
  store i32 %nl32, ptr %zlp, align 4
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %countv = load i64, ptr %cntp, align 8
  %count32 = trunc i64 %countv to i32
  %rank0 = load i64, ptr %rankv, align 8
  br label %grow.head

grow.head:
  %gi = phi i64 [ %oldlev, %fill ], [ %gi1, %grow.body ]
  %gcond = icmp ult i64 %gi, %nl
  br i1 %gcond, label %grow.body, label %grow.done

grow.body:
  %gup = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %gi
  store i64 0, ptr %gup, align 8
  %grp = getelementptr inbounds nuw [16 x i64], ptr %rankv, i64 0, i64 %gi
  store i64 0, ptr %grp, align 8
  ; head.span[gi] = count  (head node = base2 + 0)
  %hspan = getelementptr inbounds nuw i8, ptr %base2, i64 88
  %hsp = getelementptr inbounds nuw i32, ptr %hspan, i64 %gi
  store i32 %count32, ptr %hsp, align 4
  %gi1 = add nuw nsw i64 %gi, 1
  br label %grow.head

grow.done:
  %newlev = call i64 @llvm.umax.i64(i64 %oldlev, i64 %nl)
  %newlev32 = trunc i64 %newlev to i32
  %levp2 = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i32 %newlev32, ptr %levp2, align 4
  br label %link.head

link.head:
  %li = phi i64 [ 0, %grow.done ], [ %li1, %link.body ]
  %lcond = icmp ult i64 %li, %nl
  br i1 %lcond, label %link.body, label %link.done

link.body:
  %lup = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %li
  %u = load i64, ptr %lup, align 8
  %lrp = getelementptr inbounds nuw [16 x i64], ptr %rankv, i64 0, i64 %li
  %ru = load i64, ptr %lrp, align 8
  %uoff = mul nuw i64 %u, 152
  %unode = getelementptr inbounds nuw i8, ptr %base2, i64 %uoff
  %ufbase = getelementptr inbounds nuw i8, ptr %unode, i64 24
  %ufp = getelementptr inbounds nuw i32, ptr %ufbase, i64 %li
  %fu = load i32, ptr %ufp, align 4
  %zfbase = getelementptr inbounds nuw i8, ptr %znode, i64 24
  %zfp = getelementptr inbounds nuw i32, ptr %zfbase, i64 %li
  store i32 %fu, ptr %zfp, align 4
  %z32 = trunc i64 %z to i32
  store i32 %z32, ptr %ufp, align 4
  %usbase = getelementptr inbounds nuw i8, ptr %unode, i64 88
  %usp = getelementptr inbounds nuw i32, ptr %usbase, i64 %li
  %su32 = load i32, ptr %usp, align 4
  %su = zext i32 %su32 to i64
  %zsbase = getelementptr inbounds nuw i8, ptr %znode, i64 88
  %zsp = getelementptr inbounds nuw i32, ptr %zsbase, i64 %li
  %delta = sub i64 %rank0, %ru
  %zspan = sub i64 %su, %delta
  %zspan32 = trunc i64 %zspan to i32
  store i32 %zspan32, ptr %zsp, align 4
  %uspan = add i64 %delta, 1
  %uspan32 = trunc i64 %uspan to i32
  store i32 %uspan32, ptr %usp, align 4
  %li1 = add nuw nsw i64 %li, 1
  br label %link.head

link.done:
  br label %incr.head

incr.head:
  %ii = phi i64 [ %nl, %link.done ], [ %ii1, %incr.body ]
  %icond = icmp ult i64 %ii, %oldlev
  br i1 %icond, label %incr.body, label %incr.done

incr.body:
  %iup = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %ii
  %iu = load i64, ptr %iup, align 8
  %iuoff = mul nuw i64 %iu, 152
  %iunode = getelementptr inbounds nuw i8, ptr %base2, i64 %iuoff
  %iusbase = getelementptr inbounds nuw i8, ptr %iunode, i64 88
  %iusp = getelementptr inbounds nuw i32, ptr %iusbase, i64 %ii
  %isu = load i32, ptr %iusp, align 4
  %isu1 = add i32 %isu, 1
  store i32 %isu1, ptr %iusp, align 4
  %ii1 = add nuw nsw i64 %ii, 1
  br label %incr.head

incr.done:
  %cnt2 = load i64, ptr %cntp, align 8
  %cnt3 = add i64 %cnt2, 1
  store i64 %cnt3, ptr %cntp, align 8
  ret i32 0
}

; ===========================================================================
; delete
; ===========================================================================
define i32 @universe_ds_skiplist_delete(ptr %h, i64 %key) local_unnamed_addr #1 {
entry:
  %update = alloca [16 x i64], align 8
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %lev = zext i32 %lev32 to i64
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %key
  br i1 %lt, label %adv, label %stop

adv:
  br label %inner

stop:
  %up = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %i
  store i64 %xc, ptr %up, align 8
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %afterdescend, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

afterdescend:
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %cand = load i32, ptr %f0, align 4
  %cnil = icmp eq i32 %cand, -1
  br i1 %cnil, label %notfound, label %chkeq

chkeq:
  %cand64 = zext i32 %cand to i64
  %coff = mul nuw i64 %cand64, 152
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %ckey = load i64, ptr %cnode, align 8
  %eq = icmp eq i64 %ckey, %key
  br i1 %eq, label %splice.head, label %notfound

notfound:
  ret i32 5

splice.head:
  %si = phi i64 [ 0, %chkeq ], [ %si1, %splice.next ]
  %scond = icmp ult i64 %si, %lev
  br i1 %scond, label %splice.body, label %splice.done

splice.body:
  %sup = getelementptr inbounds nuw [16 x i64], ptr %update, i64 0, i64 %si
  %u = load i64, ptr %sup, align 8
  %uoff = mul nuw i64 %u, 152
  %unode = getelementptr inbounds nuw i8, ptr %base, i64 %uoff
  %ufbase = getelementptr inbounds nuw i8, ptr %unode, i64 24
  %ufp = getelementptr inbounds nuw i32, ptr %ufbase, i64 %si
  %fu = load i32, ptr %ufp, align 4
  %usbase = getelementptr inbounds nuw i8, ptr %unode, i64 88
  %usp = getelementptr inbounds nuw i32, ptr %usbase, i64 %si
  %su = load i32, ptr %usp, align 4
  %pointsat = icmp eq i32 %fu, %cand
  br i1 %pointsat, label %unlink, label %shrinkspan

unlink:
  ; u.span[si] += cand.span[si] - 1 ; forward[u][si] = forward[cand][si]
  %csbase = getelementptr inbounds nuw i8, ptr %cnode, i64 88
  %csp = getelementptr inbounds nuw i32, ptr %csbase, i64 %si
  %sc = load i32, ptr %csp, align 4
  %sum = add i32 %su, %sc
  %newu = add i32 %sum, -1
  store i32 %newu, ptr %usp, align 4
  %cfbase = getelementptr inbounds nuw i8, ptr %cnode, i64 24
  %cfp = getelementptr inbounds nuw i32, ptr %cfbase, i64 %si
  %fc = load i32, ptr %cfp, align 4
  store i32 %fc, ptr %ufp, align 4
  br label %splice.next

shrinkspan:
  %sdec = add i32 %su, -1
  store i32 %sdec, ptr %usp, align 4
  br label %splice.next

splice.next:
  %si1 = add nuw nsw i64 %si, 1
  br label %splice.head

splice.done:
  ; free node, count--, lower level while head.forward[level-1]==-1 and level>1
  call void @sl_free_node(ptr %h, i64 %cand64)
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %cntd = add i64 %cnt, -1
  store i64 %cntd, ptr %cntp, align 8
  br label %lower.head

lower.head:
  %curlev = load i32, ptr %levp, align 4
  %gt1 = icmp ugt i32 %curlev, 1
  br i1 %gt1, label %lower.chk, label %lower.done

lower.chk:
  %clm1 = sub nsw i32 %curlev, 1
  %clm1.64 = sext i32 %clm1 to i64
  %hf = getelementptr inbounds nuw i8, ptr %base, i64 24
  %hfp = getelementptr inbounds nuw i32, ptr %hf, i64 %clm1.64
  %hfwd = load i32, ptr %hfp, align 4
  %headnil = icmp eq i32 %hfwd, -1
  br i1 %headnil, label %lower.dec, label %lower.done

lower.dec:
  store i32 %clm1, ptr %levp, align 4
  br label %lower.head

lower.done:
  ret i32 0
}

; ===========================================================================
; min / max
; ===========================================================================
define i32 @universe_ds_skiplist_min(ptr %h, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  ; first node = forward[head][0]
  %hf = getelementptr inbounds nuw i8, ptr %base, i64 24
  %first = load i32, ptr %hf, align 4
  %empty = icmp eq i32 %first, -1
  br i1 %empty, label %err.empty, label %emit, !prof !0

err.empty:
  ret i32 4

emit:
  %f64 = zext i32 %first to i64
  %off = mul nuw i64 %f64, 152
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

define i32 @universe_ds_skiplist_max(ptr %h, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %err.empty, label %scan.setup, !prof !0

err.empty:
  ret i32 4

scan.setup:
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %scan.setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %scan.setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %adv

adv:
  %fwd64 = zext i32 %fwd to i64
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %emit, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

emit:
  ; xc is the last node (max)
  %k = load i64, ptr %xcnode, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %doval, label %storek

storek:
  store i64 %k, ptr %ok, align 8
  br label %doval

doval:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %ret0, label %storeval

storeval:
  %vp = getelementptr inbounds nuw i8, ptr %xcnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  br label %ret0

ret0:
  ret i32 0
}

; ===========================================================================
; floor (largest <= key) / ceiling (smallest >= key)
; ===========================================================================
define i32 @universe_ds_skiplist_floor(ptr %h, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %le = icmp sle i64 %fkey, %key
  br i1 %le, label %adv, label %stop

adv:
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %examine, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

examine:
  ; xc is the last node with key <= search; head (0) means none
  %ishead = icmp eq i64 %xc, 0
  br i1 %ishead, label %notfound, label %emit

notfound:
  ret i32 5

emit:
  %k = load i64, ptr %xcnode, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %doval, label %storek

storek:
  store i64 %k, ptr %ok, align 8
  br label %doval

doval:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %ret0, label %storeval

storeval:
  %vp = getelementptr inbounds nuw i8, ptr %xcnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  br label %ret0

ret0:
  ret i32 0
}

define i32 @universe_ds_skiplist_ceiling(ptr %h, i64 %key, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %key
  br i1 %lt, label %adv, label %stop

adv:
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %examine, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

examine:
  ; candidate = forward[xc][0] is the smallest key >= search
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %cand = load i32, ptr %f0, align 4
  %cnil = icmp eq i32 %cand, -1
  br i1 %cnil, label %notfound, label %emit

notfound:
  ret i32 5

emit:
  %cand64 = zext i32 %cand to i64
  %coff = mul nuw i64 %cand64, 152
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %k = load i64, ptr %cnode, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %doval, label %storek

storek:
  store i64 %k, ptr %ok, align 8
  br label %doval

doval:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %ret0, label %storeval

storeval:
  %vp = getelementptr inbounds nuw i8, ptr %cnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  br label %ret0

ret0:
  ret i32 0
}

; ===========================================================================
; rank (0-based index of key) / select (key/val at index)
; ===========================================================================
define i32 @universe_ds_skiplist_rank(ptr %h, i64 %key, ptr %out) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  %rank = phi i64 [ 0, %setup ], [ %rc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %rc = phi i64 [ %rank, %lvl.head ], [ %radv, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %key
  br i1 %lt, label %adv, label %stop

adv:
  %sbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 88
  %sp = getelementptr inbounds nuw i32, ptr %sbase, i64 %i
  %s32 = load i32, ptr %sp, align 4
  %s64 = zext i32 %s32 to i64
  %radv = add i64 %rc, %s64
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %examine, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

examine:
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %cand = load i32, ptr %f0, align 4
  %cnil = icmp eq i32 %cand, -1
  br i1 %cnil, label %notfound, label %chkeq

chkeq:
  %cand64 = zext i32 %cand to i64
  %coff = mul nuw i64 %cand64, 152
  %cnode = getelementptr inbounds nuw i8, ptr %base, i64 %coff
  %ckey = load i64, ptr %cnode, align 8
  %eq = icmp eq i64 %ckey, %key
  br i1 %eq, label %found, label %notfound

found:
  ; element index = rank (rc = position of predecessor, candidate at rc+1,
  ; excluding head sentinel => index rc)
  store i64 %rc, ptr %out, align 8
  ret i32 0

notfound:
  ret i32 5
}

define i32 @universe_ds_skiplist_select(ptr %h, i64 %idx, ptr %ok, ptr %ov) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %base = load ptr, ptr %h, align 8
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  %oob = icmp uge i64 %idx, %cnt
  br i1 %oob, label %err.idx, label %scan.setup, !prof !0

err.idx:
  ret i32 7

scan.setup:
  %target = add i64 %idx, 1
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %scan.setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %scan.setup ], [ %xc, %lvl.next ]
  %pos = phi i64 [ 0, %scan.setup ], [ %pc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %pc = phi i64 [ %pos, %lvl.head ], [ %padv, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkspan

chkspan:
  %sbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 88
  %sp = getelementptr inbounds nuw i32, ptr %sbase, i64 %i
  %s32 = load i32, ptr %sp, align 4
  %s64 = zext i32 %s32 to i64
  %padv = add i64 %pc, %s64
  %le = icmp ule i64 %padv, %target
  br i1 %le, label %adv, label %stop

adv:
  %fwd64 = zext i32 %fwd to i64
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %examine, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

examine:
  ; xc at position pc; if pc == target it is the wanted element
  %hit = icmp eq i64 %pc, %target
  br i1 %hit, label %emit, label %err.idx2

err.idx2:
  ret i32 7

emit:
  %k = load i64, ptr %xcnode, align 8
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %doval, label %storek

storek:
  store i64 %k, ptr %ok, align 8
  br label %doval

doval:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %ret0, label %storeval

storeval:
  %vp = getelementptr inbounds nuw i8, ptr %xcnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %ov, align 8
  br label %ret0

ret0:
  ret i32 0
}

; ===========================================================================
; range: keys in [lo, hi) into ok/ov (either may be null), capped at max.
; Returns the number written.
; ===========================================================================
define i64 @universe_ds_skiplist_range(ptr %h, i64 %lo, i64 %hi, ptr %ok, ptr %ov, i64 %max) local_unnamed_addr #0 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %ret.zero, label %setup, !prof !0

ret.zero:
  ret i64 0

setup:
  %base = load ptr, ptr %h, align 8
  %levp = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lev32 = load i32, ptr %levp, align 4
  %levm1 = sub nsw i32 %lev32, 1
  %levm1.64 = sext i32 %levm1 to i64
  br label %lvl.head

lvl.head:
  %i = phi i64 [ %levm1.64, %setup ], [ %i.dec, %lvl.next ]
  %x = phi i64 [ 0, %setup ], [ %xc, %lvl.next ]
  br label %inner

inner:
  %xc = phi i64 [ %x, %lvl.head ], [ %fwd64, %adv ]
  %xcoff = mul nuw i64 %xc, 152
  %xcnode = getelementptr inbounds nuw i8, ptr %base, i64 %xcoff
  %fbase = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %fp = getelementptr inbounds nuw i32, ptr %fbase, i64 %i
  %fwd = load i32, ptr %fp, align 4
  %isnil = icmp eq i32 %fwd, -1
  br i1 %isnil, label %stop, label %chkkey

chkkey:
  %fwd64 = zext i32 %fwd to i64
  %foff = mul nuw i64 %fwd64, 152
  %fnode = getelementptr inbounds nuw i8, ptr %base, i64 %foff
  %fkey = load i64, ptr %fnode, align 8
  %lt = icmp slt i64 %fkey, %lo
  br i1 %lt, label %adv, label %stop

adv:
  br label %inner

stop:
  %atbottom = icmp eq i64 %i, 0
  br i1 %atbottom, label %walk.start, label %lvl.next

lvl.next:
  %i.dec = add i64 %i, -1
  br label %lvl.head

walk.start:
  ; first candidate = forward[xc][0]
  %f0 = getelementptr inbounds nuw i8, ptr %xcnode, i64 24
  %start = load i32, ptr %f0, align 4
  br label %walk

walk:
  %cur = phi i32 [ %start, %walk.start ], [ %nxt, %wnext ]
  %n = phi i64 [ 0, %walk.start ], [ %n1, %wnext ]
  %curnil = icmp eq i32 %cur, -1
  br i1 %curnil, label %walk.done, label %walk.chk

walk.chk:
  %atmax = icmp uge i64 %n, %max
  br i1 %atmax, label %walk.done, label %walk.key

walk.key:
  %cur64 = zext i32 %cur to i64
  %curoff = mul nuw i64 %cur64, 152
  %curnode = getelementptr inbounds nuw i8, ptr %base, i64 %curoff
  %ck = load i64, ptr %curnode, align 8
  %below = icmp slt i64 %ck, %hi
  br i1 %below, label %writeit, label %walk.done

writeit:
  %oknull = icmp eq ptr %ok, null
  br i1 %oknull, label %wov, label %wok

wok:
  %okp = getelementptr inbounds nuw i64, ptr %ok, i64 %n
  store i64 %ck, ptr %okp, align 8
  br label %wov

wov:
  %ovnull = icmp eq ptr %ov, null
  br i1 %ovnull, label %wnext, label %wovs

wovs:
  %cvp = getelementptr inbounds nuw i8, ptr %curnode, i64 8
  %cv = load i64, ptr %cvp, align 8
  %ovp = getelementptr inbounds nuw i64, ptr %ov, i64 %n
  store i64 %cv, ptr %ovp, align 8
  br label %wnext

wnext:
  %cfp = getelementptr inbounds nuw i8, ptr %curnode, i64 24
  %nxt = load i32, ptr %cfp, align 4
  %n1 = add i64 %n, 1
  br label %walk

walk.done:
  ret i64 %n
}

define i64 @universe_ds_skiplist_size(ptr %h) local_unnamed_addr #2 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %zero, label %load, !prof !0

zero:
  ret i64 0

load:
  %cntp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %cnt = load i64, ptr %cntp, align 8
  ret i64 %cnt
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #4 = { nounwind willreturn norecurse nosync }
attributes #5 = { nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
