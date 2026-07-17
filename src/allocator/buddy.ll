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

; Buddy allocator: power-of-two blocks, O(1) find-fit, xor coalescing.
;
; DESIGN (vs the typical C implementation):
;   * SIZED FREE API (caller passes the size, like C++ sized delete): zero
;     per-block metadata, so a 64B block costs exactly 64B.
;   * `avail_mask` is a u64 with bit k set iff order-k free list is nonempty:
;     find-fit = one shift + one cttz. No list scanning ever.
;   * Buddy of block at offset `off`, order k: `off ^ (min<<k)` — one xor.
;   * Free/split state is one bit per (order, index) in a compact bitmap;
;     bit base for order k is the closed form 2N - (2N >> k) — no tables.
;   * Free blocks carry intrusive {next,prev} offsets (doubly linked ⇒ O(1)
;     unlink of a coalescing buddy). min block 32B accommodates them.
;   * Header: total@0 min@8 K@16 log2min@24 avail@32 live@40 payoff@48;
;     heads[K+1] @64; bitmap after; payload aligned to 64.
;
; API:
;   ptr  universe_alloc_buddy_create(i64 total_size, i64 min_block)
;   ptr  universe_alloc_buddy_alloc(ptr b, i64 size)
;   void universe_alloc_buddy_free(ptr b, ptr p, i64 size)  ; size as at alloc
;   i64  universe_alloc_buddy_live(ptr b)
;   void universe_alloc_buddy_destroy(ptr b)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i64 @llvm.cttz.i64(i64, i1 immarg)

; ---- tiny helpers (all alwaysinline; the optimizer flattens everything) ----

; ceil_log2(x) for x>=1 (returns 0 for x==1)
define internal i64 @clog2(i64 %x) #0 {
entry:
  %xm1 = add i64 %x, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %xm1, i1 false)
  %r = sub i64 64, %lz
  %is1 = icmp ult i64 %x, 2
  %res = select i1 %is1, i64 0, i64 %r
  ret i64 %res
}

; bitmap word ptr + mask for (order k, offset off)
define internal { ptr, i64 } @bitref(ptr %b, i64 %k, i64 %off) #0 {
entry:
  %min = getelementptr inbounds nuw i8, ptr %b, i64 8
  %minv = load i64, ptr %min, align 8
  %l2m = getelementptr inbounds nuw i8, ptr %b, i64 24
  %l2mv = load i64, ptr %l2m, align 8
  %total.p = getelementptr inbounds nuw i8, ptr %b, i64 0
  %total = load i64, ptr %total.p, align 8
  %N = udiv i64 %total, %minv
  %N2 = shl i64 %N, 1
  %shifted = lshr i64 %N2, %k
  %base = sub i64 %N2, %shifted
  %shift.amt = add i64 %l2mv, %k
  %idx = lshr i64 %off, %shift.amt
  %pos = add i64 %base, %idx
  %word = lshr i64 %pos, 6
  %bit = and i64 %pos, 63
  %mask = shl i64 1, %bit
  %K.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %K = load i64, ptr %K.p, align 8
  %heads.bytes = shl i64 %K, 3
  %heads.end = add i64 %heads.bytes, 72       ; 64 + (K+1)*8
  %bm.base = getelementptr inbounds nuw i8, ptr %b, i64 %heads.end
  %w.off = shl i64 %word, 3
  %w.p = getelementptr inbounds nuw i8, ptr %bm.base, i64 %w.off
  %r0 = insertvalue { ptr, i64 } poison, ptr %w.p, 0
  %r1 = insertvalue { ptr, i64 } %r0, i64 %mask, 1
  ret { ptr, i64 } %r1
}

define internal ptr @payload_base(ptr %b) #0 {
entry:
  %po.p = getelementptr inbounds nuw i8, ptr %b, i64 48
  %po = load i64, ptr %po.p, align 8
  %p = getelementptr inbounds nuw i8, ptr %b, i64 %po
  ret ptr %p
}

define internal ptr @head_slot(ptr %b, i64 %k) #0 {
entry:
  %off = shl i64 %k, 3
  %base = getelementptr inbounds nuw i8, ptr %b, i64 64
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %off
  ret ptr %p
}

; push free block `off` onto list k; set avail bit and free bit
define internal void @blk_push(ptr %b, i64 %k, i64 %off) #0 {
entry:
  %hs = call ptr @head_slot(ptr %b, i64 %k)
  %old = load i64, ptr %hs, align 8
  %pay = call ptr @payload_base(ptr %b)
  %blk = getelementptr inbounds nuw i8, ptr %pay, i64 %off
  store i64 %old, ptr %blk, align 8                  ; next
  %prev.p = getelementptr inbounds nuw i8, ptr %blk, i64 8
  store i64 -1, ptr %prev.p, align 8                 ; prev
  %old.none = icmp eq i64 %old, -1
  br i1 %old.none, label %set.head, label %backlink

backlink:
  %oldblk = getelementptr inbounds nuw i8, ptr %pay, i64 %old
  %oldprev.p = getelementptr inbounds nuw i8, ptr %oldblk, i64 8
  store i64 %off, ptr %oldprev.p, align 8
  br label %set.head

set.head:
  store i64 %off, ptr %hs, align 8
  %av.p = getelementptr inbounds nuw i8, ptr %b, i64 32
  %av = load i64, ptr %av.p, align 8
  %kbit = shl i64 1, %k
  %av.n = or i64 %av, %kbit
  store i64 %av.n, ptr %av.p, align 8
  %br = call { ptr, i64 } @bitref(ptr %b, i64 %k, i64 %off)
  %w.p = extractvalue { ptr, i64 } %br, 0
  %mask = extractvalue { ptr, i64 } %br, 1
  %w = load i64, ptr %w.p, align 8
  %w.n = or i64 %w, %mask
  store i64 %w.n, ptr %w.p, align 8
  ret void
}

; remove specific free block `off` from list k; clear free bit; fix avail
define internal void @blk_remove(ptr %b, i64 %k, i64 %off) #0 {
entry:
  %pay = call ptr @payload_base(ptr %b)
  %blk = getelementptr inbounds nuw i8, ptr %pay, i64 %off
  %next = load i64, ptr %blk, align 8
  %prev.p = getelementptr inbounds nuw i8, ptr %blk, i64 8
  %prev = load i64, ptr %prev.p, align 8
  %p.none = icmp eq i64 %prev, -1
  br i1 %p.none, label %fix.head, label %fix.prev

fix.prev:
  %pblk = getelementptr inbounds nuw i8, ptr %pay, i64 %prev
  store i64 %next, ptr %pblk, align 8
  br label %mid

fix.head:
  %hs = call ptr @head_slot(ptr %b, i64 %k)
  store i64 %next, ptr %hs, align 8
  br label %mid

mid:
  %n.none = icmp eq i64 %next, -1
  br i1 %n.none, label %maybe.clear, label %fix.next

fix.next:
  %nblk = getelementptr inbounds nuw i8, ptr %pay, i64 %next
  %nprev.p = getelementptr inbounds nuw i8, ptr %nblk, i64 8
  store i64 %prev, ptr %nprev.p, align 8
  br label %clear.bit

maybe.clear:                                 ; list may now be empty
  br i1 %p.none, label %clear.avail, label %clear.bit

clear.avail:
  %av.p = getelementptr inbounds nuw i8, ptr %b, i64 32
  %av = load i64, ptr %av.p, align 8
  %kbit = shl i64 1, %k
  %kbit.not = xor i64 %kbit, -1
  %av.n = and i64 %av, %kbit.not
  store i64 %av.n, ptr %av.p, align 8
  br label %clear.bit

clear.bit:
  %br2 = call { ptr, i64 } @bitref(ptr %b, i64 %k, i64 %off)
  %w.p = extractvalue { ptr, i64 } %br2, 0
  %mask = extractvalue { ptr, i64 } %br2, 1
  %w = load i64, ptr %w.p, align 8
  %mask.not = xor i64 %mask, -1
  %w.n = and i64 %w, %mask.not
  store i64 %w.n, ptr %w.p, align 8
  ret void
}

; ---------------------------------------------------------------------------

define noalias ptr @universe_alloc_buddy_create(i64 %total_size, i64 %min_block) local_unnamed_addr #1 {
entry:
  %min.lo = call i64 @llvm.umax.i64(i64 %min_block, i64 32)
  %l2min = call i64 @clog2(i64 %min.lo)
  %minv = shl i64 1, %l2min
  %tot.lo = call i64 @llvm.umax.i64(i64 %total_size, i64 %minv)
  %too.big = icmp ugt i64 %tot.lo, 1099511627776    ; 1 TiB cap
  br i1 %too.big, label %fail, label %shape, !prof !0

shape:
  %l2tot = call i64 @clog2(i64 %tot.lo)
  %total = shl i64 1, %l2tot
  %K = sub i64 %l2tot, %l2min
  %N = lshr i64 %total, %l2min
  ; bitmap bytes = ceil(2N/8) rounded to 8
  %N2 = shl i64 %N, 1
  %bm.bits = add i64 %N2, 63
  %bm.words = lshr i64 %bm.bits, 6
  %bm.bytes = shl i64 %bm.words, 3
  %heads.bytes = shl i64 %K, 3
  %heads.end = add i64 %heads.bytes, 72             ; 64 + (K+1)*8
  %pay.raw = add i64 %heads.end, %bm.bytes
  %pay.al = add i64 %pay.raw, 63
  %payoff = and i64 %pay.al, -64
  %grand = add i64 %payoff, %total
  ; OS-request floor (ALLOC_OS_MIN=16384): never malloc less than 16 KiB for the
  ; backing region. The buddy tree still manages exactly %total payload bytes at
  ; %payoff; a floored request over a small region is harmless slack.
  %os.req = call i64 @llvm.umax.i64(i64 %grand, i64 16384)
  %mem = call ptr @malloc(i64 %os.req)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %total, ptr %mem, align 8
  %min.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %minv, ptr %min.p, align 8
  %K.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %K, ptr %K.p, align 8
  %l2m.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %l2min, ptr %l2m.p, align 8
  %av.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i64 0, ptr %av.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %mem, i64 40
  store i64 0, ptr %live.p, align 8
  %po.p = getelementptr inbounds nuw i8, ptr %mem, i64 48
  store i64 %payoff, ptr %po.p, align 8
  ; heads := -1
  %heads.base = getelementptr inbounds nuw i8, ptr %mem, i64 64
  %hb = add i64 %heads.bytes, 8                     ; (K+1)*8
  call void @llvm.memset.p0.i64(ptr %heads.base, i8 -1, i64 %hb, i1 false)
  ; bitmap := 0
  %bm.base = getelementptr inbounds nuw i8, ptr %mem, i64 %heads.end
  call void @llvm.memset.p0.i64(ptr %bm.base, i8 0, i64 %bm.bytes, i1 false)
  ; one free block: whole arena at order K
  call void @blk_push(ptr %mem, i64 %K, i64 0)
  ret ptr %mem

fail:
  ret ptr null
}

define ptr @universe_alloc_buddy_alloc(ptr %b, i64 %size) local_unnamed_addr #1 {
entry:
  %b.null = icmp eq ptr %b, null
  br i1 %b.null, label %fail, label %order, !prof !0

order:
  %sz = call i64 @llvm.umax.i64(i64 %size, i64 1)
  %l2m.p = getelementptr inbounds nuw i8, ptr %b, i64 24
  %l2min = load i64, ptr %l2m.p, align 8
  %l2sz = call i64 @clog2(i64 %sz)
  %l2need = call i64 @llvm.umax.i64(i64 %l2sz, i64 %l2min)
  %k0 = sub i64 %l2need, %l2min
  %K.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %K = load i64, ptr %K.p, align 8
  %fits = icmp ule i64 %k0, %K
  br i1 %fits, label %find, label %fail, !prof !0

find:
  %av.p = getelementptr inbounds nuw i8, ptr %b, i64 32
  %av = load i64, ptr %av.p, align 8
  %cands = lshr i64 %av, %k0
  %none = icmp eq i64 %cands, 0
  br i1 %none, label %fail, label %take, !prof !0

take:
  %tz = call i64 @llvm.cttz.i64(i64 %cands, i1 true)
  %k = add i64 %k0, %tz
  %hs = call ptr @head_slot(ptr %b, i64 %k)
  %off = load i64, ptr %hs, align 8
  call void @blk_remove(ptr %b, i64 %k, i64 %off)
  br label %split

split:                                       ; while k > k0: halve, push buddy
  %kc = phi i64 [ %k, %take ], [ %kc.n, %split.body ]
  %done.split = icmp ule i64 %kc, %k0
  br i1 %done.split, label %give, label %split.body

split.body:
  %kc.n = add i64 %kc, -1
  %half.shift = add i64 %l2min, %kc.n
  %half = shl i64 1, %half.shift
  %buddy.off = add i64 %off, %half
  call void @blk_push(ptr %b, i64 %kc.n, i64 %buddy.off)
  br label %split

give:
  %live.p = getelementptr inbounds nuw i8, ptr %b, i64 40
  %live = load i64, ptr %live.p, align 8
  %live.n = add i64 %live, 1
  store i64 %live.n, ptr %live.p, align 8
  %pay = call ptr @payload_base(ptr %b)
  %blk = getelementptr inbounds nuw i8, ptr %pay, i64 %off
  ret ptr %blk

fail:
  ret ptr null
}

define void @universe_alloc_buddy_free(ptr %b, ptr %p, i64 %size) local_unnamed_addr #1 {
entry:
  %b.null = icmp eq ptr %b, null
  %p.null = icmp eq ptr %p, null
  %any.null = or i1 %b.null, %p.null
  br i1 %any.null, label %done, label %order, !prof !0

order:
  %sz = call i64 @llvm.umax.i64(i64 %size, i64 1)
  %l2m.p = getelementptr inbounds nuw i8, ptr %b, i64 24
  %l2min = load i64, ptr %l2m.p, align 8
  %l2sz = call i64 @clog2(i64 %sz)
  %l2need = call i64 @llvm.umax.i64(i64 %l2sz, i64 %l2min)
  %k0 = sub i64 %l2need, %l2min
  %K.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %K = load i64, ptr %K.p, align 8
  %pay = call ptr @payload_base(ptr %b)
  %pay.i = ptrtoint ptr %pay to i64
  %p.i = ptrtoint ptr %p to i64
  %off0 = sub i64 %p.i, %pay.i
  br label %coalesce

coalesce:
  %k = phi i64 [ %k0, %order ], [ %k.n, %merge ]
  %off = phi i64 [ %off0, %order ], [ %off.merged, %merge ]
  %at.top = icmp uge i64 %k, %K
  br i1 %at.top, label %park, label %try.buddy

try.buddy:
  %bs.shift = add i64 %l2min, %k
  %bsize = shl i64 1, %bs.shift
  %buddy.off = xor i64 %off, %bsize
  %br2 = call { ptr, i64 } @bitref(ptr %b, i64 %k, i64 %buddy.off)
  %w.p = extractvalue { ptr, i64 } %br2, 0
  %mask = extractvalue { ptr, i64 } %br2, 1
  %w = load i64, ptr %w.p, align 8
  %hit = and i64 %w, %mask
  %buddy.free = icmp ne i64 %hit, 0
  br i1 %buddy.free, label %merge, label %park

merge:
  call void @blk_remove(ptr %b, i64 %k, i64 %buddy.off)
  %off.merged = call i64 @llvm.umin.i64(i64 %off, i64 %buddy.off)
  %k.n = add i64 %k, 1
  br label %coalesce

park:
  call void @blk_push(ptr %b, i64 %k, i64 %off)
  %live.p = getelementptr inbounds nuw i8, ptr %b, i64 40
  %live = load i64, ptr %live.p, align 8
  %live.n = add i64 %live, -1
  store i64 %live.n, ptr %live.p, align 8
  br label %done

done:
  ret void
}

define i64 @universe_alloc_buddy_live(ptr %b) local_unnamed_addr #2 {
entry:
  %live.p = getelementptr inbounds nuw i8, ptr %b, i64 40
  %live = load i64, ptr %live.p, align 8
  ret i64 %live
}

define void @universe_alloc_buddy_destroy(ptr %b) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %b, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %b)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
