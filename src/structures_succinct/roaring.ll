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

; Roaring bitmap: a compressed set of u32 values.
;
; DESIGN (functional spec: a dense/sparse-adaptive set of u32 keys with fast
;         membership, cardinality, ascending iteration, and O(n+m) set algebra):
;   * A u32 value splits into a 16-bit CHUNK KEY (high bits) and a 16-bit LOW
;     value. Values sharing a chunk key live together in one CONTAINER holding
;     only the low 16 bits. The top level is a DIRECTORY of (key, container)
;     kept sorted by key, so two roarings merge in a single linear pass and a
;     lookup is one binary search + one container probe.
;   * Directory layout is SoA: a u16 key array + a parallel container-pointer
;     array, both grown by doubling. Keys packed contiguously keep the binary
;     search in a couple of cache lines; the pointer array is touched only
;     after the key is located. Header { ptr keys@0, ptr conts@8, i64 count@16,
;     i64 cap@24 }. Directory count is bounded by 65536 (u16 key space) so all
;     directory size math is structurally overflow-free.
;   * Two ADAPTIVE container kinds, chosen by cardinality — the crossover is
;     4096 because 4096 u16 (array) == 8192 bytes == the dense bitmap:
;       - ARRAY (sparse, card <= 4096): a SORTED u16 array. Membership is a
;         binary search; insert keeps it sorted with one memmove. Layout
;         { i32 kind=0 @0, i32 card@4, i64 cap@8, u16 data[]@16 }, one malloc,
;         grown by realloc.
;       - BITMAP (dense, card > 4096): a 1024 x i64 = 65536-bit dense bitmap.
;         Membership is a branch-free word+bit test; cardinality is popcount.
;         Layout { i32 kind=1 @0, i32 card@4, pad, i64 words[1024]@64 }, one
;         zeroed malloc of 8256 bytes; words at +64 stay 16-byte aligned for
;         wide vector moves.
;     An ARRAY promotes to BITMAP on the insert that would make it exceed 4096.
;   * SIMD-first hot path (mandatory): the dense bitmap-vs-bitmap AND/OR/ANDNOT
;     are portable <2 x i64> vector loops (SSE2 on AMD64, NEON on AArch64, no
;     runtime check) exported as universe_ds_roaring_bitmap_{and,or,andnot};
;     each ships a scalar twin (*_scalar) that is BOTH the reference oracle
;     (vector==scalar cross-check in tests) and a portable fallback. Set algebra
;     over two bitmap chunks reads the operands' words directly (zero copy) into
;     these kernels.
;   * Set algebra (union/intersection/difference) walks both sorted directories
;     once. Matching keys combine their containers through a scratch 65536-bit
;     bitmap: each operand is EXPANDED to words (a bitmap points at its own
;     words with no copy; an array is scattered into scratch), the vector kernel
;     runs, then the result is NORMALIZED back to an array or bitmap by its
;     popcount. Result keys are produced ascending, so the result directory is
;     appended in order (no per-insert shift).
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 8 INVALID_ARG):
;   ptr  universe_ds_roaring_create(void)
;   void universe_ds_roaring_destroy(ptr r)
;   i32  universe_ds_roaring_add(ptr r, i32 v)          ; idempotent
;   i32  universe_ds_roaring_remove(ptr r, i32 v)       ; absent => OK
;   i32  universe_ds_roaring_contains(ptr r, i32 v)     ; 0/1, 0 if null
;   i64  universe_ds_roaring_cardinality(ptr r)         ; # of values, 0 if null
;   i64  universe_ds_roaring_to_array(ptr r, ptr out_u32, i64 out_cap)
;                                                       ; ascending; ret total
;   i32  universe_ds_roaring_container_kind(ptr r, i32 v) ; 0 arr,1 bmp,-1 none
;   ptr  universe_ds_roaring_union(ptr a, ptr b)        ; a | b  (new roaring)
;   ptr  universe_ds_roaring_intersection(ptr a, ptr b) ; a & b
;   ptr  universe_ds_roaring_difference(ptr a, ptr b)   ; a & ~b (andnot)
;   ; dense 65536-bit block kernels (ptr = 1024 x i64), dst may alias a or b:
;   void universe_ds_roaring_bitmap_or(ptr dst, ptr a, ptr b)
;   void universe_ds_roaring_bitmap_and(ptr dst, ptr a, ptr b)
;   void universe_ds_roaring_bitmap_andnot(ptr dst, ptr a, ptr b) ; a & ~b
;   void universe_ds_roaring_bitmap_or_scalar(ptr dst, ptr a, ptr b)
;   void universe_ds_roaring_bitmap_and_scalar(ptr dst, ptr a, ptr b)
;   void universe_ds_roaring_bitmap_andnot_scalar(ptr dst, ptr a, ptr b)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.cttz.i64(i64, i1 immarg)

; =========================================================================
; Dense 65536-bit block kernels (1024 x i64). SIMD-first + scalar oracle.
; =========================================================================

define void @universe_ds_roaring_bitmap_or(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw <2 x i64>, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %ap, align 8
  %bp = getelementptr inbounds nuw <2 x i64>, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %bp, align 8
  %r = or <2 x i64> %va, %vb
  %dp = getelementptr inbounds nuw <2 x i64>, ptr %dst, i64 %i
  store <2 x i64> %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 512
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @universe_ds_roaring_bitmap_and(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw <2 x i64>, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %ap, align 8
  %bp = getelementptr inbounds nuw <2 x i64>, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %bp, align 8
  %r = and <2 x i64> %va, %vb
  %dp = getelementptr inbounds nuw <2 x i64>, ptr %dst, i64 %i
  store <2 x i64> %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 512
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @universe_ds_roaring_bitmap_andnot(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw <2 x i64>, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %ap, align 8
  %bp = getelementptr inbounds nuw <2 x i64>, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %bp, align 8
  %nb = xor <2 x i64> %vb, splat (i64 -1)
  %r = and <2 x i64> %va, %nb
  %dp = getelementptr inbounds nuw <2 x i64>, ptr %dst, i64 %i
  store <2 x i64> %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 512
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @universe_ds_roaring_bitmap_or_scalar(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %a, i64 %i
  %va = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds nuw i64, ptr %b, i64 %i
  %vb = load i64, ptr %bp, align 8
  %r = or i64 %va, %vb
  %dp = getelementptr inbounds nuw i64, ptr %dst, i64 %i
  store i64 %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @universe_ds_roaring_bitmap_and_scalar(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %a, i64 %i
  %va = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds nuw i64, ptr %b, i64 %i
  %vb = load i64, ptr %bp, align 8
  %r = and i64 %va, %vb
  %dp = getelementptr inbounds nuw i64, ptr %dst, i64 %i
  store i64 %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @universe_ds_roaring_bitmap_andnot_scalar(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %a, i64 %i
  %va = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds nuw i64, ptr %b, i64 %i
  %vb = load i64, ptr %bp, align 8
  %nb = xor i64 %vb, -1
  %r = and i64 %va, %nb
  %dp = getelementptr inbounds nuw i64, ptr %dst, i64 %i
  store i64 %r, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %loop, label %done
done:
  ret void
}

; popcount over 1024 words.
define internal i64 @roar_words_popcount(ptr %w) #3 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %wp = getelementptr inbounds nuw i64, ptr %w, i64 %i
  %v = load i64, ptr %wp, align 8
  %pc = call i64 @llvm.ctpop.i64(i64 %v)
  %acc.n = add i64 %acc, %pc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

; =========================================================================
; Container helpers.
; =========================================================================

; lower_bound over a sorted u16 array [data, data+n) for key (0..65535 in i32).
; returns first index with data[idx] >= key.
define internal i64 @roar_u16_lower_bound(ptr %data, i64 %n, i32 %key) #3 {
entry:
  br label %loop
loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %body ]
  %hi = phi i64 [ %n, %entry ], [ %hi.n, %body ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %body, label %done
body:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %mp = getelementptr inbounds nuw i16, ptr %data, i64 %mid
  %mv = load i16, ptr %mp, align 2
  %mv32 = zext i16 %mv to i32
  %less = icmp ult i32 %mv32, %key
  %mid1 = add nuw i64 %mid, 1
  %lo.n = select i1 %less, i64 %mid1, i64 %lo
  %hi.n = select i1 %less, i64 %hi, i64 %mid
  br label %loop
done:
  ret i64 %lo
}

; create a fresh ARRAY container holding the single low value; null on OOM.
define internal ptr @roar_arr_new(i32 %low) #1 {
entry:
  %c = call ptr @malloc(i64 24)            ; 16 hdr + 4 elems * 2
  %null = icmp eq ptr %c, null
  br i1 %null, label %fail, label %init, !prof !0
init:
  store i32 0, ptr %c, align 8             ; kind = ARRAY
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  store i32 1, ptr %cardp, align 4
  %capp = getelementptr inbounds nuw i8, ptr %c, i64 8
  store i64 4, ptr %capp, align 8
  %d0 = getelementptr inbounds nuw i8, ptr %c, i64 16
  %low16 = trunc i32 %low to i16
  store i16 %low16, ptr %d0, align 2
  ret ptr %c
fail:
  ret ptr null
}

; promote an ARRAY container to a BITMAP holding the same values; null on OOM.
; does NOT free the array; caller owns that.
define internal ptr @roar_arr_to_bitmap(ptr %arr) #1 {
entry:
  %bm = call ptr @calloc(i64 1, i64 8256)  ; 64 hdr + 1024*8 words, zeroed
  %null = icmp eq ptr %bm, null
  br i1 %null, label %fail, label %init, !prof !0
init:
  store i32 1, ptr %bm, align 8            ; kind = BITMAP
  %cardp = getelementptr inbounds nuw i8, ptr %arr, i64 4
  %card = load i32, ptr %cardp, align 4
  %bcardp = getelementptr inbounds nuw i8, ptr %bm, i64 4
  store i32 %card, ptr %bcardp, align 4
  %data = getelementptr inbounds nuw i8, ptr %arr, i64 16
  %words = getelementptr inbounds nuw i8, ptr %bm, i64 64
  %card64 = zext i32 %card to i64
  %empty = icmp eq i64 %card64, 0
  br i1 %empty, label %done, label %loop
loop:
  %i = phi i64 [ 0, %init ], [ %i.n, %loop ]
  %ep = getelementptr inbounds nuw i16, ptr %data, i64 %i
  %ev = load i16, ptr %ep, align 2
  %low = zext i16 %ev to i64
  %wi = lshr i64 %low, 6
  %bit = and i64 %low, 63
  %mask = shl nuw i64 1, %bit
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %cur = load i64, ptr %wp, align 8
  %new = or i64 %cur, %mask
  store i64 %new, ptr %wp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %card64
  br i1 %more, label %loop, label %done
done:
  ret ptr %bm
fail:
  ret ptr null
}

; add %low to the container at *%slotp. 0 added, 9 already present, 2 OOM.
; may realloc/promote and write the new pointer back to *%slotp.
define internal i32 @roar_cont_add(ptr %slotp, i32 %low) #1 {
entry:
  %c = load ptr, ptr %slotp, align 8
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr

bmp:
  %low64 = zext i32 %low to i64
  %wi = lshr i64 %low64, 6
  %bit = and i64 %low64, 63
  %mask = shl nuw i64 1, %bit
  %words = getelementptr inbounds nuw i8, ptr %c, i64 64
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %cur = load i64, ptr %wp, align 8
  %hasbit = and i64 %cur, %mask
  %set = icmp ne i64 %hasbit, 0
  br i1 %set, label %dup, label %bmp.add
bmp.add:
  %new = or i64 %cur, %mask
  store i64 %new, ptr %wp, align 8
  %bcardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %bcard = load i32, ptr %bcardp, align 4
  %bcard.n = add i32 %bcard, 1
  store i32 %bcard.n, ptr %bcardp, align 4
  ret i32 0

arr:
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %card64 = zext i32 %card to i64
  %data = getelementptr inbounds nuw i8, ptr %c, i64 16
  %pos = call i64 @roar_u16_lower_bound(ptr %data, i64 %card64, i32 %low)
  %inrange = icmp ult i64 %pos, %card64
  br i1 %inrange, label %chkdup, label %insert
chkdup:
  %pp = getelementptr inbounds nuw i16, ptr %data, i64 %pos
  %pv = load i16, ptr %pp, align 2
  %pv32 = zext i16 %pv to i32
  %same = icmp eq i32 %pv32, %low
  br i1 %same, label %dup, label %insert
insert:
  %full = icmp uge i32 %card, 4096
  br i1 %full, label %promote, label %maybegrow

promote:
  %bm = call ptr @roar_arr_to_bitmap(ptr %c)
  %bmnull = icmp eq ptr %bm, null
  br i1 %bmnull, label %oom, label %promote.set, !prof !0
promote.set:
  %plow64 = zext i32 %low to i64
  %pwi = lshr i64 %plow64, 6
  %pbit = and i64 %plow64, 63
  %pmask = shl nuw i64 1, %pbit
  %pwords = getelementptr inbounds nuw i8, ptr %bm, i64 64
  %pwp = getelementptr inbounds nuw i64, ptr %pwords, i64 %pwi
  %pcur = load i64, ptr %pwp, align 8
  %pnew = or i64 %pcur, %pmask
  store i64 %pnew, ptr %pwp, align 8
  %pcardp = getelementptr inbounds nuw i8, ptr %bm, i64 4
  %pcard = load i32, ptr %pcardp, align 4
  %pcard.n = add i32 %pcard, 1
  store i32 %pcard.n, ptr %pcardp, align 4
  call void @free(ptr %c)
  store ptr %bm, ptr %slotp, align 8
  ret i32 0

maybegrow:
  %capp = getelementptr inbounds nuw i8, ptr %c, i64 8
  %cap = load i64, ptr %capp, align 8
  %atcap = icmp uge i64 %card64, %cap
  br i1 %atcap, label %grow, label %shift
grow:
  %cap2 = shl i64 %cap, 1                   ; cap <= 4096, product fits i64
  %bytes = shl i64 %cap2, 1
  %bytes.tot = add i64 %bytes, 16
  %nc = call ptr @realloc(ptr %c, i64 %bytes.tot)
  %ncnull = icmp eq ptr %nc, null
  br i1 %ncnull, label %oom, label %grew, !prof !0
grew:
  %ncapp = getelementptr inbounds nuw i8, ptr %nc, i64 8
  store i64 %cap2, ptr %ncapp, align 8
  store ptr %nc, ptr %slotp, align 8
  br label %shift
shift:
  %cc = phi ptr [ %c, %maybegrow ], [ %nc, %grew ]
  %data2 = getelementptr inbounds nuw i8, ptr %cc, i64 16
  %src = getelementptr inbounds nuw i16, ptr %data2, i64 %pos
  %dst = getelementptr inbounds nuw i16, ptr %data2, i64 %pos
  %dst1 = getelementptr inbounds nuw i16, ptr %dst, i64 1
  %tail = sub i64 %card64, %pos
  %tailbytes = shl i64 %tail, 1
  call void @llvm.memmove.p0.p0.i64(ptr %dst1, ptr %src, i64 %tailbytes, i1 false)
  %low16 = trunc i32 %low to i16
  store i16 %low16, ptr %src, align 2
  %cardp2 = getelementptr inbounds nuw i8, ptr %cc, i64 4
  %card.n = add i32 %card, 1
  store i32 %card.n, ptr %cardp2, align 4
  ret i32 0

dup:
  ret i32 9
oom:
  ret i32 2
}

; remove %low from container at *%slotp. 0 removed, 5 not present.
define internal i32 @roar_cont_remove(ptr %slotp, i32 %low) #1 {
entry:
  %c = load ptr, ptr %slotp, align 8
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr

bmp:
  %low64 = zext i32 %low to i64
  %wi = lshr i64 %low64, 6
  %bit = and i64 %low64, 63
  %mask = shl nuw i64 1, %bit
  %words = getelementptr inbounds nuw i8, ptr %c, i64 64
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %cur = load i64, ptr %wp, align 8
  %hasbit = and i64 %cur, %mask
  %set = icmp ne i64 %hasbit, 0
  br i1 %set, label %bmp.clr, label %none
bmp.clr:
  %nmask = xor i64 %mask, -1
  %new = and i64 %cur, %nmask
  store i64 %new, ptr %wp, align 8
  %bcardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %bcard = load i32, ptr %bcardp, align 4
  %bcard.n = sub i32 %bcard, 1
  store i32 %bcard.n, ptr %bcardp, align 4
  ret i32 0

arr:
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %card64 = zext i32 %card to i64
  %data = getelementptr inbounds nuw i8, ptr %c, i64 16
  %pos = call i64 @roar_u16_lower_bound(ptr %data, i64 %card64, i32 %low)
  %inrange = icmp ult i64 %pos, %card64
  br i1 %inrange, label %chk, label %none
chk:
  %pp = getelementptr inbounds nuw i16, ptr %data, i64 %pos
  %pv = load i16, ptr %pp, align 2
  %pv32 = zext i16 %pv to i32
  %same = icmp eq i32 %pv32, %low
  br i1 %same, label %arr.del, label %none
arr.del:
  %pos1 = add nuw i64 %pos, 1
  %next = getelementptr inbounds nuw i16, ptr %data, i64 %pos1
  %tail = sub i64 %card64, %pos1
  %tailbytes = shl i64 %tail, 1
  call void @llvm.memmove.p0.p0.i64(ptr %pp, ptr %next, i64 %tailbytes, i1 false)
  %card.n = add i32 %card, -1
  store i32 %card.n, ptr %cardp, align 4
  ret i32 0

none:
  ret i32 5
}

; membership: 1 if %low in container, else 0.
define internal i32 @roar_cont_contains(ptr %c, i32 %low) #2 {
entry:
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr
bmp:
  %low64 = zext i32 %low to i64
  %wi = lshr i64 %low64, 6
  %bit = and i64 %low64, 63
  %words = getelementptr inbounds nuw i8, ptr %c, i64 64
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %cur = load i64, ptr %wp, align 8
  %sh = lshr i64 %cur, %bit
  %b = and i64 %sh, 1
  %r = trunc i64 %b to i32
  ret i32 %r
arr:
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %card64 = zext i32 %card to i64
  %data = getelementptr inbounds nuw i8, ptr %c, i64 16
  %pos = call i64 @roar_u16_lower_bound(ptr %data, i64 %card64, i32 %low)
  %inrange = icmp ult i64 %pos, %card64
  br i1 %inrange, label %chk, label %miss
chk:
  %pp = getelementptr inbounds nuw i16, ptr %data, i64 %pos
  %pv = load i16, ptr %pp, align 2
  %pv32 = zext i16 %pv to i32
  %same = icmp eq i32 %pv32, %low
  %r2 = zext i1 %same to i32
  ret i32 %r2
miss:
  ret i32 0
}

; byte size of a container's single allocation.
define internal i64 @roar_cont_bytes(ptr %c) #2 {
entry:
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr
bmp:
  ret i64 8256
arr:
  %capp = getelementptr inbounds nuw i8, ptr %c, i64 8
  %cap = load i64, ptr %capp, align 8
  %bytes = shl i64 %cap, 1
  %tot = add i64 %bytes, 16
  ret i64 %tot
}

; deep-copy a container; null on OOM.
define internal ptr @roar_cont_clone(ptr %c) #1 {
entry:
  %bytes = call i64 @roar_cont_bytes(ptr %c)
  %nc = call ptr @malloc(i64 %bytes)
  %null = icmp eq ptr %nc, null
  br i1 %null, label %fail, label %copy, !prof !0
copy:
  call void @llvm.memcpy.p0.p0.i64(ptr %nc, ptr %c, i64 %bytes, i1 false)
  ret ptr %nc
fail:
  ret ptr null
}

; expand a container into words: bitmap -> its own words (no copy); array ->
; scattered into %scratch (zeroed then set). returns the words pointer.
define internal ptr @roar_cont_expand(ptr %c, ptr %scratch) #1 {
entry:
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr
bmp:
  %words = getelementptr inbounds nuw i8, ptr %c, i64 64
  ret ptr %words
arr:
  call void @llvm.memset.p0.i64(ptr %scratch, i8 0, i64 8192, i1 false)
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %card64 = zext i32 %card to i64
  %data = getelementptr inbounds nuw i8, ptr %c, i64 16
  %empty = icmp eq i64 %card64, 0
  br i1 %empty, label %done, label %loop
loop:
  %i = phi i64 [ 0, %arr ], [ %i.n, %loop ]
  %ep = getelementptr inbounds nuw i16, ptr %data, i64 %i
  %ev = load i16, ptr %ep, align 2
  %low = zext i16 %ev to i64
  %wi = lshr i64 %low, 6
  %bit = and i64 %low, 63
  %mask = shl nuw i64 1, %bit
  %wp = getelementptr inbounds nuw i64, ptr %scratch, i64 %wi
  %cur = load i64, ptr %wp, align 8
  %new = or i64 %cur, %mask
  store i64 %new, ptr %wp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %card64
  br i1 %more, label %loop, label %done
done:
  ret ptr %scratch
}

; build a container from a 1024-word bitmap with known cardinality.
; card==0 -> null; card<=4096 -> ARRAY; else -> BITMAP. null also on OOM
; (indistinguishable from empty; callers treat null as "no container").
define internal ptr @roar_normalize(ptr %words, i32 %card) #1 {
entry:
  %zero = icmp eq i32 %card, 0
  br i1 %zero, label %empty, label %choose
choose:
  %dense = icmp ugt i32 %card, 4096
  br i1 %dense, label %mkbmp, label %mkarr

mkbmp:
  %bm = call ptr @malloc(i64 8256)
  %bmnull = icmp eq ptr %bm, null
  br i1 %bmnull, label %empty, label %bmp.fill, !prof !0
bmp.fill:
  store i32 1, ptr %bm, align 8
  %bcardp = getelementptr inbounds nuw i8, ptr %bm, i64 4
  store i32 %card, ptr %bcardp, align 4
  %bwords = getelementptr inbounds nuw i8, ptr %bm, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %bwords, ptr %words, i64 8192, i1 false)
  ret ptr %bm

mkarr:
  %card64 = zext i32 %card to i64
  %abytes = shl i64 %card64, 1
  %atot = add i64 %abytes, 16
  %arr = call ptr @malloc(i64 %atot)
  %arrnull = icmp eq ptr %arr, null
  br i1 %arrnull, label %empty, label %arr.fill, !prof !0
arr.fill:
  store i32 0, ptr %arr, align 8
  %acardp = getelementptr inbounds nuw i8, ptr %arr, i64 4
  store i32 %card, ptr %acardp, align 4
  %acapp = getelementptr inbounds nuw i8, ptr %arr, i64 8
  store i64 %card64, ptr %acapp, align 8
  %adata = getelementptr inbounds nuw i8, ptr %arr, i64 16
  br label %scan
scan:
  %wi = phi i64 [ 0, %arr.fill ], [ %wi.n, %scan.cont ]
  %oi = phi i64 [ 0, %arr.fill ], [ %oi.w, %scan.cont ]
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %word = load i64, ptr %wp, align 8
  %wbase = shl i64 %wi, 6
  br label %bits
bits:
  %w = phi i64 [ %word, %scan ], [ %w.n, %emit ]
  %oi2 = phi i64 [ %oi, %scan ], [ %oi2.n, %emit ]
  %nz = icmp ne i64 %w, 0
  br i1 %nz, label %emit, label %bits.done
emit:
  %tz = call i64 @llvm.cttz.i64(i64 %w, i1 true)
  %val = add i64 %wbase, %tz
  %val16 = trunc i64 %val to i16
  %op = getelementptr inbounds nuw i16, ptr %adata, i64 %oi2
  store i16 %val16, ptr %op, align 2
  %oi2.n = add nuw i64 %oi2, 1
  %wm1 = sub i64 %w, 1
  %w.n = and i64 %w, %wm1
  br label %bits
bits.done:
  br label %scan.cont
scan.cont:
  %oi.w = phi i64 [ %oi2, %bits.done ]
  %wi.n = add nuw i64 %wi, 1
  %more = icmp ult i64 %wi.n, 1024
  br i1 %more, label %scan, label %arr.ret
arr.ret:
  ret ptr %arr

empty:
  ret ptr null
}

; =========================================================================
; Directory helpers.
; =========================================================================

; ensure the directory can hold one more entry; 0 ok / 2 OOM.
define internal i32 @roar_dir_reserve(ptr %r) #1 {
entry:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %capp, align 8
  %full = icmp uge i64 %count, %cap
  br i1 %full, label %grow, label %ok
grow:
  %z = icmp eq i64 %cap, 0
  %cap2 = shl i64 %cap, 1
  %newcap = select i1 %z, i64 4, i64 %cap2   ; cap <= 65536, fits i64
  %kbytes = shl i64 %newcap, 1
  %oldkeys = load ptr, ptr %r, align 8
  %nk = call ptr @realloc(ptr %oldkeys, i64 %kbytes)
  %nknull = icmp eq ptr %nk, null
  br i1 %nknull, label %oom, label %grow.c, !prof !0
grow.c:
  store ptr %nk, ptr %r, align 8
  %cbytes = shl i64 %newcap, 3
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %oldconts = load ptr, ptr %contsp, align 8
  %nc = call ptr @realloc(ptr %oldconts, i64 %cbytes)
  %ncnull = icmp eq ptr %nc, null
  br i1 %ncnull, label %oom, label %grow.d, !prof !0
grow.d:
  store ptr %nc, ptr %contsp, align 8
  store i64 %newcap, ptr %capp, align 8
  br label %ok
ok:
  ret i32 0
oom:
  ret i32 2
}

; append (key, cont) at the end (caller guarantees ascending order). 0/2.
define internal i32 @roar_dir_push(ptr %r, i32 %key, ptr %cont) #1 {
entry:
  %rc = call i32 @roar_dir_reserve(ptr %r)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %do, label %oom, !prof !1
do:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %count
  %key16 = trunc i32 %key to i16
  store i16 %key16, ptr %kp, align 2
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %count
  store ptr %cont, ptr %cp, align 8
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %countp, align 8
  ret i32 0
oom:
  ret i32 2
}

; insert (key, cont) at sorted position %pos (shift the tail). 0/2.
define internal i32 @roar_dir_insert(ptr %r, i64 %pos, i32 %key, ptr %cont) #1 {
entry:
  %rc = call i32 @roar_dir_reserve(ptr %r)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %do, label %oom, !prof !1
do:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %keys = load ptr, ptr %r, align 8
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %tail = sub i64 %count, %pos
  ; shift keys
  %ksrc = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %pos1 = add nuw i64 %pos, 1
  %kdst = getelementptr inbounds nuw i16, ptr %keys, i64 %pos1
  %ktbytes = shl i64 %tail, 1
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %ktbytes, i1 false)
  %key16 = trunc i32 %key to i16
  store i16 %key16, ptr %ksrc, align 2
  ; shift conts
  %csrc = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %cdst = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos1
  %ctbytes = shl i64 %tail, 3
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %ctbytes, i1 false)
  store ptr %cont, ptr %csrc, align 8
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %countp, align 8
  ret i32 0
oom:
  ret i32 2
}

; remove directory entry at %pos (does NOT free the container).
define internal void @roar_dir_erase(ptr %r, i64 %pos) #1 {
entry:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %keys = load ptr, ptr %r, align 8
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %pos1 = add nuw i64 %pos, 1
  %tail = sub i64 %count, %pos1
  %kdst = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %ksrc = getelementptr inbounds nuw i16, ptr %keys, i64 %pos1
  %ktbytes = shl i64 %tail, 1
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %ktbytes, i1 false)
  %cdst = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %csrc = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos1
  %ctbytes = shl i64 %tail, 3
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %ctbytes, i1 false)
  %count.n = add i64 %count, -1
  store i64 %count.n, ptr %countp, align 8
  ret void
}

; directory lower_bound over the u16 key array.
define internal i64 @roar_dir_lower_bound(ptr %r, i32 %key) #2 {
entry:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %keys = load ptr, ptr %r, align 8
  %pos = call i64 @roar_u16_lower_bound(ptr %keys, i64 %count, i32 %key)
  ret i64 %pos
}

; =========================================================================
; Public API.
; =========================================================================

define noalias ptr @universe_ds_roaring_create() local_unnamed_addr #1 {
entry:
  %r = call ptr @malloc(i64 32)
  %null = icmp eq ptr %r, null
  br i1 %null, label %fail, label %init, !prof !0
init:
  store ptr null, ptr %r, align 8           ; keys
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  store ptr null, ptr %contsp, align 8      ; conts
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  store i64 0, ptr %countp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %r, i64 24
  store i64 0, ptr %capp, align 8
  ret ptr %r
fail:
  ret ptr null
}

define void @universe_ds_roaring_destroy(ptr %r) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %done, label %pre, !prof !0
pre:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %some = icmp ne i64 %count, 0
  br i1 %some, label %loop, label %free.arrays
loop:
  %i = phi i64 [ 0, %pre ], [ %i.n, %loop ]
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %i
  %c = load ptr, ptr %cp, align 8
  call void @free(ptr %c)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %free.arrays
free.arrays:
  %keys = load ptr, ptr %r, align 8
  call void @free(ptr %keys)
  call void @free(ptr %conts)
  call void @free(ptr nonnull %r)
  br label %done
done:
  ret void
}

define i32 @universe_ds_roaring_add(ptr %r, i32 %v) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %err.null, label %split, !prof !0
err.null:
  ret i32 1
split:
  %key = lshr i32 %v, 16
  %low = and i32 %v, 65535
  %pos = call i64 @roar_dir_lower_bound(ptr %r, i32 %key)
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %inrange = icmp ult i64 %pos, %count
  br i1 %inrange, label %chk, label %newchunk
chk:
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %kv = load i16, ptr %kp, align 2
  %kv32 = zext i16 %kv to i32
  %hit = icmp eq i32 %kv32, %key
  br i1 %hit, label %existing, label %newchunk
existing:
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %slotp = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %rc = call i32 @roar_cont_add(ptr %slotp, i32 %low)
  ; 0 added, 9 dup (idempotent -> OK), 2 OOM
  %isoom = icmp eq i32 %rc, 2
  %ret = select i1 %isoom, i32 2, i32 0
  ret i32 %ret
newchunk:
  %nc = call ptr @roar_arr_new(i32 %low)
  %ncnull = icmp eq ptr %nc, null
  br i1 %ncnull, label %oom, label %ins, !prof !0
ins:
  %irc = call i32 @roar_dir_insert(ptr %r, i64 %pos, i32 %key, ptr %nc)
  %iok = icmp eq i32 %irc, 0
  br i1 %iok, label %ok, label %ins.oom, !prof !1
ins.oom:
  call void @free(ptr %nc)
  ret i32 2
ok:
  ret i32 0
oom:
  ret i32 2
}

define i32 @universe_ds_roaring_remove(ptr %r, i32 %v) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %err.null, label %split, !prof !0
err.null:
  ret i32 1
split:
  %key = lshr i32 %v, 16
  %low = and i32 %v, 65535
  %pos = call i64 @roar_dir_lower_bound(ptr %r, i32 %key)
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %inrange = icmp ult i64 %pos, %count
  br i1 %inrange, label %chk, label %absent
chk:
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %kv = load i16, ptr %kp, align 2
  %kv32 = zext i16 %kv to i32
  %hit = icmp eq i32 %kv32, %key
  br i1 %hit, label %existing, label %absent
existing:
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %slotp = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %rc = call i32 @roar_cont_remove(ptr %slotp, i32 %low)
  ; if the container is now empty, drop it from the directory.
  %c = load ptr, ptr %slotp, align 8
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %empty = icmp eq i32 %card, 0
  br i1 %empty, label %drop, label %absent
drop:
  call void @free(ptr %c)
  call void @roar_dir_erase(ptr %r, i64 %pos)
  br label %absent
absent:
  ret i32 0
}

define i32 @universe_ds_roaring_contains(ptr %r, i32 %v) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %no, label %split, !prof !0
no:
  ret i32 0
split:
  %key = lshr i32 %v, 16
  %low = and i32 %v, 65535
  %pos = call i64 @roar_dir_lower_bound(ptr %r, i32 %key)
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %inrange = icmp ult i64 %pos, %count
  br i1 %inrange, label %chk, label %no
chk:
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %kv = load i16, ptr %kp, align 2
  %kv32 = zext i16 %kv to i32
  %hit = icmp eq i32 %kv32, %key
  br i1 %hit, label %probe, label %no
probe:
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %c = load ptr, ptr %cp, align 8
  %r2 = call i32 @roar_cont_contains(ptr %c, i32 %low)
  ret i32 %r2
}

define i64 @universe_ds_roaring_cardinality(ptr %r) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %z, label %pre, !prof !0
z:
  ret i64 0
pre:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %some = icmp ne i64 %count, 0
  br i1 %some, label %loop, label %z
loop:
  %i = phi i64 [ 0, %pre ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %pre ], [ %acc.n, %loop ]
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %i
  %c = load ptr, ptr %cp, align 8
  %cardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %card = load i32, ptr %cardp, align 4
  %card64 = zext i32 %card to i64
  %acc.n = add i64 %acc, %card64
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

; ascending enumeration into out_u32[0..out_cap); returns TOTAL cardinality
; (values beyond out_cap are counted but not written).
define i64 @universe_ds_roaring_to_array(ptr %r, ptr %out, i64 %outcap) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %z, label %pre, !prof !0
z:
  ret i64 0
pre:
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %some = icmp ne i64 %count, 0
  br i1 %some, label %chunk, label %z
chunk:
  %ci = phi i64 [ 0, %pre ], [ %ci.n, %chunk.done ]
  %oidx = phi i64 [ 0, %pre ], [ %oidx.c, %chunk.done ]
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %ci
  %kv = load i16, ptr %kp, align 2
  %key64 = zext i16 %kv to i64
  %base = shl nuw i64 %key64, 16
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %ci
  %c = load ptr, ptr %cp, align 8
  %kind = load i32, ptr %c, align 8
  %isbm = icmp eq i32 %kind, 1
  br i1 %isbm, label %bmp, label %arr

; ---- array container enumeration ----
arr:
  %acardp = getelementptr inbounds nuw i8, ptr %c, i64 4
  %acard = load i32, ptr %acardp, align 4
  %acard64 = zext i32 %acard to i64
  %adata = getelementptr inbounds nuw i8, ptr %c, i64 16
  %aempty = icmp eq i64 %acard64, 0
  br i1 %aempty, label %chunk.done, label %arr.loop
arr.loop:
  %ai = phi i64 [ 0, %arr ], [ %ai.n, %arr.skip ]
  %aoidx = phi i64 [ %oidx, %arr ], [ %aoidx.n, %arr.skip ]
  %aep = getelementptr inbounds nuw i16, ptr %adata, i64 %ai
  %aev = load i16, ptr %aep, align 2
  %alow = zext i16 %aev to i64
  %aval = add i64 %base, %alow
  %awrite = icmp ult i64 %aoidx, %outcap
  br i1 %awrite, label %arr.store, label %arr.skip
arr.store:
  %aop = getelementptr inbounds nuw i32, ptr %out, i64 %aoidx
  %aval32 = trunc i64 %aval to i32
  store i32 %aval32, ptr %aop, align 4
  br label %arr.skip
arr.skip:
  %aoidx.n = add nuw i64 %aoidx, 1
  %ai.n = add nuw i64 %ai, 1
  %amore = icmp ult i64 %ai.n, %acard64
  br i1 %amore, label %arr.loop, label %arr.exit
arr.exit:
  br label %chunk.done.arr

; ---- bitmap container enumeration ----
bmp:
  %words = getelementptr inbounds nuw i8, ptr %c, i64 64
  br label %bmp.word
bmp.word:
  %wi = phi i64 [ 0, %bmp ], [ %wi.n, %bmp.word.cont ]
  %boidx = phi i64 [ %oidx, %bmp ], [ %boidx.w, %bmp.word.cont ]
  %wp = getelementptr inbounds nuw i64, ptr %words, i64 %wi
  %word = load i64, ptr %wp, align 8
  %wbase = shl i64 %wi, 6
  %wbase2 = add i64 %base, %wbase
  br label %bmp.bits
bmp.bits:
  %w = phi i64 [ %word, %bmp.word ], [ %w.n, %bmp.emit.done ]
  %eoidx = phi i64 [ %boidx, %bmp.word ], [ %eoidx.n, %bmp.emit.done ]
  %nz = icmp ne i64 %w, 0
  br i1 %nz, label %bmp.emit, label %bmp.bits.exit
bmp.emit:
  %tz = call i64 @llvm.cttz.i64(i64 %w, i1 true)
  %bval = add i64 %wbase2, %tz
  %bwrite = icmp ult i64 %eoidx, %outcap
  br i1 %bwrite, label %bmp.store, label %bmp.emit.done
bmp.store:
  %bop = getelementptr inbounds nuw i32, ptr %out, i64 %eoidx
  %bval32 = trunc i64 %bval to i32
  store i32 %bval32, ptr %bop, align 4
  br label %bmp.emit.done
bmp.emit.done:
  %eoidx.n = add nuw i64 %eoidx, 1
  %wm1 = sub i64 %w, 1
  %w.n = and i64 %w, %wm1
  br label %bmp.bits
bmp.bits.exit:
  br label %bmp.word.cont
bmp.word.cont:
  %boidx.w = phi i64 [ %eoidx, %bmp.bits.exit ]
  %wi.n = add nuw i64 %wi, 1
  %wmore = icmp ult i64 %wi.n, 1024
  br i1 %wmore, label %bmp.word, label %bmp.exit
bmp.exit:
  br label %chunk.done.bmp

; merge the two enumeration exits into a single successor with the updated idx.
chunk.done.arr:
  br label %chunk.done
chunk.done.bmp:
  br label %chunk.done
chunk.done:
  %oidx.c = phi i64 [ %oidx, %arr ], [ %aoidx.n, %chunk.done.arr ], [ %boidx.w, %chunk.done.bmp ]
  %ci.n = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, %count
  br i1 %cmore, label %chunk, label %done
done:
  ret i64 %oidx.c
}

define i32 @universe_ds_roaring_container_kind(ptr %r, i32 %v) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %r, null
  br i1 %null, label %none, label %split, !prof !0
none:
  ret i32 -1
split:
  %key = lshr i32 %v, 16
  %pos = call i64 @roar_dir_lower_bound(ptr %r, i32 %key)
  %countp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %count = load i64, ptr %countp, align 8
  %inrange = icmp ult i64 %pos, %count
  br i1 %inrange, label %chk, label %none
chk:
  %keys = load ptr, ptr %r, align 8
  %kp = getelementptr inbounds nuw i16, ptr %keys, i64 %pos
  %kv = load i16, ptr %kp, align 2
  %kv32 = zext i16 %kv to i32
  %hit = icmp eq i32 %kv32, %key
  br i1 %hit, label %probe, label %none
probe:
  %contsp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %conts = load ptr, ptr %contsp, align 8
  %cp = getelementptr inbounds nuw ptr, ptr %conts, i64 %pos
  %c = load ptr, ptr %cp, align 8
  %kind = load i32, ptr %c, align 8
  ret i32 %kind
}

; =========================================================================
; Set algebra. op selector: 0 union, 1 intersection, 2 difference (a\b).
; =========================================================================

define internal ptr @roar_setalg(ptr %a, ptr %b, i32 %op) #1 {
entry:
  %sA = alloca [1024 x i64], align 16
  %sB = alloca [1024 x i64], align 16
  %sR = alloca [1024 x i64], align 16
  %anull = icmp eq ptr %a, null
  %bnull = icmp eq ptr %b, null
  %bad = or i1 %anull, %bnull
  br i1 %bad, label %fail, label %mk, !prof !0
mk:
  %r = call ptr @universe_ds_roaring_create()
  %rnull = icmp eq ptr %r, null
  br i1 %rnull, label %fail, label %setup, !prof !0
setup:
  %acountp = getelementptr inbounds nuw i8, ptr %a, i64 16
  %acount = load i64, ptr %acountp, align 8
  %bcountp = getelementptr inbounds nuw i8, ptr %b, i64 16
  %bcount = load i64, ptr %bcountp, align 8
  %akeys = load ptr, ptr %a, align 8
  %bkeys = load ptr, ptr %b, align 8
  %acontsp = getelementptr inbounds nuw i8, ptr %a, i64 8
  %aconts = load ptr, ptr %acontsp, align 8
  %bcontsp = getelementptr inbounds nuw i8, ptr %b, i64 8
  %bconts = load ptr, ptr %bcontsp, align 8
  br label %merge

merge:
  %i = phi i64 [ 0, %setup ], [ %ni.a, %onlyA.next ], [ %i, %onlyB.next ], [ %ni.e, %eq.next ]
  %j = phi i64 [ 0, %setup ], [ %j, %onlyA.next ], [ %nj.b, %onlyB.next ], [ %nj.e, %eq.next ]
  %ia = icmp ult i64 %i, %acount
  %ib = icmp ult i64 %j, %bcount
  %both = and i1 %ia, %ib
  br i1 %both, label %cmp, label %tails

cmp:
  %akp = getelementptr inbounds nuw i16, ptr %akeys, i64 %i
  %akv = load i16, ptr %akp, align 2
  %aku = zext i16 %akv to i32
  %bkp = getelementptr inbounds nuw i16, ptr %bkeys, i64 %j
  %bkv = load i16, ptr %bkp, align 2
  %bku = zext i16 %bkv to i32
  %alt = icmp ult i32 %aku, %bku
  %agt = icmp ugt i32 %aku, %bku
  br i1 %alt, label %onlyA, label %cmp2
cmp2:
  br i1 %agt, label %onlyB, label %eqkey

; a-only key: union & difference keep it; intersection drops it.
onlyA:
  %keepA = icmp ne i32 %op, 1
  br i1 %keepA, label %onlyA.clone, label %onlyA.next
onlyA.clone:
  %acp = getelementptr inbounds nuw ptr, ptr %aconts, i64 %i
  %ac = load ptr, ptr %acp, align 8
  %acl = call ptr @roar_cont_clone(ptr %ac)
  %aclnull = icmp eq ptr %acl, null
  br i1 %aclnull, label %abort, label %onlyA.push, !prof !0
onlyA.push:
  %aprc = call i32 @roar_dir_push(ptr %r, i32 %aku, ptr %acl)
  %apok = icmp eq i32 %aprc, 0
  br i1 %apok, label %onlyA.next, label %abort.freeacl, !prof !1
abort.freeacl:
  call void @free(ptr %acl)
  br label %abort
onlyA.next:
  %ni.a = add nuw i64 %i, 1
  br label %merge

; b-only key: only union keeps it.
onlyB:
  %keepB = icmp eq i32 %op, 0
  br i1 %keepB, label %onlyB.clone, label %onlyB.next
onlyB.clone:
  %bcp = getelementptr inbounds nuw ptr, ptr %bconts, i64 %j
  %bc = load ptr, ptr %bcp, align 8
  %bcl = call ptr @roar_cont_clone(ptr %bc)
  %bclnull = icmp eq ptr %bcl, null
  br i1 %bclnull, label %abort, label %onlyB.push, !prof !0
onlyB.push:
  %bprc = call i32 @roar_dir_push(ptr %r, i32 %bku, ptr %bcl)
  %bpok = icmp eq i32 %bprc, 0
  br i1 %bpok, label %onlyB.next, label %abort.freebcl, !prof !1
abort.freebcl:
  call void @free(ptr %bcl)
  br label %abort
onlyB.next:
  %nj.b = add nuw i64 %j, 1
  br label %merge

; equal keys: combine containers through scratch, normalize, append.
eqkey:
  %eacp = getelementptr inbounds nuw ptr, ptr %aconts, i64 %i
  %eac = load ptr, ptr %eacp, align 8
  %ebcp = getelementptr inbounds nuw ptr, ptr %bconts, i64 %j
  %ebc = load ptr, ptr %ebcp, align 8
  %wa = call ptr @roar_cont_expand(ptr %eac, ptr %sA)
  %wb = call ptr @roar_cont_expand(ptr %ebc, ptr %sB)
  switch i32 %op, label %comb.or [ i32 1, label %comb.and
                                   i32 2, label %comb.andnot ]
comb.or:
  call void @universe_ds_roaring_bitmap_or(ptr %sR, ptr %wa, ptr %wb)
  br label %comb.norm
comb.and:
  call void @universe_ds_roaring_bitmap_and(ptr %sR, ptr %wa, ptr %wb)
  br label %comb.norm
comb.andnot:
  call void @universe_ds_roaring_bitmap_andnot(ptr %sR, ptr %wa, ptr %wb)
  br label %comb.norm
comb.norm:
  %card64 = call i64 @roar_words_popcount(ptr %sR)
  %card32 = trunc i64 %card64 to i32
  %nc = call ptr @roar_normalize(ptr %sR, i32 %card32)
  %ncnull = icmp eq ptr %nc, null
  br i1 %ncnull, label %eq.next, label %eq.push
eq.push:
  %eprc = call i32 @roar_dir_push(ptr %r, i32 %aku, ptr %nc)
  %epok = icmp eq i32 %eprc, 0
  br i1 %epok, label %eq.next, label %abort.freenc, !prof !1
abort.freenc:
  call void @free(ptr %nc)
  br label %abort
eq.next:
  %ni.e = add nuw i64 %i, 1
  %nj.e = add nuw i64 %j, 1
  br label %merge

; -------- tails: append whatever remains, honoring the op --------
tails:
  ; union/difference: drain remaining A. union: also drain remaining B.
  %drainA = icmp ne i32 %op, 1
  br i1 %drainA, label %tailA.head, label %tailB.check
tailA.head:
  %ti = phi i64 [ %i, %tails ], [ %ti.n, %tailA.cont ]
  %tia = icmp ult i64 %ti, %acount
  br i1 %tia, label %tailA.body, label %tailB.check
tailA.body:
  %takp = getelementptr inbounds nuw i16, ptr %akeys, i64 %ti
  %takv = load i16, ptr %takp, align 2
  %taku = zext i16 %takv to i32
  %tacp = getelementptr inbounds nuw ptr, ptr %aconts, i64 %ti
  %tac = load ptr, ptr %tacp, align 8
  %tacl = call ptr @roar_cont_clone(ptr %tac)
  %taclnull = icmp eq ptr %tacl, null
  br i1 %taclnull, label %abort, label %tailA.push, !prof !0
tailA.push:
  %taprc = call i32 @roar_dir_push(ptr %r, i32 %taku, ptr %tacl)
  %tapok = icmp eq i32 %taprc, 0
  br i1 %tapok, label %tailA.cont, label %abort.freetacl, !prof !1
abort.freetacl:
  call void @free(ptr %tacl)
  br label %abort
tailA.cont:
  %ti.n = add nuw i64 %ti, 1
  br label %tailA.head

tailB.check:
  %drainB = icmp eq i32 %op, 0
  br i1 %drainB, label %tailB.head, label %success
tailB.head:
  %tj = phi i64 [ %j, %tailB.check ], [ %tj.n, %tailB.cont ]
  %tjb = icmp ult i64 %tj, %bcount
  br i1 %tjb, label %tailB.body, label %success
tailB.body:
  %tbkp = getelementptr inbounds nuw i16, ptr %bkeys, i64 %tj
  %tbkv = load i16, ptr %tbkp, align 2
  %tbku = zext i16 %tbkv to i32
  %tbcp = getelementptr inbounds nuw ptr, ptr %bconts, i64 %tj
  %tbc = load ptr, ptr %tbcp, align 8
  %tbcl = call ptr @roar_cont_clone(ptr %tbc)
  %tbclnull = icmp eq ptr %tbcl, null
  br i1 %tbclnull, label %abort, label %tailB.push, !prof !0
tailB.push:
  %tbprc = call i32 @roar_dir_push(ptr %r, i32 %tbku, ptr %tbcl)
  %tbpok = icmp eq i32 %tbprc, 0
  br i1 %tbpok, label %tailB.cont, label %abort.freetbcl, !prof !1
abort.freetbcl:
  call void @free(ptr %tbcl)
  br label %abort
tailB.cont:
  %tj.n = add nuw i64 %tj, 1
  br label %tailB.head

success:
  ret ptr %r
abort:
  call void @universe_ds_roaring_destroy(ptr %r)
  br label %fail
fail:
  ret ptr null
}

define ptr @universe_ds_roaring_union(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @roar_setalg(ptr %a, ptr %b, i32 0)
  ret ptr %r
}

define ptr @universe_ds_roaring_intersection(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @roar_setalg(ptr %a, ptr %b, i32 1)
  ret ptr %r
}

define ptr @universe_ds_roaring_difference(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @roar_setalg(ptr %a, ptr %b, i32 2)
  ret ptr %r
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
