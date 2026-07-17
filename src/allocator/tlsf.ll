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

; TLSF — Two-Level Segregated Fit allocator: O(1) worst-case alloc & free with
; bounded fragmentation. General-purpose (arbitrary sizes, real free & reuse),
; unlike arena/pool/buddy which trade generality for a narrower fast path.
; This is the SDK's DEFAULT/PRIMARY malloc replacement, so it GROWS: when the
; managed region(s) cannot satisfy a request it mallocs a NEW backing region,
; links it in, and serves from it — the cold fallback; the common path is O(1).
;
; DESIGN (from first principles; not ported from any source):
;   * SEGREGATED FREE LISTS indexed by a two-level key. First level FL =
;     floor(log2(size)) (via ctlz); second level SL subdivides each power-of-two
;     class LINEARLY into 2^SLI ranges (SLI=4 -> 16 ranges). A free block of
;     size s lands in list (FL(s)-FL_MIN, SL(s)). Constants:
;       FL_MIN=5 (min block 32B), SLI=4, SL_COUNT=16, FL_COUNT=32.
;   * O(1) FIND: an FL bitmap (one i64, bit fl set iff any SL list under fl is
;     nonempty) plus per-FL SL bitmaps (i32 each). A request rounds UP so the
;     smallest block in the chosen (fl,sl) class already satisfies it, then two
;     cttz's locate the first nonempty class >= that key. No list scan ever.
;   * O(1) COALESCE: every block header carries a PREV_PHYS pointer and its
;     size; two flag bits (bit0 BLOCK_FREE, bit1 PREV_FREE) live in the low bits
;     of the 16-aligned size word. On free, the next physical block (blk+size)
;     is inspected for BLOCK_FREE and the PREV_FREE bit tells us the previous
;     physical block is free (reached via PREV_PHYS) — merge either/both in O(1)
;     and keep the "no two adjacent free blocks" invariant.
;   * MULTI-REGION GROWTH (the malloc-replacement enabler):
;       - The control block (FL/SL bitmaps + head table + bookkeeping) lives once
;         at the FRONT of the FIRST region; all regions share it. Every region
;         (region 0 and each grown region) reserves a used SENTINEL block header
;         at its very end and starts its pool with a single free block. The first
;         block of a region has prev_phys=null and PREV_FREE=0; the trailing
;         sentinel is a zero-size USED block. Together these BOUND coalescing to
;         WITHIN a region: mergeprev stops at the null-prev first block, mergenext
;         stops at the used sentinel. Because a sentinel always follows the last
;         real block, the "next physical block" is ALWAYS a valid header — no
;         pool_end comparison is needed on the hot path (removed).
;       - GROWTH is AUTOMATIC when the allocator OWNS its memory (created with
;         base==null => owns_memory=1). The OWNED initial region we malloc is
;         floored at ALLOC_OS_MIN=16384 (16 KiB) — os_size = max(size, 16384) —
;         so a tiny owned create still obtains a >= 16 KiB backing region; a
;         caller-supplied region (base!=null) is EXEMPT (the caller sizes it).
;         On an alloc that find() cannot satisfy,
;         grow() mallocs a region of size max(need+overhead, current_total,
;         1 MiB floor) capped at 2^36, carves a 16B RegionHeader {next, base} at
;         its front, links it into the region list, seeds its pool, and the alloc
;         retries find(). A CALLER-SUPPLIED region (base!=null => owns_memory=0)
;         is FIXED and NEVER grows: alloc returns null on exhaustion, exactly as
;         before. (A caller wanting a growable heap passes base==null.)
;   * Block header (16B, payload 16-aligned at +16):
;       +0  ptr prev_phys      (null for the first block of a region)
;       +8  i64 size_and_flags (total block size incl header | flags)
;     free blocks additionally use payload words:
;       +16 ptr next_free      (null sentinel)
;       +24 ptr prev_free
;     min block = 32B (header + two link words). Sentinel = 16B used block.
;   * Control block layout (offsets, size 4288 = CONTROL_SIZE):
;       +0   i64 fl_bitmap
;       +8   i64 live               (sum of block payload bytes of live blocks)
;       +16  i64 reserved           (was pool_end; unused since sentinels)
;       +24  i64 owns_memory        (1 => region 0 malloc'd + growable)
;       +32  ptr pool_start         (region 0's first pool block)
;       +40  ptr base               (== handle; region 0's malloc ptr when owned)
;       +48  ptr region_head        (list of grown RegionHeaders, null-terminated)
;       +56  i64 total_size         (sum of all region byte sizes; geometric grow)
;       +64  i32 sl_bitmap[32]      (128B, ends +192)
;       +192 ptr heads[512]         (32*16 pointers, 4096B, ends +4288)
;   * RegionHeader (grown regions only, 16B at region front):
;       +0   ptr next   (next RegionHeader or null)
;       +8   ptr base   (== the malloc ptr, freed on destroy)
;   * _live returns the summed block-PAYLOAD bytes (block_size-16) of the
;     currently-live allocations across ALL regions — deterministic.
;   * destroy walks region_head freeing every grown region, then frees region 0
;     iff owns_memory==1; a caller-supplied region 0 is never freed.
;
; ORDERINGS: single-threaded structure; no atomics (a concurrent TLSF would need
;   a striped or lock-guarded front — out of scope here).
;
; API (C ABI, nounwind):
;   ptr  universe_alloc_tlsf_create(ptr base, i64 size)  ; null on bad size/OOM
;   ptr  universe_alloc_tlsf_alloc(ptr h, i64 size)      ; 16-aligned; grows if owned
;   ptr  universe_alloc_tlsf_alloc_aligned(ptr h, i64 size, i64 align) ; align=pow2
;   void universe_alloc_tlsf_free(ptr h, ptr p)
;   i64  universe_alloc_tlsf_live(ptr h)
;   void universe_alloc_tlsf_destroy(ptr h)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i64 @llvm.cttz.i64(i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; ---- tiny address / math helpers (memory(none) or pure) ------------------

; floor(log2(x)) for x>=1
define internal i64 @fls(i64 %x) #3 {
entry:
  %lz = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %r = sub i64 63, %lz
  ret i64 %r
}

; address of free-list head for (fl,sl): heads[fl*16+sl] at +192
define internal ptr @hslot(ptr %h, i64 %fl, i64 %sl) #3 {
entry:
  %m = mul i64 %fl, 16
  %idx = add i64 %m, %sl
  %m2 = mul i64 %idx, 8
  %off = add i64 %m2, 192
  %p = getelementptr inbounds nuw i8, ptr %h, i64 %off
  ret ptr %p
}

; address of sl_bitmap[fl] (i32) at +64
define internal ptr @slslot(ptr %h, i64 %fl) #3 {
entry:
  %m = mul i64 %fl, 4
  %off = add i64 %m, 64
  %p = getelementptr inbounds nuw i8, ptr %h, i64 %off
  ret ptr %p
}

; ---- free-list insert / remove -------------------------------------------

; insert free block `blk` of total size `bsize` into its (fl,sl) list; set bits.
define internal void @ins(ptr %h, ptr %blk, i64 %bsize) #0 {
entry:
  %b = call i64 @fls(i64 %bsize)
  %fl0 = sub i64 %b, 5
  %fl = call i64 @llvm.umin.i64(i64 %fl0, i64 31)
  %sh = sub i64 %b, 4
  %shf = lshr i64 %bsize, %sh
  %sl = and i64 %shf, 15
  %hp = call ptr @hslot(ptr %h, i64 %fl, i64 %sl)
  %old = load ptr, ptr %hp, align 8
  %nfp = getelementptr inbounds nuw i8, ptr %blk, i64 16
  store ptr %old, ptr %nfp, align 8
  %pfp = getelementptr inbounds nuw i8, ptr %blk, i64 24
  store ptr null, ptr %pfp, align 8
  %oldnull = icmp eq ptr %old, null
  br i1 %oldnull, label %sethead, label %backlink

backlink:
  %oldpfp = getelementptr inbounds nuw i8, ptr %old, i64 24
  store ptr %blk, ptr %oldpfp, align 8
  br label %sethead

sethead:
  store ptr %blk, ptr %hp, align 8
  %slp = call ptr @slslot(ptr %h, i64 %fl)
  %slv = load i32, ptr %slp, align 4
  %sl32 = trunc i64 %sl to i32
  %slbit = shl i32 1, %sl32
  %sln = or i32 %slv, %slbit
  store i32 %sln, ptr %slp, align 4
  %flv = load i64, ptr %h, align 8
  %flbit = shl i64 1, %fl
  %fln = or i64 %flv, %flbit
  store i64 %fln, ptr %h, align 8
  ret void
}

; remove free block `blk` from its list; clear bits if the list becomes empty.
define internal void @rem(ptr %h, ptr %blk) #0 {
entry:
  %sfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  %sf = load i64, ptr %sfp, align 8
  %bsize = and i64 %sf, -16
  %b = call i64 @fls(i64 %bsize)
  %fl0 = sub i64 %b, 5
  %fl = call i64 @llvm.umin.i64(i64 %fl0, i64 31)
  %sh = sub i64 %b, 4
  %shf = lshr i64 %bsize, %sh
  %sl = and i64 %shf, 15
  %hp = call ptr @hslot(ptr %h, i64 %fl, i64 %sl)
  %nfp = getelementptr inbounds nuw i8, ptr %blk, i64 16
  %next = load ptr, ptr %nfp, align 8
  %pfp = getelementptr inbounds nuw i8, ptr %blk, i64 24
  %prev = load ptr, ptr %pfp, align 8
  %prevnull = icmp eq ptr %prev, null
  br i1 %prevnull, label %headupd, label %prevupd

prevupd:
  %prevnf = getelementptr inbounds nuw i8, ptr %prev, i64 16
  store ptr %next, ptr %prevnf, align 8
  br label %after

headupd:
  store ptr %next, ptr %hp, align 8
  br label %after

after:
  %nextnull = icmp eq ptr %next, null
  br i1 %nextnull, label %chkempty, label %nextupd

nextupd:
  %nextpf = getelementptr inbounds nuw i8, ptr %next, i64 24
  store ptr %prev, ptr %nextpf, align 8
  br label %chkempty

chkempty:
  %head = load ptr, ptr %hp, align 8
  %empty = icmp eq ptr %head, null
  br i1 %empty, label %clearbits, label %done

clearbits:
  %slp = call ptr @slslot(ptr %h, i64 %fl)
  %slv = load i32, ptr %slp, align 4
  %sl32 = trunc i64 %sl to i32
  %slbit = shl i32 1, %sl32
  %slbitn = xor i32 %slbit, -1
  %sln = and i32 %slv, %slbitn
  store i32 %sln, ptr %slp, align 4
  %slzero = icmp eq i32 %sln, 0
  br i1 %slzero, label %clearfl, label %done

clearfl:
  %flv = load i64, ptr %h, align 8
  %flbit = shl i64 1, %fl
  %flbitn = xor i64 %flbit, -1
  %fln = and i64 %flv, %flbitn
  store i64 %fln, ptr %h, align 8
  br label %done

done:
  ret void
}

; ---- O(1) find suitable free block ---------------------------------------

; return the head block of the first (fl,sl) class whose smallest member
; satisfies `need` (need is a total block size, >=32, multiple of 16), or null.
define internal ptr @find(ptr %h, i64 %need) #0 {
entry:
  ; search mapping: round up so the class lower bound >= need
  %b = call i64 @fls(i64 %need)
  %rsh = sub i64 %b, 4
  %one = shl i64 1, %rsh
  %round = add i64 %one, -1
  %need2 = add i64 %need, %round
  %b2 = call i64 @fls(i64 %need2)
  %fl0a = sub i64 %b2, 5
  %fl0 = call i64 @llvm.umin.i64(i64 %fl0a, i64 31)
  %sh2 = sub i64 %b2, 4
  %shf = lshr i64 %need2, %sh2
  %sl0 = and i64 %shf, 15
  ; SL bits >= sl0 in this fl
  %slp0 = call ptr @slslot(ptr %h, i64 %fl0)
  %slbm32 = load i32, ptr %slp0, align 4
  %slbm = zext i32 %slbm32 to i64
  %hishift = shl i64 -1, %sl0
  %slmasked = and i64 %slbm, %hishift
  %slmap = and i64 %slmasked, 65535
  %has = icmp ne i64 %slmap, 0
  br i1 %has, label %samefl, label %higher

samefl:
  %slA = call i64 @llvm.cttz.i64(i64 %slmap, i1 true)
  br label %found

higher:
  %flbm = load i64, ptr %h, align 8
  %fl0p1 = add i64 %fl0, 1
  %flshift = shl i64 -1, %fl0p1
  %flmap = and i64 %flbm, %flshift
  %none = icmp eq i64 %flmap, 0
  br i1 %none, label %retnull, label %pick

pick:
  %flB = call i64 @llvm.cttz.i64(i64 %flmap, i1 true)
  %slpB = call ptr @slslot(ptr %h, i64 %flB)
  %slbmB32 = load i32, ptr %slpB, align 4
  %slbmB = and i32 %slbmB32, 65535
  %slbmB64 = zext i32 %slbmB to i64
  %slB = call i64 @llvm.cttz.i64(i64 %slbmB64, i1 true)
  br label %found

found:
  %fl = phi i64 [ %fl0, %samefl ], [ %flB, %pick ]
  %sl = phi i64 [ %slA, %samefl ], [ %slB, %pick ]
  %hp = call ptr @hslot(ptr %h, i64 %fl, i64 %sl)
  %blk = load ptr, ptr %hp, align 8
  ret ptr %blk

retnull:
  ret ptr null
}

; ---- pool seeding (region setup) -----------------------------------------

; seed a pool [pstart, pstart+psize) with ONE free block + a trailing used
; sentinel, and insert the free block. psize must be a multiple of 16, >=48,
; pstart 16-aligned. The sentinel bounds coalescing at the region's high edge.
define internal void @seed_pool(ptr %h, ptr %pstart, i64 %psize) #0 {
entry:
  %freesize = sub i64 %psize, 16
  store ptr null, ptr %pstart, align 8            ; free.prev_phys = null (region low edge)
  %fsp = getelementptr inbounds nuw i8, ptr %pstart, i64 8
  %fsf = or i64 %freesize, 1                        ; free, PREV_FREE=0
  store i64 %fsf, ptr %fsp, align 8
  %sent = getelementptr inbounds nuw i8, ptr %pstart, i64 %freesize
  store ptr %pstart, ptr %sent, align 8            ; sentinel.prev_phys = free block
  %ssp = getelementptr inbounds nuw i8, ptr %sent, i64 8
  store i64 2, ptr %ssp, align 8                    ; size 0, USED (bit0=0), PREV_FREE=1
  call void @ins(ptr %h, ptr %pstart, i64 %freesize)
  ret void
}

; ---- growth: malloc a new region and seed it (COLD fallback) -------------

; If the allocator owns its memory, malloc a fresh region big enough for `need`
; (a total block size), link it, and seed its pool. Returns the region ptr, or
; null if not growable / OOM / need too large. Cold; not on the alloc fast path.
define internal ptr @grow(ptr %h, i64 %need) #1 {
entry:
  %ownsp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %owns = load i64, ptr %ownsp, align 8
  %growable = icmp eq i64 %owns, 1
  br i1 %growable, label %cont, label %retnull

cont:
  ; TLSF is a GOOD-fit: find() rounds the request UP to the next SL subclass
  ; boundary and only accepts a block whose class lower-bound >= that. So the
  ; seeded free block must be >= that rounded value, else a block barely larger
  ; than `need` lands in a LOWER subclass and find() misses it. Round `need` the
  ; same way find() does before sizing.
  %fb = call i64 @fls(i64 %need)
  %rsh = sub i64 %fb, 4
  %oneR = shl i64 1, %rsh
  %rup = add i64 %oneR, -1
  %needR = add i64 %need, %rup
  ; overhead = RegionHeader(16) + sentinel(16) + align slack(16) = 48
  %o1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %needR, i64 48)
  %need48 = extractvalue { i64, i1 } %o1, 0
  %ov1 = extractvalue { i64, i1 } %o1, 1
  %toobig = icmp ugt i64 %need48, 68719476736       ; keep seeded block < 2^36
  %bad = or i1 %ov1, %toobig
  br i1 %bad, label %retnull, label %size

size:
  %totp = getelementptr inbounds nuw i8, ptr %h, i64 56
  %tot = load i64, ptr %totp, align 8
  %w1 = call i64 @llvm.umax.i64(i64 %need48, i64 %tot)      ; geometric: ~current total
  %w2 = call i64 @llvm.umax.i64(i64 %w1, i64 1048576)       ; 1 MiB floor
  %rsize = call i64 @llvm.umin.i64(i64 %w2, i64 68719476736) ; cap 2^36
  %m = call ptr @malloc(i64 %rsize)
  %mnull = icmp eq ptr %m, null
  br i1 %mnull, label %retnull, label %setup

setup:
  ; RegionHeader at m front: +0 next (old head), +8 base (== m)
  %rhp = getelementptr inbounds nuw i8, ptr %h, i64 48
  %oldhead = load ptr, ptr %rhp, align 8
  store ptr %oldhead, ptr %m, align 8               ; region.next = old head
  %rbasep = getelementptr inbounds nuw i8, ptr %m, i64 8
  store ptr %m, ptr %rbasep, align 8                ; region.base = m
  store ptr %m, ptr %rhp, align 8                   ; region_head = m
  %totn = add i64 %tot, %rsize
  store i64 %totn, ptr %totp, align 8
  ; pool start = align_up(m + 16, 16)
  %mi = ptrtoint ptr %m to i64
  %he = add i64 %mi, 16
  %he15 = add i64 %he, 15
  %heal = and i64 %he15, -16
  %delta = sub i64 %heal, %mi
  %pstart = getelementptr inbounds nuw i8, ptr %m, i64 %delta
  %pbytes = sub i64 %rsize, %delta
  %psize = and i64 %pbytes, -16
  call void @seed_pool(ptr %h, ptr %pstart, i64 %psize)
  ret ptr %m

retnull:
  ret ptr null
}

; find a block for `need`; on miss, grow once and retry. Returns null only when
; the allocator is fixed (caller region) & full, or growth failed.
define internal ptr @find_or_grow(ptr %h, i64 %need) #1 {
entry:
  %b1 = call ptr @find(ptr %h, i64 %need)
  %ok1 = icmp ne ptr %b1, null
  br i1 %ok1, label %ret, label %trygrow, !prof !1

trygrow:
  %g = call ptr @grow(ptr %h, i64 %need)
  %gok = icmp ne ptr %g, null
  br i1 %gok, label %research, label %ret2

research:
  %b2 = call ptr @find(ptr %h, i64 %need)
  br label %ret

ret:
  %blk = phi ptr [ %b1, %entry ], [ %b2, %research ]
  ret ptr %blk

ret2:
  ret ptr null
}

; ---- place a removed block: split trailing remainder, account, return payload

; `blk` has been removed from its free list. `bsize` is its total size, `need`
; the requested total, `prevfree` the PREV_FREE bit to preserve on blk. Returns
; the user payload pointer (blk+16). The next physical block always exists (a
; real block or the region sentinel), so its back-links are always updated.
define internal ptr @place(ptr %h, ptr %blk, i64 %bsize, i64 %need, i64 %prevfree) #0 {
entry:
  %rem = sub i64 %bsize, %need
  %dosplit = icmp uge i64 %rem, 32
  %sfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  br i1 %dosplit, label %split, label %whole

split:
  %sf1 = or i64 %need, %prevfree
  store i64 %sf1, ptr %sfp, align 8              ; blk used, size=need
  %r = getelementptr inbounds nuw i8, ptr %blk, i64 %need
  store ptr %blk, ptr %r, align 8                ; r.prev_phys = blk
  %rsfp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %rsf = or i64 %rem, 1                            ; remainder free, PREV_FREE=0
  store i64 %rsf, ptr %rsfp, align 8
  call void @ins(ptr %h, ptr %r, i64 %rem)
  %nb = getelementptr inbounds nuw i8, ptr %blk, i64 %bsize
  store ptr %r, ptr %nb, align 8                 ; nb.prev_phys = r
  %nbsfp = getelementptr inbounds nuw i8, ptr %nb, i64 8
  %nbsf = load i64, ptr %nbsfp, align 8
  %nbsf2 = or i64 %nbsf, 2                         ; r is free -> PREV_FREE
  store i64 %nbsf2, ptr %nbsfp, align 8
  br label %finish

whole:
  %sf2 = or i64 %bsize, %prevfree
  store i64 %sf2, ptr %sfp, align 8              ; blk used, whole block
  %nbw = getelementptr inbounds nuw i8, ptr %blk, i64 %bsize
  store ptr %blk, ptr %nbw, align 8             ; nb.prev_phys = blk
  %nbwsfp = getelementptr inbounds nuw i8, ptr %nbw, i64 8
  %nbwsf = load i64, ptr %nbwsfp, align 8
  %clr = and i64 %nbwsf, -3                        ; blk used -> clear PREV_FREE
  store i64 %clr, ptr %nbwsfp, align 8
  br label %finish

finish:
  %used = phi i64 [ %need, %split ], [ %bsize, %whole ]
  %livep = getelementptr inbounds nuw i8, ptr %h, i64 8
  %live = load i64, ptr %livep, align 8
  %pay = sub i64 %used, 16
  %liven = add i64 %live, %pay
  store i64 %liven, ptr %livep, align 8
  %ret = getelementptr inbounds nuw i8, ptr %blk, i64 16
  ret ptr %ret
}

; ---- exported API ---------------------------------------------------------

define noalias ptr @universe_alloc_tlsf_create(ptr %base, i64 %size) local_unnamed_addr #1 {
entry:
  %toosmall = icmp ult i64 %size, 4336             ; control(4288) + sentinel(16) + min(32)
  %toobig = icmp ugt i64 %size, 68719476736        ; 2^36 cap keeps FL index < 32
  %bad = or i1 %toosmall, %toobig
  br i1 %bad, label %fail, label %ownck, !prof !0

ownck:
  %ownsb = icmp eq ptr %base, null
  br i1 %ownsb, label %domalloc, label %haveregion

domalloc:
  ; OS-request floor: an OWNED initial region we malloc is at least ALLOC_OS_MIN
  ; (16 KiB). A caller-supplied region (haveregion) is EXEMPT — the caller sizes
  ; it. The validated request (%size >= 4336) is raised, never lowered, so the
  ; toosmall/toobig guards above still hold and no wrap is possible.
  %ossize = call i64 @llvm.umax.i64(i64 %size, i64 16384)
  %m = call ptr @malloc(i64 %ossize)
  %mnull = icmp eq ptr %m, null
  br i1 %mnull, label %fail, label %setup, !prof !0

haveregion:
  br label %setup

setup:
  %reg = phi ptr [ %m, %domalloc ], [ %base, %haveregion ]
  %owns = phi i64 [ 1, %domalloc ], [ 0, %haveregion ]
  %effsize = phi i64 [ %ossize, %domalloc ], [ %size, %haveregion ]
  call void @llvm.memset.p0.i64(ptr %reg, i8 0, i64 4288, i1 false)
  %regi = ptrtoint ptr %reg to i64
  %ce = add i64 %regi, 4288
  %ce15 = add i64 %ce, 15
  %ceal = and i64 %ce15, -16
  %delta = sub i64 %ceal, %regi
  %pstart = getelementptr inbounds nuw i8, ptr %reg, i64 %delta
  %endi = add i64 %regi, %effsize
  %psize.raw = sub i64 %endi, %ceal
  %psize = and i64 %psize.raw, -16
  %psmall = icmp ult i64 %psize, 48                ; need sentinel(16) + min free(32)
  br i1 %psmall, label %failfree, label %build, !prof !0

build:
  %ownsp = getelementptr inbounds nuw i8, ptr %reg, i64 24
  store i64 %owns, ptr %ownsp, align 8
  %pstartp = getelementptr inbounds nuw i8, ptr %reg, i64 32
  store ptr %pstart, ptr %pstartp, align 8
  %basep = getelementptr inbounds nuw i8, ptr %reg, i64 40
  store ptr %reg, ptr %basep, align 8
  %totp = getelementptr inbounds nuw i8, ptr %reg, i64 56
  store i64 %effsize, ptr %totp, align 8           ; region_head @+48 stays null (memset)
  call void @seed_pool(ptr %reg, ptr %pstart, i64 %psize)
  ret ptr %reg

failfree:
  %ownsf = icmp eq i64 %owns, 1
  br i1 %ownsf, label %dofree, label %fail

dofree:
  call void @free(ptr %reg)
  br label %fail

fail:
  ret ptr null
}

define ptr @universe_alloc_tlsf_alloc(ptr %h, i64 %size) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail, label %need, !prof !0

need:
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 16)
  %av = extractvalue { i64, i1 } %a, 0
  %ao = extractvalue { i64, i1 } %a, 1
  %c = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %av, i64 15)
  %cv = extractvalue { i64, i1 } %c, 0
  %co = extractvalue { i64, i1 } %c, 1
  %need1 = and i64 %cv, -16
  %needf = call i64 @llvm.umax.i64(i64 %need1, i64 32)
  %ovf = or i1 %ao, %co
  br i1 %ovf, label %fail, label %search, !prof !0

search:
  %blk = call ptr @find_or_grow(ptr %h, i64 %needf)
  %bnull = icmp eq ptr %blk, null
  br i1 %bnull, label %fail, label %take, !prof !0

take:
  call void @rem(ptr %h, ptr %blk)
  %sfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  %sf = load i64, ptr %sfp, align 8
  %bsize = and i64 %sf, -16
  %prevfree = and i64 %sf, 2
  %ret = call ptr @place(ptr %h, ptr %blk, i64 %bsize, i64 %needf, i64 %prevfree)
  ret ptr %ret

fail:
  ret ptr null
}

define ptr @universe_alloc_tlsf_alloc_aligned(ptr %h, i64 %size, i64 %align) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail, label %chk, !prof !0

chk:
  %small = icmp ule i64 %align, 16
  br i1 %small, label %normal, label %big

normal:
  %rn = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 %size)
  ret ptr %rn

big:
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 16)
  %av = extractvalue { i64, i1 } %a, 0
  %ao = extractvalue { i64, i1 } %a, 1
  %c = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %av, i64 15)
  %cv = extractvalue { i64, i1 } %c, 0
  %co = extractvalue { i64, i1 } %c, 1
  %need1 = and i64 %cv, -16
  %needf = call i64 @llvm.umax.i64(i64 %need1, i64 32)
  %two = shl i64 %align, 1                          ; 2*align slack
  %s = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %needf, i64 %two)
  %srch = extractvalue { i64, i1 } %s, 0
  %so = extractvalue { i64, i1 } %s, 1
  %ovf0 = or i1 %ao, %co
  %ovf = or i1 %ovf0, %so
  br i1 %ovf, label %fail, label %search, !prof !0

search:
  %blk = call ptr @find_or_grow(ptr %h, i64 %srch)
  %bnull = icmp eq ptr %blk, null
  br i1 %bnull, label %fail, label %take, !prof !0

take:
  call void @rem(ptr %h, ptr %blk)
  %sfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  %sf = load i64, ptr %sfp, align 8
  %bsize = and i64 %sf, -16
  %prevfree = and i64 %sf, 2
  %p0 = getelementptr inbounds nuw i8, ptr %blk, i64 16
  %p0i = ptrtoint ptr %p0 to i64
  %am1 = add i64 %align, -1
  %sum = add i64 %p0i, %am1
  %negA = sub i64 0, %align
  %aligned = and i64 %sum, %negA
  %blki = ptrtoint ptr %blk to i64
  %hdri = sub i64 %aligned, 16
  %gap0 = sub i64 %hdri, %blki
  %gapsmall = icmp ult i64 %gap0, 32
  %gapnz = icmp ne i64 %gap0, 0
  %bump = and i1 %gapsmall, %gapnz
  %alignedB = add i64 %aligned, %align
  %alignedF = select i1 %bump, i64 %alignedB, i64 %aligned
  %hdriF = sub i64 %alignedF, 16
  %gap = sub i64 %hdriF, %blki
  %haveGap = icmp ne i64 %gap, 0
  br i1 %haveGap, label %withgap, label %nogap

withgap:
  %lsf0 = or i64 %gap, 1                            ; leading block free
  %lsf = or i64 %lsf0, %prevfree
  %lsfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  store i64 %lsf, ptr %lsfp, align 8
  call void @ins(ptr %h, ptr %blk, i64 %gap)
  %hdr = getelementptr inbounds nuw i8, ptr %blk, i64 %gap
  store ptr %blk, ptr %hdr, align 8                ; hdr.prev_phys = blk
  %newbsize = sub i64 %bsize, %gap
  %rg = call ptr @place(ptr %h, ptr %hdr, i64 %newbsize, i64 %needf, i64 2)
  ret ptr %rg

nogap:
  %rn2 = call ptr @place(ptr %h, ptr %blk, i64 %bsize, i64 %needf, i64 %prevfree)
  ret ptr %rn2

fail:
  ret ptr null
}

define void @universe_alloc_tlsf_free(ptr %h, ptr %p) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  %pnull = icmp eq ptr %p, null
  %any = or i1 %hnull, %pnull
  br i1 %any, label %done, label %work, !prof !0

work:
  %blk = getelementptr inbounds nuw i8, ptr %p, i64 -16
  %sfp = getelementptr inbounds nuw i8, ptr %blk, i64 8
  %sf = load i64, ptr %sfp, align 8
  %bsize0 = and i64 %sf, -16
  %prevfree = and i64 %sf, 2
  %livep = getelementptr inbounds nuw i8, ptr %h, i64 8
  %live = load i64, ptr %livep, align 8
  %pay = sub i64 %bsize0, 16
  %liven = sub i64 %live, %pay
  store i64 %liven, ptr %livep, align 8
  ; next physical block always exists (real block or region sentinel)
  %nb = getelementptr inbounds nuw i8, ptr %blk, i64 %bsize0
  %nbsfp = getelementptr inbounds nuw i8, ptr %nb, i64 8
  %nbsf = load i64, ptr %nbsfp, align 8
  %nbfree = and i64 %nbsf, 1
  %isfree = icmp ne i64 %nbfree, 0                  ; sentinel is used -> no cross-region merge
  br i1 %isfree, label %mergenext, label %afternext

mergenext:
  call void @rem(ptr %h, ptr %nb)
  %nbsize = and i64 %nbsf, -16
  %bsizeM = add i64 %bsize0, %nbsize
  br label %afternext

afternext:
  %bsize1 = phi i64 [ %bsize0, %work ], [ %bsizeM, %mergenext ]
  %prevfreeb = icmp ne i64 %prevfree, 0
  br i1 %prevfreeb, label %mergeprev, label %afterprev

mergeprev:
  %pb = load ptr, ptr %blk, align 8               ; prev_phys (null for region's first block)
  call void @rem(ptr %h, ptr %pb)
  %pbsfp = getelementptr inbounds nuw i8, ptr %pb, i64 8
  %pbsf = load i64, ptr %pbsfp, align 8
  %pbsize = and i64 %pbsf, -16
  %bsize2 = add i64 %bsize1, %pbsize
  br label %afterprev

afterprev:
  %blkF = phi ptr [ %blk, %afternext ], [ %pb, %mergeprev ]
  %bsizeF = phi i64 [ %bsize1, %afternext ], [ %bsize2, %mergeprev ]
  %sfpF = getelementptr inbounds nuw i8, ptr %blkF, i64 8
  %sfF = load i64, ptr %sfpF, align 8
  %pf = and i64 %sfF, 2                             ; preserve merged PREV_FREE
  %fin0 = or i64 %bsizeF, 1
  %fin = or i64 %fin0, %pf
  store i64 %fin, ptr %sfpF, align 8
  call void @ins(ptr %h, ptr %blkF, i64 %bsizeF)
  ; next physical block always exists (real block or region sentinel)
  %nb2 = getelementptr inbounds nuw i8, ptr %blkF, i64 %bsizeF
  store ptr %blkF, ptr %nb2, align 8              ; nb2.prev_phys = blkF
  %nb2sfp = getelementptr inbounds nuw i8, ptr %nb2, i64 8
  %nb2sf = load i64, ptr %nb2sfp, align 8
  %nb2sf2 = or i64 %nb2sf, 2                        ; blkF free -> PREV_FREE
  store i64 %nb2sf2, ptr %nb2sfp, align 8
  br label %done

done:
  ret void
}

define i64 @universe_alloc_tlsf_live(ptr %h) local_unnamed_addr #2 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %retz, label %work

work:
  %lp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %l = load i64, ptr %lp, align 8
  ret i64 %l

retz:
  ret i64 0
}

define void @universe_alloc_tlsf_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %done, label %freelist, !prof !0

freelist:
  ; walk grown regions, freeing each (RegionHeader is at the region front == base)
  %rhp = getelementptr inbounds nuw i8, ptr %h, i64 48
  %head0 = load ptr, ptr %rhp, align 8
  br label %loop

loop:
  %rh = phi ptr [ %head0, %freelist ], [ %next, %freenode ]
  %rhnull = icmp eq ptr %rh, null
  br i1 %rhnull, label %region0, label %freenode

freenode:
  %next = load ptr, ptr %rh, align 8               ; next BEFORE freeing (base == rh)
  %basep = getelementptr inbounds nuw i8, ptr %rh, i64 8
  %rbase = load ptr, ptr %basep, align 8
  call void @free(ptr %rbase)
  br label %loop

region0:
  %ownsp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %owns = load i64, ptr %ownsp, align 8
  %isowns = icmp eq i64 %owns, 1
  br i1 %isowns, label %dofree, label %done

dofree:
  %r0p = getelementptr inbounds nuw i8, ptr %h, i64 40
  %r0 = load ptr, ptr %r0p, align 8
  call void @free(ptr %r0)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
