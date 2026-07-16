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

; Compact integer sequences: Elias-Fano over a monotone u64 sequence, plus
; delta+varint postings and a SIMD-friendly byte-packed postings codec.
;
; DESIGN (functional spec: near-optimal-space storage of a sorted integer
;         sequence with O(1) random access and successor search; and a
;         compressed, SIMD-decodable posting list of sorted document ids):
;
;   ELIAS-FANO (universe_ds_ef_*)
;   * A monotone non-decreasing u64 sequence of N values with maximum U is
;     split per value into l = floor(log2(U/N)) LOW bits and the remaining
;     HIGH bits. Low bits are bit-packed contiguously (N*l bits). High bits are
;     stored in a UNARY-GAP bitvector: for value i we set bit (v_i>>l)+i. The
;     positions strictly increase, so the sequence uses exactly N set bits over
;     ~N + U/2^l total bits — the Elias-Fano bound of N*(2 + ceil(log2(U/N)))
;     bits, near the information-theoretic minimum.
;   * access(i): the i-th value's high part = select1(i) - i (the i-th set bit's
;     position minus i, because value i owns set-bit-ordinal i); OR the low bits
;     back in. select1 is popcount-based with a sampled index (one bit position
;     stored every 64 set bits), so a select scans O(1) words then finds the
;     target bit within one word by clearing the low set bits. No allocation,
;     no external call on the access hot path.
;   * next_geq(x): binary search using access over the monotone sequence,
;     O(log N). Returns the first value >= x, or -1 if none exists.
;   * ONE allocation, house header+payload layout. Header (64 B):
;       +0  i64 n           number of values
;       +8  i64 u           maximum value (vals[n-1], 0 if empty)
;       +16 i64 l           low bit width in [0,63]
;       +24 i64 low_words   ceil(n*l/64) logical low words
;       +32 i64 hi_words    words in the high bitvector
;       +40 i64 nsamp       number of select samples
;     Payload, in order: low bits [(low_words+1)*8 B, one pad word so a
;     boundary-spanning low read can load word+1 unconditionally], high
;     bitvector [hi_words*8], select samples [nsamp*8]. Offsets are derived
;     from the counts, not stored.
;   * Every bit-pack shift is guarded < 64 (l<=63 by construction; the spill
;     shift 64-off is only taken when off>=1, and masked-select otherwise), so
;     no shl/lshr can ever be poison (hazard: shift >= width).
;
;   POSTINGS — delta + varint (universe_ds_postings_encode/decode)
;   * Consecutive-id deltas (d0 = ids[0], d_i = ids[i]-ids[i-1]) ULEB128-coded.
;     Small gaps become 1 byte. Decode prefix-sums the deltas back to absolute
;     ids. Reuses the encoding/varint LEB128 leaves (composition, not a copy).
;
;   POSTINGS — SIMD byte-packed (universe_ds_postings_*_packed)
;   * ULEB128 is byte-serial (each byte's continuation bit gates the next), so
;     it does not vectorize. For the SIMD path we use a FRAME-OF-REFERENCE
;     byte-width block layout: deltas are grouped in blocks of 128; each block
;     picks the smallest of {1,2,4,8} bytes that holds its max delta and stores
;     all deltas at that fixed width, little-endian. Decode is then a pure
;     WIDENING LOAD (zext of a constant-width element) per block — unit stride,
;     no data-dependent control flow, no early exit — which the LLVM vectorizer
;     lowers to vpmovzx* (AVX) / ushll (NEON). Prefix-sum is a separate scalar
;     pass (an inherently sequential dependency; kept out of the vector body per
;     the compute/memory phase-split doctrine). A scalar twin decoder is the
;     cross-check oracle and portable fallback.
;   * Layout: [i64 n][ per block: i8 code (0..3 => 1,2,4,8 bytes) | count*B data ].
;
;   RLE (universe_ds_rle_*) — byte-value run-length codec, a small helper for
;   run-heavy streams: (ULEB128 run length, i8 value) pairs.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW, 6 FULL, 8 INVALID_ARG;
;      i64 entries returning a byte/element count use a NEGATIVE errno on error):
;   ptr universe_ds_ef_build(ptr vals, i64 n)      ; vals monotone non-decreasing
;   void universe_ds_ef_destroy(ptr ef)
;   i64 universe_ds_ef_access(ptr ef, i64 i)       ; vals[i], -1 if bad
;   i64 universe_ds_ef_next_geq(ptr ef, i64 x)     ; first value >= x, -1 none
;   i64 universe_ds_ef_size(ptr ef)                ; n, 0 if null
;   i64 universe_ds_ef_footprint(ptr ef)           ; bytes allocated, 0 if null
;   i64 universe_ds_postings_bound(i64 n)          ; max encode bytes, <0 overflow
;   i64 universe_ds_postings_encode(ptr ids, i64 n, ptr dst)          ; bytes, <0 err
;   i64 universe_ds_postings_decode(ptr src, i64 bytes, ptr out, i64 cap) ; n, <0 err
;   i64 universe_ds_postings_bound_packed(i64 n)   ; max packed bytes, <0 overflow
;   i64 universe_ds_postings_encode_packed(ptr ids, i64 n, ptr dst)   ; bytes, <0 err
;   i64 universe_ds_postings_decode_packed(ptr src, i64 bytes, ptr out, i64 cap)
;   i64 universe_ds_postings_decode_packed_scalar(ptr src, i64 bytes, ptr out, i64 cap)
;   i64 universe_ds_rle_encode(ptr src, i64 n, ptr dst)              ; bytes, <0 err
;   i64 universe_ds_rle_decode(ptr src, i64 bytes, ptr out, i64 cap) ; n, <0 err

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.cttz.i64(i64, i1 immarg)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

; external varint leaves (encoding domain) — composition, not duplication.
declare i64 @universe_varint_uleb_encode(ptr, i64)
declare i64 @universe_varint_uleb_decode(ptr, i64, ptr)

; ======================================================================
; Elias-Fano
; ======================================================================

; select1: position of the r-th set bit (0-indexed) in the high bitvector.
; Sampled index: samp[j] holds the bit position of the (j*64)-th set bit.
define internal i64 @ef_select1(ptr %hi, ptr %samp, i64 %r) #4 alwaysinline {
entry:
  %j = lshr i64 %r, 6
  %sp = getelementptr inbounds nuw i64, ptr %samp, i64 %j
  %sbit = load i64, ptr %sp, align 8
  %sword = lshr i64 %sbit, 6
  %soff = and i64 %sbit, 63
  ; count set bits before word %sword: (j*64) minus the set bits of word
  ; %sword lying below %soff.
  %m = shl nuw i64 1, %soff
  %m1 = add i64 %m, -1
  %w0p = getelementptr inbounds nuw i64, ptr %hi, i64 %sword
  %w0 = load i64, ptr %w0p, align 8
  %below = and i64 %w0, %m1
  %partial = call i64 @llvm.ctpop.i64(i64 %below)
  %base = shl nuw i64 %j, 6
  %cnt0 = sub i64 %base, %partial
  br label %loop

loop:
  %w = phi i64 [ %sword, %entry ], [ %w.n, %next ]
  %cnt = phi i64 [ %cnt0, %entry ], [ %cntpc, %next ]
  %wp = getelementptr inbounds nuw i64, ptr %hi, i64 %w
  %word = load i64, ptr %wp, align 8
  %pc = call i64 @llvm.ctpop.i64(i64 %word)
  %cntpc = add i64 %cnt, %pc
  %found = icmp ugt i64 %cntpc, %r
  br i1 %found, label %within, label %next

next:
  %w.n = add nuw i64 %w, 1
  br label %loop

within:
  %t = sub i64 %r, %cnt
  br label %sel.head

sel.head:
  %cur = phi i64 [ %word, %within ], [ %cur.n, %sel.body ]
  %k = phi i64 [ 0, %within ], [ %k.n, %sel.body ]
  %kdone = icmp uge i64 %k, %t
  br i1 %kdone, label %sel.done, label %sel.body

sel.body:
  %curm1 = sub i64 %cur, 1
  %cur.n = and i64 %cur, %curm1
  %k.n = add nuw i64 %k, 1
  br label %sel.head

sel.done:
  %pos = call i64 @llvm.cttz.i64(i64 %cur, i1 true)
  %wbase = shl nuw i64 %w, 6
  %res = add nuw i64 %wbase, %pos
  ret i64 %res
}

define ptr @universe_ds_ef_build(ptr %vals, i64 %n) local_unnamed_addr #1 {
entry:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %compute, label %need.vals

need.vals:
  %v.null = icmp eq ptr %vals, null
  br i1 %v.null, label %fail, label %compute, !prof !0

compute:
  ; U = (n==0) ? 0 : vals[n-1]
  br i1 %n0, label %have.u, label %load.u

load.u:
  %nm1 = sub i64 %n, 1
  %up = getelementptr inbounds nuw i64, ptr %vals, i64 %nm1
  %uval = load i64, ptr %up, align 8
  br label %have.u

have.u:
  %u = phi i64 [ 0, %compute ], [ %uval, %load.u ]
  ; l = floor(log2(U/N)), clamped to 0. (u==0 or n==0 -> l=0)
  %uz = icmp eq i64 %u, 0
  %nz2 = or i1 %n0, %uz
  br i1 %nz2, label %have.l, label %calc.l

calc.l:
  %q = udiv i64 %u, %n
  %qz = icmp eq i64 %q, 0
  %clz = call i64 @llvm.ctlz.i64(i64 %q, i1 true)
  %lg = sub i64 63, %clz
  %lsel = select i1 %qz, i64 0, i64 %lg
  br label %have.l

have.l:
  %l = phi i64 [ 0, %have.u ], [ %lsel, %calc.l ]
  ; low_words = ceil(n*l/64)
  %bits.m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 %l)
  %bits = extractvalue { i64, i1 } %bits.m, 0
  %bits.o = extractvalue { i64, i1 } %bits.m, 1
  br i1 %bits.o, label %fail, label %lw, !prof !0

lw:
  %bits63.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %bits, i64 63)
  %bits63 = extractvalue { i64, i1 } %bits63.m, 0
  %bits63.o = extractvalue { i64, i1 } %bits63.m, 1
  br i1 %bits63.o, label %fail, label %hw, !prof !0

hw:
  %low_words = lshr i64 %bits63, 6
  ; hi_bits = (U>>l) + n ; hi_words = ceil(hi_bits/64)
  %hh = lshr i64 %u, %l
  %hb.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %hh, i64 %n)
  %hi_bits = extractvalue { i64, i1 } %hb.m, 0
  %hb.o = extractvalue { i64, i1 } %hb.m, 1
  br i1 %hb.o, label %fail, label %hw2, !prof !0

hw2:
  %hb63.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %hi_bits, i64 63)
  %hb63 = extractvalue { i64, i1 } %hb63.m, 0
  %hb63.o = extractvalue { i64, i1 } %hb63.m, 1
  br i1 %hb63.o, label %fail, label %samp, !prof !0

samp:
  %hi_words = lshr i64 %hb63, 6
  ; nsamp = (n==0) ? 0 : ((n-1)>>6)+1
  %nm1s = sub i64 %n, 1
  %nsj = lshr i64 %nm1s, 6
  %nsp1 = add i64 %nsj, 1
  %nsamp = select i1 %n0, i64 0, i64 %nsp1
  ; sizes: low region = (low_words+1) words
  %low_alloc = add i64 %low_words, 1
  %low_bytes = shl i64 %low_alloc, 3
  %hi_bytes = shl i64 %hi_words, 3
  %samp_bytes = shl i64 %nsamp, 3
  ; total = 64 + low_bytes + hi_bytes + samp_bytes  (guard the running sum)
  %t1.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %low_bytes, i64 %hi_bytes)
  %t1 = extractvalue { i64, i1 } %t1.m, 0
  %t1.o = extractvalue { i64, i1 } %t1.m, 1
  br i1 %t1.o, label %fail, label %sz2, !prof !0

sz2:
  %t2.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %t1, i64 %samp_bytes)
  %t2 = extractvalue { i64, i1 } %t2.m, 0
  %t2.o = extractvalue { i64, i1 } %t2.m, 1
  br i1 %t2.o, label %fail, label %sz3, !prof !0

sz3:
  %tot.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %t2, i64 64)
  %tot = extractvalue { i64, i1 } %tot.m, 0
  %tot.o = extractvalue { i64, i1 } %tot.m, 1
  br i1 %tot.o, label %fail, label %alloc, !prof !0

alloc:
  %ef = call ptr @malloc(i64 %tot)
  %ef.null = icmp eq ptr %ef, null
  br i1 %ef.null, label %fail, label %hdr, !prof !0

hdr:
  store i64 %n, ptr %ef, align 8
  %up8 = getelementptr inbounds nuw i8, ptr %ef, i64 8
  store i64 %u, ptr %up8, align 8
  %lp = getelementptr inbounds nuw i8, ptr %ef, i64 16
  store i64 %l, ptr %lp, align 8
  %lwp = getelementptr inbounds nuw i8, ptr %ef, i64 24
  store i64 %low_words, ptr %lwp, align 8
  %hwp = getelementptr inbounds nuw i8, ptr %ef, i64 32
  store i64 %hi_words, ptr %hwp, align 8
  %nsp = getelementptr inbounds nuw i8, ptr %ef, i64 40
  store i64 %nsamp, ptr %nsp, align 8
  ; zero the whole payload (low pad + high bitvector + samples).
  %pay = getelementptr inbounds nuw i8, ptr %ef, i64 64
  %paybytes = sub i64 %tot, 64
  call void @llvm.memset.p0.i64(ptr %pay, i8 0, i64 %paybytes, i1 false)
  ; region pointers
  %low_ptr = getelementptr inbounds nuw i8, ptr %ef, i64 64
  %hi_ptr = getelementptr inbounds nuw i8, ptr %low_ptr, i64 %low_bytes
  %samp_ptr = getelementptr inbounds nuw i8, ptr %hi_ptr, i64 %hi_bytes
  br i1 %n0, label %ret.ef, label %fill.pre

fill.pre:
  ; lowmask = (l==0) ? 0 : (1<<l)-1
  %lz3 = icmp eq i64 %l, 0
  %onel = shl nuw i64 1, %l
  %maskm1 = add i64 %onel, -1
  %lowmask = select i1 %lz3, i64 0, i64 %maskm1
  br label %fill

fill:
  %i = phi i64 [ 0, %fill.pre ], [ %i.n, %fill.cont ]
  %vp = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %v = load i64, ptr %vp, align 8
  ; --- high bit: pos = (v>>l)+i
  %hpart = lshr i64 %v, %l
  %pos = add i64 %hpart, %i
  %hword = lshr i64 %pos, 6
  %hoff = and i64 %pos, 63
  %hmask = shl nuw i64 1, %hoff
  %hwp2 = getelementptr inbounds nuw i64, ptr %hi_ptr, i64 %hword
  %hcur = load i64, ptr %hwp2, align 8
  %hnew = or i64 %hcur, %hmask
  store i64 %hnew, ptr %hwp2, align 8
  ; --- sample every 64th value
  %ismod = and i64 %i, 63
  %issamp = icmp eq i64 %ismod, 0
  br i1 %issamp, label %do.samp, label %after.samp

do.samp:
  %sj = lshr i64 %i, 6
  %sptr = getelementptr inbounds nuw i64, ptr %samp_ptr, i64 %sj
  store i64 %pos, ptr %sptr, align 8
  br label %after.samp

after.samp:
  br i1 %lz3, label %fill.cont, label %do.low

do.low:
  %low = and i64 %v, %lowmask
  %bitpos = mul i64 %i, %l
  %lw2 = lshr i64 %bitpos, 6
  %lo2 = and i64 %bitpos, 63
  %lwp2 = getelementptr inbounds nuw i64, ptr %low_ptr, i64 %lw2
  %lcur = load i64, ptr %lwp2, align 8
  %lshift = shl i64 %low, %lo2
  %lnew = or i64 %lcur, %lshift
  store i64 %lnew, ptr %lwp2, align 8
  ; spill into next word if lo+l > 64
  %lol = add i64 %lo2, %l
  %spill = icmp ugt i64 %lol, 64
  br i1 %spill, label %do.spill, label %fill.cont

do.spill:
  %rsh = sub i64 64, %lo2
  %spillbits = lshr i64 %low, %rsh
  %lw3 = add nuw i64 %lw2, 1
  %lwp3 = getelementptr inbounds nuw i64, ptr %low_ptr, i64 %lw3
  %lcur2 = load i64, ptr %lwp3, align 8
  %lnew2 = or i64 %lcur2, %spillbits
  store i64 %lnew2, ptr %lwp3, align 8
  br label %fill.cont

fill.cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %ret.ef

ret.ef:
  ret ptr %ef

fail:
  ret ptr null
}

define void @universe_ds_ef_destroy(ptr %ef) local_unnamed_addr #1 {
entry:
  %isn = icmp eq ptr %ef, null
  br i1 %isn, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %ef)
  br label %done

done:
  ret void
}

define i64 @universe_ds_ef_access(ptr %ef, i64 %i) local_unnamed_addr #3 {
entry:
  %isn = icmp eq ptr %ef, null
  br i1 %isn, label %bad, label %chk, !prof !0

bad:
  ret i64 -1

chk:
  %n = load i64, ptr %ef, align 8
  %oob = icmp uge i64 %i, %n
  br i1 %oob, label %bad, label %do, !prof !0

do:
  %lp = getelementptr inbounds nuw i8, ptr %ef, i64 16
  %l = load i64, ptr %lp, align 8
  %lwp = getelementptr inbounds nuw i8, ptr %ef, i64 24
  %low_words = load i64, ptr %lwp, align 8
  %hwp = getelementptr inbounds nuw i8, ptr %ef, i64 32
  %hi_words = load i64, ptr %hwp, align 8
  %low_alloc = add i64 %low_words, 1
  %low_bytes = shl i64 %low_alloc, 3
  %hi_bytes = shl i64 %hi_words, 3
  %low_ptr = getelementptr inbounds nuw i8, ptr %ef, i64 64
  %hi_ptr = getelementptr inbounds nuw i8, ptr %low_ptr, i64 %low_bytes
  %samp_ptr = getelementptr inbounds nuw i8, ptr %hi_ptr, i64 %hi_bytes
  %r = call i64 @ef_select1(ptr %hi_ptr, ptr %samp_ptr, i64 %i)
  %h = sub i64 %r, %i
  %lz = icmp eq i64 %l, 0
  br i1 %lz, label %nolow, label %readlow

nolow:
  ; value = h (l==0 => no low bits)
  ret i64 %h

readlow:
  %onel = shl nuw i64 1, %l
  %lowmask = add i64 %onel, -1
  %bitpos = mul i64 %i, %l
  %lw = lshr i64 %bitpos, 6
  %lo = and i64 %bitpos, 63
  %w0p = getelementptr inbounds nuw i64, ptr %low_ptr, i64 %lw
  %w0 = load i64, ptr %w0p, align 8
  %lw1 = add nuw i64 %lw, 1
  %w1p = getelementptr inbounds nuw i64, ptr %low_ptr, i64 %lw1
  %w1 = load i64, ptr %w1p, align 8
  %part0 = lshr i64 %w0, %lo
  %lonz = icmp ne i64 %lo, 0
  %rsh = sub i64 64, %lo
  %rsh.safe = select i1 %lonz, i64 %rsh, i64 0
  %hi.raw = shl i64 %w1, %rsh.safe
  %lol = add i64 %lo, %l
  %spill = icmp ugt i64 %lol, 64
  %hipart = select i1 %spill, i64 %hi.raw, i64 0
  %raw = or i64 %part0, %hipart
  %low = and i64 %raw, %lowmask
  %hshift = shl i64 %h, %l
  %val = or i64 %hshift, %low
  ret i64 %val
}

define i64 @universe_ds_ef_next_geq(ptr %ef, i64 %x) local_unnamed_addr #3 {
entry:
  %isn = icmp eq ptr %ef, null
  br i1 %isn, label %none, label %chk, !prof !0

none:
  ret i64 -1

chk:
  %n = load i64, ptr %ef, align 8
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %none, label %bs

bs:
  br label %loop

loop:
  %lo = phi i64 [ 0, %bs ], [ %lo.n, %step ]
  %hi = phi i64 [ %n, %bs ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %fin

step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %vmid = call i64 @universe_ds_ef_access(ptr %ef, i64 %mid)
  %less = icmp ult i64 %vmid, %x
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %less, i64 %mid1, i64 %lo
  %hi.n = select i1 %less, i64 %hi, i64 %mid
  br label %loop

fin:
  %atend = icmp eq i64 %lo, %n
  br i1 %atend, label %none, label %hit

hit:
  %res = call i64 @universe_ds_ef_access(ptr %ef, i64 %lo)
  ret i64 %res
}

define i64 @universe_ds_ef_size(ptr %ef) local_unnamed_addr #3 {
entry:
  %isn = icmp eq ptr %ef, null
  br i1 %isn, label %z, label %do, !prof !0

z:
  ret i64 0

do:
  %n = load i64, ptr %ef, align 8
  ret i64 %n
}

define i64 @universe_ds_ef_footprint(ptr %ef) local_unnamed_addr #3 {
entry:
  %isn = icmp eq ptr %ef, null
  br i1 %isn, label %z, label %do, !prof !0

z:
  ret i64 0

do:
  %lwp = getelementptr inbounds nuw i8, ptr %ef, i64 24
  %low_words = load i64, ptr %lwp, align 8
  %hwp = getelementptr inbounds nuw i8, ptr %ef, i64 32
  %hi_words = load i64, ptr %hwp, align 8
  %nsp = getelementptr inbounds nuw i8, ptr %ef, i64 40
  %nsamp = load i64, ptr %nsp, align 8
  %low_alloc = add i64 %low_words, 1
  %sum = add i64 %low_alloc, %hi_words
  %sum2 = add i64 %sum, %nsamp
  %words_bytes = shl i64 %sum2, 3
  %tot = add i64 %words_bytes, 64
  ret i64 %tot
}

; ======================================================================
; Postings: delta + ULEB128 varint
; ======================================================================

define i64 @universe_ds_postings_bound(i64 %n) local_unnamed_addr #4 {
entry:
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 10)
  %v = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %r = select i1 %o, i64 -3, i64 %v
  ret i64 %r
}

define i64 @universe_ds_postings_encode(ptr %ids, i64 %n, ptr %dst) local_unnamed_addr #5 {
entry:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %zero, label %chk, !prof !0

zero:
  ret i64 0

chk:
  %idn = icmp eq ptr %ids, null
  %dstn = icmp eq ptr %dst, null
  %anyn = or i1 %idn, %dstn
  br i1 %anyn, label %err.null, label %loop, !prof !0

err.null:
  ret i64 -1

loop:
  %i = phi i64 [ 0, %chk ], [ %i.n, %loop ]
  %prev = phi i64 [ 0, %chk ], [ %v, %loop ]
  %off = phi i64 [ 0, %chk ], [ %off.n, %loop ]
  %vp = getelementptr inbounds nuw i64, ptr %ids, i64 %i
  %v = load i64, ptr %vp, align 8
  %d = sub i64 %v, %prev
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  %w = call i64 @universe_varint_uleb_encode(ptr %dp, i64 %d)
  %off.n = add i64 %off, %w
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret i64 %off.n
}

define i64 @universe_ds_postings_decode(ptr %src, i64 %bytes, ptr %out, i64 %cap) local_unnamed_addr #1 {
entry:
  %tmp = alloca i64, align 8
  %b0 = icmp eq i64 %bytes, 0
  br i1 %b0, label %zero, label %chk, !prof !0

zero:
  ret i64 0

chk:
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %out, null
  %anyn = or i1 %sn, %on
  br i1 %anyn, label %err.null, label %loop, !prof !0

err.null:
  ret i64 -1

loop:
  %off = phi i64 [ 0, %chk ], [ %off.n, %adv ]
  %count = phi i64 [ 0, %chk ], [ %count.n, %adv ]
  %base = phi i64 [ 0, %chk ], [ %base.n, %adv ]
  %full = icmp uge i64 %count, %cap
  br i1 %full, label %err.full, label %dec, !prof !0

err.full:
  ret i64 -6

dec:
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %off
  %rem = sub i64 %bytes, %off
  %r = call i64 @universe_varint_uleb_decode(ptr %sp, i64 %rem, ptr %tmp)
  %rneg = icmp slt i64 %r, 0
  br i1 %rneg, label %err.parse, label %adv, !prof !0

err.parse:
  ret i64 %r

adv:
  %dval = load i64, ptr %tmp, align 8
  %base.n = add i64 %base, %dval
  %outp = getelementptr inbounds nuw i64, ptr %out, i64 %count
  store i64 %base.n, ptr %outp, align 8
  %count.n = add nuw i64 %count, 1
  %off.n = add i64 %off, %r
  %done = icmp uge i64 %off.n, %bytes
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %count.n
}

; ======================================================================
; Postings: SIMD byte-packed (frame-of-reference, fixed width per block)
; ======================================================================

; code -> bytes:  0->1, 1->2, 2->4, 3->8   (B = 1<<code)
; category(maxd): smallest B in {1,2,4,8} that holds maxd (>=1 byte always).
define internal i64 @pk_code(i64 %maxd) #4 alwaysinline {
entry:
  %z = icmp eq i64 %maxd, 0
  %clz = call i64 @llvm.ctlz.i64(i64 %maxd, i1 true)
  %bits = sub i64 64, %clz
  ; nbytes = ceil(bits/8), min 1
  %b7 = add i64 %bits, 7
  %nb0 = lshr i64 %b7, 3
  %nb = select i1 %z, i64 1, i64 %nb0
  %le1 = icmp ule i64 %nb, 1
  %le2 = icmp ule i64 %nb, 2
  %le4 = icmp ule i64 %nb, 4
  ; code = le1?0 : le2?1 : le4?2 : 3
  %c0 = select i1 %le4, i64 2, i64 3
  %c1 = select i1 %le2, i64 1, i64 %c0
  %code = select i1 %le1, i64 0, i64 %c1
  ret i64 %code
}

define i64 @universe_ds_postings_bound_packed(i64 %n) local_unnamed_addr #4 {
entry:
  ; worst case: 8 bytes/value + one control byte per 128 block + 8-byte header.
  %m8 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 8)
  %v8 = extractvalue { i64, i1 } %m8, 0
  %o8 = extractvalue { i64, i1 } %m8, 1
  br i1 %o8, label %ovf, label %ctrl

ctrl:
  %nb = lshr i64 %n, 7
  %ctrlbytes = add i64 %nb, 1
  %a1.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %v8, i64 %ctrlbytes)
  %a1 = extractvalue { i64, i1 } %a1.m, 0
  %a1.o = extractvalue { i64, i1 } %a1.m, 1
  br i1 %a1.o, label %ovf, label %hdr

hdr:
  %a2.m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a1, i64 8)
  %a2 = extractvalue { i64, i1 } %a2.m, 0
  %a2.o = extractvalue { i64, i1 } %a2.m, 1
  br i1 %a2.o, label %ovf, label %done

done:
  ret i64 %a2

ovf:
  ret i64 -3
}

define i64 @universe_ds_postings_encode_packed(ptr %ids, i64 %n, ptr %dst) local_unnamed_addr #5 {
entry:
  %blk = alloca [128 x i64], align 16
  %dstn = icmp eq ptr %dst, null
  br i1 %dstn, label %err.null, label %hdr, !prof !0

err.null:
  ret i64 -1

hdr:
  ; header: n as 8-byte LE.
  store i64 %n, ptr %dst, align 1
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %ret8, label %chkids

ret8:
  ret i64 8

chkids:
  %idn = icmp eq ptr %ids, null
  br i1 %idn, label %err.null, label %blocks, !prof !0

blocks:
  %s = phi i64 [ 0, %chkids ], [ %s.n, %blk.end ]
  %off = phi i64 [ 8, %chkids ], [ %off.n, %blk.end ]
  %remain = sub i64 %n, %s
  %count = call i64 @llvm.umin.i64(i64 %remain, i64 128)
  ; prev = (s==0) ? 0 : ids[s-1]
  %s0 = icmp eq i64 %s, 0
  %sm1 = sub i64 %s, 1
  %pvp = getelementptr inbounds nuw i64, ptr %ids, i64 %sm1
  %pvload = load i64, ptr %pvp, align 8
  %prev0 = select i1 %s0, i64 0, i64 %pvload
  br label %dloop

dloop:
  %k = phi i64 [ 0, %blocks ], [ %k.n, %dloop ]
  %prev = phi i64 [ %prev0, %blocks ], [ %vv, %dloop ]
  %maxd = phi i64 [ 0, %blocks ], [ %maxd.n, %dloop ]
  %idx = add i64 %s, %k
  %vvp = getelementptr inbounds nuw i64, ptr %ids, i64 %idx
  %vv = load i64, ptr %vvp, align 8
  %dd = sub i64 %vv, %prev
  %bkp = getelementptr inbounds nuw [128 x i64], ptr %blk, i64 0, i64 %k
  store i64 %dd, ptr %bkp, align 8
  %maxd.n = call i64 @llvm.umax.i64(i64 %maxd, i64 %dd)
  %k.n = add nuw i64 %k, 1
  %kmore = icmp ult i64 %k.n, %count
  br i1 %kmore, label %dloop, label %emit

emit:
  %code = call i64 @pk_code(i64 %maxd.n)
  %codeb = trunc i64 %code to i8
  %cp = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  store i8 %codeb, ptr %cp, align 1
  %off1 = add i64 %off, 1
  %datap = getelementptr inbounds nuw i8, ptr %dst, i64 %off1
  %B = shl nuw i64 1, %code
  ; dispatch on code to a constant-width store loop.
  switch i64 %code, label %w8 [ i64 0, label %w1
                                i64 1, label %w2
                                i64 2, label %w4 ]

w1:
  br label %w1.loop
w1.loop:
  %k1 = phi i64 [ 0, %w1 ], [ %k1.n, %w1.loop ]
  %s1p = getelementptr inbounds nuw [128 x i64], ptr %blk, i64 0, i64 %k1
  %s1v = load i64, ptr %s1p, align 8
  %s1t = trunc i64 %s1v to i8
  %d1p = getelementptr inbounds nuw i8, ptr %datap, i64 %k1
  store i8 %s1t, ptr %d1p, align 1
  %k1.n = add nuw i64 %k1, 1
  %k1m = icmp ult i64 %k1.n, %count
  br i1 %k1m, label %w1.loop, label %blk.end

w2:
  br label %w2.loop
w2.loop:
  %k2 = phi i64 [ 0, %w2 ], [ %k2.n, %w2.loop ]
  %s2p = getelementptr inbounds nuw [128 x i64], ptr %blk, i64 0, i64 %k2
  %s2v = load i64, ptr %s2p, align 8
  %s2t = trunc i64 %s2v to i16
  %d2p = getelementptr inbounds nuw i16, ptr %datap, i64 %k2
  store i16 %s2t, ptr %d2p, align 1
  %k2.n = add nuw i64 %k2, 1
  %k2m = icmp ult i64 %k2.n, %count
  br i1 %k2m, label %w2.loop, label %blk.end

w4:
  br label %w4.loop
w4.loop:
  %k4 = phi i64 [ 0, %w4 ], [ %k4.n, %w4.loop ]
  %s4p = getelementptr inbounds nuw [128 x i64], ptr %blk, i64 0, i64 %k4
  %s4v = load i64, ptr %s4p, align 8
  %s4t = trunc i64 %s4v to i32
  %d4p = getelementptr inbounds nuw i32, ptr %datap, i64 %k4
  store i32 %s4t, ptr %d4p, align 1
  %k4.n = add nuw i64 %k4, 1
  %k4m = icmp ult i64 %k4.n, %count
  br i1 %k4m, label %w4.loop, label %blk.end

w8:
  br label %w8.loop
w8.loop:
  %k8 = phi i64 [ 0, %w8 ], [ %k8.n, %w8.loop ]
  %s8p = getelementptr inbounds nuw [128 x i64], ptr %blk, i64 0, i64 %k8
  %s8v = load i64, ptr %s8p, align 8
  %d8p = getelementptr inbounds nuw i64, ptr %datap, i64 %k8
  store i64 %s8v, ptr %d8p, align 1
  %k8.n = add nuw i64 %k8, 1
  %k8m = icmp ult i64 %k8.n, %count
  br i1 %k8m, label %w8.loop, label %blk.end

blk.end:
  %databytes = mul i64 %count, %B
  %off.n = add i64 %off1, %databytes
  %s.n = add i64 %s, %count
  %smore = icmp ult i64 %s.n, %n
  br i1 %smore, label %blocks, label %fin

fin:
  ret i64 %off.n
}

; SIMD decode: per-block widening (vectorizes), then a scalar prefix-sum pass.
; src/out are noalias (distinct compressed vs decoded buffers) so the widening
; loop carries no memory dependency and the vectorizer lowers it to vpmovzx*/ushll.
define i64 @universe_ds_postings_decode_packed(ptr noalias readonly %src, i64 %bytes, ptr noalias %out, i64 %cap) local_unnamed_addr #1 {
entry:
  %small = icmp ult i64 %bytes, 8
  br i1 %small, label %maybe.empty, label %chk, !prof !0

maybe.empty:
  %b0 = icmp eq i64 %bytes, 0
  br i1 %b0, label %zero, label %err.parse, !prof !0

zero:
  ret i64 0

err.parse:
  ret i64 -13

chk:
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %out, null
  %anyn = or i1 %sn, %on
  br i1 %anyn, label %err.null, label %rdhdr, !prof !0

err.null:
  ret i64 -1

rdhdr:
  %n = load i64, ptr %src, align 1
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %zero, label %capchk

capchk:
  %nofit = icmp ugt i64 %n, %cap
  br i1 %nofit, label %err.full, label %expand, !prof !0

err.full:
  ret i64 -6

expand:
  %s = phi i64 [ 0, %capchk ], [ %s.n, %blk.end ]
  %off = phi i64 [ 8, %capchk ], [ %off.n, %blk.end ]
  ; control byte bounds
  %needctrl = icmp uge i64 %off, %bytes
  br i1 %needctrl, label %err.parse, label %rdctrl, !prof !0

rdctrl:
  %cp = getelementptr inbounds nuw i8, ptr %src, i64 %off
  %code8 = load i8, ptr %cp, align 1
  %code = zext i8 %code8 to i64
  %off1 = add i64 %off, 1
  %remain = sub i64 %n, %s
  %count = call i64 @llvm.umin.i64(i64 %remain, i64 128)
  %B = shl nuw i64 1, %code
  %databytes = mul i64 %count, %B
  ; data bounds: off1 + databytes <= bytes
  %dataend = add i64 %off1, %databytes
  %overrun = icmp ugt i64 %dataend, %bytes
  br i1 %overrun, label %err.parse, label %datap.blk, !prof !0

datap.blk:
  %datap = getelementptr inbounds nuw i8, ptr %src, i64 %off1
  %outblk = getelementptr inbounds nuw i64, ptr %out, i64 %s
  ; nvec = count rounded down to a multiple of 4 (the explicit-vector span).
  %nvec = and i64 %count, -4
  %hasvec = icmp ne i64 %nvec, 0
  switch i64 %code, label %dw8 [ i64 0, label %dw1
                                 i64 1, label %dw2
                                 i64 2, label %dw4 ]

; --- B=1: explicit <4 x i8> -> <4 x i64> widen, scalar tail ---
dw1:
  br i1 %hasvec, label %dw1.vloop, label %dw1.tpre
dw1.vloop:
  %kv1 = phi i64 [ 0, %dw1 ], [ %kv1.n, %dw1.vloop ]
  %vp1 = getelementptr inbounds nuw i8, ptr %datap, i64 %kv1
  %vv1 = load <4 x i8>, ptr %vp1, align 1
  %vz1 = zext <4 x i8> %vv1 to <4 x i64>
  %vo1 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kv1
  store <4 x i64> %vz1, ptr %vo1, align 8
  %kv1.n = add nuw i64 %kv1, 4
  %vm1 = icmp ult i64 %kv1.n, %nvec
  br i1 %vm1, label %dw1.vloop, label %dw1.tpre
dw1.tpre:
  %ht1 = icmp ult i64 %nvec, %count
  br i1 %ht1, label %dw1.tloop, label %blk.end
dw1.tloop:
  %kt1 = phi i64 [ %nvec, %dw1.tpre ], [ %kt1.n, %dw1.tloop ]
  %tp1 = getelementptr inbounds nuw i8, ptr %datap, i64 %kt1
  %tv1 = load i8, ptr %tp1, align 1
  %tz1 = zext i8 %tv1 to i64
  %to1 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kt1
  store i64 %tz1, ptr %to1, align 8
  %kt1.n = add nuw i64 %kt1, 1
  %tm1 = icmp ult i64 %kt1.n, %count
  br i1 %tm1, label %dw1.tloop, label %blk.end

; --- B=2: explicit <4 x i16> -> <4 x i64> widen, scalar tail ---
dw2:
  br i1 %hasvec, label %dw2.vloop, label %dw2.tpre
dw2.vloop:
  %kv2 = phi i64 [ 0, %dw2 ], [ %kv2.n, %dw2.vloop ]
  %vp2 = getelementptr inbounds nuw i16, ptr %datap, i64 %kv2
  %vv2 = load <4 x i16>, ptr %vp2, align 1
  %vz2 = zext <4 x i16> %vv2 to <4 x i64>
  %vo2 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kv2
  store <4 x i64> %vz2, ptr %vo2, align 8
  %kv2.n = add nuw i64 %kv2, 4
  %vm2 = icmp ult i64 %kv2.n, %nvec
  br i1 %vm2, label %dw2.vloop, label %dw2.tpre
dw2.tpre:
  %ht2 = icmp ult i64 %nvec, %count
  br i1 %ht2, label %dw2.tloop, label %blk.end
dw2.tloop:
  %kt2 = phi i64 [ %nvec, %dw2.tpre ], [ %kt2.n, %dw2.tloop ]
  %tp2 = getelementptr inbounds nuw i16, ptr %datap, i64 %kt2
  %tv2 = load i16, ptr %tp2, align 1
  %tz2 = zext i16 %tv2 to i64
  %to2 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kt2
  store i64 %tz2, ptr %to2, align 8
  %kt2.n = add nuw i64 %kt2, 1
  %tm2 = icmp ult i64 %kt2.n, %count
  br i1 %tm2, label %dw2.tloop, label %blk.end

; --- B=4: explicit <4 x i32> -> <4 x i64> widen, scalar tail ---
dw4:
  br i1 %hasvec, label %dw4.vloop, label %dw4.tpre
dw4.vloop:
  %kv4 = phi i64 [ 0, %dw4 ], [ %kv4.n, %dw4.vloop ]
  %vp4 = getelementptr inbounds nuw i32, ptr %datap, i64 %kv4
  %vv4 = load <4 x i32>, ptr %vp4, align 1
  %vz4 = zext <4 x i32> %vv4 to <4 x i64>
  %vo4 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kv4
  store <4 x i64> %vz4, ptr %vo4, align 8
  %kv4.n = add nuw i64 %kv4, 4
  %vm4 = icmp ult i64 %kv4.n, %nvec
  br i1 %vm4, label %dw4.vloop, label %dw4.tpre
dw4.tpre:
  %ht4 = icmp ult i64 %nvec, %count
  br i1 %ht4, label %dw4.tloop, label %blk.end
dw4.tloop:
  %kt4 = phi i64 [ %nvec, %dw4.tpre ], [ %kt4.n, %dw4.tloop ]
  %tp4 = getelementptr inbounds nuw i32, ptr %datap, i64 %kt4
  %tv4 = load i32, ptr %tp4, align 1
  %tz4 = zext i32 %tv4 to i64
  %to4 = getelementptr inbounds nuw i64, ptr %outblk, i64 %kt4
  store i64 %tz4, ptr %to4, align 8
  %kt4.n = add nuw i64 %kt4, 1
  %tm4 = icmp ult i64 %kt4.n, %count
  br i1 %tm4, label %dw4.tloop, label %blk.end

; --- B=8: no widening, block memcpy (already optimal) ---
dw8:
  %db8 = shl nuw i64 %count, 3
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %outblk, ptr align 1 %datap, i64 %db8, i1 false)
  br label %blk.end

blk.end:
  %off.n = add i64 %off1, %databytes
  %s.n = add i64 %s, %count
  %smore = icmp ult i64 %s.n, %n
  br i1 %smore, label %expand, label %prefix

prefix:
  ; scalar prefix-sum: out[i] += out[i-1]  (out[0] already = ids[0]).
  %multi = icmp ugt i64 %n, 1
  br i1 %multi, label %ps.loop, label %fin

ps.loop:
  %pi = phi i64 [ 1, %prefix ], [ %pi.n, %ps.loop ]
  %pip = getelementptr inbounds nuw i64, ptr %out, i64 %pi
  %pd = load i64, ptr %pip, align 8
  %pi0 = sub i64 %pi, 1
  %pp0 = getelementptr inbounds nuw i64, ptr %out, i64 %pi0
  %pbaseval = load i64, ptr %pp0, align 8
  %psum = add i64 %pd, %pbaseval
  store i64 %psum, ptr %pip, align 8
  %pi.n = add nuw i64 %pi, 1
  %psmore = icmp ult i64 %pi.n, %n
  br i1 %psmore, label %ps.loop, label %fin

fin:
  ret i64 %n
}

; Scalar reference decoder for the same packed format (cross-check oracle).
define i64 @universe_ds_postings_decode_packed_scalar(ptr %src, i64 %bytes, ptr %out, i64 %cap) local_unnamed_addr #1 {
entry:
  %small = icmp ult i64 %bytes, 8
  br i1 %small, label %maybe.empty, label %chk, !prof !0

maybe.empty:
  %b0 = icmp eq i64 %bytes, 0
  br i1 %b0, label %zero, label %err.parse, !prof !0

zero:
  ret i64 0

err.parse:
  ret i64 -13

chk:
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %out, null
  %anyn = or i1 %sn, %on
  br i1 %anyn, label %err.null, label %rdhdr, !prof !0

err.null:
  ret i64 -1

rdhdr:
  %n = load i64, ptr %src, align 1
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %zero, label %capchk

capchk:
  %nofit = icmp ugt i64 %n, %cap
  br i1 %nofit, label %err.full, label %vloop, !prof !0

err.full:
  ret i64 -6

vloop:
  %s = phi i64 [ 0, %capchk ], [ %s.n, %blk.end ]
  %off = phi i64 [ 8, %capchk ], [ %off.n, %blk.end ]
  %base = phi i64 [ 0, %capchk ], [ %base.b, %blk.end ]
  %needctrl = icmp uge i64 %off, %bytes
  br i1 %needctrl, label %err.parse, label %rdctrl, !prof !0

rdctrl:
  %cp = getelementptr inbounds nuw i8, ptr %src, i64 %off
  %code8 = load i8, ptr %cp, align 1
  %code = zext i8 %code8 to i64
  %off1 = add i64 %off, 1
  %remain = sub i64 %n, %s
  %count = call i64 @llvm.umin.i64(i64 %remain, i64 128)
  %B = shl nuw i64 1, %code
  %databytes = mul i64 %count, %B
  %dataend = add i64 %off1, %databytes
  %overrun = icmp ugt i64 %dataend, %bytes
  br i1 %overrun, label %err.parse, label %vk, !prof !0

vk:
  %k = phi i64 [ 0, %rdctrl ], [ %k.n, %vk.tail ]
  %kbase = phi i64 [ %base, %rdctrl ], [ %absv, %vk.tail ]
  ; read B bytes little-endian at datap + k*B
  %koff = mul i64 %k, %B
  %vpos = add i64 %off1, %koff
  br label %byte.loop

byte.loop:
  %bi = phi i64 [ 0, %vk ], [ %bi.n, %byte.loop ]
  %acc = phi i64 [ 0, %vk ], [ %acc.n, %byte.loop ]
  %bpos = add i64 %vpos, %bi
  %bptr = getelementptr inbounds nuw i8, ptr %src, i64 %bpos
  %bv = load i8, ptr %bptr, align 1
  %bz = zext i8 %bv to i64
  %sh = shl i64 %bi, 3
  %bshift = shl i64 %bz, %sh
  %acc.n = or i64 %acc, %bshift
  %bi.n = add nuw i64 %bi, 1
  %bimore = icmp ult i64 %bi.n, %B
  br i1 %bimore, label %byte.loop, label %vstore

vstore:
  %absv = add i64 %kbase, %acc.n
  %idx = add i64 %s, %k
  %op = getelementptr inbounds nuw i64, ptr %out, i64 %idx
  store i64 %absv, ptr %op, align 8
  br label %vk.tail

vk.tail:
  %k.n = add nuw i64 %k, 1
  %kmore = icmp ult i64 %k.n, %count
  br i1 %kmore, label %vk, label %blk.end

blk.end:
  %base.b = phi i64 [ %absv, %vk.tail ]
  %off.n = add i64 %off1, %databytes
  %s.n = add i64 %s, %count
  %smore = icmp ult i64 %s.n, %n
  br i1 %smore, label %vloop, label %fin

fin:
  ret i64 %n
}

; ======================================================================
; RLE: byte-value run-length codec — (ULEB128 run length, i8 value) pairs
; ======================================================================

define i64 @universe_ds_rle_encode(ptr %src, i64 %n, ptr %dst) local_unnamed_addr #5 {
entry:
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %zero, label %chk, !prof !0

zero:
  ret i64 0

chk:
  %sn = icmp eq ptr %src, null
  %dn = icmp eq ptr %dst, null
  %anyn = or i1 %sn, %dn
  br i1 %anyn, label %err.null, label %run, !prof !0

err.null:
  ret i64 -1

run:
  %i = phi i64 [ 0, %chk ], [ %i.n2, %emit ]
  %off = phi i64 [ 0, %chk ], [ %off.n, %emit ]
  %vp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %v = load i8, ptr %vp, align 1
  %istart = add nuw i64 %i, 1
  br label %scan

scan:
  %j = phi i64 [ %istart, %run ], [ %j.n, %scan.cont ]
  %atend = icmp uge i64 %j, %n
  br i1 %atend, label %emit.pre, label %scan.chk

scan.chk:
  %jp = getelementptr inbounds nuw i8, ptr %src, i64 %j
  %jv = load i8, ptr %jp, align 1
  %same = icmp eq i8 %jv, %v
  br i1 %same, label %scan.cont, label %emit.pre

scan.cont:
  %j.n = add nuw i64 %j, 1
  br label %scan

emit.pre:
  %runlen = sub i64 %j, %i
  br label %emit

emit:
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  %w = call i64 @universe_varint_uleb_encode(ptr %dp, i64 %runlen)
  %offw = add i64 %off, %w
  %valp = getelementptr inbounds nuw i8, ptr %dst, i64 %offw
  store i8 %v, ptr %valp, align 1
  %off.n = add i64 %offw, 1
  %i.n2 = add i64 %i, %runlen
  %more = icmp ult i64 %i.n2, %n
  br i1 %more, label %run, label %done

done:
  ret i64 %off.n
}

define i64 @universe_ds_rle_decode(ptr %src, i64 %bytes, ptr %out, i64 %cap) local_unnamed_addr #1 {
entry:
  %tmp = alloca i64, align 8
  %b0 = icmp eq i64 %bytes, 0
  br i1 %b0, label %zero, label %chk, !prof !0

zero:
  ret i64 0

chk:
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %out, null
  %anyn = or i1 %sn, %on
  br i1 %anyn, label %err.null, label %loop, !prof !0

err.null:
  ret i64 -1

loop:
  %off = phi i64 [ 0, %chk ], [ %off.n, %fill.done ]
  %count = phi i64 [ 0, %chk ], [ %count.n, %fill.done ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %off
  %rem = sub i64 %bytes, %off
  %r = call i64 @universe_varint_uleb_decode(ptr %sp, i64 %rem, ptr %tmp)
  %rneg = icmp slt i64 %r, 0
  br i1 %rneg, label %err.parse, label %chkval, !prof !0

err.parse:
  ret i64 %r

chkval:
  %valoff = add i64 %off, %r
  %novalue = icmp uge i64 %valoff, %bytes
  br i1 %novalue, label %err.parse, label %readval, !prof !0

readval:
  %runlen = load i64, ptr %tmp, align 8
  %vptr = getelementptr inbounds nuw i8, ptr %src, i64 %valoff
  %v = load i8, ptr %vptr, align 1
  %newcount = add i64 %count, %runlen
  %overcap = icmp ugt i64 %newcount, %cap
  br i1 %overcap, label %err.full, label %fill, !prof !0

err.full:
  ret i64 -6

fill:
  %rl0 = icmp eq i64 %runlen, 0
  br i1 %rl0, label %fill.done, label %fill.loop

fill.loop:
  %fk = phi i64 [ 0, %fill ], [ %fk.n, %fill.loop ]
  %oidx = add i64 %count, %fk
  %op = getelementptr inbounds nuw i8, ptr %out, i64 %oidx
  store i8 %v, ptr %op, align 1
  %fk.n = add nuw i64 %fk, 1
  %fkmore = icmp ult i64 %fk.n, %runlen
  br i1 %fkmore, label %fill.loop, label %fill.done

fill.done:
  %count.n = add i64 %count, %runlen
  %off.n = add i64 %valoff, 1
  %done = icmp uge i64 %off.n, %bytes
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %count.n
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nofree memory(argmem: read) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #5 = { nounwind willreturn norecurse nofree }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
