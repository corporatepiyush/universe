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

; UTF-16 codec (LE/BE) and UTF-8 <-> UTF-16 transcoding. Pure compute over
; caller-supplied byte ranges: the caller pre-sizes `dst` (via the len_* helpers)
; and we never allocate, never do IO.
;
; DESIGN:
;   * A UTF-16 code unit is a 16-bit value; a "unit index" addresses units, byte
;     offset = idx*2. Endianness is a runtime i32 param (0 = little-endian,
;     nonzero = big-endian). All our targets are little-endian, so a native i16
;     load/store IS the LE form; the BE form is one `llvm.bswap.i16` selected
;     branchlessly. Loads/stores use align 1 (buffers are arbitrary byte ranges).
;   * Scalar validity is a single deep helper (svalid): reject the surrogate
;     block U+D800..U+DFFF and anything above U+10FFFF. Encoding a valid scalar
;     is either one BMP unit or a hi/lo surrogate pair:
;       s2 = s - 0x10000;  hi = 0xD800 | (s2>>10);  lo = 0xDC00 | (s2 & 0x3FF)
;     Decoding inverts it: BMP passes through; a high surrogate (D800..DBFF)
;     must be followed by a low surrogate (DC00..DFFF) or it is malformed; a
;     lone low surrogate, a reversed pair, and a truncated pair are all PARSE.
;   * Compute/memory separation: transcode is buffer-to-buffer. from_utf8 first
;     calls the sibling UTF-8 module's validate (single well-formedness pass,
;     reused not reimplemented), then a decode+emit pass that steps by
;     byte_len_of_codepoint (also reused) and trusts validity. to_utf8 decodes
;     each UTF-16 scalar with our own decoder and emits UTF-8 inline. Length
;     helpers pre-measure so the caller can size dst exactly, then the copy pass
;     only bounds-checks against dcap (cold FULL exit).
;   * The hot loops keep only the ASCII/BMP fall-through and cold error/FULL
;     branches (!prof weighted); scalar classification is branchless selects.
;   * SIMD-first transcode fast path (from_utf8 / to_utf8): a portable
;     <16 x i8>/<16 x i16> ASCII lane is the PRIMARY path and the scalar
;     codepoint loop is the tail + oracle. At a codepoint/unit boundary, when
;     the output is little-endian (be==0) and 16 source bytes/units and 16 dst
;     units/bytes fit, we vector-probe the chunk: from_utf8 loads 16 bytes,
;     `icmp slt` vs 0 flags any high bit (>=0x80); if none, `zext <16 x i8> ->
;     <16 x i16>` widens 16 ASCII bytes to 16 LE code units in one store.
;     to_utf8 loads 16 code units, `icmp ugt` vs 127 flags any non-ASCII; if
;     none, `trunc <16 x i16> -> <16 x i8>` narrows 16 units to 16 bytes in one
;     store. The instant a lane is non-ASCII (or fewer than 16 remain, or be!=0)
;     we fall to the scalar per-codepoint path, which handles every multibyte
;     sequence and surrogate pair and is the cross-check oracle in tests. ASCII
;     lanes are never surrogates/overlong, so accept/reject and the emitted
;     bytes are byte-identical to the scalar path. Both lower to SSE2 / NEON at
;     the 128-bit baseline (no runtime feature check).
;
; ERROR CONVENTION (documented; matches sibling encoding modules): i64-returning
; entry points return a NON-NEGATIVE result on success (a scalar value, a unit
; count, a byte count, or units/bytes written) and a NEGATIVE error code on
; failure: -8 INVALID_ARG (encoding an invalid scalar; decoding an empty range),
; -13 PARSE (malformed UTF-16: lone/reversed/truncated surrogate, or invalid
; UTF-8 input), -6 FULL (dst capacity too small). scalar_units returns i32
; (1|2, or -8). bom_detect returns i32 (0 none, 1 LE BOM FF FE, 2 BE BOM FE FF).
;
; API:
;   i32 universe_utf16_scalar_units(i32 scalar)                       ; 1|2, or -8
;   i64 universe_utf16_encode_scalar(ptr dst, i32 scalar, i32 be)     ; units, or -8
;   i64 universe_utf16_decode_scalar(ptr src, i64 navail, i32 be)     ; scalar, or -8/-13
;   i32 universe_utf16_bom_detect(ptr src, i64 nbytes)                ; 0|1|2
;   i64 universe_utf16_len_from_utf8(ptr src, i64 n)                  ; utf16 units, or -13
;   i64 universe_utf16_len_to_utf8(ptr src, i64 nunits, i32 be)       ; utf8 bytes, or -13/-8
;   i64 universe_utf16_from_utf8(ptr dst, i64 dcap, ptr src, i64 n, i32 be)      ; units, or <0
;   i64 universe_utf16_to_utf8(ptr dst, i64 dcap, ptr src, i64 nunits, i32 be)   ; bytes, or <0

declare i16 @llvm.bswap.i16(i16)
declare i1 @llvm.vector.reduce.or.v16i1(<16 x i1>)

; reused from the sibling UTF-8 module (functional reuse, not reimplemented)
declare i64 @universe_utf8_validate(ptr, i64)
declare i32 @universe_utf8_byte_len_of_codepoint(i8)

; ---------------------------------------------------------------- u16 load/store
define internal i32 @u16ld(ptr %base, i64 %idx, i32 %be) #4 {
entry:
  %boff = shl nuw i64 %idx, 1
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %boff
  %v = load i16, ptr %p, align 1
  %sw = call i16 @llvm.bswap.i16(i16 %v)
  %isbe = icmp ne i32 %be, 0
  %sel = select i1 %isbe, i16 %sw, i16 %v
  %z = zext i16 %sel to i32
  ret i32 %z
}

define internal void @u16st(ptr %base, i64 %idx, i32 %val, i32 %be) #5 {
entry:
  %boff = shl nuw i64 %idx, 1
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %boff
  %v16 = trunc i32 %val to i16
  %sw = call i16 @llvm.bswap.i16(i16 %v16)
  %isbe = icmp ne i32 %be, 0
  %sel = select i1 %isbe, i16 %sw, i16 %v16
  store i16 %sel, ptr %p, align 1
  ret void
}

; ---------------------------------------------------------------- scalar helpers
; valid scalar: not a surrogate (U+D800..U+DFFF) and <= U+10FFFF
define internal i1 @svalid(i32 %s) #6 {
entry:
  %hi = icmp ugt i32 %s, 1114111
  %sa = icmp uge i32 %s, 55296
  %sb = icmp ule i32 %s, 57343
  %sur = and i1 %sa, %sb
  %bad = or i1 %hi, %sur
  %ok = xor i1 %bad, true
  ret i1 %ok
}

; UTF-8 byte length of a valid scalar: 1/2/3/4
define internal i32 @u8len(i32 %s) #6 {
entry:
  %a = icmp ult i32 %s, 128
  %b = icmp ult i32 %s, 2048
  %c = icmp ult i32 %s, 65536
  %r3 = select i1 %c, i32 3, i32 4
  %r2 = select i1 %b, i32 2, i32 %r3
  %r1 = select i1 %a, i32 1, i32 %r2
  ret i32 %r1
}

; decode a known-valid UTF-8 sequence of length %len at byte index %i -> scalar
define internal i32 @u8dec(ptr %src, i64 %i, i64 %len) #4 {
entry:
  %p0 = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i32
  switch i64 %len, label %one [ i64 2, label %two
                                i64 3, label %three
                                i64 4, label %four ]
one:
  ret i32 %z0

two:
  %t.a = and i32 %z0, 31
  %t.as = shl nuw i32 %t.a, 6
  %t.i1 = add nuw i64 %i, 1
  %t.p1 = getelementptr inbounds nuw i8, ptr %src, i64 %t.i1
  %t.b1 = load i8, ptr %t.p1, align 1
  %t.z1 = zext i8 %t.b1 to i32
  %t.c1 = and i32 %t.z1, 63
  %t.r = or disjoint i32 %t.as, %t.c1
  ret i32 %t.r

three:
  %h.a = and i32 %z0, 15
  %h.as = shl nuw i32 %h.a, 12
  %h.i1 = add nuw i64 %i, 1
  %h.p1 = getelementptr inbounds nuw i8, ptr %src, i64 %h.i1
  %h.b1 = load i8, ptr %h.p1, align 1
  %h.z1 = zext i8 %h.b1 to i32
  %h.c1 = and i32 %h.z1, 63
  %h.c1s = shl nuw i32 %h.c1, 6
  %h.i2 = add nuw i64 %i, 2
  %h.p2 = getelementptr inbounds nuw i8, ptr %src, i64 %h.i2
  %h.b2 = load i8, ptr %h.p2, align 1
  %h.z2 = zext i8 %h.b2 to i32
  %h.c2 = and i32 %h.z2, 63
  %h.o1 = or disjoint i32 %h.as, %h.c1s
  %h.r = or disjoint i32 %h.o1, %h.c2
  ret i32 %h.r

four:
  %f.a = and i32 %z0, 7
  %f.as = shl nuw i32 %f.a, 18
  %f.i1 = add nuw i64 %i, 1
  %f.p1 = getelementptr inbounds nuw i8, ptr %src, i64 %f.i1
  %f.b1 = load i8, ptr %f.p1, align 1
  %f.z1 = zext i8 %f.b1 to i32
  %f.c1 = and i32 %f.z1, 63
  %f.c1s = shl nuw i32 %f.c1, 12
  %f.i2 = add nuw i64 %i, 2
  %f.p2 = getelementptr inbounds nuw i8, ptr %src, i64 %f.i2
  %f.b2 = load i8, ptr %f.p2, align 1
  %f.z2 = zext i8 %f.b2 to i32
  %f.c2 = and i32 %f.z2, 63
  %f.c2s = shl nuw i32 %f.c2, 6
  %f.i3 = add nuw i64 %i, 3
  %f.p3 = getelementptr inbounds nuw i8, ptr %src, i64 %f.i3
  %f.b3 = load i8, ptr %f.p3, align 1
  %f.z3 = zext i8 %f.b3 to i32
  %f.c3 = and i32 %f.z3, 63
  %f.o1 = or disjoint i32 %f.as, %f.c1s
  %f.o2 = or disjoint i32 %f.o1, %f.c2s
  %f.r = or disjoint i32 %f.o2, %f.c3
  ret i32 %f.r
}

; write a valid scalar as UTF-8 at byte index %off; returns bytes written (1..4)
define internal i32 @u8put(ptr %dst, i64 %off, i32 %s) #5 {
entry:
  %len = call i32 @u8len(i32 %s)
  switch i32 %len, label %one [ i32 2, label %two
                                i32 3, label %three
                                i32 4, label %four ]
one:
  %o.b0 = trunc i32 %s to i8
  %o.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  store i8 %o.b0, ptr %o.p0, align 1
  ret i32 1

two:
  %t.h = lshr i32 %s, 6
  %t.b0i = or i32 %t.h, 192
  %t.b0 = trunc i32 %t.b0i to i8
  %t.l = and i32 %s, 63
  %t.b1i = or disjoint i32 %t.l, 128
  %t.b1 = trunc i32 %t.b1i to i8
  %t.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  store i8 %t.b0, ptr %t.p0, align 1
  %t.o1 = add nuw i64 %off, 1
  %t.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %t.o1
  store i8 %t.b1, ptr %t.p1, align 1
  ret i32 2

three:
  %h.h = lshr i32 %s, 12
  %h.b0i = or i32 %h.h, 224
  %h.b0 = trunc i32 %h.b0i to i8
  %h.m = lshr i32 %s, 6
  %h.m6 = and i32 %h.m, 63
  %h.b1i = or disjoint i32 %h.m6, 128
  %h.b1 = trunc i32 %h.b1i to i8
  %h.l = and i32 %s, 63
  %h.b2i = or disjoint i32 %h.l, 128
  %h.b2 = trunc i32 %h.b2i to i8
  %h.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  store i8 %h.b0, ptr %h.p0, align 1
  %h.o1 = add nuw i64 %off, 1
  %h.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %h.o1
  store i8 %h.b1, ptr %h.p1, align 1
  %h.o2 = add nuw i64 %off, 2
  %h.p2 = getelementptr inbounds nuw i8, ptr %dst, i64 %h.o2
  store i8 %h.b2, ptr %h.p2, align 1
  ret i32 3

four:
  %f.h = lshr i32 %s, 18
  %f.b0i = or i32 %f.h, 240
  %f.b0 = trunc i32 %f.b0i to i8
  %f.m1 = lshr i32 %s, 12
  %f.m1a = and i32 %f.m1, 63
  %f.b1i = or disjoint i32 %f.m1a, 128
  %f.b1 = trunc i32 %f.b1i to i8
  %f.m2 = lshr i32 %s, 6
  %f.m2a = and i32 %f.m2, 63
  %f.b2i = or disjoint i32 %f.m2a, 128
  %f.b2 = trunc i32 %f.b2i to i8
  %f.l = and i32 %s, 63
  %f.b3i = or disjoint i32 %f.l, 128
  %f.b3 = trunc i32 %f.b3i to i8
  %f.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %off
  store i8 %f.b0, ptr %f.p0, align 1
  %f.o1 = add nuw i64 %off, 1
  %f.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %f.o1
  store i8 %f.b1, ptr %f.p1, align 1
  %f.o2 = add nuw i64 %off, 2
  %f.p2 = getelementptr inbounds nuw i8, ptr %dst, i64 %f.o2
  store i8 %f.b2, ptr %f.p2, align 1
  %f.o3 = add nuw i64 %off, 3
  %f.p3 = getelementptr inbounds nuw i8, ptr %dst, i64 %f.o3
  store i8 %f.b3, ptr %f.p3, align 1
  ret i32 4
}

; ---------------------------------------------------------------- scalar_units
define i32 @universe_utf16_scalar_units(i32 %s) local_unnamed_addr #3 {
entry:
  %ok = call i1 @svalid(i32 %s)
  br i1 %ok, label %good, label %bad, !prof !0

bad:                                              ; cold
  ret i32 -8

good:
  %astral = icmp uge i32 %s, 65536
  %u = select i1 %astral, i32 2, i32 1
  ret i32 %u
}

; ---------------------------------------------------------------- encode_scalar
define i64 @universe_utf16_encode_scalar(ptr %dst, i32 %s, i32 %be) local_unnamed_addr #1 {
entry:
  %ok = call i1 @svalid(i32 %s)
  br i1 %ok, label %good, label %bad, !prof !0

bad:                                              ; cold
  ret i64 -8

good:
  %astral = icmp uge i32 %s, 65536
  br i1 %astral, label %pair, label %single

single:
  call void @u16st(ptr %dst, i64 0, i32 %s, i32 %be)
  ret i64 1

pair:
  %s2 = sub nuw i32 %s, 65536
  %h = lshr i32 %s2, 10
  %hi = or disjoint i32 %h, 55296
  %l = and i32 %s2, 1023
  %lo = or disjoint i32 %l, 56320
  call void @u16st(ptr %dst, i64 0, i32 %hi, i32 %be)
  call void @u16st(ptr %dst, i64 1, i32 %lo, i32 %be)
  ret i64 2
}

; ---------------------------------------------------------------- decode_scalar
define i64 @universe_utf16_decode_scalar(ptr %src, i64 %navail, i32 %be) local_unnamed_addr #0 {
entry:
  %has = icmp uge i64 %navail, 1
  br i1 %has, label %load0, label %bad8, !prof !1

bad8:                                             ; cold
  ret i64 -8

load0:
  %u0 = call i32 @u16ld(ptr %src, i64 0, i32 %be)
  %hia = icmp uge i32 %u0, 55296
  %hib = icmp ule i32 %u0, 56319
  %isHi = and i1 %hia, %hib
  %loa = icmp uge i32 %u0, 56320
  %lob = icmp ule i32 %u0, 57343
  %isLo = and i1 %loa, %lob
  %isSurr = or i1 %isHi, %isLo
  br i1 %isSurr, label %surr, label %bmp, !prof !1

bmp:
  %z = zext i32 %u0 to i64
  ret i64 %z

surr:
  br i1 %isHi, label %hi, label %badP, !prof !0

hi:
  %has2 = icmp uge i64 %navail, 2
  br i1 %has2, label %load1, label %badP, !prof !0

load1:
  %u1 = call i32 @u16ld(ptr %src, i64 1, i32 %be)
  %l1a = icmp uge i32 %u1, 56320
  %l1b = icmp ule i32 %u1, 57343
  %loOk = and i1 %l1a, %l1b
  br i1 %loOk, label %combine, label %badP, !prof !0

combine:
  %hd = sub nuw i32 %u0, 55296
  %hs = shl nuw i32 %hd, 10
  %ld = sub nuw i32 %u1, 56320
  %sum = add nuw i32 %hs, %ld
  %sc = add nuw i32 %sum, 65536
  %z2 = zext i32 %sc to i64
  ret i64 %z2

badP:                                             ; cold — lone/reversed/truncated surrogate
  ret i64 -13
}

; ---------------------------------------------------------------- bom_detect
define i32 @universe_utf16_bom_detect(ptr %src, i64 %nbytes) local_unnamed_addr #0 {
entry:
  %has2 = icmp uge i64 %nbytes, 2
  br i1 %has2, label %chk, label %none

none:
  ret i32 0

chk:
  %p0 = getelementptr inbounds nuw i8, ptr %src, i64 0
  %b0 = load i8, ptr %p0, align 1
  %p1 = getelementptr inbounds nuw i8, ptr %src, i64 1
  %b1 = load i8, ptr %p1, align 1
  %z0 = zext i8 %b0 to i32
  %z1 = zext i8 %b1 to i32
  %ff0 = icmp eq i32 %z0, 255
  %fe1 = icmp eq i32 %z1, 254
  %le = and i1 %ff0, %fe1
  %fe0 = icmp eq i32 %z0, 254
  %ff1 = icmp eq i32 %z1, 255
  %be = and i1 %fe0, %ff1
  %r1 = select i1 %be, i32 2, i32 0
  %r = select i1 %le, i32 1, i32 %r1
  ret i32 %r
}

; ---------------------------------------------------------------- len_from_utf8
; number of UTF-16 code units needed to represent a valid UTF-8 buffer
define i64 @universe_utf16_len_from_utf8(ptr %src, i64 %n) local_unnamed_addr #0 {
entry:
  %vr = call i64 @universe_utf8_validate(ptr %src, i64 %n)
  %valid = icmp eq i64 %vr, -1
  br i1 %valid, label %pre, label %bad, !prof !0

bad:                                              ; cold
  ret i64 -13

pre:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %zero, label %loop

zero:
  ret i64 0

loop:
  %i = phi i64 [ 0, %pre ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %pre ], [ %acc.next, %loop ]
  %lp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %lead = load i8, ptr %lp, align 1
  %len32 = call i32 @universe_utf8_byte_len_of_codepoint(i8 %lead)
  %len = zext i32 %len32 to i64
  %is4 = icmp eq i64 %len, 4
  %u = select i1 %is4, i64 2, i64 1
  %acc.next = add nuw i64 %acc, %u
  %i.next = add nuw i64 %i, %len
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %fin

fin:
  ret i64 %acc.next
}

; ---------------------------------------------------------------- len_to_utf8
; number of UTF-8 bytes needed to represent a UTF-16 unit buffer
define i64 @universe_utf16_len_to_utf8(ptr %src, i64 %nunits, i32 %be) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %nunits, 0
  br i1 %empty, label %zero, label %loop

zero:
  ret i64 0

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %cont ]
  %boff = shl nuw i64 %i, 1
  %srcp = getelementptr inbounds nuw i8, ptr %src, i64 %boff
  %avail = sub nuw i64 %nunits, %i
  %sc64 = call i64 @universe_utf16_decode_scalar(ptr %srcp, i64 %avail, i32 %be)
  %err = icmp slt i64 %sc64, 0
  br i1 %err, label %bad, label %ok, !prof !1

bad:                                              ; cold
  ret i64 %sc64

ok:
  %sc = trunc i64 %sc64 to i32
  %blen32 = call i32 @u8len(i32 %sc)
  %blen = zext i32 %blen32 to i64
  %astral = icmp uge i32 %sc, 65536
  %uc = select i1 %astral, i64 2, i64 1
  br label %cont

cont:
  %acc.next = add nuw i64 %acc, %blen
  %i.next = add nuw i64 %i, %uc
  %more = icmp ult i64 %i.next, %nunits
  br i1 %more, label %loop, label %fin

fin:
  ret i64 %acc.next
}

; ---------------------------------------------------------------- from_utf8
; transcode UTF-8 -> UTF-16 into caller-sized dst (dcap units); units written or <0
define i64 @universe_utf16_from_utf8(ptr %dst, i64 %dcap, ptr %src, i64 %n, i32 %be) local_unnamed_addr #2 {
entry:
  %vr = call i64 @universe_utf8_validate(ptr %src, i64 %n)
  %valid = icmp eq i64 %vr, -1
  br i1 %valid, label %loop, label %bad, !prof !0

bad:                                              ; cold — malformed UTF-8
  ret i64 -13

; head-checked loop over codepoint boundaries; %out accumulates units written.
; An empty range falls straight through to %fin returning 0.
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ], [ %vi2, %vemit ]
  %out = phi i64 [ 0, %entry ], [ %oend, %cont ], [ %vout2, %vemit ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %fin, label %vprobe

fin:
  ret i64 %out

; SIMD-first ASCII fast path: little-endian output, 16 src bytes and 16 dst
; units in range. Non-ASCII lane / short remainder / big-endian -> scalar body.
vprobe:
  %isle = icmp eq i32 %be, 0
  %i16 = add nuw i64 %i, 16
  %ifit = icmp ule i64 %i16, %n
  %vc0 = and i1 %isle, %ifit
  %o16 = add nuw i64 %out, 16
  %ofit = icmp ule i64 %o16, %dcap
  %vok = and i1 %vc0, %ofit
  br i1 %vok, label %vtry, label %body

vtry:
  %vp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %vv = load <16 x i8>, ptr %vp, align 1
  %vneg = icmp slt <16 x i8> %vv, zeroinitializer   ; high bit set -> >= 0x80
  %vany = call i1 @llvm.vector.reduce.or.v16i1(<16 x i1> %vneg)
  br i1 %vany, label %body, label %vemit, !prof !1

vemit:
  %vw = zext <16 x i8> %vv to <16 x i16>
  %obytes = shl nuw i64 %out, 1
  %vdp = getelementptr inbounds nuw i8, ptr %dst, i64 %obytes
  store <16 x i16> %vw, ptr %vdp, align 1
  %vi2 = add nuw i64 %i, 16
  %vout2 = add nuw i64 %out, 16
  br label %loop

body:
  %lp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %lead = load i8, ptr %lp, align 1
  %len32 = call i32 @universe_utf8_byte_len_of_codepoint(i8 %lead)
  %len = zext i32 %len32 to i64
  %scalar = call i32 @u8dec(ptr %src, i64 %i, i64 %len)
  %astral = icmp uge i32 %scalar, 65536
  %units = select i1 %astral, i64 2, i64 1
  %oend = add nuw i64 %out, %units
  %fits = icmp ule i64 %oend, %dcap
  br i1 %fits, label %emit, label %full, !prof !0

full:                                             ; cold — dst too small
  ret i64 -6

emit:
  br i1 %astral, label %emit2, label %emit1

emit1:
  call void @u16st(ptr %dst, i64 %out, i32 %scalar, i32 %be)
  br label %cont

emit2:
  %s2 = sub nuw i32 %scalar, 65536
  %h10 = lshr i32 %s2, 10
  %hiu = or disjoint i32 %h10, 55296
  %l10 = and i32 %s2, 1023
  %lou = or disjoint i32 %l10, 56320
  call void @u16st(ptr %dst, i64 %out, i32 %hiu, i32 %be)
  %out1 = add nuw i64 %out, 1
  call void @u16st(ptr %dst, i64 %out1, i32 %lou, i32 %be)
  br label %cont

cont:
  %i.next = add nuw i64 %i, %len
  br label %loop
}

; ---------------------------------------------------------------- to_utf8
; transcode UTF-16 -> UTF-8 into caller-sized dst (dcap bytes); bytes written or <0
define i64 @universe_utf16_to_utf8(ptr %dst, i64 %dcap, ptr %src, i64 %nunits, i32 %be) local_unnamed_addr #2 {
entry:
  br label %loop

; head-checked loop over unit boundaries; %out accumulates bytes written. An
; empty range falls straight through to %fin returning 0.
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ], [ %vi2, %vemit ]
  %out = phi i64 [ 0, %entry ], [ %oend, %cont ], [ %vout2, %vemit ]
  %done = icmp uge i64 %i, %nunits
  br i1 %done, label %fin, label %vprobe

fin:
  ret i64 %out

; SIMD-first ASCII fast path: little-endian input, 16 src units and 16 dst
; bytes in range. ASCII units (< 0x80) are never surrogates, so no malformed
; case can hide here. Non-ASCII lane / short remainder / big-endian -> scalar.
vprobe:
  %isle = icmp eq i32 %be, 0
  %i16 = add nuw i64 %i, 16
  %ifit = icmp ule i64 %i16, %nunits
  %vc0 = and i1 %isle, %ifit
  %o16 = add nuw i64 %out, 16
  %ofit = icmp ule i64 %o16, %dcap
  %vok = and i1 %vc0, %ofit
  br i1 %vok, label %vtry, label %body

vtry:
  %sboff = shl nuw i64 %i, 1
  %vp = getelementptr inbounds nuw i8, ptr %src, i64 %sboff
  %vv = load <16 x i16>, ptr %vp, align 1
  %vhi = icmp ugt <16 x i16> %vv, splat (i16 127)   ; any unit >= 0x80
  %vany = call i1 @llvm.vector.reduce.or.v16i1(<16 x i1> %vhi)
  br i1 %vany, label %body, label %vemit, !prof !1

vemit:
  %vn = trunc <16 x i16> %vv to <16 x i8>
  %vdp = getelementptr inbounds nuw i8, ptr %dst, i64 %out
  store <16 x i8> %vn, ptr %vdp, align 1
  %vi2 = add nuw i64 %i, 16
  %vout2 = add nuw i64 %out, 16
  br label %loop

body:
  %boff = shl nuw i64 %i, 1
  %srcp = getelementptr inbounds nuw i8, ptr %src, i64 %boff
  %avail = sub nuw i64 %nunits, %i
  %sc64 = call i64 @universe_utf16_decode_scalar(ptr %srcp, i64 %avail, i32 %be)
  %err = icmp slt i64 %sc64, 0
  br i1 %err, label %bad, label %ok, !prof !1

bad:                                              ; cold — malformed UTF-16
  ret i64 %sc64

ok:
  %sc = trunc i64 %sc64 to i32
  %blen32 = call i32 @u8len(i32 %sc)
  %blen = zext i32 %blen32 to i64
  %oend = add nuw i64 %out, %blen
  %fits = icmp ule i64 %oend, %dcap
  br i1 %fits, label %emit, label %full, !prof !0

full:                                             ; cold — dst too small
  ret i64 -6

emit:
  %w = call i32 @u8put(ptr %dst, i64 %out, i32 %sc)
  %astral = icmp uge i32 %sc, 65536
  %uc = select i1 %astral, i64 2, i64 1
  br label %cont

cont:
  %i.next = add nuw i64 %i, %uc
  br label %loop
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #5 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #6 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
