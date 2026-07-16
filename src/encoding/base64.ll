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

; Base64 encode/decode, standard AND url-safe alphabets. Pure compute over
; caller-supplied byte ranges; the caller pre-sizes `dst` (via encode_len /
; decode_len) and we never allocate.
;
; DESIGN:
;   * ALPHABET SELECT (documented): a single `i32 urlsafe` flag arg picks the
;     alphabet. 0 = standard ('+' 0x2B, '/' 0x2F). non-zero = url-safe
;     ('-' 0x2D, '_' 0x5F). Only symbols 62/63 differ; every other code point
;     is shared, so the flag folds into two `select`s that feed the same
;     branchless transform. BOTH alphabets emit '=' padding, so decode_len is
;     exact for either.
;   * ERROR CONVENTION (documented): i64 return, NEGATIVE on failure.
;     encode_len returns -1 (SIZE_OVERFLOW) if the padded length overflows i64.
;     decode_len / decode return -1 (PARSE) for a length not a multiple of 4,
;     misplaced padding, or any character outside the selected alphabet.
;   * SIMD-FIRST (128-bit primary + scalar fallback, per the house rule). The
;     PUBLIC encode/decode are the VECTOR entries; each ships a `_scalar` twin
;     that is the fallback, the ragged-tail / padding handler, AND the cross-
;     check oracle (tests assert vector == scalar for every length + both
;     alphabets, on top of the RFC 4648 KATs).
;       - encode: each iteration turns 12 input bytes into 16 output chars.
;         A `shufflevector` (NEON `tbl` / x86 `pshufb`) reshuffles the 12 bytes
;         so each 32-bit lane holds one group; two 16-bit-lane multiplies
;         (mulhi/mullo — the classic field-spread trick) extract the four 6-bit
;         indices per group; a branch-free vector `select` chain (cmeq/cmhi +
;         bsl) maps 0..63 -> ascii. Runs while >= 16 src bytes remain (the 16-B
;         load is in-bounds); the final <16 bytes + padding go to the scalar
;         twin.
;       - decode: each iteration turns 16 input chars into 12 bytes. A
;         vectorized range-classify (same logic as @b64_dec) translates ascii
;         -> 6-bit and reduces an OR of the invalid lanes into one bad flag;
;         per-lane i32 arithmetic repacks four 6-bit values into 3 bytes, and a
;         `shufflevector` gathers the 12 output bytes. Runs on the interior
;         groups only; the final group (which may carry '=' padding) is scalar.
;   * Compute/memory separation: batch the group loads, compute in vector
;     registers, batch the stores — never load-compute-store per output char.
;
; API (i32 urlsafe: 0 = standard, non-zero = url-safe):
;   i64 universe_base64_encode_len(i64 n)                    ; padded len, or -1
;   i64 universe_base64_decode_len(ptr src, i64 n)           ; exact, or -1
;   i64 universe_base64_encode(ptr dst, ptr src, i64 n, i32 urlsafe)         ; vector
;   i64 universe_base64_encode_scalar(ptr dst, ptr src, i64 n, i32 urlsafe)  ; oracle
;   i64 universe_base64_decode(ptr dst, ptr src, i64 n, i32 urlsafe)         ; vector
;   i64 universe_base64_decode_scalar(ptr dst, ptr src, i64 n, i32 urlsafe)  ; oracle

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; 6-bit index -> ascii, branchless, alphabet-parameterized.
define internal i32 @b64_enc(i32 %i, i32 %plus, i32 %slash) #3 {
entry:
  %c0 = add nuw nsw i32 %i, 65                  ; 'A'..  (0..25)
  %g25 = icmp ugt i32 %i, 25
  %av = add nuw nsw i32 %i, 71                  ; 'a'-26 = 71   (26..51)
  %c1 = select i1 %g25, i32 %av, i32 %c0
  %g51 = icmp ugt i32 %i, 51
  %dv = add nsw i32 %i, -4                       ; '0'-52 = -4  (52..61)
  %c2 = select i1 %g51, i32 %dv, i32 %c1
  %e62 = icmp eq i32 %i, 62
  %c3 = select i1 %e62, i32 %plus, i32 %c2
  %e63 = icmp eq i32 %i, 63
  %c4 = select i1 %e63, i32 %slash, i32 %c3
  ret i32 %c4
}

; ascii byte -> 6-bit value in 0..63, or -1 if not in the selected alphabet.
define internal i32 @b64_dec(i32 %c, i32 %plus, i32 %slash) #3 {
entry:
  %uge = icmp uge i32 %c, 65
  %ule = icmp ule i32 %c, 90
  %uok = and i1 %uge, %ule                       ; 'A'..'Z' -> c-65
  %uv = sub nsw i32 %c, 65
  %lge = icmp uge i32 %c, 97
  %lle = icmp ule i32 %c, 122
  %lok = and i1 %lge, %lle                       ; 'a'..'z' -> c-71
  %lv = sub nsw i32 %c, 71
  %dge = icmp uge i32 %c, 48
  %dle = icmp ule i32 %c, 57
  %dok = and i1 %dge, %dle                        ; '0'..'9' -> c+4
  %dvv = add nsw i32 %c, 4
  %pok = icmp eq i32 %c, %plus
  %sok = icmp eq i32 %c, %slash
  %r0 = select i1 %uok, i32 %uv, i32 -1
  %r1 = select i1 %lok, i32 %lv, i32 %r0
  %r2 = select i1 %dok, i32 %dvv, i32 %r1
  %r3 = select i1 %pok, i32 62, i32 %r2
  %r4 = select i1 %sok, i32 63, i32 %r3
  ret i32 %r4
}

; ------------------------------------------------------------------ encode_len
define i64 @universe_base64_encode_len(i64 %n) local_unnamed_addr #2 {
entry:
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %n, i64 2)
  %np2 = extractvalue { i64, i1 } %a, 0
  %ao = extractvalue { i64, i1 } %a, 1
  %groups = udiv i64 %np2, 3
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %groups, i64 4)
  %out = extractvalue { i64, i1 } %m, 0
  %mo = extractvalue { i64, i1 } %m, 1
  %ovf = or i1 %ao, %mo
  %r = select i1 %ovf, i64 -1, i64 %out
  ret i64 %r
}

; ------------------------------------------------------------------ decode_len
; Exact decoded size: (n/4)*3 minus the trailing '=' count. -1 if n%4 != 0 or
; padding is malformed (second-from-last is '=' but last is not).
define i64 @universe_base64_decode_len(ptr %src, i64 %n) local_unnamed_addr #4 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %chk, !prof !1

zero:
  ret i64 0

chk:
  %rem = and i64 %n, 3
  %bad.len = icmp ne i64 %rem, 0
  br i1 %bad.len, label %err, label %work, !prof !0

err:                                              ; cold
  ret i64 -1

work:
  %ng = lshr i64 %n, 2
  %base = mul nuw i64 %ng, 3
  %off.last = sub nuw i64 %n, 1
  %lp2 = getelementptr inbounds nuw i8, ptr %src, i64 %off.last
  %last = load i8, ptr %lp2, align 1
  %off.second = sub nuw i64 %n, 2
  %sp2 = getelementptr inbounds nuw i8, ptr %src, i64 %off.second
  %second = load i8, ptr %sp2, align 1
  %pl = icmp eq i8 %last, 61                       ; '='
  %ps = icmp eq i8 %second, 61
  ; malformed: second is '=' but last is not
  %not.pl = xor i1 %pl, true
  %malformed = and i1 %ps, %not.pl
  br i1 %malformed, label %err, label %count, !prof !0

count:
  %pl.i = zext i1 %pl to i64
  %ps.i = zext i1 %ps to i64
  %pad = add nuw nsw i64 %pl.i, %ps.i
  %out = sub nuw i64 %base, %pad
  ret i64 %out
}

; --------------------------------------------------------------- encode (scalar)
; The scalar path is the FALLBACK, the tail handler (final partial group +
; padding), AND the cross-check oracle for the vector entry below.
define i64 @universe_base64_encode_scalar(ptr %dst, ptr %src, i64 %n, i32 %urlsafe) local_unnamed_addr #0 {
entry:
  %url = icmp ne i32 %urlsafe, 0
  %plus = select i1 %url, i32 45, i32 43         ; '-' : '+'
  %slash = select i1 %url, i32 95, i32 47        ; '_' : '/'
  %full = udiv i64 %n, 3
  %rem = urem i64 %n, 3
  %has.groups = icmp ne i64 %full, 0
  br i1 %has.groups, label %loop, label %tail

loop:
  %g = phi i64 [ 0, %entry ], [ %g.next, %loop ]
  %soff = mul nuw i64 %g, 3
  %sp0 = getelementptr inbounds nuw i8, ptr %src, i64 %soff
  %b0 = load i8, ptr %sp0, align 1
  %sp1 = getelementptr inbounds nuw i8, ptr %sp0, i64 1
  %b1 = load i8, ptr %sp1, align 1
  %sp2 = getelementptr inbounds nuw i8, ptr %sp0, i64 2
  %b2 = load i8, ptr %sp2, align 1
  %z0 = zext i8 %b0 to i32
  %z1 = zext i8 %b1 to i32
  %z2 = zext i8 %b2 to i32
  ; four 6-bit indices
  %i0 = lshr i32 %z0, 2
  %t0a = and i32 %z0, 3
  %t0b = shl nuw nsw i32 %t0a, 4
  %t0c = lshr i32 %z1, 4
  %i1 = or disjoint i32 %t0b, %t0c
  %t1a = and i32 %z1, 15
  %t1b = shl nuw nsw i32 %t1a, 2
  %t1c = lshr i32 %z2, 6
  %i2 = or disjoint i32 %t1b, %t1c
  %i3 = and i32 %z2, 63
  %e0 = call i32 @b64_enc(i32 %i0, i32 %plus, i32 %slash)
  %e1 = call i32 @b64_enc(i32 %i1, i32 %plus, i32 %slash)
  %e2 = call i32 @b64_enc(i32 %i2, i32 %plus, i32 %slash)
  %e3 = call i32 @b64_enc(i32 %i3, i32 %plus, i32 %slash)
  %ooff = mul nuw i64 %g, 4
  %dp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %ooff
  %o0 = trunc i32 %e0 to i8
  store i8 %o0, ptr %dp0, align 1
  %dp1 = getelementptr inbounds nuw i8, ptr %dp0, i64 1
  %o1 = trunc i32 %e1 to i8
  store i8 %o1, ptr %dp1, align 1
  %dp2 = getelementptr inbounds nuw i8, ptr %dp0, i64 2
  %o2 = trunc i32 %e2 to i8
  store i8 %o2, ptr %dp2, align 1
  %dp3 = getelementptr inbounds nuw i8, ptr %dp0, i64 3
  %o3 = trunc i32 %e3 to i8
  store i8 %o3, ptr %dp3, align 1
  %g.next = add nuw i64 %g, 1
  %more = icmp ult i64 %g.next, %full
  br i1 %more, label %loop, label %tail

tail:
  %toff.s = mul nuw i64 %full, 3
  %toff.o = mul nuw i64 %full, 4
  ; dispatch on remainder 0/1/2
  switch i64 %rem, label %ret.none [ i64 1, label %one
                                     i64 2, label %two ]

one:
  %o.sp = getelementptr inbounds nuw i8, ptr %src, i64 %toff.s
  %o.b0 = load i8, ptr %o.sp, align 1
  %o.z0 = zext i8 %o.b0 to i32
  %o.i0 = lshr i32 %o.z0, 2
  %o.t = and i32 %o.z0, 3
  %o.i1 = shl nuw nsw i32 %o.t, 4
  %o.e0 = call i32 @b64_enc(i32 %o.i0, i32 %plus, i32 %slash)
  %o.e1 = call i32 @b64_enc(i32 %o.i1, i32 %plus, i32 %slash)
  %o.dp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %toff.o
  %o.o0 = trunc i32 %o.e0 to i8
  store i8 %o.o0, ptr %o.dp0, align 1
  %o.dp1 = getelementptr inbounds nuw i8, ptr %o.dp0, i64 1
  %o.o1 = trunc i32 %o.e1 to i8
  store i8 %o.o1, ptr %o.dp1, align 1
  %o.dp2 = getelementptr inbounds nuw i8, ptr %o.dp0, i64 2
  store i8 61, ptr %o.dp2, align 1
  %o.dp3 = getelementptr inbounds nuw i8, ptr %o.dp0, i64 3
  store i8 61, ptr %o.dp3, align 1
  %o.out = add nuw i64 %toff.o, 4
  ret i64 %o.out

two:
  %w.sp = getelementptr inbounds nuw i8, ptr %src, i64 %toff.s
  %w.b0 = load i8, ptr %w.sp, align 1
  %w.sp1 = getelementptr inbounds nuw i8, ptr %w.sp, i64 1
  %w.b1 = load i8, ptr %w.sp1, align 1
  %w.z0 = zext i8 %w.b0 to i32
  %w.z1 = zext i8 %w.b1 to i32
  %w.i0 = lshr i32 %w.z0, 2
  %w.t0 = and i32 %w.z0, 3
  %w.t0s = shl nuw nsw i32 %w.t0, 4
  %w.t0c = lshr i32 %w.z1, 4
  %w.i1 = or disjoint i32 %w.t0s, %w.t0c
  %w.t1 = and i32 %w.z1, 15
  %w.i2 = shl nuw nsw i32 %w.t1, 2
  %w.e0 = call i32 @b64_enc(i32 %w.i0, i32 %plus, i32 %slash)
  %w.e1 = call i32 @b64_enc(i32 %w.i1, i32 %plus, i32 %slash)
  %w.e2 = call i32 @b64_enc(i32 %w.i2, i32 %plus, i32 %slash)
  %w.dp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %toff.o
  %w.o0 = trunc i32 %w.e0 to i8
  store i8 %w.o0, ptr %w.dp0, align 1
  %w.dp1 = getelementptr inbounds nuw i8, ptr %w.dp0, i64 1
  %w.o1 = trunc i32 %w.e1 to i8
  store i8 %w.o1, ptr %w.dp1, align 1
  %w.dp2 = getelementptr inbounds nuw i8, ptr %w.dp0, i64 2
  %w.o2 = trunc i32 %w.e2 to i8
  store i8 %w.o2, ptr %w.dp2, align 1
  %w.dp3 = getelementptr inbounds nuw i8, ptr %w.dp0, i64 3
  store i8 61, ptr %w.dp3, align 1
  %w.out = add nuw i64 %toff.o, 4
  ret i64 %w.out

ret.none:
  ret i64 %toff.o
}

; --------------------------------------------------------------- decode (scalar)
; Scalar FALLBACK + final-group/padding handler + cross-check oracle.
define i64 @universe_base64_decode_scalar(ptr %dst, ptr %src, i64 %n, i32 %urlsafe) local_unnamed_addr #1 {
entry:
  %url = icmp ne i32 %urlsafe, 0
  %plus = select i1 %url, i32 45, i32 43
  %slash = select i1 %url, i32 95, i32 47
  %z = icmp eq i64 %n, 0
  br i1 %z, label %zero, label %chk, !prof !1

zero:
  ret i64 0

chk:
  %rem = and i64 %n, 3
  %bad.len = icmp ne i64 %rem, 0
  br i1 %bad.len, label %err, label %setup, !prof !0

err:                                              ; cold
  ret i64 -1

setup:
  %ng = lshr i64 %n, 2
  ; padding from the final two bytes
  %off.last = sub nuw i64 %n, 1
  %lp2 = getelementptr inbounds nuw i8, ptr %src, i64 %off.last
  %last = load i8, ptr %lp2, align 1
  %off.second = sub nuw i64 %n, 2
  %sp2 = getelementptr inbounds nuw i8, ptr %src, i64 %off.second
  %second = load i8, ptr %sp2, align 1
  %pl = icmp eq i8 %last, 61
  %ps = icmp eq i8 %second, 61
  %not.pl = xor i1 %pl, true
  %malformed = and i1 %ps, %not.pl
  br i1 %malformed, label %err, label %prep, !prof !0

prep:
  %pl.i = zext i1 %pl to i64
  %ps.i = zext i1 %ps to i64
  %pad = add nuw nsw i64 %pl.i, %ps.i
  %base = mul nuw i64 %ng, 3
  %outlen = sub nuw i64 %base, %pad
  %mg = sub nuw i64 %ng, 1                        ; full groups before the last
  %has.full = icmp ne i64 %mg, 0
  br i1 %has.full, label %loop, label %last.grp

loop:
  %g = phi i64 [ 0, %prep ], [ %g.next, %loop ]
  %bad = phi i32 [ 0, %prep ], [ %bad.next, %loop ]
  %soff = shl nuw i64 %g, 2
  %sp0 = getelementptr inbounds nuw i8, ptr %src, i64 %soff
  %c0 = load i8, ptr %sp0, align 1
  %sp1 = getelementptr inbounds nuw i8, ptr %sp0, i64 1
  %c1 = load i8, ptr %sp1, align 1
  %sp2b = getelementptr inbounds nuw i8, ptr %sp0, i64 2
  %c2 = load i8, ptr %sp2b, align 1
  %sp3 = getelementptr inbounds nuw i8, ptr %sp0, i64 3
  %c3 = load i8, ptr %sp3, align 1
  %zc0 = zext i8 %c0 to i32
  %zc1 = zext i8 %c1 to i32
  %zc2 = zext i8 %c2 to i32
  %zc3 = zext i8 %c3 to i32
  %v0 = call i32 @b64_dec(i32 %zc0, i32 %plus, i32 %slash)
  %v1 = call i32 @b64_dec(i32 %zc1, i32 %plus, i32 %slash)
  %v2 = call i32 @b64_dec(i32 %zc2, i32 %plus, i32 %slash)
  %v3 = call i32 @b64_dec(i32 %zc3, i32 %plus, i32 %slash)
  ; OR of the four values; sign bit set iff any was -1
  %or01 = or i32 %v0, %v1
  %or23 = or i32 %v2, %v3
  %orall = or i32 %or01, %or23
  %anybad = lshr i32 %orall, 31                   ; 1 if any negative
  %bad.next = or i32 %bad, %anybad
  ; reconstruct 3 bytes
  %r0a = shl nsw i32 %v0, 2
  %r0b = lshr i32 %v1, 4
  %r0 = or disjoint i32 %r0a, %r0b
  %r1a = shl nsw i32 %v1, 4
  %r1b = lshr i32 %v2, 2
  %r1 = or i32 %r1a, %r1b
  %r2a = shl nsw i32 %v2, 6
  %r2 = or i32 %r2a, %v3
  %ooff = mul nuw i64 %g, 3
  %dp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %ooff
  %b0 = trunc i32 %r0 to i8
  store i8 %b0, ptr %dp0, align 1
  %dp1 = getelementptr inbounds nuw i8, ptr %dp0, i64 1
  %b1 = trunc i32 %r1 to i8
  store i8 %b1, ptr %dp1, align 1
  %dp2 = getelementptr inbounds nuw i8, ptr %dp0, i64 2
  %b2 = trunc i32 %r2 to i8
  store i8 %b2, ptr %dp2, align 1
  %g.next = add nuw i64 %g, 1
  %more = icmp ult i64 %g.next, %mg
  br i1 %more, label %loop, label %last.pre

last.pre:
  br label %last.grp

last.grp:
  %bad.in = phi i32 [ 0, %prep ], [ %bad.next, %last.pre ]
  %lsoff = shl nuw i64 %mg, 2
  %lsp0 = getelementptr inbounds nuw i8, ptr %src, i64 %lsoff
  %lc0 = load i8, ptr %lsp0, align 1
  %lsp1 = getelementptr inbounds nuw i8, ptr %lsp0, i64 1
  %lc1 = load i8, ptr %lsp1, align 1
  %lsp2 = getelementptr inbounds nuw i8, ptr %lsp0, i64 2
  %lc2 = load i8, ptr %lsp2, align 1
  %lsp3 = getelementptr inbounds nuw i8, ptr %lsp0, i64 3
  %lc3 = load i8, ptr %lsp3, align 1
  %lz0 = zext i8 %lc0 to i32
  %lz1 = zext i8 %lc1 to i32
  %lz2 = zext i8 %lc2 to i32
  %lz3 = zext i8 %lc3 to i32
  %lv0 = call i32 @b64_dec(i32 %lz0, i32 %plus, i32 %slash)
  %lv1 = call i32 @b64_dec(i32 %lz1, i32 %plus, i32 %slash)
  ; first two data chars are always required
  %lor01 = or i32 %lv0, %lv1
  %lbad01 = lshr i32 %lor01, 31
  %lo.off = mul nuw i64 %mg, 3
  %ldp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %lo.off
  ; byte 0 = (v0<<2)|(v1>>4) — valid whenever there are >=2 data chars
  %lr0a = shl nsw i32 %lv0, 2
  %lr0b = lshr i32 %lv1, 4
  %lr0 = or i32 %lr0a, %lr0b
  %lb0 = trunc i32 %lr0 to i8
  store i8 %lb0, ptr %ldp0, align 1
  switch i64 %pad, label %pad0.case [ i64 1, label %pad1.case
                                      i64 2, label %pad2.case ]

pad0.case:
  %p0.v2 = call i32 @b64_dec(i32 %lz2, i32 %plus, i32 %slash)
  %p0.v3 = call i32 @b64_dec(i32 %lz3, i32 %plus, i32 %slash)
  %p0.or = or i32 %p0.v2, %p0.v3
  %p0.bad = lshr i32 %p0.or, 31
  %p0.b01 = or i32 %lbad01, %p0.bad
  %p0.badf = or i32 %bad.in, %p0.b01
  %p0.r1a = shl nsw i32 %lv1, 4
  %p0.r1b = lshr i32 %p0.v2, 2
  %p0.r1 = or i32 %p0.r1a, %p0.r1b
  %p0.dp1 = getelementptr inbounds nuw i8, ptr %ldp0, i64 1
  %p0.b1 = trunc i32 %p0.r1 to i8
  store i8 %p0.b1, ptr %p0.dp1, align 1
  %p0.r2a = shl nsw i32 %p0.v2, 6
  %p0.r2 = or i32 %p0.r2a, %p0.v3
  %p0.dp2 = getelementptr inbounds nuw i8, ptr %ldp0, i64 2
  %p0.b2 = trunc i32 %p0.r2 to i8
  store i8 %p0.b2, ptr %p0.dp2, align 1
  br label %finish

pad1.case:
  %p1.v2 = call i32 @b64_dec(i32 %lz2, i32 %plus, i32 %slash)
  %p1.bad = lshr i32 %p1.v2, 31
  %p1.b01 = or i32 %lbad01, %p1.bad
  %p1.badf = or i32 %bad.in, %p1.b01
  %p1.r1a = shl nsw i32 %lv1, 4
  %p1.r1b = lshr i32 %p1.v2, 2
  %p1.r1 = or i32 %p1.r1a, %p1.r1b
  %p1.dp1 = getelementptr inbounds nuw i8, ptr %ldp0, i64 1
  %p1.b1 = trunc i32 %p1.r1 to i8
  store i8 %p1.b1, ptr %p1.dp1, align 1
  br label %finish

pad2.case:
  %p2.badf = or i32 %bad.in, %lbad01
  br label %finish

finish:
  %badf = phi i32 [ %p0.badf, %pad0.case ], [ %p1.badf, %pad1.case ], [ %p2.badf, %pad2.case ]
  %is.bad = icmp ne i32 %badf, 0
  br i1 %is.bad, label %err.parse, label %ok, !prof !0

err.parse:                                        ; cold
  ret i64 -1

ok:
  ret i64 %outlen
}

; =========================================================================
; VECTOR HELPERS (128-bit, pure)
; =========================================================================

declare i1 @llvm.vector.reduce.or.v16i1(<16 x i1>)

; unsigned high 16 bits of a 16-bit lane multiply (SSE _mm_mulhi_epu16 shape)
define internal <8 x i16> @b64_mulhi_u16(<8 x i16> %a, <8 x i16> %b) #3 {
entry:
  %az = zext <8 x i16> %a to <8 x i32>
  %bz = zext <8 x i16> %b to <8 x i32>
  %p = mul <8 x i32> %az, %bz
  %hi = lshr <8 x i32> %p, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %r = trunc <8 x i32> %hi to <8 x i16>
  ret <8 x i16> %r
}

; sixteen 6-bit indices (0..63) -> ascii, alphabet-parameterized. Lane-parallel
; twin of @b64_enc.
define internal <16 x i8> @b64_enc_vec(<16 x i8> %i, <16 x i8> %plus, <16 x i8> %slash) #3 {
entry:
  %c0 = add <16 x i8> %i, <i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65>
  %g25 = icmp ugt <16 x i8> %i, <i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25>
  %av = add <16 x i8> %i, <i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71>
  %c1 = select <16 x i1> %g25, <16 x i8> %av, <16 x i8> %c0
  %g51 = icmp ugt <16 x i8> %i, <i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51, i8 51>
  %dv = add <16 x i8> %i, <i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4, i8 -4>
  %c2 = select <16 x i1> %g51, <16 x i8> %dv, <16 x i8> %c1
  %e62 = icmp eq <16 x i8> %i, <i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62>
  %c3 = select <16 x i1> %e62, <16 x i8> %plus, <16 x i8> %c2
  %e63 = icmp eq <16 x i8> %i, <i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63>
  %c4 = select <16 x i1> %e63, <16 x i8> %slash, <16 x i8> %c3
  ret <16 x i8> %c4
}

; sixteen ascii bytes -> 6-bit values (0..63), 0xff for any char outside the
; selected alphabet. Lane-parallel twin of @b64_dec.
define internal <16 x i8> @b64_dec_vec(<16 x i8> %c, i8 %plus, i8 %slash) #3 {
entry:
  %pv = insertelement <16 x i8> poison, i8 %plus, i64 0
  %plusS = shufflevector <16 x i8> %pv, <16 x i8> poison, <16 x i32> zeroinitializer
  %sv = insertelement <16 x i8> poison, i8 %slash, i64 0
  %slashS = shufflevector <16 x i8> %sv, <16 x i8> poison, <16 x i32> zeroinitializer
  %uge = icmp uge <16 x i8> %c, <i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65>
  %ule = icmp ule <16 x i8> %c, <i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90, i8 90>
  %uok = and <16 x i1> %uge, %ule
  %uv = sub <16 x i8> %c, <i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65, i8 65>
  %lge = icmp uge <16 x i8> %c, <i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97>
  %lle = icmp ule <16 x i8> %c, <i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122, i8 122>
  %lok = and <16 x i1> %lge, %lle
  %lv = sub <16 x i8> %c, <i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71, i8 71>
  %dge = icmp uge <16 x i8> %c, <i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48, i8 48>
  %dle = icmp ule <16 x i8> %c, <i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57, i8 57>
  %dok = and <16 x i1> %dge, %dle
  %dv = add <16 x i8> %c, <i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4, i8 4>
  %pok = icmp eq <16 x i8> %c, %plusS
  %sok = icmp eq <16 x i8> %c, %slashS
  %r0 = select <16 x i1> %uok, <16 x i8> %uv, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %r1 = select <16 x i1> %lok, <16 x i8> %lv, <16 x i8> %r0
  %r2 = select <16 x i1> %dok, <16 x i8> %dv, <16 x i8> %r1
  %r3 = select <16 x i1> %pok, <16 x i8> <i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62, i8 62>, <16 x i8> %r2
  %r4 = select <16 x i1> %sok, <16 x i8> <i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63, i8 63>, <16 x i8> %r3
  ret <16 x i8> %r4
}

; ---------------------------------------------------------------- encode (vector)
; 12 src bytes -> 16 chars per iteration; the final <16 bytes + padding go to
; the scalar twin (which is also the cross-check oracle).
define i64 @universe_base64_encode(ptr %dst, ptr %src, i64 %n, i32 %urlsafe) local_unnamed_addr #0 {
entry:
  %url = icmp ne i32 %urlsafe, 0
  %plusb = select i1 %url, i8 45, i8 43
  %slashb = select i1 %url, i8 95, i8 47
  %pv = insertelement <16 x i8> poison, i8 %plusb, i64 0
  %plusS = shufflevector <16 x i8> %pv, <16 x i8> poison, <16 x i32> zeroinitializer
  %sv = insertelement <16 x i8> poison, i8 %slashb, i64 0
  %slashS = shufflevector <16 x i8> %sv, <16 x i8> poison, <16 x i32> zeroinitializer
  %can = icmp uge i64 %n, 16
  br i1 %can, label %vloop, label %scalar.all

vloop:
  %inoff = phi i64 [ 0, %entry ], [ %inoff.n, %vloop ]
  %outoff = phi i64 [ 0, %entry ], [ %outoff.n, %vloop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %inoff
  %in = load <16 x i8>, ptr %sp, align 1
  %sh = shufflevector <16 x i8> %in, <16 x i8> poison, <16 x i32> <i32 1, i32 0, i32 2, i32 1, i32 4, i32 3, i32 5, i32 4, i32 7, i32 6, i32 8, i32 7, i32 10, i32 9, i32 11, i32 10>
  %sh32 = bitcast <16 x i8> %sh to <4 x i32>
  %t0 = and <4 x i32> %sh32, <i32 264305664, i32 264305664, i32 264305664, i32 264305664>
  %t0_16 = bitcast <4 x i32> %t0 to <8 x i16>
  %c1_16 = bitcast <4 x i32> <i32 67108928, i32 67108928, i32 67108928, i32 67108928> to <8 x i16>
  %t1_16 = call <8 x i16> @b64_mulhi_u16(<8 x i16> %t0_16, <8 x i16> %c1_16)
  %t1 = bitcast <8 x i16> %t1_16 to <4 x i32>
  %t2 = and <4 x i32> %sh32, <i32 4129776, i32 4129776, i32 4129776, i32 4129776>
  %t2_16 = bitcast <4 x i32> %t2 to <8 x i16>
  %c3_16 = bitcast <4 x i32> <i32 16777232, i32 16777232, i32 16777232, i32 16777232> to <8 x i16>
  %t3_16 = mul <8 x i16> %t2_16, %c3_16
  %t3 = bitcast <8 x i16> %t3_16 to <4 x i32>
  %idx32 = or <4 x i32> %t1, %t3
  %idx = bitcast <4 x i32> %idx32 to <16 x i8>
  %ascii = call <16 x i8> @b64_enc_vec(<16 x i8> %idx, <16 x i8> %plusS, <16 x i8> %slashS)
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %outoff
  store <16 x i8> %ascii, ptr %dp, align 1
  %inoff.n = add nuw i64 %inoff, 12
  %outoff.n = add nuw i64 %outoff, 16
  %lim = add nuw i64 %inoff.n, 16
  %more = icmp ule i64 %lim, %n
  br i1 %more, label %vloop, label %vdone

vdone:
  %rem = sub nuw i64 %n, %inoff.n
  %tsp = getelementptr inbounds nuw i8, ptr %src, i64 %inoff.n
  %tdp = getelementptr inbounds nuw i8, ptr %dst, i64 %outoff.n
  %tw = call i64 @universe_base64_encode_scalar(ptr %tdp, ptr %tsp, i64 %rem, i32 %urlsafe)
  %tot = add nuw i64 %outoff.n, %tw
  ret i64 %tot

scalar.all:
  %all = call i64 @universe_base64_encode_scalar(ptr %dst, ptr %src, i64 %n, i32 %urlsafe)
  ret i64 %all
}

; ---------------------------------------------------------------- decode (vector)
; 16 interior chars -> 12 bytes per iteration; the final group (which may carry
; '=' padding) and any remainder go to the scalar twin. Invalid interior chars
; are OR-reduced into a bad flag and surface as -1, matching the scalar path.
define i64 @universe_base64_decode(ptr %dst, ptr %src, i64 %n, i32 %urlsafe) local_unnamed_addr #1 {
entry:
  %url = icmp ne i32 %urlsafe, 0
  %plusb = select i1 %url, i8 45, i8 43
  %slashb = select i1 %url, i8 95, i8 47
  %rem4 = and i64 %n, 3
  %lenok = icmp eq i64 %rem4, 0
  %big = icmp uge i64 %n, 20
  %can = and i1 %lenok, %big
  br i1 %can, label %vloop, label %scalar.all

vloop:
  %inoff = phi i64 [ 0, %entry ], [ %inoff.n, %vloop ]
  %outoff = phi i64 [ 0, %entry ], [ %outoff.n, %vloop ]
  %bad = phi i32 [ 0, %entry ], [ %bad.n, %vloop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %inoff
  %cin = load <16 x i8>, ptr %sp, align 1
  %vals = call <16 x i8> @b64_dec_vec(<16 x i8> %cin, i8 %plusb, i8 %slashb)
  %isbad = icmp eq <16 x i8> %vals, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %anybad1 = call i1 @llvm.vector.reduce.or.v16i1(<16 x i1> %isbad)
  %anybad = zext i1 %anybad1 to i32
  %bad.n = or i32 %bad, %anybad
  %L = bitcast <16 x i8> %vals to <4 x i32>
  %v0 = and <4 x i32> %L, <i32 63, i32 63, i32 63, i32 63>
  %s8 = lshr <4 x i32> %L, <i32 8, i32 8, i32 8, i32 8>
  %v1 = and <4 x i32> %s8, <i32 63, i32 63, i32 63, i32 63>
  %s16 = lshr <4 x i32> %L, <i32 16, i32 16, i32 16, i32 16>
  %v2 = and <4 x i32> %s16, <i32 63, i32 63, i32 63, i32 63>
  %s24 = lshr <4 x i32> %L, <i32 24, i32 24, i32 24, i32 24>
  %v3 = and <4 x i32> %s24, <i32 63, i32 63, i32 63, i32 63>
  %w0 = shl <4 x i32> %v0, <i32 18, i32 18, i32 18, i32 18>
  %w1 = shl <4 x i32> %v1, <i32 12, i32 12, i32 12, i32 12>
  %w2 = shl <4 x i32> %v2, <i32 6, i32 6, i32 6, i32 6>
  %w01 = or <4 x i32> %w0, %w1
  %w23 = or <4 x i32> %w2, %v3
  %W = or <4 x i32> %w01, %w23
  %Wb = bitcast <4 x i32> %W to <16 x i8>
  %out = shufflevector <16 x i8> %Wb, <16 x i8> poison, <16 x i32> <i32 2, i32 1, i32 0, i32 6, i32 5, i32 4, i32 10, i32 9, i32 8, i32 14, i32 13, i32 12, i32 0, i32 0, i32 0, i32 0>
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %outoff
  %out64 = bitcast <16 x i8> %out to <2 x i64>
  %lo = extractelement <2 x i64> %out64, i64 0
  store i64 %lo, ptr %dp, align 1
  %dp8 = getelementptr inbounds nuw i8, ptr %dp, i64 8
  %out32 = bitcast <16 x i8> %out to <4 x i32>
  %hi = extractelement <4 x i32> %out32, i64 2
  store i32 %hi, ptr %dp8, align 1
  %inoff.n = add nuw i64 %inoff, 16
  %outoff.n = add nuw i64 %outoff, 12
  %lim = add nuw i64 %inoff.n, 20
  %more = icmp ule i64 %lim, %n
  br i1 %more, label %vloop, label %vdone

vdone:
  %tail = sub nuw i64 %n, %inoff.n
  %tsp = getelementptr inbounds nuw i8, ptr %src, i64 %inoff.n
  %tdp = getelementptr inbounds nuw i8, ptr %dst, i64 %outoff.n
  %tw = call i64 @universe_base64_decode_scalar(ptr %tdp, ptr %tsp, i64 %tail, i32 %urlsafe)
  %tbad = icmp slt i64 %tw, 0
  %vbad = icmp ne i32 %bad.n, 0
  %fail = or i1 %tbad, %vbad
  br i1 %fail, label %err.v, label %ok.v, !prof !0

ok.v:
  %tot = add nuw i64 %outoff.n, %tw
  ret i64 %tot

err.v:
  ret i64 -1

scalar.all:
  %all = call i64 @universe_base64_decode_scalar(ptr %dst, ptr %src, i64 %n, i32 %urlsafe)
  ret i64 %all
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #3 = { alwaysinline nounwind willreturn nosync nofree norecurse memory(none) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 20}
