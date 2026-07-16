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

; Hex (base16) encode/decode. Pure compute over caller-supplied byte ranges:
; the caller pre-sizes `dst` (via encode_len / n/2) and we never allocate.
;
; DESIGN:
;   * Compute/memory separation: each iteration is pure register arithmetic on
;     a loaded byte, then a store — no data-dependent branch on the hot path,
;     so the backend can unroll/vectorize the loop freely.
;   * Branchless nibble->ascii (NO lookup table, NO branch): for a nibble v in
;     0..15,  ascii = v + '0' + ((9 - v) >>a 31) & 0x27 . (9-v) is negative
;     exactly when v>9, so the arithmetic-shift-right by 31 yields a 0/-1 mask
;     that adds 0x27 (='a'-'0'-10) only for the letter nibbles. Two nibbles per
;     byte, both computed the same way — pure ALU, vectorizes.
;   * Branchless ascii->nibble on decode: digit_ok and alpha_ok are computed
;     with range compares; the value is a case-folded (`| 0x20`) subtraction
;     selected between the two. Invalidity is OR-accumulated across the whole
;     buffer into one flag and checked ONCE at the end (no per-char branch to a
;     cold exit inside the loop).
;   * ERROR CONVENTION (documented): the i64-returning entry points return the
;     produced length on success and a NEGATIVE value on failure — decode
;     returns -1 (PARSE) for an odd input length or any non-hex character.
;     encode_len returns -1 (SIZE_OVERFLOW) if 2n overflows i64.
;
; API:
;   i64 universe_hex_encode_len(i64 n)              ; = 2n, or -1 on overflow
;   i64 universe_hex_encode(ptr dst, ptr src, i64 n); lowercase; returns 2n
;   i64 universe_hex_decode(ptr dst, ptr src, i64 n); returns n/2, or -1 PARSE

declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ------------------------------------------------------------------ encode_len
define i64 @universe_hex_encode_len(i64 %n) local_unnamed_addr #2 {
entry:
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 2)
  %v = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %r = select i1 %o, i64 -1, i64 %v
  ret i64 %r
}

; ---------------------------------------------------------------------- encode
define i64 @universe_hex_encode(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b = load i8, ptr %sp, align 1
  %bz = zext i8 %b to i32
  ; high nibble
  %hi = lshr i32 %bz, 4
  %hi.c0 = add nuw nsw i32 %hi, 48
  %hi.t = sub nsw i32 9, %hi
  %hi.m = ashr i32 %hi.t, 31
  %hi.a = and i32 %hi.m, 39
  %hi.c = add nsw i32 %hi.c0, %hi.a
  %hi.b = trunc i32 %hi.c to i8
  ; low nibble
  %lo = and i32 %bz, 15
  %lo.c0 = add nuw nsw i32 %lo, 48
  %lo.t = sub nsw i32 9, %lo
  %lo.m = ashr i32 %lo.t, 31
  %lo.a = and i32 %lo.m, 39
  %lo.c = add nsw i32 %lo.c0, %lo.a
  %lo.b = trunc i32 %lo.c to i8
  ; store both output bytes at 2*i, 2*i+1
  %oo = shl nuw i64 %i, 1
  %dp0 = getelementptr inbounds nuw i8, ptr %dst, i64 %oo
  store i8 %hi.b, ptr %dp0, align 1
  %oo1 = or disjoint i64 %oo, 1
  %dp1 = getelementptr inbounds nuw i8, ptr %dst, i64 %oo1
  store i8 %lo.b, ptr %dp1, align 1
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %done

done:
  %out = shl i64 %n, 1
  ret i64 %out
}

; ---------------------------------------------------------------------- decode
; SIMD-first with scalar fallback: the vector body validates+packs 16 hex chars
; (= 8 output bytes) per step with portable <16 x i8> ops (lowers to SSE2 / NEON,
; baseline on every target — no runtime check). Per lane the nibble value is a
; branchless digit-or-alpha map (sub+ult range tests, house style), validity is
; OR-accumulated via a movemask, and the two nibbles of each byte are
; deinterleaved with a shufflevector (NEON uzp1/uzp2) then packed hi<<4|lo and
; stored as <8 x i8>. The scalar loop below is the fallback for the sub-16 tail
; AND the reference the vector path is cross-checked against in tests.
define i64 @universe_hex_decode(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %odd = and i64 %n, 1
  %is.odd = icmp ne i64 %odd, 0
  br i1 %is.odd, label %err, label %chk.zero, !prof !0

err:                                              ; cold
  ret i64 -1

chk.zero:
  %half = lshr i64 %n, 1
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %vec.head

; ---- vector fast path: 16 source chars -> 8 output bytes per step ----
vec.head:
  %vsi = phi i64 [ 0, %chk.zero ], [ %vsi.next, %vec.body ]   ; source index
  %voi = phi i64 [ 0, %chk.zero ], [ %voi.next, %vec.body ]   ; output index
  %vbad = phi i32 [ 0, %chk.zero ], [ %vbad.next, %vec.body ]
  %vroom = add nuw i64 %vsi, 16
  %vfits = icmp ule i64 %vroom, %n
  br i1 %vfits, label %vec.body, label %scalar.pre

vec.body:
  %vsp = getelementptr inbounds nuw i8, ptr %src, i64 %vsi
  %vv = load <16 x i8>, ptr %vsp, align 1
  ; digit path: dsub = c - '0'; valid digit when dsub < 10; nibble = dsub
  %vdsub = sub <16 x i8> %vv, splat (i8 48)
  %vdok = icmp ult <16 x i8> %vdsub, splat (i8 10)
  ; alpha path (case-folded): lc = c | 0x20; asub = lc - 'a'; valid when asub < 6
  %vlc = or <16 x i8> %vv, splat (i8 32)
  %vasub = sub <16 x i8> %vlc, splat (i8 97)
  %vaok = icmp ult <16 x i8> %vasub, splat (i8 6)
  %vaval = add <16 x i8> %vasub, splat (i8 10)
  ; nibble value = digit ? dsub : (alpha ? aval : 0)
  %vsela = select <16 x i1> %vaok, <16 x i8> %vaval, <16 x i8> zeroinitializer
  %vnib = select <16 x i1> %vdok, <16 x i8> %vdsub, <16 x i8> %vsela
  ; validity: ok lane = digit | alpha; accumulate the bad lanes via movemask
  %vok = or <16 x i1> %vdok, %vaok
  %vokm = bitcast <16 x i1> %vok to i16
  %vbadm = xor i16 %vokm, -1                       ; bits set where lane is bad
  %vbadz = zext i16 %vbadm to i32
  %vbad.next = or i32 %vbad, %vbadz
  ; pack: even lanes are hi nibbles, odd lanes are lo nibbles
  %veven = shufflevector <16 x i8> %vnib, <16 x i8> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %vodd = shufflevector <16 x i8> %vnib, <16 x i8> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %vhi = shl <8 x i8> %veven, splat (i8 4)
  %vout = or <8 x i8> %vhi, %vodd
  %vop = getelementptr inbounds nuw i8, ptr %dst, i64 %voi
  store <8 x i8> %vout, ptr %vop, align 1
  %vsi.next = add nuw i64 %vsi, 16
  %voi.next = add nuw i64 %voi, 8
  br label %vec.head

scalar.pre:
  %more.tail = icmp ult i64 %voi, %half
  br i1 %more.tail, label %loop, label %check

loop:
  %i = phi i64 [ %voi, %scalar.pre ], [ %i.next, %loop ]
  %bad = phi i32 [ %vbad, %scalar.pre ], [ %bad.next, %loop ]
  %soff = shl nuw i64 %i, 1
  %sp0 = getelementptr inbounds nuw i8, ptr %src, i64 %soff
  %c0 = load i8, ptr %sp0, align 1
  %soff1 = or disjoint i64 %soff, 1
  %sp1 = getelementptr inbounds nuw i8, ptr %src, i64 %soff1
  %c1 = load i8, ptr %sp1, align 1
  ; decode high char (c0)
  %h.c = zext i8 %c0 to i32
  %h.dge = icmp uge i32 %h.c, 48
  %h.dle = icmp ule i32 %h.c, 57
  %h.dok = and i1 %h.dge, %h.dle
  %h.dv = sub nsw i32 %h.c, 48
  %h.lc = or i32 %h.c, 32
  %h.age = icmp uge i32 %h.lc, 97
  %h.ale = icmp ule i32 %h.lc, 102
  %h.aok = and i1 %h.age, %h.ale
  %h.av = sub nsw i32 %h.lc, 87
  %h.sel = select i1 %h.aok, i32 %h.av, i32 0
  %h.val = select i1 %h.dok, i32 %h.dv, i32 %h.sel
  %h.ok = or i1 %h.dok, %h.aok
  ; decode low char (c1)
  %l.c = zext i8 %c1 to i32
  %l.dge = icmp uge i32 %l.c, 48
  %l.dle = icmp ule i32 %l.c, 57
  %l.dok = and i1 %l.dge, %l.dle
  %l.dv = sub nsw i32 %l.c, 48
  %l.lc = or i32 %l.c, 32
  %l.age = icmp uge i32 %l.lc, 97
  %l.ale = icmp ule i32 %l.lc, 102
  %l.aok = and i1 %l.age, %l.ale
  %l.av = sub nsw i32 %l.lc, 87
  %l.sel = select i1 %l.aok, i32 %l.av, i32 0
  %l.val = select i1 %l.dok, i32 %l.dv, i32 %l.sel
  %l.ok = or i1 %l.dok, %l.aok
  ; assemble output byte
  %hi.sh = shl nsw i32 %h.val, 4
  %ov = or disjoint i32 %hi.sh, %l.val
  %ob = trunc i32 %ov to i8
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %ob, ptr %dp, align 1
  ; accumulate invalidity (branchless)
  %ok.both = and i1 %h.ok, %l.ok
  %bad.bit = zext i1 %ok.both to i32           ; 1 when OK, 0 when bad
  %bad.inv = xor i32 %bad.bit, 1               ; 1 when bad
  %bad.next = or i32 %bad, %bad.inv
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %half
  br i1 %more, label %loop, label %check

check:
  %bad.final = phi i32 [ %vbad, %scalar.pre ], [ %bad.next, %loop ]
  %is.bad = icmp ne i32 %bad.final, 0
  br i1 %is.bad, label %err.parse, label %done, !prof !0

err.parse:                                        ; cold
  ret i64 -1

done:
  ret i64 %half
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
