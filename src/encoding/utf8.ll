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

; UTF-8 validation and codepoint counting over a caller-supplied byte range.
; Pure compute over memory; no allocation, no IO.
;
; DESIGN:
;   * FULL RFC 3629 well-formedness, not just a continuation-bit check:
;     rejects overlong forms (C0/C1, E0 80..9F, F0 80..8F), surrogates
;     (ED A0..BF => U+D800..DFFF), and out-of-range (F5.. and F4 90..).
;     This is enforced with the "lead byte selects the SECOND byte's legal
;     range" technique: only the 2nd continuation byte has a lead-dependent
;     bound (lo2/hi2); the 3rd and 4th are always 0x80..0xBF. So a handful of
;     `select`s compute {len, lo2, hi2} from the lead, then the range checks
;     are uniform.
;   * HOT LOOP branches minimally: an ASCII fast path (byte < 0x80 advances by
;     1) falls through as a straight line; multi-byte classification is
;     branchless (selects), and the loop advances by the decoded codepoint
;     length so each sequence is visited once. The only in-loop branches are
;     the ASCII test and the (cold, !prof) invalid exits.
;   * count_codepoints reuses validate for correctness, then counts leads in a
;     second, branch-light pass: a codepoint boundary is any byte that is NOT a
;     continuation byte, i.e. (b & 0xC0) != 0x80. That predicate vectorizes.
;   * byte_len_of_codepoint is a pure classifier: 1/2/3/4 for a valid lead,
;     0 for a continuation byte or an invalid lead (C0,C1,F5..FF).
;
; ERROR CONVENTION (documented): validate returns the i64 index of the first
; invalid byte, or -1 when the whole range is well-formed. For a truncated
; multibyte sequence at the end of the buffer, the reported index is the LEAD
; byte that begins the incomplete sequence. count_codepoints returns the count,
; or -1 if the range is not well-formed.
;
; API:
;   i64 universe_utf8_validate(ptr src, i64 n)          ; first bad idx, or -1
;   i64 universe_utf8_count_codepoints(ptr src, i64 n)  ; count, or -1
;   i32 universe_utf8_byte_len_of_codepoint(i8 lead)    ; 1..4, or 0

; ---------------------------------------------------------------- byte_len
define i32 @universe_utf8_byte_len_of_codepoint(i8 %lead) local_unnamed_addr #2 {
entry:
  %u = zext i8 %lead to i32
  %ascii = icmp ult i32 %u, 128
  %is2a = icmp uge i32 %u, 194                    ; 0xC2
  %is2b = icmp ule i32 %u, 223                    ; 0xDF
  %is2 = and i1 %is2a, %is2b
  %is3a = icmp uge i32 %u, 224                    ; 0xE0
  %is3b = icmp ule i32 %u, 239                    ; 0xEF
  %is3 = and i1 %is3a, %is3b
  %is4a = icmp uge i32 %u, 240                    ; 0xF0
  %is4b = icmp ule i32 %u, 244                    ; 0xF4
  %is4 = and i1 %is4a, %is4b
  %r4 = select i1 %is4, i32 4, i32 0
  %r3 = select i1 %is3, i32 3, i32 %r4
  %r2 = select i1 %is2, i32 2, i32 %r3
  %r1 = select i1 %ascii, i32 1, i32 %r2
  ret i32 %r1
}

; ---------------------------------------------------------------- validate
; SIMD-first with scalar fallback: a portable <16 x i8> ASCII fast-path skips
; runs of ASCII 16 bytes at a time (high-bit test via signed `icmp slt` +
; movemask; lowers to SSE2 pmovmskb / NEON cmlt+reduce — baseline, no runtime
; check). The instant any lane is >= 0x80, or fewer than 16 bytes remain, we
; fall into the scalar RFC-3629 state machine (the fallback + oracle) for that
; multibyte sequence, then return to the vector probe. Pure-ASCII input never
; touches the scalar path; mixed input vectorizes every ASCII run between
; multibyte sequences.
define i64 @universe_utf8_validate(ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  br label %head.vec

; Vector entry: reached only from entry / an ASCII byte / a vector skip — where
; an ASCII run is plausible — so the probe never burdens the multibyte path.
head.vec:
  %pv = phi i64 [ 0, %entry ], [ %pos.a1, %ascii ], [ %pos.v, %vec.adv ]
  %done.v = icmp uge i64 %pv, %n
  br i1 %done.v, label %ok, label %vscan

ok:
  ret i64 -1

vscan:
  %vroom = add nuw i64 %pv, 16
  %vfits = icmp ule i64 %vroom, %n
  br i1 %vfits, label %vload, label %body

vload:
  %vp = getelementptr inbounds nuw i8, ptr %src, i64 %pv
  %vv = load <16 x i8>, ptr %vp, align 1
  %vneg = icmp slt <16 x i8> %vv, zeroinitializer   ; high bit set -> >= 0x80
  %vmm = bitcast <16 x i1> %vneg to i16
  %vany = icmp ne i16 %vmm, 0
  br i1 %vany, label %body, label %vec.adv, !prof !0

vec.adv:
  %pos.v = add nuw i64 %pv, 16
  br label %head.vec

; Scalar entry: reached only after a completed multibyte sequence, so it drops
; straight into the state machine with no vector probe (no per-codepoint cost).
head.sc:
  %ps = phi i64 [ %end, %cont.more ], [ %end, %cont2.more ], [ %end, %cont3 ]
  %done.s = icmp uge i64 %ps, %n
  br i1 %done.s, label %ok, label %body

body:
  %pos = phi i64 [ %pv, %vscan ], [ %pv, %vload ], [ %ps, %head.sc ]
  %p0 = getelementptr inbounds nuw i8, ptr %src, i64 %pos
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i32
  %is.ascii = icmp ult i32 %z0, 128
  br i1 %is.ascii, label %ascii, label %multi, !prof !1

ascii:
  %pos.a1 = add nuw i64 %pos, 1
  br label %head.vec

multi:
  ; classify lead -> len, lo2, hi2
  %c2 = icmp uge i32 %z0, 194
  %d2 = icmp ule i32 %z0, 223
  %ok2 = and i1 %c2, %d2
  %c3 = icmp uge i32 %z0, 224
  %d3 = icmp ule i32 %z0, 239
  %ok3 = and i1 %c3, %d3
  %c4 = icmp uge i32 %z0, 240
  %d4 = icmp ule i32 %z0, 244
  %ok4 = and i1 %c4, %d4
  %len4 = select i1 %ok4, i64 4, i64 0
  %len3 = select i1 %ok3, i64 3, i64 %len4
  %len = select i1 %ok2, i64 2, i64 %len3
  %bad.lead = icmp eq i64 %len, 0
  br i1 %bad.lead, label %invalid.here, label %len.ok, !prof !0

len.ok:
  ; second-byte legal range depends on the lead
  %is.e0 = icmp eq i32 %z0, 224                    ; E0 -> lo2 = 0xA0
  %is.f0 = icmp eq i32 %z0, 240                    ; F0 -> lo2 = 0x90
  %is.ed = icmp eq i32 %z0, 237                    ; ED -> hi2 = 0x9F
  %is.f4 = icmp eq i32 %z0, 244                    ; F4 -> hi2 = 0x8F
  %lo.e0 = select i1 %is.e0, i32 160, i32 128
  %lo2 = select i1 %is.f0, i32 144, i32 %lo.e0
  %hi.ed = select i1 %is.ed, i32 159, i32 191
  %hi2 = select i1 %is.f4, i32 143, i32 %hi.ed
  ; bounds: need pos + len <= n
  %end = add nuw i64 %pos, %len
  %trunc = icmp ugt i64 %end, %n
  br i1 %trunc, label %invalid.here, label %cont1, !prof !0

cont1:
  %p1i = add nuw i64 %pos, 1
  %p1 = getelementptr inbounds nuw i8, ptr %src, i64 %p1i
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i32
  %b1.ge = icmp uge i32 %z1, %lo2
  %b1.le = icmp ule i32 %z1, %hi2
  %b1.ok = and i1 %b1.ge, %b1.le
  br i1 %b1.ok, label %cont.more, label %invalid.p1, !prof !0

cont.more:
  %need3 = icmp uge i64 %len, 3
  br i1 %need3, label %cont2, label %head.sc

cont2:
  %p2i = add nuw i64 %pos, 2
  %p2 = getelementptr inbounds nuw i8, ptr %src, i64 %p2i
  %b2 = load i8, ptr %p2, align 1
  %z2 = zext i8 %b2 to i32
  %b2.ge = icmp uge i32 %z2, 128
  %b2.le = icmp ule i32 %z2, 191
  %b2.ok = and i1 %b2.ge, %b2.le
  br i1 %b2.ok, label %cont2.more, label %invalid.p2, !prof !0

cont2.more:
  %need4 = icmp uge i64 %len, 4
  br i1 %need4, label %cont3, label %head.sc

cont3:
  %p3i = add nuw i64 %pos, 3
  %p3 = getelementptr inbounds nuw i8, ptr %src, i64 %p3i
  %b3 = load i8, ptr %p3, align 1
  %z3 = zext i8 %b3 to i32
  %b3.ge = icmp uge i32 %z3, 128
  %b3.le = icmp ule i32 %z3, 191
  %b3.ok = and i1 %b3.ge, %b3.le
  br i1 %b3.ok, label %head.sc, label %invalid.p3, !prof !0

invalid.here:                                     ; cold — lead invalid / truncated
  ret i64 %pos

invalid.p1:                                       ; cold
  %ip1 = add nuw i64 %pos, 1
  ret i64 %ip1

invalid.p2:                                       ; cold
  %ip2 = add nuw i64 %pos, 2
  ret i64 %ip2

invalid.p3:                                       ; cold
  %ip3 = add nuw i64 %pos, 3
  ret i64 %ip3
}

; --------------------------------------------------------- count_codepoints
define i64 @universe_utf8_count_codepoints(ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %vr = call i64 @universe_utf8_validate(ptr %src, i64 %n)
  %valid = icmp eq i64 %vr, -1
  br i1 %valid, label %count.pre, label %bad, !prof !2

bad:                                              ; cold
  ret i64 -1

count.pre:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %zero, label %vc.head

zero:
  ret i64 0

; SIMD-first: a codepoint boundary is any byte with (b & 0xC0) != 0x80. The
; vector body counts lead bytes in a 16-lane tile via a per-lane compare and a
; horizontal reduce.add (SSE2 / NEON addv); the scalar loop is the <16 tail and
; the fallback + oracle.
vc.head:
  %vi = phi i64 [ 0, %count.pre ], [ %vi.next, %vc.body ]
  %vacc = phi i64 [ 0, %count.pre ], [ %vacc.next, %vc.body ]
  %vroom = add nuw i64 %vi, 16
  %vfits = icmp ule i64 %vroom, %n
  br i1 %vfits, label %vc.body, label %sc.pre

vc.body:
  %vcp = getelementptr inbounds nuw i8, ptr %src, i64 %vi
  %vcv = load <16 x i8>, ptr %vcp, align 1
  %vmask = and <16 x i8> %vcv, splat (i8 -64)      ; b & 0xC0
  %viscont = icmp eq <16 x i8> %vmask, splat (i8 -128) ; == 0x80
  %vlead = select <16 x i1> %viscont, <16 x i8> zeroinitializer, <16 x i8> splat (i8 1)
  %vcnt8 = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %vlead)
  %vcntz = zext i8 %vcnt8 to i64
  %vacc.next = add nuw i64 %vacc, %vcntz
  %vi.next = add nuw i64 %vi, 16
  br label %vc.head

sc.pre:
  %rem = icmp ult i64 %vi, %n
  br i1 %rem, label %loop, label %fin

loop:
  %i = phi i64 [ %vi, %sc.pre ], [ %i.next, %loop ]
  %acc = phi i64 [ %vacc, %sc.pre ], [ %acc.next, %loop ]
  %cp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %cb = load i8, ptr %cp, align 1
  %ci = zext i8 %cb to i32
  %masked = and i32 %ci, 192                       ; b & 0xC0
  %is.cont = icmp eq i32 %masked, 128              ; == 0x80 -> continuation
  %is.lead = xor i1 %is.cont, true
  %inc = zext i1 %is.lead to i64
  %acc.next = add nuw i64 %acc, %inc
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %fin

fin:
  %rr = phi i64 [ %vacc, %sc.pre ], [ %acc.next, %loop ]
  ret i64 %rr
}

declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)

attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 2000, i32 1}
