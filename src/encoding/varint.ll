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

; Variable-length integer codec (LEB128 family). Pure compute over caller
; buffers; never allocates. Three encodings on a 64-bit word:
;   * ULEB128 — unsigned, little-endian base-128, 7 data bits/byte, high bit is
;     the continuation flag. 1..10 bytes for a u64.
;   * SLEB128 — signed, two's-complement, sign-extended on the terminal group.
;     1..10 bytes for an i64.
;   * zigzag  — maps signed to unsigned so small-magnitude negatives stay short
;     ( 0,-1,1,-2,2 -> 0,1,2,3,4 ); pair with ULEB128 for a compact signed wire
;     form that is branch-free to fold.
;
; DESIGN:
;   * Compute/memory separation: the value math is register-resident; each loop
;     iteration reads/derives a byte then does ONE store. The encode loops carry
;     a genuine data dependency (val >>= 7 each step), so they are inherently
;     sequential — we keep the body minimal instead of pretending it vectorizes.
;   * uleb_len is branch-free via ctlz: significant bits = 64-clz(val); byte
;     count = ceil(bits/7) computed as (bits+6)/7, floored to 1 for val==0.
;     No loop, no store — a leaf the caller uses to pre-size the destination
;     (same "caller pre-sizes dst" contract as hex/base64).
;   * Decode is bounds-safe: it reads at most `n` bytes and at most 10 groups.
;     Shifts are guarded (select on shift<64) so an over-long/garbage stream can
;     never form a poison `shl i64 %x, >=64`. Two failure modes, both cold:
;       - truncated: continuation bit set but the buffer ran out;
;       - overflow: a 64-bit word cannot hold the encoded magnitude.
;   * ERROR CONVENTION (documented, matches hex): the i64-returning entries
;     return the byte count on success (encode: bytes written; decode: bytes
;     consumed) and a NEGATIVE errno on failure. Decode returns -13 (PARSE) for
;     truncated input and -3 (SIZE_OVERFLOW) for a value too large for 64 bits.
;     Decoded value is written through out_val ONLY on success.
;   * zigzag is pure ALU (shift/xor), no branch, single instruction pair each
;     way; exported in its own right so callers can compose their own wire form.
;
; API:
;   i64 universe_varint_uleb_len(i64 val)                    ; 1..10
;   i64 universe_varint_sleb_len(i64 val)                    ; 1..10
;   i64 universe_varint_uleb_encode(ptr dst, i64 val)        ; bytes written
;   i64 universe_varint_sleb_encode(ptr dst, i64 val)        ; bytes written
;   i64 universe_varint_uleb_decode(ptr src, i64 n, ptr out) ; consumed, or <0
;   i64 universe_varint_sleb_decode(ptr src, i64 n, ptr out) ; consumed, or <0
;   i64 universe_varint_zigzag_encode(i64 v)                 ; signed  -> unsigned
;   i64 universe_varint_zigzag_decode(i64 u)                 ; unsigned-> signed

declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.umax.i64(i64, i64)

; ------------------------------------------------------------------- uleb_len
; significant bits = 64 - clz(val); bytes = max(1, (bits+6)/7).
define i64 @universe_varint_uleb_len(i64 %val) local_unnamed_addr #2 {
entry:
  %lz = call i64 @llvm.ctlz.i64(i64 %val, i1 false)
  %bits = sub nuw nsw i64 64, %lz
  %t = add nuw nsw i64 %bits, 6
  %div = udiv i64 %t, 7
  %len = call i64 @llvm.umax.i64(i64 %div, i64 1)
  ret i64 %len
}

; ------------------------------------------------------------------- sleb_len
; Counts the groups the sleb encoder would emit, without storing. At most 10
; iterations; not a hot path, so a straightforward loop is correct and cheap.
define i64 @universe_varint_sleb_len(i64 %val) local_unnamed_addr #3 {
entry:
  br label %loop

loop:
  %v = phi i64 [ %val, %entry ], [ %vn, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %b = and i64 %v, 127
  %vn = ashr i64 %v, 7
  %sign = and i64 %b, 64
  ; done when (vn==0 && sign==0) || (vn==-1 && sign!=0)
  %vz = icmp eq i64 %vn, 0
  %s0 = icmp eq i64 %sign, 0
  %done.pos = and i1 %vz, %s0
  %vn1 = icmp eq i64 %vn, -1
  %s1 = icmp ne i64 %sign, 0
  %done.neg = and i1 %vn1, %s1
  %done = or i1 %done.pos, %done.neg
  %i.next = add nuw i64 %i, 1
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %i.next
}

; ---------------------------------------------------------------- uleb_encode
define i64 @universe_varint_uleb_encode(ptr %dst, i64 %val) local_unnamed_addr #0 {
entry:
  br label %loop

loop:
  %v = phi i64 [ %val, %entry ], [ %vn, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %low = and i64 %v, 127
  %vn = lshr i64 %v, 7
  %more = icmp ne i64 %vn, 0
  %cont = select i1 %more, i64 128, i64 0
  %ov = or disjoint i64 %low, %cont
  %ob = trunc i64 %ov to i8
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %ob, ptr %dp, align 1
  %i.next = add nuw i64 %i, 1
  br i1 %more, label %loop, label %fin

fin:
  ret i64 %i.next
}

; ---------------------------------------------------------------- sleb_encode
define i64 @universe_varint_sleb_encode(ptr %dst, i64 %val) local_unnamed_addr #0 {
entry:
  br label %loop

loop:
  %v = phi i64 [ %val, %entry ], [ %vn, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %low = and i64 %v, 127
  %vn = ashr i64 %v, 7
  %sign = and i64 %low, 64
  %vz = icmp eq i64 %vn, 0
  %s0 = icmp eq i64 %sign, 0
  %done.pos = and i1 %vz, %s0
  %vn1 = icmp eq i64 %vn, -1
  %s1 = icmp ne i64 %sign, 0
  %done.neg = and i1 %vn1, %s1
  %done = or i1 %done.pos, %done.neg
  %cont = select i1 %done, i64 0, i64 128
  %ov = or disjoint i64 %low, %cont
  %ob = trunc i64 %ov to i8
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %ob, ptr %dp, align 1
  %i.next = add nuw i64 %i, 1
  br i1 %done, label %fin, label %loop

fin:
  ret i64 %i.next
}

; ---------------------------------------------------------------- uleb_decode
define i64 @universe_varint_uleb_decode(ptr %src, i64 %n, ptr %out) local_unnamed_addr #1 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %err.trunc, label %loop, !prof !0

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %chk.room ]
  %shift = phi i64 [ 0, %entry ], [ %shift.next, %chk.room ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %chk.room ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b = load i8, ptr %sp, align 1
  %bz = zext i8 %b to i64
  %low = and i64 %bz, 127
  ; guarded contribution: only OR when shift < 64 (shift only reaches 63 here,
  ; but keep the shl in-range regardless so it can never be poison).
  %sh.ok = icmp ult i64 %shift, 64
  %sh.safe = select i1 %sh.ok, i64 %shift, i64 0
  %piece = shl i64 %low, %sh.safe
  %acc.next = or i64 %acc, %piece
  %hi = and i64 %bz, 128
  %has.more = icmp ne i64 %hi, 0
  %i.next = add nuw i64 %i, 1
  br i1 %has.more, label %cont, label %success

; terminal byte: reject if it set bits above bit 63 (shift==63 && low>1).
success:
  %at63 = icmp eq i64 %shift, 63
  %hibits = icmp ugt i64 %low, 1
  %ovf = and i1 %at63, %hibits
  br i1 %ovf, label %err.ovf, label %store, !prof !0

store:
  store i64 %acc.next, ptr %out, align 8
  ret i64 %i.next

cont:
  %shift.next = add nuw nsw i64 %shift, 7
  %ranout = icmp uge i64 %i.next, %n
  br i1 %ranout, label %err.trunc, label %chk.room, !prof !0

chk.room:
  ; a further byte would live at bit >= 64 -> the word cannot hold it.
  %toobig = icmp uge i64 %shift.next, 64
  br i1 %toobig, label %err.ovf, label %loop, !prof !0

err.trunc:                                        ; cold
  ret i64 -13

err.ovf:                                          ; cold
  ret i64 -3
}

; ---------------------------------------------------------------- sleb_decode
define i64 @universe_varint_sleb_decode(ptr %src, i64 %n, ptr %out) local_unnamed_addr #1 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %err.trunc, label %loop, !prof !0

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %chk.room ]
  %shift = phi i64 [ 0, %entry ], [ %shift.next, %chk.room ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %chk.room ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b = load i8, ptr %sp, align 1
  %bz = zext i8 %b to i64
  %low = and i64 %bz, 127
  %sh.ok = icmp ult i64 %shift, 64
  %sh.safe = select i1 %sh.ok, i64 %shift, i64 0
  %piece.raw = shl i64 %low, %sh.safe
  %piece = select i1 %sh.ok, i64 %piece.raw, i64 0
  %acc.next = or i64 %acc, %piece
  %hi = and i64 %bz, 128
  %has.more = icmp ne i64 %hi, 0
  %i.next = add nuw i64 %i, 1
  br i1 %has.more, label %cont, label %success

success:
  ; sign-extend from the terminal group if its sign bit (0x40) is set and there
  ; are bits above the group to fill (shift < 64).
  %sign = and i64 %low, 64
  %neg = icmp ne i64 %sign, 0
  %fill = and i1 %neg, %sh.ok
  %ones.raw = shl i64 -1, %sh.safe
  %ext = select i1 %fill, i64 %ones.raw, i64 0
  %final = or i64 %acc.next, %ext
  store i64 %final, ptr %out, align 8
  ret i64 %i.next

cont:
  %shift.next = add nuw nsw i64 %shift, 7
  %ranout = icmp uge i64 %i.next, %n
  br i1 %ranout, label %err.trunc, label %chk.room, !prof !0

chk.room:
  %toobig = icmp uge i64 %shift.next, 64
  br i1 %toobig, label %err.ovf, label %loop, !prof !0

err.trunc:                                        ; cold
  ret i64 -13

err.ovf:                                          ; cold
  ret i64 -3
}

; -------------------------------------------------------------- zigzag_encode
; (v << 1) ^ (v >> 63)  — arithmetic shift makes the mask 0 or all-ones.
define i64 @universe_varint_zigzag_encode(i64 %v) local_unnamed_addr #4 {
entry:
  %sh = shl i64 %v, 1
  %mask = ashr i64 %v, 63
  %r = xor i64 %sh, %mask
  ret i64 %r
}

; -------------------------------------------------------------- zigzag_decode
; (u >>u 1) ^ -(u & 1)
define i64 @universe_varint_zigzag_decode(i64 %u) local_unnamed_addr #4 {
entry:
  %sh = lshr i64 %u, 1
  %lsb = and i64 %u, 1
  %neg = sub nsw i64 0, %lsb
  %r = xor i64 %sh, %neg
  ret i64 %r
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
