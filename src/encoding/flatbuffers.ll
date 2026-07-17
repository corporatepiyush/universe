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

; FlatBuffers binary-wire reader — zero-copy, allocation-free, bounds-checked
; navigation over a caller-owned byte buffer. Implements the official wire
; format (all little-endian, which every target here already is, so a byte
; view IS the native scalar):
;   * uoffset u32  — forward reference, relative to the field that stores it
;                    (object = self_loc + uoffset).
;   * soffset i32  — table->vtable link, SIGNED (vtable = table - soffset); the
;                    canonical builder emits a positive soffset (vtable before
;                    table) but a real Arrow stream stores a NEGATIVE soffset,
;                    so the i32 is sign-extended, never zero-extended.
;   * voffset u16  — a field's byte offset within its table, held in the vtable.
;   * Buffer root: uoffset at byte 0 -> root table at 0+uoffset. A size-prefixed
;                  buffer puts a u32 byte-count at 0, then the flatbuffer proper
;                  at byte 4 (all its relative offsets are shift-invariant).
;   * Table @ T: soffset S at T; vtable V = T - S. vtable =
;       [u16 vtable_bytes][u16 table_bytes][u16 field0_voffset]...
;     field_count = (vtable_bytes-4)/2. Field i present iff its voffset != 0;
;     the field's data lives at T + voffset. Absent -> caller's default.
;   * Scalar field: value stored inline at T+voffset.
;   * String @ P: [u32 len][len bytes][implicit NUL]. Data view = (P+4, len).
;   * Vector @ P: [u32 count][elements]. Element stride = scalar size, or 4
;     (a uoffset) for vectors of tables/strings.
;   * A sub-object field (table/string/vector) stores a uoffset at T+voffset;
;     follow it with `indirect` (object = loc + u32@loc). `string`/`vector`
;     fold that indirection in, so they take the loc that HOLDS the uoffset
;     (a table field loc, or a vector-of-strings element loc) directly.
;
; DESIGN:
;   * ZERO allocation, ZERO copy: every accessor returns either an i64 byte
;     location into the caller's buffer, or a (ptr,len) VIEW into it. The buffer
;     is the source of truth; nothing is materialized.
;   * A "loc" is an i64 byte offset from buf. A NEGATIVE loc means absent (field
;     not in the vtable) OR malformed (a hostile offset that would escape the
;     buffer) — both collapse to "not present", which is always SAFE: the scalar
;     readers substitute the caller's default for a negative loc, so a corrupt
;     buffer degrades to defaults, never a wild read. Callers that must tell
;     absent from present test `loc < 0`.
;   * UNTRUSTED INPUT is the whole point. Four internal leaf loaders
;     (fb.u16/fb.u32/fb.i32/fb.read_bits) are the ONLY places that dereference
;     the buffer, and each range-checks pos>=0 && pos+width<=len BEFORE the GEP+
;     load (hazard #15: an OOB `inbounds` load is UB the optimizer weaponizes).
;     A hostile uoffset/soffset/voffset can move `pos` anywhere, but it can never
;     produce an in-bounds `inbounds` GEP that reads outside the slice — the
;     check gates the load. The sanitizer gate + a truncate/mutate fuzz loop
;     prove the memory safety; here every public entry returns a documented code.
;   * Compute/memory separation: the loaders are `internal alwaysinline` leaves,
;     so within this module the bounds-check seam vanishes — a scalar read folds
;     to a predicted range branch + one native load. The width `n` passed to
;     fb.read_bits is a call-site constant, so its memcpy lowers to a single
;     typed load/store per type (verified via opt -O3).
;   * Signedness: soffset uses fb.i32 (sext); every offset/length/voffset is
;     unsigned (fb.u16/fb.u32, zext) so a large value stays large, not negative,
;     and the range check rejects it. No shift math here, so hazard #10 N/A.
;   * Error map (i32-returning entries): 0 OK, 1 NULL_PTR, 8 INVALID_ARG for any
;     malformed/truncated/OOB wire data. Loc-returning entries use <0 for
;     absent/malformed. `vector_elem` overflow-checks its index math and returns
;     -1 on overflow.
;
; API:
;   i64 universe_encoding_flatbuffers_root(ptr buf, i64 len)
;   i64 universe_encoding_flatbuffers_sized_root(ptr buf, i64 len)
;   i64 universe_encoding_flatbuffers_field(ptr buf, i64 len, i64 table, i32 field_id)
;   i32 universe_encoding_flatbuffers_read_i8 (ptr buf,i64 len,i64 loc,i8  dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_u8 (ptr buf,i64 len,i64 loc,i8  dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_i16(ptr buf,i64 len,i64 loc,i16 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_u16(ptr buf,i64 len,i64 loc,i16 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_i32(ptr buf,i64 len,i64 loc,i32 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_u32(ptr buf,i64 len,i64 loc,i32 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_i64(ptr buf,i64 len,i64 loc,i64 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_u64(ptr buf,i64 len,i64 loc,i64 dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_f32(ptr buf,i64 len,i64 loc,float  dflt,ptr out)
;   i32 universe_encoding_flatbuffers_read_f64(ptr buf,i64 len,i64 loc,double dflt,ptr out)
;   i64 universe_encoding_flatbuffers_indirect(ptr buf, i64 len, i64 loc)
;   i32 universe_encoding_flatbuffers_string(ptr buf,i64 len,i64 loc,ptr out_ptr,ptr out_len)
;   i32 universe_encoding_flatbuffers_vector(ptr buf,i64 len,i64 loc,ptr out_start,ptr out_count)
;   i64 universe_encoding_flatbuffers_vector_elem(i64 vec_start, i32 stride, i64 i)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; =========================================================== leaf loaders
; The ONLY functions that dereference the buffer. Each range-checks before the
; load; a false return means out-of-range (the load never happened).

; fb.u16: zero-extended little-endian u16 at pos.
define internal i1 @fb.u16(ptr %buf, i64 %len, i64 %pos, ptr %out) #0 {
entry:
  %neg = icmp slt i64 %pos, 0
  %end = add nsw i64 %pos, 2
  %past = icmp ugt i64 %end, %len
  %bad = or i1 %neg, %past
  br i1 %bad, label %fail, label %ok, !prof !0
ok:
  %p = getelementptr inbounds i8, ptr %buf, i64 %pos
  %v = load i16, ptr %p, align 1
  %z = zext i16 %v to i64
  store i64 %z, ptr %out, align 8
  ret i1 true
fail:                                             ; cold
  ret i1 false
}

; fb.u32: zero-extended little-endian u32 at pos.
define internal i1 @fb.u32(ptr %buf, i64 %len, i64 %pos, ptr %out) #0 {
entry:
  %neg = icmp slt i64 %pos, 0
  %end = add nsw i64 %pos, 4
  %past = icmp ugt i64 %end, %len
  %bad = or i1 %neg, %past
  br i1 %bad, label %fail, label %ok, !prof !0
ok:
  %p = getelementptr inbounds i8, ptr %buf, i64 %pos
  %v = load i32, ptr %p, align 1
  %z = zext i32 %v to i64
  store i64 %z, ptr %out, align 8
  ret i1 true
fail:                                             ; cold
  ret i1 false
}

; fb.i32: SIGN-extended little-endian i32 at pos (soffset).
define internal i1 @fb.i32(ptr %buf, i64 %len, i64 %pos, ptr %out) #0 {
entry:
  %neg = icmp slt i64 %pos, 0
  %end = add nsw i64 %pos, 4
  %past = icmp ugt i64 %end, %len
  %bad = or i1 %neg, %past
  br i1 %bad, label %fail, label %ok, !prof !0
ok:
  %p = getelementptr inbounds i8, ptr %buf, i64 %pos
  %v = load i32, ptr %p, align 1
  %s = sext i32 %v to i64
  store i64 %s, ptr %out, align 8
  ret i1 true
fail:                                             ; cold
  ret i1 false
}

; fb.read_bits: shared body of every scalar reader. loc<0 -> store the low n
; bytes of %dflt (little-endian) to out; otherwise range-check and copy n bytes
; from buf+loc. n is a call-site constant so the memcpy folds to one typed op.
define internal i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %dflt, ptr %out, i64 %n) #1 {
entry:
  %dslot = alloca i64, align 8
  %on = icmp eq ptr %out, null
  br i1 %on, label %err.null, label %chk.loc, !prof !0
chk.loc:
  %absent = icmp slt i64 %loc, 0
  br i1 %absent, label %use.dflt, label %present, !prof !0
present:
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %err.null, label %range, !prof !0
range:
  %end = add nsw i64 %loc, %n
  %oob = icmp ugt i64 %end, %len
  br i1 %oob, label %err.oob, label %copy, !prof !0
copy:
  %src = getelementptr inbounds i8, ptr %buf, i64 %loc
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %out, ptr align 1 %src, i64 %n, i1 false)
  ret i32 0
use.dflt:
  store i64 %dflt, ptr %dslot, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %out, ptr align 8 %dslot, i64 %n, i1 false)
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.oob:                                          ; cold
  ret i32 8
}

; ==================================================================== root
; Root table location = uoffset at byte 0. Validates that the table's soffset is
; itself readable (table+4 <= len); the vtable walk in `field` checks the rest.
define i64 @universe_encoding_flatbuffers_root(ptr %buf, i64 %len) #2 {
entry:
  %slot = alloca i64, align 8
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %fail, label %read, !prof !0
read:
  %ok = call i1 @fb.u32(ptr %buf, i64 %len, i64 0, ptr %slot)
  br i1 %ok, label %chk, label %fail, !prof !0
chk:
  %t = load i64, ptr %slot, align 8
  %end = add nsw i64 %t, 4
  %past = icmp ugt i64 %end, %len
  br i1 %past, label %fail, label %ret, !prof !0
ret:
  ret i64 %t
fail:                                             ; cold
  ret i64 -1
}

; ============================================================== sized_root
; Size-prefixed root: u32 byte-count at 0, flatbuffer proper at byte 4, so the
; root uoffset is at byte 4 and the table = 4 + uoffset. Validates 4+size<=len.
define i64 @universe_encoding_flatbuffers_sized_root(ptr %buf, i64 %len) #2 {
entry:
  %szslot = alloca i64, align 8
  %ooslot = alloca i64, align 8
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %fail, label %size, !prof !0
size:
  %oks = call i1 @fb.u32(ptr %buf, i64 %len, i64 0, ptr %szslot)
  br i1 %oks, label %chk.size, label %fail, !prof !0
chk.size:
  %sz = load i64, ptr %szslot, align 8
  %bodyend = add nsw i64 %sz, 4
  %spast = icmp ugt i64 %bodyend, %len
  br i1 %spast, label %fail, label %uoff, !prof !0
uoff:
  %oko = call i1 @fb.u32(ptr %buf, i64 %len, i64 4, ptr %ooslot)
  br i1 %oko, label %chk.tab, label %fail, !prof !0
chk.tab:
  %uo = load i64, ptr %ooslot, align 8
  %t = add nsw i64 %uo, 4
  %tend = add nsw i64 %t, 4
  %tpast = icmp ugt i64 %tend, %len
  br i1 %tpast, label %fail, label %ret, !prof !0
ret:
  ret i64 %t
fail:                                             ; cold
  ret i64 -1
}

; =================================================================== field
; Field data location for table field `field_id`, or -1 (absent/malformed).
define i64 @universe_encoding_flatbuffers_field(ptr %buf, i64 %len, i64 %table, i32 %field_id) #2 {
entry:
  %soslot = alloca i64, align 8
  %vsslot = alloca i64, align 8
  %voslot = alloca i64, align 8
  %bn = icmp eq ptr %buf, null
  %tn = icmp slt i64 %table, 0
  %fn = icmp slt i32 %field_id, 0
  %n0 = or i1 %bn, %tn
  %bad0 = or i1 %n0, %fn
  br i1 %bad0, label %fail, label %soff, !prof !0
soff:
  %oks = call i1 @fb.i32(ptr %buf, i64 %len, i64 %table, ptr %soslot)
  br i1 %oks, label %vt, label %fail, !prof !0
vt:
  %so = load i64, ptr %soslot, align 8
  %vtab = sub nsw i64 %table, %so
  %okvs = call i1 @fb.u16(ptr %buf, i64 %len, i64 %vtab, ptr %vsslot)
  br i1 %okvs, label %count, label %fail, !prof !0
count:
  %vsize = load i64, ptr %vsslot, align 8
  %small = icmp ult i64 %vsize, 4
  br i1 %small, label %fail, label %fc, !prof !0
fc:
  %datab = sub nsw i64 %vsize, 4
  %fcount = lshr i64 %datab, 1
  %fid64 = zext i32 %field_id to i64
  %oorange = icmp uge i64 %fid64, %fcount
  br i1 %oorange, label %fail, label %vopos, !prof !0
vopos:
  %fx2 = shl nuw nsw i64 %fid64, 1
  %vo.at = add nsw i64 %vtab, 4
  %vo.pos = add nsw i64 %vo.at, %fx2
  %okvo = call i1 @fb.u16(ptr %buf, i64 %len, i64 %vo.pos, ptr %voslot)
  br i1 %okvo, label %voff, label %fail, !prof !0
voff:
  %voffv = load i64, ptr %voslot, align 8
  %isz = icmp eq i64 %voffv, 0
  br i1 %isz, label %fail, label %loc, !prof !0
loc:
  %fl = add nsw i64 %table, %voffv
  ret i64 %fl
fail:                                             ; cold  (absent or malformed)
  ret i64 -1
}

; ================================================================ indirect
; Follow a uoffset stored at `loc`: object = loc + u32@loc, or -1.
define i64 @universe_encoding_flatbuffers_indirect(ptr %buf, i64 %len, i64 %loc) #2 {
entry:
  %slot = alloca i64, align 8
  %bn = icmp eq ptr %buf, null
  %ln = icmp slt i64 %loc, 0
  %bad = or i1 %bn, %ln
  br i1 %bad, label %fail, label %read, !prof !0
read:
  %ok = call i1 @fb.u32(ptr %buf, i64 %len, i64 %loc, ptr %slot)
  br i1 %ok, label %chk, label %fail, !prof !0
chk:
  %uo = load i64, ptr %slot, align 8
  %obj = add nsw i64 %loc, %uo
  %past = icmp ugt i64 %obj, %len
  br i1 %past, label %fail, label %ret, !prof !0
ret:
  ret i64 %obj
fail:                                             ; cold
  ret i64 -1
}

; ---- scalar readers: thin typed wrappers over fb.read_bits ----

define i32 @universe_encoding_flatbuffers_read_i8(ptr %buf, i64 %len, i64 %loc, i8 %dflt, ptr %out) #1 {
entry:
  %d = zext i8 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 1)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_u8(ptr %buf, i64 %len, i64 %loc, i8 %dflt, ptr %out) #1 {
entry:
  %d = zext i8 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 1)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_i16(ptr %buf, i64 %len, i64 %loc, i16 %dflt, ptr %out) #1 {
entry:
  %d = zext i16 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 2)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_u16(ptr %buf, i64 %len, i64 %loc, i16 %dflt, ptr %out) #1 {
entry:
  %d = zext i16 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 2)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_i32(ptr %buf, i64 %len, i64 %loc, i32 %dflt, ptr %out) #1 {
entry:
  %d = zext i32 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 4)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_u32(ptr %buf, i64 %len, i64 %loc, i32 %dflt, ptr %out) #1 {
entry:
  %d = zext i32 %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 4)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_i64(ptr %buf, i64 %len, i64 %loc, i64 %dflt, ptr %out) #1 {
entry:
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %dflt, ptr %out, i64 8)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_u64(ptr %buf, i64 %len, i64 %loc, i64 %dflt, ptr %out) #1 {
entry:
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %dflt, ptr %out, i64 8)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_f32(ptr %buf, i64 %len, i64 %loc, float %dflt, ptr %out) #1 {
entry:
  %bits = bitcast float %dflt to i32
  %d = zext i32 %bits to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 4)
  ret i32 %r
}
define i32 @universe_encoding_flatbuffers_read_f64(ptr %buf, i64 %len, i64 %loc, double %dflt, ptr %out) #1 {
entry:
  %d = bitcast double %dflt to i64
  %r = call i32 @fb.read_bits(ptr %buf, i64 %len, i64 %loc, i64 %d, ptr %out, i64 8)
  ret i32 %r
}

; ================================================================== string
; Loc holds a uoffset -> string object [u32 len][bytes]. Returns a (ptr,len)
; view. loc<0 (absent) -> (null,0), status 0. Malformed/OOB -> 8.
define i32 @universe_encoding_flatbuffers_string(ptr %buf, i64 %len, i64 %loc, ptr %out_ptr, ptr %out_len) #3 {
entry:
  %uoslot = alloca i64, align 8
  %lnslot = alloca i64, align 8
  %pn = icmp eq ptr %out_ptr, null
  %qn = icmp eq ptr %out_len, null
  %nn = or i1 %pn, %qn
  br i1 %nn, label %err.null, label %chk.loc, !prof !0
chk.loc:
  %absent = icmp slt i64 %loc, 0
  br i1 %absent, label %empty, label %ind, !prof !0
ind:
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %err.null, label %follow, !prof !0
follow:
  %oku = call i1 @fb.u32(ptr %buf, i64 %len, i64 %loc, ptr %uoslot)
  br i1 %oku, label %obj, label %err.bad, !prof !0
obj:
  %uo = load i64, ptr %uoslot, align 8
  %objpos = add nsw i64 %loc, %uo
  %okl = call i1 @fb.u32(ptr %buf, i64 %len, i64 %objpos, ptr %lnslot)
  br i1 %okl, label %data, label %err.bad, !prof !0
data:
  %slen = load i64, ptr %lnslot, align 8
  %datapos = add nsw i64 %objpos, 4
  %end = add nsw i64 %datapos, %slen
  %past = icmp ugt i64 %end, %len
  br i1 %past, label %err.bad, label %store, !prof !0
store:
  %dp = getelementptr inbounds i8, ptr %buf, i64 %datapos
  store ptr %dp, ptr %out_ptr, align 8
  store i64 %slen, ptr %out_len, align 8
  ret i32 0
empty:
  store ptr null, ptr %out_ptr, align 8
  store i64 0, ptr %out_len, align 8
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ================================================================== vector
; Loc holds a uoffset -> vector object [u32 count][elements]. Returns the loc of
; element 0 (out_start) and the element count. loc<0 -> (-1,0), status 0. Each
; element is bounds-checked when the caller reads it, so only the header is
; validated here. Malformed/OOB header -> 8.
define i32 @universe_encoding_flatbuffers_vector(ptr %buf, i64 %len, i64 %loc, ptr %out_start, ptr %out_count) #3 {
entry:
  %uoslot = alloca i64, align 8
  %cnslot = alloca i64, align 8
  %sn = icmp eq ptr %out_start, null
  %cn = icmp eq ptr %out_count, null
  %nn = or i1 %sn, %cn
  br i1 %nn, label %err.null, label %chk.loc, !prof !0
chk.loc:
  %absent = icmp slt i64 %loc, 0
  br i1 %absent, label %empty, label %ind, !prof !0
ind:
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %err.null, label %follow, !prof !0
follow:
  %oku = call i1 @fb.u32(ptr %buf, i64 %len, i64 %loc, ptr %uoslot)
  br i1 %oku, label %obj, label %err.bad, !prof !0
obj:
  %uo = load i64, ptr %uoslot, align 8
  %objpos = add nsw i64 %loc, %uo
  %okc = call i1 @fb.u32(ptr %buf, i64 %len, i64 %objpos, ptr %cnslot)
  br i1 %okc, label %store, label %err.bad, !prof !0
store:
  %count = load i64, ptr %cnslot, align 8
  %start = add nsw i64 %objpos, 4
  store i64 %start, ptr %out_start, align 8
  store i64 %count, ptr %out_count, align 8
  ret i32 0
empty:
  store i64 -1, ptr %out_start, align 8
  store i64 0, ptr %out_count, align 8
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ============================================================= vector_elem
; Element loc = vec_start + stride*i, overflow-checked. -1 on bad start/overflow.
; The caller follows a uoffset here for a vector of tables/strings.
define i64 @universe_encoding_flatbuffers_vector_elem(i64 %vec_start, i32 %stride, i64 %i) #4 {
entry:
  %bad.start = icmp slt i64 %vec_start, 0
  br i1 %bad.start, label %fail, label %mul, !prof !0
mul:
  %st = zext i32 %stride to i64
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %st, i64 %i)
  %off = extractvalue { i64, i1 } %m, 0
  %movf = extractvalue { i64, i1 } %m, 1
  br i1 %movf, label %fail, label %add, !prof !0
add:
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %vec_start, i64 %off)
  %eloc = extractvalue { i64, i1 } %a, 0
  %aovf = extractvalue { i64, i1 } %a, 1
  br i1 %aovf, label %fail, label %ret, !prof !0
ret:
  ret i64 %eloc
fail:                                             ; cold
  ret i64 -1
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(none) }

!0 = !{!"branch_weights", i32 2000, i32 1}
