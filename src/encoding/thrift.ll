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

; Thrift compact-protocol reader — read-only, allocation-free cursor over a
; caller-owned byte buffer. The compact protocol frames structs on the wire as:
;   * unsigned LEB128 varints (7 data bits/byte, MSB = continuation),
;   * zig-zag transform for signed i16/i32/i64,
;   * field headers as (delta << 4) | type, with a zig-zag varint field id when
;     the 4-bit delta is 0,
;   * collection headers as (size << 4) | elem-type, with a varint size when the
;     4-bit size == 15,
;   * doubles as 8 little-endian bytes, binary/string as varint-length + bytes.
;
; DESIGN:
;   * ZERO allocation, ZERO copy. Reader state is a 32-byte struct in CALLER
;     memory; binary/string reads return a VIEW (ptr,len) into the input buffer,
;     never a strdup. This is the "parse zero-copy: return slices into the read
;     buffer" rule — the buffer is the source of truth.
;   * Reader layout (bytes, computed by hand so it is identical on every target):
;       +0  base : ptr    (start of the borrowed buffer)
;       +8  len  : i64    (buffer length)
;       +16 pos  : i64    (cursor)
;       +24 last_field_id : i16   (compact-protocol field-id delta base)
;       (padded to 32)
;   * UNTRUSTED INPUT: every byte read is bounds-checked against len. The two
;     leaf accessors (thrift.byte / thrift.take) are the ONLY places that touch
;     the buffer; both refuse to index past the slice and signal truncation
;     (byte -> -1, take -> null). No public read can OOB — the sanitizer gate
;     proves it. Malformed/truncated wire data returns INVALID_ARG (8); a null
;     pointer arg returns NULL_PTR (1). Nothing panics.
;   * varint decode reuses varint.ll's in-range-shift discipline (hazard #10):
;     the shl amount is select-clamped to <64 so an over-long/garbage stream can
;     never form a poison `shl i64 %x, >=64`; a continuation past bit 63 is
;     rejected as malformed. Bound-then-guard, belt and suspenders.
;   * SKIP is DEPTH-LIMITED (max 64): a hostile deeply-nested struct/list/map
;     cannot exhaust the call stack — recursion returns INVALID_ARG once depth
;     exceeds the cap, before reading anything. skip is the one recursive symbol
;     here (NOT norecurse); every other function is a norecurse leaf/near-leaf.
;   * Compute is register-resident: the varint value math never stores mid-loop.
;     Leaf accessors are `internal alwaysinline` so within this module the seam
;     to bounds-checking vanishes — a typed read is a load + a predicted branch.
;   * Bool encoding is context-dependent (spec): a BOOL struct FIELD carries its
;     value in the header type nibble (1 = true, 2 = false, no separate byte); a
;     bool inside a collection is a full byte. `field` surfaces the nibble as the
;     ctype; `skip` reads a byte per bool collection element.
;
; Compact type nibbles: STOP=0 BOOL_TRUE=1 BOOL_FALSE=2 I8=3 I16=4 I32=5 I64=6
;   DOUBLE=7 BINARY/STRING=8 LIST=9 SET=10 MAP=11 STRUCT=12.
;
; API (all i32-returning entries: 0 OK, 1 NULL_PTR, 8 INVALID_ARG=malformed/
; truncated). STOP is surfaced by `field` as out_ctype == 0.
;   void universe_encoding_thrift_init(ptr reader, ptr buf, i64 len)
;   i32  universe_encoding_thrift_read_uvarint(ptr reader, ptr out_u64)
;   i32  universe_encoding_thrift_read_i8 (ptr reader, ptr out_i8)   ; raw byte
;   i32  universe_encoding_thrift_read_i16(ptr reader, ptr out_i16)  ; zig-zag
;   i32  universe_encoding_thrift_read_i32(ptr reader, ptr out_i32)  ; zig-zag
;   i32  universe_encoding_thrift_read_i64(ptr reader, ptr out_i64)  ; zig-zag
;   i32  universe_encoding_thrift_read_double(ptr reader, ptr out_f64)
;   i32  universe_encoding_thrift_read_binary(ptr reader, ptr out_ptr, ptr out_len)
;   i32  universe_encoding_thrift_field(ptr reader, ptr out_ctype_u8, ptr out_fid_i16)
;   i32  universe_encoding_thrift_collection(ptr reader, ptr out_etype_u8, ptr out_size_i32)
;   i32  universe_encoding_thrift_skip(ptr reader, i32 ctype, i32 depth)

declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; ============================================================ leaf accessors
; The ONLY two functions that dereference the buffer. Both bounds-check.

; thrift.byte: read one byte and advance pos; returns 0..255, or -1 truncated.
define internal i64 @thrift.byte(ptr %r) #0 {
entry:
  %pp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %pos = load i64, ptr %pp, align 8
  %lp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %len = load i64, ptr %lp, align 8
  %oob = icmp uge i64 %pos, %len
  br i1 %oob, label %trunc, label %ok, !prof !0
ok:
  %base = load ptr, ptr %r, align 8
  %bp = getelementptr inbounds nuw i8, ptr %base, i64 %pos
  %b = load i8, ptr %bp, align 1
  %bz = zext i8 %b to i64
  %posn = add nuw i64 %pos, 1
  store i64 %posn, ptr %pp, align 8
  ret i64 %bz
trunc:                                            ; cold
  ret i64 -1
}

; thrift.take: borrow n bytes (view), advance pos; returns base+pos, or null.
define internal ptr @thrift.take(ptr %r, i64 %n) #0 {
entry:
  %pp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %pos = load i64, ptr %pp, align 8
  %lp = getelementptr inbounds nuw i8, ptr %r, i64 8
  %len = load i64, ptr %lp, align 8
  %ovs = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pos, i64 %n)
  %end = extractvalue { i64, i1 } %ovs, 0
  %ov = extractvalue { i64, i1 } %ovs, 1
  br i1 %ov, label %bad, label %chk, !prof !0
chk:
  %past = icmp ugt i64 %end, %len
  br i1 %past, label %bad, label %ok, !prof !0
ok:
  %base = load ptr, ptr %r, align 8
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %pos
  store i64 %end, ptr %pp, align 8
  ret ptr %p
bad:                                              ; cold
  ret ptr null
}

; ==================================================================== init
define void @universe_encoding_thrift_init(ptr %reader, ptr %buf, i64 %len) #2 {
entry:
  %isnull = icmp eq ptr %reader, null
  br i1 %isnull, label %ret, label %store, !prof !0
store:
  store ptr %buf, ptr %reader, align 8
  %lp = getelementptr inbounds nuw i8, ptr %reader, i64 8
  store i64 %len, ptr %lp, align 8
  %pp = getelementptr inbounds nuw i8, ptr %reader, i64 16
  store i64 0, ptr %pp, align 8
  %fp = getelementptr inbounds nuw i8, ptr %reader, i64 24
  store i16 0, ptr %fp, align 8
  br label %ret
ret:
  ret void
}

; ============================================================ read_uvarint
; Unsigned LEB128, capped at 64 bits. Over-long encodings are rejected.
define i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %out) #1 {
entry:
  %rn = icmp eq ptr %r, null
  %on = icmp eq ptr %out, null
  %anynull = or i1 %rn, %on
  br i1 %anynull, label %err.null, label %loop, !prof !0

loop:
  %shift = phi i64 [ 0, %entry ], [ %shn, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %accn, %cont ]
  %b = call i64 @thrift.byte(ptr %r)
  %trunc = icmp slt i64 %b, 0
  br i1 %trunc, label %err.bad, label %chkshift, !prof !0
chkshift:
  ; a continuation would place bits at shift>=64 -> the word cannot hold it.
  %toolong = icmp uge i64 %shift, 64
  br i1 %toolong, label %err.bad, label %accum, !prof !0
accum:
  %low = and i64 %b, 127
  %sh.ok = icmp ult i64 %shift, 64
  %sh.safe = select i1 %sh.ok, i64 %shift, i64 0
  %piece = shl i64 %low, %sh.safe
  %accn = or i64 %acc, %piece
  %hi = and i64 %b, 128
  %more = icmp ne i64 %hi, 0
  br i1 %more, label %cont, label %done
cont:
  %shn = add nuw nsw i64 %shift, 7
  br label %loop
done:
  store i64 %accn, ptr %out, align 8
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ================================================================= read_i8
; A raw signed byte (compact I8 has no zig-zag / varint framing).
define i32 @universe_encoding_thrift_read_i8(ptr %r, ptr %out) #1 {
entry:
  %rn = icmp eq ptr %r, null
  %on = icmp eq ptr %out, null
  %anynull = or i1 %rn, %on
  br i1 %anynull, label %err.null, label %read, !prof !0
read:
  %b = call i64 @thrift.byte(ptr %r)
  %trunc = icmp slt i64 %b, 0
  br i1 %trunc, label %err.bad, label %store, !prof !0
store:
  %b8 = trunc i64 %b to i8
  store i8 %b8, ptr %out, align 1
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ================================================================ read_i64
; zig-zag varint: (u>>1) ^ -(u&1).
define i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %out) #1 {
entry:
  %uslot = alloca i64, align 8
  %on = icmp eq ptr %out, null
  br i1 %on, label %err.null, label %decode, !prof !0
decode:
  %rc = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %uslot)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %conv, label %ret.rc, !prof !0
conv:
  %u = load i64, ptr %uslot, align 8
  %sh = lshr i64 %u, 1
  %lsb = and i64 %u, 1
  %neg = sub nsw i64 0, %lsb
  %v = xor i64 %sh, %neg
  store i64 %v, ptr %out, align 8
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
}

; ================================================================ read_i32
define i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %out) #1 {
entry:
  %uslot = alloca i64, align 8
  %on = icmp eq ptr %out, null
  br i1 %on, label %err.null, label %decode, !prof !0
decode:
  %rc = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %uslot)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %conv, label %ret.rc, !prof !0
conv:
  %u = load i64, ptr %uslot, align 8
  %sh = lshr i64 %u, 1
  %lsb = and i64 %u, 1
  %neg = sub nsw i64 0, %lsb
  %v = xor i64 %sh, %neg
  ; range-check to i32: sext(trunc(v)) must equal v.
  %v32 = trunc i64 %v to i32
  %back = sext i32 %v32 to i64
  %inrange = icmp eq i64 %back, %v
  br i1 %inrange, label %store, label %err.bad, !prof !0
store:
  store i32 %v32, ptr %out, align 4
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ================================================================ read_i16
define i32 @universe_encoding_thrift_read_i16(ptr %r, ptr %out) #1 {
entry:
  %uslot = alloca i64, align 8
  %on = icmp eq ptr %out, null
  br i1 %on, label %err.null, label %decode, !prof !0
decode:
  %rc = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %uslot)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %conv, label %ret.rc, !prof !0
conv:
  %u = load i64, ptr %uslot, align 8
  %sh = lshr i64 %u, 1
  %lsb = and i64 %u, 1
  %neg = sub nsw i64 0, %lsb
  %v = xor i64 %sh, %neg
  %v16 = trunc i64 %v to i16
  %back = sext i16 %v16 to i64
  %inrange = icmp eq i64 %back, %v
  br i1 %inrange, label %store, label %err.bad, !prof !0
store:
  store i16 %v16, ptr %out, align 2
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ============================================================== read_double
; 8 little-endian bytes -> f64. Targets are all little-endian, so the byte view
; is the native bit pattern; copy the 64 bits through to the caller's slot.
define i32 @universe_encoding_thrift_read_double(ptr %r, ptr %out) #1 {
entry:
  %rn = icmp eq ptr %r, null
  %on = icmp eq ptr %out, null
  %anynull = or i1 %rn, %on
  br i1 %anynull, label %err.null, label %take, !prof !0
take:
  %p = call ptr @thrift.take(ptr %r, i64 8)
  %pn = icmp eq ptr %p, null
  br i1 %pn, label %err.bad, label %store, !prof !0
store:
  %bits = load i64, ptr %p, align 1
  store i64 %bits, ptr %out, align 8
  ret i32 0
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ============================================================== read_binary
; Length-prefixed (varint) view into the buffer — zero copy.
define i32 @universe_encoding_thrift_read_binary(ptr %r, ptr %out_ptr, ptr %out_len) #1 {
entry:
  %nslot = alloca i64, align 8
  %rn = icmp eq ptr %r, null
  %pn = icmp eq ptr %out_ptr, null
  %ln = icmp eq ptr %out_len, null
  %n0 = or i1 %rn, %pn
  %anynull = or i1 %n0, %ln
  br i1 %anynull, label %err.null, label %len, !prof !0
len:
  %rc = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %nslot)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %take, label %ret.rc, !prof !0
take:
  %n = load i64, ptr %nslot, align 8
  %p = call ptr @thrift.take(ptr %r, i64 %n)
  %pnull = icmp eq ptr %p, null
  br i1 %pnull, label %err.bad, label %store, !prof !0
store:
  store ptr %p, ptr %out_ptr, align 8
  store i64 %n, ptr %out_len, align 8
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ==================================================================== field
; Struct field header: (delta<<4)|type. delta==0 -> explicit zig-zag i16 id.
; STOP (byte 0) is surfaced as out_ctype == 0 with return 0.
define i32 @universe_encoding_thrift_field(ptr %r, ptr %out_ctype, ptr %out_fid) #1 {
entry:
  %idslot = alloca i16, align 2
  %rn = icmp eq ptr %r, null
  %cn = icmp eq ptr %out_ctype, null
  %fn = icmp eq ptr %out_fid, null
  %n0 = or i1 %rn, %cn
  %anynull = or i1 %n0, %fn
  br i1 %anynull, label %err.null, label %head, !prof !0
head:
  %b = call i64 @thrift.byte(ptr %r)
  %trunc = icmp slt i64 %b, 0
  br i1 %trunc, label %err.bad, label %notrunc, !prof !0
notrunc:
  %isstop = icmp eq i64 %b, 0
  br i1 %isstop, label %stop, label %cont, !prof !0
stop:
  store i8 0, ptr %out_ctype, align 1
  store i16 0, ptr %out_fid, align 2
  ret i32 0
cont:
  %ctype = and i64 %b, 15
  %delta = lshr i64 %b, 4
  %explicit = icmp eq i64 %delta, 0
  br i1 %explicit, label %read.id, label %delta.blk, !prof !0
read.id:
  %rc = call i32 @universe_encoding_thrift_read_i16(ptr %r, ptr %idslot)
  %rcok = icmp eq i32 %rc, 0
  br i1 %rcok, label %got.explicit, label %ret.rc, !prof !0
got.explicit:
  %eid = load i16, ptr %idslot, align 2
  br label %finish
delta.blk:
  %lfp = getelementptr inbounds nuw i8, ptr %r, i64 24
  %last = load i16, ptr %lfp, align 2
  %d16 = trunc i64 %delta to i16
  %did = add i16 %last, %d16
  br label %finish
finish:
  %idf = phi i16 [ %eid, %got.explicit ], [ %did, %delta.blk ]
  %lfp2 = getelementptr inbounds nuw i8, ptr %r, i64 24
  store i16 %idf, ptr %lfp2, align 2
  %ct8 = trunc i64 %ctype to i8
  store i8 %ct8, ptr %out_ctype, align 1
  store i16 %idf, ptr %out_fid, align 2
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; =============================================================== collection
; List/set/map header: (size<<4)|elem-type, varint size when the nibble == 15.
define i32 @universe_encoding_thrift_collection(ptr %r, ptr %out_etype, ptr %out_size) #1 {
entry:
  %sslot = alloca i64, align 8
  %rn = icmp eq ptr %r, null
  %en = icmp eq ptr %out_etype, null
  %sn = icmp eq ptr %out_size, null
  %n0 = or i1 %rn, %en
  %anynull = or i1 %n0, %sn
  br i1 %anynull, label %err.null, label %head, !prof !0
head:
  %b = call i64 @thrift.byte(ptr %r)
  %trunc = icmp slt i64 %b, 0
  br i1 %trunc, label %err.bad, label %split, !prof !0
split:
  %etype = and i64 %b, 15
  %snib = lshr i64 %b, 4
  %is15 = icmp eq i64 %snib, 15
  br i1 %is15, label %big, label %store, !prof !0
big:
  %rc = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %sslot)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %big.ok, label %ret.rc, !prof !0
big.ok:
  %sz = load i64, ptr %sslot, align 8
  %toobig = icmp ugt i64 %sz, 4294967295
  br i1 %toobig, label %err.bad, label %store, !prof !0
store:
  %size = phi i64 [ %snib, %split ], [ %sz, %big.ok ]
  %et8 = trunc i64 %etype to i8
  store i8 %et8, ptr %out_etype, align 1
  %sz32 = trunc i64 %size to i32
  store i32 %sz32, ptr %out_size, align 4
  ret i32 0
ret.rc:
  ret i32 %rc
err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

; ===================================================================== skip
; Depth-limited recursive skip of a value of compact type %ctype. Consumes the
; value (and any nested containers) from the reader. Returns 8 on truncation /
; malformed / depth overflow. NOT norecurse.
define i32 @universe_encoding_thrift_skip(ptr %r, i32 %ctype, i32 %depth) #3 {
entry:
  %slot = alloca i64, align 8
  %et8 = alloca i8, align 1
  %sz32 = alloca i32, align 4
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %rn = icmp eq ptr %r, null
  br i1 %rn, label %err.null, label %chkdepth, !prof !0
chkdepth:
  %deep = icmp ugt i32 %depth, 64
  br i1 %deep, label %err.bad, label %dispatch, !prof !0
dispatch:
  %d1 = add nuw nsw i32 %depth, 1
  switch i32 %ctype, label %err.bad [
    i32 1, label %noop         ; bool_true  (value in nibble)
    i32 2, label %noop         ; bool_false
    i32 3, label %skip.byte    ; i8
    i32 4, label %skip.var     ; i16
    i32 5, label %skip.var     ; i32
    i32 6, label %skip.var     ; i64
    i32 7, label %skip.dbl     ; double
    i32 8, label %skip.bin     ; binary/string
    i32 9, label %skip.list    ; list
    i32 10, label %skip.list   ; set
    i32 11, label %skip.map    ; map
    i32 12, label %skip.struct ; struct
  ]

noop:
  ret i32 0

skip.byte:
  %bb = call i64 @thrift.byte(ptr %r)
  %bbt = icmp slt i64 %bb, 0
  br i1 %bbt, label %err.bad, label %noop, !prof !0

skip.var:
  %rcv = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %slot)
  ret i32 %rcv

skip.dbl:
  %pd = call ptr @thrift.take(ptr %r, i64 8)
  %pdn = icmp eq ptr %pd, null
  br i1 %pdn, label %err.bad, label %noop, !prof !0

skip.bin:
  %rcl = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %slot)
  %rclok = icmp eq i32 %rcl, 0
  br i1 %rclok, label %skip.bin.take, label %ret.rcl, !prof !0
skip.bin.take:
  %bn = load i64, ptr %slot, align 8
  %pb = call ptr @thrift.take(ptr %r, i64 %bn)
  %pbn = icmp eq ptr %pb, null
  br i1 %pbn, label %err.bad, label %noop, !prof !0
ret.rcl:
  ret i32 %rcl

; ---- list / set: header, then size elements of the element type ----
skip.list:
  %rch = call i32 @universe_encoding_thrift_collection(ptr %r, ptr %et8, ptr %sz32)
  %rchok = icmp eq i32 %rch, 0
  br i1 %rchok, label %list.setup, label %ret.rch, !prof !0
list.setup:
  %etl = load i8, ptr %et8, align 1
  %etl32 = zext i8 %etl to i32
  %szl = load i32, ptr %sz32, align 4
  %szlc = zext i32 %szl to i64
  %etl.b1 = icmp eq i32 %etl32, 1
  %etl.b2 = icmp eq i32 %etl32, 2
  %etl.isbool = or i1 %etl.b1, %etl.b2
  br label %list.head
list.head:
  %li = phi i64 [ 0, %list.setup ], [ %li.n, %list.cont ]
  %ldone = icmp uge i64 %li, %szlc
  br i1 %ldone, label %noop, label %list.body
list.body:
  br i1 %etl.isbool, label %list.bool, label %list.skip, !prof !0
list.bool:
  %lbb = call i64 @thrift.byte(ptr %r)
  %lbbt = icmp slt i64 %lbb, 0
  br i1 %lbbt, label %err.bad, label %list.cont, !prof !0
list.skip:
  %lrc = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %etl32, i32 %d1)
  %lrcok = icmp eq i32 %lrc, 0
  br i1 %lrcok, label %list.cont, label %ret.lrc, !prof !0
list.cont:
  %li.n = add nuw i64 %li, 1
  br label %list.head
ret.lrc:
  ret i32 %lrc
ret.rch:
  ret i32 %rch

; ---- map: varint size, then a key/value type byte, then size*(k,v) ----
skip.map:
  %rcm = call i32 @universe_encoding_thrift_read_uvarint(ptr %r, ptr %slot)
  %rcmok = icmp eq i32 %rcm, 0
  br i1 %rcmok, label %map.chk, label %ret.rcm, !prof !0
map.chk:
  %msz = load i64, ptr %slot, align 8
  %msz0 = icmp eq i64 %msz, 0
  br i1 %msz0, label %noop, label %map.kv, !prof !0
map.kv:
  %kvb = call i64 @thrift.byte(ptr %r)
  %kvt = icmp slt i64 %kvb, 0
  br i1 %kvt, label %err.bad, label %map.types, !prof !0
map.types:
  %ktype = lshr i64 %kvb, 4
  %ktype32 = trunc i64 %ktype to i32
  %vtype = and i64 %kvb, 15
  %vtype32 = trunc i64 %vtype to i32
  br label %map.head
map.head:
  %mi = phi i64 [ 0, %map.types ], [ %mi.n, %map.vdone ]
  %mdone = icmp uge i64 %mi, %msz
  br i1 %mdone, label %noop, label %map.key
map.key:
  %krc = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ktype32, i32 %d1)
  %krcok = icmp eq i32 %krc, 0
  br i1 %krcok, label %map.val, label %ret.krc, !prof !0
map.val:
  %vrc = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %vtype32, i32 %d1)
  %vrcok = icmp eq i32 %vrc, 0
  br i1 %vrcok, label %map.vdone, label %ret.vrc, !prof !0
map.vdone:
  %mi.n = add nuw i64 %mi, 1
  br label %map.head
ret.krc:
  ret i32 %krc
ret.vrc:
  ret i32 %vrc
ret.rcm:
  ret i32 %rcm

; ---- struct: save/reset field-id base, read fields until STOP ----
skip.struct:
  %lfp = getelementptr inbounds nuw i8, ptr %r, i64 24
  %prev = load i16, ptr %lfp, align 2
  store i16 0, ptr %lfp, align 2
  br label %struct.head
struct.head:
  %frc = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %frcok = icmp eq i32 %frc, 0
  br i1 %frcok, label %struct.chk, label %struct.err, !prof !0
struct.chk:
  %fct = load i8, ptr %ct8, align 1
  %fstop = icmp eq i8 %fct, 0
  br i1 %fstop, label %struct.done, label %struct.skip, !prof !0
struct.skip:
  %fct32 = zext i8 %fct to i32
  %src = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %fct32, i32 %d1)
  %srcok = icmp eq i32 %src, 0
  br i1 %srcok, label %struct.head, label %struct.err2, !prof !0
struct.done:
  store i16 %prev, ptr %lfp, align 2
  ret i32 0
struct.err:                                       ; cold: propagate field error
  ret i32 %frc
struct.err2:                                      ; cold: propagate skip error
  ret i32 %src

err.null:                                         ; cold
  ret i32 1
err.bad:                                          ; cold
  ret i32 8
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #3 = { nounwind nosync nofree memory(argmem: readwrite) }

!0 = !{!"branch_weights", i32 1, i32 2000}
