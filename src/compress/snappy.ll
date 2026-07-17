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

; Snappy raw block codec: block decompress + a single-pass greedy compressor.
; This is the framed Snappy that Parquet's SNAPPY codec uses — a varint
; decompressed-length preamble followed by a stream of literal/copy elements —
; NOT the stream/CRC framing.
;
; DESIGN:
;   * ONE-SHOT, whole-buffer. The whole compressed block sits in a caller
;     buffer; the whole decoded output goes to a caller buffer with an explicit
;     cap. No streaming, no per-item allocation (the compute/memory/IO split:
;     the caller owns IO, this code is pure compute over memory it owns). The
;     only heap use is a single scratch hash table in the COMPRESSOR (freed
;     before return).
;   * BLOCK FORMAT. A block starts with the decompressed length as a base-128
;     little-endian varint (meaningful up to 32 bits). Then a stream of
;     elements, each led by a tag byte whose low 2 bits select the kind:
;       00 literal   — (len-1) in the top 6 bits of the tag when that value is
;                      0..59; when it is 60..63 the top bits give the count of
;                      trailing little-endian bytes (1..4) that hold (len-1).
;                      Then `len` verbatim bytes.
;       01 copy1     — len = ((tag>>2)&7)+4  (4..11); 11-bit offset built from
;                      (tag>>5) as the high 3 bits and one trailing byte as low 8.
;       10 copy2     — len = (tag>>2)+1  (1..64); 16-bit LE offset (2 bytes).
;       11 copy4     — len = (tag>>2)+1  (1..64); 32-bit LE offset (4 bytes).
;   * DECODE (@universe_compress_snappy_decode) parses the varint, rejects a
;     declared length that exceeds the output cap (FULL) or 32 bits (PARSE),
;     then replays elements left to right. EVERY literal read is bounds-checked
;     against slen and EVERY write against the declared length (which is <= cap)
;     BEFORE it happens, so a truncated or malformed block returns a negative
;     error and never reads or writes out of bounds (the property AddressSanitizer
;     verifies on hostile input). A copy offset must be >= 1 and <= the bytes
;     already emitted. Copy is OVERLAP-CORRECT: offset >= len cannot overlap so
;     we emit one llvm.memcpy; offset < len (RLE run, e.g. offset==1) is copied
;     forward byte-by-byte so freshly written bytes feed the copy. At the end
;     the produced length must equal the declared length, else PARSE.
;   * ENCODE (@universe_compress_snappy_encode) writes the varint preamble then
;     runs a single greedy pass with a 4-byte-hash match table (2^16 entries of
;     i32, storing position+1 so 0 is "empty"). At each position we hash the next
;     4 bytes, look up the most recent same-hash position, and if it is within a
;     64 KiB window and the 4 bytes truly match we extend the match forward, emit
;     the pending literals, then emit the match as 2-byte-offset copies (chunked
;     to the 64-byte copy maximum); otherwise we advance one byte. The tail is
;     always emitted as literals. Every block it produces is spec-valid and the
;     reference Snappy decoder accepts it. Inputs < 4 bytes are a single literal.
;   * Error convention (i64 return): decoded/compressed length on success,
;     NEGATIVE on failure: -1 NULL, -2 OOM (table alloc), -6 output exceeds cap
;     (FULL), -13 malformed block (PARSE), -15 truncated input (IO).
;
; API:
;   i64 universe_compress_snappy_decode(ptr dst, i64 dcap, ptr src, i64 slen)
;   i64 universe_compress_snappy_encode(ptr dst, i64 dcap, ptr src, i64 slen)
;   i64 universe_compress_snappy_bound(i64 slen)   ; worst-case compressed size

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

; --------------------------------------------------------------- decode
define i64 @universe_compress_snappy_decode(ptr %dst, i64 %dcap, ptr %src, i64 %slen) #0 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %ret_null, label %chkempty

ret_null:
  ret i64 -1

chkempty:
  ; a valid block always carries at least the varint preamble byte
  %empty = icmp eq i64 %slen, 0
  br i1 %empty, label %err_trunc, label %vh

; ---- varint preamble: base-128 little-endian decompressed length ----
vh:
  %vip = phi i64 [ 0, %chkempty ], [ %vip1, %vcont ]
  %vval = phi i64 [ 0, %chkempty ], [ %vval.n, %vcont ]
  %vshift = phi i64 [ 0, %chkempty ], [ %vshift.n, %vcont ]
  %vavail = icmp ult i64 %vip, %slen
  br i1 %vavail, label %vbody, label %err_trunc

vbody:
  %vbp = getelementptr inbounds i8, ptr %src, i64 %vip
  %vb8 = load i8, ptr %vbp, align 1
  %vb = zext i8 %vb8 to i64
  %vip1 = add i64 %vip, 1
  %vlow = and i64 %vb, 127
  %vpiece = shl i64 %vlow, %vshift
  %vval.n = or i64 %vval, %vpiece
  %vhigh = and i64 %vb, 128
  %vmore = icmp ne i64 %vhigh, 0
  br i1 %vmore, label %vcont, label %chk_declared

vcont:
  %vshift.n = add i64 %vshift, 7
  ; after 5 bytes shift reaches 35 (> 28); a 6th continuation byte is malformed
  %vtoobig = icmp ugt i64 %vshift.n, 28
  br i1 %vtoobig, label %err_parse, label %vh

chk_declared:
  ; declared length must be representable in 32 bits
  %vtoolarge = icmp ugt i64 %vval.n, 4294967295
  br i1 %vtoolarge, label %err_parse, label %chk_cap

chk_cap:
  %overcap = icmp ugt i64 %vval.n, %dcap
  br i1 %overcap, label %err_full, label %elem.head

; ---- element stream ----
elem.head:
  %ip = phi i64 [ %vip1, %chk_cap ], [ %srcend, %lit.do ], [ %ipC, %copy.next ]
  %op = phi i64 [ 0, %chk_cap ], [ %dstend, %lit.do ], [ %op.cp, %copy.next ]
  %atend = icmp uge i64 %ip, %slen
  br i1 %atend, label %ret_check, label %read.tag

ret_check:
  %exact = icmp eq i64 %op, %vval.n
  br i1 %exact, label %ret_op, label %err_parse

ret_op:
  ret i64 %op

read.tag:
  %tp = getelementptr inbounds i8, ptr %src, i64 %ip
  %tag8 = load i8, ptr %tp, align 1
  %tag = zext i8 %tag8 to i64
  %ip.t = add i64 %ip, 1
  %kind = and i64 %tag, 3
  switch i64 %kind, label %err_parse [
    i64 0, label %lit
    i64 1, label %copy1
    i64 2, label %copy2
    i64 3, label %copy4
  ]

; ---- literal ----
lit:
  %t6 = lshr i64 %tag, 2
  %litbig = icmp uge i64 %t6, 60
  br i1 %litbig, label %lit.ext, label %lit.small

lit.small:
  %litlen.s = add i64 %t6, 1
  br label %lit.copy

lit.ext:
  %nb = sub i64 %t6, 59
  %extend = add i64 %ip.t, %nb
  %extshort = icmp ugt i64 %extend, %slen
  br i1 %extshort, label %err_trunc, label %le.head

le.head:
  %lek = phi i64 [ 0, %lit.ext ], [ %lek.n, %le.body ]
  %lev = phi i64 [ 0, %lit.ext ], [ %lev.n, %le.body ]
  %lekdone = icmp uge i64 %lek, %nb
  br i1 %lekdone, label %le.done, label %le.body

le.body:
  %leoff = add i64 %ip.t, %lek
  %lebp = getelementptr inbounds i8, ptr %src, i64 %leoff
  %leb8 = load i8, ptr %lebp, align 1
  %leb = zext i8 %leb8 to i64
  %lesh = shl i64 %lek, 3
  %lepiece = shl i64 %leb, %lesh
  %lev.n = or i64 %lev, %lepiece
  %lek.n = add i64 %lek, 1
  br label %le.head

le.done:
  ; lev holds (litlen-1); reject values that would overflow 32-bit length
  %levbig = icmp ugt i64 %lev, 4294967294
  br i1 %levbig, label %err_parse, label %lit.ext.ok

lit.ext.ok:
  %litlen.e = add i64 %lev, 1
  br label %lit.copy

lit.copy:
  %litlen = phi i64 [ %litlen.s, %lit.small ], [ %litlen.e, %lit.ext.ok ]
  %ipL = phi i64 [ %ip.t, %lit.small ], [ %extend, %lit.ext.ok ]
  %srcend = add i64 %ipL, %litlen
  %litoosrc = icmp ugt i64 %srcend, %slen
  br i1 %litoosrc, label %err_trunc, label %lit.dst

lit.dst:
  %dstend = add i64 %op, %litlen
  %litoodst = icmp ugt i64 %dstend, %vval.n
  br i1 %litoodst, label %err_parse, label %lit.do

lit.do:
  %ldst = getelementptr inbounds i8, ptr %dst, i64 %op
  %lsrc = getelementptr inbounds i8, ptr %src, i64 %ipL
  call void @llvm.memcpy.p0.p0.i64(ptr %ldst, ptr %lsrc, i64 %litlen, i1 false)
  br label %elem.head

; ---- copy with 1-byte offset ----
copy1:
  %c1short = icmp uge i64 %ip.t, %slen
  br i1 %c1short, label %err_trunc, label %copy1.go

copy1.go:
  %c1l0 = lshr i64 %tag, 2
  %c1l1 = and i64 %c1l0, 7
  %len1 = add i64 %c1l1, 4
  %c1bp = getelementptr inbounds i8, ptr %src, i64 %ip.t
  %c1b8 = load i8, ptr %c1bp, align 1
  %c1b = zext i8 %c1b8 to i64
  %c1hi = lshr i64 %tag, 5
  %c1hi8 = shl i64 %c1hi, 8
  %off1 = or i64 %c1hi8, %c1b
  %ipC1 = add i64 %ip.t, 1
  br label %copy.do

; ---- copy with 2-byte offset ----
copy2:
  %c2end = add i64 %ip.t, 2
  %c2short = icmp ugt i64 %c2end, %slen
  br i1 %c2short, label %err_trunc, label %copy2.go

copy2.go:
  %c2t = lshr i64 %tag, 2
  %len2 = add i64 %c2t, 1
  %c2p0 = getelementptr inbounds i8, ptr %src, i64 %ip.t
  %c2b0v = load i8, ptr %c2p0, align 1
  %c2b0 = zext i8 %c2b0v to i64
  %c2i1 = add i64 %ip.t, 1
  %c2p1 = getelementptr inbounds i8, ptr %src, i64 %c2i1
  %c2b1v = load i8, ptr %c2p1, align 1
  %c2b1 = zext i8 %c2b1v to i64
  %c2b1s = shl i64 %c2b1, 8
  %off2 = or i64 %c2b0, %c2b1s
  br label %copy.do

; ---- copy with 4-byte offset ----
copy4:
  %c4end = add i64 %ip.t, 4
  %c4short = icmp ugt i64 %c4end, %slen
  br i1 %c4short, label %err_trunc, label %copy4.go

copy4.go:
  %c4t = lshr i64 %tag, 2
  %len4 = add i64 %c4t, 1
  %c4p0 = getelementptr inbounds i8, ptr %src, i64 %ip.t
  %c4b0v = load i8, ptr %c4p0, align 1
  %c4b0 = zext i8 %c4b0v to i64
  %c4i1 = add i64 %ip.t, 1
  %c4p1 = getelementptr inbounds i8, ptr %src, i64 %c4i1
  %c4b1v = load i8, ptr %c4p1, align 1
  %c4b1 = zext i8 %c4b1v to i64
  %c4b1s = shl i64 %c4b1, 8
  %c4i2 = add i64 %ip.t, 2
  %c4p2 = getelementptr inbounds i8, ptr %src, i64 %c4i2
  %c4b2v = load i8, ptr %c4p2, align 1
  %c4b2 = zext i8 %c4b2v to i64
  %c4b2s = shl i64 %c4b2, 16
  %c4i3 = add i64 %ip.t, 3
  %c4p3 = getelementptr inbounds i8, ptr %src, i64 %c4i3
  %c4b3v = load i8, ptr %c4p3, align 1
  %c4b3 = zext i8 %c4b3v to i64
  %c4b3s = shl i64 %c4b3, 24
  %c4o01 = or i64 %c4b0, %c4b1s
  %c4o23 = or i64 %c4b2s, %c4b3s
  %off4 = or i64 %c4o01, %c4o23
  br label %copy.do

; ---- shared copy execution ----
copy.do:
  %clen = phi i64 [ %len1, %copy1.go ], [ %len2, %copy2.go ], [ %len4, %copy4.go ]
  %coff = phi i64 [ %off1, %copy1.go ], [ %off2, %copy2.go ], [ %off4, %copy4.go ]
  %ipC = phi i64 [ %ipC1, %copy1.go ], [ %c2end, %copy2.go ], [ %c4end, %copy4.go ]
  %offzero = icmp eq i64 %coff, 0
  br i1 %offzero, label %err_parse, label %copy.chk

copy.chk:
  %offbad = icmp ugt i64 %coff, %op
  br i1 %offbad, label %err_parse, label %copy.room

copy.room:
  %cdstend = add i64 %op, %clen
  %coodst = icmp ugt i64 %cdstend, %vval.n
  br i1 %coodst, label %err_parse, label %copy.exec

copy.exec:
  %cfrom = sub i64 %op, %coff
  %nonoverlap = icmp uge i64 %coff, %clen
  br i1 %nonoverlap, label %c.fast, label %c.byte

c.fast:
  %cfsrc = getelementptr inbounds i8, ptr %dst, i64 %cfrom
  %cfdst = getelementptr inbounds i8, ptr %dst, i64 %op
  call void @llvm.memcpy.p0.p0.i64(ptr %cfdst, ptr %cfsrc, i64 %clen, i1 false)
  br label %copy.next

c.byte:
  br label %cb.loop

cb.loop:
  %ck = phi i64 [ 0, %c.byte ], [ %ck.n, %cb.loop ]
  %ckf = add i64 %cfrom, %ck
  %ckt = add i64 %op, %ck
  %cksp = getelementptr inbounds i8, ptr %dst, i64 %ckf
  %cksv = load i8, ptr %cksp, align 1
  %ckdp = getelementptr inbounds i8, ptr %dst, i64 %ckt
  store i8 %cksv, ptr %ckdp, align 1
  %ck.n = add i64 %ck, 1
  %ckmore = icmp ult i64 %ck.n, %clen
  br i1 %ckmore, label %cb.loop, label %copy.next

copy.next:
  %op.cp = add i64 %op, %clen
  br label %elem.head

err_trunc:
  ret i64 -15

err_full:
  ret i64 -6

err_parse:
  ret i64 -13
}

; --------------------------------------------------------------- bound
; Worst-case compressed size: preamble varint (<=5) plus the reference
; slack of slen + slen/6, rounded up with a fixed 32-byte cushion.
define i64 @universe_compress_snappy_bound(i64 %slen) #2 {
entry:
  %d6 = udiv i64 %slen, 6
  %t = add i64 %slen, %d6
  %r = add i64 %t, 32
  ret i64 %r
}

; ------------------------------------------------ varint preamble writer
; Append a base-128 little-endian varint; returns new op, or -6 on FULL.
define internal i64 @snappy_put_varint(ptr %dst, i64 %dcap, i64 %op, i64 %val) #1 {
entry:
  br label %loop

loop:
  %v = phi i64 [ %val, %entry ], [ %v.n, %cont ]
  %o = phi i64 [ %op, %entry ], [ %o1, %cont ]
  %big = icmp uge i64 %v, 128
  br i1 %big, label %chk, label %last

chk:
  %room = icmp ult i64 %o, %dcap
  br i1 %room, label %cont, label %full

cont:
  %lo = and i64 %v, 127
  %byte = or i64 %lo, 128
  %bp = getelementptr inbounds i8, ptr %dst, i64 %o
  %b8 = trunc i64 %byte to i8
  store i8 %b8, ptr %bp, align 1
  %o1 = add i64 %o, 1
  %v.n = lshr i64 %v, 7
  br label %loop

last:
  %room2 = icmp ult i64 %o, %dcap
  br i1 %room2, label %dolast, label %full

dolast:
  %lp = getelementptr inbounds i8, ptr %dst, i64 %o
  %lv8 = trunc i64 %v to i8
  store i8 %lv8, ptr %lp, align 1
  %o2 = add i64 %o, 1
  ret i64 %o2

full:
  ret i64 -6
}

; ------------------------------------------------ literal emitter
; Emit a literal element (tag + optional extended length + raw bytes).
; Returns new op, or -6 on FULL. count == 0 is a no-op.
define internal i64 @snappy_emit_lit(ptr %dst, i64 %dcap, i64 %op, ptr %src,
                                     i64 %from, i64 %count) #1 {
entry:
  %zero = icmp eq i64 %count, 0
  br i1 %zero, label %ret.op, label %hdr

ret.op:
  ret i64 %op

hdr:
  %ln = sub i64 %count, 1
  %small = icmp ult i64 %ln, 60
  br i1 %small, label %wr.small, label %wr.big

wr.small:
  %need1 = add i64 %op, 1
  %nr1 = icmp ugt i64 %need1, %dcap
  br i1 %nr1, label %full, label %do.small

do.small:
  %tags = shl i64 %ln, 2
  %tsp = getelementptr inbounds i8, ptr %dst, i64 %op
  %ts8 = trunc i64 %tags to i8
  store i8 %ts8, ptr %tsp, align 1
  %op.s = add i64 %op, 1
  br label %copy.pre

wr.big:
  %isb2 = icmp uge i64 %ln, 256
  %isb3 = icmp uge i64 %ln, 65536
  %isb4 = icmp uge i64 %ln, 16777216
  %n1 = zext i1 %isb2 to i64
  %n2 = zext i1 %isb3 to i64
  %n3 = zext i1 %isb4 to i64
  %nb0 = add i64 1, %n1
  %nb1 = add i64 %nb0, %n2
  %nb = add i64 %nb1, %n3
  %tv0 = add i64 59, %nb
  %tv = shl i64 %tv0, 2
  %needb0 = add i64 %op, 1
  %needb = add i64 %needb0, %nb
  %nrb = icmp ugt i64 %needb, %dcap
  br i1 %nrb, label %full, label %do.big

do.big:
  %tvp = getelementptr inbounds i8, ptr %dst, i64 %op
  %tv8 = trunc i64 %tv to i8
  store i8 %tv8, ptr %tvp, align 1
  %op.b0 = add i64 %op, 1
  br label %ble.head

ble.head:
  %bk = phi i64 [ 0, %do.big ], [ %bk.n, %ble.body ]
  %bkdone = icmp uge i64 %bk, %nb
  br i1 %bkdone, label %ble.done, label %ble.body

ble.body:
  %bsh = shl i64 %bk, 3
  %bval = lshr i64 %ln, %bsh
  %bpos = add i64 %op.b0, %bk
  %bdp = getelementptr inbounds i8, ptr %dst, i64 %bpos
  %bv8 = trunc i64 %bval to i8
  store i8 %bv8, ptr %bdp, align 1
  %bk.n = add i64 %bk, 1
  br label %ble.head

ble.done:
  %op.b = add i64 %op.b0, %nb
  br label %copy.pre

copy.pre:
  %opc = phi i64 [ %op.s, %do.small ], [ %op.b, %ble.done ]
  %needc = add i64 %opc, %count
  %nrc = icmp ugt i64 %needc, %dcap
  br i1 %nrc, label %full, label %do.copy

do.copy:
  %cd = getelementptr inbounds i8, ptr %dst, i64 %opc
  %cs = getelementptr inbounds i8, ptr %src, i64 %from
  call void @llvm.memcpy.p0.p0.i64(ptr %cd, ptr %cs, i64 %count, i1 false)
  %opf = add i64 %opc, %count
  ret i64 %opf

full:
  ret i64 -6
}

; ------------------------------------------------ copy emitter
; Emit a match of `len` bytes at distance `dist` (1..65535) as 2-byte-offset
; copies, chunked to the 64-byte copy maximum. Returns new op, or -6 on FULL.
define internal i64 @snappy_emit_copy(ptr %dst, i64 %dcap, i64 %op, i64 %dist, i64 %len) #1 {
entry:
  br label %loop

loop:
  %o = phi i64 [ %op, %entry ], [ %o.n, %wr ]
  %rem = phi i64 [ %len, %entry ], [ %rem.n, %wr ]
  %done = icmp eq i64 %rem, 0
  br i1 %done, label %ret.ok, label %pick

ret.ok:
  ret i64 %o

pick:
  %big = icmp ugt i64 %rem, 64
  %chunk = select i1 %big, i64 64, i64 %rem
  %need = add i64 %o, 3
  %noroom = icmp ugt i64 %need, %dcap
  br i1 %noroom, label %full, label %wr

wr:
  %cm1 = sub i64 %chunk, 1
  %csh = shl i64 %cm1, 2
  %tag = or i64 %csh, 2
  %tp = getelementptr inbounds i8, ptr %dst, i64 %o
  %t8 = trunc i64 %tag to i8
  store i8 %t8, ptr %tp, align 1
  %o1 = add i64 %o, 1
  %lop = getelementptr inbounds i8, ptr %dst, i64 %o1
  %lo8 = trunc i64 %dist to i8
  store i8 %lo8, ptr %lop, align 1
  %o2 = add i64 %o, 2
  %hip = getelementptr inbounds i8, ptr %dst, i64 %o2
  %hi64 = lshr i64 %dist, 8
  %hi8 = trunc i64 %hi64 to i8
  store i8 %hi8, ptr %hip, align 1
  %o.n = add i64 %o, 3
  %rem.n = sub i64 %rem, %chunk
  br label %loop

full:
  ret i64 -6
}

; --------------------------------------------------------------- encode
define i64 @universe_compress_snappy_encode(ptr %dst, i64 %dcap, ptr %src, i64 %slen) #0 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %ret_null, label %pre

ret_null:
  ret i64 -1

pre:
  %op0 = call i64 @snappy_put_varint(ptr %dst, i64 %dcap, i64 0, i64 %slen)
  %pverr = icmp slt i64 %op0, 0
  br i1 %pverr, label %ret_full, label %chk0

ret_full:
  ret i64 -6

chk0:
  %empty = icmp eq i64 %slen, 0
  br i1 %empty, label %ret_op0, label %chk4

ret_op0:
  ret i64 %op0

chk4:
  %tooshort = icmp ult i64 %slen, 4
  br i1 %tooshort, label %only.lit, label %setup

only.lit:
  %olr = call i64 @snappy_emit_lit(ptr %dst, i64 %dcap, i64 %op0, ptr %src, i64 0, i64 %slen)
  ret i64 %olr

setup:
  %tbl = call ptr @malloc(i64 262144)
  %tn = icmp eq ptr %tbl, null
  br i1 %tn, label %ret_oom, label %setup2

ret_oom:
  ret i64 -2

setup2:
  call void @llvm.memset.p0.i64(ptr %tbl, i8 0, i64 262144, i1 false)
  %hlimit = sub i64 %slen, 4
  br label %scan.head

scan.head:
  %ip = phi i64 [ 0, %setup2 ], [ %ip.adv, %no.match ], [ %ip.after, %emit.done ]
  %op = phi i64 [ %op0, %setup2 ], [ %op, %no.match ], [ %opC, %emit.done ]
  %anchor = phi i64 [ 0, %setup2 ], [ %anchor, %no.match ], [ %ip.after, %emit.done ]
  %go = icmp ule i64 %ip, %hlimit
  br i1 %go, label %scan.find, label %final

scan.find:
  %s4p = getelementptr inbounds i8, ptr %src, i64 %ip
  %seq = load i32, ptr %s4p, align 1
  %mul = mul i32 %seq, -1640531535
  %hidx32 = lshr i32 %mul, 16
  %hidx = zext i32 %hidx32 to i64
  %tep = getelementptr inbounds i32, ptr %tbl, i64 %hidx
  %cand = load i32, ptr %tep, align 4
  %ip32 = trunc i64 %ip to i32
  %store = add i32 %ip32, 1
  store i32 %store, ptr %tep, align 4
  %isempty = icmp eq i32 %cand, 0
  br i1 %isempty, label %no.match, label %cand.chk

cand.chk:
  %refpos32 = sub i32 %cand, 1
  %refpos = zext i32 %refpos32 to i64
  %dist = sub i64 %ip, %refpos
  %toofar = icmp ugt i64 %dist, 65535
  br i1 %toofar, label %no.match, label %cmp4

cmp4:
  %rp = getelementptr inbounds i8, ptr %src, i64 %refpos
  %seqref = load i32, ptr %rp, align 1
  %match4 = icmp eq i32 %seq, %seqref
  br i1 %match4, label %ext.head, label %no.match

no.match:
  %ip.adv = add i64 %ip, 1
  br label %scan.head

ext.head:
  %ml = phi i64 [ 4, %cmp4 ], [ %ml.n, %ext.body ]
  %ipm = add i64 %ip, %ml
  %inb = icmp ult i64 %ipm, %slen
  br i1 %inb, label %ext.cmp, label %ext.done

ext.cmp:
  %refm = add i64 %refpos, %ml
  %ap = getelementptr inbounds i8, ptr %src, i64 %ipm
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %src, i64 %refm
  %bv = load i8, ptr %bp, align 1
  %beq = icmp eq i8 %av, %bv
  br i1 %beq, label %ext.body, label %ext.done

ext.body:
  %ml.n = add i64 %ml, 1
  br label %ext.head

ext.done:
  %litcount = sub i64 %ip, %anchor
  %opL = call i64 @snappy_emit_lit(ptr %dst, i64 %dcap, i64 %op, ptr %src, i64 %anchor, i64 %litcount)
  %lerr = icmp slt i64 %opL, 0
  br i1 %lerr, label %ret_full_free, label %emit.copy

emit.copy:
  %opC = call i64 @snappy_emit_copy(ptr %dst, i64 %dcap, i64 %opL, i64 %dist, i64 %ml)
  %cerr = icmp slt i64 %opC, 0
  br i1 %cerr, label %ret_full_free, label %emit.done

emit.done:
  %ip.after = add i64 %ip, %ml
  br label %scan.head

final:
  %fcount = sub i64 %slen, %anchor
  %opF = call i64 @snappy_emit_lit(ptr %dst, i64 %dcap, i64 %op, ptr %src, i64 %anchor, i64 %fcount)
  call void @free(ptr %tbl)
  ret i64 %opF

ret_full_free:
  call void @free(ptr %tbl)
  ret i64 -6
}

attributes #0 = { nounwind }
attributes #1 = { nounwind alwaysinline }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }
