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

; DEFLATE / zlib / gzip decoder (RFC 1951 / 1950 / 1952) + adler32 / crc32,
; plus a stored-block and fixed-Huffman DEFLATE encoder for round-trip tests.
;
; DESIGN:
;   * ONE-SHOT, whole-buffer decode. The entire compressed input sits in a
;     caller buffer and the entire output goes to a caller buffer with an
;     explicit size cap. No streaming, no allocation: every scratch table
;     (Huffman counts/symbols, code-length work) lives in stack allocas sized
;     to the RFC maxima (litlen 288, dist 32, code-length 19). This matches
;     the compute/memory/IO split: the caller does IO (fills the input, drains
;     the output); this code is pure compute over memory it already owns.
;   * BIT READER (@z_getbits) is LSB-first per RFC 1951 3.1.1. State is a small
;     stack struct {pos,len,src,bitbuf,bitcnt,err}. Bits accumulate low-to-high
;     in a 64-bit register; we only ever request <=16 bits at once, and refill
;     one byte at a time while bitcnt<need, so every shift amount is provably
;     < 64 (hazard #10: a shift >= width is poison). On a truncated stream the
;     reader sets err and returns -1; since the returned value is always masked
;     to >= 0, the -1 sentinel is unambiguous and every caller checks slt 0.
;   * CANONICAL HUFFMAN DECODE (@z_decode) is the count/first/index walk derived
;     directly from the RFC's code-construction rule: read one bit at a time,
;     maintain the first code value of the current length and the running index
;     into a length-sorted symbol table. No lookup table to build/size; correct
;     for incomplete codes; O(code length) per symbol (<=15 bits).
;   * @z_build turns a per-symbol length array into {count[16], symbol[]} using
;     a bucket-offset pass (offs[len] = start of that length's symbols).
;   * LZ77 back-reference copy is OVERLAP-CORRECT: when dist>=len the two ranges
;     cannot overlap so we emit one llvm.memcpy; when dist<len (RLE-style run,
;     e.g. dist==1) we MUST copy forward byte-by-byte so freshly written bytes
;     feed the copy — a memmove would give wrong LZ77 semantics.
;   * Error convention (i64 return): decoded length on success, NEGATIVE on
;     failure: -1 NULL, -6 output would exceed cap (FULL), -13 malformed stream
;     / checksum mismatch (PARSE), -14 unsupported wrapper, -15 truncated (IO).
;
; API (raw DEFLATE + wrappers + checksums + encoders):
;   i64 universe_compress_inflate(ptr dst, i64 dstcap, ptr src, i64 srclen)
;   i64 universe_compress_inflate_zlib(ptr dst, i64 dstcap, ptr src, i64 srclen)
;   i64 universe_compress_inflate_gzip(ptr dst, i64 dstcap, ptr src, i64 srclen)
;   i64 universe_compress_adler32(ptr data, i64 len)     ; 32-bit value in i64
;   i64 universe_compress_crc32(ptr data, i64 len)       ; 32-bit value in i64
;   i64 universe_compress_deflate_stored(ptr dst, i64 dstcap, ptr src, i64 n)
;   i64 universe_compress_deflate_fixed(ptr dst, i64 dstcap, ptr src, i64 n)

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

; ---- RFC 1951 length/distance base+extra tables -------------------------------
; length codes 257..285  (index = symbol-257)
@z.len_base  = internal constant [29 x i16]
  [i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 13,
   i16 15, i16 17, i16 19, i16 23, i16 27, i16 31, i16 35, i16 43, i16 51,
   i16 59, i16 67, i16 83, i16 99, i16 115, i16 131, i16 163, i16 195, i16 227,
   i16 258]
@z.len_extra = internal constant [29 x i8]
  [i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 1, i8 1, i8 1, i8 1,
   i8 2, i8 2, i8 2, i8 2, i8 3, i8 3, i8 3, i8 3, i8 4, i8 4, i8 4, i8 4,
   i8 5, i8 5, i8 5, i8 5, i8 0]
; distance codes 0..29
@z.dist_base = internal constant [30 x i32]
  [i32 1, i32 2, i32 3, i32 4, i32 5, i32 7, i32 9, i32 13, i32 17, i32 25,
   i32 33, i32 49, i32 65, i32 97, i32 129, i32 193, i32 257, i32 385, i32 513,
   i32 769, i32 1025, i32 1537, i32 2049, i32 3073, i32 4097, i32 6145,
   i32 8193, i32 12289, i32 16385, i32 24577]
@z.dist_extra = internal constant [30 x i8]
  [i8 0, i8 0, i8 0, i8 0, i8 1, i8 1, i8 2, i8 2, i8 3, i8 3, i8 4, i8 4,
   i8 5, i8 5, i8 6, i8 6, i8 7, i8 7, i8 8, i8 8, i8 9, i8 9, i8 10, i8 10,
   i8 11, i8 11, i8 12, i8 12, i8 13, i8 13]
; code-length alphabet permutation (RFC 1951 3.2.7)
@z.clc_order = internal constant [19 x i8]
  [i8 16, i8 17, i8 18, i8 0, i8 8, i8 7, i8 9, i8 6, i8 10, i8 5, i8 11, i8 4,
   i8 12, i8 3, i8 13, i8 2, i8 14, i8 1, i8 15]

; ------------------------------------------------------------------- bit reader
; State: pos@0(i64) len@8(i64) src@16(ptr) bitbuf@24(i64) bitcnt@32(i64) err@40(i32)
define internal i64 @z_getbits(ptr %st, i32 %n) #0 {
entry:
  %n64 = zext i32 %n to i64
  br label %rf.head

rf.head:
  %cntp = getelementptr inbounds i8, ptr %st, i64 32
  %cnt = load i64, ptr %cntp, align 8
  %need = icmp ult i64 %cnt, %n64
  br i1 %need, label %rf.chk, label %ready

rf.chk:
  %posp = getelementptr inbounds i8, ptr %st, i64 0
  %pos = load i64, ptr %posp, align 8
  %lenp = getelementptr inbounds i8, ptr %st, i64 8
  %len = load i64, ptr %lenp, align 8
  %avail = icmp ult i64 %pos, %len
  br i1 %avail, label %rf.do, label %trunc

trunc:
  %errp = getelementptr inbounds i8, ptr %st, i64 40
  store i32 15, ptr %errp, align 4
  ret i64 -1

rf.do:
  %srcp = getelementptr inbounds i8, ptr %st, i64 16
  %src = load ptr, ptr %srcp, align 8
  %bytep = getelementptr inbounds i8, ptr %src, i64 %pos
  %byte = load i8, ptr %bytep, align 1
  %bz = zext i8 %byte to i64
  %bufp = getelementptr inbounds i8, ptr %st, i64 24
  %buf = load i64, ptr %bufp, align 8
  %sh = shl i64 %bz, %cnt
  %buf2 = or i64 %buf, %sh
  store i64 %buf2, ptr %bufp, align 8
  %pos2 = add i64 %pos, 1
  store i64 %pos2, ptr %posp, align 8
  %cnt2 = add i64 %cnt, 8
  store i64 %cnt2, ptr %cntp, align 8
  br label %rf.head

ready:
  %isz = icmp eq i32 %n, 0
  br i1 %isz, label %retz, label %doret

retz:
  ret i64 0

doret:
  %bufp2 = getelementptr inbounds i8, ptr %st, i64 24
  %buf3 = load i64, ptr %bufp2, align 8
  %one = shl i64 1, %n64
  %mask = add i64 %one, -1
  %val = and i64 %buf3, %mask
  %buf4 = lshr i64 %buf3, %n64
  store i64 %buf4, ptr %bufp2, align 8
  %cntp2 = getelementptr inbounds i8, ptr %st, i64 32
  %cnt3 = load i64, ptr %cntp2, align 8
  %cnt4 = sub i64 %cnt3, %n64
  store i64 %cnt4, ptr %cntp2, align 8
  ret i64 %val
}

; ---------------------------------------------------- build canonical huffman
; count: [16 x i16], symbol: [n x i16], lengths: [n x i8], n symbols
define internal i32 @z_build(ptr %count, ptr %symbol, ptr %lengths, i32 %n) #1 {
entry:
  %offs = alloca [16 x i16], align 2
  call void @llvm.memset.p0.i64(ptr %count, i8 0, i64 32, i1 false)
  %n64 = zext i32 %n to i64
  %hasn = icmp ugt i64 %n64, 0
  br i1 %hasn, label %count.loop, label %off.pre

count.loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %count.loop ]
  %lp = getelementptr inbounds i8, ptr %lengths, i64 %i
  %lv8 = load i8, ptr %lp, align 1
  %lv = zext i8 %lv8 to i64
  %cp = getelementptr inbounds i16, ptr %count, i64 %lv
  %cv = load i16, ptr %cp, align 2
  %cv1 = add i16 %cv, 1
  store i16 %cv1, ptr %cp, align 2
  %i.n = add i64 %i, 1
  %imore = icmp ult i64 %i.n, %n64
  br i1 %imore, label %count.loop, label %off.pre

off.pre:
  %o1 = getelementptr inbounds [16 x i16], ptr %offs, i64 0, i64 1
  store i16 0, ptr %o1, align 2
  br label %off.loop

off.loop:
  %L = phi i64 [ 1, %off.pre ], [ %L.n, %off.loop ]
  %ocur = getelementptr inbounds [16 x i16], ptr %offs, i64 0, i64 %L
  %ocurv = load i16, ptr %ocur, align 2
  %ccur = getelementptr inbounds i16, ptr %count, i64 %L
  %ccurv = load i16, ptr %ccur, align 2
  %osum = add i16 %ocurv, %ccurv
  %L.n = add i64 %L, 1
  %onext = getelementptr inbounds [16 x i16], ptr %offs, i64 0, i64 %L.n
  store i16 %osum, ptr %onext, align 2
  %omore = icmp ult i64 %L, 14
  br i1 %omore, label %off.loop, label %place.pre

place.pre:
  %hasn2 = icmp ugt i64 %n64, 0
  br i1 %hasn2, label %place.loop, label %done

place.loop:
  %j = phi i64 [ 0, %place.pre ], [ %j.n, %place.cont ]
  %ljp = getelementptr inbounds i8, ptr %lengths, i64 %j
  %lj8 = load i8, ptr %ljp, align 1
  %lj = zext i8 %lj8 to i64
  %nz = icmp ne i64 %lj, 0
  br i1 %nz, label %place.do, label %place.cont

place.do:
  %ojp = getelementptr inbounds [16 x i16], ptr %offs, i64 0, i64 %lj
  %ojv16 = load i16, ptr %ojp, align 2
  %ojv = zext i16 %ojv16 to i64
  %symp = getelementptr inbounds i16, ptr %symbol, i64 %ojv
  %jt = trunc i64 %j to i16
  store i16 %jt, ptr %symp, align 2
  %ojv1 = add i16 %ojv16, 1
  store i16 %ojv1, ptr %ojp, align 2
  br label %place.cont

place.cont:
  %j.n = add i64 %j, 1
  %jmore = icmp ult i64 %j.n, %n64
  br i1 %jmore, label %place.loop, label %done

done:
  ret i32 0
}

; -------------------------------------------------------- decode one symbol
define internal i32 @z_decode(ptr %st, ptr %count, ptr %symbol) #1 {
entry:
  br label %loop

loop:
  %len = phi i64 [ 1, %entry ], [ %len.n, %cont ]
  %code = phi i32 [ 0, %entry ], [ %code.n, %cont ]
  %first = phi i32 [ 0, %entry ], [ %first.n, %cont ]
  %index = phi i32 [ 0, %entry ], [ %index.n, %cont ]
  %bit = call i64 @z_getbits(ptr %st, i32 1)
  %neg = icmp slt i64 %bit, 0
  br i1 %neg, label %err, label %ok

err:
  ret i32 -1

ok:
  %bit32 = trunc i64 %bit to i32
  %code1 = or i32 %code, %bit32
  %cptr = getelementptr inbounds i16, ptr %count, i64 %len
  %cnt16 = load i16, ptr %cptr, align 2
  %cnt = zext i16 %cnt16 to i32
  %diff = sub i32 %code1, %first
  %in = icmp ult i32 %diff, %cnt
  br i1 %in, label %found, label %cont

found:
  %sidx = add i32 %index, %diff
  %sidx64 = zext i32 %sidx to i64
  %sptr = getelementptr inbounds i16, ptr %symbol, i64 %sidx64
  %sym16 = load i16, ptr %sptr, align 2
  %sym = zext i16 %sym16 to i32
  ret i32 %sym

cont:
  %index.n = add i32 %index, %cnt
  %first0 = add i32 %first, %cnt
  %first.n = shl i32 %first0, 1
  %code.n = shl i32 %code1, 1
  %len.n = add i64 %len, 1
  %over = icmp ugt i64 %len.n, 15
  br i1 %over, label %err2, label %loop

err2:
  ret i32 -1
}

; --------------------------------------------- decode one block body (types 1/2)
; returns new output position or a negative error code
define internal i64 @z_block(ptr %st, ptr %lc, ptr %ls, ptr %dc, ptr %ds,
                             ptr %dst, i64 %dstcap, i64 %op0) #1 {
entry:
  br label %bloop

bloop:
  %op = phi i64 [ %op0, %entry ], [ %op.n, %litdone ], [ %op.ac, %copydone ]
  %sym = call i32 @z_decode(ptr %st, ptr %lc, ptr %ls)
  %symneg = icmp slt i32 %sym, 0
  br i1 %symneg, label %errp, label %chk256

errp:
  ret i64 -13

chk256:
  %is256 = icmp eq i32 %sym, 256
  br i1 %is256, label %eob, label %chklit

eob:
  ret i64 %op

chklit:
  %islit = icmp ult i32 %sym, 256
  br i1 %islit, label %lit, label %length

lit:
  %litroom = icmp ult i64 %op, %dstcap
  br i1 %litroom, label %litstore, label %full

full:
  ret i64 -6

litstore:
  %sym8 = trunc i32 %sym to i8
  %dp = getelementptr inbounds i8, ptr %dst, i64 %op
  store i8 %sym8, ptr %dp, align 1
  %op.n = add i64 %op, 1
  br label %litdone

litdone:
  br label %bloop

length:
  %toobig = icmp ugt i32 %sym, 285
  br i1 %toobig, label %errp2, label %lenok

errp2:
  ret i64 -13

lenok:
  %li = sub i32 %sym, 257
  %li64 = zext i32 %li to i64
  %lbp = getelementptr inbounds [29 x i16], ptr @z.len_base, i64 0, i64 %li64
  %lb16 = load i16, ptr %lbp, align 2
  %lbase = zext i16 %lb16 to i64
  %lep = getelementptr inbounds [29 x i8], ptr @z.len_extra, i64 0, i64 %li64
  %le8 = load i8, ptr %lep, align 1
  %le32 = zext i8 %le8 to i32
  %eb = call i64 @z_getbits(ptr %st, i32 %le32)
  %ebneg = icmp slt i64 %eb, 0
  br i1 %ebneg, label %errio, label %lenok2

errio:
  ret i64 -15

lenok2:
  %len = add i64 %lbase, %eb
  %dsym = call i32 @z_decode(ptr %st, ptr %dc, ptr %ds)
  %dneg = icmp slt i32 %dsym, 0
  br i1 %dneg, label %errp3, label %distok

errp3:
  ret i64 -13

distok:
  %dtoobig = icmp ugt i32 %dsym, 29
  br i1 %dtoobig, label %errp3, label %distok2

distok2:
  %dsym64 = zext i32 %dsym to i64
  %dbp = getelementptr inbounds [30 x i32], ptr @z.dist_base, i64 0, i64 %dsym64
  %db32 = load i32, ptr %dbp, align 4
  %dbase = zext i32 %db32 to i64
  %dep = getelementptr inbounds [30 x i8], ptr @z.dist_extra, i64 0, i64 %dsym64
  %de8 = load i8, ptr %dep, align 1
  %de32 = zext i8 %de8 to i32
  %deb = call i64 @z_getbits(ptr %st, i32 %de32)
  %debneg = icmp slt i64 %deb, 0
  br i1 %debneg, label %errio2, label %distok3

errio2:
  ret i64 -15

distok3:
  %dist = add i64 %dbase, %deb
  %distok_r = icmp ule i64 %dist, %op
  br i1 %distok_r, label %roomchk, label %errp4

errp4:
  ret i64 -13

roomchk:
  %end = add i64 %op, %len
  %roomok = icmp ule i64 %end, %dstcap
  br i1 %roomok, label %copy, label %full2

full2:
  ret i64 -6

copy:
  %from = sub i64 %op, %dist
  %nonoverlap = icmp uge i64 %dist, %len
  br i1 %nonoverlap, label %fastcopy, label %bytecopy

fastcopy:
  %srcpp = getelementptr inbounds i8, ptr %dst, i64 %from
  %dstpp = getelementptr inbounds i8, ptr %dst, i64 %op
  call void @llvm.memcpy.p0.p0.i64(ptr %dstpp, ptr %srcpp, i64 %len, i1 false)
  br label %copydone

bytecopy:
  br label %bc.loop

bc.loop:
  %k = phi i64 [ 0, %bytecopy ], [ %k.n, %bc.loop ]
  %sfrom = add i64 %from, %k
  %sdst = add i64 %op, %k
  %scp = getelementptr inbounds i8, ptr %dst, i64 %sfrom
  %sbv = load i8, ptr %scp, align 1
  %dcp = getelementptr inbounds i8, ptr %dst, i64 %sdst
  store i8 %sbv, ptr %dcp, align 1
  %k.n = add i64 %k, 1
  %kmore = icmp ult i64 %k.n, %len
  br i1 %kmore, label %bc.loop, label %copydone

copydone:
  %op.ac = phi i64 [ %end, %fastcopy ], [ %end, %bc.loop ]
  br label %bloop
}

; ------------------------------------------------------------- raw DEFLATE
define i64 @universe_compress_inflate(ptr %dst, i64 %dstcap, ptr %src, i64 %srclen) #2 {
entry:
  %st = alloca [48 x i8], align 8
  %litcount = alloca [16 x i16], align 2
  %litsym = alloca [288 x i16], align 2
  %distcount = alloca [16 x i16], align 2
  %distsym = alloca [32 x i16], align 2
  %clcount = alloca [16 x i16], align 2
  %clsym = alloca [19 x i16], align 2
  %lengths = alloca [320 x i8], align 1
  %cllen = alloca [19 x i8], align 1
  %dnull = icmp eq ptr %dst, null
  %snull = icmp eq ptr %src, null
  %anynull = or i1 %dnull, %snull
  br i1 %anynull, label %ret_null, label %init

ret_null:
  ret i64 -1

init:
  %posp = getelementptr inbounds i8, ptr %st, i64 0
  store i64 0, ptr %posp, align 8
  %lenp = getelementptr inbounds i8, ptr %st, i64 8
  store i64 %srclen, ptr %lenp, align 8
  %srcp = getelementptr inbounds i8, ptr %st, i64 16
  store ptr %src, ptr %srcp, align 8
  %bufp = getelementptr inbounds i8, ptr %st, i64 24
  store i64 0, ptr %bufp, align 8
  %cntp = getelementptr inbounds i8, ptr %st, i64 32
  store i64 0, ptr %cntp, align 8
  %errp = getelementptr inbounds i8, ptr %st, i64 40
  store i32 0, ptr %errp, align 4
  br label %blockloop

blockloop:
  %outpos = phi i64 [ 0, %init ], [ %outpos.end, %blockend ]
  %bf = call i64 @z_getbits(ptr %st, i32 1)
  %bfneg = icmp slt i64 %bf, 0
  br i1 %bfneg, label %ret_io, label %rdtype

ret_io:
  ret i64 -15

rdtype:
  %bt = call i64 @z_getbits(ptr %st, i32 2)
  %btneg = icmp slt i64 %bt, 0
  br i1 %btneg, label %ret_io, label %dispatch

dispatch:
  switch i64 %bt, label %ret_parse [ i64 0, label %stored
                                      i64 1, label %fixed
                                      i64 2, label %dynamic ]

ret_parse:
  ret i64 -13

; ---- stored (type 0) ----
stored:
  %scnt = load i64, ptr %cntp, align 8
  %drop = and i64 %scnt, 7
  %sbuf = load i64, ptr %bufp, align 8
  %sbuf2 = lshr i64 %sbuf, %drop
  store i64 %sbuf2, ptr %bufp, align 8
  %scnt2 = sub i64 %scnt, %drop
  store i64 %scnt2, ptr %cntp, align 8
  %LEN = call i64 @z_getbits(ptr %st, i32 16)
  %LENneg = icmp slt i64 %LEN, 0
  br i1 %LENneg, label %ret_io, label %stored2

stored2:
  %NLEN = call i64 @z_getbits(ptr %st, i32 16)
  %NLENneg = icmp slt i64 %NLEN, 0
  br i1 %NLENneg, label %ret_io, label %stored3

stored3:
  %lenxor = xor i64 %LEN, 65535
  %nlmatch = icmp eq i64 %lenxor, %NLEN
  br i1 %nlmatch, label %stored4, label %ret_parse

stored4:
  %sendpos = add i64 %outpos, %LEN
  %sroom = icmp ule i64 %sendpos, %dstcap
  br i1 %sroom, label %drain.head, label %ret_full

ret_full:
  ret i64 -6

drain.head:
  %L = phi i64 [ %LEN, %stored4 ], [ %L.n, %drain.body ]
  %dop = phi i64 [ %outpos, %stored4 ], [ %dop.n, %drain.body ]
  %dbuf = phi i64 [ %sbuf2, %stored4 ], [ %dbuf.n, %drain.body ]
  %dcnt = phi i64 [ %scnt2, %stored4 ], [ %dcnt.n, %drain.body ]
  %Lpos = icmp ugt i64 %L, 0
  %cpos = icmp ugt i64 %dcnt, 0
  %bothd = and i1 %Lpos, %cpos
  br i1 %bothd, label %drain.body, label %drain.done

drain.body:
  %dbyte = and i64 %dbuf, 255
  %dbyte8 = trunc i64 %dbyte to i8
  %ddp = getelementptr inbounds i8, ptr %dst, i64 %dop
  store i8 %dbyte8, ptr %ddp, align 1
  %dbuf.n = lshr i64 %dbuf, 8
  %dcnt.n = sub i64 %dcnt, 8
  %dop.n = add i64 %dop, 1
  %L.n = sub i64 %L, 1
  br label %drain.head

drain.done:
  store i64 %dbuf, ptr %bufp, align 8
  store i64 %dcnt, ptr %cntp, align 8
  %remain = icmp ugt i64 %L, 0
  br i1 %remain, label %bulk, label %stored.fin

bulk:
  %bpos = load i64, ptr %posp, align 8
  %bend = add i64 %bpos, %L
  %bavail = icmp ule i64 %bend, %srclen
  br i1 %bavail, label %bulk2, label %ret_io

bulk2:
  %bsrc = getelementptr inbounds i8, ptr %src, i64 %bpos
  %bdst = getelementptr inbounds i8, ptr %dst, i64 %dop
  call void @llvm.memcpy.p0.p0.i64(ptr %bdst, ptr %bsrc, i64 %L, i1 false)
  store i64 %bend, ptr %posp, align 8
  %bnewop = add i64 %dop, %L
  br label %stored.fin

stored.fin:
  %stored.out = phi i64 [ %dop, %drain.done ], [ %bnewop, %bulk2 ]
  br label %blockend

; ---- fixed (type 1) ----
fixed:
  call void @llvm.memset.p0.i64(ptr %lengths, i8 8, i64 144, i1 false)
  %lp144 = getelementptr inbounds i8, ptr %lengths, i64 144
  call void @llvm.memset.p0.i64(ptr %lp144, i8 9, i64 112, i1 false)
  %lp256 = getelementptr inbounds i8, ptr %lengths, i64 256
  call void @llvm.memset.p0.i64(ptr %lp256, i8 7, i64 24, i1 false)
  %lp280 = getelementptr inbounds i8, ptr %lengths, i64 280
  call void @llvm.memset.p0.i64(ptr %lp280, i8 8, i64 8, i1 false)
  %fb1 = call i32 @z_build(ptr %litcount, ptr %litsym, ptr %lengths, i32 288)
  call void @llvm.memset.p0.i64(ptr %lengths, i8 5, i64 30, i1 false)
  %fb2 = call i32 @z_build(ptr %distcount, ptr %distsym, ptr %lengths, i32 30)
  %fixed.out = call i64 @z_block(ptr %st, ptr %litcount, ptr %litsym,
                                 ptr %distcount, ptr %distsym, ptr %dst,
                                 i64 %dstcap, i64 %outpos)
  %fixed.neg = icmp slt i64 %fixed.out, 0
  br i1 %fixed.neg, label %ret_fixedneg, label %fixed.fin

ret_fixedneg:
  ret i64 %fixed.out

fixed.fin:
  br label %blockend

; ---- dynamic (type 2) ----
dynamic:
  %hlit0 = call i64 @z_getbits(ptr %st, i32 5)
  %hlneg = icmp slt i64 %hlit0, 0
  br i1 %hlneg, label %ret_io, label %dyn1

dyn1:
  %hlit = add i64 %hlit0, 257
  %hdist0 = call i64 @z_getbits(ptr %st, i32 5)
  %hdneg = icmp slt i64 %hdist0, 0
  br i1 %hdneg, label %ret_io, label %dyn2

dyn2:
  %hdist = add i64 %hdist0, 1
  %hclen0 = call i64 @z_getbits(ptr %st, i32 4)
  %hcneg = icmp slt i64 %hclen0, 0
  br i1 %hcneg, label %ret_io, label %dyn3

dyn3:
  %hclen = add i64 %hclen0, 4
  call void @llvm.memset.p0.i64(ptr %cllen, i8 0, i64 19, i1 false)
  br label %clread.head

clread.head:
  %ci = phi i64 [ 0, %dyn3 ], [ %ci.n, %clstore ]
  %cimore = icmp ult i64 %ci, %hclen
  br i1 %cimore, label %clread.body, label %clbuild

clread.body:
  %clv = call i64 @z_getbits(ptr %st, i32 3)
  %clvneg = icmp slt i64 %clv, 0
  br i1 %clvneg, label %ret_io, label %clstore

clstore:
  %ordp = getelementptr inbounds [19 x i8], ptr @z.clc_order, i64 0, i64 %ci
  %ord8 = load i8, ptr %ordp, align 1
  %ord = zext i8 %ord8 to i64
  %clvp = getelementptr inbounds i8, ptr %cllen, i64 %ord
  %clv8 = trunc i64 %clv to i8
  store i8 %clv8, ptr %clvp, align 1
  %ci.n = add i64 %ci, 1
  br label %clread.head

clbuild:
  %cb = call i32 @z_build(ptr %clcount, ptr %clsym, ptr %cllen, i32 19)
  %total = add i64 %hlit, %hdist
  br label %cl.head

cl.head:
  %ti = phi i64 [ 0, %clbuild ], [ %ti.lit, %cl.lit ], [ %fillend, %cl.cont.rep ]
  %timore = icmp ult i64 %ti, %total
  br i1 %timore, label %cl.decode, label %cl.done

cl.decode:
  %csym = call i32 @z_decode(ptr %st, ptr %clcount, ptr %clsym)
  %csneg = icmp slt i32 %csym, 0
  br i1 %csneg, label %ret_parse2, label %cl.class

ret_parse2:
  ret i64 -13

cl.class:
  %islit2 = icmp ult i32 %csym, 16
  br i1 %islit2, label %cl.lit, label %cl.rep

cl.lit:
  %csym8 = trunc i32 %csym to i8
  %tlp = getelementptr inbounds i8, ptr %lengths, i64 %ti
  store i8 %csym8, ptr %tlp, align 1
  %ti.lit = add i64 %ti, 1
  br label %cl.head

cl.rep:
  %is16 = icmp eq i32 %csym, 16
  br i1 %is16, label %rep16, label %rep_other

rep16:
  %tinz = icmp ugt i64 %ti, 0
  br i1 %tinz, label %rep16b, label %ret_parse3

ret_parse3:
  ret i64 -13

rep16b:
  %r16 = call i64 @z_getbits(ptr %st, i32 2)
  %r16neg = icmp slt i64 %r16, 0
  br i1 %r16neg, label %ret_io, label %rep16c

rep16c:
  %rep16v = add i64 %r16, 3
  %previ = sub i64 %ti, 1
  %prevp = getelementptr inbounds i8, ptr %lengths, i64 %previ
  %prev8 = load i8, ptr %prevp, align 1
  br label %fill.pre

rep_other:
  %is17 = icmp eq i32 %csym, 17
  br i1 %is17, label %rep17, label %rep18

rep17:
  %r17 = call i64 @z_getbits(ptr %st, i32 3)
  %r17neg = icmp slt i64 %r17, 0
  br i1 %r17neg, label %ret_io, label %rep17c

rep17c:
  %rep17v = add i64 %r17, 3
  br label %fill.pre

rep18:
  %is18 = icmp eq i32 %csym, 18
  br i1 %is18, label %rep18b, label %ret_parse4

ret_parse4:
  ret i64 -13

rep18b:
  %r18 = call i64 @z_getbits(ptr %st, i32 7)
  %r18neg = icmp slt i64 %r18, 0
  br i1 %r18neg, label %ret_io, label %rep18c

rep18c:
  %rep18v = add i64 %r18, 11
  br label %fill.pre

fill.pre:
  %repcount = phi i64 [ %rep16v, %rep16c ], [ %rep17v, %rep17c ], [ %rep18v, %rep18c ]
  %fillval = phi i8 [ %prev8, %rep16c ], [ 0, %rep17c ], [ 0, %rep18c ]
  %fillend = add i64 %ti, %repcount
  %fillok = icmp ule i64 %fillend, %total
  br i1 %fillok, label %fill.loop, label %ret_parse5

ret_parse5:
  ret i64 -13

fill.loop:
  %fi = phi i64 [ %ti, %fill.pre ], [ %fi.n, %fill.loop ]
  %fp = getelementptr inbounds i8, ptr %lengths, i64 %fi
  store i8 %fillval, ptr %fp, align 1
  %fi.n = add i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, %fillend
  br i1 %fmore, label %fill.loop, label %cl.cont.rep

cl.cont.rep:
  br label %cl.head

cl.done:
  %hlit32 = trunc i64 %hlit to i32
  %lb = call i32 @z_build(ptr %litcount, ptr %litsym, ptr %lengths, i32 %hlit32)
  %distlenp = getelementptr inbounds i8, ptr %lengths, i64 %hlit
  %hdist32 = trunc i64 %hdist to i32
  %db_ = call i32 @z_build(ptr %distcount, ptr %distsym, ptr %distlenp, i32 %hdist32)
  %dyn.out = call i64 @z_block(ptr %st, ptr %litcount, ptr %litsym,
                               ptr %distcount, ptr %distsym, ptr %dst,
                               i64 %dstcap, i64 %outpos)
  %dyn.neg = icmp slt i64 %dyn.out, 0
  br i1 %dyn.neg, label %ret_dynneg, label %dyn.fin

ret_dynneg:
  ret i64 %dyn.out

dyn.fin:
  br label %blockend

; ---- end of block ----
blockend:
  %outpos.end = phi i64 [ %stored.out, %stored.fin ], [ %fixed.out, %fixed.fin ], [ %dyn.out, %dyn.fin ]
  %fin = icmp eq i64 %bf, 1
  br i1 %fin, label %done, label %blockloop

done:
  ret i64 %outpos.end
}

; ---------------------------------------------------------------- adler32
define i64 @universe_compress_adler32(ptr %data, i64 %len) #3 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %a = phi i64 [ 1, %entry ], [ %a.m, %loop ]
  %b = phi i64 [ 0, %entry ], [ %b.m, %loop ]
  %p = getelementptr inbounds i8, ptr %data, i64 %i
  %v8 = load i8, ptr %p, align 1
  %v = zext i8 %v8 to i64
  %a1 = add i64 %a, %v
  %age = icmp uge i64 %a1, 65521
  %a1s = sub i64 %a1, 65521
  %a.m = select i1 %age, i64 %a1s, i64 %a1
  %b1 = add i64 %b, %a.m
  %bge = icmp uge i64 %b1, 65521
  %b1s = sub i64 %b1, 65521
  %b.m = select i1 %bge, i64 %b1s, i64 %b1
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %len
  br i1 %more, label %loop, label %done

done:
  %af = phi i64 [ 1, %entry ], [ %a.m, %loop ]
  %bf = phi i64 [ 0, %entry ], [ %b.m, %loop ]
  %bsh = shl i64 %bf, 16
  %r = or i64 %bsh, %af
  ret i64 %r
}

; ---------------------------------------------------------------- crc32 (IEEE)
define i64 @universe_compress_crc32(ptr %data, i64 %len) #3 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %byteend ]
  %crc = phi i32 [ -1, %entry ], [ %bcrc.n, %byteend ]
  %p = getelementptr inbounds i8, ptr %data, i64 %i
  %v8 = load i8, ptr %p, align 1
  %v = zext i8 %v8 to i32
  %cx = xor i32 %crc, %v
  br label %bit

bit:
  %bcrc = phi i32 [ %cx, %loop ], [ %bcrc.n, %bit ]
  %bj = phi i64 [ 0, %loop ], [ %bj.n, %bit ]
  %lsb = and i32 %bcrc, 1
  %m = sub i32 0, %lsb
  %poly = and i32 %m, -306674912
  %sh = lshr i32 %bcrc, 1
  %bcrc.n = xor i32 %sh, %poly
  %bj.n = add i64 %bj, 1
  %bjmore = icmp ult i64 %bj.n, 8
  br i1 %bjmore, label %bit, label %byteend

byteend:
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %len
  br i1 %more, label %loop, label %done

done:
  %crcf = phi i32 [ -1, %entry ], [ %bcrc.n, %byteend ]
  %fin = xor i32 %crcf, -1
  %finz = zext i32 %fin to i64
  ret i64 %finz
}

; ---------------------------------------------------------------- zlib wrapper
define i64 @universe_compress_inflate_zlib(ptr %dst, i64 %dstcap, ptr %src, i64 %srclen) #2 {
entry:
  %tooshort = icmp ult i64 %srclen, 6
  br i1 %tooshort, label %rp, label %hdr

rp:
  ret i64 -13

hdr:
  %cmf8 = load i8, ptr %src, align 1
  %cmf = zext i8 %cmf8 to i32
  %flgp = getelementptr inbounds i8, ptr %src, i64 1
  %flg8 = load i8, ptr %flgp, align 1
  %flg = zext i8 %flg8 to i32
  %cm = and i32 %cmf, 15
  %cmok = icmp eq i32 %cm, 8
  br i1 %cmok, label %chk1, label %unsup

unsup:
  ret i64 -14

chk1:
  %hi = shl i32 %cmf, 8
  %chkv = or i32 %hi, %flg
  %rem = urem i32 %chkv, 31
  %remok = icmp eq i32 %rem, 0
  br i1 %remok, label %chk2, label %rp

chk2:
  %fdict = and i32 %flg, 32
  %hasdict = icmp ne i32 %fdict, 0
  br i1 %hasdict, label %unsup, label %doinf

doinf:
  %defsrc = getelementptr inbounds i8, ptr %src, i64 2
  %deflen = sub i64 %srclen, 2
  %n = call i64 @universe_compress_inflate(ptr %dst, i64 %dstcap, ptr %defsrc, i64 %deflen)
  %nneg = icmp slt i64 %n, 0
  br i1 %nneg, label %propn, label %adler

propn:
  ret i64 %n

adler:
  %ab = sub i64 %srclen, 4
  %a0p = getelementptr inbounds i8, ptr %src, i64 %ab
  %a0 = load i8, ptr %a0p, align 1
  %a0z = zext i8 %a0 to i32
  %ab1 = add i64 %ab, 1
  %a1p = getelementptr inbounds i8, ptr %src, i64 %ab1
  %a1 = load i8, ptr %a1p, align 1
  %a1z = zext i8 %a1 to i32
  %ab2 = add i64 %ab, 2
  %a2p = getelementptr inbounds i8, ptr %src, i64 %ab2
  %a2 = load i8, ptr %a2p, align 1
  %a2z = zext i8 %a2 to i32
  %ab3 = add i64 %ab, 3
  %a3p = getelementptr inbounds i8, ptr %src, i64 %ab3
  %a3 = load i8, ptr %a3p, align 1
  %a3z = zext i8 %a3 to i32
  %w0 = shl i32 %a0z, 24
  %w1 = shl i32 %a1z, 16
  %w2 = shl i32 %a2z, 8
  %w01 = or i32 %w0, %w1
  %w23 = or i32 %w2, %a3z
  %want = or i32 %w01, %w23
  %got = call i64 @universe_compress_adler32(ptr %dst, i64 %n)
  %got32 = trunc i64 %got to i32
  %match = icmp eq i32 %want, %got32
  br i1 %match, label %propn, label %rp
}

; ---------------------------------------------------------------- gzip wrapper
define i64 @universe_compress_inflate_gzip(ptr %dst, i64 %dstcap, ptr %src, i64 %srclen) #2 {
entry:
  %short = icmp ult i64 %srclen, 18
  br i1 %short, label %rp, label %hdr

rp:
  ret i64 -13

hdr:
  %m0 = load i8, ptr %src, align 1
  %m1p = getelementptr inbounds i8, ptr %src, i64 1
  %m1 = load i8, ptr %m1p, align 1
  %cmp2 = getelementptr inbounds i8, ptr %src, i64 2
  %cm = load i8, ptr %cmp2, align 1
  %flgp = getelementptr inbounds i8, ptr %src, i64 3
  %flg8 = load i8, ptr %flgp, align 1
  %flg = zext i8 %flg8 to i32
  %ism0 = icmp eq i8 %m0, 31
  %ism1 = icmp eq i8 %m1, -117
  %magic = and i1 %ism0, %ism1
  br i1 %magic, label %hdr2, label %rp

hdr2:
  %iscm = icmp eq i8 %cm, 8
  br i1 %iscm, label %extra, label %unsup

unsup:
  ret i64 -14

extra:
  %fx = and i32 %flg, 4
  %hasx = icmp ne i32 %fx, 0
  br i1 %hasx, label %extraB, label %afterextra

extraB:
  %ex0off = add i64 10, 1
  %x0p = getelementptr inbounds i8, ptr %src, i64 10
  %x0 = load i8, ptr %x0p, align 1
  %x0z = zext i8 %x0 to i64
  %x1p = getelementptr inbounds i8, ptr %src, i64 %ex0off
  %x1 = load i8, ptr %x1p, align 1
  %x1z = zext i8 %x1 to i64
  %x1s = shl i64 %x1z, 8
  %xlen = or i64 %x0z, %x1s
  %off2a = add i64 12, %xlen
  %xok = icmp ule i64 %off2a, %srclen
  br i1 %xok, label %afterextra, label %rp

afterextra:
  %offA = phi i64 [ 10, %extra ], [ %off2a, %extraB ]
  %fn = and i32 %flg, 8
  %hasn = icmp ne i32 %fn, 0
  br i1 %hasn, label %name.head, label %aftername

name.head:
  %ni = phi i64 [ %offA, %afterextra ], [ %ni.n, %name.body ]
  %nin = icmp ult i64 %ni, %srclen
  br i1 %nin, label %name.body, label %rp

name.body:
  %nbp = getelementptr inbounds i8, ptr %src, i64 %ni
  %nb = load i8, ptr %nbp, align 1
  %ni.n = add i64 %ni, 1
  %nzero = icmp eq i8 %nb, 0
  br i1 %nzero, label %aftername, label %name.head

aftername:
  %offB = phi i64 [ %offA, %afterextra ], [ %ni.n, %name.body ]
  %fc = and i32 %flg, 16
  %hasc = icmp ne i32 %fc, 0
  br i1 %hasc, label %cmt.head, label %aftercomment

cmt.head:
  %mi = phi i64 [ %offB, %aftername ], [ %mi.n, %cmt.body ]
  %cin = icmp ult i64 %mi, %srclen
  br i1 %cin, label %cmt.body, label %rp

cmt.body:
  %cbp = getelementptr inbounds i8, ptr %src, i64 %mi
  %cbyte = load i8, ptr %cbp, align 1
  %mi.n = add i64 %mi, 1
  %czero = icmp eq i8 %cbyte, 0
  br i1 %czero, label %aftercomment, label %cmt.head

aftercomment:
  %offC = phi i64 [ %offB, %aftername ], [ %mi.n, %cmt.body ]
  %fh = and i32 %flg, 2
  %hashc = icmp ne i32 %fh, 0
  %offCh = add i64 %offC, 2
  %offD = select i1 %hashc, i64 %offCh, i64 %offC
  %trailer = add i64 %offD, 8
  %dok = icmp ule i64 %trailer, %srclen
  br i1 %dok, label %doinf, label %rp

doinf:
  %defsrc = getelementptr inbounds i8, ptr %src, i64 %offD
  %deflen = sub i64 %srclen, %offD
  %n = call i64 @universe_compress_inflate(ptr %dst, i64 %dstcap, ptr %defsrc, i64 %deflen)
  %nneg = icmp slt i64 %n, 0
  br i1 %nneg, label %propn, label %crcchk

propn:
  ret i64 %n

crcchk:
  %cb = sub i64 %srclen, 8
  %c0p = getelementptr inbounds i8, ptr %src, i64 %cb
  %c0 = load i8, ptr %c0p, align 1
  %c0z = zext i8 %c0 to i32
  %cb1 = add i64 %cb, 1
  %c1p = getelementptr inbounds i8, ptr %src, i64 %cb1
  %c1 = load i8, ptr %c1p, align 1
  %c1z = zext i8 %c1 to i32
  %cb2 = add i64 %cb, 2
  %c2p = getelementptr inbounds i8, ptr %src, i64 %cb2
  %c2 = load i8, ptr %c2p, align 1
  %c2z = zext i8 %c2 to i32
  %cb3 = add i64 %cb, 3
  %c3p = getelementptr inbounds i8, ptr %src, i64 %cb3
  %c3 = load i8, ptr %c3p, align 1
  %c3z = zext i8 %c3 to i32
  %cw1 = shl i32 %c1z, 8
  %cw2 = shl i32 %c2z, 16
  %cw3 = shl i32 %c3z, 24
  %cw01 = or i32 %c0z, %cw1
  %cw23 = or i32 %cw2, %cw3
  %cwant = or i32 %cw01, %cw23
  %cgot = call i64 @universe_compress_crc32(ptr %dst, i64 %n)
  %cgot32 = trunc i64 %cgot to i32
  %cmatch = icmp eq i32 %cwant, %cgot32
  br i1 %cmatch, label %isizechk, label %rp

isizechk:
  %ib = sub i64 %srclen, 4
  %i0p = getelementptr inbounds i8, ptr %src, i64 %ib
  %i0 = load i8, ptr %i0p, align 1
  %i0z = zext i8 %i0 to i64
  %ib1 = add i64 %ib, 1
  %i1p = getelementptr inbounds i8, ptr %src, i64 %ib1
  %i1 = load i8, ptr %i1p, align 1
  %i1z = zext i8 %i1 to i64
  %ib2 = add i64 %ib, 2
  %i2p = getelementptr inbounds i8, ptr %src, i64 %ib2
  %i2 = load i8, ptr %i2p, align 1
  %i2z = zext i8 %i2 to i64
  %ib3 = add i64 %ib, 3
  %i3p = getelementptr inbounds i8, ptr %src, i64 %ib3
  %i3 = load i8, ptr %i3p, align 1
  %i3z = zext i8 %i3 to i64
  %iw1 = shl i64 %i1z, 8
  %iw2 = shl i64 %i2z, 16
  %iw3 = shl i64 %i3z, 24
  %iw01 = or i64 %i0z, %iw1
  %iw23 = or i64 %iw2, %iw3
  %isize = or i64 %iw01, %iw23
  %nlow = and i64 %n, 4294967295
  %imatch = icmp eq i64 %isize, %nlow
  br i1 %imatch, label %propn, label %rp
}

; --------------------------------------------------------- DEFLATE: stored enc
define i64 @universe_compress_deflate_stored(ptr %dst, i64 %dstcap, ptr %src, i64 %srclen) #2 {
entry:
  %dnull = icmp eq ptr %dst, null
  br i1 %dnull, label %rn, label %chunk.head

rn:
  ret i64 -1

chunk.head:
  %rem = phi i64 [ %srclen, %entry ], [ %rem.n, %chunk.cont ]
  %ip = phi i64 [ 0, %entry ], [ %ip.n, %chunk.cont ]
  %outp = phi i64 [ 0, %entry ], [ %outp.n, %chunk.cont ]
  %big = icmp ugt i64 %rem, 65535
  %chunk = select i1 %big, i64 65535, i64 %rem
  %final = icmp ule i64 %rem, 65535
  %need5 = add i64 %outp, 5
  %needt = add i64 %need5, %chunk
  %roomok = icmp ule i64 %needt, %dstcap
  br i1 %roomok, label %write, label %full

full:
  ret i64 -6

write:
  %hbyte = select i1 %final, i8 1, i8 0
  %hp = getelementptr inbounds i8, ptr %dst, i64 %outp
  store i8 %hbyte, ptr %hp, align 1
  %clo = trunc i64 %chunk to i8
  %chi64 = lshr i64 %chunk, 8
  %chi = trunc i64 %chi64 to i8
  %o1 = add i64 %outp, 1
  %lp0 = getelementptr inbounds i8, ptr %dst, i64 %o1
  store i8 %clo, ptr %lp0, align 1
  %o2 = add i64 %outp, 2
  %lp1 = getelementptr inbounds i8, ptr %dst, i64 %o2
  store i8 %chi, ptr %lp1, align 1
  %nlen = xor i64 %chunk, 65535
  %nlo = trunc i64 %nlen to i8
  %nhi64 = lshr i64 %nlen, 8
  %nhi = trunc i64 %nhi64 to i8
  %o3 = add i64 %outp, 3
  %lp2 = getelementptr inbounds i8, ptr %dst, i64 %o3
  store i8 %nlo, ptr %lp2, align 1
  %o4 = add i64 %outp, 4
  %lp3 = getelementptr inbounds i8, ptr %dst, i64 %o4
  store i8 %nhi, ptr %lp3, align 1
  %ddp = getelementptr inbounds i8, ptr %dst, i64 %need5
  %ssp = getelementptr inbounds i8, ptr %src, i64 %ip
  call void @llvm.memcpy.p0.p0.i64(ptr %ddp, ptr %ssp, i64 %chunk, i1 false)
  %outp.n = add i64 %needt, 0
  %ip.n = add i64 %ip, %chunk
  %rem.n = sub i64 %rem, %chunk
  br i1 %final, label %done, label %chunk.cont

chunk.cont:
  br label %chunk.head

done:
  ret i64 %needt
}

; -------------------------------------------------------- bit writer (encoder)
; ws: outp@0(i64) dstcap@8(i64) dst@16(ptr) bitbuf@24(i64) bitcnt@32(i64) err@40(i32)
define internal void @z_putbits(ptr %ws, i32 %val, i32 %nbits) #1 {
entry:
  %bbp = getelementptr inbounds i8, ptr %ws, i64 24
  %bb = load i64, ptr %bbp, align 8
  %bcp = getelementptr inbounds i8, ptr %ws, i64 32
  %bc = load i64, ptr %bcp, align 8
  %v64 = zext i32 %val to i64
  %sh = shl i64 %v64, %bc
  %bb2 = or i64 %bb, %sh
  %nb64 = zext i32 %nbits to i64
  %bc2 = add i64 %bc, %nb64
  br label %flush.head

flush.head:
  %cbb = phi i64 [ %bb2, %entry ], [ %cbb.n, %flush.body ]
  %cbc = phi i64 [ %bc2, %entry ], [ %cbc.n, %flush.body ]
  %hasbyte = icmp uge i64 %cbc, 8
  br i1 %hasbyte, label %flush.chk, label %flush.done

flush.chk:
  %outpp = getelementptr inbounds i8, ptr %ws, i64 0
  %outp = load i64, ptr %outpp, align 8
  %capp = getelementptr inbounds i8, ptr %ws, i64 8
  %cap = load i64, ptr %capp, align 8
  %room = icmp ult i64 %outp, %cap
  br i1 %room, label %flush.body, label %seterr

seterr:
  %errp = getelementptr inbounds i8, ptr %ws, i64 40
  store i32 6, ptr %errp, align 4
  br label %flush.done

flush.body:
  %dstpp = getelementptr inbounds i8, ptr %ws, i64 16
  %dstp = load ptr, ptr %dstpp, align 8
  %obyte = and i64 %cbb, 255
  %obyte8 = trunc i64 %obyte to i8
  %odp = getelementptr inbounds i8, ptr %dstp, i64 %outp
  store i8 %obyte8, ptr %odp, align 1
  %outp2 = add i64 %outp, 1
  store i64 %outp2, ptr %outpp, align 8
  %cbb.n = lshr i64 %cbb, 8
  %cbc.n = sub i64 %cbc, 8
  br label %flush.head

flush.done:
  store i64 %cbb, ptr %bbp, align 8
  store i64 %cbc, ptr %bcp, align 8
  ret void
}

; reverse the low %len bits of %code (DEFLATE packs Huffman codes MSB-first)
define internal i32 @z_rev(i32 %code, i32 %len) #1 {
entry:
  %z = icmp eq i32 %len, 0
  br i1 %z, label %retz, label %loop

retz:
  ret i32 0

loop:
  %i = phi i32 [ 0, %entry ], [ %i.n, %loop ]
  %r = phi i32 [ 0, %entry ], [ %r.n, %loop ]
  %c = phi i32 [ %code, %entry ], [ %c.n, %loop ]
  %bit = and i32 %c, 1
  %rsh = shl i32 %r, 1
  %r.n = or i32 %rsh, %bit
  %c.n = lshr i32 %c, 1
  %i.n = add i32 %i, 1
  %more = icmp ult i32 %i.n, %len
  br i1 %more, label %loop, label %done

done:
  ret i32 %r.n
}

; --------------------------------------------------- DEFLATE: fixed-huffman enc
; Literals only (no back-references): a valid, fully-decodable fixed block that
; exercises the fixed-Huffman DECODE path from a self-produced bitstream.
define i64 @universe_compress_deflate_fixed(ptr %dst, i64 %dstcap, ptr %src, i64 %srclen) #2 {
entry:
  %ws = alloca [48 x i8], align 8
  %dnull = icmp eq ptr %dst, null
  br i1 %dnull, label %rn, label %init

rn:
  ret i64 -1

init:
  %op0p = getelementptr inbounds i8, ptr %ws, i64 0
  store i64 0, ptr %op0p, align 8
  %capp = getelementptr inbounds i8, ptr %ws, i64 8
  store i64 %dstcap, ptr %capp, align 8
  %dstpp = getelementptr inbounds i8, ptr %ws, i64 16
  store ptr %dst, ptr %dstpp, align 8
  %bbp = getelementptr inbounds i8, ptr %ws, i64 24
  store i64 0, ptr %bbp, align 8
  %bcp = getelementptr inbounds i8, ptr %ws, i64 32
  store i64 0, ptr %bcp, align 8
  %errp = getelementptr inbounds i8, ptr %ws, i64 40
  store i32 0, ptr %errp, align 4
  call void @z_putbits(ptr %ws, i32 3, i32 3)
  br label %lit.head

lit.head:
  %i = phi i64 [ 0, %init ], [ %i.n, %lit.cont ]
  %more = icmp ult i64 %i, %srclen
  br i1 %more, label %lit.body, label %emitend

lit.body:
  %sp = getelementptr inbounds i8, ptr %src, i64 %i
  %b8 = load i8, ptr %sp, align 1
  %b = zext i8 %b8 to i32
  %small = icmp ule i32 %b, 143
  br i1 %small, label %litA, label %litB

litA:
  %codeA = add i32 %b, 48
  %revA = call i32 @z_rev(i32 %codeA, i32 8)
  call void @z_putbits(ptr %ws, i32 %revA, i32 8)
  br label %lit.cont

litB:
  %codeB0 = sub i32 %b, 144
  %codeB = add i32 %codeB0, 400
  %revB = call i32 @z_rev(i32 %codeB, i32 9)
  call void @z_putbits(ptr %ws, i32 %revB, i32 9)
  br label %lit.cont

lit.cont:
  %i.n = add i64 %i, 1
  br label %lit.head

emitend:
  call void @z_putbits(ptr %ws, i32 0, i32 7)
  %fbcp = getelementptr inbounds i8, ptr %ws, i64 32
  %fbc = load i64, ptr %fbcp, align 8
  %hasrem = icmp ugt i64 %fbc, 0
  br i1 %hasrem, label %pad, label %finish

pad:
  %foutp = getelementptr inbounds i8, ptr %ws, i64 0
  %fout = load i64, ptr %foutp, align 8
  %fcapp = getelementptr inbounds i8, ptr %ws, i64 8
  %fcap = load i64, ptr %fcapp, align 8
  %proom = icmp ult i64 %fout, %fcap
  br i1 %proom, label %padwrite, label %errfull

errfull:
  ret i64 -6

padwrite:
  %fbbp = getelementptr inbounds i8, ptr %ws, i64 24
  %fbb = load i64, ptr %fbbp, align 8
  %fbyte = and i64 %fbb, 255
  %fbyte8 = trunc i64 %fbyte to i8
  %fdstpp = getelementptr inbounds i8, ptr %ws, i64 16
  %fdstp = load ptr, ptr %fdstpp, align 8
  %fdp = getelementptr inbounds i8, ptr %fdstp, i64 %fout
  store i8 %fbyte8, ptr %fdp, align 1
  %fout2 = add i64 %fout, 1
  store i64 %fout2, ptr %foutp, align 8
  br label %finish

finish:
  %efp = getelementptr inbounds i8, ptr %ws, i64 40
  %eflag = load i32, ptr %efp, align 4
  %ehas = icmp ne i32 %eflag, 0
  br i1 %ehas, label %errfull, label %ok

ok:
  %routp = getelementptr inbounds i8, ptr %ws, i64 0
  %rout = load i64, ptr %routp, align 8
  ret i64 %rout
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
