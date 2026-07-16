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

; ZIP archive reader over a caller-owned buffer (APPNOTE local/central layout).
;
; DESIGN:
;   * ZERO-COPY, ZERO-ALLOC. The whole .zip sits in the caller buffer; every
;     entry name is reported as a {ptr,len} slice INTO that buffer, never
;     strdup'd. The only caller state is a tiny reader struct and a per-entry
;     struct the caller allocates. This is exactly the shape the XLSX/PDF
;     parsers need: locate the End-Of-Central-Directory, walk the central
;     directory, then inflate a chosen member's DEFLATE data in place.
;   * All multi-byte fields are little-endian per APPNOTE; @z_rd16/@z_rd32 read
;     them by byte so layout is identical on every target (no struct padding
;     assumptions). Bounds are checked before each header read.
;   * EOCD is found by scanning backward from the end for its signature (a
;     trailing archive comment of up to 65535 bytes may follow it).
;   * extract() dispatches on the stored method: 0 = raw copy, 8 = DEFLATE via
;     universe_compress_inflate. Compressed/uncompressed sizes are validated.
;
;   Reader struct (32 B, caller-allocated): buf@0(ptr) len@8(i64)
;                                            cd_off@16(i64) count@24(i64)
;   Entry struct  (48 B, caller-allocated): name@0(ptr) namelen@8(i64)
;                                            method@16(i64) compsize@24(i64)
;                                            uncompsize@32(i64) localoff@40(i64)
;
; API:
;   i32 universe_compress_zip_open(ptr buf, i64 len, ptr reader)  ; 0/1/13
;   i64 universe_compress_zip_count(ptr reader)
;   i32 universe_compress_zip_entry(ptr reader, i64 index, ptr entry) ; 0/7/13
;   i64 universe_compress_zip_extract(ptr reader, ptr entry, ptr dst, i64 cap)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare i64 @universe_compress_inflate(ptr, i64, ptr, i64)

define internal i32 @z_rd16(ptr %base, i64 %off) #0 {
entry:
  %p0 = getelementptr inbounds i8, ptr %base, i64 %off
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i32
  %o1 = add i64 %off, 1
  %p1 = getelementptr inbounds i8, ptr %base, i64 %o1
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i32
  %s1 = shl i32 %z1, 8
  %r = or i32 %z0, %s1
  ret i32 %r
}

define internal i64 @z_rd32(ptr %base, i64 %off) #0 {
entry:
  %p0 = getelementptr inbounds i8, ptr %base, i64 %off
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i64
  %o1 = add i64 %off, 1
  %p1 = getelementptr inbounds i8, ptr %base, i64 %o1
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i64
  %s1 = shl i64 %z1, 8
  %o2 = add i64 %off, 2
  %p2 = getelementptr inbounds i8, ptr %base, i64 %o2
  %b2 = load i8, ptr %p2, align 1
  %z2 = zext i8 %b2 to i64
  %s2 = shl i64 %z2, 16
  %o3 = add i64 %off, 3
  %p3 = getelementptr inbounds i8, ptr %base, i64 %o3
  %b3 = load i8, ptr %p3, align 1
  %z3 = zext i8 %b3 to i64
  %s3 = shl i64 %z3, 24
  %r01 = or i64 %z0, %s1
  %r23 = or i64 %s2, %s3
  %r = or i64 %r01, %r23
  ret i64 %r
}

; ------------------------------------------------------------------------- open
define i32 @universe_compress_zip_open(ptr %buf, i64 %len, ptr %reader) #1 {
entry:
  %bnull = icmp eq ptr %buf, null
  %rnull = icmp eq ptr %reader, null
  %anynull = or i1 %bnull, %rnull
  br i1 %anynull, label %ret_null, label %chk

ret_null:
  ret i32 1

chk:
  %tooshort = icmp ult i64 %len, 22
  br i1 %tooshort, label %ret_parse, label %scan.pre

ret_parse:
  ret i32 13

scan.pre:
  %start = sub i64 %len, 22
  ; low bound = max(0, len - 22 - 65535)
  %window = add i64 65535, 22
  %hasfloor = icmp ugt i64 %len, %window
  %floor0 = sub i64 %len, %window
  %floor = select i1 %hasfloor, i64 %floor0, i64 0
  br label %scan.loop

scan.loop:
  %i = phi i64 [ %start, %scan.pre ], [ %i.n, %scan.cont ]
  %sig = call i64 @z_rd32(ptr %buf, i64 %i)
  %iseocd = icmp eq i64 %sig, 101010256
  br i1 %iseocd, label %found, label %scan.cont

scan.cont:
  %atfloor = icmp ule i64 %i, %floor
  %i.n = sub i64 %i, 1
  br i1 %atfloor, label %ret_parse, label %scan.loop

found:
  ; total entries at eocd+10
  %o_cnt = add i64 %i, 10
  %cnt = call i32 @z_rd16(ptr %buf, i64 %o_cnt)
  %cnt64 = zext i32 %cnt to i64
  %o_off = add i64 %i, 16
  %cdoff = call i64 @z_rd32(ptr %buf, i64 %o_off)
  ; store reader
  %r_buf = getelementptr inbounds i8, ptr %reader, i64 0
  store ptr %buf, ptr %r_buf, align 8
  %r_len = getelementptr inbounds i8, ptr %reader, i64 8
  store i64 %len, ptr %r_len, align 8
  %r_cd = getelementptr inbounds i8, ptr %reader, i64 16
  store i64 %cdoff, ptr %r_cd, align 8
  %r_cnt = getelementptr inbounds i8, ptr %reader, i64 24
  store i64 %cnt64, ptr %r_cnt, align 8
  ret i32 0
}

; ------------------------------------------------------------------------ count
define i64 @universe_compress_zip_count(ptr %reader) #2 {
entry:
  %rnull = icmp eq ptr %reader, null
  br i1 %rnull, label %z, label %ld
z:
  ret i64 0
ld:
  %r_cnt = getelementptr inbounds i8, ptr %reader, i64 24
  %cnt = load i64, ptr %r_cnt, align 8
  ret i64 %cnt
}

; ------------------------------------------------------------------------ entry
define i32 @universe_compress_zip_entry(ptr %reader, i64 %index, ptr %entry) #1 {
begin:
  %rnull = icmp eq ptr %reader, null
  %enull = icmp eq ptr %entry, null
  %anynull = or i1 %rnull, %enull
  br i1 %anynull, label %ret_null, label %ld

ret_null:
  ret i32 1

ld:
  %r_buf = getelementptr inbounds i8, ptr %reader, i64 0
  %buf = load ptr, ptr %r_buf, align 8
  %r_len = getelementptr inbounds i8, ptr %reader, i64 8
  %len = load i64, ptr %r_len, align 8
  %r_cd = getelementptr inbounds i8, ptr %reader, i64 16
  %cdoff = load i64, ptr %r_cd, align 8
  %r_cnt = getelementptr inbounds i8, ptr %reader, i64 24
  %cnt = load i64, ptr %r_cnt, align 8
  %oob = icmp uge i64 %index, %cnt
  br i1 %oob, label %ret_idx, label %walk.head

ret_idx:
  ret i32 7

ret_parse:
  ret i32 13

walk.head:
  %off = phi i64 [ %cdoff, %ld ], [ %off.n, %walk.cont ]
  %k = phi i64 [ 0, %ld ], [ %k.n, %walk.cont ]
  ; need 46 bytes of central header
  %need = add i64 %off, 46
  %fits = icmp ule i64 %need, %len
  br i1 %fits, label %walk.sig, label %ret_parse

walk.sig:
  %sig = call i64 @z_rd32(ptr %buf, i64 %off)
  %issig = icmp eq i64 %sig, 33639248
  br i1 %issig, label %walk.body, label %ret_parse

walk.body:
  %attarget = icmp eq i64 %k, %index
  br i1 %attarget, label %fill, label %walk.cont

walk.cont:
  %o_nl = add i64 %off, 28
  %nl = call i32 @z_rd16(ptr %buf, i64 %o_nl)
  %nl64 = zext i32 %nl to i64
  %o_el = add i64 %off, 30
  %el = call i32 @z_rd16(ptr %buf, i64 %o_el)
  %el64 = zext i32 %el to i64
  %o_cl = add i64 %off, 32
  %cl = call i32 @z_rd16(ptr %buf, i64 %o_cl)
  %cl64 = zext i32 %cl to i64
  %adv0 = add i64 %off, 46
  %adv1 = add i64 %adv0, %nl64
  %adv2 = add i64 %adv1, %el64
  %off.n = add i64 %adv2, %cl64
  %k.n = add i64 %k, 1
  br label %walk.head

fill:
  %o_m = add i64 %off, 10
  %method = call i32 @z_rd16(ptr %buf, i64 %o_m)
  %method64 = zext i32 %method to i64
  %o_cs = add i64 %off, 20
  %compsize = call i64 @z_rd32(ptr %buf, i64 %o_cs)
  %o_us = add i64 %off, 24
  %uncompsize = call i64 @z_rd32(ptr %buf, i64 %o_us)
  %o_nl2 = add i64 %off, 28
  %namelen = call i32 @z_rd16(ptr %buf, i64 %o_nl2)
  %namelen64 = zext i32 %namelen to i64
  %o_lo = add i64 %off, 42
  %localoff = call i64 @z_rd32(ptr %buf, i64 %o_lo)
  ; name pointer = buf + off + 46 ; verify name in-bounds
  %nameoff = add i64 %off, 46
  %nameend = add i64 %nameoff, %namelen64
  %namefits = icmp ule i64 %nameend, %len
  br i1 %namefits, label %fill2, label %ret_parse

fill2:
  %nameptr = getelementptr inbounds i8, ptr %buf, i64 %nameoff
  %e_name = getelementptr inbounds i8, ptr %entry, i64 0
  store ptr %nameptr, ptr %e_name, align 8
  %e_nl = getelementptr inbounds i8, ptr %entry, i64 8
  store i64 %namelen64, ptr %e_nl, align 8
  %e_m = getelementptr inbounds i8, ptr %entry, i64 16
  store i64 %method64, ptr %e_m, align 8
  %e_cs = getelementptr inbounds i8, ptr %entry, i64 24
  store i64 %compsize, ptr %e_cs, align 8
  %e_us = getelementptr inbounds i8, ptr %entry, i64 32
  store i64 %uncompsize, ptr %e_us, align 8
  %e_lo = getelementptr inbounds i8, ptr %entry, i64 40
  store i64 %localoff, ptr %e_lo, align 8
  ret i32 0
}

; ---------------------------------------------------------------------- extract
define i64 @universe_compress_zip_extract(ptr %reader, ptr %entry, ptr %dst, i64 %dstcap) #1 {
begin:
  %rnull = icmp eq ptr %reader, null
  %enull = icmp eq ptr %entry, null
  %dnull = icmp eq ptr %dst, null
  %n0 = or i1 %rnull, %enull
  %anynull = or i1 %n0, %dnull
  br i1 %anynull, label %ret_null, label %ld

ret_null:
  ret i64 -1

ld:
  %r_buf = getelementptr inbounds i8, ptr %reader, i64 0
  %buf = load ptr, ptr %r_buf, align 8
  %r_len = getelementptr inbounds i8, ptr %reader, i64 8
  %len = load i64, ptr %r_len, align 8
  %e_m = getelementptr inbounds i8, ptr %entry, i64 16
  %method = load i64, ptr %e_m, align 8
  %e_cs = getelementptr inbounds i8, ptr %entry, i64 24
  %compsize = load i64, ptr %e_cs, align 8
  %e_us = getelementptr inbounds i8, ptr %entry, i64 32
  %uncompsize = load i64, ptr %e_us, align 8
  %e_lo = getelementptr inbounds i8, ptr %entry, i64 40
  %localoff = load i64, ptr %e_lo, align 8
  ; local header: need 30 bytes
  %lhend = add i64 %localoff, 30
  %lhfits = icmp ule i64 %lhend, %len
  br i1 %lhfits, label %lhsig, label %ret_parse

ret_parse:
  ret i64 -13

lhsig:
  %sig = call i64 @z_rd32(ptr %buf, i64 %localoff)
  %issig = icmp eq i64 %sig, 67324752
  br i1 %issig, label %lhread, label %ret_parse

lhread:
  %o_nl = add i64 %localoff, 26
  %lnl = call i32 @z_rd16(ptr %buf, i64 %o_nl)
  %lnl64 = zext i32 %lnl to i64
  %o_el = add i64 %localoff, 28
  %lel = call i32 @z_rd16(ptr %buf, i64 %o_el)
  %lel64 = zext i32 %lel to i64
  %d0 = add i64 %localoff, 30
  %d1 = add i64 %d0, %lnl64
  %dataoff = add i64 %d1, %lel64
  %dataend = add i64 %dataoff, %compsize
  %datafits = icmp ule i64 %dataend, %len
  br i1 %datafits, label %dispatch, label %ret_parse

dispatch:
  %datap = getelementptr inbounds i8, ptr %buf, i64 %dataoff
  %isstored = icmp eq i64 %method, 0
  br i1 %isstored, label %stored, label %chkdef

stored:
  %sizematch = icmp eq i64 %compsize, %uncompsize
  br i1 %sizematch, label %stored2, label %ret_parse

stored2:
  %caps = icmp ule i64 %uncompsize, %dstcap
  br i1 %caps, label %stored3, label %ret_full

ret_full:
  ret i64 -6

stored3:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %datap, i64 %uncompsize, i1 false)
  ret i64 %uncompsize

chkdef:
  %isdef = icmp eq i64 %method, 8
  br i1 %isdef, label %deflate, label %ret_unsup

ret_unsup:
  ret i64 -14

deflate:
  %n = call i64 @universe_compress_inflate(ptr %dst, i64 %dstcap, ptr %datap, i64 %compsize)
  %nneg = icmp slt i64 %n, 0
  br i1 %nneg, label %propn, label %verify

propn:
  ret i64 %n

verify:
  %sizeok = icmp eq i64 %n, %uncompsize
  br i1 %sizeok, label %retn, label %ret_parse

retn:
  ret i64 %n
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
