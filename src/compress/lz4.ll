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

; LZ4 block codec: raw-block decompress + a single-pass greedy compressor.
;
; DESIGN:
;   * ONE-SHOT, whole-buffer. The whole compressed block sits in a caller
;     buffer; the whole decoded output goes to a caller buffer with an explicit
;     cap. No streaming, no per-item allocation. This is the compute/memory/IO
;     split: the caller owns IO (fill input, drain output); this code is pure
;     compute over memory it already owns. The only heap use is a single scratch
;     hash table in the COMPRESSOR (freed before return).
;   * LZ4 has NO entropy coder — it is a byte-oriented LZ77 with a compact token.
;     A block is a series of SEQUENCES. Each sequence is:
;       token  = (literal_len_nibble << 4) | match_len_nibble
;       [extended literal length]  0xFF chain when nibble == 15
;       literals                   literal_len raw bytes
;       offset                     2-byte little-endian, >= 1  (absent on the
;                                  final sequence, which is literals-only)
;       [extended match length]    0xFF chain when nibble == 15
;     The stored match length is (real_match_len - 4): min-match is 4.
;   * DECODE (@universe_compress_lz4_decode) processes sequences left to right.
;     The final sequence has no offset/match — the block always ends with a
;     literal run (last 5 bytes are literals per the format), so when the input
;     is exhausted right after a literal copy we are DONE. Every read is bounds
;     checked against slen and every write against dcap BEFORE it happens, so a
;     truncated or malformed block returns a negative error and never reads or
;     writes out of bounds (the property AddressSanitizer verifies on hostile
;     input). Match copy is OVERLAP-CORRECT: offset >= len cannot overlap so we
;     emit one llvm.memcpy; offset < len (RLE run, e.g. offset==1) is copied
;     forward byte-by-byte so freshly written bytes feed the copy.
;   * ENCODE (@universe_compress_lz4_encode) is a single greedy pass with a
;     4-byte-hash match table (2^16 entries of i32, storing position+1 so 0 is
;     "empty"). At each position we hash the next 4 bytes, look up the most
;     recent same-hash position, and if it is within the 64 KiB window and the
;     4 bytes truly match we extend the match forward and emit a sequence;
;     otherwise we advance one byte. Matches are confined to end >= 5 bytes
;     before EOF (matchlimit) and are only searched while > 12 bytes remain
;     (mflimit), so the final 5 bytes are always literals — a spec-valid block
;     that the reference LZ4 decoder also accepts. Inputs < 13 bytes are emitted
;     as a single literal run.
;   * Error convention (i64 return): decoded/compressed length on success,
;     NEGATIVE on failure: -1 NULL, -2 OOM (table alloc), -6 output exceeds cap
;     (FULL), -13 malformed block (bad/zero offset), -15 truncated input (IO).
;
; API:
;   i64 universe_compress_lz4_decode(ptr dst, i64 dcap, ptr src, i64 slen)
;   i64 universe_compress_lz4_encode(ptr dst, i64 dcap, ptr src, i64 slen)
;   i64 universe_compress_lz4_bound(i64 slen)   ; worst-case compressed size

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

; --------------------------------------------------------------- decode
define i64 @universe_compress_lz4_decode(ptr %dst, i64 %dcap, ptr %src, i64 %slen) #2 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %ret_null, label %chkempty

ret_null:
  ret i64 -1

chkempty:
  %empty = icmp eq i64 %slen, 0
  br i1 %empty, label %ret_zero, label %seq.head

ret_zero:
  ret i64 0

seq.head:
  %ip = phi i64 [ 0, %chkempty ], [ %ip.next, %match.done ]
  %op = phi i64 [ 0, %chkempty ], [ %op.next, %match.done ]
  %atend = icmp uge i64 %ip, %slen
  br i1 %atend, label %ret_op, label %read.token

ret_op:
  ret i64 %op

read.token:
  %tp = getelementptr inbounds i8, ptr %src, i64 %ip
  %tok8 = load i8, ptr %tp, align 1
  %tok = zext i8 %tok8 to i64
  %ip1 = add i64 %ip, 1
  %litnib = lshr i64 %tok, 4
  %litext = icmp eq i64 %litnib, 15
  br i1 %litext, label %ll.head, label %lit.len.done

ll.head:
  %ll = phi i64 [ 15, %read.token ], [ %ll.next, %ll.body ]
  %llip = phi i64 [ %ip1, %read.token ], [ %llip.next, %ll.body ]
  %llavail = icmp ult i64 %llip, %slen
  br i1 %llavail, label %ll.body, label %err_trunc

ll.body:
  %ebp = getelementptr inbounds i8, ptr %src, i64 %llip
  %eb8 = load i8, ptr %ebp, align 1
  %eb = zext i8 %eb8 to i64
  %ll.next = add i64 %ll, %eb
  %llip.next = add i64 %llip, 1
  %is255 = icmp eq i8 %eb8, -1
  br i1 %is255, label %ll.head, label %lit.len.done

lit.len.done:
  %litlen = phi i64 [ %litnib, %read.token ], [ %ll.next, %ll.body ]
  %ipL = phi i64 [ %ip1, %read.token ], [ %llip.next, %ll.body ]
  %srcrem = sub i64 %slen, %ipL
  %litoosrc = icmp ugt i64 %litlen, %srcrem
  br i1 %litoosrc, label %err_trunc, label %lit.room

lit.room:
  %dstrem = sub i64 %dcap, %op
  %litoodst = icmp ugt i64 %litlen, %dstrem
  br i1 %litoodst, label %err_full, label %lit.copy

lit.copy:
  %ldst = getelementptr inbounds i8, ptr %dst, i64 %op
  %lsrc = getelementptr inbounds i8, ptr %src, i64 %ipL
  call void @llvm.memcpy.p0.p0.i64(ptr %ldst, ptr %lsrc, i64 %litlen, i1 false)
  %opA = add i64 %op, %litlen
  %ipM = add i64 %ipL, %litlen
  %endnow = icmp uge i64 %ipM, %slen
  br i1 %endnow, label %ret_opA, label %match.off

ret_opA:
  ret i64 %opA

match.off:
  %offrem = sub i64 %slen, %ipM
  %offshort = icmp ult i64 %offrem, 2
  br i1 %offshort, label %err_trunc, label %read.off

read.off:
  %o0p = getelementptr inbounds i8, ptr %src, i64 %ipM
  %o0v = load i8, ptr %o0p, align 1
  %o0z = zext i8 %o0v to i64
  %ipM1 = add i64 %ipM, 1
  %o1p = getelementptr inbounds i8, ptr %src, i64 %ipM1
  %o1v = load i8, ptr %o1p, align 1
  %o1z = zext i8 %o1v to i64
  %o1s = shl i64 %o1z, 8
  %offset = or i64 %o0z, %o1s
  %ipO = add i64 %ipM, 2
  %offzero = icmp eq i64 %offset, 0
  br i1 %offzero, label %err_parse, label %off.ok

off.ok:
  %offbad = icmp ugt i64 %offset, %opA
  br i1 %offbad, label %err_parse, label %match.len

match.len:
  %mnib = and i64 %tok, 15
  %mext = icmp eq i64 %mnib, 15
  br i1 %mext, label %ml.head, label %match.len.done

ml.head:
  %ml = phi i64 [ 15, %match.len ], [ %ml.next, %ml.body ]
  %mlip = phi i64 [ %ipO, %match.len ], [ %mlip.next, %ml.body ]
  %mlavail = icmp ult i64 %mlip, %slen
  br i1 %mlavail, label %ml.body, label %err_trunc

ml.body:
  %mbp = getelementptr inbounds i8, ptr %src, i64 %mlip
  %mb8 = load i8, ptr %mbp, align 1
  %mb = zext i8 %mb8 to i64
  %ml.next = add i64 %ml, %mb
  %mlip.next = add i64 %mlip, 1
  %mis255 = icmp eq i8 %mb8, -1
  br i1 %mis255, label %ml.head, label %match.len.done

match.len.done:
  %mlraw = phi i64 [ %mnib, %match.len ], [ %ml.next, %ml.body ]
  %ipN = phi i64 [ %ipO, %match.len ], [ %mlip.next, %ml.body ]
  %matchlen = add i64 %mlraw, 4
  %mdstrem = sub i64 %dcap, %opA
  %moodst = icmp ugt i64 %matchlen, %mdstrem
  br i1 %moodst, label %err_full, label %match.copy

match.copy:
  %from = sub i64 %opA, %offset
  %nonoverlap = icmp uge i64 %offset, %matchlen
  br i1 %nonoverlap, label %m.fast, label %m.byte

m.fast:
  %mfsrc = getelementptr inbounds i8, ptr %dst, i64 %from
  %mfdst = getelementptr inbounds i8, ptr %dst, i64 %opA
  call void @llvm.memcpy.p0.p0.i64(ptr %mfdst, ptr %mfsrc, i64 %matchlen, i1 false)
  br label %match.done

m.byte:
  br label %mb.loop

mb.loop:
  %k = phi i64 [ 0, %m.byte ], [ %k.n, %mb.loop ]
  %kf = add i64 %from, %k
  %kt = add i64 %opA, %k
  %ksp = getelementptr inbounds i8, ptr %dst, i64 %kf
  %ksv = load i8, ptr %ksp, align 1
  %kdp = getelementptr inbounds i8, ptr %dst, i64 %kt
  store i8 %ksv, ptr %kdp, align 1
  %k.n = add i64 %k, 1
  %kmore = icmp ult i64 %k.n, %matchlen
  br i1 %kmore, label %mb.loop, label %match.done

match.done:
  %ip.next = phi i64 [ %ipN, %m.fast ], [ %ipN, %mb.loop ]
  %op.next = add i64 %opA, %matchlen
  br label %seq.head

err_trunc:
  ret i64 -15

err_full:
  ret i64 -6

err_parse:
  ret i64 -13
}

; ------------------------------------------------ length-extension writer
; Append an 0xFF-chained extended length; returns new op, or -6 on FULL.
define internal i64 @lz4_put_ext(ptr %dst, i64 %dcap, i64 %op, i64 %val) #1 {
entry:
  br label %loop

loop:
  %v = phi i64 [ %val, %entry ], [ %v.n, %do255 ]
  %o = phi i64 [ %op, %entry ], [ %o1, %do255 ]
  %big = icmp uge i64 %v, 255
  br i1 %big, label %chk255, label %chklast

chk255:
  %room = icmp ult i64 %o, %dcap
  br i1 %room, label %do255, label %full

do255:
  %p255 = getelementptr inbounds i8, ptr %dst, i64 %o
  store i8 -1, ptr %p255, align 1
  %o1 = add i64 %o, 1
  %v.n = sub i64 %v, 255
  br label %loop

chklast:
  %room2 = icmp ult i64 %o, %dcap
  br i1 %room2, label %dolast, label %full

dolast:
  %plast = getelementptr inbounds i8, ptr %dst, i64 %o
  %v8 = trunc i64 %v to i8
  store i8 %v8, ptr %plast, align 1
  %o2 = add i64 %o, 1
  ret i64 %o2

full:
  ret i64 -6
}

; ------------------------------------------------ literals-only emitter
; Emit a token with match nibble 0, extended literal length and the raw
; literals. Returns new op, or -6 on FULL.
define internal i64 @lz4_emit_literals(ptr %dst, i64 %dcap, i64 %op, ptr %src,
                                       i64 %from, i64 %count) #1 {
entry:
  %big = icmp uge i64 %count, 15
  %nib = select i1 %big, i64 15, i64 %count
  %tokv = shl i64 %nib, 4
  %troom = icmp ult i64 %op, %dcap
  br i1 %troom, label %wtok, label %full

full:
  ret i64 -6

wtok:
  %tp = getelementptr inbounds i8, ptr %dst, i64 %op
  %t8 = trunc i64 %tokv to i8
  store i8 %t8, ptr %tp, align 1
  %op1 = add i64 %op, 1
  br i1 %big, label %ext, label %cp.pre

ext:
  %extval = sub i64 %count, 15
  %ope = call i64 @lz4_put_ext(ptr %dst, i64 %dcap, i64 %op1, i64 %extval)
  %ef = icmp slt i64 %ope, 0
  br i1 %ef, label %full, label %cp.pre

cp.pre:
  %opc = phi i64 [ %op1, %wtok ], [ %ope, %ext ]
  %room = sub i64 %dcap, %opc
  %over = icmp ugt i64 %count, %room
  br i1 %over, label %full, label %cp

cp:
  %d = getelementptr inbounds i8, ptr %dst, i64 %opc
  %s = getelementptr inbounds i8, ptr %src, i64 %from
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %count, i1 false)
  %opf = add i64 %opc, %count
  ret i64 %opf
}

; --------------------------------------------------------------- bound
; Worst-case compressed size for slen input bytes: a single all-literal
; sequence is 1 token + ceil((slen-15)/255) length bytes + slen literals.
define i64 @universe_compress_lz4_bound(i64 %slen) #3 {
entry:
  %t = add i64 %slen, 16
  %extra = udiv i64 %slen, 255
  %r = add i64 %t, %extra
  ret i64 %r
}

; --------------------------------------------------------------- encode
define i64 @universe_compress_lz4_encode(ptr %dst, i64 %dcap, ptr %src, i64 %slen) #4 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %ret_null, label %sizechk

ret_null:
  ret i64 -1

sizechk:
  %small = icmp ult i64 %slen, 13
  br i1 %small, label %all.lit, label %setup

all.lit:
  %r0 = call i64 @lz4_emit_literals(ptr %dst, i64 %dcap, i64 0, ptr %src, i64 0, i64 %slen)
  ret i64 %r0

setup:
  %tbl = call ptr @malloc(i64 262144)
  %tn = icmp eq ptr %tbl, null
  br i1 %tn, label %ret_oom, label %setup2

ret_oom:
  ret i64 -2

setup2:
  call void @llvm.memset.p0.i64(ptr %tbl, i8 0, i64 262144, i1 false)
  %mflimit = sub i64 %slen, 12
  %matchlimit = sub i64 %slen, 5
  br label %scan.head

scan.head:
  %ip = phi i64 [ 0, %setup2 ], [ %ipp1, %no.match ], [ %ip.after, %emit.done ]
  %op = phi i64 [ 0, %setup2 ], [ %op, %no.match ], [ %op.after, %emit.done ]
  %anchor = phi i64 [ 0, %setup2 ], [ %anchor, %no.match ], [ %ip.after, %emit.done ]
  %go = icmp ult i64 %ip, %mflimit
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
  %ipp1 = add i64 %ip, 1
  br label %scan.head

ext.head:
  %ml = phi i64 [ 4, %cmp4 ], [ %ml.n, %ext.body ]
  %ipm = add i64 %ip, %ml
  %inb = icmp ult i64 %ipm, %matchlimit
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
  %litlen = sub i64 %ip, %anchor
  %mlcode = sub i64 %ml, 4
  %litbig = icmp uge i64 %litlen, 15
  %litnib = select i1 %litbig, i64 15, i64 %litlen
  %mbig = icmp uge i64 %mlcode, 15
  %mnibv = select i1 %mbig, i64 15, i64 %mlcode
  %tokhi = shl i64 %litnib, 4
  %tokv = or i64 %tokhi, %mnibv
  %troom = icmp ult i64 %op, %dcap
  br i1 %troom, label %wtok, label %ret_full_free

wtok:
  %tokp = getelementptr inbounds i8, ptr %dst, i64 %op
  %tok8 = trunc i64 %tokv to i8
  store i8 %tok8, ptr %tokp, align 1
  %op.t = add i64 %op, 1
  br i1 %litbig, label %wlitext, label %wlits

wlitext:
  %litextval = sub i64 %litlen, 15
  %op.le = call i64 @lz4_put_ext(ptr %dst, i64 %dcap, i64 %op.t, i64 %litextval)
  %lefull = icmp slt i64 %op.le, 0
  br i1 %lefull, label %ret_full_free, label %wlits

wlits:
  %op.lit = phi i64 [ %op.t, %wtok ], [ %op.le, %wlitext ]
  %litroom = sub i64 %dcap, %op.lit
  %litover = icmp ugt i64 %litlen, %litroom
  br i1 %litover, label %ret_full_free, label %wlits.cp

wlits.cp:
  %ldst = getelementptr inbounds i8, ptr %dst, i64 %op.lit
  %lsrc = getelementptr inbounds i8, ptr %src, i64 %anchor
  call void @llvm.memcpy.p0.p0.i64(ptr %ldst, ptr %lsrc, i64 %litlen, i1 false)
  %op.al = add i64 %op.lit, %litlen
  %need2 = add i64 %op.al, 2
  %offroom = icmp ule i64 %need2, %dcap
  br i1 %offroom, label %woff, label %ret_full_free

woff:
  %oflo = trunc i64 %dist to i8
  %ofhi64 = lshr i64 %dist, 8
  %ofhi = trunc i64 %ofhi64 to i8
  %o0dp = getelementptr inbounds i8, ptr %dst, i64 %op.al
  store i8 %oflo, ptr %o0dp, align 1
  %op.a1 = add i64 %op.al, 1
  %o1dp = getelementptr inbounds i8, ptr %dst, i64 %op.a1
  store i8 %ofhi, ptr %o1dp, align 1
  %op.o = add i64 %op.al, 2
  br i1 %mbig, label %wmext, label %emit.done

wmext:
  %mextval = sub i64 %mlcode, 15
  %op.me = call i64 @lz4_put_ext(ptr %dst, i64 %dcap, i64 %op.o, i64 %mextval)
  %mefull = icmp slt i64 %op.me, 0
  br i1 %mefull, label %ret_full_free, label %emit.done

emit.done:
  %op.after = phi i64 [ %op.o, %woff ], [ %op.me, %wmext ]
  %ip.after = add i64 %ip, %ml
  br label %scan.head

final:
  %flen = sub i64 %slen, %anchor
  %fr = call i64 @lz4_emit_literals(ptr %dst, i64 %dcap, i64 %op, ptr %src, i64 %anchor, i64 %flen)
  call void @free(ptr %tbl)
  ret i64 %fr

ret_full_free:
  call void @free(ptr %tbl)
  ret i64 -6
}

attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #4 = { nounwind }
