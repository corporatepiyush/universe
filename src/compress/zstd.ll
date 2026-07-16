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

; Zstandard (RFC 8878) frame decoder — RAW and RLE blocks (PARTIAL).
;
; SUPPORTED SUBSET (be explicit): this decoder parses a full Zstd frame header
; (magic, Frame_Header_Descriptor, Window_Descriptor, Dictionary_ID,
; Frame_Content_Size) and the per-block headers, and fully decodes blocks of
; type RAW (0, verbatim copy) and RLE (1, single byte repeated). It intentionally
; does NOT yet implement COMPRESSED blocks (type 2): the FSE (tANS) + Huffman
; entropy stages and the literals/sequences execution are a separate, larger
; wave. A COMPRESSED block returns UNSUPPORTED (-14) cleanly — never OOB. This
; is the honest state: RAW/RLE + frame/block framing are solid and interop-tested
; against the reference `zstd` tool; compressed decoding is future work.
;
; DESIGN:
;   * ONE-SHOT, whole-buffer, no allocation. Same compute/memory/IO split as the
;     LZ4 and DEFLATE decoders in this domain: the caller owns IO; this is pure
;     compute over memory it already owns, into a caller output buffer with an
;     explicit cap.
;   * Every field read advances a running input cursor that is bounds-checked
;     against slen BEFORE the read, and every output write is checked against
;     dcap BEFORE the write, so a truncated or malformed frame returns a negative
;     error and never reads or writes out of bounds (ASan-verified on hostile and
;     truncated input).
;   * Block header (RFC 8878 3.1.1.2): 3 little-endian bytes = Last_Block(bit0),
;     Block_Type(bits1-2), Block_Size(bits3-23). RAW copies Block_Size bytes;
;     RLE reads one byte and repeats it Block_Size times (via llvm.memset).
;   * The optional 4-byte content checksum is skipped (not verified) — a
;     correctness-neutral simplification noted here.
;   * Error convention (i64 return): decoded length on success, NEGATIVE on
;     failure: -1 NULL, -6 output exceeds cap (FULL), -13 malformed frame
;     (bad magic / reserved block type), -14 unsupported (compressed block),
;     -15 truncated input (IO).
;
; API:
;   i64 universe_compress_zstd_decode(ptr dst, i64 dcap, ptr src, i64 slen)

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

; Read a little-endian unsigned integer of %n bytes (1..8) from src+pos.
; Caller must have already bounds-checked pos+n <= slen.
define internal i64 @zstd_le(ptr %src, i64 %pos, i64 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %p = add i64 %pos, %i
  %bp = getelementptr inbounds i8, ptr %src, i64 %p
  %b8 = load i8, ptr %bp, align 1
  %b = zext i8 %b8 to i64
  %sh8 = mul i64 %i, 8
  %bs = shl i64 %b, %sh8
  %acc.n = or i64 %acc, %bs
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n
}

; --------------------------------------------------------------- decode
define i64 @universe_compress_zstd_decode(ptr %dst, i64 %dcap, ptr %src, i64 %slen) #1 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %anynull = or i1 %dn, %sn
  br i1 %anynull, label %ret_null, label %chkmagic

ret_null:
  ret i64 -1

chkmagic:
  ; need magic (4) + descriptor (1) = 5 bytes minimum
  %tooshort = icmp ult i64 %slen, 5
  br i1 %tooshort, label %err_trunc, label %magic

magic:
  %m = call i64 @zstd_le(ptr %src, i64 0, i64 4)
  %magicok = icmp eq i64 %m, 4247762216   ; 0xFD2FB528
  br i1 %magicok, label %desc, label %err_parse

desc:
  %dp = getelementptr inbounds i8, ptr %src, i64 4
  %desc8 = load i8, ptr %dp, align 1
  %descv = zext i8 %desc8 to i64
  %fcs_flag = lshr i64 %descv, 6
  %ss0 = lshr i64 %descv, 5
  %single_seg = and i64 %ss0, 1
  %ck0 = lshr i64 %descv, 2
  %checksum = and i64 %ck0, 1
  %dictid_flag = and i64 %descv, 3
  ; pos after descriptor
  br label %win

win:
  ; window descriptor present iff NOT single-segment
  %haswin = icmp eq i64 %single_seg, 0
  %pos_w0 = add i64 5, 1              ; if window present
  %pos_after_win = select i1 %haswin, i64 %pos_w0, i64 5
  ; dictionary id size: flag 0->0,1->1,2->2,3->4
  %dsz1 = icmp eq i64 %dictid_flag, 3
  %dsz_base = select i1 %dsz1, i64 4, i64 %dictid_flag
  %pos_after_dict = add i64 %pos_after_win, %dsz_base
  ; frame content size field size
  ; fcs_flag: 0 -> (single?1:0), 1 -> 2, 2 -> 4, 3 -> 8
  %fcs_is0 = icmp eq i64 %fcs_flag, 0
  %ss_one = select i1 %haswin, i64 0, i64 1   ; single-seg -> 1 byte when flag 0
  %fcs_pow = shl i64 1, %fcs_flag             ; 1,2,4,8 for flag 0,1,2,3
  %fcs_nonzero = select i1 %fcs_is0, i64 %ss_one, i64 %fcs_pow
  %pos_after_fcs = add i64 %pos_after_dict, %fcs_nonzero
  ; validate header fits
  %hdrok = icmp ule i64 %pos_after_fcs, %slen
  br i1 %hdrok, label %block.head, label %err_trunc

block.head:
  %pos = phi i64 [ %pos_after_fcs, %win ], [ %pos.n, %block.cont ]
  %op = phi i64 [ 0, %win ], [ %op.n, %block.cont ]
  ; block header = 3 bytes
  %bh_end = add i64 %pos, 3
  %bh_ok = icmp ule i64 %bh_end, %slen
  br i1 %bh_ok, label %block.read, label %err_trunc

block.read:
  %bh = call i64 @zstd_le(ptr %src, i64 %pos, i64 3)
  %last = and i64 %bh, 1
  %bt0 = lshr i64 %bh, 1
  %btype = and i64 %bt0, 3
  %bsize = lshr i64 %bh, 3
  %pos_bh = add i64 %pos, 3
  switch i64 %btype, label %err_parse [ i64 0, label %blk.raw
                                        i64 1, label %blk.rle
                                        i64 2, label %err_unsup ]

blk.raw:
  %raw_end = add i64 %pos_bh, %bsize
  %raw_src_ok = icmp ule i64 %raw_end, %slen
  br i1 %raw_src_ok, label %blk.raw2, label %err_trunc

blk.raw2:
  %raw_op_end = add i64 %op, %bsize
  %raw_dst_ok = icmp ule i64 %raw_op_end, %dcap
  br i1 %raw_dst_ok, label %blk.raw3, label %err_full

blk.raw3:
  %rdst = getelementptr inbounds i8, ptr %dst, i64 %op
  %rsrc = getelementptr inbounds i8, ptr %src, i64 %pos_bh
  call void @llvm.memcpy.p0.p0.i64(ptr %rdst, ptr %rsrc, i64 %bsize, i1 false)
  br label %block.cont

blk.rle:
  ; need 1 source byte
  %rle_src_ok = icmp ult i64 %pos_bh, %slen
  br i1 %rle_src_ok, label %blk.rle2, label %err_trunc

blk.rle2:
  %rle_op_end = add i64 %op, %bsize
  %rle_dst_ok = icmp ule i64 %rle_op_end, %dcap
  br i1 %rle_dst_ok, label %blk.rle3, label %err_full

blk.rle3:
  %rlebp = getelementptr inbounds i8, ptr %src, i64 %pos_bh
  %rleb = load i8, ptr %rlebp, align 1
  %rledst = getelementptr inbounds i8, ptr %dst, i64 %op
  call void @llvm.memset.p0.i64(ptr %rledst, i8 %rleb, i64 %bsize, i1 false)
  %pos_bh1 = add i64 %pos_bh, 1
  br label %block.cont

block.cont:
  %pos.n = phi i64 [ %raw_end, %blk.raw3 ], [ %pos_bh1, %blk.rle3 ]
  %op.n = phi i64 [ %raw_op_end, %blk.raw3 ], [ %rle_op_end, %blk.rle3 ]
  %islast = icmp eq i64 %last, 1
  br i1 %islast, label %ret_op, label %block.head

ret_op:
  ret i64 %op.n

err_trunc:
  ret i64 -15

err_full:
  ret i64 -6

err_parse:
  ret i64 -13

err_unsup:
  ret i64 -14
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
