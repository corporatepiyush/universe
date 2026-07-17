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

; Parquet column-encoding decoder kernels (parquet-format Encodings.md).
; Pure compute over caller byte buffers; the only allocations are private
; scratch length arrays in the two DELTA byte-array kernels (freed on exit).
;
; DESIGN:
;   * SELF-CONTAINED — no cross-domain calls. A local ULEB128 reader
;     (@pq_uleb), a zig-zag reader (@pq_zigzag) and an LSB-first bit reader
;     (@pq_read_bits) are inlined here rather than depending on encoding/varint,
;     so this file stays a leaf of the dependency graph.
;   * UNTRUSTED INPUT. Every read is bounds-checked against `dlen`; a run
;     header or an over-long value that would step past the buffer returns
;     INVALID_ARG (8). Shifts are guarded so a hostile stream can never form a
;     poison `shl x, >=64` (hazard #10); the bit reader zero-pads past the end
;     rather than reading OOB (hazard #15). Every offset add/mul is checked with
;     llvm.uadd/umul.with.overflow.i64 → SIZE_OVERFLOW (3).
;   * The RLE/bit-packing hybrid outer loop always consumes at least the 1-byte
;     run header per iteration, so `pos` strictly increases and a malformed
;     never-completing stream terminates by running out of buffer (→ 8) rather
;     than hanging.
;   * DELTA core (@pq_delta_core) is shared by delta_binary_packed,
;     delta_length_byte_array and delta_byte_array. It requires the header
;     `total` count to fit `out_cap` (else 8): this bounds the decode loop by a
;     real allocation and defeats the width-0 miniblock DoS (a huge `total` with
;     zero-width deltas would otherwise spin). It also reports bytes consumed so
;     the two byte-array kernels can find the data that follows the length block.
;   * The `raw` bit-packed delta is an UNSIGNED value; value += min_delta + raw
;     with wrapping adds (no nuw/nsw) per the spec's two's-complement semantics.
;   * bit_width 0 is explicit: the hybrid emits all-zero output with no data
;     bytes; a zero-width delta miniblock contributes only min_delta.
;
; Physical type codes (phys_type, matches parquet Type enum):
;   0 BOOLEAN  1 INT32  2 INT64  3 INT96  4 FLOAT  5 DOUBLE
;   6 BYTE_ARRAY  7 FIXED_LEN_BYTE_ARRAY
;
; Output layouts:
;   rle_hybrid            → u32[count]      (levels / dictionary indices)
;   plain BOOLEAN         → i8[count]       (0/1)
;   plain INT32/FLOAT     → 4*count LE bytes (copied verbatim)
;   plain INT64/DOUBLE    → 8*count LE bytes
;   plain INT96           → 12*count LE bytes
;   plain BYTE_ARRAY      → [u32 len][bytes] repeated
;   plain FIXED_LEN_BYTE_ARRAY → type_len*count bytes
;   delta_binary_packed   → i64[out_count]
;   delta_length/byte_array → [u32 len][bytes] repeated (reconstructed values)
;   byte_stream_split     → width*count bytes (values reassembled, LE)
;
; API:
;   i32 universe_docparse_parquet_rle_hybrid(ptr data,i64 dlen,i32 bit_width,i64 count,ptr out_u32)
;   i32 universe_docparse_parquet_plain(ptr data,i64 dlen,i32 phys_type,i32 type_len,i64 count,ptr out,i64 out_cap,ptr out_used)
;   i32 universe_docparse_parquet_delta_binary_packed(ptr data,i64 dlen,ptr out_i64,i64 out_cap,ptr out_count)
;   i32 universe_docparse_parquet_delta_length_byte_array(ptr data,i64 dlen,i64 count,ptr out,i64 out_cap,ptr out_used)
;   i32 universe_docparse_parquet_delta_byte_array(ptr data,i64 dlen,i64 count,ptr out,i64 out_cap,ptr out_used)
;   i32 universe_docparse_parquet_byte_stream_split(ptr data,i64 dlen,i32 width,i64 count,ptr out)

declare i64 @llvm.umin.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"

; ============================================================ ULEB128 reader
; Reads one ULEB128 at *pos from data[..dlen); on success updates *pos and
; *out and returns true. Returns false on truncation or >64-bit magnitude.
define internal i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos, ptr %out) #2 {
entry:
  %p0 = load i64, ptr %pos, align 8
  br label %loop

loop:
  %i = phi i64 [ %p0, %entry ], [ %i.next, %cont ]
  %shift = phi i64 [ 0, %entry ], [ %shift.next, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %cont ]
  %inb = icmp ult i64 %i, %dlen
  br i1 %inb, label %read, label %fail, !prof !0

read:
  %bp = getelementptr inbounds nuw i8, ptr %data, i64 %i
  %b = load i8, ptr %bp, align 1
  %bz = zext i8 %b to i64
  %low = and i64 %bz, 127
  %sh.ok = icmp ult i64 %shift, 64
  %sh.safe = select i1 %sh.ok, i64 %shift, i64 0
  %piece.raw = shl i64 %low, %sh.safe
  %piece = select i1 %sh.ok, i64 %piece.raw, i64 0
  %acc.next = or i64 %acc, %piece
  %hi = and i64 %bz, 128
  %more = icmp ne i64 %hi, 0
  %i.next = add nuw i64 %i, 1
  br i1 %more, label %cont, label %done

cont:
  %shift.next = add nuw nsw i64 %shift, 7
  %toobig = icmp uge i64 %shift.next, 64
  br i1 %toobig, label %fail, label %loop, !prof !0

done:
  store i64 %acc.next, ptr %out, align 8
  store i64 %i.next, ptr %pos, align 8
  ret i1 true

fail:                                             ; cold
  ret i1 false
}

; =========================================================== zig-zag reader
define internal i1 @pq_zigzag(ptr %data, i64 %dlen, ptr %pos, ptr %out) #2 {
entry:
  %u.slot = alloca i64, align 8
  %ok = call i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos, ptr %u.slot)
  br i1 %ok, label %dec, label %fail, !prof !1

dec:
  %u = load i64, ptr %u.slot, align 8
  %sh = lshr i64 %u, 1
  %lsb = and i64 %u, 1
  %neg = sub nsw i64 0, %lsb
  %r = xor i64 %sh, %neg
  store i64 %r, ptr %out, align 8
  ret i1 true

fail:                                             ; cold
  ret i1 false
}

; ==================================================== LSB-first bit reader
; Reads `width` (0..64) bits starting at absolute bit offset `bitpos`, LSB
; first, zero-padding any bits past `dlen`. Pure leaf, no state mutation.
define internal i64 @pq_read_bits(ptr %data, i64 %dlen, i64 %bitpos, i32 %width) #0 {
entry:
  %w = zext i32 %width to i64
  %wz = icmp eq i64 %w, 0
  br i1 %wz, label %ret0, label %init

init:
  %bytepos0 = lshr i64 %bitpos, 3
  %bitoff0 = and i64 %bitpos, 7
  br label %loop

loop:
  %bytepos = phi i64 [ %bytepos0, %init ], [ %bytepos.n, %body ]
  %bitoff = phi i64 [ %bitoff0, %init ], [ %bitoff.n, %body ]
  %got = phi i64 [ 0, %init ], [ %got.n, %body ]
  %result = phi i64 [ 0, %init ], [ %result.n, %body ]
  %done = icmp uge i64 %got, %w
  br i1 %done, label %fin, label %chk

chk:
  %oob = icmp uge i64 %bytepos, %dlen
  br i1 %oob, label %fin, label %body

body:
  %avail = sub nuw i64 8, %bitoff
  %remain = sub nuw i64 %w, %got
  %take = call i64 @llvm.umin.i64(i64 %remain, i64 %avail)
  %m1 = shl i64 1, %take
  %mask = add i64 %m1, -1
  %bp = getelementptr inbounds nuw i8, ptr %data, i64 %bytepos
  %bb = load i8, ptr %bp, align 1
  %bz = zext i8 %bb to i64
  %shifted = lshr i64 %bz, %bitoff
  %bits = and i64 %shifted, %mask
  %contrib = shl i64 %bits, %got
  %result.n = or i64 %result, %contrib
  %got.n = add nuw i64 %got, %take
  %bitoff.tmp = add nuw i64 %bitoff, %take
  %wrap = icmp eq i64 %bitoff.tmp, 8
  %bitoff.n = select i1 %wrap, i64 0, i64 %bitoff.tmp
  %bytepos.inc = add nuw i64 %bytepos, 1
  %bytepos.n = select i1 %wrap, i64 %bytepos.inc, i64 %bytepos
  br label %loop

fin:
  ret i64 %result

ret0:
  ret i64 0
}

; ===================================================== RLE / bit-pack hybrid
define i32 @universe_docparse_parquet_rle_hybrid(ptr %data, i64 %dlen, i32 %bit_width, i64 %count, ptr %out) #1 {
entry:
  %cz = icmp eq i64 %count, 0
  br i1 %cz, label %ok, label %notzero

notzero:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %nn = or i1 %dn, %on
  br i1 %nn, label %err.null, label %chkbw

chkbw:
  %bwlt = icmp slt i32 %bit_width, 0
  %bwgt = icmp sgt i32 %bit_width, 32
  %bwbad = or i1 %bwlt, %bwgt
  br i1 %bwbad, label %err.arg, label %chkzero

chkzero:
  %bw = zext i32 %bit_width to i64
  %iszero = icmp eq i64 %bw, 0
  br i1 %iszero, label %zerowidth, label %mainloop.pre

zerowidth:
  ; all values are 0; no data consumed. out is u32[count].
  %zbytes = shl i64 %count, 2
  call void @llvm.memset.p0.i64(ptr %out, i8 0, i64 %zbytes, i1 false)
  ret i32 0

mainloop.pre:
  %nbytes = udiv i64 %bw, 8
  %rem8 = urem i64 %bw, 8
  %rnz = icmp ne i64 %rem8, 0
  %rext = zext i1 %rnz to i64
  %nb = add i64 %nbytes, %rext           ; ceil(bw/8), 1..4
  %pos.slot = alloca i64, align 8
  store i64 0, ptr %pos.slot, align 8
  %hdr.slot = alloca i64, align 8
  br label %run.head

run.head:
  %produced = phi i64 [ 0, %mainloop.pre ], [ %produced.next, %run.cont ]
  %pmore = icmp ult i64 %produced, %count
  br i1 %pmore, label %run.body, label %ok

run.body:
  %okh = call i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos.slot, ptr %hdr.slot)
  br i1 %okh, label %run.decode, label %err.arg, !prof !1

run.decode:
  %hdr = load i64, ptr %hdr.slot, align 8
  %isbp = and i64 %hdr, 1
  %hval = lshr i64 %hdr, 1
  ; h&1==0 → bit-packed run; h&1==1 → RLE run.
  %is_bitpacked = icmp eq i64 %isbp, 0
  br i1 %is_bitpacked, label %packed.run, label %rle.run

; -------- RLE run: `hval` repeats of a value stored in `nb` LE bytes --------
rle.run:
  %rpos = load i64, ptr %pos.slot, align 8
  %rend = add i64 %rpos, %nb
  %rovf = icmp ult i64 %rend, %rpos
  br i1 %rovf, label %err.size, label %rle.bounds

rle.bounds:
  %rtoobig = icmp ugt i64 %rend, %dlen
  br i1 %rtoobig, label %err.arg, label %rle.vloop

rle.vloop:
  %vk = phi i64 [ 0, %rle.bounds ], [ %vk.next, %rle.vbody ]
  %vacc = phi i64 [ 0, %rle.bounds ], [ %vacc.next, %rle.vbody ]
  %vk.done = icmp uge i64 %vk, %nb
  br i1 %vk.done, label %rle.store.pre, label %rle.vbody

rle.vbody:
  %vidx = add i64 %rpos, %vk
  %vbp = getelementptr inbounds nuw i8, ptr %data, i64 %vidx
  %vb = load i8, ptr %vbp, align 1
  %vbz = zext i8 %vb to i64
  %vshift = shl i64 %vk, 3               ; 8*k, k<=3 -> <=24
  %vpiece = shl i64 %vbz, %vshift
  %vacc.next = or i64 %vacc, %vpiece
  %vk.next = add nuw i64 %vk, 1
  br label %rle.vloop

rle.store.pre:
  ; n = min(hval, count - produced)
  %rremain = sub i64 %count, %produced
  %rn = call i64 @llvm.umin.i64(i64 %hval, i64 %rremain)
  %rval32 = trunc i64 %vacc to i32
  br label %rle.sloop

rle.sloop:
  %sk = phi i64 [ 0, %rle.store.pre ], [ %sk.next, %rle.sbody ]
  %sk.done = icmp uge i64 %sk, %rn
  br i1 %sk.done, label %rle.fin, label %rle.sbody

rle.sbody:
  %sidx = add i64 %produced, %sk
  %sop = getelementptr inbounds nuw i32, ptr %out, i64 %sidx
  store i32 %rval32, ptr %sop, align 4
  %sk.next = add nuw i64 %sk, 1
  br label %rle.sloop

rle.fin:
  store i64 %rend, ptr %pos.slot, align 8
  %produced.rle = add i64 %produced, %rn
  br label %run.cont

; -------- bit-packed run: (hval) groups of 8 values, each bw bits ----------
packed.run:
  %tot.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %hval, i64 8)
  %total = extractvalue { i64, i1 } %tot.chk, 0
  %tot.ovf = extractvalue { i64, i1 } %tot.chk, 1
  br i1 %tot.ovf, label %err.size, label %packed.need

packed.need:
  %need.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %hval, i64 %bw)
  %need = extractvalue { i64, i1 } %need.chk, 0
  %need.ovf = extractvalue { i64, i1 } %need.chk, 1
  br i1 %need.ovf, label %err.size, label %packed.bounds

packed.bounds:
  %ppos = load i64, ptr %pos.slot, align 8
  %pend = add i64 %ppos, %need
  %povf = icmp ult i64 %pend, %ppos
  br i1 %povf, label %err.size, label %packed.bchk

packed.bchk:
  %ptoobig = icmp ugt i64 %pend, %dlen
  br i1 %ptoobig, label %err.arg, label %packed.decode

packed.decode:
  %pbitbase = shl i64 %ppos, 3
  %bwi32 = trunc i64 %bw to i32
  br label %pk.loop

pk.loop:
  %pj = phi i64 [ 0, %packed.decode ], [ %pj.next, %pk.cont ]
  %pprod = phi i64 [ %produced, %packed.decode ], [ %pprod.next, %pk.cont ]
  %pj.done = icmp uge i64 %pj, %total
  br i1 %pj.done, label %packed.fin, label %pk.body

pk.body:
  %joff = mul i64 %pj, %bw
  %jbit = add i64 %pbitbase, %joff
  %pv = call i64 @pq_read_bits(ptr %data, i64 %dlen, i64 %jbit, i32 %bwi32)
  %pstore = icmp ult i64 %pprod, %count
  br i1 %pstore, label %pk.store, label %pk.cont

pk.store:
  %pv32 = trunc i64 %pv to i32
  %pop = getelementptr inbounds nuw i32, ptr %out, i64 %pprod
  store i32 %pv32, ptr %pop, align 4
  %pprod.inc = add i64 %pprod, 1
  br label %pk.cont

pk.cont:
  %pprod.next = phi i64 [ %pprod.inc, %pk.store ], [ %pprod, %pk.body ]
  %pj.next = add nuw i64 %pj, 1
  br label %pk.loop

packed.fin:
  store i64 %pend, ptr %pos.slot, align 8
  br label %run.cont

run.cont:
  %produced.next = phi i64 [ %produced.rle, %rle.fin ], [ %pprod, %packed.fin ]
  br label %run.head

ok:
  ret i32 0

err.null:
  ret i32 1

err.size:
  ret i32 3

err.arg:
  ret i32 8
}

; ================================================================== PLAIN
define i32 @universe_docparse_parquet_plain(ptr %data, i64 %dlen, i32 %phys_type, i32 %type_len, i64 %count, ptr %out, i64 %out_cap, ptr %out_used) #1 {
entry:
  %un = icmp eq ptr %out_used, null
  br i1 %un, label %err.null, label %chkcount

chkcount:
  %cz = icmp eq i64 %count, 0
  br i1 %cz, label %empty, label %notzero

empty:
  store i64 0, ptr %out_used, align 8
  ret i32 0

notzero:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %nn = or i1 %dn, %on
  br i1 %nn, label %err.null, label %dispatch

dispatch:
  switch i32 %phys_type, label %err.arg [
    i32 0, label %do.bool
    i32 1, label %fx.i32
    i32 4, label %fx.i32
    i32 2, label %fx.i64
    i32 5, label %fx.i64
    i32 3, label %fx.i96
    i32 6, label %do.barray
    i32 7, label %do.flba
  ]

; ---- fixed-width types funnel into a common verbatim copy ----
fx.i32:
  br label %fixed
fx.i64:
  br label %fixed
fx.i96:
  br label %fixed

fixed:
  %fsize = phi i64 [ 4, %fx.i32 ], [ 8, %fx.i64 ], [ 12, %fx.i96 ]
  %fneed.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %fsize, i64 %count)
  %fneed = extractvalue { i64, i1 } %fneed.chk, 0
  %fneed.ovf = extractvalue { i64, i1 } %fneed.chk, 1
  br i1 %fneed.ovf, label %err.size, label %fixed.bounds

fixed.bounds:
  %fin.big = icmp ugt i64 %fneed, %dlen
  br i1 %fin.big, label %err.arg, label %fixed.cap

fixed.cap:
  %fout.big = icmp ugt i64 %fneed, %out_cap
  br i1 %fout.big, label %err.full, label %fixed.copy

fixed.copy:
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %data, i64 %fneed, i1 false)
  store i64 %fneed, ptr %out_used, align 8
  ret i32 0

; ---- BOOLEAN: bit-packed 1-bit LSB-first, one i8 (0/1) per value ----
do.bool:
  %needbits.hi = add i64 %count, 7
  %needbytes = lshr i64 %needbits.hi, 3
  %bbig = icmp ugt i64 %needbytes, %dlen
  br i1 %bbig, label %err.arg, label %bool.cap

bool.cap:
  %bcap = icmp ugt i64 %count, %out_cap
  br i1 %bcap, label %err.full, label %bool.loop

bool.loop:
  %bi = phi i64 [ 0, %bool.cap ], [ %bi.next, %bool.body ]
  %bi.done = icmp uge i64 %bi, %count
  br i1 %bi.done, label %bool.fin, label %bool.body

bool.body:
  %bit = call i64 @pq_read_bits(ptr %data, i64 %dlen, i64 %bi, i32 1)
  %bit8 = trunc i64 %bit to i8
  %bop = getelementptr inbounds nuw i8, ptr %out, i64 %bi
  store i8 %bit8, ptr %bop, align 1
  %bi.next = add nuw i64 %bi, 1
  br label %bool.loop

bool.fin:
  store i64 %count, ptr %out_used, align 8
  ret i32 0

; ---- BYTE_ARRAY: [u32 len][bytes] repeated ----
do.barray:
  br label %ba.loop

ba.loop:
  %ba.i = phi i64 [ 0, %do.barray ], [ %ba.i.next, %ba.cont ]
  %ba.ip = phi i64 [ 0, %do.barray ], [ %ba.dend, %ba.cont ]
  %ba.op = phi i64 [ 0, %do.barray ], [ %ba.oend, %ba.cont ]
  %ba.done = icmp uge i64 %ba.i, %count
  br i1 %ba.done, label %ba.fin, label %ba.hdr

ba.hdr:
  %ba.hend = add i64 %ba.ip, 4
  %ba.hovf = icmp ult i64 %ba.hend, %ba.ip
  br i1 %ba.hovf, label %err.size, label %ba.hchk

ba.hchk:
  %ba.hbig = icmp ugt i64 %ba.hend, %dlen
  br i1 %ba.hbig, label %err.arg, label %ba.readlen

ba.readlen:
  %ba.lp = getelementptr inbounds nuw i8, ptr %data, i64 %ba.ip
  %ba.len32 = load i32, ptr %ba.lp, align 1
  %ba.len = zext i32 %ba.len32 to i64
  %ba.dend = add i64 %ba.hend, %ba.len
  %ba.dovf = icmp ult i64 %ba.dend, %ba.hend
  br i1 %ba.dovf, label %err.size, label %ba.dchk

ba.dchk:
  %ba.dbig = icmp ugt i64 %ba.dend, %dlen
  br i1 %ba.dbig, label %err.arg, label %ba.outchk

ba.outchk:
  %ba.orec = add i64 %ba.op, 4
  %ba.oend = add i64 %ba.orec, %ba.len
  %ba.oovf = icmp ult i64 %ba.oend, %ba.op
  br i1 %ba.oovf, label %err.size, label %ba.capchk

ba.capchk:
  %ba.obig = icmp ugt i64 %ba.oend, %out_cap
  br i1 %ba.obig, label %err.full, label %ba.write

ba.write:
  %ba.owp = getelementptr inbounds nuw i8, ptr %out, i64 %ba.op
  store i32 %ba.len32, ptr %ba.owp, align 1
  %ba.odp = getelementptr inbounds nuw i8, ptr %out, i64 %ba.orec
  %ba.sdp = getelementptr inbounds nuw i8, ptr %data, i64 %ba.hend
  call void @llvm.memcpy.p0.p0.i64(ptr %ba.odp, ptr %ba.sdp, i64 %ba.len, i1 false)
  br label %ba.cont

ba.cont:
  %ba.i.next = add nuw i64 %ba.i, 1
  br label %ba.loop

ba.fin:
  store i64 %ba.op, ptr %out_used, align 8
  ret i32 0

; ---- FIXED_LEN_BYTE_ARRAY: type_len bytes each, verbatim ----
do.flba:
  %tl = sext i32 %type_len to i64
  %tlbad = icmp slt i64 %tl, 1
  br i1 %tlbad, label %err.arg, label %flba.need

flba.need:
  %fl.need.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %tl, i64 %count)
  %fl.need = extractvalue { i64, i1 } %fl.need.chk, 0
  %fl.ovf = extractvalue { i64, i1 } %fl.need.chk, 1
  br i1 %fl.ovf, label %err.size, label %flba.bounds

flba.bounds:
  %fl.big = icmp ugt i64 %fl.need, %dlen
  br i1 %fl.big, label %err.arg, label %flba.cap

flba.cap:
  %fl.obig = icmp ugt i64 %fl.need, %out_cap
  br i1 %fl.obig, label %err.full, label %flba.copy

flba.copy:
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %data, i64 %fl.need, i1 false)
  store i64 %fl.need, ptr %out_used, align 8
  ret i32 0

err.null:
  ret i32 1
err.size:
  ret i32 3
err.full:
  ret i32 6
err.arg:
  ret i32 8
}

; ================================================= DELTA_BINARY_PACKED core
; Decodes the whole delta_binary_packed stream. Requires header `total` ≤
; out_cap. Writes `total` i64 values to `out`, *out_count = total, *out_consumed
; = bytes used. Returns 0 / 3 / 8.
define internal i32 @pq_delta_core(ptr %data, i64 %dlen, ptr %out, i64 %out_cap, ptr %out_count, ptr %out_consumed) #3 {
entry:
  %pos.slot = alloca i64, align 8
  store i64 0, ptr %pos.slot, align 8
  %tmp.slot = alloca i64, align 8
  %fv.slot = alloca i64, align 8
  %ok1 = call i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos.slot, ptr %tmp.slot)
  br i1 %ok1, label %h2, label %err.arg, !prof !1
h2:
  %block_size = load i64, ptr %tmp.slot, align 8
  %ok2 = call i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos.slot, ptr %tmp.slot)
  br i1 %ok2, label %h3, label %err.arg, !prof !1
h3:
  %miniblocks = load i64, ptr %tmp.slot, align 8
  %ok3 = call i1 @pq_uleb(ptr %data, i64 %dlen, ptr %pos.slot, ptr %tmp.slot)
  br i1 %ok3, label %h4, label %err.arg, !prof !1
h4:
  %total = load i64, ptr %tmp.slot, align 8
  %ok4 = call i1 @pq_zigzag(ptr %data, i64 %dlen, ptr %pos.slot, ptr %fv.slot)
  br i1 %ok4, label %validate, label %err.arg, !prof !1

validate:
  %first = load i64, ptr %fv.slot, align 8
  %mbz = icmp eq i64 %miniblocks, 0
  %bsz = icmp eq i64 %block_size, 0
  %badhdr = or i1 %mbz, %bsz
  br i1 %badhdr, label %err.arg, label %v2

v2:
  %rem = urem i64 %block_size, %miniblocks
  %remnz = icmp ne i64 %rem, 0
  br i1 %remnz, label %err.arg, label %v3

v3:
  %toobig = icmp ugt i64 %total, %out_cap
  br i1 %toobig, label %err.arg, label %v4

v4:
  %vpm = udiv i64 %block_size, %miniblocks
  store i64 %total, ptr %out_count, align 8
  %tz = icmp eq i64 %total, 0
  br i1 %tz, label %fin, label %seed

seed:
  store i64 %first, ptr %out, align 8
  br label %block.head

block.head:
  %written = phi i64 [ 1, %seed ], [ %written.m, %block.cont ]
  %value = phi i64 [ %first, %seed ], [ %value.m, %block.cont ]
  %ptot = phi i64 [ 1, %seed ], [ %ptot.m, %block.cont ]
  %more.blk = icmp ult i64 %ptot, %total
  br i1 %more.blk, label %block.body, label %fin

block.body:
  %okmd = call i1 @pq_zigzag(ptr %data, i64 %dlen, ptr %pos.slot, ptr %tmp.slot)
  br i1 %okmd, label %block.widths, label %err.arg, !prof !1

block.widths:
  %min_delta = load i64, ptr %tmp.slot, align 8
  %wbase = load i64, ptr %pos.slot, align 8
  %wend = add i64 %wbase, %miniblocks
  %wovf = icmp ult i64 %wend, %wbase
  br i1 %wovf, label %err.size, label %block.wchk
block.wchk:
  %wbig = icmp ugt i64 %wend, %dlen
  br i1 %wbig, label %err.arg, label %block.wadv
block.wadv:
  store i64 %wend, ptr %pos.slot, align 8
  br label %mb.head

mb.head:
  %m = phi i64 [ 0, %block.wadv ], [ %m.next, %mb.cont ]
  %written.m = phi i64 [ %written, %block.wadv ], [ %written.k, %mb.cont ]
  %value.m = phi i64 [ %value, %block.wadv ], [ %value.k, %mb.cont ]
  %ptot.m = phi i64 [ %ptot, %block.wadv ], [ %ptot.k, %mb.cont ]
  %m.done = icmp uge i64 %m, %miniblocks
  %p.done = icmp uge i64 %ptot.m, %total
  %mstop = or i1 %m.done, %p.done
  br i1 %mstop, label %block.cont, label %mb.body

mb.body:
  %widx = add i64 %wbase, %m
  %wp = getelementptr inbounds nuw i8, ptr %data, i64 %widx
  %w8 = load i8, ptr %wp, align 1
  %width = zext i8 %w8 to i64
  %wtoo = icmp ugt i64 %width, 64
  br i1 %wtoo, label %err.arg, label %mb.need

mb.need:
  %prod.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %vpm, i64 %width)
  %prodbits = extractvalue { i64, i1 } %prod.chk, 0
  %prod.ovf = extractvalue { i64, i1 } %prod.chk, 1
  br i1 %prod.ovf, label %err.size, label %mb.need2
mb.need2:
  %pb7 = add i64 %prodbits, 7
  %need = lshr i64 %pb7, 3
  %vpos = load i64, ptr %pos.slot, align 8
  %vend = add i64 %vpos, %need
  %vovf = icmp ult i64 %vend, %vpos
  br i1 %vovf, label %err.size, label %mb.bchk
mb.bchk:
  %vbig = icmp ugt i64 %vend, %dlen
  br i1 %vbig, label %err.arg, label %mb.decode

mb.decode:
  %bitbase = shl i64 %vpos, 3
  %wi32 = trunc i64 %width to i32
  br label %val.head

val.head:
  %k = phi i64 [ 0, %mb.decode ], [ %k.next, %val.body ]
  %written.k = phi i64 [ %written.m, %mb.decode ], [ %written.kc, %val.body ]
  %value.k = phi i64 [ %value.m, %mb.decode ], [ %value.kc, %val.body ]
  %ptot.k = phi i64 [ %ptot.m, %mb.decode ], [ %ptot.kc, %val.body ]
  %k.done = icmp uge i64 %k, %vpm
  %p.done2 = icmp uge i64 %ptot.k, %total
  %vstop = or i1 %k.done, %p.done2
  br i1 %vstop, label %mb.cont, label %val.body

val.body:
  %koff = mul i64 %k, %width
  %kbit = add i64 %bitbase, %koff
  %raw = call i64 @pq_read_bits(ptr %data, i64 %dlen, i64 %kbit, i32 %wi32)
  %delta = add i64 %min_delta, %raw
  %value.kc = add i64 %value.k, %delta
  %vop = getelementptr inbounds nuw i64, ptr %out, i64 %written.k
  store i64 %value.kc, ptr %vop, align 8
  %written.kc = add i64 %written.k, 1
  %ptot.kc = add i64 %ptot.k, 1
  %k.next = add nuw i64 %k, 1
  br label %val.head

mb.cont:
  %vposn = add i64 %vpos, %need
  store i64 %vposn, ptr %pos.slot, align 8
  %m.next = add nuw i64 %m, 1
  br label %mb.head

block.cont:
  br label %block.head

fin:
  %posf = load i64, ptr %pos.slot, align 8
  store i64 %posf, ptr %out_consumed, align 8
  ret i32 0

err.size:
  ret i32 3
err.arg:
  ret i32 8
}

; ================================================ DELTA_BINARY_PACKED (public)
define i32 @universe_docparse_parquet_delta_binary_packed(ptr %data, i64 %dlen, ptr %out, i64 %out_cap, ptr %out_count) #1 {
entry:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %cn = icmp eq ptr %out_count, null
  %n1 = or i1 %dn, %on
  %nn = or i1 %n1, %cn
  br i1 %nn, label %err.null, label %go

go:
  %cons.slot = alloca i64, align 8
  %r = call i32 @pq_delta_core(ptr %data, i64 %dlen, ptr %out, i64 %out_cap, ptr %out_count, ptr %cons.slot)
  ret i32 %r

err.null:
  ret i32 1
}

; ============================================ DELTA_LENGTH_BYTE_ARRAY (public)
define i32 @universe_docparse_parquet_delta_length_byte_array(ptr %data, i64 %dlen, i64 %count, ptr %out, i64 %out_cap, ptr %out_used) #1 {
entry:
  %un = icmp eq ptr %out_used, null
  br i1 %un, label %err.null.early, label %chkcount

chkcount:
  %cz = icmp eq i64 %count, 0
  br i1 %cz, label %empty, label %notzero

empty:
  store i64 0, ptr %out_used, align 8
  ret i32 0

notzero:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %nn = or i1 %dn, %on
  br i1 %nn, label %err.null.early, label %alloc

alloc:
  %bytes.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %count, i64 8)
  %bytes = extractvalue { i64, i1 } %bytes.chk, 0
  %bytes.ovf = extractvalue { i64, i1 } %bytes.chk, 1
  br i1 %bytes.ovf, label %err.size.early, label %alloc2

alloc2:
  %lens = call ptr @malloc(i64 %bytes)
  %ln = icmp eq ptr %lens, null
  br i1 %ln, label %err.oom.early, label %decode

decode:
  %nlen.slot = alloca i64, align 8
  %cons.slot = alloca i64, align 8
  %dr = call i32 @pq_delta_core(ptr %data, i64 %dlen, ptr %lens, i64 %count, ptr %nlen.slot, ptr %cons.slot)
  %drok = icmp eq i32 %dr, 0
  br i1 %drok, label %check.n, label %fail.decode

fail.decode:
  call void @free(ptr %lens)
  ret i32 %dr

check.n:
  %nlen = load i64, ptr %nlen.slot, align 8
  %nmatch = icmp eq i64 %nlen, %count
  br i1 %nmatch, label %build, label %fail.arg

build:
  %consumed = load i64, ptr %cons.slot, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %build ], [ %i.next, %cont ]
  %p = phi i64 [ %consumed, %build ], [ %pend, %cont ]
  %op = phi i64 [ 0, %build ], [ %oend, %cont ]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %fin, label %body

body:
  %lp = getelementptr inbounds nuw i64, ptr %lens, i64 %i
  %len = load i64, ptr %lp, align 8
  %lneg = icmp slt i64 %len, 0
  br i1 %lneg, label %fail.arg, label %body.in

body.in:
  %pend = add i64 %p, %len
  %povf = icmp ult i64 %pend, %p
  br i1 %povf, label %fail.size, label %body.inchk
body.inchk:
  %pbig = icmp ugt i64 %pend, %dlen
  br i1 %pbig, label %fail.arg, label %body.outchk

body.outchk:
  %orec = add i64 %op, 4
  %oend = add i64 %orec, %len
  %oovf = icmp ult i64 %oend, %op
  br i1 %oovf, label %fail.size, label %body.capchk
body.capchk:
  %obig = icmp ugt i64 %oend, %out_cap
  br i1 %obig, label %fail.full, label %body.write

body.write:
  %len32 = trunc i64 %len to i32
  %owp = getelementptr inbounds nuw i8, ptr %out, i64 %op
  store i32 %len32, ptr %owp, align 1
  %odp = getelementptr inbounds nuw i8, ptr %out, i64 %orec
  %sdp = getelementptr inbounds nuw i8, ptr %data, i64 %p
  call void @llvm.memcpy.p0.p0.i64(ptr %odp, ptr %sdp, i64 %len, i1 false)
  br label %cont

cont:
  %i.next = add nuw i64 %i, 1
  br label %loop

fin:
  store i64 %op, ptr %out_used, align 8
  call void @free(ptr %lens)
  ret i32 0

fail.size:
  call void @free(ptr %lens)
  ret i32 3
fail.full:
  call void @free(ptr %lens)
  ret i32 6
fail.arg:
  call void @free(ptr %lens)
  ret i32 8

err.null.early:
  ret i32 1
err.size.early:
  ret i32 3
err.oom.early:
  ret i32 2
}

; =================================================== DELTA_BYTE_ARRAY (public)
define i32 @universe_docparse_parquet_delta_byte_array(ptr %data, i64 %dlen, i64 %count, ptr %out, i64 %out_cap, ptr %out_used) #1 {
entry:
  %un = icmp eq ptr %out_used, null
  br i1 %un, label %err.null.early, label %chkcount

chkcount:
  %cz = icmp eq i64 %count, 0
  br i1 %cz, label %empty, label %notzero

empty:
  store i64 0, ptr %out_used, align 8
  ret i32 0

notzero:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %nn = or i1 %dn, %on
  br i1 %nn, label %err.null.early, label %alloc

alloc:
  %bytes.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %count, i64 8)
  %bytes = extractvalue { i64, i1 } %bytes.chk, 0
  %bytes.ovf = extractvalue { i64, i1 } %bytes.chk, 1
  br i1 %bytes.ovf, label %err.size.early, label %alloc.pre

alloc.pre:
  %pre = call ptr @malloc(i64 %bytes)
  %pren = icmp eq ptr %pre, null
  br i1 %pren, label %err.oom.early, label %alloc.suf

alloc.suf:
  %suf = call ptr @malloc(i64 %bytes)
  %sufn = icmp eq ptr %suf, null
  br i1 %sufn, label %free.pre.oom, label %decode.pre

decode.pre:
  %n1.slot = alloca i64, align 8
  %c1.slot = alloca i64, align 8
  %n2.slot = alloca i64, align 8
  %c2.slot = alloca i64, align 8
  %r1 = call i32 @pq_delta_core(ptr %data, i64 %dlen, ptr %pre, i64 %count, ptr %n1.slot, ptr %c1.slot)
  %r1ok = icmp eq i32 %r1, 0
  br i1 %r1ok, label %chk.n1, label %fail.r1

fail.r1:
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 %r1

chk.n1:
  %n1 = load i64, ptr %n1.slot, align 8
  %n1match = icmp eq i64 %n1, %count
  br i1 %n1match, label %decode.suf, label %fail.arg

decode.suf:
  %c1 = load i64, ptr %c1.slot, align 8
  %data2 = getelementptr inbounds nuw i8, ptr %data, i64 %c1
  %dlen2 = sub i64 %dlen, %c1
  %r2 = call i32 @pq_delta_core(ptr %data2, i64 %dlen2, ptr %suf, i64 %count, ptr %n2.slot, ptr %c2.slot)
  %r2ok = icmp eq i32 %r2, 0
  br i1 %r2ok, label %chk.n2, label %fail.r2

fail.r2:
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 %r2

chk.n2:
  %n2 = load i64, ptr %n2.slot, align 8
  %n2match = icmp eq i64 %n2, %count
  br i1 %n2match, label %build, label %fail.arg

build:
  %c2 = load i64, ptr %c2.slot, align 8
  %sufbase = add i64 %c1, %c2
  br label %loop

loop:
  %i = phi i64 [ 0, %build ], [ %i.next, %cont ]
  %sp = phi i64 [ %sufbase, %build ], [ %spend, %cont ]
  %op = phi i64 [ 0, %build ], [ %oend, %cont ]
  %prev.off = phi i64 [ 0, %build ], [ %orec, %cont ]
  %prev.len = phi i64 [ 0, %build ], [ %tlen, %cont ]
  %done = icmp uge i64 %i, %count
  br i1 %done, label %fin, label %body

body:
  %pp = getelementptr inbounds nuw i64, ptr %pre, i64 %i
  %plen = load i64, ptr %pp, align 8
  %plneg = icmp slt i64 %plen, 0
  br i1 %plneg, label %fail.arg, label %body.pchk
body.pchk:
  %ptoo = icmp ugt i64 %plen, %prev.len
  br i1 %ptoo, label %fail.arg, label %body.suf

body.suf:
  %sfp = getelementptr inbounds nuw i64, ptr %suf, i64 %i
  %slen = load i64, ptr %sfp, align 8
  %slneg = icmp slt i64 %slen, 0
  br i1 %slneg, label %fail.arg, label %body.sfchk
body.sfchk:
  %spend = add i64 %sp, %slen
  %spovf = icmp ult i64 %spend, %sp
  br i1 %spovf, label %fail.size, label %body.sfbnd
body.sfbnd:
  %spbig = icmp ugt i64 %spend, %dlen
  br i1 %spbig, label %fail.arg, label %body.tlen

body.tlen:
  %tlen.chk = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %plen, i64 %slen)
  %tlen = extractvalue { i64, i1 } %tlen.chk, 0
  %tlen.ovf = extractvalue { i64, i1 } %tlen.chk, 1
  br i1 %tlen.ovf, label %fail.size, label %body.outchk

body.outchk:
  %orec = add i64 %op, 4
  %oend = add i64 %orec, %tlen
  %oovf = icmp ult i64 %oend, %op
  br i1 %oovf, label %fail.size, label %body.capchk
body.capchk:
  %obig = icmp ugt i64 %oend, %out_cap
  br i1 %obig, label %fail.full, label %body.write

body.write:
  %tlen32 = trunc i64 %tlen to i32
  %owp = getelementptr inbounds nuw i8, ptr %out, i64 %op
  store i32 %tlen32, ptr %owp, align 1
  ; prefix bytes copied from the previous value's data region (non-overlapping:
  ; %orec >= prev.off + prev.len >= prev.off + plen).
  %pdst = getelementptr inbounds nuw i8, ptr %out, i64 %orec
  %psrc = getelementptr inbounds nuw i8, ptr %out, i64 %prev.off
  call void @llvm.memcpy.p0.p0.i64(ptr %pdst, ptr %psrc, i64 %plen, i1 false)
  %sdst.off = add i64 %orec, %plen
  %sdst = getelementptr inbounds nuw i8, ptr %out, i64 %sdst.off
  %ssrc = getelementptr inbounds nuw i8, ptr %data, i64 %sp
  call void @llvm.memcpy.p0.p0.i64(ptr %sdst, ptr %ssrc, i64 %slen, i1 false)
  br label %cont

cont:
  %i.next = add nuw i64 %i, 1
  br label %loop

fin:
  store i64 %op, ptr %out_used, align 8
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 0

fail.size:
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 3
fail.full:
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 6
fail.arg:
  call void @free(ptr %pre)
  call void @free(ptr %suf)
  ret i32 8

free.pre.oom:
  call void @free(ptr %pre)
  ret i32 2

err.null.early:
  ret i32 1
err.size.early:
  ret i32 3
err.oom.early:
  ret i32 2
}

; ==================================================== BYTE_STREAM_SPLIT
; K = width byte-planes each `count` bytes; value j byte b lives at
; data[b*count + j]. Reassemble out[j*width + b] = data[b*count + j].
define i32 @universe_docparse_parquet_byte_stream_split(ptr %data, i64 %dlen, i32 %width, i64 %count, ptr %out) #1 {
entry:
  %cz = icmp eq i64 %count, 0
  br i1 %cz, label %ok, label %notzero

notzero:
  %dn = icmp eq ptr %data, null
  %on = icmp eq ptr %out, null
  %nn = or i1 %dn, %on
  br i1 %nn, label %err.null, label %chkw

chkw:
  %wbad = icmp slt i32 %width, 1
  br i1 %wbad, label %err.arg, label %need

need:
  %wv = zext i32 %width to i64
  %need.chk = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %wv, i64 %count)
  %needb = extractvalue { i64, i1 } %need.chk, 0
  %need.ovf = extractvalue { i64, i1 } %need.chk, 1
  br i1 %need.ovf, label %err.size, label %bounds

bounds:
  %big = icmp ugt i64 %needb, %dlen
  br i1 %big, label %err.arg, label %jloop

jloop:
  %j = phi i64 [ 0, %bounds ], [ %j.next, %jcont ]
  %j.done = icmp uge i64 %j, %count
  br i1 %j.done, label %ok, label %jbody

jbody:
  %jw = mul i64 %j, %wv
  br label %bloop

bloop:
  %b = phi i64 [ 0, %jbody ], [ %b.next, %bbody ]
  %b.done = icmp uge i64 %b, %wv
  br i1 %b.done, label %jcont, label %bbody

bbody:
  %bc = mul i64 %b, %count
  %sidx = add i64 %bc, %j
  %sp = getelementptr inbounds nuw i8, ptr %data, i64 %sidx
  %byte = load i8, ptr %sp, align 1
  %didx = add i64 %jw, %b
  %dp = getelementptr inbounds nuw i8, ptr %out, i64 %didx
  store i8 %byte, ptr %dp, align 1
  %b.next = add nuw i64 %b, 1
  br label %bloop

jcont:
  %j.next = add nuw i64 %j, 1
  br label %jloop

ok:
  ret i32 0
err.null:
  ret i32 1
err.size:
  ret i32 3
err.arg:
  ret i32 8
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #3 = { nounwind }

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 2000, i32 1}
