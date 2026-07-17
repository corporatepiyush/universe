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

; Apache Parquet metadata reader — footer location + Thrift-compact navigation.
;
; A Parquet file is framed by the 4-byte magic "PAR1" at BOTH ends; the tail is
;   [ FileMetaData (thrift-compact) ][ u32 footer_len (LE) ][ "PAR1" ].
; FileMetaData names the schema and, per row group, each column chunk's physical
; type, codec, encodings, and page byte-offsets. Each column chunk is a sequence
; of pages, every page prefixed by a thrift-compact PageHeader.
;
; This module is a READ-ONLY, ALLOCATION-FREE navigator over a caller-owned
; buffer. It parses the FileMetaData thrift stream lazily via the shared
; encoding/thrift reader (universe_encoding_thrift_*): the metadata is tiny, so
; random access to row group / column N re-scans from the recorded list start
; (skip N structs, then parse the target) rather than materializing everything.
; Every parsed record lands in a CALLER-PROVIDED fixed struct whose byte layout
; is defined explicitly with `getelementptr i8` + documented offsets, so the
; layout is identical on every target (portable-layout rule).
;
; DESIGN:
;   * Thrift-compact field ids (from the OFFICIAL parquet.thrift):
;     - FileMetaData: 1 version(i32), 2 schema(list<SchemaElement>),
;       3 num_rows(i64), 4 row_groups(list<RowGroup>), 6 created_by(string).
;     - SchemaElement: 1 type(i32), 2 type_length(i32), 3 repetition_type(i32),
;       4 name(string), 5 num_children(i32), 6 converted_type(i32).
;     - RowGroup: 1 columns(list<ColumnChunk>), 2 total_byte_size(i64),
;       3 num_rows(i64).
;     - ColumnChunk: 3 meta_data(ColumnMetaData).
;     - ColumnMetaData: 1 type(i32), 2 encodings(list<i32>), 4 codec(i32),
;       5 num_values(i64), 7 total_compressed_size(i64), 9 data_page_offset(i64),
;       11 dictionary_page_offset(i64).
;     - PageHeader: 1 type(i32), 2 uncompressed_page_size(i32),
;       3 compressed_page_size(i32), 5 DATA_PAGE header, 7 DICTIONARY header,
;       8 DATA_PAGE_V2 header. NOTE: the field ids for the DATA and DICTIONARY
;       page headers are 5 and 7 respectively (verified against real DuckDB
;       output AND parquet.thrift) — a common mislabel swaps them.
;     - DataPageHeader / DictionaryPageHeader / DataPageHeaderV2 all carry
;       num_values at field 1; encoding is field 2 (v1/dict) or field 4 (v2).
;   * UNTRUSTED INPUT: every buffer touch goes through the thrift reader, which
;     bounds-checks each byte and returns 8 (INVALID_ARG) on truncation. This
;     module adds only its own framing checks (magic, footer length range).
;     Malformed / truncated / out-of-range → 8; null pointer arg → 1;
;     out-of-range index → 7; iteration past end → 5. Nothing panics/OOB-reads.
;   * Zero allocation: the thrift reader is a 32-byte stack struct; string reads
;     return (ptr,len) VIEWS into the caller buffer.
;
; Caller struct byte layouts (all fields written explicitly by offset):
;   FileMeta handle (fmd, 64 B): +0 base(ptr) +8 len(i64) +16 version(i32)
;     +20 schema_count(i32) +24 schema_start(i64) +32 rg_count(i32)
;     +40 rg_start(i64) +48 num_rows(i64).
;   Schema cursor (16 B): +0 pos(i64) +8 remaining(i32) +12 flag(i32). Caller
;     zero-initializes; flag==0 means "start from fmd.schema_start".
;   SchemaElement out (48 B): +0 type(i32,-1=absent) +4 type_length(i32)
;     +8 repetition_type(i32) +12 num_children(i32) +16 converted_type(i32,-1)
;     +24 name_ptr(ptr) +32 name_len(i64).
;   RowGroup out (48 B): +0 base(ptr) +8 len(i64) +16 col_count(i32)
;     +24 col_start(i64) +32 num_rows(i64) +40 total_byte_size(i64).
;   ColumnChunk out (56 B): +0 type(i32) +4 codec(i32) +8 first_encoding(i32,-1)
;     +16 num_values(i64) +24 data_page_offset(i64)
;     +32 dictionary_page_offset(i64,-1=absent) +40 total_compressed_size(i64).
;   PageHeader out (24 B): +0 type(i32) +4 uncompressed_page_size(i32)
;     +8 compressed_page_size(i32) +12 num_values(i32) +16 encoding(i32,-1).
;
; API (i32-returning: 0 OK, 1 NULL_PTR, 5 NOT_FOUND=end of iteration,
;   7 INVALID_INDEX, 8 INVALID_ARG=malformed/truncated framing):
;   i32 universe_docparse_parquet_footer(ptr buf,i64 len, ptr out_meta_ptr, ptr out_meta_len)
;   i32 universe_docparse_parquet_meta_open(ptr meta_buf,i64 len, ptr fmd)
;   i64 universe_docparse_parquet_meta_num_rows(ptr fmd)
;   i64 universe_docparse_parquet_meta_row_group_count(ptr fmd)
;   i32 universe_docparse_parquet_schema_next(ptr fmd, ptr cursor, ptr out_elem)
;   i32 universe_docparse_parquet_row_group(ptr fmd, i64 rg_idx, ptr out_rg)
;   i32 universe_docparse_parquet_column(ptr out_rg, i64 col_idx, ptr out_cc)
;   i32 universe_docparse_parquet_page_header(ptr buf,i64 len,i64 pos, ptr out_ph, ptr out_next_pos)

declare void @universe_encoding_thrift_init(ptr, ptr, i64)
declare i32 @universe_encoding_thrift_read_i32(ptr, ptr)
declare i32 @universe_encoding_thrift_read_i64(ptr, ptr)
declare i32 @universe_encoding_thrift_read_binary(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_field(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_collection(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_skip(ptr, i32, i32)

; ==================================================== internal helpers

; Seat a fresh thrift reader over %buf[0,%len) at cursor %pos (field-id base 0).
define internal void @pq.reader_at(ptr %r, ptr %buf, i64 %len, i64 %pos) #0 {
entry:
  call void @universe_encoding_thrift_init(ptr %r, ptr %buf, i64 %len)
  %pp = getelementptr inbounds nuw i8, ptr %r, i64 16
  store i64 %pos, ptr %pp, align 8
  ret void
}

; Current cursor position of the reader.
define internal i64 @pq.pos(ptr %r) #0 {
entry:
  %pp = getelementptr inbounds nuw i8, ptr %r, i64 16
  %pos = load i64, ptr %pp, align 8
  ret i64 %pos
}

; Skip %n consecutive elements of thrift type %etype starting at the cursor.
define internal i32 @pq.skip_n(ptr %r, i32 %etype, i32 %n) #0 {
entry:
  %nonpos = icmp sle i32 %n, 0
  br i1 %nonpos, label %done, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %in, %next ]
  %rc = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %etype, i32 0)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %next, label %err
next:
  %in = add nuw nsw i32 %i, 1
  %fin = icmp sge i32 %in, %n
  br i1 %fin, label %done, label %loop
done:
  ret i32 0
err:
  ret i32 %rc
}

; Parse a ColumnMetaData struct (reader positioned at its first field header;
; the enclosing STRUCT field header was just consumed). Saves/resets/restores
; the field-id delta base per the compact-struct discipline.
define internal i32 @pq.parse_col_meta(ptr %r, ptr %cc) #0 {
entry:
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %v32 = alloca i32, align 4
  %et8 = alloca i8, align 1
  %sz32 = alloca i32, align 4
  %fbp = getelementptr inbounds nuw i8, ptr %r, i64 24
  %saved = load i16, ptr %fbp, align 2
  store i16 0, ptr %fbp, align 2
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_type
    i32 2, label %f_enc
    i32 4, label %f_codec
    i32 5, label %f_nvals
    i32 7, label %f_tcs
    i32 9, label %f_dpo
    i32 11, label %f_dico
  ]
f_type:
  %rt = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okt = icmp eq i32 %rt, 0
  br i1 %okt, label %f_type.st, label %err
f_type.st:
  %tv = load i32, ptr %v32, align 4
  %p_type = getelementptr inbounds nuw i8, ptr %cc, i64 0
  store i32 %tv, ptr %p_type, align 4
  br label %loop
f_enc:
  %rce = call i32 @universe_encoding_thrift_collection(ptr %r, ptr %et8, ptr %sz32)
  %oke = icmp eq i32 %rce, 0
  br i1 %oke, label %f_enc.chk, label %err
f_enc.chk:
  %esz = load i32, ptr %sz32, align 4
  %eempty = icmp sle i32 %esz, 0
  br i1 %eempty, label %loop, label %f_enc.first
f_enc.first:
  %rce1 = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %oke1 = icmp eq i32 %rce1, 0
  br i1 %oke1, label %f_enc.first.st, label %err
f_enc.first.st:
  %ev = load i32, ptr %v32, align 4
  %p_enc = getelementptr inbounds nuw i8, ptr %cc, i64 8
  store i32 %ev, ptr %p_enc, align 4
  %erem = sub nsw i32 %esz, 1
  %et = load i8, ptr %et8, align 1
  %et32 = zext i8 %et to i32
  %rcs = call i32 @pq.skip_n(ptr %r, i32 %et32, i32 %erem)
  %oks = icmp eq i32 %rcs, 0
  br i1 %oks, label %loop, label %err
f_codec:
  %rcc = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okc = icmp eq i32 %rcc, 0
  br i1 %okc, label %f_codec.st, label %err
f_codec.st:
  %cv = load i32, ptr %v32, align 4
  %p_codec = getelementptr inbounds nuw i8, ptr %cc, i64 4
  store i32 %cv, ptr %p_codec, align 4
  br label %loop
f_nvals:
  %p_nv = getelementptr inbounds nuw i8, ptr %cc, i64 16
  %rcn = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %p_nv)
  %okn = icmp eq i32 %rcn, 0
  br i1 %okn, label %loop, label %err
f_tcs:
  %p_tcs = getelementptr inbounds nuw i8, ptr %cc, i64 40
  %rctcs = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %p_tcs)
  %oktcs = icmp eq i32 %rctcs, 0
  br i1 %oktcs, label %loop, label %err
f_dpo:
  %p_dpo = getelementptr inbounds nuw i8, ptr %cc, i64 24
  %rcd = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %p_dpo)
  %okd = icmp eq i32 %rcd, 0
  br i1 %okd, label %loop, label %err
f_dico:
  %p_dico = getelementptr inbounds nuw i8, ptr %cc, i64 32
  %rcdi = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %p_dico)
  %okdi = icmp eq i32 %rcdi, 0
  br i1 %okdi, label %loop, label %err
skip:
  %ct32.skip = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32.skip, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err
done:
  store i16 %saved, ptr %fbp, align 2
  ret i32 0
err:
  store i16 %saved, ptr %fbp, align 2
  ret i32 8
}

; Parse a page sub-header struct (DataPageHeader / DictionaryPageHeader /
; DataPageHeaderV2): field 1 num_values(i32), field 2 or 4 encoding(i32).
; Stores into out_ph +12 (num_values) and +16 (encoding).
define internal i32 @pq.parse_page_sub(ptr %r, ptr %ph) #0 {
entry:
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %v32 = alloca i32, align 4
  %fbp = getelementptr inbounds nuw i8, ptr %r, i64 24
  %saved = load i16, ptr %fbp, align 2
  store i16 0, ptr %fbp, align 2
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_nv
    i32 2, label %f_enc
    i32 4, label %f_enc
  ]
f_nv:
  %rcn = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okn = icmp eq i32 %rcn, 0
  br i1 %okn, label %f_nv.st, label %err
f_nv.st:
  %nv = load i32, ptr %v32, align 4
  %p_nv = getelementptr inbounds nuw i8, ptr %ph, i64 12
  store i32 %nv, ptr %p_nv, align 4
  br label %loop
f_enc:
  %rce = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %oke = icmp eq i32 %rce, 0
  br i1 %oke, label %f_enc.st, label %err
f_enc.st:
  %ev = load i32, ptr %v32, align 4
  %p_enc = getelementptr inbounds nuw i8, ptr %ph, i64 16
  store i32 %ev, ptr %p_enc, align 4
  br label %loop
skip:
  %ct32.skip = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32.skip, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err
done:
  store i16 %saved, ptr %fbp, align 2
  ret i32 0
err:
  store i16 %saved, ptr %fbp, align 2
  ret i32 8
}

; ==================================================================== footer
define i32 @universe_docparse_parquet_footer(ptr %buf, i64 %len, ptr %omp, ptr %oml) #1 {
entry:
  %bn = icmp eq ptr %buf, null
  %pn = icmp eq ptr %omp, null
  %ln = icmp eq ptr %oml, null
  %n0 = or i1 %bn, %pn
  %anynull = or i1 %n0, %ln
  br i1 %anynull, label %err.null, label %chklen, !prof !0
chklen:
  %tooshort = icmp ult i64 %len, 12
  br i1 %tooshort, label %err.bad, label %magic0, !prof !0
magic0:
  %m0 = load i32, ptr %buf, align 1
  %m0ok = icmp eq i32 %m0, 827474256
  br i1 %m0ok, label %magic1, label %err.bad, !prof !0
magic1:
  %endoff = sub nuw i64 %len, 4
  %endp = getelementptr inbounds nuw i8, ptr %buf, i64 %endoff
  %m1 = load i32, ptr %endp, align 1
  %m1ok = icmp eq i32 %m1, 827474256
  br i1 %m1ok, label %flen, label %err.bad, !prof !0
flen:
  %lenoff = sub nuw i64 %len, 8
  %lenp = getelementptr inbounds nuw i8, ptr %buf, i64 %lenoff
  %flen32 = load i32, ptr %lenp, align 1
  %flen64 = zext i32 %flen32 to i64
  ; meta_start = (len-8) - footer_len; must not underflow and must be >= 4.
  %fits = icmp ule i64 %flen64, %lenoff
  br i1 %fits, label %compute, label %err.bad, !prof !0
compute:
  %mstart = sub nuw i64 %lenoff, %flen64
  %minok = icmp uge i64 %mstart, 4
  br i1 %minok, label %store, label %err.bad, !prof !0
store:
  %mp = getelementptr inbounds nuw i8, ptr %buf, i64 %mstart
  store ptr %mp, ptr %omp, align 8
  store i64 %flen64, ptr %oml, align 8
  ret i32 0
err.null:
  ret i32 1
err.bad:
  ret i32 8
}

; ================================================================= meta_open
define i32 @universe_docparse_parquet_meta_open(ptr %mbuf, i64 %len, ptr %fmd) #1 {
entry:
  %r = alloca [32 x i8], align 8
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %v32 = alloca i32, align 4
  %et8 = alloca i8, align 1
  %sz32 = alloca i32, align 4
  %bn = icmp eq ptr %mbuf, null
  %fn = icmp eq ptr %fmd, null
  %anynull = or i1 %bn, %fn
  br i1 %anynull, label %err.null, label %init, !prof !0
init:
  %p_base = getelementptr inbounds nuw i8, ptr %fmd, i64 0
  store ptr %mbuf, ptr %p_base, align 8
  %p_len = getelementptr inbounds nuw i8, ptr %fmd, i64 8
  store i64 %len, ptr %p_len, align 8
  %p_ver = getelementptr inbounds nuw i8, ptr %fmd, i64 16
  store i32 0, ptr %p_ver, align 4
  %p_sc = getelementptr inbounds nuw i8, ptr %fmd, i64 20
  store i32 0, ptr %p_sc, align 4
  %p_ss = getelementptr inbounds nuw i8, ptr %fmd, i64 24
  store i64 0, ptr %p_ss, align 8
  %p_rc = getelementptr inbounds nuw i8, ptr %fmd, i64 32
  store i32 0, ptr %p_rc, align 4
  %p_rs = getelementptr inbounds nuw i8, ptr %fmd, i64 40
  store i64 0, ptr %p_rs, align 8
  %p_nr = getelementptr inbounds nuw i8, ptr %fmd, i64 48
  store i64 0, ptr %p_nr, align 8
  call void @pq.reader_at(ptr %r, ptr %mbuf, i64 %len, i64 0)
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err.bad
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_ver
    i32 2, label %f_schema
    i32 3, label %f_nrows
    i32 4, label %f_rgs
  ]
f_ver:
  %rcv = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okv = icmp eq i32 %rcv, 0
  br i1 %okv, label %f_ver.st, label %err.bad
f_ver.st:
  %vv = load i32, ptr %v32, align 4
  store i32 %vv, ptr %p_ver, align 4
  br label %loop
f_schema:
  %rccs = call i32 @universe_encoding_thrift_collection(ptr %r, ptr %et8, ptr %sz32)
  %okcs = icmp eq i32 %rccs, 0
  br i1 %okcs, label %f_schema.rec, label %err.bad
f_schema.rec:
  %ssz = load i32, ptr %sz32, align 4
  store i32 %ssz, ptr %p_sc, align 4
  %spos = call i64 @pq.pos(ptr %r)
  store i64 %spos, ptr %p_ss, align 8
  %set = load i8, ptr %et8, align 1
  %set32 = zext i8 %set to i32
  %rcsk = call i32 @pq.skip_n(ptr %r, i32 %set32, i32 %ssz)
  %oksk = icmp eq i32 %rcsk, 0
  br i1 %oksk, label %loop, label %err.bad
f_nrows:
  %rcn = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %p_nr)
  %okn = icmp eq i32 %rcn, 0
  br i1 %okn, label %loop, label %err.bad
f_rgs:
  %rccr = call i32 @universe_encoding_thrift_collection(ptr %r, ptr %et8, ptr %sz32)
  %okcr = icmp eq i32 %rccr, 0
  br i1 %okcr, label %f_rgs.rec, label %err.bad
f_rgs.rec:
  %rsz = load i32, ptr %sz32, align 4
  store i32 %rsz, ptr %p_rc, align 4
  %rpos = call i64 @pq.pos(ptr %r)
  store i64 %rpos, ptr %p_rs, align 8
  %ret = load i8, ptr %et8, align 1
  %ret32 = zext i8 %ret to i32
  %rcrsk = call i32 @pq.skip_n(ptr %r, i32 %ret32, i32 %rsz)
  %okrsk = icmp eq i32 %rcrsk, 0
  br i1 %okrsk, label %loop, label %err.bad
skip:
  %ct32 = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err.bad
done:
  ret i32 0
err.null:
  ret i32 1
err.bad:
  ret i32 8
}

; ============================================================= meta_num_rows
define i64 @universe_docparse_parquet_meta_num_rows(ptr %fmd) #2 {
entry:
  %n = icmp eq ptr %fmd, null
  br i1 %n, label %null, label %ok, !prof !0
ok:
  %p = getelementptr inbounds nuw i8, ptr %fmd, i64 48
  %v = load i64, ptr %p, align 8
  ret i64 %v
null:
  ret i64 -1
}

; ======================================================= meta_row_group_count
define i64 @universe_docparse_parquet_meta_row_group_count(ptr %fmd) #2 {
entry:
  %n = icmp eq ptr %fmd, null
  br i1 %n, label %null, label %ok, !prof !0
ok:
  %p = getelementptr inbounds nuw i8, ptr %fmd, i64 32
  %v = load i32, ptr %p, align 4
  %v64 = sext i32 %v to i64
  ret i64 %v64
null:
  ret i64 -1
}

; =============================================================== schema_next
define i32 @universe_docparse_parquet_schema_next(ptr %fmd, ptr %cur, ptr %elem) #1 {
entry:
  %r = alloca [32 x i8], align 8
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %v32 = alloca i32, align 4
  %nptr = alloca ptr, align 8
  %nlen = alloca i64, align 8
  %fn = icmp eq ptr %fmd, null
  %cn = icmp eq ptr %cur, null
  %en = icmp eq ptr %elem, null
  %n0 = or i1 %fn, %cn
  %anynull = or i1 %n0, %en
  br i1 %anynull, label %err.null, label %state, !prof !0
state:
  %p_flag = getelementptr inbounds nuw i8, ptr %cur, i64 12
  %flag = load i32, ptr %p_flag, align 4
  %fresh = icmp eq i32 %flag, 0
  br i1 %fresh, label %sinit, label %resume
sinit:
  %p_ss = getelementptr inbounds nuw i8, ptr %fmd, i64 24
  %ipos = load i64, ptr %p_ss, align 8
  %p_sc = getelementptr inbounds nuw i8, ptr %fmd, i64 20
  %irem = load i32, ptr %p_sc, align 4
  store i32 1, ptr %p_flag, align 4
  br label %check
resume:
  %p_pos = getelementptr inbounds nuw i8, ptr %cur, i64 0
  %rpos = load i64, ptr %p_pos, align 8
  %p_rem = getelementptr inbounds nuw i8, ptr %cur, i64 8
  %rrem = load i32, ptr %p_rem, align 4
  br label %check
check:
  %pos = phi i64 [ %ipos, %sinit ], [ %rpos, %resume ]
  %rem = phi i32 [ %irem, %sinit ], [ %rrem, %resume ]
  %endit = icmp sle i32 %rem, 0
  br i1 %endit, label %eof, label %open
open:
  %base = load ptr, ptr %fmd, align 8
  %p_len = getelementptr inbounds nuw i8, ptr %fmd, i64 8
  %len = load i64, ptr %p_len, align 8
  call void @pq.reader_at(ptr %r, ptr %base, i64 %len, i64 %pos)
  %e_type = getelementptr inbounds nuw i8, ptr %elem, i64 0
  store i32 -1, ptr %e_type, align 4
  %e_tlen = getelementptr inbounds nuw i8, ptr %elem, i64 4
  store i32 0, ptr %e_tlen, align 4
  %e_rep = getelementptr inbounds nuw i8, ptr %elem, i64 8
  store i32 0, ptr %e_rep, align 4
  %e_nc = getelementptr inbounds nuw i8, ptr %elem, i64 12
  store i32 0, ptr %e_nc, align 4
  %e_conv = getelementptr inbounds nuw i8, ptr %elem, i64 16
  store i32 -1, ptr %e_conv, align 4
  %e_np = getelementptr inbounds nuw i8, ptr %elem, i64 24
  store ptr null, ptr %e_np, align 8
  %e_nl = getelementptr inbounds nuw i8, ptr %elem, i64 32
  store i64 0, ptr %e_nl, align 8
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err.bad
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %advance, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_type
    i32 2, label %f_tlen
    i32 3, label %f_rep
    i32 4, label %f_name
    i32 5, label %f_nc
    i32 6, label %f_conv
  ]
f_type:
  %rt = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okt = icmp eq i32 %rt, 0
  br i1 %okt, label %f_type.st, label %err.bad
f_type.st:
  %tv = load i32, ptr %v32, align 4
  store i32 %tv, ptr %e_type, align 4
  br label %loop
f_tlen:
  %rtl = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %oktl = icmp eq i32 %rtl, 0
  br i1 %oktl, label %f_tlen.st, label %err.bad
f_tlen.st:
  %tlv = load i32, ptr %v32, align 4
  store i32 %tlv, ptr %e_tlen, align 4
  br label %loop
f_rep:
  %rr = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okr = icmp eq i32 %rr, 0
  br i1 %okr, label %f_rep.st, label %err.bad
f_rep.st:
  %rv = load i32, ptr %v32, align 4
  store i32 %rv, ptr %e_rep, align 4
  br label %loop
f_name:
  %rn = call i32 @universe_encoding_thrift_read_binary(ptr %r, ptr %nptr, ptr %nlen)
  %okn = icmp eq i32 %rn, 0
  br i1 %okn, label %f_name.st, label %err.bad
f_name.st:
  %npv = load ptr, ptr %nptr, align 8
  store ptr %npv, ptr %e_np, align 8
  %nlv = load i64, ptr %nlen, align 8
  store i64 %nlv, ptr %e_nl, align 8
  br label %loop
f_nc:
  %rnc = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %oknc = icmp eq i32 %rnc, 0
  br i1 %oknc, label %f_nc.st, label %err.bad
f_nc.st:
  %ncv = load i32, ptr %v32, align 4
  store i32 %ncv, ptr %e_nc, align 4
  br label %loop
f_conv:
  %rcv = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okcv = icmp eq i32 %rcv, 0
  br i1 %okcv, label %f_conv.st, label %err.bad
f_conv.st:
  %cvv = load i32, ptr %v32, align 4
  store i32 %cvv, ptr %e_conv, align 4
  br label %loop
skip:
  %ct32 = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err.bad
advance:
  %npos = call i64 @pq.pos(ptr %r)
  %p_pos2 = getelementptr inbounds nuw i8, ptr %cur, i64 0
  store i64 %npos, ptr %p_pos2, align 8
  %remn = sub nsw i32 %rem, 1
  %p_rem2 = getelementptr inbounds nuw i8, ptr %cur, i64 8
  store i32 %remn, ptr %p_rem2, align 4
  ret i32 0
eof:
  ret i32 5
err.null:
  ret i32 1
err.bad:
  ret i32 8
}

; ================================================================= row_group
define i32 @universe_docparse_parquet_row_group(ptr %fmd, i64 %idx, ptr %rg) #1 {
entry:
  %r = alloca [32 x i8], align 8
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %et8 = alloca i8, align 1
  %sz32 = alloca i32, align 4
  %fn = icmp eq ptr %fmd, null
  %gn = icmp eq ptr %rg, null
  %anynull = or i1 %fn, %gn
  br i1 %anynull, label %err.null, label %bounds, !prof !0
bounds:
  %p_rc = getelementptr inbounds nuw i8, ptr %fmd, i64 32
  %rgc = load i32, ptr %p_rc, align 4
  %rgc64 = sext i32 %rgc to i64
  %neg = icmp slt i64 %idx, 0
  %over = icmp sge i64 %idx, %rgc64
  %oob = or i1 %neg, %over
  br i1 %oob, label %err.idx, label %seek, !prof !0
seek:
  %base = load ptr, ptr %fmd, align 8
  %p_len = getelementptr inbounds nuw i8, ptr %fmd, i64 8
  %len = load i64, ptr %p_len, align 8
  %p_rs = getelementptr inbounds nuw i8, ptr %fmd, i64 40
  %rstart = load i64, ptr %p_rs, align 8
  call void @pq.reader_at(ptr %r, ptr %base, i64 %len, i64 %rstart)
  %idx32 = trunc i64 %idx to i32
  %rcsk = call i32 @pq.skip_n(ptr %r, i32 12, i32 %idx32)
  %oksk = icmp eq i32 %rcsk, 0
  br i1 %oksk, label %prep, label %err.bad
prep:
  %g_base = getelementptr inbounds nuw i8, ptr %rg, i64 0
  store ptr %base, ptr %g_base, align 8
  %g_len = getelementptr inbounds nuw i8, ptr %rg, i64 8
  store i64 %len, ptr %g_len, align 8
  %g_cc = getelementptr inbounds nuw i8, ptr %rg, i64 16
  store i32 0, ptr %g_cc, align 4
  %g_cs = getelementptr inbounds nuw i8, ptr %rg, i64 24
  store i64 0, ptr %g_cs, align 8
  %g_nr = getelementptr inbounds nuw i8, ptr %rg, i64 32
  store i64 0, ptr %g_nr, align 8
  %g_tbs = getelementptr inbounds nuw i8, ptr %rg, i64 40
  store i64 0, ptr %g_tbs, align 8
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err.bad
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_cols
    i32 2, label %f_tbs
    i32 3, label %f_nr
  ]
f_cols:
  %rcc = call i32 @universe_encoding_thrift_collection(ptr %r, ptr %et8, ptr %sz32)
  %okc = icmp eq i32 %rcc, 0
  br i1 %okc, label %f_cols.rec, label %err.bad
f_cols.rec:
  %csz = load i32, ptr %sz32, align 4
  store i32 %csz, ptr %g_cc, align 4
  %cpos = call i64 @pq.pos(ptr %r)
  store i64 %cpos, ptr %g_cs, align 8
  %cet = load i8, ptr %et8, align 1
  %cet32 = zext i8 %cet to i32
  %rcsk2 = call i32 @pq.skip_n(ptr %r, i32 %cet32, i32 %csz)
  %oksk2 = icmp eq i32 %rcsk2, 0
  br i1 %oksk2, label %loop, label %err.bad
f_tbs:
  %rct = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %g_tbs)
  %okt = icmp eq i32 %rct, 0
  br i1 %okt, label %loop, label %err.bad
f_nr:
  %rcn = call i32 @universe_encoding_thrift_read_i64(ptr %r, ptr %g_nr)
  %okn = icmp eq i32 %rcn, 0
  br i1 %okn, label %loop, label %err.bad
skip:
  %ct32 = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err.bad
done:
  ret i32 0
err.null:
  ret i32 1
err.idx:
  ret i32 7
err.bad:
  ret i32 8
}

; ==================================================================== column
define i32 @universe_docparse_parquet_column(ptr %rg, i64 %idx, ptr %cc) #1 {
entry:
  %r = alloca [32 x i8], align 8
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %gn = icmp eq ptr %rg, null
  %cn = icmp eq ptr %cc, null
  %anynull = or i1 %gn, %cn
  br i1 %anynull, label %err.null, label %bounds, !prof !0
bounds:
  %g_cc = getelementptr inbounds nuw i8, ptr %rg, i64 16
  %colc = load i32, ptr %g_cc, align 4
  %colc64 = sext i32 %colc to i64
  %neg = icmp slt i64 %idx, 0
  %over = icmp sge i64 %idx, %colc64
  %oob = or i1 %neg, %over
  br i1 %oob, label %err.idx, label %seek, !prof !0
seek:
  %base = load ptr, ptr %rg, align 8
  %g_len = getelementptr inbounds nuw i8, ptr %rg, i64 8
  %len = load i64, ptr %g_len, align 8
  %g_cs = getelementptr inbounds nuw i8, ptr %rg, i64 24
  %cstart = load i64, ptr %g_cs, align 8
  call void @pq.reader_at(ptr %r, ptr %base, i64 %len, i64 %cstart)
  %idx32 = trunc i64 %idx to i32
  %rcsk = call i32 @pq.skip_n(ptr %r, i32 12, i32 %idx32)
  %oksk = icmp eq i32 %rcsk, 0
  br i1 %oksk, label %prep, label %err.bad
prep:
  %c_type = getelementptr inbounds nuw i8, ptr %cc, i64 0
  store i32 0, ptr %c_type, align 4
  %c_codec = getelementptr inbounds nuw i8, ptr %cc, i64 4
  store i32 0, ptr %c_codec, align 4
  %c_enc = getelementptr inbounds nuw i8, ptr %cc, i64 8
  store i32 -1, ptr %c_enc, align 4
  %c_nv = getelementptr inbounds nuw i8, ptr %cc, i64 16
  store i64 0, ptr %c_nv, align 8
  %c_dpo = getelementptr inbounds nuw i8, ptr %cc, i64 24
  store i64 0, ptr %c_dpo, align 8
  %c_dico = getelementptr inbounds nuw i8, ptr %cc, i64 32
  store i64 -1, ptr %c_dico, align 8
  %c_tcs = getelementptr inbounds nuw i8, ptr %cc, i64 40
  store i64 0, ptr %c_tcs, align 8
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err.bad
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  %ismeta = icmp eq i32 %id32, 3
  br i1 %ismeta, label %f_meta, label %skip
f_meta:
  %rcm = call i32 @pq.parse_col_meta(ptr %r, ptr %cc)
  %okm = icmp eq i32 %rcm, 0
  br i1 %okm, label %loop, label %err.bad
skip:
  %ct32 = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err.bad
done:
  ret i32 0
err.null:
  ret i32 1
err.idx:
  ret i32 7
err.bad:
  ret i32 8
}

; =============================================================== page_header
define i32 @universe_docparse_parquet_page_header(ptr %buf, i64 %len, i64 %pos, ptr %ph, ptr %onext) #1 {
entry:
  %r = alloca [32 x i8], align 8
  %ct8 = alloca i8, align 1
  %fid = alloca i16, align 2
  %v32 = alloca i32, align 4
  %bn = icmp eq ptr %buf, null
  %pn = icmp eq ptr %ph, null
  %nn = icmp eq ptr %onext, null
  %n0 = or i1 %bn, %pn
  %anynull = or i1 %n0, %nn
  br i1 %anynull, label %err.null, label %chkpos, !prof !0
chkpos:
  %neg = icmp slt i64 %pos, 0
  %over = icmp sge i64 %pos, %len
  %oob = or i1 %neg, %over
  br i1 %oob, label %err.bad, label %pinit, !prof !0
pinit:
  call void @pq.reader_at(ptr %r, ptr %buf, i64 %len, i64 %pos)
  %h_type = getelementptr inbounds nuw i8, ptr %ph, i64 0
  store i32 0, ptr %h_type, align 4
  %h_us = getelementptr inbounds nuw i8, ptr %ph, i64 4
  store i32 0, ptr %h_us, align 4
  %h_cs = getelementptr inbounds nuw i8, ptr %ph, i64 8
  store i32 0, ptr %h_cs, align 4
  %h_nv = getelementptr inbounds nuw i8, ptr %ph, i64 12
  store i32 0, ptr %h_nv, align 4
  %h_enc = getelementptr inbounds nuw i8, ptr %ph, i64 16
  store i32 -1, ptr %h_enc, align 4
  br label %loop
loop:
  %rcf = call i32 @universe_encoding_thrift_field(ptr %r, ptr %ct8, ptr %fid)
  %okf = icmp eq i32 %rcf, 0
  br i1 %okf, label %chk, label %err.bad
chk:
  %ct = load i8, ptr %ct8, align 1
  %stop = icmp eq i8 %ct, 0
  br i1 %stop, label %done, label %disp
disp:
  %id = load i16, ptr %fid, align 2
  %id32 = sext i16 %id to i32
  switch i32 %id32, label %skip [
    i32 1, label %f_type
    i32 2, label %f_us
    i32 3, label %f_cs
    i32 5, label %f_sub
    i32 7, label %f_sub
    i32 8, label %f_sub
  ]
f_type:
  %rt = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okt = icmp eq i32 %rt, 0
  br i1 %okt, label %f_type.st, label %err.bad
f_type.st:
  %tv = load i32, ptr %v32, align 4
  store i32 %tv, ptr %h_type, align 4
  br label %loop
f_us:
  %rus = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okus = icmp eq i32 %rus, 0
  br i1 %okus, label %f_us.st, label %err.bad
f_us.st:
  %usv = load i32, ptr %v32, align 4
  store i32 %usv, ptr %h_us, align 4
  br label %loop
f_cs:
  %rcs = call i32 @universe_encoding_thrift_read_i32(ptr %r, ptr %v32)
  %okcs = icmp eq i32 %rcs, 0
  br i1 %okcs, label %f_cs.st, label %err.bad
f_cs.st:
  %csv = load i32, ptr %v32, align 4
  store i32 %csv, ptr %h_cs, align 4
  br label %loop
f_sub:
  %rsub = call i32 @pq.parse_page_sub(ptr %r, ptr %ph)
  %oksub = icmp eq i32 %rsub, 0
  br i1 %oksub, label %loop, label %err.bad
skip:
  %ct32 = zext i8 %ct to i32
  %rck = call i32 @universe_encoding_thrift_skip(ptr %r, i32 %ct32, i32 0)
  %okk = icmp eq i32 %rck, 0
  br i1 %okk, label %loop, label %err.bad
done:
  %np = call i64 @pq.pos(ptr %r)
  store i64 %np, ptr %onext, align 8
  ret i32 0
err.null:
  ret i32 1
err.bad:
  ret i32 8
}

attributes #0 = { nounwind norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
