; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; KAT for docparse/parquet_meta: decode REAL DuckDB-written Parquet files and
; assert their metadata against ground truth extracted from the files (mirrors
; the RAG doc_parquet_test.zig expectations). Plus framing error paths and a
; truncated-footer fuzz loop (run under ASan for OOB coverage).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()

declare i32 @universe_docparse_parquet_footer(ptr, i64, ptr, ptr)
declare i32 @universe_docparse_parquet_meta_open(ptr, i64, ptr)
declare i64 @universe_docparse_parquet_meta_num_rows(ptr)
declare i64 @universe_docparse_parquet_meta_row_group_count(ptr)
declare i32 @universe_docparse_parquet_schema_next(ptr, ptr, ptr)
declare i32 @universe_docparse_parquet_row_group(ptr, i64, ptr)
declare i32 @universe_docparse_parquet_column(ptr, i64, ptr)
declare i32 @universe_docparse_parquet_page_header(ptr, i64, i64, ptr, ptr)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; Fixtures (embedded real Parquet files) + their byte lengths.
@fixture_pq_flat_uncompressed = external constant [0 x i8]
@fixture_pq_flat_uncompressed_len = external constant i64
@fixture_pq_flat_snappy = external constant [0 x i8]
@fixture_pq_flat_snappy_len = external constant i64
@fixture_pq_flat_gzip = external constant [0 x i8]
@fixture_pq_flat_gzip_len = external constant i64
@fixture_pq_flat_zstd = external constant [0 x i8]
@fixture_pq_flat_zstd_len = external constant i64
@fixture_pq_dict = external constant [0 x i8]
@fixture_pq_dict_len = external constant i64
@fixture_pq_types = external constant [0 x i8]
@fixture_pq_types_len = external constant i64

@m.footer   = private constant [17 x i8] c"footer ok       \00"
@m.open     = private constant [17 x i8] c"meta_open ok    \00"
@m.metalen  = private constant [17 x i8] c"meta_len > 0    \00"
@m.numrows  = private constant [17 x i8] c"num_rows        \00"
@m.rgcount  = private constant [17 x i8] c"rg_count == 1   \00"
@m.schcount = private constant [17 x i8] c"schema count    \00"
@m.roottype = private constant [17 x i8] c"root type == -1 \00"
@m.col0type = private constant [17 x i8] c"schema col0 type\00"
@m.rgrc     = private constant [17 x i8] c"row_group ok    \00"
@m.rgnr     = private constant [17 x i8] c"rg.num_rows     \00"
@m.rgcc     = private constant [17 x i8] c"rg.col_count    \00"
@m.colrc    = private constant [17 x i8] c"column ok       \00"
@m.cctype   = private constant [17 x i8] c"cc.type         \00"
@m.cccodec  = private constant [17 x i8] c"cc.codec        \00"
@m.ccdpo    = private constant [17 x i8] c"cc.data_page_off\00"
@m.phrc     = private constant [17 x i8] c"page_header ok  \00"
@m.phtype   = private constant [17 x i8] c"page type       \00"
@m.phnv     = private constant [17 x i8] c"page num_values \00"
@m.schrc    = private constant [17 x i8] c"schema_next rc  \00"

@m.enonpar  = private constant [17 x i8] c"non-PAR1 -> 8   \00"
@m.eshort   = private constant [17 x i8] c"len<12 -> 8     \00"
@m.etrunc   = private constant [17 x i8] c"trunc tail -> 8 \00"
@m.ebigflen = private constant [17 x i8] c"big flen -> 8   \00"
@m.eopen0   = private constant [17 x i8] c"open len0 -> 8  \00"
@m.fuzz     = private constant [17 x i8] c"fuzz rc in {0,8}\00"

@m.tconv1   = private constant [17 x i8] c"types conv[1]=5 \00"
@m.tconv2   = private constant [17 x i8] c"types conv[2]=6 \00"
@m.tconv3   = private constant [17 x i8] c"types conv[3]10 \00"
@m.tconv4   = private constant [17 x i8] c"types conv[4]17 \00"

@bad.nonpar = private constant [20 x i8] c"not a parquet file!!"

; ============================================================ check_file
; Full metadata assertion for one fixture buffer.
define void @check_file(ptr %buf, i64 %len, i64 %exp_nr, i32 %exp_sc, i32 %exp_t0, i32 %exp_codec, i64 %exp_dpo, i32 %exp_ptype, i32 %exp_pnv) {
entry:
  %mptr = alloca ptr, align 8
  %mlen = alloca i64, align 8
  %fmd  = alloca [64 x i8], align 8
  %cur  = alloca [16 x i8], align 8
  %elem = alloca [48 x i8], align 8
  %rg   = alloca [48 x i8], align 8
  %cc   = alloca [56 x i8], align 8
  %ph   = alloca [24 x i8], align 8
  %next = alloca i64, align 8

  %rcf = call i32 @universe_docparse_parquet_footer(ptr %buf, i64 %len, ptr %mptr, ptr %mlen)
  %rcf0 = icmp eq i32 %rcf, 0
  call void @ut_check(i1 %rcf0, ptr @m.footer)
  %mp = load ptr, ptr %mptr, align 8
  %ml = load i64, ptr %mlen, align 8
  %mlpos = icmp sgt i64 %ml, 0
  call void @ut_check(i1 %mlpos, ptr @m.metalen)

  %rco = call i32 @universe_docparse_parquet_meta_open(ptr %mp, i64 %ml, ptr %fmd)
  %rco0 = icmp eq i32 %rco, 0
  call void @ut_check(i1 %rco0, ptr @m.open)

  %nr = call i64 @universe_docparse_parquet_meta_num_rows(ptr %fmd)
  call void @ut_check_eq(i64 %nr, i64 %exp_nr, ptr @m.numrows)
  %rgc = call i64 @universe_docparse_parquet_meta_row_group_count(ptr %fmd)
  call void @ut_check_eq(i64 %rgc, i64 1, ptr @m.rgcount)

  ; ---- schema iteration ----
  call void @llvm.memset.p0.i64(ptr %cur, i8 0, i64 16, i1 false)
  br label %sloop
sloop:
  %idx = phi i32 [ 0, %entry ], [ %idxn, %scont ]
  %rcs = call i32 @universe_docparse_parquet_schema_next(ptr %fmd, ptr %cur, ptr %elem)
  %isend = icmp eq i32 %rcs, 5
  br i1 %isend, label %sdone, label %sgo
sgo:
  %rcsok = icmp eq i32 %rcs, 0
  call void @ut_check(i1 %rcsok, ptr @m.schrc)
  %etype = load i32, ptr %elem, align 4
  %etype64 = sext i32 %etype to i64
  ; index 0 = root (type absent = -1); index 1 = first column (exp_t0)
  %is0 = icmp eq i32 %idx, 0
  br i1 %is0, label %chk0, label %chk1q
chk0:
  call void @ut_check_eq(i64 %etype64, i64 -1, ptr @m.roottype)
  br label %scont
chk1q:
  %is1 = icmp eq i32 %idx, 1
  br i1 %is1, label %chk1, label %scont
chk1:
  %t064 = sext i32 %exp_t0 to i64
  call void @ut_check_eq(i64 %etype64, i64 %t064, ptr @m.col0type)
  br label %scont
scont:
  %idxn = add nuw nsw i32 %idx, 1
  br label %sloop
sdone:
  %sc64 = sext i32 %exp_sc to i64
  %idx64 = sext i32 %idx to i64
  call void @ut_check_eq(i64 %idx64, i64 %sc64, ptr @m.schcount)

  ; ---- row group 0 ----
  %rcrg = call i32 @universe_docparse_parquet_row_group(ptr %fmd, i64 0, ptr %rg)
  %rcrg0 = icmp eq i32 %rcrg, 0
  call void @ut_check(i1 %rcrg0, ptr @m.rgrc)
  %p_rgnr = getelementptr inbounds i8, ptr %rg, i64 32
  %rgnr = load i64, ptr %p_rgnr, align 8
  call void @ut_check_eq(i64 %rgnr, i64 %exp_nr, ptr @m.rgnr)
  %p_rgcc = getelementptr inbounds i8, ptr %rg, i64 16
  %rgcolc = load i32, ptr %p_rgcc, align 4
  %rgcolc64 = sext i32 %rgcolc to i64
  %expcols = sub nsw i32 %exp_sc, 1
  %expcols64 = sext i32 %expcols to i64
  call void @ut_check_eq(i64 %rgcolc64, i64 %expcols64, ptr @m.rgcc)

  ; ---- column 0 ----
  %rccol = call i32 @universe_docparse_parquet_column(ptr %rg, i64 0, ptr %cc)
  %rccol0 = icmp eq i32 %rccol, 0
  call void @ut_check(i1 %rccol0, ptr @m.colrc)
  %cctype = load i32, ptr %cc, align 4
  %cctype64 = sext i32 %cctype to i64
  %t064b = sext i32 %exp_t0 to i64
  call void @ut_check_eq(i64 %cctype64, i64 %t064b, ptr @m.cctype)
  %p_codec = getelementptr inbounds i8, ptr %cc, i64 4
  %codec = load i32, ptr %p_codec, align 4
  %codec64 = sext i32 %codec to i64
  %ecodec64 = sext i32 %exp_codec to i64
  call void @ut_check_eq(i64 %codec64, i64 %ecodec64, ptr @m.cccodec)
  %p_dpo = getelementptr inbounds i8, ptr %cc, i64 24
  %dpo = load i64, ptr %p_dpo, align 8
  call void @ut_check_eq(i64 %dpo, i64 %exp_dpo, ptr @m.ccdpo)

  ; ---- page header at data_page_offset (into the whole file buffer) ----
  %rcph = call i32 @universe_docparse_parquet_page_header(ptr %buf, i64 %len, i64 %exp_dpo, ptr %ph, ptr %next)
  %rcph0 = icmp eq i32 %rcph, 0
  call void @ut_check(i1 %rcph0, ptr @m.phrc)
  %phtype = load i32, ptr %ph, align 4
  %phtype64 = sext i32 %phtype to i64
  %eptype64 = sext i32 %exp_ptype to i64
  call void @ut_check_eq(i64 %phtype64, i64 %eptype64, ptr @m.phtype)
  %p_phnv = getelementptr inbounds i8, ptr %ph, i64 12
  %phnv = load i32, ptr %p_phnv, align 4
  %phnv64 = sext i32 %phnv to i64
  %epnv64 = sext i32 %exp_pnv to i64
  call void @ut_check_eq(i64 %phnv64, i64 %epnv64, ptr @m.phnv)
  ret void
}

; nth schema element's converted_type (offset 16) for the types fixture check.
define i64 @schema_conv_at(ptr %fmd, i32 %want) {
entry:
  %cur  = alloca [16 x i8], align 8
  %elem = alloca [48 x i8], align 8
  call void @llvm.memset.p0.i64(ptr %cur, i8 0, i64 16, i1 false)
  br label %loop
loop:
  %idx = phi i32 [ 0, %entry ], [ %idxn, %cont ]
  %rc = call i32 @universe_docparse_parquet_schema_next(ptr %fmd, ptr %cur, ptr %elem)
  %end = icmp ne i32 %rc, 0
  br i1 %end, label %fail, label %go
go:
  %hit = icmp eq i32 %idx, %want
  br i1 %hit, label %found, label %cont
found:
  %p = getelementptr inbounds i8, ptr %elem, i64 16
  %v = load i32, ptr %p, align 4
  %v64 = sext i32 %v to i64
  ret i64 %v64
cont:
  %idxn = add nuw nsw i32 %idx, 1
  br label %loop
fail:
  ret i64 -999
}

; ==================================================================== main
define i32 @main(i32 %argc, ptr %argv) {
entry:
  %rngst = alloca i64, align 8
  %fmd   = alloca [64 x i8], align 8
  %mptr  = alloca ptr, align 8
  %mlen  = alloca i64, align 8
  %scratch = alloca [1024 x i8], align 8

  ; flat, every codec: id BIGINT(2), name UTF8(6), score DOUBLE(5), flag BOOL(0)
  %u = getelementptr inbounds i8, ptr @fixture_pq_flat_uncompressed, i64 0
  %ul = load i64, ptr @fixture_pq_flat_uncompressed_len, align 8
  call void @check_file(ptr %u, i64 %ul, i64 4, i32 5, i32 2, i32 0, i64 4, i32 0, i32 4)

  %s = getelementptr inbounds i8, ptr @fixture_pq_flat_snappy, i64 0
  %sl = load i64, ptr @fixture_pq_flat_snappy_len, align 8
  call void @check_file(ptr %s, i64 %sl, i64 4, i32 5, i32 2, i32 1, i64 4, i32 0, i32 4)

  %g = getelementptr inbounds i8, ptr @fixture_pq_flat_gzip, i64 0
  %gl = load i64, ptr @fixture_pq_flat_gzip_len, align 8
  call void @check_file(ptr %g, i64 %gl, i64 4, i32 5, i32 2, i32 2, i64 4, i32 0, i32 4)

  %z = getelementptr inbounds i8, ptr @fixture_pq_flat_zstd, i64 0
  %zl = load i64, ptr @fixture_pq_flat_zstd_len, align 8
  call void @check_file(ptr %z, i64 %zl, i64 4, i32 5, i32 2, i32 6, i64 4, i32 0, i32 4)

  ; dict: id BIGINT(2), color UTF8(6); 1000 rows; col0 dict page@4, data page@43
  %d = getelementptr inbounds i8, ptr @fixture_pq_dict, i64 0
  %dl = load i64, ptr @fixture_pq_dict_len, align 8
  call void @check_file(ptr %d, i64 %dl, i64 1000, i32 3, i32 2, i32 1, i64 43, i32 0, i32 1000)

  ; types: amount DECIMAL INT32(1), d DATE INT32(1), ts TS INT64(2), n INT32(1)
  %t = getelementptr inbounds i8, ptr @fixture_pq_types, i64 0
  %tl = load i64, ptr @fixture_pq_types_len, align 8
  call void @check_file(ptr %t, i64 %tl, i64 1, i32 5, i32 1, i32 0, i64 4, i32 0, i32 1)

  ; types fixture: verify converted_type per schema element (5,6,10,17).
  %rcft = call i32 @universe_docparse_parquet_footer(ptr %t, i64 %tl, ptr %mptr, ptr %mlen)
  %tmp = load ptr, ptr %mptr, align 8
  %tml = load i64, ptr %mlen, align 8
  %rcot = call i32 @universe_docparse_parquet_meta_open(ptr %tmp, i64 %tml, ptr %fmd)
  %c1 = call i64 @schema_conv_at(ptr %fmd, i32 1)
  call void @ut_check_eq(i64 %c1, i64 5, ptr @m.tconv1)
  %c2 = call i64 @schema_conv_at(ptr %fmd, i32 2)
  call void @ut_check_eq(i64 %c2, i64 6, ptr @m.tconv2)
  %c3 = call i64 @schema_conv_at(ptr %fmd, i32 3)
  call void @ut_check_eq(i64 %c3, i64 10, ptr @m.tconv3)
  %c4 = call i64 @schema_conv_at(ptr %fmd, i32 4)
  call void @ut_check_eq(i64 %c4, i64 17, ptr @m.tconv4)

  ; ---- error: non-PAR1 buffer -> 8 ----
  %e1 = call i32 @universe_docparse_parquet_footer(ptr @bad.nonpar, i64 20, ptr %mptr, ptr %mlen)
  %e1ok = icmp eq i32 %e1, 8
  call void @ut_check(i1 %e1ok, ptr @m.enonpar)

  ; ---- error: len < 12 -> 8 (use a valid fixture but a tiny length) ----
  %e2 = call i32 @universe_docparse_parquet_footer(ptr %u, i64 4, ptr %mptr, ptr %mlen)
  %e2ok = icmp eq i32 %e2, 8
  call void @ut_check(i1 %e2ok, ptr @m.eshort)

  ; ---- error: truncated tail (drop the last byte -> tail magic misaligned) -> 8 ----
  %ulm1 = sub i64 %ul, 1
  %e3 = call i32 @universe_docparse_parquet_footer(ptr %u, i64 %ulm1, ptr %mptr, ptr %mlen)
  %e3ok = icmp eq i32 %e3, 8
  call void @ut_check(i1 %e3ok, ptr @m.etrunc)

  ; ---- error: oversized footer_len -> 8. Copy fixture, patch len-8 u32 = huge ----
  call void @llvm.memcpy.p0.p0.i64(ptr %scratch, ptr %u, i64 %ul, i1 false)
  %off = sub i64 %ul, 8
  %flp = getelementptr inbounds i8, ptr %scratch, i64 %off
  store i32 65535, ptr %flp, align 1
  %e4 = call i32 @universe_docparse_parquet_footer(ptr %scratch, i64 %ul, ptr %mptr, ptr %mlen)
  %e4ok = icmp eq i32 %e4, 8
  call void @ut_check(i1 %e4ok, ptr @m.ebigflen)

  ; ---- error: meta_open over zero-length meta -> 8 ----
  %e5 = call i32 @universe_docparse_parquet_meta_open(ptr %u, i64 0, ptr %fmd)
  %e5ok = icmp eq i32 %e5, 8
  call void @ut_check(i1 %e5ok, ptr @m.eopen0)

  ; ---- fuzz: mutate footer_len + reported length over the valid framing;
  ;      footer must return 0 or 8 (never crash / OOB — the point under ASan).
  ;      If it returns 0, meta_open on the slice must not OOB either. Aggregate
  ;      any undocumented return into one violation counter. ----
  store i64 88172645463325252, ptr %rngst, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %scratch, ptr %u, i64 %ul, i1 false)
  br label %floop
floop:
  %fi = phi i32 [ 0, %entry ], [ %fin, %fnext ]
  %viol = phi i32 [ 0, %entry ], [ %violn, %fnext ]
  %rv = call i64 @ut_rand(ptr %rngst)
  %rv32 = trunc i64 %rv to i32
  %flp2 = getelementptr inbounds i8, ptr %scratch, i64 %off
  store i32 %rv32, ptr %flp2, align 1
  %rv2 = call i64 @ut_rand(ptr %rngst)
  %span = sub i64 %ul, 11
  %rem = urem i64 %rv2, %span
  %flen2 = add i64 %rem, 12
  %frc = call i32 @universe_docparse_parquet_footer(ptr %scratch, i64 %flen2, ptr %mptr, ptr %mlen)
  %f0 = icmp eq i32 %frc, 0
  %f8 = icmp eq i32 %frc, 8
  %fok = or i1 %f0, %f8
  %badinc = select i1 %fok, i32 0, i32 1
  %violm = add nuw nsw i32 %viol, %badinc
  br i1 %f0, label %fopen, label %fnext
fopen:
  %sp = load ptr, ptr %mptr, align 8
  %sl2 = load i64, ptr %mlen, align 8
  %orc = call i32 @universe_docparse_parquet_meta_open(ptr %sp, i64 %sl2, ptr %fmd)
  ; return value unimportant here; ASan verifies no OOB on hostile slices.
  br label %fnext
fnext:
  %violn = phi i32 [ %violm, %floop ], [ %violm, %fopen ]
  %fin = add nuw nsw i32 %fi, 1
  %fdone = icmp sge i32 %fin, 4000
  br i1 %fdone, label %fend, label %floop
fend:
  %violn64 = sext i32 %violn to i64
  call void @ut_check_eq(i64 %violn64, i64 0, ptr @m.fuzz)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
