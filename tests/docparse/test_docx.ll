; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Test driver for the DOCX text extractor. Fixtures are Python-built .docx
; ZIPs embedded via fixture_docx.ll (main: two paragraphs, a tab, a br, and an
; &amp; entity; empty: an empty <w:body>). The XLSX fixture (a valid ZIP with
; NO word/document.xml) is reused for the NOT_FOUND case.

declare ptr @fixture_docx_ptr()
declare i64 @fixture_docx_len()
declare ptr @fixture_docx_empty_ptr()
declare i64 @fixture_docx_empty_len()
declare ptr @fixture_xlsx_ptr()
declare i64 @fixture_xlsx_len()

declare i32 @universe_docparse_docx_text(ptr, i64, ptr, i64, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare i64 @ut_rand(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

@expected = private constant [34 x i8] c"Hello & World\0ALine1\09After\0ANewLine\0A"
@junk     = private constant [16 x i8] c"not a zip file!!"

@m.open   = private constant [16 x i8] c"docx open rc=0\0A\00"
@m.len    = private constant [13 x i8] c"docx out_len\00"
@m.text   = private constant [16 x i8] c"docx text bytes\00"
@m.nf     = private constant [20 x i8] c"docx NOT_FOUND (5)\0A\00"
@m.notzip = private constant [20 x i8] c"docx not-a-zip (8)\0A\00"
@m.full   = private constant [15 x i8] c"docx FULL (6)\0A\00"
@m.empty  = private constant [20 x i8] c"docx empty body rc\0A\00"
@m.emptyl = private constant [22 x i8] c"docx empty out_len==0\00"
@m.null   = private constant [18 x i8] c"docx NULL_PTR (1)\00"
@m.fuzz   = private constant [20 x i8] c"docx fuzz rc valid\0A\00"

@obuf = internal global [256 x i8] zeroinitializer, align 16
@fbuf = internal global [700 x i8] zeroinitializer, align 16

; compare (buf,len) to (cstr,clen)
define internal i1 @seq(ptr %buf, i64 %len, ptr %cstr, i64 %clen) {
entry:
  %lok = icmp eq i64 %len, %clen
  br i1 %lok, label %loop, label %no
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %d = icmp uge i64 %i, %len
  br i1 %d, label %yes, label %body
body:
  %ap = getelementptr inbounds i8, ptr %buf, i64 %i
  %a = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %cstr, i64 %i
  %b = load i8, ptr %bp, align 1
  %eq = icmp eq i8 %a, %b
  br i1 %eq, label %cont, label %no
cont:
  %i.n = add i64 %i, 1
  br label %loop
yes:
  ret i1 true
no:
  ret i1 false
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %outlen = alloca i64, align 8

  ; ---- happy path: main fixture ----
  %buf = call ptr @fixture_docx_ptr()
  %len = call i64 @fixture_docx_len()
  %rc = call i32 @universe_docparse_docx_text(ptr %buf, i64 %len, ptr @obuf, i64 256, ptr %outlen)
  %rcok = icmp eq i32 %rc, 0
  call void @ut_check(i1 %rcok, ptr @m.open)
  %ol = load i64, ptr %outlen, align 8
  call void @ut_check_eq(i64 %ol, i64 34, ptr @m.len)
  %texteq = call i1 @seq(ptr @obuf, i64 %ol, ptr @expected, i64 34)
  call void @ut_check(i1 %texteq, ptr @m.text)

  ; ---- NOT_FOUND: a valid ZIP with no word/document.xml (reuse xlsx) ----
  %xbuf = call ptr @fixture_xlsx_ptr()
  %xlen = call i64 @fixture_xlsx_len()
  %rcnf = call i32 @universe_docparse_docx_text(ptr %xbuf, i64 %xlen, ptr @obuf, i64 256, ptr %outlen)
  %nfok = icmp eq i32 %rcnf, 5
  call void @ut_check(i1 %nfok, ptr @m.nf)

  ; ---- not a zip -> malformed (8) ----
  %rcnz = call i32 @universe_docparse_docx_text(ptr @junk, i64 16, ptr @obuf, i64 256, ptr %outlen)
  %nzok = icmp eq i32 %rcnz, 8
  call void @ut_check(i1 %nzok, ptr @m.notzip)

  ; ---- out_cap one byte too small -> FULL (6) ----
  %rcf = call i32 @universe_docparse_docx_text(ptr %buf, i64 %len, ptr @obuf, i64 33, ptr %outlen)
  %fok = icmp eq i32 %rcf, 6
  call void @ut_check(i1 %fok, ptr @m.full)

  ; ---- empty body -> rc 0, out_len 0 ----
  %ebuf = call ptr @fixture_docx_empty_ptr()
  %elen = call i64 @fixture_docx_empty_len()
  %rce = call i32 @universe_docparse_docx_text(ptr %ebuf, i64 %elen, ptr @obuf, i64 256, ptr %outlen)
  %eok = icmp eq i32 %rce, 0
  call void @ut_check(i1 %eok, ptr @m.empty)
  %eol = load i64, ptr %outlen, align 8
  %eolok = icmp eq i64 %eol, 0
  call void @ut_check(i1 %eolok, ptr @m.emptyl)

  ; ---- NULL args -> 1 ----
  %rcnull = call i32 @universe_docparse_docx_text(ptr null, i64 0, ptr @obuf, i64 256, ptr %outlen)
  %nullok = icmp eq i32 %rcnull, 1
  call void @ut_check(i1 %nullok, ptr @m.null)

  ; ---- fuzz: truncated + byte-mutated copies of the main zip ----
  ; every call must return a documented code (never crash / never a stray code).
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  br label %fz.loop
fz.loop:
  %fi = phi i64 [ 0, %entry ], [ %fi.n, %fz.cont ]
  %bad = phi i64 [ 0, %entry ], [ %bad.i, %fz.cont ]
  %fdone = icmp uge i64 %fi, 3000
  br i1 %fdone, label %fz.done, label %fz.body
fz.body:
  ; fresh copy of the main zip into fbuf
  call void @llvm.memcpy.p0.p0.i64(ptr @fbuf, ptr %buf, i64 %len, i1 false)
  ; random truncation length in [0, len]
  %r0 = call i64 @ut_rand(ptr %seed)
  %tmod = add i64 %len, 1
  %tlen = urem i64 %r0, %tmod
  ; mutate a random byte within the truncated region (if non-empty)
  %hasb = icmp ugt i64 %tlen, 0
  br i1 %hasb, label %fz.mut, label %fz.call
fz.mut:
  %r1 = call i64 @ut_rand(ptr %seed)
  %pos = urem i64 %r1, %tlen
  %r2 = call i64 @ut_rand(ptr %seed)
  %bv = trunc i64 %r2 to i8
  %mp = getelementptr inbounds i8, ptr @fbuf, i64 %pos
  store i8 %bv, ptr %mp, align 1
  br label %fz.call
fz.call:
  %frc = call i32 @universe_docparse_docx_text(ptr @fbuf, i64 %tlen, ptr @obuf, i64 256, ptr %outlen)
  ; valid codes: 0,2,5,6,8 (never 1 here: pointers are non-null)
  %v0 = icmp eq i32 %frc, 0
  %v2 = icmp eq i32 %frc, 2
  %v5 = icmp eq i32 %frc, 5
  %v6 = icmp eq i32 %frc, 6
  %v8 = icmp eq i32 %frc, 8
  %va = or i1 %v0, %v2
  %vb = or i1 %v5, %v6
  %vc = or i1 %va, %vb
  %valid = or i1 %vc, %v8
  %isbad = xor i1 %valid, true
  %badinc = zext i1 %isbad to i64
  %bad.i = add i64 %bad, %badinc
  br label %fz.cont
fz.cont:
  %fi.n = add i64 %fi, 1
  br label %fz.loop
fz.done:
  call void @ut_check_eq(i64 %bad, i64 0, ptr @m.fuzz)

  %code = call i32 @ut_summary()
  ret i32 %code
}
