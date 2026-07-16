; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Test driver for the PDF reader (xref/trailer, FlateDecode, text extract).

declare ptr @fixture_pdf_ptr()
declare i64 @fixture_pdf_len()
declare ptr @fixture_pdf_content_ptr()
declare i64 @fixture_pdf_content_len()

declare i32 @universe_docparse_pdf_open(ptr, i64, ptr)
declare void @universe_docparse_pdf_close(ptr)
declare i64 @universe_docparse_pdf_object_count(ptr)
declare i32 @universe_docparse_pdf_object_offset(ptr, i64, ptr)
declare i64 @universe_docparse_pdf_stream(ptr, i64, ptr, i64)
declare i64 @universe_docparse_pdf_extract_text(ptr, i64, ptr, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()

@s.text = private constant [9 x i8] c"Hello PDF"

@m.open = private constant [12 x i8] c"open failed\00"
@m.cnt  = private constant [9 x i8] c"objcount\00"
@m.off  = private constant [10 x i8] c"obj4 off\0A\00"
@m.strm = private constant [12 x i8] c"stream len\0A\00"
@m.smatch = private constant [12 x i8] c"stream eq \0A\00"
@m.text = private constant [10 x i8] c"text len\0A\00"
@m.tmatch = private constant [8 x i8] c"text eq\00"
@m.free = private constant [10 x i8] c"free ent\0A\00"

define internal i1 @seq(ptr %a, i64 %alen, ptr %b, i64 %blen) {
entry:
  %lok = icmp eq i64 %alen, %blen
  br i1 %lok, label %loop, label %no
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %d = icmp uge i64 %i, %alen
  br i1 %d, label %yes, label %body
body:
  %ap = getelementptr inbounds i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %eq = icmp eq i8 %av, %bv
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
  %doc = alloca [40 x i8], align 8
  %off = alloca i64, align 8
  %sbuf = alloca [256 x i8], align 8
  %tbuf = alloca [64 x i8], align 8

  %buf = call ptr @fixture_pdf_ptr()
  %len = call i64 @fixture_pdf_len()
  %oc = call i32 @universe_docparse_pdf_open(ptr %buf, i64 %len, ptr %doc)
  %ocok = icmp eq i32 %oc, 0
  call void @ut_check(i1 %ocok, ptr @m.open)

  ; object count == 5 (/Size)
  %cnt = call i64 @universe_docparse_pdf_object_count(ptr %doc)
  call void @ut_check_eq(i64 %cnt, i64 5, ptr @m.cnt)

  ; object 4 offset == 202
  %o4 = call i32 @universe_docparse_pdf_object_offset(ptr %doc, i64 4, ptr %off)
  %o4ok = icmp eq i32 %o4, 0
  call void @ut_check(i1 %o4ok, ptr @m.off)
  %o4v = load i64, ptr %off, align 8
  call void @ut_check_eq(i64 %o4v, i64 202, ptr @m.off)

  ; object 0 is free -> NOT_FOUND(5)
  %o0 = call i32 @universe_docparse_pdf_object_offset(ptr %doc, i64 0, ptr %off)
  %o0free = icmp eq i32 %o0, 5
  call void @ut_check(i1 %o0free, ptr @m.free)

  ; stream 4: FlateDecode -> original content
  %sn = call i64 @universe_docparse_pdf_stream(ptr %doc, i64 4, ptr %sbuf, i64 256)
  %clen = call i64 @fixture_pdf_content_len()
  call void @ut_check_eq(i64 %sn, i64 %clen, ptr @m.strm)
  %cptr = call ptr @fixture_pdf_content_ptr()
  %smok = call i1 @seq(ptr %sbuf, i64 %sn, ptr %cptr, i64 %clen)
  call void @ut_check(i1 %smok, ptr @m.smatch)

  ; extract text from the decoded content stream -> "Hello PDF"
  %tn = call i64 @universe_docparse_pdf_extract_text(ptr %sbuf, i64 %sn, ptr %tbuf, i64 64)
  call void @ut_check_eq(i64 %tn, i64 9, ptr @m.text)
  %tmok = call i1 @seq(ptr %tbuf, i64 %tn, ptr @s.text, i64 9)
  call void @ut_check(i1 %tmok, ptr @m.tmatch)

  call void @universe_docparse_pdf_close(ptr %doc)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
