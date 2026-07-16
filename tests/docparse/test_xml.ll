; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Test driver for the minimal XML pull-tokenizer.

declare void @universe_docparse_xml_init(ptr, ptr, i64)
declare i32 @universe_docparse_xml_next(ptr, ptr)
declare i32 @universe_docparse_xml_attr_next(ptr, ptr, i64, ptr)
declare i64 @universe_docparse_xml_decode(ptr, ptr, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @memcmp(ptr, ptr, i64)

; <root a="1"><t>Hello</t><self/><t>A&amp;B</t><!--c--><?pi?></root>
@doc = private constant [66 x i8] c"<root a=\221\22><t>Hello</t><self/><t>A&amp;B</t><!--c--><?pi?></root>"
@doclen = private constant i64 66

@s.root = private constant [4 x i8] c"root"
@s.t    = private constant [1 x i8] c"t"
@s.self = private constant [4 x i8] c"self"
@s.a    = private constant [1 x i8] c"a"
@s.one  = private constant [1 x i8] c"1"
@s.hello = private constant [5 x i8] c"Hello"
@s.amp  = private constant [7 x i8] c"A&amp;B"
@s.ab   = private constant [3 x i8] c"A&B"
@s.num  = private constant [11 x i8] c"&#65;&#x42;"

@m.type = private constant [10 x i8] c"tok type\0A\00"
@m.name = private constant [10 x i8] c"tok name\0A\00"
@m.attr = private constant [10 x i8] c"attr nam\0A\00"
@m.aval = private constant [10 x i8] c"attr val\0A\00"
@m.dec  = private constant [10 x i8] c"decode  \0A\00"
@m.declen = private constant [10 x i8] c"declen  \0A\00"

; compare buffer slice (buf+off,len) to (cstr,clen); returns i1 equal
define internal i1 @slice_eq(ptr %buf, i64 %off, i64 %len, ptr %cstr, i64 %clen) {
entry:
  %lenok = icmp eq i64 %len, %clen
  br i1 %lenok, label %cmp, label %no
cmp:
  %p = getelementptr inbounds i8, ptr %buf, i64 %off
  %r = call i32 @memcmp(ptr %p, ptr %cstr, i64 %len)
  %eq = icmp eq i32 %r, 0
  ret i1 %eq
no:
  ret i1 false
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %sc = alloca [24 x i8], align 8
  %tok = alloca [40 x i8], align 8
  %attr = alloca [32 x i8], align 8
  %cur = alloca i64, align 8
  %dbuf = alloca [64 x i8], align 8

  %buf = getelementptr inbounds i8, ptr @doc, i64 0
  %len = load i64, ptr @doclen, align 8
  call void @universe_docparse_xml_init(ptr %sc, ptr %buf, i64 %len)

  ; token pointers
  %ptype = getelementptr inbounds i8, ptr %tok, i64 0
  %pnoff = getelementptr inbounds i8, ptr %tok, i64 8
  %pnlen = getelementptr inbounds i8, ptr %tok, i64 16
  %paoff = getelementptr inbounds i8, ptr %tok, i64 24
  %palen = getelementptr inbounds i8, ptr %tok, i64 32

  ; ---- token 1: start "root" with attr a="1"
  %r1 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t1 = zext i32 %r1 to i64
  call void @ut_check_eq(i64 %t1, i64 0, ptr @m.type)
  %n1o = load i64, ptr %pnoff, align 8
  %n1l = load i64, ptr %pnlen, align 8
  %rootok = call i1 @slice_eq(ptr %buf, i64 %n1o, i64 %n1l, ptr @s.root, i64 4)
  call void @ut_check(i1 %rootok, ptr @m.name)
  ; attributes
  %a1o = load i64, ptr %paoff, align 8
  %a1l = load i64, ptr %palen, align 8
  %aend = add i64 %a1o, %a1l
  store i64 %a1o, ptr %cur, align 8
  %ar = call i32 @universe_docparse_xml_attr_next(ptr %buf, ptr %cur, i64 %aend, ptr %attr)
  %arok = icmp eq i32 %ar, 1
  call void @ut_check(i1 %arok, ptr @m.attr)
  %ano = load i64, ptr %attr, align 8
  %anlp = getelementptr inbounds i8, ptr %attr, i64 8
  %anl = load i64, ptr %anlp, align 8
  %anok = call i1 @slice_eq(ptr %buf, i64 %ano, i64 %anl, ptr @s.a, i64 1)
  call void @ut_check(i1 %anok, ptr @m.attr)
  %avop = getelementptr inbounds i8, ptr %attr, i64 16
  %avo = load i64, ptr %avop, align 8
  %avlp = getelementptr inbounds i8, ptr %attr, i64 24
  %avl = load i64, ptr %avlp, align 8
  %avok = call i1 @slice_eq(ptr %buf, i64 %avo, i64 %avl, ptr @s.one, i64 1)
  call void @ut_check(i1 %avok, ptr @m.aval)
  ; no more attrs
  %ar2 = call i32 @universe_docparse_xml_attr_next(ptr %buf, ptr %cur, i64 %aend, ptr %attr)
  %ar2ok = icmp eq i32 %ar2, 0
  call void @ut_check(i1 %ar2ok, ptr @m.attr)

  ; ---- token 2: start "t"
  %r2 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t2 = zext i32 %r2 to i64
  call void @ut_check_eq(i64 %t2, i64 0, ptr @m.type)
  %n2o = load i64, ptr %pnoff, align 8
  %n2l = load i64, ptr %pnlen, align 8
  %tok2 = call i1 @slice_eq(ptr %buf, i64 %n2o, i64 %n2l, ptr @s.t, i64 1)
  call void @ut_check(i1 %tok2, ptr @m.name)

  ; ---- token 3: text "Hello"
  %r3 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t3 = zext i32 %r3 to i64
  call void @ut_check_eq(i64 %t3, i64 3, ptr @m.type)
  %n3o = load i64, ptr %pnoff, align 8
  %n3l = load i64, ptr %pnlen, align 8
  %hok = call i1 @slice_eq(ptr %buf, i64 %n3o, i64 %n3l, ptr @s.hello, i64 5)
  call void @ut_check(i1 %hok, ptr @m.name)

  ; ---- token 4: end "t"
  %r4 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t4 = zext i32 %r4 to i64
  call void @ut_check_eq(i64 %t4, i64 1, ptr @m.type)

  ; ---- token 5: self "self"
  %r5 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t5 = zext i32 %r5 to i64
  call void @ut_check_eq(i64 %t5, i64 2, ptr @m.type)
  %n5o = load i64, ptr %pnoff, align 8
  %n5l = load i64, ptr %pnlen, align 8
  %selfok = call i1 @slice_eq(ptr %buf, i64 %n5o, i64 %n5l, ptr @s.self, i64 4)
  call void @ut_check(i1 %selfok, ptr @m.name)

  ; ---- token 6: start "t"
  %r6 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t6 = zext i32 %r6 to i64
  call void @ut_check_eq(i64 %t6, i64 0, ptr @m.type)

  ; ---- token 7: text "A&amp;B" -> decode "A&B"
  %r7 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t7 = zext i32 %r7 to i64
  call void @ut_check_eq(i64 %t7, i64 3, ptr @m.type)
  %n7o = load i64, ptr %pnoff, align 8
  %n7l = load i64, ptr %pnlen, align 8
  %rawok = call i1 @slice_eq(ptr %buf, i64 %n7o, i64 %n7l, ptr @s.amp, i64 7)
  call void @ut_check(i1 %rawok, ptr @m.name)
  %srcp = getelementptr inbounds i8, ptr %buf, i64 %n7o
  %dl = call i64 @universe_docparse_xml_decode(ptr %dbuf, ptr %srcp, i64 %n7l)
  call void @ut_check_eq(i64 %dl, i64 3, ptr @m.declen)
  %decok = call i1 @slice_eq(ptr %dbuf, i64 0, i64 %dl, ptr @s.ab, i64 3)
  call void @ut_check(i1 %decok, ptr @m.dec)

  ; ---- token 8: end "t"
  %r8 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t8 = zext i32 %r8 to i64
  call void @ut_check_eq(i64 %t8, i64 1, ptr @m.type)

  ; ---- token 9: end "root" (comment + PI skipped)
  %r9 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t9 = zext i32 %r9 to i64
  call void @ut_check_eq(i64 %t9, i64 1, ptr @m.type)
  %n9o = load i64, ptr %pnoff, align 8
  %n9l = load i64, ptr %pnlen, align 8
  %endrootok = call i1 @slice_eq(ptr %buf, i64 %n9o, i64 %n9l, ptr @s.root, i64 4)
  call void @ut_check(i1 %endrootok, ptr @m.name)

  ; ---- token 10: eof
  %r10 = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %t10 = zext i32 %r10 to i64
  call void @ut_check_eq(i64 %t10, i64 4, ptr @m.type)

  ; ---- numeric entity decode "&#65;&#x42;" -> "AB"
  %numsrc = getelementptr inbounds i8, ptr @s.num, i64 0
  %dl2 = call i64 @universe_docparse_xml_decode(ptr %dbuf, ptr %numsrc, i64 11)
  call void @ut_check_eq(i64 %dl2, i64 2, ptr @m.declen)
  %p0 = getelementptr inbounds i8, ptr %dbuf, i64 0
  %c0 = load i8, ptr %p0, align 1
  %c0ok = icmp eq i8 %c0, 65
  call void @ut_check(i1 %c0ok, ptr @m.dec)
  %p1 = getelementptr inbounds i8, ptr %dbuf, i64 1
  %c1 = load i8, ptr %p1, align 1
  %c1ok = icmp eq i8 %c1, 66
  call void @ut_check(i1 %c1ok, ptr @m.dec)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
