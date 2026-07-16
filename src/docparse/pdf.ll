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

; PDF 2.0 (ISO 32000-2) reader: classic xref/trailer, indirect objects,
; FlateDecode content streams, and (...)Tj / [...]TJ text extraction.
;
; DESIGN:
;   * BUILDS ON compress/inflate (FlateDecode == zlib stream). The whole PDF
;     is in the caller buffer; open() locates `startxref` from the tail, reads
;     the cross-reference table, and builds a flat object->byte-offset array
;     (indexed by object number, -1 for free). This is the classic xref-TABLE
;     form (ISO 32000-2 7.5.4); cross-reference STREAMS are not yet handled
;     (documented limitation). Everything is zero-copy except the one object
;     array and the caller's stream output buffer.
;   * stream() finds an object's `stream`..`endstream` span, detects a
;     /FlateDecode filter in the object dictionary, and inflates the zlib
;     payload via universe_compress_inflate_zlib into the caller buffer;
;     unfiltered streams are copied. Length is derived from the endstream
;     marker so an indirect /Length need not be resolved.
;   * extract_text() walks a (decoded) content stream and pulls the bytes of
;     every literal `( ... )` string — the operands of the Tj/TJ text-showing
;     operators — decoding PDF string escapes (\n \r \t \b \f \( \) \\ and
;     octal \ddd) and honouring nested parens and line-continuations. Hex
;     `<..>` strings are not yet decoded (documented limitation).
;
; Document struct (40 B, caller-allocated):
;   buf@0(ptr) len@8(i64) xref_off@16(i64) obj_ptr@24(ptr) obj_count@32(i64)
;
; API:
;   i32 universe_docparse_pdf_open(ptr buf, i64 len, ptr doc)     ; 0/1/13/2
;   void universe_docparse_pdf_close(ptr doc)
;   i64 universe_docparse_pdf_object_count(ptr doc)
;   i32 universe_docparse_pdf_object_offset(ptr doc, i64 num, ptr out) ; 0/5/7
;   i64 universe_docparse_pdf_stream(ptr doc, i64 objnum, ptr dst, i64 cap)
;   i64 universe_docparse_pdf_extract_text(ptr src, i64 srclen, ptr dst, i64 cap)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare i64 @universe_compress_inflate_zlib(ptr, i64, ptr, i64)

@p.startxref = private constant [9 x i8] c"startxref"
@p.xref      = private constant [4 x i8] c"xref"
@p.trailer   = private constant [7 x i8] c"trailer"
@p.size      = private constant [5 x i8] c"/Size"
@p.stream    = private constant [6 x i8] c"stream"
@p.endstream = private constant [9 x i8] c"endstream"
@p.flate     = private constant [11 x i8] c"FlateDecode"

; ===================================================================== helpers

define internal i1 @pdf_memeq(ptr %a, ptr %b, i64 %n) #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %yes, label %body
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

; forward substring search in [from,to); -1 if none.
define internal i64 @pdf_find(ptr %buf, i64 %from, i64 %to, ptr %pat, i64 %patlen) #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ %from, %entry ], [ %i.n, %cont ]
  %end = add i64 %i, %patlen
  %fits = icmp ule i64 %end, %to
  br i1 %fits, label %chk, label %none
chk:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %m = call i1 @pdf_memeq(ptr %p, ptr %pat, i64 %patlen)
  br i1 %m, label %found, label %cont
cont:
  %i.n = add i64 %i, 1
  br label %loop
found:
  ret i64 %i
none:
  ret i64 -1
}

; backward substring search over [0,len); -1 if none.
define internal i64 @pdf_rfind(ptr %buf, i64 %len, ptr %pat, i64 %patlen) #0 {
entry:
  %tooshort = icmp ult i64 %len, %patlen
  br i1 %tooshort, label %none, label %init
init:
  %start = sub i64 %len, %patlen
  br label %loop
loop:
  %i = phi i64 [ %start, %init ], [ %i.n, %dec ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %m = call i1 @pdf_memeq(ptr %p, ptr %pat, i64 %patlen)
  br i1 %m, label %found, label %cont
cont:
  %atzero = icmp eq i64 %i, 0
  br i1 %atzero, label %none, label %dec
dec:
  %i.n = sub i64 %i, 1
  br label %loop
found:
  ret i64 %i
none:
  ret i64 -1
}

; skip PDF whitespace and %comments starting at pos.
define internal i64 @pdf_skipws(ptr %buf, i64 %len, i64 %pos) #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ %pos, %entry ], [ %i.n, %cont ], [ %j.n, %cmt.cont ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %done, label %body
body:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %ispct = icmp eq i8 %c, 37
  br i1 %ispct, label %cmt, label %chkws
chkws:
  %cz = zext i8 %c to i32
  %w0 = icmp eq i32 %cz, 0
  %w9 = icmp eq i32 %cz, 9
  %w10 = icmp eq i32 %cz, 10
  %w12 = icmp eq i32 %cz, 12
  %w13 = icmp eq i32 %cz, 13
  %w32 = icmp eq i32 %cz, 32
  %wa = or i1 %w0, %w9
  %wb = or i1 %w10, %w12
  %wc = or i1 %w13, %w32
  %wd = or i1 %wa, %wb
  %isws = or i1 %wd, %wc
  br i1 %isws, label %cont, label %done
cont:
  %i.n = add i64 %i, 1
  br label %loop
cmt:
  br label %cmt.loop
cmt.loop:
  %j = phi i64 [ %i, %cmt ], [ %j.n2, %cmt.step ]
  %jend = icmp uge i64 %j, %len
  br i1 %jend, label %done, label %cmt.body
cmt.body:
  %jp = getelementptr inbounds i8, ptr %buf, i64 %j
  %jc = load i8, ptr %jp, align 1
  %jnl = icmp eq i8 %jc, 10
  %jcr = icmp eq i8 %jc, 13
  %jeol = or i1 %jnl, %jcr
  br i1 %jeol, label %cmt.cont, label %cmt.step
cmt.step:
  %j.n2 = add i64 %j, 1
  br label %cmt.loop
cmt.cont:
  %j.n = add i64 %j, 1
  br label %loop
done:
  ret i64 %i
}

; parse unsigned integer at pos; store value; return end position.
define internal i64 @pdf_uint(ptr %buf, i64 %len, i64 %pos, ptr %pval) #6 {
entry:
  br label %loop
loop:
  %i = phi i64 [ %pos, %entry ], [ %i.n, %acc ]
  %val = phi i64 [ 0, %entry ], [ %val.n, %acc ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %done, label %body
body:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i64
  %d = sub i64 %cz, 48
  %isd = icmp ult i64 %d, 10
  br i1 %isd, label %acc, label %done
acc:
  %m = mul i64 %val, 10
  %val.n = add i64 %m, %d
  %i.n = add i64 %i, 1
  br label %loop
done:
  store i64 %val, ptr %pval, align 8
  ret i64 %i
}

; ======================================================================= open
define i32 @universe_docparse_pdf_open(ptr %buf, i64 %len, ptr %doc) #1 {
entry:
  %bn = icmp eq ptr %buf, null
  %dn = icmp eq ptr %doc, null
  %anull = or i1 %bn, %dn
  br i1 %anull, label %ret_null, label %init
ret_null:
  ret i32 1
ret_parse:
  ret i32 13
init:
  store ptr %buf, ptr %doc, align 8
  %lp = getelementptr inbounds i8, ptr %doc, i64 8
  store i64 %len, ptr %lp, align 8
  %op = getelementptr inbounds i8, ptr %doc, i64 24
  store ptr null, ptr %op, align 8
  %cp = getelementptr inbounds i8, ptr %doc, i64 32
  store i64 0, ptr %cp, align 8
  ; startxref
  %sx = call i64 @pdf_rfind(ptr %buf, i64 %len, ptr @p.startxref, i64 9)
  %sxbad = icmp slt i64 %sx, 0
  br i1 %sxbad, label %ret_parse, label %sxread
sxread:
  %pval = alloca i64, align 8
  %after = add i64 %sx, 9
  %ws1 = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %after)
  %xrp = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %ws1, ptr %pval)
  %xref_off = load i64, ptr %pval, align 8
  %xp = getelementptr inbounds i8, ptr %doc, i64 16
  store i64 %xref_off, ptr %xp, align 8
  ; /Size from trailer
  %szpos = call i64 @pdf_find(ptr %buf, i64 %xref_off, i64 %len, ptr @p.size, i64 5)
  %szbad = icmp slt i64 %szpos, 0
  br i1 %szbad, label %ret_parse, label %szread
szread:
  %szafter = add i64 %szpos, 5
  %ws2 = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %szafter)
  %szend = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %ws2, ptr %pval)
  %size = load i64, ptr %pval, align 8
  %sizeok = icmp ugt i64 %size, 0
  br i1 %sizeok, label %doalloc, label %ret_parse
doalloc:
  %ccp = getelementptr inbounds i8, ptr %doc, i64 32
  store i64 %size, ptr %ccp, align 8
  %bytes = mul i64 %size, 8
  %arr = call ptr @malloc(i64 %bytes)
  %arrnull = icmp eq ptr %arr, null
  br i1 %arrnull, label %ret_oom, label %initarr
ret_oom:
  ret i32 2
initarr:
  %aop = getelementptr inbounds i8, ptr %doc, i64 24
  store ptr %arr, ptr %aop, align 8
  br label %izloop
izloop:
  %zi = phi i64 [ 0, %initarr ], [ %zi.n, %izstore ]
  %zdone = icmp uge i64 %zi, %size
  %zslot = getelementptr inbounds i64, ptr %arr, i64 %zi
  %zi.n = add i64 %zi, 1
  br i1 %zdone, label %xref.begin, label %izstore
izstore:
  store i64 -1, ptr %zslot, align 8
  br label %izloop
xref.begin:
  ; expect "xref" at xref_off (after ws)
  %xw = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %xref_off)
  %xhead = getelementptr inbounds i8, ptr %buf, i64 %xw
  %isxref = call i1 @pdf_memeq(ptr %xhead, ptr @p.xref, i64 4)
  br i1 %isxref, label %sub.pre, label %ret_ok
ret_ok:
  ; no classic xref table; offsets stay -1 (object_offset returns 5).
  ret i32 0
sub.pre:
  %cur0 = add i64 %xw, 4
  br label %sub.loop
sub.loop:
  %cur = phi i64 [ %cur0, %sub.pre ], [ %cur.e, %ent.after ]
  %subws = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %cur)
  ; trailer?
  %trend = add i64 %subws, 7
  %trfits = icmp ule i64 %trend, %len
  br i1 %trfits, label %sub.chktrail, label %fin_ok
sub.chktrail:
  %trp = getelementptr inbounds i8, ptr %buf, i64 %subws
  %istrail = call i1 @pdf_memeq(ptr %trp, ptr @p.trailer, i64 7)
  br i1 %istrail, label %fin_ok, label %sub.head
fin_ok:
  ret i32 0
sub.head:
  %pstart = alloca i64, align 8
  %pcount = alloca i64, align 8
  %hs = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %subws, ptr %pstart)
  %hws = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %hs)
  %hc = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %hws, ptr %pcount)
  %substart = load i64, ptr %pstart, align 8
  %subcount = load i64, ptr %pcount, align 8
  br label %ent.loop
ent.loop:
  %ecur = phi i64 [ %hc, %sub.head ], [ %ecur.n, %ent.store ]
  %k = phi i64 [ 0, %sub.head ], [ %k.n, %ent.store ]
  %kdone = icmp uge i64 %k, %subcount
  br i1 %kdone, label %ent.after, label %ent.read
ent.read:
  %poff = alloca i64, align 8
  %pgen = alloca i64, align 8
  %ew1 = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %ecur)
  %eo = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %ew1, ptr %poff)
  %ew2 = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %eo)
  %eg = call i64 @pdf_uint(ptr %buf, i64 %len, i64 %ew2, ptr %pgen)
  %ew3 = call i64 @pdf_skipws(ptr %buf, i64 %len, i64 %eg)
  ; type char
  %tp = getelementptr inbounds i8, ptr %buf, i64 %ew3
  %tc = load i8, ptr %tp, align 1
  %ecur.n = add i64 %ew3, 1
  %isn = icmp eq i8 %tc, 110
  %objnum = add i64 %substart, %k
  %inrange = icmp ult i64 %objnum, %size
  %doset = and i1 %isn, %inrange
  br i1 %doset, label %ent.store_n, label %ent.store
ent.store_n:
  %offval = load i64, ptr %poff, align 8
  %slot = getelementptr inbounds i64, ptr %arr, i64 %objnum
  store i64 %offval, ptr %slot, align 8
  br label %ent.store
ent.store:
  %k.n = add i64 %k, 1
  br label %ent.loop
ent.after:
  %cur.e = phi i64 [ %ecur, %ent.loop ]
  br label %sub.loop
}

; ====================================================================== close
define void @universe_docparse_pdf_close(ptr %doc) #1 {
entry:
  %n = icmp eq ptr %doc, null
  br i1 %n, label %ret, label %free
free:
  %op = getelementptr inbounds i8, ptr %doc, i64 24
  %o = load ptr, ptr %op, align 8
  call void @free(ptr %o)
  store ptr null, ptr %op, align 8
  ret void
ret:
  ret void
}

; =============================================================== object_count
define i64 @universe_docparse_pdf_object_count(ptr %doc) #1 {
entry:
  %n = icmp eq ptr %doc, null
  br i1 %n, label %z, label %ld
z:
  ret i64 0
ld:
  %cp = getelementptr inbounds i8, ptr %doc, i64 32
  %c = load i64, ptr %cp, align 8
  ret i64 %c
}

; ============================================================== object_offset
define i32 @universe_docparse_pdf_object_offset(ptr %doc, i64 %num, ptr %out) #1 {
entry:
  %dn = icmp eq ptr %doc, null
  %on = icmp eq ptr %out, null
  %an = or i1 %dn, %on
  br i1 %an, label %ret_arg, label %ld
ret_arg:
  ret i32 -8
ld:
  %cp = getelementptr inbounds i8, ptr %doc, i64 32
  %cnt = load i64, ptr %cp, align 8
  %oob = icmp uge i64 %num, %cnt
  br i1 %oob, label %ret_idx, label %get
ret_idx:
  ret i32 7
get:
  %ap = getelementptr inbounds i8, ptr %doc, i64 24
  %arr = load ptr, ptr %ap, align 8
  %slot = getelementptr inbounds i64, ptr %arr, i64 %num
  %off = load i64, ptr %slot, align 8
  %free = icmp slt i64 %off, 0
  br i1 %free, label %ret_nf, label %ok
ret_nf:
  ret i32 5
ok:
  store i64 %off, ptr %out, align 8
  ret i32 0
}

; ==================================================================== stream
define i64 @universe_docparse_pdf_stream(ptr %doc, i64 %objnum, ptr %dst, i64 %cap) #1 {
entry:
  %dn = icmp eq ptr %doc, null
  %sn = icmp eq ptr %dst, null
  %an = or i1 %dn, %sn
  br i1 %an, label %ret_null, label %ld
ret_null:
  ret i64 -1
ret_parse:
  ret i64 -13
ld:
  %buf = load ptr, ptr %doc, align 8
  %lp = getelementptr inbounds i8, ptr %doc, i64 8
  %len = load i64, ptr %lp, align 8
  %cp = getelementptr inbounds i8, ptr %doc, i64 32
  %cnt = load i64, ptr %cp, align 8
  %oob = icmp uge i64 %objnum, %cnt
  br i1 %oob, label %ret_parse, label %get
get:
  %ap = getelementptr inbounds i8, ptr %doc, i64 24
  %arr = load ptr, ptr %ap, align 8
  %slot = getelementptr inbounds i64, ptr %arr, i64 %objnum
  %off = load i64, ptr %slot, align 8
  %free = icmp slt i64 %off, 0
  br i1 %free, label %ret_parse, label %find_stream
find_stream:
  %spos = call i64 @pdf_find(ptr %buf, i64 %off, i64 %len, ptr @p.stream, i64 6)
  %sbad = icmp slt i64 %spos, 0
  br i1 %sbad, label %ret_parse, label %chk_filter
chk_filter:
  %fdpos = call i64 @pdf_find(ptr %buf, i64 %off, i64 %spos, ptr @p.flate, i64 11)
  %isflate = icmp sge i64 %fdpos, 0
  ; data start: after "stream" + one EOL (CRLF or LF)
  %ds0 = add i64 %spos, 6
  %ds0p = getelementptr inbounds i8, ptr %buf, i64 %ds0
  %ds0c = load i8, ptr %ds0p, align 1
  %iscr = icmp eq i8 %ds0c, 13
  %islf = icmp eq i8 %ds0c, 10
  %ds1 = add i64 %ds0, 1
  %ds1p = getelementptr inbounds i8, ptr %buf, i64 %ds1
  %ds1c = load i8, ptr %ds1p, align 1
  %crlf = icmp eq i8 %ds1c, 10
  %crlfboth = and i1 %iscr, %crlf
  %adv = select i1 %crlfboth, i64 2, i64 1
  %advany = or i1 %iscr, %islf
  %advn = select i1 %advany, i64 %adv, i64 0
  %ds = add i64 %ds0, %advn
  ; endstream
  %epos = call i64 @pdf_find(ptr %buf, i64 %ds, i64 %len, ptr @p.endstream, i64 9)
  %ebad = icmp slt i64 %epos, 0
  br i1 %ebad, label %ret_parse, label %trim
trim:
  ; strip up to one trailing EOL before endstream
  %e1 = sub i64 %epos, 1
  %e1p = getelementptr inbounds i8, ptr %buf, i64 %e1
  %e1c = load i8, ptr %e1p, align 1
  %e1lf = icmp eq i8 %e1c, 10
  %de1 = select i1 %e1lf, i64 %e1, i64 %epos
  %de1m = sub i64 %de1, 1
  %de1mp = getelementptr inbounds i8, ptr %buf, i64 %de1m
  %de1mc = load i8, ptr %de1mp, align 1
  %de1cr = icmp eq i8 %de1mc, 13
  %strip2 = and i1 %e1lf, %de1cr
  %de = select i1 %strip2, i64 %de1m, i64 %de1
  %comp = sub i64 %de, %ds
  %datap = getelementptr inbounds i8, ptr %buf, i64 %ds
  br i1 %isflate, label %inflate, label %rawcopy
inflate:
  %n = call i64 @universe_compress_inflate_zlib(ptr %dst, i64 %cap, ptr %datap, i64 %comp)
  ret i64 %n
rawcopy:
  %fits = icmp ule i64 %comp, %cap
  br i1 %fits, label %docopy, label %ret_full
ret_full:
  ret i64 -6
docopy:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %datap, i64 %comp, i1 false)
  ret i64 %comp
}

; =============================================================== extract_text
define i64 @universe_docparse_pdf_extract_text(ptr %src, i64 %srclen, ptr %dst, i64 %cap) #6 {
entry:
  %sn = icmp eq ptr %src, null
  %dn = icmp eq ptr %dst, null
  %an = or i1 %sn, %dn
  br i1 %an, label %ret_null, label %setup
ret_null:
  ret i64 -1
setup:
  %pdi = alloca i64, align 8
  %pdepth = alloca i64, align 8
  store i64 0, ptr %pdi, align 8
  br label %scan
scan:
  %si = phi i64 [ 0, %setup ], [ %si.n, %scan.cont ], [ %resume, %str.done ]
  %atend = icmp uge i64 %si, %srclen
  br i1 %atend, label %done, label %scan.body
scan.body:
  %sp = getelementptr inbounds i8, ptr %src, i64 %si
  %sc = load i8, ptr %sp, align 1
  %isopen = icmp eq i8 %sc, 40
  br i1 %isopen, label %str.begin, label %scan.cont
scan.cont:
  %si.n = add i64 %si, 1
  br label %scan
done:
  %di = load i64, ptr %pdi, align 8
  ret i64 %di
; ---- literal string ----
str.begin:
  store i64 1, ptr %pdepth, align 8
  %after = add i64 %si, 1
  br label %str.loop
str.loop:
  %ci = phi i64 [ %after, %str.begin ], [ %ci.n, %str.plain ], [ %ci.escA, %str.esc.adv ], [ %ci.escB, %str.oct.join ], [ %ci.n2, %str.open ], [ %ci.n3, %str.closeemit ]
  %cend = icmp uge i64 %ci, %srclen
  br i1 %cend, label %str.done, label %str.body
str.body:
  %cp = getelementptr inbounds i8, ptr %src, i64 %ci
  %cc = load i8, ptr %cp, align 1
  %isbs = icmp eq i8 %cc, 92
  br i1 %isbs, label %str.esc, label %str.notesc
str.notesc:
  %isopenp = icmp eq i8 %cc, 40
  br i1 %isopenp, label %str.open, label %str.notopen
str.notopen:
  %isclosep = icmp eq i8 %cc, 41
  br i1 %isclosep, label %str.close, label %str.plain
str.plain:
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %cc)
  %ci.n = add i64 %ci, 1
  br label %str.loop
str.open:
  %dO = load i64, ptr %pdepth, align 8
  %dO1 = add i64 %dO, 1
  store i64 %dO1, ptr %pdepth, align 8
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 40)
  %ci.n2 = add i64 %ci, 1
  br label %str.loop
str.close:
  %dC = load i64, ptr %pdepth, align 8
  %dC1 = sub i64 %dC, 1
  store i64 %dC1, ptr %pdepth, align 8
  %endstr = icmp eq i64 %dC1, 0
  br i1 %endstr, label %str.finish, label %str.closeemit
str.closeemit:
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 41)
  %ci.n3 = add i64 %ci, 1
  br label %str.loop
str.finish:
  %sfin = add i64 %ci, 1
  br label %str.done
str.done:
  %resume = phi i64 [ %sfin, %str.finish ], [ %ci, %str.loop ], [ %ci, %str.esc ]
  br label %scan
; ---- escape sequence ----
str.esc:
  %e1 = add i64 %ci, 1
  %e1end = icmp uge i64 %e1, %srclen
  br i1 %e1end, label %str.done, label %str.escbody
str.escbody:
  %ep = getelementptr inbounds i8, ptr %src, i64 %e1
  %ec = load i8, ptr %ep, align 1
  ; octal?
  %od = sub i8 %ec, 48
  %isoct = icmp ult i8 %od, 8
  br i1 %isoct, label %str.octal, label %str.escsimple
str.escsimple:
  ; map n r t b f, else literal char
  %isn = icmp eq i8 %ec, 110
  %isr = icmp eq i8 %ec, 114
  %ist = icmp eq i8 %ec, 116
  %isb = icmp eq i8 %ec, 98
  %isf = icmp eq i8 %ec, 102
  %isnl = icmp eq i8 %ec, 10
  %iscr2 = icmp eq i8 %ec, 13
  %mapf = select i1 %isf, i8 12, i8 %ec
  %mapb = select i1 %isb, i8 8, i8 %mapf
  %mapt = select i1 %ist, i8 9, i8 %mapb
  %mapr = select i1 %isr, i8 13, i8 %mapt
  %mapn = select i1 %isn, i8 10, i8 %mapr
  ; line continuation: backslash-newline emits nothing
  %iscont = or i1 %isnl, %iscr2
  br i1 %iscont, label %str.esc.adv, label %str.esc.emit
str.esc.emit:
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %mapn)
  br label %str.esc.adv
str.esc.adv:
  %ci.escA = add i64 %ci, 2
  br label %str.loop
str.octal:
  ; parse up to 3 octal digits starting at e1
  %o0 = zext i8 %od to i64
  %o1i = add i64 %e1, 1
  %o1end = icmp uge i64 %o1i, %srclen
  br i1 %o1end, label %str.oct.emit1, label %str.oct2
str.oct2:
  %o1p = getelementptr inbounds i8, ptr %src, i64 %o1i
  %o1c = load i8, ptr %o1p, align 1
  %o1d = sub i8 %o1c, 48
  %o1oct = icmp ult i8 %o1d, 8
  br i1 %o1oct, label %str.oct2y, label %str.oct.emit1
str.oct2y:
  %o1v = zext i8 %o1d to i64
  %acc1 = mul i64 %o0, 8
  %acc1v = add i64 %acc1, %o1v
  %o2i = add i64 %e1, 2
  %o2end = icmp uge i64 %o2i, %srclen
  br i1 %o2end, label %str.oct.emit2, label %str.oct3
str.oct3:
  %o2p = getelementptr inbounds i8, ptr %src, i64 %o2i
  %o2c = load i8, ptr %o2p, align 1
  %o2d = sub i8 %o2c, 48
  %o2oct = icmp ult i8 %o2d, 8
  br i1 %o2oct, label %str.oct3y, label %str.oct.emit2
str.oct3y:
  %o2v = zext i8 %o2d to i64
  %acc2 = mul i64 %acc1v, 8
  %acc2v = add i64 %acc2, %o2v
  %ob3 = trunc i64 %acc2v to i8
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %ob3)
  %ci.oct3 = add i64 %ci, 4
  br label %str.oct.join
str.oct.emit2:
  %ob2 = trunc i64 %acc1v to i8
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %ob2)
  %ci.oct2 = add i64 %ci, 3
  br label %str.oct.join
str.oct.emit1:
  %ob1 = trunc i64 %o0 to i8
  call void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %ob1)
  %ci.oct1 = add i64 %ci, 2
  br label %str.oct.join
str.oct.join:
  %ci.escB = phi i64 [ %ci.oct3, %str.oct3y ], [ %ci.oct2, %str.oct.emit2 ], [ %ci.oct1, %str.oct.emit1 ]
  br label %str.loop
}

; append one byte to dst if under cap; always bump the count.
define internal void @pdf_emit(ptr %dst, ptr %pdi, i64 %cap, i8 %b) #6 {
entry:
  %di = load i64, ptr %pdi, align 8
  %fits = icmp ult i64 %di, %cap
  br i1 %fits, label %store, label %skip
store:
  %p = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %b, ptr %p, align 1
  br label %skip
skip:
  %di1 = add i64 %di, 1
  store i64 %di1, ptr %pdi, align 8
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #6 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
