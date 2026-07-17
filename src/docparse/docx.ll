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

; DOCX (Office Open XML / WordprocessingML) visible-text extraction.
;
; DESIGN:
;   * BUILDS ON compress/zip (+ inflate) and docparse/xml, exactly like the
;     XLSX reader. A .docx is a ZIP in the caller buffer; the body lives in the
;     member "word/document.xml". We locate that member and inflate/copy it ONCE
;     into a private malloc buffer (the only allocation), then make a single
;     linear pass over the XML pull-token stream — NO DOM is built. This is the
;     canonical IO/compute split: zip+inflate is the IO/decompress phase, the
;     token walk is pure compute writing into the caller's out buffer.
;   * The token walk is a tiny state machine. WordprocessingML emits visible
;     text only inside <w:t>...</w:t> runs; everything else (run/paragraph
;     properties, styles, bookmarks, section props) is skipped. Emission rules:
;       <w:t>text</w:t>  -> the XML-entity-decoded run text
;       <w:tab/>         -> '\t'
;       <w:br/> <w:cr/>  -> '\n'
;       </w:p>           -> '\n'   (paragraph boundary)
;     The tokenizer reports the tag NAME including its prefix ("w:t", "w:tab",
;     ...); exact length-checked compares keep "w:t" from matching "w:tab"/"w:tr".
;   * UNTRUSTED INPUT. Every read of the extracted XML is bounded by the token
;     slice / buffer length (the zip + xml layers already bounds-check their own
;     reads), and every write to `out` is capacity-checked: the emit helpers
;     return -1 the instant a write would exceed out_cap, which the API turns
;     into FULL(6). Entity decode is done in-module (the xml module's decoder is
;     `internal` and unbounded) so the FULL boundary is exact on decoded bytes.
;     A decoded byte count is always <= its source byte count (every entity —
;     named or numeric — is longer than its expansion), so the pass never grows.
;
; API:
;   i32 universe_docparse_docx_text(ptr zip_buf, i64 zip_len,
;                                   ptr out, i64 out_cap, ptr out_len)
;     0 OK / 1 NULL_PTR / 2 OOM / 5 NOT_FOUND (no word/document.xml)
;     / 6 FULL (out_cap too small) / 8 INVALID_ARG (not a zip / malformed)
;   On OK, *out_len = number of bytes written to out.

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"

declare i32 @universe_compress_zip_open(ptr, i64, ptr)
declare i64 @universe_compress_zip_count(ptr)
declare i32 @universe_compress_zip_entry(ptr, i64, ptr)
declare i64 @universe_compress_zip_extract(ptr, ptr, ptr, i64)

declare void @universe_docparse_xml_init(ptr, ptr, i64)
declare i32 @universe_docparse_xml_next(ptr, ptr)

@d.doc  = private constant [17 x i8] c"word/document.xml"
@d.wt   = private constant [3 x i8] c"w:t"
@d.wp   = private constant [3 x i8] c"w:p"
@d.wtab = private constant [5 x i8] c"w:tab"
@d.wbr  = private constant [4 x i8] c"w:br"
@d.wcr  = private constant [4 x i8] c"w:cr"

; ===================================================================== helpers

; compare buffer slice (buf+off,len) to (cstr,clen). No libc.
define internal i1 @docx_slice_eq(ptr %buf, i64 %off, i64 %len, ptr %cstr, i64 %clen) #0 {
entry:
  %lok = icmp eq i64 %len, %clen
  br i1 %lok, label %loop, label %no
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %done = icmp uge i64 %i, %len
  br i1 %done, label %yes, label %body
body:
  %ao = add i64 %off, %i
  %ap = getelementptr inbounds i8, ptr %buf, i64 %ao
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

; buf[i] as i32 0..255, or -1 when i >= end (bounded read of the extracted XML).
define internal i32 @docx_at(ptr %buf, i64 %end, i64 %i) #0 {
entry:
  %oob = icmp uge i64 %i, %end
  br i1 %oob, label %no, label %yes
no:
  ret i32 -1
yes:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %z = zext i8 %c to i32
  ret i32 %z
}

; append one byte to out[di], bounded by cap. Returns new di, or -1 if full.
; A di of -1 fed back in stays -1 (unsigned uge cap), so callers can chain.
define internal i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %b) #2 {
entry:
  %full = icmp uge i64 %di, %cap
  br i1 %full, label %isfull, label %ok
isfull:
  ret i64 -1
ok:
  %p = getelementptr inbounds i8, ptr %out, i64 %di
  store i8 %b, ptr %p, align 1
  %n = add i64 %di, 1
  ret i64 %n
}

; UTF-8 encode code point %cp into out[di], bounded. Returns new di or -1.
define internal i64 @docx_put_cp(ptr %out, i64 %cap, i64 %di, i32 %cp) #2 {
entry:
  %lt80 = icmp ult i32 %cp, 128
  br i1 %lt80, label %one, label %chk800
one:
  %c0 = trunc i32 %cp to i8
  %r1 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %c0)
  ret i64 %r1
chk800:
  %lt800 = icmp ult i32 %cp, 2048
  br i1 %lt800, label %two, label %chk10000
two:
  %t_hi = lshr i32 %cp, 6
  %t_b0i = or i32 %t_hi, 192
  %t_b0 = trunc i32 %t_b0i to i8
  %t_lo = and i32 %cp, 63
  %t_b1i = or i32 %t_lo, 128
  %t_b1 = trunc i32 %t_b1i to i8
  %a1 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %t_b0)
  %a2 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %a1, i8 %t_b1)
  ret i64 %a2
chk10000:
  %lt10000 = icmp ult i32 %cp, 65536
  br i1 %lt10000, label %three, label %four
three:
  %h_hi = lshr i32 %cp, 12
  %h_b0i = or i32 %h_hi, 224
  %h_b0 = trunc i32 %h_b0i to i8
  %h_m0 = lshr i32 %cp, 6
  %h_m = and i32 %h_m0, 63
  %h_b1i = or i32 %h_m, 128
  %h_b1 = trunc i32 %h_b1i to i8
  %h_lo = and i32 %cp, 63
  %h_b2i = or i32 %h_lo, 128
  %h_b2 = trunc i32 %h_b2i to i8
  %b1 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %h_b0)
  %b2 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %b1, i8 %h_b1)
  %b3 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %b2, i8 %h_b2)
  ret i64 %b3
four:
  %f_hi = lshr i32 %cp, 18
  %f_b0i = or i32 %f_hi, 240
  %f_b0 = trunc i32 %f_b0i to i8
  %f_m0 = lshr i32 %cp, 12
  %f_m0m = and i32 %f_m0, 63
  %f_b1i = or i32 %f_m0m, 128
  %f_b1 = trunc i32 %f_b1i to i8
  %f_m1 = lshr i32 %cp, 6
  %f_m1m = and i32 %f_m1, 63
  %f_b2i = or i32 %f_m1m, 128
  %f_b2 = trunc i32 %f_b2i to i8
  %f_lo = and i32 %cp, 63
  %f_b3i = or i32 %f_lo, 128
  %f_b3 = trunc i32 %f_b3i to i8
  %c1 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %f_b0)
  %c2 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %c1, i8 %f_b1)
  %c3 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %c2, i8 %f_b2)
  %c4 = call i64 @docx_putc(ptr %out, i64 %cap, i64 %c3, i8 %f_b3)
  ret i64 %c4
}

; Emit XML text span buf[start..end) into out[di0..], decoding entities.
; Returns new di, or -1 if out_cap would be exceeded (FULL).
define internal i64 @docx_emit_text(ptr %out, i64 %cap, i64 %di0, ptr %buf, i64 %start, i64 %end) #2 {
entry:
  br label %loop
loop:
  %i = phi i64 [ %start, %entry ], [ %icand, %adv ]
  %di = phi i64 [ %di0, %entry ], [ %dcand, %adv ]
  %atend = icmp uge i64 %i, %end
  br i1 %atend, label %done, label %body
done:
  ret i64 %di
body:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %isamp = icmp eq i8 %c, 38
  br i1 %isamp, label %ent, label %plain
plain:
  %d.plain = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %c)
  %i.plain = add i64 %i, 1
  br label %merge
ent:
  %n1 = add i64 %i, 1
  %b1 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n1)
  %ishash = icmp eq i32 %b1, 35
  br i1 %ishash, label %numeric, label %named
named:
  %n2 = add i64 %i, 2
  %n3 = add i64 %i, 3
  %n4 = add i64 %i, 4
  %n5 = add i64 %i, 5
  %e1 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n1)
  %e2 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n2)
  %e3 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n3)
  %e4 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n4)
  %e5 = call i32 @docx_at(ptr %buf, i64 %end, i64 %n5)
  ; &amp; -> '&'
  %amp_a = icmp eq i32 %e1, 97
  %amp_b = icmp eq i32 %e2, 109
  %amp_c = icmp eq i32 %e3, 112
  %amp_d = icmp eq i32 %e4, 59
  %amp1 = and i1 %amp_a, %amp_b
  %amp2 = and i1 %amp_c, %amp_d
  %amp = and i1 %amp1, %amp2
  ; &lt; -> '<'
  %lt_a = icmp eq i32 %e1, 108
  %lt_b = icmp eq i32 %e2, 116
  %lt_c = icmp eq i32 %e3, 59
  %lt1 = and i1 %lt_a, %lt_b
  %lt = and i1 %lt1, %lt_c
  ; &gt; -> '>'
  %gt_a = icmp eq i32 %e1, 103
  %gt_b = icmp eq i32 %e2, 116
  %gt_c = icmp eq i32 %e3, 59
  %gt1 = and i1 %gt_a, %gt_b
  %gt = and i1 %gt1, %gt_c
  ; &quot; -> '"'
  %qu_a = icmp eq i32 %e1, 113
  %qu_b = icmp eq i32 %e2, 117
  %qu_c = icmp eq i32 %e3, 111
  %qu_d = icmp eq i32 %e4, 116
  %qu_e = icmp eq i32 %e5, 59
  %qu1 = and i1 %qu_a, %qu_b
  %qu2 = and i1 %qu_c, %qu_d
  %qu3 = and i1 %qu1, %qu2
  %quot = and i1 %qu3, %qu_e
  ; &apos; -> '\''
  %ap_a = icmp eq i32 %e1, 97
  %ap_b = icmp eq i32 %e2, 112
  %ap_c = icmp eq i32 %e3, 111
  %ap_d = icmp eq i32 %e4, 115
  %ap_e = icmp eq i32 %e5, 59
  %ap1 = and i1 %ap_a, %ap_b
  %ap2 = and i1 %ap_c, %ap_d
  %ap3 = and i1 %ap1, %ap2
  %apos = and i1 %ap3, %ap_e
  ; default: literal '&', consume 1
  %ch_ap = select i1 %apos, i32 39, i32 38
  %co_ap = select i1 %apos, i64 6, i64 1
  %ch_qu = select i1 %quot, i32 34, i32 %ch_ap
  %co_qu = select i1 %quot, i64 6, i64 %co_ap
  %ch_gt = select i1 %gt, i32 62, i32 %ch_qu
  %co_gt = select i1 %gt, i64 4, i64 %co_qu
  %ch_lt = select i1 %lt, i32 60, i32 %ch_gt
  %co_lt = select i1 %lt, i64 4, i64 %co_gt
  %ch = select i1 %amp, i32 38, i32 %ch_lt
  %co = select i1 %amp, i64 5, i64 %co_lt
  %chb = trunc i32 %ch to i8
  %d.named = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 %chb)
  %i.named = add i64 %i, %co
  br label %merge
numeric:
  %m2 = add i64 %i, 2
  %m3 = add i64 %i, 3
  %bx = call i32 @docx_at(ptr %buf, i64 %end, i64 %m2)
  %isx = icmp eq i32 %bx, 120
  %isX = icmp eq i32 %bx, 88
  %ishex = or i1 %isx, %isX
  %dstart = select i1 %ishex, i64 %m3, i64 %m2
  br label %num.loop
num.loop:
  %ni = phi i64 [ %dstart, %numeric ], [ %ni.n, %num.cont ]
  %acc = phi i32 [ 0, %numeric ], [ %acc.n, %num.cont ]
  %nb = call i32 @docx_at(ptr %buf, i64 %end, i64 %ni)
  %issemi = icmp eq i32 %nb, 59
  br i1 %issemi, label %num.done, label %num.dig
num.dig:
  %isoob = icmp eq i32 %nb, -1
  br i1 %isoob, label %num.bad, label %num.dig2
num.dig2:
  %d09 = sub i32 %nb, 48
  %is09 = icmp ult i32 %d09, 10
  %lc = or i32 %nb, 32
  %dhex = sub i32 %lc, 87
  %hl0 = icmp uge i32 %lc, 97
  %hl1 = icmp ule i32 %lc, 102
  %ishexletter = and i1 %hl0, %hl1
  br i1 %ishex, label %num.hex, label %num.dec
num.dec:
  br i1 %is09, label %num.accd, label %num.bad
num.accd:
  %accd = mul i32 %acc, 10
  %accdn = add i32 %accd, %d09
  br label %num.cont
num.hex:
  %validhex = or i1 %is09, %ishexletter
  br i1 %validhex, label %num.acch, label %num.bad
num.acch:
  %digv = select i1 %is09, i32 %d09, i32 %dhex
  %acch = shl i32 %acc, 4
  %acchn = add i32 %acch, %digv
  br label %num.cont
num.cont:
  %acc.n = phi i32 [ %accdn, %num.accd ], [ %acchn, %num.acch ]
  %ni.n = add i64 %ni, 1
  br label %num.loop
num.done:
  %d.num = call i64 @docx_put_cp(ptr %out, i64 %cap, i64 %di, i32 %acc)
  %i.num = add i64 %ni, 1
  br label %merge
num.bad:
  ; malformed numeric entity: emit literal '&', consume 1 (lenient)
  %d.bad = call i64 @docx_putc(ptr %out, i64 %cap, i64 %di, i8 38)
  %i.bad = add i64 %i, 1
  br label %merge
merge:
  %dcand = phi i64 [ %d.plain, %plain ], [ %d.named, %named ], [ %d.num, %num.done ], [ %d.bad, %num.bad ]
  %icand = phi i64 [ %i.plain, %plain ], [ %i.named, %named ], [ %i.num, %num.done ], [ %i.bad, %num.bad ]
  %full = icmp slt i64 %dcand, 0
  br i1 %full, label %isfull, label %adv
isfull:
  ret i64 -1
adv:
  br label %loop
}

; locate "word/document.xml", inflate/copy into a fresh malloc buffer.
; Stores {ptr,len} at *pbuf/*plen. Returns 0 OK / 5 NOT_FOUND / 2 OOM / 8 malformed.
define internal i32 @docx_find_document(ptr %reader, ptr %pbuf, ptr %plen) #1 {
entry:
  %e = alloca [48 x i8], align 8
  %cnt = call i64 @universe_compress_zip_count(ptr %reader)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %next ]
  %done = icmp uge i64 %i, %cnt
  br i1 %done, label %notfound, label %get
get:
  %rc = call i32 @universe_compress_zip_entry(ptr %reader, i64 %i, ptr %e)
  %rcok = icmp eq i32 %rc, 0
  br i1 %rcok, label %chk, label %next
chk:
  %ename = load ptr, ptr %e, align 8
  %enlp = getelementptr inbounds i8, ptr %e, i64 8
  %enl = load i64, ptr %enlp, align 8
  %match = call i1 @docx_slice_eq(ptr %ename, i64 0, i64 %enl, ptr @d.doc, i64 17)
  br i1 %match, label %found, label %next
next:
  %i.n = add i64 %i, 1
  br label %loop
notfound:
  ret i32 5
found:
  %usp = getelementptr inbounds i8, ptr %e, i64 32
  %us = load i64, ptr %usp, align 8
  %isz = icmp eq i64 %us, 0
  %sz = select i1 %isz, i64 1, i64 %us
  %p = call ptr @malloc(i64 %sz)
  %pnull = icmp eq ptr %p, null
  br i1 %pnull, label %oom, label %ext
oom:
  ret i32 2
ext:
  %n = call i64 @universe_compress_zip_extract(ptr %reader, ptr %e, ptr %p, i64 %us)
  %nneg = icmp slt i64 %n, 0
  br i1 %nneg, label %exterr, label %ok
exterr:
  call void @free(ptr %p)
  ret i32 8
ok:
  store ptr %p, ptr %pbuf, align 8
  store i64 %n, ptr %plen, align 8
  ret i32 0
}

; ======================================================================= text
define i32 @universe_docparse_docx_text(ptr %zip_buf, i64 %zip_len, ptr %out, i64 %out_cap, ptr %out_len) #1 {
entry:
  %zn = icmp eq ptr %zip_buf, null
  %on = icmp eq ptr %out, null
  %ln = icmp eq ptr %out_len, null
  %n0 = or i1 %zn, %on
  %anull = or i1 %n0, %ln
  br i1 %anull, label %ret_null, label %setup
ret_null:
  ret i32 1
setup:
  %reader = alloca [32 x i8], align 8
  %pbuf = alloca ptr, align 8
  %plen = alloca i64, align 8
  %sc = alloca [24 x i8], align 8
  %tok = alloca [40 x i8], align 8
  %pdi = alloca i64, align 8
  %pinwt = alloca i32, align 4
  %zr = call i32 @universe_compress_zip_open(ptr %zip_buf, i64 %zip_len, ptr %reader)
  %zrok = icmp eq i32 %zr, 0
  br i1 %zrok, label %find, label %ret_malformed
ret_malformed:
  ret i32 8
find:
  %fr = call i32 @docx_find_document(ptr %reader, ptr %pbuf, ptr %plen)
  %frok = icmp eq i32 %fr, 0
  br i1 %frok, label %scan, label %ret_find
ret_find:
  ret i32 %fr
scan:
  %buf = load ptr, ptr %pbuf, align 8
  %len = load i64, ptr %plen, align 8
  call void @universe_docparse_xml_init(ptr %sc, ptr %buf, i64 %len)
  store i64 0, ptr %pdi, align 8
  store i32 0, ptr %pinwt, align 4
  br label %loop
loop:
  %rc = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %ty = load i32, ptr %tok, align 4
  %eof = icmp eq i32 %ty, 4
  br i1 %eof, label %finish, label %tokbody
tokbody:
  %nop = getelementptr inbounds i8, ptr %tok, i64 8
  %noff = load i64, ptr %nop, align 8
  %nlp = getelementptr inbounds i8, ptr %tok, i64 16
  %nlen = load i64, ptr %nlp, align 8
  %isstart = icmp eq i32 %ty, 0
  br i1 %isstart, label %onstart, label %chk.text
onstart:
  %s.iswt = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wt, i64 3)
  br i1 %s.iswt, label %set.inwt, label %loop
set.inwt:
  store i32 1, ptr %pinwt, align 4
  br label %loop
chk.text:
  %istext = icmp eq i32 %ty, 3
  br i1 %istext, label %ontext, label %chk.end
ontext:
  %inwt = load i32, ptr %pinwt, align 4
  %inwtb = icmp ne i32 %inwt, 0
  br i1 %inwtb, label %do.text, label %loop
do.text:
  %di0 = load i64, ptr %pdi, align 8
  %tend = add i64 %noff, %nlen
  %newdi = call i64 @docx_emit_text(ptr %out, i64 %out_cap, i64 %di0, ptr %buf, i64 %noff, i64 %tend)
  %tfull = icmp slt i64 %newdi, 0
  br i1 %tfull, label %ret_full, label %store.text
store.text:
  store i64 %newdi, ptr %pdi, align 8
  br label %loop
chk.end:
  %isend = icmp eq i32 %ty, 1
  br i1 %isend, label %onend, label %chk.self
onend:
  %e.iswt = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wt, i64 3)
  br i1 %e.iswt, label %clr.inwt, label %chk.endp
clr.inwt:
  store i32 0, ptr %pinwt, align 4
  br label %loop
chk.endp:
  %e.iswp = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wp, i64 3)
  br i1 %e.iswp, label %emit.nl.p, label %loop
emit.nl.p:
  %pdi.p = load i64, ptr %pdi, align 8
  %nl.p = call i64 @docx_putc(ptr %out, i64 %out_cap, i64 %pdi.p, i8 10)
  %pfull = icmp slt i64 %nl.p, 0
  br i1 %pfull, label %ret_full, label %store.nl.p
store.nl.p:
  store i64 %nl.p, ptr %pdi, align 8
  br label %loop
chk.self:
  %isself = icmp eq i32 %ty, 2
  br i1 %isself, label %onself, label %loop
onself:
  %istab = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wtab, i64 5)
  br i1 %istab, label %emit.tab, label %chk.brcr
emit.tab:
  %pdi.t = load i64, ptr %pdi, align 8
  %tb = call i64 @docx_putc(ptr %out, i64 %out_cap, i64 %pdi.t, i8 9)
  %tfull2 = icmp slt i64 %tb, 0
  br i1 %tfull2, label %ret_full, label %store.tab
store.tab:
  store i64 %tb, ptr %pdi, align 8
  br label %loop
chk.brcr:
  %isbr = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wbr, i64 4)
  %iscr = call i1 @docx_slice_eq(ptr %buf, i64 %noff, i64 %nlen, ptr @d.wcr, i64 4)
  %isbrcr = or i1 %isbr, %iscr
  br i1 %isbrcr, label %emit.nl.b, label %loop
emit.nl.b:
  %pdi.b = load i64, ptr %pdi, align 8
  %nl.b = call i64 @docx_putc(ptr %out, i64 %out_cap, i64 %pdi.b, i8 10)
  %bfull = icmp slt i64 %nl.b, 0
  br i1 %bfull, label %ret_full, label %store.nl.b
store.nl.b:
  store i64 %nl.b, ptr %pdi, align 8
  br label %loop
ret_full:
  %fb = load ptr, ptr %pbuf, align 8
  call void @free(ptr %fb)
  ret i32 6
finish:
  %fdi = load i64, ptr %pdi, align 8
  store i64 %fdi, ptr %out_len, align 8
  %ffb = load ptr, ptr %pbuf, align 8
  call void @free(ptr %ffb)
  ret i32 0
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
