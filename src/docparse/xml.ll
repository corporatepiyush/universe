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

; Minimal zero-copy XML pull-tokenizer sufficient for OOXML (SpreadsheetML).
;
; DESIGN:
;   * ZERO-COPY, ZERO-ALLOC scanning. The whole document sits in a caller
;     buffer; every token reports {type, name/text slice, attribute-region
;     slice} as byte OFFSETS into that buffer — never strdup'd. The only
;     caller state is a 24-byte scanner {buf, len, pos}; _next advances pos
;     and returns exactly one meaningful token per call. This is the
;     compute/memory/IO split the OOXML layer needs: fill buffer (IO by the
;     ZIP/inflate layer), walk it (pure compute), emit slices.
;   * NOT a validating parser. It recognises: start-tag `<n a="v">`, end-tag
;     `</n>`, self-closing `<n/>`, text content, CDATA (as text), and SKIPS
;     comments `<!-- -->`, PIs `<? ?>` and DOCTYPE `<! >` transparently
;     (looping until it produces a real token). Attribute values may contain
;     `>` inside quotes; the tag scan tracks quote state so it does not stop
;     early. Enough to walk sheetN.xml + sharedStrings.xml.
;   * Attribute iteration (_attr_next) is a second cursor over a tag's
;     attribute region: it yields {name-slice, value-slice} pairs; the value
;     is RAW (still escaped) — feed to _decode for entity resolution.
;   * Entity decode (_decode) resolves &amp; &lt; &gt; &quot; &apos; and
;     numeric &#NNN; / &#xHH; into UTF-8; unknown entities are copied
;     literally (lenient). Pure compute into a caller output buffer.
;
; Token struct (40 B, caller-allocated):
;   type@0(i32) [0 start, 1 end, 2 self-close, 3 text, 4 eof]
;   name_off@8(i64) name_len@16(i64)  [text: the text span]
;   attr_off@24(i64) attr_len@32(i64) [start/self-close only, else 0]
; Scanner struct (24 B, caller-allocated): buf@0 len@8 pos@16
; Attr struct (32 B, caller-allocated): name_off@0 name_len@8 val_off@16 val_len@24
;
; API:
;   void universe_docparse_xml_init(ptr sc, ptr buf, i64 len)
;   i32  universe_docparse_xml_next(ptr sc, ptr tok)      ; 0/1/2/3/4, -8, -13
;   i32  universe_docparse_xml_attr_next(ptr buf, ptr pcur, i64 end, ptr out) ; 1/0/-13/-8
;   i64  universe_docparse_xml_decode(ptr dst, ptr src, i64 len) ; len or -1

; ===================================================================== helpers

; true when c is XML whitespace (space/tab/LF/CR).
define internal i1 @xml_ws(i32 %c) #2 {
entry:
  %a = icmp eq i32 %c, 32
  %b = icmp eq i32 %c, 9
  %d = icmp eq i32 %c, 10
  %e = icmp eq i32 %c, 13
  %ab = or i1 %a, %b
  %de = or i1 %d, %e
  %r = or i1 %ab, %de
  ret i1 %r
}

; buf[i] as i32 0..255, or -1 if i>=len.
define internal i32 @xml_byte(ptr %buf, i64 %len, i64 %i) #0 {
entry:
  %oob = icmp uge i64 %i, %len
  br i1 %oob, label %no, label %yes
no:
  ret i32 -1
yes:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %z = zext i8 %c to i32
  ret i32 %z
}

; store one token's fields.
define internal void @xml_emit(ptr %tok, i32 %ty, i64 %noff, i64 %nlen, i64 %aoff, i64 %alen) #3 {
entry:
  store i32 %ty, ptr %tok, align 4
  %p1 = getelementptr inbounds i8, ptr %tok, i64 8
  store i64 %noff, ptr %p1, align 8
  %p2 = getelementptr inbounds i8, ptr %tok, i64 16
  store i64 %nlen, ptr %p2, align 8
  %p3 = getelementptr inbounds i8, ptr %tok, i64 24
  store i64 %aoff, ptr %p3, align 8
  %p4 = getelementptr inbounds i8, ptr %tok, i64 32
  store i64 %alen, ptr %p4, align 8
  ret void
}

; UTF-8 encode code point %cp into dst[di..]; return byte count (1..4).
define internal i64 @xml_utf8(ptr %dst, i64 %di, i32 %cp) #4 {
entry:
  %lt80 = icmp ult i32 %cp, 128
  br i1 %lt80, label %one, label %chk800
one:
  %c0 = trunc i32 %cp to i8
  %p0 = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %c0, ptr %p0, align 1
  ret i64 1
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
  %tp0 = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %t_b0, ptr %tp0, align 1
  %tdi1 = add i64 %di, 1
  %tp1 = getelementptr inbounds i8, ptr %dst, i64 %tdi1
  store i8 %t_b1, ptr %tp1, align 1
  ret i64 2
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
  %hp0 = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %h_b0, ptr %hp0, align 1
  %hdi1 = add i64 %di, 1
  %hp1 = getelementptr inbounds i8, ptr %dst, i64 %hdi1
  store i8 %h_b1, ptr %hp1, align 1
  %hdi2 = add i64 %di, 2
  %hp2 = getelementptr inbounds i8, ptr %dst, i64 %hdi2
  store i8 %h_b2, ptr %hp2, align 1
  ret i64 3
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
  %fp0 = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %f_b0, ptr %fp0, align 1
  %fdi1 = add i64 %di, 1
  %fp1 = getelementptr inbounds i8, ptr %dst, i64 %fdi1
  store i8 %f_b1, ptr %fp1, align 1
  %fdi2 = add i64 %di, 2
  %fp2 = getelementptr inbounds i8, ptr %dst, i64 %fdi2
  store i8 %f_b2, ptr %fp2, align 1
  %fdi3 = add i64 %di, 3
  %fp3 = getelementptr inbounds i8, ptr %dst, i64 %fdi3
  store i8 %f_b3, ptr %fp3, align 1
  ret i64 4
}

; match a named entity at src[si]; return (consumed<<8)|char, or 0 if none.
define internal i64 @xml_named(ptr %src, i64 %len, i64 %si) #0 {
entry:
  %i1 = add i64 %si, 1
  %i2 = add i64 %si, 2
  %i3 = add i64 %si, 3
  %i4 = add i64 %si, 4
  %i5 = add i64 %si, 5
  %b1 = call i32 @xml_byte(ptr %src, i64 %len, i64 %i1)
  %b2 = call i32 @xml_byte(ptr %src, i64 %len, i64 %i2)
  %b3 = call i32 @xml_byte(ptr %src, i64 %len, i64 %i3)
  %b4 = call i32 @xml_byte(ptr %src, i64 %len, i64 %i4)
  %b5 = call i32 @xml_byte(ptr %src, i64 %len, i64 %i5)
  ; amp -> '&'
  %a1 = icmp eq i32 %b1, 97
  %a2 = icmp eq i32 %b2, 109
  %a3 = icmp eq i32 %b3, 112
  %a4 = icmp eq i32 %b4, 59
  %aa = and i1 %a1, %a2
  %ab = and i1 %a3, %a4
  %amp = and i1 %aa, %ab
  ; lt -> '<'
  %l1 = icmp eq i32 %b1, 108
  %l2 = icmp eq i32 %b2, 116
  %l3 = icmp eq i32 %b3, 59
  %ll = and i1 %l1, %l2
  %lt = and i1 %ll, %l3
  ; gt -> '>'
  %g1 = icmp eq i32 %b1, 103
  %g2 = icmp eq i32 %b2, 116
  %g3 = icmp eq i32 %b3, 59
  %gg = and i1 %g1, %g2
  %gt = and i1 %gg, %g3
  ; quot -> '"'
  %q1 = icmp eq i32 %b1, 113
  %q2 = icmp eq i32 %b2, 117
  %q3 = icmp eq i32 %b3, 111
  %q4 = icmp eq i32 %b4, 116
  %q5 = icmp eq i32 %b5, 59
  %qa = and i1 %q1, %q2
  %qb = and i1 %q3, %q4
  %qc = and i1 %qa, %qb
  %quot = and i1 %qc, %q5
  ; apos -> '\''
  %s1 = icmp eq i32 %b1, 97
  %s2 = icmp eq i32 %b2, 112
  %s3 = icmp eq i32 %b3, 111
  %s4 = icmp eq i32 %b4, 115
  %s5 = icmp eq i32 %b5, 59
  %sa = and i1 %s1, %s2
  %sb = and i1 %s3, %s4
  %sc = and i1 %sa, %sb
  %apos = and i1 %sc, %s5
  ; encode: (consumed<<8)|char
  %r_apos = select i1 %apos, i64 1575, i64 0            ; (6<<8)|39
  %r_quot = select i1 %quot, i64 1570, i64 %r_apos      ; (6<<8)|34
  %r_gt   = select i1 %gt,   i64 1086, i64 %r_quot      ; (4<<8)|62
  %r_lt   = select i1 %lt,   i64 1084, i64 %r_gt        ; (4<<8)|60
  %r_amp  = select i1 %amp,  i64 1318, i64 %r_lt        ; (5<<8)|38
  ret i64 %r_amp
}

; ======================================================================= init
define void @universe_docparse_xml_init(ptr %sc, ptr %buf, i64 %len) #1 {
entry:
  %n = icmp eq ptr %sc, null
  br i1 %n, label %ret, label %store
store:
  store ptr %buf, ptr %sc, align 8
  %pl = getelementptr inbounds i8, ptr %sc, i64 8
  store i64 %len, ptr %pl, align 8
  %pp = getelementptr inbounds i8, ptr %sc, i64 16
  store i64 0, ptr %pp, align 8
  ret void
ret:
  ret void
}

; ======================================================================= next
define i32 @universe_docparse_xml_next(ptr %sc, ptr %tok) #1 {
entry:
  %scnull = icmp eq ptr %sc, null
  %tknull = icmp eq ptr %tok, null
  %anynull = or i1 %scnull, %tknull
  br i1 %anynull, label %ret_arg, label %load
ret_arg:
  ret i32 -8
ret_parse:
  ret i32 -13
load:
  %buf = load ptr, ptr %sc, align 8
  %plen = getelementptr inbounds i8, ptr %sc, i64 8
  %len = load i64, ptr %plen, align 8
  %ppos = getelementptr inbounds i8, ptr %sc, i64 16
  %pos0 = load i64, ptr %ppos, align 8
  br label %scan

scan:
  %pos = phi i64 [ %pos0, %load ], [ %posC, %cm.close ], [ %posP, %pi.close ], [ %posD, %dt.close ]
  %atend = icmp uge i64 %pos, %len
  br i1 %atend, label %emit_eof, label %peek

emit_eof:
  call void @xml_emit(ptr %tok, i32 4, i64 %pos, i64 0, i64 0, i64 0)
  store i64 %pos, ptr %ppos, align 8
  ret i32 4

peek:
  %pc = getelementptr inbounds i8, ptr %buf, i64 %pos
  %c = load i8, ptr %pc, align 1
  %islt = icmp eq i8 %c, 60
  br i1 %islt, label %tag, label %text

; ---- text content ----
text:
  br label %text.loop
text.loop:
  %tk = phi i64 [ %pos, %text ], [ %tk.n, %text.cont ]
  %tend = icmp uge i64 %tk, %len
  br i1 %tend, label %text.emit, label %text.body
text.body:
  %tp = getelementptr inbounds i8, ptr %buf, i64 %tk
  %tc = load i8, ptr %tp, align 1
  %tislt = icmp eq i8 %tc, 60
  br i1 %tislt, label %text.emit, label %text.cont
text.cont:
  %tk.n = add i64 %tk, 1
  br label %text.loop
text.emit:
  %tlen = sub i64 %tk, %pos
  call void @xml_emit(ptr %tok, i32 3, i64 %pos, i64 %tlen, i64 0, i64 0)
  store i64 %tk, ptr %ppos, align 8
  ret i32 3

; ---- a tag: dispatch on second char ----
tag:
  %pos1 = add i64 %pos, 1
  %p1end = icmp uge i64 %pos1, %len
  br i1 %p1end, label %ret_parse, label %tag.peek
tag.peek:
  %c1p = getelementptr inbounds i8, ptr %buf, i64 %pos1
  %c1 = load i8, ptr %c1p, align 1
  %isslash = icmp eq i8 %c1, 47
  br i1 %isslash, label %endtag, label %tag.chk_bang
tag.chk_bang:
  %isbang = icmp eq i8 %c1, 33
  br i1 %isbang, label %bang, label %tag.chk_pi
tag.chk_pi:
  %isq = icmp eq i8 %c1, 63
  br i1 %isq, label %pi, label %starttag

; ---- start tag ----
starttag:
  %sns = add i64 %pos, 1
  br label %st.nloop
st.nloop:
  %ni = phi i64 [ %sns, %starttag ], [ %ni.n, %st.ncont ]
  %nend = icmp uge i64 %ni, %len
  br i1 %nend, label %ret_parse, label %st.nbody
st.nbody:
  %np = getelementptr inbounds i8, ptr %buf, i64 %ni
  %nc = load i8, ptr %np, align 1
  %ncz = zext i8 %nc to i32
  %nws = call i1 @xml_ws(i32 %ncz)
  %nsl = icmp eq i8 %nc, 47
  %ngt = icmp eq i8 %nc, 62
  %nt1 = or i1 %nws, %nsl
  %nterm = or i1 %nt1, %ngt
  br i1 %nterm, label %st.namedone, label %st.ncont
st.ncont:
  %ni.n = add i64 %ni, 1
  br label %st.nloop
st.namedone:
  br label %st.qloop
st.qloop:
  %j = phi i64 [ %ni, %st.namedone ], [ %j.n, %st.qcont ]
  %inq = phi i1 [ false, %st.namedone ], [ %inq.next, %st.qcont ]
  %qch = phi i8 [ 0, %st.namedone ], [ %qch.next, %st.qcont ]
  %jend = icmp uge i64 %j, %len
  br i1 %jend, label %ret_parse, label %st.qbody
st.qbody:
  %jp = getelementptr inbounds i8, ptr %buf, i64 %j
  %jc = load i8, ptr %jp, align 1
  br i1 %inq, label %st.inq, label %st.outq
st.inq:
  %closeq = icmp eq i8 %jc, %qch
  %inq.i = xor i1 %closeq, true
  br label %st.qcont
st.outq:
  %isgt = icmp eq i8 %jc, 62
  br i1 %isgt, label %st.close, label %st.outq2
st.outq2:
  %isdq = icmp eq i8 %jc, 34
  %issq = icmp eq i8 %jc, 39
  %isquote = or i1 %isdq, %issq
  %qch.o = select i1 %isquote, i8 %jc, i8 %qch
  br label %st.qcont
st.qcont:
  %inq.next = phi i1 [ %inq.i, %st.inq ], [ %isquote, %st.outq2 ]
  %qch.next = phi i8 [ %qch, %st.inq ], [ %qch.o, %st.outq2 ]
  %j.n = add i64 %j, 1
  br label %st.qloop
st.close:
  ; back-scan from j-1 skipping ws, looking for '/'
  %bi0 = sub i64 %j, 1
  br label %st.back
st.back:
  %bi = phi i64 [ %bi0, %st.close ], [ %bi.n, %st.backcont ]
  %inrange = icmp uge i64 %bi, %ni
  br i1 %inrange, label %st.backbody, label %st.notself
st.backbody:
  %bp = getelementptr inbounds i8, ptr %buf, i64 %bi
  %bc = load i8, ptr %bp, align 1
  %bcz = zext i8 %bc to i32
  %bws = call i1 @xml_ws(i32 %bcz)
  br i1 %bws, label %st.backcont, label %st.backcheck
st.backcont:
  %bi.n = sub i64 %bi, 1
  br label %st.back
st.backcheck:
  %isslash2 = icmp eq i8 %bc, 47
  br i1 %isslash2, label %st.self, label %st.notself
st.self:
  br label %st.finish
st.notself:
  br label %st.finish
st.finish:
  %ty = phi i32 [ 2, %st.self ], [ 0, %st.notself ]
  %aend = phi i64 [ %bi, %st.self ], [ %j, %st.notself ]
  %alen = sub i64 %aend, %ni
  %nlen = sub i64 %ni, %sns
  %posn = add i64 %j, 1
  call void @xml_emit(ptr %tok, i32 %ty, i64 %sns, i64 %nlen, i64 %ni, i64 %alen)
  store i64 %posn, ptr %ppos, align 8
  ret i32 %ty

; ---- end tag ----
endtag:
  %ens = add i64 %pos, 2
  br label %et.nloop
et.nloop:
  %ei = phi i64 [ %ens, %endtag ], [ %ei.n, %et.ncont ]
  %eend = icmp uge i64 %ei, %len
  br i1 %eend, label %ret_parse, label %et.nbody
et.nbody:
  %ep = getelementptr inbounds i8, ptr %buf, i64 %ei
  %ec = load i8, ptr %ep, align 1
  %ecz = zext i8 %ec to i32
  %ews = call i1 @xml_ws(i32 %ecz)
  %egt = icmp eq i8 %ec, 62
  %et = or i1 %ews, %egt
  br i1 %et, label %et.namedone, label %et.ncont
et.ncont:
  %ei.n = add i64 %ei, 1
  br label %et.nloop
et.namedone:
  %enlen = sub i64 %ei, %ens
  br label %et.gloop
et.gloop:
  %gi = phi i64 [ %ei, %et.namedone ], [ %gi.n, %et.gcont ]
  %gend = icmp uge i64 %gi, %len
  br i1 %gend, label %ret_parse, label %et.gbody
et.gbody:
  %gp = getelementptr inbounds i8, ptr %buf, i64 %gi
  %gc = load i8, ptr %gp, align 1
  %ggt = icmp eq i8 %gc, 62
  br i1 %ggt, label %et.close, label %et.gcont
et.gcont:
  %gi.n = add i64 %gi, 1
  br label %et.gloop
et.close:
  %etposn = add i64 %gi, 1
  call void @xml_emit(ptr %tok, i32 1, i64 %ens, i64 %enlen, i64 0, i64 0)
  store i64 %etposn, ptr %ppos, align 8
  ret i32 1

; ---- <! ... > : comment, CDATA, or DOCTYPE ----
bang:
  %p2 = add i64 %pos, 2
  %p3 = add i64 %pos, 3
  %have3 = icmp ult i64 %p3, %len
  br i1 %have3, label %bang.chk, label %bang.doctype
bang.chk:
  %c2p = getelementptr inbounds i8, ptr %buf, i64 %p2
  %c2 = load i8, ptr %c2p, align 1
  %c3p = getelementptr inbounds i8, ptr %buf, i64 %p3
  %c3 = load i8, ptr %c3p, align 1
  %isdash2 = icmp eq i8 %c2, 45
  %isdash3 = icmp eq i8 %c3, 45
  %iscomment = and i1 %isdash2, %isdash3
  br i1 %iscomment, label %comment, label %bang.cdata
bang.cdata:
  %p8 = add i64 %pos, 8
  %have8 = icmp ult i64 %p8, %len
  br i1 %have8, label %cdata.chk, label %bang.doctype
cdata.chk:
  %d2p = getelementptr inbounds i8, ptr %buf, i64 %p2
  %d2 = load i8, ptr %d2p, align 1
  %d3o = add i64 %pos, 3
  %d3p = getelementptr inbounds i8, ptr %buf, i64 %d3o
  %d3 = load i8, ptr %d3p, align 1
  %d4o = add i64 %pos, 4
  %d4p = getelementptr inbounds i8, ptr %buf, i64 %d4o
  %d4 = load i8, ptr %d4p, align 1
  %d5o = add i64 %pos, 5
  %d5p = getelementptr inbounds i8, ptr %buf, i64 %d5o
  %d5 = load i8, ptr %d5p, align 1
  %d6o = add i64 %pos, 6
  %d6p = getelementptr inbounds i8, ptr %buf, i64 %d6o
  %d6 = load i8, ptr %d6p, align 1
  %d7o = add i64 %pos, 7
  %d7p = getelementptr inbounds i8, ptr %buf, i64 %d7o
  %d7 = load i8, ptr %d7p, align 1
  %d8p = getelementptr inbounds i8, ptr %buf, i64 %p8
  %d8 = load i8, ptr %d8p, align 1
  %m2 = icmp eq i8 %d2, 91
  %m3 = icmp eq i8 %d3, 67
  %m4 = icmp eq i8 %d4, 68
  %m5 = icmp eq i8 %d5, 65
  %m6 = icmp eq i8 %d6, 84
  %m7 = icmp eq i8 %d7, 65
  %m8 = icmp eq i8 %d8, 91
  %ma = and i1 %m2, %m3
  %mb = and i1 %m4, %m5
  %mc = and i1 %m6, %m7
  %md = and i1 %ma, %mb
  %me = and i1 %mc, %m8
  %iscdata = and i1 %md, %me
  br i1 %iscdata, label %cdata, label %bang.doctype
cdata:
  %cds = add i64 %pos, 9
  br label %cd.loop
cd.loop:
  %ci = phi i64 [ %cds, %cdata ], [ %ci.n, %cd.cont ]
  %ci2 = add i64 %ci, 2
  %cok = icmp ult i64 %ci2, %len
  br i1 %cok, label %cd.body, label %ret_parse
cd.body:
  %cb0p = getelementptr inbounds i8, ptr %buf, i64 %ci
  %cb0 = load i8, ptr %cb0p, align 1
  %ci1 = add i64 %ci, 1
  %cb1p = getelementptr inbounds i8, ptr %buf, i64 %ci1
  %cb1 = load i8, ptr %cb1p, align 1
  %cb2p = getelementptr inbounds i8, ptr %buf, i64 %ci2
  %cb2 = load i8, ptr %cb2p, align 1
  %cm0 = icmp eq i8 %cb0, 93
  %cm1 = icmp eq i8 %cb1, 93
  %cm2 = icmp eq i8 %cb2, 62
  %cma = and i1 %cm0, %cm1
  %cmatch = and i1 %cma, %cm2
  br i1 %cmatch, label %cd.close, label %cd.cont
cd.cont:
  %ci.n = add i64 %ci, 1
  br label %cd.loop
cd.close:
  %cdlen = sub i64 %ci, %cds
  %cdposn = add i64 %ci, 3
  call void @xml_emit(ptr %tok, i32 3, i64 %cds, i64 %cdlen, i64 0, i64 0)
  store i64 %cdposn, ptr %ppos, align 8
  ret i32 3

comment:
  %cms = add i64 %pos, 4
  br label %cm.loop
cm.loop:
  %mi = phi i64 [ %cms, %comment ], [ %mi.n, %cm.cont ]
  %mi2 = add i64 %mi, 2
  %mok = icmp ult i64 %mi2, %len
  br i1 %mok, label %cm.body, label %ret_parse
cm.body:
  %mb0p = getelementptr inbounds i8, ptr %buf, i64 %mi
  %mb0 = load i8, ptr %mb0p, align 1
  %mi1 = add i64 %mi, 1
  %mb1p = getelementptr inbounds i8, ptr %buf, i64 %mi1
  %mb1 = load i8, ptr %mb1p, align 1
  %mb2p = getelementptr inbounds i8, ptr %buf, i64 %mi2
  %mb2 = load i8, ptr %mb2p, align 1
  %mm0 = icmp eq i8 %mb0, 45
  %mm1 = icmp eq i8 %mb1, 45
  %mm2 = icmp eq i8 %mb2, 62
  %mma = and i1 %mm0, %mm1
  %mmatch = and i1 %mma, %mm2
  br i1 %mmatch, label %cm.close, label %cm.cont
cm.cont:
  %mi.n = add i64 %mi, 1
  br label %cm.loop
cm.close:
  %posC = add i64 %mi, 3
  br label %scan

pi:
  %pis = add i64 %pos, 2
  br label %pi.loop
pi.loop:
  %pii = phi i64 [ %pis, %pi ], [ %pii.n, %pi.cont ]
  %pii1 = add i64 %pii, 1
  %piok = icmp ult i64 %pii1, %len
  br i1 %piok, label %pi.body, label %ret_parse
pi.body:
  %pb0p = getelementptr inbounds i8, ptr %buf, i64 %pii
  %pb0 = load i8, ptr %pb0p, align 1
  %pb1p = getelementptr inbounds i8, ptr %buf, i64 %pii1
  %pb1 = load i8, ptr %pb1p, align 1
  %pm0 = icmp eq i8 %pb0, 63
  %pm1 = icmp eq i8 %pb1, 62
  %pmatch = and i1 %pm0, %pm1
  br i1 %pmatch, label %pi.close, label %pi.cont
pi.cont:
  %pii.n = add i64 %pii, 1
  br label %pi.loop
pi.close:
  %posP = add i64 %pii, 2
  br label %scan

bang.doctype:
  %dts = add i64 %pos, 2
  br label %dt.loop
dt.loop:
  %dqi = phi i64 [ %dts, %bang.doctype ], [ %dqi.n, %dt.cont ]
  %dtend = icmp uge i64 %dqi, %len
  br i1 %dtend, label %ret_parse, label %dt.body
dt.body:
  %dtp = getelementptr inbounds i8, ptr %buf, i64 %dqi
  %dtc = load i8, ptr %dtp, align 1
  %dtgt = icmp eq i8 %dtc, 62
  br i1 %dtgt, label %dt.close, label %dt.cont
dt.cont:
  %dqi.n = add i64 %dqi, 1
  br label %dt.loop
dt.close:
  %posD = add i64 %dqi, 1
  br label %scan
}

; ================================================================= attr_next
define i32 @universe_docparse_xml_attr_next(ptr %buf, ptr %pcur, i64 %end, ptr %out) #1 {
entry:
  %bn = icmp eq ptr %buf, null
  %cn = icmp eq ptr %pcur, null
  %on = icmp eq ptr %out, null
  %n0 = or i1 %bn, %cn
  %anull = or i1 %n0, %on
  br i1 %anull, label %ret_arg, label %ld
ret_arg:
  ret i32 -8
parse_err:
  ret i32 -13
none:
  ret i32 0
ld:
  %cur0 = load i64, ptr %pcur, align 8
  br label %ws.loop
ws.loop:
  %wi = phi i64 [ %cur0, %ld ], [ %wi.n, %ws.cont ]
  %wge = icmp uge i64 %wi, %end
  br i1 %wge, label %none, label %ws.body
ws.body:
  %wp = getelementptr inbounds i8, ptr %buf, i64 %wi
  %wc = load i8, ptr %wp, align 1
  %wcz = zext i8 %wc to i32
  %wisws = call i1 @xml_ws(i32 %wcz)
  br i1 %wisws, label %ws.cont, label %ws.done
ws.cont:
  %wi.n = add i64 %wi, 1
  br label %ws.loop
ws.done:
  ; name = [wi .. name-end); terminator is ws or '='
  br label %nm.loop
nm.loop:
  %mi2 = phi i64 [ %wi, %ws.done ], [ %mi2.n, %nm.cont ]
  %mge = icmp uge i64 %mi2, %end
  br i1 %mge, label %parse_err, label %nm.body
nm.body:
  %mp = getelementptr inbounds i8, ptr %buf, i64 %mi2
  %mc = load i8, ptr %mp, align 1
  %mcz = zext i8 %mc to i32
  %misws = call i1 @xml_ws(i32 %mcz)
  %miseq = icmp eq i8 %mc, 61
  %mterm = or i1 %misws, %miseq
  br i1 %mterm, label %nm.done, label %nm.cont
nm.cont:
  %mi2.n = add i64 %mi2, 1
  br label %nm.loop
nm.done:
  %namelen = sub i64 %mi2, %wi
  br label %eq.loop
eq.loop:
  %qi2 = phi i64 [ %mi2, %nm.done ], [ %qi2.n, %eq.cont ]
  %qge = icmp uge i64 %qi2, %end
  br i1 %qge, label %parse_err, label %eq.body
eq.body:
  %qp = getelementptr inbounds i8, ptr %buf, i64 %qi2
  %qc = load i8, ptr %qp, align 1
  %qcz = zext i8 %qc to i32
  %qisws = call i1 @xml_ws(i32 %qcz)
  br i1 %qisws, label %eq.cont, label %eq.chk
eq.cont:
  %qi2.n = add i64 %qi2, 1
  br label %eq.loop
eq.chk:
  %qiseq = icmp eq i8 %qc, 61
  br i1 %qiseq, label %eq.found, label %parse_err
eq.found:
  %aftereq = add i64 %qi2, 1
  br label %qv.loop
qv.loop:
  %vi = phi i64 [ %aftereq, %eq.found ], [ %vi.n, %qv.cont ]
  %vge = icmp uge i64 %vi, %end
  br i1 %vge, label %parse_err, label %qv.body
qv.body:
  %vp = getelementptr inbounds i8, ptr %buf, i64 %vi
  %vc = load i8, ptr %vp, align 1
  %vcz = zext i8 %vc to i32
  %visws = call i1 @xml_ws(i32 %vcz)
  br i1 %visws, label %qv.cont, label %qv.chk
qv.cont:
  %vi.n = add i64 %vi, 1
  br label %qv.loop
qv.chk:
  %visdq = icmp eq i8 %vc, 34
  %vissq = icmp eq i8 %vc, 39
  %visquote = or i1 %visdq, %vissq
  br i1 %visquote, label %qv.found, label %parse_err
qv.found:
  %valstart = add i64 %vi, 1
  br label %val.loop
val.loop:
  %xi = phi i64 [ %valstart, %qv.found ], [ %xi.n, %val.cont ]
  %xge = icmp uge i64 %xi, %end
  br i1 %xge, label %parse_err, label %val.body
val.body:
  %xp = getelementptr inbounds i8, ptr %buf, i64 %xi
  %xc = load i8, ptr %xp, align 1
  %xcl = icmp eq i8 %xc, %vc
  br i1 %xcl, label %val.done, label %val.cont
val.cont:
  %xi.n = add i64 %xi, 1
  br label %val.loop
val.done:
  %vallen = sub i64 %xi, %valstart
  %newcur = add i64 %xi, 1
  store i64 %newcur, ptr %pcur, align 8
  store i64 %wi, ptr %out, align 8
  %o1 = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %namelen, ptr %o1, align 8
  %o2 = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %valstart, ptr %o2, align 8
  %o3 = getelementptr inbounds i8, ptr %out, i64 24
  store i64 %vallen, ptr %o3, align 8
  ret i32 1
}

; =================================================================== decode
define i64 @universe_docparse_xml_decode(ptr %dst, ptr %src, i64 %len) #4 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %an = or i1 %dn, %sn
  br i1 %an, label %ret_null, label %loop
ret_null:
  ret i64 -1
loop:
  %si = phi i64 [ 0, %entry ], [ %si.n2, %adv ]
  %di = phi i64 [ 0, %entry ], [ %di.n2, %adv ]
  %atend = icmp uge i64 %si, %len
  br i1 %atend, label %done, label %body
done:
  ret i64 %di
body:
  %sp = getelementptr inbounds i8, ptr %src, i64 %si
  %c = load i8, ptr %sp, align 1
  %isamp = icmp eq i8 %c, 38
  br i1 %isamp, label %ent, label %copy
copy:
  %dp = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %c, ptr %dp, align 1
  %si.c = add i64 %si, 1
  %di.c = add i64 %di, 1
  br label %adv
ent:
  %e1 = add i64 %si, 1
  %b1e = call i32 @xml_byte(ptr %src, i64 %len, i64 %e1)
  %ishash = icmp eq i32 %b1e, 35
  br i1 %ishash, label %numeric, label %tryname
tryname:
  %nm = call i64 @xml_named(ptr %src, i64 %len, i64 %si)
  %matched = icmp ne i64 %nm, 0
  br i1 %matched, label %name.emit, label %literal
name.emit:
  %ch = and i64 %nm, 255
  %cons = lshr i64 %nm, 8
  %chb = trunc i64 %ch to i8
  %ndp = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 %chb, ptr %ndp, align 1
  %si.e1 = add i64 %si, %cons
  %di.e1 = add i64 %di, 1
  br label %adv
literal:
  %ldp = getelementptr inbounds i8, ptr %dst, i64 %di
  store i8 38, ptr %ldp, align 1
  %si.e2 = add i64 %si, 1
  %di.e2 = add i64 %di, 1
  br label %adv
numeric:
  %e2 = add i64 %si, 2
  %b2e = call i32 @xml_byte(ptr %src, i64 %len, i64 %e2)
  %isx = icmp eq i32 %b2e, 120
  %isX = icmp eq i32 %b2e, 88
  %ishex = or i1 %isx, %isX
  %e3 = add i64 %si, 3
  %digstart = select i1 %ishex, i64 %e3, i64 %e2
  br label %num.loop
num.loop:
  %nui = phi i64 [ %digstart, %numeric ], [ %nui.n, %num.cont ]
  %acc = phi i32 [ 0, %numeric ], [ %acc.n, %num.cont ]
  %nb = call i32 @xml_byte(ptr %src, i64 %len, i64 %nui)
  %issemi = icmp eq i32 %nb, 59
  br i1 %issemi, label %num.done, label %num.digit
num.digit:
  %isoob = icmp eq i32 %nb, -1
  br i1 %isoob, label %literal, label %num.dig2
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
  br i1 %is09, label %num.acc_dec, label %literal
num.acc_dec:
  %acc.d = mul i32 %acc, 10
  %acc.dn = add i32 %acc.d, %d09
  br label %num.cont
num.hex:
  %validhex = or i1 %is09, %ishexletter
  br i1 %validhex, label %num.acc_hex, label %literal
num.acc_hex:
  %digv = select i1 %is09, i32 %d09, i32 %dhex
  %acc.h = shl i32 %acc, 4
  %acc.hn = add i32 %acc.h, %digv
  br label %num.cont
num.cont:
  %acc.n = phi i32 [ %acc.dn, %num.acc_dec ], [ %acc.hn, %num.acc_hex ]
  %nui.n = add i64 %nui, 1
  br label %num.loop
num.done:
  %nbytes = call i64 @xml_utf8(ptr %dst, i64 %di, i32 %acc)
  %si.e3 = add i64 %nui, 1
  %di.e3 = add i64 %di, %nbytes
  br label %adv
adv:
  %si.n2 = phi i64 [ %si.c, %copy ], [ %si.e1, %name.emit ], [ %si.e2, %literal ], [ %si.e3, %num.done ]
  %di.n2 = phi i64 [ %di.c, %copy ], [ %di.e1, %name.emit ], [ %di.e2, %literal ], [ %di.e3, %num.done ]
  br label %loop
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(none) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
