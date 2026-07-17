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

; CSV / TSV field iterator (RFC 4180 + a configurable delimiter) over a
; caller-owned byte buffer.
;
; DESIGN:
;   * ZERO-COPY, ZERO-ALLOC. Every field is reported as {offset, length} INTO
;     the caller buffer; nothing is copied. The scanner state is a tiny 32-byte
;     struct the CALLER owns (a global, an alloca, or a slab slot) — this module
;     allocates NOTHING. That keeps the parse phase pure compute over memory the
;     caller already filled via IO, per the compute/memory/IO split: fill the
;     buffer once (one big read), then iterate fields at cache speed.
;   * One delimiter byte parameterizes CSV (',') vs TSV ('\t') vs anything else;
;     no separate code path.
;   * SIMD-first field scan: both the unquoted-field scan (stop at delim/CR/LF)
;     and the quoted-content scan (stop at ") are 128-bit <16 x i8> classify
;     loops as the PRIMARY path (SSE2 pcmpeqb / NEON cmeq.16b — baseline, no
;     runtime check), advancing by whole 16-byte chunks; a vector `icmp` builds
;     the lane mask, `bitcast` folds it to a movemask, and `llvm.cttz.i16`
;     locates the first significant byte. The scalar path handles the sub-16
;     remainder AND is the cross-check oracle. The exported
;     @universe_parse_csv_scan_field (+ _scalar twin) exposes the classify
;     primitive (delim / quote / CR / LF) for indexers and the test.
;   * RFC 4180 quoting: a field opening with '"' is a quoted field; a doubled
;     quote `""` inside it is a literal quote; commas, CR, LF and the delimiter
;     are literal inside quotes (embedded newlines/commas handled). The reported
;     offset/length for a quoted field span the CONTENT between the outer quotes
;     and the `needs_unquote` flag is set so the caller can collapse `""`->`"`
;     via @universe_parse_csv_unquote (still zero-copy until it chooses to).
;   * Record framing: LF, CRLF and bare CR all end a record; EOF ends the last
;     record whether or not a trailing newline is present. A trailing delimiter
;     yields a final empty field (RFC: "a," is two fields).
;   * ERROR CONVENTION: @..._next_field returns i32 status:
;       0 CSV_FIELD       a field was produced; the record continues (a
;                         delimiter followed the field)
;       1 CSV_RECORD_END  a field was produced and the record ends here
;       2 CSV_EOF         no field; the buffer is exhausted
;      13 PARSE           malformed quoting (e.g. text after a closing quote)
;       8 INVALID_ARG     null scanner/out
;     @..._next_record fills a caller array of field records for ONE record and
;     returns 0 (record produced), 2 (EOF), 6 (FULL: more fields than cap), 13,
;     or 8. Field record layout: off@0(i64), len@8(i64), needs_unquote@16(i32).
;
; Scanner layout (32 bytes, caller-provided): buf@0, len@8, pos@16, delim@24.
;
; API:
;   void universe_parse_csv_init(ptr sc, ptr buf, i64 len, i32 delim)
;   i32  universe_parse_csv_next_field(ptr sc, ptr out_field)   ; 0/1/2/13/8
;   i32  universe_parse_csv_next_record(ptr sc, ptr out_arr, i64 cap, ptr out_count)
;   i64  universe_parse_csv_unquote(ptr dst, ptr src, i64 len)  ; collapse ""->"

; SIMD structural-scan support: cttz locates the first matching lane in the
; 16-bit movemask of a 128-bit classify.
declare i16 @llvm.cttz.i16(i16, i1)

; ---------------------------------------------------------------- SIMD helpers
; First index in [pos,len) whose byte terminates an UNQUOTED field — the
; delimiter, LF (10) or CR (13) — else len. PRIMARY 128-bit classify + scalar
; tail. The delimiter is a runtime byte broadcast across the vector.
define internal i64 @csv_find_plain(ptr readonly %buf, i64 %len, i64 %pos, i8 %delim) #4 {
entry:
  %ins = insertelement <16 x i8> poison, i8 %delim, i64 0
  %dv = shufflevector <16 x i8> %ins, <16 x i8> poison, <16 x i32> zeroinitializer
  %end16 = add i64 %pos, 16
  %has16 = icmp ule i64 %end16, %len
  br i1 %has16, label %vloop, label %tail.head
vloop:
  %i = phi i64 [ %pos, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %ed = icmp eq <16 x i8> %v, %dv
  %e10 = icmp eq <16 x i8> %v, splat (i8 10)
  %e13 = icmp eq <16 x i8> %v, splat (i8 13)
  %o1 = or <16 x i1> %ed, %e10
  %sm = or <16 x i1> %o1, %e13
  %mm = bitcast <16 x i1> %sm to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont
locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx
vcont:
  %i.next = add nuw i64 %i, 16
  %lim = sub nuw i64 %len, 16
  %more = icmp ule i64 %i.next, %lim
  br i1 %more, label %vloop, label %tail.head
tail.head:
  %ti = phi i64 [ %pos, %entry ], [ %i.next, %vcont ], [ %ti.next, %cont ]
  %atend = icmp uge i64 %ti, %len
  br i1 %atend, label %miss, label %body
body:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %ti
  %c = load i8, ptr %p, align 1
  %isd = icmp eq i8 %c, %delim
  %islf = icmp eq i8 %c, 10
  %iscr = icmp eq i8 %c, 13
  %t1 = or i1 %isd, %islf
  %term = or i1 %t1, %iscr
  br i1 %term, label %found, label %cont
cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head
found:
  ret i64 %ti
miss:
  ret i64 %len
}

; First index in [pos,len) whose byte is a double-quote (34) — else len.
; PRIMARY 128-bit classify + scalar tail.
define internal i64 @csv_find_quote(ptr readonly %buf, i64 %len, i64 %pos) #4 {
entry:
  %end16 = add i64 %pos, 16
  %has16 = icmp ule i64 %end16, %len
  br i1 %has16, label %vloop, label %tail.head
vloop:
  %i = phi i64 [ %pos, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %eq = icmp eq <16 x i8> %v, splat (i8 34)
  %mm = bitcast <16 x i1> %eq to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont
locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx
vcont:
  %i.next = add nuw i64 %i, 16
  %lim = sub nuw i64 %len, 16
  %more = icmp ule i64 %i.next, %lim
  br i1 %more, label %vloop, label %tail.head
tail.head:
  %ti = phi i64 [ %pos, %entry ], [ %i.next, %vcont ], [ %ti.next, %cont ]
  %atend = icmp uge i64 %ti, %len
  br i1 %atend, label %miss, label %body
body:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %ti
  %c = load i8, ptr %p, align 1
  %isq = icmp eq i8 %c, 34
  br i1 %isq, label %found, label %cont
cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head
found:
  ret i64 %ti
miss:
  ret i64 %len
}

; ------------------------------------------------------------------------ init
define void @universe_parse_csv_init(ptr %sc, ptr %buf, i64 %len, i32 %delim) local_unnamed_addr #0 {
entry:
  %null = icmp eq ptr %sc, null
  br i1 %null, label %done, label %do
do:
  store ptr %buf, ptr %sc, align 8
  %lp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  store i64 %len, ptr %lp, align 8
  %pp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  store i64 0, ptr %pp, align 8
  %dp = getelementptr inbounds nuw i8, ptr %sc, i64 24
  %db = trunc i32 %delim to i8
  store i8 %db, ptr %dp, align 1
  br label %done
done:
  ret void
}

; --------------------------------------------------------------- one field step
; Reads one field from sc into out_field {off, len, needs_unquote}; advances pos
; past the field and its terminator. Returns 0/1/2/13.
define internal i32 @csv_step(ptr %sc, ptr %out) #1 {
entry:
  %buf = load ptr, ptr %sc, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  %len = load i64, ptr %lenp, align 8
  %posp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  %pos = load i64, ptr %posp, align 8
  %dp = getelementptr inbounds nuw i8, ptr %sc, i64 24
  %delim = load i8, ptr %dp, align 1
  %atend = icmp uge i64 %pos, %len
  br i1 %atend, label %eof, label %start
eof:
  ret i32 2
start:
  %sp = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  %sc0 = load i8, ptr %sp, align 1
  %isQuote = icmp eq i8 %sc0, 34
  br i1 %isQuote, label %quoted, label %plain

; ---- unquoted field: scan to delimiter / CR / LF / EOF ----
; SIMD-first: the 128-bit classify jumps whole 16-byte runs to the terminator.
plain:
  %pi = call i64 @csv_find_plain(ptr %buf, i64 %len, i64 %pos, i8 %delim)
  br label %plain.term
plain.term:
  %plain.flen = sub i64 %pi, %pos
  call void @csv_write(ptr %out, i64 %pos, i64 %plain.flen, i32 0)
  %pr = call i32 @csv_term(ptr %sc, ptr %buf, i64 %len, i8 %delim, i64 %pi)
  ret i32 %pr

; ---- quoted field: content between the outer quotes, "" is a literal quote ----
; SIMD-first: the 128-bit classify jumps whole 16-byte runs to the next quote;
; the scalar code then resolves doubled-quote ("") vs field close.
quoted:
  %qstart = add nuw i64 %pos, 1
  br label %q.head
q.head:
  %qs = phi i64 [ %qstart, %quoted ], [ %qi.n2, %q.dbl ]
  %qi = call i64 @csv_find_quote(ptr %buf, i64 %len, i64 %qs)
  %qi.end = icmp uge i64 %qi, %len
  br i1 %qi.end, label %q.unterm, label %q.quote
q.quote:
  ; a quote inside: doubled ("") => literal, else it closes the field
  %qj = add nuw i64 %qi, 1
  %qj.end = icmp uge i64 %qj, %len
  br i1 %qj.end, label %q.close, label %q.chkdbl
q.chkdbl:
  %qjp = getelementptr inbounds nuw i8, ptr %buf, i64 %qj
  %qjc = load i8, ptr %qjp, align 1
  %isdbl = icmp eq i8 %qjc, 34
  br i1 %isdbl, label %q.dbl, label %q.close
q.dbl:
  %qi.n2 = add nuw i64 %qi, 2
  br label %q.head
q.close:
  ; content = [qstart, qi) ; cursor after closing quote = qi+1
  %qlen = sub i64 %qi, %qstart
  call void @csv_write(ptr %out, i64 %qstart, i64 %qlen, i32 1)
  %qafter = add nuw i64 %qi, 1
  ; a quoted field must be followed by delimiter / CR / LF / EOF
  %qa.end = icmp uge i64 %qafter, %len
  br i1 %qa.end, label %q.ok.eof, label %q.chkterm
q.ok.eof:
  store i64 %qafter, ptr %posp, align 8
  ret i32 1
q.chkterm:
  %qap = getelementptr inbounds nuw i8, ptr %buf, i64 %qafter
  %qac = load i8, ptr %qap, align 1
  %qa.delim = icmp eq i8 %qac, %delim
  %qa.lf = icmp eq i8 %qac, 10
  %qa.cr = icmp eq i8 %qac, 13
  %qt1 = or i1 %qa.delim, %qa.lf
  %qa.term = or i1 %qt1, %qa.cr
  br i1 %qa.term, label %q.doterm, label %q.badterm
q.badterm:
  ret i32 13
q.doterm:
  %qr = call i32 @csv_term(ptr %sc, ptr %buf, i64 %len, i8 %delim, i64 %qafter)
  ret i32 %qr
q.unterm:
  ; unterminated quoted field: treat remaining as content, record end at EOF
  %qulen = sub i64 %len, %qstart
  call void @csv_write(ptr %out, i64 %qstart, i64 %qulen, i32 1)
  store i64 %len, ptr %posp, align 8
  ret i32 1
}

; Store a field record and consume the terminator at %ti; returns 0 (delimiter,
; more fields) or 1 (record end via CR/LF/EOF). Updates scanner pos.
define internal i32 @csv_term(ptr %sc, ptr readonly %buf, i64 %len, i8 %delim, i64 %ti) #1 {
entry:
  %posp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  %ti.end = icmp uge i64 %ti, %len
  br i1 %ti.end, label %at.eof, label %rd
at.eof:
  store i64 %ti, ptr %posp, align 8
  ret i32 1
rd:
  %tp = getelementptr inbounds nuw i8, ptr %buf, i64 %ti
  %tc = load i8, ptr %tp, align 1
  %isdelim = icmp eq i8 %tc, %delim
  br i1 %isdelim, label %do.delim, label %chk.lf
do.delim:
  %after.d = add nuw i64 %ti, 1
  store i64 %after.d, ptr %posp, align 8
  ret i32 0
chk.lf:
  %islf = icmp eq i8 %tc, 10
  br i1 %islf, label %do.lf, label %do.cr
do.lf:
  %after.lf = add nuw i64 %ti, 1
  store i64 %after.lf, ptr %posp, align 8
  ret i32 1
do.cr:
  ; CR, maybe CRLF
  %after.cr = add nuw i64 %ti, 1
  %has.next = icmp ult i64 %after.cr, %len
  br i1 %has.next, label %cr.peek, label %cr.only
cr.peek:
  %np = getelementptr inbounds nuw i8, ptr %buf, i64 %after.cr
  %nc = load i8, ptr %np, align 1
  %nlf = icmp eq i8 %nc, 10
  %after.crlf = add nuw i64 %ti, 2
  %after = select i1 %nlf, i64 %after.crlf, i64 %after.cr
  store i64 %after, ptr %posp, align 8
  ret i32 1
cr.only:
  store i64 %after.cr, ptr %posp, align 8
  ret i32 1
}

; Write {off, len, needs_unquote} into a field record.
define internal void @csv_write(ptr %out, i64 %off, i64 %flen, i32 %unq) #2 {
entry:
  store i64 %off, ptr %out, align 8
  %lp = getelementptr inbounds nuw i8, ptr %out, i64 8
  store i64 %flen, ptr %lp, align 8
  %up = getelementptr inbounds nuw i8, ptr %out, i64 16
  store i32 %unq, ptr %up, align 4
  ret void
}

; ================================================================= public API
define i32 @universe_parse_csv_next_field(ptr %sc, ptr %out) local_unnamed_addr #1 {
entry:
  %scn = icmp eq ptr %sc, null
  %on = icmp eq ptr %out, null
  %bad = or i1 %scn, %on
  br i1 %bad, label %argerr, label %go
argerr:
  ret i32 8
go:
  %r = call i32 @csv_step(ptr %sc, ptr %out)
  ret i32 %r
}

define i32 @universe_parse_csv_next_record(ptr %sc, ptr %arr, i64 %cap, ptr %count) local_unnamed_addr #1 {
entry:
  %scn = icmp eq ptr %sc, null
  %an = icmp eq ptr %arr, null
  %cn = icmp eq ptr %count, null
  %b1 = or i1 %scn, %an
  %bad = or i1 %b1, %cn
  br i1 %bad, label %argerr, label %loop
argerr:
  ret i32 8
loop:
  %n = phi i64 [ 0, %entry ], [ %n.next, %cont ]
  %cap.ok = icmp ult i64 %n, %cap
  br i1 %cap.ok, label %step, label %full
step:
  %slot = getelementptr inbounds nuw { i64, i64, i32, i32 }, ptr %arr, i64 %n
  %st = call i32 @csv_step(ptr %sc, ptr %slot)
  ; st: 0 field(more), 1 field(record end), 2 eof, 13 parse
  switch i32 %st, label %perr [
    i32 0, label %cont
    i32 1, label %recend
    i32 2, label %eofchk
  ]
cont:
  %n.next = add nuw i64 %n, 1
  br label %loop
recend:
  %n.re = add nuw i64 %n, 1
  store i64 %n.re, ptr %count, align 8
  ret i32 0
eofchk:
  ; EOF hit: if we already gathered fields, it's a record end; else true EOF
  %some = icmp ugt i64 %n, 0
  br i1 %some, label %eof.rec, label %eof.none
eof.rec:
  store i64 %n, ptr %count, align 8
  ret i32 0
eof.none:
  store i64 0, ptr %count, align 8
  ret i32 2
full:
  store i64 %n, ptr %count, align 8
  ret i32 6
perr:
  store i64 %n, ptr %count, align 8
  ret i32 13
}

; Collapse doubled quotes ("") into a single quote while copying content to dst.
; src/len is the between-quotes content of a quoted field. Returns dst length.
define i64 @universe_parse_csv_unquote(ptr %dst, ptr readonly %src, i64 %len) local_unnamed_addr #3 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %bad = or i1 %dn, %sn
  br i1 %bad, label %argerr, label %loop
argerr:
  ret i64 -1
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n1, %plain ], [ %i.n2, %dbl ]
  %o = phi i64 [ 0, %entry ], [ %o.n1, %plain ], [ %o.n2, %dbl ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %fin, label %rd
fin:
  ret i64 %o
rd:
  %p = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %c = load i8, ptr %p, align 1
  %isq = icmp eq i8 %c, 34
  br i1 %isq, label %qchk, label %plain
qchk:
  %j = add nuw i64 %i, 1
  %j.ok = icmp ult i64 %j, %len
  br i1 %j.ok, label %qpeek, label %plain
qpeek:
  %jp = getelementptr inbounds nuw i8, ptr %src, i64 %j
  %jc = load i8, ptr %jp, align 1
  %isdbl = icmp eq i8 %jc, 34
  br i1 %isdbl, label %dbl, label %plain
plain:
  %op = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %c, ptr %op, align 1
  %i.n1 = add nuw i64 %i, 1
  %o.n1 = add nuw i64 %o, 1
  br label %loop
dbl:
  %op2 = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 34, ptr %op2, align 1
  %i.n2 = add nuw i64 %i, 2
  %o.n2 = add nuw i64 %o, 1
  br label %loop
}

; ============================================ exported SIMD classify primitive
; First index in [pos,len) whose byte is CSV-significant — the delimiter, a
; double-quote (34), CR (13) or LF (10) — else len. Exposes the classify scan
; for indexers and is the vector-vs-scalar cross-check surface in the test.

; VECTOR primary.
define i64 @universe_parse_csv_scan_field(ptr readonly %buf, i64 %len, i64 %pos, i32 %delim) local_unnamed_addr #4 {
entry:
  %db = trunc i32 %delim to i8
  %ins = insertelement <16 x i8> poison, i8 %db, i64 0
  %dv = shufflevector <16 x i8> %ins, <16 x i8> poison, <16 x i32> zeroinitializer
  %end16 = add i64 %pos, 16
  %has16 = icmp ule i64 %end16, %len
  br i1 %has16, label %vloop, label %tail.head
vloop:
  %i = phi i64 [ %pos, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %ed = icmp eq <16 x i8> %v, %dv
  %eqt = icmp eq <16 x i8> %v, splat (i8 34)
  %e10 = icmp eq <16 x i8> %v, splat (i8 10)
  %e13 = icmp eq <16 x i8> %v, splat (i8 13)
  %o1 = or <16 x i1> %ed, %eqt
  %o2 = or <16 x i1> %e10, %e13
  %sm = or <16 x i1> %o1, %o2
  %mm = bitcast <16 x i1> %sm to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont
locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx
vcont:
  %i.next = add nuw i64 %i, 16
  %lim = sub nuw i64 %len, 16
  %more = icmp ule i64 %i.next, %lim
  br i1 %more, label %vloop, label %tail.head
tail.head:
  %ti = phi i64 [ %pos, %entry ], [ %i.next, %vcont ], [ %ti.next, %cont ]
  %atend = icmp uge i64 %ti, %len
  br i1 %atend, label %miss, label %body
body:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %ti
  %c = load i8, ptr %p, align 1
  %isd = icmp eq i8 %c, %db
  %isq = icmp eq i8 %c, 34
  %islf = icmp eq i8 %c, 10
  %iscr = icmp eq i8 %c, 13
  %t1 = or i1 %isd, %isq
  %t2 = or i1 %islf, %iscr
  %term = or i1 %t1, %t2
  br i1 %term, label %found, label %cont
cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head
found:
  ret i64 %ti
miss:
  ret i64 %len
}

; SCALAR ORACLE: bit-identical twin of universe_parse_csv_scan_field.
define i64 @universe_parse_csv_scan_field_scalar(ptr readonly %buf, i64 %len, i64 %pos, i32 %delim) local_unnamed_addr #4 {
entry:
  %db = trunc i32 %delim to i8
  br label %loop
loop:
  %i = phi i64 [ %pos, %entry ], [ %i.next, %cont ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %miss, label %body
body:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %isd = icmp eq i8 %c, %db
  %isq = icmp eq i8 %c, 34
  %islf = icmp eq i8 %c, 10
  %iscr = icmp eq i8 %c, 13
  %t1 = or i1 %isd, %isq
  %t2 = or i1 %islf, %iscr
  %term = or i1 %t1, %t2
  br i1 %term, label %found, label %cont
cont:
  %i.next = add nuw i64 %i, 1
  br label %loop
found:
  ret i64 %i
miss:
  ret i64 %len
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
