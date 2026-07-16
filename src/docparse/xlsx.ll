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

; XLSX (ISO/IEC 29500 SpreadsheetML) reader over a caller ZIP buffer.
;
; DESIGN:
;   * BUILDS ON compress/zip + compress/inflate. The whole .xlsx is a ZIP in
;     the caller buffer; we locate `xl/sharedStrings.xml` and
;     `xl/worksheets/sheet1.xml`, and inflate each ONCE into a private malloc
;     buffer (the only allocations). Everything after is zero-copy: cell
;     values are reported as {ptr,len} slices INTO the inflated sheet or the
;     inflated shared-string table. This is the canonical IO/compute split:
;     zip+inflate does the IO/decompress, the XML walk is pure compute.
;   * The shared-string table is pre-indexed once at open() into a flat array
;     of (offset,length) pairs (relative to the shared buffer) — cell type
;     t="s" is then an O(1) array lookup. Rich-text <si> (multiple <r><t>)
;     resolves to the FIRST <t> run (documented limitation for now).
;   * Cell iteration is a small state machine over the sheet's XML pull-token
;     stream: <row r="N"> sets the current row; <c r="A1" t="..."> yields one
;     cell whose value comes from the inner <v> (number / shared index /
;     boolean / formula string) or <is><t> (inline string). A1 refs are
;     decoded to (col,row) by base-26 letters + decimal digits.
;   * Numbers/booleans are parsed to a double in-place (no libc strtod). String
;     slices stay RAW (escaped) — feed to universe_docparse_xml_decode.
;
; Workbook struct (80 B, caller-allocated):
;   reader@0 (32 B, zip reader) shared_buf@32(ptr) shared_len@40(i64)
;   sheet_buf@48(ptr) sheet_len@56(i64) sst_ptr@64(ptr) sst_count@72(i64)
; Cursor struct (32 B, caller-allocated): xml-scanner@0 (24 B) cur_row@24(i64)
; Cell struct (56 B, caller-allocated):
;   row@0(i64) col@8(i64) type@16(i32) [0 num,1 shared,2 bool,3 string,4 empty]
;   val_ptr@24(ptr) val_len@32(i64) num@40(double) sidx@48(i64)
;
; API:
;   i32 universe_docparse_xlsx_open(ptr buf, i64 len, ptr wb)   ; 0/1/2/5/13
;   void universe_docparse_xlsx_close(ptr wb)
;   i64 universe_docparse_xlsx_sst_count(ptr wb)
;   i32 universe_docparse_xlsx_sst_get(ptr wb, i64 idx, ptr out16) ; 0/7
;   void universe_docparse_xlsx_sheet_init(ptr wb, ptr cursor)
;   i32 universe_docparse_xlsx_cell_next(ptr wb, ptr cursor, ptr cell) ; 1/0/-13/-8

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"

declare i32 @universe_compress_zip_open(ptr, i64, ptr)
declare i64 @universe_compress_zip_count(ptr)
declare i32 @universe_compress_zip_entry(ptr, i64, ptr)
declare i64 @universe_compress_zip_extract(ptr, ptr, ptr, i64)

declare void @universe_docparse_xml_init(ptr, ptr, i64)
declare i32 @universe_docparse_xml_next(ptr, ptr)
declare i32 @universe_docparse_xml_attr_next(ptr, ptr, i64, ptr)

@x.shared = private constant [20 x i8] c"xl/sharedStrings.xml"
@x.sheet  = private constant [24 x i8] c"xl/worksheets/sheet1.xml"
@x.si  = private constant [2 x i8] c"si"
@x.t   = private constant [1 x i8] c"t"
@x.c   = private constant [1 x i8] c"c"
@x.v   = private constant [1 x i8] c"v"
@x.row = private constant [3 x i8] c"row"
@x.r   = private constant [1 x i8] c"r"
@x.sv  = private constant [1 x i8] c"s"
@x.bv  = private constant [1 x i8] c"b"
@x.inl = private constant [9 x i8] c"inlineStr"
@x.strv = private constant [3 x i8] c"str"

; ===================================================================== helpers

; compare buffer slice (buf+off,len) to (cstr,clen). No libc.
define internal i1 @xlsx_slice_eq(ptr %buf, i64 %off, i64 %len, ptr %cstr, i64 %clen) #0 {
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

; parse an A1-style reference at (p,len) -> *pcol (1-based), *prow.
define internal void @xlsx_parse_ref(ptr %p, i64 %len, ptr %pcol, ptr %prow) #6 {
entry:
  br label %col.loop
col.loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %col.cont ]
  %col = phi i64 [ 0, %entry ], [ %col.n, %col.cont ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %fin, label %col.body
col.body:
  %cp = getelementptr inbounds i8, ptr %p, i64 %i
  %c = load i8, ptr %cp, align 1
  %cz = zext i8 %c to i64
  %ge = icmp uge i64 %cz, 65
  %le = icmp ule i64 %cz, 90
  %isupper = and i1 %ge, %le
  %gel = icmp uge i64 %cz, 97
  %lel = icmp ule i64 %cz, 122
  %islower = and i1 %gel, %lel
  %isletter = or i1 %isupper, %islower
  br i1 %isletter, label %col.acc, label %row.start
col.acc:
  %base = select i1 %isupper, i64 65, i64 97
  %lv = sub i64 %cz, %base
  %lv1 = add i64 %lv, 1
  %m = mul i64 %col, 26
  %col.n = add i64 %m, %lv1
  br label %col.cont
col.cont:
  %i.n = add i64 %i, 1
  br label %col.loop
fin:
  store i64 %col, ptr %pcol, align 8
  store i64 0, ptr %prow, align 8
  ret void
row.start:
  br label %row.loop
row.loop:
  %j = phi i64 [ %i, %row.start ], [ %j.n, %row.cont ]
  %row = phi i64 [ 0, %row.start ], [ %row.n, %row.cont ]
  %rend = icmp uge i64 %j, %len
  br i1 %rend, label %store, label %row.body
row.body:
  %rp = getelementptr inbounds i8, ptr %p, i64 %j
  %rc = load i8, ptr %rp, align 1
  %rcz = zext i8 %rc to i64
  %d = sub i64 %rcz, 48
  %isd = icmp ult i64 %d, 10
  br i1 %isd, label %row.acc, label %store
row.acc:
  %rm = mul i64 %row, 10
  %row.n = add i64 %rm, %d
  br label %row.cont
row.cont:
  %j.n = add i64 %j, 1
  br label %row.loop
store:
  store i64 %col, ptr %pcol, align 8
  store i64 %row, ptr %prow, align 8
  ret void
}

; parse a signed integer at (p,len).
define internal i64 @xlsx_atoi(ptr %p, i64 %len) #6 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret0, label %sign
ret0:
  ret i64 0
sign:
  %c0 = load i8, ptr %p, align 1
  %isneg = icmp eq i8 %c0, 45
  %start = select i1 %isneg, i64 1, i64 0
  %sgn = select i1 %isneg, i64 -1, i64 1
  br label %loop
loop:
  %i = phi i64 [ %start, %sign ], [ %i.n, %acc ]
  %val = phi i64 [ 0, %sign ], [ %val.n, %acc ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %done, label %body
body:
  %cp = getelementptr inbounds i8, ptr %p, i64 %i
  %c = load i8, ptr %cp, align 1
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
  %r = mul i64 %val, %sgn
  ret i64 %r
}

; parse a floating value at (p,len). Handles sign/int/frac/exponent.
define internal double @xlsx_atof(ptr %p, i64 %len) #7 {
entry:
  %pi = alloca i64, align 8
  %pval = alloca double, align 8
  %psign = alloca double, align 8
  %pscale = alloca double, align 8
  %pesign = alloca i64, align 8
  %pexp = alloca i64, align 8
  %pk = alloca i64, align 8
  store i64 0, ptr %pi, align 8
  store double 0.0, ptr %pval, align 8
  store double 1.0, ptr %psign, align 8
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret0, label %sign
ret0:
  ret double 0.0
sign:
  %c0 = load i8, ptr %p, align 1
  %isneg = icmp eq i8 %c0, 45
  %isplus = icmp eq i8 %c0, 43
  br i1 %isneg, label %setneg, label %chkplus
setneg:
  store double -1.0, ptr %psign, align 8
  store i64 1, ptr %pi, align 8
  br label %int.loop
chkplus:
  br i1 %isplus, label %setplus, label %int.loop
setplus:
  store i64 1, ptr %pi, align 8
  br label %int.loop
int.loop:
  %ii = load i64, ptr %pi, align 8
  %iend = icmp uge i64 %ii, %len
  br i1 %iend, label %finish, label %int.body
int.body:
  %icp = getelementptr inbounds i8, ptr %p, i64 %ii
  %ic = load i8, ptr %icp, align 1
  %icz = zext i8 %ic to i64
  %idd = sub i64 %icz, 48
  %iisd = icmp ult i64 %idd, 10
  br i1 %iisd, label %int.acc, label %frac.dot
int.acc:
  %iv = load double, ptr %pval, align 8
  %iv10 = fmul double %iv, 1.000000e+01
  %idf = uitofp i64 %idd to double
  %iv2 = fadd double %iv10, %idf
  store double %iv2, ptr %pval, align 8
  %ii1 = add i64 %ii, 1
  store i64 %ii1, ptr %pi, align 8
  br label %int.loop
frac.dot:
  %fcp = getelementptr inbounds i8, ptr %p, i64 %ii
  %fc = load i8, ptr %fcp, align 1
  %isdot = icmp eq i8 %fc, 46
  br i1 %isdot, label %frac.start, label %exp.check
frac.start:
  %fi1 = add i64 %ii, 1
  store i64 %fi1, ptr %pi, align 8
  store double 1.000000e-01, ptr %pscale, align 8
  br label %frac.loop
frac.loop:
  %fj = load i64, ptr %pi, align 8
  %fend = icmp uge i64 %fj, %len
  br i1 %fend, label %finish, label %frac.body
frac.body:
  %fjp = getelementptr inbounds i8, ptr %p, i64 %fj
  %fjc = load i8, ptr %fjp, align 1
  %fjz = zext i8 %fjc to i64
  %fdd = sub i64 %fjz, 48
  %fisd = icmp ult i64 %fdd, 10
  br i1 %fisd, label %frac.acc, label %exp.check
frac.acc:
  %fv = load double, ptr %pval, align 8
  %fsc = load double, ptr %pscale, align 8
  %fdf = uitofp i64 %fdd to double
  %fterm = fmul double %fdf, %fsc
  %fv2 = fadd double %fv, %fterm
  store double %fv2, ptr %pval, align 8
  %fsc2 = fmul double %fsc, 1.000000e-01
  store double %fsc2, ptr %pscale, align 8
  %fj1 = add i64 %fj, 1
  store i64 %fj1, ptr %pi, align 8
  br label %frac.loop
exp.check:
  %ei = load i64, ptr %pi, align 8
  %eend = icmp uge i64 %ei, %len
  br i1 %eend, label %finish, label %exp.chk2
exp.chk2:
  %ecp = getelementptr inbounds i8, ptr %p, i64 %ei
  %ec = load i8, ptr %ecp, align 1
  %ise = icmp eq i8 %ec, 101
  %isE = icmp eq i8 %ec, 69
  %isexp = or i1 %ise, %isE
  br i1 %isexp, label %exp.start, label %finish
exp.start:
  %es1 = add i64 %ei, 1
  store i64 %es1, ptr %pi, align 8
  store i64 1, ptr %pesign, align 8
  store i64 0, ptr %pexp, align 8
  %spos = load i64, ptr %pi, align 8
  %spend = icmp uge i64 %spos, %len
  br i1 %spend, label %exp.apply, label %exp.sign
exp.sign:
  %scp = getelementptr inbounds i8, ptr %p, i64 %spos
  %sc = load i8, ptr %scp, align 1
  %sneg = icmp eq i8 %sc, 45
  %splus = icmp eq i8 %sc, 43
  br i1 %sneg, label %exp.setneg, label %exp.chkplus2
exp.setneg:
  store i64 -1, ptr %pesign, align 8
  %sp1 = add i64 %spos, 1
  store i64 %sp1, ptr %pi, align 8
  br label %exp.dloop
exp.chkplus2:
  br i1 %splus, label %exp.setplus2, label %exp.dloop
exp.setplus2:
  %sp2 = add i64 %spos, 1
  store i64 %sp2, ptr %pi, align 8
  br label %exp.dloop
exp.dloop:
  %ej = load i64, ptr %pi, align 8
  %ejend = icmp uge i64 %ej, %len
  br i1 %ejend, label %exp.apply, label %exp.dbody
exp.dbody:
  %ejp = getelementptr inbounds i8, ptr %p, i64 %ej
  %ejc = load i8, ptr %ejp, align 1
  %ejz = zext i8 %ejc to i64
  %edd = sub i64 %ejz, 48
  %eisd = icmp ult i64 %edd, 10
  br i1 %eisd, label %exp.dacc, label %exp.apply
exp.dacc:
  %ev = load i64, ptr %pexp, align 8
  %ev10 = mul i64 %ev, 10
  %ev2 = add i64 %ev10, %edd
  store i64 %ev2, ptr %pexp, align 8
  %ej1 = add i64 %ej, 1
  store i64 %ej1, ptr %pi, align 8
  br label %exp.dloop
exp.apply:
  %expv = load i64, ptr %pexp, align 8
  %esg = load i64, ptr %pesign, align 8
  %isnege = icmp slt i64 %esg, 0
  store i64 0, ptr %pk, align 8
  br label %exp.aploop
exp.aploop:
  %kk = load i64, ptr %pk, align 8
  %kdone = icmp uge i64 %kk, %expv
  br i1 %kdone, label %finish, label %exp.apbody
exp.apbody:
  %vv = load double, ptr %pval, align 8
  %vmul = fmul double %vv, 1.000000e+01
  %vdiv = fdiv double %vv, 1.000000e+01
  %vsel = select i1 %isnege, double %vdiv, double %vmul
  store double %vsel, ptr %pval, align 8
  %kk1 = add i64 %kk, 1
  store i64 %kk1, ptr %pk, align 8
  br label %exp.aploop
finish:
  %fval = load double, ptr %pval, align 8
  %fsg = load double, ptr %psign, align 8
  %res = fmul double %fsg, %fval
  ret double %res
}

; find entry by exact name, inflate into a fresh malloc buffer; store {ptr,len}.
define internal i32 @xlsx_find(ptr %reader, ptr %name, i64 %namelen, ptr %pbuf, ptr %plen) #1 {
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
  %match = call i1 @xlsx_slice_eq(ptr %ename, i64 0, i64 %enl, ptr %name, i64 %namelen)
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
  ret i32 13
ok:
  store ptr %p, ptr %pbuf, align 8
  store i64 %n, ptr %plen, align 8
  ret i32 0
}

; index the shared-string table: flat (off,len) pairs into wb->sst_ptr.
define internal i32 @xlsx_build_sst(ptr %wb) #1 {
entry:
  %sbp = getelementptr inbounds i8, ptr %wb, i64 32
  %sb = load ptr, ptr %sbp, align 8
  %slp = getelementptr inbounds i8, ptr %wb, i64 40
  %sl = load i64, ptr %slp, align 8
  %isnull = icmp eq ptr %sb, null
  br i1 %isnull, label %empty, label %alloc_state
empty:
  ret i32 0
alloc_state:
  %sc = alloca [24 x i8], align 8
  %tok = alloca [40 x i8], align 8
  ; ---- count pass ----
  call void @universe_docparse_xml_init(ptr %sc, ptr %sb, i64 %sl)
  br label %cloop
cloop:
  %ncount = phi i64 [ 0, %alloc_state ], [ %ncount.n, %cloop.cont ]
  %crc = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %cty = load i32, ptr %tok, align 4
  %ceof = icmp eq i32 %cty, 4
  br i1 %ceof, label %cdone, label %cbody
cbody:
  %cisstart = icmp eq i32 %cty, 0
  br i1 %cisstart, label %cchk, label %cloop.keep
cchk:
  %cnop = getelementptr inbounds i8, ptr %tok, i64 8
  %cnoff = load i64, ptr %cnop, align 8
  %cnlp = getelementptr inbounds i8, ptr %tok, i64 16
  %cnlen = load i64, ptr %cnlp, align 8
  %cissi = call i1 @xlsx_slice_eq(ptr %sb, i64 %cnoff, i64 %cnlen, ptr @x.si, i64 2)
  br i1 %cissi, label %cinc, label %cloop.keep
cinc:
  %ncount.i = add i64 %ncount, 1
  br label %cloop.cont
cloop.keep:
  br label %cloop.cont
cloop.cont:
  %ncount.n = phi i64 [ %ncount.i, %cinc ], [ %ncount, %cloop.keep ]
  br label %cloop
cdone:
  %hasany = icmp ugt i64 %ncount, 0
  br i1 %hasany, label %doalloc, label %ret_ok
ret_ok:
  ret i32 0
doalloc:
  %bytes = mul i64 %ncount, 16
  %arr = call ptr @malloc(i64 %bytes)
  %arrnull = icmp eq ptr %arr, null
  br i1 %arrnull, label %ret_oom, label %store_arr
ret_oom:
  ret i32 2
store_arr:
  %ap = getelementptr inbounds i8, ptr %wb, i64 64
  store ptr %arr, ptr %ap, align 8
  %cp = getelementptr inbounds i8, ptr %wb, i64 72
  store i64 %ncount, ptr %cp, align 8
  ; ---- fill pass (state via allocas) ----
  %pinsi = alloca i32, align 4
  %pcap = alloca i32, align 4
  %pexp = alloca i32, align 4
  %pidx = alloca i64, align 8
  %poff = alloca i64, align 8
  %plen2 = alloca i64, align 8
  store i32 0, ptr %pinsi, align 4
  store i32 0, ptr %pcap, align 4
  store i32 0, ptr %pexp, align 4
  store i64 0, ptr %pidx, align 8
  store i64 0, ptr %poff, align 8
  store i64 0, ptr %plen2, align 8
  call void @universe_docparse_xml_init(ptr %sc, ptr %sb, i64 %sl)
  br label %floop
floop:
  %frc = call i32 @universe_docparse_xml_next(ptr %sc, ptr %tok)
  %fty = load i32, ptr %tok, align 4
  %feof = icmp eq i32 %fty, 4
  br i1 %feof, label %fdone, label %fbody
fbody:
  %fnop = getelementptr inbounds i8, ptr %tok, i64 8
  %fnoff = load i64, ptr %fnop, align 8
  %fnlp = getelementptr inbounds i8, ptr %tok, i64 16
  %fnlen = load i64, ptr %fnlp, align 8
  %fisstart = icmp eq i32 %fty, 0
  br i1 %fisstart, label %fstart, label %fchktext
fstart:
  %fsi = call i1 @xlsx_slice_eq(ptr %sb, i64 %fnoff, i64 %fnlen, ptr @x.si, i64 2)
  br i1 %fsi, label %fnewsi, label %fstart_t
fnewsi:
  store i32 1, ptr %pinsi, align 4
  store i32 0, ptr %pcap, align 4
  store i32 0, ptr %pexp, align 4
  store i64 0, ptr %poff, align 8
  store i64 0, ptr %plen2, align 8
  br label %floop
fstart_t:
  %ft = call i1 @xlsx_slice_eq(ptr %sb, i64 %fnoff, i64 %fnlen, ptr @x.t, i64 1)
  %insiv = load i32, ptr %pinsi, align 4
  %insib = icmp ne i32 %insiv, 0
  %capv = load i32, ptr %pcap, align 4
  %capb = icmp eq i32 %capv, 0
  %ct1 = and i1 %ft, %insib
  %ct2 = and i1 %ct1, %capb
  br i1 %ct2, label %fexpect, label %floop
fexpect:
  store i32 1, ptr %pexp, align 4
  br label %floop
fchktext:
  %fistext = icmp eq i32 %fty, 3
  br i1 %fistext, label %ftext, label %fchkend
ftext:
  %tinsi = load i32, ptr %pinsi, align 4
  %tinsib = icmp ne i32 %tinsi, 0
  %texp = load i32, ptr %pexp, align 4
  %texpb = icmp ne i32 %texp, 0
  %tcap = load i32, ptr %pcap, align 4
  %tcapb = icmp eq i32 %tcap, 0
  %tt1 = and i1 %tinsib, %texpb
  %tt2 = and i1 %tt1, %tcapb
  br i1 %tt2, label %fcapture, label %floop
fcapture:
  store i64 %fnoff, ptr %poff, align 8
  store i64 %fnlen, ptr %plen2, align 8
  store i32 1, ptr %pcap, align 4
  store i32 0, ptr %pexp, align 4
  br label %floop
fchkend:
  %fisend = icmp eq i32 %fty, 1
  br i1 %fisend, label %fend, label %floop
fend:
  %fesi = call i1 @xlsx_slice_eq(ptr %sb, i64 %fnoff, i64 %fnlen, ptr @x.si, i64 2)
  %einsi = load i32, ptr %pinsi, align 4
  %einsib = icmp ne i32 %einsi, 0
  %eboth = and i1 %fesi, %einsib
  br i1 %eboth, label %frecord, label %floop
frecord:
  %ridx = load i64, ptr %pidx, align 8
  %roff = load i64, ptr %poff, align 8
  %rlen = load i64, ptr %plen2, align 8
  %pairbase = mul i64 %ridx, 16
  %offslot = getelementptr inbounds i8, ptr %arr, i64 %pairbase
  store i64 %roff, ptr %offslot, align 8
  %lenoff = add i64 %pairbase, 8
  %lenslot = getelementptr inbounds i8, ptr %arr, i64 %lenoff
  store i64 %rlen, ptr %lenslot, align 8
  %ridx1 = add i64 %ridx, 1
  store i64 %ridx1, ptr %pidx, align 8
  store i32 0, ptr %pinsi, align 4
  br label %floop
fdone:
  ret i32 0
}

; ======================================================================= open
define i32 @universe_docparse_xlsx_open(ptr %buf, i64 %len, ptr %wb) #1 {
entry:
  %bn = icmp eq ptr %buf, null
  %wn = icmp eq ptr %wb, null
  %anull = or i1 %bn, %wn
  br i1 %anull, label %ret_null, label %zero
ret_null:
  ret i32 1
zero:
  %z32 = getelementptr inbounds i8, ptr %wb, i64 32
  store ptr null, ptr %z32, align 8
  %z40 = getelementptr inbounds i8, ptr %wb, i64 40
  store i64 0, ptr %z40, align 8
  %z48 = getelementptr inbounds i8, ptr %wb, i64 48
  store ptr null, ptr %z48, align 8
  %z56 = getelementptr inbounds i8, ptr %wb, i64 56
  store i64 0, ptr %z56, align 8
  %z64 = getelementptr inbounds i8, ptr %wb, i64 64
  store ptr null, ptr %z64, align 8
  %z72 = getelementptr inbounds i8, ptr %wb, i64 72
  store i64 0, ptr %z72, align 8
  %zr = call i32 @universe_compress_zip_open(ptr %buf, i64 %len, ptr %wb)
  %zrok = icmp eq i32 %zr, 0
  br i1 %zrok, label %do_shared, label %ret_parse
ret_parse:
  ret i32 13
do_shared:
  %sbp = getelementptr inbounds i8, ptr %wb, i64 32
  %slp = getelementptr inbounds i8, ptr %wb, i64 40
  %fs = call i32 @xlsx_find(ptr %wb, ptr @x.shared, i64 20, ptr %sbp, ptr %slp)
  %isoom1 = icmp eq i32 %fs, 2
  br i1 %isoom1, label %ret_oom, label %do_sheet
ret_oom:
  ret i32 2
do_sheet:
  %shbp = getelementptr inbounds i8, ptr %wb, i64 48
  %shlp = getelementptr inbounds i8, ptr %wb, i64 56
  %fh = call i32 @xlsx_find(ptr %wb, ptr @x.sheet, i64 24, ptr %shbp, ptr %shlp)
  %fhok = icmp eq i32 %fh, 0
  br i1 %fhok, label %do_sst, label %sheet_fail
sheet_fail:
  ret i32 5
do_sst:
  %bs = call i32 @xlsx_build_sst(ptr %wb)
  %bsok = icmp eq i32 %bs, 0
  br i1 %bsok, label %ok, label %ret_bs
ret_bs:
  ret i32 %bs
ok:
  ret i32 0
}

; ====================================================================== close
define void @universe_docparse_xlsx_close(ptr %wb) #1 {
entry:
  %n = icmp eq ptr %wb, null
  br i1 %n, label %ret, label %free_shared
free_shared:
  %sbp = getelementptr inbounds i8, ptr %wb, i64 32
  %sb = load ptr, ptr %sbp, align 8
  call void @free(ptr %sb)
  store ptr null, ptr %sbp, align 8
  %shp = getelementptr inbounds i8, ptr %wb, i64 48
  %sh = load ptr, ptr %shp, align 8
  call void @free(ptr %sh)
  store ptr null, ptr %shp, align 8
  %ap = getelementptr inbounds i8, ptr %wb, i64 64
  %a = load ptr, ptr %ap, align 8
  call void @free(ptr %a)
  store ptr null, ptr %ap, align 8
  ret void
ret:
  ret void
}

; ================================================================= sst_count
define i64 @universe_docparse_xlsx_sst_count(ptr %wb) #1 {
entry:
  %n = icmp eq ptr %wb, null
  br i1 %n, label %z, label %ld
z:
  ret i64 0
ld:
  %cp = getelementptr inbounds i8, ptr %wb, i64 72
  %c = load i64, ptr %cp, align 8
  ret i64 %c
}

; =================================================================== sst_get
define i32 @universe_docparse_xlsx_sst_get(ptr %wb, i64 %idx, ptr %out) #1 {
entry:
  %wn = icmp eq ptr %wb, null
  %on = icmp eq ptr %out, null
  %an = or i1 %wn, %on
  br i1 %an, label %ret_arg, label %ld
ret_arg:
  ret i32 -8
ld:
  %cp = getelementptr inbounds i8, ptr %wb, i64 72
  %cnt = load i64, ptr %cp, align 8
  %oob = icmp uge i64 %idx, %cnt
  br i1 %oob, label %ret_idx, label %get
ret_idx:
  ret i32 7
get:
  %ap = getelementptr inbounds i8, ptr %wb, i64 64
  %arr = load ptr, ptr %ap, align 8
  %sbp = getelementptr inbounds i8, ptr %wb, i64 32
  %sb = load ptr, ptr %sbp, align 8
  %pairbase = mul i64 %idx, 16
  %offslot = getelementptr inbounds i8, ptr %arr, i64 %pairbase
  %off = load i64, ptr %offslot, align 8
  %lenoff = add i64 %pairbase, 8
  %lenslot = getelementptr inbounds i8, ptr %arr, i64 %lenoff
  %slen = load i64, ptr %lenslot, align 8
  %vptr = getelementptr inbounds i8, ptr %sb, i64 %off
  store ptr %vptr, ptr %out, align 8
  %o8 = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %slen, ptr %o8, align 8
  ret i32 0
}

; ================================================================ sheet_init
define void @universe_docparse_xlsx_sheet_init(ptr %wb, ptr %cur) #1 {
entry:
  %wn = icmp eq ptr %wb, null
  %cn = icmp eq ptr %cur, null
  %an = or i1 %wn, %cn
  br i1 %an, label %ret, label %init
init:
  %shp = getelementptr inbounds i8, ptr %wb, i64 48
  %sh = load ptr, ptr %shp, align 8
  %slp = getelementptr inbounds i8, ptr %wb, i64 56
  %sl = load i64, ptr %slp, align 8
  call void @universe_docparse_xml_init(ptr %cur, ptr %sh, i64 %sl)
  %rp = getelementptr inbounds i8, ptr %cur, i64 24
  store i64 0, ptr %rp, align 8
  ret void
ret:
  ret void
}

; ================================================================= cell_next
define i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell) #1 {
entry:
  %wn = icmp eq ptr %wb, null
  %cn = icmp eq ptr %cur, null
  %en = icmp eq ptr %cell, null
  %n0 = or i1 %wn, %cn
  %anull = or i1 %n0, %en
  br i1 %anull, label %ret_arg, label %setup
ret_arg:
  ret i32 -8
setup:
  %tok = alloca [40 x i8], align 8
  %attr = alloca [32 x i8], align 8
  %acur = alloca i64, align 8
  %pcol = alloca i64, align 8
  %prow = alloca i64, align 8
  %pctype = alloca i32, align 8
  %pvoff = alloca i64, align 8
  %pvlen = alloca i64, align 8
  %phave = alloca i32, align 8
  %pexpect = alloca i32, align 8
  %shp = getelementptr inbounds i8, ptr %wb, i64 48
  %sheet = load ptr, ptr %shp, align 8
  %rowp = getelementptr inbounds i8, ptr %cur, i64 24
  br label %outer
outer:
  %orc = call i32 @universe_docparse_xml_next(ptr %cur, ptr %tok)
  %oty = load i32, ptr %tok, align 4
  %oeof = icmp eq i32 %oty, 4
  br i1 %oeof, label %ret_done, label %obody
ret_done:
  ret i32 0
obody:
  %onop = getelementptr inbounds i8, ptr %tok, i64 8
  %onoff = load i64, ptr %onop, align 8
  %onlp = getelementptr inbounds i8, ptr %tok, i64 16
  %onlen = load i64, ptr %onlp, align 8
  %oaop = getelementptr inbounds i8, ptr %tok, i64 24
  %oaoff = load i64, ptr %oaop, align 8
  %oalp = getelementptr inbounds i8, ptr %tok, i64 32
  %oalen = load i64, ptr %oalp, align 8
  %oisstart = icmp eq i32 %oty, 0
  br i1 %oisstart, label %ostart, label %oselfchk
ostart:
  %isrow = call i1 @xlsx_slice_eq(ptr %sheet, i64 %onoff, i64 %onlen, ptr @x.row, i64 3)
  br i1 %isrow, label %dorow, label %ostart_c
dorow:
  ; parse r="N" attribute of the row
  store i64 %oaoff, ptr %acur, align 8
  %rowend = add i64 %oaoff, %oalen
  br label %row.attrloop
row.attrloop:
  %rar = call i32 @universe_docparse_xml_attr_next(ptr %sheet, ptr %acur, i64 %rowend, ptr %attr)
  %rarok = icmp eq i32 %rar, 1
  br i1 %rarok, label %row.attrchk, label %outer
row.attrchk:
  %ranoff = load i64, ptr %attr, align 8
  %ranlp = getelementptr inbounds i8, ptr %attr, i64 8
  %ranlen = load i64, ptr %ranlp, align 8
  %risr = call i1 @xlsx_slice_eq(ptr %sheet, i64 %ranoff, i64 %ranlen, ptr @x.r, i64 1)
  br i1 %risr, label %row.setr, label %row.attrloop
row.setr:
  %ravop = getelementptr inbounds i8, ptr %attr, i64 16
  %ravoff = load i64, ptr %ravop, align 8
  %ravlp = getelementptr inbounds i8, ptr %attr, i64 24
  %ravlen = load i64, ptr %ravlp, align 8
  %rvptr = getelementptr inbounds i8, ptr %sheet, i64 %ravoff
  %rownum = call i64 @xlsx_atoi(ptr %rvptr, i64 %ravlen)
  store i64 %rownum, ptr %rowp, align 8
  br label %outer
ostart_c:
  %isc = call i1 @xlsx_slice_eq(ptr %sheet, i64 %onoff, i64 %onlen, ptr @x.c, i64 1)
  br i1 %isc, label %cell.begin, label %outer
oselfchk:
  %oisself = icmp eq i32 %oty, 2
  br i1 %oisself, label %oselfc, label %outer
oselfc:
  %isselfc = call i1 @xlsx_slice_eq(ptr %sheet, i64 %onoff, i64 %onlen, ptr @x.c, i64 1)
  br i1 %isselfc, label %cell.empty, label %outer

; ---- begin a <c> cell: parse r + t attributes ----
cell.begin:
  %currow = load i64, ptr %rowp, align 8
  store i64 0, ptr %pcol, align 8
  store i64 %currow, ptr %prow, align 8
  store i32 0, ptr %pctype, align 4
  store i64 %oaoff, ptr %acur, align 8
  %cattrend = add i64 %oaoff, %oalen
  br label %c.attrloop
c.attrloop:
  %car = call i32 @universe_docparse_xml_attr_next(ptr %sheet, ptr %acur, i64 %cattrend, ptr %attr)
  %carok = icmp eq i32 %car, 1
  br i1 %carok, label %c.attrchk, label %cell.readval
c.attrchk:
  %canoff = load i64, ptr %attr, align 8
  %canlp = getelementptr inbounds i8, ptr %attr, i64 8
  %canlen = load i64, ptr %canlp, align 8
  %cavop = getelementptr inbounds i8, ptr %attr, i64 16
  %cavoff = load i64, ptr %cavop, align 8
  %cavlp = getelementptr inbounds i8, ptr %attr, i64 24
  %cavlen = load i64, ptr %cavlp, align 8
  %cisr = call i1 @xlsx_slice_eq(ptr %sheet, i64 %canoff, i64 %canlen, ptr @x.r, i64 1)
  br i1 %cisr, label %c.setref, label %c.chkt
c.setref:
  %crefp = getelementptr inbounds i8, ptr %sheet, i64 %cavoff
  call void @xlsx_parse_ref(ptr %crefp, i64 %cavlen, ptr %pcol, ptr %prow)
  br label %c.attrloop
c.chkt:
  %cist = call i1 @xlsx_slice_eq(ptr %sheet, i64 %canoff, i64 %canlen, ptr @x.t, i64 1)
  br i1 %cist, label %c.sett, label %c.attrloop
c.sett:
  %cts = call i1 @xlsx_slice_eq(ptr %sheet, i64 %cavoff, i64 %cavlen, ptr @x.sv, i64 1)
  %ctb = call i1 @xlsx_slice_eq(ptr %sheet, i64 %cavoff, i64 %cavlen, ptr @x.bv, i64 1)
  %ctinl = call i1 @xlsx_slice_eq(ptr %sheet, i64 %cavoff, i64 %cavlen, ptr @x.inl, i64 9)
  %ctstr = call i1 @xlsx_slice_eq(ptr %sheet, i64 %cavoff, i64 %cavlen, ptr @x.strv, i64 3)
  %tstr = or i1 %ctinl, %ctstr
  ; 0 number,1 shared,2 bool,3 string
  %tv0 = select i1 %tstr, i32 3, i32 0
  %tv1 = select i1 %ctb, i32 2, i32 %tv0
  %tv2 = select i1 %cts, i32 1, i32 %tv1
  store i32 %tv2, ptr %pctype, align 4
  br label %c.attrloop

; ---- read inner value tokens up to </c> ----
cell.readval:
  store i32 0, ptr %phave, align 4
  store i32 0, ptr %pexpect, align 4
  store i64 0, ptr %pvoff, align 8
  store i64 0, ptr %pvlen, align 8
  br label %iloop
iloop:
  %irc = call i32 @universe_docparse_xml_next(ptr %cur, ptr %tok)
  %ity = load i32, ptr %tok, align 4
  %ieof = icmp eq i32 %ity, 4
  br i1 %ieof, label %cell.resolve, label %ibody
ibody:
  %inop = getelementptr inbounds i8, ptr %tok, i64 8
  %inoff = load i64, ptr %inop, align 8
  %inlp = getelementptr inbounds i8, ptr %tok, i64 16
  %inlen = load i64, ptr %inlp, align 8
  %iisend = icmp eq i32 %ity, 1
  br i1 %iisend, label %ichkendc, label %iisstartchk
ichkendc:
  %iendc = call i1 @xlsx_slice_eq(ptr %sheet, i64 %inoff, i64 %inlen, ptr @x.c, i64 1)
  br i1 %iendc, label %cell.resolve, label %iloop
iisstartchk:
  %iisstart = icmp eq i32 %ity, 0
  br i1 %iisstart, label %istart, label %ichktext
istart:
  %isv = call i1 @xlsx_slice_eq(ptr %sheet, i64 %inoff, i64 %inlen, ptr @x.v, i64 1)
  %ist = call i1 @xlsx_slice_eq(ptr %sheet, i64 %inoff, i64 %inlen, ptr @x.t, i64 1)
  %isvt = or i1 %isv, %ist
  br i1 %isvt, label %iexpect, label %iloop
iexpect:
  store i32 1, ptr %pexpect, align 4
  br label %iloop
ichktext:
  %iistext = icmp eq i32 %ity, 3
  br i1 %iistext, label %itext, label %iloop
itext:
  %iexp = load i32, ptr %pexpect, align 4
  %iexpb = icmp ne i32 %iexp, 0
  %ihv = load i32, ptr %phave, align 4
  %ihvb = icmp eq i32 %ihv, 0
  %icap = and i1 %iexpb, %ihvb
  br i1 %icap, label %icapture, label %iloop
icapture:
  store i64 %inoff, ptr %pvoff, align 8
  store i64 %inlen, ptr %pvlen, align 8
  store i32 1, ptr %phave, align 4
  store i32 0, ptr %pexpect, align 4
  br label %iloop

; ---- resolve captured value into the cell ----
cell.resolve:
  %rcol = load i64, ptr %pcol, align 8
  %rrow = load i64, ptr %prow, align 8
  %rctype = load i32, ptr %pctype, align 4
  %rhave = load i32, ptr %phave, align 4
  %rvoff = load i64, ptr %pvoff, align 8
  %rvlen = load i64, ptr %pvlen, align 8
  ; store row/col
  store i64 %rrow, ptr %cell, align 8
  %ccolp = getelementptr inbounds i8, ptr %cell, i64 8
  store i64 %rcol, ptr %ccolp, align 8
  %chavb = icmp eq i32 %rhave, 0
  br i1 %chavb, label %res.empty, label %res.switch
res.empty:
  %ctp0 = getelementptr inbounds i8, ptr %cell, i64 16
  store i32 4, ptr %ctp0, align 4
  %cvp0 = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr null, ptr %cvp0, align 8
  %cvl0 = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 0, ptr %cvl0, align 8
  %cnm0 = getelementptr inbounds i8, ptr %cell, i64 40
  store double 0.0, ptr %cnm0, align 8
  %csx0 = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 0, ptr %csx0, align 8
  ret i32 1
res.switch:
  %valp = getelementptr inbounds i8, ptr %sheet, i64 %rvoff
  %isshared = icmp eq i32 %rctype, 1
  br i1 %isshared, label %res.shared, label %res.notshared
res.shared:
  %sidx = call i64 @xlsx_atoi(ptr %valp, i64 %rvlen)
  %sstcp = getelementptr inbounds i8, ptr %wb, i64 72
  %sstc = load i64, ptr %sstcp, align 8
  %sidxok = icmp ult i64 %sidx, %sstc
  br i1 %sidxok, label %res.sharedok, label %res.sharedbad
res.sharedok:
  %arrp = getelementptr inbounds i8, ptr %wb, i64 64
  %arr = load ptr, ptr %arrp, align 8
  %sbp2 = getelementptr inbounds i8, ptr %wb, i64 32
  %sb2 = load ptr, ptr %sbp2, align 8
  %pb = mul i64 %sidx, 16
  %offs = getelementptr inbounds i8, ptr %arr, i64 %pb
  %soff = load i64, ptr %offs, align 8
  %lenoff2 = add i64 %pb, 8
  %lens = getelementptr inbounds i8, ptr %arr, i64 %lenoff2
  %slen = load i64, ptr %lens, align 8
  %sptr = getelementptr inbounds i8, ptr %sb2, i64 %soff
  %ctp1 = getelementptr inbounds i8, ptr %cell, i64 16
  store i32 1, ptr %ctp1, align 4
  %cvp1 = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr %sptr, ptr %cvp1, align 8
  %cvl1 = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 %slen, ptr %cvl1, align 8
  %cnm1 = getelementptr inbounds i8, ptr %cell, i64 40
  store double 0.0, ptr %cnm1, align 8
  %csx1 = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 %sidx, ptr %csx1, align 8
  ret i32 1
res.sharedbad:
  %ctp2 = getelementptr inbounds i8, ptr %cell, i64 16
  store i32 1, ptr %ctp2, align 4
  %cvp2 = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr null, ptr %cvp2, align 8
  %cvl2 = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 0, ptr %cvl2, align 8
  %cnm2 = getelementptr inbounds i8, ptr %cell, i64 40
  store double 0.0, ptr %cnm2, align 8
  %csx2 = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 %sidx, ptr %csx2, align 8
  ret i32 1
res.notshared:
  ; string/inline (type 3): raw slice, no numeric
  %isstr = icmp eq i32 %rctype, 3
  br i1 %isstr, label %res.str, label %res.numeric
res.str:
  %ctp3 = getelementptr inbounds i8, ptr %cell, i64 16
  store i32 3, ptr %ctp3, align 4
  %cvp3 = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr %valp, ptr %cvp3, align 8
  %cvl3 = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 %rvlen, ptr %cvl3, align 8
  %cnm3 = getelementptr inbounds i8, ptr %cell, i64 40
  store double 0.0, ptr %cnm3, align 8
  %csx3 = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 0, ptr %csx3, align 8
  ret i32 1
res.numeric:
  ; number (0) or boolean (2)
  %isbool = icmp eq i32 %rctype, 2
  %num = call double @xlsx_atof(ptr %valp, i64 %rvlen)
  %ctp4 = getelementptr inbounds i8, ptr %cell, i64 16
  %tystore = select i1 %isbool, i32 2, i32 0
  store i32 %tystore, ptr %ctp4, align 4
  %cvp4 = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr %valp, ptr %cvp4, align 8
  %cvl4 = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 %rvlen, ptr %cvl4, align 8
  %cnm4 = getelementptr inbounds i8, ptr %cell, i64 40
  store double %num, ptr %cnm4, align 8
  %csx4 = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 0, ptr %csx4, align 8
  ret i32 1

; ---- empty self-closed cell ----
cell.empty:
  store i64 0, ptr %pcol, align 8
  %ecurrow = load i64, ptr %rowp, align 8
  store i64 %ecurrow, ptr %prow, align 8
  ; parse r attribute for col/row
  store i64 %oaoff, ptr %acur, align 8
  %ecattrend = add i64 %oaoff, %oalen
  br label %e.attrloop
e.attrloop:
  %ear = call i32 @universe_docparse_xml_attr_next(ptr %sheet, ptr %acur, i64 %ecattrend, ptr %attr)
  %earok = icmp eq i32 %ear, 1
  br i1 %earok, label %e.attrchk, label %e.finish
e.attrchk:
  %eanoff = load i64, ptr %attr, align 8
  %eanlp = getelementptr inbounds i8, ptr %attr, i64 8
  %eanlen = load i64, ptr %eanlp, align 8
  %eisr = call i1 @xlsx_slice_eq(ptr %sheet, i64 %eanoff, i64 %eanlen, ptr @x.r, i64 1)
  br i1 %eisr, label %e.setref, label %e.attrloop
e.setref:
  %eavop = getelementptr inbounds i8, ptr %attr, i64 16
  %eavoff = load i64, ptr %eavop, align 8
  %eavlp = getelementptr inbounds i8, ptr %attr, i64 24
  %eavlen = load i64, ptr %eavlp, align 8
  %erefp = getelementptr inbounds i8, ptr %sheet, i64 %eavoff
  call void @xlsx_parse_ref(ptr %erefp, i64 %eavlen, ptr %pcol, ptr %prow)
  br label %e.attrloop
e.finish:
  %efcol = load i64, ptr %pcol, align 8
  %efrow = load i64, ptr %prow, align 8
  store i64 %efrow, ptr %cell, align 8
  %efcolp = getelementptr inbounds i8, ptr %cell, i64 8
  store i64 %efcol, ptr %efcolp, align 8
  %eftp = getelementptr inbounds i8, ptr %cell, i64 16
  store i32 4, ptr %eftp, align 4
  %efvp = getelementptr inbounds i8, ptr %cell, i64 24
  store ptr null, ptr %efvp, align 8
  %efvl = getelementptr inbounds i8, ptr %cell, i64 32
  store i64 0, ptr %efvl, align 8
  %efnm = getelementptr inbounds i8, ptr %cell, i64 40
  store double 0.0, ptr %efnm, align 8
  %efsx = getelementptr inbounds i8, ptr %cell, i64 48
  store i64 0, ptr %efsx, align 8
  ret i32 1
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind }
attributes #6 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #7 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
