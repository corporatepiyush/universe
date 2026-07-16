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

; Tests for universe_parse_csv_*: a fixed CSV document cross-checked field by
; field (quoted comma, doubled quote, CRLF, LF, empty/trailing fields), a
; quoted embedded-newline field, TSV via a tab delimiter, unquote, next_record,
; and a --bench over a ~64 KiB CSV.

declare void @universe_parse_csv_init(ptr, ptr, i64, i32)
declare i32  @universe_parse_csv_next_field(ptr, ptr)
declare i32  @universe_parse_csv_next_record(ptr, ptr, i64, ptr)
declare i64  @universe_parse_csv_unquote(ptr, ptr, i64)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

; fixed CSV doc: a,bb,ccc<CRLF>"q,w","d""e",<LF>,x
@doc = private unnamed_addr constant [26 x i8] c"a,bb,ccc\0D\0A\22q,w\22,\22d\22\22e\22,\0A,x"
; quoted field with an embedded newline: "line1<LF>line2",b
@doc2 = private unnamed_addr constant [15 x i8] c"\22line1\0Aline2\22,b"
; TSV: p<TAB>q<TAB>r<LF>s<TAB>t
@tsv = private unnamed_addr constant [9 x i8] c"p\09q\09r\0As\09t"

@exp.len  = private unnamed_addr constant [8 x i64] [ i64 1, i64 2, i64 3, i64 3, i64 4, i64 0, i64 0, i64 1 ]
@exp.unq  = private unnamed_addr constant [8 x i32] [ i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0 ]
@exp.stat = private unnamed_addr constant [8 x i32] [ i32 0, i32 0, i32 1, i32 0, i32 0, i32 1, i32 0, i32 1 ]

; expected content for the two quoted fields (RAW, between-quotes)
@raw.qw   = private unnamed_addr constant [3 x i8] c"q,w"
@raw.de   = private unnamed_addr constant [4 x i8] c"d\22\22e"
@unq.de   = private unnamed_addr constant [3 x i8] c"d\22e"
@raw.nl   = private unnamed_addr constant [11 x i8] c"line1\0Aline2"

@g.len  = internal global [32 x i64] zeroinitializer, align 16
@g.off  = internal global [32 x i64] zeroinitializer, align 16
@g.unq  = internal global [32 x i32] zeroinitializer, align 16
@g.stat = internal global [32 x i32] zeroinitializer, align 16
@g.dst  = internal global [64 x i8] zeroinitializer, align 16
@g.rec  = internal global [16 x { i64, i64, i32, i32 }] zeroinitializer, align 16
@g.csv  = internal global [65536 x i8] zeroinitializer, align 16

@m.count = private unnamed_addr constant [17 x i8] c"csv field count\0A\00"
@m.lens  = private unnamed_addr constant [16 x i8] c"csv field lens\0A\00"
@m.unqs  = private unnamed_addr constant [16 x i8] c"csv unq flags\0A\00\00"
@m.stats = private unnamed_addr constant [16 x i8] c"csv rec status\0A\00"
@m.qw    = private unnamed_addr constant [15 x i8] c"quoted comma\0A\00\00"
@m.deraw = private unnamed_addr constant [16 x i8] c"dbl quote raw\0A\00\00"
@m.deunq = private unnamed_addr constant [17 x i8] c"dbl quote unq\0A\00\00\00"
@m.nllen = private unnamed_addr constant [21 x i8] c"embedded newline len\00"
@m.nlval = private unnamed_addr constant [21 x i8] c"embedded newline val\00"
@m.tsvc  = private unnamed_addr constant [16 x i8] c"tsv field count\00"
@m.tsvv  = private unnamed_addr constant [15 x i8] c"tsv first val\0A\00"
@m.reccnt = private unnamed_addr constant [17 x i8] c"next_record cnt\0A\00"
@m.receof = private unnamed_addr constant [17 x i8] c"next_record eof\0A\00"
@csv.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.csv = private unnamed_addr constant [22 x i8] c"csv parse 3-field row\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %sc = alloca [32 x i8], align 8
  %fld = alloca [24 x i8], align 8

  ; ---- collect all fields of the fixed doc ----
  call void @universe_parse_csv_init(ptr %sc, ptr @doc, i64 26, i32 44)
  br label %col.head
col.head:
  %ci = phi i64 [ 0, %entry ], [ %ci.next, %col.body ]
  %st = call i32 @universe_parse_csv_next_field(ptr %sc, ptr %fld)
  %iseof = icmp eq i32 %st, 2
  br i1 %iseof, label %col.done, label %col.body
col.body:
  %o.p = getelementptr inbounds nuw i8, ptr %fld, i64 0
  %o.v = load i64, ptr %o.p, align 8
  %l.p = getelementptr inbounds nuw i8, ptr %fld, i64 8
  %l.v = load i64, ptr %l.p, align 8
  %u.p = getelementptr inbounds nuw i8, ptr %fld, i64 16
  %u.v = load i32, ptr %u.p, align 4
  %go = getelementptr inbounds nuw [32 x i64], ptr @g.off, i64 0, i64 %ci
  store i64 %o.v, ptr %go, align 8
  %gl = getelementptr inbounds nuw [32 x i64], ptr @g.len, i64 0, i64 %ci
  store i64 %l.v, ptr %gl, align 8
  %gu = getelementptr inbounds nuw [32 x i32], ptr @g.unq, i64 0, i64 %ci
  store i32 %u.v, ptr %gu, align 4
  %gs = getelementptr inbounds nuw [32 x i32], ptr @g.stat, i64 0, i64 %ci
  store i32 %st, ptr %gs, align 4
  %ci.next = add nuw i64 %ci, 1
  br label %col.head
col.done:
  call void @ut_check_eq(i64 %ci, i64 8, ptr @m.count)

  ; compare len/unq/stat arrays
  br label %cmp.head
cmp.head:
  %mi = phi i64 [ 0, %col.done ], [ %mi.next, %cmp.body ]
  %mlen = phi i64 [ 0, %col.done ], [ %mlen.n, %cmp.body ]
  %munq = phi i64 [ 0, %col.done ], [ %munq.n, %cmp.body ]
  %mst  = phi i64 [ 0, %col.done ], [ %mst.n, %cmp.body ]
  %mdone = icmp uge i64 %mi, 8
  br i1 %mdone, label %cmp.fin, label %cmp.body
cmp.body:
  %el.p = getelementptr inbounds nuw [8 x i64], ptr @exp.len, i64 0, i64 %mi
  %el = load i64, ptr %el.p, align 8
  %gl2 = getelementptr inbounds nuw [32 x i64], ptr @g.len, i64 0, i64 %mi
  %gl2v = load i64, ptr %gl2, align 8
  %lne = icmp ne i64 %el, %gl2v
  %lneb = zext i1 %lne to i64
  %mlen.n = add i64 %mlen, %lneb
  %eu.p = getelementptr inbounds nuw [8 x i32], ptr @exp.unq, i64 0, i64 %mi
  %eu = load i32, ptr %eu.p, align 4
  %gu2 = getelementptr inbounds nuw [32 x i32], ptr @g.unq, i64 0, i64 %mi
  %gu2v = load i32, ptr %gu2, align 4
  %une = icmp ne i32 %eu, %gu2v
  %uneb = zext i1 %une to i64
  %munq.n = add i64 %munq, %uneb
  %es.p = getelementptr inbounds nuw [8 x i32], ptr @exp.stat, i64 0, i64 %mi
  %es = load i32, ptr %es.p, align 4
  %gs2 = getelementptr inbounds nuw [32 x i32], ptr @g.stat, i64 0, i64 %mi
  %gs2v = load i32, ptr %gs2, align 4
  %sne = icmp ne i32 %es, %gs2v
  %sneb = zext i1 %sne to i64
  %mst.n = add i64 %mst, %sneb
  %mi.next = add nuw i64 %mi, 1
  br label %cmp.head
cmp.fin:
  call void @ut_check_eq(i64 %mlen, i64 0, ptr @m.lens)
  call void @ut_check_eq(i64 %munq, i64 0, ptr @m.unqs)
  call void @ut_check_eq(i64 %mst, i64 0, ptr @m.stats)

  ; ---- content of quoted fields (index 3 = "q,w", index 4 = d""e) ----
  %o3 = getelementptr inbounds nuw [32 x i64], ptr @g.off, i64 0, i64 3
  %o3v = load i64, ptr %o3, align 8
  %p3 = getelementptr inbounds nuw i8, ptr @doc, i64 %o3v
  %qw.cmp = call i32 @memcmp(ptr %p3, ptr @raw.qw, i64 3)
  %qw.ok = icmp eq i32 %qw.cmp, 0
  call void @ut_check(i1 %qw.ok, ptr @m.qw)

  %o4 = getelementptr inbounds nuw [32 x i64], ptr @g.off, i64 0, i64 4
  %o4v = load i64, ptr %o4, align 8
  %p4 = getelementptr inbounds nuw i8, ptr @doc, i64 %o4v
  %de.cmp = call i32 @memcmp(ptr %p4, ptr @raw.de, i64 4)
  %de.ok = icmp eq i32 %de.cmp, 0
  call void @ut_check(i1 %de.ok, ptr @m.deraw)

  ; unquote the d""e field -> d"e (len 3)
  %uq = call i64 @universe_parse_csv_unquote(ptr @g.dst, ptr %p4, i64 4)
  %uq.len = icmp eq i64 %uq, 3
  %uq.cmp = call i32 @memcmp(ptr @g.dst, ptr @unq.de, i64 3)
  %uq.cok = icmp eq i32 %uq.cmp, 0
  %uq.ok = and i1 %uq.len, %uq.cok
  call void @ut_check(i1 %uq.ok, ptr @m.deunq)

  ; ---- embedded-newline quoted field ----
  call void @universe_parse_csv_init(ptr %sc, ptr @doc2, i64 15, i32 44)
  %nl.st = call i32 @universe_parse_csv_next_field(ptr %sc, ptr %fld)
  %nl.op = getelementptr inbounds nuw i8, ptr %fld, i64 0
  %nl.o = load i64, ptr %nl.op, align 8
  %nl.lp = getelementptr inbounds nuw i8, ptr %fld, i64 8
  %nl.l = load i64, ptr %nl.lp, align 8
  %nl.lenok = icmp eq i64 %nl.l, 11
  call void @ut_check(i1 %nl.lenok, ptr @m.nllen)
  %nl.p = getelementptr inbounds nuw i8, ptr @doc2, i64 %nl.o
  %nl.cmp = call i32 @memcmp(ptr %nl.p, ptr @raw.nl, i64 11)
  %nl.vok = icmp eq i32 %nl.cmp, 0
  call void @ut_check(i1 %nl.vok, ptr @m.nlval)

  ; ---- TSV via tab delimiter ----
  call void @universe_parse_csv_init(ptr %sc, ptr @tsv, i64 9, i32 9)
  br label %t.head
t.head:
  %ti = phi i64 [ 0, %cmp.fin ], [ %ti.next, %t.body ]
  %tfirst = phi i64 [ 0, %cmp.fin ], [ %tf.upd, %t.body ]
  %tst = call i32 @universe_parse_csv_next_field(ptr %sc, ptr %fld)
  %teof = icmp eq i32 %tst, 2
  br i1 %teof, label %t.done, label %t.body
t.body:
  %tf0 = icmp eq i64 %ti, 0
  %tlp = getelementptr inbounds nuw i8, ptr %fld, i64 8
  %tlv = load i64, ptr %tlp, align 8
  %tf.upd = select i1 %tf0, i64 %tlv, i64 %tfirst
  %ti.next = add nuw i64 %ti, 1
  br label %t.head
t.done:
  call void @ut_check_eq(i64 %ti, i64 5, ptr @m.tsvc)
  call void @ut_check_eq(i64 %tfirst, i64 1, ptr @m.tsvv)

  ; ---- next_record over the fixed doc: first record has 3 fields ----
  call void @universe_parse_csv_init(ptr %sc, ptr @doc, i64 26, i32 44)
  %cntp = alloca i64, align 8
  %rr = call i32 @universe_parse_csv_next_record(ptr %sc, ptr @g.rec, i64 16, ptr %cntp)
  %rcnt = load i64, ptr %cntp, align 8
  %rr.ok = icmp eq i32 %rr, 0
  %rcnt.ok = icmp eq i64 %rcnt, 3
  %rec.ok = and i1 %rr.ok, %rcnt.ok
  call void @ut_check(i1 %rec.ok, ptr @m.reccnt)
  ; drain remaining records (R2, R3) then expect EOF
  %rr2 = call i32 @universe_parse_csv_next_record(ptr %sc, ptr @g.rec, i64 16, ptr %cntp)
  %rr3 = call i32 @universe_parse_csv_next_record(ptr %sc, ptr @g.rec, i64 16, ptr %cntp)
  %rr4 = call i32 @universe_parse_csv_next_record(ptr %sc, ptr @g.rec, i64 16, ptr %cntp)
  %rr4.eof = icmp eq i32 %rr4, 2
  call void @ut_check(i1 %rr4.eof, ptr @m.receof)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; fill with rows "aaa,bbb,ccc\n" (12 bytes each)
  br label %bf.head
bf.head:
  %bp = phi i64 [ 0, %bench ], [ %bp.next, %bf.body ]
  %room = add nuw i64 %bp, 12
  %fits = icmp ult i64 %room, 65536
  br i1 %fits, label %bf.body, label %bf.done
bf.body:
  %r0 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %bp
  ; "aaa,bbb,ccc\n"
  store i8 97, ptr %r0, align 1
  %q1 = add nuw i64 %bp, 1
  %r1 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q1
  store i8 97, ptr %r1, align 1
  %q2 = add nuw i64 %bp, 2
  %r2 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q2
  store i8 97, ptr %r2, align 1
  %q3 = add nuw i64 %bp, 3
  %r3 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q3
  store i8 44, ptr %r3, align 1
  %q4 = add nuw i64 %bp, 4
  %r4 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q4
  store i8 98, ptr %r4, align 1
  %q5 = add nuw i64 %bp, 5
  %r5 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q5
  store i8 98, ptr %r5, align 1
  %q6 = add nuw i64 %bp, 6
  %r6 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q6
  store i8 98, ptr %r6, align 1
  %q7 = add nuw i64 %bp, 7
  %r7 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q7
  store i8 44, ptr %r7, align 1
  %q8 = add nuw i64 %bp, 8
  %r8 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q8
  store i8 99, ptr %r8, align 1
  %q9 = add nuw i64 %bp, 9
  %r9 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q9
  store i8 99, ptr %r9, align 1
  %q10 = add nuw i64 %bp, 10
  %r10 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q10
  store i8 99, ptr %r10, align 1
  %q11 = add nuw i64 %bp, 11
  %r11 = getelementptr inbounds nuw [65536 x i8], ptr @g.csv, i64 0, i64 %q11
  store i8 10, ptr %r11, align 1
  %bp.next = add nuw i64 %bp, 12
  br label %bf.head
bf.done:
  %scb = alloca [32 x i8], align 8
  %fldb = alloca [24 x i8], align 8
  ; 17 reps of a 200-parse batch over the filled buffer; discard rep 0 (warm-up),
  ; report over the remaining 16. ops/rep = %bp * 200 bytes (ns per parsed byte).
  br label %csv.rep
csv.rep:
  %crep = phi i64 [ 0, %bf.done ], [ %crep.n, %csv.next ]
  %t0 = call double @ut_now_sec()
  br label %biter.head
biter.head:
  %it = phi i64 [ 0, %csv.rep ], [ %it.next, %biter.next ]
  %itd = icmp uge i64 %it, 200
  br i1 %itd, label %csv.rep.done, label %biter.body
biter.body:
  call void @universe_parse_csv_init(ptr %scb, ptr @g.csv, i64 %bp, i32 44)
  br label %bp.field
bp.field:
  %fst = call i32 @universe_parse_csv_next_field(ptr %scb, ptr %fldb)
  %fe = icmp eq i32 %fst, 2
  br i1 %fe, label %biter.next, label %bp.field
biter.next:
  %it.next = add nuw i64 %it, 1
  br label %biter.head
csv.rep.done:
  %t1 = call double @ut_now_sec()
  %cel = fsub double %t1, %t0
  %ckeep = icmp ugt i64 %crep, 0
  br i1 %ckeep, label %csv.store, label %csv.next
csv.store:
  %cidx = sub i64 %crep, 1
  %csp = getelementptr inbounds [16 x double], ptr @csv.samp, i64 0, i64 %cidx
  store double %cel, ptr %csp, align 8
  br label %csv.next
csv.next:
  %crep.n = add nuw i64 %crep, 1
  %cmore = icmp ult i64 %crep.n, 17
  br i1 %cmore, label %csv.rep, label %csv.report
csv.report:
  %csv.ops = mul i64 %bp, 200
  call void @ut_report_dist(ptr @csv.samp, i64 16, i64 %csv.ops, ptr @lbl.csv)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
