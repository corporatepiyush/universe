; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Test driver for the XLSX reader. Fixture .xlsx is built by Python and
; embedded via fixture_xlsx.ll (@fixture_xlsx_ptr / @fixture_xlsx_len).

declare ptr @fixture_xlsx_ptr()
declare i64 @fixture_xlsx_len()
declare ptr @fixture_big_ptr()
declare i64 @fixture_big_len()

declare i32 @universe_docparse_xlsx_open(ptr, i64, ptr)
declare void @universe_docparse_xlsx_close(ptr)
declare i64 @universe_docparse_xlsx_sst_count(ptr)
declare i32 @universe_docparse_xlsx_sst_get(ptr, i64, ptr)
declare void @universe_docparse_xlsx_sheet_init(ptr, ptr)
declare i32 @universe_docparse_xlsx_cell_next(ptr, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)

@s.hello = private constant [5 x i8] c"Hello"
@s.world = private constant [5 x i8] c"World"
@s.inl   = private constant [3 x i8] c"inl"

@m.open  = private constant [12 x i8] c"open failed\00"
@m.sstc  = private constant [10 x i8] c"sst count\00"
@m.sst0  = private constant [8 x i8] c"sst[0]\0A\00"
@m.sst1  = private constant [8 x i8] c"sst[1]\0A\00"
@m.cell  = private constant [10 x i8] c"cell val\0A\00"
@m.row   = private constant [5 x i8] c"row\0A\00"
@m.col   = private constant [5 x i8] c"col\0A\00"
@m.type  = private constant [7 x i8] c"ctype\0A\00"
@m.num   = private constant [6 x i8] c"cnum\0A\00"
@xlsx.samp = internal global [16 x double] zeroinitializer, align 8
@xlsx.cells = internal global i64 0, align 8
@lbl.xlsx = private unnamed_addr constant [18 x i8] c"xlsx cell iterate\00"

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
  %wb = alloca [80 x i8], align 8
  %cur = alloca [32 x i8], align 8
  %cell = alloca [56 x i8], align 8
  %slice = alloca [16 x i8], align 8

  %buf = call ptr @fixture_xlsx_ptr()
  %len = call i64 @fixture_xlsx_len()
  %oc = call i32 @universe_docparse_xlsx_open(ptr %buf, i64 %len, ptr %wb)
  %ocok = icmp eq i32 %oc, 0
  call void @ut_check(i1 %ocok, ptr @m.open)

  ; shared string table
  %sc = call i64 @universe_docparse_xlsx_sst_count(ptr %wb)
  call void @ut_check_eq(i64 %sc, i64 2, ptr @m.sstc)
  %g0 = call i32 @universe_docparse_xlsx_sst_get(ptr %wb, i64 0, ptr %slice)
  %s0p = load ptr, ptr %slice, align 8
  %s0lp = getelementptr inbounds i8, ptr %slice, i64 8
  %s0l = load i64, ptr %s0lp, align 8
  %s0ok = call i1 @seq(ptr %s0p, i64 %s0l, ptr @s.hello, i64 5)
  call void @ut_check(i1 %s0ok, ptr @m.sst0)
  %g1 = call i32 @universe_docparse_xlsx_sst_get(ptr %wb, i64 1, ptr %slice)
  %s1p = load ptr, ptr %slice, align 8
  %s1lp = getelementptr inbounds i8, ptr %slice, i64 8
  %s1l = load i64, ptr %s1lp, align 8
  %s1ok = call i1 @seq(ptr %s1p, i64 %s1l, ptr @s.world, i64 5)
  call void @ut_check(i1 %s1ok, ptr @m.sst1)

  ; iterate cells
  call void @universe_docparse_xlsx_sheet_init(ptr %wb, ptr %cur)

  ; cell pointers
  %prow = getelementptr inbounds i8, ptr %cell, i64 0
  %pcol = getelementptr inbounds i8, ptr %cell, i64 8
  %ptype = getelementptr inbounds i8, ptr %cell, i64 16
  %pvptr = getelementptr inbounds i8, ptr %cell, i64 24
  %pvlen = getelementptr inbounds i8, ptr %cell, i64 32
  %pnum = getelementptr inbounds i8, ptr %cell, i64 40

  ; A1 = shared "Hello" (row1 col1 type1)
  %r1 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r1ok = icmp eq i32 %r1, 1
  call void @ut_check(i1 %r1ok, ptr @m.cell)
  %a1row = load i64, ptr %prow, align 8
  call void @ut_check_eq(i64 %a1row, i64 1, ptr @m.row)
  %a1col = load i64, ptr %pcol, align 8
  call void @ut_check_eq(i64 %a1col, i64 1, ptr @m.col)
  %a1ty = load i32, ptr %ptype, align 4
  %a1ty64 = zext i32 %a1ty to i64
  call void @ut_check_eq(i64 %a1ty64, i64 1, ptr @m.type)
  %a1vp = load ptr, ptr %pvptr, align 8
  %a1vl = load i64, ptr %pvlen, align 8
  %a1ok = call i1 @seq(ptr %a1vp, i64 %a1vl, ptr @s.hello, i64 5)
  call void @ut_check(i1 %a1ok, ptr @m.cell)

  ; B1 = number 42 (col2 type0 num42)
  %r2 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r2ok = icmp eq i32 %r2, 1
  call void @ut_check(i1 %r2ok, ptr @m.cell)
  %b1col = load i64, ptr %pcol, align 8
  call void @ut_check_eq(i64 %b1col, i64 2, ptr @m.col)
  %b1ty = load i32, ptr %ptype, align 4
  %b1ty64 = zext i32 %b1ty to i64
  call void @ut_check_eq(i64 %b1ty64, i64 0, ptr @m.type)
  %b1num = load double, ptr %pnum, align 8
  %b1i = fptosi double %b1num to i64
  call void @ut_check_eq(i64 %b1i, i64 42, ptr @m.num)

  ; C1 = bool TRUE (col3 type2 num1)
  %r3 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r3ok = icmp eq i32 %r3, 1
  call void @ut_check(i1 %r3ok, ptr @m.cell)
  %c1col = load i64, ptr %pcol, align 8
  call void @ut_check_eq(i64 %c1col, i64 3, ptr @m.col)
  %c1ty = load i32, ptr %ptype, align 4
  %c1ty64 = zext i32 %c1ty to i64
  call void @ut_check_eq(i64 %c1ty64, i64 2, ptr @m.type)
  %c1num = load double, ptr %pnum, align 8
  %c1i = fptosi double %c1num to i64
  call void @ut_check_eq(i64 %c1i, i64 1, ptr @m.num)

  ; A2 = shared "World" (row2 col1 type1)
  %r4 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r4ok = icmp eq i32 %r4, 1
  call void @ut_check(i1 %r4ok, ptr @m.cell)
  %a2row = load i64, ptr %prow, align 8
  call void @ut_check_eq(i64 %a2row, i64 2, ptr @m.row)
  %a2vp = load ptr, ptr %pvptr, align 8
  %a2vl = load i64, ptr %pvlen, align 8
  %a2ok = call i1 @seq(ptr %a2vp, i64 %a2vl, ptr @s.world, i64 5)
  call void @ut_check(i1 %a2ok, ptr @m.cell)

  ; B2 = number 3.14 (col2)
  %r5 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r5ok = icmp eq i32 %r5, 1
  call void @ut_check(i1 %r5ok, ptr @m.cell)
  %b2num = load double, ptr %pnum, align 8
  ; 3.14 * 100 rounded == 314
  %b2x = fmul double %b2num, 1.000000e+02
  %b2i = fptosi double %b2x to i64
  call void @ut_check_eq(i64 %b2i, i64 314, ptr @m.num)

  ; C2 = inline "inl" (col3 type3)
  %r6 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r6ok = icmp eq i32 %r6, 1
  call void @ut_check(i1 %r6ok, ptr @m.cell)
  %c2ty = load i32, ptr %ptype, align 4
  %c2ty64 = zext i32 %c2ty to i64
  call void @ut_check_eq(i64 %c2ty64, i64 3, ptr @m.type)
  %c2vp = load ptr, ptr %pvptr, align 8
  %c2vl = load i64, ptr %pvlen, align 8
  %c2ok = call i1 @seq(ptr %c2vp, i64 %c2vl, ptr @s.inl, i64 3)
  call void @ut_check(i1 %c2ok, ptr @m.cell)

  ; D2 = empty self-close (col4 type4)
  %r7 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r7ok = icmp eq i32 %r7, 1
  call void @ut_check(i1 %r7ok, ptr @m.cell)
  %d2col = load i64, ptr %pcol, align 8
  call void @ut_check_eq(i64 %d2col, i64 4, ptr @m.col)
  %d2ty = load i32, ptr %ptype, align 4
  %d2ty64 = zext i32 %d2ty to i64
  call void @ut_check_eq(i64 %d2ty64, i64 4, ptr @m.type)

  ; end of cells
  %r8 = call i32 @universe_docparse_xlsx_cell_next(ptr %wb, ptr %cur, ptr %cell)
  %r8end = icmp eq i32 %r8, 0
  call void @ut_check(i1 %r8end, ptr @m.cell)

  call void @universe_docparse_xlsx_close(ptr %wb)

  ; ---- bench ----
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %fin

bench:
  ; open the big (1000-row) sheet ONCE (inflate+index), then time pure
  ; cell iteration by re-initialising the cursor and walking all cells.
  %wb2 = alloca [80 x i8], align 8
  %cur2 = alloca [32 x i8], align 8
  %cell2 = alloca [56 x i8], align 8
  %bbuf = call ptr @fixture_big_ptr()
  %blen = call i64 @fixture_big_len()
  %ocb = call i32 @universe_docparse_xlsx_open(ptr %bbuf, i64 %blen, ptr %wb2)
  ; 17 reps of a 400-pass cell walk over the pre-opened sheet; discard rep 0
  ; (warm-up), report over the remaining 16. ops/rep = total cells walked
  ; (400 * cells-per-pass), captured at runtime (ns per cell).
  br label %xl.rep
xl.rep:
  %xrep = phi i64 [ 0, %bench ], [ %xrep.n, %xl.next ]
  %t0 = call double @ut_now_sec()
  br label %bloop
bloop:
  %it = phi i64 [ 0, %xl.rep ], [ %it.n, %bloop.tail ]
  %cells = phi i64 [ 0, %xl.rep ], [ %cells.n, %bloop.tail ]
  %itdone = icmp uge i64 %it, 400
  br i1 %itdone, label %xl.rep.done, label %biter
biter:
  call void @universe_docparse_xlsx_sheet_init(ptr %wb2, ptr %cur2)
  br label %cellloop
cellloop:
  %cc = phi i64 [ 0, %biter ], [ %cc.n, %cellcont ]
  %cr = call i32 @universe_docparse_xlsx_cell_next(ptr %wb2, ptr %cur2, ptr %cell2)
  %crmore = icmp eq i32 %cr, 1
  br i1 %crmore, label %cellcont, label %cellend
cellcont:
  %cc.n = add i64 %cc, 1
  br label %cellloop
cellend:
  br label %bloop.tail
bloop.tail:
  %cells.n = add i64 %cells, %cc
  %it.n = add i64 %it, 1
  br label %bloop
xl.rep.done:
  %t1 = call double @ut_now_sec()
  store i64 %cells, ptr @xlsx.cells, align 8
  %xel = fsub double %t1, %t0
  %xkeep = icmp ugt i64 %xrep, 0
  br i1 %xkeep, label %xl.store, label %xl.next
xl.store:
  %xidx = sub i64 %xrep, 1
  %xsp = getelementptr inbounds [16 x double], ptr @xlsx.samp, i64 0, i64 %xidx
  store double %xel, ptr %xsp, align 8
  br label %xl.next
xl.next:
  %xrep.n = add nuw i64 %xrep, 1
  %xmore = icmp ult i64 %xrep.n, 17
  br i1 %xmore, label %xl.rep, label %xl.report
xl.report:
  call void @universe_docparse_xlsx_close(ptr %wb2)
  %xl.ops = load i64, ptr @xlsx.cells, align 8
  call void @ut_report_dist(ptr @xlsx.samp, i64 16, i64 %xl.ops, ptr @lbl.xlsx)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
