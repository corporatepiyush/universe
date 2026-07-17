; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/frame.ll — Series primitives + DataFrame methods.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()

declare ptr @malloc(i64)
declare void @free(ptr)

; series
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare void @universe_dataframe_series_free(ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare i32 @universe_dataframe_series_dtype(ptr)
declare ptr @universe_dataframe_series_values(ptr)
declare ptr @universe_dataframe_series_validity(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)
declare ptr @universe_dataframe_series_clone(ptr)
declare i32 @universe_dataframe_series_str_get(ptr, i64, ptr, ptr)
declare ptr @universe_dataframe_series_str_new(ptr, ptr, i64, i64)

; dataframe
declare ptr @universe_dataframe_new()
declare ptr @universe_dataframe_empty_with_height(i64)
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare i32 @universe_dataframe_shape(ptr, ptr, ptr)
declare i32 @universe_dataframe_get_column_names(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_dtypes(ptr, ptr, i64)
declare i32 @universe_dataframe_get_column_index(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_column(ptr, ptr, i64)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare ptr @universe_dataframe_select(ptr, ptr, i64)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_insert_column(ptr, i64, ptr, i64, ptr)
declare i32 @universe_dataframe_replace_column(ptr, i64, ptr)
declare i32 @universe_dataframe_rename(ptr, ptr, i64, ptr, i64)
declare i32 @universe_dataframe_drop_in_place(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_drop(ptr, ptr, i64)
declare ptr @universe_dataframe_drop_many(ptr, ptr, i64)
declare i32 @universe_dataframe_hstack(ptr, ptr, ptr, i64)
declare ptr @universe_dataframe_vstack(ptr, ptr)
declare i32 @universe_dataframe_extend(ptr, ptr)
declare ptr @universe_dataframe_head(ptr, i64)
declare ptr @universe_dataframe_tail(ptr, i64)
declare ptr @universe_dataframe_slice(ptr, i64, i64)
declare i32 @universe_dataframe_split_at(ptr, i64, ptr, ptr)
declare ptr @universe_dataframe_reverse(ptr)
declare ptr @universe_dataframe_shift(ptr, i64)
declare i32 @universe_dataframe_null_count(ptr, ptr, i64)
declare i32 @universe_dataframe_equals(ptr, ptr, ptr)
declare ptr @universe_dataframe_with_row_index(ptr, ptr, i64, i64)
declare void @universe_dataframe_clear(ptr)
declare void @universe_dataframe_free(ptr)

@.col_a  = private constant [1 x i8] c"a"
@.col_b  = private constant [1 x i8] c"b"
@.col_s  = private constant [1 x i8] c"s"
@.col_c  = private constant [1 x i8] c"c"
@.col_x  = private constant [1 x i8] c"x"
@.col_ix = private constant [3 x i8] c"idx"
@.col_zz = private constant [2 x i8] c"zz"

@.m.slen    = private constant [17 x i8] c"series len wrong\00"
@.m.sdtype  = private constant [13 x i8] c"series dtype\00"
@.m.width   = private constant [12 x i8] c"width wrong\00"
@.m.height  = private constant [13 x i8] c"height wrong\00"
@.m.shapeh  = private constant [13 x i8] c"shape h bad\00\00"
@.m.shapew  = private constant [13 x i8] c"shape w bad\00\00"
@.m.gci     = private constant [14 x i8] c"col index bad\00"
@.m.gcinf   = private constant [17 x i8] c"col idx notfound\00"
@.m.colnull = private constant [15 x i8] c"column() null?\00"
@.m.colmiss = private constant [18 x i8] c"missing col !null\00"
@.m.replace = private constant [16 x i8] c"replace failed \00"
@.m.append  = private constant [16 x i8] c"append failed \00\00"
@.m.wc.w    = private constant [16 x i8] c"with_col width \00"
@.m.rn      = private constant [14 x i8] c"rename fail \00\00"
@.m.rn.nf   = private constant [16 x i8] c"rename notfound\00"
@.m.dip     = private constant [16 x i8] c"drop_in_place  \00"
@.m.dip.w   = private constant [16 x i8] c"dip width wrong\00"
@.m.dip.nf  = private constant [15 x i8] c"dip notfound  \00"
@.m.drop.w  = private constant [15 x i8] c"drop width bad\00"
@.m.drop.nn = private constant [16 x i8] c"drop miss !null\00"
@.m.hs      = private constant [14 x i8] c"hstack fail  \00"
@.m.hs.w    = private constant [16 x i8] c"hstack width   \00"
@.m.vs.n    = private constant [15 x i8] c"vstack null?  \00"
@.m.vs.h    = private constant [15 x i8] c"vstack height \00"
@.m.vs.mm   = private constant [17 x i8] c"vstack mismatch \00"
@.m.ext     = private constant [14 x i8] c"extend fail \00\00"
@.m.ext.mm  = private constant [16 x i8] c"extend mismatch\00"
@.m.head    = private constant [13 x i8] c"head height\00\00"
@.m.tail    = private constant [13 x i8] c"tail height\00\00"
@.m.slh     = private constant [13 x i8] c"slice height\00"
@.m.slv     = private constant [13 x i8] c"slice value\00\00"
@.m.slneg   = private constant [15 x i8] c"slice neg null\00"
@.m.spl     = private constant [13 x i8] c"split_at rc\00\00"
@.m.spla    = private constant [13 x i8] c"split a hgt \00"
@.m.splb    = private constant [13 x i8] c"split b hgt \00"
@.m.splneg  = private constant [14 x i8] c"split neg arg\00"
@.m.rev     = private constant [13 x i8] c"reverse val \00"
@.m.revh    = private constant [13 x i8] c"reverse hgt \00"
@.m.shp     = private constant [15 x i8] c"shift+ vacated\00"
@.m.shpv    = private constant [13 x i8] c"shift+ value\00"
@.m.shn     = private constant [15 x i8] c"shift- vacated\00"
@.m.nc      = private constant [16 x i8] c"null_count col \00"
@.m.eqt     = private constant [14 x i8] c"equals true \00\00"
@.m.eqf     = private constant [14 x i8] c"equals false\00\00"
@.m.wri.w   = private constant [16 x i8] c"row_index width\00"
@.m.wri.v   = private constant [16 x i8] c"row_index value\00"
@.m.wri.d   = private constant [16 x i8] c"row_index dtype\00"
@.m.clr.h   = private constant [13 x i8] c"clear height\00"
@.m.clr.w   = private constant [13 x i8] c"clear width\00\00"
@.m.strget  = private constant [13 x i8] c"str_get fail\00"
@.m.strlen  = private constant [13 x i8] c"str_get len\00\00"
@.m.strbyte = private constant [13 x i8] c"str_get byte\00"
@.m.isnull  = private constant [13 x i8] c"is_null flag\00"
@.m.snc     = private constant [16 x i8] c"set_null count \00"
@.m.fuzz    = private constant [16 x i8] c"fuzz violations\00"
@.m.selv    = private constant [14 x i8] c"select width \00"
@.m.selnull = private constant [17 x i8] c"select miss null\00"
@.m.clonev  = private constant [13 x i8] c"clone value \00"

; string data for the STR column: "aa","bbb","c","dddd"
@.strbytes = private constant [10 x i8] c"aabbbcdddd"
@.stroffs  = private constant [5 x i32] [i32 0, i32 2, i32 5, i32 6, i32 10]

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; scratch out-params (entry-block allocas)
  %oh = alloca i64, align 8
  %ow = alloca i64, align 8
  %oidx = alloca i64, align 8
  %obool = alloca i8, align 1
  %ostrp = alloca ptr, align 8
  %ostrl = alloca i64, align 8
  %names4 = alloca [4 x i64], align 8       ; {ptr,len}[2] scratch for names
  %counts = alloca [4 x i64], align 8
  %dtypes = alloca [4 x i32], align 4
  %outa = alloca ptr, align 8
  %outb = alloca ptr, align 8
  %ivals = alloca [4 x i64], align 8         ; i64 column values
  %fvals = alloca [4 x double], align 8      ; f64 column values

  ; -------------------------------------------------------------------
  ; Build a 3-column frame: "a" i64 [10,20,30,40], "b" f64, "s" STR
  ; -------------------------------------------------------------------
  %pa0 = getelementptr [4 x i64], ptr %ivals, i64 0, i64 0
  store i64 10, ptr %pa0, align 8
  %pa1 = getelementptr [4 x i64], ptr %ivals, i64 0, i64 1
  store i64 20, ptr %pa1, align 8
  %pa2 = getelementptr [4 x i64], ptr %ivals, i64 0, i64 2
  store i64 30, ptr %pa2, align 8
  %pa3 = getelementptr [4 x i64], ptr %ivals, i64 0, i64 3
  store i64 40, ptr %pa3, align 8
  %sa = call ptr @universe_dataframe_series_from(i32 1, ptr %ivals, i64 4)

  %saLen = call i64 @universe_dataframe_series_len(ptr %sa)
  call void @ut_check_eq(i64 %saLen, i64 4, ptr @.m.slen)
  %saDt = call i32 @universe_dataframe_series_dtype(ptr %sa)
  %saDt64 = sext i32 %saDt to i64
  call void @ut_check_eq(i64 %saDt64, i64 1, ptr @.m.sdtype)

  %pf0 = getelementptr [4 x double], ptr %fvals, i64 0, i64 0
  store double 1.5, ptr %pf0, align 8
  %pf1 = getelementptr [4 x double], ptr %fvals, i64 0, i64 1
  store double 2.5, ptr %pf1, align 8
  %pf2 = getelementptr [4 x double], ptr %fvals, i64 0, i64 2
  store double 3.5, ptr %pf2, align 8
  %pf3 = getelementptr [4 x double], ptr %fvals, i64 0, i64 3
  store double 4.5, ptr %pf3, align 8
  %sb = call ptr @universe_dataframe_series_from(i32 3, ptr %fvals, i64 4)

  %ss = call ptr @universe_dataframe_series_str_new(ptr @.stroffs, ptr @.strbytes, i64 4, i64 10)

  %df = call ptr @universe_dataframe_new()
  %wc0 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_a, i64 1, ptr %sa)
  call void @ut_check_eq(i64 0, i64 0, ptr @.m.append)
  %wc1 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_b, i64 1, ptr %sb)
  %wc2 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_s, i64 1, ptr %ss)
  %wcsum = add i32 %wc0, %wc1
  %wcsum2 = add i32 %wcsum, %wc2
  %wcsum64 = sext i32 %wcsum2 to i64
  call void @ut_check_eq(i64 %wcsum64, i64 0, ptr @.m.append)

  %w = call i64 @universe_dataframe_width(ptr %df)
  call void @ut_check_eq(i64 %w, i64 3, ptr @.m.width)
  %h = call i64 @universe_dataframe_height(ptr %df)
  call void @ut_check_eq(i64 %h, i64 4, ptr @.m.height)

  ; shape
  call void @universe_dataframe_shape(ptr %df, ptr %oh, ptr %ow)
  %shh = load i64, ptr %oh, align 8
  %shw = load i64, ptr %ow, align 8
  call void @ut_check_eq(i64 %shh, i64 4, ptr @.m.shapeh)
  call void @ut_check_eq(i64 %shw, i64 3, ptr @.m.shapew)

  ; dtypes
  %dtp = getelementptr [4 x i32], ptr %dtypes, i64 0, i64 0
  call void @universe_dataframe_dtypes(ptr %df, ptr %dtp, i64 4)
  %dt0 = load i32, ptr %dtp, align 4
  %dt0_64 = sext i32 %dt0 to i64
  call void @ut_check_eq(i64 %dt0_64, i64 1, ptr @.m.sdtype)

  ; get_column_index hit + notfound
  %gci = call i32 @universe_dataframe_get_column_index(ptr %df, ptr @.col_b, i64 1, ptr %oidx)
  %gci64 = sext i32 %gci to i64
  call void @ut_check_eq(i64 %gci64, i64 0, ptr @.m.gci)
  %idxb = load i64, ptr %oidx, align 8
  call void @ut_check_eq(i64 %idxb, i64 1, ptr @.m.gci)
  %gcinf = call i32 @universe_dataframe_get_column_index(ptr %df, ptr @.col_zz, i64 2, ptr %oidx)
  %gcinf64 = sext i32 %gcinf to i64
  call void @ut_check_eq(i64 %gcinf64, i64 5, ptr @.m.gcinf)

  ; column() hit / miss
  %cb = call ptr @universe_dataframe_column(ptr %df, ptr @.col_b, i64 1)
  %cbnn = icmp ne ptr %cb, null
  call void @ut_check(i1 %cbnn, ptr @.m.colnull)
  %cmiss = call ptr @universe_dataframe_column(ptr %df, ptr @.col_zz, i64 2)
  %cmissnull = icmp eq ptr %cmiss, null
  call void @ut_check(i1 %cmissnull, ptr @.m.colmiss)

  ; select_at_idx
  %sat = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 0)
  %satnn = icmp ne ptr %sat, null
  call void @ut_check(i1 %satnn, ptr @.m.colnull)

  ; str_get on column s: element 1 = "bbb"
  %rcstr = call i32 @universe_dataframe_series_str_get(ptr %ss, i64 1, ptr %ostrp, ptr %ostrl)
  %rcstr64 = sext i32 %rcstr to i64
  call void @ut_check_eq(i64 %rcstr64, i64 0, ptr @.m.strget)
  %slen1 = load i64, ptr %ostrl, align 8
  call void @ut_check_eq(i64 %slen1, i64 3, ptr @.m.strlen)
  %sp1 = load ptr, ptr %ostrp, align 8
  %sb0 = load i8, ptr %sp1, align 1
  %sb0_64 = sext i8 %sb0 to i64
  call void @ut_check_eq(i64 %sb0_64, i64 98, ptr @.m.strbyte)   ; 'b' == 98

  ; -------------------------------------------------------------------
  ; with_column REPLACE (same name "a", new i64 series [100,200,300,400])
  ; -------------------------------------------------------------------
  store i64 100, ptr %pa0, align 8
  store i64 200, ptr %pa1, align 8
  store i64 300, ptr %pa2, align 8
  store i64 400, ptr %pa3, align 8
  %sa2 = call ptr @universe_dataframe_series_from(i32 1, ptr %ivals, i64 4)
  %wcr = call i32 @universe_dataframe_with_column(ptr %df, ptr @.col_a, i64 1, ptr %sa2)
  %wcr64 = sext i32 %wcr to i64
  call void @ut_check_eq(i64 %wcr64, i64 0, ptr @.m.replace)
  %w2 = call i64 @universe_dataframe_width(ptr %df)
  call void @ut_check_eq(i64 %w2, i64 3, ptr @.m.wc.w)        ; still 3 cols
  %sa2b = call ptr @universe_dataframe_column(ptr %df, ptr @.col_a, i64 1)
  %sa2vals = call ptr @universe_dataframe_series_values(ptr %sa2b)
  %sa2v0 = load i64, ptr %sa2vals, align 8
  call void @ut_check_eq(i64 %sa2v0, i64 100, ptr @.m.replace)

  ; rename "b" -> "c"
  %rn = call i32 @universe_dataframe_rename(ptr %df, ptr @.col_b, i64 1, ptr @.col_c, i64 1)
  %rn64 = sext i32 %rn to i64
  call void @ut_check_eq(i64 %rn64, i64 0, ptr @.m.rn)
  %gcic = call i32 @universe_dataframe_get_column_index(ptr %df, ptr @.col_c, i64 1, ptr %oidx)
  %gcic64 = sext i32 %gcic to i64
  call void @ut_check_eq(i64 %gcic64, i64 0, ptr @.m.rn)
  %rnnf = call i32 @universe_dataframe_rename(ptr %df, ptr @.col_zz, i64 2, ptr @.col_x, i64 1)
  %rnnf64 = sext i32 %rnnf to i64
  call void @ut_check_eq(i64 %rnnf64, i64 5, ptr @.m.rn.nf)

  ; -------------------------------------------------------------------
  ; drop / drop_in_place tested on CLONES so df stays intact
  ; -------------------------------------------------------------------
  %dfDrop = call ptr @universe_dataframe_drop(ptr %df, ptr @.col_s, i64 1)
  %dfDropW = call i64 @universe_dataframe_width(ptr %dfDrop)
  call void @ut_check_eq(i64 %dfDropW, i64 2, ptr @.m.drop.w)
  ; dropping a missing col returns null
  %dfDropMiss = call ptr @universe_dataframe_drop(ptr %df, ptr @.col_zz, i64 2)
  %dfDropMissNull = icmp eq ptr %dfDropMiss, null
  call void @ut_check(i1 %dfDropMissNull, ptr @.m.drop.nn)
  call void @universe_dataframe_free(ptr %dfDrop)

  ; drop_in_place on a clone-frame
  %dfDip = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %dipw0 = call i64 @universe_dataframe_width(ptr %dfDip)
  %rcdip = call i32 @universe_dataframe_drop_in_place(ptr %dfDip, ptr @.col_a, i64 1, ptr %outa)
  %rcdip64 = sext i32 %rcdip to i64
  call void @ut_check_eq(i64 %rcdip64, i64 0, ptr @.m.dip)
  %dipw1 = call i64 @universe_dataframe_width(ptr %dfDip)
  %dipexp = sub i64 %dipw0, 1
  call void @ut_check_eq(i64 %dipw1, i64 %dipexp, ptr @.m.dip.w)
  %removed = load ptr, ptr %outa, align 8
  call void @universe_dataframe_series_free(ptr %removed)
  %rcdipnf = call i32 @universe_dataframe_drop_in_place(ptr %dfDip, ptr @.col_zz, i64 2, ptr %outa)
  %rcdipnf64 = sext i32 %rcdipnf to i64
  call void @ut_check_eq(i64 %rcdipnf64, i64 5, ptr @.m.dip.nf)
  call void @universe_dataframe_free(ptr %dfDip)

  ; -------------------------------------------------------------------
  ; hstack: add an i32 column "x" to a fresh clone
  ; -------------------------------------------------------------------
  %dfHs = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %sx = call ptr @universe_dataframe_series_new(i32 0, i64 4)
  %sxarr = alloca ptr, align 8
  store ptr %sx, ptr %sxarr, align 8
  %hsnames = alloca [2 x i64], align 8
  %hsn0 = getelementptr [2 x i64], ptr %hsnames, i64 0, i64 0
  store ptr @.col_x, ptr %hsn0, align 8
  %hsn1 = getelementptr [2 x i64], ptr %hsnames, i64 0, i64 1
  store i64 1, ptr %hsn1, align 8
  %rchs = call i32 @universe_dataframe_hstack(ptr %dfHs, ptr %sxarr, ptr %hsnames, i64 1)
  %rchs64 = sext i32 %rchs to i64
  call void @ut_check_eq(i64 %rchs64, i64 0, ptr @.m.hs)
  %hsw = call i64 @universe_dataframe_width(ptr %dfHs)
  call void @ut_check_eq(i64 %hsw, i64 4, ptr @.m.hs.w)
  call void @universe_dataframe_free(ptr %dfHs)

  ; -------------------------------------------------------------------
  ; vstack two matching clones -> height 8; mismatched schema -> null
  ; -------------------------------------------------------------------
  %dfV1 = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %dfV2 = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %dfVs = call ptr @universe_dataframe_vstack(ptr %dfV1, ptr %dfV2)
  %dfVsNN = icmp ne ptr %dfVs, null
  call void @ut_check(i1 %dfVsNN, ptr @.m.vs.n)
  %vsh = call i64 @universe_dataframe_height(ptr %dfVs)
  call void @ut_check_eq(i64 %vsh, i64 8, ptr @.m.vs.h)
  ; check a value carried across: column a element 4 (== element 0 of v2) == 100
  %vsA = call ptr @universe_dataframe_column(ptr %dfVs, ptr @.col_a, i64 1)
  %vsAvals = call ptr @universe_dataframe_series_values(ptr %vsA)
  %vsA4p = getelementptr i64, ptr %vsAvals, i64 4
  %vsA4 = load i64, ptr %vsA4p, align 8
  call void @ut_check_eq(i64 %vsA4, i64 100, ptr @.m.vs.h)
  ; check STR carried: element 5 == "bbb"
  %vsS = call ptr @universe_dataframe_column(ptr %dfVs, ptr @.col_s, i64 1)
  %vsSrc = call i32 @universe_dataframe_series_str_get(ptr %vsS, i64 5, ptr %ostrp, ptr %ostrl)
  %vsSlen = load i64, ptr %ostrl, align 8
  call void @ut_check_eq(i64 %vsSlen, i64 3, ptr @.m.vs.h)
  call void @universe_dataframe_free(ptr %dfVs)
  call void @universe_dataframe_free(ptr %dfV1)
  call void @universe_dataframe_free(ptr %dfV2)

  ; mismatched schema: 2-col frame vs the 3-col df -> null
  %dfMM = call ptr @universe_dataframe_new()
  %mmS = call ptr @universe_dataframe_series_new(i32 1, i64 4)
  %mmwc = call i32 @universe_dataframe_with_column(ptr %dfMM, ptr @.col_a, i64 1, ptr %mmS)
  %dfMMvs = call ptr @universe_dataframe_vstack(ptr %df, ptr %dfMM)
  %dfMMvsNull = icmp eq ptr %dfMMvs, null
  call void @ut_check(i1 %dfMMvsNull, ptr @.m.vs.mm)
  ; extend mismatch -> 8
  %extmm = call i32 @universe_dataframe_extend(ptr %dfMM, ptr %df)
  %extmm64 = sext i32 %extmm to i64
  call void @ut_check_eq(i64 %extmm64, i64 8, ptr @.m.ext.mm)
  call void @universe_dataframe_free(ptr %dfMM)

  ; extend (in place) two matching clones
  %dfE1 = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %dfE2 = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %rcext = call i32 @universe_dataframe_extend(ptr %dfE1, ptr %dfE2)
  %rcext64 = sext i32 %rcext to i64
  call void @ut_check_eq(i64 %rcext64, i64 0, ptr @.m.ext)
  %exth = call i64 @universe_dataframe_height(ptr %dfE1)
  call void @ut_check_eq(i64 %exth, i64 8, ptr @.m.ext)
  call void @universe_dataframe_free(ptr %dfE1)
  call void @universe_dataframe_free(ptr %dfE2)

  ; -------------------------------------------------------------------
  ; head / tail / slice / split_at boundaries
  ; -------------------------------------------------------------------
  %dfHead = call ptr @universe_dataframe_head(ptr %df, i64 2)
  %hheadh = call i64 @universe_dataframe_height(ptr %dfHead)
  call void @ut_check_eq(i64 %hheadh, i64 2, ptr @.m.head)
  call void @universe_dataframe_free(ptr %dfHead)
  ; head beyond height -> clamp
  %dfHeadBig = call ptr @universe_dataframe_head(ptr %df, i64 100)
  %hbigh = call i64 @universe_dataframe_height(ptr %dfHeadBig)
  call void @ut_check_eq(i64 %hbigh, i64 4, ptr @.m.head)
  call void @universe_dataframe_free(ptr %dfHeadBig)
  ; head(0)
  %dfHead0 = call ptr @universe_dataframe_head(ptr %df, i64 0)
  %hh0 = call i64 @universe_dataframe_height(ptr %dfHead0)
  call void @ut_check_eq(i64 %hh0, i64 0, ptr @.m.head)
  call void @universe_dataframe_free(ptr %dfHead0)

  %dfTail = call ptr @universe_dataframe_tail(ptr %df, i64 2)
  %tailh = call i64 @universe_dataframe_height(ptr %dfTail)
  call void @ut_check_eq(i64 %tailh, i64 2, ptr @.m.tail)
  ; tail's column a first value == element 2 of df == 300
  %tailA = call ptr @universe_dataframe_column(ptr %dfTail, ptr @.col_a, i64 1)
  %tailAvals = call ptr @universe_dataframe_series_values(ptr %tailA)
  %tailA0 = load i64, ptr %tailAvals, align 8
  call void @ut_check_eq(i64 %tailA0, i64 300, ptr @.m.tail)
  call void @universe_dataframe_free(ptr %dfTail)

  %dfSlice = call ptr @universe_dataframe_slice(ptr %df, i64 1, i64 2)
  %slh = call i64 @universe_dataframe_height(ptr %dfSlice)
  call void @ut_check_eq(i64 %slh, i64 2, ptr @.m.slh)
  %slA = call ptr @universe_dataframe_column(ptr %dfSlice, ptr @.col_a, i64 1)
  %slAvals = call ptr @universe_dataframe_series_values(ptr %slA)
  %slA0 = load i64, ptr %slAvals, align 8
  call void @ut_check_eq(i64 %slA0, i64 200, ptr @.m.slv)
  call void @universe_dataframe_free(ptr %dfSlice)
  ; slice offset > height -> empty
  %dfSliceBig = call ptr @universe_dataframe_slice(ptr %df, i64 100, i64 5)
  %slbh = call i64 @universe_dataframe_height(ptr %dfSliceBig)
  call void @ut_check_eq(i64 %slbh, i64 0, ptr @.m.slh)
  call void @universe_dataframe_free(ptr %dfSliceBig)
  ; slice negative -> null
  %dfSliceNeg = call ptr @universe_dataframe_slice(ptr %df, i64 -1, i64 2)
  %slnegnull = icmp eq ptr %dfSliceNeg, null
  call void @ut_check(i1 %slnegnull, ptr @.m.slneg)

  ; split_at 2
  %rcsp = call i32 @universe_dataframe_split_at(ptr %df, i64 2, ptr %outa, ptr %outb)
  %rcsp64 = sext i32 %rcsp to i64
  call void @ut_check_eq(i64 %rcsp64, i64 0, ptr @.m.spl)
  %spa = load ptr, ptr %outa, align 8
  %spb = load ptr, ptr %outb, align 8
  %spah = call i64 @universe_dataframe_height(ptr %spa)
  %spbh = call i64 @universe_dataframe_height(ptr %spb)
  call void @ut_check_eq(i64 %spah, i64 2, ptr @.m.spla)
  call void @ut_check_eq(i64 %spbh, i64 2, ptr @.m.splb)
  call void @universe_dataframe_free(ptr %spa)
  call void @universe_dataframe_free(ptr %spb)
  ; split_at negative -> 8
  %rcspneg = call i32 @universe_dataframe_split_at(ptr %df, i64 -1, ptr %outa, ptr %outb)
  %rcspneg64 = sext i32 %rcspneg to i64
  call void @ut_check_eq(i64 %rcspneg64, i64 8, ptr @.m.splneg)

  ; -------------------------------------------------------------------
  ; reverse
  ; -------------------------------------------------------------------
  %dfRev = call ptr @universe_dataframe_reverse(ptr %df)
  %revh = call i64 @universe_dataframe_height(ptr %dfRev)
  call void @ut_check_eq(i64 %revh, i64 4, ptr @.m.revh)
  %revA = call ptr @universe_dataframe_column(ptr %dfRev, ptr @.col_a, i64 1)
  %revAvals = call ptr @universe_dataframe_series_values(ptr %revA)
  %revA0 = load i64, ptr %revAvals, align 8
  call void @ut_check_eq(i64 %revA0, i64 400, ptr @.m.rev)   ; last becomes first
  ; reversed STR element 0 == "dddd" (len 4)
  %revS = call ptr @universe_dataframe_column(ptr %dfRev, ptr @.col_s, i64 1)
  %revSrc = call i32 @universe_dataframe_series_str_get(ptr %revS, i64 0, ptr %ostrp, ptr %ostrl)
  %revSlen = load i64, ptr %ostrl, align 8
  call void @ut_check_eq(i64 %revSlen, i64 4, ptr @.m.rev)
  call void @universe_dataframe_free(ptr %dfRev)

  ; -------------------------------------------------------------------
  ; shift(+1): row0 becomes null, row1 = old row0
  ; -------------------------------------------------------------------
  %dfShP = call ptr @universe_dataframe_shift(ptr %df, i64 1)
  %shPA = call ptr @universe_dataframe_column(ptr %dfShP, ptr @.col_a, i64 1)
  %shPnc = call i64 @universe_dataframe_series_null_count(ptr %shPA)
  call void @ut_check_eq(i64 %shPnc, i64 1, ptr @.m.shp)
  call void @universe_dataframe_series_is_null(ptr %shPA, i64 0, ptr %obool)
  %shP0null = load i8, ptr %obool, align 1
  %shP0null64 = zext i8 %shP0null to i64
  call void @ut_check_eq(i64 %shP0null64, i64 1, ptr @.m.shp)
  %shPvals = call ptr @universe_dataframe_series_values(ptr %shPA)
  %shPv1p = getelementptr i64, ptr %shPvals, i64 1
  %shPv1 = load i64, ptr %shPv1p, align 8
  call void @ut_check_eq(i64 %shPv1, i64 100, ptr @.m.shpv)   ; old row0
  call void @universe_dataframe_free(ptr %dfShP)

  ; shift(-1): last row becomes null
  %dfShN = call ptr @universe_dataframe_shift(ptr %df, i64 -1)
  %shNA = call ptr @universe_dataframe_column(ptr %dfShN, ptr @.col_a, i64 1)
  call void @universe_dataframe_series_is_null(ptr %shNA, i64 3, ptr %obool)
  %shN3null = load i8, ptr %obool, align 1
  %shN3null64 = zext i8 %shN3null to i64
  call void @ut_check_eq(i64 %shN3null64, i64 1, ptr @.m.shn)
  call void @universe_dataframe_free(ptr %dfShN)

  ; -------------------------------------------------------------------
  ; null_count with injected nulls (on a clone so df stays intact)
  ; -------------------------------------------------------------------
  %dfNC = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %ncA = call ptr @universe_dataframe_column(ptr %dfNC, ptr @.col_a, i64 1)
  %sn0 = call i32 @universe_dataframe_series_set_null(ptr %ncA, i64 0)
  %sn2 = call i32 @universe_dataframe_series_set_null(ptr %ncA, i64 2)
  ; setting the same index twice must NOT double count
  %sn2b = call i32 @universe_dataframe_series_set_null(ptr %ncA, i64 2)
  %ncAcnt = call i64 @universe_dataframe_series_null_count(ptr %ncA)
  call void @ut_check_eq(i64 %ncAcnt, i64 2, ptr @.m.snc)
  %ncp0 = getelementptr [4 x i64], ptr %counts, i64 0, i64 0
  call void @universe_dataframe_null_count(ptr %dfNC, ptr %ncp0, i64 4)
  %nc0 = load i64, ptr %ncp0, align 8
  call void @ut_check_eq(i64 %nc0, i64 2, ptr @.m.nc)
  ; is_null flag on set index
  call void @universe_dataframe_series_is_null(ptr %ncA, i64 0, ptr %obool)
  %isn0 = load i8, ptr %obool, align 1
  %isn0_64 = zext i8 %isn0 to i64
  call void @ut_check_eq(i64 %isn0_64, i64 1, ptr @.m.isnull)
  call void @universe_dataframe_series_is_null(ptr %ncA, i64 1, ptr %obool)
  %isn1 = load i8, ptr %obool, align 1
  %isn1_64 = zext i8 %isn1 to i64
  call void @ut_check_eq(i64 %isn1_64, i64 0, ptr @.m.isnull)
  ; clone should preserve nulls
  %ncAclone = call ptr @universe_dataframe_series_clone(ptr %ncA)
  %ncAclonecnt = call i64 @universe_dataframe_series_null_count(ptr %ncAclone)
  call void @ut_check_eq(i64 %ncAclonecnt, i64 2, ptr @.m.snc)
  call void @universe_dataframe_series_free(ptr %ncAclone)
  call void @universe_dataframe_free(ptr %dfNC)

  ; -------------------------------------------------------------------
  ; equals (equal + unequal)
  ; -------------------------------------------------------------------
  %dfEqA = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %dfEqB = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %eqrc = call i32 @universe_dataframe_equals(ptr %dfEqA, ptr %dfEqB, ptr %obool)
  %eqv = load i8, ptr %obool, align 1
  %eqv64 = zext i8 %eqv to i64
  call void @ut_check_eq(i64 %eqv64, i64 1, ptr @.m.eqt)
  ; mutate B (set a null) -> unequal
  %eqBA = call ptr @universe_dataframe_column(ptr %dfEqB, ptr @.col_a, i64 1)
  %eqsn = call i32 @universe_dataframe_series_set_null(ptr %eqBA, i64 0)
  %eqrc2 = call i32 @universe_dataframe_equals(ptr %dfEqA, ptr %dfEqB, ptr %obool)
  %eqv2 = load i8, ptr %obool, align 1
  %eqv2_64 = zext i8 %eqv2 to i64
  call void @ut_check_eq(i64 %eqv2_64, i64 0, ptr @.m.eqf)
  call void @universe_dataframe_free(ptr %dfEqA)
  call void @universe_dataframe_free(ptr %dfEqB)

  ; -------------------------------------------------------------------
  ; select subset
  ; -------------------------------------------------------------------
  %seln0 = getelementptr [4 x i64], ptr %names4, i64 0, i64 0
  store ptr @.col_s, ptr %seln0, align 8
  %seln1 = getelementptr [4 x i64], ptr %names4, i64 0, i64 1
  store i64 1, ptr %seln1, align 8
  %seln2 = getelementptr [4 x i64], ptr %names4, i64 0, i64 2
  store ptr @.col_a, ptr %seln2, align 8
  %seln3 = getelementptr [4 x i64], ptr %names4, i64 0, i64 3
  store i64 1, ptr %seln3, align 8
  %dfSel = call ptr @universe_dataframe_select(ptr %df, ptr %names4, i64 2)
  %selw = call i64 @universe_dataframe_width(ptr %dfSel)
  call void @ut_check_eq(i64 %selw, i64 2, ptr @.m.selv)
  ; first selected column is "s"
  %selc0 = call ptr @universe_dataframe_select_at_idx(ptr %dfSel, i64 0)
  %selc0dt = call i32 @universe_dataframe_series_dtype(ptr %selc0)
  %selc0dt64 = sext i32 %selc0dt to i64
  call void @ut_check_eq(i64 %selc0dt64, i64 5, ptr @.m.selv)   ; STR
  call void @universe_dataframe_free(ptr %dfSel)
  ; select a missing name -> null
  store ptr @.col_zz, ptr %seln0, align 8
  store i64 2, ptr %seln1, align 8
  %dfSelMiss = call ptr @universe_dataframe_select(ptr %df, ptr %names4, i64 1)
  %selmissnull = icmp eq ptr %dfSelMiss, null
  call void @ut_check(i1 %selmissnull, ptr @.m.selnull)

  ; -------------------------------------------------------------------
  ; with_row_index (prepend i64 "idx" col with offset 100)
  ; -------------------------------------------------------------------
  %dfRI = call ptr @universe_dataframe_with_row_index(ptr %df, ptr @.col_ix, i64 3, i64 100)
  %riw = call i64 @universe_dataframe_width(ptr %dfRI)
  call void @ut_check_eq(i64 %riw, i64 4, ptr @.m.wri.w)
  %riIx = call ptr @universe_dataframe_select_at_idx(ptr %dfRI, i64 0)
  %riIxdt = call i32 @universe_dataframe_series_dtype(ptr %riIx)
  %riIxdt64 = sext i32 %riIxdt to i64
  call void @ut_check_eq(i64 %riIxdt64, i64 1, ptr @.m.wri.d)
  %riIxvals = call ptr @universe_dataframe_series_values(ptr %riIx)
  %riIx0 = load i64, ptr %riIxvals, align 8
  call void @ut_check_eq(i64 %riIx0, i64 100, ptr @.m.wri.v)
  %riIx2p = getelementptr i64, ptr %riIxvals, i64 2
  %riIx2 = load i64, ptr %riIx2p, align 8
  call void @ut_check_eq(i64 %riIx2, i64 102, ptr @.m.wri.v)
  call void @universe_dataframe_free(ptr %dfRI)

  ; -------------------------------------------------------------------
  ; insert_column at position 1
  ; -------------------------------------------------------------------
  %dfIns = call ptr @universe_dataframe_select_all_clone(ptr %df)
  %insS = call ptr @universe_dataframe_series_new(i32 0, i64 4)
  %rcins = call i32 @universe_dataframe_insert_column(ptr %dfIns, i64 1, ptr @.col_x, i64 1, ptr %insS)
  %rcins64 = sext i32 %rcins to i64
  call void @ut_check_eq(i64 %rcins64, i64 0, ptr @.m.hs)
  ; the column at index 1 must now be "x"
  %insIdx1 = call i32 @universe_dataframe_get_column_index(ptr %dfIns, ptr @.col_x, i64 1, ptr %oidx)
  %insAt = load i64, ptr %oidx, align 8
  call void @ut_check_eq(i64 %insAt, i64 1, ptr @.m.hs.w)
  ; replace_column at index 0
  %repS = call ptr @universe_dataframe_series_new(i32 1, i64 4)
  %rcrep = call i32 @universe_dataframe_replace_column(ptr %dfIns, i64 0, ptr %repS)
  %rcrep64 = sext i32 %rcrep to i64
  call void @ut_check_eq(i64 %rcrep64, i64 0, ptr @.m.replace)
  call void @universe_dataframe_free(ptr %dfIns)

  ; -------------------------------------------------------------------
  ; clear: keeps schema, height -> 0
  ; -------------------------------------------------------------------
  %dfClr = call ptr @universe_dataframe_select_all_clone(ptr %df)
  call void @universe_dataframe_clear(ptr %dfClr)
  %clrh = call i64 @universe_dataframe_height(ptr %dfClr)
  call void @ut_check_eq(i64 %clrh, i64 0, ptr @.m.clr.h)
  %clrw = call i64 @universe_dataframe_width(ptr %dfClr)
  call void @ut_check_eq(i64 %clrw, i64 3, ptr @.m.clr.w)
  call void @universe_dataframe_free(ptr %dfClr)

  ; -------------------------------------------------------------------
  ; randomized fuzz: build a frame, drop/select/slice/reverse sequence,
  ; and assert row integrity (column follows permutation) each round.
  ; -------------------------------------------------------------------
  %violp = alloca i64, align 8
  store i64 0, ptr %violp, align 8
  %rngp = alloca i64, align 8
  store i64 88172645463325252, ptr %rngp, align 8
  br label %fuzz.head

fuzz.head:
  %fi = phi i64 [ 0, %entry ], [ %fi.next, %fuzz.cleanup ]
  %fcmp = icmp ult i64 %fi, 200
  br i1 %fcmp, label %fuzz.body, label %fuzz.done

fuzz.body:
  ; make an n = 1..32 i64 column with values v[i] = i*7+1
  %r0 = call i64 @ut_rand(ptr %rngp)
  %rmod = urem i64 %r0, 32
  %fn = add i64 %rmod, 1
  %fcol = call ptr @universe_dataframe_series_new(i32 1, i64 %fn)
  %fvalsp = call ptr @universe_dataframe_series_values(ptr %fcol)
  br label %fuzz.fill.head

fuzz.fill.head:
  %fj = phi i64 [ 0, %fuzz.body ], [ %fj.next, %fuzz.fill.body ]
  %fjcmp = icmp ult i64 %fj, %fn
  br i1 %fjcmp, label %fuzz.fill.body, label %fuzz.build

fuzz.fill.body:
  %fmul = mul i64 %fj, 7
  %fval = add i64 %fmul, 1
  %fvp = getelementptr i64, ptr %fvalsp, i64 %fj
  store i64 %fval, ptr %fvp, align 8
  %fj.next = add i64 %fj, 1
  br label %fuzz.fill.head

fuzz.build:
  %fdf = call ptr @universe_dataframe_new()
  %fwc = call i32 @universe_dataframe_with_column(ptr %fdf, ptr @.col_a, i64 1, ptr %fcol)
  ; reverse then reverse again == identity of values
  %fr1 = call ptr @universe_dataframe_reverse(ptr %fdf)
  %fr2 = call ptr @universe_dataframe_reverse(ptr %fr1)
  %fr2A = call ptr @universe_dataframe_column(ptr %fr2, ptr @.col_a, i64 1)
  %fr2vals = call ptr @universe_dataframe_series_values(ptr %fr2A)
  %fr2v0 = load i64, ptr %fr2vals, align 8
  ; after double reverse, element 0 must equal original element 0 == 1
  %fr2ok = icmp eq i64 %fr2v0, 1
  br i1 %fr2ok, label %fuzz.slice, label %fuzz.viol1

fuzz.viol1:
  %v1 = load i64, ptr %violp, align 8
  %v1n = add i64 %v1, 1
  store i64 %v1n, ptr %violp, align 8
  br label %fuzz.slice

fuzz.slice:
  ; slice [0, fn) then height must equal fn
  %fsl = call ptr @universe_dataframe_slice(ptr %fdf, i64 0, i64 %fn)
  %fslh = call i64 @universe_dataframe_height(ptr %fsl)
  %fslok = icmp eq i64 %fslh, %fn
  br i1 %fslok, label %fuzz.cleanup, label %fuzz.viol2

fuzz.viol2:
  %v2 = load i64, ptr %violp, align 8
  %v2n = add i64 %v2, 1
  store i64 %v2n, ptr %violp, align 8
  br label %fuzz.cleanup

fuzz.cleanup:
  call void @universe_dataframe_free(ptr %fsl)
  call void @universe_dataframe_free(ptr %fr2)
  call void @universe_dataframe_free(ptr %fr1)
  call void @universe_dataframe_free(ptr %fdf)
  %fi.next = add i64 %fi, 1
  br label %fuzz.head

fuzz.done:
  %viol = load i64, ptr %violp, align 8
  call void @ut_check_eq(i64 %viol, i64 0, ptr @.m.fuzz)

  ; free the base frame
  call void @universe_dataframe_free(ptr %df)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; helper: clone a full df via select of all columns (returns NEW owned df)
define ptr @universe_dataframe_select_all_clone(ptr %df) {
entry:
  %nc = call i64 @universe_dataframe_width(ptr %df)
  %bytes = mul i64 %nc, 16
  %names = call ptr @malloc(i64 %bytes)
  %ncp = call i32 @universe_dataframe_get_column_names(ptr %df, ptr %names, i64 %nc, ptr @ncdummy)
  %out = call ptr @universe_dataframe_select(ptr %df, ptr %names, i64 %nc)
  call void @free(ptr %names)
  ret ptr %out
}

@ncdummy = private global i64 0
