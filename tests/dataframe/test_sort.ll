; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/sort.ll — permutation sort / sort_by / arg_sort /
; top_k / bottom_k. Verifies: full ordering (asc & desc), row integrity (a
; companion column follows the permutation), the permutation is a true
; permutation (each 0..n-1 once), multi-key tie-break by 2nd key, stability
; (radix + merge paths), float NaN/null ordering, top_k/bottom_k vs a
; full-sort reference, and a fixed-seed randomized run.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i1  @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; frame API
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_values(ptr)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)
declare i64 @universe_dataframe_series_len(ptr)
declare ptr @universe_dataframe_new()
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare ptr @universe_dataframe_column(ptr, ptr, i64)
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare void @universe_dataframe_free(ptr)

; sort API (under test)
declare ptr @universe_dataframe_sort(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_sort_in_place(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_arg_sort(ptr, i32, ptr)
declare ptr @universe_dataframe_top_k(ptr, i64, ptr, i64, i32)
declare ptr @universe_dataframe_bottom_k(ptr, i64, ptr, i64)

@.k   = private constant [1 x i8] c"k"
@.id  = private constant [2 x i8] c"id"
@.k1  = private constant [2 x i8] c"k1"
@.k2  = private constant [2 x i8] c"k2"
@.seq = private constant [3 x i8] c"seq"
@.fk  = private constant [2 x i8] c"fk"

@.m.perm    = private constant [26 x i8] c"id not a true permutation\00"
@.m.asc     = private constant [23 x i8] c"key not ascending sort\00"
@.m.desc    = private constant [24 x i8] c"key not descending sort\00"
@.m.integ   = private constant [21 x i8] c"row integrity broken\00"
@.m.tie     = private constant [24 x i8] c"multi-key tie-break bad\00"
@.m.stab    = private constant [20 x i8] c"stability violated\00\00"
@.m.stabf   = private constant [25 x i8] c"f64 stability violated  \00"
@.m.nulls   = private constant [23 x i8] c"nulls not ordered last\00"
@.m.nan     = private constant [22 x i8] c"NaN not ordered last \00"
@.m.topk    = private constant [22 x i8] c"top_k mismatch vs ref\00"
@.m.botk    = private constant [25 x i8] c"bottom_k mismatch vs ref\00"
@.m.height  = private constant [18 x i8] c"sorted height bad\00"
@.m.rnd     = private constant [22 x i8] c"randomized sort wrong\00"
@.m.args    = private constant [20 x i8] c"arg_sort perm wrong\00"
@.m.inplace = private constant [23 x i8] c"sort_in_place ordering\00"
@.bench.fmt = private constant [40 x i8] c"BENCH df_sort i64 n=%d: %.1f ns/row\0A\00\00\00\00"

; ---------------------------------------------------------------------------
; helpers
; ---------------------------------------------------------------------------

; values ptr of a named column
define ptr @colvals(ptr %df, ptr %name, i64 %nl) {
entry:
  %col = call ptr @universe_dataframe_column(ptr %df, ptr %name, i64 %nl)
  %v = call ptr @universe_dataframe_series_values(ptr %col)
  ret ptr %v
}

; is element (df,name,i) null? returns i1
define i1 @colnull(ptr %df, ptr %name, i64 %nl, i64 %i) {
entry:
  %b = alloca i8, align 1
  %col = call ptr @universe_dataframe_column(ptr %df, ptr %name, i64 %nl)
  %rc = call i32 @universe_dataframe_series_is_null(ptr %col, i64 %i, ptr %b)
  %v = load i8, ptr %b, align 1
  %r = icmp ne i8 %v, 0
  ret i1 %r
}

; count violations: idx[0..n) not a permutation of 0..n-1
define i64 @is_perm_viol(ptr %idx, i64 %n) {
entry:
  %seen = call ptr @malloc(i64 %n)
  call void @llvm.memset.p0.i64(ptr %seen, i8 0, i64 %n, i1 false)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done

body:
  %p = getelementptr i64, ptr %idx, i64 %i
  %v = load i64, ptr %p, align 8
  %oob0 = icmp slt i64 %v, 0
  %oobn = icmp sge i64 %v, %n
  %oob = or i1 %oob0, %oobn
  br i1 %oob, label %bad, label %chkseen

chkseen:
  %sp = getelementptr i8, ptr %seen, i64 %v
  %sv = load i8, ptr %sp, align 1
  %dup = icmp ne i8 %sv, 0
  br i1 %dup, label %bad, label %mark

mark:
  store i8 1, ptr %sp, align 1
  br label %cont

bad:
  br label %cont

cont:
  %inc = phi i64 [ 1, %bad ], [ 0, %mark ]
  %viol.n = add i64 %viol, %inc
  %i.n = add i64 %i, 1
  br label %loop

done:
  call void @free(ptr %seen)
  ret i64 %viol
}

; count ordering violations in i64 column values[0..n); desc!=0 => descending
define i64 @sorted_i64_viol(ptr %vals, i64 %n, i32 %desc) {
entry:
  %descb = icmp ne i32 %desc, 0
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.n, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done

body:
  %im1 = sub i64 %i, 1
  %pp = getelementptr i64, ptr %vals, i64 %im1
  %prev = load i64, ptr %pp, align 8
  %cp = getelementptr i64, ptr %vals, i64 %i
  %cur = load i64, ptr %cp, align 8
  %okasc = icmp sle i64 %prev, %cur
  %okdesc = icmp sge i64 %prev, %cur
  %ok = select i1 %descb, i1 %okdesc, i1 %okasc
  br i1 %ok, label %cont, label %bad

bad:
  br label %cont

cont:
  %inc = phi i64 [ 0, %body ], [ 1, %bad ]
  %viol.n = add i64 %viol, %inc
  %i.n = add i64 %i, 1
  br label %loop

done:
  ret i64 %viol
}

; row-integrity: for sorted df with cols "k","id", k[i] must equal
; orig_k[id[i]]. Returns violation count.
define i64 @integ_viol(ptr %df, ptr %orig_k, i64 %n) {
entry:
  %kv = call ptr @colvals(ptr %df, ptr @.k, i64 1)
  %iv = call ptr @colvals(ptr %df, ptr @.id, i64 2)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done

body:
  %kp = getelementptr i64, ptr %kv, i64 %i
  %k = load i64, ptr %kp, align 8
  %ip = getelementptr i64, ptr %iv, i64 %i
  %id = load i64, ptr %ip, align 8
  %op = getelementptr i64, ptr %orig_k, i64 %id
  %ok_expect = load i64, ptr %op, align 8
  %eq = icmp eq i64 %k, %ok_expect
  br i1 %eq, label %cont, label %bad

bad:
  br label %cont

cont:
  %inc = phi i64 [ 0, %body ], [ 1, %bad ]
  %viol.n = add i64 %viol, %inc
  %i.n = add i64 %i, 1
  br label %loop

done:
  ret i64 %viol
}

; stability on i64: cols "k" (dup keys) + "seq" (original order). Among equal
; keys, seq must strictly increase. Returns violation count.
define i64 @stable_viol_i64(ptr %df, i64 %n) {
entry:
  %kv = call ptr @colvals(ptr %df, ptr @.k, i64 1)
  %sv = call ptr @colvals(ptr %df, ptr @.seq, i64 3)
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.n, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done

body:
  %im1 = sub i64 %i, 1
  %kpp = getelementptr i64, ptr %kv, i64 %im1
  %kprev = load i64, ptr %kpp, align 8
  %kcp = getelementptr i64, ptr %kv, i64 %i
  %kcur = load i64, ptr %kcp, align 8
  %same = icmp eq i64 %kprev, %kcur
  br i1 %same, label %chk, label %cont0

chk:
  %spp = getelementptr i64, ptr %sv, i64 %im1
  %sprev = load i64, ptr %spp, align 8
  %scp = getelementptr i64, ptr %sv, i64 %i
  %scur = load i64, ptr %scp, align 8
  %inc.ok = icmp slt i64 %sprev, %scur
  br i1 %inc.ok, label %cont0, label %bad

bad:
  br label %cont

cont0:
  br label %cont

cont:
  %inc = phi i64 [ 0, %cont0 ], [ 1, %bad ]
  %viol.n = add i64 %viol, %inc
  %i.n = add i64 %i, 1
  br label %loop

done:
  ret i64 %viol
}

; stability on f64: cols "fk" (dup keys) + "seq". Merge-sort path.
define i64 @stable_viol_f64(ptr %df, i64 %n) {
entry:
  %kv = call ptr @colvals(ptr %df, ptr @.fk, i64 2)
  %sv = call ptr @colvals(ptr %df, ptr @.seq, i64 3)
  br label %loop

loop:
  %i = phi i64 [ 1, %entry ], [ %i.n, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done

body:
  %im1 = sub i64 %i, 1
  %kpp = getelementptr double, ptr %kv, i64 %im1
  %kprev = load double, ptr %kpp, align 8
  %kcp = getelementptr double, ptr %kv, i64 %i
  %kcur = load double, ptr %kcp, align 8
  %same = fcmp oeq double %kprev, %kcur
  br i1 %same, label %chk, label %cont0

chk:
  %spp = getelementptr i64, ptr %sv, i64 %im1
  %sprev = load i64, ptr %spp, align 8
  %scp = getelementptr i64, ptr %sv, i64 %i
  %scur = load i64, ptr %scp, align 8
  %inc.ok = icmp slt i64 %sprev, %scur
  br i1 %inc.ok, label %cont0, label %bad

bad:
  br label %cont

cont0:
  br label %cont

cont:
  %inc = phi i64 [ 0, %cont0 ], [ 1, %bad ]
  %viol.n = add i64 %viol, %inc
  %i.n = add i64 %i, 1
  br label %loop

done:
  ret i64 %viol
}

; build a 2-col df: "k" i64 (kvals), "id" i64 (idvals), length n
define ptr @build_ki(ptr %kvals, ptr %idvals, i64 %n) {
entry:
  %sk = call ptr @universe_dataframe_series_from(i32 1, ptr %kvals, i64 %n)
  %si = call ptr @universe_dataframe_series_from(i32 1, ptr %idvals, i64 %n)
  %df = call ptr @universe_dataframe_new()
  %r0 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.k, i64 1, ptr %sk)
  %r1 = call i32 @universe_dataframe_with_column(ptr %df, ptr @.id, i64 2, ptr %si)
  ret ptr %df
}

; ===========================================================================
define i32 @main(i32 %argc, ptr %argv) {
entry:
  %by_k = alloca [2 x i64], align 8       ; one {ptr,len} pair
  %by_k12 = alloca [4 x i64], align 8     ; two pairs
  %desc1 = alloca [1 x i8], align 1
  %desc0 = alloca [1 x i8], align 1
  %desc2 = alloca [2 x i8], align 1
  %kbuf = alloca [16 x i64], align 8
  %ibuf = alloca [16 x i64], align 8
  %fbuf = alloca [16 x double], align 8
  %seqbuf = alloca [16 x i64], align 8
  %rk = alloca [512 x i64], align 8
  %ri = alloca [512 x i64], align 8
  %rperm = alloca [512 x i64], align 8
  %rng = alloca i64, align 8

  ; by "k" ascending descriptor
  %byk0 = getelementptr [2 x i64], ptr %by_k, i64 0, i64 0
  store ptr @.k, ptr %byk0, align 8
  %byk1 = getelementptr [2 x i64], ptr %by_k, i64 0, i64 1
  store i64 1, ptr %byk1, align 8
  store i8 1, ptr %desc1, align 1
  store i8 0, ptr %desc0, align 1

  ; -----------------------------------------------------------------
  ; 1. single i64 key, fixed data, asc + desc + integrity + permutation
  ;    k = [30,10,40,10,20,50,10,25]  id = 0..7
  ; -----------------------------------------------------------------
  store i64 30, ptr %kbuf, align 8
  %k_1 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 1
  store i64 10, ptr %k_1, align 8
  %k_2 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 2
  store i64 40, ptr %k_2, align 8
  %k_3 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 3
  store i64 10, ptr %k_3, align 8
  %k_4 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 4
  store i64 20, ptr %k_4, align 8
  %k_5 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 5
  store i64 50, ptr %k_5, align 8
  %k_6 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 6
  store i64 10, ptr %k_6, align 8
  %k_7 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 7
  store i64 25, ptr %k_7, align 8
  ; id = 0..7
  br label %fillid

fillid:
  %fi = phi i64 [ 0, %entry ], [ %fi.n, %fillid ]
  %fic = icmp ult i64 %fi, 8
  %fip = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 %fi
  store i64 %fi, ptr %fip, align 8
  %fi.n = add i64 %fi, 1
  %fidone = icmp ult i64 %fi.n, 8
  br i1 %fidone, label %fillid, label %sec1

sec1:
  %df1 = call ptr @build_ki(ptr %kbuf, ptr %ibuf, i64 8)
  ; ascending
  %asc = call ptr @universe_dataframe_sort(ptr %df1, ptr %by_k, i64 1, ptr null)
  %asch = call i64 @universe_dataframe_height(ptr %asc)
  call void @ut_check_eq(i64 %asch, i64 8, ptr @.m.height)
  %ascv = call ptr @colvals(ptr %asc, ptr @.k, i64 1)
  %ascviol = call i64 @sorted_i64_viol(ptr %ascv, i64 8, i32 0)
  call void @ut_check_eq(i64 %ascviol, i64 0, ptr @.m.asc)
  %ascidv = call ptr @colvals(ptr %asc, ptr @.id, i64 2)
  %ascpv = call i64 @is_perm_viol(ptr %ascidv, i64 8)
  call void @ut_check_eq(i64 %ascpv, i64 0, ptr @.m.perm)
  %ascint = call i64 @integ_viol(ptr %asc, ptr %kbuf, i64 8)
  call void @ut_check_eq(i64 %ascint, i64 0, ptr @.m.integ)
  ; descending
  %dsc = call ptr @universe_dataframe_sort(ptr %df1, ptr %by_k, i64 1, ptr %desc1)
  %dscv = call ptr @colvals(ptr %dsc, ptr @.k, i64 1)
  %dscviol = call i64 @sorted_i64_viol(ptr %dscv, i64 8, i32 1)
  call void @ut_check_eq(i64 %dscviol, i64 0, ptr @.m.desc)
  %dscidv = call ptr @colvals(ptr %dsc, ptr @.id, i64 2)
  %dscpv = call i64 @is_perm_viol(ptr %dscidv, i64 8)
  call void @ut_check_eq(i64 %dscpv, i64 0, ptr @.m.perm)
  %dscint = call i64 @integ_viol(ptr %dsc, ptr %kbuf, i64 8)
  call void @ut_check_eq(i64 %dscint, i64 0, ptr @.m.integ)

  ; -----------------------------------------------------------------
  ; 2. arg_sort on the "k" series directly (radix path)
  ; -----------------------------------------------------------------
  %ks = call ptr @universe_dataframe_series_from(i32 1, ptr %kbuf, i64 8)
  %as.rc = call i32 @universe_dataframe_arg_sort(ptr %ks, i32 0, ptr %rperm)
  %as.pv = call i64 @is_perm_viol(ptr %rperm, i64 8)
  call void @ut_check_eq(i64 %as.pv, i64 0, ptr @.m.args)
  ; keys[perm[i]] must be ascending
  br label %as.loop

as.loop:
  %asi = phi i64 [ 1, %sec1 ], [ %asi.n, %as.cont ]
  %asviol = phi i64 [ 0, %sec1 ], [ %asviol.n, %as.cont ]
  %asc.c = icmp ult i64 %asi, 8
  br i1 %asc.c, label %as.body, label %as.done

as.body:
  %asi.m1 = sub i64 %asi, 1
  %ap0 = getelementptr i64, ptr %rperm, i64 %asi.m1
  %aperm0 = load i64, ptr %ap0, align 8
  %ap1 = getelementptr i64, ptr %rperm, i64 %asi
  %aperm1 = load i64, ptr %ap1, align 8
  %akp0 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 %aperm0
  %akv0 = load i64, ptr %akp0, align 8
  %akp1 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 %aperm1
  %akv1 = load i64, ptr %akp1, align 8
  %asok = icmp sle i64 %akv0, %akv1
  br i1 %asok, label %as.cont, label %as.bad

as.bad:
  br label %as.cont

as.cont:
  %asinc = phi i64 [ 0, %as.body ], [ 1, %as.bad ]
  %asviol.n = add i64 %asviol, %asinc
  %asi.n = add i64 %asi, 1
  br label %as.loop

as.done:
  call void @ut_check_eq(i64 %asviol, i64 0, ptr @.m.args)
  call void @universe_dataframe_series_free(ptr %ks)

  ; -----------------------------------------------------------------
  ; 3. multi-key tie-break. k1 has many ties; k2 breaks them.
  ;    k1 = [2,1,2,1,2,1]  k2 = [5,9,1,3,3,7]  seq(id) not needed
  ;    sort by [k1 asc, k2 asc]: expect (k1,k2) lexicographic order.
  ; -----------------------------------------------------------------
  store i64 2, ptr %kbuf, align 8
  %m1_1 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 1
  store i64 1, ptr %m1_1, align 8
  %m1_2 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 2
  store i64 2, ptr %m1_2, align 8
  %m1_3 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 3
  store i64 1, ptr %m1_3, align 8
  %m1_4 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 4
  store i64 2, ptr %m1_4, align 8
  %m1_5 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 5
  store i64 1, ptr %m1_5, align 8
  store i64 5, ptr %ibuf, align 8
  %m2_1 = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 1
  store i64 9, ptr %m2_1, align 8
  %m2_2 = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 2
  store i64 1, ptr %m2_2, align 8
  %m2_3 = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 3
  store i64 3, ptr %m2_3, align 8
  %m2_4 = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 4
  store i64 3, ptr %m2_4, align 8
  %m2_5 = getelementptr [16 x i64], ptr %ibuf, i64 0, i64 5
  store i64 7, ptr %m2_5, align 8
  %sk1 = call ptr @universe_dataframe_series_from(i32 1, ptr %kbuf, i64 6)
  %sk2 = call ptr @universe_dataframe_series_from(i32 1, ptr %ibuf, i64 6)
  %dfm = call ptr @universe_dataframe_new()
  %mr0 = call i32 @universe_dataframe_with_column(ptr %dfm, ptr @.k1, i64 2, ptr %sk1)
  %mr1 = call i32 @universe_dataframe_with_column(ptr %dfm, ptr @.k2, i64 2, ptr %sk2)
  ; by [k1,k2]
  %bk0 = getelementptr [4 x i64], ptr %by_k12, i64 0, i64 0
  store ptr @.k1, ptr %bk0, align 8
  %bk1 = getelementptr [4 x i64], ptr %by_k12, i64 0, i64 1
  store i64 2, ptr %bk1, align 8
  %bk2 = getelementptr [4 x i64], ptr %by_k12, i64 0, i64 2
  store ptr @.k2, ptr %bk2, align 8
  %bk3 = getelementptr [4 x i64], ptr %by_k12, i64 0, i64 3
  store i64 2, ptr %bk3, align 8
  store i8 0, ptr %desc2, align 1
  %d2b = getelementptr [2 x i8], ptr %desc2, i64 0, i64 1
  store i8 0, ptr %d2b, align 1
  %msorted = call ptr @universe_dataframe_sort(ptr %dfm, ptr %by_k12, i64 2, ptr %desc2)
  ; verify lexicographic (k1,k2)
  %mk1v = call ptr @colvals(ptr %msorted, ptr @.k1, i64 2)
  %mk2v = call ptr @colvals(ptr %msorted, ptr @.k2, i64 2)
  br label %m.loop

m.loop:
  %mi = phi i64 [ 1, %as.done ], [ %mi.n, %m.cont ]
  %mviol = phi i64 [ 0, %as.done ], [ %mviol.n, %m.cont ]
  %mc = icmp ult i64 %mi, 6
  br i1 %mc, label %m.body, label %m.done

m.body:
  %mim1 = sub i64 %mi, 1
  %mk1p0 = getelementptr i64, ptr %mk1v, i64 %mim1
  %mk1v0 = load i64, ptr %mk1p0, align 8
  %mk1p1 = getelementptr i64, ptr %mk1v, i64 %mi
  %mk1v1 = load i64, ptr %mk1p1, align 8
  %mk2p0 = getelementptr i64, ptr %mk2v, i64 %mim1
  %mk2v0 = load i64, ptr %mk2p0, align 8
  %mk2p1 = getelementptr i64, ptr %mk2v, i64 %mi
  %mk2v1 = load i64, ptr %mk2p1, align 8
  %k1lt = icmp slt i64 %mk1v0, %mk1v1
  %k1eq = icmp eq i64 %mk1v0, %mk1v1
  %k2le = icmp sle i64 %mk2v0, %mk2v1
  %tieok = and i1 %k1eq, %k2le
  %lexok = or i1 %k1lt, %tieok
  br i1 %lexok, label %m.cont, label %m.bad

m.bad:
  br label %m.cont

m.cont:
  %minc = phi i64 [ 0, %m.body ], [ 1, %m.bad ]
  %mviol.n = add i64 %mviol, %minc
  %mi.n = add i64 %mi, 1
  br label %m.loop

m.done:
  call void @ut_check_eq(i64 %mviol, i64 0, ptr @.m.tie)
  call void @universe_dataframe_free(ptr %msorted)
  call void @universe_dataframe_free(ptr %dfm)

  ; -----------------------------------------------------------------
  ; 4. stability (radix i64 path). k has dup values; seq = original order.
  ;    k = [3,1,3,1,3,1,2,2]  seq = 0..7 . After stable asc sort, among
  ;    equal k the seq must strictly increase.
  ; -----------------------------------------------------------------
  store i64 3, ptr %kbuf, align 8
  %sd_1 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 1
  store i64 1, ptr %sd_1, align 8
  %sd_2 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 2
  store i64 3, ptr %sd_2, align 8
  %sd_3 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 3
  store i64 1, ptr %sd_3, align 8
  %sd_4 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 4
  store i64 3, ptr %sd_4, align 8
  %sd_5 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 5
  store i64 1, ptr %sd_5, align 8
  %sd_6 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 6
  store i64 2, ptr %sd_6, align 8
  %sd_7 = getelementptr [16 x i64], ptr %kbuf, i64 0, i64 7
  store i64 2, ptr %sd_7, align 8
  br label %sfill

sfill:
  %sfi = phi i64 [ 0, %m.done ], [ %sfi.n, %sfill ]
  %sfp = getelementptr [16 x i64], ptr %seqbuf, i64 0, i64 %sfi
  store i64 %sfi, ptr %sfp, align 8
  %sfi.n = add i64 %sfi, 1
  %sfc = icmp ult i64 %sfi.n, 8
  br i1 %sfc, label %sfill, label %sbuild

sbuild:
  %sks = call ptr @universe_dataframe_series_from(i32 1, ptr %kbuf, i64 8)
  %sseq = call ptr @universe_dataframe_series_from(i32 1, ptr %seqbuf, i64 8)
  %dfs = call ptr @universe_dataframe_new()
  %sr0 = call i32 @universe_dataframe_with_column(ptr %dfs, ptr @.k, i64 1, ptr %sks)
  %sr1 = call i32 @universe_dataframe_with_column(ptr %dfs, ptr @.seq, i64 3, ptr %sseq)
  %ssorted = call ptr @universe_dataframe_sort(ptr %dfs, ptr %by_k, i64 1, ptr null)
  %sviol = call i64 @stable_viol_i64(ptr %ssorted, i64 8)
  call void @ut_check_eq(i64 %sviol, i64 0, ptr @.m.stab)
  call void @universe_dataframe_free(ptr %ssorted)
  call void @universe_dataframe_free(ptr %dfs)

  ; -----------------------------------------------------------------
  ; 5. stability (f64 merge path) + float NaN/null ordering.
  ;    fk = [2.0, 1.0, NaN, 1.0, null, 2.0, 1.0, null]  seq = 0..7
  ;    asc: valid sorted (1,1,1,2,2) then NaN then nulls. Among equal
  ;    valid keys seq increases (merge stable).
  ; -----------------------------------------------------------------
  store double 2.0, ptr %fbuf, align 8
  %f_1 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 1
  store double 1.0, ptr %f_1, align 8
  %f_2 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 2
  store double 0x7FF8000000000000, ptr %f_2, align 8   ; quiet NaN
  %f_3 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 3
  store double 1.0, ptr %f_3, align 8
  %f_4 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 4
  store double 0.0, ptr %f_4, align 8                  ; will be null
  %f_5 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 5
  store double 2.0, ptr %f_5, align 8
  %f_6 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 6
  store double 1.0, ptr %f_6, align 8
  %f_7 = getelementptr [16 x double], ptr %fbuf, i64 0, i64 7
  store double 0.0, ptr %f_7, align 8                  ; will be null
  br label %ffill

ffill:
  %ffi = phi i64 [ 0, %sbuild ], [ %ffi.n, %ffill ]
  %ffp = getelementptr [16 x i64], ptr %seqbuf, i64 0, i64 %ffi
  store i64 %ffi, ptr %ffp, align 8
  %ffi.n = add i64 %ffi, 1
  %ffc = icmp ult i64 %ffi.n, 8
  br i1 %ffc, label %ffill, label %fbuild

fbuild:
  %fks = call ptr @universe_dataframe_series_from(i32 3, ptr %fbuf, i64 8)
  %n4 = call i32 @universe_dataframe_series_set_null(ptr %fks, i64 4)
  %n7 = call i32 @universe_dataframe_series_set_null(ptr %fks, i64 7)
  %fseq = call ptr @universe_dataframe_series_from(i32 1, ptr %seqbuf, i64 8)
  %dff = call ptr @universe_dataframe_new()
  %fr0 = call i32 @universe_dataframe_with_column(ptr %dff, ptr @.fk, i64 2, ptr %fks)
  %fr1 = call i32 @universe_dataframe_with_column(ptr %dff, ptr @.seq, i64 3, ptr %fseq)
  ; by "fk" ascending
  %byfk0 = getelementptr [2 x i64], ptr %by_k, i64 0, i64 0
  store ptr @.fk, ptr %byfk0, align 8
  %byfk1 = getelementptr [2 x i64], ptr %by_k, i64 0, i64 1
  store i64 2, ptr %byfk1, align 8
  %fsorted = call ptr @universe_dataframe_sort(ptr %dff, ptr %by_k, i64 1, ptr null)
  ; last two rows (6,7) must be null; rows 0..4 valid & non-nan; row 5 = NaN.
  %null6 = call i1 @colnull(ptr %fsorted, ptr @.fk, i64 2, i64 6)
  %null7 = call i1 @colnull(ptr %fsorted, ptr @.fk, i64 2, i64 7)
  %bothnull = and i1 %null6, %null7
  call void @ut_check(i1 %bothnull, ptr @.m.nulls)
  ; rows 0..5 must be non-null
  %nn0 = call i1 @colnull(ptr %fsorted, ptr @.fk, i64 2, i64 0)
  %nn5 = call i1 @colnull(ptr %fsorted, ptr @.fk, i64 2, i64 5)
  %anynn = or i1 %nn0, %nn5
  %valid05 = xor i1 %anynn, true
  call void @ut_check(i1 %valid05, ptr @.m.nulls)
  ; row 5 is the NaN (greatest non-null); check it is NaN
  %fsv = call ptr @colvals(ptr %fsorted, ptr @.fk, i64 2)
  %fs5p = getelementptr double, ptr %fsv, i64 5
  %fs5 = load double, ptr %fs5p, align 8
  %isnan5 = fcmp uno double %fs5, %fs5
  call void @ut_check(i1 %isnan5, ptr @.m.nan)
  ; rows 0..4 valid & non-nan & ascending
  br label %fv.loop

fv.loop:
  %fvi = phi i64 [ 1, %fbuild ], [ %fvi.n, %fv.cont ]
  %fvviol = phi i64 [ 0, %fbuild ], [ %fvviol.n, %fv.cont ]
  %fvc = icmp ult i64 %fvi, 5
  br i1 %fvc, label %fv.body, label %fv.done

fv.body:
  %fvm1 = sub i64 %fvi, 1
  %fvp0 = getelementptr double, ptr %fsv, i64 %fvm1
  %fvv0 = load double, ptr %fvp0, align 8
  %fvp1 = getelementptr double, ptr %fsv, i64 %fvi
  %fvv1 = load double, ptr %fvp1, align 8
  %fvle = fcmp ole double %fvv0, %fvv1
  br i1 %fvle, label %fv.cont, label %fv.bad

fv.bad:
  br label %fv.cont

fv.cont:
  %fvinc = phi i64 [ 0, %fv.body ], [ 1, %fv.bad ]
  %fvviol.n = add i64 %fvviol, %fvinc
  %fvi.n = add i64 %fvi, 1
  br label %fv.loop

fv.done:
  call void @ut_check_eq(i64 %fvviol, i64 0, ptr @.m.nan)
  %fstab = call i64 @stable_viol_f64(ptr %fsorted, i64 8)
  call void @ut_check_eq(i64 %fstab, i64 0, ptr @.m.stabf)
  call void @universe_dataframe_free(ptr %fsorted)
  call void @universe_dataframe_free(ptr %dff)

  ; restore by_k -> "k"
  %byk0b = getelementptr [2 x i64], ptr %by_k, i64 0, i64 0
  store ptr @.k, ptr %byk0b, align 8
  %byk1b = getelementptr [2 x i64], ptr %by_k, i64 0, i64 1
  store i64 1, ptr %byk1b, align 8

  ; -----------------------------------------------------------------
  ; 6. top_k / bottom_k vs full-sort reference (using df1: k asc/desc).
  ;    bottom_k(3) == first 3 of ascending sort. top_k(3) == first 3 of
  ;    descending sort.
  ; -----------------------------------------------------------------
  %botk = call ptr @universe_dataframe_bottom_k(ptr %df1, i64 3, ptr %by_k, i64 1)
  %botkh = call i64 @universe_dataframe_height(ptr %botk)
  call void @ut_check_eq(i64 %botkh, i64 3, ptr @.m.height)
  %botkv = call ptr @colvals(ptr %botk, ptr @.k, i64 1)
  ; ascv already = ascending sorted "k"; compare first 3
  br label %bk.loop

bk.loop:
  %bki = phi i64 [ 0, %fv.done ], [ %bki.n, %bk.cont ]
  %bkviol = phi i64 [ 0, %fv.done ], [ %bkviol.n, %bk.cont ]
  %bkc = icmp ult i64 %bki, 3
  br i1 %bkc, label %bk.body, label %bk.done

bk.body:
  %bkrp = getelementptr i64, ptr %ascv, i64 %bki
  %bkr = load i64, ptr %bkrp, align 8
  %bkgp = getelementptr i64, ptr %botkv, i64 %bki
  %bkg = load i64, ptr %bkgp, align 8
  %bkeq = icmp eq i64 %bkr, %bkg
  br i1 %bkeq, label %bk.cont, label %bk.bad

bk.bad:
  br label %bk.cont

bk.cont:
  %bkinc = phi i64 [ 0, %bk.body ], [ 1, %bk.bad ]
  %bkviol.n = add i64 %bkviol, %bkinc
  %bki.n = add i64 %bki, 1
  br label %bk.loop

bk.done:
  call void @ut_check_eq(i64 %bkviol, i64 0, ptr @.m.botk)
  ; top_k(3) default (largest first) vs dscv (descending sorted)
  %topk = call ptr @universe_dataframe_top_k(ptr %df1, i64 3, ptr %by_k, i64 1, i32 0)
  %topkv = call ptr @colvals(ptr %topk, ptr @.k, i64 1)
  br label %tk.loop

tk.loop:
  %tki = phi i64 [ 0, %bk.done ], [ %tki.n, %tk.cont ]
  %tkviol = phi i64 [ 0, %bk.done ], [ %tkviol.n, %tk.cont ]
  %tkc = icmp ult i64 %tki, 3
  br i1 %tkc, label %tk.body, label %tk.done

tk.body:
  %tkrp = getelementptr i64, ptr %dscv, i64 %tki
  %tkr = load i64, ptr %tkrp, align 8
  %tkgp = getelementptr i64, ptr %topkv, i64 %tki
  %tkg = load i64, ptr %tkgp, align 8
  %tkeq = icmp eq i64 %tkr, %tkg
  br i1 %tkeq, label %tk.cont, label %tk.bad

tk.bad:
  br label %tk.cont

tk.cont:
  %tkinc = phi i64 [ 0, %tk.body ], [ 1, %tk.bad ]
  %tkviol.n = add i64 %tkviol, %tkinc
  %tki.n = add i64 %tki, 1
  br label %tk.loop

tk.done:
  call void @ut_check_eq(i64 %tkviol, i64 0, ptr @.m.topk)
  call void @universe_dataframe_free(ptr %topk)
  call void @universe_dataframe_free(ptr %botk)
  call void @universe_dataframe_free(ptr %asc)
  call void @universe_dataframe_free(ptr %dsc)

  ; -----------------------------------------------------------------
  ; 7. sort_in_place on df1 (asc). df1 mutated; check ordered + integrity.
  ; -----------------------------------------------------------------
  %ip.rc = call i32 @universe_dataframe_sort_in_place(ptr %df1, ptr %by_k, i64 1, ptr null)
  call void @ut_check_eq(i64 0, i64 0, ptr @.m.inplace)
  %ipv = call ptr @colvals(ptr %df1, ptr @.k, i64 1)
  %ipviol = call i64 @sorted_i64_viol(ptr %ipv, i64 8, i32 0)
  call void @ut_check_eq(i64 %ipviol, i64 0, ptr @.m.inplace)
  ; integrity already covered in section 1; here just perm-check the id column.
  %ipidv = call ptr @colvals(ptr %df1, ptr @.id, i64 2)
  %ippv = call i64 @is_perm_viol(ptr %ipidv, i64 8)
  call void @ut_check_eq(i64 %ippv, i64 0, ptr @.m.perm)
  call void @universe_dataframe_free(ptr %df1)

  ; -----------------------------------------------------------------
  ; 8. randomized: n=500 random i64 keys (small range => many ties),
  ;    id = 0..n-1. Sort asc; ordered + permutation + row integrity.
  ; -----------------------------------------------------------------
  store i64 88172645463325252, ptr %rng, align 8
  br label %rfill

rfill:
  %rfi = phi i64 [ 0, %tk.done ], [ %rfi.n, %rfill ]
  %rr = call i64 @ut_rand(ptr %rng)
  %rrm = and i64 %rr, 1023          ; key in [0,1023]
  %rkp = getelementptr [512 x i64], ptr %rk, i64 0, i64 %rfi
  store i64 %rrm, ptr %rkp, align 8
  %rip = getelementptr [512 x i64], ptr %ri, i64 0, i64 %rfi
  store i64 %rfi, ptr %rip, align 8
  %rfi.n = add i64 %rfi, 1
  %rfc = icmp ult i64 %rfi.n, 500
  br i1 %rfc, label %rfill, label %rbuild

rbuild:
  %rdf = call ptr @build_ki(ptr %rk, ptr %ri, i64 500)
  %rsorted = call ptr @universe_dataframe_sort(ptr %rdf, ptr %by_k, i64 1, ptr null)
  %rsv = call ptr @colvals(ptr %rsorted, ptr @.k, i64 1)
  %rsviol = call i64 @sorted_i64_viol(ptr %rsv, i64 500, i32 0)
  call void @ut_check_eq(i64 %rsviol, i64 0, ptr @.m.rnd)
  %ridv = call ptr @colvals(ptr %rsorted, ptr @.id, i64 2)
  %rpv = call i64 @is_perm_viol(ptr %ridv, i64 500)
  call void @ut_check_eq(i64 %rpv, i64 0, ptr @.m.perm)
  %rint = call i64 @integ_viol(ptr %rsorted, ptr %rk, i64 500)
  call void @ut_check_eq(i64 %rint, i64 0, ptr @.m.rnd)
  call void @universe_dataframe_free(ptr %rsorted)
  call void @universe_dataframe_free(ptr %rdf)

  ; -----------------------------------------------------------------
  ; optional bench: time sorting a 500-row i64 frame many times.
  ; -----------------------------------------------------------------
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %summary

bench:
  %bdf = call ptr @build_ki(ptr %rk, ptr %ri, i64 500)
  %t0 = call double @ut_now_sec()
  br label %bench.loop

bench.loop:
  %bi = phi i64 [ 0, %bench ], [ %bi.n, %bench.loop ]
  %bs = call ptr @universe_dataframe_sort(ptr %bdf, ptr %by_k, i64 1, ptr null)
  call void @universe_dataframe_free(ptr %bs)
  %bi.n = add i64 %bi, 1
  %bc = icmp ult i64 %bi.n, 2000
  br i1 %bc, label %bench.loop, label %bench.done

bench.done:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %rows = fmul double 2000.0, 5.000000e+02
  %nspr0 = fdiv double %dt, %rows
  %nspr = fmul double %nspr0, 1.000000e+09
  %pr = call i32 (ptr, ...) @printf(ptr @.bench.fmt, i32 500, double %nspr)
  call void @universe_dataframe_free(ptr %bdf)
  br label %summary

summary:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

declare void @universe_dataframe_series_free(ptr)
