; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
;
; Tests for src/dataframe/join.ll — relational hash join.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()

declare ptr @malloc(i64)
declare void @free(ptr)

declare ptr @universe_dataframe_new()
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare i32 @universe_dataframe_get_column_index(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare void @universe_dataframe_free(ptr)

declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare i64 @universe_dataframe_series_len(ptr)
declare ptr @universe_dataframe_series_values(ptr)
declare i64 @universe_dataframe_series_null_count(ptr)
declare i32 @universe_dataframe_series_is_null(ptr, i64, ptr)

declare ptr @universe_dataframe_join(ptr, ptr, ptr, i64, ptr, i64, i32)
declare ptr @universe_dataframe_inner_join(ptr, ptr, ptr, i64)
declare ptr @universe_dataframe_left_join(ptr, ptr, ptr, i64)
declare ptr @universe_dataframe_outer_join(ptr, ptr, ptr, i64)
declare ptr @universe_dataframe_cross_join(ptr, ptr)

; column names
@.n_id = private constant [2 x i8] c"id"
@.n_lv = private constant [2 x i8] c"lv"
@.n_rv = private constant [2 x i8] c"rv"
@.n_k1 = private constant [2 x i8] c"k1"
@.n_k2 = private constant [2 x i8] c"k2"
@.n_v  = private constant [1 x i8] c"v"
@.n_w  = private constant [1 x i8] c"w"
@.n_idr = private constant [8 x i8] c"id_right"

; data
@.Lid = private constant [5 x i64] [i64 1, i64 2, i64 3, i64 2, i64 5]
@.Lv  = private constant [5 x i64] [i64 10, i64 20, i64 30, i64 40, i64 50]
@.Rid = private constant [4 x i64] [i64 2, i64 3, i64 3, i64 6]
@.Rv  = private constant [4 x i64] [i64 200, i64 300, i64 301, i64 600]

@.Lk1 = private constant [3 x i64] [i64 1, i64 1, i64 2]
@.Lk2 = private constant [3 x i64] [i64 1, i64 2, i64 2]
@.Lv2 = private constant [3 x i64] [i64 100, i64 200, i64 300]
@.Rk1 = private constant [3 x i64] [i64 1, i64 2, i64 1]
@.Rk2 = private constant [3 x i64] [i64 1, i64 2, i64 3]
@.Rw  = private constant [3 x i64] [i64 7, i64 8, i64 9]

@.Dup2 = private constant [2 x i64] [i64 7, i64 7]
@.DupA = private constant [2 x i64] [i64 1, i64 2]
@.DupB = private constant [2 x i64] [i64 3, i64 4]
@.Nm   = private constant [2 x i64] [i64 100, i64 101]
@.NmV  = private constant [2 x i64] [i64 1, i64 2]

; messages
@.m.inh = private constant [16 x i8] c"inner height   \00"
@.m.inw = private constant [16 x i8] c"inner width    \00"
@.m.inlv = private constant [16 x i8] c"inner lv sum   \00"
@.m.inrv = private constant [16 x i8] c"inner rv sum   \00"
@.m.lfh = private constant [16 x i8] c"left height    \00"
@.m.lflv = private constant [16 x i8] c"left lv sum    \00"
@.m.lfrv = private constant [16 x i8] c"left rv sum    \00"
@.m.lfnc = private constant [16 x i8] c"left rv nulls  \00"
@.m.oth = private constant [16 x i8] c"outer height   \00"
@.m.otrv = private constant [16 x i8] c"outer rv sum   \00"
@.m.otidn = private constant [16 x i8] c"outer id nulls \00"
@.m.crh = private constant [16 x i8] c"cross height   \00"
@.m.crw = private constant [16 x i8] c"cross width    \00"
@.m.cridr = private constant [17 x i8] c"cross id_right  \00"
@.m.mkh = private constant [16 x i8] c"multikey height\00"
@.m.mkw = private constant [16 x i8] c"multikey width \00"
@.m.mkv = private constant [16 x i8] c"multikey v sum \00"
@.m.mkw2 = private constant [16 x i8] c"multikey w sum \00"
@.m.duph = private constant [16 x i8] c"dup height     \00"
@.m.nmh = private constant [16 x i8] c"nomatch height \00"
@.m.rndh = private constant [16 x i8] c"rand height    \00"
@.m.rnds = private constant [16 x i8] c"rand sum       \00"
@.m.nulljoin = private constant [16 x i8] c"null-arg join  \00"

; ---- helpers ----

define internal ptr @t_ser(ptr %arr, i64 %n) {
entry:
  %s = call ptr @universe_dataframe_series_from(i32 1, ptr %arr, i64 %n)
  ret ptr %s
}

define internal ptr @t_df2(ptr %n0, i64 %l0, ptr %s0, ptr %n1, i64 %l1, ptr %s1) {
entry:
  %df = call ptr @universe_dataframe_new()
  %r0 = call i32 @universe_dataframe_with_column(ptr %df, ptr %n0, i64 %l0, ptr %s0)
  %r1 = call i32 @universe_dataframe_with_column(ptr %df, ptr %n1, i64 %l1, ptr %s1)
  ret ptr %df
}

define internal ptr @t_df3(ptr %n0, i64 %l0, ptr %s0, ptr %n1, i64 %l1, ptr %s1, ptr %n2, i64 %l2, ptr %s2) {
entry:
  %df = call ptr @universe_dataframe_new()
  %r0 = call i32 @universe_dataframe_with_column(ptr %df, ptr %n0, i64 %l0, ptr %s0)
  %r1 = call i32 @universe_dataframe_with_column(ptr %df, ptr %n1, i64 %l1, ptr %s1)
  %r2 = call i32 @universe_dataframe_with_column(ptr %df, ptr %n2, i64 %l2, ptr %s2)
  ret ptr %df
}

; sum i64 values of a series (raw; nulls read as 0 after gather memset)
define internal i64 @t_sum(ptr %s) {
entry:
  %len = call i64 @universe_dataframe_series_len(ptr %s)
  %vals = call ptr @universe_dataframe_series_values(ptr %s)
  br label %loop
loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %body ]
  %acc = phi i64 [ 0, %entry ], [ %accn, %body ]
  %c = icmp ult i64 %k, %len
  br i1 %c, label %body, label %done
body:
  %p = getelementptr inbounds i64, ptr %vals, i64 %k
  %v = load i64, ptr %p, align 8
  %accn = add i64 %acc, %v
  %kn = add i64 %k, 1
  br label %loop
done:
  ret i64 %acc
}

; write a {ptr,i64} name descriptor at buf
define internal void @t_on(ptr %buf, ptr %nm, i64 %nl) {
entry:
  store ptr %nm, ptr %buf, align 8
  %p8 = getelementptr inbounds i8, ptr %buf, i64 8
  store i64 %nl, ptr %p8, align 8
  ret void
}

define i32 @main() {
entry:
  %on1 = alloca [16 x i8], align 8
  %on2 = alloca [32 x i8], align 8
  %rng = alloca i64, align 8

  ; ---- build main frames: L(id,lv), R(id,rv) ----
  %Lid.s = call ptr @t_ser(ptr @.Lid, i64 5)
  %Lv.s  = call ptr @t_ser(ptr @.Lv,  i64 5)
  %L = call ptr @t_df2(ptr @.n_id, i64 2, ptr %Lid.s, ptr @.n_lv, i64 2, ptr %Lv.s)
  %Rid.s = call ptr @t_ser(ptr @.Rid, i64 4)
  %Rv.s  = call ptr @t_ser(ptr @.Rv,  i64 4)
  %R = call ptr @t_df2(ptr @.n_id, i64 2, ptr %Rid.s, ptr @.n_rv, i64 2, ptr %Rv.s)

  ; on-array for single key "id"
  call void @t_on(ptr %on1, ptr @.n_id, i64 2)

  ; ---- inner join ----
  %inner = call ptr @universe_dataframe_inner_join(ptr %L, ptr %R, ptr %on1, i64 1)
  %inh = call i64 @universe_dataframe_height(ptr %inner)
  call void @ut_check_eq(i64 %inh, i64 4, ptr @.m.inh)
  %inw = call i64 @universe_dataframe_width(ptr %inner)
  call void @ut_check_eq(i64 %inw, i64 3, ptr @.m.inw)
  %in.lv = call ptr @universe_dataframe_select_at_idx(ptr %inner, i64 1)
  %in.lvsum = call i64 @t_sum(ptr %in.lv)
  call void @ut_check_eq(i64 %in.lvsum, i64 120, ptr @.m.inlv)
  %in.rv = call ptr @universe_dataframe_select_at_idx(ptr %inner, i64 2)
  %in.rvsum = call i64 @t_sum(ptr %in.rv)
  call void @ut_check_eq(i64 %in.rvsum, i64 1001, ptr @.m.inrv)
  call void @universe_dataframe_free(ptr %inner)

  ; ---- left join ----
  %left = call ptr @universe_dataframe_left_join(ptr %L, ptr %R, ptr %on1, i64 1)
  %lfh = call i64 @universe_dataframe_height(ptr %left)
  call void @ut_check_eq(i64 %lfh, i64 6, ptr @.m.lfh)
  %lf.lv = call ptr @universe_dataframe_select_at_idx(ptr %left, i64 1)
  %lf.lvsum = call i64 @t_sum(ptr %lf.lv)
  call void @ut_check_eq(i64 %lf.lvsum, i64 180, ptr @.m.lflv)
  %lf.rv = call ptr @universe_dataframe_select_at_idx(ptr %left, i64 2)
  %lf.rvsum = call i64 @t_sum(ptr %lf.rv)
  call void @ut_check_eq(i64 %lf.rvsum, i64 1001, ptr @.m.lfrv)
  %lf.rvnc = call i64 @universe_dataframe_series_null_count(ptr %lf.rv)
  call void @ut_check_eq(i64 %lf.rvnc, i64 2, ptr @.m.lfnc)
  call void @universe_dataframe_free(ptr %left)

  ; ---- outer join ----
  %outer = call ptr @universe_dataframe_outer_join(ptr %L, ptr %R, ptr %on1, i64 1)
  %oth = call i64 @universe_dataframe_height(ptr %outer)
  call void @ut_check_eq(i64 %oth, i64 7, ptr @.m.oth)
  %ot.rv = call ptr @universe_dataframe_select_at_idx(ptr %outer, i64 2)
  %ot.rvsum = call i64 @t_sum(ptr %ot.rv)
  call void @ut_check_eq(i64 %ot.rvsum, i64 1601, ptr @.m.otrv)
  %ot.id = call ptr @universe_dataframe_select_at_idx(ptr %outer, i64 0)
  %ot.idnc = call i64 @universe_dataframe_series_null_count(ptr %ot.id)
  call void @ut_check_eq(i64 %ot.idnc, i64 1, ptr @.m.otidn)
  call void @universe_dataframe_free(ptr %outer)

  ; ---- cross join ----
  %cross = call ptr @universe_dataframe_cross_join(ptr %L, ptr %R)
  %crh = call i64 @universe_dataframe_height(ptr %cross)
  call void @ut_check_eq(i64 %crh, i64 20, ptr @.m.crh)
  %crw = call i64 @universe_dataframe_width(ptr %cross)
  call void @ut_check_eq(i64 %crw, i64 4, ptr @.m.crw)
  ; right "id" must have been renamed to "id_right"
  %idridx = alloca i64, align 8
  %crfind = call i32 @universe_dataframe_get_column_index(ptr %cross, ptr @.n_idr, i64 8, ptr %idridx)
  %crfound = icmp eq i32 %crfind, 0
  call void @ut_check(i1 %crfound, ptr @.m.cridr)
  call void @universe_dataframe_free(ptr %cross)

  ; ---- multi-key join ----
  %Lk1.s = call ptr @t_ser(ptr @.Lk1, i64 3)
  %Lk2.s = call ptr @t_ser(ptr @.Lk2, i64 3)
  %Lv2.s = call ptr @t_ser(ptr @.Lv2, i64 3)
  %ML = call ptr @t_df3(ptr @.n_k1, i64 2, ptr %Lk1.s, ptr @.n_k2, i64 2, ptr %Lk2.s, ptr @.n_v, i64 1, ptr %Lv2.s)
  %Rk1.s = call ptr @t_ser(ptr @.Rk1, i64 3)
  %Rk2.s = call ptr @t_ser(ptr @.Rk2, i64 3)
  %Rw.s  = call ptr @t_ser(ptr @.Rw,  i64 3)
  %MR = call ptr @t_df3(ptr @.n_k1, i64 2, ptr %Rk1.s, ptr @.n_k2, i64 2, ptr %Rk2.s, ptr @.n_w, i64 1, ptr %Rw.s)
  call void @t_on(ptr %on2, ptr @.n_k1, i64 2)
  %on2b = getelementptr inbounds i8, ptr %on2, i64 16
  call void @t_on(ptr %on2b, ptr @.n_k2, i64 2)
  %mk = call ptr @universe_dataframe_inner_join(ptr %ML, ptr %MR, ptr %on2, i64 2)
  %mkh = call i64 @universe_dataframe_height(ptr %mk)
  call void @ut_check_eq(i64 %mkh, i64 2, ptr @.m.mkh)
  %mkw = call i64 @universe_dataframe_width(ptr %mk)
  ; k1,k2,v,w = 4
  call void @ut_check_eq(i64 %mkw, i64 4, ptr @.m.mkw)
  %mk.v = call ptr @universe_dataframe_select_at_idx(ptr %mk, i64 2)
  %mk.vsum = call i64 @t_sum(ptr %mk.v)
  call void @ut_check_eq(i64 %mk.vsum, i64 400, ptr @.m.mkv)
  %mk.w = call ptr @universe_dataframe_select_at_idx(ptr %mk, i64 3)
  %mk.wsum = call i64 @t_sum(ptr %mk.w)
  call void @ut_check_eq(i64 %mk.wsum, i64 15, ptr @.m.mkw2)
  call void @universe_dataframe_free(ptr %mk)
  call void @universe_dataframe_free(ptr %ML)
  call void @universe_dataframe_free(ptr %MR)

  ; ---- duplicate keys on BOTH sides (2x2 cartesian) ----
  %Dl.id = call ptr @t_ser(ptr @.Dup2, i64 2)
  %Dl.v  = call ptr @t_ser(ptr @.DupA, i64 2)
  %DL = call ptr @t_df2(ptr @.n_id, i64 2, ptr %Dl.id, ptr @.n_lv, i64 2, ptr %Dl.v)
  %Dr.id = call ptr @t_ser(ptr @.Dup2, i64 2)
  %Dr.v  = call ptr @t_ser(ptr @.DupB, i64 2)
  %DR = call ptr @t_df2(ptr @.n_id, i64 2, ptr %Dr.id, ptr @.n_rv, i64 2, ptr %Dr.v)
  %dup = call ptr @universe_dataframe_inner_join(ptr %DL, ptr %DR, ptr %on1, i64 1)
  %duph = call i64 @universe_dataframe_height(ptr %dup)
  call void @ut_check_eq(i64 %duph, i64 4, ptr @.m.duph)
  call void @universe_dataframe_free(ptr %dup)
  call void @universe_dataframe_free(ptr %DL)
  call void @universe_dataframe_free(ptr %DR)

  ; ---- no-match inner ----
  %Nm.id = call ptr @t_ser(ptr @.Nm, i64 2)
  %Nm.v  = call ptr @t_ser(ptr @.NmV, i64 2)
  %NL = call ptr @t_df2(ptr @.n_id, i64 2, ptr %Nm.id, ptr @.n_lv, i64 2, ptr %Nm.v)
  %nomatch = call ptr @universe_dataframe_inner_join(ptr %NL, ptr %R, ptr %on1, i64 1)
  %nmh = call i64 @universe_dataframe_height(ptr %nomatch)
  call void @ut_check_eq(i64 %nmh, i64 0, ptr @.m.nmh)
  call void @universe_dataframe_free(ptr %nomatch)
  call void @universe_dataframe_free(ptr %NL)

  ; ---- null-arg join returns null ----
  %nj = call ptr @universe_dataframe_join(ptr null, ptr %R, ptr %on1, i64 1, ptr %on1, i64 1, i32 0)
  %njnull = icmp eq ptr %nj, null
  call void @ut_check(i1 %njnull, ptr @.m.nulljoin)

  call void @universe_dataframe_free(ptr %L)
  call void @universe_dataframe_free(ptr %R)

  ; ---- randomized cross-check vs brute-force nested loop ----
  ; N left rows, M right rows, id in [0,8)
  %N = add i64 0, 64
  %M = add i64 0, 48
  %La = call ptr @malloc(i64 512)   ; N*8
  %Lb = call ptr @malloc(i64 512)
  %Ra = call ptr @malloc(i64 384)   ; M*8
  %Rb = call ptr @malloc(i64 384)
  store i64 88172645463325252, ptr %rng, align 8

  ; fill left
  br label %fl.head
fl.head:
  %fli = phi i64 [ 0, %entry ], [ %flin, %fl.body ]
  %flc = icmp ult i64 %fli, %N
  br i1 %flc, label %fl.body, label %fr.head
fl.body:
  %r1 = call i64 @ut_rand(ptr %rng)
  %id1 = and i64 %r1, 7
  %lap = getelementptr inbounds i64, ptr %La, i64 %fli
  store i64 %id1, ptr %lap, align 8
  %r2 = call i64 @ut_rand(ptr %rng)
  %v1 = and i64 %r2, 1023
  %lbp = getelementptr inbounds i64, ptr %Lb, i64 %fli
  store i64 %v1, ptr %lbp, align 8
  %flin = add i64 %fli, 1
  br label %fl.head
fr.head:
  %fri = phi i64 [ 0, %fl.head ], [ %frin, %fr.body ]
  %frc = icmp ult i64 %fri, %M
  br i1 %frc, label %fr.body, label %bf.head
fr.body:
  %r3 = call i64 @ut_rand(ptr %rng)
  %id2 = and i64 %r3, 7
  %rap = getelementptr inbounds i64, ptr %Ra, i64 %fri
  store i64 %id2, ptr %rap, align 8
  %r4 = call i64 @ut_rand(ptr %rng)
  %v2 = and i64 %r4, 1023
  %rbp = getelementptr inbounds i64, ptr %Rb, i64 %fri
  store i64 %v2, ptr %rbp, align 8
  %frin = add i64 %fri, 1
  br label %fr.head

  ; brute-force count + sum(Lb[i]+Rb[j]) over matches
bf.head:
  %bi = phi i64 [ 0, %fr.head ], [ %bin, %bf.icont ]
  %bcnt = phi i64 [ 0, %fr.head ], [ %bcnt.i, %bf.icont ]
  %bsum = phi i64 [ 0, %fr.head ], [ %bsum.i, %bf.icont ]
  %bic = icmp ult i64 %bi, %N
  br i1 %bic, label %bf.iload, label %bf.done
bf.iload:
  %blap = getelementptr inbounds i64, ptr %La, i64 %bi
  %bla = load i64, ptr %blap, align 8
  %blbp = getelementptr inbounds i64, ptr %Lb, i64 %bi
  %blb = load i64, ptr %blbp, align 8
  br label %bf.jhead
bf.jhead:
  %bj = phi i64 [ 0, %bf.iload ], [ %bjn, %bf.jcont ]
  %bcnt.j = phi i64 [ %bcnt, %bf.iload ], [ %bcnt.jn, %bf.jcont ]
  %bsum.j = phi i64 [ %bsum, %bf.iload ], [ %bsum.jn, %bf.jcont ]
  %bjc = icmp ult i64 %bj, %M
  br i1 %bjc, label %bf.jbody, label %bf.icont
bf.jbody:
  %brap = getelementptr inbounds i64, ptr %Ra, i64 %bj
  %bra = load i64, ptr %brap, align 8
  %beq = icmp eq i64 %bla, %bra
  br i1 %beq, label %bf.match, label %bf.jcont
bf.match:
  %brbp = getelementptr inbounds i64, ptr %Rb, i64 %bj
  %brb = load i64, ptr %brbp, align 8
  %pairsum = add i64 %blb, %brb
  %bcnt.m = add i64 %bcnt.j, 1
  %bsum.m = add i64 %bsum.j, %pairsum
  br label %bf.jcont
bf.jcont:
  %bcnt.jn = phi i64 [ %bcnt.m, %bf.match ], [ %bcnt.j, %bf.jbody ]
  %bsum.jn = phi i64 [ %bsum.m, %bf.match ], [ %bsum.j, %bf.jbody ]
  %bjn = add i64 %bj, 1
  br label %bf.jhead
bf.icont:
  %bcnt.i = phi i64 [ %bcnt.j, %bf.jhead ]
  %bsum.i = phi i64 [ %bsum.j, %bf.jhead ]
  %bin = add i64 %bi, 1
  br label %bf.head
bf.done:
  ; build frames and inner join
  %RLid = call ptr @t_ser(ptr %La, i64 %N)
  %RLv  = call ptr @t_ser(ptr %Lb, i64 %N)
  %RLdf = call ptr @t_df2(ptr @.n_id, i64 2, ptr %RLid, ptr @.n_lv, i64 2, ptr %RLv)
  %RRid = call ptr @t_ser(ptr %Ra, i64 %M)
  %RRv  = call ptr @t_ser(ptr %Rb, i64 %M)
  %RRdf = call ptr @t_df2(ptr @.n_id, i64 2, ptr %RRid, ptr @.n_rv, i64 2, ptr %RRv)
  %rj = call ptr @universe_dataframe_inner_join(ptr %RLdf, ptr %RRdf, ptr %on1, i64 1)
  %rjh = call i64 @universe_dataframe_height(ptr %rj)
  call void @ut_check_eq(i64 %rjh, i64 %bcnt, ptr @.m.rndh)
  %rj.lv = call ptr @universe_dataframe_select_at_idx(ptr %rj, i64 1)
  %rj.lvsum = call i64 @t_sum(ptr %rj.lv)
  %rj.rv = call ptr @universe_dataframe_select_at_idx(ptr %rj, i64 2)
  %rj.rvsum = call i64 @t_sum(ptr %rj.rv)
  %rj.tot = add i64 %rj.lvsum, %rj.rvsum
  call void @ut_check_eq(i64 %rj.tot, i64 %bsum, ptr @.m.rnds)
  call void @universe_dataframe_free(ptr %rj)
  call void @universe_dataframe_free(ptr %RLdf)
  call void @universe_dataframe_free(ptr %RRdf)
  call void @free(ptr %La)
  call void @free(ptr %Lb)
  call void @free(ptr %Ra)
  call void @free(ptr %Rb)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
