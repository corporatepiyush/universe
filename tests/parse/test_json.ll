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

; Tests for universe_parse_json_*: valid docs (objects/arrays/nesting/unicode/
; surrogate/numbers), malformed docs each -> PARSE, depth-guard trip, a token
; sequence walk, unescape (simple/BMP/surrogate), number->double, and --bench.

declare ptr  @universe_parse_json_scanner_create(ptr, i64, i32)
declare void @universe_parse_json_scanner_destroy(ptr)
declare void @universe_parse_json_scanner_reset(ptr)
declare i32  @universe_parse_json_next(ptr, ptr)
declare i64  @universe_parse_json_error_offset(ptr)
declare i64  @universe_parse_json_unescape(ptr, ptr, i64)
declare i32  @universe_parse_json_number_double(ptr, i64, ptr)
declare i64  @universe_parse_json_scan_ws(ptr, i64, i64)
declare i64  @universe_parse_json_scan_ws_scalar(ptr, i64, i64)
declare i64  @universe_parse_json_scan_structural(ptr, i64, i64)
declare i64  @universe_parse_json_scan_structural_scalar(ptr, i64, i64)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

; ------------------------------------------------------------------- doc data
@v.obj  = private unnamed_addr constant [26 x i8] c"{\22a\22:1,\22b\22:[2,3],\22c\22:true}"
@v.nest = private unnamed_addr constant [15 x i8] c"[1,[2,[3,[4]]]]"
@v.uni  = private unnamed_addr constant [14 x i8] c"\22\5Cu0041\5Cu00e9\22"
@v.surr = private unnamed_addr constant [14 x i8] c"\22\5CuD83D\5CuDE00\22"
@v.nums = private unnamed_addr constant [21 x i8] c"[-1,2.5,3e2,1.5e-3,0]"
@v.walk = private unnamed_addr constant [31 x i8] c"{\22a\22:1,\22b\22:[true,null],\22c\22:\22x\22}"

@m.tcobj = private unnamed_addr constant [8 x i8]  c"{\22a\22:1,}"
@m.tcarr = private unnamed_addr constant [4 x i8]  c"[1,]"
@m.unstr = private unnamed_addr constant [4 x i8]  c"\22abc"
@m.unarr = private unnamed_addr constant [4 x i8]  c"[1,2"
@m.bade  = private unnamed_addr constant [6 x i8]  c"\22a\5Cqb\22"
@m.lone  = private unnamed_addr constant [8 x i8]  c"\22\5CuD800\22"
@m.n01   = private unnamed_addr constant [2 x i8]  c"01"
@m.ndot  = private unnamed_addr constant [2 x i8]  c"1."
@m.dot5  = private unnamed_addr constant [2 x i8]  c".5"
@m.ncol  = private unnamed_addr constant [7 x i8]  c"{\22a\22 1}"
@m.deep  = private unnamed_addr constant [16 x i8] c"[[[[[[[[[[[[[[[["

; unescape vectors (content between quotes)
@u.simple = private unnamed_addr constant [8 x i8]  c"a\5Cnb\5Ct\5C\22"
@e.simple = private unnamed_addr constant [5 x i8]  c"a\0Ab\09\22"
@u.uni    = private unnamed_addr constant [12 x i8] c"\5Cu0041\5Cu00e9"
@e.uni    = private unnamed_addr constant [3 x i8]  c"A\C3\A9"
@u.surr   = private unnamed_addr constant [12 x i8] c"\5CuD83D\5CuDE00"
@e.surr   = private unnamed_addr constant [4 x i8]  c"\F0\9F\98\80"

; number->double vectors
@n.a = private unnamed_addr constant [3 x i8] c"1.5"
@n.b = private unnamed_addr constant [4 x i8] c"-2.5"
@n.c = private unnamed_addr constant [3 x i8] c"2e3"
@n.d = private unnamed_addr constant [3 x i8] c"0.5"

@walk.exp = private unnamed_addr constant [11 x i32] [ i32 0, i32 9, i32 5, i32 9, i32 2, i32 6, i32 8, i32 3, i32 9, i32 4, i32 1 ]

; SIMD cross-check byte map: whitespace, structural chars, quote, backslash, and
; a few ordinary bytes — biased so structural/ws bytes land inside 16-byte chunks.
@sc.map = private unnamed_addr constant [16 x i8] [ i8 32, i8 9, i8 10, i8 13, i8 123, i8 125, i8 91, i8 93, i8 58, i8 44, i8 34, i8 92, i8 97, i8 98, i8 49, i8 50 ]
; adversarial fixed doc: quotes, escapes, embedded ws/structural, CRLF, control.
@sc.adv = private unnamed_addr constant [44 x i8] c"  {\22a\5Cn\22: [1,\09-2],\0D\0A \22x\5C\22y\22} , : \22\22 \5C\5C  end\00"
@sc.msg = private unnamed_addr constant [32 x i8] c"json simd scan == scalar oracle\00"

@sc.seed = internal global i64 0, align 8
@sc.viol = internal global i64 0, align 8

@g.types = internal global [64 x i32] zeroinitializer, align 16
@g.dst   = internal global [256 x i8] zeroinitializer, align 16
@g.json  = internal global [65536 x i8] zeroinitializer, align 16

@m.obj   = private unnamed_addr constant [17 x i8] c"valid object ok\0A\00"
@m.nest  = private unnamed_addr constant [17 x i8] c"valid nesting ok\00"
@m.uni   = private unnamed_addr constant [15 x i8] c"valid unicode\0A\00"
@m.surr  = private unnamed_addr constant [16 x i8] c"valid surrogate\00"
@m.nums  = private unnamed_addr constant [15 x i8] c"valid numbers\0A\00"
@m.malf  = private unnamed_addr constant [20 x i8] c"malformed -> PARSE\0A\00"
@m.deepm = private unnamed_addr constant [22 x i8] c"depth guard -> PARSE\0A\00"
@m.walkm = private unnamed_addr constant [18 x i8] c"token walk count\0A\00"
@m.walks = private unnamed_addr constant [16 x i8] c"token walk seq\0A\00"
@m.erof  = private unnamed_addr constant [18 x i8] c"error offset = 3\0A\00"
@m.uesc  = private unnamed_addr constant [17 x i8] c"unescape simple\0A\00"
@m.uuni  = private unnamed_addr constant [14 x i8] c"unescape bmp\0A\00"
@m.usur  = private unnamed_addr constant [19 x i8] c"unescape surrogate\00"
@m.numa  = private unnamed_addr constant [12 x i8] c"num 1.5 ok\0A\00"
@m.numb  = private unnamed_addr constant [13 x i8] c"num -2.5 ok\0A\00"
@m.numc  = private unnamed_addr constant [12 x i8] c"num 2e3 ok\0A\00"
@m.numd  = private unnamed_addr constant [12 x i8] c"num 0.5 ok\0A\00"
@json.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.json = private unnamed_addr constant [20 x i8] c"json scan int array\00"

; Parse a whole doc to END; return 0 if fully valid, else the error status.
define internal i32 @run(ptr %buf, i64 %len, i32 %md) {
entry:
  %sc = call ptr @universe_parse_json_scanner_create(ptr %buf, i64 %len, i32 %md)
  %tok = alloca [24 x i8], align 8
  br label %loop
loop:
  %r = call i32 @universe_parse_json_next(ptr %sc, ptr %tok)
  %err = icmp ne i32 %r, 0
  br i1 %err, label %ret, label %chk
chk:
  %ty = load i32, ptr %tok, align 4
  %end = icmp eq i32 %ty, 10
  br i1 %end, label %ok, label %loop
ok:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i32 0
ret:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i32 %r
}

; Walk a doc collecting token types into g.types; returns count, or -1 on error.
define internal i64 @walk(ptr %buf, i64 %len) {
entry:
  %sc = call ptr @universe_parse_json_scanner_create(ptr %buf, i64 %len, i32 64)
  %tok = alloca [24 x i8], align 8
  br label %loop
loop:
  %n = phi i64 [ 0, %entry ], [ %n.next, %store ]
  %r = call i32 @universe_parse_json_next(ptr %sc, ptr %tok)
  %err = icmp ne i32 %r, 0
  br i1 %err, label %fail, label %chk
chk:
  %ty = load i32, ptr %tok, align 4
  %end = icmp eq i32 %ty, 10
  br i1 %end, label %done, label %store
store:
  %slot = getelementptr inbounds nuw [64 x i32], ptr @g.types, i64 0, i64 %n
  store i32 %ty, ptr %slot, align 4
  %n.next = add nuw i64 %n, 1
  br label %loop
done:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i64 %n
fail:
  call void @universe_parse_json_scanner_destroy(ptr %sc)
  ret i64 -1
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- valid docs ----
  %r.obj = call i32 @run(ptr @v.obj, i64 26, i32 64)
  %ok.obj = icmp eq i32 %r.obj, 0
  call void @ut_check(i1 %ok.obj, ptr @m.obj)

  %r.nest = call i32 @run(ptr @v.nest, i64 15, i32 64)
  %ok.nest = icmp eq i32 %r.nest, 0
  call void @ut_check(i1 %ok.nest, ptr @m.nest)

  %r.uni = call i32 @run(ptr @v.uni, i64 14, i32 64)
  %ok.uni = icmp eq i32 %r.uni, 0
  call void @ut_check(i1 %ok.uni, ptr @m.uni)

  %r.surr = call i32 @run(ptr @v.surr, i64 14, i32 64)
  %ok.surr = icmp eq i32 %r.surr, 0
  call void @ut_check(i1 %ok.surr, ptr @m.surr)

  %r.nums = call i32 @run(ptr @v.nums, i64 21, i32 64)
  %ok.nums = icmp eq i32 %r.nums, 0
  call void @ut_check(i1 %ok.nums, ptr @m.nums)

  ; ---- malformed -> PARSE (aggregate) ----
  %e1 = call i32 @run(ptr @m.tcobj, i64 8, i32 64)
  %e2 = call i32 @run(ptr @m.tcarr, i64 4, i32 64)
  %e3 = call i32 @run(ptr @m.unstr, i64 4, i32 64)
  %e4 = call i32 @run(ptr @m.unarr, i64 4, i32 64)
  %e5 = call i32 @run(ptr @m.bade, i64 6, i32 64)
  %e6 = call i32 @run(ptr @m.lone, i64 8, i32 64)
  %e7 = call i32 @run(ptr @m.n01, i64 2, i32 64)
  %e8 = call i32 @run(ptr @m.ndot, i64 2, i32 64)
  %e9 = call i32 @run(ptr @m.dot5, i64 2, i32 64)
  %e10 = call i32 @run(ptr @m.ncol, i64 7, i32 64)
  ; count how many correctly returned 13
  %c1 = icmp eq i32 %e1, 13
  %c2 = icmp eq i32 %e2, 13
  %c3 = icmp eq i32 %e3, 13
  %c4 = icmp eq i32 %e4, 13
  %c5 = icmp eq i32 %e5, 13
  %c6 = icmp eq i32 %e6, 13
  %c7 = icmp eq i32 %e7, 13
  %c8 = icmp eq i32 %e8, 13
  %c9 = icmp eq i32 %e9, 13
  %c10 = icmp eq i32 %e10, 13
  %z1 = zext i1 %c1 to i64
  %z2 = zext i1 %c2 to i64
  %z3 = zext i1 %c3 to i64
  %z4 = zext i1 %c4 to i64
  %z5 = zext i1 %c5 to i64
  %z6 = zext i1 %c6 to i64
  %z7 = zext i1 %c7 to i64
  %z8 = zext i1 %c8 to i64
  %z9 = zext i1 %c9 to i64
  %z10 = zext i1 %c10 to i64
  %s1 = add i64 %z1, %z2
  %s2 = add i64 %s1, %z3
  %s3 = add i64 %s2, %z4
  %s4 = add i64 %s3, %z5
  %s5 = add i64 %s4, %z6
  %s6 = add i64 %s5, %z7
  %s7 = add i64 %s6, %z8
  %s8 = add i64 %s7, %z9
  %s9 = add i64 %s8, %z10
  call void @ut_check_eq(i64 %s9, i64 10, ptr @m.malf)

  ; ---- depth guard: 16 '[' with max_depth 4 -> PARSE ----
  %r.deep = call i32 @run(ptr @m.deep, i64 16, i32 4)
  %ok.deep = icmp eq i32 %r.deep, 13
  call void @ut_check(i1 %ok.deep, ptr @m.deepm)

  ; ---- token walk ----
  %wc = call i64 @walk(ptr @v.walk, i64 31)
  call void @ut_check_eq(i64 %wc, i64 11, ptr @m.walkm)
  ; compare types
  br label %wcmp.head
wcmp.head:
  %wi = phi i64 [ 0, %entry ], [ %wi.next, %wcmp.body ]
  %wmis = phi i64 [ 0, %entry ], [ %wmis.next, %wcmp.body ]
  %wdone = icmp uge i64 %wi, 11
  br i1 %wdone, label %wcmp.fin, label %wcmp.body
wcmp.body:
  %gp = getelementptr inbounds nuw [64 x i32], ptr @g.types, i64 0, i64 %wi
  %gv = load i32, ptr %gp, align 4
  %ep = getelementptr inbounds nuw [11 x i32], ptr @walk.exp, i64 0, i64 %wi
  %ev = load i32, ptr %ep, align 4
  %ne = icmp ne i32 %gv, %ev
  %neb = zext i1 %ne to i64
  %wmis.next = add i64 %wmis, %neb
  %wi.next = add nuw i64 %wi, 1
  br label %wcmp.head
wcmp.fin:
  call void @ut_check_eq(i64 %wmis, i64 0, ptr @m.walks)

  ; ---- error offset for bad escape "a\qb" (q at index 3) ----
  %sc.e = call ptr @universe_parse_json_scanner_create(ptr @m.bade, i64 6, i32 64)
  %tok.e = alloca [24 x i8], align 8
  %re = call i32 @universe_parse_json_next(ptr %sc.e, ptr %tok.e)
  %eoff = call i64 @universe_parse_json_error_offset(ptr %sc.e)
  call void @universe_parse_json_scanner_destroy(ptr %sc.e)
  %eoff.ok = icmp eq i64 %eoff, 3
  call void @ut_check(i1 %eoff.ok, ptr @m.erof)

  ; ---- unescape simple ----
  %us = call i64 @universe_parse_json_unescape(ptr @g.dst, ptr @u.simple, i64 8)
  %us.len = icmp eq i64 %us, 5
  %us.cmp = call i32 @memcmp(ptr @g.dst, ptr @e.simple, i64 5)
  %us.cok = icmp eq i32 %us.cmp, 0
  %us.ok = and i1 %us.len, %us.cok
  call void @ut_check(i1 %us.ok, ptr @m.uesc)

  ; ---- unescape BMP \u ----
  %uu = call i64 @universe_parse_json_unescape(ptr @g.dst, ptr @u.uni, i64 12)
  %uu.len = icmp eq i64 %uu, 3
  %uu.cmp = call i32 @memcmp(ptr @g.dst, ptr @e.uni, i64 3)
  %uu.cok = icmp eq i32 %uu.cmp, 0
  %uu.ok = and i1 %uu.len, %uu.cok
  call void @ut_check(i1 %uu.ok, ptr @m.uuni)

  ; ---- unescape surrogate pair ----
  %usr = call i64 @universe_parse_json_unescape(ptr @g.dst, ptr @u.surr, i64 12)
  %usr.len = icmp eq i64 %usr, 4
  %usr.cmp = call i32 @memcmp(ptr @g.dst, ptr @e.surr, i64 4)
  %usr.cok = icmp eq i32 %usr.cmp, 0
  %usr.ok = and i1 %usr.len, %usr.cok
  call void @ut_check(i1 %usr.ok, ptr @m.usur)

  ; ---- number -> double ----
  %od = alloca double, align 8
  %na.rc = call i32 @universe_parse_json_number_double(ptr @n.a, i64 3, ptr %od)
  %na.v = load double, ptr %od, align 8
  %na.eq = fcmp oeq double %na.v, 1.500000e+00
  call void @ut_check(i1 %na.eq, ptr @m.numa)

  %nb.rc = call i32 @universe_parse_json_number_double(ptr @n.b, i64 4, ptr %od)
  %nb.v = load double, ptr %od, align 8
  %nb.eq = fcmp oeq double %nb.v, -2.500000e+00
  call void @ut_check(i1 %nb.eq, ptr @m.numb)

  %nc.rc = call i32 @universe_parse_json_number_double(ptr @n.c, i64 3, ptr %od)
  %nc.v = load double, ptr %od, align 8
  %nc.eq = fcmp oeq double %nc.v, 2.000000e+03
  call void @ut_check(i1 %nc.eq, ptr @m.numc)

  %nd.rc = call i32 @universe_parse_json_number_double(ptr @n.d, i64 3, ptr %od)
  %nd.v = load double, ptr %od, align 8
  %nd.eq = fcmp oeq double %nd.v, 5.000000e-01
  call void @ut_check(i1 %nd.eq, ptr @m.numd)

  ; ---- SIMD structural-scan cross-check: vector path == scalar oracle ----
  ; fill a 2048-byte buffer with map[rand&15] (structural/ws bytes land inside
  ; and across 16-byte chunk boundaries), then for EVERY start position assert
  ; the vector scan_ws / scan_structural equal their scalar oracles. Also sweep
  ; the adversarial fixed doc (quotes, escapes, CRLF, control, embedded chars).
  store i64 2463534242, ptr @sc.seed, align 8
  br label %sc.fill
sc.fill:
  %fk = phi i64 [ 0, %wcmp.fin ], [ %fk.n, %sc.fillc ]
  %fdone = icmp uge i64 %fk, 2048
  br i1 %fdone, label %sc.chk0, label %sc.fillb
sc.fillb:
  %rv = call i64 @ut_rand(ptr @sc.seed)
  %ridx = and i64 %rv, 15
  %mp = getelementptr inbounds nuw [16 x i8], ptr @sc.map, i64 0, i64 %ridx
  %mb = load i8, ptr %mp, align 1
  %dp = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %fk
  store i8 %mb, ptr %dp, align 1
  br label %sc.fillc
sc.fillc:
  %fk.n = add nuw i64 %fk, 1
  br label %sc.fill
sc.chk0:
  store i64 0, ptr @sc.viol, align 8
  br label %sc.rloop
sc.rloop:
  %ri = phi i64 [ 0, %sc.chk0 ], [ %ri.n, %sc.rcont ]
  %rdone = icmp ugt i64 %ri, 2048
  br i1 %rdone, label %sc.aloop.pre, label %sc.rbody
sc.rbody:
  %ws.v = call i64 @universe_parse_json_scan_ws(ptr @g.json, i64 2048, i64 %ri)
  %ws.s = call i64 @universe_parse_json_scan_ws_scalar(ptr @g.json, i64 2048, i64 %ri)
  %ws.bad = icmp ne i64 %ws.v, %ws.s
  %st.v = call i64 @universe_parse_json_scan_structural(ptr @g.json, i64 2048, i64 %ri)
  %st.s = call i64 @universe_parse_json_scan_structural_scalar(ptr @g.json, i64 2048, i64 %ri)
  %st.bad = icmp ne i64 %st.v, %st.s
  %r.any = or i1 %ws.bad, %st.bad
  br i1 %r.any, label %sc.rbad, label %sc.rcont
sc.rbad:
  %rv0 = load i64, ptr @sc.viol, align 8
  %rv1 = add i64 %rv0, 1
  store i64 %rv1, ptr @sc.viol, align 8
  br label %sc.rcont
sc.rcont:
  %ri.n = add nuw i64 %ri, 1
  br label %sc.rloop
sc.aloop.pre:
  br label %sc.aloop
sc.aloop:
  %ai = phi i64 [ 0, %sc.aloop.pre ], [ %ai.n, %sc.acont ]
  %adone = icmp ugt i64 %ai, 40
  br i1 %adone, label %sc.fin, label %sc.abody
sc.abody:
  %aws.v = call i64 @universe_parse_json_scan_ws(ptr @sc.adv, i64 40, i64 %ai)
  %aws.s = call i64 @universe_parse_json_scan_ws_scalar(ptr @sc.adv, i64 40, i64 %ai)
  %aws.bad = icmp ne i64 %aws.v, %aws.s
  %ast.v = call i64 @universe_parse_json_scan_structural(ptr @sc.adv, i64 40, i64 %ai)
  %ast.s = call i64 @universe_parse_json_scan_structural_scalar(ptr @sc.adv, i64 40, i64 %ai)
  %ast.bad = icmp ne i64 %ast.v, %ast.s
  %a.any = or i1 %aws.bad, %ast.bad
  br i1 %a.any, label %sc.abad, label %sc.acont
sc.abad:
  %av0 = load i64, ptr @sc.viol, align 8
  %av1 = add i64 %av0, 1
  store i64 %av1, ptr @sc.viol, align 8
  br label %sc.acont
sc.acont:
  %ai.n = add nuw i64 %ai, 1
  br label %sc.aloop
sc.fin:
  %viol = load i64, ptr @sc.viol, align 8
  %viol.ok = icmp eq i64 %viol, 0
  call void @ut_check(i1 %viol.ok, ptr @sc.msg)
  br label %sc.done

sc.done:
  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; build "[12345,12345,...]" filling the buffer
  store i8 91, ptr @g.json, align 1                    ; '['
  br label %bb.head
bb.head:
  %bp = phi i64 [ 1, %bench ], [ %bp.next, %bb.body ]
  ; stop when next group (6 bytes "12345,") would exceed 65535 (leave room for ']')
  %room = add nuw i64 %bp, 6
  %fitsz = icmp ult i64 %room, 65535
  br i1 %fitsz, label %bb.body, label %bb.close
bb.body:
  %d0 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %bp
  store i8 49, ptr %d0, align 1                         ; '1'
  %p1 = add nuw i64 %bp, 1
  %d1 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %p1
  store i8 50, ptr %d1, align 1                         ; '2'
  %p2 = add nuw i64 %bp, 2
  %d2 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %p2
  store i8 51, ptr %d2, align 1                         ; '3'
  %p3 = add nuw i64 %bp, 3
  %d3 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %p3
  store i8 52, ptr %d3, align 1                         ; '4'
  %p4 = add nuw i64 %bp, 4
  %d4 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %p4
  store i8 53, ptr %d4, align 1                         ; '5'
  %p5 = add nuw i64 %bp, 5
  %d5 = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %p5
  store i8 44, ptr %d5, align 1                         ; ','
  %bp.next = add nuw i64 %bp, 6
  br label %bb.head
bb.close:
  ; overwrite the last ',' at bp-1 with ']'
  %lastc = sub i64 %bp, 1
  %dc = getelementptr inbounds nuw [65536 x i8], ptr @g.json, i64 0, i64 %lastc
  store i8 93, ptr %dc, align 1                         ; ']'
  br label %bench.run
bench.run:
  %sc.b = call ptr @universe_parse_json_scanner_create(ptr @g.json, i64 %bp, i32 64)
  %tok.b = alloca [24 x i8], align 8
  ; 17 reps of a 200-scan batch over the filled buffer; discard rep 0 (warm-up),
  ; report over the remaining 16. ops/rep = %bp * 200 bytes (ns per scanned byte).
  br label %json.rep
json.rep:
  %jrep = phi i64 [ 0, %bench.run ], [ %jrep.n, %json.next ]
  %t0 = call double @ut_now_sec()
  br label %bi.head
bi.head:
  %it = phi i64 [ 0, %json.rep ], [ %it.next, %bi.next ]
  %itdone = icmp uge i64 %it, 200
  br i1 %itdone, label %json.rep.done, label %bi.body
bi.body:
  call void @universe_parse_json_scanner_reset(ptr %sc.b)
  br label %bp.head
bp.head:
  %pr = call i32 @universe_parse_json_next(ptr %sc.b, ptr %tok.b)
  %pty = load i32, ptr %tok.b, align 4
  %pend = icmp eq i32 %pty, 10
  %perr = icmp ne i32 %pr, 0
  %pstop = or i1 %pend, %perr
  br i1 %pstop, label %bi.next, label %bp.head
bi.next:
  %it.next = add nuw i64 %it, 1
  br label %bi.head
json.rep.done:
  %t1 = call double @ut_now_sec()
  %jel = fsub double %t1, %t0
  %jkeep = icmp ugt i64 %jrep, 0
  br i1 %jkeep, label %json.store, label %json.next
json.store:
  %jidx = sub i64 %jrep, 1
  %jsp = getelementptr inbounds [16 x double], ptr @json.samp, i64 0, i64 %jidx
  store double %jel, ptr %jsp, align 8
  br label %json.next
json.next:
  %jrep.n = add nuw i64 %jrep, 1
  %jmore = icmp ult i64 %jrep.n, 17
  br i1 %jmore, label %json.rep, label %json.report
json.report:
  call void @universe_parse_json_scanner_destroy(ptr %sc.b)
  %json.ops = mul i64 %bp, 200
  call void @ut_report_dist(ptr @json.samp, i64 16, i64 %json.ops, ptr @lbl.json)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
