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

; Tests for universe_ds_art (single-threaded adaptive radix tree).

declare ptr @universe_ds_art_create()
declare void @universe_ds_art_destroy(ptr)
declare i32 @universe_ds_art_insert(ptr, ptr, i64, i64)
declare i32 @universe_ds_art_get(ptr, ptr, i64, ptr)
declare i32 @universe_ds_art_contains(ptr, ptr, i64)
declare i32 @universe_ds_art_delete(ptr, ptr, i64)
declare i64 @universe_ds_art_count(ptr)
declare i64 @universe_ds_art_iterate(ptr, ptr, ptr)
declare i64 @universe_ds_art_prefix_scan(ptr, ptr, i64, ptr, ptr)
declare i32 @universe_ds_art_min(ptr, ptr, ptr, ptr)
declare i32 @universe_ds_art_max(ptr, ptr, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare void @ut_report_dist(ptr, i64, i64, ptr)

@art.get.samp = internal global [16 x double] zeroinitializer, align 8
@art.btf.samp = internal global [16 x double] zeroinitializer, align 8
@art.psc.samp = internal global [16 x double] zeroinitializer, align 8
@art.itr.samp = internal global [16 x double] zeroinitializer, align 8
@art.psc.cnt  = internal global i64 0, align 8
@lbl.art.get = private unnamed_addr constant [17 x i8] c"art get 32k odds\00"
@lbl.art.btf = private unnamed_addr constant [20 x i8] c"btree find 32k odds\00"
@lbl.art.psc = private unnamed_addr constant [16 x i8] c"art prefix_scan\00"
@lbl.art.itr = private unnamed_addr constant [19 x i8] c"art iterate+filter\00"

; btree reference (same domain) for the get bench
declare ptr @universe_ds_btree_create()
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)
declare void @universe_ds_btree_destroy(ptr)

@filterbyte = global i8 1, align 1
@filtercount = global i64 0, align 8

; fullscan filter: count keys whose first byte == filterbyte
define i32 @filter_cb(ptr %ctx, ptr %key, i64 %klen, i64 %val) {
entry:
  %b0 = load i8, ptr %key, align 1
  %fb = load i8, ptr @filterbyte, align 1
  %m = icmp eq i8 %b0, %fb
  br i1 %m, label %inc, label %done
inc:
  %c = load i64, ptr @filtercount, align 8
  %c1 = add i64 %c, 1
  store i64 %c1, ptr @filtercount, align 8
  br label %done
done:
  ret i32 0
}

; hand keys
@k.empty = private unnamed_addr constant [1 x i8] c"\00", align 1
@k.app = private unnamed_addr constant [3 x i8] c"app", align 1
@k.apple = private unnamed_addr constant [5 x i8] c"apple", align 1
@k.apply = private unnamed_addr constant [5 x i8] c"apply", align 1
@k.application = private unnamed_addr constant [11 x i8] c"application", align 1
@k.banana = private unnamed_addr constant [6 x i8] c"banana", align 1
@k.band = private unnamed_addr constant [4 x i8] c"band", align 1
@k.bandana = private unnamed_addr constant [7 x i8] c"bandana", align 1
@k.b = private unnamed_addr constant [1 x i8] c"b", align 1
@k.appl = private unnamed_addr constant [4 x i8] c"appl", align 1
@k.ban = private unnamed_addr constant [3 x i8] c"ban", align 1
@k.zzz = private unnamed_addr constant [3 x i8] c"zzz", align 1
@k.bandanaX = private unnamed_addr constant [8 x i8] c"bandanaX", align 1

@p.app = private unnamed_addr constant [3 x i8] c"app", align 1
@p.ban = private unnamed_addr constant [3 x i8] c"ban", align 1
@p.x = private unnamed_addr constant [1 x i8] c"x", align 1

@m.count = private unnamed_addr constant [17 x i8] c"count after hand\00", align 1
@m.get = private unnamed_addr constant [10 x i8] c"get value\00", align 1
@m.absent = private unnamed_addr constant [11 x i8] c"absent key\00", align 1
@m.iter = private unnamed_addr constant [13 x i8] c"iterate sort\00", align 1
@m.itercnt = private unnamed_addr constant [15 x i8] c"iterate count\0A\00", align 1
@m.pscan = private unnamed_addr constant [13 x i8] c"prefix scan\0A\00", align 1
@m.min = private unnamed_addr constant [8 x i8] c"min key\00", align 1
@m.max = private unnamed_addr constant [8 x i8] c"max key\00", align 1
@m.fan = private unnamed_addr constant [14 x i8] c"fanout growth\00", align 1
@m.fandel = private unnamed_addr constant [15 x i8] c"fanout shrink\0A\00", align 1
@m.big = private unnamed_addr constant [13 x i8] c"big get all\0A\00", align 1
@m.bigdel = private unnamed_addr constant [13 x i8] c"big deleted\0A\00", align 1
@m.bigcnt = private unnamed_addr constant [11 x i8] c"big count\0A\00", align 1
@m.empty = private unnamed_addr constant [12 x i8] c"empty tree\0A\00", align 1
@m.overwrite = private unnamed_addr constant [11 x i8] c"overwrite\0A\00", align 1

; unsigned byte-wise lexicographic compare: -1 / 0 / 1
define i32 @lexcmp(ptr %a, i64 %alen, ptr %b, i64 %blen) {
entry:
  %lim = call i64 @llvm.umin.i64(i64 %alen, i64 %blen)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %done = icmp uge i64 %i, %lim
  br i1 %done, label %tail, label %step
step:
  %ap = getelementptr inbounds i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %lt = icmp ult i8 %av, %bv
  br i1 %lt, label %retm1, label %chkgt
chkgt:
  %gt = icmp ugt i8 %av, %bv
  br i1 %gt, label %retp1, label %cont
cont:
  %in = add i64 %i, 1
  br label %loop
tail:
  %ltl = icmp ult i64 %alen, %blen
  br i1 %ltl, label %retm1, label %chkgtl
chkgtl:
  %gtl = icmp ugt i64 %alen, %blen
  br i1 %gtl, label %retp1, label %ret0
retm1:
  ret i32 -1
retp1:
  ret i32 1
ret0:
  ret i32 0
}
declare i64 @llvm.umin.i64(i64, i64)

; iterate callback ctx: {i64 count@0, i64 viol@8, ptr prev@16, i64 prevlen@24, i64 have@32}
define i32 @iter_cb(ptr %ctx, ptr %key, i64 %klen, i64 %val) {
entry:
  %cntp = getelementptr inbounds i8, ptr %ctx, i64 0
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  %havep = getelementptr inbounds i8, ptr %ctx, i64 32
  %have = load i64, ptr %havep, align 8
  %has = icmp ne i64 %have, 0
  br i1 %has, label %cmp, label %setprev
cmp:
  %prevp = getelementptr inbounds i8, ptr %ctx, i64 16
  %prev = load ptr, ptr %prevp, align 8
  %prevlp = getelementptr inbounds i8, ptr %ctx, i64 24
  %prevl = load i64, ptr %prevlp, align 8
  %c = call i32 @lexcmp(ptr %prev, i64 %prevl, ptr %key, i64 %klen)
  ; require strictly increasing: prev < key => c == -1
  %bad = icmp sge i32 %c, 0
  br i1 %bad, label %viol, label %setprev
viol:
  %vp = getelementptr inbounds i8, ptr %ctx, i64 8
  %v = load i64, ptr %vp, align 8
  %v1 = add i64 %v, 1
  store i64 %v1, ptr %vp, align 8
  br label %setprev
setprev:
  %pp = getelementptr inbounds i8, ptr %ctx, i64 16
  store ptr %key, ptr %pp, align 8
  %plp = getelementptr inbounds i8, ptr %ctx, i64 24
  store i64 %klen, ptr %plp, align 8
  %hp = getelementptr inbounds i8, ptr %ctx, i64 32
  store i64 1, ptr %hp, align 8
  ret i32 0
}

; prefix callback ctx: {i64 count@0, i64 viol@8, ptr prefix@16, i64 plen@24}
define i32 @prefix_cb(ptr %ctx, ptr %key, i64 %klen, i64 %val) {
entry:
  %cntp = getelementptr inbounds i8, ptr %ctx, i64 0
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  %pfxp = getelementptr inbounds i8, ptr %ctx, i64 16
  %pfx = load ptr, ptr %pfxp, align 8
  %plp = getelementptr inbounds i8, ptr %ctx, i64 24
  %plen = load i64, ptr %plp, align 8
  %short = icmp ult i64 %klen, %plen
  br i1 %short, label %viol, label %cmp
cmp:
  ; first plen bytes must equal prefix
  br label %loop
loop:
  %i = phi i64 [ 0, %cmp ], [ %in, %ccont ]
  %done = icmp uge i64 %i, %plen
  br i1 %done, label %ok, label %cstep
cstep:
  %kp = getelementptr inbounds i8, ptr %key, i64 %i
  %kv = load i8, ptr %kp, align 1
  %pp = getelementptr inbounds i8, ptr %pfx, i64 %i
  %pv = load i8, ptr %pp, align 1
  %ne = icmp ne i8 %kv, %pv
  br i1 %ne, label %viol, label %ccont
ccont:
  %in = add i64 %i, 1
  br label %loop
viol:
  %vp = getelementptr inbounds i8, ptr %ctx, i64 8
  %v = load i64, ptr %vp, align 8
  %v1 = add i64 %v, 1
  store i64 %v1, ptr %vp, align 8
  br label %ok
ok:
  ret i32 0
}

; helper: insert + assert ok
define void @ins(ptr %t, ptr %k, i64 %len, i64 %val) {
entry:
  %r = call i32 @universe_ds_art_insert(ptr %t, ptr %k, i64 %len, i64 %val)
  ret void
}

; helper: get and check value == want
define void @chkget(ptr %t, ptr %k, i64 %len, i64 %want) {
entry:
  %slot = alloca i64, align 8
  %r = call i32 @universe_ds_art_get(ptr %t, ptr %k, i64 %len, ptr %slot)
  %ok = icmp eq i32 %r, 0
  call void @ut_check(i1 %ok, ptr @m.get)
  %v = load i64, ptr %slot, align 8
  call void @ut_check_eq(i64 %v, i64 %want, ptr @m.get)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %ictx = alloca [40 x i8], align 8
  %pctx = alloca [32 x i8], align 8
  %kbuf = alloca [8 x i8], align 8
  %okey = alloca ptr, align 8
  %olen = alloca i64, align 8
  %oval = alloca i64, align 8
  %vslot = alloca i64, align 8

  %t = call ptr @universe_ds_art_create()

  ; --- empty tree ---
  %c0 = call i64 @universe_ds_art_count(ptr %t)
  call void @ut_check_eq(i64 %c0, i64 0, ptr @m.empty)
  %g0 = call i32 @universe_ds_art_get(ptr %t, ptr @k.app, i64 3, ptr null)
  %g0nf = icmp eq i32 %g0, 5
  call void @ut_check(i1 %g0nf, ptr @m.empty)
  %mn0 = call i32 @universe_ds_art_min(ptr %t, ptr %okey, ptr %olen, ptr %oval)
  %mn0e = icmp eq i32 %mn0, 4
  call void @ut_check(i1 %mn0e, ptr @m.empty)

  ; --- insert hand keys ---
  call void @ins(ptr %t, ptr @k.empty, i64 0, i64 100)
  call void @ins(ptr %t, ptr @k.app, i64 3, i64 101)
  call void @ins(ptr %t, ptr @k.apple, i64 5, i64 102)
  call void @ins(ptr %t, ptr @k.apply, i64 5, i64 103)
  call void @ins(ptr %t, ptr @k.application, i64 11, i64 104)
  call void @ins(ptr %t, ptr @k.banana, i64 6, i64 105)
  call void @ins(ptr %t, ptr @k.band, i64 4, i64 106)
  call void @ins(ptr %t, ptr @k.bandana, i64 7, i64 107)
  call void @ins(ptr %t, ptr @k.b, i64 1, i64 108)

  %ch = call i64 @universe_ds_art_count(ptr %t)
  call void @ut_check_eq(i64 %ch, i64 9, ptr @m.count)

  ; get present
  call void @chkget(ptr %t, ptr @k.empty, i64 0, i64 100)
  call void @chkget(ptr %t, ptr @k.app, i64 3, i64 101)
  call void @chkget(ptr %t, ptr @k.apple, i64 5, i64 102)
  call void @chkget(ptr %t, ptr @k.apply, i64 5, i64 103)
  call void @chkget(ptr %t, ptr @k.application, i64 11, i64 104)
  call void @chkget(ptr %t, ptr @k.banana, i64 6, i64 105)
  call void @chkget(ptr %t, ptr @k.band, i64 4, i64 106)
  call void @chkget(ptr %t, ptr @k.bandana, i64 7, i64 107)
  call void @chkget(ptr %t, ptr @k.b, i64 1, i64 108)

  ; get absent
  %ga = call i32 @universe_ds_art_get(ptr %t, ptr @k.appl, i64 4, ptr null)
  %ganf = icmp eq i32 %ga, 5
  call void @ut_check(i1 %ganf, ptr @m.absent)
  %gb = call i32 @universe_ds_art_get(ptr %t, ptr @k.ban, i64 3, ptr null)
  %gbnf = icmp eq i32 %gb, 5
  call void @ut_check(i1 %gbnf, ptr @m.absent)
  %gz = call i32 @universe_ds_art_get(ptr %t, ptr @k.zzz, i64 3, ptr null)
  %gznf = icmp eq i32 %gz, 5
  call void @ut_check(i1 %gznf, ptr @m.absent)
  %gx = call i32 @universe_ds_art_get(ptr %t, ptr @k.bandanaX, i64 8, ptr null)
  %gxnf = icmp eq i32 %gx, 5
  call void @ut_check(i1 %gxnf, ptr @m.absent)

  ; overwrite
  call void @ins(ptr %t, ptr @k.app, i64 3, i64 201)
  call void @chkget(ptr %t, ptr @k.app, i64 3, i64 201)
  %ch2 = call i64 @universe_ds_art_count(ptr %t)
  call void @ut_check_eq(i64 %ch2, i64 9, ptr @m.overwrite)

  ; --- ordered iteration ---
  call void @llvm.memset.p0.i64(ptr %ictx, i8 0, i64 40, i1 false)
  %itn = call i64 @universe_ds_art_iterate(ptr %t, ptr @iter_cb, ptr %ictx)
  call void @ut_check_eq(i64 %itn, i64 9, ptr @m.itercnt)
  %iviolp = getelementptr inbounds i8, ptr %ictx, i64 8
  %iviol = load i64, ptr %iviolp, align 8
  call void @ut_check_eq(i64 %iviol, i64 0, ptr @m.iter)

  ; --- prefix scan "app" -> app, apple, application, apply = 4 ---
  call void @llvm.memset.p0.i64(ptr %pctx, i8 0, i64 32, i1 false)
  %pfp = getelementptr inbounds i8, ptr %pctx, i64 16
  store ptr @p.app, ptr %pfp, align 8
  %plp = getelementptr inbounds i8, ptr %pctx, i64 24
  store i64 3, ptr %plp, align 8
  %ps1 = call i64 @universe_ds_art_prefix_scan(ptr %t, ptr @p.app, i64 3, ptr @prefix_cb, ptr %pctx)
  call void @ut_check_eq(i64 %ps1, i64 4, ptr @m.pscan)
  %pv1p = getelementptr inbounds i8, ptr %pctx, i64 8
  %pv1 = load i64, ptr %pv1p, align 8
  call void @ut_check_eq(i64 %pv1, i64 0, ptr @m.pscan)

  ; prefix "ban" -> banana, band, bandana = 3
  call void @llvm.memset.p0.i64(ptr %pctx, i8 0, i64 32, i1 false)
  %pfp2 = getelementptr inbounds i8, ptr %pctx, i64 16
  store ptr @p.ban, ptr %pfp2, align 8
  %plp2 = getelementptr inbounds i8, ptr %pctx, i64 24
  store i64 3, ptr %plp2, align 8
  %ps2 = call i64 @universe_ds_art_prefix_scan(ptr %t, ptr @p.ban, i64 3, ptr @prefix_cb, ptr %pctx)
  call void @ut_check_eq(i64 %ps2, i64 3, ptr @m.pscan)

  ; prefix "" -> all 9
  %ps3 = call i64 @universe_ds_art_prefix_scan(ptr %t, ptr @k.empty, i64 0, ptr @prefix_cb, ptr %pctx)
  call void @ut_check_eq(i64 %ps3, i64 9, ptr @m.pscan)

  ; prefix "x" -> 0
  %ps4 = call i64 @universe_ds_art_prefix_scan(ptr %t, ptr @p.x, i64 1, ptr @prefix_cb, ptr %pctx)
  call void @ut_check_eq(i64 %ps4, i64 0, ptr @m.pscan)

  ; --- min / max ---
  %mn = call i32 @universe_ds_art_min(ptr %t, ptr %okey, ptr %olen, ptr %oval)
  %mnok = icmp eq i32 %mn, 0
  call void @ut_check(i1 %mnok, ptr @m.min)
  %mnlen = load i64, ptr %olen, align 8
  call void @ut_check_eq(i64 %mnlen, i64 0, ptr @m.min)
  %mnval = load i64, ptr %oval, align 8
  call void @ut_check_eq(i64 %mnval, i64 100, ptr @m.min)

  %mx = call i32 @universe_ds_art_max(ptr %t, ptr %okey, ptr %olen, ptr %oval)
  %mxok = icmp eq i32 %mx, 0
  call void @ut_check(i1 %mxok, ptr @m.max)
  %mxlen = load i64, ptr %olen, align 8
  call void @ut_check_eq(i64 %mxlen, i64 7, ptr @m.max)
  %mxval = load i64, ptr %oval, align 8
  call void @ut_check_eq(i64 %mxval, i64 107, ptr @m.max)
  %mxkey = load ptr, ptr %okey, align 8
  %mxcmp = call i32 @lexcmp(ptr %mxkey, i64 7, ptr @k.bandana, i64 7)
  %mxeq = icmp eq i32 %mxcmp, 0
  call void @ut_check(i1 %mxeq, ptr @m.max)

  ; --- delete a few, re-check ---
  %d1 = call i32 @universe_ds_art_delete(ptr %t, ptr @k.app, i64 3)
  %d1ok = icmp eq i32 %d1, 0
  call void @ut_check(i1 %d1ok, ptr @m.get)
  %gd1 = call i32 @universe_ds_art_get(ptr %t, ptr @k.app, i64 3, ptr null)
  %gd1nf = icmp eq i32 %gd1, 5
  call void @ut_check(i1 %gd1nf, ptr @m.get)
  ; siblings still present
  call void @chkget(ptr %t, ptr @k.apple, i64 5, i64 102)
  call void @chkget(ptr %t, ptr @k.application, i64 11, i64 104)
  ; delete "band" (has "bandana" under it via term structure)
  %d2 = call i32 @universe_ds_art_delete(ptr %t, ptr @k.band, i64 4)
  %d2ok = icmp eq i32 %d2, 0
  call void @ut_check(i1 %d2ok, ptr @m.get)
  call void @chkget(ptr %t, ptr @k.bandana, i64 7, i64 107)
  call void @chkget(ptr %t, ptr @k.banana, i64 6, i64 105)
  ; delete empty key
  %d3 = call i32 @universe_ds_art_delete(ptr %t, ptr @k.empty, i64 0)
  %d3ok = icmp eq i32 %d3, 0
  call void @ut_check(i1 %d3ok, ptr @m.get)
  %gd3 = call i32 @universe_ds_art_get(ptr %t, ptr @k.empty, i64 0, ptr null)
  %gd3nf = icmp eq i32 %gd3, 5
  call void @ut_check(i1 %gd3nf, ptr @m.get)
  ; delete absent
  %d4 = call i32 @universe_ds_art_delete(ptr %t, ptr @k.zzz, i64 3)
  %d4nf = icmp eq i32 %d4, 5
  call void @ut_check(i1 %d4nf, ptr @m.get)
  %chd = call i64 @universe_ds_art_count(ptr %t)
  call void @ut_check_eq(i64 %chd, i64 6, ptr @m.count)

  ; --- fanout growth: 256 two-byte keys {0x30, b} -> Node4->16->48->256 ---
  %t2 = call ptr @universe_ds_art_create()
  br label %fanins

fanins:
  %fi = phi i64 [ 0, %entry ], [ %fin, %fanins ]
  %kb0 = getelementptr inbounds i8, ptr %kbuf, i64 0
  store i8 48, ptr %kb0, align 1
  %fi8 = trunc i64 %fi to i8
  %kb1 = getelementptr inbounds i8, ptr %kbuf, i64 1
  store i8 %fi8, ptr %kb1, align 1
  %fval = add i64 %fi, 200
  %fr = call i32 @universe_ds_art_insert(ptr %t2, ptr %kbuf, i64 2, i64 %fval)
  %fin = add i64 %fi, 1
  %fmore = icmp ult i64 %fin, 256
  br i1 %fmore, label %fanins, label %fanchk

fanchk:
  %fc = call i64 @universe_ds_art_count(ptr %t2)
  call void @ut_check_eq(i64 %fc, i64 256, ptr @m.fan)
  ; verify each get
  br label %fanget

fanget:
  %gi = phi i64 [ 0, %fanchk ], [ %gin, %fangetc ]
  %giv = phi i64 [ 0, %fanchk ], [ %givn, %fangetc ]
  %gkb0 = getelementptr inbounds i8, ptr %kbuf, i64 0
  store i8 48, ptr %gkb0, align 1
  %gi8 = trunc i64 %gi to i8
  %gkb1 = getelementptr inbounds i8, ptr %kbuf, i64 1
  store i8 %gi8, ptr %gkb1, align 1
  %gr = call i32 @universe_ds_art_get(ptr %t2, ptr %kbuf, i64 2, ptr %vslot)
  %grok = icmp eq i32 %gr, 0
  %gv = load i64, ptr %vslot, align 8
  %gwant = add i64 %gi, 200
  %gvok = icmp eq i64 %gv, %gwant
  %gboth = and i1 %grok, %gvok
  %gbz = zext i1 %gboth to i64
  %givn = add i64 %giv, %gbz
  %gin = add i64 %gi, 1
  br label %fangetc
fangetc:
  %gmore = icmp ult i64 %gin, 256
  br i1 %gmore, label %fanget, label %fandone
fandone:
  call void @ut_check_eq(i64 %givn, i64 256, ptr @m.fan)

  ; prefix scan single byte {0x30} -> 256
  call void @llvm.memset.p0.i64(ptr %pctx, i8 0, i64 32, i1 false)
  %fpfp = getelementptr inbounds i8, ptr %pctx, i64 16
  %fkb0 = getelementptr inbounds i8, ptr %kbuf, i64 0
  store i8 48, ptr %fkb0, align 1
  store ptr %kbuf, ptr %fpfp, align 8
  %fplp = getelementptr inbounds i8, ptr %pctx, i64 24
  store i64 1, ptr %fplp, align 8
  %fps = call i64 @universe_ds_art_prefix_scan(ptr %t2, ptr %kbuf, i64 1, ptr @prefix_cb, ptr %pctx)
  call void @ut_check_eq(i64 %fps, i64 256, ptr @m.pscan)

  ; delete 226 keys (b=0..225) -> forces 256->48->16 shrink
  br label %fandel

fandel:
  %di = phi i64 [ 0, %fandone ], [ %din, %fandel ]
  %dkb0 = getelementptr inbounds i8, ptr %kbuf, i64 0
  store i8 48, ptr %dkb0, align 1
  %di8 = trunc i64 %di to i8
  %dkb1 = getelementptr inbounds i8, ptr %kbuf, i64 1
  store i8 %di8, ptr %dkb1, align 1
  %dr = call i32 @universe_ds_art_delete(ptr %t2, ptr %kbuf, i64 2)
  %din = add i64 %di, 1
  %dmore = icmp ult i64 %din, 226
  br i1 %dmore, label %fandel, label %fandelchk

fandelchk:
  %dc = call i64 @universe_ds_art_count(ptr %t2)
  call void @ut_check_eq(i64 %dc, i64 30, ptr @m.fandel)
  ; remaining b=226..255 present, deleted absent
  br label %fandelget
fandelget:
  %ei = phi i64 [ 0, %fandelchk ], [ %ein, %fandelgetc ]
  %eviol = phi i64 [ 0, %fandelchk ], [ %eviol2, %fandelgetc ]
  %ekb0 = getelementptr inbounds i8, ptr %kbuf, i64 0
  store i8 48, ptr %ekb0, align 1
  %ei8 = trunc i64 %ei to i8
  %ekb1 = getelementptr inbounds i8, ptr %kbuf, i64 1
  store i8 %ei8, ptr %ekb1, align 1
  %er = call i32 @universe_ds_art_get(ptr %t2, ptr %kbuf, i64 2, ptr null)
  %epresent = icmp eq i32 %er, 0
  %edeleted = icmp ult i64 %ei, 226
  ; expected present == not deleted
  %eexp = xor i1 %edeleted, true
  %emismatch = xor i1 %epresent, %eexp
  %emz = zext i1 %emismatch to i64
  %eviol2 = add i64 %eviol, %emz
  %ein = add i64 %ei, 1
  br label %fandelgetc
fandelgetc:
  %emore = icmp ult i64 %ein, 256
  br i1 %emore, label %fandelget, label %fandeldone
fandeldone:
  call void @ut_check_eq(i64 %eviol, i64 0, ptr @m.fandel)
  call void @universe_ds_art_destroy(ptr %t2)

  ; --- big: 65536 distinct 8-byte keys ---
  %t3 = call ptr @universe_ds_art_create()
  br label %biginsert

biginsert:
  %bi = phi i64 [ 0, %fandeldone ], [ %bin, %biginsert ]
  store i64 %bi, ptr %kbuf, align 8
  %br = call i32 @universe_ds_art_insert(ptr %t3, ptr %kbuf, i64 8, i64 %bi)
  %bin = add i64 %bi, 1
  %bmore = icmp ult i64 %bin, 65536
  br i1 %bmore, label %biginsert, label %bigcount

bigcount:
  %bc = call i64 @universe_ds_art_count(ptr %t3)
  call void @ut_check_eq(i64 %bc, i64 65536, ptr @m.bigcnt)
  br label %bigget

bigget:
  %gbi = phi i64 [ 0, %bigcount ], [ %gbin, %biggetc ]
  %gbviol = phi i64 [ 0, %bigcount ], [ %gbviol2, %biggetc ]
  store i64 %gbi, ptr %kbuf, align 8
  %gbr = call i32 @universe_ds_art_get(ptr %t3, ptr %kbuf, i64 8, ptr %vslot)
  %gbok = icmp eq i32 %gbr, 0
  %gbv = load i64, ptr %vslot, align 8
  %gbveq = icmp eq i64 %gbv, %gbi
  %gbgood = and i1 %gbok, %gbveq
  %gbbad = xor i1 %gbgood, true
  %gbbadz = zext i1 %gbbad to i64
  %gbviol2 = add i64 %gbviol, %gbbadz
  %gbin = add i64 %gbi, 1
  br label %biggetc
biggetc:
  %gbmore = icmp ult i64 %gbin, 65536
  br i1 %gbmore, label %bigget, label %biggetdone
biggetdone:
  call void @ut_check_eq(i64 %gbviol, i64 0, ptr @m.big)

  ; iterate big -> ordered, count 65536
  call void @llvm.memset.p0.i64(ptr %ictx, i8 0, i64 40, i1 false)
  %bitn = call i64 @universe_ds_art_iterate(ptr %t3, ptr @iter_cb, ptr %ictx)
  call void @ut_check_eq(i64 %bitn, i64 65536, ptr @m.iter)
  %biviolp = getelementptr inbounds i8, ptr %ictx, i64 8
  %biviol = load i64, ptr %biviolp, align 8
  call void @ut_check_eq(i64 %biviol, i64 0, ptr @m.iter)

  ; delete even keys
  br label %bigdel
bigdel:
  %ddi = phi i64 [ 0, %biggetdone ], [ %ddin, %bigdel ]
  store i64 %ddi, ptr %kbuf, align 8
  %ddr = call i32 @universe_ds_art_delete(ptr %t3, ptr %kbuf, i64 8)
  %ddin = add i64 %ddi, 2
  %ddmore = icmp ult i64 %ddin, 65536
  br i1 %ddmore, label %bigdel, label %bigdelchk
bigdelchk:
  %dbc = call i64 @universe_ds_art_count(ptr %t3)
  call void @ut_check_eq(i64 %dbc, i64 32768, ptr @m.bigdel)
  ; verify evens absent, odds present
  br label %bigdelget
bigdelget:
  %fdi = phi i64 [ 0, %bigdelchk ], [ %fdin, %bigdelgetc ]
  %fdviol = phi i64 [ 0, %bigdelchk ], [ %fdviol2, %bigdelgetc ]
  store i64 %fdi, ptr %kbuf, align 8
  %fdr = call i32 @universe_ds_art_get(ptr %t3, ptr %kbuf, i64 8, ptr null)
  %fdpresent = icmp eq i32 %fdr, 0
  %fdlow = and i64 %fdi, 1
  %fdodd = icmp eq i64 %fdlow, 1
  ; odd should be present
  %fdmis = xor i1 %fdpresent, %fdodd
  %fdmz = zext i1 %fdmis to i64
  %fdviol2 = add i64 %fdviol, %fdmz
  %fdin = add i64 %fdi, 1
  br label %bigdelgetc
bigdelgetc:
  %fdmore = icmp ult i64 %fdin, 65536
  br i1 %fdmore, label %bigdelget, label %bigdeldone
bigdeldone:
  call void @ut_check_eq(i64 %fdviol, i64 0, ptr @m.bigdel)

  ; --- optional bench ---
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %alldone

bench:
  ; reference: btree keyed on the same i64 values (odds present)
  %tb = call ptr @universe_ds_btree_create()
  br label %bbins
bbins:
  %bii = phi i64 [ 1, %bench ], [ %biin, %bbins ]
  %bir = call i32 @universe_ds_btree_insert(ptr %tb, i64 %bii, i64 %bii)
  %biin = add i64 %bii, 2
  %bimore = icmp ult i64 %biin, 65536
  br i1 %bimore, label %bbins, label %art.get.rep
; --- ART get distribution (warm-up rep discarded) ---
art.get.rep:
  %grep = phi i64 [ 0, %bbins ], [ %grep.n, %grep.cont ]
  %at0 = call double @ut_now_sec()
  br label %bloop
bloop:
  %qi = phi i64 [ 1, %art.get.rep ], [ %qin, %bloop ]
  %qacc = phi i64 [ 0, %art.get.rep ], [ %qacc2, %bloop ]
  store i64 %qi, ptr %kbuf, align 8
  %qr = call i32 @universe_ds_art_get(ptr %t3, ptr %kbuf, i64 8, ptr %vslot)
  %qv = load i64, ptr %vslot, align 8
  %qacc2 = add i64 %qacc, %qv
  %qin = add i64 %qi, 2
  %qmore = icmp ult i64 %qin, 65536
  br i1 %qmore, label %bloop, label %grep.done
grep.done:
  %at1 = call double @ut_now_sec()
  store volatile i64 %qacc2, ptr %vslot, align 8
  %gelapsed = fsub double %at1, %at0
  %gwarm = icmp eq i64 %grep, 0
  br i1 %gwarm, label %grep.cont, label %grep.store
grep.store:
  %gsi = sub i64 %grep, 1
  %gsp = getelementptr inbounds double, ptr @art.get.samp, i64 %gsi
  store double %gelapsed, ptr %gsp, align 8
  br label %grep.cont
grep.cont:
  %grep.n = add nuw i64 %grep, 1
  %grmore = icmp ult i64 %grep.n, 17
  br i1 %grmore, label %art.get.rep, label %art.get.report
art.get.report:
  call void @ut_report_dist(ptr @art.get.samp, i64 16, i64 32768, ptr @lbl.art.get)
  br label %art.btf.rep
; --- btree find distribution ---
art.btf.rep:
  %brep = phi i64 [ 0, %art.get.report ], [ %brep.n, %brep.cont ]
  %bt0 = call double @ut_now_sec()
  br label %btloop
btloop:
  %ti = phi i64 [ 1, %art.btf.rep ], [ %tin, %btloop ]
  %tacc = phi i64 [ 0, %art.btf.rep ], [ %tacc2, %btloop ]
  %tr = call i32 @universe_ds_btree_find(ptr %tb, i64 %ti, ptr %vslot)
  %tv = load i64, ptr %vslot, align 8
  %tacc2 = add i64 %tacc, %tv
  %tin = add i64 %ti, 2
  %tmore = icmp ult i64 %tin, 65536
  br i1 %tmore, label %btloop, label %brep.done
brep.done:
  %bt1 = call double @ut_now_sec()
  store volatile i64 %tacc2, ptr %vslot, align 8
  %belapsed = fsub double %bt1, %bt0
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.cont, label %brep.store
brep.store:
  %bsi = sub i64 %brep, 1
  %bsp = getelementptr inbounds double, ptr @art.btf.samp, i64 %bsi
  store double %belapsed, ptr %bsp, align 8
  br label %brep.cont
brep.cont:
  %brep.n = add nuw i64 %brep, 1
  %brmore = icmp ult i64 %brep.n, 17
  br i1 %brmore, label %art.btf.rep, label %art.btf.report
art.btf.report:
  call void @ut_report_dist(ptr @art.btf.samp, i64 16, i64 32768, ptr @lbl.art.btf)
  call void @universe_ds_btree_destroy(ptr %tb)
  br label %art.psc.rep

  ; prefix_scan (visits only the matching subtree) vs full iterate+filter.
  ; 1-byte prefix 0x01 selects keys whose low byte == 1 (256 keys, odd => present)
; --- prefix_scan distribution ---
art.psc.rep:
  %prep = phi i64 [ 0, %art.btf.report ], [ %prep.n, %prep.cont ]
  store i8 1, ptr @filterbyte, align 1
  store i8 1, ptr %kbuf, align 1
  %pt0 = call double @ut_now_sec()
  %pcnt = call i64 @universe_ds_art_prefix_scan(ptr %t3, ptr %kbuf, i64 1, ptr @prefix_cb, ptr %pctx)
  %pt1 = call double @ut_now_sec()
  store i64 %pcnt, ptr @art.psc.cnt, align 8
  %pelapsed = fsub double %pt1, %pt0
  %pwarm = icmp eq i64 %prep, 0
  br i1 %pwarm, label %prep.cont, label %prep.store
prep.store:
  %psi = sub i64 %prep, 1
  %psp = getelementptr inbounds double, ptr @art.psc.samp, i64 %psi
  store double %pelapsed, ptr %psp, align 8
  br label %prep.cont
prep.cont:
  %prep.n = add nuw i64 %prep, 1
  %pmore = icmp ult i64 %prep.n, 17
  br i1 %pmore, label %art.psc.rep, label %art.psc.report
art.psc.report:
  call void @ut_report_dist(ptr @art.psc.samp, i64 16, i64 1, ptr @lbl.art.psc)
  br label %art.itr.rep
; --- full iterate+filter distribution ---
art.itr.rep:
  %irep = phi i64 [ 0, %art.psc.report ], [ %irep.n, %irep.cont ]
  store i64 0, ptr @filtercount, align 8
  %ft0 = call double @ut_now_sec()
  %fs = call i64 @universe_ds_art_iterate(ptr %t3, ptr @filter_cb, ptr null)
  %ft1 = call double @ut_now_sec()
  %ielapsed = fsub double %ft1, %ft0
  %iwarm = icmp eq i64 %irep, 0
  br i1 %iwarm, label %irep.cont, label %irep.store
irep.store:
  %isi = sub i64 %irep, 1
  %isp = getelementptr inbounds double, ptr @art.itr.samp, i64 %isi
  store double %ielapsed, ptr %isp, align 8
  br label %irep.cont
irep.cont:
  %irep.n = add nuw i64 %irep, 1
  %imore = icmp ult i64 %irep.n, 17
  br i1 %imore, label %art.itr.rep, label %art.itr.report
art.itr.report:
  call void @ut_report_dist(ptr @art.itr.samp, i64 16, i64 1, ptr @lbl.art.itr)
  ; sanity: both find the same number of matches
  %fcnt = load i64, ptr @filtercount, align 8
  %pcnt.f = load i64, ptr @art.psc.cnt, align 8
  %scmatch = icmp eq i64 %pcnt.f, %fcnt
  call void @ut_check(i1 %scmatch, ptr @m.pscan)
  br label %alldone

alldone:
  call void @universe_ds_art_destroy(ptr %t3)
  call void @universe_ds_art_destroy(ptr %t)
  call void @test_interleaved()
  %r = call i32 @ut_summary()
  ret i32 %r
}

declare ptr @calloc(i64, i64)
declare void @free(ptr)

; randomized INTERLEAVED insert/delete vs a bounded-domain shadow. Each integer
; key 0..4095 is encoded as a fixed 8-byte key. Data-dependent deletes exercise
; arbitrary node-shrink (256->48->16->4) and prefix-collapse orderings, unlike
; the fixed delete schedules elsewhere. Invariant after every op:
; contains(k) == shadow[k]; plus a final full-domain scan and count.
@m.il.state = private unnamed_addr constant [25 x i8] c"interleaved contains==sh\00"
@m.il.scan  = private unnamed_addr constant [22 x i8] c"interleaved full scan\00"
@m.il.count = private unnamed_addr constant [22 x i8] c"interleaved count ok \00"

define void @test_interleaved() {
entry:
  %seed = alloca i64, align 8
  store i64 -4265267296055464877, ptr %seed, align 8
  %kbuf = alloca [8 x i8], align 8
  %vslot = alloca i64, align 8
  %present = call ptr @calloc(i64 4096, i64 1)
  %t = call ptr @universe_ds_art_create()
  br label %op.head

op.head:
  %oi = phi i64 [ 0, %entry ], [ %oi.n, %op.tail ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %op.tail ]
  %cnt = phi i64 [ 0, %entry ], [ %cnt.n, %op.tail ]
  %r = call i64 @ut_rand(ptr %seed)
  %k = urem i64 %r, 4096
  store i64 %k, ptr %kbuf, align 8
  %r2 = call i64 @ut_rand(ptr %seed)
  %op = and i64 %r2, 1
  %pp = getelementptr inbounds nuw i8, ptr %present, i64 %k
  %was = load i8, ptr %pp, align 1
  %km = mul i64 %k, 3
  %val = add i64 %km, 7
  %isins = icmp eq i64 %op, 1
  br i1 %isins, label %do.ins, label %do.del

do.ins:
  %irc = call i32 @universe_ds_art_insert(ptr %t, ptr %kbuf, i64 8, i64 %val)
  %was0 = icmp eq i8 %was, 0
  %cinc = zext i1 %was0 to i64
  %cnt.ins = add i64 %cnt, %cinc
  store i8 1, ptr %pp, align 1
  br label %chk

do.del:
  %drc = call i32 @universe_ds_art_delete(ptr %t, ptr %kbuf, i64 8)
  %was1 = icmp eq i8 %was, 1
  %cdec = zext i1 %was1 to i64
  %cnt.del = sub i64 %cnt, %cdec
  store i8 0, ptr %pp, align 1
  br label %chk

chk:
  %cnt.n = phi i64 [ %cnt.ins, %do.ins ], [ %cnt.del, %do.del ]
  %exp = phi i8 [ 1, %do.ins ], [ 0, %do.del ]
  %c = call i32 @universe_ds_art_contains(ptr %t, ptr %kbuf, i64 8)
  %expi = zext i8 %exp to i32
  %cbad = icmp ne i32 %c, %expi
  %cbadz = zext i1 %cbad to i64
  %viol.n1 = add i64 %viol, %cbadz
  %ispres = icmp eq i8 %exp, 1
  br i1 %ispres, label %vchk, label %op.tail

vchk:
  %frc = call i32 @universe_ds_art_get(ptr %t, ptr %kbuf, i64 8, ptr %vslot)
  %fv = load i64, ptr %vslot, align 8
  %frcok = icmp eq i32 %frc, 0
  %fvok = icmp eq i64 %fv, %val
  %fok = and i1 %frcok, %fvok
  %fbad = xor i1 %fok, true
  %fbadz = zext i1 %fbad to i64
  br label %op.tail

op.tail:
  %vadd = phi i64 [ 0, %chk ], [ %fbadz, %vchk ]
  %viol.n = add i64 %viol.n1, %vadd
  %oi.n = add nuw i64 %oi, 1
  %omore = icmp ult i64 %oi.n, 300000
  br i1 %omore, label %op.head, label %op.done

op.done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.il.state)
  br label %scan

scan:
  %si = phi i64 [ 0, %op.done ], [ %si.n, %scan ]
  %sviol = phi i64 [ 0, %op.done ], [ %sviol.n, %scan ]
  store i64 %si, ptr %kbuf, align 8
  %spp = getelementptr inbounds nuw i8, ptr %present, i64 %si
  %sp = load i8, ptr %spp, align 1
  %sc = call i32 @universe_ds_art_contains(ptr %t, ptr %kbuf, i64 8)
  %sexp = zext i8 %sp to i32
  %sbad = icmp ne i32 %sc, %sexp
  %sbadz = zext i1 %sbad to i64
  %sviol.n = add i64 %sviol, %sbadz
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 4096
  br i1 %smore, label %scan, label %scan.done

scan.done:
  call void @ut_check_eq(i64 %sviol.n, i64 0, ptr @m.il.scan)
  %fcnt = call i64 @universe_ds_art_count(ptr %t)
  call void @ut_check_eq(i64 %fcnt, i64 %cnt.n, ptr @m.il.count)
  call void @universe_ds_art_destroy(ptr %t)
  call void @free(ptr %present)
  ret void
}

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
