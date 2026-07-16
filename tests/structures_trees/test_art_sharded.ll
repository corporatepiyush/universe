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

; Tests for universe_ds_art_sharded (hash-sharded concurrent ART).

declare ptr @universe_ds_art_sharded_create(i64)
declare void @universe_ds_art_sharded_destroy(ptr)
declare i32 @universe_ds_art_sharded_put(ptr, ptr, i64, i64)
declare i32 @universe_ds_art_sharded_get(ptr, ptr, i64, ptr)
declare i32 @universe_ds_art_sharded_delete(ptr, ptr, i64)
declare i32 @universe_ds_art_sharded_contains(ptr, ptr, i64)
declare i64 @universe_ds_art_sharded_len(ptr)
declare i64 @universe_ds_art_sharded_shards(ptr)
declare i64 @universe_ds_art_sharded_prefix_scan(ptr, ptr, i64, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(ptr, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)

@art8.bench.samp = internal global [16 x double] zeroinitializer, align 8
@art1.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.art8.bench = private unnamed_addr constant [23 x i8] c"art_sharded 8-shard 4T\00"
@lbl.art1.bench = private unnamed_addr constant [23 x i8] c"art_sharded 1-shard 4T\00"

@k.alpha = private unnamed_addr constant [5 x i8] c"alpha", align 1
@k.beta = private unnamed_addr constant [4 x i8] c"beta", align 1
@k.gamma = private unnamed_addr constant [5 x i8] c"gamma", align 1
@k.alp = private unnamed_addr constant [3 x i8] c"alp", align 1
@k.absent = private unnamed_addr constant [4 x i8] c"nope", align 1

@m.shards = private unnamed_addr constant [12 x i8] c"shard count\00", align 1
@m.put = private unnamed_addr constant [4 x i8] c"put\00", align 1
@m.get = private unnamed_addr constant [4 x i8] c"get\00", align 1
@m.len = private unnamed_addr constant [4 x i8] c"len\00", align 1
@m.del = private unnamed_addr constant [7 x i8] c"delete\00", align 1
@m.pscan = private unnamed_addr constant [12 x i8] c"prefix scan\00", align 1
@m.dj = private unnamed_addr constant [18 x i8] c"disjoint no-loss\0A\00", align 1
@m.djcnt = private unnamed_addr constant [16 x i8] c"disjoint count\0A\00", align 1
@m.ov = private unnamed_addr constant [18 x i8] c"overlap value ok\0A\00", align 1
@m.ovlen = private unnamed_addr constant [16 x i8] c"overlap len ok\0A\00", align 1

; tuning
; THREADS=4  PER=16384 (disjoint keys/thread)  OPS=100000  SHARED=8192  ITERS=10

; arg struct {ptr map@0, i64 tid@8, i64 mode@16, i64 viol@24} stride 32

; disjoint worker: insert PER distinct keys, then get+verify them
define ptr @worker(ptr %arg) {
entry:
  %kbuf = alloca [8 x i8], align 8
  %vslot = alloca i64, align 8
  %state = alloca i64, align 8
  %mapp = getelementptr inbounds i8, ptr %arg, i64 0
  %map = load ptr, ptr %mapp, align 8
  %tidp = getelementptr inbounds i8, ptr %arg, i64 8
  %tid = load i64, ptr %tidp, align 8
  %modep = getelementptr inbounds i8, ptr %arg, i64 16
  %mode = load i64, ptr %modep, align 8
  %isov = icmp eq i64 %mode, 1
  br i1 %isov, label %overlap, label %disjoint

disjoint:
  %base = mul i64 %tid, 16384
  br label %insloop
insloop:
  %ii = phi i64 [ 0, %disjoint ], [ %iin, %insloop ]
  %idx = add i64 %base, %ii
  store i64 %idx, ptr %kbuf, align 8
  %pr = call i32 @universe_ds_art_sharded_put(ptr %map, ptr %kbuf, i64 8, i64 %idx)
  %iin = add i64 %ii, 1
  %imore = icmp ult i64 %iin, 16384
  br i1 %imore, label %insloop, label %getloop
getloop:
  %gi = phi i64 [ 0, %insloop ], [ %gin, %getloop ]
  %gviol = phi i64 [ 0, %insloop ], [ %gviol2, %getloop ]
  %gidx = add i64 %base, %gi
  store i64 %gidx, ptr %kbuf, align 8
  %gr = call i32 @universe_ds_art_sharded_get(ptr %map, ptr %kbuf, i64 8, ptr %vslot)
  %gok = icmp eq i32 %gr, 0
  %gv = load i64, ptr %vslot, align 8
  %gveq = icmp eq i64 %gv, %gidx
  %ggood = and i1 %gok, %gveq
  %gbad = xor i1 %ggood, true
  %gbz = zext i1 %gbad to i64
  %gviol2 = add i64 %gviol, %gbz
  %gin = add i64 %gi, 1
  %gmore = icmp ult i64 %gin, 16384
  br i1 %gmore, label %getloop, label %djdone
djdone:
  %vp = getelementptr inbounds i8, ptr %arg, i64 24
  store i64 %gviol2, ptr %vp, align 8
  ret ptr null

overlap:
  %seed = mul i64 %tid, 2654435761
  %seed1 = add i64 %seed, 12345
  store i64 %seed1, ptr %state, align 8
  br label %oploop
oploop:
  %oi = phi i64 [ 0, %overlap ], [ %oin, %opcont ]
  %r = call i64 @ut_rand(ptr %state)
  %k = urem i64 %r, 8192
  %act0 = lshr i64 %r, 20
  %act = and i64 %act0, 3
  store i64 %k, ptr %kbuf, align 8
  %isput = icmp ult i64 %act, 2
  br i1 %isput, label %doput, label %chkget
doput:
  %pv = add i64 %k, 1000000
  %opr = call i32 @universe_ds_art_sharded_put(ptr %map, ptr %kbuf, i64 8, i64 %pv)
  br label %opcont
chkget:
  %isget = icmp eq i64 %act, 2
  br i1 %isget, label %doget, label %dodel
doget:
  %ogr = call i32 @universe_ds_art_sharded_get(ptr %map, ptr %kbuf, i64 8, ptr %vslot)
  br label %opcont
dodel:
  %odr = call i32 @universe_ds_art_sharded_delete(ptr %map, ptr %kbuf, i64 8)
  br label %opcont
opcont:
  %oin = add i64 %oi, 1
  %omore = icmp ult i64 %oin, 100000
  br i1 %omore, label %oploop, label %ovdone
ovdone:
  %vp2 = getelementptr inbounds i8, ptr %arg, i64 24
  store i64 0, ptr %vp2, align 8
  ret ptr null
}

; noop prefix callback (counts internally in ctx i64@0)
define i32 @cnt_cb(ptr %ctx, ptr %key, i64 %klen, i64 %val) {
entry:
  %c = load i64, ptr %ctx, align 8
  %c1 = add i64 %c, 1
  store i64 %c1, ptr %ctx, align 8
  ret i32 0
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %handles = alloca [4 x i64], align 8
  %args = alloca [128 x i8], align 8    ; 4 * 32
  %kbuf = alloca [8 x i8], align 8
  %vslot = alloca i64, align 8

  ; ---- single-thread mirror ----
  %m0 = call ptr @universe_ds_art_sharded_create(i64 8)
  %sh = call i64 @universe_ds_art_sharded_shards(ptr %m0)
  call void @ut_check_eq(i64 %sh, i64 8, ptr @m.shards)

  %p1 = call i32 @universe_ds_art_sharded_put(ptr %m0, ptr @k.alpha, i64 5, i64 11)
  %p2 = call i32 @universe_ds_art_sharded_put(ptr %m0, ptr @k.beta, i64 4, i64 22)
  %p3 = call i32 @universe_ds_art_sharded_put(ptr %m0, ptr @k.gamma, i64 5, i64 33)
  %p1ok = icmp eq i32 %p1, 0
  call void @ut_check(i1 %p1ok, ptr @m.put)

  %g1 = call i32 @universe_ds_art_sharded_get(ptr %m0, ptr @k.alpha, i64 5, ptr %vslot)
  %g1ok = icmp eq i32 %g1, 0
  call void @ut_check(i1 %g1ok, ptr @m.get)
  %g1v = load i64, ptr %vslot, align 8
  call void @ut_check_eq(i64 %g1v, i64 11, ptr @m.get)

  %ct = call i32 @universe_ds_art_sharded_contains(ptr %m0, ptr @k.beta, i64 4)
  %cteq = icmp eq i32 %ct, 1
  call void @ut_check(i1 %cteq, ptr @m.get)
  %ct2 = call i32 @universe_ds_art_sharded_contains(ptr %m0, ptr @k.absent, i64 4)
  %ct2eq = icmp eq i32 %ct2, 0
  call void @ut_check(i1 %ct2eq, ptr @m.get)

  %l0 = call i64 @universe_ds_art_sharded_len(ptr %m0)
  call void @ut_check_eq(i64 %l0, i64 3, ptr @m.len)

  ; prefix_scan "alp" -> alpha (1) [collected across shards]
  %pctx = alloca i64, align 8
  store i64 0, ptr %pctx, align 8
  %ps = call i64 @universe_ds_art_sharded_prefix_scan(ptr %m0, ptr @k.alp, i64 3, ptr @cnt_cb, ptr %pctx)
  call void @ut_check_eq(i64 %ps, i64 1, ptr @m.pscan)

  %d1 = call i32 @universe_ds_art_sharded_delete(ptr %m0, ptr @k.beta, i64 4)
  %d1ok = icmp eq i32 %d1, 0
  call void @ut_check(i1 %d1ok, ptr @m.del)
  %l1 = call i64 @universe_ds_art_sharded_len(ptr %m0)
  call void @ut_check_eq(i64 %l1, i64 2, ptr @m.len)
  call void @universe_ds_art_sharded_destroy(ptr %m0)

  ; ---- concurrent stress: 10 iterations ----
  br label %iter

iter:
  %it = phi i64 [ 0, %entry ], [ %itn, %iterend ]
  %djviol = phi i64 [ 0, %entry ], [ %djviol.n, %iterend ]
  %djcntbad = phi i64 [ 0, %entry ], [ %djcntbad.n, %iterend ]
  %ovviol = phi i64 [ 0, %entry ], [ %ovviol.n, %iterend ]
  %ovlenbad = phi i64 [ 0, %entry ], [ %ovlenbad.n, %iterend ]

  ; --- disjoint phase (mode 0) ---
  %mapA = call ptr @universe_ds_art_sharded_create(i64 8)
  br label %spawnA
spawnA:
  %sa = phi i64 [ 0, %iter ], [ %san, %spawnA ]
  %aoff = shl i64 %sa, 5
  %ap = getelementptr inbounds i8, ptr %args, i64 %aoff
  store ptr %mapA, ptr %ap, align 8
  %atidp = getelementptr inbounds i8, ptr %ap, i64 8
  store i64 %sa, ptr %atidp, align 8
  %amodep = getelementptr inbounds i8, ptr %ap, i64 16
  store i64 0, ptr %amodep, align 8
  %hp = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %sa
  %crc = call i32 @pthread_create(ptr %hp, ptr null, ptr @worker, ptr %ap)
  %san = add i64 %sa, 1
  %smore = icmp ult i64 %san, 4
  br i1 %smore, label %spawnA, label %joinA
joinA:
  %ja = phi i64 [ 0, %spawnA ], [ %jan, %joinA ]
  %jhp = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %ja
  %jh = load i64, ptr %jhp, align 8
  %jhpx = inttoptr i64 %jh to ptr
  %jrc = call i32 @pthread_join(ptr %jhpx, ptr null)
  %jan = add i64 %ja, 1
  %jmore = icmp ult i64 %jan, 4
  br i1 %jmore, label %joinA, label %verifyA
verifyA:
  ; sum worker viols
  br label %valoop
valoop:
  %vi = phi i64 [ 0, %verifyA ], [ %vin, %valoop ]
  %vacc = phi i64 [ 0, %verifyA ], [ %vacc2, %valoop ]
  %voff = shl i64 %vi, 5
  %vap = getelementptr inbounds i8, ptr %args, i64 %voff
  %vvp = getelementptr inbounds i8, ptr %vap, i64 24
  %vv = load i64, ptr %vvp, align 8
  %vacc2 = add i64 %vacc, %vv
  %vin = add i64 %vi, 1
  %vmore = icmp ult i64 %vin, 4
  br i1 %vmore, label %valoop, label %vadone
vadone:
  %djviol.n = add i64 %djviol, %vacc2
  %lenA = call i64 @universe_ds_art_sharded_len(ptr %mapA)
  ; expect 4*16384 = 65536
  %lenAok = icmp eq i64 %lenA, 65536
  %lenAbad = xor i1 %lenAok, true
  %lenAbadz = zext i1 %lenAbad to i64
  %djcntbad.n = add i64 %djcntbad, %lenAbadz
  call void @universe_ds_art_sharded_destroy(ptr %mapA)

  ; --- overlapping phase (mode 1) ---
  %mapB = call ptr @universe_ds_art_sharded_create(i64 8)
  br label %spawnB
spawnB:
  %sb = phi i64 [ 0, %vadone ], [ %sbn, %spawnB ]
  %boff = shl i64 %sb, 5
  %bp = getelementptr inbounds i8, ptr %args, i64 %boff
  store ptr %mapB, ptr %bp, align 8
  %btidp = getelementptr inbounds i8, ptr %bp, i64 8
  store i64 %sb, ptr %btidp, align 8
  %bmodep = getelementptr inbounds i8, ptr %bp, i64 16
  store i64 1, ptr %bmodep, align 8
  %bhp = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %sb
  %bcrc = call i32 @pthread_create(ptr %bhp, ptr null, ptr @worker, ptr %bp)
  %sbn = add i64 %sb, 1
  %sbmore = icmp ult i64 %sbn, 4
  br i1 %sbmore, label %spawnB, label %joinB
joinB:
  %jb = phi i64 [ 0, %spawnB ], [ %jbn, %joinB ]
  %jbhp = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %jb
  %jbh = load i64, ptr %jbhp, align 8
  %jbhpx = inttoptr i64 %jbh to ptr
  %jbrc = call i32 @pthread_join(ptr %jbhpx, ptr null)
  %jbn = add i64 %jb, 1
  %jbmore = icmp ult i64 %jbn, 4
  br i1 %jbmore, label %joinB, label %verifyB
verifyB:
  ; scan shared key space: present keys must have value k+1000000; count present
  br label %vbloop
vbloop:
  %ki = phi i64 [ 0, %verifyB ], [ %kin, %vbcont ]
  %present = phi i64 [ 0, %verifyB ], [ %present2, %vbcont ]
  %badval = phi i64 [ 0, %verifyB ], [ %badval2, %vbcont ]
  store i64 %ki, ptr %kbuf, align 8
  %br = call i32 @universe_ds_art_sharded_get(ptr %mapB, ptr %kbuf, i64 8, ptr %vslot)
  %bok = icmp eq i32 %br, 0
  br i1 %bok, label %vbpresent, label %vbcont2
vbpresent:
  %bv = load i64, ptr %vslot, align 8
  %want = add i64 %ki, 1000000
  %bvok = icmp eq i64 %bv, %want
  %bvbad = xor i1 %bvok, true
  %bvbadz = zext i1 %bvbad to i64
  br label %vbcont2
vbcont2:
  %pinc = phi i64 [ 1, %vbpresent ], [ 0, %vbloop ]
  %vinc = phi i64 [ %bvbadz, %vbpresent ], [ 0, %vbloop ]
  br label %vbcont
vbcont:
  %present2 = add i64 %present, %pinc
  %badval2 = add i64 %badval, %vinc
  %kin = add i64 %ki, 1
  %kmore = icmp ult i64 %kin, 8192
  br i1 %kmore, label %vbloop, label %vbdone
vbdone:
  %ovviol.n = add i64 %ovviol, %badval2
  %lenB = call i64 @universe_ds_art_sharded_len(ptr %mapB)
  %lenBok = icmp eq i64 %lenB, %present2
  %lenBbad = xor i1 %lenBok, true
  %lenBbadz = zext i1 %lenBbad to i64
  %ovlenbad.n = add i64 %ovlenbad, %lenBbadz
  call void @universe_ds_art_sharded_destroy(ptr %mapB)
  br label %iterend

iterend:
  %itn = add i64 %it, 1
  %itmore = icmp ult i64 %itn, 10
  br i1 %itmore, label %iter, label %summary

summary:
  call void @ut_check_eq(i64 %djviol.n, i64 0, ptr @m.dj)
  call void @ut_check_eq(i64 %djcntbad.n, i64 0, ptr @m.djcnt)
  call void @ut_check_eq(i64 %ovviol.n, i64 0, ptr @m.ov)
  call void @ut_check_eq(i64 %ovlenbad.n, i64 0, ptr @m.ovlen)

  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %finish

bench:
  ; 4 threads x (16384 put + 16384 get) disjoint = 131072 ops
  ; 8 shards (parallel) vs 1 shard (single-lock); re-create map each rep
  br label %a8.rep.head
a8.rep.head:
  %a8rep = phi i64 [ 0, %bench ], [ %a8rep.n, %a8.rep.cont ]
  %m8 = call ptr @universe_ds_art_sharded_create(i64 8)
  %bt0 = call double @ut_now_sec()
  br label %bsp8
bsp8:
  %s8 = phi i64 [ 0, %a8.rep.head ], [ %s8n, %bsp8 ]
  %o8 = shl i64 %s8, 5
  %ap8 = getelementptr inbounds i8, ptr %args, i64 %o8
  store ptr %m8, ptr %ap8, align 8
  %tp8 = getelementptr inbounds i8, ptr %ap8, i64 8
  store i64 %s8, ptr %tp8, align 8
  %mp8 = getelementptr inbounds i8, ptr %ap8, i64 16
  store i64 0, ptr %mp8, align 8
  %h8 = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %s8
  %c8 = call i32 @pthread_create(ptr %h8, ptr null, ptr @worker, ptr %ap8)
  %s8n = add i64 %s8, 1
  %s8m = icmp ult i64 %s8n, 4
  br i1 %s8m, label %bsp8, label %bj8
bj8:
  %j8 = phi i64 [ 0, %bsp8 ], [ %j8n, %bj8 ]
  %jh8p = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %j8
  %jh8 = load i64, ptr %jh8p, align 8
  %jh8x = inttoptr i64 %jh8 to ptr
  %jr8 = call i32 @pthread_join(ptr %jh8x, ptr null)
  %j8n = add i64 %j8, 1
  %j8m = icmp ult i64 %j8n, 4
  br i1 %j8m, label %bj8, label %bmid
bmid:
  %bt1 = call double @ut_now_sec()
  call void @universe_ds_art_sharded_destroy(ptr %m8)
  %dt8 = fsub double %bt1, %bt0
  %a8warm = icmp eq i64 %a8rep, 0
  br i1 %a8warm, label %a8.rep.cont, label %a8.rep.store
a8.rep.store:
  %a8si = sub i64 %a8rep, 1
  %a8sp = getelementptr inbounds double, ptr @art8.bench.samp, i64 %a8si
  store double %dt8, ptr %a8sp, align 8
  br label %a8.rep.cont
a8.rep.cont:
  %a8rep.n = add nuw i64 %a8rep, 1
  %a8more = icmp ult i64 %a8rep.n, 17
  br i1 %a8more, label %a8.rep.head, label %a8.report
a8.report:
  call void @ut_report_dist(ptr @art8.bench.samp, i64 16, i64 131072, ptr @lbl.art8.bench)
  br label %a1.rep.head
a1.rep.head:
  %a1rep = phi i64 [ 0, %a8.report ], [ %a1rep.n, %a1.rep.cont ]
  %m1 = call ptr @universe_ds_art_sharded_create(i64 1)
  %ct0 = call double @ut_now_sec()
  br label %bsp1
bsp1:
  %s1 = phi i64 [ 0, %a1.rep.head ], [ %s1n, %bsp1 ]
  %o1 = shl i64 %s1, 5
  %ap1 = getelementptr inbounds i8, ptr %args, i64 %o1
  store ptr %m1, ptr %ap1, align 8
  %tp1 = getelementptr inbounds i8, ptr %ap1, i64 8
  store i64 %s1, ptr %tp1, align 8
  %mp1 = getelementptr inbounds i8, ptr %ap1, i64 16
  store i64 0, ptr %mp1, align 8
  %h1 = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %s1
  %c1 = call i32 @pthread_create(ptr %h1, ptr null, ptr @worker, ptr %ap1)
  %s1n = add i64 %s1, 1
  %s1m = icmp ult i64 %s1n, 4
  br i1 %s1m, label %bsp1, label %bj1
bj1:
  %j1 = phi i64 [ 0, %bsp1 ], [ %j1n, %bj1 ]
  %jh1p = getelementptr inbounds [4 x i64], ptr %handles, i64 0, i64 %j1
  %jh1 = load i64, ptr %jh1p, align 8
  %jh1x = inttoptr i64 %jh1 to ptr
  %jr1 = call i32 @pthread_join(ptr %jh1x, ptr null)
  %j1n = add i64 %j1, 1
  %j1m = icmp ult i64 %j1n, 4
  br i1 %j1m, label %bj1, label %bfin
bfin:
  %ct1 = call double @ut_now_sec()
  call void @universe_ds_art_sharded_destroy(ptr %m1)
  %dt1 = fsub double %ct1, %ct0
  %a1warm = icmp eq i64 %a1rep, 0
  br i1 %a1warm, label %a1.rep.cont, label %a1.rep.store
a1.rep.store:
  %a1si = sub i64 %a1rep, 1
  %a1sp = getelementptr inbounds double, ptr @art1.bench.samp, i64 %a1si
  store double %dt1, ptr %a1sp, align 8
  br label %a1.rep.cont
a1.rep.cont:
  %a1rep.n = add nuw i64 %a1rep, 1
  %a1more = icmp ult i64 %a1rep.n, 17
  br i1 %a1more, label %a1.rep.head, label %a1.report
a1.report:
  call void @ut_report_dist(ptr @art1.bench.samp, i64 16, i64 131072, ptr @lbl.art1.bench)
  br label %finish

finish:
  %r = call i32 @ut_summary()
  ret i32 %r
}
