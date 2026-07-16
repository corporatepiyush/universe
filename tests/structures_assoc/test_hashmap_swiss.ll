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

; Tests for the SIMD tag-group open-addressing hash map.

declare ptr  @universe_ds_hashmap_swiss_create(i64, i64)
declare i32  @universe_ds_hashmap_swiss_put(ptr, i64, ptr)
declare i32  @universe_ds_hashmap_swiss_get(ptr, i64, ptr)
declare i32  @universe_ds_hashmap_swiss_contains(ptr, i64)
declare i32  @universe_ds_hashmap_swiss_remove(ptr, i64)
declare i64  @universe_ds_hashmap_swiss_len(ptr)
declare i64  @universe_ds_hashmap_swiss_capacity(ptr)
declare void @universe_ds_hashmap_swiss_destroy(ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64  @ut_rand(ptr)
declare double @ut_now_sec()
declare i1   @ut_want_bench(i32, ptr)
declare i32  @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@swiss.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lin.bench.samp   = internal global [16 x double] zeroinitializer, align 8
@lbl.swiss.bench = private unnamed_addr constant [13 x i8] c"swiss-get 4M\00"
@lbl.lin.bench   = private unnamed_addr constant [15 x i8] c"linear-scan 8k\00"

declare ptr  @malloc(i64)
declare void @free(ptr)
declare i32  @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)

@msg.c1     = private constant [18 x i8] c"roundtrip get val\00"
@msg.c1rc   = private constant [17 x i8] c"roundtrip get rc\00"
@msg.ow.rc  = private constant [13 x i8] c"overwrite rc\00"
@msg.ow.val = private constant [14 x i8] c"overwrite val\00"
@msg.ow.len = private constant [16 x i8] c"overwrite count\00"
@msg.abs.g  = private constant [16 x i8] c"absent get is 5\00"
@msg.abs.r  = private constant [19 x i8] c"absent remove is 5\00"
@msg.abs.c  = private constant [18 x i8] c"absent contains 5\00"
@msg.pres.c = private constant [17 x i8] c"present cont is0\00"
@msg.rm.rc  = private constant [13 x i8] c"remove ok rc\00"
@msg.rm.get = private constant [18 x i8] c"after remove get5\00"
@msg.rm.len = private constant [16 x i8] c"after remove ct\00"
@msg.rep.rc = private constant [13 x i8] c"re-put rc ok\00"
@msg.rep.v  = private constant [13 x i8] c"re-put value\00"
@msg.rep.l  = private constant [13 x i8] c"re-put count\00"

@msg.gr.len = private constant [15 x i8] c"grow len 100k\00\00"
@msg.gr.cap = private constant [15 x i8] c"grow cap grew\00\00"
@msg.gr.vio = private constant [18 x i8] c"grow verify viols\00"

@msg.ch.vio = private constant [19 x i8] c"churn get mismatch\00"
@msg.ch.mem = private constant [21 x i8] c"churn final mismatch\00"
@msg.ch.len = private constant [16 x i8] c"churn len==pop\00\00"

@msg.co.vio = private constant [16 x i8] c"collision viols\00"
@msg.co.len = private constant [14 x i8] c"collision len\00"


; ---- Tests 1-4: roundtrip, overwrite, absent, remove/re-put --------------
define internal void @t_basic() {
entry:
  %m = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 16)
  %vbuf = alloca i64, align 8
  %obuf = alloca i64, align 8

  ; roundtrip put(1)=100
  store i64 100, ptr %vbuf, align 8
  %p1 = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 1, ptr %vbuf)
  %g1 = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 1, ptr %obuf)
  %rc1ok = icmp eq i32 %g1, 0
  call void @ut_check(i1 %rc1ok, ptr @msg.c1rc)
  %o1 = load i64, ptr %obuf, align 8
  %v1ok = icmp eq i64 %o1, 100
  call void @ut_check(i1 %v1ok, ptr @msg.c1)

  ; overwrite put(1)=200; count stays 1
  store i64 200, ptr %vbuf, align 8
  %p2 = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 1, ptr %vbuf)
  %p2ok = icmp eq i32 %p2, 0
  call void @ut_check(i1 %p2ok, ptr @msg.ow.rc)
  %g2 = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 1, ptr %obuf)
  %o2 = load i64, ptr %obuf, align 8
  %v2ok = icmp eq i64 %o2, 200
  call void @ut_check(i1 %v2ok, ptr @msg.ow.val)
  %len2 = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len2, i64 1, ptr @msg.ow.len)

  ; absent get/remove/contains -> 5 ; present contains -> 0
  %ga = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 999, ptr %obuf)
  %gaok = icmp eq i32 %ga, 5
  call void @ut_check(i1 %gaok, ptr @msg.abs.g)
  %ra = call i32 @universe_ds_hashmap_swiss_remove(ptr %m, i64 999)
  %raok = icmp eq i32 %ra, 5
  call void @ut_check(i1 %raok, ptr @msg.abs.r)
  %ca = call i32 @universe_ds_hashmap_swiss_contains(ptr %m, i64 999)
  %caok = icmp eq i32 %ca, 5
  call void @ut_check(i1 %caok, ptr @msg.abs.c)
  %cp = call i32 @universe_ds_hashmap_swiss_contains(ptr %m, i64 1)
  %cpok = icmp eq i32 %cp, 0
  call void @ut_check(i1 %cpok, ptr @msg.pres.c)

  ; remove then re-get -> 5 ; re-put works
  %rm = call i32 @universe_ds_hashmap_swiss_remove(ptr %m, i64 1)
  %rmok = icmp eq i32 %rm, 0
  call void @ut_check(i1 %rmok, ptr @msg.rm.rc)
  %g3 = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 1, ptr %obuf)
  %g3ok = icmp eq i32 %g3, 5
  call void @ut_check(i1 %g3ok, ptr @msg.rm.get)
  %len3 = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len3, i64 0, ptr @msg.rm.len)
  store i64 300, ptr %vbuf, align 8
  %p4 = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 1, ptr %vbuf)
  %p4ok = icmp eq i32 %p4, 0
  call void @ut_check(i1 %p4ok, ptr @msg.rep.rc)
  %g4 = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 1, ptr %obuf)
  %o4 = load i64, ptr %obuf, align 8
  %v4ok = icmp eq i64 %o4, 300
  call void @ut_check(i1 %v4ok, ptr @msg.rep.v)
  %len4 = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len4, i64 1, ptr @msg.rep.l)

  call void @universe_ds_hashmap_swiss_destroy(ptr %m)
  ret void
}

; ---- Test 5: grow correctness (100k keys, value = key*2654435761) --------
define internal void @t_grow() {
entry:
  %m = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 16)
  %vbuf = alloca i64, align 8
  %obuf = alloca i64, align 8
  br label %ins

ins:
  %i = phi i64 [ 1, %entry ], [ %i.n, %ins ]
  %val = mul i64 %i, 2654435761
  store i64 %val, ptr %vbuf, align 8
  %p = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 %i, ptr %vbuf)
  %i.n = add nuw i64 %i, 1
  %ins.done = icmp ugt i64 %i.n, 100000
  br i1 %ins.done, label %chklen, label %ins

chklen:
  %len = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len, i64 100000, ptr @msg.gr.len)
  %cap = call i64 @universe_ds_hashmap_swiss_capacity(ptr %m)
  %capok = icmp uge i64 %cap, 131072
  call void @ut_check(i1 %capok, ptr @msg.gr.cap)
  br label %ver

ver:
  %j = phi i64 [ 1, %chklen ], [ %j.n, %ver.cont ]
  %vio = phi i64 [ 0, %chklen ], [ %vio.n, %ver.cont ]
  %g = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 %j, ptr %obuf)
  %exp = mul i64 %j, 2654435761
  %got = load i64, ptr %obuf, align 8
  %rcbad = icmp ne i32 %g, 0
  %valbad = icmp ne i64 %got, %exp
  %bad = or i1 %rcbad, %valbad
  %badi = zext i1 %bad to i64
  %vio.n = add i64 %vio, %badi
  %j.n = add nuw i64 %j, 1
  %ver.done = icmp ugt i64 %j.n, 100000
  br i1 %ver.done, label %fin, label %ver.cont

ver.cont:
  br label %ver

fin:
  call void @ut_check_eq(i64 %vio.n, i64 0, ptr @msg.gr.vio)
  call void @universe_ds_hashmap_swiss_destroy(ptr %m)
  ret void
}

; ---- Test 6: tombstone churn vs shadow bitset (domain 4096) --------------
define internal void @t_churn() {
entry:
  %m = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 16)
  %vbuf = alloca i64, align 8
  %obuf = alloca i64, align 8
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  %shadow = alloca [4096 x i8], align 16
  call void @llvm.memset.p0.i64(ptr %shadow, i8 0, i64 4096, i1 false)
  br label %loop

loop:
  %it = phi i64 [ 0, %entry ], [ %it.n, %loop.cont ]
  %vio = phi i64 [ 0, %entry ], [ %vio.n, %loop.cont ]
  %r = call i64 @ut_rand(ptr %state)
  %key = and i64 %r, 4095
  %opsh = lshr i64 %r, 12
  %op = and i64 %opsh, 3
  %sh.p = getelementptr inbounds [4096 x i8], ptr %shadow, i64 0, i64 %key
  %val = mul i64 %key, 2654435761
  %is.ins = icmp ule i64 %op, 1
  br i1 %is.ins, label %do.ins, label %not.ins

do.ins:
  store i64 %val, ptr %vbuf, align 8
  %pi = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 %key, ptr %vbuf)
  store i8 1, ptr %sh.p, align 1
  br label %loop.cont

not.ins:
  %is.rm = icmp eq i64 %op, 2
  br i1 %is.rm, label %do.rm, label %do.get

do.rm:
  %rr = call i32 @universe_ds_hashmap_swiss_remove(ptr %m, i64 %key)
  store i8 0, ptr %sh.p, align 1
  br label %loop.cont

do.get:
  %gr = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 %key, ptr %obuf)
  %present8 = load i8, ptr %sh.p, align 1
  %present = icmp ne i8 %present8, 0
  %got = load i64, ptr %obuf, align 8
  ; expected: present => rc==0 && got==val ; absent => rc==5
  %rc0 = icmp eq i32 %gr, 0
  %valok = icmp eq i64 %got, %val
  %hit.ok = and i1 %rc0, %valok
  %rc5 = icmp eq i32 %gr, 5
  %get.ok = select i1 %present, i1 %hit.ok, i1 %rc5
  %get.bad = xor i1 %get.ok, true
  %gb = zext i1 %get.bad to i64
  %vio.get = add i64 %vio, %gb
  br label %loop.cont

loop.cont:
  %vio.n = phi i64 [ %vio, %do.ins ], [ %vio, %do.rm ], [ %vio.get, %do.get ]
  %it.n = add nuw i64 %it, 1
  %loop.done = icmp uge i64 %it.n, 200000
  br i1 %loop.done, label %final, label %loop

final:
  call void @ut_check_eq(i64 %vio.n, i64 0, ptr @msg.ch.vio)
  br label %scan

scan:
  %k = phi i64 [ 0, %final ], [ %k.n, %scan.cont ]
  %mm = phi i64 [ 0, %final ], [ %mm.n, %scan.cont ]
  %pop = phi i64 [ 0, %final ], [ %pop.n, %scan.cont ]
  %s.p = getelementptr inbounds [4096 x i8], ptr %shadow, i64 0, i64 %k
  %s8 = load i8, ptr %s.p, align 1
  %s = icmp ne i8 %s8, 0
  %c = call i32 @universe_ds_hashmap_swiss_contains(ptr %m, i64 %k)
  %c.has = icmp eq i32 %c, 0
  %match = icmp eq i1 %c.has, %s
  %mism = xor i1 %match, true
  %mi = zext i1 %mism to i64
  %mm.n = add i64 %mm, %mi
  %si = zext i1 %s to i64
  %pop.n = add i64 %pop, %si
  %k.n = add nuw i64 %k, 1
  %scan.done = icmp uge i64 %k.n, 4096
  br i1 %scan.done, label %scan.fin, label %scan.cont

scan.cont:
  br label %scan

scan.fin:
  call void @ut_check_eq(i64 %mm.n, i64 0, ptr @msg.ch.mem)
  %len = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len, i64 %pop.n, ptr @msg.ch.len)
  call void @universe_ds_hashmap_swiss_destroy(ptr %m)
  ret void
}

; ---- Test 7: collision stress (keys share low 20 bits) -------------------
define internal void @t_collision() {
entry:
  %m = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 16)
  %vbuf = alloca i64, align 8
  %obuf = alloca i64, align 8
  br label %ins

ins:
  %i = phi i64 [ 1, %entry ], [ %i.n, %ins ]
  %key = shl i64 %i, 20
  store i64 %i, ptr %vbuf, align 8
  %p = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 %key, ptr %vbuf)
  %i.n = add nuw i64 %i, 1
  %ins.done = icmp ugt i64 %i.n, 3000
  br i1 %ins.done, label %ver, label %ins

ver:
  %j = phi i64 [ 1, %ins ], [ %j.n, %ver.cont ]
  %vio = phi i64 [ 0, %ins ], [ %vio.n, %ver.cont ]
  %kj = shl i64 %j, 20
  %g = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 %kj, ptr %obuf)
  %got = load i64, ptr %obuf, align 8
  %rcbad = icmp ne i32 %g, 0
  %valbad = icmp ne i64 %got, %j
  %bad = or i1 %rcbad, %valbad
  %badi = zext i1 %bad to i64
  %vio.n = add i64 %vio, %badi
  %j.n = add nuw i64 %j, 1
  %ver.done = icmp ugt i64 %j.n, 3000
  br i1 %ver.done, label %fin, label %ver.cont

ver.cont:
  br label %ver

fin:
  call void @ut_check_eq(i64 %vio.n, i64 0, ptr @msg.co.vio)
  %len = call i64 @universe_ds_hashmap_swiss_len(ptr %m)
  call void @ut_check_eq(i64 %len, i64 3000, ptr @msg.co.len)
  call void @universe_ds_hashmap_swiss_destroy(ptr %m)
  ret void
}

; ---- Bench: hot get throughput vs linear scan ----------------------------
define internal void @run_bench() {
entry:
  %N = add i64 0, 65536
  %m = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 %N)
  %vbuf = alloca i64, align 8
  %obuf = alloca i64, align 8
  %karr = call ptr @malloc(i64 524288)      ; 65536 * 8
  %varr = call ptr @malloc(i64 524288)
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %key = add i64 %i, 1
  %val = mul i64 %key, 2654435761
  store i64 %val, ptr %vbuf, align 8
  %p = call i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 %key, ptr %vbuf)
  %kp = getelementptr inbounds i64, ptr %karr, i64 %i
  store i64 %key, ptr %kp, align 8
  %vp = getelementptr inbounds i64, ptr %varr, i64 %i
  store i64 %val, ptr %vp, align 8
  %i.n = add nuw i64 %i, 1
  %fill.done = icmp uge i64 %i.n, 65536
  br i1 %fill.done, label %hot.rep.head, label %fill

; --- swiss get distribution (read-only, warm-up rep discarded) ---
hot.rep.head:
  %hrep = phi i64 [ 0, %fill ], [ %hrep.n, %hot.rep.cont ]
  %t0 = call double @ut_now_sec()
  br label %hot

hot:
  %q = phi i64 [ 0, %hot.rep.head ], [ %q.n, %hot ]
  %acc = phi i64 [ 0, %hot.rep.head ], [ %acc.n, %hot ]
  %qk.m = and i64 %q, 65535
  %qk = add i64 %qk.m, 1
  %gr = call i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 %qk, ptr %obuf)
  %got = load i64, ptr %obuf, align 8
  %acc.n = add i64 %acc, %got
  %q.n = add nuw i64 %q, 1
  %hot.done = icmp uge i64 %q.n, 4194304
  br i1 %hot.done, label %hot.rep.done, label %hot

hot.rep.done:
  store volatile i64 %acc.n, ptr %vbuf, align 8
  %t1 = call double @ut_now_sec()
  %hot.dt = fsub double %t1, %t0
  %hot.warm = icmp eq i64 %hrep, 0
  br i1 %hot.warm, label %hot.rep.cont, label %hot.rep.store
hot.rep.store:
  %hot.si = sub i64 %hrep, 1
  %hot.sp = getelementptr inbounds double, ptr @swiss.bench.samp, i64 %hot.si
  store double %hot.dt, ptr %hot.sp, align 8
  br label %hot.rep.cont
hot.rep.cont:
  %hrep.n = add nuw i64 %hrep, 1
  %hot.more = icmp ult i64 %hrep.n, 17
  br i1 %hot.more, label %hot.rep.head, label %hot.report
hot.report:
  call void @ut_report_dist(ptr @swiss.bench.samp, i64 16, i64 4194304, ptr @lbl.swiss.bench)
  br label %lin.rep.head

; --- linear scan reference distribution: 8192 queries, each scans up to N ---
lin.rep.head:
  %lrep = phi i64 [ 0, %hot.report ], [ %lrep.n, %lin.rep.cont ]
  %t2 = call double @ut_now_sec()
  br label %lin

lin:
  %lq = phi i64 [ 0, %lin.rep.head ], [ %lq.n, %lin.cont ]
  %lacc = phi i64 [ 0, %lin.rep.head ], [ %lacc.next, %lin.cont ]
  %lqk.m = and i64 %lq, 65535
  %lqk = add i64 %lqk.m, 1
  br label %scan

scan:
  %s = phi i64 [ 0, %lin ], [ %s.n, %scan.cont ]
  %sacc = phi i64 [ %lacc, %lin ], [ %sacc.sel, %scan.cont ]
  %skp = getelementptr inbounds i64, ptr %karr, i64 %s
  %sk = load i64, ptr %skp, align 8
  %hitq = icmp eq i64 %sk, %lqk
  %svp = getelementptr inbounds i64, ptr %varr, i64 %s
  %sv = load i64, ptr %svp, align 8
  %sacc.hit = add i64 %sacc, %sv
  %sacc.sel = select i1 %hitq, i64 %sacc.hit, i64 %sacc
  %s.n = add nuw i64 %s, 1
  %scan.done = icmp uge i64 %s.n, 65536
  br i1 %scan.done, label %lin.cont, label %scan.cont

scan.cont:
  br label %scan

lin.cont:
  %lacc.next = phi i64 [ %sacc.sel, %scan ]
  %lq.n = add nuw i64 %lq, 1
  %lin.done = icmp uge i64 %lq.n, 8192
  br i1 %lin.done, label %lin.rep.done, label %lin

lin.rep.done:
  store volatile i64 %lacc.next, ptr %vbuf, align 8
  %t3 = call double @ut_now_sec()
  %lin.dt = fsub double %t3, %t2
  %lin.warm = icmp eq i64 %lrep, 0
  br i1 %lin.warm, label %lin.rep.cont, label %lin.rep.store
lin.rep.store:
  %lin.si = sub i64 %lrep, 1
  %lin.sp = getelementptr inbounds double, ptr @lin.bench.samp, i64 %lin.si
  store double %lin.dt, ptr %lin.sp, align 8
  br label %lin.rep.cont
lin.rep.cont:
  %lrep.n = add nuw i64 %lrep, 1
  %lin.more = icmp ult i64 %lrep.n, 17
  br i1 %lin.more, label %lin.rep.head, label %lin.report
lin.report:
  call void @ut_report_dist(ptr @lin.bench.samp, i64 16, i64 8192, ptr @lbl.lin.bench)
  call void @free(ptr %karr)
  call void @free(ptr %varr)
  call void @universe_ds_hashmap_swiss_destroy(ptr %m)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @t_basic()
  call void @t_grow()
  call void @t_churn()
  call void @t_collision()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary

do.bench:
  call void @run_bench()
  br label %summary

summary:
  %r = call i32 @ut_summary()
  ret i32 %r
}
