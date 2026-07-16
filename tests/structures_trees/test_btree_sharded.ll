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

; Tests for the range-sharded concurrent B-tree (universe_ds_btree_sharded_*).

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(ptr, ptr)

declare i64 @ut_rand(ptr)
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@sh.bench.samp = internal global [16 x double] zeroinitializer, align 8
@ol.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.sh.bench = private unnamed_addr constant [22 x i8] c"btree sharded put+get\00"
@lbl.ol.bench = private unnamed_addr constant [21 x i8] c"btree 1-lock put+get\00"

declare ptr @universe_ds_btree_sharded_create(i64)
declare void @universe_ds_btree_sharded_destroy(ptr)
declare i32 @universe_ds_btree_sharded_put(ptr, i64, i64)
declare i32 @universe_ds_btree_sharded_get(ptr, i64, ptr)
declare i32 @universe_ds_btree_sharded_delete(ptr, i64)
declare i32 @universe_ds_btree_sharded_contains(ptr, i64)
declare i64 @universe_ds_btree_sharded_len(ptr)
declare i32 @universe_ds_btree_sharded_min(ptr, ptr, ptr)
declare i32 @universe_ds_btree_sharded_max(ptr, ptr, ptr)
declare i32 @universe_ds_btree_sharded_floor(ptr, i64, ptr, ptr)
declare i32 @universe_ds_btree_sharded_ceiling(ptr, i64, ptr, ptr)
declare i64 @universe_ds_btree_sharded_range(ptr, i64, i64, ptr, ptr, i64)
declare i64 @universe_ds_btree_sharded_shards(ptr)

; single-threaded btree for the bench baseline (one lock == the whole map)
declare ptr @universe_ds_btree_create()
declare void @universe_ds_btree_destroy(ptr)
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)

@msg.shards   = private unnamed_addr constant [16 x i8] c"shards pow2 >=1\00"
@msg.st.put   = private unnamed_addr constant [16 x i8] c"st: put rc == 0\00"
@msg.st.len   = private unnamed_addr constant [13 x i8] c"st: len == N\00"
@msg.st.get   = private unnamed_addr constant [16 x i8] c"st: get correct\00"
@msg.st.abs   = private unnamed_addr constant [16 x i8] c"st: absent == 5\00"
@msg.st.mm    = private unnamed_addr constant [14 x i8] c"st: min & max\00"
@msg.st.fl    = private unnamed_addr constant [16 x i8] c"st: floor/ceil \00"
@msg.st.rng   = private unnamed_addr constant [16 x i8] c"st: range order\00"
@msg.st.del   = private unnamed_addr constant [14 x i8] c"st: delete ok\00"

@msg.mt.lost  = private unnamed_addr constant [22 x i8] c"mt: disjoint no loss \00"
@msg.mt.val   = private unnamed_addr constant [22 x i8] c"mt: values consistent\00"
@msg.mt.len   = private unnamed_addr constant [19 x i8] c"mt: len consistent\00"


; ---- thread argument record (48 B): map@0 base@8 nops@16 seed@24 span@32 viol@40
; each thread owns keys [base, base+span) — DISJOINT ranges => no lost inserts.

define ptr @stress_worker(ptr %arg) {
entry:
  %seedslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %mapp = getelementptr inbounds nuw i8, ptr %arg, i64 0
  %map = load ptr, ptr %mapp, align 8
  %basep = getelementptr inbounds nuw i8, ptr %arg, i64 8
  %base = load i64, ptr %basep, align 8
  %nopsp = getelementptr inbounds nuw i8, ptr %arg, i64 16
  %nops = load i64, ptr %nopsp, align 8
  %seedp = getelementptr inbounds nuw i8, ptr %arg, i64 24
  %seed = load i64, ptr %seedp, align 8
  store i64 %seed, ptr %seedslot, align 8
  %spanp = getelementptr inbounds nuw i8, ptr %arg, i64 32
  %span = load i64, ptr %spanp, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %cont ]
  %r = call i64 @ut_rand(ptr %seedslot)
  %slot = urem i64 %r, %span
  %key = add i64 %base, %slot
  ; value encodes the key so any reader can validate: val = key*2+1
  %val = mul i64 %key, 2
  %val1 = add i64 %val, 1
  %op = and i64 %r, 3
  ; op 0,1 -> put ; op 2 -> delete ; op 3 -> get+validate
  %isput = icmp ult i64 %op, 2
  br i1 %isput, label %doput, label %chkdel

doput:
  %prc = call i32 @universe_ds_btree_sharded_put(ptr %map, i64 %key, i64 %val1)
  %pbad = icmp ne i32 %prc, 0
  %pbadz = zext i1 %pbad to i64
  br label %cont

chkdel:
  %isdel = icmp eq i64 %op, 2
  br i1 %isdel, label %dodel, label %doget

dodel:
  %drc = call i32 @universe_ds_btree_sharded_delete(ptr %map, i64 %key)
  ; rc is 0 (present) or 5 (absent) — both legal
  %dok = icmp eq i32 %drc, 0
  %dok5 = icmp eq i32 %drc, 5
  %dokall = or i1 %dok, %dok5
  %dbad = xor i1 %dokall, true
  %dbadz = zext i1 %dbad to i64
  br label %cont

doget:
  %grc = call i32 @universe_ds_btree_sharded_get(ptr %map, i64 %key, ptr %vslot)
  %gv = load i64, ptr %vslot, align 8
  %gpresent = icmp eq i32 %grc, 0
  ; if present, value MUST equal key*2+1
  %gvok = icmp eq i64 %gv, %val1
  %gbad0 = and i1 %gpresent, %gvok
  ; bad only when present but wrong value
  %gwrong = and i1 %gpresent, %gvok
  %gcorrupt = xor i1 %gvok, true
  %greport = and i1 %gpresent, %gcorrupt
  %gbadz = zext i1 %greport to i64
  br label %cont

cont:
  %step = phi i64 [ %pbadz, %doput ], [ %dbadz, %dodel ], [ %gbadz, %doget ]
  %viol.n = add i64 %viol, %step
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %nops
  br i1 %more, label %loop, label %fin

fin:
  %violp = getelementptr inbounds nuw i8, ptr %arg, i64 40
  store i64 %viol.n, ptr %violp, align 8
  ret ptr null
}

; disjoint-insert worker: inserts EVERY key in its own range exactly once, then
; verifies each is present with the right value (no other thread touches it).
define ptr @disjoint_worker(ptr %arg) {
entry:
  %vslot = alloca i64, align 8
  %mapp = getelementptr inbounds nuw i8, ptr %arg, i64 0
  %map = load ptr, ptr %mapp, align 8
  %basep = getelementptr inbounds nuw i8, ptr %arg, i64 8
  %base = load i64, ptr %basep, align 8
  %spanp = getelementptr inbounds nuw i8, ptr %arg, i64 32
  %span = load i64, ptr %spanp, align 8
  br label %insloop

insloop:
  %i = phi i64 [ 0, %entry ], [ %in, %insloop ]
  %key = add i64 %base, %i
  %val = mul i64 %key, 2
  %val1 = add i64 %val, 1
  %rc = call i32 @universe_ds_btree_sharded_put(ptr %map, i64 %key, i64 %val1)
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %span
  br i1 %more, label %insloop, label %verloop

verloop:
  %j = phi i64 [ 0, %insloop ], [ %jn, %verloop ]
  %viol = phi i64 [ 0, %insloop ], [ %viol.n, %verloop ]
  %vkey = add i64 %base, %j
  %vexp = mul i64 %vkey, 2
  %vexp1 = add i64 %vexp, 1
  %grc = call i32 @universe_ds_btree_sharded_get(ptr %map, i64 %vkey, ptr %vslot)
  %gv = load i64, ptr %vslot, align 8
  %gok0 = icmp eq i32 %grc, 0
  %gok1 = icmp eq i64 %gv, %vexp1
  %gok = and i1 %gok0, %gok1
  %gbad = xor i1 %gok, true
  %gbadz = zext i1 %gbad to i64
  %viol.n = add i64 %viol, %gbadz
  %jn = add i64 %j, 1
  %jmore = icmp ult i64 %jn, %span
  br i1 %jmore, label %verloop, label %fin

fin:
  %violp = getelementptr inbounds nuw i8, ptr %arg, i64 40
  store i64 %viol, ptr %violp, align 8
  ret ptr null
}

; ---------------------------------------------------------------------------
; single-thread mirror of the array-btree semantics through the sharded API
; ---------------------------------------------------------------------------
define void @test_single() {
entry:
  %kslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %okb = alloca [64 x i64], align 8
  %ovb = alloca [64 x i64], align 8
  %m = call ptr @universe_ds_btree_sharded_create(i64 8)
  %ns = call i64 @universe_ds_btree_sharded_shards(ptr %m)
  %nsok = icmp eq i64 %ns, 8
  call void @ut_check(i1 %nsok, ptr @msg.shards)

  ; put keys 0..999 (spans several shards), val = key*3
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %in, %fill ]
  %viput = phi i64 [ 0, %entry ], [ %viput.n, %fill ]
  %val = mul i64 %i, 3
  %rc = call i32 @universe_ds_btree_sharded_put(ptr %m, i64 %i, i64 %val)
  %bad = icmp ne i32 %rc, 0
  %badz = zext i1 %bad to i64
  %viput.n = add i64 %viput, %badz
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, 1000
  br i1 %more, label %fill, label %fill.done

fill.done:
  call void @ut_check_eq(i64 %viput, i64 0, ptr @msg.st.put)
  %len = call i64 @universe_ds_btree_sharded_len(ptr %m)
  call void @ut_check_eq(i64 %len, i64 1000, ptr @msg.st.len)
  br label %getall

getall:
  %g.i = phi i64 [ 0, %fill.done ], [ %g.in, %getall ]
  %g.vi = phi i64 [ 0, %fill.done ], [ %g.vi.n, %getall ]
  %grc = call i32 @universe_ds_btree_sharded_get(ptr %m, i64 %g.i, ptr %vslot)
  %gv = load i64, ptr %vslot, align 8
  %gexp = mul i64 %g.i, 3
  %g0 = icmp eq i32 %grc, 0
  %g1 = icmp eq i64 %gv, %gexp
  %gok = and i1 %g0, %g1
  %gbad = xor i1 %gok, true
  %gbadz = zext i1 %gbad to i64
  %g.vi.n = add i64 %g.vi, %gbadz
  %g.in = add i64 %g.i, 1
  %g.more = icmp ult i64 %g.in, 1000
  br i1 %g.more, label %getall, label %getall.done

getall.done:
  call void @ut_check_eq(i64 %g.vi, i64 0, ptr @msg.st.get)
  ; absent
  %ab = call i32 @universe_ds_btree_sharded_get(ptr %m, i64 100000, ptr %vslot)
  %abok = icmp eq i32 %ab, 5
  %cn = call i32 @universe_ds_btree_sharded_contains(ptr %m, i64 100000)
  %cnok = icmp eq i32 %cn, 0
  %aball = and i1 %abok, %cnok
  call void @ut_check(i1 %aball, ptr @msg.st.abs)

  ; min == 0, max == 999
  %mnrc = call i32 @universe_ds_btree_sharded_min(ptr %m, ptr %kslot, ptr %vslot)
  %mnk = load i64, ptr %kslot, align 8
  %mxrc = call i32 @universe_ds_btree_sharded_max(ptr %m, ptr %kslot, ptr %vslot)
  %mxk = load i64, ptr %kslot, align 8
  %mnok = icmp eq i64 %mnk, 0
  %mxok = icmp eq i64 %mxk, 999
  %mmok = and i1 %mnok, %mxok
  call void @ut_check(i1 %mmok, ptr @msg.st.mm)

  ; floor(500)=500, ceiling(500)=500, floor(-5)=none, ceiling(100000)=none
  %fl = call i32 @universe_ds_btree_sharded_floor(ptr %m, i64 500, ptr %kslot, ptr %vslot)
  %flk = load i64, ptr %kslot, align 8
  %fl0 = icmp eq i32 %fl, 0
  %fl1 = icmp eq i64 %flk, 500
  %flok = and i1 %fl0, %fl1
  %ce = call i32 @universe_ds_btree_sharded_ceiling(ptr %m, i64 500, ptr %kslot, ptr %vslot)
  %cek = load i64, ptr %kslot, align 8
  %ce0 = icmp eq i32 %ce, 0
  %ce1 = icmp eq i64 %cek, 500
  %ceok = and i1 %ce0, %ce1
  %fln = call i32 @universe_ds_btree_sharded_floor(ptr %m, i64 -5, ptr %kslot, ptr %vslot)
  %flnok = icmp eq i32 %fln, 5
  %cen = call i32 @universe_ds_btree_sharded_ceiling(ptr %m, i64 100000, ptr %kslot, ptr %vslot)
  %cenok = icmp eq i32 %cen, 5
  %fa = and i1 %flok, %ceok
  %fb = and i1 %fa, %flnok
  %fc = and i1 %fb, %cenok
  call void @ut_check(i1 %fc, ptr @msg.st.fl)

  ; range(100,159) crosses shard boundaries -> exactly 60 keys, globally sorted
  %rn = call i64 @universe_ds_btree_sharded_range(ptr %m, i64 100, i64 159, ptr %okb, ptr %ovb, i64 64)
  %rnok = icmp eq i64 %rn, 60
  br label %rchk

rchk:
  %r.i = phi i64 [ 0, %getall.done ], [ %r.in, %rchk ]
  %r.vi = phi i64 [ 0, %getall.done ], [ %r.vi.n, %rchk ]
  %r.kp = getelementptr inbounds nuw i64, ptr %okb, i64 %r.i
  %r.k = load i64, ptr %r.kp, align 8
  %r.exp = add i64 %r.i, 100
  %r.bad = icmp ne i64 %r.k, %r.exp
  %r.badz = zext i1 %r.bad to i64
  %r.vi.n = add i64 %r.vi, %r.badz
  %r.in = add i64 %r.i, 1
  %r.more = icmp ult i64 %r.in, 60
  br i1 %r.more, label %rchk, label %rchk.done

rchk.done:
  %rvzero = icmp eq i64 %r.vi, 0
  %rall = and i1 %rnok, %rvzero
  call void @ut_check(i1 %rall, ptr @msg.st.rng)

  ; delete evens, check
  br label %del

del:
  %d.k = phi i64 [ 0, %rchk.done ], [ %d.kn, %del ]
  %d.vi = phi i64 [ 0, %rchk.done ], [ %d.vi.n, %del ]
  %drc = call i32 @universe_ds_btree_sharded_delete(ptr %m, i64 %d.k)
  %d0 = icmp eq i32 %drc, 0
  %dbad = xor i1 %d0, true
  %dbadz = zext i1 %dbad to i64
  %d.vi.n = add i64 %d.vi, %dbadz
  %d.kn = add i64 %d.k, 2
  %d.more = icmp ult i64 %d.kn, 1000
  br i1 %d.more, label %del, label %del.done

del.done:
  %len2 = call i64 @universe_ds_btree_sharded_len(ptr %m)
  %len2ok = icmp eq i64 %len2, 500
  %ev = call i32 @universe_ds_btree_sharded_get(ptr %m, i64 500, ptr %vslot)
  %evok = icmp eq i32 %ev, 5
  %od = call i32 @universe_ds_btree_sharded_get(ptr %m, i64 501, ptr %vslot)
  %odok = icmp eq i32 %od, 0
  %delvz = icmp eq i64 %d.vi, 0
  %da = and i1 %len2ok, %evok
  %db = and i1 %da, %odok
  %dc = and i1 %db, %delvz
  call void @ut_check(i1 %dc, ptr @msg.st.del)

  call void @universe_ds_btree_sharded_destroy(ptr %m)
  ret void
}

; ---------------------------------------------------------------------------
; multithread stress: T threads, disjoint ranges, mixed put/get/delete;
; plus a disjoint-only pass that must never lose an insert. Looped by caller.
; ---------------------------------------------------------------------------
define i64 @stress_once(i64 %nthreads, i64 %nops, i64 %seedbase, i64 %overlap) {
entry:
  ; args: T records of 48 B; tids: T x 8B
  %argsz = mul i64 %nthreads, 48
  %args = call ptr @malloc(i64 %argsz)
  %tidsz = mul i64 %nthreads, 8
  %tids = call ptr @malloc(i64 %tidsz)
  %m = call ptr @universe_ds_btree_sharded_create(i64 64)
  ; each thread owns a span of 20000 keys. In disjoint mode each thread's base
  ; is si<<60 => distinct TOP bits => distinct shard (real parallelism). In
  ; overlap mode every thread hammers the SAME range [0,20000) => same shards
  ; => maximum lock contention (value invariant still holds: val == key*2+1).
  %span = add i64 0, 20000
  %isov = icmp ne i64 %overlap, 0
  br label %spawn

spawn:
  %si = phi i64 [ 0, %entry ], [ %si.n, %spawn ]
  %argi = mul i64 %si, 48
  %arg = getelementptr inbounds nuw i8, ptr %args, i64 %argi
  %ap.map = getelementptr inbounds nuw i8, ptr %arg, i64 0
  store ptr %m, ptr %ap.map, align 8
  %base.spread = shl i64 %si, 60
  %base = select i1 %isov, i64 0, i64 %base.spread
  %ap.base = getelementptr inbounds nuw i8, ptr %arg, i64 8
  store i64 %base, ptr %ap.base, align 8
  %ap.nops = getelementptr inbounds nuw i8, ptr %arg, i64 16
  store i64 %nops, ptr %ap.nops, align 8
  %seed = add i64 %seedbase, %si
  %seed2 = or i64 %seed, 1
  %ap.seed = getelementptr inbounds nuw i8, ptr %arg, i64 24
  store i64 %seed2, ptr %ap.seed, align 8
  %ap.span = getelementptr inbounds nuw i8, ptr %arg, i64 32
  store i64 %span, ptr %ap.span, align 8
  %ap.viol = getelementptr inbounds nuw i8, ptr %arg, i64 40
  store i64 0, ptr %ap.viol, align 8
  %tidi = mul i64 %si, 8
  %tid = getelementptr inbounds nuw i8, ptr %tids, i64 %tidi
  %cr = call i32 @pthread_create(ptr %tid, ptr null, ptr @stress_worker, ptr %arg)
  %si.n = add i64 %si, 1
  %smore = icmp ult i64 %si.n, %nthreads
  br i1 %smore, label %spawn, label %join

join:
  %ji = phi i64 [ 0, %spawn ], [ %ji.n, %join ]
  %tidji = mul i64 %ji, 8
  %tidj = getelementptr inbounds nuw i8, ptr %tids, i64 %tidji
  %tv = load ptr, ptr %tidj, align 8
  %jr = call i32 @pthread_join(ptr %tv, ptr null)
  %ji.n = add i64 %ji, 1
  %jmore = icmp ult i64 %ji.n, %nthreads
  br i1 %jmore, label %join, label %tally

tally:
  %ti = phi i64 [ 0, %join ], [ %ti.n, %tally ]
  %tviol = phi i64 [ 0, %join ], [ %tviol.n, %tally ]
  %targi = mul i64 %ti, 48
  %targ = getelementptr inbounds nuw i8, ptr %args, i64 %targi
  %tvp = getelementptr inbounds nuw i8, ptr %targ, i64 40
  %tvv = load i64, ptr %tvp, align 8
  %tviol.n = add i64 %tviol, %tvv
  %ti.n = add i64 %ti, 1
  %tmore = icmp ult i64 %ti.n, %nthreads
  br i1 %tmore, label %tally, label %done

done:
  call void @universe_ds_btree_sharded_destroy(ptr %m)
  call void @free(ptr %args)
  call void @free(ptr %tids)
  ret i64 %tviol
}

; disjoint pass: every key inserted exactly once, all must survive with value.
define i64 @disjoint_once(i64 %nthreads) {
entry:
  %argsz = mul i64 %nthreads, 48
  %args = call ptr @malloc(i64 %argsz)
  %tidsz = mul i64 %nthreads, 8
  %tids = call ptr @malloc(i64 %tidsz)
  %m = call ptr @universe_ds_btree_sharded_create(i64 32)
  %span = add i64 0, 30000
  br label %spawn

spawn:
  %si = phi i64 [ 0, %entry ], [ %si.n, %spawn ]
  %argi = mul i64 %si, 48
  %arg = getelementptr inbounds nuw i8, ptr %args, i64 %argi
  %ap.map = getelementptr inbounds nuw i8, ptr %arg, i64 0
  store ptr %m, ptr %ap.map, align 8
  %base = shl i64 %si, 60
  %ap.base = getelementptr inbounds nuw i8, ptr %arg, i64 8
  store i64 %base, ptr %ap.base, align 8
  %ap.span = getelementptr inbounds nuw i8, ptr %arg, i64 32
  store i64 %span, ptr %ap.span, align 8
  %ap.viol = getelementptr inbounds nuw i8, ptr %arg, i64 40
  store i64 0, ptr %ap.viol, align 8
  %tidi = mul i64 %si, 8
  %tid = getelementptr inbounds nuw i8, ptr %tids, i64 %tidi
  %cr = call i32 @pthread_create(ptr %tid, ptr null, ptr @disjoint_worker, ptr %arg)
  %si.n = add i64 %si, 1
  %smore = icmp ult i64 %si.n, %nthreads
  br i1 %smore, label %spawn, label %join

join:
  %ji = phi i64 [ 0, %spawn ], [ %ji.n, %join ]
  %tidji = mul i64 %ji, 8
  %tidj = getelementptr inbounds nuw i8, ptr %tids, i64 %tidji
  %tv = load ptr, ptr %tidj, align 8
  %jr = call i32 @pthread_join(ptr %tv, ptr null)
  %ji.n = add i64 %ji, 1
  %jmore = icmp ult i64 %ji.n, %nthreads
  br i1 %jmore, label %join, label %tally

tally:
  %ti = phi i64 [ 0, %join ], [ %ti.n, %tally ]
  %tviol = phi i64 [ 0, %join ], [ %tviol.n, %tally ]
  %targi = mul i64 %ti, 48
  %targ = getelementptr inbounds nuw i8, ptr %args, i64 %targi
  %tvp = getelementptr inbounds nuw i8, ptr %targ, i64 40
  %tvv = load i64, ptr %tvp, align 8
  %tviol.n = add i64 %tviol, %tvv
  %ti.n = add i64 %ti, 1
  %tmore = icmp ult i64 %ti.n, %nthreads
  br i1 %tmore, label %tally, label %done

done:
  ; total len must equal nthreads*span (no lost inserts, no dup keys)
  %len = call i64 @universe_ds_btree_sharded_len(ptr %m)
  %expect = mul i64 %nthreads, %span
  %lenbad = icmp ne i64 %len, %expect
  %lenbadz = zext i1 %lenbad to i64
  %total = add i64 %tviol, %lenbadz
  call void @universe_ds_btree_sharded_destroy(ptr %m)
  call void @free(ptr %args)
  call void @free(ptr %tids)
  ret i64 %total
}

define void @test_mt() {
entry:
  ; disjoint no-loss pass, looped >=10x at 4 threads
  br label %dloop

dloop:
  %di = phi i64 [ 0, %entry ], [ %di.n, %dloop ]
  %dviol = phi i64 [ 0, %entry ], [ %dviol.n, %dloop ]
  %dr = call i64 @disjoint_once(i64 4)
  %dviol.n = add i64 %dviol, %dr
  %di.n = add i64 %di, 1
  %dmore = icmp ult i64 %di.n, 10
  br i1 %dmore, label %dloop, label %dloop.done

dloop.done:
  call void @ut_check_eq(i64 %dviol, i64 0, ptr @msg.mt.lost)
  ; mixed put/get/delete stress: value-consistency invariant, looped >=10x
  br label %sloop

sloop:
  ; disjoint mixed (overlap=0) AND overlapping mixed (overlap=1), each iter,
  ; looped >=10x. Both must leave every present key with val==key*2+1.
  %ci = phi i64 [ 0, %dloop.done ], [ %ci.n, %sloop ]
  %cviol = phi i64 [ 0, %dloop.done ], [ %cviol.n, %sloop ]
  %seedb = mul i64 %ci, 7919
  %sr.dj = call i64 @stress_once(i64 4, i64 100000, i64 %seedb, i64 0)
  %seedb2 = add i64 %seedb, 104729
  %sr.ov = call i64 @stress_once(i64 4, i64 100000, i64 %seedb2, i64 1)
  %sr = add i64 %sr.dj, %sr.ov
  %cviol.n = add i64 %cviol, %sr
  %ci.n = add i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, 10
  br i1 %cmore, label %sloop, label %sloop.done

sloop.done:
  call void @ut_check_eq(i64 %cviol, i64 0, ptr @msg.mt.val)
  ret void
}

; ---------------------------------------------------------------------------
; bench: sharded put+get at T threads vs a single btree under one lock
; ---------------------------------------------------------------------------
define void @do_bench(i32 %argc, ptr %argv) {
entry:
  %sink = alloca i64, align 8
  %want = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %want, label %go, label %skip
skip:
  ret void
go:
  %T = add i64 0, 8
  %nops = add i64 0, 400000
  %totalops = mul i64 %T, %nops
  ; sharded run (warm-up rep discarded); each rep is self-contained
  br label %sh.rep.head
sh.rep.head:
  %shrep = phi i64 [ 0, %go ], [ %shrep.n, %sh.rep.cont ]
  %t0 = call double @ut_now_sec()
  %sv = call i64 @stress_once(i64 %T, i64 %nops, i64 12345, i64 0)
  %t1 = call double @ut_now_sec()
  store volatile i64 %sv, ptr %sink, align 8
  %sdt = fsub double %t1, %t0
  %shwarm = icmp eq i64 %shrep, 0
  br i1 %shwarm, label %sh.rep.cont, label %sh.rep.store
sh.rep.store:
  %shsi = sub i64 %shrep, 1
  %shsp = getelementptr inbounds double, ptr @sh.bench.samp, i64 %shsi
  store double %sdt, ptr %shsp, align 8
  br label %sh.rep.cont
sh.rep.cont:
  %shrep.n = add nuw i64 %shrep, 1
  %shmore = icmp ult i64 %shrep.n, 17
  br i1 %shmore, label %sh.rep.head, label %sh.report
sh.report:
  call void @ut_report_dist(ptr @sh.bench.samp, i64 16, i64 %totalops, ptr @lbl.sh.bench)
  br label %ol.rep.head
; single-lock baseline: one btree, T threads all hammering it
ol.rep.head:
  %olrep = phi i64 [ 0, %sh.report ], [ %olrep.n, %ol.rep.cont ]
  %t2 = call double @ut_now_sec()
  %bv = call i64 @onelock_once(i64 %T, i64 %nops)
  %t3 = call double @ut_now_sec()
  store volatile i64 %bv, ptr %sink, align 8
  %bdt = fsub double %t3, %t2
  %olwarm = icmp eq i64 %olrep, 0
  br i1 %olwarm, label %ol.rep.cont, label %ol.rep.store
ol.rep.store:
  %olsi = sub i64 %olrep, 1
  %olsp = getelementptr inbounds double, ptr @ol.bench.samp, i64 %olsi
  store double %bdt, ptr %olsp, align 8
  br label %ol.rep.cont
ol.rep.cont:
  %olrep.n = add nuw i64 %olrep, 1
  %olmore = icmp ult i64 %olrep.n, 17
  br i1 %olmore, label %ol.rep.head, label %ol.report
ol.report:
  call void @ut_report_dist(ptr @ol.bench.samp, i64 16, i64 %totalops, ptr @lbl.ol.bench)
  ret void
}

; one shared btree behind ONE spinlock — the anti-pattern the sharding beats.
@onelock.word = internal global i32 0, align 4

define ptr @onelock_worker(ptr %arg) {
entry:
  %seedslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %treep = getelementptr inbounds nuw i8, ptr %arg, i64 0
  %tree = load ptr, ptr %treep, align 8
  %basep = getelementptr inbounds nuw i8, ptr %arg, i64 8
  %base = load i64, ptr %basep, align 8
  %nopsp = getelementptr inbounds nuw i8, ptr %arg, i64 16
  %nops = load i64, ptr %nopsp, align 8
  %seedp = getelementptr inbounds nuw i8, ptr %arg, i64 24
  %seed = load i64, ptr %seedp, align 8
  store i64 %seed, ptr %seedslot, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %unlock ]
  %r = call i64 @ut_rand(ptr %seedslot)
  %slot = urem i64 %r, 20000
  %key = add i64 %base, %slot
  %val = mul i64 %key, 2
  %val1 = add i64 %val, 1
  br label %acq

acq:
  %cx = cmpxchg weak ptr @onelock.word, i32 0, i32 1 acquire monotonic
  %ok = extractvalue { i32, i1 } %cx, 1
  br i1 %ok, label %crit, label %spin

spin:
  %sv = load atomic i32, ptr @onelock.word monotonic, align 4
  %sf = icmp eq i32 %sv, 0
  br i1 %sf, label %acq, label %spin

crit:
  %isput = and i64 %r, 1
  %putc = icmp eq i64 %isput, 0
  br i1 %putc, label %doput, label %doget

doput:
  %prc = call i32 @universe_ds_btree_insert(ptr %tree, i64 %key, i64 %val1)
  br label %unlock

doget:
  %grc = call i32 @universe_ds_btree_find(ptr %tree, i64 %key, ptr %vslot)
  br label %unlock

unlock:
  store atomic i32 0, ptr @onelock.word release, align 4
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %nops
  br i1 %more, label %loop, label %fin

fin:
  ret ptr null
}

define i64 @onelock_once(i64 %nthreads, i64 %nops) {
entry:
  %argsz = mul i64 %nthreads, 48
  %args = call ptr @malloc(i64 %argsz)
  %tidsz = mul i64 %nthreads, 8
  %tids = call ptr @malloc(i64 %tidsz)
  %tree = call ptr @universe_ds_btree_create()
  store i32 0, ptr @onelock.word, align 4
  %span = add i64 0, 20000
  br label %spawn

spawn:
  %si = phi i64 [ 0, %entry ], [ %si.n, %spawn ]
  %argi = mul i64 %si, 48
  %arg = getelementptr inbounds nuw i8, ptr %args, i64 %argi
  %ap.tree = getelementptr inbounds nuw i8, ptr %arg, i64 0
  store ptr %tree, ptr %ap.tree, align 8
  %base = shl i64 %si, 60
  %ap.base = getelementptr inbounds nuw i8, ptr %arg, i64 8
  store i64 %base, ptr %ap.base, align 8
  %ap.nops = getelementptr inbounds nuw i8, ptr %arg, i64 16
  store i64 %nops, ptr %ap.nops, align 8
  %seed = or i64 %si, 1
  %ap.seed = getelementptr inbounds nuw i8, ptr %arg, i64 24
  store i64 %seed, ptr %ap.seed, align 8
  %tidi = mul i64 %si, 8
  %tid = getelementptr inbounds nuw i8, ptr %tids, i64 %tidi
  %cr = call i32 @pthread_create(ptr %tid, ptr null, ptr @onelock_worker, ptr %arg)
  %si.n = add i64 %si, 1
  %smore = icmp ult i64 %si.n, %nthreads
  br i1 %smore, label %spawn, label %join

join:
  %ji = phi i64 [ 0, %spawn ], [ %ji.n, %join ]
  %tidji = mul i64 %ji, 8
  %tidj = getelementptr inbounds nuw i8, ptr %tids, i64 %tidji
  %tv = load ptr, ptr %tidj, align 8
  %jr = call i32 @pthread_join(ptr %tv, ptr null)
  %ji.n = add i64 %ji, 1
  %jmore = icmp ult i64 %ji.n, %nthreads
  br i1 %jmore, label %join, label %done

done:
  call void @universe_ds_btree_destroy(ptr %tree)
  call void @free(ptr %args)
  call void @free(ptr %tids)
  ret i64 0
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single()
  call void @test_mt()
  call void @do_bench(i32 %argc, ptr %argv)
  %r = call i32 @ut_summary()
  ret i32 %r
}
