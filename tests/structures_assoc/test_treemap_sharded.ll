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

; Tests for the sharded (concurrent) ordered map.

declare ptr  @universe_ds_treemap_sharded_create(i64, i64)
declare void @universe_ds_treemap_sharded_destroy(ptr)
declare i32  @universe_ds_treemap_sharded_put(ptr, i64, i64)
declare i32  @universe_ds_treemap_sharded_get(ptr, i64, ptr)
declare i32  @universe_ds_treemap_sharded_contains(ptr, i64)
declare i32  @universe_ds_treemap_sharded_delete(ptr, i64)
declare i64  @universe_ds_treemap_sharded_size(ptr)
declare i64  @universe_ds_treemap_sharded_shards(ptr)
declare i32  @universe_ds_treemap_sharded_min(ptr, ptr, ptr)
declare i32  @universe_ds_treemap_sharded_max(ptr, ptr, ptr)
declare i32  @universe_ds_treemap_sharded_floor(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_sharded_ceiling(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_sharded_higher(ptr, i64, ptr, ptr)
declare i32  @universe_ds_treemap_sharded_lower(ptr, i64, ptr, ptr)
declare i64  @universe_ds_treemap_sharded_range(ptr, i64, i64, ptr, ptr, i64)
declare void @universe_ds_treemap_sharded_foreach(ptr, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64  @ut_rand(ptr)
declare double @ut_now_sec()
declare i1   @ut_want_bench(i32, ptr)
declare i32  @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@tms64.bench.samp = internal global [16 x double] zeroinitializer, align 8
@tms1.bench.samp  = internal global [16 x double] zeroinitializer, align 8
@lbl.tms64.bench = private unnamed_addr constant [26 x i8] c"treemap sharded64 put+get\00"
@lbl.tms1.bench  = private unnamed_addr constant [23 x i8] c"treemap 1-lock put+get\00"

declare ptr  @malloc(i64)
declare void @free(ptr)
declare i32  @printf(ptr, ...)
declare void @qsort(ptr, i64, i64, ptr)
declare i32  @pthread_create(ptr, ptr, ptr, ptr)
declare i32  @pthread_join(i64, ptr)

@g.map  = internal global ptr null, align 8
@g.tids = internal global [8 x i64] zeroinitializer, align 8
@g.err  = internal global i64 0, align 8
@g.bacc = internal global i64 0, align 8

@msg.sh.n   = private constant [15 x i8] c"shards is pow2\00"
@msg.si.get = private constant [17 x i8] c"single get viols\00"
@msg.si.sz  = private constant [16 x i8] c"single size ==M\00"
@msg.si.mn  = private constant [17 x i8] c"single min==ref0\00"
@msg.si.mx  = private constant [17 x i8] c"single max==refL\00"
@msg.si.ord = private constant [18 x i8] c"single iter order\00"
@msg.si.cnt = private constant [16 x i8] c"single iter cnt\00"
@msg.si.rc  = private constant [17 x i8] c"xshard range cnt\00"
@msg.si.rv  = private constant [17 x i8] c"xshard range val\00"
@msg.si.emp = private constant [17 x i8] c"xshard range emp\00"
@msg.si.fl  = private constant [17 x i8] c"single floor  ok\00"
@msg.si.ce  = private constant [17 x i8] c"single ceil   ok\00"

@msg.st.err = private constant [16 x i8] c"stress op viols\00"
@msg.st.sz  = private constant [15 x i8] c"stress size ok\00"
@msg.st.rng = private constant [17 x i8] c"stress range==sz\00"
@msg.st.ord = private constant [16 x i8] c"stress iter ord\00"
@msg.st.cnt = private constant [17 x i8] c"stress iter cnt\00\00"
@msg.st.vf  = private constant [17 x i8] c"stress verify st\00"


; splitmix64(tid*10000000 + j) -> spread key across the whole 64-bit space
define internal i64 @tkey(i64 %t, i64 %j) {
entry:
  %a = mul i64 %t, 10000000
  %x = add i64 %a, %j
  %z0 = add i64 %x, -7046029254386353131
  %s1 = lshr i64 %z0, 30
  %x1 = xor i64 %z0, %s1
  %m1 = mul i64 %x1, -4658895280553007687
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %m1, %s2
  %m2 = mul i64 %x2, -7723592293110705685
  %s3 = lshr i64 %m2, 31
  %z = xor i64 %m2, %s3
  ret i64 %z
}

; common (overlapping) key set, disjoint from every tkey() input range
define internal i64 @ckey(i64 %c) {
entry:
  %x = add i64 %c, 2863267840
  %z0 = add i64 %x, -7046029254386353131
  %s1 = lshr i64 %z0, 30
  %x1 = xor i64 %z0, %s1
  %m1 = mul i64 %x1, -4658895280553007687
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %m1, %s2
  %m2 = mul i64 %x2, -7723592293110705685
  %s3 = lshr i64 %m2, 31
  %z = xor i64 %m2, %s3
  ret i64 %z
}

; second spread family for the single-thread test
define internal i64 @tkey2(i64 %i) {
entry:
  %x = add i64 %i, 78187493520
  %z0 = add i64 %x, -7046029254386353131
  %s1 = lshr i64 %z0, 30
  %x1 = xor i64 %z0, %s1
  %m1 = mul i64 %x1, -4658895280553007687
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %m1, %s2
  %m2 = mul i64 %x2, -7723592293110705685
  %s3 = lshr i64 %m2, 31
  %z = xor i64 %m2, %s3
  ret i64 %z
}

; unsigned i64 comparator for qsort
define internal i32 @cmp_u64(ptr %a, ptr %b) {
entry:
  %x = load i64, ptr %a, align 8
  %y = load i64, ptr %b, align 8
  %lt = icmp ult i64 %x, %y
  %gt = icmp ugt i64 %x, %y
  %r0 = select i1 %gt, i32 1, i32 0
  %r = select i1 %lt, i32 -1, i32 %r0
  ret i32 %r
}

; foreach monotone-ascending callback; ctx = {prev,first,vio,count}
define internal void @cb_ordered(ptr %ctx, i64 %k, i64 %v) {
entry:
  %prev.p = getelementptr inbounds nuw i8, ptr %ctx, i64 0
  %first.p = getelementptr inbounds nuw i8, ptr %ctx, i64 8
  %vio.p = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  %cnt.p = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %first = load i64, ptr %first.p, align 8
  %isfirst = icmp eq i64 %first, 0
  br i1 %isfirst, label %skip, label %chk

chk:
  %prev = load i64, ptr %prev.p, align 8
  %le = icmp ule i64 %k, %prev
  br i1 %le, label %bump, label %skip

bump:
  %vio = load i64, ptr %vio.p, align 8
  %vio.n = add i64 %vio, 1
  store i64 %vio.n, ptr %vio.p, align 8
  br label %skip

skip:
  store i64 %k, ptr %prev.p, align 8
  store i64 1, ptr %first.p, align 8
  %cnt = load i64, ptr %cnt.p, align 8
  %cnt.n = add i64 %cnt, 1
  store i64 %cnt.n, ptr %cnt.p, align 8
  ret void
}

; ref lower_bound (first idx, arr[idx] >= k) over sorted unsigned array
define internal i64 @ref_lb(ptr %arr, i64 %n, i64 %k) {
entry:
  br label %loop
loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %n, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done
step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %kp = getelementptr inbounds nuw i64, ptr %arr, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %less = icmp ult i64 %kv, %k
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %less, i64 %mid1, i64 %lo
  %hi.n = select i1 %less, i64 %hi, i64 %mid
  br label %loop
done:
  ret i64 %lo
}

; ref upper_bound (first idx, arr[idx] > k)
define internal i64 @ref_ub(ptr %arr, i64 %n, i64 %k) {
entry:
  br label %loop
loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %n, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done
step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %kp = getelementptr inbounds nuw i64, ptr %arr, i64 %mid
  %kv = load i64, ptr %kp, align 8
  %le = icmp ule i64 %kv, %k
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %le, i64 %mid1, i64 %lo
  %hi.n = select i1 %le, i64 %hi, i64 %mid
  br label %loop
done:
  ret i64 %lo
}

; ===========================================================================
; Test A: single-thread mirror incl. cross-shard range + ordered iterate
; ===========================================================================
define internal void @t_sh_single() {
entry:
  %m = call ptr @universe_ds_treemap_sharded_create(i64 16, i64 64)
  %ob = alloca i64, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  ; shards should be power of two
  %ns = call i64 @universe_ds_treemap_sharded_shards(ptr %m)
  %nsm1 = add i64 %ns, -1
  %andv = and i64 %ns, %nsm1
  %pow2 = icmp eq i64 %andv, 0
  call void @ut_check(i1 %pow2, ptr @msg.sh.n)

  %M = add i64 0, 20000
  %refk = call ptr @malloc(i64 160000)   ; 20000*8
  %outk = call ptr @malloc(i64 160000)
  %outv = call ptr @malloc(i64 160000)
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %key = call i64 @tkey2(i64 %i)
  %val = xor i64 %key, 21845          ; 0x5555
  %p = call i32 @universe_ds_treemap_sharded_put(ptr %m, i64 %key, i64 %val)
  %rp = getelementptr inbounds nuw i64, ptr %refk, i64 %i
  store i64 %key, ptr %rp, align 8
  %i.n = add nuw i64 %i, 1
  %fdone = icmp uge i64 %i.n, 20000
  br i1 %fdone, label %sortref, label %fill

sortref:
  call void @qsort(ptr %refk, i64 20000, i64 8, ptr @cmp_u64)
  ; size == M (tkey2 is a bijection -> distinct)
  %sz = call i64 @universe_ds_treemap_sharded_size(ptr %m)
  call void @ut_check_eq(i64 %sz, i64 20000, ptr @msg.si.sz)
  br label %getv

getv:
  %j = phi i64 [ 0, %sortref ], [ %j.n, %getv ]
  %gvio = phi i64 [ 0, %sortref ], [ %gvio.n, %getv ]
  %rkp = getelementptr inbounds nuw i64, ptr %refk, i64 %j
  %rk = load i64, ptr %rkp, align 8
  %g = call i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %rk, ptr %ob)
  %got = load i64, ptr %ob, align 8
  %exp = xor i64 %rk, 21845
  %rcbad = icmp ne i32 %g, 0
  %vbad = icmp ne i64 %got, %exp
  %bad = or i1 %rcbad, %vbad
  %badi = zext i1 %bad to i64
  %gvio.n = add i64 %gvio, %badi
  %j.n = add nuw i64 %j, 1
  %gdone = icmp uge i64 %j.n, 20000
  br i1 %gdone, label %gfin, label %getv

gfin:
  call void @ut_check_eq(i64 %gvio, i64 0, ptr @msg.si.get)

  ; min == refk[0], max == refk[M-1]
  %mnr = call i32 @universe_ds_treemap_sharded_min(ptr %m, ptr %ok, ptr %ov)
  %mnk = load i64, ptr %ok, align 8
  %r0 = load i64, ptr %refk, align 8
  %mnok = icmp eq i64 %mnk, %r0
  call void @ut_check(i1 %mnok, ptr @msg.si.mn)
  %mxr = call i32 @universe_ds_treemap_sharded_max(ptr %m, ptr %ok, ptr %ov)
  %mxk = load i64, ptr %ok, align 8
  %rlp = getelementptr inbounds nuw i64, ptr %refk, i64 19999
  %rl = load i64, ptr %rlp, align 8
  %mxok = icmp eq i64 %mxk, %rl
  call void @ut_check(i1 %mxok, ptr @msg.si.mx)

  ; ordered iterate
  %ctx = alloca [4 x i64], align 8
  store i64 0, ptr %ctx, align 8
  %f.p = getelementptr inbounds nuw i8, ptr %ctx, i64 8
  store i64 0, ptr %f.p, align 8
  %v.p = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  store i64 0, ptr %v.p, align 8
  %c.p = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  store i64 0, ptr %c.p, align 8
  call void @universe_ds_treemap_sharded_foreach(ptr %m, ptr @cb_ordered, ptr %ctx)
  %ivio = load i64, ptr %v.p, align 8
  call void @ut_check_eq(i64 %ivio, i64 0, ptr @msg.si.ord)
  %icnt = load i64, ptr %c.p, align 8
  call void @ut_check_eq(i64 %icnt, i64 20000, ptr @msg.si.cnt)

  ; cross-shard range: [refk[5000], refk[15000]] inclusive -> indices 5000..15000
  %lop = getelementptr inbounds nuw i64, ptr %refk, i64 5000
  %lo = load i64, ptr %lop, align 8
  %hip = getelementptr inbounds nuw i64, ptr %refk, i64 15000
  %hi = load i64, ptr %hip, align 8
  %rcount = call i64 @universe_ds_treemap_sharded_range(ptr %m, i64 %lo, i64 %hi, ptr %outk, ptr %outv, i64 20000)
  ; reference count = ub(hi) - lb(lo)
  %rlb = call i64 @ref_lb(ptr %refk, i64 20000, i64 %lo)
  %rub = call i64 @ref_ub(ptr %refk, i64 20000, i64 %hi)
  %refcnt = sub i64 %rub, %rlb
  call void @ut_check_eq(i64 %rcount, i64 %refcnt, ptr @msg.si.rc)
  ; verify out keys == refk[rlb .. rub) and ascending, val == key^0x5555
  %rvio = call i64 @verify_slice(ptr %outk, ptr %outv, i64 %rcount, ptr %refk, i64 %rlb)
  call void @ut_check_eq(i64 %rvio, i64 0, ptr @msg.si.rv)

  ; empty range lo>hi
  %er = call i64 @universe_ds_treemap_sharded_range(ptr %m, i64 %hi, i64 %lo, ptr %outk, ptr %outv, i64 20000)
  ; only guaranteed empty if hi>lo which it is (refk sorted, index15000>5000)
  %erok = icmp eq i64 %er, 0
  call void @ut_check(i1 %erok, ptr @msg.si.emp)

  ; floor/ceiling: floor(refk[100]+? ) — use exact ref key so floor==itself
  %fkp = getelementptr inbounds nuw i64, ptr %refk, i64 100
  %fk = load i64, ptr %fkp, align 8
  %fr = call i32 @universe_ds_treemap_sharded_floor(ptr %m, i64 %fk, ptr %ok, ptr %ov)
  %frk = load i64, ptr %ok, align 8
  %frok0 = icmp eq i32 %fr, 0
  %frok1 = icmp eq i64 %frk, %fk
  %frok = and i1 %frok0, %frok1
  call void @ut_check(i1 %frok, ptr @msg.si.fl)
  ; ceiling(refk[101]) == refk[101]
  %ckp = getelementptr inbounds nuw i64, ptr %refk, i64 101
  %ck = load i64, ptr %ckp, align 8
  %cr = call i32 @universe_ds_treemap_sharded_ceiling(ptr %m, i64 %ck, ptr %ok, ptr %ov)
  %crk = load i64, ptr %ok, align 8
  %crok0 = icmp eq i32 %cr, 0
  %crok1 = icmp eq i64 %crk, %ck
  %crok = and i1 %crok0, %crok1
  call void @ut_check(i1 %crok, ptr @msg.si.ce)

  call void @free(ptr %refk)
  call void @free(ptr %outk)
  call void @free(ptr %outv)
  call void @universe_ds_treemap_sharded_destroy(ptr %m)
  ret void
}

; verify out[i]==refk[start+i], ascending, val[i]==key^0x5555; return #viols
define internal i64 @verify_slice(ptr %outk, ptr %outv, i64 %cnt, ptr %refk, i64 %start) {
entry:
  %z = icmp eq i64 %cnt, 0
  br i1 %z, label %zret, label %loop
zret:
  ret i64 0
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %vio = phi i64 [ 0, %entry ], [ %vio.n, %loop ]
  %okp = getelementptr inbounds nuw i64, ptr %outk, i64 %i
  %ok = load i64, ptr %okp, align 8
  %ridx = add i64 %start, %i
  %rkp = getelementptr inbounds nuw i64, ptr %refk, i64 %ridx
  %rk = load i64, ptr %rkp, align 8
  %kbad = icmp ne i64 %ok, %rk
  %ovp = getelementptr inbounds nuw i64, ptr %outv, i64 %i
  %ov = load i64, ptr %ovp, align 8
  %expv = xor i64 %ok, 21845
  %vbad = icmp ne i64 %ov, %expv
  %bad = or i1 %kbad, %vbad
  %badi = zext i1 %bad to i64
  %vio.n = add i64 %vio, %badi
  %i.n = add nuw i64 %i, 1
  %done = icmp uge i64 %i.n, %cnt
  br i1 %done, label %fin, label %loop
fin:
  ret i64 %vio.n
}

; ===========================================================================
; Test B: pthread stress — 4 threads, disjoint keys + shared common keys
; ===========================================================================
; worker: tid = ptrtoint(arg). K=50000 private keys via tkey(tid,j).
define internal ptr @stress_worker(ptr %arg) {
entry:
  %t = ptrtoint ptr %arg to i64
  %m = load ptr, ptr @g.map, align 8
  %ob = alloca i64, align 8
  br label %p1

; phase 1: put private keys, val = key
p1:
  %j = phi i64 [ 0, %entry ], [ %j.n, %p1 ]
  %k1 = call i64 @tkey(i64 %t, i64 %j)
  %r1 = call i32 @universe_ds_treemap_sharded_put(ptr %m, i64 %k1, i64 %k1)
  %b1 = icmp ne i32 %r1, 0
  %e1 = zext i1 %b1 to i64
  %o1 = atomicrmw add ptr @g.err, i64 %e1 monotonic, align 8
  %j.n = add nuw i64 %j, 1
  %m1 = icmp ult i64 %j.n, 50000
  br i1 %m1, label %p1, label %p2

; phase 2: get-verify own keys (stable; disjoint)
p2:
  %j2 = phi i64 [ 0, %p1 ], [ %j2.n, %p2 ]
  %k2 = call i64 @tkey(i64 %t, i64 %j2)
  %r2 = call i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %k2, ptr %ob)
  %gv = load i64, ptr %ob, align 8
  %rb = icmp ne i32 %r2, 0
  %vb = icmp ne i64 %gv, %k2
  %b2 = or i1 %rb, %vb
  %e2 = zext i1 %b2 to i64
  %o2 = atomicrmw add ptr @g.err, i64 %e2 monotonic, align 8
  %j2.n = add nuw i64 %j2, 1
  %m2 = icmp ult i64 %j2.n, 50000
  br i1 %m2, label %p2, label %p3

; phase 3: delete odd-index private keys
p3:
  %j3 = phi i64 [ 0, %p2 ], [ %j3.n, %p3.cont ]
  %odd = and i64 %j3, 1
  %isodd = icmp eq i64 %odd, 1
  br i1 %isodd, label %p3.del, label %p3.cont

p3.del:
  %k3 = call i64 @tkey(i64 %t, i64 %j3)
  %r3 = call i32 @universe_ds_treemap_sharded_delete(ptr %m, i64 %k3)
  %b3 = icmp ne i32 %r3, 0
  %e3 = zext i1 %b3 to i64
  %o3 = atomicrmw add ptr @g.err, i64 %e3 monotonic, align 8
  br label %p3.cont

p3.cont:
  %j3.n = add nuw i64 %j3, 1
  %m3 = icmp ult i64 %j3.n, 50000
  br i1 %m3, label %p3, label %p4

; phase 4: put shared overlapping keys (all threads, same key/value 777)
p4:
  %j4 = phi i64 [ 0, %p3.cont ], [ %j4.n, %p4 ]
  %k4 = call i64 @ckey(i64 %j4)
  %r4 = call i32 @universe_ds_treemap_sharded_put(ptr %m, i64 %k4, i64 777)
  %b4 = icmp ne i32 %r4, 0
  %e4 = zext i1 %b4 to i64
  %o4 = atomicrmw add ptr @g.err, i64 %e4 monotonic, align 8
  %j4.n = add nuw i64 %j4, 1
  %m4 = icmp ult i64 %j4.n, 256
  br i1 %m4, label %p4, label %done

done:
  ret ptr null
}

define internal void @spawn_join() {
entry:
  br label %spawn
spawn:
  %i = phi i64 [ 0, %entry ], [ %i.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %i
  %arg = inttoptr i64 %i to ptr
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr @stress_worker, ptr %arg)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 4
  br i1 %more, label %spawn, label %join
join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 4
  br i1 %jmore, label %join, label %done
done:
  ret void
}

; single-threaded post-join verification: returns violation count
define internal i64 @verify_stress(ptr %m) {
entry:
  %ob = alloca i64, align 8
  br label %t.loop
t.loop:
  %t = phi i64 [ 0, %entry ], [ %t.n, %t.done ]
  %viol.t = phi i64 [ 0, %entry ], [ %viol.td, %t.done ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %t.loop ], [ %j.n, %j.cont ]
  %viol.j = phi i64 [ %viol.t, %t.loop ], [ %viol.jn, %j.cont ]
  %k = call i64 @tkey(i64 %t, i64 %j)
  %g = call i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %k, ptr %ob)
  %gv = load i64, ptr %ob, align 8
  ; even j -> present (0) & val==k ; odd j -> absent (5)
  %odd = and i64 %j, 1
  %isodd = icmp ne i64 %odd, 0
  %pres.ok0 = icmp eq i32 %g, 0
  %pres.ok1 = icmp eq i64 %gv, %k
  %pres.ok = and i1 %pres.ok0, %pres.ok1
  %abs.ok = icmp eq i32 %g, 5
  %want.ok = select i1 %isodd, i1 %abs.ok, i1 %pres.ok
  %bad = xor i1 %want.ok, true
  %badi = zext i1 %bad to i64
  %viol.jn = add i64 %viol.j, %badi
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 50000
  br i1 %jmore, label %j.cont, label %t.done
j.cont:
  br label %j.loop
t.done:
  %viol.td = phi i64 [ %viol.jn, %j.loop ]
  %t.n = add nuw i64 %t, 1
  %tmore = icmp ult i64 %t.n, 4
  br i1 %tmore, label %t.loop, label %ckeys
ckeys:
  br label %c.loop
c.loop:
  %c = phi i64 [ 0, %ckeys ], [ %c.n, %c.loop ]
  %viol.c = phi i64 [ %viol.td, %ckeys ], [ %viol.cn, %c.loop ]
  %ck = call i64 @ckey(i64 %c)
  %cg = call i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %ck, ptr %ob)
  %cgv = load i64, ptr %ob, align 8
  %cok0 = icmp eq i32 %cg, 0
  %cok1 = icmp eq i64 %cgv, 777
  %cok = and i1 %cok0, %cok1
  %cbad = xor i1 %cok, true
  %cbadi = zext i1 %cbad to i64
  %viol.cn = add i64 %viol.c, %cbadi
  %c.n = add nuw i64 %c, 1
  %cmore = icmp ult i64 %c.n, 256
  br i1 %cmore, label %c.loop, label %fin
fin:
  ret i64 %viol.cn
}

define internal void @t_sh_stress() {
entry:
  br label %run
run:
  %run.i = phi i64 [ 0, %entry ], [ %run.n, %run ]
  %m = call ptr @universe_ds_treemap_sharded_create(i64 16, i64 64)
  store ptr %m, ptr @g.map, align 8
  store i64 0, ptr @g.err, align 8
  call void @spawn_join()
  ; op-time errors accumulated in g.err
  %err = load i64, ptr @g.err, align 8
  call void @ut_check_eq(i64 %err, i64 0, ptr @msg.st.err)
  ; size == 4*25000 evens + 256 common
  %sz = call i64 @universe_ds_treemap_sharded_size(ptr %m)
  call void @ut_check_eq(i64 %sz, i64 100256, ptr @msg.st.sz)
  ; post-join single-thread verify
  %vf = call i64 @verify_stress(ptr %m)
  call void @ut_check_eq(i64 %vf, i64 0, ptr @msg.st.vf)
  ; cross-shard full range count == size
  %rc = call i64 @universe_ds_treemap_sharded_range(ptr %m, i64 0, i64 -1, ptr null, ptr null, i64 0)
  call void @ut_check_eq(i64 %rc, i64 %sz, ptr @msg.st.rng)
  ; ordered foreach count == size, monotone
  %ctx = alloca [4 x i64], align 8
  store i64 0, ptr %ctx, align 8
  %f.p = getelementptr inbounds nuw i8, ptr %ctx, i64 8
  store i64 0, ptr %f.p, align 8
  %v.p = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  store i64 0, ptr %v.p, align 8
  %c.p = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  store i64 0, ptr %c.p, align 8
  call void @universe_ds_treemap_sharded_foreach(ptr %m, ptr @cb_ordered, ptr %ctx)
  %ivio = load i64, ptr %v.p, align 8
  call void @ut_check_eq(i64 %ivio, i64 0, ptr @msg.st.ord)
  %icnt = load i64, ptr %c.p, align 8
  call void @ut_check_eq(i64 %icnt, i64 %sz, ptr @msg.st.cnt)
  call void @universe_ds_treemap_sharded_destroy(ptr %m)
  %run.n = add nuw i64 %run.i, 1
  %rmore = icmp ult i64 %run.n, 12
  br i1 %rmore, label %run, label %fin
fin:
  ret void
}

; ===========================================================================
; Bench: sharded(64) vs single-lock (sharded 1) under 4-thread put/get load
; ===========================================================================
define internal ptr @bench_worker(ptr %arg) {
entry:
  %t = ptrtoint ptr %arg to i64
  %m = load ptr, ptr @g.map, align 8
  %ob = alloca i64, align 8
  br label %loop
loop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %k = call i64 @tkey(i64 %t, i64 %j)
  %p = call i32 @universe_ds_treemap_sharded_put(ptr %m, i64 %k, i64 %k)
  %g = call i32 @universe_ds_treemap_sharded_get(ptr %m, i64 %k, ptr %ob)
  %v = load i64, ptr %ob, align 8
  %acc.n = add i64 %acc, %v
  %j.n = add nuw i64 %j, 1
  %more = icmp ult i64 %j.n, 100000
  br i1 %more, label %loop, label %done
done:
  %o = atomicrmw add ptr @g.bacc, i64 %acc monotonic, align 8
  ret ptr null
}

define internal void @bspawn_join() {
entry:
  br label %spawn
spawn:
  %i = phi i64 [ 0, %entry ], [ %i.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %i
  %arg = inttoptr i64 %i to ptr
  %r = call i32 @pthread_create(ptr %slot, ptr null, ptr @bench_worker, ptr %arg)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 4
  br i1 %more, label %spawn, label %join
join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr @g.tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 4
  br i1 %jmore, label %join, label %done
done:
  ret void
}

define internal void @run_bench() {
entry:
  ; sharded 64 (destructive: re-create map each rep; warm-up rep discarded)
  br label %s64.rep.head
s64.rep.head:
  %s64rep = phi i64 [ 0, %entry ], [ %s64rep.n, %s64.rep.cont ]
  %m64 = call ptr @universe_ds_treemap_sharded_create(i64 64, i64 256)
  store ptr %m64, ptr @g.map, align 8
  %t0 = call double @ut_now_sec()
  call void @bspawn_join()
  %t1 = call double @ut_now_sec()
  call void @universe_ds_treemap_sharded_destroy(ptr %m64)
  %dt64 = fsub double %t1, %t0
  %s64.warm = icmp eq i64 %s64rep, 0
  br i1 %s64.warm, label %s64.rep.cont, label %s64.rep.store
s64.rep.store:
  %s64.si = sub i64 %s64rep, 1
  %s64.sp = getelementptr inbounds double, ptr @tms64.bench.samp, i64 %s64.si
  store double %dt64, ptr %s64.sp, align 8
  br label %s64.rep.cont
s64.rep.cont:
  %s64rep.n = add nuw i64 %s64rep, 1
  %s64.more = icmp ult i64 %s64rep.n, 17
  br i1 %s64.more, label %s64.rep.head, label %s64.report
s64.report:
  call void @ut_report_dist(ptr @tms64.bench.samp, i64 16, i64 800000, ptr @lbl.tms64.bench)
  br label %s1.rep.head
; single lock (1 shard)
s1.rep.head:
  %s1rep = phi i64 [ 0, %s64.report ], [ %s1rep.n, %s1.rep.cont ]
  %m1 = call ptr @universe_ds_treemap_sharded_create(i64 1, i64 256)
  store ptr %m1, ptr @g.map, align 8
  %t2 = call double @ut_now_sec()
  call void @bspawn_join()
  %t3 = call double @ut_now_sec()
  call void @universe_ds_treemap_sharded_destroy(ptr %m1)
  %dt1 = fsub double %t3, %t2
  %s1.warm = icmp eq i64 %s1rep, 0
  br i1 %s1.warm, label %s1.rep.cont, label %s1.rep.store
s1.rep.store:
  %s1.si = sub i64 %s1rep, 1
  %s1.sp = getelementptr inbounds double, ptr @tms1.bench.samp, i64 %s1.si
  store double %dt1, ptr %s1.sp, align 8
  br label %s1.rep.cont
s1.rep.cont:
  %s1rep.n = add nuw i64 %s1rep, 1
  %s1.more = icmp ult i64 %s1rep.n, 17
  br i1 %s1.more, label %s1.rep.head, label %s1.report
s1.report:
  call void @ut_report_dist(ptr @tms1.bench.samp, i64 16, i64 800000, ptr @lbl.tms1.bench)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @t_sh_single()
  call void @t_sh_stress()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary
do.bench:
  call void @run_bench()
  br label %summary
summary:
  %r = call i32 @ut_summary()
  ret i32 %r
}
