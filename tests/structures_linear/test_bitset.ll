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

; Tests for universe_ds_bitset: single-bit ops, error codes, tail masking on
; set_all/complement for non-word-multiple sizes, popcount + find scan vs a
; shadow model, word-parallel set algebra vs brute force. --bench sweeps 1M
; bits.

declare ptr @universe_ds_bitset_create(i64)
declare void @universe_ds_bitset_destroy(ptr)
declare i32 @universe_ds_bitset_set(ptr, i64)
declare i32 @universe_ds_bitset_clear(ptr, i64)
declare i32 @universe_ds_bitset_toggle(ptr, i64)
declare i32 @universe_ds_bitset_test(ptr, i64)
declare i32 @universe_ds_bitset_set_all(ptr)
declare i32 @universe_ds_bitset_clear_all(ptr)
declare i64 @universe_ds_bitset_popcount(ptr)
declare i64 @universe_ds_bitset_nbits(ptr)
declare i64 @universe_ds_bitset_find_first_set(ptr)
declare i64 @universe_ds_bitset_find_next_set(ptr, i64)
declare i32 @universe_ds_bitset_union(ptr, ptr, ptr)
declare i32 @universe_ds_bitset_intersection(ptr, ptr, ptr)
declare i32 @universe_ds_bitset_difference(ptr, ptr, ptr)
declare i32 @universe_ds_bitset_complement(ptr, ptr)

declare i32 @printf(ptr, ...)
declare i64 @ut_rand(ptr)
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.bits    = private unnamed_addr constant [21 x i8] c"set/test/clear/tggl \00"
@m.errs    = private unnamed_addr constant [19 x i8] c"error codes 1/7/-1\00"
@m.setall  = private unnamed_addr constant [24 x i8] c"set_all popcount==nbits\00"
@m.compl   = private unnamed_addr constant [21 x i8] c"complement tail-safe\00"
@m.pcrand  = private unnamed_addr constant [21 x i8] c"popcount vs shadow  \00"
@m.scan    = private unnamed_addr constant [22 x i8] c"find scan enumerates \00"
@m.union   = private unnamed_addr constant [14 x i8] c"union vs ref \00"
@m.inter   = private unnamed_addr constant [21 x i8] c"intersection vs ref \00"
@m.diff    = private unnamed_addr constant [19 x i8] c"difference vs ref \00"
@m.inplace = private unnamed_addr constant [19 x i8] c"in-place union a=a\00"
@lbl.bitset = private unnamed_addr constant [40 x i8] c"bitset popcount+scan (200 calls/rep) ns\00"
@bitset.samp = internal global [16 x double] zeroinitializer, align 8
@bitset.sink = internal global i64 0, align 8

; --- single-bit ops -------------------------------------------------------
define internal void @test_bits() {
entry:
  %bs = call ptr @universe_ds_bitset_create(i64 200)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 0)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 1)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 63)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 64)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 65)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 199)
  %t0 = call i32 @universe_ds_bitset_test(ptr %bs, i64 0)
  %t63 = call i32 @universe_ds_bitset_test(ptr %bs, i64 63)
  %t64 = call i32 @universe_ds_bitset_test(ptr %bs, i64 64)
  %t199 = call i32 @universe_ds_bitset_test(ptr %bs, i64 199)
  %t100 = call i32 @universe_ds_bitset_test(ptr %bs, i64 100)
  %pc = call i64 @universe_ds_bitset_popcount(ptr %bs)
  %ff = call i64 @universe_ds_bitset_find_first_set(ptr %bs)
  ; expect t0=t63=t64=t199=1, t100=0, popcount=6, first=0
  %c0 = icmp eq i32 %t0, 1
  %c1 = icmp eq i32 %t63, 1
  %c2 = icmp eq i32 %t64, 1
  %c3 = icmp eq i32 %t199, 1
  %c4 = icmp eq i32 %t100, 0
  %c5 = icmp eq i64 %pc, 6
  %c6 = icmp eq i64 %ff, 0
  %a0 = and i1 %c0, %c1
  %a1 = and i1 %a0, %c2
  %a2 = and i1 %a1, %c3
  %a3 = and i1 %a2, %c4
  %a4 = and i1 %a3, %c5
  %a5 = and i1 %a4, %c6
  ; clear + toggle round-trip on bit 63
  call i32 @universe_ds_bitset_clear(ptr %bs, i64 63)
  %tc = call i32 @universe_ds_bitset_test(ptr %bs, i64 63)
  %pc2 = call i64 @universe_ds_bitset_popcount(ptr %bs)
  call i32 @universe_ds_bitset_toggle(ptr %bs, i64 63)
  %tt = call i32 @universe_ds_bitset_test(ptr %bs, i64 63)
  %pc3 = call i64 @universe_ds_bitset_popcount(ptr %bs)
  %c7 = icmp eq i32 %tc, 0
  %c8 = icmp eq i64 %pc2, 5
  %c9 = icmp eq i32 %tt, 1
  %c10 = icmp eq i64 %pc3, 6
  %b0 = and i1 %a5, %c7
  %b1 = and i1 %b0, %c8
  %b2 = and i1 %b1, %c9
  %b3 = and i1 %b2, %c10
  call void @ut_check(i1 %b3, ptr @m.bits)
  ; clear_all drops popcount to 0
  call i32 @universe_ds_bitset_clear_all(ptr %bs)
  %pc4 = call i64 @universe_ds_bitset_popcount(ptr %bs)
  %z = icmp eq i64 %pc4, 0
  call void @ut_check(i1 %z, ptr @m.bits)
  call void @universe_ds_bitset_destroy(ptr %bs)
  ret void
}

; --- error codes ----------------------------------------------------------
define internal void @test_errors() {
entry:
  %bs = call ptr @universe_ds_bitset_create(i64 100)
  %e1 = call i32 @universe_ds_bitset_set(ptr null, i64 0)        ; NULL -> 1
  %e2 = call i32 @universe_ds_bitset_set(ptr %bs, i64 100)       ; OOB -> 7
  %e3 = call i32 @universe_ds_bitset_set(ptr %bs, i64 999)       ; OOB -> 7
  %e4 = call i32 @universe_ds_bitset_test(ptr %bs, i64 100)      ; OOB -> -1
  %e5 = call i32 @universe_ds_bitset_test(ptr null, i64 0)       ; NULL -> -1
  %ea = call ptr @universe_ds_bitset_create(i64 100)
  %eb = call ptr @universe_ds_bitset_create(i64 64)
  %e6 = call i32 @universe_ds_bitset_union(ptr %bs, ptr %ea, ptr %eb) ; size mismatch -> 8
  %e7 = call i32 @universe_ds_bitset_union(ptr null, ptr %ea, ptr %eb) ; NULL -> 1
  %o1 = icmp eq i32 %e1, 1
  %o2 = icmp eq i32 %e2, 7
  %o3 = icmp eq i32 %e3, 7
  %o4 = icmp eq i32 %e4, -1
  %o5 = icmp eq i32 %e5, -1
  %o6 = icmp eq i32 %e6, 8
  %o7 = icmp eq i32 %e7, 1
  %p0 = and i1 %o1, %o2
  %p1 = and i1 %p0, %o3
  %p2 = and i1 %p1, %o4
  %p3 = and i1 %p2, %o5
  %p4 = and i1 %p3, %o6
  %p5 = and i1 %p4, %o7
  call void @ut_check(i1 %p5, ptr @m.errs)
  call void @universe_ds_bitset_destroy(ptr %bs)
  call void @universe_ds_bitset_destroy(ptr %ea)
  call void @universe_ds_bitset_destroy(ptr %eb)
  call void @universe_ds_bitset_destroy(ptr null)
  ret void
}

; --- set_all tail masking for non-word-multiple sizes ---------------------
define internal void @test_setall() {
entry:
  %sizes = alloca [7 x i64], align 8
  store i64 1, ptr %sizes, align 8
  %s1 = getelementptr inbounds nuw i64, ptr %sizes, i64 1
  store i64 63, ptr %s1, align 8
  %s2 = getelementptr inbounds nuw i64, ptr %sizes, i64 2
  store i64 64, ptr %s2, align 8
  %s3 = getelementptr inbounds nuw i64, ptr %sizes, i64 3
  store i64 65, ptr %s3, align 8
  %s4 = getelementptr inbounds nuw i64, ptr %sizes, i64 4
  store i64 100, ptr %s4, align 8
  %s5 = getelementptr inbounds nuw i64, ptr %sizes, i64 5
  store i64 128, ptr %s5, align 8
  %s6 = getelementptr inbounds nuw i64, ptr %sizes, i64 6
  store i64 1000, ptr %s6, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %loop ]
  %sp = getelementptr inbounds nuw i64, ptr %sizes, i64 %i
  %n = load i64, ptr %sp, align 8
  %bs = call ptr @universe_ds_bitset_create(i64 %n)
  call i32 @universe_ds_bitset_set_all(ptr %bs)
  %pc = call i64 @universe_ds_bitset_popcount(ptr %bs)
  ; tail bits above n must NOT count: popcount must equal exactly n.
  %bad = icmp ne i64 %pc, %n
  %inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %inc
  call void @universe_ds_bitset_destroy(ptr %bs)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 7
  br i1 %more, label %loop, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.setall)
  ret void
}

; --- complement keeps tail zero, dst=~a ----------------------------------
define internal void @test_complement() {
entry:
  ; n=100: set bits {5,50,99}, complement -> popcount == 100 - 3 == 97,
  ; and bits 5/50/99 become 0 while a non-set bit becomes 1.
  %a = call ptr @universe_ds_bitset_create(i64 100)
  %d = call ptr @universe_ds_bitset_create(i64 100)
  call i32 @universe_ds_bitset_set(ptr %a, i64 5)
  call i32 @universe_ds_bitset_set(ptr %a, i64 50)
  call i32 @universe_ds_bitset_set(ptr %a, i64 99)
  %rc = call i32 @universe_ds_bitset_complement(ptr %d, ptr %a)
  %pc = call i64 @universe_ds_bitset_popcount(ptr %d)
  %t5 = call i32 @universe_ds_bitset_test(ptr %d, i64 5)
  %t7 = call i32 @universe_ds_bitset_test(ptr %d, i64 7)
  %t99 = call i32 @universe_ds_bitset_test(ptr %d, i64 99)
  %c0 = icmp eq i32 %rc, 0
  %c1 = icmp eq i64 %pc, 97
  %c2 = icmp eq i32 %t5, 0
  %c3 = icmp eq i32 %t7, 1
  %c4 = icmp eq i32 %t99, 0
  %a0 = and i1 %c0, %c1
  %a1 = and i1 %a0, %c2
  %a2 = and i1 %a1, %c3
  %a3 = and i1 %a2, %c4
  call void @ut_check(i1 %a3, ptr @m.compl)
  call void @universe_ds_bitset_destroy(ptr %a)
  call void @universe_ds_bitset_destroy(ptr %d)
  ret void
}

; --- popcount and find scan vs a shadow byte model ------------------------
define internal void @test_scan() {
entry:
  %shadow = alloca [1024 x i8], align 16
  %seed = alloca i64, align 8
  store i64 424242, ptr %seed, align 8
  ; zero shadow[0..1000)
  br label %zero

zero:
  %zi = phi i64 [ 0, %entry ], [ %zi.n, %zero ]
  %zp = getelementptr inbounds nuw i8, ptr %shadow, i64 %zi
  store i8 0, ptr %zp, align 1
  %zi.n = add nuw i64 %zi, 1
  %zmore = icmp ult i64 %zi.n, 1000
  br i1 %zmore, label %zero, label %build

build:
  %bs = call ptr @universe_ds_bitset_create(i64 1000)
  br label %fill

fill:
  %fi = phi i64 [ 0, %build ], [ %fi.n, %fill ]
  %r = call i64 @ut_rand(ptr %seed)
  %bit = urem i64 %r, 1000
  call i32 @universe_ds_bitset_set(ptr %bs, i64 %bit)
  %shp = getelementptr inbounds nuw i8, ptr %shadow, i64 %bit
  store i8 1, ptr %shp, align 1
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 4000
  br i1 %fmore, label %fill, label %count

count:
  ; reference popcount = sum(shadow)
  %ci = phi i64 [ 0, %fill ], [ %ci.n, %count ]
  %cacc = phi i64 [ 0, %fill ], [ %cacc.n, %count ]
  %cp = getelementptr inbounds nuw i8, ptr %shadow, i64 %ci
  %cb = load i8, ptr %cp, align 1
  %cbz = zext i8 %cb to i64
  %cacc.n = add nuw i64 %cacc, %cbz
  %ci.n = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, 1000
  br i1 %cmore, label %count, label %check.pc

check.pc:
  %pc = call i64 @universe_ds_bitset_popcount(ptr %bs)
  call void @ut_check_eq(i64 %pc, i64 %cacc.n, ptr @m.pcrand)
  ; walk find_first / find_next; every returned idx must be shadowed set,
  ; count must equal popcount, indices strictly increasing (guaranteed by >=).
  %first = call i64 @universe_ds_bitset_find_first_set(ptr %bs)
  br label %scan

scan:
  %idx = phi i64 [ %first, %check.pc ], [ %nxt, %scan.body ]
  %sviol = phi i64 [ 0, %check.pc ], [ %sviol2, %scan.body ]
  %scnt = phi i64 [ 0, %check.pc ], [ %scnt.n, %scan.body ]
  %done.scan = icmp eq i64 %idx, -1
  br i1 %done.scan, label %scan.end, label %scan.body

scan.body:
  %shp2 = getelementptr inbounds nuw i8, ptr %shadow, i64 %idx
  %sb = load i8, ptr %shp2, align 1
  %isset = icmp eq i8 %sb, 1
  %bad = xor i1 %isset, true
  %binc = zext i1 %bad to i64
  %sviol2 = add nuw i64 %sviol, %binc
  %scnt.n = add nuw i64 %scnt, 1
  %from = add nuw i64 %idx, 1
  %nxt = call i64 @universe_ds_bitset_find_next_set(ptr %bs, i64 %from)
  br label %scan

scan.end:
  %cnt.ok = icmp eq i64 %scnt, %cacc.n
  %cnt.bad = xor i1 %cnt.ok, true
  %cntinc = zext i1 %cnt.bad to i64
  %tot = add nuw i64 %sviol, %cntinc
  call void @ut_check_eq(i64 %tot, i64 0, ptr @m.scan)
  call void @universe_ds_bitset_destroy(ptr %bs)
  ret void
}

; --- word-parallel set algebra vs brute force -----------------------------
define internal void @test_setalg() {
entry:
  %sa = alloca [512 x i8], align 16
  %sb = alloca [512 x i8], align 16
  %seed = alloca i64, align 8
  store i64 987654321, ptr %seed, align 8
  br label %zero

zero:
  %zi = phi i64 [ 0, %entry ], [ %zi.n, %zero ]
  %zap = getelementptr inbounds nuw i8, ptr %sa, i64 %zi
  %zbp = getelementptr inbounds nuw i8, ptr %sb, i64 %zi
  store i8 0, ptr %zap, align 1
  store i8 0, ptr %zbp, align 1
  %zi.n = add nuw i64 %zi, 1
  %zmore = icmp ult i64 %zi.n, 500
  br i1 %zmore, label %zero, label %build

build:
  %a = call ptr @universe_ds_bitset_create(i64 500)
  %b = call ptr @universe_ds_bitset_create(i64 500)
  %ru = call ptr @universe_ds_bitset_create(i64 500)
  %ri = call ptr @universe_ds_bitset_create(i64 500)
  %rd = call ptr @universe_ds_bitset_create(i64 500)
  br label %fill

fill:
  %fi = phi i64 [ 0, %build ], [ %fi.n, %fill ]
  %r1 = call i64 @ut_rand(ptr %seed)
  %x = urem i64 %r1, 500
  call i32 @universe_ds_bitset_set(ptr %a, i64 %x)
  %sap = getelementptr inbounds nuw i8, ptr %sa, i64 %x
  store i8 1, ptr %sap, align 1
  %r2 = call i64 @ut_rand(ptr %seed)
  %y = urem i64 %r2, 500
  call i32 @universe_ds_bitset_set(ptr %b, i64 %y)
  %sbp = getelementptr inbounds nuw i8, ptr %sb, i64 %y
  store i8 1, ptr %sbp, align 1
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 2000
  br i1 %fmore, label %fill, label %compute

compute:
  %rcu = call i32 @universe_ds_bitset_union(ptr %ru, ptr %a, ptr %b)
  %rci = call i32 @universe_ds_bitset_intersection(ptr %ri, ptr %a, ptr %b)
  %rcd = call i32 @universe_ds_bitset_difference(ptr %rd, ptr %a, ptr %b)
  br label %verify

verify:
  %vi = phi i64 [ 0, %compute ], [ %vi.n, %verify ]
  %uv = phi i64 [ 0, %compute ], [ %uv.n, %verify ]
  %iv = phi i64 [ 0, %compute ], [ %iv.n, %verify ]
  %dv = phi i64 [ 0, %compute ], [ %dv.n, %verify ]
  %ap = getelementptr inbounds nuw i8, ptr %sa, i64 %vi
  %bp = getelementptr inbounds nuw i8, ptr %sb, i64 %vi
  %av = load i8, ptr %ap, align 1
  %bv = load i8, ptr %bp, align 1
  %ab = icmp ne i8 %av, 0
  %bb = icmp ne i8 %bv, 0
  %eu = or i1 %ab, %bb
  %ei = and i1 %ab, %bb
  %notb = xor i1 %bb, true
  %ed = and i1 %ab, %notb
  %gu = call i32 @universe_ds_bitset_test(ptr %ru, i64 %vi)
  %gi = call i32 @universe_ds_bitset_test(ptr %ri, i64 %vi)
  %gd = call i32 @universe_ds_bitset_test(ptr %rd, i64 %vi)
  %gub = icmp ne i32 %gu, 0
  %gib = icmp ne i32 %gi, 0
  %gdb = icmp ne i32 %gd, 0
  %mu = xor i1 %gub, %eu
  %mi = xor i1 %gib, %ei
  %md = xor i1 %gdb, %ed
  %muz = zext i1 %mu to i64
  %miz = zext i1 %mi to i64
  %mdz = zext i1 %md to i64
  %uv.n = add nuw i64 %uv, %muz
  %iv.n = add nuw i64 %iv, %miz
  %dv.n = add nuw i64 %dv, %mdz
  %vi.n = add nuw i64 %vi, 1
  %vmore = icmp ult i64 %vi.n, 500
  br i1 %vmore, label %verify, label %report

report:
  call void @ut_check_eq(i64 %uv.n, i64 0, ptr @m.union)
  call void @ut_check_eq(i64 %iv.n, i64 0, ptr @m.inter)
  call void @ut_check_eq(i64 %dv.n, i64 0, ptr @m.diff)
  ; in-place: union(a, a, b) must equal the earlier ru result.
  %rcp = call i32 @universe_ds_bitset_union(ptr %a, ptr %a, ptr %b)
  br label %inp

inp:
  %ii = phi i64 [ 0, %report ], [ %ii.n, %inp ]
  %iviol = phi i64 [ 0, %report ], [ %iviol.n, %inp ]
  %ga = call i32 @universe_ds_bitset_test(ptr %a, i64 %ii)
  %gru = call i32 @universe_ds_bitset_test(ptr %ru, i64 %ii)
  %ine = icmp ne i32 %ga, %gru
  %iinc = zext i1 %ine to i64
  %iviol.n = add nuw i64 %iviol, %iinc
  %ii.n = add nuw i64 %ii, 1
  %imore = icmp ult i64 %ii.n, 500
  br i1 %imore, label %inp, label %inp.done

inp.done:
  call void @ut_check_eq(i64 %iviol.n, i64 0, ptr @m.inplace)
  call void @universe_ds_bitset_destroy(ptr %a)
  call void @universe_ds_bitset_destroy(ptr %b)
  call void @universe_ds_bitset_destroy(ptr %ru)
  call void @universe_ds_bitset_destroy(ptr %ri)
  call void @universe_ds_bitset_destroy(ptr %rd)
  ret void
}

; --- bench: popcount + full scan over 1M bits -----------------------------
define internal void @bench() {
entry:
  %bs = call ptr @universe_ds_bitset_create(i64 1000000)
  %seed = alloca i64, align 8
  store i64 13, ptr %seed, align 8
  br label %fill

fill:
  %fi = phi i64 [ 0, %entry ], [ %fi.n, %fill ]
  %r = call i64 @ut_rand(ptr %seed)
  %bit = urem i64 %r, 1000000
  call i32 @universe_ds_bitset_set(ptr %bs, i64 %bit)
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 200000
  br i1 %fmore, label %fill, label %orep

orep:
  %orep.i = phi i64 [ 0, %fill ], [ %orep.n, %orep.next ]
  %t0 = call double @ut_now_sec()
  br label %rep

rep:
  %ri = phi i64 [ 0, %orep ], [ %ri.n, %rep ]
  %acc = phi i64 [ 0, %orep ], [ %acc.n, %rep ]
  %pc = call i64 @universe_ds_bitset_popcount(ptr %bs)
  ; also touch the scan path once per rep
  %ff = call i64 @universe_ds_bitset_find_first_set(ptr %bs)
  %mix = add i64 %pc, %ff
  %acc.n = add i64 %acc, %mix
  %ri.n = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, 100
  br i1 %rmore, label %rep, label %orep.done

orep.done:
  %t1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr @bitset.sink, align 8
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %orep.i, 0
  br i1 %warm, label %orep.next, label %orep.store

orep.store:
  %sidx = sub i64 %orep.i, 1
  %sp = getelementptr inbounds double, ptr @bitset.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %orep.next

orep.next:
  %orep.n = add nuw nsw i64 %orep.i, 1
  %orep.more = icmp ult i64 %orep.n, 17
  br i1 %orep.more, label %orep, label %report

report:
  call void @universe_ds_bitset_destroy(ptr %bs)
  call void @ut_report_dist(ptr @bitset.samp, i64 16, i64 200, ptr @lbl.bitset)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_bits()
  call void @test_errors()
  call void @test_setall()
  call void @test_complement()
  call void @test_scan()
  call void @test_setalg()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
