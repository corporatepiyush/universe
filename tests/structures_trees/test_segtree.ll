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

; Tests for universe_ds_segtree: create/create_from + range-sum with point-SET
; correctness, every error code and index boundary (0, n-1, n, huge), edge
; sizes (1,2,3,~100k), and a 10k-op fixed-seed cross-check on n=1000 vs a plain
; i64 brute-force array (replay same SET updates, recompute range sums by brute
; force). --bench included.

declare ptr @universe_ds_segtree_create(i64)
declare ptr @universe_ds_segtree_create_from(ptr, i64)
declare i32 @universe_ds_segtree_point_update(ptr, i64, i64)
declare i64 @universe_ds_segtree_range_sum(ptr, i64, i64)
declare i64 @universe_ds_segtree_size(ptr)
declare void @universe_ds_segtree_destroy(ptr)

declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@seg.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.seg.bench = private unnamed_addr constant [24 x i8] c"segtree update+rangesum\00"

@vals8 = private unnamed_addr constant [8 x i64] [i64 3, i64 1, i64 4, i64 1, i64 5, i64 9, i64 2, i64 6]
@vals2 = private unnamed_addr constant [2 x i64] [i64 10, i64 20]

@m.basic  = private unnamed_addr constant [18 x i8] c"segtree basic ops\00"
@m.from   = private unnamed_addr constant [20 x i8] c"segtree create_from\00"
@m.err    = private unnamed_addr constant [20 x i8] c"segtree error codes\00"
@m.edge   = private unnamed_addr constant [19 x i8] c"segtree edge sizes\00"
@m.rand   = private unnamed_addr constant [26 x i8] c"segtree random crosscheck\00"
@m.size   = private unnamed_addr constant [13 x i8] c"segtree size\00"

; brute range sum over bf[lo .. hi)
define internal i64 @bf_range(ptr %bf, i64 %lo, i64 %hi) {
entry:
  %e = icmp uge i64 %lo, %hi
  br i1 %e, label %ret0, label %loop
ret0:
  ret i64 0
loop:
  %i = phi i64 [ %lo, %entry ], [ %i.n, %loop ]
  %s = phi i64 [ 0, %entry ], [ %s.n, %loop ]
  %p = getelementptr inbounds i64, ptr %bf, i64 %i
  %v = load i64, ptr %p, align 8
  %s.n = add i64 %s, %v
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %hi
  br i1 %more, label %loop, label %fin
fin:
  ret i64 %s.n
}

define internal void @bump_if(i1 %bad, ptr %viol) {
entry:
  br i1 %bad, label %inc, label %done
inc:
  %v = load i64, ptr %viol, align 8
  %v.n = add nuw i64 %v, 1
  store i64 %v.n, ptr %viol, align 8
  br label %done
done:
  ret void
}

define internal void @test_basic() {
entry:
  ; create empty n=8 (zeroed), then SET values [3,1,4,1,5,9,2,6]
  %s = call ptr @universe_ds_segtree_create(i64 8)
  %z = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 8)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 0, i64 3)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 1, i64 1)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 2, i64 4)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 3, i64 1)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 4, i64 5)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 5, i64 9)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 6, i64 2)
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 7, i64 6)
  %all = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 8)
  %r25 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 2, i64 5)
  %r01 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 1)
  %one = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 5, i64 6)
  ; SET (overwrite, NOT add) index 5 from 9 to -4
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 5, i64 -4)
  %all2 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 8)
  %one2 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 5, i64 6)
  %sz = call i64 @universe_ds_segtree_size(ptr %s)
  %ok.z = icmp eq i64 %z, 0
  %ok.all = icmp eq i64 %all, 31
  %ok.r25 = icmp eq i64 %r25, 10
  %ok.r01 = icmp eq i64 %r01, 3
  %ok.one = icmp eq i64 %one, 9
  %ok.all2 = icmp eq i64 %all2, 18
  %ok.one2 = icmp eq i64 %one2, -4
  %a1 = and i1 %ok.z, %ok.all
  %a2 = and i1 %a1, %ok.r25
  %a3 = and i1 %a2, %ok.r01
  %a4 = and i1 %a3, %ok.one
  %a5 = and i1 %a4, %ok.all2
  %a6 = and i1 %a5, %ok.one2
  call void @ut_check(i1 %a6, ptr @m.basic)
  %ok.sz = icmp eq i64 %sz, 8
  call void @ut_check(i1 %ok.sz, ptr @m.size)
  call void @universe_ds_segtree_destroy(ptr %s)
  ret void
}

define internal void @test_from() {
entry:
  ; build directly from [3,1,4,1,5,9,2,6]
  %s = call ptr @universe_ds_segtree_create_from(ptr @vals8, i64 8)
  %all = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 8)
  %r14 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 1, i64 4)
  %r77 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 7, i64 8)
  ; update then re-query
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 0, i64 100)
  %all2 = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 3)
  %ok.all = icmp eq i64 %all, 31
  %ok.r14 = icmp eq i64 %r14, 6
  %ok.r77 = icmp eq i64 %r77, 6
  %ok.all2 = icmp eq i64 %all2, 105
  %a1 = and i1 %ok.all, %ok.r14
  %a2 = and i1 %a1, %ok.r77
  %a3 = and i1 %a2, %ok.all2
  call void @ut_check(i1 %a3, ptr @m.from)
  call void @universe_ds_segtree_destroy(ptr %s)
  ret void
}

define internal void @test_errors() {
entry:
  %s = call ptr @universe_ds_segtree_create(i64 4)
  ; create_from null vals -> null ptr
  %fnull = call ptr @universe_ds_segtree_create_from(ptr null, i64 4)
  ; point_update null -> 1
  %u.null = call i32 @universe_ds_segtree_point_update(ptr null, i64 0, i64 1)
  ; boundary indices: 0 valid, n-1=3 valid, n=4 invalid, huge invalid
  %u.lo = call i32 @universe_ds_segtree_point_update(ptr %s, i64 0, i64 1)
  %u.hi = call i32 @universe_ds_segtree_point_update(ptr %s, i64 3, i64 1)
  %u.n = call i32 @universe_ds_segtree_point_update(ptr %s, i64 4, i64 1)
  %u.huge = call i32 @universe_ds_segtree_point_update(ptr %s, i64 -1, i64 1)
  ; range_sum null -> 0
  %q.null = call i64 @universe_ds_segtree_range_sum(ptr null, i64 0, i64 4)
  ; range_sum with lo>=hi -> 0
  %q.empty = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 3, i64 2)
  ; range_sum hi past n is clamped: [0,100) == [0,4)
  %q.clamp = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 100)
  ; only indices 0 and 3 were set to 1 => sum 2
  %ok.fnull = icmp eq ptr %fnull, null
  %ok.unull = icmp eq i32 %u.null, 1
  %ok.ulo = icmp eq i32 %u.lo, 0
  %ok.uhi = icmp eq i32 %u.hi, 0
  %ok.un = icmp eq i32 %u.n, 7
  %ok.uhuge = icmp eq i32 %u.huge, 7
  %ok.qnull = icmp eq i64 %q.null, 0
  %ok.qempty = icmp eq i64 %q.empty, 0
  %ok.qclamp = icmp eq i64 %q.clamp, 2
  %b1 = and i1 %ok.fnull, %ok.unull
  %b2 = and i1 %b1, %ok.ulo
  %b3 = and i1 %b2, %ok.uhi
  %b4 = and i1 %b3, %ok.un
  %b5 = and i1 %b4, %ok.uhuge
  %b6 = and i1 %b5, %ok.qnull
  %b7 = and i1 %b6, %ok.qempty
  %b8 = and i1 %b7, %ok.qclamp
  call void @ut_check(i1 %b8, ptr @m.err)
  call void @universe_ds_segtree_destroy(ptr %s)
  call void @universe_ds_segtree_destroy(ptr null)
  ret void
}

define internal void @test_edges() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  ; n=1
  %s1 = call ptr @universe_ds_segtree_create(i64 1)
  call i32 @universe_ds_segtree_point_update(ptr %s1, i64 0, i64 42)
  %r1 = call i64 @universe_ds_segtree_range_sum(ptr %s1, i64 0, i64 1)
  %bad1 = icmp ne i64 %r1, 42
  call void @bump_if(i1 %bad1, ptr %viol)
  %sz1 = call i64 @universe_ds_segtree_size(ptr %s1)
  %bad1s = icmp ne i64 %sz1, 1
  call void @bump_if(i1 %bad1s, ptr %viol)
  call void @universe_ds_segtree_destroy(ptr %s1)
  ; n=2 via create_from [10,20]
  %s2 = call ptr @universe_ds_segtree_create_from(ptr @vals2, i64 2)
  %r2a = call i64 @universe_ds_segtree_range_sum(ptr %s2, i64 0, i64 2)
  %bad2a = icmp ne i64 %r2a, 30
  call void @bump_if(i1 %bad2a, ptr %viol)
  %r2b = call i64 @universe_ds_segtree_range_sum(ptr %s2, i64 1, i64 2)
  %bad2b = icmp ne i64 %r2b, 20
  call void @bump_if(i1 %bad2b, ptr %viol)
  call void @universe_ds_segtree_destroy(ptr %s2)
  ; n=3
  %s3 = call ptr @universe_ds_segtree_create(i64 3)
  call i32 @universe_ds_segtree_point_update(ptr %s3, i64 0, i64 7)
  call i32 @universe_ds_segtree_point_update(ptr %s3, i64 1, i64 8)
  call i32 @universe_ds_segtree_point_update(ptr %s3, i64 2, i64 9)
  %r3a = call i64 @universe_ds_segtree_range_sum(ptr %s3, i64 0, i64 3)
  %bad3a = icmp ne i64 %r3a, 24
  call void @bump_if(i1 %bad3a, ptr %viol)
  %r3b = call i64 @universe_ds_segtree_range_sum(ptr %s3, i64 1, i64 3)
  %bad3b = icmp ne i64 %r3b, 17
  call void @bump_if(i1 %bad3b, ptr %viol)
  call void @universe_ds_segtree_destroy(ptr %s3)
  ; large n=100000: set leaf i = i; range(0,n)=n(n-1)/2, range(1000,2000)=1499500
  %sL = call ptr @universe_ds_segtree_create(i64 100000)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  call i32 @universe_ds_segtree_point_update(ptr %sL, i64 %i, i64 %i)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %fill, label %chkL
chkL:
  %rLa = call i64 @universe_ds_segtree_range_sum(ptr %sL, i64 0, i64 100000)
  %badLa = icmp ne i64 %rLa, 4999950000
  call void @bump_if(i1 %badLa, ptr %viol)
  %rLb = call i64 @universe_ds_segtree_range_sum(ptr %sL, i64 1000, i64 2000)
  %badLb = icmp ne i64 %rLb, 1499500
  call void @bump_if(i1 %badLb, ptr %viol)
  call void @universe_ds_segtree_destroy(ptr %sL)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.edge)
  ret void
}

; 10k random SET updates on n=1000, cross-checked vs a plain i64 brute array.
define internal void @test_random() {
entry:
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  %n = add i64 0, 1000
  %s = call ptr @universe_ds_segtree_create(i64 %n)
  %bf = call ptr @calloc(i64 %n, i64 8)
  %np1 = add i64 %n, 1
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %k.n, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %loop ]
  ; pick index and a signed value in [-500,500]
  %r0 = call i64 @ut_rand(ptr %state)
  %idx = urem i64 %r0, %n
  %r1 = call i64 @ut_rand(ptr %state)
  %m = urem i64 %r1, 1001
  %val = sub i64 %m, 500
  ; SET on both
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 %idx, i64 %val)
  %bp = getelementptr inbounds i64, ptr %bf, i64 %idx
  store i64 %val, ptr %bp, align 8
  ; pick lo,hi in [0,n]
  %r2 = call i64 @ut_rand(ptr %state)
  %r3 = call i64 @ut_rand(ptr %state)
  %aa = urem i64 %r2, %np1
  %bb = urem i64 %r3, %np1
  %swap = icmp ugt i64 %aa, %bb
  %lo = select i1 %swap, i64 %bb, i64 %aa
  %hi = select i1 %swap, i64 %aa, i64 %bb
  %got = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 %lo, i64 %hi)
  %want = call i64 @bf_range(ptr %bf, i64 %lo, i64 %hi)
  %bad = icmp ne i64 %got, %want
  %inc = zext i1 %bad to i64
  %viol.n = add i64 %viol, %inc
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 10000
  br i1 %more, label %loop, label %fin

fin:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.rand)
  call void @universe_ds_segtree_destroy(ptr %s)
  call void @free(ptr %bf)
  ret void
}

define internal void @bench() {
entry:
  %state = alloca i64, align 8
  %vsink = alloca i64, align 8
  %n = add i64 0, 100000
  %s = call ptr @universe_ds_segtree_create(i64 %n)
  br label %rep.head
rep.head:
  %rep = phi i64 [ 0, %entry ], [ %rep.n, %rep.cont ]
  store i64 12345678901234567, ptr %state, align 8
  %rt0 = call double @ut_now_sec()
  br label %loop
loop:
  %k = phi i64 [ 0, %rep.head ], [ %k.n, %loop ]
  %acc = phi i64 [ 0, %rep.head ], [ %acc.n, %loop ]
  %r0 = call i64 @ut_rand(ptr %state)
  %idx = urem i64 %r0, %n
  call i32 @universe_ds_segtree_point_update(ptr %s, i64 %idx, i64 1)
  %r1 = call i64 @ut_rand(ptr %state)
  %c = urem i64 %r1, %n
  %q = call i64 @universe_ds_segtree_range_sum(ptr %s, i64 0, i64 %c)
  %acc.n = add i64 %acc, %q
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 1000000
  br i1 %more, label %loop, label %rep.done
rep.done:
  %rt1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr %vsink, align 8
  %elapsed = fsub double %rt1, %rt0
  %warm = icmp eq i64 %rep, 0
  br i1 %warm, label %rep.cont, label %rep.store
rep.store:
  %sidx = sub i64 %rep, 1
  %sp = getelementptr inbounds double, ptr @seg.bench.samp, i64 %sidx
  store double %elapsed, ptr %sp, align 8
  br label %rep.cont
rep.cont:
  %rep.n = add nuw i64 %rep, 1
  %repmore = icmp ult i64 %rep.n, 17
  br i1 %repmore, label %rep.head, label %rep.report
rep.report:
  call void @ut_report_dist(ptr @seg.bench.samp, i64 16, i64 2000000, ptr @lbl.seg.bench)
  call void @universe_ds_segtree_destroy(ptr %s)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_from()
  call void @test_errors()
  call void @test_edges()
  call void @test_random()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish
do.bench:
  call void @bench()
  br label %finish
finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
