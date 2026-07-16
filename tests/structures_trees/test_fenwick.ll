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

; Tests for universe_ds_fenwick: basic prefix/range/point correctness, every
; error code and index boundary (0, n-1, n, huge), edge sizes (1,2,3,~100k),
; a 10k-op fixed-seed cross-check vs a brute-force i64 array, and --bench.

declare ptr @universe_ds_fenwick_create(i64)
declare i32 @universe_ds_fenwick_update(ptr, i64, i64)
declare i64 @universe_ds_fenwick_prefix_sum(ptr, i64)
declare i64 @universe_ds_fenwick_range_sum(ptr, i64, i64)
declare i64 @universe_ds_fenwick_point_get(ptr, i64)
declare i64 @universe_ds_fenwick_size(ptr)
declare void @universe_ds_fenwick_destroy(ptr)

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

@fen.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.fen.bench = private unnamed_addr constant [22 x i8] c"fenwick update+prefix\00"

@m.basic  = private unnamed_addr constant [18 x i8] c"fenwick basic ops\00"
@m.err    = private unnamed_addr constant [20 x i8] c"fenwick error codes\00"
@m.edge   = private unnamed_addr constant [19 x i8] c"fenwick edge sizes\00"
@m.rand   = private unnamed_addr constant [26 x i8] c"fenwick random crosscheck\00"
@m.size   = private unnamed_addr constant [13 x i8] c"fenwick size\00"

; brute-force prefix: sum of bf[0 .. count)
define internal i64 @bf_prefix(ptr %bf, i64 %count) {
entry:
  %z = icmp eq i64 %count, 0
  br i1 %z, label %ret0, label %loop
ret0:
  ret i64 0
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %s = phi i64 [ 0, %entry ], [ %s.n, %loop ]
  %p = getelementptr inbounds i64, ptr %bf, i64 %i
  %v = load i64, ptr %p, align 8
  %s.n = add i64 %s, %v
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %fin
fin:
  ret i64 %s.n
}

define internal void @test_basic() {
entry:
  ; n=8, set element values via update (add): [3,1,4,1,5,9,2,6]
  %f = call ptr @universe_ds_fenwick_create(i64 8)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 0, i64 3)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 1, i64 1)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 2, i64 4)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 3, i64 1)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 4, i64 5)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 5, i64 9)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 6, i64 2)
  call i32 @universe_ds_fenwick_update(ptr %f, i64 7, i64 6)
  ; prefix_sum(0)=0, (1)=3, (4)=9, (8)=31
  %p0 = call i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 0)
  %p1 = call i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 1)
  %p4 = call i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 4)
  %p8 = call i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 8)
  ; range_sum(2,5)=4+1+5=10 ; point_get(5)=9
  %r25 = call i64 @universe_ds_fenwick_range_sum(ptr %f, i64 2, i64 5)
  %g5 = call i64 @universe_ds_fenwick_point_get(ptr %f, i64 5)
  ; extra: negative delta then re-check point
  call i32 @universe_ds_fenwick_update(ptr %f, i64 5, i64 -4)
  %g5b = call i64 @universe_ds_fenwick_point_get(ptr %f, i64 5)
  %c0 = icmp eq i64 %p0, 0
  %c1 = icmp eq i64 %p1, 3
  %c2 = icmp eq i64 %p4, 9
  %c3 = icmp eq i64 %p8, 31
  %c4 = icmp eq i64 %r25, 10
  %c5 = icmp eq i64 %g5, 9
  %c6 = icmp eq i64 %g5b, 5
  %a1 = and i1 %c0, %c1
  %a2 = and i1 %a1, %c2
  %a3 = and i1 %a2, %c3
  %a4 = and i1 %a3, %c4
  %a5 = and i1 %a4, %c5
  %a6 = and i1 %a5, %c6
  call void @ut_check(i1 %a6, ptr @m.basic)
  %sz = call i64 @universe_ds_fenwick_size(ptr %f)
  %szok = icmp eq i64 %sz, 8
  call void @ut_check(i1 %szok, ptr @m.size)
  call void @universe_ds_fenwick_destroy(ptr %f)
  ret void
}

define internal void @test_errors() {
entry:
  %f = call ptr @universe_ds_fenwick_create(i64 4)
  ; null handle -> 1
  %e.null = call i32 @universe_ds_fenwick_update(ptr null, i64 0, i64 1)
  ; index 0 (valid) -> 0 ; index n-1=3 (valid) -> 0
  %e.lo = call i32 @universe_ds_fenwick_update(ptr %f, i64 0, i64 1)
  %e.hi = call i32 @universe_ds_fenwick_update(ptr %f, i64 3, i64 1)
  ; index n=4 -> 7 ; huge -> 7
  %e.n = call i32 @universe_ds_fenwick_update(ptr %f, i64 4, i64 1)
  %e.huge = call i32 @universe_ds_fenwick_update(ptr %f, i64 -1, i64 1)
  ; null-safe value queries -> 0
  %q.null = call i64 @universe_ds_fenwick_prefix_sum(ptr null, i64 3)
  %g.null = call i64 @universe_ds_fenwick_point_get(ptr null, i64 0)
  ; point_get out of range -> 0
  %g.oob = call i64 @universe_ds_fenwick_point_get(ptr %f, i64 4)
  %ok1 = icmp eq i32 %e.null, 1
  %ok2 = icmp eq i32 %e.lo, 0
  %ok3 = icmp eq i32 %e.hi, 0
  %ok4 = icmp eq i32 %e.n, 7
  %ok5 = icmp eq i32 %e.huge, 7
  %ok6 = icmp eq i64 %q.null, 0
  %ok7 = icmp eq i64 %g.null, 0
  %ok8 = icmp eq i64 %g.oob, 0
  %b1 = and i1 %ok1, %ok2
  %b2 = and i1 %b1, %ok3
  %b3 = and i1 %b2, %ok4
  %b4 = and i1 %b3, %ok5
  %b5 = and i1 %b4, %ok6
  %b6 = and i1 %b5, %ok7
  %b7 = and i1 %b6, %ok8
  call void @ut_check(i1 %b7, ptr @m.err)
  call void @universe_ds_fenwick_destroy(ptr %f)
  call void @universe_ds_fenwick_destroy(ptr null)
  ret void
}

; Exercise n = 1,2,3 and a large ~100k tree; verify prefix totals.
define internal void @test_edges() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  ; n=1
  %f1 = call ptr @universe_ds_fenwick_create(i64 1)
  call i32 @universe_ds_fenwick_update(ptr %f1, i64 0, i64 42)
  %s1 = call i64 @universe_ds_fenwick_prefix_sum(ptr %f1, i64 1)
  %bad1 = icmp ne i64 %s1, 42
  call void @bump_if(i1 %bad1, ptr %viol)
  call void @universe_ds_fenwick_destroy(ptr %f1)
  ; n=2
  %f2 = call ptr @universe_ds_fenwick_create(i64 2)
  call i32 @universe_ds_fenwick_update(ptr %f2, i64 0, i64 5)
  call i32 @universe_ds_fenwick_update(ptr %f2, i64 1, i64 7)
  %s2 = call i64 @universe_ds_fenwick_range_sum(ptr %f2, i64 0, i64 2)
  %bad2 = icmp ne i64 %s2, 12
  call void @bump_if(i1 %bad2, ptr %viol)
  call void @universe_ds_fenwick_destroy(ptr %f2)
  ; n=3
  %f3 = call ptr @universe_ds_fenwick_create(i64 3)
  call i32 @universe_ds_fenwick_update(ptr %f3, i64 2, i64 100)
  %g3 = call i64 @universe_ds_fenwick_point_get(ptr %f3, i64 2)
  %bad3 = icmp ne i64 %g3, 100
  call void @bump_if(i1 %bad3, ptr %viol)
  call void @universe_ds_fenwick_destroy(ptr %f3)
  ; large n=100000: update every index by its own index; prefix(n) = n(n-1)/2
  %fL = call ptr @universe_ds_fenwick_create(i64 100000)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  call i32 @universe_ds_fenwick_update(ptr %fL, i64 %i, i64 %i)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %fill, label %chkL
chkL:
  %sL = call i64 @universe_ds_fenwick_prefix_sum(ptr %fL, i64 100000)
  ; sum 0..99999 = 4999950000
  %badL = icmp ne i64 %sL, 4999950000
  call void @bump_if(i1 %badL, ptr %viol)
  ; a mid range [1000,2000): sum i for i in 1000..1999 = (1000+1999)*1000/2 = 1499500
  %mL = call i64 @universe_ds_fenwick_range_sum(ptr %fL, i64 1000, i64 2000)
  %badM = icmp ne i64 %mL, 1499500
  call void @bump_if(i1 %badM, ptr %viol)
  call void @universe_ds_fenwick_destroy(ptr %fL)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.edge)
  ret void
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

; 10k random ops on n=1000, cross-checked against a brute-force i64 array.
define internal void @test_random() {
entry:
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  %n = add i64 0, 1000
  %f = call ptr @universe_ds_fenwick_create(i64 %n)
  %bf = call ptr @calloc(i64 %n, i64 8)
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %k.n, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %viol.f, %loop ]
  ; pick index
  %r0 = call i64 @ut_rand(ptr %state)
  %idx = urem i64 %r0, %n
  ; pick delta in [-100,100]
  %r1 = call i64 @ut_rand(ptr %state)
  %m = urem i64 %r1, 201
  %delta = sub i64 %m, 100
  ; apply to both
  call i32 @universe_ds_fenwick_update(ptr %f, i64 %idx, i64 %delta)
  %bp = getelementptr inbounds i64, ptr %bf, i64 %idx
  %bv = load i64, ptr %bp, align 8
  %bv.n = add i64 %bv, %delta
  store i64 %bv.n, ptr %bp, align 8
  ; pick lo,hi
  %r2 = call i64 @ut_rand(ptr %state)
  %r3 = call i64 @ut_rand(ptr %state)
  %np1 = add i64 %n, 1
  %a = urem i64 %r2, %np1
  %b = urem i64 %r3, %np1
  %swap = icmp ugt i64 %a, %b
  %lo = select i1 %swap, i64 %b, i64 %a
  %hi = select i1 %swap, i64 %a, i64 %b
  ; compare range_sum vs brute
  %got = call i64 @universe_ds_fenwick_range_sum(ptr %f, i64 %lo, i64 %hi)
  %ph = call i64 @bf_prefix(ptr %bf, i64 %hi)
  %pl = call i64 @bf_prefix(ptr %bf, i64 %lo)
  %want = sub i64 %ph, %pl
  %bad.r = icmp ne i64 %got, %want
  ; compare point_get(idx) vs bf[idx]
  %gp = call i64 @universe_ds_fenwick_point_get(ptr %f, i64 %idx)
  %bad.p = icmp ne i64 %gp, %bv.n
  %bad = or i1 %bad.r, %bad.p
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 10000
  br i1 %more, label %loop, label %fin

fin:
  call void @ut_check_eq(i64 %viol.f, i64 0, ptr @m.rand)
  call void @universe_ds_fenwick_destroy(ptr %f)
  call void @free(ptr %bf)
  ret void
}

define internal void @bench() {
entry:
  %state = alloca i64, align 8
  %vsink = alloca i64, align 8
  %n = add i64 0, 100000
  %f = call ptr @universe_ds_fenwick_create(i64 %n)
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
  call i32 @universe_ds_fenwick_update(ptr %f, i64 %idx, i64 1)
  %r1 = call i64 @ut_rand(ptr %state)
  %c = urem i64 %r1, %n
  %q = call i64 @universe_ds_fenwick_prefix_sum(ptr %f, i64 %c)
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
  %sp = getelementptr inbounds double, ptr @fen.bench.samp, i64 %sidx
  store double %elapsed, ptr %sp, align 8
  br label %rep.cont
rep.cont:
  %rep.n = add nuw i64 %rep, 1
  %repmore = icmp ult i64 %rep.n, 17
  br i1 %repmore, label %rep.head, label %rep.report
rep.report:
  call void @ut_report_dist(ptr @fen.bench.samp, i64 16, i64 2000000, ptr @lbl.fen.bench)
  call void @universe_ds_fenwick_destroy(ptr %f)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
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
