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

; Tests for universe_ds_unionfind: basic find/unite/connected/count_sets/
; set_size correctness, every error code and index boundary (0, n-1, n, huge),
; edge sizes (1,2,3,~100k), and a 10k-op fixed-seed cross-check on n=1000 vs an
; O(n) shadow label array with relabel-on-union. --bench included.

declare ptr @universe_ds_unionfind_create(i64)
declare i64 @universe_ds_unionfind_find(ptr, i64)
declare i32 @universe_ds_unionfind_unite(ptr, i64, i64)
declare i32 @universe_ds_unionfind_connected(ptr, i64, i64)
declare i64 @universe_ds_unionfind_count_sets(ptr)
declare i64 @universe_ds_unionfind_set_size(ptr, i64)
declare void @universe_ds_unionfind_destroy(ptr)

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

@uf.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.uf.bench = private unnamed_addr constant [26 x i8] c"unionfind unite+connected\00"

@m.basic  = private unnamed_addr constant [20 x i8] c"unionfind basic ops\00"
@m.err    = private unnamed_addr constant [22 x i8] c"unionfind error codes\00"
@m.edge   = private unnamed_addr constant [21 x i8] c"unionfind edge sizes\00"
@m.rand   = private unnamed_addr constant [28 x i8] c"unionfind random crosscheck\00"
@m.cnt    = private unnamed_addr constant [25 x i8] c"unionfind final setcount\00"

; shadow: set every label[i]==from to to, over i in [0,n).
define internal void @relabel(ptr %label, i64 %n, i64 %from, i64 %to) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %next ]
  %p = getelementptr inbounds i64, ptr %label, i64 %i
  %v = load i64, ptr %p, align 8
  %hit = icmp eq i64 %v, %from
  br i1 %hit, label %set, label %next
set:
  store i64 %to, ptr %p, align 8
  br label %next
next:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

define internal void @test_basic() {
entry:
  ; n=6, singletons initially
  %u = call ptr @universe_ds_unionfind_create(i64 6)
  %c0 = call i64 @universe_ds_unionfind_count_sets(ptr %u)
  ; unite 0-1, 2-3, then 1-2 => {0,1,2,3},{4},{5}
  %r01 = call i32 @universe_ds_unionfind_unite(ptr %u, i64 0, i64 1)
  %r23 = call i32 @universe_ds_unionfind_unite(ptr %u, i64 2, i64 3)
  %r12 = call i32 @universe_ds_unionfind_unite(ptr %u, i64 1, i64 2)
  ; unite already-together => 9
  %rdup = call i32 @universe_ds_unionfind_unite(ptr %u, i64 0, i64 3)
  %cn = call i64 @universe_ds_unionfind_count_sets(ptr %u)
  ; connectivity
  %c03 = call i32 @universe_ds_unionfind_connected(ptr %u, i64 0, i64 3)
  %c04 = call i32 @universe_ds_unionfind_connected(ptr %u, i64 0, i64 4)
  ; set sizes
  %sz0 = call i64 @universe_ds_unionfind_set_size(ptr %u, i64 0)
  %sz4 = call i64 @universe_ds_unionfind_set_size(ptr %u, i64 4)
  ; find agreement: 0,1,2,3 share one root
  %f0 = call i64 @universe_ds_unionfind_find(ptr %u, i64 0)
  %f3 = call i64 @universe_ds_unionfind_find(ptr %u, i64 3)
  %ok.c0 = icmp eq i64 %c0, 6
  %ok.r01 = icmp eq i32 %r01, 0
  %ok.r23 = icmp eq i32 %r23, 0
  %ok.r12 = icmp eq i32 %r12, 0
  %ok.dup = icmp eq i32 %rdup, 9
  %ok.cn = icmp eq i64 %cn, 3
  %ok.c03 = icmp eq i32 %c03, 1
  %ok.c04 = icmp eq i32 %c04, 0
  %ok.sz0 = icmp eq i64 %sz0, 4
  %ok.sz4 = icmp eq i64 %sz4, 1
  %ok.find = icmp eq i64 %f0, %f3
  %a1 = and i1 %ok.c0, %ok.r01
  %a2 = and i1 %a1, %ok.r23
  %a3 = and i1 %a2, %ok.r12
  %a4 = and i1 %a3, %ok.dup
  %a5 = and i1 %a4, %ok.cn
  %a6 = and i1 %a5, %ok.c03
  %a7 = and i1 %a6, %ok.c04
  %a8 = and i1 %a7, %ok.sz0
  %a9 = and i1 %a8, %ok.sz4
  %a10 = and i1 %a9, %ok.find
  call void @ut_check(i1 %a10, ptr @m.basic)
  call void @universe_ds_unionfind_destroy(ptr %u)
  ret void
}

define internal void @test_errors() {
entry:
  %u = call ptr @universe_ds_unionfind_create(i64 4)
  ; null handle conventions
  %f.null = call i64 @universe_ds_unionfind_find(ptr null, i64 0)
  %u.null = call i32 @universe_ds_unionfind_unite(ptr null, i64 0, i64 1)
  %c.null = call i32 @universe_ds_unionfind_connected(ptr null, i64 0, i64 1)
  %s.null = call i64 @universe_ds_unionfind_set_size(ptr null, i64 0)
  %n.null = call i64 @universe_ds_unionfind_count_sets(ptr null)
  ; boundary indices: 0 valid, n-1=3 valid, n=4 invalid, huge invalid
  %f.lo = call i64 @universe_ds_unionfind_find(ptr %u, i64 0)
  %f.hi = call i64 @universe_ds_unionfind_find(ptr %u, i64 3)
  %f.n = call i64 @universe_ds_unionfind_find(ptr %u, i64 4)
  %f.huge = call i64 @universe_ds_unionfind_find(ptr %u, i64 -1)
  ; unite oob on either arg -> 7
  %u.oa = call i32 @universe_ds_unionfind_unite(ptr %u, i64 4, i64 0)
  %u.ob = call i32 @universe_ds_unionfind_unite(ptr %u, i64 0, i64 4)
  ; connected oob -> -1
  %c.oob = call i32 @universe_ds_unionfind_connected(ptr %u, i64 0, i64 4)
  ; set_size oob -> 0
  %s.oob = call i64 @universe_ds_unionfind_set_size(ptr %u, i64 4)
  ; valid unite at boundary index n-1
  %u.ok = call i32 @universe_ds_unionfind_unite(ptr %u, i64 0, i64 3)
  %ok1 = icmp eq i64 %f.null, -1
  %ok2 = icmp eq i32 %u.null, 1
  %ok3 = icmp eq i32 %c.null, -1
  %ok4 = icmp eq i64 %s.null, 0
  %ok5 = icmp eq i64 %n.null, 0
  %ok6 = icmp eq i64 %f.lo, 0
  %ok7 = icmp eq i64 %f.hi, 3
  %ok8 = icmp eq i64 %f.n, -1
  %ok9 = icmp eq i64 %f.huge, -1
  %ok10 = icmp eq i32 %u.oa, 7
  %ok11 = icmp eq i32 %u.ob, 7
  %ok12 = icmp eq i32 %c.oob, -1
  %ok13 = icmp eq i64 %s.oob, 0
  %ok14 = icmp eq i32 %u.ok, 0
  %b1 = and i1 %ok1, %ok2
  %b2 = and i1 %b1, %ok3
  %b3 = and i1 %b2, %ok4
  %b4 = and i1 %b3, %ok5
  %b5 = and i1 %b4, %ok6
  %b6 = and i1 %b5, %ok7
  %b7 = and i1 %b6, %ok8
  %b8 = and i1 %b7, %ok9
  %b9 = and i1 %b8, %ok10
  %b10 = and i1 %b9, %ok11
  %b11 = and i1 %b10, %ok12
  %b12 = and i1 %b11, %ok13
  %b13 = and i1 %b12, %ok14
  call void @ut_check(i1 %b13, ptr @m.err)
  call void @universe_ds_unionfind_destroy(ptr %u)
  call void @universe_ds_unionfind_destroy(ptr null)
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

; n=1,2,3 and a large ~100k structure.
define internal void @test_edges() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  ; n=1: single set, self connected, size 1, count 1
  %u1 = call ptr @universe_ds_unionfind_create(i64 1)
  %c1 = call i32 @universe_ds_unionfind_connected(ptr %u1, i64 0, i64 0)
  %s1 = call i64 @universe_ds_unionfind_set_size(ptr %u1, i64 0)
  %n1 = call i64 @universe_ds_unionfind_count_sets(ptr %u1)
  %bad1a = icmp ne i32 %c1, 1
  call void @bump_if(i1 %bad1a, ptr %viol)
  %bad1b = icmp ne i64 %s1, 1
  call void @bump_if(i1 %bad1b, ptr %viol)
  %bad1c = icmp ne i64 %n1, 1
  call void @bump_if(i1 %bad1c, ptr %viol)
  call void @universe_ds_unionfind_destroy(ptr %u1)
  ; n=2: unite then one set of size 2
  %u2 = call ptr @universe_ds_unionfind_create(i64 2)
  call i32 @universe_ds_unionfind_unite(ptr %u2, i64 0, i64 1)
  %s2 = call i64 @universe_ds_unionfind_set_size(ptr %u2, i64 1)
  %n2 = call i64 @universe_ds_unionfind_count_sets(ptr %u2)
  %bad2a = icmp ne i64 %s2, 2
  call void @bump_if(i1 %bad2a, ptr %viol)
  %bad2b = icmp ne i64 %n2, 1
  call void @bump_if(i1 %bad2b, ptr %viol)
  call void @universe_ds_unionfind_destroy(ptr %u2)
  ; n=3: unite 0-2, leaves {0,2},{1}
  %u3 = call ptr @universe_ds_unionfind_create(i64 3)
  call i32 @universe_ds_unionfind_unite(ptr %u3, i64 0, i64 2)
  %c3 = call i32 @universe_ds_unionfind_connected(ptr %u3, i64 0, i64 2)
  %c3b = call i32 @universe_ds_unionfind_connected(ptr %u3, i64 0, i64 1)
  %n3 = call i64 @universe_ds_unionfind_count_sets(ptr %u3)
  %bad3a = icmp ne i32 %c3, 1
  call void @bump_if(i1 %bad3a, ptr %viol)
  %bad3b = icmp ne i32 %c3b, 0
  call void @bump_if(i1 %bad3b, ptr %viol)
  %bad3c = icmp ne i64 %n3, 2
  call void @bump_if(i1 %bad3c, ptr %viol)
  call void @universe_ds_unionfind_destroy(ptr %u3)
  ; large n=100000: chain-unite i with i+1, ending in one set of size n
  %uL = call ptr @universe_ds_unionfind_create(i64 100000)
  br label %chain
chain:
  %i = phi i64 [ 0, %entry ], [ %i.n, %chain ]
  %j = add nuw i64 %i, 1
  call i32 @universe_ds_unionfind_unite(ptr %uL, i64 %i, i64 %j)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 99999
  br i1 %more, label %chain, label %chkL
chkL:
  %nL = call i64 @universe_ds_unionfind_count_sets(ptr %uL)
  %badL1 = icmp ne i64 %nL, 1
  call void @bump_if(i1 %badL1, ptr %viol)
  %szL = call i64 @universe_ds_unionfind_set_size(ptr %uL, i64 50000)
  %badL2 = icmp ne i64 %szL, 100000
  call void @bump_if(i1 %badL2, ptr %viol)
  %cL = call i32 @universe_ds_unionfind_connected(ptr %uL, i64 0, i64 99999)
  %badL3 = icmp ne i32 %cL, 1
  call void @bump_if(i1 %badL3, ptr %viol)
  call void @universe_ds_unionfind_destroy(ptr %uL)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.edge)
  ret void
}

; 10k random unions on n=1000 cross-checked vs an O(n) shadow label array;
; connected() must match label equality and count_sets must match the shadow.
define internal void @test_random() {
entry:
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  %n = add i64 0, 1000
  %u = call ptr @universe_ds_unionfind_create(i64 %n)
  %label = call ptr @calloc(i64 %n, i64 8)
  br label %fill
fill:
  %fi = phi i64 [ 0, %entry ], [ %fi.n, %fill ]
  %fp = getelementptr inbounds i64, ptr %label, i64 %fi
  store i64 %fi, ptr %fp, align 8
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, %n
  br i1 %fmore, label %fill, label %loop

loop:
  %k = phi i64 [ 0, %fill ], [ %k.n, %after ]
  %viol = phi i64 [ 0, %fill ], [ %viol.n, %after ]
  %setc = phi i64 [ 1000, %fill ], [ %setc.n, %after ]
  ; pick a,b
  %r0 = call i64 @ut_rand(ptr %state)
  %a = urem i64 %r0, %n
  %r1 = call i64 @ut_rand(ptr %state)
  %b = urem i64 %r1, %n
  %la.p = getelementptr inbounds i64, ptr %label, i64 %a
  %la = load i64, ptr %la.p, align 8
  %lb.p = getelementptr inbounds i64, ptr %label, i64 %b
  %lb = load i64, ptr %lb.p, align 8
  %same = icmp eq i64 %la, %lb
  %rc = call i32 @universe_ds_unionfind_unite(ptr %u, i64 %a, i64 %b)
  %exp = select i1 %same, i32 9, i32 0
  %bad.rc = icmp ne i32 %rc, %exp
  %inc.rc = zext i1 %bad.rc to i64
  br i1 %same, label %skip, label %do

do:
  call void @relabel(ptr %label, i64 %n, i64 %lb, i64 %la)
  %setc.dec = sub i64 %setc, 1
  br label %merge

skip:
  br label %merge

merge:
  %setc.n = phi i64 [ %setc.dec, %do ], [ %setc, %skip ]
  ; after unite a,b must be connected
  %cab = call i32 @universe_ds_unionfind_connected(ptr %u, i64 %a, i64 %b)
  %bad.cab = icmp ne i32 %cab, 1
  %inc.cab = zext i1 %bad.cab to i64
  ; independent query pair vs shadow
  %r2 = call i64 @ut_rand(ptr %state)
  %c = urem i64 %r2, %n
  %r3 = call i64 @ut_rand(ptr %state)
  %d = urem i64 %r3, %n
  %lc.p = getelementptr inbounds i64, ptr %label, i64 %c
  %lc = load i64, ptr %lc.p, align 8
  %ld.p = getelementptr inbounds i64, ptr %label, i64 %d
  %ld = load i64, ptr %ld.p, align 8
  %want.eq = icmp eq i64 %lc, %ld
  %want = zext i1 %want.eq to i32
  %qc = call i32 @universe_ds_unionfind_connected(ptr %u, i64 %c, i64 %d)
  %bad.q = icmp ne i32 %qc, %want
  %inc.q = zext i1 %bad.q to i64
  br label %after

after:
  %v1 = add i64 %viol, %inc.rc
  %v2 = add i64 %v1, %inc.cab
  %viol.n = add i64 %v2, %inc.q
  %k.n = add nuw i64 %k, 1
  %kmore = icmp ult i64 %k.n, 10000
  br i1 %kmore, label %loop, label %fin

fin:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.rand)
  %cs = call i64 @universe_ds_unionfind_count_sets(ptr %u)
  call void @ut_check_eq(i64 %cs, i64 %setc.n, ptr @m.cnt)
  call void @universe_ds_unionfind_destroy(ptr %u)
  call void @free(ptr %label)
  ret void
}

define internal void @bench() {
entry:
  %state = alloca i64, align 8
  %vsink = alloca i64, align 8
  %n = add i64 0, 100000
  br label %rep.head
rep.head:
  %rep = phi i64 [ 0, %entry ], [ %rep.n, %rep.cont ]
  store i64 12345678901234567, ptr %state, align 8
  ; unite is destructive (structure collapses to one set) — re-create each rep
  %u = call ptr @universe_ds_unionfind_create(i64 %n)
  %rt0 = call double @ut_now_sec()
  br label %loop
loop:
  %k = phi i64 [ 0, %rep.head ], [ %k.n, %loop ]
  %acc = phi i64 [ 0, %rep.head ], [ %acc.n, %loop ]
  %r0 = call i64 @ut_rand(ptr %state)
  %a = urem i64 %r0, %n
  %r1 = call i64 @ut_rand(ptr %state)
  %b = urem i64 %r1, %n
  call i32 @universe_ds_unionfind_unite(ptr %u, i64 %a, i64 %b)
  %c = call i32 @universe_ds_unionfind_connected(ptr %u, i64 %a, i64 %b)
  %ce = zext i32 %c to i64
  %acc.n = add i64 %acc, %ce
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 1000000
  br i1 %more, label %loop, label %rep.done
rep.done:
  %rt1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr %vsink, align 8
  call void @universe_ds_unionfind_destroy(ptr %u)
  %elapsed = fsub double %rt1, %rt0
  %warm = icmp eq i64 %rep, 0
  br i1 %warm, label %rep.cont, label %rep.store
rep.store:
  %sidx = sub i64 %rep, 1
  %sp = getelementptr inbounds double, ptr @uf.bench.samp, i64 %sidx
  store double %elapsed, ptr %sp, align 8
  br label %rep.cont
rep.cont:
  %rep.n = add nuw i64 %rep, 1
  %repmore = icmp ult i64 %rep.n, 17
  br i1 %repmore, label %rep.head, label %rep.report
rep.report:
  call void @ut_report_dist(ptr @uf.bench.samp, i64 16, i64 2000000, ptr @lbl.uf.bench)
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
