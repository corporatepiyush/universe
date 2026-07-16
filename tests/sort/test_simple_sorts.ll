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

; Combined tests for universe_sort_{bubble, selection, shell}: each is
; cross-checked against libc qsort on fixed-seed random data plus
; presorted/reverse/all-equal edges. Bubble also gets a stability probe.

declare i32 @universe_sort_bubble(ptr, i64, i64, ptr)
declare i32 @universe_sort_selection(ptr, i64, i64, ptr)
declare i32 @universe_sort_shell(ptr, i64, i64, ptr)
declare i32 @universe_sort_merge(ptr, i64, i64, ptr)
declare i32 @universe_sort_quick(ptr, i64, i64, ptr)

declare void @qsort(ptr, i64, i64, ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)

@g.master = internal global [2048 x i32] zeroinitializer, align 16
@g.work   = internal global [2048 x i32] zeroinitializer, align 16
@g.ref    = internal global [2048 x i32] zeroinitializer, align 16
@g.pairs  = internal global [512 x i64] zeroinitializer, align 16

@m.bubble  = private unnamed_addr constant [19 x i8] c"bubble match qsort\00"
@m.select  = private unnamed_addr constant [22 x i8] c"selection match qsort\00"
@m.shell   = private unnamed_addr constant [18 x i8] c"shell match qsort\00"
@m.b.edge  = private unnamed_addr constant [13 x i8] c"bubble edges\00"
@m.s.edge  = private unnamed_addr constant [16 x i8] c"selection edges\00"
@m.h.edge  = private unnamed_addr constant [12 x i8] c"shell edges\00"
@m.b.stab  = private unnamed_addr constant [14 x i8] c"bubble stable\00"
@m.merge   = private unnamed_addr constant [18 x i8] c"merge match qsort\00"
@m.m.edge  = private unnamed_addr constant [12 x i8] c"merge edges\00"
@m.m.stab  = private unnamed_addr constant [13 x i8] c"merge stable\00"
@m.quick   = private unnamed_addr constant [18 x i8] c"quick match qsort\00"
@m.q.edge  = private unnamed_addr constant [12 x i8] c"quick edges\00"
@m.q.equal = private unnamed_addr constant [22 x i8] c"quick all-equal safe \00"
@m.errs    = private unnamed_addr constant [16 x i8] c"null args all 1\00"

define i32 @cmp_i32(ptr %a, ptr %b) {
entry:
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %gt = icmp sgt i32 %x, %y
  %lt = icmp slt i32 %x, %y
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

define i32 @cmp_pair_key(ptr %a, ptr %b) {
entry:
  %x = load i64, ptr %a, align 8
  %y = load i64, ptr %b, align 8
  %kx = and i64 %x, 4294967295
  %ky = and i64 %y, 4294967295
  %gt = icmp ugt i64 %kx, %ky
  %lt = icmp ult i64 %kx, %ky
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

; run %fn over a fresh copy of master, compare against qsort'd ref
define internal void @check_against_qsort(ptr %fn, ptr %msg) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 8192, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.ref, ptr align 16 @g.master, i64 8192, i1 false)
  %rc = call i32 %fn(ptr @g.work, i64 2048, i64 4, ptr @cmp_i32)
  call void @qsort(ptr @g.ref, i64 2048, i64 4, ptr @cmp_i32)
  %mc = call i32 @memcmp(ptr @g.work, ptr @g.ref, i64 8192)
  %rc.ok = icmp eq i32 %rc, 0
  %mc.ok = icmp eq i32 %mc, 0
  %ok = and i1 %rc.ok, %mc.ok
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

; presorted + reverse + all-equal for %fn; returns violations
define internal void @check_edges(ptr %fn, ptr %msg) {
entry:
  br label %fill.asc

fill.asc:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill.asc ]
  %v = trunc i64 %i to i32
  %p = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 300
  br i1 %c, label %fill.asc, label %sort.asc

sort.asc:
  %rc0 = call i32 %fn(ptr @g.work, i64 300, i64 4, ptr @cmp_i32)
  br label %chk.asc

chk.asc:
  %j = phi i64 [ 0, %sort.asc ], [ %j.n, %chk.asc ]
  %bad = phi i64 [ 0, %sort.asc ], [ %bad.n, %chk.asc ]
  %p2 = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %j
  %got = load i32, ptr %p2, align 4
  %want = trunc i64 %j to i32
  %ne = icmp ne i32 %got, %want
  %inc = zext i1 %ne to i64
  %bad.n = add nuw i64 %bad, %inc
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 300
  br i1 %c2, label %chk.asc, label %fill.desc

fill.desc:
  %k = phi i64 [ 0, %chk.asc ], [ %k.n, %fill.desc ]
  %dv64 = sub nsw i64 299, %k
  %dv = trunc i64 %dv64 to i32
  %p3 = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %k
  store i32 %dv, ptr %p3, align 4
  %k.n = add nuw nsw i64 %k, 1
  %c3 = icmp ult i64 %k.n, 300
  br i1 %c3, label %fill.desc, label %sort.desc

sort.desc:
  %rc1 = call i32 %fn(ptr @g.work, i64 300, i64 4, ptr @cmp_i32)
  br label %chk.desc

chk.desc:
  %m = phi i64 [ 0, %sort.desc ], [ %m.n, %chk.desc ]
  %bad2 = phi i64 [ %bad.n, %sort.desc ], [ %bad2.n, %chk.desc ]
  %p4 = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %m
  %got2 = load i32, ptr %p4, align 4
  %want2 = trunc i64 %m to i32
  %ne2 = icmp ne i32 %got2, %want2
  %inc2 = zext i1 %ne2 to i64
  %bad2.n = add nuw i64 %bad2, %inc2
  %m.n = add nuw nsw i64 %m, 1
  %c4 = icmp ult i64 %m.n, 300
  br i1 %c4, label %chk.desc, label %report

report:
  call void @ut_check_eq(i64 %bad2.n, i64 0, ptr %msg)
  ret void
}

; stability probe over key|seq pairs for any sorter claiming stability
define internal void @check_stability(ptr %fn, i64 %seedval, ptr %msg) {
entry:
  %seed = alloca i64, align 8
  store i64 %seedval, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %key = urem i64 %r, 8
  %seq = shl nuw i64 %i, 32
  %pair = or disjoint i64 %key, %seq
  %p = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %i
  store i64 %pair, ptr %p, align 8
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 512
  br i1 %c, label %fill, label %run

run:
  %rc = call i32 %fn(ptr @g.pairs, i64 512, i64 8, ptr @cmp_pair_key)
  br label %chk

chk:
  %j = phi i64 [ 1, %run ], [ %j.n, %chk ]
  %bad = phi i64 [ 0, %run ], [ %bad.n, %chk ]
  %j.prev = add nsw i64 %j, -1
  %pp = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %j.prev
  %cp = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %j
  %prev = load i64, ptr %pp, align 8
  %cur = load i64, ptr %cp, align 8
  %pk = and i64 %prev, 4294967295
  %ck = and i64 %cur, 4294967295
  %ord.bad = icmp ugt i64 %pk, %ck
  %same = icmp eq i64 %pk, %ck
  %ps = lshr i64 %prev, 32
  %cs = lshr i64 %cur, 32
  %seq.rev = icmp uge i64 %ps, %cs
  %stab.bad = and i1 %same, %seq.rev
  %either = or i1 %ord.bad, %stab.bad
  %inc = zext i1 %either to i64
  %bad.n = add nuw i64 %bad, %inc
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 512
  br i1 %c2, label %chk, label %rep

rep:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr %msg)
  ret void
}

define internal void @test_bubble_stability() {
entry:
  %seed = alloca i64, align 8
  store i64 7, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %key = urem i64 %r, 8
  %seq = shl nuw i64 %i, 32
  %pair = or disjoint i64 %key, %seq
  %p = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %i
  store i64 %pair, ptr %p, align 8
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 512
  br i1 %c, label %fill, label %run

run:
  %rc = call i32 @universe_sort_bubble(ptr @g.pairs, i64 512, i64 8, ptr @cmp_pair_key)
  br label %chk

chk:
  %j = phi i64 [ 1, %run ], [ %j.n, %chk ]
  %bad = phi i64 [ 0, %run ], [ %bad.n, %chk ]
  %j.prev = add nsw i64 %j, -1
  %pp = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %j.prev
  %cp = getelementptr inbounds nuw [512 x i64], ptr @g.pairs, i64 0, i64 %j
  %prev = load i64, ptr %pp, align 8
  %cur = load i64, ptr %cp, align 8
  %pk = and i64 %prev, 4294967295
  %ck = and i64 %cur, 4294967295
  %ord.bad = icmp ugt i64 %pk, %ck
  %same = icmp eq i64 %pk, %ck
  %ps = lshr i64 %prev, 32
  %cs = lshr i64 %cur, 32
  %seq.rev = icmp uge i64 %ps, %cs
  %stab.bad = and i1 %same, %seq.rev
  %either = or i1 %ord.bad, %stab.bad
  %inc = zext i1 %either to i64
  %bad.n = add nuw i64 %bad, %inc
  %j.n = add nuw nsw i64 %j, 1
  %c2 = icmp ult i64 %j.n, 512
  br i1 %c2, label %chk, label %rep

rep:
  call void @ut_check_eq(i64 %bad.n, i64 0, ptr @m.b.stab)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; fixed-seed random master
  %seed = alloca i64, align 8
  store i64 42, ptr %seed, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %v = trunc i64 %r to i32
  %p = getelementptr inbounds nuw [2048 x i32], ptr @g.master, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.n, 2048
  br i1 %c, label %fill, label %run

run:
  call void @check_against_qsort(ptr @universe_sort_bubble, ptr @m.bubble)
  call void @check_against_qsort(ptr @universe_sort_selection, ptr @m.select)
  call void @check_against_qsort(ptr @universe_sort_shell, ptr @m.shell)
  call void @check_against_qsort(ptr @universe_sort_merge, ptr @m.merge)
  call void @check_edges(ptr @universe_sort_bubble, ptr @m.b.edge)
  call void @check_edges(ptr @universe_sort_selection, ptr @m.s.edge)
  call void @check_edges(ptr @universe_sort_shell, ptr @m.h.edge)
  call void @check_edges(ptr @universe_sort_merge, ptr @m.m.edge)
  call void @test_bubble_stability()
  call void @check_stability(ptr @universe_sort_merge, i64 13, ptr @m.m.stab)
  call void @check_against_qsort(ptr @universe_sort_quick, ptr @m.quick)
  call void @check_edges(ptr @universe_sort_quick, ptr @m.q.edge)
  br label %all.equal

all.equal:                                  ; quick's pathological input
  %q = phi i64 [ 0, %run ], [ %q.n, %all.equal ]
  %qp = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %q
  store i32 7, ptr %qp, align 4
  %q.n = add nuw nsw i64 %q, 1
  %qc = icmp ult i64 %q.n, 2048
  br i1 %qc, label %all.equal, label %all.equal.run

all.equal.run:
  %qrc = call i32 @universe_sort_quick(ptr @g.work, i64 2048, i64 4, ptr @cmp_i32)
  br label %all.equal.chk

all.equal.chk:
  %e = phi i64 [ 0, %all.equal.run ], [ %e.n, %all.equal.chk ]
  %ebad = phi i64 [ 0, %all.equal.run ], [ %ebad.n, %all.equal.chk ]
  %ep = getelementptr inbounds nuw [2048 x i32], ptr @g.work, i64 0, i64 %e
  %ev = load i32, ptr %ep, align 4
  %ene = icmp ne i32 %ev, 7
  %einc = zext i1 %ene to i64
  %ebad.n = add nuw i64 %ebad, %einc
  %e.n = add nuw nsw i64 %e, 1
  %ec = icmp ult i64 %e.n, 2048
  br i1 %ec, label %all.equal.chk, label %all.equal.rep

all.equal.rep:
  %qrc.bad = icmp ne i32 %qrc, 0
  %qrc.inc = zext i1 %qrc.bad to i64
  %etotal = add nuw i64 %ebad.n, %qrc.inc
  call void @ut_check_eq(i64 %etotal, i64 0, ptr @m.q.equal)

  %buf = alloca i64, align 8
  %e1 = call i32 @universe_sort_bubble(ptr null, i64 2, i64 8, ptr @cmp_i32)
  %e2 = call i32 @universe_sort_selection(ptr null, i64 2, i64 8, ptr @cmp_i32)
  %e3 = call i32 @universe_sort_shell(ptr nonnull %buf, i64 2, i64 8, ptr null)
  %s1 = add nuw i32 %e1, %e2
  %s = add nuw i32 %s1, %e3
  %s.w = zext i32 %s to i64
  call void @ut_check_eq(i64 %s.w, i64 3, ptr @m.errs)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
