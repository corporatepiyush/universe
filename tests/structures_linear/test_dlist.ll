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

; Tests for universe_ds_dlist: order across chunk growth (>256 nodes),
; insert_after, remove head/mid/tail with node recycling, zero-copy data
; view, forward+backward walks, --bench.

declare ptr @universe_ds_dlist_create(i64)
declare ptr @universe_ds_dlist_push_front(ptr, ptr)
declare ptr @universe_ds_dlist_push_back(ptr, ptr)
declare ptr @universe_ds_dlist_insert_after(ptr, ptr, ptr)
declare i32 @universe_ds_dlist_remove(ptr, ptr, ptr)
declare ptr @universe_ds_dlist_first(ptr)
declare ptr @universe_ds_dlist_last(ptr)
declare ptr @universe_ds_dlist_next(ptr)
declare ptr @universe_ds_dlist_prev(ptr)
declare ptr @universe_ds_dlist_data(ptr)
declare i64 @universe_ds_dlist_count(ptr)
declare void @universe_ds_dlist_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.fwd    = private unnamed_addr constant [25 x i8] c"1000 fwd walk ascending \00"
@m.bwd    = private unnamed_addr constant [25 x i8] c"1000 bwd walk descending\00"
@m.insaft = private unnamed_addr constant [20 x i8] c"insert_after splice\00"
@m.rm     = private unnamed_addr constant [25 x i8] c"remove head/mid/tail ok \00"
@m.recyc  = private unnamed_addr constant [23 x i8] c"nodes recycle after rm\00"
@m.zc     = private unnamed_addr constant [22 x i8] c"data view is in-place\00"
@lbl.dlist = private unnamed_addr constant [34 x i8] c"dlist push_back+walk (2M ops/rep)\00"
@dlist.samp = internal global [16 x double] zeroinitializer, align 8
@dlist.sink = internal global i64 0, align 8

define internal void @test_walks() {
entry:
  %l = call ptr @universe_ds_dlist_create(i64 8)
  %v = alloca i64, align 8
  br label %fill

fill:                                        ; 1000 > 256: three chunk growths
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %node = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %fill, label %fwd.pre

fwd.pre:
  %first = call ptr @universe_ds_dlist_first(ptr %l)
  br label %fwd

fwd:
  %n = phi ptr [ %first, %fwd.pre ], [ %nn, %fwd.body ]
  %want = phi i64 [ 0, %fwd.pre ], [ %want.n, %fwd.body ]
  %viol = phi i64 [ 0, %fwd.pre ], [ %viol.n, %fwd.body ]
  %n.null = icmp eq ptr %n, null
  br i1 %n.null, label %fwd.done, label %fwd.body

fwd.body:
  %d = call ptr @universe_ds_dlist_data(ptr nonnull %n)
  %got = load i64, ptr %d, align 8
  %bad = icmp ne i64 %got, %want
  %inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %inc
  %want.n = add nuw i64 %want, 1
  %nn = call ptr @universe_ds_dlist_next(ptr nonnull %n)
  br label %fwd

fwd.done:
  %fwd.complete = icmp eq i64 %want, 1000
  %fwd.clean = icmp eq i64 %viol, 0
  %f.ok = and i1 %fwd.complete, %fwd.clean
  call void @ut_check(i1 %f.ok, ptr @m.fwd)
  %last = call ptr @universe_ds_dlist_last(ptr %l)
  br label %bwd

bwd:
  %b = phi ptr [ %last, %fwd.done ], [ %bp, %bwd.body ]
  %bwant = phi i64 [ 999, %fwd.done ], [ %bwant.n, %bwd.body ]
  %bviol = phi i64 [ 0, %fwd.done ], [ %bviol.n, %bwd.body ]
  %b.null = icmp eq ptr %b, null
  br i1 %b.null, label %bwd.done, label %bwd.body

bwd.body:
  %bd = call ptr @universe_ds_dlist_data(ptr nonnull %b)
  %bgot = load i64, ptr %bd, align 8
  %bbad = icmp ne i64 %bgot, %bwant
  %binc = zext i1 %bbad to i64
  %bviol.n = add nuw i64 %bviol, %binc
  %bwant.n = add i64 %bwant, -1
  %bp = call ptr @universe_ds_dlist_prev(ptr nonnull %b)
  br label %bwd

bwd.done:
  %b.complete = icmp eq i64 %bwant, -1
  %b.clean = icmp eq i64 %bviol, 0
  %b.ok = and i1 %b.complete, %b.clean
  call void @ut_check(i1 %b.ok, ptr @m.bwd)
  call void @universe_ds_dlist_destroy(ptr %l)
  ret void
}

define internal void @test_splice_remove() {
entry:
  %l = call ptr @universe_ds_dlist_create(i64 8)
  %v = alloca i64, align 8
  ; [1] -> insert_after(1, 2) -> [1,2] -> push_back 3 -> [1,2,3]
  store i64 1, ptr %v, align 8
  %n1 = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  store i64 2, ptr %v, align 8
  %n2 = call ptr @universe_ds_dlist_insert_after(ptr %l, ptr %n1, ptr nonnull %v)
  store i64 3, ptr %v, align 8
  %n3 = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  ; verify 1->2->3
  %d1 = call ptr @universe_ds_dlist_data(ptr %n1)
  %n1next = call ptr @universe_ds_dlist_next(ptr %n1)
  %same2 = icmp eq ptr %n1next, %n2
  %n2next = call ptr @universe_ds_dlist_next(ptr %n2)
  %same3 = icmp eq ptr %n2next, %n3
  %sp = and i1 %same2, %same3
  call void @ut_check(i1 %sp, ptr @m.insaft)

  ; zero-copy: writing through data view changes the element
  store i64 42, ptr %d1, align 8
  %d1b = call ptr @universe_ds_dlist_data(ptr %n1)
  %zc = load i64, ptr %d1b, align 8
  %zc.ok = icmp eq i64 %zc, 42
  call void @ut_check(i1 %zc.ok, ptr @m.zc)

  ; remove mid (n2, out captured), then head (n1), then tail (n3)
  store i64 0, ptr %v, align 8
  %r1 = call i32 @universe_ds_dlist_remove(ptr %l, ptr %n2, ptr nonnull %v)
  %rv = load i64, ptr %v, align 8
  %r2 = call i32 @universe_ds_dlist_remove(ptr %l, ptr %n1, ptr null)
  %r3 = call i32 @universe_ds_dlist_remove(ptr %l, ptr %n3, ptr null)
  %cnt = call i64 @universe_ds_dlist_count(ptr %l)
  %first = call ptr @universe_ds_dlist_first(ptr %l)
  %rm1 = icmp eq i64 %rv, 2
  %rm2 = icmp eq i64 %cnt, 0
  %rm3 = icmp eq ptr %first, null
  %ra = and i1 %rm1, %rm2
  %rb = and i1 %ra, %rm3
  call void @ut_check(i1 %rb, ptr @m.rm)

  ; recycling: removed nodes must be reused (no new chunk needed): push 3
  store i64 9, ptr %v, align 8
  %m1 = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  %m2 = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  %m3 = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  ; the recycled nodes are exactly {n1,n2,n3} in some order — check set
  %e11 = icmp eq ptr %m1, %n1
  %e12 = icmp eq ptr %m1, %n2
  %e13 = icmp eq ptr %m1, %n3
  %o1a = or i1 %e11, %e12
  %o1 = or i1 %o1a, %e13
  %e21 = icmp eq ptr %m2, %n1
  %e22 = icmp eq ptr %m2, %n2
  %e23 = icmp eq ptr %m2, %n3
  %o2a = or i1 %e21, %e22
  %o2 = or i1 %o2a, %e23
  %e31 = icmp eq ptr %m3, %n1
  %e32 = icmp eq ptr %m3, %n2
  %e33 = icmp eq ptr %m3, %n3
  %o3a = or i1 %e31, %e32
  %o3 = or i1 %o3a, %e33
  %all1 = and i1 %o1, %o2
  %all = and i1 %all1, %o3
  call void @ut_check(i1 %all, ptr @m.recyc)
  call void @universe_ds_dlist_destroy(ptr %l)
  ret void
}

define internal void @bench() {
entry:
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %l = call ptr @universe_ds_dlist_create(i64 8)
  %t0 = call double @ut_now_sec()
  br label %fill

fill:
  %i = phi i64 [ 0, %rep ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %node = call ptr @universe_ds_dlist_push_back(ptr %l, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %fill, label %walk.pre

walk.pre:
  %first = call ptr @universe_ds_dlist_first(ptr %l)
  br label %walk

walk:
  %n = phi ptr [ %first, %walk.pre ], [ %nn, %walk.body ]
  %sum = phi i64 [ 0, %walk.pre ], [ %sum.n, %walk.body ]
  %n.null = icmp eq ptr %n, null
  br i1 %n.null, label %rep.done, label %walk.body

walk.body:
  %d = call ptr @universe_ds_dlist_data(ptr nonnull %n)
  %x = load i64, ptr %d, align 8
  %sum.n = add i64 %sum, %x
  %nn = call ptr @universe_ds_dlist_next(ptr nonnull %n)
  br label %walk

rep.done:
  %t1 = call double @ut_now_sec()
  call void @universe_ds_dlist_destroy(ptr %l)
  store volatile i64 %sum, ptr @dlist.sink, align 8
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %rep.i, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %sidx = sub i64 %rep.i, 1
  %sp = getelementptr inbounds double, ptr @dlist.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @ut_report_dist(ptr @dlist.samp, i64 16, i64 2000000, ptr @lbl.dlist)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_walks()
  call void @test_splice_remove()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
