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

; Tests for universe_ds_slist: FIFO via push_back, LIFO via push_front,
; pop_front / remove_after semantics, chunk growth, --bench.

declare ptr @universe_ds_slist_create(i64)
declare ptr @universe_ds_slist_push_front(ptr, ptr)
declare ptr @universe_ds_slist_push_back(ptr, ptr)
declare ptr @universe_ds_slist_insert_after(ptr, ptr, ptr)
declare i32 @universe_ds_slist_pop_front(ptr, ptr)
declare i32 @universe_ds_slist_remove_after(ptr, ptr, ptr)
declare ptr @universe_ds_slist_first(ptr)
declare ptr @universe_ds_slist_next(ptr)
declare ptr @universe_ds_slist_data(ptr)
declare i64 @universe_ds_slist_count(ptr)
declare void @universe_ds_slist_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.fifo   = private unnamed_addr constant [26 x i8] c"1000 push_back pops FIFO \00"
@m.lifo   = private unnamed_addr constant [26 x i8] c"100 push_front pops LIFO \00"
@m.rmaft  = private unnamed_addr constant [22 x i8] c"remove_after mid+tail\00"
@m.empty  = private unnamed_addr constant [17 x i8] c"pop empty -> 4  \00"
@lbl.slist = private unnamed_addr constant [34 x i8] c"slist push_front+pop (2M ops/rep)\00"
@slist.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_fifo() {
entry:
  %l = call ptr @universe_ds_slist_create(i64 8)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %node = call ptr @universe_ds_slist_push_back(ptr %l, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %fill, label %drain

drain:
  %j = phi i64 [ 0, %fill ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %fill ], [ %viol.n, %drain ]
  %rc = call i32 @universe_ds_slist_pop_front(ptr %l, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %bad1 = icmp ne i32 %rc, 0
  %bad2 = icmp ne i64 %got, %j
  %anybad = or i1 %bad1, %bad2
  %inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000
  br i1 %more2, label %drain, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.fifo)
  %rc2 = call i32 @universe_ds_slist_pop_front(ptr %l, ptr nonnull %v)
  %rc2.w = zext i32 %rc2 to i64
  call void @ut_check_eq(i64 %rc2.w, i64 4, ptr @m.empty)
  call void @universe_ds_slist_destroy(ptr %l)
  ret void
}

define internal void @test_lifo() {
entry:
  %l = call ptr @universe_ds_slist_create(i64 8)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %node = call ptr @universe_ds_slist_push_front(ptr %l, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100
  br i1 %more, label %fill, label %drain

drain:
  %j = phi i64 [ 0, %fill ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %fill ], [ %viol.n, %drain ]
  %rc = call i32 @universe_ds_slist_pop_front(ptr %l, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %want = sub nuw i64 99, %j
  %bad1 = icmp ne i32 %rc, 0
  %bad2 = icmp ne i64 %got, %want
  %anybad = or i1 %bad1, %bad2
  %inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 100
  br i1 %more2, label %drain, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.lifo)
  call void @universe_ds_slist_destroy(ptr %l)
  ret void
}

define internal void @test_remove_after() {
entry:
  ; [1,2,3]: remove_after(n1) kills 2; remove_after(n1) kills 3 (tail);
  ; remove_after(n1) -> 5 (no successor)
  %l = call ptr @universe_ds_slist_create(i64 8)
  %v = alloca i64, align 8
  store i64 1, ptr %v, align 8
  %n1 = call ptr @universe_ds_slist_push_back(ptr %l, ptr nonnull %v)
  store i64 2, ptr %v, align 8
  %n2 = call ptr @universe_ds_slist_push_back(ptr %l, ptr nonnull %v)
  store i64 3, ptr %v, align 8
  %n3 = call ptr @universe_ds_slist_push_back(ptr %l, ptr nonnull %v)

  store i64 0, ptr %v, align 8
  %r1 = call i32 @universe_ds_slist_remove_after(ptr %l, ptr %n1, ptr nonnull %v)
  %rv1 = load i64, ptr %v, align 8
  %r2 = call i32 @universe_ds_slist_remove_after(ptr %l, ptr %n1, ptr nonnull %v)
  %rv2 = load i64, ptr %v, align 8
  %r3 = call i32 @universe_ds_slist_remove_after(ptr %l, ptr %n1, ptr nonnull %v)
  %cnt = call i64 @universe_ds_slist_count(ptr %l)
  %ok1 = icmp eq i64 %rv1, 2
  %ok2 = icmp eq i64 %rv2, 3
  %ok3 = icmp eq i32 %r3, 5
  %ok4 = icmp eq i64 %cnt, 1
  %a1 = and i1 %ok1, %ok2
  %a2 = and i1 %a1, %ok3
  %a3 = and i1 %a2, %ok4
  call void @ut_check(i1 %a3, ptr @m.rmaft)
  call void @universe_ds_slist_destroy(ptr %l)
  ret void
}

define internal void @bench() {
entry:
  %l = call ptr @universe_ds_slist_create(i64 8)
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  store i64 %i, ptr %v, align 8
  %node = call ptr @universe_ds_slist_push_front(ptr %l, ptr nonnull %v)
  %rc = call i32 @universe_ds_slist_pop_front(ptr %l, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %loop, label %rep.done

rep.done:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %rep.i, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %sidx = sub i64 %rep.i, 1
  %sp = getelementptr inbounds double, ptr @slist.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_slist_destroy(ptr %l)
  call void @ut_report_dist(ptr @slist.samp, i64 16, i64 2000000, ptr @lbl.slist)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_fifo()
  call void @test_lifo()
  call void @test_remove_after()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
