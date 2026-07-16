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

; Tests for universe_ds_stack: LIFO order across many growths, peek,
; empty errors, large elements, --bench.

declare ptr @universe_ds_stack_create(i64, i64)
declare i32 @universe_ds_stack_push(ptr, ptr)
declare i32 @universe_ds_stack_pop(ptr, ptr)
declare i32 @universe_ds_stack_peek(ptr, ptr)
declare i64 @universe_ds_stack_count(ptr)
declare i64 @universe_ds_stack_capacity(ptr)
declare void @universe_ds_stack_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.lifo   = private unnamed_addr constant [26 x i8] c"LIFO across 65536 + grows\00"
@m.count  = private unnamed_addr constant [15 x i8] c"count tracks  \00"
@m.peek   = private unnamed_addr constant [20 x i8] c"peek == top nondest\00"
@m.empty  = private unnamed_addr constant [15 x i8] c"pop empty -> 4\00"
@m.errs   = private unnamed_addr constant [16 x i8] c"null args -> 1 \00"
@m.big    = private unnamed_addr constant [20 x i8] c"64B elements intact\00"
@lbl.stack = private unnamed_addr constant [28 x i8] c"stack push+pop (2M ops/rep)\00"
@stack.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_lifo() {
entry:                                       ; start tiny: force many growths
  %st = call ptr @universe_ds_stack_create(i64 8, i64 8)
  %v = alloca i64, align 8
  br label %push

push:
  %i = phi i64 [ 0, %entry ], [ %i.n, %push ]
  %errs = phi i64 [ 0, %entry ], [ %errs.n, %push ]
  store i64 %i, ptr %v, align 8
  %rc = call i32 @universe_ds_stack_push(ptr %st, ptr nonnull %v)
  %bad = icmp ne i32 %rc, 0
  %e.inc = zext i1 %bad to i64
  %errs.n = add nuw i64 %errs, %e.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 65536
  br i1 %more, label %push, label %pushed

pushed:
  %count = call i64 @universe_ds_stack_count(ptr %st)
  call void @ut_check_eq(i64 %count, i64 65536, ptr @m.count)

  ; peek should not remove
  %rc.peek = call i32 @universe_ds_stack_peek(ptr %st, ptr nonnull %v)
  %top = load i64, ptr %v, align 8
  %count2 = call i64 @universe_ds_stack_count(ptr %st)
  %peek.rc = icmp eq i32 %rc.peek, 0
  %peek.val = icmp eq i64 %top, 65535
  %peek.cnt = icmp eq i64 %count2, 65536
  %pk1 = and i1 %peek.rc, %peek.val
  %pk = and i1 %pk1, %peek.cnt
  call void @ut_check(i1 %pk, ptr @m.peek)
  br label %pop

pop:
  %j = phi i64 [ 0, %pushed ], [ %j.n, %pop ]
  %viol = phi i64 [ %errs.n, %pushed ], [ %viol.n, %pop ]
  %rc2 = call i32 @universe_ds_stack_pop(ptr %st, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %want = sub nuw i64 65535, %j
  %bad1 = icmp ne i32 %rc2, 0
  %bad2 = icmp ne i64 %got, %want
  %anybad = or i1 %bad1, %bad2
  %v.inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 65536
  br i1 %more2, label %pop, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.lifo)
  %rc3 = call i32 @universe_ds_stack_pop(ptr %st, ptr nonnull %v)
  %rc3.w = zext i32 %rc3 to i64
  call void @ut_check_eq(i64 %rc3.w, i64 4, ptr @m.empty)
  call void @universe_ds_stack_destroy(ptr %st)
  ret void
}

define internal void @test_big_elems() {
entry:                                       ; 64-byte elements survive growth
  %st = call ptr @universe_ds_stack_create(i64 64, i64 4)
  %buf = alloca [8 x i64], align 16
  %seed = alloca i64, align 8
  store i64 5, ptr %seed, align 8
  br label %push

push:
  %i = phi i64 [ 0, %entry ], [ %i.n, %dopush ]
  ; element = 8 copies of i
  br label %fill

fill:
  %k = phi i64 [ 0, %push ], [ %k.n, %fill ]
  %slot = getelementptr inbounds nuw [8 x i64], ptr %buf, i64 0, i64 %k
  store i64 %i, ptr %slot, align 8
  %k.n = add nuw nsw i64 %k, 1
  %kmore = icmp ult i64 %k.n, 8
  br i1 %kmore, label %fill, label %dopush

dopush:
  %rc = call i32 @universe_ds_stack_push(ptr %st, ptr nonnull %buf)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100
  br i1 %more, label %push, label %pop

pop:
  %j = phi i64 [ 0, %dopush ], [ %j.n, %chk.done ]
  %viol = phi i64 [ 0, %dopush ], [ %viol.n2, %chk.done ]
  %rc2 = call i32 @universe_ds_stack_pop(ptr %st, ptr nonnull %buf)
  %want = sub nuw i64 99, %j
  br label %chk

chk:                                         ; all 8 lanes must equal want
  %m = phi i64 [ 0, %pop ], [ %m.n, %chk ]
  %viol.in = phi i64 [ %viol, %pop ], [ %viol.n, %chk ]
  %slot2 = getelementptr inbounds nuw [8 x i64], ptr %buf, i64 0, i64 %m
  %got = load i64, ptr %slot2, align 8
  %bad = icmp ne i64 %got, %want
  %v.inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol.in, %v.inc
  %m.n = add nuw nsw i64 %m, 1
  %mmore = icmp ult i64 %m.n, 8
  br i1 %mmore, label %chk, label %chk.done

chk.done:
  %viol.n2 = phi i64 [ %viol.n, %chk ]
  %j.n = add nuw nsw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 100
  br i1 %jmore, label %pop, label %done

done:
  call void @ut_check_eq(i64 %viol.n2, i64 0, ptr @m.big)
  call void @universe_ds_stack_destroy(ptr %st)
  ret void
}

define internal void @test_errors() {
entry:
  %v = alloca i64, align 8
  %e1 = call i32 @universe_ds_stack_push(ptr null, ptr nonnull %v)
  %st = call ptr @universe_ds_stack_create(i64 8, i64 8)
  %e2 = call i32 @universe_ds_stack_push(ptr %st, ptr null)
  %z = call ptr @universe_ds_stack_create(i64 0, i64 8)
  %z.null = icmp eq ptr %z, null
  %e1.ok = icmp eq i32 %e1, 1
  %e2.ok = icmp eq i32 %e2, 1
  %a1 = and i1 %e1.ok, %e2.ok
  %all = and i1 %a1, %z.null
  call void @ut_check(i1 %all, ptr @m.errs)
  call void @universe_ds_stack_destroy(ptr %st)
  call void @universe_ds_stack_destroy(ptr null)
  ret void
}

define internal void @bench() {
entry:
  %st = call ptr @universe_ds_stack_create(i64 8, i64 1024)
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  store i64 %i, ptr %v, align 8
  %rc1 = call i32 @universe_ds_stack_push(ptr %st, ptr nonnull %v)
  %rc2 = call i32 @universe_ds_stack_pop(ptr %st, ptr nonnull %v)
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
  %sp = getelementptr inbounds double, ptr @stack.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_stack_destroy(ptr %st)
  call void @ut_report_dist(ptr @stack.samp, i64 16, i64 2000000, ptr @lbl.stack)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_lifo()
  call void @test_big_elems()
  call void @test_errors()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
