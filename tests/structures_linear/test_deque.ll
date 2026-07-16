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

; Tests for universe_ds_deque: both-end ops, wrap below zero (push_front on
; fresh deque), growth while wrapped, palindrome drain, errors, --bench.

declare ptr @universe_ds_deque_create(i64, i64)
declare i32 @universe_ds_deque_push_front(ptr, ptr)
declare i32 @universe_ds_deque_push_back(ptr, ptr)
declare i32 @universe_ds_deque_pop_front(ptr, ptr)
declare i32 @universe_ds_deque_pop_back(ptr, ptr)
declare i32 @universe_ds_deque_peek_front(ptr, ptr)
declare i32 @universe_ds_deque_peek_back(ptr, ptr)
declare i64 @universe_ds_deque_count(ptr)
declare void @universe_ds_deque_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.mirror = private unnamed_addr constant [26 x i8] c"mirror fill drains sorted\00"
@m.wrap   = private unnamed_addr constant [25 x i8] c"head wraps below zero ok\00"
@m.peek   = private unnamed_addr constant [16 x i8] c"peeks both ends\00"
@m.empty  = private unnamed_addr constant [16 x i8] c"empty pops -> 4\00"
@m.grow   = private unnamed_addr constant [26 x i8] c"grows wrapped, order kept\00"
@lbl.deque = private unnamed_addr constant [30 x i8] c"deque mixed push/pop (4M/rep)\00"
@deque.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_mirror() {
entry:
  ; push_front 4..0 and push_back 5..9 => 0..9; drain front asc
  %d = call ptr @universe_ds_deque_create(i64 8, i64 4)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %front.val = sub nsw i64 4, %i
  store i64 %front.val, ptr %v, align 8
  %r1 = call i32 @universe_ds_deque_push_front(ptr %d, ptr nonnull %v)
  %back.val = add nuw i64 %i, 5
  store i64 %back.val, ptr %v, align 8
  %r2 = call i32 @universe_ds_deque_push_back(ptr %d, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 5
  br i1 %more, label %fill, label %peeks

peeks:
  %pf = call i32 @universe_ds_deque_peek_front(ptr %d, ptr nonnull %v)
  %fv = load i64, ptr %v, align 8
  %pb = call i32 @universe_ds_deque_peek_back(ptr %d, ptr nonnull %v)
  %bv = load i64, ptr %v, align 8
  %cnt = call i64 @universe_ds_deque_count(ptr %d)
  %ok1 = icmp eq i64 %fv, 0
  %ok2 = icmp eq i64 %bv, 9
  %ok3 = icmp eq i64 %cnt, 10
  %a1 = and i1 %ok1, %ok2
  %a2 = and i1 %a1, %ok3
  call void @ut_check(i1 %a2, ptr @m.peek)
  br label %drain

drain:
  %j = phi i64 [ 0, %peeks ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %peeks ], [ %viol.n, %drain ]
  %rc = call i32 @universe_ds_deque_pop_front(ptr %d, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %bad1 = icmp ne i32 %rc, 0
  %bad2 = icmp ne i64 %got, %j
  %anybad = or i1 %bad1, %bad2
  %inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 10
  br i1 %more2, label %drain, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.mirror)
  %e1 = call i32 @universe_ds_deque_pop_front(ptr %d, ptr nonnull %v)
  %e2 = call i32 @universe_ds_deque_pop_back(ptr %d, ptr nonnull %v)
  %s = add nuw i32 %e1, %e2
  %s.w = zext i32 %s to i64
  call void @ut_check_eq(i64 %s.w, i64 8, ptr @m.empty)
  call void @universe_ds_deque_destroy(ptr %d)
  ret void
}

define internal void @test_wrap_and_grow() {
entry:
  ; fresh deque: push_front first (head goes 0 -> -1: unsigned wrap)
  %d = call ptr @universe_ds_deque_create(i64 8, i64 8)
  %v = alloca i64, align 8
  store i64 77, ptr %v, align 8
  %r1 = call i32 @universe_ds_deque_push_front(ptr %d, ptr nonnull %v)
  store i64 0, ptr %v, align 8
  %r2 = call i32 @universe_ds_deque_pop_back(ptr %d, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %w1 = icmp eq i32 %r1, 0
  %w2 = icmp eq i32 %r2, 0
  %w3 = icmp eq i64 %got, 77
  %wa = and i1 %w1, %w2
  %wb = and i1 %wa, %w3
  call void @ut_check(i1 %wb, ptr @m.wrap)

  ; interleave front/back pushes far past capacity: 0..63 back, -1..-64 front
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %bv = add nuw i64 %i, 1000
  store i64 %bv, ptr %v, align 8
  %r3 = call i32 @universe_ds_deque_push_back(ptr %d, ptr nonnull %v)
  %fv = sub nsw i64 999, %i
  store i64 %fv, ptr %v, align 8
  %r4 = call i32 @universe_ds_deque_push_front(ptr %d, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 64
  br i1 %more, label %fill, label %drain

drain:                                       ; must come out 936..1063 asc
  %j = phi i64 [ 0, %fill ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %fill ], [ %viol.n, %drain ]
  %rc = call i32 @universe_ds_deque_pop_front(ptr %d, ptr nonnull %v)
  %got2 = load i64, ptr %v, align 8
  %want = add nuw i64 %j, 936
  %bad1 = icmp ne i32 %rc, 0
  %bad2 = icmp ne i64 %got2, %want
  %anybad = or i1 %bad1, %bad2
  %inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 128
  br i1 %more2, label %drain, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.grow)
  call void @universe_ds_deque_destroy(ptr %d)
  ret void
}

define internal void @bench() {
entry:
  %d = call ptr @universe_ds_deque_create(i64 8, i64 1024)
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  store i64 %i, ptr %v, align 8
  %r1 = call i32 @universe_ds_deque_push_back(ptr %d, ptr nonnull %v)
  %r2 = call i32 @universe_ds_deque_push_front(ptr %d, ptr nonnull %v)
  %r3 = call i32 @universe_ds_deque_pop_back(ptr %d, ptr nonnull %v)
  %r4 = call i32 @universe_ds_deque_pop_front(ptr %d, ptr nonnull %v)
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
  %sp = getelementptr inbounds double, ptr @deque.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_deque_destroy(ptr %d)
  call void @ut_report_dist(ptr @deque.samp, i64 16, i64 4000000, ptr @lbl.deque)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_mirror()
  call void @test_wrap_and_grow()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
