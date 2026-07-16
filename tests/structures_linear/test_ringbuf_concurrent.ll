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

; Tests for universe_ds_cringbuf (SPSC lock-free ring).
; Stress: producer thread pushes 0..1M-1 while the main thread consumes;
; the consumer must observe EXACTLY the sequence 0,1,2,... — any lost,
; duplicated, torn or reordered element trips the violation counter.
; This is the memory-ordering proof on weakly-ordered ARM64.

declare ptr @universe_ds_cringbuf_create(i64, i64)
declare i32 @universe_ds_cringbuf_push(ptr, ptr)
declare i32 @universe_ds_cringbuf_pop(ptr, ptr)
declare i64 @universe_ds_cringbuf_count(ptr)
declare i64 @universe_ds_cringbuf_capacity(ptr)
declare void @universe_ds_cringbuf_destroy(ptr)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @sched_yield()

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)

@g.rb = internal global ptr null, align 8

@m.single = private unnamed_addr constant [20 x i8] c"single-thread paths\00"
@m.spawn  = private unnamed_addr constant [15 x i8] c"producer spawn\00"
@m.seq    = private unnamed_addr constant [25 x i8] c"1M in exact FIFO order  \00"
@m.drain  = private unnamed_addr constant [21 x i8] c"ring empty at finish\00"

define internal void @test_single() {
entry:
  %rb = call ptr @universe_ds_cringbuf_create(i64 4, i64 8)
  %v = alloca i64, align 8
  store i64 7, ptr %v, align 8
  %r1 = call i32 @universe_ds_cringbuf_push(ptr %rb, ptr nonnull %v)
  %r2 = call i32 @universe_ds_cringbuf_push(ptr %rb, ptr nonnull %v)
  store i64 0, ptr %v, align 8
  %r3 = call i32 @universe_ds_cringbuf_pop(ptr %rb, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  ; drain second, then EMPTY
  %r4 = call i32 @universe_ds_cringbuf_pop(ptr %rb, ptr nonnull %v)
  %r5 = call i32 @universe_ds_cringbuf_pop(ptr %rb, ptr nonnull %v)
  %ok1 = icmp eq i32 %r1, 0
  %ok2 = icmp eq i32 %r2, 0
  %ok3 = icmp eq i32 %r3, 0
  %ok4 = icmp eq i64 %got, 7
  %ok5 = icmp eq i32 %r5, 4
  %a1 = and i1 %ok1, %ok2
  %a2 = and i1 %a1, %ok3
  %a3 = and i1 %a2, %ok4
  %a4 = and i1 %a3, %ok5
  call void @ut_check(i1 %a4, ptr @m.single)
  call void @universe_ds_cringbuf_destroy(ptr %rb)
  ret void
}

define internal ptr @producer(ptr %arg) {
entry:
  %rb = load ptr, ptr @g.rb, align 8
  %v = alloca i64, align 8
  br label %next

next:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  store i64 %i, ptr %v, align 8
  br label %try

try:
  %rc = call i32 @universe_ds_cringbuf_push(ptr %rb, ptr nonnull %v)
  %full = icmp ne i32 %rc, 0
  br i1 %full, label %yield, label %pushed

yield:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %next, label %done

done:
  ret ptr null
}

define internal void @test_stress() {
entry:
  %rb = call ptr @universe_ds_cringbuf_create(i64 1024, i64 8)
  store ptr %rb, ptr @g.rb, align 8
  %tid.slot = alloca i64, align 8
  %rc = call i32 @pthread_create(ptr nonnull %tid.slot, ptr null, ptr @producer, ptr null)
  %rc.w = zext i32 %rc to i64
  call void @ut_check_eq(i64 %rc.w, i64 0, ptr @m.spawn)
  %v = alloca i64, align 8
  br label %consume

consume:
  %i = phi i64 [ 0, %entry ], [ %i.n, %got ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %got ]
  br label %try

try:
  %prc = call i32 @universe_ds_cringbuf_pop(ptr %rb, ptr nonnull %v)
  %empty = icmp ne i32 %prc, 0
  br i1 %empty, label %yield, label %got

yield:
  %y = call i32 @sched_yield()
  br label %try

got:
  %val = load i64, ptr %v, align 8
  %bad = icmp ne i64 %val, %i
  %v.inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %consume, label %finish

finish:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.seq)
  %tid = load i64, ptr %tid.slot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %left = call i64 @universe_ds_cringbuf_count(ptr %rb)
  call void @ut_check_eq(i64 %left, i64 0, ptr @m.drain)
  call void @universe_ds_cringbuf_destroy(ptr %rb)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_single()
  call void @test_stress()
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
