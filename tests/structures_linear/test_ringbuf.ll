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

; Tests for universe_ds_ringbuf: FIFO order, wrap-around, full/empty/peek
; errors, capacity rounding, --bench push/pop throughput.

declare ptr @universe_ds_ringbuf_create(i64, i64)
declare i32 @universe_ds_ringbuf_push(ptr, ptr)
declare i32 @universe_ds_ringbuf_pop(ptr, ptr)
declare i32 @universe_ds_ringbuf_peek(ptr, i64, ptr)
declare i64 @universe_ds_ringbuf_count(ptr)
declare i64 @universe_ds_ringbuf_capacity(ptr)
declare void @universe_ds_ringbuf_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.cap     = private unnamed_addr constant [22 x i8] c"capacity rounds to 16\00"
@m.fill    = private unnamed_addr constant [14 x i8] c"16 pushes OK \00"
@m.full    = private unnamed_addr constant [13 x i8] c"17th is FULL\00"
@m.fifo    = private unnamed_addr constant [11 x i8] c"FIFO order\00"
@m.empty   = private unnamed_addr constant [15 x i8] c"pop empty gets\00"
@m.peek    = private unnamed_addr constant [8 x i8] c"peek(0)\00"
@m.peekoob = private unnamed_addr constant [9 x i8] c"peek oob\00"
@m.wrap    = private unnamed_addr constant [20 x i8] c"1000-op wrap intact\00"
@m.errs    = private unnamed_addr constant [11 x i8] c"null/zero \00"
@lbl.ringbuf = private unnamed_addr constant [30 x i8] c"ringbuf push+pop (2M ops/rep)\00"
@ringbuf.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_basic() {
entry:
  %rb = call ptr @universe_ds_ringbuf_create(i64 10, i64 8)
  %cap = call i64 @universe_ds_ringbuf_capacity(ptr %rb)
  call void @ut_check_eq(i64 %cap, i64 16, ptr @m.cap)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %errs = phi i64 [ 0, %entry ], [ %errs.n, %fill ]
  store i64 %i, ptr %v, align 8
  %rc = call i32 @universe_ds_ringbuf_push(ptr %rb, ptr nonnull %v)
  %bad = icmp ne i32 %rc, 0
  %e.inc = zext i1 %bad to i64
  %errs.n = add nuw i64 %errs, %e.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 16
  br i1 %more, label %fill, label %filled

filled:
  call void @ut_check_eq(i64 %errs.n, i64 0, ptr @m.fill)
  store i64 99, ptr %v, align 8
  %rc.full = call i32 @universe_ds_ringbuf_push(ptr %rb, ptr nonnull %v)
  %rc.full.w = zext i32 %rc.full to i64
  call void @ut_check_eq(i64 %rc.full.w, i64 6, ptr @m.full)

  ; peek oldest and out-of-range
  %rc.peek = call i32 @universe_ds_ringbuf_peek(ptr %rb, i64 0, ptr nonnull %v)
  %peeked = load i64, ptr %v, align 8
  %rc.ok = icmp eq i32 %rc.peek, 0
  %val.ok = icmp eq i64 %peeked, 0
  %peek.ok = and i1 %rc.ok, %val.ok
  call void @ut_check(i1 %peek.ok, ptr @m.peek)
  %rc.oob = call i32 @universe_ds_ringbuf_peek(ptr %rb, i64 16, ptr nonnull %v)
  %rc.oob.w = zext i32 %rc.oob to i64
  call void @ut_check_eq(i64 %rc.oob.w, i64 7, ptr @m.peekoob)
  br label %drain

drain:
  %j = phi i64 [ 0, %filled ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %filled ], [ %viol.n, %drain ]
  %rc2 = call i32 @universe_ds_ringbuf_pop(ptr %rb, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %rc2.bad = icmp ne i32 %rc2, 0
  %got.bad = icmp ne i64 %got, %j
  %anybad = or i1 %rc2.bad, %got.bad
  %v.inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 16
  br i1 %more2, label %drain, label %drained

drained:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.fifo)
  %rc3 = call i32 @universe_ds_ringbuf_pop(ptr %rb, ptr nonnull %v)
  %rc3.w = zext i32 %rc3 to i64
  call void @ut_check_eq(i64 %rc3.w, i64 4, ptr @m.empty)
  call void @universe_ds_ringbuf_destroy(ptr %rb)
  ret void
}

define internal void @test_wrap() {
entry:                                      ; cap 8; 1000 interleaved ops wrap
  %rb = call ptr @universe_ds_ringbuf_create(i64 8, i64 8)
  %v = alloca i64, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %loop ]
  store i64 %i, ptr %v, align 8
  %rc1 = call i32 @universe_ds_ringbuf_push(ptr %rb, ptr nonnull %v)
  store i64 -1, ptr %v, align 8
  %rc2 = call i32 @universe_ds_ringbuf_pop(ptr %rb, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %bad1 = icmp ne i32 %rc1, 0
  %bad2 = icmp ne i32 %rc2, 0
  %bad3 = icmp ne i64 %got, %i
  %b12 = or i1 %bad1, %bad2
  %bad = or i1 %b12, %bad3
  %v.inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %loop, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.wrap)
  call void @universe_ds_ringbuf_destroy(ptr %rb)
  ret void
}

define internal void @test_errors() {
entry:
  %v = alloca i64, align 8
  %e1 = call i32 @universe_ds_ringbuf_push(ptr null, ptr nonnull %v)
  %rb0 = call ptr @universe_ds_ringbuf_create(i64 0, i64 8)
  %rb0.null = icmp eq ptr %rb0, null
  %rb1 = call ptr @universe_ds_ringbuf_create(i64 8, i64 0)
  %rb1.null = icmp eq ptr %rb1, null
  %e1.ok = icmp eq i32 %e1, 1
  %a = and i1 %rb0.null, %rb1.null
  %all = and i1 %a, %e1.ok
  call void @ut_check(i1 %all, ptr @m.errs)
  call void @universe_ds_ringbuf_destroy(ptr null)
  ret void
}

define internal void @bench() {
entry:
  %rb = call ptr @universe_ds_ringbuf_create(i64 1024, i64 8)
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  store i64 %i, ptr %v, align 8
  %rc1 = call i32 @universe_ds_ringbuf_push(ptr %rb, ptr nonnull %v)
  %rc2 = call i32 @universe_ds_ringbuf_pop(ptr %rb, ptr nonnull %v)
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
  %sp = getelementptr inbounds double, ptr @ringbuf.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_ringbuf_destroy(ptr %rb)
  call void @ut_report_dist(ptr @ringbuf.samp, i64 16, i64 2000000, ptr @lbl.ringbuf)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_wrap()
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
