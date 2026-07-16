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

; Tests for universe_ds_queue: FIFO across growth while WRAPPED (the
; two-memcpy linearization path), interleaved churn, errors, --bench.

declare ptr @universe_ds_queue_create(i64, i64)
declare i32 @universe_ds_queue_enqueue(ptr, ptr)
declare i32 @universe_ds_queue_dequeue(ptr, ptr)
declare i32 @universe_ds_queue_peek(ptr, ptr)
declare i64 @universe_ds_queue_count(ptr)
declare i64 @universe_ds_queue_capacity(ptr)
declare void @universe_ds_queue_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.wrapgrow = private unnamed_addr constant [26 x i8] c"FIFO thru wrapped growth \00"
@m.churn    = private unnamed_addr constant [24 x i8] c"100k interleaved intact\00"
@m.peek     = private unnamed_addr constant [21 x i8] c"peek front nondestru\00"
@m.empty    = private unnamed_addr constant [16 x i8] c"dequeue empty 4\00"
@m.errs     = private unnamed_addr constant [15 x i8] c"null/zero args\00"
@lbl.queue  = private unnamed_addr constant [27 x i8] c"queue enq+deq (2M ops/rep)\00"
@queue.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_wrapped_growth() {
entry:
  ; cap 8; advance head by 5 so the ring is wrapped when growth hits
  %q = call ptr @universe_ds_queue_create(i64 8, i64 8)
  %v = alloca i64, align 8
  br label %prefill

prefill:                                     ; enqueue 0..4, dequeue them
  %i = phi i64 [ 0, %entry ], [ %i.n, %prefill ]
  store i64 %i, ptr %v, align 8
  %rc1 = call i32 @universe_ds_queue_enqueue(ptr %q, ptr nonnull %v)
  %rc2 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 5
  br i1 %more, label %prefill, label %fill

fill:                                        ; now push 100..131 (wraps, grows twice)
  %j = phi i64 [ 0, %prefill ], [ %j.n, %fill ]
  %errs = phi i64 [ 0, %prefill ], [ %errs.n, %fill ]
  %val = add nuw i64 %j, 100
  store i64 %val, ptr %v, align 8
  %rc3 = call i32 @universe_ds_queue_enqueue(ptr %q, ptr nonnull %v)
  %bad = icmp ne i32 %rc3, 0
  %e.inc = zext i1 %bad to i64
  %errs.n = add nuw i64 %errs, %e.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 32
  br i1 %more2, label %fill, label %peek

peek:
  %rc.p = call i32 @universe_ds_queue_peek(ptr %q, ptr nonnull %v)
  %front = load i64, ptr %v, align 8
  %cnt = call i64 @universe_ds_queue_count(ptr %q)
  %pk.rc = icmp eq i32 %rc.p, 0
  %pk.val = icmp eq i64 %front, 100
  %pk.cnt = icmp eq i64 %cnt, 32
  %pk1 = and i1 %pk.rc, %pk.val
  %pk = and i1 %pk1, %pk.cnt
  call void @ut_check(i1 %pk, ptr @m.peek)
  br label %drain

drain:                                       ; must come out 100..131 in order
  %k = phi i64 [ 0, %peek ], [ %k.n, %drain ]
  %viol = phi i64 [ %errs.n, %peek ], [ %viol.n, %drain ]
  %rc4 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %want = add nuw i64 %k, 100
  %bad1 = icmp ne i32 %rc4, 0
  %bad2 = icmp ne i64 %got, %want
  %anybad = or i1 %bad1, %bad2
  %v.inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %k.n = add nuw nsw i64 %k, 1
  %more3 = icmp ult i64 %k.n, 32
  br i1 %more3, label %drain, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.wrapgrow)
  %rc5 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr nonnull %v)
  %rc5.w = zext i32 %rc5 to i64
  call void @ut_check_eq(i64 %rc5.w, i64 4, ptr @m.empty)
  call void @universe_ds_queue_destroy(ptr %q)
  ret void
}

define internal void @test_churn() {
entry:                                       ; 2 in, 1 out x100k: grows + drains
  %q = call ptr @universe_ds_queue_create(i64 8, i64 16)
  %v = alloca i64, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %inseq = phi i64 [ 0, %entry ], [ %inseq.n2, %loop ]
  %outseq = phi i64 [ 0, %entry ], [ %outseq.n, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %loop ]
  store i64 %inseq, ptr %v, align 8
  %rc1 = call i32 @universe_ds_queue_enqueue(ptr %q, ptr nonnull %v)
  %inseq.n = add nuw i64 %inseq, 1
  store i64 %inseq.n, ptr %v, align 8
  %rc2 = call i32 @universe_ds_queue_enqueue(ptr %q, ptr nonnull %v)
  %inseq.n2 = add nuw i64 %inseq.n, 1
  %rc3 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %bad1 = icmp ne i32 %rc3, 0
  %bad2 = icmp ne i64 %got, %outseq
  %anybad = or i1 %bad1, %bad2
  %v.inc = zext i1 %anybad to i64
  %viol.n = add nuw i64 %viol, %v.inc
  %outseq.n = add nuw i64 %outseq, 1
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %loop, label %done

done:
  call void @ut_check_eq(i64 %viol.n, i64 0, ptr @m.churn)
  %left = call i64 @universe_ds_queue_count(ptr %q)
  call void @ut_check_eq(i64 %left, i64 100000, ptr @m.churn)
  call void @universe_ds_queue_destroy(ptr %q)
  ret void
}

define internal void @test_errors() {
entry:
  %v = alloca i64, align 8
  %e1 = call i32 @universe_ds_queue_enqueue(ptr null, ptr nonnull %v)
  %q = call ptr @universe_ds_queue_create(i64 8, i64 8)
  %e2 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr null)
  %z = call ptr @universe_ds_queue_create(i64 0, i64 8)
  %z.null = icmp eq ptr %z, null
  %e1.ok = icmp eq i32 %e1, 1
  %e2.ok = icmp eq i32 %e2, 1
  %a1 = and i1 %e1.ok, %e2.ok
  %all = and i1 %a1, %z.null
  call void @ut_check(i1 %all, ptr @m.errs)
  call void @universe_ds_queue_destroy(ptr %q)
  call void @universe_ds_queue_destroy(ptr null)
  ret void
}

define internal void @bench() {
entry:
  %q = call ptr @universe_ds_queue_create(i64 8, i64 1024)
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %rep ], [ %i.n, %loop ]
  store i64 %i, ptr %v, align 8
  %rc1 = call i32 @universe_ds_queue_enqueue(ptr %q, ptr nonnull %v)
  %rc2 = call i32 @universe_ds_queue_dequeue(ptr %q, ptr nonnull %v)
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
  %sp = getelementptr inbounds double, ptr @queue.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_queue_destroy(ptr %q)
  call void @ut_report_dist(ptr @queue.samp, i64 16, i64 2000000, ptr @lbl.queue)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_wrapped_growth()
  call void @test_churn()
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
