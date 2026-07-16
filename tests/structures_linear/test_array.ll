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

; Tests for universe_ds_array: push/get/set roundtrip through growth,
; insert/remove memmove correctness at head/middle/tail, error codes,
; --bench.

declare ptr @universe_ds_array_create(i64, i64)
declare i32 @universe_ds_array_push(ptr, ptr)
declare i32 @universe_ds_array_get(ptr, i64, ptr)
declare i32 @universe_ds_array_set(ptr, i64, ptr)
declare i32 @universe_ds_array_insert(ptr, i64, ptr)
declare i32 @universe_ds_array_remove(ptr, i64, ptr)
declare i64 @universe_ds_array_count(ptr)
declare void @universe_ds_array_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.roundtrip = private unnamed_addr constant [25 x i8] c"push/get 65536 roundtrip\00"
@m.set       = private unnamed_addr constant [15 x i8] c"set overwrites\00"
@m.inshead   = private unnamed_addr constant [20 x i8] c"insert head shifts \00"
@m.instail   = private unnamed_addr constant [19 x i8] c"insert tail append\00"
@m.remove    = private unnamed_addr constant [22 x i8] c"remove closes the gap\00"
@m.errs      = private unnamed_addr constant [17 x i8] c"error codes 1/7 \00"
@lbl.array   = private unnamed_addr constant [24 x i8] c"array push+get (2M/rep)\00"
@array.samp  = internal global [16 x double] zeroinitializer, align 8

define internal void @test_roundtrip() {
entry:
  %a = call ptr @universe_ds_array_create(i64 8, i64 4)
  %v = alloca i64, align 8
  br label %push

push:
  %i = phi i64 [ 0, %entry ], [ %i.n, %push ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %push ]
  %val = mul i64 %i, 7
  store i64 %val, ptr %v, align 8
  %rc = call i32 @universe_ds_array_push(ptr %a, ptr nonnull %v)
  %bad = icmp ne i32 %rc, 0
  %b.inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %b.inc
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 65536
  br i1 %more, label %push, label %verify

verify:
  %j = phi i64 [ 0, %push ], [ %j.n, %verify ]
  %viol2 = phi i64 [ %viol.n, %push ], [ %viol2.n, %verify ]
  %grc = call i32 @universe_ds_array_get(ptr %a, i64 %j, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %want = mul i64 %j, 7
  %bad1 = icmp ne i32 %grc, 0
  %bad2 = icmp ne i64 %got, %want
  %anybad = or i1 %bad1, %bad2
  %v.inc = zext i1 %anybad to i64
  %viol2.n = add nuw i64 %viol2, %v.inc
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 65536
  br i1 %more2, label %verify, label %report

report:
  call void @ut_check_eq(i64 %viol2.n, i64 0, ptr @m.roundtrip)
  ; set index 100 then read back
  store i64 -5, ptr %v, align 8
  %src = call i32 @universe_ds_array_set(ptr %a, i64 100, ptr nonnull %v)
  store i64 0, ptr %v, align 8
  %grc2 = call i32 @universe_ds_array_get(ptr %a, i64 100, ptr nonnull %v)
  %got2 = load i64, ptr %v, align 8
  %s.ok = icmp eq i64 %got2, -5
  call void @ut_check(i1 %s.ok, ptr @m.set)
  call void @universe_ds_array_destroy(ptr %a)
  ret void
}

define internal void @test_insert_remove() {
entry:
  %a = call ptr @universe_ds_array_create(i64 8, i64 4)
  %v = alloca i64, align 8
  ; [10,20,30]
  store i64 10, ptr %v, align 8
  %r1 = call i32 @universe_ds_array_push(ptr %a, ptr nonnull %v)
  store i64 20, ptr %v, align 8
  %r2 = call i32 @universe_ds_array_push(ptr %a, ptr nonnull %v)
  store i64 30, ptr %v, align 8
  %r3 = call i32 @universe_ds_array_push(ptr %a, ptr nonnull %v)
  ; insert 5 at head -> [5,10,20,30]
  store i64 5, ptr %v, align 8
  %r4 = call i32 @universe_ds_array_insert(ptr %a, i64 0, ptr nonnull %v)
  %g0 = call i32 @universe_ds_array_get(ptr %a, i64 0, ptr nonnull %v)
  %v0 = load i64, ptr %v, align 8
  %g1 = call i32 @universe_ds_array_get(ptr %a, i64 1, ptr nonnull %v)
  %v1 = load i64, ptr %v, align 8
  %g3 = call i32 @universe_ds_array_get(ptr %a, i64 3, ptr nonnull %v)
  %v3 = load i64, ptr %v, align 8
  %h1 = icmp eq i64 %v0, 5
  %h2 = icmp eq i64 %v1, 10
  %h3 = icmp eq i64 %v3, 30
  %ha = and i1 %h1, %h2
  %hb = and i1 %ha, %h3
  call void @ut_check(i1 %hb, ptr @m.inshead)
  ; insert 40 at tail (index == count) -> [5,10,20,30,40]
  store i64 40, ptr %v, align 8
  %cnt = call i64 @universe_ds_array_count(ptr %a)
  %r5 = call i32 @universe_ds_array_insert(ptr %a, i64 %cnt, ptr nonnull %v)
  %g4 = call i32 @universe_ds_array_get(ptr %a, i64 4, ptr nonnull %v)
  %v4 = load i64, ptr %v, align 8
  %t.ok = icmp eq i64 %v4, 40
  call void @ut_check(i1 %t.ok, ptr @m.instail)
  ; remove index 2 (20) -> [5,10,30,40]; out captures removed value
  store i64 0, ptr %v, align 8
  %r6 = call i32 @universe_ds_array_remove(ptr %a, i64 2, ptr nonnull %v)
  %rv = load i64, ptr %v, align 8
  %g2b = call i32 @universe_ds_array_get(ptr %a, i64 2, ptr nonnull %v)
  %v2b = load i64, ptr %v, align 8
  %cnt2 = call i64 @universe_ds_array_count(ptr %a)
  %rm1 = icmp eq i64 %rv, 20
  %rm2 = icmp eq i64 %v2b, 30
  %rm3 = icmp eq i64 %cnt2, 4
  %ra = and i1 %rm1, %rm2
  %rb = and i1 %ra, %rm3
  call void @ut_check(i1 %rb, ptr @m.remove)
  call void @universe_ds_array_destroy(ptr %a)
  ret void
}

define internal void @test_errors() {
entry:
  %v = alloca i64, align 8
  %a = call ptr @universe_ds_array_create(i64 8, i64 4)
  %e1 = call i32 @universe_ds_array_push(ptr null, ptr nonnull %v)
  %e2 = call i32 @universe_ds_array_get(ptr %a, i64 0, ptr nonnull %v)   ; empty -> 7
  %e3 = call i32 @universe_ds_array_insert(ptr %a, i64 1, ptr nonnull %v) ; > count -> 7
  %e4 = call i32 @universe_ds_array_remove(ptr %a, i64 0, ptr null)      ; empty -> 7
  %ok1 = icmp eq i32 %e1, 1
  %ok2 = icmp eq i32 %e2, 7
  %ok3 = icmp eq i32 %e3, 7
  %ok4 = icmp eq i32 %e4, 7
  %a1 = and i1 %ok1, %ok2
  %a2 = and i1 %a1, %ok3
  %a3 = and i1 %a2, %ok4
  call void @ut_check(i1 %a3, ptr @m.errs)
  call void @universe_ds_array_destroy(ptr %a)
  call void @universe_ds_array_destroy(ptr null)
  ret void
}

define internal void @bench() {
entry:
  %v = alloca i64, align 8
  br label %rep

rep:
  %rep.i = phi i64 [ 0, %entry ], [ %rep.n, %rep.next ]
  %a = call ptr @universe_ds_array_create(i64 8, i64 1024)
  %t0 = call double @ut_now_sec()
  br label %push

push:
  %i = phi i64 [ 0, %rep ], [ %i.n, %push ]
  store i64 %i, ptr %v, align 8
  %rc = call i32 @universe_ds_array_push(ptr %a, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %push, label %reads

reads:
  %j = phi i64 [ 0, %push ], [ %j.n, %reads ]
  %grc = call i32 @universe_ds_array_get(ptr %a, i64 %j, ptr nonnull %v)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000000
  br i1 %more2, label %reads, label %rep.done

rep.done:
  %t1 = call double @ut_now_sec()
  call void @universe_ds_array_destroy(ptr %a)
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %rep.i, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %sidx = sub i64 %rep.i, 1
  %sp = getelementptr inbounds double, ptr @array.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @ut_report_dist(ptr @array.samp, i64 16, i64 2000000, ptr @lbl.array)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_roundtrip()
  call void @test_insert_remove()
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
