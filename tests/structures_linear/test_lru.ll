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

; Tests for universe_ds_lru: hit/miss, update-in-place, eviction order with
; recency refresh, self-validating churn (value == key*31), --bench.

declare ptr @universe_ds_lru_create(i64, i64)
declare i32 @universe_ds_lru_put(ptr, i64, ptr)
declare i32 @universe_ds_lru_get(ptr, i64, ptr)
declare i32 @universe_ds_lru_contains(ptr, i64)
declare i64 @universe_ds_lru_count(ptr)
declare void @universe_ds_lru_destroy(ptr)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.basic  = private unnamed_addr constant [18 x i8] c"put/get roundtrip\00"
@m.update = private unnamed_addr constant [16 x i8] c"update in place\00"
@m.evict  = private unnamed_addr constant [26 x i8] c"eviction respects recency\00"
@m.count  = private unnamed_addr constant [15 x i8] c"count == cap  \00"
@m.churn  = private unnamed_addr constant [26 x i8] c"100k churn self-validates\00"
@m.miss   = private unnamed_addr constant [15 x i8] c"miss returns 5\00"
@lbl.lru = private unnamed_addr constant [25 x i8] c"lru get hot (1M ops/rep)\00"
@lru.samp = internal global [16 x double] zeroinitializer, align 8

define internal void @test_basic() {
entry:
  %c = call ptr @universe_ds_lru_create(i64 3, i64 8)
  %v = alloca i64, align 8

  store i64 100, ptr %v, align 8
  %p1 = call i32 @universe_ds_lru_put(ptr %c, i64 1, ptr nonnull %v)
  store i64 0, ptr %v, align 8
  %g1 = call i32 @universe_ds_lru_get(ptr %c, i64 1, ptr nonnull %v)
  %got = load i64, ptr %v, align 8
  %ok1 = icmp eq i32 %p1, 0
  %ok2 = icmp eq i32 %g1, 0
  %ok3 = icmp eq i64 %got, 100
  %a1 = and i1 %ok1, %ok2
  %a2 = and i1 %a1, %ok3
  call void @ut_check(i1 %a2, ptr @m.basic)

  ; update same key: value replaced, count unchanged
  store i64 200, ptr %v, align 8
  %p2 = call i32 @universe_ds_lru_put(ptr %c, i64 1, ptr nonnull %v)
  store i64 0, ptr %v, align 8
  %g2 = call i32 @universe_ds_lru_get(ptr %c, i64 1, ptr nonnull %v)
  %got2 = load i64, ptr %v, align 8
  %cnt = call i64 @universe_ds_lru_count(ptr %c)
  %u1 = icmp eq i64 %got2, 200
  %u2 = icmp eq i64 %cnt, 1
  %u = and i1 %u1, %u2
  call void @ut_check(i1 %u, ptr @m.update)

  %miss = call i32 @universe_ds_lru_get(ptr %c, i64 999, ptr nonnull %v)
  %miss.w = zext i32 %miss to i64
  call void @ut_check_eq(i64 %miss.w, i64 5, ptr @m.miss)
  call void @universe_ds_lru_destroy(ptr %c)
  ret void
}

define internal void @test_eviction() {
entry:
  %c = call ptr @universe_ds_lru_create(i64 3, i64 8)
  %v = alloca i64, align 8
  ; insert 1,2,3 (recency: 3>2>1). get(1) refreshes. put(4) must evict 2.
  store i64 10, ptr %v, align 8
  %p1 = call i32 @universe_ds_lru_put(ptr %c, i64 1, ptr nonnull %v)
  store i64 20, ptr %v, align 8
  %p2 = call i32 @universe_ds_lru_put(ptr %c, i64 2, ptr nonnull %v)
  store i64 30, ptr %v, align 8
  %p3 = call i32 @universe_ds_lru_put(ptr %c, i64 3, ptr nonnull %v)
  %g1 = call i32 @universe_ds_lru_get(ptr %c, i64 1, ptr nonnull %v)
  store i64 40, ptr %v, align 8
  %p4 = call i32 @universe_ds_lru_put(ptr %c, i64 4, ptr nonnull %v)

  %has1 = call i32 @universe_ds_lru_contains(ptr %c, i64 1)
  %has2 = call i32 @universe_ds_lru_contains(ptr %c, i64 2)
  %has3 = call i32 @universe_ds_lru_contains(ptr %c, i64 3)
  %has4 = call i32 @universe_ds_lru_contains(ptr %c, i64 4)
  %e1 = icmp eq i32 %has1, 0
  %e2 = icmp eq i32 %has2, 5
  %e3 = icmp eq i32 %has3, 0
  %e4 = icmp eq i32 %has4, 0
  %x1 = and i1 %e1, %e2
  %x2 = and i1 %x1, %e3
  %x3 = and i1 %x2, %e4
  call void @ut_check(i1 %x3, ptr @m.evict)
  %cnt = call i64 @universe_ds_lru_count(ptr %c)
  call void @ut_check_eq(i64 %cnt, i64 3, ptr @m.count)
  call void @universe_ds_lru_destroy(ptr %c)
  ret void
}

define internal void @test_churn() {
entry:                                     ; keyspace 2x cap; value must always
  %c = call ptr @universe_ds_lru_create(i64 64, i64 8)   ; equal key*31
  %v = alloca i64, align 8
  %seed = alloca i64, align 8
  store i64 77, ptr %seed, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %next ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n2, %next ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %key = urem i64 %r, 128
  %do.put = and i64 %r, 256
  %is.put = icmp ne i64 %do.put, 0
  br i1 %is.put, label %put, label %get

put:
  %stamp = mul i64 %key, 31
  store i64 %stamp, ptr %v, align 8
  %prc = call i32 @universe_ds_lru_put(ptr %c, i64 %key, ptr nonnull %v)
  %prc.bad = icmp ne i32 %prc, 0
  %pv = zext i1 %prc.bad to i64
  %viol.p = add nuw i64 %viol, %pv
  br label %next

get:
  store i64 -1, ptr %v, align 8
  %grc = call i32 @universe_ds_lru_get(ptr %c, i64 %key, ptr nonnull %v)
  %hit = icmp eq i32 %grc, 0
  br i1 %hit, label %check.val, label %ok.miss

check.val:
  %got = load i64, ptr %v, align 8
  %want = mul i64 %key, 31
  %bad = icmp ne i64 %got, %want
  %gv = zext i1 %bad to i64
  %viol.g = add nuw i64 %viol, %gv
  br label %next

ok.miss:
  br label %next

next:
  %viol.n2 = phi i64 [ %viol.p, %put ], [ %viol.g, %check.val ], [ %viol, %ok.miss ]
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %loop, label %done

done:
  call void @ut_check_eq(i64 %viol.n2, i64 0, ptr @m.churn)
  call void @universe_ds_lru_destroy(ptr %c)
  ret void
}

define internal void @bench() {
entry:
  %c = call ptr @universe_ds_lru_create(i64 1024, i64 8)
  %v = alloca i64, align 8
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  store i64 %i, ptr %v, align 8
  %prc = call i32 @universe_ds_lru_put(ptr %c, i64 %i, ptr nonnull %v)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %fill, label %rep

rep:
  %rep.i = phi i64 [ 0, %fill ], [ %rep.n, %rep.next ]
  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %j = phi i64 [ 0, %rep ], [ %j.n, %loop ]
  %key = and i64 %j, 1023
  %grc = call i32 @universe_ds_lru_get(ptr %c, i64 %key, ptr nonnull %v)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000000
  br i1 %more2, label %loop, label %rep.done

rep.done:
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %rep.i, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %sidx = sub i64 %rep.i, 1
  %sp = getelementptr inbounds double, ptr @lru.samp, i64 %sidx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %rep.n = add nuw nsw i64 %rep.i, 1
  %rep.more = icmp ult i64 %rep.n, 17
  br i1 %rep.more, label %rep, label %report

report:
  call void @universe_ds_lru_destroy(ptr %c)
  call void @ut_report_dist(ptr @lru.samp, i64 16, i64 1000000, ptr @lbl.lru)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_eviction()
  call void @test_churn()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
