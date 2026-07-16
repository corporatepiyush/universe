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

; Universe test harness. Linked into every test binary; not part of the lib.
; CLOCK id 0 = CLOCK_REALTIME on macOS, Linux and FreeBSD (portable; benches
; only need coarse deltas).

@ut.checks = internal global i64 0, align 8
@ut.failures = internal global i64 0, align 8

@ut.fmt.fail = private unnamed_addr constant [10 x i8] c"FAIL: %s\0A\00", align 1
@ut.fmt.faileq = private unnamed_addr constant [25 x i8] c"FAIL: %s (%lld != %lld)\0A\00", align 1
@ut.fmt.bad = private unnamed_addr constant [25 x i8] c"%lld/%lld checks FAILED\0A\00", align 1
@ut.fmt.ok = private unnamed_addr constant [17 x i8] c"ok: %lld checks\0A\00", align 1
@ut.str.bench = private unnamed_addr constant [8 x i8] c"--bench\00", align 1

declare i32 @printf(ptr, ...)
declare i32 @strcmp(ptr, ptr)
declare i32 @clock_gettime(i32, ptr)

define void @ut_check(i1 %cond, ptr %msg) {
entry:
  %c = load i64, ptr @ut.checks, align 8
  %c.next = add nuw i64 %c, 1
  store i64 %c.next, ptr @ut.checks, align 8
  br i1 %cond, label %pass, label %fail, !prof !0

fail:
  %f = load i64, ptr @ut.failures, align 8
  %f.next = add nuw i64 %f, 1
  store i64 %f.next, ptr @ut.failures, align 8
  %r = call i32 (ptr, ...) @printf(ptr @ut.fmt.fail, ptr %msg)
  br label %pass

pass:
  ret void
}

define void @ut_check_eq(i64 %a, i64 %b, ptr %msg) {
entry:
  %c = load i64, ptr @ut.checks, align 8
  %c.next = add nuw i64 %c, 1
  store i64 %c.next, ptr @ut.checks, align 8
  %eq = icmp eq i64 %a, %b
  br i1 %eq, label %pass, label %fail, !prof !0

fail:
  %f = load i64, ptr @ut.failures, align 8
  %f.next = add nuw i64 %f, 1
  store i64 %f.next, ptr @ut.failures, align 8
  %r = call i32 (ptr, ...) @printf(ptr @ut.fmt.faileq, ptr %msg, i64 %a, i64 %b)
  br label %pass

pass:
  ret void
}

; MMIX LCG; deterministic per fixed seed. Returns state >> 8 (better bits).
define i64 @ut_rand(ptr captures(none) %state) {
entry:
  %s = load i64, ptr %state, align 8
  %m = mul i64 %s, 6364136223846793005
  %n = add i64 %m, 1442695040888963407
  store i64 %n, ptr %state, align 8
  %r = lshr i64 %n, 8
  ret i64 %r
}

define double @ut_now_sec() {
entry:
  %ts = alloca { i64, i64 }, align 8
  %rc = call i32 @clock_gettime(i32 0, ptr nonnull %ts)
  %sec = load i64, ptr %ts, align 8
  %nsec.ptr = getelementptr inbounds nuw i8, ptr %ts, i64 8
  %nsec = load i64, ptr %nsec.ptr, align 8
  %sec.f = sitofp i64 %sec to double
  %nsec.f = sitofp i64 %nsec to double
  %nsec.s = fmul double %nsec.f, 1.000000e-09
  %t = fadd double %sec.f, %nsec.s
  ret double %t
}

; ut_report_dist(samples, n, ops, label): given N per-repetition elapsed times
; (seconds, in %samples), sort them and print the DISTRIBUTION as ns/op —
; min, p50, p95, p99 — per the bench-rigor convention (a single number hides
; variance; p99 is what a caller feels). Sorts %samples in place (insertion
; sort; N is small). Discard the warm-up rep before calling this.
@ut.fmt.dist = private unnamed_addr constant [65 x i8] c"BENCH %s: min=%.2f p50=%.2f p95=%.2f p99=%.2f ns/op (%lld reps)\0A\00"

define void @ut_report_dist(ptr %samples, i64 %n, i64 %ops, ptr %label) {
entry:
  %empty = icmp ult i64 %n, 1
  br i1 %empty, label %ret, label %isort.pre

isort.pre:
  %one = icmp eq i64 %n, 1
  br i1 %one, label %pct, label %isort.outer

isort.outer:
  %i = phi i64 [ 1, %isort.pre ], [ %i.next, %isort.cont ]
  %keyp = getelementptr inbounds double, ptr %samples, i64 %i
  %key = load double, ptr %keyp, align 8
  %jm1 = sub i64 %i, 1
  br label %isort.inner

isort.inner:
  %j = phi i64 [ %jm1, %isort.outer ], [ %j.next, %isort.shift ]
  %jok = icmp sge i64 %j, 0
  br i1 %jok, label %isort.cmp, label %isort.place

isort.cmp:
  %ajp = getelementptr inbounds double, ptr %samples, i64 %j
  %aj = load double, ptr %ajp, align 8
  %gt = fcmp ogt double %aj, %key
  br i1 %gt, label %isort.shift, label %isort.place

isort.shift:
  %jp1 = add i64 %j, 1
  %ajp1 = getelementptr inbounds double, ptr %samples, i64 %jp1
  store double %aj, ptr %ajp1, align 8
  %j.next = sub i64 %j, 1
  br label %isort.inner

isort.place:
  %jplace = add i64 %j, 1
  %placep = getelementptr inbounds double, ptr %samples, i64 %jplace
  store double %key, ptr %placep, align 8
  br label %isort.cont

isort.cont:
  %i.next = add i64 %i, 1
  %imore = icmp ult i64 %i.next, %n
  br i1 %imore, label %isort.outer, label %pct

pct:
  %nm1 = sub i64 %n, 1
  %i50 = udiv i64 %n, 2
  %t95 = mul i64 %n, 95
  %i95 = udiv i64 %t95, 100
  %t99 = mul i64 %n, 99
  %i99 = udiv i64 %t99, 100
  %pmin = load double, ptr %samples, align 8
  %p50p = getelementptr inbounds double, ptr %samples, i64 %i50
  %p50 = load double, ptr %p50p, align 8
  %p95p = getelementptr inbounds double, ptr %samples, i64 %i95
  %p95 = load double, ptr %p95p, align 8
  %p99p = getelementptr inbounds double, ptr %samples, i64 %i99
  %p99 = load double, ptr %p99p, align 8
  ; seconds -> ns/op:  v * (1e9 / ops)
  %opsf = uitofp i64 %ops to double
  %scale = fdiv double 1.000000e+09, %opsf
  %nmin = fmul double %pmin, %scale
  %n50 = fmul double %p50, %scale
  %n95 = fmul double %p95, %scale
  %n99 = fmul double %p99, %scale
  %pr = call i32 (ptr, ...) @printf(ptr @ut.fmt.dist, ptr %label, double %nmin, double %n50, double %n95, double %n99, i64 %n)
  br label %ret

ret:
  ret void
}

define i1 @ut_want_bench(i32 %argc, ptr %argv) {
entry:
  %has.arg = icmp sgt i32 %argc, 1
  br i1 %has.arg, label %check, label %no

check:
  %arg1.ptr = getelementptr inbounds nuw i8, ptr %argv, i64 8
  %arg1 = load ptr, ptr %arg1.ptr, align 8
  %r = call i32 @strcmp(ptr %arg1, ptr @ut.str.bench)
  %eq = icmp eq i32 %r, 0
  ret i1 %eq

no:
  ret i1 false
}

define i32 @ut_summary() {
entry:
  %f = load i64, ptr @ut.failures, align 8
  %c = load i64, ptr @ut.checks, align 8
  %bad = icmp ne i64 %f, 0
  br i1 %bad, label %failed, label %passed, !prof !1

failed:
  %r0 = call i32 (ptr, ...) @printf(ptr @ut.fmt.bad, i64 %f, i64 %c)
  ret i32 1

passed:
  %r1 = call i32 (ptr, ...) @printf(ptr @ut.fmt.ok, i64 %c)
  ret i32 0
}

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
