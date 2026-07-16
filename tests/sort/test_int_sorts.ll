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

; Tests for universe_sort_{counting, radix} (i32 arrays): negatives included,
; qsort cross-check, counting range guard, radix sign pass, --bench.

declare i32 @universe_sort_counting(ptr, i64)
declare i32 @universe_sort_radix(ptr, i64)

declare void @qsort(ptr, i64, i64, ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()

@g.master = internal global [65536 x i32] zeroinitializer, align 16
@g.work   = internal global [65536 x i32] zeroinitializer, align 16
@g.ref    = internal global [65536 x i32] zeroinitializer, align 16

@m.count.small = private unnamed_addr constant [28 x i8] c"counting small-range random\00"
@m.count.neg   = private unnamed_addr constant [24 x i8] c"counting negatives sort\00"
@m.count.guard = private unnamed_addr constant [26 x i8] c"counting wide range -> 8 \00"
@m.radix.full  = private unnamed_addr constant [25 x i8] c"radix full-range + signs\00"
@m.radix.smallr = private unnamed_addr constant [25 x i8] c"radix small-range random\00"
@m.errs        = private unnamed_addr constant [15 x i8] c"null args -> 1\00"
@m.edge.tiny   = private unnamed_addr constant [9 x i8]  c"sort n=1\00"
@m.edge.eq     = private unnamed_addr constant [15 x i8] c"sort all-equal\00"
@m.edge.asc    = private unnamed_addr constant [15 x i8] c"sort presorted\00"
@m.edge.rev    = private unnamed_addr constant [17 x i8] c"sort reverse+neg\00"
@m.edge.sparse = private unnamed_addr constant [17 x i8] c"sort sparse-wide\00"
@fmt.bench = private unnamed_addr constant [53 x i8] c"bench n=65536: radix=%.3fms qsort=%.3fms (i32 keys)\0A\00"
@sort.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.radix = private unnamed_addr constant [18 x i8] c"radix i32 n=65536\00"
@lbl.qsort = private unnamed_addr constant [18 x i8] c"qsort i32 n=65536\00"

define i32 @cmp_i32(ptr %a, ptr %b) {
entry:
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %gt = icmp sgt i32 %x, %y
  %lt = icmp slt i32 %x, %y
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

; fill master with n values in [lo, lo+span), seed fixed
define internal void @fill_master(i64 %n, i64 %span, i64 %lo, i64 %seedval) {
entry:
  %seed = alloca i64, align 8
  store i64 %seedval, ptr %seed, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %m = urem i64 %r, %span
  %v64 = add i64 %m, %lo
  %v = trunc i64 %v64 to i32
  %p = getelementptr inbounds nuw [65536 x i32], ptr @g.master, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; fill master[0..n) with base + i*step (step 0 = all-equal, +1 = presorted,
; -1 = reverse). Used for the small/edge cases that the large random fills miss.
define internal void @fill_ramp(i64 %n, i32 %base, i32 %step) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %i32 = trunc i64 %i to i32
  %off = mul i32 %i32, %step
  %v = add i32 %base, %off
  %p = getelementptr inbounds nuw [65536 x i32], ptr @g.master, i64 0, i64 %i
  store i32 %v, ptr %p, align 4
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; run %fn(work, n) over master copy; qsort ref; compare
define internal void @check_fn(ptr %fn, i64 %n, ptr %msg) {
entry:
  %bytes = shl i64 %n, 2
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 %bytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.ref, ptr align 16 @g.master, i64 %bytes, i1 false)
  %rc = call i32 %fn(ptr @g.work, i64 %n)
  call void @qsort(ptr @g.ref, i64 %n, i64 4, ptr @cmp_i32)
  %mc = call i32 @memcmp(ptr @g.work, ptr @g.ref, i64 %bytes)
  %rc.ok = icmp eq i32 %rc, 0
  %mc.ok = icmp eq i32 %mc, 0
  %ok = and i1 %rc.ok, %mc.ok
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; --- edge/tiny inputs the big random fills never reach (hazard #18: buckets
  ;     sized to KEY DOMAIN not N; only trips on small n / wide keys). Every
  ;     case is cross-checked against qsort for BOTH counting and radix. ---
  ; n=1
  call void @fill_ramp(i64 1, i32 42, i32 0)
  call void @check_fn(ptr @universe_sort_counting, i64 1, ptr @m.edge.tiny)
  call void @check_fn(ptr @universe_sort_radix,    i64 1, ptr @m.edge.tiny)
  ; all-equal (range = 1)
  call void @fill_ramp(i64 5, i32 7, i32 0)
  call void @check_fn(ptr @universe_sort_counting, i64 5, ptr @m.edge.eq)
  call void @check_fn(ptr @universe_sort_radix,    i64 5, ptr @m.edge.eq)
  ; presorted ascending 0..7
  call void @fill_ramp(i64 8, i32 0, i32 1)
  call void @check_fn(ptr @universe_sort_counting, i64 8, ptr @m.edge.asc)
  call void @check_fn(ptr @universe_sort_radix,    i64 8, ptr @m.edge.asc)
  ; reverse with negatives: 2,1,0,-1,-2,-3
  call void @fill_ramp(i64 6, i32 2, i32 -1)
  call void @check_fn(ptr @universe_sort_counting, i64 6, ptr @m.edge.rev)
  call void @check_fn(ptr @universe_sort_radix,    i64 6, ptr @m.edge.rev)
  ; small n, wide keys (key domain >> N): 0,50,100,150 — the hazard-#18 shape
  call void @fill_ramp(i64 4, i32 0, i32 50)
  call void @check_fn(ptr @universe_sort_counting, i64 4, ptr @m.edge.sparse)
  call void @check_fn(ptr @universe_sort_radix,    i64 4, ptr @m.edge.sparse)

  ; counting: small range incl negatives
  call void @fill_master(i64 20000, i64 2000, i64 -1000, i64 42)
  call void @check_fn(ptr @universe_sort_counting, i64 20000, ptr @m.count.small)
  call void @check_fn(ptr @universe_sort_radix, i64 20000, ptr @m.radix.smallr)

  ; all-negative band
  call void @fill_master(i64 5000, i64 500, i64 -100000, i64 7)
  call void @check_fn(ptr @universe_sort_counting, i64 5000, ptr @m.count.neg)

  ; counting range guard: two extreme values
  %p0 = getelementptr inbounds nuw [65536 x i32], ptr @g.work, i64 0, i64 0
  store i32 -2147483648, ptr %p0, align 4
  %p1 = getelementptr inbounds nuw [65536 x i32], ptr @g.work, i64 0, i64 1
  store i32 2147483647, ptr %p1, align 4
  %grc = call i32 @universe_sort_counting(ptr @g.work, i64 2)
  %grc.w = zext i32 %grc to i64
  call void @ut_check_eq(i64 %grc.w, i64 8, ptr @m.count.guard)

  ; radix: full 32-bit range with signs
  call void @fill_master(i64 65536, i64 4294967296, i64 -2147483648, i64 99)
  call void @check_fn(ptr @universe_sort_radix, i64 65536, ptr @m.radix.full)

  %e1 = call i32 @universe_sort_counting(ptr null, i64 4)
  %e2 = call i32 @universe_sort_radix(ptr null, i64 4)
  %es = add nuw i32 %e1, %e2
  %es.w = zext i32 %es to i64
  call void @ut_check_eq(i64 %es.w, i64 2, ptr @m.errs)

  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %finish

bench:
  ; 17 reps each of (refill 65536 keys, time one sort); discard rep 0 (warm-up),
  ; report the distribution over the remaining 16. The sort is destructive, so
  ; each rep re-copies the master. ops/rep = 65536 (ns per key).
  call void @fill_master(i64 65536, i64 4294967296, i64 -2147483648, i64 1)
  br label %br.rep

br.rep:
  %rrep = phi i64 [ 0, %bench ], [ %rrep.n, %br.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 262144, i1 false)
  %rt0 = call double @ut_now_sec()
  %rrc = call i32 @universe_sort_radix(ptr @g.work, i64 65536)
  %rt1 = call double @ut_now_sec()
  %rel = fsub double %rt1, %rt0
  %rkeep = icmp ugt i64 %rrep, 0
  br i1 %rkeep, label %br.store, label %br.next

br.store:
  %ridx = sub i64 %rrep, 1
  %rsp = getelementptr inbounds [16 x double], ptr @sort.samp, i64 0, i64 %ridx
  store double %rel, ptr %rsp, align 8
  br label %br.next

br.next:
  %rrep.n = add nuw i64 %rrep, 1
  %rmore = icmp ult i64 %rrep.n, 17
  br i1 %rmore, label %br.rep, label %br.report

br.report:
  call void @ut_report_dist(ptr @sort.samp, i64 16, i64 65536, ptr @lbl.radix)
  ; qsort baseline over the same distribution
  br label %bq.rep

bq.rep:
  %qrep = phi i64 [ 0, %br.report ], [ %qrep.n, %bq.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @g.work, ptr align 16 @g.master, i64 262144, i1 false)
  %qt0 = call double @ut_now_sec()
  call void @qsort(ptr @g.work, i64 65536, i64 4, ptr @cmp_i32)
  %qt1 = call double @ut_now_sec()
  %qel = fsub double %qt1, %qt0
  %qkeep = icmp ugt i64 %qrep, 0
  br i1 %qkeep, label %bq.store, label %bq.next

bq.store:
  %qidx = sub i64 %qrep, 1
  %qsp = getelementptr inbounds [16 x double], ptr @sort.samp, i64 0, i64 %qidx
  store double %qel, ptr %qsp, align 8
  br label %bq.next

bq.next:
  %qrep.n = add nuw i64 %qrep, 1
  %qmore = icmp ult i64 %qrep.n, 17
  br i1 %qmore, label %bq.rep, label %bq.report

bq.report:
  call void @ut_report_dist(ptr @sort.samp, i64 16, i64 65536, ptr @lbl.qsort)
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
