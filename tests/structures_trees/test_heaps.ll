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

; Tests for the heap family (d-ary, radix/monotone, pairing): happy paths,
; every reachable error code, edge sizes, capacity growth, fixed-seed random
; runs cross-checked against libc qsort, decrease-key / meld semantics, and a
; --bench comparing d-ary vs the binary heap and radix on a monotone workload.

; ---- d-ary heap ----
declare ptr @universe_ds_dheap_create(i64, i64)
declare void @universe_ds_dheap_destroy(ptr)
declare i32 @universe_ds_dheap_push(ptr, i64)
declare i32 @universe_ds_dheap_pop(ptr, ptr)
declare i32 @universe_ds_dheap_peek(ptr, ptr)
declare i64 @universe_ds_dheap_len(ptr)
; ---- radix heap ----
declare ptr @universe_ds_radixheap_create(i64)
declare void @universe_ds_radixheap_destroy(ptr)
declare i32 @universe_ds_radixheap_push(ptr, i64, i64)
declare i32 @universe_ds_radixheap_pop(ptr, ptr, ptr)
declare i32 @universe_ds_radixheap_peek(ptr, ptr)
declare i64 @universe_ds_radixheap_len(ptr)
; ---- pairing heap ----
declare ptr @universe_ds_pairing_create()
declare void @universe_ds_pairing_destroy(ptr)
declare i64 @universe_ds_pairing_push(ptr, i64, i64)
declare i32 @universe_ds_pairing_pop(ptr, ptr, ptr)
declare i32 @universe_ds_pairing_peek(ptr, ptr, ptr)
declare i32 @universe_ds_pairing_decrease_key(ptr, i64, i64)
declare i32 @universe_ds_pairing_meld(ptr, ptr)
declare i64 @universe_ds_pairing_len(ptr)
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @qsort(ptr, i64, i64, ptr)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@dh1.bench.samp = internal global [16 x double] zeroinitializer, align 8
@bh.bench.samp  = internal global [16 x double] zeroinitializer, align 8
@rx.bench.samp  = internal global [16 x double] zeroinitializer, align 8
@dh2.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.dh1.bench = private unnamed_addr constant [21 x i8] c"dheap4 push+pop 500k\00"
@lbl.bh.bench  = private unnamed_addr constant [22 x i8] c"binheap push+pop 500k\00"
@lbl.rx.bench  = private unnamed_addr constant [24 x i8] c"radixheap monotone 500k\00"
@lbl.dh2.bench = private unnamed_addr constant [21 x i8] c"dheap4 monotone 500k\00"

@m.dheap    = private unnamed_addr constant [22 x i8] c"dheap sort crosscheck\00"
@m.dheap.pk = private unnamed_addr constant [16 x i8] c"dheap peek==min\00"
@m.dheap.er = private unnamed_addr constant [16 x i8] c"dheap err/edges\00"
@m.dheap.d8 = private unnamed_addr constant [17 x i8] c"dheap d=8 sorted\00"
@m.rh       = private unnamed_addr constant [26 x i8] c"radixheap sort crosscheck\00"
@m.rh.mono  = private unnamed_addr constant [22 x i8] c"radixheap monotone dj\00"
@m.rh.er    = private unnamed_addr constant [19 x i8] c"radixheap err/edge\00"
@m.pr       = private unnamed_addr constant [24 x i8] c"pairing sort crosscheck\00"
@m.pr.dk    = private unnamed_addr constant [21 x i8] c"pairing decrease-key\00"
@m.pr.meld  = private unnamed_addr constant [13 x i8] c"pairing meld\00"
@m.pr.er    = private unnamed_addr constant [17 x i8] c"pairing err/edge\00"

define internal i32 @cmp_i64(ptr %a, ptr %b) {
entry:
  %av = load i64, ptr %a, align 8
  %bv = load i64, ptr %b, align 8
  %lt = icmp slt i64 %av, %bv
  %gt = icmp sgt i64 %av, %bv
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub i32 %g, %l
  ret i32 %r
}

; ===========================================================================
; d-ary heap
; ===========================================================================
define internal void @test_dheap() {
entry:
  %state = alloca i64, align 8
  store i64 11400714819323198485, ptr %state, align 8
  %n = add i64 0, 2000
  %ref = call ptr @malloc(i64 16000)   ; 2000 i64
  %out = call ptr @malloc(i64 16000)
  %h = call ptr @universe_ds_dheap_create(i64 8, i64 0)  ; d=0 -> default 4
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr %state)
  %k = and i64 %r, 1152921504606846975   ; keep positive-ish, arbitrary
  %rp = getelementptr inbounds i64, ptr %ref, i64 %i
  store i64 %k, ptr %rp, align 8
  call i32 @universe_ds_dheap_push(ptr %h, i64 %k)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %peekchk

peekchk:
  ; sort ref; min should equal peek
  call void @qsort(ptr %ref, i64 %n, i64 8, ptr @cmp_i64)
  %pk = alloca i64, align 8
  call i32 @universe_ds_dheap_peek(ptr %h, ptr %pk)
  %pkv = load i64, ptr %pk, align 8
  %ref0 = load i64, ptr %ref, align 8
  %pkok = icmp eq i64 %pkv, %ref0
  call void @ut_check(i1 %pkok, ptr @m.dheap.pk)
  br label %drain

drain:
  %j = phi i64 [ 0, %peekchk ], [ %j.n, %drain ]
  %op = getelementptr inbounds i64, ptr %out, i64 %j
  call i32 @universe_ds_dheap_pop(ptr %h, ptr %op)
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %n
  br i1 %more2, label %drain, label %verify

verify:
  br label %vloop

vloop:
  %v = phi i64 [ 0, %verify ], [ %v.n, %vloop ]
  %viol = phi i64 [ 0, %verify ], [ %viol.f, %vloop ]
  %ovp = getelementptr inbounds i64, ptr %out, i64 %v
  %ov = load i64, ptr %ovp, align 8
  %rvp = getelementptr inbounds i64, ptr %ref, i64 %v
  %rv = load i64, ptr %rvp, align 8
  %bad = icmp ne i64 %ov, %rv
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %v.n = add nuw i64 %v, 1
  %more3 = icmp ult i64 %v.n, %n
  br i1 %more3, label %vloop, label %vdone

vdone:
  call void @ut_check_eq(i64 %viol.f, i64 0, ptr @m.dheap)
  call void @universe_ds_dheap_destroy(ptr %h)
  call void @free(ptr %ref)
  call void @free(ptr %out)
  ret void
}

; d=8 variant: smaller run, confirm sorted output
define internal void @test_dheap_d8() {
entry:
  %state = alloca i64, align 8
  store i64 987654321123456789, ptr %state, align 8
  %n = add i64 0, 500
  %ref = call ptr @malloc(i64 4000)
  %out = call ptr @malloc(i64 4000)
  %h = call ptr @universe_ds_dheap_create(i64 8, i64 8)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr %state)
  %k = urem i64 %r, 100000
  %rp = getelementptr inbounds i64, ptr %ref, i64 %i
  store i64 %k, ptr %rp, align 8
  call i32 @universe_ds_dheap_push(ptr %h, i64 %k)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %sort
sort:
  call void @qsort(ptr %ref, i64 %n, i64 8, ptr @cmp_i64)
  br label %drain
drain:
  %j = phi i64 [ 0, %sort ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %sort ], [ %viol.f, %drain ]
  %op = alloca i64, align 8
  call i32 @universe_ds_dheap_pop(ptr %h, ptr %op)
  %ov = load i64, ptr %op, align 8
  %rvp = getelementptr inbounds i64, ptr %ref, i64 %j
  %rv = load i64, ptr %rvp, align 8
  %bad = icmp ne i64 %ov, %rv
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %n
  br i1 %more2, label %drain, label %done
done:
  call void @ut_check_eq(i64 %viol.f, i64 0, ptr @m.dheap.d8)
  call void @universe_ds_dheap_destroy(ptr %h)
  call void @free(ptr %ref)
  call void @free(ptr %out)
  ret void
}

define internal void @test_dheap_errors() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %slot = alloca i64, align 8
  ; null handle
  %e.pushn = call i32 @universe_ds_dheap_push(ptr null, i64 1)
  %e.popn = call i32 @universe_ds_dheap_pop(ptr null, ptr %slot)
  %b1 = icmp ne i32 %e.pushn, 1
  %b2 = icmp ne i32 %e.popn, 1
  call void @bump_if(i1 %b1, ptr %viol)
  call void @bump_if(i1 %b2, ptr %viol)
  ; empty heap
  %h = call ptr @universe_ds_dheap_create(i64 8, i64 4)
  %e.empty = call i32 @universe_ds_dheap_pop(ptr %h, ptr %slot)
  %e.peek0 = call i32 @universe_ds_dheap_peek(ptr %h, ptr %slot)
  %b3 = icmp ne i32 %e.empty, 4
  %b4 = icmp ne i32 %e.peek0, 4
  call void @bump_if(i1 %b3, ptr %viol)
  call void @bump_if(i1 %b4, ptr %viol)
  ; single element
  call i32 @universe_ds_dheap_push(ptr %h, i64 42)
  %e.pop1 = call i32 @universe_ds_dheap_pop(ptr %h, ptr %slot)
  %v1 = load i64, ptr %slot, align 8
  %b5 = icmp ne i64 %v1, 42
  call void @bump_if(i1 %b5, ptr %viol)
  ; two elements out of order -> sorted
  call i32 @universe_ds_dheap_push(ptr %h, i64 7)
  call i32 @universe_ds_dheap_push(ptr %h, i64 3)
  %pa = alloca i64, align 8
  %pb = alloca i64, align 8
  call i32 @universe_ds_dheap_pop(ptr %h, ptr %pa)
  call i32 @universe_ds_dheap_pop(ptr %h, ptr %pb)
  %va = load i64, ptr %pa, align 8
  %vb = load i64, ptr %pb, align 8
  %b6 = icmp ne i64 %va, 3
  %b7 = icmp ne i64 %vb, 7
  call void @bump_if(i1 %b6, ptr %viol)
  call void @bump_if(i1 %b7, ptr %viol)
  ; growth across capacity: push 100 into cap-8 heap, len==100
  br label %grow
grow:
  %g = phi i64 [ 0, %entry ], [ %g.n, %grow ]
  call i32 @universe_ds_dheap_push(ptr %h, i64 %g)
  %g.n = add nuw i64 %g, 1
  %gm = icmp ult i64 %g.n, 100
  br i1 %gm, label %grow, label %glen
glen:
  %ln = call i64 @universe_ds_dheap_len(ptr %h)
  %b8 = icmp ne i64 %ln, 100
  call void @bump_if(i1 %b8, ptr %viol)
  call void @universe_ds_dheap_destroy(ptr %h)
  call void @universe_ds_dheap_destroy(ptr null)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.dheap.er)
  ret void
}

; ===========================================================================
; radix / monotone heap
; ===========================================================================
define internal void @test_radix() {
entry:
  %state = alloca i64, align 8
  store i64 1234567891011121314, ptr %state, align 8
  %n = add i64 0, 2000
  %ref = call ptr @malloc(i64 16000)
  %out = call ptr @malloc(i64 16000)
  %h = call ptr @universe_ds_radixheap_create(i64 16)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr %state)
  %k = urem i64 %r, 1000000          ; non-negative bounded weights
  %rp = getelementptr inbounds i64, ptr %ref, i64 %i
  store i64 %k, ptr %rp, align 8
  call i32 @universe_ds_radixheap_push(ptr %h, i64 %k, i64 %i)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %sort
sort:
  call void @qsort(ptr %ref, i64 %n, i64 8, ptr @cmp_i64)
  br label %drain
drain:
  %j = phi i64 [ 0, %sort ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %sort ], [ %viol.f, %drain ]
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  call i32 @universe_ds_radixheap_pop(ptr %h, ptr %ok, ptr %ov)
  %kv = load i64, ptr %ok, align 8
  %rvp = getelementptr inbounds i64, ptr %ref, i64 %j
  %rv = load i64, ptr %rvp, align 8
  %bad = icmp ne i64 %kv, %rv
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %n
  br i1 %more2, label %drain, label %done
done:
  call void @ut_check_eq(i64 %viol.f, i64 0, ptr @m.rh)
  call void @universe_ds_radixheap_destroy(ptr %h)
  call void @free(ptr %ref)
  call void @free(ptr %out)
  ret void
}

; Dijkstra-like monotone workload: pop a node, relax by pushing key+weight
; (always >= last popped). Verify pops are non-decreasing.
define internal void @test_radix_monotone() {
entry:
  %state = alloca i64, align 8
  store i64 555555555512345678, ptr %state, align 8
  %h = call ptr @universe_ds_radixheap_create(i64 16)
  ; seed with a few keys
  call i32 @universe_ds_radixheap_push(ptr %h, i64 0, i64 0)
  call i32 @universe_ds_radixheap_push(ptr %h, i64 5, i64 1)
  call i32 @universe_ds_radixheap_push(ptr %h, i64 3, i64 2)
  %prev = alloca i64, align 8
  store i64 0, ptr %prev, align 8
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  br label %loop
loop:
  %pushed = phi i64 [ 0, %entry ], [ %pushed.n, %relax ], [ %pushed, %check ]
  %r = call i32 @universe_ds_radixheap_pop(ptr %h, ptr %ok, ptr %ov)
  %empty = icmp eq i32 %r, 4
  br i1 %empty, label %fin, label %check
check:
  %kv = load i64, ptr %ok, align 8
  %pv = load i64, ptr %prev, align 8
  %bad = icmp slt i64 %kv, %pv           ; must be non-decreasing
  %inc = zext i1 %bad to i64
  %vv = load i64, ptr %viol, align 8
  %vv.n = add i64 %vv, %inc
  store i64 %vv.n, ptr %viol, align 8
  store i64 %kv, ptr %prev, align 8
  ; relax: push up to 200 new keys total, each = kv + weight
  %cap = icmp uge i64 %pushed, 200
  br i1 %cap, label %loop, label %relax
relax:
  %w0 = call i64 @ut_rand(ptr %state)
  %w = urem i64 %w0, 10                   ; weight 0..9
  %nk = add i64 %kv, %w                   ; >= kv >= last -> monotone-safe
  call i32 @universe_ds_radixheap_push(ptr %h, i64 %nk, i64 %pushed)
  %pushed.n = add nuw i64 %pushed, 1
  br label %loop
fin:
  %fv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %fv, i64 0, ptr @m.rh.mono)
  call void @universe_ds_radixheap_destroy(ptr %h)
  ret void
}

define internal void @test_radix_errors() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  ; null
  %e.pn = call i32 @universe_ds_radixheap_push(ptr null, i64 1, i64 0)
  %b1 = icmp ne i32 %e.pn, 1
  call void @bump_if(i1 %b1, ptr %viol)
  %h = call ptr @universe_ds_radixheap_create(i64 16)
  ; empty pop / peek
  %e.emp = call i32 @universe_ds_radixheap_pop(ptr %h, ptr %ok, ptr %ov)
  %e.pk0 = call i32 @universe_ds_radixheap_peek(ptr %h, ptr %ok)
  %b2 = icmp ne i32 %e.emp, 4
  %b3 = icmp ne i32 %e.pk0, 4
  call void @bump_if(i1 %b2, ptr %viol)
  call void @bump_if(i1 %b3, ptr %viol)
  ; push then peek == min
  call i32 @universe_ds_radixheap_push(ptr %h, i64 10, i64 0)
  call i32 @universe_ds_radixheap_push(ptr %h, i64 4, i64 0)
  call i32 @universe_ds_radixheap_push(ptr %h, i64 7, i64 0)
  %e.pk = call i32 @universe_ds_radixheap_peek(ptr %h, ptr %ok)
  %pkv = load i64, ptr %ok, align 8
  %b4 = icmp ne i64 %pkv, 4
  call void @bump_if(i1 %b4, ptr %viol)
  ; pop the 4, raising last to 4; pushing 2 (<last) must return 8
  call i32 @universe_ds_radixheap_pop(ptr %h, ptr %ok, ptr %ov)
  %e.arg = call i32 @universe_ds_radixheap_push(ptr %h, i64 2, i64 0)
  %b5 = icmp ne i32 %e.arg, 8
  call void @bump_if(i1 %b5, ptr %viol)
  call void @universe_ds_radixheap_destroy(ptr %h)
  call void @universe_ds_radixheap_destroy(ptr null)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.rh.er)
  ret void
}

; ===========================================================================
; pairing heap
; ===========================================================================
define internal void @test_pairing() {
entry:
  %state = alloca i64, align 8
  store i64 246813579111315171, ptr %state, align 8
  %n = add i64 0, 2000
  %ref = call ptr @malloc(i64 16000)
  %h = call ptr @universe_ds_pairing_create()
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r = call i64 @ut_rand(ptr %state)
  %k = and i64 %r, 1152921504606846975
  %rp = getelementptr inbounds i64, ptr %ref, i64 %i
  store i64 %k, ptr %rp, align 8
  %hnd = call i64 @universe_ds_pairing_push(ptr %h, i64 %k, i64 %i)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %peek
peek:
  call void @qsort(ptr %ref, i64 %n, i64 8, ptr @cmp_i64)
  %pk = alloca i64, align 8
  call i32 @universe_ds_pairing_peek(ptr %h, ptr %pk, ptr null)
  %pkv = load i64, ptr %pk, align 8
  %ref0 = load i64, ptr %ref, align 8
  %pkok = icmp eq i64 %pkv, %ref0
  call void @bump_notok(i1 %pkok)
  br label %drain
drain:
  %j = phi i64 [ 0, %peek ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %peek ], [ %viol.f, %drain ]
  %ok = alloca i64, align 8
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr null)
  %kv = load i64, ptr %ok, align 8
  %rvp = getelementptr inbounds i64, ptr %ref, i64 %j
  %rv = load i64, ptr %rvp, align 8
  %bad = icmp ne i64 %kv, %rv
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %n
  br i1 %more2, label %drain, label %done
done:
  call void @ut_check_eq(i64 %viol.f, i64 0, ptr @m.pr)
  call void @universe_ds_pairing_destroy(ptr %h)
  call void @free(ptr %ref)
  ret void
}

define internal void @test_pairing_decrease() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %h = call ptr @universe_ds_pairing_create()
  ; push keys 100,200,300,400,500 keep handle of the 500
  %h1 = call i64 @universe_ds_pairing_push(ptr %h, i64 100, i64 1)
  %h2 = call i64 @universe_ds_pairing_push(ptr %h, i64 200, i64 2)
  %h3 = call i64 @universe_ds_pairing_push(ptr %h, i64 300, i64 3)
  %h4 = call i64 @universe_ds_pairing_push(ptr %h, i64 400, i64 4)
  %h5 = call i64 @universe_ds_pairing_push(ptr %h, i64 500, i64 5)
  ; decrease 500 -> 50 (now global min)
  %dr = call i32 @universe_ds_pairing_decrease_key(ptr %h, i64 %h5, i64 50)
  %b0 = icmp ne i32 %dr, 0
  call void @bump_if(i1 %b0, ptr %viol)
  ; increasing key must fail (8)
  %dr2 = call i32 @universe_ds_pairing_decrease_key(ptr %h, i64 %h1, i64 999)
  %b1 = icmp ne i32 %dr2, 8
  call void @bump_if(i1 %b1, ptr %viol)
  ; pop order should be 50,100,200,300,400
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov)
  %p0 = load i64, ptr %ok, align 8
  %v0 = load i64, ptr %ov, align 8
  %e0 = icmp ne i64 %p0, 50
  %ev0 = icmp ne i64 %v0, 5      ; payload of the decreased node
  call void @bump_if(i1 %e0, ptr %viol)
  call void @bump_if(i1 %ev0, ptr %viol)
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov)
  %p1 = load i64, ptr %ok, align 8
  %e1 = icmp ne i64 %p1, 100
  call void @bump_if(i1 %e1, ptr %viol)
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov)
  %p2 = load i64, ptr %ok, align 8
  %e2 = icmp ne i64 %p2, 200
  call void @bump_if(i1 %e2, ptr %viol)
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov)
  %p3 = load i64, ptr %ok, align 8
  %e3 = icmp ne i64 %p3, 300
  call void @bump_if(i1 %e3, ptr %viol)
  call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr %ov)
  %p4 = load i64, ptr %ok, align 8
  %e4 = icmp ne i64 %p4, 400
  call void @bump_if(i1 %e4, ptr %viol)
  call void @universe_ds_pairing_destroy(ptr %h)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.pr.dk)
  ret void
}

define internal void @test_pairing_meld() {
entry:
  %state = alloca i64, align 8
  store i64 314159265358979323, ptr %state, align 8
  %n = add i64 0, 1000
  %ref = call ptr @malloc(i64 16000)   ; up to 2000 combined
  %a = call ptr @universe_ds_pairing_create()
  %b = call ptr @universe_ds_pairing_create()
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %r1 = call i64 @ut_rand(ptr %state)
  %k1 = urem i64 %r1, 1000000
  %r2 = call i64 @ut_rand(ptr %state)
  %k2 = urem i64 %r2, 1000000
  %i2 = mul i64 %i, 2
  %rp1 = getelementptr inbounds i64, ptr %ref, i64 %i2
  store i64 %k1, ptr %rp1, align 8
  %i2p1 = add i64 %i2, 1
  %rp2 = getelementptr inbounds i64, ptr %ref, i64 %i2p1
  store i64 %k2, ptr %rp2, align 8
  %ha = call i64 @universe_ds_pairing_push(ptr %a, i64 %k1, i64 0)
  %hb = call i64 @universe_ds_pairing_push(ptr %b, i64 %k2, i64 0)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %meld
meld:
  %total = mul i64 %n, 2
  call void @qsort(ptr %ref, i64 %total, i64 8, ptr @cmp_i64)
  %mr = call i32 @universe_ds_pairing_meld(ptr %a, ptr %b)
  ; b now empty, a has all 2000
  %lenb = call i64 @universe_ds_pairing_len(ptr %b)
  %lena = call i64 @universe_ds_pairing_len(ptr %a)
  %bad.lb = icmp ne i64 %lenb, 0
  %bad.la = icmp ne i64 %lena, %total
  br label %drain
drain:
  %j = phi i64 [ 0, %meld ], [ %j.n, %drain ]
  %viol = phi i64 [ 0, %meld ], [ %viol.f, %drain ]
  %ok = alloca i64, align 8
  call i32 @universe_ds_pairing_pop(ptr %a, ptr %ok, ptr null)
  %kv = load i64, ptr %ok, align 8
  %rvp = getelementptr inbounds i64, ptr %ref, i64 %j
  %rv = load i64, ptr %rvp, align 8
  %bad = icmp ne i64 %kv, %rv
  %inc = zext i1 %bad to i64
  %viol.f = add i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, %total
  br i1 %more2, label %drain, label %done
done:
  %lbc = zext i1 %bad.lb to i64
  %lac = zext i1 %bad.la to i64
  %s1 = add i64 %viol.f, %lbc
  %s2 = add i64 %s1, %lac
  call void @ut_check_eq(i64 %s2, i64 0, ptr @m.pr.meld)
  call void @universe_ds_pairing_destroy(ptr %a)
  call void @universe_ds_pairing_destroy(ptr %b)
  call void @free(ptr %ref)
  ret void
}

define internal void @test_pairing_errors() {
entry:
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %ok = alloca i64, align 8
  ; null
  %hn = call i64 @universe_ds_pairing_push(ptr null, i64 1, i64 0)
  %b0 = icmp ne i64 %hn, -1
  call void @bump_if(i1 %b0, ptr %viol)
  %e.popn = call i32 @universe_ds_pairing_pop(ptr null, ptr %ok, ptr null)
  %b1 = icmp ne i32 %e.popn, 1
  call void @bump_if(i1 %b1, ptr %viol)
  ; empty
  %h = call ptr @universe_ds_pairing_create()
  %e.emp = call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr null)
  %e.pk = call i32 @universe_ds_pairing_peek(ptr %h, ptr %ok, ptr null)
  %b2 = icmp ne i32 %e.emp, 4
  %b3 = icmp ne i32 %e.pk, 4
  call void @bump_if(i1 %b2, ptr %viol)
  call void @bump_if(i1 %b3, ptr %viol)
  ; single push/pop
  %hnd = call i64 @universe_ds_pairing_push(ptr %h, i64 77, i64 0)
  %e.p1 = call i32 @universe_ds_pairing_pop(ptr %h, ptr %ok, ptr null)
  %v1 = load i64, ptr %ok, align 8
  %b4 = icmp ne i64 %v1, 77
  call void @bump_if(i1 %b4, ptr %viol)
  %ln = call i64 @universe_ds_pairing_len(ptr %h)
  %b5 = icmp ne i64 %ln, 0
  call void @bump_if(i1 %b5, ptr %viol)
  call void @universe_ds_pairing_destroy(ptr %h)
  call void @universe_ds_pairing_destroy(ptr null)
  %vv = load i64, ptr %viol, align 8
  call void @ut_check_eq(i64 %vv, i64 0, ptr @m.pr.er)
  ret void
}

define internal void @bump_if(i1 %bad, ptr %viol) {
entry:
  br i1 %bad, label %inc, label %done
inc:
  %v = load i64, ptr %viol, align 8
  %v.n = add nuw i64 %v, 1
  store i64 %v.n, ptr %viol, align 8
  br label %done
done:
  ret void
}

define internal void @bump_notok(i1 %ok) {
entry:
  call void @ut_check(i1 %ok, ptr @m.dheap.pk)
  ret void
}

; ---- inline reference binary min-heap (array-backed) for the bench baseline ----
define internal void @refbh_push(ptr %data, ptr %lenp, i64 %key) {
entry:
  %len = load i64, ptr %lenp, align 8
  br label %loop
loop:
  %hole = phi i64 [ %len, %entry ], [ %parent, %shift ]
  %atroot = icmp eq i64 %hole, 0
  br i1 %atroot, label %settle, label %probe
probe:
  %hm1 = sub i64 %hole, 1
  %parent = lshr i64 %hm1, 1
  %pp = getelementptr inbounds i64, ptr %data, i64 %parent
  %pv = load i64, ptr %pp, align 8
  %need = icmp slt i64 %key, %pv
  br i1 %need, label %shift, label %settle
shift:
  %hp = getelementptr inbounds i64, ptr %data, i64 %hole
  store i64 %pv, ptr %hp, align 8
  br label %loop
settle:
  %sp = getelementptr inbounds i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  %len1 = add i64 %len, 1
  store i64 %len1, ptr %lenp, align 8
  ret void
}

define internal i64 @refbh_pop(ptr %data, ptr %lenp) {
entry:
  %len = load i64, ptr %lenp, align 8
  %min = load i64, ptr %data, align 8
  %len2 = sub i64 %len, 1
  store i64 %len2, ptr %lenp, align 8
  %lastp = getelementptr inbounds i64, ptr %data, i64 %len2
  %key = load i64, ptr %lastp, align 8
  %waslast = icmp eq i64 %len2, 0
  br i1 %waslast, label %done, label %loop
loop:
  %hole = phi i64 [ 0, %entry ], [ %child, %descend ]
  %left = shl i64 %hole, 1
  %left1 = or i64 %left, 1
  %hasleft = icmp ult i64 %left1, %len2
  br i1 %hasleft, label %pick, label %settle
pick:
  %lp = getelementptr inbounds i64, ptr %data, i64 %left1
  %lv = load i64, ptr %lp, align 8
  %right = add i64 %left1, 1
  %hasright = icmp ult i64 %right, %len2
  br i1 %hasright, label %pickr, label %chosel
pickr:
  %rp = getelementptr inbounds i64, ptr %data, i64 %right
  %rv = load i64, ptr %rp, align 8
  %rsm = icmp slt i64 %rv, %lv
  %rc = select i1 %rsm, i64 %right, i64 %left1
  %rcv = select i1 %rsm, i64 %rv, i64 %lv
  br label %cmp
chosel:
  br label %cmp
cmp:
  %child = phi i64 [ %rc, %pickr ], [ %left1, %chosel ]
  %cv = phi i64 [ %rcv, %pickr ], [ %lv, %chosel ]
  %need = icmp slt i64 %cv, %key
  br i1 %need, label %descend, label %settle
descend:
  %hp = getelementptr inbounds i64, ptr %data, i64 %hole
  store i64 %cv, ptr %hp, align 8
  br label %loop
settle:
  %sp = getelementptr inbounds i64, ptr %data, i64 %hole
  store i64 %key, ptr %sp, align 8
  br label %done
done:
  ret i64 %min
}

; ===========================================================================
; bench
; ===========================================================================
define internal void @bench() {
entry:
  %state = alloca i64, align 8
  %slot = alloca i64, align 8
  %ok = alloca i64, align 8
  %ov = alloca i64, align 8
  %bhlen = alloca i64, align 8
  %vsink = alloca i64, align 8
  %n = add i64 0, 500000
  br label %d.rep.head
; --- d-ary (d=4) push+pop (destructive: re-create each rep) ---
d.rep.head:
  %drep = phi i64 [ 0, %entry ], [ %drep.n, %d.rep.cont ]
  store i64 99887766554433221, ptr %state, align 8
  %hd = call ptr @universe_ds_dheap_create(i64 1024, i64 4)
  %t0 = call double @ut_now_sec()
  br label %d.push
d.push:
  %i = phi i64 [ 0, %d.rep.head ], [ %i.n, %d.push ]
  %r = call i64 @ut_rand(ptr %state)
  %k = urem i64 %r, 100000000
  call i32 @universe_ds_dheap_push(ptr %hd, i64 %k)
  %i.n = add nuw i64 %i, 1
  %m1 = icmp ult i64 %i.n, %n
  br i1 %m1, label %d.push, label %d.pop
d.pop:
  %j = phi i64 [ 0, %d.push ], [ %j.n, %d.pop ]
  %acc = phi i64 [ 0, %d.push ], [ %acc.n, %d.pop ]
  call i32 @universe_ds_dheap_pop(ptr %hd, ptr %slot)
  %pv = load i64, ptr %slot, align 8
  %acc.n = add i64 %acc, %pv
  %j.n = add nuw i64 %j, 1
  %m2 = icmp ult i64 %j.n, %n
  br i1 %m2, label %d.pop, label %d.rep.done
d.rep.done:
  %t1 = call double @ut_now_sec()
  store volatile i64 %acc.n, ptr %vsink, align 8
  call void @universe_ds_dheap_destroy(ptr %hd)
  %delapsed = fsub double %t1, %t0
  %dwarm = icmp eq i64 %drep, 0
  br i1 %dwarm, label %d.rep.cont, label %d.rep.store
d.rep.store:
  %dsi = sub i64 %drep, 1
  %dsp = getelementptr inbounds double, ptr @dh1.bench.samp, i64 %dsi
  store double %delapsed, ptr %dsp, align 8
  br label %d.rep.cont
d.rep.cont:
  %drep.n = add nuw i64 %drep, 1
  %dmore = icmp ult i64 %drep.n, 17
  br i1 %dmore, label %d.rep.head, label %d.rep.report
d.rep.report:
  call void @ut_report_dist(ptr @dh1.bench.samp, i64 16, i64 1000000, ptr @lbl.dh1.bench)
  br label %b.rep.head
; --- reference binary heap push+pop (same sequence, re-create each rep) ---
b.rep.head:
  %brep = phi i64 [ 0, %d.rep.report ], [ %brep.n, %b.rep.cont ]
  store i64 99887766554433221, ptr %state, align 8
  %bharr = call ptr @malloc(i64 4000000)   ; 500k i64
  store i64 0, ptr %bhlen, align 8
  %bt0 = call double @ut_now_sec()
  br label %b.push
b.push:
  %bi = phi i64 [ 0, %b.rep.head ], [ %bi.n, %b.push ]
  %br = call i64 @ut_rand(ptr %state)
  %bk = urem i64 %br, 100000000
  call void @refbh_push(ptr %bharr, ptr %bhlen, i64 %bk)
  %bi.n = add nuw i64 %bi, 1
  %bm1 = icmp ult i64 %bi.n, %n
  br i1 %bm1, label %b.push, label %b.pop
b.pop:
  %bj = phi i64 [ 0, %b.push ], [ %bj.n, %b.pop ]
  %bacc = phi i64 [ 0, %b.push ], [ %bacc.n, %b.pop ]
  %bpv = call i64 @refbh_pop(ptr %bharr, ptr %bhlen)
  %bacc.n = add i64 %bacc, %bpv
  %bj.n = add nuw i64 %bj, 1
  %bm2 = icmp ult i64 %bj.n, %n
  br i1 %bm2, label %b.pop, label %b.rep.done
b.rep.done:
  %bt1 = call double @ut_now_sec()
  store volatile i64 %bacc.n, ptr %vsink, align 8
  call void @free(ptr %bharr)
  %belapsed = fsub double %bt1, %bt0
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %b.rep.cont, label %b.rep.store
b.rep.store:
  %bsi = sub i64 %brep, 1
  %bsp = getelementptr inbounds double, ptr @bh.bench.samp, i64 %bsi
  store double %belapsed, ptr %bsp, align 8
  br label %b.rep.cont
b.rep.cont:
  %brep.n = add nuw i64 %brep, 1
  %bmore = icmp ult i64 %brep.n, 17
  br i1 %bmore, label %b.rep.head, label %b.rep.report
b.rep.report:
  call void @ut_report_dist(ptr @bh.bench.samp, i64 16, i64 1000000, ptr @lbl.bh.bench)
  br label %r.rep.head
; --- radix on monotone workload (re-create each rep) ---
r.rep.head:
  %rrep = phi i64 [ 0, %b.rep.report ], [ %rrep.n, %r.rep.cont ]
  %hr = call ptr @universe_ds_radixheap_create(i64 1024)
  store i64 424242424242424242, ptr %state, align 8
  %rt0 = call double @ut_now_sec()
  br label %r.push
r.push:
  %ri = phi i64 [ 0, %r.rep.head ], [ %ri.n, %r.push ]
  %rr = call i64 @ut_rand(ptr %state)
  %rk = urem i64 %rr, 100000000
  call i32 @universe_ds_radixheap_push(ptr %hr, i64 %rk, i64 %ri)
  %ri.n = add nuw i64 %ri, 1
  %rm1 = icmp ult i64 %ri.n, %n
  br i1 %rm1, label %r.push, label %r.pop
r.pop:
  %rj = phi i64 [ 0, %r.push ], [ %rj.n, %r.pop ]
  %racc = phi i64 [ 0, %r.push ], [ %racc.n, %r.pop ]
  call i32 @universe_ds_radixheap_pop(ptr %hr, ptr %ok, ptr %ov)
  %rpv = load i64, ptr %ok, align 8
  %racc.n = add i64 %racc, %rpv
  %rj.n = add nuw i64 %rj, 1
  %rm2 = icmp ult i64 %rj.n, %n
  br i1 %rm2, label %r.pop, label %r.rep.done
r.rep.done:
  %rt1 = call double @ut_now_sec()
  store volatile i64 %racc.n, ptr %vsink, align 8
  call void @universe_ds_radixheap_destroy(ptr %hr)
  %relapsed = fsub double %rt1, %rt0
  %rwarm = icmp eq i64 %rrep, 0
  br i1 %rwarm, label %r.rep.cont, label %r.rep.store
r.rep.store:
  %rsi = sub i64 %rrep, 1
  %rsp = getelementptr inbounds double, ptr @rx.bench.samp, i64 %rsi
  store double %relapsed, ptr %rsp, align 8
  br label %r.rep.cont
r.rep.cont:
  %rrep.n = add nuw i64 %rrep, 1
  %rmore = icmp ult i64 %rrep.n, 17
  br i1 %rmore, label %r.rep.head, label %r.rep.report
r.rep.report:
  call void @ut_report_dist(ptr @rx.bench.samp, i64 16, i64 1000000, ptr @lbl.rx.bench)
  br label %r2.rep.head
; --- d-ary on the same push-all-then-pop-all workload (re-create each rep) ---
r2.rep.head:
  %r2rep = phi i64 [ 0, %r.rep.report ], [ %r2rep.n, %r2.rep.cont ]
  %hd2 = call ptr @universe_ds_dheap_create(i64 1024, i64 4)
  store i64 424242424242424242, ptr %state, align 8
  %dt0 = call double @ut_now_sec()
  br label %r2.push
r2.push:
  %r2i = phi i64 [ 0, %r2.rep.head ], [ %r2i.n, %r2.push ]
  %r2r = call i64 @ut_rand(ptr %state)
  %r2k = urem i64 %r2r, 100000000
  call i32 @universe_ds_dheap_push(ptr %hd2, i64 %r2k)
  %r2i.n = add nuw i64 %r2i, 1
  %r2m1 = icmp ult i64 %r2i.n, %n
  br i1 %r2m1, label %r2.push, label %r2.pop
r2.pop:
  %r2j = phi i64 [ 0, %r2.push ], [ %r2j.n, %r2.pop ]
  %r2acc = phi i64 [ 0, %r2.push ], [ %r2acc.n, %r2.pop ]
  call i32 @universe_ds_dheap_pop(ptr %hd2, ptr %slot)
  %r2pv = load i64, ptr %slot, align 8
  %r2acc.n = add i64 %r2acc, %r2pv
  %r2j.n = add nuw i64 %r2j, 1
  %r2m2 = icmp ult i64 %r2j.n, %n
  br i1 %r2m2, label %r2.pop, label %r2.rep.done
r2.rep.done:
  %dt1 = call double @ut_now_sec()
  store volatile i64 %r2acc.n, ptr %vsink, align 8
  call void @universe_ds_dheap_destroy(ptr %hd2)
  %d2elapsed = fsub double %dt1, %dt0
  %d2warm = icmp eq i64 %r2rep, 0
  br i1 %d2warm, label %r2.rep.cont, label %r2.rep.store
r2.rep.store:
  %d2si = sub i64 %r2rep, 1
  %d2sp = getelementptr inbounds double, ptr @dh2.bench.samp, i64 %d2si
  store double %d2elapsed, ptr %d2sp, align 8
  br label %r2.rep.cont
r2.rep.cont:
  %r2rep.n = add nuw i64 %r2rep, 1
  %d2more = icmp ult i64 %r2rep.n, 17
  br i1 %d2more, label %r2.rep.head, label %r2.rep.report
r2.rep.report:
  call void @ut_report_dist(ptr @dh2.bench.samp, i64 16, i64 1000000, ptr @lbl.dh2.bench)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_dheap()
  call void @test_dheap_d8()
  call void @test_dheap_errors()
  call void @test_radix()
  call void @test_radix_monotone()
  call void @test_radix_errors()
  call void @test_pairing()
  call void @test_pairing_decrease()
  call void @test_pairing_meld()
  call void @test_pairing_errors()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish
do.bench:
  call void @bench()
  br label %finish
finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
