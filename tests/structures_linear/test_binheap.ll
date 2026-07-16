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

; Tests for universe_ds_binheap: pop order + multiset vs libc qsort over a
; fixed-seed random sequence, peek==min, edges 0/1/2, growth across the
; initial capacity boundary, EMPTY/NULL error codes. --bench times 100k
; push+pop vs qsort of the same keys.

declare ptr @universe_ds_binheap_create(i64)
declare void @universe_ds_binheap_destroy(ptr)
declare i32 @universe_ds_binheap_push(ptr, i64)
declare i32 @universe_ds_binheap_pop(ptr, ptr)
declare i32 @universe_ds_binheap_peek(ptr, ptr)
declare i64 @universe_ds_binheap_len(ptr)
declare i64 @universe_ds_binheap_capacity(ptr)

declare i32 @printf(ptr, ...)
declare void @qsort(ptr, i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.order  = private unnamed_addr constant [24 x i8] c"pop order == qsort 1000\00"
@m.edges  = private unnamed_addr constant [16 x i8] c"edges 0/1/2 min\00"
@m.grow   = private unnamed_addr constant [24 x i8] c"grow past initial cap  \00"
@m.errs   = private unnamed_addr constant [17 x i8] c"error codes 1/4 \00"
@lbl.binheap_heap  = private unnamed_addr constant [32 x i8] c"binheap push+pop (200k ops/rep)\00"
@lbl.binheap_qsort = private unnamed_addr constant [31 x i8] c"qsort ref 100k keys (100k/rep)\00"
@binheap.hsamp = internal global [16 x double] zeroinitializer, align 8
@binheap.qsamp = internal global [16 x double] zeroinitializer, align 8
@binheap.sink  = internal global i64 0, align 8

; signed i64 three-way comparator for qsort.
define internal i32 @cmp_i64(ptr %a, ptr %b) {
entry:
  %av = load i64, ptr %a, align 8
  %bv = load i64, ptr %b, align 8
  %gt = icmp sgt i64 %av, %bv
  %lt = icmp slt i64 %av, %bv
  %g = zext i1 %gt to i32
  %l = zext i1 %lt to i32
  %r = sub nsw i32 %g, %l
  ret i32 %r
}

; --- pop order + multiset vs qsort ---------------------------------------
define internal void @test_order() {
entry:
  %arr = alloca [1000 x i64], align 16
  %seed = alloca i64, align 8
  %out = alloca i64, align 8
  store i64 20260716, ptr %seed, align 8
  %h = call ptr @universe_ds_binheap_create(i64 8)
  br label %push

push:
  %i = phi i64 [ 0, %entry ], [ %i.n, %push ]
  %r = call i64 @ut_rand(ptr %seed)
  %ap = getelementptr inbounds nuw i64, ptr %arr, i64 %i
  store i64 %r, ptr %ap, align 8
  call i32 @universe_ds_binheap_push(ptr %h, i64 %r)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000
  br i1 %more, label %push, label %sort

sort:
  %lenc = call i64 @universe_ds_binheap_len(ptr %h)
  call void @qsort(ptr %arr, i64 1000, i64 8, ptr @cmp_i64)
  br label %pop

pop:
  %j = phi i64 [ 0, %sort ], [ %j.n, %pop ]
  %viol = phi i64 [ 0, %sort ], [ %viol.n, %pop ]
  %rc = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %v = load i64, ptr %out, align 8
  %ep = getelementptr inbounds nuw i64, ptr %arr, i64 %j
  %want = load i64, ptr %ep, align 8
  %bad.rc = icmp ne i32 %rc, 0
  %bad.v = icmp ne i64 %v, %want
  %bad = or i1 %bad.rc, %bad.v
  %inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000
  br i1 %more2, label %pop, label %report

report:
  %len0 = call i64 @universe_ds_binheap_len(ptr %h)
  %len.ok = icmp eq i64 %len0, 0
  %lenc.ok = icmp eq i64 %lenc, 1000
  %pre = and i1 %len.ok, %lenc.ok
  %prez = zext i1 %pre to i64
  %prebad = xor i64 %prez, 1
  %tot = add nuw i64 %viol.n, %prebad
  call void @ut_check_eq(i64 %tot, i64 0, ptr @m.order)
  call void @universe_ds_binheap_destroy(ptr %h)
  ret void
}

; --- edges 0/1/2 ----------------------------------------------------------
define internal void @test_edges() {
entry:
  %out = alloca i64, align 8
  %h = call ptr @universe_ds_binheap_create(i64 8)
  ; empty peek/pop -> EMPTY(4)
  %pe = call i32 @universe_ds_binheap_peek(ptr %h, ptr %out)
  %po = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  ; one element
  call i32 @universe_ds_binheap_push(ptr %h, i64 42)
  %pk1 = call i32 @universe_ds_binheap_peek(ptr %h, ptr %out)
  %v1 = load i64, ptr %out, align 8
  %pp1 = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %pv1 = load i64, ptr %out, align 8
  ; two elements, out of order -> min first
  call i32 @universe_ds_binheap_push(ptr %h, i64 99)
  call i32 @universe_ds_binheap_push(ptr %h, i64 7)
  %pkm = call i32 @universe_ds_binheap_peek(ptr %h, ptr %out)
  %vm = load i64, ptr %out, align 8
  %pa = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %va = load i64, ptr %out, align 8
  %pb = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %vb = load i64, ptr %out, align 8
  %c0 = icmp eq i32 %pe, 4
  %c1 = icmp eq i32 %po, 4
  %c2 = icmp eq i64 %v1, 42
  %c3 = icmp eq i64 %pv1, 42
  %c4 = icmp eq i64 %vm, 7
  %c5 = icmp eq i64 %va, 7
  %c6 = icmp eq i64 %vb, 99
  %a0 = and i1 %c0, %c1
  %a1 = and i1 %a0, %c2
  %a2 = and i1 %a1, %c3
  %a3 = and i1 %a2, %c4
  %a4 = and i1 %a3, %c5
  %a5 = and i1 %a4, %c6
  call void @ut_check(i1 %a5, ptr @m.edges)
  call void @universe_ds_binheap_destroy(ptr %h)
  ret void
}

; --- growth across the initial capacity boundary --------------------------
define internal void @test_grow() {
entry:
  %out = alloca i64, align 8
  %h = call ptr @universe_ds_binheap_create(i64 8)
  %cap0 = call i64 @universe_ds_binheap_capacity(ptr %h)
  br label %push

push:
  ; push 8..1 descending so each insert sifts to the root; 20 items forces
  ; at least one grow (8 -> 16 -> 32).
  %i = phi i64 [ 0, %entry ], [ %i.n, %push ]
  %key = sub nsw i64 40, %i
  call i32 @universe_ds_binheap_push(ptr %h, i64 %key)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 20
  br i1 %more, label %push, label %check

check:
  %len = call i64 @universe_ds_binheap_len(ptr %h)
  %cap = call i64 @universe_ds_binheap_capacity(ptr %h)
  br label %pop

pop:
  %j = phi i64 [ 0, %check ], [ %j.n, %pop ]
  %prev = phi i64 [ -9223372036854775808, %check ], [ %v, %pop ]
  %viol = phi i64 [ 0, %check ], [ %viol.n, %pop ]
  %rc = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %v = load i64, ptr %out, align 8
  %desc = icmp slt i64 %v, %prev
  %inc = zext i1 %desc to i64
  %viol.n = add nuw i64 %viol, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 20
  br i1 %more2, label %pop, label %report

report:
  ; smallest key pushed was 40-19 = 21
  %min.ok = icmp eq i64 %prev, 40         ; prev holds last (largest) popped
  %len.ok = icmp eq i64 %len, 20
  %cap.ok = icmp uge i64 %cap, 32
  %cap0.ok = icmp eq i64 %cap0, 8
  %p0 = and i1 %len.ok, %cap.ok
  %p1 = and i1 %p0, %cap0.ok
  %p1z = zext i1 %p1 to i64
  %p1bad = xor i64 %p1z, 1
  %tot = add nuw i64 %viol, %p1bad
  call void @ut_check_eq(i64 %tot, i64 0, ptr @m.grow)
  call void @universe_ds_binheap_destroy(ptr %h)
  ret void
}

; --- error codes ----------------------------------------------------------
define internal void @test_errors() {
entry:
  %out = alloca i64, align 8
  %h = call ptr @universe_ds_binheap_create(i64 8)
  %e1 = call i32 @universe_ds_binheap_push(ptr null, i64 0)
  %e2 = call i32 @universe_ds_binheap_pop(ptr null, ptr %out)
  %e3 = call i32 @universe_ds_binheap_pop(ptr %h, ptr null)
  %e4 = call i32 @universe_ds_binheap_peek(ptr null, ptr %out)
  %e5 = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)   ; empty -> 4
  %e6 = call i32 @universe_ds_binheap_peek(ptr %h, ptr %out)  ; empty -> 4
  %o1 = icmp eq i32 %e1, 1
  %o2 = icmp eq i32 %e2, 1
  %o3 = icmp eq i32 %e3, 1
  %o4 = icmp eq i32 %e4, 1
  %o5 = icmp eq i32 %e5, 4
  %o6 = icmp eq i32 %e6, 4
  %p0 = and i1 %o1, %o2
  %p1 = and i1 %p0, %o3
  %p2 = and i1 %p1, %o4
  %p3 = and i1 %p2, %o5
  %p4 = and i1 %p3, %o6
  call void @ut_check(i1 %p4, ptr @m.errs)
  call void @universe_ds_binheap_destroy(ptr %h)
  call void @universe_ds_binheap_destroy(ptr null)
  ret void
}

; --- bench: 100k push+pop vs qsort ---------------------------------------
define internal void @bench() {
entry:
  %arr = alloca [100000 x i64], align 16
  %seed = alloca i64, align 8
  %out = alloca i64, align 8
  store i64 7, ptr %seed, align 8
  br label %gen

gen:
  %gi = phi i64 [ 0, %entry ], [ %gi.n, %gen ]
  %r = call i64 @ut_rand(ptr %seed)
  %ap = getelementptr inbounds nuw i64, ptr %arr, i64 %gi
  store i64 %r, ptr %ap, align 8
  %gi.n = add nuw i64 %gi, 1
  %gmore = icmp ult i64 %gi.n, 100000
  br i1 %gmore, label %gen, label %hrep

; --- heap push+pop distribution (array unchanged by heap ops) ---
hrep:
  %hrep.i = phi i64 [ 0, %gen ], [ %hrep.n, %hrep.next ]
  %h = call ptr @universe_ds_binheap_create(i64 1024)
  %t0 = call double @ut_now_sec()
  br label %hpush

hpush:
  %pi = phi i64 [ 0, %hrep ], [ %pi.n, %hpush ]
  %pp = getelementptr inbounds nuw i64, ptr %arr, i64 %pi
  %pv = load i64, ptr %pp, align 8
  call i32 @universe_ds_binheap_push(ptr %h, i64 %pv)
  %pi.n = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, 100000
  br i1 %pmore, label %hpush, label %hpop

hpop:
  %oi = phi i64 [ 0, %hpush ], [ %oi.n, %hpop ]
  %acc = phi i64 [ 0, %hpush ], [ %acc.n, %hpop ]
  %rc = call i32 @universe_ds_binheap_pop(ptr %h, ptr %out)
  %ov = load i64, ptr %out, align 8
  %acc.n = add i64 %acc, %ov
  %oi.n = add nuw i64 %oi, 1
  %omore = icmp ult i64 %oi.n, 100000
  br i1 %omore, label %hpop, label %hrep.done

hrep.done:
  %t1 = call double @ut_now_sec()
  call void @universe_ds_binheap_destroy(ptr %h)
  store volatile i64 %acc.n, ptr @binheap.sink, align 8
  %hdt = fsub double %t1, %t0
  %hwarm = icmp eq i64 %hrep.i, 0
  br i1 %hwarm, label %hrep.next, label %hrep.store

hrep.store:
  %hsidx = sub i64 %hrep.i, 1
  %hsp = getelementptr inbounds double, ptr @binheap.hsamp, i64 %hsidx
  store double %hdt, ptr %hsp, align 8
  br label %hrep.next

hrep.next:
  %hrep.n = add nuw nsw i64 %hrep.i, 1
  %hrep.more = icmp ult i64 %hrep.n, 17
  br i1 %hrep.more, label %hrep, label %hreport

hreport:
  call void @ut_report_dist(ptr @binheap.hsamp, i64 16, i64 200000, ptr @lbl.binheap_heap)
  br label %qrep

; --- qsort reference distribution (regenerate the same random array each rep) ---
qrep:
  %qrep.i = phi i64 [ 0, %hreport ], [ %qrep.n, %qrep.next ]
  store i64 7, ptr %seed, align 8
  br label %qgen

qgen:
  %qgi = phi i64 [ 0, %qrep ], [ %qgi.n, %qgen ]
  %qr = call i64 @ut_rand(ptr %seed)
  %qap = getelementptr inbounds nuw i64, ptr %arr, i64 %qgi
  store i64 %qr, ptr %qap, align 8
  %qgi.n = add nuw i64 %qgi, 1
  %qgmore = icmp ult i64 %qgi.n, 100000
  br i1 %qgmore, label %qgen, label %qtime

qtime:
  %qt0 = call double @ut_now_sec()
  call void @qsort(ptr %arr, i64 100000, i64 8, ptr @cmp_i64)
  %qt1 = call double @ut_now_sec()
  %qdt = fsub double %qt1, %qt0
  %qwarm = icmp eq i64 %qrep.i, 0
  br i1 %qwarm, label %qrep.next, label %qrep.store

qrep.store:
  %qsidx = sub i64 %qrep.i, 1
  %qsp = getelementptr inbounds double, ptr @binheap.qsamp, i64 %qsidx
  store double %qdt, ptr %qsp, align 8
  br label %qrep.next

qrep.next:
  %qrep.n = add nuw nsw i64 %qrep.i, 1
  %qrep.more = icmp ult i64 %qrep.n, 17
  br i1 %qrep.more, label %qrep, label %qreport

qreport:
  call void @ut_report_dist(ptr @binheap.qsamp, i64 16, i64 100000, ptr @lbl.binheap_qsort)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_order()
  call void @test_edges()
  call void @test_grow()
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
