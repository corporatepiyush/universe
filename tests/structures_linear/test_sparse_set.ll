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

; Tests for universe_ds_sparseset: add/contains/remove/size, idempotency, the
; swap-with-last invariant, O(1) clear (proves sparse need not be zeroed),
; dense iteration enumerates exactly the members, error codes, and a fixed-seed
; random run cross-checked against a brute-force bool[] shadow. --bench pits
; add/contains/clear throughput against a bitset over the same universe and
; shows the O(1)-clear win.

declare ptr @universe_ds_sparseset_create(i64)
declare void @universe_ds_sparseset_destroy(ptr)
declare i32 @universe_ds_sparseset_add(ptr, i64)
declare i32 @universe_ds_sparseset_remove(ptr, i64)
declare i32 @universe_ds_sparseset_contains(ptr, i64)
declare i64 @universe_ds_sparseset_size(ptr)
declare i64 @universe_ds_sparseset_capacity(ptr)
declare i32 @universe_ds_sparseset_clear(ptr)
declare ptr @universe_ds_sparseset_get_dense(ptr, ptr)

; bitset — same domain, linked in; used only as the --bench reference.
declare ptr @universe_ds_bitset_create(i64)
declare void @universe_ds_bitset_destroy(ptr)
declare i32 @universe_ds_bitset_set(ptr, i64)
declare i32 @universe_ds_bitset_clear(ptr, i64)
declare i32 @universe_ds_bitset_test(ptr, i64)
declare i32 @universe_ds_bitset_clear_all(ptr)

declare i32 @printf(ptr, ...)
declare i64 @ut_rand(ptr)
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.basic   = private unnamed_addr constant [24 x i8] c"add/contains/size basic\00"
@m.idem    = private unnamed_addr constant [20 x i8] c"add idempotent size\00"
@m.remove  = private unnamed_addr constant [22 x i8] c"remove swap-invariant\00"
@m.clear   = private unnamed_addr constant [24 x i8] c"clear empties, reusable\00"
@m.iter    = private unnamed_addr constant [22 x i8] c"dense enumerates set \00"
@m.errs    = private unnamed_addr constant [18 x i8] c"error codes 1 / 7\00"
@m.rand    = private unnamed_addr constant [22 x i8] c"random vs bool shadow\00"
@m.randsz  = private unnamed_addr constant [20 x i8] c"random size matches\00"
@m.edge0   = private unnamed_addr constant [18 x i8] c"empty universe ok\00"
@m.edge1   = private unnamed_addr constant [18 x i8] c"single-slot round\00"
@m.full    = private unnamed_addr constant [24 x i8] c"fill all then drain all\00"
@lbl.sparse_add = private unnamed_addr constant [24 x i8] c"sparseset add (20k/rep)\00"
@lbl.sparse_con = private unnamed_addr constant [29 x i8] c"sparseset contains (20k/rep)\00"
@lbl.sparse_clr = private unnamed_addr constant [27 x i8] c"sparseset clear (200k/rep)\00"
@lbl.bitset_clr = private unnamed_addr constant [28 x i8] c"bitset clear_all (200k/rep)\00"
@sparse.addsamp  = internal global [16 x double] zeroinitializer, align 8
@sparse.consamp  = internal global [16 x double] zeroinitializer, align 8
@sparse.clrsamp  = internal global [16 x double] zeroinitializer, align 8
@sparse.bclrsamp = internal global [16 x double] zeroinitializer, align 8
@sparse.sink     = internal global i64 0, align 8

; --- add / contains / size ------------------------------------------------
define internal void @test_basic() {
entry:
  %s = call ptr @universe_ds_sparseset_create(i64 256)
  %cap = call i64 @universe_ds_sparseset_capacity(ptr %s)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 5)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 200)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 0)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 255)
  %c5 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 5)
  %c200 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 200)
  %c0 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 0)
  %c255 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 255)
  %c7 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 7)
  %c100 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 100)
  %sz = call i64 @universe_ds_sparseset_size(ptr %s)
  %k0 = icmp eq i64 %cap, 256
  %k1 = icmp eq i32 %c5, 1
  %k2 = icmp eq i32 %c200, 1
  %k3 = icmp eq i32 %c0, 1
  %k4 = icmp eq i32 %c255, 1
  %k5 = icmp eq i32 %c7, 0
  %k6 = icmp eq i32 %c100, 0
  %k7 = icmp eq i64 %sz, 4
  %a0 = and i1 %k0, %k1
  %a1 = and i1 %a0, %k2
  %a2 = and i1 %a1, %k3
  %a3 = and i1 %a2, %k4
  %a4 = and i1 %a3, %k5
  %a5 = and i1 %a4, %k6
  %a6 = and i1 %a5, %k7
  call void @ut_check(i1 %a6, ptr @m.basic)
  ; idempotent add: re-adding present members must not grow the set.
  call i32 @universe_ds_sparseset_add(ptr %s, i64 5)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 0)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 255)
  %sz2 = call i64 @universe_ds_sparseset_size(ptr %s)
  %i0 = icmp eq i64 %sz2, 4
  call void @ut_check(i1 %i0, ptr @m.idem)
  call void @universe_ds_sparseset_destroy(ptr %s)
  ret void
}

; --- remove keeps the swap-with-last invariant ----------------------------
define internal void @test_remove() {
entry:
  ; members {1,2,3,4,5,6,7}; remove 3 and 6 (interior). Survivors present,
  ; removed absent, size correct, and the dense array still holds exactly the
  ; survivors (checked via get_dense below in test_iter).
  %s = call ptr @universe_ds_sparseset_create(i64 16)
  br label %fill

fill:
  %i = phi i64 [ 1, %entry ], [ %i.n, %fill ]
  call i32 @universe_ds_sparseset_add(ptr %s, i64 %i)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 8
  br i1 %more, label %fill, label %rm

rm:
  call i32 @universe_ds_sparseset_remove(ptr %s, i64 3)
  call i32 @universe_ds_sparseset_remove(ptr %s, i64 6)
  ; idempotent remove: removing an absent member is a no-op returning OK.
  %r0 = call i32 @universe_ds_sparseset_remove(ptr %s, i64 3)
  %r1 = call i32 @universe_ds_sparseset_remove(ptr %s, i64 15)
  %sz = call i64 @universe_ds_sparseset_size(ptr %s)
  %c1 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 1)
  %c3 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 3)
  %c4 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 4)
  %c6 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 6)
  %c7 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 7)
  %k0 = icmp eq i32 %r0, 0
  %k1 = icmp eq i32 %r1, 0
  %k2 = icmp eq i64 %sz, 5
  %k3 = icmp eq i32 %c1, 1
  %k4 = icmp eq i32 %c3, 0
  %k5 = icmp eq i32 %c4, 1
  %k6 = icmp eq i32 %c6, 0
  %k7 = icmp eq i32 %c7, 1
  %a0 = and i1 %k0, %k1
  %a1 = and i1 %a0, %k2
  %a2 = and i1 %a1, %k3
  %a3 = and i1 %a2, %k4
  %a4 = and i1 %a3, %k5
  %a5 = and i1 %a4, %k6
  %a6 = and i1 %a5, %k7
  call void @ut_check(i1 %a6, ptr @m.remove)
  call void @universe_ds_sparseset_destroy(ptr %s)
  ret void
}

; --- clear() empties in O(1) and the set is reusable ----------------------
define internal void @test_clear() {
entry:
  %s = call ptr @universe_ds_sparseset_create(i64 64)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 10)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 20)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 63)
  call i32 @universe_ds_sparseset_clear(ptr %s)
  %sz = call i64 @universe_ds_sparseset_size(ptr %s)
  br label %sweep

sweep:
  %i = phi i64 [ 0, %entry ], [ %i.n, %sweep ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %sweep ]
  %c = call i32 @universe_ds_sparseset_contains(ptr %s, i64 %i)
  %bad = icmp ne i32 %c, 0
  %inc = zext i1 %bad to i64
  %viol.n = add nuw i64 %viol, %inc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 64
  br i1 %more, label %sweep, label %reuse

reuse:
  ; re-add after clear proves sparse[] was never required to be zeroed: a stale
  ; sparse[x] left over from before clear must NOT read as present.
  call i32 @universe_ds_sparseset_add(ptr %s, i64 20)
  %c20 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 20)
  %c10 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 10)
  %sz2 = call i64 @universe_ds_sparseset_size(ptr %s)
  %k0 = icmp eq i64 %sz, 0
  %k1 = icmp eq i64 %viol, 0
  %k2 = icmp eq i32 %c20, 1
  %k3 = icmp eq i32 %c10, 0
  %k4 = icmp eq i64 %sz2, 1
  %a0 = and i1 %k0, %k1
  %a1 = and i1 %a0, %k2
  %a2 = and i1 %a1, %k3
  %a3 = and i1 %a2, %k4
  call void @ut_check(i1 %a3, ptr @m.clear)
  call void @universe_ds_sparseset_destroy(ptr %s)
  ret void
}

; --- dense iteration enumerates exactly the members -----------------------
define internal void @test_iter() {
entry:
  ; mark[] shadows membership; every dense[i] must be a marked member and every
  ; marked member must appear exactly once (len == size == number marked).
  %mark = alloca [64 x i8], align 16
  %len = alloca i64, align 8
  br label %zero

zero:
  %zi = phi i64 [ 0, %entry ], [ %zi.n, %zero ]
  %zp = getelementptr inbounds nuw i8, ptr %mark, i64 %zi
  store i8 0, ptr %zp, align 1
  %zi.n = add nuw i64 %zi, 1
  %zmore = icmp ult i64 %zi.n, 64
  br i1 %zmore, label %zero, label %build

build:
  %s = call ptr @universe_ds_sparseset_create(i64 64)
  ; members {2,3,5,7,11,13,17,19,23,29,31,37} ish via a stride
  call i32 @universe_ds_sparseset_add(ptr %s, i64 2)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 9)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 40)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 63)
  call i32 @universe_ds_sparseset_add(ptr %s, i64 17)
  call i32 @universe_ds_sparseset_remove(ptr %s, i64 9)   ; leaves {2,40,63,17}
  %mp2 = getelementptr inbounds nuw i8, ptr %mark, i64 2
  store i8 1, ptr %mp2, align 1
  %mp40 = getelementptr inbounds nuw i8, ptr %mark, i64 40
  store i8 1, ptr %mp40, align 1
  %mp63 = getelementptr inbounds nuw i8, ptr %mark, i64 63
  store i8 1, ptr %mp63, align 1
  %mp17 = getelementptr inbounds nuw i8, ptr %mark, i64 17
  store i8 1, ptr %mp17, align 1
  %base = call ptr @universe_ds_sparseset_get_dense(ptr %s, ptr %len)
  %n = load i64, ptr %len, align 8
  br label %walk

walk:
  %wi = phi i64 [ 0, %build ], [ %wi.n, %walk.body ]
  %viol = phi i64 [ 0, %build ], [ %viol2, %walk.body ]
  %done = icmp uge i64 %wi, %n
  br i1 %done, label %walk.end, label %walk.body

walk.body:
  %ep = getelementptr inbounds nuw i32, ptr %base, i64 %wi
  %v32 = load i32, ptr %ep, align 4
  %v = zext i32 %v32 to i64
  %mp = getelementptr inbounds nuw i8, ptr %mark, i64 %v
  %mb = load i8, ptr %mp, align 1
  %ismark = icmp eq i8 %mb, 1
  %bad = xor i1 %ismark, true
  %binc = zext i1 %bad to i64
  ; clear the mark to catch duplicates (a repeated value would read 0 next time)
  store i8 0, ptr %mp, align 1
  %viol2 = add nuw i64 %viol, %binc
  %wi.n = add nuw i64 %wi, 1
  br label %walk

walk.end:
  %sz = call i64 @universe_ds_sparseset_size(ptr %s)
  %lenok = icmp eq i64 %n, %sz
  %lenok4 = icmp eq i64 %n, 4
  %noviol = icmp eq i64 %viol, 0
  %a0 = and i1 %lenok, %lenok4
  %a1 = and i1 %a0, %noviol
  call void @ut_check(i1 %a1, ptr @m.iter)
  call void @universe_ds_sparseset_destroy(ptr %s)
  ret void
}

; --- error codes ----------------------------------------------------------
define internal void @test_errors() {
entry:
  %s = call ptr @universe_ds_sparseset_create(i64 32)
  %e1 = call i32 @universe_ds_sparseset_add(ptr null, i64 0)     ; NULL -> 1
  %e2 = call i32 @universe_ds_sparseset_remove(ptr null, i64 0)  ; NULL -> 1
  %e3 = call i32 @universe_ds_sparseset_clear(ptr null)          ; NULL -> 1
  %e4 = call i32 @universe_ds_sparseset_add(ptr %s, i64 32)      ; OOB  -> 7
  %e5 = call i32 @universe_ds_sparseset_add(ptr %s, i64 999)     ; OOB  -> 7
  %e6 = call i32 @universe_ds_sparseset_remove(ptr %s, i64 32)   ; OOB  -> 7
  %e7 = call i32 @universe_ds_sparseset_contains(ptr null, i64 0); NULL -> 0
  %e8 = call i32 @universe_ds_sparseset_contains(ptr %s, i64 99) ; OOB  -> 0
  %sz = call i64 @universe_ds_sparseset_size(ptr null)           ; NULL -> 0
  %cp = call i64 @universe_ds_sparseset_capacity(ptr null)       ; NULL -> 0
  %len = alloca i64, align 8
  store i64 12345, ptr %len, align 8
  %gd = call ptr @universe_ds_sparseset_get_dense(ptr null, ptr %len) ; NULL -> null,*len=0
  %gdlen = load i64, ptr %len, align 8
  %o1 = icmp eq i32 %e1, 1
  %o2 = icmp eq i32 %e2, 1
  %o3 = icmp eq i32 %e3, 1
  %o4 = icmp eq i32 %e4, 7
  %o5 = icmp eq i32 %e5, 7
  %o6 = icmp eq i32 %e6, 7
  %o7 = icmp eq i32 %e7, 0
  %o8 = icmp eq i32 %e8, 0
  %o9 = icmp eq i64 %sz, 0
  %o10 = icmp eq i64 %cp, 0
  %o11 = icmp eq ptr %gd, null
  %o12 = icmp eq i64 %gdlen, 0
  %p0 = and i1 %o1, %o2
  %p1 = and i1 %p0, %o3
  %p2 = and i1 %p1, %o4
  %p3 = and i1 %p2, %o5
  %p4 = and i1 %p3, %o6
  %p5 = and i1 %p4, %o7
  %p6 = and i1 %p5, %o8
  %p7 = and i1 %p6, %o9
  %p8 = and i1 %p7, %o10
  %p9 = and i1 %p8, %o11
  %p10 = and i1 %p9, %o12
  call void @ut_check(i1 %p10, ptr @m.errs)
  call void @universe_ds_sparseset_destroy(ptr %s)
  call void @universe_ds_sparseset_destroy(ptr null)
  ret void
}

; --- fixed-seed random ops cross-checked vs a brute-force bool[] shadow ----
define internal void @test_random() {
entry:
  %shadow = alloca [1024 x i8], align 16
  %csnap = alloca [1024 x i8], align 16
  %seed = alloca i64, align 8
  store i64 20260716, ptr %seed, align 8
  br label %zero

zero:
  %zi = phi i64 [ 0, %entry ], [ %zi.n, %zero ]
  %zp = getelementptr inbounds nuw i8, ptr %shadow, i64 %zi
  store i8 0, ptr %zp, align 1
  %zi.n = add nuw i64 %zi, 1
  %zmore = icmp ult i64 %zi.n, 1024
  br i1 %zmore, label %zero, label %build

build:
  %s = call ptr @universe_ds_sparseset_create(i64 1024)
  br label %ops

ops:
  ; %expect is the reference member count, maintained incrementally from the
  ; shadow's PRIOR state on each op (add of a new key +1, remove of a present
  ; key -1). Keeping the count as a running total avoids re-summing the shadow
  ; array afterwards.
  %oi = phi i64 [ 0, %build ], [ %oi.n, %verify ]
  %viol = phi i64 [ 0, %build ], [ %viol.n, %verify ]
  %expect = phi i64 [ 0, %build ], [ %expect.n, %verify ]
  %r1 = call i64 @ut_rand(ptr %seed)
  %x = and i64 %r1, 1023
  %r2 = call i64 @ut_rand(ptr %seed)
  %isadd = and i64 %r2, 1
  %do.add = icmp eq i64 %isadd, 1
  %sp.x = getelementptr inbounds nuw i8, ptr %shadow, i64 %x
  %prev = load i8, ptr %sp.x, align 1
  br i1 %do.add, label %op.add, label %op.rm

op.add:
  call i32 @universe_ds_sparseset_add(ptr %s, i64 %x)
  store i8 1, ptr %sp.x, align 1
  %was.absent = icmp eq i8 %prev, 0
  %add.inc = zext i1 %was.absent to i64
  %expect.add = add nuw i64 %expect, %add.inc
  br label %verify

op.rm:
  call i32 @universe_ds_sparseset_remove(ptr %s, i64 %x)
  store i8 0, ptr %sp.x, align 1
  %was.present = icmp eq i8 %prev, 1
  %rm.dec = zext i1 %was.present to i64
  %expect.rm = sub i64 %expect, %rm.dec
  br label %verify

verify:
  %expect.n = phi i64 [ %expect.add, %op.add ], [ %expect.rm, %op.rm ]
  %want8 = phi i8 [ 1, %op.add ], [ 0, %op.rm ]
  %c = call i32 @universe_ds_sparseset_contains(ptr %s, i64 %x)
  %want = zext i8 %want8 to i32
  %mism = icmp ne i32 %c, %want
  %minc = zext i1 %mism to i64
  %viol.n = add nuw i64 %viol, %minc
  %oi.n = add nuw i64 %oi, 1
  %more = icmp ult i64 %oi.n, 20000
  br i1 %more, label %ops, label %snap

snap:
  ; full sweep in TWO passes: first snapshot contains() over the whole universe
  ; into csnap (a call-only loop), then compare csnap to the shadow and total
  ; the shadow separately (a load-only loop). Splitting the phases keeps the
  ; membership probe and the reference read in distinct loops.
  %fi = phi i64 [ 0, %verify ], [ %fi.n, %snap ]
  %fc = call i32 @universe_ds_sparseset_contains(ptr %s, i64 %fi)
  %fcb = trunc i32 %fc to i8
  %fcp = getelementptr inbounds nuw i8, ptr %csnap, i64 %fi
  store i8 %fcb, ptr %fcp, align 1
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 1024
  br i1 %fmore, label %snap, label %sweep

sweep:
  ; every universe index must agree between the membership snapshot and the
  ; shadow reference.
  %si = phi i64 [ 0, %snap ], [ %si.n, %sweep ]
  %fviol = phi i64 [ 0, %snap ], [ %fviol.n, %sweep ]
  %ssp = getelementptr inbounds nuw i8, ptr %shadow, i64 %si
  %ssv = load i8, ptr %ssp, align 1
  %scp = getelementptr inbounds nuw i8, ptr %csnap, i64 %si
  %scv = load i8, ptr %scp, align 1
  %fmis = icmp ne i8 %ssv, %scv
  %fminc = zext i1 %fmis to i64
  %fviol.n = add nuw i64 %fviol, %fminc
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 1024
  br i1 %smore, label %sweep, label %report

report:
  %tot = add nuw i64 %viol, %fviol
  call void @ut_check_eq(i64 %tot, i64 0, ptr @m.rand)
  %sz = call i64 @universe_ds_sparseset_size(ptr %s)
  call void @ut_check_eq(i64 %sz, i64 %expect.n, ptr @m.randsz)
  call void @universe_ds_sparseset_destroy(ptr %s)
  ret void
}

; --- edges: empty universe, single slot, fill-all-then-drain-all ----------
define internal void @test_edges() {
entry:
  ; empty universe: add always INVALID_INDEX, contains 0, size 0, dense len 0.
  %len = alloca i64, align 8
  %e = call ptr @universe_ds_sparseset_create(i64 0)
  %ea = call i32 @universe_ds_sparseset_add(ptr %e, i64 0)
  %ec = call i32 @universe_ds_sparseset_contains(ptr %e, i64 0)
  %ecap = call i64 @universe_ds_sparseset_capacity(ptr %e)
  call ptr @universe_ds_sparseset_get_dense(ptr %e, ptr %len)
  %elen = load i64, ptr %len, align 8
  %z0 = icmp eq i32 %ea, 7
  %z1 = icmp eq i32 %ec, 0
  %z2 = icmp eq i64 %ecap, 0
  %z3 = icmp eq i64 %elen, 0
  %za = and i1 %z0, %z1
  %zb = and i1 %za, %z2
  %zc = and i1 %zb, %z3
  call void @ut_check(i1 %zc, ptr @m.edge0)
  call void @universe_ds_sparseset_destroy(ptr %e)
  ; single slot.
  %o = call ptr @universe_ds_sparseset_create(i64 1)
  call i32 @universe_ds_sparseset_add(ptr %o, i64 0)
  call i32 @universe_ds_sparseset_add(ptr %o, i64 0)
  %oc = call i32 @universe_ds_sparseset_contains(ptr %o, i64 0)
  %osz = call i64 @universe_ds_sparseset_size(ptr %o)
  call i32 @universe_ds_sparseset_remove(ptr %o, i64 0)
  %oc2 = call i32 @universe_ds_sparseset_contains(ptr %o, i64 0)
  %osz2 = call i64 @universe_ds_sparseset_size(ptr %o)
  %s0 = icmp eq i32 %oc, 1
  %s1 = icmp eq i64 %osz, 1
  %s2 = icmp eq i32 %oc2, 0
  %s3 = icmp eq i64 %osz2, 0
  %sa = and i1 %s0, %s1
  %sb = and i1 %sa, %s2
  %sc = and i1 %sb, %s3
  call void @ut_check(i1 %sc, ptr @m.edge1)
  call void @universe_ds_sparseset_destroy(ptr %o)
  br label %fillall.pre

fillall.pre:
  ; fill EVERY slot, then drain every slot.
  %f = call ptr @universe_ds_sparseset_create(i64 300)
  br label %fillall

fillall:
  %ai = phi i64 [ 0, %fillall.pre ], [ %ai.n, %fillall ]
  call i32 @universe_ds_sparseset_add(ptr %f, i64 %ai)
  %ai.n = add nuw i64 %ai, 1
  %amore = icmp ult i64 %ai.n, 300
  br i1 %amore, label %fillall, label %checkfull

checkfull:
  %szfull = call i64 @universe_ds_sparseset_size(ptr %f)
  br label %presence

presence:
  %pi = phi i64 [ 0, %checkfull ], [ %pi.n, %presence ]
  %pviol = phi i64 [ 0, %checkfull ], [ %pviol.n, %presence ]
  %pc = call i32 @universe_ds_sparseset_contains(ptr %f, i64 %pi)
  %pbad = icmp ne i32 %pc, 1
  %pinc = zext i1 %pbad to i64
  %pviol.n = add nuw i64 %pviol, %pinc
  %pi.n = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, 300
  br i1 %pmore, label %presence, label %drain

drain:
  %di = phi i64 [ 0, %presence ], [ %di.n, %drain ]
  call i32 @universe_ds_sparseset_remove(ptr %f, i64 %di)
  %di.n = add nuw i64 %di, 1
  %dmore = icmp ult i64 %di.n, 300
  br i1 %dmore, label %drain, label %checkempty

checkempty:
  %szempty = call i64 @universe_ds_sparseset_size(ptr %f)
  br label %absence

absence:
  %bi = phi i64 [ 0, %checkempty ], [ %bi.n, %absence ]
  %bviol = phi i64 [ 0, %checkempty ], [ %bviol.n, %absence ]
  %bc = call i32 @universe_ds_sparseset_contains(ptr %f, i64 %bi)
  %bbad = icmp ne i32 %bc, 0
  %binc = zext i1 %bbad to i64
  %bviol.n = add nuw i64 %bviol, %binc
  %bi.n = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bi.n, 300
  br i1 %bmore, label %absence, label %edge.report

edge.report:
  %f0 = icmp eq i64 %szfull, 300
  %f1 = icmp eq i64 %pviol, 0
  %f2 = icmp eq i64 %szempty, 0
  %f3 = icmp eq i64 %bviol, 0
  %fa = and i1 %f0, %f1
  %fb = and i1 %fa, %f2
  %fc2 = and i1 %fb, %f3
  call void @ut_check(i1 %fc2, ptr @m.full)
  call void @universe_ds_sparseset_destroy(ptr %f)
  ret void
}

; --- bench: throughput + the O(1)-clear win vs a bitset -------------------
define internal void @bench() {
entry:
  %members = alloca [20000 x i32], align 16
  %seed = alloca i64, align 8
  store i64 99991, ptr %seed, align 8
  %s = call ptr @universe_ds_sparseset_create(i64 100000)
  %bs = call ptr @universe_ds_bitset_create(i64 100000)
  br label %gen

gen:
  %gi = phi i64 [ 0, %entry ], [ %gi.n, %gen ]
  %r = call i64 @ut_rand(ptr %seed)
  %m = urem i64 %r, 100000
  %m32 = trunc i64 %m to i32
  %mp = getelementptr inbounds nuw i32, ptr %members, i64 %gi
  store i32 %m32, ptr %mp, align 4
  %gi.n = add nuw i64 %gi, 1
  %gmore = icmp ult i64 %gi.n, 20000
  br i1 %gmore, label %gen, label %warm

warm:
  ; warm up: one add pass into each structure (also leaves both populated).
  %wi = phi i64 [ 0, %gen ], [ %wi.n, %warm ]
  %wp = getelementptr inbounds nuw i32, ptr %members, i64 %wi
  %wv32 = load i32, ptr %wp, align 4
  %wv = zext i32 %wv32 to i64
  call i32 @universe_ds_sparseset_add(ptr %s, i64 %wv)
  call i32 @universe_ds_bitset_set(ptr %bs, i64 %wv)
  %wi.n = add nuw i64 %wi, 1
  %wmore = icmp ult i64 %wi.n, 20000
  br i1 %wmore, label %warm, label %arep

; --- add distribution (clear then 20k adds each rep; destructive) ---
arep:
  %arep.i = phi i64 [ 0, %warm ], [ %arep.n, %arep.next ]
  call i32 @universe_ds_sparseset_clear(ptr %s)
  %at0 = call double @ut_now_sec()
  br label %add.loop

add.loop:
  %ai = phi i64 [ 0, %arep ], [ %ai.n, %add.loop ]
  %ap = getelementptr inbounds nuw i32, ptr %members, i64 %ai
  %av32 = load i32, ptr %ap, align 4
  %av = zext i32 %av32 to i64
  call i32 @universe_ds_sparseset_add(ptr %s, i64 %av)
  %ai.n = add nuw i64 %ai, 1
  %amore = icmp ult i64 %ai.n, 20000
  br i1 %amore, label %add.loop, label %arep.done

arep.done:
  %at1 = call double @ut_now_sec()
  %adt = fsub double %at1, %at0
  %awarm = icmp eq i64 %arep.i, 0
  br i1 %awarm, label %arep.next, label %arep.store

arep.store:
  %asidx = sub i64 %arep.i, 1
  %asp = getelementptr inbounds double, ptr @sparse.addsamp, i64 %asidx
  store double %adt, ptr %asp, align 8
  br label %arep.next

arep.next:
  %arep.n = add nuw nsw i64 %arep.i, 1
  %arep.more = icmp ult i64 %arep.n, 17
  br i1 %arep.more, label %arep, label %areport

areport:
  call void @ut_report_dist(ptr @sparse.addsamp, i64 16, i64 20000, ptr @lbl.sparse_add)
  br label %crep

; --- contains distribution (read-only; s populated by add loop) ---
crep:
  %crep.i = phi i64 [ 0, %areport ], [ %crep.n, %crep.next ]
  %ct0 = call double @ut_now_sec()
  br label %con.loop

con.loop:
  %ci = phi i64 [ 0, %crep ], [ %ci.n, %con.loop ]
  %cacc = phi i64 [ 0, %crep ], [ %cacc.n, %con.loop ]
  %cp = getelementptr inbounds nuw i32, ptr %members, i64 %ci
  %cv32 = load i32, ptr %cp, align 4
  %cv = zext i32 %cv32 to i64
  %hit = call i32 @universe_ds_sparseset_contains(ptr %s, i64 %cv)
  %hit64 = zext i32 %hit to i64
  %cacc.n = add i64 %cacc, %hit64
  %ci.n = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, 20000
  br i1 %cmore, label %con.loop, label %crep.done

crep.done:
  %ct1 = call double @ut_now_sec()
  store volatile i64 %cacc.n, ptr @sparse.sink, align 8
  %cdt = fsub double %ct1, %ct0
  %cwarm = icmp eq i64 %crep.i, 0
  br i1 %cwarm, label %crep.next, label %crep.store

crep.store:
  %csidx = sub i64 %crep.i, 1
  %csp = getelementptr inbounds double, ptr @sparse.consamp, i64 %csidx
  store double %cdt, ptr %csp, align 8
  br label %crep.next

crep.next:
  %crep.n = add nuw nsw i64 %crep.i, 1
  %crep.more = icmp ult i64 %crep.n, 17
  br i1 %crep.more, label %crep, label %creport

creport:
  call void @ut_report_dist(ptr @sparse.consamp, i64 16, i64 20000, ptr @lbl.sparse_con)
  br label %srep

; --- sparse clear distribution (single-store clear; idempotent) ---
srep:
  %srep.i = phi i64 [ 0, %creport ], [ %srep.n, %srep.next ]
  %st0 = call double @ut_now_sec()
  br label %sclr

sclr:
  %si = phi i64 [ 0, %srep ], [ %si.n, %sclr ]
  call i32 @universe_ds_sparseset_clear(ptr %s)
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 200000
  br i1 %smore, label %sclr, label %srep.done

srep.done:
  %st1 = call double @ut_now_sec()
  %sdt = fsub double %st1, %st0
  %swarm = icmp eq i64 %srep.i, 0
  br i1 %swarm, label %srep.next, label %srep.store

srep.store:
  %ssidx = sub i64 %srep.i, 1
  %ssp = getelementptr inbounds double, ptr @sparse.clrsamp, i64 %ssidx
  store double %sdt, ptr %ssp, align 8
  br label %srep.next

srep.next:
  %srep.n = add nuw nsw i64 %srep.i, 1
  %srep.more = icmp ult i64 %srep.n, 17
  br i1 %srep.more, label %srep, label %sreport

sreport:
  call void @ut_report_dist(ptr @sparse.clrsamp, i64 16, i64 200000, ptr @lbl.sparse_clr)
  br label %brep

; --- bitset clear_all distribution (memsets each call; idempotent) ---
brep:
  %brep.i = phi i64 [ 0, %sreport ], [ %brep.n, %brep.next ]
  %bt0 = call double @ut_now_sec()
  br label %bclr

bclr:
  %bi = phi i64 [ 0, %brep ], [ %bi.n, %bclr ]
  call i32 @universe_ds_bitset_clear_all(ptr %bs)
  %bi.n = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bi.n, 200000
  br i1 %bmore, label %bclr, label %brep.done

brep.done:
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  %bwarm = icmp eq i64 %brep.i, 0
  br i1 %bwarm, label %brep.next, label %brep.store

brep.store:
  %bsidx = sub i64 %brep.i, 1
  %bsp = getelementptr inbounds double, ptr @sparse.bclrsamp, i64 %bsidx
  store double %bdt, ptr %bsp, align 8
  br label %brep.next

brep.next:
  %brep.n = add nuw nsw i64 %brep.i, 1
  %brep.more = icmp ult i64 %brep.n, 17
  br i1 %brep.more, label %brep, label %breport

breport:
  call void @ut_report_dist(ptr @sparse.bclrsamp, i64 16, i64 200000, ptr @lbl.bitset_clr)
  call void @universe_ds_sparseset_destroy(ptr %s)
  call void @universe_ds_bitset_destroy(ptr %bs)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_remove()
  call void @test_clear()
  call void @test_iter()
  call void @test_errors()
  call void @test_random()
  call void @test_edges()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
