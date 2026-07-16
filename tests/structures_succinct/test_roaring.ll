; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; See the License for the specific language governing permissions and
; limitations under the License.

; Tests for the Roaring bitmap. Cross-checks every operation against a plain
; dense bitset reference over a bounded universe, exercises both container
; kinds and the array->bitmap promotion, and asserts the dense vector block
; kernels equal their scalar oracle twin.

declare ptr @universe_ds_roaring_create()
declare void @universe_ds_roaring_destroy(ptr)
declare i32 @universe_ds_roaring_add(ptr, i32)
declare i32 @universe_ds_roaring_remove(ptr, i32)
declare i32 @universe_ds_roaring_contains(ptr, i32)
declare i64 @universe_ds_roaring_cardinality(ptr)
declare i64 @universe_ds_roaring_to_array(ptr, ptr, i64)
declare i32 @universe_ds_roaring_container_kind(ptr, i32)
declare ptr @universe_ds_roaring_union(ptr, ptr)
declare ptr @universe_ds_roaring_intersection(ptr, ptr)
declare ptr @universe_ds_roaring_difference(ptr, ptr)
declare void @universe_ds_roaring_bitmap_or(ptr, ptr, ptr)
declare void @universe_ds_roaring_bitmap_and(ptr, ptr, ptr)
declare void @universe_ds_roaring_bitmap_andnot(ptr, ptr, ptr)
declare void @universe_ds_roaring_bitmap_or_scalar(ptr, ptr, ptr)
declare void @universe_ds_roaring_bitmap_and_scalar(ptr, ptr, ptr)
declare void @universe_ds_roaring_bitmap_andnot_scalar(ptr, ptr, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.cttz.i64(i64, i1)

@msg.add.null   = private constant [15 x i8] c"add null == 1\0A\00"
@msg.cont.null  = private constant [20 x i8] c"contains null == 0\0A\00"
@msg.rem.null   = private constant [15 x i8] c"rem null == 1\0A\00"
@msg.card.null  = private constant [16 x i8] c"card null == 0\0A\00"
@msg.kind.null  = private constant [17 x i8] c"kind null == -1\0A\00"
@msg.empty.card = private constant [14 x i8] c"empty card 0\0A\00"
@msg.empty.arr  = private constant [15 x i8] c"empty toarr 0\0A\00"
@msg.empty.cont = private constant [16 x i8] c"empty contains\0A\00"
@msg.one.has    = private constant [12 x i8] c"one has 42\0A\00"
@msg.one.card   = private constant [12 x i8] c"one card 1\0A\00"
@msg.one.miss   = private constant [13 x i8] c"one miss 43\0A\00"
@msg.one.kind   = private constant [14 x i8] c"one is array\0A\00"
@msg.idem       = private constant [16 x i8] c"idempotent add\0A\00"
@msg.rem.gone   = private constant [12 x i8] c"removed 42\0A\00"
@msg.rem.card   = private constant [14 x i8] c"card 0 after\0A\00"
@msg.sparse.k   = private constant [17 x i8] c"sparse is array\0A\00"
@msg.sparse.c   = private constant [13 x i8] c"sparse card\0A\00"
@msg.sparse.m   = private constant [16 x i8] c"sparse members\0A\00"
@msg.prom.arr   = private constant [17 x i8] c"pre-promote arr\0A\00"
@msg.prom.bmp   = private constant [14 x i8] c"promoted bmp\0A\00"
@msg.prom.card  = private constant [14 x i8] c"promote card\0A\00"
@msg.prom.mem   = private constant [14 x i8] c"promote memb\0A\00"
@msg.set1       = private constant [16 x i8] c"single-set cmp\0A\00"
@msg.rem.cmp    = private constant [18 x i8] c"after-remove cmp\0A\00"
@msg.uni        = private constant [11 x i8] c"union cmp\0A\00"
@msg.inter      = private constant [15 x i8] c"intersect cmp\0A\00"
@msg.diff       = private constant [10 x i8] c"diff cmp\0A\00"
@msg.disj       = private constant [16 x i8] c"disjoint union\0A\00"
@msg.vec.or     = private constant [16 x i8] c"vec==scalar or\0A\00"
@msg.vec.and    = private constant [17 x i8] c"vec==scalar and\0A\00"
@msg.vec.andn   = private constant [20 x i8] c"vec==scalar andnot\0A\00"
@lbl.r_add   = private unnamed_addr constant [23 x i8] c"roaring add (1M/rep)\00\00\00"
@lbl.r_union = private unnamed_addr constant [26 x i8] c"roaring union (whole op)\00\00"
@lbl.b_union = private unnamed_addr constant [25 x i8] c"bitset union (whole op)\00\00"
@lbl.r_inter = private unnamed_addr constant [30 x i8] c"roaring intersect (whole op)\00\00"
@lbl.b_inter = private unnamed_addr constant [28 x i8] c"bitset intersect (whole op)\00"
@roar.addsamp = internal global [16 x double] zeroinitializer, align 8
@roar.rusamp  = internal global [16 x double] zeroinitializer, align 8
@roar.busamp  = internal global [16 x double] zeroinitializer, align 8
@roar.risamp  = internal global [16 x double] zeroinitializer, align 8
@roar.bisamp  = internal global [16 x double] zeroinitializer, align 8
@roar.sink    = internal global i64 0, align 8

; ---- reference bitset over i64 words ----

define internal void @tref_set(ptr %bs, i32 %v) {
entry:
  %v64 = zext i32 %v to i64
  %w = lshr i64 %v64, 6
  %b = and i64 %v64, 63
  %mask = shl nuw i64 1, %b
  %wp = getelementptr inbounds i64, ptr %bs, i64 %w
  %cur = load i64, ptr %wp, align 8
  %new = or i64 %cur, %mask
  store i64 %new, ptr %wp, align 8
  ret void
}

define internal void @tref_clear(ptr %bs, i32 %v) {
entry:
  %v64 = zext i32 %v to i64
  %w = lshr i64 %v64, 6
  %b = and i64 %v64, 63
  %mask = shl nuw i64 1, %b
  %nmask = xor i64 %mask, -1
  %wp = getelementptr inbounds i64, ptr %bs, i64 %w
  %cur = load i64, ptr %wp, align 8
  %new = and i64 %cur, %nmask
  store i64 %new, ptr %wp, align 8
  ret void
}

define internal i32 @tref_test(ptr %bs, i32 %v) {
entry:
  %v64 = zext i32 %v to i64
  %w = lshr i64 %v64, 6
  %b = and i64 %v64, 63
  %wp = getelementptr inbounds i64, ptr %bs, i64 %w
  %cur = load i64, ptr %wp, align 8
  %sh = lshr i64 %cur, %b
  %bit = and i64 %sh, 1
  %r = trunc i64 %bit to i32
  ret i32 %r
}

define internal i64 @tref_popcount(ptr %bs, i64 %nwords) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %wp = getelementptr inbounds i64, ptr %bs, i64 %i
  %w = load i64, ptr %wp, align 8
  %pc = call i64 @llvm.ctpop.i64(i64 %w)
  %acc.n = add i64 %acc, %pc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nwords
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

; enumerate set bits ascending into out (u32); return count.
define internal i64 @tref_toarray(ptr %bs, i64 %nwords, ptr %out) {
entry:
  br label %word
word:
  %wi = phi i64 [ 0, %entry ], [ %wi.n, %word.cont ]
  %oi = phi i64 [ 0, %entry ], [ %oi.w, %word.cont ]
  %wp = getelementptr inbounds i64, ptr %bs, i64 %wi
  %w = load i64, ptr %wp, align 8
  %wbase = shl i64 %wi, 6
  br label %bits
bits:
  %cur = phi i64 [ %w, %word ], [ %cur.n, %emit ]
  %oi2 = phi i64 [ %oi, %word ], [ %oi2.n, %emit ]
  %nz = icmp ne i64 %cur, 0
  br i1 %nz, label %emit, label %bits.done
emit:
  %tz = call i64 @llvm.cttz.i64(i64 %cur, i1 true)
  %val = add i64 %wbase, %tz
  %val32 = trunc i64 %val to i32
  %op = getelementptr inbounds i32, ptr %out, i64 %oi2
  store i32 %val32, ptr %op, align 4
  %oi2.n = add nuw i64 %oi2, 1
  %m1 = sub i64 %cur, 1
  %cur.n = and i64 %cur, %m1
  br label %bits
bits.done:
  br label %word.cont
word.cont:
  %oi.w = phi i64 [ %oi2, %bits.done ]
  %wi.n = add nuw i64 %wi, 1
  %more = icmp ult i64 %wi.n, %nwords
  br i1 %more, label %word, label %done
done:
  ret i64 %oi.w
}

; word-parallel reference set algebra. op: 0 or, 1 and, 2 andnot.
define internal void @tref_alg(ptr %d, ptr %a, ptr %b, i64 %nwords, i32 %op) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %av = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds i64, ptr %b, i64 %i
  %bv = load i64, ptr %bp, align 8
  %orv = or i64 %av, %bv
  %andv = and i64 %av, %bv
  %nb = xor i64 %bv, -1
  %anv = and i64 %av, %nb
  %is1 = icmp eq i32 %op, 1
  %is2 = icmp eq i32 %op, 2
  %sel1 = select i1 %is1, i64 %andv, i64 %orv
  %sel2 = select i1 %is2, i64 %anv, i64 %sel1
  %dp = getelementptr inbounds i64, ptr %d, i64 %i
  store i64 %sel2, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nwords
  br i1 %more, label %loop, label %done
done:
  ret void
}

; fill roaring %r and reference %ref with %n random values masked to %mask.
define internal void @t_fill(ptr %r, ptr %ref, i64 %n, i64 %mask, ptr %state) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %rv = call i64 @ut_rand(ptr %state)
  %m = and i64 %rv, %mask
  %v32 = trunc i64 %m to i32
  %arc = call i32 @universe_ds_roaring_add(ptr %r, i32 %v32)
  call void @tref_set(ptr %ref, i32 %v32)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; compare roaring %r against reference %ref (over %nwords words). outR/outRef
; are scratch u32 arrays sized to hold the whole set. returns #mismatches.
define internal i64 @t_cmp(ptr %r, ptr %ref, i64 %nwords, ptr %outR, ptr %outRef) {
entry:
  %pc = call i64 @tref_popcount(ptr %ref, i64 %nwords)
  %rc = call i64 @universe_ds_roaring_cardinality(ptr %r)
  %cardbad = icmp ne i64 %rc, %pc
  %m0 = zext i1 %cardbad to i64
  %bits = shl i64 %nwords, 6
  %nr = call i64 @universe_ds_roaring_to_array(ptr %r, ptr %outR, i64 %bits)
  %nref = call i64 @tref_toarray(ptr %ref, i64 %nwords, ptr %outRef)
  %nrbad = icmp ne i64 %nr, %pc
  %nrefbad = icmp ne i64 %nref, %pc
  %a1 = zext i1 %nrbad to i64
  %a2 = zext i1 %nrefbad to i64
  %m1 = add i64 %m0, %a1
  %m2 = add i64 %m1, %a2
  %empty = icmp eq i64 %pc, 0
  br i1 %empty, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %mm = phi i64 [ %m2, %entry ], [ %mm.n, %loop ]
  %rp = getelementptr inbounds i32, ptr %outR, i64 %i
  %rv = load i32, ptr %rp, align 4
  %fp = getelementptr inbounds i32, ptr %outRef, i64 %i
  %fv = load i32, ptr %fp, align 4
  %ne = icmp ne i32 %rv, %fv
  %inc = zext i1 %ne to i64
  %mm.n = add i64 %mm, %inc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %pc
  br i1 %more, label %loop, label %done
done:
  %res = phi i64 [ %m2, %entry ], [ %mm.n, %loop ]
  ret i64 %res
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %state = alloca i64, align 8

  ; ================= null guards =================
  %g.add = call i32 @universe_ds_roaring_add(ptr null, i32 0)
  %g.add.ok = icmp eq i32 %g.add, 1
  call void @ut_check(i1 %g.add.ok, ptr @msg.add.null)
  %g.cont = call i32 @universe_ds_roaring_contains(ptr null, i32 0)
  %g.cont.ok = icmp eq i32 %g.cont, 0
  call void @ut_check(i1 %g.cont.ok, ptr @msg.cont.null)
  %g.rem = call i32 @universe_ds_roaring_remove(ptr null, i32 0)
  %g.rem.ok = icmp eq i32 %g.rem, 1
  call void @ut_check(i1 %g.rem.ok, ptr @msg.rem.null)
  %g.card = call i64 @universe_ds_roaring_cardinality(ptr null)
  %g.card.ok = icmp eq i64 %g.card, 0
  call void @ut_check(i1 %g.card.ok, ptr @msg.card.null)
  %g.kind = call i32 @universe_ds_roaring_container_kind(ptr null, i32 0)
  %g.kind.ok = icmp eq i32 %g.kind, -1
  call void @ut_check(i1 %g.kind.ok, ptr @msg.kind.null)

  ; ================= empty roaring =================
  %r0 = call ptr @universe_ds_roaring_create()
  %e.card = call i64 @universe_ds_roaring_cardinality(ptr %r0)
  %e.card.ok = icmp eq i64 %e.card, 0
  call void @ut_check(i1 %e.card.ok, ptr @msg.empty.card)
  %smallbuf = call ptr @malloc(i64 64)
  %e.arr = call i64 @universe_ds_roaring_to_array(ptr %r0, ptr %smallbuf, i64 16)
  %e.arr.ok = icmp eq i64 %e.arr, 0
  call void @ut_check(i1 %e.arr.ok, ptr @msg.empty.arr)
  %e.cont = call i32 @universe_ds_roaring_contains(ptr %r0, i32 123)
  %e.cont.ok = icmp eq i32 %e.cont, 0
  call void @ut_check(i1 %e.cont.ok, ptr @msg.empty.cont)

  ; ================= single element =================
  %s.add = call i32 @universe_ds_roaring_add(ptr %r0, i32 42)
  %s.has = call i32 @universe_ds_roaring_contains(ptr %r0, i32 42)
  %s.has.ok = icmp eq i32 %s.has, 1
  call void @ut_check(i1 %s.has.ok, ptr @msg.one.has)
  %s.card = call i64 @universe_ds_roaring_cardinality(ptr %r0)
  %s.card.ok = icmp eq i64 %s.card, 1
  call void @ut_check(i1 %s.card.ok, ptr @msg.one.card)
  %s.miss = call i32 @universe_ds_roaring_contains(ptr %r0, i32 43)
  %s.miss.ok = icmp eq i32 %s.miss, 0
  call void @ut_check(i1 %s.miss.ok, ptr @msg.one.miss)
  %s.kind = call i32 @universe_ds_roaring_container_kind(ptr %r0, i32 42)
  %s.kind.ok = icmp eq i32 %s.kind, 0
  call void @ut_check(i1 %s.kind.ok, ptr @msg.one.kind)
  ; idempotent
  %s.add2 = call i32 @universe_ds_roaring_add(ptr %r0, i32 42)
  %s.card2 = call i64 @universe_ds_roaring_cardinality(ptr %r0)
  %s.card2.ok = icmp eq i64 %s.card2, 1
  call void @ut_check(i1 %s.card2.ok, ptr @msg.idem)
  ; remove
  %s.rem = call i32 @universe_ds_roaring_remove(ptr %r0, i32 42)
  %s.gone = call i32 @universe_ds_roaring_contains(ptr %r0, i32 42)
  %s.gone.ok = icmp eq i32 %s.gone, 0
  call void @ut_check(i1 %s.gone.ok, ptr @msg.rem.gone)
  %s.card3 = call i64 @universe_ds_roaring_cardinality(ptr %r0)
  %s.card3.ok = icmp eq i64 %s.card3, 0
  call void @ut_check(i1 %s.card3.ok, ptr @msg.rem.card)
  call void @universe_ds_roaring_destroy(ptr %r0)

  ; ================= sparse chunk (array container) =================
  ; 100 values in chunk key=7: base = 7<<16 = 458752.
  %rs = call ptr @universe_ds_roaring_create()
  br label %sp.loop
sp.loop:
  %sp.i = phi i32 [ 0, %entry ], [ %sp.i.n, %sp.loop ]
  %sp.v = add i32 458752, %sp.i
  %sp.rc = call i32 @universe_ds_roaring_add(ptr %rs, i32 %sp.v)
  %sp.i.n = add nuw i32 %sp.i, 1
  %sp.more = icmp ult i32 %sp.i.n, 100
  br i1 %sp.more, label %sp.loop, label %sp.check
sp.check:
  %sp.kind = call i32 @universe_ds_roaring_container_kind(ptr %rs, i32 458752)
  %sp.kind.ok = icmp eq i32 %sp.kind, 0
  call void @ut_check(i1 %sp.kind.ok, ptr @msg.sparse.k)
  %sp.card = call i64 @universe_ds_roaring_cardinality(ptr %rs)
  %sp.card.ok = icmp eq i64 %sp.card, 100
  call void @ut_check(i1 %sp.card.ok, ptr @msg.sparse.c)
  ; verify members
  br label %sp.mem
sp.mem:
  %sm.i = phi i32 [ 0, %sp.check ], [ %sm.i.n, %sp.mem ]
  %sm.miss = phi i64 [ 0, %sp.check ], [ %sm.miss.n, %sp.mem ]
  %sm.v = add i32 458752, %sm.i
  %sm.has = call i32 @universe_ds_roaring_contains(ptr %rs, i32 %sm.v)
  %sm.bad = icmp ne i32 %sm.has, 1
  %sm.inc = zext i1 %sm.bad to i64
  %sm.miss.n = add i64 %sm.miss, %sm.inc
  %sm.i.n = add nuw i32 %sm.i, 1
  %sm.more = icmp ult i32 %sm.i.n, 100
  br i1 %sm.more, label %sp.mem, label %sp.mem.done
sp.mem.done:
  call void @ut_check_eq(i64 %sm.miss.n, i64 0, ptr @msg.sparse.m)
  call void @universe_ds_roaring_destroy(ptr %rs)

  ; ================= array->bitmap promotion =================
  ; add 0..4095 into chunk 0 -> array; then 4096 -> promote to bitmap.
  %rp = call ptr @universe_ds_roaring_create()
  br label %pr.loop
pr.loop:
  %pr.i = phi i32 [ 0, %sp.mem.done ], [ %pr.i.n, %pr.loop ]
  %pr.rc = call i32 @universe_ds_roaring_add(ptr %rp, i32 %pr.i)
  %pr.i.n = add nuw i32 %pr.i, 1
  %pr.more = icmp ult i32 %pr.i.n, 4096
  br i1 %pr.more, label %pr.loop, label %pr.check
pr.check:
  %pr.k1 = call i32 @universe_ds_roaring_container_kind(ptr %rp, i32 0)
  %pr.k1.ok = icmp eq i32 %pr.k1, 0
  call void @ut_check(i1 %pr.k1.ok, ptr @msg.prom.arr)
  %pr.add = call i32 @universe_ds_roaring_add(ptr %rp, i32 4096)
  %pr.k2 = call i32 @universe_ds_roaring_container_kind(ptr %rp, i32 0)
  %pr.k2.ok = icmp eq i32 %pr.k2, 1
  call void @ut_check(i1 %pr.k2.ok, ptr @msg.prom.bmp)
  %pr.card = call i64 @universe_ds_roaring_cardinality(ptr %rp)
  %pr.card.ok = icmp eq i64 %pr.card, 4097
  call void @ut_check(i1 %pr.card.ok, ptr @msg.prom.card)
  ; verify all 0..4096 present after promotion
  br label %pr.mem
pr.mem:
  %pm.i = phi i32 [ 0, %pr.check ], [ %pm.i.n, %pr.mem ]
  %pm.miss = phi i64 [ 0, %pr.check ], [ %pm.miss.n, %pr.mem ]
  %pm.has = call i32 @universe_ds_roaring_contains(ptr %rp, i32 %pm.i)
  %pm.bad = icmp ne i32 %pm.has, 1
  %pm.inc = zext i1 %pm.bad to i64
  %pm.miss.n = add i64 %pm.miss, %pm.inc
  %pm.i.n = add nuw i32 %pm.i, 1
  %pm.more = icmp ult i32 %pm.i.n, 4097
  br i1 %pm.more, label %pr.mem, label %pr.mem.done
pr.mem.done:
  call void @ut_check_eq(i64 %pm.miss.n, i64 0, ptr @msg.prom.mem)
  call void @universe_ds_roaring_destroy(ptr %rp)

  ; ================= single-set random cross-check =================
  ; universe U = 2^18 (262144), UWORDS = 4096. keys 0..3, mostly bitmap.
  %refA = call ptr @calloc(i64 4096, i64 8)
  %outR = call ptr @malloc(i64 1048576)     ; 262144 * 4
  %outF = call ptr @malloc(i64 1048576)
  %ra = call ptr @universe_ds_roaring_create()
  store i64 88172645463325252, ptr %state, align 8
  call void @t_fill(ptr %ra, ptr %refA, i64 30000, i64 262143, ptr %state)
  %cmp1 = call i64 @t_cmp(ptr %ra, ptr %refA, i64 4096, ptr %outR, ptr %outF)
  call void @ut_check_eq(i64 %cmp1, i64 0, ptr @msg.set1)

  ; remove every 5th enumerated value, keep reference in sync, re-check.
  %rcount = call i64 @universe_ds_roaring_to_array(ptr %ra, ptr %outR, i64 262144)
  br label %rm.loop
rm.loop:
  %rm.i = phi i64 [ 0, %pr.mem.done ], [ %rm.i.n, %rm.loop ]
  %rm.p = getelementptr inbounds i32, ptr %outR, i64 %rm.i
  %rm.v = load i32, ptr %rm.p, align 4
  call i32 @universe_ds_roaring_remove(ptr %ra, i32 %rm.v)
  call void @tref_clear(ptr %refA, i32 %rm.v)
  %rm.i.n = add nuw i64 %rm.i, 5
  %rm.more = icmp ult i64 %rm.i.n, %rcount
  br i1 %rm.more, label %rm.loop, label %rm.done
rm.done:
  %cmp2 = call i64 @t_cmp(ptr %ra, ptr %refA, i64 4096, ptr %outR, ptr %outF)
  call void @ut_check_eq(i64 %cmp2, i64 0, ptr @msg.rem.cmp)

  ; ================= set algebra vs reference =================
  %refB = call ptr @calloc(i64 4096, i64 8)
  %refC = call ptr @calloc(i64 4096, i64 8)      ; scratch for op result
  %refA2 = call ptr @calloc(i64 4096, i64 8)
  %rA = call ptr @universe_ds_roaring_create()
  %rB = call ptr @universe_ds_roaring_create()
  store i64 12345678901234567, ptr %state, align 8
  call void @t_fill(ptr %rA, ptr %refA2, i64 12000, i64 262143, ptr %state)
  call void @t_fill(ptr %rB, ptr %refB, i64 12000, i64 262143, ptr %state)

  ; union
  %rU = call ptr @universe_ds_roaring_union(ptr %rA, ptr %rB)
  call void @tref_alg(ptr %refC, ptr %refA2, ptr %refB, i64 4096, i32 0)
  %cmpU = call i64 @t_cmp(ptr %rU, ptr %refC, i64 4096, ptr %outR, ptr %outF)
  call void @ut_check_eq(i64 %cmpU, i64 0, ptr @msg.uni)
  call void @universe_ds_roaring_destroy(ptr %rU)

  ; intersection
  %rI = call ptr @universe_ds_roaring_intersection(ptr %rA, ptr %rB)
  call void @tref_alg(ptr %refC, ptr %refA2, ptr %refB, i64 4096, i32 1)
  %cmpI = call i64 @t_cmp(ptr %rI, ptr %refC, i64 4096, ptr %outR, ptr %outF)
  call void @ut_check_eq(i64 %cmpI, i64 0, ptr @msg.inter)
  call void @universe_ds_roaring_destroy(ptr %rI)

  ; difference
  %rD = call ptr @universe_ds_roaring_difference(ptr %rA, ptr %rB)
  call void @tref_alg(ptr %refC, ptr %refA2, ptr %refB, i64 4096, i32 2)
  %cmpD = call i64 @t_cmp(ptr %rD, ptr %refC, i64 4096, ptr %outR, ptr %outF)
  call void @ut_check_eq(i64 %cmpD, i64 0, ptr @msg.diff)
  call void @universe_ds_roaring_destroy(ptr %rD)

  call void @universe_ds_roaring_destroy(ptr %rA)
  call void @universe_ds_roaring_destroy(ptr %rB)

  ; ================= disjoint chunk keys =================
  %dA = call ptr @universe_ds_roaring_create()
  %dB = call ptr @universe_ds_roaring_create()
  ; A in chunk 1 (65536..), B in chunk 2 (131072..)
  call i32 @universe_ds_roaring_add(ptr %dA, i32 65536)
  call i32 @universe_ds_roaring_add(ptr %dA, i32 65600)
  call i32 @universe_ds_roaring_add(ptr %dB, i32 131072)
  %dU = call ptr @universe_ds_roaring_union(ptr %dA, ptr %dB)
  %d.card = call i64 @universe_ds_roaring_cardinality(ptr %dU)
  %d.h1 = call i32 @universe_ds_roaring_contains(ptr %dU, i32 65536)
  %d.h2 = call i32 @universe_ds_roaring_contains(ptr %dU, i32 131072)
  %d.c.ok = icmp eq i64 %d.card, 3
  %d.h1.ok = icmp eq i32 %d.h1, 1
  %d.h2.ok = icmp eq i32 %d.h2, 1
  %d.ok1 = and i1 %d.c.ok, %d.h1.ok
  %d.ok = and i1 %d.ok1, %d.h2.ok
  call void @ut_check(i1 %d.ok, ptr @msg.disj)
  call void @universe_ds_roaring_destroy(ptr %dA)
  call void @universe_ds_roaring_destroy(ptr %dB)
  call void @universe_ds_roaring_destroy(ptr %dU)

  ; ================= dense block kernels: vector == scalar =================
  %wA = call ptr @malloc(i64 8192)
  %wB = call ptr @malloc(i64 8192)
  %dv = call ptr @malloc(i64 8192)
  %ds = call ptr @malloc(i64 8192)
  store i64 999999937, ptr %state, align 8
  br label %fillw
fillw:
  %fw.i = phi i64 [ 0, %rm.done ], [ %fw.i.n, %fillw ]
  %fw.a = call i64 @ut_rand(ptr %state)
  %fw.b = call i64 @ut_rand(ptr %state)
  %fw.ap = getelementptr inbounds i64, ptr %wA, i64 %fw.i
  store i64 %fw.a, ptr %fw.ap, align 8
  %fw.bp = getelementptr inbounds i64, ptr %wB, i64 %fw.i
  store i64 %fw.b, ptr %fw.bp, align 8
  %fw.i.n = add nuw i64 %fw.i, 1
  %fw.more = icmp ult i64 %fw.i.n, 1024
  br i1 %fw.more, label %fillw, label %vec.or
vec.or:
  call void @universe_ds_roaring_bitmap_or(ptr %dv, ptr %wA, ptr %wB)
  call void @universe_ds_roaring_bitmap_or_scalar(ptr %ds, ptr %wA, ptr %wB)
  %mor = call i64 @wcmp(ptr %dv, ptr %ds)
  call void @ut_check_eq(i64 %mor, i64 0, ptr @msg.vec.or)
  call void @universe_ds_roaring_bitmap_and(ptr %dv, ptr %wA, ptr %wB)
  call void @universe_ds_roaring_bitmap_and_scalar(ptr %ds, ptr %wA, ptr %wB)
  %mand = call i64 @wcmp(ptr %dv, ptr %ds)
  call void @ut_check_eq(i64 %mand, i64 0, ptr @msg.vec.and)
  call void @universe_ds_roaring_bitmap_andnot(ptr %dv, ptr %wA, ptr %wB)
  call void @universe_ds_roaring_bitmap_andnot_scalar(ptr %ds, ptr %wA, ptr %wB)
  %mandn = call i64 @wcmp(ptr %dv, ptr %ds)
  call void @ut_check_eq(i64 %mandn, i64 0, ptr @msg.vec.andn)

  ; cleanup
  call void @free(ptr %smallbuf)
  call void @free(ptr %refA)
  call void @free(ptr %refB)
  call void @free(ptr %refC)
  call void @free(ptr %refA2)
  call void @free(ptr %outR)
  call void @free(ptr %outF)
  call void @free(ptr %wA)
  call void @free(ptr %wB)
  call void @free(ptr %dv)
  call void @free(ptr %ds)
  call void @universe_ds_roaring_destroy(ptr %ra)

  ; ================= bench =================
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary

do.bench:
  call void @bench_run(ptr %state)
  br label %summary

summary:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; compare two 1024-word buffers; return #differing words.
define internal i64 @wcmp(ptr %a, ptr %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %av = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds i64, ptr %b, i64 %i
  %bv = load i64, ptr %bp, align 8
  %ne = icmp ne i64 %av, %bv
  %inc = zext i1 %ne to i64
  %acc.n = add i64 %acc, %inc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1024
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

define internal void @bench_run(ptr %state) {
entry:
  ; universe 2^24 -> 256 chunks; UWORDS = 262144 for the bitset reference.
  %bsA = call ptr @calloc(i64 262144, i64 8)
  %bsB = call ptr @calloc(i64 262144, i64 8)
  %bsR = call ptr @calloc(i64 262144, i64 8)
  %rA = call ptr @universe_ds_roaring_create()
  %rB = call ptr @universe_ds_roaring_create()

  ; ---- setup (untimed): build rA/bsA from seed S, then rB/bsB from the
  ; continued RNG stream. The add distribution below replays the SAME 1M
  ; draws (reseed to S) into throwaway sets so rA/bsA stay intact for the
  ; union/intersect distributions. ----
  store i64 424242424242, ptr %state, align 8
  br label %refA
refA:
  %ra.i = phi i64 [ 0, %entry ], [ %ra.i.n, %refA ]
  %ra.rv = call i64 @ut_rand(ptr %state)
  %ra.m = and i64 %ra.rv, 16777215
  %ra.v = trunc i64 %ra.m to i32
  call i32 @universe_ds_roaring_add(ptr %rA, i32 %ra.v)
  call void @tref_set(ptr %bsA, i32 %ra.v)
  %ra.i.n = add nuw i64 %ra.i, 1
  %ra.more = icmp ult i64 %ra.i.n, 1000000
  br i1 %ra.more, label %refA, label %refB
refB:
  %rb.i = phi i64 [ 0, %refA ], [ %rb.i.n, %refB ]
  %rb.rv = call i64 @ut_rand(ptr %state)
  %rb.m = and i64 %rb.rv, 16777215
  %rb.v = trunc i64 %rb.m to i32
  call i32 @universe_ds_roaring_add(ptr %rB, i32 %rb.v)
  call void @tref_set(ptr %bsB, i32 %rb.v)
  %rb.i.n = add nuw i64 %rb.i, 1
  %rb.more = icmp ult i64 %rb.i.n, 1000000
  br i1 %rb.more, label %refB, label %arep

  ; ---- roaring add distribution (throwaway set, replays 1M draws each rep) ----
arep:
  %arep.i = phi i64 [ 0, %refB ], [ %arep.n, %arep.next ]
  %rT = call ptr @universe_ds_roaring_create()
  store i64 424242424242, ptr %state, align 8
  %at0 = call double @ut_now_sec()
  br label %addT
addT:
  %at.i = phi i64 [ 0, %arep ], [ %at.i.n, %addT ]
  %at.rv = call i64 @ut_rand(ptr %state)
  %at.m = and i64 %at.rv, 16777215
  %at.v = trunc i64 %at.m to i32
  call i32 @universe_ds_roaring_add(ptr %rT, i32 %at.v)
  %at.i.n = add nuw i64 %at.i, 1
  %at.more = icmp ult i64 %at.i.n, 1000000
  br i1 %at.more, label %addT, label %arep.done
arep.done:
  %at1 = call double @ut_now_sec()
  %acard = call i64 @universe_ds_roaring_cardinality(ptr %rT)
  store volatile i64 %acard, ptr @roar.sink, align 8
  call void @universe_ds_roaring_destroy(ptr %rT)
  %adt = fsub double %at1, %at0
  %awarm = icmp eq i64 %arep.i, 0
  br i1 %awarm, label %arep.next, label %arep.store
arep.store:
  %asidx = sub i64 %arep.i, 1
  %asp = getelementptr inbounds double, ptr @roar.addsamp, i64 %asidx
  store double %adt, ptr %asp, align 8
  br label %arep.next
arep.next:
  %arep.n = add nuw nsw i64 %arep.i, 1
  %arep.more = icmp ult i64 %arep.n, 17
  br i1 %arep.more, label %arep, label %ureport.pre
ureport.pre:
  call void @ut_report_dist(ptr @roar.addsamp, i64 16, i64 1000000, ptr @lbl.r_add)
  br label %urep

  ; ---- roaring union distribution ----
urep:
  %urep.i = phi i64 [ 0, %ureport.pre ], [ %urep.n, %urep.next ]
  %ut0 = call double @ut_now_sec()
  %rU = call ptr @universe_ds_roaring_union(ptr %rA, ptr %rB)
  %ut1 = call double @ut_now_sec()
  %ucard = call i64 @universe_ds_roaring_cardinality(ptr %rU)
  store volatile i64 %ucard, ptr @roar.sink, align 8
  call void @universe_ds_roaring_destroy(ptr %rU)
  %udt = fsub double %ut1, %ut0
  %uwarm = icmp eq i64 %urep.i, 0
  br i1 %uwarm, label %urep.next, label %urep.store
urep.store:
  %usidx = sub i64 %urep.i, 1
  %usp = getelementptr inbounds double, ptr @roar.rusamp, i64 %usidx
  store double %udt, ptr %usp, align 8
  br label %urep.next
urep.next:
  %urep.n = add nuw nsw i64 %urep.i, 1
  %urep.more = icmp ult i64 %urep.n, 17
  br i1 %urep.more, label %urep, label %ureport
ureport:
  call void @ut_report_dist(ptr @roar.rusamp, i64 16, i64 1, ptr @lbl.r_union)
  br label %burep

  ; ---- bitset union distribution ----
burep:
  %burep.i = phi i64 [ 0, %ureport ], [ %burep.n, %burep.next ]
  %but0 = call double @ut_now_sec()
  call void @tref_alg(ptr %bsR, ptr %bsA, ptr %bsB, i64 262144, i32 0)
  %but1 = call double @ut_now_sec()
  %bucard = call i64 @tref_popcount(ptr %bsR, i64 262144)
  store volatile i64 %bucard, ptr @roar.sink, align 8
  %budt = fsub double %but1, %but0
  %buwarm = icmp eq i64 %burep.i, 0
  br i1 %buwarm, label %burep.next, label %burep.store
burep.store:
  %busidx = sub i64 %burep.i, 1
  %busp = getelementptr inbounds double, ptr @roar.busamp, i64 %busidx
  store double %budt, ptr %busp, align 8
  br label %burep.next
burep.next:
  %burep.n = add nuw nsw i64 %burep.i, 1
  %burep.more = icmp ult i64 %burep.n, 17
  br i1 %burep.more, label %burep, label %bureport
bureport:
  call void @ut_report_dist(ptr @roar.busamp, i64 16, i64 1, ptr @lbl.b_union)
  br label %nrep

  ; ---- roaring intersect distribution ----
nrep:
  %nrep.i = phi i64 [ 0, %bureport ], [ %nrep.n, %nrep.next ]
  %nt0 = call double @ut_now_sec()
  %rN = call ptr @universe_ds_roaring_intersection(ptr %rA, ptr %rB)
  %nt1 = call double @ut_now_sec()
  %ncard = call i64 @universe_ds_roaring_cardinality(ptr %rN)
  store volatile i64 %ncard, ptr @roar.sink, align 8
  call void @universe_ds_roaring_destroy(ptr %rN)
  %ndt = fsub double %nt1, %nt0
  %nwarm = icmp eq i64 %nrep.i, 0
  br i1 %nwarm, label %nrep.next, label %nrep.store
nrep.store:
  %nsidx = sub i64 %nrep.i, 1
  %nsp = getelementptr inbounds double, ptr @roar.risamp, i64 %nsidx
  store double %ndt, ptr %nsp, align 8
  br label %nrep.next
nrep.next:
  %nrep.n = add nuw nsw i64 %nrep.i, 1
  %nrep.more = icmp ult i64 %nrep.n, 17
  br i1 %nrep.more, label %nrep, label %nreport
nreport:
  call void @ut_report_dist(ptr @roar.risamp, i64 16, i64 1, ptr @lbl.r_inter)
  br label %bnrep

  ; ---- bitset intersect distribution ----
bnrep:
  %bnrep.i = phi i64 [ 0, %nreport ], [ %bnrep.n, %bnrep.next ]
  %bnt0 = call double @ut_now_sec()
  call void @tref_alg(ptr %bsR, ptr %bsA, ptr %bsB, i64 262144, i32 1)
  %bnt1 = call double @ut_now_sec()
  %bncard = call i64 @tref_popcount(ptr %bsR, i64 262144)
  store volatile i64 %bncard, ptr @roar.sink, align 8
  %bndt = fsub double %bnt1, %bnt0
  %bnwarm = icmp eq i64 %bnrep.i, 0
  br i1 %bnwarm, label %bnrep.next, label %bnrep.store
bnrep.store:
  %bnsidx = sub i64 %bnrep.i, 1
  %bnsp = getelementptr inbounds double, ptr @roar.bisamp, i64 %bnsidx
  store double %bndt, ptr %bnsp, align 8
  br label %bnrep.next
bnrep.next:
  %bnrep.n = add nuw nsw i64 %bnrep.i, 1
  %bnrep.more = icmp ult i64 %bnrep.n, 17
  br i1 %bnrep.more, label %bnrep, label %bnreport
bnreport:
  call void @ut_report_dist(ptr @roar.bisamp, i64 16, i64 1, ptr @lbl.b_inter)

  call void @universe_ds_roaring_destroy(ptr %rA)
  call void @universe_ds_roaring_destroy(ptr %rB)
  call void @free(ptr %bsA)
  call void @free(ptr %bsB)
  call void @free(ptr %bsR)
  ret void
}
