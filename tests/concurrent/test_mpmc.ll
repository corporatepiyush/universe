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

; Tests for universe_conc_spmc / universe_conc_mpmc (per-slot-sequence rings).
;   * single-thread correctness: FIFO order, FULL/EMPTY, wrap, count.
;   * pthread CONSERVATION stress (the gate): N producers x 4 consumers each
;     produce tagged items (id<<32 | seq); consumers drain and set a bit in a
;     global bitmap per item. Every item must be consumed EXACTLY ONCE — a
;     re-set bit is a DUPLICATE, a final set-count != total is a LOSS. Per
;     consumer we also verify per-producer seq is monotone (per-producer FIFO).
;     Looped 10x at -O0 and -O3 on ARM64's weak memory model.

declare ptr  @universe_conc_mpmc_create(i64, i64)
declare i32  @universe_conc_mpmc_enqueue(ptr, ptr)
declare i32  @universe_conc_mpmc_dequeue(ptr, ptr)
declare i64  @universe_conc_mpmc_count(ptr)
declare i64  @universe_conc_mpmc_capacity(ptr)
declare void @universe_conc_mpmc_destroy(ptr)

declare ptr  @universe_conc_spmc_create(i64, i64)
declare i32  @universe_conc_spmc_enqueue(ptr, ptr)
declare i32  @universe_conc_spmc_dequeue(ptr, ptr)
declare i64  @universe_conc_spmc_count(ptr)
declare i64  @universe_conc_spmc_capacity(ptr)
declare void @universe_conc_spmc_destroy(ptr)

declare ptr  @malloc(i64)
declare void @free(ptr)
declare i32  @pthread_create(ptr, ptr, ptr, ptr)
declare i32  @pthread_join(i64, ptr)
declare i32  @sched_yield()
declare i32  @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64  @llvm.ctpop.i64(i64)

declare void   @ut_check(i1, ptr)
declare void   @ut_check_eq(i64, i64, ptr)
declare i32    @ut_summary()
declare i1     @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void   @ut_report_dist(ptr, i64, i64, ptr)

@mpmc.samp  = internal global [16 x double] zeroinitializer, align 8
@lbl.mpmc   = private unnamed_addr constant [22 x i8] c"mpmc 4x4 enqueue+deq \00"

; ---- shared stress config / accumulators ----------------------------------
@g.q        = internal global ptr null, align 8
@g.nper     = internal global i64 0, align 8      ; items per producer
@g.nprod    = internal global i64 0, align 8      ; producer count
@g.consumed = internal global i64 0, align 8
@g.dup      = internal global i64 0, align 8      ; duplicate-item violations
@g.order    = internal global i64 0, align 8      ; per-producer order violations
@g.pdone    = internal global i64 0, align 8      ; producers finished
@g.bitmap   = internal global ptr null, align 8   ; 6250 x i64 = 400000 bits

@m.mpmc.single = private unnamed_addr constant [22 x i8] c"mpmc single-thread ok\00"
@m.spmc.single = private unnamed_addr constant [22 x i8] c"spmc single-thread ok\00"
@m.null        = private unnamed_addr constant [16 x i8] c"null-arg guards\00"
@m.mpmc.stress = private unnamed_addr constant [25 x i8] c"mpmc 4x4 conservation ok\00"
@m.spmc.stress = private unnamed_addr constant [25 x i8] c"spmc 1x4 conservation ok\00"

; ===========================================================================
; single-thread correctness (MPMC)
; ===========================================================================
define internal void @test_single_mpmc() {
entry:
  %q = call ptr @universe_conc_mpmc_create(i64 8, i64 8)
  %v = alloca i64, align 8
  %cap = call i64 @universe_conc_mpmc_capacity(ptr %q)
  %cap.ok = icmp eq i64 %cap, 8
  %vcap = zext i1 %cap.ok to i64          ; want 1
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %vf = phi i64 [ 0, %entry ], [ %vf.n, %fill ]
  store i64 %i, ptr %v, align 8
  %rf = call i32 @universe_conc_mpmc_enqueue(ptr %q, ptr %v)
  %bad.f = icmp ne i32 %rf, 0
  %inc.f = zext i1 %bad.f to i64
  %vf.n = add i64 %vf, %inc.f
  %i.n = add nuw i64 %i, 1
  %more.f = icmp ult i64 %i.n, 8
  br i1 %more.f, label %fill, label %full

full:
  store i64 99, ptr %v, align 8
  %rfull = call i32 @universe_conc_mpmc_enqueue(ptr %q, ptr %v)
  %full.ok = icmp eq i32 %rfull, 6
  %vfull = zext i1 %full.ok to i64
  %cnt = call i64 @universe_conc_mpmc_count(ptr %q)
  %cnt.ok = icmp eq i64 %cnt, 8
  %vcnt = zext i1 %cnt.ok to i64
  br label %drain

drain:
  %j = phi i64 [ 0, %full ], [ %j.n, %drain ]
  %vd = phi i64 [ 0, %full ], [ %vd.n, %drain ]
  %rd = call i32 @universe_conc_mpmc_dequeue(ptr %q, ptr %v)
  %got = load i64, ptr %v, align 8
  %bad.rd = icmp ne i32 %rd, 0
  %bad.ord = icmp ne i64 %got, %j
  %bad.d = or i1 %bad.rd, %bad.ord
  %inc.d = zext i1 %bad.d to i64
  %vd.n = add i64 %vd, %inc.d
  %j.n = add nuw i64 %j, 1
  %more.d = icmp ult i64 %j.n, 8
  br i1 %more.d, label %drain, label %empty

empty:
  %rempty = call i32 @universe_conc_mpmc_dequeue(ptr %q, ptr %v)
  %empty.ok = icmp eq i32 %rempty, 4
  %vempty = zext i1 %empty.ok to i64
  br label %wrap

wrap:                                   ; re-enqueue 8, slots wrap via mask
  %k = phi i64 [ 0, %empty ], [ %k.n, %wrap ]
  %vw = phi i64 [ 0, %empty ], [ %vw.n, %wrap ]
  store i64 %k, ptr %v, align 8
  %rw = call i32 @universe_conc_mpmc_enqueue(ptr %q, ptr %v)
  %bad.w = icmp ne i32 %rw, 0
  %inc.w = zext i1 %bad.w to i64
  %vw.n = add i64 %vw, %inc.w
  %k.n = add nuw i64 %k, 1
  %more.w = icmp ult i64 %k.n, 8
  br i1 %more.w, label %wrap, label %wdrain

wdrain:
  %m = phi i64 [ 0, %wrap ], [ %m.n, %wdrain ]
  %vwd = phi i64 [ 0, %wrap ], [ %vwd.n, %wdrain ]
  %rwd = call i32 @universe_conc_mpmc_dequeue(ptr %q, ptr %v)
  %gotw = load i64, ptr %v, align 8
  %bad.rwd = icmp ne i32 %rwd, 0
  %bad.ordw = icmp ne i64 %gotw, %m
  %bad.wd = or i1 %bad.rwd, %bad.ordw
  %inc.wd = zext i1 %bad.wd to i64
  %vwd.n = add i64 %vwd, %inc.wd
  %m.n = add nuw i64 %m, 1
  %more.wd = icmp ult i64 %m.n, 8
  br i1 %more.wd, label %wdrain, label %verdict

verdict:
  ; want all-clear: vcap==1 and every violation counter==0 and ok-flags set
  %s0 = add i64 %vf.n, %vd.n
  %s1 = add i64 %s0, %vw.n
  %s2 = add i64 %s1, %vwd.n
  %oktot = add i64 %vfull, %vcnt
  %oktot2 = add i64 %oktot, %vempty
  %oktot3 = add i64 %oktot2, %vcap
  ; oktot3 should be 4 (four ok flags), s2 should be 0
  %ok.viol = icmp ne i64 %oktot3, 4
  %cnt.viol = icmp ne i64 %s2, 0
  %anybad = or i1 %ok.viol, %cnt.viol
  %pass = xor i1 %anybad, true
  call void @ut_check(i1 %pass, ptr @m.mpmc.single)
  call void @universe_conc_mpmc_destroy(ptr %q)
  ret void
}

; ===========================================================================
; single-thread correctness (SPMC) — identical shape, spmc_* API
; ===========================================================================
define internal void @test_single_spmc() {
entry:
  %q = call ptr @universe_conc_spmc_create(i64 8, i64 8)
  %v = alloca i64, align 8
  %cap = call i64 @universe_conc_spmc_capacity(ptr %q)
  %cap.ok = icmp eq i64 %cap, 8
  %vcap = zext i1 %cap.ok to i64
  br label %fill

fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %vf = phi i64 [ 0, %entry ], [ %vf.n, %fill ]
  store i64 %i, ptr %v, align 8
  %rf = call i32 @universe_conc_spmc_enqueue(ptr %q, ptr %v)
  %bad.f = icmp ne i32 %rf, 0
  %inc.f = zext i1 %bad.f to i64
  %vf.n = add i64 %vf, %inc.f
  %i.n = add nuw i64 %i, 1
  %more.f = icmp ult i64 %i.n, 8
  br i1 %more.f, label %fill, label %full

full:
  store i64 99, ptr %v, align 8
  %rfull = call i32 @universe_conc_spmc_enqueue(ptr %q, ptr %v)
  %full.ok = icmp eq i32 %rfull, 6
  %vfull = zext i1 %full.ok to i64
  %cnt = call i64 @universe_conc_spmc_count(ptr %q)
  %cnt.ok = icmp eq i64 %cnt, 8
  %vcnt = zext i1 %cnt.ok to i64
  br label %drain

drain:
  %j = phi i64 [ 0, %full ], [ %j.n, %drain ]
  %vd = phi i64 [ 0, %full ], [ %vd.n, %drain ]
  %rd = call i32 @universe_conc_spmc_dequeue(ptr %q, ptr %v)
  %got = load i64, ptr %v, align 8
  %bad.rd = icmp ne i32 %rd, 0
  %bad.ord = icmp ne i64 %got, %j
  %bad.d = or i1 %bad.rd, %bad.ord
  %inc.d = zext i1 %bad.d to i64
  %vd.n = add i64 %vd, %inc.d
  %j.n = add nuw i64 %j, 1
  %more.d = icmp ult i64 %j.n, 8
  br i1 %more.d, label %drain, label %empty

empty:
  %rempty = call i32 @universe_conc_spmc_dequeue(ptr %q, ptr %v)
  %empty.ok = icmp eq i32 %rempty, 4
  %vempty = zext i1 %empty.ok to i64
  br label %wrap

wrap:
  %k = phi i64 [ 0, %empty ], [ %k.n, %wrap ]
  %vw = phi i64 [ 0, %empty ], [ %vw.n, %wrap ]
  store i64 %k, ptr %v, align 8
  %rw = call i32 @universe_conc_spmc_enqueue(ptr %q, ptr %v)
  %bad.w = icmp ne i32 %rw, 0
  %inc.w = zext i1 %bad.w to i64
  %vw.n = add i64 %vw, %inc.w
  %k.n = add nuw i64 %k, 1
  %more.w = icmp ult i64 %k.n, 8
  br i1 %more.w, label %wrap, label %wdrain

wdrain:
  %m = phi i64 [ 0, %wrap ], [ %m.n, %wdrain ]
  %vwd = phi i64 [ 0, %wrap ], [ %vwd.n, %wdrain ]
  %rwd = call i32 @universe_conc_spmc_dequeue(ptr %q, ptr %v)
  %gotw = load i64, ptr %v, align 8
  %bad.rwd = icmp ne i32 %rwd, 0
  %bad.ordw = icmp ne i64 %gotw, %m
  %bad.wd = or i1 %bad.rwd, %bad.ordw
  %inc.wd = zext i1 %bad.wd to i64
  %vwd.n = add i64 %vwd, %inc.wd
  %m.n = add nuw i64 %m, 1
  %more.wd = icmp ult i64 %m.n, 8
  br i1 %more.wd, label %wdrain, label %verdict

verdict:
  %s0 = add i64 %vf.n, %vd.n
  %s1 = add i64 %s0, %vw.n
  %s2 = add i64 %s1, %vwd.n
  %oktot = add i64 %vfull, %vcnt
  %oktot2 = add i64 %oktot, %vempty
  %oktot3 = add i64 %oktot2, %vcap
  %ok.viol = icmp ne i64 %oktot3, 4
  %cnt.viol = icmp ne i64 %s2, 0
  %anybad = or i1 %ok.viol, %cnt.viol
  %pass = xor i1 %anybad, true
  call void @ut_check(i1 %pass, ptr @m.spmc.single)
  call void @universe_conc_spmc_destroy(ptr %q)
  ret void
}

; ===========================================================================
; null-argument guards
; ===========================================================================
define internal void @test_null() {
entry:
  %q = call ptr @universe_conc_mpmc_create(i64 8, i64 8)
  %v = alloca i64, align 8
  %e1 = call i32 @universe_conc_mpmc_enqueue(ptr null, ptr %v)
  %e2 = call i32 @universe_conc_mpmc_enqueue(ptr %q, ptr null)
  %e3 = call i32 @universe_conc_mpmc_dequeue(ptr null, ptr %v)
  %e4 = call i32 @universe_conc_mpmc_dequeue(ptr %q, ptr null)
  %e5 = call i32 @universe_conc_spmc_enqueue(ptr null, ptr %v)
  %e6 = call i32 @universe_conc_spmc_dequeue(ptr %q, ptr null)
  %o1 = icmp eq i32 %e1, 1
  %o2 = icmp eq i32 %e2, 1
  %o3 = icmp eq i32 %e3, 1
  %o4 = icmp eq i32 %e4, 1
  %o5 = icmp eq i32 %e5, 1
  %o6 = icmp eq i32 %e6, 1
  %a1 = and i1 %o1, %o2
  %a2 = and i1 %a1, %o3
  %a3 = and i1 %a2, %o4
  %a4 = and i1 %a3, %o5
  %a5 = and i1 %a4, %o6
  call void @ut_check(i1 %a5, ptr @m.null)
  call void @universe_conc_mpmc_destroy(ptr %q)
  ret void
}

; ===========================================================================
; stress thread bodies
; ===========================================================================
define internal ptr @producer_mpmc(ptr %arg) {
entry:
  %id = ptrtoint ptr %arg to i64
  %q = load ptr, ptr @g.q, align 8
  %nper = load i64, ptr @g.nper, align 8
  %vp = alloca i64, align 8
  %idsh = shl i64 %id, 32
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  %done = icmp uge i64 %i, %nper
  br i1 %done, label %fin, label %body

body:
  %val = or i64 %idsh, %i
  store i64 %val, ptr %vp, align 8
  br label %try

try:
  %rc = call i32 @universe_conc_mpmc_enqueue(ptr %q, ptr %vp)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %pushed, label %full

full:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  %pd = atomicrmw add ptr @g.pdone, i64 1 monotonic, align 8
  ret ptr null
}

define internal ptr @producer_spmc(ptr %arg) {
entry:
  %id = ptrtoint ptr %arg to i64
  %q = load ptr, ptr @g.q, align 8
  %nper = load i64, ptr @g.nper, align 8
  %vp = alloca i64, align 8
  %idsh = shl i64 %id, 32
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pushed ]
  %done = icmp uge i64 %i, %nper
  br i1 %done, label %fin, label %body

body:
  %val = or i64 %idsh, %i
  store i64 %val, ptr %vp, align 8
  br label %try

try:
  %rc = call i32 @universe_conc_spmc_enqueue(ptr %q, ptr %vp)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %pushed, label %full

full:
  %y = call i32 @sched_yield()
  br label %try

pushed:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  %pd = atomicrmw add ptr @g.pdone, i64 1 monotonic, align 8
  ret ptr null
}

; consumer: drain, record each item's bit, check per-producer order.
define internal ptr @consumer_mpmc(ptr %arg) {
entry:
  %q = load ptr, ptr @g.q, align 8
  %nper = load i64, ptr @g.nper, align 8
  %nprod = load i64, ptr @g.nprod, align 8
  %bm = load ptr, ptr @g.bitmap, align 8
  %valp = alloca i64, align 8
  %last = alloca [4 x i64], align 8
  %l0 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 0
  store i64 -1, ptr %l0, align 8
  %l1 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 1
  store i64 -1, ptr %l1, align 8
  %l2 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 2
  store i64 -1, ptr %l2, align 8
  %l3 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 3
  store i64 -1, ptr %l3, align 8
  br label %loop

loop:
  %edone = phi i64 [ 0, %entry ], [ %edone.n, %emptyc ], [ 0, %proc ]
  %c = load atomic i64, ptr @g.consumed monotonic, align 8
  %reached = icmp uge i64 %c, 400000
  br i1 %reached, label %exit, label %work

work:
  %rc = call i32 @universe_conc_mpmc_dequeue(ptr %q, ptr %valp)
  %got = icmp eq i32 %rc, 0
  br i1 %got, label %proc, label %mt

mt:
  %d = load atomic i64, ptr @g.pdone monotonic, align 8
  %alldone = icmp eq i64 %d, %nprod
  %ainc = zext i1 %alldone to i64
  %edone.n = add i64 %edone, %ainc
  %guard = icmp ugt i64 %edone.n, 10000000
  br i1 %guard, label %exit, label %emptyc

emptyc:
  %y = call i32 @sched_yield()
  br label %loop

proc:
  %cadd = atomicrmw add ptr @g.consumed, i64 1 monotonic, align 8
  %val = load i64, ptr %valp, align 8
  %p = lshr i64 %val, 32
  %seq = and i64 %val, 4294967295
  ; per-producer monotone order (this consumer's view)
  %lp = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 %p
  %prev = load i64, ptr %lp, align 8
  %ord.bad = icmp sle i64 %seq, %prev
  %ord.inc = zext i1 %ord.bad to i64
  %ov = atomicrmw add ptr @g.order, i64 %ord.inc monotonic, align 8
  store i64 %seq, ptr %lp, align 8
  ; bitmap: gid = p*nper + seq
  %pn = mul i64 %p, %nper
  %gid = add i64 %pn, %seq
  %word = lshr i64 %gid, 6
  %bit = and i64 %gid, 63
  %mask = shl i64 1, %bit
  %wp = getelementptr inbounds i64, ptr %bm, i64 %word
  %old = atomicrmw or ptr %wp, i64 %mask monotonic, align 8
  %dupbit = and i64 %old, %mask
  %isdup = icmp ne i64 %dupbit, 0
  %dup.inc = zext i1 %isdup to i64
  %dv = atomicrmw add ptr @g.dup, i64 %dup.inc monotonic, align 8
  br label %loop

exit:
  ret ptr null
}

define internal ptr @consumer_spmc(ptr %arg) {
entry:
  %q = load ptr, ptr @g.q, align 8
  %nper = load i64, ptr @g.nper, align 8
  %nprod = load i64, ptr @g.nprod, align 8
  %bm = load ptr, ptr @g.bitmap, align 8
  %valp = alloca i64, align 8
  %last = alloca [4 x i64], align 8
  %l0 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 0
  store i64 -1, ptr %l0, align 8
  %l1 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 1
  store i64 -1, ptr %l1, align 8
  %l2 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 2
  store i64 -1, ptr %l2, align 8
  %l3 = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 3
  store i64 -1, ptr %l3, align 8
  br label %loop

loop:
  %edone = phi i64 [ 0, %entry ], [ %edone.n, %emptyc ], [ 0, %proc ]
  %c = load atomic i64, ptr @g.consumed monotonic, align 8
  %reached = icmp uge i64 %c, 400000
  br i1 %reached, label %exit, label %work

work:
  %rc = call i32 @universe_conc_spmc_dequeue(ptr %q, ptr %valp)
  %got = icmp eq i32 %rc, 0
  br i1 %got, label %proc, label %mt

mt:
  %d = load atomic i64, ptr @g.pdone monotonic, align 8
  %alldone = icmp eq i64 %d, %nprod
  %ainc = zext i1 %alldone to i64
  %edone.n = add i64 %edone, %ainc
  %guard = icmp ugt i64 %edone.n, 10000000
  br i1 %guard, label %exit, label %emptyc

emptyc:
  %y = call i32 @sched_yield()
  br label %loop

proc:
  %cadd = atomicrmw add ptr @g.consumed, i64 1 monotonic, align 8
  %val = load i64, ptr %valp, align 8
  %p = lshr i64 %val, 32
  %seq = and i64 %val, 4294967295
  %lp = getelementptr inbounds [4 x i64], ptr %last, i64 0, i64 %p
  %prev = load i64, ptr %lp, align 8
  %ord.bad = icmp sle i64 %seq, %prev
  %ord.inc = zext i1 %ord.bad to i64
  %ov = atomicrmw add ptr @g.order, i64 %ord.inc monotonic, align 8
  store i64 %seq, ptr %lp, align 8
  %pn = mul i64 %p, %nper
  %gid = add i64 %pn, %seq
  %word = lshr i64 %gid, 6
  %bit = and i64 %gid, 63
  %mask = shl i64 1, %bit
  %wp = getelementptr inbounds i64, ptr %bm, i64 %word
  %old = atomicrmw or ptr %wp, i64 %mask monotonic, align 8
  %dupbit = and i64 %old, %mask
  %isdup = icmp ne i64 %dupbit, 0
  %dup.inc = zext i1 %isdup to i64
  %dv = atomicrmw add ptr @g.dup, i64 %dup.inc monotonic, align 8
  br label %loop

exit:
  ret ptr null
}

; ---- popcount over the bitmap ---------------------------------------------
define internal i64 @popcount_bm(ptr %bm, i64 %words) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %wp = getelementptr inbounds i64, ptr %bm, i64 %i
  %w = load i64, ptr %wp, align 8
  %pc = call i64 @llvm.ctpop.i64(i64 %w)
  %acc.n = add i64 %acc, %pc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %words
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n
}

; ---- spawn 4 producers + 4 consumers (MPMC), join all ----------------------
define internal void @spawn_join_mpmc() {
entry:
  %tids = alloca [8 x i64], align 8
  br label %prod

prod:
  %pk = phi i64 [ 0, %entry ], [ %pk.n, %prod ]
  %pslot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %pk
  %parg = inttoptr i64 %pk to ptr
  %pc = call i32 @pthread_create(ptr %pslot, ptr null, ptr @producer_mpmc, ptr %parg)
  %pk.n = add nuw i64 %pk, 1
  %pmore = icmp ult i64 %pk.n, 4
  br i1 %pmore, label %prod, label %cons

cons:
  %ck = phi i64 [ 0, %prod ], [ %ck.n, %cons ]
  %cidx = add i64 %ck, 4
  %cslot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %cidx
  %cc = call i32 @pthread_create(ptr %cslot, ptr null, ptr @consumer_mpmc, ptr null)
  %ck.n = add nuw i64 %ck, 1
  %cmore = icmp ult i64 %ck.n, 4
  br i1 %cmore, label %cons, label %join

join:
  %jk = phi i64 [ 0, %cons ], [ %jk.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %jk
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %jk.n = add nuw i64 %jk, 1
  %jmore = icmp ult i64 %jk.n, 8
  br i1 %jmore, label %join, label %done

done:
  ret void
}

; ---- spawn 1 producer + 4 consumers (SPMC), join all -----------------------
define internal void @spawn_join_spmc() {
entry:
  %tids = alloca [5 x i64], align 8
  %pslot = getelementptr inbounds [5 x i64], ptr %tids, i64 0, i64 0
  %pc = call i32 @pthread_create(ptr %pslot, ptr null, ptr @producer_spmc, ptr null)
  br label %cons

cons:
  %ck = phi i64 [ 0, %entry ], [ %ck.n, %cons ]
  %cidx = add i64 %ck, 1
  %cslot = getelementptr inbounds [5 x i64], ptr %tids, i64 0, i64 %cidx
  %cc = call i32 @pthread_create(ptr %cslot, ptr null, ptr @consumer_spmc, ptr null)
  %ck.n = add nuw i64 %ck, 1
  %cmore = icmp ult i64 %ck.n, 4
  br i1 %cmore, label %cons, label %join

join:
  %jk = phi i64 [ 0, %cons ], [ %jk.n, %join ]
  %jslot = getelementptr inbounds [5 x i64], ptr %tids, i64 0, i64 %jk
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %jk.n = add nuw i64 %jk, 1
  %jmore = icmp ult i64 %jk.n, 5
  br i1 %jmore, label %join, label %done

done:
  ret void
}

; ---- run 10 MPMC stress rounds; return total violations --------------------
define internal i64 @run_mpmc() {
entry:
  %q = call ptr @universe_conc_mpmc_create(i64 1024, i64 8)
  store ptr %q, ptr @g.q, align 8
  store i64 100000, ptr @g.nper, align 8
  store i64 4, ptr @g.nprod, align 8
  %bm = load ptr, ptr @g.bitmap, align 8
  br label %round

round:
  %r = phi i64 [ 0, %entry ], [ %r.n, %round ]
  %tv = phi i64 [ 0, %entry ], [ %tv.n, %round ]
  call void @llvm.memset.p0.i64(ptr %bm, i8 0, i64 50000, i1 false)
  store i64 0, ptr @g.consumed, align 8
  store i64 0, ptr @g.dup, align 8
  store i64 0, ptr @g.order, align 8
  store i64 0, ptr @g.pdone, align 8
  call void @spawn_join_mpmc()
  %distinct = call i64 @popcount_bm(ptr %bm, i64 6250)
  %consumed = load i64, ptr @g.consumed, align 8
  %dup = load i64, ptr @g.dup, align 8
  %order = load i64, ptr @g.order, align 8
  %cnt = call i64 @universe_conc_mpmc_count(ptr %q)
  %v.dist = icmp ne i64 %distinct, 400000
  %v.cons = icmp ne i64 %consumed, 400000
  %v.cnt = icmp ne i64 %cnt, 0
  %iv0 = zext i1 %v.dist to i64
  %iv1 = zext i1 %v.cons to i64
  %iv2 = zext i1 %v.cnt to i64
  %rv0 = add i64 %dup, %order
  %rv1 = add i64 %rv0, %iv0
  %rv2 = add i64 %rv1, %iv1
  %rv3 = add i64 %rv2, %iv2
  %tv.n = add i64 %tv, %rv3
  %r.n = add nuw i64 %r, 1
  %more = icmp ult i64 %r.n, 10
  br i1 %more, label %round, label %done

done:
  call void @universe_conc_mpmc_destroy(ptr %q)
  ret i64 %tv.n
}

; ---- run 10 SPMC stress rounds; return total violations --------------------
define internal i64 @run_spmc() {
entry:
  %q = call ptr @universe_conc_spmc_create(i64 1024, i64 8)
  store ptr %q, ptr @g.q, align 8
  store i64 400000, ptr @g.nper, align 8
  store i64 1, ptr @g.nprod, align 8
  %bm = load ptr, ptr @g.bitmap, align 8
  br label %round

round:
  %r = phi i64 [ 0, %entry ], [ %r.n, %round ]
  %tv = phi i64 [ 0, %entry ], [ %tv.n, %round ]
  call void @llvm.memset.p0.i64(ptr %bm, i8 0, i64 50000, i1 false)
  store i64 0, ptr @g.consumed, align 8
  store i64 0, ptr @g.dup, align 8
  store i64 0, ptr @g.order, align 8
  store i64 0, ptr @g.pdone, align 8
  call void @spawn_join_spmc()
  %distinct = call i64 @popcount_bm(ptr %bm, i64 6250)
  %consumed = load i64, ptr @g.consumed, align 8
  %dup = load i64, ptr @g.dup, align 8
  %order = load i64, ptr @g.order, align 8
  %cnt = call i64 @universe_conc_spmc_count(ptr %q)
  %v.dist = icmp ne i64 %distinct, 400000
  %v.cons = icmp ne i64 %consumed, 400000
  %v.cnt = icmp ne i64 %cnt, 0
  %iv0 = zext i1 %v.dist to i64
  %iv1 = zext i1 %v.cons to i64
  %iv2 = zext i1 %v.cnt to i64
  %rv0 = add i64 %dup, %order
  %rv1 = add i64 %rv0, %iv0
  %rv2 = add i64 %rv1, %iv1
  %rv3 = add i64 %rv2, %iv2
  %tv.n = add i64 %tv, %rv3
  %r.n = add nuw i64 %r, 1
  %more = icmp ult i64 %r.n, 10
  br i1 %more, label %round, label %done

done:
  call void @universe_conc_spmc_destroy(ptr %q)
  ret i64 %tv.n
}

; ---- MPMC 4x4 throughput bench --------------------------------------------
; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) of the
; 4-producer x 4-consumer parallel batch. 17 reps: rep 0 warm-up (discarded),
; reps 1..16 recorded. Each rep recreates the queue and resets accumulators.
; batch = 4*100k enqueue + 4*100k dequeue; ops_per_rep = 800k.
define internal void @bench() {
entry:
  %bm = load ptr, ptr @g.bitmap, align 8
  br label %rep

rep:
  %r = phi i64 [ 0, %entry ], [ %r.n, %rep.next ]
  %q = call ptr @universe_conc_mpmc_create(i64 1024, i64 8)
  store ptr %q, ptr @g.q, align 8
  store i64 100000, ptr @g.nper, align 8
  store i64 4, ptr @g.nprod, align 8
  call void @llvm.memset.p0.i64(ptr %bm, i8 0, i64 50000, i1 false)
  store i64 0, ptr @g.consumed, align 8
  store i64 0, ptr @g.dup, align 8
  store i64 0, ptr @g.order, align 8
  store i64 0, ptr @g.pdone, align 8
  %t0 = call double @ut_now_sec()
  call void @spawn_join_mpmc()
  %t1 = call double @ut_now_sec()
  call void @universe_conc_mpmc_destroy(ptr %q)
  %dt = fsub double %t1, %t0
  %warm = icmp eq i64 %r, 0
  br i1 %warm, label %rep.next, label %rep.store

rep.store:
  %idx = sub i64 %r, 1
  %sp = getelementptr inbounds [16 x double], ptr @mpmc.samp, i64 0, i64 %idx
  store double %dt, ptr %sp, align 8
  br label %rep.next

rep.next:
  %r.n = add nuw nsw i64 %r, 1
  %rmore = icmp ult i64 %r.n, 17
  br i1 %rmore, label %rep, label %done

done:
  call void @ut_report_dist(ptr @mpmc.samp, i64 16, i64 800000, ptr @lbl.mpmc)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %bm = call ptr @malloc(i64 50000)
  store ptr %bm, ptr @g.bitmap, align 8
  call void @test_single_mpmc()
  call void @test_single_spmc()
  call void @test_null()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %dobench, label %dostress

dobench:
  call void @bench()
  br label %fin

dostress:
  %vm = call i64 @run_mpmc()
  call void @ut_check_eq(i64 %vm, i64 0, ptr @m.mpmc.stress)
  %vs = call i64 @run_spmc()
  call void @ut_check_eq(i64 %vs, i64 0, ptr @m.spmc.stress)
  br label %fin

fin:
  %bm2 = load ptr, ptr @g.bitmap, align 8
  call void @free(ptr %bm2)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
