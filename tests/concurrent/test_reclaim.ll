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

; Tests for universe_conc_ebr_* / universe_conc_hp_* / universe_conc_seqlock_*.
;   * The GATE is a real use-after-free stress on a Treiber stack: mutators
;     push fresh nodes and pop+RETIRE nodes; readers concurrently DEREFERENCE
;     the (possibly just-popped) nodes under a reclamation guard. Every node
;     carries a canary set to MAGIC on alloc and overwritten to POISON just
;     before its memory is freed. A guarded reader that observes anything but
;     MAGIC would be reading a freed node -> counted as a violation (and,
;     under ASan, a hard read-after-free error). EBR readers do the HARD case
;     (deep whole-stack traversal, safe because the pin protects everything
;     reachable); HP readers use the sound single-node protect+validate.
;     4 mutators x 4 readers x 100k ops, 10 rounds, at -O0 and -O3 and TSAN.
;     Invariants: zero canary violations; zero leaks (allocated == freed after
;     quiescence).
;   * Single-thread sanity: EBR enter/exit/retire/advance bag accounting;
;     HP protect/clear/scan-frees-only-unprotected; SeqLock snapshot.
;   * --bench: EBR reader cost vs HP reader cost; retire+reclaim rate.

; ---- reclaim toolkit under test -------------------------------------------
declare void @universe_conc_ebr_register(i64)
declare void @universe_conc_ebr_enter(i64)
declare void @universe_conc_ebr_exit(i64)
declare void @universe_conc_ebr_retire(i64, ptr, ptr)
declare i32  @universe_conc_ebr_try_advance(i64)
declare i64  @universe_conc_ebr_epoch()
declare void @universe_conc_ebr_collect_all()
declare void @universe_conc_ebr_reset()
declare void @universe_conc_hp_protect(i64, i64, ptr)
declare void @universe_conc_hp_clear(i64, i64)
declare void @universe_conc_hp_retire(i64, ptr, ptr)
declare void @universe_conc_hp_collect(i64)
declare void @universe_conc_hp_reset()
declare void @universe_conc_seqlock_write_begin(ptr)
declare void @universe_conc_seqlock_write_end(ptr)
declare i64  @universe_conc_seqlock_read_begin(ptr)
declare i1   @universe_conc_seqlock_read_retry(ptr, i64)

; ---- libc + harness --------------------------------------------------------
declare ptr  @malloc(i64)
declare void @free(ptr)
declare i32  @pthread_create(ptr, ptr, ptr, ptr)
declare i32  @pthread_join(i64, ptr)
declare i32  @printf(ptr, ...)

declare void   @ut_check(i1, ptr)
declare void   @ut_check_eq(i64, i64, ptr)
declare i32    @ut_summary()
declare i1     @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void   @ut_report_dist(ptr, i64, i64, ptr)

; ---- bench distribution samples (warm-up + 16 reps) ----
@ebr.samp = internal global [16 x double] zeroinitializer, align 8
@hp.samp  = internal global [16 x double] zeroinitializer, align 8
@rc.samp  = internal global [16 x double] zeroinitializer, align 8
@lbl.ebr  = private unnamed_addr constant [16 x i8] c"ebr reader read\00"
@lbl.hp   = private unnamed_addr constant [18 x i8] c"hp reader protect\00"
@lbl.rc   = private unnamed_addr constant [20 x i8] c"ebr retire+reclaim \00"

; ---- shared state ----------------------------------------------------------
@g.head       = internal global ptr null, align 8   ; Treiber stack top (atomic)
@g.allocated  = internal global i64 0, align 8      ; nodes malloc'd (atomic)
@g.freed      = internal global i64 0, align 8      ; nodes freed  (atomic)
@g.canary_viol= internal global i64 0, align 8      ; UAF observations (atomic)
@g.ops        = internal global i64 0, align 8      ; ops per worker thread
@g.sink       = internal global i64 0, align 8      ; bench DCE sink (volatile)

; node layout: +0 next(ptr)  +8 canary(i64)  +16 value(i64) ; malloc 32
; MAGIC = 1234567890123456789 ; POISON = -1

@m.ebr.stress  = private unnamed_addr constant [26 x i8] c"ebr UAF stress: no violat\00"
@m.hp.stress   = private unnamed_addr constant [26 x i8] c"hp  UAF stress: no violat\00"
@m.ebr.san     = private unnamed_addr constant [22 x i8] c"ebr bag accounting ok\00"
@m.ebr.san2    = private unnamed_addr constant [24 x i8] c"ebr advance frees on +2\00"
@m.hp.san      = private unnamed_addr constant [27 x i8] c"hp scan frees unprotected \00"
@m.hp.san2     = private unnamed_addr constant [25 x i8] c"hp clear then free rest \00"
@m.sl.san      = private unnamed_addr constant [20 x i8] c"seqlock snapshot ok\00"
@m.dbg.fmt     = private unnamed_addr constant [47 x i8] c"  [dbg] alloc=%lld freed=%lld canaryviol=%lld\0A\00"

; ===========================================================================
; node helpers
; ===========================================================================
define internal ptr @node_alloc() {
entry:
  %n = call ptr @malloc(i64 32)
  %cp = getelementptr inbounds nuw i8, ptr %n, i64 8
  store i64 1234567890123456789, ptr %cp, align 8
  %a = atomicrmw add ptr @g.allocated, i64 1 monotonic, align 8
  ret ptr %n
}

; freefn: poison the canary, count, and release the memory.
define internal void @node_free(ptr %n) {
entry:
  %cp = getelementptr inbounds nuw i8, ptr %n, i64 8
  store i64 -1, ptr %cp, align 8
  %f = atomicrmw add ptr @g.freed, i64 1 monotonic, align 8
  call void @free(ptr %n)
  ret void
}

; ===========================================================================
; Treiber stack
; ===========================================================================
define internal void @stack_push(ptr %node) {
entry:
  br label %loop

loop:
  %old = load atomic ptr, ptr @g.head acquire, align 8
  store ptr %old, ptr %node, align 8                  ; node->next = old
  %cx = cmpxchg ptr @g.head, ptr %old, ptr %node release monotonic
  %ok = extractvalue { ptr, i1 } %cx, 1
  br i1 %ok, label %done, label %loop

done:
  ret void
}

; pop under an already-held EBR pin: dereference old->next is safe.
define internal ptr @stack_pop_ebr() {
entry:
  br label %loop

loop:
  %old = load atomic ptr, ptr @g.head acquire, align 8
  %isnull = icmp eq ptr %old, null
  br i1 %isnull, label %retnull, label %body

body:
  %next = load ptr, ptr %old, align 8                 ; old->next  (safe: pinned)
  %cx = cmpxchg ptr @g.head, ptr %old, ptr %next acq_rel acquire
  %ok = extractvalue { ptr, i1 } %cx, 1
  br i1 %ok, label %got, label %loop

got:
  ret ptr %old

retnull:
  ret ptr null
}

; pop under hazard pointers: protect old, validate it is still head, then deref.
define internal ptr @stack_pop_hp(i64 %tid) {
entry:
  br label %loop

loop:
  %old = load atomic ptr, ptr @g.head acquire, align 8
  %isnull = icmp eq ptr %old, null
  br i1 %isnull, label %retnull, label %prot

prot:
  call void @universe_conc_hp_protect(i64 %tid, i64 0, ptr %old)
  %old2 = load atomic ptr, ptr @g.head acquire, align 8   ; re-validate
  %stale = icmp ne ptr %old2, %old
  br i1 %stale, label %loop, label %deref

deref:
  %next = load ptr, ptr %old, align 8                 ; old->next (safe: protected)
  %cx = cmpxchg ptr @g.head, ptr %old, ptr %next acq_rel acquire
  %ok = extractvalue { ptr, i1 } %cx, 1
  br i1 %ok, label %got, label %loop

got:
  call void @universe_conc_hp_clear(i64 %tid, i64 0)
  ret ptr %old

retnull:
  call void @universe_conc_hp_clear(i64 %tid, i64 0)   ; drop any stale protect
  ret ptr null
}

; deep EBR traversal: dereference every reachable node's canary under the pin.
define internal void @reader_scan_ebr(i64 %tid) {
entry:
  call void @universe_conc_ebr_enter(i64 %tid)
  %p0 = load atomic ptr, ptr @g.head acquire, align 8
  br label %loop

loop:
  %p = phi ptr [ %p0, %entry ], [ %next, %cont ]
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %fin, label %chk

chk:
  %cp = getelementptr inbounds nuw i8, ptr %p, i64 8
  %c = load i64, ptr %cp, align 8
  %bad = icmp ne i64 %c, 1234567890123456789
  br i1 %bad, label %viol, label %cont, !prof !0

viol:
  %v = atomicrmw add ptr @g.canary_viol, i64 1 monotonic, align 8
  br label %cont

cont:
  %next = load ptr, ptr %p, align 8                   ; p->next (frozen chain)
  br label %loop

fin:
  call void @universe_conc_ebr_exit(i64 %tid)
  ret void
}

; ===========================================================================
; worker thread bodies
; ===========================================================================
define internal ptr @mutator_ebr(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  call void @universe_conc_ebr_register(i64 %tid)
  %ops = load i64, ptr @g.ops, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %iter ]
  %done = icmp uge i64 %i, %ops
  br i1 %done, label %fin, label %body

body:
  call void @universe_conc_ebr_enter(i64 %tid)
  %n = call ptr @node_alloc()
  call void @stack_push(ptr %n)
  %popped = call ptr @stack_pop_ebr()
  %pnull = icmp eq ptr %popped, null
  br i1 %pnull, label %afterpop, label %doretire

doretire:
  call void @universe_conc_ebr_retire(i64 %tid, ptr %popped, ptr @node_free)
  br label %afterpop

afterpop:
  call void @universe_conc_ebr_exit(i64 %tid)
  %m = and i64 %i, 63
  %adv = icmp eq i64 %m, 0
  br i1 %adv, label %doadv, label %iter

doadv:
  %r = call i32 @universe_conc_ebr_try_advance(i64 %tid)
  br label %iter

iter:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  ret ptr null
}

define internal ptr @reader_ebr(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  call void @universe_conc_ebr_register(i64 %tid)
  %ops = load i64, ptr @g.ops, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %iter ]
  %done = icmp uge i64 %i, %ops
  br i1 %done, label %fin, label %body

body:
  call void @reader_scan_ebr(i64 %tid)
  br label %iter

iter:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  ret ptr null
}

define internal ptr @mutator_hp(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %ops = load i64, ptr @g.ops, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %iter ]
  %done = icmp uge i64 %i, %ops
  br i1 %done, label %fin, label %body

body:
  %n = call ptr @node_alloc()
  call void @stack_push(ptr %n)
  %popped = call ptr @stack_pop_hp(i64 %tid)
  %pnull = icmp eq ptr %popped, null
  br i1 %pnull, label %iter, label %doretire

doretire:
  call void @universe_conc_hp_retire(i64 %tid, ptr %popped, ptr @node_free)
  br label %iter

iter:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  ret ptr null
}

define internal ptr @reader_hp(ptr %arg) {
entry:
  %tid = ptrtoint ptr %arg to i64
  %ops = load i64, ptr @g.ops, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %iter ]
  %done = icmp uge i64 %i, %ops
  br i1 %done, label %fin, label %body

body:
  %old = load atomic ptr, ptr @g.head acquire, align 8
  %isnull = icmp eq ptr %old, null
  br i1 %isnull, label %iter, label %prot

prot:
  call void @universe_conc_hp_protect(i64 %tid, i64 0, ptr %old)
  %old2 = load atomic ptr, ptr @g.head acquire, align 8
  %eq = icmp eq ptr %old2, %old
  br i1 %eq, label %readc, label %clr

readc:
  %cp = getelementptr inbounds nuw i8, ptr %old, i64 8
  %c = load i64, ptr %cp, align 8
  %bad = icmp ne i64 %c, 1234567890123456789
  br i1 %bad, label %viol, label %clr, !prof !0

viol:
  %v = atomicrmw add ptr @g.canary_viol, i64 1 monotonic, align 8
  br label %clr

clr:
  call void @universe_conc_hp_clear(i64 %tid, i64 0)
  br label %iter

iter:
  %i.n = add nuw i64 %i, 1
  br label %loop

fin:
  ret ptr null
}

; ===========================================================================
; round setup / teardown
; ===========================================================================
define internal void @stack_init(i64 %count) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %n = call ptr @node_alloc()
  call void @stack_push(ptr %n)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %done

done:
  ret void
}

; drain remaining stack (single-threaded now) and reclaim everything (EBR).
define internal void @teardown_drain_ebr() {
entry:
  call void @universe_conc_ebr_register(i64 0)
  call void @universe_conc_ebr_enter(i64 0)
  br label %loop

loop:
  %old = load ptr, ptr @g.head, align 8
  %isnull = icmp eq ptr %old, null
  br i1 %isnull, label %flush, label %body

body:
  %next = load ptr, ptr %old, align 8
  store ptr %next, ptr @g.head, align 8
  call void @universe_conc_ebr_retire(i64 0, ptr %old, ptr @node_free)
  br label %loop

flush:
  call void @universe_conc_ebr_exit(i64 0)
  ; quiescent now: force-free every pending node in every bag.
  call void @universe_conc_ebr_collect_all()
  br label %done

done:
  ret void
}

; drain remaining stack and reclaim everything (HP): retire the rest, then
; collect every thread's retire list (no hazards live -> frees all).
define internal void @teardown_drain_hp() {
entry:
  br label %loop

loop:
  %old = load ptr, ptr @g.head, align 8
  %isnull = icmp eq ptr %old, null
  br i1 %isnull, label %collect, label %body

body:
  %next = load ptr, ptr %old, align 8
  store ptr %next, ptr @g.head, align 8
  call void @universe_conc_hp_retire(i64 0, ptr %old, ptr @node_free)
  br label %loop

collect:
  %t = phi i64 [ 0, %loop ], [ %t.n, %collect ]
  call void @universe_conc_hp_collect(i64 %t)
  %t.n = add nuw i64 %t, 1
  %more = icmp ult i64 %t.n, 8
  br i1 %more, label %collect, label %done

done:
  ret void
}

; one stress round (EBR): 4 mutators + 4 readers; returns total violations
; (canary observations + 1 if allocated != freed after reclamation).
define internal i64 @run_round_ebr() {
entry:
  store ptr null, ptr @g.head, align 8
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  store i64 0, ptr @g.canary_viol, align 8
  call void @stack_init(i64 64)
  %tids = alloca [8 x i64], align 8
  br label %spawn

spawn:
  %k = phi i64 [ 0, %entry ], [ %k.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %k
  %arg = inttoptr i64 %k to ptr
  %ismut = icmp ult i64 %k, 4
  %fn = select i1 %ismut, ptr @mutator_ebr, ptr @reader_ebr
  %pc = call i32 @pthread_create(ptr %slot, ptr null, ptr %fn, ptr %arg)
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 8
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 8
  br i1 %jmore, label %join, label %teardown

teardown:
  call void @teardown_drain_ebr()
  %a = load i64, ptr @g.allocated, align 8
  %f = load i64, ptr @g.freed, align 8
  %v = load i64, ptr @g.canary_viol, align 8
  %leak = icmp ne i64 %a, %f
  %leak.i = zext i1 %leak to i64
  %tot = add i64 %v, %leak.i
  %bad = icmp ne i64 %tot, 0
  br i1 %bad, label %dbg, label %ok

dbg:
  %pr = call i32 (ptr, ...) @printf(ptr @m.dbg.fmt, i64 %a, i64 %f, i64 %v)
  br label %ok

ok:
  ret i64 %tot
}

; one stress round (HP): 4 mutators + 4 readers.
define internal i64 @run_round_hp() {
entry:
  store ptr null, ptr @g.head, align 8
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  store i64 0, ptr @g.canary_viol, align 8
  call void @stack_init(i64 64)
  %tids = alloca [8 x i64], align 8
  br label %spawn

spawn:
  %k = phi i64 [ 0, %entry ], [ %k.n, %spawn ]
  %slot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %k
  %arg = inttoptr i64 %k to ptr
  %ismut = icmp ult i64 %k, 4
  %fn = select i1 %ismut, ptr @mutator_hp, ptr @reader_hp
  %pc = call i32 @pthread_create(ptr %slot, ptr null, ptr %fn, ptr %arg)
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, 8
  br i1 %more, label %spawn, label %join

join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jslot = getelementptr inbounds [8 x i64], ptr %tids, i64 0, i64 %j
  %tid = load i64, ptr %jslot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 8
  br i1 %jmore, label %join, label %teardown

teardown:
  call void @teardown_drain_hp()
  %a = load i64, ptr @g.allocated, align 8
  %f = load i64, ptr @g.freed, align 8
  %v = load i64, ptr @g.canary_viol, align 8
  %leak = icmp ne i64 %a, %f
  %leak.i = zext i1 %leak to i64
  %tot = add i64 %v, %leak.i
  ret i64 %tot
}

; ===========================================================================
; single-thread sanity
; ===========================================================================
define internal void @test_ebr_sanity() {
entry:
  call void @universe_conc_ebr_reset()
  call void @universe_conc_ebr_register(i64 0)
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  ; epoch 0 after reset
  %e0 = call i64 @universe_conc_ebr_epoch()
  %e0.ok = icmp eq i64 %e0, 0
  ; retire a node at epoch 0 -> bag[0], tagged epoch 0
  call void @universe_conc_ebr_enter(i64 0)
  %n0 = call ptr @node_alloc()
  call void @universe_conc_ebr_retire(i64 0, ptr %n0, ptr @node_free)
  call void @universe_conc_ebr_exit(i64 0)
  ; nothing freed yet (reclamation is owner-lazy on slot reuse)
  %f0 = load i64, ptr @g.freed, align 8
  %f0.ok = icmp eq i64 %f0, 0
  ; advance the epoch to 3 (quiescent -> each succeeds)
  %r1 = call i32 @universe_conc_ebr_try_advance(i64 0)
  %r2 = call i32 @universe_conc_ebr_try_advance(i64 0)
  %r3 = call i32 @universe_conc_ebr_try_advance(i64 0)
  %e3 = call i64 @universe_conc_ebr_epoch()
  %e3.ok = icmp eq i64 %e3, 3
  %r1.ok = icmp eq i32 %r1, 1
  %r2.ok = icmp eq i32 %r2, 1
  %r3.ok = icmp eq i32 %r3, 1
  %ra = and i1 %r1.ok, %r2.ok
  %rb = and i1 %ra, %r3.ok
  %s0 = and i1 %e0.ok, %f0.ok
  %s1 = and i1 %s0, %rb
  %s2 = and i1 %s1, %e3.ok
  call void @ut_check(i1 %s2, ptr @m.ebr.san)
  ; retire at epoch 3 -> slot 0 recycles n0 (epoch 0, safe) -> freed becomes 1
  call void @universe_conc_ebr_enter(i64 0)
  %n1 = call ptr @node_alloc()
  call void @universe_conc_ebr_retire(i64 0, ptr %n1, ptr @node_free)
  call void @universe_conc_ebr_exit(i64 0)
  %f1 = load i64, ptr @g.freed, align 8
  %f1.ok = icmp eq i64 %f1, 1
  ; collect the rest (n1); allocated == freed == 2
  call void @universe_conc_ebr_collect_all()
  %f2 = load i64, ptr @g.freed, align 8
  %a2 = load i64, ptr @g.allocated, align 8
  %f2.ok = icmp eq i64 %f2, 2
  %a2.ok = icmp eq i64 %a2, 2
  %t0 = and i1 %f1.ok, %f2.ok
  %t1 = and i1 %t0, %a2.ok
  call void @ut_check(i1 %t1, ptr @m.ebr.san2)
  ret void
}

define internal void @test_hp_sanity() {
entry:
  call void @universe_conc_hp_reset()
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  %n1 = call ptr @node_alloc()
  %n2 = call ptr @node_alloc()
  ; protect n1, retire both
  call void @universe_conc_hp_protect(i64 0, i64 0, ptr %n1)
  call void @universe_conc_hp_retire(i64 0, ptr %n1, ptr @node_free)
  call void @universe_conc_hp_retire(i64 0, ptr %n2, ptr @node_free)
  ; force scan: n1 protected (kept), n2 freed
  call void @universe_conc_hp_collect(i64 0)
  %f = load i64, ptr @g.freed, align 8
  %f.ok = icmp eq i64 %f, 1
  call void @ut_check(i1 %f.ok, ptr @m.hp.san)
  ; clear protection, collect again: n1 now freed
  call void @universe_conc_hp_clear(i64 0, i64 0)
  call void @universe_conc_hp_collect(i64 0)
  %f2 = load i64, ptr @g.freed, align 8
  %a = load i64, ptr @g.allocated, align 8
  %f2.ok = icmp eq i64 %f2, 2
  %a.ok = icmp eq i64 %a, 2
  %ok = and i1 %f2.ok, %a.ok
  call void @ut_check(i1 %ok, ptr @m.hp.san2)
  ret void
}

define internal void @test_seqlock_sanity() {
entry:
  %s = alloca i64, align 8
  store i64 0, ptr %s, align 8
  %d = alloca [2 x i64], align 8
  %d0 = getelementptr inbounds [2 x i64], ptr %d, i64 0, i64 0
  %d1 = getelementptr inbounds [2 x i64], ptr %d, i64 0, i64 1
  ; write a consistent pair
  call void @universe_conc_seqlock_write_begin(ptr %s)
  store i64 111, ptr %d0, align 8
  store i64 222, ptr %d1, align 8
  call void @universe_conc_seqlock_write_end(ptr %s)
  br label %rloop

rloop:
  %s1 = call i64 @universe_conc_seqlock_read_begin(ptr %s)
  %v0 = load i64, ptr %d0, align 8
  %v1 = load i64, ptr %d1, align 8
  %retry = call i1 @universe_conc_seqlock_read_retry(ptr %s, i64 %s1)
  br i1 %retry, label %rloop, label %chk

chk:
  %ok0 = icmp eq i64 %v0, 111
  %ok1 = icmp eq i64 %v1, 222
  %ok = and i1 %ok0, %ok1
  call void @ut_check(i1 %ok, ptr @m.sl.san)
  ret void
}

; ===========================================================================
; --bench : EBR reader cost vs HP reader cost vs reclaim rate
; ===========================================================================
; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) for each of
; the three single-threaded reader/reclaim loops. 17 reps: rep 0 is warm-up
; (discarded), reps 1..16 recorded. batch M = 1M ops; ops_per_rep = 1M.
define internal void @bench() {
entry:
  call void @universe_conc_ebr_reset()
  call void @universe_conc_hp_reset()
  store ptr null, ptr @g.head, align 8
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  %n = call ptr @node_alloc()
  call void @stack_push(ptr %n)
  call void @universe_conc_ebr_register(i64 0)
  br label %ebr.rep

; ---- EBR reader ----
ebr.rep:
  %er = phi i64 [ 0, %entry ], [ %er.n, %ebr.rep.next ]
  %et0 = call double @ut_now_sec()
  br label %ebr.loop

ebr.loop:
  %ei = phi i64 [ 0, %ebr.rep ], [ %ei.n, %ebr.loop ]
  %eacc = phi i64 [ 0, %ebr.rep ], [ %eacc.n, %ebr.loop ]
  call void @universe_conc_ebr_enter(i64 0)
  %ep = load atomic ptr, ptr @g.head acquire, align 8
  %ecp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  %ec = load i64, ptr %ecp, align 8
  call void @universe_conc_ebr_exit(i64 0)
  %eacc.n = xor i64 %eacc, %ec
  %ei.n = add nuw i64 %ei, 1
  %emore = icmp ult i64 %ei.n, 1000000
  br i1 %emore, label %ebr.loop, label %ebr.end

ebr.end:
  store volatile i64 %eacc.n, ptr @g.sink, align 8
  %et1 = call double @ut_now_sec()
  %edt = fsub double %et1, %et0
  %ewarm = icmp eq i64 %er, 0
  br i1 %ewarm, label %ebr.rep.next, label %ebr.rep.store

ebr.rep.store:
  %eidx = sub i64 %er, 1
  %esp = getelementptr inbounds [16 x double], ptr @ebr.samp, i64 0, i64 %eidx
  store double %edt, ptr %esp, align 8
  br label %ebr.rep.next

ebr.rep.next:
  %er.n = add nuw nsw i64 %er, 1
  %ermore = icmp ult i64 %er.n, 17
  br i1 %ermore, label %ebr.rep, label %ebr.done

ebr.done:
  call void @ut_report_dist(ptr @ebr.samp, i64 16, i64 1000000, ptr @lbl.ebr)
  br label %hp.rep

; ---- HP reader ----
hp.rep:
  %hr = phi i64 [ 0, %ebr.done ], [ %hr.n, %hp.rep.next ]
  %ht0 = call double @ut_now_sec()
  br label %hp.loop

hp.loop:
  %hi = phi i64 [ 0, %hp.rep ], [ %hi.n, %hp.loop ]
  %hacc = phi i64 [ 0, %hp.rep ], [ %hacc.n, %hp.loop ]
  %hold = load atomic ptr, ptr @g.head acquire, align 8
  call void @universe_conc_hp_protect(i64 0, i64 0, ptr %hold)
  %hold2 = load atomic ptr, ptr @g.head acquire, align 8
  %hcp = getelementptr inbounds nuw i8, ptr %hold, i64 8
  %hc = load i64, ptr %hcp, align 8
  call void @universe_conc_hp_clear(i64 0, i64 0)
  %hacc.n = xor i64 %hacc, %hc
  %hi.n = add nuw i64 %hi, 1
  %hmore = icmp ult i64 %hi.n, 1000000
  br i1 %hmore, label %hp.loop, label %hp.end

hp.end:
  store volatile i64 %hacc.n, ptr @g.sink, align 8
  %ht1 = call double @ut_now_sec()
  %hdt = fsub double %ht1, %ht0
  %hwarm = icmp eq i64 %hr, 0
  br i1 %hwarm, label %hp.rep.next, label %hp.rep.store

hp.rep.store:
  %hidx = sub i64 %hr, 1
  %hsp = getelementptr inbounds [16 x double], ptr @hp.samp, i64 0, i64 %hidx
  store double %hdt, ptr %hsp, align 8
  br label %hp.rep.next

hp.rep.next:
  %hr.n = add nuw nsw i64 %hr, 1
  %hrmore = icmp ult i64 %hr.n, 17
  br i1 %hrmore, label %hp.rep, label %hp.done

hp.done:
  call void @ut_report_dist(ptr @hp.samp, i64 16, i64 1000000, ptr @lbl.hp)
  br label %rc.rep

; ---- reclaim rate: retire + advance ----
rc.rep:
  %rr = phi i64 [ 0, %hp.done ], [ %rr.n, %rc.rep.next ]
  store i64 0, ptr @g.allocated, align 8
  store i64 0, ptr @g.freed, align 8
  %rt0 = call double @ut_now_sec()
  br label %rc.loop

rc.loop:
  %ri = phi i64 [ 0, %rc.rep ], [ %ri.n, %rc.iter ]
  call void @universe_conc_ebr_enter(i64 0)
  %rn = call ptr @node_alloc()
  call void @universe_conc_ebr_retire(i64 0, ptr %rn, ptr @node_free)
  call void @universe_conc_ebr_exit(i64 0)
  %rm = and i64 %ri, 255
  %radv = icmp eq i64 %rm, 0
  br i1 %radv, label %rc.adv, label %rc.iter

rc.adv:
  %rx = call i32 @universe_conc_ebr_try_advance(i64 0)
  br label %rc.iter

rc.iter:
  %ri.n = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, 1000000
  br i1 %rmore, label %rc.loop, label %rc.flush

rc.flush:
  %fr = phi i64 [ 0, %rc.iter ], [ %fr.n, %rc.flush ]
  %fx = call i32 @universe_conc_ebr_try_advance(i64 0)
  %fr.n = add nuw i64 %fr, 1
  %fmore = icmp ult i64 %fr.n, 16
  br i1 %fmore, label %rc.flush, label %rc.end

rc.end:
  %rt1 = call double @ut_now_sec()
  %rdt = fsub double %rt1, %rt0
  %rwarm = icmp eq i64 %rr, 0
  br i1 %rwarm, label %rc.rep.next, label %rc.rep.store

rc.rep.store:
  %ridx = sub i64 %rr, 1
  %rsp = getelementptr inbounds [16 x double], ptr @rc.samp, i64 0, i64 %ridx
  store double %rdt, ptr %rsp, align 8
  br label %rc.rep.next

rc.rep.next:
  %rr.n = add nuw nsw i64 %rr, 1
  %rrmore = icmp ult i64 %rr.n, 17
  br i1 %rrmore, label %rc.rep, label %rc.done

rc.done:
  call void @ut_report_dist(ptr @rc.samp, i64 16, i64 1000000, ptr @lbl.rc)
  ; drain the initial bench node from the stack.
  call void @teardown_drain_ebr()
  ret void
}

; ===========================================================================
; main
; ===========================================================================
define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_ebr_sanity()
  call void @test_hp_sanity()
  call void @test_seqlock_sanity()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %dobench, label %dostress

dobench:
  call void @bench()
  br label %fin

dostress:
  store i64 100000, ptr @g.ops, align 8
  ; EBR rounds
  br label %ebr.round

ebr.round:
  %er = phi i64 [ 0, %dostress ], [ %er.n, %ebr.round ]
  %ev = phi i64 [ 0, %dostress ], [ %ev.n, %ebr.round ]
  %erv = call i64 @run_round_ebr()
  %ev.n = add i64 %ev, %erv
  %er.n = add nuw i64 %er, 1
  %emore = icmp ult i64 %er.n, 10
  br i1 %emore, label %ebr.round, label %ebr.verdict

ebr.verdict:
  call void @ut_check_eq(i64 %ev.n, i64 0, ptr @m.ebr.stress)
  br label %hp.round

hp.round:
  %hr = phi i64 [ 0, %ebr.verdict ], [ %hr.n, %hp.round ]
  %hv = phi i64 [ 0, %ebr.verdict ], [ %hv.n, %hp.round ]
  %hrv = call i64 @run_round_hp()
  %hv.n = add i64 %hv, %hrv
  %hr.n = add nuw i64 %hr, 1
  %hmore = icmp ult i64 %hr.n, 10
  br i1 %hmore, label %hp.round, label %hp.verdict

hp.verdict:
  call void @ut_check_eq(i64 %hv.n, i64 0, ptr @m.hp.stress)
  br label %fin

fin:
  call void @universe_conc_ebr_reset()
  call void @universe_conc_hp_reset()
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

!0 = !{!"branch_weights", i32 1, i32 2000}
