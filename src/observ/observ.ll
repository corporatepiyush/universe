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

; universe_observ_* — observability primitives: async logging, striped
; metric counters + gauges, and an HDR-style latency histogram.
;
; ============================================================================
; DESIGN
; ----------------------------------------------------------------------------
; The single hard rule for observability is the compute/memory/IO separation
; (CLAUDE.md): the code that PRODUCES a signal must never format, allocate on
; the heap, or block. Formatting (integer -> text) and the write(2) live in
; the DRAIN, run by a background writer thread — never in the logging thread.
;
; --- Async logging -----------------------------------------------------------
;   * A logger owns a bounded lock-free MPSC ring (src/concurrent/mpsc.ll):
;     many worker threads push, one drainer pops. The producer side is
;     wait-free (XADD ticket) so a burst of logs never serialises callers.
;   * The LEVEL GATE is ONE atomic monotonic load + a compare. A below-
;     threshold call (DEBUG on an INFO build) returns before touching the
;     queue, the clock, or the record — a true no-op. `monotonic` is correct:
;     the threshold gates nothing but its own value; it publishes no other
;     memory, so no acquire/release is owed.
;   * A LOG RECORD is a fixed 64-byte struct filled on the caller's STACK
;     (no heap alloc) and memcpy'd into the ring by push:
;         +0  ts (logical sequence)  +8  msg (ptr to static NUL string)
;         +16 level(i32) +20 tid(i32) +24 argc(i32)  +32 args[4] i64
;     The timestamp is a LOGICAL sequence (one atomicrmw add on the logger),
;     NOT clock_gettime — a syscall in the hot path would violate the IO
;     separation. Global order across producers is preserved by the sequence
;     (and, redundantly, by the ring ticket the drainer pops in order).
;   * DRAIN formats each record into the buffered writer (src/io/bufio.ll),
;     which batches bytes and issues one write(2) per bufferful — never a
;     syscall per record. Consecutive IDENTICAL records (same msg/level/tid/
;     args, ts excluded) COLLAPSE to a single line with an "xN" repeat count.
;
; --- Metrics -----------------------------------------------------------------
;   * Counters are STRIPED: each (counter_id, shard) pair owns its own 128 B
;     cache line, so distinct threads incrementing distinct shards never
;     share a line (no coherency ping-pong). inc = one `atomicrmw add`
;     monotonic (LSE ldadd on ARM). sum() merges the shards lazily (cold).
;   * Gauges are single atomic i64 (set/get), monotonic.
;   * All orderings monotonic: a statistics counter carries no happens-before
;     for other memory; we need atomicity/conservation of the count only.
;
; --- HDR-style histogram -----------------------------------------------------
;   * Constant-memory latency histogram with `sig_digits` relative precision
;     over [min, max]. record() is O(1): a count-leading-zeros picks the
;     exponent bucket, a shift picks the linear sub-bucket, then ONE array
;     increment — no loops, no calls, no allocation on the record path.
;   * Layout (one allocation): parameters, running total/min/max/sum, then a
;     flat i64 counts array of (buckets+1)*subBucketHalfCount entries.
;   * percentile()/min/max/count/mean read the counts; merge() adds two
;     histograms of identical shape (== recording the union). Single-threaded
;     (concurrency deferred): record increments are plain, not atomic.
;
; HARDENING-TODO: none crypto here; logging drop-accounting is best-effort.

declare ptr  @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr  @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32  @posix_memalign(ptr, i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64  @llvm.ctlz.i64(i64, i1 immarg)
declare i64  @llvm.umax.i64(i64, i64)
declare i64  @llvm.umin.i64(i64, i64)
declare double @llvm.ceil.f64(double)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; siblings (linked, not modified)
declare ptr  @universe_conc_mpsc_ring_create(i64, i64)
declare i32  @universe_conc_mpsc_ring_push(ptr, ptr)
declare i32  @universe_conc_mpsc_ring_pop(ptr, ptr)
declare i64  @universe_conc_mpsc_ring_count(ptr)
declare void @universe_conc_mpsc_ring_destroy(ptr)
declare i64  @universe_io_writer_write(ptr, ptr, i64)
declare i32  @universe_io_writer_flush(ptr)

@obs.sp = private unnamed_addr constant [1 x i8] c" "
@obs.nl = private unnamed_addr constant [1 x i8] c"\0A"
@obs.sx = private unnamed_addr constant [2 x i8] c" x"

; ===========================================================================
; internal formatting helpers (DRAIN side only — cold, never on a hot path)
; ===========================================================================

; obs_u64dec(dst, val) -> len : write unsigned decimal, return byte count.
define internal i64 @obs_u64dec(ptr %dst, i64 %val) #2 {
entry:
  %z = icmp eq i64 %val, 0
  br i1 %z, label %zero, label %conv

zero:
  store i8 48, ptr %dst, align 1
  ret i64 1

conv:
  %tmp = alloca [24 x i8], align 1
  br label %loop

loop:
  %i = phi i64 [ 24, %conv ], [ %i.n, %loop ]
  %v = phi i64 [ %val, %conv ], [ %v.n, %loop ]
  %i.n = sub i64 %i, 1
  %d = urem i64 %v, 10
  %v.n = udiv i64 %v, 10
  %dc = trunc i64 %d to i8
  %ch = add i8 %dc, 48
  %p = getelementptr inbounds nuw [24 x i8], ptr %tmp, i64 0, i64 %i.n
  store i8 %ch, ptr %p, align 1
  %more = icmp ne i64 %v.n, 0
  br i1 %more, label %loop, label %done

done:
  %len = sub i64 24, %i.n
  %src = getelementptr inbounds nuw [24 x i8], ptr %tmp, i64 0, i64 %i.n
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %len, i1 false)
  ret i64 %len
}

; obs_i64dec(dst, val) -> len : signed decimal.
define internal i64 @obs_i64dec(ptr %dst, i64 %val) #2 {
entry:
  %neg = icmp slt i64 %val, 0
  br i1 %neg, label %negb, label %posb

posb:
  %l0 = call i64 @obs_u64dec(ptr %dst, i64 %val)
  ret i64 %l0

negb:
  store i8 45, ptr %dst, align 1
  %u = sub i64 0, %val
  %d1 = getelementptr inbounds nuw i8, ptr %dst, i64 1
  %l1 = call i64 @obs_u64dec(ptr %d1, i64 %u)
  %tot = add i64 %l1, 1
  ret i64 %tot
}

; obs_wnum(w, val) : format signed decimal into the buffered writer.
define internal void @obs_wnum(ptr %w, i64 %val) #2 {
entry:
  %buf = alloca [24 x i8], align 1
  %len = call i64 @obs_i64dec(ptr %buf, i64 %val)
  %r = call i64 @universe_io_writer_write(ptr %w, ptr %buf, i64 %len)
  ret void
}

define internal i64 @obs_strlen(ptr %s) #2 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %p = getelementptr inbounds nuw i8, ptr %s, i64 %i
  %c = load i8, ptr %p, align 1
  %z = icmp eq i8 %c, 0
  br i1 %z, label %done, label %cont
cont:
  %i.n = add nuw i64 %i, 1
  br label %loop
done:
  ret i64 %i
}

; obs_reccmp(a, b) -> i1 : true if records equal on bytes [8,64) (identity
; excluding the ts field at [0,8)).
define internal i1 @obs_reccmp(ptr %a, ptr %b) #1 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 8, %entry ], [ %i.n, %cont ]
  %pa = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %pb = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %va = load i64, ptr %pa, align 8
  %vb = load i64, ptr %pb, align 8
  %ne = icmp ne i64 %va, %vb
  br i1 %ne, label %diff, label %cont
cont:
  %i.n = add nuw i64 %i, 8
  %more = icmp ult i64 %i.n, 64
  br i1 %more, label %loop, label %same
diff:
  ret i1 false
same:
  ret i1 true
}

; obs_emit(w, rec, repeat) : format one record line into the writer.
define internal void @obs_emit(ptr %w, ptr %rec, i64 %repeat) #2 {
entry:
  %ts = load i64, ptr %rec, align 8
  %msgp = getelementptr inbounds nuw i8, ptr %rec, i64 8
  %msg = load ptr, ptr %msgp, align 8
  %lvlp = getelementptr inbounds nuw i8, ptr %rec, i64 16
  %lvl = load i32, ptr %lvlp, align 4
  %tidp = getelementptr inbounds nuw i8, ptr %rec, i64 20
  %tid = load i32, ptr %tidp, align 4
  %argcp = getelementptr inbounds nuw i8, ptr %rec, i64 24
  %argc = load i32, ptr %argcp, align 4
  %argsbase = getelementptr inbounds nuw i8, ptr %rec, i64 32

  call void @obs_wnum(ptr %w, i64 %ts)
  %w1 = call i64 @universe_io_writer_write(ptr %w, ptr @obs.sp, i64 1)
  %lvl64 = zext i32 %lvl to i64
  call void @obs_wnum(ptr %w, i64 %lvl64)
  %w2 = call i64 @universe_io_writer_write(ptr %w, ptr @obs.sp, i64 1)
  %tid64 = zext i32 %tid to i64
  call void @obs_wnum(ptr %w, i64 %tid64)
  %w3 = call i64 @universe_io_writer_write(ptr %w, ptr @obs.sp, i64 1)
  %ml = call i64 @obs_strlen(ptr %msg)
  %w4 = call i64 @universe_io_writer_write(ptr %w, ptr %msg, i64 %ml)

  %argc64 = zext i32 %argc to i64
  %hasargs = icmp ugt i64 %argc64, 0
  br i1 %hasargs, label %aloop, label %afterargs

aloop:
  %j = phi i64 [ 0, %entry ], [ %j.n, %aloop ]
  %ws = call i64 @universe_io_writer_write(ptr %w, ptr @obs.sp, i64 1)
  %ap = getelementptr inbounds nuw i64, ptr %argsbase, i64 %j
  %av = load i64, ptr %ap, align 8
  call void @obs_wnum(ptr %w, i64 %av)
  %j.n = add nuw i64 %j, 1
  %amore = icmp ult i64 %j.n, %argc64
  br i1 %amore, label %aloop, label %afterargs

afterargs:
  %rep = icmp ugt i64 %repeat, 0
  br i1 %rep, label %dorep, label %nl

dorep:
  %wx = call i64 @universe_io_writer_write(ptr %w, ptr @obs.sx, i64 2)
  %repcount = add i64 %repeat, 1
  call void @obs_wnum(ptr %w, i64 %repcount)
  br label %nl

nl:
  %wn = call i64 @universe_io_writer_write(ptr %w, ptr @obs.nl, i64 1)
  ret void
}

; ===========================================================================
; Async logging — logger layout (malloc 192):
;   +0    queue : ptr (mpsc ring, elem_size 64)
;   +8    threshold : atomic i32   (the level gate)
;   +64   seq : atomic i64         (logical timestamp source, own line)
;   +128  dropped : atomic i64     (queue-full drops, own line)
; ===========================================================================

define noalias ptr @universe_observ_logger_create(i64 %capacity, i32 %threshold) local_unnamed_addr #3 {
entry:
  %q = call ptr @universe_conc_mpsc_ring_create(i64 %capacity, i64 64)
  %q.null = icmp eq ptr %q, null
  br i1 %q.null, label %fail, label %alloc, !prof !0

alloc:
  %lg = call ptr @calloc(i64 1, i64 192)
  %lg.null = icmp eq ptr %lg, null
  br i1 %lg.null, label %freeq, label %init, !prof !0

freeq:
  call void @universe_conc_mpsc_ring_destroy(ptr %q)
  br label %fail

init:
  store ptr %q, ptr %lg, align 8
  %thr.p = getelementptr inbounds nuw i8, ptr %lg, i64 8
  store atomic i32 %threshold, ptr %thr.p monotonic, align 4
  ret ptr %lg

fail:
  ret ptr null
}

define void @universe_observ_logger_destroy(ptr %lg) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %ret, label %do, !prof !0
do:
  %q = load ptr, ptr %lg, align 8
  call void @universe_conc_mpsc_ring_destroy(ptr %q)
  call void @free(ptr nonnull %lg)
  br label %ret
ret:
  ret void
}

define void @universe_observ_log_set_level(ptr %lg, i32 %threshold) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %ret, label %do, !prof !0
do:
  %thr.p = getelementptr inbounds nuw i8, ptr %lg, i64 8
  store atomic i32 %threshold, ptr %thr.p monotonic, align 4
  br label %ret
ret:
  ret void
}

; log_enabled(lg, level) -> i1 : the gate — one atomic load + compare.
define i1 @universe_observ_log_enabled(ptr %lg, i32 %level) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %no, label %chk, !prof !0
no:
  ret i1 false
chk:
  %thr.p = getelementptr inbounds nuw i8, ptr %lg, i64 8
  %thr = load atomic i32, ptr %thr.p monotonic, align 4
  %en = icmp sge i32 %level, %thr
  ret i1 %en
}

; log(lg, level, tid, msg, argc, a0..a3) -> i32 (0 ok, 1 null, 6 dropped/full).
; Hot path: gate load+compare; if enabled, one seq XADD, fill a 64B stack
; record, and one bounded MPSC push. No format, no heap alloc, no syscall.
define i32 @universe_observ_log(ptr %lg, i32 %level, i32 %tid, ptr %msg, i64 %argc, i64 %a0, i64 %a1, i64 %a2, i64 %a3) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %err.null, label %gate, !prof !0

err.null:
  ret i32 1

gate:
  %thr.p = getelementptr inbounds nuw i8, ptr %lg, i64 8
  %thr = load atomic i32, ptr %thr.p monotonic, align 4
  %en = icmp sge i32 %level, %thr
  br i1 %en, label %build, label %suppressed, !prof !2

suppressed:
  ret i32 0

build:
  %seq.p = getelementptr inbounds nuw i8, ptr %lg, i64 64
  %seq = atomicrmw add ptr %seq.p, i64 1 monotonic, align 8
  %rec = alloca [64 x i8], align 8
  store i64 %seq, ptr %rec, align 8
  %msgp = getelementptr inbounds nuw i8, ptr %rec, i64 8
  store ptr %msg, ptr %msgp, align 8
  %lvlp = getelementptr inbounds nuw i8, ptr %rec, i64 16
  store i32 %level, ptr %lvlp, align 4
  %tidp = getelementptr inbounds nuw i8, ptr %rec, i64 20
  store i32 %tid, ptr %tidp, align 4
  %argc32 = trunc i64 %argc to i32
  %argcp = getelementptr inbounds nuw i8, ptr %rec, i64 24
  store i32 %argc32, ptr %argcp, align 4
  %a0p = getelementptr inbounds nuw i8, ptr %rec, i64 32
  store i64 %a0, ptr %a0p, align 8
  %a1p = getelementptr inbounds nuw i8, ptr %rec, i64 40
  store i64 %a1, ptr %a1p, align 8
  %a2p = getelementptr inbounds nuw i8, ptr %rec, i64 48
  store i64 %a2, ptr %a2p, align 8
  %a3p = getelementptr inbounds nuw i8, ptr %rec, i64 56
  store i64 %a3, ptr %a3p, align 8
  %q = load ptr, ptr %lg, align 8
  %rc = call i32 @universe_conc_mpsc_ring_push(ptr %q, ptr %rec)
  %ok = icmp eq i32 %rc, 0
  br i1 %ok, label %done, label %drop, !prof !1

drop:
  %drop.p = getelementptr inbounds nuw i8, ptr %lg, i64 128
  %od = atomicrmw add ptr %drop.p, i64 1 monotonic, align 8
  ret i32 %rc

done:
  ret i32 0
}

; log_drain(lg, writer) -> i64 records drained. Formats every popped record
; into the buffered writer, collapsing consecutive identical records.
define i64 @universe_observ_log_drain(ptr %lg, ptr %w) local_unnamed_addr #3 {
entry:
  %lg.null = icmp eq ptr %lg, null
  %w.null = icmp eq ptr %w, null
  %anynull = or i1 %lg.null, %w.null
  br i1 %anynull, label %ret0, label %setup, !prof !0

ret0:
  ret i64 0

setup:
  %q = load ptr, ptr %lg, align 8
  %prev = alloca [64 x i8], align 8
  %cur = alloca [64 x i8], align 8
  br label %loop

loop:
  %count = phi i64 [ 0, %setup ], [ %count.a, %newrec ], [ %count.b, %firstrec ], [ %count.c, %samerec ]
  %have = phi i1 [ false, %setup ], [ true, %newrec ], [ true, %firstrec ], [ true, %samerec ]
  %repeat = phi i64 [ 0, %setup ], [ 0, %newrec ], [ 0, %firstrec ], [ %repeat.n, %samerec ]
  %rc = call i32 @universe_conc_mpsc_ring_pop(ptr %q, ptr %cur)
  %empty = icmp ne i32 %rc, 0
  br i1 %empty, label %flush, label %got

got:
  %count.now = add i64 %count, 1
  br i1 %have, label %cmp, label %first

first:
  %count.b = add i64 0, %count.now
  call void @llvm.memcpy.p0.p0.i64(ptr %prev, ptr %cur, i64 64, i1 false)
  br label %firstrec

firstrec:
  br label %loop

cmp:
  %same = call i1 @obs_reccmp(ptr %prev, ptr %cur)
  br i1 %same, label %same.b, label %diff

same.b:
  %count.c = add i64 0, %count.now
  %repeat.n = add i64 %repeat, 1
  br label %samerec

samerec:
  br label %loop

diff:
  %count.a = add i64 0, %count.now
  call void @obs_emit(ptr %w, ptr %prev, i64 %repeat)
  call void @llvm.memcpy.p0.p0.i64(ptr %prev, ptr %cur, i64 64, i1 false)
  br label %newrec

newrec:
  br label %loop

flush:
  br i1 %have, label %emitlast, label %doflush

emitlast:
  call void @obs_emit(ptr %w, ptr %prev, i64 %repeat)
  br label %doflush

doflush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  ret i64 %count
}

define i64 @universe_observ_log_pending(ptr %lg) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %z, label %do, !prof !0
z:
  ret i64 0
do:
  %q = load ptr, ptr %lg, align 8
  %n = call i64 @universe_conc_mpsc_ring_count(ptr %q)
  ret i64 %n
}

define i64 @universe_observ_log_dropped(ptr %lg) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %lg, null
  br i1 %is.null, label %z, label %do, !prof !0
z:
  ret i64 0
do:
  %drop.p = getelementptr inbounds nuw i8, ptr %lg, i64 128
  %d = load atomic i64, ptr %drop.p monotonic, align 8
  ret i64 %d
}

; ===========================================================================
; Metrics — striped counters + gauges. Header (posix_memalign 128):
;   +0 ncnt  +8 ngauge  +16 nshards  +24 mask  +32 stride(=nshards*128)
;   +40 gaugeoff(=128+ncnt*stride)
;   counters at 128 + cid*stride + shard*128 ; gauges at gaugeoff + gid*8
; ===========================================================================

define noalias ptr @universe_observ_metrics_create(i64 %ncnt, i64 %ngauge, i64 %nshards) local_unnamed_addr #3 {
entry:
  %is0 = icmp eq i64 %nshards, 0
  %req = select i1 %is0, i64 64, i64 %nshards
  %clamp = call i64 @llvm.umin.i64(i64 %req, i64 1048576)
  %n0 = call i64 @llvm.umax.i64(i64 %clamp, i64 1)
  %nm1 = add i64 %n0, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %nm1, i1 false)
  %shift = sub nuw nsw i64 64, %lz
  %nsh = shl nuw i64 1, %shift
  %mask = add i64 %nsh, -1
  ; stride = nsh * 128
  %st = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nsh, i64 128)
  %stride = extractvalue { i64, i1 } %st, 0
  %st.o = extractvalue { i64, i1 } %st, 1
  ; cbytes = ncnt * stride
  %cb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ncnt, i64 %stride)
  %cbytes = extractvalue { i64, i1 } %cb, 0
  %cb.o = extractvalue { i64, i1 } %cb, 1
  ; gbytes = ngauge * 8
  %gb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %ngauge, i64 8)
  %gbytes = extractvalue { i64, i1 } %gb, 0
  %gb.o = extractvalue { i64, i1 } %gb, 1
  ; gaugeoff = 128 + cbytes
  %go = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %cbytes, i64 128)
  %gaugeoff = extractvalue { i64, i1 } %go, 0
  %go.o = extractvalue { i64, i1 } %go, 1
  ; total = gaugeoff + gbytes
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %gaugeoff, i64 %gbytes)
  %total = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %o0 = or i1 %st.o, %cb.o
  %o1 = or i1 %o0, %gb.o
  %o2 = or i1 %o1, %go.o
  %ovf = or i1 %o2, %tt.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %slot = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot, i64 128, i64 %total)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %fail, label %chk, !prof !0

chk:
  %mem = load ptr, ptr %slot, align 8
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  call void @llvm.memset.p0.i64(ptr %mem, i8 0, i64 %total, i1 false)
  store i64 %ncnt, ptr %mem, align 8
  %ng.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %ngauge, ptr %ng.p, align 8
  %ns.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %nsh, ptr %ns.p, align 8
  %mk.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %mask, ptr %mk.p, align 8
  %sd.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i64 %stride, ptr %sd.p, align 8
  %gf.p = getelementptr inbounds nuw i8, ptr %mem, i64 40
  store i64 %gaugeoff, ptr %gf.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define void @universe_observ_metrics_destroy(ptr %m) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %do, !prof !0
do:
  call void @free(ptr nonnull %m)
  br label %ret
ret:
  ret void
}

; counter_inc — one atomicrmw on the caller's shard line.
define void @universe_observ_counter_inc(ptr %m, i64 %cid, i64 %shard, i64 %delta) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %go, !prof !0
go:
  %mk.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %mask = load i64, ptr %mk.p, align 8
  %sd.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %stride = load i64, ptr %sd.p, align 8
  %sidx = and i64 %shard, %mask
  %cbase = mul i64 %cid, %stride
  %soff = shl i64 %sidx, 7
  %off0 = add i64 %cbase, %soff
  %off = add i64 %off0, 128
  %cnt.p = getelementptr inbounds nuw i8, ptr %m, i64 %off
  %old = atomicrmw add ptr %cnt.p, i64 %delta monotonic, align 8
  br label %ret
ret:
  ret void
}

; counter_sum — merge all shards for a counter (cold).
define i64 @universe_observ_counter_sum(ptr %m, i64 %cid) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %z, label %setup, !prof !0
z:
  ret i64 0
setup:
  %ns.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %nsh = load i64, ptr %ns.p, align 8
  %sd.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %stride = load i64, ptr %sd.p, align 8
  %cbase = mul i64 %cid, %stride
  %base = add i64 %cbase, 128
  br label %loop
loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %setup ], [ %acc.n, %loop ]
  %soff = shl i64 %i, 7
  %off = add i64 %base, %soff
  %cnt.p = getelementptr inbounds nuw i8, ptr %m, i64 %off
  %v = load atomic i64, ptr %cnt.p monotonic, align 8
  %acc.n = add i64 %acc, %v
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %nsh
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

define void @universe_observ_gauge_set(ptr %m, i64 %gid, i64 %value) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %ret, label %go, !prof !0
go:
  %gf.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %goff = load i64, ptr %gf.p, align 8
  %eoff = shl i64 %gid, 3
  %off = add i64 %goff, %eoff
  %g.p = getelementptr inbounds nuw i8, ptr %m, i64 %off
  store atomic i64 %value, ptr %g.p monotonic, align 8
  br label %ret
ret:
  ret void
}

define i64 @universe_observ_gauge_get(ptr %m, i64 %gid) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %z, label %go, !prof !0
z:
  ret i64 0
go:
  %gf.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %goff = load i64, ptr %gf.p, align 8
  %eoff = shl i64 %gid, 3
  %off = add i64 %goff, %eoff
  %g.p = getelementptr inbounds nuw i8, ptr %m, i64 %off
  %v = load atomic i64, ptr %g.p monotonic, align 8
  ret i64 %v
}

; ===========================================================================
; HDR-style histogram — layout (calloc):
;   +0 subBucketHalfCount  +8 subBucketHalfCountMagnitude  +16 subBucketMask
;   +24 unitMagnitude      +32 leadingZeroCountBase        +40 countsLen
;   +48 totalCount  +56 minValue  +64 maxValue  +72 sumValue
;   +80 counts[countsLen] i64
; ===========================================================================

define noalias ptr @universe_observ_hist_create(i64 %min, i64 %max, i64 %sig) local_unnamed_addr #3 {
entry:
  %min1 = call i64 @llvm.umax.i64(i64 %min, i64 1)
  %sigc = call i64 @llvm.umin.i64(i64 %sig, i64 5)
  ; unitMagnitude = 63 - clz(min1)
  %clzmin = call i64 @llvm.ctlz.i64(i64 %min1, i1 false)
  %unitMag = sub i64 63, %clzmin
  br label %powloop

powloop:
  %pi = phi i64 [ 0, %entry ], [ %pi.n, %powbody ]
  %pv = phi i64 [ 1, %entry ], [ %pv.n, %powbody ]
  %pdone = icmp uge i64 %pi, %sigc
  br i1 %pdone, label %afterpow, label %powbody

powbody:
  %pv.n = mul i64 %pv, 10
  %pi.n = add nuw i64 %pi, 1
  br label %powloop

afterpow:
  ; lv = 2 * 10^sig ; subBucketCountMagnitude = ceil(log2(lv)) = 64 - clz(lv-1)
  %lv = shl i64 %pv, 1
  %lvm1 = sub i64 %lv, 1
  %clzlv = call i64 @llvm.ctlz.i64(i64 %lvm1, i1 false)
  %scm = sub i64 64, %clzlv
  %scmMax = call i64 @llvm.umax.i64(i64 %scm, i64 1)
  %shcm = sub i64 %scmMax, 1
  %shc = shl i64 1, %shcm
  %scm1 = add i64 %shcm, 1
  %sbc = shl i64 1, %scm1
  %sbcm1 = sub i64 %sbc, 1
  %sbMask = shl i64 %sbcm1, %unitMag
  ; leadingZeroCountBase = 64 - unitMag - shcm - 1
  %lz0 = sub i64 64, %unitMag
  %lz1 = sub i64 %lz0, %shcm
  %lzcb = sub i64 %lz1, 1
  ; buckets needed
  %smallest0 = shl i64 %sbc, %unitMag
  br label %bloop

bloop:
  %bn = phi i64 [ 1, %afterpow ], [ %bn.n, %bbody ]
  %sm = phi i64 [ %smallest0, %afterpow ], [ %sm.n, %bbody ]
  %fits = icmp ule i64 %sm, %max
  %cap = icmp ult i64 %bn, 64
  %cont = and i1 %fits, %cap
  br i1 %cont, label %bbody, label %afterb

bbody:
  %sm.n = shl i64 %sm, 1
  %bn.n = add nuw i64 %bn, 1
  br label %bloop

afterb:
  %bn1 = add i64 %bn, 1
  %countsLen = mul i64 %bn1, %shc
  ; total = 80 + countsLen*8
  %cbt = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %countsLen, i64 8)
  %cbytes = extractvalue { i64, i1 } %cbt, 0
  %cbt.o = extractvalue { i64, i1 } %cbt, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %cbytes, i64 80)
  %total = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %ovf = or i1 %cbt.o, %tt.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @calloc(i64 1, i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %shc, ptr %mem, align 8
  %f8 = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %shcm, ptr %f8, align 8
  %f16 = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %sbMask, ptr %f16, align 8
  %f24 = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %unitMag, ptr %f24, align 8
  %f32 = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i64 %lzcb, ptr %f32, align 8
  %f40 = getelementptr inbounds nuw i8, ptr %mem, i64 40
  store i64 %countsLen, ptr %f40, align 8
  ; totalCount=0 (calloc), minValue = INT64_MAX, maxValue=0, sum=0
  %f56 = getelementptr inbounds nuw i8, ptr %mem, i64 56
  store i64 9223372036854775807, ptr %f56, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define void @universe_observ_hist_destroy(ptr %h) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %ret, label %do, !prof !0
do:
  call void @free(ptr nonnull %h)
  br label %ret
ret:
  ret void
}

; hist_record — O(1): clz bucket + shift sub-bucket + one array increment.
define void @universe_observ_hist_record(ptr %h, i64 %value) local_unnamed_addr #0 {
entry:
  %is.null = icmp eq ptr %h, null
  %neg = icmp slt i64 %value, 0
  %skip = or i1 %is.null, %neg
  br i1 %skip, label %ret, label %go, !prof !0

go:
  %sbMask.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %sbMask = load i64, ptr %sbMask.p, align 8
  %unitMag.p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %unitMag = load i64, ptr %unitMag.p, align 8
  %lzcb.p = getelementptr inbounds nuw i8, ptr %h, i64 32
  %lzcb = load i64, ptr %lzcb.p, align 8
  %shcm.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %shcm = load i64, ptr %shcm.p, align 8
  %shc = load i64, ptr %h, align 8

  %vor = or i64 %value, %sbMask
  %clzv = call i64 @llvm.ctlz.i64(i64 %vor, i1 false)
  %b = sub i64 %lzcb, %clzv
  %sh = add i64 %b, %unitMag
  %sbi = lshr i64 %value, %sh
  %b1 = add i64 %b, 1
  %bb = shl i64 %b1, %shcm
  %off = sub i64 %sbi, %shc
  %idx = add i64 %bb, %off

  ; clamp to [0, countsLen-1]
  %countsLen.p = getelementptr inbounds nuw i8, ptr %h, i64 40
  %countsLen = load i64, ptr %countsLen.p, align 8
  %last = sub i64 %countsLen, 1
  %over = icmp uge i64 %idx, %countsLen
  %idxc = select i1 %over, i64 %last, i64 %idx

  %counts = getelementptr inbounds nuw i8, ptr %h, i64 80
  %slot = getelementptr inbounds nuw i64, ptr %counts, i64 %idxc
  %cv = load i64, ptr %slot, align 8
  %cv.n = add i64 %cv, 1
  store i64 %cv.n, ptr %slot, align 8

  %tc.p = getelementptr inbounds nuw i8, ptr %h, i64 48
  %tc = load i64, ptr %tc.p, align 8
  %tc.n = add i64 %tc, 1
  store i64 %tc.n, ptr %tc.p, align 8

  %mn.p = getelementptr inbounds nuw i8, ptr %h, i64 56
  %mn = load i64, ptr %mn.p, align 8
  %mn.n = call i64 @llvm.umin.i64(i64 %mn, i64 %value)
  store i64 %mn.n, ptr %mn.p, align 8

  %mx.p = getelementptr inbounds nuw i8, ptr %h, i64 64
  %mx = load i64, ptr %mx.p, align 8
  %mx.n = call i64 @llvm.umax.i64(i64 %mx, i64 %value)
  store i64 %mx.n, ptr %mx.p, align 8

  %sm.p = getelementptr inbounds nuw i8, ptr %h, i64 72
  %sm = load i64, ptr %sm.p, align 8
  %sm.n = add i64 %sm, %value
  store i64 %sm.n, ptr %sm.p, align 8
  br label %ret

ret:
  ret void
}

; obs_hist_valfromidx(h, idx) -> value (lowest value equivalent to bucket).
define internal i64 @obs_hist_valfromidx(ptr %h, i64 %idx) #1 {
entry:
  %shcm.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  %shcm = load i64, ptr %shcm.p, align 8
  %shc = load i64, ptr %h, align 8
  %unitMag.p = getelementptr inbounds nuw i8, ptr %h, i64 24
  %unitMag = load i64, ptr %unitMag.p, align 8
  %bi0 = lshr i64 %idx, %shcm
  %bi = sub i64 %bi0, 1               ; bucketIndex, may be -1 as unsigned wrap
  %shcm1 = sub i64 %shc, 1
  %lowbits = and i64 %idx, %shcm1
  %sbi0 = add i64 %lowbits, %shc
  ; if bucketIndex < 0 (i.e. bi0 == 0): subBucketIndex -= shc, bucketIndex = 0
  %isfirst = icmp eq i64 %bi0, 0
  %sbi = select i1 %isfirst, i64 %lowbits, i64 %sbi0
  %bic = select i1 %isfirst, i64 0, i64 %bi
  %sh = add i64 %bic, %unitMag
  %val = shl i64 %sbi, %sh
  ret i64 %val
}

; hist_percentile(h, p) -> value at the p-th percentile (p in [0,100]).
define i64 @universe_observ_hist_percentile(ptr %h, double %p) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %z, label %chk, !prof !0
z:
  ret i64 0
chk:
  %tc.p = getelementptr inbounds nuw i8, ptr %h, i64 48
  %total = load i64, ptr %tc.p, align 8
  %empty = icmp eq i64 %total, 0
  br i1 %empty, label %z, label %go, !prof !0
go:
  %totd = uitofp i64 %total to double
  %frac = fdiv double %p, 1.000000e+02
  %td = fmul double %frac, %totd
  %tcl = call double @llvm.ceil.f64(double %td)
  %ti = fptoui double %tcl to i64
  %tmin = call i64 @llvm.umin.i64(i64 %ti, i64 %total)
  %target = call i64 @llvm.umax.i64(i64 %tmin, i64 1)
  %countsLen.p = getelementptr inbounds nuw i8, ptr %h, i64 40
  %countsLen = load i64, ptr %countsLen.p, align 8
  %counts = getelementptr inbounds nuw i8, ptr %h, i64 80
  br label %loop
loop:
  %i = phi i64 [ 0, %go ], [ %i.n, %next ]
  %run = phi i64 [ 0, %go ], [ %run.n, %next ]
  %slot = getelementptr inbounds nuw i64, ptr %counts, i64 %i
  %c = load i64, ptr %slot, align 8
  %run.n = add i64 %run, %c
  %reached = icmp uge i64 %run.n, %target
  br i1 %reached, label %hit, label %next
hit:
  %v = call i64 @obs_hist_valfromidx(ptr %h, i64 %i)
  ret i64 %v
next:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %countsLen
  br i1 %more, label %loop, label %fallback
fallback:
  %mx.p = getelementptr inbounds nuw i8, ptr %h, i64 64
  %mx = load i64, ptr %mx.p, align 8
  ret i64 %mx
}

define i64 @universe_observ_hist_count(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %z, label %go, !prof !0
z:
  ret i64 0
go:
  %tc.p = getelementptr inbounds nuw i8, ptr %h, i64 48
  %v = load i64, ptr %tc.p, align 8
  ret i64 %v
}

define i64 @universe_observ_hist_min(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %z, label %go, !prof !0
z:
  ret i64 0
go:
  %tc.p = getelementptr inbounds nuw i8, ptr %h, i64 48
  %total = load i64, ptr %tc.p, align 8
  %empty = icmp eq i64 %total, 0
  br i1 %empty, label %z, label %val, !prof !0
val:
  %mn.p = getelementptr inbounds nuw i8, ptr %h, i64 56
  %v = load i64, ptr %mn.p, align 8
  ret i64 %v
}

define i64 @universe_observ_hist_max(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %z, label %go, !prof !0
z:
  ret i64 0
go:
  %mx.p = getelementptr inbounds nuw i8, ptr %h, i64 64
  %v = load i64, ptr %mx.p, align 8
  ret i64 %v
}

define double @universe_observ_hist_mean(ptr %h) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %h, null
  br i1 %is.null, label %z, label %go, !prof !0
z:
  ret double 0.000000e+00
go:
  %tc.p = getelementptr inbounds nuw i8, ptr %h, i64 48
  %total = load i64, ptr %tc.p, align 8
  %empty = icmp eq i64 %total, 0
  br i1 %empty, label %z, label %val, !prof !0
val:
  %sm.p = getelementptr inbounds nuw i8, ptr %h, i64 72
  %sum = load i64, ptr %sm.p, align 8
  %sumd = uitofp i64 %sum to double
  %totd = uitofp i64 %total to double
  %mean = fdiv double %sumd, %totd
  ret double %mean
}

; hist_merge(dst, src) -> i32 (0 ok, 1 null, 8 incompatible shape).
define i32 @universe_observ_hist_merge(ptr %dst, ptr %src) local_unnamed_addr #3 {
entry:
  %d.null = icmp eq ptr %dst, null
  %s.null = icmp eq ptr %src, null
  %anynull = or i1 %d.null, %s.null
  br i1 %anynull, label %err.null, label %chk, !prof !0
err.null:
  ret i32 1
chk:
  %dl.p = getelementptr inbounds nuw i8, ptr %dst, i64 40
  %dl = load i64, ptr %dl.p, align 8
  %sl.p = getelementptr inbounds nuw i8, ptr %src, i64 40
  %sl = load i64, ptr %sl.p, align 8
  %bad = icmp ne i64 %dl, %sl
  br i1 %bad, label %err.arg, label %merge, !prof !0
err.arg:
  ret i32 8
merge:
  %dcounts = getelementptr inbounds nuw i8, ptr %dst, i64 80
  %scounts = getelementptr inbounds nuw i8, ptr %src, i64 80
  br label %loop
loop:
  %i = phi i64 [ 0, %merge ], [ %i.n, %loop ]
  %dp = getelementptr inbounds nuw i64, ptr %dcounts, i64 %i
  %sp = getelementptr inbounds nuw i64, ptr %scounts, i64 %i
  %dv = load i64, ptr %dp, align 8
  %sv = load i64, ptr %sp, align 8
  %nv = add i64 %dv, %sv
  store i64 %nv, ptr %dp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %dl
  br i1 %more, label %loop, label %after
after:
  ; totals
  %dtc.p = getelementptr inbounds nuw i8, ptr %dst, i64 48
  %stc.p = getelementptr inbounds nuw i8, ptr %src, i64 48
  %dtc = load i64, ptr %dtc.p, align 8
  %stc = load i64, ptr %stc.p, align 8
  %ntc = add i64 %dtc, %stc
  store i64 %ntc, ptr %dtc.p, align 8
  ; sum
  %dsm.p = getelementptr inbounds nuw i8, ptr %dst, i64 72
  %ssm.p = getelementptr inbounds nuw i8, ptr %src, i64 72
  %dsm = load i64, ptr %dsm.p, align 8
  %ssm = load i64, ptr %ssm.p, align 8
  %nsm = add i64 %dsm, %ssm
  store i64 %nsm, ptr %dsm.p, align 8
  ; min
  %dmn.p = getelementptr inbounds nuw i8, ptr %dst, i64 56
  %smn.p = getelementptr inbounds nuw i8, ptr %src, i64 56
  %dmn = load i64, ptr %dmn.p, align 8
  %smn = load i64, ptr %smn.p, align 8
  %nmn = call i64 @llvm.umin.i64(i64 %dmn, i64 %smn)
  store i64 %nmn, ptr %dmn.p, align 8
  ; max
  %dmx.p = getelementptr inbounds nuw i8, ptr %dst, i64 64
  %smx.p = getelementptr inbounds nuw i8, ptr %src, i64 64
  %dmx = load i64, ptr %dmx.p, align 8
  %smx = load i64, ptr %smx.p, align 8
  %nmx = call i64 @llvm.umax.i64(i64 %dmx, i64 %smx)
  store i64 %nmx, ptr %dmx.p, align 8
  ret i32 0
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 1000, i32 1000}
