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

; io_uring backend test  ---  LINUX ONLY at runtime.
;
; The whole body is gated on universe_ioring_ring_available(): on macOS / an
; old kernel it prints a skip line and returns a clean pass (0 failures) -- a
; genuine SKIP, never a faked run. On Linux >= 5.1 it exercises: a file write
; via the ring, a read-back verify, a batch of 4 SQEs submitted at once with
; all CQEs reaped by user_data, and a socketpair send/recv round-trip.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32  @ut_summary()
declare i1   @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@ring.samp  = internal global [16 x double] zeroinitializer, align 8
@pread.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.ring   = private unnamed_addr constant [22 x i8] c"io_uring batched read\00"
@lbl.pread  = private unnamed_addr constant [20 x i8] c"posix pread syscall\00"

declare ptr @universe_ioring_ring_setup(i64, i64)
declare i1  @universe_ioring_ring_available()
declare i32 @universe_ioring_prep_read(ptr, i32, ptr, i32, i64, i64)
declare i32 @universe_ioring_prep_write(ptr, i32, ptr, i32, i64, i64)
declare i32 @universe_ioring_prep_recv(ptr, i32, ptr, i32, i32, i64)
declare i32 @universe_ioring_prep_send(ptr, i32, ptr, i32, i32, i64)
declare i32 @universe_ioring_submit(ptr)
declare i32 @universe_ioring_submit_and_wait(ptr, i64)
declare i32 @universe_ioring_peek_cqe(ptr, ptr)
declare i32 @universe_ioring_wait_cqe(ptr, ptr)
declare i32 @universe_ioring_cqe_seen(ptr)
declare void @universe_ioring_ring_destroy(ptr)

declare i32 @printf(ptr, ...)
declare i32 @open(ptr, i32, ...)
declare i32 @close(i32)
declare i32 @socketpair(i32, i32, i32, ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i64 @pread(i32, ptr, i64, i64)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

@t.path   = private unnamed_addr constant [25 x i8] c"/tmp/uni_ioring_test.dat\00", align 1
@t.skip1  = private unnamed_addr constant [58 x i8] c"ioring: io_uring unavailable, skipping (Linux 5.1+ only)\0A\00", align 1
@t.skip2  = private unnamed_addr constant [37 x i8] c"ioring: ring_setup failed, skipping\0A\00", align 1
@t.mwres  = private unnamed_addr constant [17 x i8] c"write res != len\00", align 1
@t.mwud   = private unnamed_addr constant [25 x i8] c"write user_data mismatch\00", align 1
@t.mrres  = private unnamed_addr constant [16 x i8] c"read res != len\00", align 1
@t.mrdat  = private unnamed_addr constant [19 x i8] c"read data mismatch\00", align 1
@t.mball  = private unnamed_addr constant [32 x i8] c"batch: not all completions seen\00", align 1
@t.mbres  = private unnamed_addr constant [28 x i8] c"batch: a completion bad res\00", align 1
@t.msres  = private unnamed_addr constant [16 x i8] c"send res != len\00", align 1
@t.mrr2   = private unnamed_addr constant [16 x i8] c"recv res != len\00", align 1
@t.mrd2   = private unnamed_addr constant [19 x i8] c"recv data mismatch\00", align 1

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %wbuf = alloca [64 x i8], align 8
  %rbuf = alloca [64 x i8], align 8
  %out  = alloca [16 x i8], align 8
  %sv   = alloca [2 x i32], align 8
  %sbuf = alloca [32 x i8], align 8
  %rbuf2 = alloca [32 x i8], align 8
  %avail = call i1 @universe_ioring_ring_available()
  br i1 %avail, label %have, label %skip

skip:
  %ps = call i32 (ptr, ...) @printf(ptr @t.skip1)
  br label %summary

have:
  %ring = call ptr @universe_ioring_ring_setup(i64 8, i64 0)
  %ring.null = icmp eq ptr %ring, null
  br i1 %ring.null, label %skip.b, label %run

skip.b:
  %ps2 = call i32 (ptr, ...) @printf(ptr @t.skip2)
  br label %summary

run:
  %fd = call i32 (ptr, i32, ...) @open(ptr @t.path, i32 578, i32 420)
  ; ---- WRITE test ----
  call void @llvm.memset.p0.i64(ptr %wbuf, i8 -85, i64 64, i1 false)     ; 0xAB
  %w.pr0 = call i32 @universe_ioring_prep_write(ptr %ring, i32 %fd, ptr %wbuf, i32 64, i64 0, i64 4369) ; ud 0x1111
  %w.sub = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 1)
  %w.peek = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %w.resp = getelementptr inbounds nuw i8, ptr %out, i64 8
  %w.res = load i32, ptr %w.resp, align 4
  %w.res64 = sext i32 %w.res to i64
  call void @ut_check_eq(i64 %w.res64, i64 64, ptr @t.mwres)
  %w.ud = load i64, ptr %out, align 8
  call void @ut_check_eq(i64 %w.ud, i64 4369, ptr @t.mwud)
  %w.seen = call i32 @universe_ioring_cqe_seen(ptr %ring)
  ; ---- READ test ----
  call void @llvm.memset.p0.i64(ptr %rbuf, i8 0, i64 64, i1 false)
  %r.pr0 = call i32 @universe_ioring_prep_read(ptr %ring, i32 %fd, ptr %rbuf, i32 64, i64 0, i64 8738) ; ud 0x2222
  %r.sub = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 1)
  %r.peek = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %r.resp = getelementptr inbounds nuw i8, ptr %out, i64 8
  %r.res = load i32, ptr %r.resp, align 4
  %r.res64 = sext i32 %r.res to i64
  call void @ut_check_eq(i64 %r.res64, i64 64, ptr @t.mrres)
  %r.cmp = call i32 @memcmp(ptr %wbuf, ptr %rbuf, i64 64)
  %r.eq = icmp eq i32 %r.cmp, 0
  call void @ut_check(i1 %r.eq, ptr @t.mrdat)
  %r.seen = call i32 @universe_ioring_cqe_seen(ptr %ring)
  ; ---- BATCH test: queue 4 writes at distinct offsets, submit once ----
  br label %b.qloop

b.qloop:
  %b.q = phi i64 [ 0, %run ], [ %b.q.n, %b.qiter ]
  %b.qcond = icmp ult i64 %b.q, 4
  br i1 %b.qcond, label %b.qiter, label %b.submit

b.qiter:
  %b.off = shl i64 %b.q, 6                      ; q * 64
  %b.ud = add i64 %b.q, 45056                   ; 0xB000 + q
  %b.pr = call i32 @universe_ioring_prep_write(ptr %ring, i32 %fd, ptr %wbuf, i32 64, i64 %b.off, i64 %b.ud)
  %b.q.n = add i64 %b.q, 1
  br label %b.qloop

b.submit:
  %b.sub = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 4)
  br label %b.rloop

b.rloop:
  %b.i = phi i64 [ 0, %b.submit ], [ %b.i.n, %b.riter ]
  %b.seen = phi i32 [ 0, %b.submit ], [ %b.seen.n, %b.riter ]
  %b.bad = phi i32 [ 0, %b.submit ], [ %b.bad.n, %b.riter ]
  %b.rcond = icmp ult i64 %b.i, 4
  br i1 %b.rcond, label %b.riter, label %b.after

b.riter:
  %b.peek = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %b.resp = getelementptr inbounds nuw i8, ptr %out, i64 8
  %b.res = load i32, ptr %b.resp, align 4
  %b.res.ok = icmp eq i32 %b.res, 64
  %b.badinc = select i1 %b.res.ok, i32 0, i32 1
  %b.bad.n = add i32 %b.bad, %b.badinc
  %b.udv = load i64, ptr %out, align 8
  %b.idx = and i64 %b.udv, 15
  %b.idx32 = trunc i64 %b.idx to i32
  %b.bit = shl i32 1, %b.idx32
  %b.seen.n = or i32 %b.seen, %b.bit
  %b.cseen = call i32 @universe_ioring_cqe_seen(ptr %ring)
  %b.i.n = add i64 %b.i, 1
  br label %b.rloop

b.after:
  %b.seen64 = zext i32 %b.seen to i64
  call void @ut_check_eq(i64 %b.seen64, i64 15, ptr @t.mball)
  %b.bad64 = zext i32 %b.bad to i64
  call void @ut_check_eq(i64 %b.bad64, i64 0, ptr @t.mbres)
  ; ---- SOCKET send/recv round-trip ----
  %s.sr = call i32 @socketpair(i32 1, i32 1, i32 0, ptr %sv)
  %s.ok = icmp eq i32 %s.sr, 0
  br i1 %s.ok, label %s.do, label %maybe.bench

s.do:
  %s.sfd = load i32, ptr %sv, align 4
  %s.rfdp = getelementptr inbounds nuw i8, ptr %sv, i64 4
  %s.rfd = load i32, ptr %s.rfdp, align 4
  call void @llvm.memset.p0.i64(ptr %sbuf, i8 -57, i64 32, i1 false)   ; 0xC7
  call void @llvm.memset.p0.i64(ptr %rbuf2, i8 0, i64 32, i1 false)
  %s.psend = call i32 @universe_ioring_prep_send(ptr %ring, i32 %s.sfd, ptr %sbuf, i32 32, i32 0, i64 94)  ; ud 0x5E
  %s.precv = call i32 @universe_ioring_prep_recv(ptr %ring, i32 %s.rfd, ptr %rbuf2, i32 32, i32 0, i64 78) ; ud 0x4E
  %s.sub = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 2)
  br label %s.rloop

s.rloop:
  %s.i = phi i64 [ 0, %s.do ], [ %s.i.n, %s.riter ]
  %s.send = phi i32 [ -1, %s.do ], [ %s.send.n, %s.riter ]
  %s.recv = phi i32 [ -1, %s.do ], [ %s.recv.n, %s.riter ]
  %s.rcond = icmp ult i64 %s.i, 2
  br i1 %s.rcond, label %s.riter, label %s.after

s.riter:
  %s.peek = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %s.udv = load i64, ptr %out, align 8
  %s.resp = getelementptr inbounds nuw i8, ptr %out, i64 8
  %s.res = load i32, ptr %s.resp, align 4
  %s.is.recv = icmp eq i64 %s.udv, 78
  %s.recv.n = select i1 %s.is.recv, i32 %s.res, i32 %s.recv
  %s.send.n = select i1 %s.is.recv, i32 %s.send, i32 %s.res
  %s.cseen = call i32 @universe_ioring_cqe_seen(ptr %ring)
  %s.i.n = add i64 %s.i, 1
  br label %s.rloop

s.after:
  %s.send64 = sext i32 %s.send to i64
  call void @ut_check_eq(i64 %s.send64, i64 32, ptr @t.msres)
  %s.recv64 = sext i32 %s.recv to i64
  call void @ut_check_eq(i64 %s.recv64, i64 32, ptr @t.mrr2)
  %s.cmp = call i32 @memcmp(ptr %sbuf, ptr %rbuf2, i64 32)
  %s.eq = icmp eq i32 %s.cmp, 0
  call void @ut_check(i1 %s.eq, ptr @t.mrd2)
  %s.c0 = call i32 @close(i32 %s.sfd)
  %s.c1 = call i32 @close(i32 %s.rfd)
  br label %maybe.bench

maybe.bench:
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %bench.do, label %cleanup

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) for the
; io_uring batched-read path vs the posix pread reference. 17 reps: rep 0 is
; warm-up (discarded), reps 1..16 recorded. ops_per_rep = 2000 reads each.
bench.do:
  br label %rr.rep

rr.rep:
  %rr.r = phi i64 [ 0, %bench.do ], [ %rr.r.n, %rr.next ]
  %t0 = call double @ut_now_sec()
  br label %br.outer

br.outer:
  %br.o = phi i64 [ 0, %rr.rep ], [ %br.o.n, %br.rafter ]
  %br.ocond = icmp ult i64 %br.o, 250
  br i1 %br.ocond, label %br.fill, label %br.done

br.fill:
  br label %br.qloop

br.qloop:
  %br.q = phi i64 [ 0, %br.fill ], [ %br.q.n, %br.qiter ]
  %br.qcond = icmp ult i64 %br.q, 8
  br i1 %br.qcond, label %br.qiter, label %br.submit

br.qiter:
  %br.pr = call i32 @universe_ioring_prep_read(ptr %ring, i32 %fd, ptr %rbuf, i32 64, i64 0, i64 %br.q)
  %br.q.n = add i64 %br.q, 1
  br label %br.qloop

br.submit:
  %br.sub = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 8)
  br label %br.rloop

br.rloop:
  %br.r = phi i64 [ 0, %br.submit ], [ %br.r.n, %br.riter ]
  %br.rcond = icmp ult i64 %br.r, 8
  br i1 %br.rcond, label %br.riter, label %br.rafter

br.riter:
  %br.peek = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %out)
  %br.cseen = call i32 @universe_ioring_cqe_seen(ptr %ring)
  %br.r.n = add i64 %br.r, 1
  br label %br.rloop

br.rafter:
  %br.o.n = add i64 %br.o, 1
  br label %br.outer

br.done:
  %t1 = call double @ut_now_sec()
  %rr.dt = fsub double %t1, %t0
  %rr.warm = icmp eq i64 %rr.r, 0
  br i1 %rr.warm, label %rr.next, label %rr.store

rr.store:
  %rr.idx = sub i64 %rr.r, 1
  %rr.sp = getelementptr inbounds [16 x double], ptr @ring.samp, i64 0, i64 %rr.idx
  store double %rr.dt, ptr %rr.sp, align 8
  br label %rr.next

rr.next:
  %rr.r.n = add nuw nsw i64 %rr.r, 1
  %rr.more = icmp ult i64 %rr.r.n, 17
  br i1 %rr.more, label %rr.rep, label %rr.done

rr.done:
  call void @ut_report_dist(ptr @ring.samp, i64 16, i64 2000, ptr @lbl.ring)
  br label %pr.rep

pr.rep:
  %pr.r = phi i64 [ 0, %rr.done ], [ %pr.r.n, %pr.next ]
  %pt0 = call double @ut_now_sec()
  br label %bp.loop

bp.loop:
  %bp.i = phi i64 [ 0, %pr.rep ], [ %bp.i.n, %bp.iter ]
  %bp.cond = icmp ult i64 %bp.i, 2000
  br i1 %bp.cond, label %bp.iter, label %bp.done

bp.iter:
  %bp.r = call i64 @pread(i32 %fd, ptr %rbuf, i64 64, i64 0)
  %bp.i.n = add i64 %bp.i, 1
  br label %bp.loop

bp.done:
  %pt1 = call double @ut_now_sec()
  %pr.dt = fsub double %pt1, %pt0
  %pr.warm = icmp eq i64 %pr.r, 0
  br i1 %pr.warm, label %pr.next, label %pr.store

pr.store:
  %pr.idx = sub i64 %pr.r, 1
  %pr.sp = getelementptr inbounds [16 x double], ptr @pread.samp, i64 0, i64 %pr.idx
  store double %pr.dt, ptr %pr.sp, align 8
  br label %pr.next

pr.next:
  %pr.r.n = add nuw nsw i64 %pr.r, 1
  %pr.more = icmp ult i64 %pr.r.n, 17
  br i1 %pr.more, label %pr.rep, label %pr.done

pr.done:
  call void @ut_report_dist(ptr @pread.samp, i64 16, i64 2000, ptr @lbl.pread)
  br label %cleanup

cleanup:
  %c.fd = call i32 @close(i32 %fd)
  call void @universe_ioring_ring_destroy(ptr %ring)
  br label %summary

summary:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
