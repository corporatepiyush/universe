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

; io_uring-driven HTTP/1.1 server  ---  LINUX ONLY at runtime (needs io_uring,
; kernel >= 5.1; tested >= 6.15). Same server semantics as the blocking
; universe_http_serve_conn/accept_loop path, but the transport is a single
; SQ/CQ event loop instead of thread-per-connection blocking recv/send. The
; REQUEST PARSING and RESPONSE SERIALIZATION are REUSED verbatim from
; src/http/http.ll (universe_http_parse_request + universe_http_write_response);
; only the byte movement is swapped from tcp_recv/writer_flush to
; io_uring prep_recv/prep_send. The listening socket still comes from
; src/net tcp_listen (caller supplies the listen fd).
;
; On non-Linux the module COMPILES (pure IR, no target lines); at runtime
; universe_http_uring_available() forwards ring_available() == false so callers
; pick the posix path. NEVER assume this executes anywhere but Linux.
;
; DESIGN:
;   * COMPUTE/MEMORY/IO SEPARATION (CLAUDE.md). The loop is three phases per
;     turn: (1) IO — one io_uring_enter that submits every SQE prepared last
;     turn AND waits for >=1 completion (submit_and_wait); (2) compute+memory —
;     drain ALL ready CQEs, parsing zero-copy over each connection's persistent
;     recv buffer and serialising responses into that connection's persistent
;     send buffer; (3) IO — the next turn's submit_and_wait flushes every SQE we
;     queued while draining. No recv/send syscall ever sits inside the parse or
;     serialise compute; submissions are BATCHED (one enter per turn, never one
;     per SQE), exactly the "fill buffer -> compute -> flush" structure.
;   * SINGLE-THREADED ASYNC. Matches the deferred-concurrency rule: no worker
;     pool, no locks, no atomics of our own. The only cross-CPU ordering is the
;     ring protocol itself, which lives inside src/ioring (acquire/release on
;     the shared head/tail — justified there). This module issues plain calls.
;   * USER_DATA = (slot << 3) | op. op in the low 3 bits selects the state
;     machine edge (0 ACCEPT, 1 RECV, 2 SEND, 3 CLOSE); slot is the connection
;     index. An "unslotted close" (accept with no free slot) encodes slot ==
;     nslots so the CLOSE handler skips slot bookkeeping. Encode/decode and the
;     RECV/SEND decision functions are pure and exported so they are unit-
;     testable WITHOUT a live ring (real coverage on the macOS skip path).
;   * CONNECTION SLOT (64 B, one per in-flight connection, pooled and REUSED —
;     no per-connection malloc after warm-up, per the zero-alloc doctrine):
;       +0  i32 fd            +4  i32 state (0 FREE,1 RECV,2 SEND,3 CLOSING)
;       +8  ptr recv_buf      +16 i64 recv_cap
;       +24 i64 recv_len      +32 ptr send_writer (bufio writer as mem buffer)
;       +40 i64 send_off      +48 i64 send_total   +56 i32 keepalive
;     recv_buf and send_writer are allocated lazily on first use of a slot and
;     kept across keep-alive requests AND across slot reuse; freed only at
;     shutdown. This is the "chunked node pool" idea applied to connections.
;   * RESPONSE SERIALISATION reuses the bufio writer as a MEMORY sink: we reset
;     its length (documented layout fd@0/len@8/cap@16/buf@64) to 0, call
;     universe_http_write_response which accumulates the whole response into the
;     writer buffer, then prep_send the buffer [writer+64, writer.len). The
;     writer is sized >= bufcap so a small response never triggers a synchronous
;     flush. (LIMITATION: a response larger than the writer capacity would make
;     write_response flush synchronously to the fd, splitting the payload; keep
;     bufcap comfortably above the largest response. Documented, not defended.)
;   * BODY COMPLETENESS. After a clean head parse we require
;     recv_len >= head_len + body_len (Content-Length) before dispatching;
;     otherwise we re-arm RECV appending into the same buffer. Chunked request
;     bodies (body_len == 0 from the parser) are dispatched on head only —
;     acceptable for a first cut (requests rarely stream chunked); the response
;     path fully supports keep-alive and Connection: close.
;
;   Error codes (i32): 0 OK (served maxreq requests / clean stop), 2 OOM,
;   8 INVALID_ARG, 15 IO (ring setup / enter failed). HARDENING-TODO: no
;   per-connection timeouts, no slow-loris / header-flood limits beyond the
;   buffer cap, no request smuggling defenses. Deferred with the rest of http.
;
; API (universe_http_uring_*):
;   i1  available()                                  ; == ioring ring_available
;   i64 ud_encode(i64 slot, i64 op)                  ; (slot<<3)|(op&7)
;   i64 ud_op(i64 ud)                                ; ud & 7
;   i64 ud_slot(i64 ud)                              ; ud >> 3
;   i32 recv_action(i64 res, i32 parse_status, i1 body_complete)
;         -> 0 CLOSE, 1 RECV_MORE, 2 DISPATCH        ; pure RECV-edge decision
;   i32 send_action(i64 res, i1 fully_sent, i32 keepalive)
;         -> 0 CLOSE, 1 SEND_MORE, 2 KEEPALIVE_RECV  ; pure SEND-edge decision
;   i32 serve(i32 listenfd, ptr handler, ptr userdata,
;             i64 bufcap, i64 entries, i64 maxreq)
;         handler: void(userdata, req_msg, hdrs, count, resp_spec) — identical
;         to the blocking server's handler ABI. maxreq<0 => run forever.

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

; src/ioring
declare i1  @universe_ioring_ring_available()
declare ptr @universe_ioring_ring_setup(i64, i64)
declare void @universe_ioring_ring_destroy(ptr)
declare i32 @universe_ioring_prep_accept(ptr, i32, ptr, ptr, i32, i64)
declare i32 @universe_ioring_prep_recv(ptr, i32, ptr, i32, i32, i64)
declare i32 @universe_ioring_prep_send(ptr, i32, ptr, i32, i32, i64)
declare i32 @universe_ioring_prep_close(ptr, i32, i64)
declare i32 @universe_ioring_submit_and_wait(ptr, i64)
declare i32 @universe_ioring_peek_cqe(ptr, ptr)
declare i32 @universe_ioring_cqe_seen(ptr)

; src/http parser + serializer (REUSED, not reimplemented)
declare i32 @universe_http_parse_request(ptr, i64, ptr, ptr, i64)
declare i32 @universe_http_write_response(ptr, i64, ptr, i64, ptr, i64, ptr, i64, i32)

; src/io writer (used as an in-memory serialisation sink)
declare ptr @universe_io_writer_create(i32, i64)
declare void @universe_io_writer_destroy(ptr)

; ============================================================ pure helpers

; available: forward the ioring probe so callers choose posix vs uring.
define i1 @universe_http_uring_available() local_unnamed_addr #1 {
entry:
  %r = tail call i1 @universe_ioring_ring_available()
  ret i1 %r
}

; ud_encode(slot, op) -> (slot<<3)|(op&7)
define i64 @universe_http_uring_ud_encode(i64 %slot, i64 %op) local_unnamed_addr #2 {
entry:
  %sh = shl i64 %slot, 3
  %ol = and i64 %op, 7
  %ud = or i64 %sh, %ol
  ret i64 %ud
}

; ud_op(ud) -> ud & 7
define i64 @universe_http_uring_ud_op(i64 %ud) local_unnamed_addr #2 {
entry:
  %op = and i64 %ud, 7
  ret i64 %op
}

; ud_slot(ud) -> ud >> 3
define i64 @universe_http_uring_ud_slot(i64 %ud) local_unnamed_addr #2 {
entry:
  %slot = lshr i64 %ud, 3
  ret i64 %slot
}

; recv_action(res, parse_status, body_complete)
;   res<=0                     -> 0 CLOSE (peer EOF or error)
;   status == 11 INCOMPLETE    -> 1 RECV_MORE
;   status == 0  OK & complete -> 2 DISPATCH
;   status == 0  OK & !complete-> 1 RECV_MORE (need the body)
;   anything else (1,13,...)   -> 0 CLOSE (bad request)
define i32 @universe_http_uring_recv_action(i64 %res, i32 %status, i1 %complete) local_unnamed_addr #2 {
entry:
  %eof = icmp sle i64 %res, 0
  br i1 %eof, label %close, label %chk

chk:
  %inc = icmp eq i32 %status, 11
  br i1 %inc, label %more, label %chkok

chkok:
  %ok = icmp eq i32 %status, 0
  br i1 %ok, label %okpath, label %close

okpath:
  ; complete ? DISPATCH(2) : RECV_MORE(1)
  %r = select i1 %complete, i32 2, i32 1
  ret i32 %r

more:
  ret i32 1

close:
  ret i32 0
}

; send_action(res, fully_sent, keepalive)
;   res<0        -> 0 CLOSE
;   !fully_sent  -> 1 SEND_MORE
;   keepalive    -> 2 KEEPALIVE_RECV  else 0 CLOSE
define i32 @universe_http_uring_send_action(i64 %res, i1 %fully, i32 %keepalive) local_unnamed_addr #2 {
entry:
  %bad = icmp slt i64 %res, 0
  br i1 %bad, label %close, label %chk

chk:
  br i1 %fully, label %done, label %more

done:
  %ka = icmp ne i32 %keepalive, 0
  %r = select i1 %ka, i32 2, i32 0
  ret i32 %r

more:
  ret i32 1

close:
  ret i32 0
}

; ============================================================ event loop

define i32 @universe_http_uring_serve(i32 %listenfd, ptr %handler, ptr %userdata, i64 %bufcap, i64 %entries, i64 %maxreq) local_unnamed_addr #1 {
entry:
  %cqe = alloca [16 x i8], align 8
  %req = alloca [96 x i8], align 8
  %hdrs = alloca [2048 x i8], align 8
  %resp = alloca [56 x i8], align 8
  %servedp = alloca i64, align 8
  %hn = icmp eq ptr %handler, null
  br i1 %hn, label %err.arg, label %norm

err.arg:
  ret i32 8

norm:
  %bc.is0 = icmp eq i64 %bufcap, 0
  %bc = select i1 %bc.is0, i64 65536, i64 %bufcap
  %ns.is0 = icmp eq i64 %entries, 0
  %nslots = select i1 %ns.is0, i64 1, i64 %entries
  %ring = call ptr @universe_ioring_ring_setup(i64 %entries, i64 0)
  %ring.null = icmp eq ptr %ring, null
  br i1 %ring.null, label %err.io, label %alloc.slots

err.io:
  ret i32 15

alloc.slots:
  %slotsz = shl i64 %nslots, 6
  %slots = call ptr @malloc(i64 %slotsz)
  %slots.null = icmp eq ptr %slots, null
  br i1 %slots.null, label %destroy.oom, label %init.slots

destroy.oom:
  call void @universe_ioring_ring_destroy(ptr %ring)
  ret i32 2

init.slots:
  call void @llvm.memset.p0.i64(ptr %slots, i8 0, i64 %slotsz, i1 false)
  store i64 0, ptr %servedp, align 8
  ; prime the pump: one ACCEPT in flight (ud = op ACCEPT(0), slot 0)
  %a0 = call i32 @universe_ioring_prep_accept(ptr %ring, i32 %listenfd, ptr null, ptr null, i32 0, i64 0)
  br label %loop

; ------------------------------------------------------------ turn: submit+wait
loop:
  %saw = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 1)
  %saw.neg = icmp slt i32 %saw, 0
  br i1 %saw.neg, label %saw.err, label %peek

saw.err:
  %eintr = icmp eq i32 %saw, -4
  br i1 %eintr, label %loop, label %shutdown

; ------------------------------------------------------------ drain all CQEs
peek:
  %pr = call i32 @universe_ioring_peek_cqe(ptr %ring, ptr %cqe)
  %empty = icmp ne i32 %pr, 0
  br i1 %empty, label %loop, label %got

got:
  %ud = load i64, ptr %cqe, align 8
  %resp32.p = getelementptr inbounds nuw i8, ptr %cqe, i64 8
  %res32 = load i32, ptr %resp32.p, align 4
  %res = sext i32 %res32 to i64
  %op = and i64 %ud, 7
  %slot = lshr i64 %ud, 3
  switch i64 %op, label %seen [ i64 0, label %do.accept
                                i64 1, label %do.recv
                                i64 2, label %do.send
                                i64 3, label %do.close ]

; ------------------------------------------------------------ ACCEPT
do.accept:
  ; always re-arm the listener
  %ra = call i32 @universe_ioring_prep_accept(ptr %ring, i32 %listenfd, ptr null, ptr null, i32 0, i64 0)
  %acc.bad = icmp slt i64 %res, 0
  br i1 %acc.bad, label %seen, label %find.free

find.free:
  %newfd = trunc i64 %res to i32
  br label %fscan

fscan:
  %fi = phi i64 [ 0, %find.free ], [ %fi.n, %fscan.cont ]
  %fdone = icmp uge i64 %fi, %nslots
  br i1 %fdone, label %no.slot, label %fscan.body

fscan.body:
  %fsp.off = shl i64 %fi, 6
  %fsp = getelementptr inbounds nuw i8, ptr %slots, i64 %fsp.off
  %fstate.p = getelementptr inbounds nuw i8, ptr %fsp, i64 4
  %fstate = load i32, ptr %fstate.p, align 4
  %ffree = icmp eq i32 %fstate, 0
  br i1 %ffree, label %use.slot, label %fscan.cont

fscan.cont:
  %fi.n = add i64 %fi, 1
  br label %fscan

no.slot:
  ; no capacity: close the accepted fd, tracked as an unslotted CLOSE.
  %ud.uc = call i64 @universe_http_uring_ud_encode(i64 %nslots, i64 3)
  %uc = call i32 @universe_ioring_prep_close(ptr %ring, i32 %newfd, i64 %ud.uc)
  br label %seen

use.slot:
  ; lazily allocate recv_buf + send_writer, reuse across conns.
  %us.rbp = getelementptr inbounds nuw i8, ptr %fsp, i64 8
  %us.rb = load ptr, ptr %us.rbp, align 8
  %us.rb.null = icmp eq ptr %us.rb, null
  br i1 %us.rb.null, label %alloc.rb, label %have.rb

alloc.rb:
  %nrb = call ptr @malloc(i64 %bc)
  %nrb.null = icmp eq ptr %nrb, null
  br i1 %nrb.null, label %slot.oom, label %store.rb

store.rb:
  store ptr %nrb, ptr %us.rbp, align 8
  %rcapp0 = getelementptr inbounds nuw i8, ptr %fsp, i64 16
  store i64 %bc, ptr %rcapp0, align 8
  br label %have.rb

have.rb:
  %rb = phi ptr [ %us.rb, %use.slot ], [ %nrb, %store.rb ]
  %us.wp = getelementptr inbounds nuw i8, ptr %fsp, i64 32
  %us.w = load ptr, ptr %us.wp, align 8
  %us.w.null = icmp eq ptr %us.w, null
  br i1 %us.w.null, label %alloc.w, label %have.w

alloc.w:
  %nw = call ptr @universe_io_writer_create(i32 %newfd, i64 %bc)
  %nw.null = icmp eq ptr %nw, null
  br i1 %nw.null, label %slot.oom, label %store.w

store.w:
  store ptr %nw, ptr %us.wp, align 8
  br label %have.w

have.w:
  %w = phi ptr [ %us.w, %have.rb ], [ %nw, %store.w ]
  ; writer fd may be stale from a previous conn; keep it correct (fd@0).
  %newfd64 = zext i32 %newfd to i64
  store i64 %newfd64, ptr %w, align 8
  ; init slot: fd, state=RECV(1), recv_len=0
  store i32 %newfd, ptr %fsp, align 4
  %ustate.p = getelementptr inbounds nuw i8, ptr %fsp, i64 4
  store i32 1, ptr %ustate.p, align 4
  %urlen.p = getelementptr inbounds nuw i8, ptr %fsp, i64 24
  store i64 0, ptr %urlen.p, align 8
  ; arm the first RECV for this connection
  %bc32 = trunc i64 %bc to i32
  %ud.rv = call i64 @universe_http_uring_ud_encode(i64 %fi, i64 1)
  %rvr = call i32 @universe_ioring_prep_recv(ptr %ring, i32 %newfd, ptr %rb, i32 %bc32, i32 0, i64 %ud.rv)
  br label %seen

slot.oom:
  ; couldn't set the connection up; drop it, leave slot FREE.
  %ud.oc = call i64 @universe_http_uring_ud_encode(i64 %nslots, i64 3)
  %ocr = call i32 @universe_ioring_prep_close(ptr %ring, i32 %newfd, i64 %ud.oc)
  br label %seen

; ------------------------------------------------------------ RECV completion
do.recv:
  %rv.off = shl i64 %slot, 6
  %rsp = getelementptr inbounds nuw i8, ptr %slots, i64 %rv.off
  %rfd = load i32, ptr %rsp, align 4
  %rlen.p = getelementptr inbounds nuw i8, ptr %rsp, i64 24
  %rlen0 = load i64, ptr %rlen.p, align 8
  %rbp = getelementptr inbounds nuw i8, ptr %rsp, i64 8
  %rbuf = load ptr, ptr %rbp, align 8
  %rcap.p = getelementptr inbounds nuw i8, ptr %rsp, i64 16
  %rcap = load i64, ptr %rcap.p, align 8
  ; res<=0 short-circuits to close via recv_action, but avoid touching len then
  %rv.eof = icmp sle i64 %res, 0
  br i1 %rv.eof, label %go.close.pre, label %rv.append

rv.append:
  %rlen1 = add i64 %rlen0, %res
  store i64 %rlen1, ptr %rlen.p, align 8
  %pst = call i32 @universe_http_parse_request(ptr %rbuf, i64 %rlen1, ptr %req, ptr %hdrs, i64 64)
  ; complete = (pst==0) && rlen1 >= head_len + body_len
  %pst.ok = icmp eq i32 %pst, 0
  br i1 %pst.ok, label %rv.needcalc, label %rv.decide

rv.needcalc:
  %hl.p = getelementptr inbounds nuw i8, ptr %req, i64 72
  %hl = load i64, ptr %hl.p, align 8
  %bl.p = getelementptr inbounds nuw i8, ptr %req, i64 64
  %bl = load i64, ptr %bl.p, align 8
  %need = add i64 %hl, %bl
  %cmpl = icmp uge i64 %rlen1, %need
  br label %rv.decide

rv.decide:
  %complete = phi i1 [ false, %rv.append ], [ %cmpl, %rv.needcalc ]
  %act = call i32 @universe_http_uring_recv_action(i64 %res, i32 %pst, i1 %complete)
  switch i32 %act, label %go.close [ i32 1, label %rv.more
                                     i32 2, label %rv.dispatch ]

rv.more:
  %space = sub i64 %rcap, %rlen1
  %nospace = icmp eq i64 %space, 0
  br i1 %nospace, label %go.close, label %rv.rearm

rv.rearm:
  %rdst = getelementptr inbounds nuw i8, ptr %rbuf, i64 %rlen1
  %space32 = trunc i64 %space to i32
  %ud.rm = call i64 @universe_http_uring_ud_encode(i64 %slot, i64 1)
  %rmr = call i32 @universe_ioring_prep_recv(ptr %ring, i32 %rfd, ptr %rdst, i32 %space32, i32 0, i64 %ud.rm)
  br label %seen

rv.dispatch:
  ; keep-alive bit from the request flags (bit1)
  %flp = getelementptr inbounds nuw i8, ptr %req, i64 88
  %flags = load i64, ptr %flp, align 8
  %kabit = and i64 %flags, 2
  %ka = icmp ne i64 %kabit, 0
  %ka32 = zext i1 %ka to i32
  %kap = getelementptr inbounds nuw i8, ptr %rsp, i64 56
  store i32 %ka32, ptr %kap, align 4
  %hcp = getelementptr inbounds nuw i8, ptr %req, i64 48
  %hc = load i64, ptr %hcp, align 8
  ; zero the response spec (56 B) so a lazy handler yields a valid empty reply
  call void @llvm.memset.p0.i64(ptr %resp, i8 0, i64 56, i1 false)
  call void %handler(ptr %userdata, ptr %req, ptr %hdrs, i64 %hc, ptr %resp)
  ; load response spec
  %d.status = load i64, ptr %resp, align 8
  %d.reap = getelementptr inbounds nuw i8, ptr %resp, i64 8
  %d.reason = load ptr, ptr %d.reap, align 8
  %d.relp = getelementptr inbounds nuw i8, ptr %resp, i64 16
  %d.rel = load i64, ptr %d.relp, align 8
  %d.rhp = getelementptr inbounds nuw i8, ptr %resp, i64 24
  %d.rh = load ptr, ptr %d.rhp, align 8
  %d.rhcp = getelementptr inbounds nuw i8, ptr %resp, i64 32
  %d.rhc = load i64, ptr %d.rhcp, align 8
  %d.rbp = getelementptr inbounds nuw i8, ptr %resp, i64 40
  %d.rb = load ptr, ptr %d.rbp, align 8
  %d.rblp = getelementptr inbounds nuw i8, ptr %resp, i64 48
  %d.rbl = load i64, ptr %d.rblp, align 8
  ; serialise into the connection's send writer (reset its length first)
  %d.wp = getelementptr inbounds nuw i8, ptr %rsp, i64 32
  %d.w = load ptr, ptr %d.wp, align 8
  %d.wlen.p = getelementptr inbounds nuw i8, ptr %d.w, i64 8
  store i64 0, ptr %d.wlen.p, align 8
  %ws = call i32 @universe_http_write_response(ptr %d.w, i64 %d.status, ptr %d.reason, i64 %d.rel, ptr %d.rh, i64 %d.rhc, ptr %d.rb, i64 %d.rbl, i32 %ka32)
  %d.total = load i64, ptr %d.wlen.p, align 8
  ; slot -> SENDING, send_off=0, send_total=total
  %d.state.p = getelementptr inbounds nuw i8, ptr %rsp, i64 4
  store i32 2, ptr %d.state.p, align 4
  %d.off.p = getelementptr inbounds nuw i8, ptr %rsp, i64 40
  store i64 0, ptr %d.off.p, align 8
  %d.tot.p = getelementptr inbounds nuw i8, ptr %rsp, i64 48
  store i64 %d.total, ptr %d.tot.p, align 8
  %d.sptr = getelementptr inbounds nuw i8, ptr %d.w, i64 64
  %d.total32 = trunc i64 %d.total to i32
  %ud.sd = call i64 @universe_http_uring_ud_encode(i64 %slot, i64 2)
  %sdr = call i32 @universe_ioring_prep_send(ptr %ring, i32 %rfd, ptr %d.sptr, i32 %d.total32, i32 0, i64 %ud.sd)
  br label %seen

; ------------------------------------------------------------ SEND completion
do.send:
  %sd.off0 = shl i64 %slot, 6
  %ssp = getelementptr inbounds nuw i8, ptr %slots, i64 %sd.off0
  %sfd = load i32, ptr %ssp, align 4
  %sd.bad = icmp slt i64 %res, 0
  br i1 %sd.bad, label %go.close, label %sd.adv

sd.adv:
  %soff.p = getelementptr inbounds nuw i8, ptr %ssp, i64 40
  %soff0 = load i64, ptr %soff.p, align 8
  %soff1 = add i64 %soff0, %res
  %stot.p = getelementptr inbounds nuw i8, ptr %ssp, i64 48
  %stot = load i64, ptr %stot.p, align 8
  %fully = icmp uge i64 %soff1, %stot
  %skap = getelementptr inbounds nuw i8, ptr %ssp, i64 56
  %ska = load i32, ptr %skap, align 4
  %sact = call i32 @universe_http_uring_send_action(i64 %res, i1 %fully, i32 %ska)
  switch i32 %sact, label %sd.count [ i32 1, label %sd.more ]

sd.more:
  store i64 %soff1, ptr %soff.p, align 8
  %swp = getelementptr inbounds nuw i8, ptr %ssp, i64 32
  %sw = load ptr, ptr %swp, align 8
  %sbase = getelementptr inbounds nuw i8, ptr %sw, i64 64
  %sptr2 = getelementptr inbounds nuw i8, ptr %sbase, i64 %soff1
  %srem = sub i64 %stot, %soff1
  %srem32 = trunc i64 %srem to i32
  %ud.sm = call i64 @universe_http_uring_ud_encode(i64 %slot, i64 2)
  %smr = call i32 @universe_ioring_prep_send(ptr %ring, i32 %sfd, ptr %sptr2, i32 %srem32, i32 0, i64 %ud.sm)
  br label %seen

sd.count:
  ; a full response went out: count one served request, then close or recycle.
  %sv0 = load i64, ptr %servedp, align 8
  %sv1 = add i64 %sv0, 1
  store i64 %sv1, ptr %servedp, align 8
  %recycle = icmp eq i32 %sact, 2
  br i1 %recycle, label %sd.keepalive, label %go.close

sd.keepalive:
  ; reset the connection for the next pipelined/keep-alive request
  %ka.rlen.p = getelementptr inbounds nuw i8, ptr %ssp, i64 24
  store i64 0, ptr %ka.rlen.p, align 8
  %ka.off.p = getelementptr inbounds nuw i8, ptr %ssp, i64 40
  store i64 0, ptr %ka.off.p, align 8
  %ka.state.p = getelementptr inbounds nuw i8, ptr %ssp, i64 4
  store i32 1, ptr %ka.state.p, align 4
  %ka.rbp = getelementptr inbounds nuw i8, ptr %ssp, i64 8
  %ka.rb = load ptr, ptr %ka.rbp, align 8
  %ka.rcap.p = getelementptr inbounds nuw i8, ptr %ssp, i64 16
  %ka.rcap = load i64, ptr %ka.rcap.p, align 8
  %ka.rcap32 = trunc i64 %ka.rcap to i32
  %ud.ka = call i64 @universe_http_uring_ud_encode(i64 %slot, i64 1)
  %kar = call i32 @universe_ioring_prep_recv(ptr %ring, i32 %sfd, ptr %ka.rb, i32 %ka.rcap32, i32 0, i64 %ud.ka)
  br label %seen

; ------------------------------------------------------------ close paths
go.close.pre:
  br label %go.close

go.close:
  ; slot is valid here (RECV/SEND edges); mark CLOSING and queue prep_close.
  %gc.off = shl i64 %slot, 6
  %gcp = getelementptr inbounds nuw i8, ptr %slots, i64 %gc.off
  %gcfd = load i32, ptr %gcp, align 4
  %gc.state.p = getelementptr inbounds nuw i8, ptr %gcp, i64 4
  store i32 3, ptr %gc.state.p, align 4
  %ud.gc = call i64 @universe_http_uring_ud_encode(i64 %slot, i64 3)
  %gcr = call i32 @universe_ioring_prep_close(ptr %ring, i32 %gcfd, i64 %ud.gc)
  br label %seen

do.close:
  ; free the slot for reuse (buffers kept). Unslotted closes (slot>=nslots) noop.
  %c.valid = icmp ult i64 %slot, %nslots
  br i1 %c.valid, label %do.close.slot, label %seen

do.close.slot:
  %c.off = shl i64 %slot, 6
  %ccp = getelementptr inbounds nuw i8, ptr %slots, i64 %c.off
  %c.state.p = getelementptr inbounds nuw i8, ptr %ccp, i64 4
  store i32 0, ptr %c.state.p, align 4
  br label %seen

; ------------------------------------------------------------ per-CQE tail
seen:
  %cs = call i32 @universe_ioring_cqe_seen(ptr %ring)
  ; stop after maxreq served requests (maxreq<0 => forever)
  %unbounded = icmp slt i64 %maxreq, 0
  br i1 %unbounded, label %peek, label %chk.limit

chk.limit:
  %sv = load i64, ptr %servedp, align 8
  %reached = icmp uge i64 %sv, %maxreq
  br i1 %reached, label %shutdown, label %peek

; ------------------------------------------------------------ teardown
shutdown:
  br label %sd.free

sd.free:
  %ti = phi i64 [ 0, %shutdown ], [ %ti.n, %sd.next ]
  %tdone = icmp uge i64 %ti, %nslots
  br i1 %tdone, label %sd.done, label %sd.free.body

sd.free.body:
  %tsp.off = shl i64 %ti, 6
  %tsp = getelementptr inbounds nuw i8, ptr %slots, i64 %tsp.off
  %t.state.p = getelementptr inbounds nuw i8, ptr %tsp, i64 4
  %t.state = load i32, ptr %t.state.p, align 4
  %t.busy = icmp ne i32 %t.state, 0
  %t.rbp = getelementptr inbounds nuw i8, ptr %tsp, i64 8
  %t.rb = load ptr, ptr %t.rbp, align 8
  %t.rb.some = icmp ne ptr %t.rb, null
  br i1 %t.rb.some, label %sd.free.rb, label %sd.free.w

sd.free.rb:
  call void @free(ptr %t.rb)
  br label %sd.free.w

sd.free.w:
  %t.wp = getelementptr inbounds nuw i8, ptr %tsp, i64 32
  %t.w = load ptr, ptr %t.wp, align 8
  %t.w.some = icmp ne ptr %t.w, null
  br i1 %t.w.some, label %sd.free.wd, label %sd.free.cont

sd.free.wd:
  call void @universe_io_writer_destroy(ptr %t.w)
  br label %sd.free.cont

sd.free.cont:
  ; leak-free teardown of any still-open conn fd via one more close SQE
  br i1 %t.busy, label %sd.close.fd, label %sd.next

sd.close.fd:
  %t.fd = load i32, ptr %tsp, align 4
  %t.udc = call i64 @universe_http_uring_ud_encode(i64 %nslots, i64 3)
  %t.cr = call i32 @universe_ioring_prep_close(ptr %ring, i32 %t.fd, i64 %t.udc)
  br label %sd.next

sd.next:
  %ti.n = add i64 %ti, 1
  br label %sd.free

sd.done:
  ; best-effort flush of the teardown close SQEs, then tear down the ring
  %fin = call i32 @universe_ioring_submit_and_wait(ptr %ring, i64 0)
  call void @universe_ioring_ring_destroy(ptr %ring)
  call void @free(ptr %slots)
  ret i32 0
}

attributes #1 = { nounwind }
attributes #2 = { alwaysinline nounwind willreturn memory(none) }
