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

; Tests for src/http/http_uring.ll.
;   * PURE LOGIC (runs everywhere, incl. macOS): user_data encode/decode round
;     trips and the RECV/SEND state-machine decision functions. This is real
;     coverage of the module even when io_uring is unavailable.
;   * LIVE SERVER (LINUX only): if ring_available() is false we print a skip
;     line and return ut_summary() (a CLEAN skip — zero failures, NOT a fake
;     pass). When available we bring the uring server up in a pthread, drive it
;     with the EXISTING posix http client (GET keep-alive + POST echo), and
;     assert status + body, then let the server self-shutdown after maxreq.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare i32 @ut_summary()

declare i1 @universe_simd_equal(ptr, ptr, i64)

; module under test
declare i1 @universe_http_uring_available()
declare i64 @universe_http_uring_ud_encode(i64, i64)
declare i64 @universe_http_uring_ud_op(i64)
declare i64 @universe_http_uring_ud_slot(i64)
declare i32 @universe_http_uring_recv_action(i64, i32, i1)
declare i32 @universe_http_uring_send_action(i64, i1, i32)
declare i32 @universe_http_uring_serve(i32, ptr, ptr, i64, i64, i64)

; existing posix http client (drives the server over loopback)
declare ptr @universe_http_connect(i32, i32, i64)
declare void @universe_http_conn_destroy(ptr)
declare i32 @universe_http_client_request(ptr, ptr, i64, ptr, i64, i64, ptr, i64, ptr, i64, i32, ptr, ptr, i64)

declare i32 @universe_net_tcp_listen(i32, i32, i32)
declare i32 @universe_net_tcp_close(i32)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(ptr, ptr)
declare i32 @printf(ptr, ...)

; ---------------------------------------------------------------- string data
@s.get   = private unnamed_addr constant [3 x i8] c"GET"
@s.slash = private unnamed_addr constant [1 x i8] c"/"
@s.post  = private unnamed_addr constant [4 x i8] c"POST"
@s.echo  = private unnamed_addr constant [5 x i8] c"/echo"
@s.ping  = private unnamed_addr constant [4 x i8] c"ping"
@s.pong  = private unnamed_addr constant [4 x i8] c"pong"
@s.ok    = private unnamed_addr constant [2 x i8] c"OK"

@m.skip  = private unnamed_addr constant [45 x i8] c"SKIP: io_uring unavailable (non-Linux host)\0A\00"
@m.udenc = private unnamed_addr constant [18 x i8] c"ud_encode/op/slot\00"
@m.udop  = private unnamed_addr constant [6 x i8] c"ud_op\00"
@m.udsl  = private unnamed_addr constant [8 x i8] c"ud_slot\00"
@m.rac   = private unnamed_addr constant [12 x i8] c"recv_action\00"
@m.sac   = private unnamed_addr constant [12 x i8] c"send_action\00"
@m.st1   = private unnamed_addr constant [17 x i8] c"live GET status\0A\00"
@m.st1b  = private unnamed_addr constant [14 x i8] c"live GET body\00"
@m.st2   = private unnamed_addr constant [17 x i8] c"live POST status\00"
@m.st2b  = private unnamed_addr constant [15 x i8] c"live POST echo\00"
@m.conn  = private unnamed_addr constant [16 x i8] c"client connect\0A\00"

; ================================================================ handler
; void echo_handler(userdata, req, hdrs, count, resp): GET -> "pong"; else echo.
define void @echo_handler(ptr %ud, ptr %req, ptr %hdrs, i64 %count, ptr %resp) {
entry:
  %c = load i64, ptr %ud, align 8
  %c.n = add i64 %c, 1
  store i64 %c.n, ptr %ud, align 8
  %mp = load ptr, ptr %req, align 8
  %mlp = getelementptr inbounds nuw i8, ptr %req, i64 8
  %ml = load i64, ptr %mlp, align 8
  %len3 = icmp eq i64 %ml, 3
  br i1 %len3, label %cmp, label %echo

cmp:
  %eq = call i1 @universe_simd_equal(ptr %mp, ptr @s.get, i64 3)
  br i1 %eq, label %getc, label %echo

getc:
  br label %setresp

echo:
  %bp = getelementptr inbounds nuw i8, ptr %req, i64 56
  %body = load ptr, ptr %bp, align 8
  %blp = getelementptr inbounds nuw i8, ptr %req, i64 64
  %blen = load i64, ptr %blp, align 8
  br label %setresp

setresp:
  %rbody = phi ptr [ @s.pong, %getc ], [ %body, %echo ]
  %rblen = phi i64 [ 4, %getc ], [ %blen, %echo ]
  store i64 200, ptr %resp, align 8
  %r8 = getelementptr inbounds nuw i8, ptr %resp, i64 8
  store ptr @s.ok, ptr %r8, align 8
  %r16 = getelementptr inbounds nuw i8, ptr %resp, i64 16
  store i64 2, ptr %r16, align 8
  %r24 = getelementptr inbounds nuw i8, ptr %resp, i64 24
  store ptr null, ptr %r24, align 8
  %r32 = getelementptr inbounds nuw i8, ptr %resp, i64 32
  store i64 0, ptr %r32, align 8
  %r40 = getelementptr inbounds nuw i8, ptr %resp, i64 40
  store ptr %rbody, ptr %r40, align 8
  %r48 = getelementptr inbounds nuw i8, ptr %resp, i64 48
  store i64 %rblen, ptr %r48, align 8
  ret void
}

; void* server_thread(arg): arg = {i32 lf@0, ptr ud@8, i64 bufcap@16,
;                                  i64 entries@24, i64 maxreq@32}
define ptr @server_thread(ptr %arg) {
entry:
  %lf = load i32, ptr %arg, align 4
  %udp = getelementptr inbounds nuw i8, ptr %arg, i64 8
  %ud = load ptr, ptr %udp, align 8
  %bcp = getelementptr inbounds nuw i8, ptr %arg, i64 16
  %bc = load i64, ptr %bcp, align 8
  %enp = getelementptr inbounds nuw i8, ptr %arg, i64 24
  %en = load i64, ptr %enp, align 8
  %mrp = getelementptr inbounds nuw i8, ptr %arg, i64 32
  %mr = load i64, ptr %mrp, align 8
  %r = call i32 @universe_http_uring_serve(i32 %lf, ptr @echo_handler, ptr %ud, i64 %bc, i64 %en, i64 %mr)
  ret ptr null
}

; check [gp,gl) == [ep,el)
define void @check_slice(ptr %gp, i64 %gl, ptr %ep, i64 %el, ptr %m) {
entry:
  %leneq = icmp eq i64 %gl, %el
  br i1 %leneq, label %cmp, label %fail
cmp:
  %eq = call i1 @universe_simd_equal(ptr %gp, ptr %ep, i64 %el)
  call void @ut_check(i1 %eq, ptr %m)
  ret void
fail:
  call void @ut_check(i1 false, ptr %m)
  ret void
}

; ================================================================ main
define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- pure user_data round trips ----
  %e0 = call i64 @universe_http_uring_ud_encode(i64 5, i64 2)
  %e0ok = icmp eq i64 %e0, 42
  call void @ut_check(i1 %e0ok, ptr @m.udenc)
  %op0 = call i64 @universe_http_uring_ud_op(i64 %e0)
  call void @ut_check_eq(i64 %op0, i64 2, ptr @m.udop)
  %sl0 = call i64 @universe_http_uring_ud_slot(i64 %e0)
  call void @ut_check_eq(i64 %sl0, i64 5, ptr @m.udsl)
  %e1 = call i64 @universe_http_uring_ud_encode(i64 100, i64 3)
  %op1 = call i64 @universe_http_uring_ud_op(i64 %e1)
  call void @ut_check_eq(i64 %op1, i64 3, ptr @m.udop)
  %sl1 = call i64 @universe_http_uring_ud_slot(i64 %e1)
  call void @ut_check_eq(i64 %sl1, i64 100, ptr @m.udsl)

  ; ---- recv_action truth table ----
  ; res<=0 => 0 CLOSE
  %ra0 = call i32 @universe_http_uring_recv_action(i64 0, i32 0, i1 true)
  %ra0v = zext i32 %ra0 to i64
  call void @ut_check_eq(i64 %ra0v, i64 0, ptr @m.rac)
  %ra1 = call i32 @universe_http_uring_recv_action(i64 -1, i32 11, i1 false)
  %ra1v = zext i32 %ra1 to i64
  call void @ut_check_eq(i64 %ra1v, i64 0, ptr @m.rac)
  ; status 11 INCOMPLETE => 1 RECV_MORE
  %ra2 = call i32 @universe_http_uring_recv_action(i64 10, i32 11, i1 false)
  %ra2v = zext i32 %ra2 to i64
  call void @ut_check_eq(i64 %ra2v, i64 1, ptr @m.rac)
  ; ok + complete => 2 DISPATCH
  %ra3 = call i32 @universe_http_uring_recv_action(i64 10, i32 0, i1 true)
  %ra3v = zext i32 %ra3 to i64
  call void @ut_check_eq(i64 %ra3v, i64 2, ptr @m.rac)
  ; ok + not complete => 1 RECV_MORE (need body)
  %ra4 = call i32 @universe_http_uring_recv_action(i64 10, i32 0, i1 false)
  %ra4v = zext i32 %ra4 to i64
  call void @ut_check_eq(i64 %ra4v, i64 1, ptr @m.rac)
  ; parse error 13 => 0 CLOSE
  %ra5 = call i32 @universe_http_uring_recv_action(i64 10, i32 13, i1 true)
  %ra5v = zext i32 %ra5 to i64
  call void @ut_check_eq(i64 %ra5v, i64 0, ptr @m.rac)

  ; ---- send_action truth table ----
  %sa0 = call i32 @universe_http_uring_send_action(i64 -1, i1 true, i32 1)
  %sa0v = zext i32 %sa0 to i64
  call void @ut_check_eq(i64 %sa0v, i64 0, ptr @m.sac)
  %sa1 = call i32 @universe_http_uring_send_action(i64 5, i1 false, i32 1)
  %sa1v = zext i32 %sa1 to i64
  call void @ut_check_eq(i64 %sa1v, i64 1, ptr @m.sac)
  %sa2 = call i32 @universe_http_uring_send_action(i64 5, i1 true, i32 1)
  %sa2v = zext i32 %sa2 to i64
  call void @ut_check_eq(i64 %sa2v, i64 2, ptr @m.sac)
  %sa3 = call i32 @universe_http_uring_send_action(i64 5, i1 true, i32 0)
  %sa3v = zext i32 %sa3 to i64
  call void @ut_check_eq(i64 %sa3v, i64 0, ptr @m.sac)

  ; ---- live server gate ----
  %avail = call i1 @universe_http_uring_available()
  br i1 %avail, label %live, label %skip

skip:
  %ps = call i32 (ptr, ...) @printf(ptr @m.skip)
  br label %fin

; ------------------------------------------------------------ live (Linux)
live:
  %counter = alloca i64, align 8
  store i64 0, ptr %counter, align 8
  %targ = alloca [40 x i8], align 8
  %rmsg = alloca [96 x i8], align 8
  %rhdrs = alloca [2048 x i8], align 8
  %th = alloca i64, align 8

  ; listen on 127.0.0.1:18090
  %lf = call i32 @universe_net_tcp_listen(i32 2130706433, i32 18090, i32 64)
  %lfbad = icmp slt i32 %lf, 0
  br i1 %lfbad, label %fin, label %spawn

spawn:
  store i32 %lf, ptr %targ, align 4
  %tud = getelementptr inbounds nuw i8, ptr %targ, i64 8
  store ptr %counter, ptr %tud, align 8
  %tbc = getelementptr inbounds nuw i8, ptr %targ, i64 16
  store i64 65536, ptr %tbc, align 8
  %ten = getelementptr inbounds nuw i8, ptr %targ, i64 24
  store i64 32, ptr %ten, align 8
  %tmr = getelementptr inbounds nuw i8, ptr %targ, i64 32
  store i64 2, ptr %tmr, align 8      ; serve exactly 2 requests then shutdown
  %pc = call i32 @pthread_create(ptr %th, ptr null, ptr @server_thread, ptr %targ)

  ; connect (keep-alive) and issue request 1: GET / -> "pong"
  %conn = call ptr @universe_http_connect(i32 2130706433, i32 18090, i64 0)
  %cnull = icmp eq ptr %conn, null
  br i1 %cnull, label %joinbad, label %req1

req1:
  %st1 = call i32 @universe_http_client_request(ptr %conn, ptr @s.get, i64 3, ptr @s.slash, i64 1, i64 1, ptr null, i64 0, ptr null, i64 0, i32 1, ptr %rmsg, ptr %rhdrs, i64 64)
  %st1ok = icmp eq i32 %st1, 0
  call void @ut_check(i1 %st1ok, ptr @m.st1)
  %code1p = getelementptr inbounds nuw i8, ptr %rmsg, i64 40
  %code1 = load i64, ptr %code1p, align 8
  call void @ut_check_eq(i64 %code1, i64 200, ptr @m.st1)
  %b1p = getelementptr inbounds nuw i8, ptr %rmsg, i64 56
  %b1 = load ptr, ptr %b1p, align 8
  %b1lp = getelementptr inbounds nuw i8, ptr %rmsg, i64 64
  %b1l = load i64, ptr %b1lp, align 8
  call void @check_slice(ptr %b1, i64 %b1l, ptr @s.pong, i64 4, ptr @m.st1b)

  ; request 2 on the SAME keep-alive connection: POST /echo "ping" -> "ping"
  %st2 = call i32 @universe_http_client_request(ptr %conn, ptr @s.post, i64 4, ptr @s.echo, i64 5, i64 1, ptr null, i64 0, ptr @s.ping, i64 4, i32 1, ptr %rmsg, ptr %rhdrs, i64 64)
  %st2ok = icmp eq i32 %st2, 0
  call void @ut_check(i1 %st2ok, ptr @m.st2)
  %code2p = getelementptr inbounds nuw i8, ptr %rmsg, i64 40
  %code2 = load i64, ptr %code2p, align 8
  call void @ut_check_eq(i64 %code2, i64 200, ptr @m.st2)
  %b2p = getelementptr inbounds nuw i8, ptr %rmsg, i64 56
  %b2 = load ptr, ptr %b2p, align 8
  %b2lp = getelementptr inbounds nuw i8, ptr %rmsg, i64 64
  %b2l = load i64, ptr %b2lp, align 8
  call void @check_slice(ptr %b2, i64 %b2l, ptr @s.ping, i64 4, ptr @m.st2b)

  call void @universe_http_conn_destroy(ptr %conn)
  br label %join

joinbad:
  call void @ut_check(i1 false, ptr @m.conn)
  br label %join

join:
  %pj = call i32 @pthread_join(ptr %th, ptr null)
  ; handler must have run exactly twice
  %cnt = load i64, ptr %counter, align 8
  call void @ut_check_eq(i64 %cnt, i64 2, ptr @m.st2b)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
