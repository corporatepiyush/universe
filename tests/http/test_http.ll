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

; Tests for src/http/http.ll: static parse (request/response, case-insensitive
; header lookup, malformed), chunked decode + pipe encode round-trip, and a full
; loopback round-trip (GET/POST, keep-alive, Connection: close) against a server
; running in a pthread. --bench times keep-alive round-trips over loopback.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

declare i1 @universe_simd_equal(ptr, ptr, i64)

declare i64 @universe_http_find_head_end(ptr, i64)
declare i32 @universe_http_parse_request(ptr, i64, ptr, ptr, i64)
declare i32 @universe_http_parse_response(ptr, i64, ptr, ptr, i64)
declare i32 @universe_http_header_get(ptr, i64, ptr, i64, ptr, ptr)
declare i32 @universe_http_chunked_decode(ptr, i64, ptr, i64, ptr)
declare i32 @universe_http_chunked_encode(ptr, ptr, i64)
declare ptr @universe_http_connect(i32, i32, i64)
declare void @universe_http_conn_destroy(ptr)
declare i32 @universe_http_client_request(ptr, ptr, i64, ptr, i64, i64, ptr, i64, ptr, i64, i32, ptr, ptr, i64)
declare i32 @universe_http_accept_loop(i32, ptr, ptr, i64, i64)

declare ptr @universe_io_writer_create(i32, i64)
declare void @universe_io_writer_destroy(ptr)
declare i32 @universe_io_writer_flush(ptr)

declare i32 @universe_net_tcp_listen(i32, i32, i32)
declare i32 @universe_net_tcp_close(i32)

declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @pipe(ptr)
declare i64 @read(i32, ptr, i64)
declare i32 @close(i32)
declare i32 @printf(ptr, ...)

; ---------------------------------------------------------------- test data
@rq = private unnamed_addr constant [87 x i8] c"GET /index.html?q=1 HTTP/1.1\0D\0AHost: example.com\0D\0AContent-Length: 5\0D\0AX-Test: hi\0D\0A\0D\0Ahello"
@rp = private unnamed_addr constant [67 x i8] c"HTTP/1.1 404 Not Found\0D\0AContent-Length: 3\0D\0AConnection: close\0D\0A\0D\0Aabc"
@rq_inc = private unnamed_addr constant [25 x i8] c"GET / HTTP/1.1\0D\0AHost: x\0D\0A"
@rq_bad = private unnamed_addr constant [16 x i8] c"GET/HTTP/1.1\0D\0A\0D\0A"
@rq_badver = private unnamed_addr constant [18 x i8] c"GET / HTTP/2.0\0D\0A\0D\0A"
@chk = private unnamed_addr constant [24 x i8] c"4\0D\0AWiki\0D\0A5\0D\0Apedia\0D\0A0\0D\0A\0D\0A"

@x.get   = private unnamed_addr constant [3 x i8]  c"GET"
@x.tgt   = private unnamed_addr constant [15 x i8] c"/index.html?q=1"
@x.hello = private unnamed_addr constant [5 x i8]  c"hello"
@x.reason= private unnamed_addr constant [9 x i8]  c"Not Found"
@x.abc   = private unnamed_addr constant [3 x i8]  c"abc"
@x.wiki  = private unnamed_addr constant [9 x i8]  c"Wikipedia"

@n.host  = private unnamed_addr constant [4 x i8]  c"host"
@n.cl    = private unnamed_addr constant [14 x i8] c"Content-Length"
@n.xt    = private unnamed_addr constant [6 x i8]  c"X-TEST"
@n.miss  = private unnamed_addr constant [7 x i8]  c"Missing"
@v.example = private unnamed_addr constant [11 x i8] c"example.com"
@v.five  = private unnamed_addr constant [1 x i8]  c"5"
@v.hi    = private unnamed_addr constant [2 x i8]  c"hi"

@s.get   = private unnamed_addr constant [3 x i8]  c"GET"
@s.slash = private unnamed_addr constant [1 x i8]  c"/"
@s.post  = private unnamed_addr constant [4 x i8]  c"POST"
@s.echo  = private unnamed_addr constant [5 x i8]  c"/echo"
@s.ping  = private unnamed_addr constant [4 x i8]  c"ping"
@s.pong  = private unnamed_addr constant [4 x i8]  c"pong"
@s.ok    = private unnamed_addr constant [2 x i8]  c"OK"

; ---------------------------------------------------------------- messages
@m.rq_ok   = private unnamed_addr constant [16 x i8] c"parse_request ok"
@m.method  = private unnamed_addr constant [11 x i8] c"req method\00"
@m.target  = private unnamed_addr constant [11 x i8] c"req target\00"
@m.minor   = private unnamed_addr constant [10 x i8] c"req minor\00"
@m.hcount  = private unnamed_addr constant [11 x i8] c"req hcount\00"
@m.cl      = private unnamed_addr constant [8 x i8]  c"req cl\0A\00"
@m.body    = private unnamed_addr constant [9 x i8]  c"req body\00"
@m.hg_host = private unnamed_addr constant [9 x i8]  c"hg host\0A\00"
@m.hg_cl   = private unnamed_addr constant [7 x i8]  c"hg cl\0A\00"
@m.hg_xt   = private unnamed_addr constant [7 x i8]  c"hg xt\0A\00"
@m.hg_miss = private unnamed_addr constant [9 x i8]  c"hg miss\0A\00"
@m.rp_ok   = private unnamed_addr constant [13 x i8] c"parse_resp ok"
@m.code    = private unnamed_addr constant [9 x i8]  c"rp code\0A\00"
@m.reason  = private unnamed_addr constant [10 x i8] c"rp reason\00"
@m.rp_cl   = private unnamed_addr constant [7 x i8]  c"rp cl\0A\00"
@m.rp_body = private unnamed_addr constant [8 x i8]  c"rp body\00"
@m.rp_ka   = private unnamed_addr constant [12 x i8] c"rp closeflag"
@m.inc     = private unnamed_addr constant [12 x i8] c"incomplete\0A\00"
@m.bad     = private unnamed_addr constant [9 x i8]  c"bad line\00"
@m.badver  = private unnamed_addr constant [8 x i8]  c"bad ver\00"
@m.chk_ok  = private unnamed_addr constant [12 x i8] c"chunk decode"
@m.chk_len = private unnamed_addr constant [11 x i8] c"chunk len\0A\00"
@m.enc_ok  = private unnamed_addr constant [12 x i8] c"chunk encode"
@m.listen  = private unnamed_addr constant [10 x i8] c"listen ok\00"
@m.conn    = private unnamed_addr constant [8 x i8]  c"connect\00"
@m.rt_st   = private unnamed_addr constant [11 x i8] c"rt status\0A\00"
@m.rt_code = private unnamed_addr constant [9 x i8]  c"rt code\0A\00"
@m.rt_body = private unnamed_addr constant [8 x i8]  c"rt body\00"
@m.rt_cnt  = private unnamed_addr constant [12 x i8] c"rt reqcount\00"
@http.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.http  = private unnamed_addr constant [22 x i8] c"http keepalive rt 20k\00"

; ================================================================ handler
; void echo_handler(userdata, req, hdrs, count, resp)
;   userdata = ptr to i64 request counter. GET -> body "pong"; else echo body.
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

; void* server_thread(arg): arg = {i32 listenfd@0, ptr ud@8, i64 maxconns@16, i64 bufcap@24}
define ptr @server_thread(ptr %arg) {
entry:
  %lf = load i32, ptr %arg, align 4
  %udp = getelementptr inbounds nuw i8, ptr %arg, i64 8
  %ud = load ptr, ptr %udp, align 8
  %mcp = getelementptr inbounds nuw i8, ptr %arg, i64 16
  %mc = load i64, ptr %mcp, align 8
  %bcp = getelementptr inbounds nuw i8, ptr %arg, i64 24
  %bc = load i64, ptr %bcp, align 8
  %r = call i32 @universe_http_accept_loop(i32 %lf, ptr @echo_handler, ptr %ud, i64 %bc, i64 %mc)
  ret ptr null
}

; check that [gp,gl) == [ep,el)
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

define i64 @ld64(ptr %base, i64 %off) {
entry:
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define ptr @ldp(ptr %base, i64 %off) {
entry:
  %p = getelementptr inbounds nuw i8, ptr %base, i64 %off
  %v = load ptr, ptr %p, align 8
  ret ptr %v
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %msg = alloca [96 x i8], align 8
  %hdrs = alloca [2048 x i8], align 8
  %valp = alloca ptr, align 8
  %vall = alloca i64, align 8
  %decbuf = alloca [64 x i8], align 1
  %declen = alloca i64, align 8
  %pipefds = alloca [2 x i32], align 4
  %rdbuf = alloca [256 x i8], align 1
  %counter = alloca i64, align 8
  %targ = alloca [32 x i8], align 8
  %th = alloca ptr, align 8
  ; response structs for round-trip
  %rmsg = alloca [96 x i8], align 8
  %rhdrs = alloca [2048 x i8], align 8

  ; ---------------- parse_request ----------------
  %prq = call i32 @universe_http_parse_request(ptr @rq, i64 83, ptr %msg, ptr %hdrs, i64 64)
  %prq.ok = icmp eq i32 %prq, 0
  call void @ut_check(i1 %prq.ok, ptr @m.rq_ok)
  %mp = load ptr, ptr %msg, align 8
  %ml = call i64 @ld64(ptr %msg, i64 8)
  call void @check_slice(ptr %mp, i64 %ml, ptr @x.get, i64 3, ptr @m.method)
  %tp = call ptr @ldp(ptr %msg, i64 16)
  %tl = call i64 @ld64(ptr %msg, i64 24)
  call void @check_slice(ptr %tp, i64 %tl, ptr @x.tgt, i64 15, ptr @m.target)
  %minor = call i64 @ld64(ptr %msg, i64 32)
  call void @ut_check_eq(i64 %minor, i64 1, ptr @m.minor)
  %hcount = call i64 @ld64(ptr %msg, i64 48)
  call void @ut_check_eq(i64 %hcount, i64 3, ptr @m.hcount)
  %clv = call i64 @ld64(ptr %msg, i64 80)
  call void @ut_check_eq(i64 %clv, i64 5, ptr @m.cl)
  %bp = call ptr @ldp(ptr %msg, i64 56)
  %bl = call i64 @ld64(ptr %msg, i64 64)
  call void @check_slice(ptr %bp, i64 %bl, ptr @x.hello, i64 5, ptr @m.body)

  ; case-insensitive header lookup
  %hg1 = call i32 @universe_http_header_get(ptr %hdrs, i64 3, ptr @n.host, i64 4, ptr %valp, ptr %vall)
  %hg1.ok = icmp eq i32 %hg1, 0
  call void @ut_check(i1 %hg1.ok, ptr @m.hg_host)
  %h1p = load ptr, ptr %valp, align 8
  %h1l = load i64, ptr %vall, align 8
  call void @check_slice(ptr %h1p, i64 %h1l, ptr @v.example, i64 11, ptr @m.hg_host)
  %hg2 = call i32 @universe_http_header_get(ptr %hdrs, i64 3, ptr @n.cl, i64 14, ptr %valp, ptr %vall)
  %h2p = load ptr, ptr %valp, align 8
  %h2l = load i64, ptr %vall, align 8
  call void @check_slice(ptr %h2p, i64 %h2l, ptr @v.five, i64 1, ptr @m.hg_cl)
  %hg3 = call i32 @universe_http_header_get(ptr %hdrs, i64 3, ptr @n.xt, i64 6, ptr %valp, ptr %vall)
  %h3p = load ptr, ptr %valp, align 8
  %h3l = load i64, ptr %vall, align 8
  call void @check_slice(ptr %h3p, i64 %h3l, ptr @v.hi, i64 2, ptr @m.hg_xt)
  %hg4 = call i32 @universe_http_header_get(ptr %hdrs, i64 3, ptr @n.miss, i64 7, ptr %valp, ptr %vall)
  %hg4.nf = icmp eq i32 %hg4, 5
  call void @ut_check(i1 %hg4.nf, ptr @m.hg_miss)

  ; ---------------- parse_response ----------------
  %prp = call i32 @universe_http_parse_response(ptr @rp, i64 65, ptr %msg, ptr %hdrs, i64 64)
  %prp.ok = icmp eq i32 %prp, 0
  call void @ut_check(i1 %prp.ok, ptr @m.rp_ok)
  %code = call i64 @ld64(ptr %msg, i64 40)
  call void @ut_check_eq(i64 %code, i64 404, ptr @m.code)
  %rsp = load ptr, ptr %msg, align 8
  %rsl = call i64 @ld64(ptr %msg, i64 8)
  call void @check_slice(ptr %rsp, i64 %rsl, ptr @x.reason, i64 9, ptr @m.reason)
  %rcl = call i64 @ld64(ptr %msg, i64 80)
  call void @ut_check_eq(i64 %rcl, i64 3, ptr @m.rp_cl)
  %rbp = call ptr @ldp(ptr %msg, i64 56)
  %rbl = call i64 @ld64(ptr %msg, i64 64)
  call void @check_slice(ptr %rbp, i64 %rbl, ptr @x.abc, i64 3, ptr @m.rp_body)
  ; Connection: close -> keep-alive flag (bit1) must be 0
  %rflags = call i64 @ld64(ptr %msg, i64 88)
  %rka = and i64 %rflags, 2
  %rka0 = icmp eq i64 %rka, 0
  call void @ut_check(i1 %rka0, ptr @m.rp_ka)

  ; ---------------- malformed ----------------
  %inc = call i32 @universe_http_parse_request(ptr @rq_inc, i64 24, ptr %msg, ptr %hdrs, i64 64)
  %inc.ok = icmp eq i32 %inc, 11
  call void @ut_check(i1 %inc.ok, ptr @m.inc)
  %bad = call i32 @universe_http_parse_request(ptr @rq_bad, i64 16, ptr %msg, ptr %hdrs, i64 64)
  %bad.ok = icmp eq i32 %bad, 13
  call void @ut_check(i1 %bad.ok, ptr @m.bad)
  %bv = call i32 @universe_http_parse_request(ptr @rq_badver, i64 18, ptr %msg, ptr %hdrs, i64 64)
  %bv.ok = icmp eq i32 %bv, 13
  call void @ut_check(i1 %bv.ok, ptr @m.badver)

  ; ---------------- chunked decode ----------------
  %cd = call i32 @universe_http_chunked_decode(ptr @chk, i64 24, ptr %decbuf, i64 64, ptr %declen)
  %cd.ok = icmp eq i32 %cd, 0
  call void @ut_check(i1 %cd.ok, ptr @m.chk_ok)
  %dl = load i64, ptr %declen, align 8
  call void @ut_check_eq(i64 %dl, i64 9, ptr @m.chk_len)
  call void @check_slice(ptr %decbuf, i64 %dl, ptr @x.wiki, i64 9, ptr @m.chk_ok)

  ; ---------------- chunked encode via pipe ----------------
  %pr = call i32 @pipe(ptr %pipefds)
  %rfd = load i32, ptr %pipefds, align 4
  %wfdp = getelementptr inbounds nuw i8, ptr %pipefds, i64 4
  %wfd = load i32, ptr %wfdp, align 4
  %ew = call ptr @universe_io_writer_create(i32 %wfd, i64 0)
  %enc = call i32 @universe_http_chunked_encode(ptr %ew, ptr @x.wiki, i64 9)
  %efl = call i32 @universe_io_writer_flush(ptr %ew)
  call void @universe_io_writer_destroy(ptr %ew)
  %cwfd = call i32 @close(i32 %wfd)
  %rn = call i64 @read(i32 %rfd, ptr %rdbuf, i64 256)
  %crfd = call i32 @close(i32 %rfd)
  %dec2 = call i32 @universe_http_chunked_decode(ptr %rdbuf, i64 %rn, ptr %decbuf, i64 64, ptr %declen)
  %dec2.ok = icmp eq i32 %dec2, 0
  %dl2 = load i64, ptr %declen, align 8
  %dl2.ok = icmp eq i64 %dl2, 9
  %eq2 = call i1 @universe_simd_equal(ptr %decbuf, ptr @x.wiki, i64 9)
  %enc.all0 = and i1 %dec2.ok, %dl2.ok
  %enc.all = and i1 %enc.all0, %eq2
  call void @ut_check(i1 %enc.all, ptr @m.enc_ok)

  ; ================= full round-trip over loopback =================
  ; ip = 127.0.0.1 = 0x7f000001 = 2130706433 ; functional port 18080
  %lf = call i32 @universe_net_tcp_listen(i32 2130706433, i32 18080, i32 64)
  %lf.ok = icmp sge i32 %lf, 0
  call void @ut_check(i1 %lf.ok, ptr @m.listen)
  store i64 0, ptr %counter, align 8
  store i32 %lf, ptr %targ, align 4
  %targ.ud = getelementptr inbounds nuw i8, ptr %targ, i64 8
  store ptr %counter, ptr %targ.ud, align 8
  %targ.mc = getelementptr inbounds nuw i8, ptr %targ, i64 16
  store i64 2, ptr %targ.mc, align 8
  %targ.bc = getelementptr inbounds nuw i8, ptr %targ, i64 24
  store i64 0, ptr %targ.bc, align 8
  %pc = call i32 @pthread_create(ptr %th, ptr null, ptr @server_thread, ptr %targ)

  ; connection A: two keep-alive requests
  %connA = call ptr @universe_http_connect(i32 2130706433, i32 18080, i64 0)
  %connA.ok = icmp ne ptr %connA, null
  call void @ut_check(i1 %connA.ok, ptr @m.conn)
  ; req1 GET / keepalive -> body "pong"
  %st1 = call i32 @universe_http_client_request(ptr %connA, ptr @s.get, i64 3, ptr @s.slash, i64 1, i64 1, ptr null, i64 0, ptr null, i64 0, i32 1, ptr %rmsg, ptr %rhdrs, i64 64)
  %st1.ok = icmp eq i32 %st1, 0
  call void @ut_check(i1 %st1.ok, ptr @m.rt_st)
  %code1 = call i64 @ld64(ptr %rmsg, i64 40)
  call void @ut_check_eq(i64 %code1, i64 200, ptr @m.rt_code)
  %b1p = call ptr @ldp(ptr %rmsg, i64 56)
  %b1l = call i64 @ld64(ptr %rmsg, i64 64)
  call void @check_slice(ptr %b1p, i64 %b1l, ptr @s.pong, i64 4, ptr @m.rt_body)
  ; req2 POST /echo body "ping" -> body "ping"
  %st2 = call i32 @universe_http_client_request(ptr %connA, ptr @s.post, i64 4, ptr @s.echo, i64 5, i64 1, ptr null, i64 0, ptr @s.ping, i64 4, i32 1, ptr %rmsg, ptr %rhdrs, i64 64)
  %st2.ok = icmp eq i32 %st2, 0
  call void @ut_check(i1 %st2.ok, ptr @m.rt_st)
  %b2p = call ptr @ldp(ptr %rmsg, i64 56)
  %b2l = call i64 @ld64(ptr %rmsg, i64 64)
  call void @check_slice(ptr %b2p, i64 %b2l, ptr @s.ping, i64 4, ptr @m.rt_body)
  call void @universe_http_conn_destroy(ptr %connA)

  ; connection B: single Connection: close request
  %connB = call ptr @universe_http_connect(i32 2130706433, i32 18080, i64 0)
  %st3 = call i32 @universe_http_client_request(ptr %connB, ptr @s.get, i64 3, ptr @s.slash, i64 1, i64 1, ptr null, i64 0, ptr null, i64 0, i32 0, ptr %rmsg, ptr %rhdrs, i64 64)
  %st3.ok = icmp eq i32 %st3, 0
  call void @ut_check(i1 %st3.ok, ptr @m.rt_st)
  %b3p = call ptr @ldp(ptr %rmsg, i64 56)
  %b3l = call i64 @ld64(ptr %rmsg, i64 64)
  call void @check_slice(ptr %b3p, i64 %b3l, ptr @s.pong, i64 4, ptr @m.rt_body)
  call void @universe_http_conn_destroy(ptr %connB)

  %tid = load i64, ptr %th, align 8
  %pj = call i32 @pthread_join(i64 %tid, ptr null)
  %served = load i64, ptr %counter, align 8
  call void @ut_check_eq(i64 %served, i64 3, ptr @m.rt_cnt)
  %cloself = call i32 @universe_net_tcp_close(i32 %lf)

  ; ---------------- bench ----------------
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %bh.rep.head

  ; keep-alive round-trip distribution (ops_per_rep = 20000 requests over one
  ; connection). Each rep re-listens/re-spawns/re-connects (mc=1): a single
  ; keep-alive connection per rep, so no ephemeral-port/TIME_WAIT pressure.
bh.rep.head:
  %bh.rep = phi i64 [ 0, %bench ], [ %bh.rep.n, %bh.rep.cont ]
  %blf = call i32 @universe_net_tcp_listen(i32 2130706433, i32 18081, i32 128)
  store i64 0, ptr %counter, align 8
  store i32 %blf, ptr %targ, align 4
  store ptr %counter, ptr %targ.ud, align 8
  store i64 1, ptr %targ.mc, align 8
  store i64 0, ptr %targ.bc, align 8
  %bpc = call i32 @pthread_create(ptr %th, ptr null, ptr @server_thread, ptr %targ)
  %bconn = call ptr @universe_http_connect(i32 2130706433, i32 18081, i64 0)
  %t0 = call double @ut_now_sec()
  br label %bloop

bloop:
  %bi = phi i64 [ 0, %bh.rep.head ], [ %bi.n, %bloop ]
  %bst = call i32 @universe_http_client_request(ptr %bconn, ptr @s.get, i64 3, ptr @s.slash, i64 1, i64 1, ptr null, i64 0, ptr null, i64 0, i32 1, ptr %rmsg, ptr %rhdrs, i64 64)
  %bi.n = add i64 %bi, 1
  %bmore = icmp ult i64 %bi.n, 20000
  br i1 %bmore, label %bloop, label %bdone

bdone:
  %t1 = call double @ut_now_sec()
  call void @universe_http_conn_destroy(ptr %bconn)
  %btid = load i64, ptr %th, align 8
  %bpj = call i32 @pthread_join(i64 %btid, ptr null)
  %bclf = call i32 @universe_net_tcp_close(i32 %blf)
  %bh.dt = fsub double %t1, %t0
  %bh.warm = icmp eq i64 %bh.rep, 0
  br i1 %bh.warm, label %bh.rep.cont, label %bh.rep.store
bh.rep.store:
  %bh.idx = sub i64 %bh.rep, 1
  %bh.sp = getelementptr inbounds [16 x double], ptr @http.samp, i64 0, i64 %bh.idx
  store double %bh.dt, ptr %bh.sp, align 8
  br label %bh.rep.cont
bh.rep.cont:
  %bh.rep.n = add i64 %bh.rep, 1
  %bh.more = icmp ult i64 %bh.rep.n, 17
  br i1 %bh.more, label %bh.rep.head, label %bh.rep.end
bh.rep.end:
  call void @ut_report_dist(ptr @http.samp, i64 16, i64 20000, ptr @lbl.http)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
