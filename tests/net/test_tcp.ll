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

; Tests for src/net/tcp.ll + src/net/pool.ll over 127.0.0.1 loopback.

declare i32 @universe_net_tcp_listen(i32, i32, i32)
declare i32 @universe_net_tcp_accept(i32, ptr)
declare i32 @universe_net_tcp_connect(i32, i32)
declare i32 @universe_net_tcp_set_nodelay(i32, i32)
declare i32 @universe_net_tcp_set_nonblocking(i32, i32)
declare i32 @universe_net_tcp_shutdown(i32, i32)
declare i32 @universe_net_tcp_close(i32)
declare i64 @universe_net_tcp_send_all(i32, ptr, i64)
declare i64 @universe_net_tcp_recv(i32, ptr, i64)

declare ptr @universe_net_pool_create(i64)
declare void @universe_net_pool_destroy(ptr)
declare i32 @universe_net_pool_get(ptr, i32, i32)
declare i32 @universe_net_pool_put(ptr, i32, i32, i32)
declare i64 @universe_net_pool_count(ptr)

declare i32 @socket(i32, i32, i32)
declare i32 @close(i32)
declare i32 @getsockname(i32, ptr, ptr)
declare i32 @fcntl(i32, i32, ...)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @usleep(i32)
declare i16 @llvm.bswap.i16(i16)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)

@g.listenfd = internal global i32 0, align 4
@g.port     = internal global i32 0, align 4
@g.srverr   = internal global i64 0, align 8
@g.benchn   = internal global i64 0, align 8

@LOOPBACK = private constant i32 2130706433          ; 0x7F000001 = 127.0.0.1

@s.hello  = private unnamed_addr constant [5 x i8] c"hello"

@m.listen  = private unnamed_addr constant [16 x i8] c"listen fd valid\00"
@m.getsn   = private unnamed_addr constant [16 x i8] c"getsockname ok \00"
@m.connect = private unnamed_addr constant [16 x i8] c"connect fd ok  \00"
@m.nodelay = private unnamed_addr constant [12 x i8] c"nodelay ok \00"
@m.nonblk  = private unnamed_addr constant [16 x i8] c"nonblocking ok \00"
@m.snd5    = private unnamed_addr constant [16 x i8] c"send_all 5 == 5\00"
@m.echo5   = private unnamed_addr constant [16 x i8] c"echo 5 matches \00"
@m.bigsnd  = private unnamed_addr constant [20 x i8] c"send_all 1MB done  \00"
@m.bigcnt  = private unnamed_addr constant [20 x i8] c"server got 1MB     \00"
@m.bigsum  = private unnamed_addr constant [20 x i8] c"byte-sum matches   \00"
@m.srvok   = private unnamed_addr constant [16 x i8] c"server no error\00"
@m.errpath = private unnamed_addr constant [24 x i8] c"connect closed -> err  \00"

@m.p.create = private unnamed_addr constant [16 x i8] c"pool nonnull   \00"
@m.p.put2   = private unnamed_addr constant [16 x i8] c"put 2 count==2 \00"
@m.p.getB   = private unnamed_addr constant [16 x i8] c"get reuse s2   \00"
@m.p.cnt1   = private unnamed_addr constant [16 x i8] c"count==1       \00"
@m.p.unk    = private unnamed_addr constant [16 x i8] c"get unknown=-1 \00"
@m.p.getA   = private unnamed_addr constant [16 x i8] c"get reuse s1   \00"
@m.p.cnt0   = private unnamed_addr constant [16 x i8] c"count==0       \00"
@m.p.evcnt  = private unnamed_addr constant [16 x i8] c"evict count==2 \00"
@m.p.evcl   = private unnamed_addr constant [20 x i8] c"evicted fd closed  \00"
@m.p.evg1   = private unnamed_addr constant [20 x i8] c"evicted key gone   \00"
@m.p.evg2   = private unnamed_addr constant [16 x i8] c"survivor s2 ok \00"
@m.p.evg3   = private unnamed_addr constant [16 x i8] c"survivor s3 ok \00"
@m.p.dead   = private unnamed_addr constant [24 x i8] c"dead peer -> get = -1  \00"

@fmt.bench1 = private unnamed_addr constant [47 x i8] c"bench %lld conn/echo/close: %.3f us/roundtrip\0A\00"
@tcppool.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.tcppool = private unnamed_addr constant [21 x i8] c"net pool get/put x8w\00"

; --- helpers ---------------------------------------------------------------

; read exactly %want bytes (or fewer on EOF/error); returns bytes read
define internal i64 @recv_exact(i32 %fd, ptr %buf, i64 %want) {
entry:
  br label %loop

loop:
  %got = phi i64 [ 0, %entry ], [ %got.n, %cont ]
  %rem = sub i64 %want, %got
  %done = icmp eq i64 %rem, 0
  br i1 %done, label %ret, label %do

do:
  %p = getelementptr inbounds i8, ptr %buf, i64 %got
  %n = call i64 @universe_net_tcp_recv(i32 %fd, ptr %p, i64 %rem)
  %short = icmp sle i64 %n, 0
  br i1 %short, label %ret, label %cont

cont:
  %got.n = add i64 %got, %n
  br label %loop

ret:
  %r = phi i64 [ %want, %loop ], [ %got, %do ]
  ret i64 %r
}

; additive sum of %n bytes at %buf
define internal i64 @byte_sum(ptr %buf, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %ret0, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %bp = getelementptr inbounds i8, ptr %buf, i64 %i
  %b = load i8, ptr %bp, align 1
  %bz = zext i8 %b to i64
  %acc.n = add i64 %acc, %bz
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %ret

ret:
  ret i64 %acc.n

ret0:
  ret i64 0
}

; --- correctness server thread: phase1 echo 5, phase2 drain + report -------
define internal ptr @srv_main(ptr %arg) {
entry:
  %buf5 = alloca [8 x i8], align 8
  %big = alloca [65536 x i8], align 16
  %out = alloca [16 x i8], align 8
  %lf = load i32, ptr @g.listenfd, align 4
  %cf = call i32 @universe_net_tcp_accept(i32 %lf, ptr null)
  %acc.bad = icmp slt i32 %cf, 0
  br i1 %acc.bad, label %fail, label %phase1

phase1:
  %g5 = call i64 @recv_exact(i32 %cf, ptr %buf5, i64 5)
  %g5.ok = icmp eq i64 %g5, 5
  br i1 %g5.ok, label %echo, label %fail

echo:
  %se = call i64 @universe_net_tcp_send_all(i32 %cf, ptr %buf5, i64 5)
  %se.ok = icmp eq i64 %se, 5
  br i1 %se.ok, label %drain, label %fail

drain:
  %tot = phi i64 [ 0, %echo ], [ %tot.n, %more ]
  %sum = phi i64 [ 0, %echo ], [ %sum.n, %more ]
  %n = call i64 @universe_net_tcp_recv(i32 %cf, ptr %big, i64 65536)
  %eof = icmp eq i64 %n, 0
  br i1 %eof, label %report, label %chk

chk:
  %rerr = icmp slt i64 %n, 0
  br i1 %rerr, label %fail, label %more

more:
  %s = call i64 @byte_sum(ptr %big, i64 %n)
  %sum.n = add i64 %sum, %s
  %tot.n = add i64 %tot, %n
  br label %drain

report:
  store i64 %tot, ptr %out, align 8
  %out8 = getelementptr inbounds nuw i8, ptr %out, i64 8
  store i64 %sum, ptr %out8, align 8
  %sr = call i64 @universe_net_tcp_send_all(i32 %cf, ptr %out, i64 16)
  %cc = call i32 @universe_net_tcp_close(i32 %cf)
  ret ptr null

fail:
  store i64 1, ptr @g.srverr, align 8
  %cc2 = call i32 @universe_net_tcp_close(i32 %cf)
  ret ptr null
}

; --- bench server thread: N * (accept, recv64, echo64, close) --------------
define internal ptr @srv_bench(ptr %arg) {
entry:
  %buf = alloca [64 x i8], align 8
  %lf = load i32, ptr @g.listenfd, align 4
  %n = load i64, ptr @g.benchn, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %cf = call i32 @universe_net_tcp_accept(i32 %lf, ptr null)
  %bad = icmp slt i32 %cf, 0
  br i1 %bad, label %cont, label %serve

serve:
  %g = call i64 @recv_exact(i32 %cf, ptr %buf, i64 64)
  %se = call i64 @universe_net_tcp_send_all(i32 %cf, ptr %buf, i64 64)
  %cc = call i32 @universe_net_tcp_close(i32 %cf)
  br label %cont

cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret ptr null
}

; --- correctness test ------------------------------------------------------
define internal void @test_roundtrip() {
entry:
  %tid = alloca i64, align 8
  %ss = alloca [128 x i8], align 8
  %slen = alloca i32, align 4
  %rbuf5 = alloca [8 x i8], align 8
  %res = alloca [16 x i8], align 8
  store i64 0, ptr @g.srverr, align 8

  %ip = load i32, ptr @LOOPBACK, align 4
  %lf = call i32 @universe_net_tcp_listen(i32 %ip, i32 0, i32 128)
  %lf.ok = icmp sge i32 %lf, 0
  call void @ut_check(i1 %lf.ok, ptr @m.listen)
  br i1 %lf.ok, label %getport, label %ret

getport:
  store i32 128, ptr %slen, align 4
  %gr = call i32 @getsockname(i32 %lf, ptr nonnull %ss, ptr nonnull %slen)
  %gr.ok = icmp eq i32 %gr, 0
  call void @ut_check(i1 %gr.ok, ptr @m.getsn)
  %port.p = getelementptr inbounds nuw i8, ptr %ss, i64 2
  %pbe = load i16, ptr %port.p, align 2
  %ph = call i16 @llvm.bswap.i16(i16 %pbe)
  %port = zext i16 %ph to i32
  store i32 %port, ptr @g.port, align 4
  store i32 %lf, ptr @g.listenfd, align 4
  %cr = call i32 @pthread_create(ptr nonnull %tid, ptr null, ptr @srv_main, ptr null)
  br label %connect

connect:
  %cf = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %cf.ok = icmp sge i32 %cf, 0
  call void @ut_check(i1 %cf.ok, ptr @m.connect)
  br i1 %cf.ok, label %opts, label %joinret

opts:
  %nd = call i32 @universe_net_tcp_set_nodelay(i32 %cf, i32 1)
  %nd.ok = icmp eq i32 %nd, 0
  call void @ut_check(i1 %nd.ok, ptr @m.nodelay)

  ; phase 1: send "hello", read the echo back
  %s5 = call i64 @universe_net_tcp_send_all(i32 %cf, ptr @s.hello, i64 5)
  %s5.ok = icmp eq i64 %s5, 5
  call void @ut_check(i1 %s5.ok, ptr @m.snd5)
  %g5 = call i64 @recv_exact(i32 %cf, ptr %rbuf5, i64 5)
  %cmp = call i32 @memcmp(ptr %rbuf5, ptr @s.hello, i64 5)
  %g5.ok = icmp eq i64 %g5, 5
  %cmp.ok = icmp eq i32 %cmp, 0
  %echo.ok = and i1 %g5.ok, %cmp.ok
  call void @ut_check(i1 %echo.ok, ptr @m.echo5)

  ; phase 2: 1MB partial-write stress
  %big = call ptr @malloc(i64 1048576)
  br label %fill

fill:
  %i = phi i64 [ 0, %opts ], [ %i.n, %fill ]
  %exp = phi i64 [ 0, %opts ], [ %exp.n, %fill ]
  %m1 = mul i64 %i, 131
  %m2 = add i64 %m1, 7
  %byte = trunc i64 %m2 to i8
  %bp = getelementptr inbounds i8, ptr %big, i64 %i
  store i8 %byte, ptr %bp, align 1
  %bz = zext i8 %byte to i64
  %exp.n = add i64 %exp, %bz
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1048576
  br i1 %more, label %fill, label %sendbig

sendbig:
  %sb = call i64 @universe_net_tcp_send_all(i32 %cf, ptr %big, i64 1048576)
  %sb.ok = icmp eq i64 %sb, 1048576
  call void @ut_check(i1 %sb.ok, ptr @m.bigsnd)
  %sd = call i32 @universe_net_tcp_shutdown(i32 %cf, i32 1)     ; SHUT_WR
  %rr = call i64 @recv_exact(i32 %cf, ptr %res, i64 16)
  %tot = load i64, ptr %res, align 8
  %res8 = getelementptr inbounds nuw i8, ptr %res, i64 8
  %rsum = load i64, ptr %res8, align 8
  %tot.ok = icmp eq i64 %tot, 1048576
  call void @ut_check(i1 %tot.ok, ptr @m.bigcnt)
  %sum.ok = icmp eq i64 %rsum, %exp.n
  call void @ut_check(i1 %sum.ok, ptr @m.bigsum)
  call void @free(ptr %big)
  %cc = call i32 @universe_net_tcp_close(i32 %cf)
  br label %joinret

joinret:
  %t = load i64, ptr %tid, align 8
  %jr = call i32 @pthread_join(i64 %t, ptr null)
  %serr = load i64, ptr @g.srverr, align 8
  %serr.ok = icmp eq i64 %serr, 0
  call void @ut_check(i1 %serr.ok, ptr @m.srvok)
  %lc = call i32 @universe_net_tcp_close(i32 %lf)
  ret void

ret:
  ret void
}

; --- error path: connect to a closed port ----------------------------------
define internal void @test_errpath() {
entry:
  %ss = alloca [128 x i8], align 8
  %slen = alloca i32, align 4
  %ip = load i32, ptr @LOOPBACK, align 4
  %lf = call i32 @universe_net_tcp_listen(i32 %ip, i32 0, i32 1)
  %lf.ok = icmp sge i32 %lf, 0
  br i1 %lf.ok, label %getp, label %skip

getp:
  store i32 128, ptr %slen, align 4
  %gr = call i32 @getsockname(i32 %lf, ptr nonnull %ss, ptr nonnull %slen)
  %port.p = getelementptr inbounds nuw i8, ptr %ss, i64 2
  %pbe = load i16, ptr %port.p, align 2
  %ph = call i16 @llvm.bswap.i16(i16 %pbe)
  %port = zext i16 %ph to i32
  %cl = call i32 @universe_net_tcp_close(i32 %lf)
  %cf = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %failed = icmp slt i32 %cf, 0
  call void @ut_check(i1 %failed, ptr @m.errpath)
  br i1 %failed, label %skip, label %closeit

closeit:
  %cc = call i32 @universe_net_tcp_close(i32 %cf)
  br label %skip

skip:
  ret void
}

; --- pool reuse + eviction -------------------------------------------------
define internal void @test_pool() {
entry:
  %ip = load i32, ptr @LOOPBACK, align 4
  %s1 = call i32 @socket(i32 2, i32 1, i32 0)
  %s2 = call i32 @socket(i32 2, i32 1, i32 0)
  %s3 = call i32 @socket(i32 2, i32 1, i32 0)
  %pool = call ptr @universe_net_pool_create(i64 2)
  %p.ok = icmp ne ptr %pool, null
  call void @ut_check(i1 %p.ok, ptr @m.p.create)
  br i1 %p.ok, label %fill, label %ret

fill:
  %r1 = call i32 @universe_net_pool_put(ptr %pool, i32 %s1, i32 %ip, i32 1001)
  %r2 = call i32 @universe_net_pool_put(ptr %pool, i32 %s2, i32 %ip, i32 1002)
  %c2 = call i64 @universe_net_pool_count(ptr %pool)
  %c2.ok = icmp eq i64 %c2, 2
  call void @ut_check(i1 %c2.ok, ptr @m.p.put2)

  %gB = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 1002)
  %gB.ok = icmp eq i32 %gB, %s2
  call void @ut_check(i1 %gB.ok, ptr @m.p.getB)
  %c1 = call i64 @universe_net_pool_count(ptr %pool)
  %c1.ok = icmp eq i64 %c1, 1
  call void @ut_check(i1 %c1.ok, ptr @m.p.cnt1)

  %gu = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 9999)
  %gu.ok = icmp eq i32 %gu, -1
  call void @ut_check(i1 %gu.ok, ptr @m.p.unk)

  %gA = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 1001)
  %gA.ok = icmp eq i32 %gA, %s1
  call void @ut_check(i1 %gA.ok, ptr @m.p.getA)
  %c0 = call i64 @universe_net_pool_count(ptr %pool)
  %c0.ok = icmp eq i64 %c0, 0
  call void @ut_check(i1 %c0.ok, ptr @m.p.cnt0)

  ; eviction: put s1(1001), s2(1002), s3(1003) into cap-2 pool -> s1 evicted
  %e1 = call i32 @universe_net_pool_put(ptr %pool, i32 %s1, i32 %ip, i32 1001)
  %e2 = call i32 @universe_net_pool_put(ptr %pool, i32 %s2, i32 %ip, i32 1002)
  %e3 = call i32 @universe_net_pool_put(ptr %pool, i32 %s3, i32 %ip, i32 1003)
  %ec = call i64 @universe_net_pool_count(ptr %pool)
  %ec.ok = icmp eq i64 %ec, 2
  call void @ut_check(i1 %ec.ok, ptr @m.p.evcnt)

  ; s1 was LRU -> evicted and closed: fcntl F_GETFL must fail (EBADF)
  %fl = call i32 (i32, i32, ...) @fcntl(i32 %s1, i32 3)
  %closed = icmp slt i32 %fl, 0
  call void @ut_check(i1 %closed, ptr @m.p.evcl)

  %eg1 = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 1001)
  %eg1.ok = icmp eq i32 %eg1, -1
  call void @ut_check(i1 %eg1.ok, ptr @m.p.evg1)
  %eg2 = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 1002)
  %eg2.ok = icmp eq i32 %eg2, %s2
  call void @ut_check(i1 %eg2.ok, ptr @m.p.evg2)
  %eg3 = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 1003)
  %eg3.ok = icmp eq i32 %eg3, %s3
  call void @ut_check(i1 %eg3.ok, ptr @m.p.evg3)

  call void @universe_net_pool_destroy(ptr %pool)
  %x2 = call i32 @universe_net_tcp_close(i32 %s2)
  %x3 = call i32 @universe_net_tcp_close(i32 %s3)
  br label %ret

ret:
  ret void
}

; --- pool dead-peer detection ----------------------------------------------
define internal void @test_pool_dead() {
entry:
  %ss = alloca [128 x i8], align 8
  %slen = alloca i32, align 4
  %ip = load i32, ptr @LOOPBACK, align 4
  %lf = call i32 @universe_net_tcp_listen(i32 %ip, i32 0, i32 4)
  %lf.ok = icmp sge i32 %lf, 0
  br i1 %lf.ok, label %conn, label %ret

conn:
  store i32 128, ptr %slen, align 4
  %gr = call i32 @getsockname(i32 %lf, ptr nonnull %ss, ptr nonnull %slen)
  %port.p = getelementptr inbounds nuw i8, ptr %ss, i64 2
  %pbe = load i16, ptr %port.p, align 2
  %ph = call i16 @llvm.bswap.i16(i16 %pbe)
  %port = zext i16 %ph to i32
  %c = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %a = call i32 @universe_net_tcp_accept(i32 %lf, ptr null)
  %ac.ok = icmp sge i32 %a, 0
  br i1 %ac.ok, label %kill, label %ret

kill:
  ; close the server side: client %c now has a dead peer (FIN)
  %sd = call i32 @universe_net_tcp_shutdown(i32 %a, i32 2)
  %ca = call i32 @universe_net_tcp_close(i32 %a)
  %slp = call i32 @usleep(i32 50000)              ; let the FIN reach %c on loopback
  %pool = call ptr @universe_net_pool_create(i64 4)
  %pp = call i32 @universe_net_pool_put(ptr %pool, i32 %c, i32 %ip, i32 5555)
  %g = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 5555)
  %dead.ok = icmp eq i32 %g, -1
  call void @ut_check(i1 %dead.ok, ptr @m.p.dead)
  call void @universe_net_pool_destroy(ptr %pool)
  %lc = call i32 @universe_net_tcp_close(i32 %lf)
  br label %ret

ret:
  ret void
}

; --- benchmarks ------------------------------------------------------------
define internal void @run_bench() {
entry:
  %tid = alloca i64, align 8
  %ss = alloca [128 x i8], align 8
  %slen = alloca i32, align 4
  %buf = alloca [64 x i8], align 8
  %rbuf = alloca [64 x i8], align 8
  %ip = load i32, ptr @LOOPBACK, align 4
  %N = add i64 0, 4000
  store i64 %N, ptr @g.benchn, align 8

  %lf = call i32 @universe_net_tcp_listen(i32 %ip, i32 0, i32 128)
  store i32 %lf, ptr @g.listenfd, align 4
  store i32 128, ptr %slen, align 4
  %gr = call i32 @getsockname(i32 %lf, ptr nonnull %ss, ptr nonnull %slen)
  %port.p = getelementptr inbounds nuw i8, ptr %ss, i64 2
  %pbe = load i16, ptr %port.p, align 2
  %ph = call i16 @llvm.bswap.i16(i16 %pbe)
  %port = zext i16 %ph to i32
  %cr = call i32 @pthread_create(ptr nonnull %tid, ptr null, ptr @srv_bench, ptr null)

  %t0 = call double @ut_now_sec()
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %cf = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %s = call i64 @universe_net_tcp_send_all(i32 %cf, ptr %buf, i64 64)
  %g = call i64 @recv_exact(i32 %cf, ptr %rbuf, i64 64)
  %cc = call i32 @universe_net_tcp_close(i32 %cf)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %N
  br i1 %more, label %loop, label %after

after:
  %t1 = call double @ut_now_sec()
  %t = load i64, ptr %tid, align 8
  %jr = call i32 @pthread_join(i64 %t, ptr null)
  %lc = call i32 @universe_net_tcp_close(i32 %lf)
  %dt = fsub double %t1, %t0
  %us = fmul double %dt, 1.000000e+06
  %nf = sitofp i64 %N to double
  %per = fdiv double %us, %nf
  %pr = call i32 (ptr, ...) @printf(ptr @fmt.bench1, i64 %N, double %per)

  ; pool get/put churn
  %sfd = call i32 @socket(i32 2, i32 1, i32 0)
  %pool = call ptr @universe_net_pool_create(i64 8)
  %M = add i64 0, 200000
  br label %pl.rep.head

pl.rep.head:
  %pl.rep = phi i64 [ 0, %after ], [ %pl.rep.n, %pl.rep.cont ]
  %pt0 = call double @ut_now_sec()
  br label %ploop

ploop:
  %j = phi i64 [ 0, %pl.rep.head ], [ %j.n, %ploop ]
  %kk = and i64 %j, 7
  %kk32 = trunc i64 %kk to i32
  %pput = call i32 @universe_net_pool_put(ptr %pool, i32 %sfd, i32 %ip, i32 %kk32)
  %pget = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 %kk32)
  %j.n = add nuw i64 %j, 1
  %pmore = icmp ult i64 %j.n, %M
  br i1 %pmore, label %ploop, label %pafter

pafter:
  %pt1 = call double @ut_now_sec()
  %pl.dt = fsub double %pt1, %pt0
  %pl.warm = icmp eq i64 %pl.rep, 0
  br i1 %pl.warm, label %pl.rep.cont, label %pl.rep.store
pl.rep.store:
  %pl.idx = sub i64 %pl.rep, 1
  %pl.sp = getelementptr inbounds [16 x double], ptr @tcppool.samp, i64 0, i64 %pl.idx
  store double %pl.dt, ptr %pl.sp, align 8
  br label %pl.rep.cont
pl.rep.cont:
  %pl.rep.n = add i64 %pl.rep, 1
  %pl.more = icmp ult i64 %pl.rep.n, 17
  br i1 %pl.more, label %pl.rep.head, label %pl.rep.end
pl.rep.end:
  ; ops_per_rep = 200000 pool put/get pairs (8-way key churn)
  call void @ut_report_dist(ptr @tcppool.samp, i64 16, i64 200000, ptr @lbl.tcppool)
  call void @universe_net_pool_destroy(ptr %pool)
  %xs = call i32 @universe_net_tcp_close(i32 %sfd)
  ret void
}

declare ptr @malloc(i64)
declare void @free(ptr)

; --- set_nonblocking on a throwaway socket ---------------------------------
define internal void @test_nonblock() {
entry:
  %s = call i32 @socket(i32 2, i32 1, i32 0)
  %ok = icmp sge i32 %s, 0
  br i1 %ok, label %flip, label %ret

flip:
  %on = call i32 @universe_net_tcp_set_nonblocking(i32 %s, i32 1)
  %off = call i32 @universe_net_tcp_set_nonblocking(i32 %s, i32 0)
  %on.ok = icmp eq i32 %on, 0
  %off.ok = icmp eq i32 %off, 0
  %both = and i1 %on.ok, %off.ok
  call void @ut_check(i1 %both, ptr @m.nonblk)
  %cc = call i32 @universe_net_tcp_close(i32 %s)
  br label %ret

ret:
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_roundtrip()
  call void @test_nonblock()
  call void @test_errpath()
  call void @test_pool()
  call void @test_pool_dead()
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %summary

do.bench:
  call void @run_bench()
  br label %summary

summary:
  %r = call i32 @ut_summary()
  ret i32 %r
}
