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

; POSIX TCP sockets (blocking) with clean error mapping — thin, zero-alloc
; wrappers over the kernel socket API. IPv4 only (127.0.0.1-friendly).
;
; DESIGN:
;   * Return convention: a non-negative i32 is a live fd (or 0 on success for
;     the void-ish ops); a NEGATIVE i32/i64 is the negated universe error code
;     (-15 = IO, -8 = INVALID_ARG). Callers test `< 0`. fds are always >= 0
;     from the kernel, so the sign channel is unambiguous.
;   * EVERY OS-divergent primitive (errno accessor, sockaddr_in byte layout,
;     SOL_SOCKET/SO_REUSEADDR/O_NONBLOCK/MSG_DONTWAIT constant values) lives in
;     src/net/osconst_{bsd,linux}.ll, of which the build links exactly one per
;     host (Makefile selects by uname). This module bakes NO OS constant, so it
;     is byte-identical and correct on macOS, FreeBSD, and Linux. Portable
;     across all three and used inline here: AF_INET=2, SOCK_STREAM=1,
;     IPPROTO_TCP=6, TCP_NODELAY=1, F_GETFL=3, F_SETFL=4, SHUT_*=0/1/2, EINTR=4.
;     sin_addr sits at byte offset 4 on every target, so the accept peer read
;     below is OS-independent; only the WRITE side (sin_len/family) diverges and
;     is delegated to universe_net_os_build_sockaddr.
;   * send_all/recv retry on EINTR and handle partial IO; no syscall-per-byte.
;
; API (fd >= 0 ok, negative = -errorcode):
;   i32 universe_net_tcp_listen(i32 ip, i32 port, i32 backlog)
;   i32 universe_net_tcp_accept(i32 listenfd, ptr out_peer_ip)   ; out may be null
;   i32 universe_net_tcp_connect(i32 ip, i32 port)
;   i32 universe_net_tcp_set_nodelay(i32 fd, i32 on)
;   i32 universe_net_tcp_set_nonblocking(i32 fd, i32 on)
;   i32 universe_net_tcp_shutdown(i32 fd, i32 how)               ; 0/1/2
;   i32 universe_net_tcp_close(i32 fd)
;   i64 universe_net_tcp_send_all(i32 fd, ptr buf, i64 len)      ; == len ok
;   i64 universe_net_tcp_recv(i32 fd, ptr buf, i64 len)          ; bytes, 0=EOF

declare i32 @socket(i32, i32, i32)
declare i32 @bind(i32, ptr, i32)
declare i32 @listen(i32, i32)
declare i32 @accept(i32, ptr, ptr)
declare i32 @connect(i32, ptr, i32)
declare i32 @setsockopt(i32, i32, i32, ptr, i32)
declare i32 @shutdown(i32, i32)
declare i32 @close(i32)
declare i64 @read(i32, ptr, i64)
declare i64 @write(i32, ptr, i64)
declare i32 @fcntl(i32, i32, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i32 @llvm.bswap.i32(i32)

; OS-divergent primitives — resolved to osconst_{bsd,linux}.ll by the build.
declare i32 @universe_net_os_errno()
declare void @universe_net_os_build_sockaddr(ptr, i32, i32)
declare i32 @universe_net_os_enable_reuseaddr(i32)
declare i32 @universe_net_os_nonblock_flags(i32, i32)

define i32 @universe_net_tcp_listen(i32 %ip, i32 %port, i32 %backlog) local_unnamed_addr #2 {
entry:
  %fd = call i32 @socket(i32 2, i32 1, i32 0)      ; AF_INET, SOCK_STREAM
  %sock.bad = icmp slt i32 %fd, 0
  br i1 %sock.bad, label %err.io, label %opt, !prof !0

opt:
  %so = call i32 @universe_net_os_enable_reuseaddr(i32 %fd)
  %so.bad = icmp slt i32 %so, 0
  br i1 %so.bad, label %err.close, label %do.bind, !prof !0

do.bind:
  %sa = alloca [16 x i8], align 8
  call void @universe_net_os_build_sockaddr(ptr nonnull %sa, i32 %ip, i32 %port)
  %br = call i32 @bind(i32 %fd, ptr nonnull %sa, i32 16)
  %b.bad = icmp slt i32 %br, 0
  br i1 %b.bad, label %err.close, label %do.listen, !prof !0

do.listen:
  %lr = call i32 @listen(i32 %fd, i32 %backlog)
  %l.bad = icmp slt i32 %lr, 0
  br i1 %l.bad, label %err.close, label %ok, !prof !0

ok:
  ret i32 %fd

err.close:
  %ign = call i32 @close(i32 %fd)
  br label %err.io

err.io:
  ret i32 -15
}

define i32 @universe_net_tcp_accept(i32 %listenfd, ptr %out) local_unnamed_addr #2 {
entry:
  %ss = alloca [128 x i8], align 8            ; sockaddr_storage
  %slen.p = alloca i32, align 4
  br label %retry

retry:
  store i32 128, ptr %slen.p, align 4
  %fd = call i32 @accept(i32 %listenfd, ptr nonnull %ss, ptr nonnull %slen.p)
  %bad = icmp slt i32 %fd, 0
  br i1 %bad, label %chk, label %good, !prof !0

chk:
  %e = call i32 @universe_net_os_errno()
  %eintr = icmp eq i32 %e, 4
  br i1 %eintr, label %retry, label %err.io, !prof !1

good:
  %has.out = icmp ne ptr %out, null
  br i1 %has.out, label %store.peer, label %done

store.peer:
  ; BSD sockaddr_in: peer addr big-endian at offset 4; hand back host order
  %addr.p = getelementptr inbounds nuw i8, ptr %ss, i64 4
  %abe = load i32, ptr %addr.p, align 4
  %ah = call i32 @llvm.bswap.i32(i32 %abe)
  store i32 %ah, ptr %out, align 4
  br label %done

done:
  ret i32 %fd

err.io:
  ret i32 -15
}

define i32 @universe_net_tcp_connect(i32 %ip, i32 %port) local_unnamed_addr #2 {
entry:
  %fd = call i32 @socket(i32 2, i32 1, i32 0)
  %sock.bad = icmp slt i32 %fd, 0
  br i1 %sock.bad, label %err.io, label %do.connect, !prof !0

do.connect:
  %sa = alloca [16 x i8], align 8
  call void @universe_net_os_build_sockaddr(ptr nonnull %sa, i32 %ip, i32 %port)
  %cr = call i32 @connect(i32 %fd, ptr nonnull %sa, i32 16)
  %c.bad = icmp slt i32 %cr, 0
  br i1 %c.bad, label %err.close, label %ok, !prof !0

ok:
  ret i32 %fd

err.close:
  %ign = call i32 @close(i32 %fd)
  br label %err.io

err.io:
  ret i32 -15
}

define i32 @universe_net_tcp_set_nodelay(i32 %fd, i32 %on) local_unnamed_addr #2 {
entry:
  %v.p = alloca i32, align 4
  store i32 %on, ptr %v.p, align 4
  ; IPPROTO_TCP=6, TCP_NODELAY=1 (portable)
  %r = call i32 @setsockopt(i32 %fd, i32 6, i32 1, ptr nonnull %v.p, i32 4)
  %bad = icmp slt i32 %r, 0
  %ret = select i1 %bad, i32 -15, i32 0
  ret i32 %ret
}

define i32 @universe_net_tcp_set_nonblocking(i32 %fd, i32 %on) local_unnamed_addr #2 {
entry:
  %fl = call i32 (i32, i32, ...) @fcntl(i32 %fd, i32 3)     ; F_GETFL
  %fl.bad = icmp slt i32 %fl, 0
  br i1 %fl.bad, label %err.io, label %set, !prof !0

set:
  %newfl = call i32 @universe_net_os_nonblock_flags(i32 %fl, i32 %on)
  %r = call i32 (i32, i32, ...) @fcntl(i32 %fd, i32 4, i32 %newfl)   ; F_SETFL
  %bad = icmp slt i32 %r, 0
  br i1 %bad, label %err.io, label %ok, !prof !0

ok:
  ret i32 0

err.io:
  ret i32 -15
}

define i32 @universe_net_tcp_shutdown(i32 %fd, i32 %how) local_unnamed_addr #2 {
entry:
  %r = call i32 @shutdown(i32 %fd, i32 %how)
  %bad = icmp slt i32 %r, 0
  %ret = select i1 %bad, i32 -15, i32 0
  ret i32 %ret
}

define i32 @universe_net_tcp_close(i32 %fd) local_unnamed_addr #2 {
entry:
  %r = call i32 @close(i32 %fd)
  %bad = icmp slt i32 %r, 0
  %ret = select i1 %bad, i32 -15, i32 0
  ret i32 %ret
}

define i64 @universe_net_tcp_send_all(i32 %fd, ptr %buf, i64 %len) local_unnamed_addr #2 {
entry:
  %buf.null = icmp eq ptr %buf, null
  br i1 %buf.null, label %err.arg, label %loop, !prof !0

loop:
  %sent = phi i64 [ 0, %entry ], [ %sent.n, %advance ], [ %sent, %chk ]
  %rem = sub nuw i64 %len, %sent
  %done = icmp eq i64 %rem, 0
  br i1 %done, label %ok, label %do.write

do.write:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %sent
  %n = call i64 @write(i32 %fd, ptr %p, i64 %rem)
  %n.bad = icmp slt i64 %n, 0
  br i1 %n.bad, label %chk, label %advance, !prof !0

chk:
  %e = call i32 @universe_net_os_errno()
  %eintr = icmp eq i32 %e, 4
  br i1 %eintr, label %loop, label %err.io, !prof !1

advance:
  %sent.n = add nuw i64 %sent, %n
  br label %loop

ok:
  ret i64 %len

err.arg:
  ret i64 -8

err.io:
  ret i64 -15
}

define i64 @universe_net_tcp_recv(i32 %fd, ptr %buf, i64 %len) local_unnamed_addr #2 {
entry:
  %buf.null = icmp eq ptr %buf, null
  br i1 %buf.null, label %err.arg, label %retry, !prof !0

retry:
  %n = call i64 @read(i32 %fd, ptr %buf, i64 %len)
  %n.bad = icmp slt i64 %n, 0
  br i1 %n.bad, label %chk, label %ok, !prof !0

chk:
  %e = call i32 @universe_net_os_errno()
  %eintr = icmp eq i32 %e, 4
  br i1 %eintr, label %retry, label %err.io, !prof !1

ok:
  ret i64 %n

err.arg:
  ret i64 -8

err.io:
  ret i64 -15
}

attributes #0 = { alwaysinline nounwind willreturn memory(argmem: write) }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 2000}
