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

; OS-divergent socket primitives — BSD variant (macOS / FreeBSD).
;
; DESIGN:
;   LLVM IR has no #ifdef, and the socket ABI diverges by OS: errno accessor,
;   sockaddr_in byte layout, and several socket-option/flag CONSTANT VALUES are
;   different on BSD vs Linux, so a single .ll with baked values is correct on
;   ONE OS and silently wrong at RUNTIME on the other (crosscheck/codegen cannot
;   catch it — only a real run does). We therefore isolate EVERY divergent
;   operation behind these fixed symbols; the build links osconst_bsd.ll on
;   macOS/FreeBSD and osconst_linux.ll on Linux (Makefile selects by uname).
;   src/net/tcp.ll and src/net/pool.ll bake NO OS constant — they call here.
;   These are cold, once-per-connection ops, so a cross-module call is free.
;
;   BSD values used here:
;     errno accessor = __error()
;     sockaddr_in    = [0]=sin_len(u8=16) [1]=sin_family(u8=AF_INET=2)
;                      [2..3]=sin_port(BE) [4..7]=sin_addr(BE) [8..15]=0
;     SOL_SOCKET=0xffff  SO_REUSEADDR=4  O_NONBLOCK=0x4  MSG_DONTWAIT=0x80

declare ptr @__error()
declare i32 @setsockopt(i32, i32, i32, ptr, i32)
declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)

; thread-local errno value (macOS/FreeBSD __error)
define i32 @universe_net_os_errno() #0 {
entry:
  %ep = call ptr @__error()
  %e = load i32, ptr %ep, align 4
  ret i32 %e
}

; build a BSD sockaddr_in into a caller-owned 16-byte buffer
define void @universe_net_os_build_sockaddr(ptr %sa, i32 %ip, i32 %port) #1 {
entry:
  store i64 0, ptr %sa, align 8
  %z8 = getelementptr inbounds nuw i8, ptr %sa, i64 8
  store i64 0, ptr %z8, align 8
  ; sin_len = 16, sin_family = AF_INET (2)
  store i8 16, ptr %sa, align 1
  %fam.p = getelementptr inbounds nuw i8, ptr %sa, i64 1
  store i8 2, ptr %fam.p, align 1
  ; sin_port = htons(port)
  %p16 = trunc i32 %port to i16
  %pbe = call i16 @llvm.bswap.i16(i16 %p16)
  %port.p = getelementptr inbounds nuw i8, ptr %sa, i64 2
  store i16 %pbe, ptr %port.p, align 2
  ; sin_addr = htonl(ip)
  %abe = call i32 @llvm.bswap.i32(i32 %ip)
  %addr.p = getelementptr inbounds nuw i8, ptr %sa, i64 4
  store i32 %abe, ptr %addr.p, align 4
  ret void
}

; SO_REUSEADDR(4) at SOL_SOCKET(0xffff); returns the raw setsockopt rc (<0 = err)
define i32 @universe_net_os_enable_reuseaddr(i32 %fd) #2 {
entry:
  %one.p = alloca i32, align 4
  store i32 1, ptr %one.p, align 4
  %r = call i32 @setsockopt(i32 %fd, i32 65535, i32 4, ptr nonnull %one.p, i32 4)
  ret i32 %r
}

; apply O_NONBLOCK(0x4 BSD) to fcntl F_GETFL flags, returning the F_SETFL value
define i32 @universe_net_os_nonblock_flags(i32 %fl, i32 %on) #3 {
entry:
  %want = icmp ne i32 %on, 0
  %with = or i32 %fl, 4
  %without = and i32 %fl, -5
  %newfl = select i1 %want, i32 %with, i32 %without
  ret i32 %newfl
}

; MSG_PEEK(2) | MSG_DONTWAIT(0x80 BSD) = 0x82 for a non-blocking liveness peek
define i32 @universe_net_os_msg_peek_dontwait() #3 {
entry:
  ret i32 130
}

attributes #0 = { nounwind memory(read) }
attributes #1 = { nounwind memory(argmem: write) }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn memory(none) }
