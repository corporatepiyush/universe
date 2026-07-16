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

; OS-divergent socket primitives — Linux variant. Symbol-for-symbol twin of
; src/net/osconst_bsd.ll (see its DESIGN block for why this split exists). The
; Makefile links exactly one of the two per host (uname). All values here are
; the Linux/glibc ones:
;     errno accessor = __errno_location()
;     sockaddr_in    = [0..1]=sin_family(u16=AF_INET=2, LE) [2..3]=sin_port(BE)
;                      [4..7]=sin_addr(BE) [8..15]=0   (no sin_len byte)
;     SOL_SOCKET=1  SO_REUSEADDR=2  O_NONBLOCK=0x800  MSG_DONTWAIT=0x40

declare ptr @__errno_location()
declare i32 @setsockopt(i32, i32, i32, ptr, i32)
declare i16 @llvm.bswap.i16(i16)
declare i32 @llvm.bswap.i32(i32)

; thread-local errno value (glibc/musl __errno_location)
define i32 @universe_net_os_errno() #0 {
entry:
  %ep = call ptr @__errno_location()
  %e = load i32, ptr %ep, align 4
  ret i32 %e
}

; build a Linux sockaddr_in into a caller-owned 16-byte buffer
define void @universe_net_os_build_sockaddr(ptr %sa, i32 %ip, i32 %port) #1 {
entry:
  store i64 0, ptr %sa, align 8
  %z8 = getelementptr inbounds nuw i8, ptr %sa, i64 8
  store i64 0, ptr %z8, align 8
  ; sin_family = AF_INET (2) as a little-endian u16 at offset 0 (no sin_len)
  store i16 2, ptr %sa, align 2
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

; SO_REUSEADDR(2) at SOL_SOCKET(1); returns the raw setsockopt rc (<0 = err)
define i32 @universe_net_os_enable_reuseaddr(i32 %fd) #2 {
entry:
  %one.p = alloca i32, align 4
  store i32 1, ptr %one.p, align 4
  %r = call i32 @setsockopt(i32 %fd, i32 1, i32 2, ptr nonnull %one.p, i32 4)
  ret i32 %r
}

; apply O_NONBLOCK(0x800 Linux) to fcntl F_GETFL flags, returning the F_SETFL value
define i32 @universe_net_os_nonblock_flags(i32 %fl, i32 %on) #3 {
entry:
  %want = icmp ne i32 %on, 0
  %with = or i32 %fl, 2048
  %without = and i32 %fl, -2049
  %newfl = select i1 %want, i32 %with, i32 %without
  ret i32 %newfl
}

; MSG_PEEK(2) | MSG_DONTWAIT(0x40 Linux) = 0x42 for a non-blocking liveness peek
define i32 @universe_net_os_msg_peek_dontwait() #3 {
entry:
  ret i32 66
}

attributes #0 = { nounwind memory(read) }
attributes #1 = { nounwind memory(argmem: write) }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn memory(none) }
