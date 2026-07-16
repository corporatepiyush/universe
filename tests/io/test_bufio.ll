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

; Test driver for src/io/bufio.ll. Uses pipe(2) for real fds (portable; no
; platform-specific open flags), and a pthread feeder for the large --bench
; run so the pipe never blocks.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

declare i64 @universe_io_scan_byte(ptr, i64, i8)
declare ptr @universe_io_reader_create(i32, i64)
declare void @universe_io_reader_destroy(ptr)
declare i64 @universe_io_reader_fill(ptr)
declare i64 @universe_io_reader_read(ptr, ptr, i64)
declare i32 @universe_io_reader_read_exact(ptr, ptr, i64)
declare i32 @universe_io_reader_read_byte(ptr)
declare ptr @universe_io_reader_peek(ptr, i64, ptr)
declare i32 @universe_io_reader_read_until(ptr, i8, ptr, ptr)
declare i32 @universe_io_reader_read_line(ptr, ptr, ptr)

declare ptr @universe_io_writer_create(i32, i64)
declare void @universe_io_writer_destroy(ptr)
declare i32 @universe_io_writer_flush(ptr)
declare i64 @universe_io_writer_write(ptr, ptr, i64)
declare i32 @universe_io_writer_write_byte(ptr, i8)
declare i32 @universe_io_writer_write_all(ptr, ptr, i64)
declare i32 @universe_io_writer_flush_vectored(ptr, ptr, i32)

declare i32 @pipe(ptr)
declare i32 @close(i32)
declare i64 @read(i32, ptr, i64)
declare i64 @write(i32, ptr, i64)
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @memset(ptr, i32, i64)
declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(ptr, ptr)
declare i32 @printf(ptr, ...)

@msg.scan = private unnamed_addr constant [22 x i8] c"scan vector==scalar\00\00\00", align 1
@msg.scan0 = private unnamed_addr constant [16 x i8] c"scan at index 0\00", align 1
@msg.scan16 = private unnamed_addr constant [17 x i8] c"scan at index 16\00", align 1
@msg.scannone = private unnamed_addr constant [14 x i8] c"scan no match\00", align 1
@msg.rbyte = private unnamed_addr constant [14 x i8] c"read_byte[0]\00\00", align 1
@msg.rexact = private unnamed_addr constant [19 x i8] c"read_exact xrefill\00", align 1
@msg.peekv = private unnamed_addr constant [14 x i8] c"peek noconsum\00", align 1
@msg.peekc = private unnamed_addr constant [13 x i8] c"peek content\00", align 1
@msg.rrest = private unnamed_addr constant [13 x i8] c"read to EOF \00", align 1
@msg.lineln = private unnamed_addr constant [11 x i8] c"line len  \00", align 1
@msg.linect = private unnamed_addr constant [13 x i8] c"line content\00", align 1
@msg.lineeof = private unnamed_addr constant [13 x i8] c"final EOF ln\00", align 1
@msg.lineend = private unnamed_addr constant [12 x i8] c"empty @ EOF\00", align 1
@msg.fulltot = private unnamed_addr constant [16 x i8] c"read_until FULL\00", align 1
@msg.fullst = private unnamed_addr constant [16 x i8] c"FULL final stat\00", align 1
@msg.wsimple = private unnamed_addr constant [14 x i8] c"writer bufmix\00", align 1
@msg.wvec = private unnamed_addr constant [15 x i8] c"flush_vectored\00", align 1
@msg.wvlen = private unnamed_addr constant [14 x i8] c"vectored len \00", align 1
@msg.benchcnt = private unnamed_addr constant [15 x i8] c"bench line cnt\00", align 1
@bufio.samp = internal global [16 x double] zeroinitializer, align 8
@naive.samp = internal global [16 x double] zeroinitializer, align 8
@bufio.lc = internal global i64 0, align 8
@bufio.nc = internal global i64 0, align 8
@lbl.bufio = private unnamed_addr constant [20 x i8] c"bufio read_line 2MB\00"
@lbl.naive = private unnamed_addr constant [20 x i8] c"naive read_byte 2MB\00"

; --- scalar reference (the oracle) --------------------------------------
define i64 @scan_scalar(ptr %base, i64 %len, i8 %delim) {
entry:
  br label %head
head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %done = icmp uge i64 %i, %len
  br i1 %done, label %none, label %body
body:
  %p = getelementptr inbounds i8, ptr %base, i64 %i
  %b = load i8, ptr %p, align 1
  %m = icmp eq i8 %b, %delim
  br i1 %m, label %found, label %cont
found:
  ret i64 %i
cont:
  %i.n = add i64 %i, 1
  br label %head
none:
  ret i64 %len
}

; write all n bytes to fd, handling partial writes. returns n or -1.
define i64 @write_all_fd(i32 %fd, ptr %buf, i64 %n) {
entry:
  br label %head
head:
  %off = phi i64 [ 0, %entry ], [ %off.n, %adv ]
  %rem = sub i64 %n, %off
  %done = icmp eq i64 %rem, 0
  br i1 %done, label %ok, label %do
do:
  %p = getelementptr inbounds i8, ptr %buf, i64 %off
  %k = call i64 @write(i32 %fd, ptr %p, i64 %rem)
  %bad = icmp slt i64 %k, 1
  br i1 %bad, label %err, label %adv
err:
  ret i64 -1
adv:
  %off.n = add i64 %off, %k
  br label %head
ok:
  ret i64 %n
}

; drain fd into buf (cap bytes) until EOF; returns total bytes read.
define i64 @read_all_fd(i32 %fd, ptr %buf, i64 %cap) {
entry:
  br label %head
head:
  %off = phi i64 [ 0, %entry ], [ %off.n, %adv ]
  %space = sub i64 %cap, %off
  %full = icmp eq i64 %space, 0
  br i1 %full, label %done, label %do
do:
  %p = getelementptr inbounds i8, ptr %buf, i64 %off
  %k = call i64 @read(i32 %fd, ptr %p, i64 %space)
  %eof = icmp sle i64 %k, 0
  br i1 %eof, label %done, label %adv
adv:
  %off.n = add i64 %off, %k
  br label %head
done:
  ret i64 %off
}

; buf[i] = i & 0xff
define void @fill_seq(ptr %buf, i64 %n) {
entry:
  br label %head
head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %body ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %ret, label %body
body:
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %b = trunc i64 %i to i8
  store i8 %b, ptr %p, align 1
  %i.n = add i64 %i, 1
  br label %head
ret:
  ret void
}

; return 1 if all n bytes at p equal ch
define i1 @all_eq(ptr %p, i8 %ch, i64 %n) {
entry:
  br label %head
head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %done = icmp uge i64 %i, %n
  br i1 %done, label %ok, label %body
body:
  %pp = getelementptr inbounds i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %eq = icmp eq i8 %b, %ch
  br i1 %eq, label %cont, label %bad
bad:
  ret i1 false
cont:
  %i.n = add i64 %i, 1
  br label %head
ok:
  ret i1 true
}

; pthread feeder: arg = { i32 fd@0, ptr buf@8, i64 n@16 }; write all, close.
define ptr @writer_thread(ptr %arg) {
entry:
  %fd = load i32, ptr %arg, align 4
  %bufp = getelementptr inbounds i8, ptr %arg, i64 8
  %buf = load ptr, ptr %bufp, align 8
  %np = getelementptr inbounds i8, ptr %arg, i64 16
  %n = load i64, ptr %np, align 8
  %w = call i64 @write_all_fd(i32 %fd, ptr %buf, i64 %n)
  %c = call i32 @close(i32 %fd)
  ret ptr null
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  %scanbuf = alloca [4096 x i8], align 16
  %outp = alloca ptr, align 8
  %outl = alloca i64, align 8
  %avail = alloca i64, align 8
  %fds = alloca [2 x i32], align 4
  %tmp = alloca [256 x i8], align 16

  ; ---- Test A: SIMD scan cross-check vs scalar oracle ----
  br label %scan.head
scan.head:
  %it = phi i64 [ 0, %entry ], [ %it.n, %scan.cont ]
  %mism = phi i64 [ 0, %entry ], [ %mism.n, %scan.cont ]
  %it.done = icmp uge i64 %it, 1500
  br i1 %it.done, label %scan.fill.done, label %scan.gen
scan.gen:
  ; length in [0,4096]
  %rl = call i64 @ut_rand(ptr %seed)
  %len = urem i64 %rl, 4097
  ; delimiter byte (small alphabet so hits are frequent)
  %rd = call i64 @ut_rand(ptr %seed)
  %dmod = urem i64 %rd, 8
  %delim8 = trunc i64 %dmod to i8
  ; fill buffer with bytes in [0,8)
  br label %scan.fill
scan.fill:
  %fi = phi i64 [ 0, %scan.gen ], [ %fi.n, %scan.fbody ]
  %fdone = icmp uge i64 %fi, %len
  br i1 %fdone, label %scan.run, label %scan.fbody
scan.fbody:
  %rv = call i64 @ut_rand(ptr %seed)
  %rvm = urem i64 %rv, 8
  %rvb = trunc i64 %rvm to i8
  %fp = getelementptr inbounds [4096 x i8], ptr %scanbuf, i64 0, i64 %fi
  store i8 %rvb, ptr %fp, align 1
  %fi.n = add i64 %fi, 1
  br label %scan.fill
scan.run:
  %vp = getelementptr inbounds [4096 x i8], ptr %scanbuf, i64 0, i64 0
  %vres = call i64 @universe_io_scan_byte(ptr %vp, i64 %len, i8 %delim8)
  %sres = call i64 @scan_scalar(ptr %vp, i64 %len, i8 %delim8)
  %eq = icmp eq i64 %vres, %sres
  br i1 %eq, label %scan.cont, label %scan.bad
scan.bad:
  %mism.b = add i64 %mism, 1
  br label %scan.cont
scan.cont:
  %mism.n = phi i64 [ %mism, %scan.run ], [ %mism.b, %scan.bad ]
  %it.n = add i64 %it, 1
  br label %scan.head
scan.fill.done:
  call void @ut_check_eq(i64 %mism, i64 0, ptr @msg.scan)

  ; explicit edge positions using a known buffer
  %kp = getelementptr inbounds [256 x i8], ptr %tmp, i64 0, i64 0
  call ptr @memset(ptr %kp, i32 65, i64 64)          ; 64 * 'A'
  %k0 = getelementptr inbounds i8, ptr %kp, i64 0
  store i8 88, ptr %k0, align 1                       ; 'X' at index 0
  %r0 = call i64 @universe_io_scan_byte(ptr %kp, i64 64, i8 88)
  call void @ut_check_eq(i64 %r0, i64 0, ptr @msg.scan0)
  store i8 65, ptr %k0, align 1                       ; restore
  %k16 = getelementptr inbounds i8, ptr %kp, i64 16
  store i8 88, ptr %k16, align 1                      ; 'X' at index 16 (2nd vec)
  %r16 = call i64 @universe_io_scan_byte(ptr %kp, i64 64, i8 88)
  call void @ut_check_eq(i64 %r16, i64 16, ptr @msg.scan16)
  store i8 65, ptr %k16, align 1
  %rn = call i64 @universe_io_scan_byte(ptr %kp, i64 64, i8 88)   ; none
  call void @ut_check_eq(i64 %rn, i64 64, ptr @msg.scannone)

  ; ---- Test B: reader content over a pipe (tiny buffer -> many refills) ----
  %data = call ptr @malloc(i64 2000)
  call void @fill_seq(ptr %data, i64 2000)
  %pr = call i32 @pipe(ptr %fds)
  %rfd.p = getelementptr inbounds [2 x i32], ptr %fds, i64 0, i64 0
  %wfd.p = getelementptr inbounds [2 x i32], ptr %fds, i64 0, i64 1
  %rfd = load i32, ptr %rfd.p, align 4
  %wfd = load i32, ptr %wfd.p, align 4
  %wall = call i64 @write_all_fd(i32 %wfd, ptr %data, i64 2000)
  %cw = call i32 @close(i32 %wfd)
  %rdr = call ptr @universe_io_reader_create(i32 %rfd, i64 16)

  ; read_byte -> data[0] == 0
  %b0 = call i32 @universe_io_reader_read_byte(ptr %rdr)
  %b0ok = icmp eq i32 %b0, 0
  call void @ut_check(i1 %b0ok, ptr @msg.rbyte)

  ; read_exact 50 bytes crossing the 16-byte buffer boundary -> data[1..51]
  %rx = call i32 @universe_io_reader_read_exact(ptr %rdr, ptr %tmp, i64 50)
  %rxok0 = icmp eq i32 %rx, 0
  %exp1 = getelementptr inbounds i8, ptr %data, i64 1
  %mc1 = call i32 @memcmp(ptr %tmp, ptr %exp1, i64 50)
  %mc1ok = icmp eq i32 %mc1, 0
  %rxok = and i1 %rxok0, %mc1ok
  call void @ut_check(i1 %rxok, ptr @msg.rexact)

  ; peek 10 (does not consume) -> data[51..61]
  %pv = call ptr @universe_io_reader_peek(ptr %rdr, i64 10, ptr %avail)
  %av = load i64, ptr %avail, align 8
  %avok = icmp uge i64 %av, 10
  call void @ut_check(i1 %avok, ptr @msg.peekv)
  %exp51 = getelementptr inbounds i8, ptr %data, i64 51
  %mcp = call i32 @memcmp(ptr %pv, ptr %exp51, i64 10)
  %mcpok = icmp eq i32 %mcp, 0
  call void @ut_check(i1 %mcpok, ptr @msg.peekc)

  ; read the remaining 1949 bytes (51..2000) and compare
  %rest = call i64 @universe_io_reader_read(ptr %rdr, ptr %tmp, i64 200)
  ; (only 200 into tmp; verify prefix matches data[51..251] and count)
  %mcr = call i32 @memcmp(ptr %tmp, ptr %exp51, i64 200)
  %mcrok0 = icmp eq i32 %mcr, 0
  %restok = icmp eq i64 %rest, 200
  %mcrok = and i1 %mcrok0, %restok
  call void @ut_check(i1 %mcrok, ptr @msg.rrest)
  call void @universe_io_reader_destroy(ptr %rdr)
  %crd = call i32 @close(i32 %rfd)
  call void @free(ptr %data)

  ; ---- Test C: read_line / read_until over buffer boundaries ----
  ; build 74-byte buffer: aaa\n + b*20\n + c*5\n + d*40\n + ee(no nl)
  %ld = call ptr @malloc(i64 74)
  call ptr @memset(ptr %ld, i32 97, i64 3)
  %ld3 = getelementptr inbounds i8, ptr %ld, i64 3
  store i8 10, ptr %ld3, align 1
  %ld4 = getelementptr inbounds i8, ptr %ld, i64 4
  call ptr @memset(ptr %ld4, i32 98, i64 20)
  %ld24 = getelementptr inbounds i8, ptr %ld, i64 24
  store i8 10, ptr %ld24, align 1
  %ld25 = getelementptr inbounds i8, ptr %ld, i64 25
  call ptr @memset(ptr %ld25, i32 99, i64 5)
  %ld30 = getelementptr inbounds i8, ptr %ld, i64 30
  store i8 10, ptr %ld30, align 1
  %ld31 = getelementptr inbounds i8, ptr %ld, i64 31
  call ptr @memset(ptr %ld31, i32 100, i64 40)
  %ld71 = getelementptr inbounds i8, ptr %ld, i64 71
  store i8 10, ptr %ld71, align 1
  %ld72 = getelementptr inbounds i8, ptr %ld, i64 72
  call ptr @memset(ptr %ld72, i32 101, i64 2)

  %pr2 = call i32 @pipe(ptr %fds)
  %rfd2 = load i32, ptr %rfd.p, align 4
  %wfd2 = load i32, ptr %wfd.p, align 4
  %wl = call i64 @write_all_fd(i32 %wfd2, ptr %ld, i64 74)
  %cw2 = call i32 @close(i32 %wfd2)
  %rdr2 = call ptr @universe_io_reader_create(i32 %rfd2, i64 64)

  ; line 0: "aaa" len 3
  %s0 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l0 = load i64, ptr %outl, align 8
  %p0 = load ptr, ptr %outp, align 8
  call void @ut_check_eq(i64 %l0, i64 3, ptr @msg.lineln)
  %e0 = call i1 @all_eq(ptr %p0, i8 97, i64 %l0)
  call void @ut_check(i1 %e0, ptr @msg.linect)

  ; line 1: 'b'*20 len 20
  %s1 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l1 = load i64, ptr %outl, align 8
  %p1 = load ptr, ptr %outp, align 8
  call void @ut_check_eq(i64 %l1, i64 20, ptr @msg.lineln)
  %e1 = call i1 @all_eq(ptr %p1, i8 98, i64 %l1)
  call void @ut_check(i1 %e1, ptr @msg.linect)

  ; line 2: 'c'*5 len 5
  %s2 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l2 = load i64, ptr %outl, align 8
  %p2 = load ptr, ptr %outp, align 8
  call void @ut_check_eq(i64 %l2, i64 5, ptr @msg.lineln)
  %e2 = call i1 @all_eq(ptr %p2, i8 99, i64 %l2)
  call void @ut_check(i1 %e2, ptr @msg.linect)

  ; line 3: 'd'*40 len 40 (straddles the 64-byte buffer -> compaction+refill)
  %s3 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l3 = load i64, ptr %outl, align 8
  %p3 = load ptr, ptr %outp, align 8
  call void @ut_check_eq(i64 %l3, i64 40, ptr @msg.lineln)
  %e3 = call i1 @all_eq(ptr %p3, i8 100, i64 %l3)
  call void @ut_check(i1 %e3, ptr @msg.linect)

  ; line 4: 'e'*2 no newline -> status 4 (EOF), len 2
  %s4 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l4 = load i64, ptr %outl, align 8
  %p4 = load ptr, ptr %outp, align 8
  %s4ok = icmp eq i32 %s4, 4
  %l4ok = icmp eq i64 %l4, 2
  %e4 = call i1 @all_eq(ptr %p4, i8 101, i64 %l4)
  %s4a = and i1 %s4ok, %l4ok
  %s4b = and i1 %s4a, %e4
  call void @ut_check(i1 %s4b, ptr @msg.lineeof)

  ; next read_line: EOF, empty
  %s5 = call i32 @universe_io_reader_read_line(ptr %rdr2, ptr %outp, ptr %outl)
  %l5 = load i64, ptr %outl, align 8
  %s5ok = icmp eq i32 %s5, 4
  %l5ok = icmp eq i64 %l5, 0
  %s5a = and i1 %s5ok, %l5ok
  call void @ut_check(i1 %s5a, ptr @msg.lineend)
  call void @universe_io_reader_destroy(ptr %rdr2)
  %crd2 = call i32 @close(i32 %rfd2)
  call void @free(ptr %ld)

  ; ---- Test D: read_until FULL path (token longer than buffer) ----
  ; data = 'x'*50 + '\n', buffer 16
  %fd_data = call ptr @malloc(i64 51)
  call ptr @memset(ptr %fd_data, i32 120, i64 50)
  %fdl = getelementptr inbounds i8, ptr %fd_data, i64 50
  store i8 10, ptr %fdl, align 1
  %pr3 = call i32 @pipe(ptr %fds)
  %rfd3 = load i32, ptr %rfd.p, align 4
  %wfd3 = load i32, ptr %wfd.p, align 4
  %wf = call i64 @write_all_fd(i32 %wfd3, ptr %fd_data, i64 51)
  %cw3 = call i32 @close(i32 %wfd3)
  %rdr3 = call ptr @universe_io_reader_create(i32 %rfd3, i64 16)
  br label %full.head
full.head:
  %ftot = phi i64 [ 0, %scan.fill.done ], [ %ftot.n, %full.cont ]
  %fbad = phi i64 [ 0, %scan.fill.done ], [ %fbad.n, %full.cont ]
  br label %full.body
full.body:
  %fu = call i32 @universe_io_reader_read_until(ptr %rdr3, i8 10, ptr %outp, ptr %outl)
  %ful = load i64, ptr %outl, align 8
  %fup = load ptr, ptr %outp, align 8
  ; every returned chunk must be all 'x'
  %fxe = call i1 @all_eq(ptr %fup, i8 120, i64 %ful)
  %fxbad = xor i1 %fxe, true
  %fbad.i = zext i1 %fxbad to i64
  %fbad.n = add i64 %fbad, %fbad.i
  %ftot.n = add i64 %ftot, %ful
  %fu.done = icmp eq i32 %fu, 0
  %ferr = icmp eq i32 %fu, 4
  %fstop = or i1 %fu.done, %ferr
  br i1 %fstop, label %full.exit, label %full.cont
full.cont:
  br label %full.head
full.exit:
  call void @ut_check_eq(i64 %ftot.n, i64 50, ptr @msg.fulltot)
  %fstatok = icmp eq i32 %fu, 0
  %fnobad = icmp eq i64 %fbad.n, 0
  %fallok = and i1 %fstatok, %fnobad
  call void @ut_check(i1 %fallok, ptr @msg.fullst)
  call void @universe_io_reader_destroy(ptr %rdr3)
  %crd3 = call i32 @close(i32 %rfd3)
  call void @free(ptr %fd_data)

  ; ---- Test E: writer buffered/direct mix ----
  %wdata = call ptr @malloc(i64 400)
  call void @fill_seq(ptr %wdata, i64 400)
  %pr4 = call i32 @pipe(ptr %fds)
  %rfd4 = load i32, ptr %rfd.p, align 4
  %wfd4 = load i32, ptr %wfd.p, align 4
  %wtr = call ptr @universe_io_writer_create(i32 %wfd4, i64 32)
  ; first 5 bytes via write_byte
  br label %wb.head
wb.head:
  %wi = phi i64 [ 0, %full.exit ], [ %wi.n, %wb.body ]
  %wbd = icmp uge i64 %wi, 5
  br i1 %wbd, label %wb.done, label %wb.body
wb.body:
  %wbp = getelementptr inbounds i8, ptr %wdata, i64 %wi
  %wbv = load i8, ptr %wbp, align 1
  %wbr = call i32 @universe_io_writer_write_byte(ptr %wtr, i8 %wbv)
  %wi.n = add i64 %wi, 1
  br label %wb.head
wb.done:
  ; next 95 bytes as nineteen 5-byte buffered writes
  br label %ws.head
ws.head:
  %si = phi i64 [ 0, %wb.done ], [ %si.n, %ws.body ]
  %sdone = icmp uge i64 %si, 19
  br i1 %sdone, label %ws.done, label %ws.body
ws.body:
  %soff = mul i64 %si, 5
  %soff2 = add i64 %soff, 5
  %sp = getelementptr inbounds i8, ptr %wdata, i64 %soff2
  %swr = call i64 @universe_io_writer_write(ptr %wtr, ptr %sp, i64 5)
  %si.n = add i64 %si, 1
  br label %ws.head
ws.done:
  ; 100 bytes (> cap 32 -> direct)
  %d100 = getelementptr inbounds i8, ptr %wdata, i64 100
  %w100 = call i64 @universe_io_writer_write(ptr %wtr, ptr %d100, i64 100)
  ; 200 bytes (direct)
  %d200 = getelementptr inbounds i8, ptr %wdata, i64 200
  %w200 = call i32 @universe_io_writer_write_all(ptr %wtr, ptr %d200, i64 200)
  %flr = call i32 @universe_io_writer_flush(ptr %wtr)
  call void @universe_io_writer_destroy(ptr %wtr)
  %cw4 = call i32 @close(i32 %wfd4)
  ; read back all 400 and compare
  %back = call i64 @read_all_fd(i32 %rfd4, ptr %tmp, i64 256)
  ; tmp only 256; compare first 256 bytes and that at least 256 were produced
  %mcw = call i32 @memcmp(ptr %tmp, ptr %wdata, i64 200)
  %mcwok0 = icmp eq i32 %mcw, 0
  %backok = icmp eq i64 %back, 256
  %mcwok = and i1 %mcwok0, %backok
  call void @ut_check(i1 %mcwok, ptr @msg.wsimple)
  %crd4 = call i32 @close(i32 %rfd4)
  call void @free(ptr %wdata)

  ; ---- Test F: flush_vectored ----
  ; buffered prefix "AAA"(3) then 3 iovecs: 'h'*5, 'W'*7, 'z'*20
  %pr5 = call i32 @pipe(ptr %fds)
  %rfd5 = load i32, ptr %rfd.p, align 4
  %wfd5 = load i32, ptr %wfd.p, align 4
  %wtr5 = call ptr @universe_io_writer_create(i32 %wfd5, i64 64)
  %pref = alloca [3 x i8], align 1
  call ptr @memset(ptr %pref, i32 65, i64 3)
  %wp = call i64 @universe_io_writer_write(ptr %wtr5, ptr %pref, i64 3)
  %v1 = alloca [5 x i8], align 1
  %v2 = alloca [7 x i8], align 1
  %v3 = alloca [20 x i8], align 1
  call ptr @memset(ptr %v1, i32 104, i64 5)     ; 'h'
  call ptr @memset(ptr %v2, i32 87, i64 7)      ; 'W'
  call ptr @memset(ptr %v3, i32 122, i64 20)    ; 'z'
  %iov = alloca [48 x i8], align 8
  %iov0b = getelementptr inbounds i8, ptr %iov, i64 0
  store ptr %v1, ptr %iov0b, align 8
  %iov0l = getelementptr inbounds i8, ptr %iov, i64 8
  store i64 5, ptr %iov0l, align 8
  %iov1b = getelementptr inbounds i8, ptr %iov, i64 16
  store ptr %v2, ptr %iov1b, align 8
  %iov1l = getelementptr inbounds i8, ptr %iov, i64 24
  store i64 7, ptr %iov1l, align 8
  %iov2b = getelementptr inbounds i8, ptr %iov, i64 32
  store ptr %v3, ptr %iov2b, align 8
  %iov2l = getelementptr inbounds i8, ptr %iov, i64 40
  store i64 20, ptr %iov2l, align 8
  %fvr = call i32 @universe_io_writer_flush_vectored(ptr %wtr5, ptr %iov, i32 3)
  %fvok = icmp eq i32 %fvr, 0
  call void @ut_check(i1 %fvok, ptr @msg.wvec)
  call void @universe_io_writer_destroy(ptr %wtr5)
  %cw5 = call i32 @close(i32 %wfd5)
  ; expect 3 + 5 + 7 + 20 = 35 bytes: AAA hhhhh WWWWWWW zzzz...
  %vb = call i64 @read_all_fd(i32 %rfd5, ptr %tmp, i64 256)
  call void @ut_check_eq(i64 %vb, i64 35, ptr @msg.wvlen)
  %vA = call i1 @all_eq(ptr %tmp, i8 65, i64 3)
  %vh.p = getelementptr inbounds i8, ptr %tmp, i64 3
  %vh = call i1 @all_eq(ptr %vh.p, i8 104, i64 5)
  %vW.p = getelementptr inbounds i8, ptr %tmp, i64 8
  %vW = call i1 @all_eq(ptr %vW.p, i8 87, i64 7)
  %vz.p = getelementptr inbounds i8, ptr %tmp, i64 15
  %vz = call i1 @all_eq(ptr %vz.p, i8 122, i64 20)
  %vc0 = and i1 %vA, %vh
  %vc1 = and i1 %vc0, %vW
  %vc2 = and i1 %vc1, %vz
  call void @ut_check(i1 %vc2, ptr @msg.wvec)
  %crd5 = call i32 @close(i32 %rfd5)

  ; ---- optional bench ----
  %wb2 = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb2, label %bench, label %fin

bench:
  ; 2,000,000 bytes, newline every 40 -> 50,000 lines
  %BN = add i64 0, 2000000
  %big = call ptr @malloc(i64 %BN)
  br label %bg.head
bg.head:
  %bi = phi i64 [ 0, %bench ], [ %bi.n, %bg.body ]
  %bdone = icmp uge i64 %bi, %BN
  br i1 %bdone, label %bg.done, label %bg.body
bg.body:
  %bmod = urem i64 %bi, 40
  %bnl = icmp eq i64 %bmod, 39
  %bch = select i1 %bnl, i8 10, i8 97
  %bp = getelementptr inbounds i8, ptr %big, i64 %bi
  store i8 %bch, ptr %bp, align 1
  %bi.n = add i64 %bi, 1
  br label %bg.head
bg.done:
  %targ = alloca [24 x i8], align 8
  %th = alloca ptr, align 8
  br label %b1.rep.head

  ; ---- run 1: bufio read_line (distribution; ops_per_rep = 2,000,000 bytes) ----
b1.rep.head:
  %b1.rep = phi i64 [ 0, %bg.done ], [ %b1.rep.n, %b1.rep.cont ]
  %bpr1 = call i32 @pipe(ptr %fds)
  %brfd1 = load i32, ptr %rfd.p, align 4
  %bwfd1 = load i32, ptr %wfd.p, align 4
  store i32 %bwfd1, ptr %targ, align 4
  %tbufp = getelementptr inbounds i8, ptr %targ, i64 8
  store ptr %big, ptr %tbufp, align 8
  %tnp = getelementptr inbounds i8, ptr %targ, i64 16
  store i64 %BN, ptr %tnp, align 8
  %pc1 = call i32 @pthread_create(ptr %th, ptr null, ptr @writer_thread, ptr %targ)
  %t0 = call double @ut_now_sec()
  %brdr = call ptr @universe_io_reader_create(i32 %brfd1, i64 65536)
  br label %bl.head
bl.head:
  %lc = phi i64 [ 0, %b1.rep.head ], [ %lc.n, %bl.cont ]
  %ls = call i32 @universe_io_reader_read_line(ptr %brdr, ptr %outp, ptr %outl)
  %ll = load i64, ptr %outl, align 8
  %lgot = icmp eq i32 %ls, 0
  %leof = icmp eq i32 %ls, 4
  %lhaslen = icmp ugt i64 %ll, 0
  %leofline = and i1 %leof, %lhaslen
  %lcount = or i1 %lgot, %leofline
  %lc.i = zext i1 %lcount to i64
  %lc.n = add i64 %lc, %lc.i
  %lstop = or i1 %leof, %lgot
  ; continue while status==0 (got a line); stop on EOF (4) or error
  br i1 %lgot, label %bl.cont, label %bl.check.eof
bl.check.eof:
  br i1 %leof, label %bl.done, label %bl.done
bl.cont:
  br label %bl.head
bl.done:
  %t1 = call double @ut_now_sec()
  %thv1 = load ptr, ptr %th, align 8
  %tj1 = call i32 @pthread_join(ptr %thv1, ptr null)
  call void @universe_io_reader_destroy(ptr %brdr)
  %bcr1 = call i32 @close(i32 %brfd1)
  store i64 %lc, ptr @bufio.lc, align 8
  %b1.dt = fsub double %t1, %t0
  %b1.warm = icmp eq i64 %b1.rep, 0
  br i1 %b1.warm, label %b1.rep.cont, label %b1.rep.store
b1.rep.store:
  %b1.idx = sub i64 %b1.rep, 1
  %b1.sp = getelementptr inbounds [16 x double], ptr @bufio.samp, i64 0, i64 %b1.idx
  store double %b1.dt, ptr %b1.sp, align 8
  br label %b1.rep.cont
b1.rep.cont:
  %b1.rep.n = add i64 %b1.rep, 1
  %b1.more = icmp ult i64 %b1.rep.n, 17
  br i1 %b1.more, label %b1.rep.head, label %b1.rep.end
b1.rep.end:
  call void @ut_report_dist(ptr @bufio.samp, i64 16, i64 2000000, ptr @lbl.bufio)
  br label %b2.rep.head

  ; ---- run 2: naive one-byte reads (distribution; ops_per_rep = 2,000,000 bytes) ----
b2.rep.head:
  %b2.rep = phi i64 [ 0, %b1.rep.end ], [ %b2.rep.n, %b2.rep.cont ]
  %bpr2 = call i32 @pipe(ptr %fds)
  %brfd2 = load i32, ptr %rfd.p, align 4
  %bwfd2 = load i32, ptr %wfd.p, align 4
  store i32 %bwfd2, ptr %targ, align 4
  %tbufp2 = getelementptr inbounds i8, ptr %targ, i64 8
  store ptr %big, ptr %tbufp2, align 8
  %tnp2 = getelementptr inbounds i8, ptr %targ, i64 16
  store i64 %BN, ptr %tnp2, align 8
  %pc2 = call i32 @pthread_create(ptr %th, ptr null, ptr @writer_thread, ptr %targ)
  %t2 = call double @ut_now_sec()
  %nrdr = call ptr @universe_io_reader_create(i32 %brfd2, i64 65536)
  br label %nb.head
nb.head:
  %nc = phi i64 [ 0, %b2.rep.head ], [ %nc.n, %nb.cont ]
  %nby = call i32 @universe_io_reader_read_byte(ptr %nrdr)
  %neof = icmp slt i32 %nby, 0
  br i1 %neof, label %nb.done, label %nb.body
nb.body:
  %nnl = icmp eq i32 %nby, 10
  %nc.i = zext i1 %nnl to i64
  %nc.n = add i64 %nc, %nc.i
  br label %nb.cont
nb.cont:
  br label %nb.head
nb.done:
  %t3 = call double @ut_now_sec()
  %thv2 = load ptr, ptr %th, align 8
  %tj2 = call i32 @pthread_join(ptr %thv2, ptr null)
  call void @universe_io_reader_destroy(ptr %nrdr)
  %bcr2 = call i32 @close(i32 %brfd2)
  store i64 %nc, ptr @bufio.nc, align 8
  %b2.dt = fsub double %t3, %t2
  %b2.warm = icmp eq i64 %b2.rep, 0
  br i1 %b2.warm, label %b2.rep.cont, label %b2.rep.store
b2.rep.store:
  %b2.idx = sub i64 %b2.rep, 1
  %b2.sp = getelementptr inbounds [16 x double], ptr @naive.samp, i64 0, i64 %b2.idx
  store double %b2.dt, ptr %b2.sp, align 8
  br label %b2.rep.cont
b2.rep.cont:
  %b2.rep.n = add i64 %b2.rep, 1
  %b2.more = icmp ult i64 %b2.rep.n, 17
  br i1 %b2.more, label %b2.rep.head, label %b2.rep.end
b2.rep.end:
  call void @ut_report_dist(ptr @naive.samp, i64 16, i64 2000000, ptr @lbl.naive)

  ; line counts must agree (bufio read_line vs naive byte scan)
  %lc.f = load i64, ptr @bufio.lc, align 8
  %nc.f = load i64, ptr @bufio.nc, align 8
  call void @ut_check_eq(i64 %lc.f, i64 %nc.f, ptr @msg.benchcnt)
  call void @free(ptr %big)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
