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

; Buffered reader + writer over raw fds (regular files, pipes, sockets).
;
; DESIGN:
;   Both a reader and a writer own ONE allocation: a <=1 cache-line header at
;   +0 followed by the data buffer at +64. No pointer webs; the buffer never
;   shares a line with the hot cursors.
;
;   Reader layout (bytes): { fd@0 (i64), rpos@8, wpos@16, cap@24 }, buf@64.
;     * Linear-refill scheme: [rpos, wpos) is the unconsumed window inside the
;       buffer; both are absolute byte indices in [0, cap]. Empty <=> rpos==wpos.
;     * fill/read/read_byte pull one read(2) into the buffer when it drains.
;       Large reads (>= cap) bypass the buffer and read(2) straight into the
;       caller's dst — no double copy (keeps IO out of the compute path).
;     * peek / read_until / read_line return ZERO-COPY views into the buffer.
;       When a token straddles the current window the buffer is COMPACTED
;       (memmove the tail to +0) then refilled, so the token becomes contiguous
;       and a slice can be handed back. A token longer than the whole buffer
;       cannot be viewed zero-copy: read_until returns FULL(6) with the buffer
;       contents so the caller can drain and retry.
;     * Views are valid only until the next reader call that may refill/compact.
;
;   Writer layout (bytes): { fd@0 (i64), len@8, cap@16 }, buf@64.
;     * write() accumulates small pieces and flushes (one write(2)) when the
;       buffer fills; writes >= cap bypass the buffer and go straight to the fd.
;       Never one syscall per byte.
;     * flush drains with a partial-write loop. flush_vectored assembles the
;       scattered pieces and issues writev(2) once (with a partial-writev retry
;       loop over a private copy of the iovec array), after flushing the
;       internal buffer to preserve byte order.
;
;   SIMD/scalar split (universe_io_scan_byte, the delimiter kernel):
;     * PRIMARY vector path: load 16 bytes as <16 x i8>, `icmp eq` against the
;       splatted delimiter, `bitcast <16 x i1> -> i16` gives a movemask, and
;       llvm.cttz.i16 locates the first hit. Branch-free inside the 16-lane body
;       (one predictable "any hit?" test per chunk). Lowers to NEON cmeq+addv /
;       SSE2 pcmpeqb+pmovmskb.
;     * SCALAR fallback: the < 16 byte tail, and the cross-check oracle in tests.
;
;   Error convention (i32 error codes: 0 OK, 1 NULL_PTR, 4 EMPTY/EOF, 6 FULL,
;   8 INVALID_ARG, 15 IO). ssize_t-style APIs (read/fill/write) return a
;   non-negative byte count on success or a negative error: -1 NULL, -15 IO;
;   read_byte returns the byte 0..255, -1 on EOF, -15 on IO error.

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i16 @llvm.cttz.i16(i16, i1 immarg)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

declare i64 @read(i32, ptr, i64) nounwind
declare i64 @write(i32, ptr, i64) nounwind
declare i64 @writev(i32, ptr, i32) nounwind

; ---------------------------------------------------------------------------
; universe_io_scan_byte(base, len, delim) -> i64
;   Index of the first byte == delim in [base, base+len), or `len` if none.
;   SIMD-first: 16-wide vector compare + movemask; scalar tail for the < 16
;   remainder. Also the public delimiter kernel used by the reader.
; ---------------------------------------------------------------------------
define i64 @universe_io_scan_byte(ptr readonly captures(none) %base, i64 %len, i8 %delim) local_unnamed_addr #0 {
entry:
  %dv0 = insertelement <16 x i8> poison, i8 %delim, i64 0
  %dv = shufflevector <16 x i8> %dv0, <16 x i8> poison, <16 x i32> zeroinitializer
  br label %vloop

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %rem = sub i64 %len, %i
  %has16 = icmp uge i64 %rem, 16
  br i1 %has16, label %vec, label %tail

vec:
  %vp = getelementptr inbounds nuw i8, ptr %base, i64 %i
  %g = load <16 x i8>, ptr %vp, align 1
  %eq = icmp eq <16 x i8> %g, %dv
  %mm = bitcast <16 x i1> %eq to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %vfound, label %vcont

vfound:
  %bit = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %bit64 = zext i16 %bit to i64
  %vidx = add i64 %i, %bit64
  ret i64 %vidx

vcont:
  %i.next = add nuw i64 %i, 16
  br label %vloop

tail:
  br label %thead

thead:
  %j = phi i64 [ %i, %tail ], [ %j.next, %tcont ]
  %tdone = icmp uge i64 %j, %len
  br i1 %tdone, label %none, label %tbody

tbody:
  %tp = getelementptr inbounds nuw i8, ptr %base, i64 %j
  %tb = load i8, ptr %tp, align 1
  %tmatch = icmp eq i8 %tb, %delim
  br i1 %tmatch, label %tfound, label %tcont

tfound:
  ret i64 %j

tcont:
  %j.next = add nuw i64 %j, 1
  br label %thead

none:
  ret i64 %len
}

; ---------------------------------------------------------------------------
; Reader
; ---------------------------------------------------------------------------

; universe_io_reader_create(fd, bufsize) -> ptr (null on failure)
;   bufsize 0 => 65536 default; otherwise rounded UP to a power of two (min 8).
define noalias ptr @universe_io_reader_create(i32 %fd, i64 %bufsize) local_unnamed_addr #2 {
entry:
  %is0 = icmp eq i64 %bufsize, 0
  %req = select i1 %is0, i64 65536, i64 %bufsize
  %too.big = icmp ugt i64 %req, 1099511627776
  br i1 %too.big, label %fail, label %shape, !prof !0

shape:
  %req2 = call i64 @llvm.umax.i64(i64 %req, i64 8)
  %m1 = add i64 %req2, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %m1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %cap, i64 64)
  %total = extractvalue { i64, i1 } %tot, 0
  %ovf = extractvalue { i64, i1 } %tot, 1
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  %fd64 = zext i32 %fd to i64
  store i64 %fd64, ptr %mem, align 8
  %rpos.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %rpos.p, align 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 0, ptr %wpos.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %cap, ptr %cap.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define void @universe_io_reader_destroy(ptr %r) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %r, null
  br i1 %is.null, label %done, label %do.free, !prof !0
do.free:
  call void @free(ptr nonnull %r)
  br label %done
done:
  ret void
}

; universe_io_reader_fill(r) -> i64 available (0 = EOF, -1 null, -15 IO)
define i64 @universe_io_reader_fill(ptr %r) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  br i1 %rnull, label %err.null, label %body, !prof !0
err.null:
  ret i64 -1
body:
  %rpos.p = getelementptr inbounds nuw i8, ptr %r, i64 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %r, i64 16
  %rpos = load i64, ptr %rpos.p, align 8
  %wpos = load i64, ptr %wpos.p, align 8
  %avail = sub i64 %wpos, %rpos
  %have = icmp ugt i64 %avail, 0
  br i1 %have, label %ret.avail, label %refill
ret.avail:
  ret i64 %avail
refill:
  %fd64 = load i64, ptr %r, align 8
  %fd = trunc i64 %fd64 to i32
  %cap.p = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %buf = getelementptr inbounds nuw i8, ptr %r, i64 64
  %k = call i64 @read(i32 %fd, ptr %buf, i64 %cap)
  %kneg = icmp slt i64 %k, 0
  br i1 %kneg, label %err.io, label %store
err.io:
  ret i64 -15
store:
  store i64 0, ptr %rpos.p, align 8
  store i64 %k, ptr %wpos.p, align 8
  ret i64 %k
}

; universe_io_reader_read(r, dst, n) -> i64 bytes (short = EOF), -1 null, -15 IO
define i64 @universe_io_reader_read(ptr %r, ptr %dst, i64 %n) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  %dnull = icmp eq ptr %dst, null
  %anynull = or i1 %rnull, %dnull
  br i1 %anynull, label %err.null, label %setup, !prof !0
err.null:
  ret i64 -1
setup:
  %npos = icmp sgt i64 %n, 0
  br i1 %npos, label %pre, label %ret.zero
ret.zero:
  ret i64 0
pre:
  %rpos.p = getelementptr inbounds nuw i8, ptr %r, i64 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %r, i64 16
  %cap.p = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %buf = getelementptr inbounds nuw i8, ptr %r, i64 64
  %fd64 = load i64, ptr %r, align 8
  %fd = trunc i64 %fd64 to i32
  br label %loop

loop:
  %total = phi i64 [ 0, %pre ], [ %total, %f.store ], [ %total.d, %d.adv ], [ %total.c, %copy ]
  %done = icmp uge i64 %total, %n
  br i1 %done, label %ret.total, label %check
ret.total:
  ret i64 %total
check:
  %rpos = load i64, ptr %rpos.p, align 8
  %wpos = load i64, ptr %wpos.p, align 8
  %avail = sub i64 %wpos, %rpos
  %hasdata = icmp ugt i64 %avail, 0
  br i1 %hasdata, label %copy, label %refill

refill:
  %need = sub i64 %n, %total
  %bigread = icmp uge i64 %need, %cap
  br i1 %bigread, label %direct, label %fillbuf

direct:
  %ddst = getelementptr inbounds nuw i8, ptr %dst, i64 %total
  %dk = call i64 @read(i32 %fd, ptr %ddst, i64 %need)
  %dkneg = icmp slt i64 %dk, 0
  br i1 %dkneg, label %d.err, label %d.chk
d.err:
  %dgot = icmp ugt i64 %total, 0
  %dret = select i1 %dgot, i64 %total, i64 -15
  ret i64 %dret
d.chk:
  %deof = icmp eq i64 %dk, 0
  br i1 %deof, label %ret.total, label %d.adv
d.adv:
  %total.d = add i64 %total, %dk
  br label %loop

fillbuf:
  %fk = call i64 @read(i32 %fd, ptr %buf, i64 %cap)
  %fkneg = icmp slt i64 %fk, 0
  br i1 %fkneg, label %f.err, label %f.chk
f.err:
  %fgot = icmp ugt i64 %total, 0
  %fret = select i1 %fgot, i64 %total, i64 -15
  ret i64 %fret
f.chk:
  %feof = icmp eq i64 %fk, 0
  br i1 %feof, label %ret.total, label %f.store
f.store:
  store i64 0, ptr %rpos.p, align 8
  store i64 %fk, ptr %wpos.p, align 8
  br label %loop

copy:
  %remaining = sub i64 %n, %total
  %take = call i64 @llvm.umin.i64(i64 %remaining, i64 %avail)
  %cdst = getelementptr inbounds nuw i8, ptr %dst, i64 %total
  %csrc = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos
  call void @llvm.memcpy.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %take, i1 false)
  %rpos.n = add i64 %rpos, %take
  store i64 %rpos.n, ptr %rpos.p, align 8
  %total.c = add i64 %total, %take
  br label %loop
}

; universe_io_reader_read_exact(r, dst, n) -> i32 (0 OK, 1 null, 4 short/EOF, 15 IO)
define i32 @universe_io_reader_read_exact(ptr %r, ptr %dst, i64 %n) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  %dnull = icmp eq ptr %dst, null
  %anynull = or i1 %rnull, %dnull
  br i1 %anynull, label %err.null, label %body, !prof !0
err.null:
  ret i32 1
body:
  %got = call i64 @universe_io_reader_read(ptr %r, ptr %dst, i64 %n)
  %ioerr = icmp slt i64 %got, 0
  br i1 %ioerr, label %err.io, label %chk
err.io:
  ret i32 15
chk:
  %eq = icmp eq i64 %got, %n
  br i1 %eq, label %ok, label %short
ok:
  ret i32 0
short:
  ret i32 4
}

; universe_io_reader_read_byte(r) -> i32 (0..255, -1 EOF, -15 IO/null)
define i32 @universe_io_reader_read_byte(ptr %r) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  br i1 %rnull, label %err, label %body, !prof !0
err:
  ret i32 -15
body:
  %rpos.p = getelementptr inbounds nuw i8, ptr %r, i64 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %r, i64 16
  %rpos = load i64, ptr %rpos.p, align 8
  %wpos = load i64, ptr %wpos.p, align 8
  %avail = sub i64 %wpos, %rpos
  %empty = icmp eq i64 %avail, 0
  br i1 %empty, label %refill, label %take
refill:
  %fd64 = load i64, ptr %r, align 8
  %fd = trunc i64 %fd64 to i32
  %cap.p = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %buf0 = getelementptr inbounds nuw i8, ptr %r, i64 64
  %k = call i64 @read(i32 %fd, ptr %buf0, i64 %cap)
  %kneg = icmp slt i64 %k, 0
  br i1 %kneg, label %err, label %chk.eof
chk.eof:
  %eof = icmp eq i64 %k, 0
  br i1 %eof, label %ret.eof, label %filled
ret.eof:
  ret i32 -1
filled:
  store i64 0, ptr %rpos.p, align 8
  store i64 %k, ptr %wpos.p, align 8
  br label %take
take:
  %rp = phi i64 [ %rpos, %body ], [ 0, %filled ]
  %buf = getelementptr inbounds nuw i8, ptr %r, i64 64
  %bp = getelementptr inbounds nuw i8, ptr %buf, i64 %rp
  %b = load i8, ptr %bp, align 1
  %rp.n = add i64 %rp, 1
  store i64 %rp.n, ptr %rpos.p, align 8
  %bz = zext i8 %b to i32
  ret i32 %bz
}

; universe_io_reader_peek(r, n, out_avail) -> ptr view (null on error)
;   Ensures up to min(n, cap) bytes are contiguous at the returned pointer;
;   *out_avail = bytes actually available. Does NOT consume.
define ptr @universe_io_reader_peek(ptr %r, i64 %n, ptr %out_avail) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  %onull = icmp eq ptr %out_avail, null
  %anynull = or i1 %rnull, %onull
  br i1 %anynull, label %err, label %setup, !prof !0
err:
  ret ptr null
setup:
  %rpos.p = getelementptr inbounds nuw i8, ptr %r, i64 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %r, i64 16
  %cap.p = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %buf = getelementptr inbounds nuw i8, ptr %r, i64 64
  %fd64 = load i64, ptr %r, align 8
  %fd = trunc i64 %fd64 to i32
  %want = call i64 @llvm.umin.i64(i64 %n, i64 %cap)
  br label %loop
loop:
  %rpos = load i64, ptr %rpos.p, align 8
  %wpos = load i64, ptr %wpos.p, align 8
  %avail = sub i64 %wpos, %rpos
  %enough = icmp uge i64 %avail, %want
  br i1 %enough, label %ret.view, label %grow
grow:
  %need.compact = icmp ugt i64 %rpos, 0
  br i1 %need.compact, label %compact, label %after
compact:
  %csrc = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos
  call void @llvm.memmove.p0.p0.i64(ptr %buf, ptr %csrc, i64 %avail, i1 false)
  store i64 0, ptr %rpos.p, align 8
  store i64 %avail, ptr %wpos.p, align 8
  br label %after
after:
  %rpos2 = load i64, ptr %rpos.p, align 8
  %wpos2 = load i64, ptr %wpos.p, align 8
  %space = sub i64 %cap, %wpos2
  %full = icmp eq i64 %space, 0
  br i1 %full, label %ret.view, label %doread
doread:
  %dst = getelementptr inbounds nuw i8, ptr %buf, i64 %wpos2
  %k = call i64 @read(i32 %fd, ptr %dst, i64 %space)
  %kneg = icmp slt i64 %k, 0
  br i1 %kneg, label %err, label %chk.eof
chk.eof:
  %eof = icmp eq i64 %k, 0
  br i1 %eof, label %ret.view, label %advance
advance:
  %wnew = add i64 %wpos2, %k
  store i64 %wnew, ptr %wpos.p, align 8
  br label %loop
ret.view:
  %frpos = load i64, ptr %rpos.p, align 8
  %fwpos = load i64, ptr %wpos.p, align 8
  %favail = sub i64 %fwpos, %frpos
  store i64 %favail, ptr %out_avail, align 8
  %view = getelementptr inbounds nuw i8, ptr %buf, i64 %frpos
  ret ptr %view
}

; universe_io_reader_read_until(r, delim, out_ptr, out_len) -> i32
;   0  = delim found: [*out_ptr, *out_ptr+*out_len) is the slice before the
;        delimiter; the delimiter is consumed.
;   4  = EOF before any delimiter: *out_len (>=0) is the trailing partial slice.
;   6  = token longer than the buffer: *out_ptr/*out_len give the buffer's worth
;        of data (already consumed); call again to continue.
;   1  = null arg; 15 = IO error.
define i32 @universe_io_reader_read_until(ptr %r, i8 %delim, ptr %out_ptr, ptr %out_len) local_unnamed_addr #1 {
entry:
  %rnull = icmp eq ptr %r, null
  %pnull = icmp eq ptr %out_ptr, null
  %lnull = icmp eq ptr %out_len, null
  %n01 = or i1 %rnull, %pnull
  %anynull = or i1 %n01, %lnull
  br i1 %anynull, label %err.null, label %setup, !prof !0
err.null:
  ret i32 1
setup:
  %rpos.p = getelementptr inbounds nuw i8, ptr %r, i64 8
  %wpos.p = getelementptr inbounds nuw i8, ptr %r, i64 16
  %cap.p = getelementptr inbounds nuw i8, ptr %r, i64 24
  %cap = load i64, ptr %cap.p, align 8
  %buf = getelementptr inbounds nuw i8, ptr %r, i64 64
  %fd64 = load i64, ptr %r, align 8
  %fd = trunc i64 %fd64 to i32
  br label %loop
loop:
  %rpos = load i64, ptr %rpos.p, align 8
  %wpos = load i64, ptr %wpos.p, align 8
  %avail = sub i64 %wpos, %rpos
  %scanbase = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos
  %idx = call i64 @universe_io_scan_byte(ptr %scanbase, i64 %avail, i8 %delim)
  %found = icmp ult i64 %idx, %avail
  br i1 %found, label %hit, label %more
hit:
  store ptr %scanbase, ptr %out_ptr, align 8
  store i64 %idx, ptr %out_len, align 8
  %past = add i64 %idx, 1
  %newr = add i64 %rpos, %past
  store i64 %newr, ptr %rpos.p, align 8
  ret i32 0
more:
  %need.compact = icmp ugt i64 %rpos, 0
  br i1 %need.compact, label %compact, label %after
compact:
  %csrc = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos
  call void @llvm.memmove.p0.p0.i64(ptr %buf, ptr %csrc, i64 %avail, i1 false)
  store i64 0, ptr %rpos.p, align 8
  store i64 %avail, ptr %wpos.p, align 8
  br label %after
after:
  %rpos2 = load i64, ptr %rpos.p, align 8
  %wpos2 = load i64, ptr %wpos.p, align 8
  %space = sub i64 %cap, %wpos2
  %full = icmp eq i64 %space, 0
  br i1 %full, label %toolong, label %doread
toolong:
  %tview = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos2
  %tlen = sub i64 %wpos2, %rpos2
  store ptr %tview, ptr %out_ptr, align 8
  store i64 %tlen, ptr %out_len, align 8
  store i64 %wpos2, ptr %rpos.p, align 8
  ret i32 6
doread:
  %dst = getelementptr inbounds nuw i8, ptr %buf, i64 %wpos2
  %k = call i64 @read(i32 %fd, ptr %dst, i64 %space)
  %kneg = icmp slt i64 %k, 0
  br i1 %kneg, label %err.io, label %chk.eof
err.io:
  ret i32 15
chk.eof:
  %eof = icmp eq i64 %k, 0
  br i1 %eof, label %ateof, label %advance
ateof:
  %eview = getelementptr inbounds nuw i8, ptr %buf, i64 %rpos2
  %elen = sub i64 %wpos2, %rpos2
  store ptr %eview, ptr %out_ptr, align 8
  store i64 %elen, ptr %out_len, align 8
  store i64 %wpos2, ptr %rpos.p, align 8
  ret i32 4
advance:
  %wnew = add i64 %wpos2, %k
  store i64 %wnew, ptr %wpos.p, align 8
  br label %loop
}

; universe_io_reader_read_line(r, out_ptr, out_len) -> i32 (delim '\n')
define i32 @universe_io_reader_read_line(ptr %r, ptr %out_ptr, ptr %out_len) local_unnamed_addr #1 {
entry:
  %s = tail call i32 @universe_io_reader_read_until(ptr %r, i8 10, ptr %out_ptr, ptr %out_len)
  ret i32 %s
}

; ---------------------------------------------------------------------------
; Writer
; ---------------------------------------------------------------------------

; universe_io_writer_create(fd, bufsize) -> ptr (null on failure)
define noalias ptr @universe_io_writer_create(i32 %fd, i64 %bufsize) local_unnamed_addr #2 {
entry:
  %is0 = icmp eq i64 %bufsize, 0
  %req = select i1 %is0, i64 65536, i64 %bufsize
  %too.big = icmp ugt i64 %req, 1099511627776
  br i1 %too.big, label %fail, label %shape, !prof !0
shape:
  %req2 = call i64 @llvm.umax.i64(i64 %req, i64 8)
  %m1 = add i64 %req2, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %m1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %cap, i64 64)
  %total = extractvalue { i64, i1 } %tot, 0
  %ovf = extractvalue { i64, i1 } %tot, 1
  br i1 %ovf, label %fail, label %alloc, !prof !0
alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0
init:
  %fd64 = zext i32 %fd to i64
  store i64 %fd64, ptr %mem, align 8
  %len.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %len.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %cap, ptr %cap.p, align 8
  ret ptr %mem
fail:
  ret ptr null
}

; universe_io_writer_flush(w) -> i32 (0 OK, 1 null, 15 IO)
define i32 @universe_io_writer_flush(ptr %w) local_unnamed_addr #1 {
entry:
  %wnull = icmp eq ptr %w, null
  br i1 %wnull, label %err.null, label %body, !prof !0
err.null:
  ret i32 1
body:
  %len.p = getelementptr inbounds nuw i8, ptr %w, i64 8
  %len = load i64, ptr %len.p, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %ok, label %prep
ok:
  ret i32 0
prep:
  %fd64 = load i64, ptr %w, align 8
  %fd = trunc i64 %fd64 to i32
  %buf = getelementptr inbounds nuw i8, ptr %w, i64 64
  br label %loop
loop:
  %off = phi i64 [ 0, %prep ], [ %off.n, %advance ]
  %rem = sub i64 %len, %off
  %done = icmp eq i64 %rem, 0
  br i1 %done, label %drained, label %do.write
do.write:
  %src = getelementptr inbounds nuw i8, ptr %buf, i64 %off
  %k = call i64 @write(i32 %fd, ptr %src, i64 %rem)
  %bad = icmp slt i64 %k, 1
  br i1 %bad, label %err.io, label %advance
err.io:
  ; keep the unwritten tail so a retry is possible
  %tail.src = getelementptr inbounds nuw i8, ptr %buf, i64 %off
  call void @llvm.memmove.p0.p0.i64(ptr %buf, ptr %tail.src, i64 %rem, i1 false)
  store i64 %rem, ptr %len.p, align 8
  ret i32 15
advance:
  %off.n = add i64 %off, %k
  br label %loop
drained:
  store i64 0, ptr %len.p, align 8
  ret i32 0
}

; universe_io_writer_write(w, src, n) -> i64 bytes written (=n), -1 null, -15 IO
define i64 @universe_io_writer_write(ptr %w, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %wnull = icmp eq ptr %w, null
  %snull = icmp eq ptr %src, null
  %anynull = or i1 %wnull, %snull
  br i1 %anynull, label %err.null, label %setup, !prof !0
err.null:
  ret i64 -1
setup:
  %pos = icmp sgt i64 %n, 0
  br i1 %pos, label %body, label %ret.zero
ret.zero:
  ret i64 0
body:
  %len.p = getelementptr inbounds nuw i8, ptr %w, i64 8
  %cap.p = getelementptr inbounds nuw i8, ptr %w, i64 16
  %len = load i64, ptr %len.p, align 8
  %cap = load i64, ptr %cap.p, align 8
  %buf = getelementptr inbounds nuw i8, ptr %w, i64 64
  %space = sub i64 %cap, %len
  %fits = icmp ule i64 %n, %space
  br i1 %fits, label %buffer, label %spill
buffer:
  %dst = getelementptr inbounds nuw i8, ptr %buf, i64 %len
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  %len.n = add i64 %len, %n
  store i64 %len.n, ptr %len.p, align 8
  ret i64 %n
spill:
  %has.buf = icmp ugt i64 %len, 0
  br i1 %has.buf, label %flush, label %after.flush
flush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  %fbad = icmp ne i32 %fs, 0
  br i1 %fbad, label %err.io, label %after.flush
err.io:
  ret i64 -15
after.flush:
  ; buffer is now empty
  %big = icmp uge i64 %n, %cap
  br i1 %big, label %direct, label %stash
stash:
  call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr %src, i64 %n, i1 false)
  store i64 %n, ptr %len.p, align 8
  ret i64 %n
direct:
  %fd64 = load i64, ptr %w, align 8
  %fd = trunc i64 %fd64 to i32
  br label %dloop
dloop:
  %off = phi i64 [ 0, %direct ], [ %off.n, %dadv ]
  %rem = sub i64 %n, %off
  %ddone = icmp eq i64 %rem, 0
  br i1 %ddone, label %dret, label %dwrite
dwrite:
  %dsrc = getelementptr inbounds nuw i8, ptr %src, i64 %off
  %k = call i64 @write(i32 %fd, ptr %dsrc, i64 %rem)
  %bad = icmp slt i64 %k, 1
  br i1 %bad, label %err.io2, label %dadv
err.io2:
  ret i64 -15
dadv:
  %off.n = add i64 %off, %k
  br label %dloop
dret:
  ret i64 %n
}

; universe_io_writer_write_byte(w, b) -> i32 (0 OK, 1 null, 15 IO)
define i32 @universe_io_writer_write_byte(ptr %w, i8 %b) local_unnamed_addr #1 {
entry:
  %wnull = icmp eq ptr %w, null
  br i1 %wnull, label %err.null, label %body, !prof !0
err.null:
  ret i32 1
body:
  %len.p = getelementptr inbounds nuw i8, ptr %w, i64 8
  %cap.p = getelementptr inbounds nuw i8, ptr %w, i64 16
  %len = load i64, ptr %len.p, align 8
  %cap = load i64, ptr %cap.p, align 8
  %full = icmp uge i64 %len, %cap
  br i1 %full, label %flush, label %store
flush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  %fbad = icmp ne i32 %fs, 0
  br i1 %fbad, label %err.io, label %store0
err.io:
  ret i32 15
store0:
  br label %store
store:
  %len2 = phi i64 [ %len, %body ], [ 0, %store0 ]
  %buf = getelementptr inbounds nuw i8, ptr %w, i64 64
  %dst = getelementptr inbounds nuw i8, ptr %buf, i64 %len2
  store i8 %b, ptr %dst, align 1
  %len.n = add i64 %len2, 1
  store i64 %len.n, ptr %len.p, align 8
  ret i32 0
}

; universe_io_writer_write_all(w, src, n) -> i32 (0 OK, 1 null, 15 IO)
define i32 @universe_io_writer_write_all(ptr %w, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %got = call i64 @universe_io_writer_write(ptr %w, ptr %src, i64 %n)
  %isnull = icmp eq i64 %got, -1
  br i1 %isnull, label %ret.null, label %chk
ret.null:
  ret i32 1
chk:
  %ok = icmp eq i64 %got, %n
  br i1 %ok, label %good, label %bad
good:
  ret i32 0
bad:
  ret i32 15
}

; universe_io_writer_flush_vectored(w, iov, count) -> i32
;   (0 OK, 1 null, 8 INVALID_ARG (count>1024), 15 IO)
;   Flushes the internal buffer, then issues writev(2) over a private copy of
;   the iovec array (base@0, len@8; 16 bytes/entry) with a partial-write retry.
;   count is capped at 1024 (IOV_MAX-class) so the work copy is a bounded
;   fixed stack buffer (never a loop/dynamic alloca).
define i32 @universe_io_writer_flush_vectored(ptr %w, ptr %iov, i32 %count) local_unnamed_addr #1 {
entry:
  %work = alloca [16384 x i8], align 16
  %wnull = icmp eq ptr %w, null
  br i1 %wnull, label %err.null, label %chk.iov, !prof !0
err.null:
  ret i32 1
chk.iov:
  %cpos = icmp sgt i32 %count, 0
  %iovnull = icmp eq ptr %iov, null
  %bad.iov = and i1 %cpos, %iovnull
  br i1 %bad.iov, label %err.null, label %chk.max
chk.max:
  %toomany = icmp sgt i32 %count, 1024
  br i1 %toomany, label %err.arg, label %do.flush, !prof !0
err.arg:
  ret i32 8
do.flush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  %fbad = icmp ne i32 %fs, 0
  br i1 %fbad, label %ret.flush, label %chk.count
ret.flush:
  ret i32 %fs
chk.count:
  br i1 %cpos, label %prep, label %ret.ok
ret.ok:
  ret i32 0
prep:
  %cnt64 = sext i32 %count to i64
  %bytes = shl i64 %cnt64, 4
  call void @llvm.memcpy.p0.p0.i64(ptr %work, ptr %iov, i64 %bytes, i1 false)
  %fd64 = load i64, ptr %w, align 8
  %fd = trunc i64 %fd64 to i32
  br label %wv.loop
wv.loop:
  %vi = phi i64 [ 0, %prep ], [ %cvi, %cdone ], [ %cvi, %partial ]
  %remain = sub i64 %cnt64, %vi
  %rdone = icmp eq i64 %remain, 0
  br i1 %rdone, label %ret.ok2, label %issue
ret.ok2:
  ret i32 0
issue:
  %eoff = shl i64 %vi, 4
  %ep = getelementptr inbounds nuw i8, ptr %work, i64 %eoff
  %rem32 = trunc i64 %remain to i32
  %k = call i64 @writev(i32 %fd, ptr %ep, i32 %rem32)
  %kneg = icmp slt i64 %k, 0
  br i1 %kneg, label %err.io, label %chk.zero
err.io:
  ret i32 15
chk.zero:
  %kzero = icmp eq i64 %k, 0
  br i1 %kzero, label %ret.ok2, label %consume
consume:
  %cvi = phi i64 [ %vi, %chk.zero ], [ %cvi.n, %full ]
  %crem = phi i64 [ %k, %chk.zero ], [ %crem.n, %full ]
  %cdrained = icmp eq i64 %crem, 0
  br i1 %cdrained, label %cdone, label %consume.body
consume.body:
  %coff = shl i64 %cvi, 4
  %entryp = getelementptr inbounds nuw i8, ptr %work, i64 %coff
  %lenp = getelementptr inbounds nuw i8, ptr %entryp, i64 8
  %l = load i64, ptr %lenp, align 8
  %ge = icmp uge i64 %crem, %l
  br i1 %ge, label %full, label %partial
full:
  %crem.n = sub i64 %crem, %l
  %cvi.n = add i64 %cvi, 1
  br label %consume
partial:
  %base = load ptr, ptr %entryp, align 8
  %newbase = getelementptr inbounds nuw i8, ptr %base, i64 %crem
  store ptr %newbase, ptr %entryp, align 8
  %newlen = sub i64 %l, %crem
  store i64 %newlen, ptr %lenp, align 8
  br label %wv.loop
cdone:
  br label %wv.loop
}

define void @universe_io_writer_destroy(ptr %w) local_unnamed_addr #2 {
entry:
  %is.null = icmp eq ptr %w, null
  br i1 %is.null, label %done, label %do.flush, !prof !0
do.flush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  call void @free(ptr nonnull %w)
  br label %done
done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
