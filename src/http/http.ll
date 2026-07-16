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

; HTTP/1.1 client + server (plaintext, single-threaded). Zero-copy head parse
; over a contiguous accumulation buffer; SIMD line/token scans via src/simd;
; buffered output via src/io writer; sockets via src/net; keep-alive via
; src/net/pool. NO TLS (deferred to hardening). Concurrency DEFERRED: the
; server is a blocking accept-then-serve loop with no threads/locks.
;
; DESIGN:
;   * COMPUTE/MEMORY/IO SEPARATION. Parsing is a pure function over a caller
;     buffer: `universe_http_parse_request/response` take (buf,len) and return
;     zero-copy slices (ptr+len into buf) for method/target/version/reason and
;     every header. The IO layer (conn_read) only fills the buffer with
;     tcp_recv and re-invokes the pure parser until it stops returning
;     INCOMPLETE(11). No syscalls sit inside the parse compute.
;   * WHY A DEDICATED CONTIGUOUS BUFFER (not bufio reader for input). A parsed
;     request exposes MANY header slices that must stay simultaneously valid
;     while the handler runs AND while the body (already partly read) is
;     consumed. bufio's reader compacts/refills on the next call, invalidating
;     live slices, and offers no cheap "advance N" for keep-alive pipelining.
;     A single owned buffer keeps head+body+next-pipelined-bytes contiguous and
;     stable; leftover bytes are memmove-compacted to the front between
;     messages. This IS the canonical "fill buffer (IO) -> parse over buffer
;     (compute) -> emit (IO)" structure. Output DOES use the bufio writer
;     (accumulate + one flush per response) — never a syscall per field.
;   * SIMD-FIRST SCANS. The request/status line is split with
;     `universe_simd_find_byte` (SP/':' locate) and every header line boundary
;     with `universe_simd_find_crlf` (the vectorized CRLF kernel). Version
;     prefix check is `universe_simd_equal`. The scalar work left in this
;     module (OWS trim, decimal/hex conversion, case-fold compare) is over
;     already-located short tokens, not the byte stream.
;   * ERROR CODES (i32, per conventions.md): 0 OK, 1 NULL, 2 OOM, 4 EOF/EMPTY,
;     5 NOT_FOUND, 6 FULL (head/body exceeds buffer), 8 INVALID_ARG,
;     11 INVALID_STATE (== INCOMPLETE: need more bytes), 13 PARSE, 15 IO.
;   * message struct (msg, 96 B, caller-allocated):
;       +0  f0_ptr   (request: method  / response: reason)
;       +8  f0_len
;       +16 f1_ptr   (request: target  / response: unused)
;       +24 f1_len
;       +32 minor    (HTTP/1.<minor>)
;       +40 code     (response status; request: 0)
;       +48 header_count
;       +56 body_ptr (into buf, at head end)
;       +64 body_len (Content-Length, or raw chunked span, or EOF-delimited len)
;       +72 head_len (bytes of head incl. terminating CRLFCRLF)
;       +80 content_length (-1 if absent)
;       +88 flags    bit0 chunked, bit1 keep-alive, bit2 content-length-present
;   * header entry (32 B): name_ptr@0 name_len@8 val_ptr@16 val_len@24.
;   * response-spec (handler output, 56 B):
;       +0 status +8 reason_ptr +16 reason_len +24 hdrs_ptr +32 hdrs_count
;       +40 body_ptr +48 body_len
;   * conn (owned, header 64B + buffer@64):
;       +0 fd(i64) +8 cap(i64) +16 len(i64) +24 pos(i64) +32 writer(ptr)
;
; HARDENING-TODO: no request smuggling defenses (duplicate CL, CL+TE conflict),
; no header/URI size limits beyond the buffer cap, no timeouts. Deferred.

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

; src/simd
declare i64 @universe_simd_find_byte(ptr readonly, i64, i8)
declare i64 @universe_simd_find_crlf(ptr readonly, i64)
declare i1 @universe_simd_equal(ptr readonly, ptr readonly, i64)

; src/io writer
declare ptr @universe_io_writer_create(i32, i64)
declare void @universe_io_writer_destroy(ptr)
declare i32 @universe_io_writer_write_all(ptr, ptr, i64)
declare i32 @universe_io_writer_write_byte(ptr, i8)
declare i32 @universe_io_writer_flush(ptr)

; src/net
declare i32 @universe_net_tcp_connect(i32, i32)
declare i32 @universe_net_tcp_accept(i32, ptr)
declare i32 @universe_net_tcp_close(i32)
declare i64 @universe_net_tcp_recv(i32, ptr, i64)

; src/net/pool
declare i32 @universe_net_pool_get(ptr, i32, i32)
declare i32 @universe_net_pool_put(ptr, i32, i32, i32)

; ------------------------------------------------------------ string constants
@http.verpfx    = private unnamed_addr constant [7 x i8]  c"HTTP/1."
@http.crlf      = private unnamed_addr constant [2 x i8]  c"\0D\0A"
@http.colonsp   = private unnamed_addr constant [2 x i8]  c": "
@http.cl        = private unnamed_addr constant [16 x i8] c"Content-Length: "
@http.te        = private unnamed_addr constant [28 x i8] c"Transfer-Encoding: chunked\0D\0A"
@http.conn_ka   = private unnamed_addr constant [24 x i8] c"Connection: keep-alive\0D\0A"
@http.conn_cl   = private unnamed_addr constant [19 x i8] c"Connection: close\0D\0A"
@http.lastchunk = private unnamed_addr constant [5 x i8]  c"0\0D\0A\0D\0A"
@http.lit_cl    = private unnamed_addr constant [14 x i8] c"content-length"
@http.lit_te    = private unnamed_addr constant [17 x i8] c"transfer-encoding"
@http.lit_conn  = private unnamed_addr constant [10 x i8] c"connection"
@http.lit_chunk = private unnamed_addr constant [7 x i8]  c"chunked"
@http.lit_close = private unnamed_addr constant [5 x i8]  c"close"
@http.lit_ka    = private unnamed_addr constant [10 x i8] c"keep-alive"

; ============================================================ numeric helpers

; fmt_dec(val, dst) -> len : write decimal digits of val into dst (<=20), len.
define internal i64 @fmt_dec(i64 %val, ptr %dst) #0 {
entry:
  %tmp = alloca [20 x i8], align 1
  %z = icmp eq i64 %val, 0
  br i1 %z, label %zero, label %conv

zero:
  store i8 48, ptr %dst, align 1
  ret i64 1

conv:
  br label %loop

loop:
  %v = phi i64 [ %val, %conv ], [ %v.n, %loop ]
  %i = phi i64 [ 0, %conv ], [ %i.n, %loop ]
  %d = urem i64 %v, 10
  %v.n = udiv i64 %v, 10
  %dc = trunc i64 %d to i8
  %ch = add i8 %dc, 48
  %pos = sub i64 19, %i
  %tp = getelementptr inbounds [20 x i8], ptr %tmp, i64 0, i64 %pos
  store i8 %ch, ptr %tp, align 1
  %i.n = add i64 %i, 1
  %more = icmp ne i64 %v.n, 0
  br i1 %more, label %loop, label %done

done:
  %start = sub i64 20, %i.n
  %sp = getelementptr inbounds [20 x i8], ptr %tmp, i64 0, i64 %start
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %sp, i64 %i.n, i1 false)
  ret i64 %i.n
}

; fmt_hex(val, dst) -> len : lowercase hex digits into dst (<=16).
define internal i64 @fmt_hex(i64 %val, ptr %dst) #0 {
entry:
  %tmp = alloca [16 x i8], align 1
  %z = icmp eq i64 %val, 0
  br i1 %z, label %zero, label %conv

zero:
  store i8 48, ptr %dst, align 1
  ret i64 1

conv:
  br label %loop

loop:
  %v = phi i64 [ %val, %conv ], [ %v.n, %loop ]
  %i = phi i64 [ 0, %conv ], [ %i.n, %loop ]
  %d = and i64 %v, 15
  %v.n = lshr i64 %v, 4
  %is9 = icmp ult i64 %d, 10
  %dd = trunc i64 %d to i8
  %digit = add i8 %dd, 48
  %alpha = add i8 %dd, 87
  %ch = select i1 %is9, i8 %digit, i8 %alpha
  %pos = sub i64 15, %i
  %tp = getelementptr inbounds [16 x i8], ptr %tmp, i64 0, i64 %pos
  store i8 %ch, ptr %tp, align 1
  %i.n = add i64 %i, 1
  %more = icmp ne i64 %v.n, 0
  br i1 %more, label %loop, label %done

done:
  %start = sub i64 16, %i.n
  %sp = getelementptr inbounds [16 x i8], ptr %tmp, i64 0, i64 %start
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %sp, i64 %i.n, i1 false)
  ret i64 %i.n
}

; parse_dec(s, n) -> i64 value, or -1 on empty/non-digit.
define internal i64 @parse_dec(ptr readonly %s, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %bad, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %cont ]
  %p = getelementptr inbounds nuw i8, ptr %s, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i64
  %d = sub i64 %cz, 48
  %bad9 = icmp ugt i64 %d, 9
  br i1 %bad9, label %bad, label %cont

cont:
  %acc.m = mul i64 %acc, 10
  %acc.n = add i64 %acc.m, %d
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n

bad:
  ret i64 -1
}

; parse_hex(s, n) -> i64 value, or -1 if no leading hex digit. Stops at first
; non-hex byte (chunk extensions after ';' or spaces are ignored).
define internal i64 @parse_hex(ptr readonly %s, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %bad, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %cont ]
  %p = getelementptr inbounds nuw i8, ptr %s, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i64
  %d0 = sub i64 %cz, 48
  %is09 = icmp ult i64 %d0, 10
  %lc = or i64 %cz, 32
  %d1 = sub i64 %lc, 97
  %isaf = icmp ult i64 %d1, 6
  %ishex = or i1 %is09, %isaf
  br i1 %ishex, label %cont, label %stop

cont:
  %af = add i64 %d1, 10
  %hv = select i1 %is09, i64 %d0, i64 %af
  %acc.m = shl i64 %acc, 4
  %acc.n = add i64 %acc.m, %hv
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n

stop:
  %none = icmp eq i64 %i, 0
  br i1 %none, label %bad, label %retacc

retacc:
  ret i64 %acc

bad:
  ret i64 -1
}

; ci_eq(s, slen, lit, litlen) -> i1 : ASCII case-insensitive equality; `lit`
; must already be lowercase.
define internal i1 @ci_eq(ptr readonly %s, i64 %slen, ptr readonly %lit, i64 %litlen) #1 {
entry:
  %leneq = icmp eq i64 %slen, %litlen
  br i1 %leneq, label %chklen, label %ne

chklen:
  %z = icmp eq i64 %slen, 0
  br i1 %z, label %eq, label %loop

loop:
  %i = phi i64 [ 0, %chklen ], [ %i.n, %cont ]
  %sp = getelementptr inbounds nuw i8, ptr %s, i64 %i
  %sc = load i8, ptr %sp, align 1
  %scz = zext i8 %sc to i64
  %sub = sub i64 %scz, 65
  %isup = icmp ult i64 %sub, 26
  %scl = or i64 %scz, 32
  %folded = select i1 %isup, i64 %scl, i64 %scz
  %lp = getelementptr inbounds nuw i8, ptr %lit, i64 %i
  %lb = load i8, ptr %lp, align 1
  %lbz = zext i8 %lb to i64
  %lsub = sub i64 %lbz, 65
  %lisup = icmp ult i64 %lsub, 26
  %lcl = or i64 %lbz, 32
  %lfolded = select i1 %lisup, i64 %lcl, i64 %lbz
  %cheq = icmp eq i64 %folded, %lfolded
  br i1 %cheq, label %cont, label %ne

cont:
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %slen
  br i1 %more, label %loop, label %eq

eq:
  ret i1 true

ne:
  ret i1 false
}

; skip_ows_front(p, start, end) -> i64 first index >= start that is not SP/HT.
define internal i64 @skip_ows_front(ptr readonly %p, i64 %start, i64 %end) #1 {
entry:
  br label %head

head:
  %i = phi i64 [ %start, %entry ], [ %i.n, %cont ]
  %atend = icmp uge i64 %i, %end
  br i1 %atend, label %done, label %body

body:
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %sp = icmp eq i8 %b, 32
  %ht = icmp eq i8 %b, 9
  %ws = or i1 %sp, %ht
  br i1 %ws, label %cont, label %done

cont:
  %i.n = add i64 %i, 1
  br label %head

done:
  ret i64 %i
}

; trim_ows_back(p, start, end) -> i64 index one past the last non-OWS byte.
define internal i64 @trim_ows_back(ptr readonly %p, i64 %start, i64 %end) #1 {
entry:
  br label %head

head:
  %e = phi i64 [ %end, %entry ], [ %e.n, %cont ]
  %empty = icmp ule i64 %e, %start
  br i1 %empty, label %done, label %body

body:
  %em1 = sub i64 %e, 1
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %em1
  %b = load i8, ptr %pp, align 1
  %sp = icmp eq i8 %b, 32
  %ht = icmp eq i8 %b, 9
  %ws = or i1 %sp, %ht
  br i1 %ws, label %cont, label %done

cont:
  %e.n = sub i64 %e, 1
  br label %head

done:
  ret i64 %e
}

; ======================================================== head-end + headers

; universe_http_find_head_end(buf, len) -> i64 : index just past the CRLFCRLF
; terminating the message head, or -1 if not yet present.
define i64 @universe_http_find_head_end(ptr readonly %buf, i64 %len) local_unnamed_addr #2 {
entry:
  %small = icmp ult i64 %len, 4
  br i1 %small, label %none, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %k1, %adv ]
  %rem = sub i64 %len, %i
  %base = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %rel = call i64 @universe_simd_find_byte(ptr %base, i64 %rem, i8 10)
  %miss = icmp slt i64 %rel, 0
  br i1 %miss, label %none, label %chk

chk:
  %k = add i64 %i, %rel
  %k3 = icmp uge i64 %k, 3
  br i1 %k3, label %pat, label %adv

pat:
  %km3 = sub i64 %k, 3
  %pa = getelementptr inbounds nuw i8, ptr %buf, i64 %km3
  %a = load i8, ptr %pa, align 1
  %km2 = sub i64 %k, 2
  %pb = getelementptr inbounds nuw i8, ptr %buf, i64 %km2
  %bb = load i8, ptr %pb, align 1
  %km1 = sub i64 %k, 1
  %pc = getelementptr inbounds nuw i8, ptr %buf, i64 %km1
  %cc = load i8, ptr %pc, align 1
  %aok = icmp eq i8 %a, 13
  %bok = icmp eq i8 %bb, 10
  %cok = icmp eq i8 %cc, 13
  %ab = and i1 %aok, %bok
  %abc = and i1 %ab, %cok
  br i1 %abc, label %found, label %adv

found:
  %he = add i64 %k, 1
  ret i64 %he

adv:
  %k1 = add i64 %k, 1
  %past = icmp uge i64 %k1, %len
  br i1 %past, label %none, label %loop

none:
  ret i64 -1
}

; parse_headers_block(buf, headlen, pos0, hdrs, cap) -> i64 stored-count, -1 bad.
; Walks header lines with the SIMD CRLF scanner until the empty line.
define internal i64 @parse_headers_block(ptr %buf, i64 %headlen, i64 %pos0, ptr %hdrs, i64 %cap) #1 {
entry:
  br label %loop

loop:
  %pos = phi i64 [ %pos0, %entry ], [ %pos.n, %adv ]
  %count = phi i64 [ 0, %entry ], [ %count.n, %adv ]
  %avail = sub i64 %headlen, %pos
  %linebase = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  %r = call i64 @universe_simd_find_crlf(ptr %linebase, i64 %avail)
  %rbad = icmp slt i64 %r, 0
  br i1 %rbad, label %bad, label %chkempty

chkempty:
  %empty = icmp eq i64 %r, 0
  br i1 %empty, label %done, label %parse

parse:
  %colon = call i64 @universe_simd_find_byte(ptr %linebase, i64 %r, i8 58)
  %cbad = icmp slt i64 %colon, 0
  br i1 %cbad, label %bad, label %haveval

haveval:
  ; name = [pos, pos+colon); value = OWS-trimmed remainder up to CRLF
  %vstart0 = add i64 %pos, %colon
  %vstart1 = add i64 %vstart0, 1
  %lineend = add i64 %pos, %r
  %vs = call i64 @skip_ows_front(ptr %buf, i64 %vstart1, i64 %lineend)
  %ve = call i64 @trim_ows_back(ptr %buf, i64 %vs, i64 %lineend)
  %vallen = sub i64 %ve, %vs
  %cansave = icmp ult i64 %count, %cap
  br i1 %cansave, label %save, label %adv

save:
  %eoff = shl i64 %count, 5
  %ep = getelementptr inbounds nuw i8, ptr %hdrs, i64 %eoff
  store ptr %linebase, ptr %ep, align 8
  %nlp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  store i64 %colon, ptr %nlp, align 8
  %vpp = getelementptr inbounds nuw i8, ptr %ep, i64 16
  %valptr = getelementptr inbounds nuw i8, ptr %buf, i64 %vs
  store ptr %valptr, ptr %vpp, align 8
  %vlp = getelementptr inbounds nuw i8, ptr %ep, i64 24
  store i64 %vallen, ptr %vlp, align 8
  %count.s = add i64 %count, 1
  br label %adv

adv:
  %count.n = phi i64 [ %count, %haveval ], [ %count.s, %save ]
  %pos.step = add i64 %r, 2
  %pos.n = add i64 %pos, %pos.step
  br label %loop

done:
  ret i64 %count

bad:
  ret i64 -1
}

; validate_version(p, vlen) -> i64 minor (0 or 1), or -1 if not HTTP/1.0|1.1.
define internal i64 @validate_version(ptr readonly %p, i64 %vlen) #1 {
entry:
  %len8 = icmp eq i64 %vlen, 8
  br i1 %len8, label %pfx, label %bad

pfx:
  %ok = call i1 @universe_simd_equal(ptr %p, ptr @http.verpfx, i64 7)
  br i1 %ok, label %last, label %bad

last:
  %lp = getelementptr inbounds nuw i8, ptr %p, i64 7
  %lb = load i8, ptr %lp, align 1
  %is0 = icmp eq i8 %lb, 48
  %is1 = icmp eq i8 %lb, 49
  br i1 %is0, label %ret0, label %chk1

chk1:
  br i1 %is1, label %ret1, label %bad

ret0:
  ret i64 0

ret1:
  ret i64 1

bad:
  ret i64 -1
}

; scan_special(hdrs, count, minor, msg): sets msg.content_length(+80) and
; msg.flags(+88) from Content-Length / Transfer-Encoding / Connection.
define internal void @scan_special(ptr %hdrs, i64 %count, i64 %minor, ptr %msg) #1 {
entry:
  %kadef = icmp uge i64 %minor, 1
  %z = icmp eq i64 %count, 0
  br i1 %z, label %finish, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %cl = phi i64 [ -1, %entry ], [ %cl.n, %cont ]
  %chunk = phi i1 [ false, %entry ], [ %chunk.n, %cont ]
  %cclose = phi i1 [ false, %entry ], [ %cclose.n, %cont ]
  %cka = phi i1 [ false, %entry ], [ %cka.n, %cont ]
  %eoff = shl i64 %i, 5
  %ep = getelementptr inbounds nuw i8, ptr %hdrs, i64 %eoff
  %np = load ptr, ptr %ep, align 8
  %nlp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  %nl = load i64, ptr %nlp, align 8
  %vpp = getelementptr inbounds nuw i8, ptr %ep, i64 16
  %vp = load ptr, ptr %vpp, align 8
  %vlp = getelementptr inbounds nuw i8, ptr %ep, i64 24
  %vl = load i64, ptr %vlp, align 8
  %is.cl = call i1 @ci_eq(ptr %np, i64 %nl, ptr @http.lit_cl, i64 14)
  br i1 %is.cl, label %do.cl, label %chk.te

do.cl:
  %clv = call i64 @parse_dec(ptr %vp, i64 %vl)
  %clv.ok = icmp sge i64 %clv, 0
  %cl.set = select i1 %clv.ok, i64 %clv, i64 %cl
  br label %cont

chk.te:
  %is.te = call i1 @ci_eq(ptr %np, i64 %nl, ptr @http.lit_te, i64 17)
  br i1 %is.te, label %do.te, label %chk.conn

do.te:
  %is.chunk = call i1 @ci_eq(ptr %vp, i64 %vl, ptr @http.lit_chunk, i64 7)
  %chunk.set = or i1 %chunk, %is.chunk
  br label %cont

chk.conn:
  %is.conn = call i1 @ci_eq(ptr %np, i64 %nl, ptr @http.lit_conn, i64 10)
  br i1 %is.conn, label %do.conn, label %cont

do.conn:
  %is.close = call i1 @ci_eq(ptr %vp, i64 %vl, ptr @http.lit_close, i64 5)
  %is.ka = call i1 @ci_eq(ptr %vp, i64 %vl, ptr @http.lit_ka, i64 10)
  %cclose.set = or i1 %cclose, %is.close
  %cka.set = or i1 %cka, %is.ka
  br label %cont

cont:
  %cl.n = phi i64 [ %cl.set, %do.cl ], [ %cl, %do.te ], [ %cl, %do.conn ], [ %cl, %chk.conn ]
  %chunk.n = phi i1 [ %chunk, %do.cl ], [ %chunk.set, %do.te ], [ %chunk, %do.conn ], [ %chunk, %chk.conn ]
  %cclose.n = phi i1 [ %cclose, %do.cl ], [ %cclose, %do.te ], [ %cclose.set, %do.conn ], [ %cclose, %chk.conn ]
  %cka.n = phi i1 [ %cka, %do.cl ], [ %cka, %do.te ], [ %cka.set, %do.conn ], [ %cka, %chk.conn ]
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %finish

finish:
  %cl.f = phi i64 [ -1, %entry ], [ %cl.n, %cont ]
  %chunk.f = phi i1 [ false, %entry ], [ %chunk.n, %cont ]
  %cclose.f = phi i1 [ false, %entry ], [ %cclose.n, %cont ]
  %cka.f = phi i1 [ false, %entry ], [ %cka.n, %cont ]
  ; keep-alive: default from version, Connection: close forces off, keep-alive on
  %ka0 = and i1 %kadef, true
  %ka1 = select i1 %cclose.f, i1 false, i1 %ka0
  %ka2 = select i1 %cka.f, i1 true, i1 %ka1
  %f.chunk = select i1 %chunk.f, i64 1, i64 0
  %f.ka = select i1 %ka2, i64 2, i64 0
  %clpresent = icmp sge i64 %cl.f, 0
  %f.cl = select i1 %clpresent, i64 4, i64 0
  %f01 = or i64 %f.chunk, %f.ka
  %flags = or i64 %f01, %f.cl
  %clp = getelementptr inbounds nuw i8, ptr %msg, i64 80
  store i64 %cl.f, ptr %clp, align 8
  %flp = getelementptr inbounds nuw i8, ptr %msg, i64 88
  store i64 %flags, ptr %flp, align 8
  ret void
}

; universe_http_parse_request(buf, len, msg, hdrs, cap) -> i32
;   0 OK, 1 NULL, 11 INCOMPLETE (need more bytes), 13 PARSE.
define i32 @universe_http_parse_request(ptr %buf, i64 %len, ptr %msg, ptr %hdrs, i64 %cap) local_unnamed_addr #2 {
entry:
  %bn = icmp eq ptr %buf, null
  %mn = icmp eq ptr %msg, null
  %hn = icmp eq ptr %hdrs, null
  %n0 = or i1 %bn, %mn
  %anynull = or i1 %n0, %hn
  br i1 %anynull, label %err.null, label %head

err.null:
  ret i32 1

head:
  %he = call i64 @universe_http_find_head_end(ptr %buf, i64 %len)
  %incomplete = icmp slt i64 %he, 0
  br i1 %incomplete, label %err.inc, label %reqline

err.inc:
  ret i32 11

reqline:
  %r = call i64 @universe_simd_find_crlf(ptr %buf, i64 %he)
  %rbad = icmp slt i64 %r, 0
  br i1 %rbad, label %err.parse, label %method

method:
  %s1 = call i64 @universe_simd_find_byte(ptr %buf, i64 %r, i8 32)
  %s1bad = icmp slt i64 %s1, 0
  br i1 %s1bad, label %err.parse, label %target

target:
  %rest = add i64 %s1, 1
  %base2 = getelementptr inbounds nuw i8, ptr %buf, i64 %rest
  %len2 = sub i64 %r, %rest
  %s2 = call i64 @universe_simd_find_byte(ptr %base2, i64 %len2, i8 32)
  %s2bad = icmp slt i64 %s2, 0
  br i1 %s2bad, label %err.parse, label %version

version:
  %vstart = add i64 %rest, %s2
  %vstart1 = add i64 %vstart, 1
  %vp = getelementptr inbounds nuw i8, ptr %buf, i64 %vstart1
  %vlen = sub i64 %r, %vstart1
  %minor = call i64 @validate_version(ptr %vp, i64 %vlen)
  %mbad = icmp slt i64 %minor, 0
  br i1 %mbad, label %err.parse, label %store

store:
  ; f0 = method [buf, s1]
  store ptr %buf, ptr %msg, align 8
  %f0l = getelementptr inbounds nuw i8, ptr %msg, i64 8
  store i64 %s1, ptr %f0l, align 8
  ; f1 = target [buf+rest, s2]
  %f1p = getelementptr inbounds nuw i8, ptr %msg, i64 16
  store ptr %base2, ptr %f1p, align 8
  %f1l = getelementptr inbounds nuw i8, ptr %msg, i64 24
  store i64 %s2, ptr %f1l, align 8
  %mp = getelementptr inbounds nuw i8, ptr %msg, i64 32
  store i64 %minor, ptr %mp, align 8
  %cp = getelementptr inbounds nuw i8, ptr %msg, i64 40
  store i64 0, ptr %cp, align 8
  ; headers
  %pos0 = add i64 %r, 2
  %count = call i64 @parse_headers_block(ptr %buf, i64 %he, i64 %pos0, ptr %hdrs, i64 %cap)
  %cbad = icmp slt i64 %count, 0
  br i1 %cbad, label %err.parse, label %hdrs.ok

hdrs.ok:
  %hcp = getelementptr inbounds nuw i8, ptr %msg, i64 48
  store i64 %count, ptr %hcp, align 8
  call void @scan_special(ptr %hdrs, i64 %count, i64 %minor, ptr %msg)
  ; head_len
  %hlp = getelementptr inbounds nuw i8, ptr %msg, i64 72
  store i64 %he, ptr %hlp, align 8
  ; body_ptr = buf+he
  %bodyp = getelementptr inbounds nuw i8, ptr %buf, i64 %he
  %bpp = getelementptr inbounds nuw i8, ptr %msg, i64 56
  store ptr %bodyp, ptr %bpp, align 8
  ; body_len = content_length if present else 0
  %flp = getelementptr inbounds nuw i8, ptr %msg, i64 88
  %flags = load i64, ptr %flp, align 8
  %hascl = and i64 %flags, 4
  %clpresent = icmp ne i64 %hascl, 0
  %clp = getelementptr inbounds nuw i8, ptr %msg, i64 80
  %clv = load i64, ptr %clp, align 8
  %blen = select i1 %clpresent, i64 %clv, i64 0
  %blp = getelementptr inbounds nuw i8, ptr %msg, i64 64
  store i64 %blen, ptr %blp, align 8
  ret i32 0

err.parse:
  ret i32 13
}

; universe_http_parse_response(buf, len, msg, hdrs, cap) -> i32 (same codes).
define i32 @universe_http_parse_response(ptr %buf, i64 %len, ptr %msg, ptr %hdrs, i64 %cap) local_unnamed_addr #2 {
entry:
  %bn = icmp eq ptr %buf, null
  %mn = icmp eq ptr %msg, null
  %hn = icmp eq ptr %hdrs, null
  %n0 = or i1 %bn, %mn
  %anynull = or i1 %n0, %hn
  br i1 %anynull, label %err.null, label %head

err.null:
  ret i32 1

head:
  %he = call i64 @universe_http_find_head_end(ptr %buf, i64 %len)
  %incomplete = icmp slt i64 %he, 0
  br i1 %incomplete, label %err.inc, label %statusline

err.inc:
  ret i32 11

statusline:
  %r = call i64 @universe_simd_find_crlf(ptr %buf, i64 %he)
  %rbad = icmp slt i64 %r, 0
  br i1 %rbad, label %err.parse, label %ver

ver:
  %s1 = call i64 @universe_simd_find_byte(ptr %buf, i64 %r, i8 32)
  %s1bad = icmp slt i64 %s1, 0
  br i1 %s1bad, label %err.parse, label %vok

vok:
  %minor = call i64 @validate_version(ptr %buf, i64 %s1)
  %mbad = icmp slt i64 %minor, 0
  br i1 %mbad, label %err.parse, label %code

code:
  %codestart = add i64 %s1, 1
  %cbase = getelementptr inbounds nuw i8, ptr %buf, i64 %codestart
  %clen2 = sub i64 %r, %codestart
  %s2 = call i64 @universe_simd_find_byte(ptr %cbase, i64 %clen2, i8 32)
  %s2bad = icmp slt i64 %s2, 0
  br i1 %s2bad, label %noreason, label %withreason

noreason:
  ; whole remainder is the status code, empty reason
  %codev.nr = call i64 @parse_dec(ptr %cbase, i64 %clen2)
  %rend.nr = getelementptr inbounds nuw i8, ptr %buf, i64 %r
  br label %codechk

withreason:
  %codev.wr = call i64 @parse_dec(ptr %cbase, i64 %s2)
  %rstart = add i64 %codestart, %s2
  %rstart1 = add i64 %rstart, 1
  %reasonp.wr = getelementptr inbounds nuw i8, ptr %buf, i64 %rstart1
  %reasonlen.wr = sub i64 %r, %rstart1
  br label %codechk

codechk:
  %codev = phi i64 [ %codev.nr, %noreason ], [ %codev.wr, %withreason ]
  %reasonp = phi ptr [ %rend.nr, %noreason ], [ %reasonp.wr, %withreason ]
  %reasonlen = phi i64 [ 0, %noreason ], [ %reasonlen.wr, %withreason ]
  %codebad = icmp slt i64 %codev, 0
  br i1 %codebad, label %err.parse, label %store

store:
  ; f0 = reason
  store ptr %reasonp, ptr %msg, align 8
  %f0l = getelementptr inbounds nuw i8, ptr %msg, i64 8
  store i64 %reasonlen, ptr %f0l, align 8
  %f1p = getelementptr inbounds nuw i8, ptr %msg, i64 16
  store ptr null, ptr %f1p, align 8
  %f1l = getelementptr inbounds nuw i8, ptr %msg, i64 24
  store i64 0, ptr %f1l, align 8
  %mp = getelementptr inbounds nuw i8, ptr %msg, i64 32
  store i64 %minor, ptr %mp, align 8
  %cp = getelementptr inbounds nuw i8, ptr %msg, i64 40
  store i64 %codev, ptr %cp, align 8
  %pos0 = add i64 %r, 2
  %count = call i64 @parse_headers_block(ptr %buf, i64 %he, i64 %pos0, ptr %hdrs, i64 %cap)
  %cbad = icmp slt i64 %count, 0
  br i1 %cbad, label %err.parse, label %hdrs.ok

hdrs.ok:
  %hcp = getelementptr inbounds nuw i8, ptr %msg, i64 48
  store i64 %count, ptr %hcp, align 8
  call void @scan_special(ptr %hdrs, i64 %count, i64 %minor, ptr %msg)
  %hlp = getelementptr inbounds nuw i8, ptr %msg, i64 72
  store i64 %he, ptr %hlp, align 8
  %bodyp = getelementptr inbounds nuw i8, ptr %buf, i64 %he
  %bpp = getelementptr inbounds nuw i8, ptr %msg, i64 56
  store ptr %bodyp, ptr %bpp, align 8
  %flp = getelementptr inbounds nuw i8, ptr %msg, i64 88
  %flags = load i64, ptr %flp, align 8
  %hascl = and i64 %flags, 4
  %clpresent = icmp ne i64 %hascl, 0
  %clp = getelementptr inbounds nuw i8, ptr %msg, i64 80
  %clv = load i64, ptr %clp, align 8
  %blen = select i1 %clpresent, i64 %clv, i64 0
  %blp = getelementptr inbounds nuw i8, ptr %msg, i64 64
  store i64 %blen, ptr %blp, align 8
  ret i32 0

err.parse:
  ret i32 13
}

; universe_http_header_get(hdrs, count, name, name_len, out_val_ptr, out_val_len)
;   -> i32: 0 found, 5 NOT_FOUND, 1 NULL. Case-insensitive name match.
define i32 @universe_http_header_get(ptr %hdrs, i64 %count, ptr %name, i64 %name_len, ptr %out_val_ptr, ptr %out_val_len) local_unnamed_addr #2 {
entry:
  %hn = icmp eq ptr %hdrs, null
  %nn = icmp eq ptr %name, null
  %anynull = or i1 %hn, %nn
  br i1 %anynull, label %err.null, label %setup

err.null:
  ret i32 1

setup:
  %z = icmp eq i64 %count, 0
  br i1 %z, label %notfound, label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %cont ]
  %eoff = shl i64 %i, 5
  %ep = getelementptr inbounds nuw i8, ptr %hdrs, i64 %eoff
  %np = load ptr, ptr %ep, align 8
  %nlp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  %nl = load i64, ptr %nlp, align 8
  %match = call i1 @ci_eq(ptr %np, i64 %nl, ptr %name, i64 %name_len)
  br i1 %match, label %found, label %cont

found:
  %vpp = getelementptr inbounds nuw i8, ptr %ep, i64 16
  %vp = load ptr, ptr %vpp, align 8
  %vlp = getelementptr inbounds nuw i8, ptr %ep, i64 24
  %vl = load i64, ptr %vlp, align 8
  %ovpn = icmp eq ptr %out_val_ptr, null
  br i1 %ovpn, label %skip.p, label %write.p

write.p:
  store ptr %vp, ptr %out_val_ptr, align 8
  br label %skip.p

skip.p:
  %ovln = icmp eq ptr %out_val_len, null
  br i1 %ovln, label %ret.found, label %write.l

write.l:
  store i64 %vl, ptr %out_val_len, align 8
  br label %ret.found

ret.found:
  ret i32 0

cont:
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %notfound

notfound:
  ret i32 5
}

; ================================================================ serializers

; emit_headers(w, hdrs, count) -> i32 : write "name: value\r\n" for each entry.
define internal i32 @emit_headers(ptr %w, ptr %hdrs, i64 %count) #1 {
entry:
  %z = icmp eq i64 %count, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %e = phi i32 [ 0, %entry ], [ %e.n, %loop ]
  %eoff = shl i64 %i, 5
  %ep = getelementptr inbounds nuw i8, ptr %hdrs, i64 %eoff
  %np = load ptr, ptr %ep, align 8
  %nlp = getelementptr inbounds nuw i8, ptr %ep, i64 8
  %nl = load i64, ptr %nlp, align 8
  %vpp = getelementptr inbounds nuw i8, ptr %ep, i64 16
  %vp = load ptr, ptr %vpp, align 8
  %vlp = getelementptr inbounds nuw i8, ptr %ep, i64 24
  %vl = load i64, ptr %vlp, align 8
  %ea = call i32 @universe_io_writer_write_all(ptr %w, ptr %np, i64 %nl)
  %eb = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.colonsp, i64 2)
  %ec = call i32 @universe_io_writer_write_all(ptr %w, ptr %vp, i64 %vl)
  %ed = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %e1 = or i32 %e, %ea
  %e2 = or i32 %e1, %eb
  %e3 = or i32 %e2, %ec
  %e.n = or i32 %e3, %ed
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %loop, label %done

done:
  %eret = phi i32 [ 0, %entry ], [ %e.n, %loop ]
  ret i32 %eret
}

; emit_tail(w, blen, keepalive, body) -> i32 : Content-Length, Connection, the
; blank line ending the head, then the body.
define internal i32 @emit_tail(ptr %w, i64 %blen, i32 %keepalive, ptr %body) #1 {
entry:
  %buf = alloca [24 x i8], align 1
  %eA = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.cl, i64 16)
  %l = call i64 @fmt_dec(i64 %blen, ptr %buf)
  %eB = call i32 @universe_io_writer_write_all(ptr %w, ptr %buf, i64 %l)
  %eC = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %ka = icmp ne i32 %keepalive, 0
  br i1 %ka, label %kab, label %clb

kab:
  %eD1 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.conn_ka, i64 24)
  br label %cj

clb:
  %eD2 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.conn_cl, i64 19)
  br label %cj

cj:
  %eD = phi i32 [ %eD1, %kab ], [ %eD2, %clb ]
  %eE = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %hasbody = icmp sgt i64 %blen, 0
  br i1 %hasbody, label %wb, label %nb

wb:
  %eF = call i32 @universe_io_writer_write_all(ptr %w, ptr %body, i64 %blen)
  br label %bj

nb:
  br label %bj

bj:
  %eF2 = phi i32 [ %eF, %wb ], [ 0, %nb ]
  %t1 = or i32 %eA, %eB
  %t2 = or i32 %t1, %eC
  %t3 = or i32 %t2, %eD
  %t4 = or i32 %t3, %eE
  %tall = or i32 %t4, %eF2
  ret i32 %tall
}

; universe_http_write_request(w, method,mlen, target,tlen, minor, hdrs,count,
;   body,blen, keepalive) -> i32 : 0 OK, 1 NULL, 15 IO. Content-Length and
;   Connection are emitted automatically; do not include them in hdrs.
define i32 @universe_http_write_request(ptr %w, ptr %method, i64 %mlen, ptr %target, i64 %tlen, i64 %minor, ptr %hdrs, i64 %count, ptr %body, i64 %blen, i32 %keepalive) local_unnamed_addr #2 {
entry:
  %wn = icmp eq ptr %w, null
  br i1 %wn, label %err.null, label %line

err.null:
  ret i32 1

line:
  %e1 = call i32 @universe_io_writer_write_all(ptr %w, ptr %method, i64 %mlen)
  %e2 = call i32 @universe_io_writer_write_byte(ptr %w, i8 32)
  %e3 = call i32 @universe_io_writer_write_all(ptr %w, ptr %target, i64 %tlen)
  %e4 = call i32 @universe_io_writer_write_byte(ptr %w, i8 32)
  %e5 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.verpfx, i64 7)
  %mtr = trunc i64 %minor to i8
  %dch = add i8 %mtr, 48
  %e6 = call i32 @universe_io_writer_write_byte(ptr %w, i8 %dch)
  %e7 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %eh = call i32 @emit_headers(ptr %w, ptr %hdrs, i64 %count)
  %et = call i32 @emit_tail(ptr %w, i64 %blen, i32 %keepalive, ptr %body)
  %a1 = or i32 %e1, %e2
  %a2 = or i32 %a1, %e3
  %a3 = or i32 %a2, %e4
  %a4 = or i32 %a3, %e5
  %a5 = or i32 %a4, %e6
  %a6 = or i32 %a5, %e7
  %a7 = or i32 %a6, %eh
  %aall = or i32 %a7, %et
  %bad = icmp ne i32 %aall, 0
  %ret = select i1 %bad, i32 15, i32 0
  ret i32 %ret
}

; universe_http_write_response(w, status, reason,rlen, hdrs,count, body,blen,
;   keepalive) -> i32. Emits "HTTP/1.1 <code> <reason>" then headers, auto
;   Content-Length + Connection, blank line, body.
define i32 @universe_http_write_response(ptr %w, i64 %status, ptr %reason, i64 %rlen, ptr %hdrs, i64 %count, ptr %body, i64 %blen, i32 %keepalive) local_unnamed_addr #2 {
entry:
  %buf = alloca [24 x i8], align 1
  %wn = icmp eq ptr %w, null
  br i1 %wn, label %err.null, label %line

err.null:
  ret i32 1

line:
  %e1 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.verpfx, i64 7)
  %e2 = call i32 @universe_io_writer_write_byte(ptr %w, i8 49)
  %e3 = call i32 @universe_io_writer_write_byte(ptr %w, i8 32)
  %l = call i64 @fmt_dec(i64 %status, ptr %buf)
  %e4 = call i32 @universe_io_writer_write_all(ptr %w, ptr %buf, i64 %l)
  %e5 = call i32 @universe_io_writer_write_byte(ptr %w, i8 32)
  %hasreason = icmp sgt i64 %rlen, 0
  br i1 %hasreason, label %wr, label %skipr

wr:
  %e6a = call i32 @universe_io_writer_write_all(ptr %w, ptr %reason, i64 %rlen)
  br label %rj

skipr:
  br label %rj

rj:
  %e6 = phi i32 [ %e6a, %wr ], [ 0, %skipr ]
  %e7 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %eh = call i32 @emit_headers(ptr %w, ptr %hdrs, i64 %count)
  %et = call i32 @emit_tail(ptr %w, i64 %blen, i32 %keepalive, ptr %body)
  %a1 = or i32 %e1, %e2
  %a2 = or i32 %a1, %e3
  %a3 = or i32 %a2, %e4
  %a4 = or i32 %a3, %e5
  %a5 = or i32 %a4, %e6
  %a6 = or i32 %a5, %e7
  %a7 = or i32 %a6, %eh
  %aall = or i32 %a7, %et
  %bad = icmp ne i32 %aall, 0
  %ret = select i1 %bad, i32 15, i32 0
  ret i32 %ret
}

; ================================================================ chunked codec

; universe_http_chunked_encode(w, src, len) -> i32 : one data chunk (if len>0)
;   plus the terminating zero chunk "0\r\n\r\n".
define i32 @universe_http_chunked_encode(ptr %w, ptr %src, i64 %len) local_unnamed_addr #2 {
entry:
  %buf = alloca [16 x i8], align 1
  %wn = icmp eq ptr %w, null
  br i1 %wn, label %err.null, label %chk

err.null:
  ret i32 1

chk:
  %has = icmp sgt i64 %len, 0
  br i1 %has, label %chunk, label %last

chunk:
  %hl = call i64 @fmt_hex(i64 %len, ptr %buf)
  %e1 = call i32 @universe_io_writer_write_all(ptr %w, ptr %buf, i64 %hl)
  %e2 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %e3 = call i32 @universe_io_writer_write_all(ptr %w, ptr %src, i64 %len)
  %e4 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.crlf, i64 2)
  %c1 = or i32 %e1, %e2
  %c2 = or i32 %c1, %e3
  %c3 = or i32 %c2, %e4
  br label %last

last:
  %eacc = phi i32 [ 0, %chk ], [ %c3, %chunk ]
  %e5 = call i32 @universe_io_writer_write_all(ptr %w, ptr @http.lastchunk, i64 5)
  %all = or i32 %eacc, %e5
  %bad = icmp ne i32 %all, 0
  %ret = select i1 %bad, i32 15, i32 0
  ret i32 %ret
}

; chunked_span(src, srclen, out_raw) -> i32 : 0 complete (out_raw = raw bytes
;   consumed), 11 INCOMPLETE, 13 PARSE. No copy; used to size a chunked body.
define internal i32 @chunked_span(ptr %src, i64 %srclen, ptr %outraw) #1 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %dnext ]
  %avail = sub i64 %srclen, %i
  %base = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %r = call i64 @universe_simd_find_crlf(ptr %base, i64 %avail)
  %rbad = icmp slt i64 %r, 0
  br i1 %rbad, label %inc, label %sz

sz:
  %size = call i64 @parse_hex(ptr %base, i64 %r)
  %szbad = icmp slt i64 %size, 0
  br i1 %szbad, label %bad, label %cont

cont:
  %linelen = add i64 %r, 2
  %datastart = add i64 %i, %linelen
  %iszero = icmp eq i64 %size, 0
  br i1 %iszero, label %last, label %data

last:
  %rem = sub i64 %srclen, %datastart
  %rem2 = icmp ult i64 %rem, 2
  br i1 %rem2, label %inc, label %lastchk

lastchk:
  %la = getelementptr inbounds nuw i8, ptr %src, i64 %datastart
  %lab = load i8, ptr %la, align 1
  %ds1 = add i64 %datastart, 1
  %lb = getelementptr inbounds nuw i8, ptr %src, i64 %ds1
  %lbb = load i8, ptr %lb, align 1
  %lcr = icmp eq i8 %lab, 13
  %llf = icmp eq i8 %lbb, 10
  %lok = and i1 %lcr, %llf
  br i1 %lok, label %complete, label %bad

complete:
  %end = add i64 %datastart, 2
  store i64 %end, ptr %outraw, align 8
  ret i32 0

data:
  %dataend = add i64 %datastart, %size
  %need = add i64 %dataend, 2
  %short = icmp ult i64 %srclen, %need
  br i1 %short, label %inc, label %datachk

datachk:
  %da = getelementptr inbounds nuw i8, ptr %src, i64 %dataend
  %dab = load i8, ptr %da, align 1
  %de1 = add i64 %dataend, 1
  %db = getelementptr inbounds nuw i8, ptr %src, i64 %de1
  %dbb = load i8, ptr %db, align 1
  %dcr = icmp eq i8 %dab, 13
  %dlf = icmp eq i8 %dbb, 10
  %dok = and i1 %dcr, %dlf
  br i1 %dok, label %dnext, label %bad

dnext:
  %i.n = add i64 %dataend, 2
  br label %loop

inc:
  ret i32 11

bad:
  ret i32 13
}

; universe_http_chunked_decode(src, srclen, dst, dstcap, out_len) -> i32 :
;   0 complete (*out_len = decoded bytes), 11 INCOMPLETE, 13 PARSE, 6 FULL.
define i32 @universe_http_chunked_decode(ptr %src, i64 %srclen, ptr %dst, i64 %dstcap, ptr %out_len) local_unnamed_addr #2 {
entry:
  %sn = icmp eq ptr %src, null
  %dn = icmp eq ptr %dst, null
  %anynull = or i1 %sn, %dn
  br i1 %anynull, label %err.null, label %loop

err.null:
  ret i32 1

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %copy ]
  %o = phi i64 [ 0, %entry ], [ %o.n, %copy ]
  %avail = sub i64 %srclen, %i
  %base = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %r = call i64 @universe_simd_find_crlf(ptr %base, i64 %avail)
  %rbad = icmp slt i64 %r, 0
  br i1 %rbad, label %inc, label %sz

sz:
  %size = call i64 @parse_hex(ptr %base, i64 %r)
  %szbad = icmp slt i64 %size, 0
  br i1 %szbad, label %bad, label %cont

cont:
  %linelen = add i64 %r, 2
  %datastart = add i64 %i, %linelen
  %iszero = icmp eq i64 %size, 0
  br i1 %iszero, label %last, label %data

last:
  %rem = sub i64 %srclen, %datastart
  %rem2 = icmp ult i64 %rem, 2
  br i1 %rem2, label %inc, label %lastchk

lastchk:
  %la = getelementptr inbounds nuw i8, ptr %src, i64 %datastart
  %lab = load i8, ptr %la, align 1
  %ds1 = add i64 %datastart, 1
  %lb = getelementptr inbounds nuw i8, ptr %src, i64 %ds1
  %lbb = load i8, ptr %lb, align 1
  %lcr = icmp eq i8 %lab, 13
  %llf = icmp eq i8 %lbb, 10
  %lok = and i1 %lcr, %llf
  br i1 %lok, label %complete, label %bad

complete:
  %onn = icmp eq ptr %out_len, null
  br i1 %onn, label %ret0, label %wo

wo:
  store i64 %o, ptr %out_len, align 8
  br label %ret0

ret0:
  ret i32 0

data:
  %dataend = add i64 %datastart, %size
  %need = add i64 %dataend, 2
  %short = icmp ult i64 %srclen, %need
  br i1 %short, label %inc, label %datachk

datachk:
  %da = getelementptr inbounds nuw i8, ptr %src, i64 %dataend
  %dab = load i8, ptr %da, align 1
  %de1 = add i64 %dataend, 1
  %db = getelementptr inbounds nuw i8, ptr %src, i64 %de1
  %dbb = load i8, ptr %db, align 1
  %dcr = icmp eq i8 %dab, 13
  %dlf = icmp eq i8 %dbb, 10
  %dok = and i1 %dcr, %dlf
  br i1 %dok, label %fits, label %bad

fits:
  %o.after = add i64 %o, %size
  %over = icmp ugt i64 %o.after, %dstcap
  br i1 %over, label %full, label %copy

copy:
  %cdst = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  %csrc = getelementptr inbounds nuw i8, ptr %src, i64 %datastart
  call void @llvm.memcpy.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %size, i1 false)
  %o.n = add i64 %o, %size
  %i.n = add i64 %dataend, 2
  br label %loop

full:
  ret i32 6

inc:
  ret i32 11

bad:
  ret i32 13
}

; ================================================================ connection

; universe_http_conn_create(fd, bufcap) -> ptr (null on failure). bufcap 0 =>
;   65536. Owns the read accumulation buffer and a bufio writer on the fd.
define noalias ptr @universe_http_conn_create(i32 %fd, i64 %bufcap) local_unnamed_addr #2 {
entry:
  %is0 = icmp eq i64 %bufcap, 0
  %cap = select i1 %is0, i64 65536, i64 %bufcap
  %toobig = icmp ugt i64 %cap, 1099511627776
  br i1 %toobig, label %fail, label %alloc

alloc:
  %total = add i64 %cap, 64
  %mem = call ptr @malloc(i64 %total)
  %memnull = icmp eq ptr %mem, null
  br i1 %memnull, label %fail, label %mkw

mkw:
  %w = call ptr @universe_io_writer_create(i32 %fd, i64 0)
  %wnull = icmp eq ptr %w, null
  br i1 %wnull, label %freemem, label %init

freemem:
  call void @free(ptr nonnull %mem)
  br label %fail

init:
  %fd64 = zext i32 %fd to i64
  store i64 %fd64, ptr %mem, align 8
  %capp = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %cap, ptr %capp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 0, ptr %lenp, align 8
  %posp = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 0, ptr %posp, align 8
  %wp = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store ptr %w, ptr %wp, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_http_conn_fd(ptr %conn) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %conn, null
  br i1 %n, label %bad, label %ok
ok:
  %fd64 = load i64, ptr %conn, align 8
  %fd = trunc i64 %fd64 to i32
  ret i32 %fd
bad:
  ret i32 -1
}

define ptr @universe_http_conn_writer(ptr %conn) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %conn, null
  br i1 %n, label %bad, label %ok
ok:
  %wp = getelementptr inbounds nuw i8, ptr %conn, i64 32
  %w = load ptr, ptr %wp, align 8
  ret ptr %w
bad:
  ret ptr null
}

; destroy: flush+free the writer, close the fd, free the connection.
define void @universe_http_conn_destroy(ptr %conn) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %conn, null
  br i1 %n, label %done, label %do

do:
  %wp = getelementptr inbounds nuw i8, ptr %conn, i64 32
  %w = load ptr, ptr %wp, align 8
  call void @universe_io_writer_destroy(ptr %w)
  %fd64 = load i64, ptr %conn, align 8
  %fd = trunc i64 %fd64 to i32
  %ign = call i32 @universe_net_tcp_close(i32 %fd)
  call void @free(ptr nonnull %conn)
  br label %done

done:
  ret void
}

; release: flush+free writer and free the conn, WITHOUT closing the fd; returns
; the fd so the caller can pool it for keep-alive reuse.
define i32 @universe_http_conn_release(ptr %conn) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %conn, null
  br i1 %n, label %bad, label %do
bad:
  ret i32 -1
do:
  %wp = getelementptr inbounds nuw i8, ptr %conn, i64 32
  %w = load ptr, ptr %wp, align 8
  call void @universe_io_writer_destroy(ptr %w)
  %fd64 = load i64, ptr %conn, align 8
  %fd = trunc i64 %fd64 to i32
  call void @free(ptr nonnull %conn)
  ret i32 %fd
}

; universe_http_connect(ip, port, bufcap) -> ptr conn (null on failure).
define ptr @universe_http_connect(i32 %ip, i32 %port, i64 %bufcap) local_unnamed_addr #2 {
entry:
  %fd = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %bad = icmp slt i32 %fd, 0
  br i1 %bad, label %fail, label %wrap
wrap:
  %conn = call ptr @universe_http_conn_create(i32 %fd, i64 %bufcap)
  %cn = icmp eq ptr %conn, null
  br i1 %cn, label %closefd, label %ok
closefd:
  %ign = call i32 @universe_net_tcp_close(i32 %fd)
  br label %fail
ok:
  ret ptr %conn
fail:
  ret ptr null
}

; universe_http_conn_read(conn, is_resp, msg, hdrs, cap) -> i32 : fill the
;   buffer with tcp_recv, parse the head (request/response by is_resp), then the
;   body per Content-Length / chunked / EOF. 0 OK, 4 clean EOF, 6 FULL, 13
;   PARSE, 15 IO.
define i32 @universe_http_conn_read(ptr %conn, i32 %is_resp, ptr %msg, ptr %hdrs, i64 %cap) local_unnamed_addr #2 {
entry:
  %rawp = alloca i64, align 8
  %fd64 = load i64, ptr %conn, align 8
  %fd = trunc i64 %fd64 to i32
  %capp = getelementptr inbounds nuw i8, ptr %conn, i64 8
  %capb = load i64, ptr %capp, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %conn, i64 16
  %len0 = load i64, ptr %lenp, align 8
  %posp = getelementptr inbounds nuw i8, ptr %conn, i64 24
  %pos0 = load i64, ptr %posp, align 8
  %buf = getelementptr inbounds nuw i8, ptr %conn, i64 64
  ; compact leftover [pos0, len0) to front
  %needcompact = icmp ugt i64 %pos0, 0
  br i1 %needcompact, label %compact, label %headloop

compact:
  %rem = sub i64 %len0, %pos0
  %csrc = getelementptr inbounds nuw i8, ptr %buf, i64 %pos0
  call void @llvm.memmove.p0.p0.i64(ptr %buf, ptr %csrc, i64 %rem, i1 false)
  br label %headloop

headloop:
  %len = phi i64 [ %len0, %entry ], [ %rem, %compact ], [ %len.h, %hadv ]
  %resp = icmp ne i32 %is_resp, 0
  br i1 %resp, label %pr, label %pq

pq:
  %stq = call i32 @universe_http_parse_request(ptr %buf, i64 %len, ptr %msg, ptr %hdrs, i64 %cap)
  br label %pjoin

pr:
  %str = call i32 @universe_http_parse_response(ptr %buf, i64 %len, ptr %msg, ptr %hdrs, i64 %cap)
  br label %pjoin

pjoin:
  %st = phi i32 [ %stq, %pq ], [ %str, %pr ]
  %ok = icmp eq i32 %st, 0
  br i1 %ok, label %body, label %chkinc

chkinc:
  %isinc = icmp eq i32 %st, 11
  br i1 %isinc, label %hmore, label %rethead

rethead:
  ret i32 %st

hmore:
  %hfull = icmp uge i64 %len, %capb
  br i1 %hfull, label %retfull, label %hrecv

hrecv:
  %hspace = sub i64 %capb, %len
  %hdst = getelementptr inbounds nuw i8, ptr %buf, i64 %len
  %hn = call i64 @universe_net_tcp_recv(i32 %fd, ptr %hdst, i64 %hspace)
  %hneg = icmp slt i64 %hn, 0
  br i1 %hneg, label %retio, label %hchk

hchk:
  %heof = icmp eq i64 %hn, 0
  br i1 %heof, label %hcloses, label %hadv

hcloses:
  %empty = icmp eq i64 %len, 0
  %code = select i1 %empty, i32 4, i32 13
  ret i32 %code

hadv:
  %len.h = add i64 %len, %hn
  br label %headloop

body:
  %headlen = call i64 @load_headlen(ptr %msg)
  %flagsp = getelementptr inbounds nuw i8, ptr %msg, i64 88
  %flags = load i64, ptr %flagsp, align 8
  %clp = getelementptr inbounds nuw i8, ptr %msg, i64 80
  %cl = load i64, ptr %clp, align 8
  %haschunk = and i64 %flags, 1
  %ischunk = icmp ne i64 %haschunk, 0
  %hascl = and i64 %flags, 4
  %iscl = icmp ne i64 %hascl, 0
  br i1 %iscl, label %clbody, label %chkchunk

clbody:
  %need = add i64 %headlen, %cl
  %needbig = icmp ugt i64 %need, %capb
  br i1 %needbig, label %retfull, label %clloop

clloop:
  %clen = phi i64 [ %len, %clbody ], [ %clen.n, %cladv ]
  %clhave = icmp uge i64 %clen, %need
  br i1 %clhave, label %clfin, label %clrecv

clrecv:
  %clspace = sub i64 %capb, %clen
  %cldst = getelementptr inbounds nuw i8, ptr %buf, i64 %clen
  %cln = call i64 @universe_net_tcp_recv(i32 %fd, ptr %cldst, i64 %clspace)
  %clneg = icmp slt i64 %cln, 0
  br i1 %clneg, label %retio, label %clchk

clchk:
  %cleof = icmp eq i64 %cln, 0
  br i1 %cleof, label %retparse, label %cladv

cladv:
  %clen.n = add i64 %clen, %cln
  br label %clloop

clfin:
  call void @store_state(ptr %conn, i64 %clen, i64 %need)
  ret i32 0

chkchunk:
  br i1 %ischunk, label %chloop, label %eofmode

chloop:
  %chlen = phi i64 [ %len, %chkchunk ], [ %chlen.n, %chadv ]
  %chraw = sub i64 %chlen, %headlen
  %chbase = getelementptr inbounds nuw i8, ptr %buf, i64 %headlen
  %chst = call i32 @chunked_span(ptr %chbase, i64 %chraw, ptr %rawp)
  %chdone = icmp eq i32 %chst, 0
  br i1 %chdone, label %chfin, label %chkchinc

chkchinc:
  %chisinc = icmp eq i32 %chst, 11
  br i1 %chisinc, label %chmore, label %retparse

chmore:
  %chfull = icmp uge i64 %chlen, %capb
  br i1 %chfull, label %retfull, label %chrecv

chrecv:
  %chspace = sub i64 %capb, %chlen
  %chdst = getelementptr inbounds nuw i8, ptr %buf, i64 %chlen
  %chn = call i64 @universe_net_tcp_recv(i32 %fd, ptr %chdst, i64 %chspace)
  %chneg = icmp slt i64 %chn, 0
  br i1 %chneg, label %retio, label %chnchk

chnchk:
  %cheof = icmp eq i64 %chn, 0
  br i1 %cheof, label %retparse, label %chadv

chadv:
  %chlen.n = add i64 %chlen, %chn
  br label %chloop

chfin:
  %raw = load i64, ptr %rawp, align 8
  %blp.ch = getelementptr inbounds nuw i8, ptr %msg, i64 64
  store i64 %raw, ptr %blp.ch, align 8
  %chnewpos = add i64 %headlen, %raw
  call void @store_state(ptr %conn, i64 %chlen, i64 %chnewpos)
  ret i32 0

eofmode:
  br i1 %resp, label %eofloop, label %nobody

eofloop:
  %elen = phi i64 [ %len, %eofmode ], [ %elen.n, %eadv ]
  %efull = icmp uge i64 %elen, %capb
  br i1 %efull, label %eofdone, label %erecv

erecv:
  %espace = sub i64 %capb, %elen
  %edst = getelementptr inbounds nuw i8, ptr %buf, i64 %elen
  %en = call i64 @universe_net_tcp_recv(i32 %fd, ptr %edst, i64 %espace)
  %eneg = icmp slt i64 %en, 0
  br i1 %eneg, label %retio, label %enchk

enchk:
  %eeof = icmp eq i64 %en, 0
  br i1 %eeof, label %eofdone, label %eadv

eadv:
  %elen.n = add i64 %elen, %en
  br label %eofloop

eofdone:
  %efin = phi i64 [ %elen, %eofloop ], [ %elen, %enchk ]
  %ebl = sub i64 %efin, %headlen
  %blp.e = getelementptr inbounds nuw i8, ptr %msg, i64 64
  store i64 %ebl, ptr %blp.e, align 8
  call void @store_state(ptr %conn, i64 %efin, i64 %efin)
  ret i32 0

nobody:
  call void @store_state(ptr %conn, i64 %len, i64 %headlen)
  ret i32 0

retparse:
  ret i32 13

retfull:
  ret i32 6

retio:
  ret i32 15
}

; small internal accessors/mutators to keep conn_read readable
define internal i64 @load_headlen(ptr %msg) #0 {
entry:
  %p = getelementptr inbounds nuw i8, ptr %msg, i64 72
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

define internal void @store_state(ptr %conn, i64 %len, i64 %pos) #0 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %conn, i64 16
  store i64 %len, ptr %lp, align 8
  %pp = getelementptr inbounds nuw i8, ptr %conn, i64 24
  store i64 %pos, ptr %pp, align 8
  ret void
}

; ================================================================ client

; universe_http_client_request(conn, method,mlen, target,tlen, minor, hdrs,
;   count, body,blen, keepalive, resp_msg, resp_hdrs, resp_cap) -> i32.
define i32 @universe_http_client_request(ptr %conn, ptr %method, i64 %mlen, ptr %target, i64 %tlen, i64 %minor, ptr %hdrs, i64 %count, ptr %body, i64 %blen, i32 %keepalive, ptr %resp_msg, ptr %resp_hdrs, i64 %resp_cap) local_unnamed_addr #2 {
entry:
  %cn = icmp eq ptr %conn, null
  br i1 %cn, label %err.null, label %send

err.null:
  ret i32 1

send:
  %wp = getelementptr inbounds nuw i8, ptr %conn, i64 32
  %w = load ptr, ptr %wp, align 8
  %st = call i32 @universe_http_write_request(ptr %w, ptr %method, i64 %mlen, ptr %target, i64 %tlen, i64 %minor, ptr %hdrs, i64 %count, ptr %body, i64 %blen, i32 %keepalive)
  %stbad = icmp ne i32 %st, 0
  br i1 %stbad, label %retst, label %flush

retst:
  ret i32 %st

flush:
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  %fbad = icmp ne i32 %fs, 0
  br i1 %fbad, label %retio, label %read

retio:
  ret i32 15

read:
  %r = call i32 @universe_http_conn_read(ptr %conn, i32 1, ptr %resp_msg, ptr %resp_hdrs, i64 %resp_cap)
  ret i32 %r
}

; ================================================================ server

; universe_http_serve_conn(conn, handler, userdata) -> i32 : blocking keep-alive
;   loop. handler: void(userdata, req_msg, hdrs, count, resp_spec).
define i32 @universe_http_serve_conn(ptr %conn, ptr %handler, ptr %userdata) local_unnamed_addr #2 {
entry:
  %req = alloca [96 x i8], align 8
  %hdrs = alloca [2048 x i8], align 8
  %resp = alloca [56 x i8], align 8
  %cn = icmp eq ptr %conn, null
  %hn = icmp eq ptr %handler, null
  %anynull = or i1 %cn, %hn
  br i1 %anynull, label %err.null, label %setup

err.null:
  ret i32 1

setup:
  %wp = getelementptr inbounds nuw i8, ptr %conn, i64 32
  %w = load ptr, ptr %wp, align 8
  br label %loop

loop:
  %st = call i32 @universe_http_conn_read(ptr %conn, i32 0, ptr %req, ptr %hdrs, i64 64)
  %iseof = icmp eq i32 %st, 4
  br i1 %iseof, label %clean, label %chkok

clean:
  ret i32 0

chkok:
  %isok = icmp eq i32 %st, 0
  br i1 %isok, label %handle, label %reterr

reterr:
  ret i32 %st

handle:
  %rflp = getelementptr inbounds nuw i8, ptr %req, i64 88
  %rflags = load i64, ptr %rflp, align 8
  %kabit = and i64 %rflags, 2
  %kalive = icmp ne i64 %kabit, 0
  %hcp = getelementptr inbounds nuw i8, ptr %req, i64 48
  %hc = load i64, ptr %hcp, align 8
  ; zero the resp spec so an inattentive handler yields a valid empty response
  store i64 0, ptr %resp, align 8
  %rz1 = getelementptr inbounds nuw i8, ptr %resp, i64 8
  store i64 0, ptr %rz1, align 8
  %rz2 = getelementptr inbounds nuw i8, ptr %resp, i64 16
  store i64 0, ptr %rz2, align 8
  %rz3 = getelementptr inbounds nuw i8, ptr %resp, i64 24
  store i64 0, ptr %rz3, align 8
  %rz4 = getelementptr inbounds nuw i8, ptr %resp, i64 32
  store i64 0, ptr %rz4, align 8
  %rz5 = getelementptr inbounds nuw i8, ptr %resp, i64 40
  store i64 0, ptr %rz5, align 8
  %rz6 = getelementptr inbounds nuw i8, ptr %resp, i64 48
  store i64 0, ptr %rz6, align 8
  call void %handler(ptr %userdata, ptr %req, ptr %hdrs, i64 %hc, ptr %resp)
  %status = load i64, ptr %resp, align 8
  %reap = getelementptr inbounds nuw i8, ptr %resp, i64 8
  %reason = load ptr, ptr %reap, align 8
  %relp = getelementptr inbounds nuw i8, ptr %resp, i64 16
  %rel = load i64, ptr %relp, align 8
  %rhp = getelementptr inbounds nuw i8, ptr %resp, i64 24
  %rh = load ptr, ptr %rhp, align 8
  %rhcp = getelementptr inbounds nuw i8, ptr %resp, i64 32
  %rhc = load i64, ptr %rhcp, align 8
  %rbp = getelementptr inbounds nuw i8, ptr %resp, i64 40
  %rb = load ptr, ptr %rbp, align 8
  %rblp = getelementptr inbounds nuw i8, ptr %resp, i64 48
  %rbl = load i64, ptr %rblp, align 8
  %kai = zext i1 %kalive to i32
  %wst = call i32 @universe_http_write_response(ptr %w, i64 %status, ptr %reason, i64 %rel, ptr %rh, i64 %rhc, ptr %rb, i64 %rbl, i32 %kai)
  %fs = call i32 @universe_io_writer_flush(ptr %w)
  %e1 = or i32 %wst, %fs
  %ebad = icmp ne i32 %e1, 0
  br i1 %ebad, label %retio, label %cont

retio:
  ret i32 15

cont:
  br i1 %kalive, label %loop, label %clean
}

; universe_http_accept_loop(listenfd, handler, userdata, bufcap, max_conns)
;   -> i32. Single-threaded accept/serve loop; max_conns<0 => run forever.
define i32 @universe_http_accept_loop(i32 %listenfd, ptr %handler, ptr %userdata, i64 %bufcap, i64 %max_conns) local_unnamed_addr #2 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %after ]
  %unbounded = icmp slt i64 %max_conns, 0
  %within = icmp ult i64 %i, %max_conns
  %go = or i1 %unbounded, %within
  br i1 %go, label %accept, label %done

accept:
  %fd = call i32 @universe_net_tcp_accept(i32 %listenfd, ptr null)
  %fdbad = icmp slt i32 %fd, 0
  br i1 %fdbad, label %retio, label %mkconn

mkconn:
  %conn = call ptr @universe_http_conn_create(i32 %fd, i64 %bufcap)
  %cn = icmp eq ptr %conn, null
  br i1 %cn, label %closefd, label %serve

closefd:
  %ign = call i32 @universe_net_tcp_close(i32 %fd)
  ret i32 2

serve:
  %sc = call i32 @universe_http_serve_conn(ptr %conn, ptr %handler, ptr %userdata)
  call void @universe_http_conn_destroy(ptr %conn)
  br label %after

after:
  %i.n = add i64 %i, 1
  br label %loop

done:
  ret i32 0

retio:
  ret i32 15
}

; universe_http_request_pooled(pool, ip, port, method,mlen, target,tlen, minor,
;   hdrs,count, body,blen, resp_msg, resp_hdrs, resp_cap, bufcap) -> i32.
;   Reuses an idle pooled connection when available; returns the fd to the pool
;   for keep-alive when the response permits.
define i32 @universe_http_request_pooled(ptr %pool, i32 %ip, i32 %port, ptr %method, i64 %mlen, ptr %target, i64 %tlen, i64 %minor, ptr %hdrs, i64 %count, ptr %body, i64 %blen, ptr %resp_msg, ptr %resp_hdrs, i64 %resp_cap, i64 %bufcap) local_unnamed_addr #2 {
entry:
  %got = call i32 @universe_net_pool_get(ptr %pool, i32 %ip, i32 %port)
  %hasconn = icmp sge i32 %got, 0
  br i1 %hasconn, label %havefd, label %dial

dial:
  %nfd = call i32 @universe_net_tcp_connect(i32 %ip, i32 %port)
  %nbad = icmp slt i32 %nfd, 0
  br i1 %nbad, label %retio, label %havefd

havefd:
  %fd = phi i32 [ %got, %entry ], [ %nfd, %dial ]
  %conn = call ptr @universe_http_conn_create(i32 %fd, i64 %bufcap)
  %cn = icmp eq ptr %conn, null
  br i1 %cn, label %closefd, label %req

closefd:
  %ign = call i32 @universe_net_tcp_close(i32 %fd)
  ret i32 2

req:
  %st = call i32 @universe_http_client_request(ptr %conn, ptr %method, i64 %mlen, ptr %target, i64 %tlen, i64 %minor, ptr %hdrs, i64 %count, ptr %body, i64 %blen, i32 1, ptr %resp_msg, ptr %resp_hdrs, i64 %resp_cap)
  %stbad = icmp ne i32 %st, 0
  br i1 %stbad, label %failconn, label %keep

failconn:
  call void @universe_http_conn_destroy(ptr %conn)
  ret i32 %st

keep:
  %rflp = getelementptr inbounds nuw i8, ptr %resp_msg, i64 88
  %rflags = load i64, ptr %rflp, align 8
  %kabit = and i64 %rflags, 2
  %kalive = icmp ne i64 %kabit, 0
  %relfd = call i32 @universe_http_conn_release(ptr %conn)
  br i1 %kalive, label %pool.put, label %pool.close

pool.put:
  %pr = call i32 @universe_net_pool_put(ptr %pool, i32 %relfd, i32 %ip, i32 %port)
  ret i32 0

pool.close:
  %ign2 = call i32 @universe_net_tcp_close(i32 %relfd)
  ret i32 0

retio:
  ret i32 15
}

attributes #0 = { alwaysinline nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
