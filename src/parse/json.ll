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

; JSON pull/SAX tokenizer (RFC 8259) over a caller-owned byte buffer.
;
; DESIGN:
;   * ZERO-COPY, ZERO-ALLOC hot path. The scanner never copies input: every
;     token reports {type, byte-offset, byte-length} as offsets INTO the
;     caller buffer. The only allocation is the one-shot scanner object
;     (header + a small container-type stack), created once and reused for the
;     whole document — no per-token/per-syscall allocation. This is the
;     canonical compute/memory/IO split: the caller fills the buffer (IO), the
;     scanner walks it (pure compute over memory it already owns), the caller
;     emits results; no read/write ever sits in the scan loop.
;   * PULL, not recursive-descent: `_next` returns exactly ONE meaningful token
;     per call and transparently consumes the structural separators ':' and ','
;     and whitespace between tokens. Nesting is tracked iteratively in an
;     explicit byte stack (0=object, 1=array) so there is NO native call-stack
;     recursion to blow — a configurable max-depth bounds the stack.
;   * The whitespace skip (@js_skip_ws) and string char-class scan
;     (@js_scan_string) are the hot leaves: tight countup loops, branch-lean,
;     no calls on the common byte (hex4 only on a rare '\u').
;   * A small state machine (7 states) validates grammar: object/array framing,
;     key:value pairing, comma separation, no trailing commas, single top value.
;   * ERROR CONVENTION: `_next` returns i32 status — 0 OK (token written; a
;     valid end-of-input yields a TOK_END token, type 10), 13 PARSE on any
;     malformed input (the failing byte offset is stored in the scanner and
;     readable via @universe_parse_json_error_offset), 8 INVALID_ARG for a null
;     scanner/token. The scalar helpers (@..._unescape, @..._number_double)
;     return a length / status and a NEGATIVE value (or 13) on malformed input.
;
; Token types (i32 at token+0): 0 begin-object, 1 end-object, 2 begin-array,
;   3 end-array, 4 string, 5 number, 6 true, 7 false, 8 null, 9 key, 10 end.
;   token+8  = i64 byte offset into buffer; token+16 = i64 byte length.
;   For string/key the offset/length span the CONTENT between the quotes
;   (still escaped — feed to @universe_parse_json_unescape to decode).
;
; Scanner layout (single malloc): buf@0, len@8, pos@16, err_off@24,
;   depth@32(i32), max_depth@36(i32), state@40(i32), _pad@44, stack@48.
;
; API:
;   ptr  universe_parse_json_scanner_create(ptr buf, i64 len, i32 max_depth)
;   void universe_parse_json_scanner_destroy(ptr sc)
;   void universe_parse_json_scanner_reset(ptr sc)          ; rewind to start
;   i32  universe_parse_json_next(ptr sc, ptr out_token)    ; 0/13/8
;   i64  universe_parse_json_error_offset(ptr sc)
;   i64  universe_parse_json_unescape(ptr dst, ptr src, i64 len)   ; len or -1
;   i32  universe_parse_json_number_double(ptr src, i64 len, ptr out) ; 0/13

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"

; ------------------------------------------------------------- literal tables
@js.true  = private unnamed_addr constant [4 x i8] c"true", align 1
@js.false = private unnamed_addr constant [5 x i8] c"false", align 1
@js.null  = private unnamed_addr constant [4 x i8] c"null", align 1

; ===================================================================== helpers

; Skip whitespace (space/tab/LF/CR). Returns index of first non-ws byte (or len).
; HOT LEAF: tight countup, no calls, branch-lean char class.
define internal i64 @js_skip_ws(ptr readonly %buf, i64 %len, i64 %pos) #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ %pos, %entry ], [ %i.next, %cont ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %done, label %body
body:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i32
  %sp = icmp eq i32 %cz, 32
  %tb = icmp eq i32 %cz, 9
  %nl = icmp eq i32 %cz, 10
  %cr = icmp eq i32 %cz, 13
  %w1 = or i1 %sp, %tb
  %w2 = or i1 %nl, %cr
  %ws = or i1 %w1, %w2
  br i1 %ws, label %cont, label %done
cont:
  %i.next = add nuw i64 %i, 1
  br label %loop
done:
  ret i64 %i
}

; Parse 4 hex digits at buf[off..off+3]. Returns 0..65535 or -1.
define internal i32 @js_hex4(ptr readonly %buf, i64 %off) #0 {
entry:
  br label %loop
loop:
  %k = phi i64 [ 0, %entry ], [ %k.next, %cont ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %cont ]
  %done = icmp uge i64 %k, 4
  br i1 %done, label %good, label %body
body:
  %i = add nuw i64 %off, %k
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i32
  %d09 = add i32 %cz, -48
  %is09 = icmp ult i32 %d09, 10
  %lc = or i32 %cz, 32
  %daf = add i32 %lc, -87
  %af.lo = icmp uge i32 %lc, 97
  %af.hi = icmp ule i32 %lc, 102
  %isaf = and i1 %af.lo, %af.hi
  %valid = or i1 %is09, %isaf
  br i1 %valid, label %cont, label %bad
cont:
  %val = select i1 %is09, i32 %d09, i32 %daf
  %sh = shl i32 %acc, 4
  %acc.next = add i32 %sh, %val
  %k.next = add nuw i64 %k, 1
  br label %loop
good:
  ret i32 %acc
bad:
  ret i32 -1
}

; Scan a JSON string starting at the opening quote %pos. Validates escapes and
; \uXXXX incl. surrogate pairs; rejects raw control chars. Returns the index
; just PAST the closing quote on success, or -(errpos+1) on any error.
; HOT LEAF for the common byte.
define internal i64 @js_scan_string(ptr readonly %buf, i64 %len, i64 %pos) #0 {
entry:
  %start1 = add nuw i64 %pos, 1
  br label %loop
loop:
  %i = phi i64 [ %start1, %entry ], [ %i.adv, %normal ], [ %i.esc2, %esc.simple ], [ %i.u.res, %resume_u ]
  %end = icmp uge i64 %i, %len
  br i1 %end, label %fail_at_i, label %rd
rd:
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i32
  %isQuote = icmp eq i32 %cz, 34
  br i1 %isQuote, label %close, label %chkesc
close:
  %endpos = add nuw i64 %i, 1
  ret i64 %endpos
chkesc:
  %isBS = icmp eq i32 %cz, 92
  br i1 %isBS, label %esc, label %chkctrl
chkctrl:
  %isCtrl = icmp ult i32 %cz, 32
  br i1 %isCtrl, label %fail_at_i, label %normal
normal:
  %i.adv = add nuw i64 %i, 1
  br label %loop
esc:
  %j = add nuw i64 %i, 1
  %jend = icmp uge i64 %j, %len
  br i1 %jend, label %fail_at_i, label %esc.rd
esc.rd:
  %pj = getelementptr inbounds nuw i8, ptr %buf, i64 %j
  %ej = load i8, ptr %pj, align 1
  %ejz = zext i8 %ej to i32
  %isU = icmp eq i32 %ejz, 117
  br i1 %isU, label %esc.u, label %esc.simplechk
esc.simplechk:
  switch i32 %ejz, label %fail_at_j [
    i32 34,  label %esc.simple
    i32 92,  label %esc.simple
    i32 47,  label %esc.simple
    i32 98,  label %esc.simple
    i32 102, label %esc.simple
    i32 110, label %esc.simple
    i32 114, label %esc.simple
    i32 116, label %esc.simple
  ]
esc.simple:
  %i.esc2 = add nuw i64 %j, 1
  br label %loop
esc.u:
  %h0 = add nuw i64 %j, 1
  %need = add nuw i64 %h0, 4
  %needbad = icmp ugt i64 %need, %len
  br i1 %needbad, label %fail_at_i, label %u.hex
u.hex:
  %cp1 = call i32 @js_hex4(ptr %buf, i64 %h0)
  %bad1 = icmp slt i32 %cp1, 0
  br i1 %bad1, label %fail_at_i, label %u.class
u.class:
  %hi.lo = icmp uge i32 %cp1, 55296
  %hi.hi = icmp ule i32 %cp1, 56319
  %isHigh = and i1 %hi.lo, %hi.hi
  %lo.lo = icmp uge i32 %cp1, 56320
  %lo.hi = icmp ule i32 %cp1, 57343
  %isLow = and i1 %lo.lo, %lo.hi
  br i1 %isLow, label %fail_at_i, label %u.chkhigh
u.chkhigh:
  br i1 %isHigh, label %need_pair, label %u.single
u.single:
  br label %resume_u
need_pair:
  %q0 = add nuw i64 %h0, 4
  %need2 = add nuw i64 %q0, 6
  %need2bad = icmp ugt i64 %need2, %len
  br i1 %need2bad, label %fail_at_i, label %pair.read
pair.read:
  %pbs = getelementptr inbounds nuw i8, ptr %buf, i64 %q0
  %bs = load i8, ptr %pbs, align 1
  %isbs = icmp eq i8 %bs, 92
  %q1 = add nuw i64 %q0, 1
  %puc = getelementptr inbounds nuw i8, ptr %buf, i64 %q1
  %uc = load i8, ptr %puc, align 1
  %isu2 = icmp eq i8 %uc, 117
  %both = and i1 %isbs, %isu2
  br i1 %both, label %pair.hex, label %fail_at_i
pair.hex:
  %h2 = add nuw i64 %q0, 2
  %cp2 = call i32 @js_hex4(ptr %buf, i64 %h2)
  %bad2 = icmp slt i32 %cp2, 0
  br i1 %bad2, label %fail_at_i, label %pair.class
pair.class:
  %l2.lo = icmp uge i32 %cp2, 56320
  %l2.hi = icmp ule i32 %cp2, 57343
  %isLow2 = and i1 %l2.lo, %l2.hi
  br i1 %isLow2, label %pair.done, label %fail_at_i
pair.done:
  br label %resume_u
resume_u:
  %i.u.res = phi i64 [ %need, %u.single ], [ %need2, %pair.done ]
  br label %loop
fail_at_i:
  %neg.i = sub nsw i64 -1, %i
  ret i64 %neg.i
fail_at_j:
  %neg.j = sub nsw i64 -1, %j
  ret i64 %neg.j
}

; Scan a JSON number at %pos (first char '-' or digit). Returns end index or
; -(errpos+1). Enforces: no leading zeros, digit required after '.', digit
; required in exponent, optional sign only after 'e'/'E'.
define internal i64 @js_scan_number(ptr readonly %buf, i64 %len, i64 %pos) #0 {
entry:
  %p0 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  %c0 = load i8, ptr %p0, align 1
  %c0z = zext i8 %c0 to i32
  %isMinus = icmp eq i32 %c0z, 45
  %i0 = select i1 %isMinus, i64 1, i64 0
  %si = add nuw i64 %pos, %i0
  %si.end = icmp uge i64 %si, %len
  br i1 %si.end, label %fail_at_si, label %int1
int1:
  %ps = getelementptr inbounds nuw i8, ptr %buf, i64 %si
  %cs = load i8, ptr %ps, align 1
  %csz = zext i8 %cs to i32
  %isZero = icmp eq i32 %csz, 48
  %d19 = add i32 %csz, -49
  %is19 = icmp ult i32 %d19, 9
  br i1 %isZero, label %after.int, label %chk19
chk19:
  br i1 %is19, label %int.more.pre, label %fail_at_si
int.more.pre:
  %si1 = add nuw i64 %si, 1
  br label %int.more
int.more:
  %im = phi i64 [ %si1, %int.more.pre ], [ %im.next, %int.more.c ]
  %im.end = icmp uge i64 %im, %len
  br i1 %im.end, label %num.done, label %int.more.rd
int.more.rd:
  %pim = getelementptr inbounds nuw i8, ptr %buf, i64 %im
  %cim = load i8, ptr %pim, align 1
  %cimz = zext i8 %cim to i32
  %imd = add i32 %cimz, -48
  %im.isd = icmp ult i32 %imd, 10
  br i1 %im.isd, label %int.more.c, label %after.int.from.more
int.more.c:
  %im.next = add nuw i64 %im, 1
  br label %int.more
; after single '0' the cursor is si+1
after.int:
  %ai0 = add nuw i64 %si, 1
  br label %frac
after.int.from.more:
  br label %frac
frac:
  %fi = phi i64 [ %ai0, %after.int ], [ %im, %after.int.from.more ]
  %fi.end = icmp uge i64 %fi, %len
  br i1 %fi.end, label %num.done.f, label %frac.rd
frac.rd:
  %pfi = getelementptr inbounds nuw i8, ptr %buf, i64 %fi
  %cfi = load i8, ptr %pfi, align 1
  %cfiz = zext i8 %cfi to i32
  %isDot = icmp eq i32 %cfiz, 46
  br i1 %isDot, label %frac.dig1, label %expo
frac.dig1:
  %fd1 = add nuw i64 %fi, 1
  %fd1.end = icmp uge i64 %fd1, %len
  br i1 %fd1.end, label %fail_at_fi, label %frac.dig1.rd
frac.dig1.rd:
  %pfd1 = getelementptr inbounds nuw i8, ptr %buf, i64 %fd1
  %cfd1 = load i8, ptr %pfd1, align 1
  %cfd1z = zext i8 %cfd1 to i32
  %fd1d = add i32 %cfd1z, -48
  %fd1.isd = icmp ult i32 %fd1d, 10
  br i1 %fd1.isd, label %frac.more.pre, label %fail_at_fi
frac.more.pre:
  %fm1 = add nuw i64 %fd1, 1
  br label %frac.more
frac.more:
  %fm = phi i64 [ %fm1, %frac.more.pre ], [ %fm.next, %frac.more.c ]
  %fm.end = icmp uge i64 %fm, %len
  br i1 %fm.end, label %num.done.fm, label %frac.more.rd
frac.more.rd:
  %pfm = getelementptr inbounds nuw i8, ptr %buf, i64 %fm
  %cfm = load i8, ptr %pfm, align 1
  %cfmz = zext i8 %cfm to i32
  %fmd = add i32 %cfmz, -48
  %fm.isd = icmp ult i32 %fmd, 10
  br i1 %fm.isd, label %frac.more.c, label %expo.from.fm
frac.more.c:
  %fm.next = add nuw i64 %fm, 1
  br label %frac.more
expo.from.fm:
  br label %expo
expo:
  %ei = phi i64 [ %fi, %frac.rd ], [ %fm, %expo.from.fm ]
  %ei.end = icmp uge i64 %ei, %len
  br i1 %ei.end, label %num.done.e, label %expo.rd
expo.rd:
  %pei = getelementptr inbounds nuw i8, ptr %buf, i64 %ei
  %cei = load i8, ptr %pei, align 1
  %ceiz = zext i8 %cei to i32
  %ise = icmp eq i32 %ceiz, 101
  %isE = icmp eq i32 %ceiz, 69
  %ise.any = or i1 %ise, %isE
  br i1 %ise.any, label %expo.sign, label %num.done.e
expo.sign:
  %es = add nuw i64 %ei, 1
  %es.end = icmp uge i64 %es, %len
  br i1 %es.end, label %fail_at_ei, label %expo.sign.rd
expo.sign.rd:
  %pes = getelementptr inbounds nuw i8, ptr %buf, i64 %es
  %ces = load i8, ptr %pes, align 1
  %cesz = zext i8 %ces to i32
  %isPlus = icmp eq i32 %cesz, 43
  %isNeg = icmp eq i32 %cesz, 45
  %issign = or i1 %isPlus, %isNeg
  %es2 = select i1 %issign, i64 1, i64 0
  %ed0 = add nuw i64 %es, %es2
  %ed0.end = icmp uge i64 %ed0, %len
  br i1 %ed0.end, label %fail_at_ei, label %expo.dig1
expo.dig1:
  %ped0 = getelementptr inbounds nuw i8, ptr %buf, i64 %ed0
  %ced0 = load i8, ptr %ped0, align 1
  %ced0z = zext i8 %ced0 to i32
  %ed0d = add i32 %ced0z, -48
  %ed0.isd = icmp ult i32 %ed0d, 10
  br i1 %ed0.isd, label %expo.more.pre, label %fail_at_ei
expo.more.pre:
  %em1 = add nuw i64 %ed0, 1
  br label %expo.more
expo.more:
  %em = phi i64 [ %em1, %expo.more.pre ], [ %em.next, %expo.more.c ]
  %em.end = icmp uge i64 %em, %len
  br i1 %em.end, label %num.done.em, label %expo.more.rd
expo.more.rd:
  %pem = getelementptr inbounds nuw i8, ptr %buf, i64 %em
  %cem = load i8, ptr %pem, align 1
  %cemz = zext i8 %cem to i32
  %emd = add i32 %cemz, -48
  %em.isd = icmp ult i32 %emd, 10
  br i1 %em.isd, label %expo.more.c, label %num.done.em
expo.more.c:
  %em.next = add nuw i64 %em, 1
  br label %expo.more
num.done:
  ret i64 %im
num.done.f:
  ret i64 %fi
num.done.fm:
  ret i64 %fm
num.done.e:
  ret i64 %ei
num.done.em:
  ret i64 %em
fail_at_si:
  %neg.si = sub nsw i64 -1, %si
  ret i64 %neg.si
fail_at_fi:
  %neg.fi = sub nsw i64 -1, %fi
  ret i64 %neg.fi
fail_at_ei:
  %neg.ei = sub nsw i64 -1, %ei
  ret i64 %neg.ei
}

; Compare buf[pos..] against a literal; returns true on exact prefix match.
define internal i1 @js_lit_ok(ptr readonly %buf, i64 %len, i64 %pos, ptr readonly %lit, i64 %litlen) #0 {
entry:
  %need = add i64 %pos, %litlen
  %ovf = icmp ugt i64 %need, %len
  br i1 %ovf, label %no, label %loop
loop:
  %k = phi i64 [ 0, %entry ], [ %k.next, %cont ]
  %done = icmp uge i64 %k, %litlen
  br i1 %done, label %yes, label %cmp
cmp:
  %bi = add nuw i64 %pos, %k
  %bp = getelementptr inbounds nuw i8, ptr %buf, i64 %bi
  %bc = load i8, ptr %bp, align 1
  %lp = getelementptr inbounds nuw i8, ptr %lit, i64 %k
  %lc = load i8, ptr %lp, align 1
  %eq = icmp eq i8 %bc, %lc
  br i1 %eq, label %cont, label %no
cont:
  %k.next = add nuw i64 %k, 1
  br label %loop
yes:
  ret i1 true
no:
  ret i1 false
}

; Store err_off and return PARSE(13).
define internal i32 @js_fail(ptr %sc, i64 %errpos) #1 {
entry:
  %ep = getelementptr inbounds nuw i8, ptr %sc, i64 24
  store i64 %errpos, ptr %ep, align 8
  ret i32 13
}

; Write a token and advance scanner (pos + state); returns 0.
define internal i32 @js_emit(ptr %sc, ptr %tok, i32 %type, i64 %off, i64 %tlen, i64 %newpos, i32 %newstate) #1 {
entry:
  store i32 %type, ptr %tok, align 4
  %op = getelementptr inbounds nuw i8, ptr %tok, i64 8
  store i64 %off, ptr %op, align 8
  %lp = getelementptr inbounds nuw i8, ptr %tok, i64 16
  store i64 %tlen, ptr %lp, align 8
  %pp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  store i64 %newpos, ptr %pp, align 8
  %sp = getelementptr inbounds nuw i8, ptr %sc, i64 40
  store i32 %newstate, ptr %sp, align 4
  ret i32 0
}

; Post-value state from current depth/stack: 7 done (top), 5 obj-next, 6 arr-next.
define internal i32 @js_post_state(ptr readonly %sc) #1 {
entry:
  %dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  %d = load i32, ptr %dp, align 4
  %top0 = icmp eq i32 %d, 0
  br i1 %top0, label %done7, label %chk
done7:
  ret i32 7
chk:
  %dm1 = sub i32 %d, 1
  %dm1x = zext i32 %dm1 to i64
  %stk = getelementptr inbounds nuw i8, ptr %sc, i64 48
  %slot = getelementptr inbounds nuw i8, ptr %stk, i64 %dm1x
  %tv = load i8, ptr %slot, align 1
  %isObj = icmp eq i8 %tv, 0
  %st = select i1 %isObj, i32 5, i32 6
  ret i32 %st
}

; Dispatch a value that starts at %vpos (whitespace already skipped, vpos<len).
define internal i32 @js_dispatch(ptr %sc, ptr %tok, i64 %vpos) #1 {
entry:
  %buf = load ptr, ptr %sc, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  %len = load i64, ptr %lenp, align 8
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %vpos
  %c = load i8, ptr %p, align 1
  %cz = zext i8 %c to i32
  switch i32 %cz, label %maybe.num [
    i32 123, label %beginobj
    i32 91,  label %beginarr
    i32 34,  label %vstr
    i32 116, label %vtrue
    i32 102, label %vfalse
    i32 110, label %vnull
  ]
beginobj:
  %o.dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  %o.d = load i32, ptr %o.dp, align 4
  %o.mp = getelementptr inbounds nuw i8, ptr %sc, i64 36
  %o.md = load i32, ptr %o.mp, align 4
  %o.full = icmp sge i32 %o.d, %o.md
  br i1 %o.full, label %o.depthfail, label %o.push
o.depthfail:
  %o.df = call i32 @js_fail(ptr %sc, i64 %vpos)
  ret i32 %o.df
o.push:
  %o.dx = zext i32 %o.d to i64
  %o.stk = getelementptr inbounds nuw i8, ptr %sc, i64 48
  %o.slot = getelementptr inbounds nuw i8, ptr %o.stk, i64 %o.dx
  store i8 0, ptr %o.slot, align 1
  %o.d1 = add i32 %o.d, 1
  store i32 %o.d1, ptr %o.dp, align 4
  %o.np = add nuw i64 %vpos, 1
  %o.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 0, i64 %vpos, i64 1, i64 %o.np, i32 1)
  ret i32 %o.r
beginarr:
  %a.dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  %a.d = load i32, ptr %a.dp, align 4
  %a.mp = getelementptr inbounds nuw i8, ptr %sc, i64 36
  %a.md = load i32, ptr %a.mp, align 4
  %a.full = icmp sge i32 %a.d, %a.md
  br i1 %a.full, label %a.depthfail, label %a.push
a.depthfail:
  %a.df = call i32 @js_fail(ptr %sc, i64 %vpos)
  ret i32 %a.df
a.push:
  %a.dx = zext i32 %a.d to i64
  %a.stk = getelementptr inbounds nuw i8, ptr %sc, i64 48
  %a.slot = getelementptr inbounds nuw i8, ptr %a.stk, i64 %a.dx
  store i8 1, ptr %a.slot, align 1
  %a.d1 = add i32 %a.d, 1
  store i32 %a.d1, ptr %a.dp, align 4
  %a.np = add nuw i64 %vpos, 1
  %a.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 2, i64 %vpos, i64 1, i64 %a.np, i32 4)
  ret i32 %a.r
vstr:
  %s.endq = call i64 @js_scan_string(ptr %buf, i64 %len, i64 %vpos)
  %s.bad = icmp slt i64 %s.endq, 0
  br i1 %s.bad, label %s.fail, label %s.ok
s.fail:
  %s.negr = sub i64 0, %s.endq
  %s.ep = sub i64 %s.negr, 1
  %s.f = call i32 @js_fail(ptr %sc, i64 %s.ep)
  ret i32 %s.f
s.ok:
  %s.coff = add nuw i64 %vpos, 1
  %s.clen0 = sub i64 %s.endq, %s.coff
  %s.clen = sub i64 %s.clen0, 1
  %s.ns = call i32 @js_post_state(ptr %sc)
  %s.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 4, i64 %s.coff, i64 %s.clen, i64 %s.endq, i32 %s.ns)
  ret i32 %s.r
maybe.num:
  %n.mm = icmp eq i32 %cz, 45
  %n.sub = add i32 %cz, -48
  %n.isd = icmp ult i32 %n.sub, 10
  %n.isnum = or i1 %n.mm, %n.isd
  br i1 %n.isnum, label %vnum, label %cfail
cfail:
  %cf = call i32 @js_fail(ptr %sc, i64 %vpos)
  ret i32 %cf
vnum:
  %n.endn = call i64 @js_scan_number(ptr %buf, i64 %len, i64 %vpos)
  %n.bad = icmp slt i64 %n.endn, 0
  br i1 %n.bad, label %n.fail, label %n.ok
n.fail:
  %n.negr = sub i64 0, %n.endn
  %n.ep = sub i64 %n.negr, 1
  %n.f = call i32 @js_fail(ptr %sc, i64 %n.ep)
  ret i32 %n.f
n.ok:
  %n.len = sub i64 %n.endn, %vpos
  %n.ns = call i32 @js_post_state(ptr %sc)
  %n.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 5, i64 %vpos, i64 %n.len, i64 %n.endn, i32 %n.ns)
  ret i32 %n.r
vtrue:
  %t.ok = call i1 @js_lit_ok(ptr %buf, i64 %len, i64 %vpos, ptr @js.true, i64 4)
  br i1 %t.ok, label %t.go, label %cfail
t.go:
  %t.np = add nuw i64 %vpos, 4
  %t.ns = call i32 @js_post_state(ptr %sc)
  %t.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 6, i64 %vpos, i64 4, i64 %t.np, i32 %t.ns)
  ret i32 %t.r
vfalse:
  %f.ok = call i1 @js_lit_ok(ptr %buf, i64 %len, i64 %vpos, ptr @js.false, i64 5)
  br i1 %f.ok, label %f.go, label %cfail
f.go:
  %f.np = add nuw i64 %vpos, 5
  %f.ns = call i32 @js_post_state(ptr %sc)
  %f.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 7, i64 %vpos, i64 5, i64 %f.np, i32 %f.ns)
  ret i32 %f.r
vnull:
  %u.ok = call i1 @js_lit_ok(ptr %buf, i64 %len, i64 %vpos, ptr @js.null, i64 4)
  br i1 %u.ok, label %u.go, label %cfail
u.go:
  %u.np = add nuw i64 %vpos, 4
  %u.ns = call i32 @js_post_state(ptr %sc)
  %u.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 8, i64 %vpos, i64 4, i64 %u.np, i32 %u.ns)
  ret i32 %u.r
}

; Scan an object key (string) starting at %kpos; sets state to ST_COLON(3).
define internal i32 @js_scan_key(ptr %sc, ptr %tok, i64 %kpos) #1 {
entry:
  %buf = load ptr, ptr %sc, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  %len = load i64, ptr %lenp, align 8
  %endq = call i64 @js_scan_string(ptr %buf, i64 %len, i64 %kpos)
  %bad = icmp slt i64 %endq, 0
  br i1 %bad, label %fail, label %ok
fail:
  %negr = sub i64 0, %endq
  %ep = sub i64 %negr, 1
  %f = call i32 @js_fail(ptr %sc, i64 %ep)
  ret i32 %f
ok:
  %coff = add nuw i64 %kpos, 1
  %clen0 = sub i64 %endq, %coff
  %clen = sub i64 %clen0, 1
  %r = call i32 @js_emit(ptr %sc, ptr %tok, i32 9, i64 %coff, i64 %clen, i64 %endq, i32 3)
  ret i32 %r
}

; ================================================================= public API

define noalias ptr @universe_parse_json_scanner_create(ptr %buf, i64 %len, i32 %max_depth) local_unnamed_addr #2 {
entry:
  %bad = icmp sle i32 %max_depth, 0
  %md = select i1 %bad, i32 128, i32 %max_depth
  %mdx = zext i32 %md to i64
  %sz = add nuw i64 48, %mdx
  %sc = call ptr @malloc(i64 %sz)
  %null = icmp eq ptr %sc, null
  br i1 %null, label %fail, label %init
fail:
  ret ptr null
init:
  store ptr %buf, ptr %sc, align 8
  %lp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  store i64 %len, ptr %lp, align 8
  %pp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  store i64 0, ptr %pp, align 8
  %ep = getelementptr inbounds nuw i8, ptr %sc, i64 24
  store i64 0, ptr %ep, align 8
  %dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  store i32 0, ptr %dp, align 4
  %mp = getelementptr inbounds nuw i8, ptr %sc, i64 36
  store i32 %md, ptr %mp, align 4
  %stp = getelementptr inbounds nuw i8, ptr %sc, i64 40
  store i32 0, ptr %stp, align 4
  ret ptr %sc
}

define void @universe_parse_json_scanner_destroy(ptr %sc) local_unnamed_addr #2 {
entry:
  call void @free(ptr %sc)
  ret void
}

define void @universe_parse_json_scanner_reset(ptr %sc) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %sc, null
  br i1 %null, label %done, label %do
do:
  %pp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  store i64 0, ptr %pp, align 8
  %ep = getelementptr inbounds nuw i8, ptr %sc, i64 24
  store i64 0, ptr %ep, align 8
  %dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  store i32 0, ptr %dp, align 4
  %stp = getelementptr inbounds nuw i8, ptr %sc, i64 40
  store i32 0, ptr %stp, align 4
  br label %done
done:
  ret void
}

define i64 @universe_parse_json_error_offset(ptr readonly %sc) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %sc, null
  br i1 %null, label %z, label %rd
z:
  ret i64 0
rd:
  %ep = getelementptr inbounds nuw i8, ptr %sc, i64 24
  %e = load i64, ptr %ep, align 8
  ret i64 %e
}

define i32 @universe_parse_json_next(ptr %sc, ptr %tok) local_unnamed_addr #1 {
entry:
  %sc.null = icmp eq ptr %sc, null
  %tok.null = icmp eq ptr %tok, null
  %anynull = or i1 %sc.null, %tok.null
  br i1 %anynull, label %arg.err, label %load
arg.err:
  ret i32 8
load:
  %buf = load ptr, ptr %sc, align 8
  %lenp = getelementptr inbounds nuw i8, ptr %sc, i64 8
  %len = load i64, ptr %lenp, align 8
  %posp = getelementptr inbounds nuw i8, ptr %sc, i64 16
  %pos = load i64, ptr %posp, align 8
  %statep = getelementptr inbounds nuw i8, ptr %sc, i64 40
  %state = load i32, ptr %statep, align 4
  %p0 = call i64 @js_skip_ws(ptr %buf, i64 %len, i64 %pos)
  switch i32 %state, label %st.bad [
    i32 0, label %L.start
    i32 1, label %L.keyfirst
    i32 3, label %L.colon
    i32 4, label %L.valfirst
    i32 5, label %L.objnext
    i32 6, label %L.arrnext
    i32 7, label %L.done
  ]
st.bad:
  %sb = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %sb

L.start:
  %st.eof = icmp uge i64 %p0, %len
  br i1 %st.eof, label %st.err, label %st.disp
st.err:
  %st.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %st.e
st.disp:
  %st.r = call i32 @js_dispatch(ptr %sc, ptr %tok, i64 %p0)
  ret i32 %st.r

L.done:
  %dn.eof = icmp uge i64 %p0, %len
  br i1 %dn.eof, label %dn.end, label %dn.err
dn.end:
  %dn.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 10, i64 %p0, i64 0, i64 %p0, i32 7)
  ret i32 %dn.r
dn.err:
  %dn.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %dn.e

L.valfirst:
  %vf.eof = icmp uge i64 %p0, %len
  br i1 %vf.eof, label %vf.err, label %vf.rd
vf.err:
  %vf.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %vf.e
vf.rd:
  %vf.p = getelementptr inbounds nuw i8, ptr %buf, i64 %p0
  %vf.c = load i8, ptr %vf.p, align 1
  %vf.close = icmp eq i8 %vf.c, 93
  br i1 %vf.close, label %close.arr, label %vf.disp
vf.disp:
  %vf.r = call i32 @js_dispatch(ptr %sc, ptr %tok, i64 %p0)
  ret i32 %vf.r

L.keyfirst:
  %kf.eof = icmp uge i64 %p0, %len
  br i1 %kf.eof, label %kf.err, label %kf.rd
kf.err:
  %kf.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %kf.e
kf.rd:
  %kf.p = getelementptr inbounds nuw i8, ptr %buf, i64 %p0
  %kf.c = load i8, ptr %kf.p, align 1
  %kf.close = icmp eq i8 %kf.c, 125
  br i1 %kf.close, label %close.obj, label %kf.q
kf.q:
  %kf.quote = icmp eq i8 %kf.c, 34
  br i1 %kf.quote, label %kf.key, label %kf.err2
kf.err2:
  %kf.e2 = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %kf.e2
kf.key:
  %kf.r = call i32 @js_scan_key(ptr %sc, ptr %tok, i64 %p0)
  ret i32 %kf.r

L.colon:
  %co.eof = icmp uge i64 %p0, %len
  br i1 %co.eof, label %co.err, label %co.rd
co.err:
  %co.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %co.e
co.rd:
  %co.p = getelementptr inbounds nuw i8, ptr %buf, i64 %p0
  %co.c = load i8, ptr %co.p, align 1
  %co.colon = icmp eq i8 %co.c, 58
  br i1 %co.colon, label %co.go, label %co.err2
co.err2:
  %co.e2 = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %co.e2
co.go:
  %co.p1 = add nuw i64 %p0, 1
  %co.p2 = call i64 @js_skip_ws(ptr %buf, i64 %len, i64 %co.p1)
  %co.eof2 = icmp uge i64 %co.p2, %len
  br i1 %co.eof2, label %co.err3, label %co.disp
co.err3:
  %co.e3 = call i32 @js_fail(ptr %sc, i64 %co.p2)
  ret i32 %co.e3
co.disp:
  %co.r = call i32 @js_dispatch(ptr %sc, ptr %tok, i64 %co.p2)
  ret i32 %co.r

L.objnext:
  %on.eof = icmp uge i64 %p0, %len
  br i1 %on.eof, label %on.err, label %on.rd
on.err:
  %on.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %on.e
on.rd:
  %on.p = getelementptr inbounds nuw i8, ptr %buf, i64 %p0
  %on.c = load i8, ptr %on.p, align 1
  %on.close = icmp eq i8 %on.c, 125
  br i1 %on.close, label %close.obj, label %on.comma.chk
on.comma.chk:
  %on.iscomma = icmp eq i8 %on.c, 44
  br i1 %on.iscomma, label %on.comma, label %on.err2
on.err2:
  %on.e2 = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %on.e2
on.comma:
  %on.p1 = add nuw i64 %p0, 1
  %on.p2 = call i64 @js_skip_ws(ptr %buf, i64 %len, i64 %on.p1)
  %on.eof2 = icmp uge i64 %on.p2, %len
  br i1 %on.eof2, label %on.err3, label %on.keychk
on.err3:
  %on.e3 = call i32 @js_fail(ptr %sc, i64 %on.p2)
  ret i32 %on.e3
on.keychk:
  %on.kp = getelementptr inbounds nuw i8, ptr %buf, i64 %on.p2
  %on.kc = load i8, ptr %on.kp, align 1
  %on.quote = icmp eq i8 %on.kc, 34
  br i1 %on.quote, label %on.key, label %on.err4
on.err4:
  %on.e4 = call i32 @js_fail(ptr %sc, i64 %on.p2)
  ret i32 %on.e4
on.key:
  %on.r = call i32 @js_scan_key(ptr %sc, ptr %tok, i64 %on.p2)
  ret i32 %on.r

L.arrnext:
  %an.eof = icmp uge i64 %p0, %len
  br i1 %an.eof, label %an.err, label %an.rd
an.err:
  %an.e = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %an.e
an.rd:
  %an.p = getelementptr inbounds nuw i8, ptr %buf, i64 %p0
  %an.c = load i8, ptr %an.p, align 1
  %an.close = icmp eq i8 %an.c, 93
  br i1 %an.close, label %close.arr, label %an.comma.chk
an.comma.chk:
  %an.iscomma = icmp eq i8 %an.c, 44
  br i1 %an.iscomma, label %an.comma, label %an.err2
an.err2:
  %an.e2 = call i32 @js_fail(ptr %sc, i64 %p0)
  ret i32 %an.e2
an.comma:
  %an.p1 = add nuw i64 %p0, 1
  %an.p2 = call i64 @js_skip_ws(ptr %buf, i64 %len, i64 %an.p1)
  %an.eof2 = icmp uge i64 %an.p2, %len
  br i1 %an.eof2, label %an.err3, label %an.disp
an.err3:
  %an.e3 = call i32 @js_fail(ptr %sc, i64 %an.p2)
  ret i32 %an.e3
an.disp:
  %an.r = call i32 @js_dispatch(ptr %sc, ptr %tok, i64 %an.p2)
  ret i32 %an.r

close.arr:
  %ca.dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  %ca.d = load i32, ptr %ca.dp, align 4
  %ca.d1 = sub i32 %ca.d, 1
  store i32 %ca.d1, ptr %ca.dp, align 4
  %ca.np = add nuw i64 %p0, 1
  %ca.ns = call i32 @js_post_state(ptr %sc)
  %ca.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 3, i64 %p0, i64 1, i64 %ca.np, i32 %ca.ns)
  ret i32 %ca.r

close.obj:
  %cb.dp = getelementptr inbounds nuw i8, ptr %sc, i64 32
  %cb.d = load i32, ptr %cb.dp, align 4
  %cb.d1 = sub i32 %cb.d, 1
  store i32 %cb.d1, ptr %cb.dp, align 4
  %cb.np = add nuw i64 %p0, 1
  %cb.ns = call i32 @js_post_state(ptr %sc)
  %cb.r = call i32 @js_emit(ptr %sc, ptr %tok, i32 1, i64 %p0, i64 1, i64 %cb.np, i32 %cb.ns)
  ret i32 %cb.r
}

; ------------------------------------------------------------------- unescape
; UTF-8 encode a code point into dst[o..]; returns bytes written (1..4).
define internal i64 @js_utf8(ptr %dst, i64 %o, i32 %cp) #1 {
entry:
  %lt80 = icmp ult i32 %cp, 128
  br i1 %lt80, label %one, label %chk2
one:
  %o1p = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  %o1b = trunc i32 %cp to i8
  store i8 %o1b, ptr %o1p, align 1
  ret i64 1
chk2:
  %lt800 = icmp ult i32 %cp, 2048
  br i1 %lt800, label %two, label %chk3
two:
  %t.b0 = lshr i32 %cp, 6
  %t.b0m = or i32 %t.b0, 192
  %t.b0t = trunc i32 %t.b0m to i8
  %t.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %t.b0t, ptr %t.p0, align 1
  %t.b1 = and i32 %cp, 63
  %t.b1m = or i32 %t.b1, 128
  %t.b1t = trunc i32 %t.b1m to i8
  %t.o1 = add nuw i64 %o, 1
  %t.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %t.o1
  store i8 %t.b1t, ptr %t.p1, align 1
  ret i64 2
chk3:
  %lt10000 = icmp ult i32 %cp, 65536
  br i1 %lt10000, label %three, label %four
three:
  %th.b0 = lshr i32 %cp, 12
  %th.b0m = or i32 %th.b0, 224
  %th.b0t = trunc i32 %th.b0m to i8
  %th.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %th.b0t, ptr %th.p0, align 1
  %th.s1 = lshr i32 %cp, 6
  %th.b1 = and i32 %th.s1, 63
  %th.b1m = or i32 %th.b1, 128
  %th.b1t = trunc i32 %th.b1m to i8
  %th.o1 = add nuw i64 %o, 1
  %th.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %th.o1
  store i8 %th.b1t, ptr %th.p1, align 1
  %th.b2 = and i32 %cp, 63
  %th.b2m = or i32 %th.b2, 128
  %th.b2t = trunc i32 %th.b2m to i8
  %th.o2 = add nuw i64 %o, 2
  %th.p2 = getelementptr inbounds nuw i8, ptr %dst, i64 %th.o2
  store i8 %th.b2t, ptr %th.p2, align 1
  ret i64 3
four:
  %fo.b0 = lshr i32 %cp, 18
  %fo.b0m = or i32 %fo.b0, 240
  %fo.b0t = trunc i32 %fo.b0m to i8
  %fo.p0 = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %fo.b0t, ptr %fo.p0, align 1
  %fo.s1 = lshr i32 %cp, 12
  %fo.b1 = and i32 %fo.s1, 63
  %fo.b1m = or i32 %fo.b1, 128
  %fo.b1t = trunc i32 %fo.b1m to i8
  %fo.o1 = add nuw i64 %o, 1
  %fo.p1 = getelementptr inbounds nuw i8, ptr %dst, i64 %fo.o1
  store i8 %fo.b1t, ptr %fo.p1, align 1
  %fo.s2 = lshr i32 %cp, 6
  %fo.b2 = and i32 %fo.s2, 63
  %fo.b2m = or i32 %fo.b2, 128
  %fo.b2t = trunc i32 %fo.b2m to i8
  %fo.o2 = add nuw i64 %o, 2
  %fo.p2 = getelementptr inbounds nuw i8, ptr %dst, i64 %fo.o2
  store i8 %fo.b2t, ptr %fo.p2, align 1
  %fo.b3 = and i32 %cp, 63
  %fo.b3m = or i32 %fo.b3, 128
  %fo.b3t = trunc i32 %fo.b3m to i8
  %fo.o3 = add nuw i64 %o, 3
  %fo.p3 = getelementptr inbounds nuw i8, ptr %dst, i64 %fo.o3
  store i8 %fo.b3t, ptr %fo.p3, align 1
  ret i64 4
}

; Decode a JSON string CONTENT (between quotes) into dst. Returns length or -1.
define i64 @universe_parse_json_unescape(ptr %dst, ptr readonly %src, i64 %len) local_unnamed_addr #3 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %bad = or i1 %dn, %sn
  br i1 %bad, label %argerr, label %loop
argerr:
  ret i64 -1
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next.p, %plain ], [ %i.next.s, %simple ], [ %i.u.next, %u.emit ]
  %o = phi i64 [ 0, %entry ], [ %o.next.p, %plain ], [ %o.next.s, %simple ], [ %o.u.next, %u.emit ]
  %atend = icmp uge i64 %i, %len
  br i1 %atend, label %fin, label %rd
fin:
  ret i64 %o
rd:
  %p = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %c = load i8, ptr %p, align 1
  %isBS = icmp eq i8 %c, 92
  br i1 %isBS, label %esc, label %plain
plain:
  %op = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %c, ptr %op, align 1
  %i.next.p = add nuw i64 %i, 1
  %o.next.p = add nuw i64 %o, 1
  br label %loop
esc:
  %j = add nuw i64 %i, 1
  %jend = icmp uge i64 %j, %len
  br i1 %jend, label %err, label %esc.rd
esc.rd:
  %pj = getelementptr inbounds nuw i8, ptr %src, i64 %j
  %ej = load i8, ptr %pj, align 1
  %ejz = zext i8 %ej to i32
  switch i32 %ejz, label %err [
    i32 34,  label %e.quote
    i32 92,  label %e.bs
    i32 47,  label %e.slash
    i32 98,  label %e.b
    i32 102, label %e.f
    i32 110, label %e.n
    i32 114, label %e.r
    i32 116, label %e.t
    i32 117, label %e.u
  ]
e.quote:
  br label %simple.emit
e.bs:
  br label %simple.emit
e.slash:
  br label %simple.emit
e.b:
  br label %simple.emit
e.f:
  br label %simple.emit
e.n:
  br label %simple.emit
e.r:
  br label %simple.emit
e.t:
  br label %simple.emit
simple.emit:
  %sv = phi i8 [ 34, %e.quote ], [ 92, %e.bs ], [ 47, %e.slash ], [ 8, %e.b ], [ 12, %e.f ], [ 10, %e.n ], [ 13, %e.r ], [ 9, %e.t ]
  br label %simple
simple:
  %so = getelementptr inbounds nuw i8, ptr %dst, i64 %o
  store i8 %sv, ptr %so, align 1
  %i.next.s = add nuw i64 %j, 1
  %o.next.s = add nuw i64 %o, 1
  br label %loop
e.u:
  %h0 = add nuw i64 %j, 1
  %need = add nuw i64 %h0, 4
  %needbad = icmp ugt i64 %need, %len
  br i1 %needbad, label %err, label %u.hex
u.hex:
  %cp1 = call i32 @js_hex4(ptr %src, i64 %h0)
  %cp1.bad = icmp slt i32 %cp1, 0
  br i1 %cp1.bad, label %err, label %u.class
u.class:
  %hi.lo = icmp uge i32 %cp1, 55296
  %hi.hi = icmp ule i32 %cp1, 56319
  %isHigh = and i1 %hi.lo, %hi.hi
  %lo.lo = icmp uge i32 %cp1, 56320
  %lo.hi = icmp ule i32 %cp1, 57343
  %isLow = and i1 %lo.lo, %lo.hi
  br i1 %isLow, label %err, label %u.chkhigh
u.chkhigh:
  br i1 %isHigh, label %u.pair, label %u.single
u.single:
  br label %u.emit
u.pair:
  %q0 = add nuw i64 %h0, 4
  %need2 = add nuw i64 %q0, 6
  %need2bad = icmp ugt i64 %need2, %len
  br i1 %need2bad, label %err, label %u.pair.rd
u.pair.rd:
  %pbs = getelementptr inbounds nuw i8, ptr %src, i64 %q0
  %bs = load i8, ptr %pbs, align 1
  %isbs = icmp eq i8 %bs, 92
  %q1 = add nuw i64 %q0, 1
  %puc = getelementptr inbounds nuw i8, ptr %src, i64 %q1
  %uc = load i8, ptr %puc, align 1
  %isu2 = icmp eq i8 %uc, 117
  %both = and i1 %isbs, %isu2
  br i1 %both, label %u.pair.hex, label %err
u.pair.hex:
  %h2 = add nuw i64 %q0, 2
  %cp2 = call i32 @js_hex4(ptr %src, i64 %h2)
  %cp2.bad = icmp slt i32 %cp2, 0
  br i1 %cp2.bad, label %err, label %u.pair.class
u.pair.class:
  %l2.lo = icmp uge i32 %cp2, 56320
  %l2.hi = icmp ule i32 %cp2, 57343
  %isLow2 = and i1 %l2.lo, %l2.hi
  br i1 %isLow2, label %u.combine, label %err
u.combine:
  %hi.off = sub i32 %cp1, 55296
  %hi.sh = shl i32 %hi.off, 10
  %lo.off = sub i32 %cp2, 56320
  %sum = add i32 %hi.sh, %lo.off
  %cpp = add i32 %sum, 65536
  %q6 = add nuw i64 %q0, 6
  br label %u.emit
u.emit:
  %cpv = phi i32 [ %cp1, %u.single ], [ %cpp, %u.combine ]
  %i.u.next = phi i64 [ %need, %u.single ], [ %q6, %u.combine ]
  %nb = call i64 @js_utf8(ptr %dst, i64 %o, i32 %cpv)
  %o.u.next = add nuw i64 %o, %nb
  br label %loop
err:
  ret i64 -1
}

; ------------------------------------------------------------- number->double
define i32 @universe_parse_json_number_double(ptr readonly %src, i64 %len, ptr %out) local_unnamed_addr #3 {
entry:
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %out, null
  %zl = icmp eq i64 %len, 0
  %bad = or i1 %sn, %on
  %bad2 = or i1 %bad, %zl
  br i1 %bad2, label %perr, label %sign
perr:
  ret i32 13
sign:
  %c0p = getelementptr inbounds nuw i8, ptr %src, i64 0
  %c0 = load i8, ptr %c0p, align 1
  %isMinus = icmp eq i8 %c0, 45
  %i0 = select i1 %isMinus, i64 1, i64 0
  br label %int.head
int.head:
  %ii = phi i64 [ %i0, %sign ], [ %ii.next, %int.body ]
  %m1 = phi double [ 0.000000e+00, %sign ], [ %m1.next, %int.body ]
  %ii.end = icmp uge i64 %ii, %len
  br i1 %ii.end, label %frac.chk, label %int.rd
int.rd:
  %pii = getelementptr inbounds nuw i8, ptr %src, i64 %ii
  %cii = load i8, ptr %pii, align 1
  %ciiz = zext i8 %cii to i32
  %iid = add i32 %ciiz, -48
  %ii.isd = icmp ult i32 %iid, 10
  br i1 %ii.isd, label %int.body, label %frac.chk
int.body:
  %m1.x = fmul double %m1, 1.000000e+01
  %iidf = uitofp i32 %iid to double
  %m1.next = fadd double %m1.x, %iidf
  %ii.next = add nuw i64 %ii, 1
  br label %int.head
frac.chk:
  %fbad = icmp uge i64 %ii, %len
  br i1 %fbad, label %expo.chk, label %frac.rd0
frac.rd0:
  %pdot = getelementptr inbounds nuw i8, ptr %src, i64 %ii
  %cdot = load i8, ptr %pdot, align 1
  %isDot = icmp eq i8 %cdot, 46
  br i1 %isDot, label %frac.head.pre, label %expo.chk
frac.head.pre:
  %fstart = add nuw i64 %ii, 1
  br label %frac.head
frac.head:
  %fi = phi i64 [ %fstart, %frac.head.pre ], [ %fi.next, %frac.body ]
  %m2 = phi double [ %m1, %frac.head.pre ], [ %m2.next, %frac.body ]
  %fdig = phi i64 [ 0, %frac.head.pre ], [ %fdig.next, %frac.body ]
  %fi.end = icmp uge i64 %fi, %len
  br i1 %fi.end, label %expo.chk.f, label %frac.rd
frac.rd:
  %pfi = getelementptr inbounds nuw i8, ptr %src, i64 %fi
  %cfi = load i8, ptr %pfi, align 1
  %cfiz = zext i8 %cfi to i32
  %fid = add i32 %cfiz, -48
  %fi.isd = icmp ult i32 %fid, 10
  br i1 %fi.isd, label %frac.body, label %expo.chk.f
frac.body:
  %m2.x = fmul double %m2, 1.000000e+01
  %fidf = uitofp i32 %fid to double
  %m2.next = fadd double %m2.x, %fidf
  %fdig.next = add nuw i64 %fdig, 1
  %fi.next = add nuw i64 %fi, 1
  br label %frac.head
expo.chk.f:
  br label %expo.merge
expo.chk:
  br label %expo.merge
expo.merge:
  %ei = phi i64 [ %fi, %expo.chk.f ], [ %ii, %expo.chk ]
  %mant = phi double [ %m2, %expo.chk.f ], [ %m1, %expo.chk ]
  %fdigf = phi i64 [ %fdig, %expo.chk.f ], [ 0, %expo.chk ]
  %ei.end = icmp uge i64 %ei, %len
  br i1 %ei.end, label %scale.pre, label %expo.rd
expo.rd:
  %pei = getelementptr inbounds nuw i8, ptr %src, i64 %ei
  %cei = load i8, ptr %pei, align 1
  %ceiz = zext i8 %cei to i32
  %ise = icmp eq i32 %ceiz, 101
  %isE = icmp eq i32 %ceiz, 69
  %ise.any = or i1 %ise, %isE
  br i1 %ise.any, label %expo.sign, label %scale.pre
expo.sign:
  %es0 = add nuw i64 %ei, 1
  %es0.end = icmp uge i64 %es0, %len
  br i1 %es0.end, label %perr, label %expo.sign.rd
expo.sign.rd:
  %pes = getelementptr inbounds nuw i8, ptr %src, i64 %es0
  %ces = load i8, ptr %pes, align 1
  %isPlus = icmp eq i8 %ces, 43
  %isNeg = icmp eq i8 %ces, 45
  %issign = or i1 %isPlus, %isNeg
  %es1 = select i1 %issign, i64 1, i64 0
  %edstart = add nuw i64 %es0, %es1
  %ed.end0 = icmp uge i64 %edstart, %len
  br i1 %ed.end0, label %perr, label %expo.head
expo.head:
  %edi = phi i64 [ %edstart, %expo.sign.rd ], [ %edi.next, %expo.body ]
  %eacc = phi i64 [ 0, %expo.sign.rd ], [ %eacc.next, %expo.body ]
  %edi.end = icmp uge i64 %edi, %len
  br i1 %edi.end, label %expo.fin, label %expo.body.rd
expo.body.rd:
  %pedi = getelementptr inbounds nuw i8, ptr %src, i64 %edi
  %cedi = load i8, ptr %pedi, align 1
  %cediz = zext i8 %cedi to i32
  %edid = add i32 %cediz, -48
  %edi.isd = icmp ult i32 %edid, 10
  br i1 %edi.isd, label %expo.body, label %expo.fin
expo.body:
  %eacc.x = mul i64 %eacc, 10
  %edidx = zext i32 %edid to i64
  %eacc.next = add i64 %eacc.x, %edidx
  %edi.next = add nuw i64 %edi, 1
  br label %expo.head
expo.fin:
  %eaccclamp = call i64 @llvm.umin.i64(i64 %eacc, i64 1000)
  %esigned = sub nsw i64 0, %eaccclamp
  %efinal = select i1 %isNeg, i64 %esigned, i64 %eaccclamp
  br label %scale.merge
scale.pre:
  br label %scale.merge
scale.merge:
  %totexp0 = phi i64 [ %efinal, %expo.fin ], [ 0, %scale.pre ]
  %mant2 = phi double [ %mant, %expo.fin ], [ %mant, %scale.pre ]
  %te = sub nsw i64 %totexp0, %fdigf
  ; clamp te to [-400,400]
  %te.lo = call i64 @llvm.smax.i64(i64 %te, i64 -400)
  %te.c = call i64 @llvm.smin.i64(i64 %te.lo, i64 400)
  %teneg = icmp slt i64 %te.c, 0
  %cnt0 = sub nsw i64 0, %te.c
  %cnt = select i1 %teneg, i64 %cnt0, i64 %te.c
  br i1 %teneg, label %neg.head, label %pos.head
pos.head:
  %pm = phi double [ %mant2, %scale.merge ], [ %pm.next, %pos.body ]
  %pk = phi i64 [ 0, %scale.merge ], [ %pk.next, %pos.body ]
  %pk.done = icmp uge i64 %pk, %cnt
  br i1 %pk.done, label %finish, label %pos.body
pos.body:
  %pm.next = fmul double %pm, 1.000000e+01
  %pk.next = add nuw i64 %pk, 1
  br label %pos.head
neg.head:
  %nm = phi double [ %mant2, %scale.merge ], [ %nm.next, %neg.body ]
  %nk = phi i64 [ 0, %scale.merge ], [ %nk.next, %neg.body ]
  %nk.done = icmp uge i64 %nk, %cnt
  br i1 %nk.done, label %finish, label %neg.body
neg.body:
  %nm.next = fdiv double %nm, 1.000000e+01
  %nk.next = add nuw i64 %nk, 1
  br label %neg.head
finish:
  %res0 = phi double [ %pm, %pos.head ], [ %nm, %neg.head ]
  %neg.res = fneg double %res0
  %final = select i1 %isMinus, double %neg.res, double %res0
  store double %final, ptr %out, align 8
  ret i32 0
}

declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.smax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn norecurse nosync }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
