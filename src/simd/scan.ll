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

; universe_simd_* — reusable SIMD scan/compare/transform primitives over a
; caller-supplied byte range. Pure compute over memory it already owns; never
; allocates. These are the shared building blocks the HTTP parser and existing
; encoders/parsers call for memchr/memcmp/case-fold/ascii-validate work.
;
; DESIGN — SIMD-first with scalar fallback (the canonical embodiment):
;   * PRIMARY PATH is a portable 128-bit <16 x i8> vector loop. It lowers to
;     SSE2 on AMD64 and NEON on AArch64 — both baseline everywhere we ship — so
;     it needs NO runtime CPU check. The per-lane work is branch-free: a vector
;     `icmp` builds a 16-lane mask, then either
;       - `bitcast <16 x i1> -> i16` (movemask: x86 pmovmskb / ARM lowered) with
;         `llvm.cttz.i16` to LOCATE the first matching lane, or
;       - `sext`+`llvm.vector.reduce.add.v16i8` (x86 psadbw / ARM addv) to COUNT.
;     The only branch in the vector body is the once-per-16-bytes "this chunk
;     hit / another full chunk fits" test — never a per-byte branch.
;   * SCALAR PATH is the fallback AND the oracle. Every op ships a
;     `*_scalar` twin used to (a) process the sub-16 remainder (tail) and
;     (b) cross-check the vector result on fixed-seed random buffers in tests.
;     No op is "done" until vector == scalar for every length.
;   * VECTOR/SCALAR CONTRACT (what callers rely on): the vector entry and its
;     `_scalar` twin return BIT-IDENTICAL results for every input. find_* return
;     the first matching index or -1. compare returns the signed difference of
;     the first differing unsigned byte (memcmp sign). count returns the exact
;     match count. to_lower/to_upper allow dst==src (in-place); they touch only
;     ASCII A-Z / a-z, leaving every other byte (incl. UTF-8 continuation
;     bytes >= 0x80) untouched. is_ascii / validate_ascii test the high bit.
;   * Wider vectors (AVX2/AVX-512/SVE) are runtime-dispatched ADD-ONS for a
;     later wave; this module is the 128-bit baseline and stays branch-free of
;     any -march assumption.
;
; API:
;   i64  universe_simd_find_byte(ptr, i64 n, i8 c)              ; memchr, -1 miss
;   i64  universe_simd_find_byte_scalar(ptr, i64 n, i8 c)
;   i64  universe_simd_find_crlf(ptr, i64 n)                    ; first "\r\n", -1
;   i64  universe_simd_find_crlf_scalar(ptr, i64 n)
;   i64  universe_simd_index_of_any(ptr, i64 n, ptr set, i64 m) ; first in set,-1
;   i64  universe_simd_index_of_any_scalar(ptr, i64 n, ptr set, i64 m)
;   i64  universe_simd_count_byte(ptr, i64 n, i8 c)            ; # of matches
;   i64  universe_simd_count_byte_scalar(ptr, i64 n, i8 c)
;   i1   universe_simd_equal(ptr a, ptr b, i64 n)
;   i1   universe_simd_equal_scalar(ptr a, ptr b, i64 n)
;   i32  universe_simd_compare(ptr a, ptr b, i64 n)           ; memcmp sign
;   i32  universe_simd_compare_scalar(ptr a, ptr b, i64 n)
;   void universe_simd_to_lower_ascii(ptr dst, ptr src, i64 n)
;   void universe_simd_to_lower_ascii_scalar(ptr dst, ptr src, i64 n)
;   void universe_simd_to_upper_ascii(ptr dst, ptr src, i64 n)
;   void universe_simd_to_upper_ascii_scalar(ptr dst, ptr src, i64 n)
;   i1   universe_simd_is_ascii(ptr, i64 n)
;   i1   universe_simd_is_ascii_scalar(ptr, i64 n)
;   i64  universe_simd_validate_ascii(ptr, i64 n)             ; idx of 1st non-ascii, -1 all ok
;   i64  universe_simd_validate_ascii_scalar(ptr, i64 n)

declare i16 @llvm.cttz.i16(i16, i1)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)

; ================================================================= find_byte
define i64 @universe_simd_find_byte(ptr readonly %p, i64 %n, i8 %c) local_unnamed_addr #0 {
entry:
  %ins = insertelement <16 x i8> poison, i8 %c, i64 0
  %spl = shufflevector <16 x i8> %ins, <16 x i8> poison, <16 x i32> zeroinitializer
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %cmp = icmp eq <16 x i8> %v, %spl
  %mm = bitcast <16 x i1> %cmp to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont

locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %notfound, label %tail.body

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  %teq = icmp eq i8 %tb, %c
  br i1 %teq, label %tail.found, label %tail.cont

tail.found:
  ret i64 %ti

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

notfound:
  ret i64 -1
}

define i64 @universe_simd_find_byte_scalar(ptr readonly %p, i64 %n, i8 %c) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %nf, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %eq = icmp eq i8 %b, %c
  br i1 %eq, label %found, label %cont

found:
  ret i64 %i

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %nf

nf:
  ret i64 -1
}

; ================================================================= find_crlf
; The vector loop needs BOTH v1 = p[i..i+15] and v2 = p[i+1..i+16], so it only
; runs while i+16 <= n-1, i.e. n >= 17; the scalar tail checks the remaining
; pairs. cr-lane & lf-shifted-lane AND'd, then movemask+cttz locates the pair.
define i64 @universe_simd_find_crlf(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  %has17 = icmp uge i64 %n, 17
  br i1 %has17, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v1 = load <16 x i8>, ptr %vp, align 1
  %i1 = add nuw i64 %i, 1
  %vp2 = getelementptr inbounds nuw i8, ptr %p, i64 %i1
  %v2 = load <16 x i8>, ptr %vp2, align 1
  %crm = icmp eq <16 x i8> %v1, splat (i8 13)
  %lfm = icmp eq <16 x i8> %v2, splat (i8 10)
  %both = and <16 x i1> %crm, %lfm
  %mm = bitcast <16 x i1> %both to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont

locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 17
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %ti1 = add nuw i64 %ti, 1
  %inrange = icmp ult i64 %ti1, %n
  br i1 %inrange, label %tail.body, label %notfound

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  %iscr = icmp eq i8 %tb, 13
  br i1 %iscr, label %chklf, label %tail.cont

chklf:
  %tp1 = getelementptr inbounds nuw i8, ptr %p, i64 %ti1
  %tb1 = load i8, ptr %tp1, align 1
  %islf = icmp eq i8 %tb1, 10
  br i1 %islf, label %tail.found, label %tail.cont

tail.found:
  ret i64 %ti

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

notfound:
  ret i64 -1
}

define i64 @universe_simd_find_crlf_scalar(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  br label %head

head:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %i1 = add nuw i64 %i, 1
  %inrange = icmp ult i64 %i1, %n
  br i1 %inrange, label %body, label %nf

body:
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %iscr = icmp eq i8 %b, 13
  br i1 %iscr, label %chk, label %cont

chk:
  %pp1 = getelementptr inbounds nuw i8, ptr %p, i64 %i1
  %b1 = load i8, ptr %pp1, align 1
  %islf = icmp eq i8 %b1, 10
  br i1 %islf, label %found, label %cont

found:
  ret i64 %i

cont:
  %i.next = add nuw i64 %i, 1
  br label %head

nf:
  ret i64 -1
}

; ============================================================== index_of_any
define i64 @universe_simd_index_of_any(ptr readonly %p, i64 %n, ptr readonly %set, i64 %setlen) local_unnamed_addr #0 {
entry:
  %emptyset = icmp eq i64 %setlen, 0
  br i1 %emptyset, label %notfound, label %start

start:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %start ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  br label %set.head

set.head:
  %si = phi i64 [ 0, %vloop ], [ %si.next, %set.body ]
  %acc = phi i16 [ 0, %vloop ], [ %acc.next, %set.body ]
  %sdone = icmp uge i64 %si, %setlen
  br i1 %sdone, label %set.done, label %set.body

set.body:
  %scp = getelementptr inbounds nuw i8, ptr %set, i64 %si
  %sc = load i8, ptr %scp, align 1
  %sins = insertelement <16 x i8> poison, i8 %sc, i64 0
  %sspl = shufflevector <16 x i8> %sins, <16 x i8> poison, <16 x i32> zeroinitializer
  %seq = icmp eq <16 x i8> %v, %sspl
  %seqm = bitcast <16 x i1> %seq to i16
  %acc.next = or i16 %acc, %seqm
  %si.next = add nuw i64 %si, 1
  br label %set.head

set.done:
  %hit = icmp ne i16 %acc, 0
  br i1 %hit, label %locate, label %vcont

locate:
  %tz = call i16 @llvm.cttz.i16(i16 %acc, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %start ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %notfound, label %tail.body

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  br label %mem.head

mem.head:
  %mi = phi i64 [ 0, %tail.body ], [ %mi.next, %mem.cont ]
  %mdone = icmp uge i64 %mi, %setlen
  br i1 %mdone, label %tail.cont, label %mem.body

mem.body:
  %mcp = getelementptr inbounds nuw i8, ptr %set, i64 %mi
  %mc = load i8, ptr %mcp, align 1
  %meq = icmp eq i8 %mc, %tb
  br i1 %meq, label %tail.found, label %mem.cont

mem.cont:
  %mi.next = add nuw i64 %mi, 1
  br label %mem.head

tail.found:
  ret i64 %ti

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

notfound:
  ret i64 -1
}

define i64 @universe_simd_index_of_any_scalar(ptr readonly %p, i64 %n, ptr readonly %set, i64 %setlen) local_unnamed_addr #0 {
entry:
  %emptyset = icmp eq i64 %setlen, 0
  %noscan = icmp eq i64 %n, 0
  %bail = or i1 %emptyset, %noscan
  br i1 %bail, label %nf, label %head

head:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  br label %mem.head

mem.head:
  %mi = phi i64 [ 0, %head ], [ %mi.next, %mem.cont ]
  %mdone = icmp uge i64 %mi, %setlen
  br i1 %mdone, label %cont, label %mem.body

mem.body:
  %mcp = getelementptr inbounds nuw i8, ptr %set, i64 %mi
  %mc = load i8, ptr %mcp, align 1
  %meq = icmp eq i8 %mc, %b
  br i1 %meq, label %found, label %mem.cont

mem.cont:
  %mi.next = add nuw i64 %mi, 1
  br label %mem.head

found:
  ret i64 %i

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %head, label %nf

nf:
  ret i64 -1
}

; ================================================================ count_byte
define i64 @universe_simd_count_byte(ptr readonly %p, i64 %n, i8 %c) local_unnamed_addr #0 {
entry:
  %ins = insertelement <16 x i8> poison, i8 %c, i64 0
  %spl = shufflevector <16 x i8> %ins, <16 x i8> poison, <16 x i32> zeroinitializer
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vloop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %vloop ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %cmp = icmp eq <16 x i8> %v, %spl
  %m8 = sext <16 x i1> %cmp to <16 x i8>
  %red = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %m8)
  %cnt8 = sub i8 0, %red
  %cnt = zext i8 %cnt8 to i64
  %acc.next = add nuw i64 %acc, %cnt
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vloop ], [ %ti.next, %tail.body ]
  %tacc = phi i64 [ 0, %entry ], [ %acc.next, %vloop ], [ %tacc.next, %tail.body ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %done, label %tail.body

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  %teq = icmp eq i8 %tb, %c
  %tinc = zext i1 %teq to i64
  %tacc.next = add nuw i64 %tacc, %tinc
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

done:
  ret i64 %tacc
}

define i64 @universe_simd_count_byte_scalar(ptr readonly %p, i64 %n, i8 %c) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %eq = icmp eq i8 %b, %c
  %inc = zext i1 %eq to i64
  %acc.next = add nuw i64 %acc, %inc
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  ret i64 %r
}

; ===================================================================== equal
define i1 @universe_simd_equal(ptr readonly %a, ptr readonly %b, i64 %n) local_unnamed_addr #0 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load <16 x i8>, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load <16 x i8>, ptr %bp, align 1
  %ne = icmp ne <16 x i8> %av, %bv
  %mm = bitcast <16 x i1> %ne to i16
  %diff = icmp ne i16 %mm, 0
  br i1 %diff, label %nequal, label %vcont

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %equal, label %tail.body

tail.body:
  %tap = getelementptr inbounds nuw i8, ptr %a, i64 %ti
  %ta = load i8, ptr %tap, align 1
  %tbp = getelementptr inbounds nuw i8, ptr %b, i64 %ti
  %tb = load i8, ptr %tbp, align 1
  %tne = icmp ne i8 %ta, %tb
  br i1 %tne, label %nequal, label %tail.cont

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

equal:
  ret i1 true

nequal:
  ret i1 false
}

define i1 @universe_simd_equal_scalar(ptr readonly %a, ptr readonly %b, i64 %n) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %equal, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %nequal, label %cont

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %equal

equal:
  ret i1 true

nequal:
  ret i1 false
}

; =================================================================== compare
define i32 @universe_simd_compare(ptr readonly %a, ptr readonly %b, i64 %n) local_unnamed_addr #0 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load <16 x i8>, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load <16 x i8>, ptr %bp, align 1
  %ne = icmp ne <16 x i8> %av, %bv
  %mm = bitcast <16 x i1> %ne to i16
  %diff = icmp ne i16 %mm, 0
  br i1 %diff, label %locate, label %vcont

locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  %lap = getelementptr inbounds nuw i8, ptr %a, i64 %idx
  %lab = load i8, ptr %lap, align 1
  %lau = zext i8 %lab to i32
  %lbp = getelementptr inbounds nuw i8, ptr %b, i64 %idx
  %lbb = load i8, ptr %lbp, align 1
  %lbu = zext i8 %lbb to i32
  %ld = sub nsw i32 %lau, %lbu
  ret i32 %ld

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %equal, label %tail.body

tail.body:
  %tap = getelementptr inbounds nuw i8, ptr %a, i64 %ti
  %tab = load i8, ptr %tap, align 1
  %tau = zext i8 %tab to i32
  %tbp = getelementptr inbounds nuw i8, ptr %b, i64 %ti
  %tbb = load i8, ptr %tbp, align 1
  %tbu = zext i8 %tbb to i32
  %tne = icmp ne i32 %tau, %tbu
  br i1 %tne, label %tdiff, label %tail.cont

tdiff:
  %td = sub nsw i32 %tau, %tbu
  ret i32 %td

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

equal:
  ret i32 0
}

define i32 @universe_simd_compare_scalar(ptr readonly %a, ptr readonly %b, i64 %n) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %equal, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %ab = load i8, ptr %ap, align 1
  %au = zext i8 %ab to i32
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bb = load i8, ptr %bp, align 1
  %bu = zext i8 %bb to i32
  %ne = icmp ne i32 %au, %bu
  br i1 %ne, label %diff, label %cont

diff:
  %d = sub nsw i32 %au, %bu
  ret i32 %d

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %equal

equal:
  ret i32 0
}

; ============================================================== to_lower_ascii
define void @universe_simd_to_lower_ascii(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vloop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %v = load <16 x i8>, ptr %sp, align 1
  %sub = sub <16 x i8> %v, splat (i8 65)
  %isupper = icmp ult <16 x i8> %sub, splat (i8 26)
  %delta = select <16 x i1> %isupper, <16 x i8> splat (i8 32), <16 x i8> zeroinitializer
  %res = add <16 x i8> %v, %delta
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store <16 x i8> %res, ptr %dp, align 1
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vloop ], [ %ti.next, %tail.body ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %done, label %tail.body

tail.body:
  %tsp = getelementptr inbounds nuw i8, ptr %src, i64 %ti
  %tb = load i8, ptr %tsp, align 1
  %tsub = sub i8 %tb, 65
  %tisu = icmp ult i8 %tsub, 26
  %tdelta = select i1 %tisu, i8 32, i8 0
  %tres = add i8 %tb, %tdelta
  %tdp = getelementptr inbounds nuw i8, ptr %dst, i64 %ti
  store i8 %tres, ptr %tdp, align 1
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

done:
  ret void
}

define void @universe_simd_to_lower_ascii_scalar(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b = load i8, ptr %sp, align 1
  %sub = sub i8 %b, 65
  %isu = icmp ult i8 %sub, 26
  %delta = select i1 %isu, i8 32, i8 0
  %res = add i8 %b, %delta
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %res, ptr %dp, align 1
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ============================================================== to_upper_ascii
define void @universe_simd_to_upper_ascii(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vloop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %v = load <16 x i8>, ptr %sp, align 1
  %sub = sub <16 x i8> %v, splat (i8 97)
  %islower = icmp ult <16 x i8> %sub, splat (i8 26)
  %delta = select <16 x i1> %islower, <16 x i8> splat (i8 32), <16 x i8> zeroinitializer
  %res = sub <16 x i8> %v, %delta
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store <16 x i8> %res, ptr %dp, align 1
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vloop ], [ %ti.next, %tail.body ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %done, label %tail.body

tail.body:
  %tsp = getelementptr inbounds nuw i8, ptr %src, i64 %ti
  %tb = load i8, ptr %tsp, align 1
  %tsub = sub i8 %tb, 97
  %tisl = icmp ult i8 %tsub, 26
  %tdelta = select i1 %tisl, i8 32, i8 0
  %tres = sub i8 %tb, %tdelta
  %tdp = getelementptr inbounds nuw i8, ptr %dst, i64 %ti
  store i8 %tres, ptr %tdp, align 1
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

done:
  ret void
}

define void @universe_simd_to_upper_ascii_scalar(ptr %dst, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %b = load i8, ptr %sp, align 1
  %sub = sub i8 %b, 97
  %isl = icmp ult i8 %sub, 26
  %delta = select i1 %isl, i8 32, i8 0
  %res = sub i8 %b, %delta
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %res, ptr %dp, align 1
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %done

done:
  ret void
}

; ================================================================== is_ascii
define i1 @universe_simd_is_ascii(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vloop ]
  %acc = phi i16 [ 0, %entry ], [ %acc.next, %vloop ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %neg = icmp slt <16 x i8> %v, zeroinitializer
  %mm = bitcast <16 x i1> %neg to i16
  %acc.next = or i16 %acc, %mm
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vloop ], [ %ti.next, %tail.body ]
  %tacc = phi i16 [ 0, %entry ], [ %acc.next, %vloop ], [ %tacc.next, %tail.body ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %check, label %tail.body

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  %thi = icmp slt i8 %tb, 0
  %thiz = zext i1 %thi to i16
  %tacc.next = or i16 %tacc, %thiz
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

check:
  %bad = icmp ne i16 %tacc, 0
  %ok = xor i1 %bad, true
  ret i1 %ok
}

define i1 @universe_simd_is_ascii_scalar(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %ok, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %hi = icmp slt i8 %b, 0
  br i1 %hi, label %bad, label %cont

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %ok

ok:
  ret i1 true

bad:
  ret i1 false
}

; ============================================================ validate_ascii
; Locate the first non-ASCII (high-bit-set) byte; -1 if the range is all ASCII.
define i64 @universe_simd_validate_ascii(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %tail.head

vloop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vcont ]
  %vp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %v = load <16 x i8>, ptr %vp, align 1
  %neg = icmp slt <16 x i8> %v, zeroinitializer
  %mm = bitcast <16 x i1> %neg to i16
  %hit = icmp ne i16 %mm, 0
  br i1 %hit, label %locate, label %vcont

locate:
  %tz = call i16 @llvm.cttz.i16(i16 %mm, i1 true)
  %tz64 = zext i16 %tz to i64
  %idx = add nuw i64 %i, %tz64
  ret i64 %idx

vcont:
  %i.next = add nuw i64 %i, 16
  %limit = sub nuw i64 %n, 16
  %more = icmp ule i64 %i.next, %limit
  br i1 %more, label %vloop, label %tail.head

tail.head:
  %ti = phi i64 [ 0, %entry ], [ %i.next, %vcont ], [ %ti.next, %tail.cont ]
  %tdone = icmp uge i64 %ti, %n
  br i1 %tdone, label %notfound, label %tail.body

tail.body:
  %tp = getelementptr inbounds nuw i8, ptr %p, i64 %ti
  %tb = load i8, ptr %tp, align 1
  %thi = icmp slt i8 %tb, 0
  br i1 %thi, label %tail.found, label %tail.cont

tail.found:
  ret i64 %ti

tail.cont:
  %ti.next = add nuw i64 %ti, 1
  br label %tail.head

notfound:
  ret i64 -1
}

define i64 @universe_simd_validate_ascii_scalar(ptr readonly %p, i64 %n) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %nf, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %cont ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %hi = icmp slt i8 %b, 0
  br i1 %hi, label %found, label %cont

found:
  ret i64 %i

cont:
  %i.next = add nuw i64 %i, 1
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %loop, label %nf

nf:
  ret i64 -1
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
