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

; Dynamic bit set over a packed array of i64 words.
;
; DESIGN (functional spec: a fixed-length dynamic set of bit flags):
;   * ONE allocation: 64-byte header { i64 nbits@0, i64 nwords@8 } and the
;     word payload starting at +64 (the house header+payload@64 layout), so a
;     bitset is a single malloc / single free and the words are one cache-line
;     aligned run the backend can vectorize over.
;   * nwords = ceil(nbits/64). Word k holds bits [k*64, k*64+64). The bits of
;     the last word ABOVE nbits are "tail" bits and are held INVARIANT-ZERO:
;     every mutating op that could dirty them (set_all, complement) re-masks
;     the last word. Because the invariant holds, popcount / find just sweep
;     whole words with no per-word range test — the subtle correctness point
;     is concentrated in two masks, not scattered across the scans.
;   * Bit address math is pure register work: word = i>>6, bit = i&63,
;     mask = 1<<bit (shift amount always < 64, so never poison). The set/clear/
;     toggle/test leaves are branch-free past their cold guards.
;   * Word-parallel union/intersection/difference/complement are plain
;     element-wise word loops over the destination — no early exit, unit
;     stride, no aliasing hazard for distinct operands — so -O3 auto-vectorizes
;     them to NEON/AVX. dst may alias a or b (in-place is well defined).
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW, 7 INVALID_INDEX,
;      8 INVALID_ARG):
;   ptr universe_ds_bitset_create(i64 nbits)          ; nbits==0 allowed
;   void universe_ds_bitset_destroy(ptr bs)
;   i32 universe_ds_bitset_set(ptr bs, i64 i)
;   i32 universe_ds_bitset_clear(ptr bs, i64 i)
;   i32 universe_ds_bitset_toggle(ptr bs, i64 i)
;   i32 universe_ds_bitset_test(ptr bs, i64 i)        ; 0/1 bit, -1 if bad
;   i32 universe_ds_bitset_set_all(ptr bs)
;   i32 universe_ds_bitset_clear_all(ptr bs)
;   i64 universe_ds_bitset_popcount(ptr bs)           ; # of set bits, 0 if null
;   i64 universe_ds_bitset_nbits(ptr bs)
;   i64 universe_ds_bitset_find_first_set(ptr bs)     ; index, -1 if none
;   i64 universe_ds_bitset_find_next_set(ptr bs, i64 from) ; >= from, -1 none
;   i32 universe_ds_bitset_union(ptr dst, ptr a, ptr b)        ; dst = a | b
;   i32 universe_ds_bitset_intersection(ptr dst, ptr a, ptr b) ; dst = a & b
;   i32 universe_ds_bitset_difference(ptr dst, ptr a, ptr b)   ; dst = a & ~b
;   i32 universe_ds_bitset_complement(ptr dst, ptr a)          ; dst = ~a

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.ctpop.i64(i64)
declare i64 @llvm.cttz.i64(i64, i1 immarg)

; ---- construction --------------------------------------------------------

define noalias ptr @universe_ds_bitset_create(i64 %nbits) local_unnamed_addr #1 {
entry:
  ; nwords = (nbits + 63) >> 6 ; guard the +63 add and the *8 byte product.
  %add = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %nbits, i64 63)
  %add.v = extractvalue { i64, i1 } %add, 0
  %add.o = extractvalue { i64, i1 } %add, 1
  br i1 %add.o, label %fail, label %words, !prof !0

words:
  %nwords = lshr i64 %add.v, 6
  %pay = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nwords, i64 8)
  %pay.v = extractvalue { i64, i1 } %pay, 0
  %pay.o = extractvalue { i64, i1 } %pay, 1
  br i1 %pay.o, label %fail, label %total, !prof !0

total:
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pay.v, i64 64)
  %tot.v = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  br i1 %tot.o, label %fail, label %alloc, !prof !0

alloc:
  %bs = call ptr @malloc(i64 %tot.v)
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %fail, label %init, !prof !0

init:
  store i64 %nbits, ptr %bs, align 8
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  store i64 %nwords, ptr %nwords.p, align 8
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  call void @llvm.memset.p0.i64(ptr %wp, i8 0, i64 %pay.v, i1 false)
  ret ptr %bs

fail:
  ret ptr null
}

define void @universe_ds_bitset_destroy(ptr %bs) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %bs, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %bs)
  br label %done

done:
  ret void
}

; ---- single-bit leaves ---------------------------------------------------

define i32 @universe_ds_bitset_set(ptr %bs, i64 %i) local_unnamed_addr #0 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %nbits = load i64, ptr %bs, align 8
  %oob = icmp uge i64 %i, %nbits
  br i1 %oob, label %err.idx, label %do, !prof !0

err.idx:
  ret i32 7

do:
  %word = lshr i64 %i, 6
  %bit = and i64 %i, 63
  %mask = shl nuw i64 1, %bit
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %new = or i64 %cur, %mask
  store i64 %new, ptr %wep, align 8
  ret i32 0
}

define i32 @universe_ds_bitset_clear(ptr %bs, i64 %i) local_unnamed_addr #0 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %nbits = load i64, ptr %bs, align 8
  %oob = icmp uge i64 %i, %nbits
  br i1 %oob, label %err.idx, label %do, !prof !0

err.idx:
  ret i32 7

do:
  %word = lshr i64 %i, 6
  %bit = and i64 %i, 63
  %mask = shl nuw i64 1, %bit
  %notmask = xor i64 %mask, -1
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %new = and i64 %cur, %notmask
  store i64 %new, ptr %wep, align 8
  ret i32 0
}

define i32 @universe_ds_bitset_toggle(ptr %bs, i64 %i) local_unnamed_addr #0 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %nbits = load i64, ptr %bs, align 8
  %oob = icmp uge i64 %i, %nbits
  br i1 %oob, label %err.idx, label %do, !prof !0

err.idx:
  ret i32 7

do:
  %word = lshr i64 %i, 6
  %bit = and i64 %i, 63
  %mask = shl nuw i64 1, %bit
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %new = xor i64 %cur, %mask
  store i64 %new, ptr %wep, align 8
  ret i32 0
}

define i32 @universe_ds_bitset_test(ptr %bs, i64 %i) local_unnamed_addr #2 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err, label %check, !prof !0

err:
  ret i32 -1

check:
  %nbits = load i64, ptr %bs, align 8
  %oob = icmp uge i64 %i, %nbits
  br i1 %oob, label %err, label %do, !prof !0

do:
  %word = lshr i64 %i, 6
  %bit = and i64 %i, 63
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %word
  %cur = load i64, ptr %wep, align 8
  %sh = lshr i64 %cur, %bit
  %b = and i64 %sh, 1
  %r = trunc i64 %b to i32
  ret i32 %r
}

; ---- bulk set / clear ----------------------------------------------------

define i32 @universe_ds_bitset_clear_all(ptr %bs) local_unnamed_addr #0 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %bytes = shl nuw i64 %nwords, 3
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  call void @llvm.memset.p0.i64(ptr %wp, i8 0, i64 %bytes, i1 false)
  ret i32 0
}

define i32 @universe_ds_bitset_set_all(ptr %bs) local_unnamed_addr #0 {
entry:
  %bs.null = icmp eq ptr %bs, null
  br i1 %bs.null, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %nbits = load i64, ptr %bs, align 8
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %bytes = shl nuw i64 %nwords, 3
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  call void @llvm.memset.p0.i64(ptr %wp, i8 -1, i64 %bytes, i1 false)
  ; re-establish the tail-zero invariant on the last word.
  %rem = and i64 %nbits, 63
  %has.tail = icmp ne i64 %rem, 0
  %some = icmp ne i64 %nwords, 0
  %need = and i1 %has.tail, %some
  br i1 %need, label %mask, label %done

mask:
  %tmask = shl nuw i64 1, %rem
  %tmask.m1 = add i64 %tmask, -1
  %last = add i64 %nwords, -1
  %lep = getelementptr inbounds nuw i64, ptr %wp, i64 %last
  %lw = load i64, ptr %lep, align 8
  %lw.m = and i64 %lw, %tmask.m1
  store i64 %lw.m, ptr %lep, align 8
  br label %done

done:
  ret i32 0
}

; ---- queries -------------------------------------------------------------

define i64 @universe_ds_bitset_nbits(ptr %bs) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %bs, null
  br i1 %is.null, label %z, label %do, !prof !0

z:
  ret i64 0

do:
  %nbits = load i64, ptr %bs, align 8
  ret i64 %nbits
}

define i64 @universe_ds_bitset_popcount(ptr %bs) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %bs, null
  br i1 %is.null, label %z, label %pre, !prof !0

z:
  ret i64 0

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %z, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %loop ]
  %acc = phi i64 [ 0, %pre ], [ %acc.n, %loop ]
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %k
  %w = load i64, ptr %wep, align 8
  %pc = call i64 @llvm.ctpop.i64(i64 %w)
  %acc.n = add i64 %acc, %pc
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %done

done:
  ret i64 %acc.n
}

define i64 @universe_ds_bitset_find_first_set(ptr %bs) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %bs, null
  br i1 %is.null, label %none, label %pre, !prof !0

none:
  ret i64 -1

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %none, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %next ]
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %k
  %w = load i64, ptr %wep, align 8
  %nz = icmp ne i64 %w, 0
  br i1 %nz, label %hit, label %next

next:
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %none

hit:
  %tz = call i64 @llvm.cttz.i64(i64 %w, i1 true)
  %base = shl nuw i64 %k, 6
  %idx = add nuw i64 %base, %tz
  ret i64 %idx
}

define i64 @universe_ds_bitset_find_next_set(ptr %bs, i64 %from) local_unnamed_addr #3 {
entry:
  %is.null = icmp eq ptr %bs, null
  br i1 %is.null, label %none, label %pre, !prof !0

none:
  ret i64 -1

pre:
  %nbits = load i64, ptr %bs, align 8
  %past = icmp uge i64 %from, %nbits
  br i1 %past, label %none, label %setup, !prof !0

setup:
  %nwords.p = getelementptr inbounds nuw i8, ptr %bs, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %wp = getelementptr inbounds nuw i8, ptr %bs, i64 64
  %word0 = lshr i64 %from, 6
  %bitoff = and i64 %from, 63
  ; mask off the bits below bitoff in the first probed word.
  %lowmask = shl i64 -1, %bitoff
  %w0ep = getelementptr inbounds nuw i64, ptr %wp, i64 %word0
  %w0 = load i64, ptr %w0ep, align 8
  %w0m = and i64 %w0, %lowmask
  %nz0 = icmp ne i64 %w0m, 0
  br i1 %nz0, label %hit0, label %scan

hit0:
  %tz0 = call i64 @llvm.cttz.i64(i64 %w0m, i1 true)
  %base0 = shl nuw i64 %word0, 6
  %idx0 = add nuw i64 %base0, %tz0
  ret i64 %idx0

scan:
  %k0 = add nuw i64 %word0, 1
  %more0 = icmp ult i64 %k0, %nwords
  br i1 %more0, label %loop, label %none

loop:
  %k = phi i64 [ %k0, %scan ], [ %k.n, %next ]
  %wep = getelementptr inbounds nuw i64, ptr %wp, i64 %k
  %w = load i64, ptr %wep, align 8
  %nz = icmp ne i64 %w, 0
  br i1 %nz, label %hit, label %next

next:
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %none

hit:
  %tz = call i64 @llvm.cttz.i64(i64 %w, i1 true)
  %base = shl nuw i64 %k, 6
  %idx = add nuw i64 %base, %tz
  ret i64 %idx
}

; ---- word-parallel set algebra ------------------------------------------

define i32 @universe_ds_bitset_union(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %d.null = icmp eq ptr %dst, null
  %a.null = icmp eq ptr %a, null
  %b.null = icmp eq ptr %b, null
  %n1 = or i1 %d.null, %a.null
  %anynull = or i1 %n1, %b.null
  br i1 %anynull, label %err.null, label %sizes, !prof !0

err.null:
  ret i32 1

sizes:
  %dn = load i64, ptr %dst, align 8
  %an = load i64, ptr %a, align 8
  %bn = load i64, ptr %b, align 8
  %e1 = icmp eq i64 %dn, %an
  %e2 = icmp eq i64 %dn, %bn
  %ok = and i1 %e1, %e2
  br i1 %ok, label %pre, label %err.arg, !prof !1

err.arg:
  ret i32 8

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %dwp = getelementptr inbounds nuw i8, ptr %dst, i64 64
  %awp = getelementptr inbounds nuw i8, ptr %a, i64 64
  %bwp = getelementptr inbounds nuw i8, ptr %b, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %done, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %awp, i64 %k
  %bp = getelementptr inbounds nuw i64, ptr %bwp, i64 %k
  %dp = getelementptr inbounds nuw i64, ptr %dwp, i64 %k
  %av = load i64, ptr %ap, align 8
  %bv = load i64, ptr %bp, align 8
  %rv = or i64 %av, %bv
  store i64 %rv, ptr %dp, align 8
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %done

done:
  ret i32 0
}

define i32 @universe_ds_bitset_intersection(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %d.null = icmp eq ptr %dst, null
  %a.null = icmp eq ptr %a, null
  %b.null = icmp eq ptr %b, null
  %n1 = or i1 %d.null, %a.null
  %anynull = or i1 %n1, %b.null
  br i1 %anynull, label %err.null, label %sizes, !prof !0

err.null:
  ret i32 1

sizes:
  %dn = load i64, ptr %dst, align 8
  %an = load i64, ptr %a, align 8
  %bn = load i64, ptr %b, align 8
  %e1 = icmp eq i64 %dn, %an
  %e2 = icmp eq i64 %dn, %bn
  %ok = and i1 %e1, %e2
  br i1 %ok, label %pre, label %err.arg, !prof !1

err.arg:
  ret i32 8

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %dwp = getelementptr inbounds nuw i8, ptr %dst, i64 64
  %awp = getelementptr inbounds nuw i8, ptr %a, i64 64
  %bwp = getelementptr inbounds nuw i8, ptr %b, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %done, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %awp, i64 %k
  %bp = getelementptr inbounds nuw i64, ptr %bwp, i64 %k
  %dp = getelementptr inbounds nuw i64, ptr %dwp, i64 %k
  %av = load i64, ptr %ap, align 8
  %bv = load i64, ptr %bp, align 8
  %rv = and i64 %av, %bv
  store i64 %rv, ptr %dp, align 8
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %done

done:
  ret i32 0
}

define i32 @universe_ds_bitset_difference(ptr %dst, ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %d.null = icmp eq ptr %dst, null
  %a.null = icmp eq ptr %a, null
  %b.null = icmp eq ptr %b, null
  %n1 = or i1 %d.null, %a.null
  %anynull = or i1 %n1, %b.null
  br i1 %anynull, label %err.null, label %sizes, !prof !0

err.null:
  ret i32 1

sizes:
  %dn = load i64, ptr %dst, align 8
  %an = load i64, ptr %a, align 8
  %bn = load i64, ptr %b, align 8
  %e1 = icmp eq i64 %dn, %an
  %e2 = icmp eq i64 %dn, %bn
  %ok = and i1 %e1, %e2
  br i1 %ok, label %pre, label %err.arg, !prof !1

err.arg:
  ret i32 8

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %dwp = getelementptr inbounds nuw i8, ptr %dst, i64 64
  %awp = getelementptr inbounds nuw i8, ptr %a, i64 64
  %bwp = getelementptr inbounds nuw i8, ptr %b, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %done, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %awp, i64 %k
  %bp = getelementptr inbounds nuw i64, ptr %bwp, i64 %k
  %dp = getelementptr inbounds nuw i64, ptr %dwp, i64 %k
  %av = load i64, ptr %ap, align 8
  %bv = load i64, ptr %bp, align 8
  %nb = xor i64 %bv, -1
  %rv = and i64 %av, %nb
  store i64 %rv, ptr %dp, align 8
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %done

done:
  ret i32 0
}

define i32 @universe_ds_bitset_complement(ptr %dst, ptr %a) local_unnamed_addr #0 {
entry:
  %d.null = icmp eq ptr %dst, null
  %a.null = icmp eq ptr %a, null
  %anynull = or i1 %d.null, %a.null
  br i1 %anynull, label %err.null, label %sizes, !prof !0

err.null:
  ret i32 1

sizes:
  %dn = load i64, ptr %dst, align 8
  %an = load i64, ptr %a, align 8
  %ok = icmp eq i64 %dn, %an
  br i1 %ok, label %pre, label %err.arg, !prof !1

err.arg:
  ret i32 8

pre:
  %nwords.p = getelementptr inbounds nuw i8, ptr %a, i64 8
  %nwords = load i64, ptr %nwords.p, align 8
  %dwp = getelementptr inbounds nuw i8, ptr %dst, i64 64
  %awp = getelementptr inbounds nuw i8, ptr %a, i64 64
  %empty = icmp eq i64 %nwords, 0
  br i1 %empty, label %done, label %loop

loop:
  %k = phi i64 [ 0, %pre ], [ %k.n, %loop ]
  %ap = getelementptr inbounds nuw i64, ptr %awp, i64 %k
  %dp = getelementptr inbounds nuw i64, ptr %dwp, i64 %k
  %av = load i64, ptr %ap, align 8
  %rv = xor i64 %av, -1
  store i64 %rv, ptr %dp, align 8
  %k.n = add nuw i64 %k, 1
  %more = icmp ult i64 %k.n, %nwords
  br i1 %more, label %loop, label %tail

tail:
  ; complement dirties the tail bits of the last word: re-mask them to 0.
  %rem = and i64 %dn, 63
  %has.tail = icmp ne i64 %rem, 0
  br i1 %has.tail, label %mask, label %done

mask:
  %tmask = shl nuw i64 1, %rem
  %tmask.m1 = add i64 %tmask, -1
  %last = add i64 %nwords, -1
  %lep = getelementptr inbounds nuw i64, ptr %dwp, i64 %last
  %lw = load i64, ptr %lep, align 8
  %lw.m = and i64 %lw, %tmask.m1
  store i64 %lw.m, ptr %lep, align 8
  br label %done

done:
  ret i32 0
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
