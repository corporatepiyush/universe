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

; Probabilistic membership filters: cuckoo, XOR, and blocked (cache-line) Bloom.
;
; A membership filter answers "is key x in the set?" with NO false negatives
; (a key that was inserted always tests present) and a bounded false-positive
; rate, using far less space than an exact set. All three variants here share
; one fast integer mixer (splitmix64 finalizer) and derive every hash position
; by slicing bits out of one 64-bit mix.
;
; DESIGN (three variants, chosen by workload):
;   * CUCKOO (dynamic; add/contains/DELETE): buckets of 4 x 8-bit fingerprints.
;     A key has two candidate buckets via PARTIAL-KEY cuckoo hashing:
;       i1 = (mix(key) >> 32) & mask ; i2 = (i1 ^ (mix(fp) & mask)) & mask
;     so alt(alt(b)) == b for either bucket without storing the key. add finds
;     an empty slot in i1 or i2, else evicts a random victim and relocates it
;     (bounded MAX_KICKS=500 -> FULL). contains is SWAR: one i32 load per bucket
;     + a haszero test (no per-slot branch). delete removes ONE matching
;     fingerprint. Layout: header{ i64 nbuckets@0, i64 mask@8, i64 count@16,
;     i64 rng@24 }, 4-byte buckets at +64, one calloc. Power-of-two nbuckets.
;       NOTE: on FULL the last evicted victim is dropped (classic cuckoo) — a
;       previously-present key can then test absent; stop inserting on FULL.
;       delete removes one fingerprint; a distinct key sharing that
;       fingerprint+bucket can then test absent. Delete only keys known present.
;   * XOR (static; build-once/contains): built from a set of DISTINCT keys.
;     b[] holds 8-bit fingerprints across 3 equal segments of blockLength each
;     (arrayLength = 3*blockLength ~= 1.23 bytes/key). Each key maps to one
;     slot per segment (h0,h1,h2) and its fingerprint is stored so that
;     fp(x) == b[h0]^b[h1]^b[h2]. Construction PEELS the hypergraph (repeatedly
;     remove a slot touched by exactly one key), then assigns fingerprints in
;     reverse peel order; a mapping cycle triggers a reseeded retry. contains is
;     3 byte loads + xor + compare. Layout: header{ i64 seed@0,
;     i64 blockLength@8, i64 arrayLength@16, i64 size@24 }, b[] at +64.
;   * BLOCKED BLOOM (dynamic; add/contains): the filter is an array of 64-byte
;     (512-bit) blocks. ONE key -> ONE block (high mix bits & mask); k=8 bits
;     are set within that block via double hashing (a + i*b, b odd). Every query
;     therefore touches EXACTLY ONE cache line. Layout: header{ i64 nblocks@0,
;     i64 mask@8 }, 64-byte blocks at +64, one calloc. Power-of-two nblocks.
;
; API (0 OK, 1 NULL_PTR, 2 OUT_OF_MEMORY, 3 SIZE_OVERFLOW, 5 NOT_FOUND,
;      6 FULL, 8 INVALID_ARG):
;   ; -- cuckoo --
;   ptr universe_ds_cuckoo_create(i64 capacity_hint)   ; expected #keys
;   void universe_ds_cuckoo_destroy(ptr f)
;   i32 universe_ds_cuckoo_add(ptr f, i64 key)         ; 0 OK, 6 FULL
;   i32 universe_ds_cuckoo_contains(ptr f, i64 key)    ; 0/1, 0 if null
;   i32 universe_ds_cuckoo_delete(ptr f, i64 key)      ; 0 OK, 5 NOT_FOUND
;   i64 universe_ds_cuckoo_count(ptr f)                ; live fingerprints
;   i64 universe_ds_cuckoo_capacity(ptr f)             ; 4*nbuckets slots
;   ; -- xor --
;   ptr universe_ds_xor_build(ptr keys_u64, i64 n)     ; null on OOM/fail
;   void universe_ds_xor_destroy(ptr f)
;   i32 universe_ds_xor_contains(ptr f, i64 key)       ; 0/1, 0 if null
;   ; -- blocked bloom --
;   ptr universe_ds_bbloom_create(i64 nkeys, i64 bits_per_key)
;   void universe_ds_bbloom_destroy(ptr f)
;   i32 universe_ds_bbloom_add(ptr f, i64 key)         ; 0 OK
;   i32 universe_ds_bbloom_contains(ptr f, i64 key)    ; 0/1, 0 if null

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @calloc(i64, i64) allockind("alloc,zeroed") allocsize(0,1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ======================================================================
;  shared pure helpers (value-in / value-out; inline to zero cost)
; ======================================================================

; splitmix64 finalizer with golden-gamma bias so mix(0) != 0.
define internal i64 @filt_mix(i64 %x) #3 {
entry:
  %z = add i64 %x, 11400714819323198485
  %s1 = lshr i64 %z, 30
  %x1 = xor i64 %z, %s1
  %m1 = mul i64 %x1, 13787848793156543929
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %m1, %s2
  %m2 = mul i64 %x2, 10723151780598845931
  %s3 = lshr i64 %m2, 31
  %r = xor i64 %m2, %s3
  ret i64 %r
}

; smallest power of two >= n (>= 1). n assumed < 2^63.
define internal i64 @filt_pow2ceil(i64 %n) #3 {
entry:
  %le1 = icmp ule i64 %n, 1
  br i1 %le1, label %one, label %calc

one:
  ret i64 1

calc:
  %m = sub i64 %n, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %m, i1 false)
  %sh = sub i64 64, %lz
  %p = shl i64 1, %sh
  ret i64 %p
}

; XOR-filter position triple for a key hash H (segment-partitioned fastrange).
define internal { i64, i64, i64 } @filt_xor_pos(i64 %h, i64 %bl) #3 {
entry:
  ; segment 0
  %w0 = and i64 %h, 4294967295
  %pr0 = mul i64 %w0, %bl
  %p0 = lshr i64 %pr0, 32
  ; segment 1: rotl(h,21)
  %r1hi = shl i64 %h, 21
  %r1lo = lshr i64 %h, 43
  %r1 = or i64 %r1hi, %r1lo
  %w1 = and i64 %r1, 4294967295
  %pr1 = mul i64 %w1, %bl
  %p1r = lshr i64 %pr1, 32
  %p1 = add i64 %p1r, %bl
  ; segment 2: rotl(h,42)
  %r2hi = shl i64 %h, 42
  %r2lo = lshr i64 %h, 22
  %r2 = or i64 %r2hi, %r2lo
  %w2 = and i64 %r2, 4294967295
  %pr2 = mul i64 %w2, %bl
  %p2r = lshr i64 %pr2, 32
  %bl2 = shl i64 %bl, 1
  %p2 = add i64 %p2r, %bl2
  %t0 = insertvalue { i64, i64, i64 } poison, i64 %p0, 0
  %t1 = insertvalue { i64, i64, i64 } %t0, i64 %p1, 1
  %t2 = insertvalue { i64, i64, i64 } %t1, i64 %p2, 2
  ret { i64, i64, i64 } %t2
}

; ======================================================================
;  CUCKOO FILTER
; ======================================================================

; Try to place fp in an empty (==0) slot of bucket %b. Returns 1 if placed.
define internal i1 @filt_cuckoo_place(ptr %pay, i64 %b, i8 %fp) #0 {
entry:
  %off = shl i64 %b, 2
  %base = getelementptr inbounds nuw i8, ptr %pay, i64 %off
  br label %loop

loop:
  %s = phi i64 [ 0, %entry ], [ %s.n, %next ]
  %sp = getelementptr inbounds nuw i8, ptr %base, i64 %s
  %cur = load i8, ptr %sp, align 1
  %empty = icmp eq i8 %cur, 0
  br i1 %empty, label %put, label %next

put:
  store i8 %fp, ptr %sp, align 1
  ret i1 true

next:
  %s.n = add nuw nsw i64 %s, 1
  %more = icmp ult i64 %s.n, 4
  br i1 %more, label %loop, label %full

full:
  ret i1 false
}

define noalias ptr @universe_ds_cuckoo_create(i64 %hint) local_unnamed_addr #1 {
entry:
  ; slots needed ~ hint ; nbuckets = pow2ceil(ceil(hint/4)), min 1.
  %h4 = add i64 %hint, 3
  %need = lshr i64 %h4, 2
  %nb = call i64 @filt_pow2ceil(i64 %need)
  ; payload bytes = nbuckets*4 ; total = 64 + payload (overflow-checked).
  %pay = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nb, i64 4)
  %pay.v = extractvalue { i64, i1 } %pay, 0
  %pay.o = extractvalue { i64, i1 } %pay, 1
  br i1 %pay.o, label %fail, label %tot, !prof !0

tot:
  %t = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pay.v, i64 64)
  %t.v = extractvalue { i64, i1 } %t, 0
  %t.o = extractvalue { i64, i1 } %t, 1
  br i1 %t.o, label %fail, label %alloc, !prof !0

alloc:
  %f = call ptr @calloc(i64 1, i64 %t.v)
  %isnull = icmp eq ptr %f, null
  br i1 %isnull, label %fail, label %init, !prof !0

init:
  store i64 %nb, ptr %f, align 8
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = sub i64 %nb, 1
  store i64 %mask, ptr %maskp, align 8
  ; count@16 already 0 from calloc ; rng@24 seed to a nonzero constant.
  %rngp = getelementptr inbounds nuw i8, ptr %f, i64 24
  store i64 88172645463325252, ptr %rngp, align 8
  ret ptr %f

fail:
  ret ptr null
}

define void @universe_ds_cuckoo_destroy(ptr %f) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %done, label %fr, !prof !0
fr:
  call void @free(ptr nonnull %f)
  br label %done
done:
  ret void
}

define i32 @universe_ds_cuckoo_add(ptr %f, i64 %key) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %err.null, label %pre, !prof !0

err.null:
  ret i32 1

pre:
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = load i64, ptr %maskp, align 8
  %pay = getelementptr inbounds nuw i8, ptr %f, i64 64
  ; fingerprint in [1,255]
  %h = call i64 @filt_mix(i64 %key)
  %fpr = and i64 %h, 255
  %isz = icmp eq i64 %fpr, 0
  %fp64 = select i1 %isz, i64 1, i64 %fpr
  %fp = trunc i64 %fp64 to i8
  ; i1 = (h>>32)&mask ; i2 = i1 ^ (mix(fp)&mask)
  %hh = lshr i64 %h, 32
  %i1 = and i64 %hh, %mask
  %hf0 = call i64 @filt_mix(i64 %fp64)
  %hf = and i64 %hf0, %mask
  %i2 = xor i64 %i1, %hf
  ; try i1 then i2
  %ok1 = call i1 @filt_cuckoo_place(ptr %pay, i64 %i1, i8 %fp)
  br i1 %ok1, label %inserted, label %try2

try2:
  %ok2 = call i1 @filt_cuckoo_place(ptr %pay, i64 %i2, i8 %fp)
  br i1 %ok2, label %inserted, label %kickstart

inserted:
  %cp = getelementptr inbounds nuw i8, ptr %f, i64 16
  %c = load i64, ptr %cp, align 8
  %c.n = add i64 %c, 1
  store i64 %c.n, ptr %cp, align 8
  ret i32 0

kickstart:
  %rngp = getelementptr inbounds nuw i8, ptr %f, i64 24
  %rng0 = load i64, ptr %rngp, align 8
  ; advance rng once to choose starting bucket
  %ra1 = shl i64 %rng0, 13
  %rb1 = xor i64 %rng0, %ra1
  %rc1 = lshr i64 %rb1, 7
  %rd1 = xor i64 %rb1, %rc1
  %re1 = shl i64 %rd1, 17
  %rng1 = xor i64 %rd1, %re1
  %pick = and i64 %rng1, 1
  %pk = icmp eq i64 %pick, 1
  %cur0 = select i1 %pk, i64 %i2, i64 %i1
  br label %kick

kick:
  %n.k = phi i64 [ 0, %kickstart ], [ %n.kn, %miss ]
  %cur = phi i64 [ %cur0, %kickstart ], [ %cur2, %miss ]
  %curfp = phi i8 [ %fp, %kickstart ], [ %old, %miss ]
  %rng = phi i64 [ %rng1, %kickstart ], [ %rngN, %miss ]
  ; advance rng, pick slot 0..3
  %ka = shl i64 %rng, 13
  %kb = xor i64 %rng, %ka
  %kc = lshr i64 %kb, 7
  %kd = xor i64 %kb, %kc
  %ke = shl i64 %kd, 17
  %rngN = xor i64 %kd, %ke
  %slot = and i64 %rngN, 3
  %boff = shl i64 %cur, 2
  %bbase = getelementptr inbounds nuw i8, ptr %pay, i64 %boff
  %slp = getelementptr inbounds nuw i8, ptr %bbase, i64 %slot
  %old = load i8, ptr %slp, align 1
  store i8 %curfp, ptr %slp, align 1
  ; relocate old: alt = cur ^ (mix(old)&mask)
  %old64 = zext i8 %old to i64
  %hfo0 = call i64 @filt_mix(i64 %old64)
  %hfo = and i64 %hfo0, %mask
  %cur2 = xor i64 %cur, %hfo
  %placed = call i1 @filt_cuckoo_place(ptr %pay, i64 %cur2, i8 %old)
  br i1 %placed, label %kickdone, label %miss

miss:
  %n.kn = add nuw nsw i64 %n.k, 1
  %more = icmp ult i64 %n.kn, 500
  br i1 %more, label %kick, label %kfull

kickdone:
  store i64 %rngN, ptr %rngp, align 8
  %cp2 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %c2 = load i64, ptr %cp2, align 8
  %c2.n = add i64 %c2, 1
  store i64 %c2.n, ptr %cp2, align 8
  ret i32 0

kfull:
  store i64 %rngN, ptr %rngp, align 8
  ret i32 6
}

define i32 @universe_ds_cuckoo_contains(ptr %f, i64 %key) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %absent, label %pre, !prof !0

absent:
  ret i32 0

pre:
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = load i64, ptr %maskp, align 8
  %pay = getelementptr inbounds nuw i8, ptr %f, i64 64
  %h = call i64 @filt_mix(i64 %key)
  %fpr = and i64 %h, 255
  %isz = icmp eq i64 %fpr, 0
  %fp64 = select i1 %isz, i64 1, i64 %fpr
  %fp32 = trunc i64 %fp64 to i32
  %hh = lshr i64 %h, 32
  %i1 = and i64 %hh, %mask
  %hf0 = call i64 @filt_mix(i64 %fp64)
  %hf = and i64 %hf0, %mask
  %i2 = xor i64 %i1, %hf
  ; SWAR: broadcast fp, haszero(word ^ pat)
  %pat = mul i32 %fp32, 16843009
  %o1 = shl i64 %i1, 2
  %b1p = getelementptr inbounds nuw i8, ptr %pay, i64 %o1
  %w1 = load i32, ptr %b1p, align 4
  %v1 = xor i32 %w1, %pat
  %v1m = sub i32 %v1, 16843009
  %nv1 = xor i32 %v1, -1
  %a1 = and i32 %v1m, %nv1
  %hz1 = and i32 %a1, -2139062144
  %m1 = icmp ne i32 %hz1, 0
  %o2 = shl i64 %i2, 2
  %b2p = getelementptr inbounds nuw i8, ptr %pay, i64 %o2
  %w2 = load i32, ptr %b2p, align 4
  %v2 = xor i32 %w2, %pat
  %v2m = sub i32 %v2, 16843009
  %nv2 = xor i32 %v2, -1
  %a2 = and i32 %v2m, %nv2
  %hz2 = and i32 %a2, -2139062144
  %m2 = icmp ne i32 %hz2, 0
  %hit = or i1 %m1, %m2
  %r = zext i1 %hit to i32
  ret i32 %r
}

define i32 @universe_ds_cuckoo_delete(ptr %f, i64 %key) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %err.null, label %pre, !prof !0

err.null:
  ret i32 1

pre:
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = load i64, ptr %maskp, align 8
  %pay = getelementptr inbounds nuw i8, ptr %f, i64 64
  %h = call i64 @filt_mix(i64 %key)
  %fpr = and i64 %h, 255
  %isz = icmp eq i64 %fpr, 0
  %fp64 = select i1 %isz, i64 1, i64 %fpr
  %fp = trunc i64 %fp64 to i8
  %hh = lshr i64 %h, 32
  %i1 = and i64 %hh, %mask
  %hf0 = call i64 @filt_mix(i64 %fp64)
  %hf = and i64 %hf0, %mask
  %i2 = xor i64 %i1, %hf
  ; scan bucket i1 slots
  %o1 = shl i64 %i1, 2
  %base1 = getelementptr inbounds nuw i8, ptr %pay, i64 %o1
  br label %l1

l1:
  %s1 = phi i64 [ 0, %pre ], [ %s1.n, %n1 ]
  %sp1 = getelementptr inbounds nuw i8, ptr %base1, i64 %s1
  %c1 = load i8, ptr %sp1, align 1
  %eq1 = icmp eq i8 %c1, %fp
  br i1 %eq1, label %del1, label %n1

del1:
  store i8 0, ptr %sp1, align 1
  br label %dec

n1:
  %s1.n = add nuw nsw i64 %s1, 1
  %m1 = icmp ult i64 %s1.n, 4
  br i1 %m1, label %l1, label %scan2

scan2:
  %o2 = shl i64 %i2, 2
  %base2 = getelementptr inbounds nuw i8, ptr %pay, i64 %o2
  br label %l2

l2:
  %s2 = phi i64 [ 0, %scan2 ], [ %s2.n, %n2 ]
  %sp2 = getelementptr inbounds nuw i8, ptr %base2, i64 %s2
  %c2 = load i8, ptr %sp2, align 1
  %eq2 = icmp eq i8 %c2, %fp
  br i1 %eq2, label %del2, label %n2

del2:
  store i8 0, ptr %sp2, align 1
  br label %dec

n2:
  %s2.n = add nuw nsw i64 %s2, 1
  %m2 = icmp ult i64 %s2.n, 4
  br i1 %m2, label %l2, label %notfound

dec:
  %cp = getelementptr inbounds nuw i8, ptr %f, i64 16
  %c = load i64, ptr %cp, align 8
  %c.n = sub i64 %c, 1
  store i64 %c.n, ptr %cp, align 8
  ret i32 0

notfound:
  ret i32 5
}

define i64 @universe_ds_cuckoo_count(ptr %f) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %z, label %do, !prof !0
z:
  ret i64 0
do:
  %cp = getelementptr inbounds nuw i8, ptr %f, i64 16
  %c = load i64, ptr %cp, align 8
  ret i64 %c
}

define i64 @universe_ds_cuckoo_capacity(ptr %f) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %z, label %do, !prof !0
z:
  ret i64 0
do:
  %nb = load i64, ptr %f, align 8
  %cap = shl i64 %nb, 2
  ret i64 %cap
}

; ======================================================================
;  XOR FILTER (static, build-once)
; ======================================================================

define ptr @universe_ds_xor_build(ptr %keys, i64 %n) local_unnamed_addr #1 {
entry:
  %kn = icmp eq ptr %keys, null
  %n0 = icmp eq i64 %n, 0
  ; keys==null only allowed when n==0.
  br i1 %kn, label %chknull, label %sizes

chknull:
  br i1 %n0, label %sizes, label %fail0

fail0:
  ret ptr null

sizes:
  ; cap = 32 + (n*123)/100 ; blockLength = ceil(cap/3) ; arrayLength=3*bl.
  %n123 = mul i64 %n, 123
  %ndiv = udiv i64 %n123, 100
  %cap = add i64 %ndiv, 32
  %cap2 = add i64 %cap, 2
  %bl = udiv i64 %cap2, 3
  %blz = icmp eq i64 %bl, 0
  %blk = select i1 %blz, i64 1, i64 %bl
  %m = mul i64 %blk, 3
  ; scratch: H_xor(m*8), H_cnt(m*4), Q(m*4), stackH(n*8), stackP(n*4)
  %mx8 = shl i64 %m, 3
  %hxor = call ptr @malloc(i64 %mx8)
  %e1 = icmp eq ptr %hxor, null
  br i1 %e1, label %fail0, label %a2

a2:
  %mx4 = shl i64 %m, 2
  %hcnt = call ptr @malloc(i64 %mx4)
  %e2 = icmp eq ptr %hcnt, null
  br i1 %e2, label %f1, label %a3

a3:
  %queue = call ptr @malloc(i64 %mx4)
  %e3 = icmp eq ptr %queue, null
  br i1 %e3, label %f2, label %a4

a4:
  ; stack sizes: use max(n,1) so malloc(0) never returns ambiguous.
  %nz = icmp eq i64 %n, 0
  %ns = select i1 %nz, i64 1, i64 %n
  %ns8 = shl i64 %ns, 3
  %stackh = call ptr @malloc(i64 %ns8)
  %e4 = icmp eq ptr %stackh, null
  br i1 %e4, label %f3, label %a5

a5:
  %ns4 = shl i64 %ns, 2
  %stackp = call ptr @malloc(i64 %ns4)
  %e5 = icmp eq ptr %stackp, null
  br i1 %e5, label %f4, label %attempt

; ---- reseeding attempt loop ----
attempt:
  %seed = phi i64 [ 0, %a5 ], [ %seed.n, %retry ]
  %att = phi i64 [ 0, %a5 ], [ %att.n, %retry ]
  ; zero H_xor and H_cnt
  call void @llvm.memset.p0.i64(ptr %hxor, i8 0, i64 %mx8, i1 false)
  call void @llvm.memset.p0.i64(ptr %hcnt, i8 0, i64 %mx4, i1 false)
  %seedmix = call i64 @filt_mix(i64 %seed)
  br i1 %n0, label %peeldone, label %scatter

scatter:
  %si = phi i64 [ 0, %attempt ], [ %si.n, %scatter ]
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %si
  %kv = load i64, ptr %kp, align 8
  %kmix = xor i64 %kv, %seedmix
  %kh = call i64 @filt_mix(i64 %kmix)
  %tp = call { i64, i64, i64 } @filt_xor_pos(i64 %kh, i64 %blk)
  %sp0 = extractvalue { i64, i64, i64 } %tp, 0
  %sp1 = extractvalue { i64, i64, i64 } %tp, 1
  %sp2 = extractvalue { i64, i64, i64 } %tp, 2
  ; H_xor[p]^=kh ; H_cnt[p]++
  %xp0 = getelementptr inbounds nuw i64, ptr %hxor, i64 %sp0
  %xv0 = load i64, ptr %xp0, align 8
  %xn0 = xor i64 %xv0, %kh
  store i64 %xn0, ptr %xp0, align 8
  %cp0 = getelementptr inbounds nuw i32, ptr %hcnt, i64 %sp0
  %cv0 = load i32, ptr %cp0, align 4
  %cn0 = add i32 %cv0, 1
  store i32 %cn0, ptr %cp0, align 4
  %xp1 = getelementptr inbounds nuw i64, ptr %hxor, i64 %sp1
  %xv1 = load i64, ptr %xp1, align 8
  %xn1 = xor i64 %xv1, %kh
  store i64 %xn1, ptr %xp1, align 8
  %cp1 = getelementptr inbounds nuw i32, ptr %hcnt, i64 %sp1
  %cv1 = load i32, ptr %cp1, align 4
  %cn1 = add i32 %cv1, 1
  store i32 %cn1, ptr %cp1, align 4
  %xp2 = getelementptr inbounds nuw i64, ptr %hxor, i64 %sp2
  %xv2 = load i64, ptr %xp2, align 8
  %xn2 = xor i64 %xv2, %kh
  store i64 %xn2, ptr %xp2, align 8
  %cp2 = getelementptr inbounds nuw i32, ptr %hcnt, i64 %sp2
  %cv2 = load i32, ptr %cp2, align 4
  %cn2 = add i32 %cv2, 1
  store i32 %cn2, ptr %cp2, align 4
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, %n
  br i1 %smore, label %scatter, label %qinit

; seed the queue with all positions of count==1
qinit:
  br label %qscan

qscan:
  %qi = phi i64 [ 0, %qinit ], [ %qi.n, %qnext ]
  %qsz = phi i64 [ 0, %qinit ], [ %qsz2, %qnext ]
  %qcp = getelementptr inbounds nuw i32, ptr %hcnt, i64 %qi
  %qcv = load i32, ptr %qcp, align 4
  %is1 = icmp eq i32 %qcv, 1
  br i1 %is1, label %qpush, label %qnext

qpush:
  %qwp = getelementptr inbounds nuw i32, ptr %queue, i64 %qsz
  %qi32 = trunc i64 %qi to i32
  store i32 %qi32, ptr %qwp, align 4
  %qsz.p = add nuw i64 %qsz, 1
  br label %qnext

qnext:
  %qsz2 = phi i64 [ %qsz.p, %qpush ], [ %qsz, %qscan ]
  %qi.n = add nuw i64 %qi, 1
  %qmore = icmp ult i64 %qi.n, %m
  br i1 %qmore, label %qscan, label %peelinit

peelinit:
  br label %peel

; ---- peel: LIFO over queue; stack records (kh, outpos) ----
peel:
  %qhead = phi i64 [ %qsz2, %peelinit ], [ %qhead2, %peelcont ]
  %top = phi i64 [ 0, %peelinit ], [ %top2, %peelcont ]
  %hasq = icmp ugt i64 %qhead, 0
  br i1 %hasq, label %pop, label %peeldone

pop:
  %qhead.d = sub i64 %qhead, 1
  %popp = getelementptr inbounds nuw i32, ptr %queue, i64 %qhead.d
  %pos32 = load i32, ptr %popp, align 4
  %pos = zext i32 %pos32 to i64
  ; stale? cnt must still be 1
  %pcp = getelementptr inbounds nuw i32, ptr %hcnt, i64 %pos
  %pcv = load i32, ptr %pcp, align 4
  %still1 = icmp eq i32 %pcv, 1
  br i1 %still1, label %emit, label %peelcont.stale

peelcont.stale:
  br label %peelcont_j

emit:
  ; recover the lone key hash, push to stack
  %hxp = getelementptr inbounds nuw i64, ptr %hxor, i64 %pos
  %kh2 = load i64, ptr %hxp, align 8
  %shp = getelementptr inbounds nuw i64, ptr %stackh, i64 %top
  store i64 %kh2, ptr %shp, align 8
  %spp = getelementptr inbounds nuw i32, ptr %stackp, i64 %top
  store i32 %pos32, ptr %spp, align 4
  %top.n = add nuw i64 %top, 1
  ; remove key from all 3 positions
  %tp2 = call { i64, i64, i64 } @filt_xor_pos(i64 %kh2, i64 %blk)
  %ep0 = extractvalue { i64, i64, i64 } %tp2, 0
  %ep1 = extractvalue { i64, i64, i64 } %tp2, 1
  %ep2 = extractvalue { i64, i64, i64 } %tp2, 2
  ; process position ep0
  %d0cp = getelementptr inbounds nuw i32, ptr %hcnt, i64 %ep0
  %d0cv = load i32, ptr %d0cp, align 4
  %d0cn = sub i32 %d0cv, 1
  store i32 %d0cn, ptr %d0cp, align 4
  %d0xp = getelementptr inbounds nuw i64, ptr %hxor, i64 %ep0
  %d0xv = load i64, ptr %d0xp, align 8
  %d0xn = xor i64 %d0xv, %kh2
  store i64 %d0xn, ptr %d0xp, align 8
  %d0is1 = icmp eq i32 %d0cn, 1
  br i1 %d0is1, label %pu0, label %af0

pu0:
  %qw0 = getelementptr inbounds nuw i32, ptr %queue, i64 %qhead.d
  %ep0.32 = trunc i64 %ep0 to i32
  store i32 %ep0.32, ptr %qw0, align 4
  %qh0.n = add nuw i64 %qhead.d, 1
  br label %af0

af0:
  %qhA = phi i64 [ %qh0.n, %pu0 ], [ %qhead.d, %emit ]
  ; process position ep1
  %d1cp = getelementptr inbounds nuw i32, ptr %hcnt, i64 %ep1
  %d1cv = load i32, ptr %d1cp, align 4
  %d1cn = sub i32 %d1cv, 1
  store i32 %d1cn, ptr %d1cp, align 4
  %d1xp = getelementptr inbounds nuw i64, ptr %hxor, i64 %ep1
  %d1xv = load i64, ptr %d1xp, align 8
  %d1xn = xor i64 %d1xv, %kh2
  store i64 %d1xn, ptr %d1xp, align 8
  %d1is1 = icmp eq i32 %d1cn, 1
  br i1 %d1is1, label %pu1, label %af1

pu1:
  %qw1 = getelementptr inbounds nuw i32, ptr %queue, i64 %qhA
  %ep1.32 = trunc i64 %ep1 to i32
  store i32 %ep1.32, ptr %qw1, align 4
  %qhA.n = add nuw i64 %qhA, 1
  br label %af1

af1:
  %qhB = phi i64 [ %qhA.n, %pu1 ], [ %qhA, %af0 ]
  ; process position ep2
  %d2cp = getelementptr inbounds nuw i32, ptr %hcnt, i64 %ep2
  %d2cv = load i32, ptr %d2cp, align 4
  %d2cn = sub i32 %d2cv, 1
  store i32 %d2cn, ptr %d2cp, align 4
  %d2xp = getelementptr inbounds nuw i64, ptr %hxor, i64 %ep2
  %d2xv = load i64, ptr %d2xp, align 8
  %d2xn = xor i64 %d2xv, %kh2
  store i64 %d2xn, ptr %d2xp, align 8
  %d2is1 = icmp eq i32 %d2cn, 1
  br i1 %d2is1, label %pu2, label %af2

pu2:
  %qw2 = getelementptr inbounds nuw i32, ptr %queue, i64 %qhB
  %ep2.32 = trunc i64 %ep2 to i32
  store i32 %ep2.32, ptr %qw2, align 4
  %qhB.n = add nuw i64 %qhB, 1
  br label %af2

af2:
  %qhC = phi i64 [ %qhB.n, %pu2 ], [ %qhB, %af1 ]
  br label %peelcont_j

; merge stale (no stack change) and emit (stack grew) back into loop
peelcont_j:
  %qhead2 = phi i64 [ %qhead.d, %peelcont.stale ], [ %qhC, %af2 ]
  %top2 = phi i64 [ %top, %peelcont.stale ], [ %top.n, %af2 ]
  br label %peelcont

peelcont:
  br label %peel

peeldone:
  %fintop = phi i64 [ 0, %attempt ], [ %top, %peel ]
  %all = icmp eq i64 %fintop, %n
  br i1 %all, label %assign, label %retry

retry:
  %att.n = add nuw i64 %att, 1
  %seed.n = add i64 %seed, 1
  %retrymore = icmp ult i64 %att.n, 100
  br i1 %retrymore, label %attempt, label %failscratch

; ---- assignment: build the filter, fill b[] in reverse peel order ----
assign:
  %albytes = add i64 %m, 64
  %filt = call ptr @calloc(i64 1, i64 %albytes)
  %fe = icmp eq ptr %filt, null
  br i1 %fe, label %failscratch, label %filt.init

filt.init:
  store i64 %seed, ptr %filt, align 8
  %blp = getelementptr inbounds nuw i8, ptr %filt, i64 8
  store i64 %blk, ptr %blp, align 8
  %alp = getelementptr inbounds nuw i8, ptr %filt, i64 16
  store i64 %m, ptr %alp, align 8
  %szp = getelementptr inbounds nuw i8, ptr %filt, i64 24
  store i64 %n, ptr %szp, align 8
  %bb = getelementptr inbounds nuw i8, ptr %filt, i64 64
  ; if n==0, nothing to assign
  br i1 %n0, label %succeed, label %astep

astep:
  ; t from fintop downto 1, operating on stack[t-1]
  %tt = phi i64 [ %fintop, %filt.init ], [ %tt.d, %acont ]
  %tt.d = sub i64 %tt, 1
  %shp2 = getelementptr inbounds nuw i64, ptr %stackh, i64 %tt.d
  %akh = load i64, ptr %shp2, align 8
  %spp2 = getelementptr inbounds nuw i32, ptr %stackp, i64 %tt.d
  %apos32 = load i32, ptr %spp2, align 4
  %apos = zext i32 %apos32 to i64
  %atp = call { i64, i64, i64 } @filt_xor_pos(i64 %akh, i64 %blk)
  %ap0 = extractvalue { i64, i64, i64 } %atp, 0
  %ap1 = extractvalue { i64, i64, i64 } %atp, 1
  %ap2 = extractvalue { i64, i64, i64 } %atp, 2
  ; fp = low byte of (kh ^ (kh>>32))
  %khs = lshr i64 %akh, 32
  %khx = xor i64 %akh, %khs
  %fp8 = trunc i64 %khx to i8
  ; val = fp ^ b[p0] ^ b[p1] ^ b[p2]  (b[apos] currently 0)
  %bp0 = getelementptr inbounds nuw i8, ptr %bb, i64 %ap0
  %bv0 = load i8, ptr %bp0, align 1
  %bp1 = getelementptr inbounds nuw i8, ptr %bb, i64 %ap1
  %bv1 = load i8, ptr %bp1, align 1
  %bp2 = getelementptr inbounds nuw i8, ptr %bb, i64 %ap2
  %bv2 = load i8, ptr %bp2, align 1
  %x01 = xor i8 %bv0, %bv1
  %x012 = xor i8 %x01, %bv2
  %val = xor i8 %fp8, %x012
  %bpos = getelementptr inbounds nuw i8, ptr %bb, i64 %apos
  store i8 %val, ptr %bpos, align 1
  br label %acont

acont:
  %amore = icmp ugt i64 %tt.d, 0
  br i1 %amore, label %astep, label %succeed

succeed:
  ; free scratch, return filter
  call void @free(ptr %hxor)
  call void @free(ptr %hcnt)
  call void @free(ptr %queue)
  call void @free(ptr %stackh)
  call void @free(ptr %stackp)
  ret ptr %filt

failscratch:
  call void @free(ptr %hxor)
  call void @free(ptr %hcnt)
  call void @free(ptr %queue)
  call void @free(ptr %stackh)
  call void @free(ptr %stackp)
  ret ptr null

; staged frees for early alloc failures
f4:
  call void @free(ptr %stackh)
  br label %f3
f3:
  call void @free(ptr %queue)
  br label %f2
f2:
  call void @free(ptr %hcnt)
  br label %f1
f1:
  call void @free(ptr %hxor)
  ret ptr null
}

define void @universe_ds_xor_destroy(ptr %f) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %done, label %fr, !prof !0
fr:
  call void @free(ptr nonnull %f)
  br label %done
done:
  ret void
}

define i32 @universe_ds_xor_contains(ptr %f, i64 %key) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %absent, label %pre, !prof !0

absent:
  ret i32 0

pre:
  %seed = load i64, ptr %f, align 8
  %blp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %bl = load i64, ptr %blp, align 8
  %bb = getelementptr inbounds nuw i8, ptr %f, i64 64
  %seedmix = call i64 @filt_mix(i64 %seed)
  %kmix = xor i64 %key, %seedmix
  %kh = call i64 @filt_mix(i64 %kmix)
  %tp = call { i64, i64, i64 } @filt_xor_pos(i64 %kh, i64 %bl)
  %p0 = extractvalue { i64, i64, i64 } %tp, 0
  %p1 = extractvalue { i64, i64, i64 } %tp, 1
  %p2 = extractvalue { i64, i64, i64 } %tp, 2
  %khs = lshr i64 %kh, 32
  %khx = xor i64 %kh, %khs
  %fp8 = trunc i64 %khx to i8
  %bp0 = getelementptr inbounds nuw i8, ptr %bb, i64 %p0
  %bv0 = load i8, ptr %bp0, align 1
  %bp1 = getelementptr inbounds nuw i8, ptr %bb, i64 %p1
  %bv1 = load i8, ptr %bp1, align 1
  %bp2 = getelementptr inbounds nuw i8, ptr %bb, i64 %p2
  %bv2 = load i8, ptr %bp2, align 1
  %x01 = xor i8 %bv0, %bv1
  %x012 = xor i8 %x01, %bv2
  %eq = icmp eq i8 %x012, %fp8
  %r = zext i1 %eq to i32
  ret i32 %r
}

; ======================================================================
;  BLOCKED (CACHE-LINE) BLOOM   k = 8 bits within one 512-bit block
; ======================================================================

define noalias ptr @universe_ds_bbloom_create(i64 %nkeys, i64 %bpk) local_unnamed_addr #1 {
entry:
  ; total_bits = nkeys*bpk (overflow-checked) ; nblocks=pow2ceil(ceil(/512))
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nkeys, i64 %bpk)
  %tb.v = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  br i1 %tb.o, label %fail, label %blocks, !prof !0

blocks:
  %tb511 = add i64 %tb.v, 511
  %need = lshr i64 %tb511, 9
  %nb = call i64 @filt_pow2ceil(i64 %need)
  ; payload = nblocks*64 (overflow-checked) ; total = 64 + payload
  %pay = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %nb, i64 64)
  %pay.v = extractvalue { i64, i1 } %pay, 0
  %pay.o = extractvalue { i64, i1 } %pay, 1
  br i1 %pay.o, label %fail, label %tot, !prof !0

tot:
  %t = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %pay.v, i64 64)
  %t.v = extractvalue { i64, i1 } %t, 0
  %t.o = extractvalue { i64, i1 } %t, 1
  br i1 %t.o, label %fail, label %alloc, !prof !0

alloc:
  %f = call ptr @calloc(i64 1, i64 %t.v)
  %isnull = icmp eq ptr %f, null
  br i1 %isnull, label %fail, label %init, !prof !0

init:
  store i64 %nb, ptr %f, align 8
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = sub i64 %nb, 1
  store i64 %mask, ptr %maskp, align 8
  ret ptr %f

fail:
  ret ptr null
}

define void @universe_ds_bbloom_destroy(ptr %f) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %done, label %fr, !prof !0
fr:
  call void @free(ptr nonnull %f)
  br label %done
done:
  ret void
}

define i32 @universe_ds_bbloom_add(ptr %f, i64 %key) local_unnamed_addr #0 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %err.null, label %pre, !prof !0

err.null:
  ret i32 1

pre:
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = load i64, ptr %maskp, align 8
  %pay = getelementptr inbounds nuw i8, ptr %f, i64 64
  %h1 = call i64 @filt_mix(i64 %key)
  %h2 = call i64 @filt_mix(i64 %h1)
  %blk = lshr i64 %h1, 32
  %block = and i64 %blk, %mask
  %boff = shl i64 %block, 6
  %bbase = getelementptr inbounds nuw i8, ptr %pay, i64 %boff
  %bodd = or i64 %h2, 1
  br label %loop

loop:
  %i = phi i64 [ 0, %pre ], [ %i.n, %loop ]
  %ib = mul i64 %i, %bodd
  %bit = add i64 %h1, %ib
  %pos = and i64 %bit, 511
  %word = lshr i64 %pos, 6
  %bitidx = and i64 %pos, 63
  %wp = getelementptr inbounds nuw i64, ptr %bbase, i64 %word
  %cur = load i64, ptr %wp, align 8
  %m1 = shl nuw i64 1, %bitidx
  %new = or i64 %cur, %m1
  store i64 %new, ptr %wp, align 8
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 8
  br i1 %more, label %loop, label %done

done:
  ret i32 0
}

define i32 @universe_ds_bbloom_contains(ptr %f, i64 %key) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %f, null
  br i1 %n, label %absent, label %pre, !prof !0

absent:
  ret i32 0

pre:
  %maskp = getelementptr inbounds nuw i8, ptr %f, i64 8
  %mask = load i64, ptr %maskp, align 8
  %pay = getelementptr inbounds nuw i8, ptr %f, i64 64
  %h1 = call i64 @filt_mix(i64 %key)
  %h2 = call i64 @filt_mix(i64 %h1)
  %blk = lshr i64 %h1, 32
  %block = and i64 %blk, %mask
  %boff = shl i64 %block, 6
  %bbase = getelementptr inbounds nuw i8, ptr %pay, i64 %boff
  %bodd = or i64 %h2, 1
  br label %loop

loop:
  %i = phi i64 [ 0, %pre ], [ %i.n, %loop ]
  %acc = phi i64 [ 1, %pre ], [ %acc.n, %loop ]
  %ib = mul i64 %i, %bodd
  %bit = add i64 %h1, %ib
  %pos = and i64 %bit, 511
  %word = lshr i64 %pos, 6
  %bitidx = and i64 %pos, 63
  %wp = getelementptr inbounds nuw i64, ptr %bbase, i64 %word
  %cur = load i64, ptr %wp, align 8
  %sh = lshr i64 %cur, %bitidx
  %isset = and i64 %sh, 1
  %acc.n = and i64 %acc, %isset
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 8
  br i1 %more, label %loop, label %done

done:
  %r = trunc i64 %acc.n to i32
  ret i32 %r
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync memory(none) alwaysinline }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
