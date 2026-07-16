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

; Minimal perfect hash (MPH) over a STATIC set of byte-string keys.
;
; WHEN TO CHOOSE THIS:
;   * A fixed, known-at-build-time set of N DISTINCT byte-string keys that must
;     map to the dense range [0,N) with ZERO collisions and O(1), probe-free
;     lookup — e.g. "HTTP header name -> id", opcode tables, keyword -> token.
;   * Read-only after build. Not a mutable map (no insert/delete) — for that use
;     the swiss hashmap. The win here is that a member lookup is a single string
;     hash + one table load + a mix + one mod, with NO probe loop and NO branch
;     back-and-forth: strictly fewer memory touches than any open-addressing get.
;
; DESIGN (first principles; a compress-hash-displace / displacement-array MPH):
;   * A perfect hash is built in two levels. Level 1 buckets the keys; level 2
;     stores, per bucket, a small DISPLACEMENT that resolves that bucket's keys
;     into distinct free slots of the output range [0,N). Lookup replays the two
;     levels: bucket = hash(key) & rmask; d = G[bucket]; slot = f(key, d).
;   * Hashing: one FNV-1a pass over the key bytes seeded by a per-build seed S,
;     then a splitmix64 finalizer for avalanche => h. The bucket index is the low
;     bits of h; a SECOND finalize of (h ^ GOLDEN) => hslot, a slot-space hash
;     decorrelated from the bucket bits. For displacement d, slot(key,d) =
;     (hslot ^ splitmix64(d)) mod N. Different d perturbs every bucket member's
;     slot deterministically; lookup recomputes splitmix64(d) from the stored d.
;   * BUCKETS = next_pow2(max(N,2)) * 2  => load factor 0.25..0.5. A low load
;     makes construction converge fast: most buckets are empty or singletons,
;     multi-key buckets are tiny. rmask = BUCKETS-1 so the level-1 index is a
;     mask (no division); only the level-2 slot needs one `urem N` (unavoidable
;     for a MINIMAL — exactly-[0,N) — hash).
;   * G[bucket] (i32, one per bucket) encodes level 2:
;       G == 0  : empty bucket (default). A non-member landing here still gets a
;                 valid slot = hslot mod N (a perfect hash is not a membership
;                 test — see lookup_checked to disambiguate).
;       G  > 0  : displacement d for a MULTI-key bucket; slot = f(key,d).
;       G  < 0  : a SINGLETON bucket assigned a slot directly; slot = -1 - G.
;                 Singletons are placed into the lowest free slot via a cursor,
;                 which makes construction cheap (no displacement search for the
;                 common size-1 bucket).
;   * CONSTRUCTION: hash all keys, group by bucket (CSR), counting-sort buckets
;     LARGEST-FIRST (big buckets placed while the table is near-empty), then for
;     each bucket: size 1 -> next free slot; size>1 -> search d=1,2,.. until all
;     members land on distinct, still-free slots (an occupancy bitmap + a
;     per-slot generation stamp reject collisions in O(1)). If a bucket exhausts
;     the displacement budget (MAXD), the whole attempt fails and we RETRY with a
;     fresh seed S (up to MAX_ATTEMPT). Duplicate input keys collide identically
;     in every attempt and therefore cause build failure (returns null) — the
;     contract requires DISTINCT keys.
;   * ONE allocation holds the immutable structure: a 64-byte header, the G array
;     (BUCKETS x i32), a per-slot key DESCRIPTOR array (N x {i32 off, i32 len})
;     and a KEY BLOB (all key bytes concatenated). The descriptor + blob let
;     lookup_checked compare the stored owner of a slot against the query to
;     answer membership. Build scratch is separate (freed before returning).
;
; Header (64 B): n@0 buckets@8 rmask@16 seed@24 goff@32 descoff@40 bloboff@48 _@56
; Block: [hdr 64] [G: buckets x i32] [desc: n x {off,len}] [blob: sum(keylens)]
;
; API (error/sentinels: build -> null on failure; lookup_checked -> -1 miss):
;   ptr  universe_ds_mph_build(ptr keys, ptr keylens, i64 n)   ; keys: ptr[n],
;                                                    keylens: i64[n]; NULL on fail
;   i64  universe_ds_mph_lookup(ptr m, ptr key, i64 len)   ; slot in [0,n)
;   i64  universe_ds_mph_lookup_checked(ptr m, ptr key, i64 len) ; slot or -1
;   i64  universe_ds_mph_slot_count(ptr m)                 ; n
;   void universe_ds_mph_destroy(ptr m)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; splitmix64 finalizer — strong avalanche, no division.
define internal i64 @mph_fin(i64 %x) #0 {
entry:
  %a1 = lshr i64 %x, 30
  %a2 = xor i64 %a1, %x
  %a3 = mul i64 %a2, -4658895280553007687      ; 0xbf58476d1ce4e5b9
  %a4 = lshr i64 %a3, 27
  %a5 = xor i64 %a4, %a3
  %a6 = mul i64 %a5, -7723592293110705685      ; 0x94d049bb133111eb
  %a7 = lshr i64 %a6, 31
  %a8 = xor i64 %a7, %a6
  ret i64 %a8
}

; Seeded FNV-1a variant hashing 8 bytes per step (a dependent multiply chain of
; length ceil(len/8), not len), with a masked byte tail; splitmix64-finalized for
; avalanche => 64-bit key hash. Word loads are always in bounds (i+8 <= len);
; the <8-byte tail is assembled byte-by-byte, never over-reading the buffer.
define internal i64 @mph_strhash(ptr %key, i64 %len, i64 %seed) #3 {
entry:
  %acc0 = xor i64 -3750763034362895579, %seed  ; FNV offset basis ^ seed
  br label %wloop.head

wloop.head:
  %i = phi i64 [ 0, %entry ], [ %i8, %wloop.body ]
  %acc = phi i64 [ %acc0, %entry ], [ %acc.n, %wloop.body ]
  %i8 = add nuw i64 %i, 8
  %hasword = icmp ule i64 %i8, %len
  br i1 %hasword, label %wloop.body, label %tail

wloop.body:
  %wp = getelementptr inbounds i8, ptr %key, i64 %i
  %w = load i64, ptr %wp, align 1
  %wx = xor i64 %acc, %w
  %acc.n = mul i64 %wx, 1099511628211          ; FNV prime 0x100000001b3
  br label %wloop.head

tail:
  %rem = sub i64 %len, %i
  %hastail = icmp ne i64 %rem, 0
  br i1 %hastail, label %tloop.head, label %finalize

tloop.head:
  %ti = phi i64 [ 0, %tail ], [ %ti.next, %tloop.body ]
  %tw = phi i64 [ 0, %tail ], [ %tw.n, %tloop.body ]
  %tdone = icmp eq i64 %ti, %rem
  br i1 %tdone, label %tmix, label %tloop.body

tloop.body:
  %tidx = add i64 %i, %ti
  %tbp = getelementptr inbounds i8, ptr %key, i64 %tidx
  %tb = load i8, ptr %tbp, align 1
  %tz = zext i8 %tb to i64
  %tshamt = shl nuw nsw i64 %ti, 3             ; ti in [0,7] -> shift [0,48]
  %tsh = shl i64 %tz, %tshamt
  %tw.n = or i64 %tw, %tsh
  %ti.next = add nuw i64 %ti, 1
  br label %tloop.head

tmix:
  %tx = xor i64 %acc, %tw
  %acc.tail = mul i64 %tx, 1099511628211
  br label %finalize

finalize:
  %accf = phi i64 [ %acc, %tail ], [ %acc.tail, %tmix ]
  %mixed = xor i64 %accf, %len
  %h = call i64 @mph_fin(i64 %mixed)
  ret i64 %h
}

; Byte-equality of two buffers of the given length (keys are short).
define internal i1 @mph_bytes_eq(ptr %a, ptr %b, i64 %len) #3 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %eq, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %pb = getelementptr inbounds i8, ptr %b, i64 %i
  %ca = load i8, ptr %pa, align 1
  %cb = load i8, ptr %pb, align 1
  %ne = icmp ne i8 %ca, %cb
  br i1 %ne, label %neq, label %cont

cont:
  %i.n = add nuw i64 %i, 1
  %d = icmp eq i64 %i.n, %len
  br i1 %d, label %eq, label %loop

eq:
  ret i1 true

neq:
  ret i1 false
}

; ===========================================================================
; build
; ===========================================================================
define ptr @universe_ds_mph_build(ptr %keys, ptr %keylens, i64 %n) local_unnamed_addr #1 {
entry:
  %wk = alloca [14 x ptr], align 8
  %gen.slot = alloca i64, align 8
  %fc.slot = alloca i64, align 8
  %null.keys = icmp eq ptr %keys, null
  %null.lens = icmp eq ptr %keylens, null
  %badptr = or i1 %null.keys, %null.lens
  br i1 %badptr, label %ret.null.direct, label %chk.n, !prof !0

chk.n:
  %n.zero = icmp eq i64 %n, 0
  %n.big = icmp ugt i64 %n, 268435456          ; cap 2^28 keys
  %n.bad = or i1 %n.zero, %n.big
  br i1 %n.bad, label %ret.null.direct, label %compute, !prof !0

compute:
  %maxn2 = call i64 @llvm.umax.i64(i64 %n, i64 2)
  %bm1 = sub i64 %maxn2, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %bm1, i1 false)
  %shift = sub i64 64, %lz
  %r0 = shl i64 1, %shift
  %r = shl i64 %r0, 1                           ; buckets = next_pow2 * 2
  %rmask = sub i64 %r, 1
  %n4 = shl i64 %n, 2
  %n8 = shl i64 %n, 3
  %r4 = shl i64 %r, 2
  %np1 = add i64 %n, 1
  %np14 = shl i64 %np1, 2
  %nadd = add i64 %n, 63
  %nw = lshr i64 %nadd, 6
  %nw8 = shl i64 %nw, 3
  br label %bt.head

bt.head:
  %bti = phi i64 [ 0, %compute ], [ %bti.next, %bt.cont ]
  %btrun = phi i64 [ 0, %compute ], [ %btrun.next, %bt.cont ]
  %btdone = icmp eq i64 %bti, %n
  br i1 %btdone, label %alloc.wk, label %bt.body

bt.body:
  %btlp = getelementptr inbounds i64, ptr %keylens, i64 %bti
  %btlen = load i64, ptr %btlp, align 8
  %btlen.big = icmp ugt i64 %btlen, 4294967295
  br i1 %btlen.big, label %ret.null.direct, label %bt.add, !prof !0

bt.add:
  %btrun.next = add i64 %btrun, %btlen
  %btrun.big = icmp ugt i64 %btrun.next, 4294967295
  br i1 %btrun.big, label %ret.null.direct, label %bt.cont, !prof !0

bt.cont:
  %bti.next = add nuw i64 %bti, 1
  br label %bt.head

ret.null.direct:
  ret ptr null

alloc.wk:
  call void @llvm.memset.p0.i64(ptr %wk, i8 0, i64 112, i1 false)
  %a0 = call ptr @malloc(i64 %n8)               ; 0 hslot i64[n]
  %s0 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 0
  store ptr %a0, ptr %s0, align 8
  %a1 = call ptr @malloc(i64 %n4)               ; 1 bucketof i32[n]
  %s1 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 1
  store ptr %a1, ptr %s1, align 8
  %a2 = call ptr @malloc(i64 %n4)               ; 2 bkeys i32[n]
  %s2 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 2
  store ptr %a2, ptr %s2, align 8
  %a3 = call ptr @malloc(i64 %n4)               ; 3 keyslot i32[n]
  %s3 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 3
  store ptr %a3, ptr %s3, align 8
  %a4 = call ptr @malloc(i64 %n8)               ; 4 stamp i64[n]
  %s4 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 4
  store ptr %a4, ptr %s4, align 8
  %a5 = call ptr @malloc(i64 %n4)               ; 5 trial i32[n]
  %s5 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 5
  store ptr %a5, ptr %s5, align 8
  %a6 = call ptr @malloc(i64 %r4)               ; 6 bsize i32[r]
  %s6 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 6
  store ptr %a6, ptr %s6, align 8
  %a7 = call ptr @malloc(i64 %r4)               ; 7 bstart i32[r]
  %s7 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 7
  store ptr %a7, ptr %s7, align 8
  %a8 = call ptr @malloc(i64 %r4)               ; 8 order i32[r]
  %s8 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 8
  store ptr %a8, ptr %s8, align 8
  %a9 = call ptr @malloc(i64 %r4)               ; 9 gtmp i32[r]
  %s9 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 9
  store ptr %a9, ptr %s9, align 8
  %a10 = call ptr @malloc(i64 %np14)            ; 10 cnt i32[n+1]
  %s10 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 10
  store ptr %a10, ptr %s10, align 8
  %a11 = call ptr @malloc(i64 %np14)            ; 11 pos i32[n+1]
  %s11 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 11
  store ptr %a11, ptr %s11, align 8
  %a12 = call ptr @malloc(i64 %nw8)             ; 12 occ i64[nw]
  %s12 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 12
  store ptr %a12, ptr %s12, align 8
  %a13 = call ptr @malloc(i64 %r4)              ; 13 cursor i32[r]
  %s13 = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 13
  store ptr %a13, ptr %s13, align 8
  br label %chk.head

chk.head:
  %ci = phi i64 [ 0, %alloc.wk ], [ %ci.next, %chk.next ]
  %cdone = icmp eq i64 %ci, 14
  br i1 %cdone, label %alloc.ok, label %chk.body

chk.body:
  %cwp = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 %ci
  %cp = load ptr, ptr %cwp, align 8
  %cnull = icmp eq ptr %cp, null
  br i1 %cnull, label %finish, label %chk.next, !prof !0

chk.next:
  %ci.next = add nuw i64 %ci, 1
  br label %chk.head

alloc.ok:
  %hslot = load ptr, ptr %s0, align 8
  %bucketof = load ptr, ptr %s1, align 8
  %bkeys = load ptr, ptr %s2, align 8
  %keyslot = load ptr, ptr %s3, align 8
  %stamp = load ptr, ptr %s4, align 8
  %trial = load ptr, ptr %s5, align 8
  %bsize = load ptr, ptr %s6, align 8
  %bstart = load ptr, ptr %s7, align 8
  %order = load ptr, ptr %s8, align 8
  %gtmp = load ptr, ptr %s9, align 8
  %cnt = load ptr, ptr %s10, align 8
  %pos = load ptr, ptr %s11, align 8
  %occ = load ptr, ptr %s12, align 8
  %cursor = load ptr, ptr %s13, align 8
  call void @llvm.memset.p0.i64(ptr %stamp, i8 0, i64 %n8, i1 false)
  store i64 0, ptr %gen.slot, align 8
  br label %attempt.head

; ---- per-attempt construction ----------------------------------------------
attempt.head:
  %att = phi i64 [ 0, %alloc.ok ], [ %att.next, %attempt.fail ]
  %attmul = mul i64 %att, 1442695040888963407
  %S = add i64 2611923443488327891, %attmul
  call void @llvm.memset.p0.i64(ptr %occ, i8 0, i64 %nw8, i1 false)
  call void @llvm.memset.p0.i64(ptr %gtmp, i8 0, i64 %r4, i1 false)
  call void @llvm.memset.p0.i64(ptr %bsize, i8 0, i64 %r4, i1 false)
  store i64 0, ptr %fc.slot, align 8
  br label %hash.head

hash.head:
  %hi = phi i64 [ 0, %attempt.head ], [ %hi.next, %hash.cont ]
  %hidone = icmp eq i64 %hi, %n
  br i1 %hidone, label %prefix.init, label %hash.body

hash.body:
  %hkp = getelementptr inbounds ptr, ptr %keys, i64 %hi
  %hkey = load ptr, ptr %hkp, align 8
  %hlp = getelementptr inbounds i64, ptr %keylens, i64 %hi
  %hlen = load i64, ptr %hlp, align 8
  %hh = call i64 @mph_strhash(ptr %hkey, i64 %hlen, i64 %S)
  ; slot-hash = the full key hash; bucket uses its low bits (mask), the level-2
  ; fastrange uses its high bits, and fin(d) perturbs all bits -> decorrelated.
  %hslp = getelementptr inbounds i64, ptr %hslot, i64 %hi
  store i64 %hh, ptr %hslp, align 8
  %hb = and i64 %hh, %rmask
  %hb32 = trunc i64 %hb to i32
  %hbop = getelementptr inbounds i32, ptr %bucketof, i64 %hi
  store i32 %hb32, ptr %hbop, align 4
  %hbsp = getelementptr inbounds i32, ptr %bsize, i64 %hb
  %hbs = load i32, ptr %hbsp, align 4
  %hbs.n = add i32 %hbs, 1
  store i32 %hbs.n, ptr %hbsp, align 4
  br label %hash.cont

hash.cont:
  %hi.next = add nuw i64 %hi, 1
  br label %hash.head

prefix.init:
  br label %pf.head

pf.head:
  %pi = phi i64 [ 0, %prefix.init ], [ %pi.next, %pf.cont ]
  %pacc = phi i64 [ 0, %prefix.init ], [ %pacc.next, %pf.cont ]
  %pmax = phi i32 [ 0, %prefix.init ], [ %pmax.next, %pf.cont ]
  %pdone = icmp eq i64 %pi, %r
  br i1 %pdone, label %fillbk.init, label %pf.body

pf.body:
  %pbsp = getelementptr inbounds i32, ptr %bsize, i64 %pi
  %pbs = load i32, ptr %pbsp, align 4
  %pacc32 = trunc i64 %pacc to i32
  %pstp = getelementptr inbounds i32, ptr %bstart, i64 %pi
  store i32 %pacc32, ptr %pstp, align 4
  %pbs64 = zext i32 %pbs to i64
  %pacc.next = add i64 %pacc, %pbs64
  %pgt = icmp ugt i32 %pbs, %pmax
  %pmax.next = select i1 %pgt, i32 %pbs, i32 %pmax
  br label %pf.cont

pf.cont:
  %pi.next = add nuw i64 %pi, 1
  br label %pf.head

fillbk.init:
  call void @llvm.memcpy.p0.p0.i64(ptr %cursor, ptr %bstart, i64 %r4, i1 false)
  br label %fb.head

fb.head:
  %fbi = phi i64 [ 0, %fillbk.init ], [ %fbi.next, %fb.cont ]
  %fbdone = icmp eq i64 %fbi, %n
  br i1 %fbdone, label %csort.init, label %fb.body

fb.body:
  %fbbp = getelementptr inbounds i32, ptr %bucketof, i64 %fbi
  %fbb32 = load i32, ptr %fbbp, align 4
  %fbb = zext i32 %fbb32 to i64
  %fbcp = getelementptr inbounds i32, ptr %cursor, i64 %fbb
  %fbpos32 = load i32, ptr %fbcp, align 4
  %fbpos = zext i32 %fbpos32 to i64
  %fbkp = getelementptr inbounds i32, ptr %bkeys, i64 %fbpos
  %fbi32 = trunc i64 %fbi to i32
  store i32 %fbi32, ptr %fbkp, align 4
  %fbpos.n = add i32 %fbpos32, 1
  store i32 %fbpos.n, ptr %fbcp, align 4
  br label %fb.cont

fb.cont:
  %fbi.next = add nuw i64 %fbi, 1
  br label %fb.head

csort.init:
  %maxsz64 = zext i32 %pmax to i64
  %maxp1 = add nuw i64 %maxsz64, 1
  %cntbytes = shl nuw i64 %maxp1, 2
  call void @llvm.memset.p0.i64(ptr %cnt, i8 0, i64 %cntbytes, i1 false)
  br label %cc.head

cc.head:
  %cci = phi i64 [ 0, %csort.init ], [ %cci.next, %cc.cont ]
  %ccdone = icmp eq i64 %cci, %r
  br i1 %ccdone, label %cp.init, label %cc.body

cc.body:
  %ccsp = getelementptr inbounds i32, ptr %bsize, i64 %cci
  %ccs32 = load i32, ptr %ccsp, align 4
  %ccs = zext i32 %ccs32 to i64
  %cccp = getelementptr inbounds i32, ptr %cnt, i64 %ccs
  %ccc = load i32, ptr %cccp, align 4
  %ccc.n = add i32 %ccc, 1
  store i32 %ccc.n, ptr %cccp, align 4
  br label %cc.cont

cc.cont:
  %cci.next = add nuw i64 %cci, 1
  br label %cc.head

cp.init:
  %posmaxp = getelementptr inbounds i32, ptr %pos, i64 %maxsz64
  store i32 0, ptr %posmaxp, align 4
  %s0d = sub i64 %maxsz64, 1
  br label %cp.head

cp.head:
  %cs = phi i64 [ %s0d, %cp.init ], [ %cs.next, %cp.cont ]
  %csneg = icmp slt i64 %cs, 0
  br i1 %csneg, label %ord.init, label %cp.body

cp.body:
  %csp1 = add i64 %cs, 1
  %pp1 = getelementptr inbounds i32, ptr %pos, i64 %csp1
  %pv1 = load i32, ptr %pp1, align 4
  %cp1 = getelementptr inbounds i32, ptr %cnt, i64 %csp1
  %cv1 = load i32, ptr %cp1, align 4
  %pssum = add i32 %pv1, %cv1
  %psp = getelementptr inbounds i32, ptr %pos, i64 %cs
  store i32 %pssum, ptr %psp, align 4
  br label %cp.cont

cp.cont:
  %cs.next = sub i64 %cs, 1
  br label %cp.head

ord.init:
  br label %ord.head

ord.head:
  %oi = phi i64 [ 0, %ord.init ], [ %oi.next, %ord.cont ]
  %odone = icmp eq i64 %oi, %r
  br i1 %odone, label %proc.loop, label %ord.body

ord.body:
  %osp = getelementptr inbounds i32, ptr %bsize, i64 %oi
  %os32 = load i32, ptr %osp, align 4
  %os = zext i32 %os32 to i64
  %opp = getelementptr inbounds i32, ptr %pos, i64 %os
  %oidx32 = load i32, ptr %opp, align 4
  %oidx = zext i32 %oidx32 to i64
  %oop = getelementptr inbounds i32, ptr %order, i64 %oidx
  %oi32 = trunc i64 %oi to i32
  store i32 %oi32, ptr %oop, align 4
  %oidx.n = add i32 %oidx32, 1
  store i32 %oidx.n, ptr %opp, align 4
  br label %ord.cont

ord.cont:
  %oi.next = add nuw i64 %oi, 1
  br label %ord.head

; ---- place buckets, largest first ------------------------------------------
proc.loop:
  %k = phi i64 [ 0, %ord.head ], [ %k.next, %proc.cont ]
  %kdone = icmp eq i64 %k, %r
  br i1 %kdone, label %build_final, label %proc.body

proc.body:
  %op = getelementptr inbounds i32, ptr %order, i64 %k
  %b32 = load i32, ptr %op, align 4
  %b = zext i32 %b32 to i64
  %szp = getelementptr inbounds i32, ptr %bsize, i64 %b
  %sz32 = load i32, ptr %szp, align 4
  %sz = zext i32 %sz32 to i64
  %iszero = icmp eq i64 %sz, 0
  br i1 %iszero, label %proc.cont, label %proc.nonzero

proc.nonzero:
  %stp = getelementptr inbounds i32, ptr %bstart, i64 %b
  %start32 = load i32, ptr %stp, align 4
  %start = zext i32 %start32 to i64
  %issingle = icmp eq i64 %sz, 1
  br i1 %issingle, label %single, label %multi.enter

single:
  %skip = getelementptr inbounds i32, ptr %bkeys, i64 %start
  %ski32 = load i32, ptr %skip, align 4
  %ski = zext i32 %ski32 to i64
  br label %fc.head

fc.head:
  %fc = load i64, ptr %fc.slot, align 8
  %fwordidx = lshr i64 %fc, 6
  %fwp = getelementptr inbounds i64, ptr %occ, i64 %fwordidx
  %fword = load i64, ptr %fwp, align 8
  %fbitpos = and i64 %fc, 63
  %fbit = lshr i64 %fword, %fbitpos
  %focc = and i64 %fbit, 1
  %fisocc = icmp ne i64 %focc, 0
  br i1 %fisocc, label %fc.adv, label %fc.found

fc.adv:
  %fc.n = add nuw i64 %fc, 1
  store i64 %fc.n, ptr %fc.slot, align 8
  br label %fc.head

fc.found:
  %fmaskbit = shl nuw i64 1, %fbitpos
  %fword.n = or i64 %fword, %fmaskbit
  store i64 %fword.n, ptr %fwp, align 8
  %fc.after = add nuw i64 %fc, 1
  store i64 %fc.after, ptr %fc.slot, align 8
  %sslot32 = trunc i64 %fc to i32
  %sksp = getelementptr inbounds i32, ptr %keyslot, i64 %ski
  store i32 %sslot32, ptr %sksp, align 4
  %sneg = sub i32 -1, %sslot32                  ; encode -1 - slot
  %sgbp = getelementptr inbounds i32, ptr %gtmp, i64 %b
  store i32 %sneg, ptr %sgbp, align 4
  br label %proc.cont

multi.enter:
  br label %dsearch.head

dsearch.head:
  %d = phi i64 [ 1, %multi.enter ], [ %d.next, %d.cont ]
  %dmix = call i64 @mph_fin(i64 %d)
  %gen.cur0 = load i64, ptr %gen.slot, align 8
  %gen.cur = add i64 %gen.cur0, 1
  store i64 %gen.cur, ptr %gen.slot, align 8
  br label %in.head

in.head:
  %j = phi i64 [ 0, %dsearch.head ], [ %j.next, %in.cont ]
  %jend = icmp eq i64 %j, %sz
  br i1 %jend, label %commit, label %in.body

in.body:
  %bkidx = add i64 %start, %j
  %kip = getelementptr inbounds i32, ptr %bkeys, i64 %bkidx
  %ki32 = load i32, ptr %kip, align 4
  %ki = zext i32 %ki32 to i64
  %hsp = getelementptr inbounds i64, ptr %hslot, i64 %ki
  %hs = load i64, ptr %hsp, align 8
  %xr = xor i64 %hs, %dmix
  ; fastrange: slot = (xr * n) >> 64  -> maps a 64-bit value into [0,n) with no
  ; division (lowers to umulh on AArch64, mul+shift on AMD64).
  %xr128 = zext i64 %xr to i128
  %n128 = zext i64 %n to i128
  %prod = mul i128 %xr128, %n128
  %prodhi = lshr i128 %prod, 64
  %slot = trunc i128 %prodhi to i64
  %wordidx = lshr i64 %slot, 6
  %wp = getelementptr inbounds i64, ptr %occ, i64 %wordidx
  %word = load i64, ptr %wp, align 8
  %bitpos = and i64 %slot, 63
  %bit = lshr i64 %word, %bitpos
  %occd = and i64 %bit, 1
  %isocc = icmp ne i64 %occd, 0
  br i1 %isocc, label %d.fail, label %in.stampchk

in.stampchk:
  %sp = getelementptr inbounds i64, ptr %stamp, i64 %slot
  %st = load i64, ptr %sp, align 8
  %dup = icmp eq i64 %st, %gen.cur
  br i1 %dup, label %d.fail, label %in.mark

in.mark:
  store i64 %gen.cur, ptr %sp, align 8
  %tp = getelementptr inbounds i32, ptr %trial, i64 %j
  %slot32 = trunc i64 %slot to i32
  store i32 %slot32, ptr %tp, align 4
  br label %in.cont

in.cont:
  %j.next = add nuw i64 %j, 1
  br label %in.head

d.fail:
  %d.next = add nuw i64 %d, 1
  %dover = icmp ugt i64 %d.next, 100000         ; MAXD displacement budget
  br i1 %dover, label %attempt.fail, label %d.cont, !prof !0

d.cont:
  br label %dsearch.head

commit:
  br label %commit.loop

commit.loop:
  %cj = phi i64 [ 0, %commit ], [ %cj.next, %commit.body2 ]
  %cjend = icmp eq i64 %cj, %sz
  br i1 %cjend, label %commit.done, label %commit.body

commit.body:
  %ctp = getelementptr inbounds i32, ptr %trial, i64 %cj
  %cslot32 = load i32, ptr %ctp, align 4
  %cslot = zext i32 %cslot32 to i64
  %cwordidx = lshr i64 %cslot, 6
  %cwordp = getelementptr inbounds i64, ptr %occ, i64 %cwordidx
  %cword = load i64, ptr %cwordp, align 8
  %cbitpos = and i64 %cslot, 63
  %cmaskbit = shl nuw i64 1, %cbitpos
  %cword.n = or i64 %cword, %cmaskbit
  store i64 %cword.n, ptr %cwordp, align 8
  %cbkidx = add i64 %start, %cj
  %ckip = getelementptr inbounds i32, ptr %bkeys, i64 %cbkidx
  %cki32 = load i32, ptr %ckip, align 4
  %cki = zext i32 %cki32 to i64
  %ckslotp = getelementptr inbounds i32, ptr %keyslot, i64 %cki
  store i32 %cslot32, ptr %ckslotp, align 4
  br label %commit.body2

commit.body2:
  %cj.next = add nuw i64 %cj, 1
  br label %commit.loop

commit.done:
  %gd32 = trunc i64 %d to i32
  %gbp = getelementptr inbounds i32, ptr %gtmp, i64 %b
  store i32 %gd32, ptr %gbp, align 4
  br label %proc.cont

proc.cont:
  %k.next = add nuw i64 %k, 1
  br label %proc.loop

attempt.fail:
  %att.next = add nuw i64 %att, 1
  %att.more = icmp ult i64 %att.next, 64        ; MAX_ATTEMPT
  br i1 %att.more, label %attempt.head, label %attfail_all, !prof !1

attfail_all:
  br label %finish

; ---- success: materialize the immutable structure --------------------------
build_final:
  %f.hdrg = add i64 64, %r4
  %f.descoff.r = add i64 %f.hdrg, 7
  %f.descoff = and i64 %f.descoff.r, -8
  %f.dsz = shl nuw i64 %n, 3
  %f.bloboff = add i64 %f.descoff, %f.dsz
  %f.total = add i64 %f.bloboff, %btrun
  %struct = call ptr @malloc(i64 %f.total)
  %struct.null = icmp eq ptr %struct, null
  br i1 %struct.null, label %finish, label %fill.hdr, !prof !0

fill.hdr:
  store i64 %n, ptr %struct, align 8
  %f.rp = getelementptr inbounds nuw i8, ptr %struct, i64 8
  store i64 %r, ptr %f.rp, align 8
  %f.rmp = getelementptr inbounds nuw i8, ptr %struct, i64 16
  store i64 %rmask, ptr %f.rmp, align 8
  %f.sp = getelementptr inbounds nuw i8, ptr %struct, i64 24
  store i64 %S, ptr %f.sp, align 8
  %f.gop = getelementptr inbounds nuw i8, ptr %struct, i64 32
  store i64 64, ptr %f.gop, align 8
  %f.dop = getelementptr inbounds nuw i8, ptr %struct, i64 40
  store i64 %f.descoff, ptr %f.dop, align 8
  %f.bop = getelementptr inbounds nuw i8, ptr %struct, i64 48
  store i64 %f.bloboff, ptr %f.bop, align 8
  %f.spp = getelementptr inbounds nuw i8, ptr %struct, i64 56
  store i64 0, ptr %f.spp, align 8
  %f.gdst = getelementptr inbounds nuw i8, ptr %struct, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %f.gdst, ptr %gtmp, i64 %r4, i1 false)
  %f.desc = getelementptr inbounds nuw i8, ptr %struct, i64 %f.descoff
  %f.blob = getelementptr inbounds nuw i8, ptr %struct, i64 %f.bloboff
  br label %bf.head

bf.head:
  %bfi = phi i64 [ 0, %fill.hdr ], [ %bfi.next, %bf.cont ]
  %bfrun = phi i64 [ 0, %fill.hdr ], [ %bfrun.next, %bf.cont ]
  %bfdone = icmp eq i64 %bfi, %n
  br i1 %bfdone, label %success, label %bf.body

bf.body:
  %bfkp = getelementptr inbounds ptr, ptr %keys, i64 %bfi
  %bfkey = load ptr, ptr %bfkp, align 8
  %bflp = getelementptr inbounds i64, ptr %keylens, i64 %bfi
  %bflen = load i64, ptr %bflp, align 8
  %bfdst = getelementptr inbounds i8, ptr %f.blob, i64 %bfrun
  call void @llvm.memcpy.p0.p0.i64(ptr %bfdst, ptr %bfkey, i64 %bflen, i1 false)
  %bfslp = getelementptr inbounds i32, ptr %keyslot, i64 %bfi
  %bfslot32 = load i32, ptr %bfslp, align 4
  %bfslot = zext i32 %bfslot32 to i64
  %bfsl8 = shl nuw i64 %bfslot, 3
  %bfde = getelementptr inbounds i8, ptr %f.desc, i64 %bfsl8
  %bfrun32 = trunc i64 %bfrun to i32
  store i32 %bfrun32, ptr %bfde, align 4
  %bfde4 = getelementptr inbounds i8, ptr %bfde, i64 4
  %bflen32 = trunc i64 %bflen to i32
  store i32 %bflen32, ptr %bfde4, align 4
  %bfrun.next = add i64 %bfrun, %bflen
  br label %bf.cont

bf.cont:
  %bfi.next = add nuw i64 %bfi, 1
  br label %bf.head

success:
  br label %finish

; ---- shared exit: free scratch, return result ------------------------------
finish:
  %result = phi ptr [ null, %chk.body ], [ null, %attfail_all ], [ null, %build_final ], [ %struct, %success ]
  br label %free.loop

free.loop:
  %fi = phi i64 [ 0, %finish ], [ %fi.next, %free.next ]
  %fdone = icmp eq i64 %fi, 14
  br i1 %fdone, label %ret.blk, label %free.body

free.body:
  %fwpp = getelementptr inbounds nuw [14 x ptr], ptr %wk, i64 0, i64 %fi
  %fp = load ptr, ptr %fwpp, align 8
  %fpn = icmp eq ptr %fp, null
  br i1 %fpn, label %free.next, label %do.free

do.free:
  call void @free(ptr %fp)
  br label %free.next

free.next:
  %fi.next = add nuw i64 %fi, 1
  br label %free.loop

ret.blk:
  ret ptr %result
}

; ===========================================================================
; lookup — perfect hash; returns a slot in [0,n) for any key (member or not)
; ===========================================================================
define i64 @universe_ds_mph_lookup(ptr %m, ptr %key, i64 %len) local_unnamed_addr #2 {
entry:
  %n = load i64, ptr %m, align 8
  %rmask.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %rmask = load i64, ptr %rmask.p, align 8
  %seed.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %seed = load i64, ptr %seed.p, align 8
  %goff.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %goff = load i64, ptr %goff.p, align 8
  %G = getelementptr inbounds nuw i8, ptr %m, i64 %goff
  %h = call i64 @mph_strhash(ptr %key, i64 %len, i64 %seed)
  %b = and i64 %h, %rmask
  %gp = getelementptr inbounds i32, ptr %G, i64 %b
  %g = load i32, ptr %gp, align 4
  %isdirect = icmp slt i32 %g, 0
  br i1 %isdirect, label %direct, label %seedpath

direct:
  %ds = sub i32 -1, %g                          ; decode slot = -1 - g
  %ds64 = zext i32 %ds to i64
  ret i64 %ds64

seedpath:
  %gd = zext i32 %g to i64
  %dmix = call i64 @mph_fin(i64 %gd)
  %xr = xor i64 %h, %dmix
  ; fastrange (see build): slot = (xr * n) >> 64, no division.
  %xr128 = zext i64 %xr to i128
  %n128 = zext i64 %n to i128
  %prod = mul i128 %xr128, %n128
  %prodhi = lshr i128 %prod, 64
  %slot = trunc i128 %prodhi to i64
  ret i64 %slot
}

; ===========================================================================
; lookup_checked — disambiguate members from non-members via stored key
; ===========================================================================
define i64 @universe_ds_mph_lookup_checked(ptr %m, ptr %key, i64 %len) local_unnamed_addr #2 {
entry:
  %slot = call i64 @universe_ds_mph_lookup(ptr %m, ptr %key, i64 %len)
  %descoff.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %descoff = load i64, ptr %descoff.p, align 8
  %bloboff.p = getelementptr inbounds nuw i8, ptr %m, i64 48
  %bloboff = load i64, ptr %bloboff.p, align 8
  %desc = getelementptr inbounds nuw i8, ptr %m, i64 %descoff
  %sl8 = shl nuw i64 %slot, 3
  %dentry = getelementptr inbounds i8, ptr %desc, i64 %sl8
  %off = load i32, ptr %dentry, align 4
  %dlen.p = getelementptr inbounds i8, ptr %dentry, i64 4
  %dlen = load i32, ptr %dlen.p, align 4
  %dlen64 = zext i32 %dlen to i64
  %lenmatch = icmp eq i64 %dlen64, %len
  br i1 %lenmatch, label %cmp, label %notfound

cmp:
  %blob = getelementptr inbounds nuw i8, ptr %m, i64 %bloboff
  %off64 = zext i32 %off to i64
  %kp = getelementptr inbounds i8, ptr %blob, i64 %off64
  %eq = call i1 @mph_bytes_eq(ptr %kp, ptr %key, i64 %len)
  br i1 %eq, label %hit, label %notfound

hit:
  ret i64 %slot

notfound:
  ret i64 -1
}

define i64 @universe_ds_mph_slot_count(ptr %m) local_unnamed_addr #2 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %zero, label %load

load:
  %n = load i64, ptr %m, align 8
  ret i64 %n

zero:
  ret i64 0
}

define void @universe_ds_mph_destroy(ptr %m) local_unnamed_addr #1 {
entry:
  %isnull = icmp eq ptr %m, null
  br i1 %isnull, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %m)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(none) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 2000}
