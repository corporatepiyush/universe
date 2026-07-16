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

; Suffix array + LCP of a byte string, with substring find/count.
;
; DESIGN (algorithm class + layout):
;   * Suffix array by PREFIX DOUBLING with radix (counting) sort — O(N log N).
;     Each round sorts suffixes by the pair (rank[i], rank[i+k]); a stable
;     two-pass LSD counting sort over N+1 buckets does the pair sort in O(N),
;     so log N rounds give O(N log N). Ranks in [0,N) are used directly as
;     radix digits (base N) — no comparisons, the right class for this job
;     (comparison sorting suffixes is O(N log^2 N) with big constants). We stop
;     early the round all ranks become distinct.
;   * LCP by KASAI — O(N) using the inverse permutation (rank), reusing the
;     h-1 carry between adjacent text positions.
;   * find/count are BINARY SEARCH over the sorted suffixes: O(M + log N) find,
;     two lower-bound searches for the occurrence RANGE in count. The compare
;     is a tight byte loop against the pattern.
;   * ONE allocation for the persistent handle: 32-byte header + sa[N] + lcp[N]
;     as flat i32 arrays (indices, never pointers). Prefix-doubling scratch
;     (rank, tmp, sorted, keys, inverse, counts) lives in a SINGLE temp block
;     freed at the end of build. The text buffer is BORROWED (caller owns it;
;     it must outlive the handle) — find/count also take text explicitly.
;
;   Header (bytes): [0] ptr text  [8] i64 len(N)  [16] ptr sa  [24] ptr lcp.
;
; API (build returns null on bad args / OOM; find/count/at return -1 on miss):
;   ptr universe_search_sa_build(ptr text, i64 len)
;   i64 universe_search_sa_find(ptr sa, ptr text, ptr pattern, i64 plen)
;       -> a starting offset of an occurrence, or -1 if absent.
;   i64 universe_search_sa_count(ptr sa, ptr text, ptr pattern, i64 plen)
;   i64 universe_search_sa_at(ptr sa, i64 i)       -> sa[i], or -1 if OOB
;   i64 universe_search_sa_lcp_at(ptr sa, i64 i)   -> lcp[i], or -1 if OOB
;   i64 universe_search_sa_len(ptr sa)             -> N
;   void universe_search_sa_destroy(ptr sa)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; rank2(idx) = (idx+k < N) ? rank[idx+k]+1 : 0  (clamped index, OOB-safe)
define internal i64 @sa_r2(ptr %rank, i64 %N, i64 %k, i64 %idx) #4 {
entry:
  %jk = add i64 %idx, %k
  %in = icmp ult i64 %jk, %N
  %safe = select i1 %in, i64 %jk, i64 0
  %rp = getelementptr inbounds i32, ptr %rank, i64 %safe
  %rv = load i32, ptr %rp, align 4
  %rvz = zext i32 %rv to i64
  %r1 = add nuw i64 %rvz, 1
  %key = select i1 %in, i64 %r1, i64 0
  ret i64 %key
}

; stable counting sort of arr[0..N) -> out, key(e) = keyarr[e], nb buckets.
define internal void @sa_csort(ptr %arr, ptr %out, ptr %keyarr, ptr %cnt, i64 %N, i64 %nb) #5 {
entry:
  %nbb = shl i64 %nb, 2
  call void @llvm.memset.p0.i64(ptr %cnt, i8 0, i64 %nbb, i1 false)
  br label %count.head

count.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %count.head ]
  %ap = getelementptr inbounds i32, ptr %arr, i64 %i
  %e = load i32, ptr %ap, align 4
  %ez = zext i32 %e to i64
  %kp = getelementptr inbounds i32, ptr %keyarr, i64 %ez
  %key = load i32, ptr %kp, align 4
  %kz = zext i32 %key to i64
  %cp = getelementptr inbounds i32, ptr %cnt, i64 %kz
  %cv = load i32, ptr %cp, align 4
  %cv.n = add i32 %cv, 1
  store i32 %cv.n, ptr %cp, align 4
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %N
  br i1 %more, label %count.head, label %pref.head

pref.head:                                          ; exclusive prefix sum
  %b = phi i64 [ 0, %count.head ], [ %b.n, %pref.head ]
  %acc = phi i32 [ 0, %count.head ], [ %acc.n, %pref.head ]
  %pcp = getelementptr inbounds i32, ptr %cnt, i64 %b
  %t = load i32, ptr %pcp, align 4
  store i32 %acc, ptr %pcp, align 4
  %acc.n = add i32 %acc, %t
  %b.n = add nuw i64 %b, 1
  %bmore = icmp ult i64 %b.n, %nb
  br i1 %bmore, label %pref.head, label %scat.head

scat.head:
  %j = phi i64 [ 0, %pref.head ], [ %j.n, %scat.head ]
  %ap2 = getelementptr inbounds i32, ptr %arr, i64 %j
  %e2 = load i32, ptr %ap2, align 4
  %e2z = zext i32 %e2 to i64
  %kp2 = getelementptr inbounds i32, ptr %keyarr, i64 %e2z
  %key2 = load i32, ptr %kp2, align 4
  %k2z = zext i32 %key2 to i64
  %cp2 = getelementptr inbounds i32, ptr %cnt, i64 %k2z
  %p = load i32, ptr %cp2, align 4
  %pz = zext i32 %p to i64
  %op = getelementptr inbounds i32, ptr %out, i64 %pz
  store i32 %e2, ptr %op, align 4
  %p.n = add i32 %p, 1
  store i32 %p.n, ptr %cp2, align 4
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, %N
  br i1 %jmore, label %scat.head, label %done

done:
  ret void
}

; compare suffix starting at %s vs pattern[0..plen): -1 lt, 0 pattern is prefix, 1 gt
define internal i32 @sa_cmp(ptr %text, i64 %N, i64 %s, ptr %pattern, i64 %plen) #4 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %eqc ]
  %done = icmp uge i64 %i, %plen
  br i1 %done, label %ret0, label %body

body:
  %si = add i64 %s, %i
  %short = icmp uge i64 %si, %N
  br i1 %short, label %retlt, label %cmp

cmp:
  %tp = getelementptr inbounds i8, ptr %text, i64 %si
  %ta = load i8, ptr %tp, align 1
  %taz = zext i8 %ta to i32
  %pp = getelementptr inbounds i8, ptr %pattern, i64 %i
  %pb = load i8, ptr %pp, align 1
  %pbz = zext i8 %pb to i32
  %lt = icmp ult i32 %taz, %pbz
  br i1 %lt, label %retlt, label %gtc

gtc:
  %gt = icmp ugt i32 %taz, %pbz
  br i1 %gt, label %retgt, label %eqc

eqc:
  %i.n = add nuw i64 %i, 1
  br label %loop

ret0:
  ret i32 0

retlt:
  ret i32 -1

retgt:
  ret i32 1
}

; lower bound: first SA index whose suffix compares (upper? >0 : >=0) to pattern
define internal i64 @sa_bound(ptr %sap, ptr %text, i64 %N, ptr %pattern, i64 %plen, i1 %upper) #6 {
entry:
  br label %loop

loop:
  %lo = phi i64 [ 0, %entry ], [ %lo.n, %step ]
  %hi = phi i64 [ %N, %entry ], [ %hi.n, %step ]
  %go = icmp ult i64 %lo, %hi
  br i1 %go, label %step, label %done

step:
  %sum = add i64 %lo, %hi
  %mid = lshr i64 %sum, 1
  %mp = getelementptr inbounds i32, ptr %sap, i64 %mid
  %sv = load i32, ptr %mp, align 4
  %s = zext i32 %sv to i64
  %c = call i32 @sa_cmp(ptr %text, i64 %N, i64 %s, ptr %pattern, i64 %plen)
  %lt = icmp slt i32 %c, 0
  %le = icmp sle i32 %c, 0
  %right = select i1 %upper, i1 %le, i1 %lt
  %mid1 = add i64 %mid, 1
  %lo.n = select i1 %right, i64 %mid1, i64 %lo
  %hi.n = select i1 %right, i64 %hi, i64 %mid
  br label %loop

done:
  ret i64 %lo
}

; ---------------------------------------------------------------------------
define ptr @universe_search_sa_build(ptr %text, i64 %len) local_unnamed_addr #1 {
entry:
  %hist = alloca [256 x i32], align 4
  %tnull = icmp eq ptr %text, null
  %haslen = icmp ne i64 %len, 0
  %bad = and i1 %tnull, %haslen
  br i1 %bad, label %fail, label %sizes, !prof !0

fail:
  ret ptr null

sizes:
  %n8 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %len, i64 8)
  %n8.v = extractvalue { i64, i1 } %n8, 0
  %n8.o = extractvalue { i64, i1 } %n8, 1
  br i1 %n8.o, label %fail, label %alloc, !prof !0

alloc:
  %bsz = add nuw i64 %n8.v, 32
  %base = call ptr @malloc(i64 %bsz)
  %base.null = icmp eq ptr %base, null
  br i1 %base.null, label %fail, label %hdr, !prof !0

hdr:
  %n4 = shl i64 %len, 2
  %sap = getelementptr inbounds nuw i8, ptr %base, i64 32
  %lcpp = getelementptr inbounds nuw i8, ptr %sap, i64 %n4
  store ptr %text, ptr %base, align 8
  %h.len = getelementptr inbounds nuw i8, ptr %base, i64 8
  store i64 %len, ptr %h.len, align 8
  %h.sa = getelementptr inbounds nuw i8, ptr %base, i64 16
  store ptr %sap, ptr %h.sa, align 8
  %h.lcp = getelementptr inbounds nuw i8, ptr %base, i64 24
  store ptr %lcpp, ptr %h.lcp, align 8
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %ret.base, label %tmp.alloc, !prof !0

ret.base:
  ret ptr %base

tmp.alloc:                                          ; scratch: 6N+1 i32
  %six = mul nuw i64 %len, 6
  %telem = add nuw i64 %six, 1
  %tbytes = shl i64 %telem, 2
  %tmpb = call ptr @malloc(i64 %tbytes)
  %tmpb.null = icmp eq ptr %tmpb, null
  br i1 %tmpb.null, label %free.base, label %init, !prof !0

free.base:
  call void @free(ptr %base)
  ret ptr null

init:
  %rank = getelementptr inbounds nuw i32, ptr %tmpb, i64 0
  %tmpa = getelementptr inbounds nuw i32, ptr %tmpb, i64 %len
  %two.n = shl i64 %len, 1
  %sa2 = getelementptr inbounds nuw i32, ptr %tmpb, i64 %two.n
  %three.n = mul nuw i64 %len, 3
  %key2 = getelementptr inbounds nuw i32, ptr %tmpb, i64 %three.n
  %four.n = shl i64 %len, 2
  %pos = getelementptr inbounds nuw i32, ptr %tmpb, i64 %four.n
  %five.n = mul nuw i64 %len, 5
  %cnt = getelementptr inbounds nuw i32, ptr %tmpb, i64 %five.n
  ; normalize raw bytes -> dense ranks in [0,distinct) so the first radix
  ; round's buckets (N / N+1) suffice even when N < 256.
  call void @llvm.memset.p0.i64(ptr %hist, i8 0, i64 1024, i1 false)
  br label %hist.mark

hist.mark:                                          ; mark present bytes
  %hi = phi i64 [ 0, %init ], [ %hi.n, %hist.mark ]
  %hcp = getelementptr inbounds i8, ptr %text, i64 %hi
  %hb = load i8, ptr %hcp, align 1
  %hbz = zext i8 %hb to i64
  %hbp = getelementptr inbounds [256 x i32], ptr %hist, i64 0, i64 %hbz
  store i32 1, ptr %hbp, align 4
  %hi.n = add nuw i64 %hi, 1
  %himore = icmp ult i64 %hi.n, %len
  br i1 %himore, label %hist.mark, label %hist.pref

hist.pref:                                          ; exclusive prefix sum -> dense rank map
  %hpb = phi i64 [ 0, %hist.mark ], [ %hpb.n, %hist.pref ]
  %hacc = phi i32 [ 0, %hist.mark ], [ %hacc.n, %hist.pref ]
  %hpp = getelementptr inbounds [256 x i32], ptr %hist, i64 0, i64 %hpb
  %ht = load i32, ptr %hpp, align 4
  store i32 %hacc, ptr %hpp, align 4
  %hacc.n = add i32 %hacc, %ht
  %hpb.n = add nuw i64 %hpb, 1
  %hpmore = icmp ult i64 %hpb.n, 256
  br i1 %hpmore, label %hist.pref, label %init.head

init.head:
  %ii = phi i64 [ 0, %hist.pref ], [ %ii.n, %init.head ]
  %saip = getelementptr inbounds i32, ptr %sap, i64 %ii
  %ii32 = trunc i64 %ii to i32
  store i32 %ii32, ptr %saip, align 4
  %tcp = getelementptr inbounds i8, ptr %text, i64 %ii
  %tc = load i8, ptr %tcp, align 1
  %tcz = zext i8 %tc to i64
  %tmapp = getelementptr inbounds [256 x i32], ptr %hist, i64 0, i64 %tcz
  %tdense = load i32, ptr %tmapp, align 4
  %rip = getelementptr inbounds i32, ptr %rank, i64 %ii
  store i32 %tdense, ptr %rip, align 4
  %ii.n = add nuw i64 %ii, 1
  %imore = icmp ult i64 %ii.n, %len
  br i1 %imore, label %init.head, label %pd.head

; ---- prefix doubling -------------------------------------------------------
pd.head:
  %k = phi i64 [ 1, %init.head ], [ %k2, %pd.next ]
  %kgo = icmp ult i64 %k, %len
  br i1 %kgo, label %pd.keys, label %kasai.init

pd.keys:                                            ; key2[e] = rank2(e)
  %ke = phi i64 [ 0, %pd.head ], [ %ke.n, %pd.keys ]
  %r2 = call i64 @sa_r2(ptr %rank, i64 %len, i64 %k, i64 %ke)
  %r2.32 = trunc i64 %r2 to i32
  %k2p = getelementptr inbounds i32, ptr %key2, i64 %ke
  store i32 %r2.32, ptr %k2p, align 4
  %ke.n = add nuw i64 %ke, 1
  %kemore = icmp ult i64 %ke.n, %len
  br i1 %kemore, label %pd.keys, label %pd.sort

pd.sort:
  %nb1 = add nuw i64 %len, 1
  call void @sa_csort(ptr %sap, ptr %sa2, ptr %key2, ptr %cnt, i64 %len, i64 %nb1)
  call void @sa_csort(ptr %sa2, ptr %sap, ptr %rank, ptr %cnt, i64 %len, i64 %len)
  ; recompute ranks into tmp
  %sa0p = getelementptr inbounds i32, ptr %sap, i64 0
  %sa0 = load i32, ptr %sa0p, align 4
  %sa0z = zext i32 %sa0 to i64
  %tmp0p = getelementptr inbounds i32, ptr %tmpa, i64 %sa0z
  store i32 0, ptr %tmp0p, align 4
  br label %rank.head

rank.head:
  %ri = phi i64 [ 1, %pd.sort ], [ %ri.n, %rank.head ]
  %rim1 = add i64 %ri, -1
  %sap.a = getelementptr inbounds i32, ptr %sap, i64 %rim1
  %av = load i32, ptr %sap.a, align 4
  %az = zext i32 %av to i64
  %sap.b = getelementptr inbounds i32, ptr %sap, i64 %ri
  %bv = load i32, ptr %sap.b, align 4
  %bz = zext i32 %bv to i64
  %r1a.p = getelementptr inbounds i32, ptr %rank, i64 %az
  %r1a = load i32, ptr %r1a.p, align 4
  %r1b.p = getelementptr inbounds i32, ptr %rank, i64 %bz
  %r1b = load i32, ptr %r1b.p, align 4
  %k2a.p = getelementptr inbounds i32, ptr %key2, i64 %az
  %k2a = load i32, ptr %k2a.p, align 4
  %k2b.p = getelementptr inbounds i32, ptr %key2, i64 %bz
  %k2b = load i32, ptr %k2b.p, align 4
  %e1 = icmp eq i32 %r1a, %r1b
  %e2 = icmp eq i32 %k2a, %k2b
  %same = and i1 %e1, %e2
  %d = select i1 %same, i32 0, i32 1
  %tmpa.a = getelementptr inbounds i32, ptr %tmpa, i64 %az
  %prev = load i32, ptr %tmpa.a, align 4
  %newr = add i32 %prev, %d
  %tmpa.b = getelementptr inbounds i32, ptr %tmpa, i64 %bz
  store i32 %newr, ptr %tmpa.b, align 4
  %ri.n = add nuw i64 %ri, 1
  %rimore = icmp ult i64 %ri.n, %len
  br i1 %rimore, label %rank.head, label %rank.copy

rank.copy:
  call void @llvm.memcpy.p0.p0.i64(ptr %rank, ptr %tmpa, i64 %n4, i1 false)
  ; max rank = rank[sa[N-1]]
  %nm1 = add i64 %len, -1
  %saLp = getelementptr inbounds i32, ptr %sap, i64 %nm1
  %saL = load i32, ptr %saLp, align 4
  %saLz = zext i32 %saL to i64
  %maxp = getelementptr inbounds i32, ptr %rank, i64 %saLz
  %maxr = load i32, ptr %maxp, align 4
  %nm1.32 = trunc i64 %nm1 to i32
  %distinct = icmp eq i32 %maxr, %nm1.32
  br i1 %distinct, label %kasai.init, label %pd.next

pd.next:
  %k2 = shl i64 %k, 1
  br label %pd.head

; ---- Kasai LCP -------------------------------------------------------------
kasai.init:
  ; pos[sa[i]] = i  (inverse permutation)
  br label %pos.head

pos.head:
  %pi = phi i64 [ 0, %kasai.init ], [ %pi.n, %pos.head ]
  %pos.sap = getelementptr inbounds i32, ptr %sap, i64 %pi
  %pos.sa = load i32, ptr %pos.sap, align 4
  %pos.saz = zext i32 %pos.sa to i64
  %posp = getelementptr inbounds i32, ptr %pos, i64 %pos.saz
  %pi.32 = trunc i64 %pi to i32
  store i32 %pi.32, ptr %posp, align 4
  %pi.n = add nuw i64 %pi, 1
  %pimore = icmp ult i64 %pi.n, %len
  br i1 %pimore, label %pos.head, label %lcp0

lcp0:
  store i32 0, ptr %lcpp, align 4                   ; lcp[0] = 0
  br label %kas.head

kas.head:
  %ki = phi i64 [ 0, %lcp0 ], [ %ki.n, %kas.cont ]
  %h = phi i64 [ 0, %lcp0 ], [ %h.next, %kas.cont ]
  %ki.posp = getelementptr inbounds i32, ptr %pos, i64 %ki
  %ri.rank = load i32, ptr %ki.posp, align 4
  %ri.z = zext i32 %ri.rank to i64
  %has.prev = icmp ugt i32 %ri.rank, 0
  br i1 %has.prev, label %kas.ext, label %kas.zero

kas.zero:                                           ; rank 0: no predecessor, h=0
  br label %kas.cont

kas.ext:
  %rm1 = add i64 %ri.z, -1
  %jprev.p = getelementptr inbounds i32, ptr %sap, i64 %rm1
  %jprev = load i32, ptr %jprev.p, align 4
  %j = zext i32 %jprev to i64
  br label %ext.loop

ext.loop:
  %hh = phi i64 [ %h, %kas.ext ], [ %hh.n, %ext.cont ]
  %iph = add i64 %ki, %hh
  %jph = add i64 %j, %hh
  %i.ok = icmp ult i64 %iph, %len
  %j.ok = icmp ult i64 %jph, %len
  %both = and i1 %i.ok, %j.ok
  br i1 %both, label %ext.cmp, label %ext.store

ext.cmp:
  %ci.p = getelementptr inbounds i8, ptr %text, i64 %iph
  %ci = load i8, ptr %ci.p, align 1
  %cj.p = getelementptr inbounds i8, ptr %text, i64 %jph
  %cj = load i8, ptr %cj.p, align 1
  %ceq = icmp eq i8 %ci, %cj
  br i1 %ceq, label %ext.cont, label %ext.store

ext.cont:
  %hh.n = add nuw i64 %hh, 1
  br label %ext.loop

ext.store:
  %lcp.rp = getelementptr inbounds i32, ptr %lcpp, i64 %ri.z
  %hh.32 = trunc i64 %hh to i32
  store i32 %hh.32, ptr %lcp.rp, align 4
  %hpos = icmp ugt i64 %hh, 0
  %hh.m1 = add i64 %hh, -1
  %h.dec = select i1 %hpos, i64 %hh.m1, i64 0
  br label %kas.cont

kas.cont:
  %h.next = phi i64 [ 0, %kas.zero ], [ %h.dec, %ext.store ]
  %ki.n = add nuw i64 %ki, 1
  %kimore = icmp ult i64 %ki.n, %len
  br i1 %kimore, label %kas.head, label %kas.done

kas.done:
  call void @free(ptr %tmpb)
  ret ptr %base
}

; ---------------------------------------------------------------------------
define i64 @universe_search_sa_find(ptr %sa, ptr %text, ptr %pattern, i64 %plen) local_unnamed_addr #6 {
entry:
  %sa.null = icmp eq ptr %sa, null
  br i1 %sa.null, label %miss, label %go, !prof !0

miss:
  ret i64 -1

go:
  %h.len = getelementptr inbounds nuw i8, ptr %sa, i64 8
  %N = load i64, ptr %h.len, align 8
  %h.sa = getelementptr inbounds nuw i8, ptr %sa, i64 16
  %sap = load ptr, ptr %h.sa, align 8
  %empty = icmp eq i64 %N, 0
  br i1 %empty, label %miss, label %search, !prof !0

search:
  %lo = call i64 @sa_bound(ptr %sap, ptr %text, i64 %N, ptr %pattern, i64 %plen, i1 false)
  %inb = icmp ult i64 %lo, %N
  br i1 %inb, label %check, label %miss

check:
  %mp = getelementptr inbounds i32, ptr %sap, i64 %lo
  %sv = load i32, ptr %mp, align 4
  %s = zext i32 %sv to i64
  %c = call i32 @sa_cmp(ptr %text, i64 %N, i64 %s, ptr %pattern, i64 %plen)
  %match = icmp eq i32 %c, 0
  %res = select i1 %match, i64 %s, i64 -1
  ret i64 %res
}

define i64 @universe_search_sa_count(ptr %sa, ptr %text, ptr %pattern, i64 %plen) local_unnamed_addr #6 {
entry:
  %sa.null = icmp eq ptr %sa, null
  br i1 %sa.null, label %zero, label %go, !prof !0

zero:
  ret i64 0

go:
  %h.len = getelementptr inbounds nuw i8, ptr %sa, i64 8
  %N = load i64, ptr %h.len, align 8
  %h.sa = getelementptr inbounds nuw i8, ptr %sa, i64 16
  %sap = load ptr, ptr %h.sa, align 8
  %empty = icmp eq i64 %N, 0
  br i1 %empty, label %zero, label %search, !prof !0

search:
  %lo = call i64 @sa_bound(ptr %sap, ptr %text, i64 %N, ptr %pattern, i64 %plen, i1 false)
  %hi = call i64 @sa_bound(ptr %sap, ptr %text, i64 %N, ptr %pattern, i64 %plen, i1 true)
  %cnt = sub i64 %hi, %lo
  ret i64 %cnt
}

define i64 @universe_search_sa_at(ptr %sa, i64 %i) local_unnamed_addr #7 {
entry:
  %sa.null = icmp eq ptr %sa, null
  br i1 %sa.null, label %miss, label %go, !prof !0

miss:
  ret i64 -1

go:
  %h.len = getelementptr inbounds nuw i8, ptr %sa, i64 8
  %N = load i64, ptr %h.len, align 8
  %oob = icmp uge i64 %i, %N
  br i1 %oob, label %miss, label %read, !prof !0

read:
  %h.sa = getelementptr inbounds nuw i8, ptr %sa, i64 16
  %sap = load ptr, ptr %h.sa, align 8
  %p = getelementptr inbounds i32, ptr %sap, i64 %i
  %v = load i32, ptr %p, align 4
  %vz = zext i32 %v to i64
  ret i64 %vz
}

define i64 @universe_search_sa_lcp_at(ptr %sa, i64 %i) local_unnamed_addr #7 {
entry:
  %sa.null = icmp eq ptr %sa, null
  br i1 %sa.null, label %miss, label %go, !prof !0

miss:
  ret i64 -1

go:
  %h.len = getelementptr inbounds nuw i8, ptr %sa, i64 8
  %N = load i64, ptr %h.len, align 8
  %oob = icmp uge i64 %i, %N
  br i1 %oob, label %miss, label %read, !prof !0

read:
  %h.lcp = getelementptr inbounds nuw i8, ptr %sa, i64 24
  %lcpp = load ptr, ptr %h.lcp, align 8
  %p = getelementptr inbounds i32, ptr %lcpp, i64 %i
  %v = load i32, ptr %p, align 4
  %vz = zext i32 %v to i64
  ret i64 %vz
}

define i64 @universe_search_sa_len(ptr %sa) local_unnamed_addr #7 {
entry:
  %sa.null = icmp eq ptr %sa, null
  br i1 %sa.null, label %zero, label %go, !prof !0

zero:
  ret i64 0

go:
  %h.len = getelementptr inbounds nuw i8, ptr %sa, i64 8
  %N = load i64, ptr %h.len, align 8
  ret i64 %N
}

define void @universe_search_sa_destroy(ptr %sa) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %sa, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %sa)
  br label %done

done:
  ret void
}

attributes #1 = { nounwind willreturn }
attributes #4 = { nounwind willreturn norecurse nosync alwaysinline memory(argmem: read) }
attributes #5 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #6 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #7 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
