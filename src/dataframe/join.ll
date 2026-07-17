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

; DataFrame relational hash join. Functional inspiration from Polars method
; NAMES (join / inner_/left_/outer_/cross_join); layout + algorithm are ours.
;
; ============================ DESIGN ============================
; RELIES on frame.ll's documented byte layout (read directly here):
;   Series: +0 i32 dtype; +8 i64 len; +16 i64 null_count; +24 ptr values;
;           +32 ptr validity(1=valid,null=all valid); +40 ptr strdata; +48 i64 strdata_len.
;   DataFrame: +0 i64 n_cols; +16 i64 height; +24 ptr names({ptr,i64}[16B each});
;           +32 ptr columns.
;   STR: values = i32 offsets[len+1]; string i = strdata[off[i]..off[i+1]).
;
; ALGORITHM CLASS: build-probe hash join. Build an inline MULTIMAP over the
; RIGHT key columns as key-hash -> chain of right-row indices, realized with
; two flat index arrays (no pointers, no tombstones):
;   head[nbuckets] (i64, -1 = empty), next[rlen] (i64 chain link).
; nbuckets = next_pow2(2*rlen); bucket = hash & (nbuckets-1) (mask, not modulo).
; Insert right row j: next[j]=head[b]; head[b]=j  (a stack -> acyclic chains).
; Probe with each LEFT row: walk its bucket chain, key-equality-check each
; candidate, emit matched (left_row, right_row) index pairs into growable
; arrays. Output columns are built by GATHER over those index arrays (indices,
; not bytes); a -1 index gathers a NULL element (reuses the frame gather idiom).
;
; JOIN SEMANTICS (how): 0 inner (matches only), 1 left (all left rows; misses
; get a right-side null row = ridx -1), 2 outer (inner + left misses + right
; misses, ridx/lidx -1 respectively), 3 cross (full n*m product, keys ignored).
; NULL keys never match (Polars default join_nulls=false): any row with a null
; in ANY key column is treated as non-matching (never inserted / never joined).
;
; OUTPUT COLUMNS: all LEFT columns (names kept), then RIGHT NON-KEY columns
; (right key columns dropped for keyed joins; for cross none are dropped). A
; right column whose name already exists in LEFT gets a "_right" suffix (Polars).
;
; KEY HASH/EQ: FNV-1a over the raw value bytes of each key column at the row
; (fixed dtypes: width bytes; STR: the string bytes), folded across key columns.
; Equality compares the SAME bytes (memcmp on fixed width / string bytes) so
; hash and equality are consistent. Floats compared bitwise (documented: -0.0
; != +0.0, NaN != NaN under this scheme; acceptable for v1 join semantics).
;
; Errors: constructors return null on failure (NULL args, key not found,
; key dtype mismatch, size overflow). No partial DataFrame is leaked on error.
; ===============================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @memcmp(ptr captures(none), ptr captures(none), i64)

; frame.ll public API
declare ptr @universe_dataframe_new()
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare i32 @universe_dataframe_get_column_index(ptr, ptr, i64, ptr)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)

@join.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"
@join.suffix = internal constant [6 x i8] c"_right"

; ---------------------------------------------------------------------------
; small helpers
; ---------------------------------------------------------------------------

define internal ptr @join_xmalloc(i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

; validity bitmap (all valid = 0xFF) for len elements
define internal ptr @join_bm(i64 %len) #1 {
entry:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %p = call ptr @join_xmalloc(i64 %nb)
  call void @llvm.memset.p0.i64(ptr %p, i8 -1, i64 %nb, i1 false)
  ret ptr %p
}

define internal void @join_bmclr(ptr %bm, i64 %k) #4 {
entry:
  %bi = lshr i64 %k, 3
  %bp = getelementptr inbounds i8, ptr %bm, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %k, 7
  %sh8 = trunc i64 %sh to i8
  %m = shl i8 1, %sh8
  %nm = xor i8 %m, -1
  %b2 = and i8 %b, %nm
  store i8 %b2, ptr %bp, align 1
  ret void
}

; true if row i of series s is valid (non-null). validity==null => all valid.
define internal i1 @join_valid(ptr %s, i64 %i) #6 {
entry:
  %vp = getelementptr inbounds i8, ptr %s, i64 32
  %v = load ptr, ptr %vp, align 8
  %vn = icmp eq ptr %v, null
  br i1 %vn, label %yes, label %chk

yes:
  ret i1 true

chk:
  %bi = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %v, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %i, 7
  %sh8 = trunc i64 %sh to i8
  %bit = lshr i8 %b, %sh8
  %lo = and i8 %bit, 1
  %r = icmp ne i8 %lo, 0
  ret i1 %r
}

define internal i64 @join_width(i32 %dt) #5 {
entry:
  %i = zext i32 %dt to i64
  %p = getelementptr inbounds [6 x i8], ptr @join.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

; FNV-1a fold of len bytes at p into hash h0
define internal i64 @join_fnv(i64 %h0, ptr %p, i64 %len) #6 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %body ]
  %h = phi i64 [ %h0, %entry ], [ %hn, %body ]
  %c = icmp ult i64 %k, %len
  br i1 %c, label %body, label %done

body:
  %bp = getelementptr inbounds i8, ptr %p, i64 %k
  %b = load i8, ptr %bp, align 1
  %bz = zext i8 %b to i64
  %x = xor i64 %h, %bz
  %hn = mul i64 %x, 1099511628211
  %kn = add i64 %k, 1
  br label %loop

done:
  ret i64 %h
}

; smallest power of two >= max(x,1), clamped to 2^62
define internal i64 @join_np2(i64 %x) #7 {
entry:
  %x1 = call i64 @llvm.umax.i64(i64 %x, i64 1)
  %xm = sub i64 %x1, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %xm, i1 false)
  %sh0 = sub i64 64, %lz
  %sh = call i64 @llvm.umin.i64(i64 %sh0, i64 62)
  %res = shl i64 1, %sh
  ret i64 %res
}

; hash the key tuple of `row` across nk key columns (ser = ptr[] of nk series)
define internal i64 @join_hash_row(ptr %ser, i64 %nk, i64 %row) #3 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %h = phi i64 [ 14695981039346656037, %entry ], [ %hn, %cont ]
  %c = icmp ult i64 %k, %nk
  br i1 %c, label %body, label %done

body:
  %sp = getelementptr inbounds ptr, ptr %ser, i64 %k
  %s = load ptr, ptr %sp, align 8
  %dt = load i32, ptr %s, align 8
  %isstr = icmp eq i32 %dt, 5
  br i1 %isstr, label %str, label %fix

str:
  %offp = getelementptr inbounds i8, ptr %s, i64 24
  %offs = load ptr, ptr %offp, align 8
  %o0p = getelementptr inbounds i32, ptr %offs, i64 %row
  %o0 = load i32, ptr %o0p, align 4
  %row1 = add i64 %row, 1
  %o1p = getelementptr inbounds i32, ptr %offs, i64 %row1
  %o1 = load i32, ptr %o1p, align 4
  %o0z = zext i32 %o0 to i64
  %o1z = zext i32 %o1 to i64
  %slen = sub i64 %o1z, %o0z
  %sdp = getelementptr inbounds i8, ptr %s, i64 40
  %sd = load ptr, ptr %sdp, align 8
  %strp = getelementptr inbounds i8, ptr %sd, i64 %o0z
  %hs = call i64 @join_fnv(i64 %h, ptr %strp, i64 %slen)
  br label %cont

fix:
  %w = call i64 @join_width(i32 %dt)
  %vp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vp, align 8
  %off = mul i64 %row, %w
  %ep = getelementptr inbounds i8, ptr %vals, i64 %off
  %hf = call i64 @join_fnv(i64 %h, ptr %ep, i64 %w)
  br label %cont

cont:
  %hn = phi i64 [ %hs, %str ], [ %hf, %fix ]
  %kn = add i64 %k, 1
  br label %loop

done:
  ret i64 %h
}

; true if `row` has a null in ANY of the nk key columns
define internal i1 @join_row_null(ptr %ser, i64 %nk, i64 %row) #3 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %c = icmp ult i64 %k, %nk
  br i1 %c, label %body, label %no

body:
  %sp = getelementptr inbounds ptr, ptr %ser, i64 %k
  %s = load ptr, ptr %sp, align 8
  %v = call i1 @join_valid(ptr %s, i64 %row)
  br i1 %v, label %cont, label %yes

cont:
  %kn = add i64 %k, 1
  br label %loop

yes:
  ret i1 true

no:
  ret i1 false
}

; true if left row i and right row j have equal key tuples (both assumed
; non-null keys; dtypes verified equal pairwise by the caller)
define internal i1 @join_keq(ptr %lser, i64 %i, ptr %rser, i64 %j, i64 %nk) #3 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %c = icmp ult i64 %k, %nk
  br i1 %c, label %body, label %eq

body:
  %lsp = getelementptr inbounds ptr, ptr %lser, i64 %k
  %ls = load ptr, ptr %lsp, align 8
  %rsp = getelementptr inbounds ptr, ptr %rser, i64 %k
  %rs = load ptr, ptr %rsp, align 8
  %dt = load i32, ptr %ls, align 8
  %isstr = icmp eq i32 %dt, 5
  br i1 %isstr, label %str, label %fix

str:
  %loffp = getelementptr inbounds i8, ptr %ls, i64 24
  %loffs = load ptr, ptr %loffp, align 8
  %lo0p = getelementptr inbounds i32, ptr %loffs, i64 %i
  %lo0 = load i32, ptr %lo0p, align 4
  %i1 = add i64 %i, 1
  %lo1p = getelementptr inbounds i32, ptr %loffs, i64 %i1
  %lo1 = load i32, ptr %lo1p, align 4
  %lo0z = zext i32 %lo0 to i64
  %lo1z = zext i32 %lo1 to i64
  %llen = sub i64 %lo1z, %lo0z
  %lsdp = getelementptr inbounds i8, ptr %ls, i64 40
  %lsd = load ptr, ptr %lsdp, align 8
  %lp = getelementptr inbounds i8, ptr %lsd, i64 %lo0z
  %roffp = getelementptr inbounds i8, ptr %rs, i64 24
  %roffs = load ptr, ptr %roffp, align 8
  %ro0p = getelementptr inbounds i32, ptr %roffs, i64 %j
  %ro0 = load i32, ptr %ro0p, align 4
  %j1 = add i64 %j, 1
  %ro1p = getelementptr inbounds i32, ptr %roffs, i64 %j1
  %ro1 = load i32, ptr %ro1p, align 4
  %ro0z = zext i32 %ro0 to i64
  %ro1z = zext i32 %ro1 to i64
  %rlen = sub i64 %ro1z, %ro0z
  %rsdp = getelementptr inbounds i8, ptr %rs, i64 40
  %rsd = load ptr, ptr %rsdp, align 8
  %rp = getelementptr inbounds i8, ptr %rsd, i64 %ro0z
  %lenne = icmp ne i64 %llen, %rlen
  br i1 %lenne, label %ne, label %strcmp

strcmp:
  %scmp = call i32 @memcmp(ptr %lp, ptr %rp, i64 %llen)
  %scne = icmp ne i32 %scmp, 0
  br i1 %scne, label %ne, label %cont

fix:
  %w = call i64 @join_width(i32 %dt)
  %lvp = getelementptr inbounds i8, ptr %ls, i64 24
  %lvals = load ptr, ptr %lvp, align 8
  %loff = mul i64 %i, %w
  %lep = getelementptr inbounds i8, ptr %lvals, i64 %loff
  %rvp = getelementptr inbounds i8, ptr %rs, i64 24
  %rvals = load ptr, ptr %rvp, align 8
  %roff = mul i64 %j, %w
  %rep = getelementptr inbounds i8, ptr %rvals, i64 %roff
  ; width-dispatched integer compare (1/4/8) — call-free hot leaf, monomorphic
  ; per column so the width branch is perfectly predicted.
  %isw8 = icmp eq i64 %w, 8
  br i1 %isw8, label %cmp8, label %fixsmall

cmp8:
  %l8 = load i64, ptr %lep, align 1
  %r8 = load i64, ptr %rep, align 1
  %e8 = icmp eq i64 %l8, %r8
  br i1 %e8, label %cont, label %ne

fixsmall:
  %isw4 = icmp eq i64 %w, 4
  br i1 %isw4, label %cmp4, label %cmp1

cmp4:
  %l4 = load i32, ptr %lep, align 1
  %r4 = load i32, ptr %rep, align 1
  %e4 = icmp eq i32 %l4, %r4
  br i1 %e4, label %cont, label %ne

cmp1:
  %l1v = load i8, ptr %lep, align 1
  %r1v = load i8, ptr %rep, align 1
  %e1 = icmp eq i8 %l1v, %r1v
  br i1 %e1, label %cont, label %ne

cont:
  %kn = add i64 %k, 1
  br label %loop

eq:
  ret i1 true

ne:
  ret i1 false
}

; linear membership over a small i64 set (right key col indices). n==0 => false.
define internal i1 @join_in_set(ptr %set, i64 %n, i64 %v) #2 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %c = icmp ult i64 %k, %n
  br i1 %c, label %body, label %no

body:
  %ep = getelementptr inbounds i64, ptr %set, i64 %k
  %e = load i64, ptr %ep, align 8
  %eq = icmp eq i64 %e, %v
  br i1 %eq, label %yes, label %cont

cont:
  %kn = add i64 %k, 1
  br label %loop

yes:
  ret i1 true

no:
  ret i1 false
}

; ---------------------------------------------------------------------------
; growable (lidx,ridx) pair buffer. state = [4 x i64]: +0 lbuf,+8 rbuf,
; +16 count,+24 cap.
; ---------------------------------------------------------------------------
define internal void @join_push(ptr %st, i64 %l, i64 %r) #1 {
entry:
  %cntp = getelementptr inbounds i8, ptr %st, i64 16
  %cnt = load i64, ptr %cntp, align 8
  %capp = getelementptr inbounds i8, ptr %st, i64 24
  %cap = load i64, ptr %capp, align 8
  %full = icmp eq i64 %cnt, %cap
  br i1 %full, label %grow, label %store

grow:
  %z = icmp eq i64 %cap, 0
  %cap2 = shl i64 %cap, 1
  %ncap = select i1 %z, i64 16, i64 %cap2
  %nb = shl i64 %ncap, 3
  %lp = getelementptr inbounds i8, ptr %st, i64 0
  %oldl = load ptr, ptr %lp, align 8
  %newl = call ptr @realloc(ptr %oldl, i64 %nb)
  store ptr %newl, ptr %lp, align 8
  %rp = getelementptr inbounds i8, ptr %st, i64 8
  %oldr = load ptr, ptr %rp, align 8
  %newr = call ptr @realloc(ptr %oldr, i64 %nb)
  store ptr %newr, ptr %rp, align 8
  store i64 %ncap, ptr %capp, align 8
  br label %store

store:
  %lp2 = getelementptr inbounds i8, ptr %st, i64 0
  %lbuf = load ptr, ptr %lp2, align 8
  %ls = getelementptr inbounds i64, ptr %lbuf, i64 %cnt
  store i64 %l, ptr %ls, align 8
  %rp2 = getelementptr inbounds i8, ptr %st, i64 8
  %rbuf = load ptr, ptr %rp2, align 8
  %rs = getelementptr inbounds i64, ptr %rbuf, i64 %cnt
  store i64 %r, ptr %rs, align 8
  %cn = add i64 %cnt, 1
  store i64 %cn, ptr %cntp, align 8
  ret void
}

; ---------------------------------------------------------------------------
; join_gather: NEW series = rows of s selected by idx[0..n) (i64 each);
; idx entry < 0 => a null / empty element. Self-contained (frame's gather is
; internal to frame.ll). Handles fixed dtypes + STR + validity.
; ---------------------------------------------------------------------------
define internal ptr @join_gather(ptr %s, ptr %idx, i64 %n) #1 {
entry:
  %dtype = load i32, ptr %s, align 8
  %out = call ptr @malloc(i64 56)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %head0, !prof !0

head0:
  call void @llvm.memset.p0.i64(ptr %out, i8 0, i64 56, i1 false)
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %n, ptr %olp, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %w = call i64 @join_width(i32 %dtype)
  %fbytes = mul i64 %n, %w
  %fvals = call ptr @join_xmalloc(i64 %fbytes)
  %fvpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %fvals, ptr %fvpp, align 8
  %svpp = getelementptr inbounds i8, ptr %s, i64 24
  %svals = load ptr, ptr %svpp, align 8
  br label %f1h

f1h:
  %f1k = phi i64 [ 0, %fixed ], [ %f1kn, %f1i ]
  %f1nulls = phi i64 [ 0, %fixed ], [ %f1nn, %f1i ]
  %f1c = icmp ult i64 %f1k, %n
  br i1 %f1c, label %f1b, label %f1d

f1b:
  %f1jp = getelementptr inbounds i64, ptr %idx, i64 %f1k
  %f1j = load i64, ptr %f1jp, align 8
  %f1neg = icmp slt i64 %f1j, 0
  br i1 %f1neg, label %f1i, label %f1chk

f1chk:
  %f1v = call i1 @join_valid(ptr %s, i64 %f1j)
  %f1sn = xor i1 %f1v, true
  br label %f1i

f1i:
  %f1null = phi i1 [ true, %f1b ], [ %f1sn, %f1chk ]
  %f1inc = zext i1 %f1null to i64
  %f1nn = add i64 %f1nulls, %f1inc
  %f1kn = add i64 %f1k, 1
  br label %f1h

f1d:
  %fhn = icmp ugt i64 %f1nulls, 0
  br i1 %fhn, label %fmkbm, label %f2h

fmkbm:
  %fbm = call ptr @join_bm(i64 %n)
  %fbmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %fbm, ptr %fbmp, align 8
  %fncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %f1nulls, ptr %fncp, align 8
  br label %f2h

f2h:
  %f2bm = phi ptr [ null, %f1d ], [ %fbm, %fmkbm ]
  br label %f2l

f2l:
  %f2k = phi i64 [ 0, %f2h ], [ %f2kn, %f2cont ]
  %f2c = icmp ult i64 %f2k, %n
  br i1 %f2c, label %f2b, label %done

f2b:
  %f2off = mul i64 %f2k, %w
  %f2dst = getelementptr inbounds i8, ptr %fvals, i64 %f2off
  %f2jp = getelementptr inbounds i64, ptr %idx, i64 %f2k
  %f2j = load i64, ptr %f2jp, align 8
  %f2neg = icmp slt i64 %f2j, 0
  br i1 %f2neg, label %f2vac, label %f2cp

f2vac:
  call void @llvm.memset.p0.i64(ptr %f2dst, i8 0, i64 %w, i1 false)
  call void @join_bmclr(ptr %f2bm, i64 %f2k)
  br label %f2cont

f2cp:
  %f2soff = mul i64 %f2j, %w
  %f2src = getelementptr inbounds i8, ptr %svals, i64 %f2soff
  call void @llvm.memcpy.p0.p0.i64(ptr %f2dst, ptr %f2src, i64 %w, i1 false)
  %f2v = call i1 @join_valid(ptr %s, i64 %f2j)
  br i1 %f2v, label %f2cont, label %f2mn

f2mn:
  call void @join_bmclr(ptr %f2bm, i64 %f2k)
  br label %f2cont

f2cont:
  %f2kn = add i64 %f2k, 1
  br label %f2l

str:
  %ssvpp = getelementptr inbounds i8, ptr %s, i64 24
  %ssoffs = load ptr, ptr %ssvpp, align 8
  %sssdp = getelementptr inbounds i8, ptr %s, i64 40
  %sssd = load ptr, ptr %sssdp, align 8
  br label %s1h

s1h:
  %s1k = phi i64 [ 0, %str ], [ %s1kn, %s1acc ]
  %s1total = phi i64 [ 0, %str ], [ %s1tn, %s1acc ]
  %s1nulls = phi i64 [ 0, %str ], [ %s1nn, %s1acc ]
  %s1c = icmp ult i64 %s1k, %n
  br i1 %s1c, label %s1b, label %s1d

s1b:
  %s1jp = getelementptr inbounds i64, ptr %idx, i64 %s1k
  %s1j = load i64, ptr %s1jp, align 8
  %s1neg = icmp slt i64 %s1j, 0
  br i1 %s1neg, label %s1acc, label %s1chk

s1chk:
  %s1j1 = add i64 %s1j, 1
  %s1o0p = getelementptr inbounds i32, ptr %ssoffs, i64 %s1j
  %s1o0 = load i32, ptr %s1o0p, align 4
  %s1o1p = getelementptr inbounds i32, ptr %ssoffs, i64 %s1j1
  %s1o1 = load i32, ptr %s1o1p, align 4
  %s1o0z = zext i32 %s1o0 to i64
  %s1o1z = zext i32 %s1o1 to i64
  %s1slen = sub i64 %s1o1z, %s1o0z
  %s1v = call i1 @join_valid(ptr %s, i64 %s1j)
  %s1sn = xor i1 %s1v, true
  br label %s1acc

s1acc:
  %s1addb = phi i64 [ 0, %s1b ], [ %s1slen, %s1chk ]
  %s1null = phi i1 [ true, %s1b ], [ %s1sn, %s1chk ]
  %s1tn = add i64 %s1total, %s1addb
  %s1ninc = zext i1 %s1null to i64
  %s1nn = add i64 %s1nulls, %s1ninc
  %s1kn = add i64 %s1k, 1
  br label %s1h

s1d:
  %sob = add i64 %n, 1
  %sobytes = mul i64 %sob, 4
  %soffs = call ptr @join_xmalloc(i64 %sobytes)
  %sovpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %soffs, ptr %sovpp, align 8
  %ssd2 = call ptr @join_xmalloc(i64 %s1total)
  %ssdp2 = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %ssd2, ptr %ssdp2, align 8
  %ssdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %s1total, ptr %ssdlp, align 8
  %shn = icmp ugt i64 %s1nulls, 0
  br i1 %shn, label %smkbm, label %s2h

smkbm:
  %sbm = call ptr @join_bm(i64 %n)
  %sbmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %sbm, ptr %sbmp, align 8
  %sncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %s1nulls, ptr %sncp, align 8
  br label %s2h

s2h:
  %s2bm = phi ptr [ null, %s1d ], [ %sbm, %smkbm ]
  br label %s2l

s2l:
  %s2k = phi i64 [ 0, %s2h ], [ %s2kn, %s2cont ]
  %s2cur = phi i64 [ 0, %s2h ], [ %s2curn, %s2cont ]
  %s2c = icmp ult i64 %s2k, %n
  br i1 %s2c, label %s2b, label %s2fin

s2b:
  %s2op = getelementptr inbounds i32, ptr %soffs, i64 %s2k
  %s2cur32 = trunc i64 %s2cur to i32
  store i32 %s2cur32, ptr %s2op, align 4
  %s2jp = getelementptr inbounds i64, ptr %idx, i64 %s2k
  %s2j = load i64, ptr %s2jp, align 8
  %s2neg = icmp slt i64 %s2j, 0
  br i1 %s2neg, label %s2vac, label %s2cp

s2vac:
  call void @join_bmclr(ptr %s2bm, i64 %s2k)
  br label %s2c0

s2cp:
  %s2j1 = add i64 %s2j, 1
  %s2o0p = getelementptr inbounds i32, ptr %ssoffs, i64 %s2j
  %s2o0 = load i32, ptr %s2o0p, align 4
  %s2o1p = getelementptr inbounds i32, ptr %ssoffs, i64 %s2j1
  %s2o1 = load i32, ptr %s2o1p, align 4
  %s2o0z = zext i32 %s2o0 to i64
  %s2o1z = zext i32 %s2o1 to i64
  %s2slen = sub i64 %s2o1z, %s2o0z
  %s2src = getelementptr inbounds i8, ptr %sssd, i64 %s2o0z
  %s2dst = getelementptr inbounds i8, ptr %ssd2, i64 %s2cur
  call void @llvm.memcpy.p0.p0.i64(ptr %s2dst, ptr %s2src, i64 %s2slen, i1 false)
  %s2v = call i1 @join_valid(ptr %s, i64 %s2j)
  br i1 %s2v, label %s2c1, label %s2mn

s2mn:
  call void @join_bmclr(ptr %s2bm, i64 %s2k)
  br label %s2c1

s2c0:
  br label %s2cont

s2c1:
  br label %s2cont

s2cont:
  %s2add = phi i64 [ 0, %s2c0 ], [ %s2slen, %s2c1 ]
  %s2curn = add i64 %s2cur, %s2add
  %s2kn = add i64 %s2k, 1
  br label %s2l

s2fin:
  %s2lastp = getelementptr inbounds i32, ptr %soffs, i64 %n
  %s2last32 = trunc i64 %s2cur to i32
  store i32 %s2last32, ptr %s2lastp, align 4
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

; ---------------------------------------------------------------------------
; hash-join core: build over right key cols, probe with left, emit index
; pairs into `st`. how in {0 inner,1 left,2 outer}. Returns 0.
; ---------------------------------------------------------------------------
define internal i32 @join_hashjoin(ptr %left, ptr %right, ptr %lser, ptr %rser, i64 %nk, i32 %how, ptr %st) #1 {
entry:
  %llen = call i64 @universe_dataframe_height(ptr %left)
  %rlen = call i64 @universe_dataframe_height(ptr %right)
  %isouter = icmp eq i32 %how, 2
  %r2 = shl i64 %rlen, 1
  %nb = call i64 @join_np2(i64 %r2)
  %mask = sub i64 %nb, 1
  %hbytes = shl i64 %nb, 3
  %head = call ptr @join_xmalloc(i64 %hbytes)
  call void @llvm.memset.p0.i64(ptr %head, i8 -1, i64 %hbytes, i1 false)
  %nbytes = shl i64 %rlen, 3
  %next = call ptr @join_xmalloc(i64 %nbytes)
  %rm = call ptr @join_xmalloc(i64 %rlen)
  call void @llvm.memset.p0.i64(ptr %rm, i8 0, i64 %rlen, i1 false)
  br label %b.head

b.head:
  %bj = phi i64 [ 0, %entry ], [ %bjn, %b.cont ]
  %bc = icmp ult i64 %bj, %rlen
  br i1 %bc, label %b.body, label %p.head

b.body:
  %bnull = call i1 @join_row_null(ptr %rser, i64 %nk, i64 %bj)
  br i1 %bnull, label %b.cont, label %b.ins

b.ins:
  %bh = call i64 @join_hash_row(ptr %rser, i64 %nk, i64 %bj)
  %bb = and i64 %bh, %mask
  %bhs = getelementptr inbounds i64, ptr %head, i64 %bb
  %bold = load i64, ptr %bhs, align 8
  %bns = getelementptr inbounds i64, ptr %next, i64 %bj
  store i64 %bold, ptr %bns, align 8
  store i64 %bj, ptr %bhs, align 8
  br label %b.cont

b.cont:
  %bjn = add i64 %bj, 1
  br label %b.head

p.head:
  %pi = phi i64 [ 0, %b.head ], [ %pin, %p.after ]
  %pc = icmp ult i64 %pi, %llen
  br i1 %pc, label %p.body, label %o.head

p.body:
  %pnull = call i1 @join_row_null(ptr %lser, i64 %nk, i64 %pi)
  br i1 %pnull, label %p.checkextra, label %p.probe

p.probe:
  %ph = call i64 @join_hash_row(ptr %lser, i64 %nk, i64 %pi)
  %pb = and i64 %ph, %mask
  %phs = getelementptr inbounds i64, ptr %head, i64 %pb
  %pj0 = load i64, ptr %phs, align 8
  br label %w.head

w.head:
  %wj = phi i64 [ %pj0, %p.probe ], [ %wjn, %w.cont ]
  %wm = phi i1 [ false, %p.probe ], [ %wm2, %w.cont ]
  %wc = icmp sge i64 %wj, 0
  br i1 %wc, label %w.body, label %w.done

w.body:
  %weq = call i1 @join_keq(ptr %lser, i64 %pi, ptr %rser, i64 %wj, i64 %nk)
  br i1 %weq, label %w.match, label %w.next

w.match:
  call void @join_push(ptr %st, i64 %pi, i64 %wj)
  br i1 %isouter, label %w.mark, label %w.next2

w.mark:
  %rmp = getelementptr inbounds i8, ptr %rm, i64 %wj
  store i8 1, ptr %rmp, align 1
  br label %w.next2

w.next2:
  br label %w.cont

w.next:
  br label %w.cont

w.cont:
  %wm2 = phi i1 [ true, %w.next2 ], [ %wm, %w.next ]
  %wjp = getelementptr inbounds i64, ptr %next, i64 %wj
  %wjn = load i64, ptr %wjp, align 8
  br label %w.head

w.done:
  br i1 %wm, label %p.after, label %p.checkextra

p.checkextra:
  %isleftouter = icmp uge i32 %how, 1
  br i1 %isleftouter, label %p.emitnull, label %p.after

p.emitnull:
  call void @join_push(ptr %st, i64 %pi, i64 -1)
  br label %p.after

p.after:
  %pin = add i64 %pi, 1
  br label %p.head

o.head:
  br i1 %isouter, label %o.loop, label %fin

o.loop:
  %oj = phi i64 [ 0, %o.head ], [ %ojn, %o.cont ]
  %oc = icmp ult i64 %oj, %rlen
  br i1 %oc, label %o.body, label %fin

o.body:
  %omp = getelementptr inbounds i8, ptr %rm, i64 %oj
  %omv = load i8, ptr %omp, align 1
  %omatched = icmp ne i8 %omv, 0
  br i1 %omatched, label %o.cont, label %o.emit

o.emit:
  call void @join_push(ptr %st, i64 -1, i64 %oj)
  br label %o.cont

o.cont:
  %ojn = add i64 %oj, 1
  br label %o.loop

fin:
  call void @free(ptr %head)
  call void @free(ptr %next)
  call void @free(ptr %rm)
  ret i32 0
}

; ---------------------------------------------------------------------------
; build the output DataFrame: gather all LEFT columns by lidx, then all RIGHT
; NON-KEY columns by ridx (with _right suffix on name collision with left).
; ---------------------------------------------------------------------------
define internal ptr @join_build_output(ptr %left, ptr %right, ptr %lidx, ptr %ridx, i64 %cnt, ptr %rkidx, i64 %nk) #1 {
entry:
  %tmp = alloca i64, align 8
  %out = call ptr @universe_dataframe_new()
  %lw = call i64 @universe_dataframe_width(ptr %left)
  %rw = call i64 @universe_dataframe_width(ptr %right)
  %rnp = getelementptr inbounds i8, ptr %right, i64 24
  %rnames = load ptr, ptr %rnp, align 8
  %lnp = getelementptr inbounds i8, ptr %left, i64 24
  %lnames = load ptr, ptr %lnp, align 8
  br label %lc.head

lc.head:
  %lc = phi i64 [ 0, %entry ], [ %lcn, %lc.cont ]
  %lcc = icmp ult i64 %lc, %lw
  br i1 %lcc, label %lc.body, label %rc.head

lc.body:
  %lcol = call ptr @universe_dataframe_select_at_idx(ptr %left, i64 %lc)
  %lg = call ptr @join_gather(ptr %lcol, ptr %lidx, i64 %cnt)
  %lnoff = mul i64 %lc, 16
  %lnslot = getelementptr inbounds i8, ptr %lnames, i64 %lnoff
  %lnptr = load ptr, ptr %lnslot, align 8
  %lnlenp = getelementptr inbounds i8, ptr %lnslot, i64 8
  %lnlen = load i64, ptr %lnlenp, align 8
  %lrc = call i32 @universe_dataframe_with_column(ptr %out, ptr %lnptr, i64 %lnlen, ptr %lg)
  br label %lc.cont

lc.cont:
  %lcn = add i64 %lc, 1
  br label %lc.head

rc.head:
  %rc = phi i64 [ 0, %lc.head ], [ %rcn, %rc.cont ]
  %rcc = icmp ult i64 %rc, %rw
  br i1 %rcc, label %rc.body, label %done

rc.body:
  %iskey = call i1 @join_in_set(ptr %rkidx, i64 %nk, i64 %rc)
  br i1 %iskey, label %rc.cont, label %rc.emit

rc.emit:
  %rcol = call ptr @universe_dataframe_select_at_idx(ptr %right, i64 %rc)
  %rg = call ptr @join_gather(ptr %rcol, ptr %ridx, i64 %cnt)
  %rnoff = mul i64 %rc, 16
  %rnslot = getelementptr inbounds i8, ptr %rnames, i64 %rnoff
  %rnptr = load ptr, ptr %rnslot, align 8
  %rnlenp = getelementptr inbounds i8, ptr %rnslot, i64 8
  %rnlen = load i64, ptr %rnlenp, align 8
  %ci = call i32 @universe_dataframe_get_column_index(ptr %left, ptr %rnptr, i64 %rnlen, ptr %tmp)
  %collide = icmp eq i32 %ci, 0
  br i1 %collide, label %rc.suffix, label %rc.plain

rc.plain:
  %prc = call i32 @universe_dataframe_with_column(ptr %out, ptr %rnptr, i64 %rnlen, ptr %rg)
  br label %rc.cont

rc.suffix:
  %snlen = add i64 %rnlen, 6
  %snbuf = call ptr @join_xmalloc(i64 %snlen)
  call void @llvm.memcpy.p0.p0.i64(ptr %snbuf, ptr %rnptr, i64 %rnlen, i1 false)
  %sufp = getelementptr inbounds i8, ptr %snbuf, i64 %rnlen
  call void @llvm.memcpy.p0.p0.i64(ptr %sufp, ptr @join.suffix, i64 6, i1 false)
  %src2 = call i32 @universe_dataframe_with_column(ptr %out, ptr %snbuf, i64 %snlen, ptr %rg)
  call void @free(ptr %snbuf)
  br label %rc.cont

rc.cont:
  %rcn = add i64 %rc, 1
  br label %rc.head

done:
  ret ptr %out
}

; ---------------------------------------------------------------------------
; public API
; ---------------------------------------------------------------------------
define ptr @universe_dataframe_join(ptr %left, ptr %right, ptr %left_on, i64 %nl, ptr %right_on, i64 %nr, i32 %how) local_unnamed_addr #1 {
entry:
  %st = alloca [4 x i64], align 8
  %ktmp = alloca i64, align 8
  %ln = icmp eq ptr %left, null
  %rn = icmp eq ptr %right, null
  %anynull = or i1 %ln, %rn
  br i1 %anynull, label %fail, label %chk, !prof !0

chk:
  call void @llvm.memset.p0.i64(ptr %st, i8 0, i64 32, i1 false)
  %iscross = icmp eq i32 %how, 3
  br i1 %iscross, label %cross, label %keyed

cross:
  %clen = call i64 @universe_dataframe_height(ptr %left)
  %crlen = call i64 @universe_dataframe_height(ptr %right)
  %cm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %clen, i64 %crlen)
  %cov = extractvalue { i64, i1 } %cm, 1
  br i1 %cov, label %fail, label %cx.i.head, !prof !0

cx.i.head:
  %cxi = phi i64 [ 0, %cross ], [ %cxin, %cx.i.cont ]
  %cxic = icmp ult i64 %cxi, %clen
  br i1 %cxic, label %cx.j.head, label %cx.build

cx.j.head:
  %cxj = phi i64 [ 0, %cx.i.head ], [ %cxjn, %cx.j.body ]
  %cxjc = icmp ult i64 %cxj, %crlen
  br i1 %cxjc, label %cx.j.body, label %cx.i.cont

cx.j.body:
  call void @join_push(ptr %st, i64 %cxi, i64 %cxj)
  %cxjn = add i64 %cxj, 1
  br label %cx.j.head

cx.i.cont:
  %cxin = add i64 %cxi, 1
  br label %cx.i.head

cx.build:
  %cl0 = getelementptr inbounds i8, ptr %st, i64 0
  %clbuf = load ptr, ptr %cl0, align 8
  %cr0 = getelementptr inbounds i8, ptr %st, i64 8
  %crbuf = load ptr, ptr %cr0, align 8
  %ccntp = getelementptr inbounds i8, ptr %st, i64 16
  %ccnt = load i64, ptr %ccntp, align 8
  %cout = call ptr @join_build_output(ptr %left, ptr %right, ptr %clbuf, ptr %crbuf, i64 %ccnt, ptr null, i64 0)
  call void @free(ptr %clbuf)
  call void @free(ptr %crbuf)
  ret ptr %cout

keyed:
  %nleqnr = icmp eq i64 %nl, %nr
  %nlpos = icmp ugt i64 %nl, 0
  %okv = and i1 %nleqnr, %nlpos
  br i1 %okv, label %kalloc, label %fail, !prof !0

kalloc:
  %kb = shl i64 %nl, 3
  %lser = call ptr @join_xmalloc(i64 %kb)
  %rser = call ptr @join_xmalloc(i64 %kb)
  %rkidx = call ptr @join_xmalloc(i64 %kb)
  br label %kr.head

kr.head:
  %kk = phi i64 [ 0, %kalloc ], [ %kkn, %kr.cont ]
  %kkc = icmp ult i64 %kk, %nl
  br i1 %kkc, label %kr.body, label %kr.done

kr.body:
  %loff = mul i64 %kk, 16
  %lonslot = getelementptr inbounds i8, ptr %left_on, i64 %loff
  %lonptr = load ptr, ptr %lonslot, align 8
  %lonlenp = getelementptr inbounds i8, ptr %lonslot, i64 8
  %lonlen = load i64, ptr %lonlenp, align 8
  %lidxr = call i32 @universe_dataframe_get_column_index(ptr %left, ptr %lonptr, i64 %lonlen, ptr %ktmp)
  %lok = icmp eq i32 %lidxr, 0
  br i1 %lok, label %kr.right, label %kr.err

kr.right:
  %kli = load i64, ptr %ktmp, align 8
  %klser = call ptr @universe_dataframe_select_at_idx(ptr %left, i64 %kli)
  %lserslot = getelementptr inbounds ptr, ptr %lser, i64 %kk
  store ptr %klser, ptr %lserslot, align 8
  %roff = mul i64 %kk, 16
  %ronslot = getelementptr inbounds i8, ptr %right_on, i64 %roff
  %ronptr = load ptr, ptr %ronslot, align 8
  %ronlenp = getelementptr inbounds i8, ptr %ronslot, i64 8
  %ronlen = load i64, ptr %ronlenp, align 8
  %ridxr = call i32 @universe_dataframe_get_column_index(ptr %right, ptr %ronptr, i64 %ronlen, ptr %ktmp)
  %rok = icmp eq i32 %ridxr, 0
  br i1 %rok, label %kr.store, label %kr.err

kr.store:
  %kri = load i64, ptr %ktmp, align 8
  %krser = call ptr @universe_dataframe_select_at_idx(ptr %right, i64 %kri)
  %rserslot = getelementptr inbounds ptr, ptr %rser, i64 %kk
  store ptr %krser, ptr %rserslot, align 8
  %rkslot = getelementptr inbounds i64, ptr %rkidx, i64 %kk
  store i64 %kri, ptr %rkslot, align 8
  %kldt = load i32, ptr %klser, align 8
  %krdt = load i32, ptr %krser, align 8
  %dtne = icmp ne i32 %kldt, %krdt
  br i1 %dtne, label %kr.err, label %kr.cont

kr.cont:
  %kkn = add i64 %kk, 1
  br label %kr.head

kr.err:
  call void @free(ptr %lser)
  call void @free(ptr %rser)
  call void @free(ptr %rkidx)
  br label %fail

kr.done:
  %hjrc = call i32 @join_hashjoin(ptr %left, ptr %right, ptr %lser, ptr %rser, i64 %nl, i32 %how, ptr %st)
  %kl0 = getelementptr inbounds i8, ptr %st, i64 0
  %klbuf = load ptr, ptr %kl0, align 8
  %kr0 = getelementptr inbounds i8, ptr %st, i64 8
  %krbuf = load ptr, ptr %kr0, align 8
  %kcntp = getelementptr inbounds i8, ptr %st, i64 16
  %kcnt = load i64, ptr %kcntp, align 8
  %kout = call ptr @join_build_output(ptr %left, ptr %right, ptr %klbuf, ptr %krbuf, i64 %kcnt, ptr %rkidx, i64 %nl)
  call void @free(ptr %klbuf)
  call void @free(ptr %krbuf)
  call void @free(ptr %lser)
  call void @free(ptr %rser)
  call void @free(ptr %rkidx)
  ret ptr %kout

fail:
  ret ptr null
}

define ptr @universe_dataframe_inner_join(ptr %l, ptr %r, ptr %on, i64 %n) local_unnamed_addr #1 {
entry:
  %o = call ptr @universe_dataframe_join(ptr %l, ptr %r, ptr %on, i64 %n, ptr %on, i64 %n, i32 0)
  ret ptr %o
}

define ptr @universe_dataframe_left_join(ptr %l, ptr %r, ptr %on, i64 %n) local_unnamed_addr #1 {
entry:
  %o = call ptr @universe_dataframe_join(ptr %l, ptr %r, ptr %on, i64 %n, ptr %on, i64 %n, i32 1)
  ret ptr %o
}

define ptr @universe_dataframe_outer_join(ptr %l, ptr %r, ptr %on, i64 %n) local_unnamed_addr #1 {
entry:
  %o = call ptr @universe_dataframe_join(ptr %l, ptr %r, ptr %on, i64 %n, ptr %on, i64 %n, i32 2)
  ret ptr %o
}

define ptr @universe_dataframe_cross_join(ptr %l, ptr %r) local_unnamed_addr #1 {
entry:
  %o = call ptr @universe_dataframe_join(ptr %l, ptr %r, ptr null, i64 0, ptr null, i64 0, i32 3)
  ret ptr %o
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #5 = { alwaysinline nounwind willreturn norecurse nosync memory(read) }
attributes #6 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #7 = { alwaysinline nounwind willreturn norecurse nosync memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
