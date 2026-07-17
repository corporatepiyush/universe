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

; DataFrame row-selection engine: filter / take / sample / drop_nulls /
; fill_null / unique / is_unique / is_duplicated. Wave-2 consumer of frame.ll.
; Functional inspiration from Polars (method NAMES only); layout is our own.
;
; DESIGN
; ------
; Contract (from frame.ll, coded against by byte offset):
;   Series = 56B header:  +0 i32 dtype ; +8 i64 len ; +16 i64 null_count ;
;     +24 ptr values ; +32 ptr validity (null=>all valid, bit=1 VALID) ;
;     +40 ptr strdata ; +48 i64 strdata_len.
;   DataFrame = 40B:  +0 i64 n_cols ; +8 i64 cap_cols ; +16 i64 height ;
;     +24 ptr names ({ptr,i64}[]) ; +32 ptr columns (Series ptr[]).
;   DType: I32=0 I64=1 F32=2 F64=3 BOOL=4 (1 byte/val) STR=5 (i32 offsets[len+1]
;     in values, bytes in strdata). Fixed widths 4/8/4/8/1.
;
; frame.ll's gather/build helpers are `internal` (unreachable across the
; per-object archive link), so this module RE-IMPLEMENTS the proven shapes
; (series_gather, df_new/append/gather) directly against the documented layout
; — the disciplined choice per the "reuse the shape" hot-path rule. Only the
; public STR accessor (universe_dataframe_series_str_get) is called across the
; module edge (cold, per-row string equality/hash).
;
; ALGORITHM CLASSES
;   filter  : one pass building a compacted i64 index list of kept rows (mask
;             true AND valid), then a per-column index gather. Null mask entry
;             counts as false (Polars semantics).
;   take    : validate every index in [0,height) then gather; any OOB => the
;             ptr-return failure signal (null). (error class 8 has no ptr slot.)
;   sample_n: splitmix64 PRNG seeded by `seed` (bit-deterministic). With
;             replacement: idx = rng % height. Without: partial Fisher-Yates
;             over a 0..height permutation (n>height => null).
;   drop_null: keep rows with NO null in the subset (all columns if sn==0).
;   fill_null: in-place per-null value write + validity bit set; STR => 8.
;   unique  : inline open-addressing set (power-of-two, mask, ~<=50% load) over
;             per-row FNV-1a of the subset key columns; first-appearance output
;             order; keep_first vs keep_last updates the representative row.
;   is_unique/is_duplicated: build a key->count map over ALL columns (2 passes),
;             emit a BOOL mask (count==1 vs count>1) into a caller BOOL series.
; Row hashing/equality fold null-flag + value bytes (fixed) or len+bytes (STR).
; Float NaN compares bitwise (same bits => equal); documented.
;
; Errors i32: 0 OK, 1 NULL_PTR, 5 NOT_FOUND, 8 INVALID_ARG. ptr-returning ops
; signal failure with null. No atomics (single-threaded engine).

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i32 @memcmp(ptr captures(none), ptr captures(none), i64)
declare i32 @universe_dataframe_series_str_get(ptr, i64, ptr, ptr)

@sel.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"

; ---------------------------------------------------------------------------
; low-level helpers (mirror frame.ll shapes)
; ---------------------------------------------------------------------------

define internal i64 @sel_fixed_width(i32 %dtype) #5 {
entry:
  %i = zext i32 %dtype to i64
  %p = getelementptr inbounds [6 x i8], ptr @sel.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

define internal ptr @sel_xmalloc(i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

define internal ptr @sel_hdr_alloc() #1 {
entry:
  %p = call ptr @malloc(i64 56)
  %n = icmp eq ptr %p, null
  br i1 %n, label %done, label %zero, !prof !0
zero:
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 56, i1 false)
  br label %done
done:
  ret ptr %p
}

define internal ptr @sel_bm_alloc(i64 %len) #1 {
entry:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %p = call ptr @sel_xmalloc(i64 %nb)
  call void @llvm.memset.p0.i64(ptr %p, i8 -1, i64 %nb, i1 false)
  ret ptr %p
}

define internal void @sel_bm_clear(ptr %bm, i64 %k) #4 {
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

define internal void @sel_bm_set(ptr %bm, i64 %k) #4 {
entry:
  %bi = lshr i64 %k, 3
  %bp = getelementptr inbounds i8, ptr %bm, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %k, 7
  %sh8 = trunc i64 %sh to i8
  %m = shl i8 1, %sh8
  %b2 = or i8 %b, %m
  store i8 %b2, ptr %bp, align 1
  ret void
}

define internal i1 @sel_valid_at(ptr %s, i64 %i) #6 {
entry:
  %vp = getelementptr inbounds i8, ptr %s, i64 32
  %v = load ptr, ptr %vp, align 8
  %vn = icmp eq ptr %v, null
  br i1 %vn, label %valid, label %check
valid:
  ret i1 true
check:
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

define internal i64 @sel_colidx(ptr %cols, i64 %c) #7 {
entry:
  %n = icmp eq ptr %cols, null
  br i1 %n, label %id, label %arr
id:
  ret i64 %c
arr:
  %p = getelementptr inbounds i64, ptr %cols, i64 %c
  %v = load i64, ptr %p, align 8
  ret i64 %v
}

; smallest power of two >= max(2*n, 8)
define internal i64 @sel_pow2(i64 %n) #3 {
entry:
  %n2 = shl i64 %n, 1
  %m = call i64 @llvm.umax.i64(i64 %n2, i64 8)
  %m1 = sub i64 %m, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %m1, i1 false)
  %sh = sub i64 64, %lz
  %p = shl i64 1, %sh
  ret i64 %p
}

; splitmix64: advance *statep, return next value
define internal i64 @sel_rng(ptr %statep) #0 {
entry:
  %s = load i64, ptr %statep, align 8
  %s2 = add i64 %s, 11400714819323198485
  store i64 %s2, ptr %statep, align 8
  %z1a = lshr i64 %s2, 30
  %z1 = xor i64 %s2, %z1a
  %z2 = mul i64 %z1, 13787848793156543929
  %z3a = lshr i64 %z2, 27
  %z3 = xor i64 %z2, %z3a
  %z4 = mul i64 %z3, 10723151780598845931
  %z5a = lshr i64 %z4, 31
  %z5 = xor i64 %z4, %z5a
  ret i64 %z5
}

; ---------------------------------------------------------------------------
; series gather: NEW series = rows selected by idx[0..n) (i64 each, all valid
; in-range; callers guarantee this). fixed + STR + validity.
; ---------------------------------------------------------------------------
define internal ptr @sel_series_gather(ptr %s, ptr %idx, i64 %n) #1 {
entry:
  %dtype = load i32, ptr %s, align 8
  %out = call ptr @sel_hdr_alloc()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %head, !prof !0
head:
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %n, ptr %olp, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %w = call i64 @sel_fixed_width(i32 %dtype)
  %fbytes = mul i64 %n, %w
  %fvals = call ptr @sel_xmalloc(i64 %fbytes)
  %fvpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %fvals, ptr %fvpp, align 8
  %svpp = getelementptr inbounds i8, ptr %s, i64 24
  %svals = load ptr, ptr %svpp, align 8
  br label %f1.head
f1.head:
  %f1.k = phi i64 [ 0, %fixed ], [ %f1.kn, %f1.cont ]
  %f1.nulls = phi i64 [ 0, %fixed ], [ %f1.nn, %f1.cont ]
  %f1.cmp = icmp ult i64 %f1.k, %n
  br i1 %f1.cmp, label %f1.body, label %f1.done
f1.body:
  %f1.jp = getelementptr inbounds i64, ptr %idx, i64 %f1.k
  %f1.j = load i64, ptr %f1.jp, align 8
  %f1.valid = call i1 @sel_valid_at(ptr %s, i64 %f1.j)
  %f1.isnull = xor i1 %f1.valid, true
  br label %f1.cont
f1.cont:
  %f1.inc = zext i1 %f1.isnull to i64
  %f1.nn = add i64 %f1.nulls, %f1.inc
  %f1.kn = add i64 %f1.k, 1
  br label %f1.head
f1.done:
  %f.has = icmp ugt i64 %f1.nulls, 0
  br i1 %f.has, label %f.mkbm, label %f2.head
f.mkbm:
  %f.bm = call ptr @sel_bm_alloc(i64 %n)
  %f.bmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %f.bm, ptr %f.bmp, align 8
  %f.ncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %f1.nulls, ptr %f.ncp, align 8
  br label %f2.head
f2.head:
  %f2.bm = phi ptr [ null, %f1.done ], [ %f.bm, %f.mkbm ]
  br label %f2.loop
f2.loop:
  %f2.k = phi i64 [ 0, %f2.head ], [ %f2.kn, %f2.cont ]
  %f2.cmp = icmp ult i64 %f2.k, %n
  br i1 %f2.cmp, label %f2.body, label %done
f2.body:
  %f2.off = mul i64 %f2.k, %w
  %f2.dst = getelementptr inbounds i8, ptr %fvals, i64 %f2.off
  %f2.jp = getelementptr inbounds i64, ptr %idx, i64 %f2.k
  %f2.j = load i64, ptr %f2.jp, align 8
  %f2.soff = mul i64 %f2.j, %w
  %f2.src = getelementptr inbounds i8, ptr %svals, i64 %f2.soff
  call void @llvm.memcpy.p0.p0.i64(ptr %f2.dst, ptr %f2.src, i64 %w, i1 false)
  %f2.valid = call i1 @sel_valid_at(ptr %s, i64 %f2.j)
  br i1 %f2.valid, label %f2.cont, label %f2.mknull
f2.mknull:
  call void @sel_bm_clear(ptr %f2.bm, i64 %f2.k)
  br label %f2.cont
f2.cont:
  %f2.kn = add i64 %f2.k, 1
  br label %f2.loop

str:
  %s.svpp = getelementptr inbounds i8, ptr %s, i64 24
  %s.soffs = load ptr, ptr %s.svpp, align 8
  %s.ssdp = getelementptr inbounds i8, ptr %s, i64 40
  %s.ssd = load ptr, ptr %s.ssdp, align 8
  br label %s1.head
s1.head:
  %s1.k = phi i64 [ 0, %str ], [ %s1.kn, %s1.cont ]
  %s1.total = phi i64 [ 0, %str ], [ %s1.tn, %s1.cont ]
  %s1.nulls = phi i64 [ 0, %str ], [ %s1.nn, %s1.cont ]
  %s1.cmp = icmp ult i64 %s1.k, %n
  br i1 %s1.cmp, label %s1.body, label %s1.done
s1.body:
  %s1.jp = getelementptr inbounds i64, ptr %idx, i64 %s1.k
  %s1.j = load i64, ptr %s1.jp, align 8
  %s1.j1 = add i64 %s1.j, 1
  %s1.o0p = getelementptr inbounds i32, ptr %s.soffs, i64 %s1.j
  %s1.o0 = load i32, ptr %s1.o0p, align 4
  %s1.o1p = getelementptr inbounds i32, ptr %s.soffs, i64 %s1.j1
  %s1.o1 = load i32, ptr %s1.o1p, align 4
  %s1.o0z = zext i32 %s1.o0 to i64
  %s1.o1z = zext i32 %s1.o1 to i64
  %s1.slen = sub i64 %s1.o1z, %s1.o0z
  %s1.valid = call i1 @sel_valid_at(ptr %s, i64 %s1.j)
  %s1.isnull = xor i1 %s1.valid, true
  br label %s1.cont
s1.cont:
  %s1.tn = add i64 %s1.total, %s1.slen
  %s1.ninc = zext i1 %s1.isnull to i64
  %s1.nn = add i64 %s1.nulls, %s1.ninc
  %s1.kn = add i64 %s1.k, 1
  br label %s1.head
s1.done:
  %s.slots = add i64 %n, 1
  %s.obytes = mul i64 %s.slots, 4
  %s.offs = call ptr @sel_xmalloc(i64 %s.obytes)
  %s.ovpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %s.offs, ptr %s.ovpp, align 8
  %s.sd = call ptr @sel_xmalloc(i64 %s1.total)
  %s.sdp = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %s.sd, ptr %s.sdp, align 8
  %s.sdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %s1.total, ptr %s.sdlp, align 8
  %s.has = icmp ugt i64 %s1.nulls, 0
  br i1 %s.has, label %s.mkbm, label %s2.head
s.mkbm:
  %s.bm = call ptr @sel_bm_alloc(i64 %n)
  %s.bmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %s.bm, ptr %s.bmp, align 8
  %s.ncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %s1.nulls, ptr %s.ncp, align 8
  br label %s2.head
s2.head:
  %s2.bm = phi ptr [ null, %s1.done ], [ %s.bm, %s.mkbm ]
  br label %s2.loop
s2.loop:
  %s2.k = phi i64 [ 0, %s2.head ], [ %s2.kn, %s2.cont ]
  %s2.cur = phi i64 [ 0, %s2.head ], [ %s2.curn, %s2.cont ]
  %s2.cmp = icmp ult i64 %s2.k, %n
  br i1 %s2.cmp, label %s2.body, label %s2.fin
s2.body:
  %s2.op = getelementptr inbounds i32, ptr %s.offs, i64 %s2.k
  %s2.cur32 = trunc i64 %s2.cur to i32
  store i32 %s2.cur32, ptr %s2.op, align 4
  %s2.jp = getelementptr inbounds i64, ptr %idx, i64 %s2.k
  %s2.j = load i64, ptr %s2.jp, align 8
  %s2.j1 = add i64 %s2.j, 1
  %s2.o0p = getelementptr inbounds i32, ptr %s.soffs, i64 %s2.j
  %s2.o0 = load i32, ptr %s2.o0p, align 4
  %s2.o1p = getelementptr inbounds i32, ptr %s.soffs, i64 %s2.j1
  %s2.o1 = load i32, ptr %s2.o1p, align 4
  %s2.o0z = zext i32 %s2.o0 to i64
  %s2.o1z = zext i32 %s2.o1 to i64
  %s2.slen = sub i64 %s2.o1z, %s2.o0z
  %s2.src = getelementptr inbounds i8, ptr %s.ssd, i64 %s2.o0z
  %s2.dst = getelementptr inbounds i8, ptr %s.sd, i64 %s2.cur
  call void @llvm.memcpy.p0.p0.i64(ptr %s2.dst, ptr %s2.src, i64 %s2.slen, i1 false)
  %s2.valid = call i1 @sel_valid_at(ptr %s, i64 %s2.j)
  br i1 %s2.valid, label %s2.cont, label %s2.mknull
s2.mknull:
  call void @sel_bm_clear(ptr %s2.bm, i64 %s2.k)
  br label %s2.cont
s2.cont:
  %s2.curn = add i64 %s2.cur, %s2.slen
  %s2.kn = add i64 %s2.k, 1
  br label %s2.loop
s2.fin:
  %s2.lastp = getelementptr inbounds i32, ptr %s.offs, i64 %n
  %s2.last32 = trunc i64 %s2.cur to i32
  store i32 %s2.last32, ptr %s2.lastp, align 4
  br label %done

done:
  ret ptr %out
fail:
  ret ptr null
}

; ---------------------------------------------------------------------------
; dataframe build helpers
; ---------------------------------------------------------------------------
define internal ptr @sel_df_new(i64 %height, i64 %capcols) #1 {
entry:
  %cap = call i64 @llvm.umax.i64(i64 %capcols, i64 1)
  %hdr = call ptr @malloc(i64 40)
  %hn = icmp eq ptr %hdr, null
  br i1 %hn, label %fail, label %bufs, !prof !0
bufs:
  %nb = mul i64 %cap, 16
  %cb = mul i64 %cap, 8
  %names = call ptr @malloc(i64 %nb)
  %nn = icmp eq ptr %names, null
  br i1 %nn, label %freehdr, label %cols, !prof !0
cols:
  %columns = call ptr @malloc(i64 %cb)
  %cn = icmp eq ptr %columns, null
  br i1 %cn, label %freenames, label %init, !prof !0
init:
  store i64 0, ptr %hdr, align 8
  %capp = getelementptr inbounds i8, ptr %hdr, i64 8
  store i64 %cap, ptr %capp, align 8
  %hp = getelementptr inbounds i8, ptr %hdr, i64 16
  store i64 %height, ptr %hp, align 8
  %np = getelementptr inbounds i8, ptr %hdr, i64 24
  store ptr %names, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %hdr, i64 32
  store ptr %columns, ptr %cp, align 8
  ret ptr %hdr
freenames:
  call void @free(ptr %names)
  br label %freehdr
freehdr:
  call void @free(ptr %hdr)
  br label %fail
fail:
  ret ptr null
}

; append column (TAKES OWNERSHIP of series, COPIES name). Preallocated cap
; guarantees no growth; enforces len==height.
define internal i32 @sel_df_append(ptr %df, ptr %name, i64 %nl, ptr %series) #1 {
entry:
  %nc = load i64, ptr %df, align 8
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %slp = getelementptr inbounds i8, ptr %series, i64 8
  %slen = load i64, ptr %slp, align 8
  %isfirst = icmp eq i64 %nc, 0
  %h0 = icmp eq i64 %height, 0
  %setheight = and i1 %isfirst, %h0
  br i1 %setheight, label %seth, label %checkh
seth:
  store i64 %slen, ptr %hp, align 8
  br label %store
checkh:
  %hmatch = icmp eq i64 %slen, %height
  br i1 %hmatch, label %store, label %errarg, !prof !0
errarg:
  ret i32 8
store:
  %namecopy = call ptr @sel_xmalloc(i64 %nl)
  call void @llvm.memcpy.p0.p0.i64(ptr %namecopy, ptr %name, i64 %nl, i1 false)
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %noff = mul i64 %nc, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  store ptr %namecopy, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  store i64 %nl, ptr %nlenslot, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %nc
  store ptr %series, ptr %cslot, align 8
  %ncn = add i64 %nc, 1
  store i64 %ncn, ptr %df, align 8
  ret i32 0
}

; gather ALL columns of df by idx[0..n) -> NEW df of height n
define internal ptr @sel_df_gather(ptr %df, ptr %idx, i64 %n) #1 {
entry:
  %nc = load i64, ptr %df, align 8
  %out = call ptr @sel_df_new(i64 %n, i64 %nc)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %setup, !prof !0
setup:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop
loop:
  %k = phi i64 [ 0, %setup ], [ %kn, %cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %body, label %done
body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @sel_series_gather(ptr %col, ptr %idx, i64 %n)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlp = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlp, align 8
  %rc = call i32 @sel_df_append(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %cont
cont:
  %kn = add i64 %k, 1
  br label %loop
done:
  ret ptr %out
fail:
  ret ptr null
}

; find column index by name; returns index or -1
define internal i64 @sel_df_find(ptr %df, ptr %name, i64 %nl) #3 {
entry:
  %nc = load i64, ptr %df, align 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  br label %loop
loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %body, label %notfound
body:
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %lm = icmp eq i64 %nlen, %nl
  br i1 %lm, label %cmpb, label %cont
cmpb:
  %r = call i32 @memcmp(ptr %nptr, ptr %name, i64 %nl)
  %eq = icmp eq i32 %r, 0
  br i1 %eq, label %found, label %cont
found:
  ret i64 %k
cont:
  %kn = add i64 %k, 1
  br label %loop
notfound:
  ret i64 -1
}

; ---------------------------------------------------------------------------
; per-row key hash (FNV-1a) over subset columns (colsptr==null => identity 0..n)
; ---------------------------------------------------------------------------
define internal i64 @sel_row_hash(ptr %df, ptr %colsptr, i64 %ncols, i64 %row) #1 {
entry:
  %pslot = alloca ptr, align 8
  %lslot = alloca i64, align 8
  %colsp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %colsp, align 8
  br label %loop
loop:
  %c = phi i64 [ 0, %entry ], [ %cn, %cont ]
  %h = phi i64 [ 14695981039346656037, %entry ], [ %hnext, %cont ]
  %cmp = icmp ult i64 %c, %ncols
  br i1 %cmp, label %body, label %done
body:
  %ci = call i64 @sel_colidx(ptr %colsptr, i64 %c)
  %colslot = getelementptr inbounds ptr, ptr %cols, i64 %ci
  %col = load ptr, ptr %colslot, align 8
  %valid = call i1 @sel_valid_at(ptr %col, i64 %row)
  br i1 %valid, label %hval, label %hnull
hnull:
  %hn1 = xor i64 %h, 255
  %hn2 = mul i64 %hn1, 1099511628211
  br label %cont
hval:
  %hp1 = xor i64 %h, 1
  %hp2 = mul i64 %hp1, 1099511628211
  %dtype = load i32, ptr %col, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %pstr, label %pfix
pfix:
  %w = call i64 @sel_fixed_width(i32 %dtype)
  %valsp = getelementptr inbounds i8, ptr %col, i64 24
  %vals = load ptr, ptr %valsp, align 8
  %off = mul i64 %row, %w
  %base = getelementptr inbounds i8, ptr %vals, i64 %off
  br label %fixloop
fixloop:
  %bi = phi i64 [ 0, %pfix ], [ %bin, %fixbody ]
  %hf = phi i64 [ %hp2, %pfix ], [ %hfn, %fixbody ]
  %fcmp = icmp ult i64 %bi, %w
  br i1 %fcmp, label %fixbody, label %endfix
fixbody:
  %bp = getelementptr inbounds i8, ptr %base, i64 %bi
  %bb = load i8, ptr %bp, align 1
  %bz = zext i8 %bb to i64
  %hx = xor i64 %hf, %bz
  %hfn = mul i64 %hx, 1099511628211
  %bin = add i64 %bi, 1
  br label %fixloop
endfix:
  br label %cont
pstr:
  %rc = call i32 @universe_dataframe_series_str_get(ptr %col, i64 %row, ptr %pslot, ptr %lslot)
  %sp = load ptr, ptr %pslot, align 8
  %sl = load i64, ptr %lslot, align 8
  %hlx = xor i64 %hp2, %sl
  %hl = mul i64 %hlx, 1099511628211
  br label %strloop
strloop:
  %si = phi i64 [ 0, %pstr ], [ %sin, %strbody ]
  %hs = phi i64 [ %hl, %pstr ], [ %hsn, %strbody ]
  %scmp = icmp ult i64 %si, %sl
  br i1 %scmp, label %strbody, label %endstr
strbody:
  %sbp = getelementptr inbounds i8, ptr %sp, i64 %si
  %sb = load i8, ptr %sbp, align 1
  %sbz = zext i8 %sb to i64
  %sx = xor i64 %hs, %sbz
  %hsn = mul i64 %sx, 1099511628211
  %sin = add i64 %si, 1
  br label %strloop
endstr:
  br label %cont
cont:
  %hnext = phi i64 [ %hn2, %hnull ], [ %hf, %endfix ], [ %hs, %endstr ]
  %cn = add i64 %c, 1
  br label %loop
done:
  ret i64 %h
}

; per-row key equality over subset columns
define internal i1 @sel_row_eq(ptr %df, ptr %colsptr, i64 %ncols, i64 %r1, i64 %r2) #1 {
entry:
  %p1 = alloca ptr, align 8
  %l1 = alloca i64, align 8
  %p2 = alloca ptr, align 8
  %l2 = alloca i64, align 8
  %colsp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %colsp, align 8
  br label %loop
loop:
  %c = phi i64 [ 0, %entry ], [ %cn, %cont ]
  %cmp = icmp ult i64 %c, %ncols
  br i1 %cmp, label %body, label %eq
body:
  %ci = call i64 @sel_colidx(ptr %colsptr, i64 %c)
  %colslot = getelementptr inbounds ptr, ptr %cols, i64 %ci
  %col = load ptr, ptr %colslot, align 8
  %v1 = call i1 @sel_valid_at(ptr %col, i64 %r1)
  %v2 = call i1 @sel_valid_at(ptr %col, i64 %r2)
  %vne = xor i1 %v1, %v2
  br i1 %vne, label %neq, label %vsame
vsame:
  br i1 %v1, label %cmpval, label %cont
cmpval:
  %dtype = load i32, ptr %col, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %cstr, label %cfix
cfix:
  %w = call i64 @sel_fixed_width(i32 %dtype)
  %valsp = getelementptr inbounds i8, ptr %col, i64 24
  %vals = load ptr, ptr %valsp, align 8
  %o1 = mul i64 %r1, %w
  %o2 = mul i64 %r2, %w
  %a = getelementptr inbounds i8, ptr %vals, i64 %o1
  %b = getelementptr inbounds i8, ptr %vals, i64 %o2
  %mc = call i32 @memcmp(ptr %a, ptr %b, i64 %w)
  %mne = icmp ne i32 %mc, 0
  br i1 %mne, label %neq, label %cont
cstr:
  %rc1 = call i32 @universe_dataframe_series_str_get(ptr %col, i64 %r1, ptr %p1, ptr %l1)
  %rc2 = call i32 @universe_dataframe_series_str_get(ptr %col, i64 %r2, ptr %p2, ptr %l2)
  %sp1 = load ptr, ptr %p1, align 8
  %sl1 = load i64, ptr %l1, align 8
  %sp2 = load ptr, ptr %p2, align 8
  %sl2 = load i64, ptr %l2, align 8
  %lne = icmp ne i64 %sl1, %sl2
  br i1 %lne, label %neq, label %cmpbytes
cmpbytes:
  %zl = icmp eq i64 %sl1, 0
  br i1 %zl, label %cont, label %docmp
docmp:
  %mc2 = call i32 @memcmp(ptr %sp1, ptr %sp2, i64 %sl1)
  %mne2 = icmp ne i32 %mc2, 0
  br i1 %mne2, label %neq, label %cont
cont:
  %cn = add i64 %c, 1
  br label %loop
eq:
  ret i1 true
neq:
  ret i1 false
}

; ---------------------------------------------------------------------------
; public API
; ---------------------------------------------------------------------------

; filter(df, mask BOOL series) -> NEW df of rows where mask is true & valid
define ptr @universe_dataframe_filter(ptr %df, ptr %mask) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %mn = icmp eq ptr %mask, null
  %bad = or i1 %dn, %mn
  br i1 %bad, label %fail, label %chk, !prof !0
chk:
  %mdt = load i32, ptr %mask, align 8
  %notbool = icmp ne i32 %mdt, 4
  br i1 %notbool, label %fail, label %chklen, !prof !0
chklen:
  %mlp = getelementptr inbounds i8, ptr %mask, i64 8
  %mlen = load i64, ptr %mlp, align 8
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %lne = icmp ne i64 %mlen, %height
  br i1 %lne, label %fail, label %build, !prof !0
build:
  %ib = mul i64 %height, 8
  %idx = call ptr @sel_xmalloc(i64 %ib)
  %mvp = getelementptr inbounds i8, ptr %mask, i64 24
  %mvals = load ptr, ptr %mvp, align 8
  br label %loop
loop:
  %r = phi i64 [ 0, %build ], [ %rn, %cont ]
  %cnt = phi i64 [ 0, %build ], [ %cntn, %cont ]
  %cmp = icmp ult i64 %r, %height
  br i1 %cmp, label %body, label %doneL
body:
  %bp = getelementptr inbounds i8, ptr %mvals, i64 %r
  %bv = load i8, ptr %bp, align 1
  %nz = icmp ne i8 %bv, 0
  %valid = call i1 @sel_valid_at(ptr %mask, i64 %r)
  %keep = and i1 %nz, %valid
  br i1 %keep, label %sel, label %cont
sel:
  %sp = getelementptr inbounds i64, ptr %idx, i64 %cnt
  store i64 %r, ptr %sp, align 8
  %cnt1 = add i64 %cnt, 1
  br label %cont
cont:
  %cntn = phi i64 [ %cnt, %body ], [ %cnt1, %sel ]
  %rn = add i64 %r, 1
  br label %loop
doneL:
  %out = call ptr @sel_df_gather(ptr %df, ptr %idx, i64 %cnt)
  call void @free(ptr %idx)
  ret ptr %out
fail:
  ret ptr null
}

; take(df, idx i64[], n) -> gather rows by explicit index; OOB => null
define ptr @universe_dataframe_take(ptr %df, ptr %idx, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %in = icmp eq ptr %idx, null
  %bad0 = or i1 %dn, %in
  %nneg = icmp slt i64 %n, 0
  %bad = or i1 %bad0, %nneg
  br i1 %bad, label %fail, label %setup, !prof !0
setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  br label %loop
loop:
  %k = phi i64 [ 0, %setup ], [ %kn, %cont ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %body, label %ok
body:
  %jp = getelementptr inbounds i64, ptr %idx, i64 %k
  %j = load i64, ptr %jp, align 8
  %neg = icmp slt i64 %j, 0
  %oob = icmp sge i64 %j, %height
  %bad2 = or i1 %neg, %oob
  br i1 %bad2, label %fail, label %cont, !prof !0
cont:
  %kn = add i64 %k, 1
  br label %loop
ok:
  %out = call ptr @sel_df_gather(ptr %df, ptr %idx, i64 %n)
  ret ptr %out
fail:
  ret ptr null
}

; sample_n(df, n, with_replacement, seed) -> NEW df of n sampled rows
define ptr @universe_dataframe_sample_n(ptr %df, i64 %n, i32 %with_repl, i64 %seed) local_unnamed_addr #1 {
entry:
  %statep = alloca i64, align 8
  %dn = icmp eq ptr %df, null
  %nneg = icmp slt i64 %n, 0
  %bad = or i1 %dn, %nneg
  br i1 %bad, label %fail, label %chk, !prof !0
chk:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %wr = icmp ne i32 %with_repl, 0
  %tooBig = icmp ugt i64 %n, %height
  %norepl = xor i1 %wr, true
  %failNR = and i1 %norepl, %tooBig
  br i1 %failNR, label %fail, label %chkH, !prof !0
chkH:
  %h0 = icmp eq i64 %height, 0
  %nz = icmp ne i64 %n, 0
  %failH = and i1 %h0, %nz
  br i1 %failH, label %fail, label %alloc, !prof !0
alloc:
  store i64 %seed, ptr %statep, align 8
  %ib = mul i64 %n, 8
  %idx = call ptr @sel_xmalloc(i64 %ib)
  br i1 %wr, label %withrepl, label %norep
withrepl:
  br label %wr.loop
wr.loop:
  %wk = phi i64 [ 0, %withrepl ], [ %wkn, %wr.body ]
  %wcmp = icmp ult i64 %wk, %n
  br i1 %wcmp, label %wr.body, label %gather
wr.body:
  %rv = call i64 @sel_rng(ptr %statep)
  %ri = urem i64 %rv, %height
  %wsp = getelementptr inbounds i64, ptr %idx, i64 %wk
  store i64 %ri, ptr %wsp, align 8
  %wkn = add i64 %wk, 1
  br label %wr.loop
norep:
  %pb = mul i64 %height, 8
  %perm = call ptr @sel_xmalloc(i64 %pb)
  br label %pf.loop
pf.loop:
  %pk = phi i64 [ 0, %norep ], [ %pkn, %pf.body ]
  %pcmp = icmp ult i64 %pk, %height
  br i1 %pcmp, label %pf.body, label %fy
pf.body:
  %pp = getelementptr inbounds i64, ptr %perm, i64 %pk
  store i64 %pk, ptr %pp, align 8
  %pkn = add i64 %pk, 1
  br label %pf.loop
fy:
  br label %fy.loop
fy.loop:
  %fk = phi i64 [ 0, %fy ], [ %fkn, %fy.body ]
  %fcmp = icmp ult i64 %fk, %n
  br i1 %fcmp, label %fy.body, label %fy.done
fy.body:
  %rem = sub i64 %height, %fk
  %rv2 = call i64 @sel_rng(ptr %statep)
  %offr = urem i64 %rv2, %rem
  %jj = add i64 %fk, %offr
  %pfk = getelementptr inbounds i64, ptr %perm, i64 %fk
  %pjj = getelementptr inbounds i64, ptr %perm, i64 %jj
  %vfk = load i64, ptr %pfk, align 8
  %vjj = load i64, ptr %pjj, align 8
  store i64 %vjj, ptr %pfk, align 8
  store i64 %vfk, ptr %pjj, align 8
  %isp = getelementptr inbounds i64, ptr %idx, i64 %fk
  store i64 %vjj, ptr %isp, align 8
  %fkn = add i64 %fk, 1
  br label %fy.loop
fy.done:
  call void @free(ptr %perm)
  br label %gather
gather:
  %out = call ptr @sel_df_gather(ptr %df, ptr %idx, i64 %n)
  call void @free(ptr %idx)
  ret ptr %out
fail:
  ret ptr null
}

; drop_nulls(df, subset {ptr,i64}[], sn) -> NEW df; rows with any null in subset dropped
define ptr @universe_dataframe_drop_nulls(ptr %df, ptr %subset, i64 %sn) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %setup, !prof !0
setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %width = load i64, ptr %df, align 8
  %subn = icmp eq ptr %subset, null
  %sn0 = icmp eq i64 %sn, 0
  %useall = or i1 %subn, %sn0
  br i1 %useall, label %allcols, label %resolve
allcols:
  br label %build
resolve:
  %rb = mul i64 %sn, 8
  %rcols = call ptr @sel_xmalloc(i64 %rb)
  br label %res.loop
res.loop:
  %rk = phi i64 [ 0, %resolve ], [ %rkn, %res.cont ]
  %rcmp = icmp ult i64 %rk, %sn
  br i1 %rcmp, label %res.body, label %build
res.body:
  %noff = mul i64 %rk, 16
  %nslot = getelementptr inbounds i8, ptr %subset, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlp = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlp, align 8
  %fi = call i64 @sel_df_find(ptr %df, ptr %nptr, i64 %nlen)
  %nf = icmp slt i64 %fi, 0
  br i1 %nf, label %res.fail, label %res.store, !prof !0
res.store:
  %rp = getelementptr inbounds i64, ptr %rcols, i64 %rk
  store i64 %fi, ptr %rp, align 8
  br label %res.cont
res.cont:
  %rkn = add i64 %rk, 1
  br label %res.loop
res.fail:
  call void @free(ptr %rcols)
  br label %fail
build:
  %colsptr = phi ptr [ null, %allcols ], [ %rcols, %res.loop ]
  %ncols = phi i64 [ %width, %allcols ], [ %sn, %res.loop ]
  %ib = mul i64 %height, 8
  %idx = call ptr @sel_xmalloc(i64 %ib)
  %colsbase = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %colsbase, align 8
  br label %rloop
rloop:
  %r = phi i64 [ 0, %build ], [ %rn, %rcont ]
  %cnt = phi i64 [ 0, %build ], [ %cntn, %rcont ]
  %rcmp2 = icmp ult i64 %r, %height
  br i1 %rcmp2, label %rbody, label %doneL
rbody:
  br label %cloop
cloop:
  %c = phi i64 [ 0, %rbody ], [ %ccn, %ccont ]
  %ccmp = icmp ult i64 %c, %ncols
  br i1 %ccmp, label %cbody, label %keeprow
cbody:
  %ci = call i64 @sel_colidx(ptr %colsptr, i64 %c)
  %colslot = getelementptr inbounds ptr, ptr %cols, i64 %ci
  %col = load ptr, ptr %colslot, align 8
  %valid = call i1 @sel_valid_at(ptr %col, i64 %r)
  br i1 %valid, label %ccont, label %droprow
ccont:
  %ccn = add i64 %c, 1
  br label %cloop
keeprow:
  %sp = getelementptr inbounds i64, ptr %idx, i64 %cnt
  store i64 %r, ptr %sp, align 8
  %cnt1 = add i64 %cnt, 1
  br label %rcont
droprow:
  br label %rcont
rcont:
  %cntn = phi i64 [ %cnt1, %keeprow ], [ %cnt, %droprow ]
  %rn = add i64 %r, 1
  br label %rloop
doneL:
  %out = call ptr @sel_df_gather(ptr %df, ptr %idx, i64 %cnt)
  call void @free(ptr %idx)
  %allocated = icmp ne ptr %colsptr, null
  br i1 %allocated, label %freecols, label %ret
freecols:
  call void @free(ptr %colsptr)
  br label %ret
ret:
  ret ptr %out
fail:
  ret ptr null
}

; fill_null(df, name, nl, value) -> i32; per-null value write for fixed dtype
define i32 @universe_dataframe_fill_null(ptr %df, ptr %name, i64 %nl, ptr %value) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %vn = icmp eq ptr %value, null
  %bad = or i1 %dn, %vn
  br i1 %bad, label %errnull, label %find, !prof !0
errnull:
  ret i32 1
find:
  %fi = call i64 @sel_df_find(ptr %df, ptr %name, i64 %nl)
  %nf = icmp slt i64 %fi, 0
  br i1 %nf, label %notfound, label %got, !prof !0
notfound:
  ret i32 5
got:
  %colsbase = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %colsbase, align 8
  %colslot = getelementptr inbounds ptr, ptr %cols, i64 %fi
  %col = load ptr, ptr %colslot, align 8
  %dtype = load i32, ptr %col, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %errarg, label %fixed
errarg:
  ret i32 8
fixed:
  %vldp = getelementptr inbounds i8, ptr %col, i64 32
  %vld = load ptr, ptr %vldp, align 8
  %novld = icmp eq ptr %vld, null
  br i1 %novld, label %noop, label %dofill
noop:
  ret i32 0
dofill:
  %w = call i64 @sel_fixed_width(i32 %dtype)
  %valsp = getelementptr inbounds i8, ptr %col, i64 24
  %vals = load ptr, ptr %valsp, align 8
  %lp = getelementptr inbounds i8, ptr %col, i64 8
  %len = load i64, ptr %lp, align 8
  %ncp = getelementptr inbounds i8, ptr %col, i64 16
  %nc0 = load i64, ptr %ncp, align 8
  br label %loop
loop:
  %r = phi i64 [ 0, %dofill ], [ %rn, %lcont ]
  %nc = phi i64 [ %nc0, %dofill ], [ %ncn, %lcont ]
  %cmp = icmp ult i64 %r, %len
  br i1 %cmp, label %lbody, label %fin
lbody:
  %valid = call i1 @sel_valid_at(ptr %col, i64 %r)
  br i1 %valid, label %skip, label %fillit
fillit:
  %off = mul i64 %r, %w
  %dst = getelementptr inbounds i8, ptr %vals, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %value, i64 %w, i1 false)
  call void @sel_bm_set(ptr %vld, i64 %r)
  %nc1 = sub i64 %nc, 1
  br label %lcont
skip:
  br label %lcont
lcont:
  %ncn = phi i64 [ %nc1, %fillit ], [ %nc, %skip ]
  %rn = add i64 %r, 1
  br label %loop
fin:
  store i64 %nc, ptr %ncp, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; unique (open-addressing set over key columns; first-appearance order)
; ---------------------------------------------------------------------------
define internal ptr @sel_unique_core(ptr %df, ptr %subset, i64 %sn, i32 %keep_first) #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %setup, !prof !0
setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %width = load i64, ptr %df, align 8
  %subn = icmp eq ptr %subset, null
  %sn0 = icmp eq i64 %sn, 0
  %useall = or i1 %subn, %sn0
  br i1 %useall, label %allcols, label %resolve
allcols:
  br label %build
resolve:
  %rb = mul i64 %sn, 8
  %rcols = call ptr @sel_xmalloc(i64 %rb)
  br label %res.loop
res.loop:
  %rk = phi i64 [ 0, %resolve ], [ %rkn, %res.cont ]
  %rcmp = icmp ult i64 %rk, %sn
  br i1 %rcmp, label %res.body, label %build
res.body:
  %noff = mul i64 %rk, 16
  %nslot = getelementptr inbounds i8, ptr %subset, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlp = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlp, align 8
  %fi = call i64 @sel_df_find(ptr %df, ptr %nptr, i64 %nlen)
  %nf = icmp slt i64 %fi, 0
  br i1 %nf, label %res.fail, label %res.store, !prof !0
res.store:
  %rp = getelementptr inbounds i64, ptr %rcols, i64 %rk
  store i64 %fi, ptr %rp, align 8
  br label %res.cont
res.cont:
  %rkn = add i64 %rk, 1
  br label %res.loop
res.fail:
  call void @free(ptr %rcols)
  br label %fail
build:
  %colsptr = phi ptr [ null, %allcols ], [ %rcols, %res.loop ]
  %ncols = phi i64 [ %width, %allcols ], [ %sn, %res.loop ]
  %cap = call i64 @sel_pow2(i64 %height)
  %mask = sub i64 %cap, 1
  %sb = mul i64 %cap, 8
  %slots = call ptr @sel_xmalloc(i64 %sb)
  call void @llvm.memset.p0.i64(ptr %slots, i8 -1, i64 %sb, i1 false)
  %kb0 = call i64 @llvm.umax.i64(i64 %height, i64 1)
  %kb = mul i64 %kb0, 8
  %kept = call ptr @sel_xmalloc(i64 %kb)
  br label %loop
loop:
  %r = phi i64 [ 0, %build ], [ %rn, %rcont ]
  %nkept = phi i64 [ 0, %build ], [ %nkeptn, %rcont ]
  %cmp = icmp ult i64 %r, %height
  br i1 %cmp, label %rbody, label %doneL
rbody:
  %hraw = call i64 @sel_row_hash(ptr %df, ptr %colsptr, i64 %ncols, i64 %r)
  %h0 = and i64 %hraw, %mask
  br label %probe
probe:
  %h = phi i64 [ %h0, %rbody ], [ %hn, %pcont ]
  %sslot = getelementptr inbounds i64, ptr %slots, i64 %h
  %sv = load i64, ptr %sslot, align 8
  %empty = icmp eq i64 %sv, -1
  br i1 %empty, label %insert, label %occupied
insert:
  store i64 %nkept, ptr %sslot, align 8
  %kp = getelementptr inbounds i64, ptr %kept, i64 %nkept
  store i64 %r, ptr %kp, align 8
  %nkept1 = add i64 %nkept, 1
  br label %rcont
occupied:
  %repp = getelementptr inbounds i64, ptr %kept, i64 %sv
  %rep = load i64, ptr %repp, align 8
  %eq = call i1 @sel_row_eq(ptr %df, ptr %colsptr, i64 %ncols, i64 %rep, i64 %r)
  br i1 %eq, label %match, label %pcont
match:
  %kf = icmp ne i32 %keep_first, 0
  br i1 %kf, label %rcont, label %updrep
updrep:
  store i64 %r, ptr %repp, align 8
  br label %rcont
pcont:
  %hinc = add i64 %h, 1
  %hn = and i64 %hinc, %mask
  br label %probe
rcont:
  %nkeptn = phi i64 [ %nkept1, %insert ], [ %nkept, %match ], [ %nkept, %updrep ]
  %rn = add i64 %r, 1
  br label %loop
doneL:
  %out = call ptr @sel_df_gather(ptr %df, ptr %kept, i64 %nkept)
  call void @free(ptr %slots)
  call void @free(ptr %kept)
  %alloc = icmp ne ptr %colsptr, null
  br i1 %alloc, label %freecols, label %ret
freecols:
  call void @free(ptr %colsptr)
  br label %ret
ret:
  ret ptr %out
fail:
  ret ptr null
}

define ptr @universe_dataframe_unique(ptr %df, ptr %subset, i64 %sn, i32 %keep_first) local_unnamed_addr #1 {
entry:
  %r = call ptr @sel_unique_core(ptr %df, ptr %subset, i64 %sn, i32 %keep_first)
  ret ptr %r
}

define ptr @universe_dataframe_unique_stable(ptr %df, ptr %subset, i64 %sn) local_unnamed_addr #1 {
entry:
  %r = call ptr @sel_unique_core(ptr %df, ptr %subset, i64 %sn, i32 1)
  ret ptr %r
}

; ---------------------------------------------------------------------------
; is_unique / is_duplicated (key->count map over ALL columns -> BOOL mask)
; ---------------------------------------------------------------------------
define internal i32 @sel_dupmask(ptr %df, ptr %out_mask, i32 %want_unique) #1 {
entry:
  %dn = icmp eq ptr %df, null
  %on = icmp eq ptr %out_mask, null
  %bad = or i1 %dn, %on
  br i1 %bad, label %errnull, label %chk, !prof !0
errnull:
  ret i32 1
chk:
  %odt = load i32, ptr %out_mask, align 8
  %notbool = icmp ne i32 %odt, 4
  br i1 %notbool, label %errarg, label %chklen, !prof !0
errarg:
  ret i32 8
chklen:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %olp = getelementptr inbounds i8, ptr %out_mask, i64 8
  %olen = load i64, ptr %olp, align 8
  %lne = icmp ne i64 %olen, %height
  br i1 %lne, label %errarg, label %setup, !prof !0
setup:
  %width = load i64, ptr %df, align 8
  %cap = call i64 @sel_pow2(i64 %height)
  %maskv = sub i64 %cap, 1
  %sb = mul i64 %cap, 8
  %slots = call ptr @sel_xmalloc(i64 %sb)
  call void @llvm.memset.p0.i64(ptr %slots, i8 -1, i64 %sb, i1 false)
  %scnt = call ptr @sel_xmalloc(i64 %sb)
  call void @llvm.memset.p0.i64(ptr %scnt, i8 0, i64 %sb, i1 false)
  br label %p1.loop
p1.loop:
  %r = phi i64 [ 0, %setup ], [ %rn, %p1.cont ]
  %cmp = icmp ult i64 %r, %height
  br i1 %cmp, label %p1.body, label %p2.init
p1.body:
  %hraw = call i64 @sel_row_hash(ptr %df, ptr null, i64 %width, i64 %r)
  %h0 = and i64 %hraw, %maskv
  br label %p1.probe
p1.probe:
  %h = phi i64 [ %h0, %p1.body ], [ %hn, %p1.pc ]
  %sslot = getelementptr inbounds i64, ptr %slots, i64 %h
  %sv = load i64, ptr %sslot, align 8
  %empty = icmp eq i64 %sv, -1
  br i1 %empty, label %p1.ins, label %p1.occ
p1.ins:
  store i64 %r, ptr %sslot, align 8
  %cntslot0 = getelementptr inbounds i64, ptr %scnt, i64 %h
  store i64 1, ptr %cntslot0, align 8
  br label %p1.cont
p1.occ:
  %eq = call i1 @sel_row_eq(ptr %df, ptr null, i64 %width, i64 %sv, i64 %r)
  br i1 %eq, label %p1.bump, label %p1.pc
p1.bump:
  %cntslot = getelementptr inbounds i64, ptr %scnt, i64 %h
  %cv = load i64, ptr %cntslot, align 8
  %cv1 = add i64 %cv, 1
  store i64 %cv1, ptr %cntslot, align 8
  br label %p1.cont
p1.pc:
  %hinc = add i64 %h, 1
  %hn = and i64 %hinc, %maskv
  br label %p1.probe
p1.cont:
  %rn = add i64 %r, 1
  br label %p1.loop
p2.init:
  %ovp = getelementptr inbounds i8, ptr %out_mask, i64 24
  %ovals = load ptr, ptr %ovp, align 8
  %wu = icmp ne i32 %want_unique, 0
  %notwu = xor i1 %wu, true
  br label %p2.loop
p2.loop:
  %r2 = phi i64 [ 0, %p2.init ], [ %r2n, %p2.cont ]
  %cmp2 = icmp ult i64 %r2, %height
  br i1 %cmp2, label %p2.body, label %fin
p2.body:
  %hraw2 = call i64 @sel_row_hash(ptr %df, ptr null, i64 %width, i64 %r2)
  %h20 = and i64 %hraw2, %maskv
  br label %p2.probe
p2.probe:
  %hh = phi i64 [ %h20, %p2.body ], [ %hhn, %p2.pc ]
  %sslot2 = getelementptr inbounds i64, ptr %slots, i64 %hh
  %sv2 = load i64, ptr %sslot2, align 8
  %eq2 = call i1 @sel_row_eq(ptr %df, ptr null, i64 %width, i64 %sv2, i64 %r2)
  br i1 %eq2, label %p2.found, label %p2.pc
p2.found:
  %cntslot2 = getelementptr inbounds i64, ptr %scnt, i64 %hh
  %c = load i64, ptr %cntslot2, align 8
  %isuni = icmp eq i64 %c, 1
  %res = xor i1 %isuni, %notwu
  %res8 = zext i1 %res to i8
  %op = getelementptr inbounds i8, ptr %ovals, i64 %r2
  store i8 %res8, ptr %op, align 1
  br label %p2.cont
p2.pc:
  %hhinc = add i64 %hh, 1
  %hhn = and i64 %hhinc, %maskv
  br label %p2.probe
p2.cont:
  %r2n = add i64 %r2, 1
  br label %p2.loop
fin:
  call void @free(ptr %slots)
  call void @free(ptr %scnt)
  ret i32 0
}

define i32 @universe_dataframe_is_unique(ptr %df, ptr %out_mask) local_unnamed_addr #1 {
entry:
  %r = call i32 @sel_dupmask(ptr %df, ptr %out_mask, i32 1)
  ret i32 %r
}

define i32 @universe_dataframe_is_duplicated(ptr %df, ptr %out_mask) local_unnamed_addr #1 {
entry:
  %r = call i32 @sel_dupmask(ptr %df, ptr %out_mask, i32 0)
  ret i32 %r
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #3 = { nounwind willreturn norecurse nosync }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #5 = { alwaysinline nounwind willreturn norecurse nosync memory(read) }
attributes #6 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #7 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
