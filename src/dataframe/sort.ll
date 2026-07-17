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

; DataFrame ordering: permutation-based sort / sort_by / arg_sort / top_k /
; bottom_k. We NEVER move row bytes to order a frame — we compute an index
; permutation from the key column(s), then apply that permutation to EVERY
; column with a per-dtype gather (out[i] = in[perm[i]], validity permuted too).
; `reverse` is already provided by frame.ll (universe_dataframe_reverse) and is
; intentionally NOT redefined here.
;
; ============================ DESIGN ============================
; Contract (matches frame.ll DOWNSTREAM CONTRACT, offsets relied on):
;   Series header (56B): +0 i32 dtype; +8 i64 len; +16 i64 null_count;
;     +24 ptr values; +32 ptr validity (null=>all valid, bit=1 VALID);
;     +40 ptr strdata; +48 i64 strdata_len. DType: I32=0,I64=1,F32=2,F64=3,
;     BOOL=4(one byte/value),STR=5. STR `values` = i32 offsets[len+1].
;   DataFrame header (40B): +0 n_cols; +8 cap; +16 height; +24 names
;     ({ptr,i64}[16B]); +32 columns (Series ptr[]).
;
; Algorithm class (measured doctrine: integer keys => radix beats compare ~20x):
;   * SINGLE numeric key (I32/I64), no nulls: LSD radix on a sortable-u64
;     transform of the key (stable, 8 passes x 8 bits). Sign bit flipped so
;     signed order == unsigned order; `descending` = bitwise-NOT of the key
;     (a bijection, so equal keys stay equal => stability preserved).
;   * Everything else (multi-key, float, bool, str, or any nulls): a STABLE
;     bottom-up merge sort over the index array, comparing rows through a
;     multi-key comparator (left wins ties => stable). Descending negates the
;     per-key comparison. This is the general, always-correct path; radix is
;     the fast specialisation of it.
;
; Ordering conventions (documented, total order):
;   * NULL is ordered as GREATER than any value (ascending => nulls last).
;   * NaN is ordered as GREATER than any non-NaN number (ascending => NaN last).
;   * `descending` negates the ENTIRE per-key comparison, so under a descending
;     key nulls and NaN lead. Strings compare lexicographically by bytes, the
;     shorter string first on a shared prefix.
;
; top_k / bottom_k: compute the full ordering (argsort), then GATHER only the
;   first min(k,height) rows of the permutation. top_k returns the k LARGEST
;   rows (largest first); its scalar `descending` reverses that to the k
;   smallest. bottom_k returns the k SMALLEST (smallest first). (A bounded
;   O(n log k) heap is a future optimisation; the full-sort-then-head form is
;   used here for correctness and matches a full-sort reference exactly.)
;
; Ownership: sort / top_k / bottom_k build a NEW DataFrame via
;   universe_dataframe_new + _with_column (freshly gathered, owned columns).
;   sort_in_place gathers each column then frees the old + stores the new in
;   the SAME handle (names/height unchanged).
;
; Errors i32: 0 OK, 1 NULL_PTR, 2 OOM, 5 NOT_FOUND (missing key column).
;   Constructors returning ptr signal failure with null.
; ===============================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i32 @memcmp(ptr captures(none), ptr captures(none), i64)

; frame.ll exports we compose with
declare ptr @universe_dataframe_new()
declare ptr @universe_dataframe_column(ptr, ptr, i64)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare void @universe_dataframe_series_free(ptr)

@srt.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"

; ---------------------------------------------------------------------------
; low-level helpers (duplicated inline shapes of frame.ll internals; internal
; linkage => no symbol collision. Kept byte-identical in behaviour.)
; ---------------------------------------------------------------------------

define internal ptr @srt_xmalloc(i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

define internal i64 @srt_fixed_width(i32 %dtype) #0 {
entry:
  %i = zext i32 %dtype to i64
  %p = getelementptr inbounds [6 x i8], ptr @srt.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

define internal { i64, i1 } @srt_valbytes(i32 %dtype, i64 %len) #0 {
entry:
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fix

str:
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %len, i64 1)
  %slots = extractvalue { i64, i1 } %a, 0
  %ao = extractvalue { i64, i1 } %a, 1
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %slots, i64 4)
  %sbytes = extractvalue { i64, i1 } %m, 0
  %mo = extractvalue { i64, i1 } %m, 1
  %sov = or i1 %ao, %mo
  %sr0 = insertvalue { i64, i1 } undef, i64 %sbytes, 0
  %sr = insertvalue { i64, i1 } %sr0, i1 %sov, 1
  ret { i64, i1 } %sr

fix:
  %w = call i64 @srt_fixed_width(i32 %dtype)
  %fm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %len, i64 %w)
  %fbytes = extractvalue { i64, i1 } %fm, 0
  %fo = extractvalue { i64, i1 } %fm, 1
  %fr0 = insertvalue { i64, i1 } undef, i64 %fbytes, 0
  %fr = insertvalue { i64, i1 } %fr0, i1 %fo, 1
  ret { i64, i1 } %fr
}

define internal ptr @srt_hdr_alloc() #0 {
entry:
  %p = call ptr @malloc(i64 56)
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %done, label %zero, !prof !0

zero:
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 56, i1 false)
  br label %done

done:
  ret ptr %p
}

define internal ptr @srt_bm_alloc(i64 %len) #0 {
entry:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %p = call ptr @srt_xmalloc(i64 %nb)
  call void @llvm.memset.p0.i64(ptr %p, i8 -1, i64 %nb, i1 false)
  ret ptr %p
}

define internal void @srt_bm_clear(ptr %bm, i64 %k) #0 {
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

define internal i1 @srt_valid_at(ptr %s, i64 %i) #0 {
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

; ---------------------------------------------------------------------------
; srt_series_gather: NEW series = rows selected by idx[0..n) (i64 each);
; idx entry < 0 => a null / empty element. Monomorphic per dtype. (Duplicated
; inline shape of frame.ll's series_gather.)
; ---------------------------------------------------------------------------
define internal ptr @srt_series_gather(ptr %s, ptr %idx, i64 %n) #0 {
entry:
  %dtype = load i32, ptr %s, align 8
  %out = call ptr @srt_hdr_alloc()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %head, !prof !0

head:
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %n, ptr %olp, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %w = call i64 @srt_fixed_width(i32 %dtype)
  %fbytes = mul i64 %n, %w
  %fvals = call ptr @srt_xmalloc(i64 %fbytes)
  %fvpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %fvals, ptr %fvpp, align 8
  %svpp = getelementptr inbounds i8, ptr %s, i64 24
  %svals = load ptr, ptr %svpp, align 8
  br label %f1.head

f1.head:
  %f1.k = phi i64 [ 0, %fixed ], [ %f1.kn, %f1.isnull ]
  %f1.nulls = phi i64 [ 0, %fixed ], [ %f1.nn, %f1.isnull ]
  %f1.cmp = icmp ult i64 %f1.k, %n
  br i1 %f1.cmp, label %f1.body, label %f1.done

f1.body:
  %f1.jp = getelementptr inbounds i64, ptr %idx, i64 %f1.k
  %f1.j = load i64, ptr %f1.jp, align 8
  %f1.neg = icmp slt i64 %f1.j, 0
  br i1 %f1.neg, label %f1.isnull, label %f1.chk

f1.chk:
  %f1.valid = call i1 @srt_valid_at(ptr %s, i64 %f1.j)
  %f1.srcnull = xor i1 %f1.valid, true
  br label %f1.isnull

f1.isnull:
  %f1.null = phi i1 [ true, %f1.body ], [ %f1.srcnull, %f1.chk ]
  %f1.inc = zext i1 %f1.null to i64
  %f1.nn = add i64 %f1.nulls, %f1.inc
  %f1.kn = add i64 %f1.k, 1
  br label %f1.head

f1.done:
  %f.hasnull = icmp ugt i64 %f1.nulls, 0
  br i1 %f.hasnull, label %f.mkbm, label %f2.head

f.mkbm:
  %f.bm = call ptr @srt_bm_alloc(i64 %n)
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
  %f2.neg = icmp slt i64 %f2.j, 0
  br i1 %f2.neg, label %f2.vacated, label %f2.copy

f2.vacated:
  call void @llvm.memset.p0.i64(ptr %f2.dst, i8 0, i64 %w, i1 false)
  call void @srt_bm_clear(ptr %f2.bm, i64 %f2.k)
  br label %f2.cont

f2.copy:
  %f2.soff = mul i64 %f2.j, %w
  %f2.src = getelementptr inbounds i8, ptr %svals, i64 %f2.soff
  call void @llvm.memcpy.p0.p0.i64(ptr %f2.dst, ptr %f2.src, i64 %w, i1 false)
  %f2.valid = call i1 @srt_valid_at(ptr %s, i64 %f2.j)
  br i1 %f2.valid, label %f2.cont, label %f2.mknull

f2.mknull:
  call void @srt_bm_clear(ptr %f2.bm, i64 %f2.k)
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
  %s1.k = phi i64 [ 0, %str ], [ %s1.kn, %s1.acc ]
  %s1.total = phi i64 [ 0, %str ], [ %s1.tn, %s1.acc ]
  %s1.nulls = phi i64 [ 0, %str ], [ %s1.nn, %s1.acc ]
  %s1.cmp = icmp ult i64 %s1.k, %n
  br i1 %s1.cmp, label %s1.body, label %s1.done

s1.body:
  %s1.jp = getelementptr inbounds i64, ptr %idx, i64 %s1.k
  %s1.j = load i64, ptr %s1.jp, align 8
  %s1.neg = icmp slt i64 %s1.j, 0
  br i1 %s1.neg, label %s1.acc, label %s1.chk

s1.chk:
  %s1.j1 = add i64 %s1.j, 1
  %s1.o0p = getelementptr inbounds i32, ptr %s.soffs, i64 %s1.j
  %s1.o0 = load i32, ptr %s1.o0p, align 4
  %s1.o1p = getelementptr inbounds i32, ptr %s.soffs, i64 %s1.j1
  %s1.o1 = load i32, ptr %s1.o1p, align 4
  %s1.o0z = zext i32 %s1.o0 to i64
  %s1.o1z = zext i32 %s1.o1 to i64
  %s1.slen = sub i64 %s1.o1z, %s1.o0z
  %s1.valid = call i1 @srt_valid_at(ptr %s, i64 %s1.j)
  %s1.srcnull = xor i1 %s1.valid, true
  br label %s1.acc

s1.acc:
  %s1.addbytes = phi i64 [ 0, %s1.body ], [ %s1.slen, %s1.chk ]
  %s1.null = phi i1 [ true, %s1.body ], [ %s1.srcnull, %s1.chk ]
  %s1.tn = add i64 %s1.total, %s1.addbytes
  %s1.ninc = zext i1 %s1.null to i64
  %s1.nn = add i64 %s1.nulls, %s1.ninc
  %s1.kn = add i64 %s1.k, 1
  br label %s1.head

s1.done:
  %s.obr = call { i64, i1 } @srt_valbytes(i32 5, i64 %n)
  %s.obytes = extractvalue { i64, i1 } %s.obr, 0
  %s.offs = call ptr @srt_xmalloc(i64 %s.obytes)
  %s.ovpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %s.offs, ptr %s.ovpp, align 8
  %s.sd = call ptr @srt_xmalloc(i64 %s1.total)
  %s.sdp = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %s.sd, ptr %s.sdp, align 8
  %s.sdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %s1.total, ptr %s.sdlp, align 8
  %s.hasnull = icmp ugt i64 %s1.nulls, 0
  br i1 %s.hasnull, label %s.mkbm, label %s2.head

s.mkbm:
  %s.bm = call ptr @srt_bm_alloc(i64 %n)
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
  %s2.neg = icmp slt i64 %s2.j, 0
  br i1 %s2.neg, label %s2.vacated, label %s2.copy

s2.vacated:
  call void @srt_bm_clear(ptr %s2.bm, i64 %s2.k)
  br label %s2.cont0

s2.copy:
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
  %s2.valid = call i1 @srt_valid_at(ptr %s, i64 %s2.j)
  br i1 %s2.valid, label %s2.cont1, label %s2.mknull

s2.mknull:
  call void @srt_bm_clear(ptr %s2.bm, i64 %s2.k)
  br label %s2.cont1

s2.cont0:
  br label %s2.cont

s2.cont1:
  br label %s2.cont

s2.cont:
  %s2.add = phi i64 [ 0, %s2.cont0 ], [ %s2.slen, %s2.cont1 ]
  %s2.curn = add i64 %s2.cur, %s2.add
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
; srt_cmp_col: base ASCENDING comparison of column %col rows %a vs %b.
; Returns i32 -1/0/1. NULL and NaN order as GREATER than any value.
; ---------------------------------------------------------------------------
define internal i32 @srt_cmp_col(ptr %col, i64 %a, i64 %b) #0 {
entry:
  %va = call i1 @srt_valid_at(ptr %col, i64 %a)
  %vb = call i1 @srt_valid_at(ptr %col, i64 %b)
  br i1 %va, label %a_ok, label %a_null

a_null:
  br i1 %vb, label %ret_pos1, label %ret_zero

a_ok:
  br i1 %vb, label %both_ok, label %ret_neg1

ret_zero:
  ret i32 0
ret_pos1:
  ret i32 1
ret_neg1:
  ret i32 -1

both_ok:
  %dtype = load i32, ptr %col, align 8
  %vpp = getelementptr inbounds i8, ptr %col, i64 24
  %values = load ptr, ptr %vpp, align 8
  switch i32 %dtype, label %d_i32 [ i32 0, label %d_i32
                                    i32 1, label %d_i64
                                    i32 2, label %d_f32
                                    i32 3, label %d_f64
                                    i32 4, label %d_bool
                                    i32 5, label %d_str ]

d_i32:
  %i32.pa = getelementptr inbounds i32, ptr %values, i64 %a
  %i32.la = load i32, ptr %i32.pa, align 4
  %i32.pb = getelementptr inbounds i32, ptr %values, i64 %b
  %i32.lb = load i32, ptr %i32.pb, align 4
  %i32.lt = icmp slt i32 %i32.la, %i32.lb
  %i32.gt = icmp sgt i32 %i32.la, %i32.lb
  %i32.g = select i1 %i32.gt, i32 1, i32 0
  %i32.r = select i1 %i32.lt, i32 -1, i32 %i32.g
  ret i32 %i32.r

d_i64:
  %i64.pa = getelementptr inbounds i64, ptr %values, i64 %a
  %i64.la = load i64, ptr %i64.pa, align 8
  %i64.pb = getelementptr inbounds i64, ptr %values, i64 %b
  %i64.lb = load i64, ptr %i64.pb, align 8
  %i64.lt = icmp slt i64 %i64.la, %i64.lb
  %i64.gt = icmp sgt i64 %i64.la, %i64.lb
  %i64.g = select i1 %i64.gt, i32 1, i32 0
  %i64.r = select i1 %i64.lt, i32 -1, i32 %i64.g
  ret i32 %i64.r

d_bool:
  %bl.pa = getelementptr inbounds i8, ptr %values, i64 %a
  %bl.la = load i8, ptr %bl.pa, align 1
  %bl.pb = getelementptr inbounds i8, ptr %values, i64 %b
  %bl.lb = load i8, ptr %bl.pb, align 1
  %bl.lt = icmp ult i8 %bl.la, %bl.lb
  %bl.gt = icmp ugt i8 %bl.la, %bl.lb
  %bl.g = select i1 %bl.gt, i32 1, i32 0
  %bl.r = select i1 %bl.lt, i32 -1, i32 %bl.g
  ret i32 %bl.r

d_f32:
  %f32.pa = getelementptr inbounds float, ptr %values, i64 %a
  %f32.fa = load float, ptr %f32.pa, align 4
  %f32.pb = getelementptr inbounds float, ptr %values, i64 %b
  %f32.fb = load float, ptr %f32.pb, align 4
  %f32.na = fcmp uno float %f32.fa, %f32.fa
  %f32.nb = fcmp uno float %f32.fb, %f32.fb
  br i1 %f32.na, label %f32_anan, label %f32_anum

f32_anan:
  br i1 %f32.nb, label %ret_zero, label %ret_pos1

f32_anum:
  br i1 %f32.nb, label %ret_neg1, label %f32_cmp

f32_cmp:
  %f32.lt = fcmp olt float %f32.fa, %f32.fb
  %f32.gt = fcmp ogt float %f32.fa, %f32.fb
  %f32.g = select i1 %f32.gt, i32 1, i32 0
  %f32.r = select i1 %f32.lt, i32 -1, i32 %f32.g
  ret i32 %f32.r

d_f64:
  %f64.pa = getelementptr inbounds double, ptr %values, i64 %a
  %f64.fa = load double, ptr %f64.pa, align 8
  %f64.pb = getelementptr inbounds double, ptr %values, i64 %b
  %f64.fb = load double, ptr %f64.pb, align 8
  %f64.na = fcmp uno double %f64.fa, %f64.fa
  %f64.nb = fcmp uno double %f64.fb, %f64.fb
  br i1 %f64.na, label %f64_anan, label %f64_anum

f64_anan:
  br i1 %f64.nb, label %ret_zero, label %ret_pos1

f64_anum:
  br i1 %f64.nb, label %ret_neg1, label %f64_cmp

f64_cmp:
  %f64.lt = fcmp olt double %f64.fa, %f64.fb
  %f64.gt = fcmp ogt double %f64.fa, %f64.fb
  %f64.g = select i1 %f64.gt, i32 1, i32 0
  %f64.r = select i1 %f64.lt, i32 -1, i32 %f64.g
  ret i32 %f64.r

d_str:
  %st.sdp = getelementptr inbounds i8, ptr %col, i64 40
  %st.sd = load ptr, ptr %st.sdp, align 8
  %st.a1 = add i64 %a, 1
  %st.b1 = add i64 %b, 1
  %st.oap = getelementptr inbounds i32, ptr %values, i64 %a
  %st.oa = load i32, ptr %st.oap, align 4
  %st.oa1p = getelementptr inbounds i32, ptr %values, i64 %st.a1
  %st.oa1 = load i32, ptr %st.oa1p, align 4
  %st.obp = getelementptr inbounds i32, ptr %values, i64 %b
  %st.ob = load i32, ptr %st.obp, align 4
  %st.ob1p = getelementptr inbounds i32, ptr %values, i64 %st.b1
  %st.ob1 = load i32, ptr %st.ob1p, align 4
  %st.oaz = zext i32 %st.oa to i64
  %st.oa1z = zext i32 %st.oa1 to i64
  %st.obz = zext i32 %st.ob to i64
  %st.ob1z = zext i32 %st.ob1 to i64
  %st.la = sub i64 %st.oa1z, %st.oaz
  %st.lb = sub i64 %st.ob1z, %st.obz
  %st.pa = getelementptr inbounds i8, ptr %st.sd, i64 %st.oaz
  %st.pb = getelementptr inbounds i8, ptr %st.sd, i64 %st.obz
  %st.min = call i64 @llvm.umin.i64(i64 %st.la, i64 %st.lb)
  %st.mc = call i32 @memcmp(ptr %st.pa, ptr %st.pb, i64 %st.min)
  %st.ne = icmp ne i32 %st.mc, 0
  br i1 %st.ne, label %str_diff, label %str_lencmp

str_diff:
  %st.neg = icmp slt i32 %st.mc, 0
  %st.dr = select i1 %st.neg, i32 -1, i32 1
  ret i32 %st.dr

str_lencmp:
  %st.llt = icmp ult i64 %st.la, %st.lb
  %st.lgt = icmp ugt i64 %st.la, %st.lb
  %st.lg = select i1 %st.lgt, i32 1, i32 0
  %st.lr = select i1 %st.llt, i32 -1, i32 %st.lg
  ret i32 %st.lr
}

; ---------------------------------------------------------------------------
; srt_cmp: multi-key row comparator. keys = ptr[nkeys], descs = i8[nkeys]
; (1 => descending). Returns i32 -1/0/1; left wins ties at the caller.
; ---------------------------------------------------------------------------
define internal i32 @srt_cmp(ptr %keys, ptr %descs, i64 %nkeys, i64 %a, i64 %b) #0 {
entry:
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %jn, %cont ]
  %jc = icmp ult i64 %j, %nkeys
  br i1 %jc, label %body, label %eq

body:
  %kp = getelementptr inbounds ptr, ptr %keys, i64 %j
  %col = load ptr, ptr %kp, align 8
  %dp = getelementptr inbounds i8, ptr %descs, i64 %j
  %d8 = load i8, ptr %dp, align 1
  %descb = icmp ne i8 %d8, 0
  %c = call i32 @srt_cmp_col(ptr %col, i64 %a, i64 %b)
  %cneg = sub i32 0, %c
  %c2 = select i1 %descb, i32 %cneg, i32 %c
  %nz = icmp ne i32 %c2, 0
  br i1 %nz, label %ret, label %cont

ret:
  ret i32 %c2

cont:
  %jn = add i64 %j, 1
  br label %loop

eq:
  ret i32 0
}

; ---------------------------------------------------------------------------
; srt_iota: out[i] = i for i in 0..n
; ---------------------------------------------------------------------------
define internal void @srt_iota(ptr %out, i64 %n) #0 {
entry:
  br label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %body ]
  %c = icmp ult i64 %k, %n
  br i1 %c, label %body, label %done

body:
  %p = getelementptr inbounds i64, ptr %out, i64 %k
  store i64 %k, ptr %p, align 8
  %kn = add i64 %k, 1
  br label %loop

done:
  ret void
}

; ---------------------------------------------------------------------------
; srt_merge_sort: stable bottom-up merge sort of idx[0..n) via srt_cmp.
; ---------------------------------------------------------------------------
define internal void @srt_merge_sort(ptr %idx, i64 %n, ptr %keys, ptr %descs, i64 %nkeys) #0 {
entry:
  %lt2 = icmp ult i64 %n, 2
  br i1 %lt2, label %ret, label %init

init:
  %auxbytes = mul i64 %n, 8
  %aux = call ptr @srt_xmalloc(i64 %auxbytes)
  br label %w.head

w.head:
  %src = phi ptr [ %idx, %init ], [ %dst, %w.next ]
  %dst = phi ptr [ %aux, %init ], [ %src, %w.next ]
  %width = phi i64 [ 1, %init ], [ %width2, %w.next ]
  %wcmp = icmp ult i64 %width, %n
  br i1 %wcmp, label %pass, label %w.done

pass:
  %twowidth = shl i64 %width, 1
  br label %run.head

run.head:
  %lo = phi i64 [ 0, %pass ], [ %hi, %run.cont ]
  %locmp = icmp ult i64 %lo, %n
  br i1 %locmp, label %run.body, label %pass.done

run.body:
  %midtmp = add i64 %lo, %width
  %mid = call i64 @llvm.umin.i64(i64 %midtmp, i64 %n)
  %hitmp = add i64 %lo, %twowidth
  %hi = call i64 @llvm.umin.i64(i64 %hitmp, i64 %n)
  br label %m.head

m.head:
  %i = phi i64 [ %lo, %run.body ], [ %i.n, %m.adv ]
  %j = phi i64 [ %mid, %run.body ], [ %j.n, %m.adv ]
  %kk = phi i64 [ %lo, %run.body ], [ %kk.n, %m.adv ]
  %iok = icmp ult i64 %i, %mid
  %jok = icmp ult i64 %j, %hi
  %both = and i1 %iok, %jok
  br i1 %both, label %m.cmp, label %m.tail

m.cmp:
  %si.p = getelementptr inbounds i64, ptr %src, i64 %i
  %si = load i64, ptr %si.p, align 8
  %sj.p = getelementptr inbounds i64, ptr %src, i64 %j
  %sj = load i64, ptr %sj.p, align 8
  %c = call i32 @srt_cmp(ptr %keys, ptr %descs, i64 %nkeys, i64 %si, i64 %sj)
  %takeleft = icmp sle i32 %c, 0
  br i1 %takeleft, label %m.left, label %m.right

m.left:
  %dl = getelementptr inbounds i64, ptr %dst, i64 %kk
  store i64 %si, ptr %dl, align 8
  %iinc = add i64 %i, 1
  br label %m.adv

m.right:
  %dr = getelementptr inbounds i64, ptr %dst, i64 %kk
  store i64 %sj, ptr %dr, align 8
  %jinc = add i64 %j, 1
  br label %m.adv

m.adv:
  %i.n = phi i64 [ %iinc, %m.left ], [ %i, %m.right ]
  %j.n = phi i64 [ %j, %m.left ], [ %jinc, %m.right ]
  %kk.n = add i64 %kk, 1
  br label %m.head

m.tail:
  br label %ti.head

ti.head:
  %ti = phi i64 [ %i, %m.tail ], [ %ti.n, %ti.body ]
  %tk = phi i64 [ %kk, %m.tail ], [ %tk.n, %ti.body ]
  %tic = icmp ult i64 %ti, %mid
  br i1 %tic, label %ti.body, label %ti.done

ti.body:
  %ti.sp = getelementptr inbounds i64, ptr %src, i64 %ti
  %ti.v = load i64, ptr %ti.sp, align 8
  %ti.dp = getelementptr inbounds i64, ptr %dst, i64 %tk
  store i64 %ti.v, ptr %ti.dp, align 8
  %ti.n = add i64 %ti, 1
  %tk.n = add i64 %tk, 1
  br label %ti.head

ti.done:
  br label %tj.head

tj.head:
  %tj = phi i64 [ %j, %ti.done ], [ %tj.n, %tj.body ]
  %tk2 = phi i64 [ %tk, %ti.done ], [ %tk2.n, %tj.body ]
  %tjc = icmp ult i64 %tj, %hi
  br i1 %tjc, label %tj.body, label %tj.done

tj.body:
  %tj.sp = getelementptr inbounds i64, ptr %src, i64 %tj
  %tj.v = load i64, ptr %tj.sp, align 8
  %tj.dp = getelementptr inbounds i64, ptr %dst, i64 %tk2
  store i64 %tj.v, ptr %tj.dp, align 8
  %tj.n = add i64 %tj, 1
  %tk2.n = add i64 %tk2, 1
  br label %tj.head

tj.done:
  br label %run.cont

run.cont:
  br label %run.head

pass.done:
  br label %w.next

w.next:
  %width2 = shl i64 %width, 1
  br label %w.head

w.done:
  %isidx = icmp eq ptr %src, %idx
  br i1 %isidx, label %free, label %copyback

copyback:
  %cbbytes = mul i64 %n, 8
  call void @llvm.memcpy.p0.p0.i64(ptr %idx, ptr %src, i64 %cbbytes, i1 false)
  br label %free

free:
  call void @free(ptr %aux)
  br label %ret

ret:
  ret void
}

; ---------------------------------------------------------------------------
; srt_radix1: single numeric (I32/I64, no nulls) key -> stable permutation in
; out_idx via LSD radix (8 passes x 8 bits) over a sortable-u64 transform.
; ---------------------------------------------------------------------------
define internal void @srt_radix1(ptr %series, i8 %desc, ptr %out_idx, i64 %n) #0 {
entry:
  %cnt = alloca [256 x i64], align 8
  %n0 = icmp eq i64 %n, 0
  br i1 %n0, label %ret, label %setup

setup:
  %dtype = load i32, ptr %series, align 8
  %vpp = getelementptr inbounds i8, ptr %series, i64 24
  %values = load ptr, ptr %vpp, align 8
  %isI64 = icmp eq i32 %dtype, 1
  %descb = icmp ne i8 %desc, 0
  %bytes = mul i64 %n, 8
  %keyA = call ptr @srt_xmalloc(i64 %bytes)
  %keyB = call ptr @srt_xmalloc(i64 %bytes)
  %idxB = call ptr @srt_xmalloc(i64 %bytes)
  br label %fill.head

fill.head:
  %fk = phi i64 [ 0, %setup ], [ %fk.n, %fill.cont ]
  %fc = icmp ult i64 %fk, %n
  br i1 %fc, label %fill.body, label %pass.init

fill.body:
  br i1 %isI64, label %fill.i64, label %fill.i32

fill.i64:
  %fi64.p = getelementptr inbounds i64, ptr %values, i64 %fk
  %fi64.v = load i64, ptr %fi64.p, align 8
  %fi64.u = xor i64 %fi64.v, -9223372036854775808
  br label %fill.have

fill.i32:
  %fi32.p = getelementptr inbounds i32, ptr %values, i64 %fk
  %fi32.v = load i32, ptr %fi32.p, align 4
  %fi32.x = xor i32 %fi32.v, -2147483648
  %fi32.u = zext i32 %fi32.x to i64
  br label %fill.have

fill.have:
  %fu0 = phi i64 [ %fi64.u, %fill.i64 ], [ %fi32.u, %fill.i32 ]
  %fu.inv = xor i64 %fu0, -1
  %fu = select i1 %descb, i64 %fu.inv, i64 %fu0
  %fkp = getelementptr inbounds i64, ptr %keyA, i64 %fk
  store i64 %fu, ptr %fkp, align 8
  %fip = getelementptr inbounds i64, ptr %out_idx, i64 %fk
  store i64 %fk, ptr %fip, align 8
  br label %fill.cont

fill.cont:
  %fk.n = add i64 %fk, 1
  br label %fill.head

pass.init:
  br label %p.head

p.head:
  %src_k = phi ptr [ %keyA, %pass.init ], [ %dst_k, %p.next ]
  %dst_k = phi ptr [ %keyB, %pass.init ], [ %src_k, %p.next ]
  %src_i = phi ptr [ %out_idx, %pass.init ], [ %dst_i, %p.next ]
  %dst_i = phi ptr [ %idxB, %pass.init ], [ %src_i, %p.next ]
  %p = phi i64 [ 0, %pass.init ], [ %p.plus, %p.next ]
  %pc = icmp ult i64 %p, 8
  br i1 %pc, label %p.body, label %p.done

p.body:
  call void @llvm.memset.p0.i64(ptr %cnt, i8 0, i64 2048, i1 false)
  %shift = shl i64 %p, 3
  br label %c.head

c.head:
  %ck = phi i64 [ 0, %p.body ], [ %ck.n, %c.body ]
  %cc = icmp ult i64 %ck, %n
  br i1 %cc, label %c.body, label %prefix

c.body:
  %ckp = getelementptr inbounds i64, ptr %src_k, i64 %ck
  %ckv = load i64, ptr %ckp, align 8
  %cbyte = lshr i64 %ckv, %shift
  %cidx = and i64 %cbyte, 255
  %cslot = getelementptr inbounds [256 x i64], ptr %cnt, i64 0, i64 %cidx
  %cold = load i64, ptr %cslot, align 8
  %cnew = add i64 %cold, 1
  store i64 %cnew, ptr %cslot, align 8
  %ck.n = add i64 %ck, 1
  br label %c.head

prefix:
  br label %pf.head

pf.head:
  %pb = phi i64 [ 0, %prefix ], [ %pb.n, %pf.body ]
  %prun = phi i64 [ 0, %prefix ], [ %prun.n, %pf.body ]
  %pfc = icmp ult i64 %pb, 256
  br i1 %pfc, label %pf.body, label %scatter

pf.body:
  %pfslot = getelementptr inbounds [256 x i64], ptr %cnt, i64 0, i64 %pb
  %pfcount = load i64, ptr %pfslot, align 8
  store i64 %prun, ptr %pfslot, align 8
  %prun.n = add i64 %prun, %pfcount
  %pb.n = add i64 %pb, 1
  br label %pf.head

scatter:
  br label %sc.head

sc.head:
  %sk = phi i64 [ 0, %scatter ], [ %sk.n, %sc.body ]
  %scc = icmp ult i64 %sk, %n
  br i1 %scc, label %sc.body, label %p.next

sc.body:
  %skp = getelementptr inbounds i64, ptr %src_k, i64 %sk
  %skv = load i64, ptr %skp, align 8
  %sbyte = lshr i64 %skv, %shift
  %sidx = and i64 %sbyte, 255
  %sslot = getelementptr inbounds [256 x i64], ptr %cnt, i64 0, i64 %sidx
  %spos = load i64, ptr %sslot, align 8
  %spos1 = add i64 %spos, 1
  store i64 %spos1, ptr %sslot, align 8
  %sdkp = getelementptr inbounds i64, ptr %dst_k, i64 %spos
  store i64 %skv, ptr %sdkp, align 8
  %sivp = getelementptr inbounds i64, ptr %src_i, i64 %sk
  %siv = load i64, ptr %sivp, align 8
  %sdip = getelementptr inbounds i64, ptr %dst_i, i64 %spos
  store i64 %siv, ptr %sdip, align 8
  %sk.n = add i64 %sk, 1
  br label %sc.head

p.next:
  %p.plus = add i64 %p, 1
  br label %p.head

p.done:
  %need_copy = icmp ne ptr %src_i, %out_idx
  br i1 %need_copy, label %docopy, label %freeall

docopy:
  call void @llvm.memcpy.p0.p0.i64(ptr %out_idx, ptr %src_i, i64 %bytes, i1 false)
  br label %freeall

freeall:
  call void @free(ptr %keyA)
  call void @free(ptr %keyB)
  call void @free(ptr %idxB)
  br label %ret

ret:
  ret void
}

; ---------------------------------------------------------------------------
; srt_argsort_into: fill out_idx[0..n) with the ordering permutation.
; Chooses radix fast path (single numeric key, no nulls) else stable merge.
; ---------------------------------------------------------------------------
define internal void @srt_argsort_into(ptr %keys, ptr %descs, i64 %nkeys, i64 %n, ptr %out_idx) #0 {
entry:
  %single = icmp eq i64 %nkeys, 1
  br i1 %single, label %chk, label %slow

chk:
  %col = load ptr, ptr %keys, align 8
  %dtype = load i32, ptr %col, align 8
  %isI32 = icmp eq i32 %dtype, 0
  %isI64 = icmp eq i32 %dtype, 1
  %isint = or i1 %isI32, %isI64
  %ncp = getelementptr inbounds i8, ptr %col, i64 16
  %nc = load i64, ptr %ncp, align 8
  %nonull = icmp eq i64 %nc, 0
  %fast = and i1 %isint, %nonull
  br i1 %fast, label %dofast, label %slow

dofast:
  %desc8 = load i8, ptr %descs, align 1
  call void @srt_radix1(ptr %col, i8 %desc8, ptr %out_idx, i64 %n)
  ret void

slow:
  call void @srt_iota(ptr %out_idx, i64 %n)
  call void @srt_merge_sort(ptr %out_idx, i64 %n, ptr %keys, ptr %descs, i64 %nkeys)
  ret void
}

; ---------------------------------------------------------------------------
; srt_build_keys: resolve %nkeys by-name columns into caller-supplied %keys
; (ptr[]) and %descs (i8[]). Returns 0 OK, 5 NOT_FOUND. %desc may be null.
; ---------------------------------------------------------------------------
define internal i32 @srt_build_keys(ptr %df, ptr %by, i64 %nkeys, ptr %desc, ptr %keys, ptr %descs) #0 {
entry:
  %descnull = icmp eq ptr %desc, null
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %jn, %cont ]
  %jc = icmp ult i64 %j, %nkeys
  br i1 %jc, label %body, label %ok

body:
  %noff = mul i64 %j, 16
  %nslot = getelementptr inbounds i8, ptr %by, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %col = call ptr @universe_dataframe_column(ptr %df, ptr %nptr, i64 %nlen)
  %notfound = icmp eq ptr %col, null
  br i1 %notfound, label %err, label %store

store:
  %kslot = getelementptr inbounds ptr, ptr %keys, i64 %j
  store ptr %col, ptr %kslot, align 8
  br i1 %descnull, label %d0, label %dload

dload:
  %dp = getelementptr inbounds i8, ptr %desc, i64 %j
  %dv = load i8, ptr %dp, align 1
  %dnz = icmp ne i8 %dv, 0
  %d1 = zext i1 %dnz to i8
  br label %dstore

d0:
  br label %dstore

dstore:
  %dfinal = phi i8 [ 0, %d0 ], [ %d1, %dload ]
  %dsslot = getelementptr inbounds i8, ptr %descs, i64 %j
  store i8 %dfinal, ptr %dsslot, align 1
  br label %cont

cont:
  %jn = add i64 %j, 1
  br label %loop

err:
  ret i32 5

ok:
  ret i32 0
}

; ---------------------------------------------------------------------------
; srt_gather_new_df: NEW df = every column of %df gathered by idx[0..n2).
; ---------------------------------------------------------------------------
define internal ptr @srt_gather_new_df(ptr %df, ptr %idx, i64 %n2) #0 {
entry:
  %nc = load i64, ptr %df, align 8
  %namesp = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %namesp, align 8
  %out = call ptr @universe_dataframe_new()
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %loop

loop:
  %k = phi i64 [ 0, %entry ], [ %kn, %cont ]
  %kc = icmp ult i64 %k, %nc
  br i1 %kc, label %body, label %done

body:
  %col = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %k)
  %newcol = call ptr @srt_series_gather(ptr %col, ptr %idx, i64 %n2)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @universe_dataframe_with_column(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %cont

cont:
  %kn = add i64 %k, 1
  br label %loop

done:
  ret ptr %out

fail:
  ret ptr null
}

; ===========================================================================
; PUBLIC API
; ===========================================================================

; permutation of a single series into caller's out_idx (i64[len]).
define i32 @universe_dataframe_arg_sort(ptr %series, i32 %descending, ptr %out_idx) local_unnamed_addr #0 {
entry:
  %k1 = alloca [1 x ptr], align 8
  %d1 = alloca [1 x i8], align 1
  %sn = icmp eq ptr %series, null
  %on = icmp eq ptr %out_idx, null
  %bad = or i1 %sn, %on
  br i1 %bad, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %lp = getelementptr inbounds i8, ptr %series, i64 8
  %n = load i64, ptr %lp, align 8
  store ptr %series, ptr %k1, align 8
  %dnz = icmp ne i32 %descending, 0
  %d8 = zext i1 %dnz to i8
  store i8 %d8, ptr %d1, align 1
  call void @srt_argsort_into(ptr %k1, ptr %d1, i64 1, i64 %n, ptr %out_idx)
  ret i32 0
}

; NEW sorted df ordered by the named key columns.
define ptr @universe_dataframe_sort(ptr %df, ptr %by, i64 %nkeys, ptr %desc) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %bn = icmp eq ptr %by, null
  %bad = or i1 %dn, %bn
  br i1 %bad, label %fail, label %setup, !prof !0

setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %h = load i64, ptr %hp, align 8
  %kbytes = mul i64 %nkeys, 8
  %keys = call ptr @srt_xmalloc(i64 %kbytes)
  %descs = call ptr @srt_xmalloc(i64 %nkeys)
  %rc = call i32 @srt_build_keys(ptr %df, ptr %by, i64 %nkeys, ptr %desc, ptr %keys, ptr %descs)
  %rcok = icmp eq i32 %rc, 0
  br i1 %rcok, label %perm, label %freekeys, !prof !0

perm:
  %ibytes = mul i64 %h, 8
  %idx = call ptr @srt_xmalloc(i64 %ibytes)
  call void @srt_argsort_into(ptr %keys, ptr %descs, i64 %nkeys, i64 %h, ptr %idx)
  %out = call ptr @srt_gather_new_df(ptr %df, ptr %idx, i64 %h)
  call void @free(ptr %idx)
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  ret ptr %out

freekeys:
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  br label %fail

fail:
  ret ptr null
}

; in-place reorder of an existing df by the named key columns.
define i32 @universe_dataframe_sort_in_place(ptr %df, ptr %by, i64 %nkeys, ptr %desc) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %bn = icmp eq ptr %by, null
  %bad = or i1 %dn, %bn
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %h = load i64, ptr %hp, align 8
  %ncp2 = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp2, align 8
  %kbytes = mul i64 %nkeys, 8
  %keys = call ptr @srt_xmalloc(i64 %kbytes)
  %descs = call ptr @srt_xmalloc(i64 %nkeys)
  %rc = call i32 @srt_build_keys(ptr %df, ptr %by, i64 %nkeys, ptr %desc, ptr %keys, ptr %descs)
  %rcok = icmp eq i32 %rc, 0
  br i1 %rcok, label %perm, label %freekeys, !prof !0

perm:
  %ibytes = mul i64 %h, 8
  %idx = call ptr @srt_xmalloc(i64 %ibytes)
  call void @srt_argsort_into(ptr %keys, ptr %descs, i64 %nkeys, i64 %h, ptr %idx)
  %colsp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %colsp, align 8
  br label %loop

loop:
  %k = phi i64 [ 0, %perm ], [ %kn, %body ]
  %kc = icmp ult i64 %k, %nc
  br i1 %kc, label %body, label %done

body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %oldcol = load ptr, ptr %cslot, align 8
  %newcol = call ptr @srt_series_gather(ptr %oldcol, ptr %idx, i64 %h)
  call void @universe_dataframe_series_free(ptr %oldcol)
  store ptr %newcol, ptr %cslot, align 8
  %kn = add i64 %k, 1
  br label %loop

done:
  call void @free(ptr %idx)
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  ret i32 0

freekeys:
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  ret i32 5
}

; k LARGEST rows (largest first); scalar descending reverses to k smallest.
define ptr @universe_dataframe_top_k(ptr %df, i64 %k, ptr %by, i64 %nkeys, i32 %descending) local_unnamed_addr #0 {
entry:
  %rev = icmp ne i32 %descending, 0
  %flag = select i1 %rev, i8 0, i8 1
  %r = call ptr @srt_topbot(ptr %df, i64 %k, ptr %by, i64 %nkeys, i8 %flag)
  ret ptr %r
}

; k SMALLEST rows (smallest first).
define ptr @universe_dataframe_bottom_k(ptr %df, i64 %k, ptr %by, i64 %nkeys) local_unnamed_addr #0 {
entry:
  %r = call ptr @srt_topbot(ptr %df, i64 %k, ptr %by, i64 %nkeys, i8 0)
  ret ptr %r
}

; shared top/bottom: %descall applied to every key (1 => descending).
define internal ptr @srt_topbot(ptr %df, i64 %k, ptr %by, i64 %nkeys, i8 %descall) #0 {
entry:
  %dn = icmp eq ptr %df, null
  %bn = icmp eq ptr %by, null
  %bad = or i1 %dn, %bn
  br i1 %bad, label %fail, label %setup, !prof !0

setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %h = load i64, ptr %hp, align 8
  %kbytes = mul i64 %nkeys, 8
  %keys = call ptr @srt_xmalloc(i64 %kbytes)
  %descs = call ptr @srt_xmalloc(i64 %nkeys)
  br label %dfill.head

dfill.head:
  %dj = phi i64 [ 0, %setup ], [ %dj.n, %dfill.body ]
  %djc = icmp ult i64 %dj, %nkeys
  br i1 %djc, label %dfill.body, label %resolve

dfill.body:
  %djp = getelementptr inbounds i8, ptr %descs, i64 %dj
  store i8 %descall, ptr %djp, align 1
  %dj.n = add i64 %dj, 1
  br label %dfill.head

resolve:
  %rc = call i32 @srt_build_keys(ptr %df, ptr %by, i64 %nkeys, ptr %descs, ptr %keys, ptr %descs)
  %rcok = icmp eq i32 %rc, 0
  br i1 %rcok, label %perm, label %freekeys, !prof !0

perm:
  %ibytes = mul i64 %h, 8
  %idx = call ptr @srt_xmalloc(i64 %ibytes)
  call void @srt_argsort_into(ptr %keys, ptr %descs, i64 %nkeys, i64 %h, ptr %idx)
  %n2 = call i64 @llvm.umin.i64(i64 %k, i64 %h)
  %out = call ptr @srt_gather_new_df(ptr %df, ptr %idx, i64 %n2)
  call void @free(ptr %idx)
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  ret ptr %out

freekeys:
  call void @free(ptr %keys)
  call void @free(ptr %descs)
  br label %fail

fail:
  ret ptr null
}

attributes #0 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
