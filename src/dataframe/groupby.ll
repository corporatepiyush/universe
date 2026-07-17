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

; DataFrame grouping + aggregation (Polars method NAMES; our own low-level
; design). Depends only on frame.ll's EXPORTED symbols (internal frame helpers
; are module-private, so we re-derive the small ones we need here).
;
; ============================ DESIGN ========================================
; ALGORITHM CLASS: hash grouping. Each row's key columns are hashed with an
; INLINE FNV-1a byte mix into an open-addressing map (power-of-two capacity,
; mask wrap, linear probe, <=70% load) whose slots hold a group id (i64, -1 =
; empty). A colliding slot is resolved by comparing the row's key values to the
; group's REPRESENTATIVE (first) row. Group ids are handed out in ROW ORDER, so
; group order == FIRST-APPEARANCE order: group_by and group_by_stable share the
; same core (stable is a strict-superset guarantee we always meet).
;
; Per-group member row lists are built as CSR (group_off[G+1] + members[height])
; with the two-pass count-then-scatter shape (same as ivf/parquet_meta). Each
; aggregate is a reduce over a group's contiguous member slice.
;
; The hot leaf (row hash + probe + equality) is MONOMORPHIC and fully inlined
; (alwaysinline helpers, no fn-ptr / hashmap cross-call) per the CLAUDE.md
; hot-leaf rule. Key column value/validity/dtype/strdata pointers are resolved
; ONCE into flat arrays before the row loop, so the loop body has no calls into
; frame.ll accessors.
;
; GroupBy handle = ONE malloc'd 64-byte header:
;   +0   ptr df            ; BORROWED source frame
;   +8   i64 n_groups      ; G
;   +16  i64 height        ; source row count
;   +24  i64 nk            ; number of key columns
;   +32  ptr key_idx       ; i64[nk] source column indices
;   +40  ptr group_off     ; i64[G+1] CSR offsets into members
;   +48  ptr members       ; i64[height] row indices grouped (row-order within grp)
;   +56  ptr group_rep     ; i64[G] first (representative) row per group
;
; API / CONTRACT:
;   group_by(df,key_names,nk)         -> opaque GroupBy* (null on error)
;   group_by_stable(df,key_names,nk)  -> same (first-appearance order)
;     key_names: array of {ptr name,i64 len} pairs (frame's name representation).
;     nk must be in [1,width]; unknown name / bad args => null.
;   group_by_agg(gb,agg_cols,nc,agg_ops,out_df) -> i32 status
;     agg_cols: i64[nc] SOURCE column indices (NOT names). agg_ops: i32[nc]
;     0 sum,1 mean,2 min,3 max,4 count,5 n_unique. Output frame = key columns
;     (one row per group) followed by one column per (agg_col,agg_op), named
;     "<colname>_<opsuffix>". Aggregate dtypes: sum/mean/min/max -> F64;
;     count/n_unique -> I64. Aggregates skip nulls; empty (all-null) group =>
;     sum 0, count/n_unique 0, mean/min/max NULL. Agg columns must be numeric
;     (I32/I64/F32/F64/BOOL) else INVALID_ARG(8). *out_df receives the new frame.
;   partition_by(df,key_names,nk,out_frames,cap,out_n) -> i32 status
;     Splits df into per-group sub-frames (all columns, member rows gathered).
;     *out_n receives G. If G>cap => FULL(6) (out_n still set for a retry).
;     out_frames[g] receives a NEW frame (caller owns; free with dataframe_free).
;   group_by_free(gb) -> void  (frees handle+arrays; NOT the borrowed df)
;
; Errors i32: 0 OK,1 NULL_PTR,2 OOM,3 SIZE_OVERFLOW,5 NOT_FOUND,6 FULL,
;   7 INVALID_INDEX,8 INVALID_ARG.
; ===========================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i32 @memcmp(ptr captures(none), ptr captures(none), i64)

; frame.ll exported API
declare ptr @universe_dataframe_new() local_unnamed_addr
declare i64 @universe_dataframe_height(ptr) local_unnamed_addr
declare i64 @universe_dataframe_width(ptr) local_unnamed_addr
declare i32 @universe_dataframe_get_column_index(ptr, ptr, i64, ptr) local_unnamed_addr
declare ptr @universe_dataframe_select_at_idx(ptr, i64) local_unnamed_addr
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr) local_unnamed_addr
declare void @universe_dataframe_free(ptr) local_unnamed_addr
declare ptr @universe_dataframe_series_new(i32, i64) local_unnamed_addr
declare ptr @universe_dataframe_series_values(ptr) local_unnamed_addr
declare i32 @universe_dataframe_series_dtype(ptr) local_unnamed_addr
declare i64 @universe_dataframe_series_len(ptr) local_unnamed_addr
declare i32 @universe_dataframe_series_set_null(ptr, i64) local_unnamed_addr
declare ptr @universe_dataframe_series_str_new(ptr, ptr, i64, i64) local_unnamed_addr

@gb.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"

; op-suffix table (for output aggregate column names)
@gb.sfx.sum   = internal constant [3 x i8] c"sum"
@gb.sfx.mean  = internal constant [4 x i8] c"mean"
@gb.sfx.min   = internal constant [3 x i8] c"min"
@gb.sfx.max   = internal constant [3 x i8] c"max"
@gb.sfx.cnt   = internal constant [5 x i8] c"count"
@gb.sfx.nuq   = internal constant [8 x i8] c"n_unique"

; ---------------------------------------------------------------------------
; small internal helpers (re-derived; frame's are module-private)
; ---------------------------------------------------------------------------

define internal ptr @gb_xmalloc(i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

define internal i64 @gb_width(i32 %dtype) #2 {
entry:
  %i = zext i32 %dtype to i64
  %p = getelementptr inbounds [6 x i8], ptr @gb.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

; validity ptr (may be null => all valid); test bit for element i (1 = valid)
define internal i1 @gb_bit_valid(ptr %bm, i64 %i) #1 {
entry:
  %vn = icmp eq ptr %bm, null
  br i1 %vn, label %valid, label %check

valid:
  ret i1 true

check:
  %bi = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %bm, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %i, 7
  %sh8 = trunc i64 %sh to i8
  %bit = lshr i8 %b, %sh8
  %lo = and i8 %bit, 1
  %r = icmp ne i8 %lo, 0
  ret i1 %r
}

; validity via a series header (validity ptr at +32)
define internal i1 @gb_valid_at(ptr %s, i64 %i) #1 {
entry:
  %vp = getelementptr inbounds i8, ptr %s, i64 32
  %v = load ptr, ptr %vp, align 8
  %r = call i1 @gb_bit_valid(ptr %v, i64 %i)
  ret i1 %r
}

; FNV-1a byte hash: h' = fold(h, bytes[0..len))
define internal i64 @gb_hash_bytes(i64 %h0, ptr %p, i64 %len) #1 {
entry:
  br label %loop

loop:
  %h = phi i64 [ %h0, %entry ], [ %hn, %body ]
  %k = phi i64 [ 0, %entry ], [ %kn, %body ]
  %c = icmp ult i64 %k, %len
  br i1 %c, label %body, label %done

body:
  %bp = getelementptr inbounds i8, ptr %p, i64 %k
  %b = load i8, ptr %bp, align 1
  %bz = zext i8 %b to i64
  %hx = xor i64 %h, %bz
  %hn = mul i64 %hx, 1099511628211
  %kn = add i64 %k, 1
  br label %loop

done:
  ret i64 %h
}

; fold a single sentinel byte into the running hash
define internal i64 @gb_mix1(i64 %h, i64 %byte) #1 {
entry:
  %hx = xor i64 %h, %byte
  %hn = mul i64 %hx, 1099511628211
  ret i64 %hn
}

; hash a row's key tuple. kvals/kvalid/kstr are ptr[nk]; kdtype is i32[nk].
define internal i64 @gb_row_hash(ptr %kvals, ptr %kvalid, ptr %kstr, ptr %kdtype, i64 %nk, i64 %row) #1 {
entry:
  br label %loop

loop:
  %h = phi i64 [ -3750763034362895579, %entry ], [ %hsep, %cont ]
  %c = phi i64 [ 0, %entry ], [ %cn, %cont ]
  %cmp = icmp ult i64 %c, %nk
  br i1 %cmp, label %body, label %done

body:
  %dtp = getelementptr inbounds i32, ptr %kdtype, i64 %c
  %dt = load i32, ptr %dtp, align 4
  %valp = getelementptr inbounds ptr, ptr %kvals, i64 %c
  %vals = load ptr, ptr %valp, align 8
  %vlp = getelementptr inbounds ptr, ptr %kvalid, i64 %c
  %vld = load ptr, ptr %vlp, align 8
  %isvalid = call i1 @gb_bit_valid(ptr %vld, i64 %row)
  br i1 %isvalid, label %present, label %null

null:
  %hnull = call i64 @gb_mix1(i64 %h, i64 158)
  br label %cont

present:
  %isstr = icmp eq i32 %dt, 5
  br i1 %isstr, label %str, label %fix

fix:
  %w = call i64 @gb_width(i32 %dt)
  %foff = mul i64 %row, %w
  %fptr = getelementptr inbounds i8, ptr %vals, i64 %foff
  %hfix = call i64 @gb_hash_bytes(i64 %h, ptr %fptr, i64 %w)
  br label %cont

str:
  %o0p = getelementptr inbounds i32, ptr %vals, i64 %row
  %o0 = load i32, ptr %o0p, align 4
  %row1 = add i64 %row, 1
  %o1p = getelementptr inbounds i32, ptr %vals, i64 %row1
  %o1 = load i32, ptr %o1p, align 4
  %o0z = zext i32 %o0 to i64
  %o1z = zext i32 %o1 to i64
  %slen = sub i64 %o1z, %o0z
  %strp0 = getelementptr inbounds ptr, ptr %kstr, i64 %c
  %sd = load ptr, ptr %strp0, align 8
  %sptr = getelementptr inbounds i8, ptr %sd, i64 %o0z
  %hstr = call i64 @gb_hash_bytes(i64 %h, ptr %sptr, i64 %slen)
  br label %cont

cont:
  %hcol = phi i64 [ %hnull, %null ], [ %hfix, %fix ], [ %hstr, %str ]
  %hsep = call i64 @gb_mix1(i64 %hcol, i64 1)
  %cn = add i64 %c, 1
  br label %loop

done:
  ret i64 %h
}

; equality of two rows' key tuples
define internal i1 @gb_row_eq(ptr %kvals, ptr %kvalid, ptr %kstr, ptr %kdtype, i64 %nk, i64 %ra, i64 %rb) #1 {
entry:
  br label %loop

loop:
  %c = phi i64 [ 0, %entry ], [ %cn, %cont ]
  %cmp = icmp ult i64 %c, %nk
  br i1 %cmp, label %body, label %eqtrue

body:
  %dtp = getelementptr inbounds i32, ptr %kdtype, i64 %c
  %dt = load i32, ptr %dtp, align 4
  %valp = getelementptr inbounds ptr, ptr %kvals, i64 %c
  %vals = load ptr, ptr %valp, align 8
  %vlp = getelementptr inbounds ptr, ptr %kvalid, i64 %c
  %vld = load ptr, ptr %vlp, align 8
  %va = call i1 @gb_bit_valid(ptr %vld, i64 %ra)
  %vb = call i1 @gb_bit_valid(ptr %vld, i64 %rb)
  %vsame = icmp eq i1 %va, %vb
  br i1 %vsame, label %vok, label %neq

vok:
  ; both same validity; if both null this column is equal
  br i1 %va, label %cmpval, label %cont

cmpval:
  %isstr = icmp eq i32 %dt, 5
  br i1 %isstr, label %str, label %fix

; typed fixed-width equality (bitwise; no data-dependent memcmp call in the
; hot probe loop). default (fix64) covers I64(1)/F64(3).
fix:
  switch i32 %dt, label %fix64 [ i32 0, label %fix32
                                 i32 2, label %fix32
                                 i32 4, label %fix8 ]

fix32:
  %a32p = getelementptr inbounds i32, ptr %vals, i64 %ra
  %a32 = load i32, ptr %a32p, align 4
  %b32p = getelementptr inbounds i32, ptr %vals, i64 %rb
  %b32 = load i32, ptr %b32p, align 4
  %eq32 = icmp eq i32 %a32, %b32
  br i1 %eq32, label %cont, label %neq

fix8:
  %a8p = getelementptr inbounds i8, ptr %vals, i64 %ra
  %a8 = load i8, ptr %a8p, align 1
  %b8p = getelementptr inbounds i8, ptr %vals, i64 %rb
  %b8 = load i8, ptr %b8p, align 1
  %eq8 = icmp eq i8 %a8, %b8
  br i1 %eq8, label %cont, label %neq

fix64:
  %a64p = getelementptr inbounds i64, ptr %vals, i64 %ra
  %a64 = load i64, ptr %a64p, align 8
  %b64p = getelementptr inbounds i64, ptr %vals, i64 %rb
  %b64 = load i64, ptr %b64p, align 8
  %eq64 = icmp eq i64 %a64, %b64
  br i1 %eq64, label %cont, label %neq

str:
  %a0p = getelementptr inbounds i32, ptr %vals, i64 %ra
  %a0 = load i32, ptr %a0p, align 4
  %ra1 = add i64 %ra, 1
  %a1p = getelementptr inbounds i32, ptr %vals, i64 %ra1
  %a1 = load i32, ptr %a1p, align 4
  %b0p = getelementptr inbounds i32, ptr %vals, i64 %rb
  %b0 = load i32, ptr %b0p, align 4
  %rb1 = add i64 %rb, 1
  %b1p = getelementptr inbounds i32, ptr %vals, i64 %rb1
  %b1 = load i32, ptr %b1p, align 4
  %alen = sub i32 %a1, %a0
  %blen = sub i32 %b1, %b0
  %leneq = icmp eq i32 %alen, %blen
  br i1 %leneq, label %strbytes, label %neq

strbytes:
  %a0z = zext i32 %a0 to i64
  %b0z = zext i32 %b0 to i64
  %alenz = zext i32 %alen to i64
  %strp0 = getelementptr inbounds ptr, ptr %kstr, i64 %c
  %sd = load ptr, ptr %strp0, align 8
  %sap = getelementptr inbounds i8, ptr %sd, i64 %a0z
  %sbp = getelementptr inbounds i8, ptr %sd, i64 %b0z
  %smc = call i32 @memcmp(ptr %sap, ptr %sbp, i64 %alenz)
  %seq = icmp eq i32 %smc, 0
  br i1 %seq, label %cont, label %neq

cont:
  %cn = add i64 %c, 1
  br label %loop

eqtrue:
  ret i1 true

neq:
  ret i1 false
}

; read a numeric element as f64 (I32/I64/F32/F64/BOOL). vals = series values ptr.
define internal double @gb_elem_f64(ptr %vals, i32 %dt, i64 %row) #2 {
entry:
  switch i32 %dt, label %d64 [ i32 0, label %d0
                               i32 1, label %d1
                               i32 2, label %d2
                               i32 4, label %d4 ]

d0:
  %p0 = getelementptr inbounds i32, ptr %vals, i64 %row
  %v0 = load i32, ptr %p0, align 4
  %f0 = sitofp i32 %v0 to double
  ret double %f0

d1:
  %p1 = getelementptr inbounds i64, ptr %vals, i64 %row
  %v1 = load i64, ptr %p1, align 8
  %f1 = sitofp i64 %v1 to double
  ret double %f1

d2:
  %p2 = getelementptr inbounds float, ptr %vals, i64 %row
  %v2 = load float, ptr %p2, align 4
  %f2 = fpext float %v2 to double
  ret double %f2

d4:
  %p4 = getelementptr inbounds i8, ptr %vals, i64 %row
  %v4 = load i8, ptr %p4, align 1
  %f4 = uitofp i8 %v4 to double
  ret double %f4

d64:
  %p8 = getelementptr inbounds double, ptr %vals, i64 %row
  %v8 = load double, ptr %p8, align 8
  ret double %v8
}

; ---------------------------------------------------------------------------
; gb_gather_series: NEW series = src rows selected by idx[0..n) (all idx valid,
; but the SELECTED element may be null -> propagated). Supports fixed + STR.
; ---------------------------------------------------------------------------
define internal ptr @gb_gather_series(ptr %src, ptr %idx, i64 %n) #0 {
entry:
  %dt = call i32 @universe_dataframe_series_dtype(ptr %src)
  %isstr = icmp eq i32 %dt, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %new = call ptr @universe_dataframe_series_new(i32 %dt, i64 %n)
  %nn = icmp eq ptr %new, null
  br i1 %nn, label %fail, label %fsetup

fsetup:
  %w = call i64 @gb_width(i32 %dt)
  %dst = call ptr @universe_dataframe_series_values(ptr %new)
  %sv = call ptr @universe_dataframe_series_values(ptr %src)
  br label %f.loop

f.loop:
  %fk = phi i64 [ 0, %fsetup ], [ %fkn, %f.cont ]
  %fc = icmp ult i64 %fk, %n
  br i1 %fc, label %f.body, label %done

f.body:
  %fjp = getelementptr inbounds i64, ptr %idx, i64 %fk
  %fj = load i64, ptr %fjp, align 8
  %fdo = mul i64 %fk, %w
  %fso = mul i64 %fj, %w
  %fdp = getelementptr inbounds i8, ptr %dst, i64 %fdo
  %fsp = getelementptr inbounds i8, ptr %sv, i64 %fso
  call void @llvm.memcpy.p0.p0.i64(ptr %fdp, ptr %fsp, i64 %w, i1 false)
  %fvalid = call i1 @gb_valid_at(ptr %src, i64 %fj)
  br i1 %fvalid, label %f.cont, label %f.null

f.null:
  %frc = call i32 @universe_dataframe_series_set_null(ptr %new, i64 %fk)
  br label %f.cont

f.cont:
  %fkn = add i64 %fk, 1
  br label %f.loop

str:
  %soffs = call ptr @universe_dataframe_series_values(ptr %src)
  %ssdp = getelementptr inbounds i8, ptr %src, i64 40
  %ssd = load ptr, ptr %ssdp, align 8
  br label %s1.loop

; pass 1: total data bytes
s1.loop:
  %s1k = phi i64 [ 0, %str ], [ %s1kn, %s1.cont ]
  %s1tot = phi i64 [ 0, %str ], [ %s1tn, %s1.cont ]
  %s1c = icmp ult i64 %s1k, %n
  br i1 %s1c, label %s1.body, label %s1.done

s1.body:
  %s1jp = getelementptr inbounds i64, ptr %idx, i64 %s1k
  %s1j = load i64, ptr %s1jp, align 8
  %s1a = getelementptr inbounds i32, ptr %soffs, i64 %s1j
  %s1o0 = load i32, ptr %s1a, align 4
  %s1j1 = add i64 %s1j, 1
  %s1b = getelementptr inbounds i32, ptr %soffs, i64 %s1j1
  %s1o1 = load i32, ptr %s1b, align 4
  %s1len = sub i32 %s1o1, %s1o0
  %s1lenz = zext i32 %s1len to i64
  br label %s1.cont

s1.cont:
  %s1tn = add i64 %s1tot, %s1lenz
  %s1kn = add i64 %s1k, 1
  br label %s1.loop

s1.done:
  %n1 = add i64 %n, 1
  %obytes = mul i64 %n1, 4
  %offout = call ptr @gb_xmalloc(i64 %obytes)
  %datout = call ptr @gb_xmalloc(i64 %s1tot)
  br label %s2.loop

; pass 2: fill offsets + data
s2.loop:
  %s2k = phi i64 [ 0, %s1.done ], [ %s2kn, %s2.cont ]
  %s2cur = phi i64 [ 0, %s1.done ], [ %s2curn, %s2.cont ]
  %s2c = icmp ult i64 %s2k, %n
  br i1 %s2c, label %s2.body, label %s2.fin

s2.body:
  %s2op = getelementptr inbounds i32, ptr %offout, i64 %s2k
  %s2cur32 = trunc i64 %s2cur to i32
  store i32 %s2cur32, ptr %s2op, align 4
  %s2jp = getelementptr inbounds i64, ptr %idx, i64 %s2k
  %s2j = load i64, ptr %s2jp, align 8
  %s2a = getelementptr inbounds i32, ptr %soffs, i64 %s2j
  %s2o0 = load i32, ptr %s2a, align 4
  %s2j1 = add i64 %s2j, 1
  %s2b = getelementptr inbounds i32, ptr %soffs, i64 %s2j1
  %s2o1 = load i32, ptr %s2b, align 4
  %s2o0z = zext i32 %s2o0 to i64
  %s2len32 = sub i32 %s2o1, %s2o0
  %s2len = zext i32 %s2len32 to i64
  %s2src = getelementptr inbounds i8, ptr %ssd, i64 %s2o0z
  %s2dst = getelementptr inbounds i8, ptr %datout, i64 %s2cur
  call void @llvm.memcpy.p0.p0.i64(ptr %s2dst, ptr %s2src, i64 %s2len, i1 false)
  br label %s2.cont

s2.cont:
  %s2curn = add i64 %s2cur, %s2len
  %s2kn = add i64 %s2k, 1
  br label %s2.loop

s2.fin:
  %s2lastp = getelementptr inbounds i32, ptr %offout, i64 %n
  %s2last32 = trunc i64 %s2cur to i32
  store i32 %s2last32, ptr %s2lastp, align 4
  %snew = call ptr @universe_dataframe_series_str_new(ptr %offout, ptr %datout, i64 %n, i64 %s2cur)
  call void @free(ptr %offout)
  call void @free(ptr %datout)
  %snn = icmp eq ptr %snew, null
  br i1 %snn, label %fail, label %s3.loop

; propagate nulls
s3.loop:
  %s3k = phi i64 [ 0, %s2.fin ], [ %s3kn, %s3.cont ]
  %s3c = icmp ult i64 %s3k, %n
  br i1 %s3c, label %s3.body, label %strdone

s3.body:
  %s3jp = getelementptr inbounds i64, ptr %idx, i64 %s3k
  %s3j = load i64, ptr %s3jp, align 8
  %s3valid = call i1 @gb_valid_at(ptr %src, i64 %s3j)
  br i1 %s3valid, label %s3.cont, label %s3.null

s3.null:
  %s3rc = call i32 @universe_dataframe_series_set_null(ptr %snew, i64 %s3k)
  br label %s3.cont

s3.cont:
  %s3kn = add i64 %s3k, 1
  br label %s3.loop

strdone:
  ret ptr %snew

done:
  ret ptr %new

fail:
  ret ptr null
}

; capacity: smallest pow2 with height <= 0.7*cap, min 16
define internal i64 @gb_cap(i64 %height) #2 {
entry:
  br label %loop

loop:
  %cap = phi i64 [ 16, %entry ], [ %cap2, %grow ]
  %h10 = mul i64 %height, 10
  %c7 = mul i64 %cap, 7
  %need = icmp ugt i64 %h10, %c7
  br i1 %need, label %grow, label %done

grow:
  %cap2 = shl i64 %cap, 1
  br label %loop

done:
  ret i64 %cap
}

; ---------------------------------------------------------------------------
; gb_build: core grouping. Returns GroupBy handle (null on error).
; ---------------------------------------------------------------------------
define internal ptr @gb_build(ptr %df, ptr %key_names, i64 %nk) #0 {
entry:
  %dfnull = icmp eq ptr %df, null
  %knnull = icmp eq ptr %key_names, null
  %e1 = or i1 %dfnull, %knnull
  br i1 %e1, label %fail0, label %chknk

chknk:
  %width = call i64 @universe_dataframe_width(ptr %df)
  %nk0 = icmp eq i64 %nk, 0
  %nkbig = icmp ugt i64 %nk, %width
  %badnk = or i1 %nk0, %nkbig
  br i1 %badnk, label %fail0, label %alloc

alloc:
  %height = call i64 @universe_dataframe_height(ptr %df)
  ; per-key arrays
  %kib = mul i64 %nk, 8
  %key_idx = call ptr @gb_xmalloc(i64 %kib)
  %kvals = call ptr @gb_xmalloc(i64 %kib)
  %kvalid = call ptr @gb_xmalloc(i64 %kib)
  %kstr = call ptr @gb_xmalloc(i64 %kib)
  %kdb = mul i64 %nk, 4
  %kdtype = call ptr @gb_xmalloc(i64 %kdb)
  br label %kloop

; resolve each key column: index + value/validity/dtype/strdata pointers
kloop:
  %kc = phi i64 [ 0, %alloc ], [ %kcn, %kcont ]
  %kcmp = icmp ult i64 %kc, %nk
  br i1 %kcmp, label %kbody, label %kdone

kbody:
  %noff = mul i64 %kc, 16
  %nslot = getelementptr inbounds i8, ptr %key_names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlp = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlp, align 8
  %idxslot = getelementptr inbounds i64, ptr %key_idx, i64 %kc
  %grc = call i32 @universe_dataframe_get_column_index(ptr %df, ptr %nptr, i64 %nlen, ptr %idxslot)
  %notfound = icmp ne i32 %grc, 0
  br i1 %notfound, label %failres, label %kresolve

kresolve:
  %cidx = load i64, ptr %idxslot, align 8
  %ks = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %cidx)
  %kv = call ptr @universe_dataframe_series_values(ptr %ks)
  %kvslot = getelementptr inbounds ptr, ptr %kvals, i64 %kc
  store ptr %kv, ptr %kvslot, align 8
  %kvldp = getelementptr inbounds i8, ptr %ks, i64 32
  %kvld = load ptr, ptr %kvldp, align 8
  %kvldslot = getelementptr inbounds ptr, ptr %kvalid, i64 %kc
  store ptr %kvld, ptr %kvldslot, align 8
  %ksdp = getelementptr inbounds i8, ptr %ks, i64 40
  %ksd = load ptr, ptr %ksdp, align 8
  %kstrslot = getelementptr inbounds ptr, ptr %kstr, i64 %kc
  store ptr %ksd, ptr %kstrslot, align 8
  %kdt = call i32 @universe_dataframe_series_dtype(ptr %ks)
  %kdtslot = getelementptr inbounds i32, ptr %kdtype, i64 %kc
  store i32 %kdt, ptr %kdtslot, align 4
  br label %kcont

kcont:
  %kcn = add i64 %kc, 1
  br label %kloop

kdone:
  ; grouping scratch
  %hb = mul i64 %height, 8
  %group_of = call ptr @gb_xmalloc(i64 %hb)
  %group_rep = call ptr @gb_xmalloc(i64 %hb)
  %group_cnt = call ptr @gb_xmalloc(i64 %hb)
  %cap = call i64 @gb_cap(i64 %height)
  %capb = mul i64 %cap, 8
  %slots = call ptr @gb_xmalloc(i64 %capb)
  call void @llvm.memset.p0.i64(ptr %slots, i8 -1, i64 %capb, i1 false)
  %mask = sub i64 %cap, 1
  br label %rloop

rloop:
  %r = phi i64 [ 0, %kdone ], [ %rn, %rcont ]
  %G = phi i64 [ 0, %kdone ], [ %Gc, %rcont ]
  %rcmp = icmp ult i64 %r, %height
  br i1 %rcmp, label %rbody, label %rdone

rbody:
  %h = call i64 @gb_row_hash(ptr %kvals, ptr %kvalid, ptr %kstr, ptr %kdtype, i64 %nk, i64 %r)
  %slot0 = and i64 %h, %mask
  br label %probe

probe:
  %slot = phi i64 [ %slot0, %rbody ], [ %slotn, %pcont ]
  %sp = getelementptr inbounds i64, ptr %slots, i64 %slot
  %g = load i64, ptr %sp, align 8
  %empty = icmp eq i64 %g, -1
  br i1 %empty, label %newgrp, label %checkeq

checkeq:
  %repp = getelementptr inbounds i64, ptr %group_rep, i64 %g
  %rep = load i64, ptr %repp, align 8
  %eq = call i1 @gb_row_eq(ptr %kvals, ptr %kvalid, ptr %kstr, ptr %kdtype, i64 %nk, i64 %r, i64 %rep)
  br i1 %eq, label %found, label %pcont

pcont:
  %slot.inc = add i64 %slot, 1
  %slotn = and i64 %slot.inc, %mask
  br label %probe

newgrp:
  store i64 %G, ptr %sp, align 8
  %reppn = getelementptr inbounds i64, ptr %group_rep, i64 %G
  store i64 %r, ptr %reppn, align 8
  %cntpn = getelementptr inbounds i64, ptr %group_cnt, i64 %G
  store i64 1, ptr %cntpn, align 8
  %gofn = getelementptr inbounds i64, ptr %group_of, i64 %r
  store i64 %G, ptr %gofn, align 8
  %Gnew = add i64 %G, 1
  br label %rcont

found:
  %goff = getelementptr inbounds i64, ptr %group_of, i64 %r
  store i64 %g, ptr %goff, align 8
  %cntp = getelementptr inbounds i64, ptr %group_cnt, i64 %g
  %cnt = load i64, ptr %cntp, align 8
  %cnt2 = add i64 %cnt, 1
  store i64 %cnt2, ptr %cntp, align 8
  br label %rcont

rcont:
  %Gc = phi i64 [ %Gnew, %newgrp ], [ %G, %found ]
  %rn = add i64 %r, 1
  br label %rloop

rdone:
  ; CSR offsets (prefix sum of counts)
  %G1 = add i64 %G, 8
  %gob = mul i64 %G1, 8
  %group_off = call ptr @gb_xmalloc(i64 %gob)
  br label %oloop

oloop:
  %og = phi i64 [ 0, %rdone ], [ %ogn, %ocont ]
  %oacc = phi i64 [ 0, %rdone ], [ %oaccn, %ocont ]
  %ocmp = icmp ult i64 %og, %G
  br i1 %ocmp, label %obody, label %ofin

obody:
  %offp = getelementptr inbounds i64, ptr %group_off, i64 %og
  store i64 %oacc, ptr %offp, align 8
  %cp = getelementptr inbounds i64, ptr %group_cnt, i64 %og
  %cv = load i64, ptr %cp, align 8
  br label %ocont

ocont:
  %oaccn = add i64 %oacc, %cv
  %ogn = add i64 %og, 1
  br label %oloop

ofin:
  %lastp = getelementptr inbounds i64, ptr %group_off, i64 %G
  store i64 %oacc, ptr %lastp, align 8
  ; scatter members using a cursor copy of offsets
  %members = call ptr @gb_xmalloc(i64 %hb)
  %curb = mul i64 %G, 8
  %cursor = call ptr @gb_xmalloc(i64 %curb)
  br label %cinit

cinit:
  %ci = phi i64 [ 0, %ofin ], [ %cin, %cibody ]
  %cicmp = icmp ult i64 %ci, %G
  br i1 %cicmp, label %cibody, label %sloop

cibody:
  %cisrc = getelementptr inbounds i64, ptr %group_off, i64 %ci
  %civ = load i64, ptr %cisrc, align 8
  %cidst = getelementptr inbounds i64, ptr %cursor, i64 %ci
  store i64 %civ, ptr %cidst, align 8
  %cin = add i64 %ci, 1
  br label %cinit

sloop:
  %sr = phi i64 [ 0, %cinit ], [ %srn, %sbody ]
  %srcmp = icmp ult i64 %sr, %height
  br i1 %srcmp, label %sbody, label %buildgb

sbody:
  %sgp = getelementptr inbounds i64, ptr %group_of, i64 %sr
  %sg = load i64, ptr %sgp, align 8
  %scp = getelementptr inbounds i64, ptr %cursor, i64 %sg
  %spos = load i64, ptr %scp, align 8
  %smp = getelementptr inbounds i64, ptr %members, i64 %spos
  store i64 %sr, ptr %smp, align 8
  %spos1 = add i64 %spos, 1
  store i64 %spos1, ptr %scp, align 8
  %srn = add i64 %sr, 1
  br label %sloop

buildgb:
  call void @free(ptr %group_of)
  call void @free(ptr %group_cnt)
  call void @free(ptr %slots)
  call void @free(ptr %cursor)
  %gb = call ptr @malloc(i64 64)
  %gbn = icmp eq ptr %gb, null
  br i1 %gbn, label %failbuilt, label %fillgb

fillgb:
  store ptr %df, ptr %gb, align 8
  %ngp = getelementptr inbounds i8, ptr %gb, i64 8
  store i64 %G, ptr %ngp, align 8
  %htp = getelementptr inbounds i8, ptr %gb, i64 16
  store i64 %height, ptr %htp, align 8
  %nkp = getelementptr inbounds i8, ptr %gb, i64 24
  store i64 %nk, ptr %nkp, align 8
  %kip = getelementptr inbounds i8, ptr %gb, i64 32
  store ptr %key_idx, ptr %kip, align 8
  %gofp = getelementptr inbounds i8, ptr %gb, i64 40
  store ptr %group_off, ptr %gofp, align 8
  %memp = getelementptr inbounds i8, ptr %gb, i64 48
  store ptr %members, ptr %memp, align 8
  %grepp = getelementptr inbounds i8, ptr %gb, i64 56
  store ptr %group_rep, ptr %grepp, align 8
  ; free the transient per-key resolution arrays (not needed after build)
  call void @free(ptr %kvals)
  call void @free(ptr %kvalid)
  call void @free(ptr %kstr)
  call void @free(ptr %kdtype)
  ret ptr %gb

failbuilt:
  call void @free(ptr %group_off)
  call void @free(ptr %members)
  call void @free(ptr %group_rep)
  call void @free(ptr %key_idx)
  call void @free(ptr %kvals)
  call void @free(ptr %kvalid)
  call void @free(ptr %kstr)
  call void @free(ptr %kdtype)
  ret ptr null

failres:
  call void @free(ptr %key_idx)
  call void @free(ptr %kvals)
  call void @free(ptr %kvalid)
  call void @free(ptr %kstr)
  call void @free(ptr %kdtype)
  ret ptr null

fail0:
  ret ptr null
}

; ---------------------------------------------------------------------------
; public: group_by / group_by_stable (same first-appearance core)
; ---------------------------------------------------------------------------
define ptr @universe_dataframe_group_by(ptr %df, ptr %key_names, i64 %nk) local_unnamed_addr #0 {
entry:
  %gb = call ptr @gb_build(ptr %df, ptr %key_names, i64 %nk)
  ret ptr %gb
}

define ptr @universe_dataframe_group_by_stable(ptr %df, ptr %key_names, i64 %nk) local_unnamed_addr #0 {
entry:
  %gb = call ptr @gb_build(ptr %df, ptr %key_names, i64 %nk)
  ret ptr %gb
}

define void @universe_dataframe_group_by_free(ptr %gb) local_unnamed_addr #0 {
entry:
  %n = icmp eq ptr %gb, null
  br i1 %n, label %done, label %dofree

dofree:
  %kip = getelementptr inbounds i8, ptr %gb, i64 32
  %ki = load ptr, ptr %kip, align 8
  call void @free(ptr %ki)
  %gofp = getelementptr inbounds i8, ptr %gb, i64 40
  %gof = load ptr, ptr %gofp, align 8
  call void @free(ptr %gof)
  %memp = getelementptr inbounds i8, ptr %gb, i64 48
  %mem = load ptr, ptr %memp, align 8
  call void @free(ptr %mem)
  %grepp = getelementptr inbounds i8, ptr %gb, i64 56
  %grep = load ptr, ptr %grepp, align 8
  call void @free(ptr %grep)
  call void @free(ptr %gb)
  br label %done

done:
  ret void
}

; op-suffix ptr+len (returns via out params)
define internal void @gb_op_suffix(i32 %op, ptr %out_ptr, ptr %out_len) #0 {
entry:
  switch i32 %op, label %def [ i32 0, label %sum
                               i32 1, label %mean
                               i32 2, label %min
                               i32 3, label %max
                               i32 4, label %cnt
                               i32 5, label %nuq ]
sum:
  store ptr @gb.sfx.sum, ptr %out_ptr, align 8
  store i64 3, ptr %out_len, align 8
  ret void
mean:
  store ptr @gb.sfx.mean, ptr %out_ptr, align 8
  store i64 4, ptr %out_len, align 8
  ret void
min:
  store ptr @gb.sfx.min, ptr %out_ptr, align 8
  store i64 3, ptr %out_len, align 8
  ret void
max:
  store ptr @gb.sfx.max, ptr %out_ptr, align 8
  store i64 3, ptr %out_len, align 8
  ret void
cnt:
  store ptr @gb.sfx.cnt, ptr %out_ptr, align 8
  store i64 5, ptr %out_len, align 8
  ret void
nuq:
  store ptr @gb.sfx.nuq, ptr %out_ptr, align 8
  store i64 8, ptr %out_len, align 8
  ret void
def:
  store ptr @gb.sfx.sum, ptr %out_ptr, align 8
  store i64 3, ptr %out_len, align 8
  ret void
}

; ---------------------------------------------------------------------------
; public: group_by_agg
; ---------------------------------------------------------------------------
define i32 @universe_dataframe_group_by_agg(ptr %gb, ptr %agg_cols, i64 %nc, ptr %agg_ops, ptr %out_df) local_unnamed_addr #0 {
entry:
  %sfxpp = alloca ptr, align 8
  %sfxlp = alloca i64, align 8
  %gbn = icmp eq ptr %gb, null
  %odn = icmp eq ptr %out_df, null
  %e0 = or i1 %gbn, %odn
  br i1 %e0, label %err.null, label %chkargs

chkargs:
  %hasc = icmp ugt i64 %nc, 0
  %acn = icmp eq ptr %agg_cols, null
  %aon = icmp eq ptr %agg_ops, null
  %anull = or i1 %acn, %aon
  %badargs = and i1 %hasc, %anull
  br i1 %badargs, label %err.null, label %load

err.null:
  ret i32 1

load:
  %df = load ptr, ptr %gb, align 8
  %ngp = getelementptr inbounds i8, ptr %gb, i64 8
  %G = load i64, ptr %ngp, align 8
  %nkp = getelementptr inbounds i8, ptr %gb, i64 24
  %nk = load i64, ptr %nkp, align 8
  %kip = getelementptr inbounds i8, ptr %gb, i64 32
  %key_idx = load ptr, ptr %kip, align 8
  %gofp = getelementptr inbounds i8, ptr %gb, i64 40
  %group_off = load ptr, ptr %gofp, align 8
  %memp = getelementptr inbounds i8, ptr %gb, i64 48
  %members = load ptr, ptr %memp, align 8
  %grepp = getelementptr inbounds i8, ptr %gb, i64 56
  %group_rep = load ptr, ptr %grepp, align 8
  %dfnamesp = getelementptr inbounds i8, ptr %df, i64 24
  %dfnames = load ptr, ptr %dfnamesp, align 8
  %out = call ptr @universe_dataframe_new()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %err.oom, label %kcols

err.oom:
  ret i32 2

; --- key columns (one row per group) ---
kcols:
  %kc = phi i64 [ 0, %load ], [ %kcn, %kccont ]
  %kcmp = icmp ult i64 %kc, %nk
  br i1 %kcmp, label %kcbody, label %acols

kcbody:
  %kidxp = getelementptr inbounds i64, ptr %key_idx, i64 %kc
  %kidx = load i64, ptr %kidxp, align 8
  %ks = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %kidx)
  %knewser = call ptr @gb_gather_series(ptr %ks, ptr %group_rep, i64 %G)
  %knoff = mul i64 %kidx, 16
  %knslot = getelementptr inbounds i8, ptr %dfnames, i64 %knoff
  %knptr = load ptr, ptr %knslot, align 8
  %knlp = getelementptr inbounds i8, ptr %knslot, i64 8
  %knlen = load i64, ptr %knlp, align 8
  %kwrc = call i32 @universe_dataframe_with_column(ptr %out, ptr %knptr, i64 %knlen, ptr %knewser)
  br label %kccont

kccont:
  %kcn = add i64 %kc, 1
  br label %kcols

; --- aggregate columns ---
acols:
  %ac = phi i64 [ 0, %kcols ], [ %acn2, %accont ]
  %acmp = icmp ult i64 %ac, %nc
  br i1 %acmp, label %acbody, label %finish

acbody:
  %acolp = getelementptr inbounds i64, ptr %agg_cols, i64 %ac
  %acol = load i64, ptr %acolp, align 8
  %aopp = getelementptr inbounds i32, ptr %agg_ops, i64 %ac
  %aop = load i32, ptr %aopp, align 4
  %as = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %acol)
  %adt = call i32 @universe_dataframe_series_dtype(ptr %as)
  %isstrcol = icmp eq i32 %adt, 5
  br i1 %isstrcol, label %err.arg, label %mkout

err.arg:
  call void @universe_dataframe_free(ptr %out)
  ret i32 8

mkout:
  ; output dtype: count(4)/n_unique(5) -> I64(1); else F64(3)
  %iscount = icmp eq i32 %aop, 4
  %isnuq = icmp eq i32 %aop, 5
  %isint = or i1 %iscount, %isnuq
  %outdt = select i1 %isint, i32 1, i32 3
  %newser = call ptr @universe_dataframe_series_new(i32 %outdt, i64 %G)
  %nsn = icmp eq ptr %newser, null
  br i1 %nsn, label %err.oom2, label %getvals

err.oom2:
  call void @universe_dataframe_free(ptr %out)
  ret i32 2

getvals:
  %avals = call ptr @universe_dataframe_series_values(ptr %as)
  %odvals = call ptr @universe_dataframe_series_values(ptr %newser)
  br label %gloop

; reduce over each group's member slice
gloop:
  %gi = phi i64 [ 0, %getvals ], [ %gin, %gcont ]
  %gcmp = icmp ult i64 %gi, %G
  br i1 %gcmp, label %gbody, label %addcol

gbody:
  %offp = getelementptr inbounds i64, ptr %group_off, i64 %gi
  %off = load i64, ptr %offp, align 8
  %gi1 = add i64 %gi, 1
  %off1p = getelementptr inbounds i64, ptr %group_off, i64 %gi1
  %off1 = load i64, ptr %off1p, align 8
  br label %mloop

; per-group reduce: sum,cnt,min,max tracked together
mloop:
  %mi = phi i64 [ %off, %gbody ], [ %min2, %mcont ]
  %msum = phi double [ 0.0, %gbody ], [ %msumn, %mcont ]
  %mcnt = phi i64 [ 0, %gbody ], [ %mcntn, %mcont ]
  %mmin = phi double [ 0x7FF0000000000000, %gbody ], [ %mminn, %mcont ]
  %mmax = phi double [ 0xFFF0000000000000, %gbody ], [ %mmaxn, %mcont ]
  %mcmp = icmp ult i64 %mi, %off1
  br i1 %mcmp, label %mbody, label %greduce

mbody:
  %mrp = getelementptr inbounds i64, ptr %members, i64 %mi
  %mrow = load i64, ptr %mrp, align 8
  %mvalid = call i1 @gb_valid_at(ptr %as, i64 %mrow)
  br i1 %mvalid, label %macc, label %mcont

macc:
  %mval = call double @gb_elem_f64(ptr %avals, i32 %adt, i64 %mrow)
  %msum1 = fadd double %msum, %mval
  %mcnt1 = add i64 %mcnt, 1
  %isless = fcmp olt double %mval, %mmin
  %mmin1 = select i1 %isless, double %mval, double %mmin
  %ismore = fcmp ogt double %mval, %mmax
  %mmax1 = select i1 %ismore, double %mval, double %mmax
  br label %mcont

mcont:
  %msumn = phi double [ %msum1, %macc ], [ %msum, %mbody ]
  %mcntn = phi i64 [ %mcnt1, %macc ], [ %mcnt, %mbody ]
  %mminn = phi double [ %mmin1, %macc ], [ %mmin, %mbody ]
  %mmaxn = phi double [ %mmax1, %macc ], [ %mmax, %mbody ]
  %min2 = add i64 %mi, 1
  br label %mloop

greduce:
  %cntf = uitofp i64 %mcnt to double
  %meanf = fdiv double %msum, %cntf
  %hasany = icmp ne i64 %mcnt, 0
  switch i32 %aop, label %st.sum [ i32 0, label %st.sum
                                   i32 1, label %st.mean
                                   i32 2, label %st.min
                                   i32 3, label %st.max
                                   i32 4, label %st.count
                                   i32 5, label %st.nuq ]

st.sum:
  %sump = getelementptr inbounds double, ptr %odvals, i64 %gi
  store double %msum, ptr %sump, align 8
  br label %gcont

st.mean:
  br i1 %hasany, label %st.mean.v, label %st.null

st.mean.v:
  %meanp = getelementptr inbounds double, ptr %odvals, i64 %gi
  store double %meanf, ptr %meanp, align 8
  br label %gcont

st.min:
  br i1 %hasany, label %st.min.v, label %st.null

st.min.v:
  %minp = getelementptr inbounds double, ptr %odvals, i64 %gi
  store double %mmin, ptr %minp, align 8
  br label %gcont

st.max:
  br i1 %hasany, label %st.max.v, label %st.null

st.max.v:
  %maxp = getelementptr inbounds double, ptr %odvals, i64 %gi
  store double %mmax, ptr %maxp, align 8
  br label %gcont

st.count:
  %cntp = getelementptr inbounds i64, ptr %odvals, i64 %gi
  store i64 %mcnt, ptr %cntp, align 8
  br label %gcont

st.nuq:
  ; distinct non-null values in [off,off1); O(m^2)
  br label %uloop

uloop:
  %ui = phi i64 [ %off, %st.nuq ], [ %uin, %ucont ]
  %uniq = phi i64 [ 0, %st.nuq ], [ %uniqn, %ucont ]
  %ucmp = icmp ult i64 %ui, %off1
  br i1 %ucmp, label %ubody, label %ustore

ubody:
  %urp = getelementptr inbounds i64, ptr %members, i64 %ui
  %urow = load i64, ptr %urp, align 8
  %uvalid = call i1 @gb_valid_at(ptr %as, i64 %urow)
  br i1 %uvalid, label %ucheck, label %ucont0

ucheck:
  %uval = call double @gb_elem_f64(ptr %avals, i32 %adt, i64 %urow)
  br label %pinner

; scan earlier members for an equal valid value
pinner:
  %pi = phi i64 [ %off, %ucheck ], [ %pin, %picont ]
  %seen = phi i1 [ false, %ucheck ], [ %seen2, %picont ]
  %pcmp = icmp ult i64 %pi, %ui
  %notseen = xor i1 %seen, true
  %doscan = and i1 %pcmp, %notseen
  br i1 %doscan, label %pbody, label %pdone

pbody:
  %prp = getelementptr inbounds i64, ptr %members, i64 %pi
  %prow = load i64, ptr %prp, align 8
  %pvalid = call i1 @gb_valid_at(ptr %as, i64 %prow)
  br i1 %pvalid, label %pcmpval, label %picont

pcmpval:
  %pval = call double @gb_elem_f64(ptr %avals, i32 %adt, i64 %prow)
  %pequal = fcmp oeq double %pval, %uval
  br label %picont

picont:
  %seen2 = phi i1 [ %seen, %pbody ], [ %pequal, %pcmpval ]
  %pin = add i64 %pi, 1
  br label %pinner

pdone:
  %isnew = xor i1 %seen, true
  %uinc = zext i1 %isnew to i64
  %uniq1 = add i64 %uniq, %uinc
  br label %ucont

ucont0:
  br label %ucont

ucont:
  %uniqn = phi i64 [ %uniq1, %pdone ], [ %uniq, %ucont0 ]
  %uin = add i64 %ui, 1
  br label %uloop

ustore:
  %nuqp = getelementptr inbounds i64, ptr %odvals, i64 %gi
  store i64 %uniq, ptr %nuqp, align 8
  br label %gcont

st.null:
  %snrc = call i32 @universe_dataframe_series_set_null(ptr %newser, i64 %gi)
  br label %gcont

gcont:
  %gin = add i64 %gi, 1
  br label %gloop

addcol:
  ; output name = "<colname>_<suffix>"
  %cnoff = mul i64 %acol, 16
  %cnslot = getelementptr inbounds i8, ptr %dfnames, i64 %cnoff
  %cnptr = load ptr, ptr %cnslot, align 8
  %cnlp = getelementptr inbounds i8, ptr %cnslot, i64 8
  %cnlen = load i64, ptr %cnlp, align 8
  call void @gb_op_suffix(i32 %aop, ptr %sfxpp, ptr %sfxlp)
  %sfxp = load ptr, ptr %sfxpp, align 8
  %sfxl = load i64, ptr %sfxlp, align 8
  %namelen = add i64 %cnlen, %sfxl
  %namelen1 = add i64 %namelen, 1
  %namebuf = call ptr @gb_xmalloc(i64 %namelen1)
  call void @llvm.memcpy.p0.p0.i64(ptr %namebuf, ptr %cnptr, i64 %cnlen, i1 false)
  %usp = getelementptr inbounds i8, ptr %namebuf, i64 %cnlen
  store i8 95, ptr %usp, align 1
  %usp1 = getelementptr inbounds i8, ptr %usp, i64 1
  call void @llvm.memcpy.p0.p0.i64(ptr %usp1, ptr %sfxp, i64 %sfxl, i1 false)
  %awrc = call i32 @universe_dataframe_with_column(ptr %out, ptr %namebuf, i64 %namelen1, ptr %newser)
  call void @free(ptr %namebuf)
  br label %accont

accont:
  %acn2 = add i64 %ac, 1
  br label %acols

finish:
  store ptr %out, ptr %out_df, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; public: partition_by
; ---------------------------------------------------------------------------
define i32 @universe_dataframe_partition_by(ptr %df, ptr %key_names, i64 %nk, ptr %out_frames, i64 %cap, ptr %out_n) local_unnamed_addr #0 {
entry:
  %ofn = icmp eq ptr %out_frames, null
  %onn = icmp eq ptr %out_n, null
  %e0 = or i1 %ofn, %onn
  br i1 %e0, label %err.null, label %buildpb

err.null:
  ret i32 1

buildpb:
  %gb = call ptr @gb_build(ptr %df, ptr %key_names, i64 %nk)
  %gbn = icmp eq ptr %gb, null
  br i1 %gbn, label %err.arg, label %got

err.arg:
  ret i32 8

got:
  %ngp = getelementptr inbounds i8, ptr %gb, i64 8
  %G = load i64, ptr %ngp, align 8
  store i64 %G, ptr %out_n, align 8
  %toobig = icmp ugt i64 %G, %cap
  br i1 %toobig, label %full, label %setup

full:
  call void @universe_dataframe_group_by_free(ptr %gb)
  ret i32 6

setup:
  %gofp = getelementptr inbounds i8, ptr %gb, i64 40
  %group_off = load ptr, ptr %gofp, align 8
  %memp = getelementptr inbounds i8, ptr %gb, i64 48
  %members = load ptr, ptr %memp, align 8
  %width = call i64 @universe_dataframe_width(ptr %df)
  %dfnamesp = getelementptr inbounds i8, ptr %df, i64 24
  %dfnames = load ptr, ptr %dfnamesp, align 8
  br label %gloop

gloop:
  %gi = phi i64 [ 0, %setup ], [ %gin, %gcont ]
  %gcmp = icmp ult i64 %gi, %G
  br i1 %gcmp, label %gbody, label %done

gbody:
  %offp = getelementptr inbounds i64, ptr %group_off, i64 %gi
  %off = load i64, ptr %offp, align 8
  %gi1 = add i64 %gi, 1
  %off1p = getelementptr inbounds i64, ptr %group_off, i64 %gi1
  %off1 = load i64, ptr %off1p, align 8
  %m = sub i64 %off1, %off
  %idxbase = getelementptr inbounds i64, ptr %members, i64 %off
  %sub = call ptr @universe_dataframe_new()
  br label %cloop

cloop:
  %cc = phi i64 [ 0, %gbody ], [ %ccn, %cccont ]
  %ccmp = icmp ult i64 %cc, %width
  br i1 %ccmp, label %ccbody, label %storeframe

ccbody:
  %cs = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %cc)
  %cnew = call ptr @gb_gather_series(ptr %cs, ptr %idxbase, i64 %m)
  %cnoff = mul i64 %cc, 16
  %cnslot = getelementptr inbounds i8, ptr %dfnames, i64 %cnoff
  %cnptr = load ptr, ptr %cnslot, align 8
  %cnlp = getelementptr inbounds i8, ptr %cnslot, i64 8
  %cnlen = load i64, ptr %cnlp, align 8
  %cwrc = call i32 @universe_dataframe_with_column(ptr %sub, ptr %cnptr, i64 %cnlen, ptr %cnew)
  br label %cccont

cccont:
  %ccn = add i64 %cc, 1
  br label %cloop

storeframe:
  %fslot = getelementptr inbounds ptr, ptr %out_frames, i64 %gi
  store ptr %sub, ptr %fslot, align 8
  br label %gcont

gcont:
  %gin = add i64 %gi, 1
  br label %gloop

done:
  call void @universe_dataframe_group_by_free(ptr %gb)
  ret i32 0
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind norecurse nosync }
attributes #2 = { alwaysinline nounwind norecurse nosync memory(read) }
