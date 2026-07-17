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

; DataFrame foundation: columnar Series (typed column) + DataFrame container.
; Single-chunk, contiguous, eager, struct-of-arrays. Functional inspiration
; from Polars' DataFrame (method NAMES kept); layout is our own.
;
; ============================ DOWNSTREAM CONTRACT ============================
; DType enum (i32): I32=0, I64=1, F32=2, F64=3, BOOL=4, STR=5.
;   fixed element widths: I32=4, I64=8, F32=4, F64=8, BOOL=1 (one byte/value,
;   0/1, SIMD-friendly). STR: `values` holds i32 offsets[len+1]; `strdata`
;   holds concatenated UTF-8 bytes; string i = strdata[offsets[i]..offsets[i+1]).
;
; Series handle = ONE malloc'd 56-byte header:
;   +0   i32  dtype
;   +4   (pad)
;   +8   i64  len
;   +16  i64  null_count
;   +24  ptr  values      ; fixed: len*width bytes ; STR: (len+1)*4 offset bytes
;   +32  ptr  validity    ; null if no nulls; else bitmap ceil(len/8) bytes, 1=valid
;   +40  ptr  strdata     ; STR only: concatenated bytes (else null)
;   +48  i64  strdata_len ; STR only: byte length of strdata (else 0)
; One buffer alloc per non-null array; validity lazily allocated on first null.
;
; DataFrame handle = ONE malloc'd 40-byte header:
;   +0   i64  n_cols
;   +8   i64  cap_cols     ; geometric growth (doubling)
;   +16  i64  height       ; every column len == height (enforced on add)
;   +24  ptr  names        ; array of {ptr name, i64 len} pairs (16 bytes each)
;   +32  ptr  columns      ; array of Series handles (8 bytes each)
; The DataFrame OWNS its column Series and name-byte copies. with_column /
; hstack / insert_column / replace_column TAKE OWNERSHIP of the passed Series.
; NEW-frame producers (slice/head/tail/reverse/shift/vstack/select/drop/
; with_row_index) CLONE columns for simple ownership (documented per Polars v1).
; height set by first column; empty_with_height presets it.
;
; Errors i32: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY, 5 NOT_FOUND,
;   7 INVALID_INDEX, 8 INVALID_ARG (also dtype/schema mismatch). Constructors
;   returning ptr signal failure with null.
; ===========================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.smax.i64(i64, i64)
declare i32 @memcmp(ptr captures(none), ptr captures(none), i64)

@df.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"

; ---------------------------------------------------------------------------
; internal helpers
; ---------------------------------------------------------------------------

define internal ptr @df_xmalloc(i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

define internal i64 @df_fixed_width(i32 %dtype) #5 {
entry:
  %i = zext i32 %dtype to i64
  %p = getelementptr inbounds [6 x i8], ptr @df.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

; returns { bytes, overflow } for the values buffer of (dtype,len)
define internal { i64, i1 } @df_valbytes(i32 %dtype, i64 %len) #3 {
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
  %w = call i64 @df_fixed_width(i32 %dtype)
  %fm = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %len, i64 %w)
  %fbytes = extractvalue { i64, i1 } %fm, 0
  %fo = extractvalue { i64, i1 } %fm, 1
  %fr0 = insertvalue { i64, i1 } undef, i64 %fbytes, 0
  %fr = insertvalue { i64, i1 } %fr0, i1 %fo, 1
  ret { i64, i1 } %fr
}

define internal ptr @series_hdr_alloc() #1 {
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

; allocate a validity bitmap (all valid = 0xFF) for len elements
define internal ptr @df_bm_alloc(i64 %len) #1 {
entry:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %p = call ptr @df_xmalloc(i64 %nb)
  call void @llvm.memset.p0.i64(ptr %p, i8 -1, i64 %nb, i1 false)
  ret ptr %p
}

define internal void @df_bm_clear(ptr %bm, i64 %k) #4 {
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

; true if element i is valid (non-null). validity==null => all valid.
define internal i1 @series_valid_at(ptr %s, i64 %i) #6 {
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
; Series primitives
; ---------------------------------------------------------------------------

define ptr @universe_dataframe_series_new(i32 %dtype, i64 %len) local_unnamed_addr #1 {
entry:
  %bad = icmp ugt i32 %dtype, 5
  br i1 %bad, label %fail, label %vb, !prof !0

vb:
  %vbr = call { i64, i1 } @df_valbytes(i32 %dtype, i64 %len)
  %bytes = extractvalue { i64, i1 } %vbr, 0
  %ov = extractvalue { i64, i1 } %vbr, 1
  br i1 %ov, label %fail, label %alloc, !prof !0

alloc:
  %s = call ptr @series_hdr_alloc()
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %fail, label %setv, !prof !0

setv:
  store i32 %dtype, ptr %s, align 8
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  store i64 %len, ptr %lp, align 8
  %vals = call ptr @df_xmalloc(i64 %bytes)
  %valn = icmp eq ptr %vals, null
  br i1 %valn, label %freehdr, label %storev, !prof !0

storev:
  call void @llvm.memset.p0.i64(ptr %vals, i8 0, i64 %bytes, i1 false)
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  store ptr %vals, ptr %vpp, align 8
  ret ptr %s

freehdr:
  call void @free(ptr %s)
  br label %fail

fail:
  ret ptr null
}

define ptr @universe_dataframe_series_from(i32 %dtype, ptr %values, i64 %len) local_unnamed_addr #1 {
entry:
  %bad = icmp ugt i32 %dtype, 5
  %isstr = icmp eq i32 %dtype, 5
  %vn = icmp eq ptr %values, null
  %bad2 = or i1 %bad, %isstr
  %bad3 = or i1 %bad2, %vn
  br i1 %bad3, label %fail, label %build, !prof !0

build:
  %s = call ptr @universe_dataframe_series_new(i32 %dtype, i64 %len)
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %fail, label %copy, !prof !0

copy:
  %vbr = call { i64, i1 } @df_valbytes(i32 %dtype, i64 %len)
  %bytes = extractvalue { i64, i1 } %vbr, 0
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %dst = load ptr, ptr %vpp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %values, i64 %bytes, i1 false)
  ret ptr %s

fail:
  ret ptr null
}

define ptr @universe_dataframe_series_str_new(ptr %offsets, ptr %data, i64 %len, i64 %data_len) local_unnamed_addr #1 {
entry:
  %on = icmp eq ptr %offsets, null
  br i1 %on, label %fail, label %alloc, !prof !0

alloc:
  %obr = call { i64, i1 } @df_valbytes(i32 5, i64 %len)
  %obytes = extractvalue { i64, i1 } %obr, 0
  %oov = extractvalue { i64, i1 } %obr, 1
  br i1 %oov, label %fail, label %hdr, !prof !0

hdr:
  %s = call ptr @series_hdr_alloc()
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %fail, label %fill, !prof !0

fill:
  store i32 5, ptr %s, align 8
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  store i64 %len, ptr %lp, align 8
  %vals = call ptr @df_xmalloc(i64 %obytes)
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  store ptr %vals, ptr %vpp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %vals, ptr %offsets, i64 %obytes, i1 false)
  %sd = call ptr @df_xmalloc(i64 %data_len)
  %sdp = getelementptr inbounds i8, ptr %s, i64 40
  store ptr %sd, ptr %sdp, align 8
  %hasdata = icmp ugt i64 %data_len, 0
  br i1 %hasdata, label %copydata, label %setlen

copydata:
  %datan = icmp eq ptr %data, null
  br i1 %datan, label %setlen, label %docopy

docopy:
  call void @llvm.memcpy.p0.p0.i64(ptr %sd, ptr %data, i64 %data_len, i1 false)
  br label %setlen

setlen:
  %sdlp = getelementptr inbounds i8, ptr %s, i64 48
  store i64 %data_len, ptr %sdlp, align 8
  ret ptr %s

fail:
  ret ptr null
}

define void @universe_dataframe_series_free(ptr %s) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %done, label %free, !prof !0

free:
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  call void @free(ptr %vals)
  %vlp = getelementptr inbounds i8, ptr %s, i64 32
  %vld = load ptr, ptr %vlp, align 8
  call void @free(ptr %vld)
  %sdp = getelementptr inbounds i8, ptr %s, i64 40
  %sd = load ptr, ptr %sdp, align 8
  call void @free(ptr %sd)
  call void @free(ptr %s)
  br label %done

done:
  ret void
}

define i64 @universe_dataframe_series_len(ptr %s) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %zero, label %load

load:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %l = load i64, ptr %lp, align 8
  ret i64 %l

zero:
  ret i64 0
}

define i32 @universe_dataframe_series_dtype(ptr %s) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %neg, label %load

load:
  %d = load i32, ptr %s, align 8
  ret i32 %d

neg:
  ret i32 -1
}

define ptr @universe_dataframe_series_values(ptr %s) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %null, label %load

load:
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %v = load ptr, ptr %vpp, align 8
  ret ptr %v

null:
  ret ptr null
}

define ptr @universe_dataframe_series_validity(ptr %s) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %null, label %load

load:
  %vpp = getelementptr inbounds i8, ptr %s, i64 32
  %v = load ptr, ptr %vpp, align 8
  ret ptr %v

null:
  ret ptr null
}

define i64 @universe_dataframe_series_null_count(ptr %s) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %zero, label %load

load:
  %cp = getelementptr inbounds i8, ptr %s, i64 16
  %c = load i64, ptr %cp, align 8
  ret i64 %c

zero:
  ret i64 0
}

define i32 @universe_dataframe_series_set_null(ptr %s, i64 %i) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %oob = icmp uge i64 %i, %len
  br i1 %oob, label %err.idx, label %ensure, !prof !0

err.idx:
  ret i32 7

ensure:
  %vlp = getelementptr inbounds i8, ptr %s, i64 32
  %bm0 = load ptr, ptr %vlp, align 8
  %hasbm = icmp ne ptr %bm0, null
  br i1 %hasbm, label %have, label %make

make:
  %bmn = call ptr @df_bm_alloc(i64 %len)
  store ptr %bmn, ptr %vlp, align 8
  br label %have

have:
  %bm = phi ptr [ %bm0, %ensure ], [ %bmn, %make ]
  %bi = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %bm, i64 %bi
  %b = load i8, ptr %bp, align 1
  %sh = and i64 %i, 7
  %sh8 = trunc i64 %sh to i8
  %m = shl i8 1, %sh8
  %isvalid = and i8 %b, %m
  %wasvalid = icmp ne i8 %isvalid, 0
  br i1 %wasvalid, label %clear, label %done

clear:
  %nm = xor i8 %m, -1
  %b2 = and i8 %b, %nm
  store i8 %b2, ptr %bp, align 1
  %cp = getelementptr inbounds i8, ptr %s, i64 16
  %c = load i64, ptr %cp, align 8
  %c2 = add i64 %c, 1
  store i64 %c2, ptr %cp, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_dataframe_series_is_null(ptr %s, i64 %i, ptr %out_bool) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %obn = icmp eq ptr %out_bool, null
  %anynull = or i1 %sn, %obn
  br i1 %anynull, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %oob = icmp uge i64 %i, %len
  br i1 %oob, label %err.idx, label %do, !prof !0

err.idx:
  ret i32 7

do:
  %valid = call i1 @series_valid_at(ptr %s, i64 %i)
  %isnull = xor i1 %valid, true
  %v8 = zext i1 %isnull to i8
  store i8 %v8, ptr %out_bool, align 1
  ret i32 0
}

define ptr @universe_dataframe_series_clone(ptr %s) local_unnamed_addr #1 {
entry:
  %n = icmp eq ptr %s, null
  br i1 %n, label %fail, label %read, !prof !0

read:
  %dtype = load i32, ptr %s, align 8
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %ncp = getelementptr inbounds i8, ptr %s, i64 16
  %nc = load i64, ptr %ncp, align 8
  %sdlp = getelementptr inbounds i8, ptr %s, i64 48
  %sdl = load i64, ptr %sdlp, align 8
  %out = call ptr @series_hdr_alloc()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %fill, !prof !0

fill:
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %len, ptr %olp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nc, ptr %oncp, align 8
  %osdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %sdl, ptr %osdlp, align 8
  %vbr = call { i64, i1 } @df_valbytes(i32 %dtype, i64 %len)
  %vbytes = extractvalue { i64, i1 } %vbr, 0
  %ovals = call ptr @df_xmalloc(i64 %vbytes)
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %ovals, ptr %ovpp, align 8
  %svpp = getelementptr inbounds i8, ptr %s, i64 24
  %svals = load ptr, ptr %svpp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %ovals, ptr %svals, i64 %vbytes, i1 false)
  %svlp = getelementptr inbounds i8, ptr %s, i64 32
  %svld = load ptr, ptr %svlp, align 8
  %hasvld = icmp ne ptr %svld, null
  br i1 %hasvld, label %copyvld, label %strchk

copyvld:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %ovld = call ptr @df_xmalloc(i64 %nb)
  %ovlp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %ovld, ptr %ovlp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %ovld, ptr %svld, i64 %nb, i1 false)
  br label %strchk

strchk:
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %copystr, label %done

copystr:
  %ssdp = getelementptr inbounds i8, ptr %s, i64 40
  %ssd = load ptr, ptr %ssdp, align 8
  %hassd = icmp ne ptr %ssd, null
  br i1 %hassd, label %dostr, label %done

dostr:
  %osd = call ptr @df_xmalloc(i64 %sdl)
  %osdp = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %osd, ptr %osdp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %osd, ptr %ssd, i64 %sdl, i1 false)
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_series_str_get(ptr %s, i64 %i, ptr %out_ptr, ptr %out_len) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %opn = icmp eq ptr %out_ptr, null
  %oln = icmp eq ptr %out_len, null
  %n1 = or i1 %sn, %opn
  %n2 = or i1 %n1, %oln
  br i1 %n2, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %dtype = load i32, ptr %s, align 8
  %notstr = icmp ne i32 %dtype, 5
  br i1 %notstr, label %err.arg, label %chklen, !prof !0

err.arg:
  ret i32 8

chklen:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %oob = icmp uge i64 %i, %len
  br i1 %oob, label %err.idx, label %do, !prof !0

err.idx:
  ret i32 7

do:
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %offs = load ptr, ptr %vpp, align 8
  %o0p = getelementptr inbounds i32, ptr %offs, i64 %i
  %o0 = load i32, ptr %o0p, align 4
  %i1 = add i64 %i, 1
  %o1p = getelementptr inbounds i32, ptr %offs, i64 %i1
  %o1 = load i32, ptr %o1p, align 4
  %o0z = zext i32 %o0 to i64
  %o1z = zext i32 %o1 to i64
  %slen = sub i64 %o1z, %o0z
  %sdp = getelementptr inbounds i8, ptr %s, i64 40
  %sd = load ptr, ptr %sdp, align 8
  %strp = getelementptr inbounds i8, ptr %sd, i64 %o0z
  store ptr %strp, ptr %out_ptr, align 8
  store i64 %slen, ptr %out_len, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; series_gather: build a NEW series = rows selected by idx[0..n) (i64 each);
; idx entry < 0 => a null / empty element. Used by all row-permutation ops.
; ---------------------------------------------------------------------------

define internal ptr @series_gather(ptr %s, ptr %idx, i64 %n) #1 {
entry:
  %dtype = load i32, ptr %s, align 8
  %out = call ptr @series_hdr_alloc()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %head, !prof !0

head:
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %n, ptr %olp, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %w = call i64 @df_fixed_width(i32 %dtype)
  %fbytes = mul i64 %n, %w
  %fvals = call ptr @df_xmalloc(i64 %fbytes)
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
  %f1.valid = call i1 @series_valid_at(ptr %s, i64 %f1.j)
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
  %f.bm = call ptr @df_bm_alloc(i64 %n)
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
  call void @df_bm_clear(ptr %f2.bm, i64 %f2.k)
  br label %f2.cont

f2.copy:
  %f2.soff = mul i64 %f2.j, %w
  %f2.src = getelementptr inbounds i8, ptr %svals, i64 %f2.soff
  call void @llvm.memcpy.p0.p0.i64(ptr %f2.dst, ptr %f2.src, i64 %w, i1 false)
  %f2.valid = call i1 @series_valid_at(ptr %s, i64 %f2.j)
  br i1 %f2.valid, label %f2.cont, label %f2.mknull

f2.mknull:
  call void @df_bm_clear(ptr %f2.bm, i64 %f2.k)
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
  %s1.valid = call i1 @series_valid_at(ptr %s, i64 %s1.j)
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
  %s.obr = call { i64, i1 } @df_valbytes(i32 5, i64 %n)
  %s.obytes = extractvalue { i64, i1 } %s.obr, 0
  %s.offs = call ptr @df_xmalloc(i64 %s.obytes)
  %s.ovpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %s.offs, ptr %s.ovpp, align 8
  %s.sd = call ptr @df_xmalloc(i64 %s1.total)
  %s.sdp = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %s.sd, ptr %s.sdp, align 8
  %s.sdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %s1.total, ptr %s.sdlp, align 8
  %s.hasnull = icmp ugt i64 %s1.nulls, 0
  br i1 %s.hasnull, label %s.mkbm, label %s2.head

s.mkbm:
  %s.bm = call ptr @df_bm_alloc(i64 %n)
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
  call void @df_bm_clear(ptr %s2.bm, i64 %s2.k)
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
  %s2.valid = call i1 @series_valid_at(ptr %s, i64 %s2.j)
  br i1 %s2.valid, label %s2.cont1, label %s2.mknull

s2.mknull:
  call void @df_bm_clear(ptr %s2.bm, i64 %s2.k)
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

; series_concat: NEW series = a followed by b (same dtype assumed by caller)
define internal ptr @series_concat(ptr %a, ptr %b) #1 {
entry:
  %dtype = load i32, ptr %a, align 8
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %n = add i64 %la, %lb
  %ancp = getelementptr inbounds i8, ptr %a, i64 16
  %anc = load i64, ptr %ancp, align 8
  %bncp = getelementptr inbounds i8, ptr %b, i64 16
  %bnc = load i64, ptr %bncp, align 8
  %nc = add i64 %anc, %bnc
  %out = call ptr @series_hdr_alloc()
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %head

head:
  store i32 %dtype, ptr %out, align 8
  %olp = getelementptr inbounds i8, ptr %out, i64 8
  store i64 %n, ptr %olp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nc, ptr %oncp, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %str, label %fixed

fixed:
  %w = call i64 @df_fixed_width(i32 %dtype)
  %abytes = mul i64 %la, %w
  %bbytes = mul i64 %lb, %w
  %tbytes = mul i64 %n, %w
  %fvals = call ptr @df_xmalloc(i64 %tbytes)
  %fvpp = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %fvals, ptr %fvpp, align 8
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %bvpp = getelementptr inbounds i8, ptr %b, i64 24
  %bvals = load ptr, ptr %bvpp, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %fvals, ptr %avals, i64 %abytes, i1 false)
  %fvals.b = getelementptr inbounds i8, ptr %fvals, i64 %abytes
  call void @llvm.memcpy.p0.p0.i64(ptr %fvals.b, ptr %bvals, i64 %bbytes, i1 false)
  br label %validity

str:
  %asdlp = getelementptr inbounds i8, ptr %a, i64 48
  %asdl = load i64, ptr %asdlp, align 8
  %bsdlp = getelementptr inbounds i8, ptr %b, i64 48
  %bsdl = load i64, ptr %bsdlp, align 8
  %total = add i64 %asdl, %bsdl
  %obr = call { i64, i1 } @df_valbytes(i32 5, i64 %n)
  %obytes = extractvalue { i64, i1 } %obr, 0
  %soffs = call ptr @df_xmalloc(i64 %obytes)
  %svpp2 = getelementptr inbounds i8, ptr %out, i64 24
  store ptr %soffs, ptr %svpp2, align 8
  %ssd = call ptr @df_xmalloc(i64 %total)
  %ssdp = getelementptr inbounds i8, ptr %out, i64 40
  store ptr %ssd, ptr %ssdp, align 8
  %ssdlp = getelementptr inbounds i8, ptr %out, i64 48
  store i64 %total, ptr %ssdlp, align 8
  %aoffp = getelementptr inbounds i8, ptr %a, i64 24
  %aoff = load ptr, ptr %aoffp, align 8
  %boffp = getelementptr inbounds i8, ptr %b, i64 24
  %boff = load ptr, ptr %boffp, align 8
  %acopyn = add i64 %la, 1
  %acopyb = mul i64 %acopyn, 4
  call void @llvm.memcpy.p0.p0.i64(ptr %soffs, ptr %aoff, i64 %acopyb, i1 false)
  br label %bo.head

bo.head:
  %bo.k = phi i64 [ 1, %str ], [ %bo.kn, %bo.body ]
  %bo.cmp = icmp ule i64 %bo.k, %lb
  br i1 %bo.cmp, label %bo.body, label %sdata

bo.body:
  %bo.sp = getelementptr inbounds i32, ptr %boff, i64 %bo.k
  %bo.v = load i32, ptr %bo.sp, align 4
  %bo.vz = zext i32 %bo.v to i64
  %bo.biased = add i64 %bo.vz, %asdl
  %bo.b32 = trunc i64 %bo.biased to i32
  %bo.di = add i64 %la, %bo.k
  %bo.dp = getelementptr inbounds i32, ptr %soffs, i64 %bo.di
  store i32 %bo.b32, ptr %bo.dp, align 4
  %bo.kn = add i64 %bo.k, 1
  br label %bo.head

sdata:
  %asd.p = getelementptr inbounds i8, ptr %a, i64 40
  %asd = load ptr, ptr %asd.p, align 8
  %bsd.p = getelementptr inbounds i8, ptr %b, i64 40
  %bsd = load ptr, ptr %bsd.p, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %ssd, ptr %asd, i64 %asdl, i1 false)
  %ssd.b = getelementptr inbounds i8, ptr %ssd, i64 %asdl
  call void @llvm.memcpy.p0.p0.i64(ptr %ssd.b, ptr %bsd, i64 %bsdl, i1 false)
  br label %validity

validity:
  %needbm = icmp ugt i64 %nc, 0
  br i1 %needbm, label %mkbm, label %done

mkbm:
  %bm = call ptr @df_bm_alloc(i64 %n)
  %bmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %bmp, align 8
  br label %va.head

va.head:
  %va.k = phi i64 [ 0, %mkbm ], [ %va.kn, %va.cont ]
  %va.cmp = icmp ult i64 %va.k, %la
  br i1 %va.cmp, label %va.body, label %vb.head

va.body:
  %va.valid = call i1 @series_valid_at(ptr %a, i64 %va.k)
  br i1 %va.valid, label %va.cont, label %va.clear

va.clear:
  call void @df_bm_clear(ptr %bm, i64 %va.k)
  br label %va.cont

va.cont:
  %va.kn = add i64 %va.k, 1
  br label %va.head

vb.head:
  %vb.k = phi i64 [ 0, %va.head ], [ %vb.kn, %vb.cont ]
  %vb.cmp = icmp ult i64 %vb.k, %lb
  br i1 %vb.cmp, label %vb.body, label %done

vb.body:
  %vb.valid = call i1 @series_valid_at(ptr %b, i64 %vb.k)
  br i1 %vb.valid, label %vb.cont, label %vb.clear

vb.clear:
  %vb.di = add i64 %la, %vb.k
  call void @df_bm_clear(ptr %bm, i64 %vb.di)
  br label %vb.cont

vb.cont:
  %vb.kn = add i64 %vb.k, 1
  br label %vb.head

done:
  ret ptr %out

fail:
  ret ptr null
}

; series_equals: true if same dtype/len and same values+null pattern
define internal i1 @series_equals(ptr %a, ptr %b) #3 {
entry:
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %dne = icmp ne i32 %da, %db
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %lne = icmp ne i64 %la, %lb
  %hdrne = or i1 %dne, %lne
  br i1 %hdrne, label %ret.false, label %setup

setup:
  %isstr = icmp eq i32 %da, 5
  %aoffp = getelementptr inbounds i8, ptr %a, i64 24
  %aoff = load ptr, ptr %aoffp, align 8
  %boffp = getelementptr inbounds i8, ptr %b, i64 24
  %boff = load ptr, ptr %boffp, align 8
  %asdp = getelementptr inbounds i8, ptr %a, i64 40
  %asd = load ptr, ptr %asdp, align 8
  %bsdp = getelementptr inbounds i8, ptr %b, i64 40
  %bsd = load ptr, ptr %bsdp, align 8
  %w = call i64 @df_fixed_width(i32 %da)
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %la
  br i1 %cmp, label %loop.body, label %ret.true

loop.body:
  %av = call i1 @series_valid_at(ptr %a, i64 %k)
  %bv = call i1 @series_valid_at(ptr %b, i64 %k)
  %vne = xor i1 %av, %bv
  br i1 %vne, label %ret.false, label %bothsame

bothsame:
  br i1 %av, label %cmpval, label %loop.cont

cmpval:
  br i1 %isstr, label %cmpstr, label %cmpfix

cmpfix:
  %f.off = mul i64 %k, %w
  %f.ap = getelementptr inbounds i8, ptr %aoff, i64 %f.off
  %f.bp = getelementptr inbounds i8, ptr %boff, i64 %f.off
  %f.r = call i32 @memcmp(ptr %f.ap, ptr %f.bp, i64 %w)
  %f.ne = icmp ne i32 %f.r, 0
  br i1 %f.ne, label %ret.false, label %loop.cont

cmpstr:
  %k1 = add i64 %k, 1
  %sa.o0p = getelementptr inbounds i32, ptr %aoff, i64 %k
  %sa.o0 = load i32, ptr %sa.o0p, align 4
  %sa.o1p = getelementptr inbounds i32, ptr %aoff, i64 %k1
  %sa.o1 = load i32, ptr %sa.o1p, align 4
  %sb.o0p = getelementptr inbounds i32, ptr %boff, i64 %k
  %sb.o0 = load i32, ptr %sb.o0p, align 4
  %sb.o1p = getelementptr inbounds i32, ptr %boff, i64 %k1
  %sb.o1 = load i32, ptr %sb.o1p, align 4
  %sa.len = sub i32 %sa.o1, %sa.o0
  %sb.len = sub i32 %sb.o1, %sb.o0
  %slne = icmp ne i32 %sa.len, %sb.len
  br i1 %slne, label %ret.false, label %cmpstrbytes

cmpstrbytes:
  %sa.o0z = zext i32 %sa.o0 to i64
  %sb.o0z = zext i32 %sb.o0 to i64
  %sa.lenz = zext i32 %sa.len to i64
  %sa.p = getelementptr inbounds i8, ptr %asd, i64 %sa.o0z
  %sb.p = getelementptr inbounds i8, ptr %bsd, i64 %sb.o0z
  %s.r = call i32 @memcmp(ptr %sa.p, ptr %sb.p, i64 %sa.lenz)
  %s.ne = icmp ne i32 %s.r, 0
  br i1 %s.ne, label %ret.false, label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

ret.true:
  ret i1 true

ret.false:
  ret i1 false
}

; ---------------------------------------------------------------------------
; DataFrame internal helpers
; ---------------------------------------------------------------------------

define internal ptr @df_new_empty(i64 %height, i64 %capcols) #1 {
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

define internal i1 @df_ensure_cap(ptr %df, i64 %need) #1 {
entry:
  %capp = getelementptr inbounds i8, ptr %df, i64 8
  %cap = load i64, ptr %capp, align 8
  %ok = icmp uge i64 %cap, %need
  br i1 %ok, label %retok, label %grow

grow:
  %cap2 = shl i64 %cap, 1
  %newcap = call i64 @llvm.umax.i64(i64 %cap2, i64 %need)
  %nb = mul i64 %newcap, 16
  %cb = mul i64 %newcap, 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %oldnames = load ptr, ptr %np, align 8
  %newnames = call ptr @realloc(ptr %oldnames, i64 %nb)
  %nn = icmp eq ptr %newnames, null
  br i1 %nn, label %retbad, label %growcols

growcols:
  store ptr %newnames, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %oldcols = load ptr, ptr %cp, align 8
  %newcols = call ptr @realloc(ptr %oldcols, i64 %cb)
  %cn = icmp eq ptr %newcols, null
  br i1 %cn, label %retbad, label %commit

commit:
  store ptr %newcols, ptr %cp, align 8
  store i64 %newcap, ptr %capp, align 8
  br label %retok

retok:
  ret i1 true

retbad:
  ret i1 false
}

define internal ptr @df_name_dup(ptr %name, i64 %nl) #1 {
entry:
  %p = call ptr @df_xmalloc(i64 %nl)
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %name, i64 %nl, i1 false)
  ret ptr %p
}

; append a column (TAKES OWNERSHIP of series, COPIES name). enforces height.
define internal i32 @df_append_col(ptr %df, ptr %name, i64 %nl, ptr %series) #1 {
entry:
  %ncp = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp, align 8
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
  br label %grow

checkh:
  %hmatch = icmp eq i64 %slen, %height
  br i1 %hmatch, label %grow, label %err.arg, !prof !0

err.arg:
  ret i32 8

grow:
  %need = add i64 %nc, 1
  %grown = call i1 @df_ensure_cap(ptr %df, i64 %need)
  br i1 %grown, label %store, label %err.oom, !prof !0

err.oom:
  ret i32 2

store:
  %namecopy = call ptr @df_name_dup(ptr %name, i64 %nl)
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
  store i64 %ncn, ptr %ncp, align 8
  ret i32 0
}

; find column index by name; returns index or -1
define internal i64 @df_find(ptr %df, ptr %name, i64 %nl) #3 {
entry:
  %ncp = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp, align 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %notfound

loop.body:
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %lenmatch = icmp eq i64 %nlen, %nl
  br i1 %lenmatch, label %cmpbytes, label %loop.cont

cmpbytes:
  %r = call i32 @memcmp(ptr %nptr, ptr %name, i64 %nl)
  %eq = icmp eq i32 %r, 0
  br i1 %eq, label %found, label %loop.cont

found:
  ret i64 %k

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

notfound:
  ret i64 -1
}

; gather ALL columns of df by idx[0..n) -> NEW df of height n
define internal ptr @df_gather(ptr %df, ptr %idx, i64 %n) #1 {
entry:
  %ncp = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp, align 8
  %out = call ptr @df_new_empty(i64 %n, i64 %nc)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %setup, !prof !0

setup:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @series_gather(ptr %col, ptr %idx, i64 %n)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @df_append_col(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret ptr %out

fail:
  ret ptr null
}

; build an index array [start, start+1, ... start+n-1] (contiguous)
define internal ptr @df_idx_range(i64 %start, i64 %n) #1 {
entry:
  %bytes = mul i64 %n, 8
  %idx = call ptr @df_xmalloc(i64 %bytes)
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop.body ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %v = add i64 %start, %k
  %p = getelementptr inbounds i64, ptr %idx, i64 %k
  store i64 %v, ptr %p, align 8
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret ptr %idx
}

; ---------------------------------------------------------------------------
; DataFrame public API
; ---------------------------------------------------------------------------

define ptr @universe_dataframe_new() local_unnamed_addr #1 {
entry:
  %df = call ptr @df_new_empty(i64 0, i64 8)
  ret ptr %df
}

define ptr @universe_dataframe_empty_with_height(i64 %h) local_unnamed_addr #1 {
entry:
  %df = call ptr @df_new_empty(i64 %h, i64 8)
  ret ptr %df
}

define i64 @universe_dataframe_height(ptr %df) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %df, null
  br i1 %n, label %zero, label %load

load:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %h = load i64, ptr %hp, align 8
  ret i64 %h

zero:
  ret i64 0
}

define i64 @universe_dataframe_width(ptr %df) local_unnamed_addr #2 {
entry:
  %n = icmp eq ptr %df, null
  br i1 %n, label %zero, label %load

load:
  %w = load i64, ptr %df, align 8
  ret i64 %w

zero:
  ret i64 0
}

define i32 @universe_dataframe_shape(ptr %df, ptr %out_h, ptr %out_w) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %hn = icmp eq ptr %out_h, null
  %wn = icmp eq ptr %out_w, null
  %n1 = or i1 %dn, %hn
  %n2 = or i1 %n1, %wn
  br i1 %n2, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %h = load i64, ptr %hp, align 8
  %w = load i64, ptr %df, align 8
  store i64 %h, ptr %out_h, align 8
  store i64 %w, ptr %out_w, align 8
  ret i32 0
}

define i32 @universe_dataframe_get_column_names(ptr %df, ptr %out_names, i64 %cap, ptr %out_n) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %on = icmp eq ptr %out_names, null
  %nn = icmp eq ptr %out_n, null
  %e1 = or i1 %dn, %on
  %e2 = or i1 %e1, %nn
  br i1 %e2, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %nc = load i64, ptr %df, align 8
  store i64 %nc, ptr %out_n, align 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %lim = call i64 @llvm.umin.i64(i64 %nc, i64 %cap)
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %lim
  br i1 %cmp, label %loop.cont, label %done

loop.cont:
  %soff = mul i64 %k, 16
  %sslot = getelementptr inbounds i8, ptr %names, i64 %soff
  %sptr = load ptr, ptr %sslot, align 8
  %slenslot = getelementptr inbounds i8, ptr %sslot, i64 8
  %slen = load i64, ptr %slenslot, align 8
  %dslot = getelementptr inbounds i8, ptr %out_names, i64 %soff
  store ptr %sptr, ptr %dslot, align 8
  %dlenslot = getelementptr inbounds i8, ptr %dslot, i64 8
  store i64 %slen, ptr %dlenslot, align 8
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret i32 0
}

define i32 @universe_dataframe_dtypes(ptr %df, ptr %out_i32, i64 %cap) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %on = icmp eq ptr %out_i32, null
  %e = or i1 %dn, %on
  br i1 %e, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %nc = load i64, ptr %df, align 8
  %lim = call i64 @llvm.umin.i64(i64 %nc, i64 %cap)
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %lim
  br i1 %cmp, label %loop.cont, label %done

loop.cont:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %dt = load i32, ptr %col, align 8
  %dslot = getelementptr inbounds i32, ptr %out_i32, i64 %k
  store i32 %dt, ptr %dslot, align 4
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret i32 0
}

define i32 @universe_dataframe_get_column_index(ptr %df, ptr %name, i64 %name_len, ptr %out_idx) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %on = icmp eq ptr %out_idx, null
  %e1 = or i1 %dn, %nn
  %e2 = or i1 %e1, %on
  br i1 %e2, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %idx = call i64 @df_find(ptr %df, ptr %name, i64 %name_len)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %notfound, label %found, !prof !0

notfound:
  ret i32 5

found:
  store i64 %idx, ptr %out_idx, align 8
  ret i32 0
}

define ptr @universe_dataframe_column(ptr %df, ptr %name, i64 %name_len) local_unnamed_addr #2 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %e = or i1 %dn, %nn
  br i1 %e, label %null, label %do

do:
  %idx = call i64 @df_find(ptr %df, ptr %name, i64 %name_len)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %null, label %found

found:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %idx
  %col = load ptr, ptr %cslot, align 8
  ret ptr %col

null:
  ret ptr null
}

define ptr @universe_dataframe_select_at_idx(ptr %df, i64 %i) local_unnamed_addr #2 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %null, label %do

do:
  %nc = load i64, ptr %df, align 8
  %oob = icmp uge i64 %i, %nc
  br i1 %oob, label %null, label %found

found:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %i
  %col = load ptr, ptr %cslot, align 8
  ret ptr %col

null:
  ret ptr null
}

define ptr @universe_dataframe_select(ptr %df, ptr %names, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %setup

setup:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %out = call ptr @df_new_empty(i64 %height, i64 %n)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %soff = mul i64 %k, 16
  %sslot = getelementptr inbounds i8, ptr %names, i64 %soff
  %sptr = load ptr, ptr %sslot, align 8
  %slenslot = getelementptr inbounds i8, ptr %sslot, i64 8
  %slen = load i64, ptr %slenslot, align 8
  %idx = call i64 @df_find(ptr %df, ptr %sptr, i64 %slen)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %freefail, label %clone

clone:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %idx
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @universe_dataframe_series_clone(ptr %col)
  %rc = call i32 @df_append_col(ptr %out, ptr %sptr, i64 %slen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

freefail:
  call void @universe_dataframe_free(ptr %out)
  br label %fail

done:
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_with_column(ptr %df, ptr %name, i64 %name_len, ptr %series) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %sn = icmp eq ptr %series, null
  %e1 = or i1 %dn, %nn
  %e2 = or i1 %e1, %sn
  br i1 %e2, label %err.null, label %find, !prof !0

err.null:
  ret i32 1

find:
  %idx = call i64 @df_find(ptr %df, ptr %name, i64 %name_len)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %append, label %replace

append:
  %rc = call i32 @df_append_col(ptr %df, ptr %name, i64 %name_len, ptr %series)
  ret i32 %rc

replace:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %slp = getelementptr inbounds i8, ptr %series, i64 8
  %slen = load i64, ptr %slp, align 8
  %hmatch = icmp eq i64 %slen, %height
  br i1 %hmatch, label %do, label %err.arg, !prof !0

err.arg:
  ret i32 8

do:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %idx
  %old = load ptr, ptr %cslot, align 8
  call void @universe_dataframe_series_free(ptr %old)
  store ptr %series, ptr %cslot, align 8
  ret i32 0
}

define i32 @universe_dataframe_insert_column(ptr %df, i64 %at, ptr %name, i64 %nl, ptr %series) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %sn = icmp eq ptr %series, null
  %e1 = or i1 %dn, %nn
  %e2 = or i1 %e1, %sn
  br i1 %e2, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %ncp = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp, align 8
  %oob = icmp ugt i64 %at, %nc
  br i1 %oob, label %err.idx, label %chkh, !prof !0

err.idx:
  ret i32 7

chkh:
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
  br label %grow

checkh:
  %hmatch = icmp eq i64 %slen, %height
  br i1 %hmatch, label %grow, label %err.arg, !prof !0

err.arg:
  ret i32 8

grow:
  %need = add i64 %nc, 1
  %grown = call i1 @df_ensure_cap(ptr %df, i64 %need)
  br i1 %grown, label %shift, label %err.oom, !prof !0

err.oom:
  ret i32 2

shift:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %tail = sub i64 %nc, %at
  %natoff = mul i64 %at, 16
  %nsrc = getelementptr inbounds i8, ptr %names, i64 %natoff
  %ndst = getelementptr inbounds i8, ptr %nsrc, i64 16
  %nbytes = mul i64 %tail, 16
  call void @llvm.memmove.p0.p0.i64(ptr %ndst, ptr %nsrc, i64 %nbytes, i1 false)
  %catoff = mul i64 %at, 8
  %csrc = getelementptr inbounds i8, ptr %cols, i64 %catoff
  %cdst = getelementptr inbounds i8, ptr %csrc, i64 8
  %cbytes = mul i64 %tail, 8
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %cbytes, i1 false)
  %namecopy = call ptr @df_name_dup(ptr %name, i64 %nl)
  store ptr %namecopy, ptr %nsrc, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nsrc, i64 8
  store i64 %nl, ptr %nlenslot, align 8
  store ptr %series, ptr %csrc, align 8
  %ncn = add i64 %nc, 1
  store i64 %ncn, ptr %ncp, align 8
  ret i32 0
}

define i32 @universe_dataframe_replace_column(ptr %df, i64 %i, ptr %series) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %sn = icmp eq ptr %series, null
  %e = or i1 %dn, %sn
  br i1 %e, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %nc = load i64, ptr %df, align 8
  %oob = icmp uge i64 %i, %nc
  br i1 %oob, label %err.idx, label %chkh, !prof !0

err.idx:
  ret i32 7

chkh:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %slp = getelementptr inbounds i8, ptr %series, i64 8
  %slen = load i64, ptr %slp, align 8
  %hmatch = icmp eq i64 %slen, %height
  br i1 %hmatch, label %do, label %err.arg, !prof !0

err.arg:
  ret i32 8

do:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %i
  %old = load ptr, ptr %cslot, align 8
  call void @universe_dataframe_series_free(ptr %old)
  store ptr %series, ptr %cslot, align 8
  ret i32 0
}

define i32 @universe_dataframe_rename(ptr %df, ptr %old, i64 %ol, ptr %new, i64 %nl) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %on = icmp eq ptr %old, null
  %nn = icmp eq ptr %new, null
  %e1 = or i1 %dn, %on
  %e2 = or i1 %e1, %nn
  br i1 %e2, label %err.null, label %find, !prof !0

err.null:
  ret i32 1

find:
  %idx = call i64 @df_find(ptr %df, ptr %old, i64 %ol)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %notfound, label %do, !prof !0

notfound:
  ret i32 5

do:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %noff = mul i64 %idx, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %oldname = load ptr, ptr %nslot, align 8
  call void @free(ptr %oldname)
  %namecopy = call ptr @df_name_dup(ptr %new, i64 %nl)
  store ptr %namecopy, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  store i64 %nl, ptr %nlenslot, align 8
  ret i32 0
}

; remove column, shifting the tail down. returns the removed series via out.
define internal void @df_remove_at(ptr %df, i64 %idx, ptr %out_series) #1 {
entry:
  %ncp = getelementptr inbounds i8, ptr %df, i64 0
  %nc = load i64, ptr %ncp, align 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %noff = mul i64 %idx, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nameptr = load ptr, ptr %nslot, align 8
  call void @free(ptr %nameptr)
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %idx
  %series = load ptr, ptr %cslot, align 8
  %hasout = icmp ne ptr %out_series, null
  br i1 %hasout, label %storeout, label %shift

storeout:
  store ptr %series, ptr %out_series, align 8
  br label %shift

shift:
  %idx1 = add i64 %idx, 1
  %tail = sub i64 %nc, %idx1
  %ndst = getelementptr inbounds i8, ptr %names, i64 %noff
  %nsrc = getelementptr inbounds i8, ptr %ndst, i64 16
  %nbytes = mul i64 %tail, 16
  call void @llvm.memmove.p0.p0.i64(ptr %ndst, ptr %nsrc, i64 %nbytes, i1 false)
  %cdstoff = mul i64 %idx, 8
  %cdst = getelementptr inbounds i8, ptr %cols, i64 %cdstoff
  %csrc = getelementptr inbounds i8, ptr %cdst, i64 8
  %cbytes = mul i64 %tail, 8
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %cbytes, i1 false)
  %ncn = sub i64 %nc, 1
  store i64 %ncn, ptr %ncp, align 8
  ret void
}

define i32 @universe_dataframe_drop_in_place(ptr %df, ptr %name, i64 %nl, ptr %out_series) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %e = or i1 %dn, %nn
  br i1 %e, label %err.null, label %find, !prof !0

err.null:
  ret i32 1

find:
  %idx = call i64 @df_find(ptr %df, ptr %name, i64 %nl)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %notfound, label %do, !prof !0

notfound:
  ret i32 5

do:
  %want = icmp ne ptr %out_series, null
  br i1 %want, label %keep, label %drop

keep:
  call void @df_remove_at(ptr %df, i64 %idx, ptr %out_series)
  ret i32 0

drop:
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %idx
  %series = load ptr, ptr %cslot, align 8
  call void @df_remove_at(ptr %df, i64 %idx, ptr null)
  call void @universe_dataframe_series_free(ptr %series)
  ret i32 0
}

define ptr @universe_dataframe_drop(ptr %df, ptr %name, i64 %name_len) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %find

find:
  %idx = call i64 @df_find(ptr %df, ptr %name, i64 %name_len)
  %nf = icmp slt i64 %idx, 0
  br i1 %nf, label %fail, label %build

build:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %nc = load i64, ptr %df, align 8
  %out = call ptr @df_new_empty(i64 %height, i64 %nc)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %setup

setup:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %skip = icmp eq i64 %k, %idx
  br i1 %skip, label %loop.cont, label %copy

copy:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @universe_dataframe_series_clone(ptr %col)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @df_append_col(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret ptr %out

fail:
  ret ptr null
}

; check whether name (ptr,len) is in the drop-list names[0..n)
define internal i1 @df_in_list(ptr %names, i64 %n, ptr %name, i64 %nl) #3 {
entry:
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %loop.body, label %notfound

loop.body:
  %soff = mul i64 %k, 16
  %sslot = getelementptr inbounds i8, ptr %names, i64 %soff
  %sptr = load ptr, ptr %sslot, align 8
  %slenslot = getelementptr inbounds i8, ptr %sslot, i64 8
  %slen = load i64, ptr %slenslot, align 8
  %lenmatch = icmp eq i64 %slen, %nl
  br i1 %lenmatch, label %cmpbytes, label %loop.cont

cmpbytes:
  %r = call i32 @memcmp(ptr %sptr, ptr %name, i64 %nl)
  %eq = icmp eq i32 %r, 0
  br i1 %eq, label %found, label %loop.cont

found:
  ret i1 true

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

notfound:
  ret i1 false
}

define ptr @universe_dataframe_drop_many(ptr %df, ptr %names, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %build

build:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %nc = load i64, ptr %df, align 8
  %out = call ptr @df_new_empty(i64 %height, i64 %nc)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %setup

setup:
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %mynames = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %mynames, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %indrop = call i1 @df_in_list(ptr %names, i64 %n, ptr %nptr, i64 %nlen)
  br i1 %indrop, label %loop.cont, label %copy

copy:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @universe_dataframe_series_clone(ptr %col)
  %rc = call i32 @df_append_col(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_hstack(ptr %df, ptr %series_arr, ptr %names, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %sn = icmp eq ptr %series_arr, null
  %nn = icmp eq ptr %names, null
  %e1 = or i1 %dn, %sn
  %e2 = or i1 %e1, %nn
  br i1 %e2, label %err.null, label %loop.head, !prof !0

err.null:
  ret i32 1

loop.head:
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %sslot = getelementptr inbounds ptr, ptr %series_arr, i64 %k
  %series = load ptr, ptr %sslot, align 8
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @df_append_col(ptr %df, ptr %nptr, i64 %nlen, ptr %series)
  %bad = icmp ne i32 %rc, 0
  br i1 %bad, label %ret.err, label %loop.cont

ret.err:
  ret i32 %rc

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret i32 0
}

; schema-compatible check: same width, same names, same dtypes
define internal i1 @df_schema_match(ptr %a, ptr %b) #3 {
entry:
  %anc = load i64, ptr %a, align 8
  %bnc = load i64, ptr %b, align 8
  %wne = icmp ne i64 %anc, %bnc
  br i1 %wne, label %ret.false, label %setup

setup:
  %anp = getelementptr inbounds i8, ptr %a, i64 24
  %anames = load ptr, ptr %anp, align 8
  %bnp = getelementptr inbounds i8, ptr %b, i64 24
  %bnames = load ptr, ptr %bnp, align 8
  %acp = getelementptr inbounds i8, ptr %a, i64 32
  %acols = load ptr, ptr %acp, align 8
  %bcp = getelementptr inbounds i8, ptr %b, i64 32
  %bcols = load ptr, ptr %bcp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %anc
  br i1 %cmp, label %loop.body, label %ret.true

loop.body:
  %noff = mul i64 %k, 16
  %anslot = getelementptr inbounds i8, ptr %anames, i64 %noff
  %anptr = load ptr, ptr %anslot, align 8
  %anlenslot = getelementptr inbounds i8, ptr %anslot, i64 8
  %anlen = load i64, ptr %anlenslot, align 8
  %bnslot = getelementptr inbounds i8, ptr %bnames, i64 %noff
  %bnptr = load ptr, ptr %bnslot, align 8
  %bnlenslot = getelementptr inbounds i8, ptr %bnslot, i64 8
  %bnlen = load i64, ptr %bnlenslot, align 8
  %lenne = icmp ne i64 %anlen, %bnlen
  br i1 %lenne, label %ret.false, label %cmpname

cmpname:
  %r = call i32 @memcmp(ptr %anptr, ptr %bnptr, i64 %anlen)
  %namene = icmp ne i32 %r, 0
  br i1 %namene, label %ret.false, label %cmpdtype

cmpdtype:
  %acslot = getelementptr inbounds ptr, ptr %acols, i64 %k
  %acol = load ptr, ptr %acslot, align 8
  %adt = load i32, ptr %acol, align 8
  %bcslot = getelementptr inbounds ptr, ptr %bcols, i64 %k
  %bcol = load ptr, ptr %bcslot, align 8
  %bdt = load i32, ptr %bcol, align 8
  %dtne = icmp ne i32 %adt, %bdt
  br i1 %dtne, label %ret.false, label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

ret.true:
  ret i1 true

ret.false:
  ret i1 false
}

define ptr @universe_dataframe_vstack(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %e = or i1 %an, %bn
  br i1 %e, label %fail, label %chk

chk:
  %match = call i1 @df_schema_match(ptr %a, ptr %b)
  br i1 %match, label %build, label %fail

build:
  %ahp = getelementptr inbounds i8, ptr %a, i64 16
  %ah = load i64, ptr %ahp, align 8
  %bhp = getelementptr inbounds i8, ptr %b, i64 16
  %bh = load i64, ptr %bhp, align 8
  %nh = add i64 %ah, %bh
  %nc = load i64, ptr %a, align 8
  %out = call ptr @df_new_empty(i64 %nh, i64 %nc)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %setup

setup:
  %anp = getelementptr inbounds i8, ptr %a, i64 24
  %anames = load ptr, ptr %anp, align 8
  %acp = getelementptr inbounds i8, ptr %a, i64 32
  %acols = load ptr, ptr %acp, align 8
  %bcp = getelementptr inbounds i8, ptr %b, i64 32
  %bcols = load ptr, ptr %bcp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %acslot = getelementptr inbounds ptr, ptr %acols, i64 %k
  %acol = load ptr, ptr %acslot, align 8
  %bcslot = getelementptr inbounds ptr, ptr %bcols, i64 %k
  %bcol = load ptr, ptr %bcslot, align 8
  %newcol = call ptr @series_concat(ptr %acol, ptr %bcol)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %anames, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @df_append_col(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_extend(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %e = or i1 %an, %bn
  br i1 %e, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %match = call i1 @df_schema_match(ptr %a, ptr %b)
  br i1 %match, label %do, label %err.arg

err.arg:
  ret i32 8

do:
  %nc = load i64, ptr %a, align 8
  %acp = getelementptr inbounds i8, ptr %a, i64 32
  %acols = load ptr, ptr %acp, align 8
  %bcp = getelementptr inbounds i8, ptr %b, i64 32
  %bcols = load ptr, ptr %bcp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %setheight

loop.body:
  %acslot = getelementptr inbounds ptr, ptr %acols, i64 %k
  %acol = load ptr, ptr %acslot, align 8
  %bcslot = getelementptr inbounds ptr, ptr %bcols, i64 %k
  %bcol = load ptr, ptr %bcslot, align 8
  %newcol = call ptr @series_concat(ptr %acol, ptr %bcol)
  call void @universe_dataframe_series_free(ptr %acol)
  store ptr %newcol, ptr %acslot, align 8
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

setheight:
  %ahp = getelementptr inbounds i8, ptr %a, i64 16
  %ah = load i64, ptr %ahp, align 8
  %bhp = getelementptr inbounds i8, ptr %b, i64 16
  %bh = load i64, ptr %bhp, align 8
  %nh = add i64 %ah, %bh
  store i64 %nh, ptr %ahp, align 8
  ret i32 0
}

define ptr @universe_dataframe_slice(ptr %df, i64 %offset, i64 %len) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %offbad = icmp slt i64 %offset, 0
  %lenbad = icmp slt i64 %len, 0
  %e1 = or i1 %dn, %offbad
  %e2 = or i1 %e1, %lenbad
  br i1 %e2, label %fail, label %clamp

clamp:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %effoff = call i64 @llvm.umin.i64(i64 %offset, i64 %height)
  %avail = sub i64 %height, %effoff
  %efflen = call i64 @llvm.umin.i64(i64 %len, i64 %avail)
  %idx = call ptr @df_idx_range(i64 %effoff, i64 %efflen)
  %out = call ptr @df_gather(ptr %df, ptr %idx, i64 %efflen)
  call void @free(ptr %idx)
  ret ptr %out

fail:
  ret ptr null
}

define ptr @universe_dataframe_head(ptr %df, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %compute

compute:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %neg = icmp slt i64 %n, 0
  br i1 %neg, label %negcase, label %poscase

poscase:
  %cntp = call i64 @llvm.umin.i64(i64 %n, i64 %height)
  br label %do

negcase:
  %hplusn = add i64 %height, %n
  %cntn = call i64 @llvm.smax.i64(i64 %hplusn, i64 0)
  br label %do

do:
  %cnt = phi i64 [ %cntp, %poscase ], [ %cntn, %negcase ]
  %out = call ptr @universe_dataframe_slice(ptr %df, i64 0, i64 %cnt)
  ret ptr %out

fail:
  ret ptr null
}

define ptr @universe_dataframe_tail(ptr %df, i64 %n) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %compute

compute:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %neg = icmp slt i64 %n, 0
  br i1 %neg, label %negcase, label %poscase

poscase:
  %cntp = call i64 @llvm.umin.i64(i64 %n, i64 %height)
  %startp = sub i64 %height, %cntp
  br label %do

negcase:
  %negn = sub i64 0, %n
  %startn = call i64 @llvm.umin.i64(i64 %negn, i64 %height)
  %cntn = sub i64 %height, %startn
  br label %do

do:
  %start = phi i64 [ %startp, %poscase ], [ %startn, %negcase ]
  %cnt = phi i64 [ %cntp, %poscase ], [ %cntn, %negcase ]
  %out = call ptr @universe_dataframe_slice(ptr %df, i64 %start, i64 %cnt)
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_split_at(ptr %df, i64 %at, ptr %out_a, ptr %out_b) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %an = icmp eq ptr %out_a, null
  %bn = icmp eq ptr %out_b, null
  %e1 = or i1 %dn, %an
  %e2 = or i1 %e1, %bn
  br i1 %e2, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %neg = icmp slt i64 %at, 0
  br i1 %neg, label %err.arg, label %do, !prof !0

err.arg:
  ret i32 8

do:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %eat = call i64 @llvm.umin.i64(i64 %at, i64 %height)
  %rest = sub i64 %height, %eat
  %a = call ptr @universe_dataframe_slice(ptr %df, i64 0, i64 %eat)
  %b = call ptr @universe_dataframe_slice(ptr %df, i64 %eat, i64 %rest)
  store ptr %a, ptr %out_a, align 8
  store ptr %b, ptr %out_b, align 8
  ret i32 0
}

define ptr @universe_dataframe_reverse(ptr %df) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %do

do:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %bytes = mul i64 %height, 8
  %idx = call ptr @df_xmalloc(i64 %bytes)
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.body ]
  %cmp = icmp ult i64 %k, %height
  br i1 %cmp, label %loop.body, label %gather

loop.body:
  %hm1 = sub i64 %height, 1
  %v = sub i64 %hm1, %k
  %p = getelementptr inbounds i64, ptr %idx, i64 %k
  store i64 %v, ptr %p, align 8
  %k.next = add i64 %k, 1
  br label %loop.head

gather:
  %out = call ptr @df_gather(ptr %df, ptr %idx, i64 %height)
  call void @free(ptr %idx)
  ret ptr %out

fail:
  ret ptr null
}

define ptr @universe_dataframe_shift(ptr %df, i64 %periods) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %do

do:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %bytes = mul i64 %height, 8
  %idx = call ptr @df_xmalloc(i64 %bytes)
  %isneg = icmp slt i64 %periods, 0
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %storeidx ]
  %cmp = icmp ult i64 %k, %height
  br i1 %cmp, label %loop.body, label %gather

loop.body:
  %p = getelementptr inbounds i64, ptr %idx, i64 %k
  br i1 %isneg, label %negshift, label %posshift

posshift:
  %ge = icmp sge i64 %k, %periods
  %srcpos = sub i64 %k, %periods
  %valpos = select i1 %ge, i64 %srcpos, i64 -1
  br label %storeidx

negshift:
  %m = sub i64 0, %periods
  %srcneg = add i64 %k, %m
  %inrange = icmp ult i64 %srcneg, %height
  %valneg = select i1 %inrange, i64 %srcneg, i64 -1
  br label %storeidx

storeidx:
  %v = phi i64 [ %valpos, %posshift ], [ %valneg, %negshift ]
  store i64 %v, ptr %p, align 8
  %k.next = add i64 %k, 1
  br label %loop.head

gather:
  %out = call ptr @df_gather(ptr %df, ptr %idx, i64 %height)
  call void @free(ptr %idx)
  ret ptr %out

fail:
  ret ptr null
}

define i32 @universe_dataframe_null_count(ptr %df, ptr %out_counts, i64 %cap) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  %on = icmp eq ptr %out_counts, null
  %e = or i1 %dn, %on
  br i1 %e, label %err.null, label %do, !prof !0

err.null:
  ret i32 1

do:
  %nc = load i64, ptr %df, align 8
  %lim = call i64 @llvm.umin.i64(i64 %nc, i64 %cap)
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %lim
  br i1 %cmp, label %loop.cont, label %done

loop.cont:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %ncp = getelementptr inbounds i8, ptr %col, i64 16
  %cnt = load i64, ptr %ncp, align 8
  %dslot = getelementptr inbounds i64, ptr %out_counts, i64 %k
  store i64 %cnt, ptr %dslot, align 8
  %k.next = add i64 %k, 1
  br label %loop.head

done:
  ret i32 0
}

define i32 @universe_dataframe_equals(ptr %a, ptr %b, ptr %out_bool) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %obn = icmp eq ptr %out_bool, null
  %e1 = or i1 %an, %bn
  %e2 = or i1 %e1, %obn
  br i1 %e2, label %err.null, label %shapechk, !prof !0

err.null:
  ret i32 1

shapechk:
  %anc = load i64, ptr %a, align 8
  %bnc = load i64, ptr %b, align 8
  %wne = icmp ne i64 %anc, %bnc
  %ahp = getelementptr inbounds i8, ptr %a, i64 16
  %ah = load i64, ptr %ahp, align 8
  %bhp = getelementptr inbounds i8, ptr %b, i64 16
  %bh = load i64, ptr %bhp, align 8
  %hne = icmp ne i64 %ah, %bh
  %shapene = or i1 %wne, %hne
  br i1 %shapene, label %notequal, label %setup

setup:
  %anp = getelementptr inbounds i8, ptr %a, i64 24
  %anames = load ptr, ptr %anp, align 8
  %bnp = getelementptr inbounds i8, ptr %b, i64 24
  %bnames = load ptr, ptr %bnp, align 8
  %acp = getelementptr inbounds i8, ptr %a, i64 32
  %acols = load ptr, ptr %acp, align 8
  %bcp = getelementptr inbounds i8, ptr %b, i64 32
  %bcols = load ptr, ptr %bcp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %setup ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %anc
  br i1 %cmp, label %loop.body, label %equal

loop.body:
  %noff = mul i64 %k, 16
  %anslot = getelementptr inbounds i8, ptr %anames, i64 %noff
  %anptr = load ptr, ptr %anslot, align 8
  %anlenslot = getelementptr inbounds i8, ptr %anslot, i64 8
  %anlen = load i64, ptr %anlenslot, align 8
  %bnslot = getelementptr inbounds i8, ptr %bnames, i64 %noff
  %bnptr = load ptr, ptr %bnslot, align 8
  %bnlenslot = getelementptr inbounds i8, ptr %bnslot, i64 8
  %bnlen = load i64, ptr %bnlenslot, align 8
  %lenne = icmp ne i64 %anlen, %bnlen
  br i1 %lenne, label %notequal, label %cmpname

cmpname:
  %r = call i32 @memcmp(ptr %anptr, ptr %bnptr, i64 %anlen)
  %namene = icmp ne i32 %r, 0
  br i1 %namene, label %notequal, label %cmpvals

cmpvals:
  %acslot = getelementptr inbounds ptr, ptr %acols, i64 %k
  %acol = load ptr, ptr %acslot, align 8
  %bcslot = getelementptr inbounds ptr, ptr %bcols, i64 %k
  %bcol = load ptr, ptr %bcslot, align 8
  %eq = call i1 @series_equals(ptr %acol, ptr %bcol)
  br i1 %eq, label %loop.cont, label %notequal

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

equal:
  store i8 1, ptr %out_bool, align 1
  ret i32 0

notequal:
  store i8 0, ptr %out_bool, align 1
  ret i32 0
}

define ptr @universe_dataframe_with_row_index(ptr %df, ptr %name, i64 %nl, i64 %offset) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  %nn = icmp eq ptr %name, null
  %e = or i1 %dn, %nn
  br i1 %e, label %fail, label %build

build:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  %height = load i64, ptr %hp, align 8
  %nc = load i64, ptr %df, align 8
  %capneed = add i64 %nc, 1
  %out = call ptr @df_new_empty(i64 %height, i64 %capneed)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %mkidx

mkidx:
  %idxcol = call ptr @universe_dataframe_series_new(i32 1, i64 %height)
  %icn = icmp eq ptr %idxcol, null
  br i1 %icn, label %freefail, label %fillidx

fillidx:
  %vpp = getelementptr inbounds i8, ptr %idxcol, i64 24
  %vals = load ptr, ptr %vpp, align 8
  br label %fill.head

fill.head:
  %fk = phi i64 [ 0, %fillidx ], [ %fk.next, %fill.body ]
  %fcmp = icmp ult i64 %fk, %height
  br i1 %fcmp, label %fill.body, label %appendidx

fill.body:
  %fv = add i64 %offset, %fk
  %fp = getelementptr inbounds i64, ptr %vals, i64 %fk
  store i64 %fv, ptr %fp, align 8
  %fk.next = add i64 %fk, 1
  br label %fill.head

appendidx:
  %rc0 = call i32 @df_append_col(ptr %out, ptr %name, i64 %nl, ptr %idxcol)
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %appendidx ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %done

loop.body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %newcol = call ptr @universe_dataframe_series_clone(ptr %col)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  %nlenslot = getelementptr inbounds i8, ptr %nslot, i64 8
  %nlen = load i64, ptr %nlenslot, align 8
  %rc = call i32 @df_append_col(ptr %out, ptr %nptr, i64 %nlen, ptr %newcol)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

freefail:
  call void @universe_dataframe_free(ptr %out)
  br label %fail

done:
  ret ptr %out

fail:
  ret ptr null
}

define void @universe_dataframe_clear(ptr %df) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %done, label %do

do:
  %nc = load i64, ptr %df, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %seth

loop.body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  %dt = load i32, ptr %col, align 8
  %newcol = call ptr @universe_dataframe_series_new(i32 %dt, i64 0)
  call void @universe_dataframe_series_free(ptr %col)
  store ptr %newcol, ptr %cslot, align 8
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

seth:
  %hp = getelementptr inbounds i8, ptr %df, i64 16
  store i64 0, ptr %hp, align 8
  br label %done

done:
  ret void
}

define void @universe_dataframe_free(ptr %df) local_unnamed_addr #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %done, label %do

do:
  %nc = load i64, ptr %df, align 8
  %np = getelementptr inbounds i8, ptr %df, i64 24
  %names = load ptr, ptr %np, align 8
  %cp = getelementptr inbounds i8, ptr %df, i64 32
  %cols = load ptr, ptr %cp, align 8
  br label %loop.head

loop.head:
  %k = phi i64 [ 0, %do ], [ %k.next, %loop.cont ]
  %cmp = icmp ult i64 %k, %nc
  br i1 %cmp, label %loop.body, label %freebufs

loop.body:
  %cslot = getelementptr inbounds ptr, ptr %cols, i64 %k
  %col = load ptr, ptr %cslot, align 8
  call void @universe_dataframe_series_free(ptr %col)
  %noff = mul i64 %k, 16
  %nslot = getelementptr inbounds i8, ptr %names, i64 %noff
  %nptr = load ptr, ptr %nslot, align 8
  call void @free(ptr %nptr)
  br label %loop.cont

loop.cont:
  %k.next = add i64 %k, 1
  br label %loop.head

freebufs:
  call void @free(ptr %names)
  call void @free(ptr %cols)
  call void @free(ptr %df)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #5 = { alwaysinline nounwind willreturn norecurse nosync memory(read) }
attributes #6 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
