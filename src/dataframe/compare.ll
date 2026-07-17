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

; DataFrame SIMD compute layer: predicates + boolean algebra. This is the piece
; that makes `filter` (select.ll) SIMD-predicated — the vector `icmp`/`fcmp`
; kernels build the BOOL mask that filter compacts. Functional inspiration from
; numpy ufuncs / Polars kernels / Julia SIMD.jl (SEMANTICS only); layout+IR ours.
;
; ============================ SCOPE (state plainly) =========================
; This module gives the DataFrame its VECTORIZED predicate + boolean-algebra
; surface. The compaction inside filter/take stays scalar (portable baseline has
; no SIMD compress; vpcompress/SVE is a future dispatched add-on) — but the
; PREDICATE that drives filter is now SIMD. Together with arith.ll/cast.ll/
; reduce.ll this is the numpy/Polars/Julia element-wise compute layer.
;
; DESIGN — SIMD-first with scalar fallback/oracle (per CLAUDE.md MANDATORY):
;   * PRIMARY PATH is a portable 128-bit vector loop per dtype:
;       I32 -> <4 x i32>, I64 -> <2 x i64>, F32 -> <4 x float>, F64 -> <2 x
;       double>, BOOL -> <16 x i8>. These lower to SSE2 (x86) and NEON (ARM),
;       both baseline everywhere we ship — NO runtime CPU probe. Compares:
;       vector icmp/fcmp -> <N x i1> -> `zext` to <N x i8> 0/1 -> store into the
;       BOOL result values buffer. (The BOOL contract is one 0/1 byte per value;
;       `zext` yields exactly 0/1 so bitwise and/or/xor over the byte column is
;       self-consistent and `not` is xor-with-1.)
;   * SCALAR TAIL handles len % lanes and is the fallback/oracle: the tests
;       cross-check the vector result against an independent scalar oracle over
;       fixed-seed inputs (both multiple-of-lane and odd lengths).
;   * FLOAT COMPARES ARE ORDERED (oeq/one/olt/ole/ogt/oge): any NaN operand ->
;       false for EVERY predicate (documented). Ordered predicates are used
;       directly (NOT derived as gt = !(lt|eq), which is wrong for NaN).
;   * INT COMPARES ARE SIGNED (I32/I64 are signed dtypes): eq/ne/slt/sle/sgt/sge.
;
; NULL STRATEGY (Polars/arrow2 trick): compute the comparison DENSELY over EVERY
;   lane (including null lanes — the garbage 0/1 there is harmless because the
;   result validity marks it null). The result BOOL validity is combined
;   SEPARATELY: out.validity = a.validity AND b.validity (a null input lane is
;   null in the result, NOT true). Scalar broadcast keeps a's validity. This
;   keeps the value loop branch-free and fully vectorized — no per-lane null
;   test inside the SIMD body. A null validity ptr means "all valid".
;
; API (all col-col/col-scalar comparators return a NEW BOOL Series, null on err):
;   ptr universe_dataframe_eq  (a,b)  neq lt lte gt gte  ; Series (x) Series
;   ptr universe_dataframe_eq_scalar (a, i32 stype, i64 ival, double fval) ; +5
;   ptr universe_dataframe_and (a,b)  or xor             ; BOOL (x) BOOL
;   ptr universe_dataframe_not (a)                       ; BOOL
;   i32 universe_dataframe_any    (bool_series, ptr out_bool)  ; reduce.or
;   i32 universe_dataframe_all    (bool_series, ptr out_bool)  ; reduce.and
;   i32 universe_dataframe_sum_bool(bool_series, ptr out_i64)  ; popcount of true
;   ptr universe_dataframe_zip_with(mask_bool, a, b)     ; branchless per-lane sel
;
; DType (i32): I32=0 I64=1 F32=2 F64=3 BOOL=4 STR=5. Comparators require a,b same
; numeric dtype (0..3) — else null. Boolean ops require BOOL (4). zip_with allows
; any fixed dtype (0..4), same for a,b; STR (5) -> null.
;
; Errors i32 (for the i32-returning reductions): 0 OK, 1 NULL_PTR, 8 INVALID_ARG.
; ptr-returning ops signal failure with null. No atomics (single-threaded).
; ===========================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare i8 @llvm.vector.reduce.or.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.and.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)

@cmp.widths = internal constant [6 x i8] c"\04\08\04\08\01\04"

; op codes for the compare engines (shared by col-col + col-scalar):
;   0 eq   1 neq   2 lt   3 lte   4 gt   5 gte

; ---------------------------------------------------------------------------
; small shared helpers (mirror frame.ll shapes)
; ---------------------------------------------------------------------------

define internal i64 @cmp_width(i32 %dtype) #5 {
entry:
  %i = zext i32 %dtype to i64
  %p = getelementptr inbounds [6 x i8], ptr @cmp.widths, i64 0, i64 %i
  %w8 = load i8, ptr %p, align 1
  %w = zext i8 %w8 to i64
  ret i64 %w
}

define internal ptr @cmp_xmalloc(i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  %sz = select i1 %z, i64 1, i64 %n
  %p = call ptr @malloc(i64 %sz)
  ret ptr %p
}

define internal i1 @cmp_valid_at(ptr %s, i64 %i) #6 {
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

define internal void @cmp_bm_clear(ptr %bm, i64 %k) #4 {
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

; allocate a fresh BOOL series (dtype 4) of length n, values buffer = n bytes
; (uninitialized — the caller writes every lane). validity/strdata cleared.
define internal ptr @cmp_mk_bool(i64 %n) #1 {
entry:
  %hdr = call ptr @malloc(i64 56)
  %hn = icmp eq ptr %hdr, null
  br i1 %hn, label %fail, label %init, !prof !0
init:
  call void @llvm.memset.p0.i64(ptr %hdr, i8 0, i64 56, i1 false)
  store i32 4, ptr %hdr, align 8
  %lp = getelementptr inbounds i8, ptr %hdr, i64 8
  store i64 %n, ptr %lp, align 8
  %vals = call ptr @cmp_xmalloc(i64 %n)
  %vn = icmp eq ptr %vals, null
  br i1 %vn, label %freehdr, label %setv, !prof !0
setv:
  %vpp = getelementptr inbounds i8, ptr %hdr, i64 24
  store ptr %vals, ptr %vpp, align 8
  ret ptr %hdr
freehdr:
  call void @free(ptr %hdr)
  br label %fail
fail:
  ret ptr null
}

; allocate a fresh series of arbitrary fixed dtype + length (values buffer only).
define internal ptr @cmp_mk_fixed(i32 %dtype, i64 %n) #1 {
entry:
  %hdr = call ptr @malloc(i64 56)
  %hn = icmp eq ptr %hdr, null
  br i1 %hn, label %fail, label %init, !prof !0
init:
  call void @llvm.memset.p0.i64(ptr %hdr, i8 0, i64 56, i1 false)
  store i32 %dtype, ptr %hdr, align 8
  %lp = getelementptr inbounds i8, ptr %hdr, i64 8
  store i64 %n, ptr %lp, align 8
  %w = call i64 @cmp_width(i32 %dtype)
  %bytes = mul i64 %n, %w
  %vals = call ptr @cmp_xmalloc(i64 %bytes)
  %vn = icmp eq ptr %vals, null
  br i1 %vn, label %freehdr, label %setv, !prof !0
setv:
  %vpp = getelementptr inbounds i8, ptr %hdr, i64 24
  store ptr %vals, ptr %vpp, align 8
  ret ptr %hdr
freehdr:
  call void @free(ptr %hdr)
  br label %fail
fail:
  ret ptr null
}

; out.validity = a.validity AND b.validity ; sets out.null_count. Cold: only
; touches memory when at least one input has nulls. Row loop (once per op).
define internal void @cmp_validity_and(ptr %out, ptr %a, ptr %b, i64 %n) #1 {
entry:
  %avp = getelementptr inbounds i8, ptr %a, i64 32
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 32
  %bv = load ptr, ptr %bvp, align 8
  %an = icmp eq ptr %av, null
  %bn = icmp eq ptr %bv, null
  %both = and i1 %an, %bn
  br i1 %both, label %done, label %build
build:
  %t = add i64 %n, 7
  %nb = lshr i64 %t, 3
  %bm = call ptr @cmp_xmalloc(i64 %nb)
  call void @llvm.memset.p0.i64(ptr %bm, i8 -1, i64 %nb, i1 false)
  %ovp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %ovp, align 8
  br label %loop
loop:
  %k = phi i64 [ 0, %build ], [ %kn, %cont ]
  %nulls = phi i64 [ 0, %build ], [ %nn, %cont ]
  %cmp = icmp ult i64 %k, %n
  br i1 %cmp, label %body, label %fin
body:
  %va = call i1 @cmp_valid_at(ptr %a, i64 %k)
  %vb = call i1 @cmp_valid_at(ptr %b, i64 %k)
  %valid = and i1 %va, %vb
  br i1 %valid, label %cont, label %isnull
isnull:
  call void @cmp_bm_clear(ptr %bm, i64 %k)
  br label %cont
cont:
  %inc = zext i1 %valid to i64
  %nnv = xor i64 %inc, 1
  %nn = add i64 %nulls, %nnv
  %kn = add i64 %k, 1
  br label %loop
fin:
  %ncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nulls, ptr %ncp, align 8
  ret void
done:
  ret void
}

; out.validity = copy of a.validity (scalar-broadcast comparators keep a's nulls)
define internal void @cmp_validity_copy(ptr %out, ptr %a, i64 %n) #1 {
entry:
  %avp = getelementptr inbounds i8, ptr %a, i64 32
  %av = load ptr, ptr %avp, align 8
  %an = icmp eq ptr %av, null
  br i1 %an, label %done, label %build
build:
  %t = add i64 %n, 7
  %nb = lshr i64 %t, 3
  %bm = call ptr @cmp_xmalloc(i64 %nb)
  call void @llvm.memcpy.p0.p0.i64(ptr %bm, ptr %av, i64 %nb, i1 false)
  %ovp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %ovp, align 8
  %ancp = getelementptr inbounds i8, ptr %a, i64 16
  %anc = load i64, ptr %ancp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %anc, ptr %oncp, align 8
  ret void
done:
  ret void
}

; ---------------------------------------------------------------------------
; compare engines (col-col). Vector 128-bit primary + scalar tail. The op-switch
; is loop-invariant: at -O3 it unswitches/folds to a single packed compare.
; ---------------------------------------------------------------------------

define internal void @cmp_eng_i32(ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 4
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %va = load <4 x i32>, ptr %pa, align 4
  %pb = getelementptr inbounds i32, ptr %b, i64 %i
  %vb = load <4 x i32>, ptr %pb, align 4
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = icmp eq <4 x i32> %va, %vb
  br label %vsel
v_ne:
  %m_ne = icmp ne <4 x i32> %va, %vb
  br label %vsel
v_lt:
  %m_lt = icmp slt <4 x i32> %va, %vb
  br label %vsel
v_le:
  %m_le = icmp sle <4 x i32> %va, %vb
  br label %vsel
v_gt:
  %m_gt = icmp sgt <4 x i32> %va, %vb
  br label %vsel
v_ge:
  %m_ge = icmp sge <4 x i32> %va, %vb
  br label %vsel
vsel:
  %m = phi <4 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <4 x i1> %m to <4 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <4 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 4
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds i32, ptr %a, i64 %j
  %sa = load i32, ptr %ta, align 4
  %tb = getelementptr inbounds i32, ptr %b, i64 %j
  %sb = load i32, ptr %tb, align 4
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = icmp eq i32 %sa, %sb
  br label %ssel
s_ne:
  %r_ne = icmp ne i32 %sa, %sb
  br label %ssel
s_lt:
  %r_lt = icmp slt i32 %sa, %sb
  br label %ssel
s_le:
  %r_le = icmp sle i32 %sa, %sb
  br label %ssel
s_gt:
  %r_gt = icmp sgt i32 %sa, %sb
  br label %ssel
s_ge:
  %r_ge = icmp sge i32 %sa, %sb
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_i64(ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 2
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i64, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %pb, align 8
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = icmp eq <2 x i64> %va, %vb
  br label %vsel
v_ne:
  %m_ne = icmp ne <2 x i64> %va, %vb
  br label %vsel
v_lt:
  %m_lt = icmp slt <2 x i64> %va, %vb
  br label %vsel
v_le:
  %m_le = icmp sle <2 x i64> %va, %vb
  br label %vsel
v_gt:
  %m_gt = icmp sgt <2 x i64> %va, %vb
  br label %vsel
v_ge:
  %m_ge = icmp sge <2 x i64> %va, %vb
  br label %vsel
vsel:
  %m = phi <2 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <2 x i1> %m to <2 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <2 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 2
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds i64, ptr %a, i64 %j
  %sa = load i64, ptr %ta, align 8
  %tb = getelementptr inbounds i64, ptr %b, i64 %j
  %sb = load i64, ptr %tb, align 8
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = icmp eq i64 %sa, %sb
  br label %ssel
s_ne:
  %r_ne = icmp ne i64 %sa, %sb
  br label %ssel
s_lt:
  %r_lt = icmp slt i64 %sa, %sb
  br label %ssel
s_le:
  %r_le = icmp sle i64 %sa, %sb
  br label %ssel
s_gt:
  %r_gt = icmp sgt i64 %sa, %sb
  br label %ssel
s_ge:
  %r_ge = icmp sge i64 %sa, %sb
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_f32(ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 4
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds float, ptr %a, i64 %i
  %va = load <4 x float>, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %b, i64 %i
  %vb = load <4 x float>, ptr %pb, align 4
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = fcmp oeq <4 x float> %va, %vb
  br label %vsel
v_ne:
  %m_ne = fcmp one <4 x float> %va, %vb
  br label %vsel
v_lt:
  %m_lt = fcmp olt <4 x float> %va, %vb
  br label %vsel
v_le:
  %m_le = fcmp ole <4 x float> %va, %vb
  br label %vsel
v_gt:
  %m_gt = fcmp ogt <4 x float> %va, %vb
  br label %vsel
v_ge:
  %m_ge = fcmp oge <4 x float> %va, %vb
  br label %vsel
vsel:
  %m = phi <4 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <4 x i1> %m to <4 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <4 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 4
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds float, ptr %a, i64 %j
  %sa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds float, ptr %b, i64 %j
  %sb = load float, ptr %tb, align 4
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = fcmp oeq float %sa, %sb
  br label %ssel
s_ne:
  %r_ne = fcmp one float %sa, %sb
  br label %ssel
s_lt:
  %r_lt = fcmp olt float %sa, %sb
  br label %ssel
s_le:
  %r_le = fcmp ole float %sa, %sb
  br label %ssel
s_gt:
  %r_gt = fcmp ogt float %sa, %sb
  br label %ssel
s_ge:
  %r_ge = fcmp oge float %sa, %sb
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_f64(ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 2
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds double, ptr %a, i64 %i
  %va = load <2 x double>, ptr %pa, align 8
  %pb = getelementptr inbounds double, ptr %b, i64 %i
  %vb = load <2 x double>, ptr %pb, align 8
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = fcmp oeq <2 x double> %va, %vb
  br label %vsel
v_ne:
  %m_ne = fcmp one <2 x double> %va, %vb
  br label %vsel
v_lt:
  %m_lt = fcmp olt <2 x double> %va, %vb
  br label %vsel
v_le:
  %m_le = fcmp ole <2 x double> %va, %vb
  br label %vsel
v_gt:
  %m_gt = fcmp ogt <2 x double> %va, %vb
  br label %vsel
v_ge:
  %m_ge = fcmp oge <2 x double> %va, %vb
  br label %vsel
vsel:
  %m = phi <2 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <2 x i1> %m to <2 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <2 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 2
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds double, ptr %a, i64 %j
  %sa = load double, ptr %ta, align 8
  %tb = getelementptr inbounds double, ptr %b, i64 %j
  %sb = load double, ptr %tb, align 8
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = fcmp oeq double %sa, %sb
  br label %ssel
s_ne:
  %r_ne = fcmp one double %sa, %sb
  br label %ssel
s_lt:
  %r_lt = fcmp olt double %sa, %sb
  br label %ssel
s_le:
  %r_le = fcmp ole double %sa, %sb
  br label %ssel
s_gt:
  %r_gt = fcmp ogt double %sa, %sb
  br label %ssel
s_ge:
  %r_ge = fcmp oge double %sa, %sb
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

; ---------------------------------------------------------------------------
; compare engines (col-scalar): b broadcast to a splat vector.
; ---------------------------------------------------------------------------

define internal void @cmp_eng_i32_s(ptr readonly %a, i32 %sv, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  %s0 = insertelement <4 x i32> undef, i32 %sv, i64 0
  %vb = shufflevector <4 x i32> %s0, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 4
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %va = load <4 x i32>, ptr %pa, align 4
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = icmp eq <4 x i32> %va, %vb
  br label %vsel
v_ne:
  %m_ne = icmp ne <4 x i32> %va, %vb
  br label %vsel
v_lt:
  %m_lt = icmp slt <4 x i32> %va, %vb
  br label %vsel
v_le:
  %m_le = icmp sle <4 x i32> %va, %vb
  br label %vsel
v_gt:
  %m_gt = icmp sgt <4 x i32> %va, %vb
  br label %vsel
v_ge:
  %m_ge = icmp sge <4 x i32> %va, %vb
  br label %vsel
vsel:
  %m = phi <4 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <4 x i1> %m to <4 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <4 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 4
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds i32, ptr %a, i64 %j
  %sa = load i32, ptr %ta, align 4
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = icmp eq i32 %sa, %sv
  br label %ssel
s_ne:
  %r_ne = icmp ne i32 %sa, %sv
  br label %ssel
s_lt:
  %r_lt = icmp slt i32 %sa, %sv
  br label %ssel
s_le:
  %r_le = icmp sle i32 %sa, %sv
  br label %ssel
s_gt:
  %r_gt = icmp sgt i32 %sa, %sv
  br label %ssel
s_ge:
  %r_ge = icmp sge i32 %sa, %sv
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_i64_s(ptr readonly %a, i64 %sv, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  %s0 = insertelement <2 x i64> undef, i64 %sv, i64 0
  %vb = shufflevector <2 x i64> %s0, <2 x i64> undef, <2 x i32> zeroinitializer
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 2
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i64, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %pa, align 8
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = icmp eq <2 x i64> %va, %vb
  br label %vsel
v_ne:
  %m_ne = icmp ne <2 x i64> %va, %vb
  br label %vsel
v_lt:
  %m_lt = icmp slt <2 x i64> %va, %vb
  br label %vsel
v_le:
  %m_le = icmp sle <2 x i64> %va, %vb
  br label %vsel
v_gt:
  %m_gt = icmp sgt <2 x i64> %va, %vb
  br label %vsel
v_ge:
  %m_ge = icmp sge <2 x i64> %va, %vb
  br label %vsel
vsel:
  %m = phi <2 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <2 x i1> %m to <2 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <2 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 2
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds i64, ptr %a, i64 %j
  %sa = load i64, ptr %ta, align 8
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = icmp eq i64 %sa, %sv
  br label %ssel
s_ne:
  %r_ne = icmp ne i64 %sa, %sv
  br label %ssel
s_lt:
  %r_lt = icmp slt i64 %sa, %sv
  br label %ssel
s_le:
  %r_le = icmp sle i64 %sa, %sv
  br label %ssel
s_gt:
  %r_gt = icmp sgt i64 %sa, %sv
  br label %ssel
s_ge:
  %r_ge = icmp sge i64 %sa, %sv
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_f32_s(ptr readonly %a, float %sv, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  %s0 = insertelement <4 x float> undef, float %sv, i64 0
  %vb = shufflevector <4 x float> %s0, <4 x float> undef, <4 x i32> zeroinitializer
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 4
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds float, ptr %a, i64 %i
  %va = load <4 x float>, ptr %pa, align 4
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = fcmp oeq <4 x float> %va, %vb
  br label %vsel
v_ne:
  %m_ne = fcmp one <4 x float> %va, %vb
  br label %vsel
v_lt:
  %m_lt = fcmp olt <4 x float> %va, %vb
  br label %vsel
v_le:
  %m_le = fcmp ole <4 x float> %va, %vb
  br label %vsel
v_gt:
  %m_gt = fcmp ogt <4 x float> %va, %vb
  br label %vsel
v_ge:
  %m_ge = fcmp oge <4 x float> %va, %vb
  br label %vsel
vsel:
  %m = phi <4 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <4 x i1> %m to <4 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <4 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 4
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds float, ptr %a, i64 %j
  %sa = load float, ptr %ta, align 4
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = fcmp oeq float %sa, %sv
  br label %ssel
s_ne:
  %r_ne = fcmp one float %sa, %sv
  br label %ssel
s_lt:
  %r_lt = fcmp olt float %sa, %sv
  br label %ssel
s_le:
  %r_le = fcmp ole float %sa, %sv
  br label %ssel
s_gt:
  %r_gt = fcmp ogt float %sa, %sv
  br label %ssel
s_ge:
  %r_ge = fcmp oge float %sa, %sv
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal void @cmp_eng_f64_s(ptr readonly %a, double %sv, ptr writeonly noalias %out, i64 %n, i32 %op) #7 {
entry:
  %s0 = insertelement <2 x double> undef, double %sv, i64 0
  %vb = shufflevector <2 x double> %s0, <2 x double> undef, <2 x i32> zeroinitializer
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 2
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds double, ptr %a, i64 %i
  %va = load <2 x double>, ptr %pa, align 8
  switch i32 %op, label %v_eq [ i32 0, label %v_eq
                                i32 1, label %v_ne
                                i32 2, label %v_lt
                                i32 3, label %v_le
                                i32 4, label %v_gt
                                i32 5, label %v_ge ]
v_eq:
  %m_eq = fcmp oeq <2 x double> %va, %vb
  br label %vsel
v_ne:
  %m_ne = fcmp one <2 x double> %va, %vb
  br label %vsel
v_lt:
  %m_lt = fcmp olt <2 x double> %va, %vb
  br label %vsel
v_le:
  %m_le = fcmp ole <2 x double> %va, %vb
  br label %vsel
v_gt:
  %m_gt = fcmp ogt <2 x double> %va, %vb
  br label %vsel
v_ge:
  %m_ge = fcmp oge <2 x double> %va, %vb
  br label %vsel
vsel:
  %m = phi <2 x i1> [ %m_eq, %v_eq ], [ %m_ne, %v_ne ], [ %m_lt, %v_lt ], [ %m_le, %v_le ], [ %m_gt, %v_gt ], [ %m_ge, %v_ge ]
  %b8 = zext <2 x i1> %m to <2 x i8>
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <2 x i8> %b8, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 2
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds double, ptr %a, i64 %j
  %sa = load double, ptr %ta, align 8
  switch i32 %op, label %s_eq [ i32 0, label %s_eq
                                i32 1, label %s_ne
                                i32 2, label %s_lt
                                i32 3, label %s_le
                                i32 4, label %s_gt
                                i32 5, label %s_ge ]
s_eq:
  %r_eq = fcmp oeq double %sa, %sv
  br label %ssel
s_ne:
  %r_ne = fcmp one double %sa, %sv
  br label %ssel
s_lt:
  %r_lt = fcmp olt double %sa, %sv
  br label %ssel
s_le:
  %r_le = fcmp ole double %sa, %sv
  br label %ssel
s_gt:
  %r_gt = fcmp ogt double %sa, %sv
  br label %ssel
s_ge:
  %r_ge = fcmp oge double %sa, %sv
  br label %ssel
ssel:
  %r = phi i1 [ %r_eq, %s_eq ], [ %r_ne, %s_ne ], [ %r_lt, %s_lt ], [ %r_le, %s_le ], [ %r_gt, %s_gt ], [ %r_ge, %s_ge ]
  %r8 = zext i1 %r to i8
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %r8, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

; ---------------------------------------------------------------------------
; col-col dispatcher + public comparators
; ---------------------------------------------------------------------------

define internal ptr @cmp_dispatch(ptr %a, ptr %b, i32 %op) #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %bad0 = or i1 %an, %bn
  br i1 %bad0, label %fail, label %chk, !prof !0
chk:
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %dne = icmp ne i32 %da, %db
  %dnum = icmp ugt i32 %da, 3
  %bad1 = or i1 %dne, %dnum
  br i1 %bad1, label %fail, label %chklen, !prof !0
chklen:
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %lne = icmp ne i64 %la, %lb
  br i1 %lne, label %fail, label %alloc, !prof !0
alloc:
  %out = call ptr @cmp_mk_bool(i64 %la)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %run, !prof !0
run:
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %bvpp = getelementptr inbounds i8, ptr %b, i64 24
  %bvals = load ptr, ptr %bvpp, align 8
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  switch i32 %da, label %d_i32 [ i32 0, label %d_i32
                                 i32 1, label %d_i64
                                 i32 2, label %d_f32
                                 i32 3, label %d_f64 ]
d_i32:
  call void @cmp_eng_i32(ptr %avals, ptr %bvals, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_i64:
  call void @cmp_eng_i64(ptr %avals, ptr %bvals, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_f32:
  call void @cmp_eng_f32(ptr %avals, ptr %bvals, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_f64:
  call void @cmp_eng_f64(ptr %avals, ptr %bvals, ptr %ovals, i64 %la, i32 %op)
  br label %vld
vld:
  call void @cmp_validity_and(ptr %out, ptr %a, ptr %b, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

define ptr @universe_dataframe_eq(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 0)
  ret ptr %r
}
define ptr @universe_dataframe_neq(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 1)
  ret ptr %r
}
define ptr @universe_dataframe_lt(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 2)
  ret ptr %r
}
define ptr @universe_dataframe_lte(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 3)
  ret ptr %r
}
define ptr @universe_dataframe_gt(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 4)
  ret ptr %r
}
define ptr @universe_dataframe_gte(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch(ptr %a, ptr %b, i32 5)
  ret ptr %r
}

; ---------------------------------------------------------------------------
; col-scalar dispatcher + public comparators. stype documents the caller's
; intent; the broadcast type is derived from the COLUMN dtype (int cols read
; ival, float cols read fval).
; ---------------------------------------------------------------------------

define internal ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 %op) #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %chk, !prof !0
chk:
  %da = load i32, ptr %a, align 8
  %dnum = icmp ugt i32 %da, 3
  br i1 %dnum, label %fail, label %alloc, !prof !0
alloc:
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %out = call ptr @cmp_mk_bool(i64 %la)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %run, !prof !0
run:
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  switch i32 %da, label %d_i32 [ i32 0, label %d_i32
                                 i32 1, label %d_i64
                                 i32 2, label %d_f32
                                 i32 3, label %d_f64 ]
d_i32:
  %sv32 = trunc i64 %ival to i32
  call void @cmp_eng_i32_s(ptr %avals, i32 %sv32, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_i64:
  call void @cmp_eng_i64_s(ptr %avals, i64 %ival, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_f32:
  %svf = fptrunc double %fval to float
  call void @cmp_eng_f32_s(ptr %avals, float %svf, ptr %ovals, i64 %la, i32 %op)
  br label %vld
d_f64:
  call void @cmp_eng_f64_s(ptr %avals, double %fval, ptr %ovals, i64 %la, i32 %op)
  br label %vld
vld:
  call void @cmp_validity_copy(ptr %out, ptr %a, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

define ptr @universe_dataframe_eq_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 0)
  ret ptr %r
}
define ptr @universe_dataframe_neq_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 1)
  ret ptr %r
}
define ptr @universe_dataframe_lt_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 2)
  ret ptr %r
}
define ptr @universe_dataframe_lte_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 3)
  ret ptr %r
}
define ptr @universe_dataframe_gt_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 4)
  ret ptr %r
}
define ptr @universe_dataframe_gte_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_dispatch_scalar(ptr %a, i32 %stype, i64 %ival, double %fval, i32 5)
  ret ptr %r
}

; ---------------------------------------------------------------------------
; boolean algebra over BOOL columns (<16 x i8> bitwise). Values are 0/1, so
; bitwise and/or/xor is exactly logical and/or/xor and stays 0/1.
;   bop: 0 and   1 or   2 xor
; ---------------------------------------------------------------------------

define internal void @cmp_bool_eng(ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n, i32 %bop) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vcont ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %pb = getelementptr inbounds i8, ptr %b, i64 %i
  %vb = load <16 x i8>, ptr %pb, align 1
  switch i32 %bop, label %v_and [ i32 0, label %v_and
                                  i32 1, label %v_or
                                  i32 2, label %v_xor ]
v_and:
  %r_and = and <16 x i8> %va, %vb
  br label %vsel
v_or:
  %r_or = or <16 x i8> %va, %vb
  br label %vsel
v_xor:
  %r_xor = xor <16 x i8> %va, %vb
  br label %vsel
vsel:
  %rv = phi <16 x i8> [ %r_and, %v_and ], [ %r_or, %v_or ], [ %r_xor, %v_xor ]
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <16 x i8> %rv, ptr %po, align 1
  br label %vcont
vcont:
  %inext = add i64 %i, 16
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tcont ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %ta = getelementptr inbounds i8, ptr %a, i64 %j
  %sa = load i8, ptr %ta, align 1
  %tb = getelementptr inbounds i8, ptr %b, i64 %j
  %sb = load i8, ptr %tb, align 1
  switch i32 %bop, label %s_and [ i32 0, label %s_and
                                  i32 1, label %s_or
                                  i32 2, label %s_xor ]
s_and:
  %sr_and = and i8 %sa, %sb
  br label %ssel
s_or:
  %sr_or = or i8 %sa, %sb
  br label %ssel
s_xor:
  %sr_xor = xor i8 %sa, %sb
  br label %ssel
ssel:
  %sr = phi i8 [ %sr_and, %s_and ], [ %sr_or, %s_or ], [ %sr_xor, %s_xor ]
  %pj = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %sr, ptr %pj, align 1
  br label %tcont
tcont:
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define internal ptr @cmp_bool_dispatch(ptr %a, ptr %b, i32 %bop) #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %bad0 = or i1 %an, %bn
  br i1 %bad0, label %fail, label %chk, !prof !0
chk:
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %nba = icmp ne i32 %da, 4
  %nbb = icmp ne i32 %db, 4
  %bad1 = or i1 %nba, %nbb
  br i1 %bad1, label %fail, label %chklen, !prof !0
chklen:
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %lne = icmp ne i64 %la, %lb
  br i1 %lne, label %fail, label %alloc, !prof !0
alloc:
  %out = call ptr @cmp_mk_bool(i64 %la)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %run, !prof !0
run:
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %bvpp = getelementptr inbounds i8, ptr %b, i64 24
  %bvals = load ptr, ptr %bvpp, align 8
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  call void @cmp_bool_eng(ptr %avals, ptr %bvals, ptr %ovals, i64 %la, i32 %bop)
  call void @cmp_validity_and(ptr %out, ptr %a, ptr %b, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

define ptr @universe_dataframe_and(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_bool_dispatch(ptr %a, ptr %b, i32 0)
  ret ptr %r
}
define ptr @universe_dataframe_or(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_bool_dispatch(ptr %a, ptr %b, i32 1)
  ret ptr %r
}
define ptr @universe_dataframe_xor(ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %r = call ptr @cmp_bool_dispatch(ptr %a, ptr %b, i32 2)
  ret ptr %r
}

; not(a): logical negation of a 0/1 BOOL column = xor with 1. validity copied.
define ptr @universe_dataframe_not(ptr %a) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %chk, !prof !0
chk:
  %da = load i32, ptr %a, align 8
  %nb = icmp ne i32 %da, 4
  br i1 %nb, label %fail, label %alloc, !prof !0
alloc:
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %out = call ptr @cmp_mk_bool(i64 %la)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %run, !prof !0
run:
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  %ones = insertelement <16 x i8> undef, i8 1, i64 0
  %onev = shufflevector <16 x i8> %ones, <16 x i8> undef, <16 x i32> zeroinitializer
  br label %vhead
vhead:
  %i = phi i64 [ 0, %run ], [ %inext, %vbody ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %la
  br i1 %fits, label %vbody, label %thead
vbody:
  %pa = getelementptr inbounds i8, ptr %avals, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %nv = xor <16 x i8> %va, %onev
  %po = getelementptr inbounds i8, ptr %ovals, i64 %i
  store <16 x i8> %nv, ptr %po, align 1
  %inext = add i64 %i, 16
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tbody ]
  %tc = icmp ult i64 %j, %la
  br i1 %tc, label %tbody, label %vldp
tbody:
  %ta = getelementptr inbounds i8, ptr %avals, i64 %j
  %sa = load i8, ptr %ta, align 1
  %sn = xor i8 %sa, 1
  %pj = getelementptr inbounds i8, ptr %ovals, i64 %j
  store i8 %sn, ptr %pj, align 1
  %jn = add i64 %j, 1
  br label %thead
vldp:
  call void @cmp_validity_copy(ptr %out, ptr %a, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

; ---------------------------------------------------------------------------
; horizontal boolean reductions. Nulls are IGNORED (Kleene-ish): a null lane is
; not "true" for any, not counted for sum, and does not force all false.
; ---------------------------------------------------------------------------

; any(bool_series, out_bool) -> i32. SIMD reduce.or on the no-null fast path.
define i32 @universe_dataframe_any(ptr %s, ptr %out_bool) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out_bool, null
  %bad = or i1 %sn, %on
  br i1 %bad, label %errnull, label %chk, !prof !0
errnull:
  ret i32 1
chk:
  %da = load i32, ptr %s, align 8
  %nb = icmp ne i32 %da, 4
  br i1 %nb, label %errarg, label %setup, !prof !0
errarg:
  ret i32 8
setup:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %n = load i64, ptr %lp, align 8
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  %vldp = getelementptr inbounds i8, ptr %s, i64 32
  %vld = load ptr, ptr %vldp, align 8
  %hasnull = icmp ne ptr %vld, null
  br i1 %hasnull, label %slow, label %fast
fast:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %fast ], [ %inext, %vbody ]
  %acc = phi i8 [ 0, %fast ], [ %accn, %vbody ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %ftail
vbody:
  %pa = getelementptr inbounds i8, ptr %vals, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %vne = icmp ne <16 x i8> %va, zeroinitializer
  %v01 = zext <16 x i1> %vne to <16 x i8>
  %red = call i8 @llvm.vector.reduce.or.v16i8(<16 x i8> %v01)
  %accn = or i8 %acc, %red
  %inext = add i64 %i, 16
  br label %vhead
ftail:
  %j = phi i64 [ %i, %vhead ], [ %jn, %ftbody ]
  %tacc = phi i8 [ %acc, %vhead ], [ %taccn, %ftbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %ftbody, label %fastfin
ftbody:
  %ta = getelementptr inbounds i8, ptr %vals, i64 %j
  %sa = load i8, ptr %ta, align 1
  %snz = icmp ne i8 %sa, 0
  %s01 = zext i1 %snz to i8
  %taccn = or i8 %tacc, %s01
  %jn = add i64 %j, 1
  br label %ftail
fastfin:
  %anyf = icmp ne i8 %tacc, 0
  %rf = zext i1 %anyf to i8
  store i8 %rf, ptr %out_bool, align 1
  ret i32 0
slow:
  br label %shead
shead:
  %sj = phi i64 [ 0, %slow ], [ %sjn, %scont ]
  %sfound = phi i1 [ false, %slow ], [ %sfn, %scont ]
  %scmp = icmp ult i64 %sj, %n
  br i1 %scmp, label %sbody, label %slowfin
sbody:
  %sv = call i1 @cmp_valid_at(ptr %s, i64 %sj)
  %svp = getelementptr inbounds i8, ptr %vals, i64 %sj
  %svv = load i8, ptr %svp, align 1
  %svnz = icmp ne i8 %svv, 0
  %istrue = and i1 %sv, %svnz
  br label %scont
scont:
  %sfn = or i1 %sfound, %istrue
  %sjn = add i64 %sj, 1
  br label %shead
slowfin:
  %rs = zext i1 %sfound to i8
  store i8 %rs, ptr %out_bool, align 1
  ret i32 0
}

; all(bool_series, out_bool) -> i32. SIMD reduce.and on the no-null fast path.
define i32 @universe_dataframe_all(ptr %s, ptr %out_bool) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out_bool, null
  %bad = or i1 %sn, %on
  br i1 %bad, label %errnull, label %chk, !prof !0
errnull:
  ret i32 1
chk:
  %da = load i32, ptr %s, align 8
  %nb = icmp ne i32 %da, 4
  br i1 %nb, label %errarg, label %setup, !prof !0
errarg:
  ret i32 8
setup:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %n = load i64, ptr %lp, align 8
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  %vldp = getelementptr inbounds i8, ptr %s, i64 32
  %vld = load ptr, ptr %vldp, align 8
  %hasnull = icmp ne ptr %vld, null
  br i1 %hasnull, label %slow, label %fast
fast:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %fast ], [ %inext, %vbody ]
  %acc = phi i8 [ 1, %fast ], [ %accn, %vbody ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %ftail
vbody:
  %pa = getelementptr inbounds i8, ptr %vals, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %vne = icmp ne <16 x i8> %va, zeroinitializer
  %v01 = zext <16 x i1> %vne to <16 x i8>
  %red = call i8 @llvm.vector.reduce.and.v16i8(<16 x i8> %v01)
  %accn = and i8 %acc, %red
  %inext = add i64 %i, 16
  br label %vhead
ftail:
  %j = phi i64 [ %i, %vhead ], [ %jn, %ftbody ]
  %tacc = phi i8 [ %acc, %vhead ], [ %taccn, %ftbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %ftbody, label %fastfin
ftbody:
  %ta = getelementptr inbounds i8, ptr %vals, i64 %j
  %sa = load i8, ptr %ta, align 1
  %snz = icmp ne i8 %sa, 0
  %s01 = zext i1 %snz to i8
  %taccn = and i8 %tacc, %s01
  %jn = add i64 %j, 1
  br label %ftail
fastfin:
  %allf = icmp ne i8 %tacc, 0
  %rf = zext i1 %allf to i8
  store i8 %rf, ptr %out_bool, align 1
  ret i32 0
slow:
  br label %shead
shead:
  %sj = phi i64 [ 0, %slow ], [ %sjn, %scont ]
  %sall = phi i1 [ true, %slow ], [ %san, %scont ]
  %scmp = icmp ult i64 %sj, %n
  br i1 %scmp, label %sbody, label %slowfin
sbody:
  %sv = call i1 @cmp_valid_at(ptr %s, i64 %sj)
  %svp = getelementptr inbounds i8, ptr %vals, i64 %sj
  %svv = load i8, ptr %svp, align 1
  %svz = icmp eq i8 %svv, 0
  %isfalse = and i1 %sv, %svz
  %keep = xor i1 %isfalse, true
  br label %scont
scont:
  %san = and i1 %sall, %keep
  %sjn = add i64 %sj, 1
  br label %shead
slowfin:
  %rs = zext i1 %sall to i8
  store i8 %rs, ptr %out_bool, align 1
  ret i32 0
}

; sum_bool(bool_series, out_i64) -> i32. Count of true(&valid) lanes.
define i32 @universe_dataframe_sum_bool(ptr %s, ptr %out_i64) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out_i64, null
  %bad = or i1 %sn, %on
  br i1 %bad, label %errnull, label %chk, !prof !0
errnull:
  ret i32 1
chk:
  %da = load i32, ptr %s, align 8
  %nb = icmp ne i32 %da, 4
  br i1 %nb, label %errarg, label %setup, !prof !0
errarg:
  ret i32 8
setup:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %n = load i64, ptr %lp, align 8
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  %vldp = getelementptr inbounds i8, ptr %s, i64 32
  %vld = load ptr, ptr %vldp, align 8
  %hasnull = icmp ne ptr %vld, null
  br i1 %hasnull, label %slow, label %fast
fast:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %fast ], [ %inext, %vbody ]
  %acc = phi i64 [ 0, %fast ], [ %accn, %vbody ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %ftail
vbody:
  %pa = getelementptr inbounds i8, ptr %vals, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %vne = icmp ne <16 x i8> %va, zeroinitializer
  %v01 = zext <16 x i1> %vne to <16 x i8>
  %red = call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %v01)
  %redz = zext i8 %red to i64
  %accn = add i64 %acc, %redz
  %inext = add i64 %i, 16
  br label %vhead
ftail:
  %j = phi i64 [ %i, %vhead ], [ %jn, %ftbody ]
  %tacc = phi i64 [ %acc, %vhead ], [ %taccn, %ftbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %ftbody, label %fastfin
ftbody:
  %ta = getelementptr inbounds i8, ptr %vals, i64 %j
  %sa = load i8, ptr %ta, align 1
  %snz = icmp ne i8 %sa, 0
  %s01 = zext i1 %snz to i64
  %taccn = add i64 %tacc, %s01
  %jn = add i64 %j, 1
  br label %ftail
fastfin:
  store i64 %tacc, ptr %out_i64, align 8
  ret i32 0
slow:
  br label %shead
shead:
  %sj = phi i64 [ 0, %slow ], [ %sjn, %scont ]
  %scnt = phi i64 [ 0, %slow ], [ %scntn, %scont ]
  %scmp = icmp ult i64 %sj, %n
  br i1 %scmp, label %sbody, label %slowfin
sbody:
  %sv = call i1 @cmp_valid_at(ptr %s, i64 %sj)
  %svp = getelementptr inbounds i8, ptr %vals, i64 %sj
  %svv = load i8, ptr %svp, align 1
  %svnz = icmp ne i8 %svv, 0
  %istrue = and i1 %sv, %svnz
  br label %scont
scont:
  %inc = zext i1 %istrue to i64
  %scntn = add i64 %scnt, %inc
  %sjn = add i64 %sj, 1
  br label %shead
slowfin:
  store i64 %scnt, ptr %out_i64, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; zip_with(mask, a, b): per-lane branchless select — result[i] = (mask[i] valid
; & true) ? a[i] : b[i]. result dtype = a.dtype. Values selected with a vector
; `select`; validity chosen per lane from the picked source.
; ---------------------------------------------------------------------------

; width-4 value select (<4 x i32> covers I32 and F32 bit-for-bit)
define internal void @cmp_zip_w4(ptr readonly %mask, ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vbody ]
  %lim = add i64 %i, 4
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pm = getelementptr inbounds i8, ptr %mask, i64 %i
  %vm8 = load <4 x i8>, ptr %pm, align 1
  %vm = icmp ne <4 x i8> %vm8, zeroinitializer
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %va = load <4 x i32>, ptr %pa, align 4
  %pb = getelementptr inbounds i32, ptr %b, i64 %i
  %vb = load <4 x i32>, ptr %pb, align 4
  %vr = select <4 x i1> %vm, <4 x i32> %va, <4 x i32> %vb
  %po = getelementptr inbounds i32, ptr %out, i64 %i
  store <4 x i32> %vr, ptr %po, align 4
  %inext = add i64 %i, 4
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %tm = getelementptr inbounds i8, ptr %mask, i64 %j
  %sm = load i8, ptr %tm, align 1
  %smnz = icmp ne i8 %sm, 0
  %ta = getelementptr inbounds i32, ptr %a, i64 %j
  %sa = load i32, ptr %ta, align 4
  %tb = getelementptr inbounds i32, ptr %b, i64 %j
  %sb = load i32, ptr %tb, align 4
  %sr = select i1 %smnz, i32 %sa, i32 %sb
  %po2 = getelementptr inbounds i32, ptr %out, i64 %j
  store i32 %sr, ptr %po2, align 4
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

; width-8 value select (<2 x i64> covers I64 and F64 bit-for-bit)
define internal void @cmp_zip_w8(ptr readonly %mask, ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vbody ]
  %lim = add i64 %i, 2
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pm = getelementptr inbounds i8, ptr %mask, i64 %i
  %vm8 = load <2 x i8>, ptr %pm, align 1
  %vm = icmp ne <2 x i8> %vm8, zeroinitializer
  %pa = getelementptr inbounds i64, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %pb, align 8
  %vr = select <2 x i1> %vm, <2 x i64> %va, <2 x i64> %vb
  %po = getelementptr inbounds i64, ptr %out, i64 %i
  store <2 x i64> %vr, ptr %po, align 8
  %inext = add i64 %i, 2
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %tm = getelementptr inbounds i8, ptr %mask, i64 %j
  %sm = load i8, ptr %tm, align 1
  %smnz = icmp ne i8 %sm, 0
  %ta = getelementptr inbounds i64, ptr %a, i64 %j
  %sa = load i64, ptr %ta, align 8
  %tb = getelementptr inbounds i64, ptr %b, i64 %j
  %sb = load i64, ptr %tb, align 8
  %sr = select i1 %smnz, i64 %sa, i64 %sb
  %po2 = getelementptr inbounds i64, ptr %out, i64 %j
  store i64 %sr, ptr %po2, align 8
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

; width-1 value select (<16 x i8> for BOOL)
define internal void @cmp_zip_w1(ptr readonly %mask, ptr readonly %a, ptr readonly %b, ptr writeonly noalias %out, i64 %n) #7 {
entry:
  br label %vhead
vhead:
  %i = phi i64 [ 0, %entry ], [ %inext, %vbody ]
  %lim = add i64 %i, 16
  %fits = icmp ule i64 %lim, %n
  br i1 %fits, label %vbody, label %thead
vbody:
  %pm = getelementptr inbounds i8, ptr %mask, i64 %i
  %vm8 = load <16 x i8>, ptr %pm, align 1
  %vm = icmp ne <16 x i8> %vm8, zeroinitializer
  %pa = getelementptr inbounds i8, ptr %a, i64 %i
  %va = load <16 x i8>, ptr %pa, align 1
  %pb = getelementptr inbounds i8, ptr %b, i64 %i
  %vb = load <16 x i8>, ptr %pb, align 1
  %vr = select <16 x i1> %vm, <16 x i8> %va, <16 x i8> %vb
  %po = getelementptr inbounds i8, ptr %out, i64 %i
  store <16 x i8> %vr, ptr %po, align 1
  %inext = add i64 %i, 16
  br label %vhead
thead:
  %j = phi i64 [ %i, %vhead ], [ %jn, %tbody ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tbody, label %ret
tbody:
  %tm = getelementptr inbounds i8, ptr %mask, i64 %j
  %sm = load i8, ptr %tm, align 1
  %smnz = icmp ne i8 %sm, 0
  %ta = getelementptr inbounds i8, ptr %a, i64 %j
  %sa = load i8, ptr %ta, align 1
  %tb = getelementptr inbounds i8, ptr %b, i64 %j
  %sb = load i8, ptr %tb, align 1
  %sr = select i1 %smnz, i8 %sa, i8 %sb
  %po2 = getelementptr inbounds i8, ptr %out, i64 %j
  store i8 %sr, ptr %po2, align 1
  %jn = add i64 %j, 1
  br label %thead
ret:
  ret void
}

define ptr @universe_dataframe_zip_with(ptr %mask, ptr %a, ptr %b) local_unnamed_addr #1 {
entry:
  %mn = icmp eq ptr %mask, null
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %b0 = or i1 %mn, %an
  %b1 = or i1 %b0, %bn
  br i1 %b1, label %fail, label %chk, !prof !0
chk:
  %dm = load i32, ptr %mask, align 8
  %nbm = icmp ne i32 %dm, 4
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %dne = icmp ne i32 %da, %db
  %isstr = icmp eq i32 %da, 5
  %bad0 = or i1 %nbm, %dne
  %bad1 = or i1 %bad0, %isstr
  br i1 %bad1, label %fail, label %chklen, !prof !0
chklen:
  %mlp = getelementptr inbounds i8, ptr %mask, i64 8
  %mlen = load i64, ptr %mlp, align 8
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %lne0 = icmp ne i64 %mlen, %la
  %lne1 = icmp ne i64 %la, %lb
  %lne = or i1 %lne0, %lne1
  br i1 %lne, label %fail, label %alloc, !prof !0
alloc:
  %out = call ptr @cmp_mk_fixed(i32 %da, i64 %la)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %run, !prof !0
run:
  %mvpp = getelementptr inbounds i8, ptr %mask, i64 24
  %mvals = load ptr, ptr %mvpp, align 8
  %avpp = getelementptr inbounds i8, ptr %a, i64 24
  %avals = load ptr, ptr %avpp, align 8
  %bvpp = getelementptr inbounds i8, ptr %b, i64 24
  %bvals = load ptr, ptr %bvpp, align 8
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  %w = call i64 @cmp_width(i32 %da)
  switch i64 %w, label %w1 [ i64 4, label %w4
                             i64 8, label %w8
                             i64 1, label %w1 ]
w4:
  call void @cmp_zip_w4(ptr %mvals, ptr %avals, ptr %bvals, ptr %ovals, i64 %la)
  br label %vld
w8:
  call void @cmp_zip_w8(ptr %mvals, ptr %avals, ptr %bvals, ptr %ovals, i64 %la)
  br label %vld
w1:
  call void @cmp_zip_w1(ptr %mvals, ptr %avals, ptr %bvals, ptr %ovals, i64 %la)
  br label %vld
vld:
  ; per-lane validity: pick a-side validity where (mask valid & true), else
  ; b-side validity. Build a bitmap only if any lane is null. (cold path.)
  %t = add i64 %la, 7
  %nbytes = lshr i64 %t, 3
  %bm = call ptr @cmp_xmalloc(i64 %nbytes)
  call void @llvm.memset.p0.i64(ptr %bm, i8 -1, i64 %nbytes, i1 false)
  br label %vloop
vloop:
  %k = phi i64 [ 0, %vld ], [ %kn, %next ]
  %nulls = phi i64 [ 0, %vld ], [ %nn, %next ]
  %cmp = icmp ult i64 %k, %la
  br i1 %cmp, label %vbody, label %vfin
vbody:
  %mvalid = call i1 @cmp_valid_at(ptr %mask, i64 %k)
  %mvp = getelementptr inbounds i8, ptr %mvals, i64 %k
  %mv = load i8, ptr %mvp, align 1
  %mnz = icmp ne i8 %mv, 0
  %pickA = and i1 %mvalid, %mnz
  br i1 %pickA, label %fromA, label %fromB
fromA:
  %av = call i1 @cmp_valid_at(ptr %a, i64 %k)
  br label %vcont
fromB:
  %bv = call i1 @cmp_valid_at(ptr %b, i64 %k)
  br label %vcont
vcont:
  %lanevalid = phi i1 [ %av, %fromA ], [ %bv, %fromB ]
  br i1 %lanevalid, label %isvalid, label %isnull
isnull:
  call void @cmp_bm_clear(ptr %bm, i64 %k)
  br label %next
isvalid:
  br label %next
next:
  %inc = zext i1 %lanevalid to i64
  %nnv = xor i64 %inc, 1
  %nn = add i64 %nulls, %nnv
  %kn = add i64 %k, 1
  br label %vloop
vfin:
  %anynull = icmp ugt i64 %nulls, 0
  br i1 %anynull, label %keepbm, label %dropbm
keepbm:
  %ovldp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %ovldp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nulls, ptr %oncp, align 8
  ret ptr %out
dropbm:
  call void @free(ptr %bm)
  ret ptr %out
fail:
  ret ptr null
}

attributes #0 = { nounwind willreturn norecurse nosync }
attributes #1 = { nounwind willreturn }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #5 = { alwaysinline nounwind willreturn norecurse nosync memory(read) }
attributes #6 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #7 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }

!0 = !{!"branch_weights", i32 1, i32 2000}
