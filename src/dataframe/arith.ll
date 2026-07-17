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

; DataFrame SIMD element-wise arithmetic (numpy-ufunc / Polars-kernel analog).
; Operates on Series handles (frame.ll DOWNSTREAM CONTRACT). Hand-written IR.
;
; ============================ DESIGN ============================
; Series layout relied upon (frame.ll):
;   +0 i32 dtype (I32=0,I64=1,F32=2,F64=3,BOOL=4,STR=5) ; +8 i64 len
;   +16 i64 null_count ; +24 ptr values ; +32 ptr validity
;   validity: null => all valid; else bitmap ceil(len/8) bytes, bit=1 VALID.
;
; ALGORITHM CLASS — SIMD-first, scalar tail/oracle (CLAUDE.md "SIMD-first with
; scalar fallback (MANDATORY)"):
;   * VALUE HOT PATH is a portable 128-bit vector loop per dtype:
;       I32 -> <4 x i32>, I64 -> <2 x i64>, F32 -> <4 x float>, F64 -> <2 x
;       double>. These lower to SSE2 on x86 and NEON on AArch64 (both baseline
;       everywhere we ship) so NO runtime CPU check — the 128-bit path is the
;       DEFAULT, not optional SIMD. add/sub/mul (int+float) and div (float) are
;       genuinely packed (paddd/addps/addpd, subps, pmulld/mulps/mulpd, divps/
;       divpd). A scalar tail finishes the len % lanes remainder AND is the
;       reference the tests cross-check the vector path against.
;   * NULL STRATEGY (Polars/arrow2): compute VALUES DENSELY over EVERY lane
;     (garbage in null lanes is harmless); combine the input validity bitmaps
;     SEPARATELY with a `<16 x i8>` bitwise AND (a_combine_validity). A null
;     validity ptr means "all valid" (treated as all-ones 0xFF). This keeps the
;     value loop branch-free and fully vectorized — NO per-element null test in
;     the SIMD body. out.validity = a.validity AND b.validity; out.null_count is
;     recomputed from the combined bitmap (padding bits stay 1 => not counted).
;   * INTEGER divide/remainder has NO SIMD instruction on either ISA (SSE/NEON
;     have no packed integer divide) — vectorizing sdiv/srem only scalarizes.
;     So homogeneous int div/rem run a SCALAR branchless-guarded loop that:
;     masks zero divisors AND the INT_MIN/-1 overflow case (both are poison for
;     sdiv/srem) by forcing the divisor to 1 for those lanes, computes safely,
;     then marks the lane NULL (Polars: int div-by-zero => null). Validity for
;     this path = valid_a AND valid_b AND (divisor != 0 && !overflow), built in
;     the same pass. FLOAT div/rem by zero follow IEEE (inf/nan), NOT null.
;   * frem on <N x float>/<N x double> lowers to a per-lane libcall (fmodf/fmod)
;     — inherent to remainder; documented, not a scalarization bug.
;
; DTYPE PROMOTION (documented, uniform, simple):
;   * SAME dtype (da==db) -> that dtype (homogeneous fast SIMD path).
;   * ANY heterogeneous pair -> F64 (values widened to double via sitofp/fpext,
;     computed in double, stored F64). This subsumes "mixed int/float -> F64"
;     and also sends mixed int widths (I32+I64) to F64. Precision note: a large
;     I64 is not exactly representable in F64 — heterogeneous integer arithmetic
;     is lossy by this rule; keep operands same-dtype for exact integer results.
;     The heterogeneous path is a COLD scalar-double loop (rare in practice).
;
; SCALAR BROADCAST (col op scalar): result dtype = the COLUMN's dtype (no
;   promotion); the scalar is taken in that dtype — int column uses `ival`
;   (truncated to width), float column uses `fval`. Values computed by a SIMD
;   splat+vector-op loop; validity is COPIED from the column (a scalar operand
;   introduces no new nulls) EXCEPT int div/rem by a zero scalar => whole column
;   null. add/sub/mul/div broadcast are provided.
;
; UNARY (neg/abs): result dtype = input dtype; validity copied. neg int = 0-x
;   (vector sub, wrapping), neg float = fneg; abs int = llvm.abs (is_int_min_
;   poison=false, so abs(INT_MIN)=INT_MIN), abs float = llvm.fabs.
;
; HORIZONTAL (min/max across all numeric columns, per row): output F64 Series of
;   height rows (uniform with reduce.ll's *_horizontal). Row-wise across columns
;   is a different shape from element-parallel buffer kernels, so this is a cold
;   scalar accumulate (nulls skipped; a row with zero numeric cells => null).
;
; Integer add/sub/mul WRAP on overflow (plain add/sub/mul, no nsw/nuw — hazard
;   #1). Errors mapped to null return for constructors: NULL_PTR (any null
;   input), INVALID_ARG (non-numeric dtype, length mismatch), OOM.
; ===============================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)

declare <4 x i32> @llvm.abs.v4i32(<4 x i32>, i1 immarg)
declare <2 x i64> @llvm.abs.v2i64(<2 x i64>, i1 immarg)
declare i32 @llvm.abs.i32(i32, i1 immarg)
declare i64 @llvm.abs.i64(i64, i1 immarg)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare <2 x double> @llvm.fabs.v2f64(<2 x double>)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare i8 @llvm.ctpop.i8(i8)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.maxnum.f64(double, double)

; frame.ll exports we compose with (cold surface only)
declare ptr @universe_dataframe_series_new(i32, i64)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)

; ---------------------------------------------------------------------------
; internal helpers
; ---------------------------------------------------------------------------

; true if element i is valid (non-null). validity==null => all valid.
define internal i1 @a_valid_at(ptr %s, i64 %i) #2 {
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

; load element i of a fixed-width numeric column (dtype 0..3) as a double.
define internal double @a_load_f64(ptr %vals, i32 %dtype, i64 %i) #2 {
entry:
  switch i32 %dtype, label %dflt [ i32 0, label %l0
                                   i32 1, label %l1
                                   i32 2, label %l2
                                   i32 3, label %l3 ]
l0:
  %p0 = getelementptr inbounds i32, ptr %vals, i64 %i
  %r0 = load i32, ptr %p0, align 4
  %d0 = sitofp i32 %r0 to double
  ret double %d0
l1:
  %p1 = getelementptr inbounds i64, ptr %vals, i64 %i
  %r1 = load i64, ptr %p1, align 8
  %d1 = sitofp i64 %r1 to double
  ret double %d1
l2:
  %p2 = getelementptr inbounds float, ptr %vals, i64 %i
  %r2 = load float, ptr %p2, align 4
  %d2 = fpext float %r2 to double
  ret double %d2
l3:
  %p3 = getelementptr inbounds double, ptr %vals, i64 %i
  %d3 = load double, ptr %p3, align 8
  ret double %d3
dflt:
  ret double 0.0
}

; allocate a validity bitmap (all valid = 0xFF) for len elements.
define internal ptr @a_bm_alloc(i64 %len) #1 {
entry:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %z = icmp eq i64 %nb, 0
  %msz = select i1 %z, i64 1, i64 %nb
  %p = call ptr @malloc(i64 %msz)
  call void @llvm.memset.p0.i64(ptr %p, i8 -1, i64 %nb, i1 false)
  ret ptr %p
}

; clear (mark null) bit k in bitmap bm.
define internal void @a_bm_clear(ptr %bm, i64 %k) #3 {
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

; load a <16 x i8> chunk from p, or an all-0xFF splat when isnull.
define internal <16 x i8> @a_load16(ptr %p, i1 %isnull) #2 {
entry:
  br i1 %isnull, label %ones, label %ld
ld:
  %v = load <16 x i8>, ptr %p, align 1
  ret <16 x i8> %v
ones:
  ret <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
}

; load byte base[i], or 0xFF when isnull.
define internal i8 @a_load8(ptr %base, i64 %i, i1 %isnull) #2 {
entry:
  br i1 %isnull, label %ones, label %ld
ld:
  %p = getelementptr inbounds i8, ptr %base, i64 %i
  %v = load i8, ptr %p, align 1
  ret i8 %v
ones:
  ret i8 -1
}

; combine input validity bitmaps into out: out.validity = a.val AND b.val, via
; a <16 x i8> bitwise AND over the bitmap bytes (scalar-byte tail). A null
; validity ptr is treated as all-ones. If BOTH inputs are all-valid, out is left
; with no bitmap. null_count recomputed from the combined bitmap.
define internal void @a_combine_validity(ptr %out, ptr %a, ptr %b, i64 %len) #1 {
entry:
  %avp = getelementptr inbounds i8, ptr %a, i64 32
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 32
  %bv = load ptr, ptr %bvp, align 8
  %an = icmp eq ptr %av, null
  %bn = icmp eq ptr %bv, null
  %both = and i1 %an, %bn
  br i1 %both, label %ret, label %build
build:
  %t = add i64 %len, 7
  %nbytes = lshr i64 %t, 3
  %bm = call ptr @a_bm_alloc(i64 %len)
  %vbulk = and i64 %nbytes, -16
  br label %vh
vh:
  %vi = phi i64 [ 0, %build ], [ %vin, %vb ]
  %vc = icmp ult i64 %vi, %vbulk
  br i1 %vc, label %vb, label %th
vb:
  %vap = getelementptr inbounds i8, ptr %av, i64 %vi
  %voa = call <16 x i8> @a_load16(ptr %vap, i1 %an)
  %vbp = getelementptr inbounds i8, ptr %bv, i64 %vi
  %vob = call <16 x i8> @a_load16(ptr %vbp, i1 %bn)
  %vand = and <16 x i8> %voa, %vob
  %vop = getelementptr inbounds i8, ptr %bm, i64 %vi
  store <16 x i8> %vand, ptr %vop, align 1
  %vin = add nuw i64 %vi, 16
  br label %vh
th:
  %ti = phi i64 [ %vbulk, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %nbytes
  br i1 %tc, label %tb, label %count
tb:
  %tba = call i8 @a_load8(ptr %av, i64 %ti, i1 %an)
  %tbb = call i8 @a_load8(ptr %bv, i64 %ti, i1 %bn)
  %tband = and i8 %tba, %tbb
  %top = getelementptr inbounds i8, ptr %bm, i64 %ti
  store i8 %tband, ptr %top, align 1
  %tin = add nuw i64 %ti, 1
  br label %th
count:
  br label %ch
ch:
  %ci = phi i64 [ 0, %count ], [ %cin, %cb ]
  %ones = phi i64 [ 0, %count ], [ %onesn, %cb ]
  %cc = icmp ult i64 %ci, %nbytes
  br i1 %cc, label %cb, label %cfin
cb:
  %cbp = getelementptr inbounds i8, ptr %bm, i64 %ci
  %cbyte = load i8, ptr %cbp, align 1
  %pc = call i8 @llvm.ctpop.i8(i8 %cbyte)
  %pcz = zext i8 %pc to i64
  %onesn = add i64 %ones, %pcz
  %cin = add nuw i64 %ci, 1
  br label %ch
cfin:
  %totbits = shl i64 %nbytes, 3
  %nulls = sub i64 %totbits, %ones
  %ncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nulls, ptr %ncp, align 8
  %obmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %obmp, align 8
  ret void
ret:
  ret void
}

; copy src validity to out (unary/scalar broadcast: validity unchanged).
define internal void @a_copy_validity(ptr %out, ptr %src, i64 %len) #1 {
entry:
  %svp = getelementptr inbounds i8, ptr %src, i64 32
  %sv = load ptr, ptr %svp, align 8
  %sn = icmp eq ptr %sv, null
  br i1 %sn, label %ret, label %copy
copy:
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %bm = call ptr @a_bm_alloc(i64 %len)
  call void @llvm.memcpy.p0.p0.i64(ptr %bm, ptr %sv, i64 %nb, i1 false)
  %obmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %obmp, align 8
  %sncp = getelementptr inbounds i8, ptr %src, i64 16
  %snc = load i64, ptr %sncp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %snc, ptr %oncp, align 8
  ret void
ret:
  ret void
}

; ---------------------------------------------------------------------------
; homogeneous SIMD value kernels — add(0)/sub(1)/mul(2) for int, +div(3)/rem(4)
; for float. Op switched ONCE before the loop (cold); each loop monomorphic.
; ---------------------------------------------------------------------------

define internal void @a_bin_i32(i32 %op, ptr %a, ptr %b, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -4
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw i32, ptr %a, i64 %ai
  %ava = load <4 x i32>, ptr %apa, align 4
  %apb = getelementptr inbounds nuw i32, ptr %b, i64 %ai
  %avb = load <4 x i32>, ptr %apb, align 4
  %ar = add <4 x i32> %ava, %avb
  %apo = getelementptr inbounds nuw i32, ptr %o, i64 %ai
  store <4 x i32> %ar, ptr %apo, align 4
  %ain = add nuw i64 %ai, 4
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw i32, ptr %a, i64 %aj
  %atva = load i32, ptr %atpa, align 4
  %atpb = getelementptr inbounds nuw i32, ptr %b, i64 %aj
  %atvb = load i32, ptr %atpb, align 4
  %atr = add i32 %atva, %atvb
  %atpo = getelementptr inbounds nuw i32, ptr %o, i64 %aj
  store i32 %atr, ptr %atpo, align 4
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %st
sb:
  %spa = getelementptr inbounds nuw i32, ptr %a, i64 %si
  %sva = load <4 x i32>, ptr %spa, align 4
  %spb = getelementptr inbounds nuw i32, ptr %b, i64 %si
  %svb = load <4 x i32>, ptr %spb, align 4
  %sr = sub <4 x i32> %sva, %svb
  %spo = getelementptr inbounds nuw i32, ptr %o, i64 %si
  store <4 x i32> %sr, ptr %spo, align 4
  %sin = add nuw i64 %si, 4
  br label %sl
st:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw i32, ptr %a, i64 %sj
  %stva = load i32, ptr %stpa, align 4
  %stpb = getelementptr inbounds nuw i32, ptr %b, i64 %sj
  %stvb = load i32, ptr %stpb, align 4
  %str = sub i32 %stva, %stvb
  %stpo = getelementptr inbounds nuw i32, ptr %o, i64 %sj
  store i32 %str, ptr %stpo, align 4
  %sjn = add nuw i64 %sj, 1
  br label %st
mh:
  br label %ml
ml:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw i32, ptr %a, i64 %mi
  %mva = load <4 x i32>, ptr %mpa, align 4
  %mpb = getelementptr inbounds nuw i32, ptr %b, i64 %mi
  %mvb = load <4 x i32>, ptr %mpb, align 4
  %mr = mul <4 x i32> %mva, %mvb
  %mpo = getelementptr inbounds nuw i32, ptr %o, i64 %mi
  store <4 x i32> %mr, ptr %mpo, align 4
  %min = add nuw i64 %mi, 4
  br label %ml
mt:
  %mj = phi i64 [ %nbulk, %ml ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw i32, ptr %a, i64 %mj
  %mtva = load i32, ptr %mtpa, align 4
  %mtpb = getelementptr inbounds nuw i32, ptr %b, i64 %mj
  %mtvb = load i32, ptr %mtpb, align 4
  %mtr = mul i32 %mtva, %mtvb
  %mtpo = getelementptr inbounds nuw i32, ptr %o, i64 %mj
  store i32 %mtr, ptr %mtpo, align 4
  %mjn = add nuw i64 %mj, 1
  br label %mt
done:
  ret void
}

define internal void @a_bin_i64(i32 %op, ptr %a, ptr %b, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -2
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw i64, ptr %a, i64 %ai
  %ava = load <2 x i64>, ptr %apa, align 8
  %apb = getelementptr inbounds nuw i64, ptr %b, i64 %ai
  %avb = load <2 x i64>, ptr %apb, align 8
  %ar = add <2 x i64> %ava, %avb
  %apo = getelementptr inbounds nuw i64, ptr %o, i64 %ai
  store <2 x i64> %ar, ptr %apo, align 8
  %ain = add nuw i64 %ai, 2
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw i64, ptr %a, i64 %aj
  %atva = load i64, ptr %atpa, align 8
  %atpb = getelementptr inbounds nuw i64, ptr %b, i64 %aj
  %atvb = load i64, ptr %atpb, align 8
  %atr = add i64 %atva, %atvb
  %atpo = getelementptr inbounds nuw i64, ptr %o, i64 %aj
  store i64 %atr, ptr %atpo, align 8
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %st
sb:
  %spa = getelementptr inbounds nuw i64, ptr %a, i64 %si
  %sva = load <2 x i64>, ptr %spa, align 8
  %spb = getelementptr inbounds nuw i64, ptr %b, i64 %si
  %svb = load <2 x i64>, ptr %spb, align 8
  %sr = sub <2 x i64> %sva, %svb
  %spo = getelementptr inbounds nuw i64, ptr %o, i64 %si
  store <2 x i64> %sr, ptr %spo, align 8
  %sin = add nuw i64 %si, 2
  br label %sl
st:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw i64, ptr %a, i64 %sj
  %stva = load i64, ptr %stpa, align 8
  %stpb = getelementptr inbounds nuw i64, ptr %b, i64 %sj
  %stvb = load i64, ptr %stpb, align 8
  %str = sub i64 %stva, %stvb
  %stpo = getelementptr inbounds nuw i64, ptr %o, i64 %sj
  store i64 %str, ptr %stpo, align 8
  %sjn = add nuw i64 %sj, 1
  br label %st
mh:
  br label %ml
ml:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw i64, ptr %a, i64 %mi
  %mva = load <2 x i64>, ptr %mpa, align 8
  %mpb = getelementptr inbounds nuw i64, ptr %b, i64 %mi
  %mvb = load <2 x i64>, ptr %mpb, align 8
  %mr = mul <2 x i64> %mva, %mvb
  %mpo = getelementptr inbounds nuw i64, ptr %o, i64 %mi
  store <2 x i64> %mr, ptr %mpo, align 8
  %min = add nuw i64 %mi, 2
  br label %ml
mt:
  %mj = phi i64 [ %nbulk, %ml ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw i64, ptr %a, i64 %mj
  %mtva = load i64, ptr %mtpa, align 8
  %mtpb = getelementptr inbounds nuw i64, ptr %b, i64 %mj
  %mtvb = load i64, ptr %mtpb, align 8
  %mtr = mul i64 %mtva, %mtvb
  %mtpo = getelementptr inbounds nuw i64, ptr %o, i64 %mj
  store i64 %mtr, ptr %mtpo, align 8
  %mjn = add nuw i64 %mj, 1
  br label %mt
done:
  ret void
}

define internal void @a_bin_f32(i32 %op, ptr %a, ptr %b, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -4
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh
                                i32 3, label %dh  i32 4, label %rh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw float, ptr %a, i64 %ai
  %ava = load <4 x float>, ptr %apa, align 4
  %apb = getelementptr inbounds nuw float, ptr %b, i64 %ai
  %avb = load <4 x float>, ptr %apb, align 4
  %ar = fadd <4 x float> %ava, %avb
  %apo = getelementptr inbounds nuw float, ptr %o, i64 %ai
  store <4 x float> %ar, ptr %apo, align 4
  %ain = add nuw i64 %ai, 4
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw float, ptr %a, i64 %aj
  %atva = load float, ptr %atpa, align 4
  %atpb = getelementptr inbounds nuw float, ptr %b, i64 %aj
  %atvb = load float, ptr %atpb, align 4
  %atr = fadd float %atva, %atvb
  %atpo = getelementptr inbounds nuw float, ptr %o, i64 %aj
  store float %atr, ptr %atpo, align 4
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw float, ptr %a, i64 %si
  %sva = load <4 x float>, ptr %spa, align 4
  %spb = getelementptr inbounds nuw float, ptr %b, i64 %si
  %svb = load <4 x float>, ptr %spb, align 4
  %sr = fsub <4 x float> %sva, %svb
  %spo = getelementptr inbounds nuw float, ptr %o, i64 %si
  store <4 x float> %sr, ptr %spo, align 4
  %sin = add nuw i64 %si, 4
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw float, ptr %a, i64 %sj
  %stva = load float, ptr %stpa, align 4
  %stpb = getelementptr inbounds nuw float, ptr %b, i64 %sj
  %stvb = load float, ptr %stpb, align 4
  %str = fsub float %stva, %stvb
  %stpo = getelementptr inbounds nuw float, ptr %o, i64 %sj
  store float %str, ptr %stpo, align 4
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %ml
ml:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw float, ptr %a, i64 %mi
  %mva = load <4 x float>, ptr %mpa, align 4
  %mpb = getelementptr inbounds nuw float, ptr %b, i64 %mi
  %mvb = load <4 x float>, ptr %mpb, align 4
  %mr = fmul <4 x float> %mva, %mvb
  %mpo = getelementptr inbounds nuw float, ptr %o, i64 %mi
  store <4 x float> %mr, ptr %mpo, align 4
  %min = add nuw i64 %mi, 4
  br label %ml
mt:
  %mj = phi i64 [ %nbulk, %ml ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw float, ptr %a, i64 %mj
  %mtva = load float, ptr %mtpa, align 4
  %mtpb = getelementptr inbounds nuw float, ptr %b, i64 %mj
  %mtvb = load float, ptr %mtpb, align 4
  %mtr = fmul float %mtva, %mtvb
  %mtpo = getelementptr inbounds nuw float, ptr %o, i64 %mj
  store float %mtr, ptr %mtpo, align 4
  %mjn = add nuw i64 %mj, 1
  br label %mt
dh:
  br label %dl
dl:
  %di = phi i64 [ 0, %dh ], [ %din, %db ]
  %dc = icmp ult i64 %di, %nbulk
  br i1 %dc, label %db, label %dt
db:
  %dpa = getelementptr inbounds nuw float, ptr %a, i64 %di
  %dva = load <4 x float>, ptr %dpa, align 4
  %dpb = getelementptr inbounds nuw float, ptr %b, i64 %di
  %dvb = load <4 x float>, ptr %dpb, align 4
  %dr = fdiv <4 x float> %dva, %dvb
  %dpo = getelementptr inbounds nuw float, ptr %o, i64 %di
  store <4 x float> %dr, ptr %dpo, align 4
  %din = add nuw i64 %di, 4
  br label %dl
dt:
  %dj = phi i64 [ %nbulk, %dl ], [ %djn, %dtb ]
  %dtc = icmp ult i64 %dj, %n
  br i1 %dtc, label %dtb, label %done
dtb:
  %dtpa = getelementptr inbounds nuw float, ptr %a, i64 %dj
  %dtva = load float, ptr %dtpa, align 4
  %dtpb = getelementptr inbounds nuw float, ptr %b, i64 %dj
  %dtvb = load float, ptr %dtpb, align 4
  %dtr = fdiv float %dtva, %dtvb
  %dtpo = getelementptr inbounds nuw float, ptr %o, i64 %dj
  store float %dtr, ptr %dtpo, align 4
  %djn = add nuw i64 %dj, 1
  br label %dt
rh:
  br label %rl
rl:
  %ri = phi i64 [ 0, %rh ], [ %rin, %rb ]
  %rc = icmp ult i64 %ri, %nbulk
  br i1 %rc, label %rb, label %rt
rb:
  %rpa = getelementptr inbounds nuw float, ptr %a, i64 %ri
  %rva = load <4 x float>, ptr %rpa, align 4
  %rpb = getelementptr inbounds nuw float, ptr %b, i64 %ri
  %rvb = load <4 x float>, ptr %rpb, align 4
  %rr = frem <4 x float> %rva, %rvb
  %rpo = getelementptr inbounds nuw float, ptr %o, i64 %ri
  store <4 x float> %rr, ptr %rpo, align 4
  %rin = add nuw i64 %ri, 4
  br label %rl
rt:
  %rj = phi i64 [ %nbulk, %rl ], [ %rjn, %rtb ]
  %rtc = icmp ult i64 %rj, %n
  br i1 %rtc, label %rtb, label %done
rtb:
  %rtpa = getelementptr inbounds nuw float, ptr %a, i64 %rj
  %rtva = load float, ptr %rtpa, align 4
  %rtpb = getelementptr inbounds nuw float, ptr %b, i64 %rj
  %rtvb = load float, ptr %rtpb, align 4
  %rtr = frem float %rtva, %rtvb
  %rtpo = getelementptr inbounds nuw float, ptr %o, i64 %rj
  store float %rtr, ptr %rtpo, align 4
  %rjn = add nuw i64 %rj, 1
  br label %rt
done:
  ret void
}

define internal void @a_bin_f64(i32 %op, ptr %a, ptr %b, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -2
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh
                                i32 3, label %dh  i32 4, label %rh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw double, ptr %a, i64 %ai
  %ava = load <2 x double>, ptr %apa, align 8
  %apb = getelementptr inbounds nuw double, ptr %b, i64 %ai
  %avb = load <2 x double>, ptr %apb, align 8
  %ar = fadd <2 x double> %ava, %avb
  %apo = getelementptr inbounds nuw double, ptr %o, i64 %ai
  store <2 x double> %ar, ptr %apo, align 8
  %ain = add nuw i64 %ai, 2
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw double, ptr %a, i64 %aj
  %atva = load double, ptr %atpa, align 8
  %atpb = getelementptr inbounds nuw double, ptr %b, i64 %aj
  %atvb = load double, ptr %atpb, align 8
  %atr = fadd double %atva, %atvb
  %atpo = getelementptr inbounds nuw double, ptr %o, i64 %aj
  store double %atr, ptr %atpo, align 8
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw double, ptr %a, i64 %si
  %sva = load <2 x double>, ptr %spa, align 8
  %spb = getelementptr inbounds nuw double, ptr %b, i64 %si
  %svb = load <2 x double>, ptr %spb, align 8
  %sr = fsub <2 x double> %sva, %svb
  %spo = getelementptr inbounds nuw double, ptr %o, i64 %si
  store <2 x double> %sr, ptr %spo, align 8
  %sin = add nuw i64 %si, 2
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw double, ptr %a, i64 %sj
  %stva = load double, ptr %stpa, align 8
  %stpb = getelementptr inbounds nuw double, ptr %b, i64 %sj
  %stvb = load double, ptr %stpb, align 8
  %str = fsub double %stva, %stvb
  %stpo = getelementptr inbounds nuw double, ptr %o, i64 %sj
  store double %str, ptr %stpo, align 8
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %ml
ml:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw double, ptr %a, i64 %mi
  %mva = load <2 x double>, ptr %mpa, align 8
  %mpb = getelementptr inbounds nuw double, ptr %b, i64 %mi
  %mvb = load <2 x double>, ptr %mpb, align 8
  %mr = fmul <2 x double> %mva, %mvb
  %mpo = getelementptr inbounds nuw double, ptr %o, i64 %mi
  store <2 x double> %mr, ptr %mpo, align 8
  %min = add nuw i64 %mi, 2
  br label %ml
mt:
  %mj = phi i64 [ %nbulk, %ml ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw double, ptr %a, i64 %mj
  %mtva = load double, ptr %mtpa, align 8
  %mtpb = getelementptr inbounds nuw double, ptr %b, i64 %mj
  %mtvb = load double, ptr %mtpb, align 8
  %mtr = fmul double %mtva, %mtvb
  %mtpo = getelementptr inbounds nuw double, ptr %o, i64 %mj
  store double %mtr, ptr %mtpo, align 8
  %mjn = add nuw i64 %mj, 1
  br label %mt
dh:
  br label %dl
dl:
  %di = phi i64 [ 0, %dh ], [ %din, %db ]
  %dc = icmp ult i64 %di, %nbulk
  br i1 %dc, label %db, label %dt
db:
  %dpa = getelementptr inbounds nuw double, ptr %a, i64 %di
  %dva = load <2 x double>, ptr %dpa, align 8
  %dpb = getelementptr inbounds nuw double, ptr %b, i64 %di
  %dvb = load <2 x double>, ptr %dpb, align 8
  %dr = fdiv <2 x double> %dva, %dvb
  %dpo = getelementptr inbounds nuw double, ptr %o, i64 %di
  store <2 x double> %dr, ptr %dpo, align 8
  %din = add nuw i64 %di, 2
  br label %dl
dt:
  %dj = phi i64 [ %nbulk, %dl ], [ %djn, %dtb ]
  %dtc = icmp ult i64 %dj, %n
  br i1 %dtc, label %dtb, label %done
dtb:
  %dtpa = getelementptr inbounds nuw double, ptr %a, i64 %dj
  %dtva = load double, ptr %dtpa, align 8
  %dtpb = getelementptr inbounds nuw double, ptr %b, i64 %dj
  %dtvb = load double, ptr %dtpb, align 8
  %dtr = fdiv double %dtva, %dtvb
  %dtpo = getelementptr inbounds nuw double, ptr %o, i64 %dj
  store double %dtr, ptr %dtpo, align 8
  %djn = add nuw i64 %dj, 1
  br label %dt
rh:
  br label %rl
rl:
  %ri = phi i64 [ 0, %rh ], [ %rin, %rb ]
  %rc = icmp ult i64 %ri, %nbulk
  br i1 %rc, label %rb, label %rt
rb:
  %rpa = getelementptr inbounds nuw double, ptr %a, i64 %ri
  %rva = load <2 x double>, ptr %rpa, align 8
  %rpb = getelementptr inbounds nuw double, ptr %b, i64 %ri
  %rvb = load <2 x double>, ptr %rpb, align 8
  %rr = frem <2 x double> %rva, %rvb
  %rpo = getelementptr inbounds nuw double, ptr %o, i64 %ri
  store <2 x double> %rr, ptr %rpo, align 8
  %rin = add nuw i64 %ri, 2
  br label %rl
rt:
  %rj = phi i64 [ %nbulk, %rl ], [ %rjn, %rtb ]
  %rtc = icmp ult i64 %rj, %n
  br i1 %rtc, label %rtb, label %done
rtb:
  %rtpa = getelementptr inbounds nuw double, ptr %a, i64 %rj
  %rtva = load double, ptr %rtpa, align 8
  %rtpb = getelementptr inbounds nuw double, ptr %b, i64 %rj
  %rtvb = load double, ptr %rtpb, align 8
  %rtr = frem double %rtva, %rtvb
  %rtpo = getelementptr inbounds nuw double, ptr %o, i64 %rj
  store double %rtr, ptr %rtpo, align 8
  %rjn = add nuw i64 %rj, 1
  br label %rt
done:
  ret void
}

; ---------------------------------------------------------------------------
; homogeneous integer div/rem — scalar branchless-guarded, builds validity.
; op: 3=div, 4=rem. Marks a lane null on zero divisor / INT_MIN,-1 overflow.
; ---------------------------------------------------------------------------
define internal void @a_intdiv_i32(i32 %op, ptr %a, ptr %b, ptr %out, i64 %n) #1 {
entry:
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 24
  %bv = load ptr, ptr %bvp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  %isrem = icmp eq i32 %op, 4
  %bm = call ptr @a_bm_alloc(i64 %n)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %nulls = phi i64 [ 0, %entry ], [ %nn, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %fin
body:
  %pa = getelementptr inbounds i32, ptr %av, i64 %i
  %x = load i32, ptr %pa, align 4
  %pb = getelementptr inbounds i32, ptr %bv, i64 %i
  %y = load i32, ptr %pb, align 4
  %zero = icmp eq i32 %y, 0
  %xmin = icmp eq i32 %x, -2147483648
  %ym1 = icmp eq i32 %y, -1
  %ovf = and i1 %xmin, %ym1
  %bad = or i1 %zero, %ovf
  %d = select i1 %bad, i32 1, i32 %y
  %q = sdiv i32 %x, %d
  %rr = srem i32 %x, %d
  %res = select i1 %isrem, i32 %rr, i32 %q
  %po = getelementptr inbounds i32, ptr %ov, i64 %i
  store i32 %res, ptr %po, align 4
  %va = call i1 @a_valid_at(ptr %a, i64 %i)
  %vb = call i1 @a_valid_at(ptr %b, i64 %i)
  %vin = and i1 %va, %vb
  %notbad = xor i1 %bad, true
  %lanevalid = and i1 %vin, %notbad
  br i1 %lanevalid, label %cont, label %mknull
mknull:
  call void @a_bm_clear(ptr %bm, i64 %i)
  %nn1 = add i64 %nulls, 1
  br label %cont
cont:
  %nn = phi i64 [ %nulls, %body ], [ %nn1, %mknull ]
  %in = add nuw i64 %i, 1
  br label %loop
fin:
  %z = icmp eq i64 %nulls, 0
  br i1 %z, label %freebm, label %keepbm
freebm:
  call void @free(ptr %bm)
  ret void
keepbm:
  %obmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %obmp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nulls, ptr %oncp, align 8
  ret void
}

define internal void @a_intdiv_i64(i32 %op, ptr %a, ptr %b, ptr %out, i64 %n) #1 {
entry:
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 24
  %bv = load ptr, ptr %bvp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  %isrem = icmp eq i32 %op, 4
  %bm = call ptr @a_bm_alloc(i64 %n)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %nulls = phi i64 [ 0, %entry ], [ %nn, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %fin
body:
  %pa = getelementptr inbounds i64, ptr %av, i64 %i
  %x = load i64, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %bv, i64 %i
  %y = load i64, ptr %pb, align 8
  %zero = icmp eq i64 %y, 0
  %xmin = icmp eq i64 %x, -9223372036854775808
  %ym1 = icmp eq i64 %y, -1
  %ovf = and i1 %xmin, %ym1
  %bad = or i1 %zero, %ovf
  %d = select i1 %bad, i64 1, i64 %y
  %q = sdiv i64 %x, %d
  %rr = srem i64 %x, %d
  %res = select i1 %isrem, i64 %rr, i64 %q
  %po = getelementptr inbounds i64, ptr %ov, i64 %i
  store i64 %res, ptr %po, align 8
  %va = call i1 @a_valid_at(ptr %a, i64 %i)
  %vb = call i1 @a_valid_at(ptr %b, i64 %i)
  %vin = and i1 %va, %vb
  %notbad = xor i1 %bad, true
  %lanevalid = and i1 %vin, %notbad
  br i1 %lanevalid, label %cont, label %mknull
mknull:
  call void @a_bm_clear(ptr %bm, i64 %i)
  %nn1 = add i64 %nulls, 1
  br label %cont
cont:
  %nn = phi i64 [ %nulls, %body ], [ %nn1, %mknull ]
  %in = add nuw i64 %i, 1
  br label %loop
fin:
  %z = icmp eq i64 %nulls, 0
  br i1 %z, label %freebm, label %keepbm
freebm:
  call void @free(ptr %bm)
  ret void
keepbm:
  %obmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %bm, ptr %obmp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %nulls, ptr %oncp, align 8
  ret void
}

; ---------------------------------------------------------------------------
; heterogeneous (mixed-dtype) COLD path -> F64. Scalar double loop. Caller
; combines validity afterwards (float div/rem by zero => IEEE, no null).
; ---------------------------------------------------------------------------
define internal void @a_mixed_f64(i32 %op, ptr %a, ptr %b, ptr %out, i64 %n) #1 {
entry:
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 24
  %bv = load ptr, ptr %bvp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %c = icmp ult i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %xa = call double @a_load_f64(ptr %av, i32 %da, i64 %i)
  %xb = call double @a_load_f64(ptr %bv, i32 %db, i64 %i)
  switch i32 %op, label %o.add [ i32 1, label %o.sub  i32 2, label %o.mul
                                 i32 3, label %o.div  i32 4, label %o.rem ]
o.add:
  %radd = fadd double %xa, %xb
  br label %store
o.sub:
  %rsub = fsub double %xa, %xb
  br label %store
o.mul:
  %rmul = fmul double %xa, %xb
  br label %store
o.div:
  %rdiv = fdiv double %xa, %xb
  br label %store
o.rem:
  %rrem = frem double %xa, %xb
  br label %store
store:
  %r = phi double [ %radd, %o.add ], [ %rsub, %o.sub ], [ %rmul, %o.mul ], [ %rdiv, %o.div ], [ %rrem, %o.rem ]
  %po = getelementptr inbounds double, ptr %ov, i64 %i
  store double %r, ptr %po, align 8
  br label %cont
cont:
  %in = add nuw i64 %i, 1
  br label %loop
done:
  ret void
}

; ---------------------------------------------------------------------------
; unary neg(0)/abs(1) per dtype (SIMD; validity copied by caller).
; ---------------------------------------------------------------------------
define internal void @a_un_i32(i32 %op, ptr %a, ptr %o, i64 %n) #1 {
entry:
  %isabs = icmp eq i32 %op, 1
  %nbulk = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %nbulk
  br i1 %c, label %vb, label %th
vb:
  %pa = getelementptr inbounds nuw i32, ptr %a, i64 %i
  %va = load <4 x i32>, ptr %pa, align 4
  %neg = sub <4 x i32> zeroinitializer, %va
  %abs = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %va, i1 false)
  %r = select i1 %isabs, <4 x i32> %abs, <4 x i32> %neg
  %po = getelementptr inbounds nuw i32, ptr %o, i64 %i
  store <4 x i32> %r, ptr %po, align 4
  %in = add nuw i64 %i, 4
  br label %vh
th:
  %j = phi i64 [ %nbulk, %vh ], [ %jn, %tb ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tb, label %done
tb:
  %tpa = getelementptr inbounds nuw i32, ptr %a, i64 %j
  %tva = load i32, ptr %tpa, align 4
  %tneg = sub i32 0, %tva
  %tabs = call i32 @llvm.abs.i32(i32 %tva, i1 false)
  %tr = select i1 %isabs, i32 %tabs, i32 %tneg
  %tpo = getelementptr inbounds nuw i32, ptr %o, i64 %j
  store i32 %tr, ptr %tpo, align 4
  %jn = add nuw i64 %j, 1
  br label %th
done:
  ret void
}

define internal void @a_un_i64(i32 %op, ptr %a, ptr %o, i64 %n) #1 {
entry:
  %isabs = icmp eq i32 %op, 1
  %nbulk = and i64 %n, -2
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %nbulk
  br i1 %c, label %vb, label %th
vb:
  %pa = getelementptr inbounds nuw i64, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %pa, align 8
  %neg = sub <2 x i64> zeroinitializer, %va
  %abs = call <2 x i64> @llvm.abs.v2i64(<2 x i64> %va, i1 false)
  %r = select i1 %isabs, <2 x i64> %abs, <2 x i64> %neg
  %po = getelementptr inbounds nuw i64, ptr %o, i64 %i
  store <2 x i64> %r, ptr %po, align 8
  %in = add nuw i64 %i, 2
  br label %vh
th:
  %j = phi i64 [ %nbulk, %vh ], [ %jn, %tb ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tb, label %done
tb:
  %tpa = getelementptr inbounds nuw i64, ptr %a, i64 %j
  %tva = load i64, ptr %tpa, align 8
  %tneg = sub i64 0, %tva
  %tabs = call i64 @llvm.abs.i64(i64 %tva, i1 false)
  %tr = select i1 %isabs, i64 %tabs, i64 %tneg
  %tpo = getelementptr inbounds nuw i64, ptr %o, i64 %j
  store i64 %tr, ptr %tpo, align 8
  %jn = add nuw i64 %j, 1
  br label %th
done:
  ret void
}

define internal void @a_un_f32(i32 %op, ptr %a, ptr %o, i64 %n) #1 {
entry:
  %isabs = icmp eq i32 %op, 1
  %nbulk = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %nbulk
  br i1 %c, label %vb, label %th
vb:
  %pa = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va = load <4 x float>, ptr %pa, align 4
  %neg = fneg <4 x float> %va
  %abs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %va)
  %r = select i1 %isabs, <4 x float> %abs, <4 x float> %neg
  %po = getelementptr inbounds nuw float, ptr %o, i64 %i
  store <4 x float> %r, ptr %po, align 4
  %in = add nuw i64 %i, 4
  br label %vh
th:
  %j = phi i64 [ %nbulk, %vh ], [ %jn, %tb ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tb, label %done
tb:
  %tpa = getelementptr inbounds nuw float, ptr %a, i64 %j
  %tva = load float, ptr %tpa, align 4
  %tneg = fneg float %tva
  %tabs = call float @llvm.fabs.f32(float %tva)
  %tr = select i1 %isabs, float %tabs, float %tneg
  %tpo = getelementptr inbounds nuw float, ptr %o, i64 %j
  store float %tr, ptr %tpo, align 4
  %jn = add nuw i64 %j, 1
  br label %th
done:
  ret void
}

define internal void @a_un_f64(i32 %op, ptr %a, ptr %o, i64 %n) #1 {
entry:
  %isabs = icmp eq i32 %op, 1
  %nbulk = and i64 %n, -2
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %nbulk
  br i1 %c, label %vb, label %th
vb:
  %pa = getelementptr inbounds nuw double, ptr %a, i64 %i
  %va = load <2 x double>, ptr %pa, align 8
  %neg = fneg <2 x double> %va
  %abs = call <2 x double> @llvm.fabs.v2f64(<2 x double> %va)
  %r = select i1 %isabs, <2 x double> %abs, <2 x double> %neg
  %po = getelementptr inbounds nuw double, ptr %o, i64 %i
  store <2 x double> %r, ptr %po, align 8
  %in = add nuw i64 %i, 2
  br label %vh
th:
  %j = phi i64 [ %nbulk, %vh ], [ %jn, %tb ]
  %tc = icmp ult i64 %j, %n
  br i1 %tc, label %tb, label %done
tb:
  %tpa = getelementptr inbounds nuw double, ptr %a, i64 %j
  %tva = load double, ptr %tpa, align 8
  %tneg = fneg double %tva
  %tabs = call double @llvm.fabs.f64(double %tva)
  %tr = select i1 %isabs, double %tabs, double %tneg
  %tpo = getelementptr inbounds nuw double, ptr %o, i64 %j
  store double %tr, ptr %tpo, align 8
  %jn = add nuw i64 %j, 1
  br label %th
done:
  ret void
}

; ---------------------------------------------------------------------------
; scalar broadcast kernels. int: add/sub/mul vector-splat; div/rem scalar
; (caller guarantees sv != 0, guards INT_MIN/-1). float: add/sub/mul/div splat.
; ---------------------------------------------------------------------------
define internal void @a_scal_i32(i32 %op, ptr %a, i32 %sv, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -4
  %e0 = insertelement <4 x i32> poison, i32 %sv, i64 0
  %sp = shufflevector <4 x i32> %e0, <4 x i32> poison, <4 x i32> zeroinitializer
  switch i32 %op, label %divh [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh  i32 4, label %divh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw i32, ptr %a, i64 %ai
  %ava = load <4 x i32>, ptr %apa, align 4
  %ar = add <4 x i32> %ava, %sp
  %apo = getelementptr inbounds nuw i32, ptr %o, i64 %ai
  store <4 x i32> %ar, ptr %apo, align 4
  %ain = add nuw i64 %ai, 4
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw i32, ptr %a, i64 %aj
  %atva = load i32, ptr %atpa, align 4
  %atr = add i32 %atva, %sv
  %atpo = getelementptr inbounds nuw i32, ptr %o, i64 %aj
  store i32 %atr, ptr %atpo, align 4
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw i32, ptr %a, i64 %si
  %sva = load <4 x i32>, ptr %spa, align 4
  %sr = sub <4 x i32> %sva, %sp
  %spo = getelementptr inbounds nuw i32, ptr %o, i64 %si
  store <4 x i32> %sr, ptr %spo, align 4
  %sin = add nuw i64 %si, 4
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw i32, ptr %a, i64 %sj
  %stva = load i32, ptr %stpa, align 4
  %str = sub i32 %stva, %sv
  %stpo = getelementptr inbounds nuw i32, ptr %o, i64 %sj
  store i32 %str, ptr %stpo, align 4
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %mlp
mlp:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw i32, ptr %a, i64 %mi
  %mva = load <4 x i32>, ptr %mpa, align 4
  %mr = mul <4 x i32> %mva, %sp
  %mpo = getelementptr inbounds nuw i32, ptr %o, i64 %mi
  store <4 x i32> %mr, ptr %mpo, align 4
  %min = add nuw i64 %mi, 4
  br label %mlp
mt:
  %mj = phi i64 [ %nbulk, %mlp ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw i32, ptr %a, i64 %mj
  %mtva = load i32, ptr %mtpa, align 4
  %mtr = mul i32 %mtva, %sv
  %mtpo = getelementptr inbounds nuw i32, ptr %o, i64 %mj
  store i32 %mtr, ptr %mtpo, align 4
  %mjn = add nuw i64 %mj, 1
  br label %mt
divh:
  %isrem = icmp eq i32 %op, 4
  %svm1 = icmp eq i32 %sv, -1
  br label %dl
dl:
  %di = phi i64 [ 0, %divh ], [ %din, %dbdy ]
  %dc = icmp ult i64 %di, %n
  br i1 %dc, label %dbdy, label %done
dbdy:
  %dpa = getelementptr inbounds nuw i32, ptr %a, i64 %di
  %dx = load i32, ptr %dpa, align 4
  %dxmin = icmp eq i32 %dx, -2147483648
  %dovf = and i1 %svm1, %dxmin
  %dd = select i1 %dovf, i32 1, i32 %sv
  %dq = sdiv i32 %dx, %dd
  %drr = srem i32 %dx, %dd
  %dqf = select i1 %dovf, i32 -2147483648, i32 %dq
  %drf = select i1 %dovf, i32 0, i32 %drr
  %dres = select i1 %isrem, i32 %drf, i32 %dqf
  %dpo = getelementptr inbounds nuw i32, ptr %o, i64 %di
  store i32 %dres, ptr %dpo, align 4
  %din = add nuw i64 %di, 1
  br label %dl
done:
  ret void
}

define internal void @a_scal_i64(i32 %op, ptr %a, i64 %sv, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -2
  %e0 = insertelement <2 x i64> poison, i64 %sv, i64 0
  %sp = shufflevector <2 x i64> %e0, <2 x i64> poison, <2 x i32> zeroinitializer
  switch i32 %op, label %divh [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh  i32 4, label %divh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw i64, ptr %a, i64 %ai
  %ava = load <2 x i64>, ptr %apa, align 8
  %ar = add <2 x i64> %ava, %sp
  %apo = getelementptr inbounds nuw i64, ptr %o, i64 %ai
  store <2 x i64> %ar, ptr %apo, align 8
  %ain = add nuw i64 %ai, 2
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw i64, ptr %a, i64 %aj
  %atva = load i64, ptr %atpa, align 8
  %atr = add i64 %atva, %sv
  %atpo = getelementptr inbounds nuw i64, ptr %o, i64 %aj
  store i64 %atr, ptr %atpo, align 8
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw i64, ptr %a, i64 %si
  %sva = load <2 x i64>, ptr %spa, align 8
  %sr = sub <2 x i64> %sva, %sp
  %spo = getelementptr inbounds nuw i64, ptr %o, i64 %si
  store <2 x i64> %sr, ptr %spo, align 8
  %sin = add nuw i64 %si, 2
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw i64, ptr %a, i64 %sj
  %stva = load i64, ptr %stpa, align 8
  %str = sub i64 %stva, %sv
  %stpo = getelementptr inbounds nuw i64, ptr %o, i64 %sj
  store i64 %str, ptr %stpo, align 8
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %mlp
mlp:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw i64, ptr %a, i64 %mi
  %mva = load <2 x i64>, ptr %mpa, align 8
  %mr = mul <2 x i64> %mva, %sp
  %mpo = getelementptr inbounds nuw i64, ptr %o, i64 %mi
  store <2 x i64> %mr, ptr %mpo, align 8
  %min = add nuw i64 %mi, 2
  br label %mlp
mt:
  %mj = phi i64 [ %nbulk, %mlp ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw i64, ptr %a, i64 %mj
  %mtva = load i64, ptr %mtpa, align 8
  %mtr = mul i64 %mtva, %sv
  %mtpo = getelementptr inbounds nuw i64, ptr %o, i64 %mj
  store i64 %mtr, ptr %mtpo, align 8
  %mjn = add nuw i64 %mj, 1
  br label %mt
divh:
  %isrem = icmp eq i32 %op, 4
  %svm1 = icmp eq i64 %sv, -1
  br label %dl
dl:
  %di = phi i64 [ 0, %divh ], [ %din, %dbdy ]
  %dc = icmp ult i64 %di, %n
  br i1 %dc, label %dbdy, label %done
dbdy:
  %dpa = getelementptr inbounds nuw i64, ptr %a, i64 %di
  %dx = load i64, ptr %dpa, align 8
  %dxmin = icmp eq i64 %dx, -9223372036854775808
  %dovf = and i1 %svm1, %dxmin
  %dd = select i1 %dovf, i64 1, i64 %sv
  %dq = sdiv i64 %dx, %dd
  %drr = srem i64 %dx, %dd
  %dqf = select i1 %dovf, i64 -9223372036854775808, i64 %dq
  %drf = select i1 %dovf, i64 0, i64 %drr
  %dres = select i1 %isrem, i64 %drf, i64 %dqf
  %dpo = getelementptr inbounds nuw i64, ptr %o, i64 %di
  store i64 %dres, ptr %dpo, align 8
  %din = add nuw i64 %di, 1
  br label %dl
done:
  ret void
}

define internal void @a_scal_f32(i32 %op, ptr %a, float %sv, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -4
  %e0 = insertelement <4 x float> poison, float %sv, i64 0
  %sp = shufflevector <4 x float> %e0, <4 x float> poison, <4 x i32> zeroinitializer
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh  i32 3, label %dh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw float, ptr %a, i64 %ai
  %ava = load <4 x float>, ptr %apa, align 4
  %ar = fadd <4 x float> %ava, %sp
  %apo = getelementptr inbounds nuw float, ptr %o, i64 %ai
  store <4 x float> %ar, ptr %apo, align 4
  %ain = add nuw i64 %ai, 4
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw float, ptr %a, i64 %aj
  %atva = load float, ptr %atpa, align 4
  %atr = fadd float %atva, %sv
  %atpo = getelementptr inbounds nuw float, ptr %o, i64 %aj
  store float %atr, ptr %atpo, align 4
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw float, ptr %a, i64 %si
  %sva = load <4 x float>, ptr %spa, align 4
  %sr = fsub <4 x float> %sva, %sp
  %spo = getelementptr inbounds nuw float, ptr %o, i64 %si
  store <4 x float> %sr, ptr %spo, align 4
  %sin = add nuw i64 %si, 4
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw float, ptr %a, i64 %sj
  %stva = load float, ptr %stpa, align 4
  %str = fsub float %stva, %sv
  %stpo = getelementptr inbounds nuw float, ptr %o, i64 %sj
  store float %str, ptr %stpo, align 4
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %mlp
mlp:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw float, ptr %a, i64 %mi
  %mva = load <4 x float>, ptr %mpa, align 4
  %mr = fmul <4 x float> %mva, %sp
  %mpo = getelementptr inbounds nuw float, ptr %o, i64 %mi
  store <4 x float> %mr, ptr %mpo, align 4
  %min = add nuw i64 %mi, 4
  br label %mlp
mt:
  %mj = phi i64 [ %nbulk, %mlp ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw float, ptr %a, i64 %mj
  %mtva = load float, ptr %mtpa, align 4
  %mtr = fmul float %mtva, %sv
  %mtpo = getelementptr inbounds nuw float, ptr %o, i64 %mj
  store float %mtr, ptr %mtpo, align 4
  %mjn = add nuw i64 %mj, 1
  br label %mt
dh:
  br label %dlp
dlp:
  %ddi = phi i64 [ 0, %dh ], [ %ddin, %ddb ]
  %ddc = icmp ult i64 %ddi, %nbulk
  br i1 %ddc, label %ddb, label %ddt
ddb:
  %ddpa = getelementptr inbounds nuw float, ptr %a, i64 %ddi
  %ddva = load <4 x float>, ptr %ddpa, align 4
  %ddr = fdiv <4 x float> %ddva, %sp
  %ddpo = getelementptr inbounds nuw float, ptr %o, i64 %ddi
  store <4 x float> %ddr, ptr %ddpo, align 4
  %ddin = add nuw i64 %ddi, 4
  br label %dlp
ddt:
  %ddj = phi i64 [ %nbulk, %dlp ], [ %ddjn, %ddtb ]
  %ddtc = icmp ult i64 %ddj, %n
  br i1 %ddtc, label %ddtb, label %done
ddtb:
  %ddtpa = getelementptr inbounds nuw float, ptr %a, i64 %ddj
  %ddtva = load float, ptr %ddtpa, align 4
  %ddtr = fdiv float %ddtva, %sv
  %ddtpo = getelementptr inbounds nuw float, ptr %o, i64 %ddj
  store float %ddtr, ptr %ddtpo, align 4
  %ddjn = add nuw i64 %ddj, 1
  br label %ddt
done:
  ret void
}

define internal void @a_scal_f64(i32 %op, ptr %a, double %sv, ptr %o, i64 %n) #1 {
entry:
  %nbulk = and i64 %n, -2
  %e0 = insertelement <2 x double> poison, double %sv, i64 0
  %sp = shufflevector <2 x double> %e0, <2 x double> poison, <2 x i32> zeroinitializer
  switch i32 %op, label %done [ i32 0, label %ah  i32 1, label %sh  i32 2, label %mh  i32 3, label %dh ]
ah:
  br label %al
al:
  %ai = phi i64 [ 0, %ah ], [ %ain, %ab ]
  %ac = icmp ult i64 %ai, %nbulk
  br i1 %ac, label %ab, label %at
ab:
  %apa = getelementptr inbounds nuw double, ptr %a, i64 %ai
  %ava = load <2 x double>, ptr %apa, align 8
  %ar = fadd <2 x double> %ava, %sp
  %apo = getelementptr inbounds nuw double, ptr %o, i64 %ai
  store <2 x double> %ar, ptr %apo, align 8
  %ain = add nuw i64 %ai, 2
  br label %al
at:
  %aj = phi i64 [ %nbulk, %al ], [ %ajn, %atb ]
  %atc = icmp ult i64 %aj, %n
  br i1 %atc, label %atb, label %done
atb:
  %atpa = getelementptr inbounds nuw double, ptr %a, i64 %aj
  %atva = load double, ptr %atpa, align 8
  %atr = fadd double %atva, %sv
  %atpo = getelementptr inbounds nuw double, ptr %o, i64 %aj
  store double %atr, ptr %atpo, align 8
  %ajn = add nuw i64 %aj, 1
  br label %at
sh:
  br label %sl
sl:
  %si = phi i64 [ 0, %sh ], [ %sin, %sb ]
  %sc = icmp ult i64 %si, %nbulk
  br i1 %sc, label %sb, label %stt
sb:
  %spa = getelementptr inbounds nuw double, ptr %a, i64 %si
  %sva = load <2 x double>, ptr %spa, align 8
  %sr = fsub <2 x double> %sva, %sp
  %spo = getelementptr inbounds nuw double, ptr %o, i64 %si
  store <2 x double> %sr, ptr %spo, align 8
  %sin = add nuw i64 %si, 2
  br label %sl
stt:
  %sj = phi i64 [ %nbulk, %sl ], [ %sjn, %stb ]
  %stc = icmp ult i64 %sj, %n
  br i1 %stc, label %stb, label %done
stb:
  %stpa = getelementptr inbounds nuw double, ptr %a, i64 %sj
  %stva = load double, ptr %stpa, align 8
  %str = fsub double %stva, %sv
  %stpo = getelementptr inbounds nuw double, ptr %o, i64 %sj
  store double %str, ptr %stpo, align 8
  %sjn = add nuw i64 %sj, 1
  br label %stt
mh:
  br label %mlp
mlp:
  %mi = phi i64 [ 0, %mh ], [ %min, %mb ]
  %mc = icmp ult i64 %mi, %nbulk
  br i1 %mc, label %mb, label %mt
mb:
  %mpa = getelementptr inbounds nuw double, ptr %a, i64 %mi
  %mva = load <2 x double>, ptr %mpa, align 8
  %mr = fmul <2 x double> %mva, %sp
  %mpo = getelementptr inbounds nuw double, ptr %o, i64 %mi
  store <2 x double> %mr, ptr %mpo, align 8
  %min = add nuw i64 %mi, 2
  br label %mlp
mt:
  %mj = phi i64 [ %nbulk, %mlp ], [ %mjn, %mtb ]
  %mtc = icmp ult i64 %mj, %n
  br i1 %mtc, label %mtb, label %done
mtb:
  %mtpa = getelementptr inbounds nuw double, ptr %a, i64 %mj
  %mtva = load double, ptr %mtpa, align 8
  %mtr = fmul double %mtva, %sv
  %mtpo = getelementptr inbounds nuw double, ptr %o, i64 %mj
  store double %mtr, ptr %mtpo, align 8
  %mjn = add nuw i64 %mj, 1
  br label %mt
dh:
  br label %dlp
dlp:
  %ddi = phi i64 [ 0, %dh ], [ %ddin, %ddb ]
  %ddc = icmp ult i64 %ddi, %nbulk
  br i1 %ddc, label %ddb, label %ddt
ddb:
  %ddpa = getelementptr inbounds nuw double, ptr %a, i64 %ddi
  %ddva = load <2 x double>, ptr %ddpa, align 8
  %ddr = fdiv <2 x double> %ddva, %sp
  %ddpo = getelementptr inbounds nuw double, ptr %o, i64 %ddi
  store <2 x double> %ddr, ptr %ddpo, align 8
  %ddin = add nuw i64 %ddi, 2
  br label %dlp
ddt:
  %ddj = phi i64 [ %nbulk, %dlp ], [ %ddjn, %ddtb ]
  %ddtc = icmp ult i64 %ddj, %n
  br i1 %ddtc, label %ddtb, label %done
ddtb:
  %ddtpa = getelementptr inbounds nuw double, ptr %a, i64 %ddj
  %ddtva = load double, ptr %ddtpa, align 8
  %ddtr = fdiv double %ddtva, %sv
  %ddtpo = getelementptr inbounds nuw double, ptr %o, i64 %ddj
  store double %ddtr, ptr %ddtpo, align 8
  %ddjn = add nuw i64 %ddj, 1
  br label %ddt
done:
  ret void
}

; ---------------------------------------------------------------------------
; internal drivers
; ---------------------------------------------------------------------------

; validate + dispatch a binary col-col op. op: 0 add,1 sub,2 mul,3 div,4 rem.
define internal ptr @a_binary(i32 %op, ptr %a, ptr %b) #1 {
entry:
  %an = icmp eq ptr %a, null
  %bn = icmp eq ptr %b, null
  %anyn = or i1 %an, %bn
  br i1 %anyn, label %fail, label %chk
chk:
  %da = load i32, ptr %a, align 8
  %db = load i32, ptr %b, align 8
  %dabad = icmp ugt i32 %da, 3
  %dbbad = icmp ugt i32 %db, 3
  %anybad = or i1 %dabad, %dbbad
  %alp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %alp, align 8
  %blp = getelementptr inbounds i8, ptr %b, i64 8
  %lb = load i64, ptr %blp, align 8
  %lne = icmp ne i64 %la, %lb
  %argbad = or i1 %anybad, %lne
  br i1 %argbad, label %fail, label %ok
ok:
  %homog = icmp eq i32 %da, %db
  %R = select i1 %homog, i32 %da, i32 3
  %out = call ptr @universe_dataframe_series_new(i32 %R, i64 %la)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %vals
vals:
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %bvp = getelementptr inbounds i8, ptr %b, i64 24
  %bv = load ptr, ptr %bvp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  br i1 %homog, label %homogd, label %mixed
mixed:
  call void @a_mixed_f64(i32 %op, ptr %a, ptr %b, ptr %out, i64 %la)
  call void @a_combine_validity(ptr %out, ptr %a, ptr %b, i64 %la)
  ret ptr %out
homogd:
  %isdivrem = icmp uge i32 %op, 3
  switch i32 %da, label %h.f64 [ i32 0, label %h.i32  i32 1, label %h.i64  i32 2, label %h.f32 ]
h.i32:
  br i1 %isdivrem, label %h.i32.dr, label %h.i32.asm
h.i32.dr:
  call void @a_intdiv_i32(i32 %op, ptr %a, ptr %b, ptr %out, i64 %la)
  ret ptr %out
h.i32.asm:
  call void @a_bin_i32(i32 %op, ptr %av, ptr %bv, ptr %ov, i64 %la)
  br label %comb
h.i64:
  br i1 %isdivrem, label %h.i64.dr, label %h.i64.asm
h.i64.dr:
  call void @a_intdiv_i64(i32 %op, ptr %a, ptr %b, ptr %out, i64 %la)
  ret ptr %out
h.i64.asm:
  call void @a_bin_i64(i32 %op, ptr %av, ptr %bv, ptr %ov, i64 %la)
  br label %comb
h.f32:
  call void @a_bin_f32(i32 %op, ptr %av, ptr %bv, ptr %ov, i64 %la)
  br label %comb
h.f64:
  call void @a_bin_f64(i32 %op, ptr %av, ptr %bv, ptr %ov, i64 %la)
  br label %comb
comb:
  call void @a_combine_validity(ptr %out, ptr %a, ptr %b, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

; validate + dispatch a unary op. op: 0 neg, 1 abs.
define internal ptr @a_unary(i32 %op, ptr %a) #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %chk
chk:
  %da = load i32, ptr %a, align 8
  %bad = icmp ugt i32 %da, 3
  br i1 %bad, label %fail, label %ok
ok:
  %lp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %lp, align 8
  %out = call ptr @universe_dataframe_series_new(i32 %da, i64 %la)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %vals
vals:
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  switch i32 %da, label %d3 [ i32 0, label %d0  i32 1, label %d1  i32 2, label %d2 ]
d0:
  call void @a_un_i32(i32 %op, ptr %av, ptr %ov, i64 %la)
  br label %done
d1:
  call void @a_un_i64(i32 %op, ptr %av, ptr %ov, i64 %la)
  br label %done
d2:
  call void @a_un_f32(i32 %op, ptr %av, ptr %ov, i64 %la)
  br label %done
d3:
  call void @a_un_f64(i32 %op, ptr %av, ptr %ov, i64 %la)
  br label %done
done:
  call void @a_copy_validity(ptr %out, ptr %a, i64 %la)
  ret ptr %out
fail:
  ret ptr null
}

; validate + dispatch a scalar-broadcast op. op: 0 add,1 sub,2 mul,3 div.
define internal ptr @a_scalar(i32 %op, ptr %a, i64 %ival, double %fval) #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %chk
chk:
  %da = load i32, ptr %a, align 8
  %bad = icmp ugt i32 %da, 3
  br i1 %bad, label %fail, label %ok
ok:
  %lp = getelementptr inbounds i8, ptr %a, i64 8
  %la = load i64, ptr %lp, align 8
  %out = call ptr @universe_dataframe_series_new(i32 %da, i64 %la)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %vals
vals:
  %avp = getelementptr inbounds i8, ptr %a, i64 24
  %av = load ptr, ptr %avp, align 8
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %ov = load ptr, ptr %ovp, align 8
  switch i32 %da, label %d3 [ i32 0, label %d0  i32 1, label %d1  i32 2, label %d2 ]
d0:
  %sv32 = trunc i64 %ival to i32
  %d0iszero = icmp eq i32 %sv32, 0
  %d0isdiv = icmp uge i32 %op, 3
  %d0allnull = and i1 %d0iszero, %d0isdiv
  br i1 %d0allnull, label %allnull, label %d0go
d0go:
  call void @a_scal_i32(i32 %op, ptr %av, i32 %sv32, ptr %ov, i64 %la)
  br label %copyv
d1:
  %d1iszero = icmp eq i64 %ival, 0
  %d1isdiv = icmp uge i32 %op, 3
  %d1allnull = and i1 %d1iszero, %d1isdiv
  br i1 %d1allnull, label %allnull, label %d1go
d1go:
  call void @a_scal_i64(i32 %op, ptr %av, i64 %ival, ptr %ov, i64 %la)
  br label %copyv
d2:
  %svf = fptrunc double %fval to float
  call void @a_scal_f32(i32 %op, ptr %av, float %svf, ptr %ov, i64 %la)
  br label %copyv
d3:
  call void @a_scal_f64(i32 %op, ptr %av, double %fval, ptr %ov, i64 %la)
  br label %copyv
copyv:
  call void @a_copy_validity(ptr %out, ptr %a, i64 %la)
  ret ptr %out
allnull:
  %anbm = call ptr @a_bm_alloc(i64 %la)
  %t = add i64 %la, 7
  %nb = lshr i64 %t, 3
  call void @llvm.memset.p0.i64(ptr %anbm, i8 0, i64 %nb, i1 false)
  %anbmp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %anbm, ptr %anbmp, align 8
  %anncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %la, ptr %anncp, align 8
  ret ptr %out
fail:
  ret ptr null
}

; horizontal min(0)/max(1) across all numeric columns -> F64 Series (height).
define internal ptr @a_horizontal(i32 %op, ptr %df) #1 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %run
fail:
  ret ptr null
run:
  %ismax = icmp eq i32 %op, 1
  %h = call i64 @universe_dataframe_height(ptr %df)
  %w = call i64 @universe_dataframe_width(ptr %df)
  %out = call ptr @universe_dataframe_series_new(i32 3, i64 %h)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %init
init:
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %outv = load ptr, ptr %ovp, align 8
  %initv = select i1 %ismax, double 0xFFF0000000000000, double 0x7FF0000000000000
  %cbytes = shl i64 %h, 3
  %cbz = icmp eq i64 %cbytes, 0
  %cba = select i1 %cbz, i64 8, i64 %cbytes
  %cntbuf = call ptr @malloc(i64 %cba)
  %cbn = icmp eq ptr %cntbuf, null
  br i1 %cbn, label %fail, label %fillinit
fillinit:
  call void @llvm.memset.p0.i64(ptr %cntbuf, i8 0, i64 %cba, i1 false)
  br label %ih
ih:
  %ii = phi i64 [ 0, %fillinit ], [ %iin, %ib ]
  %ic = icmp ult i64 %ii, %h
  br i1 %ic, label %ib, label %cloop
ib:
  %ip = getelementptr inbounds double, ptr %outv, i64 %ii
  store double %initv, ptr %ip, align 8
  %iin = add nuw i64 %ii, 1
  br label %ih
cloop:
  %c = phi i64 [ 0, %ih ], [ %cn, %cdone ]
  %cdoneb = icmp uge i64 %c, %w
  br i1 %cdoneb, label %finalize, label %col
col:
  %s = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %c)
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %cdone, label %coltype
coltype:
  %dt = load i32, ptr %s, align 8
  %skip = icmp ugt i32 %dt, 3
  br i1 %skip, label %cdone, label %colbuf
colbuf:
  %svp = getelementptr inbounds i8, ptr %s, i64 24
  %sv = load ptr, ptr %svp, align 8
  br label %rloop
rloop:
  %r = phi i64 [ 0, %colbuf ], [ %rn, %rcont ]
  %rdone = icmp uge i64 %r, %h
  br i1 %rdone, label %cdone, label %rchk
rchk:
  %v = call i1 @a_valid_at(ptr %s, i64 %r)
  br i1 %v, label %rbody, label %rcont
rbody:
  %x = call double @a_load_f64(ptr %sv, i32 %dt, i64 %r)
  %op2 = getelementptr inbounds double, ptr %outv, i64 %r
  %cur = load double, ptr %op2, align 8
  %mn = call double @llvm.minnum.f64(double %cur, double %x)
  %mx = call double @llvm.maxnum.f64(double %cur, double %x)
  %new = select i1 %ismax, double %mx, double %mn
  store double %new, ptr %op2, align 8
  %ccp = getelementptr inbounds i64, ptr %cntbuf, i64 %r
  %cc = load i64, ptr %ccp, align 8
  %ccn = add i64 %cc, 1
  store i64 %ccn, ptr %ccp, align 8
  br label %rcont
rcont:
  %rn = add nuw i64 %r, 1
  br label %rloop
cdone:
  %cn = add nuw i64 %c, 1
  br label %cloop
finalize:
  br label %fh
fh:
  %fr = phi i64 [ 0, %finalize ], [ %frn, %fcont ]
  %fdone = icmp uge i64 %fr, %h
  br i1 %fdone, label %freecnt, label %fbody
fbody:
  %fcp = getelementptr inbounds i64, ptr %cntbuf, i64 %fr
  %fc = load i64, ptr %fcp, align 8
  %has = icmp ugt i64 %fc, 0
  br i1 %has, label %fcont, label %fnull
fnull:
  %ig = call i32 @universe_dataframe_series_set_null(ptr %out, i64 %fr)
  br label %fcont
fcont:
  %frn = add nuw i64 %fr, 1
  br label %fh
freecnt:
  call void @free(ptr %cntbuf)
  ret ptr %out
}

; ---------------------------------------------------------------------------
; public API
; ---------------------------------------------------------------------------
define ptr @universe_dataframe_add(ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_binary(i32 0, ptr %a, ptr %b)
  ret ptr %r
}
define ptr @universe_dataframe_sub(ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_binary(i32 1, ptr %a, ptr %b)
  ret ptr %r
}
define ptr @universe_dataframe_mul(ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_binary(i32 2, ptr %a, ptr %b)
  ret ptr %r
}
define ptr @universe_dataframe_div(ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_binary(i32 3, ptr %a, ptr %b)
  ret ptr %r
}
define ptr @universe_dataframe_rem(ptr %a, ptr %b) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_binary(i32 4, ptr %a, ptr %b)
  ret ptr %r
}
define ptr @universe_dataframe_neg(ptr %a) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_unary(i32 0, ptr %a)
  ret ptr %r
}
define ptr @universe_dataframe_abs(ptr %a) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_unary(i32 1, ptr %a)
  ret ptr %r
}
define ptr @universe_dataframe_add_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_scalar(i32 0, ptr %a, i64 %ival, double %fval)
  ret ptr %r
}
define ptr @universe_dataframe_sub_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_scalar(i32 1, ptr %a, i64 %ival, double %fval)
  ret ptr %r
}
define ptr @universe_dataframe_mul_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_scalar(i32 2, ptr %a, i64 %ival, double %fval)
  ret ptr %r
}
define ptr @universe_dataframe_div_scalar(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_scalar(i32 3, ptr %a, i64 %ival, double %fval)
  ret ptr %r
}
define ptr @universe_dataframe_min_horizontal(ptr %df) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_horizontal(i32 0, ptr %df)
  ret ptr %r
}
define ptr @universe_dataframe_max_horizontal(ptr %df) local_unnamed_addr #0 {
entry:
  %r = call ptr @a_horizontal(i32 1, ptr %df)
  ret ptr %r
}

attributes #0 = { nounwind }
attributes #1 = { nounwind norecurse }
attributes #2 = { alwaysinline nounwind norecurse }
attributes #3 = { alwaysinline nounwind norecurse }
