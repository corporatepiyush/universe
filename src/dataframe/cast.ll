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

; universe_dataframe_* — SIMD dtype conversion + null-aware fill/clip/mask.
; The numpy `astype` + Polars fill_null/clip analog for the columnar Series of
; src/dataframe/frame.ll. Reads that module's Series layout directly (see the
; DOWNSTREAM CONTRACT in frame.ll): dtype@+0, len@+8, null_count@+16,
; values@+24, validity@+32 (null => all valid, else bitmap ceil(len/8), 1=VALID).
; DType i32: I32=0, I64=1, F32=2, F64=3, BOOL=4 (one byte/value), STR=5.
;
; DESIGN — SIMD-first with scalar fallback/oracle:
;   * cast: PRIMARY is a portable <4 x T> convert loop per (from,to) pair — a
;     single vector instruction per body: sext/trunc (int<->int), sitofp
;     (int->float), fpext/fptrunc (float<->float), llvm.fptosi.sat (float->int,
;     SATURATING so an out-of-range float clamps to INT_MIN/MAX instead of
;     poison). These lower to packed cvt on both ISAs (cvtdq2pd/cvttpd2dq/
;     cvtps2pd on x86; scvtf/fcvtzs/fcvtl/fcvtn on NEON) with NO runtime check.
;     A scalar tail (len % 4) handles the remainder AND is the test oracle.
;     Dispatch is a ONE-TIME cold switch on key = from*4+to; the hot leaf is
;     the monomorphic per-pair loop (no per-element branch). Validity + null
;     count are carried over verbatim (a value cast never adds/removes nulls).
;   * fill_null_value: the null lanes are replaced densely with a broadcast
;     fill via a vector `select`. Groups of 8 elements are aligned to ONE
;     validity byte: broadcast the byte, AND with <1,2,4,..,128>, `icmp ne 0`
;     builds the <8 x i1> keep-original mask in one shot — no per-lane bit
;     test. select(mask, orig, fill) keeps valid lanes, overwrites nulls. The
;     result has NO nulls (validity=null, null_count=0). Scalar tail for n%8.
;   * clip: vector llvm.smin/smax (int) or llvm.minnum/maxnum (float) clamp to
;     [lo,hi] over <4 x T>; clones then clamps in place, carrying validity.
;     NaN clamps to a bound (minnum/maxnum drop NaN); documented.
;   * is_null_mask / is_not_null_mask: SIMD bit->byte expand. One validity byte
;     -> 8 BOOL bytes via the same broadcast-AND-compare trick, then xor with
;     the invert flag (1 for is_null, 0 for is_not_null). A null validity ptr
;     means all-valid => a single memset of the constant result.
;
; The 128-bit vector path is the default; AVX2/AVX-512/SVE wide variants are a
; FUTURE runtime-dispatched add-on, never the baseline (per CLAUDE.md).
; Errors: constructors return null on NULL_PTR / OOM / INVALID_ARG (unsupported
; dtype, e.g. STR cast or clip/fill on STR).

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

; frame.ll exports
declare ptr @universe_dataframe_series_new(i32, i64)
declare ptr @universe_dataframe_series_clone(ptr)
declare void @universe_dataframe_series_free(ptr)

; saturating float->int converts (vector + scalar)
declare <4 x i32> @llvm.fptosi.sat.v4i32.v4f32(<4 x float>)
declare <4 x i64> @llvm.fptosi.sat.v4i64.v4f32(<4 x float>)
declare <4 x i32> @llvm.fptosi.sat.v4i32.v4f64(<4 x double>)
declare <4 x i64> @llvm.fptosi.sat.v4i64.v4f64(<4 x double>)
declare i32 @llvm.fptosi.sat.i32.f32(float)
declare i64 @llvm.fptosi.sat.i64.f32(float)
declare i32 @llvm.fptosi.sat.i32.f64(double)
declare i64 @llvm.fptosi.sat.i64.f64(double)

; clip clamp intrinsics (vector + scalar)
declare <4 x i32> @llvm.smin.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i64> @llvm.smin.v4i64(<4 x i64>, <4 x i64>)
declare <4 x i64> @llvm.smax.v4i64(<4 x i64>, <4 x i64>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i64 @llvm.smin.i64(i64, i64)
declare i64 @llvm.smax.i64(i64, i64)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.maxnum.f64(double, double)

; ===========================================================================
; cast converts — one per (from,to) numeric pair. <4 x T> vector body + tail.
; ===========================================================================

define internal void @cast_i32_i64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i32, ptr %src, i64 %i
  %v = load <4 x i32>, ptr %sp, align 4
  %cv = sext <4 x i32> %v to <4 x i64>
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  store <4 x i64> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i32, ptr %src, i64 %ti
  %tv = load i32, ptr %tsp, align 4
  %tcv = sext i32 %tv to i64
  %tdp = getelementptr inbounds i64, ptr %dst, i64 %ti
  store i64 %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_i32_f32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i32, ptr %src, i64 %i
  %v = load <4 x i32>, ptr %sp, align 4
  %cv = sitofp <4 x i32> %v to <4 x float>
  %dp = getelementptr inbounds float, ptr %dst, i64 %i
  store <4 x float> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i32, ptr %src, i64 %ti
  %tv = load i32, ptr %tsp, align 4
  %tcv = sitofp i32 %tv to float
  %tdp = getelementptr inbounds float, ptr %dst, i64 %ti
  store float %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_i32_f64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i32, ptr %src, i64 %i
  %v = load <4 x i32>, ptr %sp, align 4
  %cv = sitofp <4 x i32> %v to <4 x double>
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store <4 x double> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i32, ptr %src, i64 %ti
  %tv = load i32, ptr %tsp, align 4
  %tcv = sitofp i32 %tv to double
  %tdp = getelementptr inbounds double, ptr %dst, i64 %ti
  store double %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_i64_i32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %v = load <4 x i64>, ptr %sp, align 8
  %cv = trunc <4 x i64> %v to <4 x i32>
  %dp = getelementptr inbounds i32, ptr %dst, i64 %i
  store <4 x i32> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i64, ptr %src, i64 %ti
  %tv = load i64, ptr %tsp, align 8
  %tcv = trunc i64 %tv to i32
  %tdp = getelementptr inbounds i32, ptr %dst, i64 %ti
  store i32 %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_i64_f32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %v = load <4 x i64>, ptr %sp, align 8
  %cv = sitofp <4 x i64> %v to <4 x float>
  %dp = getelementptr inbounds float, ptr %dst, i64 %i
  store <4 x float> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i64, ptr %src, i64 %ti
  %tv = load i64, ptr %tsp, align 8
  %tcv = sitofp i64 %tv to float
  %tdp = getelementptr inbounds float, ptr %dst, i64 %ti
  store float %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_i64_f64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %v = load <4 x i64>, ptr %sp, align 8
  %cv = sitofp <4 x i64> %v to <4 x double>
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store <4 x double> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds i64, ptr %src, i64 %ti
  %tv = load i64, ptr %tsp, align 8
  %tcv = sitofp i64 %tv to double
  %tdp = getelementptr inbounds double, ptr %dst, i64 %ti
  store double %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f32_i32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds float, ptr %src, i64 %i
  %v = load <4 x float>, ptr %sp, align 4
  %cv = call <4 x i32> @llvm.fptosi.sat.v4i32.v4f32(<4 x float> %v)
  %dp = getelementptr inbounds i32, ptr %dst, i64 %i
  store <4 x i32> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds float, ptr %src, i64 %ti
  %tv = load float, ptr %tsp, align 4
  %tcv = call i32 @llvm.fptosi.sat.i32.f32(float %tv)
  %tdp = getelementptr inbounds i32, ptr %dst, i64 %ti
  store i32 %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f32_i64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds float, ptr %src, i64 %i
  %v = load <4 x float>, ptr %sp, align 4
  %cv = call <4 x i64> @llvm.fptosi.sat.v4i64.v4f32(<4 x float> %v)
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  store <4 x i64> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds float, ptr %src, i64 %ti
  %tv = load float, ptr %tsp, align 4
  %tcv = call i64 @llvm.fptosi.sat.i64.f32(float %tv)
  %tdp = getelementptr inbounds i64, ptr %dst, i64 %ti
  store i64 %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f32_f64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds float, ptr %src, i64 %i
  %v = load <4 x float>, ptr %sp, align 4
  %cv = fpext <4 x float> %v to <4 x double>
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store <4 x double> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds float, ptr %src, i64 %ti
  %tv = load float, ptr %tsp, align 4
  %tcv = fpext float %tv to double
  %tdp = getelementptr inbounds double, ptr %dst, i64 %ti
  store double %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f64_i32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %v = load <4 x double>, ptr %sp, align 8
  %cv = call <4 x i32> @llvm.fptosi.sat.v4i32.v4f64(<4 x double> %v)
  %dp = getelementptr inbounds i32, ptr %dst, i64 %i
  store <4 x i32> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds double, ptr %src, i64 %ti
  %tv = load double, ptr %tsp, align 8
  %tcv = call i32 @llvm.fptosi.sat.i32.f64(double %tv)
  %tdp = getelementptr inbounds i32, ptr %dst, i64 %ti
  store i32 %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f64_i64(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %v = load <4 x double>, ptr %sp, align 8
  %cv = call <4 x i64> @llvm.fptosi.sat.v4i64.v4f64(<4 x double> %v)
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  store <4 x i64> %cv, ptr %dp, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds double, ptr %src, i64 %ti
  %tv = load double, ptr %tsp, align 8
  %tcv = call i64 @llvm.fptosi.sat.i64.f64(double %tv)
  %tdp = getelementptr inbounds i64, ptr %dst, i64 %ti
  store i64 %tcv, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @cast_f64_f32(ptr noalias %src, ptr noalias %dst, i64 %n) #0 {
entry:
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %v = load <4 x double>, ptr %sp, align 8
  %cv = fptrunc <4 x double> %v to <4 x float>
  %dp = getelementptr inbounds float, ptr %dst, i64 %i
  store <4 x float> %cv, ptr %dp, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tsp = getelementptr inbounds double, ptr %src, i64 %ti
  %tv = load double, ptr %tsp, align 8
  %tcv = fptrunc double %tv to float
  %tdp = getelementptr inbounds float, ptr %dst, i64 %ti
  store float %tcv, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

; ===========================================================================
; universe_dataframe_cast — new Series in to_dtype, validity carried over.
; ===========================================================================
define ptr @universe_dataframe_cast(ptr %a, i32 %to) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %read, !prof !0

read:
  %from = load i32, ptr %a, align 8
  %same = icmp eq i32 %from, %to
  br i1 %same, label %cloneit, label %validate

cloneit:
  %cl = call ptr @universe_dataframe_series_clone(ptr %a)
  ret ptr %cl

validate:
  %fbad = icmp ugt i32 %from, 3
  %tbad = icmp ugt i32 %to, 3
  %bad = or i1 %fbad, %tbad
  br i1 %bad, label %fail, label %build, !prof !0

build:
  %lp = getelementptr inbounds i8, ptr %a, i64 8
  %len = load i64, ptr %lp, align 8
  %out = call ptr @universe_dataframe_series_new(i32 %to, i64 %len)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %ptrs, !prof !0

ptrs:
  %svpp = getelementptr inbounds i8, ptr %a, i64 24
  %src = load ptr, ptr %svpp, align 8
  %dvpp = getelementptr inbounds i8, ptr %out, i64 24
  %dst = load ptr, ptr %dvpp, align 8
  %f4 = shl i32 %from, 2
  %key = or i32 %f4, %to
  switch i32 %key, label %carry [
    i32 1,  label %k1
    i32 2,  label %k2
    i32 3,  label %k3
    i32 4,  label %k4
    i32 6,  label %k6
    i32 7,  label %k7
    i32 8,  label %k8
    i32 9,  label %k9
    i32 11, label %k11
    i32 12, label %k12
    i32 13, label %k13
    i32 14, label %k14
  ]

k1:
  call void @cast_i32_i64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k2:
  call void @cast_i32_f32(ptr %src, ptr %dst, i64 %len)
  br label %carry
k3:
  call void @cast_i32_f64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k4:
  call void @cast_i64_i32(ptr %src, ptr %dst, i64 %len)
  br label %carry
k6:
  call void @cast_i64_f32(ptr %src, ptr %dst, i64 %len)
  br label %carry
k7:
  call void @cast_i64_f64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k8:
  call void @cast_f32_i32(ptr %src, ptr %dst, i64 %len)
  br label %carry
k9:
  call void @cast_f32_i64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k11:
  call void @cast_f32_f64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k12:
  call void @cast_f64_i32(ptr %src, ptr %dst, i64 %len)
  br label %carry
k13:
  call void @cast_f64_i64(ptr %src, ptr %dst, i64 %len)
  br label %carry
k14:
  call void @cast_f64_f32(ptr %src, ptr %dst, i64 %len)
  br label %carry

carry:
  %avlp = getelementptr inbounds i8, ptr %a, i64 32
  %avld = load ptr, ptr %avlp, align 8
  %hasvld = icmp ne ptr %avld, null
  %t = add i64 %len, 7
  %nb = lshr i64 %t, 3
  %hasbytes = icmp ugt i64 %nb, 0
  %docarry = and i1 %hasvld, %hasbytes
  br i1 %docarry, label %copyvld, label %done

copyvld:
  %obm = call ptr @malloc(i64 %nb)
  %obmn = icmp eq ptr %obm, null
  br i1 %obmn, label %done, label %storevld, !prof !0

storevld:
  call void @llvm.memcpy.p0.p0.i64(ptr %obm, ptr %avld, i64 %nb, i1 false)
  %ovlp = getelementptr inbounds i8, ptr %out, i64 32
  store ptr %obm, ptr %ovlp, align 8
  %ancp = getelementptr inbounds i8, ptr %a, i64 16
  %anc = load i64, ptr %ancp, align 8
  %oncp = getelementptr inbounds i8, ptr %out, i64 16
  store i64 %anc, ptr %oncp, align 8
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

; ===========================================================================
; fill_null — per-dtype dense select over the validity bitmap (groups of 8).
; ===========================================================================

define internal void @fill_i32(ptr %dst, ptr %src, ptr %vld, i64 %n, i32 %fill) #0 {
entry:
  %f0 = insertelement <8 x i32> poison, i32 %fill, i64 0
  %fv = shufflevector <8 x i32> %f0, <8 x i32> poison, <8 x i32> zeroinitializer
  %vend = and i64 %n, -8
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %g = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %valid = icmp ne <8 x i8> %mk, zeroinitializer
  %sp = getelementptr inbounds i32, ptr %src, i64 %i
  %orig = load <8 x i32>, ptr %sp, align 4
  %res = select <8 x i1> %valid, <8 x i32> %orig, <8 x i32> %fv
  %dp = getelementptr inbounds i32, ptr %dst, i64 %i
  store <8 x i32> %res, ptr %dp, align 4
  %in = add i64 %i, 8
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tbb = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tbb, %tsh8
  %tlo = and i8 %tbit, 1
  %tvalid = icmp ne i8 %tlo, 0
  %tsp = getelementptr inbounds i32, ptr %src, i64 %ti
  %to = load i32, ptr %tsp, align 4
  %tres = select i1 %tvalid, i32 %to, i32 %fill
  %tdp = getelementptr inbounds i32, ptr %dst, i64 %ti
  store i32 %tres, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @fill_i64(ptr %dst, ptr %src, ptr %vld, i64 %n, i64 %fill) #0 {
entry:
  %f0 = insertelement <8 x i64> poison, i64 %fill, i64 0
  %fv = shufflevector <8 x i64> %f0, <8 x i64> poison, <8 x i32> zeroinitializer
  %vend = and i64 %n, -8
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %g = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %valid = icmp ne <8 x i8> %mk, zeroinitializer
  %sp = getelementptr inbounds i64, ptr %src, i64 %i
  %orig = load <8 x i64>, ptr %sp, align 8
  %res = select <8 x i1> %valid, <8 x i64> %orig, <8 x i64> %fv
  %dp = getelementptr inbounds i64, ptr %dst, i64 %i
  store <8 x i64> %res, ptr %dp, align 8
  %in = add i64 %i, 8
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tbb = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tbb, %tsh8
  %tlo = and i8 %tbit, 1
  %tvalid = icmp ne i8 %tlo, 0
  %tsp = getelementptr inbounds i64, ptr %src, i64 %ti
  %to = load i64, ptr %tsp, align 8
  %tres = select i1 %tvalid, i64 %to, i64 %fill
  %tdp = getelementptr inbounds i64, ptr %dst, i64 %ti
  store i64 %tres, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @fill_f32(ptr %dst, ptr %src, ptr %vld, i64 %n, float %fill) #0 {
entry:
  %f0 = insertelement <8 x float> poison, float %fill, i64 0
  %fv = shufflevector <8 x float> %f0, <8 x float> poison, <8 x i32> zeroinitializer
  %vend = and i64 %n, -8
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %g = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %valid = icmp ne <8 x i8> %mk, zeroinitializer
  %sp = getelementptr inbounds float, ptr %src, i64 %i
  %orig = load <8 x float>, ptr %sp, align 4
  %res = select <8 x i1> %valid, <8 x float> %orig, <8 x float> %fv
  %dp = getelementptr inbounds float, ptr %dst, i64 %i
  store <8 x float> %res, ptr %dp, align 4
  %in = add i64 %i, 8
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tbb = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tbb, %tsh8
  %tlo = and i8 %tbit, 1
  %tvalid = icmp ne i8 %tlo, 0
  %tsp = getelementptr inbounds float, ptr %src, i64 %ti
  %to = load float, ptr %tsp, align 4
  %tres = select i1 %tvalid, float %to, float %fill
  %tdp = getelementptr inbounds float, ptr %dst, i64 %ti
  store float %tres, ptr %tdp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @fill_f64(ptr %dst, ptr %src, ptr %vld, i64 %n, double %fill) #0 {
entry:
  %f0 = insertelement <8 x double> poison, double %fill, i64 0
  %fv = shufflevector <8 x double> %f0, <8 x double> poison, <8 x i32> zeroinitializer
  %vend = and i64 %n, -8
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %g = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %valid = icmp ne <8 x i8> %mk, zeroinitializer
  %sp = getelementptr inbounds double, ptr %src, i64 %i
  %orig = load <8 x double>, ptr %sp, align 8
  %res = select <8 x i1> %valid, <8 x double> %orig, <8 x double> %fv
  %dp = getelementptr inbounds double, ptr %dst, i64 %i
  store <8 x double> %res, ptr %dp, align 8
  %in = add i64 %i, 8
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tbb = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tbb, %tsh8
  %tlo = and i8 %tbit, 1
  %tvalid = icmp ne i8 %tlo, 0
  %tsp = getelementptr inbounds double, ptr %src, i64 %ti
  %to = load double, ptr %tsp, align 8
  %tres = select i1 %tvalid, double %to, double %fill
  %tdp = getelementptr inbounds double, ptr %dst, i64 %ti
  store double %tres, ptr %tdp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @fill_bool(ptr %dst, ptr %src, ptr %vld, i64 %n, i8 %fill) #0 {
entry:
  %f0 = insertelement <8 x i8> poison, i8 %fill, i64 0
  %fv = shufflevector <8 x i8> %f0, <8 x i8> poison, <8 x i32> zeroinitializer
  %vend = and i64 %n, -8
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %g = lshr i64 %i, 3
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %valid = icmp ne <8 x i8> %mk, zeroinitializer
  %sp = getelementptr inbounds i8, ptr %src, i64 %i
  %orig = load <8 x i8>, ptr %sp, align 1
  %res = select <8 x i1> %valid, <8 x i8> %orig, <8 x i8> %fv
  %dp = getelementptr inbounds i8, ptr %dst, i64 %i
  store <8 x i8> %res, ptr %dp, align 1
  %in = add i64 %i, 8
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tbb = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tbb, %tsh8
  %tlo = and i8 %tbit, 1
  %tvalid = icmp ne i8 %tlo, 0
  %tsp = getelementptr inbounds i8, ptr %src, i64 %ti
  %to = load i8, ptr %tsp, align 1
  %tres = select i1 %tvalid, i8 %to, i8 %fill
  %tdp = getelementptr inbounds i8, ptr %dst, i64 %ti
  store i8 %tres, ptr %tdp, align 1
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define ptr @universe_dataframe_fill_null_value(ptr %a, i32 %stype, i64 %ival, double %fval) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %read, !prof !0

read:
  %dtype = load i32, ptr %a, align 8
  %isstr = icmp eq i32 %dtype, 5
  br i1 %isstr, label %fail, label %chkvld, !prof !0

chkvld:
  %avlp = getelementptr inbounds i8, ptr %a, i64 32
  %avld = load ptr, ptr %avlp, align 8
  %novld = icmp eq ptr %avld, null
  br i1 %novld, label %cloneit, label %build

cloneit:
  %cl = call ptr @universe_dataframe_series_clone(ptr %a)
  ret ptr %cl

build:
  %lp = getelementptr inbounds i8, ptr %a, i64 8
  %len = load i64, ptr %lp, align 8
  %out = call ptr @universe_dataframe_series_new(i32 %dtype, i64 %len)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %ptrs, !prof !0

ptrs:
  %svpp = getelementptr inbounds i8, ptr %a, i64 24
  %src = load ptr, ptr %svpp, align 8
  %dvpp = getelementptr inbounds i8, ptr %out, i64 24
  %dst = load ptr, ptr %dvpp, align 8
  switch i32 %dtype, label %done [
    i32 0, label %d_i32
    i32 1, label %d_i64
    i32 2, label %d_f32
    i32 3, label %d_f64
    i32 4, label %d_bool
  ]

d_i32:
  %fi32 = trunc i64 %ival to i32
  call void @fill_i32(ptr %dst, ptr %src, ptr %avld, i64 %len, i32 %fi32)
  br label %done
d_i64:
  call void @fill_i64(ptr %dst, ptr %src, ptr %avld, i64 %len, i64 %ival)
  br label %done
d_f32:
  %ff32 = fptrunc double %fval to float
  call void @fill_f32(ptr %dst, ptr %src, ptr %avld, i64 %len, float %ff32)
  br label %done
d_f64:
  call void @fill_f64(ptr %dst, ptr %src, ptr %avld, i64 %len, double %fval)
  br label %done
d_bool:
  %nz = icmp ne i64 %ival, 0
  %fb = zext i1 %nz to i8
  call void @fill_bool(ptr %dst, ptr %src, ptr %avld, i64 %len, i8 %fb)
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

; ===========================================================================
; clip — clone then clamp values in place, validity preserved by the clone.
; ===========================================================================

define internal void @clip_i32(ptr %vals, i64 %n, i32 %lo, i32 %hi) #2 {
entry:
  %l0 = insertelement <4 x i32> poison, i32 %lo, i64 0
  %lv = shufflevector <4 x i32> %l0, <4 x i32> poison, <4 x i32> zeroinitializer
  %h0 = insertelement <4 x i32> poison, i32 %hi, i64 0
  %hv = shufflevector <4 x i32> %h0, <4 x i32> poison, <4 x i32> zeroinitializer
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %p = getelementptr inbounds i32, ptr %vals, i64 %i
  %v = load <4 x i32>, ptr %p, align 4
  %t = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %v, <4 x i32> %hv)
  %r = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %t, <4 x i32> %lv)
  store <4 x i32> %r, ptr %p, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tp = getelementptr inbounds i32, ptr %vals, i64 %ti
  %tv = load i32, ptr %tp, align 4
  %tt = call i32 @llvm.smin.i32(i32 %tv, i32 %hi)
  %tr = call i32 @llvm.smax.i32(i32 %tt, i32 %lo)
  store i32 %tr, ptr %tp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @clip_i64(ptr %vals, i64 %n, i64 %lo, i64 %hi) #2 {
entry:
  %l0 = insertelement <4 x i64> poison, i64 %lo, i64 0
  %lv = shufflevector <4 x i64> %l0, <4 x i64> poison, <4 x i32> zeroinitializer
  %h0 = insertelement <4 x i64> poison, i64 %hi, i64 0
  %hv = shufflevector <4 x i64> %h0, <4 x i64> poison, <4 x i32> zeroinitializer
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %p = getelementptr inbounds i64, ptr %vals, i64 %i
  %v = load <4 x i64>, ptr %p, align 8
  %t = call <4 x i64> @llvm.smin.v4i64(<4 x i64> %v, <4 x i64> %hv)
  %r = call <4 x i64> @llvm.smax.v4i64(<4 x i64> %t, <4 x i64> %lv)
  store <4 x i64> %r, ptr %p, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tp = getelementptr inbounds i64, ptr %vals, i64 %ti
  %tv = load i64, ptr %tp, align 8
  %tt = call i64 @llvm.smin.i64(i64 %tv, i64 %hi)
  %tr = call i64 @llvm.smax.i64(i64 %tt, i64 %lo)
  store i64 %tr, ptr %tp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @clip_f32(ptr %vals, i64 %n, float %lo, float %hi) #2 {
entry:
  %l0 = insertelement <4 x float> poison, float %lo, i64 0
  %lv = shufflevector <4 x float> %l0, <4 x float> poison, <4 x i32> zeroinitializer
  %h0 = insertelement <4 x float> poison, float %hi, i64 0
  %hv = shufflevector <4 x float> %h0, <4 x float> poison, <4 x i32> zeroinitializer
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %p = getelementptr inbounds float, ptr %vals, i64 %i
  %v = load <4 x float>, ptr %p, align 4
  %t = call <4 x float> @llvm.minnum.v4f32(<4 x float> %v, <4 x float> %hv)
  %r = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %t, <4 x float> %lv)
  store <4 x float> %r, ptr %p, align 4
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tp = getelementptr inbounds float, ptr %vals, i64 %ti
  %tv = load float, ptr %tp, align 4
  %tt = call float @llvm.minnum.f32(float %tv, float %hi)
  %tr = call float @llvm.maxnum.f32(float %tt, float %lo)
  store float %tr, ptr %tp, align 4
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define internal void @clip_f64(ptr %vals, i64 %n, double %lo, double %hi) #2 {
entry:
  %l0 = insertelement <4 x double> poison, double %lo, i64 0
  %lv = shufflevector <4 x double> %l0, <4 x double> poison, <4 x i32> zeroinitializer
  %h0 = insertelement <4 x double> poison, double %hi, i64 0
  %hv = shufflevector <4 x double> %h0, <4 x double> poison, <4 x i32> zeroinitializer
  %vend = and i64 %n, -4
  br label %vh
vh:
  %i = phi i64 [ 0, %entry ], [ %in, %vb ]
  %c = icmp ult i64 %i, %vend
  br i1 %c, label %vb, label %th
vb:
  %p = getelementptr inbounds double, ptr %vals, i64 %i
  %v = load <4 x double>, ptr %p, align 8
  %t = call <4 x double> @llvm.minnum.v4f64(<4 x double> %v, <4 x double> %hv)
  %r = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %t, <4 x double> %lv)
  store <4 x double> %r, ptr %p, align 8
  %in = add i64 %i, 4
  br label %vh
th:
  %ti = phi i64 [ %vend, %vh ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tp = getelementptr inbounds double, ptr %vals, i64 %ti
  %tv = load double, ptr %tp, align 8
  %tt = call double @llvm.minnum.f64(double %tv, double %hi)
  %tr = call double @llvm.maxnum.f64(double %tt, double %lo)
  store double %tr, ptr %tp, align 8
  %tin = add i64 %ti, 1
  br label %th
done:
  ret void
}

define ptr @universe_dataframe_clip(ptr %a, double %lo, double %hi) local_unnamed_addr #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %read, !prof !0

read:
  %dtype = load i32, ptr %a, align 8
  %bad = icmp ugt i32 %dtype, 3
  br i1 %bad, label %fail, label %clone, !prof !0

clone:
  %out = call ptr @universe_dataframe_series_clone(ptr %a)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %ptrs, !prof !0

ptrs:
  %lp = getelementptr inbounds i8, ptr %out, i64 8
  %len = load i64, ptr %lp, align 8
  %vpp = getelementptr inbounds i8, ptr %out, i64 24
  %vals = load ptr, ptr %vpp, align 8
  switch i32 %dtype, label %done [
    i32 0, label %d_i32
    i32 1, label %d_i64
    i32 2, label %d_f32
    i32 3, label %d_f64
  ]

d_i32:
  %ilo = call i32 @llvm.fptosi.sat.i32.f64(double %lo)
  %ihi = call i32 @llvm.fptosi.sat.i32.f64(double %hi)
  call void @clip_i32(ptr %vals, i64 %len, i32 %ilo, i32 %ihi)
  br label %done
d_i64:
  %llo = call i64 @llvm.fptosi.sat.i64.f64(double %lo)
  %lhi = call i64 @llvm.fptosi.sat.i64.f64(double %hi)
  call void @clip_i64(ptr %vals, i64 %len, i64 %llo, i64 %lhi)
  br label %done
d_f32:
  %flo = fptrunc double %lo to float
  %fhi = fptrunc double %hi to float
  call void @clip_f32(ptr %vals, i64 %len, float %flo, float %fhi)
  br label %done
d_f64:
  call void @clip_f64(ptr %vals, i64 %len, double %lo, double %hi)
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

; ===========================================================================
; is_null_mask / is_not_null_mask — SIMD bit->byte expand of the validity map.
; ===========================================================================

; res[k] = valid(k) XOR invert.  invert=1 => is_null, invert=0 => is_not_null.
define internal void @build_null_mask(ptr %vld, ptr %out, i64 %n, i8 %invert) #0 {
entry:
  %iv0 = insertelement <8 x i8> poison, i8 %invert, i64 0
  %ivv = shufflevector <8 x i8> %iv0, <8 x i8> poison, <8 x i32> zeroinitializer
  %nbytes = lshr i64 %n, 3
  br label %vh
vh:
  %g = phi i64 [ 0, %entry ], [ %gn, %vb ]
  %c = icmp ult i64 %g, %nbytes
  br i1 %c, label %vb, label %th
vb:
  %bp = getelementptr inbounds i8, ptr %vld, i64 %g
  %b = load i8, ptr %bp, align 1
  %b0 = insertelement <8 x i8> poison, i8 %b, i64 0
  %bc = shufflevector <8 x i8> %b0, <8 x i8> poison, <8 x i32> zeroinitializer
  %mk = and <8 x i8> %bc, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %nz = icmp ne <8 x i8> %mk, zeroinitializer
  %valid8 = zext <8 x i1> %nz to <8 x i8>
  %res = xor <8 x i8> %valid8, %ivv
  %base = shl i64 %g, 3
  %op = getelementptr inbounds i8, ptr %out, i64 %base
  store <8 x i8> %res, ptr %op, align 1
  %gn = add i64 %g, 1
  br label %vh
th:
  %tstart = shl i64 %nbytes, 3
  br label %tl
tl:
  %ti = phi i64 [ %tstart, %th ], [ %tin, %tb ]
  %tc = icmp ult i64 %ti, %n
  br i1 %tc, label %tb, label %done
tb:
  %tg = lshr i64 %ti, 3
  %tbp = getelementptr inbounds i8, ptr %vld, i64 %tg
  %tb8 = load i8, ptr %tbp, align 1
  %tsh = and i64 %ti, 7
  %tsh8 = trunc i64 %tsh to i8
  %tbit = lshr i8 %tb8, %tsh8
  %tlo = and i8 %tbit, 1
  %tres = xor i8 %tlo, %invert
  %top = getelementptr inbounds i8, ptr %out, i64 %ti
  store i8 %tres, ptr %top, align 1
  %tin = add i64 %ti, 1
  br label %tl
done:
  ret void
}

; shared body: build a BOOL Series; invert=1 => null mask, 0 => not-null mask.
define internal ptr @nullmask_impl(ptr %a, i8 %invert) #1 {
entry:
  %an = icmp eq ptr %a, null
  br i1 %an, label %fail, label %build, !prof !0

build:
  %lp = getelementptr inbounds i8, ptr %a, i64 8
  %len = load i64, ptr %lp, align 8
  %out = call ptr @universe_dataframe_series_new(i32 4, i64 %len)
  %on = icmp eq ptr %out, null
  br i1 %on, label %fail, label %ptrs, !prof !0

ptrs:
  %ovpp = getelementptr inbounds i8, ptr %out, i64 24
  %ovals = load ptr, ptr %ovpp, align 8
  %avlp = getelementptr inbounds i8, ptr %a, i64 32
  %avld = load ptr, ptr %avlp, align 8
  %novld = icmp eq ptr %avld, null
  br i1 %novld, label %allvalid, label %expand

allvalid:
  ; all valid => is_null all 0, is_not_null all 1 => memset (invert XOR 1)
  %cval = xor i8 %invert, 1
  call void @llvm.memset.p0.i64(ptr %ovals, i8 %cval, i64 %len, i1 false)
  br label %done

expand:
  call void @build_null_mask(ptr %avld, ptr %ovals, i64 %len, i8 %invert)
  br label %done

done:
  ret ptr %out

fail:
  ret ptr null
}

define ptr @universe_dataframe_is_null_mask(ptr %a) local_unnamed_addr #1 {
entry:
  %r = call ptr @nullmask_impl(ptr %a, i8 1)
  ret ptr %r
}

define ptr @universe_dataframe_is_not_null_mask(ptr %a) local_unnamed_addr #1 {
entry:
  %r = call ptr @nullmask_impl(ptr %a, i8 0)
  ret ptr %r
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }

!0 = !{!"branch_weights", i32 1, i32 2000}
