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

; DataFrame column reductions (Polars method names, our own low-level design).
; Operates on a Series handle (frame.ll contract). Integer/float columns reduce
; to an f64 result; nulls are SKIPPED via the validity bitmap.
;
; ============================ DESIGN ============================
; Series layout relied upon (frame.ll DOWNSTREAM CONTRACT):
;   +0 i32 dtype (I32=0,I64=1,F32=2,F64=3,BOOL=4,STR=5)
;   +8 i64 len   +16 i64 null_count   +24 ptr values   +32 ptr validity
;   validity: null => all valid; else bitmap ceil(len/8) bytes, bit=1 VALID.
;
; ALGORITHM CLASS — one-pass SIMD stats, scalar oracle/fallback:
;   * The DENSE path (no validity bitmap, fixed numeric dtype 0..3) runs ONE
;     memory pass through a per-dtype vector kernel that computes sum, sum-of-
;     squares, min AND max together. Reductions stream large columns and are
;     MEMORY-BOUND, so the extra sumsq/min/max ALU work hides under load
;     latency — computing all four in a single pass minimises memory traffic
;     for whichever reduction the caller wants. Each kernel widens the loaded
;     lane vector to <4 x double> (F64 loads directly; F32 fpext; I32/I64
;     sitofp) so the reduction body is identical across dtypes. FOUR
;     accumulators (2 sum + 2 sumsq <4 x double>) break the loop-carried FP
;     dependency chain; min/max use minnum/maxnum. `fast` FP on sum/sumsq
;     (reassoc/contract/nsz/arcp) enables tree reduction + fmla; min/max use
;     UNFLAGGED minnum/maxnum. A scalar tail finishes the <8-element remainder.
;   * The SCALAR path is BOTH the null-aware fallback (validity present, or
;     BOOL which is stored one byte/value) AND the test oracle: strict
;     left-to-right fadd/fmul so the vector path is cross-checked against it.
;     It skips null lanes via the bitmap.
;   * BOOL reduces through the scalar path (rare for reductions; documented).
;
; median/quantile: extract non-null values into an f64 scratch buffer, sort
;   ascending (universe_sort_quick, cold path — reuse over reinvention), then
;   linear-interpolate the order statistic. NaN ordering is unspecified (v1).
; n_unique: inline open-addressing hash set over canonical i64 keys (float bits
;   with -0.0 normalised to +0.0 and NaN canonicalised); null counts as ONE
;   distinct value if present (Polars semantics). Power-of-two capacity, mask
;   probe, load factor < 0.5.
; sum_horizontal/mean_horizontal: column-wise accumulate into a new F64 Series
;   of length height (cache-friendly: add each numeric column's values into the
;   running row buffer). STR columns are skipped. mean divides by the per-row
;   count of non-null numeric cells; a row with zero numeric cells is null.
;
; Errors i32: 0 OK, 1 NULL_PTR, 2 OOM, 4 EMPTY (reduction of no non-null
;   values / n<2 for var-std), 8 INVALID_ARG (non-numeric dtype, q out of
;   range). Constructors returning ptr signal failure with null.
; ===============================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>)
declare double @llvm.vector.reduce.fmin.v4f64(<4 x double>)
declare double @llvm.vector.reduce.fmax.v4f64(<4 x double>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.maxnum.f64(double, double)
declare double @llvm.sqrt.f64(double)
declare double @llvm.floor.f64(double)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i64 @llvm.umax.i64(i64, i64)

; frame.ll exports we compose with (cold surface only)
declare ptr @universe_dataframe_series_new(i32, i64)
declare i32 @universe_dataframe_series_set_null(ptr, i64)
declare ptr @universe_dataframe_select_at_idx(ptr, i64)
declare i64 @universe_dataframe_height(ptr)
declare i64 @universe_dataframe_width(ptr)
; sort domain (cold path)
declare i32 @universe_sort_quick(ptr, i64, i64, ptr)

; ---------------------------------------------------------------------------
; internal helpers
; ---------------------------------------------------------------------------

; true if element i is valid (non-null). validity==null => all valid.
define internal i1 @df_valid_at(ptr %s, i64 %i) #2 {
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

; load element i of a fixed-width numeric column as a double.
define internal double @df_load_f64(ptr %vals, i32 %dtype, i64 %i) #2 {
entry:
  switch i32 %dtype, label %dflt [ i32 0, label %l0
                                   i32 1, label %l1
                                   i32 2, label %l2
                                   i32 3, label %l3
                                   i32 4, label %l4 ]
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
l4:
  %p4 = getelementptr inbounds i8, ptr %vals, i64 %i
  %r4 = load i8, ptr %p4, align 1
  %z4 = and i8 %r4, 1
  %d4 = uitofp i8 %z4 to double
  ret double %d4
dflt:
  ret double 0.0
}

define internal i64 @df_canon_bits(double %x) #3 {
entry:
  %isz = fcmp oeq double %x, 0.0
  %nz = select i1 %isz, double 0.0, double %x
  %isnan = fcmp uno double %nz, %nz
  %cn = select i1 %isnan, double 0x7FF8000000000000, double %nz
  %b = bitcast double %cn to i64
  ret i64 %b
}

; canonical i64 key for element i (distinct-by-value): ints raw, floats bits
; with -0.0 normalised to +0.0 and NaN canonicalised.
define internal i64 @df_key_i64(ptr %vals, i32 %dtype, i64 %i) #2 {
entry:
  switch i32 %dtype, label %kflt [ i32 0, label %k0
                                   i32 1, label %k1
                                   i32 2, label %k2
                                   i32 3, label %k3
                                   i32 4, label %k4 ]
k0:
  %p0 = getelementptr inbounds i32, ptr %vals, i64 %i
  %r0 = load i32, ptr %p0, align 4
  %e0 = sext i32 %r0 to i64
  ret i64 %e0
k1:
  %p1 = getelementptr inbounds i64, ptr %vals, i64 %i
  %r1 = load i64, ptr %p1, align 8
  ret i64 %r1
k2:
  %p2 = getelementptr inbounds float, ptr %vals, i64 %i
  %r2 = load float, ptr %p2, align 4
  %f2 = fpext float %r2 to double
  %b2 = call i64 @df_canon_bits(double %f2)
  ret i64 %b2
k3:
  %p3 = getelementptr inbounds double, ptr %vals, i64 %i
  %r3 = load double, ptr %p3, align 8
  %b3 = call i64 @df_canon_bits(double %r3)
  ret i64 %b3
k4:
  %p4 = getelementptr inbounds i8, ptr %vals, i64 %i
  %r4 = load i8, ptr %p4, align 1
  %z4 = and i8 %r4, 1
  %e4 = zext i8 %z4 to i64
  ret i64 %e4
kflt:
  ret i64 0
}

; double comparator for universe_sort_quick (ascending; NaN => equal, v1).
define internal i32 @df_cmp_f64(ptr %a, ptr %b) #2 {
entry:
  %la = load double, ptr %a, align 8
  %lb = load double, ptr %b, align 8
  %lt = fcmp olt double %la, %lb
  %gt = fcmp ogt double %la, %lb
  %r1 = select i1 %gt, i32 1, i32 0
  %r = select i1 %lt, i32 -1, i32 %r1
  ret i32 %r
}

; sample variance from sum, sumsq, n (n>=2). Clamped >= 0 to avoid FP-negative
; -> NaN std for near-constant data.
define internal double @df_variance(double %sum, double %sq, i64 %c) #3 {
entry:
  %n = uitofp i64 %c to double
  %ss = fmul double %sum, %sum
  %corr = fdiv double %ss, %n
  %num = fsub double %sq, %corr
  %nm1 = fsub double %n, 1.0
  %var0 = fdiv double %num, %nm1
  %var = call double @llvm.maxnum.f64(double %var0, double 0.0)
  ret double %var
}

; ---------------------------------------------------------------------------
; DENSE (no-null) SIMD stat kernels — one pass, per dtype. sum/sumsq/min/max.
; ---------------------------------------------------------------------------

define internal void @df_dense_stats_i32(ptr %vals, i64 %n, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %nbulk = and i64 %n, -8
  %has = icmp uge i64 %nbulk, 8
  br i1 %has, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %s0 = phi <4 x double> [ zeroinitializer, %entry ], [ %s0n, %main ]
  %s1 = phi <4 x double> [ zeroinitializer, %entry ], [ %s1n, %main ]
  %q0 = phi <4 x double> [ zeroinitializer, %entry ], [ %q0n, %main ]
  %q1 = phi <4 x double> [ zeroinitializer, %entry ], [ %q1n, %main ]
  %mn = phi <4 x double> [ <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>, %entry ], [ %mnn, %main ]
  %mx = phi <4 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000>, %entry ], [ %mxn, %main ]
  %pa = getelementptr inbounds nuw i32, ptr %vals, i64 %i
  %ra = load <4 x i32>, ptr %pa, align 4
  %va = sitofp <4 x i32> %ra to <4 x double>
  %i4 = add nuw i64 %i, 4
  %pb = getelementptr inbounds nuw i32, ptr %vals, i64 %i4
  %rb = load <4 x i32>, ptr %pb, align 4
  %vb = sitofp <4 x i32> %rb to <4 x double>
  %s0n = fadd fast <4 x double> %s0, %va
  %s1n = fadd fast <4 x double> %s1, %vb
  %sqa = fmul fast <4 x double> %va, %va
  %q0n = fadd fast <4 x double> %q0, %sqa
  %sqb = fmul fast <4 x double> %vb, %vb
  %q1n = fadd fast <4 x double> %q1, %sqb
  %mab = call <4 x double> @llvm.minnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mnn = call <4 x double> @llvm.minnum.v4f64(<4 x double> %mn, <4 x double> %mab)
  %xab = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mxn = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %mx, <4 x double> %xab)
  %inext = add nuw i64 %i, 8
  %more = icmp ult i64 %inext, %nbulk
  br i1 %more, label %main, label %maindone

maindone:
  %sm = fadd fast <4 x double> %s0n, %s1n
  %sv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %sm)
  %qm = fadd fast <4 x double> %q0n, %q1n
  %qv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %qm)
  %mnv = call double @llvm.vector.reduce.fmin.v4f64(<4 x double> %mnn)
  %mxv = call double @llvm.vector.reduce.fmax.v4f64(<4 x double> %mxn)
  br label %red

red:
  %bsum = phi double [ 0.0, %entry ], [ %sv, %maindone ]
  %bsq = phi double [ 0.0, %entry ], [ %qv, %maindone ]
  %bmin = phi double [ 0x7FF0000000000000, %entry ], [ %mnv, %maindone ]
  %bmax = phi double [ 0xFFF0000000000000, %entry ], [ %mxv, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %nbulk, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jn, %body ]
  %ts = phi double [ %bsum, %red ], [ %tsn, %body ]
  %tq = phi double [ %bsq, %red ], [ %tqn, %body ]
  %tmn = phi double [ %bmin, %red ], [ %tmnn, %body ]
  %tmx = phi double [ %bmax, %red ], [ %tmxn, %body ]
  %d = icmp uge i64 %j, %n
  br i1 %d, label %fin, label %body

body:
  %tp = getelementptr inbounds nuw i32, ptr %vals, i64 %j
  %tr = load i32, ptr %tp, align 4
  %x = sitofp i32 %tr to double
  %tsn = fadd fast double %ts, %x
  %xx = fmul fast double %x, %x
  %tqn = fadd fast double %tq, %xx
  %tmnn = call double @llvm.minnum.f64(double %tmn, double %x)
  %tmxn = call double @llvm.maxnum.f64(double %tmx, double %x)
  %jn = add nuw i64 %j, 1
  br label %tail

fin:
  store double %ts, ptr %osum, align 8
  store double %tq, ptr %osq, align 8
  store double %tmn, ptr %omin, align 8
  store double %tmx, ptr %omax, align 8
  ret void
}

define internal void @df_dense_stats_i64(ptr %vals, i64 %n, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %nbulk = and i64 %n, -8
  %has = icmp uge i64 %nbulk, 8
  br i1 %has, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %s0 = phi <4 x double> [ zeroinitializer, %entry ], [ %s0n, %main ]
  %s1 = phi <4 x double> [ zeroinitializer, %entry ], [ %s1n, %main ]
  %q0 = phi <4 x double> [ zeroinitializer, %entry ], [ %q0n, %main ]
  %q1 = phi <4 x double> [ zeroinitializer, %entry ], [ %q1n, %main ]
  %mn = phi <4 x double> [ <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>, %entry ], [ %mnn, %main ]
  %mx = phi <4 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000>, %entry ], [ %mxn, %main ]
  %pa = getelementptr inbounds nuw i64, ptr %vals, i64 %i
  %ra = load <4 x i64>, ptr %pa, align 8
  %va = sitofp <4 x i64> %ra to <4 x double>
  %i4 = add nuw i64 %i, 4
  %pb = getelementptr inbounds nuw i64, ptr %vals, i64 %i4
  %rb = load <4 x i64>, ptr %pb, align 8
  %vb = sitofp <4 x i64> %rb to <4 x double>
  %s0n = fadd fast <4 x double> %s0, %va
  %s1n = fadd fast <4 x double> %s1, %vb
  %sqa = fmul fast <4 x double> %va, %va
  %q0n = fadd fast <4 x double> %q0, %sqa
  %sqb = fmul fast <4 x double> %vb, %vb
  %q1n = fadd fast <4 x double> %q1, %sqb
  %mab = call <4 x double> @llvm.minnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mnn = call <4 x double> @llvm.minnum.v4f64(<4 x double> %mn, <4 x double> %mab)
  %xab = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mxn = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %mx, <4 x double> %xab)
  %inext = add nuw i64 %i, 8
  %more = icmp ult i64 %inext, %nbulk
  br i1 %more, label %main, label %maindone

maindone:
  %sm = fadd fast <4 x double> %s0n, %s1n
  %sv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %sm)
  %qm = fadd fast <4 x double> %q0n, %q1n
  %qv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %qm)
  %mnv = call double @llvm.vector.reduce.fmin.v4f64(<4 x double> %mnn)
  %mxv = call double @llvm.vector.reduce.fmax.v4f64(<4 x double> %mxn)
  br label %red

red:
  %bsum = phi double [ 0.0, %entry ], [ %sv, %maindone ]
  %bsq = phi double [ 0.0, %entry ], [ %qv, %maindone ]
  %bmin = phi double [ 0x7FF0000000000000, %entry ], [ %mnv, %maindone ]
  %bmax = phi double [ 0xFFF0000000000000, %entry ], [ %mxv, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %nbulk, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jn, %body ]
  %ts = phi double [ %bsum, %red ], [ %tsn, %body ]
  %tq = phi double [ %bsq, %red ], [ %tqn, %body ]
  %tmn = phi double [ %bmin, %red ], [ %tmnn, %body ]
  %tmx = phi double [ %bmax, %red ], [ %tmxn, %body ]
  %d = icmp uge i64 %j, %n
  br i1 %d, label %fin, label %body

body:
  %tp = getelementptr inbounds nuw i64, ptr %vals, i64 %j
  %tr = load i64, ptr %tp, align 8
  %x = sitofp i64 %tr to double
  %tsn = fadd fast double %ts, %x
  %xx = fmul fast double %x, %x
  %tqn = fadd fast double %tq, %xx
  %tmnn = call double @llvm.minnum.f64(double %tmn, double %x)
  %tmxn = call double @llvm.maxnum.f64(double %tmx, double %x)
  %jn = add nuw i64 %j, 1
  br label %tail

fin:
  store double %ts, ptr %osum, align 8
  store double %tq, ptr %osq, align 8
  store double %tmn, ptr %omin, align 8
  store double %tmx, ptr %omax, align 8
  ret void
}

define internal void @df_dense_stats_f32(ptr %vals, i64 %n, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %nbulk = and i64 %n, -8
  %has = icmp uge i64 %nbulk, 8
  br i1 %has, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %s0 = phi <4 x double> [ zeroinitializer, %entry ], [ %s0n, %main ]
  %s1 = phi <4 x double> [ zeroinitializer, %entry ], [ %s1n, %main ]
  %q0 = phi <4 x double> [ zeroinitializer, %entry ], [ %q0n, %main ]
  %q1 = phi <4 x double> [ zeroinitializer, %entry ], [ %q1n, %main ]
  %mn = phi <4 x double> [ <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>, %entry ], [ %mnn, %main ]
  %mx = phi <4 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000>, %entry ], [ %mxn, %main ]
  %pa = getelementptr inbounds nuw float, ptr %vals, i64 %i
  %ra = load <4 x float>, ptr %pa, align 4
  %va = fpext <4 x float> %ra to <4 x double>
  %i4 = add nuw i64 %i, 4
  %pb = getelementptr inbounds nuw float, ptr %vals, i64 %i4
  %rb = load <4 x float>, ptr %pb, align 4
  %vb = fpext <4 x float> %rb to <4 x double>
  %s0n = fadd fast <4 x double> %s0, %va
  %s1n = fadd fast <4 x double> %s1, %vb
  %sqa = fmul fast <4 x double> %va, %va
  %q0n = fadd fast <4 x double> %q0, %sqa
  %sqb = fmul fast <4 x double> %vb, %vb
  %q1n = fadd fast <4 x double> %q1, %sqb
  %mab = call <4 x double> @llvm.minnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mnn = call <4 x double> @llvm.minnum.v4f64(<4 x double> %mn, <4 x double> %mab)
  %xab = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mxn = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %mx, <4 x double> %xab)
  %inext = add nuw i64 %i, 8
  %more = icmp ult i64 %inext, %nbulk
  br i1 %more, label %main, label %maindone

maindone:
  %sm = fadd fast <4 x double> %s0n, %s1n
  %sv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %sm)
  %qm = fadd fast <4 x double> %q0n, %q1n
  %qv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %qm)
  %mnv = call double @llvm.vector.reduce.fmin.v4f64(<4 x double> %mnn)
  %mxv = call double @llvm.vector.reduce.fmax.v4f64(<4 x double> %mxn)
  br label %red

red:
  %bsum = phi double [ 0.0, %entry ], [ %sv, %maindone ]
  %bsq = phi double [ 0.0, %entry ], [ %qv, %maindone ]
  %bmin = phi double [ 0x7FF0000000000000, %entry ], [ %mnv, %maindone ]
  %bmax = phi double [ 0xFFF0000000000000, %entry ], [ %mxv, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %nbulk, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jn, %body ]
  %ts = phi double [ %bsum, %red ], [ %tsn, %body ]
  %tq = phi double [ %bsq, %red ], [ %tqn, %body ]
  %tmn = phi double [ %bmin, %red ], [ %tmnn, %body ]
  %tmx = phi double [ %bmax, %red ], [ %tmxn, %body ]
  %d = icmp uge i64 %j, %n
  br i1 %d, label %fin, label %body

body:
  %tp = getelementptr inbounds nuw float, ptr %vals, i64 %j
  %tr = load float, ptr %tp, align 4
  %x = fpext float %tr to double
  %tsn = fadd fast double %ts, %x
  %xx = fmul fast double %x, %x
  %tqn = fadd fast double %tq, %xx
  %tmnn = call double @llvm.minnum.f64(double %tmn, double %x)
  %tmxn = call double @llvm.maxnum.f64(double %tmx, double %x)
  %jn = add nuw i64 %j, 1
  br label %tail

fin:
  store double %ts, ptr %osum, align 8
  store double %tq, ptr %osq, align 8
  store double %tmn, ptr %omin, align 8
  store double %tmx, ptr %omax, align 8
  ret void
}

define internal void @df_dense_stats_f64(ptr %vals, i64 %n, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %nbulk = and i64 %n, -8
  %has = icmp uge i64 %nbulk, 8
  br i1 %has, label %main, label %red

main:
  %i = phi i64 [ 0, %entry ], [ %inext, %main ]
  %s0 = phi <4 x double> [ zeroinitializer, %entry ], [ %s0n, %main ]
  %s1 = phi <4 x double> [ zeroinitializer, %entry ], [ %s1n, %main ]
  %q0 = phi <4 x double> [ zeroinitializer, %entry ], [ %q0n, %main ]
  %q1 = phi <4 x double> [ zeroinitializer, %entry ], [ %q1n, %main ]
  %mn = phi <4 x double> [ <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>, %entry ], [ %mnn, %main ]
  %mx = phi <4 x double> [ <double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000, double 0xFFF0000000000000>, %entry ], [ %mxn, %main ]
  %pa = getelementptr inbounds nuw double, ptr %vals, i64 %i
  %va = load <4 x double>, ptr %pa, align 8
  %i4 = add nuw i64 %i, 4
  %pb = getelementptr inbounds nuw double, ptr %vals, i64 %i4
  %vb = load <4 x double>, ptr %pb, align 8
  %s0n = fadd fast <4 x double> %s0, %va
  %s1n = fadd fast <4 x double> %s1, %vb
  %sqa = fmul fast <4 x double> %va, %va
  %q0n = fadd fast <4 x double> %q0, %sqa
  %sqb = fmul fast <4 x double> %vb, %vb
  %q1n = fadd fast <4 x double> %q1, %sqb
  %mab = call <4 x double> @llvm.minnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mnn = call <4 x double> @llvm.minnum.v4f64(<4 x double> %mn, <4 x double> %mab)
  %xab = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %va, <4 x double> %vb)
  %mxn = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %mx, <4 x double> %xab)
  %inext = add nuw i64 %i, 8
  %more = icmp ult i64 %inext, %nbulk
  br i1 %more, label %main, label %maindone

maindone:
  %sm = fadd fast <4 x double> %s0n, %s1n
  %sv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %sm)
  %qm = fadd fast <4 x double> %q0n, %q1n
  %qv = call fast double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> %qm)
  %mnv = call double @llvm.vector.reduce.fmin.v4f64(<4 x double> %mnn)
  %mxv = call double @llvm.vector.reduce.fmax.v4f64(<4 x double> %mxn)
  br label %red

red:
  %bsum = phi double [ 0.0, %entry ], [ %sv, %maindone ]
  %bsq = phi double [ 0.0, %entry ], [ %qv, %maindone ]
  %bmin = phi double [ 0x7FF0000000000000, %entry ], [ %mnv, %maindone ]
  %bmax = phi double [ 0xFFF0000000000000, %entry ], [ %mxv, %maindone ]
  %start = phi i64 [ 0, %entry ], [ %nbulk, %maindone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %red ], [ %jn, %body ]
  %ts = phi double [ %bsum, %red ], [ %tsn, %body ]
  %tq = phi double [ %bsq, %red ], [ %tqn, %body ]
  %tmn = phi double [ %bmin, %red ], [ %tmnn, %body ]
  %tmx = phi double [ %bmax, %red ], [ %tmxn, %body ]
  %d = icmp uge i64 %j, %n
  br i1 %d, label %fin, label %body

body:
  %tp = getelementptr inbounds nuw double, ptr %vals, i64 %j
  %x = load double, ptr %tp, align 8
  %tsn = fadd fast double %ts, %x
  %xx = fmul fast double %x, %x
  %tqn = fadd fast double %tq, %xx
  %tmnn = call double @llvm.minnum.f64(double %tmn, double %x)
  %tmxn = call double @llvm.maxnum.f64(double %tmx, double %x)
  %jn = add nuw i64 %j, 1
  br label %tail

fin:
  store double %ts, ptr %osum, align 8
  store double %tq, ptr %osq, align 8
  store double %tmn, ptr %omin, align 8
  store double %tmx, ptr %omax, align 8
  ret void
}

; ---------------------------------------------------------------------------
; SCALAR null-aware pass (fallback + oracle). Strict FP. Skips null lanes.
; ---------------------------------------------------------------------------
define internal void @df_scalar_all(ptr %s, i32 %dtype, i64 %len, ptr %ocnt, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %cnt = phi i64 [ 0, %entry ], [ %cntn, %cont ]
  %sum = phi double [ 0.0, %entry ], [ %sumn, %cont ]
  %sq = phi double [ 0.0, %entry ], [ %sqn, %cont ]
  %mn = phi double [ 0x7FF0000000000000, %entry ], [ %mnn, %cont ]
  %mx = phi double [ 0xFFF0000000000000, %entry ], [ %mxn, %cont ]
  %done = icmp uge i64 %i, %len
  br i1 %done, label %fin, label %chk

chk:
  %v = call i1 @df_valid_at(ptr %s, i64 %i)
  br i1 %v, label %body, label %cont

body:
  %x = call double @df_load_f64(ptr %vals, i32 %dtype, i64 %i)
  %cntb = add i64 %cnt, 1
  %sumb = fadd double %sum, %x
  %xx = fmul double %x, %x
  %sqb = fadd double %sq, %xx
  %mnb = call double @llvm.minnum.f64(double %mn, double %x)
  %mxb = call double @llvm.maxnum.f64(double %mx, double %x)
  br label %cont

cont:
  %cntn = phi i64 [ %cnt, %chk ], [ %cntb, %body ]
  %sumn = phi double [ %sum, %chk ], [ %sumb, %body ]
  %sqn = phi double [ %sq, %chk ], [ %sqb, %body ]
  %mnn = phi double [ %mn, %chk ], [ %mnb, %body ]
  %mxn = phi double [ %mx, %chk ], [ %mxb, %body ]
  %in = add nuw i64 %i, 1
  br label %loop

fin:
  store i64 %cnt, ptr %ocnt, align 8
  store double %sum, ptr %osum, align 8
  store double %sq, ptr %osq, align 8
  store double %mn, ptr %omin, align 8
  store double %mx, ptr %omax, align 8
  ret void
}

; dispatcher: fills count/sum/sumsq/min/max for a Series. err 0/1/8.
define internal i32 @df_series_stats(ptr %s, ptr %ocnt, ptr %osum, ptr %osq, ptr %omin, ptr %omax) #1 {
entry:
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %dtype = load i32, ptr %s, align 8
  %notnum = icmp ugt i32 %dtype, 4
  br i1 %notnum, label %err.arg, label %ok, !prof !0

err.arg:
  ret i32 8

ok:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  %vlp = getelementptr inbounds i8, ptr %s, i64 32
  %validity = load ptr, ptr %vlp, align 8
  %hasnull = icmp ne ptr %validity, null
  %isbool = icmp eq i32 %dtype, 4
  %usescalar = or i1 %hasnull, %isbool
  br i1 %usescalar, label %scalar, label %dense

scalar:
  call void @df_scalar_all(ptr %s, i32 %dtype, i64 %len, ptr %ocnt, ptr %osum, ptr %osq, ptr %omin, ptr %omax)
  ret i32 0

dense:
  store i64 %len, ptr %ocnt, align 8
  switch i32 %dtype, label %d3 [ i32 0, label %d0
                                 i32 1, label %d1
                                 i32 2, label %d2 ]
d0:
  call void @df_dense_stats_i32(ptr %vals, i64 %len, ptr %osum, ptr %osq, ptr %omin, ptr %omax)
  ret i32 0
d1:
  call void @df_dense_stats_i64(ptr %vals, i64 %len, ptr %osum, ptr %osq, ptr %omin, ptr %omax)
  ret i32 0
d2:
  call void @df_dense_stats_f32(ptr %vals, i64 %len, ptr %osum, ptr %osq, ptr %omin, ptr %omax)
  ret i32 0
d3:
  call void @df_dense_stats_f64(ptr %vals, i64 %len, ptr %osum, ptr %osq, ptr %omin, ptr %omax)
  ret i32 0
}

; ---------------------------------------------------------------------------
; public reductions
; ---------------------------------------------------------------------------

define i32 @universe_dataframe_sum(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %v = load double, ptr %sum, align 8
  store double %v, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_mean(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %c = load i64, ptr %cnt, align 8
  %empty = icmp eq i64 %c, 0
  br i1 %empty, label %err.empty, label %do
err.empty:
  ret i32 4
do:
  %v = load double, ptr %sum, align 8
  %cd = uitofp i64 %c to double
  %m = fdiv double %v, %cd
  store double %m, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_min(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %c = load i64, ptr %cnt, align 8
  %empty = icmp eq i64 %c, 0
  br i1 %empty, label %err.empty, label %do
err.empty:
  ret i32 4
do:
  %v = load double, ptr %mn, align 8
  store double %v, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_max(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %c = load i64, ptr %cnt, align 8
  %empty = icmp eq i64 %c, 0
  br i1 %empty, label %err.empty, label %do
err.empty:
  ret i32 4
do:
  %v = load double, ptr %mx, align 8
  store double %v, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_var(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %c = load i64, ptr %cnt, align 8
  %few = icmp ult i64 %c, 2
  br i1 %few, label %err.empty, label %do
err.empty:
  ret i32 4
do:
  %vsum = load double, ptr %sum, align 8
  %vsq = load double, ptr %sq, align 8
  %v = call double @df_variance(double %vsum, double %vsq, i64 %c)
  store double %v, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_std(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %on = icmp eq ptr %out, null
  br i1 %on, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cnt = alloca i64, align 8
  %sum = alloca double, align 8
  %sq = alloca double, align 8
  %mn = alloca double, align 8
  %mx = alloca double, align 8
  %e = call i32 @df_series_stats(ptr %s, ptr %cnt, ptr %sum, ptr %sq, ptr %mn, ptr %mx)
  %bad = icmp ne i32 %e, 0
  br i1 %bad, label %ret.e, label %ok
ret.e:
  ret i32 %e
ok:
  %c = load i64, ptr %cnt, align 8
  %few = icmp ult i64 %c, 2
  br i1 %few, label %err.empty, label %do
err.empty:
  ret i32 4
do:
  %vsum = load double, ptr %sum, align 8
  %vsq = load double, ptr %sq, align 8
  %v = call double @df_variance(double %vsum, double %vsq, i64 %c)
  %sd = call double @llvm.sqrt.f64(double %v)
  store double %sd, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_null_count_series(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out, null
  %any = or i1 %sn, %on
  br i1 %any, label %err, label %run, !prof !0
err:
  ret i32 1
run:
  %cp = getelementptr inbounds i8, ptr %s, i64 16
  %c = load i64, ptr %cp, align 8
  store i64 %c, ptr %out, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; quantile / median (sorted-copy + linear interpolation)
; ---------------------------------------------------------------------------
define i32 @universe_dataframe_quantile(ptr %s, double %q, ptr %out) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out, null
  %any = or i1 %sn, %on
  br i1 %any, label %err.null, label %chk, !prof !0
err.null:
  ret i32 1
chk:
  %dtype = load i32, ptr %s, align 8
  %notnum = icmp ugt i32 %dtype, 4
  %qlo = fcmp olt double %q, 0.0
  %qhi = fcmp ogt double %q, 1.0
  %qbad = or i1 %qlo, %qhi
  %bad = or i1 %notnum, %qbad
  br i1 %bad, label %err.arg, label %count, !prof !0
err.arg:
  ret i32 8
count:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %ncp = getelementptr inbounds i8, ptr %s, i64 16
  %nc = load i64, ptr %ncp, align 8
  %cnt = sub i64 %len, %nc
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %err.empty, label %alloc, !prof !0
err.empty:
  ret i32 4
alloc:
  %bytes = shl i64 %cnt, 3
  %buf = call ptr @malloc(i64 %bytes)
  %bn = icmp eq ptr %buf, null
  br i1 %bn, label %err.oom, label %fill, !prof !0
err.oom:
  ret i32 2
fill:
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  br label %floop
floop:
  %i = phi i64 [ 0, %fill ], [ %in, %fcont ]
  %k = phi i64 [ 0, %fill ], [ %kn, %fcont ]
  %fdone = icmp uge i64 %i, %len
  br i1 %fdone, label %sort, label %fchk
fchk:
  %v = call i1 @df_valid_at(ptr %s, i64 %i)
  br i1 %v, label %fbody, label %fcont
fbody:
  %x = call double @df_load_f64(ptr %vals, i32 %dtype, i64 %i)
  %dp = getelementptr inbounds double, ptr %buf, i64 %k
  store double %x, ptr %dp, align 8
  %kb = add nuw i64 %k, 1
  br label %fcont
fcont:
  %kn = phi i64 [ %k, %fchk ], [ %kb, %fbody ]
  %in = add nuw i64 %i, 1
  br label %floop
sort:
  %src = call i32 @universe_sort_quick(ptr %buf, i64 %cnt, i64 8, ptr @df_cmp_f64)
  %cm1 = sub i64 %cnt, 1
  %cm1d = uitofp i64 %cm1 to double
  %pos = fmul double %q, %cm1d
  %flo = call double @llvm.floor.f64(double %pos)
  %frac = fsub double %pos, %flo
  %loi = fptoui double %flo to i64
  %hoi0 = add i64 %loi, 1
  %clamph = icmp uge i64 %hoi0, %cnt
  %hoi = select i1 %clamph, i64 %cm1, i64 %hoi0
  %plo = getelementptr inbounds double, ptr %buf, i64 %loi
  %vlo = load double, ptr %plo, align 8
  %phi = getelementptr inbounds double, ptr %buf, i64 %hoi
  %vhi = load double, ptr %phi, align 8
  %diff = fsub double %vhi, %vlo
  %scaled = fmul double %diff, %frac
  %res = fadd double %vlo, %scaled
  call void @free(ptr %buf)
  store double %res, ptr %out, align 8
  ret i32 0
}

define i32 @universe_dataframe_median(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %r = call i32 @universe_dataframe_quantile(ptr %s, double 5.000000e-01, ptr %out)
  ret i32 %r
}

; ---------------------------------------------------------------------------
; n_unique — inline open-addressing hash set over canonical i64 keys
; ---------------------------------------------------------------------------
define i32 @universe_dataframe_n_unique(ptr %s, ptr %out) local_unnamed_addr #0 {
entry:
  %sn = icmp eq ptr %s, null
  %on = icmp eq ptr %out, null
  %any = or i1 %sn, %on
  br i1 %any, label %err.null, label %chk, !prof !0
err.null:
  ret i32 1
chk:
  %dtype = load i32, ptr %s, align 8
  %notnum = icmp ugt i32 %dtype, 4
  br i1 %notnum, label %err.arg, label %count, !prof !0
err.arg:
  ret i32 8
count:
  %lp = getelementptr inbounds i8, ptr %s, i64 8
  %len = load i64, ptr %lp, align 8
  %ncp = getelementptr inbounds i8, ptr %s, i64 16
  %nc = load i64, ptr %ncp, align 8
  %cnt = sub i64 %len, %nc
  %nullpresent = icmp ugt i64 %nc, 0
  %nullextra = zext i1 %nullpresent to i64
  %novals = icmp eq i64 %cnt, 0
  br i1 %novals, label %onlynull, label %build, !prof !0
onlynull:
  store i64 %nullextra, ptr %out, align 8
  ret i32 0
build:
  ; cap = next_pow2( max(16, cnt*2) )
  %want0 = shl i64 %cnt, 1
  %want = call i64 @llvm.umax.i64(i64 %want0, i64 16)
  %wm1 = sub i64 %want, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %wm1, i1 false)
  %shift = sub i64 64, %lz
  %cap = shl i64 1, %shift
  %mask = sub i64 %cap, 1
  %slotbytes = shl i64 %cap, 3
  %slots = call ptr @malloc(i64 %slotbytes)
  %sl0 = icmp eq ptr %slots, null
  br i1 %sl0, label %err.oom, label %allococc, !prof !0
err.oom:
  ret i32 2
allococc:
  %occ = call ptr @malloc(i64 %cap)
  %oc0 = icmp eq ptr %occ, null
  br i1 %oc0, label %freeslots, label %zocc, !prof !0
freeslots:
  call void @free(ptr %slots)
  ret i32 2
zocc:
  call void @llvm.memset.p0.i64(ptr %occ, i8 0, i64 %cap, i1 false)
  %vpp = getelementptr inbounds i8, ptr %s, i64 24
  %vals = load ptr, ptr %vpp, align 8
  br label %uloop
uloop:
  %i = phi i64 [ 0, %zocc ], [ %in, %ucont ]
  %dist = phi i64 [ 0, %zocc ], [ %distn, %ucont ]
  %udone = icmp uge i64 %i, %len
  br i1 %udone, label %ufin, label %uchk
uchk:
  %v = call i1 @df_valid_at(ptr %s, i64 %i)
  br i1 %v, label %ubody, label %ucontsame
ubody:
  %key = call i64 @df_key_i64(ptr %vals, i32 %dtype, i64 %i)
  %m0 = mul i64 %key, -7046029254386353131
  %m1 = lshr i64 %m0, 29
  %m2 = xor i64 %m0, %m1
  %h = and i64 %m2, %mask
  br label %probe
probe:
  %idx = phi i64 [ %h, %ubody ], [ %idxn, %pnext ]
  %op = getelementptr inbounds i8, ptr %occ, i64 %idx
  %o = load i8, ptr %op, align 1
  %isempty = icmp eq i8 %o, 0
  br i1 %isempty, label %insert, label %maybe
insert:
  store i8 1, ptr %op, align 1
  %skp = getelementptr inbounds i64, ptr %slots, i64 %idx
  store i64 %key, ptr %skp, align 8
  %distb = add i64 %dist, 1
  br label %ucont
maybe:
  %skp2 = getelementptr inbounds i64, ptr %slots, i64 %idx
  %ek = load i64, ptr %skp2, align 8
  %eq = icmp eq i64 %ek, %key
  br i1 %eq, label %ucontsame, label %pnext
pnext:
  %idxadd = add i64 %idx, 1
  %idxn = and i64 %idxadd, %mask
  br label %probe
ucontsame:
  br label %ucont
ucont:
  %distn = phi i64 [ %dist, %ucontsame ], [ %distb, %insert ]
  %in = add nuw i64 %i, 1
  br label %uloop
ufin:
  call void @free(ptr %slots)
  call void @free(ptr %occ)
  %total = add i64 %dist, %nullextra
  store i64 %total, ptr %out, align 8
  ret i32 0
}

; ---------------------------------------------------------------------------
; horizontal (row-wise across numeric columns) reductions
; ---------------------------------------------------------------------------
define ptr @universe_dataframe_sum_horizontal(ptr %df) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %run, !prof !0
fail:
  ret ptr null
run:
  %h = call i64 @universe_dataframe_height(ptr %df)
  %w = call i64 @universe_dataframe_width(ptr %df)
  %out = call ptr @universe_dataframe_series_new(i32 3, i64 %h)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %getbuf, !prof !0
getbuf:
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %outv = load ptr, ptr %ovp, align 8
  br label %cloop
cloop:
  %c = phi i64 [ 0, %getbuf ], [ %cn, %cdone ]
  %cdoneb = icmp uge i64 %c, %w
  br i1 %cdoneb, label %ret, label %col
col:
  %s = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %c)
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %cdone, label %coltype
coltype:
  %dt = load i32, ptr %s, align 8
  %skip = icmp ugt i32 %dt, 4
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
  %v = call i1 @df_valid_at(ptr %s, i64 %r)
  br i1 %v, label %rbody, label %rcont
rbody:
  %x = call double @df_load_f64(ptr %sv, i32 %dt, i64 %r)
  %op = getelementptr inbounds double, ptr %outv, i64 %r
  %cur = load double, ptr %op, align 8
  %acc = fadd double %cur, %x
  store double %acc, ptr %op, align 8
  br label %rcont
rcont:
  %rn = add nuw i64 %r, 1
  br label %rloop
cdone:
  %cn = add nuw i64 %c, 1
  br label %cloop
ret:
  ret ptr %out
}

define ptr @universe_dataframe_mean_horizontal(ptr %df) local_unnamed_addr #0 {
entry:
  %dn = icmp eq ptr %df, null
  br i1 %dn, label %fail, label %run, !prof !0
fail:
  ret ptr null
run:
  %h = call i64 @universe_dataframe_height(ptr %df)
  %w = call i64 @universe_dataframe_width(ptr %df)
  %out = call ptr @universe_dataframe_series_new(i32 3, i64 %h)
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %fail, label %alloccnt, !prof !0
alloccnt:
  %cntbytes = shl i64 %h, 3
  %cbz = icmp eq i64 %cntbytes, 0
  %cba = select i1 %cbz, i64 8, i64 %cntbytes
  %cntbuf = call ptr @malloc(i64 %cba)
  %cbn = icmp eq ptr %cntbuf, null
  br i1 %cbn, label %fail, label %getbuf, !prof !0
getbuf:
  call void @llvm.memset.p0.i64(ptr %cntbuf, i8 0, i64 %cba, i1 false)
  %ovp = getelementptr inbounds i8, ptr %out, i64 24
  %outv = load ptr, ptr %ovp, align 8
  br label %cloop
cloop:
  %c = phi i64 [ 0, %getbuf ], [ %cn, %cdone ]
  %cdoneb = icmp uge i64 %c, %w
  br i1 %cdoneb, label %finalize, label %col
col:
  %s = call ptr @universe_dataframe_select_at_idx(ptr %df, i64 %c)
  %sn = icmp eq ptr %s, null
  br i1 %sn, label %cdone, label %coltype
coltype:
  %dt = load i32, ptr %s, align 8
  %skip = icmp ugt i32 %dt, 4
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
  %v = call i1 @df_valid_at(ptr %s, i64 %r)
  br i1 %v, label %rbody, label %rcont
rbody:
  %x = call double @df_load_f64(ptr %sv, i32 %dt, i64 %r)
  %op = getelementptr inbounds double, ptr %outv, i64 %r
  %cur = load double, ptr %op, align 8
  %acc = fadd double %cur, %x
  store double %acc, ptr %op, align 8
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
  br label %fdloop
fdloop:
  %fr = phi i64 [ 0, %finalize ], [ %frn, %fdcont ]
  %fdone = icmp uge i64 %fr, %h
  br i1 %fdone, label %freecnt, label %fdbody
fdbody:
  %fcp = getelementptr inbounds i64, ptr %cntbuf, i64 %fr
  %fc = load i64, ptr %fcp, align 8
  %has = icmp ugt i64 %fc, 0
  br i1 %has, label %fdiv, label %fnull
fdiv:
  %fop = getelementptr inbounds double, ptr %outv, i64 %fr
  %fcur = load double, ptr %fop, align 8
  %fcd = uitofp i64 %fc to double
  %fm = fdiv double %fcur, %fcd
  store double %fm, ptr %fop, align 8
  br label %fdcont
fnull:
  %ig = call i32 @universe_dataframe_series_set_null(ptr %out, i64 %fr)
  br label %fdcont
fdcont:
  %frn = add nuw i64 %fr, 1
  br label %fdloop
freecnt:
  call void @free(ptr %cntbuf)
  ret ptr %out
}

attributes #0 = { nounwind }
attributes #1 = { nounwind norecurse }
attributes #2 = { alwaysinline nounwind norecurse }
attributes #3 = { alwaysinline nounwind norecurse }

!0 = !{!"branch_weights", i32 1, i32 2000}
