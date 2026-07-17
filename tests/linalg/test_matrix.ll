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

; Tests for universe_linalg_* (dense f64 row-major). Known-answer checks
; (identity, small A*B, transpose, kron, cross, trace, dot, norms), plus the
; SIMD GEMM / matvec / dot / norm cross-checked against the exported scalar
; oracles on fixed-seed random matrices at many shapes (including edge tiles
; that are NOT multiples of 4). --bench reports GEMM GFLOP/s.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
declare double @llvm.fabs.f64(double)

declare i32 @universe_linalg_matmul(ptr, i64, i64, ptr, i64, i64, ptr)
declare i32 @universe_linalg_matmul_scalar(ptr, i64, i64, ptr, i64, i64, ptr)
declare i32 @universe_linalg_matvec(ptr, i64, i64, ptr, ptr)
declare i32 @universe_linalg_matvec_scalar(ptr, i64, i64, ptr, ptr)
declare i32 @universe_linalg_mul_scalar(ptr, i64, double, ptr)
declare i32 @universe_linalg_add(ptr, ptr, i64, ptr)
declare i32 @universe_linalg_sub(ptr, ptr, i64, ptr)
declare i32 @universe_linalg_transpose(ptr, i64, i64, ptr)
declare double @universe_linalg_dot(ptr, ptr, i64)
declare double @universe_linalg_dot_scalar(ptr, ptr, i64)
declare double @universe_linalg_norm(ptr, i64, i32)
declare double @universe_linalg_norm_scalar(ptr, i64, i32)
declare double @universe_linalg_norm_fro(ptr, i64)
declare double @universe_linalg_norm_fro_scalar(ptr, i64)
declare i32 @universe_linalg_normalize(ptr, i64, ptr)
declare i32 @universe_linalg_cross(ptr, ptr, ptr)
declare double @universe_linalg_tr(ptr, i64)
declare i32 @universe_linalg_diag(ptr, i64, ptr)
declare i32 @universe_linalg_diagm(ptr, i64, ptr)
declare i32 @universe_linalg_identity(ptr, i64)
declare i32 @universe_linalg_kron(ptr, i64, i64, ptr, i64, i64, ptr)

@msg.mm    = private unnamed_addr constant [22 x i8] c"matmul vec==oracle   \00", align 1
@msg.mv    = private unnamed_addr constant [22 x i8] c"matvec vec==oracle   \00", align 1
@msg.dot   = private unnamed_addr constant [22 x i8] c"dot vec==oracle      \00", align 1
@msg.fro   = private unnamed_addr constant [22 x i8] c"norm_fro vec==oracle \00", align 1
@msg.n2    = private unnamed_addr constant [22 x i8] c"norm2 vec==oracle    \00", align 1
@msg.n1    = private unnamed_addr constant [22 x i8] c"norm1 vec==oracle    \00", align 1
@msg.ninf  = private unnamed_addr constant [22 x i8] c"norminf vec==oracle  \00", align 1
@msg.rc    = private unnamed_addr constant [16 x i8] c"matmul rc==OK  \00", align 1
@msg.kid   = private unnamed_addr constant [16 x i8] c"identity diag  \00", align 1
@msg.kid0  = private unnamed_addr constant [16 x i8] c"identity offdg \00", align 1
@msg.kmm   = private unnamed_addr constant [16 x i8] c"KAT matmul 2x2 \00", align 1
@msg.ktr   = private unnamed_addr constant [16 x i8] c"KAT transpose  \00", align 1
@msg.kkr   = private unnamed_addr constant [16 x i8] c"KAT kron       \00", align 1
@msg.kcr   = private unnamed_addr constant [16 x i8] c"KAT cross      \00", align 1
@msg.ktc   = private unnamed_addr constant [16 x i8] c"KAT trace      \00", align 1
@msg.kdt   = private unnamed_addr constant [16 x i8] c"KAT dot        \00", align 1
@msg.kn1   = private unnamed_addr constant [16 x i8] c"KAT norm1      \00", align 1
@msg.kn2   = private unnamed_addr constant [16 x i8] c"KAT norm2      \00", align 1
@msg.kni   = private unnamed_addr constant [16 x i8] c"KAT norminf    \00", align 1
@msg.kdg   = private unnamed_addr constant [16 x i8] c"KAT diag       \00", align 1
@msg.kdm   = private unnamed_addr constant [16 x i8] c"KAT diagm      \00", align 1
@msg.knz   = private unnamed_addr constant [16 x i8] c"KAT normalize  \00", align 1
@msg.kas   = private unnamed_addr constant [16 x i8] c"KAT add/sub    \00", align 1
@msg.err   = private unnamed_addr constant [16 x i8] c"err shape mism \00", align 1
@msg.enull = private unnamed_addr constant [16 x i8] c"err null ptr   \00", align 1
@msg.enz   = private unnamed_addr constant [16 x i8] c"err normalize0 \00", align 1

; shape triples (m,k,n) — includes 1x1, tall, wide, non-square, and many
; non-multiple-of-4 dims to exercise the scalar edge tiles.
@shapes = private unnamed_addr constant [17 x [3 x i64]]
  [ [3 x i64] [i64 1, i64 1, i64 1],
    [3 x i64] [i64 2, i64 3, i64 2],
    [3 x i64] [i64 3, i64 3, i64 3],
    [3 x i64] [i64 4, i64 4, i64 4],
    [3 x i64] [i64 5, i64 5, i64 5],
    [3 x i64] [i64 8, i64 8, i64 8],
    [3 x i64] [i64 5, i64 7, i64 3],
    [3 x i64] [i64 7, i64 3, i64 5],
    [3 x i64] [i64 6, i64 6, i64 2],
    [3 x i64] [i64 9, i64 4, i64 9],
    [3 x i64] [i64 1, i64 5, i64 4],
    [3 x i64] [i64 4, i64 1, i64 4],
    [3 x i64] [i64 16, i64 16, i64 16],
    [3 x i64] [i64 17, i64 3, i64 17],
    [3 x i64] [i64 10, i64 10, i64 10],
    [3 x i64] [i64 13, i64 11, i64 7],
    [3 x i64] [i64 2, i64 2, i64 32] ], align 8

@bench.fmt = private unnamed_addr constant [44 x i8] c"BENCH gemm %ldx%ldx%ld: %.2f GFLOP/s (vec)\0A\00", align 1
@bench.fs  = private unnamed_addr constant [44 x i8] c"BENCH gemm %ldx%ldx%ld: %.2f GFLOP/s (scl)\0A\00", align 1

; ============================================================ fill random arr
; value in [-1, 1): rand20/524288 - 1
define internal void @fill(ptr %a, i64 %n, ptr %st) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %low = and i64 %r, 1048575
  %rf = uitofp i64 %low to double
  %sc = fdiv double %rf, 5.242880e+05
  %v = fsub double %sc, 1.0
  %p = getelementptr inbounds nuw double, ptr %a, i64 %i
  store double %v, ptr %p, align 8
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; close(v,s): |v-s| <= (|s|+1)*1e-9
define internal i1 @close(double %v, double %s) {
entry:
  %d = fsub double %v, %s
  %ad = call double @llvm.fabs.f64(double %d)
  %as = call double @llvm.fabs.f64(double %s)
  %b0 = fadd double %as, 1.0
  %b = fmul double %b0, 1.000000e-09
  %ok = fcmp ole double %ad, %b
  ret i1 %ok
}

; count elementwise mismatches of two arrays (rel 1e-9)
define internal i64 @arrdiff(ptr %p, ptr %q, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %viol = phi i64 [ 0, %entry ], [ %violn, %loop ]
  %pp = getelementptr inbounds nuw double, ptr %p, i64 %i
  %pv = load double, ptr %pp, align 8
  %qp = getelementptr inbounds nuw double, ptr %q, i64 %i
  %qv = load double, ptr %qp, align 8
  %ok = call i1 @close(double %pv, double %qv)
  %bad = xor i1 %ok, true
  %inc = zext i1 %bad to i64
  %violn = add nuw i64 %viol, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %violn, %loop ]
  ret i64 %r
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8
  ; scratch: bench 128x128 = 131072 bytes; use 262144 for headroom.
  %A  = call ptr @malloc(i64 262144)
  %B  = call ptr @malloc(i64 262144)
  %C1 = call ptr @malloc(i64 262144)
  %C2 = call ptr @malloc(i64 262144)
  %x  = call ptr @malloc(i64 4096)
  %y1 = call ptr @malloc(i64 4096)
  %y2 = call ptr @malloc(i64 4096)

  ; ===================== KAT: identity 3x3 =====================
  call void @universe_linalg_identity(ptr %A, i64 3)
  %id0 = load double, ptr %A, align 8            ; A[0,0]=1
  %id0p = getelementptr inbounds double, ptr %A, i64 4
  %id4 = load double, ptr %id0p, align 8         ; A[1,1]=1
  %ck_id0 = fcmp oeq double %id0, 1.0
  %ck_id4 = fcmp oeq double %id4, 1.0
  %ck_idd = and i1 %ck_id0, %ck_id4
  call void @ut_check(i1 %ck_idd, ptr @msg.kid)
  %id1p = getelementptr inbounds double, ptr %A, i64 1
  %id1 = load double, ptr %id1p, align 8         ; A[0,1]=0
  %ck_id1 = fcmp oeq double %id1, 0.0
  call void @ut_check(i1 %ck_id1, ptr @msg.kid0)

  ; ===================== KAT: A(2x3)*B(3x2) =====================
  ; A = [1 2 3; 4 5 6], B = [7 8; 9 10; 11 12]
  ; C = [58 64; 139 154]
  store double 1.0, ptr %A, align 8
  %a01 = getelementptr inbounds double, ptr %A, i64 1
  store double 2.0, ptr %a01, align 8
  %a02 = getelementptr inbounds double, ptr %A, i64 2
  store double 3.0, ptr %a02, align 8
  %a03 = getelementptr inbounds double, ptr %A, i64 3
  store double 4.0, ptr %a03, align 8
  %a04 = getelementptr inbounds double, ptr %A, i64 4
  store double 5.0, ptr %a04, align 8
  %a05 = getelementptr inbounds double, ptr %A, i64 5
  store double 6.0, ptr %a05, align 8
  store double 7.0, ptr %B, align 8
  %b01 = getelementptr inbounds double, ptr %B, i64 1
  store double 8.0, ptr %b01, align 8
  %b02 = getelementptr inbounds double, ptr %B, i64 2
  store double 9.0, ptr %b02, align 8
  %b03 = getelementptr inbounds double, ptr %B, i64 3
  store double 1.000000e+01, ptr %b03, align 8
  %b04 = getelementptr inbounds double, ptr %B, i64 4
  store double 1.100000e+01, ptr %b04, align 8
  %b05 = getelementptr inbounds double, ptr %B, i64 5
  store double 1.200000e+01, ptr %b05, align 8
  %rc.mm = call i32 @universe_linalg_matmul(ptr %A, i64 2, i64 3, ptr %B, i64 3, i64 2, ptr %C1)
  %rc.ok = icmp eq i32 %rc.mm, 0
  call void @ut_check(i1 %rc.ok, ptr @msg.rc)
  %c00 = load double, ptr %C1, align 8
  %c01p = getelementptr inbounds double, ptr %C1, i64 1
  %c01 = load double, ptr %c01p, align 8
  %c10p = getelementptr inbounds double, ptr %C1, i64 2
  %c10 = load double, ptr %c10p, align 8
  %c11p = getelementptr inbounds double, ptr %C1, i64 3
  %c11 = load double, ptr %c11p, align 8
  %e00 = fcmp oeq double %c00, 5.800000e+01
  %e01 = fcmp oeq double %c01, 6.400000e+01
  %e10 = fcmp oeq double %c10, 1.390000e+02
  %e11 = fcmp oeq double %c11, 1.540000e+02
  %k0 = and i1 %e00, %e01
  %k1 = and i1 %e10, %e11
  %kmm = and i1 %k0, %k1
  call void @ut_check(i1 %kmm, ptr @msg.kmm)

  ; ===================== KAT: transpose of A(2x3) =====================
  ; A^T is 3x2: [1 4; 2 5; 3 6]
  call void @universe_linalg_transpose(ptr %A, i64 2, i64 3, ptr %C1)
  %t0 = load double, ptr %C1, align 8            ; 1
  %t1p = getelementptr inbounds double, ptr %C1, i64 1
  %t1 = load double, ptr %t1p, align 8           ; 4
  %t2p = getelementptr inbounds double, ptr %C1, i64 2
  %t2 = load double, ptr %t2p, align 8           ; 2
  %t5p = getelementptr inbounds double, ptr %C1, i64 5
  %t5 = load double, ptr %t5p, align 8           ; 6
  %tt0 = fcmp oeq double %t0, 1.0
  %tt1 = fcmp oeq double %t1, 4.0
  %tt2 = fcmp oeq double %t2, 2.0
  %tt5 = fcmp oeq double %t5, 6.0
  %ta = and i1 %tt0, %tt1
  %tb = and i1 %tt2, %tt5
  %ktr = and i1 %ta, %tb
  call void @ut_check(i1 %ktr, ptr @msg.ktr)

  ; ===================== KAT: kron 2x2 (x) 2x2 =====================
  ; A=[1 2;3 4], B=[0 5;6 7] -> 4x4
  ; row0: 0 5 0 10 ; ... check a couple.
  store double 1.0, ptr %A, align 8
  store double 2.0, ptr %a01, align 8
  store double 3.0, ptr %a02, align 8
  store double 4.0, ptr %a03, align 8
  store double 0.0, ptr %B, align 8
  store double 5.0, ptr %b01, align 8
  store double 6.0, ptr %b02, align 8
  store double 7.0, ptr %b03, align 8
  call void @universe_linalg_kron(ptr %A, i64 2, i64 2, ptr %B, i64 2, i64 2, ptr %C1)
  ; out[0,1] = A[0,0]*B[0,1] = 1*5 = 5  (index 1)
  %kr1p = getelementptr inbounds double, ptr %C1, i64 1
  %kr1 = load double, ptr %kr1p, align 8
  ; out[0,3] = A[0,1]*B[0,1] = 2*5 = 10 (index 3)
  %kr3p = getelementptr inbounds double, ptr %C1, i64 3
  %kr3 = load double, ptr %kr3p, align 8
  ; out[3,3] = A[1,1]*B[1,1] = 4*7 = 28 (index 15)
  %kr15p = getelementptr inbounds double, ptr %C1, i64 15
  %kr15 = load double, ptr %kr15p, align 8
  ; out[2,2] = A[1,0]*B[0,0] = 3*0 = 0  (index 10)
  %kr10p = getelementptr inbounds double, ptr %C1, i64 10
  %kr10 = load double, ptr %kr10p, align 8
  %kk1 = fcmp oeq double %kr1, 5.0
  %kk3 = fcmp oeq double %kr3, 1.000000e+01
  %kk15 = fcmp oeq double %kr15, 2.800000e+01
  %kk10 = fcmp oeq double %kr10, 0.0
  %ka = and i1 %kk1, %kk3
  %kb = and i1 %kk15, %kk10
  %kkr = and i1 %ka, %kb
  call void @ut_check(i1 %kkr, ptr @msg.kkr)

  ; ===================== KAT: cross [1,0,0]x[0,1,0]=[0,0,1] =====================
  store double 1.0, ptr %A, align 8
  store double 0.0, ptr %a01, align 8
  store double 0.0, ptr %a02, align 8
  store double 0.0, ptr %B, align 8
  store double 1.0, ptr %b01, align 8
  store double 0.0, ptr %b02, align 8
  call void @universe_linalg_cross(ptr %A, ptr %B, ptr %C1)
  %cr0 = load double, ptr %C1, align 8
  %cr1pp = getelementptr inbounds double, ptr %C1, i64 1
  %cr1 = load double, ptr %cr1pp, align 8
  %cr2pp = getelementptr inbounds double, ptr %C1, i64 2
  %cr2 = load double, ptr %cr2pp, align 8
  %ccr0 = fcmp oeq double %cr0, 0.0
  %ccr1 = fcmp oeq double %cr1, 0.0
  %ccr2 = fcmp oeq double %cr2, 1.0
  %cca = and i1 %ccr0, %ccr1
  %kcr = and i1 %cca, %ccr2
  call void @ut_check(i1 %kcr, ptr @msg.kcr)

  ; ===================== KAT: trace/dot/norms/diag on a fixed 3x3 =====
  ; M = [1 2 3; 4 5 6; 7 8 9], trace=15
  store double 1.0, ptr %A, align 8
  store double 2.0, ptr %a01, align 8
  store double 3.0, ptr %a02, align 8
  store double 4.0, ptr %a03, align 8
  store double 5.0, ptr %a04, align 8
  store double 6.0, ptr %a05, align 8
  %a06 = getelementptr inbounds double, ptr %A, i64 6
  store double 7.0, ptr %a06, align 8
  %a07 = getelementptr inbounds double, ptr %A, i64 7
  store double 8.0, ptr %a07, align 8
  %a08 = getelementptr inbounds double, ptr %A, i64 8
  store double 9.0, ptr %a08, align 8
  %trv = call double @universe_linalg_tr(ptr %A, i64 3)
  %ktc = fcmp oeq double %trv, 1.500000e+01
  call void @ut_check(i1 %ktc, ptr @msg.ktc)
  ; diag extract -> [1,5,9]
  call void @universe_linalg_diag(ptr %A, i64 3, ptr %C1)
  %dg0 = load double, ptr %C1, align 8
  %dg1p = getelementptr inbounds double, ptr %C1, i64 1
  %dg1 = load double, ptr %dg1p, align 8
  %dg2p = getelementptr inbounds double, ptr %C1, i64 2
  %dg2 = load double, ptr %dg2p, align 8
  %dd0 = fcmp oeq double %dg0, 1.0
  %dd1 = fcmp oeq double %dg1, 5.0
  %dd2 = fcmp oeq double %dg2, 9.0
  %dda = and i1 %dd0, %dd1
  %kdg = and i1 %dda, %dd2
  call void @ut_check(i1 %kdg, ptr @msg.kdg)
  ; diagm of [1,5,9] -> 3x3 diagonal; off-diagonal 0, [1,1]=5
  call void @universe_linalg_diagm(ptr %C1, i64 3, ptr %C2)
  %dm4p = getelementptr inbounds double, ptr %C2, i64 4
  %dm4 = load double, ptr %dm4p, align 8        ; [1,1]=5
  %dm1p = getelementptr inbounds double, ptr %C2, i64 1
  %dm1 = load double, ptr %dm1p, align 8        ; [0,1]=0
  %dmm4 = fcmp oeq double %dm4, 5.0
  %dmm1 = fcmp oeq double %dm1, 0.0
  %kdm = and i1 %dmm4, %dmm1
  call void @ut_check(i1 %kdm, ptr @msg.kdm)

  ; dot([1,2,3],[4,5,6]) = 32
  store double 1.0, ptr %x, align 8
  %x1 = getelementptr inbounds double, ptr %x, i64 1
  store double 2.0, ptr %x1, align 8
  %x2 = getelementptr inbounds double, ptr %x, i64 2
  store double 3.0, ptr %x2, align 8
  store double 4.0, ptr %y1, align 8
  %yy1 = getelementptr inbounds double, ptr %y1, i64 1
  store double 5.0, ptr %yy1, align 8
  %yy2 = getelementptr inbounds double, ptr %y1, i64 2
  store double 6.0, ptr %yy2, align 8
  %dotv = call double @universe_linalg_dot(ptr %x, ptr %y1, i64 3)
  %kdt = fcmp oeq double %dotv, 3.200000e+01
  call void @ut_check(i1 %kdt, ptr @msg.kdt)

  ; norms of x=[3,-4] : l1=7, l2=5, linf=4
  store double 3.0, ptr %x, align 8
  store double -4.0, ptr %x1, align 8
  %n1v = call double @universe_linalg_norm(ptr %x, i64 2, i32 1)
  %n2v = call double @universe_linalg_norm(ptr %x, i64 2, i32 2)
  %niv = call double @universe_linalg_norm(ptr %x, i64 2, i32 0)
  %kn1 = fcmp oeq double %n1v, 7.0
  %kn2 = fcmp oeq double %n2v, 5.0
  %kni = fcmp oeq double %niv, 4.0
  call void @ut_check(i1 %kn1, ptr @msg.kn1)
  call void @ut_check(i1 %kn2, ptr @msg.kn2)
  call void @ut_check(i1 %kni, ptr @msg.kni)

  ; normalize x=[3,4] -> [0.6,0.8], norm 1
  store double 3.0, ptr %x, align 8
  store double 4.0, ptr %x1, align 8
  %rc.nz = call i32 @universe_linalg_normalize(ptr %x, i64 2, ptr %y1)
  %nzn = call double @universe_linalg_norm(ptr %y1, i64 2, i32 2)
  %nz_close = call i1 @close(double %nzn, double 1.0)
  %nz_rc = icmp eq i32 %rc.nz, 0
  %knz = and i1 %nz_close, %nz_rc
  call void @ut_check(i1 %knz, ptr @msg.knz)
  ; normalize a zero vector -> INVALID_ARG (8)
  store double 0.0, ptr %x, align 8
  store double 0.0, ptr %x1, align 8
  %rc.z0 = call i32 @universe_linalg_normalize(ptr %x, i64 2, ptr %y1)
  %z0ok = icmp eq i32 %rc.z0, 8
  call void @ut_check(i1 %z0ok, ptr @msg.enz)

  ; add/sub: [1,2,3,4]+[10,20,30,40]=[11,22,33,44], then sub back
  store double 1.0, ptr %A, align 8
  store double 2.0, ptr %a01, align 8
  store double 3.0, ptr %a02, align 8
  store double 4.0, ptr %a03, align 8
  store double 1.000000e+01, ptr %B, align 8
  store double 2.000000e+01, ptr %b01, align 8
  store double 3.000000e+01, ptr %b02, align 8
  store double 4.000000e+01, ptr %b03, align 8
  call void @universe_linalg_add(ptr %A, ptr %B, i64 4, ptr %C1)
  %as1p = getelementptr inbounds double, ptr %C1, i64 1
  %as1 = load double, ptr %as1p, align 8         ; 22
  call void @universe_linalg_sub(ptr %C1, ptr %B, i64 4, ptr %C2)
  %sb1p = getelementptr inbounds double, ptr %C2, i64 1
  %sb1 = load double, ptr %sb1p, align 8         ; back to 2
  %as_ok = fcmp oeq double %as1, 2.200000e+01
  %sb_ok = fcmp oeq double %sb1, 2.0
  %kas = and i1 %as_ok, %sb_ok
  call void @ut_check(i1 %kas, ptr @msg.kas)

  ; ===================== error paths =====================
  ; shape mismatch k != k2
  %rc.sm = call i32 @universe_linalg_matmul(ptr %A, i64 2, i64 3, ptr %B, i64 4, i64 2, ptr %C1)
  %sm_ok = icmp eq i32 %rc.sm, 8
  call void @ut_check(i1 %sm_ok, ptr @msg.err)
  ; null pointer
  %rc.np = call i32 @universe_linalg_matmul(ptr null, i64 2, i64 3, ptr %B, i64 3, i64 2, ptr %C1)
  %np_ok = icmp eq i32 %rc.np, 1
  call void @ut_check(i1 %np_ok, ptr @msg.enull)

  ; ===================== random cross-checks over all shapes =========
  br label %sloop

sloop:
  %si = phi i64 [ 0, %entry ], [ %sin, %scont ]
  %mp = getelementptr inbounds [17 x [3 x i64]], ptr @shapes, i64 0, i64 %si, i64 0
  %m = load i64, ptr %mp, align 8
  %kp = getelementptr inbounds [17 x [3 x i64]], ptr @shapes, i64 0, i64 %si, i64 1
  %k = load i64, ptr %kp, align 8
  %np = getelementptr inbounds [17 x [3 x i64]], ptr @shapes, i64 0, i64 %si, i64 2
  %n = load i64, ptr %np, align 8
  %mk = mul i64 %m, %k
  %kn = mul i64 %k, %n
  %mn = mul i64 %m, %n
  call void @fill(ptr %A, i64 %mk, ptr %st)
  call void @fill(ptr %B, i64 %kn, ptr %st)

  ; matmul vector vs scalar oracle
  %r1 = call i32 @universe_linalg_matmul(ptr %A, i64 %m, i64 %k, ptr %B, i64 %k, i64 %n, ptr %C1)
  %r2 = call i32 @universe_linalg_matmul_scalar(ptr %A, i64 %m, i64 %k, ptr %B, i64 %k, i64 %n, ptr %C2)
  %mmd = call i64 @arrdiff(ptr %C1, ptr %C2, i64 %mn)
  %mm_ok = icmp eq i64 %mmd, 0
  call void @ut_check(i1 %mm_ok, ptr @msg.mm)

  ; matvec vs scalar: A is m x n; x is length n; fill x
  call void @fill(ptr %x, i64 %n, ptr %st)
  call void @universe_linalg_matvec(ptr %A, i64 %m, i64 %n, ptr %x, ptr %y1)
  call void @universe_linalg_matvec_scalar(ptr %A, i64 %m, i64 %n, ptr %x, ptr %y2)
  %mvd = call i64 @arrdiff(ptr %y1, ptr %y2, i64 %m)
  %mv_ok = icmp eq i64 %mvd, 0
  call void @ut_check(i1 %mv_ok, ptr @msg.mv)

  ; dot vs scalar over length kn (a decent-sized vector)
  %dv = call double @universe_linalg_dot(ptr %A, ptr %B, i64 %kn)
  %ds = call double @universe_linalg_dot_scalar(ptr %A, ptr %B, i64 %kn)
  %dv_ok = call i1 @close(double %dv, double %ds)
  call void @ut_check(i1 %dv_ok, ptr @msg.dot)

  ; frobenius vs scalar
  %fv = call double @universe_linalg_norm_fro(ptr %A, i64 %mk)
  %fs = call double @universe_linalg_norm_fro_scalar(ptr %A, i64 %mk)
  %fv_ok = call i1 @close(double %fv, double %fs)
  call void @ut_check(i1 %fv_ok, ptr @msg.fro)

  ; norm p=2,1,inf vs scalar over length mk
  %p2v = call double @universe_linalg_norm(ptr %A, i64 %mk, i32 2)
  %p2s = call double @universe_linalg_norm_scalar(ptr %A, i64 %mk, i32 2)
  %p2_ok = call i1 @close(double %p2v, double %p2s)
  call void @ut_check(i1 %p2_ok, ptr @msg.n2)
  %p1v = call double @universe_linalg_norm(ptr %A, i64 %mk, i32 1)
  %p1s = call double @universe_linalg_norm_scalar(ptr %A, i64 %mk, i32 1)
  %p1_ok = call i1 @close(double %p1v, double %p1s)
  call void @ut_check(i1 %p1_ok, ptr @msg.n1)
  %piv = call double @universe_linalg_norm(ptr %A, i64 %mk, i32 0)
  %pis = call double @universe_linalg_norm_scalar(ptr %A, i64 %mk, i32 0)
  %pi_ok = call i1 @close(double %piv, double %pis)
  call void @ut_check(i1 %pi_ok, ptr @msg.ninf)
  br label %scont

scont:
  %sin = add i64 %si, 1
  %smore = icmp ult i64 %sin, 17
  br i1 %smore, label %sloop, label %afterloop

afterloop:
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  call void @run_bench(ptr %A, ptr %B, ptr %C1, ptr %st)
  br label %fin

fin:
  call void @free(ptr %A)
  call void @free(ptr %B)
  call void @free(ptr %C1)
  call void @free(ptr %C2)
  call void @free(ptr %x)
  call void @free(ptr %y1)
  call void @free(ptr %y2)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; ============================================================ bench (128 cube)
; GEMM GFLOP/s = 2*m*n*k / seconds. Warm up, then take the best of several reps.
define internal void @run_bench(ptr %A, ptr %B, ptr %C, ptr %st) {
entry:
  %N = add i64 0, 128
  %sz = mul i64 %N, %N
  call void @fill(ptr %A, i64 %sz, ptr %st)
  call void @fill(ptr %B, i64 %sz, ptr %st)
  ; warm-up
  %w = call i32 @universe_linalg_matmul(ptr %A, i64 %N, i64 %N, ptr %B, i64 %N, i64 %N, ptr %C)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %best = phi double [ 0x7FF0000000000000, %entry ], [ %bestn, %loop ]
  %t0 = call double @ut_now_sec()
  %r = call i32 @universe_linalg_matmul(ptr %A, i64 %N, i64 %N, ptr %B, i64 %N, i64 %N, ptr %C)
  %t1 = call double @ut_now_sec()
  %dt = fsub double %t1, %t0
  %lt = fcmp olt double %dt, %best
  %bestn = select i1 %lt, double %dt, double %best
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, 50
  br i1 %more, label %loop, label %report

report:
  ; flops = 2*N^3
  %n3 = mul i64 %sz, %N
  %flops = mul i64 %n3, 2
  %flopf = uitofp i64 %flops to double
  %gf = fdiv double %flopf, %bestn
  %gflops = fdiv double %gf, 1.000000e+09
  %pr = call i32 (ptr, ...) @printf(ptr @bench.fmt, i64 %N, i64 %N, i64 %N, double %gflops)
  ; scalar oracle for comparison
  br label %sloop

sloop:
  %si = phi i64 [ 0, %report ], [ %sin, %sloop ]
  %sbest = phi double [ 0x7FF0000000000000, %report ], [ %sbestn, %sloop ]
  %st0 = call double @ut_now_sec()
  %sr = call i32 @universe_linalg_matmul_scalar(ptr %A, i64 %N, i64 %N, ptr %B, i64 %N, i64 %N, ptr %C)
  %st1 = call double @ut_now_sec()
  %sdt = fsub double %st1, %st0
  %slt = fcmp olt double %sdt, %sbest
  %sbestn = select i1 %slt, double %sdt, double %sbest
  %sin = add i64 %si, 1
  %smore = icmp ult i64 %sin, 10
  br i1 %smore, label %sloop, label %sreport

sreport:
  %sgf = fdiv double %flopf, %sbestn
  %sgflops = fdiv double %sgf, 1.000000e+09
  %spr = call i32 (ptr, ...) @printf(ptr @bench.fs, i64 %N, i64 %N, i64 %N, double %sgflops)
  ret void
}
