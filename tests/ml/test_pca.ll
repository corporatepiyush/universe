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

; Tests for universe_ml_pca_*:
;   1. Analytic 2D case: (-4,1) (-2,-1) (2,-1) (4,1), n=4 d=2, mean 0. Covariance
;      is diag(40/3, 4/3) (zero cross-term), so eigenpairs are (40/3, (1,0)) and
;      (4/3, (0,1)) — distinct and well separated. PC1 aligns with axis (1,0),
;      eigenvalues ~= 40/3 and 4/3, total var 44/3 so evr = (10/11, 1/11),
;      components orthonormal, evr sum ~= 1. transform+reconstruct (k=d=2) is
;      exact. Standalone explained_variance_ratio matches.
;   2. Axis-aligned cloud (d=3, fixed seed): huge variance on axis 0, tiny on
;      axes 1,2. PC1 ~= (+-1,0,0), eigenvalues strictly descending, evr[0] high.
;      All k=3 components orthonormal.
;   3. Error codes: n<2, d=0, k=0, k>d -> INVALID_ARG (8).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
declare float @llvm.fabs.f32(float)

declare i32 @universe_ml_pca_fit(ptr, i64, i64, i64, ptr, ptr, ptr, i64, float)
declare i32 @universe_ml_pca_transform(ptr, i64, i64, ptr, i64, ptr)
declare void @universe_ml_pca_explained_variance_ratio(ptr, i64, float, ptr)

; analytic dataset: (-4,1) (-2,-1) (2,-1) (4,1), n=4 d=2, mean=0
; covariance diag(40/3, 4/3): eigenvectors (1,0) and (0,1)
@d2 = private unnamed_addr constant [8 x float]
  [float -4.0, float 1.0, float -2.0, float -1.0,
   float 2.0, float -1.0, float 4.0, float 1.0], align 4

@m.fit0   = private unnamed_addr constant [13 x i8] c"2D fit rc=0\0A\00", align 1
@m.pc1al  = private unnamed_addr constant [17 x i8] c"2D PC1 || (1,1)\0A\00", align 1
@m.ev0    = private unnamed_addr constant [14 x i8] c"2D eig0=40/3\0A\00", align 1
@m.ev1    = private unnamed_addr constant [16 x i8] c"2D eig1 ~= 4/3\0A\00", align 1
@m.evord  = private unnamed_addr constant [15 x i8] c"2D eig0>=eig1\0A\00", align 1
@m.n0     = private unnamed_addr constant [12 x i8] c"2D |PC0|=1\0A\00", align 1
@m.n1     = private unnamed_addr constant [12 x i8] c"2D |PC1|=1\0A\00", align 1
@m.orth   = private unnamed_addr constant [14 x i8] c"2D PC0.PC1=0\0A\00", align 1
@m.evr0   = private unnamed_addr constant [18 x i8] c"2D evr0 ~= 10/11\0A\00", align 1
@m.evrs   = private unnamed_addr constant [15 x i8] c"2D evr sum<=1\0A\00", align 1
@m.evrsa  = private unnamed_addr constant [19 x i8] c"2D evr standalone\0A\00", align 1
@m.recon  = private unnamed_addr constant [16 x i8] c"2D reconstruct\0A\00", align 1

@m.afit   = private unnamed_addr constant [13 x i8] c"3D fit rc=0\0A\00", align 1
@m.apc1   = private unnamed_addr constant [14 x i8] c"3D PC1 axis0\0A\00", align 1
@m.apc1b  = private unnamed_addr constant [17 x i8] c"3D PC1 off-axis\0A\00", align 1
@m.aord   = private unnamed_addr constant [16 x i8] c"3D eig descend\0A\00", align 1
@m.aevr   = private unnamed_addr constant [14 x i8] c"3D evr0 high\0A\00", align 1
@m.aorth  = private unnamed_addr constant [16 x i8] c"3D orthonormal\0A\00", align 1

@m.en2    = private unnamed_addr constant [13 x i8] c"n<2 -> INVAL\00", align 1
@m.ed0    = private unnamed_addr constant [13 x i8] c"d=0 -> INVAL\00", align 1
@m.ek0    = private unnamed_addr constant [13 x i8] c"k=0 -> INVAL\00", align 1
@m.ekd    = private unnamed_addr constant [13 x i8] c"k>d -> INVAL\00", align 1

@pca.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.pca = private unnamed_addr constant [20 x i8] c"pca fit n=2000 d=16\00"

; --- helpers ------------------------------------------------------------------

; |a - b| < tol ?
define internal i1 @approx(float %a, float %b, float %tol) {
entry:
  %d = fsub float %a, %b
  %ad = call float @llvm.fabs.f32(float %d)
  %r = fcmp olt float %ad, %tol
  ret i1 %r
}

; dot of two length-n f32 vectors (independent reference; not the kernel)
define internal float @refdot(ptr %a, ptr %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %accn, %loop ]
  %pa = getelementptr inbounds float, ptr %a, i64 %i
  %fa = load float, ptr %pa, align 4
  %pb = getelementptr inbounds float, ptr %b, i64 %i
  %fb = load float, ptr %pb, align 4
  %m = fmul float %fa, %fb
  %accn = fadd float %acc, %m
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret float %accn
}

; --- main ---------------------------------------------------------------------
define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; outputs for the 2D analytic case
  %comp2 = call ptr @malloc(i64 16)    ; k*d = 2*2 floats
  %eig2  = call ptr @malloc(i64 8)     ; k = 2 floats
  %evr2  = call ptr @malloc(i64 8)
  %proj2 = call ptr @malloc(i64 32)    ; n*k = 4*2 floats = 8 * 4 bytes

  %rc0 = call i32 @universe_ml_pca_fit(ptr @d2, i64 4, i64 2, i64 2,
                                       ptr %comp2, ptr %eig2, ptr %evr2,
                                       i64 100, float 0x3EB0C6F7A0000000)
  %rc0ok = icmp eq i32 %rc0, 0
  call void @ut_check(i1 %rc0ok, ptr @m.fit0)

  ; PC1 aligns with axis (1,0): |PC1[0]| ~= 1, |PC1[1]| ~= 0
  %pc1x = load float, ptr %comp2, align 4
  %pc1yp = getelementptr inbounds float, ptr %comp2, i64 1
  %pc1y = load float, ptr %pc1yp, align 4
  %apc1x = call float @llvm.fabs.f32(float %pc1x)
  %apc1y = call float @llvm.fabs.f32(float %pc1y)
  %pc1xok = fcmp ogt float %apc1x, 0x3FEF800000000000  ; > 0.984375
  %pc1yok = fcmp olt float %apc1y, 0x3FB0000000000000  ; < 1/16
  %al1 = and i1 %pc1xok, %pc1yok
  call void @ut_check(i1 %al1, ptr @m.pc1al)

  ; eigenvalue 0 ~= 40/3
  %e0 = load float, ptr %eig2, align 4
  %e40 = fdiv float 40.0, 3.0
  %ale0 = call i1 @approx(float %e0, float %e40, float 0x3FB0000000000000) ; tol 1/16
  call void @ut_check(i1 %ale0, ptr @m.ev0)

  ; eigenvalue 1 ~= 4/3
  %e1p = getelementptr inbounds float, ptr %eig2, i64 1
  %e1 = load float, ptr %e1p, align 4
  %e43 = fdiv float 4.0, 3.0
  %ale1 = call i1 @approx(float %e1, float %e43, float 0x3FB0000000000000)
  call void @ut_check(i1 %ale1, ptr @m.ev1)

  ; descending
  %ord = fcmp oge float %e0, %e1
  call void @ut_check(i1 %ord, ptr @m.evord)

  ; norms of both components ~= 1
  %n0 = call float @refdot(ptr %comp2, ptr %comp2, i64 2)
  %aln0 = call i1 @approx(float %n0, float 1.0, float 0x3F50000000000000) ; tol 1/1024
  call void @ut_check(i1 %aln0, ptr @m.n0)
  %c1 = getelementptr inbounds float, ptr %comp2, i64 2
  %n1 = call float @refdot(ptr %c1, ptr %c1, i64 2)
  %aln1 = call i1 @approx(float %n1, float 1.0, float 0x3F50000000000000)
  call void @ut_check(i1 %aln1, ptr @m.n1)

  ; orthogonality PC0.PC1 ~= 0
  %o01 = call float @refdot(ptr %comp2, ptr %c1, i64 2)
  %alo = call i1 @approx(float %o01, float 0.0, float 0x3F80000000000000) ; tol 1/64
  call void @ut_check(i1 %alo, ptr @m.orth)

  ; evr[0] ~= 10/11 (40/3 out of trace 44/3)
  %r0 = load float, ptr %evr2, align 4
  %e1011 = fdiv float 40.0, 44.0
  %alr0 = call i1 @approx(float %r0, float %e1011, float 0x3F90000000000000)
  call void @ut_check(i1 %alr0, ptr @m.evr0)

  ; evr sum ~= 1 (<= 1 + tiny)
  %r1p = getelementptr inbounds float, ptr %evr2, i64 1
  %r1 = load float, ptr %r1p, align 4
  %rsum = fadd float %r0, %r1
  %rsok = fcmp ole float %rsum, 0x3FF0100000000000  ; 1.0009765625
  call void @ut_check(i1 %rsok, ptr @m.evrs)

  ; standalone explained_variance_ratio with total_var = trace = 44/3
  %tvar = fdiv float 44.0, 3.0
  %evr2b = call ptr @malloc(i64 8)
  call void @universe_ml_pca_explained_variance_ratio(ptr %eig2, i64 2,
                                                      float %tvar, ptr %evr2b)
  %rb0 = load float, ptr %evr2b, align 4
  %alrb = call i1 @approx(float %rb0, float %r0, float 0x3F50000000000000)
  call void @ut_check(i1 %alrb, ptr @m.evrsa)

  ; transform + reconstruct (k=d=2, mean=0): rec[i] = sum_m proj[i][m]*comp[m]
  %trc = call i32 @universe_ml_pca_transform(ptr @d2, i64 4, i64 2,
                                             ptr %comp2, i64 2, ptr %proj2)
  br label %rh

rh:
  %ri = phi i64 [ 0, %entry ], [ %rin, %rdone ]
  %rviol = phi i64 [ 0, %entry ], [ %rviol2, %rdone ]
  %rgo = icmp ult i64 %ri, 4
  br i1 %rgo, label %rbody, label %rfin

rbody:
  ; reconstruct feature 0 and 1
  %pbase = mul nuw i64 %ri, 2
  %p0p = getelementptr inbounds float, ptr %proj2, i64 %pbase
  %p0 = load float, ptr %p0p, align 4
  %p1i = add nuw i64 %pbase, 1
  %p1p = getelementptr inbounds float, ptr %proj2, i64 %p1i
  %p1 = load float, ptr %p1p, align 4
  ; comp rows
  %c00 = load float, ptr %comp2, align 4
  %c01p = getelementptr inbounds float, ptr %comp2, i64 1
  %c01 = load float, ptr %c01p, align 4
  %c10p = getelementptr inbounds float, ptr %comp2, i64 2
  %c10 = load float, ptr %c10p, align 4
  %c11p = getelementptr inbounds float, ptr %comp2, i64 3
  %c11 = load float, ptr %c11p, align 4
  %rec0a = fmul float %p0, %c00
  %rec0b = fmul float %p1, %c10
  %rec0 = fadd float %rec0a, %rec0b
  %rec1a = fmul float %p0, %c01
  %rec1b = fmul float %p1, %c11
  %rec1 = fadd float %rec1a, %rec1b
  ; original (mean 0 so centered == original)
  %ox0p = getelementptr inbounds [8 x float], ptr @d2, i64 0, i64 %pbase
  %ox0 = load float, ptr %ox0p, align 4
  %ox1p = getelementptr inbounds [8 x float], ptr @d2, i64 0, i64 %p1i
  %ox1 = load float, ptr %ox1p, align 4
  %rok0 = call i1 @approx(float %rec0, float %ox0, float 0x3F80000000000000)
  %rok1 = call i1 @approx(float %rec1, float %ox1, float 0x3F80000000000000)
  %rboth = and i1 %rok0, %rok1
  %rbadb = xor i1 %rboth, true
  %rinc = zext i1 %rbadb to i64
  br label %rdone

rdone:
  %rviol2 = add nuw i64 %rviol, %rinc
  %rin = add nuw i64 %ri, 1
  br label %rh

rfin:
  %reconok = icmp eq i64 %rviol, 0
  call void @ut_check(i1 %reconok, ptr @m.recon)

  ; ========================= 3D axis-aligned cloud ============================
  ; n=200, d=3. axis0 spread [-5,5], axes1,2 tiny noise [-0.125,0.125].
  %X3 = call ptr @malloc(i64 2400)   ; 200*3 floats
  %comp3 = call ptr @malloc(i64 36)  ; 3*3 floats
  %eig3  = call ptr @malloc(i64 12)
  %evr3  = call ptr @malloc(i64 12)
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8
  br label %gh

gh:
  %gi = phi i64 [ 0, %rfin ], [ %gin, %gh ]
  ; t in [-5,5]
  %rt = call i64 @ut_rand(ptr %st)
  %rtl = and i64 %rt, 65535
  %rtf = uitofp i64 %rtl to float
  %rt10 = fmul float %rtf, 10.0
  %rt10n = fdiv float %rt10, 65535.0
  %t = fsub float %rt10n, 5.0
  ; noise j in [-0.125,0.125] = r/65535*0.25 - 0.125
  %rn1 = call i64 @ut_rand(ptr %st)
  %rn1l = and i64 %rn1, 65535
  %rn1f = uitofp i64 %rn1l to float
  %rn1a = fmul float %rn1f, 0.25
  %rn1b = fdiv float %rn1a, 65535.0
  %n1v = fsub float %rn1b, 0.125
  %rn2 = call i64 @ut_rand(ptr %st)
  %rn2l = and i64 %rn2, 65535
  %rn2f = uitofp i64 %rn2l to float
  %rn2a = fmul float %rn2f, 0.25
  %rn2b = fdiv float %rn2a, 65535.0
  %n2v = fsub float %rn2b, 0.125
  %rn0 = call i64 @ut_rand(ptr %st)
  %rn0l = and i64 %rn0, 65535
  %rn0f = uitofp i64 %rn0l to float
  %rn0a = fmul float %rn0f, 0.25
  %rn0b = fdiv float %rn0a, 65535.0
  %n0v = fsub float %rn0b, 0.125
  %f0 = fadd float %t, %n0v
  %grow = mul nuw i64 %gi, 3
  %g0p = getelementptr inbounds float, ptr %X3, i64 %grow
  store float %f0, ptr %g0p, align 4
  %g1i = add nuw i64 %grow, 1
  %g1p = getelementptr inbounds float, ptr %X3, i64 %g1i
  store float %n1v, ptr %g1p, align 4
  %g2i = add nuw i64 %grow, 2
  %g2p = getelementptr inbounds float, ptr %X3, i64 %g2i
  store float %n2v, ptr %g2p, align 4
  %gin = add nuw i64 %gi, 1
  %gmore = icmp ult i64 %gin, 200
  br i1 %gmore, label %gh, label %afit

afit:
  %arc = call i32 @universe_ml_pca_fit(ptr %X3, i64 200, i64 3, i64 3,
                                       ptr %comp3, ptr %eig3, ptr %evr3,
                                       i64 200, float 0x3EB0C6F7A0000000)
  %arcok = icmp eq i32 %arc, 0
  call void @ut_check(i1 %arcok, ptr @m.afit)

  ; PC1 aligns with axis 0: |comp3[0]| ~= 1
  %ax0 = load float, ptr %comp3, align 4
  %aax0 = call float @llvm.fabs.f32(float %ax0)
  %axok = fcmp ogt float %aax0, 0x3FEF800000000000  ; > 0.984375
  call void @ut_check(i1 %axok, ptr @m.apc1)

  ; off-axis components small
  %ax1p = getelementptr inbounds float, ptr %comp3, i64 1
  %ax1 = load float, ptr %ax1p, align 4
  %aax1 = call float @llvm.fabs.f32(float %ax1)
  %ax2p = getelementptr inbounds float, ptr %comp3, i64 2
  %ax2 = load float, ptr %ax2p, align 4
  %aax2 = call float @llvm.fabs.f32(float %ax2)
  %off1 = fcmp olt float %aax1, 0x3FB0000000000000  ; < 1/16
  %off2 = fcmp olt float %aax2, 0x3FB0000000000000
  %offok = and i1 %off1, %off2
  call void @ut_check(i1 %offok, ptr @m.apc1b)

  ; eigenvalues descending
  %ae0 = load float, ptr %eig3, align 4
  %ae1p = getelementptr inbounds float, ptr %eig3, i64 1
  %ae1 = load float, ptr %ae1p, align 4
  %ae2p = getelementptr inbounds float, ptr %eig3, i64 2
  %ae2 = load float, ptr %ae2p, align 4
  %d01 = fcmp oge float %ae0, %ae1
  %d12 = fcmp oge float %ae1, %ae2
  %desc = and i1 %d01, %d12
  call void @ut_check(i1 %desc, ptr @m.aord)

  ; evr[0] high (> 0.9)
  %ar0 = load float, ptr %evr3, align 4
  %evrhi = fcmp ogt float %ar0, 0x3FECCCCCC0000000  ; > 0.9
  call void @ut_check(i1 %evrhi, ptr @m.aevr)

  ; orthonormality of all 3 components: |ci|~1, ci.cj~0
  %r0v = getelementptr inbounds float, ptr %comp3, i64 0
  %r1v = getelementptr inbounds float, ptr %comp3, i64 3
  %r2v = getelementptr inbounds float, ptr %comp3, i64 6
  %nn0 = call float @refdot(ptr %r0v, ptr %r0v, i64 3)
  %nn1 = call float @refdot(ptr %r1v, ptr %r1v, i64 3)
  %nn2 = call float @refdot(ptr %r2v, ptr %r2v, i64 3)
  %dd01 = call float @refdot(ptr %r0v, ptr %r1v, i64 3)
  %dd02 = call float @refdot(ptr %r0v, ptr %r2v, i64 3)
  %dd12 = call float @refdot(ptr %r1v, ptr %r2v, i64 3)
  %on0 = call i1 @approx(float %nn0, float 1.0, float 0x3F50000000000000)
  %on1 = call i1 @approx(float %nn1, float 1.0, float 0x3F50000000000000)
  %on2 = call i1 @approx(float %nn2, float 1.0, float 0x3F50000000000000)
  %oz01 = call i1 @approx(float %dd01, float 0.0, float 0x3F80000000000000)
  %oz02 = call i1 @approx(float %dd02, float 0.0, float 0x3F80000000000000)
  %oz12 = call i1 @approx(float %dd12, float 0.0, float 0x3F80000000000000)
  %oa = and i1 %on0, %on1
  %ob = and i1 %oa, %on2
  %oc = and i1 %ob, %oz01
  %od = and i1 %oc, %oz02
  %oe = and i1 %od, %oz12
  call void @ut_check(i1 %oe, ptr @m.aorth)

  ; ========================= error codes ======================================
  %en2 = call i32 @universe_ml_pca_fit(ptr @d2, i64 1, i64 2, i64 1,
                                       ptr %comp2, ptr %eig2, ptr null,
                                       i64 10, float 1.0)
  %en2ok = icmp eq i32 %en2, 8
  call void @ut_check(i1 %en2ok, ptr @m.en2)
  %ed0 = call i32 @universe_ml_pca_fit(ptr @d2, i64 4, i64 0, i64 1,
                                       ptr %comp2, ptr %eig2, ptr null,
                                       i64 10, float 1.0)
  %ed0ok = icmp eq i32 %ed0, 8
  call void @ut_check(i1 %ed0ok, ptr @m.ed0)
  %ek0 = call i32 @universe_ml_pca_fit(ptr @d2, i64 4, i64 2, i64 0,
                                       ptr %comp2, ptr %eig2, ptr null,
                                       i64 10, float 1.0)
  %ek0ok = icmp eq i32 %ek0, 8
  call void @ut_check(i1 %ek0ok, ptr @m.ek0)
  %ekd = call i32 @universe_ml_pca_fit(ptr @d2, i64 4, i64 2, i64 3,
                                       ptr %comp2, ptr %eig2, ptr null,
                                       i64 10, float 1.0)
  %ekdok = icmp eq i32 %ekd, 8
  call void @ut_check(i1 %ekdok, ptr @m.ekd)

  ; ========================= bench ============================================
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %fin

bench:
  ; n=2000 d=16 k=4
  %Xb = call ptr @malloc(i64 128000)   ; 2000*16 floats
  %compb = call ptr @malloc(i64 256)   ; 4*16 floats
  %eigb  = call ptr @malloc(i64 16)
  %stb = alloca i64, align 8
  store i64 1234567, ptr %stb, align 8
  br label %bgh

bgh:
  %bi = phi i64 [ 0, %bench ], [ %bin, %bgh ]
  %br = call i64 @ut_rand(ptr %stb)
  %brl = and i64 %br, 65535
  %brf = uitofp i64 %brl to float
  %bp = getelementptr inbounds float, ptr %Xb, i64 %bi
  store float %brf, ptr %bp, align 4
  %bin = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bin, 32000
  br i1 %bmore, label %bgh, label %brun

brun:
  br label %b2.rep.head

b2.rep.head:
  %b2.rep = phi i64 [ 0, %brun ], [ %b2.rep.n, %b2.rep.cont ]
  %b2.t0 = call double @ut_now_sec()
  br label %bfit

bfit:
  %bit = phi i64 [ 0, %b2.rep.head ], [ %bitn, %bfit ]
  %rcb = call i32 @universe_ml_pca_fit(ptr %Xb, i64 2000, i64 16, i64 4,
                                       ptr %compb, ptr %eigb, ptr null,
                                       i64 50, float 0x3EB0C6F7A0000000)
  %bitn = add nuw i64 %bit, 1
  %bcont = icmp ult i64 %bitn, 20
  br i1 %bcont, label %bfit, label %bdone

bdone:
  %b2.t1 = call double @ut_now_sec()
  %b2.dt = fsub double %b2.t1, %b2.t0
  %b2.warm = icmp eq i64 %b2.rep, 0
  br i1 %b2.warm, label %b2.rep.cont, label %b2.rep.store
b2.rep.store:
  %b2.idx = sub i64 %b2.rep, 1
  %b2.sp = getelementptr inbounds [16 x double], ptr @pca.samp, i64 0, i64 %b2.idx
  store double %b2.dt, ptr %b2.sp, align 8
  br label %b2.rep.cont
b2.rep.cont:
  %b2.rep.n = add i64 %b2.rep, 1
  %b2.more = icmp ult i64 %b2.rep.n, 17
  br i1 %b2.more, label %b2.rep.head, label %b2.rep.end
b2.rep.end:
  ; ops_per_rep = 20 pca fits
  call void @ut_report_dist(ptr @pca.samp, i64 16, i64 20, ptr @lbl.pca)
  call void @free(ptr %Xb)
  call void @free(ptr %compb)
  call void @free(ptr %eigb)
  br label %fin

fin:
  call void @free(ptr %comp2)
  call void @free(ptr %eig2)
  call void @free(ptr %evr2)
  call void @free(ptr %evr2b)
  call void @free(ptr %proj2)
  call void @free(ptr %X3)
  call void @free(ptr %comp3)
  call void @free(ptr %eig3)
  call void @free(ptr %evr3)
  %r = call i32 @ut_summary()
  ret i32 %r
}
