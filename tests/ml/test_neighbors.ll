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

; Tests for universe_ml_knn_*: a tiny two-class labeled set with obvious
; nearest neighbors -> correct class (k=1 and k=3); regression returns the
; mean of the k nearest targets; error codes fire.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare float @llvm.fabs.f32(float)

declare i32 @universe_ml_knn_classify(ptr, ptr, i64, i64, ptr, i64, i64, i64, ptr)
declare i32 @universe_ml_knn_regress(ptr, ptr, i64, i64, ptr, i64, i64, ptr)

; training set: n=6, d=2. class 0 near origin, class 1 near (10,10).
@Xtrain = private unnamed_addr constant [12 x float]
  [float 0.0, float 0.0, float 1.0, float 0.0, float 0.0, float 1.0,
   float 10.0, float 10.0, float 11.0, float 10.0, float 10.0, float 11.0], align 4
@ytrain  = private unnamed_addr constant [6 x i32] [i32 0, i32 0, i32 0, i32 1, i32 1, i32 1], align 4
@ytrainf = private unnamed_addr constant [6 x float]
  [float 1.0, float 2.0, float 3.0, float 10.0, float 11.0, float 12.0], align 4
; queries: (0.25,0.25)->c0, (10.5,10.25)->c1, (0,0)->c0
@Xq = private unnamed_addr constant [6 x float]
  [float 0.25, float 0.25, float 10.5, float 10.25, float 0.0, float 0.0], align 4

@m.c1_0 = private unnamed_addr constant [15 x i8] c"k1 classify q0\00", align 1
@m.c1_1 = private unnamed_addr constant [15 x i8] c"k1 classify q1\00", align 1
@m.c1_2 = private unnamed_addr constant [15 x i8] c"k1 classify q2\00", align 1
@m.c3_0 = private unnamed_addr constant [15 x i8] c"k3 classify q0\00", align 1
@m.c3_1 = private unnamed_addr constant [15 x i8] c"k3 classify q1\00", align 1
@m.c3_2 = private unnamed_addr constant [15 x i8] c"k3 classify q2\00", align 1
@m.r1_0 = private unnamed_addr constant [15 x i8] c"k1 regress  q0\00", align 1
@m.r1_1 = private unnamed_addr constant [15 x i8] c"k1 regress  q1\00", align 1
@m.r3_0 = private unnamed_addr constant [15 x i8] c"k3 regress  q0\00", align 1
@m.r3_1 = private unnamed_addr constant [15 x i8] c"k3 regress  q1\00", align 1
@m.r3_2 = private unnamed_addr constant [15 x i8] c"k3 regress  q2\00", align 1
@m.crc  = private unnamed_addr constant [13 x i8] c"classify rc0\00", align 1
@m.rrc  = private unnamed_addr constant [13 x i8] c"regress  rc0\00", align 1
@m.ek0  = private unnamed_addr constant [11 x i8] c"k=0 INVAL\0A\00", align 1
@m.ed0  = private unnamed_addr constant [11 x i8] c"d=0 INVAL\0A\00", align 1
@m.ekn  = private unnamed_addr constant [11 x i8] c"k>n INVAL\0A\00", align 1
@m.enc  = private unnamed_addr constant [14 x i8] c"ncls=0 INVAL\0A\00", align 1

define internal void @ck(float %v, float %s, ptr %msg) {
entry:
  %d = fsub float %v, %s
  %ad = call float @llvm.fabs.f32(float %d)
  %ok = fcmp ole float %ad, 0x3F1A36E2E0000000
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %co = alloca [3 x i32], align 4
  %ro = alloca [3 x float], align 4

  ; ---- classify k=1 ----
  %rc1 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 2, ptr @Xq, i64 3, i64 1, i64 2, ptr %co)
  %rc1i = sext i32 %rc1 to i64
  call void @ut_check_eq(i64 %rc1i, i64 0, ptr @m.crc)
  %c1p0 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 0
  %c1v0 = load i32, ptr %c1p0, align 4
  %c1v0e = sext i32 %c1v0 to i64
  call void @ut_check_eq(i64 %c1v0e, i64 0, ptr @m.c1_0)
  %c1p1 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 1
  %c1v1 = load i32, ptr %c1p1, align 4
  %c1v1e = sext i32 %c1v1 to i64
  call void @ut_check_eq(i64 %c1v1e, i64 1, ptr @m.c1_1)
  %c1p2 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 2
  %c1v2 = load i32, ptr %c1p2, align 4
  %c1v2e = sext i32 %c1v2 to i64
  call void @ut_check_eq(i64 %c1v2e, i64 0, ptr @m.c1_2)

  ; ---- classify k=3 ----
  %rc3 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 2, ptr @Xq, i64 3, i64 3, i64 2, ptr %co)
  %c3p0 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 0
  %c3v0 = load i32, ptr %c3p0, align 4
  %c3v0e = sext i32 %c3v0 to i64
  call void @ut_check_eq(i64 %c3v0e, i64 0, ptr @m.c3_0)
  %c3p1 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 1
  %c3v1 = load i32, ptr %c3p1, align 4
  %c3v1e = sext i32 %c3v1 to i64
  call void @ut_check_eq(i64 %c3v1e, i64 1, ptr @m.c3_1)
  %c3p2 = getelementptr inbounds nuw [3 x i32], ptr %co, i64 0, i64 2
  %c3v2 = load i32, ptr %c3p2, align 4
  %c3v2e = sext i32 %c3v2 to i64
  call void @ut_check_eq(i64 %c3v2e, i64 0, ptr @m.c3_2)

  ; ---- regress k=1 : nearest target ----
  %rr1 = call i32 @universe_ml_knn_regress(ptr @Xtrain, ptr @ytrainf, i64 6, i64 2, ptr @Xq, i64 3, i64 1, ptr %ro)
  %rr1i = sext i32 %rr1 to i64
  call void @ut_check_eq(i64 %rr1i, i64 0, ptr @m.rrc)
  %r1p0 = getelementptr inbounds nuw [3 x float], ptr %ro, i64 0, i64 0
  %r1v0 = load float, ptr %r1p0, align 4
  call void @ck(float %r1v0, float 1.0, ptr @m.r1_0)
  %r1p1 = getelementptr inbounds nuw [3 x float], ptr %ro, i64 0, i64 1
  %r1v1 = load float, ptr %r1p1, align 4
  call void @ck(float %r1v1, float 10.0, ptr @m.r1_1)

  ; ---- regress k=3 : mean of 3 nearest targets ----
  %rr3 = call i32 @universe_ml_knn_regress(ptr @Xtrain, ptr @ytrainf, i64 6, i64 2, ptr @Xq, i64 3, i64 3, ptr %ro)
  %r3p0 = getelementptr inbounds nuw [3 x float], ptr %ro, i64 0, i64 0
  %r3v0 = load float, ptr %r3p0, align 4
  call void @ck(float %r3v0, float 2.0, ptr @m.r3_0)
  %r3p1 = getelementptr inbounds nuw [3 x float], ptr %ro, i64 0, i64 1
  %r3v1 = load float, ptr %r3p1, align 4
  call void @ck(float %r3v1, float 11.0, ptr @m.r3_1)
  %r3p2 = getelementptr inbounds nuw [3 x float], ptr %ro, i64 0, i64 2
  %r3v2 = load float, ptr %r3p2, align 4
  call void @ck(float %r3v2, float 2.0, ptr @m.r3_2)

  ; ---- error codes ----
  %e1 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 2, ptr @Xq, i64 3, i64 0, i64 2, ptr %co)
  %e1i = sext i32 %e1 to i64
  call void @ut_check_eq(i64 %e1i, i64 8, ptr @m.ek0)
  %e2 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 0, ptr @Xq, i64 3, i64 1, i64 2, ptr %co)
  %e2i = sext i32 %e2 to i64
  call void @ut_check_eq(i64 %e2i, i64 8, ptr @m.ed0)
  %e3 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 2, ptr @Xq, i64 3, i64 99, i64 2, ptr %co)
  %e3i = sext i32 %e3 to i64
  call void @ut_check_eq(i64 %e3i, i64 8, ptr @m.ekn)
  %e4 = call i32 @universe_ml_knn_classify(ptr @Xtrain, ptr @ytrain, i64 6, i64 2, ptr @Xq, i64 3, i64 1, i64 0, ptr %co)
  %e4i = sext i32 %e4 to i64
  call void @ut_check_eq(i64 %e4i, i64 8, ptr @m.enc)
  %e5 = call i32 @universe_ml_knn_regress(ptr @Xtrain, ptr @ytrainf, i64 6, i64 2, ptr @Xq, i64 3, i64 0, ptr %ro)
  %e5i = sext i32 %e5 to i64
  call void @ut_check_eq(i64 %e5i, i64 8, ptr @m.ek0)

  %rcs = call i32 @ut_summary()
  ret i32 %rcs
}
