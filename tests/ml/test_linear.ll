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

; Tests for universe_ml_linreg_* / universe_ml_logreg_*: gradient descent on a
; noiseless linear target recovers the true weights; logistic regression on a
; linearly separable set classifies the training data; error codes fire.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare ptr @malloc(i64)
declare void @free(ptr)
declare float @llvm.fabs.f32(float)

declare i32 @universe_ml_linreg_fit(ptr, ptr, i64, i64, i64, float, ptr, ptr)
declare i32 @universe_ml_linreg_predict(ptr, i64, i64, ptr, float, ptr)
declare i32 @universe_ml_logreg_fit(ptr, ptr, i64, i64, i64, float, ptr, ptr)
declare i32 @universe_ml_logreg_predict_proba(ptr, i64, i64, ptr, float, ptr)

@m.lrc  = private unnamed_addr constant [13 x i8] c"linreg rc=0\0A\00", align 1
@m.w0   = private unnamed_addr constant [12 x i8] c"w0 approx 2\00", align 1
@m.w1   = private unnamed_addr constant [12 x i8] c"w1 approx 3\00", align 1
@m.bb   = private unnamed_addr constant [11 x i8] c"b approx 1\00", align 1
@m.pred = private unnamed_addr constant [16 x i8] c"linreg preds ok\00", align 1
@m.grc  = private unnamed_addr constant [13 x i8] c"logreg rc=0\0A\00", align 1
@m.acc  = private unnamed_addr constant [16 x i8] c"logreg accurate\00", align 1
@m.pos  = private unnamed_addr constant [14 x i8] c"proba(1,1)>.5\00", align 1
@m.neg  = private unnamed_addr constant [14 x i8] c"proba(0,0)<.5\00", align 1
@m.en   = private unnamed_addr constant [13 x i8] c"n=0 -> INVAL\00", align 1
@m.ed   = private unnamed_addr constant [13 x i8] c"d=0 -> INVAL\00", align 1
@m.epd  = private unnamed_addr constant [15 x i8] c"pred d=0 INVAL\00", align 1

; n=200, d=2

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 20260716, ptr %st, align 8
  %X = call ptr @malloc(i64 1600)     ; 200*2*4
  %y = call ptr @malloc(i64 800)      ; 200*4  (linreg targets)
  %yc = call ptr @malloc(i64 800)     ; 200*4  (logreg 0/1 labels)
  %preds = call ptr @malloc(i64 800)
  %w = call ptr @malloc(i64 8)        ; 2 floats
  %b = alloca float, align 4

  ; --- generate: x in [0,1]^2; y = 2*x0 + 3*x1 + 1 ; class = (2x0+3x1-2.5>0) ---
  br label %gen

gen:
  %i = phi i64 [ 0, %entry ], [ %in, %gen ]
  %r0 = call i64 @ut_rand(ptr %st)
  %r0l = and i64 %r0, 65535
  %x0f = uitofp i64 %r0l to float
  %x0 = fmul float %x0f, 0x3EF0000000000000   ; /65536 -> [0,1)
  %r1 = call i64 @ut_rand(ptr %st)
  %r1l = and i64 %r1, 65535
  %x1f = uitofp i64 %r1l to float
  %x1 = fmul float %x1f, 0x3EF0000000000000
  %row = mul nuw i64 %i, 2
  %xp0 = getelementptr inbounds nuw float, ptr %X, i64 %row
  store float %x0, ptr %xp0, align 4
  %row1 = add nuw i64 %row, 1
  %xp1 = getelementptr inbounds nuw float, ptr %X, i64 %row1
  store float %x1, ptr %xp1, align 4
  ; y = 2 x0 + 3 x1 + 1
  %t0 = fmul float %x0, 2.0
  %t1 = fmul float %x1, 3.0
  %t2 = fadd float %t0, %t1
  %yv = fadd float %t2, 1.0
  %yp = getelementptr inbounds nuw float, ptr %y, i64 %i
  store float %yv, ptr %yp, align 4
  ; class label: 2x0+3x1-2.5 > 0
  %s = fsub float %t2, 2.5
  %ispos = fcmp ogt float %s, 0.0
  %cls = select i1 %ispos, float 1.0, float 0.0
  %ycp = getelementptr inbounds nuw float, ptr %yc, i64 %i
  store float %cls, ptr %ycp, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, 200
  br i1 %more, label %gen, label %linreg

; ---- linear regression ----
linreg:
  %lrc = call i32 @universe_ml_linreg_fit(ptr %X, ptr %y, i64 200, i64 2, i64 8000, float 5.000000e-01, ptr %w, ptr %b)
  %lrci = sext i32 %lrc to i64
  call void @ut_check_eq(i64 %lrci, i64 0, ptr @m.lrc)
  %w0 = load float, ptr %w, align 4
  %w1p = getelementptr inbounds nuw float, ptr %w, i64 1
  %w1 = load float, ptr %w1p, align 4
  %bv = load float, ptr %b, align 4
  ; |w0-2|<0.25, |w1-3|<0.25, |b-1|<0.25
  %dw0 = fsub float %w0, 2.0
  %aw0 = call float @llvm.fabs.f32(float %dw0)
  %okw0 = fcmp olt float %aw0, 2.500000e-01
  call void @ut_check(i1 %okw0, ptr @m.w0)
  %dw1 = fsub float %w1, 3.0
  %aw1 = call float @llvm.fabs.f32(float %dw1)
  %okw1 = fcmp olt float %aw1, 2.500000e-01
  call void @ut_check(i1 %okw1, ptr @m.w1)
  %db = fsub float %bv, 1.0
  %ab = call float @llvm.fabs.f32(float %db)
  %okb = fcmp olt float %ab, 2.500000e-01
  call void @ut_check(i1 %okb, ptr @m.bb)

  ; predictions close to y
  %prc = call i32 @universe_ml_linreg_predict(ptr %X, i64 200, i64 2, ptr %w, float %bv, ptr %preds)
  br label %pl

pl:
  %pi = phi i64 [ 0, %linreg ], [ %pin, %pl ]
  %pviol = phi i64 [ 0, %linreg ], [ %pvioln, %pl ]
  %pp = getelementptr inbounds nuw float, ptr %preds, i64 %pi
  %pv = load float, ptr %pp, align 4
  %yp2 = getelementptr inbounds nuw float, ptr %y, i64 %pi
  %yv2 = load float, ptr %yp2, align 4
  %pd = fsub float %pv, %yv2
  %pad = call float @llvm.fabs.f32(float %pd)
  %pbad = fcmp ogt float %pad, 1.250000e-01
  %pinc = zext i1 %pbad to i64
  %pvioln = add nuw i64 %pviol, %pinc
  %pin = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pin, 200
  br i1 %pmore, label %pl, label %pdone

pdone:
  call void @ut_check_eq(i64 %pvioln, i64 0, ptr @m.pred)

  ; ---- logistic regression ----
  %grc = call i32 @universe_ml_logreg_fit(ptr %X, ptr %yc, i64 200, i64 2, i64 8000, float 5.000000e-01, ptr %w, ptr %b)
  %grci = sext i32 %grc to i64
  call void @ut_check_eq(i64 %grci, i64 0, ptr @m.grc)
  %gbv = load float, ptr %b, align 4
  %gprc = call i32 @universe_ml_logreg_predict_proba(ptr %X, i64 200, i64 2, ptr %w, float %gbv, ptr %preds)
  br label %al

al:
  %ai = phi i64 [ 0, %pdone ], [ %ain, %al ]
  %aviol = phi i64 [ 0, %pdone ], [ %avioln, %al ]
  %app = getelementptr inbounds nuw float, ptr %preds, i64 %ai
  %apv = load float, ptr %app, align 4
  %aycp = getelementptr inbounds nuw float, ptr %yc, i64 %ai
  %aycv = load float, ptr %aycp, align 4
  ; predicted class = proba>0.5
  %ppos = fcmp ogt float %apv, 5.000000e-01
  %pcls = select i1 %ppos, float 1.0, float 0.0
  %amis = fcmp one float %pcls, %aycv
  %ainc = zext i1 %amis to i64
  %avioln = add nuw i64 %aviol, %ainc
  %ain = add nuw i64 %ai, 1
  %amore = icmp ult i64 %ain, 200
  br i1 %amore, label %al, label %adone

adone:
  ; allow up to 20 misclassifications (near-boundary points)
  %accok = icmp ule i64 %avioln, 20
  call void @ut_check(i1 %accok, ptr @m.acc)

  ; strong positive / negative queries
  %qbuf = call ptr @malloc(i64 16)   ; 2 rows x 2
  store float 1.0, ptr %qbuf, align 4
  %q0b = getelementptr inbounds nuw float, ptr %qbuf, i64 1
  store float 1.0, ptr %q0b, align 4
  %q1a = getelementptr inbounds nuw float, ptr %qbuf, i64 2
  store float 0.0, ptr %q1a, align 4
  %q1b = getelementptr inbounds nuw float, ptr %qbuf, i64 3
  store float 0.0, ptr %q1b, align 4
  %qout = call ptr @malloc(i64 8)
  %qrc = call i32 @universe_ml_logreg_predict_proba(ptr %qbuf, i64 2, i64 2, ptr %w, float %gbv, ptr %qout)
  %qp0 = load float, ptr %qout, align 4
  %qpos = fcmp ogt float %qp0, 5.000000e-01
  call void @ut_check(i1 %qpos, ptr @m.pos)
  %qp1p = getelementptr inbounds nuw float, ptr %qout, i64 1
  %qp1 = load float, ptr %qp1p, align 4
  %qneg = fcmp olt float %qp1, 5.000000e-01
  call void @ut_check(i1 %qneg, ptr @m.neg)

  ; ---- error codes ----
  %en = call i32 @universe_ml_linreg_fit(ptr %X, ptr %y, i64 0, i64 2, i64 10, float 1.0, ptr %w, ptr %b)
  %eni = sext i32 %en to i64
  call void @ut_check_eq(i64 %eni, i64 8, ptr @m.en)
  %ed = call i32 @universe_ml_linreg_fit(ptr %X, ptr %y, i64 200, i64 0, i64 10, float 1.0, ptr %w, ptr %b)
  %edi = sext i32 %ed to i64
  call void @ut_check_eq(i64 %edi, i64 8, ptr @m.ed)
  %epd = call i32 @universe_ml_linreg_predict(ptr %X, i64 200, i64 0, ptr %w, float 0.0, ptr %preds)
  %epdi = sext i32 %epd to i64
  call void @ut_check_eq(i64 %epdi, i64 8, ptr @m.epd)

  call void @free(ptr %X)
  call void @free(ptr %y)
  call void @free(ptr %yc)
  call void @free(ptr %preds)
  call void @free(ptr %w)
  call void @free(ptr %qbuf)
  call void @free(ptr %qout)
  %rcs = call i32 @ut_summary()
  ret i32 %rcs
}
