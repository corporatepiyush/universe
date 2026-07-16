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

; Tests for universe_crypto_ecdsa_p256_*: RFC 6979 A.2.5 P-256/SHA-256 vector
; (message "sample"): verify accept/reject, deterministic sign == vector (r,s),
; and sign->verify round-trip. Scalars/coords are 4-limb little-endian.

declare i32 @universe_crypto_ecdsa_p256_verify(ptr, ptr, ptr, ptr, ptr)
declare i32 @universe_crypto_ecdsa_p256_sign(ptr, ptr, ptr, ptr, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

declare void @ut_check(i1, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; SHA-256("sample") — big-endian hash bytes
@EC_HASH = private unnamed_addr constant [32 x i8] c"\af\2b\db\e1\aa\9b\6e\c1\e2\ad\e1\d6\94\f4\1f\c7\1a\83\1d\02\68\e9\89\15\62\11\3d\8a\62\ad\d1\bf"
@EC_QX = private unnamed_addr constant [32 x i8] c"\b6\9f\f2\60\2e\62\69\e6\6c\fa\61\3b\92\b8\49\c0\68\6d\35\c6\74\eb\61\c9\31\9d\5a\25\ba\d4\fe\60", align 8
@EC_QY = private unnamed_addr constant [32 x i8] c"\99\22\46\d4\94\c2\a3\77\51\9f\7e\2d\0c\b2\f1\f2\64\bc\28\56\e9\e9\1a\a4\99\bc\b8\08\10\fe\03\79", align 8
@EC_R = private unnamed_addr constant [32 x i8] c"\16\37\af\4e\a8\0e\4d\c3\91\f9\aa\56\7b\87\2c\9d\d6\81\5e\d4\9c\dd\40\11\fd\a8\b6\ac\2a\8b\d4\ef", align 8
@EC_S = private unnamed_addr constant [32 x i8] c"\a8\cd\3a\84\2f\ab\c4\4d\06\f4\af\b9\db\00\e9\f3\65\9f\e2\b6\a1\c7\36\d4\41\7c\65\2d\94\1c\cb\f7", align 8
@EC_K = private unnamed_addr constant [32 x i8] c"\60\ad\8a\3d\49\29\61\4d\f2\b0\82\33\87\aa\17\3b\4c\dd\55\83\39\38\65\08\90\be\1a\d0\7d\c5\e3\a6", align 8
@EC_D = private unnamed_addr constant [32 x i8] c"\21\67\0f\12\2b\62\8a\7b\12\9b\e8\36\db\c3\50\4e\93\d6\b1\67\57\21\5c\6b\16\75\ba\45\d8\a9\af\c9", align 8

@g.r = internal global [32 x i8] zeroinitializer, align 8
@g.s = internal global [32 x i8] zeroinitializer, align 8
@g.tr = internal global [32 x i8] zeroinitializer, align 8

@m.ver = private unnamed_addr constant [18 x i8] c"verify good sig\00\00\00"
@m.rej = private unnamed_addr constant [18 x i8] c"reject tampered\00\00\00"
@m.sr  = private unnamed_addr constant [14 x i8] c"sign r==vec\00\00\00"
@m.ss  = private unnamed_addr constant [14 x i8] c"sign s==vec\00\00\00"
@m.rt  = private unnamed_addr constant [16 x i8] c"sign->verify\00\00\00\00"
@fmt.bench = private unnamed_addr constant [30 x i8] c"bench p256 vrfy: %.0f/s\0A\00\00\00\00\00\00"
@ecdsa.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.ecdsa = private unnamed_addr constant [18 x i8] c"ecdsa-p256 verify\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; verify good signature
  %v = call i32 @universe_crypto_ecdsa_p256_verify(ptr @EC_HASH, ptr @EC_QX, ptr @EC_QY, ptr @EC_R, ptr @EC_S)
  %vok = icmp eq i32 %v, 0
  call void @ut_check(i1 %vok, ptr @m.ver)

  ; reject tampered r
  call void @llvm.memcpy.p0.p0.i64(ptr @g.tr, ptr @EC_R, i64 32, i1 false)
  %tp = getelementptr inbounds i8, ptr @g.tr, i64 0
  %tv = load i8, ptr %tp, align 1
  %tv2 = xor i8 %tv, 1
  store i8 %tv2, ptr %tp, align 1
  %rj = call i32 @universe_crypto_ecdsa_p256_verify(ptr @EC_HASH, ptr @EC_QX, ptr @EC_QY, ptr @g.tr, ptr @EC_S)
  %rjok = icmp ne i32 %rj, 0
  call void @ut_check(i1 %rjok, ptr @m.rej)

  ; deterministic sign matches vector
  %sc = call i32 @universe_crypto_ecdsa_p256_sign(ptr @EC_HASH, ptr @EC_D, ptr @EC_K, ptr @g.r, ptr @g.s)
  %crr = call i32 @memcmp(ptr @g.r, ptr @EC_R, i64 32)
  %crrok = icmp eq i32 %crr, 0
  call void @ut_check(i1 %crrok, ptr @m.sr)
  %css = call i32 @memcmp(ptr @g.s, ptr @EC_S, i64 32)
  %cssok = icmp eq i32 %css, 0
  call void @ut_check(i1 %cssok, ptr @m.ss)

  ; sign -> verify round-trip
  %v2 = call i32 @universe_crypto_ecdsa_p256_verify(ptr @EC_HASH, ptr @EC_QX, ptr @EC_QY, ptr @g.r, ptr @g.s)
  %v2ok = icmp eq i32 %v2, 0
  call void @ut_check(i1 %v2ok, ptr @m.rt)

  ; bench
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %fin
do.bench:
  br label %brep.head
brep.head:
  %brep = phi i64 [ 0, %do.bench ], [ %brep.n, %brep.tail ]
  %bt0 = call double @ut_now_sec()
  br label %b.head
b.head:
  %bi = phi i64 [ 0, %brep.head ], [ %bi.n, %b.body ]
  %bd = icmp uge i64 %bi, 100
  br i1 %bd, label %b.fin, label %b.body
b.body:
  %bv = call i32 @universe_crypto_ecdsa_p256_verify(ptr @EC_HASH, ptr @EC_QX, ptr @EC_QY, ptr @EC_R, ptr @EC_S)
  %bi.n = add i64 %bi, 1
  br label %b.head
b.fin:
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (100 verifies/rep)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore
bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @ecdsa.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail
brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report
brep.report:
  call void @ut_report_dist(ptr @ecdsa.samp, i64 16, i64 100, ptr @lbl.ecdsa)
  br label %fin
fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
