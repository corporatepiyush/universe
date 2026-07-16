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

; Tests for universe_crypto_blowfish_* and universe_crypto_bcrypt_*:
;   * Blowfish ECB known-answer vectors (Eric Young / SSLeay set) verify the
;     cipher core + pi-derived key schedule independently of bcrypt.
;   * bcrypt known-answer vectors: two are the published jBCrypt $2 hashes
;     (password ""/"abc", cost 6) with the version tag rendered "$2b$" (for
;     short passwords $2a and $2b are byte-identical past the tag); two more
;     cover cost 4/5 with distinct salts. External KAT is the correctness gate.
;   * --bench times one cost-8 bcrypt.

declare void @universe_crypto_blowfish_init(ptr, ptr, i64)
declare void @universe_crypto_blowfish_encrypt(ptr, ptr, ptr)
declare void @universe_crypto_bcrypt(ptr, i64, ptr, i32, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; ---- Blowfish ECB vectors (key, plaintext, ciphertext), 8 bytes each ----
@bk1 = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\00\00"
@bp1 = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\00\00"
@bc1 = private unnamed_addr constant [8 x i8] c"\4e\f9\97\45\61\98\dd\78"
@bk2 = private unnamed_addr constant [8 x i8] c"\ff\ff\ff\ff\ff\ff\ff\ff"
@bp2 = private unnamed_addr constant [8 x i8] c"\ff\ff\ff\ff\ff\ff\ff\ff"
@bc2 = private unnamed_addr constant [8 x i8] c"\51\86\6f\d5\b8\5e\cb\8a"
@bk3 = private unnamed_addr constant [8 x i8] c"\01\23\45\67\89\ab\cd\ef"
@bp3 = private unnamed_addr constant [8 x i8] c"\11\11\11\11\11\11\11\11"
@bc3 = private unnamed_addr constant [8 x i8] c"\61\f9\c3\80\22\81\b0\96"
@bk4 = private unnamed_addr constant [8 x i8] c"\fe\dc\ba\98\76\54\32\10"
@bp4 = private unnamed_addr constant [8 x i8] c"\01\23\45\67\89\ab\cd\ef"
@bc4 = private unnamed_addr constant [8 x i8] c"\0a\ce\ab\0f\c6\a0\a2\8d"

; ---- bcrypt vectors: salt (16 raw bytes) + expected "$2b$..." (60 chars) ----
@salt1 = private unnamed_addr constant [16 x i8] c"\14\4b\3d\69\1a\7b\4e\cf\39\cf\73\5c\7f\a7\a7\9c"
@exp1  = private unnamed_addr constant [60 x i8] c"$2b$06$DCq7YPn5Rq63x1Lad4cll.TV4S6ytwfsfvkgY8jIucDrjc8deX1s."
@pw2   = private unnamed_addr constant [3 x i8]  c"abc"
@salt2 = private unnamed_addr constant [16 x i8] c"\2a\1f\1d\c7\0a\3d\14\79\56\a4\6f\eb\e3\01\60\17"
@exp2  = private unnamed_addr constant [60 x i8] c"$2b$06$If6bvum7DFjUnE9p2uDeDu0YHzrHM6tf.iqN8.yx.jNN1ILEf7h0i"
@pw3   = private unnamed_addr constant [1 x i8]  c"a"
@salt3 = private unnamed_addr constant [16 x i8] c"\00\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f"
@exp3  = private unnamed_addr constant [60 x i8] c"$2b$04$..CA.uOD/eaGAOmJB.yMBunBitU1jIt2E6xX6kPhDtSnJIjNzugbS"
@pw4   = private unnamed_addr constant [9 x i8]  c"Test1234!"
@salt4 = private unnamed_addr constant [16 x i8] c"\10\32\54\76\98\ba\dc\fe\01\23\45\67\89\ab\cd\ef"
@exp4  = private unnamed_addr constant [60 x i8] c"$2b$05$CBHSbng41N2/GyTlgYtL5uMIGo6U93/yQUsHNzKD5276hvsw2pcCC"

@g.bfctx = internal global [4168 x i8] zeroinitializer, align 16
@g.bfout = internal global [8 x i8] zeroinitializer, align 8
@g.bcout = internal global [64 x i8] zeroinitializer, align 16

@m.ecb1 = private unnamed_addr constant [12 x i8] c"blowfish e1\00"
@m.ecb2 = private unnamed_addr constant [12 x i8] c"blowfish e2\00"
@m.ecb3 = private unnamed_addr constant [12 x i8] c"blowfish e3\00"
@m.ecb4 = private unnamed_addr constant [12 x i8] c"blowfish e4\00"
@m.bc1  = private unnamed_addr constant [18 x i8] c"bcrypt pw='' c=6\00\00"
@m.bc2  = private unnamed_addr constant [18 x i8] c"bcrypt pw=abc c=6\00"
@m.bc3  = private unnamed_addr constant [16 x i8] c"bcrypt pw=a c=4\00"
@m.bc4  = private unnamed_addr constant [16 x i8] c"bcrypt Test1234\00"
@fmt.bench = private unnamed_addr constant [32 x i8] c"bench bcrypt cost8: %.2f ms\0A\00\00\00\00"
@bcrypt.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.bcrypt = private unnamed_addr constant [13 x i8] c"bcrypt cost8\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ===== Blowfish ECB KAT =====
  call void @universe_crypto_blowfish_init(ptr @g.bfctx, ptr @bk1, i64 8)
  call void @universe_crypto_blowfish_encrypt(ptr @g.bfctx, ptr @bp1, ptr @g.bfout)
  %e1 = call i32 @memcmp(ptr @g.bfout, ptr @bc1, i64 8)
  %e1ok = icmp eq i32 %e1, 0
  call void @ut_check(i1 %e1ok, ptr @m.ecb1)

  call void @universe_crypto_blowfish_init(ptr @g.bfctx, ptr @bk2, i64 8)
  call void @universe_crypto_blowfish_encrypt(ptr @g.bfctx, ptr @bp2, ptr @g.bfout)
  %e2 = call i32 @memcmp(ptr @g.bfout, ptr @bc2, i64 8)
  %e2ok = icmp eq i32 %e2, 0
  call void @ut_check(i1 %e2ok, ptr @m.ecb2)

  call void @universe_crypto_blowfish_init(ptr @g.bfctx, ptr @bk3, i64 8)
  call void @universe_crypto_blowfish_encrypt(ptr @g.bfctx, ptr @bp3, ptr @g.bfout)
  %e3 = call i32 @memcmp(ptr @g.bfout, ptr @bc3, i64 8)
  %e3ok = icmp eq i32 %e3, 0
  call void @ut_check(i1 %e3ok, ptr @m.ecb3)

  call void @universe_crypto_blowfish_init(ptr @g.bfctx, ptr @bk4, i64 8)
  call void @universe_crypto_blowfish_encrypt(ptr @g.bfctx, ptr @bp4, ptr @g.bfout)
  %e4 = call i32 @memcmp(ptr @g.bfout, ptr @bc4, i64 8)
  %e4ok = icmp eq i32 %e4, 0
  call void @ut_check(i1 %e4ok, ptr @m.ecb4)

  ; ===== bcrypt KAT =====
  call void @universe_crypto_bcrypt(ptr @exp1, i64 0, ptr @salt1, i32 6, ptr @g.bcout)
  %b1 = call i32 @memcmp(ptr @g.bcout, ptr @exp1, i64 60)
  %b1ok = icmp eq i32 %b1, 0
  call void @ut_check(i1 %b1ok, ptr @m.bc1)

  call void @universe_crypto_bcrypt(ptr @pw2, i64 3, ptr @salt2, i32 6, ptr @g.bcout)
  %b2 = call i32 @memcmp(ptr @g.bcout, ptr @exp2, i64 60)
  %b2ok = icmp eq i32 %b2, 0
  call void @ut_check(i1 %b2ok, ptr @m.bc2)

  call void @universe_crypto_bcrypt(ptr @pw3, i64 1, ptr @salt3, i32 4, ptr @g.bcout)
  %b3 = call i32 @memcmp(ptr @g.bcout, ptr @exp3, i64 60)
  %b3ok = icmp eq i32 %b3, 0
  call void @ut_check(i1 %b3ok, ptr @m.bc3)

  call void @universe_crypto_bcrypt(ptr @pw4, i64 9, ptr @salt4, i32 5, ptr @g.bcout)
  %b4 = call i32 @memcmp(ptr @g.bcout, ptr @exp4, i64 60)
  %b4ok = icmp eq i32 %b4, 0
  call void @ut_check(i1 %b4ok, ptr @m.bc4)

  ; ===== bench =====
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %brep.head

brep.head:
  %brep = phi i64 [ 0, %bench ], [ %brep.n, %brep.tail ]
  %bt0 = call double @ut_now_sec()
  call void @universe_crypto_bcrypt(ptr @pw2, i64 3, ptr @salt2, i32 8, ptr @g.bcout)
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (1 bcrypt/rep)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore

bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @bcrypt.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail

brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report

brep.report:
  call void @ut_report_dist(ptr @bcrypt.samp, i64 16, i64 1, ptr @lbl.bcrypt)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
