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

; Tests for universe_crypto_rsa_*: a real RSA-2048 key (n,e,d) with a published
; PKCS#1 v1.5 SHA-256 signature (independent oracle). Verify accept/reject,
; sign==vector, and encrypt->decrypt round-trip. s = 32 limbs, k = 256 bytes.

declare void @universe_crypto_rsa_pkcs1_sign_sha256(ptr, i64, ptr, ptr, i64, ptr)
declare i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr, i64, ptr, ptr, i64, ptr)
declare i32 @universe_crypto_rsa_encrypt_pkcs1(ptr, i64, ptr, ptr, i64, ptr)
declare i64 @universe_crypto_rsa_decrypt_pkcs1(ptr, ptr, ptr, i64, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@RSA_N = private unnamed_addr constant [256 x i8] c"\97\51\b6\2b\e0\f4\6b\2e\2f\1d\27\13\65\2e\eb\14\90\2d\e5\1c\e7\51\2a\50\48\a6\33\34\df\f6\61\f8\7f\18\07\c2\13\6d\51\fa\86\22\02\76\0a\f1\1d\d7\b2\e7\2c\04\e4\47\50\4a\dd\a4\22\78\fe\9d\4d\5e\6e\ac\22\ae\4c\0d\c3\15\c5\88\60\8a\55\57\5c\17\ab\b4\aa\01\6f\0c\bb\a4\7c\29\fc\cc\6f\fa\35\98\a4\7e\b4\6e\94\36\97\e1\43\1e\29\df\05\5d\f7\2c\0f\cb\d0\40\43\d7\d1\b9\1e\e0\d5\25\67\a4\39\07\2d\03\e9\7c\e7\e6\5d\7d\cc\cf\d5\3a\bb\f6\72\1a\7c\0f\5c\b5\55\3d\ac\77\a2\22\17\a5\42\7b\fc\d1\fa\29\ba\64\db\fc\74\d4\85\ec\33\f0\9f\7f\16\3a\aa\d2\44\db\67\e6\14\1a\de\55\5e\01\50\de\e8\a7\35\bb\ef\b2\60\ba\1d\bc\01\85\e2\30\b3\97\7e\8a\cb\16\83\77\59\3d\0e\86\08\5e\69\94\cf\49\86\2f\89\1c\0a\95\65\09\ce\d7\a7\8c\c7\d0\36\b8\93\84\16\85\bc\8c\3d\a3\61\30\a8\19\c6\6f\e8\aa\7a\f3", align 8
@RSA_E = private unnamed_addr constant [256 x i8] c"\01\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 8
@RSA_D = private unnamed_addr constant [256 x i8] c"\01\f7\a5\e8\bc\00\60\10\37\14\be\dd\6a\b7\34\e3\fa\e4\2a\6d\62\7f\c8\63\6f\83\8d\49\1d\cd\71\c2\e5\d4\69\e4\79\4c\e0\47\86\2f\25\75\de\e9\cc\93\fb\fb\7e\52\3f\c9\1c\e9\c8\fa\be\f1\57\17\13\60\b9\12\8d\d3\b9\c6\c1\e5\b5\6d\e7\64\ab\a6\16\45\4f\57\2a\b2\98\6a\46\11\34\ce\ae\c9\90\d0\2d\81\11\2e\14\d8\fa\9c\2e\92\45\b1\33\fb\3c\d4\d2\c9\51\93\1a\87\f5\9d\90\51\26\f3\66\6b\b1\bd\9a\d6\34\dc\32\5c\61\52\ea\f8\5f\39\90\9d\18\1f\78\c1\54\c1\25\90\5e\80\ae\3b\4d\9e\31\08\b1\b4\44\38\10\cd\0b\34\20\a6\aa\96\9b\09\d7\ed\ee\34\4f\39\a6\2f\03\69\c2\f6\2e\d6\48\bd\bf\59\4a\5f\96\c2\a0\db\c1\f2\c5\ba\f8\5c\15\da\27\ef\54\8a\a6\6b\e8\be\cb\c4\f7\c6\e6\45\30\46\08\62\0f\98\78\ee\6a\84\e4\7f\62\b2\01\5c\66\ac\62\7b\d8\df\fb\f9\35\6e\9f\c0\21\3a\1e\c2\33\31\59\7c\f5\31\f8\5f", align 8
@RSA_SIG = private unnamed_addr constant [256 x i8] c"\61\aa\45\5f\66\48\d4\da\bc\ca\cf\81\92\6b\40\b8\e1\f2\4b\2a\06\10\22\5b\db\1e\b5\ad\4e\ca\61\a8\15\7c\93\a2\dc\1e\20\b8\da\5a\2b\6e\51\c0\0a\41\ee\a6\29\f8\53\41\6e\f1\6b\50\07\d1\1f\b3\7e\fc\5d\28\4a\6a\42\f7\08\98\f9\3d\61\ba\c2\28\84\7d\e5\a9\21\c6\00\b4\3d\62\24\3a\de\a2\53\13\de\e6\19\e7\b8\83\2e\6f\67\7f\98\88\a3\f5\48\d6\31\46\c4\f4\52\04\86\7e\f6\4e\64\8e\57\c6\e2\a6\71\b8\ec\31\c6\34\63\ee\0b\1a\8a\e8\a8\f2\2b\89\b9\ac\70\5c\a6\a4\54\b3\fd\44\e8\86\6e\38\fc\90\c0\8b\71\80\12\63\aa\8d\91\d6\f7\ca\30\63\97\80\10\e0\ba\4d\65\57\99\10\69\b7\6f\0e\3c\d4\b8\d3\b8\6d\cb\0d\74\21\21\97\ae\aa\82\2d\bc\83\7d\69\fd\47\66\34\2d\bf\de\f3\c7\26\b9\6c\40\c3\79\ba\e8\75\85\ec\a4\59\a4\2b\3c\3b\4e\ee\6f\3c\65\27\a4\c0\67\b2\0b\1f\6c\23\93\f4\67\2b\9c\7b\22\1a\11\54", align 8
@RSA_MSG = private unnamed_addr constant [35 x i8] c"universe RSA PKCS1 v1.5 KAT message"

@g.sig = internal global [256 x i8] zeroinitializer, align 8
@g.ct = internal global [256 x i8] zeroinitializer, align 8
@g.pt = internal global [256 x i8] zeroinitializer, align 8
@g.tsig = internal global [256 x i8] zeroinitializer, align 8

@m.ver  = private unnamed_addr constant [20 x i8] c"verify good sig\00\00\00\00\00"
@m.sign = private unnamed_addr constant [18 x i8] c"sign == vector\00\00\00\00"
@m.rej  = private unnamed_addr constant [20 x i8] c"reject tampered\00\00\00\00\00"
@m.rtlen = private unnamed_addr constant [22 x i8] c"enc/dec len\00\00\00\00\00\00\00\00\00\00\00"
@m.rtval = private unnamed_addr constant [22 x i8] c"enc/dec bytes\00\00\00\00\00\00\00\00\00"
@m.encrc = private unnamed_addr constant [16 x i8] c"encrypt rc=0\00\00\00\00"
@fmt.bench = private unnamed_addr constant [30 x i8] c"bench rsa2048 vrfy: %.0f/s\0A\00\00\00"
@rsa.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.rsa = private unnamed_addr constant [16 x i8] c"rsa-2048 verify\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; verify good signature
  %v = call i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr @RSA_MSG, i64 35, ptr @RSA_N, ptr @RSA_E, i64 32, ptr @RSA_SIG)
  %vok = icmp eq i32 %v, 0
  call void @ut_check(i1 %vok, ptr @m.ver)

  ; sign produces the vector signature
  call void @universe_crypto_rsa_pkcs1_sign_sha256(ptr @RSA_MSG, i64 35, ptr @RSA_N, ptr @RSA_D, i64 32, ptr @g.sig)
  %sc = call i32 @memcmp(ptr @g.sig, ptr @RSA_SIG, i64 256)
  %sok = icmp eq i32 %sc, 0
  call void @ut_check(i1 %sok, ptr @m.sign)

  ; reject tampered signature
  call void @llvm.memcpy.p0.p0.i64(ptr @g.tsig, ptr @RSA_SIG, i64 256, i1 false)
  %tp = getelementptr inbounds i8, ptr @g.tsig, i64 200
  %tv = load i8, ptr %tp, align 1
  %tv2 = xor i8 %tv, 1
  store i8 %tv2, ptr %tp, align 1
  %rj = call i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr @RSA_MSG, i64 35, ptr @RSA_N, ptr @RSA_E, i64 32, ptr @g.tsig)
  %rjok = icmp ne i32 %rj, 0
  call void @ut_check(i1 %rjok, ptr @m.rej)

  ; encrypt -> decrypt round-trip (encrypt with public key, decrypt with private)
  %erc = call i32 @universe_crypto_rsa_encrypt_pkcs1(ptr @RSA_MSG, i64 35, ptr @RSA_N, ptr @RSA_E, i64 32, ptr @g.ct)
  %ercok = icmp eq i32 %erc, 0
  call void @ut_check(i1 %ercok, ptr @m.encrc)
  %mlen = call i64 @universe_crypto_rsa_decrypt_pkcs1(ptr @g.ct, ptr @RSA_N, ptr @RSA_D, i64 32, ptr @g.pt)
  call void @ut_check_eq(i64 %mlen, i64 35, ptr @m.rtlen)
  %pc = call i32 @memcmp(ptr @g.pt, ptr @RSA_MSG, i64 35)
  %pcok = icmp eq i32 %pc, 0
  call void @ut_check(i1 %pcok, ptr @m.rtval)

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
  %bd = icmp uge i64 %bi, 500
  br i1 %bd, label %b.fin, label %b.body
b.body:
  %bv = call i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr @RSA_MSG, i64 35, ptr @RSA_N, ptr @RSA_E, i64 32, ptr @RSA_SIG)
  %bi.n = add i64 %bi, 1
  br label %b.head
b.fin:
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (500 verifies/rep)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore
bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @rsa.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail
brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report
brep.report:
  call void @ut_report_dist(ptr @rsa.samp, i64 16, i64 500, ptr @lbl.rsa)
  br label %fin
fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
