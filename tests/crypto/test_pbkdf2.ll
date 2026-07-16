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

; Tests for universe_crypto_pbkdf2_*: RFC 6070 (PBKDF2-HMAC-SHA1, iterations
; 1/2/4096, incl. multi-block dkLen=25 and an embedded-NUL password/salt),
; published PBKDF2-HMAC-SHA256 vectors (c=1 and c=4096, dkLen=32), the
; RFC 7914 PBKDF2-HMAC-SHA256 c=1 dkLen=64 multi-block vector, and --bench.

declare void @universe_crypto_pbkdf2_sha1(ptr, i64, ptr, i64, i64, ptr, i64)
declare void @universe_crypto_pbkdf2_sha256(ptr, i64, ptr, i64, i64, ptr, i64)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; ---- passwords / salts ----
@p.password = private unnamed_addr constant [8 x i8]  c"password"
@s.salt     = private unnamed_addr constant [4 x i8]  c"salt"
@p.PW       = private unnamed_addr constant [24 x i8] c"passwordPASSWORDpassword"
@s.saltSALT = private unnamed_addr constant [36 x i8] c"saltSALTsaltSALTsaltSALTsaltSALTsalt"
@p.passnul  = private unnamed_addr constant [9 x i8]  c"pass\00word"
@s.sanul    = private unnamed_addr constant [5 x i8]  c"sa\00lt"
@p.passwd   = private unnamed_addr constant [6 x i8]  c"passwd"

; ---- expected keys ----
@e.s1.c1   = private unnamed_addr constant [20 x i8] c"\0c\60\c8\0f\96\1f\0e\71\f3\a9\b5\24\af\60\12\06\2f\e0\37\a6"
@e.s1.c2   = private unnamed_addr constant [20 x i8] c"\ea\6c\01\4d\c7\2d\6f\8c\cd\1e\d9\2a\ce\1d\41\f0\d8\de\89\57"
@e.s1.c4096 = private unnamed_addr constant [20 x i8] c"\4b\00\79\01\b7\65\48\9a\be\ad\49\d9\26\f7\21\d0\65\a4\29\c1"
@e.s1.dk25 = private unnamed_addr constant [25 x i8] c"\3d\2e\ec\4f\e4\1c\84\9b\80\c8\d8\36\62\c0\e4\4a\8b\29\1a\96\4c\f2\f0\70\38"
@e.s1.nul  = private unnamed_addr constant [16 x i8] c"\56\fa\6a\a7\55\48\09\9d\cc\37\d7\f0\34\25\e0\c3"

@e.s256.c1   = private unnamed_addr constant [32 x i8] c"\12\0f\b6\cf\fc\f8\b3\2c\43\e7\22\52\56\c4\f8\37\a8\65\48\c9\2c\cc\35\48\08\05\98\7c\b7\0b\e1\7b"
@e.s256.c4096 = private unnamed_addr constant [32 x i8] c"\c5\e4\78\d5\92\88\c8\41\aa\53\0d\b6\84\5c\4c\8d\96\28\93\a0\01\ce\4e\11\a4\96\38\73\aa\98\13\4a"
@e.s256.dk64 = private unnamed_addr constant [64 x i8] c"\55\ac\04\6e\56\e3\08\9f\ec\16\91\c2\25\44\b6\05\f9\41\85\21\6d\de\04\65\e6\8b\9d\57\c2\0d\ac\bc\49\ca\9c\cc\f1\79\b6\45\99\16\64\b3\9d\77\ef\31\7c\71\b8\45\b1\e3\0b\d5\09\11\20\41\d3\a1\97\83"

@g.dk = internal global [64 x i8] zeroinitializer, align 16

@m.s1.c1   = private unnamed_addr constant [18 x i8] c"pbkdf2-sha1 c=1\00\00\00"
@m.s1.c2   = private unnamed_addr constant [16 x i8] c"pbkdf2-sha1 c=2\00"
@m.s1.c4096 = private unnamed_addr constant [19 x i8] c"pbkdf2-sha1 c=4096\00"
@m.s1.dk25 = private unnamed_addr constant [24 x i8] c"pbkdf2-sha1 dk=25 mblk\00\00"
@m.s1.nul  = private unnamed_addr constant [20 x i8] c"pbkdf2-sha1 embnul\00\00"
@m.s256.c1 = private unnamed_addr constant [18 x i8] c"pbkdf2-sha256 c=1\00"
@m.s256.c4096 = private unnamed_addr constant [21 x i8] c"pbkdf2-sha256 c=4096\00"
@m.s256.dk64 = private unnamed_addr constant [24 x i8] c"pbkdf2-sha256 dk=64 mb\00\00"
@fmt.bench = private unnamed_addr constant [41 x i8] c"bench pbkdf2-sha256 100k iters: %.2f ms\0A\00"
@pbkdf2.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.pbkdf2 = private unnamed_addr constant [19 x i8] c"pbkdf2-sha256 100k\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ===== PBKDF2-HMAC-SHA1, RFC 6070 =====
  call void @universe_crypto_pbkdf2_sha1(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 1, ptr @g.dk, i64 20)
  %a1 = call i32 @memcmp(ptr @g.dk, ptr @e.s1.c1, i64 20)
  %a1ok = icmp eq i32 %a1, 0
  call void @ut_check(i1 %a1ok, ptr @m.s1.c1)

  call void @universe_crypto_pbkdf2_sha1(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 2, ptr @g.dk, i64 20)
  %a2 = call i32 @memcmp(ptr @g.dk, ptr @e.s1.c2, i64 20)
  %a2ok = icmp eq i32 %a2, 0
  call void @ut_check(i1 %a2ok, ptr @m.s1.c2)

  call void @universe_crypto_pbkdf2_sha1(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 4096, ptr @g.dk, i64 20)
  %a3 = call i32 @memcmp(ptr @g.dk, ptr @e.s1.c4096, i64 20)
  %a3ok = icmp eq i32 %a3, 0
  call void @ut_check(i1 %a3ok, ptr @m.s1.c4096)

  ; multi-block dkLen=25, c=4096
  call void @universe_crypto_pbkdf2_sha1(ptr @p.PW, i64 24, ptr @s.saltSALT, i64 36, i64 4096, ptr @g.dk, i64 25)
  %a4 = call i32 @memcmp(ptr @g.dk, ptr @e.s1.dk25, i64 25)
  %a4ok = icmp eq i32 %a4, 0
  call void @ut_check(i1 %a4ok, ptr @m.s1.dk25)

  ; embedded-NUL password + salt, c=4096, dkLen=16
  call void @universe_crypto_pbkdf2_sha1(ptr @p.passnul, i64 9, ptr @s.sanul, i64 5, i64 4096, ptr @g.dk, i64 16)
  %a5 = call i32 @memcmp(ptr @g.dk, ptr @e.s1.nul, i64 16)
  %a5ok = icmp eq i32 %a5, 0
  call void @ut_check(i1 %a5ok, ptr @m.s1.nul)

  ; ===== PBKDF2-HMAC-SHA256 =====
  call void @universe_crypto_pbkdf2_sha256(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 1, ptr @g.dk, i64 32)
  %b1 = call i32 @memcmp(ptr @g.dk, ptr @e.s256.c1, i64 32)
  %b1ok = icmp eq i32 %b1, 0
  call void @ut_check(i1 %b1ok, ptr @m.s256.c1)

  call void @universe_crypto_pbkdf2_sha256(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 4096, ptr @g.dk, i64 32)
  %b2 = call i32 @memcmp(ptr @g.dk, ptr @e.s256.c4096, i64 32)
  %b2ok = icmp eq i32 %b2, 0
  call void @ut_check(i1 %b2ok, ptr @m.s256.c4096)

  ; RFC 7914 multi-block dkLen=64, c=1
  call void @universe_crypto_pbkdf2_sha256(ptr @p.passwd, i64 6, ptr @s.salt, i64 4, i64 1, ptr @g.dk, i64 64)
  %b3 = call i32 @memcmp(ptr @g.dk, ptr @e.s256.dk64, i64 64)
  %b3ok = icmp eq i32 %b3, 0
  call void @ut_check(i1 %b3ok, ptr @m.s256.dk64)

  ; ===== bench =====
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %brep.head

brep.head:
  %brep = phi i64 [ 0, %bench ], [ %brep.n, %brep.tail ]
  %bt0 = call double @ut_now_sec()
  call void @universe_crypto_pbkdf2_sha256(ptr @p.password, i64 8, ptr @s.salt, i64 4, i64 100000, ptr @g.dk, i64 32)
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (1 pbkdf2/rep, 100k iters)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore

bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @pbkdf2.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail

brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report

brep.report:
  call void @ut_report_dist(ptr @pbkdf2.samp, i64 16, i64 1, ptr @lbl.pbkdf2)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
