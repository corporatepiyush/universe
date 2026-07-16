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

; Tests for universe_crypto_hmac_*: published known-answer vectors —
; RFC 4231 (HMAC-SHA256 / HMAC-SHA512 TC1,TC2, plus the >block-size-key TC6
; for SHA256), RFC 2202 (HMAC-SHA1 TC1,TC2, and the 80-byte-key TC6), the
; well-known empty-key/empty-message HMAC-SHA256 value, streaming==one-shot
; over odd chunkings, and --bench.

declare void @universe_crypto_hmac_sha1(ptr, i64, ptr, i64, ptr)
declare void @universe_crypto_hmac_sha256(ptr, i64, ptr, i64, ptr)
declare void @universe_crypto_hmac_sha512(ptr, i64, ptr, i64, ptr)
declare void @universe_crypto_hmac_sha256_init(ptr, ptr, i64)
declare void @universe_crypto_hmac_update(ptr, ptr, i64)
declare void @universe_crypto_hmac_final(ptr, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; ---- message data ----
@d.hi   = private unnamed_addr constant [8 x i8]  c"Hi There"
@d.jefe = private unnamed_addr constant [4 x i8]  c"Jefe"
@d.what = private unnamed_addr constant [28 x i8] c"what do ya want for nothing?"
@d.big  = private unnamed_addr constant [54 x i8] c"Test Using Larger Than Block-Size Key - Hash Key First"

; ---- expected digests (RFC 4231 / RFC 2202) ----
@e.s256.tc1 = private unnamed_addr constant [32 x i8] c"\b0\34\4c\61\d8\db\38\53\5c\a8\af\ce\af\0b\f1\2b\88\1d\c2\00\c9\83\3d\a7\26\e9\37\6c\2e\32\cf\f7"
@e.s256.tc2 = private unnamed_addr constant [32 x i8] c"\5b\dc\c1\46\bf\60\75\4e\6a\04\24\26\08\95\75\c7\5a\00\3f\08\9d\27\39\83\9d\ec\58\b9\64\ec\38\43"
@e.s256.tc6 = private unnamed_addr constant [32 x i8] c"\60\e4\31\59\1e\e0\b6\7f\0d\8a\26\aa\cb\f5\b7\7f\8e\0b\c6\21\37\28\c5\14\05\46\04\0f\0e\e3\7f\54"
@e.s256.empty = private unnamed_addr constant [32 x i8] c"\b6\13\67\9a\08\14\d9\ec\77\2f\95\d7\78\c3\5f\c5\ff\16\97\c4\93\71\56\53\c6\c7\12\14\42\92\c5\ad"

@e.s512.tc1 = private unnamed_addr constant [64 x i8] c"\87\aa\7c\de\a5\ef\61\9d\4f\f0\b4\24\1a\1d\6c\b0\23\79\f4\e2\ce\4e\c2\78\7a\d0\b3\05\45\e1\7c\de\da\a8\33\b7\d6\b8\a7\02\03\8b\27\4e\ae\a3\f4\e4\be\9d\91\4e\eb\61\f1\70\2e\69\6c\20\3a\12\68\54"
@e.s512.tc2 = private unnamed_addr constant [64 x i8] c"\16\4b\7a\7b\fc\f8\19\e2\e3\95\fb\e7\3b\56\e0\a3\87\bd\64\22\2e\83\1f\d6\10\27\0c\d7\ea\25\05\54\97\58\bf\75\c0\5a\99\4a\6d\03\4f\65\f8\f0\e6\fd\ca\ea\b1\a3\4d\4a\6b\4b\63\6e\07\0a\38\bc\e7\37"

@e.s1.tc1 = private unnamed_addr constant [20 x i8] c"\b6\17\31\86\55\05\72\64\e2\8b\c0\b6\fb\37\8c\8e\f1\46\be\00"
@e.s1.tc2 = private unnamed_addr constant [20 x i8] c"\ef\fc\df\6a\e5\eb\2f\a2\d2\74\16\d5\f1\84\df\9c\25\9a\7c\79"
@e.s1.tc6 = private unnamed_addr constant [20 x i8] c"\aa\4a\e5\e1\52\72\d0\0e\95\70\56\37\ce\8a\3b\55\ed\40\21\12"

; ---- scratch ----
@g.key  = internal global [1024 x i8] zeroinitializer, align 16
@g.out  = internal global [64 x i8] zeroinitializer, align 16
@g.out2 = internal global [64 x i8] zeroinitializer, align 16
@g.hctx = internal global [440 x i8] zeroinitializer, align 16

; ---- messages ----
@m.s256.tc1 = private unnamed_addr constant [16 x i8] c"hmac-sha256 tc1\00"
@m.s256.tc2 = private unnamed_addr constant [16 x i8] c"hmac-sha256 tc2\00"
@m.s256.tc6 = private unnamed_addr constant [21 x i8] c"hmac-sha256 tc6 lkey\00"
@m.s256.emp = private unnamed_addr constant [18 x i8] c"hmac-sha256 empty\00"
@m.s512.tc1 = private unnamed_addr constant [16 x i8] c"hmac-sha512 tc1\00"
@m.s512.tc2 = private unnamed_addr constant [16 x i8] c"hmac-sha512 tc2\00"
@m.s1.tc1   = private unnamed_addr constant [14 x i8] c"hmac-sha1 tc1\00"
@m.s1.tc2   = private unnamed_addr constant [14 x i8] c"hmac-sha1 tc2\00"
@m.s1.tc6   = private unnamed_addr constant [19 x i8] c"hmac-sha1 tc6 lkey\00"
@m.stream   = private unnamed_addr constant [22 x i8] c"hmac stream==oneshot\00\00"
@fmt.bench  = private unnamed_addr constant [36 x i8] c"bench hmac-sha256 1KiB: %.2f MB/s\0A\00\00"
@hmac.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.hmac = private unnamed_addr constant [28 x i8] c"hmac-sha256 1KiB (per byte)\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ===== HMAC-SHA256 TC1: key 0x0b x20, data "Hi There" =====
  call void @llvm.memset.p0.i64(ptr @g.key, i8 11, i64 20, i1 false)
  call void @universe_crypto_hmac_sha256(ptr @g.key, i64 20, ptr @d.hi, i64 8, ptr @g.out)
  %c1 = call i32 @memcmp(ptr @g.out, ptr @e.s256.tc1, i64 32)
  %c1ok = icmp eq i32 %c1, 0
  call void @ut_check(i1 %c1ok, ptr @m.s256.tc1)

  ; ===== HMAC-SHA256 TC2: key "Jefe", data "what do ya want for nothing?" =====
  call void @universe_crypto_hmac_sha256(ptr @d.jefe, i64 4, ptr @d.what, i64 28, ptr @g.out)
  %c2 = call i32 @memcmp(ptr @g.out, ptr @e.s256.tc2, i64 32)
  %c2ok = icmp eq i32 %c2, 0
  call void @ut_check(i1 %c2ok, ptr @m.s256.tc2)

  ; ===== HMAC-SHA256 TC6: key 0xaa x131 (hashed), data 54-byte string =====
  call void @llvm.memset.p0.i64(ptr @g.key, i8 -86, i64 131, i1 false)   ; 0xaa
  call void @universe_crypto_hmac_sha256(ptr @g.key, i64 131, ptr @d.big, i64 54, ptr @g.out)
  %c6 = call i32 @memcmp(ptr @g.out, ptr @e.s256.tc6, i64 32)
  %c6ok = icmp eq i32 %c6, 0
  call void @ut_check(i1 %c6ok, ptr @m.s256.tc6)

  ; ===== HMAC-SHA256 empty key, empty message =====
  call void @universe_crypto_hmac_sha256(ptr @g.key, i64 0, ptr @g.key, i64 0, ptr @g.out)
  %ce = call i32 @memcmp(ptr @g.out, ptr @e.s256.empty, i64 32)
  %ceok = icmp eq i32 %ce, 0
  call void @ut_check(i1 %ceok, ptr @m.s256.emp)

  ; ===== HMAC-SHA512 TC1 =====
  call void @llvm.memset.p0.i64(ptr @g.key, i8 11, i64 20, i1 false)
  call void @universe_crypto_hmac_sha512(ptr @g.key, i64 20, ptr @d.hi, i64 8, ptr @g.out)
  %d1 = call i32 @memcmp(ptr @g.out, ptr @e.s512.tc1, i64 64)
  %d1ok = icmp eq i32 %d1, 0
  call void @ut_check(i1 %d1ok, ptr @m.s512.tc1)

  ; ===== HMAC-SHA512 TC2 =====
  call void @universe_crypto_hmac_sha512(ptr @d.jefe, i64 4, ptr @d.what, i64 28, ptr @g.out)
  %d2 = call i32 @memcmp(ptr @g.out, ptr @e.s512.tc2, i64 64)
  %d2ok = icmp eq i32 %d2, 0
  call void @ut_check(i1 %d2ok, ptr @m.s512.tc2)

  ; ===== HMAC-SHA1 TC1 =====
  call void @llvm.memset.p0.i64(ptr @g.key, i8 11, i64 20, i1 false)
  call void @universe_crypto_hmac_sha1(ptr @g.key, i64 20, ptr @d.hi, i64 8, ptr @g.out)
  %s1 = call i32 @memcmp(ptr @g.out, ptr @e.s1.tc1, i64 20)
  %s1ok = icmp eq i32 %s1, 0
  call void @ut_check(i1 %s1ok, ptr @m.s1.tc1)

  ; ===== HMAC-SHA1 TC2 =====
  call void @universe_crypto_hmac_sha1(ptr @d.jefe, i64 4, ptr @d.what, i64 28, ptr @g.out)
  %s2 = call i32 @memcmp(ptr @g.out, ptr @e.s1.tc2, i64 20)
  %s2ok = icmp eq i32 %s2, 0
  call void @ut_check(i1 %s2ok, ptr @m.s1.tc2)

  ; ===== HMAC-SHA1 TC6: key 0xaa x80 (hashed), data 54-byte string =====
  call void @llvm.memset.p0.i64(ptr @g.key, i8 -86, i64 80, i1 false)    ; 0xaa
  call void @universe_crypto_hmac_sha1(ptr @g.key, i64 80, ptr @d.big, i64 54, ptr @g.out)
  %s6 = call i32 @memcmp(ptr @g.out, ptr @e.s1.tc6, i64 20)
  %s6ok = icmp eq i32 %s6, 0
  call void @ut_check(i1 %s6ok, ptr @m.s1.tc6)

  ; ===== streaming == one-shot (HMAC-SHA256, key "Jefe", data @d.what) =====
  call void @universe_crypto_hmac_sha256_init(ptr @g.hctx, ptr @d.jefe, i64 4)
  br label %sc.head

sc.head:
  %si = phi i64 [ 0, %entry ], [ %si.n, %sc.body ]
  %sdone = icmp uge i64 %si, 28
  br i1 %sdone, label %sc.fin, label %sc.body

sc.body:
  %rem = sub nuw i64 28, %si
  %odd = and i64 %si, 1
  %isodd = icmp ne i64 %odd, 0
  %csz0 = select i1 %isodd, i64 5, i64 3
  %clamp = icmp ult i64 %rem, %csz0
  %csz = select i1 %clamp, i64 %rem, i64 %csz0
  %dp = getelementptr inbounds nuw i8, ptr @d.what, i64 %si
  call void @universe_crypto_hmac_update(ptr @g.hctx, ptr %dp, i64 %csz)
  %si.n = add nuw i64 %si, %csz
  br label %sc.head

sc.fin:
  call void @universe_crypto_hmac_final(ptr @g.hctx, ptr @g.out)
  call void @universe_crypto_hmac_sha256(ptr @d.jefe, i64 4, ptr @d.what, i64 28, ptr @g.out2)
  %sccmp = call i32 @memcmp(ptr @g.out, ptr @g.out2, i64 32)
  %scok = icmp eq i32 %sccmp, 0
  call void @ut_check(i1 %scok, ptr @m.stream)

  ; ===== bench =====
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  call void @llvm.memset.p0.i64(ptr @g.key, i8 66, i64 1024, i1 false)
  br label %brep.head

brep.head:
  %brep = phi i64 [ 0, %bench ], [ %brep.n, %brep.tail ]
  %bt0 = call double @ut_now_sec()
  br label %be.head

be.head:
  %bc = phi i64 [ 0, %brep.head ], [ %bc.n, %be.head ]
  call void @universe_crypto_hmac_sha256(ptr @d.jefe, i64 4, ptr @g.key, i64 1024, ptr @g.out)
  %bc.n = add nuw i64 %bc, 1
  %bmore = icmp ult i64 %bc.n, 100000
  br i1 %bmore, label %be.head, label %be.done

be.done:
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (100000*1024 bytes/rep)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore

bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @hmac.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail

brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report

brep.report:
  call void @ut_report_dist(ptr @hmac.samp, i64 16, i64 102400000, ptr @lbl.hmac)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
