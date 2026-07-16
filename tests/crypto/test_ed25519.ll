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

; Tests for universe_crypto_ed25519_*: RFC 8032 section 7.1 known-answer
; vectors (TEST 1/2/3): seed->public, sign->signature, verify accept/reject.

declare void @universe_crypto_ed25519_public_from_seed(ptr, ptr)
declare void @universe_crypto_ed25519_sign(ptr, ptr, i64, ptr)
declare i32 @universe_crypto_ed25519_verify(ptr, ptr, i64, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; ---- RFC 8032 test vectors ----
@seed1 = private unnamed_addr constant [32 x i8] c"\9d\61\b1\9d\ef\fd\5a\60\ba\84\4a\f4\92\ec\2c\c4\44\49\c5\69\7b\32\69\19\70\3b\ac\03\1c\ae\7f\60"
@pub1  = private unnamed_addr constant [32 x i8] c"\d7\5a\98\01\82\b1\0a\b7\d5\4b\fe\d3\c9\64\07\3a\0e\e1\72\f3\da\a6\23\25\af\02\1a\68\f7\07\51\1a"
@sig1  = private unnamed_addr constant [64 x i8] c"\e5\56\43\00\c3\60\ac\72\90\86\e2\cc\80\6e\82\8a\84\87\7f\1e\b8\e5\d9\74\d8\73\e0\65\22\49\01\55\5f\b8\82\15\90\a3\3b\ac\c6\1e\39\70\1c\f9\b4\6b\d2\5b\f5\f0\59\5b\be\24\65\51\41\43\8e\7a\10\0b"

@seed2 = private unnamed_addr constant [32 x i8] c"\4c\cd\08\9b\28\ff\96\da\9d\b6\c3\46\ec\11\4e\0f\5b\8a\31\9f\35\ab\a6\24\da\8c\f6\ed\4f\b8\a6\fb"
@pub2  = private unnamed_addr constant [32 x i8] c"\3d\40\17\c3\e8\43\89\5a\92\b7\0a\a7\4d\1b\7e\bc\9c\98\2c\cf\2e\c4\96\8c\c0\cd\55\f1\2a\f4\66\0c"
@msg2  = private unnamed_addr constant [1 x i8] c"\72"
@sig2  = private unnamed_addr constant [64 x i8] c"\92\a0\09\a9\f0\d4\ca\b8\72\0e\82\0b\5f\64\25\40\a2\b2\7b\54\16\50\3f\8f\b3\76\22\23\eb\db\69\da\08\5a\c1\e4\3e\15\99\6e\45\8f\36\13\d0\f1\1d\8c\38\7b\2e\ae\b4\30\2a\ee\b0\0d\29\16\12\bb\0c\00"

@seed3 = private unnamed_addr constant [32 x i8] c"\c5\aa\8d\f4\3f\9f\83\7b\ed\b7\44\2f\31\dc\b7\b1\66\d3\85\35\07\6f\09\4b\85\ce\3a\2e\0b\44\58\f7"
@pub3  = private unnamed_addr constant [32 x i8] c"\fc\51\cd\8e\62\18\a1\a3\8d\a4\7e\d0\02\30\f0\58\08\16\ed\13\ba\33\03\ac\5d\eb\91\15\48\90\80\25"
@msg3  = private unnamed_addr constant [2 x i8] c"\af\82"
@sig3  = private unnamed_addr constant [64 x i8] c"\62\91\d6\57\de\ec\24\02\48\27\e6\9c\3a\be\01\a3\0c\e5\48\a2\84\74\3a\44\5e\36\80\d7\db\5a\c3\ac\18\ff\9b\53\8d\16\f2\90\ae\67\f7\60\98\4d\c6\59\4a\7c\15\e9\71\6e\d2\8d\c0\27\be\ce\ea\1e\c4\0a"

@g.pub = internal global [32 x i8] zeroinitializer, align 8
@g.sig = internal global [64 x i8] zeroinitializer, align 8
@g.tsig = internal global [64 x i8] zeroinitializer, align 8

@m.pub1  = private unnamed_addr constant [16 x i8] c"pub from seed 1\00"
@m.sig1  = private unnamed_addr constant [12 x i8] c"sign vec 1\00\00"
@m.ver1  = private unnamed_addr constant [14 x i8] c"verify vec 1\00\00"
@m.pub2  = private unnamed_addr constant [16 x i8] c"pub from seed 2\00"
@m.sig2  = private unnamed_addr constant [12 x i8] c"sign vec 2\00\00"
@m.ver2  = private unnamed_addr constant [14 x i8] c"verify vec 2\00\00"
@m.pub3  = private unnamed_addr constant [16 x i8] c"pub from seed 3\00"
@m.sig3  = private unnamed_addr constant [12 x i8] c"sign vec 3\00\00"
@m.ver3  = private unnamed_addr constant [14 x i8] c"verify vec 3\00\00"
@m.rejsig = private unnamed_addr constant [22 x i8] c"reject tampered sig\00\00\00"
@m.rejmsg = private unnamed_addr constant [22 x i8] c"reject tampered msg\00\00\00"
@fmt.bench = private unnamed_addr constant [30 x i8] c"bench ed25519 sign: %.0f/s\0A\00\00\00"
@fmt.benchv = private unnamed_addr constant [30 x i8] c"bench ed25519 vrfy: %.0f/s\0A\00\00\00"
@ed.sign.samp = internal global [16 x double] zeroinitializer, align 8
@ed.verify.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.ed.sign = private unnamed_addr constant [13 x i8] c"ed25519 sign\00"
@lbl.ed.verify = private unnamed_addr constant [15 x i8] c"ed25519 verify\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- vector 1 (empty message) ----
  call void @universe_crypto_ed25519_public_from_seed(ptr @seed1, ptr @g.pub)
  %p1 = call i32 @memcmp(ptr @g.pub, ptr @pub1, i64 32)
  %p1ok = icmp eq i32 %p1, 0
  call void @ut_check(i1 %p1ok, ptr @m.pub1)
  call void @universe_crypto_ed25519_sign(ptr @seed1, ptr @seed1, i64 0, ptr @g.sig)
  %s1 = call i32 @memcmp(ptr @g.sig, ptr @sig1, i64 64)
  %s1ok = icmp eq i32 %s1, 0
  call void @ut_check(i1 %s1ok, ptr @m.sig1)
  %v1 = call i32 @universe_crypto_ed25519_verify(ptr @pub1, ptr @seed1, i64 0, ptr @sig1)
  %v1ok = icmp eq i32 %v1, 0
  call void @ut_check(i1 %v1ok, ptr @m.ver1)

  ; ---- vector 2 (1-byte message) ----
  call void @universe_crypto_ed25519_public_from_seed(ptr @seed2, ptr @g.pub)
  %p2 = call i32 @memcmp(ptr @g.pub, ptr @pub2, i64 32)
  %p2ok = icmp eq i32 %p2, 0
  call void @ut_check(i1 %p2ok, ptr @m.pub2)
  call void @universe_crypto_ed25519_sign(ptr @seed2, ptr @msg2, i64 1, ptr @g.sig)
  %s2 = call i32 @memcmp(ptr @g.sig, ptr @sig2, i64 64)
  %s2ok = icmp eq i32 %s2, 0
  call void @ut_check(i1 %s2ok, ptr @m.sig2)
  %v2 = call i32 @universe_crypto_ed25519_verify(ptr @pub2, ptr @msg2, i64 1, ptr @sig2)
  %v2ok = icmp eq i32 %v2, 0
  call void @ut_check(i1 %v2ok, ptr @m.ver2)

  ; ---- vector 3 (2-byte message) ----
  call void @universe_crypto_ed25519_public_from_seed(ptr @seed3, ptr @g.pub)
  %p3 = call i32 @memcmp(ptr @g.pub, ptr @pub3, i64 32)
  %p3ok = icmp eq i32 %p3, 0
  call void @ut_check(i1 %p3ok, ptr @m.pub3)
  call void @universe_crypto_ed25519_sign(ptr @seed3, ptr @msg3, i64 2, ptr @g.sig)
  %s3 = call i32 @memcmp(ptr @g.sig, ptr @sig3, i64 64)
  %s3ok = icmp eq i32 %s3, 0
  call void @ut_check(i1 %s3ok, ptr @m.sig3)
  %v3 = call i32 @universe_crypto_ed25519_verify(ptr @pub3, ptr @msg3, i64 2, ptr @sig3)
  %v3ok = icmp eq i32 %v3, 0
  call void @ut_check(i1 %v3ok, ptr @m.ver3)

  ; ---- reject tampered signature (flip a byte of sig3) ----
  call void @llvm.memcpy.p0.p0.i64(ptr @g.tsig, ptr @sig3, i64 64, i1 false)
  %tp = getelementptr inbounds i8, ptr @g.tsig, i64 40
  %tv = load i8, ptr %tp, align 1
  %tv2 = xor i8 %tv, 1
  store i8 %tv2, ptr %tp, align 1
  %rj = call i32 @universe_crypto_ed25519_verify(ptr @pub3, ptr @msg3, i64 2, ptr @g.tsig)
  %rjok = icmp ne i32 %rj, 0
  call void @ut_check(i1 %rjok, ptr @m.rejsig)

  ; ---- reject tampered message (use msg2 against sig3/pub3) ----
  %rm = call i32 @universe_crypto_ed25519_verify(ptr @pub3, ptr @msg2, i64 1, ptr @sig3)
  %rmok = icmp ne i32 %rm, 0
  call void @ut_check(i1 %rmok, ptr @m.rejmsg)

  ; ---- bench ----
  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %do.bench, label %fin

do.bench:
  br label %srep.head
srep.head:
  %srep = phi i64 [ 0, %do.bench ], [ %srep.n, %srep.tail ]
  %st0 = call double @ut_now_sec()
  br label %bs.head
bs.head:
  %bi = phi i64 [ 0, %srep.head ], [ %bi.n, %bs.body ]
  %bdone = icmp uge i64 %bi, 200
  br i1 %bdone, label %bs.fin, label %bs.body
bs.body:
  call void @universe_crypto_ed25519_sign(ptr @seed3, ptr @msg3, i64 2, ptr @g.sig)
  %bi.n = add i64 %bi, 1
  br label %bs.head
bs.fin:
  %st1 = call double @ut_now_sec()
  %sdt = fsub double %st1, %st0
  ; discard rep 0 as warm-up; store reps 1..16 (200 signs/rep)
  %swarm = icmp eq i64 %srep, 0
  br i1 %swarm, label %srep.tail, label %sstore
sstore:
  %sslot = sub i64 %srep, 1
  %ssp = getelementptr inbounds [16 x double], ptr @ed.sign.samp, i64 0, i64 %sslot
  store double %sdt, ptr %ssp, align 8
  br label %srep.tail
srep.tail:
  %srep.n = add nuw i64 %srep, 1
  %srepmore = icmp ult i64 %srep.n, 17
  br i1 %srepmore, label %srep.head, label %srep.report
srep.report:
  call void @ut_report_dist(ptr @ed.sign.samp, i64 16, i64 200, ptr @lbl.ed.sign)
  br label %vrep.head

  ; verify bench
vrep.head:
  %vrep = phi i64 [ 0, %srep.report ], [ %vrep.n, %vrep.tail ]
  %vt0 = call double @ut_now_sec()
  br label %bv.head
bv.head:
  %vi = phi i64 [ 0, %vrep.head ], [ %vi.n, %bv.body ]
  %vdone = icmp uge i64 %vi, 200
  br i1 %vdone, label %bv.fin, label %bv.body
bv.body:
  %vv = call i32 @universe_crypto_ed25519_verify(ptr @pub3, ptr @msg3, i64 2, ptr @sig3)
  %vi.n = add i64 %vi, 1
  br label %bv.head
bv.fin:
  %vt1 = call double @ut_now_sec()
  %vdt = fsub double %vt1, %vt0
  ; discard rep 0 as warm-up; store reps 1..16 (200 verifies/rep)
  %vwarm = icmp eq i64 %vrep, 0
  br i1 %vwarm, label %vrep.tail, label %vstore
vstore:
  %vslot = sub i64 %vrep, 1
  %vsp = getelementptr inbounds [16 x double], ptr @ed.verify.samp, i64 0, i64 %vslot
  store double %vdt, ptr %vsp, align 8
  br label %vrep.tail
vrep.tail:
  %vrep.n = add nuw i64 %vrep, 1
  %vrepmore = icmp ult i64 %vrep.n, 17
  br i1 %vrepmore, label %vrep.head, label %vrep.report
vrep.report:
  call void @ut_report_dist(ptr @ed.verify.samp, i64 16, i64 200, ptr @lbl.ed.verify)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
