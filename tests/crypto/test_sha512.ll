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

; Tests for universe_crypto_sha512_*: FIPS 180-4 known-answer vectors,
; streaming==one-shot over odd chunkings, fixed-seed random, and --bench.

declare void @universe_crypto_sha512_init(ptr)
declare void @universe_crypto_sha512_update(ptr, ptr, i64)
declare void @universe_crypto_sha512_final(ptr, ptr)
declare void @universe_crypto_sha512_hash(ptr, i64, ptr)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@k.abc   = private unnamed_addr constant [3 x i8] c"abc"
@k.nist2 = private unnamed_addr constant [56 x i8] c"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"

@d.empty = private unnamed_addr constant [64 x i8] c"\cf\83\e1\35\7e\ef\b8\bd\f1\54\28\50\d6\6d\80\07\d6\20\e4\05\0b\57\15\dc\83\f4\a9\21\d3\6c\e9\ce\47\d0\d1\3c\5d\85\f2\b0\ff\83\18\d2\87\7e\ec\2f\63\b9\31\bd\47\41\7a\81\a5\38\32\7a\f9\27\da\3e"
@d.abc   = private unnamed_addr constant [64 x i8] c"\dd\af\35\a1\93\61\7a\ba\cc\41\73\49\ae\20\41\31\12\e6\fa\4e\89\a9\7e\a2\0a\9e\ee\e6\4b\55\d3\9a\21\92\99\2a\27\4f\c1\a8\36\ba\3c\23\a3\fe\eb\bd\45\4d\44\23\64\3c\e8\0e\2a\9a\c9\4f\a5\4c\a4\9f"
@d.nist2 = private unnamed_addr constant [64 x i8] c"\20\4a\8f\c6\dd\a8\2f\0a\0c\ed\7b\eb\8e\08\a4\16\57\c1\6e\f4\68\b2\28\a8\27\9b\e3\31\a7\03\c3\35\96\fd\15\c1\3b\1b\07\f9\aa\1d\3b\ea\57\78\9c\a0\31\ad\85\c7\a7\1d\d7\03\54\ec\63\12\38\ca\34\45"

@g.src = internal global [70000 x i8] zeroinitializer, align 16
@g.ctx = internal global [200 x i8] zeroinitializer, align 8
@g.out = internal global [64 x i8] zeroinitializer, align 8
@g.out2 = internal global [64 x i8] zeroinitializer, align 8

@lens = internal constant [11 x i64] [ i64 0, i64 1, i64 111, i64 112, i64 127, i64 128, i64 129, i64 255, i64 256, i64 1000, i64 65536 ], align 8

@m.empty = private unnamed_addr constant [16 x i8] c"sha512(empty)\00\00\00"
@m.abc   = private unnamed_addr constant [12 x i8] c"sha512(abc)\00"
@m.nist2 = private unnamed_addr constant [14 x i8] c"sha512(nist2)\00"
@m.stream = private unnamed_addr constant [22 x i8] c"stream==oneshot chunk\00"
@m.rand  = private unnamed_addr constant [20 x i8] c"random stream==hash\00"
@fmt.bench = private unnamed_addr constant [34 x i8] c"bench sha512 64KiB: %.2f MB/s\0A\00\00\00\00"
@sha512.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.sha512 = private unnamed_addr constant [24 x i8] c"sha512 64KiB (per byte)\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @universe_crypto_sha512_hash(ptr @g.src, i64 0, ptr @g.out)
  %ce = call i32 @memcmp(ptr @g.out, ptr @d.empty, i64 64)
  %ce.ok = icmp eq i32 %ce, 0
  call void @ut_check(i1 %ce.ok, ptr @m.empty)

  call void @universe_crypto_sha512_hash(ptr @k.abc, i64 3, ptr @g.out)
  %ca = call i32 @memcmp(ptr @g.out, ptr @d.abc, i64 64)
  %ca.ok = icmp eq i32 %ca, 0
  call void @ut_check(i1 %ca.ok, ptr @m.abc)

  call void @universe_crypto_sha512_hash(ptr @k.nist2, i64 56, ptr @g.out)
  %cn = call i32 @memcmp(ptr @g.out, ptr @d.nist2, i64 64)
  %cn.ok = icmp eq i32 %cn, 0
  call void @ut_check(i1 %cn.ok, ptr @m.nist2)

  ; ---- streaming odd chunks over 56-byte msg == one-shot ----
  call void @universe_crypto_sha512_init(ptr @g.ctx)
  br label %sc.head

sc.head:
  %si = phi i64 [ 0, %entry ], [ %si.n, %sc.body ]
  %sdone = icmp uge i64 %si, 56
  br i1 %sdone, label %sc.fin, label %sc.body

sc.body:
  %rem = sub nuw i64 56, %si
  %odd = and i64 %si, 1
  %isodd = icmp ne i64 %odd, 0
  %csz0 = select i1 %isodd, i64 7, i64 3
  %clamp = icmp ult i64 %rem, %csz0
  %csz = select i1 %clamp, i64 %rem, i64 %csz0
  %dp = getelementptr inbounds nuw i8, ptr @k.nist2, i64 %si
  call void @universe_crypto_sha512_update(ptr @g.ctx, ptr %dp, i64 %csz)
  %si.n = add nuw i64 %si, %csz
  br label %sc.head

sc.fin:
  call void @universe_crypto_sha512_final(ptr @g.ctx, ptr @g.out)
  call void @universe_crypto_sha512_hash(ptr @k.nist2, i64 56, ptr @g.out2)
  %sccmp = call i32 @memcmp(ptr @g.out, ptr @g.out2, i64 64)
  %sc.ok = icmp eq i32 %sccmp, 0
  call void @ut_check(i1 %sc.ok, ptr @m.stream)

  ; ---- random streaming vs one-shot ----
  %state = alloca i64, align 8
  store i64 305419896, ptr %state, align 8
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %sc.fin ], [ %fi.n, %fill.body ]
  %fdone = icmp uge i64 %fi, 65536
  br i1 %fdone, label %rt.head, label %fill.body

fill.body:
  %rv = call i64 @ut_rand(ptr %state)
  %rb = trunc i64 %rv to i8
  %sp = getelementptr inbounds nuw [70000 x i8], ptr @g.src, i64 0, i64 %fi
  store i8 %rb, ptr %sp, align 1
  %fi.n = add nuw i64 %fi, 1
  br label %fill.head

rt.head:
  %li = phi i64 [ 0, %fill.head ], [ %li.n, %rt.next ]
  %mis = phi i64 [ 0, %fill.head ], [ %mis.n, %rt.next ]
  %lp = getelementptr inbounds nuw [11 x i64], ptr @lens, i64 0, i64 %li
  %len = load i64, ptr %lp, align 8
  call void @universe_crypto_sha512_hash(ptr @g.src, i64 %len, ptr @g.out2)
  call void @universe_crypto_sha512_init(ptr @g.ctx)
  br label %rs.head

rs.head:
  %ri = phi i64 [ 0, %rt.head ], [ %ri.n, %rs.body ]
  %rdone = icmp uge i64 %ri, %len
  br i1 %rdone, label %rs.fin, label %rs.body

rs.body:
  %rrem = sub nuw i64 %len, %ri
  %rcl = icmp ult i64 %rrem, 13
  %rcsz = select i1 %rcl, i64 %rrem, i64 13
  %rdp = getelementptr inbounds nuw [70000 x i8], ptr @g.src, i64 0, i64 %ri
  call void @universe_crypto_sha512_update(ptr @g.ctx, ptr %rdp, i64 %rcsz)
  %ri.n = add nuw i64 %ri, %rcsz
  br label %rs.head

rs.fin:
  call void @universe_crypto_sha512_final(ptr @g.ctx, ptr @g.out)
  %rcmp = call i32 @memcmp(ptr @g.out, ptr @g.out2, i64 64)
  %rbad = icmp ne i32 %rcmp, 0
  %rbad.i = zext i1 %rbad to i64
  %mis.n = add nuw i64 %mis, %rbad.i
  br label %rt.next

rt.next:
  %li.n = add nuw i64 %li, 1
  %more = icmp ult i64 %li.n, 11
  br i1 %more, label %rt.head, label %rt.done

rt.done:
  call void @ut_check_eq(i64 %mis.n, i64 0, ptr @m.rand)

  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %brep.head

brep.head:
  %brep = phi i64 [ 0, %bench ], [ %brep.n, %brep.tail ]
  %bt0 = call double @ut_now_sec()
  br label %be.head

be.head:
  %bc = phi i64 [ 0, %brep.head ], [ %bc.n, %be.head ]
  call void @universe_crypto_sha512_hash(ptr @g.src, i64 65536, ptr @g.out)
  %bc.n = add nuw i64 %bc, 1
  %bmore = icmp ult i64 %bc.n, 2000
  br i1 %bmore, label %be.head, label %be.done

be.done:
  %bt1 = call double @ut_now_sec()
  %bdt = fsub double %bt1, %bt0
  ; discard rep 0 as warm-up; store reps 1..16 (2000*65536 bytes/rep)
  %bwarm = icmp eq i64 %brep, 0
  br i1 %bwarm, label %brep.tail, label %bstore

bstore:
  %bslot = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @sha512.samp, i64 0, i64 %bslot
  store double %bdt, ptr %bsp, align 8
  br label %brep.tail

brep.tail:
  %brep.n = add nuw i64 %brep, 1
  %brepmore = icmp ult i64 %brep.n, 17
  br i1 %brepmore, label %brep.head, label %brep.report

brep.report:
  call void @ut_report_dist(ptr @sha512.samp, i64 16, i64 131072000, ptr @lbl.sha512)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
