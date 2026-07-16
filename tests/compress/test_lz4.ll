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

; Tests for the LZ4 block codec (src/compress/lz4.ll).
;   * decode of real `lz4 -1` CLI-produced raw blocks == original (3 vectors:
;     repetitive, text, single-byte run exercising overlap copy)
;   * round-trip encode->decode over fixed-seed random and highly-repetitive data
;   * truncated / too-small-cap / NULL inputs return the right negative errors
;     with no OOB access (verified under -fsanitize=address,undefined)
;   * --bench: LZ4 decode MB/s

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @malloc(i64)
declare void @free(ptr)

declare i64 @universe_compress_lz4_decode(ptr, i64, ptr, i64)
declare i64 @universe_compress_lz4_encode(ptr, i64, ptr, i64)
declare i64 @universe_compress_lz4_bound(i64)

@lz4_v1_comp = internal constant [40 x i8] c"\3f\41\42\43\03\00\ff\ad\ff\05\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\14\00\ff\65\50\20\66\6f\78\20"
@lz4_v1_orig = internal constant [850 x i8] c"\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\41\42\43\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20"
@lz4_v2_comp = internal constant [136 x i8] c"\f2\57\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\5b\00\ff\01\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\7c\00\ff\ff\ff\4f\50\71\75\61\2e\20"
@lz4_v2_orig = internal constant [992 x i8] c"\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20\4c\6f\72\65\6d\20\69\70\73\75\6d\20\64\6f\6c\6f\72\20\73\69\74\20\61\6d\65\74\2c\20\63\6f\6e\73\65\63\74\65\74\75\72\20\61\64\69\70\69\73\63\69\6e\67\20\65\6c\69\74\2e\20\53\65\64\20\64\6f\20\65\69\75\73\6d\6f\64\20\74\65\6d\70\6f\72\20\69\6e\63\69\64\69\64\75\6e\74\20\75\74\20\6c\61\62\6f\72\65\20\65\74\20\64\6f\6c\6f\72\65\20\6d\61\67\6e\61\20\61\6c\69\71\75\61\2e\20"
@lz4_v3_comp = internal constant [12 x i8] c"\1f\41\01\00\ff\14\50\41\41\41\41\41"
@lz4_v3_orig = internal constant [300 x i8] c"\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41\41"

@m.v1len  = private constant [16 x i8] c"v1 decode len\00\00\00"
@m.v1cmp  = private constant [16 x i8] c"v1 bytes match\00\00"
@m.v2len  = private constant [16 x i8] c"v2 decode len\00\00\00"
@m.v2cmp  = private constant [16 x i8] c"v2 bytes match\00\00"
@m.v3len  = private constant [16 x i8] c"v3 decode len\00\00\00"
@m.v3cmp  = private constant [16 x i8] c"v3 bytes match\00\00"
@m.rt_rnd = private constant [24 x i8] c"round-trip random len\00\00\00"
@m.rt_rc  = private constant [24 x i8] c"round-trip random cmp\00\00\00"
@m.rt_rep = private constant [24 x i8] c"round-trip repeat len\00\00\00"
@m.rt_pc  = private constant [24 x i8] c"round-trip repeat cmp\00\00\00"
@m.trunc  = private constant [24 x i8] c"truncated -> negative\00\00\00"
@m.trunc2 = private constant [24 x i8] c"trunc mid -> negative\00\00\00"
@m.full   = private constant [24 x i8] c"small cap -> FULL\00\00\00\00\00\00\00"
@m.null   = private constant [24 x i8] c"null -> -1\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
@m.badoff = private constant [24 x i8] c"bad offset -> parse\00\00\00\00\00"
@lz4.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.lz4 = private unnamed_addr constant [25 x i8] c"lz4 decode 136B->992B x1\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %out = call ptr @malloc(i64 65536)
  %tmp = call ptr @malloc(i64 65536)

  ; ---- vector 1 ----
  %d1 = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536,
        ptr @lz4_v1_comp, i64 40)
  call void @ut_check_eq(i64 %d1, i64 850, ptr @m.v1len)
  %c1 = call i32 @memcmp(ptr %out, ptr @lz4_v1_orig, i64 850)
  %c1ok = icmp eq i32 %c1, 0
  call void @ut_check(i1 %c1ok, ptr @m.v1cmp)

  ; ---- vector 2 ----
  %d2 = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536,
        ptr @lz4_v2_comp, i64 136)
  call void @ut_check_eq(i64 %d2, i64 992, ptr @m.v2len)
  %c2 = call i32 @memcmp(ptr %out, ptr @lz4_v2_orig, i64 992)
  %c2ok = icmp eq i32 %c2, 0
  call void @ut_check(i1 %c2ok, ptr @m.v2cmp)

  ; ---- vector 3 (single-byte run, overlap copy) ----
  %d3 = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536,
        ptr @lz4_v3_comp, i64 12)
  call void @ut_check_eq(i64 %d3, i64 300, ptr @m.v3len)
  %c3 = call i32 @memcmp(ptr %out, ptr @lz4_v3_orig, i64 300)
  %c3ok = icmp eq i32 %c3, 0
  call void @ut_check(i1 %c3ok, ptr @m.v3cmp)

  ; ---- round-trip: random data (all-literal path) ----
  %rnd = call ptr @malloc(i64 4096)
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  br label %rfill.head
rfill.head:
  %ri = phi i64 [ 0, %entry ], [ %ri.n, %rfill.body ]
  %rlt = icmp ult i64 %ri, 4096
  br i1 %rlt, label %rfill.body, label %rdo
rfill.body:
  %rv = call i64 @ut_rand(ptr %seed)
  %rb = trunc i64 %rv to i8
  %rpp = getelementptr inbounds i8, ptr %rnd, i64 %ri
  store i8 %rb, ptr %rpp, align 1
  %ri.n = add i64 %ri, 1
  br label %rfill.head
rdo:
  %rbound = call i64 @universe_compress_lz4_bound(i64 4096)
  %renc = call i64 @universe_compress_lz4_encode(ptr %tmp, i64 %rbound, ptr %rnd, i64 4096)
  %rencok = icmp sgt i64 %renc, 0
  %rdec = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536, ptr %tmp, i64 %renc)
  call void @ut_check_eq(i64 %rdec, i64 4096, ptr @m.rt_rnd)
  %rc = call i32 @memcmp(ptr %out, ptr %rnd, i64 4096)
  %rcok0 = icmp eq i32 %rc, 0
  %rcok = and i1 %rcok0, %rencok
  call void @ut_check(i1 %rcok, ptr @m.rt_rc)

  ; ---- round-trip: highly-repetitive data (matches + overlap) ----
  %rep = call ptr @malloc(i64 8000)
  br label %pfill.head
pfill.head:
  %pi = phi i64 [ 0, %rdo ], [ %pi.n, %pfill.body ]
  %plt = icmp ult i64 %pi, 8000
  br i1 %plt, label %pfill.body, label %pdo
pfill.body:
  ; long runs + a repeating 24-byte-ish phrase => many matches, offset==1 runs
  %seg = udiv i64 %pi, 40
  %inseg = urem i64 %pi, 40
  %isrun = icmp ult i64 %inseg, 30
  %segb = trunc i64 %seg to i8
  %runv = add i8 65, %segb
  %vv = call i64 @ut_rand(ptr %seed)
  %vvb = trunc i64 %vv to i8
  %pb = select i1 %isrun, i8 %runv, i8 %vvb
  %ppp = getelementptr inbounds i8, ptr %rep, i64 %pi
  store i8 %pb, ptr %ppp, align 1
  %pi.n = add i64 %pi, 1
  br label %pfill.head
pdo:
  %pbound = call i64 @universe_compress_lz4_bound(i64 8000)
  %penc = call i64 @universe_compress_lz4_encode(ptr %tmp, i64 %pbound, ptr %rep, i64 8000)
  %pencok = icmp sgt i64 %penc, 0
  %pdec = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536, ptr %tmp, i64 %penc)
  call void @ut_check_eq(i64 %pdec, i64 8000, ptr @m.rt_rep)
  %pc = call i32 @memcmp(ptr %out, ptr %rep, i64 8000)
  %pcok0 = icmp eq i32 %pc, 0
  %pcok = and i1 %pcok0, %pencok
  call void @ut_check(i1 %pcok, ptr @m.rt_pc)

  ; ---- truncated block (drop last byte of v2) -> negative ----
  %t1 = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536,
        ptr @lz4_v2_comp, i64 135)
  %t1neg = icmp slt i64 %t1, 0
  call void @ut_check(i1 %t1neg, ptr @m.trunc)

  ; ---- truncated mid-stream (only first 5 bytes of v2) -> negative ----
  %t2 = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536,
        ptr @lz4_v2_comp, i64 5)
  %t2neg = icmp slt i64 %t2, 0
  call void @ut_check(i1 %t2neg, ptr @m.trunc2)

  ; ---- too-small output cap -> FULL (-6) ----
  %fl = call i64 @universe_compress_lz4_decode(ptr %out, i64 100,
        ptr @lz4_v1_comp, i64 40)
  %flok = icmp eq i64 %fl, -6
  call void @ut_check(i1 %flok, ptr @m.full)

  ; ---- NULL dst -> -1 ----
  %nu = call i64 @universe_compress_lz4_decode(ptr null, i64 100,
        ptr @lz4_v1_comp, i64 40)
  %nuok = icmp eq i64 %nu, -1
  call void @ut_check(i1 %nuok, ptr @m.null)

  ; ---- crafted bad-offset block: token 0x00 (0 lit, matchcode 0), offset 0 ----
  ; bytes: 0x00, 0x00, 0x00  -> match with offset 0 (invalid) -> -13
  %bad = alloca [3 x i8], align 1
  %b0 = getelementptr inbounds i8, ptr %bad, i64 0
  store i8 0, ptr %b0, align 1
  %b1 = getelementptr inbounds i8, ptr %bad, i64 1
  store i8 0, ptr %b1, align 1
  %b2 = getelementptr inbounds i8, ptr %bad, i64 2
  store i8 0, ptr %b2, align 1
  %bo = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536, ptr %bad, i64 3)
  %book = icmp eq i64 %bo, -13
  call void @ut_check(i1 %book, ptr @m.badoff)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; 17 reps of a 2,000,000-decode batch; discard rep 0 (warm-up), report over
  ; the remaining 16. ops/rep = 2000000 * 992 = 1984000000 output bytes
  ; (ns per decompressed byte).
  br label %lz.rep
lz.rep:
  %lrep = phi i64 [ 0, %bench ], [ %lrep.n, %lz.next ]
  %t0 = call double @ut_now_sec()
  br label %bl.head
bl.head:
  %bi = phi i64 [ 0, %lz.rep ], [ %bi.n, %bl.body ]
  %blt = icmp ult i64 %bi, 2000000
  br i1 %blt, label %bl.body, label %lz.rep.done
bl.body:
  %bd = call i64 @universe_compress_lz4_decode(ptr %out, i64 65536, ptr @lz4_v2_comp, i64 136)
  %bi.n = add i64 %bi, 1
  br label %bl.head
lz.rep.done:
  %t1b = call double @ut_now_sec()
  %lel = fsub double %t1b, %t0
  %lkeep = icmp ugt i64 %lrep, 0
  br i1 %lkeep, label %lz.store, label %lz.next
lz.store:
  %lidx = sub i64 %lrep, 1
  %lsp = getelementptr inbounds [16 x double], ptr @lz4.samp, i64 0, i64 %lidx
  store double %lel, ptr %lsp, align 8
  br label %lz.next
lz.next:
  %lrep.n = add nuw i64 %lrep, 1
  %lmore = icmp ult i64 %lrep.n, 17
  br i1 %lmore, label %lz.rep, label %lz.report
lz.report:
  call void @ut_report_dist(ptr @lz4.samp, i64 16, i64 1984000000, ptr @lbl.lz4)
  br label %fin

fin:
  call void @free(ptr %out)
  call void @free(ptr %tmp)
  call void @free(ptr %rnd)
  call void @free(ptr %rep)
  %r = call i32 @ut_summary()
  ret i32 %r
}
