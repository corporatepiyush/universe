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

; Tests for the Zstd frame decoder (src/compress/zstd.ll), RAW/RLE subset.
;   * RAW block frame decodes to original (hand-crafted, verified by `zstd -d`)
;   * RLE block frame decodes to a 200-byte 0x5A run
;   * a real `zstd -1` COMPRESSED frame returns -14 (UNSUPPORTED), no OOB
;   * truncated / bad-magic / small-cap / NULL return the right negatives with
;     no OOB access (verified under -fsanitize=address,undefined)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @malloc(i64)
declare i64 @universe_compress_zstd_decode(ptr, i64, ptr, i64)

@m.rawlen = private constant [23 x i8] c"raw decode len\00\00\00\00\00\00\00\00\00"
@m.rawcmp = private constant [23 x i8] c"raw bytes match\00\00\00\00\00\00\00\00"
@m.rlelen = private constant [23 x i8] c"rle decode len\00\00\00\00\00\00\00\00\00"
@m.rlecmp = private constant [23 x i8] c"rle bytes match\00\00\00\00\00\00\00\00"
@m.unsup  = private constant [32 x i8] c"compressed -> unsupported(-14)\00\00"
@m.trunc  = private constant [24 x i8] c"truncated -> negative\00\00\00"
@m.hdr    = private constant [24 x i8] c"short header -> trunc\00\00\00"
@m.badmag = private constant [24 x i8] c"bad magic -> parse\00\00\00\00\00\00"
@m.null   = private constant [24 x i8] c"null -> -1\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
@m.full   = private constant [24 x i8] c"small cap -> FULL\00\00\00\00\00\00\00"

@zstd_raw = internal constant [47 x i8] c"\28\b5\2f\fd\20\26\31\01\00\48\65\6c\6c\6f\2c\20\5a\73\74\61\6e\64\61\72\64\20\52\41\57\20\62\6c\6f\63\6b\21\20\30\31\32\33\34\35\36\37\38\39"
@zstd_raw_orig = internal constant [38 x i8] c"\48\65\6c\6c\6f\2c\20\5a\73\74\61\6e\64\61\72\64\20\52\41\57\20\62\6c\6f\63\6b\21\20\30\31\32\33\34\35\36\37\38\39"
@zstd_rle = internal constant [10 x i8] c"\28\b5\2f\fd\20\c8\43\06\00\5a"
@zstd_rle_orig = internal constant [200 x i8] c"\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a\5a"
@zstd_comp = internal constant [65 x i8] c"\28\b5\2f\fd\60\46\04\bd\01\00\c4\02\74\68\65\20\71\75\69\63\6b\20\62\72\6f\77\6e\20\66\6f\78\20\6a\75\6d\70\73\20\6f\76\65\72\20\74\68\65\20\6c\61\7a\79\20\64\6f\67\2e\02\00\12\41\f5\01\43\98\65"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %out = call ptr @malloc(i64 65536)

  ; ---- RAW ----
  %dr = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr @zstd_raw, i64 47)
  call void @ut_check_eq(i64 %dr, i64 38, ptr @m.rawlen)
  %cr = call i32 @memcmp(ptr %out, ptr @zstd_raw_orig, i64 38)
  %crok = icmp eq i32 %cr, 0
  call void @ut_check(i1 %crok, ptr @m.rawcmp)

  ; ---- RLE ----
  %dl = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr @zstd_rle, i64 10)
  call void @ut_check_eq(i64 %dl, i64 200, ptr @m.rlelen)
  %cl = call i32 @memcmp(ptr %out, ptr @zstd_rle_orig, i64 200)
  %clok = icmp eq i32 %cl, 0
  call void @ut_check(i1 %clok, ptr @m.rlecmp)

  ; ---- COMPRESSED frame -> unsupported ----
  %du = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr @zstd_comp, i64 65)
  %duok = icmp eq i64 %du, -14
  call void @ut_check(i1 %duok, ptr @m.unsup)

  ; ---- truncated RAW (drop last byte) -> negative ----
  %dt = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr @zstd_raw, i64 46)
  %dtneg = icmp slt i64 %dt, 0
  call void @ut_check(i1 %dtneg, ptr @m.trunc)

  ; ---- short header (only magic, 4 bytes) -> trunc ----
  %dh = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr @zstd_raw, i64 4)
  %dhok = icmp eq i64 %dh, -15
  call void @ut_check(i1 %dhok, ptr @m.hdr)

  ; ---- bad magic -> parse ----
  %bad = alloca [8 x i8], align 1
  %bp0 = getelementptr inbounds i8, ptr %bad, i64 0
  store i8 0, ptr %bp0, align 1
  %bp1 = getelementptr inbounds i8, ptr %bad, i64 1
  store i8 1, ptr %bp1, align 1
  %bp2 = getelementptr inbounds i8, ptr %bad, i64 2
  store i8 2, ptr %bp2, align 1
  %bp3 = getelementptr inbounds i8, ptr %bad, i64 3
  store i8 3, ptr %bp3, align 1
  %bp4 = getelementptr inbounds i8, ptr %bad, i64 4
  store i8 0, ptr %bp4, align 1
  %dm = call i64 @universe_compress_zstd_decode(ptr %out, i64 65536, ptr %bad, i64 8)
  %dmok = icmp eq i64 %dm, -13
  call void @ut_check(i1 %dmok, ptr @m.badmag)

  ; ---- NULL -> -1 ----
  %nu = call i64 @universe_compress_zstd_decode(ptr null, i64 65536, ptr @zstd_raw, i64 47)
  %nuok = icmp eq i64 %nu, -1
  call void @ut_check(i1 %nuok, ptr @m.null)

  ; ---- small cap on RLE (dcap=10, needs 200) -> FULL ----
  %fl = call i64 @universe_compress_zstd_decode(ptr %out, i64 10, ptr @zstd_rle, i64 10)
  %flok = icmp eq i64 %fl, -6
  call void @ut_check(i1 %flok, ptr @m.full)

  %r = call i32 @ut_summary()
  ret i32 %r
}
