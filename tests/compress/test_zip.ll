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

; Tests for src/compress/zip.ll : open, count, entry metadata (name/method/
; sizes), and extract of a DEFLATE member and a STORED member.

declare i32 @universe_compress_zip_open(ptr, i64, ptr)
declare i64 @universe_compress_zip_count(ptr)
declare i32 @universe_compress_zip_entry(ptr, i64, ptr)
declare i64 @universe_compress_zip_extract(ptr, ptr, ptr, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @memcmp(ptr, ptr, i64)

@m_open  = private constant [12 x i8] c"zip open ok\00", align 1
@m_cnt   = private constant [12 x i8] c"zip count 2\00", align 1
@m_e0    = private constant [12 x i8] c"entry0 ok  \00", align 1
@m_e0n   = private constant [12 x i8] c"entry0 name\00", align 1
@m_e0m   = private constant [12 x i8] c"entry0 meth\00", align 1
@m_e0u   = private constant [12 x i8] c"entry0 usz \00", align 1
@m_e1    = private constant [12 x i8] c"entry1 ok  \00", align 1
@m_e1m   = private constant [12 x i8] c"entry1 meth\00", align 1
@m_exA   = private constant [13 x i8] c"extractA len\00", align 1
@m_exAc  = private constant [13 x i8] c"extractA cmp\00", align 1
@m_exB   = private constant [13 x i8] c"extractB len\00", align 1
@m_exBc  = private constant [13 x i8] c"extractB cmp\00", align 1
@m_oob   = private constant [12 x i8] c"entry oob  \00", align 1

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %reader = alloca [32 x i8], align 8
  %e0 = alloca [48 x i8], align 8
  %e1 = alloca [48 x i8], align 8
  %out = alloca [4096 x i8], align 16

  %rc = call i32 @universe_compress_zip_open(ptr @zip_data, i64 269, ptr %reader)
  %open_ok = icmp eq i32 %rc, 0
  call void @ut_check(i1 %open_ok, ptr @m_open)

  %cnt = call i64 @universe_compress_zip_count(ptr %reader)
  call void @ut_check_eq(i64 %cnt, i64 2, ptr @m_cnt)

  ; entry 0 = docs/a.txt (deflate)
  %r0 = call i32 @universe_compress_zip_entry(ptr %reader, i64 0, ptr %e0)
  %e0_ok = icmp eq i32 %r0, 0
  call void @ut_check(i1 %e0_ok, ptr @m_e0)
  %e0_namep = getelementptr inbounds i8, ptr %e0, i64 0
  %e0_name = load ptr, ptr %e0_namep, align 8
  %e0_nlp = getelementptr inbounds i8, ptr %e0, i64 8
  %e0_nl = load i64, ptr %e0_nlp, align 8
  %nl_ok = icmp eq i64 %e0_nl, 10
  %ncmp = call i32 @memcmp(ptr %e0_name, ptr @nameA, i64 10)
  %ncmp_ok = icmp eq i32 %ncmp, 0
  %name_ok = and i1 %nl_ok, %ncmp_ok
  call void @ut_check(i1 %name_ok, ptr @m_e0n)
  %e0_mp = getelementptr inbounds i8, ptr %e0, i64 16
  %e0_m = load i64, ptr %e0_mp, align 8
  call void @ut_check_eq(i64 %e0_m, i64 8, ptr @m_e0m)
  %e0_up = getelementptr inbounds i8, ptr %e0, i64 32
  %e0_u = load i64, ptr %e0_up, align 8
  call void @ut_check_eq(i64 %e0_u, i64 540, ptr @m_e0u)

  ; entry 1 = b.bin (stored)
  %r1 = call i32 @universe_compress_zip_entry(ptr %reader, i64 1, ptr %e1)
  %e1_ok = icmp eq i32 %r1, 0
  call void @ut_check(i1 %e1_ok, ptr @m_e1)
  %e1_mp = getelementptr inbounds i8, ptr %e1, i64 16
  %e1_m = load i64, ptr %e1_mp, align 8
  call void @ut_check_eq(i64 %e1_m, i64 0, ptr @m_e1m)

  ; extract A (deflate)
  %exA = call i64 @universe_compress_zip_extract(ptr %reader, ptr %e0, ptr %out, i64 4096)
  call void @ut_check_eq(i64 %exA, i64 540, ptr @m_exA)
  %cmpA = call i32 @memcmp(ptr %out, ptr @zA, i64 540)
  %exA_ok = icmp eq i32 %cmpA, 0
  call void @ut_check(i1 %exA_ok, ptr @m_exAc)

  ; extract B (stored)
  %exB = call i64 @universe_compress_zip_extract(ptr %reader, ptr %e1, ptr %out, i64 4096)
  call void @ut_check_eq(i64 %exB, i64 31, ptr @m_exB)
  %cmpB = call i32 @memcmp(ptr %out, ptr @zB, i64 31)
  %exB_ok = icmp eq i32 %cmpB, 0
  call void @ut_check(i1 %exB_ok, ptr @m_exBc)

  ; out-of-range entry
  %roob = call i32 @universe_compress_zip_entry(ptr %reader, i64 5, ptr %e0)
  %oob_ok = icmp eq i32 %roob, 7
  call void @ut_check(i1 %oob_ok, ptr @m_oob)

  %r = call i32 @ut_summary()
  ret i32 %r
}
@zip_data = private constant [269 x i8] c"\50\4b\03\04\14\00\00\00\08\00\00\00\21\00\c2\0d\dc\aa\22\00\00\00\1c\02\00\00\0a\00\00\00\64\6f\63\73\2f\61\2e\74\78\74\f3\48\cd\c9\c9\57\48\2b\ca\cf\55\c8\cc\2b\ce\4c\49\55\28\c9\48\55\a8\ca\2c\50\54\f0\18\95\1a\d9\52\00\50\4b\03\04\14\00\00\00\00\00\00\00\21\00\88\96\73\0f\1f\00\00\00\1f\00\00\00\05\00\00\00\62\2e\62\69\6e\73\74\6f\72\65\64\2d\66\69\6c\65\2d\63\6f\6e\74\65\6e\74\73\2d\30\31\32\33\34\35\36\37\38\39\50\4b\01\02\14\03\14\00\00\00\08\00\00\00\21\00\c2\0d\dc\aa\22\00\00\00\1c\02\00\00\0a\00\00\00\00\00\00\00\00\00\00\00\80\01\00\00\00\00\64\6f\63\73\2f\61\2e\74\78\74\50\4b\01\02\14\03\14\00\00\00\00\00\00\00\21\00\88\96\73\0f\1f\00\00\00\1f\00\00\00\05\00\00\00\00\00\00\00\00\00\00\00\80\01\4a\00\00\00\62\2e\62\69\6e\50\4b\05\06\00\00\00\00\02\00\02\00\6b\00\00\00\8c\00\00\00\00\00", align 1
@zA = private constant [540 x i8] c"\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20\48\65\6c\6c\6f\20\66\72\6f\6d\20\69\6e\73\69\64\65\20\74\68\65\20\7a\69\70\21\20", align 1
@zB = private constant [31 x i8] c"\73\74\6f\72\65\64\2d\66\69\6c\65\2d\63\6f\6e\74\65\6e\74\73\2d\30\31\32\33\34\35\36\37\38\39", align 1
@nameA = private constant [11 x i8] c"docs/a.txt\00", align 1
