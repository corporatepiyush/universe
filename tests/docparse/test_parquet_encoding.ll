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

; KATs for docparse/parquet_encoding — every value hand-derived from
; parquet-format Encodings.md, then confirmed against an independent decode.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare i32 @ut_summary()
declare i1 @ut_want_bench(i32, ptr)
declare double @ut_now_sec()

declare i32 @universe_docparse_parquet_rle_hybrid(ptr, i64, i32, i64, ptr)
declare i32 @universe_docparse_parquet_plain(ptr, i64, i32, i32, i64, ptr, i64, ptr)
declare i32 @universe_docparse_parquet_delta_binary_packed(ptr, i64, ptr, i64, ptr)
declare i32 @universe_docparse_parquet_delta_length_byte_array(ptr, i64, i64, ptr, i64, ptr)
declare i32 @universe_docparse_parquet_delta_byte_array(ptr, i64, i64, ptr, i64, ptr)
declare i32 @universe_docparse_parquet_byte_stream_split(ptr, i64, i32, i64, ptr)

; ---- KAT input constants ----
; RLE run: header 0x09 = (4<<1)|1 → run of 4; value 0x05 in 1 byte (bw=3).
@rle_run = private constant [2 x i8] c"\09\05"
; bit-packed run bw=3: header 0x02 = (1<<1) → 8 values; 0,1,..,7 packed = 88 C6 FA.
@rle_bp = private constant [4 x i8] c"\02\88\C6\FA"
; RLE run bw=32: header 0x05 = (2<<1)|1 → run 2; value 0x12345678 LE.
@rle_w32 = private constant [5 x i8] c"\05\78\56\34\12"
; plain INT32: 1, -2, 1000000.
@plain_i32 = private constant [12 x i8] c"\01\00\00\00\FE\FF\FF\FF\40\42\0F\00"
; plain DOUBLE: 3.14, 2.0 (LE bit patterns).
@plain_f64 = private constant [16 x i8] c"\1F\85\EB\51\B8\1E\09\40\00\00\00\00\00\00\00\40"
; plain BYTE_ARRAY: "hi","abc".
@plain_ba = private constant [13 x i8] c"\02\00\00\00hi\03\00\00\00abc"
; plain BOOLEAN: true,false,true,true → bits 1011 → 0x0D.
@plain_bool = private constant [1 x i8] c"\0D"
; DELTA_BINARY_PACKED: 1,2,3,4,5 (block 128, 4 miniblocks, total 5, first zz1).
@dbp = private constant [10 x i8] c"\80\01\04\05\02\02\00\00\00\00"
; DELTA_LENGTH_BYTE_ARRAY: "a","bb","ccc" (lengths [1,2,3] delta + "abbccc").
@dlba = private constant [16 x i8] c"\80\01\04\03\02\02\00\00\00\00abbccc"
; DELTA_BYTE_ARRAY: "abc","adef","adghi".
;   prefixes [0,1,2] delta block (10 B); suffix DLBA: lengths [3,3,3] (10 B) + "abcdefghi".
@dba = private constant [29 x i8] c"\80\01\04\03\00\02\00\00\00\00\80\01\04\03\06\00\00\00\00\00abcdefghi"
; BYTE_STREAM_SPLIT width=4 count=2 of floats 1.0f,2.0f. planes 00 00|00 00|80 00|3F 40.
@bss = private constant [8 x i8] c"\00\00\00\00\80\00\3F\40"

@m_rle_ret = private constant [16 x i8] c"rle_run ret ok\00\00"
@m_rle_v   = private constant [12 x i8] c"rle_run val\00"
@m_bp_ret  = private constant [12 x i8] c"rle_bp ret\00\00"
@m_bp_v    = private constant [11 x i8] c"rle_bp val\00"
@m_w32_v   = private constant [11 x i8] c"rle_w32 v \00"
@m_bw0     = private constant [12 x i8] c"rle bw0 zer\00"
@m_i32     = private constant [10 x i8] c"plain i32\00"
@m_i32u    = private constant [14 x i8] c"plain i32 use\00"
@m_f64     = private constant [10 x i8] c"plain f64\00"
@m_ba      = private constant [12 x i8] c"plain barr \00"
@m_bau     = private constant [13 x i8] c"plain barr u\00"
@m_bool    = private constant [11 x i8] c"plain bool\00"
@m_dbp     = private constant [10 x i8] c"dbp value\00"
@m_dbpc    = private constant [10 x i8] c"dbp count\00"
@m_dlba    = private constant [11 x i8] c"dlba value\00"
@m_dlbau   = private constant [10 x i8] c"dlba used\00"
@m_dba     = private constant [10 x i8] c"dba value\00"
@m_dbau    = private constant [10 x i8] c"dba usede\00"
@m_bss     = private constant [10 x i8] c"bss value\00"
@m_trunc   = private constant [13 x i8] c"trunc gives8\00"
@m_zero    = private constant [12 x i8] c"count0 ok  \00"
@m_fuzz    = private constant [14 x i8] c"fuzz codes ok\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %out32 = alloca [64 x i32], align 4
  %out64 = alloca [64 x i64], align 8
  %outb = alloca [256 x i8], align 1
  %used = alloca i64, align 8
  %cnt = alloca i64, align 8
  %rng = alloca i64, align 8
  %fuzzbuf = alloca [64 x i8], align 1

  ; ============ RLE run (bw=3) ============
  %r1 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_run, i64 2, i32 3, i64 4, ptr %out32)
  %r1ok = icmp eq i32 %r1, 0
  call void @ut_check(i1 %r1ok, ptr @m_rle_ret)
  br label %chk.rle

chk.rle:
  %rv0 = load i32, ptr %out32, align 4
  %rv0z = zext i32 %rv0 to i64
  call void @ut_check_eq(i64 %rv0z, i64 5, ptr @m_rle_v)
  %rp3 = getelementptr [64 x i32], ptr %out32, i64 0, i64 3
  %rv3 = load i32, ptr %rp3, align 4
  %rv3z = zext i32 %rv3 to i64
  call void @ut_check_eq(i64 %rv3z, i64 5, ptr @m_rle_v)

  ; ============ bit-packed run (bw=3) → 0..7 ============
  %r2 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_bp, i64 4, i32 3, i64 8, ptr %out32)
  %r2ok = icmp eq i32 %r2, 0
  call void @ut_check(i1 %r2ok, ptr @m_bp_ret)
  br label %bp.loop

bp.loop:
  %bi = phi i64 [ 0, %chk.rle ], [ %bi.n, %bp.loop ]
  %bpp = getelementptr [64 x i32], ptr %out32, i64 0, i64 %bi
  %bpv = load i32, ptr %bpp, align 4
  %bpvz = zext i32 %bpv to i64
  call void @ut_check_eq(i64 %bpvz, i64 %bi, ptr @m_bp_v)
  %bi.n = add i64 %bi, 1
  %bp.done = icmp uge i64 %bi.n, 8
  br i1 %bp.done, label %w32, label %bp.loop

  ; ============ RLE run bw=32 ============
w32:
  %r3 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_w32, i64 5, i32 32, i64 2, ptr %out32)
  %w0 = load i32, ptr %out32, align 4
  %w0z = zext i32 %w0 to i64
  call void @ut_check_eq(i64 %w0z, i64 305419896, ptr @m_w32_v)  ; 0x12345678
  %wp1 = getelementptr [64 x i32], ptr %out32, i64 0, i64 1
  %w1 = load i32, ptr %wp1, align 4
  %w1z = zext i32 %w1 to i64
  call void @ut_check_eq(i64 %w1z, i64 305419896, ptr @m_w32_v)

  ; ============ bit_width 0 → all zeros ============
  %r4 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_run, i64 2, i32 0, i64 5, ptr %out32)
  %z0 = load i32, ptr %out32, align 4
  %zp4 = getelementptr [64 x i32], ptr %out32, i64 0, i64 4
  %z4 = load i32, ptr %zp4, align 4
  %zor = or i32 %z0, %z4
  %ziszero = icmp eq i32 %zor, 0
  call void @ut_check(i1 %ziszero, ptr @m_bw0)

  ; ============ PLAIN INT32 ============
  %p1 = call i32 @universe_docparse_parquet_plain(ptr @plain_i32, i64 12, i32 1, i32 0, i64 3, ptr %outb, i64 256, ptr %used)
  %pi0 = load i32, ptr %outb, align 1
  %pi0z = sext i32 %pi0 to i64
  call void @ut_check_eq(i64 %pi0z, i64 1, ptr @m_i32)
  %pip1 = getelementptr i8, ptr %outb, i64 4
  %pi1 = load i32, ptr %pip1, align 1
  %pi1z = sext i32 %pi1 to i64
  call void @ut_check_eq(i64 %pi1z, i64 -2, ptr @m_i32)
  %pip2 = getelementptr i8, ptr %outb, i64 8
  %pi2 = load i32, ptr %pip2, align 1
  %pi2z = sext i32 %pi2 to i64
  call void @ut_check_eq(i64 %pi2z, i64 1000000, ptr @m_i32)
  %piu = load i64, ptr %used, align 8
  call void @ut_check_eq(i64 %piu, i64 12, ptr @m_i32u)

  ; ============ PLAIN DOUBLE ============
  %p2 = call i32 @universe_docparse_parquet_plain(ptr @plain_f64, i64 16, i32 5, i32 0, i64 2, ptr %outb, i64 256, ptr %used)
  %pd0 = load i64, ptr %outb, align 1
  call void @ut_check_eq(i64 %pd0, i64 4614253070214989087, ptr @m_f64)  ; 3.14 bits
  %pdp1 = getelementptr i8, ptr %outb, i64 8
  %pd1 = load i64, ptr %pdp1, align 1
  call void @ut_check_eq(i64 %pd1, i64 4611686018427387904, ptr @m_f64)  ; 2.0 bits

  ; ============ PLAIN BYTE_ARRAY ============
  %p3 = call i32 @universe_docparse_parquet_plain(ptr @plain_ba, i64 13, i32 6, i32 0, i64 2, ptr %outb, i64 256, ptr %used)
  %bl0 = load i32, ptr %outb, align 1
  %bl0z = zext i32 %bl0 to i64
  call void @ut_check_eq(i64 %bl0z, i64 2, ptr @m_ba)
  %bc0 = getelementptr i8, ptr %outb, i64 4
  %bch = load i8, ptr %bc0, align 1
  %bchz = zext i8 %bch to i64
  call void @ut_check_eq(i64 %bchz, i64 104, ptr @m_ba)  ; 'h'
  %bl1p = getelementptr i8, ptr %outb, i64 6
  %bl1 = load i32, ptr %bl1p, align 1
  %bl1z = zext i32 %bl1 to i64
  call void @ut_check_eq(i64 %bl1z, i64 3, ptr @m_ba)
  %bau = load i64, ptr %used, align 8
  call void @ut_check_eq(i64 %bau, i64 13, ptr @m_bau)

  ; ============ PLAIN BOOLEAN ============
  %p4 = call i32 @universe_docparse_parquet_plain(ptr @plain_bool, i64 1, i32 0, i32 0, i64 4, ptr %outb, i64 256, ptr %used)
  %bb0 = load i8, ptr %outb, align 1
  %bb0z = zext i8 %bb0 to i64
  call void @ut_check_eq(i64 %bb0z, i64 1, ptr @m_bool)
  %bb1p = getelementptr i8, ptr %outb, i64 1
  %bb1 = load i8, ptr %bb1p, align 1
  %bb1z = zext i8 %bb1 to i64
  call void @ut_check_eq(i64 %bb1z, i64 0, ptr @m_bool)
  %bb2p = getelementptr i8, ptr %outb, i64 2
  %bb2 = load i8, ptr %bb2p, align 1
  %bb2z = zext i8 %bb2 to i64
  call void @ut_check_eq(i64 %bb2z, i64 1, ptr @m_bool)

  ; ============ DELTA_BINARY_PACKED ============
  %d1 = call i32 @universe_docparse_parquet_delta_binary_packed(ptr @dbp, i64 10, ptr %out64, i64 64, ptr %cnt)
  %dc = load i64, ptr %cnt, align 8
  call void @ut_check_eq(i64 %dc, i64 5, ptr @m_dbpc)
  br label %dbp.loop

dbp.loop:
  %di = phi i64 [ 0, %w32 ], [ %di.n, %dbp.loop ]
  %dpp = getelementptr [64 x i64], ptr %out64, i64 0, i64 %di
  %dpv = load i64, ptr %dpp, align 8
  %dexp = add i64 %di, 1
  call void @ut_check_eq(i64 %dpv, i64 %dexp, ptr @m_dbp)
  %di.n = add i64 %di, 1
  %dbp.done = icmp uge i64 %di.n, 5
  br i1 %dbp.done, label %dlba, label %dbp.loop

  ; ============ DELTA_LENGTH_BYTE_ARRAY ============
dlba:
  %l1 = call i32 @universe_docparse_parquet_delta_length_byte_array(ptr @dlba, i64 16, i64 3, ptr %outb, i64 256, ptr %used)
  %ll0 = load i32, ptr %outb, align 1
  %ll0z = zext i32 %ll0 to i64
  call void @ut_check_eq(i64 %ll0z, i64 1, ptr @m_dlba)          ; len "a"
  %lc0 = getelementptr i8, ptr %outb, i64 4
  %lch = load i8, ptr %lc0, align 1
  %lchz = zext i8 %lch to i64
  call void @ut_check_eq(i64 %lchz, i64 97, ptr @m_dlba)         ; 'a'
  %ll1p = getelementptr i8, ptr %outb, i64 5
  %ll1 = load i32, ptr %ll1p, align 1
  %ll1z = zext i32 %ll1 to i64
  call void @ut_check_eq(i64 %ll1z, i64 2, ptr @m_dlba)          ; len "bb"
  %lc1 = getelementptr i8, ptr %outb, i64 9
  %lc1v = load i8, ptr %lc1, align 1
  %lc1z = zext i8 %lc1v to i64
  call void @ut_check_eq(i64 %lc1z, i64 98, ptr @m_dlba)         ; 'b'
  %ll2p = getelementptr i8, ptr %outb, i64 11
  %ll2 = load i32, ptr %ll2p, align 1
  %ll2z = zext i32 %ll2 to i64
  call void @ut_check_eq(i64 %ll2z, i64 3, ptr @m_dlba)          ; len "ccc"
  %lu = load i64, ptr %used, align 8
  call void @ut_check_eq(i64 %lu, i64 18, ptr @m_dlbau)

  ; ============ DELTA_BYTE_ARRAY ============
  %b1 = call i32 @universe_docparse_parquet_delta_byte_array(ptr @dba, i64 29, i64 3, ptr %outb, i64 256, ptr %used)
  ; value0 "abc"
  %ba0l = load i32, ptr %outb, align 1
  %ba0lz = zext i32 %ba0l to i64
  call void @ut_check_eq(i64 %ba0lz, i64 3, ptr @m_dba)
  %ba0c = getelementptr i8, ptr %outb, i64 4
  %ba0ch = load i8, ptr %ba0c, align 1
  %ba0chz = zext i8 %ba0ch to i64
  call void @ut_check_eq(i64 %ba0chz, i64 97, ptr @m_dba)        ; 'a'
  ; value1 "adef" at offset 7 : len@7, data@11
  %ba1lp = getelementptr i8, ptr %outb, i64 7
  %ba1l = load i32, ptr %ba1lp, align 1
  %ba1lz = zext i32 %ba1l to i64
  call void @ut_check_eq(i64 %ba1lz, i64 4, ptr @m_dba)
  %ba1c = getelementptr i8, ptr %outb, i64 11
  %ba1ch = load i8, ptr %ba1c, align 1
  %ba1chz = zext i8 %ba1ch to i64
  call void @ut_check_eq(i64 %ba1chz, i64 97, ptr @m_dba)        ; 'a' (reused prefix)
  %ba1c2p = getelementptr i8, ptr %outb, i64 12
  %ba1c2 = load i8, ptr %ba1c2p, align 1
  %ba1c2z = zext i8 %ba1c2 to i64
  call void @ut_check_eq(i64 %ba1c2z, i64 100, ptr @m_dba)       ; 'd'
  ; value2 "adghi" at offset 15 : len@15, data@19
  %ba2lp = getelementptr i8, ptr %outb, i64 15
  %ba2l = load i32, ptr %ba2lp, align 1
  %ba2lz = zext i32 %ba2l to i64
  call void @ut_check_eq(i64 %ba2lz, i64 5, ptr @m_dba)
  %ba2c = getelementptr i8, ptr %outb, i64 19
  %ba2ch = load i8, ptr %ba2c, align 1
  %ba2chz = zext i8 %ba2ch to i64
  call void @ut_check_eq(i64 %ba2chz, i64 97, ptr @m_dba)        ; 'a'
  %ba2c2p = getelementptr i8, ptr %outb, i64 21
  %ba2c2 = load i8, ptr %ba2c2p, align 1
  %ba2c2z = zext i8 %ba2c2 to i64
  call void @ut_check_eq(i64 %ba2c2z, i64 103, ptr @m_dba)       ; 'g'
  %bau2 = load i64, ptr %used, align 8
  call void @ut_check_eq(i64 %bau2, i64 24, ptr @m_dbau)

  ; ============ BYTE_STREAM_SPLIT (2 floats) ============
  %s1 = call i32 @universe_docparse_parquet_byte_stream_split(ptr @bss, i64 8, i32 4, i64 2, ptr %outb)
  %sf0 = load i32, ptr %outb, align 1
  %sf0z = zext i32 %sf0 to i64
  call void @ut_check_eq(i64 %sf0z, i64 1065353216, ptr @m_bss)  ; 1.0f = 0x3F800000
  %sf1p = getelementptr i8, ptr %outb, i64 4
  %sf1 = load i32, ptr %sf1p, align 1
  %sf1z = zext i32 %sf1 to i64
  call void @ut_check_eq(i64 %sf1z, i64 1073741824, ptr @m_bss)  ; 2.0f = 0x40000000

  ; ============ truncation → 8 ============
  ; RLE run header needs a value byte but dlen=1.
  %t1 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_run, i64 1, i32 3, i64 4, ptr %out32)
  %t1is8 = icmp eq i32 %t1, 8
  call void @ut_check(i1 %t1is8, ptr @m_trunc)
  ; delta with only 1 header byte (incomplete uleb).
  %t2 = call i32 @universe_docparse_parquet_delta_binary_packed(ptr @dbp, i64 1, ptr %out64, i64 64, ptr %cnt)
  %t2is8 = icmp eq i32 %t2, 8
  call void @ut_check(i1 %t2is8, ptr @m_trunc)

  ; ============ count 0 ============
  %c1 = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_run, i64 2, i32 3, i64 0, ptr %out32)
  %c1ok = icmp eq i32 %c1, 0
  call void @ut_check(i1 %c1ok, ptr @m_zero)
  %c2 = call i32 @universe_docparse_parquet_plain(ptr @plain_i32, i64 12, i32 1, i32 0, i64 0, ptr %outb, i64 256, ptr %used)
  %c2u = load i64, ptr %used, align 8
  %c2ok = icmp eq i64 %c2u, 0
  call void @ut_check(i1 %c2ok, ptr @m_zero)
  br label %fuzz.pre

fuzz.pre:
  ; ============ fuzz: truncated/mutated delta + rle under ASan ============
  store i64 88172645463325252, ptr %rng, align 8
  br label %fuzz.loop

fuzz.loop:
  %fi = phi i64 [ 0, %fuzz.pre ], [ %fi.n, %fuzz.iter ]
  %viol = phi i64 [ 0, %fuzz.pre ], [ %viol.n, %fuzz.iter ]
  %fdone = icmp uge i64 %fi, 20000
  br i1 %fdone, label %fuzz.fin, label %fuzz.body

fuzz.body:
  ; copy dba (29 bytes) into fuzzbuf, mutate one byte, pick a truncation length.
  call void @llvm.memcpy.p0.p0.i64(ptr %fuzzbuf, ptr @dba, i64 29, i1 false)
  %rmut = call i64 @ut_rand(ptr %rng)
  %midx = urem i64 %rmut, 29
  %rval = call i64 @ut_rand(ptr %rng)
  %mb = trunc i64 %rval to i8
  %mp = getelementptr [64 x i8], ptr %fuzzbuf, i64 0, i64 %midx
  store i8 %mb, ptr %mp, align 1
  %rtl = call i64 @ut_rand(ptr %rng)
  %tl = urem i64 %rtl, 30
  ; try each kernel on the mutated/truncated buffer; return codes must be documented.
  %f1 = call i32 @universe_docparse_parquet_delta_binary_packed(ptr %fuzzbuf, i64 %tl, ptr %out64, i64 64, ptr %cnt)
  %f1bad = call i1 @bad_code(i32 %f1)
  %v1 = zext i1 %f1bad to i64
  %f2 = call i32 @universe_docparse_parquet_delta_byte_array(ptr %fuzzbuf, i64 %tl, i64 3, ptr %outb, i64 256, ptr %used)
  %f2bad = call i1 @bad_code(i32 %f2)
  %v2 = zext i1 %f2bad to i64
  %f3 = call i32 @universe_docparse_parquet_delta_length_byte_array(ptr %fuzzbuf, i64 %tl, i64 3, ptr %outb, i64 256, ptr %used)
  %f3bad = call i1 @bad_code(i32 %f3)
  %v3 = zext i1 %f3bad to i64
  %f4 = call i32 @universe_docparse_parquet_rle_hybrid(ptr %fuzzbuf, i64 %tl, i32 5, i64 40, ptr %out32)
  %f4bad = call i1 @bad_code(i32 %f4)
  %v4 = zext i1 %f4bad to i64
  %f5 = call i32 @universe_docparse_parquet_byte_stream_split(ptr %fuzzbuf, i64 %tl, i32 4, i64 3, ptr %outb)
  %f5bad = call i1 @bad_code(i32 %f5)
  %v5 = zext i1 %f5bad to i64
  %vs1 = add i64 %v1, %v2
  %vs2 = add i64 %vs1, %v3
  %vs3 = add i64 %vs2, %v4
  %vs4 = add i64 %vs3, %v5
  br label %fuzz.iter

fuzz.iter:
  %viol.n = add i64 %viol, %vs4
  %fi.n = add i64 %fi, 1
  br label %fuzz.loop

fuzz.fin:
  call void @ut_check_eq(i64 %viol, i64 0, ptr @m_fuzz)

  %isbench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %isbench, label %bench, label %summary

bench:
  ; time the bit-packed unpack hot path.
  %t0 = call double @ut_now_sec()
  br label %bench.loop
bench.loop:
  %bii = phi i64 [ 0, %bench ], [ %bii.n, %bench.loop ]
  %bret = call i32 @universe_docparse_parquet_rle_hybrid(ptr @rle_bp, i64 4, i32 3, i64 8, ptr %out32)
  %bsink = load volatile i32, ptr %out32, align 4
  %bii.n = add i64 %bii, 1
  %bench.done = icmp uge i64 %bii.n, 1000000
  br i1 %bench.done, label %summary, label %bench.loop

summary:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; returns true if a decoder return code is NOT one of the documented values
; {0 OK, 3 SIZE_OVERFLOW, 6 FULL, 8 INVALID_ARG}. (malloc failure 2 also ok.)
define internal i1 @bad_code(i32 %c) {
entry:
  %ok0 = icmp eq i32 %c, 0
  %ok2 = icmp eq i32 %c, 2
  %ok3 = icmp eq i32 %c, 3
  %ok6 = icmp eq i32 %c, 6
  %ok8 = icmp eq i32 %c, 8
  %a = or i1 %ok0, %ok2
  %b = or i1 %a, %ok3
  %d = or i1 %b, %ok6
  %e = or i1 %d, %ok8
  %bad = xor i1 %e, true
  ret i1 %bad
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
