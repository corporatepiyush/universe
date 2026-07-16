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

; Tests for universe_simd_*: known-answer vectors; the CORE SIMD-first gate —
; for every op the VECTOR result must equal the SCALAR oracle over fixed-seed
; random buffers of many lengths (boundaries 0,1,15,16,17,31,32,...,65535);
; libc cross-checks (memchr for find_byte, memcmp for equal/compare); a --bench
; comparing vector vs scalar throughput.

; ---- module under test ----
declare i64 @universe_simd_find_byte(ptr, i64, i8)
declare i64 @universe_simd_find_byte_scalar(ptr, i64, i8)
declare i64 @universe_simd_find_crlf(ptr, i64)
declare i64 @universe_simd_find_crlf_scalar(ptr, i64)
declare i64 @universe_simd_index_of_any(ptr, i64, ptr, i64)
declare i64 @universe_simd_index_of_any_scalar(ptr, i64, ptr, i64)
declare i64 @universe_simd_count_byte(ptr, i64, i8)
declare i64 @universe_simd_count_byte_scalar(ptr, i64, i8)
declare i1  @universe_simd_equal(ptr, ptr, i64)
declare i1  @universe_simd_equal_scalar(ptr, ptr, i64)
declare i32 @universe_simd_compare(ptr, ptr, i64)
declare i32 @universe_simd_compare_scalar(ptr, ptr, i64)
declare void @universe_simd_to_lower_ascii(ptr, ptr, i64)
declare void @universe_simd_to_lower_ascii_scalar(ptr, ptr, i64)
declare void @universe_simd_to_upper_ascii(ptr, ptr, i64)
declare void @universe_simd_to_upper_ascii_scalar(ptr, ptr, i64)
declare i1  @universe_simd_is_ascii(ptr, i64)
declare i1  @universe_simd_is_ascii_scalar(ptr, i64)
declare i64 @universe_simd_validate_ascii(ptr, i64)
declare i64 @universe_simd_validate_ascii_scalar(ptr, i64)

; ---- libc ----
declare ptr @memchr(ptr, i32, i64)
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @memcpy(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

; ---- harness ----
declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

; ---- buffers ----
@g.a    = internal global [65536 x i8] zeroinitializer, align 16
@g.b    = internal global [65536 x i8] zeroinitializer, align 16
@g.asc  = internal global [65536 x i8] zeroinitializer, align 16
@g.adup = internal global [65536 x i8] zeroinitializer, align 16
@g.dv   = internal global [65536 x i8] zeroinitializer, align 16
@g.ds   = internal global [65536 x i8] zeroinitializer, align 16

@lens = internal constant [20 x i64]
  [ i64 0, i64 1, i64 2, i64 3, i64 15, i64 16, i64 17, i64 31, i64 32, i64 33,
    i64 63, i64 64, i64 65, i64 127, i64 128, i64 255, i64 256, i64 1000,
    i64 4096, i64 65535 ], align 8

; ---- mismatch counters (globals: simpler than threading phis) ----
@c.fb    = internal global i64 0, align 8
@c.fblc  = internal global i64 0, align 8
@c.fc    = internal global i64 0, align 8
@c.ioa   = internal global i64 0, align 8
@c.cnt   = internal global i64 0, align 8
@c.eq    = internal global i64 0, align 8
@c.eqt   = internal global i64 0, align 8
@c.cmp   = internal global i64 0, align 8
@c.cmplc = internal global i64 0, align 8
@c.low   = internal global i64 0, align 8
@c.up    = internal global i64 0, align 8
@c.asc   = internal global i64 0, align 8
@c.asc2  = internal global i64 0, align 8
@c.val   = internal global i64 0, align 8
@c.val2  = internal global i64 0, align 8
@c.vcons = internal global i64 0, align 8

; ---- known-answer buffers ----
@k.hello   = private unnamed_addr constant [11 x i8] c"hello world"
@k.crlf    = private unnamed_addr constant [8 x i8]  c"abc\0D\0Adef"
@k.nocrlf  = private unnamed_addr constant [7 x i8]  c"abc\0Ddef"
@k.crlf17  = private unnamed_addr constant [17 x i8] c"0123456789ABCDE\0D\0A"
@k.banana  = private unnamed_addr constant [6 x i8]  c"banana"
@k.vowels  = private unnamed_addr constant [5 x i8]  c"aeiou"
@k.xyz     = private unnamed_addr constant [3 x i8]  c"xyz"
@k.mixed   = private unnamed_addr constant [8 x i8]  c"HeLLo123"
@k.lowexp  = private unnamed_addr constant [8 x i8]  c"hello123"
@k.upexp   = private unnamed_addr constant [8 x i8]  c"HELLO123"
@k.abc     = private unnamed_addr constant [3 x i8]  c"abc"
@k.abd     = private unnamed_addr constant [3 x i8]  c"abd"
@k.nonasc  = private unnamed_addr constant [5 x i8]  c"ab\FFcd"

; ---- messages ----
@m.fbo    = private unnamed_addr constant [19 x i8] c"find_byte 'o' == 4\00"
@m.fbz    = private unnamed_addr constant [20 x i8] c"find_byte miss ==-1\00"
@m.fc     = private unnamed_addr constant [15 x i8] c"find_crlf == 3\00"
@m.fcn    = private unnamed_addr constant [20 x i8] c"find_crlf none ==-1\00"
@m.fc17   = private unnamed_addr constant [24 x i8] c"find_crlf boundary ==15\00"
@m.ioa    = private unnamed_addr constant [18 x i8] c"index_of_any == 1\00"
@m.ioan   = private unnamed_addr constant [23 x i8] c"index_of_any none ==-1\00"
@m.cnt    = private unnamed_addr constant [15 x i8] c"count 'a' == 3\00"
@m.eqt    = private unnamed_addr constant [16 x i8] c"equal same true\00"
@m.eqf    = private unnamed_addr constant [17 x i8] c"equal diff false\00"
@m.cmpn   = private unnamed_addr constant [16 x i8] c"compare abc<abd\00"
@m.cmpp   = private unnamed_addr constant [16 x i8] c"compare abd>abc\00"
@m.cmpe   = private unnamed_addr constant [16 x i8] c"compare eq == 0\00"
@m.low    = private unnamed_addr constant [15 x i8] c"to_lower known\00"
@m.up     = private unnamed_addr constant [15 x i8] c"to_upper known\00"
@m.lowip  = private unnamed_addr constant [18 x i8] c"to_lower in-place\00"
@m.asct   = private unnamed_addr constant [14 x i8] c"is_ascii true\00"
@m.ascf   = private unnamed_addr constant [15 x i8] c"is_ascii false\00"
@m.valk   = private unnamed_addr constant [20 x i8] c"validate_ascii == 2\00"
@m.valn   = private unnamed_addr constant [22 x i8] c"validate all ascii -1\00"

@r.fb    = private unnamed_addr constant [19 x i8] c"rnd find_byte v==s\00"
@r.fblc  = private unnamed_addr constant [20 x i8] c"rnd find_byte==memc\00"
@r.fc    = private unnamed_addr constant [19 x i8] c"rnd find_crlf v==s\00"
@r.ioa   = private unnamed_addr constant [17 x i8] c"rnd idx_any v==s\00"
@r.cnt   = private unnamed_addr constant [15 x i8] c"rnd count v==s\00"
@r.eq    = private unnamed_addr constant [15 x i8] c"rnd equal v==s\00"
@r.eqt   = private unnamed_addr constant [20 x i8] c"rnd equal-true v==s\00"
@r.cmp   = private unnamed_addr constant [17 x i8] c"rnd compare v==s\00"
@r.cmplc = private unnamed_addr constant [20 x i8] c"rnd compare==memcmp\00"
@r.low   = private unnamed_addr constant [18 x i8] c"rnd to_lower v==s\00"
@r.up    = private unnamed_addr constant [18 x i8] c"rnd to_upper v==s\00"
@r.asc   = private unnamed_addr constant [18 x i8] c"rnd is_ascii v==s\00"
@r.asc2  = private unnamed_addr constant [21 x i8] c"rnd is_ascii asc v=s\00"
@r.val   = private unnamed_addr constant [18 x i8] c"rnd validate v==s\00"
@r.val2  = private unnamed_addr constant [21 x i8] c"rnd validate asc v=s\00"
@r.vcons = private unnamed_addr constant [22 x i8] c"rnd validate/is_ascii\00"

@simd.fvsamp = internal global [16 x double] zeroinitializer, align 8
@simd.fssamp = internal global [16 x double] zeroinitializer, align 8
@simd.cvsamp = internal global [16 x double] zeroinitializer, align 8
@simd.cssamp = internal global [16 x double] zeroinitializer, align 8
@simd.lvsamp = internal global [16 x double] zeroinitializer, align 8
@simd.lssamp = internal global [16 x double] zeroinitializer, align 8
@lbl.simd.fv = private unnamed_addr constant [24 x i8] c"simd find_byte vec 64K \00"
@lbl.simd.fs = private unnamed_addr constant [24 x i8] c"simd find_byte scal 64K\00"
@lbl.simd.cv = private unnamed_addr constant [24 x i8] c"simd count_byte vec 64K\00"
@lbl.simd.cs = private unnamed_addr constant [25 x i8] c"simd count_byte scal 64K\00"
@lbl.simd.lv = private unnamed_addr constant [24 x i8] c"simd to_lower vec 64K  \00"
@lbl.simd.ls = private unnamed_addr constant [24 x i8] c"simd to_lower scal 64K \00"

; increment *%g by (i1 %c ? 1 : 0)
define internal void @bump(ptr %g, i1 %c) {
entry:
  %v = load i64, ptr %g, align 8
  %inc = zext i1 %c to i64
  %n = add i64 %v, %inc
  store i64 %n, ptr %g, align 8
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ============================ known-answer ============================
  ; find_byte 'o' in "hello world" -> 4 ; miss 'z' -> -1
  %fbo = call i64 @universe_simd_find_byte(ptr @k.hello, i64 11, i8 111)
  call void @ut_check_eq(i64 %fbo, i64 4, ptr @m.fbo)
  %fbz = call i64 @universe_simd_find_byte(ptr @k.hello, i64 11, i8 122)
  %fbz.ok = icmp eq i64 %fbz, -1
  call void @ut_check(i1 %fbz.ok, ptr @m.fbz)

  ; find_crlf
  %fc = call i64 @universe_simd_find_crlf(ptr @k.crlf, i64 8)
  call void @ut_check_eq(i64 %fc, i64 3, ptr @m.fc)
  %fcn = call i64 @universe_simd_find_crlf(ptr @k.nocrlf, i64 7)
  %fcn.ok = icmp eq i64 %fcn, -1
  call void @ut_check(i1 %fcn.ok, ptr @m.fcn)
  %fc17 = call i64 @universe_simd_find_crlf(ptr @k.crlf17, i64 17)
  call void @ut_check_eq(i64 %fc17, i64 15, ptr @m.fc17)

  ; index_of_any "hello" set "aeiou" -> 1 ; set "xyz" -> -1
  %ioa = call i64 @universe_simd_index_of_any(ptr @k.hello, i64 11, ptr @k.vowels, i64 5)
  call void @ut_check_eq(i64 %ioa, i64 1, ptr @m.ioa)
  %ioan = call i64 @universe_simd_index_of_any(ptr @k.hello, i64 11, ptr @k.xyz, i64 3)
  %ioan.ok = icmp eq i64 %ioan, -1
  call void @ut_check(i1 %ioan.ok, ptr @m.ioan)

  ; count 'a' in "banana" -> 3
  %cnt = call i64 @universe_simd_count_byte(ptr @k.banana, i64 6, i8 97)
  call void @ut_check_eq(i64 %cnt, i64 3, ptr @m.cnt)

  ; equal / compare
  %eqt = call i1 @universe_simd_equal(ptr @k.abc, ptr @k.abc, i64 3)
  call void @ut_check(i1 %eqt, ptr @m.eqt)
  %eqf = call i1 @universe_simd_equal(ptr @k.abc, ptr @k.abd, i64 3)
  %eqf.n = xor i1 %eqf, true
  call void @ut_check(i1 %eqf.n, ptr @m.eqf)
  %cmpn = call i32 @universe_simd_compare(ptr @k.abc, ptr @k.abd, i64 3)
  %cmpn.ok = icmp slt i32 %cmpn, 0
  call void @ut_check(i1 %cmpn.ok, ptr @m.cmpn)
  %cmpp = call i32 @universe_simd_compare(ptr @k.abd, ptr @k.abc, i64 3)
  %cmpp.ok = icmp sgt i32 %cmpp, 0
  call void @ut_check(i1 %cmpp.ok, ptr @m.cmpp)
  %cmpe = call i32 @universe_simd_compare(ptr @k.abc, ptr @k.abc, i64 3)
  %cmpe.ok = icmp eq i32 %cmpe, 0
  call void @ut_check(i1 %cmpe.ok, ptr @m.cmpe)

  ; to_lower / to_upper known
  call void @universe_simd_to_lower_ascii(ptr @g.dv, ptr @k.mixed, i64 8)
  %lc = call i32 @memcmp(ptr @g.dv, ptr @k.lowexp, i64 8)
  %lc.ok = icmp eq i32 %lc, 0
  call void @ut_check(i1 %lc.ok, ptr @m.low)
  call void @universe_simd_to_upper_ascii(ptr @g.dv, ptr @k.mixed, i64 8)
  %uc = call i32 @memcmp(ptr @g.dv, ptr @k.upexp, i64 8)
  %uc.ok = icmp eq i32 %uc, 0
  call void @ut_check(i1 %uc.ok, ptr @m.up)
  ; in-place lower
  call ptr @memcpy(ptr @g.dv, ptr @k.mixed, i64 8)
  call void @universe_simd_to_lower_ascii(ptr @g.dv, ptr @g.dv, i64 8)
  %lipc = call i32 @memcmp(ptr @g.dv, ptr @k.lowexp, i64 8)
  %lipc.ok = icmp eq i32 %lipc, 0
  call void @ut_check(i1 %lipc.ok, ptr @m.lowip)

  ; is_ascii / validate
  %asct = call i1 @universe_simd_is_ascii(ptr @k.hello, i64 11)
  call void @ut_check(i1 %asct, ptr @m.asct)
  %ascf = call i1 @universe_simd_is_ascii(ptr @k.nonasc, i64 5)
  %ascf.n = xor i1 %ascf, true
  call void @ut_check(i1 %ascf.n, ptr @m.ascf)
  %valk = call i64 @universe_simd_validate_ascii(ptr @k.nonasc, i64 5)
  call void @ut_check_eq(i64 %valk, i64 2, ptr @m.valk)
  %valn = call i64 @universe_simd_validate_ascii(ptr @k.hello, i64 11)
  %valn.ok = icmp eq i64 %valn, -1
  call void @ut_check(i1 %valn.ok, ptr @m.valn)

  ; ============================ fill random buffers ============================
  %s1 = alloca i64, align 8
  %s2 = alloca i64, align 8
  store i64 11400714819323198485, ptr %s1, align 8
  store i64 1442695040888963407, ptr %s2, align 8
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %entry ], [ %fi.next, %fill.head ]
  %r1 = call i64 @ut_rand(ptr %s1)
  %r2 = call i64 @ut_rand(ptr %s2)
  %a8 = trunc i64 %r1 to i8
  %b8 = trunc i64 %r2 to i8
  %ap = getelementptr inbounds nuw [65536 x i8], ptr @g.a, i64 0, i64 %fi
  store i8 %a8, ptr %ap, align 1
  %bp = getelementptr inbounds nuw [65536 x i8], ptr @g.b, i64 0, i64 %fi
  store i8 %b8, ptr %bp, align 1
  %dupp = getelementptr inbounds nuw [65536 x i8], ptr @g.adup, i64 0, i64 %fi
  store i8 %a8, ptr %dupp, align 1
  %asc8 = and i8 %a8, 127
  %ascp = getelementptr inbounds nuw [65536 x i8], ptr @g.asc, i64 0, i64 %fi
  store i8 %asc8, ptr %ascp, align 1
  %fi.next = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.next, 65536
  br i1 %fmore, label %fill.head, label %rnd.head

  ; ============================ random vector==scalar gate ============================
rnd.head:
  %li = phi i64 [ 0, %fill.head ], [ %li.next, %rnd.body ]
  %lp = getelementptr inbounds nuw [20 x i64], ptr @lens, i64 0, i64 %li
  %L = load i64, ptr %lp, align 8
  br label %rnd.body

rnd.body:
  ; -- find_byte (target 'A'=65) --
  %fbv = call i64 @universe_simd_find_byte(ptr @g.a, i64 %L, i8 65)
  %fbs = call i64 @universe_simd_find_byte_scalar(ptr @g.a, i64 %L, i8 65)
  %fb.mis = icmp ne i64 %fbv, %fbs
  call void @bump(ptr @c.fb, i1 %fb.mis)
  ; libc memchr cross-check
  %mc = call ptr @memchr(ptr @g.a, i32 65, i64 %L)
  %mc.null = icmp eq ptr %mc, null
  %mc.int = ptrtoint ptr %mc to i64
  %a.int = ptrtoint ptr @g.a to i64
  %mc.idx = sub i64 %mc.int, %a.int
  %mc.exp = select i1 %mc.null, i64 -1, i64 %mc.idx
  %fblc.mis = icmp ne i64 %fbv, %mc.exp
  call void @bump(ptr @c.fblc, i1 %fblc.mis)

  ; -- find_crlf --
  %fcv = call i64 @universe_simd_find_crlf(ptr @g.a, i64 %L)
  %fcs = call i64 @universe_simd_find_crlf_scalar(ptr @g.a, i64 %L)
  %fc.mis = icmp ne i64 %fcv, %fcs
  call void @bump(ptr @c.fc, i1 %fc.mis)

  ; -- index_of_any (set "aeiou") --
  %ioav = call i64 @universe_simd_index_of_any(ptr @g.a, i64 %L, ptr @k.vowels, i64 5)
  %ioas = call i64 @universe_simd_index_of_any_scalar(ptr @g.a, i64 %L, ptr @k.vowels, i64 5)
  %ioa.mis = icmp ne i64 %ioav, %ioas
  call void @bump(ptr @c.ioa, i1 %ioa.mis)

  ; -- count_byte (target 'A') --
  %cntv = call i64 @universe_simd_count_byte(ptr @g.a, i64 %L, i8 65)
  %cnts = call i64 @universe_simd_count_byte_scalar(ptr @g.a, i64 %L, i8 65)
  %cnt.mis = icmp ne i64 %cntv, %cnts
  call void @bump(ptr @c.cnt, i1 %cnt.mis)

  ; -- equal (a vs b: usually differ) --
  %eqv = call i1 @universe_simd_equal(ptr @g.a, ptr @g.b, i64 %L)
  %eqs = call i1 @universe_simd_equal_scalar(ptr @g.a, ptr @g.b, i64 %L)
  %eq.mis = xor i1 %eqv, %eqs
  call void @bump(ptr @c.eq, i1 %eq.mis)
  ; -- equal true path (a vs adup) --
  %eqtv = call i1 @universe_simd_equal(ptr @g.a, ptr @g.adup, i64 %L)
  %eqts = call i1 @universe_simd_equal_scalar(ptr @g.a, ptr @g.adup, i64 %L)
  %eqt.d = xor i1 %eqtv, %eqts
  %eqt.f = xor i1 %eqtv, true
  %eqt.mis = or i1 %eqt.d, %eqt.f
  call void @bump(ptr @c.eqt, i1 %eqt.mis)

  ; -- compare (sign) --
  %cmpv = call i32 @universe_simd_compare(ptr @g.a, ptr @g.b, i64 %L)
  %cmps = call i32 @universe_simd_compare_scalar(ptr @g.a, ptr @g.b, i64 %L)
  %vpos = icmp sgt i32 %cmpv, 0
  %spos = icmp sgt i32 %cmps, 0
  %vneg = icmp slt i32 %cmpv, 0
  %sneg = icmp slt i32 %cmps, 0
  %cmp.d1 = xor i1 %vpos, %spos
  %cmp.d2 = xor i1 %vneg, %sneg
  %cmp.mis = or i1 %cmp.d1, %cmp.d2
  call void @bump(ptr @c.cmp, i1 %cmp.mis)
  ; libc memcmp sign
  %lcmp = call i32 @memcmp(ptr @g.a, ptr @g.b, i64 %L)
  %lpos = icmp sgt i32 %lcmp, 0
  %lneg = icmp slt i32 %lcmp, 0
  %cmplc.d1 = xor i1 %vpos, %lpos
  %cmplc.d2 = xor i1 %vneg, %lneg
  %cmplc.mis = or i1 %cmplc.d1, %cmplc.d2
  call void @bump(ptr @c.cmplc, i1 %cmplc.mis)

  ; -- to_lower v==s --
  call void @universe_simd_to_lower_ascii(ptr @g.dv, ptr @g.a, i64 %L)
  call void @universe_simd_to_lower_ascii_scalar(ptr @g.ds, ptr @g.a, i64 %L)
  %low.c = call i32 @memcmp(ptr @g.dv, ptr @g.ds, i64 %L)
  %low.mis = icmp ne i32 %low.c, 0
  call void @bump(ptr @c.low, i1 %low.mis)
  ; -- to_upper v==s --
  call void @universe_simd_to_upper_ascii(ptr @g.dv, ptr @g.a, i64 %L)
  call void @universe_simd_to_upper_ascii_scalar(ptr @g.ds, ptr @g.a, i64 %L)
  %up.c = call i32 @memcmp(ptr @g.dv, ptr @g.ds, i64 %L)
  %up.mis = icmp ne i32 %up.c, 0
  call void @bump(ptr @c.up, i1 %up.mis)

  ; -- is_ascii on g.a (mostly non-ascii) --
  %ascv = call i1 @universe_simd_is_ascii(ptr @g.a, i64 %L)
  %ascs = call i1 @universe_simd_is_ascii_scalar(ptr @g.a, i64 %L)
  %asc.mis = xor i1 %ascv, %ascs
  call void @bump(ptr @c.asc, i1 %asc.mis)
  ; -- is_ascii on g.asc (all ascii: exercises no-hit vector continuation) --
  %asc2v = call i1 @universe_simd_is_ascii(ptr @g.asc, i64 %L)
  %asc2s = call i1 @universe_simd_is_ascii_scalar(ptr @g.asc, i64 %L)
  %asc2.mis = xor i1 %asc2v, %asc2s
  call void @bump(ptr @c.asc2, i1 %asc2.mis)

  ; -- validate_ascii on g.a --
  %valv = call i64 @universe_simd_validate_ascii(ptr @g.a, i64 %L)
  %vals = call i64 @universe_simd_validate_ascii_scalar(ptr @g.a, i64 %L)
  %val.mis = icmp ne i64 %valv, %vals
  call void @bump(ptr @c.val, i1 %val.mis)
  ; -- validate_ascii on g.asc (all ascii -> -1) --
  %val2v = call i64 @universe_simd_validate_ascii(ptr @g.asc, i64 %L)
  %val2s = call i64 @universe_simd_validate_ascii_scalar(ptr @g.asc, i64 %L)
  %val2.mis = icmp ne i64 %val2v, %val2s
  call void @bump(ptr @c.val2, i1 %val2.mis)
  ; -- consistency: (validate(asc)==-1) iff is_ascii(asc) --
  %val2.neg1 = icmp eq i64 %val2v, -1
  %vcons.mis = xor i1 %val2.neg1, %asc2v
  call void @bump(ptr @c.vcons, i1 %vcons.mis)

  %li.next = add nuw i64 %li, 1
  %lmore = icmp ult i64 %li.next, 20
  br i1 %lmore, label %rnd.head, label %rnd.done

rnd.done:
  %v.fb    = load i64, ptr @c.fb, align 8
  call void @ut_check_eq(i64 %v.fb, i64 0, ptr @r.fb)
  %v.fblc  = load i64, ptr @c.fblc, align 8
  call void @ut_check_eq(i64 %v.fblc, i64 0, ptr @r.fblc)
  %v.fc    = load i64, ptr @c.fc, align 8
  call void @ut_check_eq(i64 %v.fc, i64 0, ptr @r.fc)
  %v.ioa   = load i64, ptr @c.ioa, align 8
  call void @ut_check_eq(i64 %v.ioa, i64 0, ptr @r.ioa)
  %v.cnt   = load i64, ptr @c.cnt, align 8
  call void @ut_check_eq(i64 %v.cnt, i64 0, ptr @r.cnt)
  %v.eq    = load i64, ptr @c.eq, align 8
  call void @ut_check_eq(i64 %v.eq, i64 0, ptr @r.eq)
  %v.eqt   = load i64, ptr @c.eqt, align 8
  call void @ut_check_eq(i64 %v.eqt, i64 0, ptr @r.eqt)
  %v.cmp   = load i64, ptr @c.cmp, align 8
  call void @ut_check_eq(i64 %v.cmp, i64 0, ptr @r.cmp)
  %v.cmplc = load i64, ptr @c.cmplc, align 8
  call void @ut_check_eq(i64 %v.cmplc, i64 0, ptr @r.cmplc)
  %v.low   = load i64, ptr @c.low, align 8
  call void @ut_check_eq(i64 %v.low, i64 0, ptr @r.low)
  %v.up    = load i64, ptr @c.up, align 8
  call void @ut_check_eq(i64 %v.up, i64 0, ptr @r.up)
  %v.asc   = load i64, ptr @c.asc, align 8
  call void @ut_check_eq(i64 %v.asc, i64 0, ptr @r.asc)
  %v.asc2  = load i64, ptr @c.asc2, align 8
  call void @ut_check_eq(i64 %v.asc2, i64 0, ptr @r.asc2)
  %v.val   = load i64, ptr @c.val, align 8
  call void @ut_check_eq(i64 %v.val, i64 0, ptr @r.val)
  %v.val2  = load i64, ptr @c.val2, align 8
  call void @ut_check_eq(i64 %v.val2, i64 0, ptr @r.val2)
  %v.vcons = load i64, ptr @c.vcons, align 8
  call void @ut_check_eq(i64 %v.vcons, i64 0, ptr @r.vcons)

  ; ============================ bench ============================
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; Each kernel: 17 reps of a 2000-op batch over the 64 KiB all-ascii buffer
  ; (search 0xFF => never found => full scan); discard rep 0 (warm-up), report
  ; over the remaining 16. ops/rep = 2000 * 65536 = 131072000 (ns per byte).
  br label %bf.v.rep
bf.v.rep:
  %fvrep = phi i64 [ 0, %bench ], [ %fvrep.n, %bf.v.next ]
  %fvt0 = call double @ut_now_sec()
  br label %bf.v.head
bf.v.head:
  %bfvi = phi i64 [ 0, %bf.v.rep ], [ %bfvi.n, %bf.v.head ]
  %bfv = call i64 @universe_simd_find_byte(ptr @g.asc, i64 65536, i8 -1)
  %bfvi.n = add nuw i64 %bfvi, 1
  %bfv.m = icmp ult i64 %bfvi.n, 2000
  br i1 %bfv.m, label %bf.v.head, label %bf.v.rep.done
bf.v.rep.done:
  %fvt1 = call double @ut_now_sec()
  %fvel = fsub double %fvt1, %fvt0
  %fvkeep = icmp ugt i64 %fvrep, 0
  br i1 %fvkeep, label %bf.v.store, label %bf.v.next
bf.v.store:
  %fvidx = sub i64 %fvrep, 1
  %fvsp = getelementptr inbounds [16 x double], ptr @simd.fvsamp, i64 0, i64 %fvidx
  store double %fvel, ptr %fvsp, align 8
  br label %bf.v.next
bf.v.next:
  %fvrep.n = add nuw i64 %fvrep, 1
  %fvmore = icmp ult i64 %fvrep.n, 17
  br i1 %fvmore, label %bf.v.rep, label %bf.v.report
bf.v.report:
  call void @ut_report_dist(ptr @simd.fvsamp, i64 16, i64 131072000, ptr @lbl.simd.fv)
  br label %bf.s.rep

bf.s.rep:
  %fsrep = phi i64 [ 0, %bf.v.report ], [ %fsrep.n, %bf.s.next ]
  %fst0 = call double @ut_now_sec()
  br label %bf.s.head
bf.s.head:
  %bfsi = phi i64 [ 0, %bf.s.rep ], [ %bfsi.n, %bf.s.head ]
  %bfs = call i64 @universe_simd_find_byte_scalar(ptr @g.asc, i64 65536, i8 -1)
  %bfsi.n = add nuw i64 %bfsi, 1
  %bfs.m = icmp ult i64 %bfsi.n, 2000
  br i1 %bfs.m, label %bf.s.head, label %bf.s.rep.done
bf.s.rep.done:
  %fst1 = call double @ut_now_sec()
  %fsel = fsub double %fst1, %fst0
  %fskeep = icmp ugt i64 %fsrep, 0
  br i1 %fskeep, label %bf.s.store, label %bf.s.next
bf.s.store:
  %fsidx = sub i64 %fsrep, 1
  %fssp = getelementptr inbounds [16 x double], ptr @simd.fssamp, i64 0, i64 %fsidx
  store double %fsel, ptr %fssp, align 8
  br label %bf.s.next
bf.s.next:
  %fsrep.n = add nuw i64 %fsrep, 1
  %fsmore = icmp ult i64 %fsrep.n, 17
  br i1 %fsmore, label %bf.s.rep, label %bf.s.report
bf.s.report:
  call void @ut_report_dist(ptr @simd.fssamp, i64 16, i64 131072000, ptr @lbl.simd.fs)
  br label %bc.v.rep

bc.v.rep:
  %cvrep = phi i64 [ 0, %bf.s.report ], [ %cvrep.n, %bc.v.next ]
  %cvt0 = call double @ut_now_sec()
  br label %bc.v.head
bc.v.head:
  %bcvi = phi i64 [ 0, %bc.v.rep ], [ %bcvi.n, %bc.v.head ]
  %bcv = call i64 @universe_simd_count_byte(ptr @g.asc, i64 65536, i8 -1)
  %bcvi.n = add nuw i64 %bcvi, 1
  %bcv.m = icmp ult i64 %bcvi.n, 2000
  br i1 %bcv.m, label %bc.v.head, label %bc.v.rep.done
bc.v.rep.done:
  %cvt1 = call double @ut_now_sec()
  %cvel = fsub double %cvt1, %cvt0
  %cvkeep = icmp ugt i64 %cvrep, 0
  br i1 %cvkeep, label %bc.v.store, label %bc.v.next
bc.v.store:
  %cvidx = sub i64 %cvrep, 1
  %cvsp = getelementptr inbounds [16 x double], ptr @simd.cvsamp, i64 0, i64 %cvidx
  store double %cvel, ptr %cvsp, align 8
  br label %bc.v.next
bc.v.next:
  %cvrep.n = add nuw i64 %cvrep, 1
  %cvmore = icmp ult i64 %cvrep.n, 17
  br i1 %cvmore, label %bc.v.rep, label %bc.v.report
bc.v.report:
  call void @ut_report_dist(ptr @simd.cvsamp, i64 16, i64 131072000, ptr @lbl.simd.cv)
  br label %bc.s.rep

bc.s.rep:
  %csrep = phi i64 [ 0, %bc.v.report ], [ %csrep.n, %bc.s.next ]
  %cst0 = call double @ut_now_sec()
  br label %bc.s.head
bc.s.head:
  %bcsi = phi i64 [ 0, %bc.s.rep ], [ %bcsi.n, %bc.s.head ]
  %bcs = call i64 @universe_simd_count_byte_scalar(ptr @g.asc, i64 65536, i8 -1)
  %bcsi.n = add nuw i64 %bcsi, 1
  %bcs.m = icmp ult i64 %bcsi.n, 2000
  br i1 %bcs.m, label %bc.s.head, label %bc.s.rep.done
bc.s.rep.done:
  %cst1 = call double @ut_now_sec()
  %csel = fsub double %cst1, %cst0
  %cskeep = icmp ugt i64 %csrep, 0
  br i1 %cskeep, label %bc.s.store, label %bc.s.next
bc.s.store:
  %csidx = sub i64 %csrep, 1
  %cssp = getelementptr inbounds [16 x double], ptr @simd.cssamp, i64 0, i64 %csidx
  store double %csel, ptr %cssp, align 8
  br label %bc.s.next
bc.s.next:
  %csrep.n = add nuw i64 %csrep, 1
  %csmore = icmp ult i64 %csrep.n, 17
  br i1 %csmore, label %bc.s.rep, label %bc.s.report
bc.s.report:
  call void @ut_report_dist(ptr @simd.cssamp, i64 16, i64 131072000, ptr @lbl.simd.cs)
  br label %bl.v.rep

bl.v.rep:
  %lvrep = phi i64 [ 0, %bc.s.report ], [ %lvrep.n, %bl.v.next ]
  %lvt0 = call double @ut_now_sec()
  br label %bl.v.head
bl.v.head:
  %blvi = phi i64 [ 0, %bl.v.rep ], [ %blvi.n, %bl.v.head ]
  call void @universe_simd_to_lower_ascii(ptr @g.dv, ptr @g.asc, i64 65536)
  %blvi.n = add nuw i64 %blvi, 1
  %blv.m = icmp ult i64 %blvi.n, 2000
  br i1 %blv.m, label %bl.v.head, label %bl.v.rep.done
bl.v.rep.done:
  %lvt1 = call double @ut_now_sec()
  %lvel = fsub double %lvt1, %lvt0
  %lvkeep = icmp ugt i64 %lvrep, 0
  br i1 %lvkeep, label %bl.v.store, label %bl.v.next
bl.v.store:
  %lvidx = sub i64 %lvrep, 1
  %lvsp = getelementptr inbounds [16 x double], ptr @simd.lvsamp, i64 0, i64 %lvidx
  store double %lvel, ptr %lvsp, align 8
  br label %bl.v.next
bl.v.next:
  %lvrep.n = add nuw i64 %lvrep, 1
  %lvmore = icmp ult i64 %lvrep.n, 17
  br i1 %lvmore, label %bl.v.rep, label %bl.v.report
bl.v.report:
  call void @ut_report_dist(ptr @simd.lvsamp, i64 16, i64 131072000, ptr @lbl.simd.lv)
  br label %bl.s.rep

bl.s.rep:
  %lsrep = phi i64 [ 0, %bl.v.report ], [ %lsrep.n, %bl.s.next ]
  %lst0 = call double @ut_now_sec()
  br label %bl.s.head
bl.s.head:
  %blsi = phi i64 [ 0, %bl.s.rep ], [ %blsi.n, %bl.s.head ]
  call void @universe_simd_to_lower_ascii_scalar(ptr @g.ds, ptr @g.asc, i64 65536)
  %blsi.n = add nuw i64 %blsi, 1
  %bls.m = icmp ult i64 %blsi.n, 2000
  br i1 %bls.m, label %bl.s.head, label %bl.s.rep.done
bl.s.rep.done:
  %lst1 = call double @ut_now_sec()
  %lsel = fsub double %lst1, %lst0
  %lskeep = icmp ugt i64 %lsrep, 0
  br i1 %lskeep, label %bl.s.store, label %bl.s.next
bl.s.store:
  %lsidx = sub i64 %lsrep, 1
  %lssp = getelementptr inbounds [16 x double], ptr @simd.lssamp, i64 0, i64 %lsidx
  store double %lsel, ptr %lssp, align 8
  br label %bl.s.next
bl.s.next:
  %lsrep.n = add nuw i64 %lsrep, 1
  %lsmore = icmp ult i64 %lsrep.n, 17
  br i1 %lsmore, label %bl.s.rep, label %bl.s.report
bl.s.report:
  call void @ut_report_dist(ptr @simd.lssamp, i64 16, i64 131072000, ptr @lbl.simd.ls)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
