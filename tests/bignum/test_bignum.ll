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

; Tests for the bignum module. Identities over fixed-seed random inputs, an
; i128 reference for single-limb multiply, divmod reconstruction (q*v+r==u,
; r<v), Montgomery product identity, and KNOWN modexp/modinv/gcd vectors.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)

declare i64 @universe_bignum_add_n(ptr, ptr, ptr, i64)
declare i64 @universe_bignum_sub_n(ptr, ptr, ptr, i64)
declare i32 @universe_bignum_cmp_n(ptr, ptr, i64)
declare i32 @universe_bignum_is_zero_n(ptr, i64)
declare i64 @universe_bignum_bit_length_n(ptr, i64)
declare i64 @universe_bignum_shl_bits(ptr, ptr, i64, i64)
declare void @universe_bignum_shr_bits(ptr, ptr, i64, i64)
declare void @universe_bignum_mul(ptr, ptr, i64, ptr, i64)
declare i32 @universe_bignum_divmod(ptr, ptr, ptr, i64, ptr, i64)
declare i32 @universe_bignum_mod(ptr, ptr, i64, ptr, i64)
declare i64 @universe_bignum_normalize_len(ptr, i64)
declare void @universe_bignum_set_u64(ptr, i64, i64)
declare i64 @universe_bignum_mont_n0inv(ptr)
declare void @universe_bignum_mont_rr(ptr, ptr, i64)
declare void @universe_bignum_montmul(ptr, ptr, ptr, ptr, i64, i64, ptr)
declare i32 @universe_bignum_modexp(ptr, ptr, i64, ptr, i64, ptr, i64)
declare i32 @universe_bignum_modinv(ptr, ptr, ptr, i64)
declare i32 @universe_bignum_gcd(ptr, ptr, ptr, i64)

@m.addsub = private unnamed_addr constant [22 x i8] c"(a+b)-b==a violations\00", align 1
@m.mul1   = private unnamed_addr constant [24 x i8] c"1-limb mul vs i128 viol\00", align 1
@m.shl    = private unnamed_addr constant [20 x i8] c"shl==mul2^k viol   \00", align 1
@m.dm     = private unnamed_addr constant [22 x i8] c"divmod q*v+r==u viol \00", align 1
@m.dmr    = private unnamed_addr constant [18 x i8] c"divmod r<v viol  \00", align 1
@m.me1    = private unnamed_addr constant [18 x i8] c"3^7 mod 101 == 66\00", align 1
@m.me2    = private unnamed_addr constant [20 x i8] c"4^13 mod 497 == 445\00", align 1
@m.mecx   = private unnamed_addr constant [22 x i8] c"modexp vs naive viol \00", align 1
@m.mont   = private unnamed_addr constant [22 x i8] c"montmul product viol \00", align 1
@m.inv    = private unnamed_addr constant [20 x i8] c"a*inv(a) mod n ==1 \00", align 1
@m.gcd1   = private unnamed_addr constant [16 x i8] c"gcd(48,36)==12 \00", align 1
@m.gcd2   = private unnamed_addr constant [14 x i8] c"gcd(17,5)==1 \00", align 1
@m.izero  = private unnamed_addr constant [14 x i8] c"is_zero_n ok \00", align 1
@m.bitlen = private unnamed_addr constant [14 x i8] c"bit_length ok\00", align 1

@bn.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.bn = private unnamed_addr constant [21 x i8] c"bignum modexp 256bit\00"

; fill n limbs of buf with random data
define internal void @fillrand(ptr %buf, i64 %n, ptr %st) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %p = getelementptr inbounds i64, ptr %buf, i64 %i
  store i64 %r, ptr %p, align 8
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; r = (x*y) mod n  (s limbs each; tmp2s: 2s scratch)
define internal void @mulmod(ptr %r, ptr %x, ptr %y, ptr %n, i64 %s, i64 %nlen, ptr %tmp2s) {
entry:
  call void @universe_bignum_mul(ptr %tmp2s, ptr %x, i64 %s, ptr %y, i64 %s)
  %s2 = shl i64 %s, 1
  %rc = call i32 @universe_bignum_mod(ptr %r, ptr %tmp2s, i64 %s2, ptr %n, i64 %nlen)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8

  ; work buffers
  %a4 = alloca [4 x i64], align 8
  %b4 = alloca [4 x i64], align 8
  %t4 = alloca [4 x i64], align 8
  %u8 = alloca [8 x i64], align 8
  %v4 = alloca [4 x i64], align 8
  %q9 = alloca [9 x i64], align 8
  %r9 = alloca [9 x i64], align 8
  %qv9 = alloca [9 x i64], align 8
  %sum9 = alloca [9 x i64], align 8
  %uext9 = alloca [9 x i64], align 8

  %bench = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %bench, label %dobench, label %tests

tests:
  ; ---- (a+b)-b == a and cmp/add/sub, s=4, 300 iters
  br label %as.loop
as.loop:
  %as.i = phi i64 [ 0, %tests ], [ %as.in, %as.cont ]
  %as.v = phi i64 [ 0, %tests ], [ %as.vn, %as.cont ]
  call void @fillrand(ptr %a4, i64 4, ptr %seed)
  call void @fillrand(ptr %b4, i64 4, ptr %seed)
  ; t = a + b  (5-limb: store carry into a spare? use 4-limb + ignore, then subtract)
  %ascar = call i64 @universe_bignum_add_n(ptr %t4, ptr %a4, ptr %b4, i64 4)
  ; t = t - b  (borrow must equal ascar to cancel; ignore)
  %asbor = call i64 @universe_bignum_sub_n(ptr %t4, ptr %t4, ptr %b4, i64 4)
  ; compare t vs a
  %ascmp = call i32 @universe_bignum_cmp_n(ptr %t4, ptr %a4, i64 4)
  %asok = icmp eq i32 %ascmp, 0
  %asbad = xor i1 %asok, true
  %asaddv = zext i1 %asbad to i64
  %as.vn = add i64 %as.v, %asaddv
  br label %as.cont
as.cont:
  %as.in = add i64 %as.i, 1
  %as.more = icmp ult i64 %as.in, 300
  br i1 %as.more, label %as.loop, label %as.done
as.done:
  call void @ut_check_eq(i64 %as.v, i64 0, ptr @m.addsub)

  ; ---- 1-limb mul vs i128, 300 iters
  br label %m1.loop
m1.loop:
  %m1.i = phi i64 [ 0, %as.done ], [ %m1.in, %m1.cont ]
  %m1.v = phi i64 [ 0, %as.done ], [ %m1.vn, %m1.cont ]
  %m1a = call i64 @ut_rand(ptr %seed)
  %m1b = call i64 @ut_rand(ptr %seed)
  store i64 %m1a, ptr %a4, align 8
  store i64 %m1b, ptr %b4, align 8
  call void @universe_bignum_mul(ptr %t4, ptr %a4, i64 1, ptr %b4, i64 1)
  %t4lo = load i64, ptr %t4, align 8
  %t4hip = getelementptr inbounds i64, ptr %t4, i64 1
  %t4hi = load i64, ptr %t4hip, align 8
  %m1a128 = zext i64 %m1a to i128
  %m1b128 = zext i64 %m1b to i128
  %m1p = mul i128 %m1a128, %m1b128
  %m1lo = trunc i128 %m1p to i64
  %m1hish = lshr i128 %m1p, 64
  %m1hi = trunc i128 %m1hish to i64
  %loeq = icmp eq i64 %t4lo, %m1lo
  %hieq = icmp eq i64 %t4hi, %m1hi
  %m1ok = and i1 %loeq, %hieq
  %m1bad = xor i1 %m1ok, true
  %m1add = zext i1 %m1bad to i64
  %m1.vn = add i64 %m1.v, %m1add
  br label %m1.cont
m1.cont:
  %m1.in = add i64 %m1.i, 1
  %m1.more = icmp ult i64 %m1.in, 300
  br i1 %m1.more, label %m1.loop, label %m1.done
m1.done:
  call void @ut_check_eq(i64 %m1.v, i64 0, ptr @m.mul1)

  ; ---- shl_bits(k) == mul by 2^k, k in [0,63], 200 iters
  br label %sh.loop
sh.loop:
  %sh.i = phi i64 [ 0, %m1.done ], [ %sh.in, %sh.cont ]
  %sh.v = phi i64 [ 0, %m1.done ], [ %sh.vn, %sh.cont ]
  call void @fillrand(ptr %a4, i64 4, ptr %seed)
  %sh.kr = call i64 @ut_rand(ptr %seed)
  %sh.k = urem i64 %sh.kr, 64
  ; t = a << k  (returns overflow limb -> ignore; compare within 4 limbs vs mul)
  %shov = call i64 @universe_bignum_shl_bits(ptr %t4, ptr %a4, i64 4, i64 %sh.k)
  ; b = 2^k as 1-limb (only valid k<64; k up to 63 fits)
  %shpow = shl i64 1, %sh.k
  store i64 %shpow, ptr %b4, align 8
  ; qv = a * (2^k)  (4 x 1 -> 5 limbs); low 4 limbs must equal t
  call void @universe_bignum_mul(ptr %qv9, ptr %a4, i64 4, ptr %b4, i64 1)
  %shcmp = call i32 @universe_bignum_cmp_n(ptr %t4, ptr %qv9, i64 4)
  %shok = icmp eq i32 %shcmp, 0
  %shbad = xor i1 %shok, true
  %shadd = zext i1 %shbad to i64
  %sh.vn = add i64 %sh.v, %shadd
  br label %sh.cont
sh.cont:
  %sh.in = add i64 %sh.i, 1
  %sh.more = icmp ult i64 %sh.in, 200
  br i1 %sh.more, label %sh.loop, label %sh.done
sh.done:
  call void @ut_check_eq(i64 %sh.v, i64 0, ptr @m.shl)

  ; ---- divmod: u(8) / v(4), check q*v+r==u and r<v, 300 iters
  br label %dm.loop
dm.loop:
  %dm.i = phi i64 [ 0, %sh.done ], [ %dm.in, %dm.cont ]
  %dm.v = phi i64 [ 0, %sh.done ], [ %dm.vn, %dm.cont ]
  %dm.rv = phi i64 [ 0, %sh.done ], [ %dm.rvn, %dm.cont ]
  call void @fillrand(ptr %u8, i64 8, ptr %seed)
  call void @fillrand(ptr %v4, i64 4, ptr %seed)
  %vlen0 = call i64 @universe_bignum_normalize_len(ptr %v4, i64 4)
  %vlenz = icmp eq i64 %vlen0, 0
  %vlen = select i1 %vlenz, i64 1, i64 %vlen0
  br i1 %vlenz, label %dm.fixv, label %dm.go
dm.fixv:
  store i64 1, ptr %v4, align 8
  br label %dm.go
dm.go:
  ; zero q9,r9
  call void @universe_bignum_set_u64(ptr %q9, i64 9, i64 0)
  call void @universe_bignum_set_u64(ptr %r9, i64 9, i64 0)
  %dmrc = call i32 @universe_bignum_divmod(ptr %q9, ptr %r9, ptr %u8, i64 8, ptr %v4, i64 %vlen)
  ; qlen = 8 - vlen + 1
  %qlen0 = sub i64 8, %vlen
  %qlen = add i64 %qlen0, 1
  ; qv = q * v  (qlen+vlen = 9 limbs)
  call void @universe_bignum_mul(ptr %qv9, ptr %q9, i64 %qlen, ptr %v4, i64 %vlen)
  ; sum = qv + r  (r has vlen limbs, but r9 buffer is 9-wide zero-extended)
  %dmcar = call i64 @universe_bignum_add_n(ptr %sum9, ptr %qv9, ptr %r9, i64 9)
  ; uext = u zero-extended to 9
  call void @universe_bignum_set_u64(ptr %uext9, i64 9, i64 0)
  call void @llvm.memcpy.p0.p0.i64(ptr %uext9, ptr %u8, i64 64, i1 false)
  %dmcmp = call i32 @universe_bignum_cmp_n(ptr %sum9, ptr %uext9, i64 9)
  %dmok = icmp eq i32 %dmcmp, 0
  %dmbad = xor i1 %dmok, true
  %dmadd = zext i1 %dmbad to i64
  %dm.vn = add i64 %dm.v, %dmadd
  br label %dm.rcheck
dm.rcheck:
  ; r < v  (compare vlen limbs)
  %rcmp = call i32 @universe_bignum_cmp_n(ptr %r9, ptr %v4, i64 %vlen)
  %rlt = icmp eq i32 %rcmp, -1
  %rbad = xor i1 %rlt, true
  %radd = zext i1 %rbad to i64
  %dm.rvn = add i64 %dm.rv, %radd
  br label %dm.cont
dm.cont:
  %dm.in = add i64 %dm.i, 1
  %dm.more = icmp ult i64 %dm.in, 300
  br i1 %dm.more, label %dm.loop, label %dm.done
dm.done:
  call void @ut_check_eq(i64 %dm.v, i64 0, ptr @m.dm)
  call void @ut_check_eq(i64 %dm.rv, i64 0, ptr @m.dmr)

  ; ---- is_zero_n and bit_length_n
  call void @universe_bignum_set_u64(ptr %a4, i64 4, i64 0)
  %izz = call i32 @universe_bignum_is_zero_n(ptr %a4, i64 4)
  %izok0 = icmp eq i32 %izz, 1
  call void @universe_bignum_set_u64(ptr %a4, i64 4, i64 5)
  %iznz = call i32 @universe_bignum_is_zero_n(ptr %a4, i64 4)
  %izok1 = icmp eq i32 %iznz, 0
  %izok = and i1 %izok0, %izok1
  call void @ut_check(i1 %izok, ptr @m.izero)
  ; bit_length: value 2^70 + 1 -> 71 bits.  set limb1 = 64 (2^70 => limb1 bit6)
  call void @universe_bignum_set_u64(ptr %a4, i64 4, i64 1)
  %bl.p1 = getelementptr inbounds [4 x i64], ptr %a4, i64 0, i64 1
  store i64 64, ptr %bl.p1, align 8       ; 64 = 2^6 -> bit 70 overall
  %bl = call i64 @universe_bignum_bit_length_n(ptr %a4, i64 4)
  %blok = icmp eq i64 %bl, 71
  call void @ut_check(i1 %blok, ptr @m.bitlen)

  ; ---- modexp known vectors (s=1)
  ; 3^7 mod 101 == 66
  %me.base = alloca [4 x i64], align 8
  %me.exp = alloca [4 x i64], align 8
  %me.n = alloca [4 x i64], align 8
  %me.r = alloca [4 x i64], align 8
  store i64 3, ptr %me.base, align 8
  store i64 7, ptr %me.exp, align 8
  store i64 101, ptr %me.n, align 8
  %me1rc = call i32 @universe_bignum_modexp(ptr %me.r, ptr %me.base, i64 1, ptr %me.exp, i64 1, ptr %me.n, i64 1)
  %me1v = load i64, ptr %me.r, align 8
  %me1ok = icmp eq i64 %me1v, 66
  call void @ut_check(i1 %me1ok, ptr @m.me1)
  ; 4^13 mod 497 == 445
  store i64 4, ptr %me.base, align 8
  store i64 13, ptr %me.exp, align 8
  store i64 497, ptr %me.n, align 8
  %me2rc = call i32 @universe_bignum_modexp(ptr %me.r, ptr %me.base, i64 1, ptr %me.exp, i64 1, ptr %me.n, i64 1)
  %me2v = load i64, ptr %me.r, align 8
  %me2ok = icmp eq i64 %me2v, 445
  call void @ut_check(i1 %me2ok, ptr @m.me2)

  ; ---- modexp vs naive powmod, s=2, random odd n, small exp, 40 iters
  %mx.base = alloca [2 x i64], align 8
  %mx.n = alloca [2 x i64], align 8
  %mx.r = alloca [2 x i64], align 8
  %mx.acc = alloca [2 x i64], align 8
  %mx.bred = alloca [2 x i64], align 8
  %mx.tmp = alloca [4 x i64], align 8
  %mx.exp = alloca [1 x i64], align 8
  br label %mx.loop
mx.loop:
  %mx.i = phi i64 [ 0, %dm.done ], [ %mx.in, %mx.cont ]
  %mx.v = phi i64 [ 0, %dm.done ], [ %mx.vn, %mx.cont ]
  call void @fillrand(ptr %mx.n, i64 2, ptr %seed)
  ; force n odd and 2-limb (top limb nonzero)
  %mx.n0 = load i64, ptr %mx.n, align 8
  %mx.n0o = or i64 %mx.n0, 1
  store i64 %mx.n0o, ptr %mx.n, align 8
  %mx.n1p = getelementptr inbounds [2 x i64], ptr %mx.n, i64 0, i64 1
  %mx.n1 = load i64, ptr %mx.n1p, align 8
  %mx.n1o = or i64 %mx.n1, 1
  store i64 %mx.n1o, ptr %mx.n1p, align 8
  ; base reduced = base mod n
  call void @fillrand(ptr %mx.base, i64 2, ptr %seed)
  %mx.br = call i32 @universe_bignum_mod(ptr %mx.bred, ptr %mx.base, i64 2, ptr %mx.n, i64 2)
  ; exponent small: e in [0,15]
  %mx.er = call i64 @ut_rand(ptr %seed)
  %mx.e = and i64 %mx.er, 15
  store i64 %mx.e, ptr %mx.exp, align 8
  ; naive: acc = 1; repeat e times acc = acc*bred mod n
  call void @universe_bignum_set_u64(ptr %mx.acc, i64 2, i64 1)
  br label %mx.nloop
mx.nloop:
  %mx.j = phi i64 [ 0, %mx.loop ], [ %mx.jn, %mx.nbody ]
  %mx.jdone = icmp uge i64 %mx.j, %mx.e
  br i1 %mx.jdone, label %mx.checkexp, label %mx.nbody
mx.nbody:
  call void @mulmod(ptr %mx.acc, ptr %mx.acc, ptr %mx.bred, ptr %mx.n, i64 2, i64 2, ptr %mx.tmp)
  %mx.jn = add i64 %mx.j, 1
  br label %mx.nloop
mx.checkexp:
  %mx.rc = call i32 @universe_bignum_modexp(ptr %mx.r, ptr %mx.base, i64 2, ptr %mx.exp, i64 1, ptr %mx.n, i64 2)
  %mx.cmp = call i32 @universe_bignum_cmp_n(ptr %mx.r, ptr %mx.acc, i64 2)
  %mx.ok = icmp eq i32 %mx.cmp, 0
  %mx.bad = xor i1 %mx.ok, true
  %mx.a = zext i1 %mx.bad to i64
  %mx.vn = add i64 %mx.v, %mx.a
  br label %mx.cont
mx.cont:
  %mx.in = add i64 %mx.i, 1
  %mx.more = icmp ult i64 %mx.in, 40
  br i1 %mx.more, label %mx.loop, label %mx.done
mx.done:
  call void @ut_check_eq(i64 %mx.v, i64 0, ptr @m.mecx)

  ; ---- montmul product identity, s=2, 40 iters
  ; fromMont(montmul(toMont(a),toMont(b))) == (a*b) mod n
  %mm.a = alloca [2 x i64], align 8
  %mm.b = alloca [2 x i64], align 8
  %mm.n = alloca [2 x i64], align 8
  %mm.rr = alloca [2 x i64], align 8
  %mm.one = alloca [2 x i64], align 8
  %mm.am = alloca [2 x i64], align 8
  %mm.bm = alloca [2 x i64], align 8
  %mm.pm = alloca [2 x i64], align 8
  %mm.pr = alloca [2 x i64], align 8
  %mm.exp = alloca [2 x i64], align 8
  %mm.tmp = alloca [4 x i64], align 8
  %mm.sc = alloca [4 x i64], align 8
  %mm.ta = alloca [2 x i64], align 8
  %mm.tb = alloca [2 x i64], align 8
  br label %mm.loop
mm.loop:
  %mm.i = phi i64 [ 0, %mx.done ], [ %mm.in, %mm.cont ]
  %mm.v = phi i64 [ 0, %mx.done ], [ %mm.vn, %mm.cont ]
  call void @fillrand(ptr %mm.n, i64 2, ptr %seed)
  %mm.n0 = load i64, ptr %mm.n, align 8
  %mm.n0o = or i64 %mm.n0, 1
  store i64 %mm.n0o, ptr %mm.n, align 8
  %mm.n1p = getelementptr inbounds [2 x i64], ptr %mm.n, i64 0, i64 1
  %mm.n1 = load i64, ptr %mm.n1p, align 8
  %mm.n1o = or i64 %mm.n1, 1
  store i64 %mm.n1o, ptr %mm.n1p, align 8
  ; a,b reduced mod n
  call void @fillrand(ptr %mm.ta, i64 2, ptr %seed)
  call void @fillrand(ptr %mm.tb, i64 2, ptr %seed)
  %mm.rca = call i32 @universe_bignum_mod(ptr %mm.a, ptr %mm.ta, i64 2, ptr %mm.n, i64 2)
  %mm.rcb = call i32 @universe_bignum_mod(ptr %mm.b, ptr %mm.tb, i64 2, ptr %mm.n, i64 2)
  %mm.n0inv = call i64 @universe_bignum_mont_n0inv(ptr %mm.n)
  call void @universe_bignum_mont_rr(ptr %mm.rr, ptr %mm.n, i64 2)
  call void @universe_bignum_set_u64(ptr %mm.one, i64 2, i64 1)
  ; toMont
  call void @universe_bignum_montmul(ptr %mm.am, ptr %mm.a, ptr %mm.rr, ptr %mm.n, i64 2, i64 %mm.n0inv, ptr %mm.sc)
  call void @universe_bignum_montmul(ptr %mm.bm, ptr %mm.b, ptr %mm.rr, ptr %mm.n, i64 2, i64 %mm.n0inv, ptr %mm.sc)
  ; product in mont
  call void @universe_bignum_montmul(ptr %mm.pm, ptr %mm.am, ptr %mm.bm, ptr %mm.n, i64 2, i64 %mm.n0inv, ptr %mm.sc)
  ; fromMont
  call void @universe_bignum_montmul(ptr %mm.pr, ptr %mm.pm, ptr %mm.one, ptr %mm.n, i64 2, i64 %mm.n0inv, ptr %mm.sc)
  ; reference (a*b) mod n
  call void @mulmod(ptr %mm.exp, ptr %mm.a, ptr %mm.b, ptr %mm.n, i64 2, i64 2, ptr %mm.tmp)
  %mm.cmp = call i32 @universe_bignum_cmp_n(ptr %mm.pr, ptr %mm.exp, i64 2)
  %mm.ok = icmp eq i32 %mm.cmp, 0
  %mm.bad = xor i1 %mm.ok, true
  %mm.a2 = zext i1 %mm.bad to i64
  %mm.vn = add i64 %mm.v, %mm.a2
  br label %mm.cont
mm.cont:
  %mm.in = add i64 %mm.i, 1
  %mm.more = icmp ult i64 %mm.in, 40
  br i1 %mm.more, label %mm.loop, label %mm.done
mm.done:
  call void @ut_check_eq(i64 %mm.v, i64 0, ptr @m.mont)

  ; ---- modinv: n = 2^61-1 (Mersenne prime M61), s=1, random a. a*inv==1
  %iv.n = alloca [1 x i64], align 8
  %iv.a = alloca [1 x i64], align 8
  %iv.inv = alloca [1 x i64], align 8
  %iv.chk = alloca [1 x i64], align 8
  %iv.tmp = alloca [2 x i64], align 8
  store i64 2305843009213693951, ptr %iv.n, align 8
  br label %iv.loop
iv.loop:
  %iv.i = phi i64 [ 0, %mm.done ], [ %iv.in, %iv.cont ]
  %iv.v = phi i64 [ 0, %mm.done ], [ %iv.vn, %iv.cont ]
  %iv.rr = call i64 @ut_rand(ptr %seed)
  %iv.ar = urem i64 %iv.rr, 2305843009213693951
  %iv.az = icmp eq i64 %iv.ar, 0
  %iv.a1 = select i1 %iv.az, i64 1, i64 %iv.ar
  store i64 %iv.a1, ptr %iv.a, align 8
  %iv.rc = call i32 @universe_bignum_modinv(ptr %iv.inv, ptr %iv.a, ptr %iv.n, i64 1)
  ; a * inv mod n == 1
  call void @mulmod(ptr %iv.chk, ptr %iv.a, ptr %iv.inv, ptr %iv.n, i64 1, i64 1, ptr %iv.tmp)
  %iv.cv = load i64, ptr %iv.chk, align 8
  %iv.ok0 = icmp eq i64 %iv.cv, 1
  %iv.rcok = icmp eq i32 %iv.rc, 0
  %iv.ok = and i1 %iv.ok0, %iv.rcok
  %iv.bad = xor i1 %iv.ok, true
  %iv.add = zext i1 %iv.bad to i64
  %iv.vn = add i64 %iv.v, %iv.add
  br label %iv.cont
iv.cont:
  %iv.in = add i64 %iv.i, 1
  %iv.more = icmp ult i64 %iv.in, 60
  br i1 %iv.more, label %iv.loop, label %iv.done
iv.done:
  call void @ut_check_eq(i64 %iv.v, i64 0, ptr @m.inv)

  ; ---- gcd known vectors (s=1)
  %g.a = alloca [1 x i64], align 8
  %g.b = alloca [1 x i64], align 8
  %g.g = alloca [1 x i64], align 8
  store i64 48, ptr %g.a, align 8
  store i64 36, ptr %g.b, align 8
  %grc1 = call i32 @universe_bignum_gcd(ptr %g.g, ptr %g.a, ptr %g.b, i64 1)
  %gv1 = load i64, ptr %g.g, align 8
  %gok1 = icmp eq i64 %gv1, 12
  call void @ut_check(i1 %gok1, ptr @m.gcd1)
  store i64 17, ptr %g.a, align 8
  store i64 5, ptr %g.b, align 8
  %grc2 = call i32 @universe_bignum_gcd(ptr %g.g, ptr %g.a, ptr %g.b, i64 1)
  %gv2 = load i64, ptr %g.g, align 8
  %gok2 = icmp eq i64 %gv2, 1
  call void @ut_check(i1 %gok2, ptr @m.gcd2)

  %rc = call i32 @ut_summary()
  ret i32 %rc

dobench:
  ; modexp with a 256-bit (s=4) odd modulus and full-width exponent
  %bn.n = alloca [4 x i64], align 8
  %bn.base = alloca [4 x i64], align 8
  %bn.exp = alloca [4 x i64], align 8
  %bn.r = alloca [4 x i64], align 8
  call void @fillrand(ptr %bn.n, i64 4, ptr %seed)
  %bn.n0 = load i64, ptr %bn.n, align 8
  %bn.n0o = or i64 %bn.n0, 1
  store i64 %bn.n0o, ptr %bn.n, align 8
  %bn.n3p = getelementptr inbounds [4 x i64], ptr %bn.n, i64 0, i64 3
  %bn.n3 = load i64, ptr %bn.n3p, align 8
  %bn.n3o = or i64 %bn.n3, -9223372036854775808   ; set top bit -> full 256-bit
  store i64 %bn.n3o, ptr %bn.n3p, align 8
  call void @fillrand(ptr %bn.base, i64 4, ptr %seed)
  call void @fillrand(ptr %bn.exp, i64 4, ptr %seed)
  ; snapshot the initial exponent: the loop mutates exp (result feedback to block
  ; DCE), so each rep must restore the identical input before timing.
  %bn.exp0 = alloca [4 x i64], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %bn.exp0, ptr %bn.exp, i64 32, i1 false)
  ; 17 reps of a 2000-modexp batch; discard rep 0 (warm-up), report over the
  ; remaining 16. ops/rep = 2000 (ns per 256-bit modexp).
  br label %bn.rep
bn.rep:
  %bn.rep.i = phi i64 [ 0, %dobench ], [ %bn.rep.n, %bn.next ]
  call void @llvm.memcpy.p0.p0.i64(ptr %bn.exp, ptr %bn.exp0, i64 32, i1 false)
  %t0 = call double @ut_now_sec()
  br label %bn.loop
bn.loop:
  %bn.i = phi i64 [ 0, %bn.rep ], [ %bn.in, %bn.loop ]
  %bn.rc = call i32 @universe_bignum_modexp(ptr %bn.r, ptr %bn.base, i64 4, ptr %bn.exp, i64 4, ptr %bn.n, i64 4)
  ; feed result low limb back into exp to prevent DCE
  %bn.rl = load i64, ptr %bn.r, align 8
  store volatile i64 %bn.rl, ptr %bn.exp, align 8
  %bn.in = add i64 %bn.i, 1
  %bn.more = icmp ult i64 %bn.in, 2000
  br i1 %bn.more, label %bn.loop, label %bn.rep.done
bn.rep.done:
  %t1 = call double @ut_now_sec()
  %bel = fsub double %t1, %t0
  %bkeep = icmp ugt i64 %bn.rep.i, 0
  br i1 %bkeep, label %bn.store, label %bn.next
bn.store:
  %bidx = sub i64 %bn.rep.i, 1
  %bsp = getelementptr inbounds [16 x double], ptr @bn.samp, i64 0, i64 %bidx
  store double %bel, ptr %bsp, align 8
  br label %bn.next
bn.next:
  %bn.rep.n = add nuw i64 %bn.rep.i, 1
  %bmore = icmp ult i64 %bn.rep.n, 17
  br i1 %bmore, label %bn.rep, label %bn.report
bn.report:
  call void @ut_report_dist(ptr @bn.samp, i64 16, i64 2000, ptr @lbl.bn)
  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
