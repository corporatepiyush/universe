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

; Ed25519 signature scheme (RFC 8032, edwards25519, field 2^255-19).
;
; DESIGN:
;   FIELD Fp (p = 2^255-19): elements are 4 little-endian u64 limbs kept
;   CANONICAL in [0,p) after every op. We reuse the audited bignum kernels:
;   f_mul = bignum_mul (4x4 -> 8 limbs) then bignum_mod by p; f_add/f_sub use
;   bignum_add_n/sub_n + a single conditional reduce; f_inv = bignum_modinv;
;   the decode square-root exponentiation ^((p-5)/8) = ^(2^252-3) = bignum
;   modexp. This is CORRECTNESS-FIRST: mod-per-multiply (Knuth) is slower than a
;   specialized 2^255-19 reduction but is provably correct from a tested base
;   and is the KAT gate. A dedicated 38-fold reduction is a later speed pass.
;
;   CURVE (twisted Edwards -x^2+y^2 = 1+d x^2 y^2, a=-1): extended homogeneous
;   coordinates (X,Y,Z,T), point = 4 field elements = 128 bytes
;   (X@0,Y@32,Z@64,T@96). ONE unified, COMPLETE addition law (add-2008-hwcd-3,
;   k=2d) serves both add and double (double = add(P,P)); no separate/incomplete
;   doubling to get wrong on the identity. Scalar multiply is MSB-first
;   double-and-add over 256 bits (NOT constant-time).
;
;   SIGN (RFC 8032): h = SHA-512(seed); a = clamp(h[0..31]); prefix = h[32..63];
;   A = [a]B; r = SHA-512(prefix||M) mod L; R = [r]B; k = SHA-512(R||A||M) mod L;
;   S = (r + k*a) mod L; sig = enc(R) || S. Reductions mod the group order
;   L = 2^252+27742317777372353535851937790883648493 use bignum_mod.
;   VERIFY: decode A and R; require S < L; k = SHA-512(R||A||M) mod L; accept iff
;   [S]B == R + [k]A (compared via canonical 32-byte encodings).
;
; HARDENING-TODO: fast, NOT constant-time. Scalar ladder, field mod, and all
;   comparisons branch on secret data; no scratch zeroization; no small-order /
;   cofactor checks beyond decode validity. Deferred to the hardening phase.
;
; API (C ABI, nounwind):
;   void universe_crypto_ed25519_public_from_seed(ptr seed32, ptr out_pub32)
;   void universe_crypto_ed25519_sign(ptr seed32, ptr msg, i64 msglen, ptr sig64)
;   i32  universe_crypto_ed25519_verify(ptr pub32, ptr msg, i64 msglen, ptr sig64)
;        ; -> 0 valid, 1 invalid

declare void @universe_bignum_mul(ptr, ptr, i64, ptr, i64)
declare i32 @universe_bignum_mod(ptr, ptr, i64, ptr, i64)
declare i64 @universe_bignum_add_n(ptr, ptr, ptr, i64)
declare i64 @universe_bignum_sub_n(ptr, ptr, ptr, i64)
declare i32 @universe_bignum_cmp_n(ptr, ptr, i64)
declare i32 @universe_bignum_modinv(ptr, ptr, ptr, i64)
declare i32 @universe_bignum_modexp(ptr, ptr, i64, ptr, i64, ptr, i64)

declare void @universe_crypto_sha512_init(ptr)
declare void @universe_crypto_sha512_update(ptr, ptr, i64)
declare void @universe_crypto_sha512_final(ptr, ptr)
declare void @universe_crypto_sha512_hash(ptr, i64, ptr)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; --------------------------------------------------------------- constants
; All 32-byte little-endian, align 8 so they load as 4 x i64.
@ED_P      = private unnamed_addr constant [32 x i8] c"\ed\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\7f", align 8
@ED_D      = private unnamed_addr constant [32 x i8] c"\a3\78\59\13\ca\4d\eb\75\ab\d8\41\41\4d\0a\70\00\98\e8\79\77\79\40\c7\8c\73\fe\6f\2b\ee\6c\03\52", align 8
@ED_D2     = private unnamed_addr constant [32 x i8] c"\59\f1\b2\26\94\9b\d6\eb\56\b1\83\82\9a\14\e0\00\30\d1\f3\ee\f2\80\8e\19\e7\fc\df\56\dc\d9\06\24", align 8
@ED_SQRTM1 = private unnamed_addr constant [32 x i8] c"\b0\a0\0e\4a\27\1b\ee\c4\78\e4\2f\ad\06\18\43\2f\a7\d7\fb\3d\99\00\4d\2b\0b\df\c1\4f\80\24\83\2b", align 8
@ED_L      = private unnamed_addr constant [32 x i8] c"\ed\d3\f5\5c\1a\63\12\58\d6\9c\f7\a2\de\f9\de\14\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10", align 8
@ED_BX     = private unnamed_addr constant [32 x i8] c"\1a\d5\25\8f\60\2d\56\c9\b2\a7\25\95\60\c7\2c\69\5c\dc\d6\fd\31\e2\a4\c0\fe\53\6e\cd\d3\36\69\21", align 8
@ED_BY     = private unnamed_addr constant [32 x i8] c"\58\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66\66", align 8
@ED_BT     = private unnamed_addr constant [32 x i8] c"\a3\dd\b7\a5\b3\8a\de\6d\f5\52\51\77\80\9f\f0\20\7d\e3\ab\64\8e\4e\ea\66\65\76\8b\d7\0f\5f\87\67", align 8
@ED_EXP58  = private unnamed_addr constant [32 x i8] c"\fd\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\0f", align 8
@ED_ZERO   = private unnamed_addr constant [32 x i8] zeroinitializer, align 8

; --------------------------------------------------------------- field ops

define internal void @f_mul(ptr %r, ptr %a, ptr %b) #0 {
entry:
  %t = alloca [8 x i64], align 8
  call void @universe_bignum_mul(ptr %t, ptr %a, i64 4, ptr %b, i64 4)
  %rc = call i32 @universe_bignum_mod(ptr %r, ptr %t, i64 8, ptr @ED_P, i64 4)
  ret void
}

define internal void @f_sq(ptr %r, ptr %a) #0 {
entry:
  call void @f_mul(ptr %r, ptr %a, ptr %a)
  ret void
}

define internal void @f_add(ptr %r, ptr %a, ptr %b) #0 {
entry:
  %c = call i64 @universe_bignum_add_n(ptr %r, ptr %a, ptr %b, i64 4)
  %cmp = call i32 @universe_bignum_cmp_n(ptr %r, ptr @ED_P, i64 4)
  %ge = icmp sge i32 %cmp, 0
  %cnz = icmp ne i64 %c, 0
  %need = or i1 %ge, %cnz
  br i1 %need, label %sub, label %done
sub:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr %r, ptr @ED_P, i64 4)
  br label %done
done:
  ret void
}

define internal void @f_sub(ptr %r, ptr %a, ptr %b) #0 {
entry:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr %a, ptr %b, i64 4)
  %neg = icmp ne i64 %bw, 0
  br i1 %neg, label %add, label %done
add:
  %c = call i64 @universe_bignum_add_n(ptr %r, ptr %r, ptr @ED_P, i64 4)
  br label %done
done:
  ret void
}

define internal void @f_neg(ptr %r, ptr %a) #0 {
entry:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr @ED_ZERO, ptr %a, i64 4)
  %neg = icmp ne i64 %bw, 0
  br i1 %neg, label %add, label %done
add:
  %c = call i64 @universe_bignum_add_n(ptr %r, ptr %r, ptr @ED_P, i64 4)
  br label %done
done:
  ret void
}

define internal void @f_inv(ptr %r, ptr %a) #0 {
entry:
  %rc = call i32 @universe_bignum_modinv(ptr %r, ptr %a, ptr @ED_P, i64 4)
  ret void
}

; r = a ^ ((p-5)/8) mod p
define internal void @f_pow58(ptr %r, ptr %a) #0 {
entry:
  %rc = call i32 @universe_bignum_modexp(ptr %r, ptr %a, i64 4, ptr @ED_EXP58, i64 4, ptr @ED_P, i64 4)
  ret void
}

define internal i1 @f_eq(ptr %a, ptr %b) #0 {
entry:
  %c = call i32 @universe_bignum_cmp_n(ptr %a, ptr %b, i64 4)
  %eq = icmp eq i32 %c, 0
  ret i1 %eq
}

; --------------------------------------------------------------- point ops

define internal void @pt_identity(ptr %r) #0 {
entry:
  call void @llvm.memset.p0.i64(ptr %r, i8 0, i64 128, i1 false)
  %yp = getelementptr inbounds i8, ptr %r, i64 32
  store i64 1, ptr %yp, align 8
  %zp = getelementptr inbounds i8, ptr %r, i64 64
  store i64 1, ptr %zp, align 8
  ret void
}

; r = p1 + p2 (unified complete twisted-Edwards addition, a=-1, k=2d)
define internal void @pt_add(ptr %r, ptr %p1, ptr %p2) #0 {
entry:
  %A = alloca [4 x i64], align 8
  %B = alloca [4 x i64], align 8
  %C = alloca [4 x i64], align 8
  %D = alloca [4 x i64], align 8
  %E = alloca [4 x i64], align 8
  %F = alloca [4 x i64], align 8
  %G = alloca [4 x i64], align 8
  %H = alloca [4 x i64], align 8
  %t1 = alloca [4 x i64], align 8
  %t2 = alloca [4 x i64], align 8
  %X3 = alloca [4 x i64], align 8
  %Y3 = alloca [4 x i64], align 8
  %Z3 = alloca [4 x i64], align 8
  %T3 = alloca [4 x i64], align 8
  %X1 = getelementptr inbounds i8, ptr %p1, i64 0
  %Y1 = getelementptr inbounds i8, ptr %p1, i64 32
  %Z1 = getelementptr inbounds i8, ptr %p1, i64 64
  %T1 = getelementptr inbounds i8, ptr %p1, i64 96
  %X2 = getelementptr inbounds i8, ptr %p2, i64 0
  %Y2 = getelementptr inbounds i8, ptr %p2, i64 32
  %Z2 = getelementptr inbounds i8, ptr %p2, i64 64
  %T2 = getelementptr inbounds i8, ptr %p2, i64 96
  call void @f_sub(ptr %t1, ptr %Y1, ptr %X1)
  call void @f_sub(ptr %t2, ptr %Y2, ptr %X2)
  call void @f_mul(ptr %A, ptr %t1, ptr %t2)
  call void @f_add(ptr %t1, ptr %Y1, ptr %X1)
  call void @f_add(ptr %t2, ptr %Y2, ptr %X2)
  call void @f_mul(ptr %B, ptr %t1, ptr %t2)
  call void @f_mul(ptr %t1, ptr %T1, ptr @ED_D2)
  call void @f_mul(ptr %C, ptr %t1, ptr %T2)
  call void @f_mul(ptr %t1, ptr %Z1, ptr %Z2)
  call void @f_add(ptr %D, ptr %t1, ptr %t1)
  call void @f_sub(ptr %E, ptr %B, ptr %A)
  call void @f_sub(ptr %F, ptr %D, ptr %C)
  call void @f_add(ptr %G, ptr %D, ptr %C)
  call void @f_add(ptr %H, ptr %B, ptr %A)
  call void @f_mul(ptr %X3, ptr %E, ptr %F)
  call void @f_mul(ptr %Y3, ptr %G, ptr %H)
  call void @f_mul(ptr %T3, ptr %E, ptr %H)
  call void @f_mul(ptr %Z3, ptr %F, ptr %G)
  %rX = getelementptr inbounds i8, ptr %r, i64 0
  %rY = getelementptr inbounds i8, ptr %r, i64 32
  %rZ = getelementptr inbounds i8, ptr %r, i64 64
  %rT = getelementptr inbounds i8, ptr %r, i64 96
  call void @llvm.memcpy.p0.p0.i64(ptr %rX, ptr %X3, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %rY, ptr %Y3, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %rZ, ptr %Z3, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %rT, ptr %T3, i64 32, i1 false)
  ret void
}

; r = [scalar] * base ; scalar = 4 LE limbs, MSB-first double-and-add
define internal void @pt_scalarmult(ptr %r, ptr %scalar, ptr %base) #0 {
entry:
  %R = alloca [16 x i64], align 8
  call void @pt_identity(ptr %R)
  br label %loop
loop:
  %k = phi i64 [ 255, %entry ], [ %kn, %step ]
  call void @pt_add(ptr %R, ptr %R, ptr %R)
  %li = lshr i64 %k, 6
  %off = and i64 %k, 63
  %lp = getelementptr inbounds i64, ptr %scalar, i64 %li
  %lv = load i64, ptr %lp, align 8
  %sh = lshr i64 %lv, %off
  %bit = and i64 %sh, 1
  %set = icmp eq i64 %bit, 1
  br i1 %set, label %addb, label %step
addb:
  call void @pt_add(ptr %R, ptr %R, ptr %base)
  br label %step
step:
  %kn = add i64 %k, -1
  %more = icmp sge i64 %kn, 0
  br i1 %more, label %loop, label %done
done:
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %R, i64 128, i1 false)
  ret void
}

define internal void @pt_load_base(ptr %r) #0 {
entry:
  %rX = getelementptr inbounds i8, ptr %r, i64 0
  %rY = getelementptr inbounds i8, ptr %r, i64 32
  %rZ = getelementptr inbounds i8, ptr %r, i64 64
  %rT = getelementptr inbounds i8, ptr %r, i64 96
  call void @llvm.memcpy.p0.p0.i64(ptr %rX, ptr @ED_BX, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %rY, ptr @ED_BY, i64 32, i1 false)
  call void @llvm.memset.p0.i64(ptr %rZ, i8 0, i64 32, i1 false)
  store i64 1, ptr %rZ, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %rT, ptr @ED_BT, i64 32, i1 false)
  ret void
}

; encode point -> 32 bytes (y little-endian, high bit = sign of x)
define internal void @pt_encode(ptr %out, ptr %p) #0 {
entry:
  %zinv = alloca [4 x i64], align 8
  %x = alloca [4 x i64], align 8
  %y = alloca [4 x i64], align 8
  %X = getelementptr inbounds i8, ptr %p, i64 0
  %Y = getelementptr inbounds i8, ptr %p, i64 32
  %Z = getelementptr inbounds i8, ptr %p, i64 64
  call void @f_inv(ptr %zinv, ptr %Z)
  call void @f_mul(ptr %x, ptr %X, ptr %zinv)
  call void @f_mul(ptr %y, ptr %Y, ptr %zinv)
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %y, i64 32, i1 false)
  %x0 = load i64, ptr %x, align 8
  %sign = and i64 %x0, 1
  %signb = trunc i64 %sign to i8
  %hi = shl i8 %signb, 7
  %bp = getelementptr inbounds i8, ptr %out, i64 31
  %bv = load i8, ptr %bp, align 1
  %nb = or i8 %bv, %hi
  store i8 %nb, ptr %bp, align 1
  ret void
}

; decode 32 bytes -> point ; returns 1 on success, 0 on failure
define internal i1 @pt_decode(ptr %p, ptr %in) #0 {
entry:
  %y = alloca [4 x i64], align 8
  %u = alloca [4 x i64], align 8
  %v = alloca [4 x i64], align 8
  %v2 = alloca [4 x i64], align 8
  %v3 = alloca [4 x i64], align 8
  %v4 = alloca [4 x i64], align 8
  %v7 = alloca [4 x i64], align 8
  %uv7 = alloca [4 x i64], align 8
  %pw = alloca [4 x i64], align 8
  %x = alloca [4 x i64], align 8
  %x2 = alloca [4 x i64], align 8
  %vx2 = alloca [4 x i64], align 8
  %negu = alloca [4 x i64], align 8
  %tmp = alloca [4 x i64], align 8
  %one = alloca [4 x i64], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %y, ptr %in, i64 32, i1 false)
  ; sign bit and clear bit 255
  %b31p = getelementptr inbounds i8, ptr %in, i64 31
  %b31 = load i8, ptr %b31p, align 1
  %signi = lshr i8 %b31, 7
  %sign = zext i8 %signi to i64
  %y3p = getelementptr inbounds i64, ptr %y, i64 3
  %y3 = load i64, ptr %y3p, align 8
  %y3m = and i64 %y3, 9223372036854775807
  store i64 %y3m, ptr %y3p, align 8
  ; require y < p
  %cy = call i32 @universe_bignum_cmp_n(ptr %y, ptr @ED_P, i64 4)
  %yok = icmp slt i32 %cy, 0
  br i1 %yok, label %compute, label %fail
compute:
  ; one = 1
  call void @llvm.memset.p0.i64(ptr %one, i8 0, i64 32, i1 false)
  store i64 1, ptr %one, align 8
  ; u = y^2 - 1
  call void @f_sq(ptr %u, ptr %y)
  call void @f_sub(ptr %u, ptr %u, ptr %one)
  ; v = d*y^2 + 1
  call void @f_sq(ptr %tmp, ptr %y)
  call void @f_mul(ptr %v, ptr %tmp, ptr @ED_D)
  call void @f_add(ptr %v, ptr %v, ptr %one)
  ; v2=v^2 v4=v2^2 v3=v2*v v7=v4*v3
  call void @f_sq(ptr %v2, ptr %v)
  call void @f_sq(ptr %v4, ptr %v2)
  call void @f_mul(ptr %v3, ptr %v2, ptr %v)
  call void @f_mul(ptr %v7, ptr %v4, ptr %v3)
  ; uv7 = u*v7 ; pw = uv7^((p-5)/8) ; x = u*v3*pw
  call void @f_mul(ptr %uv7, ptr %u, ptr %v7)
  call void @f_pow58(ptr %pw, ptr %uv7)
  call void @f_mul(ptr %x, ptr %u, ptr %v3)
  call void @f_mul(ptr %x, ptr %x, ptr %pw)
  ; vx2 = v*x^2
  call void @f_sq(ptr %x2, ptr %x)
  call void @f_mul(ptr %vx2, ptr %v, ptr %x2)
  ; if vx2 == u ok ; elif vx2 == -u x*=sqrtm1 ; else fail
  %e1 = call i1 @f_eq(ptr %vx2, ptr %u)
  br i1 %e1, label %haveroot, label %tryneg
tryneg:
  call void @f_neg(ptr %negu, ptr %u)
  %e2 = call i1 @f_eq(ptr %vx2, ptr %negu)
  br i1 %e2, label %fixroot, label %fail
fixroot:
  call void @f_mul(ptr %x, ptr %x, ptr @ED_SQRTM1)
  br label %haveroot
haveroot:
  ; if x == 0 and sign == 1: fail
  %x0 = load i64, ptr %x, align 8
  %xlsb = and i64 %x0, 1
  %xis0 = call i32 @universe_bignum_cmp_n(ptr %x, ptr @ED_ZERO, i64 4)
  %xzero = icmp eq i32 %xis0, 0
  %s1 = icmp eq i64 %sign, 1
  %bad = and i1 %xzero, %s1
  br i1 %bad, label %fail, label %adjust
adjust:
  ; if (x & 1) != sign: x = -x
  %needneg = icmp ne i64 %xlsb, %sign
  br i1 %needneg, label %doneg, label %store
doneg:
  call void @f_neg(ptr %x, ptr %x)
  br label %store
store:
  %pX = getelementptr inbounds i8, ptr %p, i64 0
  %pY = getelementptr inbounds i8, ptr %p, i64 32
  %pZ = getelementptr inbounds i8, ptr %p, i64 64
  %pT = getelementptr inbounds i8, ptr %p, i64 96
  call void @llvm.memcpy.p0.p0.i64(ptr %pX, ptr %x, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %pY, ptr %y, i64 32, i1 false)
  call void @llvm.memset.p0.i64(ptr %pZ, i8 0, i64 32, i1 false)
  store i64 1, ptr %pZ, align 8
  call void @f_mul(ptr %pT, ptr %x, ptr %y)
  ret i1 true
fail:
  ret i1 false
}

define internal i1 @pt_eq(ptr %p1, ptr %p2) #0 {
entry:
  %e1 = alloca [32 x i8], align 8
  %e2 = alloca [32 x i8], align 8
  call void @pt_encode(ptr %e1, ptr %p1)
  call void @pt_encode(ptr %e2, ptr %p2)
  %c = call i32 @memcmp_ed(ptr %e1, ptr %e2, i64 32)
  %eq = icmp eq i32 %c, 0
  ret i1 %eq
}

; tiny internal memcmp (avoid external libc dependency in the module)
define internal i32 @memcmp_ed(ptr %a, ptr %b, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %eq, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %ap = getelementptr inbounds i8, ptr %a, i64 %i
  %bp = getelementptr inbounds i8, ptr %b, i64 %i
  %av = load i8, ptr %ap, align 1
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %diff, label %cont
cont:
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %eq
diff:
  ret i32 1
eq:
  ret i32 0
}

; --------------------------------------------------------------- scalar mod L

; out(4 limbs) = value(in, inlimbs) mod L
define internal void @sc_reduce(ptr %out, ptr %in, i64 %inlimbs) #0 {
entry:
  %rc = call i32 @universe_bignum_mod(ptr %out, ptr %in, i64 %inlimbs, ptr @ED_L, i64 4)
  ret void
}

; clamp: dst = h[0..31] with Ed25519 clamping (dst must be 32 bytes)
define internal void @ed_clamp(ptr %dst, ptr %h) #0 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %h, i64 32, i1 false)
  %b0 = load i8, ptr %dst, align 1
  %b0c = and i8 %b0, -8
  store i8 %b0c, ptr %dst, align 1
  %b31p = getelementptr inbounds i8, ptr %dst, i64 31
  %b31 = load i8, ptr %b31p, align 1
  %b31a = and i8 %b31, 127
  %b31b = or i8 %b31a, 64
  store i8 %b31b, ptr %b31p, align 1
  ret void
}

; --------------------------------------------------------------- public API

define void @universe_crypto_ed25519_public_from_seed(ptr %seed, ptr %out) local_unnamed_addr #1 {
entry:
  %h = alloca [64 x i8], align 8
  %a = alloca [32 x i8], align 8
  %B = alloca [16 x i64], align 8
  %A = alloca [16 x i64], align 8
  call void @universe_crypto_sha512_hash(ptr %seed, i64 32, ptr %h)
  call void @ed_clamp(ptr %a, ptr %h)
  call void @pt_load_base(ptr %B)
  call void @pt_scalarmult(ptr %A, ptr %a, ptr %B)
  call void @pt_encode(ptr %out, ptr %A)
  ret void
}

define void @universe_crypto_ed25519_sign(ptr %seed, ptr %msg, i64 %msglen, ptr %sig) local_unnamed_addr #1 {
entry:
  %h = alloca [64 x i8], align 8
  %a = alloca [32 x i8], align 8
  %B = alloca [16 x i64], align 8
  %A = alloca [16 x i64], align 8
  %Rpt = alloca [16 x i64], align 8
  %Aenc = alloca [32 x i8], align 8
  %ctx = alloca [200 x i8], align 8
  %hr = alloca [64 x i8], align 8
  %rsc = alloca [8 x i64], align 8
  %ksc = alloca [4 x i64], align 8
  %prod = alloca [8 x i64], align 8
  %Ssc = alloca [4 x i64], align 8
  call void @universe_crypto_sha512_hash(ptr %seed, i64 32, ptr %h)
  call void @ed_clamp(ptr %a, ptr %h)
  %prefix = getelementptr inbounds i8, ptr %h, i64 32
  ; A = [a]B
  call void @pt_load_base(ptr %B)
  call void @pt_scalarmult(ptr %A, ptr %a, ptr %B)
  call void @pt_encode(ptr %Aenc, ptr %A)
  ; rsc is 8 limbs, high 4 zeroed so the r+k*a accumulation reads in-bounds
  call void @llvm.memset.p0.i64(ptr %rsc, i8 0, i64 64, i1 false)
  ; r = SHA512(prefix || msg) mod L
  call void @universe_crypto_sha512_init(ptr %ctx)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %prefix, i64 32)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %msg, i64 %msglen)
  call void @universe_crypto_sha512_final(ptr %ctx, ptr %hr)
  call void @sc_reduce(ptr %rsc, ptr %hr, i64 8)
  ; R = [r]B ; enc(R) -> sig[0..31]
  call void @pt_scalarmult(ptr %Rpt, ptr %rsc, ptr %B)
  call void @pt_encode(ptr %sig, ptr %Rpt)
  ; k = SHA512(enc(R) || A || msg) mod L
  call void @universe_crypto_sha512_init(ptr %ctx)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %sig, i64 32)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %Aenc, i64 32)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %msg, i64 %msglen)
  call void @universe_crypto_sha512_final(ptr %ctx, ptr %hr)
  call void @sc_reduce(ptr %ksc, ptr %hr, i64 8)
  ; S = (r + k*a) mod L
  call void @universe_bignum_mul(ptr %prod, ptr %ksc, i64 4, ptr %a, i64 4)
  ; add r (4 limbs) into prod (8 limbs)
  br label %addloop
addloop:
  %i = phi i64 [ 0, %entry ], [ %in, %addloop ]
  %carry = phi i128 [ 0, %entry ], [ %chi, %addloop ]
  %pp = getelementptr inbounds i64, ptr %prod, i64 %i
  %pv = load i64, ptr %pp, align 8
  %pv128 = zext i64 %pv to i128
  %rp = getelementptr inbounds i64, ptr %rsc, i64 %i
  %rv = load i64, ptr %rp, align 8
  %rv128 = zext i64 %rv to i128
  %s1 = add i128 %pv128, %rv128
  %s2 = add i128 %s1, %carry
  %lo = trunc i128 %s2 to i64
  store i64 %lo, ptr %pp, align 8
  %chi = lshr i128 %s2, 64
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, 8
  br i1 %more, label %addloop, label %addfin
addfin:
  call void @sc_reduce(ptr %Ssc, ptr %prod, i64 8)
  %sigS = getelementptr inbounds i8, ptr %sig, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr %sigS, ptr %Ssc, i64 32, i1 false)
  ret void
}

define i32 @universe_crypto_ed25519_verify(ptr %pub, ptr %msg, i64 %msglen, ptr %sig) local_unnamed_addr #1 {
entry:
  %A = alloca [16 x i64], align 8
  %Rpt = alloca [16 x i64], align 8
  %B = alloca [16 x i64], align 8
  %S = alloca [4 x i64], align 8
  %ctx = alloca [200 x i8], align 8
  %hk = alloca [64 x i8], align 8
  %ksc = alloca [4 x i64], align 8
  %SB = alloca [16 x i64], align 8
  %kA = alloca [16 x i64], align 8
  %P2 = alloca [16 x i64], align 8
  ; decode A
  %okA = call i1 @pt_decode(ptr %A, ptr %pub)
  br i1 %okA, label %decR, label %invalid
decR:
  %okR = call i1 @pt_decode(ptr %Rpt, ptr %sig)
  br i1 %okR, label %loadS, label %invalid
loadS:
  %sigS = getelementptr inbounds i8, ptr %sig, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr %S, ptr %sigS, i64 32, i1 false)
  %cS = call i32 @universe_bignum_cmp_n(ptr %S, ptr @ED_L, i64 4)
  %Sok = icmp slt i32 %cS, 0
  br i1 %Sok, label %hashk, label %invalid
hashk:
  ; k = SHA512(enc(R) || pub || msg) mod L
  call void @universe_crypto_sha512_init(ptr %ctx)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %sig, i64 32)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %pub, i64 32)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %msg, i64 %msglen)
  call void @universe_crypto_sha512_final(ptr %ctx, ptr %hk)
  call void @sc_reduce(ptr %ksc, ptr %hk, i64 8)
  ; SB = [S]B ; kA = [k]A ; P2 = R + kA
  call void @pt_load_base(ptr %B)
  call void @pt_scalarmult(ptr %SB, ptr %S, ptr %B)
  call void @pt_scalarmult(ptr %kA, ptr %ksc, ptr %A)
  call void @pt_add(ptr %P2, ptr %Rpt, ptr %kA)
  %eq = call i1 @pt_eq(ptr %SB, ptr %P2)
  br i1 %eq, label %valid, label %invalid
valid:
  ret i32 0
invalid:
  ret i32 1
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
