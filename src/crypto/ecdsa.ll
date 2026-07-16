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

; ECDSA over NIST P-256 (FIPS 186-4 / SEC1): sign (with a supplied nonce k) and
; verify. Built on the bignum kernels; this module is the curve group law and
; the ECDSA equations.
;
; DESIGN:
;   FIELD Fp and SCALAR Fn arithmetic reuse bignum: modmul = bignum_mul (4x4 ->
;   8 limbs) then bignum_mod; modadd/modsub use add_n/sub_n + one conditional
;   fix (operands < modulus, so a single correction always suffices, including
;   the add carry-out case: when a+b overflows 2^256 the 4-limb sub_n by the
;   modulus yields exactly (a+b)-m). Inversion = bignum_modinv. CORRECTNESS-
;   FIRST (mod-per-multiply), not the fastest reduction.
;
;   GROUP LAW: AFFINE coordinates with an explicit infinity flag (point = x@0,
;   y@32, inf@64; 96 bytes). Affine trades a modular inversion per add/double
;   for exception-free, obviously-correct formulas (Weierstrass add/double with
;   the P==Q, P==-Q, and O cases handled explicitly) — the right call for a
;   from-scratch KAT-gated first version. a = -3. Scalar multiply is MSB-first
;   double-and-add over 256 bits (NOT constant-time).
;
;   ECDSA: e = the message hash reduced mod n (input hash is 32 big-endian bytes
;   = bits2int for P-256 where hashlen==qlen). verify: reject unless 0<r,s<n;
;   w=s^-1; R=[e w]G+[r w]Q; accept iff R.x mod n == r. sign(k): r=([k]G).x mod n,
;   s=k^-1(e+r d) mod n; reject zero r/s (caller must retry with a new k).
;
; HARDENING-TODO: NOT constant-time (ladder + field mod branch on secrets); the
;   nonce k is caller-supplied (no in-module RFC-6979/CSPRNG generation yet);
;   no scratch zeroization. Deferred to the hardening phase.
;
; API (C ABI, nounwind). All scalars/coords are 4-limb little-endian u64.
;   i32 universe_crypto_ecdsa_p256_verify(ptr hash_be32, ptr qx, ptr qy, ptr r, ptr s) ; 0 ok / 1 bad
;   i32 universe_crypto_ecdsa_p256_sign(ptr hash_be32, ptr priv, ptr k, ptr out_r, ptr out_s) ; 0 ok / 8 retry

declare void @universe_bignum_mul(ptr, ptr, i64, ptr, i64)
declare i32 @universe_bignum_mod(ptr, ptr, i64, ptr, i64)
declare i64 @universe_bignum_add_n(ptr, ptr, ptr, i64)
declare i64 @universe_bignum_sub_n(ptr, ptr, ptr, i64)
declare i32 @universe_bignum_cmp_n(ptr, ptr, i64)
declare i32 @universe_bignum_is_zero_n(ptr, i64)
declare i32 @universe_bignum_modinv(ptr, ptr, ptr, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; ---- constants (32-byte little-endian, align 8) ----
@ECP  = private unnamed_addr constant [32 x i8] c"\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\ff\ff\ff\ff", align 8
@ECA  = private unnamed_addr constant [32 x i8] c"\fc\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\ff\ff\ff\ff", align 8
@ECN  = private unnamed_addr constant [32 x i8] c"\51\25\63\fc\c2\ca\b9\f3\84\9e\17\a7\ad\fa\e6\bc\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\ff\ff\ff\ff", align 8
@ECGX = private unnamed_addr constant [32 x i8] c"\96\c2\98\d8\45\39\a1\f4\a0\33\eb\2d\81\7d\03\77\f2\40\a4\63\e5\e6\bc\f8\47\42\2c\e1\f2\d1\17\6b", align 8
@ECGY = private unnamed_addr constant [32 x i8] c"\f5\51\bf\37\68\40\b6\cb\ce\5e\31\6b\57\33\ce\2b\16\9e\0f\7c\4a\eb\e7\8e\9b\7f\1a\fe\e2\42\e3\4f", align 8
@ECZERO = private unnamed_addr constant [32 x i8] zeroinitializer, align 8

; ---- modular helpers (m = modulus ptr, 4 limbs) ----
define internal void @modmul(ptr %r, ptr %a, ptr %b, ptr %m) #0 {
entry:
  %t = alloca [8 x i64], align 8
  call void @universe_bignum_mul(ptr %t, ptr %a, i64 4, ptr %b, i64 4)
  %rc = call i32 @universe_bignum_mod(ptr %r, ptr %t, i64 8, ptr %m, i64 4)
  ret void
}

define internal void @modadd(ptr %r, ptr %a, ptr %b, ptr %m) #0 {
entry:
  %c = call i64 @universe_bignum_add_n(ptr %r, ptr %a, ptr %b, i64 4)
  %cnz = icmp ne i64 %c, 0
  %cmp = call i32 @universe_bignum_cmp_n(ptr %r, ptr %m, i64 4)
  %ge = icmp sge i32 %cmp, 0
  %need = or i1 %cnz, %ge
  br i1 %need, label %sub, label %done
sub:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr %r, ptr %m, i64 4)
  br label %done
done:
  ret void
}

define internal void @modsub(ptr %r, ptr %a, ptr %b, ptr %m) #0 {
entry:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr %a, ptr %b, i64 4)
  %neg = icmp ne i64 %bw, 0
  br i1 %neg, label %add, label %done
add:
  %c = call i64 @universe_bignum_add_n(ptr %r, ptr %r, ptr %m, i64 4)
  br label %done
done:
  ret void
}

define internal i1 @eq4(ptr %a, ptr %b) #0 {
entry:
  %c = call i32 @universe_bignum_cmp_n(ptr %a, ptr %b, i64 4)
  %e = icmp eq i32 %c, 0
  ret i1 %e
}

define internal i1 @is_zero4(ptr %a) #0 {
entry:
  %z = call i32 @universe_bignum_is_zero_n(ptr %a, i64 4)
  %r = icmp ne i32 %z, 0
  ret i1 %r
}

; ---- point ops (affine; point = x@0 y@32 inf@64) ----

define internal void @pt_set_inf(ptr %r) #0 {
entry:
  call void @llvm.memset.p0.i64(ptr %r, i8 0, i64 96, i1 false)
  %ip = getelementptr inbounds i8, ptr %r, i64 64
  store i64 1, ptr %ip, align 8
  ret void
}

define internal i1 @pt_is_inf(ptr %p) #0 {
entry:
  %ip = getelementptr inbounds i8, ptr %p, i64 64
  %v = load i64, ptr %ip, align 8
  %r = icmp ne i64 %v, 0
  ret i1 %r
}

; r = 2*p1
define internal void @pt_double(ptr %r, ptr %p1) #0 {
entry:
  %l = alloca [4 x i64], align 8
  %t = alloca [4 x i64], align 8
  %num = alloca [4 x i64], align 8
  %den = alloca [4 x i64], align 8
  %di = alloca [4 x i64], align 8
  %l2 = alloca [4 x i64], align 8
  %x3 = alloca [4 x i64], align 8
  %y3 = alloca [4 x i64], align 8
  %tx = alloca [4 x i64], align 8
  %x1 = getelementptr inbounds i8, ptr %p1, i64 0
  %y1 = getelementptr inbounds i8, ptr %p1, i64 32
  %inf = call i1 @pt_is_inf(ptr %p1)
  %y0 = call i1 @is_zero4(ptr %y1)
  %deg = or i1 %inf, %y0
  br i1 %deg, label %toinf, label %compute
toinf:
  call void @pt_set_inf(ptr %r)
  ret void
compute:
  ; num = 3*x1^2 + a
  call void @modmul(ptr %t, ptr %x1, ptr %x1, ptr @ECP)
  call void @modadd(ptr %num, ptr %t, ptr %t, ptr @ECP)
  call void @modadd(ptr %num, ptr %num, ptr %t, ptr @ECP)
  call void @modadd(ptr %num, ptr %num, ptr @ECA, ptr @ECP)
  ; den = 2*y1 ; di = den^-1
  call void @modadd(ptr %den, ptr %y1, ptr %y1, ptr @ECP)
  %rc = call i32 @universe_bignum_modinv(ptr %di, ptr %den, ptr @ECP, i64 4)
  call void @modmul(ptr %l, ptr %num, ptr %di, ptr @ECP)
  ; x3 = l^2 - 2*x1
  call void @modmul(ptr %l2, ptr %l, ptr %l, ptr @ECP)
  call void @modadd(ptr %tx, ptr %x1, ptr %x1, ptr @ECP)
  call void @modsub(ptr %x3, ptr %l2, ptr %tx, ptr @ECP)
  ; y3 = l*(x1-x3) - y1
  call void @modsub(ptr %tx, ptr %x1, ptr %x3, ptr @ECP)
  call void @modmul(ptr %y3, ptr %l, ptr %tx, ptr @ECP)
  call void @modsub(ptr %y3, ptr %y3, ptr %y1, ptr @ECP)
  %rx = getelementptr inbounds i8, ptr %r, i64 0
  %ry = getelementptr inbounds i8, ptr %r, i64 32
  %ri = getelementptr inbounds i8, ptr %r, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %rx, ptr %x3, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %ry, ptr %y3, i64 32, i1 false)
  store i64 0, ptr %ri, align 8
  ret void
}

; r = p1 + p2
define internal void @pt_add(ptr %r, ptr %p1, ptr %p2) #0 {
entry:
  %l = alloca [4 x i64], align 8
  %dx = alloca [4 x i64], align 8
  %dy = alloca [4 x i64], align 8
  %di = alloca [4 x i64], align 8
  %l2 = alloca [4 x i64], align 8
  %x3 = alloca [4 x i64], align 8
  %y3 = alloca [4 x i64], align 8
  %tx = alloca [4 x i64], align 8
  %i1 = call i1 @pt_is_inf(ptr %p1)
  br i1 %i1, label %copy2, label %chk2
copy2:
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %p2, i64 96, i1 false)
  ret void
chk2:
  %i2 = call i1 @pt_is_inf(ptr %p2)
  br i1 %i2, label %copy1, label %general
copy1:
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %p1, i64 96, i1 false)
  ret void
general:
  %x1 = getelementptr inbounds i8, ptr %p1, i64 0
  %y1 = getelementptr inbounds i8, ptr %p1, i64 32
  %x2 = getelementptr inbounds i8, ptr %p2, i64 0
  %y2 = getelementptr inbounds i8, ptr %p2, i64 32
  %xeq = call i1 @eq4(ptr %x1, ptr %x2)
  br i1 %xeq, label %samex, label %addg
samex:
  %yeq = call i1 @eq4(ptr %y1, ptr %y2)
  br i1 %yeq, label %dbl, label %toinf
dbl:
  call void @pt_double(ptr %r, ptr %p1)
  ret void
toinf:
  call void @pt_set_inf(ptr %r)
  ret void
addg:
  ; l = (y2-y1)/(x2-x1)
  call void @modsub(ptr %dy, ptr %y2, ptr %y1, ptr @ECP)
  call void @modsub(ptr %dx, ptr %x2, ptr %x1, ptr @ECP)
  %rc = call i32 @universe_bignum_modinv(ptr %di, ptr %dx, ptr @ECP, i64 4)
  call void @modmul(ptr %l, ptr %dy, ptr %di, ptr @ECP)
  ; x3 = l^2 - x1 - x2
  call void @modmul(ptr %l2, ptr %l, ptr %l, ptr @ECP)
  call void @modsub(ptr %x3, ptr %l2, ptr %x1, ptr @ECP)
  call void @modsub(ptr %x3, ptr %x3, ptr %x2, ptr @ECP)
  ; y3 = l*(x1-x3) - y1
  call void @modsub(ptr %tx, ptr %x1, ptr %x3, ptr @ECP)
  call void @modmul(ptr %y3, ptr %l, ptr %tx, ptr @ECP)
  call void @modsub(ptr %y3, ptr %y3, ptr %y1, ptr @ECP)
  %rx = getelementptr inbounds i8, ptr %r, i64 0
  %ry = getelementptr inbounds i8, ptr %r, i64 32
  %ri = getelementptr inbounds i8, ptr %r, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %rx, ptr %x3, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %ry, ptr %y3, i64 32, i1 false)
  store i64 0, ptr %ri, align 8
  ret void
}

; r = [scalar] * point  (scalar 4 LE limbs, MSB-first)
define internal void @pt_scalarmult(ptr %r, ptr %scalar, ptr %point) #0 {
entry:
  %R = alloca [96 x i8], align 8
  call void @pt_set_inf(ptr %R)
  br label %loop
loop:
  %k = phi i64 [ 255, %entry ], [ %kn, %step ]
  call void @pt_double(ptr %R, ptr %R)
  %li = lshr i64 %k, 6
  %off = and i64 %k, 63
  %lp = getelementptr inbounds i64, ptr %scalar, i64 %li
  %lv = load i64, ptr %lp, align 8
  %sh = lshr i64 %lv, %off
  %bit = and i64 %sh, 1
  %set = icmp eq i64 %bit, 1
  br i1 %set, label %addb, label %step
addb:
  call void @pt_add(ptr %R, ptr %R, ptr %point)
  br label %step
step:
  %kn = add i64 %k, -1
  %more = icmp sge i64 %kn, 0
  br i1 %more, label %loop, label %done
done:
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %R, i64 96, i1 false)
  ret void
}

define internal void @load_G(ptr %r) #0 {
entry:
  %rx = getelementptr inbounds i8, ptr %r, i64 0
  %ry = getelementptr inbounds i8, ptr %r, i64 32
  %ri = getelementptr inbounds i8, ptr %r, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %rx, ptr @ECGX, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %ry, ptr @ECGY, i64 32, i1 false)
  store i64 0, ptr %ri, align 8
  ret void
}

; reverse 32 big-endian bytes into a 4-limb LE buffer, then reduce mod n (once)
define internal void @hash_to_e(ptr %e, ptr %hash_be) #0 {
entry:
  br label %rev
rev:
  %i = phi i64 [ 0, %entry ], [ %in, %rev ]
  %j = sub i64 31, %i
  %sp = getelementptr inbounds i8, ptr %hash_be, i64 %j
  %v = load i8, ptr %sp, align 1
  %dp = getelementptr inbounds i8, ptr %e, i64 %i
  store i8 %v, ptr %dp, align 1
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, 32
  br i1 %more, label %rev, label %reduce
reduce:
  %cmp = call i32 @universe_bignum_cmp_n(ptr %e, ptr @ECN, i64 4)
  %ge = icmp sge i32 %cmp, 0
  br i1 %ge, label %sub, label %done
sub:
  %bw = call i64 @universe_bignum_sub_n(ptr %e, ptr %e, ptr @ECN, i64 4)
  br label %done
done:
  ret void
}

; ---- public API ----

define i32 @universe_crypto_ecdsa_p256_verify(ptr %hash, ptr %qx, ptr %qy, ptr %r, ptr %s) local_unnamed_addr #1 {
entry:
  %e = alloca [4 x i64], align 8
  %w = alloca [4 x i64], align 8
  %u1 = alloca [4 x i64], align 8
  %u2 = alloca [4 x i64], align 8
  %Q = alloca [96 x i8], align 8
  %G = alloca [96 x i8], align 8
  %P1 = alloca [96 x i8], align 8
  %P2 = alloca [96 x i8], align 8
  %Rr = alloca [96 x i8], align 8
  %v = alloca [4 x i64], align 8
  ; range: 0 < r,s < n
  %rz = call i1 @is_zero4(ptr %r)
  %sz = call i1 @is_zero4(ptr %s)
  %anyz = or i1 %rz, %sz
  br i1 %anyz, label %bad, label %rng
rng:
  %cr = call i32 @universe_bignum_cmp_n(ptr %r, ptr @ECN, i64 4)
  %rlt = icmp slt i32 %cr, 0
  %cs = call i32 @universe_bignum_cmp_n(ptr %s, ptr @ECN, i64 4)
  %slt = icmp slt i32 %cs, 0
  %both = and i1 %rlt, %slt
  br i1 %both, label %compute, label %bad
compute:
  call void @hash_to_e(ptr %e, ptr %hash)
  %ic = call i32 @universe_bignum_modinv(ptr %w, ptr %s, ptr @ECN, i64 4)
  call void @modmul(ptr %u1, ptr %e, ptr %w, ptr @ECN)
  call void @modmul(ptr %u2, ptr %r, ptr %w, ptr @ECN)
  ; build Q
  %qxp = getelementptr inbounds i8, ptr %Q, i64 0
  %qyp = getelementptr inbounds i8, ptr %Q, i64 32
  %qip = getelementptr inbounds i8, ptr %Q, i64 64
  call void @llvm.memcpy.p0.p0.i64(ptr %qxp, ptr %qx, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %qyp, ptr %qy, i64 32, i1 false)
  store i64 0, ptr %qip, align 8
  call void @load_G(ptr %G)
  call void @pt_scalarmult(ptr %P1, ptr %u1, ptr %G)
  call void @pt_scalarmult(ptr %P2, ptr %u2, ptr %Q)
  call void @pt_add(ptr %Rr, ptr %P1, ptr %P2)
  %rinf = call i1 @pt_is_inf(ptr %Rr)
  br i1 %rinf, label %bad, label %final
final:
  ; v = Rr.x mod n (one conditional subtract)
  %rx = getelementptr inbounds i8, ptr %Rr, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %v, ptr %rx, i64 32, i1 false)
  %cv = call i32 @universe_bignum_cmp_n(ptr %v, ptr @ECN, i64 4)
  %vge = icmp sge i32 %cv, 0
  br i1 %vge, label %vsub, label %vcmp
vsub:
  %bw = call i64 @universe_bignum_sub_n(ptr %v, ptr %v, ptr @ECN, i64 4)
  br label %vcmp
vcmp:
  %ok = call i1 @eq4(ptr %v, ptr %r)
  br i1 %ok, label %good, label %bad
good:
  ret i32 0
bad:
  ret i32 1
}

define i32 @universe_crypto_ecdsa_p256_sign(ptr %hash, ptr %priv, ptr %k, ptr %out_r, ptr %out_s) local_unnamed_addr #1 {
entry:
  %e = alloca [4 x i64], align 8
  %G = alloca [96 x i8], align 8
  %R1 = alloca [96 x i8], align 8
  %r = alloca [4 x i64], align 8
  %kinv = alloca [4 x i64], align 8
  %rd = alloca [4 x i64], align 8
  %esum = alloca [4 x i64], align 8
  %s = alloca [4 x i64], align 8
  call void @hash_to_e(ptr %e, ptr %hash)
  call void @load_G(ptr %G)
  call void @pt_scalarmult(ptr %R1, ptr %k, ptr %G)
  %rinf = call i1 @pt_is_inf(ptr %R1)
  br i1 %rinf, label %retry, label %getr
getr:
  ; r = R1.x mod n
  %rx = getelementptr inbounds i8, ptr %R1, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %rx, i64 32, i1 false)
  %cr = call i32 @universe_bignum_cmp_n(ptr %r, ptr @ECN, i64 4)
  %rge = icmp sge i32 %cr, 0
  br i1 %rge, label %rsub, label %rchk
rsub:
  %bw = call i64 @universe_bignum_sub_n(ptr %r, ptr %r, ptr @ECN, i64 4)
  br label %rchk
rchk:
  %rz = call i1 @is_zero4(ptr %r)
  br i1 %rz, label %retry, label %scomp
scomp:
  %ic = call i32 @universe_bignum_modinv(ptr %kinv, ptr %k, ptr @ECN, i64 4)
  call void @modmul(ptr %rd, ptr %r, ptr %priv, ptr @ECN)
  call void @modadd(ptr %esum, ptr %e, ptr %rd, ptr @ECN)
  call void @modmul(ptr %s, ptr %kinv, ptr %esum, ptr @ECN)
  %sz = call i1 @is_zero4(ptr %s)
  br i1 %sz, label %retry, label %emit
emit:
  call void @llvm.memcpy.p0.p0.i64(ptr %out_r, ptr %r, i64 32, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %out_s, ptr %s, i64 32, i1 false)
  ret i32 0
retry:
  ret i32 8
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
