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

; Big-integer arithmetic foundation (unsigned multiprecision).
; Prerequisite for RSA / ECDSA / Ed25519.
;
; DESIGN:
;   REPRESENTATION: caller-provided u64 limb buffers, little-endian limb
;   order (limb[0] is least significant), with an EXPLICIT limb count passed
;   per call. No hidden handle, no sign bit: every core op works on a raw
;   fixed-length limb array. This is the crypto-native shape — RSA/ECC operate
;   on fixed modulus widths, so allocation-free fixed-length kernels are both
;   simpler to prove correct and faster (no realloc/canonicalization in the
;   hot path). Convenience alloc/set helpers are provided but optional.
;
;   CARRY/BORROW: every add/sub/mul limb step is computed in i128 and split
;   into {lo=trunc, hi=lshr 64}. This is the single most important correctness
;   decision: it makes carry/borrow propagation total and exact with no
;   `add nuw x,-1` landmine (Hazard #1) — decrements use plain add/sub and
;   wrapping counters carry no nuw/nsw. add-of-two-limbs+carry <= 2^65-1 and
;   mul-of-two-limbs+two-limbs <= 2^128-1 both fit i128 with no overflow.
;
;   DIVISION: Knuth Algorithm D (normalized long division) for n>=2 limb
;   divisors; a dedicated i128 long-division loop for single-limb divisors.
;   The multiply-subtract and add-back steps use signed i128 differences so a
;   negative partial remainder is detected by `icmp slt i128 d, 0` — no
;   hand-rolled borrow chain to get wrong.
;
;   MONTGOMERY: CIOS form. montmul(a,b) = a*b*R^-1 mod n with R = 2^(64*s).
;   n0inv = -n[0]^-1 mod 2^64 via Newton iteration (odd modulus required).
;   R^2 mod n by 128*s modular doublings (setup only, off the hot path).
;   modexp = MSB-first square-and-multiply in the Montgomery domain — the
;   priority correctness target (RSA/ECDSA modexp).
;
;   modinv/gcd: Euclid/extended-Euclid via divmod. modinv keeps the Bezout
;   coefficient reduced in [0,n) at every step (modular subtract), so no
;   signed big integers are needed — matches the unsigned representation.
;
;   SHIFTS: bit shifts take an amount in [0,63]; the 0 case is special-cased
;   (a `>> (64-0)` would be shift-by-64 poison). Limb-granular shifts are the
;   caller's job via buffer offsetting.
;
; HARDENING-TODO: fast, NOT constant-time. Division, comparisons and the
;   square-and-multiply ladder branch on secret data; no memory zeroization of
;   scratch. Constant-time ladders, blinding and scratch scrubbing are
;   deferred to the hardening phase (this feeds crypto — see CLAUDE.md).
;
; API (0 OK, 2 OUT_OF_MEMORY, 5 NOT_FOUND, 8 INVALID_ARG). Limbs LE u64.
;   ptr  universe_bignum_alloc(i64 nlimbs)
;   void universe_bignum_free(ptr p)
;   void universe_bignum_set_u64(ptr r, i64 s, i64 v)
;   i64  universe_bignum_normalize_len(ptr a, i64 n)
;   i64  universe_bignum_add_n(ptr r, ptr a, ptr b, i64 n)   ; -> carry
;   i64  universe_bignum_sub_n(ptr r, ptr a, ptr b, i64 n)   ; -> borrow
;   i32  universe_bignum_cmp_n(ptr a, ptr b, i64 n)          ; -1/0/1
;   i32  universe_bignum_is_zero_n(ptr a, i64 n)             ; 1/0
;   i64  universe_bignum_bit_length_n(ptr a, i64 n)
;   i64  universe_bignum_shl_bits(ptr r, ptr a, i64 n, i64 bits) ; ->overflow
;   void universe_bignum_shr_bits(ptr r, ptr a, i64 n, i64 bits)
;   void universe_bignum_mul(ptr r, ptr a, i64 an, ptr b, i64 bn)
;   i32  universe_bignum_divmod(ptr q, ptr r, ptr u, i64 m, ptr v, i64 n)
;   i32  universe_bignum_mod(ptr r, ptr u, i64 m, ptr v, i64 n)
;   i64  universe_bignum_mont_n0inv(ptr n)
;   void universe_bignum_mont_rr(ptr rr, ptr n, i64 s)
;   void universe_bignum_montmul(ptr r, ptr a, ptr b, ptr n, i64 s,
;                                i64 n0inv, ptr scratch)      ; scratch: s+2
;   i32  universe_bignum_modexp(ptr r, ptr base, i64 basel,
;                               ptr exp, i64 expl, ptr n, i64 s)
;   i32  universe_bignum_modinv(ptr r, ptr a, ptr n, i64 s)
;   i32  universe_bignum_gcd(ptr g, ptr a, ptr b, i64 s)

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; ---------------------------------------------------------------- alloc/set

define ptr @universe_bignum_alloc(i64 %nlimbs) local_unnamed_addr #1 {
entry:
  %bad = icmp eq i64 %nlimbs, 0
  br i1 %bad, label %fail, label %ok, !prof !0
ok:
  %p = call ptr @calloc(i64 %nlimbs, i64 8)
  ret ptr %p
fail:
  ret ptr null
}

define void @universe_bignum_free(ptr %p) local_unnamed_addr #1 {
entry:
  call void @free(ptr %p)
  ret void
}

define void @universe_bignum_set_u64(ptr %r, i64 %s, i64 %v) local_unnamed_addr #0 {
entry:
  %bytes = shl i64 %s, 3
  call void @llvm.memset.p0.i64(ptr %r, i8 0, i64 %bytes, i1 false)
  store i64 %v, ptr %r, align 8
  ret void
}

define i64 @universe_bignum_normalize_len(ptr %a, i64 %n) local_unnamed_addr #2 {
entry:
  %i0 = add i64 %n, -1
  br label %loop
loop:
  %i = phi i64 [ %i0, %entry ], [ %in, %cont ]
  %done = icmp slt i64 %i, 0
  br i1 %done, label %zero, label %check
check:
  %p = getelementptr inbounds i64, ptr %a, i64 %i
  %x = load i64, ptr %p, align 8
  %nz = icmp ne i64 %x, 0
  br i1 %nz, label %found, label %cont
found:
  %len = add i64 %i, 1
  ret i64 %len
cont:
  %in = add i64 %i, -1
  br label %loop
zero:
  ret i64 0
}

; ---------------------------------------------------------------- add / sub

define i64 @universe_bignum_add_n(ptr %r, ptr %a, ptr %b, i64 %n) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %ret0, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %carry = phi i128 [ 0, %entry ], [ %chi, %loop ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %av = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds i64, ptr %b, i64 %i
  %bv = load i64, ptr %bp, align 8
  %a128 = zext i64 %av to i128
  %b128 = zext i64 %bv to i128
  %s1 = add i128 %a128, %b128
  %s2 = add i128 %s1, %carry
  %lo = trunc i128 %s2 to i64
  %rp = getelementptr inbounds i64, ptr %r, i64 %i
  store i64 %lo, ptr %rp, align 8
  %chi = lshr i128 %s2, 64
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %out = trunc i128 %chi to i64
  ret i64 %out
ret0:
  ret i64 0
}

define i64 @universe_bignum_sub_n(ptr %r, ptr %a, ptr %b, i64 %n) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %ret0, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %borrow = phi i128 [ 0, %entry ], [ %bnext, %loop ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %av = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds i64, ptr %b, i64 %i
  %bv = load i64, ptr %bp, align 8
  %a128 = zext i64 %av to i128
  %b128 = zext i64 %bv to i128
  %d1 = sub i128 %a128, %b128
  %d2 = sub i128 %d1, %borrow
  %lo = trunc i128 %d2 to i64
  %rp = getelementptr inbounds i64, ptr %r, i64 %i
  store i64 %lo, ptr %rp, align 8
  %neg = icmp slt i128 %d2, 0
  %bnext = zext i1 %neg to i128
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %out = zext i1 %neg to i64
  ret i64 %out
ret0:
  ret i64 0
}

; ---------------------------------------------------------------- compare

define i32 @universe_bignum_cmp_n(ptr %a, ptr %b, i64 %n) local_unnamed_addr #2 {
entry:
  %i0 = add i64 %n, -1
  br label %loop
loop:
  %i = phi i64 [ %i0, %entry ], [ %in, %cont ]
  %done = icmp slt i64 %i, 0
  br i1 %done, label %eq, label %check
check:
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %av = load i64, ptr %ap, align 8
  %bp = getelementptr inbounds i64, ptr %b, i64 %i
  %bv = load i64, ptr %bp, align 8
  %lt = icmp ult i64 %av, %bv
  br i1 %lt, label %less, label %maybegt
maybegt:
  %gt = icmp ugt i64 %av, %bv
  br i1 %gt, label %greater, label %cont
cont:
  %in = add i64 %i, -1
  br label %loop
less:
  ret i32 -1
greater:
  ret i32 1
eq:
  ret i32 0
}

define i32 @universe_bignum_is_zero_n(ptr %a, i64 %n) local_unnamed_addr #2 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %yes, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %p = getelementptr inbounds i64, ptr %a, i64 %i
  %x = load i64, ptr %p, align 8
  %nz = icmp ne i64 %x, 0
  br i1 %nz, label %no, label %cont
cont:
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %yes
yes:
  ret i32 1
no:
  ret i32 0
}

define i64 @universe_bignum_bit_length_n(ptr %a, i64 %n) local_unnamed_addr #2 {
entry:
  %i0 = add i64 %n, -1
  br label %loop
loop:
  %i = phi i64 [ %i0, %entry ], [ %in, %cont ]
  %done = icmp slt i64 %i, 0
  br i1 %done, label %zero, label %check
check:
  %p = getelementptr inbounds i64, ptr %a, i64 %i
  %x = load i64, ptr %p, align 8
  %nz = icmp ne i64 %x, 0
  br i1 %nz, label %found, label %cont
found:
  %clz = call i64 @llvm.ctlz.i64(i64 %x, i1 true)
  %hibits = sub i64 64, %clz
  %base = shl i64 %i, 6
  %len = add i64 %base, %hibits
  ret i64 %len
cont:
  %in = add i64 %i, -1
  br label %loop
zero:
  ret i64 0
}

; ---------------------------------------------------------------- shifts

define i64 @universe_bignum_shl_bits(ptr %r, ptr %a, i64 %n, i64 %bits) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %bits, 0
  br i1 %z, label %copy, label %shift
copy:
  %bytes = shl i64 %n, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %a, i64 %bytes, i1 false)
  ret i64 0
shift:
  %inv = sub i64 64, %bits
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %ret0, label %loop
loop:
  %i = phi i64 [ 0, %shift ], [ %in, %loop ]
  %carry = phi i64 [ 0, %shift ], [ %newcarry, %loop ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %x = load i64, ptr %ap, align 8
  %up = shl i64 %x, %bits
  %val = or i64 %up, %carry
  %rp = getelementptr inbounds i64, ptr %r, i64 %i
  store i64 %val, ptr %rp, align 8
  %newcarry = lshr i64 %x, %inv
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %newcarry
ret0:
  ret i64 0
}

define void @universe_bignum_shr_bits(ptr %r, ptr %a, i64 %n, i64 %bits) local_unnamed_addr #0 {
entry:
  %z = icmp eq i64 %bits, 0
  br i1 %z, label %copy, label %shift
copy:
  %bytes = shl i64 %n, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %a, i64 %bytes, i1 false)
  ret void
shift:
  %inv = sub i64 64, %bits
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop
loop:
  %i = phi i64 [ 0, %shift ], [ %in, %storeb ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %x = load i64, ptr %ap, align 8
  %lo = lshr i64 %x, %bits
  %in = add i64 %i, 1
  %hasnext = icmp ult i64 %in, %n
  br i1 %hasnext, label %withnext, label %top
withnext:
  %np = getelementptr inbounds i64, ptr %a, i64 %in
  %xn = load i64, ptr %np, align 8
  %hi = shl i64 %xn, %inv
  br label %storeb
top:
  br label %storeb
storeb:
  %hival = phi i64 [ %hi, %withnext ], [ 0, %top ]
  %val = or i64 %lo, %hival
  %rp = getelementptr inbounds i64, ptr %r, i64 %i
  store i64 %val, ptr %rp, align 8
  br i1 %hasnext, label %loop, label %done
done:
  ret void
}

; ---------------------------------------------------------------- multiply

define void @universe_bignum_mul(ptr %r, ptr %a, i64 %an, ptr %b, i64 %bn) local_unnamed_addr #0 {
entry:
  %tot = add i64 %an, %bn
  %rbytes = shl i64 %tot, 3
  call void @llvm.memset.p0.i64(ptr %r, i8 0, i64 %rbytes, i1 false)
  %az = icmp eq i64 %an, 0
  %bz = icmp eq i64 %bn, 0
  %anyz = or i1 %az, %bz
  br i1 %anyz, label %done, label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %inext, %istore ]
  %ap = getelementptr inbounds i64, ptr %a, i64 %i
  %ai = load i64, ptr %ap, align 8
  %ai128 = zext i64 %ai to i128
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %jnext, %inner ]
  %carry = phi i128 [ 0, %outer ], [ %chi, %inner ]
  %bp = getelementptr inbounds i64, ptr %b, i64 %j
  %bj = load i64, ptr %bp, align 8
  %bj128 = zext i64 %bj to i128
  %ij = add i64 %i, %j
  %rp = getelementptr inbounds i64, ptr %r, i64 %ij
  %rv = load i64, ptr %rp, align 8
  %rv128 = zext i64 %rv to i128
  %prod = mul i128 %ai128, %bj128
  %s1 = add i128 %prod, %rv128
  %s2 = add i128 %s1, %carry
  %lo = trunc i128 %s2 to i64
  store i64 %lo, ptr %rp, align 8
  %chi = lshr i128 %s2, 64
  %jnext = add i64 %j, 1
  %jmore = icmp ult i64 %jnext, %bn
  br i1 %jmore, label %inner, label %istore
istore:
  %ibn = add i64 %i, %bn
  %rtp = getelementptr inbounds i64, ptr %r, i64 %ibn
  %clo = trunc i128 %chi to i64
  store i64 %clo, ptr %rtp, align 8
  %inext = add i64 %i, 1
  %imore = icmp ult i64 %inext, %an
  br i1 %imore, label %outer, label %done
done:
  ret void
}

; ---------------------------------------------------------------- divmod

define i32 @universe_bignum_divmod(ptr %q, ptr %r, ptr %u, i64 %m, ptr %v, i64 %n) local_unnamed_addr #1 {
entry:
  %nz = icmp eq i64 %n, 0
  br i1 %nz, label %inval, label %chkm
inval:
  ret i32 8
chkm:
  %mz = icmp eq i64 %m, 0
  br i1 %mz, label %uzero, label %chkn1
uzero:
  %rbytes0 = shl i64 %n, 3
  call void @llvm.memset.p0.i64(ptr %r, i8 0, i64 %rbytes0, i1 false)
  ret i32 0
chkn1:
  %isn1 = icmp eq i64 %n, 1
  br i1 %isn1, label %single, label %multi

single:
  %v0 = load i64, ptr %v, align 8
  %v0z = icmp eq i64 %v0, 0
  br i1 %v0z, label %inval, label %sinit
sinit:
  %v0128 = zext i64 %v0 to i128
  %si0 = add i64 %m, -1
  br label %sloop
sloop:
  %si = phi i64 [ %si0, %sinit ], [ %sin, %sloop ]
  %rem = phi i128 [ 0, %sinit ], [ %newrem, %sloop ]
  %up = getelementptr inbounds i64, ptr %u, i64 %si
  %uv = load i64, ptr %up, align 8
  %uv128 = zext i64 %uv to i128
  %remsh = shl i128 %rem, 64
  %num = or i128 %remsh, %uv128
  %qd = udiv i128 %num, %v0128
  %qlo = trunc i128 %qd to i64
  %qp = getelementptr inbounds i64, ptr %q, i64 %si
  store i64 %qlo, ptr %qp, align 8
  %qv = mul i128 %qd, %v0128
  %newrem = sub i128 %num, %qv
  %sin = add i64 %si, -1
  %smore = icmp sge i64 %sin, 0
  br i1 %smore, label %sloop, label %sdone
sdone:
  %remfinal = phi i128 [ %newrem, %sloop ]
  %remlo = trunc i128 %remfinal to i64
  store i64 %remlo, ptr %r, align 8
  ret i32 0

; ---- multi-limb divisor (n >= 2): Knuth Algorithm D
multi:
  %vtopidx = add i64 %n, -1
  %vtopp = getelementptr inbounds i64, ptr %v, i64 %vtopidx
  %vtopraw = load i64, ptr %vtopp, align 8
  %vtopz = icmp eq i64 %vtopraw, 0
  br i1 %vtopz, label %inval, label %chksize
chksize:
  %mltn = icmp ult i64 %m, %n
  br i1 %mltn, label %qzero, label %knuth
qzero:
  ; quotient 0, remainder = u zero-extended to n limbs
  store i64 0, ptr %q, align 8
  %ubytes = shl i64 %m, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %u, i64 %ubytes, i1 false)
  %rtail = getelementptr inbounds i64, ptr %r, i64 %m
  %tailn = sub i64 %n, %m
  %tailb = shl i64 %tailn, 3
  call void @llvm.memset.p0.i64(ptr %rtail, i8 0, i64 %tailb, i1 false)
  ret i32 0
knuth:
  %shift = call i64 @llvm.ctlz.i64(i64 %vtopraw, i1 true)
  %m1 = add i64 %m, 1
  %unbytes = shl i64 %m1, 3
  %un = call ptr @malloc(i64 %unbytes)
  %unnull = icmp eq ptr %un, null
  br i1 %unnull, label %oom0, label %allocvn
oom0:
  ret i32 2
allocvn:
  %vnbytes = shl i64 %n, 3
  %vn = call ptr @malloc(i64 %vnbytes)
  %vnnull = icmp eq ptr %vn, null
  br i1 %vnnull, label %freeun_oom, label %normalize
freeun_oom:
  call void @free(ptr %un)
  ret i32 2
normalize:
  %vovf = call i64 @universe_bignum_shl_bits(ptr %vn, ptr %v, i64 %n, i64 %shift)
  %uovf = call i64 @universe_bignum_shl_bits(ptr %un, ptr %u, i64 %m, i64 %shift)
  %unm = getelementptr inbounds i64, ptr %un, i64 %m
  store i64 %uovf, ptr %unm, align 8
  ; constants for the loop
  %vtn1 = getelementptr inbounds i64, ptr %vn, i64 %vtopidx
  %vntop = load i64, ptr %vtn1, align 8
  %vntop128 = zext i64 %vntop to i128
  %secidx = add i64 %n, -2
  %vsecp = getelementptr inbounds i64, ptr %vn, i64 %secidx
  %vnsec = load i64, ptr %vsecp, align 8
  %vnsec128 = zext i64 %vnsec to i128
  %j0 = sub i64 %m, %n
  br label %jloop
jloop:
  %j = phi i64 [ %j0, %normalize ], [ %jnext, %jnextb ]
  %jvalid = icmp sge i64 %j, 0
  br i1 %jvalid, label %jbody, label %jdone
jbody:
  %jn = add i64 %j, %n
  %jn1 = add i64 %jn, -1
  %jn2 = add i64 %jn, -2
  %ujnp = getelementptr inbounds i64, ptr %un, i64 %jn
  %ujn = load i64, ptr %ujnp, align 8
  %ujn1p = getelementptr inbounds i64, ptr %un, i64 %jn1
  %ujn1 = load i64, ptr %ujn1p, align 8
  %ujn128 = zext i64 %ujn to i128
  %ujn1128 = zext i64 %ujn1 to i128
  %ujnsh = shl i128 %ujn128, 64
  %numj = or i128 %ujnsh, %ujn1128
  %qhat0 = udiv i128 %numj, %vntop128
  %qprod0 = mul i128 %qhat0, %vntop128
  %rhat0 = sub i128 %numj, %qprod0
  br label %corr
corr:
  %qh = phi i128 [ %qhat0, %jbody ], [ %qh2, %corrdo ]
  %rh = phi i128 [ %rhat0, %jbody ], [ %rh2, %corrdo ]
  %cond1 = icmp uge i128 %qh, 18446744073709551616
  %ujn2p = getelementptr inbounds i64, ptr %un, i64 %jn2
  %ujn2 = load i64, ptr %ujn2p, align 8
  %ujn2128 = zext i64 %ujn2 to i128
  %lhs = mul i128 %qh, %vnsec128
  %rhsh = shl i128 %rh, 64
  %rhs = or i128 %rhsh, %ujn2128
  %cond2 = icmp ugt i128 %lhs, %rhs
  %need = or i1 %cond1, %cond2
  br i1 %need, label %corrdo, label %mulsub
corrdo:
  %qh2 = sub i128 %qh, 1
  %rh2 = add i128 %rh, %vntop128
  %rhbig = icmp uge i128 %rh2, 18446744073709551616
  br i1 %rhbig, label %mulsub_break, label %corr
mulsub_break:
  br label %mulsub
mulsub:
  %qhat = phi i128 [ %qh, %corr ], [ %qh2, %mulsub_break ]
  br label %mloop
mloop:
  %i = phi i64 [ 0, %mulsub ], [ %inext, %mloop ]
  %k = phi i128 [ 0, %mulsub ], [ %knext, %mloop ]
  %borrow = phi i64 [ 0, %mulsub ], [ %bnext, %mloop ]
  %vnip = getelementptr inbounds i64, ptr %vn, i64 %i
  %vni = load i64, ptr %vnip, align 8
  %vni128 = zext i64 %vni to i128
  %pmul = mul i128 %qhat, %vni128
  %p = add i128 %pmul, %k
  %plo = trunc i128 %p to i64
  %knext = lshr i128 %p, 64
  %ij = add i64 %i, %j
  %unijp = getelementptr inbounds i64, ptr %un, i64 %ij
  %unij = load i64, ptr %unijp, align 8
  %unij128 = zext i64 %unij to i128
  %plo128 = zext i64 %plo to i128
  %bor128 = zext i64 %borrow to i128
  %d1 = sub i128 %unij128, %plo128
  %d = sub i128 %d1, %bor128
  %dlo = trunc i128 %d to i64
  store i64 %dlo, ptr %unijp, align 8
  %dneg = icmp slt i128 %d, 0
  %bnext = zext i1 %dneg to i64
  %inext = add i64 %i, 1
  %imore = icmp ult i64 %inext, %n
  br i1 %imore, label %mloop, label %mtail
mtail:
  %ujnp2 = getelementptr inbounds i64, ptr %un, i64 %jn
  %ujnv = load i64, ptr %ujnp2, align 8
  %ujnv128 = zext i64 %ujnv to i128
  %fbor128 = zext i64 %bnext to i128
  %df1 = sub i128 %ujnv128, %knext
  %df = sub i128 %df1, %fbor128
  %dflo = trunc i128 %df to i64
  store i64 %dflo, ptr %ujnp2, align 8
  %dfneg = icmp slt i128 %df, 0
  br i1 %dfneg, label %addback, label %qstore
qstore:
  %qjp = getelementptr inbounds i64, ptr %q, i64 %j
  %qval = trunc i128 %qhat to i64
  store i64 %qval, ptr %qjp, align 8
  br label %jnextb
addback:
  %qm1 = sub i128 %qhat, 1
  %qjp2 = getelementptr inbounds i64, ptr %q, i64 %j
  %qval2 = trunc i128 %qm1 to i64
  store i64 %qval2, ptr %qjp2, align 8
  br label %aloop
aloop:
  %ai = phi i64 [ 0, %addback ], [ %ainext, %aloop ]
  %ac = phi i128 [ 0, %addback ], [ %acnext, %aloop ]
  %avnp = getelementptr inbounds i64, ptr %vn, i64 %ai
  %avn = load i64, ptr %avnp, align 8
  %aij = add i64 %ai, %j
  %aunp = getelementptr inbounds i64, ptr %un, i64 %aij
  %aun = load i64, ptr %aunp, align 8
  %avn128 = zext i64 %avn to i128
  %aun128 = zext i64 %aun to i128
  %asum1 = add i128 %aun128, %avn128
  %asum = add i128 %asum1, %ac
  %aunlo = trunc i128 %asum to i64
  store i64 %aunlo, ptr %aunp, align 8
  %acnext = lshr i128 %asum, 64
  %ainext = add i64 %ai, 1
  %amore = icmp ult i64 %ainext, %n
  br i1 %amore, label %aloop, label %atail
atail:
  %curp = getelementptr inbounds i64, ptr %un, i64 %jn
  %cur = load i64, ptr %curp, align 8
  %cur128 = zext i64 %cur to i128
  %fsum = add i128 %cur128, %acnext
  %fsumlo = trunc i128 %fsum to i64
  store i64 %fsumlo, ptr %curp, align 8
  br label %jnextb
jnextb:
  %jnext = add i64 %j, -1
  br label %jloop
jdone:
  call void @universe_bignum_shr_bits(ptr %r, ptr %un, i64 %n, i64 %shift)
  call void @free(ptr %un)
  call void @free(ptr %vn)
  ret i32 0
}

; r = u mod v
define i32 @universe_bignum_mod(ptr %r, ptr %u, i64 %m, ptr %v, i64 %n) local_unnamed_addr #1 {
entry:
  %ge = icmp uge i64 %m, %n
  %diff = sub i64 %m, %n
  %diff1 = add i64 %diff, 1
  %qlen = select i1 %ge, i64 %diff1, i64 1
  %qbytes = shl i64 %qlen, 3
  %q = call ptr @malloc(i64 %qbytes)
  %qnull = icmp eq ptr %q, null
  br i1 %qnull, label %oom, label %run
oom:
  ret i32 2
run:
  %rc = call i32 @universe_bignum_divmod(ptr %q, ptr %r, ptr %u, i64 %m, ptr %v, i64 %n)
  call void @free(ptr %q)
  ret i32 %rc
}

; ---------------------------------------------------------------- montgomery

; -n[0]^-1 mod 2^64 (odd modulus). Newton iteration doubles correct bits.
define i64 @universe_bignum_mont_n0inv(ptr %n) local_unnamed_addr #2 {
entry:
  %n0 = load i64, ptr %n, align 8
  ; x = n0 (3 correct bits since n0 is odd); 6 Newton steps -> >64 bits
  %t1 = mul i64 %n0, %n0
  %s1 = sub i64 2, %t1
  %x1 = mul i64 %n0, %s1
  %t2 = mul i64 %n0, %x1
  %s2 = sub i64 2, %t2
  %x2 = mul i64 %x1, %s2
  %t3 = mul i64 %n0, %x2
  %s3 = sub i64 2, %t3
  %x3 = mul i64 %x2, %s3
  %t4 = mul i64 %n0, %x3
  %s4 = sub i64 2, %t4
  %x4 = mul i64 %x3, %s4
  %t5 = mul i64 %n0, %x4
  %s5 = sub i64 2, %t5
  %x5 = mul i64 %x4, %s5
  %t6 = mul i64 %n0, %x5
  %s6 = sub i64 2, %t6
  %x6 = mul i64 %x5, %s6
  %inv = sub i64 0, %x6
  ret i64 %inv
}

; rr = R^2 mod n, R = 2^(64*s). By 128*s modular doublings from 1.
define void @universe_bignum_mont_rr(ptr %rr, ptr %n, i64 %s) local_unnamed_addr #1 {
entry:
  %sbytes = shl i64 %s, 3
  call void @llvm.memset.p0.i64(ptr %rr, i8 0, i64 %sbytes, i1 false)
  store i64 1, ptr %rr, align 8
  %cnt = shl i64 %s, 7
  %cz = icmp eq i64 %cnt, 0
  br i1 %cz, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %iter ]
  %ov = call i64 @universe_bignum_shl_bits(ptr %rr, ptr %rr, i64 %s, i64 1)
  %ovnz = icmp ne i64 %ov, 0
  %c = call i32 @universe_bignum_cmp_n(ptr %rr, ptr %n, i64 %s)
  %cge = icmp sge i32 %c, 0
  %need = or i1 %ovnz, %cge
  br i1 %need, label %dosub, label %iter
dosub:
  %bo = call i64 @universe_bignum_sub_n(ptr %rr, ptr %rr, ptr %n, i64 %s)
  br label %iter
iter:
  %inext = add i64 %i, 1
  %more = icmp ult i64 %inext, %cnt
  br i1 %more, label %loop, label %done
done:
  ret void
}

; r = a*b*R^-1 mod n (CIOS). scratch t: s+2 limbs.
define void @universe_bignum_montmul(ptr %r, ptr %a, ptr %b, ptr %n, i64 %s, i64 %n0inv, ptr %t) local_unnamed_addr #1 {
entry:
  %tlimbs = add i64 %s, 2
  %tbytes = shl i64 %tlimbs, 3
  call void @llvm.memset.p0.i64(ptr %t, i8 0, i64 %tbytes, i1 false)
  %sm1 = add i64 %s, -1
  %sgt1 = icmp ugt i64 %s, 1
  %tsp = getelementptr inbounds i64, ptr %t, i64 %s
  %sp1 = add i64 %s, 1
  %tsp1 = getelementptr inbounds i64, ptr %t, i64 %sp1
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %inext, %ostep ]
  %bip = getelementptr inbounds i64, ptr %b, i64 %i
  %bi = load i64, ptr %bip, align 8
  %bi128 = zext i64 %bi to i128
  br label %inner1
inner1:
  %j1 = phi i64 [ 0, %outer ], [ %j1n, %inner1 ]
  %C1 = phi i128 [ 0, %outer ], [ %C1n, %inner1 ]
  %ajp = getelementptr inbounds i64, ptr %a, i64 %j1
  %aj = load i64, ptr %ajp, align 8
  %aj128 = zext i64 %aj to i128
  %tjp = getelementptr inbounds i64, ptr %t, i64 %j1
  %tj = load i64, ptr %tjp, align 8
  %tj128 = zext i64 %tj to i128
  %pa = mul i128 %aj128, %bi128
  %pa1 = add i128 %pa, %tj128
  %pa2 = add i128 %pa1, %C1
  %palo = trunc i128 %pa2 to i64
  store i64 %palo, ptr %tjp, align 8
  %C1n = lshr i128 %pa2, 64
  %j1n = add i64 %j1, 1
  %j1more = icmp ult i64 %j1n, %s
  br i1 %j1more, label %inner1, label %i1done
i1done:
  %ts = load i64, ptr %tsp, align 8
  %ts128 = zext i64 %ts to i128
  %sum1 = add i128 %ts128, %C1n
  %sum1lo = trunc i128 %sum1 to i64
  store i64 %sum1lo, ptr %tsp, align 8
  %hi1 = lshr i128 %sum1, 64
  %hi1lo = trunc i128 %hi1 to i64
  store i64 %hi1lo, ptr %tsp1, align 8
  %t0 = load i64, ptr %t, align 8
  %m = mul i64 %t0, %n0inv
  %m128 = zext i64 %m to i128
  %n0v = load i64, ptr %n, align 8
  %n0v128 = zext i64 %n0v to i128
  %p0 = mul i128 %m128, %n0v128
  %t0128 = zext i64 %t0 to i128
  %p0s = add i128 %p0, %t0128
  %C20 = lshr i128 %p0s, 64
  br i1 %sgt1, label %inner2, label %i2done
inner2:
  %j2 = phi i64 [ 1, %i1done ], [ %j2n, %inner2 ]
  %C2 = phi i128 [ %C20, %i1done ], [ %C2n, %inner2 ]
  %njp = getelementptr inbounds i64, ptr %n, i64 %j2
  %nj = load i64, ptr %njp, align 8
  %nj128 = zext i64 %nj to i128
  %tj2p = getelementptr inbounds i64, ptr %t, i64 %j2
  %tj2 = load i64, ptr %tj2p, align 8
  %tj2128 = zext i64 %tj2 to i128
  %pn = mul i128 %m128, %nj128
  %pn1 = add i128 %pn, %tj2128
  %pn2 = add i128 %pn1, %C2
  %j2m1 = add i64 %j2, -1
  %tj2m1p = getelementptr inbounds i64, ptr %t, i64 %j2m1
  %pnlo = trunc i128 %pn2 to i64
  store i64 %pnlo, ptr %tj2m1p, align 8
  %C2n = lshr i128 %pn2, 64
  %j2n = add i64 %j2, 1
  %j2more = icmp ult i64 %j2n, %s
  br i1 %j2more, label %inner2, label %i2done
i2done:
  %Cfin = phi i128 [ %C20, %i1done ], [ %C2n, %inner2 ]
  %ts2 = load i64, ptr %tsp, align 8
  %ts2128 = zext i64 %ts2 to i128
  %sum2 = add i128 %ts2128, %Cfin
  %sum2lo = trunc i128 %sum2 to i64
  %tsm1p = getelementptr inbounds i64, ptr %t, i64 %sm1
  store i64 %sum2lo, ptr %tsm1p, align 8
  %c3 = lshr i128 %sum2, 64
  %c3lo = trunc i128 %c3 to i64
  %tsp1v = load i64, ptr %tsp1, align 8
  %newts = add i64 %tsp1v, %c3lo
  store i64 %newts, ptr %tsp, align 8
  br label %ostep
ostep:
  %inext = add i64 %i, 1
  %imore = icmp ult i64 %inext, %s
  br i1 %imore, label %outer, label %reduce
reduce:
  %ext = load i64, ptr %tsp, align 8
  %extnz = icmp ne i64 %ext, 0
  %cr = call i32 @universe_bignum_cmp_n(ptr %t, ptr %n, i64 %s)
  %crge = icmp sge i32 %cr, 0
  %needsub = or i1 %extnz, %crge
  br i1 %needsub, label %dosub, label %docopy
dosub:
  %bo = call i64 @universe_bignum_sub_n(ptr %r, ptr %t, ptr %n, i64 %s)
  ret void
docopy:
  %cbytes = shl i64 %s, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %t, i64 %cbytes, i1 false)
  ret void
}

; ---------------------------------------------------------------- modexp

; r = base^exp mod n. n must be ODD with n[s-1] != 0. Returns 0, 8, or 2.
define i32 @universe_bignum_modexp(ptr %r, ptr %base, i64 %basel, ptr %exp, i64 %expl, ptr %n, i64 %s) local_unnamed_addr #1 {
entry:
  %n0 = load i64, ptr %n, align 8
  %odd = and i64 %n0, 1
  %isodd = icmp eq i64 %odd, 1
  br i1 %isodd, label %chkn, label %inval
inval:
  ret i32 8
chkn:
  %nlen = call i64 @universe_bignum_normalize_len(ptr %n, i64 %s)
  %nz = icmp eq i64 %nlen, 0
  br i1 %nz, label %inval, label %alloc
alloc:
  ; scratch: rr(s) onep(s) basem(s) res(s) basered(s) mts(s+2) = 6s+2
  %carve = mul i64 %s, 6
  %carve2 = add i64 %carve, 2
  %buf = call ptr @calloc(i64 %carve2, i64 8)
  %bufnull = icmp eq ptr %buf, null
  br i1 %bufnull, label %oom, label %setup
oom:
  ret i32 2
setup:
  %rr = getelementptr inbounds i64, ptr %buf, i64 0
  %onep = getelementptr inbounds i64, ptr %buf, i64 %s
  %s2 = mul i64 %s, 2
  %basem = getelementptr inbounds i64, ptr %buf, i64 %s2
  %s3 = mul i64 %s, 3
  %res = getelementptr inbounds i64, ptr %buf, i64 %s3
  %s4 = mul i64 %s, 4
  %basered = getelementptr inbounds i64, ptr %buf, i64 %s4
  %s5 = mul i64 %s, 5
  %mts = getelementptr inbounds i64, ptr %buf, i64 %s5
  %n0inv = call i64 @universe_bignum_mont_n0inv(ptr %n)
  call void @universe_bignum_mont_rr(ptr %rr, ptr %n, i64 %s)
  store i64 1, ptr %onep, align 8
  ; basered = base mod n
  %mc = call i32 @universe_bignum_mod(ptr %basered, ptr %base, i64 %basel, ptr %n, i64 %nlen)
  ; basem = toMont(basered) = montmul(basered, rr)
  call void @universe_bignum_montmul(ptr %basem, ptr %basered, ptr %rr, ptr %n, i64 %s, i64 %n0inv, ptr %mts)
  ; res = toMont(1) = montmul(onep, rr)
  call void @universe_bignum_montmul(ptr %res, ptr %onep, ptr %rr, ptr %n, i64 %s, i64 %n0inv, ptr %mts)
  %ebits = call i64 @universe_bignum_bit_length_n(ptr %exp, i64 %expl)
  %k0 = add i64 %ebits, -1
  br label %kloop
kloop:
  %k = phi i64 [ %k0, %setup ], [ %knext, %kstep ]
  %kvalid = icmp sge i64 %k, 0
  br i1 %kvalid, label %kbody, label %convert
kbody:
  ; res = montmul(res,res)  (square)
  call void @universe_bignum_montmul(ptr %res, ptr %res, ptr %res, ptr %n, i64 %s, i64 %n0inv, ptr %mts)
  %kdiv = lshr i64 %k, 6
  %kmod = and i64 %k, 63
  %elp = getelementptr inbounds i64, ptr %exp, i64 %kdiv
  %elv = load i64, ptr %elp, align 8
  %bitsh = lshr i64 %elv, %kmod
  %bit = and i64 %bitsh, 1
  %set = icmp eq i64 %bit, 1
  br i1 %set, label %kmul, label %kstep
kmul:
  call void @universe_bignum_montmul(ptr %res, ptr %res, ptr %basem, ptr %n, i64 %s, i64 %n0inv, ptr %mts)
  br label %kstep
kstep:
  %knext = add i64 %k, -1
  br label %kloop
convert:
  ; r = fromMont(res) = montmul(res, onep)
  call void @universe_bignum_montmul(ptr %r, ptr %res, ptr %onep, ptr %n, i64 %s, i64 %n0inv, ptr %mts)
  call void @free(ptr %buf)
  ret i32 0
}

; ---------------------------------------------------------------- gcd / modinv

; g = gcd(a,b) (magnitudes, s limbs each). g has s limbs.
define i32 @universe_bignum_gcd(ptr %g, ptr %a, ptr %b, i64 %s) local_unnamed_addr #1 {
entry:
  %carve = mul i64 %s, 4
  %buf = call ptr @calloc(i64 %carve, i64 8)
  %bufnull = icmp eq ptr %buf, null
  br i1 %bufnull, label %oom, label %init
oom:
  ret i32 2
init:
  %r0 = getelementptr inbounds i64, ptr %buf, i64 0
  %r1 = getelementptr inbounds i64, ptr %buf, i64 %s
  %s2 = mul i64 %s, 2
  %r2 = getelementptr inbounds i64, ptr %buf, i64 %s2
  %s3 = mul i64 %s, 3
  %q = getelementptr inbounds i64, ptr %buf, i64 %s3
  %sbytes = shl i64 %s, 3
  call void @llvm.memcpy.p0.p0.i64(ptr %r0, ptr %a, i64 %sbytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %r1, ptr %b, i64 %sbytes, i1 false)
  br label %loop
loop:
  %r1len = call i64 @universe_bignum_normalize_len(ptr %r1, i64 %s)
  %r1z = icmp eq i64 %r1len, 0
  br i1 %r1z, label %fin, label %step
step:
  %r0len = call i64 @universe_bignum_normalize_len(ptr %r0, i64 %s)
  call void @llvm.memset.p0.i64(ptr %r2, i8 0, i64 %sbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %q, i8 0, i64 %sbytes, i1 false)
  %dc = call i32 @universe_bignum_divmod(ptr %q, ptr %r2, ptr %r0, i64 %r0len, ptr %r1, i64 %r1len)
  call void @llvm.memcpy.p0.p0.i64(ptr %r0, ptr %r1, i64 %sbytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %r1, ptr %r2, i64 %sbytes, i1 false)
  br label %loop
fin:
  call void @llvm.memcpy.p0.p0.i64(ptr %g, ptr %r0, i64 %sbytes, i1 false)
  call void @free(ptr %buf)
  ret i32 0
}

; r = a^-1 mod n (s limbs). Returns 0 OK, 5 NOT_FOUND (gcd != 1), 2 OOM, 8.
define i32 @universe_bignum_modinv(ptr %r, ptr %a, ptr %n, i64 %s) local_unnamed_addr #1 {
entry:
  %nlen = call i64 @universe_bignum_normalize_len(ptr %n, i64 %s)
  %nz = icmp eq i64 %nlen, 0
  br i1 %nz, label %inval, label %alloc
inval:
  ret i32 8
alloc:
  ; r0 r1 r2 t0 t1 t2 q qt1 dtmp (s each) + mtmp (2s) = 11s ; alloc 12s
  %carve = mul i64 %s, 12
  %buf = call ptr @calloc(i64 %carve, i64 8)
  %bufnull = icmp eq ptr %buf, null
  br i1 %bufnull, label %oom, label %init
oom:
  ret i32 2
init:
  %sbytes = shl i64 %s, 3
  %r0 = getelementptr inbounds i64, ptr %buf, i64 0
  %r1 = getelementptr inbounds i64, ptr %buf, i64 %s
  %o2 = mul i64 %s, 2
  %r2 = getelementptr inbounds i64, ptr %buf, i64 %o2
  %o3 = mul i64 %s, 3
  %t0 = getelementptr inbounds i64, ptr %buf, i64 %o3
  %o4 = mul i64 %s, 4
  %t1 = getelementptr inbounds i64, ptr %buf, i64 %o4
  %o5 = mul i64 %s, 5
  %t2 = getelementptr inbounds i64, ptr %buf, i64 %o5
  %o6 = mul i64 %s, 6
  %q = getelementptr inbounds i64, ptr %buf, i64 %o6
  %o7 = mul i64 %s, 7
  %qt1 = getelementptr inbounds i64, ptr %buf, i64 %o7
  %o8 = mul i64 %s, 8
  %dtmp = getelementptr inbounds i64, ptr %buf, i64 %o8
  %o9 = mul i64 %s, 9
  %mtmp = getelementptr inbounds i64, ptr %buf, i64 %o9
  ; r0 = n
  call void @llvm.memcpy.p0.p0.i64(ptr %r0, ptr %n, i64 %sbytes, i1 false)
  ; r1 = a mod n
  %mc = call i32 @universe_bignum_mod(ptr %r1, ptr %a, i64 %s, ptr %n, i64 %nlen)
  ; t0 = 0 (already), t1 = 1
  store i64 1, ptr %t1, align 8
  br label %loop
loop:
  %r1len = call i64 @universe_bignum_normalize_len(ptr %r1, i64 %s)
  %r1z = icmp eq i64 %r1len, 0
  br i1 %r1z, label %fin, label %step
step:
  %r0len = call i64 @universe_bignum_normalize_len(ptr %r0, i64 %s)
  call void @llvm.memset.p0.i64(ptr %r2, i8 0, i64 %sbytes, i1 false)
  call void @llvm.memset.p0.i64(ptr %q, i8 0, i64 %sbytes, i1 false)
  %dc = call i32 @universe_bignum_divmod(ptr %q, ptr %r2, ptr %r0, i64 %r0len, ptr %r1, i64 %r1len)
  ; qt1 = (q * t1) mod n
  %qlen = call i64 @universe_bignum_normalize_len(ptr %q, i64 %s)
  call void @llvm.memset.p0.i64(ptr %qt1, i8 0, i64 %sbytes, i1 false)
  %qz = icmp eq i64 %qlen, 0
  br i1 %qz, label %modsub, label %domul
domul:
  call void @universe_bignum_mul(ptr %mtmp, ptr %q, i64 %qlen, ptr %t1, i64 %s)
  %mlen = add i64 %qlen, %s
  %mmc = call i32 @universe_bignum_mod(ptr %qt1, ptr %mtmp, i64 %mlen, ptr %n, i64 %nlen)
  br label %modsub
modsub:
  ; t2 = (t0 - qt1) mod n
  %cmp = call i32 @universe_bignum_cmp_n(ptr %t0, ptr %qt1, i64 %s)
  %ge = icmp sge i32 %cmp, 0
  br i1 %ge, label %subdirect, label %subwrap
subdirect:
  %bo1 = call i64 @universe_bignum_sub_n(ptr %t2, ptr %t0, ptr %qt1, i64 %s)
  br label %rotate
subwrap:
  %bo2 = call i64 @universe_bignum_sub_n(ptr %dtmp, ptr %n, ptr %qt1, i64 %s)
  %cy = call i64 @universe_bignum_add_n(ptr %t2, ptr %t0, ptr %dtmp, i64 %s)
  %cynz = icmp ne i64 %cy, 0
  %cmp2 = call i32 @universe_bignum_cmp_n(ptr %t2, ptr %n, i64 %s)
  %ge2 = icmp sge i32 %cmp2, 0
  %needsub = or i1 %cynz, %ge2
  br i1 %needsub, label %fixup, label %rotate
fixup:
  %bo3 = call i64 @universe_bignum_sub_n(ptr %t2, ptr %t2, ptr %n, i64 %s)
  br label %rotate
rotate:
  call void @llvm.memcpy.p0.p0.i64(ptr %r0, ptr %r1, i64 %sbytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %r1, ptr %r2, i64 %sbytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %t0, ptr %t1, i64 %sbytes, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %t1, ptr %t2, i64 %sbytes, i1 false)
  br label %loop
fin:
  ; gcd = r0; require r0 == 1
  %glen = call i64 @universe_bignum_normalize_len(ptr %r0, i64 %s)
  %isone.len = icmp eq i64 %glen, 1
  %g0 = load i64, ptr %r0, align 8
  %isone.val = icmp eq i64 %g0, 1
  %isone = and i1 %isone.len, %isone.val
  br i1 %isone, label %success, label %notfound
notfound:
  call void @free(ptr %buf)
  ret i32 5
success:
  call void @llvm.memcpy.p0.p0.i64(ptr %r, ptr %t0, i64 %sbytes, i1 false)
  call void @free(ptr %buf)
  ret i32 0
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
