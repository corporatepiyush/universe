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

; RSA PKCS#1 v1.5 (RFC 8017): RSASSA sign/verify (SHA-256) and RSAES
; encrypt/decrypt. The modular exponentiation is bignum_modexp (Montgomery
; square-and-multiply); this module is the padding + byte-order (I2OSP/OS2IP)
; layer around it.
;
; DESIGN:
;   KEY SHAPE: n, e, d are caller-provided bignum limb arrays of s u64 limbs
;   (little-endian), modulus byte length k = 8*s. RSA is defined on big-endian
;   octet strings (I2OSP/OS2IP); bignum limbs are little-endian, so conversion
;   is a whole-buffer BYTE REVERSAL (@rev_bytes): reversing k big-endian bytes
;   yields the little-endian limb image and vice-versa. No per-limb swaps.
;
;   RSASSA-PKCS1-v1_5 (SHA-256): EM = 0x00 01 PS(0xFF..) 00 || DigestInfo || H,
;   where DigestInfo is the fixed 19-byte SHA-256 AlgorithmIdentifier prefix.
;   sign: s = EM^d mod n; verify: rebuild EM from the message and compare the
;   recovered EM^e mod n byte-for-byte (encode-then-compare — no parsing of
;   attacker-controlled structure).
;
;   RSAES-PKCS1-v1_5: EM = 0x00 02 PS(nonzero) 00 || M. Decrypt scans for the
;   0x00 separator after >= 8 padding bytes.
;
; HARDENING-TODO: NOT constant-time (modexp ladder branches on secret bits; the
;   decrypt padding scan is not constant-time — Bleichenbacher-relevant). The
;   encrypt PS bytes are a DETERMINISTIC nonzero filler, NOT CSPRNG output;
;   replace with a real CSPRNG in the hardening phase. No blinding, no scratch
;   zeroization.
;
; API (C ABI, nounwind). k = 8*s bytes. sig/ct/out are k big-endian bytes.
;   void universe_crypto_rsa_pkcs1_sign_sha256(ptr msg,i64 mlen,ptr n,ptr d,i64 s,ptr sig)
;   i32  universe_crypto_rsa_pkcs1_verify_sha256(ptr msg,i64 mlen,ptr n,ptr e,i64 s,ptr sig)  ; 0 ok / 1 bad
;   i32  universe_crypto_rsa_encrypt_pkcs1(ptr msg,i64 mlen,ptr n,ptr e,i64 s,ptr out)        ; 0 ok / 8 too long
;   i64  universe_crypto_rsa_decrypt_pkcs1(ptr ct,ptr n,ptr d,i64 s,ptr out)                  ; mlen or -1

declare i32 @universe_bignum_modexp(ptr, ptr, i64, ptr, i64, ptr, i64)
declare void @universe_crypto_sha256_hash(ptr, i64, ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; SHA-256 DigestInfo prefix (19 bytes)
@RSA_DI256 = private unnamed_addr constant [19 x i8] c"\30\31\30\0d\06\09\60\86\48\01\65\03\04\02\01\05\00\04\20"

; dst[i] = src[n-1-i]  (dst and src must not alias)
define internal void @rev_bytes(ptr %dst, ptr %src, i64 %n) #0 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %nm1 = sub i64 %n, 1
  %j = sub i64 %nm1, %i
  %sp = getelementptr inbounds i8, ptr %src, i64 %j
  %v = load i8, ptr %sp, align 1
  %dp = getelementptr inbounds i8, ptr %dst, i64 %i
  store i8 %v, ptr %dp, align 1
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; build signature EM (k big-endian bytes) from a 32-byte SHA-256 hash
define internal void @rsa_em_sign(ptr %em, i64 %k, ptr %h32) #0 {
entry:
  ; pslen = k - 51 - 3
  %pslen = sub i64 %k, 54
  store i8 0, ptr %em, align 1
  %e1 = getelementptr inbounds i8, ptr %em, i64 1
  store i8 1, ptr %e1, align 1
  %ps = getelementptr inbounds i8, ptr %em, i64 2
  call void @llvm.memset.p0.i64(ptr %ps, i8 -1, i64 %pslen, i1 false)
  %sepidx = add i64 2, %pslen
  %sep = getelementptr inbounds i8, ptr %em, i64 %sepidx
  store i8 0, ptr %sep, align 1
  %diidx = add i64 %sepidx, 1
  %di = getelementptr inbounds i8, ptr %em, i64 %diidx
  call void @llvm.memcpy.p0.p0.i64(ptr %di, ptr @RSA_DI256, i64 19, i1 false)
  %hidx = add i64 %diidx, 19
  %hp = getelementptr inbounds i8, ptr %em, i64 %hidx
  call void @llvm.memcpy.p0.p0.i64(ptr %hp, ptr %h32, i64 32, i1 false)
  ret void
}

define void @universe_crypto_rsa_pkcs1_sign_sha256(ptr %msg, i64 %mlen, ptr %n, ptr %d, i64 %s, ptr %sig) local_unnamed_addr #1 {
entry:
  %h = alloca [32 x i8], align 8
  %em = alloca [512 x i8], align 8
  %le = alloca [512 x i8], align 8
  %rb = alloca [512 x i8], align 8
  %k = shl i64 %s, 3
  call void @universe_crypto_sha256_hash(ptr %msg, i64 %mlen, ptr %h)
  call void @rsa_em_sign(ptr %em, i64 %k, ptr %h)
  call void @rev_bytes(ptr %le, ptr %em, i64 %k)
  %rc = call i32 @universe_bignum_modexp(ptr %rb, ptr %le, i64 %s, ptr %d, i64 %s, ptr %n, i64 %s)
  call void @rev_bytes(ptr %sig, ptr %rb, i64 %k)
  ret void
}

define i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr %msg, i64 %mlen, ptr %n, ptr %e, i64 %s, ptr %sig) local_unnamed_addr #1 {
entry:
  %h = alloca [32 x i8], align 8
  %em = alloca [512 x i8], align 8
  %emx = alloca [512 x i8], align 8
  %le = alloca [512 x i8], align 8
  %rb = alloca [512 x i8], align 8
  %k = shl i64 %s, 3
  ; recovered EM = sig^e mod n
  call void @rev_bytes(ptr %le, ptr %sig, i64 %k)
  %rc = call i32 @universe_bignum_modexp(ptr %rb, ptr %le, i64 %s, ptr %e, i64 %s, ptr %n, i64 %s)
  call void @rev_bytes(ptr %emx, ptr %rb, i64 %k)
  ; expected EM
  call void @universe_crypto_sha256_hash(ptr %msg, i64 %mlen, ptr %h)
  call void @rsa_em_sign(ptr %em, i64 %k, ptr %h)
  ; compare k bytes
  br label %cmp
cmp:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %ap = getelementptr inbounds i8, ptr %em, i64 %i
  %bp = getelementptr inbounds i8, ptr %emx, i64 %i
  %av = load i8, ptr %ap, align 1
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %bad, label %cont
cont:
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %k
  br i1 %more, label %cmp, label %ok
ok:
  ret i32 0
bad:
  ret i32 1
}

define i32 @universe_crypto_rsa_encrypt_pkcs1(ptr %msg, i64 %mlen, ptr %n, ptr %e, i64 %s, ptr %out) local_unnamed_addr #1 {
entry:
  %em = alloca [512 x i8], align 8
  %le = alloca [512 x i8], align 8
  %rb = alloca [512 x i8], align 8
  %k = shl i64 %s, 3
  ; require mlen <= k - 11
  %maxm = sub i64 %k, 11
  %toolong = icmp ugt i64 %mlen, %maxm
  br i1 %toolong, label %err, label %build
err:
  ret i32 8
build:
  ; EM = 00 02 PS(nonzero) 00 M ; pslen = k - mlen - 3
  store i8 0, ptr %em, align 1
  %e1 = getelementptr inbounds i8, ptr %em, i64 1
  store i8 2, ptr %e1, align 1
  %mplus3 = add i64 %mlen, 3
  %pslen = sub i64 %k, %mplus3
  br label %psloop
psloop:
  %i = phi i64 [ 0, %build ], [ %in, %psloop ]
  %idx = add i64 2, %i
  %pp = getelementptr inbounds i8, ptr %em, i64 %idx
  ; deterministic nonzero filler (HARDENING-TODO: CSPRNG)
  %ib = trunc i64 %i to i8
  %nz = or i8 %ib, 1
  store i8 %nz, ptr %pp, align 1
  %in = add i64 %i, 1
  %more = icmp ult i64 %in, %pslen
  br i1 %more, label %psloop, label %psdone
psdone:
  %sepidx = add i64 2, %pslen
  %sep = getelementptr inbounds i8, ptr %em, i64 %sepidx
  store i8 0, ptr %sep, align 1
  %midx = add i64 %sepidx, 1
  %mp = getelementptr inbounds i8, ptr %em, i64 %midx
  call void @llvm.memcpy.p0.p0.i64(ptr %mp, ptr %msg, i64 %mlen, i1 false)
  call void @rev_bytes(ptr %le, ptr %em, i64 %k)
  %rc = call i32 @universe_bignum_modexp(ptr %rb, ptr %le, i64 %s, ptr %e, i64 %s, ptr %n, i64 %s)
  call void @rev_bytes(ptr %out, ptr %rb, i64 %k)
  ret i32 0
}

define i64 @universe_crypto_rsa_decrypt_pkcs1(ptr %ct, ptr %n, ptr %d, i64 %s, ptr %out) local_unnamed_addr #1 {
entry:
  %em = alloca [512 x i8], align 8
  %le = alloca [512 x i8], align 8
  %rb = alloca [512 x i8], align 8
  %k = shl i64 %s, 3
  call void @rev_bytes(ptr %le, ptr %ct, i64 %k)
  %rc = call i32 @universe_bignum_modexp(ptr %rb, ptr %le, i64 %s, ptr %d, i64 %s, ptr %n, i64 %s)
  call void @rev_bytes(ptr %em, ptr %rb, i64 %k)
  ; check 00 02 prefix
  %b0 = load i8, ptr %em, align 1
  %b0ok = icmp eq i8 %b0, 0
  %e1 = getelementptr inbounds i8, ptr %em, i64 1
  %b1 = load i8, ptr %e1, align 1
  %b1ok = icmp eq i8 %b1, 2
  %pfx = and i1 %b0ok, %b1ok
  br i1 %pfx, label %scan, label %bad
scan:
  ; find first 0x00 at index >= 2
  br label %sloop
sloop:
  %i = phi i64 [ 2, %scan ], [ %in, %scont ]
  %atend = icmp uge i64 %i, %k
  br i1 %atend, label %bad, label %sbody
sbody:
  %pp = getelementptr inbounds i8, ptr %em, i64 %i
  %bv = load i8, ptr %pp, align 1
  %is0 = icmp eq i8 %bv, 0
  br i1 %is0, label %found, label %scont
scont:
  %in = add i64 %i, 1
  br label %sloop
found:
  ; PS length = i - 2 must be >= 8
  %pslen = sub i64 %i, 2
  %pok = icmp uge i64 %pslen, 8
  br i1 %pok, label %emit, label %bad
emit:
  %midx = add i64 %i, 1
  %mp = getelementptr inbounds i8, ptr %em, i64 %midx
  %mlen = sub i64 %k, %midx
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %mp, i64 %mlen, i1 false)
  ret i64 %mlen
bad:
  ret i64 -1
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
