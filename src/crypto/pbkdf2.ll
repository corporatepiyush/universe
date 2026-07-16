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

; PBKDF2 (RFC 8018 / PKCS#5 v2.1) over HMAC.
;
;   DK = T_1 || T_2 || ... || T_l ,  l = ceil(dkLen / hLen)
;   T_i = U_1 XOR U_2 XOR ... XOR U_c
;   U_1 = PRF(P, S || INT32_BE(i)) ,  U_j = PRF(P, U_{j-1})
;   PRF = HMAC-<digest>.
;
; DESIGN:
;   * Key-setup amortization is THE optimization: HMAC's expensive part is
;     compressing the two padded key blocks (ipad/opad). We run HMAC-init ONCE
;     to build a "template" context whose inner/outer digest states already
;     hold those compressed blocks, then, for every PRF invocation, memcpy the
;     440-byte template into a working context and only feed the message +
;     finalize. Each PRF therefore costs ~2 compressions of the short input,
;     not a re-derivation of the padded key. This is why a 4096-iteration
;     PBKDF2 is ~4096*2 compressions, not ~4096*4.
;   * INT32_BE(i) is the 1-based block index, big-endian, appended to the salt
;     for U_1 only (via llvm.bswap.i32 into a 4-byte stack slot).
;   * The XOR accumulator T and the running U are both at most hLen (<=64)
;     bytes and live on the stack; the inner accumulate loop is a plain byte
;     XOR (hLen is 20/32/64 — the backend vectorizes it freely).
;   * Compute is separated from IO/layout: no allocation in the iteration
;     loop; the working context and scratch buffers are reused across blocks.
;
; HARDENING-TODO: fast, not constant-time. The iteration count is public but
;   the derived key is secret; add zeroization of T/U/working-ctx and a
;   constant-time review in the hardening phase.
;
; API (C ABI, nounwind):
;   ; generic (bring your own digest vtable + HMAC block/digest sizes)
;   void universe_crypto_pbkdf2(ptr initfp, ptr updfp, ptr finfp,
;                               i64 blocksize, i64 digestsize,
;                               ptr pass, i64 plen, ptr salt, i64 slen,
;                               i64 iters, ptr out, i64 dklen)
;   ; concrete convenience
;   void universe_crypto_pbkdf2_sha1  (ptr pass,i64 plen, ptr salt,i64 slen,
;                                      i64 iters, ptr out, i64 dklen)
;   void universe_crypto_pbkdf2_sha256(...)
;   void universe_crypto_pbkdf2_sha512(...)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.umin.i64(i64, i64)

declare void @universe_crypto_hmac_init(ptr, ptr, ptr, ptr, i64, i64, ptr, i64)
declare void @universe_crypto_hmac_update(ptr, ptr, i64)
declare void @universe_crypto_hmac_final(ptr, ptr)

declare void @universe_crypto_sha1_init(ptr)
declare void @universe_crypto_sha1_update(ptr, ptr, i64)
declare void @universe_crypto_sha1_final(ptr, ptr)
declare void @universe_crypto_sha256_init(ptr)
declare void @universe_crypto_sha256_update(ptr, ptr, i64)
declare void @universe_crypto_sha256_final(ptr, ptr)
declare void @universe_crypto_sha512_init(ptr)
declare void @universe_crypto_sha512_update(ptr, ptr, i64)
declare void @universe_crypto_sha512_final(ptr, ptr)

; --------------------------------------------------------------- generic core
define void @universe_crypto_pbkdf2(ptr %initfp, ptr %updfp, ptr %finfp, i64 %bs, i64 %ds, ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %dklen, 0
  br i1 %z, label %ret, label %setup

setup:
  %tmpl = alloca [440 x i8], align 16
  %hctx = alloca [440 x i8], align 16
  %U = alloca [64 x i8], align 16
  %T = alloca [64 x i8], align 16
  %ibuf = alloca [4 x i8], align 4
  call void @universe_crypto_hmac_init(ptr %tmpl, ptr %initfp, ptr %updfp, ptr %finfp, i64 %bs, i64 %ds, ptr %pass, i64 %plen)
  br label %blk.head

blk.head:                                         ; one output block T_i per pass
  %bi = phi i32 [ 1, %setup ], [ %bi.n, %blk.done ]
  %off = phi i64 [ 0, %setup ], [ %off.n, %blk.done ]
  %more = icmp ult i64 %off, %dklen
  br i1 %more, label %blk.body, label %ret

blk.body:
  ; U_1 = PRF(salt || INT32_BE(bi))
  call void @llvm.memcpy.p0.p0.i64(ptr %hctx, ptr %tmpl, i64 440, i1 false)
  call void @universe_crypto_hmac_update(ptr %hctx, ptr %salt, i64 %slen)
  %be = call i32 @llvm.bswap.i32(i32 %bi)
  store i32 %be, ptr %ibuf, align 4
  call void @universe_crypto_hmac_update(ptr %hctx, ptr %ibuf, i64 4)
  call void @universe_crypto_hmac_final(ptr %hctx, ptr %U)
  ; T = U_1
  call void @llvm.memcpy.p0.p0.i64(ptr %T, ptr %U, i64 %ds, i1 false)
  br label %it.head

it.head:                                          ; U_j and T ^= U_j, j=2..iters
  %j = phi i64 [ 1, %blk.body ], [ %j.n, %it.next ]
  %jc = icmp ult i64 %j, %iters
  br i1 %jc, label %it.body, label %emit

it.body:
  call void @llvm.memcpy.p0.p0.i64(ptr %hctx, ptr %tmpl, i64 440, i1 false)
  call void @universe_crypto_hmac_update(ptr %hctx, ptr %U, i64 %ds)
  call void @universe_crypto_hmac_final(ptr %hctx, ptr %U)
  br label %xor.head

xor.head:
  %k = phi i64 [ 0, %it.body ], [ %k.n, %xor.body ]
  %kc = icmp ult i64 %k, %ds
  br i1 %kc, label %xor.body, label %it.next

xor.body:
  %tp = getelementptr inbounds nuw i8, ptr %T, i64 %k
  %tv = load i8, ptr %tp, align 1
  %up = getelementptr inbounds nuw i8, ptr %U, i64 %k
  %uv = load i8, ptr %up, align 1
  %xv = xor i8 %tv, %uv
  store i8 %xv, ptr %tp, align 1
  %k.n = add nuw nsw i64 %k, 1
  br label %xor.head

it.next:
  %j.n = add nuw i64 %j, 1
  br label %it.head

emit:                                             ; copy min(hLen, remaining)
  %rem = sub nuw i64 %dklen, %off
  %n = call i64 @llvm.umin.i64(i64 %ds, i64 %rem)
  %outp = getelementptr inbounds nuw i8, ptr %out, i64 %off
  call void @llvm.memcpy.p0.p0.i64(ptr %outp, ptr %T, i64 %n, i1 false)
  br label %blk.done

blk.done:
  %off.n = add nuw i64 %off, %ds
  %bi.n = add i32 %bi, 1
  br label %blk.head

ret:
  ret void
}

; ---------------------------------------------------------- concrete wrappers
define void @universe_crypto_pbkdf2_sha1(ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_pbkdf2(ptr @universe_crypto_sha1_init, ptr @universe_crypto_sha1_update, ptr @universe_crypto_sha1_final, i64 64, i64 20, ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen)
  ret void
}

define void @universe_crypto_pbkdf2_sha256(ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_pbkdf2(ptr @universe_crypto_sha256_init, ptr @universe_crypto_sha256_update, ptr @universe_crypto_sha256_final, i64 64, i64 32, ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen)
  ret void
}

define void @universe_crypto_pbkdf2_sha512(ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_pbkdf2(ptr @universe_crypto_sha512_init, ptr @universe_crypto_sha512_update, ptr @universe_crypto_sha512_final, i64 128, i64 64, ptr %pass, i64 %plen, ptr %salt, i64 %slen, i64 %iters, ptr %out, i64 %dklen)
  ret void
}

attributes #1 = { nounwind willreturn norecurse nosync nofree }
