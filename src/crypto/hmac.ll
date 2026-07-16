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

; HMAC (RFC 2104 / FIPS 198-1), generic over any block-based digest.
;
;   HMAC(K, m) = H( (K' XOR opad) || H( (K' XOR ipad) || m ) )
;   K' = H(K) if |K| > blocksize else K, right-padded with zero bytes to
;        blocksize; ipad = 0x36 repeated, opad = 0x5c repeated.
;
; DESIGN:
;   * Digest-agnostic via a tiny value vtable stored in the HMAC context:
;     three function pointers (init/update/final, the uniform digest API
;     void(ptr) / void(ptr,ptr,i64) / void(ptr,ptr)) plus the digest's block
;     size and output size. One generic implementation drives SHA-1/256/512
;     (and any future block digest) with no per-algorithm branching.
;   * Streaming, allocation-free: the context embeds TWO digest contexts —
;     the inner one pre-fed with (K' XOR ipad), the outer pre-fed with
;     (K' XOR opad). update() feeds the inner; final() closes the inner into
;     a scratch digest, feeds it to the outer, and closes the outer into the
;     caller's buffer. Because the two padded blocks are compressed once at
;     init, key setup is not repeated per message — this is exactly what
;     PBKDF2 exploits by copying a post-init context template.
;   * CTX layout (caller-supplied, single block, NO malloc inside):
;       off   0: ptr  initfp
;       off   8: ptr  updatefp
;       off  16: ptr  finalfp
;       off  24: i64  block size   (64 for SHA-1/256, 128 for SHA-512)
;       off  32: i64  digest size  (20 / 32 / 64)
;       off  40: inner digest ctx  (<=200 B, sized for the largest digest)
;       off 240: outer digest ctx  (<=200 B)
;     => UNIVERSE_CRYPTO_HMAC_CTX_SIZE = 440 bytes, align 8. Digest contexts
;     used here: SHA-1 96, SHA-256 104, SHA-512 200 — all fit the 200-byte
;     reserve. Max block size is 128, max digest 64.
;   * Indirect vtable calls sit OUTSIDE the hot compression loop (one per
;     padded block / message chunk); the per-block cost is dominated by the
;     directly-called, fully-inlined digest compression. No hot-path penalty.
;
; HARDENING-TODO: fast, not constant-time. No secret-dependent branches, but
;   the context (holding the padded key material) is not zeroized on final;
;   add zeroization + a constant-time review in the hardening phase.
;
; API (C ABI, nounwind):
;   ; generic (bring your own digest vtable)
;   void universe_crypto_hmac_init(ptr ctx, ptr initfp, ptr updfp, ptr finfp,
;                                  i64 blocksize, i64 digestsize,
;                                  ptr key, i64 klen)
;   void universe_crypto_hmac_update(ptr ctx, ptr data, i64 len)
;   void universe_crypto_hmac_final(ptr ctx, ptr out)   ; writes digestsize B
;   ; concrete streaming inits (fill the vtable for a named digest)
;   void universe_crypto_hmac_sha1_init(ptr ctx, ptr key, i64 klen)
;   void universe_crypto_hmac_sha256_init(ptr ctx, ptr key, i64 klen)
;   void universe_crypto_hmac_sha512_init(ptr ctx, ptr key, i64 klen)
;   ; one-shot convenience
;   void universe_crypto_hmac_sha1(ptr key, i64 klen, ptr msg, i64 mlen, ptr out20)
;   void universe_crypto_hmac_sha256(ptr key, i64 klen, ptr msg, i64 mlen, ptr out32)
;   void universe_crypto_hmac_sha512(ptr key, i64 klen, ptr msg, i64 mlen, ptr out64)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; digest modules (linked; called directly by the concrete wrappers, and
; indirectly through the vtable by the generic core)
declare void @universe_crypto_sha1_init(ptr)
declare void @universe_crypto_sha1_update(ptr, ptr, i64)
declare void @universe_crypto_sha1_final(ptr, ptr)
declare void @universe_crypto_sha256_init(ptr)
declare void @universe_crypto_sha256_update(ptr, ptr, i64)
declare void @universe_crypto_sha256_final(ptr, ptr)
declare void @universe_crypto_sha512_init(ptr)
declare void @universe_crypto_sha512_update(ptr, ptr, i64)
declare void @universe_crypto_sha512_final(ptr, ptr)

; ------------------------------------------------------------------ hmac init
define void @universe_crypto_hmac_init(ptr %h, ptr %initfp, ptr %updfp, ptr %finfp, i64 %bs, i64 %ds, ptr %key, i64 %klen) local_unnamed_addr #1 {
entry:
  store ptr %initfp, ptr %h, align 8
  %pu = getelementptr inbounds nuw i8, ptr %h, i64 8
  store ptr %updfp, ptr %pu, align 8
  %pf = getelementptr inbounds nuw i8, ptr %h, i64 16
  store ptr %finfp, ptr %pf, align 8
  %pbs = getelementptr inbounds nuw i8, ptr %h, i64 24
  store i64 %bs, ptr %pbs, align 8
  %pds = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 %ds, ptr %pds, align 8
  %inner = getelementptr inbounds nuw i8, ptr %h, i64 40
  %outer = getelementptr inbounds nuw i8, ptr %h, i64 240

  ; key block K', padded with zeros to blocksize (max block = 128)
  %kb = alloca [128 x i8], align 16
  call void @llvm.memset.p0.i64(ptr %kb, i8 0, i64 128, i1 false)

  %long = icmp ugt i64 %klen, %bs
  br i1 %long, label %hashkey, label %copykey

hashkey:                                          ; K' = H(K)
  call void %initfp(ptr %inner)
  call void %updfp(ptr %inner, ptr %key, i64 %klen)
  call void %finfp(ptr %inner, ptr %kb)
  br label %pad

copykey:                                          ; K' = K (zero padded)
  call void @llvm.memcpy.p0.p0.i64(ptr %kb, ptr %key, i64 %klen, i1 false)
  br label %pad

pad:
  %ipad = alloca [128 x i8], align 16
  %opad = alloca [128 x i8], align 16
  br label %xor.head

xor.head:
  %i = phi i64 [ 0, %pad ], [ %i.n, %xor.body ]
  %c = icmp ult i64 %i, %bs
  br i1 %c, label %xor.body, label %feed

xor.body:
  %kp = getelementptr inbounds nuw i8, ptr %kb, i64 %i
  %kv = load i8, ptr %kp, align 1
  %iv = xor i8 %kv, 54                             ; 0x36 ipad
  %ov = xor i8 %kv, 92                             ; 0x5c opad
  %ip = getelementptr inbounds nuw i8, ptr %ipad, i64 %i
  store i8 %iv, ptr %ip, align 1
  %op = getelementptr inbounds nuw i8, ptr %opad, i64 %i
  store i8 %ov, ptr %op, align 1
  %i.n = add nuw nsw i64 %i, 1
  br label %xor.head

feed:                                             ; pre-compress the pad blocks
  call void %initfp(ptr %inner)
  call void %updfp(ptr %inner, ptr %ipad, i64 %bs)
  call void %initfp(ptr %outer)
  call void %updfp(ptr %outer, ptr %opad, i64 %bs)
  ret void
}

; ---------------------------------------------------------------- hmac update
define void @universe_crypto_hmac_update(ptr %h, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %pu = getelementptr inbounds nuw i8, ptr %h, i64 8
  %updfp = load ptr, ptr %pu, align 8
  %inner = getelementptr inbounds nuw i8, ptr %h, i64 40
  call void %updfp(ptr %inner, ptr %data, i64 %len)
  ret void
}

; ----------------------------------------------------------------- hmac final
define void @universe_crypto_hmac_final(ptr %h, ptr %out) local_unnamed_addr #1 {
entry:
  %pu = getelementptr inbounds nuw i8, ptr %h, i64 8
  %updfp = load ptr, ptr %pu, align 8
  %pf = getelementptr inbounds nuw i8, ptr %h, i64 16
  %finfp = load ptr, ptr %pf, align 8
  %pds = getelementptr inbounds nuw i8, ptr %h, i64 32
  %ds = load i64, ptr %pds, align 8
  %inner = getelementptr inbounds nuw i8, ptr %h, i64 40
  %outer = getelementptr inbounds nuw i8, ptr %h, i64 240
  %ihash = alloca [64 x i8], align 16
  call void %finfp(ptr %inner, ptr %ihash)         ; inner digest
  call void %updfp(ptr %outer, ptr %ihash, i64 %ds)
  call void %finfp(ptr %outer, ptr %out)           ; outer digest -> out
  ret void
}

; ----------------------------------------------- concrete streaming inits
define void @universe_crypto_hmac_sha1_init(ptr %h, ptr %key, i64 %klen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_hmac_init(ptr %h, ptr @universe_crypto_sha1_init, ptr @universe_crypto_sha1_update, ptr @universe_crypto_sha1_final, i64 64, i64 20, ptr %key, i64 %klen)
  ret void
}

define void @universe_crypto_hmac_sha256_init(ptr %h, ptr %key, i64 %klen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_hmac_init(ptr %h, ptr @universe_crypto_sha256_init, ptr @universe_crypto_sha256_update, ptr @universe_crypto_sha256_final, i64 64, i64 32, ptr %key, i64 %klen)
  ret void
}

define void @universe_crypto_hmac_sha512_init(ptr %h, ptr %key, i64 %klen) local_unnamed_addr #1 {
entry:
  call void @universe_crypto_hmac_init(ptr %h, ptr @universe_crypto_sha512_init, ptr @universe_crypto_sha512_update, ptr @universe_crypto_sha512_final, i64 128, i64 64, ptr %key, i64 %klen)
  ret void
}

; ---------------------------------------------------------- one-shot helpers
define void @universe_crypto_hmac_sha1(ptr %key, i64 %klen, ptr %msg, i64 %mlen, ptr %out) local_unnamed_addr #1 {
entry:
  %h = alloca [440 x i8], align 16
  call void @universe_crypto_hmac_sha1_init(ptr %h, ptr %key, i64 %klen)
  call void @universe_crypto_hmac_update(ptr %h, ptr %msg, i64 %mlen)
  call void @universe_crypto_hmac_final(ptr %h, ptr %out)
  ret void
}

define void @universe_crypto_hmac_sha256(ptr %key, i64 %klen, ptr %msg, i64 %mlen, ptr %out) local_unnamed_addr #1 {
entry:
  %h = alloca [440 x i8], align 16
  call void @universe_crypto_hmac_sha256_init(ptr %h, ptr %key, i64 %klen)
  call void @universe_crypto_hmac_update(ptr %h, ptr %msg, i64 %mlen)
  call void @universe_crypto_hmac_final(ptr %h, ptr %out)
  ret void
}

define void @universe_crypto_hmac_sha512(ptr %key, i64 %klen, ptr %msg, i64 %mlen, ptr %out) local_unnamed_addr #1 {
entry:
  %h = alloca [440 x i8], align 16
  call void @universe_crypto_hmac_sha512_init(ptr %h, ptr %key, i64 %klen)
  call void @universe_crypto_hmac_update(ptr %h, ptr %msg, i64 %mlen)
  call void @universe_crypto_hmac_final(ptr %h, ptr %out)
  ret void
}

attributes #1 = { nounwind willreturn norecurse nosync nofree }
