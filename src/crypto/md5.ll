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

; MD5 (RFC 1321). Streaming context + one-shot. Digest = 16 bytes.
; NOTE: MD5 is cryptographically broken (collisions); provided for legacy
; interop/checksums only, not for security. See HARDENING-TODO.
;
; DESIGN:
;   * CTX layout (caller-supplied, single allocation):
;       off  0: 4 x i32 state words        (16 B)
;       off 16: i64   total byte length     (8 B)   fill = (total & 63)
;       off 24: 64 x i8 partial buffer      (64 B)
;     => UNIVERSE_CRYPTO_MD5_CTX_SIZE = 88 bytes, align 8.
;   * ENDIANNESS: MD5 is little-endian; our targets are all little-endian, so
;     block words are loaded and digest words stored DIRECTLY (no bswap), and
;     the trailing 64-bit length is appended little-endian (direct i64 store).
;   * HOT LEAF @md5_compress: internal alwaysinline; A..D are loop-carried SSA.
;     The 64-round loop has a constant trip count so -O3 unrolls it; per-round
;     F, message index g, K[i] and shift s[i] are chosen by selects/tables on
;     the loop index and fold to constants after unroll. Rotations via
;     llvm.fshl.i32(x,x,s) -> single rol. All adds are plain wrapping i32
;     (NO nuw/nsw — a no-wrap flag would be poison, Hazard #1).
;
; HARDENING-TODO: fast, not constant-time; no context zeroization on final.
;   MD5 must not be used where collision/preimage resistance is required.
;
; API (C ABI, nounwind):
;   void universe_crypto_md5_init(ptr ctx)
;   void universe_crypto_md5_update(ptr ctx, ptr data, i64 len)
;   void universe_crypto_md5_final(ptr ctx, ptr out16)
;   void universe_crypto_md5_hash(ptr data, i64 len, ptr out16)

declare i32 @llvm.fshl.i32(i32, i32, i32)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

@md5.K = private unnamed_addr constant [64 x i32]
[ i32 -680876936, i32 -389564586, i32 606105819, i32 -1044525330, i32 -176418897, i32 1200080426, i32 -1473231341, i32 -45705983, i32 1770035416, i32 -1958414417, i32 -42063, i32 -1990404162, i32 1804603682, i32 -40341101, i32 -1502002290, i32 1236535329, i32 -165796510, i32 -1069501632, i32 643717713, i32 -373897302, i32 -701558691, i32 38016083, i32 -660478335, i32 -405537848, i32 568446438, i32 -1019803690, i32 -187363961, i32 1163531501, i32 -1444681467, i32 -51403784, i32 1735328473, i32 -1926607734, i32 -378558, i32 -2022574463, i32 1839030562, i32 -35309556, i32 -1530992060, i32 1272893353, i32 -155497632, i32 -1094730640, i32 681279174, i32 -358537222, i32 -722521979, i32 76029189, i32 -640364487, i32 -421815835, i32 530742520, i32 -995338651, i32 -198630844, i32 1126891415, i32 -1416354905, i32 -57434055, i32 1700485571, i32 -1894986606, i32 -1051523, i32 -2054922799, i32 1873313359, i32 -30611744, i32 -1560198380, i32 1309151649, i32 -145523070, i32 -1120210379, i32 718787259, i32 -343485551 ], align 16

@md5.s = private unnamed_addr constant [64 x i32]
[ i32 7, i32 12, i32 17, i32 22, i32 7, i32 12, i32 17, i32 22, i32 7, i32 12, i32 17, i32 22, i32 7, i32 12, i32 17, i32 22, i32 5, i32 9, i32 14, i32 20, i32 5, i32 9, i32 14, i32 20, i32 5, i32 9, i32 14, i32 20, i32 5, i32 9, i32 14, i32 20, i32 4, i32 11, i32 16, i32 23, i32 4, i32 11, i32 16, i32 23, i32 4, i32 11, i32 16, i32 23, i32 4, i32 11, i32 16, i32 23, i32 6, i32 10, i32 15, i32 21, i32 6, i32 10, i32 15, i32 21, i32 6, i32 10, i32 15, i32 21, i32 6, i32 10, i32 15, i32 21 ], align 16

; ------------------------------------------------------------------- compress
define internal void @md5_compress(ptr noalias %st, ptr noalias %blk) #0 {
entry:
  %sp0 = getelementptr inbounds nuw i32, ptr %st, i64 0
  %A0 = load i32, ptr %sp0, align 4
  %sp1 = getelementptr inbounds nuw i32, ptr %st, i64 1
  %B0 = load i32, ptr %sp1, align 4
  %sp2 = getelementptr inbounds nuw i32, ptr %st, i64 2
  %C0 = load i32, ptr %sp2, align 4
  %sp3 = getelementptr inbounds nuw i32, ptr %st, i64 3
  %D0 = load i32, ptr %sp3, align 4
  br label %r.head

r.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %r.body ]
  %A = phi i32 [ %A0, %entry ], [ %A.n, %r.body ]
  %B = phi i32 [ %B0, %entry ], [ %B.n, %r.body ]
  %C = phi i32 [ %C0, %entry ], [ %C.n, %r.body ]
  %D = phi i32 [ %D0, %entry ], [ %D.n, %r.body ]
  %ic = icmp ult i64 %i, 64
  br i1 %ic, label %r.body, label %comp.post

r.body:
  %ph1 = icmp ult i64 %i, 16
  %ph2 = icmp ult i64 %i, 32
  %ph3 = icmp ult i64 %i, 48
  ; F candidates
  %f0_cxd = xor i32 %C, %D
  %f0_bm = and i32 %B, %f0_cxd
  %F0 = xor i32 %D, %f0_bm                        ; (B&C)|(~B&D)
  %f1_bxc = xor i32 %B, %C
  %f1_dm = and i32 %D, %f1_bxc
  %F1 = xor i32 %C, %f1_dm                        ; (D&B)|(~D&C)
  %f2a = xor i32 %B, %C
  %F2 = xor i32 %f2a, %D                          ; B^C^D
  %d_not = xor i32 %D, -1
  %f3or = or i32 %B, %d_not
  %F3 = xor i32 %C, %f3or                         ; C^(B|~D)
  %F_s2 = select i1 %ph3, i32 %F2, i32 %F3
  %F_s1 = select i1 %ph2, i32 %F1, i32 %F_s2
  %F = select i1 %ph1, i32 %F0, i32 %F_s1
  ; message word index g
  %g0 = and i64 %i, 15
  %i5 = mul nuw nsw i64 %i, 5
  %i5p1 = add nuw nsw i64 %i5, 1
  %g1 = and i64 %i5p1, 15
  %i3 = mul nuw nsw i64 %i, 3
  %i3p5 = add nuw nsw i64 %i3, 5
  %g2 = and i64 %i3p5, 15
  %i7 = mul nuw nsw i64 %i, 7
  %g3 = and i64 %i7, 15
  %g_s2 = select i1 %ph3, i64 %g2, i64 %g3
  %g_s1 = select i1 %ph2, i64 %g1, i64 %g_s2
  %g = select i1 %ph1, i64 %g0, i64 %g_s1
  ; load M[g] (little-endian direct)
  %mp = getelementptr inbounds nuw i32, ptr %blk, i64 %g
  %m = load i32, ptr %mp, align 1
  ; K[i], s[i]
  %kp = getelementptr inbounds nuw [64 x i32], ptr @md5.K, i64 0, i64 %i
  %k = load i32, ptr %kp, align 4
  %shp = getelementptr inbounds nuw [64 x i32], ptr @md5.s, i64 0, i64 %i
  %sh = load i32, ptr %shp, align 4
  ; tmp = A + F + K + M ; B = B + rol(tmp, s)
  %t0 = add i32 %A, %F
  %t1 = add i32 %t0, %k
  %t2 = add i32 %t1, %m
  %rot = call i32 @llvm.fshl.i32(i32 %t2, i32 %t2, i32 %sh)
  %B.n = add i32 %B, %rot
  %A.n = or i32 %D, 0
  %D.n = or i32 %C, 0
  %C.n = or i32 %B, 0
  %i.n = add nuw nsw i64 %i, 1
  br label %r.head

comp.post:
  %nA = add i32 %A, %A0
  store i32 %nA, ptr %sp0, align 4
  %nB = add i32 %B, %B0
  store i32 %nB, ptr %sp1, align 4
  %nC = add i32 %C, %C0
  store i32 %nC, ptr %sp2, align 4
  %nD = add i32 %D, %D0
  store i32 %nD, ptr %sp3, align 4
  ret void
}

; ----------------------------------------------------------------------- init
define void @universe_crypto_md5_init(ptr %ctx) local_unnamed_addr #1 {
entry:
  %p0 = getelementptr inbounds nuw i32, ptr %ctx, i64 0
  store i32 1732584193, ptr %p0, align 4                      ; 0x67452301
  %p1 = getelementptr inbounds nuw i32, ptr %ctx, i64 1
  store i32 -271733879, ptr %p1, align 4                      ; 0xefcdab89
  %p2 = getelementptr inbounds nuw i32, ptr %ctx, i64 2
  store i32 -1732584194, ptr %p2, align 4                     ; 0x98badcfe
  %p3 = getelementptr inbounds nuw i32, ptr %ctx, i64 3
  store i32 271733878, ptr %p3, align 4                       ; 0x10325476
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  store i64 0, ptr %lp, align 8
  ret void
}

; --------------------------------------------------------------------- update
define void @universe_crypto_md5_update(ptr %ctx, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret, label %go

go:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  %total = load i64, ptr %lp, align 8
  %buffered = and i64 %total, 63
  %total.new = add i64 %total, %len
  store i64 %total.new, ptr %lp, align 8
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %pos.nz = icmp ne i64 %buffered, 0
  br i1 %pos.nz, label %have_partial, label %bulk_entry

have_partial:
  %need = sub nuw nsw i64 64, %buffered
  %short = icmp ult i64 %len, %need
  br i1 %short, label %copy_short, label %complete_block

copy_short:
  %dstp = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp, ptr %data, i64 %len, i1 false)
  br label %ret

complete_block:
  %dstp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp2, ptr %data, i64 %need, i1 false)
  call void @md5_compress(ptr %ctx, ptr %buf)
  %data2 = getelementptr inbounds nuw i8, ptr %data, i64 %need
  %rem2 = sub nuw i64 %len, %need
  br label %bulk_loop

bulk_entry:
  br label %bulk_loop

bulk_loop:
  %dptr = phi ptr [ %data2, %complete_block ], [ %data, %bulk_entry ], [ %dptr.n, %bulk_body ]
  %rem = phi i64 [ %rem2, %complete_block ], [ %len, %bulk_entry ], [ %rem.n, %bulk_body ]
  %big = icmp uge i64 %rem, 64
  br i1 %big, label %bulk_body, label %tail

bulk_body:
  call void @md5_compress(ptr %ctx, ptr %dptr)
  %dptr.n = getelementptr inbounds nuw i8, ptr %dptr, i64 64
  %rem.n = sub nuw i64 %rem, 64
  br label %bulk_loop

tail:
  %tz = icmp eq i64 %rem, 0
  br i1 %tz, label %ret, label %copy_tail

copy_tail:
  call void @llvm.memcpy.p0.p0.i64(ptr %buf, ptr %dptr, i64 %rem, i1 false)
  br label %ret

ret:
  ret void
}

; ---------------------------------------------------------------------- final
define void @universe_crypto_md5_final(ptr %ctx, ptr %out) local_unnamed_addr #1 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 16
  %total = load i64, ptr %lp, align 8
  %pos = and i64 %total, 63
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %p80 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  store i8 -128, ptr %p80, align 1
  %pos1 = add nuw nsw i64 %pos, 1
  ; little-endian 64-bit bit length
  %bits = shl i64 %total, 3
  %lenslot = getelementptr inbounds nuw i8, ptr %buf, i64 56
  %twoblk = icmp ugt i64 %pos1, 56
  br i1 %twoblk, label %two, label %one

one:
  %zlen = sub nuw nsw i64 56, %pos1
  %zp = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp, i8 0, i64 %zlen, i1 false)
  store i64 %bits, ptr %lenslot, align 1
  call void @md5_compress(ptr %ctx, ptr %buf)
  br label %emit

two:
  %zlen2 = sub nuw nsw i64 64, %pos1
  %zp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp2, i8 0, i64 %zlen2, i1 false)
  call void @md5_compress(ptr %ctx, ptr %buf)
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 56, i1 false)
  store i64 %bits, ptr %lenslot, align 1
  call void @md5_compress(ptr %ctx, ptr %buf)
  br label %emit

emit:                                             ; write little-endian digest
  br label %out.head

out.head:
  %oi = phi i64 [ 0, %emit ], [ %oi.n, %out.body ]
  %oc = icmp ult i64 %oi, 4
  br i1 %oc, label %out.body, label %done

out.body:
  %swp = getelementptr inbounds nuw i32, ptr %ctx, i64 %oi
  %sw = load i32, ptr %swp, align 4
  %op = getelementptr inbounds nuw i32, ptr %out, i64 %oi
  store i32 %sw, ptr %op, align 1
  %oi.n = add nuw nsw i64 %oi, 1
  br label %out.head

done:
  ret void
}

; ----------------------------------------------------------------------- hash
define void @universe_crypto_md5_hash(ptr %data, i64 %len, ptr %out) local_unnamed_addr #1 {
entry:
  %ctx = alloca [88 x i8], align 8
  call void @universe_crypto_md5_init(ptr %ctx)
  call void @universe_crypto_md5_update(ptr %ctx, ptr %data, i64 %len)
  call void @universe_crypto_md5_final(ptr %ctx, ptr %out)
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree }
