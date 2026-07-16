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

; SHA-256 (FIPS 180-4). Streaming context + one-shot. Digest = 32 bytes.
;
; DESIGN:
;   * CTX layout (caller-supplied, single allocation, NO malloc inside):
;       off  0: 8 x i32 state words  (32 B)   little-endian in memory, holding
;                                              the running big-endian hash words
;       off 32: i64   total byte length processed (8 B)
;       off 40: 64 x i8 partial-block buffer   (64 B)
;     => UNIVERSE_CRYPTO_SHA256_CTX_SIZE = 104 bytes, align 8. The current
;     fill of the buffer is (total & 63); no separate length field needed.
;   * HOT LEAF is @sha256_compress: internal alwaysinline, register-resident.
;     a..h are loop-carried SSA (phi) values so the 64 rounds keep all working
;     state in registers; the round loop has a constant trip count (64) so -O3
;     fully unrolls it. Message schedule W[0..63] lives in a stack array; the
;     backend reads W[t]/K[t] as ordinary array loads (not spills).
;   * ENDIANNESS: SHA-256 defines big-endian word loads; our targets are all
;     little-endian, so each 32-bit block word is loaded then byte-swapped with
;     llvm.bswap.i32, and the final digest words are byte-swapped on store.
;   * ROTATIONS via llvm.fshr.i32(x,x,n): a funnel-shift-right with equal
;     operands is exactly ROTR_n(x); it documents intent and lowers to a single
;     ror (arm64) / ror (x86) — verified spill-free/call-free (hot-path gate).
;   * Ch(e,f,g) = g ^ (e & (f ^ g))  and  Maj(a,b,c) = (a&b) | (c & (a^b)) are
;     the minimal-op algebraic forms of the FIPS choose/majority functions.
;   * All round arithmetic is plain wrapping i32 add (mod 2^32) — NO nuw/nsw:
;     a lying no-wrap flag would be poison here (Hazard #1).
;
; HARDENING-TODO: this is the fast, not-constant-time family. No secret-
;   dependent branches exist in the compression, but the API does not zeroize
;   the context/buffer on final; add zeroization + constant-time review in the
;   hardening phase.
;
; API (C ABI, nounwind):
;   void universe_crypto_sha256_init(ptr ctx)
;   void universe_crypto_sha256_update(ptr ctx, ptr data, i64 len)
;   void universe_crypto_sha256_final(ptr ctx, ptr out32)
;   void universe_crypto_sha256_hash(ptr data, i64 len, ptr out32)

declare i32 @llvm.bswap.i32(i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

@sha256.K = private unnamed_addr constant [64 x i32]
[ i32 1116352408, i32 1899447441, i32 3049323471, i32 3921009573, i32 961987163, i32 1508970993, i32 2453635748, i32 2870763221, i32 3624381080, i32 310598401, i32 607225278, i32 1426881987, i32 1925078388, i32 2162078206, i32 2614888103, i32 3248222580, i32 3835390401, i32 4022224774, i32 264347078, i32 604807628, i32 770255983, i32 1249150122, i32 1555081692, i32 1996064986, i32 2554220882, i32 2821834349, i32 2952996808, i32 3210313671, i32 3336571891, i32 3584528711, i32 113926993, i32 338241895, i32 666307205, i32 773529912, i32 1294757372, i32 1396182291, i32 1695183700, i32 1986661051, i32 2177026350, i32 2456956037, i32 2730485921, i32 2820302411, i32 3259730800, i32 3345764771, i32 3516065817, i32 3600352804, i32 4094571909, i32 275423344, i32 430227734, i32 506948616, i32 659060556, i32 883997877, i32 958139571, i32 1322822218, i32 1537002063, i32 1747873779, i32 1955562222, i32 2024104815, i32 2227730452, i32 2361852424, i32 2428436474, i32 2756734187, i32 3204031479, i32 3329325298 ], align 16

; ------------------------------------------------------------------- compress
define internal void @sha256_compress(ptr noalias %st, ptr noalias %blk) #0 {
entry:
  %W = alloca [64 x i32], align 16
  br label %ld.head

ld.head:                                          ; load 16 big-endian words
  %i = phi i64 [ 0, %entry ], [ %i.n, %ld.body ]
  %ic = icmp ult i64 %i, 16
  br i1 %ic, label %ld.body, label %ext.head

ld.body:
  %bp = getelementptr inbounds nuw i32, ptr %blk, i64 %i
  %raw = load i32, ptr %bp, align 1
  %be = call i32 @llvm.bswap.i32(i32 %raw)
  %wp = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %i
  store i32 %be, ptr %wp, align 4
  %i.n = add nuw nsw i64 %i, 1
  br label %ld.head

ext.head:                                         ; extend to 64 words
  %j = phi i64 [ 16, %ld.head ], [ %j.n, %ext.body ]
  %jc = icmp ult i64 %j, 64
  br i1 %jc, label %ext.body, label %comp.pre

ext.body:
  %j2 = sub nuw nsw i64 %j, 2
  %j7 = sub nuw nsw i64 %j, 7
  %j15 = sub nuw nsw i64 %j, 15
  %j16 = sub nuw nsw i64 %j, 16
  %p2 = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %j2
  %w2 = load i32, ptr %p2, align 4
  %p7 = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %j7
  %w7 = load i32, ptr %p7, align 4
  %p15 = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %j15
  %w15 = load i32, ptr %p15, align 4
  %p16 = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %j16
  %w16 = load i32, ptr %p16, align 4
  ; sigma1(w2) = ror(w2,17) ^ ror(w2,19) ^ (w2 >> 10)
  %s1a = call i32 @llvm.fshr.i32(i32 %w2, i32 %w2, i32 17)
  %s1b = call i32 @llvm.fshr.i32(i32 %w2, i32 %w2, i32 19)
  %s1c = lshr i32 %w2, 10
  %s1x = xor i32 %s1a, %s1b
  %sig1 = xor i32 %s1x, %s1c
  ; sigma0(w15) = ror(w15,7) ^ ror(w15,18) ^ (w15 >> 3)
  %s0a = call i32 @llvm.fshr.i32(i32 %w15, i32 %w15, i32 7)
  %s0b = call i32 @llvm.fshr.i32(i32 %w15, i32 %w15, i32 18)
  %s0c = lshr i32 %w15, 3
  %s0x = xor i32 %s0a, %s0b
  %sig0 = xor i32 %s0x, %s0c
  %sum0 = add i32 %sig1, %w7
  %sum1 = add i32 %sum0, %sig0
  %wj = add i32 %sum1, %w16
  %pj = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %j
  store i32 %wj, ptr %pj, align 4
  %j.n = add nuw nsw i64 %j, 1
  br label %ext.head

comp.pre:                                         ; load working state
  %sp0 = getelementptr inbounds nuw i32, ptr %st, i64 0
  %a0 = load i32, ptr %sp0, align 4
  %sp1 = getelementptr inbounds nuw i32, ptr %st, i64 1
  %b0 = load i32, ptr %sp1, align 4
  %sp2 = getelementptr inbounds nuw i32, ptr %st, i64 2
  %c0 = load i32, ptr %sp2, align 4
  %sp3 = getelementptr inbounds nuw i32, ptr %st, i64 3
  %d0 = load i32, ptr %sp3, align 4
  %sp4 = getelementptr inbounds nuw i32, ptr %st, i64 4
  %e0 = load i32, ptr %sp4, align 4
  %sp5 = getelementptr inbounds nuw i32, ptr %st, i64 5
  %f0 = load i32, ptr %sp5, align 4
  %sp6 = getelementptr inbounds nuw i32, ptr %st, i64 6
  %g0 = load i32, ptr %sp6, align 4
  %sp7 = getelementptr inbounds nuw i32, ptr %st, i64 7
  %h0 = load i32, ptr %sp7, align 4
  br label %r.head

r.head:
  %t = phi i64 [ 0, %comp.pre ], [ %t.n, %r.body ]
  %a = phi i32 [ %a0, %comp.pre ], [ %a.n, %r.body ]
  %b = phi i32 [ %b0, %comp.pre ], [ %b.n, %r.body ]
  %c = phi i32 [ %c0, %comp.pre ], [ %c.n, %r.body ]
  %d = phi i32 [ %d0, %comp.pre ], [ %d.n, %r.body ]
  %e = phi i32 [ %e0, %comp.pre ], [ %e.n, %r.body ]
  %f = phi i32 [ %f0, %comp.pre ], [ %f.n, %r.body ]
  %g = phi i32 [ %g0, %comp.pre ], [ %g.n, %r.body ]
  %h = phi i32 [ %h0, %comp.pre ], [ %h.n, %r.body ]
  %tc = icmp ult i64 %t, 64
  br i1 %tc, label %r.body, label %comp.post

r.body:
  %kp = getelementptr inbounds nuw [64 x i32], ptr @sha256.K, i64 0, i64 %t
  %k = load i32, ptr %kp, align 4
  %wp2 = getelementptr inbounds nuw [64 x i32], ptr %W, i64 0, i64 %t
  %w = load i32, ptr %wp2, align 4
  ; Sigma1(e) = ror(e,6) ^ ror(e,11) ^ ror(e,25)
  %E6 = call i32 @llvm.fshr.i32(i32 %e, i32 %e, i32 6)
  %E11 = call i32 @llvm.fshr.i32(i32 %e, i32 %e, i32 11)
  %E25 = call i32 @llvm.fshr.i32(i32 %e, i32 %e, i32 25)
  %Ex = xor i32 %E6, %E11
  %S1 = xor i32 %Ex, %E25
  ; Ch(e,f,g) = g ^ (e & (f ^ g))
  %fxg = xor i32 %f, %g
  %eand = and i32 %e, %fxg
  %ch = xor i32 %g, %eand
  ; T1 = h + S1 + ch + k + w
  %t1a = add i32 %h, %S1
  %t1b = add i32 %t1a, %ch
  %t1c = add i32 %t1b, %k
  %T1 = add i32 %t1c, %w
  ; Sigma0(a) = ror(a,2) ^ ror(a,13) ^ ror(a,22)
  %A2 = call i32 @llvm.fshr.i32(i32 %a, i32 %a, i32 2)
  %A13 = call i32 @llvm.fshr.i32(i32 %a, i32 %a, i32 13)
  %A22 = call i32 @llvm.fshr.i32(i32 %a, i32 %a, i32 22)
  %Ax = xor i32 %A2, %A13
  %S0 = xor i32 %Ax, %A22
  ; Maj(a,b,c) = (a & b) | (c & (a ^ b))
  %ab = and i32 %a, %b
  %axb = xor i32 %a, %b
  %cab = and i32 %c, %axb
  %maj = or i32 %ab, %cab
  %T2 = add i32 %S0, %maj
  %a.n = add i32 %T1, %T2
  %b.n = or i32 %a, 0
  %c.n = or i32 %b, 0
  %d.n = or i32 %c, 0
  %e.n = add i32 %d, %T1
  %f.n = or i32 %e, 0
  %g.n = or i32 %f, 0
  %h.n = or i32 %g, 0
  %t.n = add nuw nsw i64 %t, 1
  br label %r.head

comp.post:                                        ; add working state back
  %na = add i32 %a, %a0
  store i32 %na, ptr %sp0, align 4
  %nb = add i32 %b, %b0
  store i32 %nb, ptr %sp1, align 4
  %nc = add i32 %c, %c0
  store i32 %nc, ptr %sp2, align 4
  %nd = add i32 %d, %d0
  store i32 %nd, ptr %sp3, align 4
  %ne = add i32 %e, %e0
  store i32 %ne, ptr %sp4, align 4
  %nf = add i32 %f, %f0
  store i32 %nf, ptr %sp5, align 4
  %ng = add i32 %g, %g0
  store i32 %ng, ptr %sp6, align 4
  %nh = add i32 %h, %h0
  store i32 %nh, ptr %sp7, align 4
  ret void
}

; ----------------------------------------------------------------------- init
define void @universe_crypto_sha256_init(ptr %ctx) local_unnamed_addr #1 {
entry:
  %p0 = getelementptr inbounds nuw i32, ptr %ctx, i64 0
  store i32 1779033703, ptr %p0, align 4
  %p1 = getelementptr inbounds nuw i32, ptr %ctx, i64 1
  store i32 3144134277, ptr %p1, align 4
  %p2 = getelementptr inbounds nuw i32, ptr %ctx, i64 2
  store i32 1013904242, ptr %p2, align 4
  %p3 = getelementptr inbounds nuw i32, ptr %ctx, i64 3
  store i32 2773480762, ptr %p3, align 4
  %p4 = getelementptr inbounds nuw i32, ptr %ctx, i64 4
  store i32 1359893119, ptr %p4, align 4
  %p5 = getelementptr inbounds nuw i32, ptr %ctx, i64 5
  store i32 2600822924, ptr %p5, align 4
  %p6 = getelementptr inbounds nuw i32, ptr %ctx, i64 6
  store i32 528734635, ptr %p6, align 4
  %p7 = getelementptr inbounds nuw i32, ptr %ctx, i64 7
  store i32 1541459225, ptr %p7, align 4
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 32
  store i64 0, ptr %lp, align 8
  ret void
}

; --------------------------------------------------------------------- update
define void @universe_crypto_sha256_update(ptr %ctx, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret, label %go

go:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 32
  %total = load i64, ptr %lp, align 8
  %buffered = and i64 %total, 63
  %total.new = add i64 %total, %len
  store i64 %total.new, ptr %lp, align 8
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 40
  %pos.nz = icmp ne i64 %buffered, 0
  br i1 %pos.nz, label %have_partial, label %bulk_entry

have_partial:
  %need = sub nuw nsw i64 64, %buffered
  %short = icmp ult i64 %len, %need
  br i1 %short, label %copy_short, label %complete_block

copy_short:                                       ; not enough to fill a block
  %dstp = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp, ptr %data, i64 %len, i1 false)
  br label %ret

complete_block:
  %dstp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp2, ptr %data, i64 %need, i1 false)
  call void @sha256_compress(ptr %ctx, ptr %buf)
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
  call void @sha256_compress(ptr %ctx, ptr %dptr)
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
define void @universe_crypto_sha256_final(ptr %ctx, ptr %out) local_unnamed_addr #1 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 32
  %total = load i64, ptr %lp, align 8
  %pos = and i64 %total, 63
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 40
  ; append 0x80
  %p80 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  store i8 -128, ptr %p80, align 1
  %pos1 = add nuw nsw i64 %pos, 1
  ; bit length, big-endian
  %bits = shl i64 %total, 3
  %bits.be = call i64 @llvm.bswap.i64(i64 %bits)
  %lenslot = getelementptr inbounds nuw i8, ptr %buf, i64 56
  %twoblk = icmp ugt i64 %pos1, 56
  br i1 %twoblk, label %two, label %one

one:                                              ; length fits in this block
  %zlen = sub nuw nsw i64 56, %pos1
  %zp = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp, i8 0, i64 %zlen, i1 false)
  store i64 %bits.be, ptr %lenslot, align 1
  call void @sha256_compress(ptr %ctx, ptr %buf)
  br label %emit

two:                                              ; need an extra padding block
  %zlen2 = sub nuw nsw i64 64, %pos1
  %zp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp2, i8 0, i64 %zlen2, i1 false)
  call void @sha256_compress(ptr %ctx, ptr %buf)
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 56, i1 false)
  store i64 %bits.be, ptr %lenslot, align 1
  call void @sha256_compress(ptr %ctx, ptr %buf)
  br label %emit

emit:                                             ; write big-endian digest
  br label %out.head

out.head:
  %oi = phi i64 [ 0, %emit ], [ %oi.n, %out.body ]
  %oc = icmp ult i64 %oi, 8
  br i1 %oc, label %out.body, label %done

out.body:
  %swp = getelementptr inbounds nuw i32, ptr %ctx, i64 %oi
  %sw = load i32, ptr %swp, align 4
  %swb = call i32 @llvm.bswap.i32(i32 %sw)
  %op = getelementptr inbounds nuw i32, ptr %out, i64 %oi
  store i32 %swb, ptr %op, align 1
  %oi.n = add nuw nsw i64 %oi, 1
  br label %out.head

done:
  ret void
}

declare i64 @llvm.bswap.i64(i64)

; ----------------------------------------------------------------------- hash
define void @universe_crypto_sha256_hash(ptr %data, i64 %len, ptr %out) local_unnamed_addr #1 {
entry:
  %ctx = alloca [104 x i8], align 8
  call void @universe_crypto_sha256_init(ptr %ctx)
  call void @universe_crypto_sha256_update(ptr %ctx, ptr %data, i64 %len)
  call void @universe_crypto_sha256_final(ptr %ctx, ptr %out)
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree }
