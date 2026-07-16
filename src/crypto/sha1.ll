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

; SHA-1 (FIPS 180-4). Streaming context + one-shot. Digest = 20 bytes.
; NOTE: SHA-1 is cryptographically broken (collisions); provided for legacy
; interop/checksums only, not for security. See HARDENING-TODO.
;
; DESIGN:
;   * CTX layout (caller-supplied, single allocation):
;       off  0: 5 x i32 state words        (20 B)
;       off 24: i64   total byte length     (8 B)   fill = (total & 63)
;       off 32: 64 x i8 partial buffer      (64 B)
;     => UNIVERSE_CRYPTO_SHA1_CTX_SIZE = 96 bytes, align 8. (Length lives at
;     +24 for 8-byte alignment; the 4-byte hole at +20 is intentional.)
;   * HOT LEAF @sha1_compress: internal alwaysinline; a..e are loop-carried SSA
;     so the 80 unrolled rounds stay register-resident. The per-phase f/k are
;     chosen by selects on the loop index t; with the constant trip count (80)
;     -O3 unrolls and folds each select to the phase constant, leaving no
;     branch in the round. Rotations via llvm.fshl.i32(x,x,n) -> single rol.
;   * Big-endian word loads via llvm.bswap.i32. All round adds are plain
;     wrapping i32 (NO nuw/nsw — a no-wrap flag would be poison, Hazard #1).
;
; HARDENING-TODO: fast, not constant-time; no context zeroization on final.
;   SHA-1 must not be used where collision resistance is required.
;
; API (C ABI, nounwind):
;   void universe_crypto_sha1_init(ptr ctx)
;   void universe_crypto_sha1_update(ptr ctx, ptr data, i64 len)
;   void universe_crypto_sha1_final(ptr ctx, ptr out20)
;   void universe_crypto_sha1_hash(ptr data, i64 len, ptr out20)

declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; ------------------------------------------------------------------- compress
define internal void @sha1_compress(ptr noalias %st, ptr noalias %blk) #0 {
entry:
  %W = alloca [80 x i32], align 16
  br label %ld.head

ld.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %ld.body ]
  %ic = icmp ult i64 %i, 16
  br i1 %ic, label %ld.body, label %ext.head

ld.body:
  %bp = getelementptr inbounds nuw i32, ptr %blk, i64 %i
  %raw = load i32, ptr %bp, align 1
  %be = call i32 @llvm.bswap.i32(i32 %raw)
  %wp = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %i
  store i32 %be, ptr %wp, align 4
  %i.n = add nuw nsw i64 %i, 1
  br label %ld.head

ext.head:
  %j = phi i64 [ 16, %ld.head ], [ %j.n, %ext.body ]
  %jc = icmp ult i64 %j, 80
  br i1 %jc, label %ext.body, label %comp.pre

ext.body:
  ; W[j] = rol(W[j-3] ^ W[j-8] ^ W[j-14] ^ W[j-16], 1)
  %j3 = sub nuw nsw i64 %j, 3
  %j8 = sub nuw nsw i64 %j, 8
  %j14 = sub nuw nsw i64 %j, 14
  %j16 = sub nuw nsw i64 %j, 16
  %p3 = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %j3
  %w3 = load i32, ptr %p3, align 4
  %p8 = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %j8
  %w8 = load i32, ptr %p8, align 4
  %p14 = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %j14
  %w14 = load i32, ptr %p14, align 4
  %p16 = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %j16
  %w16 = load i32, ptr %p16, align 4
  %x0 = xor i32 %w3, %w8
  %x1 = xor i32 %x0, %w14
  %x2 = xor i32 %x1, %w16
  %wj = call i32 @llvm.fshl.i32(i32 %x2, i32 %x2, i32 1)
  %pj = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %j
  store i32 %wj, ptr %pj, align 4
  %j.n = add nuw nsw i64 %j, 1
  br label %ext.head

comp.pre:
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
  br label %r.head

r.head:
  %t = phi i64 [ 0, %comp.pre ], [ %t.n, %r.body ]
  %a = phi i32 [ %a0, %comp.pre ], [ %a.n, %r.body ]
  %b = phi i32 [ %b0, %comp.pre ], [ %b.n, %r.body ]
  %c = phi i32 [ %c0, %comp.pre ], [ %c.n, %r.body ]
  %d = phi i32 [ %d0, %comp.pre ], [ %d.n, %r.body ]
  %e = phi i32 [ %e0, %comp.pre ], [ %e.n, %r.body ]
  %tc = icmp ult i64 %t, 80
  br i1 %tc, label %r.body, label %comp.post

r.body:
  %wp2 = getelementptr inbounds nuw [80 x i32], ptr %W, i64 0, i64 %t
  %w = load i32, ptr %wp2, align 4
  ; phase predicates
  %ph1 = icmp ult i64 %t, 20
  %ph2 = icmp ult i64 %t, 40
  %ph3 = icmp ult i64 %t, 60
  ; f candidates
  %f_choose_cd = xor i32 %c, %d
  %f_choose_bcd = and i32 %b, %f_choose_cd
  %f_choose = xor i32 %d, %f_choose_bcd          ; (b&c)|(~b&d)
  %f_par0 = xor i32 %b, %c
  %f_parity = xor i32 %f_par0, %d                ; b^c^d
  %f_maj_bc = and i32 %b, %c
  %f_maj_bxc = xor i32 %b, %c
  %f_maj_dm = and i32 %d, %f_maj_bxc
  %f_maj = or i32 %f_maj_bc, %f_maj_dm           ; (b&c)|(b&d)|(c&d)
  ; select f by phase
  %f_s2 = select i1 %ph3, i32 %f_maj, i32 %f_parity
  %f_s1 = select i1 %ph2, i32 %f_parity, i32 %f_s2
  %f = select i1 %ph1, i32 %f_choose, i32 %f_s1
  ; select k by phase
  %k_s2 = select i1 %ph3, i32 -1894007588, i32 -899497514     ; K2=0x8F1BBCDC, K3=0xCA62C1D6
  %k_s1 = select i1 %ph2, i32 1859775393, i32 %k_s2           ; K1=0x6ED9EBA1
  %k = select i1 %ph1, i32 1518500249, i32 %k_s1              ; K0=0x5A827999
  ; temp = rol(a,5) + f + e + k + W[t]
  %rola5 = call i32 @llvm.fshl.i32(i32 %a, i32 %a, i32 5)
  %tp0 = add i32 %rola5, %f
  %tp1 = add i32 %tp0, %e
  %tp2 = add i32 %tp1, %k
  %temp = add i32 %tp2, %w
  ; rotate the working registers
  %a.n = or i32 %temp, 0
  %b.n = or i32 %a, 0
  %c.n = call i32 @llvm.fshl.i32(i32 %b, i32 %b, i32 30)      ; rol(b,30)
  %d.n = or i32 %c, 0
  %e.n = or i32 %d, 0
  %t.n = add nuw nsw i64 %t, 1
  br label %r.head

comp.post:
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
  ret void
}

; ----------------------------------------------------------------------- init
define void @universe_crypto_sha1_init(ptr %ctx) local_unnamed_addr #1 {
entry:
  %p0 = getelementptr inbounds nuw i32, ptr %ctx, i64 0
  store i32 1732584193, ptr %p0, align 4
  %p1 = getelementptr inbounds nuw i32, ptr %ctx, i64 1
  store i32 -271733879, ptr %p1, align 4                      ; 0xEFCDAB89
  %p2 = getelementptr inbounds nuw i32, ptr %ctx, i64 2
  store i32 -1732584194, ptr %p2, align 4                     ; 0x98BADCFE
  %p3 = getelementptr inbounds nuw i32, ptr %ctx, i64 3
  store i32 271733878, ptr %p3, align 4
  %p4 = getelementptr inbounds nuw i32, ptr %ctx, i64 4
  store i32 -1009589776, ptr %p4, align 4                     ; 0xC3D2E1F0
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  store i64 0, ptr %lp, align 8
  ret void
}

; --------------------------------------------------------------------- update
define void @universe_crypto_sha1_update(ptr %ctx, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret, label %go

go:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %total = load i64, ptr %lp, align 8
  %buffered = and i64 %total, 63
  %total.new = add i64 %total, %len
  store i64 %total.new, ptr %lp, align 8
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 32
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
  call void @sha1_compress(ptr %ctx, ptr %buf)
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
  call void @sha1_compress(ptr %ctx, ptr %dptr)
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
define void @universe_crypto_sha1_final(ptr %ctx, ptr %out) local_unnamed_addr #1 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 24
  %total = load i64, ptr %lp, align 8
  %pos = and i64 %total, 63
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 32
  %p80 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  store i8 -128, ptr %p80, align 1
  %pos1 = add nuw nsw i64 %pos, 1
  %bits = shl i64 %total, 3
  %bits.be = call i64 @llvm.bswap.i64(i64 %bits)
  %lenslot = getelementptr inbounds nuw i8, ptr %buf, i64 56
  %twoblk = icmp ugt i64 %pos1, 56
  br i1 %twoblk, label %two, label %one

one:
  %zlen = sub nuw nsw i64 56, %pos1
  %zp = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp, i8 0, i64 %zlen, i1 false)
  store i64 %bits.be, ptr %lenslot, align 1
  call void @sha1_compress(ptr %ctx, ptr %buf)
  br label %emit

two:
  %zlen2 = sub nuw nsw i64 64, %pos1
  %zp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp2, i8 0, i64 %zlen2, i1 false)
  call void @sha1_compress(ptr %ctx, ptr %buf)
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 56, i1 false)
  store i64 %bits.be, ptr %lenslot, align 1
  call void @sha1_compress(ptr %ctx, ptr %buf)
  br label %emit

emit:
  br label %out.head

out.head:
  %oi = phi i64 [ 0, %emit ], [ %oi.n, %out.body ]
  %oc = icmp ult i64 %oi, 5
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

; ----------------------------------------------------------------------- hash
define void @universe_crypto_sha1_hash(ptr %data, i64 %len, ptr %out) local_unnamed_addr #1 {
entry:
  %ctx = alloca [96 x i8], align 8
  call void @universe_crypto_sha1_init(ptr %ctx)
  call void @universe_crypto_sha1_update(ptr %ctx, ptr %data, i64 %len)
  call void @universe_crypto_sha1_final(ptr %ctx, ptr %out)
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree }
