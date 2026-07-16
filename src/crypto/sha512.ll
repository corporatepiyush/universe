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

; SHA-512 (FIPS 180-4). Streaming context + one-shot. Digest = 64 bytes.
;
; DESIGN:
;   * Identical structure to SHA-256 but 64-bit words, 128-byte blocks, 80
;     rounds, and a 128-bit length field. CTX layout (caller-supplied):
;       off  0: 8 x i64 state words        (64 B)
;       off 64: i64   total byte length     (8 B)   fill = (total & 127)
;       off 72: 128 x i8 partial buffer    (128 B)
;     => UNIVERSE_CRYPTO_SHA512_CTX_SIZE = 200 bytes, align 8.
;   * The 128-bit big-endian length appended in padding is
;       high = total >> 61 , low = total << 3 (total is a byte count).
;   * HOT LEAF @sha512_compress: internal alwaysinline; a..h are loop-carried
;     SSA so 80 unrolled rounds stay register-resident. Rotations via
;     llvm.fshr.i64(x,x,n) -> single ror. Big-endian word loads via
;     llvm.bswap.i64. All round adds are plain wrapping i64 (NO nuw/nsw).
;
; HARDENING-TODO: fast, not constant-time; no context zeroization on final.
;
; API (C ABI, nounwind):
;   void universe_crypto_sha512_init(ptr ctx)
;   void universe_crypto_sha512_update(ptr ctx, ptr data, i64 len)
;   void universe_crypto_sha512_final(ptr ctx, ptr out64)
;   void universe_crypto_sha512_hash(ptr data, i64 len, ptr out64)

declare i64 @llvm.bswap.i64(i64)
declare i64 @llvm.fshr.i64(i64, i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

@sha512.K = private unnamed_addr constant [80 x i64]
[ i64 4794697086780616226, i64 8158064640168781261, i64 -5349999486874862801, i64 -1606136188198331460, i64 4131703408338449720, i64 6480981068601479193, i64 -7908458776815382629, i64 -6116909921290321640, i64 -2880145864133508542, i64 1334009975649890238, i64 2608012711638119052, i64 6128411473006802146, i64 8268148722764581231, i64 -9160688886553864527, i64 -7215885187991268811, i64 -4495734319001033068, i64 -1973867731355612462, i64 -1171420211273849373, i64 1135362057144423861, i64 2597628984639134821, i64 3308224258029322869, i64 5365058923640841347, i64 6679025012923562964, i64 8573033837759648693, i64 -7476448914759557205, i64 -6327057829258317296, i64 -5763719355590565569, i64 -4658551843659510044, i64 -4116276920077217854, i64 -3051310485924567259, i64 489312712824947311, i64 1452737877330783856, i64 2861767655752347644, i64 3322285676063803686, i64 5560940570517711597, i64 5996557281743188959, i64 7280758554555802590, i64 8532644243296465576, i64 -9096487096722542874, i64 -7894198246740708037, i64 -6719396339535248540, i64 -6333637450476146687, i64 -4446306890439682159, i64 -4076793802049405392, i64 -3345356375505022440, i64 -2983346525034927856, i64 -860691631967231958, i64 1182934255886127544, i64 1847814050463011016, i64 2177327727835720531, i64 2830643537854262169, i64 3796741975233480872, i64 4115178125766777443, i64 5681478168544905931, i64 6601373596472566643, i64 7507060721942968483, i64 8399075790359081724, i64 8693463985226723168, i64 -8878714635349349518, i64 -8302665154208450068, i64 -8016688836872298968, i64 -6606660893046293015, i64 -4685533653050689259, i64 -4147400797238176981, i64 -3880063495543823972, i64 -3348786107499101689, i64 -1523767162380948706, i64 -757361751448694408, i64 500013540394364858, i64 748580250866718886, i64 1242879168328830382, i64 1977374033974150939, i64 2944078676154940804, i64 3659926193048069267, i64 4368137639120453308, i64 4836135668995329356, i64 5532061633213252278, i64 6448918945643986474, i64 6902733635092675308, i64 7801388544844847127 ], align 16

; ------------------------------------------------------------------- compress
define internal void @sha512_compress(ptr noalias %st, ptr noalias %blk) #0 {
entry:
  %W = alloca [80 x i64], align 16
  br label %ld.head

ld.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %ld.body ]
  %ic = icmp ult i64 %i, 16
  br i1 %ic, label %ld.body, label %ext.head

ld.body:
  %bp = getelementptr inbounds nuw i64, ptr %blk, i64 %i
  %raw = load i64, ptr %bp, align 1
  %be = call i64 @llvm.bswap.i64(i64 %raw)
  %wp = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %i
  store i64 %be, ptr %wp, align 8
  %i.n = add nuw nsw i64 %i, 1
  br label %ld.head

ext.head:
  %j = phi i64 [ 16, %ld.head ], [ %j.n, %ext.body ]
  %jc = icmp ult i64 %j, 80
  br i1 %jc, label %ext.body, label %comp.pre

ext.body:
  %j2 = sub nuw nsw i64 %j, 2
  %j7 = sub nuw nsw i64 %j, 7
  %j15 = sub nuw nsw i64 %j, 15
  %j16 = sub nuw nsw i64 %j, 16
  %p2 = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %j2
  %w2 = load i64, ptr %p2, align 8
  %p7 = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %j7
  %w7 = load i64, ptr %p7, align 8
  %p15 = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %j15
  %w15 = load i64, ptr %p15, align 8
  %p16 = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %j16
  %w16 = load i64, ptr %p16, align 8
  ; sigma1(w2) = ror(w2,19) ^ ror(w2,61) ^ (w2 >> 6)
  %s1a = call i64 @llvm.fshr.i64(i64 %w2, i64 %w2, i64 19)
  %s1b = call i64 @llvm.fshr.i64(i64 %w2, i64 %w2, i64 61)
  %s1c = lshr i64 %w2, 6
  %s1x = xor i64 %s1a, %s1b
  %sig1 = xor i64 %s1x, %s1c
  ; sigma0(w15) = ror(w15,1) ^ ror(w15,8) ^ (w15 >> 7)
  %s0a = call i64 @llvm.fshr.i64(i64 %w15, i64 %w15, i64 1)
  %s0b = call i64 @llvm.fshr.i64(i64 %w15, i64 %w15, i64 8)
  %s0c = lshr i64 %w15, 7
  %s0x = xor i64 %s0a, %s0b
  %sig0 = xor i64 %s0x, %s0c
  %sum0 = add i64 %sig1, %w7
  %sum1 = add i64 %sum0, %sig0
  %wj = add i64 %sum1, %w16
  %pj = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %j
  store i64 %wj, ptr %pj, align 8
  %j.n = add nuw nsw i64 %j, 1
  br label %ext.head

comp.pre:
  %sp0 = getelementptr inbounds nuw i64, ptr %st, i64 0
  %a0 = load i64, ptr %sp0, align 8
  %sp1 = getelementptr inbounds nuw i64, ptr %st, i64 1
  %b0 = load i64, ptr %sp1, align 8
  %sp2 = getelementptr inbounds nuw i64, ptr %st, i64 2
  %c0 = load i64, ptr %sp2, align 8
  %sp3 = getelementptr inbounds nuw i64, ptr %st, i64 3
  %d0 = load i64, ptr %sp3, align 8
  %sp4 = getelementptr inbounds nuw i64, ptr %st, i64 4
  %e0 = load i64, ptr %sp4, align 8
  %sp5 = getelementptr inbounds nuw i64, ptr %st, i64 5
  %f0 = load i64, ptr %sp5, align 8
  %sp6 = getelementptr inbounds nuw i64, ptr %st, i64 6
  %g0 = load i64, ptr %sp6, align 8
  %sp7 = getelementptr inbounds nuw i64, ptr %st, i64 7
  %h0 = load i64, ptr %sp7, align 8
  br label %r.head

r.head:
  %t = phi i64 [ 0, %comp.pre ], [ %t.n, %r.body ]
  %a = phi i64 [ %a0, %comp.pre ], [ %a.n, %r.body ]
  %b = phi i64 [ %b0, %comp.pre ], [ %b.n, %r.body ]
  %c = phi i64 [ %c0, %comp.pre ], [ %c.n, %r.body ]
  %d = phi i64 [ %d0, %comp.pre ], [ %d.n, %r.body ]
  %e = phi i64 [ %e0, %comp.pre ], [ %e.n, %r.body ]
  %f = phi i64 [ %f0, %comp.pre ], [ %f.n, %r.body ]
  %g = phi i64 [ %g0, %comp.pre ], [ %g.n, %r.body ]
  %h = phi i64 [ %h0, %comp.pre ], [ %h.n, %r.body ]
  %tc = icmp ult i64 %t, 80
  br i1 %tc, label %r.body, label %comp.post

r.body:
  %kp = getelementptr inbounds nuw [80 x i64], ptr @sha512.K, i64 0, i64 %t
  %k = load i64, ptr %kp, align 8
  %wp2 = getelementptr inbounds nuw [80 x i64], ptr %W, i64 0, i64 %t
  %w = load i64, ptr %wp2, align 8
  ; Sigma1(e) = ror(e,14) ^ ror(e,18) ^ ror(e,41)
  %E14 = call i64 @llvm.fshr.i64(i64 %e, i64 %e, i64 14)
  %E18 = call i64 @llvm.fshr.i64(i64 %e, i64 %e, i64 18)
  %E41 = call i64 @llvm.fshr.i64(i64 %e, i64 %e, i64 41)
  %Ex = xor i64 %E14, %E18
  %S1 = xor i64 %Ex, %E41
  ; Ch(e,f,g) = g ^ (e & (f ^ g))
  %fxg = xor i64 %f, %g
  %eand = and i64 %e, %fxg
  %ch = xor i64 %g, %eand
  %t1a = add i64 %h, %S1
  %t1b = add i64 %t1a, %ch
  %t1c = add i64 %t1b, %k
  %T1 = add i64 %t1c, %w
  ; Sigma0(a) = ror(a,28) ^ ror(a,34) ^ ror(a,39)
  %A28 = call i64 @llvm.fshr.i64(i64 %a, i64 %a, i64 28)
  %A34 = call i64 @llvm.fshr.i64(i64 %a, i64 %a, i64 34)
  %A39 = call i64 @llvm.fshr.i64(i64 %a, i64 %a, i64 39)
  %Ax = xor i64 %A28, %A34
  %S0 = xor i64 %Ax, %A39
  ; Maj(a,b,c) = (a & b) | (c & (a ^ b))
  %ab = and i64 %a, %b
  %axb = xor i64 %a, %b
  %cab = and i64 %c, %axb
  %maj = or i64 %ab, %cab
  %T2 = add i64 %S0, %maj
  %a.n = add i64 %T1, %T2
  %b.n = or i64 %a, 0
  %c.n = or i64 %b, 0
  %d.n = or i64 %c, 0
  %e.n = add i64 %d, %T1
  %f.n = or i64 %e, 0
  %g.n = or i64 %f, 0
  %h.n = or i64 %g, 0
  %t.n = add nuw nsw i64 %t, 1
  br label %r.head

comp.post:
  %na = add i64 %a, %a0
  store i64 %na, ptr %sp0, align 8
  %nb = add i64 %b, %b0
  store i64 %nb, ptr %sp1, align 8
  %nc = add i64 %c, %c0
  store i64 %nc, ptr %sp2, align 8
  %nd = add i64 %d, %d0
  store i64 %nd, ptr %sp3, align 8
  %ne = add i64 %e, %e0
  store i64 %ne, ptr %sp4, align 8
  %nf = add i64 %f, %f0
  store i64 %nf, ptr %sp5, align 8
  %ng = add i64 %g, %g0
  store i64 %ng, ptr %sp6, align 8
  %nh = add i64 %h, %h0
  store i64 %nh, ptr %sp7, align 8
  ret void
}

; ----------------------------------------------------------------------- init
define void @universe_crypto_sha512_init(ptr %ctx) local_unnamed_addr #1 {
entry:
  %p0 = getelementptr inbounds nuw i64, ptr %ctx, i64 0
  store i64 7640891576956012808, ptr %p0, align 8
  %p1 = getelementptr inbounds nuw i64, ptr %ctx, i64 1
  store i64 -4942790177534073029, ptr %p1, align 8
  %p2 = getelementptr inbounds nuw i64, ptr %ctx, i64 2
  store i64 4354685564936845355, ptr %p2, align 8
  %p3 = getelementptr inbounds nuw i64, ptr %ctx, i64 3
  store i64 -6534734903238641935, ptr %p3, align 8
  %p4 = getelementptr inbounds nuw i64, ptr %ctx, i64 4
  store i64 5840696475078001361, ptr %p4, align 8
  %p5 = getelementptr inbounds nuw i64, ptr %ctx, i64 5
  store i64 -7276294671716946913, ptr %p5, align 8
  %p6 = getelementptr inbounds nuw i64, ptr %ctx, i64 6
  store i64 2270897969802886507, ptr %p6, align 8
  %p7 = getelementptr inbounds nuw i64, ptr %ctx, i64 7
  store i64 6620516959819538809, ptr %p7, align 8
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 64
  store i64 0, ptr %lp, align 8
  ret void
}

; --------------------------------------------------------------------- update
define void @universe_crypto_sha512_update(ptr %ctx, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %ret, label %go

go:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 64
  %total = load i64, ptr %lp, align 8
  %buffered = and i64 %total, 127
  %total.new = add i64 %total, %len
  store i64 %total.new, ptr %lp, align 8
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 72
  %pos.nz = icmp ne i64 %buffered, 0
  br i1 %pos.nz, label %have_partial, label %bulk_entry

have_partial:
  %need = sub nuw nsw i64 128, %buffered
  %short = icmp ult i64 %len, %need
  br i1 %short, label %copy_short, label %complete_block

copy_short:
  %dstp = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp, ptr %data, i64 %len, i1 false)
  br label %ret

complete_block:
  %dstp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %buffered
  call void @llvm.memcpy.p0.p0.i64(ptr %dstp2, ptr %data, i64 %need, i1 false)
  call void @sha512_compress(ptr %ctx, ptr %buf)
  %data2 = getelementptr inbounds nuw i8, ptr %data, i64 %need
  %rem2 = sub nuw i64 %len, %need
  br label %bulk_loop

bulk_entry:
  br label %bulk_loop

bulk_loop:
  %dptr = phi ptr [ %data2, %complete_block ], [ %data, %bulk_entry ], [ %dptr.n, %bulk_body ]
  %rem = phi i64 [ %rem2, %complete_block ], [ %len, %bulk_entry ], [ %rem.n, %bulk_body ]
  %big = icmp uge i64 %rem, 128
  br i1 %big, label %bulk_body, label %tail

bulk_body:
  call void @sha512_compress(ptr %ctx, ptr %dptr)
  %dptr.n = getelementptr inbounds nuw i8, ptr %dptr, i64 128
  %rem.n = sub nuw i64 %rem, 128
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
define void @universe_crypto_sha512_final(ptr %ctx, ptr %out) local_unnamed_addr #1 {
entry:
  %lp = getelementptr inbounds nuw i8, ptr %ctx, i64 64
  %total = load i64, ptr %lp, align 8
  %pos = and i64 %total, 127
  %buf = getelementptr inbounds nuw i8, ptr %ctx, i64 72
  %p80 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos
  store i8 -128, ptr %p80, align 1
  %pos1 = add nuw nsw i64 %pos, 1
  ; 128-bit big-endian length: high = total>>61, low = total<<3
  %bits.lo = shl i64 %total, 3
  %bits.hi = lshr i64 %total, 61
  %lo.be = call i64 @llvm.bswap.i64(i64 %bits.lo)
  %hi.be = call i64 @llvm.bswap.i64(i64 %bits.hi)
  %hislot = getelementptr inbounds nuw i8, ptr %buf, i64 112
  %loslot = getelementptr inbounds nuw i8, ptr %buf, i64 120
  %twoblk = icmp ugt i64 %pos1, 112
  br i1 %twoblk, label %two, label %one

one:
  %zlen = sub nuw nsw i64 112, %pos1
  %zp = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp, i8 0, i64 %zlen, i1 false)
  store i64 %hi.be, ptr %hislot, align 1
  store i64 %lo.be, ptr %loslot, align 1
  call void @sha512_compress(ptr %ctx, ptr %buf)
  br label %emit

two:
  %zlen2 = sub nuw nsw i64 128, %pos1
  %zp2 = getelementptr inbounds nuw i8, ptr %buf, i64 %pos1
  call void @llvm.memset.p0.i64(ptr %zp2, i8 0, i64 %zlen2, i1 false)
  call void @sha512_compress(ptr %ctx, ptr %buf)
  call void @llvm.memset.p0.i64(ptr %buf, i8 0, i64 112, i1 false)
  store i64 %hi.be, ptr %hislot, align 1
  store i64 %lo.be, ptr %loslot, align 1
  call void @sha512_compress(ptr %ctx, ptr %buf)
  br label %emit

emit:
  br label %out.head

out.head:
  %oi = phi i64 [ 0, %emit ], [ %oi.n, %out.body ]
  %oc = icmp ult i64 %oi, 8
  br i1 %oc, label %out.body, label %done

out.body:
  %swp = getelementptr inbounds nuw i64, ptr %ctx, i64 %oi
  %sw = load i64, ptr %swp, align 8
  %swb = call i64 @llvm.bswap.i64(i64 %sw)
  %op = getelementptr inbounds nuw i64, ptr %out, i64 %oi
  store i64 %swb, ptr %op, align 1
  %oi.n = add nuw nsw i64 %oi, 1
  br label %out.head

done:
  ret void
}

; ----------------------------------------------------------------------- hash
define void @universe_crypto_sha512_hash(ptr %data, i64 %len, ptr %out) local_unnamed_addr #1 {
entry:
  %ctx = alloca [200 x i8], align 8
  call void @universe_crypto_sha512_init(ptr %ctx)
  call void @universe_crypto_sha512_update(ptr %ctx, ptr %data, i64 %len)
  call void @universe_crypto_sha512_final(ptr %ctx, ptr %out)
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree }
