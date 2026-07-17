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

; universe_ml_quant_* — vector quantization codec + int8/binary distance
; kernels. The memory axis of an embedding store: a stored vector's cost is
; D * bytes-per-component; quantization moves that 4x-48x with bounded recall
; loss. Pure compute over caller-owned memory; never allocates.
;
; DESIGN — self-describing blob:
;   [0]      scheme tag  u8  (0=f32, 1=f16, 2=int8, 3=binary)
;   [1..5)   dims        u32 little-endian
;   [5..)    payload     scheme-specific
;   payload lengths (header excluded):
;     f32    : dims*4                    (raw copy)
;     f16    : dims*2                    (IEEE half per component)
;     int8   : 4 + dims                  (f32 per-vector scale, then one i8 each)
;     binary : (dims+7)/8                (sign bit per component, LSB-first)
;   Blobs are self-describing so a corpus can mix schemes mid-migration and a
;   reader needs no out-of-band metadata. All targets are little-endian, so a
;   plain i16/i32 store IS the LE encoding; header stores use align 1 (offset 1
;   is odd) — no manual byte assembly.
;
; DESIGN — int8 symmetric quantization:
;   scale = max|x| / 127 (or 1.0 for an all-zero vector); q[i] = clamp(round(
;   x[i]/scale), -127, 127). Symmetric so the scale divides OUT of a cosine
;   ratio exactly, which is why cos_i8 needs no scale argument and approximates
;   the original-f32 cosine directly.
;
; DESIGN — SIMD-first-with-scalar-oracle (the DISTANCE kernels; the query hot
;   path). Codecs run once per stored vector (cold relative to query) and stay
;   as clean, overflow-checked scalar loops; the two distance kernels carry the
;   full SIMD-first discipline:
;   * cos_i8 PRIMARY: a <16 x i8> load, sext to <16 x i32>, WIDENING integer
;     multiply-add into three <16 x i32> accumulators (dot, |a|^2, |b|^2), then
;     llvm.vector.reduce.add.v16i32 and a scalar tail (<16 elems). Widen BEFORE
;     multiplying — i8*i8 overflows. i32 accumulation is exact for embedding
;     dims (127*127*dims stays < 2^31 for dims up to ~133k; real embeddings are
;     far smaller). Final one float divide + clamp to [-1,1].
;   * hamming PRIMARY: 16 bytes/iter as <2 x i64>, xor, llvm.ctpop.v2i64 into a
;     <2 x i64> accumulator, llvm.vector.reduce.add.v2i64 at the end + byte tail.
;     i64-lane accumulation never overflows (ctpop<=64/lane * iters).
;   * Each kernel ships a `*_scalar` twin: the sub-16 remainder handler AND the
;     cross-check oracle in tests (vector == scalar exactly on random inputs).
;
; API:
;   i32   universe_ml_quant_encode(dst, dcap, src_f32, dims, scheme, out_len)
;   i32   universe_ml_quant_decode(dst_f32, dcap_elems, src, slen, out_dims)
;   i32   universe_ml_quant_info(src, slen, out_scheme, out_dims)
;   float universe_ml_dist_cos_i8(a_i8, b_i8, n)
;   float universe_ml_dist_cos_i8_scalar(a_i8, b_i8, n)   ; oracle
;   i64   universe_ml_dist_hamming(a_u8, b_u8, nbytes)
;   i64   universe_ml_dist_hamming_scalar(a_u8, b_u8, nbytes) ; oracle
; Errors (i32): 0 OK, 1 NULL_PTR, 3 SIZE_OVERFLOW, 8 INVALID_ARG
;   (BufferTooSmall / bad dim / unknown scheme all map to INVALID_ARG).

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare float @llvm.fabs.f32(float)
declare float @llvm.round.f32(float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.sqrt.f32(float)
declare i8 @llvm.ctpop.i8(i8)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
declare i64 @llvm.vector.reduce.add.v2i64(<2 x i64>)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>)

; ======================================================= payloadLen (internal)
; Overflow-checked payload byte count for scheme in [0,3]; returns bytes and an
; overflow flag packed as {i64 bytes, i1 overflowed}. dims already validated>0.
define internal { i64, i1 } @ml_quant_paylen(i64 %dims, i32 %scheme) #2 {
entry:
  switch i32 %scheme, label %bad [ i32 0, label %f32
                                   i32 1, label %f16
                                   i32 2, label %i8s
                                   i32 3, label %bin ]
f32:
  %m4 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %dims, i64 4)
  ret { i64, i1 } %m4
f16:
  %m2 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %dims, i64 2)
  ret { i64, i1 } %m2
i8s:
  %a4 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %dims, i64 4)
  ret { i64, i1 } %a4
bin:
  %a7 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %dims, i64 7)
  %s7 = extractvalue { i64, i1 } %a7, 0
  %o7 = extractvalue { i64, i1 } %a7, 1
  %by = lshr i64 %s7, 3
  %r0 = insertvalue { i64, i1 } poison, i64 %by, 0
  %r1 = insertvalue { i64, i1 } %r0, i1 %o7, 1
  ret { i64, i1 } %r1
bad:
  ; unreachable in practice (scheme pre-validated); report overflow to be safe.
  %z0 = insertvalue { i64, i1 } poison, i64 0, 0
  %z1 = insertvalue { i64, i1 } %z0, i1 true, 1
  ret { i64, i1 } %z1
}

; ================================================================= encode API
define i32 @universe_ml_quant_encode(ptr %dst, i64 %dcap, ptr %src, i64 %dims,
                                     i32 %scheme, ptr %outlen) #0 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %outlen, null
  %n0 = or i1 %dn, %sn
  %nz = or i1 %n0, %on
  br i1 %nz, label %err.null, label %chk.dims

chk.dims:
  %dzero = icmp eq i64 %dims, 0
  %dbig = icmp ugt i64 %dims, 4294967295
  %dbad = or i1 %dzero, %dbig
  br i1 %dbad, label %err.arg, label %chk.scheme

chk.scheme:
  %sbad = icmp ugt i32 %scheme, 3
  br i1 %sbad, label %err.arg, label %calc

calc:
  %pl = call { i64, i1 } @ml_quant_paylen(i64 %dims, i32 %scheme)
  %plen = extractvalue { i64, i1 } %pl, 0
  %plov = extractvalue { i64, i1 } %pl, 1
  br i1 %plov, label %err.ovf, label %tot

tot:
  %ta = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %plen, i64 5)
  %total = extractvalue { i64, i1 } %ta, 0
  %tov = extractvalue { i64, i1 } %ta, 1
  br i1 %tov, label %err.ovf, label %cap

cap:
  %small = icmp ult i64 %dcap, %total
  br i1 %small, label %err.arg, label %hdr

hdr:
  ; header: scheme u8 @0, dims u32 LE @1
  %s8 = trunc i32 %scheme to i8
  store i8 %s8, ptr %dst, align 1
  %dhdr = getelementptr inbounds nuw i8, ptr %dst, i64 1
  %d32 = trunc i64 %dims to i32
  store i32 %d32, ptr %dhdr, align 1
  %payload = getelementptr inbounds nuw i8, ptr %dst, i64 5
  switch i32 %scheme, label %done [ i32 0, label %enc.f32
                                    i32 1, label %enc.f16
                                    i32 2, label %enc.i8
                                    i32 3, label %enc.bin ]

; ----- f32: raw copy dims*4 bytes -----
enc.f32:
  %nb = shl nuw i64 %dims, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %payload, ptr %src, i64 %nb, i1 false)
  br label %done

; ----- f16: fptrunc each float to half -----
enc.f16:
  br label %f16.loop
f16.loop:
  %fi = phi i64 [ 0, %enc.f16 ], [ %fin, %f16.loop ]
  %fsp = getelementptr inbounds nuw float, ptr %src, i64 %fi
  %fx = load float, ptr %fsp, align 4
  %fh = fptrunc float %fx to half
  %fb = bitcast half %fh to i16
  %fo = shl nuw i64 %fi, 1
  %fdp = getelementptr inbounds nuw i8, ptr %payload, i64 %fo
  store i16 %fb, ptr %fdp, align 1
  %fin = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fin, %dims
  br i1 %fmore, label %f16.loop, label %done

; ----- int8: symmetric per-vector scale -----
enc.i8:
  br label %ma.loop
ma.loop:
  %mi = phi i64 [ 0, %enc.i8 ], [ %min, %ma.loop ]
  %ma = phi float [ 0.0, %enc.i8 ], [ %man, %ma.loop ]
  %msp = getelementptr inbounds nuw float, ptr %src, i64 %mi
  %mx = load float, ptr %msp, align 4
  %mabs = call float @llvm.fabs.f32(float %mx)
  %man = call float @llvm.maxnum.f32(float %ma, float %mabs)
  %min = add nuw i64 %mi, 1
  %mmore = icmp ult i64 %min, %dims
  br i1 %mmore, label %ma.loop, label %scale

scale:
  %mzero = fcmp oeq float %man, 0.0
  %sdiv = fdiv float %man, 127.0
  %sc = select i1 %mzero, float 1.0, float %sdiv
  ; store scale f32 @payload[0]
  store float %sc, ptr %payload, align 1
  %qbase = getelementptr inbounds nuw i8, ptr %payload, i64 4
  br label %q.loop
q.loop:
  %qi = phi i64 [ 0, %scale ], [ %qin, %q.loop ]
  %qsp = getelementptr inbounds nuw float, ptr %src, i64 %qi
  %qx = load float, ptr %qsp, align 4
  %qd = fdiv float %qx, %sc
  %qr = call float @llvm.round.f32(float %qd)
  %qhi = call float @llvm.minnum.f32(float %qr, float 127.0)
  %qcl = call float @llvm.maxnum.f32(float %qhi, float -127.0)
  %qv = fptosi float %qcl to i8
  %qdp = getelementptr inbounds nuw i8, ptr %qbase, i64 %qi
  store i8 %qv, ptr %qdp, align 1
  %qin = add nuw i64 %qi, 1
  %qmore = icmp ult i64 %qin, %dims
  br i1 %qmore, label %q.loop, label %done

; ----- binary: sign bit per component -----
enc.bin:
  ; zero the packed payload first (plen bytes), then set bits.
  call void @llvm.memset.p0.i64(ptr %payload, i8 0, i64 %plen, i1 false)
  br label %bin.loop
bin.loop:
  %bi = phi i64 [ 0, %enc.bin ], [ %bin, %bin.cont ]
  %bsp = getelementptr inbounds nuw float, ptr %src, i64 %bi
  %bx = load float, ptr %bsp, align 4
  %bpos = fcmp oge float %bx, 0.0
  br i1 %bpos, label %bin.set, label %bin.cont
bin.set:
  %byi = lshr i64 %bi, 3
  %biti = and i64 %bi, 7
  %bytep = getelementptr inbounds nuw i8, ptr %payload, i64 %byi
  %cur = load i8, ptr %bytep, align 1
  %sh = trunc i64 %biti to i8
  %mask = shl nuw i8 1, %sh
  %newb = or i8 %cur, %mask
  store i8 %newb, ptr %bytep, align 1
  br label %bin.cont
bin.cont:
  %bin = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bin, %dims
  br i1 %bmore, label %bin.loop, label %done

done:
  store i64 %total, ptr %outlen, align 8
  ret i32 0

err.null:
  ret i32 1
err.arg:
  ret i32 8
err.ovf:
  ret i32 3
}

; ================================================================= decode API
define i32 @universe_ml_quant_decode(ptr %dst, i64 %dcapel, ptr %src, i64 %slen,
                                     ptr %outdims) #0 {
entry:
  %dn = icmp eq ptr %dst, null
  %sn = icmp eq ptr %src, null
  %on = icmp eq ptr %outdims, null
  %n0 = or i1 %dn, %sn
  %nz = or i1 %n0, %on
  br i1 %nz, label %err.null, label %chk.hdr

chk.hdr:
  %tooshort = icmp ult i64 %slen, 5
  br i1 %tooshort, label %err.arg, label %rdhdr

rdhdr:
  %s8 = load i8, ptr %src, align 1
  %scheme = zext i8 %s8 to i32
  %sbad = icmp ugt i32 %scheme, 3
  br i1 %sbad, label %err.arg, label %rddims

rddims:
  %dhdr = getelementptr inbounds nuw i8, ptr %src, i64 1
  %d32 = load i32, ptr %dhdr, align 1
  %dims = zext i32 %d32 to i64
  %dzero = icmp eq i64 %dims, 0
  br i1 %dzero, label %err.arg, label %chklen

chklen:
  ; expected total = 5 + payloadLen; dims<=u32 so *4 cannot overflow i64.
  %pl = call { i64, i1 } @ml_quant_paylen(i64 %dims, i32 %scheme)
  %plen = extractvalue { i64, i1 } %pl, 0
  %total = add nuw i64 %plen, 5
  %blobshort = icmp ult i64 %slen, %total
  br i1 %blobshort, label %err.arg, label %chkcap

chkcap:
  %capshort = icmp ult i64 %dcapel, %dims
  br i1 %capshort, label %err.arg, label %dispatch

dispatch:
  %payload = getelementptr inbounds nuw i8, ptr %src, i64 5
  switch i32 %scheme, label %fin [ i32 0, label %dec.f32
                                   i32 1, label %dec.f16
                                   i32 2, label %dec.i8
                                   i32 3, label %dec.bin ]

dec.f32:
  %nb = shl nuw i64 %dims, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %payload, i64 %nb, i1 false)
  br label %fin

dec.f16:
  br label %f16.loop
f16.loop:
  %fi = phi i64 [ 0, %dec.f16 ], [ %fin2, %f16.loop ]
  %fo = shl nuw i64 %fi, 1
  %fsp = getelementptr inbounds nuw i8, ptr %payload, i64 %fo
  %fb = load i16, ptr %fsp, align 1
  %fh = bitcast i16 %fb to half
  %fx = fpext half %fh to float
  %fdp = getelementptr inbounds nuw float, ptr %dst, i64 %fi
  store float %fx, ptr %fdp, align 4
  %fin2 = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fin2, %dims
  br i1 %fmore, label %f16.loop, label %fin

dec.i8:
  %sc = load float, ptr %payload, align 1
  %qbase = getelementptr inbounds nuw i8, ptr %payload, i64 4
  br label %i8.loop
i8.loop:
  %qi = phi i64 [ 0, %dec.i8 ], [ %qin, %i8.loop ]
  %qsp = getelementptr inbounds nuw i8, ptr %qbase, i64 %qi
  %qb = load i8, ptr %qsp, align 1
  %qf = sitofp i8 %qb to float
  %qv = fmul float %qf, %sc
  %qdp = getelementptr inbounds nuw float, ptr %dst, i64 %qi
  store float %qv, ptr %qdp, align 4
  %qin = add nuw i64 %qi, 1
  %qmore = icmp ult i64 %qin, %dims
  br i1 %qmore, label %i8.loop, label %fin

dec.bin:
  br label %bin.loop
bin.loop:
  %bi = phi i64 [ 0, %dec.bin ], [ %bin, %bin.loop ]
  %byi = lshr i64 %bi, 3
  %biti = and i64 %bi, 7
  %bytep = getelementptr inbounds nuw i8, ptr %payload, i64 %byi
  %byte = load i8, ptr %bytep, align 1
  %sh = trunc i64 %biti to i8
  %shifted = lshr i8 %byte, %sh
  %bit = and i8 %shifted, 1
  %isset = icmp eq i8 %bit, 1
  %val = select i1 %isset, float 1.0, float -1.0
  %bdp = getelementptr inbounds nuw float, ptr %dst, i64 %bi
  store float %val, ptr %bdp, align 4
  %bin = add nuw i64 %bi, 1
  %bmore = icmp ult i64 %bin, %dims
  br i1 %bmore, label %bin.loop, label %fin

fin:
  store i64 %dims, ptr %outdims, align 8
  ret i32 0

err.null:
  ret i32 1
err.arg:
  ret i32 8
}

; =================================================================== info API
define i32 @universe_ml_quant_info(ptr %src, i64 %slen, ptr %outscheme,
                                   ptr %outdims) #0 {
entry:
  %sn = icmp eq ptr %src, null
  %scn = icmp eq ptr %outscheme, null
  %dn = icmp eq ptr %outdims, null
  %n0 = or i1 %sn, %scn
  %nz = or i1 %n0, %dn
  br i1 %nz, label %err.null, label %chk

chk:
  %tooshort = icmp ult i64 %slen, 5
  br i1 %tooshort, label %err.arg, label %rd

rd:
  %s8 = load i8, ptr %src, align 1
  %scheme = zext i8 %s8 to i32
  %sbad = icmp ugt i32 %scheme, 3
  br i1 %sbad, label %err.arg, label %store

store:
  %dhdr = getelementptr inbounds nuw i8, ptr %src, i64 1
  %d32 = load i32, ptr %dhdr, align 1
  %dims = zext i32 %d32 to i64
  store i32 %scheme, ptr %outscheme, align 4
  store i64 %dims, ptr %outdims, align 8
  ret i32 0

err.null:
  ret i32 1
err.arg:
  ret i32 8
}

; ============================================================= cos_i8 (vector)
; Cosine similarity over symmetric-int8 quanta. Widen i8->i32 before multiply;
; accumulate dot/na/nb as <16 x i32>, reduce, scalar tail, one float divide.
define float @universe_ml_dist_cos_i8(ptr readonly %a, ptr readonly %b, i64 %n) #1 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %vred

vloop:
  %i = phi i64 [ 0, %entry ], [ %in, %vloop ]
  %vdot = phi <16 x i32> [ zeroinitializer, %entry ], [ %vdotn, %vloop ]
  %vna = phi <16 x i32> [ zeroinitializer, %entry ], [ %vnan, %vloop ]
  %vnb = phi <16 x i32> [ zeroinitializer, %entry ], [ %vnbn, %vloop ]
  %pa = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %va8 = load <16 x i8>, ptr %pa, align 1
  %pb = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %vb8 = load <16 x i8>, ptr %pb, align 1
  %xa = sext <16 x i8> %va8 to <16 x i32>
  %xb = sext <16 x i8> %vb8 to <16 x i32>
  %md = mul nsw <16 x i32> %xa, %xb
  %vdotn = add nsw <16 x i32> %vdot, %md
  %maa = mul nsw <16 x i32> %xa, %xa
  %vnan = add nsw <16 x i32> %vna, %maa
  %mbb = mul nsw <16 x i32> %xb, %xb
  %vnbn = add nsw <16 x i32> %vnb, %mbb
  %in = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %in, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hdot = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %vdotn)
  %hna = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %vnan)
  %hnb = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %vnbn)
  br label %vred

vred:
  %sdot0 = phi i32 [ 0, %entry ], [ %hdot, %vdone ]
  %sna0 = phi i32 [ 0, %entry ], [ %hna, %vdone ]
  %snb0 = phi i32 [ 0, %entry ], [ %hnb, %vdone ]
  %start = phi i64 [ 0, %entry ], [ %in, %vdone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %vred ], [ %jn, %tbody ]
  %adot = phi i32 [ %sdot0, %vred ], [ %adotn, %tbody ]
  %ana = phi i32 [ %sna0, %vred ], [ %anan, %tbody ]
  %anb = phi i32 [ %snb0, %vred ], [ %anbn, %tbody ]
  %tdone = icmp uge i64 %j, %n
  br i1 %tdone, label %finish, label %tbody

tbody:
  %tap = getelementptr inbounds nuw i8, ptr %a, i64 %j
  %ta8 = load i8, ptr %tap, align 1
  %fa = sext i8 %ta8 to i32
  %tbp = getelementptr inbounds nuw i8, ptr %b, i64 %j
  %tb8 = load i8, ptr %tbp, align 1
  %fb = sext i8 %tb8 to i32
  %pd = mul nsw i32 %fa, %fb
  %adotn = add nsw i32 %adot, %pd
  %pna = mul nsw i32 %fa, %fa
  %anan = add nsw i32 %ana, %pna
  %pnb = mul nsw i32 %fb, %fb
  %anbn = add nsw i32 %anb, %pnb
  %jn = add nuw i64 %j, 1
  br label %tail

finish:
  %naz = icmp eq i32 %ana, 0
  %nbz = icmp eq i32 %anb, 0
  %anyz = or i1 %naz, %nbz
  br i1 %anyz, label %retzero, label %compute

compute:
  %naf = sitofp i32 %ana to float
  %nbf = sitofp i32 %anb to float
  %prod = fmul float %naf, %nbf
  %den = call float @llvm.sqrt.f32(float %prod)
  %dotf = sitofp i32 %adot to float
  %sim = fdiv float %dotf, %den
  %clhi = call float @llvm.minnum.f32(float %sim, float 1.0)
  %cl = call float @llvm.maxnum.f32(float %clhi, float -1.0)
  ret float %cl

retzero:
  ret float 0.0
}

; ============================================================= cos_i8 (oracle)
define float @universe_ml_dist_cos_i8_scalar(ptr readonly %a, ptr readonly %b, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %retzero, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %adot = phi i32 [ 0, %entry ], [ %adotn, %loop ]
  %ana = phi i32 [ 0, %entry ], [ %anan, %loop ]
  %anb = phi i32 [ 0, %entry ], [ %anbn, %loop ]
  %pa = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %a8 = load i8, ptr %pa, align 1
  %fa = sext i8 %a8 to i32
  %pb = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %b8 = load i8, ptr %pb, align 1
  %fb = sext i8 %b8 to i32
  %pd = mul nsw i32 %fa, %fb
  %adotn = add nsw i32 %adot, %pd
  %pna = mul nsw i32 %fa, %fa
  %anan = add nsw i32 %ana, %pna
  %pnb = mul nsw i32 %fb, %fb
  %anbn = add nsw i32 %anb, %pnb
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %reduce

reduce:
  %naz = icmp eq i32 %anan, 0
  %nbz = icmp eq i32 %anbn, 0
  %anyz = or i1 %naz, %nbz
  br i1 %anyz, label %retzero, label %compute

compute:
  %naf = sitofp i32 %anan to float
  %nbf = sitofp i32 %anbn to float
  %prod = fmul float %naf, %nbf
  %den = call float @llvm.sqrt.f32(float %prod)
  %dotf = sitofp i32 %adotn to float
  %sim = fdiv float %dotf, %den
  %clhi = call float @llvm.minnum.f32(float %sim, float 1.0)
  %cl = call float @llvm.maxnum.f32(float %clhi, float -1.0)
  ret float %cl

retzero:
  ret float 0.0
}

; ============================================================ hamming (vector)
; popcount(a XOR b) over 16-byte <2 x i64> lanes with vector ctpop; byte tail.
define i64 @universe_ml_dist_hamming(ptr readonly %a, ptr readonly %b, i64 %n) #1 {
entry:
  %has16 = icmp uge i64 %n, 16
  br i1 %has16, label %vloop, label %vred

vloop:
  %i = phi i64 [ 0, %entry ], [ %in, %vloop ]
  %vacc = phi <2 x i64> [ zeroinitializer, %entry ], [ %vaccn, %vloop ]
  %pa = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %va = load <2 x i64>, ptr %pa, align 1
  %pb = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %vb = load <2 x i64>, ptr %pb, align 1
  %x = xor <2 x i64> %va, %vb
  %pc = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x)
  %vaccn = add <2 x i64> %vacc, %pc
  %in = add nuw i64 %i, 16
  %lim = sub nuw i64 %n, 16
  %more = icmp ule i64 %in, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hpc = call i64 @llvm.vector.reduce.add.v2i64(<2 x i64> %vaccn)
  br label %vred

vred:
  %base = phi i64 [ 0, %entry ], [ %hpc, %vdone ]
  %start = phi i64 [ 0, %entry ], [ %in, %vdone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %vred ], [ %jn, %tbody ]
  %acc = phi i64 [ %base, %vred ], [ %accn, %tbody ]
  %tdone = icmp uge i64 %j, %n
  br i1 %tdone, label %done, label %tbody

tbody:
  %tap = getelementptr inbounds nuw i8, ptr %a, i64 %j
  %ta = load i8, ptr %tap, align 1
  %tbp = getelementptr inbounds nuw i8, ptr %b, i64 %j
  %tb = load i8, ptr %tbp, align 1
  %tx = xor i8 %ta, %tb
  %tpc = call i8 @llvm.ctpop.i8(i8 %tx)
  %tpc64 = zext i8 %tpc to i64
  %accn = add nuw i64 %acc, %tpc64
  %jn = add nuw i64 %j, 1
  br label %tail

done:
  ret i64 %acc
}

; ============================================================ hamming (oracle)
define i64 @universe_ml_dist_hamming_scalar(ptr readonly %a, ptr readonly %b, i64 %n) #1 {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %accn, %loop ]
  %pa = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %ba = load i8, ptr %pa, align 1
  %pb = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bb = load i8, ptr %pb, align 1
  %x = xor i8 %ba, %bb
  %pc = call i8 @llvm.ctpop.i8(i8 %x)
  %pc64 = zext i8 %pc to i64
  %accn = add nuw i64 %acc, %pc64
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done

done:
  %r = phi i64 [ 0, %entry ], [ %accn, %loop ]
  ret i64 %r
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(none) }
