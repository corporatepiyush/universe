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

; Tests for universe_ml_quant_*: encode->decode fidelity per scheme (f32 exact,
; f16/int8/binary within bound), blob header round-trip via info, all error
; codes, and cos_i8 / hamming distance kernels cross-checked vector vs scalar
; oracle over edge sizes. --bench times the vector kernels vs the scalar twins.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare ptr @malloc(i64)
declare void @free(ptr)
declare float @llvm.fabs.f32(float)

declare i32 @universe_ml_quant_encode(ptr, i64, ptr, i64, i32, ptr)
declare i32 @universe_ml_quant_decode(ptr, i64, ptr, i64, ptr)
declare i32 @universe_ml_quant_info(ptr, i64, ptr, ptr)
declare float @universe_ml_dist_cos_i8(ptr, ptr, i64)
declare float @universe_ml_dist_cos_i8_scalar(ptr, ptr, i64)
declare i64 @universe_ml_dist_hamming(ptr, ptr, i64)
declare i64 @universe_ml_dist_hamming_scalar(ptr, ptr, i64)

@msg.enc   = private unnamed_addr constant [14 x i8] c"encode rc==0\0A\00", align 1
@msg.len   = private unnamed_addr constant [16 x i8] c"encode out_len\0A\00", align 1
@msg.infrc = private unnamed_addr constant [12 x i8] c"info rc==0\0A\00", align 1
@msg.infs  = private unnamed_addr constant [14 x i8] c"info scheme\0A\0A\00", align 1
@msg.infd  = private unnamed_addr constant [12 x i8] c"info dims\0A\0A\00", align 1
@msg.decrc = private unnamed_addr constant [14 x i8] c"decode rc==0\0A\00", align 1
@msg.decd  = private unnamed_addr constant [15 x i8] c"decode dims\0A\0A\0A\00", align 1
@msg.f32   = private unnamed_addr constant [20 x i8] c"f32 exact roundtrip\00", align 1
@msg.f16   = private unnamed_addr constant [14 x i8] c"f16 in bound\0A\00", align 1
@msg.i8b   = private unnamed_addr constant [15 x i8] c"int8 in bound\0A\00", align 1
@msg.bin   = private unnamed_addr constant [14 x i8] c"binary signs\0A\00", align 1
@msg.baddim = private unnamed_addr constant [16 x i8] c"bad dim -> 8\0A\0A\0A\00", align 1
@msg.badsch = private unnamed_addr constant [17 x i8] c"unknown sch -> 8\00", align 1
@msg.small  = private unnamed_addr constant [17 x i8] c"too small -> 8\0A\0A\00", align 1
@msg.encnul = private unnamed_addr constant [15 x i8] c"enc null -> 1\0A\00", align 1
@msg.decsh  = private unnamed_addr constant [16 x i8] c"dec short -> 8\0A\00", align 1
@msg.deccap = private unnamed_addr constant [16 x i8] c"dec cap  -> 8\0A\0A\00", align 1
@msg.decsch = private unnamed_addr constant [16 x i8] c"dec sch  -> 8\0A\0A\00", align 1
@msg.decnul = private unnamed_addr constant [15 x i8] c"dec null -> 1\0A\00", align 1
@msg.infnul = private unnamed_addr constant [15 x i8] c"inf null -> 1\0A\00", align 1
@msg.infsh  = private unnamed_addr constant [16 x i8] c"inf short -> 8\0A\00", align 1
@msg.infsch = private unnamed_addr constant [16 x i8] c"inf sch  -> 8\0A\0A\00", align 1
@msg.cos    = private unnamed_addr constant [16 x i8] c"cos vec==scal\0A\0A\00", align 1
@msg.ham    = private unnamed_addr constant [16 x i8] c"ham vec==scal\0A\0A\00", align 1
@msg.kcos   = private unnamed_addr constant [12 x i8] c"cos known1\0A\00", align 1
@msg.kham   = private unnamed_addr constant [12 x i8] c"ham known3\0A\00", align 1

@lbl.cosv = private unnamed_addr constant [13 x i8] c"cos_i8 vec\0A\0A\00"
@lbl.coss = private unnamed_addr constant [13 x i8] c"cos_i8 scal\0A\00"
@lbl.hamv = private unnamed_addr constant [13 x i8] c"hamming vec\0A\00"
@lbl.hams = private unnamed_addr constant [14 x i8] c"hamming scal\0A\00"

@cv.samp = internal global [16 x double] zeroinitializer, align 8
@cs.samp = internal global [16 x double] zeroinitializer, align 8
@hv.samp = internal global [16 x double] zeroinitializer, align 8
@hs.samp = internal global [16 x double] zeroinitializer, align 8

; codec lengths (>=1; n==0 tested as bad-dim error separately)
@codeclens = private unnamed_addr constant [15 x i64]
  [i64 1, i64 2, i64 3, i64 7, i64 8, i64 15, i64 16, i64 17, i64 63, i64 64,
   i64 100, i64 255, i64 256, i64 384, i64 1000], align 8

; distance lengths (0 and around the 16-lane boundary)
@distlens = private unnamed_addr constant [9 x i64]
  [i64 0, i64 1, i64 2, i64 15, i64 16, i64 17, i64 31, i64 64, i64 1000], align 8

; ===================================================== fill f32 in [-1,1)
define internal void @fillf(ptr %a, i64 %n, ptr %st) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %low = and i64 %r, 65535
  %rf = uitofp i64 %low to float
  %sc = fmul float %rf, 0x3F00000000000000
  %v = fsub float %sc, 1.0
  %p = getelementptr inbounds nuw float, ptr %a, i64 %i
  store float %v, ptr %p, align 4
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; ===================================================== fill random bytes
define internal void @fillb(ptr %a, i64 %n, ptr %st) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %r = call i64 @ut_rand(ptr %st)
  %b = trunc i64 %r to i8
  %p = getelementptr inbounds nuw i8, ptr %a, i64 %i
  store i8 %b, ptr %p, align 1
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; ===================================================== expected encoded length
define internal i64 @explen(i64 %n, i32 %s) {
entry:
  switch i32 %s, label %f32 [ i32 0, label %f32
                              i32 1, label %f16
                              i32 2, label %i8s
                              i32 3, label %bin ]
f32:
  %e0 = shl i64 %n, 2
  %r0 = add i64 %e0, 5
  ret i64 %r0
f16:
  %e1 = shl i64 %n, 1
  %r1 = add i64 %e1, 5
  ret i64 %r1
i8s:
  %r2 = add i64 %n, 9
  ret i64 %r2
bin:
  %e3 = add i64 %n, 7
  %d3 = lshr i64 %e3, 3
  %r3 = add i64 %d3, 5
  ret i64 %r3
}

; ===================================================== f32 exact mismatches
define internal i64 @diff_exact(ptr %p, ptr %q, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %v = phi i64 [ 0, %entry ], [ %vn, %loop ]
  %pp = getelementptr inbounds nuw float, ptr %p, i64 %i
  %pv = load float, ptr %pp, align 4
  %qp = getelementptr inbounds nuw float, ptr %q, i64 %i
  %qv = load float, ptr %qp, align 4
  %ne = fcmp one float %pv, %qv
  %inc = zext i1 %ne to i64
  %vn = add nuw i64 %v, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %vn, %loop ]
  ret i64 %r
}

; ===================================================== abs-bound mismatches
define internal i64 @diff_bound(ptr %dec, ptr %src, i64 %n, float %bound) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %v = phi i64 [ 0, %entry ], [ %vn, %loop ]
  %dp = getelementptr inbounds nuw float, ptr %dec, i64 %i
  %dv = load float, ptr %dp, align 4
  %sp = getelementptr inbounds nuw float, ptr %src, i64 %i
  %sv = load float, ptr %sp, align 4
  %d = fsub float %dv, %sv
  %ad = call float @llvm.fabs.f32(float %d)
  %bad = fcmp ogt float %ad, %bound
  %inc = zext i1 %bad to i64
  %vn = add nuw i64 %v, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %vn, %loop ]
  ret i64 %r
}

; ===================================================== binary sign mismatches
; dec[i] must be +1 when src[i] >= 0, else -1.
define internal i64 @diff_sign(ptr %dec, ptr %src, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %in, %loop ]
  %v = phi i64 [ 0, %entry ], [ %vn, %loop ]
  %sp = getelementptr inbounds nuw float, ptr %src, i64 %i
  %sv = load float, ptr %sp, align 4
  %pos = fcmp oge float %sv, 0.0
  %want = select i1 %pos, float 1.0, float -1.0
  %dp = getelementptr inbounds nuw float, ptr %dec, i64 %i
  %dv = load float, ptr %dp, align 4
  %ne = fcmp one float %dv, %want
  %inc = zext i1 %ne to i64
  %vn = add nuw i64 %v, %inc
  %in = add nuw i64 %i, 1
  %more = icmp ult i64 %in, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %vn, %loop ]
  ret i64 %r
}

; ===================================================== close float (v ~= s)
define internal void @ckclose(float %v, float %s, ptr %msg) {
entry:
  %d = fsub float %v, %s
  %ad = call float @llvm.fabs.f32(float %d)
  %as = call float @llvm.fabs.f32(float %s)
  %b0 = fadd float %as, 1.0
  %b = fmul float %b0, 0x3EB0000000000000
  %ok = fcmp ole float %ad, %b
  call void @ut_check(i1 %ok, ptr %msg)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %st = alloca i64, align 8
  store i64 88172645463325252, ptr %st, align 8
  %oscheme = alloca i32, align 4
  %odims = alloca i64, align 8
  %olen = alloca i64, align 8
  ; buffers: src/dec up to 1000 floats; blob up to 5+4000; i8 up to 1000
  %src = call ptr @malloc(i64 4096)
  %dec = call ptr @malloc(i64 4096)
  %blob = call ptr @malloc(i64 8192)
  %ai = call ptr @malloc(i64 1024)
  %bi = call ptr @malloc(i64 1024)

  ; ============================ codec roundtrip over lengths x schemes
  br label %clen
clen:
  %ci = phi i64 [ 0, %entry ], [ %cin, %clentail ]
  %clp = getelementptr inbounds nuw [15 x i64], ptr @codeclens, i64 0, i64 %ci
  %n = load i64, ptr %clp, align 8
  call void @fillf(ptr %src, i64 %n, ptr %st)
  br label %sch
sch:
  %s = phi i32 [ 0, %clen ], [ %sn, %schtail ]
  %erc = call i32 @universe_ml_quant_encode(ptr %blob, i64 8192, ptr %src, i64 %n, i32 %s, ptr %olen)
  %erc0 = icmp eq i32 %erc, 0
  call void @ut_check(i1 %erc0, ptr @msg.enc)
  %ol = load i64, ptr %olen, align 8
  %exp = call i64 @explen(i64 %n, i32 %s)
  call void @ut_check_eq(i64 %ol, i64 %exp, ptr @msg.len)

  ; info round-trip
  %irc = call i32 @universe_ml_quant_info(ptr %blob, i64 %ol, ptr %oscheme, ptr %odims)
  %irc0 = icmp eq i32 %irc, 0
  call void @ut_check(i1 %irc0, ptr @msg.infrc)
  %isch = load i32, ptr %oscheme, align 4
  %isch64 = zext i32 %isch to i64
  %s64 = zext i32 %s to i64
  call void @ut_check_eq(i64 %isch64, i64 %s64, ptr @msg.infs)
  %idim = load i64, ptr %odims, align 8
  call void @ut_check_eq(i64 %idim, i64 %n, ptr @msg.infd)

  ; decode
  %drc = call i32 @universe_ml_quant_decode(ptr %dec, i64 %n, ptr %blob, i64 %ol, ptr %odims)
  %drc0 = icmp eq i32 %drc, 0
  call void @ut_check(i1 %drc0, ptr @msg.decrc)
  %ddim = load i64, ptr %odims, align 8
  call void @ut_check_eq(i64 %ddim, i64 %n, ptr @msg.decd)

  ; per-scheme fidelity
  switch i32 %s, label %schtail [ i32 0, label %chk.f32
                                  i32 1, label %chk.f16
                                  i32 2, label %chk.i8
                                  i32 3, label %chk.bin ]
chk.f32:
  %vf32 = call i64 @diff_exact(ptr %dec, ptr %src, i64 %n)
  call void @ut_check_eq(i64 %vf32, i64 0, ptr @msg.f32)
  br label %schtail
chk.f16:
  %vf16 = call i64 @diff_bound(ptr %dec, ptr %src, i64 %n, float 0x3F90000000000000)
  call void @ut_check_eq(i64 %vf16, i64 0, ptr @msg.f16)
  br label %schtail
chk.i8:
  %vi8 = call i64 @diff_bound(ptr %dec, ptr %src, i64 %n, float 0x3FA0000000000000)
  call void @ut_check_eq(i64 %vi8, i64 0, ptr @msg.i8b)
  br label %schtail
chk.bin:
  %vbin = call i64 @diff_sign(ptr %dec, ptr %src, i64 %n)
  call void @ut_check_eq(i64 %vbin, i64 0, ptr @msg.bin)
  br label %schtail
schtail:
  %sn = add nuw i32 %s, 1
  %smore = icmp ult i32 %sn, 4
  br i1 %smore, label %sch, label %clentail
clentail:
  %cin = add nuw i64 %ci, 1
  %cmore = icmp ult i64 %cin, 15
  br i1 %cmore, label %clen, label %errs

; ============================ error paths
errs:
  ; bad dim (0) -> 8
  %ebd = call i32 @universe_ml_quant_encode(ptr %blob, i64 8192, ptr %src, i64 0, i32 0, ptr %olen)
  %ebd8 = icmp eq i32 %ebd, 8
  call void @ut_check(i1 %ebd8, ptr @msg.baddim)
  ; unknown scheme (4) -> 8
  %ebs = call i32 @universe_ml_quant_encode(ptr %blob, i64 8192, ptr %src, i64 4, i32 4, ptr %olen)
  %ebs8 = icmp eq i32 %ebs, 8
  call void @ut_check(i1 %ebs8, ptr @msg.badsch)
  ; buffer too small (cap 3, need 21) -> 8
  %esm = call i32 @universe_ml_quant_encode(ptr %blob, i64 3, ptr %src, i64 4, i32 0, ptr %olen)
  %esm8 = icmp eq i32 %esm, 8
  call void @ut_check(i1 %esm8, ptr @msg.small)
  ; null dst -> 1
  %enu = call i32 @universe_ml_quant_encode(ptr null, i64 8192, ptr %src, i64 4, i32 0, ptr %olen)
  %enu1 = icmp eq i32 %enu, 1
  call void @ut_check(i1 %enu1, ptr @msg.encnul)

  ; make a valid int8 blob (dims=8) for decode/info error crafting
  %vrc = call i32 @universe_ml_quant_encode(ptr %blob, i64 8192, ptr %src, i64 8, i32 2, ptr %olen)
  %vol = load i64, ptr %olen, align 8
  ; decode short (slen=4) -> 8
  %dsh = call i32 @universe_ml_quant_decode(ptr %dec, i64 8, ptr %blob, i64 4, ptr %odims)
  %dsh8 = icmp eq i32 %dsh, 8
  call void @ut_check(i1 %dsh8, ptr @msg.decsh)
  ; decode cap too small (dcapel=4 < dims=8) -> 8
  %dcp = call i32 @universe_ml_quant_decode(ptr %dec, i64 4, ptr %blob, i64 %vol, ptr %odims)
  %dcp8 = icmp eq i32 %dcp, 8
  call void @ut_check(i1 %dcp8, ptr @msg.deccap)
  ; decode unknown scheme: corrupt blob[0] = 9 -> 8
  store i8 9, ptr %blob, align 1
  %dsc = call i32 @universe_ml_quant_decode(ptr %dec, i64 8, ptr %blob, i64 %vol, ptr %odims)
  %dsc8 = icmp eq i32 %dsc, 8
  call void @ut_check(i1 %dsc8, ptr @msg.decsch)
  ; info unknown scheme (still corrupted) -> 8
  %isc = call i32 @universe_ml_quant_info(ptr %blob, i64 %vol, ptr %oscheme, ptr %odims)
  %isc8 = icmp eq i32 %isc, 8
  call void @ut_check(i1 %isc8, ptr @msg.infsch)
  ; decode null -> 1
  %dnu = call i32 @universe_ml_quant_decode(ptr null, i64 8, ptr %blob, i64 %vol, ptr %odims)
  %dnu1 = icmp eq i32 %dnu, 1
  call void @ut_check(i1 %dnu1, ptr @msg.decnul)
  ; info null -> 1
  %inu = call i32 @universe_ml_quant_info(ptr %blob, i64 %vol, ptr null, ptr %odims)
  %inu1 = icmp eq i32 %inu, 1
  call void @ut_check(i1 %inu1, ptr @msg.infnul)
  ; info short -> 8
  %ish = call i32 @universe_ml_quant_info(ptr %blob, i64 4, ptr %oscheme, ptr %odims)
  %ish8 = icmp eq i32 %ish, 8
  call void @ut_check(i1 %ish8, ptr @msg.infsh)

  ; ============================ distance kernels vec == scalar
  br label %dlen
dlen:
  %di = phi i64 [ 0, %errs ], [ %din, %dlen.tail ]
  %dlp = getelementptr inbounds nuw [9 x i64], ptr @distlens, i64 0, i64 %di
  %dn = load i64, ptr %dlp, align 8
  call void @fillb(ptr %ai, i64 %dn, ptr %st)
  call void @fillb(ptr %bi, i64 %dn, ptr %st)
  ; cos_i8
  %cv = call float @universe_ml_dist_cos_i8(ptr %ai, ptr %bi, i64 %dn)
  %cs = call float @universe_ml_dist_cos_i8_scalar(ptr %ai, ptr %bi, i64 %dn)
  call void @ckclose(float %cv, float %cs, ptr @msg.cos)
  ; hamming (bytes)
  %hv = call i64 @universe_ml_dist_hamming(ptr %ai, ptr %bi, i64 %dn)
  %hs = call i64 @universe_ml_dist_hamming_scalar(ptr %ai, ptr %bi, i64 %dn)
  call void @ut_check_eq(i64 %hv, i64 %hs, ptr @msg.ham)
  br label %dlen.tail
dlen.tail:
  %din = add nuw i64 %di, 1
  %dmore = icmp ult i64 %din, 9
  br i1 %dmore, label %dlen, label %known

; ============================ known-answer distance
known:
  ; a = [3,4,0,...16 vals], b same -> cosine of a vector with itself = 1.
  ; fill ai with a small deterministic pattern of 20 bytes, bi = ai.
  br label %kfill
kfill:
  %ki = phi i64 [ 0, %known ], [ %kin, %kfill ]
  %kb = trunc i64 %ki to i8
  %kv = add i8 %kb, 1
  %kap = getelementptr inbounds nuw i8, ptr %ai, i64 %ki
  store i8 %kv, ptr %kap, align 1
  %kbp = getelementptr inbounds nuw i8, ptr %bi, i64 %ki
  store i8 %kv, ptr %kbp, align 1
  %kin = add nuw i64 %ki, 1
  %kmore = icmp ult i64 %kin, 20
  br i1 %kmore, label %kfill, label %kcheck
kcheck:
  ; cos(a,a) == 1
  %kcos = call float @universe_ml_dist_cos_i8(ptr %ai, ptr %bi, i64 20)
  call void @ckclose(float %kcos, float 1.0, ptr @msg.kcos)
  ; hamming(a,a) == 0; flip 3 bits in bi[0] -> distance 3
  %b0 = load i8, ptr %bi, align 1
  %b0x = xor i8 %b0, 7            ; low 3 bits differ
  store i8 %b0x, ptr %bi, align 1
  %kham = call i64 @universe_ml_dist_hamming(ptr %ai, ptr %bi, i64 20)
  call void @ut_check_eq(i64 %kham, i64 3, ptr @msg.kham)

  ; ============================ bench
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin
bench:
  call void @run_bench(ptr %ai, ptr %bi)
  br label %fin

fin:
  call void @free(ptr %src)
  call void @free(ptr %dec)
  call void @free(ptr %blob)
  call void @free(ptr %ai)
  call void @free(ptr %bi)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; ============================================================ bench (n=1024)
define internal void @run_bench(ptr %scratchi, ptr %scratch2) {
entry:
  %N = add i64 0, 1024
  %a = call ptr @malloc(i64 %N)
  %b = call ptr @malloc(i64 %N)
  %st = alloca i64, align 8
  store i64 12345678, ptr %st, align 8
  call void @fillb(ptr %a, i64 %N, ptr %st)
  call void @fillb(ptr %b, i64 %N, ptr %st)
  %iters = add i64 0, 20000

  ; --- cos_i8 vector ---
  br label %cv.rep.head
cv.rep.head:
  %cv.rep = phi i64 [ 0, %entry ], [ %cv.rep.n, %cv.rep.cont ]
  %cv.t0 = call double @ut_now_sec()
  br label %cv.loop
cv.loop:
  %cvi = phi i64 [ 0, %cv.rep.head ], [ %cvin, %cv.loop ]
  %cvacc = phi float [ 0.0, %cv.rep.head ], [ %cvacn, %cv.loop ]
  %cvr = call float @universe_ml_dist_cos_i8(ptr %a, ptr %b, i64 %N)
  %cvacn = fadd float %cvacc, %cvr
  %cvin = add nuw i64 %cvi, 1
  %cvm = icmp ult i64 %cvin, %iters
  br i1 %cvm, label %cv.loop, label %cv.done
cv.done:
  %cv.t1 = call double @ut_now_sec()
  store volatile float %cvacn, ptr %scratchi, align 1
  %cv.dt = fsub double %cv.t1, %cv.t0
  %cv.warm = icmp eq i64 %cv.rep, 0
  br i1 %cv.warm, label %cv.rep.cont, label %cv.rep.store
cv.rep.store:
  %cv.idx = sub i64 %cv.rep, 1
  %cv.sp = getelementptr inbounds [16 x double], ptr @cv.samp, i64 0, i64 %cv.idx
  store double %cv.dt, ptr %cv.sp, align 8
  br label %cv.rep.cont
cv.rep.cont:
  %cv.rep.n = add i64 %cv.rep, 1
  %cv.more = icmp ult i64 %cv.rep.n, 17
  br i1 %cv.more, label %cv.rep.head, label %cv.end
cv.end:
  %cv.ops = mul i64 %N, %iters
  call void @ut_report_dist(ptr @cv.samp, i64 16, i64 %cv.ops, ptr @lbl.cosv)

  ; --- cos_i8 scalar ---
  br label %cs.rep.head
cs.rep.head:
  %cs.rep = phi i64 [ 0, %cv.end ], [ %cs.rep.n, %cs.rep.cont ]
  %cs.t0 = call double @ut_now_sec()
  br label %cs.loop
cs.loop:
  %csi = phi i64 [ 0, %cs.rep.head ], [ %csin, %cs.loop ]
  %csacc = phi float [ 0.0, %cs.rep.head ], [ %csacn, %cs.loop ]
  %csr = call float @universe_ml_dist_cos_i8_scalar(ptr %a, ptr %b, i64 %N)
  %csacn = fadd float %csacc, %csr
  %csin = add nuw i64 %csi, 1
  %csm = icmp ult i64 %csin, %iters
  br i1 %csm, label %cs.loop, label %cs.done
cs.done:
  %cs.t1 = call double @ut_now_sec()
  store volatile float %csacn, ptr %scratchi, align 1
  %cs.dt = fsub double %cs.t1, %cs.t0
  %cs.warm = icmp eq i64 %cs.rep, 0
  br i1 %cs.warm, label %cs.rep.cont, label %cs.rep.store
cs.rep.store:
  %cs.idx = sub i64 %cs.rep, 1
  %cs.sp = getelementptr inbounds [16 x double], ptr @cs.samp, i64 0, i64 %cs.idx
  store double %cs.dt, ptr %cs.sp, align 8
  br label %cs.rep.cont
cs.rep.cont:
  %cs.rep.n = add i64 %cs.rep, 1
  %cs.more = icmp ult i64 %cs.rep.n, 17
  br i1 %cs.more, label %cs.rep.head, label %cs.end
cs.end:
  %cs.ops = mul i64 %N, %iters
  call void @ut_report_dist(ptr @cs.samp, i64 16, i64 %cs.ops, ptr @lbl.coss)

  ; --- hamming vector ---
  br label %hv.rep.head
hv.rep.head:
  %hv.rep = phi i64 [ 0, %cs.end ], [ %hv.rep.n, %hv.rep.cont ]
  %hv.t0 = call double @ut_now_sec()
  br label %hv.loop
hv.loop:
  %hvi = phi i64 [ 0, %hv.rep.head ], [ %hvin, %hv.loop ]
  %hvacc = phi i64 [ 0, %hv.rep.head ], [ %hvacn, %hv.loop ]
  %hvr = call i64 @universe_ml_dist_hamming(ptr %a, ptr %b, i64 %N)
  %hvacn = add i64 %hvacc, %hvr
  %hvin = add nuw i64 %hvi, 1
  %hvm = icmp ult i64 %hvin, %iters
  br i1 %hvm, label %hv.loop, label %hv.done
hv.done:
  %hv.t1 = call double @ut_now_sec()
  %hvvol = trunc i64 %hvacn to i8
  store volatile i8 %hvvol, ptr %scratch2, align 1
  %hv.dt = fsub double %hv.t1, %hv.t0
  %hv.warm = icmp eq i64 %hv.rep, 0
  br i1 %hv.warm, label %hv.rep.cont, label %hv.rep.store
hv.rep.store:
  %hv.idx = sub i64 %hv.rep, 1
  %hv.sp = getelementptr inbounds [16 x double], ptr @hv.samp, i64 0, i64 %hv.idx
  store double %hv.dt, ptr %hv.sp, align 8
  br label %hv.rep.cont
hv.rep.cont:
  %hv.rep.n = add i64 %hv.rep, 1
  %hv.more = icmp ult i64 %hv.rep.n, 17
  br i1 %hv.more, label %hv.rep.head, label %hv.end
hv.end:
  %hv.ops = mul i64 %N, %iters
  call void @ut_report_dist(ptr @hv.samp, i64 16, i64 %hv.ops, ptr @lbl.hamv)

  ; --- hamming scalar ---
  br label %hs.rep.head
hs.rep.head:
  %hs.rep = phi i64 [ 0, %hv.end ], [ %hs.rep.n, %hs.rep.cont ]
  %hs.t0 = call double @ut_now_sec()
  br label %hs.loop
hs.loop:
  %hsi = phi i64 [ 0, %hs.rep.head ], [ %hsin, %hs.loop ]
  %hsacc = phi i64 [ 0, %hs.rep.head ], [ %hsacn, %hs.loop ]
  %hsr = call i64 @universe_ml_dist_hamming_scalar(ptr %a, ptr %b, i64 %N)
  %hsacn = add i64 %hsacc, %hsr
  %hsin = add nuw i64 %hsi, 1
  %hsm = icmp ult i64 %hsin, %iters
  br i1 %hsm, label %hs.loop, label %hs.done
hs.done:
  %hs.t1 = call double @ut_now_sec()
  %hsvol = trunc i64 %hsacn to i8
  store volatile i8 %hsvol, ptr %scratch2, align 1
  %hs.dt = fsub double %hs.t1, %hs.t0
  %hs.warm = icmp eq i64 %hs.rep, 0
  br i1 %hs.warm, label %hs.rep.cont, label %hs.rep.store
hs.rep.store:
  %hs.idx = sub i64 %hs.rep, 1
  %hs.sp = getelementptr inbounds [16 x double], ptr @hs.samp, i64 0, i64 %hs.idx
  store double %hs.dt, ptr %hs.sp, align 8
  br label %hs.rep.cont
hs.rep.cont:
  %hs.rep.n = add i64 %hs.rep, 1
  %hs.more = icmp ult i64 %hs.rep.n, 17
  br i1 %hs.more, label %hs.rep.head, label %hs.end
hs.end:
  %hs.ops = mul i64 %N, %iters
  call void @ut_report_dist(ptr @hs.samp, i64 16, i64 %hs.ops, ptr @lbl.hams)

  call void @free(ptr %a)
  call void @free(ptr %b)
  ret void
}
