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

; Tests for Elias-Fano + delta/varint postings (structures_succinct).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)

declare ptr @universe_ds_ef_build(ptr, i64)
declare void @universe_ds_ef_destroy(ptr)
declare i64 @universe_ds_ef_access(ptr, i64)
declare i64 @universe_ds_ef_next_geq(ptr, i64)
declare i64 @universe_ds_ef_size(ptr)
declare i64 @universe_ds_ef_footprint(ptr)
declare i64 @universe_ds_postings_bound(i64)
declare i64 @universe_ds_postings_encode(ptr, i64, ptr)
declare i64 @universe_ds_postings_decode(ptr, i64, ptr, i64)
declare i64 @universe_ds_postings_bound_packed(i64)
declare i64 @universe_ds_postings_encode_packed(ptr, i64, ptr)
declare i64 @universe_ds_postings_decode_packed(ptr, i64, ptr, i64)
declare i64 @universe_ds_postings_decode_packed_scalar(ptr, i64, ptr, i64)
declare i64 @universe_ds_rle_encode(ptr, i64, ptr)
declare i64 @universe_ds_rle_decode(ptr, i64, ptr, i64)

@m.n1        = private constant [16 x i8] c"N=1 access(0)\00\00\00"
@m.n1geq     = private constant [12 x i8] c"N=1 nextgeq\00"
@m.n1sz      = private constant [8 x i8] c"N=1 sz\00\00"
@m.eq        = private constant [16 x i8] c"all-equal acc\00\00\00"
@m.eqgeq     = private constant [16 x i8] c"all-equal geq\00\00\00"
@m.uabove    = private constant [16 x i8] c"U-above-N acc\00\00\00"
@m.empty     = private constant [16 x i8] c"empty ef size\00\00\00"
@m.emptyacc  = private constant [12 x i8] c"empty acc-1\00"
@m.acc       = private constant [16 x i8] c"random access\00\00\00"
@m.geq       = private constant [16 x i8] c"random nextgeq\00\00"
@m.foot      = private constant [16 x i8] c"footprint bound\00"
@m.penc      = private constant [16 x i8] c"postings rtrip\00\00"
@m.pn        = private constant [16 x i8] c"postings count\00\00"
@m.pk        = private constant [16 x i8] c"packed rtrip\00\00\00\00"
@m.pkn       = private constant [16 x i8] c"packed count\00\00\00\00"
@m.pkss      = private constant [16 x i8] c"packed simd=sca\00"
@m.rle       = private constant [12 x i8] c"rle rtrip\00\00\00"
@m.rlen      = private constant [12 x i8] c"rle count\00\00\00"

@lbl.ef_access = private unnamed_addr constant [17 x i8] c"EF access/val ns\00"
@lbl.ef_plain  = private unnamed_addr constant [19 x i8] c"plain array/val ns\00"
@lbl.ef_simd   = private unnamed_addr constant [26 x i8] c"postings decode SIMD /val\00"
@lbl.ef_scalar = private unnamed_addr constant [28 x i8] c"postings decode scalar /val\00"
@ef.efsamp   = internal global [16 x double] zeroinitializer, align 8
@ef.plsamp   = internal global [16 x double] zeroinitializer, align 8
@ef.simdsamp = internal global [16 x double] zeroinitializer, align 8
@ef.scsamp   = internal global [16 x double] zeroinitializer, align 8
@ef.sink     = internal global i64 0, align 8

; ---- reference helpers ---------------------------------------------------

; fill vals[0..n) monotone non-decreasing; returns U (last value).
define i64 @gen_monotone(ptr %vals, i64 %n, ptr %seed, i64 %gapmax) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %ret0, label %loop
ret0:
  ret i64 0
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %r = call i64 @ut_rand(ptr %seed)
  %g = urem i64 %r, %gapmax
  %acc.n = add i64 %acc, %g
  %vp = getelementptr inbounds i64, ptr %vals, i64 %i
  store i64 %acc.n, ptr %vp, align 8
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

; fill ids[0..n) strictly increasing (gap>=1); returns last id.
define i64 @gen_sorted(ptr %ids, i64 %n, ptr %seed, i64 %gapmax) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %ret0, label %loop
ret0:
  ret i64 0
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %r = call i64 @ut_rand(ptr %seed)
  %g0 = urem i64 %r, %gapmax
  %g = add i64 %g0, 1
  %acc.n = add i64 %acc, %g
  %vp = getelementptr inbounds i64, ptr %ids, i64 %i
  store i64 %acc.n, ptr %vp, align 8
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

; linear first value >= x, or -1.
define i64 @lin_geq(ptr %vals, i64 %n, i64 %x) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %none, label %loop
none:
  ret i64 -1
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %vp = getelementptr inbounds i64, ptr %vals, i64 %i
  %v = load i64, ptr %vp, align 8
  %ge = icmp uge i64 %v, %x
  br i1 %ge, label %hit, label %cont
hit:
  ret i64 %v
cont:
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %none
}

; ---- main ----------------------------------------------------------------

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  %fixed = alloca [16 x i64], align 8

  ; ========================= EF: N=1 =========================
  %f0 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 0
  store i64 42, ptr %f0, align 8
  %ef1 = call ptr @universe_ds_ef_build(ptr %fixed, i64 1)
  %a1 = call i64 @universe_ds_ef_access(ptr %ef1, i64 0)
  call void @ut_check_eq(i64 %a1, i64 42, ptr @m.n1)
  %g1a = call i64 @universe_ds_ef_next_geq(ptr %ef1, i64 42)
  call void @ut_check_eq(i64 %g1a, i64 42, ptr @m.n1geq)
  %g1b = call i64 @universe_ds_ef_next_geq(ptr %ef1, i64 43)
  call void @ut_check_eq(i64 %g1b, i64 -1, ptr @m.n1geq)
  %g1c = call i64 @universe_ds_ef_next_geq(ptr %ef1, i64 0)
  call void @ut_check_eq(i64 %g1c, i64 42, ptr @m.n1geq)
  %sz1 = call i64 @universe_ds_ef_size(ptr %ef1)
  call void @ut_check_eq(i64 %sz1, i64 1, ptr @m.n1sz)
  call void @universe_ds_ef_destroy(ptr %ef1)

  ; ========================= EF: all-equal =========================
  br label %eqfill
eqfill:
  %ei = phi i64 [ 0, %entry ], [ %ei.n, %eqfill ]
  %eip = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 %ei
  store i64 7, ptr %eip, align 8
  %ei.n = add i64 %ei, 1
  %eqmore = icmp ult i64 %ei.n, 5
  br i1 %eqmore, label %eqfill, label %eqtest
eqtest:
  %efe = call ptr @universe_ds_ef_build(ptr %fixed, i64 5)
  br label %eqacc
eqacc:
  %qi = phi i64 [ 0, %eqtest ], [ %qi.n, %eqacc ]
  %qbad = phi i64 [ 0, %eqtest ], [ %qbad.n, %eqacc ]
  %qa = call i64 @universe_ds_ef_access(ptr %efe, i64 %qi)
  %qne = icmp ne i64 %qa, 7
  %qinc = zext i1 %qne to i64
  %qbad.n = add i64 %qbad, %qinc
  %qi.n = add i64 %qi, 1
  %qmore = icmp ult i64 %qi.n, 5
  br i1 %qmore, label %eqacc, label %eqdone
eqdone:
  call void @ut_check_eq(i64 %qbad.n, i64 0, ptr @m.eq)
  %eg0 = call i64 @universe_ds_ef_next_geq(ptr %efe, i64 7)
  call void @ut_check_eq(i64 %eg0, i64 7, ptr @m.eqgeq)
  %eg1 = call i64 @universe_ds_ef_next_geq(ptr %efe, i64 0)
  call void @ut_check_eq(i64 %eg1, i64 7, ptr @m.eqgeq)
  %eg2 = call i64 @universe_ds_ef_next_geq(ptr %efe, i64 8)
  call void @ut_check_eq(i64 %eg2, i64 -1, ptr @m.eqgeq)
  call void @universe_ds_ef_destroy(ptr %efe)

  ; ==================== EF: U just above N ====================
  ; vals = 1,2,3,4,5,6,7,9  (N=8, U=9)
  %u0 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 0
  store i64 1, ptr %u0, align 8
  %u1 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 1
  store i64 2, ptr %u1, align 8
  %u2 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 2
  store i64 3, ptr %u2, align 8
  %u3 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 3
  store i64 4, ptr %u3, align 8
  %u4 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 4
  store i64 5, ptr %u4, align 8
  %u5 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 5
  store i64 6, ptr %u5, align 8
  %u6 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 6
  store i64 7, ptr %u6, align 8
  %u7 = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 7
  store i64 9, ptr %u7, align 8
  %efu = call ptr @universe_ds_ef_build(ptr %fixed, i64 8)
  br label %uacc
uacc:
  %ui = phi i64 [ 0, %eqdone ], [ %ui.n, %uacc ]
  %ubad = phi i64 [ 0, %eqdone ], [ %ubad.n, %uacc ]
  %uref = getelementptr inbounds [16 x i64], ptr %fixed, i64 0, i64 %ui
  %urv = load i64, ptr %uref, align 8
  %uav = call i64 @universe_ds_ef_access(ptr %efu, i64 %ui)
  %une = icmp ne i64 %uav, %urv
  %uinc = zext i1 %une to i64
  %ubad.n = add i64 %ubad, %uinc
  %ui.n = add i64 %ui, 1
  %umore = icmp ult i64 %ui.n, 8
  br i1 %umore, label %uacc, label %udone
udone:
  call void @ut_check_eq(i64 %ubad.n, i64 0, ptr @m.uabove)
  %ug = call i64 @universe_ds_ef_next_geq(ptr %efu, i64 8)
  call void @ut_check_eq(i64 %ug, i64 9, ptr @m.uabove)
  call void @universe_ds_ef_destroy(ptr %efu)

  ; ========================= EF: empty =========================
  %efz = call ptr @universe_ds_ef_build(ptr null, i64 0)
  %szz = call i64 @universe_ds_ef_size(ptr %efz)
  call void @ut_check_eq(i64 %szz, i64 0, ptr @m.empty)
  %azz = call i64 @universe_ds_ef_access(ptr %efz, i64 0)
  call void @ut_check_eq(i64 %azz, i64 -1, ptr @m.emptyacc)
  %gzz = call i64 @universe_ds_ef_next_geq(ptr %efz, i64 5)
  call void @ut_check_eq(i64 %gzz, i64 -1, ptr @m.emptyacc)
  call void @universe_ds_ef_destroy(ptr %efz)

  ; ==================== EF: large random, access-all ====================
  ; N = 1<<20, small gaps (dense) so footprint is compact.
  %NBIG = add i64 0, 1048576
  %bigbytes = shl i64 %NBIG, 3
  %valsbig = call ptr @malloc(i64 %bigbytes)
  %Ubig = call i64 @gen_monotone(ptr %valsbig, i64 %NBIG, ptr %seed, i64 64)
  %efbig = call ptr @universe_ds_ef_build(ptr %valsbig, i64 %NBIG)
  br label %bacc
bacc:
  %bi = phi i64 [ 0, %udone ], [ %bi.n, %bacc ]
  %bbad = phi i64 [ 0, %udone ], [ %bbad.n, %bacc ]
  %brefp = getelementptr inbounds i64, ptr %valsbig, i64 %bi
  %brv = load i64, ptr %brefp, align 8
  %bav = call i64 @universe_ds_ef_access(ptr %efbig, i64 %bi)
  %bne = icmp ne i64 %bav, %brv
  %binc = zext i1 %bne to i64
  %bbad.n = add i64 %bbad, %binc
  %bi.n = add i64 %bi, 1
  %bmore = icmp ult i64 %bi.n, %NBIG
  br i1 %bmore, label %bacc, label %bdone
bdone:
  call void @ut_check_eq(i64 %bbad.n, i64 0, ptr @m.acc)
  ; footprint must be more compact than a plain u64 array (dense sequence).
  %foot = call i64 @universe_ds_ef_footprint(ptr %efbig)
  %compact = icmp ult i64 %foot, %bigbytes
  call void @ut_check(i1 %compact, ptr @m.foot)

  ; ==================== EF: next_geq cross-check (medium) ====================
  %NMED = add i64 0, 5000
  %medbytes = shl i64 %NMED, 3
  %valsmed = call ptr @malloc(i64 %medbytes)
  %Umed = call i64 @gen_monotone(ptr %valsmed, i64 %NMED, ptr %seed, i64 16)
  %efmed = call ptr @universe_ds_ef_build(ptr %valsmed, i64 %NMED)
  %xmax = add i64 %Umed, 4
  br label %gloop
gloop:
  %gx = phi i64 [ 0, %bdone ], [ %gx.n, %gloop ]
  %gbad = phi i64 [ 0, %bdone ], [ %gbad.n, %gloop ]
  %ev = call i64 @universe_ds_ef_next_geq(ptr %efmed, i64 %gx)
  %lv = call i64 @lin_geq(ptr %valsmed, i64 %NMED, i64 %gx)
  %gne = icmp ne i64 %ev, %lv
  %ginc = zext i1 %gne to i64
  %gbad.n = add i64 %gbad, %ginc
  %gx.n = add i64 %gx, 1
  %gmore = icmp ule i64 %gx.n, %xmax
  br i1 %gmore, label %gloop, label %gdone
gdone:
  call void @ut_check_eq(i64 %gbad.n, i64 0, ptr @m.geq)
  call void @universe_ds_ef_destroy(ptr %efmed)
  call void @free(ptr %valsmed)

  ; ==================== Postings: LEB128 delta round-trip ====================
  ; reuse valsbig region as scratch; make a sorted id list.
  %NP = add i64 0, 100000
  %idsbytes = shl i64 %NP, 3
  %ids = call ptr @malloc(i64 %idsbytes)
  %lastid = call i64 @gen_sorted(ptr %ids, i64 %NP, ptr %seed, i64 1000)
  %pbound = call i64 @universe_ds_postings_bound(i64 %NP)
  %pdst = call ptr @malloc(i64 %pbound)
  %pout = call ptr @malloc(i64 %idsbytes)
  %penc = call i64 @universe_ds_postings_encode(ptr %ids, i64 %NP, ptr %pdst)
  %pdn = call i64 @universe_ds_postings_decode(ptr %pdst, i64 %penc, ptr %pout, i64 %NP)
  call void @ut_check_eq(i64 %pdn, i64 %NP, ptr @m.pn)
  br label %pcmp
pcmp:
  %pi = phi i64 [ 0, %gdone ], [ %pi.n, %pcmp ]
  %pbad = phi i64 [ 0, %gdone ], [ %pbad.n, %pcmp ]
  %pa = getelementptr inbounds i64, ptr %ids, i64 %pi
  %pav = load i64, ptr %pa, align 8
  %pb = getelementptr inbounds i64, ptr %pout, i64 %pi
  %pbv = load i64, ptr %pb, align 8
  %pne = icmp ne i64 %pav, %pbv
  %pinc = zext i1 %pne to i64
  %pbad.n = add i64 %pbad, %pinc
  %pi.n = add i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, %NP
  br i1 %pmore, label %pcmp, label %pdonecmp
pdonecmp:
  call void @ut_check_eq(i64 %pbad.n, i64 0, ptr @m.penc)

  ; ==================== Postings: packed (SIMD) round-trip ====================
  %kbound = call i64 @universe_ds_postings_bound_packed(i64 %NP)
  %kdst = call ptr @malloc(i64 %kbound)
  %kout = call ptr @malloc(i64 %idsbytes)
  %kout2 = call ptr @malloc(i64 %idsbytes)
  %kenc = call i64 @universe_ds_postings_encode_packed(ptr %ids, i64 %NP, ptr %kdst)
  %kdn = call i64 @universe_ds_postings_decode_packed(ptr %kdst, i64 %kenc, ptr %kout, i64 %NP)
  call void @ut_check_eq(i64 %kdn, i64 %NP, ptr @m.pkn)
  %kdn2 = call i64 @universe_ds_postings_decode_packed_scalar(ptr %kdst, i64 %kenc, ptr %kout2, i64 %NP)
  call void @ut_check_eq(i64 %kdn2, i64 %NP, ptr @m.pkn)
  br label %kcmp
kcmp:
  %ki = phi i64 [ 0, %pdonecmp ], [ %ki.n, %kcmp ]
  %kbad = phi i64 [ 0, %pdonecmp ], [ %kbad.n, %kcmp ]
  %kss = phi i64 [ 0, %pdonecmp ], [ %kss.n, %kcmp ]
  %kra = getelementptr inbounds i64, ptr %ids, i64 %ki
  %krav = load i64, ptr %kra, align 8
  %ksa = getelementptr inbounds i64, ptr %kout, i64 %ki
  %ksav = load i64, ptr %ksa, align 8
  %kca = getelementptr inbounds i64, ptr %kout2, i64 %ki
  %kcav = load i64, ptr %kca, align 8
  %kne = icmp ne i64 %krav, %ksav
  %kinc = zext i1 %kne to i64
  %kbad.n = add i64 %kbad, %kinc
  %kssne = icmp ne i64 %ksav, %kcav
  %kssinc = zext i1 %kssne to i64
  %kss.n = add i64 %kss, %kssinc
  %ki.n = add i64 %ki, 1
  %kmore = icmp ult i64 %ki.n, %NP
  br i1 %kmore, label %kcmp, label %kdonecmp
kdonecmp:
  call void @ut_check_eq(i64 %kbad.n, i64 0, ptr @m.pk)
  call void @ut_check_eq(i64 %kss.n, i64 0, ptr @m.pkss)

  ; ==================== Postings edge cases ====================
  ; empty
  %e0enc = call i64 @universe_ds_postings_encode(ptr %ids, i64 0, ptr %pdst)
  call void @ut_check_eq(i64 %e0enc, i64 0, ptr @m.pn)
  %e0dec = call i64 @universe_ds_postings_decode(ptr %pdst, i64 0, ptr %pout, i64 %NP)
  call void @ut_check_eq(i64 %e0dec, i64 0, ptr @m.pn)
  ; single + dense-consecutive + big-gaps via dedicated small arrays
  %edge = alloca [8 x i64], align 8
  %ed0 = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 0
  store i64 123456789, ptr %ed0, align 8
  call void @edge_check(ptr %edge, i64 1)
  ; dense consecutive 0..7
  br label %densefill
densefill:
  %di = phi i64 [ 0, %kdonecmp ], [ %di.n, %densefill ]
  %dip = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 %di
  store i64 %di, ptr %dip, align 8
  %di.n = add i64 %di, 1
  %dmore = icmp ult i64 %di.n, 8
  br i1 %dmore, label %densefill, label %denseck
denseck:
  call void @edge_check(ptr %edge, i64 8)
  ; big gaps
  %bg0 = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 0
  store i64 0, ptr %bg0, align 8
  %bg1 = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 1
  store i64 1000000, ptr %bg1, align 8
  %bg2 = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 2
  store i64 4000000000, ptr %bg2, align 8
  %bg3 = getelementptr inbounds [8 x i64], ptr %edge, i64 0, i64 3
  store i64 4000000005, ptr %bg3, align 8
  call void @edge_check(ptr %edge, i64 4)

  ; ==================== RLE round-trip ====================
  %rlesrc = call ptr @malloc(i64 4096)
  %rledst = call ptr @malloc(i64 8192)
  %rleout = call ptr @malloc(i64 4096)
  br label %rlefill
rlefill:
  %ri = phi i64 [ 0, %denseck ], [ %ri.n, %rlefill ]
  %rval64 = udiv i64 %ri, 40
  %rval = trunc i64 %rval64 to i8
  %rp = getelementptr inbounds i8, ptr %rlesrc, i64 %ri
  store i8 %rval, ptr %rp, align 1
  %ri.n = add i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, 4096
  br i1 %rmore, label %rlefill, label %rletest
rletest:
  %renc = call i64 @universe_ds_rle_encode(ptr %rlesrc, i64 4096, ptr %rledst)
  %rdn = call i64 @universe_ds_rle_decode(ptr %rledst, i64 %renc, ptr %rleout, i64 4096)
  call void @ut_check_eq(i64 %rdn, i64 4096, ptr @m.rlen)
  br label %rlecmp
rlecmp:
  %ci = phi i64 [ 0, %rletest ], [ %ci.n, %rlecmp ]
  %cbad = phi i64 [ 0, %rletest ], [ %cbad.n, %rlecmp ]
  %csp = getelementptr inbounds i8, ptr %rlesrc, i64 %ci
  %csv = load i8, ptr %csp, align 1
  %cop = getelementptr inbounds i8, ptr %rleout, i64 %ci
  %cov = load i8, ptr %cop, align 1
  %cne = icmp ne i8 %csv, %cov
  %cinc = zext i1 %cne to i64
  %cbad.n = add i64 %cbad, %cinc
  %ci.n = add i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, 4096
  br i1 %cmore, label %rlecmp, label %rledone
rledone:
  call void @ut_check_eq(i64 %cbad.n, i64 0, ptr @m.rle)
  call void @free(ptr %rlesrc)
  call void @free(ptr %rledst)
  call void @free(ptr %rleout)

  ; ==================== bench (optional) ====================
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %cleanup

bench:
  ; ---- EF random access distribution (read-only) ----
  br label %efrep
efrep:
  %efrep.i = phi i64 [ 0, %bench ], [ %efrep.n, %efrep.next ]
  %eft0 = call double @ut_now_sec()
  %efr = call i64 @bench_ef(ptr %efbig, i64 %NBIG)
  %eft1 = call double @ut_now_sec()
  store volatile i64 %efr, ptr @ef.sink, align 8
  %efdt = fsub double %eft1, %eft0
  %efwarm = icmp eq i64 %efrep.i, 0
  br i1 %efwarm, label %efrep.next, label %efrep.store
efrep.store:
  %efsidx = sub i64 %efrep.i, 1
  %efsp = getelementptr inbounds double, ptr @ef.efsamp, i64 %efsidx
  store double %efdt, ptr %efsp, align 8
  br label %efrep.next
efrep.next:
  %efrep.n = add nuw nsw i64 %efrep.i, 1
  %efrep.more = icmp ult i64 %efrep.n, 17
  br i1 %efrep.more, label %efrep, label %efreport
efreport:
  call void @ut_report_dist(ptr @ef.efsamp, i64 16, i64 %NBIG, ptr @lbl.ef_access)
  br label %plrep

  ; ---- plain array sum distribution ----
plrep:
  %plrep.i = phi i64 [ 0, %efreport ], [ %plrep.n, %plrep.next ]
  %plt0 = call double @ut_now_sec()
  %plr = call i64 @bench_plain(ptr %valsbig, i64 %NBIG)
  %plt1 = call double @ut_now_sec()
  store volatile i64 %plr, ptr @ef.sink, align 8
  %pldt = fsub double %plt1, %plt0
  %plwarm = icmp eq i64 %plrep.i, 0
  br i1 %plwarm, label %plrep.next, label %plrep.store
plrep.store:
  %plsidx = sub i64 %plrep.i, 1
  %plsp = getelementptr inbounds double, ptr @ef.plsamp, i64 %plsidx
  store double %pldt, ptr %plsp, align 8
  br label %plrep.next
plrep.next:
  %plrep.n = add nuw nsw i64 %plrep.i, 1
  %plrep.more = icmp ult i64 %plrep.n, 17
  br i1 %plrep.more, label %plrep, label %plreport
plreport:
  call void @ut_report_dist(ptr @ef.plsamp, i64 16, i64 %NBIG, ptr @lbl.ef_plain)
  br label %sdrep

  ; ---- postings decode: SIMD distribution ----
sdrep:
  %sdrep.i = phi i64 [ 0, %plreport ], [ %sdrep.n, %sdrep.next ]
  %sdt0 = call double @ut_now_sec()
  %sdr = call i64 @universe_ds_postings_decode_packed(ptr %kdst, i64 %kenc, ptr %kout, i64 %NP)
  %sdt1 = call double @ut_now_sec()
  store volatile i64 %sdr, ptr @ef.sink, align 8
  %sddt = fsub double %sdt1, %sdt0
  %sdwarm = icmp eq i64 %sdrep.i, 0
  br i1 %sdwarm, label %sdrep.next, label %sdrep.store
sdrep.store:
  %sdsidx = sub i64 %sdrep.i, 1
  %sdsp = getelementptr inbounds double, ptr @ef.simdsamp, i64 %sdsidx
  store double %sddt, ptr %sdsp, align 8
  br label %sdrep.next
sdrep.next:
  %sdrep.n = add nuw nsw i64 %sdrep.i, 1
  %sdrep.more = icmp ult i64 %sdrep.n, 17
  br i1 %sdrep.more, label %sdrep, label %sdreport
sdreport:
  call void @ut_report_dist(ptr @ef.simdsamp, i64 16, i64 %NP, ptr @lbl.ef_simd)
  br label %screp

  ; ---- postings decode: scalar distribution ----
screp:
  %screp.i = phi i64 [ 0, %sdreport ], [ %screp.n, %screp.next ]
  %sct0 = call double @ut_now_sec()
  %scr = call i64 @universe_ds_postings_decode_packed_scalar(ptr %kdst, i64 %kenc, ptr %kout2, i64 %NP)
  %sct1 = call double @ut_now_sec()
  store volatile i64 %scr, ptr @ef.sink, align 8
  %scdt = fsub double %sct1, %sct0
  %scwarm = icmp eq i64 %screp.i, 0
  br i1 %scwarm, label %screp.next, label %screp.store
screp.store:
  %scsidx = sub i64 %screp.i, 1
  %scsp = getelementptr inbounds double, ptr @ef.scsamp, i64 %scsidx
  store double %scdt, ptr %scsp, align 8
  br label %screp.next
screp.next:
  %screp.n = add nuw nsw i64 %screp.i, 1
  %screp.more = icmp ult i64 %screp.n, 17
  br i1 %screp.more, label %screp, label %screport
screport:
  call void @ut_report_dist(ptr @ef.scsamp, i64 16, i64 %NP, ptr @lbl.ef_scalar)
  br label %cleanup

cleanup:
  call void @universe_ds_ef_destroy(ptr %efbig)
  call void @free(ptr %valsbig)
  call void @free(ptr %ids)
  call void @free(ptr %pdst)
  call void @free(ptr %pout)
  call void @free(ptr %kdst)
  call void @free(ptr %kout)
  call void @free(ptr %kout2)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; encode both codecs for a small array and confirm both round-trip.
define void @edge_check(ptr %ids, i64 %n) {
entry:
  %bound = call i64 @universe_ds_postings_bound(i64 %n)
  %kbound = call i64 @universe_ds_postings_bound_packed(i64 %n)
  %dst = call ptr @malloc(i64 %bound)
  %kdst = call ptr @malloc(i64 %kbound)
  %obytes = shl i64 %n, 3
  %o1 = call ptr @malloc(i64 %obytes)
  %o2 = call ptr @malloc(i64 %obytes)
  %enc = call i64 @universe_ds_postings_encode(ptr %ids, i64 %n, ptr %dst)
  %dn = call i64 @universe_ds_postings_decode(ptr %dst, i64 %enc, ptr %o1, i64 %n)
  %kenc = call i64 @universe_ds_postings_encode_packed(ptr %ids, i64 %n, ptr %kdst)
  %kdn = call i64 @universe_ds_postings_decode_packed(ptr %kdst, i64 %kenc, ptr %o2, i64 %n)
  br label %cmp
cmp:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cmp ]
  %bad = phi i64 [ 0, %entry ], [ %bad.n, %cmp ]
  %rp = getelementptr inbounds i64, ptr %ids, i64 %i
  %rv = load i64, ptr %rp, align 8
  %a1p = getelementptr inbounds i64, ptr %o1, i64 %i
  %a1v = load i64, ptr %a1p, align 8
  %a2p = getelementptr inbounds i64, ptr %o2, i64 %i
  %a2v = load i64, ptr %a2p, align 8
  %b1 = icmp ne i64 %rv, %a1v
  %b2 = icmp ne i64 %rv, %a2v
  %bany = or i1 %b1, %b2
  %inc = zext i1 %bany to i64
  %bad.n = add i64 %bad, %inc
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %cmp, label %done
done:
  %ok = icmp eq i64 %bad, 0
  %msg = getelementptr inbounds [16 x i8], ptr @m.penc, i64 0, i64 0
  call void @ut_check(i1 %ok, ptr %msg)
  call void @free(ptr %dst)
  call void @free(ptr %kdst)
  call void @free(ptr %o1)
  call void @free(ptr %o2)
  ret void
}

; sum access(i) over [0,n) -> volatile sink to defeat DCE.
define i64 @bench_ef(ptr %ef, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %v = call i64 @universe_ds_ef_access(ptr %ef, i64 %i)
  %acc.n = add i64 %acc, %v
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

define i64 @bench_plain(ptr %vals, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %vp = getelementptr inbounds i64, ptr %vals, i64 %i
  %v = load volatile i64, ptr %vp, align 8
  %acc.n = add i64 %acc, %v
  %i.n = add i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc.n
}

define double @ns_per(double %dt, double %reps) {
entry:
  %ns = fmul double %dt, 1000000000.0
  %r = fdiv double %ns, %reps
  ret double %r
}

define double @mb_per(double %dt, double %bytes) {
entry:
  %mb = fdiv double %bytes, 1048576.0
  %r = fdiv double %mb, %dt
  ret double %r
}
