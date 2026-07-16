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

; Tests for universe_varint_*: ULEB/SLEB known-answer vectors, uleb_len/sleb_len
; agreement with the encoders, fixed-seed random round-trips over u64/i64,
; decode failure modes (truncated, overflow, terminal-byte overflow), zigzag
; identities, and a --bench mode timing encode+decode throughput.

declare i64 @universe_varint_uleb_len(i64)
declare i64 @universe_varint_sleb_len(i64)
declare i64 @universe_varint_uleb_encode(ptr, i64)
declare i64 @universe_varint_sleb_encode(ptr, i64)
declare i64 @universe_varint_uleb_decode(ptr, i64, ptr)
declare i64 @universe_varint_sleb_decode(ptr, i64, ptr)
declare i64 @universe_varint_zigzag_encode(i64)
declare i64 @universe_varint_zigzag_decode(i64)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@g.buf = internal global [16 x i8] zeroinitializer, align 16
@g.out = internal global i64 0, align 8

; ---- known ULEB byte patterns ----
@kv.128   = private unnamed_addr constant [2 x i8] c"\80\01"
@kv.300   = private unnamed_addr constant [2 x i8] c"\AC\02"
@kv.16384 = private unnamed_addr constant [3 x i8] c"\80\80\01"
; ---- known SLEB byte patterns ----
@sv.64    = private unnamed_addr constant [2 x i8] c"\C0\00"
@sv.n128  = private unnamed_addr constant [2 x i8] c"\80\7F"
; ---- malformed decode inputs ----
@bad.trunc = private unnamed_addr constant [2 x i8] c"\80\80"       ; both continue, ends
@bad.ovf   = private unnamed_addr constant [11 x i8] c"\80\80\80\80\80\80\80\80\80\80\00" ; 10 continues
@bad.hi    = private unnamed_addr constant [10 x i8] c"\80\80\80\80\80\80\80\80\80\02"    ; bit 64 set

@m.ulen0  = private unnamed_addr constant [16 x i8] c"uleb_len(0)==1\0A\00"
@m.ulen   = private unnamed_addr constant [22 x i8] c"uleb_len==encoded len\00"
@m.slen   = private unnamed_addr constant [22 x i8] c"sleb_len==encoded len\00"
@m.u128   = private unnamed_addr constant [17 x i8] c"uleb 128 pattern\00"
@m.u300   = private unnamed_addr constant [17 x i8] c"uleb 300 pattern\00"
@m.u16384 = private unnamed_addr constant [19 x i8] c"uleb 16384 pattern\00"
@m.s64    = private unnamed_addr constant [16 x i8] c"sleb 64 pattern\00"
@m.sn128  = private unnamed_addr constant [18 x i8] c"sleb -128 pattern\00"
@m.urt    = private unnamed_addr constant [22 x i8] c"uleb round-trip value\00"
@m.urtc   = private unnamed_addr constant [21 x i8] c"uleb round-trip cons\00"
@m.srt    = private unnamed_addr constant [22 x i8] c"sleb round-trip value\00"
@m.srtc   = private unnamed_addr constant [21 x i8] c"sleb round-trip cons\00"
@m.zz     = private unnamed_addr constant [20 x i8] c"zigzag round-trip v\00"
@m.zz0    = private unnamed_addr constant [15 x i8] c"zigzag(0)==0\0A\00\00"
@m.zzn1   = private unnamed_addr constant [15 x i8] c"zigzag(-1)==1\0A\00"
@m.zz1    = private unnamed_addr constant [14 x i8] c"zigzag(1)==2\0A\00"
@m.trunc  = private unnamed_addr constant [23 x i8] c"truncated -> -13 PARSE\00"
@m.ovf    = private unnamed_addr constant [20 x i8] c"overrun -> -3 OVFLW\00"
@m.hi     = private unnamed_addr constant [21 x i8] c"bit64 -> -3 SIZE_OVF\00"
@m.max    = private unnamed_addr constant [21 x i8] c"u64 max round-trips\0A\00"
@vi.encsamp = internal global [16 x double] zeroinitializer, align 8
@vi.decsamp = internal global [16 x double] zeroinitializer, align 8
@lbl.vienc = private unnamed_addr constant [19 x i8] c"varint uleb encode\00"
@lbl.videc = private unnamed_addr constant [19 x i8] c"varint uleb decode\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- uleb_len basics ----
  %ul0 = call i64 @universe_varint_uleb_len(i64 0)
  %ul0.ok = icmp eq i64 %ul0, 1
  call void @ut_check(i1 %ul0.ok, ptr @m.ulen0)

  ; ---- known ULEB patterns ----
  %e128 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 128)
  %c128 = call i32 @memcmp(ptr @g.buf, ptr @kv.128, i64 2)
  %c128.ok = icmp eq i32 %c128, 0
  %e128.ok = icmp eq i64 %e128, 2
  %k128 = and i1 %c128.ok, %e128.ok
  call void @ut_check(i1 %k128, ptr @m.u128)

  %e300 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 300)
  %c300 = call i32 @memcmp(ptr @g.buf, ptr @kv.300, i64 2)
  %c300.ok = icmp eq i32 %c300, 0
  call void @ut_check(i1 %c300.ok, ptr @m.u300)

  %e16384 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 16384)
  %c16384 = call i32 @memcmp(ptr @g.buf, ptr @kv.16384, i64 3)
  %c16384.ok = icmp eq i32 %c16384, 0
  %e16384.ok = icmp eq i64 %e16384, 3
  %k16384 = and i1 %c16384.ok, %e16384.ok
  call void @ut_check(i1 %k16384, ptr @m.u16384)

  ; ---- known SLEB patterns ----
  %es64 = call i64 @universe_varint_sleb_encode(ptr @g.buf, i64 64)
  %cs64 = call i32 @memcmp(ptr @g.buf, ptr @sv.64, i64 2)
  %cs64.ok = icmp eq i32 %cs64, 0
  %es64.ok = icmp eq i64 %es64, 2
  %ks64 = and i1 %cs64.ok, %es64.ok
  call void @ut_check(i1 %ks64, ptr @m.s64)

  %esn = call i64 @universe_varint_sleb_encode(ptr @g.buf, i64 -128)
  %csn = call i32 @memcmp(ptr @g.buf, ptr @sv.n128, i64 2)
  %csn.ok = icmp eq i32 %csn, 0
  %esn.ok = icmp eq i64 %esn, 2
  %ksn = and i1 %csn.ok, %esn.ok
  call void @ut_check(i1 %ksn, ptr @m.sn128)

  ; ---- decode failure modes ----
  %dt = call i64 @universe_varint_uleb_decode(ptr @bad.trunc, i64 2, ptr @g.out)
  %dt.ok = icmp eq i64 %dt, -13
  call void @ut_check(i1 %dt.ok, ptr @m.trunc)

  %dov = call i64 @universe_varint_uleb_decode(ptr @bad.ovf, i64 11, ptr @g.out)
  %dov.ok = icmp eq i64 %dov, -3
  call void @ut_check(i1 %dov.ok, ptr @m.ovf)

  %dhi = call i64 @universe_varint_uleb_decode(ptr @bad.hi, i64 10, ptr @g.out)
  %dhi.ok = icmp eq i64 %dhi, -3
  call void @ut_check(i1 %dhi.ok, ptr @m.hi)

  ; ---- u64 max round-trips exactly (10 bytes) ----
  %em = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 -1) ; 0xFFFF..FF
  %em.ok = icmp eq i64 %em, 10
  %dm = call i64 @universe_varint_uleb_decode(ptr @g.buf, i64 10, ptr @g.out)
  %dmv = load i64, ptr @g.out, align 8
  %dmv.ok = icmp eq i64 %dmv, -1
  %dmc.ok = icmp eq i64 %dm, 10
  %maxall = and i1 %em.ok, %dmv.ok
  %maxall2 = and i1 %maxall, %dmc.ok
  call void @ut_check(i1 %maxall2, ptr @m.max)

  ; ---- zigzag identities ----
  %z0 = call i64 @universe_varint_zigzag_encode(i64 0)
  %z0.ok = icmp eq i64 %z0, 0
  call void @ut_check(i1 %z0.ok, ptr @m.zz0)
  %zn1 = call i64 @universe_varint_zigzag_encode(i64 -1)
  %zn1.ok = icmp eq i64 %zn1, 1
  call void @ut_check(i1 %zn1.ok, ptr @m.zzn1)
  %z1 = call i64 @universe_varint_zigzag_encode(i64 1)
  %z1.ok = icmp eq i64 %z1, 2
  call void @ut_check(i1 %z1.ok, ptr @m.zz1)

  ; ---- randomized round-trips ----
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  br label %rt.head

rt.head:
  %it = phi i64 [ 0, %entry ], [ %it.next, %rt.head ]
  %uv.mis = phi i64 [ 0, %entry ], [ %uv.mis.n, %rt.head ]
  %uc.mis = phi i64 [ 0, %entry ], [ %uc.mis.n, %rt.head ]
  %sv.mis = phi i64 [ 0, %entry ], [ %sv.mis.n, %rt.head ]
  %sc.mis = phi i64 [ 0, %entry ], [ %sc.mis.n, %rt.head ]
  %ul.mis = phi i64 [ 0, %entry ], [ %ul.mis.n, %rt.head ]
  %sl.mis = phi i64 [ 0, %entry ], [ %sl.mis.n, %rt.head ]
  %zz.mis = phi i64 [ 0, %entry ], [ %zz.mis.n, %rt.head ]
  %rv = call i64 @ut_rand(ptr %state)

  ; --- unsigned round trip ---
  %uwrote = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %rv)
  %ulen = call i64 @universe_varint_uleb_len(i64 %rv)
  %ul.bad = icmp ne i64 %ulen, %uwrote
  %ul.b = zext i1 %ul.bad to i64
  %ul.mis.n = add nuw i64 %ul.mis, %ul.b
  %ucons = call i64 @universe_varint_uleb_decode(ptr @g.buf, i64 16, ptr @g.out)
  %uval = load i64, ptr @g.out, align 8
  %uv.bad = icmp ne i64 %uval, %rv
  %uv.b = zext i1 %uv.bad to i64
  %uv.mis.n = add nuw i64 %uv.mis, %uv.b
  %uc.bad = icmp ne i64 %ucons, %uwrote
  %uc.b = zext i1 %uc.bad to i64
  %uc.mis.n = add nuw i64 %uc.mis, %uc.b

  ; --- signed round trip (same random bits reinterpreted) ---
  %swrote = call i64 @universe_varint_sleb_encode(ptr @g.buf, i64 %rv)
  %slen = call i64 @universe_varint_sleb_len(i64 %rv)
  %sl.bad = icmp ne i64 %slen, %swrote
  %sl.b = zext i1 %sl.bad to i64
  %sl.mis.n = add nuw i64 %sl.mis, %sl.b
  %scons = call i64 @universe_varint_sleb_decode(ptr @g.buf, i64 16, ptr @g.out)
  %sval = load i64, ptr @g.out, align 8
  %sv.bad = icmp ne i64 %sval, %rv
  %sv.b = zext i1 %sv.bad to i64
  %sv.mis.n = add nuw i64 %sv.mis, %sv.b
  %sc.bad = icmp ne i64 %scons, %swrote
  %sc.b = zext i1 %sc.bad to i64
  %sc.mis.n = add nuw i64 %sc.mis, %sc.b

  ; --- zigzag identity ---
  %zenc = call i64 @universe_varint_zigzag_encode(i64 %rv)
  %zdec = call i64 @universe_varint_zigzag_decode(i64 %zenc)
  %zz.bad = icmp ne i64 %zdec, %rv
  %zz.b = zext i1 %zz.bad to i64
  %zz.mis.n = add nuw i64 %zz.mis, %zz.b

  %it.next = add nuw i64 %it, 1
  %more = icmp ult i64 %it.next, 200000
  br i1 %more, label %rt.head, label %rt.done

rt.done:
  call void @ut_check_eq(i64 %uv.mis, i64 0, ptr @m.urt)
  call void @ut_check_eq(i64 %uc.mis, i64 0, ptr @m.urtc)
  call void @ut_check_eq(i64 %sv.mis, i64 0, ptr @m.srt)
  call void @ut_check_eq(i64 %sc.mis, i64 0, ptr @m.srtc)
  call void @ut_check_eq(i64 %ul.mis, i64 0, ptr @m.ulen)
  call void @ut_check_eq(i64 %sl.mis, i64 0, ptr @m.slen)
  call void @ut_check_eq(i64 %zz.mis, i64 0, ptr @m.zz)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; ENCODE: 17 reps of a 1,000,000-encode batch; re-seed the rng each rep so the
  ; batch is identical, discard rep 0 (warm-up), report over the remaining 16.
  ; ops/rep = 1000000.
  br label %enc.rep

enc.rep:
  %erep = phi i64 [ 0, %bench ], [ %erep.n, %enc.next ]
  store i64 1, ptr %state, align 8
  %t0 = call double @ut_now_sec()
  br label %be.head

be.head:
  %bec = phi i64 [ 0, %enc.rep ], [ %bec.next, %be.head ]
  %brv = call i64 @ut_rand(ptr %state)
  %bw = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %brv)
  %bec.next = add nuw i64 %bec, 1
  %bemore = icmp ult i64 %bec.next, 1000000
  br i1 %bemore, label %be.head, label %enc.rep.done

enc.rep.done:
  %t1 = call double @ut_now_sec()
  %eel = fsub double %t1, %t0
  %ekeep = icmp ugt i64 %erep, 0
  br i1 %ekeep, label %enc.store, label %enc.next

enc.store:
  %eidx = sub i64 %erep, 1
  %esp = getelementptr inbounds [16 x double], ptr @vi.encsamp, i64 0, i64 %eidx
  store double %eel, ptr %esp, align 8
  br label %enc.next

enc.next:
  %erep.n = add nuw i64 %erep, 1
  %erm = icmp ult i64 %erep.n, 17
  br i1 %erm, label %enc.rep, label %enc.report

enc.report:
  call void @ut_report_dist(ptr @vi.encsamp, i64 16, i64 1000000, ptr @lbl.vienc)
  br label %dec.rep

dec.rep:
  %drep = phi i64 [ 0, %enc.report ], [ %drep.n, %dec.next ]
  store i64 1, ptr %state, align 8
  %dt0 = call double @ut_now_sec()
  br label %bd.head

bd.head:
  %bdc = phi i64 [ 0, %dec.rep ], [ %bdc.next, %bd.head ]
  %brv2 = call i64 @ut_rand(ptr %state)
  %bw2 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %brv2)
  %bd = call i64 @universe_varint_uleb_decode(ptr @g.buf, i64 16, ptr @g.out)
  %bdc.next = add nuw i64 %bdc, 1
  %bdmore = icmp ult i64 %bdc.next, 1000000
  br i1 %bdmore, label %bd.head, label %dec.rep.done

dec.rep.done:
  %dt1 = call double @ut_now_sec()
  %del = fsub double %dt1, %dt0
  %dkeep = icmp ugt i64 %drep, 0
  br i1 %dkeep, label %dec.store, label %dec.next

dec.store:
  %didx = sub i64 %drep, 1
  %dsp = getelementptr inbounds [16 x double], ptr @vi.decsamp, i64 0, i64 %didx
  store double %del, ptr %dsp, align 8
  br label %dec.next

dec.next:
  %drep.n = add nuw i64 %drep, 1
  %drm = icmp ult i64 %drep.n, 17
  br i1 %drm, label %dec.rep, label %dec.report

dec.report:
  call void @ut_report_dist(ptr @vi.decsamp, i64 16, i64 1000000, ptr @lbl.videc)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
