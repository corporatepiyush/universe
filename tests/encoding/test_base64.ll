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

; Tests for universe_base64_*: known-answer vectors (std + url-safe),
; encode_len/decode_len, round-trip over fixed-seed random buffers for BOTH
; alphabets, reject-invalid (bad char, bad padding, bad length), --bench.

declare i64 @universe_base64_encode_len(i64)
declare i64 @universe_base64_decode_len(ptr, i64)
declare i64 @universe_base64_encode(ptr, ptr, i64, i32)
declare i64 @universe_base64_decode(ptr, ptr, i64, i32)
declare i64 @universe_base64_encode_scalar(ptr, ptr, i64, i32)
declare i64 @universe_base64_decode_scalar(ptr, ptr, i64, i32)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@g.src = internal global [65536 x i8] zeroinitializer, align 16
@g.enc = internal global [131072 x i8] zeroinitializer, align 16
@g.dec = internal global [65536 x i8] zeroinitializer, align 16
@g.enc2 = internal global [131072 x i8] zeroinitializer, align 16
@g.dec2 = internal global [65536 x i8] zeroinitializer, align 16
@m.xcheck = private unnamed_addr constant [30 x i8] c"vector==scalar all lengths ok\00"

@t.man    = private unnamed_addr constant [3 x i8] c"Man", align 1
@e.man    = private unnamed_addr constant [4 x i8] c"TWFu", align 1
@t.f      = private unnamed_addr constant [1 x i8] c"f", align 1
@e.f      = private unnamed_addr constant [4 x i8] c"Zg==", align 1
@t.fo     = private unnamed_addr constant [2 x i8] c"fo", align 1
@e.fo     = private unnamed_addr constant [4 x i8] c"Zm8=", align 1
@t.foobar = private unnamed_addr constant [6 x i8] c"foobar", align 1
@e.foobar = private unnamed_addr constant [8 x i8] c"Zm9vYmFy", align 1
@t.url    = private unnamed_addr constant [6 x i8] c"\00\00\3E\FF\FF\FF", align 1
@e.std    = private unnamed_addr constant [8 x i8] c"AAA+////", align 1
@e.url    = private unnamed_addr constant [8 x i8] c"AAA-____", align 1

@r.badchar = private unnamed_addr constant [4 x i8] c"AB*D", align 1
@r.badpad  = private unnamed_addr constant [4 x i8] c"AB=C", align 1
@r.badlen  = private unnamed_addr constant [3 x i8] c"ABC", align 1
@r.eqfirst = private unnamed_addr constant [4 x i8] c"=AAA", align 1

@lens = internal constant [9 x i64] [ i64 0, i64 1, i64 2, i64 3, i64 4, i64 17, i64 64, i64 65, i64 65535 ], align 8

@m.el0   = private unnamed_addr constant [16 x i8] c"encode_len(0)=0\00"
@m.el1   = private unnamed_addr constant [16 x i8] c"encode_len(1)=4\00"
@m.el3   = private unnamed_addr constant [16 x i8] c"encode_len(3)=4\00"
@m.el6   = private unnamed_addr constant [16 x i8] c"encode_len(6)=8\00"
@m.dl3   = private unnamed_addr constant [18 x i8] c"decode_len TWFu=3\00"
@m.dl1   = private unnamed_addr constant [16 x i8] c"decode_len Zg==\00"
@m.dl2   = private unnamed_addr constant [16 x i8] c"decode_len Zm8=\00"
@m.dlbad = private unnamed_addr constant [19 x i8] c"decode_len bad len\00"
@m.man   = private unnamed_addr constant [13 x i8] c"enc Man=TWFu\00"
@m.manrc = private unnamed_addr constant [13 x i8] c"enc Man rc=4\00"
@m.f     = private unnamed_addr constant [11 x i8] c"enc f=Zg==\00"
@m.fo    = private unnamed_addr constant [12 x i8] c"enc fo=Zm8=\00"
@m.foobar= private unnamed_addr constant [15 x i8] c"enc foobar ans\00"
@m.std   = private unnamed_addr constant [15 x i8] c"enc std +/ ans\00"
@m.url   = private unnamed_addr constant [15 x i8] c"enc url -_ ans\00"
@m.decman= private unnamed_addr constant [13 x i8] c"dec TWFu=Man\00"
@m.decf  = private unnamed_addr constant [10 x i8] c"dec Zg==f\00"
@m.decurl= private unnamed_addr constant [16 x i8] c"dec url-safe ok\00"
@m.rt    = private unnamed_addr constant [23 x i8] c"round-trip == original\00"
@m.rtlen = private unnamed_addr constant [22 x i8] c"round-trip lengths ok\00"
@m.rbad  = private unnamed_addr constant [15 x i8] c"bad char -> -1\00"
@m.rpad  = private unnamed_addr constant [14 x i8] c"bad pad -> -1\00"
@m.rlen  = private unnamed_addr constant [17 x i8] c"bad length -> -1\00"
@m.reqf  = private unnamed_addr constant [16 x i8] c"'=' first -> -1\00"
@b64.encsamp = internal global [16 x double] zeroinitializer, align 8
@b64.decsamp = internal global [16 x double] zeroinitializer, align 8
@lbl.enc = private unnamed_addr constant [11 x i8] c"b64 encode\00"
@lbl.dec = private unnamed_addr constant [11 x i8] c"b64 decode\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- encode_len / decode_len ----
  %el0 = call i64 @universe_base64_encode_len(i64 0)
  call void @ut_check_eq(i64 %el0, i64 0, ptr @m.el0)
  %el1 = call i64 @universe_base64_encode_len(i64 1)
  call void @ut_check_eq(i64 %el1, i64 4, ptr @m.el1)
  %el3 = call i64 @universe_base64_encode_len(i64 3)
  call void @ut_check_eq(i64 %el3, i64 4, ptr @m.el3)
  %el6 = call i64 @universe_base64_encode_len(i64 6)
  call void @ut_check_eq(i64 %el6, i64 8, ptr @m.el6)
  %dl3 = call i64 @universe_base64_decode_len(ptr @e.man, i64 4)
  call void @ut_check_eq(i64 %dl3, i64 3, ptr @m.dl3)
  %dl1 = call i64 @universe_base64_decode_len(ptr @e.f, i64 4)
  call void @ut_check_eq(i64 %dl1, i64 1, ptr @m.dl1)
  %dl2 = call i64 @universe_base64_decode_len(ptr @e.fo, i64 4)
  call void @ut_check_eq(i64 %dl2, i64 2, ptr @m.dl2)
  %dlb = call i64 @universe_base64_decode_len(ptr @r.badlen, i64 3)
  %dlb.ok = icmp eq i64 %dlb, -1
  call void @ut_check(i1 %dlb.ok, ptr @m.dlbad)

  ; ---- known-answer encode (standard) ----
  %em = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.man, i64 3, i32 0)
  call void @ut_check_eq(i64 %em, i64 4, ptr @m.manrc)
  %cm = call i32 @memcmp(ptr @g.enc, ptr @e.man, i64 4)
  %cm.ok = icmp eq i32 %cm, 0
  call void @ut_check(i1 %cm.ok, ptr @m.man)
  %ef = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.f, i64 1, i32 0)
  %cf = call i32 @memcmp(ptr @g.enc, ptr @e.f, i64 4)
  %cf.ok = icmp eq i32 %cf, 0
  call void @ut_check(i1 %cf.ok, ptr @m.f)
  %efo = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.fo, i64 2, i32 0)
  %cfo = call i32 @memcmp(ptr @g.enc, ptr @e.fo, i64 4)
  %cfo.ok = icmp eq i32 %cfo, 0
  call void @ut_check(i1 %cfo.ok, ptr @m.fo)
  %efb = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.foobar, i64 6, i32 0)
  %cfb = call i32 @memcmp(ptr @g.enc, ptr @e.foobar, i64 8)
  %cfb.ok = icmp eq i32 %cfb, 0
  call void @ut_check(i1 %cfb.ok, ptr @m.foobar)

  ; ---- alphabet: std '+/' vs url '-_' ----
  %es = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.url, i64 6, i32 0)
  %cs = call i32 @memcmp(ptr @g.enc, ptr @e.std, i64 8)
  %cs.ok = icmp eq i32 %cs, 0
  call void @ut_check(i1 %cs.ok, ptr @m.std)
  %eu = call i64 @universe_base64_encode(ptr @g.enc, ptr @t.url, i64 6, i32 1)
  %cu = call i32 @memcmp(ptr @g.enc, ptr @e.url, i64 8)
  %cu.ok = icmp eq i32 %cu, 0
  call void @ut_check(i1 %cu.ok, ptr @m.url)

  ; ---- known-answer decode ----
  %dm = call i64 @universe_base64_decode(ptr @g.dec, ptr @e.man, i64 4, i32 0)
  %dm.len = icmp eq i64 %dm, 3
  %dmc = call i32 @memcmp(ptr @g.dec, ptr @t.man, i64 3)
  %dmc.z = icmp eq i32 %dmc, 0
  %dm.ok = and i1 %dm.len, %dmc.z
  call void @ut_check(i1 %dm.ok, ptr @m.decman)
  %df = call i64 @universe_base64_decode(ptr @g.dec, ptr @e.f, i64 4, i32 0)
  %df.len = icmp eq i64 %df, 1
  %dfc = call i32 @memcmp(ptr @g.dec, ptr @t.f, i64 1)
  %dfc.z = icmp eq i32 %dfc, 0
  %df.ok = and i1 %df.len, %dfc.z
  call void @ut_check(i1 %df.ok, ptr @m.decf)
  %du = call i64 @universe_base64_decode(ptr @g.dec, ptr @e.url, i64 8, i32 1)
  %du.len = icmp eq i64 %du, 6
  %duc = call i32 @memcmp(ptr @g.dec, ptr @t.url, i64 6)
  %duc.z = icmp eq i32 %duc, 0
  %du.ok = and i1 %du.len, %duc.z
  call void @ut_check(i1 %du.ok, ptr @m.decurl)

  ; ---- reject invalid ----
  %rb = call i64 @universe_base64_decode(ptr @g.dec, ptr @r.badchar, i64 4, i32 0)
  %rb.ok = icmp eq i64 %rb, -1
  call void @ut_check(i1 %rb.ok, ptr @m.rbad)
  %rp = call i64 @universe_base64_decode(ptr @g.dec, ptr @r.badpad, i64 4, i32 0)
  %rp.ok = icmp eq i64 %rp, -1
  call void @ut_check(i1 %rp.ok, ptr @m.rpad)
  %rl = call i64 @universe_base64_decode(ptr @g.dec, ptr @r.badlen, i64 3, i32 0)
  %rl.ok = icmp eq i64 %rl, -1
  call void @ut_check(i1 %rl.ok, ptr @m.rlen)
  %rq = call i64 @universe_base64_decode(ptr @g.dec, ptr @r.eqfirst, i64 4, i32 0)
  %rq.ok = icmp eq i64 %rq, -1
  call void @ut_check(i1 %rq.ok, ptr @m.reqf)

  ; ---- round-trip over many lengths, both alphabets ----
  %state = alloca i64, align 8
  store i64 1234567, ptr %state, align 8
  br label %rt.head

rt.head:
  %li = phi i64 [ 0, %entry ], [ %li.next, %rt.next ]
  %mis = phi i64 [ 0, %entry ], [ %mis.n2, %rt.next ]
  %lmis = phi i64 [ 0, %entry ], [ %lmis.n2, %rt.next ]
  %lp = getelementptr inbounds nuw [9 x i64], ptr @lens, i64 0, i64 %li
  %len = load i64, ptr %lp, align 8
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %rt.head ], [ %fi.next, %fill.body ]
  %fdone = icmp uge i64 %fi, %len
  br i1 %fdone, label %flag0, label %fill.body

fill.body:
  %rv = call i64 @ut_rand(ptr %state)
  %rb8 = trunc i64 %rv to i8
  %sp = getelementptr inbounds nuw [65536 x i8], ptr @g.src, i64 0, i64 %fi
  store i8 %rb8, ptr %sp, align 1
  %fi.next = add nuw i64 %fi, 1
  br label %fill.head

flag0:
  %e0 = call i64 @universe_base64_encode(ptr @g.enc, ptr @g.src, i64 %len, i32 0)
  %dl0 = call i64 @universe_base64_decode_len(ptr @g.enc, i64 %e0)
  %d0 = call i64 @universe_base64_decode(ptr @g.dec, ptr @g.enc, i64 %e0, i32 0)
  %l0.bad = icmp ne i64 %d0, %len
  %dl0.bad = icmp ne i64 %dl0, %len
  %c0 = call i32 @memcmp(ptr @g.dec, ptr @g.src, i64 %len)
  %c0.bad = icmp ne i32 %c0, 0
  %m0a = zext i1 %l0.bad to i64
  %m0b = zext i1 %c0.bad to i64
  %mis.n0 = add nuw i64 %mis, %m0a
  %mis.n1 = add nuw i64 %mis.n0, %m0b
  %lm0 = zext i1 %dl0.bad to i64
  %lmis.n0 = add nuw i64 %lmis, %lm0
  br label %flag1

flag1:
  %e1 = call i64 @universe_base64_encode(ptr @g.enc, ptr @g.src, i64 %len, i32 1)
  %dl1a = call i64 @universe_base64_decode_len(ptr @g.enc, i64 %e1)
  %d1 = call i64 @universe_base64_decode(ptr @g.dec, ptr @g.enc, i64 %e1, i32 1)
  %l1.bad = icmp ne i64 %d1, %len
  %dl1.bad = icmp ne i64 %dl1a, %len
  %c1 = call i32 @memcmp(ptr @g.dec, ptr @g.src, i64 %len)
  %c1.bad = icmp ne i32 %c1, 0
  %m1a = zext i1 %l1.bad to i64
  %m1b = zext i1 %c1.bad to i64
  %mis.n1b = add nuw i64 %mis.n1, %m1a
  %mis.n2 = add nuw i64 %mis.n1b, %m1b
  %lm1 = zext i1 %dl1.bad to i64
  %lmis.n2 = add nuw i64 %lmis.n0, %lm1
  br label %rt.next

rt.next:
  %li.next = add nuw i64 %li, 1
  %more = icmp ult i64 %li.next, 9
  br i1 %more, label %rt.head, label %rt.done

rt.done:
  call void @ut_check_eq(i64 %mis.n2, i64 0, ptr @m.rt)
  call void @ut_check_eq(i64 %lmis.n2, i64 0, ptr @m.rtlen)

  ; ---- SIMD-first cross-check: the VECTOR public entry must produce bit-
  ;      identical output to its scalar oracle for encode AND decode, over every
  ;      length 0..200 (exercises the vector body + the ragged scalar tail) and
  ;      both alphabets (url = length parity). Any mismatch increments %xmis. ----
  store i64 987654321, ptr %state, align 8
  br label %xc.head

xc.head:
  %xl = phi i64 [ 0, %rt.done ], [ %xl.next, %xc.next ]
  %xmis = phi i64 [ 0, %rt.done ], [ %xmis.n, %xc.next ]
  br label %xf.head

xf.head:
  %xfi = phi i64 [ 0, %xc.head ], [ %xfi.next, %xf.body ]
  %xfd = icmp uge i64 %xfi, %xl
  br i1 %xfd, label %xc.enc, label %xf.body

xf.body:
  %xrv = call i64 @ut_rand(ptr %state)
  %xrb = trunc i64 %xrv to i8
  %xsp = getelementptr inbounds nuw [65536 x i8], ptr @g.src, i64 0, i64 %xfi
  store i8 %xrb, ptr %xsp, align 1
  %xfi.next = add nuw i64 %xfi, 1
  br label %xf.head

xc.enc:
  %url64 = and i64 %xl, 1
  %url = trunc i64 %url64 to i32
  %ev = call i64 @universe_base64_encode(ptr @g.enc, ptr @g.src, i64 %xl, i32 %url)
  %es.x = call i64 @universe_base64_encode_scalar(ptr @g.enc2, ptr @g.src, i64 %xl, i32 %url)
  %elen.bad = icmp ne i64 %ev, %es.x
  %ecmp = call i32 @memcmp(ptr @g.enc, ptr @g.enc2, i64 %ev)
  %ecmp.bad = icmp ne i32 %ecmp, 0
  %dv = call i64 @universe_base64_decode(ptr @g.dec, ptr @g.enc, i64 %ev, i32 %url)
  %ds = call i64 @universe_base64_decode_scalar(ptr @g.dec2, ptr @g.enc, i64 %ev, i32 %url)
  %dlen.bad = icmp ne i64 %dv, %ds
  %dcmp = call i32 @memcmp(ptr @g.dec, ptr @g.dec2, i64 %dv)
  %dcmp.bad = icmp ne i32 %dcmp, 0
  %ocmp = call i32 @memcmp(ptr @g.dec, ptr @g.src, i64 %xl)
  %ocmp.bad = icmp ne i32 %ocmp, 0
  %rt.bad = icmp ne i64 %dv, %xl
  %x1 = zext i1 %elen.bad to i64
  %x2 = zext i1 %ecmp.bad to i64
  %x3 = zext i1 %dlen.bad to i64
  %x4 = zext i1 %dcmp.bad to i64
  %x5 = zext i1 %ocmp.bad to i64
  %x6 = zext i1 %rt.bad to i64
  %xa = add nuw i64 %x1, %x2
  %xb = add nuw i64 %xa, %x3
  %xc2 = add nuw i64 %xb, %x4
  %xd = add nuw i64 %xc2, %x5
  %xe = add nuw i64 %xd, %x6
  %xmis.n = add nuw i64 %xmis, %xe
  br label %xc.next

xc.next:
  %xl.next = add nuw i64 %xl, 1
  %xmore = icmp ult i64 %xl.next, 201
  br i1 %xmore, label %xc.head, label %xc.done

xc.done:
  call void @ut_check_eq(i64 %xmis.n, i64 0, ptr @m.xcheck)

  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  store i64 7, ptr %state, align 8
  br label %bf.head

bf.head:
  %bi = phi i64 [ 0, %bench ], [ %bi.next, %bf.body ]
  %bdone = icmp uge i64 %bi, 65536
  br i1 %bdone, label %bench.run, label %bf.body

bf.body:
  %brv = call i64 @ut_rand(ptr %state)
  %brb = trunc i64 %brv to i8
  %bsp = getelementptr inbounds nuw [65536 x i8], ptr @g.src, i64 0, i64 %bi
  store i8 %brb, ptr %bsp, align 1
  %bi.next = add nuw i64 %bi, 1
  br label %bf.head

bench.run:
  ; length for the decode side; also serves as the first warm-up encode.
  %be0 = call i64 @universe_base64_encode(ptr @g.enc, ptr @g.src, i64 65536, i32 0)
  ; ENCODE: 17 reps of a 1000-encode batch; discard rep 0 (warm-up), report the
  ; distribution of the remaining 16 (ops/rep = 1000 * 65536 = 65536000).
  br label %enc.rep

enc.rep:
  %erep = phi i64 [ 0, %bench.run ], [ %erep.n, %enc.next ]
  %et0 = call double @ut_now_sec()
  br label %enc.in

enc.in:
  %eic = phi i64 [ 0, %enc.rep ], [ %eic.n, %enc.in ]
  %bev = call i64 @universe_base64_encode(ptr @g.enc, ptr @g.src, i64 65536, i32 0)
  %eic.n = add nuw i64 %eic, 1
  %eim = icmp ult i64 %eic.n, 1000
  br i1 %eim, label %enc.in, label %enc.rep.done

enc.rep.done:
  %et1 = call double @ut_now_sec()
  %eel = fsub double %et1, %et0
  %ekeep = icmp ugt i64 %erep, 0
  br i1 %ekeep, label %enc.store, label %enc.next

enc.store:
  %eidx = sub i64 %erep, 1
  %esp = getelementptr inbounds [16 x double], ptr @b64.encsamp, i64 0, i64 %eidx
  store double %eel, ptr %esp, align 8
  br label %enc.next

enc.next:
  %erep.n = add nuw i64 %erep, 1
  %erm = icmp ult i64 %erep.n, 17
  br i1 %erm, label %enc.rep, label %enc.report

enc.report:
  call void @ut_report_dist(ptr @b64.encsamp, i64 16, i64 65536000, ptr @lbl.enc)
  br label %dec.rep

dec.rep:
  %drep = phi i64 [ 0, %enc.report ], [ %drep.n, %dec.next ]
  %dt0 = call double @ut_now_sec()
  br label %dec.in

dec.in:
  %dic = phi i64 [ 0, %dec.rep ], [ %dic.n, %dec.in ]
  %bdv = call i64 @universe_base64_decode(ptr @g.dec, ptr @g.enc, i64 %be0, i32 0)
  %dic.n = add nuw i64 %dic, 1
  %dim = icmp ult i64 %dic.n, 1000
  br i1 %dim, label %dec.in, label %dec.rep.done

dec.rep.done:
  %dt1 = call double @ut_now_sec()
  %del = fsub double %dt1, %dt0
  %dkeep = icmp ugt i64 %drep, 0
  br i1 %dkeep, label %dec.store, label %dec.next

dec.store:
  %didx = sub i64 %drep, 1
  %dsp = getelementptr inbounds [16 x double], ptr @b64.decsamp, i64 0, i64 %didx
  store double %del, ptr %dsp, align 8
  br label %dec.next

dec.next:
  %drep.n = add nuw i64 %drep, 1
  %drm = icmp ult i64 %drep.n, 17
  br i1 %drm, label %dec.rep, label %dec.report

dec.report:
  call void @ut_report_dist(ptr @b64.decsamp, i64 16, i64 65536000, ptr @lbl.dec)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
