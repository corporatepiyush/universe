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

; Tests for universe_hex_*: known-answer vectors, round-trip over fixed-seed
; random buffers of many lengths, reject-invalid (bad char, odd length),
; encode_len (incl. overflow), and a --bench mode vs a naive per-byte encoder.

declare i64 @universe_hex_encode_len(i64)
declare i64 @universe_hex_encode(ptr, ptr, i64)
declare i64 @universe_hex_decode(ptr, ptr, i64)

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

@k.in    = private unnamed_addr constant [6 x i8] c"\00\FF\DE\AD\BE\EF", align 1
@k.out   = private unnamed_addr constant [12 x i8] c"00ffdeadbeef", align 1
@k.upper = private unnamed_addr constant [8 x i8] c"DEADBEEF", align 1
@k.exp   = private unnamed_addr constant [4 x i8] c"\DE\AD\BE\EF", align 1
@k.bad   = private unnamed_addr constant [4 x i8] c"00zz", align 1
@k.odd   = private unnamed_addr constant [3 x i8] c"abc", align 1

@lens = internal constant [9 x i64] [ i64 0, i64 1, i64 2, i64 3, i64 17, i64 63, i64 64, i64 65, i64 65536 ], align 8

; Oracle: a plain per-byte scalar hex decoder, independent of the library's
; SIMD path. Same contract: -1 on odd length or any non-hex char, else n/2.
@g.ref = internal global [65536 x i8] zeroinitializer, align 16
@olens = internal constant [8 x i64] [ i64 0, i64 2, i64 16, i64 30, i64 32, i64 34, i64 200, i64 131072 ], align 8

define internal i64 @ref_hex_decode(ptr %dst, ptr %src, i64 %n) {
entry:
  %odd = and i64 %n, 1
  %o = icmp ne i64 %odd, 0
  br i1 %o, label %bad, label %pre
bad:
  ret i64 -1
pre:
  %half = lshr i64 %n, 1
  %z = icmp eq i64 %n, 0
  br i1 %z, label %fin, label %lp
lp:
  %i = phi i64 [ 0, %pre ], [ %i.n, %lp ]
  %acc = phi i32 [ 0, %pre ], [ %acc.n, %lp ]
  %so = shl nuw i64 %i, 1
  %p0 = getelementptr inbounds nuw i8, ptr %src, i64 %so
  %c0 = load i8, ptr %p0, align 1
  %so1 = or disjoint i64 %so, 1
  %p1 = getelementptr inbounds nuw i8, ptr %src, i64 %so1
  %c1 = load i8, ptr %p1, align 1
  %h = zext i8 %c0 to i32
  %hdge = icmp uge i32 %h, 48
  %hdle = icmp ule i32 %h, 57
  %hd = and i1 %hdge, %hdle
  %hdv = sub nsw i32 %h, 48
  %hlc = or i32 %h, 32
  %hage = icmp uge i32 %hlc, 97
  %hale = icmp ule i32 %hlc, 102
  %ha = and i1 %hage, %hale
  %hav = sub nsw i32 %hlc, 87
  %hs = select i1 %ha, i32 %hav, i32 0
  %hv = select i1 %hd, i32 %hdv, i32 %hs
  %hok = or i1 %hd, %ha
  %l = zext i8 %c1 to i32
  %ldge = icmp uge i32 %l, 48
  %ldle = icmp ule i32 %l, 57
  %ld = and i1 %ldge, %ldle
  %ldv = sub nsw i32 %l, 48
  %llc = or i32 %l, 32
  %lage = icmp uge i32 %llc, 97
  %lale = icmp ule i32 %llc, 102
  %la = and i1 %lage, %lale
  %lav = sub nsw i32 %llc, 87
  %ls = select i1 %la, i32 %lav, i32 0
  %lv = select i1 %ld, i32 %ldv, i32 %ls
  %lok = or i1 %ld, %la
  %sh = shl nsw i32 %hv, 4
  %ov = or disjoint i32 %sh, %lv
  %ob = trunc i32 %ov to i8
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %ob, ptr %dp, align 1
  %both = and i1 %hok, %lok
  %okb = zext i1 %both to i32
  %inv = xor i32 %okb, 1
  %acc.n = or i32 %acc, %inv
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %half
  br i1 %more, label %lp, label %chk
chk:
  %isbad = icmp ne i32 %acc.n, 0
  br i1 %isbad, label %bad2, label %fin
bad2:
  ret i64 -1
fin:
  ret i64 %half
}

@m.klen   = private unnamed_addr constant [15 x i8] c"encode_len =2n\00"
@m.kovf   = private unnamed_addr constant [20 x i8] c"encode_len overflow\00"
@m.kenc   = private unnamed_addr constant [17 x i8] c"encode known ans\00"
@m.kencrc = private unnamed_addr constant [15 x i8] c"encode returns\00"
@m.kdec   = private unnamed_addr constant [21 x i8] c"decode upper hex ans\00"
@m.kdecrc = private unnamed_addr constant [17 x i8] c"decode returns n\00"
@m.rt     = private unnamed_addr constant [23 x i8] c"round-trip == original\00"
@m.rtlen  = private unnamed_addr constant [22 x i8] c"round-trip decode len\00"
@m.bad    = private unnamed_addr constant [21 x i8] c"bad char rejected -1\00"
@m.odd    = private unnamed_addr constant [20 x i8] c"odd length rejected\00"
@m.orv    = private unnamed_addr constant [25 x i8] c"oracle decode retvals ==\00"
@m.orb    = private unnamed_addr constant [23 x i8] c"oracle decode bytes ==\00"
@hex.encsamp = internal global [16 x double] zeroinitializer, align 8
@hex.decsamp = internal global [16 x double] zeroinitializer, align 8
@lbl.hexenc = private unnamed_addr constant [19 x i8] c"hex encode n=65536\00"
@lbl.hexdec = private unnamed_addr constant [19 x i8] c"hex decode n=65536\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- encode_len ----
  %el = call i64 @universe_hex_encode_len(i64 5)
  call void @ut_check_eq(i64 %el, i64 10, ptr @m.klen)
  %elo = call i64 @universe_hex_encode_len(i64 -9223372036854775808) ; 0x8000..0
  %elo.bad = icmp eq i64 %elo, -1
  call void @ut_check(i1 %elo.bad, ptr @m.kovf)

  ; ---- known-answer encode ----
  %e1 = call i64 @universe_hex_encode(ptr @g.enc, ptr @k.in, i64 6)
  call void @ut_check_eq(i64 %e1, i64 12, ptr @m.kencrc)
  %ec = call i32 @memcmp(ptr @g.enc, ptr @k.out, i64 12)
  %ec.ok = icmp eq i32 %ec, 0
  call void @ut_check(i1 %ec.ok, ptr @m.kenc)

  ; ---- known-answer decode (uppercase accepted) ----
  %d1 = call i64 @universe_hex_decode(ptr @g.dec, ptr @k.upper, i64 8)
  call void @ut_check_eq(i64 %d1, i64 4, ptr @m.kdecrc)
  %dc = call i32 @memcmp(ptr @g.dec, ptr @k.exp, i64 4)
  %dc.ok = icmp eq i32 %dc, 0
  call void @ut_check(i1 %dc.ok, ptr @m.kdec)

  ; ---- reject invalid ----
  %rb = call i64 @universe_hex_decode(ptr @g.dec, ptr @k.bad, i64 4)
  %rb.ok = icmp eq i64 %rb, -1
  call void @ut_check(i1 %rb.ok, ptr @m.bad)
  %ro = call i64 @universe_hex_decode(ptr @g.dec, ptr @k.odd, i64 3)
  %ro.ok = icmp eq i64 %ro, -1
  call void @ut_check(i1 %ro.ok, ptr @m.odd)

  ; ---- round-trip over many lengths ----
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  br label %rt.head

rt.head:
  %li = phi i64 [ 0, %entry ], [ %li.next, %rt.next ]
  %rtmis = phi i64 [ 0, %entry ], [ %rtmis.next, %rt.next ]
  %rtlenmis = phi i64 [ 0, %entry ], [ %rtlenmis.next, %rt.next ]
  %lp = getelementptr inbounds nuw [9 x i64], ptr @lens, i64 0, i64 %li
  %len = load i64, ptr %lp, align 8
  ; fill src with random bytes
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %rt.head ], [ %fi.next, %fill.body ]
  %fdone = icmp uge i64 %fi, %len
  br i1 %fdone, label %do.enc, label %fill.body

fill.body:
  %rv = call i64 @ut_rand(ptr %state)
  %rb8 = trunc i64 %rv to i8
  %sp = getelementptr inbounds nuw [65536 x i8], ptr @g.src, i64 0, i64 %fi
  store i8 %rb8, ptr %sp, align 1
  %fi.next = add nuw i64 %fi, 1
  br label %fill.head

do.enc:
  %enclen = call i64 @universe_hex_encode(ptr @g.enc, ptr @g.src, i64 %len)
  %declen = call i64 @universe_hex_decode(ptr @g.dec, ptr @g.enc, i64 %enclen)
  %lenbad = icmp ne i64 %declen, %len
  %rtlenmis.b = zext i1 %lenbad to i64
  %rtlenmis.next = add nuw i64 %rtlenmis, %rtlenmis.b
  %cmp = call i32 @memcmp(ptr @g.dec, ptr @g.src, i64 %len)
  %cmpbad = icmp ne i32 %cmp, 0
  %rtmis.b = zext i1 %cmpbad to i64
  %rtmis.next = add nuw i64 %rtmis, %rtmis.b
  br label %rt.next

rt.next:
  %li.next = add nuw i64 %li, 1
  %more = icmp ult i64 %li.next, 9
  br i1 %more, label %rt.head, label %rt.done

rt.done:
  call void @ut_check_eq(i64 %rtmis.next, i64 0, ptr @m.rt)
  call void @ut_check_eq(i64 %rtlenmis.next, i64 0, ptr @m.rtlen)

  ; ---- oracle: library SIMD decode == independent scalar decode ----
  store i64 6364136223846793005, ptr %state, align 8
  br label %oc.head

oc.head:
  %oli = phi i64 [ 0, %rt.done ], [ %oli.n, %oc.next ]
  %vmis = phi i64 [ 0, %rt.done ], [ %vmis.c, %oc.next ]
  %bmis = phi i64 [ 0, %rt.done ], [ %bmis.n, %oc.next ]
  %olp = getelementptr inbounds nuw [8 x i64], ptr @olens, i64 0, i64 %oli
  %oL = load i64, ptr %olp, align 8
  %oh = lshr i64 %oL, 1
  br label %oc.fill.head

oc.fill.head:
  %ofi = phi i64 [ 0, %oc.head ], [ %ofi.n, %oc.fill.body ]
  %ofd = icmp uge i64 %ofi, %oh
  br i1 %ofd, label %oc.enc, label %oc.fill.body

oc.fill.body:
  %orv = call i64 @ut_rand(ptr %state)
  %orb = trunc i64 %orv to i8
  %osp = getelementptr inbounds nuw [65536 x i8], ptr @g.src, i64 0, i64 %ofi
  store i8 %orb, ptr %osp, align 1
  %ofi.n = add nuw i64 %ofi, 1
  br label %oc.fill.head

oc.enc:
  %oel = call i64 @universe_hex_encode(ptr @g.enc, ptr @g.src, i64 %oh)
  %oodd = and i64 %oli, 1
  %oisodd = icmp ne i64 %oodd, 0
  %ohas = icmp ne i64 %oL, 0
  %osmall = icmp ult i64 %oL, 100000
  %oc.a = and i1 %oisodd, %ohas
  %oc.b = and i1 %oc.a, %osmall
  br i1 %oc.b, label %oc.corrupt, label %oc.dec

oc.corrupt:
  store i8 122, ptr @g.enc, align 1                ; 'z' -> not a hex char
  br label %oc.dec

oc.dec:
  %orl = call i64 @universe_hex_decode(ptr @g.dec, ptr @g.enc, i64 %oL)
  %orr = call i64 @ref_hex_decode(ptr @g.ref, ptr @g.enc, i64 %oL)
  %ovne = icmp ne i64 %orl, %orr
  %ovb = zext i1 %ovne to i64
  %vmis.c = add nuw i64 %vmis, %ovb
  %ovalid = icmp ne i64 %orr, -1
  br i1 %ovalid, label %oc.cmp, label %oc.nocmp

oc.cmp:
  %ocm = call i32 @memcmp(ptr @g.dec, ptr @g.ref, i64 %orr)
  %ocmne = icmp ne i32 %ocm, 0
  %ocb = zext i1 %ocmne to i64
  br label %oc.next

oc.nocmp:
  br label %oc.next

oc.next:
  %ocbp = phi i64 [ %ocb, %oc.cmp ], [ 0, %oc.nocmp ]
  %bmis.n = add nuw i64 %bmis, %ocbp
  %oli.n = add nuw i64 %oli, 1
  %omore = icmp ult i64 %oli.n, 8
  br i1 %omore, label %oc.head, label %oc.fin

oc.fin:
  call void @ut_check_eq(i64 %vmis.c, i64 0, ptr @m.orv)
  call void @ut_check_eq(i64 %bmis.n, i64 0, ptr @m.orb)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; fill 65536 bytes
  store i64 1, ptr %state, align 8
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
  ; ENCODE: 17 reps of a 1000-encode batch; discard rep 0 (warm-up), report the
  ; distribution over the remaining 16. ops/rep = 1000 * 65536 = 65536000 (ns/byte).
  br label %enc.rep

enc.rep:
  %erep = phi i64 [ 0, %bench.run ], [ %erep.n, %enc.next ]
  %et0 = call double @ut_now_sec()
  br label %be.head

be.head:
  %bec = phi i64 [ 0, %enc.rep ], [ %bec.next, %be.head ]
  %be = call i64 @universe_hex_encode(ptr @g.enc, ptr @g.src, i64 65536)
  %bec.next = add nuw i64 %bec, 1
  %bemore = icmp ult i64 %bec.next, 1000
  br i1 %bemore, label %be.head, label %enc.rep.done

enc.rep.done:
  %et1 = call double @ut_now_sec()
  %eel = fsub double %et1, %et0
  %ekeep = icmp ugt i64 %erep, 0
  br i1 %ekeep, label %enc.store, label %enc.next

enc.store:
  %eidx = sub i64 %erep, 1
  %esp = getelementptr inbounds [16 x double], ptr @hex.encsamp, i64 0, i64 %eidx
  store double %eel, ptr %esp, align 8
  br label %enc.next

enc.next:
  %erep.n = add nuw i64 %erep, 1
  %erm = icmp ult i64 %erep.n, 17
  br i1 %erm, label %enc.rep, label %enc.report

enc.report:
  call void @ut_report_dist(ptr @hex.encsamp, i64 16, i64 65536000, ptr @lbl.hexenc)
  br label %dec.rep

dec.rep:
  %drep = phi i64 [ 0, %enc.report ], [ %drep.n, %dec.next ]
  %dt0 = call double @ut_now_sec()
  br label %bd.head

bd.head:
  %bdc = phi i64 [ 0, %dec.rep ], [ %bdc.next, %bd.head ]
  %bd = call i64 @universe_hex_decode(ptr @g.dec, ptr @g.enc, i64 131072)
  %bdc.next = add nuw i64 %bdc, 1
  %bdmore = icmp ult i64 %bdc.next, 1000
  br i1 %bdmore, label %bd.head, label %dec.rep.done

dec.rep.done:
  %dt1 = call double @ut_now_sec()
  %del = fsub double %dt1, %dt0
  %dkeep = icmp ugt i64 %drep, 0
  br i1 %dkeep, label %dec.store, label %dec.next

dec.store:
  %didx = sub i64 %drep, 1
  %dsp = getelementptr inbounds [16 x double], ptr @hex.decsamp, i64 0, i64 %didx
  store double %del, ptr %dsp, align 8
  br label %dec.next

dec.next:
  %drep.n = add nuw i64 %drep, 1
  %drm = icmp ult i64 %drep.n, 17
  br i1 %drm, label %dec.rep, label %dec.report

dec.report:
  call void @ut_report_dist(ptr @hex.decsamp, i64 16, i64 65536000, ptr @lbl.hexdec)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
