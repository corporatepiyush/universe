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

; Tests for the minimal perfect hash (universe_ds_mph_*).

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@mph.bench.samp   = internal global [16 x double] zeroinitializer, align 8
@mphsw.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.mph.bench   = private unnamed_addr constant [15 x i8] c"mph lookup 10k\00"
@lbl.mphsw.bench = private unnamed_addr constant [17 x i8] c"swiss lookup 10k\00"

declare ptr @universe_ds_mph_build(ptr, ptr, i64)
declare i64 @universe_ds_mph_lookup(ptr, ptr, i64)
declare i64 @universe_ds_mph_lookup_checked(ptr, ptr, i64)
declare i64 @universe_ds_mph_slot_count(ptr)
declare void @universe_ds_mph_destroy(ptr)

; swiss hashmap (i64 keys) used only as the bench reference.
declare ptr @universe_ds_hashmap_swiss_create(i64, i64)
declare i32 @universe_ds_hashmap_swiss_put(ptr, i64, ptr)
declare i32 @universe_ds_hashmap_swiss_get(ptr, i64, ptr)
declare void @universe_ds_hashmap_swiss_destroy(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
declare i32 @printf(ptr, ...)

@sink = internal global i64 0, align 8

; Mirror of mph's 8-byte-key string hash (seed 0) so the swiss reference pays the
; SAME string->i64 cost that real "string -> id" use would: you cannot key swiss
; on a variable-length string without first hashing it. This makes the bench a
; like-for-like probe-free-vs-SIMD-probe comparison over identical inputs.
define internal i64 @t_strhash8(i64 %w) {
entry:
  %wx = xor i64 -3750763034362895579, %w
  %accn = mul i64 %wx, 1099511628211
  %mixed = xor i64 %accn, 8
  %a1 = lshr i64 %mixed, 30
  %a2 = xor i64 %a1, %mixed
  %a3 = mul i64 %a2, -4658895280553007687
  %a4 = lshr i64 %a3, 27
  %a5 = xor i64 %a4, %a3
  %a6 = mul i64 %a5, -7723592293110705685
  %a7 = lshr i64 %a6, 31
  %a8 = xor i64 %a7, %a6
  ret i64 %a8
}

; ---- HTTP header name key set ----------------------------------------------
@h.host    = private constant [4 x i8]  c"host"
@h.accept  = private constant [6 x i8]  c"accept"
@h.acenc   = private constant [15 x i8] c"accept-encoding"
@h.aclang  = private constant [15 x i8] c"accept-language"
@h.ua      = private constant [10 x i8] c"user-agent"
@h.ctype   = private constant [12 x i8] c"content-type"
@h.clen    = private constant [14 x i8] c"content-length"
@h.auth    = private constant [13 x i8] c"authorization"
@h.cookie  = private constant [6 x i8]  c"cookie"
@h.scookie = private constant [10 x i8] c"set-cookie"
@h.cache   = private constant [13 x i8] c"cache-control"
@h.conn    = private constant [10 x i8] c"connection"
@h.date    = private constant [4 x i8]  c"date"
@h.etag    = private constant [4 x i8]  c"etag"
@h.loc     = private constant [8 x i8]  c"location"
@h.ref     = private constant [7 x i8]  c"referer"
@h.server  = private constant [6 x i8]  c"server"
@h.vary    = private constant [4 x i8]  c"vary"
@h.age     = private constant [3 x i8]  c"age"
@h.allow   = private constant [5 x i8]  c"allow"

@hdr.keys = private constant [20 x ptr] [
  ptr @h.host, ptr @h.accept, ptr @h.acenc, ptr @h.aclang, ptr @h.ua,
  ptr @h.ctype, ptr @h.clen, ptr @h.auth, ptr @h.cookie, ptr @h.scookie,
  ptr @h.cache, ptr @h.conn, ptr @h.date, ptr @h.etag, ptr @h.loc,
  ptr @h.ref, ptr @h.server, ptr @h.vary, ptr @h.age, ptr @h.allow ]
@hdr.lens = private constant [20 x i64] [
  i64 4, i64 6, i64 15, i64 15, i64 10, i64 12, i64 14, i64 13, i64 6, i64 10,
  i64 13, i64 10, i64 4, i64 4, i64 8, i64 7, i64 6, i64 4, i64 3, i64 5 ]

; non-members (not in the header set)
@nm.a = private constant [8 x i8]  c"x-custom"
@nm.b = private constant [6 x i8]  c"banana"
@nm.c = private constant [11 x i8] c"content-typ"
@nm.d = private constant [5 x i8]  c"hostx"
@nm.keys = private constant [4 x ptr] [ ptr @nm.a, ptr @nm.b, ptr @nm.c, ptr @nm.d ]
@nm.lens = private constant [4 x i64] [ i64 8, i64 6, i64 11, i64 5 ]

; shared-prefix key set (7-char common prefix, distinct last char)
@sp.0 = private constant [8 x i8] c"prefixaa"
@sp.1 = private constant [8 x i8] c"prefixab"
@sp.2 = private constant [8 x i8] c"prefixac"
@sp.3 = private constant [8 x i8] c"prefixad"
@sp.4 = private constant [8 x i8] c"prefixae"
@sp.5 = private constant [9 x i8] c"prefixaaa"
@sp.keys = private constant [6 x ptr] [ ptr @sp.0, ptr @sp.1, ptr @sp.2, ptr @sp.3, ptr @sp.4, ptr @sp.5 ]
@sp.lens = private constant [6 x i64] [ i64 8, i64 8, i64 8, i64 8, i64 8, i64 9 ]

@msg.nonnull  = private constant [23 x i8] c"build returned nonnull\00"
@msg.count    = private constant [17 x i8] c"slot_count == n \00"
@msg.range    = private constant [20 x i8] c"all slots in [0,n) \00"
@msg.det      = private constant [21 x i8] c"lookup deterministic\00"
@msg.chk      = private constant [24 x i8] c"checked member == slot \00"
@msg.cover    = private constant [22 x i8] c"bijection: each once \00"
@msg.n1       = private constant [13 x i8] c"N=1 build   \00"
@msg.n2       = private constant [13 x i8] c"N=2 build   \00"
@msg.dup      = private constant [22 x i8] c"duplicate keys reject\00"
@msg.nm       = private constant [22 x i8] c"non-members return -1\00"
@msg.mem      = private constant [22 x i8] c"members via checked  \00"


; ---------------------------------------------------------------------------
; verify_build: build over (keys,lens,n); assert bijection + determinism +
; checked-member identity; then destroy.
; ---------------------------------------------------------------------------
define void @verify_build(ptr %keys, ptr %lens, i64 %n) {
entry:
  %m = call ptr @universe_ds_mph_build(ptr %keys, ptr %lens, i64 %n)
  %mnn = icmp ne ptr %m, null
  call void @ut_check(i1 %mnn, ptr @msg.nonnull)
  br i1 %mnn, label %go, label %ret

go:
  %sc = call i64 @universe_ds_mph_slot_count(ptr %m)
  call void @ut_check_eq(i64 %sc, i64 %n, ptr @msg.count)
  %counts = call ptr @malloc(i64 %n)
  call void @llvm.memset.p0.i64(ptr %counts, i8 0, i64 %n, i1 false)
  %vr = alloca i64, align 8
  %vd = alloca i64, align 8
  %vc = alloca i64, align 8
  store i64 0, ptr %vr, align 8
  store i64 0, ptr %vd, align 8
  store i64 0, ptr %vc, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %go ], [ %i.next, %cont ]
  %idone = icmp eq i64 %i, %n
  br i1 %idone, label %cover, label %body

body:
  %kp = getelementptr inbounds ptr, ptr %keys, i64 %i
  %key = load ptr, ptr %kp, align 8
  %lp = getelementptr inbounds i64, ptr %lens, i64 %i
  %len = load i64, ptr %lp, align 8
  %slot = call i64 @universe_ds_mph_lookup(ptr %m, ptr %key, i64 %len)
  %ge0 = icmp sge i64 %slot, 0
  %ltn = icmp slt i64 %slot, %n
  %inrange = and i1 %ge0, %ltn
  br i1 %inrange, label %counted, label %oor

oor:
  %vr0 = load i64, ptr %vr, align 8
  %vr1 = add i64 %vr0, 1
  store i64 %vr1, ptr %vr, align 8
  br label %after

counted:
  %cptr = getelementptr inbounds i8, ptr %counts, i64 %slot
  %cv = load i8, ptr %cptr, align 1
  %cv1 = add i8 %cv, 1
  store i8 %cv1, ptr %cptr, align 1
  br label %after

after:
  %slot2 = call i64 @universe_ds_mph_lookup(ptr %m, ptr %key, i64 %len)
  %deteq = icmp eq i64 %slot2, %slot
  br i1 %deteq, label %detok, label %detbad

detbad:
  %vd0 = load i64, ptr %vd, align 8
  %vd1 = add i64 %vd0, 1
  store i64 %vd1, ptr %vd, align 8
  br label %detok

detok:
  %cs = call i64 @universe_ds_mph_lookup_checked(ptr %m, ptr %key, i64 %len)
  %chkeq = icmp eq i64 %cs, %slot
  br i1 %chkeq, label %cont, label %chkbad

chkbad:
  %vc0 = load i64, ptr %vc, align 8
  %vc1 = add i64 %vc0, 1
  store i64 %vc1, ptr %vc, align 8
  br label %cont

cont:
  %i.next = add nuw i64 %i, 1
  br label %loop

cover:
  %covv = alloca i64, align 8
  store i64 0, ptr %covv, align 8
  br label %cloop

cloop:
  %ci = phi i64 [ 0, %cover ], [ %ci.next, %ccont ]
  %cidone = icmp eq i64 %ci, %n
  br i1 %cidone, label %report, label %cbody

cbody:
  %ccptr = getelementptr inbounds i8, ptr %counts, i64 %ci
  %ccv = load i8, ptr %ccptr, align 1
  %isone = icmp eq i8 %ccv, 1
  br i1 %isone, label %ccont, label %covbad

covbad:
  %cov0 = load i64, ptr %covv, align 8
  %cov1 = add i64 %cov0, 1
  store i64 %cov1, ptr %covv, align 8
  br label %ccont

ccont:
  %ci.next = add nuw i64 %ci, 1
  br label %cloop

report:
  %fvr = load i64, ptr %vr, align 8
  call void @ut_check_eq(i64 %fvr, i64 0, ptr @msg.range)
  %fvd = load i64, ptr %vd, align 8
  call void @ut_check_eq(i64 %fvd, i64 0, ptr @msg.det)
  %fvc = load i64, ptr %vc, align 8
  call void @ut_check_eq(i64 %fvc, i64 0, ptr @msg.chk)
  %fcov = load i64, ptr %covv, align 8
  call void @ut_check_eq(i64 %fcov, i64 0, ptr @msg.cover)
  call void @free(ptr %counts)
  call void @universe_ds_mph_destroy(ptr %m)
  br label %ret

ret:
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; --- header set: full bijection verification ---
  call void @verify_build(ptr @hdr.keys, ptr @hdr.lens, i64 20)
  ; --- shared-prefix set ---
  call void @verify_build(ptr @sp.keys, ptr @sp.lens, i64 6)

  ; --- membership: header set, members hit / non-members miss ---
  %hm = call ptr @universe_ds_mph_build(ptr @hdr.keys, ptr @hdr.lens, i64 20)
  %hmnn = icmp ne ptr %hm, null
  call void @ut_check(i1 %hmnn, ptr @msg.nonnull)
  ; members
  %memv = alloca i64, align 8
  store i64 0, ptr %memv, align 8
  br label %mloop

mloop:
  %mi = phi i64 [ 0, %entry ], [ %mi.next, %mcont ]
  %midone = icmp eq i64 %mi, 20
  br i1 %midone, label %mdone, label %mbody

mbody:
  %mkp = getelementptr inbounds ptr, ptr @hdr.keys, i64 %mi
  %mkey = load ptr, ptr %mkp, align 8
  %mlp = getelementptr inbounds i64, ptr @hdr.lens, i64 %mi
  %mlen = load i64, ptr %mlp, align 8
  %mcs = call i64 @universe_ds_mph_lookup_checked(ptr %hm, ptr %mkey, i64 %mlen)
  %misok = icmp sge i64 %mcs, 0
  br i1 %misok, label %mcont, label %mbad

mbad:
  %mv0 = load i64, ptr %memv, align 8
  %mv1 = add i64 %mv0, 1
  store i64 %mv1, ptr %memv, align 8
  br label %mcont

mcont:
  %mi.next = add nuw i64 %mi, 1
  br label %mloop

mdone:
  %fmem = load i64, ptr %memv, align 8
  call void @ut_check_eq(i64 %fmem, i64 0, ptr @msg.mem)
  ; non-members
  %nmv = alloca i64, align 8
  store i64 0, ptr %nmv, align 8
  br label %nloop

nloop:
  %ni = phi i64 [ 0, %mdone ], [ %ni.next, %ncont ]
  %nidone = icmp eq i64 %ni, 4
  br i1 %nidone, label %ndone, label %nbody

nbody:
  %nkp = getelementptr inbounds ptr, ptr @nm.keys, i64 %ni
  %nkey = load ptr, ptr %nkp, align 8
  %nlp = getelementptr inbounds i64, ptr @nm.lens, i64 %ni
  %nlen = load i64, ptr %nlp, align 8
  %ncs = call i64 @universe_ds_mph_lookup_checked(ptr %hm, ptr %nkey, i64 %nlen)
  %ismiss = icmp eq i64 %ncs, -1
  br i1 %ismiss, label %ncont, label %nbad

nbad:
  %nv0 = load i64, ptr %nmv, align 8
  %nv1 = add i64 %nv0, 1
  store i64 %nv1, ptr %nmv, align 8
  br label %ncont

ncont:
  %ni.next = add nuw i64 %ni, 1
  br label %nloop

ndone:
  %fnm = load i64, ptr %nmv, align 8
  call void @ut_check_eq(i64 %fnm, i64 0, ptr @msg.nm)
  call void @universe_ds_mph_destroy(ptr %hm)

  ; --- edge N=1 and N=2 ---
  %e1keys = alloca [1 x ptr], align 8
  %e1lens = alloca [1 x i64], align 8
  %e1p0 = getelementptr inbounds [1 x ptr], ptr %e1keys, i64 0, i64 0
  store ptr @h.host, ptr %e1p0, align 8
  %e1l0 = getelementptr inbounds [1 x i64], ptr %e1lens, i64 0, i64 0
  store i64 4, ptr %e1l0, align 8
  %m1 = call ptr @universe_ds_mph_build(ptr %e1keys, ptr %e1lens, i64 1)
  %m1nn = icmp ne ptr %m1, null
  call void @ut_check(i1 %m1nn, ptr @msg.n1)
  %s1 = call i64 @universe_ds_mph_lookup(ptr %m1, ptr @h.host, i64 4)
  %s1ok = icmp eq i64 %s1, 0
  call void @ut_check(i1 %s1ok, ptr @msg.n1)
  call void @universe_ds_mph_destroy(ptr %m1)

  %e2keys = alloca [2 x ptr], align 8
  %e2lens = alloca [2 x i64], align 8
  %e2p0 = getelementptr inbounds [2 x ptr], ptr %e2keys, i64 0, i64 0
  store ptr @h.host, ptr %e2p0, align 8
  %e2p1 = getelementptr inbounds [2 x ptr], ptr %e2keys, i64 0, i64 1
  store ptr @h.accept, ptr %e2p1, align 8
  %e2l0 = getelementptr inbounds [2 x i64], ptr %e2lens, i64 0, i64 0
  store i64 4, ptr %e2l0, align 8
  %e2l1 = getelementptr inbounds [2 x i64], ptr %e2lens, i64 0, i64 1
  store i64 6, ptr %e2l1, align 8
  %m2 = call ptr @universe_ds_mph_build(ptr %e2keys, ptr %e2lens, i64 2)
  %m2nn = icmp ne ptr %m2, null
  call void @ut_check(i1 %m2nn, ptr @msg.n2)
  call void @verify_build(ptr %e2keys, ptr %e2lens, i64 2)
  call void @universe_ds_mph_destroy(ptr %m2)

  ; --- duplicate keys rejected ---
  %dkeys = alloca [3 x ptr], align 8
  %dlens = alloca [3 x i64], align 8
  %dp0 = getelementptr inbounds [3 x ptr], ptr %dkeys, i64 0, i64 0
  store ptr @h.host, ptr %dp0, align 8
  %dp1 = getelementptr inbounds [3 x ptr], ptr %dkeys, i64 0, i64 1
  store ptr @h.accept, ptr %dp1, align 8
  %dp2 = getelementptr inbounds [3 x ptr], ptr %dkeys, i64 0, i64 2
  store ptr @h.host, ptr %dp2, align 8              ; duplicate of [0]
  %dl0 = getelementptr inbounds [3 x i64], ptr %dlens, i64 0, i64 0
  store i64 4, ptr %dl0, align 8
  %dl1 = getelementptr inbounds [3 x i64], ptr %dlens, i64 0, i64 1
  store i64 6, ptr %dl1, align 8
  %dl2 = getelementptr inbounds [3 x i64], ptr %dlens, i64 0, i64 2
  store i64 4, ptr %dl2, align 8
  %dm = call ptr @universe_ds_mph_build(ptr %dkeys, ptr %dlens, i64 3)
  %disnull = icmp eq ptr %dm, null
  call void @ut_check(i1 %disnull, ptr @msg.dup)

  ; --- large randomized set: 10000 distinct 8-byte keys ---
  %N = add i64 0, 10000
  %buf = call ptr @malloc(i64 80000)             ; N*8 key bytes
  %lkeys = call ptr @malloc(i64 80000)           ; N ptr
  %llens = call ptr @malloc(i64 80000)           ; N i64
  br label %fill

fill:
  %fi = phi i64 [ 0, %ndone ], [ %fi.next, %fillbody ]
  %fdone = icmp eq i64 %fi, %N
  br i1 %fdone, label %largebuild, label %fillbody

fillbody:
  %off = shl nuw i64 %fi, 3
  %kb = getelementptr inbounds i8, ptr %buf, i64 %off
  ; distinct content: (fi * golden) ^ fi, guarantees uniqueness via bijective mix
  %mix = mul i64 %fi, -7046029254386353131
  %content = xor i64 %mix, %fi
  store i64 %content, ptr %kb, align 8
  %kpp = getelementptr inbounds ptr, ptr %lkeys, i64 %fi
  store ptr %kb, ptr %kpp, align 8
  %lpp = getelementptr inbounds i64, ptr %llens, i64 %fi
  store i64 8, ptr %lpp, align 8
  %fi.next = add nuw i64 %fi, 1
  br label %fill

largebuild:
  call void @verify_build(ptr %lkeys, ptr %llens, i64 %N)

  ; --- bench (optional) ---
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %afterbench

bench:
  %bm = call ptr @universe_ds_mph_build(ptr %lkeys, ptr %llens, i64 %N)
  %sm = call ptr @universe_ds_hashmap_swiss_create(i64 8, i64 %N)
  %vslot = alloca i64, align 8
  br label %swfill

swfill:
  %si = phi i64 [ 0, %bench ], [ %si.next, %swbody ]
  %sdone = icmp eq i64 %si, %N
  br i1 %sdone, label %mph.rep.head, label %swbody

swbody:
  %soff = shl nuw i64 %si, 3
  %skb = getelementptr inbounds i8, ptr %buf, i64 %soff
  %sw = load i64, ptr %skb, align 8
  %skey = call i64 @t_strhash8(i64 %sw)
  store i64 %si, ptr %vslot, align 8
  %pr = call i32 @universe_ds_hashmap_swiss_put(ptr %sm, i64 %skey, ptr %vslot)
  %si.next = add nuw i64 %si, 1
  br label %swfill

; --- MPH lookup distribution: one rep = one pass of N lookups (warm-up discarded) ---
mph.rep.head:
  %mrep = phi i64 [ 0, %swfill ], [ %mrep.n, %mph.rep.cont ]
  %t0 = call double @ut_now_sec()
  br label %mkloop

mkloop:
  %mki = phi i64 [ 0, %mph.rep.head ], [ %mki.next, %mkbody ]
  %mkacc = phi i64 [ 0, %mph.rep.head ], [ %mkacc.n, %mkbody ]
  %mkdone = icmp eq i64 %mki, %N
  br i1 %mkdone, label %mk.rep.done, label %mkbody

mkbody:
  %mkpp = getelementptr inbounds ptr, ptr %lkeys, i64 %mki
  %mkkey = load ptr, ptr %mkpp, align 8
  %mkslot = call i64 @universe_ds_mph_lookup(ptr %bm, ptr %mkkey, i64 8)
  %mkacc.n = add i64 %mkacc, %mkslot
  %mki.next = add nuw i64 %mki, 1
  br label %mkloop

mk.rep.done:
  store volatile i64 %mkacc, ptr @sink, align 8
  %t1 = call double @ut_now_sec()
  %mph.dt = fsub double %t1, %t0
  %mph.warm = icmp eq i64 %mrep, 0
  br i1 %mph.warm, label %mph.rep.cont, label %mph.rep.store
mph.rep.store:
  %mph.si = sub i64 %mrep, 1
  %mph.sp = getelementptr inbounds double, ptr @mph.bench.samp, i64 %mph.si
  store double %mph.dt, ptr %mph.sp, align 8
  br label %mph.rep.cont
mph.rep.cont:
  %mrep.n = add nuw i64 %mrep, 1
  %mph.more = icmp ult i64 %mrep.n, 17
  br i1 %mph.more, label %mph.rep.head, label %mph.report
mph.report:
  call void @ut_report_dist(ptr @mph.bench.samp, i64 16, i64 %N, ptr @lbl.mph.bench)
  br label %sw.rep.head

; --- swiss lookup distribution: one rep = one pass of N gets ---
sw.rep.head:
  %srep = phi i64 [ 0, %mph.report ], [ %srep.n, %sw.rep.cont ]
  %t2 = call double @ut_now_sec()
  br label %skloop

skloop:
  %ski2 = phi i64 [ 0, %sw.rep.head ], [ %ski2.next, %skbody ]
  %skacc = phi i64 [ 0, %sw.rep.head ], [ %skacc.n, %skbody ]
  %skdone = icmp eq i64 %ski2, %N
  br i1 %skdone, label %sk.rep.done, label %skbody

skbody:
  %skoff = shl nuw i64 %ski2, 3
  %skb2 = getelementptr inbounds i8, ptr %buf, i64 %skoff
  %sw2 = load i64, ptr %skb2, align 8
  %skey2 = call i64 @t_strhash8(i64 %sw2)
  %gr = call i32 @universe_ds_hashmap_swiss_get(ptr %sm, i64 %skey2, ptr %vslot)
  %gv = load i64, ptr %vslot, align 8
  %skacc.n = add i64 %skacc, %gv
  %ski2.next = add nuw i64 %ski2, 1
  br label %skloop

sk.rep.done:
  store volatile i64 %skacc, ptr @sink, align 8
  %t3 = call double @ut_now_sec()
  %sw.dt = fsub double %t3, %t2
  %sw.warm = icmp eq i64 %srep, 0
  br i1 %sw.warm, label %sw.rep.cont, label %sw.rep.store
sw.rep.store:
  %sw.si = sub i64 %srep, 1
  %sw.sp = getelementptr inbounds double, ptr @mphsw.bench.samp, i64 %sw.si
  store double %sw.dt, ptr %sw.sp, align 8
  br label %sw.rep.cont
sw.rep.cont:
  %srep.n = add nuw i64 %srep, 1
  %sw.more = icmp ult i64 %srep.n, 17
  br i1 %sw.more, label %sw.rep.head, label %sw.report
sw.report:
  call void @ut_report_dist(ptr @mphsw.bench.samp, i64 16, i64 %N, ptr @lbl.mphsw.bench)
  call void @universe_ds_mph_destroy(ptr %bm)
  call void @universe_ds_hashmap_swiss_destroy(ptr %sm)
  br label %afterbench

afterbench:
  call void @free(ptr %buf)
  call void @free(ptr %lkeys)
  call void @free(ptr %llens)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
