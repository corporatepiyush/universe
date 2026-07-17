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

; Tests for universe_alloc_tlsf. Shadow accounting oracle: per-slot stamps prove
; no two live blocks overlap; a per-alloc `live` delta is verified equal to the
; drop on free; coalescing is proven by draining the pool into min blocks,
; freeing all, then a single large alloc succeeding; alignment honored;
; exhaustion returns null; edge sizes (0,1,min,huge); caller-region + owned.

declare ptr @universe_alloc_tlsf_create(ptr, i64)
declare ptr @universe_alloc_tlsf_alloc(ptr, i64)
declare ptr @universe_alloc_tlsf_alloc_aligned(ptr, i64, i64)
declare void @universe_alloc_tlsf_free(ptr, ptr)
declare i64 @universe_alloc_tlsf_live(ptr)
declare void @universe_alloc_tlsf_destroy(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

@g.ptr   = internal global [4096 x ptr] zeroinitializer, align 16
@g.size  = internal global [256 x i64] zeroinitializer, align 16
@g.tag   = internal global [256 x i8]  zeroinitializer, align 16
@g.delta = internal global [256 x i64] zeroinitializer, align 16

@tlsf.samp   = internal global [16 x double] zeroinitializer, align 8
@malloc.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.tlsf    = private unnamed_addr constant [24 x i8] c"tlsf alloc/free 64+512B\00"
@lbl.malloc  = private unnamed_addr constant [26 x i8] c"malloc alloc/free 64+512B\00"

@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.align    = private unnamed_addr constant [17 x i8] c"payload 16-align\00"
@m.stamps   = private unnamed_addr constant [21 x i8] c"stamps intact (basic\00"
@m.live0    = private unnamed_addr constant [7 x i8] c"live 0\00"
@m.callreg  = private unnamed_addr constant [20 x i8] c"caller-region alloc\00"
@m.z0       = private unnamed_addr constant [17 x i8] c"alloc(0) nonnull\00"
@m.z1       = private unnamed_addr constant [17 x i8] c"alloc(1) nonnull\00"
@m.huge     = private unnamed_addr constant [16 x i8] c"huge alloc null\00"
@m.small    = private unnamed_addr constant [21 x i8] c"tiny create -> null \00"
@m.zero     = private unnamed_addr constant [21 x i8] c"zero create -> null \00"
@m.aalign   = private unnamed_addr constant [21 x i8] c"aligned addr honored\00"
@m.astamp   = private unnamed_addr constant [21 x i8] c"aligned no overlap  \00"
@m.drain    = private unnamed_addr constant [21 x i8] c"drain then exhausted\00"
@m.dstamp   = private unnamed_addr constant [21 x i8] c"drain stamps intact \00"
@m.coal     = private unnamed_addr constant [25 x i8] c"coalesce -> big reusable\00"
@m.overlap  = private unnamed_addr constant [20 x i8] c"churn: no overlap  \00"
@m.acct     = private unnamed_addr constant [23 x i8] c"churn: live accounting\00"
@m.churn0   = private unnamed_addr constant [23 x i8] c"churn end live 0/whole\00"

; ---------------------------------------------------------------------------

define internal void @test_basic() {
entry:
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 1052672)   ; ~1MB pool
  %ok = icmp ne ptr %h, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %go, label %done

go:
  %p1 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 100)
  %p2 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 5000)
  %p3 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 40)
  %n1 = icmp ne ptr %p1, null
  %n2 = icmp ne ptr %p2, null
  %n3 = icmp ne ptr %p3, null
  %na = and i1 %n1, %n2
  %nb = and i1 %na, %n3
  call void @ut_check(i1 %nb, ptr @m.create)
  ; 16-alignment
  %i1 = ptrtoint ptr %p1 to i64
  %i2 = ptrtoint ptr %p2 to i64
  %i3 = ptrtoint ptr %p3 to i64
  %o = or i64 %i1, %i2
  %o2 = or i64 %o, %i3
  %am = and i64 %o2, 15
  %aok = icmp eq i64 %am, 0
  call void @ut_check(i1 %aok, ptr @m.align)
  ; distinct stamps, verify no overlap
  call void @llvm.memset.p0.i64(ptr %p1, i8 -95, i64 100, i1 false)
  call void @llvm.memset.p0.i64(ptr %p2, i8 -78, i64 5000, i1 false)
  call void @llvm.memset.p0.i64(ptr %p3, i8 -61, i64 40, i1 false)
  %e1a = load i8, ptr %p1, align 1
  %p1e = getelementptr inbounds i8, ptr %p1, i64 99
  %e1b = load i8, ptr %p1e, align 1
  %e2a = load i8, ptr %p2, align 1
  %p2e = getelementptr inbounds i8, ptr %p2, i64 4999
  %e2b = load i8, ptr %p2e, align 1
  %e3a = load i8, ptr %p3, align 1
  %p3e = getelementptr inbounds i8, ptr %p3, i64 39
  %e3b = load i8, ptr %p3e, align 1
  %c1a = icmp eq i8 %e1a, -95
  %c1b = icmp eq i8 %e1b, -95
  %c2a = icmp eq i8 %e2a, -78
  %c2b = icmp eq i8 %e2b, -78
  %c3a = icmp eq i8 %e3a, -61
  %c3b = icmp eq i8 %e3b, -61
  %s1 = and i1 %c1a, %c1b
  %s2 = and i1 %c2a, %c2b
  %s3 = and i1 %c3a, %c3b
  %sa = and i1 %s1, %s2
  %sb = and i1 %sa, %s3
  call void @ut_check(i1 %sb, ptr @m.stamps)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p2)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p1)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p3)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  call void @universe_alloc_tlsf_destroy(ptr %h)
  br label %done

done:
  ret void
}

define internal void @test_caller_region() {
entry:
  ; caller-supplied region: destroy must NOT free it
  %buf = call ptr @malloc(i64 262144)
  %h = call ptr @universe_alloc_tlsf_create(ptr %buf, i64 262144)
  %ok = icmp ne ptr %h, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %go, label %freebuf

go:
  %p = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 1000)
  %n = icmp ne ptr %p, null
  call void @ut_check(i1 %n, ptr @m.callreg)
  br i1 %n, label %use, label %destroy

use:
  call void @llvm.memset.p0.i64(ptr %p, i8 55, i64 1000, i1 false)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  br label %destroy

destroy:
  call void @universe_alloc_tlsf_destroy(ptr %h)      ; must not free %buf
  br label %freebuf

freebuf:
  call void @free(ptr %buf)
  ret void
}

define internal void @test_edges() {
entry:
  ; too small / zero create -> null
  %ht = call ptr @universe_alloc_tlsf_create(ptr null, i64 100)
  %tn = icmp eq ptr %ht, null
  call void @ut_check(i1 %tn, ptr @m.small)
  %hz = call ptr @universe_alloc_tlsf_create(ptr null, i64 0)
  %zn = icmp eq ptr %hz, null
  call void @ut_check(i1 %zn, ptr @m.zero)
  ; edge sizes on a real pool
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 1052672)
  %q0 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 0)
  %q0n = icmp ne ptr %q0, null
  call void @ut_check(i1 %q0n, ptr @m.z0)
  %q1 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 1)
  %q1n = icmp ne ptr %q1, null
  call void @ut_check(i1 %q1n, ptr @m.z1)
  call void @llvm.memset.p0.i64(ptr %q1, i8 7, i64 1, i1 false)
  ; huge alloc (bigger than pool) -> null
  %qh = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 100000000)
  %qhn = icmp eq ptr %qh, null
  call void @ut_check(i1 %qhn, ptr @m.huge)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %q0)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %q1)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  call void @universe_alloc_tlsf_destroy(ptr %h)
  ret void
}

; one aligned alloc: alloc_aligned, check alignment + stamp/verify
define internal ptr @aligned_one(ptr %h, i64 %size, i64 %align, i8 %tag, ptr %viol) {
entry:
  %q = call ptr @universe_alloc_tlsf_alloc_aligned(ptr %h, i64 %size, i64 %align)
  %n = icmp ne ptr %q, null
  br i1 %n, label %chk, label %bad

chk:
  %qi = ptrtoint ptr %q to i64
  %am1 = add i64 %align, -1
  %masked = and i64 %qi, %am1
  %aligned = icmp eq i64 %masked, 0
  br i1 %aligned, label %stamp, label %bad

stamp:
  call void @llvm.memset.p0.i64(ptr %q, i8 %tag, i64 %size, i1 false)
  ret ptr %q

bad:
  %v = load i64, ptr %viol, align 8
  %v1 = add i64 %v, 1
  store i64 %v1, ptr %viol, align 8
  ret ptr null
}

define internal void @test_aligned() {
entry:
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 1052672)
  %viol = alloca i64, align 8
  store i64 0, ptr %viol, align 8
  %a = call ptr @aligned_one(ptr %h, i64 200,  i64 32,   i8 17, ptr %viol)
  %b = call ptr @aligned_one(ptr %h, i64 900,  i64 64,   i8 18, ptr %viol)
  %c = call ptr @aligned_one(ptr %h, i64 50,   i64 256,  i8 19, ptr %viol)
  %d = call ptr @aligned_one(ptr %h, i64 4000, i64 4096, i8 20, ptr %viol)
  %e = call ptr @aligned_one(ptr %h, i64 33,   i64 128,  i8 21, ptr %viol)
  %vv = load i64, ptr %viol, align 8
  %aok = icmp eq i64 %vv, 0
  call void @ut_check(i1 %aok, ptr @m.aalign)
  ; verify no overlap: re-read the first byte of each == its tag
  %ba = load i8, ptr %a, align 1
  %bb = load i8, ptr %b, align 1
  %bc = load i8, ptr %c, align 1
  %bd = load i8, ptr %d, align 1
  %be = load i8, ptr %e, align 1
  %ka = icmp eq i8 %ba, 17
  %kb = icmp eq i8 %bb, 18
  %kc = icmp eq i8 %bc, 19
  %kd = icmp eq i8 %bd, 20
  %ke = icmp eq i8 %be, 21
  %x1 = and i1 %ka, %kb
  %x2 = and i1 %x1, %kc
  %x3 = and i1 %x2, %kd
  %x4 = and i1 %x3, %ke
  call void @ut_check(i1 %x4, ptr @m.astamp)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %a)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %b)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %c)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %d)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %e)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  call void @universe_alloc_tlsf_destroy(ptr %h)
  ret void
}

; drain a 64KiB pool into 32B min blocks, verify stamps, free all, then a large
; alloc must succeed only if coalescing rebuilt one contiguous region.
define internal void @test_coalesce() {
entry:
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 69824)     ; ~64KiB pool
  %ok = icmp ne ptr %h, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %drain, label %done

drain:
  %i = phi i64 [ 0, %entry ], [ %i.n, %drain.cont ]
  %p = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 16)
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %drained, label %drain.store

drain.store:
  %slot = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %i
  store ptr %p, ptr %slot, align 8
  %tag = trunc i64 %i to i8
  store i8 %tag, ptr %p, align 1
  %pe = getelementptr inbounds i8, ptr %p, i64 15
  store i8 %tag, ptr %pe, align 1
  br label %drain.cont

drain.cont:
  %i.n = add nuw nsw i64 %i, 1
  %room = icmp ult i64 %i.n, 4096
  br i1 %room, label %drain, label %drained

drained:
  %n = phi i64 [ %i, %drain ], [ %i.n, %drain.cont ]
  ; exhausted returns null (we hit it) AND n>0
  %pos = icmp ugt i64 %n, 0
  call void @ut_check(i1 %pos, ptr @m.drain)
  br label %verify

verify:
  %j = phi i64 [ 0, %drained ], [ %j.n, %verify ]
  %vbad = phi i64 [ 0, %drained ], [ %vbad.n, %verify ]
  %vslot = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %j
  %q = load ptr, ptr %vslot, align 8
  %want = trunc i64 %j to i8
  %b0 = load i8, ptr %q, align 1
  %qe = getelementptr inbounds i8, ptr %q, i64 15
  %b15 = load i8, ptr %qe, align 1
  %ok0 = icmp eq i8 %b0, %want
  %ok15 = icmp eq i8 %b15, %want
  %okv = and i1 %ok0, %ok15
  %badb = xor i1 %okv, true
  %badi = zext i1 %badb to i64
  %vbad.n = add nuw i64 %vbad, %badi
  %j.n = add nuw nsw i64 %j, 1
  %vmore = icmp ult i64 %j.n, %n
  br i1 %vmore, label %verify, label %freeall

freeall:                                         ; free every block
  %k = phi i64 [ 0, %verify ], [ %k.n, %freeall ]
  %fslot = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %k
  %fq = load ptr, ptr %fslot, align 8
  call void @universe_alloc_tlsf_free(ptr %h, ptr %fq)
  %k.n = add nuw nsw i64 %k, 1
  %fmore = icmp ult i64 %k.n, %n
  br i1 %fmore, label %freeall, label %grand

grand:
  call void @ut_check_eq(i64 %vbad, i64 0, ptr @m.dstamp)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @ut_check_eq(i64 %live, i64 0, ptr @m.live0)
  ; a 60000B alloc succeeds only if the min blocks coalesced back to one region
  %big = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 60000)
  %bigok = icmp ne ptr %big, null
  call void @ut_check(i1 %bigok, ptr @m.coal)
  br i1 %bigok, label %freebig, label %done

freebig:
  call void @universe_alloc_tlsf_free(ptr %h, ptr %big)
  br label %done

done:
  %hok = icmp ne ptr %h, null
  br i1 %hok, label %destroy, label %ret

destroy:
  call void @universe_alloc_tlsf_destroy(ptr %h)
  br label %ret

ret:
  ret void
}

; random alloc/free churn over 256 slots with per-slot stamp + live-delta oracle.
define internal void @test_churn() {
entry:
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 2101248)   ; ~2MB pool
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  br label %clr

clr:                                             ; prior tests left stale ptrs in g.ptr
  %ci = phi i64 [ 0, %entry ], [ %ci.n, %clr ]
  %cp = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %ci
  store ptr null, ptr %cp, align 8
  %ci.n = add nuw nsw i64 %ci, 1
  %cmore = icmp ult i64 %ci.n, 256
  br i1 %cmore, label %clr, label %loop

loop:
  %i = phi i64 [ 0, %clr ], [ %i.n, %loop.cont ]
  %ov = phi i64 [ 0, %clr ], [ %ov.n, %loop.cont ]
  %av = phi i64 [ 0, %clr ], [ %av.n, %loop.cont ]
  %r = call i64 @ut_rand(ptr nonnull %seed)
  %slot = and i64 %r, 255
  %gp = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %slot
  %cur = load ptr, ptr %gp, align 8
  %empty = icmp eq ptr %cur, null
  br i1 %empty, label %do.alloc, label %do.free

do.alloc:
  %r2 = call i64 @ut_rand(ptr nonnull %seed)
  %szr = and i64 %r2, 4095
  %sz = add i64 %szr, 1
  %lb = call i64 @universe_alloc_tlsf_live(ptr %h)
  %q = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 %sz)
  %qnull = icmp eq ptr %q, null
  br i1 %qnull, label %loop.cont.a0, label %alloc.ok

alloc.ok:
  %la = call i64 @universe_alloc_tlsf_live(ptr %h)
  %qi = ptrtoint ptr %q to i64
  %qam = and i64 %qi, 15
  %qaok = icmp eq i64 %qam, 0
  %qabad = xor i1 %qaok, true
  %qab = zext i1 %qabad to i64
  %ov.a = add i64 %ov, %qab
  ; record slot
  store ptr %q, ptr %gp, align 8
  %szslot = getelementptr inbounds [256 x i64], ptr @g.size, i64 0, i64 %slot
  store i64 %sz, ptr %szslot, align 8
  %tagslot = getelementptr inbounds [256 x i8], ptr @g.tag, i64 0, i64 %slot
  %tagv = trunc i64 %slot to i8
  store i8 %tagv, ptr %tagslot, align 1
  %dslot = getelementptr inbounds [256 x i64], ptr @g.delta, i64 0, i64 %slot
  %delta = sub i64 %la, %lb
  store i64 %delta, ptr %dslot, align 8
  ; stamp first + last byte
  store i8 %tagv, ptr %q, align 1
  %szm1 = sub i64 %sz, 1
  %qe = getelementptr inbounds i8, ptr %q, i64 %szm1
  store i8 %tagv, ptr %qe, align 1
  br label %loop.cont.a1

loop.cont.a0:
  br label %loop.cont

loop.cont.a1:
  br label %loop.cont

do.free:
  %fsz.p = getelementptr inbounds [256 x i64], ptr @g.size, i64 0, i64 %slot
  %fsz = load i64, ptr %fsz.p, align 8
  %ftag.p = getelementptr inbounds [256 x i8], ptr @g.tag, i64 0, i64 %slot
  %ftag = load i8, ptr %ftag.p, align 1
  ; verify stamp
  %fb0 = load i8, ptr %cur, align 1
  %fszm1 = sub i64 %fsz, 1
  %fce = getelementptr inbounds i8, ptr %cur, i64 %fszm1
  %fb1 = load i8, ptr %fce, align 1
  %m0 = icmp eq i8 %fb0, %ftag
  %m1 = icmp eq i8 %fb1, %ftag
  %mok = and i1 %m0, %m1
  %mbad = xor i1 %mok, true
  %mbi = zext i1 %mbad to i64
  %ov.f = add i64 %ov, %mbi
  ; live delta check
  %fd.p = getelementptr inbounds [256 x i64], ptr @g.delta, i64 0, i64 %slot
  %fd = load i64, ptr %fd.p, align 8
  %lbf = call i64 @universe_alloc_tlsf_live(ptr %h)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %cur)
  %laf = call i64 @universe_alloc_tlsf_live(ptr %h)
  %drop = sub i64 %lbf, %laf
  %dok = icmp eq i64 %drop, %fd
  %dbad = xor i1 %dok, true
  %dbi = zext i1 %dbad to i64
  %av.f = add i64 %av, %dbi
  store ptr null, ptr %gp, align 8
  br label %loop.cont.f

loop.cont.f:
  br label %loop.cont

loop.cont:
  %ov.n = phi i64 [ %ov, %loop.cont.a0 ], [ %ov.a, %loop.cont.a1 ], [ %ov.f, %loop.cont.f ]
  %av.n = phi i64 [ %av, %loop.cont.a0 ], [ %av, %loop.cont.a1 ], [ %av.f, %loop.cont.f ]
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 200000
  br i1 %more, label %loop, label %drainlive

drainlive:                                       ; free every still-live slot
  %s = phi i64 [ 0, %loop.cont ], [ %s.n, %dl.cont ]
  %sp = getelementptr inbounds [4096 x ptr], ptr @g.ptr, i64 0, i64 %s
  %sv = load ptr, ptr %sp, align 8
  %svn = icmp ne ptr %sv, null
  br i1 %svn, label %dofree, label %dl.cont

dofree:
  call void @universe_alloc_tlsf_free(ptr %h, ptr %sv)
  store ptr null, ptr %sp, align 8
  br label %dl.cont

dl.cont:
  %s.n = add nuw nsw i64 %s, 1
  %smore = icmp ult i64 %s.n, 256
  br i1 %smore, label %drainlive, label %fin

fin:
  call void @ut_check_eq(i64 %ov.n, i64 0, ptr @m.overlap)
  call void @ut_check_eq(i64 %av.n, i64 0, ptr @m.acct)
  %live = call i64 @universe_alloc_tlsf_live(ptr %h)
  %live0 = icmp eq i64 %live, 0
  %whole = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 2000000)
  %wok = icmp ne ptr %whole, null
  %both = and i1 %live0, %wok
  call void @ut_check(i1 %both, ptr @m.churn0)
  call void @universe_alloc_tlsf_destroy(ptr %h)
  ret void
}

; --bench: warm-up + 16-sample distribution (min/p50/p95/p99 ns/op) vs malloc.
define internal void @bench() {
entry:
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 16781056)
  br label %td.rep

td.rep:
  %td.r = phi i64 [ 0, %entry ], [ %td.r.n, %td.next ]
  %td.t0 = call double @ut_now_sec()
  br label %tloop

tloop:
  %i = phi i64 [ 0, %td.rep ], [ %i.n, %tloop ]
  %p1 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 64)
  %p2 = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 512)
  store i64 %i, ptr %p1, align 8
  store i64 %i, ptr %p2, align 8
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p2)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p1)
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, 100000
  br i1 %more, label %tloop, label %td.mid

td.mid:
  %td.t1 = call double @ut_now_sec()
  %td.el = fsub double %td.t1, %td.t0
  %td.warm = icmp eq i64 %td.r, 0
  br i1 %td.warm, label %td.next, label %td.store

td.store:
  %td.idx = sub i64 %td.r, 1
  %td.sp = getelementptr inbounds [16 x double], ptr @tlsf.samp, i64 0, i64 %td.idx
  store double %td.el, ptr %td.sp, align 8
  br label %td.next

td.next:
  %td.r.n = add nuw nsw i64 %td.r, 1
  %td.rmore = icmp ult i64 %td.r.n, 17
  br i1 %td.rmore, label %td.rep, label %td.done

td.done:
  call void @universe_alloc_tlsf_destroy(ptr %h)
  call void @ut_report_dist(ptr @tlsf.samp, i64 16, i64 100000, ptr @lbl.tlsf)
  br label %ml.rep

ml.rep:
  %ml.r = phi i64 [ 0, %td.done ], [ %ml.r.n, %ml.next ]
  %ml.t0 = call double @ut_now_sec()
  br label %mloop

mloop:
  %j = phi i64 [ 0, %ml.rep ], [ %j.n, %mloop ]
  %q1 = call ptr @malloc(i64 64)
  %q2 = call ptr @malloc(i64 512)
  store volatile i64 %j, ptr %q1, align 8
  store volatile i64 %j, ptr %q2, align 8
  call void @free(ptr %q2)
  call void @free(ptr %q1)
  %j.n = add nuw nsw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 100000
  br i1 %more2, label %mloop, label %ml.mid

ml.mid:
  %ml.t1 = call double @ut_now_sec()
  %ml.el = fsub double %ml.t1, %ml.t0
  %ml.warm = icmp eq i64 %ml.r, 0
  br i1 %ml.warm, label %ml.next, label %ml.store

ml.store:
  %ml.idx = sub i64 %ml.r, 1
  %ml.sp = getelementptr inbounds [16 x double], ptr @malloc.samp, i64 0, i64 %ml.idx
  store double %ml.el, ptr %ml.sp, align 8
  br label %ml.next

ml.next:
  %ml.r.n = add nuw nsw i64 %ml.r, 1
  %ml.rmore = icmp ult i64 %ml.r.n, 17
  br i1 %ml.rmore, label %ml.rep, label %ml.done

ml.done:
  call void @ut_report_dist(ptr @malloc.samp, i64 16, i64 100000, ptr @lbl.malloc)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_caller_region()
  call void @test_edges()
  call void @test_aligned()
  call void @test_coalesce()
  call void @test_churn()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
