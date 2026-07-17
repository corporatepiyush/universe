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

; Tests for universe_alloc_hybrid: small->slab / large->TLSF routing (verified
; via tlsf_live + summed slab_live deltas read through the documented handle
; offsets), a boundary sweep, a shadow-oracle random mixed churn (no overlap,
; correct reuse per path), a CUSTOM tunable config, alloc_aligned on both
; paths, leak-free destroy, and a --bench (small + large, hybrid vs TLSF vs
; malloc).

declare ptr  @universe_alloc_hybrid_create(ptr)
declare ptr  @universe_alloc_hybrid_alloc(ptr, i64)
declare ptr  @universe_alloc_hybrid_alloc_aligned(ptr, i64, i64)
declare void @universe_alloc_hybrid_free(ptr, ptr)
declare i64  @universe_alloc_hybrid_live(ptr)
declare void @universe_alloc_hybrid_destroy(ptr)

declare ptr  @universe_alloc_tlsf_create(ptr, i64)
declare ptr  @universe_alloc_tlsf_alloc(ptr, i64)
declare void @universe_alloc_tlsf_free(ptr, ptr)
declare i64  @universe_alloc_tlsf_live(ptr)
declare void @universe_alloc_tlsf_destroy(ptr)

declare i64  @universe_alloc_slab_live(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr captures(none))
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare void @ut_report_dist(ptr, i64, i64, ptr)

; ---- shadow-oracle state ---------------------------------------------------
@o.ptr = internal global [2048 x ptr] zeroinitializer, align 16
@o.sz  = internal global [2048 x i64] zeroinitializer, align 16
@o.tag = internal global [2048 x i8]  zeroinitializer, align 16

; ---- custom config ---------------------------------------------------------
@cfg.classes = internal global [3 x i64] [i64 32, i64 128, i64 256], align 8
@cfg.mem     = internal global [40 x i8] zeroinitializer, align 8

; ---- bench: steady-state live working set (ring of W live allocations) ------
@ring = internal global [256 x ptr] zeroinitializer, align 16

; ---- bench sample arrays ---------------------------------------------------
@hs.samp = internal global [16 x double] zeroinitializer, align 8
@ts.samp = internal global [16 x double] zeroinitializer, align 8
@ms.samp = internal global [16 x double] zeroinitializer, align 8
@hl.samp = internal global [16 x double] zeroinitializer, align 8
@tl.samp = internal global [16 x double] zeroinitializer, align 8
@ml.samp = internal global [16 x double] zeroinitializer, align 8

@lbl.hs = private unnamed_addr constant [23 x i8] c"hybrid small 64B churn\00"
@lbl.ts = private unnamed_addr constant [23 x i8] c"tlsf   small 64B churn\00"
@lbl.ms = private unnamed_addr constant [23 x i8] c"malloc small 64B churn\00"
@lbl.hl = private unnamed_addr constant [25 x i8] c"hybrid large 4096B churn\00"
@lbl.tl = private unnamed_addr constant [25 x i8] c"tlsf   large 4096B churn\00"
@lbl.ml = private unnamed_addr constant [25 x i8] c"malloc large 4096B churn\00"

; ---- messages --------------------------------------------------------------
@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.small.t  = private unnamed_addr constant [23 x i8] c"small: tlsf unchanged \00"
@m.small.s  = private unnamed_addr constant [19 x i8] c"small: slab live+1\00"
@m.live1    = private unnamed_addr constant [15 x i8] c"live 1 (small)\00"
@m.large.t  = private unnamed_addr constant [19 x i8] c"large: tlsf grew  \00"
@m.large.s  = private unnamed_addr constant [22 x i8] c"large: slab unchanged\00"
@m.live2    = private unnamed_addr constant [15 x i8] c"live 2 (large)\00"
@m.b512.s   = private unnamed_addr constant [18 x i8] c"512 -> slab (bnd)\00"
@m.b513.t   = private unnamed_addr constant [18 x i8] c"513 -> tlsf (bnd)\00"
@m.live0    = private unnamed_addr constant [20 x i8] c"live 0 after frees \00"

@m.orc.null = private unnamed_addr constant [21 x i8] c"oracle: no null allc\00"
@m.orc.corr = private unnamed_addr constant [24 x i8] c"oracle: no overlap/corr\00"
@m.orc.live = private unnamed_addr constant [22 x i8] c"oracle: live 0 at end\00"

@m.cust.cr  = private unnamed_addr constant [20 x i8] c"custom cfg created \00"
@m.cust.sm  = private unnamed_addr constant [22 x i8] c"custom small_max==256\00"
@m.cust.us  = private unnamed_addr constant [25 x i8] c"custom 256 under -> slab\00"
@m.cust.ov  = private unnamed_addr constant [25 x i8] c"custom 257 over  -> tlsf\00"
@m.cust.mid = private unnamed_addr constant [22 x i8] c"custom 100 -> slab   \00"

@m.al16.nn  = private unnamed_addr constant [22 x i8] c"aligned 16 nonnull   \00"
@m.al16.al  = private unnamed_addr constant [22 x i8] c"aligned 16 is aligned\00"
@m.al16.s   = private unnamed_addr constant [22 x i8] c"aligned 16 -> slab   \00"
@m.al64.nn  = private unnamed_addr constant [22 x i8] c"aligned 64 nonnull   \00"
@m.al64.al  = private unnamed_addr constant [22 x i8] c"aligned 64 is aligned\00"
@m.al64.t   = private unnamed_addr constant [22 x i8] c"aligned 64 -> tlsf   \00"
@m.alL.al   = private unnamed_addr constant [24 x i8] c"aligned 64 large align \00"

@m.null.a   = private unnamed_addr constant [22 x i8] c"null handle alloc nul\00"
@m.null.l   = private unnamed_addr constant [20 x i8] c"null handle live 0 \00"

; ===========================================================================
; helper: sum of live objects across all class slabs (reads sub-handles at the
; documented handle offsets: slab_handles[] at +64 + nclasses*8).
define internal i64 @sum_slab_live(ptr %h) {
entry:
  %ncp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nc = load i64, ptr %ncp, align 8
  %csbytes = mul nuw i64 %nc, 8
  %shoff = add nuw i64 64, %csbytes
  %shbase = getelementptr inbounds nuw i8, ptr %h, i64 %shoff
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %body ]
  %acc = phi i64 [ 0, %entry ], [ %accn, %body ]
  %more = icmp ult i64 %i, %nc
  br i1 %more, label %body, label %done

body:
  %slot = getelementptr inbounds ptr, ptr %shbase, i64 %i
  %slab = load ptr, ptr %slot, align 8
  %sl = call i64 @universe_alloc_slab_live(ptr %slab)
  %accn = add i64 %acc, %sl
  %in = add i64 %i, 1
  br label %loop

done:
  ret i64 %acc
}

; ===========================================================================
define internal void @test_routing() {
entry:
  %h = call ptr @universe_alloc_hybrid_create(ptr null)
  %ok = icmp ne ptr %h, null
  call void @ut_check(i1 %ok, ptr @m.create)
  br i1 %ok, label %go, label %done

go:
  %tlsf = load ptr, ptr %h, align 8
  ; --- small alloc (64B) ---
  %t0 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %s0 = call i64 @sum_slab_live(ptr %h)
  %p1 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 64)
  %t1 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %s1 = call i64 @sum_slab_live(ptr %h)
  call void @ut_check_eq(i64 %t1, i64 %t0, ptr @m.small.t)
  %s0p1 = add i64 %s0, 1
  call void @ut_check_eq(i64 %s1, i64 %s0p1, ptr @m.small.s)
  %lv1 = call i64 @universe_alloc_hybrid_live(ptr %h)
  call void @ut_check_eq(i64 %lv1, i64 1, ptr @m.live1)

  ; --- large alloc (1000B) ---
  %p2 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 1000)
  %t2 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %s2 = call i64 @sum_slab_live(ptr %h)
  %tgrew = icmp ugt i64 %t2, %t1
  call void @ut_check(i1 %tgrew, ptr @m.large.t)
  call void @ut_check_eq(i64 %s2, i64 %s1, ptr @m.large.s)
  %lv2 = call i64 @universe_alloc_hybrid_live(ptr %h)
  call void @ut_check_eq(i64 %lv2, i64 2, ptr @m.live2)

  ; --- boundary sweep: 512 -> slab, 513 -> tlsf ---
  %s2b = call i64 @sum_slab_live(ptr %h)
  %p3 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 512)
  %s3 = call i64 @sum_slab_live(ptr %h)
  %s2bp1 = add i64 %s2b, 1
  call void @ut_check_eq(i64 %s3, i64 %s2bp1, ptr @m.b512.s)

  %t3 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %p4 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 513)
  %t4 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %tgrew2 = icmp ugt i64 %t4, %t3
  call void @ut_check(i1 %tgrew2, ptr @m.b513.t)

  ; --- free all, expect live 0 ---
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p1)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p2)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p3)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p4)
  %lv0 = call i64 @universe_alloc_hybrid_live(ptr %h)
  call void @ut_check_eq(i64 %lv0, i64 0, ptr @m.live0)
  call void @universe_alloc_hybrid_destroy(ptr %h)
  br label %done

done:
  ret void
}

; ===========================================================================
; Shadow-oracle random mixed churn. Each live allocation's payload is filled
; with a per-alloc tag byte; on free the head+tail bytes are verified to still
; equal the tag — any overlap between two live allocations (either path) would
; clobber one and be caught. count is swap-remove maintained.
define internal void @test_oracle() {
entry:
  %st = alloca i64, align 8
  store i64 305419896, ptr %st, align 8        ; 0x12345678
  %h = call ptr @universe_alloc_hybrid_create(ptr null)
  %ok = icmp ne ptr %h, null
  br i1 %ok, label %loop.head, label %skip

loop.head:
  %iter = phi i64 [ 0, %entry ], [ %iter.n, %loop.tail ]
  %count = phi i64 [ 0, %entry ], [ %count.next, %loop.tail ]
  %corrupt = phi i64 [ 0, %entry ], [ %corrupt.next, %loop.tail ]
  %nullc = phi i64 [ 0, %entry ], [ %nullc.next, %loop.tail ]
  %r = call i64 @ut_rand(ptr %st)
  %empty = icmp eq i64 %count, 0
  %notfull = icmp ult i64 %count, 2048
  %rbit = and i64 %r, 1
  %wantalloc = icmp ne i64 %rbit, 0
  %wa = and i1 %wantalloc, %notfull
  %doalloc = or i1 %empty, %wa
  br i1 %doalloc, label %alloc, label %free

alloc:
  %r2 = call i64 @ut_rand(ptr %st)
  %m900 = urem i64 %r2, 900
  %size = add i64 %m900, 1
  %p = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 %size)
  %r3 = call i64 @ut_rand(ptr %st)
  %tag = trunc i64 %r3 to i8
  %pnull = icmp eq ptr %p, null
  %ninc = zext i1 %pnull to i64
  %nullc.a = add i64 %nullc, %ninc
  br i1 %pnull, label %alloc.skip, label %alloc.store

alloc.store:
  call void @llvm.memset.p0.i64(ptr %p, i8 %tag, i64 %size, i1 false)
  %pslot = getelementptr inbounds [2048 x ptr], ptr @o.ptr, i64 0, i64 %count
  store ptr %p, ptr %pslot, align 8
  %zslot = getelementptr inbounds [2048 x i64], ptr @o.sz, i64 0, i64 %count
  store i64 %size, ptr %zslot, align 8
  %tslot = getelementptr inbounds [2048 x i8], ptr @o.tag, i64 0, i64 %count
  store i8 %tag, ptr %tslot, align 1
  %count.a = add i64 %count, 1
  br label %loop.tail

alloc.skip:
  br label %loop.tail

free:
  %r4 = call i64 @ut_rand(ptr %st)
  %j = urem i64 %r4, %count
  %jpslot = getelementptr inbounds [2048 x ptr], ptr @o.ptr, i64 0, i64 %j
  %jp = load ptr, ptr %jpslot, align 8
  %jzslot = getelementptr inbounds [2048 x i64], ptr @o.sz, i64 0, i64 %j
  %jsz = load i64, ptr %jzslot, align 8
  %jtslot = getelementptr inbounds [2048 x i8], ptr @o.tag, i64 0, i64 %j
  %jtag = load i8, ptr %jtslot, align 1
  %b0 = load i8, ptr %jp, align 1
  %szm1 = sub i64 %jsz, 1
  %pend = getelementptr inbounds nuw i8, ptr %jp, i64 %szm1
  %be = load i8, ptr %pend, align 1
  %ok0 = icmp eq i8 %b0, %jtag
  %oke = icmp eq i8 %be, %jtag
  %okboth = and i1 %ok0, %oke
  %bad = xor i1 %okboth, true
  %binc = zext i1 %bad to i64
  %corrupt.f = add i64 %corrupt, %binc
  call void @universe_alloc_hybrid_free(ptr %h, ptr %jp)
  ; swap-remove last into j
  %last = sub i64 %count, 1
  %lpslot = getelementptr inbounds [2048 x ptr], ptr @o.ptr, i64 0, i64 %last
  %lp = load ptr, ptr %lpslot, align 8
  store ptr %lp, ptr %jpslot, align 8
  %lzslot = getelementptr inbounds [2048 x i64], ptr @o.sz, i64 0, i64 %last
  %lz = load i64, ptr %lzslot, align 8
  store i64 %lz, ptr %jzslot, align 8
  %ltslot = getelementptr inbounds [2048 x i8], ptr @o.tag, i64 0, i64 %last
  %lt = load i8, ptr %ltslot, align 1
  store i8 %lt, ptr %jtslot, align 1
  br label %loop.tail

loop.tail:
  %count.next = phi i64 [ %count.a, %alloc.store ], [ %count, %alloc.skip ], [ %last, %free ]
  %corrupt.next = phi i64 [ %corrupt, %alloc.store ], [ %corrupt, %alloc.skip ], [ %corrupt.f, %free ]
  %nullc.next = phi i64 [ %nullc.a, %alloc.store ], [ %nullc.a, %alloc.skip ], [ %nullc, %free ]
  %iter.n = add i64 %iter, 1
  %more = icmp ult i64 %iter.n, 20000
  br i1 %more, label %loop.head, label %drain.head

drain.head:
  %di = phi i64 [ 0, %loop.tail ], [ %di.n, %drain.body ]
  %dcorrupt = phi i64 [ %corrupt.next, %loop.tail ], [ %dcorrupt.n, %drain.body ]
  %dmore = icmp ult i64 %di, %count.next
  br i1 %dmore, label %drain.body, label %drain.done

drain.body:
  %dpslot = getelementptr inbounds [2048 x ptr], ptr @o.ptr, i64 0, i64 %di
  %dp = load ptr, ptr %dpslot, align 8
  %dzslot = getelementptr inbounds [2048 x i64], ptr @o.sz, i64 0, i64 %di
  %dsz = load i64, ptr %dzslot, align 8
  %dtslot = getelementptr inbounds [2048 x i8], ptr @o.tag, i64 0, i64 %di
  %dtag = load i8, ptr %dtslot, align 1
  %db0 = load i8, ptr %dp, align 1
  %dszm1 = sub i64 %dsz, 1
  %dpend = getelementptr inbounds nuw i8, ptr %dp, i64 %dszm1
  %dbe = load i8, ptr %dpend, align 1
  %dok0 = icmp eq i8 %db0, %dtag
  %doke = icmp eq i8 %dbe, %dtag
  %dokboth = and i1 %dok0, %doke
  %dbad = xor i1 %dokboth, true
  %dbinc = zext i1 %dbad to i64
  %dcorrupt.n = add i64 %dcorrupt, %dbinc
  call void @universe_alloc_hybrid_free(ptr %h, ptr %dp)
  %di.n = add i64 %di, 1
  br label %drain.head

drain.done:
  call void @ut_check_eq(i64 %nullc.next, i64 0, ptr @m.orc.null)
  call void @ut_check_eq(i64 %dcorrupt, i64 0, ptr @m.orc.corr)
  %lv = call i64 @universe_alloc_hybrid_live(ptr %h)
  call void @ut_check_eq(i64 %lv, i64 0, ptr @m.orc.live)
  call void @universe_alloc_hybrid_destroy(ptr %h)
  br label %skip

skip:
  ret void
}

; ===========================================================================
; Custom tunable config: explicit class_sizes [32,128,256], slab_chunk 4096.
; small_max snaps to top class 256; 256 -> slab, 257 -> tlsf, 100 -> slab.
define internal void @test_custom() {
entry:
  ; build cfg
  %sm.p = getelementptr inbounds nuw i8, ptr @cfg.mem, i64 0
  store i64 999, ptr %sm.p, align 8              ; overridden by top class
  %nc.p = getelementptr inbounds nuw i8, ptr @cfg.mem, i64 8
  store i64 3, ptr %nc.p, align 8
  %cs.p = getelementptr inbounds nuw i8, ptr @cfg.mem, i64 16
  store ptr @cfg.classes, ptr %cs.p, align 8
  %ch.p = getelementptr inbounds nuw i8, ptr @cfg.mem, i64 24
  store i64 4096, ptr %ch.p, align 8
  %ti.p = getelementptr inbounds nuw i8, ptr @cfg.mem, i64 32
  store i64 1048576, ptr %ti.p, align 8

  %h = call ptr @universe_alloc_hybrid_create(ptr @cfg.mem)
  %ok = icmp ne ptr %h, null
  call void @ut_check(i1 %ok, ptr @m.cust.cr)
  br i1 %ok, label %go, label %done

go:
  %smhp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %smh = load i64, ptr %smhp, align 8
  call void @ut_check_eq(i64 %smh, i64 256, ptr @m.cust.sm)
  %tlsf = load ptr, ptr %h, align 8

  ; size 256 (== small_max) -> slab
  %sa = call i64 @sum_slab_live(ptr %h)
  %p256 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 256)
  %sb = call i64 @sum_slab_live(ptr %h)
  %sap1 = add i64 %sa, 1
  call void @ut_check_eq(i64 %sb, i64 %sap1, ptr @m.cust.us)

  ; size 257 (> small_max) -> tlsf
  %ta = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %p257 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 257)
  %tb = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %tgrew = icmp ugt i64 %tb, %ta
  call void @ut_check(i1 %tgrew, ptr @m.cust.ov)

  ; size 100 -> slab (class 128)
  %sc = call i64 @sum_slab_live(ptr %h)
  %p100 = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 100)
  %sd = call i64 @sum_slab_live(ptr %h)
  %scp1 = add i64 %sc, 1
  call void @ut_check_eq(i64 %sd, i64 %scp1, ptr @m.cust.mid)

  call void @universe_alloc_hybrid_free(ptr %h, ptr %p256)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p257)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %p100)
  call void @universe_alloc_hybrid_destroy(ptr %h)
  br label %done

done:
  ret void
}

; ===========================================================================
; alloc_aligned on both paths.
define internal void @test_aligned() {
entry:
  %h = call ptr @universe_alloc_hybrid_create(ptr null)
  %ok = icmp ne ptr %h, null
  br i1 %ok, label %go, label %done

go:
  %tlsf = load ptr, ptr %h, align 8

  ; align 16, size 100 -> small -> slab path (payload already 16-aligned)
  %t0 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %s0 = call i64 @sum_slab_live(ptr %h)
  %pa = call ptr @universe_alloc_hybrid_alloc_aligned(ptr %h, i64 100, i64 16)
  %pa.ok = icmp ne ptr %pa, null
  call void @ut_check(i1 %pa.ok, ptr @m.al16.nn)
  %pa.i = ptrtoint ptr %pa to i64
  %pa.lo = and i64 %pa.i, 15
  %pa.al = icmp eq i64 %pa.lo, 0
  call void @ut_check(i1 %pa.al, ptr @m.al16.al)
  %s1 = call i64 @sum_slab_live(ptr %h)
  %s0p1 = add i64 %s0, 1
  call void @ut_check_eq(i64 %s1, i64 %s0p1, ptr @m.al16.s)

  ; align 64, size 100 -> big path -> tlsf aligned
  %t1 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %pb = call ptr @universe_alloc_hybrid_alloc_aligned(ptr %h, i64 100, i64 64)
  %pb.ok = icmp ne ptr %pb, null
  call void @ut_check(i1 %pb.ok, ptr @m.al64.nn)
  %pb.i = ptrtoint ptr %pb to i64
  %pb.lo = and i64 %pb.i, 63
  %pb.al = icmp eq i64 %pb.lo, 0
  call void @ut_check(i1 %pb.al, ptr @m.al64.al)
  %t2 = call i64 @universe_alloc_tlsf_live(ptr %tlsf)
  %tgrew = icmp ugt i64 %t2, %t1
  call void @ut_check(i1 %tgrew, ptr @m.al64.t)

  ; align 64, size 2000 -> big path large
  %pc = call ptr @universe_alloc_hybrid_alloc_aligned(ptr %h, i64 2000, i64 64)
  %pc.i = ptrtoint ptr %pc to i64
  %pc.lo = and i64 %pc.i, 63
  %pc.al = icmp eq i64 %pc.lo, 0
  call void @ut_check(i1 %pc.al, ptr @m.alL.al)

  ; write through each to prove usable, then free
  store i64 1, ptr %pa, align 8
  store i64 2, ptr %pb, align 8
  store i64 3, ptr %pc, align 8
  call void @universe_alloc_hybrid_free(ptr %h, ptr %pa)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %pb)
  call void @universe_alloc_hybrid_free(ptr %h, ptr %pc)
  call void @universe_alloc_hybrid_destroy(ptr %h)
  br label %done

done:
  ret void
}

; ===========================================================================
define internal void @test_null() {
entry:
  %p = call ptr @universe_alloc_hybrid_alloc(ptr null, i64 32)
  %pn = icmp eq ptr %p, null
  call void @ut_check(i1 %pn, ptr @m.null.a)
  %l = call i64 @universe_alloc_hybrid_live(ptr null)
  call void @ut_check_eq(i64 %l, i64 0, ptr @m.null.l)
  call void @universe_alloc_hybrid_free(ptr null, ptr null)
  call void @universe_alloc_hybrid_destroy(ptr null)
  ret void
}

; ===========================================================================
; --bench: STEADY-STATE churn with a live working set. A ring of W=256 live
; allocations is pre-filled (untimed); each timed op frees one slot and allocs
; a replacement into it, so W objects stay live throughout — the slab never
; goes empty (that is the small-object common case the hybrid targets). ops =
; 100k free+alloc per rep, 17 reps (rep 0 warm-up), ns/op distribution.
; (NB: a strict alloc/free-PAIR pattern empties the slab every iteration and
;  the slab eagerly releases empty slabs to the OS — that pathological pattern
;  makes the slab path lose; steady-state churn is the representative measure.)
define internal void @bench() {
entry:
  ; ================= SMALL 64B =================
  ; ---- hybrid ----
  %hs = call ptr @universe_alloc_hybrid_create(ptr null)
  br label %hs.fill
hs.fill:
  %hs.fi = phi i64 [ 0, %entry ], [ %hs.fin, %hs.fill ]
  %hs.fp = call ptr @universe_alloc_hybrid_alloc(ptr %hs, i64 64)
  %hs.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hs.fi
  store ptr %hs.fp, ptr %hs.fs, align 8
  %hs.fin = add nuw nsw i64 %hs.fi, 1
  %hs.fmore = icmp ult i64 %hs.fin, 256
  br i1 %hs.fmore, label %hs.fill, label %hs.rep
hs.rep:
  %hs.r = phi i64 [ 0, %hs.fill ], [ %hs.rn, %hs.next ]
  %hs.t0 = call double @ut_now_sec()
  br label %hs.loop
hs.loop:
  %hs.i = phi i64 [ 0, %hs.rep ], [ %hs.in, %hs.loop ]
  %hs.idx = and i64 %hs.i, 255
  %hs.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hs.idx
  %hs.old = load ptr, ptr %hs.slot, align 8
  call void @universe_alloc_hybrid_free(ptr %hs, ptr %hs.old)
  %hs.new = call ptr @universe_alloc_hybrid_alloc(ptr %hs, i64 64)
  store volatile i64 %hs.i, ptr %hs.new, align 8
  store ptr %hs.new, ptr %hs.slot, align 8
  %hs.in = add nuw nsw i64 %hs.i, 1
  %hs.more = icmp ult i64 %hs.in, 100000
  br i1 %hs.more, label %hs.loop, label %hs.mid
hs.mid:
  %hs.t1 = call double @ut_now_sec()
  %hs.el = fsub double %hs.t1, %hs.t0
  %hs.warm = icmp eq i64 %hs.r, 0
  br i1 %hs.warm, label %hs.next, label %hs.store
hs.store:
  %hs.si = sub i64 %hs.r, 1
  %hs.sp = getelementptr inbounds [16 x double], ptr @hs.samp, i64 0, i64 %hs.si
  store double %hs.el, ptr %hs.sp, align 8
  br label %hs.next
hs.next:
  %hs.rn = add nuw nsw i64 %hs.r, 1
  %hs.rmore = icmp ult i64 %hs.rn, 17
  br i1 %hs.rmore, label %hs.rep, label %hs.drain
hs.drain:
  %hs.di = phi i64 [ 0, %hs.next ], [ %hs.din, %hs.drain ]
  %hs.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hs.di
  %hs.dp = load ptr, ptr %hs.ds, align 8
  call void @universe_alloc_hybrid_free(ptr %hs, ptr %hs.dp)
  %hs.din = add nuw nsw i64 %hs.di, 1
  %hs.dmore = icmp ult i64 %hs.din, 256
  br i1 %hs.dmore, label %hs.drain, label %hs.done
hs.done:
  call void @universe_alloc_hybrid_destroy(ptr %hs)
  call void @ut_report_dist(ptr @hs.samp, i64 16, i64 100000, ptr @lbl.hs)

  ; ---- tlsf ----
  %ts = call ptr @universe_alloc_tlsf_create(ptr null, i64 1048576)
  br label %ts.fill
ts.fill:
  %ts.fi = phi i64 [ 0, %hs.done ], [ %ts.fin, %ts.fill ]
  %ts.fp = call ptr @universe_alloc_tlsf_alloc(ptr %ts, i64 64)
  %ts.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ts.fi
  store ptr %ts.fp, ptr %ts.fs, align 8
  %ts.fin = add nuw nsw i64 %ts.fi, 1
  %ts.fmore = icmp ult i64 %ts.fin, 256
  br i1 %ts.fmore, label %ts.fill, label %ts.rep
ts.rep:
  %ts.r = phi i64 [ 0, %ts.fill ], [ %ts.rn, %ts.next ]
  %ts.t0 = call double @ut_now_sec()
  br label %ts.loop
ts.loop:
  %ts.i = phi i64 [ 0, %ts.rep ], [ %ts.in, %ts.loop ]
  %ts.idx = and i64 %ts.i, 255
  %ts.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ts.idx
  %ts.old = load ptr, ptr %ts.slot, align 8
  call void @universe_alloc_tlsf_free(ptr %ts, ptr %ts.old)
  %ts.new = call ptr @universe_alloc_tlsf_alloc(ptr %ts, i64 64)
  store volatile i64 %ts.i, ptr %ts.new, align 8
  store ptr %ts.new, ptr %ts.slot, align 8
  %ts.in = add nuw nsw i64 %ts.i, 1
  %ts.more = icmp ult i64 %ts.in, 100000
  br i1 %ts.more, label %ts.loop, label %ts.mid
ts.mid:
  %ts.t1 = call double @ut_now_sec()
  %ts.el = fsub double %ts.t1, %ts.t0
  %ts.warm = icmp eq i64 %ts.r, 0
  br i1 %ts.warm, label %ts.next, label %ts.store
ts.store:
  %ts.si = sub i64 %ts.r, 1
  %ts.sp = getelementptr inbounds [16 x double], ptr @ts.samp, i64 0, i64 %ts.si
  store double %ts.el, ptr %ts.sp, align 8
  br label %ts.next
ts.next:
  %ts.rn = add nuw nsw i64 %ts.r, 1
  %ts.rmore = icmp ult i64 %ts.rn, 17
  br i1 %ts.rmore, label %ts.rep, label %ts.drain
ts.drain:
  %ts.di = phi i64 [ 0, %ts.next ], [ %ts.din, %ts.drain ]
  %ts.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ts.di
  %ts.dp = load ptr, ptr %ts.ds, align 8
  call void @universe_alloc_tlsf_free(ptr %ts, ptr %ts.dp)
  %ts.din = add nuw nsw i64 %ts.di, 1
  %ts.dmore = icmp ult i64 %ts.din, 256
  br i1 %ts.dmore, label %ts.drain, label %ts.done
ts.done:
  call void @universe_alloc_tlsf_destroy(ptr %ts)
  call void @ut_report_dist(ptr @ts.samp, i64 16, i64 100000, ptr @lbl.ts)

  ; ---- malloc ----
  br label %ms.fill
ms.fill:
  %ms.fi = phi i64 [ 0, %ts.done ], [ %ms.fin, %ms.fill ]
  %ms.fp = call ptr @malloc(i64 64)
  %ms.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ms.fi
  store ptr %ms.fp, ptr %ms.fs, align 8
  %ms.fin = add nuw nsw i64 %ms.fi, 1
  %ms.fmore = icmp ult i64 %ms.fin, 256
  br i1 %ms.fmore, label %ms.fill, label %ms.rep
ms.rep:
  %ms.r = phi i64 [ 0, %ms.fill ], [ %ms.rn, %ms.next ]
  %ms.t0 = call double @ut_now_sec()
  br label %ms.loop
ms.loop:
  %ms.i = phi i64 [ 0, %ms.rep ], [ %ms.in, %ms.loop ]
  %ms.idx = and i64 %ms.i, 255
  %ms.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ms.idx
  %ms.old = load ptr, ptr %ms.slot, align 8
  call void @free(ptr %ms.old)
  %ms.new = call ptr @malloc(i64 64)
  store volatile i64 %ms.i, ptr %ms.new, align 8
  store ptr %ms.new, ptr %ms.slot, align 8
  %ms.in = add nuw nsw i64 %ms.i, 1
  %ms.more = icmp ult i64 %ms.in, 100000
  br i1 %ms.more, label %ms.loop, label %ms.mid
ms.mid:
  %ms.t1 = call double @ut_now_sec()
  %ms.el = fsub double %ms.t1, %ms.t0
  %ms.warm = icmp eq i64 %ms.r, 0
  br i1 %ms.warm, label %ms.next, label %ms.store
ms.store:
  %ms.si = sub i64 %ms.r, 1
  %ms.sp = getelementptr inbounds [16 x double], ptr @ms.samp, i64 0, i64 %ms.si
  store double %ms.el, ptr %ms.sp, align 8
  br label %ms.next
ms.next:
  %ms.rn = add nuw nsw i64 %ms.r, 1
  %ms.rmore = icmp ult i64 %ms.rn, 17
  br i1 %ms.rmore, label %ms.rep, label %ms.drain
ms.drain:
  %ms.di = phi i64 [ 0, %ms.next ], [ %ms.din, %ms.drain ]
  %ms.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %ms.di
  %ms.dp = load ptr, ptr %ms.ds, align 8
  call void @free(ptr %ms.dp)
  %ms.din = add nuw nsw i64 %ms.di, 1
  %ms.dmore = icmp ult i64 %ms.din, 256
  br i1 %ms.dmore, label %ms.drain, label %ms.done
ms.done:
  call void @ut_report_dist(ptr @ms.samp, i64 16, i64 100000, ptr @lbl.ms)

  ; ================= LARGE 4096B =================
  ; ---- hybrid ----
  %hl = call ptr @universe_alloc_hybrid_create(ptr null)
  br label %hl.fill
hl.fill:
  %hl.fi = phi i64 [ 0, %ms.done ], [ %hl.fin, %hl.fill ]
  %hl.fp = call ptr @universe_alloc_hybrid_alloc(ptr %hl, i64 4096)
  %hl.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hl.fi
  store ptr %hl.fp, ptr %hl.fs, align 8
  %hl.fin = add nuw nsw i64 %hl.fi, 1
  %hl.fmore = icmp ult i64 %hl.fin, 256
  br i1 %hl.fmore, label %hl.fill, label %hl.rep
hl.rep:
  %hl.r = phi i64 [ 0, %hl.fill ], [ %hl.rn, %hl.next ]
  %hl.t0 = call double @ut_now_sec()
  br label %hl.loop
hl.loop:
  %hl.i = phi i64 [ 0, %hl.rep ], [ %hl.in, %hl.loop ]
  %hl.idx = and i64 %hl.i, 255
  %hl.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hl.idx
  %hl.old = load ptr, ptr %hl.slot, align 8
  call void @universe_alloc_hybrid_free(ptr %hl, ptr %hl.old)
  %hl.new = call ptr @universe_alloc_hybrid_alloc(ptr %hl, i64 4096)
  store volatile i64 %hl.i, ptr %hl.new, align 8
  store ptr %hl.new, ptr %hl.slot, align 8
  %hl.in = add nuw nsw i64 %hl.i, 1
  %hl.more = icmp ult i64 %hl.in, 100000
  br i1 %hl.more, label %hl.loop, label %hl.mid
hl.mid:
  %hl.t1 = call double @ut_now_sec()
  %hl.el = fsub double %hl.t1, %hl.t0
  %hl.warm = icmp eq i64 %hl.r, 0
  br i1 %hl.warm, label %hl.next, label %hl.store
hl.store:
  %hl.si = sub i64 %hl.r, 1
  %hl.sp = getelementptr inbounds [16 x double], ptr @hl.samp, i64 0, i64 %hl.si
  store double %hl.el, ptr %hl.sp, align 8
  br label %hl.next
hl.next:
  %hl.rn = add nuw nsw i64 %hl.r, 1
  %hl.rmore = icmp ult i64 %hl.rn, 17
  br i1 %hl.rmore, label %hl.rep, label %hl.drain
hl.drain:
  %hl.di = phi i64 [ 0, %hl.next ], [ %hl.din, %hl.drain ]
  %hl.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %hl.di
  %hl.dp = load ptr, ptr %hl.ds, align 8
  call void @universe_alloc_hybrid_free(ptr %hl, ptr %hl.dp)
  %hl.din = add nuw nsw i64 %hl.di, 1
  %hl.dmore = icmp ult i64 %hl.din, 256
  br i1 %hl.dmore, label %hl.drain, label %hl.done
hl.done:
  call void @universe_alloc_hybrid_destroy(ptr %hl)
  call void @ut_report_dist(ptr @hl.samp, i64 16, i64 100000, ptr @lbl.hl)

  ; ---- tlsf ----
  %tl = call ptr @universe_alloc_tlsf_create(ptr null, i64 1048576)
  br label %tl.fill
tl.fill:
  %tl.fi = phi i64 [ 0, %hl.done ], [ %tl.fin, %tl.fill ]
  %tl.fp = call ptr @universe_alloc_tlsf_alloc(ptr %tl, i64 4096)
  %tl.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %tl.fi
  store ptr %tl.fp, ptr %tl.fs, align 8
  %tl.fin = add nuw nsw i64 %tl.fi, 1
  %tl.fmore = icmp ult i64 %tl.fin, 256
  br i1 %tl.fmore, label %tl.fill, label %tl.rep
tl.rep:
  %tl.r = phi i64 [ 0, %tl.fill ], [ %tl.rn, %tl.next ]
  %tl.t0 = call double @ut_now_sec()
  br label %tl.loop
tl.loop:
  %tl.i = phi i64 [ 0, %tl.rep ], [ %tl.in, %tl.loop ]
  %tl.idx = and i64 %tl.i, 255
  %tl.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %tl.idx
  %tl.old = load ptr, ptr %tl.slot, align 8
  call void @universe_alloc_tlsf_free(ptr %tl, ptr %tl.old)
  %tl.new = call ptr @universe_alloc_tlsf_alloc(ptr %tl, i64 4096)
  store volatile i64 %tl.i, ptr %tl.new, align 8
  store ptr %tl.new, ptr %tl.slot, align 8
  %tl.in = add nuw nsw i64 %tl.i, 1
  %tl.more = icmp ult i64 %tl.in, 100000
  br i1 %tl.more, label %tl.loop, label %tl.mid
tl.mid:
  %tl.t1 = call double @ut_now_sec()
  %tl.el = fsub double %tl.t1, %tl.t0
  %tl.warm = icmp eq i64 %tl.r, 0
  br i1 %tl.warm, label %tl.next, label %tl.store
tl.store:
  %tl.si = sub i64 %tl.r, 1
  %tl.sp = getelementptr inbounds [16 x double], ptr @tl.samp, i64 0, i64 %tl.si
  store double %tl.el, ptr %tl.sp, align 8
  br label %tl.next
tl.next:
  %tl.rn = add nuw nsw i64 %tl.r, 1
  %tl.rmore = icmp ult i64 %tl.rn, 17
  br i1 %tl.rmore, label %tl.rep, label %tl.drain
tl.drain:
  %tl.di = phi i64 [ 0, %tl.next ], [ %tl.din, %tl.drain ]
  %tl.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %tl.di
  %tl.dp = load ptr, ptr %tl.ds, align 8
  call void @universe_alloc_tlsf_free(ptr %tl, ptr %tl.dp)
  %tl.din = add nuw nsw i64 %tl.di, 1
  %tl.dmore = icmp ult i64 %tl.din, 256
  br i1 %tl.dmore, label %tl.drain, label %tl.done
tl.done:
  call void @universe_alloc_tlsf_destroy(ptr %tl)
  call void @ut_report_dist(ptr @tl.samp, i64 16, i64 100000, ptr @lbl.tl)

  ; ---- malloc ----
  br label %mll.fill
mll.fill:
  %mll.fi = phi i64 [ 0, %tl.done ], [ %mll.fin, %mll.fill ]
  %mll.fp = call ptr @malloc(i64 4096)
  %mll.fs = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %mll.fi
  store ptr %mll.fp, ptr %mll.fs, align 8
  %mll.fin = add nuw nsw i64 %mll.fi, 1
  %mll.fmore = icmp ult i64 %mll.fin, 256
  br i1 %mll.fmore, label %mll.fill, label %mll.rep
mll.rep:
  %mll.r = phi i64 [ 0, %mll.fill ], [ %mll.rn, %mll.next ]
  %mll.t0 = call double @ut_now_sec()
  br label %mll.loop
mll.loop:
  %mll.i = phi i64 [ 0, %mll.rep ], [ %mll.in, %mll.loop ]
  %mll.idx = and i64 %mll.i, 255
  %mll.slot = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %mll.idx
  %mll.old = load ptr, ptr %mll.slot, align 8
  call void @free(ptr %mll.old)
  %mll.new = call ptr @malloc(i64 4096)
  store volatile i64 %mll.i, ptr %mll.new, align 8
  store ptr %mll.new, ptr %mll.slot, align 8
  %mll.in = add nuw nsw i64 %mll.i, 1
  %mll.more = icmp ult i64 %mll.in, 100000
  br i1 %mll.more, label %mll.loop, label %mll.mid
mll.mid:
  %mll.t1 = call double @ut_now_sec()
  %mll.el = fsub double %mll.t1, %mll.t0
  %mll.warm = icmp eq i64 %mll.r, 0
  br i1 %mll.warm, label %mll.next, label %mll.store
mll.store:
  %mll.si = sub i64 %mll.r, 1
  %mll.sp = getelementptr inbounds [16 x double], ptr @ml.samp, i64 0, i64 %mll.si
  store double %mll.el, ptr %mll.sp, align 8
  br label %mll.next
mll.next:
  %mll.rn = add nuw nsw i64 %mll.r, 1
  %mll.rmore = icmp ult i64 %mll.rn, 17
  br i1 %mll.rmore, label %mll.rep, label %mll.drain
mll.drain:
  %mll.di = phi i64 [ 0, %mll.next ], [ %mll.din, %mll.drain ]
  %mll.ds = getelementptr inbounds [256 x ptr], ptr @ring, i64 0, i64 %mll.di
  %mll.dp = load ptr, ptr %mll.ds, align 8
  call void @free(ptr %mll.dp)
  %mll.din = add nuw nsw i64 %mll.di, 1
  %mll.dmore = icmp ult i64 %mll.din, 256
  br i1 %mll.dmore, label %mll.drain, label %mll.done
mll.done:
  call void @ut_report_dist(ptr @ml.samp, i64 16, i64 100000, ptr @lbl.ml)
  ret void
}


define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_routing()
  call void @test_oracle()
  call void @test_custom()
  call void @test_aligned()
  call void @test_null()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish

do.bench:
  call void @bench()
  br label %finish

finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
