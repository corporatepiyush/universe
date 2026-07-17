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

; HYBRID allocator — segregated SLABS for small, common, fixed size-classes +
; the growable TLSF as the general fallback. The classic small-fast /
; large-general split (from first principles): a stream of same-sized small
; objects is served by a size-class slab free list (one pointer pop, zero
; fragmentation, cache-dense); everything else delegates to TLSF (O(1),
; arbitrary sizes, real coalescing). Slab configuration is TUNABLE at create.
; This makes the SDK's DEFAULT allocator (TLSF) fast on the small-object common
; case without giving up TLSF's generality.
;
; DESIGN (own layout; composes the sibling universe_alloc_tlsf_* + _slab_*):
;   * ROUTING BY SIZE. alloc(size): if size <= small_max -> pick the smallest
;     size class >= size and pop from that class's slab; else delegate to TLSF.
;     The small path is a straight slab pop (no TLSF call, no lock) — the hot
;     common case. Single-threaded v1 (a future concurrent variant would give
;     each thread its own per-class slab magazine + a shared TLSF back end).
;   * FREE-OWNER IDENTIFICATION — chosen scheme: PREFIX HEADER (plan option
;     (a)), NOT the slab-region registry (b). Rationale: (b) would mask an
;     object DOWN to its span-aligned slab (`p & -span`), but each size class
;     has a DIFFERENT span, so you must know the class BEFORE you can find the
;     span — circular; a registry/bitmap of all slab spans adds a lookup on
;     every free. (a) is O(1), uniform across both paths, and — by also
;     recording the underlying base pointer — handles over-aligned allocations
;     (where the returned pointer is NOT base+16) with the SAME free path.
;     Every hybrid allocation carries a 16 B header just below the returned
;     pointer p:
;         p-16 : i64 tag    (-1 => TLSF-owned; 0..nclasses-1 => slab class idx)
;         p-8  : ptr base   (pointer to hand to the underlying free:
;                            slab cell for slab, tlsf raw ptr for TLSF)
;     free(p) reads {tag, base} at p-16 and routes: tag==-1 -> tlsf_free(base);
;     else slab_free(slab_handles[tag], base). No per-object slab search, no
;     cross-path free possible. Cost: 16 B/object (acceptable; the win is the
;     slab pop). The returned p is 16-aligned (slab cells and TLSF payloads are
;     16-aligned; header is exactly 16 B) so payloads keep natural alignment.
;   * alloc_aligned(size, align): align<=16 -> route by size as normal (payload
;     is already 16-aligned; slab path used for small). align>16 -> ALWAYS the
;     TLSF aligned path (regardless of size): request size+align+16 from
;     tlsf_alloc_aligned(.,align), which returns an align-aligned raw ptr; set
;     p = raw+align (still align-aligned, guarantees >=16 B header room before
;     p, all within the block), header {tag=-1, base=raw} at p-16.
;   * live() returns the TOTAL outstanding allocation COUNT across both paths
;     (a single i64 in the handle, +1/alloc, -1/free). Routing is verified in
;     the test by reading the sub-handles at the documented handle offsets and
;     querying universe_alloc_tlsf_live / _slab_live directly.
;
;   * CONFIG (caller-supplied at create; NULL => documented defaults):
;       +0  i64 small_max      ; largest size served by a slab
;       +8  i64 nclasses       ; number of size classes (used iff class_sizes!=0)
;       +16 ptr class_sizes    ; i64[nclasses] ASCENDING boundaries, or NULL
;       +24 i64 slab_chunk     ; target bytes/slab refill chunk (objs derived)
;       +32 i64 tlsf_initial   ; initial TLSF region size (growable large path)
;     Defaults (cfg==NULL): small_max=512, class_sizes=NULL, slab_chunk=64 KiB,
;     tlsf_initial=1 MiB. slab_chunk and tlsf_initial are OS-backed regions, so
;     each is CLAMPED UP to ALLOC_OS_MIN=16384 (16 KiB) after the zero=>default
;     substitution — a tuned-tiny config still honors the 16 KiB OS-request floor.
;     When class_sizes==NULL a STEP-16 schedule is generated
;     (16,32,48,64,...,ceil(small_max/16)*16); nclasses = ceil(small_max/16) and
;     small_max is snapped up to the top class. When class_sizes!=NULL the
;     provided ascending boundaries are used verbatim and small_max is set to
;     the top class (class_sizes[nclasses-1]).
;
;   * HANDLE (ONE malloc: header + class_sizes[] + slab_handles[]):
;       +0  ptr tlsf_h            ; growable TLSF (created with base==NULL)
;       +8  i64 small_max         ; routing boundary (== top class)
;       +16 i64 nclasses
;       +24 i64 live_count        ; total outstanding allocations
;       +32 i64 slab_chunk
;       +40 i64 tlsf_initial
;       +48 i64 reserved0
;       +56 i64 reserved1
;       +64 i64 class_sizes[nclasses]                 (ascending boundaries)
;       +64+8*nclasses ptr slab_handles[nclasses]     (one slab per class)
;
; ORDERINGS: single-threaded structure; no atomics.
;
; API (C ABI, nounwind):
;   ptr  universe_alloc_hybrid_create(ptr cfg)                 ; NULL => defaults
;   ptr  universe_alloc_hybrid_alloc(ptr h, i64 size)
;   ptr  universe_alloc_hybrid_alloc_aligned(ptr h, i64 size, i64 align)
;   void universe_alloc_hybrid_free(ptr h, ptr p)
;   i64  universe_alloc_hybrid_live(ptr h)
;   void universe_alloc_hybrid_destroy(ptr h)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare i64 @llvm.umax.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; sibling allocators (same domain -> resolved in libuniverse_allocator.a)
declare ptr  @universe_alloc_tlsf_create(ptr, i64)
declare ptr  @universe_alloc_tlsf_alloc(ptr, i64)
declare ptr  @universe_alloc_tlsf_alloc_aligned(ptr, i64, i64)
declare void @universe_alloc_tlsf_free(ptr, ptr)
declare void @universe_alloc_tlsf_destroy(ptr)
declare ptr  @universe_alloc_slab_create(i64, i64)
declare ptr  @universe_alloc_slab_alloc(ptr)
declare void @universe_alloc_slab_free(ptr, ptr)
declare void @universe_alloc_slab_destroy(ptr)

; ---- class lookup: smallest class index c with class_sizes[c] >= size -------
; Precondition (caller-guaranteed): size <= small_max == class_sizes[nclasses-1],
; so the loop always finds a class before running off the end; the atlast guard
; makes the final read in-bounds regardless (hazard #15: never read past end).
define internal i64 @class_index(ptr %h, i64 %size) #0 {
entry:
  %ncp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nc = load i64, ptr %ncp, align 8
  %ncm1 = add i64 %nc, -1
  %csbase = getelementptr inbounds nuw i8, ptr %h, i64 64
  br label %loop

loop:
  %c = phi i64 [ 0, %entry ], [ %cn, %cont ]
  %atlast = icmp uge i64 %c, %ncm1
  br i1 %atlast, label %retc, label %chk

chk:
  %slot = getelementptr inbounds i64, ptr %csbase, i64 %c
  %v = load i64, ptr %slot, align 8
  %ge = icmp uge i64 %v, %size
  br i1 %ge, label %retc, label %cont

cont:
  %cn = add i64 %c, 1
  br label %loop

retc:
  ret i64 %c
}

define noalias ptr @universe_alloc_hybrid_create(ptr %cfg) local_unnamed_addr #1 {
entry:
  %cfgnull = icmp eq ptr %cfg, null
  br i1 %cfgnull, label %defaults, label %readcfg

defaults:
  br label %haveconf

readcfg:
  %smrp = getelementptr inbounds nuw i8, ptr %cfg, i64 0
  %smr = load i64, ptr %smrp, align 8
  %ncrp = getelementptr inbounds nuw i8, ptr %cfg, i64 8
  %ncr = load i64, ptr %ncrp, align 8
  %csrp = getelementptr inbounds nuw i8, ptr %cfg, i64 16
  %csr = load ptr, ptr %csrp, align 8
  %chrp = getelementptr inbounds nuw i8, ptr %cfg, i64 24
  %chr = load i64, ptr %chrp, align 8
  %tirp = getelementptr inbounds nuw i8, ptr %cfg, i64 32
  %tir = load i64, ptr %tirp, align 8
  br label %haveconf

haveconf:
  %small_max0 = phi i64 [ 512, %defaults ], [ %smr, %readcfg ]
  %ncfg = phi i64 [ 0, %defaults ], [ %ncr, %readcfg ]
  %csin = phi ptr [ null, %defaults ], [ %csr, %readcfg ]
  %chunk0 = phi i64 [ 65536, %defaults ], [ %chr, %readcfg ]
  %tinit0 = phi i64 [ 1048576, %defaults ], [ %tir, %readcfg ]
  ; sanitize zero-valued knobs to defaults, then CLAMP UP to the OS-request floor
  ; ALLOC_OS_MIN=16384: a tuned-tiny slab_chunk/tlsf_initial still yields >= 16 KiB
  ; OS-backed regions (defaults 64 KiB / 1 MiB already comply). 0 => "use default".
  %chz = icmp eq i64 %chunk0, 0
  %chunk1 = select i1 %chz, i64 65536, i64 %chunk0
  %chunk = call i64 @llvm.umax.i64(i64 %chunk1, i64 16384)
  %tiz = icmp eq i64 %tinit0, 0
  %tinit1 = select i1 %tiz, i64 1048576, i64 %tinit0
  %tinit = call i64 @llvm.umax.i64(i64 %tinit1, i64 16384)
  %hascs = icmp ne ptr %csin, null
  br i1 %hascs, label %useprov, label %gen

useprov:
  %ncbad = icmp ult i64 %ncfg, 1
  br i1 %ncbad, label %fail0, label %provok, !prof !0

provok:
  %lastidx = sub i64 %ncfg, 1
  %lastp = getelementptr inbounds i64, ptr %csin, i64 %lastidx
  %smprov = load i64, ptr %lastp, align 8
  br label %haveN

gen:
  %sm16 = call i64 @llvm.umax.i64(i64 %small_max0, i64 16)
  %sm15 = add i64 %sm16, 15
  %ncgen = lshr i64 %sm15, 4
  %smgen = shl i64 %ncgen, 4
  br label %haveN

haveN:
  %nclasses = phi i64 [ %ncfg, %provok ], [ %ncgen, %gen ]
  %small_max = phi i64 [ %smprov, %provok ], [ %smgen, %gen ]
  %nctoobig = icmp ugt i64 %nclasses, 4096
  br i1 %nctoobig, label %fail0, label %sizeok, !prof !0

sizeok:
  ; nclasses <= 4096 -> arrays bounded; nuw safe
  %arrbytes = mul nuw i64 %nclasses, 16
  %hbytes = add nuw i64 %arrbytes, 64
  %h = call ptr @malloc(i64 %hbytes)
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail0, label %init, !prof !0

init:
  call void @llvm.memset.p0.i64(ptr %h, i8 0, i64 %hbytes, i1 false)
  %smsp = getelementptr inbounds nuw i8, ptr %h, i64 8
  store i64 %small_max, ptr %smsp, align 8
  %ncsp = getelementptr inbounds nuw i8, ptr %h, i64 16
  store i64 %nclasses, ptr %ncsp, align 8
  %chsp = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 %chunk, ptr %chsp, align 8
  %tisp = getelementptr inbounds nuw i8, ptr %h, i64 40
  store i64 %tinit, ptr %tisp, align 8
  %csbase = getelementptr inbounds nuw i8, ptr %h, i64 64
  %hascs2 = icmp ne ptr %csin, null
  br i1 %hascs2, label %copycs, label %gencs

copycs:
  %cpbytes = mul nuw i64 %nclasses, 8
  call void @llvm.memcpy.p0.p0.i64(ptr %csbase, ptr %csin, i64 %cpbytes, i1 false)
  br label %mktlsf

gencs:
  br label %gencs.loop

gencs.loop:
  %gi = phi i64 [ 0, %gencs ], [ %gin, %gencs.loop ]
  %val0 = add i64 %gi, 1
  %val = shl i64 %val0, 4
  %gslot = getelementptr inbounds i64, ptr %csbase, i64 %gi
  store i64 %val, ptr %gslot, align 8
  %gin = add i64 %gi, 1
  %gmore = icmp ult i64 %gin, %nclasses
  br i1 %gmore, label %gencs.loop, label %mktlsf

mktlsf:
  %tlsf = call ptr @universe_alloc_tlsf_create(ptr null, i64 %tinit)
  store ptr %tlsf, ptr %h, align 8
  %tlsfnull = icmp eq ptr %tlsf, null
  br i1 %tlsfnull, label %failh, label %mkslabs, !prof !0

mkslabs:
  %csbytes = mul nuw i64 %nclasses, 8
  %shoff = add nuw i64 64, %csbytes
  %shbase = getelementptr inbounds nuw i8, ptr %h, i64 %shoff
  br label %sl.loop

sl.loop:
  %si = phi i64 [ 0, %mkslabs ], [ %sin, %sl.cont ]
  %csslot = getelementptr inbounds i64, ptr %csbase, i64 %si
  %clsz = load i64, ptr %csslot, align 8
  %oadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %clsz, i64 16)
  %osz = extractvalue { i64, i1 } %oadd, 0
  %oov = extractvalue { i64, i1 } %oadd, 1
  br i1 %oov, label %failh, label %sl.mk, !prof !0

sl.mk:
  %opq = udiv i64 %chunk, %osz
  %ops = call i64 @llvm.umax.i64(i64 %opq, i64 1)
  %slab = call ptr @universe_alloc_slab_create(i64 %osz, i64 %ops)
  %shslot = getelementptr inbounds ptr, ptr %shbase, i64 %si
  store ptr %slab, ptr %shslot, align 8
  %slnull = icmp eq ptr %slab, null
  br i1 %slnull, label %failh, label %sl.cont, !prof !0

sl.cont:
  %sin = add i64 %si, 1
  %smore = icmp ult i64 %sin, %nclasses
  br i1 %smore, label %sl.loop, label %done

done:
  ret ptr %h

failh:
  call void @universe_alloc_hybrid_destroy(ptr %h)
  ret ptr null

fail0:
  ret ptr null
}

define ptr @universe_alloc_hybrid_alloc(ptr %h, i64 %size) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail, label %route, !prof !0

route:
  %smp = getelementptr inbounds nuw i8, ptr %h, i64 8
  %sm = load i64, ptr %smp, align 8
  %issmall = icmp ule i64 %size, %sm
  br i1 %issmall, label %small, label %large

small:
  %c = call i64 @class_index(ptr %h, i64 %size)
  %ncp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nc = load i64, ptr %ncp, align 8
  %csbytes = mul nuw i64 %nc, 8
  %shoff = add nuw i64 64, %csbytes
  %shbase = getelementptr inbounds nuw i8, ptr %h, i64 %shoff
  %shslot = getelementptr inbounds ptr, ptr %shbase, i64 %c
  %slab = load ptr, ptr %shslot, align 8
  %cell = call ptr @universe_alloc_slab_alloc(ptr %slab)
  %cnull = icmp eq ptr %cell, null
  br i1 %cnull, label %fail, label %small.hdr, !prof !0

small.hdr:
  store i64 %c, ptr %cell, align 8
  %sbasep = getelementptr inbounds nuw i8, ptr %cell, i64 8
  store ptr %cell, ptr %sbasep, align 8
  %sp = getelementptr inbounds nuw i8, ptr %cell, i64 16
  %slp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %sl = load i64, ptr %slp, align 8
  %sln = add i64 %sl, 1
  store i64 %sln, ptr %slp, align 8
  ret ptr %sp

large:
  %ladd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 16)
  %lnv = extractvalue { i64, i1 } %ladd, 0
  %lno = extractvalue { i64, i1 } %ladd, 1
  br i1 %lno, label %fail, label %large.go, !prof !0

large.go:
  %tlsf = load ptr, ptr %h, align 8
  %raw = call ptr @universe_alloc_tlsf_alloc(ptr %tlsf, i64 %lnv)
  %rnull = icmp eq ptr %raw, null
  br i1 %rnull, label %fail, label %large.hdr, !prof !0

large.hdr:
  store i64 -1, ptr %raw, align 8
  %rbasep = getelementptr inbounds nuw i8, ptr %raw, i64 8
  store ptr %raw, ptr %rbasep, align 8
  %rp = getelementptr inbounds nuw i8, ptr %raw, i64 16
  %llp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %ll = load i64, ptr %llp, align 8
  %lln = add i64 %ll, 1
  store i64 %lln, ptr %llp, align 8
  ret ptr %rp

fail:
  ret ptr null
}

define ptr @universe_alloc_hybrid_alloc_aligned(ptr %h, i64 %size, i64 %align) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %fail, label %chk, !prof !0

chk:
  %smallal = icmp ule i64 %align, 16
  br i1 %smallal, label %normal, label %big

normal:
  %rn = call ptr @universe_alloc_hybrid_alloc(ptr %h, i64 %size)
  ret ptr %rn

big:
  %a1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 %align)
  %a1v = extractvalue { i64, i1 } %a1, 0
  %a1o = extractvalue { i64, i1 } %a1, 1
  %a2 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a1v, i64 16)
  %needv = extractvalue { i64, i1 } %a2, 0
  %a2o = extractvalue { i64, i1 } %a2, 1
  %ovf = or i1 %a1o, %a2o
  br i1 %ovf, label %fail, label %big.go, !prof !0

big.go:
  %tlsf = load ptr, ptr %h, align 8
  %raw = call ptr @universe_alloc_tlsf_alloc_aligned(ptr %tlsf, i64 %needv, i64 %align)
  %rnull = icmp eq ptr %raw, null
  br i1 %rnull, label %fail, label %big.place, !prof !0

big.place:
  %p = getelementptr inbounds nuw i8, ptr %raw, i64 %align
  %hdr = getelementptr inbounds nuw i8, ptr %p, i64 -16
  store i64 -1, ptr %hdr, align 8
  %basep = getelementptr inbounds nuw i8, ptr %p, i64 -8
  store ptr %raw, ptr %basep, align 8
  %llp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %ll = load i64, ptr %llp, align 8
  %lln = add i64 %ll, 1
  store i64 %lln, ptr %llp, align 8
  ret ptr %p

fail:
  ret ptr null
}

define void @universe_alloc_hybrid_free(ptr %h, ptr %p) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  %pnull = icmp eq ptr %p, null
  %any = or i1 %hnull, %pnull
  br i1 %any, label %done, label %work, !prof !0

work:
  %hdr = getelementptr inbounds nuw i8, ptr %p, i64 -16
  %tag = load i64, ptr %hdr, align 8
  %basep = getelementptr inbounds nuw i8, ptr %p, i64 -8
  %base = load ptr, ptr %basep, align 8
  %istlsf = icmp eq i64 %tag, -1
  br i1 %istlsf, label %freetlsf, label %freeslab

freetlsf:
  %tlsf = load ptr, ptr %h, align 8
  call void @universe_alloc_tlsf_free(ptr %tlsf, ptr %base)
  br label %dec

freeslab:
  %ncp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nc = load i64, ptr %ncp, align 8
  %csbytes = mul nuw i64 %nc, 8
  %shoff = add nuw i64 64, %csbytes
  %shbase = getelementptr inbounds nuw i8, ptr %h, i64 %shoff
  %shslot = getelementptr inbounds ptr, ptr %shbase, i64 %tag
  %slab = load ptr, ptr %shslot, align 8
  call void @universe_alloc_slab_free(ptr %slab, ptr %base)
  br label %dec

dec:
  %lp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %l = load i64, ptr %lp, align 8
  %ln = add i64 %l, -1
  store i64 %ln, ptr %lp, align 8
  br label %done

done:
  ret void
}

define i64 @universe_alloc_hybrid_live(ptr %h) local_unnamed_addr #2 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %retz, label %work

work:
  %lp = getelementptr inbounds nuw i8, ptr %h, i64 24
  %l = load i64, ptr %lp, align 8
  ret i64 %l

retz:
  ret i64 0
}

define void @universe_alloc_hybrid_destroy(ptr %h) local_unnamed_addr #1 {
entry:
  %hnull = icmp eq ptr %h, null
  br i1 %hnull, label %done, label %work, !prof !0

work:
  %ncp = getelementptr inbounds nuw i8, ptr %h, i64 16
  %nc = load i64, ptr %ncp, align 8
  %csbytes = mul nuw i64 %nc, 8
  %shoff = add nuw i64 64, %csbytes
  %shbase = getelementptr inbounds nuw i8, ptr %h, i64 %shoff
  br label %loop

loop:
  %i = phi i64 [ 0, %work ], [ %in, %body ]
  %more = icmp ult i64 %i, %nc
  br i1 %more, label %body, label %afterslabs

body:
  %shslot = getelementptr inbounds ptr, ptr %shbase, i64 %i
  %slab = load ptr, ptr %shslot, align 8
  call void @universe_alloc_slab_destroy(ptr %slab)
  %in = add i64 %i, 1
  br label %loop

afterslabs:
  %tlsf = load ptr, ptr %h, align 8
  call void @universe_alloc_tlsf_destroy(ptr %tlsf)
  call void @free(ptr %h)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
