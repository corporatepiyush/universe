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

; universe_ds_art — adaptive radix tree, byte-string key -> i64 value, single
; thread. Adaptive inner nodes (Node4/16/48/256) + pessimistic path compression.
;
; ============================================================================
; DESIGN (from first principles)
; ----------------------------------------------------------------------------
;   ART indexes byte-string keys by consuming ONE key byte per tree level. Every
;   inner node fans out on the next key byte; the node KIND adapts to how many
;   distinct next-bytes actually occur so a node never wastes 256 slots for a
;   handful of children:
;       Node4   up to 4   children  (sorted keys[4]  + child[4])
;       Node16  up to 16  children  (sorted keys[16] + child[16], SIMD lookup)
;       Node48  up to 48  children  (index[256]->slot + child[48])
;       Node256 up to 256 children  (direct child[256])
;   Insert grows a full node to the next kind; delete shrinks it back when the
;   child count crosses a hysteresis threshold (256->48 at <=36, 48->16 at <=12,
;   16->4 at <=3). Leaves store the full key (in a shared arena) + the i64 value.
;
;   PATH COMPRESSION (pessimistic, capped-inline): a chain of single-child nodes
;   is collapsed into ONE node carrying the shared bytes as a "prefix". We keep
;   the TRUE prefix length (prefix_len, may be long) but store only the first
;   PREFIXLEN=16 bytes INLINE. When a descent/insert needs a prefix byte beyond
;   the inline window, it is read back from ANY descendant leaf's key (every leaf
;   under the node shares those bytes) at absolute position depth+i. This is the
;   classic "pessimistic" scheme: descent trusts the inline bytes and the final
;   full-key comparison at the leaf certifies the match, so a compressed prefix
;   never causes a false hit. Collapse on delete rewrites the merged prefix inline
;   straight from a descendant leaf (prefix byte j == leafkey[depth+j]).
;
;   KEYS THAT ARE PREFIXES OF OTHER KEYS ("app" vs "apple"): an inner node carries
;   an optional term_leaf (the value for the key that ENDS exactly at this node's
;   path). term_leaf sorts BEFORE all children (children have one more byte), which
;   also makes in-order iteration lexicographic.
;
;   INDEX-LINKED CHUNKED POOLS (no raw node pointers): one pool per kind; a pool
;   is a directory of fixed 256-node CHUNKS. A node reference is a TAGGED i32:
;   (kind<<29)|index. Chunks never move (only the directory of chunk pointers may
;   realloc), so a child-slot pointer stays valid across allocations of ANY kind
;   — this is why growth/shrink can hand the parent's slot around safely. NULL id
;   = -1. Freed nodes thread an intrusive i32 freelist through node word 0.
;
;   ORDERED / SEARCH ops exploit the radix == lexicographic byte order: iterate,
;   prefix_scan (all keys under a prefix — ART's signature strength), min, max.
;   Node4/16 keep keys sorted; Node48/256 iterate by ascending byte index.
;
;   Node16 child lookup uses a 16-lane byte compare (<16 x i8> icmp eq -> movemask
;   -> cttz) masked to the live child count; Node4 is a short scalar scan.
;
; ----------------------------------------------------------------------------
; Node reference: i32  (kind<<29)|index ; kind 0=leaf 1=N4 2=N16 3=N48 4=N256
;                 NULL = -1
; Inner header (32 B, kinds 1..4; word0 doubles as freelist-next when free):
;   num        i32   @0     child count
;   prefix_len i32   @4     TRUE compressed-prefix length (may exceed inline)
;   prefix     i8[16]@8     inline prefix bytes [8,24)
;   term_leaf  i32   @24    leaf id for key ending here, -1 = none
;   pad        i32   @28
;   N4:   keys i8[4] @32 [32,36)  child i32[4]  @36  [36,52)     stride 64
;   N16:  keys i8[16]@32 [32,48)  child i32[16] @48  [48,112)    stride 112
;   N48:  index i8[256]@32 [32,288) child i32[48]@288 [288,480)  stride 480
;   N256: child i32[256]@32 [32,1056)                            stride 1056
; Leaf record (24 B):
;   next  i32  @0   (freelist link while free)
;   keylen i32 @4
;   val   i64  @8
;   keyoff i64 @16  (byte offset into key arena)
; Header (240 B):
;   count      i64  @0
;   root       i32  @8   (-1 = empty)
;   arena_base ptr  @16
;   arena_cap  i64  @24
;   arena_used i64  @32
;   pools[5]        @40  each 40 B: dir ptr@0 dircap i64@8 nchunks i64@16
;                        nnodes i64@24 free i32@32
; ----------------------------------------------------------------------------
; API (0 OK, 1 NULL_PTR, 2 OOM, 4 EMPTY, 5 NOT_FOUND, 8 INVALID_ARG):
;   ptr  universe_ds_art_create()
;   void universe_ds_art_destroy(ptr)
;   i32  universe_ds_art_insert(ptr, ptr key, i64 klen, i64 val)  ; overwrite dup
;   i32  universe_ds_art_get(ptr, ptr key, i64 klen, ptr outval)
;   i32  universe_ds_art_contains(ptr, ptr key, i64 klen)         ; 1/0
;   i32  universe_ds_art_delete(ptr, ptr key, i64 klen)
;   i64  universe_ds_art_count(ptr)
;   i64  universe_ds_art_size(ptr)                                ; alias
;   i64  universe_ds_art_iterate(ptr, ptr cb, ptr ctx)            ; in-order
;   i64  universe_ds_art_prefix_scan(ptr, ptr pfx, i64 plen, ptr cb, ptr ctx)
;   i32  universe_ds_art_min(ptr, ptr outkeyptr, ptr outlen, ptr outval)
;   i32  universe_ds_art_max(ptr, ptr outkeyptr, ptr outlen, ptr outval)
;     callback: i32 cb(ptr ctx, ptr key, i64 klen, i64 val) ; nonzero => stop
; ============================================================================

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i16 @llvm.cttz.i16(i16, i1 immarg)

; ===========================================================================
; node addressing
; ===========================================================================
; stride in bytes for a kind (0..4)
define internal i64 @art_stride(i32 %kind) #0 {
entry:
  %isleaf = icmp eq i32 %kind, 0
  %is4 = icmp eq i32 %kind, 1
  %is16 = icmp eq i32 %kind, 2
  %is48 = icmp eq i32 %kind, 3
  %s1 = select i1 %is48, i64 480, i64 1056
  %s2 = select i1 %is16, i64 112, i64 %s1
  %s3 = select i1 %is4, i64 64, i64 %s2
  %s = select i1 %isleaf, i64 24, i64 %s3
  ret i64 %s
}

; node pointer from tagged id
define internal ptr @art_nptr(ptr %t, i32 %id) #0 {
entry:
  %kind = lshr i32 %id, 29
  %idx = and i32 %id, 536870911
  %idx64 = zext i32 %idx to i64
  %kind64 = zext i32 %kind to i64
  %ko = mul nuw i64 %kind64, 40
  %poff = add nuw i64 %ko, 40
  %poolp = getelementptr inbounds nuw i8, ptr %t, i64 %poff
  %dir = load ptr, ptr %poolp, align 8
  %stride = call i64 @art_stride(i32 %kind)
  %ci = lshr i64 %idx64, 8
  %within = and i64 %idx64, 255
  %cslot = getelementptr inbounds nuw ptr, ptr %dir, i64 %ci
  %chunk = load ptr, ptr %cslot, align 8
  %woff = mul nuw i64 %within, %stride
  %np = getelementptr inbounds nuw i8, ptr %chunk, i64 %woff
  ret ptr %np
}

; ===========================================================================
; pool allocation (chunked; nodes never move)
; ===========================================================================
; returns node index (i64) within its kind pool, or -1 on OOM
define internal i64 @art_pool_alloc(ptr %t, i32 %kind) #1 {
entry:
  %kind64 = zext i32 %kind to i64
  %ko = mul nuw i64 %kind64, 40
  %poff = add nuw i64 %ko, 40
  %poolp = getelementptr inbounds nuw i8, ptr %t, i64 %poff
  %stride = call i64 @art_stride(i32 %kind)
  %freep = getelementptr inbounds nuw i8, ptr %poolp, i64 32
  %free = load i32, ptr %freep, align 4
  %hasfree = icmp sge i32 %free, 0
  br i1 %hasfree, label %pop, label %bump

pop:
  %dir0 = load ptr, ptr %poolp, align 8
  %fi = zext i32 %free to i64
  %fci = lshr i64 %fi, 8
  %fwithin = and i64 %fi, 255
  %fcslot = getelementptr inbounds nuw ptr, ptr %dir0, i64 %fci
  %fchunk = load ptr, ptr %fcslot, align 8
  %fwoff = mul nuw i64 %fwithin, %stride
  %fnode = getelementptr inbounds nuw i8, ptr %fchunk, i64 %fwoff
  %next = load i32, ptr %fnode, align 4
  store i32 %next, ptr %freep, align 4
  ret i64 %fi

bump:
  %nnp = getelementptr inbounds nuw i8, ptr %poolp, i64 24
  %nn = load i64, ptr %nnp, align 8
  %ncp = getelementptr inbounds nuw i8, ptr %poolp, i64 16
  %nch = load i64, ptr %ncp, align 8
  %ci = lshr i64 %nn, 8
  %needchunk = icmp uge i64 %ci, %nch
  br i1 %needchunk, label %newchunk, label %place, !prof !1

newchunk:
  %dcp = getelementptr inbounds nuw i8, ptr %poolp, i64 8
  %dcap = load i64, ptr %dcp, align 8
  %dirfull = icmp uge i64 %nch, %dcap
  br i1 %dirfull, label %growdir, label %addchunk

growdir:
  %olddir = load ptr, ptr %poolp, align 8
  %dcap2 = shl i64 %dcap, 1
  %dbytes = shl i64 %dcap2, 3
  %ndir = call ptr @realloc(ptr %olddir, i64 %dbytes)
  %ndir.null = icmp eq ptr %ndir, null
  br i1 %ndir.null, label %fail, label %dirok, !prof !1

dirok:
  store ptr %ndir, ptr %poolp, align 8
  store i64 %dcap2, ptr %dcp, align 8
  br label %addchunk

addchunk:
  %cbytes = mul nuw i64 %stride, 256
  %chunk = call ptr @malloc(i64 %cbytes)
  %chunk.null = icmp eq ptr %chunk, null
  br i1 %chunk.null, label %fail, label %chunkok, !prof !1

chunkok:
  %dir1 = load ptr, ptr %poolp, align 8
  %newslot = getelementptr inbounds nuw ptr, ptr %dir1, i64 %nch
  store ptr %chunk, ptr %newslot, align 8
  %nch1 = add nuw i64 %nch, 1
  store i64 %nch1, ptr %ncp, align 8
  br label %place

place:
  %idx = load i64, ptr %nnp, align 8
  %idx1 = add nuw i64 %idx, 1
  store i64 %idx1, ptr %nnp, align 8
  ret i64 %idx

fail:
  ret i64 -1
}

; push a node onto its kind freelist
define internal void @art_pool_free(ptr %t, i32 %id) #1 {
entry:
  %kind = lshr i32 %id, 29
  %idx = and i32 %id, 536870911
  %kind64 = zext i32 %kind to i64
  %ko = mul nuw i64 %kind64, 40
  %poff = add nuw i64 %ko, 40
  %poolp = getelementptr inbounds nuw i8, ptr %t, i64 %poff
  %freep = getelementptr inbounds nuw i8, ptr %poolp, i64 32
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %fh = load i32, ptr %freep, align 4
  store i32 %fh, ptr %node, align 4
  store i32 %idx, ptr %freep, align 4
  ret void
}

; ===========================================================================
; key arena
; ===========================================================================
; append klen bytes, return offset (or -1 on OOM)
define internal i64 @art_arena_append(ptr %t, ptr %key, i64 %klen) #1 {
entry:
  %usedp = getelementptr inbounds nuw i8, ptr %t, i64 32
  %used = load i64, ptr %usedp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %t, i64 24
  %cap = load i64, ptr %capp, align 8
  %need = add i64 %used, %klen
  %fits = icmp ule i64 %need, %cap
  br i1 %fits, label %store, label %grow, !prof !2

grow:
  %cap2 = shl i64 %cap, 1
  %newcap = call i64 @llvm.umax.i64(i64 %cap2, i64 %need)
  %base0 = load ptr, ptr %t, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %base = load ptr, ptr %arbp, align 8
  %nb = call ptr @realloc(ptr %base, i64 %newcap)
  %nb.null = icmp eq ptr %nb, null
  br i1 %nb.null, label %fail, label %growok, !prof !1

growok:
  store ptr %nb, ptr %arbp, align 8
  store i64 %newcap, ptr %capp, align 8
  br label %store

store:
  %arbp2 = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp2, align 8
  %dst = getelementptr inbounds nuw i8, ptr %ab, i64 %used
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %key, i64 %klen, i1 false)
  store i64 %need, ptr %usedp, align 8
  ret i64 %used

fail:
  ret i64 -1
}

; allocate a leaf, return id (or -1)
define internal i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val) #1 {
entry:
  %idx = call i64 @art_pool_alloc(ptr %t, i32 0)
  %bad = icmp slt i64 %idx, 0
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %off = call i64 @art_arena_append(ptr %t, ptr %key, i64 %klen)
  %obad = icmp slt i64 %off, 0
  br i1 %obad, label %fail, label %fill, !prof !1

fill:
  %id = trunc i64 %idx to i32
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %lenp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %klen32 = trunc i64 %klen to i32
  store i32 %klen32, ptr %lenp, align 4
  %valp = getelementptr inbounds nuw i8, ptr %node, i64 8
  store i64 %val, ptr %valp, align 8
  %offp = getelementptr inbounds nuw i8, ptr %node, i64 16
  store i64 %off, ptr %offp, align 8
  ret i32 %id

fail:
  ret i32 -1
}

; allocate an inner node of the given kind, return id (or -1)
define internal i32 @art_alloc_node(ptr %t, i32 %kind) #1 {
entry:
  %idx = call i64 @art_pool_alloc(ptr %t, i32 %kind)
  %bad = icmp slt i64 %idx, 0
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %idx32 = trunc i64 %idx to i32
  %ksh = shl i32 %kind, 29
  %id = or i32 %ksh, %idx32
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  store i32 0, ptr %node, align 4
  %plp = getelementptr inbounds nuw i8, ptr %node, i64 4
  store i32 0, ptr %plp, align 4
  %tlp = getelementptr inbounds nuw i8, ptr %node, i64 24
  store i32 -1, ptr %tlp, align 4
  %is48 = icmp eq i32 %kind, 3
  br i1 %is48, label %init48, label %chk256

init48:
  %idxp = getelementptr inbounds nuw i8, ptr %node, i64 32
  call void @llvm.memset.p0.i64(ptr %idxp, i8 0, i64 256, i1 false)
  %chp48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  call void @llvm.memset.p0.i64(ptr %chp48, i8 -1, i64 192, i1 false)
  ret i32 %id

chk256:
  %is256 = icmp eq i32 %kind, 4
  br i1 %is256, label %init256, label %done

init256:
  %chp256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  call void @llvm.memset.p0.i64(ptr %chp256, i8 -1, i64 1024, i1 false)
  ret i32 %id

done:
  ret i32 %id

fail:
  ret i32 -1
}

; ===========================================================================
; small comparisons
; ===========================================================================
define internal i1 @art_keyeq(ptr %a, i64 %alen, ptr %b, i64 %blen) #3 {
entry:
  %lne = icmp ne i64 %alen, %blen
  br i1 %lne, label %no, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %in, %cont ]
  %done = icmp uge i64 %i, %alen
  br i1 %done, label %yes, label %step

step:
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %no, label %cont

cont:
  %in = add nuw i64 %i, 1
  br label %loop

yes:
  ret i1 true

no:
  ret i1 false
}

; common-prefix length starting at %start (bytes before %start assumed equal)
define internal i64 @art_cpl(ptr %a, i64 %alen, ptr %b, i64 %blen, i64 %start) #3 {
entry:
  %lim = call i64 @llvm.umin.i64(i64 %alen, i64 %blen)
  br label %loop

loop:
  %i = phi i64 [ %start, %entry ], [ %in, %cont ]
  %done = icmp uge i64 %i, %lim
  br i1 %done, label %ret, label %step

step:
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %ret, label %cont

cont:
  %in = add nuw i64 %i, 1
  br label %loop

ret:
  %r = sub i64 %i, %start
  ret i64 %r
}

; ===========================================================================
; child navigation
; ===========================================================================
; return pointer to the i32 child slot for byte %b, or null if absent
define internal ptr @art_findchild(ptr %t, i32 %id, i8 %b) #0 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  switch i32 %kind, label %none [ i32 1, label %f4
                                  i32 2, label %f16
                                  i32 3, label %f48
                                  i32 4, label %f256 ]

f4:
  %num4 = load i32, ptr %node, align 4
  %num4.64 = zext i32 %num4 to i64
  %keys4 = getelementptr inbounds nuw i8, ptr %node, i64 32
  %child4 = getelementptr inbounds nuw i8, ptr %node, i64 36
  br label %f4loop

f4loop:
  %j = phi i64 [ 0, %f4 ], [ %jn, %f4cont ]
  %atend = icmp uge i64 %j, %num4.64
  br i1 %atend, label %none, label %f4step

f4step:
  %kp4 = getelementptr inbounds nuw i8, ptr %keys4, i64 %j
  %kv4 = load i8, ptr %kp4, align 1
  %hit4 = icmp eq i8 %kv4, %b
  br i1 %hit4, label %f4hit, label %f4cont

f4hit:
  %slot4 = getelementptr inbounds nuw i32, ptr %child4, i64 %j
  ret ptr %slot4

f4cont:
  %jn = add nuw i64 %j, 1
  br label %f4loop

f16:
  %num16 = load i32, ptr %node, align 4
  %keys16 = getelementptr inbounds nuw i8, ptr %node, i64 32
  %vk = load <16 x i8>, ptr %keys16, align 16
  %bv0 = insertelement <16 x i8> poison, i8 %b, i64 0
  %bsplat = shufflevector <16 x i8> %bv0, <16 x i8> poison, <16 x i32> zeroinitializer
  %eqv = icmp eq <16 x i8> %vk, %bsplat
  %mask16 = bitcast <16 x i1> %eqv to i16
  %vm0 = shl nuw i32 1, %num16
  %vm1 = add i32 %vm0, -1
  %vm16 = trunc i32 %vm1 to i16
  %mask = and i16 %mask16, %vm16
  %nomatch = icmp eq i16 %mask, 0
  br i1 %nomatch, label %none, label %f16hit

f16hit:
  %tz = call i16 @llvm.cttz.i16(i16 %mask, i1 true)
  %tz64 = zext i16 %tz to i64
  %child16 = getelementptr inbounds nuw i8, ptr %node, i64 48
  %slot16 = getelementptr inbounds nuw i32, ptr %child16, i64 %tz64
  ret ptr %slot16

f48:
  %idxb = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b64 = zext i8 %b to i64
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %b64
  %iv = load i8, ptr %ixp, align 1
  %absent48 = icmp eq i8 %iv, 0
  br i1 %absent48, label %none, label %f48hit

f48hit:
  %iv64 = zext i8 %iv to i64
  %slotidx = sub nuw i64 %iv64, 1
  %child48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  %slot48 = getelementptr inbounds nuw i32, ptr %child48, i64 %slotidx
  ret ptr %slot48

f256:
  %child256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b64b = zext i8 %b to i64
  %slot256 = getelementptr inbounds nuw i32, ptr %child256, i64 %b64b
  %cv = load i32, ptr %slot256, align 4
  %absent256 = icmp eq i32 %cv, -1
  br i1 %absent256, label %none, label %f256hit

f256hit:
  ret ptr %slot256

none:
  ret ptr null
}

; first (lowest-byte) child id of an inner node
define internal i32 @art_firstchild(ptr %t, i32 %id) #4 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  %issmall = icmp ult i32 %kind, 3
  br i1 %issmall, label %small, label %big

small:
  %isk4.s = icmp eq i32 %kind, 1
  %choff.s = select i1 %isk4.s, i64 36, i64 48
  %chb.s = getelementptr inbounds nuw i8, ptr %node, i64 %choff.s
  %c0 = load i32, ptr %chb.s, align 4
  ret i32 %c0

big:
  %is48 = icmp eq i32 %kind, 3
  br i1 %is48, label %b48, label %b256

b48:
  %idxb = getelementptr inbounds nuw i8, ptr %node, i64 32
  br label %b48loop

b48loop:
  %j = phi i64 [ 0, %b48 ], [ %jn, %b48cont ]
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %j
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %b48hit, label %b48cont

b48hit:
  %iv64 = zext i8 %iv to i64
  %si = sub nuw i64 %iv64, 1
  %child48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  %sp = getelementptr inbounds nuw i32, ptr %child48, i64 %si
  %c48 = load i32, ptr %sp, align 4
  ret i32 %c48

b48cont:
  %jn = add nuw i64 %j, 1
  br label %b48loop

b256:
  %child256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  br label %b256loop

b256loop:
  %k = phi i64 [ 0, %b256 ], [ %kn, %b256cont ]
  %cp = getelementptr inbounds nuw i32, ptr %child256, i64 %k
  %cv = load i32, ptr %cp, align 4
  %present256 = icmp ne i32 %cv, -1
  br i1 %present256, label %b256hit, label %b256cont

b256hit:
  ret i32 %cv

b256cont:
  %kn = add nuw i64 %k, 1
  br label %b256loop
}

; first (lowest) child byte of an inner node
define internal i8 @art_firstbyte(ptr %t, i32 %id) #4 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  %issmall = icmp ult i32 %kind, 3
  br i1 %issmall, label %small, label %big

small:
  %keys = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b0 = load i8, ptr %keys, align 1
  ret i8 %b0

big:
  %is48 = icmp eq i32 %kind, 3
  %arrb = getelementptr inbounds nuw i8, ptr %node, i64 32
  br i1 %is48, label %b48, label %b256

b48:
  br label %b48loop

b48loop:
  %j = phi i64 [ 0, %b48 ], [ %jn, %b48cont ]
  %ixp = getelementptr inbounds nuw i8, ptr %arrb, i64 %j
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %hit, label %b48cont

b48cont:
  %jn = add nuw i64 %j, 1
  br label %b48loop

b256:
  br label %b256loop

b256loop:
  %k = phi i64 [ 0, %b256 ], [ %kn, %b256cont ]
  %cp = getelementptr inbounds nuw i32, ptr %arrb, i64 %k
  %cv = load i32, ptr %cp, align 4
  %present256 = icmp ne i32 %cv, -1
  br i1 %present256, label %hit256, label %b256cont

b256cont:
  %kn = add nuw i64 %k, 1
  br label %b256loop

hit:
  %r48 = trunc i64 %j to i8
  ret i8 %r48

hit256:
  %r256 = trunc i64 %k to i8
  ret i8 %r256
}

; last (highest-byte) child id of an inner node
define internal i32 @art_lastchild(ptr %t, i32 %id) #4 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  %issmall = icmp ult i32 %kind, 3
  br i1 %issmall, label %small, label %big

small:
  %num = load i32, ptr %node, align 4
  %num64 = zext i32 %num to i64
  %last = sub i64 %num64, 1
  %isk4.l = icmp eq i32 %kind, 1
  %choff = select i1 %isk4.l, i64 36, i64 48
  %chb = getelementptr inbounds nuw i8, ptr %node, i64 %choff
  %sp = getelementptr inbounds nuw i32, ptr %chb, i64 %last
  %c = load i32, ptr %sp, align 4
  ret i32 %c

big:
  %is48 = icmp eq i32 %kind, 3
  br i1 %is48, label %b48, label %b256

b48:
  %idxb = getelementptr inbounds nuw i8, ptr %node, i64 32
  br label %b48loop

b48loop:
  %j = phi i64 [ 255, %b48 ], [ %jn, %b48cont ]
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %j
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %b48hit, label %b48cont

b48hit:
  %iv64 = zext i8 %iv to i64
  %si = sub nuw i64 %iv64, 1
  %child48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  %spx = getelementptr inbounds nuw i32, ptr %child48, i64 %si
  %c48 = load i32, ptr %spx, align 4
  ret i32 %c48

b48cont:
  %jn = sub i64 %j, 1
  br label %b48loop

b256:
  %child256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  br label %b256loop

b256loop:
  %k = phi i64 [ 255, %b256 ], [ %kn, %b256cont ]
  %cp = getelementptr inbounds nuw i32, ptr %child256, i64 %k
  %cv = load i32, ptr %cp, align 4
  %present256 = icmp ne i32 %cv, -1
  br i1 %present256, label %b256hit, label %b256cont

b256hit:
  ret i32 %cv

b256cont:
  %kn = sub i64 %k, 1
  br label %b256loop
}

; any descendant leaf id (for prefix reconstruction)
define internal i32 @art_minleaf(ptr %t, i32 %id0) #4 {
entry:
  br label %loop

loop:
  %id = phi i32 [ %id0, %entry ], [ %nid, %step ]
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %ret, label %step

step:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %tlp = getelementptr inbounds nuw i8, ptr %node, i64 24
  %term = load i32, ptr %tlp, align 4
  %hasterm = icmp ne i32 %term, -1
  %fc = call i32 @art_firstchild(ptr %t, i32 %id)
  %nid = select i1 %hasterm, i32 %term, i32 %fc
  br label %loop

ret:
  ret i32 %id
}

; absolute prefix byte j of node (inline if j<16, else from descendant leaf)
define internal i8 @art_pbyte(ptr %node, ptr %minkey, i64 %depth, i64 %j) #0 {
entry:
  %inline = icmp ult i64 %j, 16
  br i1 %inline, label %ini, label %ext

ini:
  %pp = getelementptr inbounds nuw i8, ptr %node, i64 8
  %ipp = getelementptr inbounds nuw i8, ptr %pp, i64 %j
  %iv = load i8, ptr %ipp, align 1
  ret i8 %iv

ext:
  %pos = add i64 %depth, %j
  %ep = getelementptr inbounds nuw i8, ptr %minkey, i64 %pos
  %ev = load i8, ptr %ep, align 1
  ret i8 %ev
}

; match a node's compressed prefix against key at depth.
; out status: 0 FULL, 1 MISMATCH, 2 KEY_EXHAUSTED. returns matched byte count.
define internal i64 @art_prefix_match(ptr %t, i32 %id, ptr %key, i64 %klen, i64 %depth, ptr %status) #1 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %plp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %pl32 = load i32, ptr %plp, align 4
  %pl = zext i32 %pl32 to i64
  %zero = icmp eq i64 %pl, 0
  br i1 %zero, label %full0, label %needmin

full0:
  store i32 0, ptr %status, align 4
  ret i64 0

needmin:
  %long = icmp ugt i64 %pl, 16
  br i1 %long, label %getmin, label %loop

getmin:
  %mlid = call i32 @art_minleaf(ptr %t, i32 %id)
  %mlnode = call ptr @art_nptr(ptr %t, i32 %mlid)
  %offp = getelementptr inbounds nuw i8, ptr %mlnode, i64 16
  %moff = load i64, ptr %offp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %mk = getelementptr inbounds nuw i8, ptr %ab, i64 %moff
  br label %loop

loop:
  %minkey = phi ptr [ null, %needmin ], [ %mk, %getmin ], [ %minkey, %cont ]
  %i = phi i64 [ 0, %needmin ], [ 0, %getmin ], [ %in, %cont ]
  %done = icmp uge i64 %i, %pl
  br i1 %done, label %full, label %step

step:
  %pos = add i64 %depth, %i
  %kexh = icmp uge i64 %pos, %klen
  br i1 %kexh, label %exhausted, label %cmp

cmp:
  %a = call i8 @art_pbyte(ptr %node, ptr %minkey, i64 %depth, i64 %i)
  %kp = getelementptr inbounds nuw i8, ptr %key, i64 %pos
  %kb = load i8, ptr %kp, align 1
  %ne = icmp ne i8 %a, %kb
  br i1 %ne, label %mismatch, label %cont

cont:
  %in = add nuw i64 %i, 1
  br label %loop

full:
  store i32 0, ptr %status, align 4
  ret i64 %pl

mismatch:
  store i32 1, ptr %status, align 4
  ret i64 %i

exhausted:
  store i32 2, ptr %status, align 4
  ret i64 %i
}

; ===========================================================================
; node growth (returns new id, or -1 on OOM)
; ===========================================================================
; copy the 32-byte inner header (num,prefix_len,prefix,term_leaf) from src->dst
define internal void @art_copyhdr(ptr %dst, ptr %src) #1 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 32, i1 false)
  ret void
}

define internal i32 @art_grow4to16(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 2)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %okeys = getelementptr inbounds nuw i8, ptr %old, i64 32
  %nkeys = getelementptr inbounds nuw i8, ptr %new, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr %nkeys, ptr %okeys, i64 4, i1 false)
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 36
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 48
  call void @llvm.memcpy.p0.p0.i64(ptr %nchild, ptr %ochild, i64 16, i1 false)
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 -1
}

define internal i32 @art_grow16to48(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 3)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %okeys = getelementptr inbounds nuw i8, ptr %old, i64 32
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 48
  %nindex = getelementptr inbounds nuw i8, ptr %new, i64 32
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 288
  call void @llvm.memcpy.p0.p0.i64(ptr %nchild, ptr %ochild, i64 64, i1 false)
  br label %loop

loop:
  %j = phi i64 [ 0, %go ], [ %j1, %loop ]
  %kp = getelementptr inbounds nuw i8, ptr %okeys, i64 %j
  %b = load i8, ptr %kp, align 1
  %b64 = zext i8 %b to i64
  %ixp = getelementptr inbounds nuw i8, ptr %nindex, i64 %b64
  %j1 = add nuw i64 %j, 1
  %slotv = trunc i64 %j1 to i8
  store i8 %slotv, ptr %ixp, align 1
  %more = icmp ult i64 %j1, 16
  br i1 %more, label %loop, label %fin

fin:
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 -1
}

define internal i32 @art_grow48to256(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 4)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %oindex = getelementptr inbounds nuw i8, ptr %old, i64 32
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 288
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 32
  br label %loop

loop:
  %b = phi i64 [ 0, %go ], [ %bn, %cont ]
  %ixp = getelementptr inbounds nuw i8, ptr %oindex, i64 %b
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %set, label %cont

set:
  %iv64 = zext i8 %iv to i64
  %si = sub nuw i64 %iv64, 1
  %osp = getelementptr inbounds nuw i32, ptr %ochild, i64 %si
  %cv = load i32, ptr %osp, align 4
  %nsp = getelementptr inbounds nuw i32, ptr %nchild, i64 %b
  store i32 %cv, ptr %nsp, align 4
  br label %cont

cont:
  %bn = add nuw i64 %b, 1
  %more = icmp ult i64 %bn, 256
  br i1 %more, label %loop, label %fin

fin:
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 -1
}

; ===========================================================================
; add a child (byte not already present). handles growth; updates *ref.
; returns 0 ok, 2 OOM.
; ===========================================================================
define internal i32 @art_add_child(ptr %t, ptr %ref, i32 %id0, i8 %b, i32 %childid) #1 {
entry:
  %kind0 = lshr i32 %id0, 29
  %node0 = call ptr @art_nptr(ptr %t, i32 %id0)
  %num0 = load i32, ptr %node0, align 4
  ; grow if full
  %full4 = icmp eq i32 %num0, 4
  %isk4 = icmp eq i32 %kind0, 1
  %g4 = and i1 %isk4, %full4
  br i1 %g4, label %grow4, label %chk16

grow4:
  %n4 = call i32 @art_grow4to16(ptr %t, i32 %id0)
  %n4bad = icmp eq i32 %n4, -1
  br i1 %n4bad, label %oom, label %n4ok, !prof !1

n4ok:
  store i32 %n4, ptr %ref, align 4
  br label %dispatch

chk16:
  %full16 = icmp eq i32 %num0, 16
  %isk16 = icmp eq i32 %kind0, 2
  %g16 = and i1 %isk16, %full16
  br i1 %g16, label %grow16, label %chk48

grow16:
  %n16 = call i32 @art_grow16to48(ptr %t, i32 %id0)
  %n16bad = icmp eq i32 %n16, -1
  br i1 %n16bad, label %oom, label %n16ok, !prof !1

n16ok:
  store i32 %n16, ptr %ref, align 4
  br label %dispatch

chk48:
  %full48 = icmp eq i32 %num0, 48
  %isk48 = icmp eq i32 %kind0, 3
  %g48 = and i1 %isk48, %full48
  br i1 %g48, label %grow48, label %same

grow48:
  %n48 = call i32 @art_grow48to256(ptr %t, i32 %id0)
  %n48bad = icmp eq i32 %n48, -1
  br i1 %n48bad, label %oom, label %n48ok, !prof !1

n48ok:
  store i32 %n48, ptr %ref, align 4
  br label %dispatch

same:
  br label %dispatch

dispatch:
  %id = load i32, ptr %ref, align 4
  %kind = lshr i32 %id, 29
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %num = load i32, ptr %node, align 4
  %num64 = zext i32 %num to i64
  switch i32 %kind, label %oom [ i32 1, label %ins_small
                                 i32 2, label %ins_small
                                 i32 3, label %ins48
                                 i32 4, label %ins256 ]

ins_small:
  %isk4.i = icmp eq i32 %kind, 1
  %choff = select i1 %isk4.i, i64 36, i64 48
  %keys = getelementptr inbounds nuw i8, ptr %node, i64 32
  %child = getelementptr inbounds nuw i8, ptr %node, i64 %choff
  br label %posloop

posloop:
  %p = phi i64 [ 0, %ins_small ], [ %pn, %poscont ]
  %pend = icmp uge i64 %p, %num64
  br i1 %pend, label %doins, label %poscmp

poscmp:
  %kp = getelementptr inbounds nuw i8, ptr %keys, i64 %p
  %kv = load i8, ptr %kp, align 1
  %gt = icmp ugt i8 %kv, %b
  br i1 %gt, label %doins, label %poscont

poscont:
  %pn = add nuw i64 %p, 1
  br label %posloop

doins:
  %tail = sub i64 %num64, %p
  %p1 = add nuw i64 %p, 1
  %ksrc = getelementptr inbounds nuw i8, ptr %keys, i64 %p
  %kdst = getelementptr inbounds nuw i8, ptr %keys, i64 %p1
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %tail, i1 false)
  %csrc = getelementptr inbounds nuw i32, ptr %child, i64 %p
  %cdst = getelementptr inbounds nuw i32, ptr %child, i64 %p1
  %tailb = shl i64 %tail, 2
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %tailb, i1 false)
  store i8 %b, ptr %ksrc, align 1
  store i32 %childid, ptr %csrc, align 4
  %nums1 = add nuw i32 %num, 1
  store i32 %nums1, ptr %node, align 4
  ret i32 0

ins48:
  %idxb = getelementptr inbounds nuw i8, ptr %node, i64 32
  %child48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  br label %slotloop

slotloop:
  %s = phi i64 [ 0, %ins48 ], [ %sn, %slotcont ]
  %sp = getelementptr inbounds nuw i32, ptr %child48, i64 %s
  %sv = load i32, ptr %sp, align 4
  %free = icmp eq i32 %sv, -1
  br i1 %free, label %slotfound, label %slotcont

slotcont:
  %sn = add nuw i64 %s, 1
  br label %slotloop

slotfound:
  store i32 %childid, ptr %sp, align 4
  %b64.48 = zext i8 %b to i64
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %b64.48
  %s1 = add nuw i64 %s, 1
  %sv8 = trunc i64 %s1 to i8
  store i8 %sv8, ptr %ixp, align 1
  %num48.1 = add nuw i32 %num, 1
  store i32 %num48.1, ptr %node, align 4
  ret i32 0

ins256:
  %child256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b64.256 = zext i8 %b to i64
  %sp256 = getelementptr inbounds nuw i32, ptr %child256, i64 %b64.256
  store i32 %childid, ptr %sp256, align 4
  %num256.1 = add nuw i32 %num, 1
  store i32 %num256.1, ptr %node, align 4
  ret i32 0

oom:
  ret i32 2
}

; ===========================================================================
; remove a child (byte present). num--. arrays compacted. no shrink here.
; ===========================================================================
define internal void @art_remove_child(ptr %t, i32 %id, i8 %b) #1 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  %num = load i32, ptr %node, align 4
  %num64 = zext i32 %num to i64
  switch i32 %kind, label %ret [ i32 1, label %small
                                 i32 2, label %small
                                 i32 3, label %r48
                                 i32 4, label %r256 ]

small:
  %isk4.r = icmp eq i32 %kind, 1
  %choff = select i1 %isk4.r, i64 36, i64 48
  %keys = getelementptr inbounds nuw i8, ptr %node, i64 32
  %child = getelementptr inbounds nuw i8, ptr %node, i64 %choff
  br label %findloop

findloop:
  %j = phi i64 [ 0, %small ], [ %jn, %findcont ]
  %kp = getelementptr inbounds nuw i8, ptr %keys, i64 %j
  %kv = load i8, ptr %kp, align 1
  %hit = icmp eq i8 %kv, %b
  br i1 %hit, label %doremove, label %findcont

findcont:
  %jn = add nuw i64 %j, 1
  br label %findloop

doremove:
  %j1 = add nuw i64 %j, 1
  %tail = sub i64 %num64, %j1
  %ksrc = getelementptr inbounds nuw i8, ptr %keys, i64 %j1
  %kdst = getelementptr inbounds nuw i8, ptr %keys, i64 %j
  call void @llvm.memmove.p0.p0.i64(ptr %kdst, ptr %ksrc, i64 %tail, i1 false)
  %csrc = getelementptr inbounds nuw i32, ptr %child, i64 %j1
  %cdst = getelementptr inbounds nuw i32, ptr %child, i64 %j
  %tailb = shl i64 %tail, 2
  call void @llvm.memmove.p0.p0.i64(ptr %cdst, ptr %csrc, i64 %tailb, i1 false)
  %nums1 = sub nuw i32 %num, 1
  store i32 %nums1, ptr %node, align 4
  ret void

r48:
  %idxb = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b64 = zext i8 %b to i64
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %b64
  %iv = load i8, ptr %ixp, align 1
  %iv64 = zext i8 %iv to i64
  %si = sub nuw i64 %iv64, 1
  %child48 = getelementptr inbounds nuw i8, ptr %node, i64 288
  %sp = getelementptr inbounds nuw i32, ptr %child48, i64 %si
  store i32 -1, ptr %sp, align 4
  store i8 0, ptr %ixp, align 1
  %num48.1 = sub nuw i32 %num, 1
  store i32 %num48.1, ptr %node, align 4
  ret void

r256:
  %child256 = getelementptr inbounds nuw i8, ptr %node, i64 32
  %b64.256 = zext i8 %b to i64
  %sp256 = getelementptr inbounds nuw i32, ptr %child256, i64 %b64.256
  store i32 -1, ptr %sp256, align 4
  %num256.1 = sub nuw i32 %num, 1
  store i32 %num256.1, ptr %node, align 4
  ret void

ret:
  ret void
}

; ===========================================================================
; node shrink (returns new id, or same id if not shrunk / OOM)
; ===========================================================================
define internal i32 @art_shrink256to48(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 3)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 32
  %nindex = getelementptr inbounds nuw i8, ptr %new, i64 32
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 288
  br label %loop

loop:
  %b = phi i64 [ 0, %go ], [ %bn, %cont ]
  %s = phi i64 [ 0, %go ], [ %snext, %cont ]
  %osp = getelementptr inbounds nuw i32, ptr %ochild, i64 %b
  %cv = load i32, ptr %osp, align 4
  %present = icmp ne i32 %cv, -1
  br i1 %present, label %set, label %cont

set:
  %nsp = getelementptr inbounds nuw i32, ptr %nchild, i64 %s
  store i32 %cv, ptr %nsp, align 4
  %ixp = getelementptr inbounds nuw i8, ptr %nindex, i64 %b
  %s1 = add nuw i64 %s, 1
  %sv8 = trunc i64 %s1 to i8
  store i8 %sv8, ptr %ixp, align 1
  br label %cont

cont:
  %snext = phi i64 [ %s1, %set ], [ %s, %loop ]
  %bn = add nuw i64 %b, 1
  %more = icmp ult i64 %bn, 256
  br i1 %more, label %loop, label %fin

fin:
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 %id
}

define internal i32 @art_shrink48to16(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 2)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %oindex = getelementptr inbounds nuw i8, ptr %old, i64 32
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 288
  %nkeys = getelementptr inbounds nuw i8, ptr %new, i64 32
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 48
  br label %loop

loop:
  %b = phi i64 [ 0, %go ], [ %bn, %cont ]
  %s = phi i64 [ 0, %go ], [ %snext, %cont ]
  %ixp = getelementptr inbounds nuw i8, ptr %oindex, i64 %b
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %set, label %cont

set:
  %iv64 = zext i8 %iv to i64
  %osi = sub nuw i64 %iv64, 1
  %osp = getelementptr inbounds nuw i32, ptr %ochild, i64 %osi
  %cv = load i32, ptr %osp, align 4
  %nsp = getelementptr inbounds nuw i32, ptr %nchild, i64 %s
  store i32 %cv, ptr %nsp, align 4
  %kp = getelementptr inbounds nuw i8, ptr %nkeys, i64 %s
  %b8 = trunc i64 %b to i8
  store i8 %b8, ptr %kp, align 1
  %s1 = add nuw i64 %s, 1
  br label %cont

cont:
  %snext = phi i64 [ %s1, %set ], [ %s, %loop ]
  %bn = add nuw i64 %b, 1
  %more = icmp ult i64 %bn, 256
  br i1 %more, label %loop, label %fin

fin:
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 %id
}

define internal i32 @art_shrink16to4(ptr %t, i32 %id) #1 {
entry:
  %nid = call i32 @art_alloc_node(ptr %t, i32 1)
  %bad = icmp eq i32 %nid, -1
  br i1 %bad, label %fail, label %go, !prof !1

go:
  %old = call ptr @art_nptr(ptr %t, i32 %id)
  %new = call ptr @art_nptr(ptr %t, i32 %nid)
  call void @art_copyhdr(ptr %new, ptr %old)
  %num = load i32, ptr %old, align 4
  %num64 = zext i32 %num to i64
  %okeys = getelementptr inbounds nuw i8, ptr %old, i64 32
  %nkeys = getelementptr inbounds nuw i8, ptr %new, i64 32
  call void @llvm.memcpy.p0.p0.i64(ptr %nkeys, ptr %okeys, i64 %num64, i1 false)
  %ochild = getelementptr inbounds nuw i8, ptr %old, i64 48
  %nchild = getelementptr inbounds nuw i8, ptr %new, i64 36
  %cb = shl i64 %num64, 2
  call void @llvm.memcpy.p0.p0.i64(ptr %nchild, ptr %ochild, i64 %cb, i1 false)
  call void @art_pool_free(ptr %t, i32 %id)
  ret i32 %nid

fail:
  ret i32 %id
}

; ===========================================================================
; collapse/shrink a node after a deletion. updates *ref.
; ===========================================================================
define internal void @art_collapse(ptr %t, ptr %ref, i32 %id, i64 %depth) #1 {
entry:
  %node = call ptr @art_nptr(ptr %t, i32 %id)
  %kind = lshr i32 %id, 29
  %num = load i32, ptr %node, align 4
  %tlp = getelementptr inbounds nuw i8, ptr %node, i64 24
  %term = load i32, ptr %tlp, align 4
  %hasterm = icmp ne i32 %term, -1
  %empty = icmp eq i32 %num, 0
  br i1 %empty, label %isempty, label %chksingle

isempty:
  br i1 %hasterm, label %emptyterm, label %emptynone

emptynone:
  store i32 -1, ptr %ref, align 4
  call void @art_pool_free(ptr %t, i32 %id)
  ret void

emptyterm:
  store i32 %term, ptr %ref, align 4
  call void @art_pool_free(ptr %t, i32 %id)
  ret void

chksingle:
  %one = icmp eq i32 %num, 1
  %single = and i1 %one, %hasterm
  %singlecollapse = xor i1 %hasterm, true
  %docollapse = and i1 %one, %singlecollapse
  br i1 %docollapse, label %collapse, label %shrink

collapse:
  %ccid = call i32 @art_firstchild(ptr %t, i32 %id)
  %cckind = lshr i32 %ccid, 29
  %ccleaf = icmp eq i32 %cckind, 0
  br i1 %ccleaf, label %leafup, label %mergeup

leafup:
  store i32 %ccid, ptr %ref, align 4
  call void @art_pool_free(ptr %t, i32 %id)
  ret void

mergeup:
  ; new merged prefix bytes come from any descendant leaf at depth+j
  %mlid = call i32 @art_minleaf(ptr %t, i32 %ccid)
  %mlnode = call ptr @art_nptr(ptr %t, i32 %mlid)
  %moffp = getelementptr inbounds nuw i8, ptr %mlnode, i64 16
  %moff = load i64, ptr %moffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %mk = getelementptr inbounds nuw i8, ptr %ab, i64 %moff
  %nplp = getelementptr inbounds nuw i8, ptr %node, i64 4
  %npl32 = load i32, ptr %nplp, align 4
  %npl = zext i32 %npl32 to i64
  %ccnode = call ptr @art_nptr(ptr %t, i32 %ccid)
  %cplp = getelementptr inbounds nuw i8, ptr %ccnode, i64 4
  %cpl32 = load i32, ptr %cplp, align 4
  %cpl = zext i32 %cpl32 to i64
  %sum0 = add i64 %npl, %cpl
  %newlen = add i64 %sum0, 1
  %newlen32 = trunc i64 %newlen to i32
  store i32 %newlen32, ptr %cplp, align 4
  %copy = call i64 @llvm.umin.i64(i64 %newlen, i64 16)
  %src = getelementptr inbounds nuw i8, ptr %mk, i64 %depth
  %dstpfx = getelementptr inbounds nuw i8, ptr %ccnode, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %dstpfx, ptr %src, i64 %copy, i1 false)
  store i32 %ccid, ptr %ref, align 4
  call void @art_pool_free(ptr %t, i32 %id)
  ret void

shrink:
  %is256 = icmp eq i32 %kind, 4
  %s256 = icmp ule i32 %num, 36
  %do256 = and i1 %is256, %s256
  br i1 %do256, label %sh256, label %chk48

sh256:
  %r256 = call i32 @art_shrink256to48(ptr %t, i32 %id)
  store i32 %r256, ptr %ref, align 4
  ret void

chk48:
  %is48 = icmp eq i32 %kind, 3
  %s48 = icmp ule i32 %num, 12
  %do48 = and i1 %is48, %s48
  br i1 %do48, label %sh48, label %chk16

sh48:
  %r48 = call i32 @art_shrink48to16(ptr %t, i32 %id)
  store i32 %r48, ptr %ref, align 4
  ret void

chk16:
  %is16 = icmp eq i32 %kind, 2
  %s16 = icmp ule i32 %num, 3
  %do16 = and i1 %is16, %s16
  br i1 %do16, label %sh16, label %noshrink

sh16:
  %r16 = call i32 @art_shrink16to4(ptr %t, i32 %id)
  store i32 %r16, ptr %ref, align 4
  ret void

noshrink:
  ret void
}

; ===========================================================================
; create / destroy
; ===========================================================================
define noalias ptr @universe_ds_art_create() local_unnamed_addr #5 {
entry:
  %hdr = call ptr @malloc(i64 240)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %arena, !prof !1

arena:
  %ar = call ptr @malloc(i64 4096)
  %ar.null = icmp eq ptr %ar, null
  br i1 %ar.null, label %freehdr, label %init, !prof !1

freehdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  call void @llvm.memset.p0.i64(ptr %hdr, i8 0, i64 240, i1 false)
  store i64 0, ptr %hdr, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i32 -1, ptr %rootp, align 4
  %arbp = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store ptr %ar, ptr %arbp, align 8
  %capp = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store i64 4096, ptr %capp, align 8
  %usedp = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 0, ptr %usedp, align 8
  ; init 5 pool directories
  br label %ploop

ploop:
  %k = phi i64 [ 0, %init ], [ %kn, %pcont ]
  %ko = mul nuw i64 %k, 40
  %poff = add nuw i64 %ko, 40
  %poolp = getelementptr inbounds nuw i8, ptr %hdr, i64 %poff
  %dir = call ptr @malloc(i64 64)
  %dir.null = icmp eq ptr %dir, null
  br i1 %dir.null, label %cleanup, label %pset, !prof !1

pset:
  store ptr %dir, ptr %poolp, align 8
  %dcp = getelementptr inbounds nuw i8, ptr %poolp, i64 8
  store i64 8, ptr %dcp, align 8
  %ncp = getelementptr inbounds nuw i8, ptr %poolp, i64 16
  store i64 0, ptr %ncp, align 8
  %nnp = getelementptr inbounds nuw i8, ptr %poolp, i64 24
  store i64 0, ptr %nnp, align 8
  %freep = getelementptr inbounds nuw i8, ptr %poolp, i64 32
  store i32 -1, ptr %freep, align 4
  br label %pcont

pcont:
  %kn = add nuw i64 %k, 1
  %more = icmp ult i64 %kn, 5
  br i1 %more, label %ploop, label %done

done:
  ret ptr %hdr

cleanup:
  br label %clloop

clloop:
  %ci = phi i64 [ 0, %cleanup ], [ %cin, %clbody ]
  %cdone = icmp uge i64 %ci, %k
  br i1 %cdone, label %clfin, label %clbody

clbody:
  %cko = shl nuw i64 %ci, 5
  %cpoff = add nuw i64 %cko, 40
  %cpoolp = getelementptr inbounds nuw i8, ptr %hdr, i64 %cpoff
  %cdir = load ptr, ptr %cpoolp, align 8
  call void @free(ptr %cdir)
  %cin = add nuw i64 %ci, 1
  br label %clloop

clfin:
  call void @free(ptr %ar)
  call void @free(ptr nonnull %hdr)
  br label %fail

fail:
  ret ptr null
}

define void @universe_ds_art_destroy(ptr %t) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %done, label %pools, !prof !1

pools:
  br label %ploop

ploop:
  %k = phi i64 [ 0, %pools ], [ %kn, %pcont ]
  %ko = mul nuw i64 %k, 40
  %poff = add nuw i64 %ko, 40
  %poolp = getelementptr inbounds nuw i8, ptr %t, i64 %poff
  %dir = load ptr, ptr %poolp, align 8
  %ncp = getelementptr inbounds nuw i8, ptr %poolp, i64 16
  %nch = load i64, ptr %ncp, align 8
  br label %chloop

chloop:
  %c = phi i64 [ 0, %ploop ], [ %cn, %chbody ]
  %cdone = icmp uge i64 %c, %nch
  br i1 %cdone, label %freedir, label %chbody

chbody:
  %cslot = getelementptr inbounds nuw ptr, ptr %dir, i64 %c
  %chunk = load ptr, ptr %cslot, align 8
  call void @free(ptr %chunk)
  %cn = add nuw i64 %c, 1
  br label %chloop

freedir:
  call void @free(ptr %dir)
  br label %pcont

pcont:
  %kn = add nuw i64 %k, 1
  %more = icmp ult i64 %kn, 5
  br i1 %more, label %ploop, label %freerest

freerest:
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ar = load ptr, ptr %arbp, align 8
  call void @free(ptr %ar)
  call void @free(ptr nonnull %t)
  br label %done

done:
  ret void
}

; ===========================================================================
; insert
; ===========================================================================
define i32 @universe_ds_art_insert(ptr %t, ptr %key, i64 %klen, i64 %val) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %tn, %kn
  br i1 %bad, label %err.null, label %setup, !prof !1

err.null:
  ret i32 1

setup:
  %stslot = alloca i32, align 4
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  br label %loop

loop:
  %ref = phi ptr [ %rootp, %setup ], [ %nextref, %godown ]
  %depth = phi i64 [ 0, %setup ], [ %depth3, %godown ]
  %id = load i32, ptr %ref, align 4
  %isnull = icmp eq i32 %id, -1
  br i1 %isnull, label %emptyslot, label %notnull

emptyslot:
  %nl = call i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val)
  %nlbad = icmp eq i32 %nl, -1
  br i1 %nlbad, label %err.oom, label %emptyset, !prof !1

emptyset:
  store i32 %nl, ptr %ref, align 4
  br label %inc

err.oom:
  ret i32 2

notnull:
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %atleaf, label %atinner

atleaf:
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %eq = call i1 @art_keyeq(ptr %lkey, i64 %llen, ptr %key, i64 %klen)
  br i1 %eq, label %overwrite, label %splitleaf

overwrite:
  %ovp = getelementptr inbounds nuw i8, ptr %lnode, i64 8
  store i64 %val, ptr %ovp, align 8
  ret i32 0

splitleaf:
  %rc.sl = call i32 @art_split_leaf(ptr %t, ptr %ref, i32 %id, ptr %key, i64 %klen, i64 %val, i64 %depth)
  %sl.bad = icmp ne i32 %rc.sl, 0
  br i1 %sl.bad, label %err.oom, label %inc, !prof !1

atinner:
  %m = call i64 @art_prefix_match(ptr %t, i32 %id, ptr %key, i64 %klen, i64 %depth, ptr %stslot)
  %st = load i32, ptr %stslot, align 4
  %ismis = icmp eq i32 %st, 1
  br i1 %ismis, label %splitpfx, label %chkexh

splitpfx:
  %rc.sp = call i32 @art_split_prefix(ptr %t, ptr %ref, i32 %id, ptr %key, i64 %klen, i64 %val, i64 %depth, i64 %m, i1 false)
  %sp.bad = icmp ne i32 %rc.sp, 0
  br i1 %sp.bad, label %err.oom, label %inc, !prof !1

chkexh:
  %isexh = icmp eq i32 %st, 2
  br i1 %isexh, label %splitterm, label %fullmatch

splitterm:
  %rc.stm = call i32 @art_split_prefix(ptr %t, ptr %ref, i32 %id, ptr %key, i64 %klen, i64 %val, i64 %depth, i64 %m, i1 true)
  %stm.bad = icmp ne i32 %rc.stm, 0
  br i1 %stm.bad, label %err.oom, label %inc, !prof !1

fullmatch:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %iplp = getelementptr inbounds nuw i8, ptr %inode, i64 4
  %ipl32 = load i32, ptr %iplp, align 4
  %ipl = zext i32 %ipl32 to i64
  %depth2 = add i64 %depth, %ipl
  %atend = icmp eq i64 %depth2, %klen
  br i1 %atend, label %storeterm, label %descend

storeterm:
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  %hasterm = icmp ne i32 %term, -1
  br i1 %hasterm, label %termover, label %termnew

termnew:
  %ntl = call i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val)
  %ntl.bad = icmp eq i32 %ntl, -1
  br i1 %ntl.bad, label %err.oom, label %termset, !prof !1

termset:
  %inode2 = call ptr @art_nptr(ptr %t, i32 %id)
  %tlp2 = getelementptr inbounds nuw i8, ptr %inode2, i64 24
  store i32 %ntl, ptr %tlp2, align 4
  br label %inc

termover:
  %tnode = call ptr @art_nptr(ptr %t, i32 %term)
  %tvp = getelementptr inbounds nuw i8, ptr %tnode, i64 8
  store i64 %val, ptr %tvp, align 8
  ret i32 0

descend:
  %bp = getelementptr inbounds nuw i8, ptr %key, i64 %depth2
  %b = load i8, ptr %bp, align 1
  %slot = call ptr @art_findchild(ptr %t, i32 %id, i8 %b)
  %slotnull = icmp eq ptr %slot, null
  %depth3 = add i64 %depth2, 1
  br i1 %slotnull, label %addnew, label %godown

godown:
  %nextref = phi ptr [ %slot, %descend ]
  br label %loop

addnew:
  %anl = call i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val)
  %anl.bad = icmp eq i32 %anl, -1
  br i1 %anl.bad, label %err.oom, label %addchild, !prof !1

addchild:
  %rc.ac = call i32 @art_add_child(ptr %t, ptr %ref, i32 %id, i8 %b, i32 %anl)
  %ac.bad = icmp ne i32 %rc.ac, 0
  br i1 %ac.bad, label %err.oom, label %inc, !prof !1

inc:
  %cnt = load i64, ptr %t, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %t, align 8
  ret i32 0
}

; split an existing leaf vs a new key. returns 0 ok / 2 OOM.
define internal i32 @art_split_leaf(ptr %t, ptr %ref, i32 %oldleaf, ptr %key, i64 %klen, i64 %val, i64 %depth) #1 {
entry:
  %lnode = call ptr @art_nptr(ptr %t, i32 %oldleaf)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp0 = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab0 = load ptr, ptr %arbp0, align 8
  %k1a = getelementptr inbounds nuw i8, ptr %ab0, i64 %loff
  %cpl = call i64 @art_cpl(ptr %k1a, i64 %llen, ptr %key, i64 %klen, i64 %depth)
  %p = add i64 %depth, %cpl
  ; new leaf (appends; may move arena)
  %nl = call i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val)
  %nlbad = icmp eq i32 %nl, -1
  br i1 %nlbad, label %fail, label %n4, !prof !1

n4:
  %n4id = call i32 @art_alloc_node(ptr %t, i32 1)
  %n4bad = icmp eq i32 %n4id, -1
  br i1 %n4bad, label %fail, label %build, !prof !1

build:
  ; recompute arena-based old key pointer (arena may have moved)
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %k1 = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %n4node = call ptr @art_nptr(ptr %t, i32 %n4id)
  ; prefix = cpl bytes from k1[depth..]
  %cpl32 = trunc i64 %cpl to i32
  %n4plp = getelementptr inbounds nuw i8, ptr %n4node, i64 4
  store i32 %cpl32, ptr %n4plp, align 4
  %copy = call i64 @llvm.umin.i64(i64 %cpl, i64 16)
  %psrc = getelementptr inbounds nuw i8, ptr %k1, i64 %depth
  %pdst = getelementptr inbounds nuw i8, ptr %n4node, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %pdst, ptr %psrc, i64 %copy, i1 false)
  store i32 %n4id, ptr %ref, align 4
  %oldExh = icmp eq i64 %p, %llen
  %newExh = icmp eq i64 %p, %klen
  br i1 %oldExh, label %oldterm, label %chknew

oldterm:
  ; old key ends here -> term_leaf = oldleaf, child key[p] -> newleaf
  %tlp.o = getelementptr inbounds nuw i8, ptr %n4node, i64 24
  store i32 %oldleaf, ptr %tlp.o, align 4
  %bp.o = getelementptr inbounds nuw i8, ptr %key, i64 %p
  %nb.o = load i8, ptr %bp.o, align 1
  %rc.o = call i32 @art_add_child(ptr %t, ptr %ref, i32 %n4id, i8 %nb.o, i32 %nl)
  ret i32 %rc.o

chknew:
  br i1 %newExh, label %newterm, label %both

newterm:
  ; new key ends here -> term_leaf = newleaf, child k1[p] -> oldleaf
  %tlp.n = getelementptr inbounds nuw i8, ptr %n4node, i64 24
  store i32 %nl, ptr %tlp.n, align 4
  %bp.n = getelementptr inbounds nuw i8, ptr %k1, i64 %p
  %ob.n = load i8, ptr %bp.n, align 1
  %rc.n = call i32 @art_add_child(ptr %t, ptr %ref, i32 %n4id, i8 %ob.n, i32 %oldleaf)
  ret i32 %rc.n

both:
  %bp.ob = getelementptr inbounds nuw i8, ptr %k1, i64 %p
  %ob = load i8, ptr %bp.ob, align 1
  %rc.b1 = call i32 @art_add_child(ptr %t, ptr %ref, i32 %n4id, i8 %ob, i32 %oldleaf)
  %b1bad = icmp ne i32 %rc.b1, 0
  br i1 %b1bad, label %retb1, label %both2

both2:
  %id2 = load i32, ptr %ref, align 4
  %bp.nb = getelementptr inbounds nuw i8, ptr %key, i64 %p
  %nb = load i8, ptr %bp.nb, align 1
  %rc.b2 = call i32 @art_add_child(ptr %t, ptr %ref, i32 %id2, i8 %nb, i32 %nl)
  ret i32 %rc.b2

retb1:
  ret i32 %rc.b1

fail:
  ret i32 2
}

; split a node's prefix at mismatch/exhaust position %m. %isterm=true => the new
; key ends within the prefix (becomes term_leaf). returns 0 ok / 2 OOM.
define internal i32 @art_split_prefix(ptr %t, ptr %ref, i32 %oldid, ptr %key, i64 %klen, i64 %val, i64 %depth, i64 %m, i1 %isterm) #1 {
entry:
  %nl = call i32 @art_alloc_leaf(ptr %t, ptr %key, i64 %klen, i64 %val)
  %nlbad = icmp eq i32 %nl, -1
  br i1 %nlbad, label %fail, label %n4, !prof !1

n4:
  %n4id = call i32 @art_alloc_node(ptr %t, i32 1)
  %n4bad = icmp eq i32 %n4id, -1
  br i1 %n4bad, label %fail, label %build, !prof !1

build:
  %mlid = call i32 @art_minleaf(ptr %t, i32 %oldid)
  %mlnode = call ptr @art_nptr(ptr %t, i32 %mlid)
  %moffp = getelementptr inbounds nuw i8, ptr %mlnode, i64 16
  %moff = load i64, ptr %moffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %mk = getelementptr inbounds nuw i8, ptr %ab, i64 %moff
  %oldnode = call ptr @art_nptr(ptr %t, i32 %oldid)
  %oplp = getelementptr inbounds nuw i8, ptr %oldnode, i64 4
  %opl32 = load i32, ptr %oplp, align 4
  %opl = zext i32 %opl32 to i64
  ; new N4 prefix = m bytes from mk[depth..]
  %n4node = call ptr @art_nptr(ptr %t, i32 %n4id)
  %m32 = trunc i64 %m to i32
  %n4plp = getelementptr inbounds nuw i8, ptr %n4node, i64 4
  store i32 %m32, ptr %n4plp, align 4
  %copyn = call i64 @llvm.umin.i64(i64 %m, i64 16)
  %nsrc = getelementptr inbounds nuw i8, ptr %mk, i64 %depth
  %ndst = getelementptr inbounds nuw i8, ptr %n4node, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %ndst, ptr %nsrc, i64 %copyn, i1 false)
  ; old node discriminator byte = mk[depth+m]
  %obpos = add i64 %depth, %m
  %obp = getelementptr inbounds nuw i8, ptr %mk, i64 %obpos
  %oldbyte = load i8, ptr %obp, align 1
  ; shorten old prefix by (m+1)
  %rem0 = sub i64 %opl, %m
  %newoldlen = sub i64 %rem0, 1
  %newoldlen32 = trunc i64 %newoldlen to i32
  store i32 %newoldlen32, ptr %oplp, align 4
  %copyo = call i64 @llvm.umin.i64(i64 %newoldlen, i64 16)
  %osrcpos = add i64 %obpos, 1
  %osrc = getelementptr inbounds nuw i8, ptr %mk, i64 %osrcpos
  %odst = getelementptr inbounds nuw i8, ptr %oldnode, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %odst, ptr %osrc, i64 %copyo, i1 false)
  store i32 %n4id, ptr %ref, align 4
  br i1 %isterm, label %doterm, label %dobranch

doterm:
  %tlp = getelementptr inbounds nuw i8, ptr %n4node, i64 24
  store i32 %nl, ptr %tlp, align 4
  %rc.t = call i32 @art_add_child(ptr %t, ptr %ref, i32 %n4id, i8 %oldbyte, i32 %oldid)
  ret i32 %rc.t

dobranch:
  %rc.o = call i32 @art_add_child(ptr %t, ptr %ref, i32 %n4id, i8 %oldbyte, i32 %oldid)
  %obad = icmp ne i32 %rc.o, 0
  br i1 %obad, label %reto, label %dobranch2

dobranch2:
  %id2 = load i32, ptr %ref, align 4
  %nbpos = add i64 %depth, %m
  %nbp = getelementptr inbounds nuw i8, ptr %key, i64 %nbpos
  %newbyte = load i8, ptr %nbp, align 1
  %rc.n = call i32 @art_add_child(ptr %t, ptr %ref, i32 %id2, i8 %newbyte, i32 %nl)
  ret i32 %rc.n

reto:
  ret i32 %rc.o

fail:
  ret i32 2
}

; ===========================================================================
; get / contains
; ===========================================================================
define i32 @universe_ds_art_get(ptr %t, ptr %key, i64 %klen, ptr %out) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %tn, %kn
  br i1 %bad, label %err.null, label %setup, !prof !1

err.null:
  ret i32 1

setup:
  %stslot = alloca i32, align 4
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %root = load i32, ptr %rootp, align 4
  br label %loop

loop:
  %id = phi i32 [ %root, %setup ], [ %nid, %cont ]
  %depth = phi i64 [ 0, %setup ], [ %depth3, %cont ]
  %isnull = icmp eq i32 %id, -1
  br i1 %isnull, label %miss, label %notnull

notnull:
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %atleaf, label %atinner

atleaf:
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %eq = call i1 @art_keyeq(ptr %lkey, i64 %llen, ptr %key, i64 %klen)
  br i1 %eq, label %hit, label %miss

hit:
  %outn = icmp eq ptr %out, null
  br i1 %outn, label %ret0, label %store

store:
  %vp = getelementptr inbounds nuw i8, ptr %lnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %out, align 8
  br label %ret0

ret0:
  ret i32 0

atinner:
  %m = call i64 @art_prefix_match(ptr %t, i32 %id, ptr %key, i64 %klen, i64 %depth, ptr %stslot)
  %st = load i32, ptr %stslot, align 4
  %nofull = icmp ne i32 %st, 0
  br i1 %nofull, label %miss, label %pfxok

pfxok:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %iplp = getelementptr inbounds nuw i8, ptr %inode, i64 4
  %ipl32 = load i32, ptr %iplp, align 4
  %ipl = zext i32 %ipl32 to i64
  %depth2 = add i64 %depth, %ipl
  %atend = icmp eq i64 %depth2, %klen
  br i1 %atend, label %useterm, label %descend

useterm:
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  %termnull = icmp eq i32 %term, -1
  br i1 %termnull, label %miss, label %termleaf

termleaf:
  %tnode = call ptr @art_nptr(ptr %t, i32 %term)
  %tvpn = icmp eq ptr %out, null
  br i1 %tvpn, label %ret0b, label %tstore

tstore:
  %tvp = getelementptr inbounds nuw i8, ptr %tnode, i64 8
  %tv = load i64, ptr %tvp, align 8
  store i64 %tv, ptr %out, align 8
  br label %ret0b

ret0b:
  ret i32 0

descend:
  %bp = getelementptr inbounds nuw i8, ptr %key, i64 %depth2
  %b = load i8, ptr %bp, align 1
  %slot = call ptr @art_findchild(ptr %t, i32 %id, i8 %b)
  %slotnull = icmp eq ptr %slot, null
  %depth3 = add i64 %depth2, 1
  br i1 %slotnull, label %miss, label %cont

cont:
  %nid = load i32, ptr %slot, align 4
  br label %loop

miss:
  ret i32 5
}

define i32 @universe_ds_art_contains(ptr %t, ptr %key, i64 %klen) local_unnamed_addr #5 {
entry:
  %r = call i32 @universe_ds_art_get(ptr %t, ptr %key, i64 %klen, ptr null)
  %ok = icmp eq i32 %r, 0
  %z = zext i1 %ok to i32
  ret i32 %z
}

; ===========================================================================
; delete
; ===========================================================================
define i32 @universe_ds_art_delete(ptr %t, ptr %key, i64 %klen) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  %kn = icmp eq ptr %key, null
  %bad = or i1 %tn, %kn
  br i1 %bad, label %err.null, label %go, !prof !1

err.null:
  ret i32 1

go:
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %r = call i32 @art_delete_rec(ptr %t, ptr %rootp, ptr %key, i64 %klen, i64 0)
  %ok = icmp eq i32 %r, 0
  br i1 %ok, label %dec, label %ret

dec:
  %cnt = load i64, ptr %t, align 8
  %cnt1 = sub i64 %cnt, 1
  store i64 %cnt1, ptr %t, align 8
  br label %ret

ret:
  ret i32 %r
}

; recursive delete. returns 0 found / 5 not found. mutates *ref.
define internal i32 @art_delete_rec(ptr %t, ptr %ref, ptr %key, i64 %klen, i64 %depth) #6 {
entry:
  %stslot = alloca i32, align 4
  %id = load i32, ptr %ref, align 4
  %isnull = icmp eq i32 %id, -1
  br i1 %isnull, label %notfound, label %chk

chk:
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %leaf, label %inner

leaf:
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %eq = call i1 @art_keyeq(ptr %lkey, i64 %llen, ptr %key, i64 %klen)
  br i1 %eq, label %dodel, label %notfound

dodel:
  call void @art_pool_free(ptr %t, i32 %id)
  store i32 -1, ptr %ref, align 4
  ret i32 0

inner:
  %m = call i64 @art_prefix_match(ptr %t, i32 %id, ptr %key, i64 %klen, i64 %depth, ptr %stslot)
  %st = load i32, ptr %stslot, align 4
  %nofull = icmp ne i32 %st, 0
  br i1 %nofull, label %notfound, label %pfxok

pfxok:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %iplp = getelementptr inbounds nuw i8, ptr %inode, i64 4
  %ipl32 = load i32, ptr %iplp, align 4
  %ipl = zext i32 %ipl32 to i64
  %depth2 = add i64 %depth, %ipl
  %atend = icmp eq i64 %depth2, %klen
  br i1 %atend, label %termdel, label %descend

termdel:
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  %termnull = icmp eq i32 %term, -1
  br i1 %termnull, label %notfound, label %dotermdel

dotermdel:
  call void @art_pool_free(ptr %t, i32 %term)
  store i32 -1, ptr %tlp, align 4
  call void @art_collapse(ptr %t, ptr %ref, i32 %id, i64 %depth)
  ret i32 0

descend:
  %bp = getelementptr inbounds nuw i8, ptr %key, i64 %depth2
  %b = load i8, ptr %bp, align 1
  %slot = call ptr @art_findchild(ptr %t, i32 %id, i8 %b)
  %slotnull = icmp eq ptr %slot, null
  br i1 %slotnull, label %notfound, label %rec

rec:
  %depth3 = add i64 %depth2, 1
  %rc = call i32 @art_delete_rec(ptr %t, ptr %slot, ptr %key, i64 %klen, i64 %depth3)
  %failed = icmp ne i32 %rc, 0
  br i1 %failed, label %retrc, label %chkchild

chkchild:
  %cid = load i32, ptr %slot, align 4
  %cnull = icmp eq i32 %cid, -1
  br i1 %cnull, label %removed, label %done0

removed:
  call void @art_remove_child(ptr %t, i32 %id, i8 %b)
  call void @art_collapse(ptr %t, ptr %ref, i32 %id, i64 %depth)
  ret i32 0

done0:
  ret i32 0

retrc:
  ret i32 %rc

notfound:
  ret i32 5
}

; ===========================================================================
; count / size
; ===========================================================================
define i64 @universe_ds_art_count(ptr %t) local_unnamed_addr #7 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %z, label %go, !prof !1

z:
  ret i64 0

go:
  %c = load i64, ptr %t, align 8
  ret i64 %c
}

define i64 @universe_ds_art_size(ptr %t) local_unnamed_addr #7 {
entry:
  %r = call i64 @universe_ds_art_count(ptr %t)
  ret i64 %r
}

; ===========================================================================
; ordered iteration / prefix scan
; ===========================================================================
; emit all keys under %id in lexicographic order. returns 1 if the callback
; asked to stop. *cntp accumulates the number of emitted leaves.
define internal i1 @art_emit(ptr %t, i32 %id, ptr %cb, ptr %ctx, ptr %cntp) #6 {
entry:
  %isnull = icmp eq i32 %id, -1
  br i1 %isnull, label %nostop, label %chk

chk:
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %leaf, label %inner

leaf:
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %vp = getelementptr inbounds nuw i8, ptr %lnode, i64 8
  %v = load i64, ptr %vp, align 8
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %cnt = load i64, ptr %cntp, align 8
  %cnt1 = add i64 %cnt, 1
  store i64 %cnt1, ptr %cntp, align 8
  %r = call i32 %cb(ptr %ctx, ptr %lkey, i64 %llen, i64 %v)
  %stop = icmp ne i32 %r, 0
  ret i1 %stop

inner:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  %hasterm = icmp ne i32 %term, -1
  br i1 %hasterm, label %emitterm, label %children

emitterm:
  %ts = call i1 @art_emit(ptr %t, i32 %term, ptr %cb, ptr %ctx, ptr %cntp)
  br i1 %ts, label %stopped, label %children

children:
  %num = load i32, ptr %inode, align 4
  %num64 = zext i32 %num to i64
  %issmall = icmp ult i32 %kind, 3
  br i1 %issmall, label %small, label %big

small:
  %isk4.e = icmp eq i32 %kind, 1
  %choff = select i1 %isk4.e, i64 36, i64 48
  %child = getelementptr inbounds nuw i8, ptr %inode, i64 %choff
  br label %sloop

sloop:
  %j = phi i64 [ 0, %small ], [ %jn, %scont ]
  %sdone = icmp uge i64 %j, %num64
  br i1 %sdone, label %nostop, label %sbody

sbody:
  %sp = getelementptr inbounds nuw i32, ptr %child, i64 %j
  %cidv = load i32, ptr %sp, align 4
  %ss = call i1 @art_emit(ptr %t, i32 %cidv, ptr %cb, ptr %ctx, ptr %cntp)
  br i1 %ss, label %stopped, label %scont

scont:
  %jn = add nuw i64 %j, 1
  br label %sloop

big:
  %is48 = icmp eq i32 %kind, 3
  br i1 %is48, label %big48, label %big256

big48:
  %idxb = getelementptr inbounds nuw i8, ptr %inode, i64 32
  %child48 = getelementptr inbounds nuw i8, ptr %inode, i64 288
  br label %loop48

loop48:
  %b = phi i64 [ 0, %big48 ], [ %bn, %cont48 ]
  %ixp = getelementptr inbounds nuw i8, ptr %idxb, i64 %b
  %iv = load i8, ptr %ixp, align 1
  %present = icmp ne i8 %iv, 0
  br i1 %present, label %emit48, label %cont48

emit48:
  %iv64 = zext i8 %iv to i64
  %si = sub nuw i64 %iv64, 1
  %spx = getelementptr inbounds nuw i32, ptr %child48, i64 %si
  %c48 = load i32, ptr %spx, align 4
  %s48 = call i1 @art_emit(ptr %t, i32 %c48, ptr %cb, ptr %ctx, ptr %cntp)
  br i1 %s48, label %stopped, label %cont48

cont48:
  %bn = add nuw i64 %b, 1
  %more48 = icmp ult i64 %bn, 256
  br i1 %more48, label %loop48, label %nostop

big256:
  %child256 = getelementptr inbounds nuw i8, ptr %inode, i64 32
  br label %loop256

loop256:
  %k = phi i64 [ 0, %big256 ], [ %kn, %cont256 ]
  %cp = getelementptr inbounds nuw i32, ptr %child256, i64 %k
  %cv = load i32, ptr %cp, align 4
  %present256 = icmp ne i32 %cv, -1
  br i1 %present256, label %emit256, label %cont256

emit256:
  %s256 = call i1 @art_emit(ptr %t, i32 %cv, ptr %cb, ptr %ctx, ptr %cntp)
  br i1 %s256, label %stopped, label %cont256

cont256:
  %kn = add nuw i64 %k, 1
  %more256 = icmp ult i64 %kn, 256
  br i1 %more256, label %loop256, label %nostop

stopped:
  ret i1 true

nostop:
  ret i1 false
}

define i64 @universe_ds_art_iterate(ptr %t, ptr %cb, ptr %ctx) local_unnamed_addr #6 {
entry:
  %tn = icmp eq ptr %t, null
  %cbn = icmp eq ptr %cb, null
  %bad = or i1 %tn, %cbn
  br i1 %bad, label %z, label %go, !prof !1

z:
  ret i64 0

go:
  %cslot = alloca i64, align 8
  store i64 0, ptr %cslot, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %root = load i32, ptr %rootp, align 4
  %s = call i1 @art_emit(ptr %t, i32 %root, ptr %cb, ptr %ctx, ptr %cslot)
  %r = load i64, ptr %cslot, align 8
  ret i64 %r
}

define i64 @universe_ds_art_prefix_scan(ptr %t, ptr %pfx, i64 %plen, ptr %cb, ptr %ctx) local_unnamed_addr #6 {
entry:
  %tn = icmp eq ptr %t, null
  %cbn = icmp eq ptr %cb, null
  %bad = or i1 %tn, %cbn
  br i1 %bad, label %z, label %setup, !prof !1

z:
  ret i64 0

setup:
  %cslot = alloca i64, align 8
  %stslot = alloca i32, align 4
  store i64 0, ptr %cslot, align 8
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %root = load i32, ptr %rootp, align 4
  br label %loop

loop:
  %id = phi i32 [ %root, %setup ], [ %nid, %descend ]
  %depth = phi i64 [ 0, %setup ], [ %depth3, %descend ]
  %isnull = icmp eq i32 %id, -1
  br i1 %isnull, label %z, label %notnull

notnull:
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %atleaf, label %atinner

atleaf:
  ; leaf matches iff its full key begins with pfx
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %short = icmp ult i64 %llen, %plen
  br i1 %short, label %z, label %lcmp

lcmp:
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %pmatch = call i64 @art_cpl(ptr %lkey, i64 %llen, ptr %pfx, i64 %plen, i64 %depth)
  %consumed = add i64 %depth, %pmatch
  %ok = icmp uge i64 %consumed, %plen
  br i1 %ok, label %emitleaf, label %z

emitleaf:
  %s0 = call i1 @art_emit(ptr %t, i32 %id, ptr %cb, ptr %ctx, ptr %cslot)
  br label %fin

atinner:
  ; how much of the remaining prefix does this node cover?
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %iplp = getelementptr inbounds nuw i8, ptr %inode, i64 4
  %ipl32 = load i32, ptr %iplp, align 4
  %ipl = zext i32 %ipl32 to i64
  %remain = sub i64 %plen, %depth
  %within = icmp ule i64 %remain, %ipl
  %cmplen = call i64 @llvm.umin.i64(i64 %ipl, i64 %remain)
  ; verify the overlapping prefix bytes match
  %m = call i64 @art_prefix_match(ptr %t, i32 %id, ptr %pfx, i64 %plen, i64 %depth, ptr %stslot)
  %stv = load i32, ptr %stslot, align 4
  ; MISMATCH (1) means a byte differed -> no keys under this prefix
  %mism = icmp eq i32 %stv, 1
  br i1 %mism, label %z, label %chkwithin

chkwithin:
  br i1 %within, label %emitall, label %godeeper

emitall:
  ; prefix ends inside (or at end of) this node's compressed prefix and all
  ; overlapping bytes matched -> entire subtree qualifies
  %sa = call i1 @art_emit(ptr %t, i32 %id, ptr %cb, ptr %ctx, ptr %cslot)
  br label %fin

godeeper:
  ; status must be FULL here (prefix longer than node prefix, all matched)
  %depth2 = add i64 %depth, %ipl
  %bp = getelementptr inbounds nuw i8, ptr %pfx, i64 %depth2
  %b = load i8, ptr %bp, align 1
  %slot = call ptr @art_findchild(ptr %t, i32 %id, i8 %b)
  %slotnull = icmp eq ptr %slot, null
  %depth3 = add i64 %depth2, 1
  br i1 %slotnull, label %z, label %descend

descend:
  %nid = load i32, ptr %slot, align 4
  br label %loop

fin:
  %r = load i64, ptr %cslot, align 8
  ret i64 %r
}

; ===========================================================================
; min / max (zero-copy key view into the arena)
; ===========================================================================
define i32 @universe_ds_art_min(ptr %t, ptr %outkey, ptr %outlen, ptr %outval) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %err.null, label %setup, !prof !1

err.null:
  ret i32 1

setup:
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %root = load i32, ptr %rootp, align 4
  %empty = icmp eq i32 %root, -1
  br i1 %empty, label %isempty, label %loop, !prof !1

isempty:
  ret i32 4

loop:
  %id = phi i32 [ %root, %setup ], [ %nid, %step ]
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %atleaf, label %step

step:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  %hasterm = icmp ne i32 %term, -1
  %fc = call i32 @art_firstchild(ptr %t, i32 %id)
  %nid = select i1 %hasterm, i32 %term, i32 %fc
  br label %loop

atleaf:
  call void @art_emitkey(ptr %t, i32 %id, ptr %outkey, ptr %outlen, ptr %outval)
  ret i32 0
}

define i32 @universe_ds_art_max(ptr %t, ptr %outkey, ptr %outlen, ptr %outval) local_unnamed_addr #5 {
entry:
  %tn = icmp eq ptr %t, null
  br i1 %tn, label %err.null, label %setup, !prof !1

err.null:
  ret i32 1

setup:
  %rootp = getelementptr inbounds nuw i8, ptr %t, i64 8
  %root = load i32, ptr %rootp, align 4
  %empty = icmp eq i32 %root, -1
  br i1 %empty, label %isempty, label %loop, !prof !1

isempty:
  ret i32 4

loop:
  %id = phi i32 [ %root, %setup ], [ %nid, %step2 ]
  %kind = lshr i32 %id, 29
  %isleaf = icmp eq i32 %kind, 0
  br i1 %isleaf, label %atleaf, label %step

step:
  %inode = call ptr @art_nptr(ptr %t, i32 %id)
  %num = load i32, ptr %inode, align 4
  %haskids = icmp ne i32 %num, 0
  br i1 %haskids, label %lastkid, label %useterm

lastkid:
  %lc = call i32 @art_lastchild(ptr %t, i32 %id)
  br label %step2

useterm:
  %tlp = getelementptr inbounds nuw i8, ptr %inode, i64 24
  %term = load i32, ptr %tlp, align 4
  br label %step2

step2:
  %nid = phi i32 [ %lc, %lastkid ], [ %term, %useterm ]
  br label %loop

atleaf:
  call void @art_emitkey(ptr %t, i32 %id, ptr %outkey, ptr %outlen, ptr %outval)
  ret i32 0
}

; write a leaf's key view + len + val into the caller's out slots (any may null)
define internal void @art_emitkey(ptr %t, i32 %id, ptr %outkey, ptr %outlen, ptr %outval) #1 {
entry:
  %lnode = call ptr @art_nptr(ptr %t, i32 %id)
  %llenp = getelementptr inbounds nuw i8, ptr %lnode, i64 4
  %llen32 = load i32, ptr %llenp, align 4
  %llen = zext i32 %llen32 to i64
  %loffp = getelementptr inbounds nuw i8, ptr %lnode, i64 16
  %loff = load i64, ptr %loffp, align 8
  %arbp = getelementptr inbounds nuw i8, ptr %t, i64 16
  %ab = load ptr, ptr %arbp, align 8
  %lkey = getelementptr inbounds nuw i8, ptr %ab, i64 %loff
  %okn = icmp eq ptr %outkey, null
  br i1 %okn, label %chklen, label %wk

wk:
  store ptr %lkey, ptr %outkey, align 8
  br label %chklen

chklen:
  %oln = icmp eq ptr %outlen, null
  br i1 %oln, label %chkval, label %wl

wl:
  store i64 %llen, ptr %outlen, align 8
  br label %chkval

chkval:
  %ovn = icmp eq ptr %outval, null
  br i1 %ovn, label %done, label %wv

wv:
  %vp = getelementptr inbounds nuw i8, ptr %lnode, i64 8
  %v = load i64, ptr %vp, align 8
  store i64 %v, ptr %outval, align 8
  br label %done

done:
  ret void
}

declare void @llvm.memmove.p0.p0.i64(ptr captures(none), ptr captures(none), i64, i1 immarg)

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #4 = { nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #5 = { nounwind }
attributes #6 = { nounwind }
attributes #7 = { nounwind willreturn norecurse nosync memory(argmem: read) }

!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
