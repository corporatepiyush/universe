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

; Open-addressing hash map, i64 keys, caller-fixed value size, SIMD tag-group
; probing (the "_swiss" flavor).
;
; WHEN TO CHOOSE THIS VARIANT:
;   * Keys are i64 (or fit i64); values are POD of a fixed size chosen at
;     create time. Read-heavy or mixed workloads where get() latency matters.
;   * The tag-group probe rejects 16 slots per vector compare, so a miss or a
;     hit touches ~1 cache line of control bytes plus at most a couple of key
;     loads — far fewer branches than a per-slot linear scan. Prefer this over
;     a chained/linear map when the hot op is get() and the table is large
;     enough that probe length matters. Prefer a simple linear-probe map only
;     for tiny fixed tables where the vector setup does not pay.
;
; DESIGN (first principles; no external layout copied):
;   * ONE allocation for the table storage: a control-byte array immediately
;     followed by the key array and the value array (SoA), all in one block.
;     A separate, STABLE 64-byte header holds metadata + the current table
;     pointer, so put() may reallocate on growth without invalidating the
;     caller's handle.
;   * Control bytes (one per slot): 0x80 EMPTY, 0xFE TOMBSTONE, else a 7-bit
;     tag (high bit clear) = low 7 bits of the hash. Full/empty/tombstone are
;     distinguished by the high bit; a tag byte can never equal EMPTY/TOMB.
;   * Capacity = power of two AND a multiple of the group width 16, so the
;     table splits into cap/16 aligned groups. Probing is GROUP-ALIGNED: a
;     group load reads exactly ctrl[g*16 .. g*16+16) which is always in
;     bounds, so NO cloned/mirror control bytes are needed.
;   * Hash = splitmix64(key). h1 = hash>>7 selects the starting group
;     (h1 & (cap/16 - 1)); h2 = hash & 0x7f is the tag.
;   * HOT PROBE: load 16 control bytes as <16 x i8>; `icmp eq` vs the splatted
;     tag; `bitcast <16 x i1> -> i16` gives a movemask; iterate set bits with
;     llvm.cttz, comparing only the candidate full keys. A separate
;     `icmp eq` vs EMPTY (splat 0x80) tells us to stop (key absent). If the
;     group has neither the key nor an empty, advance by TRIANGULAR numbers:
;     group = (group + (++probe)) & (cap/16 - 1). For a power-of-two group
;     count the triangular sequence visits every group exactly once, so the
;     probe always terminates (an empty exists because load factor < 1).
;   * Load factor bound 7/8: on inserting a new key, if count+tombstones+1
;     would exceed cap*7/8 we rehash. RECLAIM POLICY: if the live count would
;     fit in half the capacity (count+1 <= cap/2) the excess is tombstones, so
;     rehash at the SAME capacity to drop them; otherwise grow 2x. A fresh
;     table has no tombstones, restoring probe locality.
;   * Insertion reuses the first EMPTY-or-TOMBSTONE slot in the probe order
;     (icmp slt vs 0 = high-bit-set lanes), reclaiming a tombstone when one is
;     hit before an empty.
;
; Header (64 B, one cache line):
;   table@0(ptr) count@8 capacity@16 maskg@24(=cap/16-1) tombstones@32 vsz@40
; Table block: ctrl[cap] | keys[cap]*i64 | values[cap]*vsz  (posix_memalign 64)
;
; API (error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 5 NOT_FOUND):
;   ptr  universe_ds_hashmap_swiss_create(i64 val_size, i64 initial_cap)
;   i32  universe_ds_hashmap_swiss_put(ptr m, i64 key, ptr val)  ; ins/overwrite
;   i32  universe_ds_hashmap_swiss_get(ptr m, i64 key, ptr out)
;   i32  universe_ds_hashmap_swiss_contains(ptr m, i64 key)
;   i32  universe_ds_hashmap_swiss_remove(ptr m, i64 key)
;   i64  universe_ds_hashmap_swiss_len(ptr m)
;   i64  universe_ds_hashmap_swiss_capacity(ptr m)
;   void universe_ds_hashmap_swiss_destroy(ptr m)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare i16 @llvm.cttz.i16(i16, i1 immarg)

; splitmix64 finalizer — excellent avalanche, no division.
define internal i64 @swm_hash(i64 %x) #3 {
entry:
  %a1 = lshr i64 %x, 30
  %a2 = xor i64 %a1, %x
  %a3 = mul i64 %a2, -4658895280553007687      ; 0xbf58476d1ce4e5b9
  %a4 = lshr i64 %a3, 27
  %a5 = xor i64 %a4, %a3
  %a6 = mul i64 %a5, -7723592293110705685      ; 0x94d049bb133111eb
  %a7 = lshr i64 %a6, 31
  %a8 = xor i64 %a7, %a6
  ret i64 %a8
}

; Return {ctrl, keys, values} for the current table block.
define internal { ptr, ptr, ptr } @swm_ptrs(ptr %m) #0 {
entry:
  %table = load ptr, ptr %m, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %cap = load i64, ptr %cap.p, align 8
  %keys = getelementptr inbounds nuw i8, ptr %table, i64 %cap
  %cap8 = shl nuw i64 %cap, 3
  %values = getelementptr inbounds nuw i8, ptr %keys, i64 %cap8
  %r0 = insertvalue { ptr, ptr, ptr } poison, ptr %table, 0
  %r1 = insertvalue { ptr, ptr, ptr } %r0, ptr %keys, 1
  %r2 = insertvalue { ptr, ptr, ptr } %r1, ptr %values, 2
  ret { ptr, ptr, ptr } %r2
}

; Find the slot holding %key, or -1. Group-aligned SIMD tag probe.
define internal i64 @swm_find(ptr %ctrl, ptr %keys, i64 %key, i64 %hash, i64 %maskg) #0 {
entry:
  %tag64 = and i64 %hash, 127
  %tag = trunc i64 %tag64 to i8
  %tv0 = insertelement <16 x i8> poison, i8 %tag, i64 0
  %tagv = shufflevector <16 x i8> %tv0, <16 x i8> poison, <16 x i32> zeroinitializer
  %h1 = lshr i64 %hash, 7
  %grp0 = and i64 %h1, %maskg
  br label %grp.loop

grp.loop:
  %grp = phi i64 [ %grp0, %entry ], [ %grp.next, %grp.cont ]
  %probe = phi i64 [ 0, %entry ], [ %probe.next, %grp.cont ]
  %base = shl nuw i64 %grp, 4
  %cptr = getelementptr inbounds nuw i8, ptr %ctrl, i64 %base
  %g = load <16 x i8>, ptr %cptr, align 16
  %meq = icmp eq <16 x i8> %g, %tagv
  %mm = bitcast <16 x i1> %meq to i16
  br label %match.loop

match.loop:
  %mask = phi i16 [ %mm, %grp.loop ], [ %mask.clr, %match.cont ]
  %nomatch = icmp eq i16 %mask, 0
  br i1 %nomatch, label %check.empty, label %match.body

match.body:
  %bit = call i16 @llvm.cttz.i16(i16 %mask, i1 true)
  %bit64 = zext i16 %bit to i64
  %cand = add nuw i64 %base, %bit64
  %kptr = getelementptr inbounds i64, ptr %keys, i64 %cand
  %k = load i64, ptr %kptr, align 8
  %hit = icmp eq i64 %k, %key
  br i1 %hit, label %found, label %match.cont, !prof !1

match.cont:
  %m1 = sub i16 %mask, 1
  %mask.clr = and i16 %mask, %m1
  br label %match.loop

check.empty:
  %emq = icmp eq <16 x i8> %g, splat (i8 -128)
  %em = bitcast <16 x i1> %emq to i16
  %hasempty = icmp ne i16 %em, 0
  br i1 %hasempty, label %notfound, label %grp.cont

grp.cont:
  %probe.next = add nuw i64 %probe, 1
  %gp = add i64 %grp, %probe.next
  %grp.next = and i64 %gp, %maskg
  br label %grp.loop

found:
  ret i64 %cand

notfound:
  ret i64 -1
}

; First EMPTY-or-TOMBSTONE slot in the probe order (guaranteed to exist).
define internal i64 @swm_insert_slot(ptr %ctrl, i64 %hash, i64 %maskg) #0 {
entry:
  %h1 = lshr i64 %hash, 7
  %grp0 = and i64 %h1, %maskg
  br label %grp.loop

grp.loop:
  %grp = phi i64 [ %grp0, %entry ], [ %grp.next, %grp.cont ]
  %probe = phi i64 [ 0, %entry ], [ %probe.next, %grp.cont ]
  %base = shl nuw i64 %grp, 4
  %cptr = getelementptr inbounds nuw i8, ptr %ctrl, i64 %base
  %g = load <16 x i8>, ptr %cptr, align 16
  %av = icmp slt <16 x i8> %g, zeroinitializer   ; high bit set = empty|tomb
  %am = bitcast <16 x i1> %av to i16
  %has = icmp ne i16 %am, 0
  br i1 %has, label %take, label %grp.cont

take:
  %bit = call i16 @llvm.cttz.i16(i16 %am, i1 true)
  %bit64 = zext i16 %bit to i64
  %slot = add nuw i64 %base, %bit64
  ret i64 %slot

grp.cont:
  %probe.next = add nuw i64 %probe, 1
  %gp = add i64 %grp, %probe.next
  %grp.next = and i64 %gp, %maskg
  br label %grp.loop
}

define noalias ptr @universe_ds_hashmap_swiss_create(i64 %val_size, i64 %initial_cap) local_unnamed_addr #1 {
entry:
  ; slots = next_pow2(max(16, ceil(initial_cap * 8 / 7)))
  %m8 = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %initial_cap, i64 8)
  %need8 = extractvalue { i64, i1 } %m8, 0
  %m8.o = extractvalue { i64, i1 } %m8, 1
  br i1 %m8.o, label %fail, label %shape, !prof !0

shape:
  %need8c = add nuw i64 %need8, 6
  %div7 = udiv i64 %need8c, 7
  %base = call i64 @llvm.umax.i64(i64 %div7, i64 16)
  %bm1 = add i64 %base, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %bm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %slots = shl nuw i64 1, %shift
  ; total bytes = slots * (9 + val_size)
  %vv = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %val_size, i64 9)
  %per = extractvalue { i64, i1 } %vv, 0
  %vv.o = extractvalue { i64, i1 } %vv, 1
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %slots, i64 %per)
  %total = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  %ovf = or i1 %vv.o, %tb.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %hdr = call ptr @malloc(i64 64)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %alloc.tbl, !prof !0

alloc.tbl:
  %slot.pp = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot.pp, i64 64, i64 %total)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %free.hdr, label %chk.tbl, !prof !0

chk.tbl:
  %table = load ptr, ptr %slot.pp, align 8
  %tbl.null = icmp eq ptr %table, null
  br i1 %tbl.null, label %free.hdr, label %init, !prof !0

free.hdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  call void @llvm.memset.p0.i64(ptr %table, i8 -128, i64 %slots, i1 false)
  store ptr %table, ptr %hdr, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 0, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store i64 %slots, ptr %cap.p, align 8
  %maskg.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  %ng = lshr i64 %slots, 4
  %maskg = add i64 %ng, -1
  store i64 %maskg, ptr %maskg.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 0, ptr %tomb.p, align 8
  %vsz.p = getelementptr inbounds nuw i8, ptr %hdr, i64 40
  store i64 %val_size, ptr %vsz.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

; Grow/reclaim: rehash into a fresh table, update header in place.
; Returns 0 OK, 2 OOM, 3 SIZE_OVERFLOW.
define internal i32 @swm_resize(ptr %m) #1 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %oldcap = load i64, ptr %cap.p, align 8
  %vsz.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %vsz = load i64, ptr %vsz.p, align 8
  %oldtable = load ptr, ptr %m, align 8
  ; new_cap: reclaim (same) if live fits in half, else grow 2x.
  %half = lshr i64 %oldcap, 1
  %cnt1 = add nuw i64 %count, 1
  %reclaim = icmp ule i64 %cnt1, %half
  %grown = shl nuw i64 %oldcap, 1
  %newcap = select i1 %reclaim, i64 %oldcap, i64 %grown
  ; total = newcap * (9 + vsz)   (vsz already validated at create; recheck)
  %vv = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %vsz, i64 9)
  %per = extractvalue { i64, i1 } %vv, 0
  %vv.o = extractvalue { i64, i1 } %vv, 1
  %tb = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %newcap, i64 %per)
  %total = extractvalue { i64, i1 } %tb, 0
  %tb.o = extractvalue { i64, i1 } %tb, 1
  %ovf = or i1 %vv.o, %tb.o
  br i1 %ovf, label %err.ovf, label %alloc, !prof !0

alloc:
  %slot.pp = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot.pp, i64 64, i64 %total)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %err.oom, label %chk, !prof !0

chk:
  %newtable = load ptr, ptr %slot.pp, align 8
  %nt.null = icmp eq ptr %newtable, null
  br i1 %nt.null, label %err.oom, label %setup, !prof !0

setup:
  call void @llvm.memset.p0.i64(ptr %newtable, i8 -128, i64 %newcap, i1 false)
  %newng = lshr i64 %newcap, 4
  %newmaskg = add i64 %newng, -1
  %oldkeys = getelementptr inbounds nuw i8, ptr %oldtable, i64 %oldcap
  %oldcap8 = shl nuw i64 %oldcap, 3
  %oldvals = getelementptr inbounds nuw i8, ptr %oldkeys, i64 %oldcap8
  %newkeys = getelementptr inbounds nuw i8, ptr %newtable, i64 %newcap
  %newcap8 = shl nuw i64 %newcap, 3
  %newvals = getelementptr inbounds nuw i8, ptr %newkeys, i64 %newcap8
  br label %scan

scan:
  %i = phi i64 [ 0, %setup ], [ %i.next, %scan.cont ]
  %done = icmp eq i64 %i, %oldcap
  br i1 %done, label %finish, label %scan.body

scan.body:
  %cb.p = getelementptr inbounds nuw i8, ptr %oldtable, i64 %i
  %cb = load i8, ptr %cb.p, align 1
  %full = icmp sge i8 %cb, 0        ; high bit clear = full
  br i1 %full, label %move, label %scan.cont

move:
  %okp = getelementptr inbounds i64, ptr %oldkeys, i64 %i
  %ok = load i64, ptr %okp, align 8
  %oh = call i64 @swm_hash(i64 %ok)
  %ns = call i64 @swm_insert_slot(ptr %newtable, i64 %oh, i64 %newmaskg)
  %ncb.p = getelementptr inbounds nuw i8, ptr %newtable, i64 %ns
  %otag64 = and i64 %oh, 127
  %otag = trunc i64 %otag64 to i8
  store i8 %otag, ptr %ncb.p, align 1
  %nkp = getelementptr inbounds i64, ptr %newkeys, i64 %ns
  store i64 %ok, ptr %nkp, align 8
  %ovoff = mul nuw i64 %i, %vsz
  %ovp = getelementptr inbounds nuw i8, ptr %oldvals, i64 %ovoff
  %nvoff = mul nuw i64 %ns, %vsz
  %nvp = getelementptr inbounds nuw i8, ptr %newvals, i64 %nvoff
  call void @llvm.memcpy.p0.p0.i64(ptr %nvp, ptr %ovp, i64 %vsz, i1 false)
  br label %scan.cont

scan.cont:
  %i.next = add nuw i64 %i, 1
  br label %scan

finish:
  call void @free(ptr %oldtable)
  store ptr %newtable, ptr %m, align 8
  store i64 %newcap, ptr %cap.p, align 8
  %maskg.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  store i64 %newmaskg, ptr %maskg.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  store i64 0, ptr %tomb.p, align 8
  ret i32 0

err.oom:
  ret i32 2

err.ovf:
  ret i32 3
}

define i32 @universe_ds_hashmap_swiss_put(ptr %m, i64 %key, ptr %val) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %chk.val, !prof !0

chk.val:
  %vsz.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %vsz = load i64, ptr %vsz.p, align 8
  %val.null = icmp eq ptr %val, null
  %vsz.nz = icmp ne i64 %vsz, 0
  %val.bad = and i1 %val.null, %vsz.nz
  br i1 %val.bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %ptrs = call { ptr, ptr, ptr } @swm_ptrs(ptr nonnull %m)
  %ctrl = extractvalue { ptr, ptr, ptr } %ptrs, 0
  %keys = extractvalue { ptr, ptr, ptr } %ptrs, 1
  %values = extractvalue { ptr, ptr, ptr } %ptrs, 2
  %maskg.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %maskg = load i64, ptr %maskg.p, align 8
  %hash = call i64 @swm_hash(i64 %key)
  %slot = call i64 @swm_find(ptr %ctrl, ptr %keys, i64 %key, i64 %hash, i64 %maskg)
  %present = icmp sge i64 %slot, 0
  br i1 %present, label %overwrite, label %absent

overwrite:
  %voff = mul nuw i64 %slot, %vsz
  %vp = getelementptr inbounds nuw i8, ptr %values, i64 %voff
  call void @llvm.memcpy.p0.p0.i64(ptr %vp, ptr %val, i64 %vsz, i1 false)
  ret i32 0

absent:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %tomb = load i64, ptr %tomb.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %cap = load i64, ptr %cap.p, align 8
  ; threshold = cap*7/8 = cap - cap/8; resize if used+1 > threshold
  %cap8 = lshr i64 %cap, 3
  %thresh = sub i64 %cap, %cap8
  %used = add nuw i64 %count, %tomb
  %used1 = add nuw i64 %used, 1
  %need.grow = icmp ugt i64 %used1, %thresh
  br i1 %need.grow, label %grow, label %place, !prof !0

grow:
  %rc = call i32 @swm_resize(ptr nonnull %m)
  %rc.ok = icmp eq i32 %rc, 0
  br i1 %rc.ok, label %reload, label %err.resize, !prof !2

err.resize:
  ret i32 %rc

reload:
  %ptrs2 = call { ptr, ptr, ptr } @swm_ptrs(ptr nonnull %m)
  %ctrl2 = extractvalue { ptr, ptr, ptr } %ptrs2, 0
  %values2 = extractvalue { ptr, ptr, ptr } %ptrs2, 2
  %maskg2 = load i64, ptr %maskg.p, align 8
  br label %place

place:
  %ctrl.f = phi ptr [ %ctrl, %absent ], [ %ctrl2, %reload ]
  %values.f = phi ptr [ %values, %absent ], [ %values2, %reload ]
  %maskg.f = phi i64 [ %maskg, %absent ], [ %maskg2, %reload ]
  %cap.cur = load i64, ptr %cap.p, align 8
  %keys.f = getelementptr inbounds nuw i8, ptr %ctrl.f, i64 %cap.cur
  %islot = call i64 @swm_insert_slot(ptr %ctrl.f, i64 %hash, i64 %maskg.f)
  %icb.p = getelementptr inbounds nuw i8, ptr %ctrl.f, i64 %islot
  %icb = load i8, ptr %icb.p, align 1
  %was.tomb = icmp eq i8 %icb, -2          ; 0xFE tombstone
  ; write control tag, key, value
  %tag64 = and i64 %hash, 127
  %tag = trunc i64 %tag64 to i8
  store i8 %tag, ptr %icb.p, align 1
  %kp = getelementptr inbounds i64, ptr %keys.f, i64 %islot
  store i64 %key, ptr %kp, align 8
  %ivoff = mul nuw i64 %islot, %vsz
  %ivp = getelementptr inbounds nuw i8, ptr %values.f, i64 %ivoff
  call void @llvm.memcpy.p0.p0.i64(ptr %ivp, ptr %val, i64 %vsz, i1 false)
  ; count++
  %cnt.now = load i64, ptr %count.p, align 8
  %cnt.new = add nuw i64 %cnt.now, 1
  store i64 %cnt.new, ptr %count.p, align 8
  br i1 %was.tomb, label %dec.tomb, label %done

dec.tomb:
  %tomb.now = load i64, ptr %tomb.p, align 8
  %tomb.new = sub i64 %tomb.now, 1
  store i64 %tomb.new, ptr %tomb.p, align 8
  br label %done

done:
  ret i32 0
}

define i32 @universe_ds_hashmap_swiss_get(ptr %m, i64 %key, ptr %out) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  %out.null = icmp eq ptr %out, null
  %bad = or i1 %m.null, %out.null
  br i1 %bad, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %ptrs = call { ptr, ptr, ptr } @swm_ptrs(ptr nonnull %m)
  %ctrl = extractvalue { ptr, ptr, ptr } %ptrs, 0
  %keys = extractvalue { ptr, ptr, ptr } %ptrs, 1
  %values = extractvalue { ptr, ptr, ptr } %ptrs, 2
  %maskg.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %maskg = load i64, ptr %maskg.p, align 8
  %hash = call i64 @swm_hash(i64 %key)
  %slot = call i64 @swm_find(ptr %ctrl, ptr %keys, i64 %key, i64 %hash, i64 %maskg)
  %miss = icmp slt i64 %slot, 0
  br i1 %miss, label %notfound, label %hit

notfound:
  ret i32 5

hit:
  %vsz.p = getelementptr inbounds nuw i8, ptr %m, i64 40
  %vsz = load i64, ptr %vsz.p, align 8
  %voff = mul nuw i64 %slot, %vsz
  %vp = getelementptr inbounds nuw i8, ptr %values, i64 %voff
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %vp, i64 %vsz, i1 false)
  ret i32 0
}

define i32 @universe_ds_hashmap_swiss_contains(ptr %m, i64 %key) local_unnamed_addr #2 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %ptrs = call { ptr, ptr, ptr } @swm_ptrs(ptr nonnull %m)
  %ctrl = extractvalue { ptr, ptr, ptr } %ptrs, 0
  %keys = extractvalue { ptr, ptr, ptr } %ptrs, 1
  %maskg.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %maskg = load i64, ptr %maskg.p, align 8
  %hash = call i64 @swm_hash(i64 %key)
  %slot = call i64 @swm_find(ptr %ctrl, ptr %keys, i64 %key, i64 %hash, i64 %maskg)
  %miss = icmp slt i64 %slot, 0
  %r = select i1 %miss, i32 5, i32 0
  ret i32 %r
}

define i32 @universe_ds_hashmap_swiss_remove(ptr %m, i64 %key) local_unnamed_addr #1 {
entry:
  %m.null = icmp eq ptr %m, null
  br i1 %m.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %ptrs = call { ptr, ptr, ptr } @swm_ptrs(ptr nonnull %m)
  %ctrl = extractvalue { ptr, ptr, ptr } %ptrs, 0
  %keys = extractvalue { ptr, ptr, ptr } %ptrs, 1
  %maskg.p = getelementptr inbounds nuw i8, ptr %m, i64 24
  %maskg = load i64, ptr %maskg.p, align 8
  %hash = call i64 @swm_hash(i64 %key)
  %slot = call i64 @swm_find(ptr %ctrl, ptr %keys, i64 %key, i64 %hash, i64 %maskg)
  %miss = icmp slt i64 %slot, 0
  br i1 %miss, label %notfound, label %erase

notfound:
  ret i32 5

erase:
  %cb.p = getelementptr inbounds nuw i8, ptr %ctrl, i64 %slot
  store i8 -2, ptr %cb.p, align 1                 ; 0xFE tombstone
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  %count.n = sub i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  %tomb.p = getelementptr inbounds nuw i8, ptr %m, i64 32
  %tomb = load i64, ptr %tomb.p, align 8
  %tomb.n = add nuw i64 %tomb, 1
  store i64 %tomb.n, ptr %tomb.p, align 8
  ret i32 0
}

define i64 @universe_ds_hashmap_swiss_len(ptr %m) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %m, i64 8
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define i64 @universe_ds_hashmap_swiss_capacity(ptr %m) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %m, i64 16
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define void @universe_ds_hashmap_swiss_destroy(ptr %m) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %m, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  %table = load ptr, ptr %m, align 8
  call void @free(ptr %table)
  call void @free(ptr nonnull %m)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync memory(none) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 2000, i32 1}
