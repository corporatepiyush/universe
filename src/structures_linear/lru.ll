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

; LRU cache: i64 keys, fixed-size values, O(1) get/put with eviction.
;
; DESIGN (vs the typical C implementation):
;   * ONE allocation, struct-of-arrays: keys[], bucket-chain links[],
;     lru prev[]/next[], values[], bucket heads[]. No per-entry malloc,
;     no pointer webs — links are i32 indices (half the size of pointers,
;     twice the entries per cache line), -1 sentinel.
;   * Hash table is open chaining over index links with 2x-capacity
;     power-of-two bucket array (load factor <= 0.5), splitmix64 finalizer
;     as the hash (3 xor-shifts + 2 multiplies, no division anywhere).
;   * Recency list is an intrusive doubly-linked index list; move-to-front
;     touches at most 6 i32 stores.
;   * Eviction reuses the victim's slot in place: zero allocation churn at
;     steady state.
;   * Header (64B): cap@0 vsz@8 count@16 hmask@24 head@32(i32) tail@36(i32)
;     + section offsets computed once at create: keys@40 hnext@48(implied)…
;     stored as byte offsets: okeys@40, ohnext@48(i64)… layout fixed:
;     keys | hnext | prev | next | buckets | values (all from base+64).
;
; API (0 OK, 1 NULL_PTR, 5 NOT_FOUND):
;   ptr universe_ds_lru_create(i64 capacity, i64 val_size)
;   i32 universe_ds_lru_put(ptr c, i64 key, ptr val)   ; insert/update, may evict
;   i32 universe_ds_lru_get(ptr c, i64 key, ptr out)   ; hit refreshes recency
;   i32 universe_ds_lru_contains(ptr c, i64 key)       ; no recency change
;   i64 universe_ds_lru_count(ptr c)
;   void universe_ds_lru_destroy(ptr c)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

; splitmix64 finalizer — excellent avalanche, 5 ops
define internal i64 @hash64(i64 %x) #0 {
entry:
  %h1 = lshr i64 %x, 30
  %x1 = xor i64 %h1, %x
  %x2 = mul i64 %x1, -4658895280553007687     ; 0xbf58476d1ce4e5b9
  %h2 = lshr i64 %x2, 27
  %x3 = xor i64 %h2, %x2
  %x4 = mul i64 %x3, -7723592293110705685     ; 0x94d049bb133111eb
  %h3 = lshr i64 %x4, 31
  %x5 = xor i64 %h3, %x4
  ret i64 %x5
}

define noalias ptr @universe_ds_lru_create(i64 %capacity, i64 %val_size) local_unnamed_addr #1 {
entry:
  %cap.bad = icmp eq i64 %capacity, 0
  %too.big = icmp ugt i64 %capacity, 2147483647     ; i32 indices
  %bad = or i1 %cap.bad, %too.big
  br i1 %bad, label %fail, label %shape, !prof !0

shape:
  ; hcap = next_pow2(2*cap)
  %cap2 = shl nuw i64 %capacity, 1
  %c.min = call i64 @llvm.umax.i64(i64 %cap2, i64 8)
  %cm1 = add i64 %c.min, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %cm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %hcap = shl nuw i64 1, %shift
  ; bytes: keys 8c + hnext 4c + prev 4c + next 4c + buckets 4h + vals vsz*c
  %b.keys = shl nuw i64 %capacity, 3
  %b.links = mul nuw i64 %capacity, 12
  %b.buckets = shl nuw i64 %hcap, 2
  %vals = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %capacity, i64 %val_size)
  %b.vals = extractvalue { i64, i1 } %vals, 0
  %vals.o = extractvalue { i64, i1 } %vals, 1
  %s0 = add nuw i64 %b.keys, %b.links
  %s1 = add nuw i64 %s0, %b.buckets
  %t0 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s1, i64 %b.vals)
  %s2 = extractvalue { i64, i1 } %t0, 0
  %t0.o = extractvalue { i64, i1 } %t0, 1
  %t1 = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %s2, i64 64)
  %total = extractvalue { i64, i1 } %t1, 0
  %t1.o = extractvalue { i64, i1 } %t1, 1
  %o0 = or i1 %vals.o, %t0.o
  %ovf = or i1 %o0, %t1.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %capacity, ptr %mem, align 8
  %vsz.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %val_size, ptr %vsz.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 0, ptr %count.p, align 8
  %hmask.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  %hmask = add i64 %hcap, -1
  store i64 %hmask, ptr %hmask.p, align 8
  %head.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i32 -1, ptr %head.p, align 4
  %tail.p = getelementptr inbounds nuw i8, ptr %mem, i64 36
  store i32 -1, ptr %tail.p, align 4
  ; buckets := -1 (0xFF fill)
  %keys.base = getelementptr inbounds nuw i8, ptr %mem, i64 64
  %hnext.base = getelementptr inbounds nuw i8, ptr %keys.base, i64 %b.keys
  %cx4 = shl nuw i64 %capacity, 2
  %prev.base = getelementptr inbounds nuw i8, ptr %hnext.base, i64 %cx4
  %next.base = getelementptr inbounds nuw i8, ptr %prev.base, i64 %cx4
  %buckets.base = getelementptr inbounds nuw i8, ptr %next.base, i64 %cx4
  call void @llvm.memset.p0.i64(ptr %buckets.base, i8 -1, i64 %b.buckets, i1 false)
  ret ptr %mem

fail:
  ret ptr null
}

; --- internal section-pointer helpers (recomputed; cheap shifts/adds) -------

define internal { ptr, ptr, ptr, ptr, ptr, ptr } @sections(ptr %c) #0 {
entry:
  %cap = load i64, ptr %c, align 8
  %keys = getelementptr inbounds nuw i8, ptr %c, i64 64
  %b.keys = shl nuw i64 %cap, 3
  %hnext = getelementptr inbounds nuw i8, ptr %keys, i64 %b.keys
  %cx4 = shl nuw i64 %cap, 2
  %prev = getelementptr inbounds nuw i8, ptr %hnext, i64 %cx4
  %next = getelementptr inbounds nuw i8, ptr %prev, i64 %cx4
  %buckets = getelementptr inbounds nuw i8, ptr %next, i64 %cx4
  %hmask.p = getelementptr inbounds nuw i8, ptr %c, i64 24
  %hmask = load i64, ptr %hmask.p, align 8
  %hcap = add i64 %hmask, 1
  %b.buckets = shl nuw i64 %hcap, 2
  %vals = getelementptr inbounds nuw i8, ptr %buckets, i64 %b.buckets
  %r0 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } poison, ptr %keys, 0
  %r1 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } %r0, ptr %hnext, 1
  %r2 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } %r1, ptr %prev, 2
  %r3 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } %r2, ptr %next, 3
  %r4 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } %r3, ptr %buckets, 4
  %r5 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr } %r4, ptr %vals, 5
  ret { ptr, ptr, ptr, ptr, ptr, ptr } %r5
}

; find index of key: returns i32 idx or -1. (chain walk; <=few links at LF 0.5)
define internal i32 @lookup(ptr %c, i64 %key, ptr %keys, ptr %hnext, ptr %buckets) #0 {
entry:
  %h = call i64 @hash64(i64 %key)
  %hmask.p = getelementptr inbounds nuw i8, ptr %c, i64 24
  %hmask = load i64, ptr %hmask.p, align 8
  %b = and i64 %h, %hmask
  %b.p = getelementptr inbounds nuw [0 x i32], ptr %buckets, i64 0, i64 %b
  %first = load i32, ptr %b.p, align 4
  br label %walk

walk:
  %idx = phi i32 [ %first, %entry ], [ %nxt, %step ]
  %miss = icmp eq i32 %idx, -1
  br i1 %miss, label %notfound, label %cmp

cmp:
  %idx.w = sext i32 %idx to i64
  %k.p = getelementptr inbounds [0 x i64], ptr %keys, i64 0, i64 %idx.w
  %k = load i64, ptr %k.p, align 8
  %hit = icmp eq i64 %k, %key
  br i1 %hit, label %found, label %step, !prof !1

step:
  %n.p = getelementptr inbounds [0 x i32], ptr %hnext, i64 0, i64 %idx.w
  %nxt = load i32, ptr %n.p, align 4
  br label %walk

found:
  ret i32 %idx

notfound:
  ret i32 -1
}

; unlink idx from recency list (idx must be linked)
define internal void @list_unlink(ptr %c, i32 %idx, ptr %prev, ptr %next) #0 {
entry:
  %idx.w = sext i32 %idx to i64
  %p.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %idx.w
  %p = load i32, ptr %p.p, align 4
  %n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %idx.w
  %n = load i32, ptr %n.p, align 4
  %p.none = icmp eq i32 %p, -1
  br i1 %p.none, label %fix.head, label %fix.prev

fix.prev:
  %p.w = sext i32 %p to i64
  %pn.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %p.w
  store i32 %n, ptr %pn.p, align 4
  br label %mid

fix.head:
  %head.p = getelementptr inbounds nuw i8, ptr %c, i64 32
  store i32 %n, ptr %head.p, align 4
  br label %mid

mid:
  %n.none = icmp eq i32 %n, -1
  br i1 %n.none, label %fix.tail, label %fix.next

fix.next:
  %n.w = sext i32 %n to i64
  %np.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %n.w
  store i32 %p, ptr %np.p, align 4
  br label %done

fix.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %c, i64 36
  store i32 %p, ptr %tail.p, align 4
  br label %done

done:
  ret void
}

; push idx at list front (most recent)
define internal void @list_push_front(ptr %c, i32 %idx, ptr %prev, ptr %next) #0 {
entry:
  %head.p = getelementptr inbounds nuw i8, ptr %c, i64 32
  %old = load i32, ptr %head.p, align 4
  %idx.w = sext i32 %idx to i64
  %p.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %idx.w
  store i32 -1, ptr %p.p, align 4
  %n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %idx.w
  store i32 %old, ptr %n.p, align 4
  store i32 %idx, ptr %head.p, align 4
  %old.none = icmp eq i32 %old, -1
  br i1 %old.none, label %set.tail, label %link.old

link.old:
  %old.w = sext i32 %old to i64
  %op.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %old.w
  store i32 %idx, ptr %op.p, align 4
  br label %done

set.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %c, i64 36
  store i32 %idx, ptr %tail.p, align 4
  br label %done

done:
  ret void
}

; remove idx from its hash bucket chain
define internal void @bucket_remove(ptr %c, i32 %idx, i64 %key, ptr %hnext, ptr %buckets) #0 {
entry:
  %h = call i64 @hash64(i64 %key)
  %hmask.p = getelementptr inbounds nuw i8, ptr %c, i64 24
  %hmask = load i64, ptr %hmask.p, align 8
  %b = and i64 %h, %hmask
  %b.p = getelementptr inbounds [0 x i32], ptr %buckets, i64 0, i64 %b
  %idx.w = sext i32 %idx to i64
  %my.n.p = getelementptr inbounds [0 x i32], ptr %hnext, i64 0, i64 %idx.w
  %my.n = load i32, ptr %my.n.p, align 4
  %first = load i32, ptr %b.p, align 4
  %is.first = icmp eq i32 %first, %idx
  br i1 %is.first, label %pop.head, label %walk

pop.head:
  store i32 %my.n, ptr %b.p, align 4
  ret void

walk:
  %cur = phi i32 [ %first, %entry ], [ %nxt, %walk.body ]
  %cur.w = sext i32 %cur to i64
  %n.p = getelementptr inbounds [0 x i32], ptr %hnext, i64 0, i64 %cur.w
  %nxt = load i32, ptr %n.p, align 4
  %found = icmp eq i32 %nxt, %idx
  br i1 %found, label %splice, label %walk.body

walk.body:
  br label %walk

splice:
  store i32 %my.n, ptr %n.p, align 4
  ret void
}

define i32 @universe_ds_lru_put(ptr %c, i64 %key, ptr %val) local_unnamed_addr #1 {
entry:
  %c.null = icmp eq ptr %c, null
  %val.null = icmp eq ptr %val, null
  %any.null = or i1 %c.null, %val.null
  br i1 %any.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %secs = call { ptr, ptr, ptr, ptr, ptr, ptr } @sections(ptr nonnull %c)
  %keys = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 0
  %hnext = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 1
  %prev = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 2
  %next = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 3
  %buckets = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 4
  %vals = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 5
  %vsz.p = getelementptr inbounds nuw i8, ptr %c, i64 8
  %vsz = load i64, ptr %vsz.p, align 8
  %idx = call i32 @lookup(ptr nonnull %c, i64 %key, ptr %keys, ptr %hnext, ptr %buckets)
  %hit = icmp ne i32 %idx, -1
  br i1 %hit, label %update, label %insert

update:                                     ; overwrite value, refresh recency
  %idx.w = sext i32 %idx to i64
  %voff = mul nuw i64 %idx.w, %vsz
  %v.p = getelementptr inbounds nuw i8, ptr %vals, i64 %voff
  call void @llvm.memcpy.p0.p0.i64(ptr %v.p, ptr %val, i64 %vsz, i1 false)
  call void @list_unlink(ptr nonnull %c, i32 %idx, ptr %prev, ptr %next)
  call void @list_push_front(ptr nonnull %c, i32 %idx, ptr %prev, ptr %next)
  ret i32 0

insert:
  %count.p = getelementptr inbounds nuw i8, ptr %c, i64 16
  %count = load i64, ptr %count.p, align 8
  %cap = load i64, ptr %c, align 8
  %room = icmp ult i64 %count, %cap
  br i1 %room, label %fresh, label %evict

fresh:
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  %slot.fresh = trunc i64 %count to i32
  br label %place

evict:                                      ; reuse LRU victim's slot
  %tail.p = getelementptr inbounds nuw i8, ptr %c, i64 36
  %victim = load i32, ptr %tail.p, align 4
  %victim.w = sext i32 %victim to i64
  %vk.p = getelementptr inbounds [0 x i64], ptr %keys, i64 0, i64 %victim.w
  %victim.key = load i64, ptr %vk.p, align 8
  call void @bucket_remove(ptr nonnull %c, i32 %victim, i64 %victim.key, ptr %hnext, ptr %buckets)
  call void @list_unlink(ptr nonnull %c, i32 %victim, ptr %prev, ptr %next)
  br label %place

place:
  %slot = phi i32 [ %slot.fresh, %fresh ], [ %victim, %evict ]
  %slot.w = sext i32 %slot to i64
  ; key + value
  %k.p = getelementptr inbounds [0 x i64], ptr %keys, i64 0, i64 %slot.w
  store i64 %key, ptr %k.p, align 8
  %voff2 = mul nuw i64 %slot.w, %vsz
  %v.p2 = getelementptr inbounds nuw i8, ptr %vals, i64 %voff2
  call void @llvm.memcpy.p0.p0.i64(ptr %v.p2, ptr %val, i64 %vsz, i1 false)
  ; bucket push-front
  %h = call i64 @hash64(i64 %key)
  %hmask.p = getelementptr inbounds nuw i8, ptr %c, i64 24
  %hmask = load i64, ptr %hmask.p, align 8
  %b = and i64 %h, %hmask
  %b.p = getelementptr inbounds [0 x i32], ptr %buckets, i64 0, i64 %b
  %first = load i32, ptr %b.p, align 4
  %hn.p = getelementptr inbounds [0 x i32], ptr %hnext, i64 0, i64 %slot.w
  store i32 %first, ptr %hn.p, align 4
  store i32 %slot, ptr %b.p, align 4
  ; recency front
  call void @list_push_front(ptr nonnull %c, i32 %slot, ptr %prev, ptr %next)
  ret i32 0
}

define i32 @universe_ds_lru_get(ptr %c, i64 %key, ptr %out) local_unnamed_addr #1 {
entry:
  %c.null = icmp eq ptr %c, null
  %out.null = icmp eq ptr %out, null
  %any.null = or i1 %c.null, %out.null
  br i1 %any.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %secs = call { ptr, ptr, ptr, ptr, ptr, ptr } @sections(ptr nonnull %c)
  %keys = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 0
  %hnext = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 1
  %prev = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 2
  %next = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 3
  %buckets = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 4
  %vals = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 5
  %idx = call i32 @lookup(ptr nonnull %c, i64 %key, ptr %keys, ptr %hnext, ptr %buckets)
  %miss = icmp eq i32 %idx, -1
  br i1 %miss, label %notfound, label %hit

notfound:
  ret i32 5

hit:
  %vsz.p = getelementptr inbounds nuw i8, ptr %c, i64 8
  %vsz = load i64, ptr %vsz.p, align 8
  %idx.w = sext i32 %idx to i64
  %voff = mul nuw i64 %idx.w, %vsz
  %v.p = getelementptr inbounds nuw i8, ptr %vals, i64 %voff
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %v.p, i64 %vsz, i1 false)
  call void @list_unlink(ptr nonnull %c, i32 %idx, ptr %prev, ptr %next)
  call void @list_push_front(ptr nonnull %c, i32 %idx, ptr %prev, ptr %next)
  ret i32 0
}

define i32 @universe_ds_lru_contains(ptr %c, i64 %key) local_unnamed_addr #1 {
entry:
  %c.null = icmp eq ptr %c, null
  br i1 %c.null, label %err.null, label %setup, !prof !0

err.null:
  ret i32 1

setup:
  %secs = call { ptr, ptr, ptr, ptr, ptr, ptr } @sections(ptr nonnull %c)
  %keys = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 0
  %hnext = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 1
  %buckets = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr } %secs, 4
  %idx = call i32 @lookup(ptr nonnull %c, i64 %key, ptr %keys, ptr %hnext, ptr %buckets)
  %miss = icmp eq i32 %idx, -1
  %r = select i1 %miss, i32 5, i32 0
  ret i32 %r
}

define i64 @universe_ds_lru_count(ptr %c) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %c, i64 16
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define void @universe_ds_lru_destroy(ptr %c) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %c, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %c)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
