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

; Single-threaded TCP connection pool: caches up to `cap` idle connections
; keyed by (ip,port), reuses live ones, evicts the least-recently-used when
; full. Concurrency is DEFERRED — NO locks, NO atomics (CLAUDE.md).
;
; DESIGN:
;   * ONE allocation (header + struct-of-arrays), index-linked recency list —
;     same idiom as the LRU cache. Links are i32 slot indices (-1 sentinel),
;     not pointers: half the footprint, no pointer chasing.
;   * A pooled entry is (key:i64, fd:i32) where key = (ip<<32)|port. Multiple
;     entries may share a key (a real pool holds several idle conns per peer),
;     so this is a keyed multiset, not a map — lookup is a short walk of the
;     recency list (cap is small; a hash would be overkill and colder).
;   * Occupied slots form an intrusive doubly-linked recency list
;     (head = MRU, tail = LRU). Free slots come from a wilderness bump cursor
;     first (cold memory untouched until needed) then a singly-linked freelist
;     of returned slots (reuses next[]). Create is O(1), no pre-linking.
;   * pool_get validates liveness with a non-blocking MSG_PEEK recv: a peer
;     that closed returns 0 (EOF) -> we close the corpse and keep scanning;
;     -1 (EAGAIN/ENOTCONN) or >0 means the socket is still usable.
;   * pool_put evicts the LRU tail when full, closing the evicted fd, and
;     reuses that slot in place — zero alloc churn at steady state.
;   * Header (64B): cap@0(i64) count@8(i64) wild@16(i64)
;                   head@32(i32) tail@36(i32) freehead@40(i32)
;     Arrays from base+64: keys[cap](i64) fds[cap](i32) prev[cap](i32)
;                          next[cap](i32).
;
; API (0 OK, negative = -errorcode; get returns fd>=0 or -1 = none):
;   ptr universe_net_pool_create(i64 cap)
;   void universe_net_pool_destroy(ptr p)
;   i32  universe_net_pool_get(ptr p, i32 ip, i32 port)
;   i32  universe_net_pool_put(ptr p, i32 fd, i32 ip, i32 port)
;   i64  universe_net_pool_count(ptr p)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @close(i32)
declare i64 @recv(i32, ptr, i64, i32)
declare i32 @universe_net_os_msg_peek_dontwait()   ; osconst_{bsd,linux}.ll
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; section pointers { keys, fds, prev, next }
define internal { ptr, ptr, ptr, ptr } @sections(ptr %p) #0 {
entry:
  %cap = load i64, ptr %p, align 8
  %keys = getelementptr inbounds nuw i8, ptr %p, i64 64
  %b.keys = shl nuw i64 %cap, 3
  %fds = getelementptr inbounds nuw i8, ptr %keys, i64 %b.keys
  %cx4 = shl nuw i64 %cap, 2
  %prev = getelementptr inbounds nuw i8, ptr %fds, i64 %cx4
  %next = getelementptr inbounds nuw i8, ptr %prev, i64 %cx4
  %r0 = insertvalue { ptr, ptr, ptr, ptr } poison, ptr %keys, 0
  %r1 = insertvalue { ptr, ptr, ptr, ptr } %r0, ptr %fds, 1
  %r2 = insertvalue { ptr, ptr, ptr, ptr } %r1, ptr %prev, 2
  %r3 = insertvalue { ptr, ptr, ptr, ptr } %r2, ptr %next, 3
  ret { ptr, ptr, ptr, ptr } %r3
}

; unlink idx from the recency list (idx must be linked)
define internal void @list_unlink(ptr %p, i32 %idx, ptr %prev, ptr %next) #0 {
entry:
  %idx.w = sext i32 %idx to i64
  %p.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %idx.w
  %pv = load i32, ptr %p.p, align 4
  %n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %idx.w
  %nx = load i32, ptr %n.p, align 4
  %p.none = icmp eq i32 %pv, -1
  br i1 %p.none, label %fix.head, label %fix.prev

fix.prev:
  %pv.w = sext i32 %pv to i64
  %pn.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %pv.w
  store i32 %nx, ptr %pn.p, align 4
  br label %mid

fix.head:
  %head.p = getelementptr inbounds nuw i8, ptr %p, i64 32
  store i32 %nx, ptr %head.p, align 4
  br label %mid

mid:
  %n.none = icmp eq i32 %nx, -1
  br i1 %n.none, label %fix.tail, label %fix.next

fix.next:
  %nx.w = sext i32 %nx to i64
  %np.p = getelementptr inbounds [0 x i32], ptr %prev, i64 0, i64 %nx.w
  store i32 %pv, ptr %np.p, align 4
  br label %done

fix.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %p, i64 36
  store i32 %pv, ptr %tail.p, align 4
  br label %done

done:
  ret void
}

; push idx at the recency front (most recent)
define internal void @list_push_front(ptr %p, i32 %idx, ptr %prev, ptr %next) #0 {
entry:
  %head.p = getelementptr inbounds nuw i8, ptr %p, i64 32
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
  %tail.p = getelementptr inbounds nuw i8, ptr %p, i64 36
  store i32 %idx, ptr %tail.p, align 4
  br label %done

done:
  ret void
}

; liveness probe: MSG_PEEK|MSG_DONTWAIT recv of 1 byte. returns 1 alive, 0 dead
define internal i32 @sock_alive(i32 %fd) #1 {
entry:
  %tmp = alloca i8, align 1
  ; MSG_PEEK | MSG_DONTWAIT — value is OS-divergent, resolved by the build
  %flags = call i32 @universe_net_os_msg_peek_dontwait()
  %r = call i64 @recv(i32 %fd, ptr nonnull %tmp, i64 1, i32 %flags)
  %eof = icmp eq i64 %r, 0
  %alive = select i1 %eof, i32 0, i32 1
  ret i32 %alive
}

define noalias ptr @universe_net_pool_create(i64 %cap) local_unnamed_addr #2 {
entry:
  %cap.bad = icmp eq i64 %cap, 0
  %too.big = icmp ugt i64 %cap, 2147483647
  %bad = or i1 %cap.bad, %too.big
  br i1 %bad, label %fail, label %shape, !prof !0

shape:
  ; total = 64 + cap*8 (keys) + cap*4*3 (fds,prev,next) = 64 + cap*20
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 20)
  %body = extractvalue { i64, i1 } %m, 0
  %m.o = extractvalue { i64, i1 } %m, 1
  %a = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %body, i64 64)
  %total = extractvalue { i64, i1 } %a, 0
  %a.o = extractvalue { i64, i1 } %a, 1
  %ovf = or i1 %m.o, %a.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %cap, ptr %mem, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %count.p, align 8
  %wild.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 0, ptr %wild.p, align 8
  %head.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i32 -1, ptr %head.p, align 4
  %tail.p = getelementptr inbounds nuw i8, ptr %mem, i64 36
  store i32 -1, ptr %tail.p, align 4
  %free.p = getelementptr inbounds nuw i8, ptr %mem, i64 40
  store i32 -1, ptr %free.p, align 4
  ret ptr %mem

fail:
  ret ptr null
}

define i64 @universe_net_pool_count(ptr %p) local_unnamed_addr #3 {
entry:
  %null = icmp eq ptr %p, null
  br i1 %null, label %zero, label %read, !prof !0

read:
  %count.p = getelementptr inbounds nuw i8, ptr %p, i64 8
  %count = load i64, ptr %count.p, align 8
  ret i64 %count

zero:
  ret i64 0
}

define i32 @universe_net_pool_get(ptr %p, i32 %ip, i32 %port) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %p, null
  br i1 %null, label %none, label %setup, !prof !0

setup:
  %ip.w = zext i32 %ip to i64
  %port.w = zext i32 %port to i64
  %ip.hi = shl nuw i64 %ip.w, 32
  %key = or i64 %ip.hi, %port.w
  %secs = call { ptr, ptr, ptr, ptr } @sections(ptr nonnull %p)
  %keys = extractvalue { ptr, ptr, ptr, ptr } %secs, 0
  %fds = extractvalue { ptr, ptr, ptr, ptr } %secs, 1
  %prev = extractvalue { ptr, ptr, ptr, ptr } %secs, 2
  %next = extractvalue { ptr, ptr, ptr, ptr } %secs, 3
  %head.p = getelementptr inbounds nuw i8, ptr %p, i64 32
  %h0 = load i32, ptr %head.p, align 4
  br label %walk

walk:
  %cur = phi i32 [ %h0, %setup ], [ %nx, %advance ]
  %end = icmp eq i32 %cur, -1
  br i1 %end, label %none, label %inspect

inspect:
  %cur.w = sext i32 %cur to i64
  %n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %cur.w
  %nx = load i32, ptr %n.p, align 4                 ; save before any unlink
  %k.p = getelementptr inbounds [0 x i64], ptr %keys, i64 0, i64 %cur.w
  %k = load i64, ptr %k.p, align 8
  %match = icmp eq i64 %k, %key
  br i1 %match, label %take, label %advance, !prof !1

take:
  ; remove cur from recency list and return its slot to the freelist
  call void @list_unlink(ptr nonnull %p, i32 %cur, ptr %prev, ptr %next)
  %fd.p = getelementptr inbounds [0 x i32], ptr %fds, i64 0, i64 %cur.w
  %fd = load i32, ptr %fd.p, align 4
  %free.p = getelementptr inbounds nuw i8, ptr %p, i64 40
  %fh = load i32, ptr %free.p, align 4
  store i32 %fh, ptr %n.p, align 4                  ; reuse next[] as freelist link
  store i32 %cur, ptr %free.p, align 4
  %count.p = getelementptr inbounds nuw i8, ptr %p, i64 8
  %cnt = load i64, ptr %count.p, align 8
  %cnt.n = sub nuw i64 %cnt, 1
  store i64 %cnt.n, ptr %count.p, align 8
  ; validate liveness
  %alive = call i32 @sock_alive(i32 %fd)
  %ok = icmp ne i32 %alive, 0
  br i1 %ok, label %hit, label %dead, !prof !2

hit:
  ret i32 %fd

dead:
  %ign = call i32 @close(i32 %fd)
  br label %advance

advance:
  br label %walk

none:
  ret i32 -1
}

define i32 @universe_net_pool_put(ptr %p, i32 %fd, i32 %ip, i32 %port) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %p, null
  %fd.bad = icmp slt i32 %fd, 0
  %bad = or i1 %null, %fd.bad
  br i1 %bad, label %err.arg, label %setup, !prof !0

setup:
  %ip.w = zext i32 %ip to i64
  %port.w = zext i32 %port to i64
  %ip.hi = shl nuw i64 %ip.w, 32
  %key = or i64 %ip.hi, %port.w
  %secs = call { ptr, ptr, ptr, ptr } @sections(ptr nonnull %p)
  %keys = extractvalue { ptr, ptr, ptr, ptr } %secs, 0
  %fds = extractvalue { ptr, ptr, ptr, ptr } %secs, 1
  %prev = extractvalue { ptr, ptr, ptr, ptr } %secs, 2
  %next = extractvalue { ptr, ptr, ptr, ptr } %secs, 3
  ; obtain a slot: freelist -> wilderness bump -> evict LRU tail
  %free.p = getelementptr inbounds nuw i8, ptr %p, i64 40
  %fh = load i32, ptr %free.p, align 4
  %has.free = icmp ne i32 %fh, -1
  br i1 %has.free, label %from.free, label %try.wild

from.free:
  %fh.w = sext i32 %fh to i64
  %fh.n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %fh.w
  %fh.n = load i32, ptr %fh.n.p, align 4
  store i32 %fh.n, ptr %free.p, align 4
  br label %bump.count

try.wild:
  %cap = load i64, ptr %p, align 8
  %wild.p = getelementptr inbounds nuw i8, ptr %p, i64 16
  %wild = load i64, ptr %wild.p, align 8
  %has.wild = icmp ult i64 %wild, %cap
  br i1 %has.wild, label %from.wild, label %evict

from.wild:
  %wild.n = add nuw i64 %wild, 1
  store i64 %wild.n, ptr %wild.p, align 8
  %wslot = trunc i64 %wild to i32
  br label %bump.count

bump.count:
  %slot.new = phi i32 [ %fh, %from.free ], [ %wslot, %from.wild ]
  %count.p = getelementptr inbounds nuw i8, ptr %p, i64 8
  %cnt = load i64, ptr %count.p, align 8
  %cnt.n = add nuw i64 %cnt, 1
  store i64 %cnt.n, ptr %count.p, align 8
  br label %place

evict:
  ; pool is full: close + reuse the LRU tail slot (count unchanged)
  %tail.p = getelementptr inbounds nuw i8, ptr %p, i64 36
  %victim = load i32, ptr %tail.p, align 4
  %victim.w = sext i32 %victim to i64
  %vfd.p = getelementptr inbounds [0 x i32], ptr %fds, i64 0, i64 %victim.w
  %vfd = load i32, ptr %vfd.p, align 4
  %vign = call i32 @close(i32 %vfd)
  call void @list_unlink(ptr nonnull %p, i32 %victim, ptr %prev, ptr %next)
  br label %place

place:
  %slot = phi i32 [ %slot.new, %bump.count ], [ %victim, %evict ]
  %slot.w = sext i32 %slot to i64
  %k.p = getelementptr inbounds [0 x i64], ptr %keys, i64 0, i64 %slot.w
  store i64 %key, ptr %k.p, align 8
  %sfd.p = getelementptr inbounds [0 x i32], ptr %fds, i64 0, i64 %slot.w
  store i32 %fd, ptr %sfd.p, align 4
  call void @list_push_front(ptr nonnull %p, i32 %slot, ptr %prev, ptr %next)
  ret i32 0

err.arg:
  ret i32 -8
}

define void @universe_net_pool_destroy(ptr %p) local_unnamed_addr #2 {
entry:
  %null = icmp eq ptr %p, null
  br i1 %null, label %done, label %setup, !prof !0

setup:
  %secs = call { ptr, ptr, ptr, ptr } @sections(ptr nonnull %p)
  %fds = extractvalue { ptr, ptr, ptr, ptr } %secs, 1
  %next = extractvalue { ptr, ptr, ptr, ptr } %secs, 3
  %head.p = getelementptr inbounds nuw i8, ptr %p, i64 32
  %h0 = load i32, ptr %head.p, align 4
  br label %walk

walk:
  %cur = phi i32 [ %h0, %setup ], [ %nx, %body ]
  %end = icmp eq i32 %cur, -1
  br i1 %end, label %free.it, label %body

body:
  %cur.w = sext i32 %cur to i64
  %n.p = getelementptr inbounds [0 x i32], ptr %next, i64 0, i64 %cur.w
  %nx = load i32, ptr %n.p, align 4
  %fd.p = getelementptr inbounds [0 x i32], ptr %fds, i64 0, i64 %cur.w
  %fd = load i32, ptr %fd.p, align 4
  %ign = call i32 @close(i32 %fd)
  br label %walk

free.it:
  call void @free(ptr nonnull %p)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 4}
!2 = !{!"branch_weights", i32 2000, i32 1}
