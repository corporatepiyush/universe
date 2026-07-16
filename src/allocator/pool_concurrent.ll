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

; Concurrent fixed-block pool: LOCK-FREE Treiber free list + wait-free
; wilderness bump.
;
; DESIGN (vs the typical C implementation):
;   * A mutex around a linked free list is the naive approach. Here: no locks at all.
;   * ABA is defeated with a generation tag packed beside a 32-bit block
;     offset in ONE 64-bit head word: { tag:32 | off16:32 }, off16 =
;     byte_offset/16 (stride is 16-aligned so /16 is exact; pool payload is
;     capped at 64 GiB, checked at create). 64-bit CAS is portable —
;     no cmpxchg16b / casp dependency.
;   * Fresh blocks come from a separate wait-free bump cursor (atomicrmw add)
;     so an empty free list never serializes allocs behind a CAS retry loop.
;   * Orderings: pop CAS is acq_rel(acquire on the successful read of the
;     head so the block's intrusive next written by the freeing thread is
;     visible), push CAS is release (publishes the block's next). Failure
;     orderings monotonic. Stats counter monotonic (gates nothing).
;   * Layout (relative offsets; 128B apart = never same cache line):
;       @0   head      (atomic i64: tag|off16)     — hot CAS line
;       @128 next_fresh(atomic i64), stride @136, limit @144, count @152
;       @256 live      (atomic i64)                — stats line
;       @384 payload
;
; API:
;   ptr  universe_alloc_cpool_create(i64 block_size, i64 block_count)
;   ptr  universe_alloc_cpool_alloc(ptr p)
;   void universe_alloc_cpool_free(ptr p, ptr block)
;   i64  universe_alloc_cpool_live(ptr p)
;   i64  universe_alloc_cpool_capacity(ptr p)
;   void universe_alloc_cpool_destroy(ptr p)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

define noalias ptr @universe_alloc_cpool_create(i64 %block_size, i64 %block_count) local_unnamed_addr #1 {
entry:
  %bs = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %block_size, i64 15)
  %bs.up = extractvalue { i64, i1 } %bs, 0
  %bs.o = extractvalue { i64, i1 } %bs, 1
  br i1 %bs.o, label %fail, label %shape, !prof !0

shape:
  %bs.rounded = and i64 %bs.up, -16
  %stride = call i64 @llvm.umax.i64(i64 %bs.rounded, i64 16)
  %lim = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %stride, i64 %block_count)
  %limit = extractvalue { i64, i1 } %lim, 0
  %lim.o = extractvalue { i64, i1 } %lim, 1
  ; off16 must fit 32 bits below the 0xFFFFFFFF sentinel
  %too.big = icmp ugt i64 %limit, 68719476720   ; (2^32-1)*16
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %limit, i64 384)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  %o01 = or i1 %lim.o, %too.big
  %ovf = or i1 %o01, %tot.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store atomic i64 4294967295, ptr %mem monotonic, align 8   ; head: tag 0, off16 = sentinel
  %nf.p = getelementptr inbounds nuw i8, ptr %mem, i64 128
  store atomic i64 0, ptr %nf.p monotonic, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %mem, i64 136
  store i64 %stride, ptr %stride.p, align 8
  %limit.p = getelementptr inbounds nuw i8, ptr %mem, i64 144
  store i64 %limit, ptr %limit.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %mem, i64 152
  store i64 %block_count, ptr %count.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %mem, i64 256
  store atomic i64 0, ptr %live.p monotonic, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define ptr @universe_alloc_cpool_alloc(ptr %pool) local_unnamed_addr #0 {
entry:
  br label %pop.try

pop.try:                                    ; lock-free free-list pop
  %h = load atomic i64, ptr %pool acquire, align 8
  %off16 = and i64 %h, 4294967295
  %empty = icmp eq i64 %off16, 4294967295
  br i1 %empty, label %fresh, label %pop.read

pop.read:
  %off = shl nuw i64 %off16, 4
  %payload = getelementptr inbounds nuw i8, ptr %pool, i64 384
  %block = getelementptr inbounds nuw i8, ptr %payload, i64 %off
  %next16 = load atomic i64, ptr %block monotonic, align 8
  %tag = lshr i64 %h, 32
  %tag.n = add nuw i64 %tag, 1
  %tag.hi = shl i64 %tag.n, 32
  %next.lo = and i64 %next16, 4294967295
  %h.new = or disjoint i64 %tag.hi, %next.lo
  %cas = cmpxchg weak ptr %pool, i64 %h, i64 %h.new acq_rel monotonic, align 8
  %won = extractvalue { i64, i1 } %cas, 1
  br i1 %won, label %counted.ret, label %pop.try, !prof !1

fresh:                                      ; wait-free wilderness bump
  %nf.p = getelementptr inbounds nuw i8, ptr %pool, i64 128
  %stride.p = getelementptr inbounds nuw i8, ptr %pool, i64 136
  %stride = load i64, ptr %stride.p, align 8
  %limit.p = getelementptr inbounds nuw i8, ptr %pool, i64 144
  %limit = load i64, ptr %limit.p, align 8
  %old = atomicrmw add ptr %nf.p, i64 %stride monotonic, align 8
  %in.range = icmp ult i64 %old, %limit
  br i1 %in.range, label %fresh.ok, label %fresh.undo, !prof !2

fresh.ok:
  %payload2 = getelementptr inbounds nuw i8, ptr %pool, i64 384
  %block2 = getelementptr inbounds nuw i8, ptr %payload2, i64 %old
  br label %counted.ret

fresh.undo:                                 ; roll back; one last free-list look
  %undo = atomicrmw sub ptr %nf.p, i64 %stride monotonic, align 8
  %h2 = load atomic i64, ptr %pool acquire, align 8
  %off2 = and i64 %h2, 4294967295
  %empty2 = icmp eq i64 %off2, 4294967295
  br i1 %empty2, label %exhausted, label %pop.try

counted.ret:
  %result = phi ptr [ %block, %pop.read ], [ %block2, %fresh.ok ]
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 256
  %inc = atomicrmw add ptr %live.p, i64 1 monotonic, align 8
  ret ptr %result

exhausted:
  ret ptr null
}

define void @universe_alloc_cpool_free(ptr %pool, ptr %block) local_unnamed_addr #0 {
entry:
  %pool.i = ptrtoint ptr %pool to i64
  %block.i = ptrtoint ptr %block to i64
  %base = add i64 %pool.i, 384
  %off = sub i64 %block.i, %base
  %off16 = lshr i64 %off, 4
  br label %push.try

push.try:
  %h = load atomic i64, ptr %pool monotonic, align 8
  %head.off = and i64 %h, 4294967295
  store atomic i64 %head.off, ptr %block monotonic, align 8   ; block.next = head
  %tag = lshr i64 %h, 32
  %tag.n = add nuw i64 %tag, 1
  %tag.hi = shl i64 %tag.n, 32
  %h.new = or i64 %tag.hi, %off16
  %cas = cmpxchg weak ptr %pool, i64 %h, i64 %h.new release monotonic, align 8
  %won = extractvalue { i64, i1 } %cas, 1
  br i1 %won, label %done, label %push.try, !prof !1

done:
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 256
  %dec = atomicrmw sub ptr %live.p, i64 1 monotonic, align 8
  ret void
}

define i64 @universe_alloc_cpool_live(ptr %pool) local_unnamed_addr #2 {
entry:
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 256
  %live = load atomic i64, ptr %live.p monotonic, align 8
  ret i64 %live
}

define i64 @universe_alloc_cpool_capacity(ptr %pool) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %pool, i64 152
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define void @universe_alloc_cpool_destroy(ptr %pool) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %pool, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %pool)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 2000, i32 1}
