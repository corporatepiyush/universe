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

; Fixed-block pool allocator. O(1) alloc, O(1) free, O(1) create.
;
; DESIGN (vs the typical C implementation):
;   * No vtable — direct calls on an opaque handle.
;   * O(1) create via the wilderness trick: blocks are NOT pre-linked into a
;     free list (an eager init loop touching every block means pointless page
;     faults + cache pollution). Fresh blocks come from a bump cursor;
;     the free list only ever holds blocks that were actually freed.
;   * Free list stores BYTE OFFSETS, not indices — alloc/free do zero
;     multiplications. The next-offset lives inside the free block itself
;     (intrusive, zero metadata overhead). -1 = empty.
;   * Header is one cache line; payload at +64; stride is 16-aligned so
;     every block is 16-byte aligned.
;   * Layout: { i64 head_off, i64 next_fresh, i64 stride, i64 limit,
;               i64 live, i64 count } + pad, payload at +64.
;
; API:
;   ptr  universe_alloc_pool_create(i64 block_size, i64 block_count)
;   ptr  universe_alloc_pool_alloc(ptr p)             ; null when exhausted
;   void universe_alloc_pool_free(ptr p, ptr block)   ; block must be from this pool
;   i64  universe_alloc_pool_live(ptr p)              ; blocks currently allocated
;   i64  universe_alloc_pool_capacity(ptr p)          ; total blocks
;   void universe_alloc_pool_destroy(ptr p)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

define noalias ptr @universe_alloc_pool_create(i64 %block_size, i64 %block_count) local_unnamed_addr #1 {
entry:
  ; stride = max(16, round_up_16(block_size)); a block must hold the 8-byte
  ; intrusive next-offset, which 16 covers.
  %bs = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %block_size, i64 15)
  %bs.up = extractvalue { i64, i1 } %bs, 0
  %bs.o = extractvalue { i64, i1 } %bs, 1
  %bs.rounded = and i64 %bs.up, -16
  %stride = call i64 @llvm.umax.i64(i64 %bs.rounded, i64 16)
  %lim = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %stride, i64 %block_count)
  %limit = extractvalue { i64, i1 } %lim, 0
  %lim.o = extractvalue { i64, i1 } %lim, 1
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %limit, i64 64)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  %o01 = or i1 %bs.o, %lim.o
  %ovf = or i1 %o01, %tot.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  ; OS-request floor: never malloc less than ALLOC_OS_MIN (16384) for backing
  ; memory. The pool bounds allocation by the stored `limit` (= stride*count),
  ; not by the malloc size, so a floored request over a small pool is harmless
  ; slack — capacity/behavior are unchanged, the OS request is just >= 16 KiB.
  %os.req = call i64 @llvm.umax.i64(i64 %total, i64 16384)
  %mem = call ptr @malloc(i64 %os.req)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 -1, ptr %mem, align 8                       ; head_off: empty
  %fresh.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 0, ptr %fresh.p, align 8                    ; next_fresh
  %stride.p = getelementptr inbounds nuw i8, ptr %mem, i64 16
  store i64 %stride, ptr %stride.p, align 8
  %limit.p = getelementptr inbounds nuw i8, ptr %mem, i64 24
  store i64 %limit, ptr %limit.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %mem, i64 32
  store i64 0, ptr %live.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %mem, i64 40
  store i64 %block_count, ptr %count.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define ptr @universe_alloc_pool_alloc(ptr %pool) local_unnamed_addr #0 {
entry:
  %head = load i64, ptr %pool, align 8
  %head.empty = icmp eq i64 %head, -1
  br i1 %head.empty, label %fresh, label %pop

pop:                                                    ; reuse a freed block
  %payload = getelementptr inbounds nuw i8, ptr %pool, i64 64
  %block = getelementptr inbounds nuw i8, ptr %payload, i64 %head
  %next = load i64, ptr %block, align 8
  store i64 %next, ptr %pool, align 8
  br label %bump.live

fresh:                                                  ; wilderness bump
  %fresh.p = getelementptr inbounds nuw i8, ptr %pool, i64 8
  %off = load i64, ptr %fresh.p, align 8
  %limit.p = getelementptr inbounds nuw i8, ptr %pool, i64 24
  %limit = load i64, ptr %limit.p, align 8
  %full = icmp uge i64 %off, %limit
  br i1 %full, label %exhausted, label %take, !prof !0

take:
  %stride.p = getelementptr inbounds nuw i8, ptr %pool, i64 16
  %stride = load i64, ptr %stride.p, align 8
  %off.next = add nuw i64 %off, %stride
  store i64 %off.next, ptr %fresh.p, align 8
  %payload2 = getelementptr inbounds nuw i8, ptr %pool, i64 64
  %block2 = getelementptr inbounds nuw i8, ptr %payload2, i64 %off
  br label %bump.live

bump.live:
  %result = phi ptr [ %block, %pop ], [ %block2, %take ]
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 32
  %live = load i64, ptr %live.p, align 8
  %live.n = add nuw i64 %live, 1
  store i64 %live.n, ptr %live.p, align 8
  ret ptr %result

exhausted:
  ret ptr null
}

define void @universe_alloc_pool_free(ptr %pool, ptr %block) local_unnamed_addr #0 {
entry:
  %head = load i64, ptr %pool, align 8
  store i64 %head, ptr %block, align 8                  ; block.next = old head
  %payload.i = ptrtoint ptr %pool to i64
  %block.i = ptrtoint ptr %block to i64
  %base = add i64 %payload.i, 64
  %off = sub i64 %block.i, %base
  store i64 %off, ptr %pool, align 8                    ; head = block offset
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 32
  %live = load i64, ptr %live.p, align 8
  %live.n = add i64 %live, -1
  store i64 %live.n, ptr %live.p, align 8
  ret void
}

define i64 @universe_alloc_pool_live(ptr %pool) local_unnamed_addr #2 {
entry:
  %live.p = getelementptr inbounds nuw i8, ptr %pool, i64 32
  %live = load i64, ptr %live.p, align 8
  ret i64 %live
}

define i64 @universe_alloc_pool_capacity(ptr %pool) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %pool, i64 40
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define void @universe_alloc_pool_destroy(ptr %pool) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %pool, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %pool)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
