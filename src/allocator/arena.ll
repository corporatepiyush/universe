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

; Arena (bump) allocator. O(1) alloc, O(1) whole-arena reset, zero per-block
; metadata.
;
; DESIGN (vs the typical C implementation):
;   * typical C allocator APIs route every call through a vtable (indirect call +
;     dependent load per op). Universe exposes direct functions on an opaque
;     handle: the alloc fast path is ~6 instructions, fully branch-predicted.
;   * Header and payload live in ONE malloc; header is exactly one cache
;     line, payload starts 64B in (16B-aligned by malloc contract, same line
;     never shared with user data).
;   * `used` is kept 16-aligned as an invariant, so the fast path needs no
;     align-up of the cursor — only the size is rounded (add+and).
;   * All size math is overflow-checked (umul/uadd.with.overflow → fail).
;   * Layout: { i64 used, i64 capacity } + 48B pad, payload at +64.
;
; API:
;   ptr  universe_alloc_arena_create(i64 capacity)          ; null on OOM/overflow
;   ptr  universe_alloc_arena_alloc(ptr a, i64 size)        ; 16-aligned; null when full
;   ptr  universe_alloc_arena_alloc_aligned(ptr a, i64 size, i64 align) ; align = pow2
;   void universe_alloc_arena_reset(ptr a)                  ; frees nothing, reuses all
;   i64  universe_alloc_arena_used(ptr a)
;   i64  universe_alloc_arena_capacity(ptr a)
;   void universe_alloc_arena_destroy(ptr a)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

define noalias ptr @universe_alloc_arena_create(i64 %capacity) local_unnamed_addr #1 {
entry:
  ; total = 64 + capacity, overflow-checked
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %capacity, i64 64)
  %total = extractvalue { i64, i1 } %tot, 0
  %ovf = extractvalue { i64, i1 } %tot, 1
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 0, ptr %mem, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %capacity, ptr %cap.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define ptr @universe_alloc_arena_alloc(ptr %arena, i64 %size) local_unnamed_addr #0 {
entry:
  %used = load i64, ptr %arena, align 8
  ; round size up to 16 (keeps the `used` cursor 16-aligned), overflow-checked
  %sz = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 15)
  %sz.up = extractvalue { i64, i1 } %sz, 0
  %sz.ovf = extractvalue { i64, i1 } %sz, 1
  %sz.rounded = and i64 %sz.up, -16
  %nxt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %used, i64 %sz.rounded)
  %next = extractvalue { i64, i1 } %nxt, 0
  %nxt.ovf = extractvalue { i64, i1 } %nxt, 1
  %any.ovf = or i1 %sz.ovf, %nxt.ovf
  %cap.p = getelementptr inbounds nuw i8, ptr %arena, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %over.cap = icmp ugt i64 %next, %cap
  %fail = or i1 %any.ovf, %over.cap
  br i1 %fail, label %exhausted, label %bump, !prof !0

bump:
  store i64 %next, ptr %arena, align 8
  %payload = getelementptr inbounds nuw i8, ptr %arena, i64 64
  %block = getelementptr inbounds nuw i8, ptr %payload, i64 %used
  ret ptr %block

exhausted:
  ret ptr null
}

define ptr @universe_alloc_arena_alloc_aligned(ptr %arena, i64 %size, i64 %align) local_unnamed_addr #0 {
entry:
  ; align must be a nonzero power of two; larger of (align,16) is honored
  %a.min = call i64 @llvm.umax.i64(i64 %align, i64 16)
  %used = load i64, ptr %arena, align 8
  %payload.i = ptrtoint ptr %arena to i64
  %base = add i64 %payload.i, 64
  %cursor = add i64 %base, %used
  ; aligned = (cursor + a-1) & -a
  %a.m1 = add i64 %a.min, -1
  %sum = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %cursor, i64 %a.m1)
  %sum.v = extractvalue { i64, i1 } %sum, 0
  %sum.o = extractvalue { i64, i1 } %sum, 1
  %a.neg = sub i64 0, %a.min
  %aligned = and i64 %sum.v, %a.neg
  %new.used.base = sub i64 %aligned, %base
  ; round size to 16 to preserve the cursor invariant
  %sz = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 15)
  %sz.up = extractvalue { i64, i1 } %sz, 0
  %sz.o = extractvalue { i64, i1 } %sz, 1
  %sz.rounded = and i64 %sz.up, -16
  %nu = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %new.used.base, i64 %sz.rounded)
  %new.used = extractvalue { i64, i1 } %nu, 0
  %nu.o = extractvalue { i64, i1 } %nu, 1
  %o01 = or i1 %sum.o, %sz.o
  %o = or i1 %o01, %nu.o
  %cap.p = getelementptr inbounds nuw i8, ptr %arena, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %over = icmp ugt i64 %new.used, %cap
  %fail = or i1 %o, %over
  br i1 %fail, label %exhausted, label %bump, !prof !0

bump:
  store i64 %new.used, ptr %arena, align 8
  ; reach the aligned block by GEP off the real arena pointer (provenance
  ; preserved) rather than inttoptr of the computed address.
  %block.off = add nuw i64 %new.used.base, 64
  %block = getelementptr inbounds nuw i8, ptr %arena, i64 %block.off
  ret ptr %block

exhausted:
  ret ptr null
}

define void @universe_alloc_arena_reset(ptr %arena) local_unnamed_addr #0 {
entry:
  store i64 0, ptr %arena, align 8
  ret void
}

define i64 @universe_alloc_arena_used(ptr %arena) local_unnamed_addr #2 {
entry:
  %used = load i64, ptr %arena, align 8
  ret i64 %used
}

define i64 @universe_alloc_arena_capacity(ptr %arena) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %arena, i64 8
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define void @universe_alloc_arena_destroy(ptr %arena) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %arena, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %arena)
  br label %done

done:
  ret void
}

declare i64 @llvm.umax.i64(i64, i64)

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
