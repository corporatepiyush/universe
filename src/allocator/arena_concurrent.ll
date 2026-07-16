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

; Concurrent arena: WAIT-FREE bump allocation from any number of threads.
;
; DESIGN (vs the typical C implementation):
;   * A mutex-guarded arena serializes every alloc. Here the entire alloc is
;     ONE `atomicrmw add` (a single LDADD instruction on AArch64 with LSE,
;     lock xadd on x86) plus a bounds check — wait-free, no convoy, no
;     syscall on contention.
;   * Ordering is `monotonic` and that is sufficient: the allocator hands
;     each caller a DISJOINT region; no data is published through the cursor
;     itself. Callers who share the block with other threads synchronize via
;     their own release/acquire, as with malloc.
;   * Over-reservation on a failed (exhausted) alloc is rolled back with an
;     atomic sub so `used` stays accurate under churn at the boundary.
;   * Header: { atomic i64 used @0, i64 capacity @8 }, payload at +128 —
;     the RMW-hot line never false-shares with user data (128B covers
;     Apple-silicon L2 line pairs).
;   * reset() is documented QUIESCENT-ONLY (no concurrent allocs), matching
;     the C contract.
;
; API:
;   ptr  universe_alloc_carena_create(i64 capacity)
;   ptr  universe_alloc_carena_alloc(ptr a, i64 size)      ; 16-aligned
;   ptr  universe_alloc_carena_alloc_aligned(ptr a, i64 size, i64 align)
;   void universe_alloc_carena_reset(ptr a)                ; quiescent only
;   i64  universe_alloc_carena_used(ptr a)
;   i64  universe_alloc_carena_capacity(ptr a)
;   void universe_alloc_carena_destroy(ptr a)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)

define noalias ptr @universe_alloc_carena_create(i64 %capacity) local_unnamed_addr #1 {
entry:
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %capacity, i64 128)
  %total = extractvalue { i64, i1 } %tot, 0
  %ovf = extractvalue { i64, i1 } %tot, 1
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store atomic i64 0, ptr %mem monotonic, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %capacity, ptr %cap.p, align 8
  ret ptr %mem

fail:
  ret ptr null
}

define ptr @universe_alloc_carena_alloc(ptr %arena, i64 %size) local_unnamed_addr #0 {
entry:
  %sz = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 15)
  %sz.up = extractvalue { i64, i1 } %sz, 0
  %sz.o = extractvalue { i64, i1 } %sz, 1
  br i1 %sz.o, label %exhausted, label %reserve, !prof !0

reserve:
  %sz.rounded = and i64 %sz.up, -16
  %old = atomicrmw add ptr %arena, i64 %sz.rounded monotonic, align 8
  %nw = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %old, i64 %sz.rounded)
  %new = extractvalue { i64, i1 } %nw, 0
  %nw.o = extractvalue { i64, i1 } %nw, 1
  %cap.p = getelementptr inbounds nuw i8, ptr %arena, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %over = icmp ugt i64 %new, %cap
  %fail = or i1 %nw.o, %over
  br i1 %fail, label %rollback, label %ok, !prof !0

ok:
  %payload = getelementptr inbounds nuw i8, ptr %arena, i64 128
  %block = getelementptr inbounds nuw i8, ptr %payload, i64 %old
  ret ptr %block

rollback:
  %undo = atomicrmw sub ptr %arena, i64 %sz.rounded monotonic, align 8
  br label %exhausted

exhausted:
  ret ptr null
}

define ptr @universe_alloc_carena_alloc_aligned(ptr %arena, i64 %size, i64 %align) local_unnamed_addr #0 {
entry:
  ; wait-free strategy: over-allocate size + (align-16), then round the
  ; returned address up. Wastes < align bytes; never loops.
  %a.min = call i64 @llvm.umax.i64(i64 %align, i64 16)
  %slack = add i64 %a.min, -16
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %size, i64 %slack)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  br i1 %tot.o, label %fail, label %do.alloc, !prof !0

do.alloc:
  %raw = call ptr @universe_alloc_carena_alloc(ptr %arena, i64 %total)
  %raw.null = icmp eq ptr %raw, null
  br i1 %raw.null, label %fail, label %round, !prof !0

round:                                        ; round %raw up to %a.min
  %raw.i = ptrtoint ptr %raw to i64
  %a.m1 = add i64 %a.min, -1
  %sum = add i64 %raw.i, %a.m1
  %a.neg = sub i64 0, %a.min
  %aligned = and i64 %sum, %a.neg
  ; GEP the delta off the real %raw pointer (provenance preserved) instead of
  ; inttoptr of the aligned address.
  %delta = sub i64 %aligned, %raw.i
  %block = getelementptr inbounds nuw i8, ptr %raw, i64 %delta
  ret ptr %block

fail:
  ret ptr null
}

define void @universe_alloc_carena_reset(ptr %arena) local_unnamed_addr #0 {
entry:
  store atomic i64 0, ptr %arena monotonic, align 8
  ret void
}

define i64 @universe_alloc_carena_used(ptr %arena) local_unnamed_addr #2 {
entry:
  %used = load atomic i64, ptr %arena monotonic, align 8
  ret i64 %used
}

define i64 @universe_alloc_carena_capacity(ptr %arena) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %arena, i64 8
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define void @universe_alloc_carena_destroy(ptr %arena) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %arena, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %arena)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
