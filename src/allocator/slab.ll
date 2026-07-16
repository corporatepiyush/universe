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

; Slab allocator: growable, fixed-size objects, O(1) alloc/free, empty slabs
; returned to the OS.
;
; DESIGN (vs the typical C implementation):
;   * Slabs are allocated with posix_memalign(span, span) where span is a
;     power of two — freeing an object finds its slab with ONE and:
;     slab = ptr & -span. No per-object headers, no slab search, no division.
;   * Per-slab intrusive free list of byte offsets + wilderness bump cursor
;     (same O(1)-create trick as the pool: fresh slabs never pre-link).
;   * Slabs form a doubly-linked list; a slab that goes empty is unlinked and
;     free()d in O(1) — memory actually returns, unlike a naive implementation.
;   * `current` points at the last slab that had space (freed-into slabs
;     become current: they're cache-hot).
;   * Handle: { ptr current, ptr head, i64 stride, i64 span, i64 live, i64 objs }
;     Slab header (64B): { i64 free_head_off, i64 next_fresh_off,
;                          i64 live_in_slab, ptr next, ptr prev }
;     objects at slab+64.
;
; API:
;   ptr  universe_alloc_slab_create(i64 obj_size, i64 objs_per_slab) ; objs clamped to [1,512]
;   ptr  universe_alloc_slab_alloc(ptr s)
;   void universe_alloc_slab_free(ptr s, ptr obj)
;   i64  universe_alloc_slab_live(ptr s)
;   void universe_alloc_slab_destroy(ptr s)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @posix_memalign(ptr, i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
declare ptr @llvm.ptrmask.p0.i64(ptr, i64)

; handle offsets: current 0, head 8, stride 16, span 24, live 32, objs 40

define noalias ptr @universe_alloc_slab_create(i64 %obj_size, i64 %objs_per_slab) local_unnamed_addr #1 {
entry:
  %sz = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %obj_size, i64 15)
  %sz.up = extractvalue { i64, i1 } %sz, 0
  %sz.o = extractvalue { i64, i1 } %sz, 1
  br i1 %sz.o, label %fail, label %clamp, !prof !0

clamp:
  %sz.rounded = and i64 %sz.up, -16
  %stride = call i64 @llvm.umax.i64(i64 %sz.rounded, i64 16)
  %objs.min = call i64 @llvm.umax.i64(i64 %objs_per_slab, i64 1)
  %objs = call i64 @llvm.umin.i64(i64 %objs.min, i64 512)
  ; span = next_pow2(64 + objs*stride)
  %body = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %stride, i64 %objs)
  %body.v = extractvalue { i64, i1 } %body, 0
  %body.o = extractvalue { i64, i1 } %body, 1
  %need = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %body.v, i64 64)
  %need.v = extractvalue { i64, i1 } %need, 0
  %need.o = extractvalue { i64, i1 } %need, 1
  %o01 = or i1 %body.o, %need.o
  ; refuse spans over 2^40 (1 TiB slab is a bug, and pow2 math must not wrap)
  %too.big = icmp ugt i64 %need.v, 1099511627776
  %ovf = or i1 %o01, %too.big
  br i1 %ovf, label %fail, label %pow2, !prof !0

pow2:
  %nm1 = add i64 %need.v, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %nm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %span = shl nuw i64 1, %shift
  %h = call ptr @malloc(i64 48)
  %h.null = icmp eq ptr %h, null
  br i1 %h.null, label %fail, label %init, !prof !0

init:
  store ptr null, ptr %h, align 8                     ; current
  %head.p = getelementptr inbounds nuw i8, ptr %h, i64 8
  store ptr null, ptr %head.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  store i64 %stride, ptr %stride.p, align 8
  %span.p = getelementptr inbounds nuw i8, ptr %h, i64 24
  store i64 %span, ptr %span.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %h, i64 32
  store i64 0, ptr %live.p, align 8
  %objs.p = getelementptr inbounds nuw i8, ptr %h, i64 40
  store i64 %objs, ptr %objs.p, align 8
  ret ptr %h

fail:
  ret ptr null
}

; Take one object out of slab %sl (caller guarantees space); returns object.
define internal ptr @slab_take(ptr %sl, ptr %h) #0 {
entry:
  %fh = load i64, ptr %sl, align 8
  %have.free = icmp ne i64 %fh, -1
  br i1 %have.free, label %pop, label %fresh

pop:
  %objbase = getelementptr inbounds nuw i8, ptr %sl, i64 64
  %obj = getelementptr inbounds nuw i8, ptr %objbase, i64 %fh
  %next = load i64, ptr %obj, align 8
  store i64 %next, ptr %sl, align 8
  br label %done

fresh:
  %nf.p = getelementptr inbounds nuw i8, ptr %sl, i64 8
  %nf = load i64, ptr %nf.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %stride = load i64, ptr %stride.p, align 8
  %nf.next = add nuw i64 %nf, %stride
  store i64 %nf.next, ptr %nf.p, align 8
  %objbase2 = getelementptr inbounds nuw i8, ptr %sl, i64 64
  %obj2 = getelementptr inbounds nuw i8, ptr %objbase2, i64 %nf
  br label %done

done:
  %result = phi ptr [ %obj, %pop ], [ %obj2, %fresh ]
  %lis.p = getelementptr inbounds nuw i8, ptr %sl, i64 16
  %lis = load i64, ptr %lis.p, align 8
  %lis.n = add nuw i64 %lis, 1
  store i64 %lis.n, ptr %lis.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %h, i64 32
  %live = load i64, ptr %live.p, align 8
  %live.n = add nuw i64 %live, 1
  store i64 %live.n, ptr %live.p, align 8
  ret ptr %result
}

; Does slab %sl have space? (free list nonempty, or wilderness not exhausted)
define internal i1 @slab_has_space(ptr %sl, ptr %h) #2 {
entry:
  %fh = load i64, ptr %sl, align 8
  %have.free = icmp ne i64 %fh, -1
  %nf.p = getelementptr inbounds nuw i8, ptr %sl, i64 8
  %nf = load i64, ptr %nf.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %h, i64 16
  %stride = load i64, ptr %stride.p, align 8
  %objs.p = getelementptr inbounds nuw i8, ptr %h, i64 40
  %objs = load i64, ptr %objs.p, align 8
  %limit = mul nuw i64 %stride, %objs
  %have.fresh = icmp ult i64 %nf, %limit
  %space = or i1 %have.free, %have.fresh
  ret i1 %space
}

define ptr @universe_alloc_slab_alloc(ptr %s) local_unnamed_addr #1 {
entry:
  %cur = load ptr, ptr %s, align 8
  %cur.null = icmp eq ptr %cur, null
  br i1 %cur.null, label %scan, label %try.cur

try.cur:
  %cur.ok = call i1 @slab_has_space(ptr nonnull %cur, ptr %s)
  br i1 %cur.ok, label %take.cur, label %scan, !prof !1

take.cur:
  %obj = call ptr @slab_take(ptr nonnull %cur, ptr %s)
  ret ptr %obj

scan:                                       ; walk list for a slab with space
  %head.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %head = load ptr, ptr %head.p, align 8
  br label %scan.loop

scan.loop:
  %sl = phi ptr [ %head, %scan ], [ %sl.next, %scan.next ]
  %sl.null = icmp eq ptr %sl, null
  br i1 %sl.null, label %grow, label %scan.check

scan.check:
  %ok = call i1 @slab_has_space(ptr nonnull %sl, ptr %s)
  br i1 %ok, label %scan.hit, label %scan.next

scan.next:
  %next.p = getelementptr inbounds nuw i8, ptr %sl, i64 24
  %sl.next = load ptr, ptr %next.p, align 8
  br label %scan.loop

scan.hit:
  store ptr %sl, ptr %s, align 8            ; current = sl
  %obj2 = call ptr @slab_take(ptr nonnull %sl, ptr %s)
  ret ptr %obj2

grow:                                       ; new slab, aligned to its span
  %span.p = getelementptr inbounds nuw i8, ptr %s, i64 24
  %span = load i64, ptr %span.p, align 8
  %slot = alloca ptr, align 8
  %rc = call i32 @posix_memalign(ptr nonnull %slot, i64 %span, i64 %span)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %fail, label %link, !prof !0

link:
  %new = load ptr, ptr %slot, align 8
  store i64 -1, ptr %new, align 8           ; free_head_off = empty
  %nf.p = getelementptr inbounds nuw i8, ptr %new, i64 8
  store i64 0, ptr %nf.p, align 8           ; next_fresh_off
  %lis.p = getelementptr inbounds nuw i8, ptr %new, i64 16
  store i64 0, ptr %lis.p, align 8          ; live_in_slab
  %head.p2 = getelementptr inbounds nuw i8, ptr %s, i64 8
  %old.head = load ptr, ptr %head.p2, align 8
  %next.p2 = getelementptr inbounds nuw i8, ptr %new, i64 24
  store ptr %old.head, ptr %next.p2, align 8
  %prev.p2 = getelementptr inbounds nuw i8, ptr %new, i64 32
  store ptr null, ptr %prev.p2, align 8
  %oh.null = icmp eq ptr %old.head, null
  br i1 %oh.null, label %set.head, label %backlink

backlink:
  %oh.prev.p = getelementptr inbounds nuw i8, ptr %old.head, i64 32
  store ptr %new, ptr %oh.prev.p, align 8
  br label %set.head

set.head:
  store ptr %new, ptr %head.p2, align 8
  store ptr %new, ptr %s, align 8           ; current = new
  %obj3 = call ptr @slab_take(ptr nonnull %new, ptr %s)
  ret ptr %obj3

fail:
  ret ptr null
}

define void @universe_alloc_slab_free(ptr %s, ptr %obj) local_unnamed_addr #1 {
entry:
  %span.p = getelementptr inbounds nuw i8, ptr %s, i64 24
  %span = load i64, ptr %span.p, align 8
  %obj.i = ptrtoint ptr %obj to i64
  %span.neg = sub i64 0, %span
  %sl.i = and i64 %obj.i, %span.neg              ; integer form for offset math below
  ; mask object DOWN to its span-aligned slab header — llvm.ptrmask keeps
  ; provenance (unlike inttoptr), so the header loads/stores stay alias-analyzable
  %sl = call ptr @llvm.ptrmask.p0.i64(ptr %obj, i64 %span.neg)

  ; push offset onto slab free list
  %fh = load i64, ptr %sl, align 8
  store i64 %fh, ptr %obj, align 8
  %base = add i64 %sl.i, 64
  %off = sub i64 %obj.i, %base
  store i64 %off, ptr %sl, align 8

  %lis.p = getelementptr inbounds nuw i8, ptr %sl, i64 16
  %lis = load i64, ptr %lis.p, align 8
  %lis.n = add i64 %lis, -1
  store i64 %lis.n, ptr %lis.p, align 8
  %live.p = getelementptr inbounds nuw i8, ptr %s, i64 32
  %live = load i64, ptr %live.p, align 8
  %live.n = add i64 %live, -1
  store i64 %live.n, ptr %live.p, align 8

  ; slab empty? release it (unless it's the only place we'd alloc from next)
  %empty = icmp eq i64 %lis.n, 0
  br i1 %empty, label %release, label %retarget, !prof !0

retarget:                                   ; freed-into slab is cache-hot
  store ptr %sl, ptr %s, align 8
  ret void

release:
  ; unlink from doubly-linked list
  %prev.p = getelementptr inbounds nuw i8, ptr %sl, i64 32
  %prev = load ptr, ptr %prev.p, align 8
  %next.p = getelementptr inbounds nuw i8, ptr %sl, i64 24
  %next = load ptr, ptr %next.p, align 8
  %prev.null = icmp eq ptr %prev, null
  br i1 %prev.null, label %fix.head, label %fix.prev

fix.prev:
  %pn.p = getelementptr inbounds nuw i8, ptr %prev, i64 24
  store ptr %next, ptr %pn.p, align 8
  br label %fix.next

fix.head:
  %head.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  store ptr %next, ptr %head.p, align 8
  br label %fix.next

fix.next:
  %next.null = icmp eq ptr %next, null
  br i1 %next.null, label %fix.cur, label %fix.next.prev

fix.next.prev:
  %np.p = getelementptr inbounds nuw i8, ptr %next, i64 32
  store ptr %prev, ptr %np.p, align 8
  br label %fix.cur

fix.cur:                                    ; current must not dangle
  %cur = load ptr, ptr %s, align 8
  %was.cur = icmp eq ptr %cur, %sl
  br i1 %was.cur, label %clear.cur, label %do.release

clear.cur:
  store ptr null, ptr %s, align 8
  br label %do.release

do.release:
  call void @free(ptr nonnull %sl)
  ret void
}

define i64 @universe_alloc_slab_live(ptr %s) local_unnamed_addr #2 {
entry:
  %live.p = getelementptr inbounds nuw i8, ptr %s, i64 32
  %live = load i64, ptr %live.p, align 8
  ret i64 %live
}

define void @universe_alloc_slab_destroy(ptr %s) local_unnamed_addr #1 {
entry:
  %s.null = icmp eq ptr %s, null
  br i1 %s.null, label %done, label %walk.pre, !prof !0

walk.pre:
  %head.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %head = load ptr, ptr %head.p, align 8
  br label %walk

walk:
  %sl = phi ptr [ %head, %walk.pre ], [ %next, %walk.body ]
  %sl.null = icmp eq ptr %sl, null
  br i1 %sl.null, label %free.handle, label %walk.body

walk.body:
  %next.p = getelementptr inbounds nuw i8, ptr %sl, i64 24
  %next = load ptr, ptr %next.p, align 8
  call void @free(ptr nonnull %sl)
  br label %walk

free.handle:
  call void @free(ptr nonnull %s)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 2000, i32 1}
