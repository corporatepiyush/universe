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

; Singly-linked list with chunked node pool and zero-copy data access.
;
; DESIGN (vs the typical C implementation):
;   * Same chunked pool as dlist (one malloc per 256 nodes, intrusive free
;     list, wilderness cursor) but nodes are half the overhead:
;     { next@0, data@8 }, stride = 8 + round8(elem).
;   * Idiomatic slist ops only: push_front / push_back (O(1) via tail ptr) /
;     pop_front / insert_after / remove_after. No O(n) arbitrary remove.
;   * Header: head@0 tail@8 free@16 chunks@24 count@32 elem@40 stride@48
;             fresh@56 fresh_end@64.
;
; API:
;   ptr universe_ds_slist_create(i64 elem_size)
;   ptr universe_ds_slist_push_front(ptr l, ptr elem)  ; -> node | null
;   ptr universe_ds_slist_push_back(ptr l, ptr elem)
;   ptr universe_ds_slist_insert_after(ptr l, ptr node, ptr elem)
;   i32 universe_ds_slist_pop_front(ptr l, ptr out)    ; 4 EMPTY
;   i32 universe_ds_slist_remove_after(ptr l, ptr node, ptr out) ; 5 NOT_FOUND if no successor
;   ptr universe_ds_slist_first(ptr l)
;   ptr universe_ds_slist_next(ptr node)
;   ptr universe_ds_slist_data(ptr node)
;   i64 universe_ds_slist_count(ptr l)
;   void universe_ds_slist_destroy(ptr l)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

define noalias ptr @universe_ds_slist_create(i64 %elem_size) local_unnamed_addr #1 {
entry:
  %elem.bad = icmp eq i64 %elem_size, 0
  %too.big = icmp ugt i64 %elem_size, 1073741824
  %bad = or i1 %elem.bad, %too.big
  br i1 %bad, label %fail, label %alloc, !prof !0

alloc:
  %hdr = call ptr @malloc(i64 72)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %init, !prof !0

init:
  store ptr null, ptr %hdr, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store ptr null, ptr %tail.p, align 8
  %free.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store ptr null, ptr %free.p, align 8
  %chunks.p = getelementptr inbounds nuw i8, ptr %hdr, i64 24
  store ptr null, ptr %chunks.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %hdr, i64 32
  store i64 0, ptr %count.p, align 8
  %elem.p = getelementptr inbounds nuw i8, ptr %hdr, i64 40
  store i64 %elem_size, ptr %elem.p, align 8
  %stride.p = getelementptr inbounds nuw i8, ptr %hdr, i64 48
  %e7 = add i64 %elem_size, 7
  %e.r = and i64 %e7, -8
  %stride = add i64 %e.r, 8
  store i64 %stride, ptr %stride.p, align 8
  %fresh.p = getelementptr inbounds nuw i8, ptr %hdr, i64 56
  store ptr null, ptr %fresh.p, align 8
  %fend.p = getelementptr inbounds nuw i8, ptr %hdr, i64 64
  store ptr null, ptr %fend.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

define internal ptr @snode_take(ptr %l) #3 {
entry:
  %free.p = getelementptr inbounds nuw i8, ptr %l, i64 16
  %freehead = load ptr, ptr %free.p, align 8
  %have.free = icmp ne ptr %freehead, null
  br i1 %have.free, label %pop, label %wilderness

pop:
  %next = load ptr, ptr %freehead, align 8
  store ptr %next, ptr %free.p, align 8
  ret ptr %freehead

wilderness:
  %fresh.p = getelementptr inbounds nuw i8, ptr %l, i64 56
  %fresh = load ptr, ptr %fresh.p, align 8
  %fend.p = getelementptr inbounds nuw i8, ptr %l, i64 64
  %fend = load ptr, ptr %fend.p, align 8
  %fresh.i = ptrtoint ptr %fresh to i64
  %fend.i = ptrtoint ptr %fend to i64
  %have.fresh = icmp ult i64 %fresh.i, %fend.i
  br i1 %have.fresh, label %carve, label %grow

carve:
  %stride.p = getelementptr inbounds nuw i8, ptr %l, i64 48
  %stride = load i64, ptr %stride.p, align 8
  %fresh.n = getelementptr inbounds nuw i8, ptr %fresh, i64 %stride
  store ptr %fresh.n, ptr %fresh.p, align 8
  ret ptr %fresh

grow:
  %stride.p2 = getelementptr inbounds nuw i8, ptr %l, i64 48
  %stride2 = load i64, ptr %stride.p2, align 8
  %body = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %stride2, i64 256)
  %body.v = extractvalue { i64, i1 } %body, 0
  %body.o = extractvalue { i64, i1 } %body, 1
  %tot = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %body.v, i64 16)
  %total = extractvalue { i64, i1 } %tot, 0
  %tot.o = extractvalue { i64, i1 } %tot, 1
  %ovf = or i1 %body.o, %tot.o
  br i1 %ovf, label %fail, label %chunk.alloc

chunk.alloc:
  %chunk = call ptr @malloc(i64 %total)
  %chunk.null = icmp eq ptr %chunk, null
  br i1 %chunk.null, label %fail, label %chunk.link

chunk.link:
  %chunks.p = getelementptr inbounds nuw i8, ptr %l, i64 24
  %old.chunks = load ptr, ptr %chunks.p, align 8
  store ptr %old.chunks, ptr %chunk, align 8
  store ptr %chunk, ptr %chunks.p, align 8
  %first = getelementptr inbounds nuw i8, ptr %chunk, i64 16
  %second = getelementptr inbounds nuw i8, ptr %first, i64 %stride2
  %end = getelementptr inbounds nuw i8, ptr %chunk, i64 %total
  %fresh.p2 = getelementptr inbounds nuw i8, ptr %l, i64 56
  store ptr %second, ptr %fresh.p2, align 8
  %fend.p2 = getelementptr inbounds nuw i8, ptr %l, i64 64
  store ptr %end, ptr %fend.p2, align 8
  ret ptr %first

fail:
  ret ptr null
}

define internal void @snode_fill(ptr %l, ptr %node, ptr %elem) #0 {
entry:
  %elem.p = getelementptr inbounds nuw i8, ptr %l, i64 40
  %esz = load i64, ptr %elem.p, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %node, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr %data.p, ptr %elem, i64 %esz, i1 false)
  %count.p = getelementptr inbounds nuw i8, ptr %l, i64 32
  %count = load i64, ptr %count.p, align 8
  %count.n = add nuw i64 %count, 1
  store i64 %count.n, ptr %count.p, align 8
  ret void
}

define ptr @universe_ds_slist_push_front(ptr %l, ptr %elem) local_unnamed_addr #1 {
entry:
  %l.null = icmp eq ptr %l, null
  %e.null = icmp eq ptr %elem, null
  %any.null = or i1 %l.null, %e.null
  br i1 %any.null, label %fail, label %take, !prof !0

take:
  %node = call ptr @snode_take(ptr nonnull %l)
  %n.null = icmp eq ptr %node, null
  br i1 %n.null, label %fail, label %link, !prof !0

link:
  call void @snode_fill(ptr nonnull %l, ptr nonnull %node, ptr %elem)
  %head = load ptr, ptr %l, align 8
  store ptr %head, ptr %node, align 8
  store ptr %node, ptr %l, align 8
  %was.empty = icmp eq ptr %head, null
  br i1 %was.empty, label %set.tail, label %done

set.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %l, i64 8
  store ptr %node, ptr %tail.p, align 8
  br label %done

done:
  ret ptr %node

fail:
  ret ptr null
}

define ptr @universe_ds_slist_push_back(ptr %l, ptr %elem) local_unnamed_addr #1 {
entry:
  %l.null = icmp eq ptr %l, null
  %e.null = icmp eq ptr %elem, null
  %any.null = or i1 %l.null, %e.null
  br i1 %any.null, label %fail, label %take, !prof !0

take:
  %node = call ptr @snode_take(ptr nonnull %l)
  %n.null = icmp eq ptr %node, null
  br i1 %n.null, label %fail, label %link, !prof !0

link:
  call void @snode_fill(ptr nonnull %l, ptr nonnull %node, ptr %elem)
  store ptr null, ptr %node, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %l, i64 8
  %tail = load ptr, ptr %tail.p, align 8
  %was.empty = icmp eq ptr %tail, null
  br i1 %was.empty, label %set.head, label %chain

chain:
  store ptr %node, ptr %tail, align 8
  br label %set.tail

set.head:
  store ptr %node, ptr %l, align 8
  br label %set.tail

set.tail:
  store ptr %node, ptr %tail.p, align 8
  ret ptr %node

fail:
  ret ptr null
}

define ptr @universe_ds_slist_insert_after(ptr %l, ptr %node, ptr %elem) local_unnamed_addr #1 {
entry:
  %l.null = icmp eq ptr %l, null
  %n.null = icmp eq ptr %node, null
  %e.null = icmp eq ptr %elem, null
  %n01 = or i1 %l.null, %n.null
  %any.null = or i1 %n01, %e.null
  br i1 %any.null, label %fail, label %take, !prof !0

take:
  %fresh = call ptr @snode_take(ptr nonnull %l)
  %f.null = icmp eq ptr %fresh, null
  br i1 %f.null, label %fail, label %link, !prof !0

link:
  call void @snode_fill(ptr nonnull %l, ptr nonnull %fresh, ptr %elem)
  %after = load ptr, ptr %node, align 8
  store ptr %after, ptr %fresh, align 8
  store ptr %fresh, ptr %node, align 8
  %was.tail = icmp eq ptr %after, null
  br i1 %was.tail, label %set.tail, label %done

set.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %l, i64 8
  store ptr %fresh, ptr %tail.p, align 8
  br label %done

done:
  ret ptr %fresh

fail:
  ret ptr null
}

; unlink %victim whose PREDECESSOR is %pred (null pred = victim is head)
define internal void @sunlink(ptr %l, ptr %pred, ptr %victim, ptr %out) #0 {
entry:
  %next = load ptr, ptr %victim, align 8
  %want.out = icmp ne ptr %out, null
  br i1 %want.out, label %copy.out, label %relink

copy.out:
  %elem.p = getelementptr inbounds nuw i8, ptr %l, i64 40
  %esz = load i64, ptr %elem.p, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %victim, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull %out, ptr %data.p, i64 %esz, i1 false)
  br label %relink

relink:
  %p.null = icmp eq ptr %pred, null
  br i1 %p.null, label %fix.head, label %fix.pred

fix.pred:
  store ptr %next, ptr %pred, align 8
  br label %tailcheck

fix.head:
  store ptr %next, ptr %l, align 8
  br label %tailcheck

tailcheck:
  %was.tail = icmp eq ptr %next, null
  br i1 %was.tail, label %fix.tail, label %recycle

fix.tail:
  %tail.p = getelementptr inbounds nuw i8, ptr %l, i64 8
  store ptr %pred, ptr %tail.p, align 8
  br label %recycle

recycle:
  %free.p = getelementptr inbounds nuw i8, ptr %l, i64 16
  %freehead = load ptr, ptr %free.p, align 8
  store ptr %freehead, ptr %victim, align 8
  store ptr %victim, ptr %free.p, align 8
  %count.p = getelementptr inbounds nuw i8, ptr %l, i64 32
  %count = load i64, ptr %count.p, align 8
  %count.n = add i64 %count, -1
  store i64 %count.n, ptr %count.p, align 8
  ret void
}

define i32 @universe_ds_slist_pop_front(ptr %l, ptr %out) local_unnamed_addr #1 {
entry:
  %l.null = icmp eq ptr %l, null
  br i1 %l.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %head = load ptr, ptr %l, align 8
  %empty = icmp eq ptr %head, null
  br i1 %empty, label %err.empty, label %take, !prof !0

err.empty:
  ret i32 4

take:
  call void @sunlink(ptr nonnull %l, ptr null, ptr nonnull %head, ptr %out)
  ret i32 0
}

define i32 @universe_ds_slist_remove_after(ptr %l, ptr %node, ptr %out) local_unnamed_addr #1 {
entry:
  %l.null = icmp eq ptr %l, null
  %n.null = icmp eq ptr %node, null
  %any.null = or i1 %l.null, %n.null
  br i1 %any.null, label %err.null, label %check, !prof !0

err.null:
  ret i32 1

check:
  %victim = load ptr, ptr %node, align 8
  %none = icmp eq ptr %victim, null
  br i1 %none, label %err.nf, label %take, !prof !0

err.nf:
  ret i32 5

take:
  call void @sunlink(ptr nonnull %l, ptr nonnull %node, ptr nonnull %victim, ptr %out)
  ret i32 0
}

define ptr @universe_ds_slist_first(ptr %l) local_unnamed_addr #2 {
entry:
  %head = load ptr, ptr %l, align 8
  ret ptr %head
}

define ptr @universe_ds_slist_next(ptr %node) local_unnamed_addr #2 {
entry:
  %next = load ptr, ptr %node, align 8
  ret ptr %next
}

define ptr @universe_ds_slist_data(ptr %node) local_unnamed_addr #2 {
entry:
  %data.p = getelementptr inbounds nuw i8, ptr %node, i64 8
  ret ptr %data.p
}

define i64 @universe_ds_slist_count(ptr %l) local_unnamed_addr #2 {
entry:
  %count.p = getelementptr inbounds nuw i8, ptr %l, i64 32
  %count = load i64, ptr %count.p, align 8
  ret i64 %count
}

define void @universe_ds_slist_destroy(ptr %l) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %l, null
  br i1 %is.null, label %done, label %walk.pre, !prof !0

walk.pre:
  %chunks.p = getelementptr inbounds nuw i8, ptr %l, i64 24
  %chunks = load ptr, ptr %chunks.p, align 8
  br label %walk

walk:
  %c = phi ptr [ %chunks, %walk.pre ], [ %next, %walk.body ]
  %c.null = icmp eq ptr %c, null
  br i1 %c.null, label %free.hdr, label %walk.body

walk.body:
  %next = load ptr, ptr %c, align 8
  call void @free(ptr nonnull %c)
  br label %walk

free.hdr:
  call void @free(ptr nonnull %l)
  br label %done

done:
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn }

!0 = !{!"branch_weights", i32 1, i32 2000}
