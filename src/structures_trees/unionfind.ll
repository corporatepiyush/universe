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

; Disjoint-set union (union-find) over the elements 0..n-1. Near-constant
; amortized find/unite via path halving + union by size.
;
; DESIGN (from first principles):
;   * ONE allocation: 64B header + two flat i64 arrays (parent[n], size[n]).
;     parent[] is the forest; size[] holds the subtree count AT A ROOT (stale
;     for non-roots, never read there). No node pool, no pointers — element
;     ids index the arrays directly, so a find is a pure pointer-chase over
;     contiguous memory.
;   * PATH HALVING (single pass, no recursion): as we climb, we repoint each
;     node to its grandparent (parent[cur] = parent[parent[cur]]); cur then
;     advances to that grandparent. This halves the path length every call
;     and needs no second pass and no stack — strictly cheaper than full path
;     compression for the same asymptotics.
;   * UNION BY SIZE keeps trees shallow: the smaller root is hung under the
;     larger, and the survivor's size becomes the sum. num_sets is maintained
;     so count_sets() is O(1).
;   * Conventions (documented, since some entries return values not codes):
;       find(x)      -> representative id, or -1 for null/out-of-range.
;       unite(a,b)   -> 0 merged, 9 already-same-set (a note, not an error),
;                       1 null handle, 7 out-of-range id.
;       connected    -> 1 connected, 0 not, -1 null/out-of-range.
;       set_size(x)  -> size of x's set, or 0 for null/out-of-range.
;   * create size math overflow-checked (umul/uadd.with.overflow → null).
;   * Layout: { i64 n@0, i64 num_sets@8, 48B pad }, parent i64[n]@64,
;     size i64[n]@(64+n*8).

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

define noalias ptr @universe_ds_unionfind_create(i64 %n) local_unnamed_addr #1 {
entry:
  ; bytes for both arrays = n * 16
  %ab = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %n, i64 16)
  %arrbytes = extractvalue { i64, i1 } %ab, 0
  %ab.o = extractvalue { i64, i1 } %ab, 1
  %tt = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %arrbytes, i64 64)
  %total = extractvalue { i64, i1 } %tt, 0
  %tt.o = extractvalue { i64, i1 } %tt, 1
  %ovf = or i1 %ab.o, %tt.o
  br i1 %ovf, label %fail, label %alloc, !prof !0

alloc:
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  store i64 %n, ptr %mem, align 8
  %ns.p = getelementptr inbounds nuw i8, ptr %mem, i64 8
  store i64 %n, ptr %ns.p, align 8
  %parent = getelementptr inbounds nuw i8, ptr %mem, i64 64
  %size = getelementptr inbounds i64, ptr %parent, i64 %n
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %ret, label %fill

fill:
  %i = phi i64 [ 0, %init ], [ %i.n, %fill ]
  %pp = getelementptr inbounds i64, ptr %parent, i64 %i
  store i64 %i, ptr %pp, align 8
  %sp = getelementptr inbounds i64, ptr %size, i64 %i
  store i64 1, ptr %sp, align 8
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %fill, label %ret

ret:
  ret ptr %mem

fail:
  ret ptr null
}

; internal: representative of x with path halving. Assumes x in range.
define internal i64 @uf_find(ptr captures(none) %parent, i64 %x) #4 {
entry:
  br label %loop

loop:
  %cur = phi i64 [ %x, %entry ], [ %gp, %half ]
  %pp = getelementptr inbounds i64, ptr %parent, i64 %cur
  %p = load i64, ptr %pp, align 8
  %isroot = icmp eq i64 %p, %cur
  br i1 %isroot, label %ret, label %half

half:
  %gpp = getelementptr inbounds i64, ptr %parent, i64 %p
  %gp = load i64, ptr %gpp, align 8
  store i64 %gp, ptr %pp, align 8
  br label %loop

ret:
  ret i64 %cur
}

define i64 @universe_ds_unionfind_find(ptr %u, i64 %x) local_unnamed_addr #0 {
entry:
  %u.null = icmp eq ptr %u, null
  br i1 %u.null, label %err, label %chk, !prof !0

err:
  ret i64 -1

chk:
  %n = load i64, ptr %u, align 8
  %oob = icmp uge i64 %x, %n
  br i1 %oob, label %err, label %go, !prof !0

go:
  %parent = getelementptr inbounds nuw i8, ptr %u, i64 64
  %r = call i64 @uf_find(ptr %parent, i64 %x)
  ret i64 %r
}

define i32 @universe_ds_unionfind_unite(ptr %u, i64 %a, i64 %b) local_unnamed_addr #0 {
entry:
  %u.null = icmp eq ptr %u, null
  br i1 %u.null, label %err.null, label %chk, !prof !0

err.null:
  ret i32 1

chk:
  %n = load i64, ptr %u, align 8
  %oa = icmp uge i64 %a, %n
  %ob = icmp uge i64 %b, %n
  %oob = or i1 %oa, %ob
  br i1 %oob, label %err.idx, label %go, !prof !0

err.idx:
  ret i32 7

go:
  %parent = getelementptr inbounds nuw i8, ptr %u, i64 64
  %ra = call i64 @uf_find(ptr %parent, i64 %a)
  %rb = call i64 @uf_find(ptr %parent, i64 %b)
  %same = icmp eq i64 %ra, %rb
  br i1 %same, label %already, label %merge, !prof !0

already:
  ret i32 9

merge:
  %size = getelementptr inbounds i64, ptr %parent, i64 %n
  %sa.p = getelementptr inbounds i64, ptr %size, i64 %ra
  %sa = load i64, ptr %sa.p, align 8
  %sb.p = getelementptr inbounds i64, ptr %size, i64 %rb
  %sb = load i64, ptr %sb.p, align 8
  %a.bigger = icmp uge i64 %sa, %sb
  %root = select i1 %a.bigger, i64 %ra, i64 %rb
  %child = select i1 %a.bigger, i64 %rb, i64 %ra
  ; parent[child] = root
  %cp = getelementptr inbounds i64, ptr %parent, i64 %child
  store i64 %root, ptr %cp, align 8
  ; size[root] = sa + sb
  %newsz = add i64 %sa, %sb
  %rsp = getelementptr inbounds i64, ptr %size, i64 %root
  store i64 %newsz, ptr %rsp, align 8
  ; num_sets -= 1
  %ns.p = getelementptr inbounds nuw i8, ptr %u, i64 8
  %ns = load i64, ptr %ns.p, align 8
  %ns2 = sub i64 %ns, 1
  store i64 %ns2, ptr %ns.p, align 8
  ret i32 0
}

define i32 @universe_ds_unionfind_connected(ptr %u, i64 %a, i64 %b) local_unnamed_addr #0 {
entry:
  %u.null = icmp eq ptr %u, null
  br i1 %u.null, label %err, label %chk, !prof !0

err:
  ret i32 -1

chk:
  %n = load i64, ptr %u, align 8
  %oa = icmp uge i64 %a, %n
  %ob = icmp uge i64 %b, %n
  %oob = or i1 %oa, %ob
  br i1 %oob, label %err, label %go, !prof !0

go:
  %parent = getelementptr inbounds nuw i8, ptr %u, i64 64
  %ra = call i64 @uf_find(ptr %parent, i64 %a)
  %rb = call i64 @uf_find(ptr %parent, i64 %b)
  %eq = icmp eq i64 %ra, %rb
  %r = zext i1 %eq to i32
  ret i32 %r
}

define i64 @universe_ds_unionfind_count_sets(ptr %u) local_unnamed_addr #2 {
entry:
  %u.null = icmp eq ptr %u, null
  br i1 %u.null, label %ret0, label %go, !prof !0

ret0:
  ret i64 0

go:
  %ns.p = getelementptr inbounds nuw i8, ptr %u, i64 8
  %ns = load i64, ptr %ns.p, align 8
  ret i64 %ns
}

define i64 @universe_ds_unionfind_set_size(ptr %u, i64 %x) local_unnamed_addr #0 {
entry:
  %u.null = icmp eq ptr %u, null
  br i1 %u.null, label %ret0, label %chk, !prof !0

ret0:
  ret i64 0

chk:
  %n = load i64, ptr %u, align 8
  %oob = icmp uge i64 %x, %n
  br i1 %oob, label %ret0, label %go, !prof !0

go:
  %parent = getelementptr inbounds nuw i8, ptr %u, i64 64
  %root = call i64 @uf_find(ptr %parent, i64 %x)
  %size = getelementptr inbounds i64, ptr %parent, i64 %n
  %sp = getelementptr inbounds i64, ptr %size, i64 %root
  %s = load i64, ptr %sp, align 8
  ret i64 %s
}

define void @universe_ds_unionfind_destroy(ptr %u) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %u, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %u)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync memory(argmem: readwrite) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #4 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: readwrite) }

!0 = !{!"branch_weights", i32 1, i32 2000}
