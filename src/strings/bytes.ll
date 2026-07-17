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

; Growable byte buffer: the foundation for strings, encoders and IO. Amortized
; O(1) append via geometric (2x) growth, zero-copy contiguous data view.
;
; DESIGN:
;   * TWO allocations by necessity, not by accident. A growable buffer whose
;     HANDLE must stay stable across growth cannot be a single block: growth
;     reallocates (and may move) the data, but callers hold the handle. So the
;     handle is a small stable header { i64 len, i64 cap, ptr data } and the
;     bytes live in a separately realloc'd block. Only `data` moves on growth
;     (exactly the contract). This mirrors the SoA/handle idiom while honoring
;     the "handle stable, data pointer moves" requirement.
;   * Header is 24 B, well under a cache line; the three hot fields sit
;     together so len()/cap()/data() and the append fast path touch one line.
;   * Growth is a SINGLE realloc (no malloc+memcpy+free by hand): realloc both
;     grows in place when it can and moves+copies when it must, in one call.
;     new_cap = umax(needed, cap*2) — geometric so N appends are amortized O(1).
;   * Append is ONE llvm.memcpy after a single reserve; never a byte loop.
;   * All size math (len+extra, len+n) is overflow-checked -> SIZE_OVERFLOW(3).
;   * cap 0 => data is null; the first grow reallocs from null (== malloc), so
;     an empty buffer costs one 24 B header and no data block until first write.
;
; SIMD scans over the live bytes (find/count/equals/compare/case-fold) DELEGATE
; to the verified 128-bit vector kernels + scalar tail in src/simd/scan.ll
; (DEPS_strings := simd) rather than a scalar byte loop — the CLAUDE "reuse the
; shared scan primitive" rule. The scalar `*_scalar` twins in scan.ll remain the
; cross-check oracle. Cross-module this is a real `bl` (blocks inlining the
; vector body into the caller) in the non-LTO build; acceptable — these are not
; proven hot leaves and the vector loop dominates; the LTO/unity variant inlines
; the seam away with no source change. len==0 => data may be null, but every
; kernel reads nothing at n==0, so the null data is never dereferenced.
;
; API (0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX):
;   ptr  universe_bytes_create(i64 initial_cap)      ; null on OOM/overflow
;   i64  universe_bytes_len(ptr b)
;   i64  universe_bytes_cap(ptr b)
;   ptr  universe_bytes_data(ptr b)                  ; contiguous zero-copy view
;   i32  universe_bytes_append(ptr b, ptr src, i64 n); grows, one memcpy
;   i32  universe_bytes_append_byte(ptr b, i8 v)
;   i32  universe_bytes_reserve(ptr b, i64 extra)    ; ensure room for `extra`
;   void universe_bytes_clear(ptr b)                 ; len=0, keep cap
;   i32  universe_bytes_truncate(ptr b, i64 newlen)  ; newlen<=len else 7
;   void universe_bytes_destroy(ptr b)
;   SIMD scans (b==null => empty sentinel: -1 / 0 / false / 0 / no-op):
;   i64  universe_bytes_index_of_byte(ptr b, i8 c)   ; first idx or -1 (memchr)
;   i64  universe_bytes_count_byte(ptr b, i8 c)      ; # of matches
;   i1   universe_bytes_equals(ptr a, ptr b)         ; byte-equal
;   i32  universe_bytes_compare(ptr a, ptr b)        ; memcmp sign, len tiebreak
;   void universe_bytes_to_lower(ptr b)              ; ASCII A-Z -> a-z in place
;   void universe_bytes_to_upper(ptr b)              ; ASCII a-z -> A-Z in place

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare ptr @realloc(ptr allocptr, i64) allockind("realloc") allocsize(1) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)

; SIMD byte kernels (src/simd/scan.ll): 128-bit vector primary + scalar tail.
declare i64 @universe_simd_find_byte(ptr readonly, i64, i8)
declare i64 @universe_simd_count_byte(ptr readonly, i64, i8)
declare i1  @universe_simd_equal(ptr readonly, ptr readonly, i64)
declare i32 @universe_simd_compare(ptr readonly, ptr readonly, i64)
declare void @universe_simd_to_lower_ascii(ptr, ptr, i64)
declare void @universe_simd_to_upper_ascii(ptr, ptr, i64)

; Ensure cap >= needed. Assumes b non-null. Returns 0 or 2 (OOM).
define internal i32 @bytes_grow(ptr %b, i64 %needed) #3 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %b, i64 8
  %cap = load i64, ptr %cap.p, align 8
  %fits = icmp ule i64 %needed, %cap
  br i1 %fits, label %ok, label %grow, !prof !1

grow:
  ; new_cap = max(needed, cap*2). plain shl may wrap; umax with needed keeps
  ; the result >= needed, so a wrapped double never yields an undersized cap.
  %dbl = shl i64 %cap, 1
  %newcap = call i64 @llvm.umax.i64(i64 %needed, i64 %dbl)
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %old = load ptr, ptr %data.p, align 8
  %new = call ptr @realloc(ptr %old, i64 %newcap)
  %null = icmp eq ptr %new, null
  br i1 %null, label %oom, label %store, !prof !1

store:
  store ptr %new, ptr %data.p, align 8
  store i64 %newcap, ptr %cap.p, align 8
  ret i32 0

oom:
  ret i32 2

ok:
  ret i32 0
}

define noalias ptr @universe_bytes_create(i64 %initial_cap) local_unnamed_addr #1 {
entry:
  %hdr = call ptr @malloc(i64 24)
  %hdr.null = icmp eq ptr %hdr, null
  br i1 %hdr.null, label %fail, label %maybe.data, !prof !1

maybe.data:
  %has.cap = icmp ne i64 %initial_cap, 0
  br i1 %has.cap, label %alloc.data, label %init

alloc.data:
  %data = call ptr @malloc(i64 %initial_cap)
  %data.null = icmp eq ptr %data, null
  br i1 %data.null, label %free.hdr, label %init, !prof !1

free.hdr:
  call void @free(ptr nonnull %hdr)
  br label %fail

init:
  %d = phi ptr [ null, %maybe.data ], [ %data, %alloc.data ]
  store i64 0, ptr %hdr, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %hdr, i64 8
  store i64 %initial_cap, ptr %cap.p, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %hdr, i64 16
  store ptr %d, ptr %data.p, align 8
  ret ptr %hdr

fail:
  ret ptr null
}

define i64 @universe_bytes_len(ptr %b) local_unnamed_addr #2 {
entry:
  %len = load i64, ptr %b, align 8
  ret i64 %len
}

define i64 @universe_bytes_cap(ptr %b) local_unnamed_addr #2 {
entry:
  %cap.p = getelementptr inbounds nuw i8, ptr %b, i64 8
  %cap = load i64, ptr %cap.p, align 8
  ret i64 %cap
}

define ptr @universe_bytes_data(ptr %b) local_unnamed_addr #2 {
entry:
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  ret ptr %data
}

define i32 @universe_bytes_reserve(ptr %b, i64 %extra) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %err.null, label %work, !prof !1

err.null:
  ret i32 1

work:
  %len = load i64, ptr %b, align 8
  %add = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %len, i64 %extra)
  %needed = extractvalue { i64, i1 } %add, 0
  %ovf = extractvalue { i64, i1 } %add, 1
  br i1 %ovf, label %err.ovf, label %do.grow, !prof !1

err.ovf:
  ret i32 3

do.grow:
  %r = call i32 @bytes_grow(ptr nonnull %b, i64 %needed)
  ret i32 %r
}

define i32 @universe_bytes_append(ptr %b, ptr %src, i64 %n) local_unnamed_addr #1 {
entry:
  %b.null = icmp eq ptr %b, null
  br i1 %b.null, label %err.null, label %chk.src, !prof !1

err.null:
  ret i32 1

chk.src:
  ; src may be null only when n == 0 (nothing is read)
  %src.null = icmp eq ptr %src, null
  %n.pos = icmp ne i64 %n, 0
  %bad.src = and i1 %src.null, %n.pos
  br i1 %bad.src, label %err.null, label %work, !prof !1

work:
  %len = load i64, ptr %b, align 8
  %add = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %len, i64 %n)
  %needed = extractvalue { i64, i1 } %add, 0
  %ovf = extractvalue { i64, i1 } %add, 1
  br i1 %ovf, label %err.ovf, label %reserve, !prof !1

err.ovf:
  ret i32 3

reserve:
  %g = call i32 @bytes_grow(ptr nonnull %b, i64 %needed)
  %g.bad = icmp ne i32 %g, 0
  br i1 %g.bad, label %ret.g, label %copy, !prof !1

ret.g:
  ret i32 %g

copy:
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %len
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %n, i1 false)
  store i64 %needed, ptr %b, align 8
  ret i32 0
}

define i32 @universe_bytes_append_byte(ptr %b, i8 %v) local_unnamed_addr #1 {
entry:
  %b.null = icmp eq ptr %b, null
  br i1 %b.null, label %err.null, label %work, !prof !1

err.null:
  ret i32 1

work:
  %len = load i64, ptr %b, align 8
  %add = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %len, i64 1)
  %needed = extractvalue { i64, i1 } %add, 0
  %ovf = extractvalue { i64, i1 } %add, 1
  br i1 %ovf, label %err.ovf, label %reserve, !prof !1

err.ovf:
  ret i32 3

reserve:
  %g = call i32 @bytes_grow(ptr nonnull %b, i64 %needed)
  %g.bad = icmp ne i32 %g, 0
  br i1 %g.bad, label %ret.g, label %put, !prof !1

ret.g:
  ret i32 %g

put:
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  %dst = getelementptr inbounds nuw i8, ptr %data, i64 %len
  store i8 %v, ptr %dst, align 1
  store i64 %needed, ptr %b, align 8
  ret i32 0
}

define void @universe_bytes_clear(ptr %b) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %done, label %do.clear, !prof !1

do.clear:
  store i64 0, ptr %b, align 8
  br label %done

done:
  ret void
}

define i32 @universe_bytes_truncate(ptr %b, i64 %newlen) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %err.null, label %work, !prof !1

err.null:
  ret i32 1

work:
  %len = load i64, ptr %b, align 8
  %bad = icmp ugt i64 %newlen, %len
  br i1 %bad, label %err.idx, label %do.trunc, !prof !1

err.idx:
  ret i32 7

do.trunc:
  store i64 %newlen, ptr %b, align 8
  ret i32 0
}

define void @universe_bytes_destroy(ptr %b) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %done, label %do.free, !prof !1

do.free:
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  call void @free(ptr %data)
  call void @free(ptr nonnull %b)
  br label %done

done:
  ret void
}

; ---------------------------------------------------------------------------
; SIMD scans over the live bytes (delegate to src/simd/scan.ll). Reads through
; the loaded `data` pointer (not just argmem), so these are memory(read) — not
; memory(argmem: read). len==0 => data may be null but n==0 dereferences nothing.
; ---------------------------------------------------------------------------

define i64 @universe_bytes_index_of_byte(ptr %b, i8 %c) local_unnamed_addr #4 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %miss, label %work, !prof !1

miss:
  ret i64 -1

work:
  %len = load i64, ptr %b, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  %r = call i64 @universe_simd_find_byte(ptr %data, i64 %len, i8 %c)
  ret i64 %r
}

define i64 @universe_bytes_count_byte(ptr %b, i8 %c) local_unnamed_addr #4 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %zero, label %work, !prof !1

zero:
  ret i64 0

work:
  %len = load i64, ptr %b, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  %r = call i64 @universe_simd_count_byte(ptr %data, i64 %len, i8 %c)
  ret i64 %r
}

define i1 @universe_bytes_equals(ptr %a, ptr %b) local_unnamed_addr #4 {
entry:
  ; identical handle (incl. both null) => equal
  %same = icmp eq ptr %a, %b
  br i1 %same, label %yes, label %chknull

chknull:
  %anull = icmp eq ptr %a, null
  %bnull = icmp eq ptr %b, null
  %either = or i1 %anull, %bnull
  br i1 %either, label %no, label %lens

lens:
  %alen = load i64, ptr %a, align 8
  %blen = load i64, ptr %b, align 8
  %leneq = icmp eq i64 %alen, %blen
  br i1 %leneq, label %scan, label %no

scan:
  %ad.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %ad = load ptr, ptr %ad.p, align 8
  %bd.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %bd = load ptr, ptr %bd.p, align 8
  %r = call i1 @universe_simd_equal(ptr %ad, ptr %bd, i64 %alen)
  ret i1 %r

yes:
  ret i1 true

no:
  ret i1 false
}

; Lexicographic unsigned compare of two buffers; a null handle is treated as an
; empty buffer. Prefix via the vector kernel over min(len); ties broken by len.
define i32 @universe_bytes_compare(ptr %a, ptr %b) local_unnamed_addr #4 {
entry:
  %anull = icmp eq ptr %a, null
  br i1 %anull, label %have.a, label %load.a

load.a:
  %alen0 = load i64, ptr %a, align 8
  %ad.p = getelementptr inbounds nuw i8, ptr %a, i64 16
  %ad0 = load ptr, ptr %ad.p, align 8
  br label %have.a

have.a:
  %alen = phi i64 [ 0, %entry ], [ %alen0, %load.a ]
  %adata = phi ptr [ null, %entry ], [ %ad0, %load.a ]
  %bnull = icmp eq ptr %b, null
  br i1 %bnull, label %have.b, label %load.b

load.b:
  %blen0 = load i64, ptr %b, align 8
  %bd.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %bd0 = load ptr, ptr %bd.p, align 8
  br label %have.b

have.b:
  %blen = phi i64 [ 0, %have.a ], [ %blen0, %load.b ]
  %bdata = phi ptr [ null, %have.a ], [ %bd0, %load.b ]
  %min = call i64 @llvm.umin.i64(i64 %alen, i64 %blen)
  %c = call i32 @universe_simd_compare(ptr %adata, ptr %bdata, i64 %min)
  %tie = icmp eq i32 %c, 0
  br i1 %tie, label %by.len, label %ret.c

ret.c:
  ret i32 %c

by.len:
  %lt = icmp ult i64 %alen, %blen
  %gt = icmp ugt i64 %alen, %blen
  %gt.i = zext i1 %gt to i32
  %lt.i = zext i1 %lt to i32
  %r = sub nsw i32 %gt.i, %lt.i
  ret i32 %r
}

define void @universe_bytes_to_lower(ptr %b) local_unnamed_addr #5 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %done, label %work, !prof !1

work:
  %len = load i64, ptr %b, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  call void @universe_simd_to_lower_ascii(ptr %data, ptr %data, i64 %len)
  br label %done

done:
  ret void
}

define void @universe_bytes_to_upper(ptr %b) local_unnamed_addr #5 {
entry:
  %null = icmp eq ptr %b, null
  br i1 %null, label %done, label %work, !prof !1

work:
  %len = load i64, ptr %b, align 8
  %data.p = getelementptr inbounds nuw i8, ptr %b, i64 16
  %data = load ptr, ptr %data.p, align 8
  call void @universe_simd_to_upper_ascii(ptr %data, ptr %data, i64 %len)
  br label %done

done:
  ret void
}

attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { alwaysinline nounwind willreturn }
attributes #4 = { nounwind willreturn norecurse nosync nofree memory(read) }
attributes #5 = { nounwind willreturn norecurse nosync nofree }

!1 = !{!"branch_weights", i32 1, i32 2000}
