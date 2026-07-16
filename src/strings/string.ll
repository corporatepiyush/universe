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

; UTF-8 strings, two representations. Bytes are opaque: every routine operates
; on raw bytes, so multibyte UTF-8 sequences are handled transparently.
;
; DESIGN — TWO variants, pick by ownership:
;
;   (1) STRING VIEW — immutable {ptr data, i64 len} passed BY VALUE as two
;       args. Pure, allocation-free slices: no ownership, no copy, no lifetime
;       beyond the borrowed bytes. Use for parsing, scanning, comparison,
;       hashing, and slicing over a buffer you already own (e.g. a
;       universe_bytes block or an mmap'd file). substring_view returns an
;       adjusted {ptr,i64} with ZERO allocation — a pointer add + length clamp.
;       Choose the view whenever the bytes outlive the operation and you only
;       read them.
;
;   (2) SSO OWNED STRING — a 24-byte value that OWNS its bytes. Strings of
;       length <= 22 live INLINE inside the 24 bytes (no heap touch at all —
;       create/read/free are pure register/stack work); longer strings hold a
;       heap block. The last byte (offset 23) is the tag: bit0 = 1 => heap,
;       bit0 = 0 => inline with length in bits[1..]. Because inline length is
;       stored as (len<<1) it is always even, so a single `tag & 1` test
;       distinguishes the two with no sentinel. Reading _data is ONE predicated
;       select (inline: the struct itself is the data; heap: the stored ptr) —
;       zero copies on either side. Choose the SSO string when a routine must
;       own/return a string and most strings are short (keys, tokens, labels).
;
;   ABI note (metal-fastest): a 24-byte struct returned by value is passed via
;   a hidden sret pointer on both SysV-x86 and AArch64 (>16 B). So we make that
;   pointer EXPLICIT: the caller supplies a 24-byte, 8-aligned `out` slot
;   (typically an alloca). This is exactly the codegen of a by-value return
;   with none of the ambiguity, and lets the caller keep the string on its own
;   stack frame.
;
;   Inline capacity is 22 (offsets 0..21); offset 22 is spare; offset 23 tag.
;   Heap layout reuses the same 24 bytes: ptr data@0, i64 len@8, tag@23.
;
; API (0 OK, 1 NULL_PTR, 2 OOM):
;   VIEW (pure):
;     i64  universe_string_len(ptr data, i64 len)
;     i1   universe_string_eq(ptr a, i64 alen, ptr b, i64 blen)
;     i32  universe_string_compare(ptr a, i64 alen, ptr b, i64 blen) ; <0/0/>0
;     i64  universe_string_index_of_byte(ptr data, i64 len, i8 byte)  ; idx or -1
;     i64  universe_string_hash(ptr data, i64 len)                    ; deterministic
;     i1   universe_string_starts_with(ptr data, i64 len, ptr pre, i64 plen)
;     {ptr,i64} universe_string_substring_view(ptr data, i64 len, i64 start, i64 count)
;   SSO (owned):
;     i32  universe_sso_create(ptr out, ptr data, i64 len)  ; fills 24-byte out
;     i64  universe_sso_len(ptr s)
;     ptr  universe_sso_data(ptr s)                         ; single select
;     void universe_sso_free(ptr s)                         ; frees heap only

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare i64 @llvm.umin.i64(i64, i64)

; SIMD-first byte kernels (src/simd/scan.ll): the view ops below DELEGATE the
; hot data-parallel scans (equal / lexicographic compare / memchr) to the
; already-verified 128-bit vector kernels + scalar tail, instead of a scalar
; byte loop. Cross-module in our non-LTO build this is a real `bl` (blocks
; inlining of the vector body into the caller); acceptable because these are
; not proven hot leaves and the vector loop dominates. The LTO/unity build
; variant will inline the seam away with no source change. DEPS_strings := simd.
declare i1  @universe_simd_equal(ptr readonly, ptr readonly, i64)
declare i32 @universe_simd_compare(ptr readonly, ptr readonly, i64)
declare i64 @universe_simd_find_byte(ptr readonly, i64, i8)

; ---------------------------------------------------------------------------
; STRING VIEW (pure, no allocation)
; ---------------------------------------------------------------------------

define i64 @universe_string_len(ptr %data, i64 %len) local_unnamed_addr #0 {
entry:
  ret i64 %len
}

; Byte-wise equality. Length check first (branch-predicted), then the vector
; equality kernel (SIMD-first) over the shared length.
define i1 @universe_string_eq(ptr %a, i64 %alen, ptr %b, i64 %blen) local_unnamed_addr #0 {
entry:
  %len.eq = icmp eq i64 %alen, %blen
  br i1 %len.eq, label %scan, label %ne

ne:
  ret i1 false

scan:
  ; simd_equal returns true for n == 0, so no empty special-case needed.
  %r = call i1 @universe_simd_equal(ptr %a, ptr %b, i64 %alen)
  ret i1 %r
}

; Lexicographic unsigned-byte compare. Returns <0, 0, or >0. The prefix scan is
; the vector kernel (memcmp sign); ties broken by length.
define i32 @universe_string_compare(ptr %a, i64 %alen, ptr %b, i64 %blen) local_unnamed_addr #0 {
entry:
  %min = call i64 @llvm.umin.i64(i64 %alen, i64 %blen)
  %c = call i32 @universe_simd_compare(ptr %a, ptr %b, i64 %min)
  %tie = icmp eq i32 %c, 0
  br i1 %tie, label %by.len, label %ret.c

ret.c:
  ret i32 %c

by.len:
  ; all compared bytes equal -> shorter string is smaller
  %lt = icmp ult i64 %alen, %blen
  %gt = icmp ugt i64 %alen, %blen
  %gt.i = zext i1 %gt to i32
  %lt.i = zext i1 %lt to i32
  %r = sub nsw i32 %gt.i, %lt.i
  ret i32 %r
}

; memchr — the vector find_byte kernel (returns first index or -1; n == 0 -> -1).
define i64 @universe_string_index_of_byte(ptr %data, i64 %len, i8 %byte) local_unnamed_addr #0 {
entry:
  %r = call i64 @universe_simd_find_byte(ptr %data, i64 %len, i8 %byte)
  ret i64 %r
}

; FNV-1a over the bytes, finished with the splitmix64 finalizer for avalanche.
; Deterministic: same bytes -> same hash on every platform.
define i64 @universe_string_hash(ptr %data, i64 %len) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %final, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %h = phi i64 [ -3750763034362895579, %entry ], [ %h.n, %loop ]  ; 0xcbf29ce484222325
  %p = getelementptr inbounds nuw i8, ptr %data, i64 %i
  %c = load i8, ptr %p, align 1
  %c64 = zext i8 %c to i64
  %hx = xor i64 %h, %c64
  %h.n = mul i64 %hx, 1099511628211                                ; 0x100000001b3
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %len
  br i1 %more, label %loop, label %final

final:
  %hv = phi i64 [ -3750763034362895579, %entry ], [ %h.n, %loop ]
  ; splitmix64 finalizer
  %s1 = lshr i64 %hv, 30
  %x1 = xor i64 %s1, %hv
  %m1 = mul i64 %x1, -4658895280553007687     ; 0xbf58476d1ce4e5b9
  %s2 = lshr i64 %m1, 27
  %x2 = xor i64 %s2, %m1
  %m2 = mul i64 %x2, -7723592293110705685     ; 0x94d049bb133111eb
  %s3 = lshr i64 %m2, 31
  %x3 = xor i64 %s3, %m2
  ret i64 %x3
}

define i1 @universe_string_starts_with(ptr %data, i64 %len, ptr %pre, i64 %plen) local_unnamed_addr #0 {
entry:
  %too.long = icmp ugt i64 %plen, %len
  br i1 %too.long, label %no, label %maybe

maybe:
  ; vector prefix equality (simd_equal returns true for plen == 0)
  %r = call i1 @universe_simd_equal(ptr %data, ptr %pre, i64 %plen)
  ret i1 %r

no:
  ret i1 false
}

; Branchless slice: clamp start to [0,len], count to [0, len-start].
; Returns { adjusted data ptr, adjusted len } — no allocation.
define { ptr, i64 } @universe_string_substring_view(ptr %data, i64 %len, i64 %start, i64 %count) local_unnamed_addr #0 {
entry:
  %start.c = call i64 @llvm.umin.i64(i64 %start, i64 %len)
  %avail = sub i64 %len, %start.c
  %count.c = call i64 @llvm.umin.i64(i64 %count, i64 %avail)
  %ndata = getelementptr inbounds nuw i8, ptr %data, i64 %start.c
  %r0 = insertvalue { ptr, i64 } poison, ptr %ndata, 0
  %r1 = insertvalue { ptr, i64 } %r0, i64 %count.c, 1
  ret { ptr, i64 } %r1
}

; ---------------------------------------------------------------------------
; SSO OWNED STRING (24-byte value; tag byte @23, bit0 = heap flag)
; ---------------------------------------------------------------------------

define i32 @universe_sso_create(ptr %out, ptr %data, i64 %len) local_unnamed_addr #1 {
entry:
  %out.null = icmp eq ptr %out, null
  br i1 %out.null, label %err.null, label %chk.src, !prof !1

err.null:
  ret i32 1

chk.src:
  %src.null = icmp eq ptr %data, null
  %n.pos = icmp ne i64 %len, 0
  %bad.src = and i1 %src.null, %n.pos
  br i1 %bad.src, label %err.null, label %classify, !prof !1

classify:
  %is.inline = icmp ule i64 %len, 22
  br i1 %is.inline, label %inline, label %heap

inline:
  ; copy bytes into the struct itself, tag = len<<1 (bit0 = 0 => inline)
  call void @llvm.memcpy.p0.p0.i64(ptr %out, ptr %data, i64 %len, i1 false)
  %tag.in = shl nuw i64 %len, 1
  %tag.in8 = trunc i64 %tag.in to i8
  %tag.p.in = getelementptr inbounds nuw i8, ptr %out, i64 23
  store i8 %tag.in8, ptr %tag.p.in, align 1
  ret i32 0

heap:
  %mem = call ptr @malloc(i64 %len)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %err.oom, label %heap.init, !prof !1

err.oom:
  ret i32 2

heap.init:
  call void @llvm.memcpy.p0.p0.i64(ptr %mem, ptr %data, i64 %len, i1 false)
  store ptr %mem, ptr %out, align 8
  %len.p = getelementptr inbounds nuw i8, ptr %out, i64 8
  store i64 %len, ptr %len.p, align 8
  %tag.p.h = getelementptr inbounds nuw i8, ptr %out, i64 23
  store i8 1, ptr %tag.p.h, align 1      ; bit0 = 1 => heap
  ret i32 0
}

define i64 @universe_sso_len(ptr %s) local_unnamed_addr #2 {
entry:
  %tag.p = getelementptr inbounds nuw i8, ptr %s, i64 23
  %tag = load i8, ptr %tag.p, align 1
  %is.heap = and i8 %tag, 1
  %heap.b = icmp ne i8 %is.heap, 0
  ; inline length = tag >> 1
  %tag64 = zext i8 %tag to i64
  %inline.len = lshr i64 %tag64, 1
  ; heap length lives at offset 8
  %len.p = getelementptr inbounds nuw i8, ptr %s, i64 8
  %heap.len = load i64, ptr %len.p, align 8
  %len = select i1 %heap.b, i64 %heap.len, i64 %inline.len
  ret i64 %len
}

; Single predicated select: inline -> the struct is the data; heap -> stored ptr.
define ptr @universe_sso_data(ptr %s) local_unnamed_addr #3 {
entry:
  %tag.p = getelementptr inbounds nuw i8, ptr %s, i64 23
  %tag = load i8, ptr %tag.p, align 1
  %is.heap = and i8 %tag, 1
  %heap.b = icmp ne i8 %is.heap, 0
  %heap.ptr = load ptr, ptr %s, align 8      ; @0 holds the heap data ptr
  %data = select i1 %heap.b, ptr %heap.ptr, ptr %s
  ret ptr %data
}

define void @universe_sso_free(ptr %s) local_unnamed_addr #1 {
entry:
  %null = icmp eq ptr %s, null
  br i1 %null, label %done, label %chk, !prof !1

chk:
  %tag.p = getelementptr inbounds nuw i8, ptr %s, i64 23
  %tag = load i8, ptr %tag.p, align 1
  %is.heap = and i8 %tag, 1
  %heap.b = icmp ne i8 %is.heap, 0
  br i1 %heap.b, label %do.free, label %done

do.free:
  %heap.ptr = load ptr, ptr %s, align 8
  call void @free(ptr %heap.ptr)
  br label %done

done:
  ret void
}

attributes #0 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #3 = { nounwind willreturn norecurse nosync nofree memory(argmem: read) }

!1 = !{!"branch_weights", i32 1, i32 2000}
