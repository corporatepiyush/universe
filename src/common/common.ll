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

; Common leaf primitives shared across modules — the canonical shapes for the
; three most-duplicated hot idioms (word rotation, big-endian word I/O,
; overflow-checked size math). Each is one intrinsic wrapped with the flags and
; attributes we want everywhere, so every module inherits identical codegen.
;
; DESIGN (why this exists AND why callers still duplicate the hot ones):
;   * These are the SINGLE SOURCE OF TRUTH for shapes that otherwise drift: a
;     rotation written as (x<<n)|(x>>(32-n)) is poison when n==0 (shift by 32),
;     so it MUST be llvm.fsh*; a big-endian load on our little-endian targets is
;     load+llvm.bswap; a size multiply MUST go through llvm.umul.with.overflow.
;     New modules import these to get the correct, flag-complete form once.
;   * NON-LTO REALITY (CLAUDE.md "cost-free abstraction has a HARD boundary at
;     the module edge"): a cross-.ll call to an exported leaf stays a real
;     bl/call in our per-object archive build — it does NOT inline. So a HOT
;     per-word caller (the SHA compression loops) still DUPLICATES the inline
;     shape locally; it does not call across the module edge. These exports are
;     for (a) cold/once-per-op callers where a bl is free, and (b) the future
;     LTO build variant, under which every caller inlines these and the
;     per-module duplication is deleted with zero runtime cost.
;   * All are `alwaysinline`: within any .ll that includes this file's text (or
;     under LTO) they fold to the single intrinsic; the out-of-line symbol
;     remains for normal cross-module linkage.
;
; API:
;   i32  universe_common_rotl32(i32 x, i32 n) / _rotr32                 ; ror/rol
;   i64  universe_common_rotl64(i64 x, i64 n) / _rotr64
;   i32  universe_common_load_be32(ptr p)   / i64 _load_be64(ptr p)     ; BE read
;   void universe_common_store_be32(ptr p, i32 v) / _store_be64(..i64)  ; BE write
;   i64  universe_common_checked_mul(i64 a, i64 b, ptr err)  ; *err=0 OK|3 OVF
;   i64  universe_common_checked_add(i64 a, i64 b, ptr err)  ; *err=0 OK|3 OVF

declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
declare i64 @llvm.fshl.i64(i64, i64, i64)
declare i64 @llvm.fshr.i64(i64, i64, i64)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.bswap.i64(i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)

; ---------------------------------------------------------------- rotations
define i32 @universe_common_rotl32(i32 %x, i32 %n) local_unnamed_addr #0 {
  %r = call i32 @llvm.fshl.i32(i32 %x, i32 %x, i32 %n)
  ret i32 %r
}
define i32 @universe_common_rotr32(i32 %x, i32 %n) local_unnamed_addr #0 {
  %r = call i32 @llvm.fshr.i32(i32 %x, i32 %x, i32 %n)
  ret i32 %r
}
define i64 @universe_common_rotl64(i64 %x, i64 %n) local_unnamed_addr #0 {
  %r = call i64 @llvm.fshl.i64(i64 %x, i64 %x, i64 %n)
  ret i64 %r
}
define i64 @universe_common_rotr64(i64 %x, i64 %n) local_unnamed_addr #0 {
  %r = call i64 @llvm.fshr.i64(i64 %x, i64 %x, i64 %n)
  ret i64 %r
}

; ----------------------------------------------------------- big-endian I/O
; align 1: callers read/write from arbitrary byte offsets in a stream buffer.
define i32 @universe_common_load_be32(ptr %p) local_unnamed_addr #1 {
  %raw = load i32, ptr %p, align 1
  %be = call i32 @llvm.bswap.i32(i32 %raw)
  ret i32 %be
}
define i64 @universe_common_load_be64(ptr %p) local_unnamed_addr #1 {
  %raw = load i64, ptr %p, align 1
  %be = call i64 @llvm.bswap.i64(i64 %raw)
  ret i64 %be
}
define void @universe_common_store_be32(ptr %p, i32 %v) local_unnamed_addr #2 {
  %be = call i32 @llvm.bswap.i32(i32 %v)
  store i32 %be, ptr %p, align 1
  ret void
}
define void @universe_common_store_be64(ptr %p, i64 %v) local_unnamed_addr #2 {
  %be = call i64 @llvm.bswap.i64(i64 %v)
  store i64 %be, ptr %p, align 1
  ret void
}

; --------------------------------------------------------- checked size math
; Returns the result; writes 0 (OK) or 3 (SIZE_OVERFLOW) to *err. The result is
; unspecified on overflow — callers must branch on *err before using it.
define i64 @universe_common_checked_mul(i64 %a, i64 %b, ptr %err) local_unnamed_addr #3 {
  %m = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %a, i64 %b)
  %v = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %code = select i1 %o, i32 3, i32 0
  store i32 %code, ptr %err, align 4
  ret i64 %v
}
define i64 @universe_common_checked_add(i64 %a, i64 %b, ptr %err) local_unnamed_addr #3 {
  %m = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %v = extractvalue { i64, i1 } %m, 0
  %o = extractvalue { i64, i1 } %m, 1
  %code = select i1 %o, i32 3, i32 0
  store i32 %code, ptr %err, align 4
  ret i64 %v
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(none) }
attributes #1 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: read) }
attributes #2 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: write) }
attributes #3 = { alwaysinline nounwind willreturn norecurse nosync nofree memory(argmem: write) }
