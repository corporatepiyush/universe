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

; Tests for universe_common_*: rotation identities (incl. n==0 and n==width,
; the poison cases a naive shift-or would hit), big-endian round-trips against
; a hand-assembled reference, and checked size math at and past the overflow
; boundary.

declare i32 @universe_common_rotl32(i32, i32)
declare i32 @universe_common_rotr32(i32, i32)
declare i64 @universe_common_rotl64(i64, i64)
declare i64 @universe_common_rotr64(i64, i64)
declare i32 @universe_common_load_be32(ptr)
declare i64 @universe_common_load_be64(ptr)
declare void @universe_common_store_be32(ptr, i32)
declare void @universe_common_store_be64(ptr, i64)
declare i64 @universe_common_checked_mul(i64, i64, ptr)
declare i64 @universe_common_checked_add(i64, i64, ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i32 @ut_summary()

@buf = internal global [8 x i8] zeroinitializer, align 8

@m.rl32   = private unnamed_addr constant [16 x i8] c"rotl32 by 8 ok\0A\00"
@m.rl32z  = private unnamed_addr constant [18 x i8] c"rotl32 by 0 ident\00"
@m.rl32w  = private unnamed_addr constant [18 x i8] c"rotl32 by 32 iden\00"
@m.rr32   = private unnamed_addr constant [14 x i8] c"rotr32 inv rl\00"
@m.rl64   = private unnamed_addr constant [13 x i8] c"rotl64 by 40\00"
@m.rr64   = private unnamed_addr constant [14 x i8] c"rotr64 inv rl\00"
@m.be32   = private unnamed_addr constant [17 x i8] c"load_be32 bytes\0A\00"
@m.be32rt = private unnamed_addr constant [16 x i8] c"be32 round-trip\00"
@m.be64rt = private unnamed_addr constant [16 x i8] c"be64 round-trip\00"
@m.mul    = private unnamed_addr constant [16 x i8] c"checked_mul val\00"
@m.mulok  = private unnamed_addr constant [16 x i8] c"checked_mul ok0\00"
@m.mulov  = private unnamed_addr constant [17 x i8] c"checked_mul ovf3\00"
@m.addov  = private unnamed_addr constant [17 x i8] c"checked_add ovf3\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- rotations ----
  ; rotl32(0x12345678, 8) = 0x34567812
  %a = call i32 @universe_common_rotl32(i32 305419896, i32 8) ; 0x12345678
  %a64 = zext i32 %a to i64
  call void @ut_check_eq(i64 %a64, i64 878082066, ptr @m.rl32) ; 0x34567812
  ; rotl32 by 0 is identity (naive (x<<0)|(x>>32) would be poison)
  %z = call i32 @universe_common_rotl32(i32 305419896, i32 0)
  %z.ok = icmp eq i32 %z, 305419896
  call void @ut_check(i1 %z.ok, ptr @m.rl32z)
  ; rotl32 by 32 is identity too (shift by width)
  %w = call i32 @universe_common_rotl32(i32 305419896, i32 32)
  %w.ok = icmp eq i32 %w, 305419896
  call void @ut_check(i1 %w.ok, ptr @m.rl32w)
  ; rotr32 undoes rotl32
  %rr = call i32 @universe_common_rotr32(i32 %a, i32 8)
  %rr.ok = icmp eq i32 %rr, 305419896
  call void @ut_check(i1 %rr.ok, ptr @m.rr32)

  ; rotl64 / rotr64 inverse over a fixed value and shift
  %b = call i64 @universe_common_rotl64(i64 81985529216486895, i64 40) ; 0x0123456789ABCDEF
  %br = call i64 @universe_common_rotr64(i64 %b, i64 40)
  %b.ok = icmp eq i64 %br, 81985529216486895
  call void @ut_check(i1 %b.ok, ptr @m.rl64)
  %b2.ok = icmp ne i64 %b, 81985529216486895 ; rotation actually moved bits
  call void @ut_check(i1 %b2.ok, ptr @m.rr64)

  ; ---- big-endian I/O ----
  ; store bytes 00 01 02 03 as BE32 of 0x00010203 == 66051
  call void @universe_common_store_be32(ptr @buf, i32 66051)
  %p0 = load i8, ptr @buf, align 1
  %p0z = zext i8 %p0 to i64
  call void @ut_check_eq(i64 %p0z, i64 0, ptr @m.be32) ; MSB first
  %ld = call i32 @universe_common_load_be32(ptr @buf)
  %ld.ok = icmp eq i32 %ld, 66051
  call void @ut_check(i1 %ld.ok, ptr @m.be32rt)

  call void @universe_common_store_be64(ptr @buf, i64 1234605616436508552) ; 0x1122334455667788
  %ld64 = call i64 @universe_common_load_be64(ptr @buf)
  %ld64.ok = icmp eq i64 %ld64, 1234605616436508552
  call void @ut_check(i1 %ld64.ok, ptr @m.be64rt)

  ; ---- checked size math ----
  %err = alloca i32, align 4
  %m1 = call i64 @universe_common_checked_mul(i64 1000, i64 2000, ptr %err)
  call void @ut_check_eq(i64 %m1, i64 2000000, ptr @m.mul)
  %e1 = load i32, ptr %err, align 4
  %e1.ok = icmp eq i32 %e1, 0
  call void @ut_check(i1 %e1.ok, ptr @m.mulok)
  ; overflow: 2^33 * 2^33 = 2^66 overflows i64
  %big = shl i64 1, 33
  %m2 = call i64 @universe_common_checked_mul(i64 %big, i64 %big, ptr %err)
  %e2 = load i32, ptr %err, align 4
  %e2.ok = icmp eq i32 %e2, 3
  call void @ut_check(i1 %e2.ok, ptr @m.mulov)
  ; add overflow: umax + 1
  %m3 = call i64 @universe_common_checked_add(i64 -1, i64 1, ptr %err)
  %e3 = load i32, ptr %err, align 4
  %e3.ok = icmp eq i32 %e3, 3
  call void @ut_check(i1 %e3.ok, ptr @m.addov)

  %rc = call i32 @ut_summary()
  ret i32 %rc
}
