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

; Tests for universe_encoding_thrift_*: hand-encoded compact-protocol KATs
; (a struct with an i32, a binary, and a list<i32>, plus an explicit-field-id
; struct), zig-zag round-trips over fixed-seed random i16/i32/i64 (buffers built
; with the varint module's encoders), a double read, truncation/malformed error
; modes (empty buffer, truncated varint, truncated binary — must return 8 and
; never read OOB), depth-limited skip on a hostile nested struct, a whole-struct
; skip, and a --bench mode.

; ---- module under test ----
declare void @universe_encoding_thrift_init(ptr, ptr, i64)
declare i32 @universe_encoding_thrift_read_uvarint(ptr, ptr)
declare i32 @universe_encoding_thrift_read_i8(ptr, ptr)
declare i32 @universe_encoding_thrift_read_i16(ptr, ptr)
declare i32 @universe_encoding_thrift_read_i32(ptr, ptr)
declare i32 @universe_encoding_thrift_read_i64(ptr, ptr)
declare i32 @universe_encoding_thrift_read_double(ptr, ptr)
declare i32 @universe_encoding_thrift_read_binary(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_field(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_collection(ptr, ptr, ptr)
declare i32 @universe_encoding_thrift_skip(ptr, i32, i32)

; ---- sibling encoders (same domain archive) to build round-trip buffers ----
declare i64 @universe_varint_zigzag_encode(i64)
declare i64 @universe_varint_uleb_encode(ptr, i64)

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

; 32-byte reader state (base,len,pos,last_field_id).
@g.reader = internal global [32 x i8] zeroinitializer, align 8
@g.buf    = internal global [32 x i8] zeroinitializer, align 8
@g.out    = internal global i64 0, align 8

; ---- KAT 1: a struct -------------------------------------------------------
;   field 1 : i32   = -1        -> hdr 0x15, zigzag(-1)=1 -> 0x01
;   field 2 : binary "hi"       -> hdr 0x18, len 0x02, 'h' 'i'
;   field 3 : list<i32>[1,2,3]  -> hdr 0x19, listhdr 0x35, zz 02 04 06
;   STOP 0x00
@kat1 = private unnamed_addr constant [12 x i8] c"\15\01\18\02\68\69\19\35\02\04\06\00"
@hi   = private unnamed_addr constant [2 x i8] c"\68\69"

; ---- KAT 2: explicit field id (delta==0) -----------------------------------
;   field id 20 (zigzag 40 -> varint 0x28), type i8 (3) -> hdr 0x03
;   i8 value 0x7F, STOP 0x00
@kat2 = private unnamed_addr constant [4 x i8] c"\03\28\7F\00"

; ---- malformed inputs ------------------------------------------------------
@bad.uvar = private unnamed_addr constant [1 x i8] c"\80"                ; continuation, ends
@bad.bin  = private unnamed_addr constant [3 x i8] c"\18\05\68"          ; binary len 5, 1 byte

; ---- double: 1.5 = 0x3FF8000000000000 (little-endian bytes) ----------------
@dbl = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\F8\3F"

; ---- deeply nested struct: 70 x 0x1C (struct-in-struct field headers) ------
@g.deep = internal global [70 x i8] zeroinitializer, align 1

; ---- messages ----
@m.f1t   = private unnamed_addr constant [17 x i8] c"kat1 f1 type i32\00"
@m.f1id  = private unnamed_addr constant [13 x i8] c"kat1 f1 id 1\00"
@m.f1v   = private unnamed_addr constant [15 x i8] c"kat1 f1 == -1 \00"
@m.f2t   = private unnamed_addr constant [17 x i8] c"kat1 f2 type bin\00"
@m.f2id  = private unnamed_addr constant [13 x i8] c"kat1 f2 id 2\00"
@m.f2len = private unnamed_addr constant [17 x i8] c"kat1 f2 len == 2\00"
@m.f2v   = private unnamed_addr constant [19 x i8] c"kat1 f2 == \22hi\22   \00"
@m.f3t   = private unnamed_addr constant [18 x i8] c"kat1 f3 type list\00"
@m.f3id  = private unnamed_addr constant [13 x i8] c"kat1 f3 id 3\00"
@m.f3et  = private unnamed_addr constant [19 x i8] c"kat1 list elem i32\00"
@m.f3sz  = private unnamed_addr constant [17 x i8] c"kat1 list sz==3 \00"
@m.f3e   = private unnamed_addr constant [16 x i8] c"kat1 list elems\00"
@m.stop  = private unnamed_addr constant [16 x i8] c"kat1 STOP ct==0\00"
@m.k2t   = private unnamed_addr constant [16 x i8] c"kat2 type i8   \00"
@m.k2id  = private unnamed_addr constant [17 x i8] c"kat2 expl id 20 \00"
@m.k2v   = private unnamed_addr constant [16 x i8] c"kat2 i8 == 127 \00"
@m.dbl   = private unnamed_addr constant [17 x i8] c"double == 1.5   \00"
@m.empty = private unnamed_addr constant [19 x i8] c"empty -> field 8  \00"
@m.tuv   = private unnamed_addr constant [20 x i8] c"trunc uvarint -> 8 \00"
@m.tbin  = private unnamed_addr constant [20 x i8] c"trunc binary -> 8  \00"
@m.deep  = private unnamed_addr constant [19 x i8] c"deep skip -> 8    \00"
@m.skip  = private unnamed_addr constant [18 x i8] c"skip struct ok   \00"
@m.skipp = private unnamed_addr constant [19 x i8] c"skip consumed all \00"
@m.rt64  = private unnamed_addr constant [17 x i8] c"i64 round-trip  \00"
@m.rt32  = private unnamed_addr constant [17 x i8] c"i32 round-trip  \00"
@m.rt16  = private unnamed_addr constant [17 x i8] c"i16 round-trip  \00"
@m.fuzz  = private unnamed_addr constant [20 x i8] c"fuzz no-crash (8/0)\00"

@vi.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.dec = private unnamed_addr constant [20 x i8] c"thrift read_i64    \00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %ct   = alloca i8, align 1
  %fid  = alloca i16, align 2
  %et   = alloca i8, align 1
  %sz   = alloca i32, align 4
  %v64  = alloca i64, align 8
  %v32  = alloca i32, align 4
  %v16  = alloca i16, align 2
  %bptr = alloca ptr, align 8
  %blen = alloca i64, align 8
  %dv   = alloca i64, align 8

  ; ================= KAT 1 =================
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @kat1, i64 12)

  ; field 1
  %r1 = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %ct1 = load i8, ptr %ct, align 1
  %id1 = load i16, ptr %fid, align 2
  %ct1.ok = icmp eq i8 %ct1, 5
  %r1.ok = icmp eq i32 %r1, 0
  %f1t = and i1 %ct1.ok, %r1.ok
  call void @ut_check(i1 %f1t, ptr @m.f1t)
  %id1.ok = icmp eq i16 %id1, 1
  call void @ut_check(i1 %id1.ok, ptr @m.f1id)
  %rv1 = call i32 @universe_encoding_thrift_read_i32(ptr @g.reader, ptr %v32)
  %val1 = load i32, ptr %v32, align 4
  %val1.ok = icmp eq i32 %val1, -1
  %rv1.ok = icmp eq i32 %rv1, 0
  %f1v = and i1 %val1.ok, %rv1.ok
  call void @ut_check(i1 %f1v, ptr @m.f1v)

  ; field 2 (binary)
  %r2 = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %ct2 = load i8, ptr %ct, align 1
  %id2 = load i16, ptr %fid, align 2
  %ct2.ok = icmp eq i8 %ct2, 8
  call void @ut_check(i1 %ct2.ok, ptr @m.f2t)
  %id2.ok = icmp eq i16 %id2, 2
  call void @ut_check(i1 %id2.ok, ptr @m.f2id)
  %rb = call i32 @universe_encoding_thrift_read_binary(ptr @g.reader, ptr %bptr, ptr %blen)
  %blv = load i64, ptr %blen, align 8
  %blv.ok = icmp eq i64 %blv, 2
  %rb.ok = icmp eq i32 %rb, 0
  %f2len = and i1 %blv.ok, %rb.ok
  call void @ut_check(i1 %f2len, ptr @m.f2len)
  %bpv = load ptr, ptr %bptr, align 8
  %cmp = call i32 @memcmp(ptr %bpv, ptr @hi, i64 2)
  %cmp.ok = icmp eq i32 %cmp, 0
  call void @ut_check(i1 %cmp.ok, ptr @m.f2v)

  ; field 3 (list<i32>)
  %r3 = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %ct3 = load i8, ptr %ct, align 1
  %id3 = load i16, ptr %fid, align 2
  %ct3.ok = icmp eq i8 %ct3, 9
  call void @ut_check(i1 %ct3.ok, ptr @m.f3t)
  %id3.ok = icmp eq i16 %id3, 3
  call void @ut_check(i1 %id3.ok, ptr @m.f3id)
  %rc = call i32 @universe_encoding_thrift_collection(ptr @g.reader, ptr %et, ptr %sz)
  %etv = load i8, ptr %et, align 1
  %szv = load i32, ptr %sz, align 4
  %etv.ok = icmp eq i8 %etv, 5
  %rc.ok = icmp eq i32 %rc, 0
  %f3et = and i1 %etv.ok, %rc.ok
  call void @ut_check(i1 %f3et, ptr @m.f3et)
  %szv.ok = icmp eq i32 %szv, 3
  call void @ut_check(i1 %szv.ok, ptr @m.f3sz)
  ; three i32 elements 1,2,3
  %e1r = call i32 @universe_encoding_thrift_read_i32(ptr @g.reader, ptr %v32)
  %e1 = load i32, ptr %v32, align 4
  %e2r = call i32 @universe_encoding_thrift_read_i32(ptr @g.reader, ptr %v32)
  %e2 = load i32, ptr %v32, align 4
  %e3r = call i32 @universe_encoding_thrift_read_i32(ptr @g.reader, ptr %v32)
  %e3 = load i32, ptr %v32, align 4
  %e1.ok = icmp eq i32 %e1, 1
  %e2.ok = icmp eq i32 %e2, 2
  %e3.ok = icmp eq i32 %e3, 3
  %e12 = and i1 %e1.ok, %e2.ok
  %e123 = and i1 %e12, %e3.ok
  call void @ut_check(i1 %e123, ptr @m.f3e)

  ; STOP
  %rs = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %cts = load i8, ptr %ct, align 1
  %cts.ok = icmp eq i8 %cts, 0
  %rs.ok = icmp eq i32 %rs, 0
  %stop.ok = and i1 %cts.ok, %rs.ok
  call void @ut_check(i1 %stop.ok, ptr @m.stop)

  ; ================= KAT 2 (explicit field id) =================
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @kat2, i64 4)
  %k2 = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %k2ct = load i8, ptr %ct, align 1
  %k2id = load i16, ptr %fid, align 2
  %k2ct.ok = icmp eq i8 %k2ct, 3
  %k2.ok = icmp eq i32 %k2, 0
  %k2t = and i1 %k2ct.ok, %k2.ok
  call void @ut_check(i1 %k2t, ptr @m.k2t)
  %k2id.ok = icmp eq i16 %k2id, 20
  call void @ut_check(i1 %k2id.ok, ptr @m.k2id)
  %k2vr = call i32 @universe_encoding_thrift_read_i8(ptr @g.reader, ptr %v32)
  %k2v = load i8, ptr %v32, align 1
  %k2v.ok = icmp eq i8 %k2v, 127
  call void @ut_check(i1 %k2v.ok, ptr @m.k2v)

  ; ================= double =================
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @dbl, i64 8)
  %rd = call i32 @universe_encoding_thrift_read_double(ptr @g.reader, ptr %dv)
  %dvv = load i64, ptr %dv, align 8
  %dvv.ok = icmp eq i64 %dvv, 4609434218613702656   ; 0x3FF8000000000000
  %rd.ok = icmp eq i32 %rd, 0
  %dbl.ok = and i1 %dvv.ok, %rd.ok
  call void @ut_check(i1 %dbl.ok, ptr @m.dbl)

  ; ================= error modes =================
  ; empty buffer: field returns 8
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 0)
  %em = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %em.ok = icmp eq i32 %em, 8
  call void @ut_check(i1 %em.ok, ptr @m.empty)

  ; truncated uvarint
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @bad.uvar, i64 1)
  %tu = call i32 @universe_encoding_thrift_read_uvarint(ptr @g.reader, ptr %v64)
  %tu.ok = icmp eq i32 %tu, 8
  call void @ut_check(i1 %tu.ok, ptr @m.tuv)

  ; truncated binary
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @bad.bin, i64 3)
  %tbf = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %tb = call i32 @universe_encoding_thrift_read_binary(ptr @g.reader, ptr %bptr, ptr %blen)
  %tb.ok = icmp eq i32 %tb, 8
  call void @ut_check(i1 %tb.ok, ptr @m.tbin)

  ; ================= depth-limited skip =================
  call void @llvm.memset.p0.i64(ptr @g.deep, i8 28, i64 70, i1 false)  ; 0x1C
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.deep, i64 70)
  %dk = call i32 @universe_encoding_thrift_skip(ptr @g.reader, i32 12, i32 0)
  %dk.ok = icmp eq i32 %dk, 8
  call void @ut_check(i1 %dk.ok, ptr @m.deep)

  ; ================= whole-struct skip =================
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @kat1, i64 12)
  %sk = call i32 @universe_encoding_thrift_skip(ptr @g.reader, i32 12, i32 0)
  %sk.ok = icmp eq i32 %sk, 0
  call void @ut_check(i1 %sk.ok, ptr @m.skip)
  %posp = getelementptr inbounds i8, ptr @g.reader, i64 16
  %posv = load i64, ptr %posp, align 8
  %pos.ok = icmp eq i64 %posv, 12
  call void @ut_check(i1 %pos.ok, ptr @m.skipp)

  ; ================= random zig-zag round-trips =================
  %state = alloca i64, align 8
  store i64 88172645463325252, ptr %state, align 8
  br label %rt.head

rt.head:
  %it = phi i64 [ 0, %entry ], [ %it.n, %rt.head ]
  %m64 = phi i64 [ 0, %entry ], [ %m64.n, %rt.head ]
  %m32 = phi i64 [ 0, %entry ], [ %m32.n, %rt.head ]
  %m16 = phi i64 [ 0, %entry ], [ %m16.n, %rt.head ]
  %rv = call i64 @ut_rand(ptr %state)

  ; --- i64 ---
  %zz64 = call i64 @universe_varint_zigzag_encode(i64 %rv)
  %w64 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %zz64)
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 32)
  %d64 = call i32 @universe_encoding_thrift_read_i64(ptr @g.reader, ptr %v64)
  %got64 = load i64, ptr %v64, align 8
  %bad64 = icmp ne i64 %got64, %rv
  %bad64.b = zext i1 %bad64 to i64
  %m64.n = add nuw i64 %m64, %bad64.b

  ; --- i32 (sign-extended low 32 bits) ---
  %lo32 = trunc i64 %rv to i32
  %sx32 = sext i32 %lo32 to i64
  %zz32 = call i64 @universe_varint_zigzag_encode(i64 %sx32)
  %w32 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %zz32)
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 32)
  %d32 = call i32 @universe_encoding_thrift_read_i32(ptr @g.reader, ptr %v32)
  %got32 = load i32, ptr %v32, align 4
  %bad32 = icmp ne i32 %got32, %lo32
  %bad32.b = zext i1 %bad32 to i64
  %m32.n = add nuw i64 %m32, %bad32.b

  ; --- i16 (sign-extended low 16 bits) ---
  %lo16 = trunc i64 %rv to i16
  %sx16 = sext i16 %lo16 to i64
  %zz16 = call i64 @universe_varint_zigzag_encode(i64 %sx16)
  %w16 = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %zz16)
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 32)
  %d16 = call i32 @universe_encoding_thrift_read_i16(ptr @g.reader, ptr %v16)
  %got16 = load i16, ptr %v16, align 2
  %bad16 = icmp ne i16 %got16, %lo16
  %bad16.b = zext i1 %bad16 to i64
  %m16.n = add nuw i64 %m16, %bad16.b

  %it.n = add nuw i64 %it, 1
  %more = icmp ult i64 %it.n, 200000
  br i1 %more, label %rt.head, label %rt.done

rt.done:
  call void @ut_check_eq(i64 %m64, i64 0, ptr @m.rt64)
  call void @ut_check_eq(i64 %m32, i64 0, ptr @m.rt32)
  call void @ut_check_eq(i64 %m16, i64 0, ptr @m.rt16)

  ; ================= fuzz: mutated/truncated inputs never crash =================
  ; Fill g.buf with random bytes, init over a random-length prefix, drive the
  ; field/skip grammar; every call must return 0 or 8 (never OOB — the sanitizer
  ; build proves the memory safety; here we assert the status invariant).
  store i64 12345, ptr %state, align 8
  br label %fz.head

fz.head:
  %fi = phi i64 [ 0, %rt.done ], [ %fi.n, %fz.tail ]
  %fbad = phi i64 [ 0, %rt.done ], [ %fbad.n, %fz.tail ]
  ; randomize 32 bytes
  %fr0 = call i64 @ut_rand(ptr %state)
  store i64 %fr0, ptr @g.buf, align 8
  %fr1 = call i64 @ut_rand(ptr %state)
  %gb1 = getelementptr inbounds i8, ptr @g.buf, i64 8
  store i64 %fr1, ptr %gb1, align 8
  %fr2 = call i64 @ut_rand(ptr %state)
  %gb2 = getelementptr inbounds i8, ptr @g.buf, i64 16
  store i64 %fr2, ptr %gb2, align 8
  %fr3 = call i64 @ut_rand(ptr %state)
  %gb3 = getelementptr inbounds i8, ptr @g.buf, i64 24
  store i64 %fr3, ptr %gb3, align 8
  %frl = call i64 @ut_rand(ptr %state)
  %flen = urem i64 %frl, 33          ; 0..32
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 %flen)
  ; try a field header, then skip whatever type it claims
  %ff = call i32 @universe_encoding_thrift_field(ptr @g.reader, ptr %ct, ptr %fid)
  %ffct = load i8, ptr %ct, align 1
  %ffct32 = zext i8 %ffct to i32
  %fs = call i32 @universe_encoding_thrift_skip(ptr @g.reader, i32 %ffct32, i32 0)
  ; status invariant: each must be 0 or 8
  %ff0 = icmp eq i32 %ff, 0
  %ff8 = icmp eq i32 %ff, 8
  %ffok = or i1 %ff0, %ff8
  %fs0 = icmp eq i32 %fs, 0
  %fs8 = icmp eq i32 %fs, 8
  %fsok = or i1 %fs0, %fs8
  %fok = and i1 %ffok, %fsok
  %fbadb = xor i1 %fok, true
  %fbadz = zext i1 %fbadb to i64
  %fbad.n = add nuw i64 %fbad, %fbadz
  br label %fz.tail
fz.tail:
  %fi.n = add nuw i64 %fi, 1
  %fmore = icmp ult i64 %fi.n, 100000
  br i1 %fmore, label %fz.head, label %fz.done
fz.done:
  call void @ut_check_eq(i64 %fbad, i64 0, ptr @m.fuzz)

  ; ================= bench =================
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; encode one fixed value once; time N re-init + read_i64 decodes.
  %bzz = call i64 @universe_varint_zigzag_encode(i64 -123456789)
  %bw = call i64 @universe_varint_uleb_encode(ptr @g.buf, i64 %bzz)
  br label %b.rep

b.rep:
  %brep = phi i64 [ 0, %bench ], [ %brep.n, %b.next ]
  %bt0 = call double @ut_now_sec()
  br label %b.loop
b.loop:
  %bc = phi i64 [ 0, %b.rep ], [ %bc.n, %b.loop ]
  call void @universe_encoding_thrift_init(ptr @g.reader, ptr @g.buf, i64 32)
  %bd = call i32 @universe_encoding_thrift_read_i64(ptr @g.reader, ptr @g.out)
  %bc.n = add nuw i64 %bc, 1
  %bmore = icmp ult i64 %bc.n, 1000000
  br i1 %bmore, label %b.loop, label %b.rep.done
b.rep.done:
  %bt1 = call double @ut_now_sec()
  %bel = fsub double %bt1, %bt0
  %bkeep = icmp ugt i64 %brep, 0
  br i1 %bkeep, label %b.store, label %b.next
b.store:
  %bidx = sub i64 %brep, 1
  %bsp = getelementptr inbounds [16 x double], ptr @vi.samp, i64 0, i64 %bidx
  store double %bel, ptr %bsp, align 8
  br label %b.next
b.next:
  %brep.n = add nuw i64 %brep, 1
  %brm = icmp ult i64 %brep.n, 17
  br i1 %brm, label %b.rep, label %b.report
b.report:
  call void @ut_report_dist(ptr @vi.samp, i64 16, i64 1000000, ptr @lbl.dec)
  br label %fin

fin:
  %code = call i32 @ut_summary()
  ret i32 %code
}
