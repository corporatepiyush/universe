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

; Tests for universe_utf16_*: known scalars (ASCII, BMP euro, astral emoji,
; U+10FFFF) encode/decode LE and BE; lone/reversed/truncated surrogates and
; out-of-range scalars are rejected; endianness differs by byte swap; random
; scalar round-trips and a UTF-8<->UTF-16<->UTF-8 round-trip cross-checked
; against the sibling utf8 module, aggregated into mismatch counters.
; --bench transcodes a 64 KiB UTF-8 buffer to UTF-16.

declare i32 @universe_utf16_scalar_units(i32)
declare i64 @universe_utf16_encode_scalar(ptr, i32, i32)
declare i64 @universe_utf16_decode_scalar(ptr, i64, i32)
declare i32 @universe_utf16_bom_detect(ptr, i64)
declare i64 @universe_utf16_len_from_utf8(ptr, i64)
declare i64 @universe_utf16_len_to_utf8(ptr, i64, i32)
declare i64 @universe_utf16_from_utf8(ptr, i64, ptr, i64, i32)
declare i64 @universe_utf16_to_utf8(ptr, i64, ptr, i64, i32)

declare i32 @printf(ptr, ...)
declare i32 @memcmp(ptr, ptr, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@g.u16 = internal global [65536 x i8] zeroinitializer, align 16
@g.u8  = internal global [65536 x i8] zeroinitializer, align 16

; ---- SIMD transcode oracle buffers (LE = vector path, BE = scalar-only) ----
@o.u16le = internal global [65536 x i8] zeroinitializer, align 16
@o.u16be = internal global [65536 x i8] zeroinitializer, align 16
@o.b1    = internal global [65536 x i8] zeroinitializer, align 16
@o.b2    = internal global [65536 x i8] zeroinitializer, align 16
@o.o1    = internal global [65536 x i8] zeroinitializer, align 16
@o.o2    = internal global [65536 x i8] zeroinitializer, align 16
@m2.tnb  = private unnamed_addr constant [22 x i8] c"to_utf8 vec==scl len\00\00"
@m2.tby  = private unnamed_addr constant [24 x i8] c"to_utf8 vec==scl bytes\00\00"
@m2.ltu  = private unnamed_addr constant [22 x i8] c"len_to_utf8 == nbytes\00"
@m2.fvu  = private unnamed_addr constant [22 x i8] c"from_utf8 vec units=U\00"
@m2.fsu  = private unnamed_addr constant [22 x i8] c"from_utf8 scl units=U\00"
@m2.fvb  = private unnamed_addr constant [24 x i8] c"from_utf8 vec le bytes\00\00"
@m2.fsb  = private unnamed_addr constant [24 x i8] c"from_utf8 scl be bytes\00\00"
@m2.lfu  = private unnamed_addr constant [20 x i8] c"len_from_utf8 == U\00\00"

; ---- UTF-16 byte buffers for decode (LE and BE) ----
@d.a.le   = private unnamed_addr constant [2 x i8] c"\41\00", align 1          ; 'A'
@d.a.be   = private unnamed_addr constant [2 x i8] c"\00\41", align 1
@d.eu.le  = private unnamed_addr constant [2 x i8] c"\AC\20", align 1          ; U+20AC
@d.eu.be  = private unnamed_addr constant [2 x i8] c"\20\AC", align 1
@d.em.le  = private unnamed_addr constant [4 x i8] c"\3D\D8\00\DE", align 1    ; U+1F600
@d.em.be  = private unnamed_addr constant [4 x i8] c"\D8\3D\DE\00", align 1
@d.mx.le  = private unnamed_addr constant [4 x i8] c"\FF\DB\FF\DF", align 1    ; U+10FFFF
@d.mx.be  = private unnamed_addr constant [4 x i8] c"\DB\FF\DF\FF", align 1

; ---- malformed UTF-16 (LE) ----
@d.lohi   = private unnamed_addr constant [2 x i8] c"\00\D8", align 1          ; lone high surrogate
@d.lolo   = private unnamed_addr constant [2 x i8] c"\00\DC", align 1          ; lone low surrogate
@d.rev    = private unnamed_addr constant [4 x i8] c"\00\DC\00\D8", align 1    ; low then high

; ---- UTF-8 reference buffer (A, euro U+20AC, emoji U+1F600, e-acute U+00E9) ----
@u8.mixed = private unnamed_addr constant [10 x i8] c"A\E2\82\AC\F0\9F\98\80\C3\A9", align 1

; ---- BOM samples ----
@bom.le   = private unnamed_addr constant [2 x i8] c"\FF\FE", align 1
@bom.be   = private unnamed_addr constant [2 x i8] c"\FE\FF", align 1
@bom.no   = private unnamed_addr constant [2 x i8] c"\41\42", align 1

@m.a.le   = private unnamed_addr constant [17 x i8] c"decode A LE=0x41\00"
@m.a.be   = private unnamed_addr constant [17 x i8] c"decode A BE=0x41\00"
@m.eu.le  = private unnamed_addr constant [14 x i8] c"decode eu LE\00\00"
@m.eu.be  = private unnamed_addr constant [14 x i8] c"decode eu BE\00\00"
@m.em.le  = private unnamed_addr constant [16 x i8] c"decode emoji LE\00"
@m.em.be  = private unnamed_addr constant [16 x i8] c"decode emoji BE\00"
@m.mx.le  = private unnamed_addr constant [17 x i8] c"decode 10FFFF LE\00"
@m.mx.be  = private unnamed_addr constant [17 x i8] c"decode 10FFFF BE\00"
@m.lohi   = private unnamed_addr constant [17 x i8] c"lone high -> -13\00"
@m.lolo   = private unnamed_addr constant [16 x i8] c"lone low -> -13\00"
@m.rev    = private unnamed_addr constant [17 x i8] c"reversed -> -13 \00"
@m.tr     = private unnamed_addr constant [18 x i8] c"trunc pair -> -13\00"
@m.emp    = private unnamed_addr constant [16 x i8] c"empty dec -> -8\00"

@m.enc1   = private unnamed_addr constant [17 x i8] c"encode A units=1\00"
@m.encb0  = private unnamed_addr constant [16 x i8] c"encode A LE b0 \00"
@m.encb1  = private unnamed_addr constant [16 x i8] c"encode A LE b1 \00"
@m.encbb0 = private unnamed_addr constant [16 x i8] c"encode A BE b0 \00"
@m.encbb1 = private unnamed_addr constant [16 x i8] c"encode A BE b1 \00"
@m.enc2   = private unnamed_addr constant [21 x i8] c"encode emoji units=2\00"
@m.encbad = private unnamed_addr constant [20 x i8] c"encode surr -> -8  \00"
@m.encbad2= private unnamed_addr constant [20 x i8] c"encode 110000 -> -8\00"

@m.su.a   = private unnamed_addr constant [16 x i8] c"units A = 1    \00"
@m.su.em  = private unnamed_addr constant [16 x i8] c"units emoji = 2\00"
@m.su.s   = private unnamed_addr constant [16 x i8] c"units surr =-8 \00"
@m.su.o   = private unnamed_addr constant [16 x i8] c"units 110000=-8\00"

@m.bom.le = private unnamed_addr constant [12 x i8] c"bom LE = 1 \00"
@m.bom.be = private unnamed_addr constant [12 x i8] c"bom BE = 2 \00"
@m.bom.no = private unnamed_addr constant [12 x i8] c"bom no = 0 \00"

@m.lfu    = private unnamed_addr constant [20 x i8] c"len_from_utf8 = 5  \00"
@m.ltu    = private unnamed_addr constant [20 x i8] c"len_to_utf8 = 10   \00"
@m.tf     = private unnamed_addr constant [20 x i8] c"from_utf8 units = 5\00"
@m.tt     = private unnamed_addr constant [20 x i8] c"to_utf8 bytes = 10 \00"
@m.rt     = private unnamed_addr constant [20 x i8] c"utf8 roundtrip == 0\00"

@m.rand   = private unnamed_addr constant [24 x i8] c"scalar roundtrip mism=0\00"

@u16.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.u16 = private unnamed_addr constant [24 x i8] c"utf16 from_utf8 n=65536\00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---------------- decode known scalars ----------------
  %da.le = call i64 @universe_utf16_decode_scalar(ptr @d.a.le, i64 1, i32 0)
  call void @ut_check_eq(i64 %da.le, i64 65, ptr @m.a.le)
  %da.be = call i64 @universe_utf16_decode_scalar(ptr @d.a.be, i64 1, i32 1)
  call void @ut_check_eq(i64 %da.be, i64 65, ptr @m.a.be)

  %deu.le = call i64 @universe_utf16_decode_scalar(ptr @d.eu.le, i64 1, i32 0)
  call void @ut_check_eq(i64 %deu.le, i64 8364, ptr @m.eu.le)          ; 0x20AC
  %deu.be = call i64 @universe_utf16_decode_scalar(ptr @d.eu.be, i64 1, i32 1)
  call void @ut_check_eq(i64 %deu.be, i64 8364, ptr @m.eu.be)

  %dem.le = call i64 @universe_utf16_decode_scalar(ptr @d.em.le, i64 2, i32 0)
  call void @ut_check_eq(i64 %dem.le, i64 128512, ptr @m.em.le)        ; 0x1F600
  %dem.be = call i64 @universe_utf16_decode_scalar(ptr @d.em.be, i64 2, i32 1)
  call void @ut_check_eq(i64 %dem.be, i64 128512, ptr @m.em.be)

  %dmx.le = call i64 @universe_utf16_decode_scalar(ptr @d.mx.le, i64 2, i32 0)
  call void @ut_check_eq(i64 %dmx.le, i64 1114111, ptr @m.mx.le)       ; 0x10FFFF
  %dmx.be = call i64 @universe_utf16_decode_scalar(ptr @d.mx.be, i64 2, i32 1)
  call void @ut_check_eq(i64 %dmx.be, i64 1114111, ptr @m.mx.be)

  ; ---------------- decode rejects ----------------
  %r.lohi = call i64 @universe_utf16_decode_scalar(ptr @d.lohi, i64 1, i32 0)
  call void @ut_check_eq(i64 %r.lohi, i64 -13, ptr @m.lohi)
  %r.lolo = call i64 @universe_utf16_decode_scalar(ptr @d.lolo, i64 1, i32 0)
  call void @ut_check_eq(i64 %r.lolo, i64 -13, ptr @m.lolo)
  %r.rev = call i64 @universe_utf16_decode_scalar(ptr @d.rev, i64 2, i32 0)
  call void @ut_check_eq(i64 %r.rev, i64 -13, ptr @m.rev)
  ; truncated pair: a high surrogate with only 1 unit available
  %r.tr = call i64 @universe_utf16_decode_scalar(ptr @d.em.le, i64 1, i32 0)
  call void @ut_check_eq(i64 %r.tr, i64 -13, ptr @m.tr)
  ; empty range
  %r.emp = call i64 @universe_utf16_decode_scalar(ptr @d.a.le, i64 0, i32 0)
  call void @ut_check_eq(i64 %r.emp, i64 -8, ptr @m.emp)

  ; ---------------- encode known scalars ----------------
  ; encode 'A' LE
  %ea.le = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 65, i32 0)
  call void @ut_check_eq(i64 %ea.le, i64 1, ptr @m.enc1)
  %ea.p0 = getelementptr inbounds nuw [65536 x i8], ptr @g.u16, i64 0, i64 0
  %ea.v0 = load i8, ptr %ea.p0, align 1
  %ea.z0 = zext i8 %ea.v0 to i64
  call void @ut_check_eq(i64 %ea.z0, i64 65, ptr @m.encb0)
  %ea.p1 = getelementptr inbounds nuw [65536 x i8], ptr @g.u16, i64 0, i64 1
  %ea.v1 = load i8, ptr %ea.p1, align 1
  %ea.z1 = zext i8 %ea.v1 to i64
  call void @ut_check_eq(i64 %ea.z1, i64 0, ptr @m.encb1)
  ; encode 'A' BE -> bytes swapped
  %eb.le = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 65, i32 1)
  %eb.p0 = getelementptr inbounds nuw [65536 x i8], ptr @g.u16, i64 0, i64 0
  %eb.v0 = load i8, ptr %eb.p0, align 1
  %eb.z0 = zext i8 %eb.v0 to i64
  call void @ut_check_eq(i64 %eb.z0, i64 0, ptr @m.encbb0)
  %eb.p1 = getelementptr inbounds nuw [65536 x i8], ptr @g.u16, i64 0, i64 1
  %eb.v1 = load i8, ptr %eb.p1, align 1
  %eb.z1 = zext i8 %eb.v1 to i64
  call void @ut_check_eq(i64 %eb.z1, i64 65, ptr @m.encbb1)
  ; encode emoji -> 2 units
  %eem = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 128512, i32 0)
  call void @ut_check_eq(i64 %eem, i64 2, ptr @m.enc2)
  ; encode invalid scalars
  %ebad = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 55296, i32 0)   ; 0xD800 surrogate
  call void @ut_check_eq(i64 %ebad, i64 -8, ptr @m.encbad)
  %ebad2 = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 1114112, i32 0) ; 0x110000
  call void @ut_check_eq(i64 %ebad2, i64 -8, ptr @m.encbad2)

  ; ---------------- scalar_units ----------------
  %su.a = call i32 @universe_utf16_scalar_units(i32 65)
  %su.a64 = sext i32 %su.a to i64
  call void @ut_check_eq(i64 %su.a64, i64 1, ptr @m.su.a)
  %su.em = call i32 @universe_utf16_scalar_units(i32 128512)
  %su.em64 = sext i32 %su.em to i64
  call void @ut_check_eq(i64 %su.em64, i64 2, ptr @m.su.em)
  %su.s = call i32 @universe_utf16_scalar_units(i32 56320)                         ; 0xDC00
  %su.s64 = sext i32 %su.s to i64
  call void @ut_check_eq(i64 %su.s64, i64 -8, ptr @m.su.s)
  %su.o = call i32 @universe_utf16_scalar_units(i32 1114112)
  %su.o64 = sext i32 %su.o to i64
  call void @ut_check_eq(i64 %su.o64, i64 -8, ptr @m.su.o)

  ; ---------------- BOM ----------------
  %bl = call i32 @universe_utf16_bom_detect(ptr @bom.le, i64 2)
  %bl64 = sext i32 %bl to i64
  call void @ut_check_eq(i64 %bl64, i64 1, ptr @m.bom.le)
  %bb = call i32 @universe_utf16_bom_detect(ptr @bom.be, i64 2)
  %bb64 = sext i32 %bb to i64
  call void @ut_check_eq(i64 %bb64, i64 2, ptr @m.bom.be)
  %bn = call i32 @universe_utf16_bom_detect(ptr @bom.no, i64 2)
  %bn64 = sext i32 %bn to i64
  call void @ut_check_eq(i64 %bn64, i64 0, ptr @m.bom.no)

  ; ---------------- length helpers + transcode round trip ----------------
  %lfu = call i64 @universe_utf16_len_from_utf8(ptr @u8.mixed, i64 10)
  call void @ut_check_eq(i64 %lfu, i64 5, ptr @m.lfu)
  ; transcode utf8 -> utf16 (LE) into g.u16
  %tf = call i64 @universe_utf16_from_utf8(ptr @g.u16, i64 5, ptr @u8.mixed, i64 10, i32 0)
  call void @ut_check_eq(i64 %tf, i64 5, ptr @m.tf)
  ; measure utf8 bytes from the 5-unit utf16 buffer
  %ltu = call i64 @universe_utf16_len_to_utf8(ptr @g.u16, i64 5, i32 0)
  call void @ut_check_eq(i64 %ltu, i64 10, ptr @m.ltu)
  ; transcode back utf16 -> utf8 into g.u8, compare to original
  %tt = call i64 @universe_utf16_to_utf8(ptr @g.u8, i64 10, ptr @g.u16, i64 5, i32 0)
  call void @ut_check_eq(i64 %tt, i64 10, ptr @m.tt)
  %cmp = call i32 @memcmp(ptr @g.u8, ptr @u8.mixed, i64 10)
  %cmp64 = sext i32 %cmp to i64
  call void @ut_check_eq(i64 %cmp64, i64 0, ptr @m.rt)

  ; ---------------- random scalar round trip ----------------
  ; single deterministic state cell (alloca in entry, never in the loop)
  %scell = alloca i64, align 8
  %gstate = alloca i64, align 8
  store i64 305419896, ptr %scell, align 8
  br label %rt.head

rt.head:
  %ri = phi i64 [ 0, %entry ], [ %ri.next, %rt.body ]
  %rmism = phi i64 [ 0, %entry ], [ %rmism.next, %rt.body ]
  %rdone = icmp uge i64 %ri, 20000
  br i1 %rdone, label %rt.fin, label %rt.body

rt.body:
  %r = call i64 @ut_rand(ptr %scell)
  ; map r into the valid scalar set [0,0x10FFFF] minus surrogates, bijectively
  %m0 = urem i64 %r, 1112032                                  ; 0x10F800
  %m32 = trunc i64 %m0 to i32
  %issur = icmp uge i32 %m32, 55296                           ; >= 0xD800
  %bump = select i1 %issur, i32 2048, i32 0                   ; +0x800 to skip surrogates
  %scalar = add nuw i32 %m32, %bump
  ; alternate endianness on the low bit
  %ber = trunc i64 %r to i32
  %be = and i32 %ber, 1
  %enc = call i64 @universe_utf16_encode_scalar(ptr @g.u16, i32 %scalar, i32 %be)
  %dec = call i64 @universe_utf16_decode_scalar(ptr @g.u16, i64 2, i32 %be)
  %sc64 = zext i32 %scalar to i64
  %eq = icmp eq i64 %dec, %sc64
  %miss = select i1 %eq, i64 0, i64 1
  %rmism.next = add nuw i64 %rmism, %miss
  %ri.next = add nuw i64 %ri, 1
  br label %rt.head

rt.fin:
  call void @ut_check_eq(i64 %rmism, i64 0, ptr @m.rand)

  ; ---------------- SIMD vector-vs-scalar transcode oracle ----------------
  ; Build ~2000 units of random VALID scalars (ASCII-biased so >=16-run ASCII
  ; chunks exercise the vector fast path, mixed with BMP/astral to force the
  ; scalar handoff). Encode each scalar both LE (into o.u16le) and BE (into
  ; o.u16be). Then:
  ;   * to_utf8(LE)   uses the SIMD path; to_utf8(BE) is scalar-only. UTF-8 is
  ;     endianness-free so the two byte streams MUST be identical -> vector==scalar.
  ;   * from_utf8(be=0) uses the SIMD path and must reproduce o.u16le exactly;
  ;     from_utf8(be=1) is scalar-only and must reproduce o.u16be exactly.
  ;   * len_* helpers must agree with the transcoded sizes.
  store i64 2862933555777941757, ptr %gstate, align 8
  br label %vo.gen

vo.gen:
  %gi = phi i64 [ 0, %rt.fin ], [ %gi.n, %vo.body ]
  %gdone = icmp uge i64 %gi, 2000
  br i1 %gdone, label %vo.run, label %vo.body

vo.body:
  %gr = call i64 @ut_rand(ptr %gstate)
  %gsel = and i64 %gr, 7
  %gp = lshr i64 %gr, 3
  ; ascii candidate 0..127
  %o.asc = and i64 %gp, 127
  %o.asc32 = trunc i64 %o.asc to i32
  ; bmp candidate: [0,0xF800) shifted past the surrogate block
  %o.bm0 = urem i64 %gp, 63488
  %o.bm32 = trunc i64 %o.bm0 to i32
  %o.bsur = icmp uge i32 %o.bm32, 55296
  %o.bbump = select i1 %o.bsur, i32 2048, i32 0
  %o.bmp32 = add nuw i32 %o.bm32, %o.bbump
  ; astral candidate: 0x10000 + [0,0x100000)
  %o.as0 = urem i64 %gp, 1048576
  %o.as32 = trunc i64 %o.as0 to i32
  %o.ast32 = add nuw i32 %o.as32, 65536
  ; select: sel==7 astral, sel==6 bmp, else ascii
  %o.isast = icmp eq i64 %gsel, 7
  %o.isbmp = icmp eq i64 %gsel, 6
  %o.sca = select i1 %o.isast, i32 %o.ast32, i32 %o.asc32
  %o.scalar = select i1 %o.isbmp, i32 %o.bmp32, i32 %o.sca
  %o.boff = shl nuw i64 %gi, 1
  %o.ple = getelementptr inbounds nuw i8, ptr @o.u16le, i64 %o.boff
  %o.pbe = getelementptr inbounds nuw i8, ptr @o.u16be, i64 %o.boff
  %o.uw = call i64 @universe_utf16_encode_scalar(ptr %o.ple, i32 %o.scalar, i32 0)
  %o.uwb = call i64 @universe_utf16_encode_scalar(ptr %o.pbe, i32 %o.scalar, i32 1)
  %gi.n = add nuw i64 %gi, %o.uw
  br label %vo.gen

vo.run:
  %U = phi i64 [ %gi, %vo.gen ]
  ; to_utf8: vector (LE) vs scalar (BE); UTF-8 output must be identical bytes
  %nb_le = call i64 @universe_utf16_to_utf8(ptr @o.b1, i64 65536, ptr @o.u16le, i64 %U, i32 0)
  %nb_be = call i64 @universe_utf16_to_utf8(ptr @o.b2, i64 65536, ptr @o.u16be, i64 %U, i32 1)
  %o.nbeq = icmp eq i64 %nb_le, %nb_be
  call void @ut_check(i1 %o.nbeq, ptr @m2.tnb)
  %o.mc1 = call i32 @memcmp(ptr @o.b1, ptr @o.b2, i64 %nb_le)
  %o.mc1z = icmp eq i32 %o.mc1, 0
  call void @ut_check(i1 %o.mc1z, ptr @m2.tby)
  %o.ltu = call i64 @universe_utf16_len_to_utf8(ptr @o.u16le, i64 %U, i32 0)
  %o.ltueq = icmp eq i64 %o.ltu, %nb_le
  call void @ut_check(i1 %o.ltueq, ptr @m2.ltu)
  ; from_utf8: vector (be=0) reproduces LE ref; scalar (be=1) reproduces BE ref
  %un_le = call i64 @universe_utf16_from_utf8(ptr @o.o1, i64 65536, ptr @o.b1, i64 %nb_le, i32 0)
  %un_be = call i64 @universe_utf16_from_utf8(ptr @o.o2, i64 65536, ptr @o.b1, i64 %nb_le, i32 1)
  %o.uleq = icmp eq i64 %un_le, %U
  call void @ut_check(i1 %o.uleq, ptr @m2.fvu)
  %o.ubeq = icmp eq i64 %un_be, %U
  call void @ut_check(i1 %o.ubeq, ptr @m2.fsu)
  %o.ub = shl nuw i64 %U, 1
  %o.mc2 = call i32 @memcmp(ptr @o.o1, ptr @o.u16le, i64 %o.ub)
  %o.mc2z = icmp eq i32 %o.mc2, 0
  call void @ut_check(i1 %o.mc2z, ptr @m2.fvb)
  %o.mc3 = call i32 @memcmp(ptr @o.o2, ptr @o.u16be, i64 %o.ub)
  %o.mc3z = icmp eq i32 %o.mc3, 0
  call void @ut_check(i1 %o.mc3z, ptr @m2.fsb)
  %o.lfu = call i64 @universe_utf16_len_from_utf8(ptr @o.b1, i64 %nb_le)
  %o.lfueq = icmp eq i64 %o.lfu, %U
  call void @ut_check(i1 %o.lfueq, ptr @m2.lfu)

  ; ---------------- bench ----------------
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %bf.head

bf.head:
  %bi = phi i64 [ 0, %bench ], [ %bi.next, %bf.body ]
  %bdone = icmp uge i64 %bi, 65536
  br i1 %bdone, label %bench.run, label %bf.body

bf.body:
  ; repeating 4-byte emoji F0 9F 98 80
  %bm = and i64 %bi, 3
  %bsp = getelementptr inbounds nuw [4 x i8], ptr @d.em.pat, i64 0, i64 %bm
  %bv = load i8, ptr %bsp, align 1
  %bdp = getelementptr inbounds nuw [65536 x i8], ptr @g.u8, i64 0, i64 %bi
  store i8 %bv, ptr %bdp, align 1
  %bi.next = add nuw i64 %bi, 1
  br label %bf.head

bench.run:
  ; 17 reps of a 1000-conversion batch; discard rep 0 (warm-up), report the
  ; distribution over the remaining 16. ops/rep = 1000 * 65536 = 65536000 (ns/byte).
  br label %cv.rep

cv.rep:
  %crep = phi i64 [ 0, %bench.run ], [ %crep.n, %cv.next ]
  %t0 = call double @ut_now_sec()
  br label %br.head

br.head:
  %bc = phi i64 [ 0, %cv.rep ], [ %bc.next, %br.head ]
  %bres = call i64 @universe_utf16_from_utf8(ptr @g.u16, i64 32768, ptr @g.u8, i64 65536, i32 0)
  %bc.next = add nuw i64 %bc, 1
  %bmore = icmp ult i64 %bc.next, 1000
  br i1 %bmore, label %br.head, label %cv.rep.done

cv.rep.done:
  %t1 = call double @ut_now_sec()
  %cel = fsub double %t1, %t0
  %ckeep = icmp ugt i64 %crep, 0
  br i1 %ckeep, label %cv.store, label %cv.next

cv.store:
  %cidx = sub i64 %crep, 1
  %csp = getelementptr inbounds [16 x double], ptr @u16.samp, i64 0, i64 %cidx
  store double %cel, ptr %csp, align 8
  br label %cv.next

cv.next:
  %crep.n = add nuw i64 %crep, 1
  %cmore = icmp ult i64 %crep.n, 17
  br i1 %cmore, label %cv.rep, label %cv.report

cv.report:
  call void @ut_report_dist(ptr @u16.samp, i64 16, i64 65536000, ptr @lbl.u16)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

@d.em.pat = private unnamed_addr constant [4 x i8] c"\F0\9F\98\80", align 1
