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

; Tests for universe_utf8_*: valid 1/2/3/4-byte sequences pass; crafted invalid
; sequences (overlong, lone continuation, truncated, surrogate, out-of-range)
; are rejected at the expected byte index; count_codepoints matches a hand
; count; byte_len_of_codepoint classifies leads. --bench vs a naive validator.

declare i64 @universe_utf8_validate(ptr, i64)
declare i64 @universe_utf8_count_codepoints(ptr, i64)
declare i32 @universe_utf8_byte_len_of_codepoint(i8)

declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@g.buf = internal global [65536 x i8] zeroinitializer, align 16

; ---- valid sequences ----
@v.ascii = private unnamed_addr constant [2 x i8] c"Aa", align 1
@v.two   = private unnamed_addr constant [2 x i8] c"\C3\A9", align 1          ; U+00E9
@v.three = private unnamed_addr constant [3 x i8] c"\E2\82\AC", align 1       ; U+20AC
@v.four  = private unnamed_addr constant [4 x i8] c"\F0\9F\98\80", align 1    ; U+1F600
@v.mixed = private unnamed_addr constant [10 x i8] c"A\E2\82\AC\F0\9F\98\80\C3\A9", align 1

; ---- invalid sequences ----
@i.cont    = private unnamed_addr constant [1 x i8] c"\80", align 1
@i.c0      = private unnamed_addr constant [2 x i8] c"\C0\80", align 1
@i.e0over  = private unnamed_addr constant [3 x i8] c"\E0\80\80", align 1
@i.f0over  = private unnamed_addr constant [4 x i8] c"\F0\80\80\80", align 1
@i.surr    = private unnamed_addr constant [3 x i8] c"\ED\A0\80", align 1
@i.f4over  = private unnamed_addr constant [4 x i8] c"\F4\90\80\80", align 1
@i.f5      = private unnamed_addr constant [4 x i8] c"\F5\80\80\80", align 1
@i.trunc2  = private unnamed_addr constant [1 x i8] c"\C3", align 1
@i.trunc3  = private unnamed_addr constant [2 x i8] c"\E2\82", align 1
@i.badcont = private unnamed_addr constant [2 x i8] c"\C3\20", align 1

@m.va   = private unnamed_addr constant [12 x i8] c"ascii valid\00"
@m.v2   = private unnamed_addr constant [13 x i8] c"2-byte valid\00"
@m.v3   = private unnamed_addr constant [13 x i8] c"3-byte valid\00"
@m.v4   = private unnamed_addr constant [13 x i8] c"4-byte valid\00"
@m.vm   = private unnamed_addr constant [12 x i8] c"mixed valid\00"
@m.ic   = private unnamed_addr constant [17 x i8] c"lone cont bad @0\00"
@m.ic0  = private unnamed_addr constant [19 x i8] c"C0 overlong bad @0\00"
@m.ie0  = private unnamed_addr constant [19 x i8] c"E0 overlong bad @1\00"
@m.if0  = private unnamed_addr constant [19 x i8] c"F0 overlong bad @1\00"
@m.isr  = private unnamed_addr constant [17 x i8] c"surrogate bad @1\00"
@m.if4  = private unnamed_addr constant [16 x i8] c"F4 range bad @1\00"
@m.if5  = private unnamed_addr constant [15 x i8] c"F5 lead bad @0\00"
@m.it2  = private unnamed_addr constant [20 x i8] c"trunc 2-byte bad @0\00"
@m.it3  = private unnamed_addr constant [20 x i8] c"trunc 3-byte bad @0\00"
@m.ibc  = private unnamed_addr constant [16 x i8] c"bad cont bad @1\00"
@m.cnt  = private unnamed_addr constant [16 x i8] c"count mixed = 4\00"
@m.cnt0 = private unnamed_addr constant [16 x i8] c"count empty = 0\00"
@m.cntb = private unnamed_addr constant [19 x i8] c"count invalid = -1\00"
@m.emp  = private unnamed_addr constant [15 x i8] c"empty valid -1\00"
@m.bl1  = private unnamed_addr constant [13 x i8] c"byte_len A=1\00"
@m.bl2  = private unnamed_addr constant [14 x i8] c"byte_len C3=2\00"
@m.bl3  = private unnamed_addr constant [14 x i8] c"byte_len E2=3\00"
@m.bl4  = private unnamed_addr constant [14 x i8] c"byte_len F0=4\00"
@m.bl0c = private unnamed_addr constant [14 x i8] c"byte_len 80=0\00"
@m.bl0x = private unnamed_addr constant [14 x i8] c"byte_len C0=0\00"
@m.bl0f = private unnamed_addr constant [14 x i8] c"byte_len F5=0\00"
@m.oav  = private unnamed_addr constant [25 x i8] c"ascii oracle valid == -1\00"
@m.oac  = private unnamed_addr constant [23 x i8] c"ascii oracle count = n\00"
@m.oai  = private unnamed_addr constant [24 x i8] c"ascii oracle bad pos ==\00"
@oalens = internal constant [7 x i64] [ i64 0, i64 15, i64 16, i64 17, i64 64, i64 1000, i64 65536 ], align 8
@u8.msamp = internal global [16 x double] zeroinitializer, align 8
@u8.asamp = internal global [16 x double] zeroinitializer, align 8
@lbl.u8mixed = private unnamed_addr constant [26 x i8] c"utf8 validate 4byte 65536\00"
@lbl.u8ascii = private unnamed_addr constant [26 x i8] c"utf8 validate ascii 65536\00"
@m.mxv = private unnamed_addr constant [24 x i8] c"mixed multibyte valid-1\00"
@m.mxc = private unnamed_addr constant [24 x i8] c"mixed count == tracked \00"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; ---- valid ----
  %a1 = call i64 @universe_utf8_validate(ptr @v.ascii, i64 2)
  call void @ut_check_eq(i64 %a1, i64 -1, ptr @m.va)
  %a2 = call i64 @universe_utf8_validate(ptr @v.two, i64 2)
  call void @ut_check_eq(i64 %a2, i64 -1, ptr @m.v2)
  %a3 = call i64 @universe_utf8_validate(ptr @v.three, i64 3)
  call void @ut_check_eq(i64 %a3, i64 -1, ptr @m.v3)
  %a4 = call i64 @universe_utf8_validate(ptr @v.four, i64 4)
  call void @ut_check_eq(i64 %a4, i64 -1, ptr @m.v4)
  %am = call i64 @universe_utf8_validate(ptr @v.mixed, i64 10)
  call void @ut_check_eq(i64 %am, i64 -1, ptr @m.vm)
  %ae = call i64 @universe_utf8_validate(ptr @v.ascii, i64 0)
  call void @ut_check_eq(i64 %ae, i64 -1, ptr @m.emp)

  ; ---- invalid, at expected index ----
  %b1 = call i64 @universe_utf8_validate(ptr @i.cont, i64 1)
  call void @ut_check_eq(i64 %b1, i64 0, ptr @m.ic)
  %b2 = call i64 @universe_utf8_validate(ptr @i.c0, i64 2)
  call void @ut_check_eq(i64 %b2, i64 0, ptr @m.ic0)
  %b3 = call i64 @universe_utf8_validate(ptr @i.e0over, i64 3)
  call void @ut_check_eq(i64 %b3, i64 1, ptr @m.ie0)
  %b4 = call i64 @universe_utf8_validate(ptr @i.f0over, i64 4)
  call void @ut_check_eq(i64 %b4, i64 1, ptr @m.if0)
  %b5 = call i64 @universe_utf8_validate(ptr @i.surr, i64 3)
  call void @ut_check_eq(i64 %b5, i64 1, ptr @m.isr)
  %b6 = call i64 @universe_utf8_validate(ptr @i.f4over, i64 4)
  call void @ut_check_eq(i64 %b6, i64 1, ptr @m.if4)
  %b7 = call i64 @universe_utf8_validate(ptr @i.f5, i64 4)
  call void @ut_check_eq(i64 %b7, i64 0, ptr @m.if5)
  %b8 = call i64 @universe_utf8_validate(ptr @i.trunc2, i64 1)
  call void @ut_check_eq(i64 %b8, i64 0, ptr @m.it2)
  %b9 = call i64 @universe_utf8_validate(ptr @i.trunc3, i64 2)
  call void @ut_check_eq(i64 %b9, i64 0, ptr @m.it3)
  %b10 = call i64 @universe_utf8_validate(ptr @i.badcont, i64 2)
  call void @ut_check_eq(i64 %b10, i64 1, ptr @m.ibc)

  ; ---- count_codepoints ----
  %c1 = call i64 @universe_utf8_count_codepoints(ptr @v.mixed, i64 10)
  call void @ut_check_eq(i64 %c1, i64 4, ptr @m.cnt)
  %c2 = call i64 @universe_utf8_count_codepoints(ptr @v.ascii, i64 0)
  call void @ut_check_eq(i64 %c2, i64 0, ptr @m.cnt0)
  %c3 = call i64 @universe_utf8_count_codepoints(ptr @i.surr, i64 3)
  call void @ut_check_eq(i64 %c3, i64 -1, ptr @m.cntb)

  ; ---- byte_len_of_codepoint ----
  %l1 = call i32 @universe_utf8_byte_len_of_codepoint(i8 65)
  %l1e = zext i32 %l1 to i64
  call void @ut_check_eq(i64 %l1e, i64 1, ptr @m.bl1)
  %l2 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -61)   ; 0xC3
  %l2e = zext i32 %l2 to i64
  call void @ut_check_eq(i64 %l2e, i64 2, ptr @m.bl2)
  %l3 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -30)   ; 0xE2
  %l3e = zext i32 %l3 to i64
  call void @ut_check_eq(i64 %l3e, i64 3, ptr @m.bl3)
  %l4 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -16)   ; 0xF0
  %l4e = zext i32 %l4 to i64
  call void @ut_check_eq(i64 %l4e, i64 4, ptr @m.bl4)
  %l5 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -128)  ; 0x80
  %l5e = zext i32 %l5 to i64
  call void @ut_check_eq(i64 %l5e, i64 0, ptr @m.bl0c)
  %l6 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -64)   ; 0xC0
  %l6e = zext i32 %l6 to i64
  call void @ut_check_eq(i64 %l6e, i64 0, ptr @m.bl0x)
  %l7 = call i32 @universe_utf8_byte_len_of_codepoint(i8 -11)   ; 0xF5
  %l7e = zext i32 %l7 to i64
  call void @ut_check_eq(i64 %l7e, i64 0, ptr @m.bl0f)

  ; ---- oracle: vector ASCII fast-path vs known answers ----
  ; Pure-ASCII buffers must validate (-1) and count == n; injecting a lone
  ; continuation byte (0x80) at a random position must be reported at that
  ; index, exercising the vector-skip -> scalar-handoff boundary.
  %ostate = alloca i64, align 8
  store i64 2862933555777941757, ptr %ostate, align 8
  br label %oo.head

oo.head:
  %oi = phi i64 [ 0, %entry ], [ %oi.n, %oo.next ]
  %vm = phi i64 [ 0, %entry ], [ %vm.n, %oo.next ]
  %cm = phi i64 [ 0, %entry ], [ %cm.n, %oo.next ]
  %im = phi i64 [ 0, %entry ], [ %im.n, %oo.next ]
  %olp = getelementptr inbounds nuw [7 x i64], ptr @oalens, i64 0, i64 %oi
  %oL = load i64, ptr %olp, align 8
  br label %of.head

of.head:
  %ofi = phi i64 [ 0, %oo.head ], [ %ofi.n, %of.body ]
  %ofd = icmp uge i64 %ofi, %oL
  br i1 %ofd, label %oo.chk, label %of.body

of.body:
  %orr = call i64 @ut_rand(ptr %ostate)
  %orb = trunc i64 %orr to i8
  %orasc = and i8 %orb, 127                         ; force ASCII
  %odp = getelementptr inbounds nuw [65536 x i8], ptr @g.buf, i64 0, i64 %ofi
  store i8 %orasc, ptr %odp, align 1
  %ofi.n = add nuw i64 %ofi, 1
  br label %of.head

oo.chk:
  %rv = call i64 @universe_utf8_validate(ptr @g.buf, i64 %oL)
  %vbad = icmp ne i64 %rv, -1
  %vb = zext i1 %vbad to i64
  %vm.n = add nuw i64 %vm, %vb
  %ocnt = call i64 @universe_utf8_count_codepoints(ptr @g.buf, i64 %oL)
  %cbad = icmp ne i64 %ocnt, %oL
  %cb = zext i1 %cbad to i64
  %cm.n = add nuw i64 %cm, %cb
  %ohas = icmp ne i64 %oL, 0
  br i1 %ohas, label %oo.inj, label %oo.next

oo.inj:
  %opr = call i64 @ut_rand(ptr %ostate)
  %opp = urem i64 %opr, %oL
  %opdp = getelementptr inbounds nuw [65536 x i8], ptr @g.buf, i64 0, i64 %opp
  store i8 -128, ptr %opdp, align 1                 ; 0x80 lone continuation
  %rv2 = call i64 @universe_utf8_validate(ptr @g.buf, i64 %oL)
  %ibad = icmp ne i64 %rv2, %opp
  %ib = zext i1 %ibad to i64
  br label %oo.next

oo.next:
  %imadd = phi i64 [ %ib, %oo.inj ], [ 0, %oo.chk ]
  %im.n = add nuw i64 %im, %imadd
  %oi.n = add nuw i64 %oi, 1
  %omore = icmp ult i64 %oi.n, 7
  br i1 %omore, label %oo.head, label %oo.fin

oo.fin:
  call void @ut_check_eq(i64 %vm.n, i64 0, ptr @m.oav)
  call void @ut_check_eq(i64 %cm.n, i64 0, ptr @m.oac)
  call void @ut_check_eq(i64 %im.n, i64 0, ptr @m.oai)

  ; ---- mixed-multibyte oracle: exercise the vector ASCII-skip -> scalar
  ; multibyte handoff. Build a buffer of random VALID codepoints (ASCII-biased
  ; so >=16-run ASCII chunks hit the vector fast path, interleaved with 2/3/4-
  ; byte sequences forcing the scalar path), tracking the exact codepoint count.
  ; The fused vector+scalar validate MUST accept (-1) and count MUST equal the
  ; independently tracked count. ----
  store i64 88172645463325252, ptr %ostate, align 8
  br label %mg.head

mg.head:
  %mgi = phi i64 [ 0, %oo.fin ], [ %mgi.nx, %mg.next ]
  %mgc = phi i64 [ 0, %oo.fin ], [ %mgc.nx, %mg.next ]
  %mgstop = icmp uge i64 %mgc, 6000
  %mgp4 = add nuw i64 %mgi, 4
  %mgroom = icmp ule i64 %mgp4, 65500
  %mgok = xor i1 %mgstop, true
  %mggo = and i1 %mgok, %mgroom
  br i1 %mggo, label %mg.body, label %mg.done

mg.body:
  %mr = call i64 @ut_rand(ptr %ostate)
  %mk = and i64 %mr, 7
  %mp = lshr i64 %mr, 3
  %dp0 = getelementptr inbounds nuw [65536 x i8], ptr @g.buf, i64 0, i64 %mgi
  switch i64 %mk, label %mg.ascii [ i64 5, label %mg.two
                                    i64 6, label %mg.three
                                    i64 7, label %mg.four ]

mg.ascii:
  %ac = and i64 %mp, 127
  %ac8 = trunc i64 %ac to i8
  store i8 %ac8, ptr %dp0, align 1
  br label %mg.next

mg.two:                                            ; U+00E9 = C3 A9
  store i8 -61, ptr %dp0, align 1
  %tp1 = getelementptr inbounds nuw i8, ptr %dp0, i64 1
  store i8 -87, ptr %tp1, align 1
  br label %mg.next

mg.three:                                          ; U+20AC = E2 82 AC
  store i8 -30, ptr %dp0, align 1
  %hp1 = getelementptr inbounds nuw i8, ptr %dp0, i64 1
  store i8 -126, ptr %hp1, align 1
  %hp2 = getelementptr inbounds nuw i8, ptr %dp0, i64 2
  store i8 -84, ptr %hp2, align 1
  br label %mg.next

mg.four:                                           ; U+1F600 = F0 9F 98 80
  store i8 -16, ptr %dp0, align 1
  %fp1 = getelementptr inbounds nuw i8, ptr %dp0, i64 1
  store i8 -97, ptr %fp1, align 1
  %fp2 = getelementptr inbounds nuw i8, ptr %dp0, i64 2
  store i8 -104, ptr %fp2, align 1
  %fp3 = getelementptr inbounds nuw i8, ptr %dp0, i64 3
  store i8 -128, ptr %fp3, align 1
  br label %mg.next

mg.next:
  %mlen = phi i64 [ 1, %mg.ascii ], [ 2, %mg.two ], [ 3, %mg.three ], [ 4, %mg.four ]
  %mgi.nx = add nuw i64 %mgi, %mlen
  %mgc.nx = add nuw i64 %mgc, 1
  br label %mg.head

mg.done:
  %mgv = call i64 @universe_utf8_validate(ptr @g.buf, i64 %mgi)
  %mgveq = icmp eq i64 %mgv, -1
  call void @ut_check(i1 %mgveq, ptr @m.mxv)
  %mgcnt = call i64 @universe_utf8_count_codepoints(ptr @g.buf, i64 %mgi)
  %mgceq = icmp eq i64 %mgcnt, %mgc
  call void @ut_check(i1 %mgceq, ptr @m.mxc)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  ; fill buffer with a repeating valid 4-byte codepoint (F0 9F 98 80)
  br label %bf.head

bf.head:
  %bi = phi i64 [ 0, %bench ], [ %bi.next, %bf.body ]
  %bdone = icmp uge i64 %bi, 65536
  br i1 %bdone, label %bench.run, label %bf.body

bf.body:
  %m = and i64 %bi, 3
  %sp = getelementptr inbounds nuw [4 x i8], ptr @v.four, i64 0, i64 %m
  %bv = load i8, ptr %sp, align 1
  %dp = getelementptr inbounds nuw [65536 x i8], ptr @g.buf, i64 0, i64 %bi
  store i8 %bv, ptr %dp, align 1
  %bi.next = add nuw i64 %bi, 1
  br label %bf.head

bench.run:
  ; 4-byte-codepoint validate: 17 reps of a 1000-validate batch; discard rep 0
  ; (warm-up), report over the remaining 16. ops/rep = 1000 * 65536 = 65536000.
  br label %mv.rep

mv.rep:
  %mrep = phi i64 [ 0, %bench.run ], [ %mrep.n, %mv.next ]
  %t0 = call double @ut_now_sec()
  br label %bv.head

bv.head:
  %bvc = phi i64 [ 0, %mv.rep ], [ %bvc.next, %bv.head ]
  %bres = call i64 @universe_utf8_validate(ptr @g.buf, i64 65536)
  %bvc.next = add nuw i64 %bvc, 1
  %bvmore = icmp ult i64 %bvc.next, 1000
  br i1 %bvmore, label %bv.head, label %mv.rep.done

mv.rep.done:
  %t1 = call double @ut_now_sec()
  %mel = fsub double %t1, %t0
  %mkeep = icmp ugt i64 %mrep, 0
  br i1 %mkeep, label %mv.store, label %mv.next

mv.store:
  %midx = sub i64 %mrep, 1
  %msp = getelementptr inbounds [16 x double], ptr @u8.msamp, i64 0, i64 %midx
  store double %mel, ptr %msp, align 8
  br label %mv.next

mv.next:
  %mrep.n = add nuw i64 %mrep, 1
  %mmore = icmp ult i64 %mrep.n, 17
  br i1 %mmore, label %mv.rep, label %mv.report

mv.report:
  call void @ut_report_dist(ptr @u8.msamp, i64 16, i64 65536000, ptr @lbl.u8mixed)
  br label %ab.head

; ---- ASCII bench: exercises the vector fast-path (common-case input) ----
ab.head:
  %abi = phi i64 [ 0, %mv.report ], [ %abi.next, %ab.head ]
  %abb = trunc i64 %abi to i8
  %aba = and i8 %abb, 127
  %abp = getelementptr inbounds nuw [65536 x i8], ptr @g.buf, i64 0, i64 %abi
  store i8 %aba, ptr %abp, align 1
  %abi.next = add nuw i64 %abi, 1
  %abmore = icmp ult i64 %abi.next, 65536
  br i1 %abmore, label %ab.head, label %av.rep

av.rep:
  %arep = phi i64 [ 0, %ab.head ], [ %arep.n, %av.next ]
  %at0 = call double @ut_now_sec()
  br label %av.head

av.head:
  %avc = phi i64 [ 0, %av.rep ], [ %avc.next, %av.head ]
  %ares = call i64 @universe_utf8_validate(ptr @g.buf, i64 65536)
  %avc.next = add nuw i64 %avc, 1
  %avmore = icmp ult i64 %avc.next, 1000
  br i1 %avmore, label %av.head, label %av.rep.done

av.rep.done:
  %at1 = call double @ut_now_sec()
  %ael = fsub double %at1, %at0
  %akeep = icmp ugt i64 %arep, 0
  br i1 %akeep, label %av.store, label %av.next

av.store:
  %aidx = sub i64 %arep, 1
  %asp = getelementptr inbounds [16 x double], ptr @u8.asamp, i64 0, i64 %aidx
  store double %ael, ptr %asp, align 8
  br label %av.next

av.next:
  %arep.n = add nuw i64 %arep, 1
  %amore = icmp ult i64 %arep.n, 17
  br i1 %amore, label %av.rep, label %av.report

av.report:
  call void @ut_report_dist(ptr @u8.asamp, i64 16, i64 65536000, ptr @lbl.u8ascii)
  br label %fin

fin:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
