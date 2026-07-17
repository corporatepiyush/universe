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

; Tests for universe_string (view + SSO). Covers view eq/compare/index/hash/
; starts_with/substring, SSO inline<->heap boundary (len 0,21,22,23,100) with
; _data verified on both sides, raw multibyte UTF-8 bytes, error codes, and a
; --bench of SSO inline create/read vs heap.

declare i64 @universe_string_len(ptr, i64)
declare i1 @universe_string_eq(ptr, i64, ptr, i64)
declare i32 @universe_string_compare(ptr, i64, ptr, i64)
declare i64 @universe_string_index_of_byte(ptr, i64, i8)
declare i64 @universe_string_count_byte(ptr, i64, i8)
declare i64 @universe_string_hash(ptr, i64)
declare i1 @universe_string_starts_with(ptr, i64, ptr, i64)
declare { ptr, i64 } @universe_string_substring_view(ptr, i64, i64, i64)

declare i32 @universe_sso_create(ptr, ptr, i64)
declare i64 @universe_sso_len(ptr)
declare ptr @universe_sso_data(ptr)
declare void @universe_sso_free(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)
declare i64 @llvm.umin.i64(i64, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@d.abc    = private unnamed_addr constant [3 x i8] c"abc"
@d.abd    = private unnamed_addr constant [3 x i8] c"abd"
@d.ab     = private unnamed_addr constant [2 x i8] c"ab"
@d.ax     = private unnamed_addr constant [2 x i8] c"ax"
@d.abcdef = private unnamed_addr constant [6 x i8] c"abcdef"
@d.hello  = private unnamed_addr constant [5 x i8] c"hello"
@d.hellp  = private unnamed_addr constant [5 x i8] c"hellp"
@d.Hello  = private unnamed_addr constant [5 x i8] c"Hello"
@d.euro   = private unnamed_addr constant [3 x i8] c"\E2\82\AC"    ; U+20AC euro sign

@s.len      = private unnamed_addr constant [9 x i8]  c"view len\00"
@s.eq       = private unnamed_addr constant [9 x i8]  c"eq equal\00"
@s.ne1      = private unnamed_addr constant [12 x i8] c"eq diff len\00"
@s.ne2      = private unnamed_addr constant [16 x i8] c"eq diff content\00"
@s.cmpeq    = private unnamed_addr constant [16 x i8] c"compare equal 0\00"
@s.cmplt    = private unnamed_addr constant [13 x i8] c"compare less\00"
@s.cmpgt    = private unnamed_addr constant [16 x i8] c"compare greater\00"
@s.cmppre   = private unnamed_addr constant [15 x i8] c"compare prefix\00"
@s.idx      = private unnamed_addr constant [12 x i8] c"index found\00"
@s.idxno    = private unnamed_addr constant [17 x i8] c"index missing -1\00"
@s.hashdet  = private unnamed_addr constant [19 x i8] c"hash deterministic\00"
@s.hashdiff = private unnamed_addr constant [14 x i8] c"hash distinct\00"
@s.sw1      = private unnamed_addr constant [16 x i8] c"starts_with yes\00"
@s.sw2      = private unnamed_addr constant [15 x i8] c"starts_with no\00"
@s.sw3      = private unnamed_addr constant [17 x i8] c"starts_with long\00"
@s.sub      = private unnamed_addr constant [15 x i8] c"substring view\00"
@s.subclamp = private unnamed_addr constant [16 x i8] c"substring clamp\00"
@s.ssolen   = private unnamed_addr constant [14 x i8] c"sso len match\00"
@s.ssodata  = private unnamed_addr constant [12 x i8] c"sso data ok\00"
@s.ssoptr   = private unnamed_addr constant [15 x i8] c"sso ptr rep ok\00"
@s.ssonull  = private unnamed_addr constant [15 x i8] c"sso null out 1\00"
@s.ssosrc   = private unnamed_addr constant [15 x i8] c"sso null src 1\00"
@s.utf8     = private unnamed_addr constant [14 x i8] c"utf8 bytes ok\00"
@o.lens = private unnamed_addr constant [8 x i64] [i64 0, i64 1, i64 15, i64 16, i64 17, i64 64, i64 255, i64 1024]
@s.oracle = private unnamed_addr constant [26 x i8] c"simd vs scalar oracle (0)\00"
@sso.shsamp = internal global [16 x double] zeroinitializer, align 8
@sso.hpsamp = internal global [16 x double] zeroinitializer, align 8
@sso.sink = internal global i64 0, align 8
@lbl.ssosh = private unnamed_addr constant [20 x i8] c"sso short create 1M\00"
@lbl.ssohp = private unnamed_addr constant [19 x i8] c"sso heap create 1M\00"

define internal void @test_view() {
entry:
  ; len
  %l = call i64 @universe_string_len(ptr @d.abc, i64 3)
  call void @ut_check_eq(i64 %l, i64 3, ptr @s.len)

  ; eq
  %e1 = call i1 @universe_string_eq(ptr @d.abc, i64 3, ptr @d.abc, i64 3)
  call void @ut_check(i1 %e1, ptr @s.eq)
  %e2 = call i1 @universe_string_eq(ptr @d.ab, i64 2, ptr @d.abc, i64 3)
  %e2.n = xor i1 %e2, true
  call void @ut_check(i1 %e2.n, ptr @s.ne1)
  %e3 = call i1 @universe_string_eq(ptr @d.abc, i64 3, ptr @d.abd, i64 3)
  %e3.n = xor i1 %e3, true
  call void @ut_check(i1 %e3.n, ptr @s.ne2)

  ; compare
  %c0 = call i32 @universe_string_compare(ptr @d.abc, i64 3, ptr @d.abc, i64 3)
  %c0.ok = icmp eq i32 %c0, 0
  call void @ut_check(i1 %c0.ok, ptr @s.cmpeq)
  %c1 = call i32 @universe_string_compare(ptr @d.abc, i64 3, ptr @d.abd, i64 3)
  %c1.ok = icmp slt i32 %c1, 0
  call void @ut_check(i1 %c1.ok, ptr @s.cmplt)
  %c2 = call i32 @universe_string_compare(ptr @d.abd, i64 3, ptr @d.abc, i64 3)
  %c2.ok = icmp sgt i32 %c2, 0
  call void @ut_check(i1 %c2.ok, ptr @s.cmpgt)
  %c3 = call i32 @universe_string_compare(ptr @d.ab, i64 2, ptr @d.abc, i64 3)
  %c3.ok = icmp slt i32 %c3, 0
  call void @ut_check(i1 %c3.ok, ptr @s.cmppre)

  ; index_of_byte
  %i0 = call i64 @universe_string_index_of_byte(ptr @d.abc, i64 3, i8 99)  ; 'c'
  call void @ut_check_eq(i64 %i0, i64 2, ptr @s.idx)
  %i1 = call i64 @universe_string_index_of_byte(ptr @d.abc, i64 3, i8 122) ; 'z'
  call void @ut_check_eq(i64 %i1, i64 -1, ptr @s.idxno)

  ; starts_with
  %w1 = call i1 @universe_string_starts_with(ptr @d.abcdef, i64 6, ptr @d.ab, i64 2)
  call void @ut_check(i1 %w1, ptr @s.sw1)
  %w2 = call i1 @universe_string_starts_with(ptr @d.abcdef, i64 6, ptr @d.ax, i64 2)
  %w2.n = xor i1 %w2, true
  call void @ut_check(i1 %w2.n, ptr @s.sw2)
  %w3 = call i1 @universe_string_starts_with(ptr @d.ab, i64 2, ptr @d.abcdef, i64 6)
  %w3.n = xor i1 %w3, true
  call void @ut_check(i1 %w3.n, ptr @s.sw3)

  ; substring_view: "abcdef"[2..2+3) = "cde"
  %sv = call { ptr, i64 } @universe_string_substring_view(ptr @d.abcdef, i64 6, i64 2, i64 3)
  %sv.p = extractvalue { ptr, i64 } %sv, 0
  %sv.l = extractvalue { ptr, i64 } %sv, 1
  %sv.b0 = load i8, ptr %sv.p, align 1
  %sv.lok = icmp eq i64 %sv.l, 3
  %sv.bok = icmp eq i8 %sv.b0, 99   ; 'c'
  %sv.ok = and i1 %sv.lok, %sv.bok
  call void @ut_check(i1 %sv.ok, ptr @s.sub)
  ; clamp: start beyond len -> len 0
  %svc = call { ptr, i64 } @universe_string_substring_view(ptr @d.abcdef, i64 6, i64 10, i64 5)
  %svc.l = extractvalue { ptr, i64 } %svc, 1
  %svc.ok = icmp eq i64 %svc.l, 0
  call void @ut_check(i1 %svc.ok, ptr @s.subclamp)
  ret void
}

define internal void @test_hash() {
entry:
  ; determinism: same bytes -> same hash
  %h1 = call i64 @universe_string_hash(ptr @d.hello, i64 5)
  %h2 = call i64 @universe_string_hash(ptr @d.hello, i64 5)
  %det = icmp eq i64 %h1, %h2
  call void @ut_check(i1 %det, ptr @s.hashdet)

  ; distinctness across a small sample (hello, hellp, Hello, empty)
  %h3 = call i64 @universe_string_hash(ptr @d.hellp, i64 5)
  %h4 = call i64 @universe_string_hash(ptr @d.Hello, i64 5)
  %h5 = call i64 @universe_string_hash(ptr @d.hello, i64 0)   ; empty
  %d12 = icmp ne i64 %h1, %h3
  %d13 = icmp ne i64 %h1, %h4
  %d14 = icmp ne i64 %h1, %h5
  %d34 = icmp ne i64 %h3, %h4
  %d35 = icmp ne i64 %h3, %h5
  %d45 = icmp ne i64 %h4, %h5
  %a0 = and i1 %d12, %d13
  %a1 = and i1 %a0, %d14
  %a2 = and i1 %a1, %d34
  %a3 = and i1 %a2, %d35
  %a4 = and i1 %a3, %d45
  call void @ut_check(i1 %a4, ptr @s.hashdiff)
  ret void
}

; Build a 24-byte SSO from ref[0..len), verify _len, _data bytes, and the
; inline/heap pointer relationship. expheap=false => _data must equal &s;
; expheap=true => _data must differ from &s (points into the heap block).
define internal void @check_sso(ptr %ref, i64 %len, i1 %expheap) {
entry:
  %s = alloca [24 x i8], align 8
  %rc = call i32 @universe_sso_create(ptr %s, ptr %ref, i64 %len)
  %l = call i64 @universe_sso_len(ptr %s)
  call void @ut_check_eq(i64 %l, i64 %len, ptr @s.ssolen)
  %d = call ptr @universe_sso_data(ptr %s)
  br label %cmp

cmp:
  %empty = icmp eq i64 %len, 0
  br i1 %empty, label %ptrcheck, label %bytecheck

bytecheck:
  %m = call i32 @memcmp(ptr %d, ptr %ref, i64 %len)
  %m.ok = icmp eq i32 %m, 0
  call void @ut_check(i1 %m.ok, ptr @s.ssodata)
  br label %ptrcheck

ptrcheck:
  %eq = icmp eq ptr %d, %s
  %ptr.ok = xor i1 %eq, %expheap    ; inline: want eq; heap: want !eq
  call void @ut_check(i1 %ptr.ok, ptr @s.ssoptr)
  call void @universe_sso_free(ptr %s)
  ret void
}

define internal void @test_sso() {
entry:
  ; reference of 100 bytes with a pattern that includes bytes > 127
  %ref = call ptr @malloc(i64 100)
  br label %fill
fill:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill ]
  %v = mul i64 %i, 37
  %v2 = add i64 %v, 129
  %b = trunc i64 %v2 to i8
  %p = getelementptr inbounds nuw i8, ptr %ref, i64 %i
  store i8 %b, ptr %p, align 1
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 100
  br i1 %more, label %fill, label %run

run:
  ; boundary sweep: 0 and 21 and 22 inline; 23 and 100 heap
  call void @check_sso(ptr %ref, i64 0,   i1 false)
  call void @check_sso(ptr %ref, i64 21,  i1 false)
  call void @check_sso(ptr %ref, i64 22,  i1 false)
  call void @check_sso(ptr %ref, i64 23,  i1 true)
  call void @check_sso(ptr %ref, i64 100, i1 true)
  call void @free(ptr %ref)

  ; explicit multibyte UTF-8 round-trip (inline): euro sign, 3 raw bytes
  %s = alloca [24 x i8], align 8
  %rc = call i32 @universe_sso_create(ptr %s, ptr @d.euro, i64 3)
  %d = call ptr @universe_sso_data(ptr %s)
  %b2.p = getelementptr inbounds nuw i8, ptr %d, i64 2
  %b0 = load i8, ptr %d, align 1
  %b2 = load i8, ptr %b2.p, align 1
  %ok0 = icmp eq i8 %b0, -30   ; 0xE2
  %ok2 = icmp eq i8 %b2, -84   ; 0xAC
  %uok = and i1 %ok0, %ok2
  call void @ut_check(i1 %uok, ptr @s.utf8)
  call void @universe_sso_free(ptr %s)
  ret void
}

define internal void @test_sso_errors() {
entry:
  ; null out -> NULL_PTR
  %e1 = call i32 @universe_sso_create(ptr null, ptr @d.abc, i64 3)
  %e1.ok = icmp eq i32 %e1, 1
  call void @ut_check(i1 %e1.ok, ptr @s.ssonull)
  ; null src with len>0 -> NULL_PTR
  %s = alloca [24 x i8], align 8
  %e2 = call i32 @universe_sso_create(ptr %s, ptr null, i64 4)
  %e2.ok = icmp eq i32 %e2, 1
  call void @ut_check(i1 %e2.ok, ptr @s.ssosrc)
  ret void
}

define internal void @bench() {
entry:
  %si = alloca [24 x i8], align 8
  %sh = alloca [24 x i8], align 8
  ; SHORT (inline) create+read: 17 reps of a 1,000,000-op batch; discard rep 0
  ; (warm-up), report over the remaining 16. ops/rep = 1000000 (ns per create).
  br label %sh.rep
sh.rep:
  %srep = phi i64 [ 0, %entry ], [ %srep.n, %sh.next ]
  %t0 = call double @ut_now_sec()
  br label %iloop
iloop:
  %i = phi i64 [ 0, %sh.rep ], [ %i.n, %iloop ]
  %acc = phi i64 [ 0, %sh.rep ], [ %acc.n, %iloop ]
  %rci = call i32 @universe_sso_create(ptr %si, ptr @d.hello, i64 5)
  %di = call ptr @universe_sso_data(ptr %si)
  %bi = load i8, ptr %di, align 1
  %bi64 = zext i8 %bi to i64
  %acc.n = add i64 %acc, %bi64
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %iloop, label %sh.rep.done
sh.rep.done:
  %t1 = call double @ut_now_sec()
  ; keep %acc live so the inline loop is not elided
  store volatile i64 %acc.n, ptr @sso.sink, align 8
  %sel = fsub double %t1, %t0
  %skeep = icmp ugt i64 %srep, 0
  br i1 %skeep, label %sh.store, label %sh.next
sh.store:
  %sidx = sub i64 %srep, 1
  %ssp = getelementptr inbounds [16 x double], ptr @sso.shsamp, i64 0, i64 %sidx
  store double %sel, ptr %ssp, align 8
  br label %sh.next
sh.next:
  %srep.n = add nuw i64 %srep, 1
  %smore = icmp ult i64 %srep.n, 17
  br i1 %smore, label %sh.rep, label %sh.report
sh.report:
  call void @ut_report_dist(ptr @sso.shsamp, i64 16, i64 1000000, ptr @lbl.ssosh)
  ; HEAP create+read+free using a 40-byte source: 17 reps of a 1,000,000-op batch.
  %src = call ptr @malloc(i64 40)
  br label %hp.rep
hp.rep:
  %hrep = phi i64 [ 0, %sh.report ], [ %hrep.n, %hp.next ]
  %ht0 = call double @ut_now_sec()
  br label %hloop
hloop:
  %j = phi i64 [ 0, %hp.rep ], [ %j.n, %hloop ]
  %rcj = call i32 @universe_sso_create(ptr %sh, ptr %src, i64 40)
  %dj = call ptr @universe_sso_data(ptr %sh)
  %bj = load volatile i8, ptr %dj, align 1
  call void @universe_sso_free(ptr %sh)
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000000
  br i1 %more2, label %hloop, label %hp.rep.done
hp.rep.done:
  %ht1 = call double @ut_now_sec()
  %hel = fsub double %ht1, %ht0
  %hkeep = icmp ugt i64 %hrep, 0
  br i1 %hkeep, label %hp.store, label %hp.next
hp.store:
  %hidx = sub i64 %hrep, 1
  %hsp = getelementptr inbounds [16 x double], ptr @sso.hpsamp, i64 0, i64 %hidx
  store double %hel, ptr %hsp, align 8
  br label %hp.next
hp.next:
  %hrep.n = add nuw i64 %hrep, 1
  %hmore = icmp ult i64 %hrep.n, 17
  br i1 %hmore, label %hp.rep, label %hp.report
hp.report:
  call void @ut_report_dist(ptr @sso.hpsamp, i64 16, i64 1000000, ptr @lbl.ssohp)
  call void @free(ptr %src)
  ret void
}

; ---------------------------------------------------------------------------
; SIMD-vs-scalar oracle. string_eq/compare/index_of_byte/count_byte route
; through the 128-bit vector kernels in src/simd/scan.ll; these in-test scalar
; references are the oracle. Fixed-seed random buffers at every edge length
; (0/1/15/16/17 straddle the 16-byte vector stride, plus 64/255/1024) must
; produce bit-identical results from the vector path and the scalar reference.
; ---------------------------------------------------------------------------

define internal i32 @sgn(i32 %x) {
entry:
  %pos = icmp sgt i32 %x, 0
  %neg = icmp slt i32 %x, 0
  %p = zext i1 %pos to i32
  %n = zext i1 %neg to i32
  %r = sub i32 %p, %n
  ret i32 %r
}

define internal i64 @ref_find(ptr %p, i64 %n, i8 %c) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %nf, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %eq = icmp eq i8 %b, %c
  br i1 %eq, label %found, label %cont
found:
  ret i64 %i
cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %nf
nf:
  ret i64 -1
}

define internal i64 @ref_count(ptr %p, i64 %n, i8 %c) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  %pp = getelementptr inbounds nuw i8, ptr %p, i64 %i
  %b = load i8, ptr %pp, align 1
  %eq = icmp eq i8 %b, %c
  %inc = zext i1 %eq to i64
  %acc.n = add nuw i64 %acc, %inc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  %r = phi i64 [ 0, %entry ], [ %acc.n, %loop ]
  ret i64 %r
}

define internal i1 @ref_eq(ptr %a, i64 %an, ptr %b, i64 %bn) {
entry:
  %leq = icmp eq i64 %an, %bn
  br i1 %leq, label %scan, label %no
scan:
  %z = icmp eq i64 %an, 0
  br i1 %z, label %yes, label %loop
loop:
  %i = phi i64 [ 0, %scan ], [ %i.n, %cont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %av = load i8, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bv = load i8, ptr %bp, align 1
  %ne = icmp ne i8 %av, %bv
  br i1 %ne, label %no, label %cont
cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %an
  br i1 %more, label %loop, label %yes
yes:
  ret i1 true
no:
  ret i1 false
}

define internal i32 @ref_cmp(ptr %a, i64 %an, ptr %b, i64 %bn) {
entry:
  %min = call i64 @llvm.umin.i64(i64 %an, i64 %bn)
  %z = icmp eq i64 %min, 0
  br i1 %z, label %bylen, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %cont ]
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %i
  %ab = load i8, ptr %ap, align 1
  %au = zext i8 %ab to i32
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %i
  %bb = load i8, ptr %bp, align 1
  %bu = zext i8 %bb to i32
  %ne = icmp ne i32 %au, %bu
  br i1 %ne, label %diff, label %cont
diff:
  %d = sub nsw i32 %au, %bu
  ret i32 %d
cont:
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %min
  br i1 %more, label %loop, label %bylen
bylen:
  %lt = icmp ult i64 %an, %bn
  %gt = icmp ugt i64 %an, %bn
  %gti = zext i1 %gt to i32
  %lti = zext i1 %lt to i32
  %r = sub nsw i32 %gti, %lti
  ret i32 %r
}

; eq + compare violations for one (a,an,b,bn) pair (0 if both agree).
define internal i64 @chk_ec(ptr %a, i64 %an, ptr %b, i64 %bn) {
entry:
  %ge = call i1 @universe_string_eq(ptr %a, i64 %an, ptr %b, i64 %bn)
  %re = call i1 @ref_eq(ptr %a, i64 %an, ptr %b, i64 %bn)
  %be = xor i1 %ge, %re
  %gc = call i32 @universe_string_compare(ptr %a, i64 %an, ptr %b, i64 %bn)
  %rc = call i32 @ref_cmp(ptr %a, i64 %an, ptr %b, i64 %bn)
  %gs = call i32 @sgn(i32 %gc)
  %rs = call i32 @sgn(i32 %rc)
  %bc = icmp ne i32 %gs, %rs
  %o = or i1 %be, %bc
  %z = zext i1 %o to i64
  ret i64 %z
}

; find + count violations for one target byte over (p,n) (0 if both agree).
define internal i64 @chk_ft(ptr %p, i64 %n, i8 %c) {
entry:
  %gf = call i64 @universe_string_index_of_byte(ptr %p, i64 %n, i8 %c)
  %rf = call i64 @ref_find(ptr %p, i64 %n, i8 %c)
  %bf = icmp ne i64 %gf, %rf
  %gc = call i64 @universe_string_count_byte(ptr %p, i64 %n, i8 %c)
  %rc = call i64 @ref_count(ptr %p, i64 %n, i8 %c)
  %bc = icmp ne i64 %gc, %rc
  %o = or i1 %bf, %bc
  %z = zext i1 %o to i64
  ret i64 %z
}

define internal void @test_simd_oracle() {
entry:
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8
  %a = call ptr @malloc(i64 1024)
  %b = call ptr @malloc(i64 1024)
  br label %L.head

L.head:
  %li = phi i64 [ 0, %entry ], [ %li.n, %L.cont ]
  %viol = phi i64 [ 0, %entry ], [ %viol.n, %L.cont ]
  %ldone = icmp uge i64 %li, 8
  br i1 %ldone, label %L.done, label %L.body

L.body:
  %lp = getelementptr inbounds [8 x i64], ptr @o.lens, i64 0, i64 %li
  %L = load i64, ptr %lp, align 8
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %L.body ], [ %fi.n, %fill.body ]
  %fdone = icmp uge i64 %fi, %L
  br i1 %fdone, label %chk, label %fill.body

fill.body:
  %rnd = call i64 @ut_rand(ptr %seed)
  %rm = and i64 %rnd, 3
  %rb = trunc i64 %rm to i8
  %ap = getelementptr inbounds nuw i8, ptr %a, i64 %fi
  store i8 %rb, ptr %ap, align 1
  %bp = getelementptr inbounds nuw i8, ptr %b, i64 %fi
  store i8 %rb, ptr %bp, align 1
  %fi.n = add nuw i64 %fi, 1
  br label %fill.head

chk:
  ; equal-length, equal-content pass
  %e0 = call i64 @chk_ec(ptr %a, i64 %L, ptr %b, i64 %L)
  %t0 = call i64 @chk_ft(ptr %a, i64 %L, i8 0)
  %t1 = call i64 @chk_ft(ptr %a, i64 %L, i8 1)
  %t2 = call i64 @chk_ft(ptr %a, i64 %L, i8 2)
  %t3 = call i64 @chk_ft(ptr %a, i64 %L, i8 3)
  %t7 = call i64 @chk_ft(ptr %a, i64 %L, i8 7)
  %s0 = add i64 %e0, %t0
  %s1 = add i64 %s0, %t1
  %s2 = add i64 %s1, %t2
  %s3 = add i64 %s2, %t3
  %v0 = add i64 %s3, %t7
  %pos = icmp ugt i64 %L, 0
  br i1 %pos, label %mut, label %after

mut:
  ; unequal length: compare (a,L) with (b,L-1) — exercises the length tiebreak
  %Lm1 = sub i64 %L, 1
  %ed = call i64 @chk_ec(ptr %a, i64 %L, ptr %b, i64 %Lm1)
  ; mutate one byte of b, then equal-length differing content
  %mid = lshr i64 %L, 1
  %mp = getelementptr inbounds nuw i8, ptr %b, i64 %mid
  %ob = load i8, ptr %mp, align 1
  %fb = xor i8 %ob, -128
  store i8 %fb, ptr %mp, align 1
  %em = call i64 @chk_ec(ptr %a, i64 %L, ptr %b, i64 %L)
  %v1 = add i64 %ed, %em
  br label %after

after:
  %vsel = phi i64 [ 0, %chk ], [ %v1, %mut ]
  %vtot = add i64 %v0, %vsel
  br label %L.cont

L.cont:
  %viol.n = add i64 %viol, %vtot
  %li.n = add nuw i64 %li, 1
  br label %L.head

L.done:
  call void @ut_check_eq(i64 %viol, i64 0, ptr @s.oracle)
  call void @free(ptr %a)
  call void @free(ptr %b)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_view()
  call void @test_hash()
  call void @test_sso()
  call void @test_sso_errors()
  call void @test_simd_oracle()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish
do.bench:
  call void @bench()
  br label %finish
finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
