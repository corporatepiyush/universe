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

; Tests for universe_bytes: create/len/cap/data, append (contents verified via
; memcmp against an in-test reference), growth across 64K appends, append_byte,
; reserve, clear, truncate, every error code, empty buffer, and a --bench of
; append_byte vs a naive per-byte store loop.

declare ptr @universe_bytes_create(i64)
declare i64 @universe_bytes_len(ptr)
declare i64 @universe_bytes_cap(ptr)
declare ptr @universe_bytes_data(ptr)
declare i32 @universe_bytes_append(ptr, ptr, i64)
declare i32 @universe_bytes_append_byte(ptr, i8)
declare i32 @universe_bytes_reserve(ptr, i64)
declare void @universe_bytes_clear(ptr)
declare i32 @universe_bytes_truncate(ptr, i64)
declare void @universe_bytes_destroy(ptr)
declare i64 @universe_bytes_index_of_byte(ptr, i8)
declare i64 @universe_bytes_count_byte(ptr, i8)
declare i1 @universe_bytes_equals(ptr, ptr)
declare i32 @universe_bytes_compare(ptr, ptr)
declare void @universe_bytes_to_lower(ptr)
declare void @universe_bytes_to_upper(ptr)

declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @memcmp(ptr, ptr, i64)
declare i32 @printf(ptr, ...)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()

@m.create   = private unnamed_addr constant [15 x i8] c"create nonnull\00"
@m.len0     = private unnamed_addr constant [12 x i8] c"fresh len 0\00"
@m.cap0     = private unnamed_addr constant [10 x i8] c"fresh cap\00"
@m.apnd     = private unnamed_addr constant [16 x i8] c"append contents\00"
@m.apndlen  = private unnamed_addr constant [11 x i8] c"append len\00"
@m.grow     = private unnamed_addr constant [19 x i8] c"grow bytes correct\00"
@m.growlen  = private unnamed_addr constant [15 x i8] c"grow len 65536\00"
@m.growcap  = private unnamed_addr constant [16 x i8] c"grow cap ok len\00"
@m.byte     = private unnamed_addr constant [16 x i8] c"append_byte val\00"
@m.reserve  = private unnamed_addr constant [18 x i8] c"reserve grows cap\00"
@m.clear    = private unnamed_addr constant [12 x i8] c"clear len 0\00"
@m.clearcap = private unnamed_addr constant [16 x i8] c"clear keeps cap\00"
@m.trunc    = private unnamed_addr constant [13 x i8] c"truncate len\00"
@m.truncbad = private unnamed_addr constant [15 x i8] c"truncate oob 7\00"
@m.nullap   = private unnamed_addr constant [14 x i8] c"null append 1\00"
@m.nullsrc  = private unnamed_addr constant [18 x i8] c"null src append 1\00"
@m.nullres  = private unnamed_addr constant [15 x i8] c"null reserve 1\00"
@m.nullbyte = private unnamed_addr constant [12 x i8] c"null byte 1\00"
@m.nulltr   = private unnamed_addr constant [13 x i8] c"null trunc 1\00"
@m.ovfres   = private unnamed_addr constant [14 x i8] c"reserve ovf 3\00"
@m.ovfap    = private unnamed_addr constant [13 x i8] c"append ovf 3\00"
@m.empty    = private unnamed_addr constant [13 x i8] c"empty buf ok\00"
@o.lens    = private unnamed_addr constant [8 x i64] [i64 0, i64 1, i64 15, i64 16, i64 17, i64 64, i64 255, i64 1024]
@o.find    = private unnamed_addr constant [22 x i8] c"simd find/count (viol\00"
@o.eqcmp   = private unnamed_addr constant [24 x i8] c"simd equals/compare vio\00"
@o.case    = private unnamed_addr constant [24 x i8] c"simd to_lower/upper vio\00"
@bytes.appsamp = internal global [16 x double] zeroinitializer, align 8
@bytes.navsamp = internal global [16 x double] zeroinitializer, align 8
@lbl.bytesapp = private unnamed_addr constant [17 x i8] c"bytes append 1M \00"
@lbl.bytesnav = private unnamed_addr constant [17 x i8] c"naive store 1M  \00"

; Fill a buffer with a deterministic byte pattern.
define internal void @fill_pattern(ptr %buf, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %v = mul i64 %i, 31
  %v2 = add i64 %v, 7
  %b = trunc i64 %v2 to i8
  %p = getelementptr inbounds nuw i8, ptr %buf, i64 %i
  store i8 %b, ptr %p, align 1
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

define internal void @test_basic() {
entry:
  %b = call ptr @universe_bytes_create(i64 8)
  %ok = icmp ne ptr %b, null
  call void @ut_check(i1 %ok, ptr @m.create)
  %l0 = call i64 @universe_bytes_len(ptr %b)
  call void @ut_check_eq(i64 %l0, i64 0, ptr @m.len0)
  %c0 = call i64 @universe_bytes_cap(ptr %b)
  call void @ut_check_eq(i64 %c0, i64 8, ptr @m.cap0)

  ; reference buffer of 5000 bytes
  %ref = call ptr @malloc(i64 5000)
  call void @fill_pattern(ptr %ref, i64 5000)

  ; append in irregular chunks to force several growths
  br label %chloop

chloop:
  %off = phi i64 [ 0, %entry ], [ %off.n, %chcont ]
  %step = phi i64 [ 17, %entry ], [ %step.n2, %chcont ]
  %rem = sub i64 5000, %off
  %take = call i64 @llvm.umin.i64(i64 %step, i64 %rem)
  %src = getelementptr inbounds nuw i8, ptr %ref, i64 %off
  %rc = call i32 @universe_bytes_append(ptr %b, ptr %src, i64 %take)
  %off.n = add i64 %off, %take
  %step.n = add i64 %step, 13
  %step.wrap = and i64 %step.n, 127
  %step.n2 = add i64 %step.wrap, 1
  %atend = icmp ult i64 %off.n, 5000
  br i1 %atend, label %chcont, label %chdone

chcont:
  br label %chloop

chdone:
  %al = call i64 @universe_bytes_len(ptr %b)
  call void @ut_check_eq(i64 %al, i64 5000, ptr @m.apndlen)
  %data = call ptr @universe_bytes_data(ptr %b)
  %cmp = call i32 @memcmp(ptr %data, ptr %ref, i64 5000)
  %cmp.ok = icmp eq i32 %cmp, 0
  call void @ut_check(i1 %cmp.ok, ptr @m.apnd)
  call void @free(ptr %ref)
  call void @universe_bytes_destroy(ptr %b)
  ret void
}

declare i64 @llvm.umin.i64(i64, i64)

; Grow across 64K single-byte appends; verify every byte and cap invariant.
define internal void @test_growth() {
entry:
  %b = call ptr @universe_bytes_create(i64 0)     ; start from null data
  br label %ap

ap:
  %i = phi i64 [ 0, %entry ], [ %i.n, %ap ]
  %v = trunc i64 %i to i8
  %rc = call i32 @universe_bytes_append_byte(ptr %b, i8 %v)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 65536
  br i1 %more, label %ap, label %check

check:
  %len = call i64 @universe_bytes_len(ptr %b)
  call void @ut_check_eq(i64 %len, i64 65536, ptr @m.growlen)
  %cap = call i64 @universe_bytes_cap(ptr %b)
  %cap.ok = icmp uge i64 %cap, 65536
  call void @ut_check(i1 %cap.ok, ptr @m.growcap)
  %data = call ptr @universe_bytes_data(ptr %b)
  br label %vloop

vloop:
  %j = phi i64 [ 0, %check ], [ %j.n, %vloop ]
  %bad = phi i64 [ 0, %check ], [ %bad.n, %vloop ]
  %p = getelementptr inbounds nuw i8, ptr %data, i64 %j
  %got = load i8, ptr %p, align 1
  %want = trunc i64 %j to i8
  %ne = icmp ne i8 %got, %want
  %inc = zext i1 %ne to i64
  %bad.n = add nuw i64 %bad, %inc
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 65536
  br i1 %more2, label %vloop, label %vdone

vdone:
  call void @ut_check_eq(i64 %bad, i64 0, ptr @m.grow)
  call void @universe_bytes_destroy(ptr %b)
  ret void
}

define internal void @test_ops() {
entry:
  %b = call ptr @universe_bytes_create(i64 4)
  ; append_byte then verify data
  %r1 = call i32 @universe_bytes_append_byte(ptr %b, i8 65)   ; 'A'
  %r2 = call i32 @universe_bytes_append_byte(ptr %b, i8 66)   ; 'B'
  %data = call ptr @universe_bytes_data(ptr %b)
  %d0 = load i8, ptr %data, align 1
  %d1.p = getelementptr inbounds nuw i8, ptr %data, i64 1
  %d1 = load i8, ptr %d1.p, align 1
  %ba = icmp eq i8 %d0, 65
  %bb = icmp eq i8 %d1, 66
  %bok = and i1 %ba, %bb
  call void @ut_check(i1 %bok, ptr @m.byte)

  ; reserve grows capacity to >= len+extra
  %rr = call i32 @universe_bytes_reserve(ptr %b, i64 1000)
  %cap = call i64 @universe_bytes_cap(ptr %b)
  %cap.ok = icmp uge i64 %cap, 1002
  call void @ut_check(i1 %cap.ok, ptr @m.reserve)
  %cap.keep = call i64 @universe_bytes_cap(ptr %b)

  ; clear: len 0, cap unchanged
  call void @universe_bytes_clear(ptr %b)
  %lc = call i64 @universe_bytes_len(ptr %b)
  call void @ut_check_eq(i64 %lc, i64 0, ptr @m.clear)
  %cc = call i64 @universe_bytes_cap(ptr %b)
  call void @ut_check_eq(i64 %cc, i64 %cap.keep, ptr @m.clearcap)

  ; truncate valid
  %ra = call i32 @universe_bytes_append_byte(ptr %b, i8 1)
  %rb = call i32 @universe_bytes_append_byte(ptr %b, i8 2)
  %rc2 = call i32 @universe_bytes_append_byte(ptr %b, i8 3)
  %tt = call i32 @universe_bytes_truncate(ptr %b, i64 2)
  %lt = call i64 @universe_bytes_len(ptr %b)
  %tt.ok = icmp eq i32 %tt, 0
  %lt.ok = icmp eq i64 %lt, 2
  %tv = and i1 %tt.ok, %lt.ok
  call void @ut_check(i1 %tv, ptr @m.trunc)

  ; truncate out of bounds -> 7
  %tb = call i32 @universe_bytes_truncate(ptr %b, i64 100)
  %tb.ok = icmp eq i32 %tb, 7
  call void @ut_check(i1 %tb.ok, ptr @m.truncbad)

  call void @universe_bytes_destroy(ptr %b)
  ret void
}

define internal void @test_errors() {
entry:
  ; NULL_PTR on mutators
  %e1 = call i32 @universe_bytes_append(ptr null, ptr null, i64 0)
  %e1.ok = icmp eq i32 %e1, 1
  call void @ut_check(i1 %e1.ok, ptr @m.nullap)

  %e2 = call i32 @universe_bytes_reserve(ptr null, i64 4)
  %e2.ok = icmp eq i32 %e2, 1
  call void @ut_check(i1 %e2.ok, ptr @m.nullres)

  %e3 = call i32 @universe_bytes_append_byte(ptr null, i8 0)
  %e3.ok = icmp eq i32 %e3, 1
  call void @ut_check(i1 %e3.ok, ptr @m.nullbyte)

  %e4 = call i32 @universe_bytes_truncate(ptr null, i64 0)
  %e4.ok = icmp eq i32 %e4, 1
  call void @ut_check(i1 %e4.ok, ptr @m.nulltr)

  ; null src with n>0 -> NULL_PTR
  %b = call ptr @universe_bytes_create(i64 4)
  %e5 = call i32 @universe_bytes_append(ptr %b, ptr null, i64 8)
  %e5.ok = icmp eq i32 %e5, 1
  call void @ut_check(i1 %e5.ok, ptr @m.nullsrc)

  ; SIZE_OVERFLOW: need len > 0 first, then extra = huge
  %r = call i32 @universe_bytes_append_byte(ptr %b, i8 7)
  %ov1 = call i32 @universe_bytes_reserve(ptr %b, i64 -1)
  %ov1.ok = icmp eq i32 %ov1, 3
  call void @ut_check(i1 %ov1.ok, ptr @m.ovfres)

  ; SIZE_OVERFLOW on append: len(1) + n(-1) overflows; src non-null
  %scratch = call ptr @malloc(i64 8)
  %ov2 = call i32 @universe_bytes_append(ptr %b, ptr %scratch, i64 -1)
  %ov2.ok = icmp eq i32 %ov2, 3
  call void @ut_check(i1 %ov2.ok, ptr @m.ovfap)
  call void @free(ptr %scratch)
  call void @universe_bytes_destroy(ptr %b)
  call void @universe_bytes_destroy(ptr null)
  ret void
}

define internal void @test_empty() {
entry:
  ; empty buffer: append 0 bytes with null src is legal, len stays 0
  %b = call ptr @universe_bytes_create(i64 0)
  %r = call i32 @universe_bytes_append(ptr %b, ptr null, i64 0)
  %len = call i64 @universe_bytes_len(ptr %b)
  %r.ok = icmp eq i32 %r, 0
  %len.ok = icmp eq i64 %len, 0
  %ok = and i1 %r.ok, %len.ok
  call void @ut_check(i1 %ok, ptr @m.empty)
  call void @universe_bytes_destroy(ptr %b)
  ret void
}

define internal void @bench() {
entry:
  %b = call ptr @universe_bytes_create(i64 1048576)
  ; APPEND: 17 reps of a 1,000,000-append batch; clear the buffer each rep to
  ; re-init the input, discard rep 0 (warm-up), report over the remaining 16.
  ; ops/rep = 1000000 (ns per append_byte).
  br label %ap.rep
ap.rep:
  %arep = phi i64 [ 0, %entry ], [ %arep.n, %ap.next ]
  call void @universe_bytes_clear(ptr %b)
  %t0 = call double @ut_now_sec()
  br label %aloop
aloop:
  %i = phi i64 [ 0, %ap.rep ], [ %i.n, %aloop ]
  %v = trunc i64 %i to i8
  %rc = call i32 @universe_bytes_append_byte(ptr %b, i8 %v)
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, 1000000
  br i1 %more, label %aloop, label %ap.rep.done
ap.rep.done:
  %t1 = call double @ut_now_sec()
  %ael = fsub double %t1, %t0
  %akeep = icmp ugt i64 %arep, 0
  br i1 %akeep, label %ap.store, label %ap.next
ap.store:
  %aidx = sub i64 %arep, 1
  %asp = getelementptr inbounds [16 x double], ptr @bytes.appsamp, i64 0, i64 %aidx
  store double %ael, ptr %asp, align 8
  br label %ap.next
ap.next:
  %arep.n = add nuw i64 %arep, 1
  %amore = icmp ult i64 %arep.n, 17
  br i1 %amore, label %ap.rep, label %ap.report
ap.report:
  call void @ut_report_dist(ptr @bytes.appsamp, i64 16, i64 1000000, ptr @lbl.bytesapp)
  ; NAIVE baseline: 17 reps of a 1,000,000-store batch into a raw malloc buffer.
  %naive = call ptr @malloc(i64 1048576)
  br label %nv.rep
nv.rep:
  %nrep = phi i64 [ 0, %ap.report ], [ %nrep.n, %nv.next ]
  %nt0 = call double @ut_now_sec()
  br label %nloop
nloop:
  %j = phi i64 [ 0, %nv.rep ], [ %j.n, %nloop ]
  %jv = trunc i64 %j to i8
  %jmask = and i64 %j, 1048575
  %np = getelementptr inbounds nuw i8, ptr %naive, i64 %jmask
  store volatile i8 %jv, ptr %np, align 1
  %j.n = add nuw i64 %j, 1
  %more2 = icmp ult i64 %j.n, 1000000
  br i1 %more2, label %nloop, label %nv.rep.done
nv.rep.done:
  %nt1 = call double @ut_now_sec()
  %nel = fsub double %nt1, %nt0
  %nkeep = icmp ugt i64 %nrep, 0
  br i1 %nkeep, label %nv.store, label %nv.next
nv.store:
  %nidx = sub i64 %nrep, 1
  %nsp = getelementptr inbounds [16 x double], ptr @bytes.navsamp, i64 0, i64 %nidx
  store double %nel, ptr %nsp, align 8
  br label %nv.next
nv.next:
  %nrep.n = add nuw i64 %nrep, 1
  %nmore = icmp ult i64 %nrep.n, 17
  br i1 %nmore, label %nv.rep, label %nv.report
nv.report:
  call void @ut_report_dist(ptr @bytes.navsamp, i64 16, i64 1000000, ptr @lbl.bytesnav)
  call void @free(ptr %naive)
  call void @universe_bytes_destroy(ptr %b)
  ret void
}

; ---------------------------------------------------------------------------
; SIMD-vs-scalar oracle for the buffer scan ops (index_of_byte/count_byte/
; equals/compare/to_lower/to_upper), which delegate to the 128-bit vector
; kernels in src/simd/scan.ll. In-test scalar references are the oracle; fixed-
; seed random buffers at every edge length (0/1/15/16/17/64/255/1024) must
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

define internal void @ref_lower(ptr %dst, ptr %src, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %c = load i8, ptr %sp, align 1
  %sub = sub i8 %c, 65
  %isu = icmp ult i8 %sub, 26
  %d = select i1 %isu, i8 32, i8 0
  %r = add i8 %c, %d
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %r, ptr %dp, align 1
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

define internal void @ref_upper(ptr %dst, ptr %src, i64 %n) {
entry:
  %z = icmp eq i64 %n, 0
  br i1 %z, label %done, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %sp = getelementptr inbounds nuw i8, ptr %src, i64 %i
  %c = load i8, ptr %sp, align 1
  %sub = sub i8 %c, 97
  %isl = icmp ult i8 %sub, 26
  %d = select i1 %isl, i8 32, i8 0
  %r = sub i8 %c, %d
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %i
  store i8 %r, ptr %dp, align 1
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %n
  br i1 %more, label %loop, label %done
done:
  ret void
}

; find + count violations for one target byte (buf op vs scalar over data,n).
define internal i64 @chk_ft_b(ptr %buf, ptr %data, i64 %n, i8 %c) {
entry:
  %gf = call i64 @universe_bytes_index_of_byte(ptr %buf, i8 %c)
  %rf = call i64 @ref_find(ptr %data, i64 %n, i8 %c)
  %bf = icmp ne i64 %gf, %rf
  %gc = call i64 @universe_bytes_count_byte(ptr %buf, i8 %c)
  %rc = call i64 @ref_count(ptr %data, i64 %n, i8 %c)
  %bc = icmp ne i64 %gc, %rc
  %o = or i1 %bf, %bc
  %z = zext i1 %o to i64
  ret i64 %z
}

define internal void @test_simd_oracle() {
entry:
  %seed = alloca i64, align 8
  store i64 1234567891234567, ptr %seed, align 8
  %a = call ptr @universe_bytes_create(i64 1024)
  %b = call ptr @universe_bytes_create(i64 1024)
  %u = call ptr @universe_bytes_create(i64 1024)
  %raw = call ptr @malloc(i64 1024)
  %refb = call ptr @malloc(i64 1024)
  br label %L.head

L.head:
  %li = phi i64 [ 0, %entry ], [ %li.n, %L.cont ]
  %vf = phi i64 [ 0, %entry ], [ %vf.n, %L.cont ]
  %vc = phi i64 [ 0, %entry ], [ %vc.n, %L.cont ]
  %vk = phi i64 [ 0, %entry ], [ %vk.n, %L.cont ]
  %ldone = icmp uge i64 %li, 8
  br i1 %ldone, label %L.done, label %L.body

L.body:
  %lp = getelementptr inbounds [8 x i64], ptr @o.lens, i64 0, i64 %li
  %L = load i64, ptr %lp, align 8
  call void @universe_bytes_clear(ptr %a)
  call void @universe_bytes_clear(ptr %b)
  call void @universe_bytes_clear(ptr %u)
  br label %fill.head

fill.head:
  %fi = phi i64 [ 0, %L.body ], [ %fi.n, %fill.body ]
  %fdone = icmp uge i64 %fi, %L
  br i1 %fdone, label %prep, label %fill.body

fill.body:
  %rnd = call i64 @ut_rand(ptr %seed)
  %rm = and i64 %rnd, 127
  %rb = trunc i64 %rm to i8
  %rp = getelementptr inbounds nuw i8, ptr %raw, i64 %fi
  store i8 %rb, ptr %rp, align 1
  %aa = call i32 @universe_bytes_append_byte(ptr %a, i8 %rb)
  %ab2 = call i32 @universe_bytes_append_byte(ptr %b, i8 %rb)
  %au2 = call i32 @universe_bytes_append_byte(ptr %u, i8 %rb)
  %fi.n = add nuw i64 %fi, 1
  br label %fill.head

prep:
  %da = call ptr @universe_bytes_data(ptr %a)
  %db = call ptr @universe_bytes_data(ptr %b)
  %du = call ptr @universe_bytes_data(ptr %u)
  ; find/count over four targets
  %f0 = call i64 @chk_ft_b(ptr %a, ptr %da, i64 %L, i8 65)
  %f1 = call i64 @chk_ft_b(ptr %a, ptr %da, i64 %L, i8 97)
  %f2 = call i64 @chk_ft_b(ptr %a, ptr %da, i64 %L, i8 0)
  %f3 = call i64 @chk_ft_b(ptr %a, ptr %da, i64 %L, i8 50)
  %fs0 = add i64 %f0, %f1
  %fs1 = add i64 %fs0, %f2
  %fsum = add i64 %fs1, %f3
  ; equals/compare — identical buffers
  %ge = call i1 @universe_bytes_equals(ptr %a, ptr %b)
  %re = call i1 @ref_eq(ptr %da, i64 %L, ptr %db, i64 %L)
  %be = xor i1 %ge, %re
  %gc = call i32 @universe_bytes_compare(ptr %a, ptr %b)
  %rc = call i32 @ref_cmp(ptr %da, i64 %L, ptr %db, i64 %L)
  %gs = call i32 @sgn(i32 %gc)
  %rs = call i32 @sgn(i32 %rc)
  %bsc = icmp ne i32 %gs, %rs
  %o0 = or i1 %be, %bsc
  %vc0 = zext i1 %o0 to i64
  %pos = icmp ugt i64 %L, 0
  br i1 %pos, label %mut, label %case

mut:
  ; equal-length differing content: flip one byte of b
  %mid = lshr i64 %L, 1
  %mp = getelementptr inbounds nuw i8, ptr %db, i64 %mid
  %ob = load i8, ptr %mp, align 1
  %fbb = xor i8 %ob, -128
  store i8 %fbb, ptr %mp, align 1
  %ge2 = call i1 @universe_bytes_equals(ptr %a, ptr %b)
  %re2 = call i1 @ref_eq(ptr %da, i64 %L, ptr %db, i64 %L)
  %be2 = xor i1 %ge2, %re2
  %gc2 = call i32 @universe_bytes_compare(ptr %a, ptr %b)
  %rc2 = call i32 @ref_cmp(ptr %da, i64 %L, ptr %db, i64 %L)
  %gs2 = call i32 @sgn(i32 %gc2)
  %rs2 = call i32 @sgn(i32 %rc2)
  %bsc2 = icmp ne i32 %gs2, %rs2
  %o2 = or i1 %be2, %bsc2
  %z2 = zext i1 %o2 to i64
  ; unequal length: truncate b to L-1
  %Lm1 = sub i64 %L, 1
  %tr = call i32 @universe_bytes_truncate(ptr %b, i64 %Lm1)
  %ge3 = call i1 @universe_bytes_equals(ptr %a, ptr %b)
  %re3 = call i1 @ref_eq(ptr %da, i64 %L, ptr %db, i64 %Lm1)
  %be3 = xor i1 %ge3, %re3
  %gc3 = call i32 @universe_bytes_compare(ptr %a, ptr %b)
  %rc3 = call i32 @ref_cmp(ptr %da, i64 %L, ptr %db, i64 %Lm1)
  %gs3 = call i32 @sgn(i32 %gc3)
  %rs3 = call i32 @sgn(i32 %rc3)
  %bsc3 = icmp ne i32 %gs3, %rs3
  %o3 = or i1 %be3, %bsc3
  %z3 = zext i1 %o3 to i64
  %vc1 = add i64 %z2, %z3
  br label %case

case:
  %vcsel = phi i64 [ 0, %prep ], [ %vc1, %mut ]
  %vcadd = add i64 %vc0, %vcsel
  ; case fold: to_lower(a) vs scalar-lowered raw; to_upper(u) vs scalar-uppered raw
  call void @ref_lower(ptr %refb, ptr %raw, i64 %L)
  call void @universe_bytes_to_lower(ptr %a)
  %ml = call i32 @memcmp(ptr %da, ptr %refb, i64 %L)
  %mlbad = icmp ne i32 %ml, 0
  %mlz = zext i1 %mlbad to i64
  call void @ref_upper(ptr %refb, ptr %raw, i64 %L)
  call void @universe_bytes_to_upper(ptr %u)
  %mu = call i32 @memcmp(ptr %du, ptr %refb, i64 %L)
  %mubad = icmp ne i32 %mu, 0
  %muz = zext i1 %mubad to i64
  %vkadd = add i64 %mlz, %muz
  br label %L.cont

L.cont:
  %vf.n = add i64 %vf, %fsum
  %vc.n = add i64 %vc, %vcadd
  %vk.n = add i64 %vk, %vkadd
  %li.n = add nuw i64 %li, 1
  br label %L.head

L.done:
  call void @ut_check_eq(i64 %vf, i64 0, ptr @o.find)
  call void @ut_check_eq(i64 %vc, i64 0, ptr @o.eqcmp)
  call void @ut_check_eq(i64 %vk, i64 0, ptr @o.case)
  call void @universe_bytes_destroy(ptr %a)
  call void @universe_bytes_destroy(ptr %b)
  call void @universe_bytes_destroy(ptr %u)
  call void @free(ptr %raw)
  call void @free(ptr %refb)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_growth()
  call void @test_ops()
  call void @test_errors()
  call void @test_empty()
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
