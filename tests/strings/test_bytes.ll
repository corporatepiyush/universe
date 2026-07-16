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

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @test_basic()
  call void @test_growth()
  call void @test_ops()
  call void @test_errors()
  call void @test_empty()
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %do.bench, label %finish
do.bench:
  call void @bench()
  br label %finish
finish:
  %rc = call i32 @ut_summary()
  ret i32 %rc
}
