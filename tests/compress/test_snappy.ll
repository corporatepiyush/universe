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

; Tests for the Snappy raw block codec (src/compress/snappy.ll).
;   * KAT decode of hand-derived Snappy blocks (literal, literal-extended,
;     1-byte copy, 2-byte copy non-overlap, offset==1 RLE overlap) == expected.
;   * round-trip encode->decode == identity over fixed-seed random, all-same,
;     and repetitive buffers at edge sizes (0,1,2,60,61,64,~64K).
;   * every reachable negative error (NULL, truncated, FULL, PARSE).
;   * a truncation/mutation fuzz loop that must always return (no OOB — checked
;     under -fsanitize=address,undefined).
;   * --bench: Snappy decode ns/byte.

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)
declare i32 @ut_summary()
declare i32 @memcmp(ptr, ptr, i64)
declare ptr @malloc(i64)
declare void @free(ptr)

declare i64 @universe_compress_snappy_decode(ptr, i64, ptr, i64)
declare i64 @universe_compress_snappy_encode(ptr, i64, ptr, i64)
declare i64 @universe_compress_snappy_bound(i64)

; ---- KAT vectors (compressed block + expected decoded bytes) ----
; K1: varint(5), literal len5 "hello"
@k1c = internal constant [7 x i8]  c"\05\10\68\65\6c\6c\6f"
@k1o = internal constant [5 x i8]  c"\68\65\6c\6c\6f"
; K2: varint(10), literal 'a', copy2 offset=1 len=9  -> 10x 'a' (RLE overlap)
@k2c = internal constant [6 x i8]  c"\0a\00\61\22\01\00"
@k2o = internal constant [10 x i8] c"\61\61\61\61\61\61\61\61\61\61"
; K3: varint(8), literal "abcd", copy2 offset=4 len=4 -> "abcdabcd" (non-overlap)
@k3c = internal constant [9 x i8]  c"\08\0c\61\62\63\64\0e\04\00"
@k3o = internal constant [8 x i8]  c"\61\62\63\64\61\62\63\64"
; K4: varint(8), literal "xyza", copy1 offset=4 len=4 -> "xyzaxyza"
@k4c = internal constant [8 x i8]  c"\08\0c\78\79\7a\61\01\04"
@k4o = internal constant [8 x i8]  c"\78\79\7a\61\78\79\7a\61"
; K5: varint(61), literal-extended (tag 0xf0, lenbyte 0x3c=60) + 61x 'a'
@k5c = internal constant [64 x i8] c"\3d\f0\3c\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61"
@k5o = internal constant [61 x i8] c"\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61\61"

@m.k1  = private constant [20 x i8] c"KAT1 literal\00\00\00\00\00\00\00\00"
@m.k1c = private constant [20 x i8] c"KAT1 bytes\00\00\00\00\00\00\00\00\00\00"
@m.k2  = private constant [20 x i8] c"KAT2 rle len\00\00\00\00\00\00\00\00"
@m.k2c = private constant [20 x i8] c"KAT2 rle bytes\00\00\00\00\00\00"
@m.k3  = private constant [20 x i8] c"KAT3 copy2 len\00\00\00\00\00\00"
@m.k3c = private constant [20 x i8] c"KAT3 copy2 bytes\00\00\00\00"
@m.k4  = private constant [20 x i8] c"KAT4 copy1 len\00\00\00\00\00\00"
@m.k4c = private constant [20 x i8] c"KAT4 copy1 bytes\00\00\00\00"
@m.k5  = private constant [20 x i8] c"KAT5 litext len\00\00\00\00\00"
@m.k5c = private constant [20 x i8] c"KAT5 litext bytes\00\00\00"

@m.rtr = private constant [24 x i8] c"round-trip random\00\00\00\00\00\00\00"
@m.rts = private constant [24 x i8] c"round-trip same-byte\00\00\00\00"
@m.rtp = private constant [24 x i8] c"round-trip repeat\00\00\00\00\00\00\00"
@m.null  = private constant [24 x i8] c"null -> -1\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
@m.trnc0 = private constant [24 x i8] c"empty -> trunc\00\00\00\00\00\00\00\00\00\00"
@m.trnc1 = private constant [24 x i8] c"trunc literal -> neg\00\00\00\00"
@m.full  = private constant [24 x i8] c"small cap -> FULL\00\00\00\00\00\00\00"
@m.badof = private constant [24 x i8] c"zero offset -> parse\00\00\00\00"
@m.over  = private constant [24 x i8] c"over-declared -> parse\00\00"
@m.fuzz  = private constant [24 x i8] c"fuzz returned always\00\00\00\00"

@sn.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.sn  = private unnamed_addr constant [24 x i8] c"snappy decode 61B x1   \00"

; ---- round-trip helper: fill `in` (mode 0=rand,1=same,2=repeat), encode into
;      `tmp`, decode into `out`, and report identity. Returns 1 on success. ----
define internal i1 @rt(i64 %size, i32 %mode, ptr %seed, ptr %in, ptr %tmp, ptr %out) {
entry:
  br label %fill.head
fill.head:
  %i = phi i64 [ 0, %entry ], [ %i.n, %fill.body ]
  %lt = icmp ult i64 %i, %size
  br i1 %lt, label %fill.body, label %doit
fill.body:
  %rv = call i64 @ut_rand(ptr %seed)
  %rb = trunc i64 %rv to i8
  ; repetitive: 30-of-40 run of a segment byte, else random
  %seg = udiv i64 %i, 40
  %inseg = urem i64 %i, 40
  %isrun = icmp ult i64 %inseg, 30
  %segb = trunc i64 %seg to i8
  %runv = add i8 65, %segb
  %repb = select i1 %isrun, i8 %runv, i8 %rb
  %ism1 = icmp eq i32 %mode, 1
  %ism2 = icmp eq i32 %mode, 2
  %b0 = select i1 %ism2, i8 %repb, i8 %rb
  %b = select i1 %ism1, i8 65, i8 %b0
  %pp = getelementptr inbounds i8, ptr %in, i64 %i
  store i8 %b, ptr %pp, align 1
  %i.n = add i64 %i, 1
  br label %fill.head
doit:
  %bound = call i64 @universe_compress_snappy_bound(i64 %size)
  %enc = call i64 @universe_compress_snappy_encode(ptr %tmp, i64 %bound, ptr %in, i64 %size)
  %encok = icmp sgt i64 %enc, -1
  br i1 %encok, label %dec, label %bad
dec:
  %d = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr %tmp, i64 %enc)
  %dok = icmp eq i64 %d, %size
  br i1 %dok, label %cmp, label %bad
cmp:
  %c = call i32 @memcmp(ptr %out, ptr %in, i64 %size)
  %cok = icmp eq i32 %c, 0
  ret i1 %cok
bad:
  ret i1 false
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %out = call ptr @malloc(i64 200000)
  %tmp = call ptr @malloc(i64 200000)
  %in  = call ptr @malloc(i64 200000)
  %seed = alloca i64, align 8
  store i64 88172645463325252, ptr %seed, align 8

  ; ---- KAT1 literal ----
  %d1 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k1c, i64 7)
  call void @ut_check_eq(i64 %d1, i64 5, ptr @m.k1)
  %c1 = call i32 @memcmp(ptr %out, ptr @k1o, i64 5)
  %c1ok = icmp eq i32 %c1, 0
  call void @ut_check(i1 %c1ok, ptr @m.k1c)

  ; ---- KAT2 RLE overlap ----
  %d2 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k2c, i64 6)
  call void @ut_check_eq(i64 %d2, i64 10, ptr @m.k2)
  %c2 = call i32 @memcmp(ptr %out, ptr @k2o, i64 10)
  %c2ok = icmp eq i32 %c2, 0
  call void @ut_check(i1 %c2ok, ptr @m.k2c)

  ; ---- KAT3 copy2 non-overlap ----
  %d3 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k3c, i64 9)
  call void @ut_check_eq(i64 %d3, i64 8, ptr @m.k3)
  %c3 = call i32 @memcmp(ptr %out, ptr @k3o, i64 8)
  %c3ok = icmp eq i32 %c3, 0
  call void @ut_check(i1 %c3ok, ptr @m.k3c)

  ; ---- KAT4 copy1 ----
  %d4 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k4c, i64 8)
  call void @ut_check_eq(i64 %d4, i64 8, ptr @m.k4)
  %c4 = call i32 @memcmp(ptr %out, ptr @k4o, i64 8)
  %c4ok = icmp eq i32 %c4, 0
  call void @ut_check(i1 %c4ok, ptr @m.k4c)

  ; ---- KAT5 literal-extended ----
  %d5 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k5c, i64 64)
  call void @ut_check_eq(i64 %d5, i64 61, ptr @m.k5)
  %c5 = call i32 @memcmp(ptr %out, ptr @k5o, i64 61)
  %c5ok = icmp eq i32 %c5, 0
  call void @ut_check(i1 %c5ok, ptr @m.k5c)

  ; ---- round-trip over edge sizes, three fill modes ----
  ; sizes: 0,1,2,60,61,64,4096,65536
  br label %rt.rand
rt.rand:
  %r0 = call i1 @rt(i64 0,     i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r1 = call i1 @rt(i64 1,     i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r2 = call i1 @rt(i64 2,     i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r3 = call i1 @rt(i64 60,    i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r4 = call i1 @rt(i64 61,    i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r5 = call i1 @rt(i64 64,    i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r6 = call i1 @rt(i64 4096,  i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %r7 = call i1 @rt(i64 65536, i32 0, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %ra = and i1 %r0, %r1
  %rb1 = and i1 %ra, %r2
  %rc1 = and i1 %rb1, %r3
  %rd1 = and i1 %rc1, %r4
  %re1 = and i1 %rd1, %r5
  %rf1 = and i1 %re1, %r6
  %rg1 = and i1 %rf1, %r7
  call void @ut_check(i1 %rg1, ptr @m.rtr)

  ; same-byte (RLE stress)
  %s3 = call i1 @rt(i64 61,    i32 1, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %s5 = call i1 @rt(i64 4096,  i32 1, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %s7 = call i1 @rt(i64 65536, i32 1, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %sa = and i1 %s3, %s5
  %sb = and i1 %sa, %s7
  call void @ut_check(i1 %sb, ptr @m.rts)

  ; repetitive (many matches + overlap)
  %p3 = call i1 @rt(i64 64,    i32 2, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %p5 = call i1 @rt(i64 8000,  i32 2, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %p7 = call i1 @rt(i64 65536, i32 2, ptr %seed, ptr %in, ptr %tmp, ptr %out)
  %pa = and i1 %p3, %p5
  %pb = and i1 %pa, %p7
  call void @ut_check(i1 %pb, ptr @m.rtp)

  ; ---- negative errors ----
  ; NULL dst -> -1
  %e1 = call i64 @universe_compress_snappy_decode(ptr null, i64 100, ptr @k1c, i64 7)
  %e1ok = icmp eq i64 %e1, -1
  call void @ut_check(i1 %e1ok, ptr @m.null)

  ; empty input -> truncated (-15)
  %e2 = call i64 @universe_compress_snappy_decode(ptr %out, i64 100, ptr @k1c, i64 0)
  %e2ok = icmp eq i64 %e2, -15
  call void @ut_check(i1 %e2ok, ptr @m.trnc0)

  ; truncated literal: K1 with only 4 bytes present (claims 5 literals) -> neg
  %e3 = call i64 @universe_compress_snappy_decode(ptr %out, i64 100, ptr @k1c, i64 4)
  %e3neg = icmp slt i64 %e3, 0
  call void @ut_check(i1 %e3neg, ptr @m.trnc1)

  ; declared length exceeds cap -> FULL (-6); K2 declares 10, cap 5
  %e4 = call i64 @universe_compress_snappy_decode(ptr %out, i64 5, ptr @k2c, i64 6)
  %e4ok = icmp eq i64 %e4, -6
  call void @ut_check(i1 %e4ok, ptr @m.full)

  ; crafted zero-offset copy -> PARSE (-13): varint(10), lit 'a', copy2 off=0 len=9
  %bad = alloca [6 x i8], align 1
  store i8 10,  ptr %bad, align 1
  %bp1 = getelementptr inbounds i8, ptr %bad, i64 1
  store i8 0,   ptr %bp1, align 1
  %bp2 = getelementptr inbounds i8, ptr %bad, i64 2
  store i8 97,  ptr %bp2, align 1
  %bp3 = getelementptr inbounds i8, ptr %bad, i64 3
  store i8 34,  ptr %bp3, align 1
  %bp4 = getelementptr inbounds i8, ptr %bad, i64 4
  store i8 0,   ptr %bp4, align 1
  %bp5 = getelementptr inbounds i8, ptr %bad, i64 5
  store i8 0,   ptr %bp5, align 1
  %e5 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr %bad, i64 6)
  %e5ok = icmp eq i64 %e5, -13
  call void @ut_check(i1 %e5ok, ptr @m.badof)

  ; over-declared literal: varint(2) but a literal of length 5 -> PARSE (-13)
  %ovr = alloca [7 x i8], align 1
  store i8 2,   ptr %ovr, align 1
  %op1 = getelementptr inbounds i8, ptr %ovr, i64 1
  store i8 16,  ptr %op1, align 1        ; literal len5 tag (4<<2)
  %op2 = getelementptr inbounds i8, ptr %ovr, i64 2
  store i8 97,  ptr %op2, align 1
  %op3 = getelementptr inbounds i8, ptr %ovr, i64 3
  store i8 97,  ptr %op3, align 1
  %op4 = getelementptr inbounds i8, ptr %ovr, i64 4
  store i8 97,  ptr %op4, align 1
  %op5 = getelementptr inbounds i8, ptr %ovr, i64 5
  store i8 97,  ptr %op5, align 1
  %op6 = getelementptr inbounds i8, ptr %ovr, i64 6
  store i8 97,  ptr %op6, align 1
  %e6 = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr %ovr, i64 7)
  %e6ok = icmp eq i64 %e6, -13
  call void @ut_check(i1 %e6ok, ptr @m.over)

  ; ---- fuzz: mutate/truncate K5 in every prefix length + random tag byte;
  ;      decode must always RETURN (no OOB, checked under ASan) ----
  %fbuf = call ptr @malloc(i64 128)
  br label %fz.head
fz.head:
  %fi = phi i64 [ 0, %rt.rand ], [ %fi.n, %fz.body.n ]
  %flt = icmp ult i64 %fi, 4000
  br i1 %flt, label %fz.body, label %fz.done
fz.body:
  ; copy first `plen` bytes of K5 into fbuf, flip one random byte
  %plen0 = call i64 @ut_rand(ptr %seed)
  %plen = urem i64 %plen0, 65
  br label %fzc.head
fzc.head:
  %fj = phi i64 [ 0, %fz.body ], [ %fj.n, %fzc.body ]
  %fjlt = icmp ult i64 %fj, %plen
  br i1 %fjlt, label %fzc.body, label %fzc.done
fzc.body:
  %sp = getelementptr inbounds i8, ptr @k5c, i64 %fj
  %sv = load i8, ptr %sp, align 1
  %dp = getelementptr inbounds i8, ptr %fbuf, i64 %fj
  store i8 %sv, ptr %dp, align 1
  %fj.n = add i64 %fj, 1
  br label %fzc.head
fzc.done:
  ; mutate one byte (if any)
  %has = icmp ugt i64 %plen, 0
  br i1 %has, label %fz.mut, label %fz.dec
fz.mut:
  %mr = call i64 @ut_rand(ptr %seed)
  %mi = urem i64 %mr, %plen
  %mvr = call i64 @ut_rand(ptr %seed)
  %mv = trunc i64 %mvr to i8
  %mp = getelementptr inbounds i8, ptr %fbuf, i64 %mi
  store i8 %mv, ptr %mp, align 1
  br label %fz.dec
fz.dec:
  %fd = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr %fbuf, i64 %plen)
  br label %fz.body.n
fz.body.n:
  %fi.n = add i64 %fi, 1
  br label %fz.head
fz.done:
  call void @ut_check(i1 true, ptr @m.fuzz)   ; reaching here == every decode returned
  call void @free(ptr %fbuf)

  ; ---- bench ----
  %wb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wb, label %bench, label %fin

bench:
  br label %sn.rep
sn.rep:
  %rep = phi i64 [ 0, %bench ], [ %rep.n, %sn.next ]
  %t0 = call double @ut_now_sec()
  br label %bl.head
bl.head:
  %bi = phi i64 [ 0, %sn.rep ], [ %bi.n, %bl.body ]
  %blt = icmp ult i64 %bi, 2000000
  br i1 %blt, label %bl.body, label %sn.rep.done
bl.body:
  %bd = call i64 @universe_compress_snappy_decode(ptr %out, i64 200000, ptr @k5c, i64 64)
  %bi.n = add i64 %bi, 1
  br label %bl.head
sn.rep.done:
  %t1b = call double @ut_now_sec()
  %el = fsub double %t1b, %t0
  %keep = icmp ugt i64 %rep, 0
  br i1 %keep, label %sn.store, label %sn.next
sn.store:
  %idx = sub i64 %rep, 1
  %sp2 = getelementptr inbounds [16 x double], ptr @sn.samp, i64 0, i64 %idx
  store double %el, ptr %sp2, align 8
  br label %sn.next
sn.next:
  %rep.n = add nuw i64 %rep, 1
  %more = icmp ult i64 %rep.n, 17
  br i1 %more, label %sn.rep, label %sn.report
sn.report:
  ; ops/rep = 2000000 * 61 decoded bytes
  call void @ut_report_dist(ptr @sn.samp, i64 16, i64 122000000, ptr @lbl.sn)
  br label %fin

fin:
  call void @free(ptr %out)
  call void @free(ptr %tmp)
  call void @free(ptr %in)
  %r = call i32 @ut_summary()
  ret i32 %r
}
