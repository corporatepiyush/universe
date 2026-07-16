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

; LSD radix sort for i32 arrays. O(4N) with 8-bit digits.
;
; DESIGN (vs the typical C implementation):
;   * Sign handled by XORing the top bit during digit extraction of the
;     final pass (negative ints order correctly with zero extra passes).
;   * Skip-pass optimization: histogram first; if a whole pass shares one
;     digit value the scatter is skipped entirely (huge win on small-range
;     data — e.g. all-positive small ints do 2 passes, not 4).
;   * Per-pass [256 x i64] histogram lives on the stack; ONE aux buffer,
;     ping-pong between passes, final memcpy only if the result sits in aux.
;
; API: i32 universe_sort_radix(ptr base /*i32*/, i64 count)

define i32 @universe_sort_radix(ptr %base, i64 %count) local_unnamed_addr #1 {
entry:
  %base.null = icmp eq ptr %base, null
  br i1 %base.null, label %err.null, label %check.trivial, !prof !0

err.null:
  ret i32 1

check.trivial:
  %small = icmp ult i64 %count, 2
  br i1 %small, label %done, label %alloc, !prof !0

alloc:
  %bytes = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %count, i64 4)
  %bytes.v = extractvalue { i64, i1 } %bytes, 0
  %bytes.o = extractvalue { i64, i1 } %bytes, 1
  br i1 %bytes.o, label %err.oom, label %do.alloc, !prof !0

do.alloc:
  %aux = call ptr @malloc(i64 %bytes.v)
  %aux.null = icmp eq ptr %aux, null
  br i1 %aux.null, label %err.oom, label %setup, !prof !0

err.oom:
  ret i32 2

setup:
  %hist = alloca [256 x i64], align 16
  br label %pass

pass:                                        ; shift = 0, 8, 16, 24
  %shift = phi i64 [ 0, %setup ], [ %shift.n, %pass.end ]
  %src = phi ptr [ %base, %setup ], [ %src.n, %pass.end ]
  %dst = phi ptr [ %aux, %setup ], [ %dst.n, %pass.end ]
  %pass.done = icmp ugt i64 %shift, 24
  br i1 %pass.done, label %settle, label %hist.zero

hist.zero:
  call void @llvm.memset.p0.i64(ptr nonnull align 16 %hist, i8 0, i64 2048, i1 false)
  %is.sign.pass = icmp eq i64 %shift, 24
  %digit.bias = select i1 %is.sign.pass, i64 128, i64 0
  br label %tally

tally:
  %i = phi i64 [ 0, %hist.zero ], [ %i.n, %tally ]
  %p = getelementptr inbounds nuw i32, ptr %src, i64 %i
  %v = load i32, ptr %p, align 4
  %v.w = zext i32 %v to i64
  %sh = lshr i64 %v.w, %shift
  %digit.raw = and i64 %sh, 255
  %digit = xor i64 %digit.raw, %digit.bias   ; flip sign bit on last pass
  %hp = getelementptr inbounds nuw [256 x i64], ptr %hist, i64 0, i64 %digit
  %h = load i64, ptr %hp, align 8
  %h.n = add nuw i64 %h, 1
  store i64 %h.n, ptr %hp, align 8
  %i.n = add nuw nsw i64 %i, 1
  %more = icmp ult i64 %i.n, %count
  br i1 %more, label %tally, label %skip.check

skip.check:                                  ; all in one bucket? skip pass
  %probe0 = getelementptr inbounds nuw i32, ptr %src, i64 0
  %v0 = load i32, ptr %probe0, align 4
  %v0.w = zext i32 %v0 to i64
  %sh0 = lshr i64 %v0.w, %shift
  %d0.raw = and i64 %sh0, 255
  %d0 = xor i64 %d0.raw, %digit.bias
  %hp0 = getelementptr inbounds nuw [256 x i64], ptr %hist, i64 0, i64 %d0
  %h0 = load i64, ptr %hp0, align 8
  %uniform = icmp eq i64 %h0, %count
  br i1 %uniform, label %pass.skip, label %prefix.pre

pass.skip:                                   ; nothing moves this pass
  br label %pass.end.skip

prefix.pre:
  br label %prefix

prefix:                                      ; exclusive prefix sums in place
  %d = phi i64 [ 0, %prefix.pre ], [ %d.n, %prefix ]
  %acc = phi i64 [ 0, %prefix.pre ], [ %acc.n, %prefix ]
  %pp = getelementptr inbounds nuw [256 x i64], ptr %hist, i64 0, i64 %d
  %pc = load i64, ptr %pp, align 8
  store i64 %acc, ptr %pp, align 8
  %acc.n = add nuw i64 %acc, %pc
  %d.n = add nuw nsw i64 %d, 1
  %pmore = icmp ult i64 %d.n, 256
  br i1 %pmore, label %prefix, label %scatter.pre

scatter.pre:
  br label %scatter

scatter:                                     ; stable scatter src -> dst
  %j = phi i64 [ 0, %scatter.pre ], [ %j.n, %scatter ]
  %sp = getelementptr inbounds nuw i32, ptr %src, i64 %j
  %sv = load i32, ptr %sp, align 4
  %sv.w = zext i32 %sv to i64
  %ssh = lshr i64 %sv.w, %shift
  %sd.raw = and i64 %ssh, 255
  %sd = xor i64 %sd.raw, %digit.bias
  %posp = getelementptr inbounds nuw [256 x i64], ptr %hist, i64 0, i64 %sd
  %pos = load i64, ptr %posp, align 8
  %pos.n = add nuw i64 %pos, 1
  store i64 %pos.n, ptr %posp, align 8
  %dp = getelementptr inbounds nuw i32, ptr %dst, i64 %pos
  store i32 %sv, ptr %dp, align 4
  %j.n = add nuw nsw i64 %j, 1
  %smore = icmp ult i64 %j.n, %count
  br i1 %smore, label %scatter, label %pass.end.swap

pass.end.swap:                               ; roles flip after a real pass
  br label %pass.end

pass.end.skip:
  br label %pass.end

pass.end:
  %src.n = phi ptr [ %dst, %pass.end.swap ], [ %src, %pass.end.skip ]
  %dst.n = phi ptr [ %src, %pass.end.swap ], [ %dst, %pass.end.skip ]
  %shift.n = add nuw i64 %shift, 8
  br label %pass

settle:
  %in.aux = icmp eq ptr %src, %aux
  br i1 %in.aux, label %copy.back, label %cleanup

copy.back:
  call void @llvm.memcpy.p0.p0.i64(ptr %base, ptr %aux, i64 %bytes.v, i1 false)
  br label %cleanup

cleanup:
  call void @free(ptr nonnull %aux)
  br label %done

done:
  ret i32 0
}

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

attributes #1 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
