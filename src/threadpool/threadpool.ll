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

; Fixed worker thread pool with a bounded inline task ring.
;
; DESIGN (vs the typical C implementation):
;   * The C pool MALLOCS A NODE PER TASK and chases a linked list — an
;     allocation plus cache miss on every enqueue/dequeue. Here tasks are
;     {fn, arg} pairs stored INLINE in a power-of-two ring: submit is a
;     store + index increment under the lock; dispatch walks memory
;     sequentially (prefetcher-friendly). Zero per-task allocation.
;   * Bounded ring gives natural backpressure: submit blocks on a full ring
;     (not_full condvar) instead of growing an unbounded list.
;   * One mutex + three condvars (not_empty / not_full / idle). Everything
;     lives in ONE malloc: header, ring, pthread_t array. Opaque pthread
;     blobs sized per docs/conventions.md (mutex 64B, cond 48B).
;   * wait() blocks until queue empty AND in-flight == 0 (true drain).
;   * destroy() drains pending tasks, then joins and frees.
;
; Layout: mutex@0[64] not_empty@64[48] not_full@112[48] idle@160[48]
;         head@208 tail@216 mask@224 cap@232 inflight@240 shutdown@248
;         nthreads@256 ring@264(ptr) threads@272(ptr); data at +320.
;
; API:
;   ptr universe_threadpool_create(i64 nthreads, i64 queue_cap)
;   i32 universe_threadpool_submit(ptr tp, ptr fn, ptr arg)  ; fn: void(ptr)
;   i32 universe_threadpool_wait(ptr tp)
;   i32 universe_threadpool_destroy(ptr tp)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare i32 @pthread_mutex_init(ptr, ptr)
declare i32 @pthread_mutex_lock(ptr)
declare i32 @pthread_mutex_unlock(ptr)
declare i32 @pthread_mutex_destroy(ptr)
declare i32 @pthread_cond_init(ptr, ptr)
declare i32 @pthread_cond_wait(ptr, ptr)
declare i32 @pthread_cond_signal(ptr)
declare i32 @pthread_cond_broadcast(ptr)
declare i32 @pthread_cond_destroy(ptr)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(i64, ptr)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.ctlz.i64(i64, i1 immarg)

define internal ptr @worker(ptr %tp) {
entry:
  %ne.p = getelementptr inbounds nuw i8, ptr %tp, i64 64
  %nf.p = getelementptr inbounds nuw i8, ptr %tp, i64 112
  %idle.p = getelementptr inbounds nuw i8, ptr %tp, i64 160
  %head.p = getelementptr inbounds nuw i8, ptr %tp, i64 208
  %tail.p = getelementptr inbounds nuw i8, ptr %tp, i64 216
  %mask.p = getelementptr inbounds nuw i8, ptr %tp, i64 224
  %inflight.p = getelementptr inbounds nuw i8, ptr %tp, i64 240
  %shut.p = getelementptr inbounds nuw i8, ptr %tp, i64 248
  %ring.p = getelementptr inbounds nuw i8, ptr %tp, i64 264
  %ring = load ptr, ptr %ring.p, align 8
  %mask = load i64, ptr %mask.p, align 8
  %l0 = call i32 @pthread_mutex_lock(ptr %tp)
  br label %sleep.check

sleep.check:                                 ; hold lock here
  %head = load i64, ptr %head.p, align 8
  %tail = load i64, ptr %tail.p, align 8
  %empty = icmp eq i64 %head, %tail
  br i1 %empty, label %maybe.sleep, label %take

maybe.sleep:
  %shut = load i32, ptr %shut.p, align 4
  %quit = icmp ne i32 %shut, 0
  br i1 %quit, label %exit, label %sleep

sleep:
  %w = call i32 @pthread_cond_wait(ptr %ne.p, ptr %tp)
  br label %sleep.check

take:
  %slot = and i64 %head, %mask
  %off = shl nuw i64 %slot, 4                ; 16B per task
  %task.p = getelementptr inbounds nuw i8, ptr %ring, i64 %off
  %fn = load ptr, ptr %task.p, align 8
  %arg.p = getelementptr inbounds nuw i8, ptr %task.p, i64 8
  %arg = load ptr, ptr %arg.p, align 8
  %head.n = add i64 %head, 1
  store i64 %head.n, ptr %head.p, align 8
  %inflight = load i64, ptr %inflight.p, align 8
  %inflight.up = add nuw i64 %inflight, 1
  store i64 %inflight.up, ptr %inflight.p, align 8
  %s1 = call i32 @pthread_cond_signal(ptr %nf.p)
  %u1 = call i32 @pthread_mutex_unlock(ptr %tp)

  call void %fn(ptr %arg) #4                 ; run task outside the lock

  %l1 = call i32 @pthread_mutex_lock(ptr %tp)
  %inflight2 = load i64, ptr %inflight.p, align 8
  %inflight.dn = add i64 %inflight2, -1
  store i64 %inflight.dn, ptr %inflight.p, align 8
  %drained.if = icmp eq i64 %inflight.dn, 0
  %head2 = load i64, ptr %head.p, align 8
  %tail2 = load i64, ptr %tail.p, align 8
  %drained.q = icmp eq i64 %head2, %tail2
  %drained = and i1 %drained.if, %drained.q
  br i1 %drained, label %ping.idle, label %sleep.check

ping.idle:
  %b = call i32 @pthread_cond_broadcast(ptr %idle.p)
  br label %sleep.check

exit:
  %u2 = call i32 @pthread_mutex_unlock(ptr %tp)
  ret ptr null
}

define noalias ptr @universe_threadpool_create(i64 %nthreads, i64 %queue_cap) local_unnamed_addr #1 {
entry:
  %nt.min = call i64 @llvm.umax.i64(i64 %nthreads, i64 1)
  %nt = call i64 @llvm.umin.i64(i64 %nt.min, i64 128)
  ; cap = next_pow2(max(queue_cap, 16)), bounded to 2^24
  %qc.min = call i64 @llvm.umax.i64(i64 %queue_cap, i64 16)
  %qc = call i64 @llvm.umin.i64(i64 %qc.min, i64 16777216)
  %qm1 = add i64 %qc, -1
  %lz = call i64 @llvm.ctlz.i64(i64 %qm1, i1 true)
  %shift = sub nuw nsw i64 64, %lz
  %cap = shl nuw i64 1, %shift
  ; total = 320 + cap*16 + nt*8   (cap<=2^24, nt<=128: never overflows)
  %ring.bytes = shl nuw i64 %cap, 4
  %thr.bytes = shl nuw i64 %nt, 3
  %t0 = add nuw i64 %ring.bytes, %thr.bytes
  %total = add nuw i64 %t0, 320
  %mem = call ptr @malloc(i64 %total)
  %mem.null = icmp eq ptr %mem, null
  br i1 %mem.null, label %fail, label %init, !prof !0

init:
  %mi = call i32 @pthread_mutex_init(ptr %mem, ptr null)
  %ne.p = getelementptr inbounds nuw i8, ptr %mem, i64 64
  %ci1 = call i32 @pthread_cond_init(ptr %ne.p, ptr null)
  %nf.p = getelementptr inbounds nuw i8, ptr %mem, i64 112
  %ci2 = call i32 @pthread_cond_init(ptr %nf.p, ptr null)
  %idle.p = getelementptr inbounds nuw i8, ptr %mem, i64 160
  %ci3 = call i32 @pthread_cond_init(ptr %idle.p, ptr null)
  %head.p = getelementptr inbounds nuw i8, ptr %mem, i64 208
  store i64 0, ptr %head.p, align 8
  %tail.p = getelementptr inbounds nuw i8, ptr %mem, i64 216
  store i64 0, ptr %tail.p, align 8
  %mask.p = getelementptr inbounds nuw i8, ptr %mem, i64 224
  %mask = add i64 %cap, -1
  store i64 %mask, ptr %mask.p, align 8
  %cap.p = getelementptr inbounds nuw i8, ptr %mem, i64 232
  store i64 %cap, ptr %cap.p, align 8
  %inflight.p = getelementptr inbounds nuw i8, ptr %mem, i64 240
  store i64 0, ptr %inflight.p, align 8
  %shut.p = getelementptr inbounds nuw i8, ptr %mem, i64 248
  store i32 0, ptr %shut.p, align 4
  %nt.p = getelementptr inbounds nuw i8, ptr %mem, i64 256
  store i64 %nt, ptr %nt.p, align 8
  %ring = getelementptr inbounds nuw i8, ptr %mem, i64 320
  %ring.p = getelementptr inbounds nuw i8, ptr %mem, i64 264
  store ptr %ring, ptr %ring.p, align 8
  %threads = getelementptr inbounds nuw i8, ptr %ring, i64 %ring.bytes
  %threads.p = getelementptr inbounds nuw i8, ptr %mem, i64 272
  store ptr %threads, ptr %threads.p, align 8
  br label %spawn

spawn:
  %i = phi i64 [ 0, %init ], [ %i.n, %spawn.ok ]
  %done.spawn = icmp uge i64 %i, %nt
  br i1 %done.spawn, label %ready, label %spawn.one

spawn.one:
  %toff = shl nuw i64 %i, 3
  %tid.slot = getelementptr inbounds nuw i8, ptr %threads, i64 %toff
  %rc = call i32 @pthread_create(ptr %tid.slot, ptr null, ptr @worker, ptr %mem)
  %rc.bad = icmp ne i32 %rc, 0
  br i1 %rc.bad, label %abort.spawn, label %spawn.ok, !prof !0

spawn.ok:
  %i.n = add nuw i64 %i, 1
  br label %spawn

abort.spawn:                                 ; join the ones that did start
  %l = call i32 @pthread_mutex_lock(ptr %mem)
  store i32 1, ptr %shut.p, align 4
  %b = call i32 @pthread_cond_broadcast(ptr %ne.p)
  %u = call i32 @pthread_mutex_unlock(ptr %mem)
  br label %abort.join

abort.join:
  %k = phi i64 [ 0, %abort.spawn ], [ %k.n, %abort.join.body ]
  %join.done = icmp uge i64 %k, %i
  br i1 %join.done, label %abort.free, label %abort.join.body

abort.join.body:
  %koff = shl nuw i64 %k, 3
  %kt.slot = getelementptr inbounds nuw i8, ptr %threads, i64 %koff
  %ktid = load i64, ptr %kt.slot, align 8
  %jr = call i32 @pthread_join(i64 %ktid, ptr null)
  %k.n = add nuw i64 %k, 1
  br label %abort.join

abort.free:
  call void @free(ptr nonnull %mem)
  br label %fail

ready:
  ret ptr %mem

fail:
  ret ptr null
}

define i32 @universe_threadpool_submit(ptr %tp, ptr %fn, ptr %arg) local_unnamed_addr #1 {
entry:
  %tp.null = icmp eq ptr %tp, null
  %fn.null = icmp eq ptr %fn, null
  %any.null = or i1 %tp.null, %fn.null
  br i1 %any.null, label %err.null, label %lock, !prof !0

err.null:
  ret i32 1                                  ; UNIVERSE_ERR_NULL_PTR

lock:
  %l = call i32 @pthread_mutex_lock(ptr %tp)
  %head.p = getelementptr inbounds nuw i8, ptr %tp, i64 208
  %tail.p = getelementptr inbounds nuw i8, ptr %tp, i64 216
  %cap.p = getelementptr inbounds nuw i8, ptr %tp, i64 232
  %cap = load i64, ptr %cap.p, align 8
  %nf.p = getelementptr inbounds nuw i8, ptr %tp, i64 112
  %shut.p = getelementptr inbounds nuw i8, ptr %tp, i64 248
  br label %full.check

full.check:
  %shut = load i32, ptr %shut.p, align 4
  %shutting = icmp ne i32 %shut, 0
  br i1 %shutting, label %rejected, label %room.check, !prof !0

room.check:
  %head = load i64, ptr %head.p, align 8
  %tail = load i64, ptr %tail.p, align 8
  %used = sub i64 %tail, %head
  %full = icmp uge i64 %used, %cap
  br i1 %full, label %wait.room, label %enqueue, !prof !0

wait.room:
  %w = call i32 @pthread_cond_wait(ptr %nf.p, ptr %tp)
  br label %full.check

enqueue:
  %mask.p = getelementptr inbounds nuw i8, ptr %tp, i64 224
  %mask = load i64, ptr %mask.p, align 8
  %ring.p = getelementptr inbounds nuw i8, ptr %tp, i64 264
  %ring = load ptr, ptr %ring.p, align 8
  %slot = and i64 %tail, %mask
  %off = shl nuw i64 %slot, 4
  %task.p = getelementptr inbounds nuw i8, ptr %ring, i64 %off
  store ptr %fn, ptr %task.p, align 8
  %argslot.p = getelementptr inbounds nuw i8, ptr %task.p, i64 8
  store ptr %arg, ptr %argslot.p, align 8
  %tail.n = add i64 %tail, 1
  store i64 %tail.n, ptr %tail.p, align 8
  %ne.p = getelementptr inbounds nuw i8, ptr %tp, i64 64
  %s = call i32 @pthread_cond_signal(ptr %ne.p)
  %u = call i32 @pthread_mutex_unlock(ptr %tp)
  ret i32 0

rejected:
  %u2 = call i32 @pthread_mutex_unlock(ptr %tp)
  ret i32 11                                 ; UNIVERSE_ERR_INVALID_STATE
}

define i32 @universe_threadpool_wait(ptr %tp) local_unnamed_addr #1 {
entry:
  %tp.null = icmp eq ptr %tp, null
  br i1 %tp.null, label %err.null, label %lock, !prof !0

err.null:
  ret i32 1

lock:
  %l = call i32 @pthread_mutex_lock(ptr %tp)
  %head.p = getelementptr inbounds nuw i8, ptr %tp, i64 208
  %tail.p = getelementptr inbounds nuw i8, ptr %tp, i64 216
  %inflight.p = getelementptr inbounds nuw i8, ptr %tp, i64 240
  %idle.p = getelementptr inbounds nuw i8, ptr %tp, i64 160
  br label %check

check:
  %head = load i64, ptr %head.p, align 8
  %tail = load i64, ptr %tail.p, align 8
  %empty = icmp eq i64 %head, %tail
  %inflight = load i64, ptr %inflight.p, align 8
  %none = icmp eq i64 %inflight, 0
  %drained = and i1 %empty, %none
  br i1 %drained, label %done, label %block

block:
  %w = call i32 @pthread_cond_wait(ptr %idle.p, ptr %tp)
  br label %check

done:
  %u = call i32 @pthread_mutex_unlock(ptr %tp)
  ret i32 0
}

define i32 @universe_threadpool_destroy(ptr %tp) local_unnamed_addr #1 {
entry:
  %tp.null = icmp eq ptr %tp, null
  br i1 %tp.null, label %err.null, label %drain, !prof !0

err.null:
  ret i32 1

drain:                                       ; finish everything pending first
  %wrc = call i32 @universe_threadpool_wait(ptr nonnull %tp)
  %l = call i32 @pthread_mutex_lock(ptr %tp)
  %shut.p = getelementptr inbounds nuw i8, ptr %tp, i64 248
  store i32 1, ptr %shut.p, align 4
  %ne.p = getelementptr inbounds nuw i8, ptr %tp, i64 64
  %b1 = call i32 @pthread_cond_broadcast(ptr %ne.p)
  %nf.p = getelementptr inbounds nuw i8, ptr %tp, i64 112
  %b2 = call i32 @pthread_cond_broadcast(ptr %nf.p)
  %u = call i32 @pthread_mutex_unlock(ptr %tp)

  %nt.p = getelementptr inbounds nuw i8, ptr %tp, i64 256
  %nt = load i64, ptr %nt.p, align 8
  %threads.p = getelementptr inbounds nuw i8, ptr %tp, i64 272
  %threads = load ptr, ptr %threads.p, align 8
  br label %join

join:
  %i = phi i64 [ 0, %drain ], [ %i.n, %join.body ]
  %join.done = icmp uge i64 %i, %nt
  br i1 %join.done, label %teardown, label %join.body

join.body:
  %toff = shl nuw i64 %i, 3
  %tid.slot = getelementptr inbounds nuw i8, ptr %threads, i64 %toff
  %tid = load i64, ptr %tid.slot, align 8
  %jr = call i32 @pthread_join(i64 %tid, ptr null)
  %i.n = add nuw i64 %i, 1
  br label %join

teardown:
  %md = call i32 @pthread_mutex_destroy(ptr %tp)
  %cd1 = call i32 @pthread_cond_destroy(ptr %ne.p)
  %cd2 = call i32 @pthread_cond_destroy(ptr %nf.p)
  %idle.p = getelementptr inbounds nuw i8, ptr %tp, i64 160
  %cd3 = call i32 @pthread_cond_destroy(ptr %idle.p)
  call void @free(ptr nonnull %tp)
  ret i32 0
}

attributes #1 = { nounwind }
attributes #4 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
