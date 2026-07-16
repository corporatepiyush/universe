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

; universe_conc_ebr_* / universe_conc_hp_* / universe_conc_seqlock_* —
; a lock-free MEMORY-RECLAMATION toolkit. Answers "when is it safe to free a
; node that a concurrent lock-free reader might still be dereferencing?" —
; the hard part of every lock-free node-based structure (Treiber stack,
; Michael-Scott queue, concurrent skiplist / ART). Two independent, composable
; schemes plus a read-mostly SeqLock; implemented from first principles.
;
; ============================================================================
; DESIGN
; ----------------------------------------------------------------------------
; THREAD MODEL. Callers pass a small dense thread id `tid` in [0, MAX_THREADS).
; This deliberately avoids thread-local storage (TLS lowering is a portability
; minefield across macOS/Linux/FreeBSD x ISA); a dense id indexes a fixed
; global registry whose per-thread control blocks live on their own cache
; lines. MAX_THREADS = 64.
;
; ----------------------------------------------------------------------------
; (1) EPOCH-BASED RECLAMATION (EBR) — near-zero reader cost.
;   A single global epoch counter (free-running i64; only its value mod 3
;   matters). THREE "bags" of deferred frees per thread, indexed by
;   epoch mod 3. The classic 3-epoch argument:
;     * enter(pin): read the global epoch, PUBLISH it into this thread's
;       local slot, then a seq_cst fence. Between enter and exit the thread is
;       "in a critical section"; any node it can reach is safe to dereference.
;     * retire: append (node, freefn) to the bag of the thread's CURRENT
;       pinned epoch. Binding the bag to the pin (not a fresh global read)
;       closes the window where a delayed retire lands in a bag that is about
;       to be reclaimed.
;     * try_advance: seq_cst fence, then scan every thread's local slot; if
;       every PINNED thread is at the current global epoch, CAS the global
;       epoch forward by one. The winner of the CAS then frees the bag two
;       epochs behind the NEW global (slot (g-1) mod 3 == (g+2) mod 3): the
;       advance condition guarantees no pinned thread is at epoch <= g-1, so
;       nothing reachable references those nodes.
;   Reader cost = one relaxed load + one relaxed store + one seq_cst fence per
;   critical section; no per-access atomics. This is why EBR is the default
;   for read-heavy structures.
;
;   ORDERINGS (each justified; the ONLY seq_cst is the pin/scan StoreLoad):
;     - enter: load global ACQUIRE (synchronize-with the advancer that
;         published this epoch, so our later plain accesses to bag metadata
;         have a happens-before edge), store local MONOTONIC, then FENCE
;         seq_cst. The fence is the Dekker/StoreLoad barrier that makes the
;         pin visible to a concurrent scan before the reader dereferences.
;     - exit: store local 0 RELEASE. Release orders our critical-section reads
;         BEFORE the unpin is observed by the advancer's acquire scan load, so
;         a free can never overtake a read we already issued.
;     - try_advance: load global ACQUIRE, FENCE seq_cst (pairs with enter's
;         fence), scan locals ACQUIRE (pairs with exit's release), CAS global
;         ACQ_REL (acquire prior advancer state, release our bag resets).
;     - retire: load own local MONOTONIC (own slot), fallback load global
;         ACQUIRE when unpinned. Bag metadata is thread-owned; the advancer
;         only touches a QUIESCENT bag (2 epochs behind), so those plain
;         accesses are race-free by construction.
;
; ----------------------------------------------------------------------------
; (2) HAZARD POINTERS (HP) — bounded memory, per-access publish cost.
;   Global array of K=8 single-writer hazard slots per thread (128 B padded so
;   a thread's slots never false-share with a neighbour). A reader PUBLISHES a
;   pointer into a slot, issues a seq_cst fence, then RE-VALIDATES that the
;   source still points at it (caller's loop); once validated the node cannot
;   be freed under it. retire adds to a per-thread retire list; when the list
;   crosses a threshold it SCANS every hazard slot and frees only nodes that
;   no slot protects. Memory is bounded: retained nodes <= total live hazards.
;
;   ORDERINGS:
;     - protect: store slot RELEASE, then FENCE seq_cst (StoreLoad: publish
;         before the caller re-reads the source; symmetric with the scan).
;     - clear: store slot null RELEASE (dropping protection can only DELAY a
;         free, never cause a UAF, so no fence needed).
;     - scan: FENCE seq_cst (pairs with protect), load each slot ACQUIRE
;         (pairs with protect's release store). retire list is thread-owned.
;
; ----------------------------------------------------------------------------
; (3) SEQLOCK — read-mostly small POD, no reader-side writes.
;   A version counter, ODD while a write is in progress. Readers snapshot the
;   counter, read the data, re-read the counter, and retry if it is odd or
;   changed. Writers are serialized by the caller (single writer or external
;   lock). Orderings: begin bumps to odd then FENCE release (data writes stay
;   after the odd publish); end publishes even with a RELEASE store; readers
;   load ACQUIRE and FENCE acquire before the re-read.
;
; ============================================================================
; LAYOUT (byte offsets computed explicitly; identical on every target)
;   @ebr  : [0]=global epoch (atomic i64, own 128 B line);
;           thread t block at 128 + t*256 :
;             +0   local epoch slot (atomic i64; (epoch<<1)|1 pinned, 0 idle)
;             +128 bag0{cnt@0,cap@8,arr@16}  (arr = {ptr node, ptr fn}[])
;             +152 bag1{cnt,cap,arr}
;             +176 bag2{cnt,cap,arr}
;   @hp_slots        : thread t slot k (atomic ptr) at t*128 + k*8, k in [0,8)
;   @hp_retire_lists : thread t {cnt@0,cap@8,arr@16} at t*32
;
; API:
;   void  universe_conc_ebr_register(i64 tid)
;   void  universe_conc_ebr_enter(i64 tid)
;   void  universe_conc_ebr_exit(i64 tid)
;   void  universe_conc_ebr_retire(i64 tid, ptr node, ptr freefn)
;   i32   universe_conc_ebr_try_advance(i64 tid)          ; 1 advanced, 0 not
;   i64   universe_conc_ebr_epoch()                       ; introspection
;   void  universe_conc_hp_protect(i64 tid, i64 k, ptr p)
;   void  universe_conc_hp_clear(i64 tid, i64 k)
;   void  universe_conc_hp_retire(i64 tid, ptr node, ptr freefn)
;   void  universe_conc_hp_collect(i64 tid)               ; force a scan+free
;   void  universe_conc_seqlock_write_begin(ptr s)
;   void  universe_conc_seqlock_write_end(ptr s)
;   i64   universe_conc_seqlock_read_begin(ptr s)
;   i1    universe_conc_seqlock_read_retry(ptr s, i64 prev)

declare ptr @realloc(ptr, i64)
declare void @free(ptr)

; @ebr size = 128 + 64*256 = 16512
@ebr = internal global [16512 x i8] zeroinitializer, align 128
; @hp_slots size = 64*128 = 8192
@hp_slots = internal global [8192 x i8] zeroinitializer, align 128
; @hp_retire_lists size = 64*32 = 2048
@hp_retire_lists = internal global [2048 x i8] zeroinitializer, align 8

; ===========================================================================
; EBR
; ===========================================================================

; register — (re)initialize this thread's local epoch slot to idle. Bags are
; zero from BSS and grown lazily on first retire, so nothing else is needed.
define void @universe_conc_ebr_register(i64 %tid) local_unnamed_addr #1 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %off = shl nuw nsw i64 %tid, 8              ; tid*256
  %base = add nuw nsw i64 %off, 128
  %le.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %base
  store atomic i64 0, ptr %le.p monotonic, align 8
  br label %ret

ret:
  ret void
}

; enter (pin) — publish the current global epoch into our slot, then a
; seq_cst fence so the pin is globally visible before we dereference anything.
define void @universe_conc_ebr_enter(i64 %tid) local_unnamed_addr #1 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %g = load atomic i64, ptr @ebr acquire, align 8      ; global epoch
  %gsh = shl i64 %g, 1
  %pinned = or i64 %gsh, 1                              ; (epoch<<1)|1
  %off = shl nuw nsw i64 %tid, 8
  %base = add nuw nsw i64 %off, 128
  %le.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %base
  store atomic i64 %pinned, ptr %le.p monotonic, align 8
  fence seq_cst
  br label %ret

ret:
  ret void
}

; exit (unpin) — mark idle with a RELEASE store; our critical-section reads
; are ordered before any advancer observes the unpin.
define void @universe_conc_ebr_exit(i64 %tid) local_unnamed_addr #1 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %off = shl nuw nsw i64 %tid, 8
  %base = add nuw nsw i64 %off, 128
  %le.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %base
  store atomic i64 0, ptr %le.p release, align 8
  br label %ret

ret:
  ret void
}

; retire — defer (node, freefn) to the bag of our CURRENT pinned epoch.
define void @universe_conc_ebr_retire(i64 %tid, ptr %node, ptr %freefn) local_unnamed_addr #1 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %off = shl nuw nsw i64 %tid, 8
  %base = add nuw nsw i64 %off, 128
  %le.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %base
  %lv = load atomic i64, ptr %le.p monotonic, align 8
  %pin.bit = and i64 %lv, 1
  %is.pinned = icmp ne i64 %pin.bit, 0
  br i1 %is.pinned, label %use.local, label %use.global

use.local:
  %epoch.l = lshr i64 %lv, 1
  br label %have.epoch

use.global:
  %epoch.g = load atomic i64, ptr @ebr acquire, align 8
  br label %have.epoch

have.epoch:
  %epoch = phi i64 [ %epoch.l, %use.local ], [ %epoch.g, %use.global ]
  %slot = urem i64 %epoch, 3
  %bag.off0 = mul nuw nsw i64 %slot, 32                 ; bag stride 32
  %bag.off = add nuw nsw i64 %bag.off0, 128            ; block-relative bag base
  %bag.abs = add nuw nsw i64 %base, %bag.off
  %cnt.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %bag.abs
  %cap.off = add nuw nsw i64 %bag.abs, 8
  %cap.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %cap.off
  %arr.off = add nuw nsw i64 %bag.abs, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %arr.off
  %ep.off = add nuw nsw i64 %bag.abs, 24
  %ep.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %ep.off
  ; Lazy owner-reclaim: if this slot holds a PRIOR generation (epoch <=
  ; current-3, guaranteed since only we write our bags and we are pinned at
  ; %epoch), its nodes are safe to free NOW (global >= %epoch >= old+3 > old+2,
  ; so no reader is pinned at <= old). Free + reset before appending. This
  ; makes reclamation OWNER-ONLY and self-sequencing => no double free and no
  ; lapping (the pitfall of freeing another thread's bag by epoch).
  %bag.epoch = load i64, ptr %ep.p, align 8
  %same = icmp eq i64 %bag.epoch, %epoch
  br i1 %same, label %append.load, label %recycle

recycle:
  %rcnt = load i64, ptr %cnt.p, align 8
  %rarr = load ptr, ptr %arr.p, align 8
  %rempty = icmp eq i64 %rcnt, 0
  br i1 %rempty, label %recycle.reset, label %rfree

rfree:
  %ri = phi i64 [ 0, %recycle ], [ %ri.n, %rfree ]
  %reoff = shl i64 %ri, 4
  %rent = getelementptr inbounds i8, ptr %rarr, i64 %reoff
  %rnode = load ptr, ptr %rent, align 8
  %rfn.p = getelementptr inbounds nuw i8, ptr %rent, i64 8
  %rfn = load ptr, ptr %rfn.p, align 8
  call void %rfn(ptr %rnode)
  %ri.n = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, %rcnt
  br i1 %rmore, label %rfree, label %recycle.reset

recycle.reset:
  store i64 0, ptr %cnt.p, align 8
  store i64 %epoch, ptr %ep.p, align 8
  br label %append.load

append.load:
  %cnt = load i64, ptr %cnt.p, align 8
  %cap = load i64, ptr %cap.p, align 8
  %arr.pre = load ptr, ptr %arr.p, align 8
  %full = icmp uge i64 %cnt, %cap
  br i1 %full, label %grow, label %write, !prof !0

grow:
  %cap.zero = icmp eq i64 %cap, 0
  %cap.dbl = shl i64 %cap, 1
  %newcap = select i1 %cap.zero, i64 16, i64 %cap.dbl
  %newbytes = shl i64 %newcap, 4                        ; newcap*16
  %new.arr = call ptr @realloc(ptr %arr.pre, i64 %newbytes)
  store i64 %newcap, ptr %cap.p, align 8
  store ptr %new.arr, ptr %arr.p, align 8
  br label %write

write:
  %arr = phi ptr [ %new.arr, %grow ], [ %arr.pre, %append.load ]
  %eoff = shl i64 %cnt, 4                                ; cnt*16
  %ent = getelementptr inbounds i8, ptr %arr, i64 %eoff
  store ptr %node, ptr %ent, align 8
  %fn.p = getelementptr inbounds nuw i8, ptr %ent, i64 8
  store ptr %freefn, ptr %fn.p, align 8
  %cnt.n = add nuw i64 %cnt, 1
  store i64 %cnt.n, ptr %cnt.p, align 8
  br label %ret

ret:
  ret void
}

; try_advance — advance the global epoch if every PINNED thread is already at
; the current epoch. Reclamation itself is NOT done here (that is owner-only,
; folded into retire's slot recycle); this only moves the epoch forward so
; those slots become reclaimable. Returns 1 if it advanced, else 0.
define i32 @universe_conc_ebr_try_advance(i64 %tid) local_unnamed_addr #3 {
entry:
  %g = load atomic i64, ptr @ebr acquire, align 8
  fence seq_cst
  br label %scan

scan:
  %t = phi i64 [ 0, %entry ], [ %t.n, %scan.cont ]
  %toff = shl nuw nsw i64 %t, 8
  %tbase = add nuw nsw i64 %toff, 128
  %tle.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %tbase
  %tv = load atomic i64, ptr %tle.p acquire, align 8
  %tpin = and i64 %tv, 1
  %t.pinned = icmp ne i64 %tpin, 0
  %tepoch = lshr i64 %tv, 1
  %t.stale = icmp ne i64 %tepoch, %g
  %block = and i1 %t.pinned, %t.stale
  br i1 %block, label %no, label %scan.cont

scan.cont:
  %t.n = add nuw nsw i64 %t, 1
  %more = icmp ult i64 %t.n, 64
  br i1 %more, label %scan, label %try.cas

no:
  ret i32 0

try.cas:
  %gn = add i64 %g, 1
  %cx = cmpxchg ptr @ebr, i64 %g, i64 %gn acq_rel acquire
  %ok = extractvalue { i64, i1 } %cx, 1
  %r = zext i1 %ok to i32
  ret i32 %r
}

; collect_all — QUIESCENT-ONLY teardown: with no thread pinned, free every
; pending node in every bag and reset counts (arrays retained; ebr_reset frees
; them). Safe only when all workers have stopped.
define void @universe_conc_ebr_collect_all() local_unnamed_addr #3 {
entry:
  br label %bag

bag:
  %bi = phi i64 [ 0, %entry ], [ %bi.n, %bag.next ]     ; 0..192 (64 threads * 3)
  %tid3 = udiv i64 %bi, 3
  %slot = urem i64 %bi, 3
  %toff = shl nuw nsw i64 %tid3, 8
  %tbase = add nuw nsw i64 %toff, 128
  %soff = mul nuw nsw i64 %slot, 32
  %srel = add nuw nsw i64 %soff, 128
  %babs = add nuw nsw i64 %tbase, %srel
  %cnt.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %babs
  %arr.rel = add nuw nsw i64 %babs, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %arr.rel
  %cnt = load i64, ptr %cnt.p, align 8
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %bag.next, label %freearr

freearr:
  %arr = load ptr, ptr %arr.p, align 8
  br label %fl

fl:
  %i = phi i64 [ 0, %freearr ], [ %i.n, %fl ]
  %eoff = shl i64 %i, 4
  %ent = getelementptr inbounds i8, ptr %arr, i64 %eoff
  %node = load ptr, ptr %ent, align 8
  %fn.p = getelementptr inbounds nuw i8, ptr %ent, i64 8
  %fn = load ptr, ptr %fn.p, align 8
  call void %fn(ptr %node)
  %i.n = add nuw i64 %i, 1
  %fmore = icmp ult i64 %i.n, %cnt
  br i1 %fmore, label %fl, label %doreset

doreset:
  store i64 0, ptr %cnt.p, align 8
  br label %bag.next

bag.next:
  %bi.n = add nuw nsw i64 %bi, 1
  %more = icmp ult i64 %bi.n, 192
  br i1 %more, label %bag, label %ret

ret:
  ret void
}

define i64 @universe_conc_ebr_epoch() local_unnamed_addr #2 {
entry:
  %g = load atomic i64, ptr @ebr acquire, align 8
  ret i64 %g
}

; ===========================================================================
; HAZARD POINTERS
; ===========================================================================

; protect — publish p into slot k, then a seq_cst fence so a concurrent scan
; sees the hazard before the caller re-validates the source.
define void @universe_conc_hp_protect(i64 %tid, i64 %k, ptr %p) local_unnamed_addr #1 {
entry:
  %oob.t = icmp uge i64 %tid, 64
  %oob.k = icmp uge i64 %k, 8
  %oob = or i1 %oob.t, %oob.k
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %toff = shl nuw nsw i64 %tid, 7                       ; tid*128
  %koff = shl nuw nsw i64 %k, 3                         ; k*8
  %off = add nuw nsw i64 %toff, %koff
  %slot.p = getelementptr inbounds nuw i8, ptr @hp_slots, i64 %off
  store atomic ptr %p, ptr %slot.p release, align 8
  fence seq_cst
  br label %ret

ret:
  ret void
}

; clear — drop protection on slot k (release; can only delay a free).
define void @universe_conc_hp_clear(i64 %tid, i64 %k) local_unnamed_addr #1 {
entry:
  %oob.t = icmp uge i64 %tid, 64
  %oob.k = icmp uge i64 %k, 8
  %oob = or i1 %oob.t, %oob.k
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %toff = shl nuw nsw i64 %tid, 7
  %koff = shl nuw nsw i64 %k, 3
  %off = add nuw nsw i64 %toff, %koff
  %slot.p = getelementptr inbounds nuw i8, ptr @hp_slots, i64 %off
  store atomic ptr null, ptr %slot.p release, align 8
  br label %ret

ret:
  ret void
}

; internal: is %node protected by ANY hazard slot? (seq_cst fence issued by
; the caller before the first invocation of the scan loop).
define internal i1 @hp_is_protected(ptr %node) #0 {
entry:
  br label %tloop

tloop:
  %t = phi i64 [ 0, %entry ], [ %t.n, %kdone ]
  %tbase = shl nuw nsw i64 %t, 7                        ; t*128
  br label %kloop

kloop:
  %kk = phi i64 [ 0, %tloop ], [ %kk.n, %kcont ]
  %koff = shl nuw nsw i64 %kk, 3
  %off = add nuw nsw i64 %tbase, %koff
  %slot.p = getelementptr inbounds nuw i8, ptr @hp_slots, i64 %off
  %h = load atomic ptr, ptr %slot.p acquire, align 8
  %hit = icmp eq ptr %h, %node
  br i1 %hit, label %yes, label %kcont

kcont:
  %kk.n = add nuw nsw i64 %kk, 1
  %kmore = icmp ult i64 %kk.n, 8
  br i1 %kmore, label %kloop, label %kdone

kdone:
  %t.n = add nuw nsw i64 %t, 1
  %tmore = icmp ult i64 %t.n, 64
  br i1 %tmore, label %tloop, label %noo

yes:
  ret i1 true

noo:
  ret i1 false
}

; internal scan — free every retired node no hazard slot protects; compact
; the survivors to the front of the retire list.
define internal void @hp_scan(i64 %tid) #3 {
entry:
  %loff = shl nuw nsw i64 %tid, 5                       ; tid*32
  %cnt.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %loff
  %arr.rel = add nuw nsw i64 %loff, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %arr.rel
  %cnt = load i64, ptr %cnt.p, align 8
  %empty = icmp eq i64 %cnt, 0
  br i1 %empty, label %ret, label %setup

setup:
  %arr = load ptr, ptr %arr.p, align 8
  fence seq_cst                                          ; pair with protect
  br label %loop

loop:
  %i = phi i64 [ 0, %setup ], [ %i.n, %cont ]
  %w = phi i64 [ 0, %setup ], [ %w.n, %cont ]
  %eoff = shl i64 %i, 4
  %ent = getelementptr inbounds i8, ptr %arr, i64 %eoff
  %node = load ptr, ptr %ent, align 8
  %fn.p = getelementptr inbounds nuw i8, ptr %ent, i64 8
  %fn = load ptr, ptr %fn.p, align 8
  %prot = call i1 @hp_is_protected(ptr %node)
  br i1 %prot, label %keep, label %drop

keep:
  ; compact survivor to slot w (skip the self-copy when w==i)
  %same = icmp eq i64 %w, %i
  br i1 %same, label %keep.done, label %keep.move

keep.move:
  %woff = shl i64 %w, 4
  %went = getelementptr inbounds i8, ptr %arr, i64 %woff
  store ptr %node, ptr %went, align 8
  %wfn.p = getelementptr inbounds nuw i8, ptr %went, i64 8
  store ptr %fn, ptr %wfn.p, align 8
  br label %keep.done

keep.done:
  %w.keep = add nuw i64 %w, 1
  br label %cont

drop:
  call void %fn(ptr %node)
  br label %cont

cont:
  %w.n = phi i64 [ %w.keep, %keep.done ], [ %w, %drop ]
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %cnt
  br i1 %more, label %loop, label %finish

finish:
  store i64 %w.n, ptr %cnt.p, align 8
  br label %ret

ret:
  ret void
}

; retire — append (node, freefn) to this thread's list; scan when it crosses
; the threshold (64). Threshold > max simultaneous live hazards guarantees
; forward progress and bounded retained memory.
define void @universe_conc_hp_retire(i64 %tid, ptr %node, ptr %freefn) local_unnamed_addr #3 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  %loff = shl nuw nsw i64 %tid, 5
  %cnt.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %loff
  %cap.rel = add nuw nsw i64 %loff, 8
  %cap.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %cap.rel
  %arr.rel = add nuw nsw i64 %loff, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %arr.rel
  %cnt = load i64, ptr %cnt.p, align 8
  %cap = load i64, ptr %cap.p, align 8
  %arr.pre = load ptr, ptr %arr.p, align 8
  %full = icmp uge i64 %cnt, %cap
  br i1 %full, label %grow, label %write, !prof !0

grow:
  %cap.zero = icmp eq i64 %cap, 0
  %cap.dbl = shl i64 %cap, 1
  %newcap = select i1 %cap.zero, i64 16, i64 %cap.dbl
  %newbytes = shl i64 %newcap, 4
  %new.arr = call ptr @realloc(ptr %arr.pre, i64 %newbytes)
  store i64 %newcap, ptr %cap.p, align 8
  store ptr %new.arr, ptr %arr.p, align 8
  br label %write

write:
  %arr = phi ptr [ %new.arr, %grow ], [ %arr.pre, %go ]
  %eoff = shl i64 %cnt, 4
  %ent = getelementptr inbounds i8, ptr %arr, i64 %eoff
  store ptr %node, ptr %ent, align 8
  %fn.p = getelementptr inbounds nuw i8, ptr %ent, i64 8
  store ptr %freefn, ptr %fn.p, align 8
  %cnt.n = add nuw i64 %cnt, 1
  store i64 %cnt.n, ptr %cnt.p, align 8
  %hit = icmp uge i64 %cnt.n, 64
  br i1 %hit, label %doscan, label %ret, !prof !0

doscan:
  call void @hp_scan(i64 %tid)
  br label %ret

ret:
  ret void
}

; collect — force an immediate scan+free (teardown / low-water flush).
define void @universe_conc_hp_collect(i64 %tid) local_unnamed_addr #3 {
entry:
  %oob = icmp uge i64 %tid, 64
  br i1 %oob, label %ret, label %go, !prof !0

go:
  call void @hp_scan(i64 %tid)
  br label %ret

ret:
  ret void
}

; ===========================================================================
; SEQLOCK
; ===========================================================================

define void @universe_conc_seqlock_write_begin(ptr %s) local_unnamed_addr #1 {
entry:
  %v = load atomic i64, ptr %s monotonic, align 8
  %v1 = add i64 %v, 1                                    ; -> odd
  store atomic i64 %v1, ptr %s monotonic, align 8
  fence release                                          ; data writes stay after
  ret void
}

define void @universe_conc_seqlock_write_end(ptr %s) local_unnamed_addr #1 {
entry:
  %v = load atomic i64, ptr %s monotonic, align 8
  %v1 = add i64 %v, 1                                    ; -> even
  store atomic i64 %v1, ptr %s release, align 8          ; publish data
  ret void
}

define i64 @universe_conc_seqlock_read_begin(ptr %s) local_unnamed_addr #2 {
entry:
  %v = load atomic i64, ptr %s acquire, align 8
  ret i64 %v
}

; read_retry — true if the caller must retry (writer active or version moved).
define i1 @universe_conc_seqlock_read_retry(ptr %s, i64 %prev) local_unnamed_addr #2 {
entry:
  fence acquire
  %v = load atomic i64, ptr %s monotonic, align 8
  %odd.bit = and i64 %prev, 1
  %was.odd = icmp ne i64 %odd.bit, 0
  %changed = icmp ne i64 %v, %prev
  %retry = or i1 %was.odd, %changed
  ret i1 %retry
}

; ===========================================================================
; TEARDOWN — free every realloc'd bag / retire-list array and zero all state.
; Not part of the hot path; used at shutdown so no memory is retained.
; ===========================================================================
define void @universe_conc_ebr_reset() local_unnamed_addr #3 {
entry:
  store atomic i64 0, ptr @ebr monotonic, align 8
  br label %tloop

tloop:
  %t = phi i64 [ 0, %entry ], [ %t.n, %tdone ]
  %off = shl nuw nsw i64 %t, 8
  %base = add nuw nsw i64 %off, 128
  %le.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %base
  store atomic i64 0, ptr %le.p monotonic, align 8
  br label %bloop

bloop:
  %b = phi i64 [ 0, %tloop ], [ %b.n, %bcont ]
  %brel0 = mul nuw nsw i64 %b, 32                        ; bag stride 32
  %brel = add nuw nsw i64 %brel0, 128
  %babs = add nuw nsw i64 %base, %brel
  %cnt.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %babs
  %cap.off = add nuw nsw i64 %babs, 8
  %cap.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %cap.off
  %arr.off = add nuw nsw i64 %babs, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %arr.off
  %ep.off = add nuw nsw i64 %babs, 24
  %ep.p = getelementptr inbounds nuw i8, ptr @ebr, i64 %ep.off
  store i64 0, ptr %ep.p, align 8
  %arr = load ptr, ptr %arr.p, align 8
  %isnull = icmp eq ptr %arr, null
  br i1 %isnull, label %bcont, label %dofree

dofree:
  call void @free(ptr %arr)
  store i64 0, ptr %cnt.p, align 8
  store i64 0, ptr %cap.p, align 8
  store ptr null, ptr %arr.p, align 8
  br label %bcont

bcont:
  %b.n = add nuw nsw i64 %b, 1
  %bmore = icmp ult i64 %b.n, 3
  br i1 %bmore, label %bloop, label %tdone

tdone:
  %t.n = add nuw nsw i64 %t, 1
  %tmore = icmp ult i64 %t.n, 64
  br i1 %tmore, label %tloop, label %ret

ret:
  ret void
}

define void @universe_conc_hp_reset() local_unnamed_addr #3 {
entry:
  br label %tloop

tloop:
  %t = phi i64 [ 0, %entry ], [ %t.n, %tcont ]
  ; free retire-list array
  %loff = shl nuw nsw i64 %t, 5
  %cnt.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %loff
  %cap.rel = add nuw nsw i64 %loff, 8
  %cap.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %cap.rel
  %arr.rel = add nuw nsw i64 %loff, 16
  %arr.p = getelementptr inbounds nuw i8, ptr @hp_retire_lists, i64 %arr.rel
  %arr = load ptr, ptr %arr.p, align 8
  %isnull = icmp eq ptr %arr, null
  br i1 %isnull, label %clearslots, label %dofree

dofree:
  call void @free(ptr %arr)
  store i64 0, ptr %cnt.p, align 8
  store i64 0, ptr %cap.p, align 8
  store ptr null, ptr %arr.p, align 8
  br label %clearslots

clearslots:
  %sbase = shl nuw nsw i64 %t, 7                         ; t*128
  br label %kloop

kloop:
  %kk = phi i64 [ 0, %clearslots ], [ %kk.n, %kloop ]
  %koff = shl nuw nsw i64 %kk, 3
  %soff = add nuw nsw i64 %sbase, %koff
  %slot.p = getelementptr inbounds nuw i8, ptr @hp_slots, i64 %soff
  store atomic ptr null, ptr %slot.p monotonic, align 8
  %kk.n = add nuw nsw i64 %kk, 1
  %kmore = icmp ult i64 %kk.n, 8
  br i1 %kmore, label %kloop, label %tcont

tcont:
  %t.n = add nuw nsw i64 %t, 1
  %tmore = icmp ult i64 %t.n, 64
  br i1 %tmore, label %tloop, label %ret

ret:
  ret void
}

attributes #0 = { nounwind willreturn norecurse }
attributes #1 = { nounwind willreturn }
attributes #2 = { nounwind willreturn norecurse memory(read) }
attributes #3 = { nounwind }

!0 = !{!"branch_weights", i32 1, i32 2000}
