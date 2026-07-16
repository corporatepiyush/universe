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

; Aho-Corasick multi-pattern matcher over a set of byte patterns.
;
; DESIGN (representation + why):
;   * FULL-DFA goto table. One flat i32 array `next[num_nodes*256]` holds the
;     goto-transformed automaton: every (state,byte) already resolves through
;     the fail links, so the search inner loop is a SINGLE table load per byte
;     (state = next[state*256 + byte]) with NO fail-follow at scan time. This
;     trades memory (1 KiB/node) for a branch-free, call-free hot leaf — the
;     right call for a scanner that runs over megabytes of text. A sparse
;     first-child/next-sibling form would halve memory for sparse alphabets
;     but adds a per-byte search loop; we choose speed (documented tradeoff).
;   * Fail links + full-DFA are built together in ONE BFS pass using the
;     classic in-place trick: because fail[u] is always shallower than u, its
;     row in `next` is already finalized when u is processed, so a missing
;     child copies next[fail[u]][c] directly (O(nodes*256), no per-node loop
;     over the fail chain).
;   * Output reporting uses dictionary-suffix links `dict[]`: dict[u] is the
;     nearest terminal proper-suffix of u (or -1). At each state we walk the
;     dict chain and emit every pattern id registered on each terminal node.
;     Terminal ids live in two flat i32 arrays `pat_id[]`/`pat_link[]` forming
;     a per-node singly-linked list rooted at `term_head[node]` — this lets
;     several patterns (e.g. duplicates or a pattern that equals another's
;     suffix ending at the same node) all report. All match reporting is COLD;
;     the no-match per-byte path is term_head[state]==-1 && dict[state]==-1.
;   * ONE allocation: a 64-byte header followed by next[], fail[], dict[],
;     term_head[], pat_id[], pat_link[]. fail[] is build-only but kept in the
;     block (freed with destroy). Node ids are i32 indices, never pointers.
;
;   Layout (bytes): [0] i64 num_nodes  [8] i64 max_nodes  [16] i64 n_patterns
;     [24] ptr next  [32] ptr dict  [40] ptr term_head  [48] ptr pat_id
;     [56] ptr pat_link.  Payload order: next, fail, dict, term_head,
;     pat_id, pat_link.
;
; API (0 OK domain; build returns null on bad args / OOM):
;   ptr universe_search_ac_build(ptr patterns, ptr lens, i64 n)
;       patterns: n pointers to byte strings; lens: n i64 lengths (each >= 1).
;   i64 universe_search_ac_search(ptr ac, ptr text, i64 len,
;                                 ptr out_ids, ptr out_ends, i64 cap)
;       reports every (pattern_id -> out_ids[k], end_offset -> out_ends[k]);
;       end_offset is the 0-based index of the match's LAST byte. Writes up to
;       cap entries; RETURNS the total match count (may exceed cap). out_ids /
;       out_ends may be null to count only. Returns -1 if ac is null.
;   void universe_search_ac_destroy(ptr ac)

declare ptr @malloc(i64) allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc"
declare void @free(ptr allocptr captures(none)) allockind("free") "alloc-family"="malloc"
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ---------------------------------------------------------------------------
define ptr @universe_search_ac_build(ptr %patterns, ptr %lens, i64 %n) local_unnamed_addr #1 {
entry:
  %n.pos = icmp ne i64 %n, 0
  %pat.null = icmp eq ptr %patterns, null
  %lens.null = icmp eq ptr %lens, null
  %ptr.bad = or i1 %pat.null, %lens.null
  %need.ptr.bad = and i1 %n.pos, %ptr.bad
  br i1 %need.ptr.bad, label %fail.null, label %validate.head, !prof !0

validate.head:                                    ; sum lens, reject empty/null
  %vi = phi i64 [ 0, %entry ], [ %vi.n, %validate.cont ]
  %total = phi i64 [ 0, %entry ], [ %total.n, %validate.cont ]
  %v.more = icmp ult i64 %vi, %n
  br i1 %v.more, label %validate.body, label %doneval

validate.body:
  %lp = getelementptr inbounds nuw i64, ptr %lens, i64 %vi
  %L = load i64, ptr %lp, align 8
  %L.bad = icmp eq i64 %L, 0
  br i1 %L.bad, label %fail.null, label %validate.pat, !prof !0

validate.pat:
  %pp = getelementptr inbounds nuw ptr, ptr %patterns, i64 %vi
  %p = load ptr, ptr %pp, align 8
  %p.null = icmp eq ptr %p, null
  br i1 %p.null, label %fail.null, label %validate.cont, !prof !0

validate.cont:
  %sum = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %total, i64 %L)
  %total.n = extractvalue { i64, i1 } %sum, 0
  %sum.o = extractvalue { i64, i1 } %sum, 1
  %vi.n = add nuw i64 %vi, 1
  br i1 %sum.o, label %fail.null, label %validate.head, !prof !0

fail.null:
  ret ptr null

doneval:
  %mn = add nuw i64 %total, 1                      ; max_nodes = total + 1
  %nbp = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %mn, i64 1024)
  %nb = extractvalue { i64, i1 } %nbp, 0           ; next bytes = mn*256*4
  %nb.o = extractvalue { i64, i1 } %nbp, 1
  %nb.huge = icmp ugt i64 %nb, 4611686018427387904
  %nb.bad = or i1 %nb.o, %nb.huge
  br i1 %nb.bad, label %fail.null, label %sizes, !prof !0

sizes:
  %pnb = mul nuw i64 %mn, 4                         ; per-node array bytes
  %patb = mul nuw i64 %n, 4                         ; pattern-id array bytes
  %off.fail = add nuw i64 64, %nb
  %off.dict = add nuw i64 %off.fail, %pnb
  %off.term = add nuw i64 %off.dict, %pnb
  %off.patid = add nuw i64 %off.term, %pnb
  %off.patlink = add nuw i64 %off.patid, %patb
  %total.bytes = add nuw i64 %off.patlink, %patb
  %base = call ptr @malloc(i64 %total.bytes)
  %base.null = icmp eq ptr %base, null
  br i1 %base.null, label %fail.null, label %setup, !prof !0

setup:
  %nextp = getelementptr inbounds nuw i8, ptr %base, i64 64
  %failp = getelementptr inbounds nuw i8, ptr %base, i64 %off.fail
  %dictp = getelementptr inbounds nuw i8, ptr %base, i64 %off.dict
  %termp = getelementptr inbounds nuw i8, ptr %base, i64 %off.term
  %patidp = getelementptr inbounds nuw i8, ptr %base, i64 %off.patid
  %patlinkp = getelementptr inbounds nuw i8, ptr %base, i64 %off.patlink
  call void @llvm.memset.p0.i64(ptr %nextp, i8 -1, i64 %nb, i1 false)
  call void @llvm.memset.p0.i64(ptr %termp, i8 -1, i64 %pnb, i1 false)
  %h.mn = getelementptr inbounds nuw i8, ptr %base, i64 8
  store i64 %mn, ptr %h.mn, align 8
  %h.n = getelementptr inbounds nuw i8, ptr %base, i64 16
  store i64 %n, ptr %h.n, align 8
  %h.next = getelementptr inbounds nuw i8, ptr %base, i64 24
  store ptr %nextp, ptr %h.next, align 8
  %h.dict = getelementptr inbounds nuw i8, ptr %base, i64 32
  store ptr %dictp, ptr %h.dict, align 8
  %h.term = getelementptr inbounds nuw i8, ptr %base, i64 40
  store ptr %termp, ptr %h.term, align 8
  %h.patid = getelementptr inbounds nuw i8, ptr %base, i64 48
  store ptr %patidp, ptr %h.patid, align 8
  %h.patlink = getelementptr inbounds nuw i8, ptr %base, i64 56
  store ptr %patlinkp, ptr %h.patlink, align 8
  br label %ins.outer.head

; ---- trie insertion --------------------------------------------------------
ins.outer.head:
  %oi = phi i64 [ 0, %setup ], [ %oi.n, %ins.reg ]
  %numnodes = phi i64 [ 1, %setup ], [ %nn, %ins.reg ]
  %o.more = icmp ult i64 %oi, %n
  br i1 %o.more, label %ins.outer.body, label %ins.done

ins.outer.body:
  %olp = getelementptr inbounds nuw i64, ptr %lens, i64 %oi
  %oL = load i64, ptr %olp, align 8
  %opp = getelementptr inbounds nuw ptr, ptr %patterns, i64 %oi
  %op = load ptr, ptr %opp, align 8
  br label %ins.inner.head

ins.inner.head:
  %ik = phi i64 [ 0, %ins.outer.body ], [ %ik.n, %ins.inner.adv ]
  %cur = phi i64 [ 0, %ins.outer.body ], [ %cur.next, %ins.inner.adv ]
  %nn = phi i64 [ %numnodes, %ins.outer.body ], [ %nn2, %ins.inner.adv ]
  %i.more = icmp ult i64 %ik, %oL
  br i1 %i.more, label %ins.inner.body, label %ins.reg

ins.inner.body:
  %cp = getelementptr inbounds nuw i8, ptr %op, i64 %ik
  %c8 = load i8, ptr %cp, align 1
  %c = zext i8 %c8 to i64
  %cur.sh = shl i64 %cur, 8
  %idx = add i64 %cur.sh, %c
  %slot = getelementptr inbounds i32, ptr %nextp, i64 %idx
  %ch = load i32, ptr %slot, align 4
  %miss = icmp eq i32 %ch, -1
  br i1 %miss, label %ins.miss, label %ins.hit

ins.miss:
  %nn.tr = trunc i64 %nn to i32
  store i32 %nn.tr, ptr %slot, align 4
  %nn.inc = add nuw i64 %nn, 1
  br label %ins.inner.adv

ins.hit:
  %ch.z = zext i32 %ch to i64
  br label %ins.inner.adv

ins.inner.adv:
  %cur.next = phi i64 [ %nn, %ins.miss ], [ %ch.z, %ins.hit ]
  %nn2 = phi i64 [ %nn.inc, %ins.miss ], [ %nn, %ins.hit ]
  %ik.n = add nuw i64 %ik, 1
  br label %ins.inner.head

ins.reg:                                          ; register pattern oi at %cur
  %tslot = getelementptr inbounds i32, ptr %termp, i64 %cur
  %old.head = load i32, ptr %tslot, align 4
  %pl = getelementptr inbounds nuw i32, ptr %patlinkp, i64 %oi
  store i32 %old.head, ptr %pl, align 4
  %pid = getelementptr inbounds nuw i32, ptr %patidp, i64 %oi
  %oi.tr = trunc i64 %oi to i32
  store i32 %oi.tr, ptr %pid, align 4
  store i32 %oi.tr, ptr %tslot, align 4
  %oi.n = add nuw i64 %oi, 1
  br label %ins.outer.head

ins.done:
  store i64 %numnodes, ptr %base, align 8          ; header num_nodes
  ; dict[0] = -1
  %d0 = getelementptr inbounds i32, ptr %dictp, i64 0
  store i32 -1, ptr %d0, align 4
  %bfsq = call ptr @malloc(i64 %pnb)                ; temp BFS queue (i32*mn)
  %bfsq.null = icmp eq ptr %bfsq, null
  br i1 %bfsq.null, label %bfs.oom, label %root.head, !prof !0

bfs.oom:                                           ; cannot build fail links
  call void @free(ptr %base)
  ret ptr null

; ---- root row: missing -> self(0); real children enqueue, fail=0 ----------
root.head:
  %rc = phi i64 [ 0, %ins.done ], [ %rc.n, %root.cont ]
  %rqt = phi i64 [ 0, %ins.done ], [ %rqt.n, %root.cont ]
  %r.more = icmp ult i64 %rc, 256
  br i1 %r.more, label %root.body, label %bfs.head

root.body:
  %rslot = getelementptr inbounds i32, ptr %nextp, i64 %rc
  %rv = load i32, ptr %rslot, align 4
  %r.miss = icmp eq i32 %rv, -1
  br i1 %r.miss, label %root.miss, label %root.child

root.miss:
  store i32 0, ptr %rslot, align 4
  br label %root.cont

root.child:
  %rv.z = zext i32 %rv to i64
  %rfp = getelementptr inbounds i32, ptr %failp, i64 %rv.z
  store i32 0, ptr %rfp, align 4
  %rqp = getelementptr inbounds i32, ptr %bfsq, i64 %rqt
  %rv.tr = trunc i64 %rv.z to i32
  store i32 %rv.tr, ptr %rqp, align 4
  %rqt.inc = add nuw i64 %rqt, 1
  br label %root.cont

root.cont:
  %rqt.n = phi i64 [ %rqt, %root.miss ], [ %rqt.inc, %root.child ]
  %rc.n = add nuw i64 %rc, 1
  br label %root.head

; ---- BFS: fail links + full-DFA rows + dict links -------------------------
bfs.head:
  %qh = phi i64 [ 0, %root.head ], [ %qh.n, %bfs.cdone ]
  %qt = phi i64 [ %rqt, %root.head ], [ %qt.new, %bfs.cdone ]
  %q.more = icmp ult i64 %qh, %qt
  br i1 %q.more, label %bfs.u, label %bfs.finish

bfs.u:
  %uqp = getelementptr inbounds i32, ptr %bfsq, i64 %qh
  %u32 = load i32, ptr %uqp, align 4
  %u = zext i32 %u32 to i64
  %qh.n = add nuw i64 %qh, 1
  %ufp = getelementptr inbounds i32, ptr %failp, i64 %u
  %fu32 = load i32, ptr %ufp, align 4
  %fu = zext i32 %fu32 to i64
  ; dict[u] = term_head[fu]!=-1 ? fu : dict[fu]
  %fu.tp = getelementptr inbounds i32, ptr %termp, i64 %fu
  %fu.th = load i32, ptr %fu.tp, align 4
  %fu.term = icmp ne i32 %fu.th, -1
  %fu.dp = getelementptr inbounds i32, ptr %dictp, i64 %fu
  %fu.dl = load i32, ptr %fu.dp, align 4
  %fu.tr = trunc i64 %fu to i32
  %dl.u = select i1 %fu.term, i32 %fu.tr, i32 %fu.dl
  %u.dp = getelementptr inbounds i32, ptr %dictp, i64 %u
  store i32 %dl.u, ptr %u.dp, align 4
  %u.sh = shl i64 %u, 8
  %fu.sh = shl i64 %fu, 8
  br label %bfs.cloop

bfs.cloop:
  %bc = phi i64 [ 0, %bfs.u ], [ %bc.n, %bfs.cadv ]
  %qti = phi i64 [ %qt, %bfs.u ], [ %qti.next, %bfs.cadv ]
  %uidx = add i64 %u.sh, %bc
  %uslot = getelementptr inbounds i32, ptr %nextp, i64 %uidx
  %v = load i32, ptr %uslot, align 4
  %fidx = add i64 %fu.sh, %bc
  %fslot = getelementptr inbounds i32, ptr %nextp, i64 %fidx
  %nf = load i32, ptr %fslot, align 4
  %v.miss = icmp eq i32 %v, -1
  br i1 %v.miss, label %bfs.copy, label %bfs.enq

bfs.copy:
  store i32 %nf, ptr %uslot, align 4
  br label %bfs.cadv

bfs.enq:
  %v.z = zext i32 %v to i64
  %vfp = getelementptr inbounds i32, ptr %failp, i64 %v.z
  store i32 %nf, ptr %vfp, align 4
  %vqp = getelementptr inbounds i32, ptr %bfsq, i64 %qti
  store i32 %v, ptr %vqp, align 4
  %qti.inc = add nuw i64 %qti, 1
  br label %bfs.cadv

bfs.cadv:
  %qti.next = phi i64 [ %qti, %bfs.copy ], [ %qti.inc, %bfs.enq ]
  %bc.n = add nuw i64 %bc, 1
  %c.more = icmp ult i64 %bc.n, 256
  br i1 %c.more, label %bfs.cloop, label %bfs.cdone

bfs.cdone:
  %qt.new = phi i64 [ %qti.next, %bfs.cadv ]
  br label %bfs.head

bfs.finish:
  call void @free(ptr %bfsq)
  ret ptr %base
}

; ---------------------------------------------------------------------------
define i64 @universe_search_ac_search(ptr %ac, ptr %text, i64 %len, ptr %out_ids, ptr %out_ends, i64 %cap) local_unnamed_addr #3 {
entry:
  %ac.null = icmp eq ptr %ac, null
  br i1 %ac.null, label %err, label %load, !prof !0

err:
  ret i64 -1

load:
  %h.next = getelementptr inbounds nuw i8, ptr %ac, i64 24
  %nextp = load ptr, ptr %h.next, align 8
  %h.dict = getelementptr inbounds nuw i8, ptr %ac, i64 32
  %dictp = load ptr, ptr %h.dict, align 8
  %h.term = getelementptr inbounds nuw i8, ptr %ac, i64 40
  %termp = load ptr, ptr %h.term, align 8
  %h.patid = getelementptr inbounds nuw i8, ptr %ac, i64 48
  %patidp = load ptr, ptr %h.patid, align 8
  %h.patlink = getelementptr inbounds nuw i8, ptr %ac, i64 56
  %patlinkp = load ptr, ptr %h.patlink, align 8
  %ids.ok = icmp ne ptr %out_ids, null
  %ends.ok = icmp ne ptr %out_ends, null
  %arr.ok = and i1 %ids.ok, %ends.ok
  %cap.pos = icmp sgt i64 %cap, 0
  %can.write = and i1 %arr.ok, %cap.pos
  %has.text = icmp ugt i64 %len, 0
  br i1 %has.text, label %scan.head, label %done0

done0:
  ret i64 0

scan.head:
  %pos = phi i64 [ 0, %load ], [ %pos.n, %rep.after ]
  %state = phi i64 [ 0, %load ], [ %state.new, %rep.after ]
  %count = phi i64 [ 0, %load ], [ %cnt.final, %rep.after ]
  %tp = getelementptr inbounds nuw i8, ptr %text, i64 %pos
  %t8 = load i8, ptr %tp, align 1
  %tc = zext i8 %t8 to i64
  %st.sh = shl i64 %state, 8
  %st.idx = add i64 %st.sh, %tc
  %st.slot = getelementptr inbounds i32, ptr %nextp, i64 %st.idx
  %st.raw = load i32, ptr %st.slot, align 4
  %state.new = zext i32 %st.raw to i64
  br label %rep.head

rep.head:                                          ; walk dict chain from node
  %node = phi i64 [ %state.new, %scan.head ], [ %dl.z, %rep.jump ]
  %cnt.in = phi i64 [ %count, %scan.head ], [ %cnt.j, %rep.jump ]
  %ntp = getelementptr inbounds i32, ptr %termp, i64 %node
  %th = load i32, ptr %ntp, align 4
  %is.term = icmp ne i32 %th, -1
  br i1 %is.term, label %list.head, label %rep.jump, !prof !1

list.head:                                         ; emit pattern-id list
  %s = phi i32 [ %th, %rep.head ], [ %sn, %list.cont ]
  %cnt.l = phi i64 [ %cnt.in, %rep.head ], [ %cnt.l2, %list.cont ]
  %s.z = sext i32 %s to i64
  %pidp = getelementptr inbounds i32, ptr %patidp, i64 %s.z
  %pid = load i32, ptr %pidp, align 4
  %below = icmp ult i64 %cnt.l, %cap
  %do.store = and i1 %can.write, %below
  br i1 %do.store, label %store.blk, label %list.cont

store.blk:
  %oip = getelementptr inbounds i32, ptr %out_ids, i64 %cnt.l
  store i32 %pid, ptr %oip, align 4
  %oep = getelementptr inbounds i64, ptr %out_ends, i64 %cnt.l
  store i64 %pos, ptr %oep, align 8
  br label %list.cont

list.cont:
  %cnt.l2 = add nuw i64 %cnt.l, 1
  %plp = getelementptr inbounds i32, ptr %patlinkp, i64 %s.z
  %sn = load i32, ptr %plp, align 4
  %s.more = icmp ne i32 %sn, -1
  br i1 %s.more, label %list.head, label %rep.jump

rep.jump:
  %cnt.j = phi i64 [ %cnt.in, %rep.head ], [ %cnt.l2, %list.cont ]
  %ndp = getelementptr inbounds i32, ptr %dictp, i64 %node
  %dl = load i32, ptr %ndp, align 4
  %dl.more = icmp ne i32 %dl, -1
  %dl.z = zext i32 %dl to i64
  br i1 %dl.more, label %rep.head, label %rep.after, !prof !1

rep.after:
  %cnt.final = phi i64 [ %cnt.j, %rep.jump ]
  %pos.n = add nuw i64 %pos, 1
  %scan.more = icmp ult i64 %pos.n, %len
  br i1 %scan.more, label %scan.head, label %done

done:
  ret i64 %cnt.final
}

; ---------------------------------------------------------------------------
define void @universe_search_ac_destroy(ptr %ac) local_unnamed_addr #1 {
entry:
  %is.null = icmp eq ptr %ac, null
  br i1 %is.null, label %done, label %do.free, !prof !0

do.free:
  call void @free(ptr nonnull %ac)
  br label %done

done:
  ret void
}

attributes #1 = { nounwind willreturn }
attributes #3 = { nounwind willreturn norecurse nosync }

!0 = !{!"branch_weights", i32 1, i32 2000}
!1 = !{!"branch_weights", i32 1, i32 2000}
