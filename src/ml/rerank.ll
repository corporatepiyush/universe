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

; universe_ml_rrf / universe_ml_mmr — two pure re-ranking kernels for the
; retrieval/query side: Reciprocal Rank Fusion (order-only score fusion of N
; ranked id lists) and greedy Maximal Marginal Relevance (relevance/diversity
; trade-off re-rank). Both compute over caller-owned memory; RRF uses ONE
; scratch calloc (freed on exit), MMR uses ONE scratch calloc (freed on exit).
;
; DESIGN
;   RRF (universe_ml_rrf):
;     * score(id) += 1/(rrf_k + rank), rank 0-based within each list; an id in
;       several lists accumulates — the whole point of hybrid lexical+semantic
;       fusion. rrf_k <= 0 (or NaN) selects the standard default 60.
;     * AGGREGATION is a single open-addressing i64->f32 map (linear probing,
;       power-of-two capacity, mask wrap). ONE calloc holds keys[cap] (i64) then
;       vals[cap] (f32) then used[cap] (i8) — calloc zeroes `used` (0 = empty)
;       and the value cells. cap = next_pow2(max(16, 2*total_ids)) so the table
;       is <=50% full: linear probing always finds an empty slot (no infinite
;       probe) and stays cache-friendly. Hash = id * 0x9E3779B97F4A7C15, mask.
;     * TOP-K SELECTION is a partial selection over the map slots: pick the max
;       score out_cap times, ties broken by LOWEST id (stable, deterministic),
;       marking a chosen slot used=2 so it is skipped next round. O(out_cap*cap);
;       out_cap is the small result window, so this beats a full sort of every
;       distinct id and needs no extra scratch. Not the SIMD hot path.
;   MMR (universe_ml_mmr):
;     * greedy: each round pick argmax over not-yet-selected i of
;       lambda*rel[i] - (1-lambda)*max_{j in selected} cos(vec_i, vec_j).
;       lambda clamped to [0,1]; ties keep the LOWEST index (strict >), so the
;       selection order is deterministic and, at lambda=1, is pure top-k by rel.
;     * the per-candidate max-similarity cache `maxsim[]` is updated
;       INCREMENTALLY: after picking p, fold cos(vec_c, vec_p) into maxsim[c]
;       for every remaining c. Total cost O(k*n*d), not O(k^2*n*d).
;     * cos() is an alwaysinline <4 x float> reduction (dot + both norms in one
;       pass, 4-wide vector accumulators, scalar tail) so the hot O(k*n*d) loop
;       has NO cross-module call; mirrors the SIMD reduction discipline of
;       src/ml/kernels.ll. denom==0 (either vector zero-magnitude) yields 0.
;
; API:
;   i32 universe_ml_rrf(ptr lists_ids, ptr lists_len, i64 nlists, float rrf_k,
;                       ptr out_ids, ptr out_scores, i64 out_cap, ptr out_n)
;   i32 universe_ml_mmr(ptr vecs, i64 n, i64 d, ptr rel, float lambda, i64 k,
;                       ptr out_idx, ptr out_n)
; Error codes: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW.

declare ptr @calloc(i64, i64)
declare void @free(ptr)
declare float @llvm.sqrt.f32(float)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.umin.i64(i64, i64)
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

; ============================================================ cosine (internal)
; cos(a, b) over d dims: dot/na/nb in one <4 x float> pass + scalar tail; 0 when
; either norm is 0. alwaysinline: folds into the MMR update loop at zero cost.
define internal float @ml_cosine(ptr readonly %a, ptr readonly %b, i64 %d) #1 {
entry:
  %has4 = icmp uge i64 %d, 4
  br i1 %has4, label %vloop, label %tailinit

vloop:
  %i = phi i64 [ 0, %entry ], [ %inext, %vloop ]
  %vdot = phi <4 x float> [ zeroinitializer, %entry ], [ %vdotn, %vloop ]
  %vna = phi <4 x float> [ zeroinitializer, %entry ], [ %vnan, %vloop ]
  %vnb = phi <4 x float> [ zeroinitializer, %entry ], [ %vnbn, %vloop ]
  %pa = getelementptr inbounds nuw float, ptr %a, i64 %i
  %va = load <4 x float>, ptr %pa, align 4
  %pb = getelementptr inbounds nuw float, ptr %b, i64 %i
  %vb = load <4 x float>, ptr %pb, align 4
  %mdot = fmul fast <4 x float> %va, %vb
  %vdotn = fadd fast <4 x float> %vdot, %mdot
  %mna = fmul fast <4 x float> %va, %va
  %vnan = fadd fast <4 x float> %vna, %mna
  %mnb = fmul fast <4 x float> %vb, %vb
  %vnbn = fadd fast <4 x float> %vnb, %mnb
  %inext = add nuw i64 %i, 4
  %lim = sub nuw i64 %d, 4
  %more = icmp ule i64 %inext, %lim
  br i1 %more, label %vloop, label %vdone

vdone:
  %hdot = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %vdotn)
  %hna = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %vnan)
  %hnb = call fast float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> %vnbn)
  br label %tailinit

tailinit:
  %start = phi i64 [ 0, %entry ], [ %inext, %vdone ]
  %dot0 = phi float [ 0.0, %entry ], [ %hdot, %vdone ]
  %na0 = phi float [ 0.0, %entry ], [ %hna, %vdone ]
  %nb0 = phi float [ 0.0, %entry ], [ %hnb, %vdone ]
  br label %tail

tail:
  %j = phi i64 [ %start, %tailinit ], [ %jnext, %tailbody ]
  %dacc = phi float [ %dot0, %tailinit ], [ %daccn, %tailbody ]
  %naacc = phi float [ %na0, %tailinit ], [ %naaccn, %tailbody ]
  %nbacc = phi float [ %nb0, %tailinit ], [ %nbaccn, %tailbody ]
  %tdone = icmp uge i64 %j, %d
  br i1 %tdone, label %finish, label %tailbody

tailbody:
  %ta = getelementptr inbounds nuw float, ptr %a, i64 %j
  %fa = load float, ptr %ta, align 4
  %tb = getelementptr inbounds nuw float, ptr %b, i64 %j
  %fb = load float, ptr %tb, align 4
  %pd = fmul fast float %fa, %fb
  %daccn = fadd fast float %dacc, %pd
  %pa2 = fmul fast float %fa, %fa
  %naaccn = fadd fast float %naacc, %pa2
  %pb2 = fmul fast float %fb, %fb
  %nbaccn = fadd fast float %nbacc, %pb2
  %jnext = add nuw i64 %j, 1
  br label %tail

finish:
  %sqa = call float @llvm.sqrt.f32(float %naacc)
  %sqb = call float @llvm.sqrt.f32(float %nbacc)
  %denom = fmul float %sqa, %sqb
  %zero = fcmp oeq float %denom, 0.0
  %sim = fdiv float %dacc, %denom
  %res = select i1 %zero, float 0.0, float %sim
  ret float %res
}

; ==================================================================== rrf API
define i32 @universe_ml_rrf(ptr %lists_ids, ptr %lists_len, i64 %nlists,
                            float %rrf_k, ptr %out_ids, ptr %out_scores,
                            i64 %out_cap, ptr %out_n) #0 {
entry:
  %n0 = icmp eq ptr %lists_ids, null
  %n1 = icmp eq ptr %lists_len, null
  %n2 = icmp eq ptr %out_ids, null
  %n3 = icmp eq ptr %out_scores, null
  %n4 = icmp eq ptr %out_n, null
  %or0 = or i1 %n0, %n1
  %or1 = or i1 %n2, %n3
  %or2 = or i1 %or0, %or1
  %anynull = or i1 %or2, %n4
  br i1 %anynull, label %err.null, label %kset

kset:
  ; rrf_k <= 0 (or NaN) => default 60
  %kok = fcmp ogt float %rrf_k, 0.0
  %kf = select i1 %kok, float %rrf_k, float 6.0e1
  ; ---- pass 1: validate list pointers + sum lengths (overflow-checked) ----
  %hasl = icmp ugt i64 %nlists, 0
  br i1 %hasl, label %sum, label %sumdone

sum:
  %sl = phi i64 [ 0, %kset ], [ %sln, %sumnext ]
  %stot = phi i64 [ 0, %kset ], [ %stotn, %sumnext ]
  %lenp = getelementptr inbounds nuw i64, ptr %lists_len, i64 %sl
  %len = load i64, ptr %lenp, align 8
  %lpp = getelementptr inbounds nuw ptr, ptr %lists_ids, i64 %sl
  %lp = load ptr, ptr %lpp, align 8
  %lennz = icmp ugt i64 %len, 0
  %lpnull = icmp eq ptr %lp, null
  %badlist = and i1 %lennz, %lpnull
  br i1 %badlist, label %err.null, label %sumcont

sumcont:
  %addr = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %stot, i64 %len)
  %stotn = extractvalue { i64, i1 } %addr, 0
  %ovf = extractvalue { i64, i1 } %addr, 1
  br i1 %ovf, label %err.ovf, label %sumnext

sumnext:
  %sln = add nuw i64 %sl, 1
  %smore = icmp ult i64 %sln, %nlists
  br i1 %smore, label %sum, label %sumdone

sumdone:
  %total = phi i64 [ 0, %kset ], [ %stotn, %sumnext ]
  %empty = icmp eq i64 %total, 0
  br i1 %empty, label %noresult, label %alloc

; ---- size + allocate the open-addressing map (single calloc) ----
alloc:
  %want0 = shl i64 %total, 1
  %want = call i64 @llvm.umax.i64(i64 %want0, i64 16)
  %wm1 = sub i64 %want, 1
  %lz = call i64 @llvm.ctlz.i64(i64 %wm1, i1 false)
  %shift = sub i64 64, %lz
  %cap = shl i64 1, %shift
  %mask = sub i64 %cap, 1
  ; bytes = cap*8 (keys) + cap*4 (vals) + cap*1 (used) = cap*13
  %bpair = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %cap, i64 13)
  %bytes = extractvalue { i64, i1 } %bpair, 0
  %bovf = extractvalue { i64, i1 } %bpair, 1
  br i1 %bovf, label %err.ovf, label %doalloc

doalloc:
  %base = call ptr @calloc(i64 1, i64 %bytes)
  %anull = icmp eq ptr %base, null
  br i1 %anull, label %err.oom, label %maps

maps:
  %keys = getelementptr inbounds nuw i8, ptr %base, i64 0
  %voff = shl i64 %cap, 3
  %vals = getelementptr inbounds nuw i8, ptr %base, i64 %voff
  %uoff0 = mul nuw i64 %cap, 12
  %used = getelementptr inbounds nuw i8, ptr %base, i64 %uoff0
  ; ---- pass 2: insert/accumulate contributions ----
  br label %il

; iterate lists
il:
  %li = phi i64 [ 0, %maps ], [ %lin, %ildone ]
  %ilenp = getelementptr inbounds nuw i64, ptr %lists_len, i64 %li
  %ilen = load i64, ptr %ilenp, align 8
  %ilpp = getelementptr inbounds nuw ptr, ptr %lists_ids, i64 %li
  %ilp = load ptr, ptr %ilpp, align 8
  %ilennz = icmp ugt i64 %ilen, 0
  br i1 %ilennz, label %rl, label %ildone

; iterate ranks within list
rl:
  %ri = phi i64 [ 0, %il ], [ %rin, %probe.done ]
  %idp = getelementptr inbounds nuw i64, ptr %ilp, i64 %ri
  %id = load i64, ptr %idp, align 8
  %rf = uitofp i64 %ri to float
  %den = fadd float %kf, %rf
  %contrib = fdiv float 1.0, %den
  ; hash
  %h = mul i64 %id, -7046029254386353131
  %slot0 = and i64 %h, %mask
  br label %probe

probe:
  %slot = phi i64 [ %slot0, %rl ], [ %slotnext, %probe.miss ]
  %up = getelementptr inbounds nuw i8, ptr %used, i64 %slot
  %u = load i8, ptr %up, align 1
  %isempty = icmp eq i8 %u, 0
  br i1 %isempty, label %probe.insert, label %probe.check

probe.check:
  %kp = getelementptr inbounds nuw i64, ptr %keys, i64 %slot
  %kk = load i64, ptr %kp, align 8
  %match = icmp eq i64 %kk, %id
  br i1 %match, label %probe.accum, label %probe.miss

probe.miss:
  %addslot = add i64 %slot, 1
  %slotnext = and i64 %addslot, %mask
  br label %probe

probe.accum:
  %vp = getelementptr inbounds nuw float, ptr %vals, i64 %slot
  %vold = load float, ptr %vp, align 4
  %vnew = fadd float %vold, %contrib
  store float %vnew, ptr %vp, align 4
  br label %probe.done

probe.insert:
  store i8 1, ptr %up, align 1
  %ikp = getelementptr inbounds nuw i64, ptr %keys, i64 %slot
  store i64 %id, ptr %ikp, align 8
  %ivp = getelementptr inbounds nuw float, ptr %vals, i64 %slot
  store float %contrib, ptr %ivp, align 4
  br label %probe.done

probe.done:
  %rin = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %rin, %ilen
  br i1 %rmore, label %rl, label %ildone

ildone:
  %lin = add nuw i64 %li, 1
  %lmore = icmp ult i64 %lin, %nlists
  br i1 %lmore, label %il, label %select

; ---- partial selection: pick top out_cap by score desc, ties by lowest id ----
select:
  %cap0 = icmp ugt i64 %out_cap, 0
  br i1 %cap0, label %pick, label %seldone

pick:
  %count = phi i64 [ 0, %select ], [ %count2, %pick.take ]
  br label %scan

scan:
  %ss = phi i64 [ 0, %pick ], [ %ssn, %scan ]
  %bslot = phi i64 [ -1, %pick ], [ %nbslot, %scan ]
  %bscore = phi float [ 0.0, %pick ], [ %nbscore, %scan ]
  %bid = phi i64 [ 0, %pick ], [ %nbid, %scan ]
  %sup = getelementptr inbounds nuw i8, ptr %used, i64 %ss
  %su = load i8, ptr %sup, align 1
  %is1 = icmp eq i8 %su, 1
  %svp = getelementptr inbounds nuw float, ptr %vals, i64 %ss
  %sv = load float, ptr %svp, align 4
  %skp = getelementptr inbounds nuw i64, ptr %keys, i64 %ss
  %sk = load i64, ptr %skp, align 8
  %first = icmp eq i64 %bslot, -1
  %sgt = fcmp ogt float %sv, %bscore
  %seq = fcmp oeq float %sv, %bscore
  %idlt = icmp ult i64 %sk, %bid
  %tiew = and i1 %seq, %idlt
  %core = or i1 %sgt, %tiew
  %cand = or i1 %first, %core
  %take = and i1 %is1, %cand
  %nbslot = select i1 %take, i64 %ss, i64 %bslot
  %nbscore = select i1 %take, float %sv, float %bscore
  %nbid = select i1 %take, i64 %sk, i64 %bid
  %ssn = add nuw i64 %ss, 1
  %scanmore = icmp ult i64 %ssn, %cap
  br i1 %scanmore, label %scan, label %scandone

scandone:
  %found = icmp ne i64 %nbslot, -1
  br i1 %found, label %pick.take, label %seldone

pick.take:
  %tup = getelementptr inbounds nuw i8, ptr %used, i64 %nbslot
  store i8 2, ptr %tup, align 1
  %oip = getelementptr inbounds nuw i64, ptr %out_ids, i64 %count
  store i64 %nbid, ptr %oip, align 8
  %osp = getelementptr inbounds nuw float, ptr %out_scores, i64 %count
  store float %nbscore, ptr %osp, align 4
  %count2 = add nuw i64 %count, 1
  %pmore = icmp ult i64 %count2, %out_cap
  br i1 %pmore, label %pick, label %seldone

seldone:
  %finalcount = phi i64 [ 0, %select ], [ %count, %scandone ], [ %count2, %pick.take ]
  store i64 %finalcount, ptr %out_n, align 8
  call void @free(ptr %base)
  ret i32 0

noresult:
  store i64 0, ptr %out_n, align 8
  ret i32 0

err.null:
  ret i32 1

err.oom:
  ret i32 2

err.ovf:
  ret i32 3
}

; ==================================================================== mmr API
define i32 @universe_ml_mmr(ptr %vecs, i64 %n, i64 %d, ptr %rel, float %lambda,
                            i64 %k, ptr %out_idx, ptr %out_n) #0 {
entry:
  %m0 = icmp eq ptr %vecs, null
  %m1 = icmp eq ptr %rel, null
  %m2 = icmp eq ptr %out_idx, null
  %m3 = icmp eq ptr %out_n, null
  %mo0 = or i1 %m0, %m1
  %mo1 = or i1 %m2, %m3
  %mnull = or i1 %mo0, %mo1
  br i1 %mnull, label %merr.null, label %clamp

clamp:
  ; want = min(k, n)
  %want = call i64 @llvm.umin.i64(i64 %k, i64 %n)
  %wz = icmp eq i64 %want, 0
  br i1 %wz, label %mnoresult, label %lam

lam:
  %llt0 = fcmp olt float %lambda, 0.0
  %lam0 = select i1 %llt0, float 0.0, float %lambda
  %lgt1 = fcmp ogt float %lam0, 1.0
  %lamc = select i1 %lgt1, float 1.0, float %lam0
  %oneminus = fsub float 1.0, %lamc
  ; scratch: maxsim[n] (f32) then chosen[n] (i8) = n*5 bytes, zeroed by calloc
  %sc = call ptr @calloc(i64 %n, i64 5)
  %scnull = icmp eq ptr %sc, null
  br i1 %scnull, label %merr.oom, label %mmaps

mmaps:
  %maxsim = getelementptr inbounds nuw i8, ptr %sc, i64 0
  %choff = shl i64 %n, 2
  %chosen = getelementptr inbounds nuw i8, ptr %sc, i64 %choff
  br label %round

; each round: argmax score over not-chosen, then fold pick into maxsim cache
round:
  %rc = phi i64 [ 0, %mmaps ], [ %rc2, %update.done ]
  br label %ascan

ascan:
  %ac = phi i64 [ 0, %round ], [ %acn, %ascan ]
  %abidx = phi i64 [ -1, %round ], [ %nabidx, %ascan ]
  %abscore = phi float [ 0.0, %round ], [ %nabscore, %ascan ]
  %chp = getelementptr inbounds nuw i8, ptr %chosen, i64 %ac
  %ch = load i8, ptr %chp, align 1
  %notch = icmp eq i8 %ch, 0
  %relp = getelementptr inbounds nuw float, ptr %rel, i64 %ac
  %relv = load float, ptr %relp, align 4
  %msp = getelementptr inbounds nuw float, ptr %maxsim, i64 %ac
  %msv = load float, ptr %msp, align 4
  %term1 = fmul float %lamc, %relv
  %term2 = fmul float %oneminus, %msv
  %score = fsub float %term1, %term2
  %afirst = icmp eq i64 %abidx, -1
  %agt = fcmp ogt float %score, %abscore
  %abetter = or i1 %afirst, %agt
  %atake = and i1 %notch, %abetter
  %nabidx = select i1 %atake, i64 %ac, i64 %abidx
  %nabscore = select i1 %atake, float %score, float %abscore
  %acn = add nuw i64 %ac, 1
  %amore = icmp ult i64 %acn, %n
  br i1 %amore, label %ascan, label %ascandone

ascandone:
  ; a not-chosen candidate always exists while rc < want <= n
  %pch = getelementptr inbounds nuw i8, ptr %chosen, i64 %nabidx
  store i8 1, ptr %pch, align 1
  %poidx = getelementptr inbounds nuw i64, ptr %out_idx, i64 %rc
  store i64 %nabidx, ptr %poidx, align 8
  ; picked row base
  %poff = mul nuw i64 %nabidx, %d
  %prow = getelementptr inbounds nuw float, ptr %vecs, i64 %poff
  br label %update

update:
  %uc = phi i64 [ 0, %ascandone ], [ %ucn, %update.cont ]
  %uchp = getelementptr inbounds nuw i8, ptr %chosen, i64 %uc
  %uch = load i8, ptr %uchp, align 1
  %unotch = icmp eq i8 %uch, 0
  br i1 %unotch, label %update.do, label %update.cont

update.do:
  %uoff = mul nuw i64 %uc, %d
  %urow = getelementptr inbounds nuw float, ptr %vecs, i64 %uoff
  %sim = call float @ml_cosine(ptr %urow, ptr %prow, i64 %d)
  %umsp = getelementptr inbounds nuw float, ptr %maxsim, i64 %uc
  %umsv = load float, ptr %umsp, align 4
  %bigger = fcmp ogt float %sim, %umsv
  %newms = select i1 %bigger, float %sim, float %umsv
  store float %newms, ptr %umsp, align 4
  br label %update.cont

update.cont:
  %ucn = add nuw i64 %uc, 1
  %umore = icmp ult i64 %ucn, %n
  br i1 %umore, label %update, label %update.done

update.done:
  %rc2 = add nuw i64 %rc, 1
  %rmore = icmp ult i64 %rc2, %want
  br i1 %rmore, label %round, label %mdone

mdone:
  store i64 %want, ptr %out_n, align 8
  call void @free(ptr %sc)
  ret i32 0

mnoresult:
  store i64 0, ptr %out_n, align 8
  ret i32 0

merr.null:
  ret i32 1

merr.oom:
  ret i32 2
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind willreturn norecurse nosync memory(argmem: read) }
