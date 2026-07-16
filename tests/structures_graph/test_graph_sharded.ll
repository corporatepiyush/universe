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

; Tests for the sharded concurrent graph: single-thread mirror, an 8-thread
; add_edge stress (disjoint + cross-shard edges, no loss, terminates) repeated
; 10x, error contract, and a --bench (sharded vs single-lock).

declare ptr @universe_ds_graph_sharded_create(i64, i64, i32)
declare void @universe_ds_graph_sharded_destroy(ptr)
declare i32 @universe_ds_graph_sharded_add_edge(ptr, i64, i64, i64)
declare i64 @universe_ds_graph_sharded_degree(ptr, i64)
declare i32 @universe_ds_graph_sharded_has_edge(ptr, i64, i64)
declare i64 @universe_ds_graph_sharded_neighbors(ptr, i64, ptr, i64)
declare i64 @universe_ds_graph_sharded_ecount(ptr)
declare i64 @universe_ds_graph_sharded_vcount(ptr)
declare i64 @universe_ds_graph_sharded_shards(ptr)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
declare i32 @pthread_create(ptr, ptr, ptr, ptr)
declare i32 @pthread_join(ptr, ptr)
declare void @ut_report_dist(ptr, i64, i64, ptr)

@gsh32.bench.samp = internal global [16 x double] zeroinitializer, align 8
@gsh1.bench.samp  = internal global [16 x double] zeroinitializer, align 8
@lbl.gsh32.bench = private unnamed_addr constant [25 x i8] c"graph sharded32 add_edge\00"
@lbl.gsh1.bench  = private unnamed_addr constant [22 x i8] c"graph 1-lock add_edge\00"

@m.mirror  = private unnamed_addr constant [15 x i8] c"mirror degrees\00", align 1
@m.mecnt   = private unnamed_addr constant [14 x i8] c"mirror ecount\00", align 1
@m.mhas    = private unnamed_addr constant [16 x i8] c"mirror has_edge\00", align 1
@m.mnbr    = private unnamed_addr constant [17 x i8] c"mirror neighbors\00", align 1
@m.mvs     = private unnamed_addr constant [20 x i8] c"mirror vcount/shard\00", align 1
@m.dir     = private unnamed_addr constant [17 x i8] c"directed sharded\00", align 1
@m.err     = private unnamed_addr constant [15 x i8] c"error contract\00", align 1
@m.sfail   = private unnamed_addr constant [17 x i8] c"stress add fails\00", align 1
@m.secnt   = private unnamed_addr constant [20 x i8] c"stress ecount total\00", align 1
@m.sdeg    = private unnamed_addr constant [18 x i8] c"stress degree sum\00", align 1
@m.spres   = private unnamed_addr constant [21 x i8] c"stress edge presence\00", align 1

; ============ deterministic (u,v) edge generator, v != u, over 4096 verts ====
; Both the worker threads and the verifier use this identical stream so a
; per-thread seed reproduces exactly the edges that thread added.
define internal void @gen_uv(ptr %state, ptr %uout, ptr %vout) {
entry:
  %r1 = call i64 @ut_rand(ptr %state)
  %u = urem i64 %r1, 4096
  %r2 = call i64 @ut_rand(ptr %state)
  %off = urem i64 %r2, 4095
  %u1 = add nuw i64 %u, 1
  %sum = add nuw i64 %u1, %off
  %v = urem i64 %sum, 4096
  store i64 %u, ptr %uout, align 8
  store i64 %v, ptr %vout, align 8
  ret void
}

; worker: arg = { ptr graph@0, i64 tid@8, i64 nedges@16, i64 fails@24 }
define ptr @stress_worker(ptr %arg) {
entry:
  %uslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %state = alloca i64, align 8
  %g = load ptr, ptr %arg, align 8
  %tid.p = getelementptr inbounds i8, ptr %arg, i64 8
  %tid = load i64, ptr %tid.p, align 8
  %ne.p = getelementptr inbounds i8, ptr %arg, i64 16
  %ne = load i64, ptr %ne.p, align 8
  %seedbase = add nuw i64 %tid, 1
  %seed = mul i64 %seedbase, 88172645463325252
  store i64 %seed, ptr %state, align 8
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.n, %loop ]
  %fails = phi i64 [ 0, %entry ], [ %fails.n, %loop ]
  call void @gen_uv(ptr %state, ptr %uslot, ptr %vslot)
  %u = load i64, ptr %uslot, align 8
  %v = load i64, ptr %vslot, align 8
  %rc = call i32 @universe_ds_graph_sharded_add_edge(ptr %g, i64 %u, i64 %v, i64 1)
  %bad = icmp ne i32 %rc, 0
  %inc = zext i1 %bad to i64
  %fails.n = add nuw i64 %fails, %inc
  %i.n = add nuw i64 %i, 1
  %more = icmp ult i64 %i.n, %ne
  br i1 %more, label %loop, label %fin
fin:
  %f.p = getelementptr inbounds i8, ptr %arg, i64 24
  store i64 %fails.n, ptr %f.p, align 8
  ret ptr null
}

; one stress round: 8 threads x 15000 edges over 4096 verts / 32 shards.
define internal void @stress_once() {
entry:
  %uslot = alloca i64, align 8
  %vslot = alloca i64, align 8
  %state = alloca i64, align 8
  %tids = alloca [8 x ptr], align 8
  %args = alloca [8 x [4 x i64]], align 8
  %g = call ptr @universe_ds_graph_sharded_create(i64 4096, i64 32, i32 0)
  br label %spawn
spawn:
  %t = phi i64 [ 0, %entry ], [ %t.n, %spawn ]
  %ap = getelementptr inbounds [8 x [4 x i64]], ptr %args, i64 0, i64 %t
  store ptr %g, ptr %ap, align 8
  %ap.tid = getelementptr inbounds i8, ptr %ap, i64 8
  store i64 %t, ptr %ap.tid, align 8
  %ap.ne = getelementptr inbounds i8, ptr %ap, i64 16
  store i64 15000, ptr %ap.ne, align 8
  %ap.f = getelementptr inbounds i8, ptr %ap, i64 24
  store i64 0, ptr %ap.f, align 8
  %tp = getelementptr inbounds [8 x ptr], ptr %tids, i64 0, i64 %t
  %cr = call i32 @pthread_create(ptr %tp, ptr null, ptr @stress_worker, ptr %ap)
  %t.n = add nuw i64 %t, 1
  %smore = icmp ult i64 %t.n, 8
  br i1 %smore, label %spawn, label %join
join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jtp = getelementptr inbounds [8 x ptr], ptr %tids, i64 0, i64 %j
  %th = load ptr, ptr %jtp, align 8
  %jr = call i32 @pthread_join(ptr %th, ptr null)
  %j.n = add nuw i64 %j, 1
  %jmore = icmp ult i64 %j.n, 8
  br i1 %jmore, label %join, label %verify.fails
verify.fails:
  br label %fl
fl:
  %fi = phi i64 [ 0, %verify.fails ], [ %fi.n, %fl ]
  %facc = phi i64 [ 0, %verify.fails ], [ %facc.n, %fl ]
  %fap = getelementptr inbounds [8 x [4 x i64]], ptr %args, i64 0, i64 %fi
  %fap.f = getelementptr inbounds i8, ptr %fap, i64 24
  %fv = load i64, ptr %fap.f, align 8
  %facc.n = add i64 %facc, %fv
  %fi.n = add nuw i64 %fi, 1
  %flm = icmp ult i64 %fi.n, 8
  br i1 %flm, label %fl, label %check.count
check.count:
  call void @ut_check_eq(i64 %facc.n, i64 0, ptr @m.sfail)
  %ec = call i64 @universe_ds_graph_sharded_ecount(ptr %g)
  call void @ut_check_eq(i64 %ec, i64 240000, ptr @m.secnt)   ; 2 * 8 * 15000
  br label %ds
ds:
  %di = phi i64 [ 0, %check.count ], [ %di.n, %ds ]
  %dacc = phi i64 [ 0, %check.count ], [ %dacc.n, %ds ]
  %dg = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 %di)
  %dacc.n = add i64 %dacc, %dg
  %di.n = add nuw i64 %di, 1
  %dm = icmp ult i64 %di.n, 4096
  br i1 %dm, label %ds, label %check.deg
check.deg:
  call void @ut_check_eq(i64 %dacc.n, i64 240000, ptr @m.sdeg)
  br label %pres.t
; edge-presence sample: first 2000 edges of each thread must be present
pres.t:
  %pt = phi i64 [ 0, %check.deg ], [ %pt.n, %pres.t.next ]
  %pviol0 = phi i64 [ 0, %check.deg ], [ %pviol.carry, %pres.t.next ]
  %pseedb = add nuw i64 %pt, 1
  %pseed = mul i64 %pseedb, 88172645463325252
  store i64 %pseed, ptr %state, align 8
  br label %pres.i
pres.i:
  %pi = phi i64 [ 0, %pres.t ], [ %pi.n, %pres.i ]
  %pviol = phi i64 [ %pviol0, %pres.t ], [ %pviol.n, %pres.i ]
  call void @gen_uv(ptr %state, ptr %uslot, ptr %vslot)
  %pu = load i64, ptr %uslot, align 8
  %pv = load i64, ptr %vslot, align 8
  %h1 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 %pu, i64 %pv)
  %h2 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 %pv, i64 %pu)
  %m1 = icmp ne i32 %h1, 1
  %m2 = icmp ne i32 %h2, 1
  %e1 = zext i1 %m1 to i64
  %e2 = zext i1 %m2 to i64
  %esum = add i64 %e1, %e2
  %pviol.n = add i64 %pviol, %esum
  %pi.n = add nuw i64 %pi, 1
  %pim = icmp ult i64 %pi.n, 2000
  br i1 %pim, label %pres.i, label %pres.t.next
pres.t.next:
  %pviol.carry = phi i64 [ %pviol.n, %pres.i ]
  %pt.n = add nuw i64 %pt, 1
  %ptm = icmp ult i64 %pt.n, 8
  br i1 %ptm, label %pres.t, label %pres.done
pres.done:
  call void @ut_check_eq(i64 %pviol.carry, i64 0, ptr @m.spres)
  call void @universe_ds_graph_sharded_destroy(ptr %g)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %nbrbuf = call ptr @malloc(i64 64)

  ; ================= single-thread mirror =================
  ; nverts 8, nshards 4. (0,4) same shard0; (1,5) same shard1; (0,1),(2,3) cross.
  %g = call ptr @universe_ds_graph_sharded_create(i64 8, i64 4, i32 0)
  %a0 = call i32 @universe_ds_graph_sharded_add_edge(ptr %g, i64 0, i64 1, i64 1)
  %a1 = call i32 @universe_ds_graph_sharded_add_edge(ptr %g, i64 0, i64 4, i64 1)
  %a2 = call i32 @universe_ds_graph_sharded_add_edge(ptr %g, i64 1, i64 5, i64 1)
  %a3 = call i32 @universe_ds_graph_sharded_add_edge(ptr %g, i64 2, i64 3, i64 1)
  %dg0 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 0)
  call void @ut_check_eq(i64 %dg0, i64 2, ptr @m.mirror)
  %dg4 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 4)
  call void @ut_check_eq(i64 %dg4, i64 1, ptr @m.mirror)
  %dg1 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 1)
  call void @ut_check_eq(i64 %dg1, i64 2, ptr @m.mirror)
  %dg2 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 2)
  call void @ut_check_eq(i64 %dg2, i64 1, ptr @m.mirror)
  %dg3 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 3)
  call void @ut_check_eq(i64 %dg3, i64 1, ptr @m.mirror)
  %dg5 = call i64 @universe_ds_graph_sharded_degree(ptr %g, i64 5)
  call void @ut_check_eq(i64 %dg5, i64 1, ptr @m.mirror)
  %ec = call i64 @universe_ds_graph_sharded_ecount(ptr %g)
  call void @ut_check_eq(i64 %ec, i64 8, ptr @m.mecnt)
  %h01 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 0, i64 1)
  %h01ok = icmp eq i32 %h01, 1
  call void @ut_check(i1 %h01ok, ptr @m.mhas)
  %h10 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 1, i64 0)
  %h10ok = icmp eq i32 %h10, 1
  call void @ut_check(i1 %h10ok, ptr @m.mhas)
  %h40 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 4, i64 0)
  %h40ok = icmp eq i32 %h40, 1
  call void @ut_check(i1 %h40ok, ptr @m.mhas)
  %h32 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 3, i64 2)
  %h32ok = icmp eq i32 %h32, 1
  call void @ut_check(i1 %h32ok, ptr @m.mhas)
  %h02 = call i32 @universe_ds_graph_sharded_has_edge(ptr %g, i64 0, i64 2)
  %h02ok = icmp eq i32 %h02, 0
  call void @ut_check(i1 %h02ok, ptr @m.mhas)
  ; neighbors(0) insertion order [1,4]
  %nc = call i64 @universe_ds_graph_sharded_neighbors(ptr %g, i64 0, ptr %nbrbuf, i64 16)
  call void @ut_check_eq(i64 %nc, i64 2, ptr @m.mnbr)
  %nn0 = load i32, ptr %nbrbuf, align 4
  %nn0z = zext i32 %nn0 to i64
  call void @ut_check_eq(i64 %nn0z, i64 1, ptr @m.mnbr)
  %nn1p = getelementptr inbounds i8, ptr %nbrbuf, i64 4
  %nn1 = load i32, ptr %nn1p, align 4
  %nn1z = zext i32 %nn1 to i64
  call void @ut_check_eq(i64 %nn1z, i64 4, ptr @m.mnbr)
  %vcnt = call i64 @universe_ds_graph_sharded_vcount(ptr %g)
  call void @ut_check_eq(i64 %vcnt, i64 8, ptr @m.mvs)
  %shn = call i64 @universe_ds_graph_sharded_shards(ptr %g)
  call void @ut_check_eq(i64 %shn, i64 4, ptr @m.mvs)
  call void @universe_ds_graph_sharded_destroy(ptr %g)

  ; ================= directed sharded =================
  %gd = call ptr @universe_ds_graph_sharded_create(i64 4, i64 4, i32 1)
  %da0 = call i32 @universe_ds_graph_sharded_add_edge(ptr %gd, i64 0, i64 1, i64 1)
  %dec = call i64 @universe_ds_graph_sharded_ecount(ptr %gd)
  call void @ut_check_eq(i64 %dec, i64 1, ptr @m.dir)         ; one record only
  %dh01 = call i32 @universe_ds_graph_sharded_has_edge(ptr %gd, i64 0, i64 1)
  %dh01ok = icmp eq i32 %dh01, 1
  call void @ut_check(i1 %dh01ok, ptr @m.dir)
  %dh10 = call i32 @universe_ds_graph_sharded_has_edge(ptr %gd, i64 1, i64 0)
  %dh10ok = icmp eq i32 %dh10, 0
  call void @ut_check(i1 %dh10ok, ptr @m.dir)
  %dd1 = call i64 @universe_ds_graph_sharded_degree(ptr %gd, i64 1)
  call void @ut_check_eq(i64 %dd1, i64 0, ptr @m.dir)
  call void @universe_ds_graph_sharded_destroy(ptr %gd)

  ; ================= error contract =================
  %en = call i32 @universe_ds_graph_sharded_add_edge(ptr null, i64 0, i64 0, i64 0)
  %en.ok = icmp eq i32 %en, 1
  call void @ut_check(i1 %en.ok, ptr @m.err)
  %ge = call ptr @universe_ds_graph_sharded_create(i64 4, i64 4, i32 0)
  %ei = call i32 @universe_ds_graph_sharded_add_edge(ptr %ge, i64 0, i64 99, i64 1)
  %ei.ok = icmp eq i32 %ei, 7
  call void @ut_check(i1 %ei.ok, ptr @m.err)
  %edg = call i64 @universe_ds_graph_sharded_degree(ptr %ge, i64 77)
  %edg.ok = icmp eq i64 %edg, -1
  call void @ut_check(i1 %edg.ok, ptr @m.err)
  %ehe = call i32 @universe_ds_graph_sharded_has_edge(ptr %ge, i64 88, i64 0)
  %ehe.ok = icmp eq i32 %ehe, 0
  call void @ut_check(i1 %ehe.ok, ptr @m.err)
  call void @universe_ds_graph_sharded_destroy(ptr %ge)

  ; ================= concurrent stress, 10 rounds =================
  br label %stress.loop
stress.loop:
  %si = phi i64 [ 0, %entry ], [ %si.n, %stress.loop ]
  call void @stress_once()
  %si.n = add nuw i64 %si, 1
  %smore = icmp ult i64 %si.n, 10
  br i1 %smore, label %stress.loop, label %bench.check
bench.check:
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %after
bench:
  call void @run_bench_sharded()
  br label %after
after:
  call void @free(ptr %nbrbuf)
  %rc = call i32 @ut_summary()
  ret i32 %rc
}

; bench: 8 threads x 20000 add_edge, sharded(32) vs single-lock(1 shard).
define internal void @run_bench_sharded() {
entry:
  %tids = alloca [8 x ptr], align 8
  %args = alloca [8 x [4 x i64]], align 8
  ; --- sharded 32 (destructive: re-create each rep; warm-up rep discarded) ---
  br label %g32.rep.head
g32.rep.head:
  %g32rep = phi i64 [ 0, %entry ], [ %g32rep.n, %g32.rep.cont ]
  %g32 = call ptr @universe_ds_graph_sharded_create(i64 4096, i64 32, i32 0)
  %t0 = call double @ut_now_sec()
  call void @bench_run(ptr %g32, ptr %tids, ptr %args)
  %t1 = call double @ut_now_sec()
  call void @universe_ds_graph_sharded_destroy(ptr %g32)
  %d32 = fsub double %t1, %t0
  %g32.warm = icmp eq i64 %g32rep, 0
  br i1 %g32.warm, label %g32.rep.cont, label %g32.rep.store
g32.rep.store:
  %g32.si = sub i64 %g32rep, 1
  %g32.sp = getelementptr inbounds double, ptr @gsh32.bench.samp, i64 %g32.si
  store double %d32, ptr %g32.sp, align 8
  br label %g32.rep.cont
g32.rep.cont:
  %g32rep.n = add nuw i64 %g32rep, 1
  %g32.more = icmp ult i64 %g32rep.n, 17
  br i1 %g32.more, label %g32.rep.head, label %g32.report
g32.report:
  call void @ut_report_dist(ptr @gsh32.bench.samp, i64 16, i64 160000, ptr @lbl.gsh32.bench)
  br label %g1.rep.head
; --- single-lock (1 shard) ---
g1.rep.head:
  %g1rep = phi i64 [ 0, %g32.report ], [ %g1rep.n, %g1.rep.cont ]
  %g1 = call ptr @universe_ds_graph_sharded_create(i64 4096, i64 1, i32 0)
  %s0 = call double @ut_now_sec()
  call void @bench_run(ptr %g1, ptr %tids, ptr %args)
  %s1 = call double @ut_now_sec()
  call void @universe_ds_graph_sharded_destroy(ptr %g1)
  %d1 = fsub double %s1, %s0
  %g1.warm = icmp eq i64 %g1rep, 0
  br i1 %g1.warm, label %g1.rep.cont, label %g1.rep.store
g1.rep.store:
  %g1.si = sub i64 %g1rep, 1
  %g1.sp = getelementptr inbounds double, ptr @gsh1.bench.samp, i64 %g1.si
  store double %d1, ptr %g1.sp, align 8
  br label %g1.rep.cont
g1.rep.cont:
  %g1rep.n = add nuw i64 %g1rep, 1
  %g1.more = icmp ult i64 %g1rep.n, 17
  br i1 %g1.more, label %g1.rep.head, label %g1.report
g1.report:
  call void @ut_report_dist(ptr @gsh1.bench.samp, i64 16, i64 160000, ptr @lbl.gsh1.bench)
  ret void
}

define internal void @bench_run(ptr %g, ptr %tids, ptr %args) {
entry:
  br label %spawn
spawn:
  %t = phi i64 [ 0, %entry ], [ %t.n, %spawn ]
  %ap = getelementptr inbounds [8 x [4 x i64]], ptr %args, i64 0, i64 %t
  store ptr %g, ptr %ap, align 8
  %ap.tid = getelementptr inbounds i8, ptr %ap, i64 8
  store i64 %t, ptr %ap.tid, align 8
  %ap.ne = getelementptr inbounds i8, ptr %ap, i64 16
  store i64 20000, ptr %ap.ne, align 8
  %ap.f = getelementptr inbounds i8, ptr %ap, i64 24
  store i64 0, ptr %ap.f, align 8
  %tp = getelementptr inbounds [8 x ptr], ptr %tids, i64 0, i64 %t
  %cr = call i32 @pthread_create(ptr %tp, ptr null, ptr @stress_worker, ptr %ap)
  %t.n = add nuw i64 %t, 1
  %sm = icmp ult i64 %t.n, 8
  br i1 %sm, label %spawn, label %join
join:
  %j = phi i64 [ 0, %spawn ], [ %j.n, %join ]
  %jtp = getelementptr inbounds [8 x ptr], ptr %tids, i64 0, i64 %j
  %th = load ptr, ptr %jtp, align 8
  %jr = call i32 @pthread_join(ptr %th, ptr null)
  %j.n = add nuw i64 %j, 1
  %jm = icmp ult i64 %j.n, 8
  br i1 %jm, label %join, label %done
done:
  ret void
}
