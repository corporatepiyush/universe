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

; Tests for the array-backed graph: adjacency, BFS/DFS order, Dijkstra
; distances, connected components, shortest path, errors, a large connected
; graph sanity, and a --bench mode.

declare ptr @universe_ds_graph_create(i64, i32)
declare void @universe_ds_graph_destroy(ptr)
declare i64 @universe_ds_graph_add_vertex(ptr)
declare i32 @universe_ds_graph_add_edge(ptr, i64, i64, i64)
declare i64 @universe_ds_graph_degree(ptr, i64)
declare i32 @universe_ds_graph_has_edge(ptr, i64, i64)
declare i64 @universe_ds_graph_neighbors(ptr, i64, ptr, i64)
declare i64 @universe_ds_graph_vcount(ptr)
declare i64 @universe_ds_graph_ecount(ptr)
declare i64 @universe_ds_graph_bfs(ptr, i64, ptr)
declare i64 @universe_ds_graph_dfs(ptr, i64, ptr)
declare i32 @universe_ds_graph_dijkstra(ptr, i64, ptr)
declare i64 @universe_ds_graph_connected_components(ptr, ptr)
declare i64 @universe_ds_graph_shortest_path(ptr, i64, i64)

declare void @ut_check(i1, ptr)
declare void @ut_check_eq(i64, i64, ptr)
declare i64 @ut_rand(ptr)
declare double @ut_now_sec()
declare i1 @ut_want_bench(i32, ptr)
declare i32 @ut_summary()
declare i32 @printf(ptr, ...)
declare void @ut_report_dist(ptr, i64, i64, ptr)

@gbfs.bench.samp = internal global [16 x double] zeroinitializer, align 8
@gdij.bench.samp = internal global [16 x double] zeroinitializer, align 8
@lbl.gbfs.bench = private unnamed_addr constant [13 x i8] c"BFS 20k-node\00"
@lbl.gdij.bench = private unnamed_addr constant [18 x i8] c"Dijkstra 20k-node\00"
declare ptr @malloc(i64)
declare void @free(ptr)

@m.vcount   = private unnamed_addr constant [16 x i8] c"graphA vcount=6\00", align 1
@m.ecount   = private unnamed_addr constant [17 x i8] c"graphA ecount=18\00", align 1
@m.deg      = private unnamed_addr constant [14 x i8] c"graphA degree\00", align 1
@m.nbr      = private unnamed_addr constant [17 x i8] c"graphA neighbors\00", align 1
@m.hase     = private unnamed_addr constant [16 x i8] c"graphA has_edge\00", align 1
@m.bfs      = private unnamed_addr constant [15 x i8] c"graphA BFS cnt\00", align 1
@m.bfso     = private unnamed_addr constant [17 x i8] c"graphA BFS order\00", align 1
@m.dfs      = private unnamed_addr constant [15 x i8] c"graphA DFS cnt\00", align 1
@m.dfso     = private unnamed_addr constant [17 x i8] c"graphA DFS order\00", align 1
@m.dij      = private unnamed_addr constant [18 x i8] c"graphA dijkstra 0\00", align 1
@m.dijd     = private unnamed_addr constant [16 x i8] c"graphA dij dist\00", align 1
@m.sp       = private unnamed_addr constant [20 x i8] c"graphA shortestpath\00", align 1
@m.dir      = private unnamed_addr constant [15 x i8] c"directed edges\00", align 1
@m.av       = private unnamed_addr constant [15 x i8] c"add_vertex ids\00", align 1
@m.cc       = private unnamed_addr constant [18 x i8] c"components k == 3\00", align 1
@m.ccl      = private unnamed_addr constant [17 x i8] c"component labels\00", align 1
@m.unreach  = private unnamed_addr constant [17 x i8] c"unreachable = -1\00", align 1
@m.err      = private unnamed_addr constant [15 x i8] c"error contract\00", align 1
@m.big.bfs  = private unnamed_addr constant [18 x i8] c"big BFS reaches N\00", align 1
@m.big.dfs  = private unnamed_addr constant [18 x i8] c"big DFS reaches N\00", align 1
@m.big.dij  = private unnamed_addr constant [19 x i8] c"big dijkstra reach\00", align 1

; add one weighted undirected edge, asserting rc==0
define internal void @tg_edge(ptr %g, i64 %u, i64 %v, i64 %w) {
entry:
  %rc = call i32 @universe_ds_graph_add_edge(ptr %g, i64 %u, i64 %v, i64 %w)
  %ok = icmp eq i32 %rc, 0
  call void @ut_check(i1 %ok, ptr @m.err)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  ; scratch buffers sized for the large graph (reused for small graphs)
  %order = call ptr @malloc(i64 80000)          ; 20000 * 4
  %dist  = call ptr @malloc(i64 160000)         ; 20000 * 8
  %labels = call ptr @malloc(i64 80000)
  %nbrbuf = call ptr @malloc(i64 64)
  %rstate = alloca i64, align 8

  ; ================= Graph A: undirected weighted 6-node =================
  %gA = call ptr @universe_ds_graph_create(i64 6, i32 0)
  call void @tg_edge(ptr %gA, i64 0, i64 1, i64 7)
  call void @tg_edge(ptr %gA, i64 0, i64 2, i64 9)
  call void @tg_edge(ptr %gA, i64 0, i64 5, i64 14)
  call void @tg_edge(ptr %gA, i64 1, i64 2, i64 10)
  call void @tg_edge(ptr %gA, i64 1, i64 3, i64 15)
  call void @tg_edge(ptr %gA, i64 2, i64 3, i64 11)
  call void @tg_edge(ptr %gA, i64 2, i64 5, i64 2)
  call void @tg_edge(ptr %gA, i64 3, i64 4, i64 6)
  call void @tg_edge(ptr %gA, i64 4, i64 5, i64 9)

  %vc = call i64 @universe_ds_graph_vcount(ptr %gA)
  call void @ut_check_eq(i64 %vc, i64 6, ptr @m.vcount)
  %ecnt = call i64 @universe_ds_graph_ecount(ptr %gA)
  call void @ut_check_eq(i64 %ecnt, i64 18, ptr @m.ecount)

  ; degrees: 0->3 1->3 2->4 3->3 4->2 5->3
  %d0 = call i64 @universe_ds_graph_degree(ptr %gA, i64 0)
  call void @ut_check_eq(i64 %d0, i64 3, ptr @m.deg)
  %d2 = call i64 @universe_ds_graph_degree(ptr %gA, i64 2)
  call void @ut_check_eq(i64 %d2, i64 4, ptr @m.deg)
  %d4 = call i64 @universe_ds_graph_degree(ptr %gA, i64 4)
  call void @ut_check_eq(i64 %d4, i64 2, ptr @m.deg)

  ; neighbors(2) in insertion order = [0,1,3,5]
  %nc = call i64 @universe_ds_graph_neighbors(ptr %gA, i64 2, ptr %nbrbuf, i64 16)
  call void @ut_check_eq(i64 %nc, i64 4, ptr @m.nbr)
  %n0 = load i32, ptr %nbrbuf, align 4
  %n0z = zext i32 %n0 to i64
  call void @ut_check_eq(i64 %n0z, i64 0, ptr @m.nbr)
  %n1p = getelementptr inbounds i8, ptr %nbrbuf, i64 4
  %n1 = load i32, ptr %n1p, align 4
  %n1z = zext i32 %n1 to i64
  call void @ut_check_eq(i64 %n1z, i64 1, ptr @m.nbr)
  %n2p = getelementptr inbounds i8, ptr %nbrbuf, i64 8
  %n2 = load i32, ptr %n2p, align 4
  %n2z = zext i32 %n2 to i64
  call void @ut_check_eq(i64 %n2z, i64 3, ptr @m.nbr)
  %n3p = getelementptr inbounds i8, ptr %nbrbuf, i64 12
  %n3 = load i32, ptr %n3p, align 4
  %n3z = zext i32 %n3 to i64
  call void @ut_check_eq(i64 %n3z, i64 5, ptr @m.nbr)

  ; has_edge (undirected symmetric)
  %he01 = call i32 @universe_ds_graph_has_edge(ptr %gA, i64 0, i64 1)
  %he01ok = icmp eq i32 %he01, 1
  call void @ut_check(i1 %he01ok, ptr @m.hase)
  %he10 = call i32 @universe_ds_graph_has_edge(ptr %gA, i64 1, i64 0)
  %he10ok = icmp eq i32 %he10, 1
  call void @ut_check(i1 %he10ok, ptr @m.hase)
  %he03 = call i32 @universe_ds_graph_has_edge(ptr %gA, i64 0, i64 3)
  %he03ok = icmp eq i32 %he03, 0
  call void @ut_check(i1 %he03ok, ptr @m.hase)
  %he45 = call i32 @universe_ds_graph_has_edge(ptr %gA, i64 4, i64 5)
  %he45ok = icmp eq i32 %he45, 1
  call void @ut_check(i1 %he45ok, ptr @m.hase)

  ; BFS from 0 == [0,1,2,5,3,4]
  %bc = call i64 @universe_ds_graph_bfs(ptr %gA, i64 0, ptr %order)
  call void @ut_check_eq(i64 %bc, i64 6, ptr @m.bfs)
  call void @check_order(ptr %order, i32 0, i32 1, i32 2, i32 5, i32 3, i32 4, ptr @m.bfso)

  ; DFS from 0 == [0,5,4,3,2,1]
  %fc = call i64 @universe_ds_graph_dfs(ptr %gA, i64 0, ptr %order)
  call void @ut_check_eq(i64 %fc, i64 6, ptr @m.dfs)
  call void @check_order(ptr %order, i32 0, i32 5, i32 4, i32 3, i32 2, i32 1, ptr @m.dfso)

  ; Dijkstra from 0 == [0,7,9,20,20,11]
  %dj = call i32 @universe_ds_graph_dijkstra(ptr %gA, i64 0, ptr %dist)
  %djok = icmp eq i32 %dj, 0
  call void @ut_check(i1 %djok, ptr @m.dij)
  call void @check_dist(ptr %dist, i64 0, i64 0)
  call void @check_dist(ptr %dist, i64 1, i64 7)
  call void @check_dist(ptr %dist, i64 2, i64 9)
  call void @check_dist(ptr %dist, i64 3, i64 20)
  call void @check_dist(ptr %dist, i64 4, i64 20)
  call void @check_dist(ptr %dist, i64 5, i64 11)

  ; shortest_path hops
  %sp04 = call i64 @universe_ds_graph_shortest_path(ptr %gA, i64 0, i64 4)
  call void @ut_check_eq(i64 %sp04, i64 2, ptr @m.sp)
  %sp03 = call i64 @universe_ds_graph_shortest_path(ptr %gA, i64 0, i64 3)
  call void @ut_check_eq(i64 %sp03, i64 2, ptr @m.sp)
  %sp00 = call i64 @universe_ds_graph_shortest_path(ptr %gA, i64 0, i64 0)
  call void @ut_check_eq(i64 %sp00, i64 0, ptr @m.sp)
  call void @universe_ds_graph_destroy(ptr %gA)

  ; ================= Directed graph =================
  %gD = call ptr @universe_ds_graph_create(i64 4, i32 1)
  %rd0 = call i32 @universe_ds_graph_add_edge(ptr %gD, i64 0, i64 1, i64 1)
  %rd1 = call i32 @universe_ds_graph_add_edge(ptr %gD, i64 1, i64 2, i64 1)
  %rd2 = call i32 @universe_ds_graph_add_edge(ptr %gD, i64 2, i64 3, i64 1)
  %ed = call i64 @universe_ds_graph_ecount(ptr %gD)
  call void @ut_check_eq(i64 %ed, i64 3, ptr @m.dir)         ; 3 records (one per edge)
  %hd01 = call i32 @universe_ds_graph_has_edge(ptr %gD, i64 0, i64 1)
  %hd01ok = icmp eq i32 %hd01, 1
  call void @ut_check(i1 %hd01ok, ptr @m.dir)
  %hd10 = call i32 @universe_ds_graph_has_edge(ptr %gD, i64 1, i64 0)
  %hd10ok = icmp eq i32 %hd10, 0                             ; directed: no reverse
  call void @ut_check(i1 %hd10ok, ptr @m.dir)
  %dgd0 = call i64 @universe_ds_graph_degree(ptr %gD, i64 0)
  call void @ut_check_eq(i64 %dgd0, i64 1, ptr @m.dir)
  %dgd3 = call i64 @universe_ds_graph_degree(ptr %gD, i64 3)
  call void @ut_check_eq(i64 %dgd3, i64 0, ptr @m.dir)
  %spd = call i64 @universe_ds_graph_shortest_path(ptr %gD, i64 0, i64 3)
  call void @ut_check_eq(i64 %spd, i64 3, ptr @m.dir)        ; 0->1->2->3
  call void @universe_ds_graph_destroy(ptr %gD)

  ; ================= add_vertex growth =================
  %gV = call ptr @universe_ds_graph_create(i64 0, i32 0)
  %av0 = call i64 @universe_ds_graph_add_vertex(ptr %gV)
  %av1 = call i64 @universe_ds_graph_add_vertex(ptr %gV)
  %av2 = call i64 @universe_ds_graph_add_vertex(ptr %gV)
  %av3 = call i64 @universe_ds_graph_add_vertex(ptr %gV)
  call void @ut_check_eq(i64 %av0, i64 0, ptr @m.av)
  call void @ut_check_eq(i64 %av3, i64 3, ptr @m.av)
  call void @tg_edge(ptr %gV, i64 0, i64 3, i64 1)
  %dv0 = call i64 @universe_ds_graph_degree(ptr %gV, i64 0)
  call void @ut_check_eq(i64 %dv0, i64 1, ptr @m.av)
  %dv3 = call i64 @universe_ds_graph_degree(ptr %gV, i64 3)
  call void @ut_check_eq(i64 %dv3, i64 1, ptr @m.av)
  call void @universe_ds_graph_destroy(ptr %gV)

  ; ================= connected components (3 comps) =================
  %gC = call ptr @universe_ds_graph_create(i64 6, i32 0)
  call void @tg_edge(ptr %gC, i64 0, i64 1, i64 1)
  call void @tg_edge(ptr %gC, i64 1, i64 2, i64 1)
  call void @tg_edge(ptr %gC, i64 3, i64 4, i64 1)
  %k = call i64 @universe_ds_graph_connected_components(ptr %gC, ptr %labels)
  call void @ut_check_eq(i64 %k, i64 3, ptr @m.cc)
  %l0 = load i32, ptr %labels, align 4
  %l1p = getelementptr inbounds i8, ptr %labels, i64 4
  %l1 = load i32, ptr %l1p, align 4
  %l2p = getelementptr inbounds i8, ptr %labels, i64 8
  %l2 = load i32, ptr %l2p, align 4
  %l3p = getelementptr inbounds i8, ptr %labels, i64 12
  %l3 = load i32, ptr %l3p, align 4
  %l4p = getelementptr inbounds i8, ptr %labels, i64 16
  %l4 = load i32, ptr %l4p, align 4
  %l5p = getelementptr inbounds i8, ptr %labels, i64 20
  %l5 = load i32, ptr %l5p, align 4
  %cc01 = icmp eq i32 %l0, %l1
  call void @ut_check(i1 %cc01, ptr @m.ccl)
  %cc12 = icmp eq i32 %l1, %l2
  call void @ut_check(i1 %cc12, ptr @m.ccl)
  %cc34 = icmp eq i32 %l3, %l4
  call void @ut_check(i1 %cc34, ptr @m.ccl)
  %cc03 = icmp ne i32 %l0, %l3
  call void @ut_check(i1 %cc03, ptr @m.ccl)
  %cc35 = icmp ne i32 %l3, %l5
  call void @ut_check(i1 %cc35, ptr @m.ccl)
  %cc05 = icmp ne i32 %l0, %l5
  call void @ut_check(i1 %cc05, ptr @m.ccl)
  ; unreachable shortest paths within disjoint components
  %spu = call i64 @universe_ds_graph_shortest_path(ptr %gC, i64 0, i64 5)
  call void @ut_check_eq(i64 %spu, i64 -1, ptr @m.unreach)
  %spu2 = call i64 @universe_ds_graph_shortest_path(ptr %gC, i64 0, i64 3)
  call void @ut_check_eq(i64 %spu2, i64 -1, ptr @m.unreach)
  call void @universe_ds_graph_destroy(ptr %gC)

  ; ================= error contract =================
  %en = call i32 @universe_ds_graph_add_edge(ptr null, i64 0, i64 0, i64 0)
  %en.ok = icmp eq i32 %en, 1
  call void @ut_check(i1 %en.ok, ptr @m.err)
  %gE = call ptr @universe_ds_graph_create(i64 4, i32 0)
  %ei = call i32 @universe_ds_graph_add_edge(ptr %gE, i64 0, i64 9, i64 1)
  %ei.ok = icmp eq i32 %ei, 7
  call void @ut_check(i1 %ei.ok, ptr @m.err)
  %edg = call i64 @universe_ds_graph_degree(ptr %gE, i64 99)
  %edg.ok = icmp eq i64 %edg, -1
  call void @ut_check(i1 %edg.ok, ptr @m.err)
  %ehe = call i32 @universe_ds_graph_has_edge(ptr %gE, i64 0, i64 77)
  %ehe.ok = icmp eq i32 %ehe, 0
  call void @ut_check(i1 %ehe.ok, ptr @m.err)
  %ebf = call i64 @universe_ds_graph_bfs(ptr %gE, i64 88, ptr %order)
  %ebf.ok = icmp eq i64 %ebf, -1
  call void @ut_check(i1 %ebf.ok, ptr @m.err)
  %edn = call i32 @universe_ds_graph_dijkstra(ptr null, i64 0, ptr %dist)
  %edn.ok = icmp eq i32 %edn, 1
  call void @ut_check(i1 %edn.ok, ptr @m.err)
  %edx = call i32 @universe_ds_graph_dijkstra(ptr %gE, i64 55, ptr %dist)
  %edx.ok = icmp eq i32 %edx, 7
  call void @ut_check(i1 %edx.ok, ptr @m.err)
  call void @universe_ds_graph_destroy(ptr %gE)

  ; ================= large connected graph (path + random) =================
  ; N=20000; path 0-1-...-(N-1) guarantees full connectivity.
  store i64 88172645463325252, ptr %rstate, align 8
  %gB = call ptr @universe_ds_graph_create(i64 20000, i32 0)
  br label %path.head
path.head:
  %pi = phi i64 [ 0, %entry ], [ %pi.n, %path.head ]
  %pv = add nuw i64 %pi, 1
  %prc = call i32 @universe_ds_graph_add_edge(ptr %gB, i64 %pi, i64 %pv, i64 1)
  %pi.n = add nuw i64 %pi, 1
  %pmore = icmp ult i64 %pi.n, 19999
  br i1 %pmore, label %path.head, label %rand.head
rand.head:
  %ri = phi i64 [ 0, %path.head ], [ %ri.n, %rand.head ]
  %r1 = call i64 @ut_rand(ptr %rstate)
  %ru = urem i64 %r1, 20000
  %r2 = call i64 @ut_rand(ptr %rstate)
  %rv = urem i64 %r2, 20000
  %rw = and i64 %r2, 15
  %rw1 = add nuw i64 %rw, 1
  %rc2 = call i32 @universe_ds_graph_add_edge(ptr %gB, i64 %ru, i64 %rv, i64 %rw1)
  %ri.n = add nuw i64 %ri, 1
  %rmore = icmp ult i64 %ri.n, 80000
  br i1 %rmore, label %rand.head, label %big.check
big.check:
  %bbfs = call i64 @universe_ds_graph_bfs(ptr %gB, i64 0, ptr %order)
  call void @ut_check_eq(i64 %bbfs, i64 20000, ptr @m.big.bfs)
  %bdfs = call i64 @universe_ds_graph_dfs(ptr %gB, i64 0, ptr %order)
  call void @ut_check_eq(i64 %bdfs, i64 20000, ptr @m.big.dfs)
  %bdij = call i32 @universe_ds_graph_dijkstra(ptr %gB, i64 0, ptr %dist)
  %bdij.ok = icmp eq i32 %bdij, 0
  call void @ut_check(i1 %bdij.ok, ptr @m.big.dij)
  ; last vertex reachable => dist < INF
  %ldp = getelementptr inbounds i8, ptr %dist, i64 159992    ; (20000-1)*8
  %ld = load i64, ptr %ldp, align 8
  %ldreach = icmp ult i64 %ld, 9223372036854775807
  call void @ut_check(i1 %ldreach, ptr @m.big.dij)

  ; ================= bench =================
  %wantb = call i1 @ut_want_bench(i32 %argc, ptr %argv)
  br i1 %wantb, label %bench, label %after.bench
bench:
  call void @run_bench(ptr %gB, ptr %order, ptr %dist)
  br label %after.bench
after.bench:
  call void @universe_ds_graph_destroy(ptr %gB)
  call void @free(ptr %order)
  call void @free(ptr %dist)
  call void @free(ptr %labels)
  call void @free(ptr %nbrbuf)
  %rcx = call i32 @ut_summary()
  ret i32 %rcx
}

; compare 6-element i32 order buffer against expected values
define internal void @check_order(ptr %buf, i32 %e0, i32 %e1, i32 %e2, i32 %e3, i32 %e4, i32 %e5, ptr %msg) {
entry:
  %p0 = load i32, ptr %buf, align 4
  %c0 = icmp eq i32 %p0, %e0
  %p1p = getelementptr inbounds i8, ptr %buf, i64 4
  %p1 = load i32, ptr %p1p, align 4
  %c1 = icmp eq i32 %p1, %e1
  %p2p = getelementptr inbounds i8, ptr %buf, i64 8
  %p2 = load i32, ptr %p2p, align 4
  %c2 = icmp eq i32 %p2, %e2
  %p3p = getelementptr inbounds i8, ptr %buf, i64 12
  %p3 = load i32, ptr %p3p, align 4
  %c3 = icmp eq i32 %p3, %e3
  %p4p = getelementptr inbounds i8, ptr %buf, i64 16
  %p4 = load i32, ptr %p4p, align 4
  %c4 = icmp eq i32 %p4, %e4
  %p5p = getelementptr inbounds i8, ptr %buf, i64 20
  %p5 = load i32, ptr %p5p, align 4
  %c5 = icmp eq i32 %p5, %e5
  %a0 = and i1 %c0, %c1
  %a1 = and i1 %a0, %c2
  %a2 = and i1 %a1, %c3
  %a3 = and i1 %a2, %c4
  %a4 = and i1 %a3, %c5
  call void @ut_check(i1 %a4, ptr %msg)
  ret void
}

define internal void @check_dist(ptr %dist, i64 %idx, i64 %exp) {
entry:
  %off = shl nuw i64 %idx, 3
  %p = getelementptr inbounds i8, ptr %dist, i64 %off
  %v = load i64, ptr %p, align 8
  call void @ut_check_eq(i64 %v, i64 %exp, ptr @m.dijd)
  ret void
}

; warm up, then time BFS and Dijkstra over the large graph; report min/mean.
define internal void @run_bench(ptr %g, ptr %order, ptr %dist) {
entry:
  %sink = alloca i64, align 8
  ; one op == one full traversal; ns/op is reported per 20000-node graph
  br label %bfs.rep.head
bfs.rep.head:
  %brep = phi i64 [ 0, %entry ], [ %brep.n, %bfs.rep.cont ]
  %t0 = call double @ut_now_sec()
  %rb = call i64 @universe_ds_graph_bfs(ptr %g, i64 0, ptr %order)
  %t1 = call double @ut_now_sec()
  store volatile i64 %rb, ptr %sink, align 8
  %dtb = fsub double %t1, %t0
  %bfs.warm = icmp eq i64 %brep, 0
  br i1 %bfs.warm, label %bfs.rep.cont, label %bfs.rep.store
bfs.rep.store:
  %bfs.si = sub i64 %brep, 1
  %bfs.sp = getelementptr inbounds double, ptr @gbfs.bench.samp, i64 %bfs.si
  store double %dtb, ptr %bfs.sp, align 8
  br label %bfs.rep.cont
bfs.rep.cont:
  %brep.n = add nuw i64 %brep, 1
  %bfs.more = icmp ult i64 %brep.n, 17
  br i1 %bfs.more, label %bfs.rep.head, label %bfs.report
bfs.report:
  call void @ut_report_dist(ptr @gbfs.bench.samp, i64 16, i64 20000, ptr @lbl.gbfs.bench)
  br label %dij.rep.head
dij.rep.head:
  %drep = phi i64 [ 0, %bfs.report ], [ %drep.n, %dij.rep.cont ]
  %s0 = call double @ut_now_sec()
  %rd = call i32 @universe_ds_graph_dijkstra(ptr %g, i64 0, ptr %dist)
  %s1 = call double @ut_now_sec()
  %rd64 = zext i32 %rd to i64
  store volatile i64 %rd64, ptr %sink, align 8
  %dtd = fsub double %s1, %s0
  %dij.warm = icmp eq i64 %drep, 0
  br i1 %dij.warm, label %dij.rep.cont, label %dij.rep.store
dij.rep.store:
  %dij.si = sub i64 %drep, 1
  %dij.sp = getelementptr inbounds double, ptr @gdij.bench.samp, i64 %dij.si
  store double %dtd, ptr %dij.sp, align 8
  br label %dij.rep.cont
dij.rep.cont:
  %drep.n = add nuw i64 %drep, 1
  %dij.more = icmp ult i64 %drep.n, 17
  br i1 %dij.more, label %dij.rep.head, label %dij.report
dij.report:
  call void @ut_report_dist(ptr @gdij.bench.samp, i64 16, i64 20000, ptr @lbl.gdij.bench)
  ret void
}

