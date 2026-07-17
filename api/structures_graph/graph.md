# structures_graph/graph

## Purpose

Array-backed graph (directed or undirected, optional i64 edge weights) with
BFS, DFS, Dijkstra, connected-components, and unweighted shortest path. The
graph is three flat i32 vertex arrays (head/tail/degree) plus one flat 16-byte
edge-record array; a vertex's out-adjacency is a singly-linked list threaded
through the edge array by i32 "next" indices, with no per-node malloc and no
pointer webs. Appending at the tail keeps neighbors in insertion order for
deterministic traversal. Vertex count is capped at 2^30 so all index math stays
in i32. Single-threaded; for concurrent edge insertion use `graph_sharded`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_graph_create(int64_t nverts, int32_t directed)` | Create with `nverts` initial vertices; `directed` != 0 for a digraph | handle, or NULL on OOM |
| `int64_t universe_ds_graph_add_vertex(void *g)` | Append a vertex | new id, or -1 on OOM |
| `int32_t universe_ds_graph_add_edge(void *g, int64_t u, int64_t v, int64_t w)` | Add edge u->v with weight `w` (both directions if undirected) | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX |
| `int64_t universe_ds_graph_degree(void *g, int64_t u)` | Out-degree of `u` | degree, or -1 bad index |
| `int32_t universe_ds_graph_has_edge(void *g, int64_t u, int64_t v)` | Adjacency test | 1 yes, 0 no |
| `int64_t universe_ds_graph_neighbors(void *g, int64_t u, int64_t *out, int64_t max)` | Copy up to `max` neighbor ids of `u` into `out` | count written |
| `int64_t universe_ds_graph_vcount(void *g)` | Vertex count | count |
| `int64_t universe_ds_graph_ecount(void *g)` | Edge-record count | count |
| `int64_t universe_ds_graph_bfs(void *g, int64_t src, int64_t *order_out)` | BFS from `src`, writes visit order | count, or -1 err |
| `int64_t universe_ds_graph_dfs(void *g, int64_t src, int64_t *order_out)` | DFS from `src`, writes visit order | count, or -1 err |
| `int32_t universe_ds_graph_dijkstra(void *g, int64_t src, int64_t *dist_out)` | Shortest-path distances (nonneg weights) into `dist_out[nverts]` | 0 OK, or error code |
| `int64_t universe_ds_graph_connected_components(void *g, int64_t *labels_out)` | Component label per vertex | k components, or -1 err |
| `int64_t universe_ds_graph_shortest_path(void *g, int64_t u, int64_t v)` | Unweighted BFS hop distance | hops, or -1 unreachable |
| `void universe_ds_graph_destroy(void *g)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_graph_create(i64, i32)
declare i32 @universe_ds_graph_add_edge(ptr, i64, i64, i64)
declare i64 @universe_ds_graph_shortest_path(ptr, i64, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read `u v` edge pairs, then print the hop distance between two vertices.

```c
// graphcli.c — build: clang -O3 graphcli.c build/libuniverse.a -lpthread -lm -o graphcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_graph_create(int64_t, int32_t);
extern int32_t universe_ds_graph_add_edge(void *, int64_t, int64_t, int64_t);
extern int64_t universe_ds_graph_shortest_path(void *, int64_t, int64_t);
extern void universe_ds_graph_destroy(void *);

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s SRC DST < edges\n", argv[0]); return 1; }
  void *g = universe_ds_graph_create(64, 0 /* undirected */);
  long long u, v;
  while (scanf("%lld %lld", &u, &v) == 2) universe_ds_graph_add_edge(g, u, v, 1);
  long long s = atoll(argv[1]), d = atoll(argv[2]);
  printf("hops(%lld,%lld)=%lld\n", s, d, (long long)universe_ds_graph_shortest_path(g, s, d));
  universe_ds_graph_destroy(g);
  return 0;
}
```

```sh
printf '0 1\n1 2\n2 3\n' | ./graphcli 0 3   # -> hops(0,3)=3
```

## Notes

- Vertex ids must be `< nverts` (grow with `add_vertex`); ids are dense i32
  internally, capped at 2^30.
- `dijkstra` requires non-negative weights; `shortest_path` ignores weights
  (unweighted hops). Traversal output buffers must hold up to `vcount` entries.
- Single-threaded; not safe for concurrent mutation.
