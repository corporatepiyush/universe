# structures_graph/graph_sharded

## Purpose

Concurrent sharded-array graph. Vertices are striped across N power-of-two
shards; each shard is a self-contained flat index-linked adjacency store (the
same layout as `graph`) guarded by its own spinlock on its own 128 B cache line.
Vertex `vid` and its out-adjacency belong to shard `vid & (N-1)`, stored at
local index `vid >> log2(N)`. A directed edge locks one shard; an undirected
cross-shard edge locks the two shards in fixed (lower-index-first) order to stay
deadlock-free. **Choose this** over `graph` when many threads insert/query edges
concurrently — a single lock over one adjacency store scales negatively with
cores, striping drops contention roughly N×. This variant exposes only the
mutation/query surface (no built-in traversal algorithms).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_graph_sharded_create(int64_t nverts, int64_t nshards, int32_t directed)` | Create a striped graph | handle, or NULL on OOM |
| `int32_t universe_ds_graph_sharded_add_edge(void *m, int64_t u, int64_t v, int64_t w)` | Add edge u->v (both directions if undirected) | 0 OK, or error code |
| `int64_t universe_ds_graph_sharded_degree(void *m, int64_t u)` | Out-degree of `u` | degree, or -1 bad index |
| `int32_t universe_ds_graph_sharded_has_edge(void *m, int64_t u, int64_t v)` | Adjacency test | 1 yes, 0 no |
| `int64_t universe_ds_graph_sharded_neighbors(void *m, int64_t u, int64_t *out, int64_t max)` | Copy up to `max` neighbor ids of `u` | count written |
| `int64_t universe_ds_graph_sharded_ecount(void *m)` | Total edge records | count |
| `int64_t universe_ds_graph_sharded_vcount(void *m)` | Vertex count | count |
| `int64_t universe_ds_graph_sharded_shards(void *m)` | Shard count | count |
| `void universe_ds_graph_sharded_destroy(void *m)` | Free all shards | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_graph_sharded_create(i64, i64, i32)
declare i32 @universe_ds_graph_sharded_add_edge(ptr, i64, i64, i64)
declare i64 @universe_ds_graph_sharded_degree(ptr, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Build a graph concurrently, then report a vertex's degree.

```c
// gscli.c — build: clang -O3 gscli.c build/libuniverse.a -lpthread -lm -o gscli
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
extern void *universe_ds_graph_sharded_create(int64_t, int64_t, int32_t);
extern int32_t universe_ds_graph_sharded_add_edge(void *, int64_t, int64_t, int64_t);
extern int64_t universe_ds_graph_sharded_degree(void *, int64_t);
extern int64_t universe_ds_graph_sharded_ecount(void *);
extern void universe_ds_graph_sharded_destroy(void *);

static void *worker(void *arg) {
  void *m = ((void **)arg)[0]; long id = (long)((void **)arg)[1];
  for (long j = 0; j < 100; j++) universe_ds_graph_sharded_add_edge(m, id, j, 1);
  return 0;
}
int main(void) {
  void *m = universe_ds_graph_sharded_create(256, 16, 1 /* directed */);
  pthread_t t[4]; void *a[4][2];
  for (int i = 0; i < 4; i++) { a[i][0] = m; a[i][1] = (void *)(long)i;
    pthread_create(&t[i], 0, worker, a[i]); }
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  printf("deg(0)=%lld ecount=%lld\n",
         (long long)universe_ds_graph_sharded_degree(m, 0),
         (long long)universe_ds_graph_sharded_ecount(m));
  universe_ds_graph_sharded_destroy(m);
  return 0;
}
```

```sh
./gscli   # deg(0)=100 ecount=400
```

## Notes

- Thread-safe writes: each shard has its own padded spinlock; cross-shard
  undirected edges lock two shards in fixed order.
- Ownership is by `vid & (N-1)`; `nverts` is fixed at create.
- No traversal algorithms here — snapshot into `graph` (single-thread) for
  BFS/DFS/Dijkstra. Link with `-lpthread`.
