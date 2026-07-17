# ml/hnsw — HNSW approximate nearest-neighbor index

## Purpose

An in-memory HNSW (Hierarchical Navigable Small World) approximate
nearest-neighbor index — a multi-layer proximity graph giving O(log N) expected
query cost vs the O(N) flat scan of [`knn`](neighbors.md). A node's level is
drawn from a geometric distribution; the sparse upper layers are a "highway"
descended greedily (ef=1) to land near the query, then layer 0 is explored
best-first with breadth `ef` to collect the candidate neighbourhood. Graph
quality comes from the neighbour-selection heuristic (keep a candidate only if it
is closer to the new node than to every already-kept neighbour — spreads links,
raises recall). SoA, flat index-linked graph (NO pointer chasing): vectors are
`capacity×dims` f32 row-major (cosine ⇒ normalized on store), labels are
`capacity` i64, distances are squared-L2 via a 4-accumulator `<4 x float>`
reduction.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ml_hnsw_create(int64_t dims, int32_t metric, int64_t m, int64_t efc)` | Build an empty index (`m` = links/node, `efc` = build breadth) | index handle, or NULL on OOM |
| `int32_t universe_ml_hnsw_insert(void *h, const float *vec, int64_t label)` | Insert a `dims`-vector with caller key `label` | 0 OK, 2 OOM, codes |
| `int32_t universe_ml_hnsw_search(void *h, const float *qvec, int64_t k, int64_t efs, int64_t *out_labels, float *out_dists, int64_t *out_n)` | k-NN search with breadth `efs` | 0 OK, codes |
| `int64_t universe_ml_hnsw_len(void *h)` | Number of inserted vectors | count |
| `void universe_ml_hnsw_destroy(void *h)` | Free the index | — |

`metric`: **0 = cosine, 1 = L2**. `out_labels` (i64) / `out_dists` (f32) are
caller arrays of ≥ `k` slots; `*out_n` is the count actually returned
(≤ `k`, ≤ index size). For cosine, distances are on normalized vectors; for L2,
`out_dists` are squared-L2. `efs ≥ k` improves recall at higher cost.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare ptr @universe_ml_hnsw_create(i64, i32, i64, i64)
declare i32 @universe_ml_hnsw_insert(ptr, ptr, i64)
declare i32 @universe_ml_hnsw_search(ptr, ptr, i64, i64, ptr, ptr, ptr)
declare void @universe_ml_hnsw_destroy(ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Build an index from a fixed 2-D set, then return the nearest label for each
stdin query point.

```c
// hnsw.c    reads "x y" per line, prints nearest label
#include <stdint.h>
#include <stdio.h>
void   *universe_ml_hnsw_create(int64_t, int32_t, int64_t, int64_t);
int32_t universe_ml_hnsw_insert(void *, const float *, int64_t);
int32_t universe_ml_hnsw_search(void *, const float *, int64_t, int64_t,
            int64_t *, float *, int64_t *);
void    universe_ml_hnsw_destroy(void *);

int main(void) {
    void *h = universe_ml_hnsw_create(/*dims*/2, /*metric L2*/1, /*m*/16, /*efc*/100);
    float pts[][2] = {{0,0},{1,1},{10,10},{11,11}};
    for (int64_t i = 0; i < 4; i++) universe_ml_hnsw_insert(h, pts[i], i);
    float q[2];
    while (scanf("%f %f", &q[0], &q[1]) == 2) {
        int64_t lbl[1], n = 0; float dist[1];
        universe_ml_hnsw_search(h, q, 1, 32, lbl, dist, &n);
        if (n) printf("label=%lld dist2=%g\n", (long long)lbl[0], dist[0]);
    }
    universe_ml_hnsw_destroy(h);
    return 0;
}
```

```
clang -O3 hnsw.c build/libuniverse.a -lpthread -lm -o hnsw
printf '0.2 0.1\n10.5 10.5\n' | ./hnsw
```

## Notes

- **dtype/layout:** f32 vectors row-major; SoA, flat index-linked graph — no
  pointer chasing on the hot path. Cosine vectors are normalized on store.
- **metric:** 0 = cosine, 1 = L2 (squared-L2 distances reported for L2).
- **Approximate:** recall rises with `m`, `efc` (build) and `efs` (query); use
  [`knn`](neighbors.md) as the exact oracle.
- **Ownership:** one owned allocation grown as needed; `destroy` frees all.
- **Threading:** single-writer; layout leaves room for a shared read lock but
  concurrency is deferred — treat as single-threaded.
