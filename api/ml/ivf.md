# ml/ivf — IVF-FLAT vector index

## Purpose

A self-contained in-memory IVF-FLAT (inverted-file) vector index — the middle
tier between an O(n·d) brute-force scan and a graph index. Training partitions
the buffered corpus into `nlist` Voronoi cells by k-means (delegated to
[`kmeans_fit`](cluster.md), a cold once-per-build call). A query scores the
`nprobe` nearest centroids, then EXHAUSTIVELY scans the members of those cells
into a bounded replace-worst top-k — cost ≈ O((nlist + nprobe·n/nlist)·d),
sub-linear when `nlist ≈ √n` and `nprobe ≪ nlist`, exact within the probed cells
(`nprobe == nlist` degenerates to an exact flat scan). Internally everything is
squared-L2 via a 4-accumulator `<4 x float>` reduction; cosine (metric 0) is
handled by L2-normalizing each vector on add and each query on search, with
spherical (re-normalized) centroids. Storage is SoA: vectors, parallel labels,
trained centroids, `list_off` prefix-sum offsets, and cell member arrays.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ml_ivf_create(int64_t dims, int32_t metric, int64_t nlist)` | Build an empty index with `nlist` cells | index handle, or NULL on OOM |
| `int32_t universe_ml_ivf_add(void *h, const float *vec, int64_t label)` | Buffer a `dims`-vector with caller key `label` | 0 OK, 2 OOM, codes |
| `int32_t universe_ml_ivf_train(void *h)` | k-means the buffered corpus into cells | 0 OK, 8 INVALID_ARG, codes |
| `int32_t universe_ml_ivf_search(void *h, const float *q, int64_t k, int64_t nprobe, int64_t *out_labels, float *out_dists, int64_t *out_n)` | Probe `nprobe` cells, exact top-k within | 0 OK, codes |
| `int64_t universe_ml_ivf_len(void *h)` | Number of added vectors | count |
| `void universe_ml_ivf_destroy(void *h)` | Free the index | — |

`metric`: **0 = cosine, 1 = L2**. Call `add` for all vectors, then `train` once,
then `search`. `out_labels` (i64) / `out_dists` (f32) are caller arrays of ≥ `k`;
`*out_n` = count returned. `out_dists` are squared-L2 (or squared-L2 on
normalized vectors for cosine).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare ptr @universe_ml_ivf_create(i64, i32, i64)
declare i32 @universe_ml_ivf_add(ptr, ptr, i64)
declare i32 @universe_ml_ivf_train(ptr)
declare i32 @universe_ml_ivf_search(ptr, ptr, i64, i64, ptr, ptr, ptr)
declare void @universe_ml_ivf_destroy(ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Add a fixed 2-D set, train, then return the nearest label per stdin query.

```c
// ivf.c    reads "x y" per line, prints nearest label
#include <stdint.h>
#include <stdio.h>
void   *universe_ml_ivf_create(int64_t, int32_t, int64_t);
int32_t universe_ml_ivf_add(void *, const float *, int64_t);
int32_t universe_ml_ivf_train(void *);
int32_t universe_ml_ivf_search(void *, const float *, int64_t, int64_t,
            int64_t *, float *, int64_t *);
void    universe_ml_ivf_destroy(void *);

int main(void) {
    void *h = universe_ml_ivf_create(/*dims*/2, /*metric L2*/1, /*nlist*/2);
    float pts[][2] = {{0,0},{1,1},{10,10},{11,11}};
    for (int64_t i = 0; i < 4; i++) universe_ml_ivf_add(h, pts[i], i);
    universe_ml_ivf_train(h);
    float q[2];
    while (scanf("%f %f", &q[0], &q[1]) == 2) {
        int64_t lbl[1], n = 0; float dist[1];
        universe_ml_ivf_search(h, q, 1, /*nprobe*/2, lbl, dist, &n);
        if (n) printf("label=%lld dist2=%g\n", (long long)lbl[0], dist[0]);
    }
    universe_ml_ivf_destroy(h);
    return 0;
}
```

```
clang -O3 ivf.c build/libuniverse.a -lpthread -lm -o ivf
printf '0.2 0.1\n10.5 10.5\n' | ./ivf
```

## Notes

- **Lifecycle:** `add` all vectors → `train` once → `search`. Searching before
  training is invalid.
- **metric:** 0 = cosine, 1 = L2; squared-L2 distances reported.
- **Tuning:** `nlist ≈ √n`, `nprobe` trades recall for speed; `nprobe == nlist`
  is an exact scan.
- **Ownership:** one owned index; `destroy` frees everything.
- **Threading:** single-threaded build + query.
