# ml/cluster — k-means clustering

## Purpose

Lloyd's k-means over row-major f32 samples. Each pass ASSIGNS every sample to its
nearest centroid by squared-L2 (no per-comparison sqrt; argmin preserved) then
UPDATES each centroid to the mean of its members; it converges when no label
changes OR total centroid movement `≤ tol`. The hot assign step (`n·k` distance
evaluations) uses an inlined 4-accumulator `<4 x float>` squared-distance
reduction that vectorizes to fmla with no cross-module call. INIT is the first-k
rows (deterministic, allocation-free, reproducible); empty clusters keep their
previous centroid. One scratch malloc holds the accumulation sums and counts,
freed on every exit.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_kmeans_fit(const float *X, int64_t n, int64_t d, int64_t k, int64_t maxit, float tol, float *C, int32_t *labels, float *inertia)` | Fit `k` centroids to `X` (`n×d` row-major) | 0 OK, 2 OOM, 8 INVALID_ARG |
| `int32_t universe_ml_kmeans_predict(const float *X, int64_t n, int64_t d, int64_t k, const float *C, int32_t *labels)` | Assign each sample to its nearest of `C` | 0 OK, codes |

`C` is out `k×d` centroids (row-major). `labels` is out `n` `int32`. `inertia`
(nullable) receives the final sum of squared assignment distances.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_ml_kmeans_fit(ptr, i64, i64, i64, i64, float, ptr, ptr, ptr)
declare i32 @universe_ml_kmeans_predict(ptr, i64, i64, i64, ptr, ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Cluster stdin points (one `d`-dim row per line) into `k` clusters; print each
point's label. Usage: `kmeans <d> <k>`.

```c
// kmeans.c   usage: kmeans <d> <k>  < points.txt
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t universe_ml_kmeans_fit(const float *, int64_t, int64_t, int64_t,
            int64_t, float, float *, int32_t *, float *);

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: kmeans <d> <k>\n"); return 2; }
    int64_t d = atoll(argv[1]), k = atoll(argv[2]);
    static float X[1 << 20];
    int64_t cnt = 0;
    while (cnt < (1 << 20) && scanf("%f", &X[cnt]) == 1) cnt++;
    int64_t n = cnt / d;
    if (n < k) { fprintf(stderr, "need n >= k\n"); return 1; }

    float *C = malloc(k * d * sizeof(float));
    int32_t *labels = malloc(n * sizeof(int32_t));
    float inertia = 0;
    int rc = universe_ml_kmeans_fit(X, n, d, k, 100, 1e-4f, C, labels, &inertia);
    if (rc) { fprintf(stderr, "fit rc=%d\n", rc); return rc; }
    for (int64_t i = 0; i < n; i++) printf("%d\n", labels[i]);
    fprintf(stderr, "inertia=%g\n", inertia);
    free(C); free(labels);
    return 0;
}
```

```
clang -O3 kmeans.c build/libuniverse.a -lpthread -lm -o kmeans
printf '0 0\n0 1\n10 10\n10 11\n' | ./kmeans 2 2
```

## Notes

- **dtype/layout:** f32, `X` is `n×d` row-major; `C` is `k×d` row-major.
- **Determinism:** first-k rows as seeds → reproducible runs.
- **Ownership:** caller allocates `X`, `C`, `labels`, `inertia`; the module's
  one scratch buffer is internal and freed on exit.
- **Threading:** single call, single-threaded; no shared state.
