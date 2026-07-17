# ml/neighbors — brute-force k-nearest-neighbors

## Purpose

Exact brute-force kNN classify/regress over row-major f32 training data. For each
query it evaluates squared-L2 to every training row (preserves nearest ordering,
skips a per-point sqrt) and keeps the k smallest with a "replace-worst" top-k
(k-slot best-distance/index array, evict the current worst) — O(n·k) selection
with an O(k) worst-scan, ideal for small k and branch-predictable. The `n·d`
distance sweep uses an inlined 4-accumulator `<4 x float>` squared distance
(fmla, no cross-module call). Classify does a uniform majority vote over the k
labels (lowest-index tie-break); regress a uniform mean of the k targets. One
scratch malloc per call, freed on exit.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_knn_classify(const float *Xtrain, const int32_t *ytrain, int64_t n, int64_t d, const float *Xq, int64_t nq, int64_t k, int64_t nclasses, int32_t *out)` | Majority-vote class for each of `nq` queries | 0 OK, 2 OOM, 8 INVALID_ARG |
| `int32_t universe_ml_knn_regress(const float *Xtrain, const float *ytrain, int64_t n, int64_t d, const float *Xq, int64_t nq, int64_t k, float *out)` | Mean-of-k target for each of `nq` queries | 0 OK, codes |

`Xtrain` is `n×d` row-major; `Xq` is `nq×d`. `ytrain` is `n` `int32` class labels
(classify, in `[0,nclasses)`) or `n` `float` targets (regress). `out` is `nq`
`int32` predicted classes / `nq` `float` predictions.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_ml_knn_classify(ptr, ptr, i64, i64, ptr, i64, i64, i64, ptr)
declare i32 @universe_ml_knn_regress(ptr, ptr, i64, i64, ptr, i64, i64, ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

1-D classify demo: a tiny fixed training set, classify each stdin value.

```c
// knn.c    prints predicted class (0 or 1) for each stdin number
#include <stdint.h>
#include <stdio.h>
int32_t universe_ml_knn_classify(const float *, const int32_t *, int64_t, int64_t,
            const float *, int64_t, int64_t, int64_t, int32_t *);

int main(void) {
    float   Xtrain[] = {0, 1, 2, 8, 9, 10};   // 6 points, d=1
    int32_t ytrain[] = {0, 0, 0, 1, 1, 1};
    float q;
    while (scanf("%f", &q) == 1) {
        int32_t out = -1;
        universe_ml_knn_classify(Xtrain, ytrain, 6, 1, &q, 1, 3, 2, &out);
        printf("%d\n", out);
    }
    return 0;
}
```

```
clang -O3 knn.c build/libuniverse.a -lpthread -lm -o knn
printf '1\n9\n5\n' | ./knn
```

## Notes

- **dtype/layout:** f32 features row-major; classify labels are `int32` in
  `[0,nclasses)`, regress targets are `float`.
- **Exact:** no index, full scan — best for modest `n` or as the ground truth for
  the approximate indexes ([hnsw](hnsw.md), [ivf](ivf.md)).
- **Ownership:** caller owns all buffers; internal scratch freed per call.
- **Threading:** stateless per call; safe on disjoint inputs.
