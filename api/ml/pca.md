# ml/pca — Principal Component Analysis

## Purpose

PCA over row-major f32 samples by covariance power-iteration with Hotelling
deflation. It (1) subtracts the per-feature mean, (2) forms the `d×d` sample
covariance `Cov = Xcᵀ·Xc / (n−1)` as a sum of rank-1 outer products (each a
length-d axpy — unit stride, feeds the SIMD kernels), then (3) extracts the top-k
eigenpairs one at a time: power-iterate `v ← normalize(Cov·v)`, read the
eigenvalue as the Rayleigh quotient `λ = vᵀ·Cov·v`, deflate
`Cov ← Cov − λ·v·vᵀ`. This is the right class when `k ≪ d`
(O(iters·k·d²), far cheaper than a full eigensolve). Every candidate is
re-orthogonalized (modified Gram-Schmidt) against found components for
robustness on repeated/near-zero eigenvalues. The orchestration reuses the
exported `universe_ml_*` kernels (gemv/dot/axpy/scale/l2_norm) as real
cross-module calls.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_pca_fit(const float *X, int64_t n, int64_t d, int64_t k, float *comp, float *eig, float *evr, int64_t iters, float tol)` | Fit top-`k` components of `X` (`n×d`, `n≥2`) | 0 OK, 2 OOM, 8 INVALID_ARG, 12 NOT_CONVERGED |
| `int32_t universe_ml_pca_transform(const float *X, int64_t n, int64_t d, const float *comp, int64_t k, float *out)` | Project `X` onto `comp` → `n×k` scores | 0 OK, codes |
| `void universe_ml_pca_explained_variance_ratio(const float *eig, int64_t k, float total, float *out)` | Compute EVR = `eig[i] / total` for each component | — |

`comp` is out `k×d` components (row-major); `eig` is out `k` eigenvalues; `evr`
(nullable) is out `k` explained-variance ratios. `out` (transform) is `n×k`.
`total` for EVR is the total variance (sum of all `d` eigenvalues / feature
variances) the caller supplies.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_ml_pca_fit(ptr, i64, i64, i64, ptr, ptr, ptr, i64, float)
declare i32 @universe_ml_pca_transform(ptr, i64, i64, ptr, i64, ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read `n×d` points (from `argv` dims) and print the top-k projected scores.
Usage: `pca <d> <k>`.

```c
// pca.c   usage: pca <d> <k>  < points.txt
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t universe_ml_pca_fit(const float *, int64_t, int64_t, int64_t,
            float *, float *, float *, int64_t, float);
int32_t universe_ml_pca_transform(const float *, int64_t, int64_t,
            const float *, int64_t, float *);

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: pca <d> <k>\n"); return 2; }
    int64_t d = atoll(argv[1]), k = atoll(argv[2]);
    static float X[1 << 20];
    int64_t cnt = 0;
    while (cnt < (1 << 20) && scanf("%f", &X[cnt]) == 1) cnt++;
    int64_t n = cnt / d;
    float *comp = malloc(k * d * sizeof(float));
    float *eig  = malloc(k * sizeof(float));
    float *scores = malloc(n * k * sizeof(float));
    if (universe_ml_pca_fit(X, n, d, k, comp, eig, NULL, 200, 1e-6f)) return 1;
    universe_ml_pca_transform(X, n, d, comp, k, scores);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) printf("%g ", scores[i * k + j]);
        putchar('\n');
    }
    free(comp); free(eig); free(scores);
    return 0;
}
```

```
clang -O3 pca.c build/libuniverse.a -lpthread -lm -o pca
printf '1 1\n2 2\n3 3\n0 0\n' | ./pca 2 1
```

## Notes

- **dtype/layout:** f32, `X` is `n×d` row-major (`n ≥ 2`); `comp` is `k×d`.
- **Method:** power-iteration + deflation + modified Gram-Schmidt; returns
  12 NOT_CONVERGED if `iters` is exhausted before `tol`.
- **Ownership:** caller allocates `X`/`comp`/`eig`/`evr`/`out`; internal scratch
  (covariance, work vectors) is freed per call.
- **Threading:** stateless per call.
