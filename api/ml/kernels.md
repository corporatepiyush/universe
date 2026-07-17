# ml/kernels — SIMD numeric kernels (BLAS-1/2 substrate)

## Purpose

The SIMD numeric substrate (BLAS-1/BLAS-2) for the ML engines (k-means, kNN,
linear/logistic, PCA). All f32, row-major, unit stride, pure compute over
caller-owned memory — never allocates. The PRIMARY path is a portable
`<4 x float>` vector loop lowering to SSE (x86) and NEON (AArch64), both baseline
everywhere, so no runtime CPU check. Reductions (dot/sum/l2/dist) run FOUR
independent accumulators (16 floats/iter) to break the FP dependency chain, fold
once, horizontal-reduce, then a scalar tail. Every kernel ships a `*_scalar`
twin: the sub-4 remainder handler AND the test oracle (a kernel is "done" only
when `|vector − scalar| < 1e-4·|scalar|`). Reduction hot paths use `fast` FP
(reassoc + contract) — deliberate and documented.

## Exported API

Each entry has a SIMD default and a `_scalar` twin with the identical signature;
only the default is listed (append `_scalar` for the oracle/fallback).

| C signature | Description |
|---|---|
| `float universe_ml_dot(const float *a, const float *b, int64_t n)` | Dot product `a·b` |
| `float universe_ml_sum(const float *x, int64_t n)` | Sum of elements |
| `float universe_ml_mean(const float *x, int64_t n)` | Arithmetic mean |
| `float universe_ml_l2_norm(const float *x, int64_t n)` | Euclidean norm `‖x‖₂` |
| `float universe_ml_l2_dist2(const float *a, const float *b, int64_t n)` | Squared L2 distance `‖a−b‖²` |
| `float universe_ml_l1_dist(const float *a, const float *b, int64_t n)` | L1 (Manhattan) distance |
| `void universe_ml_axpy(float *y, float alpha, const float *x, int64_t n)` | `y += alpha·x` (in place) |
| `void universe_ml_scale(float *x, float alpha, int64_t n)` | `x *= alpha` (in place) |
| `void universe_ml_gemv(float *y, const float *A, const float *x, int64_t m, int64_t n)` | `y = A·x`, `A` row-major `m×n` |
| `int64_t universe_ml_argmin(const float *x, int64_t n)` | Index of the minimum (lowest-index tie) |
| `int64_t universe_ml_argmax(const float *x, int64_t n)` | Index of the maximum (lowest-index tie) |

Twins: `universe_ml_dot_scalar`, `_sum_scalar`, `_mean_scalar`,
`_l2_norm_scalar`, `_l2_dist2_scalar`, `_l1_dist_scalar`, `_axpy_scalar`,
`_scale_scalar`, `_gemv_scalar`, `_argmin_scalar`, `_argmax_scalar`. No error
codes — pure math; caller guarantees valid non-null buffers and `n ≥ 0`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare float @universe_ml_dot(ptr, ptr, i64)
declare float @universe_ml_l2_dist2(ptr, ptr, i64)
declare i64   @universe_ml_argmax(ptr, i64)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read whitespace-separated floats from stdin; print sum, mean, L2 norm, argmax.

```c
// vecstat.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
float   universe_ml_sum(const float *, int64_t);
float   universe_ml_mean(const float *, int64_t);
float   universe_ml_l2_norm(const float *, int64_t);
int64_t universe_ml_argmax(const float *, int64_t);

int main(void) {
    static float v[1 << 20];
    int64_t n = 0;
    while (n < (1 << 20) && scanf("%f", &v[n]) == 1) n++;
    if (!n) return 1;
    printf("n=%lld sum=%g mean=%g l2=%g argmax=%lld\n",
           (long long)n, universe_ml_sum(v, n), universe_ml_mean(v, n),
           universe_ml_l2_norm(v, n), (long long)universe_ml_argmax(v, n));
    return 0;
}
```

```
clang -O3 vecstat.c build/libuniverse.a -lpthread -lm -o vecstat
printf '1 2 3 4 5\n' | ./vecstat
```

## Notes

- **dtype/layout:** f32, row-major, unit stride, contiguous. `gemv` treats `A`
  as `m×n` row-major with `lda = n`.
- **SIMD-first:** the default lowers to SSE2/NEON with no runtime check; the
  `_scalar` twin is both fallback and cross-check oracle.
- **FP:** reductions use `fast` (reassociation), so results may differ in the
  last ULP from a strictly-sequential sum — this is intentional.
- **No allocation, no state:** thread-safe on disjoint buffers.
