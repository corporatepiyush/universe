# universe_dataframe (reduce)

## Purpose

Column reductions (Polars method names). Integer/float columns reduce to an f64
result; nulls are skipped via the validity bitmap. The dense path (no validity,
fixed numeric dtype) makes ONE memory pass through a per-dtype vector kernel that
computes sum, sum-of-squares, min and max together — reductions are memory-bound,
so the extra ALU work hides under load latency and one pass minimizes traffic
for whichever reduction the caller wants. Four `<4 x double>` accumulators break
the FP dependency chain; `fast` FP on sum/sumsq enables tree reduction + fmla.
The scalar path is the null-aware fallback and the test oracle. `median`/
`quantile` extract non-null values, sort ascending (`universe_sort_quick`), then
linear-interpolate the order statistic.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_sum` | `int32_t universe_dataframe_sum(void *s, double *out)` | Sum (nulls skipped). | 0 / errcode |
| `universe_dataframe_mean` | `int32_t universe_dataframe_mean(void *s, double *out)` | Mean. | 0 / errcode |
| `universe_dataframe_min` / `_max` | `int32_t universe_dataframe_min(void *s, double *out)` | Min / max. | 0 / errcode |
| `universe_dataframe_var` / `_std` | `int32_t universe_dataframe_var(void *s, double *out)` | Variance / std-dev. | 0 / errcode |
| `universe_dataframe_null_count_series` | `int32_t universe_dataframe_null_count_series(void *s, int64_t *out)` | Null count. | 0 / errcode |
| `universe_dataframe_quantile` | `int32_t universe_dataframe_quantile(void *s, double q, double *out)` | Interpolated quantile `q∈[0,1]`. | 0 / errcode |
| `universe_dataframe_median` | `int32_t universe_dataframe_median(void *s, double *out)` | Median. | 0 / errcode |
| `universe_dataframe_n_unique` | `int32_t universe_dataframe_n_unique(void *s, int64_t *out)` | Distinct-value count. | 0 / errcode |
| `universe_dataframe_sum_horizontal` / `_mean_horizontal` | `void *universe_dataframe_sum_horizontal(void *df)` | Row-wise sum/mean across columns → Series. | Series* / NULL |

Scalar results are written through `out` (`double*` or `int64_t*`); the return is
a status code. Horizontal reductions return a new Series.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_dataframe_mean(ptr, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Mean of argv floats.

```c
// meancli.c — mean of stdin/argv doubles
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_mean(void*, double*);
void  universe_dataframe_series_free(void*);
int main(int argc, char **argv) {
    int64_t n = argc - 1;
    double *v = malloc(sizeof(double) * (n ? n : 1));
    for (int64_t i = 0; i < n; i++) v[i] = atof(argv[i + 1]);
    void *s = universe_dataframe_series_from(/*F64*/3, v, n);
    double m = 0.0;
    universe_dataframe_mean(s, &m);
    printf("%.6f\n", m);
    universe_dataframe_series_free(s); free(v);
    return 0;
}
```

```
clang -O3 meancli.c build/libuniverse.a -lpthread -lm -o meancli
./meancli 1 2 3 4        # 2.500000
```

## Notes

- Reductions ignore nulls; an all-null (or empty) column returns the documented
  empty result (e.g. sum 0). Results are f64 regardless of input dtype.
- `median`/`quantile` allocate an f64 scratch buffer and sort it (cold path).
- No internal threading; scalar reductions write through a caller out-pointer.
