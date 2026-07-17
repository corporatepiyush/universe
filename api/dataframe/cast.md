# universe_dataframe (cast)

## Purpose

SIMD dtype conversion plus null-aware fill/clip/null-mask (the numpy `astype` +
Polars `fill_null`/`clip` analog). `cast` is a portable `<4 x T>` convert loop
per (from,to) pair — one vector instruction per body (sext/trunc, sitofp,
fpext/fptrunc, and saturating `llvm.fptosi.sat` for float→int so out-of-range
floats clamp to `INT_MIN/MAX` instead of poison) — dispatched by a one-time cold
switch; validity and null count carry over verbatim. `fill_null_value` replaces
null lanes with a broadcast fill via a vector `select` (result has no nulls).
`clip` clamps to `[lo,hi]` with `llvm.smin/smax` (int) or `llvm.minnum/maxnum`
(float). The null-mask helpers expand one validity byte to 8 BOOL bytes.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_cast` | `void *universe_dataframe_cast(void *a, int32_t to)` | Convert a Series to DType `to`. | Series* / NULL |
| `universe_dataframe_fill_null_value` | `void *universe_dataframe_fill_null_value(void *a, int32_t stype, int64_t ival, double fval)` | Replace nulls with a scalar (result has no nulls). | Series* / NULL |
| `universe_dataframe_clip` | `void *universe_dataframe_clip(void *a, double lo, double hi)` | Clamp values to `[lo,hi]`. | Series* / NULL |
| `universe_dataframe_is_null_mask` | `void *universe_dataframe_is_null_mask(void *a)` | BOOL Series, 1 where null. | Series* / NULL |
| `universe_dataframe_is_not_null_mask` | `void *universe_dataframe_is_not_null_mask(void *a)` | BOOL Series, 1 where valid. | Series* / NULL |

`to` and `stype` use the DType enum (`I32=0,I64=1,F32=2,F64=3,BOOL=4,STR=5`).
`clip` bounds are doubles applied per the column's dtype; NaN clamps to a bound
(minnum/maxnum drop NaN).

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_cast(ptr, i32)
declare ptr @universe_dataframe_clip(ptr, double, double)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Cast an I32 column to F64 and print it.

```c
// castcli.c — I32 -> F64
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void *universe_dataframe_series_from(int32_t, const void*, int64_t);
void *universe_dataframe_cast(void*, int32_t);
void *universe_dataframe_series_values(void*);
int64_t universe_dataframe_series_len(void*);
void  universe_dataframe_series_free(void*);
int main(int argc, char **argv) {
    int64_t n = argc - 1;
    int32_t *v = malloc(sizeof(int32_t) * (n ? n : 1));
    for (int64_t i = 0; i < n; i++) v[i] = atoi(argv[i + 1]);
    void *s = universe_dataframe_series_from(/*I32*/0, v, n);
    void *r = universe_dataframe_cast(s, /*F64*/3);
    double *out = universe_dataframe_series_values(r);
    for (int64_t i = 0; i < universe_dataframe_series_len(r); i++) printf("%.1f ", out[i]);
    putchar('\n');
    universe_dataframe_series_free(s); universe_dataframe_series_free(r); free(v);
    return 0;
}
```

```
clang -O3 castcli.c build/libuniverse.a -lpthread -lm -o castcli
./castcli 1 2 3        # 1.0 2.0 3.0
```

## Notes

- A value cast never adds or removes nulls (validity + null count preserved);
  `fill_null_value` produces a null-free result. `float→int` casts saturate.
- Every result is a NEW owned Series; no internal threading; libc malloc.
