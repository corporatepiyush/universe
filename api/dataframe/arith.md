# universe_dataframe (arith)

## Purpose

SIMD element-wise arithmetic over `Series` (the numpy-ufunc / Polars-kernel
analog). The value hot path is a portable 128-bit vector loop per dtype
(`<4 x i32>`, `<2 x i64>`, `<4 x float>`, `<2 x double>`) that lowers to SSE2 /
NEON with no runtime CPU check; a scalar tail finishes the remainder and is the
test oracle. Nulls follow the Polars/arrow2 trick: values compute densely over
every lane while validity bitmaps combine separately with a `<16 x i8>` AND, so
the value loop stays branch-free. Integer divide/remainder has no packed SIMD op,
so it runs a branchless scalar loop that masks divide-by-zero and `INT_MIN/-1`
and marks those lanes null (Polars semantics). All results are freshly allocated
Series.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_add` / `_sub` / `_mul` / `_div` / `_rem` | `void *universe_dataframe_add(void *a, void *b)` | Element-wise binary op of two same-length Series. | Series* / NULL |
| `universe_dataframe_neg` / `_abs` | `void *universe_dataframe_neg(void *a)` | Unary negate / absolute value. | Series* / NULL |
| `universe_dataframe_add_scalar` / `_sub_scalar` / `_mul_scalar` / `_div_scalar` | `void *universe_dataframe_add_scalar(void *a, int32_t stype, int64_t ival, double fval)` | Op with a broadcast scalar. | Series* / NULL |
| `universe_dataframe_min_horizontal` / `_max_horizontal` | `void *universe_dataframe_min_horizontal(void *df)` | Row-wise min/max across a frame's columns. | Series* / NULL |

`stype` selects which scalar operand is live using the DType enum: for integer
columns use `stype = I32/I64` with `ival`; for float columns use `F32/F64` with
`fval`. Result validity is the AND of the input validities; int div/rem by zero
yields a null lane.

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_add(ptr, ptr)
declare ptr @universe_dataframe_mul_scalar(ptr, i32, i64, double)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Multiply an I32 column (argv ints) by a scalar and print the result values.

```c
// mulcli.c — series *= k
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void *universe_dataframe_series_from(int32_t, const void*, int64_t);
void *universe_dataframe_mul_scalar(void*, int32_t, int64_t, double);
void *universe_dataframe_series_values(void*);
int64_t universe_dataframe_series_len(void*);
void  universe_dataframe_series_free(void*);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s k v1 v2 ...\n", argv[0]); return 2; }
    int64_t k = atoll(argv[1]), n = argc - 2;
    int32_t *v = malloc(sizeof(int32_t) * n);
    for (int64_t i = 0; i < n; i++) v[i] = atoi(argv[i + 2]);
    void *s = universe_dataframe_series_from(/*I32*/0, v, n);
    void *r = universe_dataframe_mul_scalar(s, /*I32*/0, k, 0.0);
    int32_t *out = universe_dataframe_series_values(r);
    for (int64_t i = 0; i < universe_dataframe_series_len(r); i++) printf("%d ", out[i]);
    putchar('\n');
    universe_dataframe_series_free(s); universe_dataframe_series_free(r); free(v);
    return 0;
}
```

```
clang -O3 mulcli.c build/libuniverse.a -lpthread -lm -o mulcli
./mulcli 3 1 2 3        # 3 6 9
```

## Notes

- Operands must share length; binary ops on mismatched dtypes follow the module's
  documented coercion (numeric columns). Every result is a NEW owned Series —
  free it (or hand it to a frame).
- Float ops propagate NaN; integer div/rem by zero → null lane.
- No internal threading; the vector loops are single-threaded and allocation is
  one output buffer per call (libc malloc).
