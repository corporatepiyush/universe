# universe_dataframe (compare)

## Purpose

The vectorized predicate + boolean-algebra surface — the piece that makes
`filter` (select.ll) SIMD-predicated: the vector `icmp`/`fcmp` kernels build the
BOOL mask that filter compacts. Primary path is a portable 128-bit vector loop
per dtype (BOOL uses `<16 x i8>`); the `<N x i1>` result is `zext`'d to one 0/1
byte per value, so bitwise and/or/xor over the byte column is self-consistent and
`not` is xor-with-1. Float compares are ORDERED (any NaN operand ⇒ false for
every predicate); int compares are SIGNED. A scalar tail is the fallback and test
oracle. Comparison result validity is combined from the inputs; the comparison
computes densely over null lanes (garbage masked out by validity).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_eq` / `_neq` / `_lt` / `_lte` / `_gt` / `_gte` | `void *universe_dataframe_eq(void *a, void *b)` | Element-wise predicate of two Series → BOOL Series. | Series* / NULL |
| `universe_dataframe_eq_scalar` / `_neq_scalar` / `_lt_scalar` / `_lte_scalar` / `_gt_scalar` / `_gte_scalar` | `void *universe_dataframe_gt_scalar(void *a, int32_t stype, int64_t ival, double fval)` | Predicate against a broadcast scalar → BOOL Series. | Series* / NULL |
| `universe_dataframe_and` / `_or` / `_xor` | `void *universe_dataframe_and(void *a, void *b)` | Boolean algebra on two BOOL Series. | Series* / NULL |
| `universe_dataframe_not` | `void *universe_dataframe_not(void *a)` | Logical NOT of a BOOL Series. | Series* / NULL |
| `universe_dataframe_any` / `_all` | `int32_t universe_dataframe_any(void *s, int32_t *out_bool)` | Reduce a BOOL Series to a scalar. | 0 / errcode |
| `universe_dataframe_sum_bool` | `int32_t universe_dataframe_sum_bool(void *s, int64_t *out_i64)` | Count set bits (true count). | 0 / errcode |
| `universe_dataframe_zip_with` | `void *universe_dataframe_zip_with(void *mask, void *a, void *b)` | Element select: `mask ? a : b`. | Series* / NULL |

`stype`/`ival`/`fval`: same scalar-typing convention as arith (DType enum picks
the live operand).

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_gt_scalar(ptr, i32, i64, double)
declare i32 @universe_dataframe_sum_bool(ptr, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Count how many argv ints exceed a threshold.

```c
// countgt.c — count values > threshold
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void *universe_dataframe_series_from(int32_t, const void*, int64_t);
void *universe_dataframe_gt_scalar(void*, int32_t, int64_t, double);
int32_t universe_dataframe_sum_bool(void*, int64_t*);
void  universe_dataframe_series_free(void*);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s threshold v1 v2 ...\n", argv[0]); return 2; }
    int64_t t = atoll(argv[1]), n = argc - 2;
    int32_t *v = malloc(sizeof(int32_t) * n);
    for (int64_t i = 0; i < n; i++) v[i] = atoi(argv[i + 2]);
    void *s = universe_dataframe_series_from(/*I32*/0, v, n);
    void *mask = universe_dataframe_gt_scalar(s, /*I32*/0, t, 0.0);
    int64_t cnt = 0;
    universe_dataframe_sum_bool(mask, &cnt);
    printf("%lld\n", (long long)cnt);
    universe_dataframe_series_free(s); universe_dataframe_series_free(mask); free(v);
    return 0;
}
```

```
clang -O3 countgt.c build/libuniverse.a -lpthread -lm -o countgt
./countgt 5 1 6 3 9 5 8     # 3
```

## Notes

- Predicates return a BOOL Series (one 0/1 byte per value) — feed it to
  `universe_dataframe_filter` to select rows. A null input entry yields a null
  mask entry, which `filter` treats as false (Polars semantics).
- Float predicates are ordered: NaN compares false everywhere.
- Every result is a NEW owned Series; no internal threading; one output buffer
  per call (libc malloc).
