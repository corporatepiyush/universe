# universe_dataframe (select)

## Purpose

The row-selection engine: filter / take / sample / drop_nulls / fill_null /
unique / is_unique / is_duplicated. `filter` makes one pass building a compacted
`i64` index list of kept rows (mask true AND valid), then a per-column index
gather (a null mask entry counts as false, Polars). `take` validates every index
in `[0,height)` then gathers (any OOB ⇒ NULL return). `sample_n` uses a
splitmix64 PRNG seeded by `seed` (bit-deterministic): with replacement
`idx = rng % height`, without replacement a partial Fisher-Yates. `unique` uses a
hash of the subset columns. New frames are produced by gather (columns cloned).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_filter` | `void *universe_dataframe_filter(void *df, void *mask)` | Keep rows where the BOOL `mask` Series is true. | DataFrame* / NULL |
| `universe_dataframe_take` | `void *universe_dataframe_take(void *df, const int64_t *idx, int64_t n)` | Gather `n` rows by index. | DataFrame* / NULL |
| `universe_dataframe_sample_n` | `void *universe_dataframe_sample_n(void *df, int64_t n, int32_t with_repl, int64_t seed)` | Random `n`-row sample. | DataFrame* / NULL |
| `universe_dataframe_drop_nulls` | `void *universe_dataframe_drop_nulls(void *df, const void *subset, int64_t sn)` | Drop rows with a null in `subset` (all cols if `sn==0`). | DataFrame* / NULL |
| `universe_dataframe_fill_null` | `int32_t universe_dataframe_fill_null(void *df, const void *name, int64_t nl, void *value)` | Fill a column's nulls in place. | 0 / errcode |
| `universe_dataframe_unique` | `void *universe_dataframe_unique(void *df, const void *subset, int64_t sn, int32_t keep_first)` | Deduplicate rows by `subset`. | DataFrame* / NULL |
| `universe_dataframe_unique_stable` | `void *universe_dataframe_unique_stable(void *df, const void *subset, int64_t sn)` | Order-preserving dedup. | DataFrame* / NULL |
| `universe_dataframe_is_unique` / `_is_duplicated` | `int32_t universe_dataframe_is_unique(void *df, void **out_mask)` | BOOL mask of unique / duplicated rows. | 0 / errcode |

`mask` must be a BOOL Series of length `height` (e.g. from `compare`). `idx` is a
caller `int64_t[n]`. `subset` is an array of column-name descriptors (or NULL /
`sn==0` for all columns).

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_filter(ptr, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Build a frame, filter rows where the column exceeds a threshold, print the kept
row count.

```c
// filtercli.c — build I32 column, keep rows > threshold, print survivor count
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
void   *universe_dataframe_new(void);
void   *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_with_column(void*, const void*, int64_t, void*);
void   *universe_dataframe_column(void*, const void*, int64_t);
void   *universe_dataframe_gt_scalar(void*, int32_t, int64_t, double);
void   *universe_dataframe_filter(void*, void*);
int64_t universe_dataframe_height(void*);
void    universe_dataframe_series_free(void*);
void    universe_dataframe_free(void*);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s threshold v1 v2 ...\n", argv[0]); return 2; }
    int64_t t = atoll(argv[1]), n = argc - 2;
    int32_t *v = malloc(sizeof(int32_t) * n);
    for (int64_t i = 0; i < n; i++) v[i] = atoi(argv[i + 2]);
    void *df = universe_dataframe_new();
    universe_dataframe_with_column(df, "x", strlen("x"), universe_dataframe_series_from(0, v, n));
    void *col  = universe_dataframe_column(df, "x", strlen("x"));   /* borrowed */
    void *mask = universe_dataframe_gt_scalar(col, /*I32*/0, t, 0.0);
    void *out  = universe_dataframe_filter(df, mask);
    printf("%lld rows kept\n", (long long)universe_dataframe_height(out));
    universe_dataframe_series_free(mask);
    universe_dataframe_free(out); universe_dataframe_free(df); free(v);
    return 0;
}
```

```
clang -O3 filtercli.c build/libuniverse.a -lpthread -lm -o filtercli
./filtercli 4 1 5 3 9 2 7     # 3 rows kept
```

## Notes

- `filter`/`take`/`sample`/`drop_nulls`/`unique` return a NEW owned frame (columns
  cloned) — free it with `universe_dataframe_free`. `fill_null` mutates in place.
- `sample_n` is deterministic per `seed`; without replacement `n>height` ⇒ NULL.
- No internal threading; libc malloc for the index list + gathered columns.
