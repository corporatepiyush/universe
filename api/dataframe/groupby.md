# universe_dataframe (groupby)

## Purpose

DataFrame grouping + aggregation. Each row's key columns are hashed with an
inline FNV-1a mix into an open-addressing map (power-of-two capacity, mask wrap,
linear probe, ≤70% load) whose slots hold a group id; collisions resolve by
comparing to the group's representative (first) row. Group ids are handed out in
row order, so group order == first-appearance order (`group_by` and
`group_by_stable` share the same core). Per-group member lists are built as CSR
(`group_off[G+1]` + `members[height]`), and each aggregate is a reduce over a
group's contiguous member slice. The hot leaf (row hash + probe + equality) is
monomorphic and fully inlined; key column pointers are resolved once before the
row loop.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_group_by` | `void *universe_dataframe_group_by(void *df, const void *key_names, int64_t nk)` | Build a GroupBy over `nk` key columns (BORROWS `df`). | GroupBy* / NULL |
| `universe_dataframe_group_by_stable` | `void *universe_dataframe_group_by_stable(void *df, const void *key_names, int64_t nk)` | Same, first-appearance order guaranteed. | GroupBy* / NULL |
| `universe_dataframe_group_by_free` | `void universe_dataframe_group_by_free(void *gb)` | Free a GroupBy handle. | — |
| `universe_dataframe_group_by_agg` | `int32_t universe_dataframe_group_by_agg(void *gb, const int64_t *agg_cols, int64_t nc, const int32_t *agg_ops, void **out_df)` | Aggregate; write a result frame. | 0 / errcode |
| `universe_dataframe_partition_by` | `int32_t universe_dataframe_partition_by(void *df, const void *key_names, int64_t nk, void **out_frames, int64_t cap, int64_t *out_n)` | Split into per-group frames. | 0 / errcode |

`agg_cols` are SOURCE column indices (`int64_t[nc]`, NOT names). `agg_ops` are
`int32_t[nc]`: **0 sum, 1 mean, 2 min, 3 max, 4 count, 5 n_unique**. The output
frame is the key columns (one row per group) followed by one column per
`(agg_col,agg_op)`, named `"<colname>_<opsuffix>"`. Aggregate dtypes:
sum/mean/min/max → F64; count/n_unique → I64. Aggregates skip nulls; an all-null
group gives sum 0, count/n_unique 0, mean/min/max NULL. `out_frames` is a caller
array of `cap` slots; `out_n` receives the group count.

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_group_by(ptr, ptr, i64)
declare i32 @universe_dataframe_group_by_agg(ptr, ptr, i64, ptr, ptr)
declare void @universe_dataframe_group_by_free(ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Group a two-column frame (`k`, `v`) by `k`, sum `v`, and print the number of
groups. Keys/values come from argv pairs `k:v`.

```c
// groupcli.c — group by k, sum v; print group count
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef struct { const char *ptr; int64_t len; } name_t;   /* {ptr,i64} name descriptor */
void   *universe_dataframe_new(void);
void   *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_with_column(void*, const void*, int64_t, void*);
void   *universe_dataframe_group_by(void*, const void*, int64_t);
int32_t universe_dataframe_group_by_agg(void*, const int64_t*, int64_t, const int32_t*, void**);
int64_t universe_dataframe_height(void*);
void    universe_dataframe_group_by_free(void*);
void    universe_dataframe_free(void*);
int main(int argc, char **argv) {
    int64_t n = argc - 1;
    int32_t *k = malloc(sizeof(int32_t)*n), *v = malloc(sizeof(int32_t)*n);
    for (int64_t i = 0; i < n; i++) { char *c = strchr(argv[i+1], ':'); *c = 0;
        k[i] = atoi(argv[i+1]); v[i] = atoi(c + 1); }
    void *df = universe_dataframe_new();
    universe_dataframe_with_column(df, "k", 1, universe_dataframe_series_from(0, k, n));
    universe_dataframe_with_column(df, "v", 1, universe_dataframe_series_from(0, v, n));
    name_t keys[1] = {{"k", 1}};
    void *gb = universe_dataframe_group_by(df, keys, 1);
    int64_t cols[1] = {1};        /* aggregate source column index 1 ("v") */
    int32_t ops[1]  = {0};        /* 0 = sum */
    void *out = NULL;
    universe_dataframe_group_by_agg(gb, cols, 1, ops, &out);
    printf("%lld groups\n", (long long)universe_dataframe_height(out));
    universe_dataframe_free(out); universe_dataframe_group_by_free(gb);
    universe_dataframe_free(df); free(k); free(v);
    return 0;
}
```

```
clang -O3 groupcli.c build/libuniverse.a -lpthread -lm -o groupcli
./groupcli 1:10 2:20 1:30 2:5 3:7     # 3 groups
```

## Notes

- The GroupBy handle BORROWS the source frame — keep `df` alive until you free
  the GroupBy. Free the GroupBy with `group_by_free` and the agg result with
  `dataframe_free`.
- Group order is first-appearance; aggregates skip nulls.
- No internal threading; the hash map / CSR arrays are libc malloc.
