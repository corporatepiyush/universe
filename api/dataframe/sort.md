# universe_dataframe (sort)

## Purpose

DataFrame ordering: `arg_sort` / `sort` / `sort_in_place` / `top_k` / `bottom_k`.
It NEVER moves row bytes — it computes an index permutation from the key
column(s), then applies it to every column with a per-dtype gather (validity
permuted too). A single numeric key with no nulls uses LSD radix on a sortable-
u64 transform (stable, 8 passes; sign bit flipped so signed order == unsigned;
`descending` = bitwise-NOT of the key, preserving stability) — measured ~20× over
comparison sorts for integer keys. Everything else (multi-key, float, bool, str,
or any nulls) uses a stable bottom-up merge sort with a multi-key comparator.
Ordering: NULL and NaN sort GREATER than any value (ascending ⇒ last); strings
compare lexicographically by bytes.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_arg_sort` | `int32_t universe_dataframe_arg_sort(void *series, int32_t descending, int64_t *out_idx)` | Write the sort permutation of one Series. | 0 / errcode |
| `universe_dataframe_sort` | `void *universe_dataframe_sort(void *df, const void *by, int64_t nkeys, const int32_t *desc)` | New frame sorted by `nkeys` key columns. | DataFrame* / NULL |
| `universe_dataframe_sort_in_place` | `int32_t universe_dataframe_sort_in_place(void *df, const void *by, int64_t nkeys, const int32_t *desc)` | Sort `df` in place. | 0 / errcode |
| `universe_dataframe_top_k` | `void *universe_dataframe_top_k(void *df, int64_t k, const void *by, int64_t nkeys, int32_t descending)` | Top `k` rows by keys. | DataFrame* / NULL |
| `universe_dataframe_bottom_k` | `void *universe_dataframe_bottom_k(void *df, int64_t k, const void *by, int64_t nkeys)` | Bottom `k` rows by keys. | DataFrame* / NULL |

`out_idx` is a caller `int64_t[height]`. `by` is an array of `nkeys` column-name
descriptors; `desc` is an `int32_t[nkeys]` per-key descending flag. `reverse` is
provided by frame.ll (`universe_dataframe_reverse`), not redefined here.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_dataframe_arg_sort(ptr, i32, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Sort argv ints ascending via `arg_sort` and print them in order.

```c
// sortcli.c — arg_sort an I32 column, print values in sorted order
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void   *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_arg_sort(void*, int32_t, int64_t*);
void    universe_dataframe_series_free(void*);
int main(int argc, char **argv) {
    int64_t n = argc - 1;
    int32_t *v = malloc(sizeof(int32_t) * (n ? n : 1));
    int64_t *idx = malloc(sizeof(int64_t) * (n ? n : 1));
    for (int64_t i = 0; i < n; i++) v[i] = atoi(argv[i + 1]);
    void *s = universe_dataframe_series_from(/*I32*/0, v, n);
    universe_dataframe_arg_sort(s, /*ascending*/0, idx);
    for (int64_t i = 0; i < n; i++) printf("%d ", v[idx[i]]);
    putchar('\n');
    universe_dataframe_series_free(s); free(v); free(idx);
    return 0;
}
```

```
clang -O3 sortcli.c build/libuniverse.a -lpthread -lm -o sortcli
./sortcli 5 3 9 1 7        # 1 3 5 7 9
```

## Notes

- Sorting is permutation-based and stable (radix for a single clean numeric key,
  stable merge otherwise). `sort`/`top_k`/`bottom_k` return NEW owned frames;
  `sort_in_place` mutates.
- NULL and NaN order last under ascending, first under descending.
- No internal threading; scratch permutation/gather buffers via libc malloc.
