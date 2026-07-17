# universe_dataframe (frame)

## Purpose

The DataFrame foundation: a columnar `Series` (typed column) and a `DataFrame`
container, single-chunk, contiguous, eager, struct-of-arrays. This module builds
and mutates columns, queries shape/dtypes, reshapes (slice/head/tail/reverse/
shift/vstack/hstack/split), and owns lifetime. All other `dataframe/*` modules
code against its documented byte layout (see the domain
[README](README.md#shared-contract-framell)). One buffer alloc per non-null
array; validity is lazily allocated on the first null.

## Exported API

Selected functions (see source for the full set). `void*` is a `Series*` or
`DataFrame*` handle.

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_series_new` | `void *universe_dataframe_series_new(int32_t dtype, int64_t len)` | Allocate a Series of `len` (zeroed values). | Series* / NULL |
| `universe_dataframe_series_from` | `void *universe_dataframe_series_from(int32_t dtype, const void *values, int64_t len)` | Series copying `len` fixed-width values. | Series* / NULL |
| `universe_dataframe_series_str_new` | `void *universe_dataframe_series_str_new(const void *offsets, const void *data, int64_t len, int64_t data_len)` | STR Series from offsets + byte blob. | Series* / NULL |
| `universe_dataframe_series_free` | `void universe_dataframe_series_free(void *s)` | Free a standalone Series. | — |
| `universe_dataframe_series_len` | `int64_t universe_dataframe_series_len(void *s)` | Element count. | len |
| `universe_dataframe_series_dtype` | `int32_t universe_dataframe_series_dtype(void *s)` | DType code. | dtype |
| `universe_dataframe_series_values` | `void *universe_dataframe_series_values(void *s)` | Raw values pointer. | ptr |
| `universe_dataframe_series_validity` | `void *universe_dataframe_series_validity(void *s)` | Validity bitmap (or NULL). | ptr |
| `universe_dataframe_series_null_count` | `int64_t universe_dataframe_series_null_count(void *s)` | Number of nulls. | count |
| `universe_dataframe_series_set_null` | `int32_t universe_dataframe_series_set_null(void *s, int64_t i)` | Mark element `i` null. | 0 / errcode |
| `universe_dataframe_series_is_null` | `int32_t universe_dataframe_series_is_null(void *s, int64_t i, int32_t *out_bool)` | Test null. | 0 / errcode |
| `universe_dataframe_series_clone` | `void *universe_dataframe_series_clone(void *s)` | Deep copy. | Series* / NULL |
| `universe_dataframe_series_str_get` | `int32_t universe_dataframe_series_str_get(void *s, int64_t i, void **out_ptr, int64_t *out_len)` | View a string element. | 0 / errcode |
| `universe_dataframe_new` | `void *universe_dataframe_new(void)` | Empty DataFrame. | DataFrame* / NULL |
| `universe_dataframe_empty_with_height` | `void *universe_dataframe_empty_with_height(int64_t h)` | Empty frame with preset height. | DataFrame* / NULL |
| `universe_dataframe_height` / `_width` | `int64_t universe_dataframe_height(void *df)` / `int64_t universe_dataframe_width(void *df)` | Rows / columns. | count |
| `universe_dataframe_shape` | `int32_t universe_dataframe_shape(void *df, int64_t *out_h, int64_t *out_w)` | Both dims. | 0 / errcode |
| `universe_dataframe_column` | `void *universe_dataframe_column(void *df, const void *name, int64_t name_len)` | Borrow a column by name. | Series* / NULL |
| `universe_dataframe_select_at_idx` | `void *universe_dataframe_select_at_idx(void *df, int64_t i)` | Column by position (clone). | Series* / NULL |
| `universe_dataframe_with_column` | `int32_t universe_dataframe_with_column(void *df, const void *name, int64_t name_len, void *series)` | Append a column (takes ownership). | 0 / errcode |
| `universe_dataframe_replace_column` | `int32_t universe_dataframe_replace_column(void *df, int64_t i, void *series)` | Replace column `i`. | 0 / errcode |
| `universe_dataframe_rename` | `int32_t universe_dataframe_rename(void *df, const void *old, int64_t ol, const void *new, int64_t nl)` | Rename a column. | 0 / errcode |
| `universe_dataframe_drop` | `void *universe_dataframe_drop(void *df, const void *name, int64_t name_len)` | New frame without a column. | DataFrame* / NULL |
| `universe_dataframe_hstack` | `int32_t universe_dataframe_hstack(void *df, void *series_arr, void *names, int64_t n)` | Append `n` columns (takes ownership). | 0 / errcode |
| `universe_dataframe_vstack` | `void *universe_dataframe_vstack(void *a, void *b)` | Row-concatenate two frames. | DataFrame* / NULL |
| `universe_dataframe_slice` / `_head` / `_tail` | `void *universe_dataframe_slice(void *df, int64_t offset, int64_t len)` | Row window (clones). | DataFrame* / NULL |
| `universe_dataframe_reverse` / `_shift` | `void *universe_dataframe_reverse(void *df)` / `void *universe_dataframe_shift(void *df, int64_t periods)` | Reverse / shift rows. | DataFrame* / NULL |
| `universe_dataframe_with_row_index` | `void *universe_dataframe_with_row_index(void *df, const void *name, int64_t nl, int64_t offset)` | Prepend an index column. | DataFrame* / NULL |
| `universe_dataframe_equals` | `int32_t universe_dataframe_equals(void *a, void *b, int32_t *out_bool)` | Structural equality. | 0 / errcode |
| `universe_dataframe_free` | `void universe_dataframe_free(void *df)` | Free frame + owned columns + names. | — |

Also exported: `series_dtype`, `dtypes`, `get_column_names`,
`get_column_index`, `select`, `insert_column`, `drop_in_place`, `drop_many`,
`extend`, `split_at`, `null_count`, `clear`. See `src/dataframe/frame.ll`.

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_new()
declare ptr @universe_dataframe_series_from(i32, ptr, i64)
declare i32 @universe_dataframe_with_column(ptr, ptr, i64, ptr)
declare i64 @universe_dataframe_height(ptr)
declare void @universe_dataframe_free(ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Build a one-column I32 frame from argv integers and print its shape.

```c
// framecli.c — build an I32 column from argv ints, print rows x cols
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
void   *universe_dataframe_new(void);
void   *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_with_column(void*, const void*, int64_t, void*);
int64_t universe_dataframe_height(void*);
int64_t universe_dataframe_width(void*);
void    universe_dataframe_free(void*);
int main(int argc, char **argv) {
    int64_t n = argc - 1;
    int32_t *vals = malloc(sizeof(int32_t) * (n ? n : 1));
    for (int64_t i = 0; i < n; i++) vals[i] = atoi(argv[i + 1]);
    void *s = universe_dataframe_series_from(/*I32*/0, vals, n);
    void *df = universe_dataframe_new();
    universe_dataframe_with_column(df, "x", strlen("x"), s);  /* df now owns s */
    printf("%lld rows x %lld cols\n",
           (long long)universe_dataframe_height(df),
           (long long)universe_dataframe_width(df));
    universe_dataframe_free(df);
    free(vals);
    return 0;
}
```

```
clang -O3 framecli.c build/libuniverse.a -lpthread -lm -o framecli
./framecli 10 20 30 40      # 4 rows x 1 cols
```

## Notes

- **Ownership:** `with_column`/`hstack`/`insert_column`/`replace_column` take
  ownership of the passed Series — do not free it yourself; free the frame.
  `column` BORROWS (do not free the returned pointer); `select_at_idx`/`clone`
  return an owned copy.
- **Layout is fixed and portable** (byte offsets above); every sibling module
  reads it directly.
- **Thread-safety:** no internal locking — a frame is single-owner; concurrent
  mutation is the caller's responsibility. Read-only sharing across threads is
  fine once construction is complete.
- **Allocator:** libc `malloc`/`free`; one buffer per column, validity allocated
  lazily on first null.
