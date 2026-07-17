# universe_dataframe (join)

## Purpose

Relational hash join (inner / left / outer / cross). Build a multimap over the
RIGHT key columns as key-hash → chain of right-row indices, realized with two
flat index arrays (`head[nbuckets]`, `next[rlen]`) — no pointers, no tombstones,
`nbuckets = next_pow2(2*rlen)`, `bucket = hash & (nbuckets-1)`. Probe with each
LEFT row, walking its bucket chain and key-equality-checking candidates, and emit
matched `(left_row, right_row)` index pairs; output columns are built by gather
over those index arrays (a `-1` index gathers a NULL). NULL keys never match
(Polars default). Output columns: all LEFT columns, then RIGHT NON-KEY columns; a
right column whose name collides with a left column gets a `_right` suffix.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_dataframe_join` | `void *universe_dataframe_join(void *left, void *right, const void *left_on, int64_t nl, const void *right_on, int64_t nr, int32_t how)` | General join with separate left/right key names. | DataFrame* / NULL |
| `universe_dataframe_inner_join` | `void *universe_dataframe_inner_join(void *l, void *r, const void *on, int64_t n)` | Inner join on shared key names. | DataFrame* / NULL |
| `universe_dataframe_left_join` | `void *universe_dataframe_left_join(void *l, void *r, const void *on, int64_t n)` | Left join. | DataFrame* / NULL |
| `universe_dataframe_outer_join` | `void *universe_dataframe_outer_join(void *l, void *r, const void *on, int64_t n)` | Full outer join. | DataFrame* / NULL |
| `universe_dataframe_cross_join` | `void *universe_dataframe_cross_join(void *l, void *r)` | Cartesian product (keys ignored). | DataFrame* / NULL |

`how` codes: **0 inner, 1 left, 2 outer, 3 cross**. Key-name arguments (`on`,
`left_on`, `right_on`) are arrays of `{ptr,i64}` name descriptors; `nl`/`nr`/`n`
are the key counts. All variants return a NEW owned frame.

## Use in an LLVM-based environment

```llvm
declare ptr @universe_dataframe_inner_join(ptr, ptr, ptr, i64)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Inner-join two single-key frames on `id` and print the joined row count.

```c
// joincli.c — inner join two I32 "id" columns, print match count
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
typedef struct { const char *ptr; int64_t len; } name_t;
void   *universe_dataframe_new(void);
void   *universe_dataframe_series_from(int32_t, const void*, int64_t);
int32_t universe_dataframe_with_column(void*, const void*, int64_t, void*);
void   *universe_dataframe_inner_join(void*, void*, const void*, int64_t);
int64_t universe_dataframe_height(void*);
void    universe_dataframe_free(void*);
static void *mk(const int32_t *v, int64_t n) {
    void *df = universe_dataframe_new();
    universe_dataframe_with_column(df, "id", 2, universe_dataframe_series_from(0, v, n));
    return df;
}
int main(void) {
    int32_t a[] = {1, 2, 3, 4}, b[] = {2, 4, 6};
    void *l = mk(a, 4), *r = mk(b, 3);
    name_t on[1] = {{"id", 2}};
    void *j = universe_dataframe_inner_join(l, r, on, 1);
    printf("%lld matched rows\n", (long long)universe_dataframe_height(j));  /* 2 */
    universe_dataframe_free(j); universe_dataframe_free(l); universe_dataframe_free(r);
    return 0;
}
```

```
clang -O3 joincli.c build/libuniverse.a -lpthread -lm -o joincli
./joincli        # 2 matched rows
```

## Notes

- The result is a NEW owned frame; inputs are unchanged (borrowed). NULL keys
  never match; missing sides gather NULL (left/outer). Colliding right column
  names get a `_right` suffix.
- No internal threading; the hash tables + output columns are libc malloc.
