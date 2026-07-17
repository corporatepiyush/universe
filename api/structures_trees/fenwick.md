# structures_trees/fenwick

## Purpose

Fenwick tree (binary indexed tree) of i64 partial sums: O(log n) point update
and prefix query, O(1) size, contiguous cache-friendly storage. One allocation
(64-byte header + a flat i64[n+1] tree, 1-based internally); calloc gives a
zeroed tree so create is O(1) touch-free. The low-bit stride `i & -i` is a
single neg+and with no data-dependent branch in the loop body. Value queries
return i64 with no error channel (defensive: null handle -> 0, indices clamped
into `[0, n]`); only `update` reports errors. **Choose Fenwick** for cumulative
sums with point updates; for range-sum with point-set and more general monoids
see `segtree`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_fenwick_create(int64_t n)` | Create for `n` slots (all zero) | handle, or NULL on OOM |
| `int32_t universe_ds_fenwick_update(void *f, int64_t index, int64_t delta)` | Add `delta` at `index` | 0 OK, 1 NULL, 7 INVALID_INDEX |
| `int64_t universe_ds_fenwick_prefix_sum(void *f, int64_t count)` | Sum of the first `count` elements | sum (clamped) |
| `int64_t universe_ds_fenwick_range_sum(void *f, int64_t lo, int64_t hi)` | Sum over `[lo, hi)` | sum (clamped) |
| `int64_t universe_ds_fenwick_point_get(void *f, int64_t index)` | Current value at `index` | value |
| `int64_t universe_ds_fenwick_size(void *f)` | Slot count `n` | count |
| `void universe_ds_fenwick_destroy(void *f)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_fenwick_create(i64)
declare i32 @universe_ds_fenwick_update(ptr, i64, i64)
declare i64 @universe_ds_fenwick_prefix_sum(ptr, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

`add I D` adds a delta, `sum C` prints the prefix sum of the first C elements.

```c
// fwcli.c — build: clang -O3 fwcli.c build/libuniverse.a -lpthread -lm -o fwcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_fenwick_create(int64_t);
extern int32_t universe_ds_fenwick_update(void *, int64_t, int64_t);
extern int64_t universe_ds_fenwick_prefix_sum(void *, int64_t);
extern void universe_ds_fenwick_destroy(void *);

int main(void) {
  void *f = universe_ds_fenwick_create(1024);
  char op[8]; long long a, b;
  while (scanf("%7s %lld", op, &a) == 2) {
    if (!strcmp(op, "add")) { scanf("%lld", &b); universe_ds_fenwick_update(f, a, b); }
    else if (!strcmp(op, "sum")) printf("%lld\n", (long long)universe_ds_fenwick_prefix_sum(f, a));
  }
  universe_ds_fenwick_destroy(f);
  return 0;
}
```

```sh
printf 'add 0 5\nadd 2 3\nsum 3\n' | ./fwcli   # -> 8
```

## Notes

- `n` is fixed at create; `update` indices must be `< n`.
- Query functions are defensive (no error code): out-of-range counts are clamped,
  a null handle returns 0.
- Single-threaded.
