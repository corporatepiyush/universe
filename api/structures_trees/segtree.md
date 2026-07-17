# structures_trees/segtree

## Purpose

Segment tree for range-sum with point-**set** update over i64, iterative and
bottom-up (no recursion). One allocation (64-byte header + a flat i64 tree of
exactly 2n slots): leaves at `tree[n..2n)`, internal node `i` covers
`tree[2i] + tree[2i+1]`. This arbitrary-n layout needs no rounding to a power of
two, using half the memory of a `2*npow2` tree while keeping the branch-free
`>>1` climb; because sum is commutative/associative the inward walk is
order-independent for every n. `point_update` overwrites a leaf then re-sums
ancestors; `range_sum([lo,hi))` walks inward in O(log n). Choose this when you
need range aggregates with point-overwrite; use `fenwick` for the lighter
prefix-sum-with-delta case.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_segtree_create(int64_t n)` | Create for `n` zeroed leaves | handle, or NULL on OOM |
| `void *universe_ds_segtree_create_from(void *vals, int64_t n)` | Create from `n` initial `int64_t` values | handle, or NULL on OOM |
| `int32_t universe_ds_segtree_point_update(void *s, int64_t index, int64_t value)` | Set leaf `index` to `value` | 0 OK, 1 NULL, 7 INVALID_INDEX |
| `int64_t universe_ds_segtree_range_sum(void *s, int64_t lo, int64_t hi)` | Sum over `[lo, hi)` | sum |
| `int64_t universe_ds_segtree_size(void *s)` | Leaf count `n` | count |
| `void universe_ds_segtree_destroy(void *s)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_segtree_create_from(ptr, i64)
declare i32 @universe_ds_segtree_point_update(ptr, i64, i64)
declare i64 @universe_ds_segtree_range_sum(ptr, i64, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Load an array, then answer `[lo,hi)` range-sum queries.

```c
// segcli.c — build: clang -O3 segcli.c build/libuniverse.a -lpthread -lm -o segcli
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void *universe_ds_segtree_create_from(void *, int64_t);
extern int64_t universe_ds_segtree_range_sum(void *, int64_t, int64_t);
extern void universe_ds_segtree_destroy(void *);

int main(int argc, char **argv) {
  int64_t vals[1024]; int64_t n = 0; long long x;
  while (n < 1024 && scanf("%lld", &x) == 1) vals[n++] = x;
  void *s = universe_ds_segtree_create_from(vals, n);
  long long lo = argc > 1 ? atoll(argv[1]) : 0, hi = argc > 2 ? atoll(argv[2]) : n;
  printf("sum[%lld,%lld)=%lld\n", lo, hi, (long long)universe_ds_segtree_range_sum(s, lo, hi));
  universe_ds_segtree_destroy(s);
  return 0;
}
```

```sh
printf '1 2 3 4 5\n' | ./segcli 1 4   # -> sum[1,4)=9
```

## Notes

- `n` is fixed at create; updates overwrite (not add). `range_sum` uses a
  half-open `[lo, hi)` interval.
- Single-threaded.
