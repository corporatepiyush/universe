# structures_trees/skiplist

## Purpose

Indexable probabilistic ordered map, i64 key -> i64 value, single thread. A skip
list gives O(log n) search/insert/delete with no rotations or rebalancing — each
mutation touches only the O(log n) nodes on the search path — plus **rank** and
**select** (order-statistics) from per-link span counts. All nodes live in one
flat growable array and reference forward neighbours by i32 index (half the
footprint of pointers, one allocation, realloc-safe). Node 0 is the head
sentinel. Choose this for an ordered map when you want rank/select and simple
mutation; `btree` is an alternative ordered map with higher fan-out.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_skiplist_create(void)` | Create an empty list | handle, or NULL on OOM |
| `int32_t universe_ds_skiplist_put(void *h, int64_t key, int64_t val)` | Insert/overwrite | 0 OK, 1, 2 |
| `int32_t universe_ds_skiplist_get(void *h, int64_t key, int64_t *out)` | Point lookup (`out` may be NULL) | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_skiplist_contains(void *h, int64_t key)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_skiplist_delete(void *h, int64_t key)` | Remove | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_skiplist_min(void *h, int64_t *ok, int64_t *ov)` | Smallest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_skiplist_max(void *h, int64_t *ok, int64_t *ov)` | Largest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_skiplist_floor(void *h, int64_t key, int64_t *ok, int64_t *ov)` | Largest key <= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_skiplist_ceiling(void *h, int64_t key, int64_t *ok, int64_t *ov)` | Smallest key >= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_skiplist_rank(void *h, int64_t key, int64_t *out_rank)` | 0-based index of `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_skiplist_select(void *h, int64_t idx, int64_t *ok, int64_t *ov)` | Key/value at position `idx` | 0 OK, 7 INVALID_INDEX |
| `int64_t universe_ds_skiplist_range(void *h, int64_t lo, int64_t hi, int64_t *ok, int64_t *ov, int64_t max)` | Ordered copy of `lo <= key < hi` | count copied |
| `int64_t universe_ds_skiplist_size(void *h)` | Entry count | count |
| `void universe_ds_skiplist_destroy(void *h)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_skiplist_create()
declare i32 @universe_ds_skiplist_put(ptr, i64, i64)
declare i32 @universe_ds_skiplist_select(ptr, i64, ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Insert keys, then select the k-th smallest (order statistic).

```c
// slkcli.c — build: clang -O3 slkcli.c build/libuniverse.a -lpthread -lm -o slkcli
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void *universe_ds_skiplist_create(void);
extern int32_t universe_ds_skiplist_put(void *, int64_t, int64_t);
extern int32_t universe_ds_skiplist_select(void *, int64_t, int64_t *, int64_t *);
extern void universe_ds_skiplist_destroy(void *);

int main(int argc, char **argv) {
  void *h = universe_ds_skiplist_create();
  long long x;
  while (scanf("%lld", &x) == 1) universe_ds_skiplist_put(h, x, x);
  long long k = argc > 1 ? atoll(argv[1]) : 0;
  int64_t ok, ov;
  if (universe_ds_skiplist_select(h, k, &ok, &ov) == 0) printf("select(%lld)=%lld\n", k, (long long)ok);
  else printf("select(%lld)=oob\n", k);
  universe_ds_skiplist_destroy(h);
  return 0;
}
```

```sh
printf '50 10 30 20 40\n' | ./slkcli 2   # -> select(2)=30 (3rd smallest)
```

## Notes

- Keys ordered signed i64. `rank`/`select` are 0-based order statistics.
- One flat allocation grows via realloc; handle stable. Single-threaded.
