# structures_trees/btree

## Purpose

Array-backed B-tree mapping i64 keys to i64 values, single thread. Order t = 8
(up to 15 keys / 16 children per node) keeps the tree shallow (~log_8 n: 64K
keys is height <= 6). All nodes live in one flat growable array and reference
children by i32 index, not pointer — halving link footprint, keeping the whole
tree in one allocation, and letting the array double via realloc without
rewriting interior links. In-node lookup is a branch-lean binary search over a
cache-resident key run. **Choose the B-tree** for an ordered i64->i64 map when
you want high fan-out and one allocation; `skiplist` is an alternative ordered
map with rank/select, `treemap` when inserts are rare and scans dominate.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_btree_create(void)` | Create an empty tree | handle, or NULL on OOM |
| `int32_t universe_ds_btree_insert(void *t, int64_t key, int64_t val)` | Insert/update | 0 OK, 1 NULL, 2 OOM |
| `int32_t universe_ds_btree_delete(void *t, int64_t key)` | Remove | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_find(void *t, int64_t key, int64_t *out)` | Point lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_contains(void *t, int64_t key)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_btree_min(void *t, int64_t *ok, int64_t *ov)` | Smallest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_btree_max(void *t, int64_t *ok, int64_t *ov)` | Largest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_btree_floor(void *t, int64_t key, int64_t *ok, int64_t *ov)` | Largest key <= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_ceiling(void *t, int64_t key, int64_t *ok, int64_t *ov)` | Smallest key >= `key` | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_btree_range(void *t, int64_t lo, int64_t hi, int64_t *ok, int64_t *ov, int64_t max)` | Copy entries `lo <= key < hi` (up to `max`) | count copied |
| `int64_t universe_ds_btree_count(void *t)` | Entry count | count |
| `int64_t universe_ds_btree_size(void *t)` | Node/allocation size metric | value |
| `void universe_ds_btree_destroy(void *t)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_btree_create()
declare i32 @universe_ds_btree_insert(ptr, i64, i64)
declare i32 @universe_ds_btree_find(ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Insert key/value pairs, then query a range.

```c
// btcli.c — build: clang -O3 btcli.c build/libuniverse.a -lpthread -lm -o btcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_btree_create(void);
extern int32_t universe_ds_btree_insert(void *, int64_t, int64_t);
extern int64_t universe_ds_btree_range(void *, int64_t, int64_t, int64_t *, int64_t *, int64_t);
extern void universe_ds_btree_destroy(void *);

int main(int argc, char **argv) {
  void *t = universe_ds_btree_create();
  long long k;
  while (scanf("%lld", &k) == 1) universe_ds_btree_insert(t, k, k * k);
  long long lo = argc > 1 ? atoll(argv[1]) : 0, hi = argc > 2 ? atoll(argv[2]) : 100;
  int64_t ks[256], vs[256];
  int64_t n = universe_ds_btree_range(t, lo, hi, ks, vs, 256);
  for (int64_t i = 0; i < n; i++) printf("%lld->%lld\n", (long long)ks[i], (long long)vs[i]);
  universe_ds_btree_destroy(t);
  return 0;
}
```

```sh
printf '3 1 4 1 5 9\n' | ./btcli 2 6   # -> 3->9, 4->16, 5->25
```

## Notes

- Keys ordered by signed i64. `range` writes ascending into caller arrays and
  returns the count actually copied (capped at `max`).
- One allocation grows via realloc; the handle stays stable. Single-threaded —
  use `btree_sharded` for concurrency.
