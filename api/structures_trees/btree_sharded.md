# structures_trees/btree_sharded

## Purpose

Range-sharded concurrent B-tree. N power-of-two independent shards, each a full
array-backed B-tree (reused directly from `btree`) guarded by its own TTAS
spinlock on its own 128 B cache-line pair. Shards are partitioned by the **high
bits** of an order-preserving unsigned mapping of the key (`ukey = key XOR
0x8000000000000000`), so the partition is monotone and range/floor/ceiling/
min/max stay globally ordered across shards. Point ops lock exactly one shard,
so threads on different key ranges proceed fully in parallel. **Choose this** for
a concurrent ordered i64 map where writers spread across the key space; for
concurrency without ordering, hash-sharded maps balance load better under skew.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_btree_sharded_create(int64_t nshards)` | Create with `nshards` (rounded to pow2) shards | handle, or NULL on OOM |
| `int32_t universe_ds_btree_sharded_put(void *m, int64_t key, int64_t val)` | Insert/update (locks one shard) | 0 OK, 1 NULL, 2 OOM |
| `int32_t universe_ds_btree_sharded_get(void *m, int64_t key, int64_t *out)` | Point lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_sharded_delete(void *m, int64_t key)` | Remove | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_sharded_contains(void *m, int64_t key)` | Membership | 1 present, 0 absent |
| `int64_t universe_ds_btree_sharded_len(void *m)` | Total entry count | count |
| `int64_t universe_ds_btree_sharded_shards(void *m)` | Shard count | count |
| `int32_t universe_ds_btree_sharded_min(void *m, int64_t *ok, int64_t *ov)` | Global minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_btree_sharded_max(void *m, int64_t *ok, int64_t *ov)` | Global maximum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_btree_sharded_floor(void *m, int64_t key, int64_t *ok, int64_t *ov)` | Largest key <= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_btree_sharded_ceiling(void *m, int64_t key, int64_t *ok, int64_t *ov)` | Smallest key >= `key` | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_btree_sharded_range(void *m, int64_t lo, int64_t hi, int64_t *ok, int64_t *ov, int64_t max)` | Ordered copy of `lo <= key < hi` | count copied |
| `void universe_ds_btree_sharded_destroy(void *m)` | Free all shards | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_btree_sharded_create(i64)
declare i32 @universe_ds_btree_sharded_put(ptr, i64, i64)
declare i32 @universe_ds_btree_sharded_get(ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Insert concurrently, then read the global ordered extremes.

```c
// btscli.c — build: clang -O3 btscli.c build/libuniverse.a -lpthread -lm -o btscli
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
extern void *universe_ds_btree_sharded_create(int64_t);
extern int32_t universe_ds_btree_sharded_put(void *, int64_t, int64_t);
extern int32_t universe_ds_btree_sharded_min(void *, int64_t *, int64_t *);
extern int32_t universe_ds_btree_sharded_max(void *, int64_t *, int64_t *);
extern int64_t universe_ds_btree_sharded_len(void *);
extern void universe_ds_btree_sharded_destroy(void *);

static void *worker(void *arg) {
  void *m = ((void **)arg)[0]; long base = (long)((void **)arg)[1];
  for (long i = 0; i < 1000; i++) universe_ds_btree_sharded_put(m, base + i, i);
  return 0;
}
int main(void) {
  void *m = universe_ds_btree_sharded_create(16);
  pthread_t t[4]; void *a[4][2];
  for (int i = 0; i < 4; i++) { a[i][0] = m; a[i][1] = (void *)(long)(i * 1000000);
    pthread_create(&t[i], 0, worker, a[i]); }
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  int64_t k, v;
  universe_ds_btree_sharded_min(m, &k, &v); printf("min=%lld\n", (long long)k);
  universe_ds_btree_sharded_max(m, &k, &v); printf("max=%lld\n", (long long)k);
  printf("len=%lld\n", (long long)universe_ds_btree_sharded_len(m));
  universe_ds_btree_sharded_destroy(m);
  return 0;
}
```

```sh
./btscli   # min=0 max=3000999 len=4000
```

## Notes

- Thread-safe: point ops lock one shard; global ops acquire shards in fixed
  order. Range-partitioned so ordered queries stay globally correct.
- Signed key order (via sign-bit-flip mapping). Skewed key distributions can
  concentrate load on a few shards. Link with `-lpthread`.
