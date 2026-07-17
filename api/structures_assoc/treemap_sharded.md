# structures_assoc/treemap_sharded

## Purpose

Concurrent ordered map — the sharded flavor of `treemap`. It holds N
power-of-two shards, each an independent array-backed treemap (reused by direct
call) guarded by its own 128 B-padded TTAS spinlock. Shards are **range**
partitioned by the key's top `s = log2(N)` bits (`shard = key >> (64 - s)`),
which is monotone in the unsigned key, so global ascending order is preserved:
shard 0's keys precede shard 1's, etc. — the whole point of keeping it ordered.
**Choose this** for many threads doing point put/get/delete on disjoint key
ranges plus occasional ordered queries; range-striping drops contention ~N×
versus a single-lock treemap while keeping cache-dense ordered scans. Keys are
ordered **unsigned** (same convention as `treemap`).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_treemap_sharded_create(int64_t nshards, int64_t cap_per_shard)` | Create with `nshards` (rounded to pow2) shards | handle, or NULL on OOM |
| `int32_t universe_ds_treemap_sharded_put(void *m, int64_t key, int64_t val)` | Insert/update (locks one shard) | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_treemap_sharded_get(void *m, int64_t key, int64_t *out_v)` | Point lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_sharded_contains(void *m, int64_t key)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_treemap_sharded_delete(void *m, int64_t key)` | Remove | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_treemap_sharded_shards(void *m)` | Shard count | count |
| `int64_t universe_ds_treemap_sharded_size(void *m)` | Total entries across shards | count |
| `int32_t universe_ds_treemap_sharded_min(void *m, int64_t *out_k, int64_t *out_v)` | Global minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_treemap_sharded_max(void *m, int64_t *out_k, int64_t *out_v)` | Global maximum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_treemap_sharded_floor(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Largest key <= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_sharded_lower(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Largest key < `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_sharded_ceiling(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Smallest key >= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_sharded_higher(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Smallest key > `key` | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_treemap_sharded_range(void *m, int64_t lo, int64_t hi, int64_t *out_k, int64_t *out_v, int64_t out_cap)` | Ordered copy of `lo <= key < hi` | count copied |
| `void universe_ds_treemap_sharded_foreach(void *m, void *fn, void *ctx)` | Global in-order iterate; `fn` is `void(*)(void *ctx, int64_t key, int64_t val)` | — |
| `void universe_ds_treemap_sharded_destroy(void *m)` | Free all shards | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_treemap_sharded_create(i64, i64)
declare i32 @universe_ds_treemap_sharded_put(ptr, i64, i64)
declare i32 @universe_ds_treemap_sharded_get(ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Load values concurrently, then dump ordered min/max.

```c
// tmscli.c — build: clang -O3 tmscli.c build/libuniverse.a -lpthread -lm -o tmscli
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
extern void *universe_ds_treemap_sharded_create(int64_t, int64_t);
extern int32_t universe_ds_treemap_sharded_put(void *, int64_t, int64_t);
extern int32_t universe_ds_treemap_sharded_min(void *, int64_t *, int64_t *);
extern int32_t universe_ds_treemap_sharded_max(void *, int64_t *, int64_t *);
extern int64_t universe_ds_treemap_sharded_size(void *);
extern void universe_ds_treemap_sharded_destroy(void *);

static void *worker(void *arg) {
  void *m = ((void **)arg)[0]; long base = (long)((void **)arg)[1];
  for (long i = 0; i < 1000; i++) universe_ds_treemap_sharded_put(m, base + i, i);
  return 0;
}
int main(void) {
  void *m = universe_ds_treemap_sharded_create(16, 256);
  pthread_t t[4]; void *args[4][2];
  for (int i = 0; i < 4; i++) { args[i][0] = m; args[i][1] = (void *)(long)(i * 100000);
    pthread_create(&t[i], 0, worker, args[i]); }
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  int64_t k, v;
  universe_ds_treemap_sharded_min(m, &k, &v); printf("min=%lld\n", (long long)k);
  universe_ds_treemap_sharded_max(m, &k, &v); printf("max=%lld\n", (long long)k);
  printf("size=%lld\n", (long long)universe_ds_treemap_sharded_size(m));
  universe_ds_treemap_sharded_destroy(m);
  return 0;
}
```

```sh
./tmscli   # min=0 max=300999 size=4000
```

## Notes

- Thread-safe: point ops take one shard's spinlock; disjoint-range writers run
  fully parallel. Global ops (size/min/max/range/foreach) acquire shards in a
  fixed order.
- Range-partitioned (not hash), so ordered queries stay globally correct; skewed
  key distributions can concentrate load on a few shards.
- Link with `-lpthread`.
