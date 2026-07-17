# structures_trees/art_sharded

## Purpose

Hash-sharded concurrent adaptive radix tree. It stripes keys across N
power-of-two independent shards, each a full ART (reused directly from `art`)
guarded by its own TTAS spinlock on its own 128 B cache-line pair. The shard is
chosen by an FNV-1a/64 hash of the **whole key** (`shard = hash(key) & (N-1)`),
so identical keys always route to the same shard and prefix-sharing keys still
route correctly. Point ops lock exactly one shard, so threads touching keys in
different shards run fully in parallel. **Choose this** over `art` for concurrent
byte-string maps; note that because sharding is by hash, only a per-shard prefix
scan is offered (no globally ordered scan).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_art_sharded_create(int64_t nshards)` | Create with `nshards` (rounded to pow2) shards | handle, or NULL on OOM |
| `int32_t universe_ds_art_sharded_put(void *m, void *key, int64_t klen, int64_t val)` | Insert/overwrite (locks one shard) | 0 OK, 1 NULL, 2 OOM, 8 INVALID_ARG |
| `int32_t universe_ds_art_sharded_get(void *m, void *key, int64_t klen, int64_t *out)` | Lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_art_sharded_contains(void *m, void *key, int64_t klen)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_art_sharded_delete(void *m, void *key, int64_t klen)` | Remove | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_art_sharded_len(void *m)` | Total key count | count |
| `int64_t universe_ds_art_sharded_shards(void *m)` | Shard count | count |
| `int64_t universe_ds_art_sharded_prefix_scan(void *m, void *pfx, int64_t plen, void *cb, void *ctx)` | Per-shard prefix scan; `cb` is `int32_t(void *ctx, void *key, int64_t klen, int64_t val)` | keys visited |
| `void universe_ds_art_sharded_destroy(void *m)` | Free all shards | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_art_sharded_create(i64)
declare i32 @universe_ds_art_sharded_put(ptr, ptr, i64, i64)
declare i32 @universe_ds_art_sharded_get(ptr, ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Populate a string map from many threads, then query.

```c
// artscli.c — build: clang -O3 artscli.c build/libuniverse.a -lpthread -lm -o artscli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
extern void *universe_ds_art_sharded_create(int64_t);
extern int32_t universe_ds_art_sharded_put(void *, void *, int64_t, int64_t);
extern int32_t universe_ds_art_sharded_get(void *, void *, int64_t, int64_t *);
extern int64_t universe_ds_art_sharded_len(void *);
extern void universe_ds_art_sharded_destroy(void *);

static void *worker(void *arg) {
  void *m = ((void **)arg)[0]; long id = (long)((void **)arg)[1];
  char k[32];
  for (long i = 0; i < 1000; i++) { int n = sprintf(k, "t%ld_%ld", id, i);
    universe_ds_art_sharded_put(m, k, n, i); }
  return 0;
}
int main(void) {
  void *m = universe_ds_art_sharded_create(16);
  pthread_t t[4]; void *a[4][2];
  for (int i = 0; i < 4; i++) { a[i][0] = m; a[i][1] = (void *)(long)i;
    pthread_create(&t[i], 0, worker, a[i]); }
  for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
  int64_t v; universe_ds_art_sharded_get(m, "t2_500", 6, &v);
  printf("len=%lld t2_500=%lld\n", (long long)universe_ds_art_sharded_len(m), (long long)v);
  universe_ds_art_sharded_destroy(m);
  return 0;
}
```

```sh
./artscli   # len=4000 t2_500=500
```

## Notes

- Thread-safe point ops via per-shard spinlocks; disjoint keys proceed fully
  parallel.
- Hash-sharded, so there is no globally ordered iteration — `prefix_scan` runs
  per shard (order across shards is not defined). Link with `-lpthread`.
