# concurrent/sharded_map — striped-lock concurrent hash map

## Purpose
A SHARDED (striped-lock) concurrent hash map: byte-string keys (`ptr + len`) →
`int64_t` values. A single lock over one shared map serializes every writer and
scales NEGATIVELY with cores; lock striping fixes this by partitioning the key
space into a power-of-two number of shards, each a self-contained open-addressing
map guarded by its OWN test-and-test-and-set spinlock on its OWN 128 B cache
line. A key routes to `shard = hash(key) & (N-1)` (mask, never modulo); ops on
different shards run in parallel and never touch each other's lock line, so
contention drops ~N×. Two decorrelated hashes come from one FNV-1a pass (primary
for shard select, splitmix64-mixed for the in-shard probe). Per-shard grow/rehash
happens under the shard lock, invisible to others. `len()` takes every shard lock
in ascending order (deadlock-free) for a consistent snapshot.

## Exported API
Error codes: `0` OK, `1` NULL_PTR, `2` OOM, `3` SIZE_OVERFLOW, `5` NOT_FOUND.

| C signature | Description | Returns |
|---|---|---|
| `void *universe_conc_shardmap_create(int64_t nshards, int64_t cap_per_shard)` | Create a map with `nshards` (rounded to pow2) shards | handle, or `NULL` on OOM |
| `int32_t universe_conc_shardmap_put(void *m, const void *key, int64_t klen, int64_t val)` | Insert/update `key → val` (key is copied) | 0 / 1 / 2 / 3 |
| `int32_t universe_conc_shardmap_get(void *m, const void *key, int64_t klen, int64_t *out)` | Look up `key`, write value to `*out` | 0 / 1 / 5 |
| `int32_t universe_conc_shardmap_delete(void *m, const void *key, int64_t klen)` | Remove `key` | 0 / 1 / 5 |
| `int64_t universe_conc_shardmap_len(void *m)` | Total live entries (consistent snapshot) | count |
| `int64_t universe_conc_shardmap_shards(void *m)` | Number of shards | count |
| `void universe_conc_shardmap_destroy(void *m)` | Free the map and all key copies | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_conc_shardmap_create(i64, i64)
declare i32 @universe_conc_shardmap_put(ptr, ptr, i64, i64)
declare i32 @universe_conc_shardmap_get(ptr, ptr, i64, ptr)
declare void @universe_conc_shardmap_destroy(ptr)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`shardmap_demo.c` — T threads each insert N distinct keys; verify final len.
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_conc_shardmap_create(int64_t nshards, int64_t cap_per_shard);
int32_t universe_conc_shardmap_put(void *m, const void *key, int64_t klen, int64_t val);
int32_t universe_conc_shardmap_get(void *m, const void *key, int64_t klen, int64_t *out);
int64_t universe_conc_shardmap_len(void *m);
void    universe_conc_shardmap_destroy(void *m);

#define PER 20000
static int T = 4;
static void *worker(void *arg) {
    void *m = ((void **)arg)[0];
    int   id = (int)(intptr_t)((void **)arg)[1];
    char key[32];
    for (int i = 0; i < PER; i++) {
        int n = snprintf(key, sizeof key, "%d:%d", id, i);
        universe_conc_shardmap_put(m, key, n, i);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) T = atoi(argv[1]);
    void *m = universe_conc_shardmap_create(256, 64);
    pthread_t th[64]; void *args[64][2];
    for (int i = 0; i < T; i++) {
        args[i][0] = m; args[i][1] = (void *)(intptr_t)i;
        pthread_create(&th[i], NULL, worker, args[i]);
    }
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    int64_t want = (int64_t)T * PER;
    printf("len=%lld want=%lld %s\n", (long long)universe_conc_shardmap_len(m),
           (long long)want, universe_conc_shardmap_len(m) == want ? "OK" : "FAIL");
    universe_conc_shardmap_destroy(m);
    return 0;
}
```
```
clang -O3 shardmap_demo.c build/libuniverse.a -lpthread -lm -o shardmap_demo
./shardmap_demo 8
```

## Notes
- Fully thread-safe for concurrent put/get/delete. Keys are copied into a
  per-entry malloc, so callers may free their key buffers after the call.
- `nshards` rounds up to a power of two; tune to `cores × write-ratio`
  (~256 shards at 8 cores is past the knee for most workloads).
- `len` locks every shard (cold, consistent snapshot); avoid on the hot path.
- Skewed/Zipfian key sets defeat key-hash striping (hot keys collide on few
  shards) — for heavy skew shard per-thread with lazy merge instead.
