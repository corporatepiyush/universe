# concurrent/striped_counter — striped 64-bit counter

## Purpose
A STRIPED (sharded) 64-bit counter. A single shared atomic counter scales
NEGATIVELY under contention — every `atomicrmw add` bounces one cache line
between cores, so 8 cores are slower than 1. State sharding fixes this:
partition the counter into a power-of-two number of independent shards, each on
its OWN 128 B cache line. A thread increments the shard chosen by
`shard_id & (N-1)`; with distinct per-thread ids the atomics never contend.
`sum()` merges the shards (a cold, infrequent read). All orderings are
`monotonic` — a statistics counter carries no happens-before for other data;
conservation is exact because each increment is a single `atomicrmw add`.
Per-id (not key-hash) sharding is skew-proof: hot ids still spread across shards.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_conc_scounter_create(int64_t nshards)` | Create; `nshards` rounded up to pow2, clamped to [1, 2^20], 0 → default 64 | handle, or `NULL` on OOM |
| `void universe_conc_scounter_inc(void *c, int64_t shard_id, int64_t delta)` | Add `delta` to shard `shard_id & (N-1)` | — |
| `int64_t universe_conc_scounter_sum(void *c)` | Sum of all shards | total |
| `void universe_conc_scounter_reset(void *c)` | Zero every shard | — |
| `int64_t universe_conc_scounter_shards(void *c)` | Number of shards | count |
| `void universe_conc_scounter_destroy(void *c)` | Free | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_conc_scounter_create(i64)
declare void @universe_conc_scounter_inc(ptr, i64, i64)
declare i64 @universe_conc_scounter_sum(ptr)
declare void @universe_conc_scounter_destroy(ptr)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`scounter_demo.c` — T threads each add 1 a million times; verify the sum.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_conc_scounter_create(int64_t nshards);
void    universe_conc_scounter_inc(void *c, int64_t shard_id, int64_t delta);
int64_t universe_conc_scounter_sum(void *c);
void    universe_conc_scounter_destroy(void *c);

#define PER 1000000
static int T = 8;
static void *worker(void *arg) {
    void *c = ((void **)arg)[0];
    int64_t id = (int64_t)(intptr_t)((void **)arg)[1];
    for (int i = 0; i < PER; i++) universe_conc_scounter_inc(c, id, 1);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) T = atoi(argv[1]);
    void *c = universe_conc_scounter_create(T * 4);   /* ~4x threads */
    pthread_t th[64]; void *args[64][2];
    for (int i = 0; i < T; i++) {
        args[i][0] = c; args[i][1] = (void *)(intptr_t)i;
        pthread_create(&th[i], NULL, worker, args[i]);
    }
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    int64_t want = (int64_t)T * PER;
    printf("sum=%lld want=%lld %s\n", (long long)universe_conc_scounter_sum(c),
           (long long)want, universe_conc_scounter_sum(c) == want ? "OK" : "FAIL");
    universe_conc_scounter_destroy(c);
    return 0;
}
```
```
clang -O3 scounter_demo.c build/libuniverse.a -lpthread -lm -o scounter_demo
./scounter_demo 8
```

## Notes
- Pass a stable per-thread/core `shard_id` to `inc` so threads hit distinct
  lines; the module masks it to `& (N-1)`.
- Choose `nshards ≥` the number of writer threads (2×–4× cores is the sweet
  spot; past the knee it is memory for nothing).
- `inc` is wait-free (single `atomicrmw add`); `sum`/`reset` are cold and
  approximate under concurrent increment.
