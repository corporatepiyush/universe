# allocator/pool_concurrent — lock-free fixed-block pool

## Purpose
A concurrent fixed-block pool: a lock-free Treiber free list plus a wait-free
wilderness bump — no locks at all. ABA is defeated with a generation tag packed
beside a 32-bit block offset in one 64-bit head word (`{tag:32 | off16:32}`,
`off16 = byte_offset/16`), so a plain portable 64-bit CAS suffices (no
`cmpxchg16b`/`casp`). Fresh blocks come from a separate wait-free bump cursor
(`atomicrmw add`) so an empty free list never serializes allocs behind a CAS
retry loop. Choose this for uniform-size objects churned by multiple threads.
Payload is capped at 64 GiB (checked at create).

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_cpool_create(int64_t block_size, int64_t block_count)` | Create a concurrent pool | handle, or `NULL` on OOM/overflow |
| `void *universe_alloc_cpool_alloc(void *pool)` | Take one block (lock-free) | pointer, or `NULL` when exhausted |
| `void universe_alloc_cpool_free(void *pool, void *block)` | Return a block (lock-free) | — |
| `int64_t universe_alloc_cpool_live(void *pool)` | Blocks currently allocated (approx under churn) | count |
| `int64_t universe_alloc_cpool_capacity(void *pool)` | Total blocks | count |
| `void universe_alloc_cpool_destroy(void *pool)` | Free the whole pool | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_cpool_create(i64, i64)
declare ptr @universe_alloc_cpool_alloc(ptr)
declare void @universe_alloc_cpool_free(ptr, ptr)
declare void @universe_alloc_cpool_destroy(ptr)

define i32 @main() {
  %p = call ptr @universe_alloc_cpool_create(i64 64, i64 4096)
  %b = call ptr @universe_alloc_cpool_alloc(ptr %p)
  call void @universe_alloc_cpool_free(ptr %p, ptr %b)
  call void @universe_alloc_cpool_destroy(ptr %p)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`cpool_demo.c` — T threads alloc-then-free in a loop; final live must be 0.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_alloc_cpool_create(int64_t block_size, int64_t block_count);
void   *universe_alloc_cpool_alloc(void *pool);
void    universe_alloc_cpool_free(void *pool, void *block);
int64_t universe_alloc_cpool_live(void *pool);
void    universe_alloc_cpool_destroy(void *pool);

static void *worker(void *pool) {
    for (int i = 0; i < 100000; i++) {
        void *b = universe_alloc_cpool_alloc(pool);
        if (b) universe_alloc_cpool_free(pool, b);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int t = argc > 1 ? atoi(argv[1]) : 4;
    void *p = universe_alloc_cpool_create(64, 8192);
    pthread_t th[64];
    for (int i = 0; i < t; i++) pthread_create(&th[i], NULL, worker, p);
    for (int i = 0; i < t; i++) pthread_join(th[i], NULL);
    printf("final live = %lld (want 0)\n", (long long)universe_alloc_cpool_live(p));
    universe_alloc_cpool_destroy(p);
    return 0;
}
```
```
clang -O3 cpool_demo.c build/libuniverse.a -lpthread -lm -o cpool_demo
./cpool_demo 8
```

## Notes
- Thread-safe alloc/free (lock-free); `live` is a monotonic stats counter and
  is approximate under concurrent churn.
- Block stride is 16-aligned; total payload capped at 64 GiB.
- `create`/`destroy` are not concurrent with in-flight operations.
