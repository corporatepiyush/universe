# allocator/arena_concurrent — wait-free concurrent arena

## Purpose
A concurrent arena: wait-free bump allocation from any number of threads. The
entire alloc is one `atomicrmw add` on the shared cursor (a single `LDADD` on
AArch64+LSE, `lock xadd` on x86) plus a bounds check — no mutex, no convoy, no
syscall on contention. Ordering is `monotonic` because each caller receives a
disjoint region; no data is published through the cursor itself (callers who
share a block synchronize themselves, exactly as with `malloc`). The hot atomic
line lives 128 B from the payload so it never false-shares with user data.
Over-reservation on an exhausted alloc is rolled back with an atomic sub.
Choose this when many threads need cheap scratch with a shared lifetime.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_carena_create(int64_t capacity)` | Create a concurrent arena for `capacity` payload bytes | handle, or `NULL` on OOM/overflow |
| `void *universe_alloc_carena_alloc(void *arena, int64_t size)` | Wait-free bump-allocate `size` bytes, 16-aligned | pointer, or `NULL` when full |
| `void *universe_alloc_carena_alloc_aligned(void *arena, int64_t size, int64_t align)` | Bump-allocate `size` bytes with `align` (power of two) | pointer, or `NULL` when full |
| `void universe_alloc_carena_reset(void *arena)` | Reset cursor (QUIESCENT only — no concurrent allocs) | — |
| `int64_t universe_alloc_carena_used(void *arena)` | Bytes currently handed out | count |
| `int64_t universe_alloc_carena_capacity(void *arena)` | Total payload capacity | count |
| `void universe_alloc_carena_destroy(void *arena)` | Free the whole arena | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_carena_create(i64)
declare ptr @universe_alloc_carena_alloc(ptr, i64)
declare void @universe_alloc_carena_destroy(ptr)

define i32 @main() {
  %a = call ptr @universe_alloc_carena_create(i64 1048576)
  %p = call ptr @universe_alloc_carena_alloc(ptr %a, i64 128)
  call void @universe_alloc_carena_destroy(ptr %a)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`carena_demo.c` — hammer the arena from T threads, verify total bytes used.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_alloc_carena_create(int64_t capacity);
void   *universe_alloc_carena_alloc(void *arena, int64_t size);
int64_t universe_alloc_carena_used(void *arena);
void    universe_alloc_carena_destroy(void *arena);

static void *worker(void *a) {
    for (int i = 0; i < 10000; i++) universe_alloc_carena_alloc(a, 16);
    return NULL;
}

int main(int argc, char **argv) {
    int t = argc > 1 ? atoi(argv[1]) : 4;
    void *a = universe_alloc_carena_create(1 << 24);
    pthread_t th[64];
    for (int i = 0; i < t; i++) pthread_create(&th[i], NULL, worker, a);
    for (int i = 0; i < t; i++) pthread_join(th[i], NULL);
    printf("used %lld bytes across %d threads\n",
           (long long)universe_alloc_carena_used(a), t);
    universe_alloc_carena_destroy(a);
    return 0;
}
```
```
clang -O3 carena_demo.c build/libuniverse.a -lpthread -lm -o carena_demo
./carena_demo 8
```

## Notes
- Thread-safe for `alloc`/`alloc_aligned`/`used`/`capacity`. `reset` and
  `destroy` are quiescent-only — no concurrent allocations may be in flight.
- Individual blocks are never freed. Regions handed to distinct callers are
  disjoint; cross-thread sharing of a returned block needs your own
  release/acquire.
