# allocator/pool — fixed-block pool allocator

## Purpose
A fixed-block pool: O(1) alloc, O(1) free, O(1) create. Every block is the same
size. Create is O(1) via the wilderness trick — blocks are NOT pre-linked into
a free list (an eager init loop wastes page faults and pollutes cache); fresh
blocks come from a bump cursor and the free list only ever holds blocks that
were actually freed. The free list stores byte offsets (intrusive, inside the
freed block) so alloc/free do zero multiplications. Choose the pool for
uniform-size objects with individual lifetimes (nodes, cells, fixed records).
Single-threaded; see `pool_concurrent` for the lock-free variant.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_pool_create(int64_t block_size, int64_t block_count)` | Create a pool of `block_count` blocks of `block_size` bytes | handle, or `NULL` on OOM/overflow |
| `void *universe_alloc_pool_alloc(void *pool)` | Take one block | pointer, or `NULL` when exhausted |
| `void universe_alloc_pool_free(void *pool, void *block)` | Return a block (must be from this pool) | — |
| `int64_t universe_alloc_pool_live(void *pool)` | Blocks currently allocated | count |
| `int64_t universe_alloc_pool_capacity(void *pool)` | Total blocks | count |
| `void universe_alloc_pool_destroy(void *pool)` | Free the whole pool | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_pool_create(i64, i64)
declare ptr @universe_alloc_pool_alloc(ptr)
declare void @universe_alloc_pool_free(ptr, ptr)
declare void @universe_alloc_pool_destroy(ptr)

define i32 @main() {
  %p = call ptr @universe_alloc_pool_create(i64 64, i64 1024)
  %b = call ptr @universe_alloc_pool_alloc(ptr %p)
  call void @universe_alloc_pool_free(ptr %p, ptr %b)
  call void @universe_alloc_pool_destroy(ptr %p)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`pool_demo.c` — alloc/free churn, report live count.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void   *universe_alloc_pool_create(int64_t block_size, int64_t block_count);
void   *universe_alloc_pool_alloc(void *pool);
void    universe_alloc_pool_free(void *pool, void *block);
int64_t universe_alloc_pool_live(void *pool);
void    universe_alloc_pool_destroy(void *pool);

int main(int argc, char **argv) {
    int64_t bs = argc > 1 ? atoll(argv[1]) : 64;
    int64_t bc = argc > 2 ? atoll(argv[2]) : 1024;
    void *p = universe_alloc_pool_create(bs, bc);
    if (!p) { fprintf(stderr, "create failed\n"); return 1; }
    void *a = universe_alloc_pool_alloc(p);
    void *b = universe_alloc_pool_alloc(p);
    printf("live after 2 allocs = %lld\n", (long long)universe_alloc_pool_live(p));
    universe_alloc_pool_free(p, a);
    printf("live after 1 free   = %lld\n", (long long)universe_alloc_pool_live(p));
    universe_alloc_pool_free(p, b);
    universe_alloc_pool_destroy(p);
    return 0;
}
```
```
clang -O3 pool_demo.c build/libuniverse.a -lpthread -lm -o pool_demo
./pool_demo 64 1024
```

## Notes
- Every block is 16-byte aligned; block stride is the 16-aligned `block_size`.
- `free` must be passed a pointer previously returned by this pool; double-free
  detection is deferred (hardening phase).
- Not thread-safe; use `pool_concurrent` for concurrent alloc/free.
