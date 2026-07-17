# allocator/buddy — buddy allocator

## Purpose
A buddy allocator: power-of-two blocks, O(1) find-fit, XOR coalescing. It uses
a SIZED FREE API (the caller passes the size back, like C++ sized delete) so
there is zero per-block metadata — a 64 B block costs exactly 64 B. An
`avail_mask` u64 has bit k set iff the order-k free list is nonempty, so
find-fit is one shift plus one `cttz` — no list scan. A block's buddy is
`off ^ (min<<k)` (one XOR), enabling O(1) coalescing. Choose the buddy when you
need general variable-size allocation from a fixed region with fast merge and
predictable fragmentation, and can track sizes at the call site.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_buddy_create(int64_t total_size, int64_t min_block)` | Create a buddy allocator over `total_size` bytes, minimum block `min_block` | handle, or `NULL` on bad size/OOM |
| `void *universe_alloc_buddy_alloc(void *b, int64_t size)` | Allocate a block that fits `size` | pointer, or `NULL` when full |
| `void universe_alloc_buddy_free(void *b, void *p, int64_t size)` | Free `p`; `size` must equal the size passed at alloc | — |
| `int64_t universe_alloc_buddy_live(void *b)` | Bytes currently allocated | count |
| `void universe_alloc_buddy_destroy(void *b)` | Free the allocator | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_buddy_create(i64, i64)
declare ptr @universe_alloc_buddy_alloc(ptr, i64)
declare void @universe_alloc_buddy_free(ptr, ptr, i64)
declare void @universe_alloc_buddy_destroy(ptr)

define i32 @main() {
  %b = call ptr @universe_alloc_buddy_create(i64 1048576, i64 32)
  %p = call ptr @universe_alloc_buddy_alloc(ptr %b, i64 100)
  call void @universe_alloc_buddy_free(ptr %b, ptr %p, i64 100)
  call void @universe_alloc_buddy_destroy(ptr %b)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`buddy_demo.c` — allocate a few sizes, free them back (passing size), report live.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void   *universe_alloc_buddy_create(int64_t total_size, int64_t min_block);
void   *universe_alloc_buddy_alloc(void *b, int64_t size);
void    universe_alloc_buddy_free(void *b, void *p, int64_t size);
int64_t universe_alloc_buddy_live(void *b);
void    universe_alloc_buddy_destroy(void *b);

int main(int argc, char **argv) {
    int64_t total = argc > 1 ? atoll(argv[1]) : (1 << 20);
    void *b = universe_alloc_buddy_create(total, 32);
    if (!b) { fprintf(stderr, "create failed\n"); return 1; }
    int64_t sizes[] = {100, 5000, 300, 40000};
    void *p[4];
    for (int i = 0; i < 4; i++) p[i] = universe_alloc_buddy_alloc(b, sizes[i]);
    printf("live after 4 allocs = %lld\n", (long long)universe_alloc_buddy_live(b));
    for (int i = 0; i < 4; i++) universe_alloc_buddy_free(b, p[i], sizes[i]);
    printf("live after freeing all = %lld (want 0)\n",
           (long long)universe_alloc_buddy_live(b));
    universe_alloc_buddy_destroy(b);
    return 0;
}
```
```
clang -O3 buddy_demo.c build/libuniverse.a -lpthread -lm -o buddy_demo
./buddy_demo 1048576
```

## Notes
- `free` REQUIRES the exact `size` used at `alloc` (sized-free contract); a
  wrong size corrupts the free-tree state.
- `min_block` is 32 B minimum (free blocks store intrusive `{next,prev}`
  offsets). Allocation rounds up to the next power of two.
- Not thread-safe.
