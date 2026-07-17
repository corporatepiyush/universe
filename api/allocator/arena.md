# allocator/arena — bump (arena) allocator

## Purpose
An arena (bump) allocator: O(1) allocation, O(1) whole-arena reset, zero
per-block metadata. Header and payload live in one `malloc`; the header is
exactly one cache line and the payload starts 64 B in. `used` is kept
16-aligned so the fast path (~6 branch-predicted instructions) only rounds
the size, never the cursor. All size math is overflow-checked. Choose the
arena when you allocate many objects with a shared lifetime and free them all
at once (parse trees, per-request scratch, frame allocators). Single-threaded;
for concurrent use, guard it with a mutex in the caller.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_arena_create(int64_t capacity)` | Create an arena backing `capacity` payload bytes | handle, or `NULL` on OOM/overflow |
| `void *universe_alloc_arena_alloc(void *arena, int64_t size)` | Bump-allocate `size` bytes, 16-aligned | pointer, or `NULL` when full |
| `void *universe_alloc_arena_alloc_aligned(void *arena, int64_t size, int64_t align)` | Bump-allocate `size` bytes with `align` (power of two) | pointer, or `NULL` when full |
| `void universe_alloc_arena_reset(void *arena)` | Reset cursor to empty; frees nothing, reuses all memory | — |
| `int64_t universe_alloc_arena_used(void *arena)` | Bytes currently handed out | count |
| `int64_t universe_alloc_arena_capacity(void *arena)` | Total payload capacity | count |
| `void universe_alloc_arena_destroy(void *arena)` | Free the whole arena | — |

## Use in an LLVM-based environment
Declare the symbols and call them (C ABI, `nounwind`). `make lib` builds
`build/libuniverse.a`.

```llvm
declare ptr @universe_alloc_arena_create(i64)
declare ptr @universe_alloc_arena_alloc(ptr, i64)
declare void @universe_alloc_arena_destroy(ptr)

define i32 @main() {
  %a = call ptr @universe_alloc_arena_create(i64 4096)
  %p = call ptr @universe_alloc_arena_alloc(ptr %a, i64 100)
  call void @universe_alloc_arena_destroy(ptr %a)
  ret i32 0
}
```

Link line:
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`arena_demo.c` — allocate N blocks from an arena, report bytes used.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void   *universe_alloc_arena_create(int64_t capacity);
void   *universe_alloc_arena_alloc(void *arena, int64_t size);
int64_t universe_alloc_arena_used(void *arena);
void    universe_alloc_arena_destroy(void *arena);

int main(int argc, char **argv) {
    int64_t cap = argc > 1 ? atoll(argv[1]) : 65536;
    int64_t n   = argc > 2 ? atoll(argv[2]) : 100;
    void *a = universe_alloc_arena_create(cap);
    if (!a) { fprintf(stderr, "create failed\n"); return 1; }
    for (int64_t i = 0; i < n; i++) {
        void *p = universe_alloc_arena_alloc(a, 64);
        if (!p) { printf("full after %lld blocks\n", (long long)i); break; }
    }
    printf("used %lld / %lld bytes\n", (long long)universe_alloc_arena_used(a), (long long)cap);
    universe_alloc_arena_destroy(a);
    return 0;
}
```
```
clang -O3 arena_demo.c build/libuniverse.a -lpthread -lm -o arena_demo
./arena_demo 65536 100
```

## Notes
- One `malloc` per arena; payload 16-aligned. Individual blocks are never
  freed — only `reset` (reuse) or `destroy` (release everything).
- Not thread-safe; the caller adds a mutex if the arena is shared across threads.
- Pointers returned stay valid until the next `reset` or `destroy`.
