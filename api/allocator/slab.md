# allocator/slab — growable slab allocator

## Purpose
A slab allocator: growable, fixed-size objects, O(1) alloc/free, and empty
slabs are returned to the OS. Slabs are allocated with `posix_memalign(span,
span)` where `span` is a power of two, so freeing an object finds its slab with
one `and` (`slab = ptr & -span`) — no per-object headers, no slab search, no
division. Each slab keeps an intrusive free list of byte offsets plus a
wilderness bump cursor (O(1) create). Slabs form a doubly-linked list; a slab
that goes fully empty is unlinked and `free()`d in O(1), so memory actually
returns. Choose the slab over the pool when the working set grows and shrinks
and you want memory reclaimed.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_slab_create(int64_t obj_size, int64_t objs_per_slab)` | Create a slab allocator; `objs_per_slab` clamped to [1,512] | handle, or `NULL` on OOM/overflow |
| `void *universe_alloc_slab_alloc(void *s)` | Allocate one object | pointer, or `NULL` on OOM |
| `void universe_alloc_slab_free(void *s, void *obj)` | Free an object (must be from this allocator) | — |
| `int64_t universe_alloc_slab_live(void *s)` | Objects currently allocated | count |
| `void universe_alloc_slab_destroy(void *s)` | Free all slabs and the handle | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_slab_create(i64, i64)
declare ptr @universe_alloc_slab_alloc(ptr)
declare void @universe_alloc_slab_free(ptr, ptr)
declare void @universe_alloc_slab_destroy(ptr)

define i32 @main() {
  %s = call ptr @universe_alloc_slab_create(i64 48, i64 256)
  %o = call ptr @universe_alloc_slab_alloc(ptr %s)
  call void @universe_alloc_slab_free(ptr %s, ptr %o)
  call void @universe_alloc_slab_destroy(ptr %s)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`slab_demo.c` — grow by N allocs, then free half, report live count.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void   *universe_alloc_slab_create(int64_t obj_size, int64_t objs_per_slab);
void   *universe_alloc_slab_alloc(void *s);
void    universe_alloc_slab_free(void *s, void *obj);
int64_t universe_alloc_slab_live(void *s);
void    universe_alloc_slab_destroy(void *s);

int main(int argc, char **argv) {
    int64_t n = argc > 1 ? atoll(argv[1]) : 1000;
    void *s = universe_alloc_slab_create(48, 256);
    if (!s) { fprintf(stderr, "create failed\n"); return 1; }
    void **objs = malloc(sizeof(void *) * n);
    for (int64_t i = 0; i < n; i++) objs[i] = universe_alloc_slab_alloc(s);
    printf("live after %lld allocs = %lld\n", (long long)n,
           (long long)universe_alloc_slab_live(s));
    for (int64_t i = 0; i < n; i += 2) universe_alloc_slab_free(s, objs[i]);
    printf("live after freeing half = %lld\n",
           (long long)universe_alloc_slab_live(s));
    free(objs);
    universe_alloc_slab_destroy(s);
    return 0;
}
```
```
clang -O3 slab_demo.c build/libuniverse.a -lpthread -lm -o slab_demo
./slab_demo 1000
```

## Notes
- Objects are placed at `slab+64`; a slab's `span` is a power of two so
  `obj & -span` recovers the slab header.
- Empty slabs are unlinked and freed automatically, so RSS shrinks with the
  working set.
- Not thread-safe.
