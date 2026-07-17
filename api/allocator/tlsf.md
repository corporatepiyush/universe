# allocator/tlsf — Two-Level Segregated Fit allocator

## Purpose
TLSF (Two-Level Segregated Fit): O(1) worst-case alloc and free with bounded
fragmentation — a general-purpose allocator (arbitrary sizes, real free and
reuse), unlike arena/pool/buddy which trade generality for a narrower fast
path. Free blocks are indexed by a two-level key: first level `FL =
floor(log2(size))` via `ctlz`, second level `SL` subdivides each power-of-two
class linearly into 16 ranges. An FL bitmap plus per-FL SL bitmaps make find
two `cttz`s — never a list scan. Coalescing is O(1) via a `prev_phys` pointer
and two flag bits in the size word. The control block is carved from the front
of the caller's region; if `base == NULL` the allocator `malloc`s the region
itself and frees it on destroy. Choose TLSF for real-time / general workloads
needing predictable alloc/free with genuine reuse.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `void *universe_alloc_tlsf_create(void *base, int64_t size)` | Create over caller region `base` (or `NULL` to self-`malloc`) of `size` bytes | handle, or `NULL` on bad size/OOM |
| `void *universe_alloc_tlsf_alloc(void *h, int64_t size)` | Allocate `size` bytes, 16-aligned | pointer, or `NULL` when full |
| `void *universe_alloc_tlsf_alloc_aligned(void *h, int64_t size, int64_t align)` | Allocate with `align` (power of two) | pointer, or `NULL` when full |
| `void universe_alloc_tlsf_free(void *h, void *p)` | Free `p` (no size needed) | — |
| `int64_t universe_alloc_tlsf_live(void *h)` | Summed live payload bytes | count |
| `void universe_alloc_tlsf_destroy(void *h)` | Free the region if owned (self-`malloc`ed) | — |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_alloc_tlsf_create(ptr, i64)
declare ptr @universe_alloc_tlsf_alloc(ptr, i64)
declare void @universe_alloc_tlsf_free(ptr, ptr)
declare void @universe_alloc_tlsf_destroy(ptr)

define i32 @main() {
  %h = call ptr @universe_alloc_tlsf_create(ptr null, i64 1048576)
  %p = call ptr @universe_alloc_tlsf_alloc(ptr %h, i64 200)
  call void @universe_alloc_tlsf_free(ptr %h, ptr %p)
  call void @universe_alloc_tlsf_destroy(ptr %h)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`tlsf_demo.c` — mixed-size alloc/free workload, report live bytes.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void   *universe_alloc_tlsf_create(void *base, int64_t size);
void   *universe_alloc_tlsf_alloc(void *h, int64_t size);
void    universe_alloc_tlsf_free(void *h, void *p);
int64_t universe_alloc_tlsf_live(void *h);
void    universe_alloc_tlsf_destroy(void *h);

int main(int argc, char **argv) {
    int64_t total = argc > 1 ? atoll(argv[1]) : (4 << 20);
    void *h = universe_alloc_tlsf_create(NULL, total);   /* self-malloc */
    if (!h) { fprintf(stderr, "create failed\n"); return 1; }
    void *p[64];
    for (int i = 0; i < 64; i++) p[i] = universe_alloc_tlsf_alloc(h, (i + 1) * 17);
    printf("live after 64 allocs = %lld\n", (long long)universe_alloc_tlsf_live(h));
    for (int i = 0; i < 64; i += 2) universe_alloc_tlsf_free(h, p[i]);
    printf("live after freeing half = %lld\n", (long long)universe_alloc_tlsf_live(h));
    universe_alloc_tlsf_destroy(h);
    return 0;
}
```
```
clang -O3 tlsf_demo.c build/libuniverse.a -lpthread -lm -o tlsf_demo
./tlsf_demo 4194304
```

## Notes
- Pass `base = NULL` to have TLSF `malloc` its own region (freed on destroy);
  pass a caller buffer to place the allocator inside it (never freed by TLSF).
- `free` needs no size (unlike buddy). Blocks are 16-aligned; minimum block 32 B.
- `live` returns summed live payload bytes (block size minus the 16 B header).
- Single-threaded; no atomics (a concurrent TLSF would need a striped/locked front).
