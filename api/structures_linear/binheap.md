# structures_linear/binheap

## Purpose

Binary-heap priority queue over i64 keys (min-heap). An implicit binary heap in
a flat i64 array — children of `i` are `2i+1`/`2i+2` — so the whole heap is one
contiguous, prefetch-friendly run with no node pointers. Push/pop use the "hole"
sift optimization (shift into the hole, one store when it settles). The 32-byte
handle is stable across doubling growth (payload reallocated separately). Order
is **signed** i64 (`icmp slt`); for a max-heap negate keys in and out (avoid
`INT64_MIN`). For a shallower tree on very large heaps or pop-heavy workloads,
see `structures_trees/heaps` (d-ary/radix/pairing).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_binheap_create(int64_t initial_cap)` | Create a min-heap | handle, or NULL on OOM |
| `int32_t universe_ds_binheap_push(void *h, int64_t key)` | Insert `key` | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_binheap_pop(void *h, int64_t *out)` | Remove and return the minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_binheap_peek(void *h, int64_t *out)` | Read the minimum without removing | 0 OK, 4 EMPTY |
| `int64_t universe_ds_binheap_len(void *h)` | Element count | count |
| `int64_t universe_ds_binheap_capacity(void *h)` | Allocated slots | count |
| `void universe_ds_binheap_destroy(void *h)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_binheap_create(i64)
declare i32 @universe_ds_binheap_push(ptr, i64)
declare i32 @universe_ds_binheap_pop(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Heap-sort stdin ints (ascending) by pushing all, then popping.

```c
// heapcli.c — build: clang -O3 heapcli.c build/libuniverse.a -lpthread -lm -o heapcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_binheap_create(int64_t);
extern int32_t universe_ds_binheap_push(void *, int64_t);
extern int32_t universe_ds_binheap_pop(void *, int64_t *);
extern void universe_ds_binheap_destroy(void *);

int main(void) {
  void *h = universe_ds_binheap_create(16);
  long long x;
  while (scanf("%lld", &x) == 1) universe_ds_binheap_push(h, x);
  int64_t v;
  while (universe_ds_binheap_pop(h, &v) == 0) printf("%lld\n", (long long)v);
  universe_ds_binheap_destroy(h);
  return 0;
}
```

```sh
printf '5 1 3 2 4\n' | ./heapcli   # -> 1 2 3 4 5
```

## Notes

- Min-heap, signed order. Max-heap: negate on push and pop.
- Keys are bare i64 (no associated value); for key+value priority queues use the
  pairing/radix heaps in `structures_trees/heaps`.
- Single-threaded.
