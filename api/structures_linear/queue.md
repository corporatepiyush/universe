# structures_linear/queue

## Purpose

Growable FIFO queue over a power-of-two ring with free-running unsigned indices
and mask wrap (no modulo, no branchy wrap). Elements are contiguous so a dequeue
walk streams memory for the prefetcher, and growth doubles the ring, linearizing
wrapped contents with at most two memcpys. No per-node allocation. Layout is
`{ data, head, tail, mask, elem }` with a stable separately-malloc'd handle.
Choose this for pure FIFO; use `deque` when you need both ends, `ringbuf` for a
fixed (non-growing) capacity.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_queue_create(int64_t elem_size, int64_t initial_cap)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `int32_t universe_ds_queue_enqueue(void *q, void *elem)` | Append at back | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_queue_dequeue(void *q, void *out)` | Remove from front | 0 OK, 4 EMPTY |
| `int32_t universe_ds_queue_peek(void *q, void *out)` | Read front without removal | 0 OK, 4 EMPTY |
| `int64_t universe_ds_queue_count(void *q)` | Element count | count |
| `int64_t universe_ds_queue_capacity(void *q)` | Ring capacity | count |
| `void universe_ds_queue_destroy(void *q)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_queue_create(i64, i64)
declare i32 @universe_ds_queue_enqueue(ptr, ptr)
declare i32 @universe_ds_queue_dequeue(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

FIFO echo: enqueue stdin ints, then drain them in order.

```c
// qcli.c — build: clang -O3 qcli.c build/libuniverse.a -lpthread -lm -o qcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_queue_create(int64_t, int64_t);
extern int32_t universe_ds_queue_enqueue(void *, void *);
extern int32_t universe_ds_queue_dequeue(void *, void *);
extern void universe_ds_queue_destroy(void *);

int main(void) {
  void *q = universe_ds_queue_create(sizeof(int), 8);
  int x;
  while (scanf("%d", &x) == 1) universe_ds_queue_enqueue(q, &x);
  int v;
  while (universe_ds_queue_dequeue(q, &v) == 0) printf("%d\n", v);
  universe_ds_queue_destroy(q);
  return 0;
}
```

```sh
printf '1 2 3\n' | ./qcli   # -> 1 2 3
```

## Notes

- Elements are POD copied by value; handle stable across growth.
- Single-threaded. For a lock-free single-producer/single-consumer ring use
  `ringbuf_concurrent`.
