# structures_linear/ringbuf

## Purpose

Fixed-capacity ring buffer (single-threaded) for arbitrary-size elements.
Capacity is rounded up to a power of two so slot selection is `index & mask` (no
modulo, no wrap branch). Indices are free-running monotonic u64 counters, so
`count = tail - head` works across wrap and full/empty are single subtractions.
Header and slots are one allocation with the payload at +64. Unlike `queue`,
capacity is fixed — a push on a full buffer returns FULL rather than growing.
Use for bounded backpressure buffers; use `ringbuf_concurrent` for the lock-free
SPSC variant.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_ringbuf_create(int64_t capacity, int64_t elem_size)` | Create a ring (capacity rounded up to pow2) | handle, or NULL on OOM |
| `int32_t universe_ds_ringbuf_push(void *rb, void *elem)` | Append at back | 0 OK, 1 NULL, 6 FULL |
| `int32_t universe_ds_ringbuf_pop(void *rb, void *out)` | Remove oldest | 0 OK, 4 EMPTY |
| `int32_t universe_ds_ringbuf_peek(void *rb, int64_t index, void *out)` | Read the element at logical `index` (0 = oldest) | 0 OK, 4 EMPTY, 7 INVALID_INDEX |
| `int64_t universe_ds_ringbuf_count(void *rb)` | Element count | count |
| `int64_t universe_ds_ringbuf_capacity(void *rb)` | Rounded capacity | count |
| `void universe_ds_ringbuf_destroy(void *rb)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_ringbuf_create(i64, i64)
declare i32 @universe_ds_ringbuf_push(ptr, ptr)
declare i32 @universe_ds_ringbuf_pop(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A sliding window of the last N ints (drop-oldest on overflow).

```c
// rbcli.c — build: clang -O3 rbcli.c build/libuniverse.a -lpthread -lm -o rbcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_ringbuf_create(int64_t, int64_t);
extern int32_t universe_ds_ringbuf_push(void *, void *);
extern int32_t universe_ds_ringbuf_pop(void *, void *);
extern int64_t universe_ds_ringbuf_count(void *);
extern void universe_ds_ringbuf_destroy(void *);

int main(void) {
  void *rb = universe_ds_ringbuf_create(4, sizeof(int));
  int x;
  while (scanf("%d", &x) == 1) {
    if (universe_ds_ringbuf_push(rb, &x) == 6) { int drop;
      universe_ds_ringbuf_pop(rb, &drop); universe_ds_ringbuf_push(rb, &x); }
  }
  int v;
  while (universe_ds_ringbuf_pop(rb, &v) == 0) printf("%d\n", v);
  universe_ds_ringbuf_destroy(rb);
  return 0;
}
```

```sh
printf '1 2 3 4 5 6\n' | ./rbcli   # -> 3 4 5 6 (last 4)
```

## Notes

- Capacity is fixed and power-of-two rounded; `capacity()` reports the rounded
  value. Push on full returns 6 (FULL).
- Elements are POD at `elem_size` stride. Single-threaded.
