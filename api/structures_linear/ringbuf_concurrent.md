# structures_linear/ringbuf_concurrent

## Purpose

Single-producer / single-consumer (SPSC) lock-free ring buffer. No mutex: the
producer owns `tail`, the consumer owns `head`, and each side reads the other's
index with an acquire load only when its cached copy says the ring might be
full/empty. In steady state a push is one monotonic load of your own index, a
cached compare, a memcpy, and a release store — no shared-line ping-pong.
Producer and consumer state sit 128 B apart (distinct cache lines). **Use with
exactly one producer thread and one consumer thread**; for single-threaded use
`ringbuf`, for multi-producer/consumer use a dedicated MPMC queue.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_cringbuf_create(int64_t capacity, int64_t elem_size)` | Create an SPSC ring (capacity rounded to pow2) | handle, or NULL on OOM |
| `int32_t universe_ds_cringbuf_push(void *rb, void *elem)` | Producer: enqueue one element | 0 OK, 1 NULL, 6 FULL |
| `int32_t universe_ds_cringbuf_pop(void *rb, void *out)` | Consumer: dequeue one element | 0 OK, 4 EMPTY |
| `int64_t universe_ds_cringbuf_count(void *rb)` | Approximate element count | count |
| `int64_t universe_ds_cringbuf_capacity(void *rb)` | Rounded capacity | count |
| `void universe_ds_cringbuf_destroy(void *rb)` | Free (after both threads quiesce) | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_cringbuf_create(i64, i64)
declare i32 @universe_ds_cringbuf_push(ptr, ptr)
declare i32 @universe_ds_cringbuf_pop(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

One producer streams ints to one consumer that sums them.

```c
// crbcli.c — build: clang -O3 crbcli.c build/libuniverse.a -lpthread -lm -o crbcli
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
extern void *universe_ds_cringbuf_create(int64_t, int64_t);
extern int32_t universe_ds_cringbuf_push(void *, void *);
extern int32_t universe_ds_cringbuf_pop(void *, void *);
extern void universe_ds_cringbuf_destroy(void *);

#define N 1000000
static long long total;
static void *consumer(void *rb) {
  int v; long got = 0;
  while (got < N) if (universe_ds_cringbuf_pop(rb, &v) == 0) { total += v; got++; }
  return 0;
}
int main(void) {
  void *rb = universe_ds_cringbuf_create(1024, sizeof(int));
  pthread_t c; pthread_create(&c, 0, consumer, rb);
  for (int i = 0; i < N; i++) while (universe_ds_cringbuf_push(rb, &i) == 6) {}
  pthread_join(c, 0);
  printf("sum=%lld\n", total);   // 0+1+...+(N-1) = 499999500000
  universe_ds_cringbuf_destroy(rb);
  return 0;
}
```

```sh
./crbcli   # sum=499999500000
```

## Notes

- Exactly one producer and one consumer. Multiple producers or consumers are
  undefined behavior — this is SPSC only.
- `count` is a lock-free approximation; do not use it to gate correctness.
  Push on full returns 6, pop on empty returns 4.
- Link with `-lpthread`.
