# concurrent/spsc — single-producer / single-consumer queues

## Purpose
Wait-free single-producer / single-consumer queues, first member of the
concurrent-queue family. Two structures share one module:
a BOUNDED ring (`universe_conc_spsc_*`) and an UNBOUNDED segment queue
(`universe_conc_spsc_seg_*`). In the ring the producer OWNS `tail`, the consumer
OWNS `head`; each reads only its own index and a private cached copy of the
peer's, refreshing (an acquire load) only when the cache claims full/empty — so
a hot op is one monotonic load, one cached compare, one memcpy, one release
store: no shared line touched, no CAS, no retry (wait-free). The segment queue
gets the same in-chunk speed across 1024-slot chunks linked and pooled
SPSC-safely (retired chunks are recycled producer←consumer through an embedded
wait-free ring). Choose SPSC when exactly one thread enqueues and one dequeues.

## Exported API
Enqueue/dequeue return `int32_t`: `0` OK, `6` FULL, `4` EMPTY, `1` NULL.
Elements are fixed `elem_size` bytes, copied by value.

Bounded ring (power-of-two capacity):
| C signature | Description |
|---|---|
| `void *universe_conc_spsc_create(int64_t capacity, int64_t elem_size)` | Create a bounded ring |
| `int32_t universe_conc_spsc_enqueue(void *rb, const void *elem)` | Copy in one element (producer) |
| `int32_t universe_conc_spsc_dequeue(void *rb, void *out)` | Copy out one element (consumer) |
| `int64_t universe_conc_spsc_count(void *rb)` | Approximate element count |
| `int32_t universe_conc_spsc_is_empty(void *rb)` | 1 if empty |
| `int32_t universe_conc_spsc_is_full(void *rb)` | 1 if full |
| `int64_t universe_conc_spsc_capacity(void *rb)` | Capacity |
| `void universe_conc_spsc_destroy(void *rb)` | Free |

Unbounded segment queue:
| C signature | Description |
|---|---|
| `void *universe_conc_spsc_seg_create(int64_t elem_size)` | Create an unbounded segment queue |
| `int32_t universe_conc_spsc_seg_enqueue(void *q, const void *elem)` | Enqueue (grows as needed) |
| `int32_t universe_conc_spsc_seg_dequeue(void *q, void *out)` | Dequeue |
| `int64_t universe_conc_spsc_seg_count(void *q)` | Approximate count |
| `int32_t universe_conc_spsc_seg_is_empty(void *q)` | 1 if empty |
| `void universe_conc_spsc_seg_destroy(void *q)` | Free |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_conc_spsc_create(i64, i64)
declare i32 @universe_conc_spsc_enqueue(ptr, ptr)
declare i32 @universe_conc_spsc_dequeue(ptr, ptr)
declare void @universe_conc_spsc_destroy(ptr)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`spsc_demo.c` — one producer thread, one consumer thread, verify FIFO sum.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_conc_spsc_create(int64_t capacity, int64_t elem_size);
int32_t universe_conc_spsc_enqueue(void *rb, const void *elem);
int32_t universe_conc_spsc_dequeue(void *rb, void *out);
void    universe_conc_spsc_destroy(void *rb);

#define N 1000000
static void *producer(void *q) {
    for (int64_t i = 0; i < N; i++)
        while (universe_conc_spsc_enqueue(q, &i) != 0) ;   /* spin on FULL */
    return NULL;
}

int main(void) {
    void *q = universe_conc_spsc_create(1024, sizeof(int64_t));
    pthread_t p; pthread_create(&p, NULL, producer, q);
    int64_t v, sum = 0, got = 0;
    while (got < N) if (universe_conc_spsc_dequeue(q, &v) == 0) { sum += v; got++; }
    pthread_join(p, NULL);
    int64_t want = (int64_t)(N - 1) * N / 2;
    printf("sum=%lld want=%lld %s\n", (long long)sum, (long long)want,
           sum == want ? "OK" : "FAIL");
    universe_conc_spsc_destroy(q);
    return 0;
}
```
```
clang -O3 spsc_demo.c build/libuniverse.a -lpthread -lm -o spsc_demo
./spsc_demo
```

## Notes
- Exactly ONE producer thread and ONE consumer thread — not safe with more.
- `count`/`is_empty`/`is_full` are approximate snapshots under concurrency.
- Bounded ring is a single allocation, 128 B producer/consumer line separation;
  the segment queue grows in 1024-slot chunks and pools retired chunks.
- Element size is fixed at create; elements are copied by value (memcpy).
