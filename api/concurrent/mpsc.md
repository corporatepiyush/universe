# concurrent/mpsc — multi-producer / single-consumer queues

## Purpose
Lock-free multi-producer / single-consumer queues. Two variants share one
module: a BOUNDED ring (`universe_conc_mpsc_ring_*`) and an UNBOUNDED
directory-paged segment queue (`universe_conc_mpsc_seg_*`). The multi-producer
contention point — the producer position counter — is claimed with ONE
`atomicrmw add` (`ldadd`/`lock xadd`), wait-free on the claim, no CAS retry loop.
Data correctness rides on a per-slot release/acquire handshake, never on the
monotonic counter. The bounded ring detects FULL with a pre-claim peek (a
returned FULL consumes no ticket, so producers may retry). The unbounded seg
maps `chunkIdx = pos>>10` to a chunk through a flat directory (no linked-list
traversal, no front-reclamation UAF); the single consumer recycles fully-drained
chunks through a tiny spinlock-guarded freelist (cold, once per 1024 items).
Choose MPSC for many producers feeding one consumer (log sinks, work funnels).

## Exported API
Push/pop return `int32_t`: `0` OK, `6` FULL, `4` EMPTY, `1` NULL. Elements are
fixed `elem_size` bytes, copied by value.

Bounded ring (power-of-two capacity):
| C signature | Description |
|---|---|
| `void *universe_conc_mpsc_ring_create(int64_t capacity, int64_t elem_size)` | Create a bounded MPSC ring |
| `int32_t universe_conc_mpsc_ring_push(void *rb, const void *elem)` | Enqueue (any producer thread) |
| `int32_t universe_conc_mpsc_ring_pop(void *rb, void *out)` | Dequeue (single consumer) |
| `int64_t universe_conc_mpsc_ring_count(void *rb)` | Approximate count |
| `int64_t universe_conc_mpsc_ring_capacity(void *rb)` | Capacity |
| `void universe_conc_mpsc_ring_destroy(void *rb)` | Free |

Unbounded segment queue:
| C signature | Description |
|---|---|
| `void *universe_conc_mpsc_seg_create(int64_t elem_size)` | Create an unbounded MPSC queue |
| `int32_t universe_conc_mpsc_seg_push(void *q, const void *elem)` | Enqueue (any producer); FULL only at ~64M live-lifetime items |
| `int32_t universe_conc_mpsc_seg_pop(void *q, void *out)` | Dequeue (single consumer) |
| `int64_t universe_conc_mpsc_seg_count(void *q)` | Approximate count |
| `void universe_conc_mpsc_seg_destroy(void *q)` | Free |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_conc_mpsc_ring_create(i64, i64)
declare i32 @universe_conc_mpsc_ring_push(ptr, ptr)
declare i32 @universe_conc_mpsc_ring_pop(ptr, ptr)
declare void @universe_conc_mpsc_ring_destroy(ptr)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`mpsc_demo.c` — T producer threads, one consumer, verify conservation.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_conc_mpsc_ring_create(int64_t capacity, int64_t elem_size);
int32_t universe_conc_mpsc_ring_push(void *rb, const void *elem);
int32_t universe_conc_mpsc_ring_pop(void *rb, void *out);
void    universe_conc_mpsc_ring_destroy(void *rb);

#define PER 100000
static int T = 4;
static void *producer(void *q) {
    int64_t v = 1;
    for (int i = 0; i < PER; i++) while (universe_conc_mpsc_ring_push(q, &v) != 0) ;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) T = atoi(argv[1]);
    void *q = universe_conc_mpsc_ring_create(4096, sizeof(int64_t));
    pthread_t th[64];
    for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, producer, q);
    int64_t total = (int64_t)T * PER, got = 0, sum = 0, v;
    while (got < total) if (universe_conc_mpsc_ring_pop(q, &v) == 0) { sum += v; got++; }
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    printf("sum=%lld want=%lld %s\n", (long long)sum, (long long)total,
           sum == total ? "OK" : "FAIL");
    universe_conc_mpsc_ring_destroy(q);
    return 0;
}
```
```
clang -O3 mpsc_demo.c build/libuniverse.a -lpthread -lm -o mpsc_demo
./mpsc_demo 8
```

## Notes
- MANY producer threads, exactly ONE consumer thread.
- The bounded ring is lossless on FULL (no ticket is consumed); the unbounded
  seg is effectively unbounded (~64M live-lifetime items before FULL).
- `count` is an approximate snapshot. Element size fixed at create; copied by value.
