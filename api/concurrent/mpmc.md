# concurrent/mpmc — bounded MPMC and SPMC queues

## Purpose
Lock-free BOUNDED queues with per-slot sequence numbers — the SPMC and MPMC
members of the queue family. A monotone shared index alone cannot safely gate a
slot's non-atomic payload (that is a data race); each cell embeds an atomic
sequence number that carries the happens-before edge. A producer writes slot `p`
only when `seq == p` (claims `p` by advancing the enqueue index, memcpys, then
RELEASE-stores `seq = p+1` to publish); a consumer reads slot `p` only when
`seq == p+1` (ACQUIRE-loads the seq, memcpys out, RELEASE-stores `seq = p+cap`).
FULL and EMPTY fall straight out of the seq comparison — no separate flags, no
seq_cst anywhere. In the SPMC specialization the single producer owns the tail
(no CAS on the write side); in MPMC both ends CAS their shared index. Choose
MPMC when many threads on both ends; SPMC when one producer fans out to many
consumers.

## Exported API
Enqueue/dequeue return `int32_t`: `0` OK, `6` FULL, `4` EMPTY, `1` NULL.
Capacity is rounded to a power of two; elements are fixed `elem_size` bytes.

MPMC (multi-producer, multi-consumer):
| C signature | Description |
|---|---|
| `void *universe_conc_mpmc_create(int64_t capacity, int64_t elem_size)` | Create a bounded MPMC ring |
| `int32_t universe_conc_mpmc_enqueue(void *q, const void *elem)` | Enqueue (any producer) |
| `int32_t universe_conc_mpmc_dequeue(void *q, void *out)` | Dequeue (any consumer) |
| `int64_t universe_conc_mpmc_count(void *q)` | Approximate count |
| `int64_t universe_conc_mpmc_capacity(void *q)` | Capacity |
| `void universe_conc_mpmc_destroy(void *q)` | Free |

SPMC (single-producer, multi-consumer):
| C signature | Description |
|---|---|
| `void *universe_conc_spmc_create(int64_t capacity, int64_t elem_size)` | Create a bounded SPMC ring |
| `int32_t universe_conc_spmc_enqueue(void *q, const void *elem)` | Enqueue (single producer, wait-free) |
| `int32_t universe_conc_spmc_dequeue(void *q, void *out)` | Dequeue (any consumer) |
| `int64_t universe_conc_spmc_count(void *q)` | Approximate count |
| `int64_t universe_conc_spmc_capacity(void *q)` | Capacity |
| `void universe_conc_spmc_destroy(void *q)` | Free |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_conc_mpmc_create(i64, i64)
declare i32 @universe_conc_mpmc_enqueue(ptr, ptr)
declare i32 @universe_conc_mpmc_dequeue(ptr, ptr)
declare void @universe_conc_mpmc_destroy(ptr)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`mpmc_demo.c` — P producers + C consumers, verify total items conserved.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

void   *universe_conc_mpmc_create(int64_t capacity, int64_t elem_size);
int32_t universe_conc_mpmc_enqueue(void *q, const void *elem);
int32_t universe_conc_mpmc_dequeue(void *q, void *out);
void    universe_conc_mpmc_destroy(void *q);

#define PER 100000
static int P = 4, C = 4;
static int64_t consumed = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int producers_done = 0;

static void *prod(void *q) {
    int64_t v = 1;
    for (int i = 0; i < PER; i++) while (universe_conc_mpmc_enqueue(q, &v) != 0) ;
    return NULL;
}
static void *cons(void *q) {
    int64_t v, local = 0;
    for (;;) {
        if (universe_conc_mpmc_dequeue(q, &v) == 0) local += v;
        else if (producers_done) { if (universe_conc_mpmc_dequeue(q, &v) != 0) break; else local += v; }
    }
    pthread_mutex_lock(&mu); consumed += local; pthread_mutex_unlock(&mu);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 2) { P = atoi(argv[1]); C = atoi(argv[2]); }
    void *q = universe_conc_mpmc_create(4096, sizeof(int64_t));
    pthread_t pt[64], ct[64];
    for (int i = 0; i < C; i++) pthread_create(&ct[i], NULL, cons, q);
    for (int i = 0; i < P; i++) pthread_create(&pt[i], NULL, prod, q);
    for (int i = 0; i < P; i++) pthread_join(pt[i], NULL);
    producers_done = 1;
    for (int i = 0; i < C; i++) pthread_join(ct[i], NULL);
    int64_t want = (int64_t)P * PER;
    printf("consumed=%lld want=%lld %s\n", (long long)consumed, (long long)want,
           consumed == want ? "OK" : "FAIL");
    universe_conc_mpmc_destroy(q);
    return 0;
}
```
```
clang -O3 mpmc_demo.c build/libuniverse.a -lpthread -lm -o mpmc_demo
./mpmc_demo 4 4
```

## Notes
- Bounded, single allocation; producer and consumer indices sit on separate
  128 B lines. Capacity rounds up to a power of two.
- `count` is an approximate snapshot. Element size fixed at create; copied by value.
- SPMC requires exactly one producer thread (the producer side is wait-free);
  MPMC allows many on both ends.
