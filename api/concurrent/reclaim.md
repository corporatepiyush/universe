# concurrent/reclaim — lock-free memory reclamation toolkit

## Purpose
A lock-free MEMORY-RECLAMATION toolkit — it answers "when is it safe to free a
node a concurrent lock-free reader might still be dereferencing?", the hard part
of every lock-free node-based structure (Treiber stack, Michael-Scott queue,
concurrent skiplist/ART). Three independent, composable schemes:
(1) EPOCH-BASED RECLAMATION (EBR) — near-zero reader cost (one relaxed
load + one relaxed store + one seq_cst fence per critical section), the default
for read-heavy structures; (2) HAZARD POINTERS (HP) — bounded memory, per-access
publish cost; (3) SEQLOCK — read-mostly small POD, readers do no writes. Callers
pass a small dense thread id `tid` in `[0, 64)` (deliberately avoiding TLS, which
is a portability minefield); each thread's control blocks live on their own cache
lines.

## Exported API
Epoch-based reclamation (EBR):
| C signature | Description |
|---|---|
| `void universe_conc_ebr_register(int64_t tid)` | Register thread `tid` (once, before use) |
| `void universe_conc_ebr_enter(int64_t tid)` | Pin (begin a read critical section) |
| `void universe_conc_ebr_exit(int64_t tid)` | Unpin (end the section) |
| `void universe_conc_ebr_retire(int64_t tid, void *node, void (*freefn)(void *))` | Defer freeing `node` |
| `int32_t universe_conc_ebr_try_advance(int64_t tid)` | Try to advance the global epoch; 1 advanced, 0 not |
| `void universe_conc_ebr_collect_all(void)` | Drain all deferred frees (quiescent) |
| `int64_t universe_conc_ebr_epoch(void)` | Current global epoch (introspection) |
| `void universe_conc_ebr_reset(void)` | Reset all EBR state (test teardown) |

Hazard pointers (HP), K=8 slots per thread:
| C signature | Description |
|---|---|
| `void universe_conc_hp_protect(int64_t tid, int64_t k, void *p)` | Publish `p` into hazard slot `k` |
| `void universe_conc_hp_clear(int64_t tid, int64_t k)` | Clear hazard slot `k` |
| `void universe_conc_hp_retire(int64_t tid, void *node, void (*freefn)(void *))` | Retire `node`; freed once unhazarded |
| `void universe_conc_hp_collect(int64_t tid)` | Force a scan + free of the retire list |
| `void universe_conc_hp_reset(void)` | Reset all HP state (test teardown) |

SeqLock (`s` points at an 8-byte atomic version counter):
| C signature | Description |
|---|---|
| `void universe_conc_seqlock_write_begin(void *s)` | Begin a write (version → odd) |
| `void universe_conc_seqlock_write_end(void *s)` | End a write (version → even) |
| `int64_t universe_conc_seqlock_read_begin(void *s)` | Snapshot version before reading |
| `bool universe_conc_seqlock_read_retry(void *s, int64_t prev)` | True if the reader must retry |

## Use in an LLVM-based environment
```llvm
declare void @universe_conc_ebr_register(i64)
declare void @universe_conc_ebr_enter(i64)
declare void @universe_conc_ebr_exit(i64)
declare void @universe_conc_ebr_retire(i64, ptr, ptr)
declare i32  @universe_conc_ebr_try_advance(i64)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`seqlock_demo.c` — one writer thread mutates a small record, readers snapshot it
consistently.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

void    universe_conc_seqlock_write_begin(void *s);
void    universe_conc_seqlock_write_end(void *s);
int64_t universe_conc_seqlock_read_begin(void *s);
bool    universe_conc_seqlock_read_retry(void *s, int64_t prev);

static _Alignas(8) int64_t seq = 0;   /* 8-byte version counter */
static int64_t a = 0, b = 0;          /* the protected record (a == b invariant) */
static volatile int stop = 0;

static void *writer(void *_) {
    for (int64_t i = 1; !stop; i++) {
        universe_conc_seqlock_write_begin(&seq);
        a = i; b = i;
        universe_conc_seqlock_write_end(&seq);
    }
    return NULL;
}

int main(void) {
    pthread_t w; pthread_create(&w, NULL, writer, NULL);
    int64_t bad = 0;
    for (int i = 0; i < 2000000; i++) {
        int64_t v = universe_conc_seqlock_read_begin(&seq);
        int64_t ra = a, rb = b;
        if (universe_conc_seqlock_read_retry(&seq, v)) continue;
        if (ra != rb) bad++;
    }
    stop = 1; pthread_join(w, NULL);
    printf("inconsistent reads = %lld (want 0)\n", (long long)bad);
    return bad ? 1 : 0;
}
```
```
clang -O3 seqlock_demo.c build/libuniverse.a -lpthread -lm -o seqlock_demo
./seqlock_demo
```

## Notes
- Thread ids must be dense in `[0, 64)` (MAX_THREADS = 64); `register` once per
  thread before EBR use.
- EBR: `retire` binds the deferred free to the thread's CURRENT pinned epoch;
  reclamation is owner-self-sequenced (never a cross-thread double-free).
- HP: 8 hazard slots per thread; retained memory is bounded by total live hazards.
- SeqLock: writers must be serialized by the caller (single writer or external
  lock); `s` is a single 8-byte atomic counter you allocate.
- `_reset` helpers exist for test teardown; not for use under live concurrency.
