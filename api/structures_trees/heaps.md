# structures_trees/heaps

## Purpose

A family of three priority-queue variants over i64 keys, single thread — pick the
class by workload (see `docs/performance-principles.md`):

- **d-ary heap** (`dheap_*`) — general priority queue, default D=4. An implicit
  heap in a flat i64 array with D children per node, so the tree is shallower
  (log_D n) and one sift step scans a single cache line (D=4 => 32 B). Choose D=4
  for balanced push/pop, D=8+ when pops dominate. Min-heap over signed i64
  (max-heap: negate keys, avoiding INT64_MIN). Keys only (no value).
- **radix / monotone heap** (`radixheap_*`) — Dijkstra with **bounded
  non-negative integer** keys under a monotone-pop precondition (no key smaller
  than the last extracted min is inserted). O(1) amortized ops, key+value. A push
  violating monotonicity returns INVALID_ARG.
- **pairing heap** (`pairing_*`) — key+value PQ with O(1) amortized
  **decrease-key** and meld, ideal for dense-graph Dijkstra. Push returns a node
  handle used by `decrease_key`.

## Exported API

### d-ary heap

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_dheap_create(int64_t initial_cap, int64_t d)` | Create a D-ary min-heap | handle, or NULL on OOM |
| `int32_t universe_ds_dheap_push(void *h, int64_t key)` | Insert `key` | 0 OK, 1, 2, 3 |
| `int32_t universe_ds_dheap_pop(void *h, int64_t *out)` | Remove the minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_dheap_peek(void *h, int64_t *out)` | Read the minimum | 0 OK, 4 EMPTY |
| `int64_t universe_ds_dheap_len(void *h)` | Element count | count |
| `void universe_ds_dheap_destroy(void *h)` | Free | — |

### radix / monotone heap

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_radixheap_create(int64_t initial_cap)` | Create a monotone heap | handle, or NULL on OOM |
| `int32_t universe_ds_radixheap_push(void *h, int64_t key, int64_t val)` | Insert; `key < last_popped` -> 8 INVALID_ARG | 0 OK, 8 |
| `int32_t universe_ds_radixheap_pop(void *h, int64_t *out_key, int64_t *out_val)` | Remove the minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_radixheap_peek(void *h, int64_t *out_key)` | Read the minimum key | 0 OK, 4 EMPTY |
| `int64_t universe_ds_radixheap_len(void *h)` | Element count | count |
| `void universe_ds_radixheap_destroy(void *h)` | Free | — |

### pairing heap

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_pairing_create(void)` | Create a pairing heap | handle, or NULL on OOM |
| `int64_t universe_ds_pairing_push(void *h, int64_t key, int64_t val)` | Insert; returns a node handle | node id, or <0 on OOM |
| `int32_t universe_ds_pairing_peek(void *h, int64_t *ok, int64_t *ov)` | Read the minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_pairing_pop(void *h, int64_t *ok, int64_t *ov)` | Remove the minimum | 0 OK, 4 EMPTY |
| `int32_t universe_ds_pairing_decrease_key(void *h, int64_t node, int64_t newkey)` | Lower `node`'s key | 0 OK, 8 INVALID_ARG |
| `int32_t universe_ds_pairing_meld(void *dst, void *src)` | Merge `src` into `dst` | 0 OK, or error |
| `int64_t universe_ds_pairing_len(void *h)` | Element count | count |
| `void universe_ds_pairing_destroy(void *h)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_dheap_create(i64, i64)
declare i32 @universe_ds_dheap_push(ptr, i64)
declare i32 @universe_ds_dheap_pop(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Sort stdin ints with a 4-ary heap.

```c
// dhcli.c — build: clang -O3 dhcli.c build/libuniverse.a -lpthread -lm -o dhcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_dheap_create(int64_t, int64_t);
extern int32_t universe_ds_dheap_push(void *, int64_t);
extern int32_t universe_ds_dheap_pop(void *, int64_t *);
extern void universe_ds_dheap_destroy(void *);

int main(void) {
  void *h = universe_ds_dheap_create(16, 4);
  long long x;
  while (scanf("%lld", &x) == 1) universe_ds_dheap_push(h, x);
  int64_t v;
  while (universe_ds_dheap_pop(h, &v) == 0) printf("%lld\n", (long long)v);
  universe_ds_dheap_destroy(h);
  return 0;
}
```

```sh
printf '9 3 7 1 5\n' | ./dhcli   # -> 1 3 5 7 9
```

## Notes

- All three are min-heaps. d-ary keys are bare i64; radix/pairing carry a value.
- Radix heap requires non-negative keys and monotone (non-decreasing) pops —
  designed for bounded-weight Dijkstra. Pairing heap's `decrease_key` uses the
  node handle returned by `push`.
- Single-threaded.
