# structures_linear/sparse_set

## Purpose

Sparse set: O(1) membership, add, remove, and clear over a bounded integer
universe `[0, capacity)`, backed by the classic dense/sparse pair. Two u32
arrays in one allocation — `dense[]` packs the members in insertion order,
`sparse[]` maps a present member `x` to its slot in `dense` with the invariant
`dense[sparse[x]] == x`. That mutual back-pointer is the validity test, so
`sparse[]` never needs initialization and `clear` is O(1) (just reset count).
Iteration over `dense` is a unit-stride cache-dense walk (perfect locality) —
ideal for ECS component sets and per-pass "visited" sets.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_sparseset_create(int64_t capacity)` | Create over universe `[0, capacity)` | handle, or NULL on OOM |
| `int32_t universe_ds_sparseset_add(void *s, int64_t x)` | Add `x` (idempotent) | 0 OK, 7 INVALID_INDEX (x >= capacity) |
| `int32_t universe_ds_sparseset_remove(void *s, int64_t x)` | Remove `x` (swap-with-last in dense) | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_sparseset_clear(void *s)` | Empty the set in O(1) | 0 OK, 1 NULL |
| `int32_t universe_ds_sparseset_contains(void *s, int64_t x)` | Membership | 1 present, 0 absent |
| `int64_t universe_ds_sparseset_size(void *s)` | Member count | count |
| `int64_t universe_ds_sparseset_capacity(void *s)` | Universe size | count |
| `void *universe_ds_sparseset_get_dense(void *s, int64_t *out_len)` | Pointer to the packed member array; writes length to `out_len` | u32* base |
| `void universe_ds_sparseset_destroy(void *s)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_sparseset_create(i64)
declare i32 @universe_ds_sparseset_add(ptr, i64)
declare ptr @universe_ds_sparseset_get_dense(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Add/remove ids, then dump the live set via the dense array.

```c
// sscli.c — build: clang -O3 sscli.c build/libuniverse.a -lpthread -lm -o sscli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_sparseset_create(int64_t);
extern int32_t universe_ds_sparseset_add(void *, int64_t);
extern int32_t universe_ds_sparseset_remove(void *, int64_t);
extern uint32_t *universe_ds_sparseset_get_dense(void *, int64_t *);
extern void universe_ds_sparseset_destroy(void *);

int main(void) {
  void *s = universe_ds_sparseset_create(1024);
  char op[4]; long long x;
  while (scanf("%3s %lld", op, &x) == 2) {
    if (!strcmp(op, "a")) universe_ds_sparseset_add(s, x);
    else if (!strcmp(op, "r")) universe_ds_sparseset_remove(s, x);
  }
  int64_t n; uint32_t *d = universe_ds_sparseset_get_dense(s, &n);
  for (int64_t i = 0; i < n; i++) printf("%u ", d[i]);
  printf("\n");
  universe_ds_sparseset_destroy(s);
  return 0;
}
```

```sh
printf 'a 5 a 9 a 2 r 9\n' | ./sscli   # -> live members (order may vary): 5 2
```

## Notes

- Members must be `< capacity`; capacity is fixed at create.
- `remove` swaps the victim with the last dense element, so iteration order is
  not stable across removals.
- `get_dense` exposes the internal u32 array — do not free or resize it.
  Single-threaded.
