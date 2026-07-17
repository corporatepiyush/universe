# structures_linear/lru

## Purpose

LRU cache with i64 keys and fixed-size values, O(1) `get`/`put` with eviction.
One allocation, struct-of-arrays: keys, bucket-chain links, recency prev/next,
values, and bucket heads — all i32 index links (no pointer webs), -1 sentinel.
The hash table is open chaining over a 2x-capacity power-of-two bucket array
(load factor <= 0.5) with a splitmix64 hash; the recency list is an intrusive
index doubly-linked list, so move-to-front touches at most 6 i32 stores. When
full, `put` reuses the evicted victim's slot in place (zero allocation churn at
steady state).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_lru_create(int64_t capacity, int64_t val_size)` | Create a cache holding `capacity` entries of `val_size` bytes | handle, or NULL on OOM |
| `int32_t universe_ds_lru_put(void *c, int64_t key, void *val)` | Insert/update `key`; may evict the LRU entry | 0 OK, 1 NULL |
| `int32_t universe_ds_lru_get(void *c, int64_t key, void *out)` | Fetch value and mark most-recently-used | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_lru_contains(void *c, int64_t key)` | Membership (does not update recency) | 1 present, 0 absent |
| `int64_t universe_ds_lru_count(void *c)` | Live entry count | count |
| `void universe_ds_lru_destroy(void *c)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_lru_create(i64, i64)
declare i32 @universe_ds_lru_put(ptr, i64, ptr)
declare i32 @universe_ds_lru_get(ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A tiny fixed-capacity cache: `put K V` / `get K`, observing eviction.

```c
// lrucli.c — build: clang -O3 lrucli.c build/libuniverse.a -lpthread -lm -o lrucli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_lru_create(int64_t, int64_t);
extern int32_t universe_ds_lru_put(void *, int64_t, void *);
extern int32_t universe_ds_lru_get(void *, int64_t, void *);
extern void universe_ds_lru_destroy(void *);

int main(void) {
  void *c = universe_ds_lru_create(2 /* cap */, sizeof(int));
  char op[8]; long long k; int v;
  while (scanf("%7s %lld", op, &k) == 2) {
    if (!strcmp(op, "put")) { scanf("%d", &v); universe_ds_lru_put(c, k, &v); }
    else if (!strcmp(op, "get")) {
      if (universe_ds_lru_get(c, k, &v) == 0) printf("%d\n", v);
      else printf("(evicted)\n"); }
  }
  universe_ds_lru_destroy(c);
  return 0;
}
```

```sh
printf 'put 1 10\nput 2 20\nput 3 30\nget 1\nget 3\n' | ./lrucli  # (evicted) then 30
```

## Notes

- Capacity and `val_size` are fixed at create; `put` past capacity evicts the
  least-recently-used entry.
- `get` updates recency; `contains` does not.
- Single-threaded.
