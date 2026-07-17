# structures_assoc/hashmap_swiss

## Purpose

Open-addressing hash map with i64 keys and a caller-fixed value size, using
SIMD tag-group probing (the "swiss" flavor). Control bytes hold a 7-bit hash
tag per slot; a single 16-lane vector compare rejects or locates 16 slots at
once, so a hit or miss touches ~1 cache line of control bytes plus at most a
couple of key loads. Storage is one allocation (control bytes + SoA keys +
values), with a stable 64-byte header so `put` can reallocate on growth without
invalidating the caller's handle. **Choose this** for read-heavy or mixed
workloads on large tables where `get` latency matters; prefer a plain
linear-probe map only for tiny fixed tables where the vector setup does not pay.
For a static, immutable key set, use `mph` instead.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_hashmap_swiss_create(int64_t val_size, int64_t initial_cap)` | Create a map for POD values of `val_size` bytes | handle, or NULL on OOM |
| `int32_t universe_ds_hashmap_swiss_put(void *m, int64_t key, void *val)` | Insert/update `key`, copying `val_size` bytes from `val` | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_hashmap_swiss_get(void *m, int64_t key, void *out)` | Copy the value for `key` into `out` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_hashmap_swiss_contains(void *m, int64_t key)` | Membership test | 1 present, 0 absent |
| `int32_t universe_ds_hashmap_swiss_remove(void *m, int64_t key)` | Delete `key` (tombstone) | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_hashmap_swiss_len(void *m)` | Live entry count | count |
| `int64_t universe_ds_hashmap_swiss_capacity(void *m)` | Slot capacity | count |
| `void universe_ds_hashmap_swiss_destroy(void *m)` | Free the map | — |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Symbols are C ABI, `nounwind`. Declare
and call from another `.ll`:

```llvm
declare ptr @universe_ds_hashmap_swiss_create(i64, i64)
declare i32 @universe_ds_hashmap_swiss_put(ptr, i64, ptr)
declare i32 @universe_ds_hashmap_swiss_get(ptr, i64, ptr)
```

Link:

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A key/value store: `put K V` lines set entries, `get K` prints values.

```c
// swisscli.c — build: clang -O3 swisscli.c build/libuniverse.a -lpthread -lm -o swisscli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_hashmap_swiss_create(int64_t, int64_t);
extern int32_t universe_ds_hashmap_swiss_put(void *, int64_t, void *);
extern int32_t universe_ds_hashmap_swiss_get(void *, int64_t, void *);
extern void universe_ds_hashmap_swiss_destroy(void *);

int main(void) {
  void *m = universe_ds_hashmap_swiss_create(sizeof(int64_t), 16);
  char op[8]; long long k, v;
  while (scanf("%7s %lld", op, &k) == 2) {
    if (!strcmp(op, "put")) { scanf("%lld", &v);
      int64_t vv = v; universe_ds_hashmap_swiss_put(m, k, &vv); }
    else if (!strcmp(op, "get")) { int64_t vv;
      if (universe_ds_hashmap_swiss_get(m, k, &vv) == 0) printf("%lld\n", (long long)vv);
      else printf("(none)\n"); }
  }
  universe_ds_hashmap_swiss_destroy(m);
  return 0;
}
```

```sh
printf 'put 7 100\nget 7\nget 9\n' | ./swisscli   # -> 100 then (none)
```

## Notes

- Values are POD copied by value (`val_size` fixed at create); the map stores no
  pointers into caller memory.
- Single-threaded. For concurrent i64-keyed maps use a sharded tree variant.
- One allocation for table storage; `put` may realloc, but the returned handle
  stays stable. `destroy` frees everything.
