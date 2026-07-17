# structures_assoc/treemap

## Purpose

Ordered map (i64 key -> i64 value), the **array-backed** flavor: keys are kept
in one flat ascending run, so every point query is a branch-lean binary search
and every ordered query (floor/ceiling/higher/lower/range/min/max) is index
arithmetic over that run. Keys and values are stored SoA in a single allocation
so an in-order range scan is a flat sequential walk (prefetcher-friendly), and
search streams keys with no pointer chasing. Total order is **unsigned** i64
(`icmp ult`). **Choose this** when ordered/nearest-key queries and range scans
dominate and inserts are moderate; `put`/`delete` are O(n) memmoves, so prefer a
hash map when you need no ordering. For a concurrent ordered map use
`treemap_sharded`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_treemap_create(int64_t initial_cap)` | Create an ordered map | handle, or NULL on OOM |
| `int32_t universe_ds_treemap_put(void *m, int64_t key, int64_t val)` | Insert/update | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_treemap_get(void *m, int64_t key, int64_t *out_v)` | Point lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_contains(void *m, int64_t key)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_treemap_delete(void *m, int64_t key)` | Remove key | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_treemap_size(void *m)` | Entry count | count |
| `int32_t universe_ds_treemap_ceiling(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Smallest key >= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_higher(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Smallest key > `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_floor(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Largest key <= `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_lower(void *m, int64_t key, int64_t *out_k, int64_t *out_v)` | Largest key < `key` | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_treemap_min(void *m, int64_t *out_k, int64_t *out_v)` | Smallest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_treemap_max(void *m, int64_t *out_k, int64_t *out_v)` | Largest entry | 0 OK, 4 EMPTY |
| `int32_t universe_ds_treemap_first(void *m, int64_t *out_k, int64_t *out_v)` | Alias of min | 0 OK, 4 EMPTY |
| `int32_t universe_ds_treemap_last(void *m, int64_t *out_k, int64_t *out_v)` | Alias of max | 0 OK, 4 EMPTY |
| `int64_t universe_ds_treemap_range(void *m, int64_t lo, int64_t hi, int64_t *out_k, int64_t *out_v, int64_t out_cap)` | Copy entries with `lo <= key < hi` into parallel arrays | count copied |
| `void universe_ds_treemap_foreach(void *m, void *fn, void *ctx)` | In-order iterate; `fn` is `void(*)(void *ctx, int64_t key, int64_t val)` | — |
| `void universe_ds_treemap_destroy(void *m)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_treemap_create(i64)
declare i32 @universe_ds_treemap_put(ptr, i64, i64)
declare i32 @universe_ds_treemap_floor(ptr, i64, ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Insert numbers, then answer floor queries.

```c
// tmcli.c — build: clang -O3 tmcli.c build/libuniverse.a -lpthread -lm -o tmcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_treemap_create(int64_t);
extern int32_t universe_ds_treemap_put(void *, int64_t, int64_t);
extern int32_t universe_ds_treemap_floor(void *, int64_t, int64_t *, int64_t *);
extern void universe_ds_treemap_destroy(void *);

int main(void) {
  void *m = universe_ds_treemap_create(16);
  char op[8]; long long k;
  while (scanf("%7s %lld", op, &k) == 2) {
    if (!strcmp(op, "add")) universe_ds_treemap_put(m, k, k * 10);
    else if (!strcmp(op, "floor")) { int64_t ok, ov;
      if (universe_ds_treemap_floor(m, k, &ok, &ov) == 0)
        printf("floor(%lld)=%lld val=%lld\n", k, (long long)ok, (long long)ov);
      else printf("floor(%lld)=none\n", k); }
  }
  universe_ds_treemap_destroy(m);
  return 0;
}
```

```sh
printf 'add 10\nadd 20\nadd 30\nfloor 25\n' | ./tmcli   # -> floor(25)=20 val=200
```

## Notes

- Keys ordered **unsigned**; for signed order bias keys by +2^63 on the way in
  and out.
- `put`/`delete` shift with one memmove: O(n). Ordered/point queries are
  O(log n) binary searches.
- Single-threaded. Values are plain i64.
