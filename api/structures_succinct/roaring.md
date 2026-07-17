# structures_succinct/roaring

## Purpose

Roaring bitmap: a compressed set of u32 values with fast membership, cardinality,
ascending iteration, and O(n+m) set algebra. A u32 splits into a 16-bit chunk
key and a 16-bit low value; values sharing a chunk key live in one container
holding only the low 16 bits. The top level is a directory of (key, container)
sorted by key (SoA), so two roarings merge in one linear pass. Containers adapt
by cardinality: **array** (sorted u16, sparse, <= 4096) promotes to **bitmap**
(1024 x i64 dense, > 4096). The dense bitmap AND/OR/ANDNOT kernels are portable
`<2 x i64>` SIMD loops (SSE2/NEON, no runtime check), with scalar oracles
exported alongside. Choose this for large, clustered or sparse integer sets
(id sets, filters, postings) where set algebra and membership dominate.

## Exported API

### Set operations

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_roaring_create(void)` | Create an empty bitmap | handle, or NULL on OOM |
| `int32_t universe_ds_roaring_add(void *r, int32_t v)` | Add `v` (idempotent) | 0 OK, 1 NULL, 2 OOM |
| `int32_t universe_ds_roaring_remove(void *r, int32_t v)` | Remove `v` (absent is OK) | 0 OK, or error |
| `int32_t universe_ds_roaring_contains(void *r, int32_t v)` | Membership | 1 present, 0 absent (0 if null) |
| `int64_t universe_ds_roaring_cardinality(void *r)` | Number of values | count (0 if null) |
| `int64_t universe_ds_roaring_to_array(void *r, void *out_u32, int64_t out_cap)` | Copy values ascending into `out` (`uint32_t*`) | total count |
| `int32_t universe_ds_roaring_container_kind(void *r, int32_t v)` | Container kind for `v`'s chunk | 0 array, 1 bitmap, -1 none |
| `void *universe_ds_roaring_union(void *a, void *b)` | New bitmap `a \| b` | handle, or NULL on OOM |
| `void *universe_ds_roaring_intersection(void *a, void *b)` | New bitmap `a & b` | handle, or NULL on OOM |
| `void *universe_ds_roaring_difference(void *a, void *b)` | New bitmap `a & ~b` | handle, or NULL on OOM |
| `void universe_ds_roaring_destroy(void *r)` | Free | — |

### Dense block kernels (low-level, `ptr` = 1024 x i64; `dst` may alias `a`/`b`)

| C signature | Description |
|---|---|
| `void universe_ds_roaring_bitmap_or(void *dst, void *a, void *b)` | SIMD `dst = a \| b` |
| `void universe_ds_roaring_bitmap_and(void *dst, void *a, void *b)` | SIMD `dst = a & b` |
| `void universe_ds_roaring_bitmap_andnot(void *dst, void *a, void *b)` | SIMD `dst = a & ~b` |
| `void universe_ds_roaring_bitmap_or_scalar(void *dst, void *a, void *b)` | Scalar oracle for `or` |
| `void universe_ds_roaring_bitmap_and_scalar(void *dst, void *a, void *b)` | Scalar oracle for `and` |
| `void universe_ds_roaring_bitmap_andnot_scalar(void *dst, void *a, void *b)` | Scalar oracle for `andnot` |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_roaring_create()
declare i32 @universe_ds_roaring_add(ptr, i32)
declare i64 @universe_ds_roaring_cardinality(ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Add u32 values from stdin, print cardinality and the sorted set.

```c
// roarcli.c — build: clang -O3 roarcli.c build/libuniverse.a -lpthread -lm -o roarcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_roaring_create(void);
extern int32_t universe_ds_roaring_add(void *, int32_t);
extern int64_t universe_ds_roaring_cardinality(void *);
extern int64_t universe_ds_roaring_to_array(void *, void *, int64_t);
extern void universe_ds_roaring_destroy(void *);

int main(void) {
  void *r = universe_ds_roaring_create();
  long long v;
  while (scanf("%lld", &v) == 1) universe_ds_roaring_add(r, (int32_t)v);
  int64_t n = universe_ds_roaring_cardinality(r);
  static uint32_t out[4096];
  int64_t got = universe_ds_roaring_to_array(r, out, 4096);
  printf("card=%lld:", (long long)n);
  for (int64_t i = 0; i < got; i++) printf(" %u", out[i]);
  printf("\n");
  universe_ds_roaring_destroy(r);
  return 0;
}
```

```sh
printf '5 1 5 100000 3\n' | ./roarcli   # card=4: 1 3 5 100000
```

## Notes

- Values are u32; duplicates are ignored (`add` idempotent). `to_array` yields
  ascending order and returns the true count (may exceed `out_cap` — size the
  buffer to `cardinality`).
- `union`/`intersection`/`difference` allocate a new bitmap the caller must
  `destroy`.
- The dense block kernels operate on raw 1024-word (65536-bit) blocks and are
  the SIMD primitives behind container merges; `dst` may alias an input.
  Single-threaded.
