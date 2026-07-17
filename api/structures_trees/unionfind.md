# structures_trees/unionfind

## Purpose

Disjoint-set union (union-find) over elements `0..n-1` with near-constant
amortized find/unite via path halving plus union by size. One allocation
(64-byte header + two flat i64 arrays: `parent[]` forest, `size[]` subtree count
at roots). Path halving repoints each node to its grandparent in a single pass
(no recursion, no second pass); union by size keeps trees shallow. `num_sets` is
maintained so `count_sets` is O(1). The standard tool for connectivity,
Kruskal's MST, and grouping.

## Exported API

Some functions return values rather than error codes (documented below).

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_unionfind_create(int64_t n)` | Create `n` singleton sets | handle, or NULL on OOM |
| `int64_t universe_ds_unionfind_find(void *u, int64_t x)` | Representative id of `x`'s set | root id, or -1 (null/out-of-range) |
| `int32_t universe_ds_unionfind_unite(void *u, int64_t a, int64_t b)` | Merge the sets of `a` and `b` | 0 merged, 9 already same set, or error |
| `int32_t universe_ds_unionfind_connected(void *u, int64_t a, int64_t b)` | Same-set test | 1 connected, 0 not, -1 out-of-range |
| `int64_t universe_ds_unionfind_count_sets(void *u)` | Number of disjoint sets | count |
| `int64_t universe_ds_unionfind_set_size(void *u, int64_t x)` | Size of `x`'s set | size, or 0 (null/out-of-range) |
| `void universe_ds_unionfind_destroy(void *u)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_unionfind_create(i64)
declare i32 @universe_ds_unionfind_unite(ptr, i64, i64)
declare i32 @universe_ds_unionfind_connected(ptr, i64, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read `a b` pairs, union them, then print the number of connected components.

```c
// ufcli.c — build: clang -O3 ufcli.c build/libuniverse.a -lpthread -lm -o ufcli
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void *universe_ds_unionfind_create(int64_t);
extern int32_t universe_ds_unionfind_unite(void *, int64_t, int64_t);
extern int64_t universe_ds_unionfind_count_sets(void *);
extern void universe_ds_unionfind_destroy(void *);

int main(int argc, char **argv) {
  int64_t n = argc > 1 ? atoll(argv[1]) : 16;
  void *u = universe_ds_unionfind_create(n);
  long long a, b;
  while (scanf("%lld %lld", &a, &b) == 2) universe_ds_unionfind_unite(u, a, b);
  printf("components=%lld\n", (long long)universe_ds_unionfind_count_sets(u));
  universe_ds_unionfind_destroy(u);
  return 0;
}
```

```sh
printf '0 1\n2 3\n1 2\n' | ./ufcli 5   # -> components=2  ({0,1,2,3}, {4})
```

## Notes

- Elements are fixed ids in `[0, n)`. `unite` returns 9 when the two are already
  in the same set (a note, not an error).
- `find`/`connected`/`set_size` are defensive on out-of-range input (-1 / -1 / 0).
- Single-threaded.
