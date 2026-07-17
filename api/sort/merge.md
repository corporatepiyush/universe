# sort/merge — stable bottom-up merge sort (generic comparator)

## Purpose

Merge sort over an arbitrary element type via a caller comparator. O(N log N),
**stable**, one aux allocation. Bottom-up (iterative) with ping-pong buffers:
zero recursion, exactly ceil(log2 N) full linear passes — the sequential access
pattern the prefetcher loves. Roles swap each pass so there is no per-merge
copy-back. Choose this when a stable order or guaranteed O(N log N) is required.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_merge(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | Stable sort `count` elements of `elem_size` bytes at `base` by `cmp`. | 0 OK, 1 NULL, 2 OOM (aux alloc) |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_merge(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Stable-sort integers from stdin (see `sort/quick` for the full pattern; only the
function name changes).

```c
// msort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_merge(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<20]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_merge(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 msort_cli.c build/libuniverse.a -lpthread -lm -o msort_cli
printf '5 3 8 1 9 2\n' | ./msort_cli
```

## Notes

- **Stable + one aux allocation** (may return OOM). If a stable sort is not
  required, `sort/quick` avoids the allocation.
- **Threading.** Reentrant; no shared state.
