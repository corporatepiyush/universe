# sort/shell — shellsort, Knuth gaps (generic comparator)

## Purpose

Shellsort with Knuth gaps (`h = 3h+1`) over an arbitrary element type via a
caller comparator. In-place, not stable, no allocation. Each gapped insertion
holds the moving element once (stack buffer) and shifts run slots with
whole-element `memcpy`s — no byte-by-byte swapping. A good allocation-free
middle ground between insertion sort and O(N log N) sorts for medium arrays.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_shell(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | In-place shellsort of `count` elements of `elem_size` bytes by `cmp`. | 0 OK, 1 NULL |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_shell(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

```c
// shsort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_shell(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<20]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_shell(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 shsort_cli.c build/libuniverse.a -lpthread -lm -o shsort_cli
printf '5 3 8 1 9 2\n' | ./shsort_cli
```

## Notes

- **Allocation-free, in-place, not stable.** Sub-quadratic on medium arrays
  without the aux buffer merge sort needs.
- **Threading.** Reentrant; no shared state.
