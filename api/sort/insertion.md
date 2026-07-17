# sort/insertion — insertion sort (generic comparator)

## Purpose

Insertion sort over an arbitrary element type via a caller comparator. O(N²)
worst, O(N) on nearly-sorted input, stable, in-place, no allocation. The
displaced element is saved once to a stack buffer, the insertion point is found,
and the run is shifted with a single `llvm.memmove` — no byte-by-byte swapping.
The right choice for small or nearly-sorted arrays (and the finishing pass other
sorts delegate to).

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_insertion(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | Stable in-place insertion sort of `count` elements of `elem_size` bytes by `cmp`. | 0 OK, 1 NULL |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_insertion(void*, int64_t, int64_t,
                                       int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

```c
// isort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_insertion(void*, int64_t, int64_t,
                                       int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<16]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_insertion(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 isort_cli.c build/libuniverse.a -lpthread -lm -o isort_cli
printf '5 3 8 1 9 2\n' | ./isort_cli
```

## Notes

- **Best for small / nearly-sorted arrays**; O(N) when the input is already
  almost ordered. Stable, in-place, no allocation.
- **Threading.** Reentrant; no shared state.
