# sort/selection — selection sort (generic comparator)

## Purpose

Selection sort over an arbitrary element type via a caller comparator. O(N²)
compares but only O(N) element moves — the right N² sort when *moves* are
expensive (huge elements). The min-index scan touches no data beyond the
comparator, and there is exactly one whole-element swap per position. In-place,
not stable, no allocation.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_selection(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | In-place selection sort of `count` elements of `elem_size` bytes by `cmp`. | 0 OK, 1 NULL |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_selection(void*, int64_t, int64_t,
                                       int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

```c
// ssort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_selection(void*, int64_t, int64_t,
                                       int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<16]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_selection(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 ssort_cli.c build/libuniverse.a -lpthread -lm -o ssort_cli
printf '5 3 8 1 9 2\n' | ./ssort_cli
```

## Notes

- **Minimizes moves** (exactly one swap per position) — use when elements are
  large and copying dominates. In-place, no allocation, not stable.
- **Threading.** Reentrant; no shared state.
