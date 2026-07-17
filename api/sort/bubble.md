# sort/bubble — bubble sort (generic comparator)

## Purpose

Bubble sort over an arbitrary element type via a caller comparator. O(N²),
stable, in-place, no allocation. Last-swap optimization: each pass only runs to
where the previous pass last swapped, so sorted tails are skipped entirely
(strictly better than a naive early-exit boolean). Whole-element swaps go
through a 1 KiB bounce buffer, chunked for oversized elements. Mainly a
reference/teaching sort; prefer `sort/quick` or `sort/insertion` in practice.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_bubble(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | Stable in-place bubble sort of `count` elements of `elem_size` bytes by `cmp`. | 0 OK, 1 NULL |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_bubble(void*, int64_t, int64_t,
                                    int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

```c
// bsort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_bubble(void*, int64_t, int64_t,
                                    int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<14]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_bubble(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 bsort_cli.c build/libuniverse.a -lpthread -lm -o bsort_cli
printf '5 3 8 1 9 2\n' | ./bsort_cli
```

## Notes

- **Stable, in-place, no allocation**, with the last-swap optimization — but
  O(N²); use only for tiny or already-sorted inputs.
- **Threading.** Reentrant; no shared state.
