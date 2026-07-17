# sort/heap — iterative heapsort (generic comparator)

## Purpose

Heapsort over an arbitrary element type via a caller comparator. O(N log N)
worst case, in-place, not stable, no allocation. Fully iterative siftdown (no
recursion, no stack risk); whole-element swaps move through a 1 KiB stack bounce
buffer with `llvm.memcpy`, chunked for oversized elements. Chosen as the
worst-case-bounded, allocation-free fallback — `sort/quick` delegates elements
> 1 KiB here.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_heap(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | In-place heapsort of `count` elements of `elem_size` bytes by `cmp`. | 0 OK, 1 NULL |

Comparator: C convention (negative / zero / positive).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_heap(void*, int64_t, int64_t,
                                  int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

```c
// hsort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_heap(void*, int64_t, int64_t,
                                  int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<20]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_heap(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 hsort_cli.c build/libuniverse.a -lpthread -lm -o hsort_cli
printf '5 3 8 1 9 2\n' | ./hsort_cli
```

## Notes

- **In-place, worst-case O(N log N), no allocation.** Good for adversarial input
  or when heap memory must not be touched.
- **Threading.** Reentrant; no shared state.
