# sort/quick — introspective quicksort (generic comparator)

## Purpose

Quicksort over an arbitrary element type via a caller comparator. O(N log N)
expected, in-place, not stable. Median-of-3 pivot swapped to the hi slot +
Lomuto partition — provably terminates (no Hoare value-pivot livelock on
all-equal input) and always excludes the pivot slot from both subranges.
Recursion-free: an explicit 64-entry range stack pushes the larger side and
loops on the smaller, so depth ≤ log2 N. Runs ≤ 16 are left for a single final
insertion pass (nearly-sorted input finishes in one sequential sweep). Elements
> 1 KiB delegate to `universe_sort_heap` (in-place, no pivot buffer) — same
contract, no allocation either way. This is the general-purpose default sort.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_quick(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))` | Sort `count` elements of `elem_size` bytes at `base` in place, ordering by `cmp`. | 0 OK, 1 NULL (base or cmp NULL) |

The comparator follows the C convention: return negative if `*a < *b`, zero if
equal, positive if `*a > *b`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_quick(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Sort whitespace-separated integers from stdin, ascending.

```c
// qsort_cli.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern int32_t universe_sort_quick(void*, int64_t, int64_t,
                                   int (*)(const void*, const void*));
static int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}
int main(void){
    static int v[1<<20];
    int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int)x;
    universe_sort_quick(v, n, sizeof(int), cmp_int);
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 qsort_cli.c build/libuniverse.a -lpthread -lm -o qsort_cli
printf '5 3 8 1 9 2\n' | ./qsort_cli
```

## Notes

- **Generic + no allocation.** Works on any fixed-size element via the
  comparator; sorts in place. Whole-element moves use `llvm.memcpy` through a
  1 KiB bounce buffer (chunked for oversized elements).
- **Comparator cost dominates** — the indirect call blocks inlining. For `int32`
  keys prefer `sort/radix` or `sort/counting` (measured ~20× faster; comparison
  sort is the wrong class for integer keys).
- **Threading.** Reentrant; sort disjoint arrays concurrently. No shared state.
