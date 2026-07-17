# sort/counting — counting sort for int32 arrays

## Purpose

Counting sort specialized for `int32` arrays. O(N + range). Min/max are found in
one fused pass (auto-vectorizes to smin/smax reductions); output is written as
value runs directly from the histogram — no separate positions/prefix array, no
scatter, purely sequential stores. The value range is guarded (≤ 2²⁶ buckets) so
hostile inputs cannot OOM; wider ranges return `INVALID_ARG(8)` (use `sort/radix`
or `sort/quick` for those). Fastest class for small-range integer keys.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_counting(int32_t* base, int64_t count)` | In-place ascending sort of `count` `int32` values at `base`. | 0 OK, 1 NULL, 8 INVALID_ARG (range > 2²⁶) |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_counting(int32_t*, int64_t);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Sort small-range integers from stdin.

```c
// csort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_counting(int32_t*, int64_t);
int main(void){
    static int32_t v[1<<20]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int32_t)x;
    if (universe_sort_counting(v, n) != 0) { fprintf(stderr, "range too wide\n"); return 1; }
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 csort_cli.c build/libuniverse.a -lpthread -lm -o csort_cli
printf '5 3 8 1 9 2 3 3\n' | ./csort_cli
```

## Notes

- **int32 only, small range.** Falls back with `INVALID_ARG(8)` when `max-min`
  exceeds 2²⁶ buckets — route those to `sort/radix`.
- **In-place result, one histogram allocation** internal to the call. Reentrant;
  no shared state.
