# sort/radix — LSD radix sort for int32 arrays

## Purpose

LSD radix sort specialized for `int32` arrays. O(4N) with 8-bit digits (four
passes). Sign is handled by XORing the top bit during the final pass, so
negatives order correctly with zero extra passes. A skip-pass optimization
histograms first and skips the scatter for any pass whose digit is constant
(e.g. all-positive small ints do 2 passes, not 4). Per-pass `[256]` histograms
live on the stack; one aux buffer ping-pongs between passes with a final memcpy
only if the result lands in aux. Measured ~20× faster than a comparison sort on
full-range 65 K int32 — the right class for integer keys.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sort_radix(int32_t* base, int64_t count)` | In-place ascending sort of `count` `int32` values at `base`. | 0 OK, 1 NULL, 2 OOM (aux alloc) |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int32_t universe_sort_radix(int32_t*, int64_t);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Sort arbitrary-range integers from stdin.

```c
// rsort_cli.c
#include <stdint.h>
#include <stdio.h>
extern int32_t universe_sort_radix(int32_t*, int64_t);
int main(void){
    static int32_t v[1<<20]; int64_t n = 0, x;
    while (n < (int64_t)(sizeof v/sizeof *v) && scanf("%ld", &x) == 1) v[n++] = (int32_t)x;
    if (universe_sort_radix(v, n) != 0) { fprintf(stderr, "oom\n"); return 1; }
    for (int64_t i = 0; i < n; i++) printf("%d\n", v[i]);
    return 0;
}
```

```
clang -O3 rsort_cli.c build/libuniverse.a -lpthread -lm -o rsort_cli
printf '5 -3 8 -1 9 2\n' | ./rsort_cli
```

## Notes

- **int32 only, any range** (handles negatives). The default for large integer
  arrays; `sort/counting` wins only on small-range data.
- **One aux buffer** (may return OOM). In-place result. Reentrant; no shared
  state.
