# structures_linear/bitset

## Purpose

Dynamic bit set over a packed array of i64 words, one allocation (64-byte header
+ word payload at +64). Bits above `nbits` in the last word are held
invariant-zero so `popcount`/`find` sweep whole words with no per-word range
test. Bit address math is pure register work (word `i>>6`, bit `i&63`). The
word-parallel set operations (union/intersection/difference/complement) are
plain element-wise loops that `-O3` auto-vectorizes to NEON/AVX, and their `dst`
may alias an operand for in-place results.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_bitset_create(int64_t nbits)` | Create a set of `nbits` bits (all clear) | handle, or NULL on OOM |
| `int32_t universe_ds_bitset_set(void *bs, int64_t i)` | Set bit `i` | 0 OK, 1 NULL, 7 INVALID_INDEX |
| `int32_t universe_ds_bitset_clear(void *bs, int64_t i)` | Clear bit `i` | 0 OK, 1, 7 |
| `int32_t universe_ds_bitset_toggle(void *bs, int64_t i)` | Flip bit `i` | 0 OK, 1, 7 |
| `int32_t universe_ds_bitset_test(void *bs, int64_t i)` | Read bit `i` | 1 set, 0 clear (or error) |
| `int32_t universe_ds_bitset_clear_all(void *bs)` | Clear every bit | 0 OK, 1 NULL |
| `int32_t universe_ds_bitset_set_all(void *bs)` | Set every valid bit | 0 OK, 1 NULL |
| `int64_t universe_ds_bitset_nbits(void *bs)` | Bit count | count |
| `int64_t universe_ds_bitset_popcount(void *bs)` | Number of set bits | count |
| `int64_t universe_ds_bitset_find_first_set(void *bs)` | Index of lowest set bit | index, or -1 none |
| `int64_t universe_ds_bitset_find_next_set(void *bs, int64_t from)` | Lowest set bit at index >= `from` | index, or -1 none |
| `int32_t universe_ds_bitset_union(void *dst, void *a, void *b)` | `dst = a | b` | 0 OK, or error |
| `int32_t universe_ds_bitset_intersection(void *dst, void *a, void *b)` | `dst = a & b` | 0 OK, or error |
| `int32_t universe_ds_bitset_difference(void *dst, void *a, void *b)` | `dst = a & ~b` | 0 OK, or error |
| `int32_t universe_ds_bitset_complement(void *dst, void *a)` | `dst = ~a` (masked to `nbits`) | 0 OK, or error |
| `void universe_ds_bitset_destroy(void *bs)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_bitset_create(i64)
declare i32 @universe_ds_bitset_set(ptr, i64)
declare i64 @universe_ds_bitset_popcount(ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Set bits from stdin indices, print the population count and set bits.

```c
// bscli.c — build: clang -O3 bscli.c build/libuniverse.a -lpthread -lm -o bscli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_bitset_create(int64_t);
extern int32_t universe_ds_bitset_set(void *, int64_t);
extern int64_t universe_ds_bitset_popcount(void *);
extern int64_t universe_ds_bitset_find_first_set(void *);
extern int64_t universe_ds_bitset_find_next_set(void *, int64_t);
extern void universe_ds_bitset_destroy(void *);

int main(void) {
  void *bs = universe_ds_bitset_create(1024);
  long long i;
  while (scanf("%lld", &i) == 1) universe_ds_bitset_set(bs, i);
  printf("popcount=%lld\n", (long long)universe_ds_bitset_popcount(bs));
  for (int64_t b = universe_ds_bitset_find_first_set(bs); b >= 0;
       b = universe_ds_bitset_find_next_set(bs, b + 1))
    printf("%lld ", (long long)b);
  printf("\n");
  universe_ds_bitset_destroy(bs);
  return 0;
}
```

```sh
printf '3 100 7\n' | ./bscli   # popcount=3 then 3 7 100
```

## Notes

- `nbits` is fixed at create. Indices must be `< nbits`.
- Set operations require operands of the same size; `dst` may alias `a`/`b`.
- Single-threaded.
