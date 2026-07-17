# structures_succinct/eliasfano

## Purpose

Compact storage for sorted integer sequences. This module bundles three related
tools:

- **Elias-Fano** (`ef_*`) — near-optimal-space encoding of a monotone
  non-decreasing u64 sequence with O(1) random access and O(log n) successor
  search. Each value is split into low bits (bit-packed) and high bits (a
  unary-gap bitvector); a value uses ~`2 + ceil(log2(U/N))` bits, near the
  information-theoretic minimum. Access is popcount-based `select1` with sampled
  positions; no allocation on the access path.
- **Postings codec** (`postings_*`) — delta+varint compression of a sorted list
  of document ids, plus a SIMD-friendly byte-packed (frame-of-reference)
  variant whose decode vectorizes.
- **RLE** (`rle_*`) — a small byte-value run-length codec.

Choose Elias-Fano for random-access-plus-successor over a static monotone
sequence; the postings codecs for compact, streamable sorted-id lists.

## Exported API

### Elias-Fano

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_ef_build(void *vals, int64_t n)` | Build from `n` monotone non-decreasing `uint64_t` values | handle, or NULL on OOM |
| `int64_t universe_ds_ef_access(void *ef, int64_t i)` | The `i`-th value | value |
| `int64_t universe_ds_ef_next_geq(void *ef, int64_t x)` | First value >= `x` | value, or -1 if none |
| `int64_t universe_ds_ef_size(void *ef)` | Number of values N | count |
| `int64_t universe_ds_ef_footprint(void *ef)` | Total bytes used | bytes |
| `void universe_ds_ef_destroy(void *ef)` | Free | — |

### Postings (delta+varint)

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_ds_postings_bound(int64_t n)` | Max encoded bytes for `n` ids | bytes, or <0 overflow |
| `int64_t universe_ds_postings_encode(void *ids, int64_t n, void *dst)` | Encode `n` sorted `uint64_t` ids into `dst` | bytes written, or <0 err |
| `int64_t universe_ds_postings_decode(void *src, int64_t bytes, void *out, int64_t cap)` | Decode into `out` (`cap` ids) | id count, or <0 err |

### Postings (SIMD byte-packed)

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_ds_postings_bound_packed(int64_t n)` | Max packed bytes for `n` ids | bytes, or <0 overflow |
| `int64_t universe_ds_postings_encode_packed(void *ids, int64_t n, void *dst)` | Frame-of-reference block encode | bytes written, or <0 err |
| `int64_t universe_ds_postings_decode_packed(void *src, int64_t bytes, void *out, int64_t cap)` | SIMD decode (with scalar prefix-sum) | id count, or <0 err |
| `int64_t universe_ds_postings_decode_packed_scalar(void *src, int64_t bytes, void *out, int64_t cap)` | Scalar-only decode (oracle/fallback) | id count, or <0 err |

### RLE

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_ds_rle_encode(void *src, int64_t n, void *dst)` | Run-length encode `n` bytes | bytes written, or <0 err |
| `int64_t universe_ds_rle_decode(void *src, int64_t bytes, void *out, int64_t cap)` | Decode into `out` (`cap` bytes) | byte count, or <0 err |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_ef_build(ptr, i64)
declare i64 @universe_ds_ef_access(ptr, i64)
declare i64 @universe_ds_ef_next_geq(ptr, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read a sorted list of ids, build Elias-Fano, answer successor queries.

```c
// efcli.c — build: clang -O3 efcli.c build/libuniverse.a -lpthread -lm -o efcli
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void *universe_ds_ef_build(void *, int64_t);
extern int64_t universe_ds_ef_next_geq(void *, int64_t);
extern int64_t universe_ds_ef_size(void *);
extern void universe_ds_ef_destroy(void *);

int main(int argc, char **argv) {
  uint64_t vals[1024]; int64_t n = 0;
  long long x;
  while (n < 1024 && scanf("%lld", &x) == 1) vals[n++] = (uint64_t)x;  // must be sorted
  void *ef = universe_ds_ef_build(vals, n);
  long long q = argc > 1 ? atoll(argv[1]) : 0;
  printf("next_geq(%lld)=%lld  (n=%lld)\n", q,
         (long long)universe_ds_ef_next_geq(ef, q), (long long)universe_ds_ef_size(ef));
  universe_ds_ef_destroy(ef);
  return 0;
}
```

```sh
printf '1 4 9 16 25\n' | ./efcli 10   # -> next_geq(10)=16 (n=5)
```

## Notes

- Elias-Fano input **must be monotone non-decreasing** `uint64_t`. The structure
  is immutable after build.
- Postings/RLE encoders write into a caller `dst` sized by the matching `bound`
  function; decoders write up to `cap` and return the true count. All are pure
  buffer transforms (no handle to free).
- Single-threaded.
