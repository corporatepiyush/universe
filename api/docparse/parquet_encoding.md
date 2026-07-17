# universe_docparse_parquet_encoding

## Purpose

Parquet column-encoding decoder kernels (parquet-format `Encodings.md`). Pure
compute over caller byte buffers; the only allocations are private scratch length
arrays inside the two DELTA byte-array kernels (freed on exit). Self-contained —
local ULEB128 / zig-zag / LSB-first bit readers are inlined rather than depending
on `encoding/varint`, keeping this a leaf of the dependency graph. Untrusted-input
safe: every read is bounds-checked against `dlen` (→ INVALID_ARG on overrun),
shifts are guarded against poison, the bit reader zero-pads past the end, and
every offset add/mul is overflow-checked (→ SIZE_OVERFLOW). The shared DELTA core
requires the header `total` to fit `out_cap`, defeating a zero-width-miniblock DoS.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_parquet_rle_hybrid` | `int32_t universe_docparse_parquet_rle_hybrid(const void *data, int64_t dlen, int32_t bit_width, int64_t count, void *out)` | Decode the RLE/bit-packing hybrid into `count` values. | 0 / 3 / 8 |
| `universe_docparse_parquet_plain` | `int32_t universe_docparse_parquet_plain(const void *data, int64_t dlen, int32_t phys_type, int32_t type_len, int64_t count, void *out, int64_t out_cap, int64_t *out_used)` | Decode PLAIN-encoded values. | 0 / 3 / 6 / 8 |
| `universe_docparse_parquet_delta_binary_packed` | `int32_t universe_docparse_parquet_delta_binary_packed(const void *data, int64_t dlen, void *out, int64_t out_cap, int64_t *out_count)` | Decode DELTA_BINARY_PACKED ints. | 0 / 3 / 6 / 8 |
| `universe_docparse_parquet_delta_length_byte_array` | `int32_t universe_docparse_parquet_delta_length_byte_array(const void *data, int64_t dlen, int64_t count, void *out, int64_t out_cap, int64_t *out_used)` | Decode DELTA_LENGTH_BYTE_ARRAY. | 0 / 3 / 6 / 8 |
| `universe_docparse_parquet_delta_byte_array` | `int32_t universe_docparse_parquet_delta_byte_array(const void *data, int64_t dlen, int64_t count, void *out, int64_t out_cap, int64_t *out_used)` | Decode DELTA_BYTE_ARRAY (prefix+suffix). | 0 / 3 / 6 / 8 |
| `universe_docparse_parquet_byte_stream_split` | `int32_t universe_docparse_parquet_byte_stream_split(const void *data, int64_t dlen, int32_t width, int64_t count, void *out)` | Un-split a BYTE_STREAM_SPLIT column. | 0 / 3 / 8 |

**i32 codes:** 0 OK, 3 SIZE_OVERFLOW, 6 FULL (output capacity), 8 INVALID_ARG
(malformed/truncated). **Physical type codes** (`phys_type`): 0 BOOLEAN, 1 INT32,
2 INT64, 3 INT96, 4 FLOAT, 5 DOUBLE. `out`/`out_cap` is caller-owned;
`out_used`/`out_count` report how much was produced.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_docparse_parquet_rle_hybrid(ptr, i64, i32, i64, ptr)
declare i32 @universe_docparse_parquet_delta_binary_packed(ptr, i64, ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Decode a DELTA_BINARY_PACKED page (raw bytes on stdin) and print the integers.

```c
// pqdelta.c — decode a DELTA_BINARY_PACKED blob from stdin, print i64 values
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int32_t universe_docparse_parquet_delta_binary_packed(const void*, int64_t, void*, int64_t, int64_t*);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    int64_t out_cap = 1 << 20, count = 0;
    int64_t *out = malloc(out_cap * sizeof(int64_t));
    int rc = universe_docparse_parquet_delta_binary_packed(b, (int64_t)n, out, out_cap, &count);
    if (rc != 0) { fprintf(stderr, "error %d\n", rc); return rc; }
    for (int64_t i = 0; i < count; i++) printf("%lld\n", (long long)out[i]);
    free(b); free(out);
    return 0;
}
```

```
clang -O3 pqdelta.c build/libuniverse.a -lpthread -lm -o pqdelta
./pqdelta < page.bin
```

## Notes

- Pure compute over caller buffers; only the two DELTA byte-array kernels malloc a
  private scratch length array (freed before return). The caller sizes `out`;
  a too-small buffer returns FULL(6).
- `bit_width` 0 is explicit (all-zero output, no data bytes consumed). Wrapping
  delta adds follow the spec's two's-complement semantics.
- Untrusted-input hardened; reentrant.
