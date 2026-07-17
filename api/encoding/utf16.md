# universe_utf16

## Purpose

UTF-16 codec (LE/BE) and UTF-8 ⇄ UTF-16 transcoding. Pure compute over caller
byte ranges: the caller pre-sizes `dst` (via the `len_*` helpers) and the module
never allocates or does IO. Endianness is a runtime `i32` param (0 = LE, non-zero
= BE); all targets are little-endian so a native `i16` load/store IS the LE form
and the BE form is one branchlessly-selected `llvm.bswap.i16`. Scalar validity
rejects the surrogate block and anything above U+10FFFF; encoding emits one BMP
unit or a hi/lo surrogate pair. `from_utf8` first calls the sibling UTF-8
validator (reused, not reimplemented) then a decode+emit pass. A 128-bit ASCII/
BMP SIMD fast path is primary; the scalar codepoint loop is the tail + oracle.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_utf16_scalar_units` | `int32_t universe_utf16_scalar_units(int32_t scalar)` | UTF-16 units a scalar needs. | 1 or 2, or -8 |
| `universe_utf16_encode_scalar` | `int64_t universe_utf16_encode_scalar(void *dst, int32_t scalar, int32_t be)` | Encode one scalar. | units written, or -8 |
| `universe_utf16_decode_scalar` | `int64_t universe_utf16_decode_scalar(const void *src, int64_t navail, int32_t be)` | Decode one scalar. | scalar value, or -8 / -13 |
| `universe_utf16_bom_detect` | `int32_t universe_utf16_bom_detect(const void *src, int64_t nbytes)` | Detect a byte-order mark. | 0 none / 1 LE / 2 BE |
| `universe_utf16_len_from_utf8` | `int64_t universe_utf16_len_from_utf8(const void *src, int64_t n)` | UTF-16 units for a UTF-8 input. | units, or -13 (PARSE) |
| `universe_utf16_len_to_utf8` | `int64_t universe_utf16_len_to_utf8(const void *src, int64_t nunits, int32_t be)` | UTF-8 bytes for a UTF-16 input. | bytes, or -13 / -8 |
| `universe_utf16_from_utf8` | `int64_t universe_utf16_from_utf8(void *dst, int64_t dcap, const void *src, int64_t n, int32_t be)` | Transcode UTF-8 → UTF-16. | units written, or < 0 |
| `universe_utf16_to_utf8` | `int64_t universe_utf16_to_utf8(void *dst, int64_t dcap, const void *src, int64_t nunits, int32_t be)` | Transcode UTF-16 → UTF-8. | bytes written, or < 0 |

`be` = 0 little-endian, non-zero big-endian. Size `dst` with the matching
`len_*` helper; a too-small `dcap` yields a negative FULL error.

## Use in an LLVM-based environment

```llvm
declare i64 @universe_utf16_len_from_utf8(ptr, i64)
declare i64 @universe_utf16_from_utf8(ptr, i64, ptr, i64, i32)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Transcode UTF-8 stdin to little-endian UTF-16, write raw bytes to stdout.

```c
// u8to16.c — UTF-8 stdin -> UTF-16LE stdout
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_utf16_len_from_utf8(const void*, int64_t);
int64_t universe_utf16_from_utf8(void*, int64_t, const void*, int64_t, int32_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    int64_t units = universe_utf16_len_from_utf8(b, (int64_t)n);
    if (units < 0) { fprintf(stderr, "invalid utf-8\n"); return 1; }
    unsigned char *out = malloc(units * 2);
    int64_t w = universe_utf16_from_utf8(out, units * 2, b, (int64_t)n, /*LE*/0);
    if (w >= 0) fwrite(out, 2, w, stdout);
    free(b); free(out);
    return w < 0 ? 1 : 0;
}
```

```
clang -O3 u8to16.c build/libuniverse.a -lpthread -lm -o u8to16
printf 'Hi' | ./u8to16 | xxd        # 4800 6900
```

## Notes

- No allocation; `dst` is caller-owned and pre-sized with `len_from_utf8` /
  `len_to_utf8`. A unit index addresses 16-bit units (byte offset = idx*2);
  loads/stores use align 1.
- Malformed UTF-8/UTF-16 (surrogates, truncation, out-of-range) returns a negative
  PARSE code; capacity overflow returns a negative FULL code.
- Reentrant, stateless.
