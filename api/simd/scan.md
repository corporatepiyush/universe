# simd/scan — reusable SIMD scan / compare / transform primitives

## Purpose

`universe_simd_*` are the shared SIMD building blocks the HTTP parser, encoders,
and other parsers call for memchr / memcmp / case-fold / ascii-validate work.
Pure compute over a caller-supplied byte range; never allocates.

This module is the canonical embodiment of SIMD-first with scalar fallback. The
primary path is a portable 128-bit `<16 x i8>` vector loop that lowers to SSE2
on AMD64 and NEON on AArch64 — both baseline everywhere — so it needs no runtime
CPU check. Per-lane work is branch-free: a vector `icmp` builds a 16-lane mask,
then either a movemask + `llvm.cttz.i16` to LOCATE the first match, or a
reduce-add to COUNT. Every op ships a `*_scalar` twin that (a) handles the
sub-16 remainder tail and (b) is the cross-check oracle — vector and scalar
return bit-identical results for every input.

## Exported API

Each op has a vector entry (primary) and a `_scalar` twin (fallback + oracle),
returning identical results.

| C signature | Description | Return |
|---|---|---|
| `int64_t universe_simd_find_byte(const void* p, int64_t n, char c)` | Index of first byte equal to `c`. | index, or -1 |
| `int64_t universe_simd_find_byte_scalar(const void* p, int64_t n, char c)` | Scalar twin. | index, or -1 |
| `int64_t universe_simd_find_crlf(const void* p, int64_t n)` | Index of first CR or LF. | index, or -1 |
| `int64_t universe_simd_find_crlf_scalar(const void* p, int64_t n)` | Scalar twin. | index, or -1 |
| `int64_t universe_simd_index_of_any(const void* p, int64_t n, const void* set, int64_t setlen)` | Index of first byte present in `set[0..setlen)`. | index, or -1 |
| `int64_t universe_simd_index_of_any_scalar(const void* p, int64_t n, const void* set, int64_t setlen)` | Scalar twin. | index, or -1 |
| `int64_t universe_simd_count_byte(const void* p, int64_t n, char c)` | Count of bytes equal to `c`. | count |
| `int64_t universe_simd_count_byte_scalar(const void* p, int64_t n, char c)` | Scalar twin. | count |
| `int universe_simd_equal(const void* a, const void* b, int64_t n)` | `a[0..n) == b[0..n)`. | bool |
| `int universe_simd_equal_scalar(const void* a, const void* b, int64_t n)` | Scalar twin. | bool |
| `int32_t universe_simd_compare(const void* a, const void* b, int64_t n)` | memcmp sign: signed difference of first differing unsigned byte. | <0 / 0 / >0 |
| `int32_t universe_simd_compare_scalar(const void* a, const void* b, int64_t n)` | Scalar twin. | <0 / 0 / >0 |
| `void universe_simd_to_lower_ascii(void* dst, const void* src, int64_t n)` | Lowercase ASCII A–Z (`dst==src` allowed); other bytes untouched. | — |
| `void universe_simd_to_lower_ascii_scalar(void* dst, const void* src, int64_t n)` | Scalar twin. | — |
| `void universe_simd_to_upper_ascii(void* dst, const void* src, int64_t n)` | Uppercase ASCII a–z (`dst==src` allowed). | — |
| `void universe_simd_to_upper_ascii_scalar(void* dst, const void* src, int64_t n)` | Scalar twin. | — |
| `int universe_simd_is_ascii(const void* p, int64_t n)` | Are all bytes < 0x80? | bool |
| `int universe_simd_is_ascii_scalar(const void* p, int64_t n)` | Scalar twin. | bool |
| `int64_t universe_simd_validate_ascii(const void* p, int64_t n)` | Index of first non-ASCII byte (high bit set). | index, or -1 if all ASCII |
| `int64_t universe_simd_validate_ascii_scalar(const void* p, int64_t n)` | Scalar twin. | index, or -1 |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern int64_t universe_simd_find_byte(const void*, int64_t, char);
extern int universe_simd_is_ascii(const void*, int64_t);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A small text-stats tool: read stdin, report byte count, newline count, first
non-ASCII offset.

```c
// scanstat.c
#include <stdint.h>
#include <stdio.h>
extern int64_t universe_simd_count_byte(const void*, int64_t, char);
extern int64_t universe_simd_validate_ascii(const void*, int64_t);
extern int universe_simd_is_ascii(const void*, int64_t);

int main(void){
    static char buf[1<<20];
    int64_t n = fread(buf, 1, sizeof buf, stdin);
    printf("bytes=%lld lines=%lld ascii=%d first_nonascii=%lld\n",
           (long long)n,
           (long long)universe_simd_count_byte(buf, n, '\n'),
           universe_simd_is_ascii(buf, n),
           (long long)universe_simd_validate_ascii(buf, n));
    return 0;
}
```

```
clang -O3 scanstat.c build/libuniverse.a -lpthread -lm -o scanstat
printf 'hello\nworld\n' | ./scanstat
```

## Notes

- **SIMD-first.** The vector entry is the default (128-bit SSE2/NEON, no runtime
  check); the `_scalar` twin is the tail handler and the test oracle. Vector and
  scalar are contractually bit-identical.
- **No allocation, no ownership.** Pure functions over caller memory; safe to
  call concurrently on disjoint (or read-only) ranges.
- **Transform ops** touch only ASCII letters, leaving UTF-8 continuation bytes
  (≥ 0x80) untouched; `to_lower`/`to_upper` allow in-place `dst == src`.
