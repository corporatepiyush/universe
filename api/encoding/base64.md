# universe_base64

## Purpose

Base64 encode/decode, standard AND url-safe alphabets. Pure compute over caller
byte ranges; the caller pre-sizes `dst` (via `encode_len`/`decode_len`) and the
module never allocates. A single `i32 urlsafe` flag picks the alphabet (0 =
standard `+`/`/`, non-zero = url-safe `-`/`_`); only symbols 62/63 differ so the
flag folds into two `select`s. SIMD-first: the public `encode`/`decode` are the
128-bit vector entries (12 bytes→16 chars / 16 chars→12 bytes per iteration via
`shufflevector` + field-spread multiplies), each with a `_scalar` twin that is
the ragged-tail/padding handler AND the cross-check oracle. Both alphabets emit
`=` padding.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_base64_encode_len` | `int64_t universe_base64_encode_len(int64_t n)` | Padded encoded length for `n` bytes. | length, or -1 overflow |
| `universe_base64_decode_len` | `int64_t universe_base64_decode_len(const void *src, int64_t n)` | Decoded byte length. | length, or -1 (PARSE) |
| `universe_base64_encode` | `int64_t universe_base64_encode(void *dst, const void *src, int64_t n, int32_t urlsafe)` | Vector encode. | chars written |
| `universe_base64_decode` | `int64_t universe_base64_decode(void *dst, const void *src, int64_t n, int32_t urlsafe)` | Vector decode. | bytes written, or -1 (PARSE) |
| `universe_base64_encode_scalar` | `int64_t universe_base64_encode_scalar(void *dst, const void *src, int64_t n, int32_t urlsafe)` | Scalar fallback/oracle encode. | chars written |
| `universe_base64_decode_scalar` | `int64_t universe_base64_decode_scalar(void *dst, const void *src, int64_t n, int32_t urlsafe)` | Scalar fallback/oracle decode. | bytes written, or -1 (PARSE) |

`decode` returns -1 for a length not a multiple of 4, misplaced padding, or any
char outside the selected alphabet. `dst` must be pre-sized.

## Use in an LLVM-based environment

```llvm
declare i64 @universe_base64_encode_len(i64)
declare i64 @universe_base64_encode(ptr, ptr, i64, i32)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Base64-encode stdin (standard alphabet).

```c
// b64cli.c — base64-encode stdin
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_base64_encode_len(int64_t);
int64_t universe_base64_encode(void*, const void*, int64_t, int32_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    int64_t olen = universe_base64_encode_len((int64_t)n);
    char *out = malloc(olen);
    int64_t w = universe_base64_encode(out, b, (int64_t)n, /*urlsafe*/0);
    fwrite(out, 1, w, stdout); putchar('\n');
    free(b); free(out);
    return 0;
}
```

```
clang -O3 b64cli.c build/libuniverse.a -lpthread -lm -o b64cli
printf 'hello' | ./b64cli        # aGVsbG8=
```

## Notes

- No allocation; `dst` caller-owned and pre-sized. Pass `urlsafe=1` for the
  `-`/`_` alphabet; padding (`=`) is emitted either way, so `decode_len` is exact.
- The vector path is the default (SSE2/NEON, no CPU probe); the `_scalar` twins
  are the fallback and the tested oracle.
- Reentrant, stateless.
