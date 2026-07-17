# universe_hex

## Purpose

Hex (base16) encode/decode. Pure compute over caller byte ranges; the caller
pre-sizes `dst` (via `encode_len` or `n/2`) and the module never allocates. Each
loop iteration is register arithmetic on a loaded byte then a store, with no
data-dependent branch on the hot path, so the backend can unroll/vectorize.
Nibble→ascii is branchless (no lookup table): `ascii = v + '0' + ((9-v)>>a 31) &
0x27`. Decode is branchless too, OR-accumulating any invalidity across the whole
buffer into one flag checked once at the end.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_hex_encode_len` | `int64_t universe_hex_encode_len(int64_t n)` | Encoded length for `n` input bytes. | `2n`, or -1 on overflow |
| `universe_hex_encode` | `int64_t universe_hex_encode(void *dst, const void *src, int64_t n)` | Lowercase-hex encode `n` bytes. | `2n` written |
| `universe_hex_decode` | `int64_t universe_hex_decode(void *dst, const void *src, int64_t n)` | Decode `n` hex chars to bytes. | `n/2`, or -1 (PARSE) |

`decode` returns -1 for an odd length or any non-hex character. `dst` must be
pre-sized (`2n` for encode, `n/2` for decode).

## Use in an LLVM-based environment

```llvm
declare i64 @universe_hex_encode(ptr, ptr, i64)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Hex-encode stdin.

```c
// hexcli.c — hex-encode stdin (like xxd -p, no newlines)
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_hex_encode_len(int64_t);
int64_t universe_hex_encode(void*, const void*, int64_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    int64_t olen = universe_hex_encode_len((int64_t)n);
    char *out = malloc(olen);
    universe_hex_encode(out, b, (int64_t)n);
    fwrite(out, 1, olen, stdout); putchar('\n');
    free(b); free(out);
    return 0;
}
```

```
clang -O3 hexcli.c build/libuniverse.a -lpthread -lm -o hexcli
printf 'Hi' | ./hexcli        # 4869
```

## Notes

- No allocation; `dst` is caller-owned and pre-sized. Encode is always lowercase.
- Reentrant, stateless; the loops vectorize (branch-free).
