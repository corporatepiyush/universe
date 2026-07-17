# universe_crypto_sha1

## Purpose

SHA-1 (FIPS 180-4): streaming context + one-shot, 20-byte digest. Same shape as
SHA-256 — an `alwaysinline` compression leaf with 80 unrolled rounds, `a..e`
register-resident, big-endian loads via `llvm.bswap.i32`, rotations via
`llvm.fshl.i32`, wrapping `i32` adds. Context is caller-allocated, 96 bytes
(align 8). **SHA-1 is cryptographically broken (collisions); provided for legacy
interop / checksums only, not for security.**

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_sha1_init` | `void universe_crypto_sha1_init(void *ctx)` | Initialize a 96-byte context. | — |
| `universe_crypto_sha1_update` | `void universe_crypto_sha1_update(void *ctx, const void *data, int64_t len)` | Feed `len` bytes. | — |
| `universe_crypto_sha1_final` | `void universe_crypto_sha1_final(void *ctx, void *out20)` | Finalize; write 20 bytes. | — |
| `universe_crypto_sha1_hash` | `void universe_crypto_sha1_hash(const void *data, int64_t len, void *out20)` | One-shot. | — |

`ctx` is a caller buffer of **96 bytes**, align 8; `out20` ≥ 20 bytes.

## Use in an LLVM-based environment

```llvm
declare void @universe_crypto_sha1_hash(ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// sha1cli.c — hash stdin, print hex
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void    universe_crypto_sha1_hash(const void*, int64_t, void*);
int64_t universe_hex_encode(void*, const void*, int64_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    unsigned char d[20]; char h[40];
    universe_crypto_sha1_hash(b, (int64_t)n, d);
    universe_hex_encode(h, d, 20);
    fwrite(h, 1, 40, stdout); putchar('\n'); free(b); return 0;
}
```

```
clang -O3 sha1cli.c build/libuniverse.a -lpthread -lm -o sha1cli
printf 'abc' | ./sha1cli   # a9993e364706816aba3e25717850c26c9cd0d89d
```

## Notes

- Digest 20 bytes; context 96 B / align 8, caller-owned, no malloc.
- One-shot `_hash` is reentrant; a streaming context is single-threaded.
- **HARDENING-TODO:** fast, not constant-time; no context zeroization on final.
  Must not be used where collision resistance is required.
