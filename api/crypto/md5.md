# universe_crypto_md5

## Purpose

MD5 (RFC 1321): streaming context + one-shot, 16-byte digest. The `alwaysinline`
compression leaf has a constant 64-round trip count that `-O3` unrolls, folding
the per-round `F`, message index, `K[i]` and shift constants; rotations use
`llvm.fshl.i32`. MD5 is little-endian so words load/store directly (no bswap).
Context is caller-allocated, 88 bytes (align 8). **MD5 is cryptographically
broken (collisions); provided for legacy interop / checksums only, not security.**

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_md5_init` | `void universe_crypto_md5_init(void *ctx)` | Initialize an 88-byte context. | — |
| `universe_crypto_md5_update` | `void universe_crypto_md5_update(void *ctx, const void *data, int64_t len)` | Feed `len` bytes. | — |
| `universe_crypto_md5_final` | `void universe_crypto_md5_final(void *ctx, void *out16)` | Finalize; write 16 bytes. | — |
| `universe_crypto_md5_hash` | `void universe_crypto_md5_hash(const void *data, int64_t len, void *out16)` | One-shot. | — |

`ctx` is a caller buffer of **88 bytes**, align 8; `out16` ≥ 16 bytes.

## Use in an LLVM-based environment

```llvm
declare void @universe_crypto_md5_hash(ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// md5cli.c — md5 of stdin as hex
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void    universe_crypto_md5_hash(const void*, int64_t, void*);
int64_t universe_hex_encode(void*, const void*, int64_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    unsigned char d[16]; char h[32];
    universe_crypto_md5_hash(b, (int64_t)n, d);
    universe_hex_encode(h, d, 16);
    fwrite(h, 1, 32, stdout); putchar('\n'); free(b); return 0;
}
```

```
clang -O3 md5cli.c build/libuniverse.a -lpthread -lm -o md5cli
printf 'abc' | ./md5cli   # 900150983cd24fb0d6963f7d28e17f72
```

## Notes

- Digest 16 bytes; context 88 B / align 8, caller-owned, no malloc.
- One-shot `_hash` is reentrant; a streaming context is single-threaded.
- **HARDENING-TODO:** fast, not constant-time; no context zeroization on final.
  Must not be used where collision/preimage resistance is required.
