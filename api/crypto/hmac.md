# universe_crypto_hmac

## Purpose

HMAC (RFC 2104 / FIPS 198-1): `HMAC(K,m) = H((K'⊕opad) || H((K'⊕ipad) || m))`,
generic over any block-based digest. The digest is supplied as a tiny value
vtable (three function pointers `init`/`update`/`final` with the uniform
`void(ptr)` / `void(ptr,ptr,i64)` / `void(ptr,ptr)` shapes) plus its block and
output sizes, so one implementation drives SHA-1/256/512. Streaming and
allocation-free: the 440-byte caller context embeds two digest contexts,
pre-fed with the padded keys at `init` so key setup is not repeated per message —
exactly what PBKDF2 exploits by copying a post-init template.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_hmac_init` | `void universe_crypto_hmac_init(void *h, void *initfp, void *updfp, void *finfp, int64_t blocksize, int64_t digestsize, const void *key, int64_t klen)` | Generic init from a digest vtable. | — |
| `universe_crypto_hmac_update` | `void universe_crypto_hmac_update(void *h, const void *data, int64_t len)` | Feed message bytes. | — |
| `universe_crypto_hmac_final` | `void universe_crypto_hmac_final(void *h, void *out)` | Finalize; write `digestsize` bytes. | — |
| `universe_crypto_hmac_sha1_init` | `void universe_crypto_hmac_sha1_init(void *h, const void *key, int64_t klen)` | Preset init for HMAC-SHA-1. | — |
| `universe_crypto_hmac_sha256_init` | `void universe_crypto_hmac_sha256_init(void *h, const void *key, int64_t klen)` | Preset init for HMAC-SHA-256. | — |
| `universe_crypto_hmac_sha512_init` | `void universe_crypto_hmac_sha512_init(void *h, const void *key, int64_t klen)` | Preset init for HMAC-SHA-512. | — |
| `universe_crypto_hmac_sha1` | `void universe_crypto_hmac_sha1(const void *key, int64_t klen, const void *msg, int64_t mlen, void *out20)` | One-shot HMAC-SHA-1. | — |
| `universe_crypto_hmac_sha256` | `void universe_crypto_hmac_sha256(const void *key, int64_t klen, const void *msg, int64_t mlen, void *out32)` | One-shot HMAC-SHA-256. | — |
| `universe_crypto_hmac_sha512` | `void universe_crypto_hmac_sha512(const void *key, int64_t klen, const void *msg, int64_t mlen, void *out64)` | One-shot HMAC-SHA-512. | — |

`h` (context) is a caller buffer of **440 bytes**, align 8. Output length is the
digest size: 20 / 32 / 64 for SHA-1 / 256 / 512.

## Use in an LLVM-based environment

```llvm
declare void @universe_crypto_hmac_sha256(ptr, i64, ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// hmaccli.c — HMAC-SHA-256 of stdin under a key argv[1], printed as hex
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
void    universe_crypto_hmac_sha256(const void*, int64_t, const void*, int64_t, void*);
int64_t universe_hex_encode(void*, const void*, int64_t);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <key>\n", argv[0]); return 2; }
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    unsigned char mac[32]; char h[64];
    universe_crypto_hmac_sha256(argv[1], (int64_t)strlen(argv[1]), b, (int64_t)n, mac);
    universe_hex_encode(h, mac, 32);
    fwrite(h, 1, 64, stdout); putchar('\n'); free(b); return 0;
}
```

```
clang -O3 hmaccli.c build/libuniverse.a -lpthread -lm -o hmaccli
printf 'The quick brown fox jumps over the lazy dog' | ./hmaccli key
# f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
```

## Notes

- Context 440 B / align 8, caller-owned, no malloc. The generic `init` takes the
  three digest fn-pointers plus block size (64 for SHA-1/256, 128 for SHA-512)
  and digest size (20/32/64); the `_shaN_init` helpers fill those in for you.
- The indirect vtable calls sit OUTSIDE the compression loop (one per block), so
  per-block cost is the digest's, not the dispatch's.
- One-shots are reentrant; each streaming context is single-threaded.
- **HARDENING-TODO:** fast, not constant-time; add zeroization of the context in
  the hardening phase.
