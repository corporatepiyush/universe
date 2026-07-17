# universe_crypto_pbkdf2

## Purpose

PBKDF2 (RFC 8018 / PKCS#5 v2.1) over HMAC:
`DK = T_1 || … || T_l`, `T_i = U_1 ⊕ … ⊕ U_c`, `U_1 = PRF(P, S || INT32_BE(i))`,
`U_j = PRF(P, U_{j-1})`, `PRF = HMAC-<digest>`. The key optimization is
key-setup amortization: HMAC's expensive step (compressing the two padded key
blocks) runs ONCE to build a 440-byte template context; each PRF invocation then
`memcpy`s the template and only feeds the short message, so a 4096-iteration
derivation costs ~4096·2 short compressions, not ~4096·4. No allocation in the
iteration loop.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_pbkdf2` | `void universe_crypto_pbkdf2(void *initfp, void *updfp, void *finfp, int64_t blocksize, int64_t digestsize, const void *pass, int64_t plen, const void *salt, int64_t slen, int64_t iters, void *out, int64_t dklen)` | Generic PBKDF2 (bring your own digest vtable). | — |
| `universe_crypto_pbkdf2_sha1` | `void universe_crypto_pbkdf2_sha1(const void *pass, int64_t plen, const void *salt, int64_t slen, int64_t iters, void *out, int64_t dklen)` | PBKDF2-HMAC-SHA-1. | — |
| `universe_crypto_pbkdf2_sha256` | `void universe_crypto_pbkdf2_sha256(const void *pass, int64_t plen, const void *salt, int64_t slen, int64_t iters, void *out, int64_t dklen)` | PBKDF2-HMAC-SHA-256. | — |
| `universe_crypto_pbkdf2_sha512` | `void universe_crypto_pbkdf2_sha512(const void *pass, int64_t plen, const void *salt, int64_t slen, int64_t iters, void *out, int64_t dklen)` | PBKDF2-HMAC-SHA-512. | — |

`out` receives exactly `dklen` derived-key bytes. `iters` is the iteration count
(public parameter).

## Use in an LLVM-based environment

```llvm
declare void @universe_crypto_pbkdf2_sha256(ptr, i64, ptr, i64, i64, ptr, i64)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// pbkdf2cli.c — derive a key: argv[1]=password argv[2]=salt argv[3]=iters argv[4]=dklen
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
void    universe_crypto_pbkdf2_sha256(const void*, int64_t, const void*, int64_t, int64_t, void*, int64_t);
int64_t universe_hex_encode(void*, const void*, int64_t);
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s pass salt iters dklen\n", argv[0]); return 2; }
    int64_t iters = atoll(argv[3]), dklen = atoll(argv[4]);
    unsigned char *dk = malloc(dklen); char *hex = malloc(dklen * 2);
    universe_crypto_pbkdf2_sha256(argv[1], strlen(argv[1]), argv[2], strlen(argv[2]),
                                  iters, dk, dklen);
    universe_hex_encode(hex, dk, dklen);
    fwrite(hex, 1, dklen * 2, stdout); putchar('\n');
    free(dk); free(hex); return 0;
}
```

```
clang -O3 pbkdf2cli.c build/libuniverse.a -lpthread -lm -o pbkdf2cli
./pbkdf2cli password salt 4096 32
```

## Notes

- No allocation inside; scratch/working contexts live on the stack and are reused
  across output blocks. `out` is caller-owned, sized to `dklen`.
- The generic entry takes the HMAC block/digest sizes (64/128 and 20/32/64) plus
  the digest vtable; the `_shaN` helpers fill those in.
- One-shot and reentrant (no shared state).
- **HARDENING-TODO:** fast, not constant-time; the iteration count is public but
  the derived key is secret — add zeroization of `T`/`U`/working-ctx and a
  constant-time review in the hardening phase.
