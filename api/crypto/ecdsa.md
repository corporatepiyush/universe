# universe_crypto_ecdsa

## Purpose

ECDSA over NIST P-256 (FIPS 186-4 / SEC1): sign (with a caller-supplied nonce
`k`) and verify. Built on the bignum kernels — field `Fp` and scalar `Fn`
arithmetic reuse `bignum_mul` + `bignum_mod` (correctness-first, mod-per-multiply),
and inversion is `bignum_modinv`. The group law uses AFFINE coordinates with an
explicit infinity flag, trading a modular inversion per add/double for
exception-free formulas; scalar multiply is MSB-first double-and-add. All
scalars and coordinates are 4-limb little-endian `u64` (32 bytes each); the input
hash is 32 big-endian bytes.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_ecdsa_p256_verify` | `int32_t universe_crypto_ecdsa_p256_verify(const void *hash_be32, const void *qx, const void *qy, const void *r, const void *s)` | Verify `(r,s)` against public key `(qx,qy)`. | 0 OK / 1 bad |
| `universe_crypto_ecdsa_p256_sign` | `int32_t universe_crypto_ecdsa_p256_sign(const void *hash_be32, const void *priv, const void *k, void *out_r, void *out_s)` | Sign; write `(r,s)`. | 0 OK / 8 retry (zero r/s — supply a new k) |

`hash_be32` is 32 big-endian bytes. `qx`, `qy`, `priv`, `k`, `r`, `s`, `out_r`,
`out_s` are each a 4-limb little-endian `u64` array (32 bytes). Signing does NOT
generate `k` — the caller supplies it (and must retry with a fresh one on error
8).

## Use in an LLVM-based environment

```llvm
declare i32 @universe_crypto_ecdsa_p256_verify(ptr, ptr, ptr, ptr, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// ecdsaverify.c — verify a P-256 signature; each operand file is 32 raw bytes.
// argv: hash.be32 qx qy r s   (hash big-endian; qx/qy/r/s little-endian u64 limbs)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t universe_crypto_ecdsa_p256_verify(const void*, const void*, const void*, const void*, const void*);
static void rd(const char *p, unsigned char b[32]) {
    FILE *f = fopen(p, "rb"); fread(b, 1, 32, f); fclose(f);
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s hash qx qy r s\n", argv[0]); return 2; }
    unsigned char h[32], qx[32], qy[32], r[32], s[32];
    rd(argv[1], h); rd(argv[2], qx); rd(argv[3], qy); rd(argv[4], r); rd(argv[5], s);
    int v = universe_crypto_ecdsa_p256_verify(h, qx, qy, r, s);
    puts(v == 0 ? "OK" : "BAD");
    return v;
}
```

```
clang -O3 ecdsaverify.c build/libuniverse.a -lpthread -lm -o ecdsaverify
./ecdsaverify hash.be32 qx.bin qy.bin r.bin s.bin
```

## Notes

- All curve operands are 32-byte 4-limb little-endian `u64`; the message hash is
  32 big-endian bytes (`bits2int` for P-256 where hashlen == qlen).
- Caller owns the nonce lifecycle: `k` must be unique+secret per signature; a
  zero `r`/`s` returns 8 and the caller retries with a new `k`.
- Reentrant; only bignum scratch is allocated internally.
- **HARDENING-TODO:** NOT constant-time (ladder + field-mod branch on secrets);
  no in-module RFC-6979/CSPRNG nonce generation yet; no scratch zeroization.
