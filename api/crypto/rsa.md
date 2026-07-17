# universe_crypto_rsa

## Purpose

RSA PKCS#1 v1.5 (RFC 8017): RSASSA sign/verify over SHA-256 and RSAES
encrypt/decrypt. Modular exponentiation is delegated to `bignum_modexp`
(Montgomery square-and-multiply); this module is the padding + byte-order
(I2OSP/OS2IP) layer around it. Keys `n`, `e`, `d` are caller-provided bignum limb
arrays of `s` little-endian `u64` limbs; the modulus byte length is `k = 8*s`.
RSA operates on big-endian octet strings, so conversion is a whole-buffer byte
reversal (no per-limb swaps). Signature verification rebuilds and compares the
expected `EM` byte-for-byte (no parsing of attacker-controlled structure).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_rsa_pkcs1_sign_sha256` | `void universe_crypto_rsa_pkcs1_sign_sha256(const void *msg, int64_t mlen, const void *n, const void *d, int64_t s, void *sig)` | RSASSA-PKCS1-v1_5 sign (SHA-256); write `k` big-endian bytes. | — |
| `universe_crypto_rsa_pkcs1_verify_sha256` | `int32_t universe_crypto_rsa_pkcs1_verify_sha256(const void *msg, int64_t mlen, const void *n, const void *e, int64_t s, const void *sig)` | Verify a signature. | 0 OK / 1 bad |
| `universe_crypto_rsa_encrypt_pkcs1` | `int32_t universe_crypto_rsa_encrypt_pkcs1(const void *msg, int64_t mlen, const void *n, const void *e, int64_t s, void *out)` | RSAES-PKCS1-v1_5 encrypt; write `k` bytes. | 0 OK / 8 too long |
| `universe_crypto_rsa_decrypt_pkcs1` | `int64_t universe_crypto_rsa_decrypt_pkcs1(const void *ct, const void *n, const void *d, int64_t s, void *out)` | Decrypt; write recovered message. | message length, or -1 |

`s` = number of `u64` limbs in the key; `k = 8*s` = modulus byte length.
`sig` / `out` / `ct` are `k` big-endian bytes. Message must satisfy
`mlen <= k - 11` for encrypt.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_crypto_rsa_pkcs1_verify_sha256(ptr, i64, ptr, ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// rsaverify.c — verify a PKCS#1 v1.5 SHA-256 signature (keys/sig as raw limb blobs)
// argv: n.bin e.bin sig.bin msg.txt  (n,e little-endian u64 limbs; sig = k big-endian bytes)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t universe_crypto_rsa_pkcs1_verify_sha256(const void*, int64_t, const void*, const void*, int64_t, const void*);
static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); rewind(f);
    void *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s n.bin e.bin sig.bin msg\n", argv[0]); return 2; }
    long nl, el, sl, ml;
    void *n = slurp(argv[1], &nl), *e = slurp(argv[2], &el);
    void *sig = slurp(argv[3], &sl), *msg = slurp(argv[4], &ml);
    int64_t s = nl / 8;                          /* limb count from modulus blob */
    int r = universe_crypto_rsa_pkcs1_verify_sha256(msg, ml, n, e, s, sig);
    puts(r == 0 ? "OK" : "BAD");
    return r;
}
```

```
clang -O3 rsaverify.c build/libuniverse.a -lpthread -lm -o rsaverify
./rsaverify n.bin e.bin sig.bin message.txt
```

## Notes

- Keys are raw little-endian `u64` limb arrays; the caller supplies the limb
  count `s`. No key parsing (ASN.1/PEM) is included — bring decoded limbs.
- Signature/ciphertext buffers are `k = 8*s` big-endian bytes, caller-owned.
- Reentrant; the module allocates only bignum scratch internally.
- **HARDENING-TODO:** NOT constant-time — the modexp ladder branches on secret
  bits and the decrypt padding scan is not constant-time (Bleichenbacher-relevant).
  Encrypt PS bytes are deterministic filler, NOT CSPRNG; no blinding, no
  zeroization. Deferred to the hardening phase.
