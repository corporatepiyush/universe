# universe_crypto_bcrypt

## Purpose

bcrypt (Provos–Mazières EksBlowfish password hash) plus the Blowfish cipher it is
built on. Blowfish state (P-array + four S-boxes) is laid out contiguously as one
`i32[1042]` so pi-init is a single memcpy and the key schedule is a single sweep;
the Feistel encipher folds the swap into the round recurrence. The bcrypt cost
knob drives `2^cost` ExpandKey sweeps (each ~521 encipherments), scaling work
exponentially. Init constants are the fractional hex digits of pi, derived from
first principles; correctness is gated on the classic Blowfish ECB vectors and
published `$2` hashes. `BLOWFISH_CTX_SIZE = 4168` bytes, align 16.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_crypto_blowfish_init` | `void universe_crypto_blowfish_init(void *ctx, const void *key, int64_t keylen)` | Standard Blowfish key schedule into a 4168-byte context. | — |
| `universe_crypto_blowfish_encrypt` | `void universe_crypto_blowfish_encrypt(void *ctx, const void *in8, void *out8)` | Encrypt one 8-byte ECB block. | — |
| `universe_crypto_bcrypt_raw` | `void universe_crypto_bcrypt_raw(const void *pass, int64_t passlen, const void *salt16, int32_t cost, void *out23)` | EksBlowfish; write the raw 23-byte hash. | — |
| `universe_crypto_bcrypt` | `void universe_crypto_bcrypt(const void *pass, int64_t passlen, const void *salt16, int32_t cost, void *out)` | Full hash; write the 60-char `$2b$CC$…` string + NUL. | — |

`salt16` is exactly 16 raw salt bytes. `out23` ≥ 23 bytes; the framed `out` ≥ 61
bytes (60 chars + NUL). Blowfish `ctx` is a caller buffer of **4168 bytes**,
align 16.

## Use in an LLVM-based environment

```llvm
declare void @universe_crypto_bcrypt(ptr, i64, ptr, i32, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

```c
// bcryptcli.c — bcrypt a password: argv[1]=password argv[2]=cost, salt = 16 zero bytes
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
void universe_crypto_bcrypt(const void*, int64_t, const void*, int32_t, void*);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s password cost\n", argv[0]); return 2; }
    unsigned char salt[16] = {0};       /* demo: fixed salt; use a CSPRNG salt in real use */
    char out[61] = {0};
    universe_crypto_bcrypt(argv[1], (int64_t)strlen(argv[1]), salt, atoi(argv[2]), out);
    puts(out);
    return 0;
}
```

```
clang -O3 bcryptcli.c build/libuniverse.a -lpthread -lm -o bcryptcli
./bcryptcli hunter2 10          # $2b$10$....
```

## Notes

- Salt is 16 raw bytes (the framed output encodes it as 22 bcrypt-base64 chars).
  Cost 4..31; runtime is `~2^cost` — pick per your latency budget.
- Blowfish context 4168 B / align 16, caller-owned, no malloc.
- Reentrant (no shared state); each call is independent.
- **HARDENING-TODO:** fast, not constant-time. S-box lookups are data-dependent
  by design (as in every Blowfish); the key/salt/context buffers are not zeroized
  — add zeroization + a side-channel review in the hardening phase.
