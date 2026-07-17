# crypto — universe API

Hashing, MAC, key-derivation and public-key primitives, hand-written in LLVM IR
and shipped in `libuniverse.a` / `libuniverse.dylib` / `libuniverse.so`
(`make lib`). All symbols are C ABI, `nounwind`. Exported names are
`universe_crypto_<algorithm>_<op>`.

> **HARDENING status.** Every module here is marked `; HARDENING-TODO:` in its
> source: the implementations are correctness-first and **fast, NOT
> constant-time**. Scalar ladders, modular reductions, and byte comparisons
> branch on secret data; contexts are not zeroized on finalize; PS/nonce bytes
> are deterministic filler, not CSPRNG output. Do not deploy these in adversarial
> side-channel settings until the hardening phase lands. Correctness is gated on
> published KATs (FIPS/RFC), not internal round-trips.

## Modules

| Module | What it is |
| --- | --- |
| [md5](md5.md) | MD5 (RFC 1321), 16-byte digest. Legacy/checksum only — broken. |
| [sha1](sha1.md) | SHA-1 (FIPS 180-4), 20-byte digest. Legacy only — broken. |
| [sha256](sha256.md) | SHA-256 (FIPS 180-4), 32-byte digest. |
| [sha512](sha512.md) | SHA-512 (FIPS 180-4), 64-byte digest. |
| [hmac](hmac.md) | HMAC (RFC 2104), generic over any block digest + SHA-1/256/512 helpers. |
| [pbkdf2](pbkdf2.md) | PBKDF2 (RFC 8018) over HMAC, with SHA-1/256/512 helpers. |
| [bcrypt](bcrypt.md) | bcrypt (EksBlowfish) password hash + the Blowfish cipher. |
| [rsa](rsa.md) | RSA PKCS#1 v1.5 sign/verify (SHA-256) and encrypt/decrypt. |
| [ecdsa](ecdsa.md) | ECDSA over NIST P-256, sign (caller nonce) + verify. |
| [ed25519](ed25519.md) | Ed25519 signatures (RFC 8032). |

## Common conventions

- Streaming digests use a **caller-allocated context** (no malloc inside):
  MD5 88 B, SHA-1 96 B, SHA-256 104 B, SHA-512 200 B, HMAC 440 B (align 8).
- `_hash` / `_final` writes go to a caller-owned `out` buffer sized to the
  digest length.
- Link line: `clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.
