# bignum — big-integer arithmetic

Exported as `universe_bignum_*`. C ABI, `nounwind`. Link against
`build/libuniverse.a`:
`clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.

| Module | What |
|---|---|
| [bignum](bignum.md) | Unsigned multiprecision: add/sub/mul/divmod/shift, Montgomery montmul/modexp, gcd/modinv — the RSA/ECC foundation |

Representation: little-endian `uint64_t` limb arrays with an explicit limb count
per call, no sign, no hidden handle. Fast, not constant-time (hardening
deferred). Single-threaded.
