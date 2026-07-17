# bignum/bignum — unsigned multiprecision arithmetic

## Purpose
Big-integer arithmetic foundation (unsigned multiprecision) — the prerequisite
for RSA / ECDSA / Ed25519. Numbers are caller-provided `uint64_t` limb buffers
in little-endian limb order (limb[0] is least significant) with an EXPLICIT
limb count per call — no hidden handle, no sign bit. This is the crypto-native
shape: RSA/ECC operate on fixed modulus widths, so allocation-free fixed-length
kernels are simpler to prove correct and faster. Every carry/borrow/mul step is
computed in `i128` and split into `{lo, hi}`, making propagation total and exact
(no `add nuw x,-1` landmine). Division is Knuth Algorithm D; modular exponentiation
is CIOS Montgomery square-and-multiply. Convenience alloc/set helpers are optional.

`; HARDENING-TODO:` fast, NOT constant-time — division, comparisons and the
ladder branch on secret data and scratch is not zeroized (deferred to the
hardening phase).

## Exported API
Error codes: `0` OK, `2` OUT_OF_MEMORY, `5` NOT_FOUND (no inverse), `8`
INVALID_ARG. Limbs are little-endian `uint64_t`; `s`/`n`/`m` are limb counts.

| C signature | Description | Returns |
|---|---|---|
| `uint64_t *universe_bignum_alloc(int64_t nlimbs)` | Allocate a zeroed limb buffer | pointer, or `NULL` on OOM |
| `void universe_bignum_free(uint64_t *p)` | Free a buffer from `_alloc` | — |
| `void universe_bignum_set_u64(uint64_t *r, int64_t s, int64_t v)` | Set `s`-limb `r` to the value `v` | — |
| `int64_t universe_bignum_normalize_len(uint64_t *a, int64_t n)` | Significant limb count (trailing-zero-limb strip) | length |
| `int64_t universe_bignum_add_n(uint64_t *r, uint64_t *a, uint64_t *b, int64_t n)` | `r = a + b` over `n` limbs | carry out (0/1) |
| `int64_t universe_bignum_sub_n(uint64_t *r, uint64_t *a, uint64_t *b, int64_t n)` | `r = a - b` over `n` limbs | borrow out (0/1) |
| `int32_t universe_bignum_cmp_n(uint64_t *a, uint64_t *b, int64_t n)` | Compare `a` vs `b` | -1 / 0 / 1 |
| `int32_t universe_bignum_is_zero_n(uint64_t *a, int64_t n)` | Test `a == 0` | 1 / 0 |
| `int64_t universe_bignum_bit_length_n(uint64_t *a, int64_t n)` | Bit length of `a` | bits |
| `int64_t universe_bignum_shl_bits(uint64_t *r, uint64_t *a, int64_t n, int64_t bits)` | `r = a << bits` (bits in [0,63]) | overflow bits out |
| `void universe_bignum_shr_bits(uint64_t *r, uint64_t *a, int64_t n, int64_t bits)` | `r = a >> bits` (bits in [0,63]) | — |
| `void universe_bignum_mul(uint64_t *r, uint64_t *a, int64_t an, uint64_t *b, int64_t bn)` | `r = a * b` (`r` has `an+bn` limbs) | — |
| `int32_t universe_bignum_divmod(uint64_t *q, uint64_t *r, uint64_t *u, int64_t m, uint64_t *v, int64_t n)` | `q = u / v`, `r = u % v` | 0 / 8 (div by zero) |
| `int32_t universe_bignum_mod(uint64_t *r, uint64_t *u, int64_t m, uint64_t *v, int64_t n)` | `r = u % v` | 0 / 8 |
| `int64_t universe_bignum_mont_n0inv(uint64_t *n)` | `-n[0]^-1 mod 2^64` (odd modulus) | n0inv |
| `void universe_bignum_mont_rr(uint64_t *rr, uint64_t *n, int64_t s)` | `R^2 mod n`, `R = 2^(64*s)` (setup) | — |
| `void universe_bignum_montmul(uint64_t *r, uint64_t *a, uint64_t *b, uint64_t *n, int64_t s, int64_t n0inv, uint64_t *t)` | `r = a*b*R^-1 mod n` (CIOS); `t` scratch of `s+2` limbs | — |
| `int32_t universe_bignum_modexp(uint64_t *r, uint64_t *base, int64_t basel, uint64_t *exp, int64_t expl, uint64_t *n, int64_t s)` | `r = base^exp mod n` | 0 / 2 / 8 |
| `int32_t universe_bignum_modinv(uint64_t *r, uint64_t *a, uint64_t *n, int64_t s)` | `r = a^-1 mod n` | 0 / 5 (no inverse) / 8 |
| `int32_t universe_bignum_gcd(uint64_t *g, uint64_t *a, uint64_t *b, int64_t s)` | `g = gcd(a, b)` | 0 / 2 / 8 |

## Use in an LLVM-based environment
```llvm
declare ptr @universe_bignum_alloc(i64)
declare void @universe_bignum_set_u64(ptr, i64, i64)
declare i64 @universe_bignum_add_n(ptr, ptr, ptr, i64)
declare void @universe_bignum_free(ptr)

define i32 @main() {
  %a = call ptr @universe_bignum_alloc(i64 4)
  %b = call ptr @universe_bignum_alloc(i64 4)
  %r = call ptr @universe_bignum_alloc(i64 4)
  call void @universe_bignum_set_u64(ptr %a, i64 4, i64 123)
  call void @universe_bignum_set_u64(ptr %b, i64 4, i64 456)
  %carry = call i64 @universe_bignum_add_n(ptr %r, ptr %a, ptr %b, i64 4)
  call void @universe_bignum_free(ptr %a)
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`modexp_cli.c` — RSA-style `base^exp mod n` on small hex-free decimal inputs
(single-limb) to show the modexp path.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint64_t *universe_bignum_alloc(int64_t nlimbs);
void      universe_bignum_free(uint64_t *p);
void      universe_bignum_set_u64(uint64_t *r, int64_t s, int64_t v);
int32_t   universe_bignum_modexp(uint64_t *r, uint64_t *base, int64_t basel,
                                 uint64_t *exp, int64_t expl,
                                 uint64_t *n, int64_t s);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s base exp mod\n", argv[0]); return 2; }
    int64_t s = 1;  /* one limb each (values must fit u64 and mod be odd) */
    uint64_t *b = universe_bignum_alloc(s), *e = universe_bignum_alloc(s);
    uint64_t *n = universe_bignum_alloc(s), *r = universe_bignum_alloc(s);
    universe_bignum_set_u64(b, s, atoll(argv[1]));
    universe_bignum_set_u64(e, s, atoll(argv[2]));
    universe_bignum_set_u64(n, s, atoll(argv[3]));
    int rc = universe_bignum_modexp(r, b, s, e, s, n, s);
    if (rc != 0) { fprintf(stderr, "modexp rc=%d\n", rc); return 1; }
    printf("%llu\n", (unsigned long long)r[0]);
    universe_bignum_free(b); universe_bignum_free(e);
    universe_bignum_free(n); universe_bignum_free(r);
    return 0;
}
```
```
clang -O3 modexp_cli.c build/libuniverse.a -lpthread -lm -o modexp_cli
./modexp_cli 4 13 497    # 4^13 mod 497
```

## Notes
- Caller owns all limb buffers (from `universe_bignum_alloc` or your own). Output
  buffers must be sized per operation: `mul` needs `an+bn` limbs, `montmul`
  scratch needs `s+2` limbs, division `q`/`r` sized to dividend/divisor.
- Montgomery routines require an ODD modulus (`n0inv` via Newton iteration).
- Bit shifts take an amount in [0,63]; limb-granular shifts are the caller's job
  via buffer offsetting.
- Not constant-time (see HARDENING-TODO). Single-threaded; no atomics.
