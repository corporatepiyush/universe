# universe_varint

## Purpose

Variable-length integer codec (LEB128 family) on a 64-bit word. Pure compute over
caller buffers; never allocates. Three encodings: **ULEB128** (unsigned, 1..10
bytes), **SLEB128** (signed, sign-extended terminal group, 1..10 bytes), and
**zig-zag** (maps signed→unsigned so small negatives stay short; pair with ULEB128
for a compact signed wire form). `uleb_len` is branch-free via `ctlz`. Decode is
bounds-safe (reads at most `n` bytes / 10 groups) and shift-guarded (select-clamp
on shift < 64) so a garbage stream can never form a poison `shl ≥ 64`.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_varint_uleb_len` | `int64_t universe_varint_uleb_len(int64_t val)` | Byte length of `val` in ULEB128. | 1..10 |
| `universe_varint_sleb_len` | `int64_t universe_varint_sleb_len(int64_t val)` | Byte length in SLEB128. | 1..10 |
| `universe_varint_uleb_encode` | `int64_t universe_varint_uleb_encode(void *dst, int64_t val)` | Write ULEB128. | bytes written |
| `universe_varint_sleb_encode` | `int64_t universe_varint_sleb_encode(void *dst, int64_t val)` | Write SLEB128. | bytes written |
| `universe_varint_uleb_decode` | `int64_t universe_varint_uleb_decode(const void *src, int64_t n, int64_t *out)` | Read ULEB128 (≤ `n` bytes). | bytes consumed, or -13 / -3 |
| `universe_varint_sleb_decode` | `int64_t universe_varint_sleb_decode(const void *src, int64_t n, int64_t *out)` | Read SLEB128. | bytes consumed, or -13 / -3 |
| `universe_varint_zigzag_encode` | `int64_t universe_varint_zigzag_encode(int64_t v)` | Signed → zig-zag unsigned. | encoded value |
| `universe_varint_zigzag_decode` | `int64_t universe_varint_zigzag_decode(int64_t u)` | Zig-zag → signed. | decoded value |

Decode returns -13 (PARSE) for truncated input and -3 (SIZE_OVERFLOW) for a value
too large for 64 bits; `*out` is written only on success. Size `dst` with
`*_len` (max 10 bytes).

## Use in an LLVM-based environment

```llvm
declare i64 @universe_varint_uleb_encode(ptr, i64)
declare i64 @universe_varint_uleb_decode(ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Round-trip: ULEB128-encode a number from argv, then decode it back.

```c
// varintcli.c — ULEB128 round-trip of argv[1]
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_varint_uleb_encode(void*, int64_t);
int64_t universe_varint_uleb_decode(const void*, int64_t, int64_t*);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <uint>\n", argv[0]); return 2; }
    int64_t v = atoll(argv[1]);
    unsigned char buf[10];
    int64_t nw = universe_varint_uleb_encode(buf, v);
    printf("encoded %lld bytes:", (long long)nw);
    for (int64_t i = 0; i < nw; i++) printf(" %02x", buf[i]);
    int64_t out = 0, nr = universe_varint_uleb_decode(buf, nw, &out);
    printf("\ndecoded %lld (consumed %lld)\n", (long long)out, (long long)nr);
    return 0;
}
```

```
clang -O3 varintcli.c build/libuniverse.a -lpthread -lm -o varintcli
./varintcli 300        # encoded 2 bytes: ac 02 / decoded 300 (consumed 2)
```

## Notes

- No allocation; `dst` caller-owned, ≤ 10 bytes. `out` written only on a
  successful decode. The `zigzag_*` helpers are pure ALU, exported so callers can
  compose their own signed wire form (zig-zag + ULEB).
- Reentrant, stateless.
