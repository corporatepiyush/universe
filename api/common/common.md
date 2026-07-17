# common/common — shared leaf primitives

## Purpose
Common leaf primitives shared across modules — the canonical, flag-complete
shapes for the three most-duplicated hot idioms: word rotation, big-endian word
I/O, and overflow-checked size math. Each is one LLVM intrinsic wrapped with the
attributes we want everywhere (`alwaysinline`), so every module inherits
identical codegen. They are the single source of truth for shapes that otherwise
drift — a rotation written as `(x<<n)|(x>>(32-n))` is poison when `n==0`, so it
MUST be `llvm.fsh*`; a big-endian load on our little-endian targets is
`load + llvm.bswap`; a size multiply MUST go through `llvm.umul.with.overflow`.
Cold/once-per-op callers import these directly; hot per-word callers (e.g. SHA
compression) still duplicate the inline shape locally until the LTO build lands
(a cross-`.ll` call does not inline in the non-LTO archive build).

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_common_rotl32(int32_t x, int32_t n)` | Rotate-left 32-bit (`n` any value, no poison) | rotated |
| `int32_t universe_common_rotr32(int32_t x, int32_t n)` | Rotate-right 32-bit | rotated |
| `int64_t universe_common_rotl64(int64_t x, int64_t n)` | Rotate-left 64-bit | rotated |
| `int64_t universe_common_rotr64(int64_t x, int64_t n)` | Rotate-right 64-bit | rotated |
| `int32_t universe_common_load_be32(const void *p)` | Load big-endian `uint32_t` from `p` | value |
| `int64_t universe_common_load_be64(const void *p)` | Load big-endian `uint64_t` from `p` | value |
| `void universe_common_store_be32(void *p, int32_t v)` | Store `v` big-endian (4 bytes) | — |
| `void universe_common_store_be64(void *p, int64_t v)` | Store `v` big-endian (8 bytes) | — |
| `int64_t universe_common_checked_mul(int64_t a, int64_t b, int32_t *err)` | `a * b`; `*err = 0` OK, `3` SIZE_OVERFLOW | product |
| `int64_t universe_common_checked_add(int64_t a, int64_t b, int32_t *err)` | `a + b`; `*err = 0` OK, `3` SIZE_OVERFLOW | sum |

## Use in an LLVM-based environment
```llvm
declare i32 @universe_common_rotl32(i32, i32)
declare i64 @universe_common_load_be64(ptr)
declare i64 @universe_common_checked_mul(i64, i64, ptr)

define i32 @main() {
  %r = call i32 @universe_common_rotl32(i32 1, i32 8)
  %err = alloca i32
  %n = call i64 @universe_common_checked_mul(i64 1000000, i64 1000000, ptr %err)
  %e = load i32, ptr %err
  ret i32 %e
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`be_cli.c` — parse a decimal `uint32`, print its big-endian hex bytes, and
demo overflow-checked multiply.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void    universe_common_store_be32(void *p, int32_t v);
int64_t universe_common_checked_mul(int64_t a, int64_t b, int32_t *err);

int main(int argc, char **argv) {
    uint32_t v = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 0) : 0x01020304u;
    unsigned char buf[4];
    universe_common_store_be32(buf, (int32_t)v);
    printf("be bytes: %02x %02x %02x %02x\n", buf[0], buf[1], buf[2], buf[3]);

    int32_t err = 0;
    int64_t a = argc > 2 ? atoll(argv[2]) : 3000000000LL;
    int64_t p = universe_common_checked_mul(a, a, &err);
    if (err) printf("mul overflow (err=%d)\n", err);
    else     printf("mul = %lld\n", (long long)p);
    return 0;
}
```
```
clang -O3 be_cli.c build/libuniverse.a -lpthread -lm -o be_cli
./be_cli 0x01020304 3000000000
```

## Notes
- Pure leaf functions, no allocation, no state — thread-safe and reentrant.
- `checked_mul`/`checked_add` write the error code through the `err` pointer and
  still return the (possibly wrapped) result; always check `*err` first.
- Big-endian I/O is byte-exact on every target (little-endian assumed for the
  host, `llvm.bswap` handles the swap).
