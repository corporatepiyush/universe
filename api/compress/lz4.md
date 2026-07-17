# compress/lz4 — LZ4 block codec

## Purpose
An LZ4 block codec: raw-block decompress plus a single-pass greedy compressor.
LZ4 has no entropy coder — it is byte-oriented LZ77 with a compact token, so it
is very fast. Both directions are ONE-SHOT and whole-buffer: input and output
are caller buffers with an explicit cap, no streaming, no per-item allocation
(the only heap use is a scratch hash table inside the compressor, freed before
return). Decode processes sequences left to right; every read is bounds-checked
against `slen` and every write against `dcap` BEFORE it happens, so hostile or
truncated input returns a negative error and never touches out-of-bounds memory
(ASan-verified). Match copy is overlap-correct (memcpy when `offset>=len`,
forward byte copy for RLE runs). Encode is a greedy pass with a 4-byte-hash
64 KiB-window match table; the final 5 bytes are always literals, so it emits
spec-valid blocks the reference LZ4 decoder accepts.

## Exported API
Returns `int64_t`: decoded/compressed length on success, or NEGATIVE error:
`-1` NULL, `-2` OOM (table alloc), `-6` output exceeds cap (FULL), `-13`
malformed block (bad/zero offset), `-15` truncated input (IO).

| C signature | Description |
|---|---|
| `int64_t universe_compress_lz4_decode(void *dst, int64_t dcap, const void *src, int64_t slen)` | Decompress a raw LZ4 block |
| `int64_t universe_compress_lz4_encode(void *dst, int64_t dcap, const void *src, int64_t slen)` | Compress to a raw LZ4 block |
| `int64_t universe_compress_lz4_bound(int64_t slen)` | Worst-case compressed size for `slen` input bytes |

## Use in an LLVM-based environment
```llvm
declare i64 @universe_compress_lz4_encode(ptr, i64, ptr, i64)
declare i64 @universe_compress_lz4_decode(ptr, i64, ptr, i64)
declare i64 @universe_compress_lz4_bound(i64)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`lz4_cli.c` — round-trip stdin: `-c` compress, `-d` decompress; here a simple
self-check that compresses stdin then decodes and verifies.
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int64_t universe_compress_lz4_encode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_lz4_decode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_lz4_bound(int64_t slen);

int main(void) {
    size_t cap = 1 << 20, n = 0;
    unsigned char *in = malloc(cap);
    size_t r;
    while ((r = fread(in + n, 1, cap - n, stdin)) > 0) {
        n += r;
        if (n == cap) { cap <<= 1; in = realloc(in, cap); }
    }
    int64_t bound = universe_compress_lz4_bound(n);
    unsigned char *comp = malloc(bound);
    int64_t clen = universe_compress_lz4_encode(comp, bound, in, n);
    if (clen < 0) { fprintf(stderr, "encode %lld\n", (long long)clen); return 1; }
    unsigned char *back = malloc(n + 1);
    int64_t dlen = universe_compress_lz4_decode(back, n, comp, clen);
    if (dlen < 0) { fprintf(stderr, "decode %lld\n", (long long)dlen); return 1; }
    int ok = (dlen == (int64_t)n) && (memcmp(in, back, n) == 0);
    fprintf(stderr, "in=%zu comp=%lld roundtrip=%s\n",
            n, (long long)clen, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
```
```
clang -O3 lz4_cli.c build/libuniverse.a -lpthread -lm -o lz4_cli
head -c 100000 /dev/urandom | ./lz4_cli
```

## Notes
- Raw LZ4 *block* format (no frame header/magic, no checksums).
- Use `universe_compress_lz4_bound(n)` to size the compression output buffer.
- Thread-safe/reentrant; the compressor's scratch table is heap-allocated and
  freed internally (hence the `-2` OOM path).
