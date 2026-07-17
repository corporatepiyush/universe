# compress/snappy — Snappy raw block codec

## Purpose
A Snappy raw-block codec: block decompress plus a single-pass greedy
compressor. This is the framed-block Snappy that Parquet's `SNAPPY` codec uses —
a varint decompressed-length preamble followed by literal/copy elements — NOT
the stream/CRC framing. Both directions are ONE-SHOT and whole-buffer with an
explicit output cap; no streaming, no per-item allocation (only a scratch hash
table in the compressor, freed before return). Decode parses the varint length,
rejects a length beyond the cap or 32 bits, then replays elements; every read is
bounds-checked against `slen` and every write against the declared length BEFORE
it happens (ASan-verified on hostile input). Copies are overlap-correct. Encode
writes the varint preamble then a greedy pass with a 4-byte-hash 64 KiB-window
match table, emitting 2-byte-offset copies; every block is spec-valid and the
reference Snappy decoder accepts it.

## Exported API
Returns `int64_t`: decoded/compressed length on success, or NEGATIVE error:
`-1` NULL, `-2` OOM (table alloc), `-6` output exceeds cap (FULL), `-13`
malformed block (PARSE), `-15` truncated input (IO).

| C signature | Description |
|---|---|
| `int64_t universe_compress_snappy_decode(void *dst, int64_t dcap, const void *src, int64_t slen)` | Decompress a Snappy block |
| `int64_t universe_compress_snappy_encode(void *dst, int64_t dcap, const void *src, int64_t slen)` | Compress to a Snappy block |
| `int64_t universe_compress_snappy_bound(int64_t slen)` | Worst-case compressed size for `slen` input bytes |

## Use in an LLVM-based environment
```llvm
declare i64 @universe_compress_snappy_encode(ptr, i64, ptr, i64)
declare i64 @universe_compress_snappy_decode(ptr, i64, ptr, i64)
declare i64 @universe_compress_snappy_bound(i64)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`snappy_cli.c` — compress stdin to a Snappy block, write it to stdout; `-d`
decodes.
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int64_t universe_compress_snappy_encode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_snappy_decode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_snappy_bound(int64_t slen);

int main(int argc, char **argv) {
    int decode = (argc > 1 && strcmp(argv[1], "-d") == 0);
    size_t cap = 1 << 20, n = 0;
    unsigned char *in = malloc(cap), *out;
    size_t r;
    while ((r = fread(in + n, 1, cap - n, stdin)) > 0) {
        n += r;
        if (n == cap) { cap <<= 1; in = realloc(in, cap); }
    }
    int64_t rc;
    if (decode) {
        size_t ocap = 64 * 1024 * 1024; out = malloc(ocap);
        rc = universe_compress_snappy_decode(out, ocap, in, n);
    } else {
        int64_t b = universe_compress_snappy_bound(n); out = malloc(b);
        rc = universe_compress_snappy_encode(out, b, in, n);
    }
    if (rc < 0) { fprintf(stderr, "error %lld\n", (long long)rc); return 1; }
    fwrite(out, 1, rc, stdout);
    return 0;
}
```
```
clang -O3 snappy_cli.c build/libuniverse.a -lpthread -lm -o snappy_cli
printf 'aaaaaaaaaabbbbbbbbbb' | ./snappy_cli | ./snappy_cli -d
```

## Notes
- Framed-block Snappy (varint length + elements), not the stream framing.
- Size the encode buffer with `universe_compress_snappy_bound(n)`.
- Thread-safe/reentrant; compressor scratch table is heap-allocated and freed
  internally.
