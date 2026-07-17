# compress/inflate — DEFLATE / zlib / gzip decoder + checksums

## Purpose
A DEFLATE / zlib / gzip decoder (RFC 1951 / 1950 / 1952) plus adler32 / crc32
and a stored-block and fixed-Huffman DEFLATE encoder (for round-trip tests).
Decode is ONE-SHOT and whole-buffer: the entire compressed input sits in a
caller buffer and the entire output goes to a caller buffer with an explicit
size cap — no streaming, no allocation (all scratch is stack allocas sized to
RFC maxima). The LSB-first bit reader never shifts past 64 bits; canonical
Huffman decode is the count/first/index walk (no lookup table); LZ77 back-copies
are overlap-correct (memcpy when `dist>=len`, forward byte copy for RLE runs).

## Exported API
Every function returns `int64_t`: the decoded/encoded length on success, or a
NEGATIVE error: `-1` NULL, `-6` output exceeds cap (FULL), `-13` malformed
stream / checksum mismatch (PARSE), `-14` unsupported wrapper, `-15` truncated
(IO). Checksums return the 32-bit value in an `int64_t`.

| C signature | Description |
|---|---|
| `int64_t universe_compress_inflate(void *dst, int64_t dstcap, const void *src, int64_t srclen)` | Decode raw DEFLATE |
| `int64_t universe_compress_inflate_zlib(void *dst, int64_t dstcap, const void *src, int64_t srclen)` | Decode zlib-wrapped DEFLATE (adler32 checked) |
| `int64_t universe_compress_inflate_gzip(void *dst, int64_t dstcap, const void *src, int64_t srclen)` | Decode gzip-wrapped DEFLATE (crc32 checked) |
| `int64_t universe_compress_adler32(const void *data, int64_t len)` | Adler-32 of `data` |
| `int64_t universe_compress_crc32(const void *data, int64_t len)` | CRC-32 (zlib polynomial) of `data` |
| `int64_t universe_compress_deflate_stored(void *dst, int64_t dstcap, const void *src, int64_t n)` | Encode as stored (uncompressed) DEFLATE blocks |
| `int64_t universe_compress_deflate_fixed(void *dst, int64_t dstcap, const void *src, int64_t n)` | Encode as fixed-Huffman DEFLATE |

## Use in an LLVM-based environment
```llvm
declare i64 @universe_compress_inflate_gzip(ptr, i64, ptr, i64)
declare i64 @universe_compress_crc32(ptr, i64)

define i32 @main() {
  ; %dst, %src, %caps provided by caller
  ret i32 0
}
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`gunzip_cli.c` — decompress a gzip file from argv[1] to stdout.
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int64_t universe_compress_inflate_gzip(void *dst, int64_t dstcap,
                                       const void *src, int64_t srclen);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.gz\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *src = malloc(n);
    fread(src, 1, n, f); fclose(f);
    int64_t cap = 64 * 1024 * 1024;          /* 64 MiB output cap */
    unsigned char *dst = malloc(cap);
    int64_t out = universe_compress_inflate_gzip(dst, cap, src, n);
    if (out < 0) { fprintf(stderr, "inflate error %lld\n", (long long)out); return 1; }
    fwrite(dst, 1, out, stdout);
    free(src); free(dst);
    return 0;
}
```
```
clang -O3 gunzip_cli.c build/libuniverse.a -lpthread -lm -o gunzip_cli
printf 'hello world' | gzip | ./gunzip_cli /dev/stdin
```

## Notes
- Whole-buffer, zero-allocation: caller supplies both buffers and the output
  cap. Sizing the output too small returns `-6`, never an overflow.
- The zlib/gzip wrappers verify the trailing checksum; a mismatch returns `-13`.
- The encoders are for round-trip verification, not ratio; they emit
  spec-valid streams the decoder (and reference tools) accept.
- Thread-safe (no shared state); reentrant.
