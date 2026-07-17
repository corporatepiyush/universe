# compress/zstd — Zstandard frame decoder (RAW/RLE subset)

## Purpose
A Zstandard (RFC 8878) frame decoder for the RAW and RLE block types (PARTIAL).
It parses a full Zstd frame header (magic, Frame_Header_Descriptor,
Window_Descriptor, Dictionary_ID, Frame_Content_Size) and every per-block header,
and fully decodes RAW (type 0, verbatim copy) and RLE (type 1, one byte repeated)
blocks. It intentionally does NOT yet implement COMPRESSED blocks (type 2 — the
FSE/tANS + Huffman entropy stages are a separate wave); a compressed block
returns UNSUPPORTED (`-14`) cleanly, never OOB. ONE-SHOT, whole-buffer, no
allocation: every field read is bounds-checked against `slen` and every write
against `dcap` before it happens (ASan-verified on hostile and truncated input).
The optional 4-byte content checksum is skipped, not verified.

## Exported API
Returns `int64_t`: decoded length on success, or NEGATIVE error: `-1` NULL,
`-6` output exceeds cap (FULL), `-13` malformed frame (bad magic / reserved
block type), `-14` unsupported (compressed block), `-15` truncated input (IO).

| C signature | Description |
|---|---|
| `int64_t universe_compress_zstd_decode(void *dst, int64_t dcap, const void *src, int64_t slen)` | Decode a Zstd frame (RAW/RLE blocks only) |

## Use in an LLVM-based environment
```llvm
declare i64 @universe_compress_zstd_decode(ptr, i64, ptr, i64)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`zstd_cli.c` — decode a Zstd frame from argv[1] to stdout (works on RAW/RLE
frames, e.g. `zstd --no-compress` output).
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int64_t universe_compress_zstd_decode(void *dst, int64_t dcap,
                                      const void *src, int64_t slen);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.zst\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *src = malloc(n);
    fread(src, 1, n, f); fclose(f);
    int64_t cap = 64 * 1024 * 1024;
    unsigned char *dst = malloc(cap);
    int64_t out = universe_compress_zstd_decode(dst, cap, src, n);
    if (out == -14) { fprintf(stderr, "compressed blocks unsupported\n"); return 3; }
    if (out < 0)    { fprintf(stderr, "decode error %lld\n", (long long)out); return 1; }
    fwrite(dst, 1, out, stdout);
    return 0;
}
```
```
clang -O3 zstd_cli.c build/libuniverse.a -lpthread -lm -o zstd_cli
```

## Notes
- Decodes only RAW/RLE blocks; COMPRESSED (entropy-coded) blocks return `-14`.
  This is the honest current state — frame/block framing is solid and
  interop-tested against the reference `zstd` tool.
- Whole-buffer, zero-allocation; caller owns both buffers and the output cap.
- Thread-safe/reentrant.
