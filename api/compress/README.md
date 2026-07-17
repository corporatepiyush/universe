# compress — compression / decompression

Exported as `universe_compress_*`. C ABI, `nounwind`. Link against
`build/libuniverse.a`:
`clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.

All codecs are ONE-SHOT and whole-buffer: input and output are caller buffers
with an explicit output cap, no streaming. Every read/write is bounds-checked
before it happens (ASan-verified on hostile input). Codec functions return an
`int64_t` byte length on success, or a NEGATIVE error (`-1` NULL, `-2` OOM,
`-6` output exceeds cap, `-13` malformed, `-14` unsupported, `-15` truncated).

| Module | What |
|---|---|
| [inflate](inflate.md) | DEFLATE / zlib / gzip decode (RFC 1951/1950/1952) + adler32/crc32 + stored/fixed encoders |
| [lz4](lz4.md) | LZ4 block decode + greedy encode; `_bound` for output sizing |
| [snappy](snappy.md) | Snappy raw-block (Parquet SNAPPY) decode + encode; `_bound` |
| [zstd](zstd.md) | Zstandard frame decode — RAW/RLE blocks only (compressed → `-14`) |
| [zip](zip.md) | Zero-copy ZIP reader (open / count / entry / extract) over a caller buffer |

The encoder-bearing codecs (lz4, snappy) provide a `_bound(n)` helper to size
the compression output buffer.
