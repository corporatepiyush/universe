# parse — zero-copy structured-text parsers

Zero-copy, zero-alloc parsers over caller-owned byte buffers: tokens are
reported as offsets/lengths into the buffer, never copied. Hot scans are
SIMD-first (128-bit SSE2/NEON) with scalar fallbacks. Symbols are
`universe_parse_*`, C ABI, `nounwind`. Link against `build/libuniverse.a`.

| Module | Purpose |
|---|---|
| [csv](csv.md) | RFC 4180 CSV/TSV field iterator; caller-owned 32-byte scanner. |
| [json](json.md) | RFC 8259 JSON pull/SAX tokenizer; recursion-free, depth-bounded. |
