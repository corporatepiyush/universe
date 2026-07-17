# encoding — universe API

Byte-level codecs and wire-format readers: hex, base64, UTF-8/UTF-16, varint
(LEB128), and zero-copy FlatBuffers / Thrift-compact decoders. Pure compute over
caller-supplied byte ranges — the caller pre-sizes destination buffers (via the
`*_len` helpers) and these modules never allocate. Shipped in `libuniverse.a`
(`make lib`); C ABI, `nounwind`.

> **Naming note.** The classic codecs export short names —
> `universe_hex_*`, `universe_base64_*`, `universe_utf8_*`, `universe_utf16_*`,
> `universe_varint_*` — while the newer wire readers use the domain prefix:
> `universe_encoding_flatbuffers_*` and `universe_encoding_thrift_*`.

## Modules

| Module | What it is |
| --- | --- |
| [hex](hex.md) | Hex (base16) encode/decode, branchless. |
| [base64](base64.md) | Base64 encode/decode, standard + url-safe, SIMD-first. |
| [utf8](utf8.md) | UTF-8 RFC 3629 validation + codepoint counting. |
| [utf16](utf16.md) | UTF-16 (LE/BE) codec + UTF-8 ⇄ UTF-16 transcoding. |
| [varint](varint.md) | LEB128 (ULEB/SLEB) + zig-zag variable-length integers. |
| [flatbuffers](flatbuffers.md) | Zero-copy FlatBuffers binary-wire reader. |
| [thrift](thrift.md) | Zero-copy Thrift compact-protocol reader. |

## Common conventions

- **Caller pre-sizes `dst`.** Use `encode_len` / `decode_len` (or `len_*`
  helpers) to size the destination before calling; no allocation happens inside.
- **`i64` error convention** (hex / base64 / varint / utf transcode): return the
  produced/consumed byte count on success, a NEGATIVE value on failure (`-1`
  PARSE / `-3` SIZE_OVERFLOW / `-13` PARSE per module).
- **Zero-copy readers** (flatbuffers / thrift) return offsets or `(ptr,len)` views
  into the caller buffer; `i32` entries use the universe codes (0 OK, 1 NULL_PTR,
  8 INVALID_ARG). Every read is bounds-checked (sanitizer-gated).
- Hot data-parallel kernels are SIMD-first (128-bit vector primary) with a scalar
  fallback that is also the cross-check oracle.

Link line: `clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.
