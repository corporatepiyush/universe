# strings — byte buffers and UTF-8 string types

Byte/string building blocks. Bytes are opaque, so multibyte UTF-8 is handled
transparently. Hot scans delegate to the `simd/scan` kernels. C ABI, `nounwind`;
link against `build/libuniverse.a`.

| Module | Symbols | Purpose |
|---|---|---|
| [bytes](bytes.md) | `universe_bytes_*` | Growable byte buffer (stable handle, 2× growth, zero-copy view). |
| [string](string.md) | `universe_string_*`, `universe_sso_*` | Immutable borrowed string view + 24-byte SSO owned string (inline ≤ 22 bytes). |
