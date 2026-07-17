# simd — reusable SIMD scan / compare / transform kernels

Shared data-parallel primitives (memchr, memcmp, case-fold, ascii-validate) that
other domains call. Every op is SIMD-first — a portable 128-bit `<16 x i8>` loop
(SSE2 on AMD64, NEON on AArch64, baseline everywhere, no runtime check) — with a
`*_scalar` twin that handles the sub-16 tail and is the bit-identical cross-check
oracle. Pure compute over caller memory; never allocates. Symbols are
`universe_simd_*`, C ABI, `nounwind`. Link against `build/libuniverse.a`.

| Module | Purpose |
|---|---|
| [scan](scan.md) | find_byte / find_crlf / index_of_any / count_byte / equal / compare / to_lower / to_upper / is_ascii / validate_ascii (+ `_scalar` twins). |
