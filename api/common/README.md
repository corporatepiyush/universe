# common — shared leaf primitives

Exported as `universe_common_*`. C ABI, `nounwind`, `alwaysinline`. Link against
`build/libuniverse.a`:
`clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.

| Module | What |
|---|---|
| [common](common.md) | Canonical shapes for word rotation (`fsh*`), big-endian word load/store (`bswap`), and overflow-checked size math (`umul/uadd.with.overflow`) |

These are the single source of truth for the most-duplicated hot idioms. Pure,
stateless, reentrant. Cold callers import directly; hot per-word callers
duplicate the inline shape locally until the LTO build variant lands.
