# net — TCP sockets + connection pool

Blocking IPv4 TCP primitives and a single-threaded connection pool. C-ABI
`nounwind`; link against `build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [tcp](tcp.md) | Blocking IPv4 sockets: listen/accept/connect/send_all/recv, NODELAY, non-block, shutdown/close |
| [pool](pool.md) | Idle-connection pool keyed by `(ip,port)`, LRU eviction, MSG_PEEK liveness check |

Return convention: a live fd is `≥ 0`; a NEGATIVE value is the negated SDK error
code (`-15` IO, `-8` INVALID_ARG). `ip`/`port` are host-order `int32`
(`127.0.0.1` = `0x7F000001`). Single-threaded (concurrency deferred).

## Build-selected OS constants (`osconst_*`, not a public API)

TCP portability across macOS / FreeBSD / Linux hinges on OS-divergent socket
constants and struct layouts that cannot be `#ifdef`'d in target-agnostic IR
(`SOL_SOCKET` is `0xffff` on BSD but `1` on Linux; `O_NONBLOCK` `4` vs `0x800`;
`sockaddr_in` carries a leading `sin_len` byte on BSD but not Linux). These are
isolated in two symbol-for-symbol twins, `src/net/osconst_bsd.ll` and
`src/net/osconst_linux.ll`, of which the Makefile links **exactly one per host**
(selected by `uname`). They expose internal helpers
(`universe_net_os_build_sockaddr`, `_enable_reuseaddr`, `_errno`,
`_nonblock_flags`, `_msg_peek_dontwait`) consumed by `tcp.ll`/`pool.ll` — they
are a build-time portability seam, **not** part of the public API and are not
documented per-symbol here. `net/tcp` itself bakes no OS constant, so its IR is
byte-identical on every target.
