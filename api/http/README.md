# http — HTTP/1.1 client + server

Plaintext HTTP/1.1 (NO TLS, deferred to hardening). Zero-copy head parsing over
a caller-owned buffer, SIMD line/token scans (`src/simd`), buffered output
(`src/io`), sockets (`src/net`), keep-alive pooling (`src/net/pool`). C-ABI
`nounwind`; link against `build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [http](http.md) | Client + blocking server: parse/serialize, conn objects, keep-alive, chunked, pooled requests |
| [http_uring](http_uring.md) | Same server semantics over an io_uring event loop — **Linux only at runtime** |

Both share the message-struct / header-entry / response-spec layouts documented
in [http](http.md). The blocking path runs everywhere; the io_uring path is
gated on `universe_http_uring_available()`. Single-threaded (concurrency
deferred). Sockets go through `net/tcp`, so the Linux-only socket-constant
portability seam is handled by the build-selected `net/osconst_*` (see the
[net README](../net/README.md)).
