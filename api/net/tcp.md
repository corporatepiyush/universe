# net/tcp — blocking POSIX TCP sockets

## Purpose

Thin, zero-allocation wrappers over the kernel socket API for blocking IPv4 TCP
(127.0.0.1-friendly). Listen/accept/connect/send/recv plus TCP_NODELAY and
non-blocking toggles and clean shutdown/close. The module bakes NO OS constant:
every OS-divergent primitive (errno accessor, `sockaddr_in` byte layout,
`SOL_SOCKET`/`SO_REUSEADDR`/`O_NONBLOCK`/`MSG_DONTWAIT` values) is delegated to
`src/net/osconst_{bsd,linux}.ll`, of which the build links exactly one per host,
so the IR is byte-identical and correct on macOS, FreeBSD, and Linux.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_net_tcp_listen(int32_t ip, int32_t port, int32_t backlog)` | Create a listening socket bound to `ip:port` | listen fd (≥0), or negated error |
| `int32_t universe_net_tcp_accept(int32_t listenfd, int32_t *out)` | Accept one connection; `*out` = peer IPv4 addr | conn fd (≥0), or negated error |
| `int32_t universe_net_tcp_connect(int32_t ip, int32_t port)` | Blocking connect to `ip:port` | conn fd (≥0), or negated error |
| `int32_t universe_net_tcp_set_nodelay(int32_t fd, int32_t on)` | Toggle `TCP_NODELAY` | 0 OK, or negated error |
| `int32_t universe_net_tcp_set_nonblocking(int32_t fd, int32_t on)` | Toggle `O_NONBLOCK` via F_GETFL/F_SETFL | 0 OK, or negated error |
| `int32_t universe_net_tcp_shutdown(int32_t fd, int32_t how)` | `shutdown(fd, how)` (how 0/1/2 = RD/WR/RDWR) | 0 OK, or negated error |
| `int32_t universe_net_tcp_close(int32_t fd)` | `close(fd)` | 0 OK, or negated error |
| `int64_t universe_net_tcp_send_all(int32_t fd, const void *buf, int64_t len)` | Send all `len` bytes (partial-write loop) | bytes sent, or negated error |
| `int64_t universe_net_tcp_recv(int32_t fd, void *buf, int64_t len)` | One `recv` of up to `len` bytes | bytes read (0 = EOF), or negated error |

`ip` and `port` are host-order `int32` (e.g. `127.0.0.1` = `0x7F000001`). A
NEGATIVE return is the negated SDK error code (`-15` = IO, `-8` = INVALID_ARG);
callers test `< 0`. Live fds are always `≥ 0`, so the sign channel is
unambiguous.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_net_tcp_connect(i32, i32)
declare i64 @universe_net_tcp_send_all(i32, ptr, i64)
declare i64 @universe_net_tcp_recv(i32, ptr, i64)
declare i32 @universe_net_tcp_close(i32)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Minimal HTTP GET: connect to `127.0.0.1:<port>`, send a request line, print the
reply.

```c
// tget.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int32_t universe_net_tcp_connect(int32_t, int32_t);
int64_t universe_net_tcp_send_all(int32_t, const void *, int64_t);
int64_t universe_net_tcp_recv(int32_t, void *, int64_t);
int32_t universe_net_tcp_close(int32_t);

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : 8080;
    int fd = universe_net_tcp_connect(0x7F000001, port);  // 127.0.0.1
    if (fd < 0) { fprintf(stderr, "connect failed: %d\n", fd); return 1; }
    const char *req = "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n";
    universe_net_tcp_send_all(fd, req, strlen(req));
    char buf[4096];
    int64_t n;
    while ((n = universe_net_tcp_recv(fd, buf, sizeof buf)) > 0)
        fwrite(buf, 1, n, stdout);
    universe_net_tcp_close(fd);
    return 0;
}
```

```
clang -O3 tget.c build/libuniverse.a -lpthread -lm -o tget
./tget 8080
```

## Notes

- **IPv4 only**, blocking sockets; addresses/ports are host-order `int32`.
- **Portability:** all OS-divergent socket constants and `sockaddr_in` layout
  live in the build-selected `osconst_{bsd,linux}.ll` — this module is
  byte-identical across macOS/FreeBSD/Linux. See the [net README](README.md).
- **Ownership:** callers own returned fds and must `close` them (or hand them to
  `net/pool`). Zero allocation; no hidden state.
- **Threading:** stateless wrappers, safe to call from multiple threads on
  distinct fds (concurrency of higher layers is deferred).
