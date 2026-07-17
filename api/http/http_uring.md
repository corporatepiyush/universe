# http/http_uring — io_uring-driven HTTP/1.1 server (Linux only)

## Purpose

An io_uring-driven HTTP/1.1 server with the SAME semantics as the blocking
`universe_http_accept_loop` path, but the transport is a single SQ/CQ event loop
(via `src/ioring`) instead of thread-per-connection blocking recv/send. Request
PARSING and response SERIALIZATION are reused verbatim from
[`http`](http.md) (`universe_http_parse_request` +
`universe_http_write_response`); only byte movement is swapped to
`prep_recv`/`prep_send`. Each turn is three phases: (1) one
`io_uring_enter` that submits every SQE queued last turn and waits for ≥1
completion; (2) drain ALL ready CQEs, parsing zero-copy over each connection's
persistent recv buffer and serializing into its persistent send buffer;
(3) the next turn's submit flushes the SQEs queued while draining — no syscall
inside the parse/serialize compute. Single-threaded async (no worker pool, no
locks). **LINUX ONLY at runtime**; on other targets it compiles but
`universe_http_uring_available()` returns false so callers use the posix path.

The completion cookie is `USER_DATA = (slot << 3) | op`, `op` ∈ {0 ACCEPT,
1 RECV, 2 SEND, 3 CLOSE}, `slot` = connection index. The `*_action` helpers are
the pure state-machine transitions the loop drives — also usable standalone.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `bool universe_http_uring_available(void)` | Forwards `ioring_ring_available()` | true only on a supporting Linux kernel |
| `int64_t universe_http_uring_ud_encode(int64_t slot, int64_t op)` | Pack `(slot<<3)\|(op&7)` into a user-data cookie | encoded `ud` |
| `int64_t universe_http_uring_ud_op(int64_t ud)` | Extract `op` (`ud & 7`) | op 0–3 |
| `int64_t universe_http_uring_ud_slot(int64_t ud)` | Extract `slot` (`ud >> 3`) | slot index |
| `int32_t universe_http_uring_recv_action(int64_t res, int32_t parse_status, bool body_complete)` | RECV-completion transition | 0 CLOSE, 1 RECV_MORE, 2 DISPATCH |
| `int32_t universe_http_uring_send_action(int64_t res, bool fully_sent, int32_t keepalive)` | SEND-completion transition | 0 CLOSE, 1 SEND_MORE, 2 KEEPALIVE_RECV |
| `int32_t universe_http_uring_serve(int32_t listenfd, void *handler, void *userdata, int64_t bufcap, int64_t entries, int64_t maxreq)` | Run the io_uring server loop on `listenfd` | 0 OK (served `maxreq` / clean stop), 2 OOM, codes |

`handler` is the same 5-arg callback as the blocking server (`void
handler(void *userdata, void *req_msg, void *hdrs, int64_t count, void
*resp_spec)`). `listenfd` comes from `universe_net_tcp_listen`. `maxreq` bounds
how many requests are served before a clean stop (0 / large = run indefinitely).

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i1  @universe_http_uring_available()
declare i32 @universe_http_uring_serve(i32, ptr, ptr, i64, i64, i64)
declare i32 @universe_net_tcp_listen(i32, i32, i32)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A tiny fixed-response server: listen, then serve requests via io_uring on Linux
(fall back to the blocking loop elsewhere).

```c
// uring_server.c   -- io_uring path is Linux-only at runtime
#include <stdint.h>
#include <string.h>
int      universe_http_uring_available(void);
int32_t  universe_http_uring_serve(int32_t, void *, void *, int64_t, int64_t, int64_t);
int32_t  universe_http_accept_loop(int32_t, void *, void *, int64_t, int64_t);
int32_t  universe_net_tcp_listen(int32_t, int32_t, int32_t);

static const char BODY[] = "hello\n";
// resp_spec is 56 B: status@0, reason_ptr@8, reason_len@16,
//                    hdrs_ptr@24, hdrs_count@32, body_ptr@40, body_len@48
static void handler(void *ud, void *req, void *hdrs, int64_t count, void *rs) {
    char *p = (char *)rs;
    *(int64_t *)(p + 0)  = 200;
    *(const char **)(p + 8) = "OK"; *(int64_t *)(p + 16) = 2;
    *(void **)(p + 24) = 0;  *(int64_t *)(p + 32) = 0;      // no extra headers
    *(const char **)(p + 40) = BODY; *(int64_t *)(p + 48) = sizeof BODY - 1;
}

int main(void) {
    int lfd = universe_net_tcp_listen(0x7F000001, 8080, 128);  // 127.0.0.1:8080
    if (lfd < 0) return 1;
    if (universe_http_uring_available())
        return universe_http_uring_serve(lfd, handler, 0, 1 << 16, 256, 1000);
    return universe_http_accept_loop(lfd, handler, 0, 1 << 16, 256);
}
```

```
clang -O3 uring_server.c build/libuniverse.a -lpthread -lm -o uring_server
./uring_server &        # then: curl http://127.0.0.1:8080/   (Linux for the uring path)
```

## Notes

- **Platform:** runtime-Linux-only. Always gate on `uring_available()`;
  otherwise use `http_accept_loop`.
- **Reuse:** parsing/serialization are the exact `http` module functions; this
  module only swaps the transport. Per-slot recv buffer + send writer are
  allocated lazily and kept across keep-alive and slot reuse.
- **Single-threaded async:** no worker pool, no locks, no atomics of its own;
  the only cross-CPU ordering is the io_uring ring protocol.
- **Cookie helpers** (`ud_*`, `*_action`) are pure and independently testable —
  useful when writing a custom event loop over `src/ioring`.
