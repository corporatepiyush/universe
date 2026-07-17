# http/http — HTTP/1.1 client + server (plaintext)

## Purpose

Plaintext HTTP/1.1 client and server, single-threaded, NO TLS. Request/response
parsing is a PURE function over a caller-owned contiguous accumulation buffer:
`universe_http_parse_request/response` take `(buf, len)` and return ZERO-COPY
slices (ptr+len into `buf`) for method/target/version/reason and every header —
no syscalls in the parse. Header line boundaries and tokens are located with the
SIMD scans from `src/simd`. Output goes through the `src/io` buffered writer
(accumulate + one flush per message). The connection object owns one buffer that
keeps head + body + next-pipelined bytes contiguous and stable across a handler
call (leftover bytes are `memmove`-compacted between messages), realizing the
"fill buffer (IO) → parse (compute) → emit (IO)" structure. The server is a
blocking accept-then-serve loop; keep-alive reuse is via `src/net/pool`.

**Message struct** (`msg`, 96 B, caller-allocated): `f0_ptr`@0 / `f0_len`@8
(method or reason), `f1_ptr`@16 / `f1_len`@24 (target), `minor`@32, `code`@40
(response status), `header_count`@48, `body_ptr`@56, `body_len`@64,
`head_len`@72, `content_length`@80 (−1 if absent), `flags`@88 (bit0 chunked,
bit1 keep-alive, bit2 content-length-present). **Header entry** (32 B):
`name_ptr`@0, `name_len`@8, `val_ptr`@16, `val_len`@24. **Response-spec** (handler
output, 56 B): `status`@0, `reason_ptr`@8, `reason_len`@16, `hdrs_ptr`@24,
`hdrs_count`@32, `body_ptr`@40, `body_len`@48.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_http_find_head_end(const void *buf, int64_t len)` | Locate the terminating CRLFCRLF | head length, or `-1` if incomplete |
| `int32_t universe_http_parse_request(void *buf, int64_t len, void *msg, void *hdrs, int64_t cap)` | Parse a request into `msg` + `hdrs[cap]` (zero-copy) | 0 OK, 1 NULL, 11 INCOMPLETE, 13 PARSE |
| `int32_t universe_http_parse_response(void *buf, int64_t len, void *msg, void *hdrs, int64_t cap)` | Parse a response into `msg` + `hdrs[cap]` | same codes |
| `int32_t universe_http_header_get(void *hdrs, int64_t count, const void *name, int64_t name_len, void **out_val_ptr, int64_t *out_val_len)` | Case-insensitive header lookup | 0 OK, 5 NOT_FOUND |
| `int32_t universe_http_write_request(void *w, const void *method, int64_t mlen, const void *target, int64_t tlen, int64_t minor, void *hdrs, int64_t count, const void *body, int64_t blen, int32_t keepalive)` | Serialize a request into a `io` writer | 0 OK, `<0`/code on error |
| `int32_t universe_http_write_response(void *w, int64_t status, const void *reason, int64_t rlen, void *hdrs, int64_t count, const void *body, int64_t blen, int32_t keepalive)` | Serialize a response | 0 OK |
| `int32_t universe_http_chunked_encode(void *w, const void *src, int64_t len)` | Write one chunked-transfer chunk | 0 OK |
| `int32_t universe_http_chunked_decode(const void *src, int64_t srclen, void *dst, int64_t dstcap, int64_t *out_len)` | Decode a chunked body into `dst` | 0 OK, 11 INCOMPLETE, 13 PARSE, 6 FULL |
| `void *universe_http_conn_create(int32_t fd, int64_t bufcap)` | Wrap `fd` in an owned conn (buffer + writer) | conn handle, or NULL on OOM |
| `int32_t universe_http_conn_fd(void *conn)` | The connection's fd | fd |
| `void *universe_http_conn_writer(void *conn)` | The conn's `io` buffered writer | writer ptr |
| `void universe_http_conn_destroy(void *conn)` | Free the conn AND close the fd | — |
| `int32_t universe_http_conn_release(void *conn)` | Flush + free the conn WITHOUT closing the fd | fd (for reuse) |
| `void *universe_http_connect(int32_t ip, int32_t port, int64_t bufcap)` | TCP-connect + wrap in a client conn | conn handle, or NULL |
| `int32_t universe_http_conn_read(void *conn, int32_t is_resp, void *msg, void *hdrs, int64_t cap)` | Fill buffer + parse one message | 0 OK, 11 INCOMPLETE, codes |
| `int32_t universe_http_client_request(void *conn, const void *method, int64_t mlen, const void *target, int64_t tlen, int64_t minor, void *hdrs, int64_t count, const void *body, int64_t blen, int32_t keepalive, void *resp_msg, void *resp_hdrs, int64_t resp_cap)` | One request/response round-trip on `conn` | 0 OK, codes |
| `int32_t universe_http_serve_conn(void *conn, void *handler, void *userdata)` | Serve keep-alive requests on one conn via `handler` | 0 OK |
| `int32_t universe_http_accept_loop(int32_t listenfd, void *handler, void *userdata, int64_t bufcap, int64_t max_conns)` | Blocking accept-then-serve server loop | 0 OK, codes |
| `int32_t universe_http_request_pooled(void *pool, int32_t ip, int32_t port, const void *method, int64_t mlen, const void *target, int64_t tlen, int64_t minor, void *hdrs, int64_t count, const void *body, int64_t blen, void *resp_msg, void *resp_hdrs, int64_t resp_cap, int64_t bufcap)` | Round-trip reusing a `net/pool` idle conn | 0 OK, codes |

`handler` signature: `void handler(void *userdata, void *req_msg, void *hdrs,
int64_t count, void *resp_spec)` — it fills the 56-byte response-spec.
`Content-Length`/`Transfer-Encoding`/`Connection` are emitted automatically by
the writers; do not include them in `hdrs`. `ip`/`port` are host-order `int32`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare ptr @universe_http_connect(i32, i32, i64)
declare i32 @universe_http_client_request(ptr, ptr, i64, ptr, i64, i64,
             ptr, i64, ptr, i64, i32, ptr, ptr, i64)
declare i32 @universe_http_conn_release(ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

HTTP GET client against a local server: connect, one round-trip, print the
status and body length.

```c
// httpget.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void   *universe_http_connect(int32_t, int32_t, int64_t);
int32_t universe_http_client_request(void *, const void *, int64_t,
            const void *, int64_t, int64_t, void *, int64_t,
            const void *, int64_t, int32_t, void *, void *, int64_t);
void    universe_http_conn_destroy(void *);

int main(int argc, char **argv) {
    int port = argc > 1 ? atoi(argv[1]) : 8080;
    void *conn = universe_http_connect(0x7F000001, port, 1 << 16);   // 127.0.0.1
    if (!conn) { fprintf(stderr, "connect failed\n"); return 1; }
    char msg[96] = {0};           // response message struct (96 B)
    char hdrs[64 * 32];           // up to 64 header entries (32 B each)
    int rc = universe_http_client_request(conn, "GET", 3, "/", 1, 1,
                 NULL, 0, NULL, 0, /*keepalive*/0, msg, hdrs, 64);
    if (rc == 0) {
        int64_t status  = *(int64_t *)(msg + 40);
        int64_t bodylen = *(int64_t *)(msg + 64);
        printf("status=%lld body_len=%lld\n",
               (long long)status, (long long)bodylen);
    } else {
        fprintf(stderr, "request failed rc=%d\n", rc);
    }
    universe_http_conn_destroy(conn);
    return rc;
}
```

```
clang -O3 httpget.c build/libuniverse.a -lpthread -lm -o httpget
./httpget 8080
```

## Notes

- **No TLS** (deferred to hardening); plaintext only. Single-threaded blocking
  server; no worker pool.
- **Zero-copy parse:** header/body slices point INTO the conn buffer and stay
  valid for the duration of the handler; they are invalidated when the conn
  moves to the next message. Copy anything you must retain.
- **Layout:** conn is one owned allocation (64-byte header + buffer@64) holding
  fd, cap, len, pos, and the buffered writer.
- **Auto headers:** the writers emit `Content-Length`/`Transfer-Encoding`/
  `Connection` — do not duplicate them in `hdrs`.
- **Portability:** sockets come from `net/tcp`, so the Linux-only socket-constant
  concern is handled by the build-selected `net/osconst_*` (see the
  [net README](../net/README.md)). For the io_uring server transport see
  [http_uring](http_uring.md).
