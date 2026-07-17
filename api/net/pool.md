# net/pool — single-threaded TCP connection pool

## Purpose

A single-threaded TCP connection pool: caches up to `cap` idle connections keyed
by `(ip, port)`, reuses live ones, and evicts the least-recently-used when full.
ONE allocation holds a header plus a struct-of-arrays; occupied slots form an
intrusive doubly-linked recency list (head = MRU, tail = LRU) using i32 slot
indices (−1 sentinel), not pointers — half the footprint, no pointer chasing.
Free slots come from a wilderness bump cursor first, then a freelist, so `create`
is O(1) with no pre-linking. A pooled entry is a keyed multiset (several idle
conns per peer), so lookup is a short walk of the recency list. `pool_get`
validates liveness with a non-blocking `MSG_PEEK` recv — a peer that closed
returns EOF and the corpse is closed and skipped. Concurrency is DEFERRED: NO
locks, NO atomics.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_net_pool_create(int64_t cap)` | Allocate a pool holding up to `cap` idle conns | pool handle, or NULL on OOM |
| `void universe_net_pool_destroy(void *p)` | Close all cached fds and free the pool | — |
| `int64_t universe_net_pool_count(void *p)` | Number of currently cached idle connections | count (≥0) |
| `int32_t universe_net_pool_get(void *p, int32_t ip, int32_t port)` | Take a live cached fd for `ip:port` (validated via MSG_PEEK) | fd (≥0), or `<0` if none cached |
| `int32_t universe_net_pool_put(void *p, int32_t fd, int32_t ip, int32_t port)` | Return an idle `fd` to the pool for reuse | 0 OK; may evict/close the LRU when full |

`ip`/`port` are host-order `int32` matching `net/tcp`. `pool_get` returning `<0`
means no live idle connection is cached — open a fresh one with
`universe_net_tcp_connect`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare ptr @universe_net_pool_create(i64)
declare i32 @universe_net_pool_get(ptr, i32, i32)
declare i32 @universe_net_pool_put(ptr, i32, i32, i32)
declare void @universe_net_pool_destroy(ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Reuse a pooled connection across two requests to the same peer.

```c
// pooldemo.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void   *universe_net_pool_create(int64_t);
void    universe_net_pool_destroy(void *);
int64_t universe_net_pool_count(void *);
int32_t universe_net_pool_get(void *, int32_t, int32_t);
int32_t universe_net_pool_put(void *, int32_t, int32_t, int32_t);
int32_t universe_net_tcp_connect(int32_t, int32_t);
int64_t universe_net_tcp_send_all(int32_t, const void *, int64_t);

int main(void) {
    int32_t ip = 0x7F000001, port = 8080;   // 127.0.0.1:8080
    void *pool = universe_net_pool_create(16);
    for (int i = 0; i < 2; i++) {
        int fd = universe_net_pool_get(pool, ip, port);
        if (fd < 0) { fd = universe_net_tcp_connect(ip, port); }
        if (fd < 0) { fprintf(stderr, "connect failed\n"); break; }
        const char *ping = "GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n";
        universe_net_tcp_send_all(fd, ping, strlen(ping));
        universe_net_pool_put(pool, fd, ip, port);   // return for reuse
        printf("cached=%lld\n", (long long)universe_net_pool_count(pool));
    }
    universe_net_pool_destroy(pool);
    return 0;
}
```

```
clang -O3 pooldemo.c build/libuniverse.a -lpthread -lm -o pooldemo
./pooldemo
```

## Notes

- **Layout:** one allocation, index-linked SoA recency list (i32 indices, −1
  sentinel); LRU eviction closes the evicted fd.
- **Liveness:** `pool_get` MSG_PEEKs before handing back a fd; dead peers are
  closed and skipped — you always get a live fd or `<0`.
- **Ownership:** `pool_put` transfers the fd to the pool; `pool_destroy` closes
  every cached fd. Do not close a fd you have returned to the pool.
- **Threading:** single-threaded; not safe to share across threads.
