# ioring/uring — Linux io_uring async IO backend

## Purpose

A from-scratch io_uring async-IO backend that speaks the raw Linux io_uring
kernel ABI directly (`io_uring_setup`=425 / `io_uring_enter`=426, plus
`mmap`/`munmap`/`close`) — NO dependency on any userspace io_uring helper
library. `ring_setup` maps the SQ ring, CQ ring and SQE array into the process
and returns a single-allocation handle carrying every base pointer and the
head/tail/mask pointers the ring protocol needs. The `prep_*` calls fill one SQE
(read/write/readv/writev/recv/send/accept/connect/close) tagged with a 64-bit
user-data cookie; `submit`/`submit_and_wait` issue one `io_uring_enter` per turn
(batched, never one syscall per SQE); `peek_cqe`/`wait_cqe`/`cqe_seen` drain
completions. **LINUX ONLY at runtime** (kernel ≥ 5.1, tested ≥ 6.15). The module
COMPILES everywhere (pure IR, no target lines); on non-Linux
`universe_ioring_ring_available` returns false so callers fall back to a posix
path.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ioring_ring_setup(int64_t entries, int64_t flags)` | Set up + mmap a ring with `entries` SQEs | ring handle, or NULL on failure/non-Linux |
| `bool universe_ioring_ring_available(void)` | Probe whether io_uring is usable on this host | true only on a supporting Linux kernel |
| `void universe_ioring_ring_destroy(void *ring)` | Unmap the rings and free the handle | — |
| `int32_t universe_ioring_prep_read(void *ring, int32_t fd, void *buf, int32_t len, int64_t off, int64_t ud)` | Queue a `read` SQE | 0 OK, 6 FULL (SQ full) |
| `int32_t universe_ioring_prep_write(void *ring, int32_t fd, const void *buf, int32_t len, int64_t off, int64_t ud)` | Queue a `write` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_readv(void *ring, int32_t fd, const void *iov, int32_t nr, int64_t off, int64_t ud)` | Queue a `readv` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_writev(void *ring, int32_t fd, const void *iov, int32_t nr, int64_t off, int64_t ud)` | Queue a `writev` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_recv(void *ring, int32_t fd, void *buf, int32_t len, int32_t msg_flags, int64_t ud)` | Queue a `recv` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_send(void *ring, int32_t fd, const void *buf, int32_t len, int32_t msg_flags, int64_t ud)` | Queue a `send` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_accept(void *ring, int32_t fd, void *addr, void *addrlen, int32_t flags, int64_t ud)` | Queue an `accept` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_connect(void *ring, int32_t fd, const void *addr, int64_t addrlen, int64_t ud)` | Queue a `connect` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_prep_close(void *ring, int32_t fd, int64_t ud)` | Queue a `close` SQE | 0 OK, 6 FULL |
| `int32_t universe_ioring_submit(void *ring)` | `io_uring_enter` submitting all queued SQEs | count submitted, or `<0` |
| `int32_t universe_ioring_submit_and_wait(void *ring, int64_t wait_nr)` | Submit and block for `wait_nr` completions | count submitted, or `<0` |
| `int32_t universe_ioring_peek_cqe(void *ring, void *out)` | Non-blocking peek of the next CQE into `out` | 1 if a CQE was copied, 0 if none |
| `int32_t universe_ioring_wait_cqe(void *ring, void *out)` | Block for the next CQE into `out` | 1 on CQE, `<0` on error |
| `int32_t universe_ioring_cqe_seen(void *ring)` | Advance the CQ head (consume one CQE) | 0 OK |

`ud` is an arbitrary 64-bit cookie echoed back in the CQE `user_data` — use it to
route a completion to its request. A CQE is a 16-byte struct: `user_data`@0,
`res`(i32)@8, `flags`(u32)@12; `res` < 0 is `-errno`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare ptr @universe_ioring_ring_setup(i64, i64)
declare i1  @universe_ioring_ring_available()
declare i32 @universe_ioring_prep_recv(ptr, i32, ptr, i32, i32, i64)
declare i32 @universe_ioring_submit_and_wait(ptr, i64)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Async `read` of a file's first 4 KiB via a one-SQE ring (Linux only).

```c
// uring_read.c   -- run on Linux >= 5.1
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
typedef struct { uint64_t ud; int32_t res; uint32_t flags; } cqe_t;
void   *universe_ioring_ring_setup(int64_t, int64_t);
int      universe_ioring_ring_available(void);
int32_t  universe_ioring_prep_read(void*, int32_t, void*, int32_t, int64_t, int64_t);
int32_t  universe_ioring_submit_and_wait(void*, int64_t);
int32_t  universe_ioring_wait_cqe(void*, void*);
int32_t  universe_ioring_cqe_seen(void*);
void     universe_ioring_ring_destroy(void*);

int main(int argc, char **argv) {
    if (argc < 2 || !universe_ioring_ring_available()) return 1;
    int fd = open(argv[1], O_RDONLY);
    void *ring = universe_ioring_ring_setup(8, 0);
    static char buf[4096];
    universe_ioring_prep_read(ring, fd, buf, sizeof buf, 0, 0x1234);
    universe_ioring_submit_and_wait(ring, 1);
    cqe_t c;
    if (universe_ioring_wait_cqe(ring, &c) == 1 && c.res > 0)
        fwrite(buf, 1, c.res, stdout);
    universe_ioring_cqe_seen(ring);
    universe_ioring_ring_destroy(ring);
    return 0;
}
```

```
clang -O3 uring_read.c build/libuniverse.a -lpthread -lm -o uring_read
./uring_read /etc/hostname     # Linux only
```

## Notes

- **Platform:** runtime-Linux-only. Always gate use on `ring_available()`;
  compiles but degrades on macOS/FreeBSD.
- **Kernel ABI:** struct offsets and syscall numbers are the public
  `uapi/linux/io_uring.h` contract for LP64 little-endian (x86_64/aarch64),
  implemented from first principles — no liburing.
- **Batching:** one `io_uring_enter` per turn; queue many `prep_*` then one
  `submit`. Never submit per SQE.
- **Ownership:** the ring handle is one allocation; `ring_destroy` unmaps the
  rings. The fds you pass are yours to close (or queue a `prep_close`).
- **Threading:** single-threaded SQ/CQ handle; not shared across threads.
