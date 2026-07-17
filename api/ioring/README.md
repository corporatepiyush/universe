# ioring — io_uring async IO

Raw Linux io_uring backend implemented directly against the kernel ABI (no
liburing). Submission/completion ring handle plus `prep_*` opcode builders for
read/write/recv/send/accept/connect/close. C-ABI `nounwind`; link against
`build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [uring](uring.md) | io_uring ring setup, SQE builders, submit/wait/peek completions |

**Runtime-Linux-only** (kernel ≥ 5.1, tested ≥ 6.15). The module compiles on all
targets; on non-Linux `universe_ioring_ring_available()` returns false so
callers can fall back to a posix path (e.g. `net/tcp`). Always gate use on that
probe.
