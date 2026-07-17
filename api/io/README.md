# io — buffered IO

Buffered reader/writer primitives over raw file descriptors: the SDK's
"fill buffer → compute over buffer → flush" substrate that keeps syscalls out
of compute loops. All symbols are C-ABI `nounwind`; link against
`build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [bufio](bufio.md) | Buffered reader + writer over fds; zero-copy `peek`/`read_line`, vectored flush |

Single-threaded (concurrency deferred). Each reader/writer is one allocation
(header +0, buffer +64); neither opens nor closes the underlying fd.
