# io/bufio — buffered reader + writer over raw fds

## Purpose

Buffered reader and writer over raw file descriptors (regular files, pipes,
sockets). Each side owns ONE allocation: a header (≤1 cache line) at +0 and the
data buffer at +64, so the hot cursors never share a line with buffered bytes.
The reader keeps an unconsumed window `[rpos, wpos)` and pulls one `read(2)`
when it drains; reads `≥ cap` bypass the buffer and go straight into the
caller's `dst`. `peek`/`read_until`/`read_line` hand back ZERO-COPY slices into
the buffer (compacting when a token straddles the window), valid only until the
next call that may refill. The writer accumulates small pieces and flushes with
one `write(2)` when full; writes `≥ cap` bypass the buffer. This is the SDK's
"fill buffer → compute over buffer → flush" substrate — never one syscall per
byte.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_io_scan_byte(const void *base, int64_t len, uint8_t delim)` | SIMD scan for first `delim` in `[base,base+len)` | index, or `-1` if absent |
| `void *universe_io_reader_create(int32_t fd, int64_t bufsize)` | Allocate a buffered reader over `fd` | reader handle, or NULL on OOM |
| `void universe_io_reader_destroy(void *r)` | Free the reader (does not close `fd`) | — |
| `int64_t universe_io_reader_fill(void *r)` | Pull one `read(2)` when the window is empty | bytes available, or `<0` error |
| `int64_t universe_io_reader_read(void *r, void *dst, int64_t n)` | Copy up to `n` bytes into `dst` | bytes read (0 = EOF), or `<0` error |
| `int32_t universe_io_reader_read_exact(void *r, void *dst, int64_t n)` | Fill exactly `n` bytes | 0 OK, 4 EMPTY (EOF short), `<0` |
| `int32_t universe_io_reader_read_byte(void *r)` | Read one byte | byte 0–255, or `<0` on EOF/error |
| `void *universe_io_reader_peek(void *r, int64_t n, int64_t *out_avail)` | View up to `n` buffered bytes without consuming | slice ptr; `*out_avail` set |
| `int32_t universe_io_reader_read_until(void *r, uint8_t delim, void **out_ptr, int64_t *out_len)` | Zero-copy slice through the next `delim` (inclusive) | 0 OK, 6 FULL (token > buffer), 4 EMPTY |
| `int32_t universe_io_reader_read_line(void *r, void **out_ptr, int64_t *out_len)` | `read_until('\n')` convenience | same as `read_until` |
| `void *universe_io_writer_create(int32_t fd, int64_t bufsize)` | Allocate a buffered writer over `fd` | writer handle, or NULL on OOM |
| `void universe_io_writer_destroy(void *w)` | Free the writer (does not close `fd`) | — |
| `int32_t universe_io_writer_flush(void *w)` | Drain the buffer with a partial-write loop | 0 OK, `<0` error |
| `int64_t universe_io_writer_write(void *w, const void *src, int64_t n)` | Buffer/emit `n` bytes | bytes accepted, or `<0` |
| `int32_t universe_io_writer_write_byte(void *w, uint8_t b)` | Append one byte | 0 OK, `<0` |
| `int32_t universe_io_writer_write_all(void *w, const void *src, int64_t n)` | Write all `n` bytes (loops over `write`) | 0 OK, `<0` |
| `int32_t universe_io_writer_flush_vectored(void *w, const void *iov, int32_t count)` | Assemble an `iovec[count]` and `writev` once | 0 OK, `<0` |

Error codes are the SDK negatives (e.g. IO). Zero-copy slices are invalidated by
the next reader call that may refill or compact.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the C-ABI symbols (`nounwind`)
and link:

```llvm
declare ptr @universe_io_reader_create(i32, i64)
declare i32 @universe_io_reader_read_byte(ptr)
declare void @universe_io_reader_destroy(ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

`cat`-style tool: buffered-copy stdin (fd 0) to stdout (fd 1).

```c
// bufcat.c
#include <stdint.h>
void *universe_io_reader_create(int32_t, int64_t);
void  universe_io_reader_destroy(void *);
int64_t universe_io_reader_read(void *, void *, int64_t);
void *universe_io_writer_create(int32_t, int64_t);
void  universe_io_writer_destroy(void *);
int32_t universe_io_writer_write_all(void *, const void *, int64_t);
int32_t universe_io_writer_flush(void *);

int main(void) {
    void *r = universe_io_reader_create(0, 1 << 16);
    void *w = universe_io_writer_create(1, 1 << 16);
    char buf[8192];
    int64_t n;
    while ((n = universe_io_reader_read(r, buf, sizeof buf)) > 0)
        universe_io_writer_write_all(w, buf, n);
    universe_io_writer_flush(w);
    universe_io_writer_destroy(w);
    universe_io_reader_destroy(r);
    return 0;
}
```

```
clang -O3 bufcat.c build/libuniverse.a -lpthread -lm -o bufcat
echo hello | ./bufcat
```

## Notes

- **Layout:** one allocation per side, header +0, buffer +64; no pointer webs.
- **Ownership:** `create` mallocs, `destroy` frees; neither opens nor closes the
  fd — the caller owns it.
- **Zero-copy lifetime:** `peek`/`read_until`/`read_line` slices are valid only
  until the next reader call that may refill/compact. Copy out if you need them
  to persist.
- **Threading:** single-threaded; a reader/writer is not shared across threads.
