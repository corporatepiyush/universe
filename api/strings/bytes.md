# strings/bytes — growable byte buffer

## Purpose

A growable byte buffer — the foundation for strings, encoders, and IO. Amortized
O(1) append via geometric (2×) growth, with a zero-copy contiguous data view. By
necessity it is **two allocations**: a small stable 24-byte header `{ i64 len,
i64 cap, ptr data }` (so callers can hold the handle across growth) plus a
separately realloc'd data block — only `data` moves on growth. The three hot
fields sit together in one cache line, so `len()`/`cap()`/`data()` and the append
fast path touch one line. Growth is a single `realloc` (`new_cap = max(needed,
cap*2)`); append is one `llvm.memcpy` after a single reserve, never a byte loop.

## Exported API

Error convention: 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX.

| C signature | Description | Return |
|---|---|---|
| `void* universe_bytes_create(int64_t initial_cap)` | Create a buffer with `initial_cap` reserved bytes. | handle / NULL on OOM |
| `int64_t universe_bytes_len(void* b)` | Current length. | length |
| `int64_t universe_bytes_cap(void* b)` | Current capacity. | capacity |
| `void* universe_bytes_data(void* b)` | Pointer to the contiguous bytes (moves on growth). | ptr |
| `int32_t universe_bytes_reserve(void* b, int64_t extra)` | Ensure room for `extra` more bytes. | 0 / 1 / 2 / 3 |
| `int32_t universe_bytes_append(void* b, const void* src, int64_t n)` | Append `src[0..n)` (grows, one memcpy). | 0 / 1 / 2 / 3 |
| `int32_t universe_bytes_append_byte(void* b, char v)` | Append one byte. | 0 / 1 / 2 / 3 |
| `void universe_bytes_clear(void* b)` | Reset length to 0 (keeps capacity). | — |
| `int32_t universe_bytes_truncate(void* b, int64_t newlen)` | Shrink to `newlen` (`≤ len`, else 7). | 0 / 1 / 7 |
| `void universe_bytes_destroy(void* b)` | Free header + data. | — |
| `int64_t universe_bytes_index_of_byte(void* b, char c)` | First index of `c`. | index, or -1 |
| `int64_t universe_bytes_count_byte(void* b, char c)` | Count of `c`. | count |
| `int universe_bytes_equals(void* a, void* b)` | Byte-equality of two buffers. | bool |
| `int32_t universe_bytes_compare(void* a, void* b)` | Lexicographic unsigned compare (NULL handle sorts low). | <0 / 0 / >0 |
| `void universe_bytes_to_lower(void* b)` | ASCII lowercase in place. | — |
| `void universe_bytes_to_upper(void* b)` | ASCII uppercase in place. | — |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern void* universe_bytes_create(int64_t);
extern int32_t universe_bytes_append(void*, const void*, int64_t);
extern void* universe_bytes_data(void*);
extern int64_t universe_bytes_len(void*);
extern void universe_bytes_destroy(void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Slurp stdin into a buffer, uppercase it, and write it back out.

```c
// upper.c
#include <stdint.h>
#include <stdio.h>
extern void* universe_bytes_create(int64_t);
extern int32_t universe_bytes_append(void*, const void*, int64_t);
extern void universe_bytes_to_upper(void*);
extern void* universe_bytes_data(void*);
extern int64_t universe_bytes_len(void*);
extern void universe_bytes_destroy(void*);

int main(void){
    void* b = universe_bytes_create(4096);
    char chunk[65536]; size_t r;
    while ((r = fread(chunk, 1, sizeof chunk, stdin)) > 0)
        universe_bytes_append(b, chunk, (int64_t)r);
    universe_bytes_to_upper(b);
    fwrite(universe_bytes_data(b), 1, universe_bytes_len(b), stdout);
    universe_bytes_destroy(b);
    return 0;
}
```

```
clang -O3 upper.c build/libuniverse.a -lpthread -lm -o upper
printf 'Hello, World\n' | ./upper
```

## Notes

- **Two allocations, stable handle.** The handle never moves; `data()` moves on
  growth, so re-fetch it after any append/reserve. Overflow-checked size math
  (`SIZE_OVERFLOW`).
- **Ownership.** `destroy` frees both the header and the data block; call once.
- **Threading.** Not internally synchronized — one buffer per thread, or guard
  externally. Case-fold and scan ops touch ASCII only, leaving UTF-8
  continuation bytes untouched.
