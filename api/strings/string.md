# strings/string — string view + SSO owned string (UTF-8)

## Purpose

Two UTF-8 string representations, chosen by ownership. Bytes are opaque, so
multibyte sequences are handled transparently.

1. **String view** — an immutable `{ptr data, int64 len}` passed by value as two
   args. Pure, allocation-free slices: no ownership, no copy. Use for parsing,
   scanning, comparison, hashing, and slicing over a buffer you already own (a
   `universe_bytes` block, an mmap'd file). `substring_view` returns an adjusted
   `{ptr,len}` with zero allocation (pointer add + length clamp). The hot
   data-parallel scans (equal / compare / index-of) delegate to the verified
   128-bit `simd/scan` kernels.
2. **SSO owned string** — a 24-byte value that OWNS its bytes. Strings of length
   ≤ 22 live inline inside the 24 bytes (no heap touch); longer strings spill to
   the heap. `sso_data` resolves inline-vs-heap with a single `select`.

## Exported API

### String view (borrowed, allocation-free)

| C signature | Description | Return |
|---|---|---|
| `int64_t universe_string_len(const void* data, int64_t len)` | Byte length accessor for a view (returns `len`). | length |
| `int universe_string_eq(const void* a, int64_t alen, const void* b, int64_t blen)` | Byte-equality of two views. | bool |
| `int32_t universe_string_compare(const void* a, int64_t alen, const void* b, int64_t blen)` | Lexicographic unsigned compare. | <0 / 0 / >0 |
| `int64_t universe_string_index_of_byte(const void* data, int64_t len, char byte)` | First index of `byte`. | index, or -1 |
| `int64_t universe_string_count_byte(const void* data, int64_t len, char byte)` | Count of `byte`. | count |
| `int64_t universe_string_hash(const void* data, int64_t len)` | Deterministic hash (same bytes → same value on every platform). | hash |
| `int universe_string_starts_with(const void* data, int64_t len, const void* pre, int64_t plen)` | Does the view start with `pre[0..plen)`? | bool |
| `{void*, int64_t} universe_string_substring_view(const void* data, int64_t len, int64_t start, int64_t count)` | Sub-view `[start, start+count)` (clamped), zero allocation. | `{ptr, len}` |

### SSO owned string

| C signature | Description | Return |
|---|---|---|
| `int32_t universe_sso_create(void* out, const void* data, int64_t len)` | Fill 24-byte `out` with an owned copy of `data[0..len)` (inline if `len ≤ 22`). | 0 OK, 1 NULL, 2 OOM |
| `int64_t universe_sso_len(const void* s)` | Length of the owned string. | length |
| `void* universe_sso_data(const void* s)` | Pointer to the bytes (inline or heap). | ptr |
| `void universe_sso_free(void* s)` | Free the heap block (no-op for inline strings). | — |

`substring_view` returns the two-field aggregate `{ptr, i64}` by value.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. The SSO value is 24 bytes of caller
storage (e.g. `_Alignas(8) char s[24]`).

```c
#include <stdint.h>
extern int64_t universe_string_hash(const void*, int64_t);
extern int32_t universe_sso_create(void*, const void*, int64_t);
extern void*   universe_sso_data(const void*);
extern int64_t universe_sso_len(const void*);
extern void    universe_sso_free(void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read lines from stdin; for each, print its hash and whether it starts with a
prefix given on argv.

```c
// strtool.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern int64_t universe_string_hash(const void*, int64_t);
extern int universe_string_starts_with(const void*, int64_t, const void*, int64_t);
int main(int argc, char** argv){
    const char* pre = argc > 1 ? argv[1] : "";
    int64_t plen = strlen(pre);
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        int64_t n = strlen(line);
        if (n && line[n-1] == '\n') n--;
        printf("hash=%016llx starts=%d  %.*s\n",
               (unsigned long long)universe_string_hash(line, n),
               universe_string_starts_with(line, n, pre, plen),
               (int)n, line);
    }
    return 0;
}
```

```
clang -O3 strtool.c build/libuniverse.a -lpthread -lm -o strtool
printf 'hello\nworld\n' | ./strtool he
```

## Notes

- **View = borrowed, zero-alloc.** The view functions never allocate; the caller
  owns the bytes, which must outlive the calls. `substring_view` returns a
  sub-slice of the same buffer.
- **SSO = owned.** `sso_create` copies (inline ≤ 22 bytes, else one heap block);
  call `sso_free` exactly once for heap-backed strings (harmless for inline).
- **Threading.** Views are pure and reentrant. Hot scans delegate to
  `simd/scan` (SIMD-first with scalar fallback).
