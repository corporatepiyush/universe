# search/suffix_array — suffix array + LCP with substring find/count

## Purpose

Suffix array + LCP array of a byte string, with substring find/count. The
suffix array is built by **prefix doubling with radix (counting) sort** —
O(N log N): each round sorts suffixes by the pair `(rank[i], rank[i+k])` with a
stable two-pass LSD counting sort over N+1 buckets, stopping early once all
ranks are distinct. Ranks in `[0,N)` are used directly as radix digits, so no
comparisons — the right class for suffix sorting. The LCP array is built by
Kasai's algorithm in O(N). `find`/`count` are binary search over the sorted
suffixes: O(M + log N) find, two lower-bound searches for the occurrence range
in count.

The persistent handle is one allocation: a 32-byte header plus `sa[N]` and
`lcp[N]` as flat `int32` arrays. The text buffer is **borrowed** — the caller
owns it and it must outlive the handle; `find`/`count` also take `text`
explicitly.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `void* universe_search_sa_build(const void* text, int64_t len)` | Build suffix array + LCP over `text[0..len)` (text borrowed). | handle / NULL on bad args or OOM |
| `int64_t universe_search_sa_find(void* sa, const void* text, const void* pattern, int64_t plen)` | Find some occurrence of `pattern` in `text`. | start offset, or -1 if absent |
| `int64_t universe_search_sa_count(void* sa, const void* text, const void* pattern, int64_t plen)` | Count occurrences of `pattern`. | number of matches |
| `int64_t universe_search_sa_at(void* sa, int64_t i)` | `sa[i]` (i-th smallest suffix start). | value, or -1 if out of bounds |
| `int64_t universe_search_sa_lcp_at(void* sa, int64_t i)` | `lcp[i]`. | value, or -1 if out of bounds |
| `int64_t universe_search_sa_len(void* sa)` | N (text length). | length |
| `void universe_search_sa_destroy(void* sa)` | Free the handle. | — |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern void* universe_search_sa_build(const void*, int64_t);
extern int64_t universe_search_sa_count(void*, const void*, const void*, int64_t);
extern void universe_search_sa_destroy(void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Count and locate a pattern in text from stdin.

```c
// sagrep.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void* universe_search_sa_build(const void*, int64_t);
extern int64_t universe_search_sa_find(void*, const void*, const void*, int64_t);
extern int64_t universe_search_sa_count(void*, const void*, const void*, int64_t);
extern void universe_search_sa_destroy(void*);

int main(int argc, char** argv){
    if (argc < 2) { fprintf(stderr, "usage: sagrep pattern < text\n"); return 2; }
    const char* pat = argv[1];
    int64_t plen = strlen(pat);
    static char text[1<<20];
    int64_t n = fread(text, 1, sizeof text, stdin);
    void* sa = universe_search_sa_build(text, n);
    printf("count=%lld first=%lld\n",
           (long long)universe_search_sa_count(sa, text, pat, plen),
           (long long)universe_search_sa_find(sa, text, pat, plen));
    universe_search_sa_destroy(sa);
    return 0;
}
```

```
clang -O3 sagrep.c build/libuniverse.a -lpthread -lm -o sagrep
printf 'abracadabra' | ./sagrep abra
```

## Notes

- **Ownership.** The text buffer is borrowed and must outlive the handle; pass
  the same `text` pointer to `find`/`count`. One allocation for the handle
  (header + `sa` + `lcp`).
- **Complexity.** Build O(N log N) time; `find` O(M + log N); `count` two
  lower-bound searches; `at`/`lcp_at`/`len` O(1).
- **Threading.** The built handle is immutable — safe to share read-only for
  concurrent `find`/`count`.
