# search/ahocorasick — Aho-Corasick multi-pattern matcher

## Purpose

Aho-Corasick multi-pattern matcher: find every occurrence of a *set* of byte
patterns in a text in one linear pass. The automaton is a **full-DFA goto
table** — one flat `int32 next[num_nodes*256]` where every `(state,byte)` is
already resolved through the fail links, so the search inner loop is a single
table load per byte (`state = next[state*256 + byte]`) with no fail-follow at
scan time. This trades 1 KiB/node for a branch-free, call-free hot leaf — the
right call for scanning megabytes. Fail links, the full DFA, and dictionary-
suffix links (for overlapping/suffix matches) are built in one BFS pass. The
whole automaton is one allocation.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `void* universe_search_ac_build(const char** patterns, const int64_t* lens, int64_t n)` | Build the automaton from `n` patterns (`patterns[i]` byte string, `lens[i]` its length ≥ 1). | handle / NULL on OOM |
| `int64_t universe_search_ac_search(void* ac, const void* text, int64_t len, int32_t* out_ids, int64_t* out_ends, int64_t cap)` | Scan `text[0..len)`. For each match writes `out_ids[k]` = pattern id, `out_ends[k]` = end offset (up to `cap` entries). `out_ids`/`out_ends` may be NULL to count only. | total match count (may exceed `cap`); -1 if `ac` is NULL |
| `void universe_search_ac_destroy(void* ac)` | Free the automaton. | — |

The match count returned may exceed `cap` — only the first `cap` matches are
written; a larger count signals the caller to enlarge the buffers and rescan.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern void* universe_search_ac_build(const char**, const int64_t*, int64_t);
extern int64_t universe_search_ac_search(void*, const void*, int64_t,
                                         int32_t*, int64_t*, int64_t);
extern void universe_search_ac_destroy(void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A multi-pattern grep: patterns from argv, text from stdin; print each match as
`end_offset:pattern_index`.

```c
// acgrep.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void* universe_search_ac_build(const char**, const int64_t*, int64_t);
extern int64_t universe_search_ac_search(void*, const void*, int64_t,
                                         int32_t*, int64_t*, int64_t);
extern void universe_search_ac_destroy(void*);

int main(int argc, char** argv){
    int n = argc - 1;
    if (n < 1) { fprintf(stderr, "usage: acgrep pat...\n"); return 2; }
    const char** pats = malloc(n * sizeof *pats);
    int64_t* lens = malloc(n * sizeof *lens);
    for (int i = 0; i < n; i++) { pats[i] = argv[i+1]; lens[i] = strlen(argv[i+1]); }
    void* ac = universe_search_ac_build(pats, lens, n);

    static char text[1<<20];
    int64_t tn = fread(text, 1, sizeof text, stdin);
    enum { CAP = 4096 };
    int32_t ids[CAP]; int64_t ends[CAP];
    int64_t total = universe_search_ac_search(ac, text, tn, ids, ends, CAP);
    int64_t k = total < CAP ? total : CAP;
    for (int64_t i = 0; i < k; i++)
        printf("%lld:%d %s\n", (long long)ends[i], ids[i], pats[ids[i]]);
    printf("total=%lld\n", (long long)total);
    universe_search_ac_destroy(ac);
    return 0;
}
```

```
clang -O3 acgrep.c build/libuniverse.a -lpthread -lm -o acgrep
printf 'she sells seashells' | ./acgrep he she sea sells
```

## Notes

- **Memory.** Full-DFA is 1 KiB per node (`int32[256]`); prefer for scan speed
  over memory. One allocation for the whole automaton.
- **Overlaps.** Dictionary-suffix links report all overlapping and suffix
  matches, including duplicate patterns and a pattern equal to another's
  suffix.
- **Threading.** The built automaton is immutable — safe to share read-only
  across threads; each searcher supplies its own text and output buffers.
- **Ownership.** Pattern strings are borrowed at build time; the caller owns
  `text` and the `out_ids`/`out_ends` buffers.
