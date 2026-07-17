# universe_utf8

## Purpose

UTF-8 validation and codepoint counting over a caller byte range. Pure compute;
no allocation, no IO. Full RFC 3629 well-formedness (not just a continuation-bit
check): rejects overlong forms, surrogates, and out-of-range sequences, using the
"lead byte selects the second byte's legal range" technique so a handful of
`select`s compute `{len, lo2, hi2}` from the lead and the range checks are
uniform. The hot loop has an ASCII fast path (byte < 0x80 advances by 1) as a
straight fall-through; multi-byte classification is branchless.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_utf8_validate` | `int64_t universe_utf8_validate(const void *src, int64_t n)` | Check well-formedness. | index of first bad byte, or -1 if valid |
| `universe_utf8_count_codepoints` | `int64_t universe_utf8_count_codepoints(const void *src, int64_t n)` | Count codepoints. | count, or -1 if not well-formed |
| `universe_utf8_byte_len_of_codepoint` | `int32_t universe_utf8_byte_len_of_codepoint(int8_t lead)` | Sequence length of a lead byte. | 1/2/3/4, or 0 (continuation / invalid lead) |

For a truncated multibyte sequence at the end of the buffer, `validate` reports
the index of the LEAD byte that begins the incomplete sequence.

## Use in an LLVM-based environment

```llvm
declare i64 @universe_utf8_validate(ptr, i64)
declare i64 @universe_utf8_count_codepoints(ptr, i64)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Validate stdin and report codepoint count (or the first bad offset).

```c
// utf8cli.c — validate stdin, print codepoint count or first bad byte offset
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_utf8_validate(const void*, int64_t);
int64_t universe_utf8_count_codepoints(const void*, int64_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    int64_t bad = universe_utf8_validate(b, (int64_t)n);
    if (bad < 0) printf("valid, %lld codepoints\n",
                        (long long)universe_utf8_count_codepoints(b, (int64_t)n));
    else printf("invalid at byte %lld\n", (long long)bad);
    free(b);
    return bad < 0 ? 0 : 1;
}
```

```
clang -O3 utf8cli.c build/libuniverse.a -lpthread -lm -o utf8cli
printf 'caf\xc3\xa9' | ./utf8cli        # valid, 4 codepoints
```

## Notes

- Read-only over the caller buffer; no allocation, stateless, reentrant.
- `count_codepoints` re-validates for correctness; the boundary predicate
  `(b & 0xC0) != 0x80` vectorizes.
