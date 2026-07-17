# parse/json — zero-copy JSON pull/SAX tokenizer (RFC 8259)

## Purpose

A JSON pull/SAX tokenizer over a caller-owned byte buffer. The scanner never
copies input: every token reports `{type, byte-offset, byte-length}` as offsets
*into* the caller buffer. The only allocation is the one-shot scanner object
(header + a small container-type stack), created once and reused for the whole
document — no per-token allocation. `_next` returns exactly ONE meaningful token
per call, transparently consuming `:`, `,` and whitespace; nesting is tracked
iteratively in an explicit byte stack (no native recursion) bounded by a
configurable max-depth.

The whitespace skip and string char-class scan are SIMD-first: 128-bit
`<16 x i8>` classify loops (SSE2 `pcmpeqb` / NEON `cmeq.16b`, baseline) with a
`_scalar` oracle twin for the tail and cross-check.

**Token types** (i32 at `token+0`): 0 begin-object, 1 end-object, 2 begin-array,
3 end-array, 4 string, 5 number, 6 true, 7 false, 8 null, 9 key, 10 end.
`token+8` = i64 byte offset, `token+16` = i64 byte length. For string/key the
offset/length span the *content* between the quotes (still escaped — feed to
`universe_parse_json_unescape` to decode).

## Exported API

| C signature | Description | Return |
|---|---|---|
| `void* universe_parse_json_scanner_create(const void* buf, int64_t len, int32_t max_depth)` | Create a scanner over `buf[0..len)` with nesting bound `max_depth`. | handle / NULL |
| `void universe_parse_json_scanner_destroy(void* sc)` | Free scanner. | — |
| `void universe_parse_json_scanner_reset(void* sc)` | Rewind to the start of the buffer for re-scan. | — |
| `int64_t universe_parse_json_error_offset(const void* sc)` | Byte offset of the last parse error. | offset |
| `int32_t universe_parse_json_next(void* sc, void* tok)` | Write the next token (24 bytes) into `tok`. | 0 OK (type 10 = valid end), 13 PARSE, 8 INVALID_ARG |
| `int64_t universe_parse_json_unescape(void* dst, const void* src, int64_t len)` | Decode JSON string escapes from `src` content into `dst`. | bytes written, negative on malformed |
| `int32_t universe_parse_json_number_double(const void* src, int64_t len, double* out)` | Parse a JSON number token into `*out`. | 0 OK, 13 on malformed |
| `int64_t universe_parse_json_scan_ws(const void* buf, int64_t len, int64_t pos)` | SIMD: first non-whitespace index at/after `pos`. | index (or `len`) |
| `int64_t universe_parse_json_scan_ws_scalar(const void* buf, int64_t len, int64_t pos)` | Scalar oracle twin. | index (or `len`) |
| `int64_t universe_parse_json_scan_structural(const void* buf, int64_t len, int64_t pos)` | SIMD: first structural byte (`{}[]:,"`) at/after `pos`. | index (or `len`) |
| `int64_t universe_parse_json_scan_structural_scalar(const void* buf, int64_t len, int64_t pos)` | Scalar oracle twin. | index (or `len`) |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`.

```c
#include <stdint.h>
extern void* universe_parse_json_scanner_create(const void*, int64_t, int32_t);
extern int32_t universe_parse_json_next(void*, void*);
extern void universe_parse_json_scanner_destroy(void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A token dumper: read a JSON document from stdin and print each token's type and
text span.

```c
// json_cli.c
#include <stdint.h>
#include <stdio.h>
extern void* universe_parse_json_scanner_create(const void*, int64_t, int32_t);
extern int32_t universe_parse_json_next(void*, void*);
extern int64_t universe_parse_json_error_offset(const void*);
extern void universe_parse_json_scanner_destroy(void*);

static const char* NAME[] = {"obj{","}obj","arr[","]arr","str","num",
                             "true","false","null","key","end"};
int main(void){
    static char buf[1<<20];
    int64_t n = fread(buf, 1, sizeof buf, stdin);
    void* sc = universe_parse_json_scanner_create(buf, n, 64);
    struct { int32_t type; int32_t _p; int64_t off, len; } t;
    for (;;) {
        int rc = universe_parse_json_next(sc, &t);
        if (rc != 0) { printf("PARSE error @%lld\n",
            (long long)universe_parse_json_error_offset(sc)); break; }
        if (t.type == 10) { printf("end\n"); break; }
        printf("%-6s %.*s\n", NAME[t.type], (int)t.len, buf + t.off);
    }
    universe_parse_json_scanner_destroy(sc);
    return 0;
}
```

```
clang -O3 json_cli.c build/libuniverse.a -lpthread -lm -o json_cli
printf '{"a":1,"b":[true,null]}' | ./json_cli
```

## Notes

- **Zero-copy.** Token spans reference the caller buffer, which must outlive
  the scan. String/key spans are still escaped — decode with
  `universe_parse_json_unescape` into scratch.
- **Ownership / threading.** One scanner allocation, created once, reused (or
  `reset`) for the document; one scanner per thread.
- **Recursion-free.** Nesting uses an explicit byte stack bounded by
  `max_depth`, so adversarial deep input cannot blow the native stack.
- **SIMD.** `scan_ws` / `scan_structural` are the vector primitives with
  `_scalar` fallbacks and test oracles.
