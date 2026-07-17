# parse/csv — zero-copy CSV/TSV field iterator (RFC 4180)

## Purpose

A zero-copy, zero-alloc CSV/TSV field iterator over a caller-owned byte buffer.
Every field is reported as `{offset, length}` *into* the caller buffer; nothing
is copied. The scanner state is a tiny 32-byte struct the caller owns (a global,
an alloca, or a slab slot) — the module allocates nothing. One delimiter byte
parameterizes CSV (`,`) vs TSV (`\t`) vs anything else. RFC 4180 quoting is
handled: a doubled `""` inside a quoted field is a literal quote; delimiter, CR
and LF are literal inside quotes; LF, CRLF and bare CR all end a record.

The unquoted-field and quoted-content scans are SIMD-first: 128-bit `<16 x i8>`
classify loops are the primary path (SSE2 `pcmpeqb` / NEON `cmeq.16b`, baseline,
no runtime check); a scalar twin handles the sub-16 tail and is the cross-check
oracle.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `void universe_parse_csv_init(void* sc, const void* buf, int64_t len, int32_t delim)` | Initialize caller-owned 32-byte scanner `sc` over `buf[0..len)` with delimiter byte `delim`. | — |
| `int32_t universe_parse_csv_next_field(void* sc, void* out)` | Produce next field record into `out` `{off@0:i64, len@8:i64, needs_unquote@16:i32}`. | 0 field (record continues), 1 field + record ends, 2 EOF |
| `int32_t universe_parse_csv_next_record(void* sc, void* arr, int64_t cap, int64_t* count)` | Fill `arr` (array of field records) for ONE record; `*count` set to fields written. | 0 record, 2 EOF, 6 FULL (>cap), 8/13 error |
| `int64_t universe_parse_csv_unquote(void* dst, const void* src, int64_t len)` | Collapse `""`→`"` from a quoted field's content into `dst`. | bytes written |
| `int64_t universe_parse_csv_scan_field(const void* buf, int64_t len, int64_t pos, int32_t delim)` | SIMD classify primitive: index of next delim/CR/LF at/after `pos`. | index (or `len`) |
| `int64_t universe_parse_csv_scan_field_scalar(const void* buf, int64_t len, int64_t pos, int32_t delim)` | Scalar oracle twin of the above (bit-identical). | index (or `len`) |

Field record layout: `off` (i64) and `len` (i64) span the content — for a
quoted field they span the content *between* the outer quotes, and
`needs_unquote` is set when a `""` is present so the caller can call
`universe_parse_csv_unquote`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. The scanner struct is 32 bytes of
caller storage (e.g. a `char sc[32]`, 8-byte aligned).

```c
#include <stdint.h>
extern void universe_parse_csv_init(void*, const void*, int64_t, int32_t);
extern int32_t universe_parse_csv_next_field(void*, void*);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read a whole CSV file from stdin and print each field on its own line, marking
record boundaries.

```c
// csv_cli.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void universe_parse_csv_init(void*, const void*, int64_t, int32_t);
extern int32_t universe_parse_csv_next_field(void*, void*);

int main(int argc, char** argv){
    int delim = (argc > 1 && argv[1][0]=='t') ? '\t' : ',';
    static char buf[1<<20];
    int64_t n = fread(buf, 1, sizeof buf, stdin);
    _Alignas(8) char sc[32];
    universe_parse_csv_init(sc, buf, n, delim);
    struct { int64_t off, len; int32_t nq; } f;
    int rc, row = 0, col = 0;
    do {
        rc = universe_parse_csv_next_field(sc, &f);
        if (rc == 2) break;
        printf("[%d,%d] %.*s\n", row, col++, (int)f.len, buf + f.off);
        if (rc == 1) { row++; col = 0; }
    } while (1);
    return 0;
}
```

```
clang -O3 csv_cli.c build/libuniverse.a -lpthread -lm -o csv_cli
printf 'a,b,c\n1,"x,y",3\n' | ./csv_cli      # comma (default)
printf 'a\tb\n1\t2\n' | ./csv_cli t          # tab-separated
```

## Notes

- **Zero-copy / zero-alloc.** Fields are offsets into the caller buffer, which
  must outlive iteration. The module never allocates; the caller owns the
  32-byte scanner.
- **Quoting.** `needs_unquote` fields still contain the raw `""`; call
  `universe_parse_csv_unquote` into scratch to decode.
- **Threading.** Not shared — one scanner per thread/stream. Pure compute over
  memory the caller already filled (fill the buffer once with one big read,
  then iterate at cache speed).
- **SIMD.** `scan_field` is the vector primitive; `_scalar` is its fallback and
  test oracle.
