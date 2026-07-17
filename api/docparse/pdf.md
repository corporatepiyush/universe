# universe_docparse_pdf

## Purpose

PDF 2.0 (ISO 32000-2) reader: classic xref/trailer, indirect objects,
FlateDecode content streams, and `(...)Tj` / `[...]TJ` text extraction. The whole
PDF sits in the caller buffer; `open()` locates `startxref` from the tail, reads
the cross-reference TABLE, and builds a flat object→byte-offset array (the only
allocation besides the caller's stream output). `stream()` finds an object's
`stream`..`endstream` span, detects a `/FlateDecode` filter, and inflates the
zlib payload via `universe_compress_inflate_zlib`. `extract_text()` walks a
decoded content stream and pulls the bytes of every literal `(...)` string
(operands of Tj/TJ), decoding PDF string escapes and honoring nested parens.
Cross-reference STREAMS and hex `<..>` strings are documented limitations.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_pdf_open` | `int32_t universe_docparse_pdf_open(const void *buf, int64_t len, void *doc)` | Parse xref into a 40-byte doc struct. | 0 OK / 1 NULL_PTR / 2 OOM / 13 (bad/unsupported xref) |
| `universe_docparse_pdf_close` | `void universe_docparse_pdf_close(void *doc)` | Free the object-offset array. | — |
| `universe_docparse_pdf_object_count` | `int64_t universe_docparse_pdf_object_count(void *doc)` | Number of objects. | count |
| `universe_docparse_pdf_object_offset` | `int32_t universe_docparse_pdf_object_offset(void *doc, int64_t num, int64_t *out)` | Byte offset of object `num`. | 0 OK / 5 NOT_FOUND / 7 INVALID_INDEX |
| `universe_docparse_pdf_stream` | `int64_t universe_docparse_pdf_stream(void *doc, int64_t objnum, void *dst, int64_t cap)` | Inflate/copy an object's stream into `dst`. | bytes written, or negative error |
| `universe_docparse_pdf_extract_text` | `int64_t universe_docparse_pdf_extract_text(const void *src, int64_t srclen, void *dst, int64_t cap)` | Pull Tj/TJ literal text from a decoded stream. | bytes written, or negative error |

**Document struct (40 B, caller-allocated):** `buf@0`, `len@8`, `xref_off@16`,
`obj_ptr@24`, `obj_count@32`.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_docparse_pdf_open(ptr, i64, ptr)
declare i64 @universe_docparse_pdf_stream(ptr, i64, ptr, i64)
declare i64 @universe_docparse_pdf_extract_text(ptr, i64, ptr, i64)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Decode object N's content stream, then print the text it shows.

```c
// pdftext.c — extract text from object N of a PDF
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int32_t universe_docparse_pdf_open(const void*, int64_t, void*);
int64_t universe_docparse_pdf_stream(void*, int64_t, void*, int64_t);
int64_t universe_docparse_pdf_extract_text(const void*, int64_t, void*, int64_t);
void    universe_docparse_pdf_close(void*);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s file.pdf objnum\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long z = ftell(f); rewind(f);
    unsigned char *buf = malloc(z); fread(buf, 1, z, f); fclose(f);
    unsigned char doc[40];
    if (universe_docparse_pdf_open(buf, z, doc) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    int64_t cap = 1 << 20;
    char *raw = malloc(cap), *txt = malloc(cap);
    int64_t rn = universe_docparse_pdf_stream(doc, atoll(argv[2]), raw, cap);
    if (rn > 0) {
        int64_t tn = universe_docparse_pdf_extract_text(raw, rn, txt, cap);
        if (tn > 0) fwrite(txt, 1, tn, stdout);
    }
    universe_docparse_pdf_close(doc); free(buf); free(raw); free(txt);
    return 0;
}
```

```
clang -O3 pdftext.c build/libuniverse.a -lpthread -lm -o pdftext
./pdftext doc.pdf 4
```

## Notes

- `doc` is a caller-allocated 40-byte struct; `open` mallocs the object array,
  `close` frees it — always pair them. `dst` buffers for `stream`/`extract_text`
  are caller-owned; a too-small `cap` yields a negative error.
- Only classic xref TABLES are supported (not xref streams); hex `<..>` strings
  are not decoded. Untrusted-input safe (bounds-checked).
- Reentrant; `extract_text` is stateless.
