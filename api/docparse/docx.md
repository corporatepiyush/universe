# universe_docparse_docx

## Purpose

DOCX (Office Open XML / WordprocessingML) visible-text extraction. A `.docx` is a
ZIP in the caller buffer; the body lives in `word/document.xml`, which this module
locates and inflates ONCE into a private malloc buffer (the only allocation),
then walks with a single linear pass over the XML pull-token stream — no DOM.
This is the canonical IO/compute split: zip+inflate is the IO/decompress phase,
the token walk is pure compute writing into the caller's out buffer. Text is
emitted only inside `<w:t>` runs; `<w:tab/>` → `\t`, `<w:br/>`/`<w:cr/>` → `\n`,
`</w:p>` → `\n`. Writes are capacity-checked (returns FULL the instant a write
would exceed `out_cap`).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_docx_text` | `int32_t universe_docparse_docx_text(const void *zip_buf, int64_t zip_len, void *out, int64_t out_cap, int64_t *out_len)` | Extract visible text into `out`; write the byte count to `out_len`. | 0 OK / 1 NULL_PTR / 2 OOM / 5 NOT_FOUND (no `word/document.xml`) / 6 FULL |

`zip_buf`/`zip_len` is the whole `.docx` file in memory; `out`/`out_cap` is a
caller output buffer; the decoded byte count lands in `*out_len`.

## Use in an LLVM-based environment

```llvm
declare i32 @universe_docparse_docx_text(ptr, i64, ptr, i64, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Extract the text of a `.docx` file to stdout.

```c
// docxtext.c — print the visible text of a .docx
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int32_t universe_docparse_docx_text(const void*, int64_t, void*, int64_t, int64_t*);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.docx\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long z = ftell(f); rewind(f);
    unsigned char *zip = malloc(z); fread(zip, 1, z, f); fclose(f);
    int64_t cap = 1 << 20, out_len = 0;
    char *out = malloc(cap);
    int rc = universe_docparse_docx_text(zip, z, out, cap, &out_len);
    if (rc == 0) fwrite(out, 1, out_len, stdout);
    else fprintf(stderr, "error %d\n", rc);
    free(zip); free(out);
    return rc;
}
```

```
clang -O3 docxtext.c build/libuniverse.a -lpthread -lm -o docxtext
./docxtext report.docx
```

## Notes

- One private malloc for the inflated `word/document.xml`; everything else writes
  into the caller's `out`. Size `out_cap` for the expected text — 6 (FULL) means
  grow it (decoded bytes never exceed source XML bytes, so `zip_len` is a safe
  upper bound).
- Untrusted-input safe: bounds-checked reads, capacity-checked writes.
- Reentrant; no shared state.
