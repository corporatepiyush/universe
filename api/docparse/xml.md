# universe_docparse_xml

## Purpose

A minimal zero-copy, zero-alloc XML pull-tokenizer sufficient for OOXML
(SpreadsheetML / WordprocessingML). The whole document sits in a caller buffer;
every token reports `{type, name/text slice, attribute-region slice}` as byte
OFFSETS into that buffer — never strdup'd. The only caller state is a 24-byte
scanner `{buf,len,pos}`; `_next` advances `pos` and returns exactly one
meaningful token per call. It is NOT a validating parser: it recognizes start /
end / self-closing tags, text, and CDATA, and transparently skips comments, PIs
and DOCTYPE. `_attr_next` is a second cursor over a tag's attribute region;
`_decode` resolves XML entities into UTF-8.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_xml_init` | `void universe_docparse_xml_init(void *sc, const void *buf, int64_t len)` | Init a 24-byte scanner over `buf`. | — |
| `universe_docparse_xml_next` | `int32_t universe_docparse_xml_next(void *sc, void *tok)` | Emit the next token into a 40-byte struct. | token type 0..4, or -8 / -13 |
| `universe_docparse_xml_attr_next` | `int32_t universe_docparse_xml_attr_next(const void *buf, int64_t *pcur, int64_t end, void *out)` | Next attribute in a tag region → 32-byte Attr. | 1 got / 0 done / -8 / -13 |
| `universe_docparse_xml_decode` | `int64_t universe_docparse_xml_decode(void *dst, const void *src, int64_t len)` | Entity-decode `len` bytes into `dst`. | decoded length, or -1 |

**Token struct (40 B):** `type@0` (0 start, 1 end, 2 self-close, 3 text, 4 eof),
`name_off@8`, `name_len@16` (text: the text span), `attr_off@24`, `attr_len@32`
(start/self-close only). **Scanner (24 B):** `buf@0 len@8 pos@16`.
**Attr (32 B):** `name_off@0 name_len@8 val_off@16 val_len@24`. Attribute values
are RAW (still escaped) — pass to `_decode`.

## Use in an LLVM-based environment

```llvm
declare void @universe_docparse_xml_init(ptr, ptr, i64)
declare i32  @universe_docparse_xml_next(ptr, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Read XML from stdin and print every start-tag name.

```c
// xmltags.c — print start-tag names from stdin XML
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void    universe_docparse_xml_init(void*, const void*, int64_t);
int32_t universe_docparse_xml_next(void*, void*);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int ch;
    while ((ch = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=ch; }
    unsigned char sc[24];
    unsigned char tok[40];
    universe_docparse_xml_init(sc, b, (int64_t)n);
    for (;;) {
        int t = universe_docparse_xml_next(sc, tok);
        if (t < 0 || t == 4) break;               /* error or eof */
        if (t == 0 || t == 2) {                    /* start / self-close */
            int64_t off = *(int64_t*)(tok + 8), len = *(int64_t*)(tok + 16);
            fwrite(b + off, 1, len, stdout); putchar('\n');
        }
    }
    free(b);
    return 0;
}
```

```
clang -O3 xmltags.c build/libuniverse.a -lpthread -lm -o xmltags
printf '<a><b/><c>x</c></a>' | ./xmltags     # a b c
```

## Notes

- Zero-copy, zero-alloc: token fields are byte offsets into the caller buffer,
  which is the source of truth until you consume it.
- All structs (scanner/token/attr) are caller-allocated (24/40/32 B). Attribute
  and text slices are raw; use `_decode` for entity resolution.
- Reentrant; each scanner is single-threaded state.
