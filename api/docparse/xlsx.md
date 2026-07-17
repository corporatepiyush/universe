# universe_docparse_xlsx

## Purpose

XLSX (ISO/IEC 29500 SpreadsheetML) reader over a caller ZIP buffer. Builds on
`compress/zip` + `compress/inflate`: `open()` locates
`xl/sharedStrings.xml` and `xl/worksheets/sheet1.xml` and inflates each ONCE into
a private malloc buffer (the only allocations); everything after is zero-copy —
cell values are `{ptr,len}` slices INTO the inflated buffers. The shared-string
table is pre-indexed at open into a flat `(offset,length)` array, so a `t="s"`
cell is an O(1) lookup. Cell iteration is a small state machine over the sheet's
XML pull-token stream; A1 refs are decoded to `(col,row)`. Numbers/booleans are
parsed to a double in place (no libc strtod); string slices stay RAW (feed to
`universe_docparse_xml_decode`).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_xlsx_open` | `int32_t universe_docparse_xlsx_open(const void *buf, int64_t len, void *wb)` | Open a workbook into an 80-byte struct. | 0 / 1 / 2 / 5 / 13 |
| `universe_docparse_xlsx_close` | `void universe_docparse_xlsx_close(void *wb)` | Free the workbook's inflated buffers. | — |
| `universe_docparse_xlsx_sst_count` | `int64_t universe_docparse_xlsx_sst_count(void *wb)` | Shared-string count. | count |
| `universe_docparse_xlsx_sst_get` | `int32_t universe_docparse_xlsx_sst_get(void *wb, int64_t idx, void *out)` | View shared string `idx`. | 0 / errcode |
| `universe_docparse_xlsx_sheet_init` | `void universe_docparse_xlsx_sheet_init(void *wb, void *cur)` | Init a 32-byte cell cursor. | — |
| `universe_docparse_xlsx_cell_next` | `int32_t universe_docparse_xlsx_cell_next(void *wb, void *cur, void *cell)` | Next cell into a 56-byte struct. | 0 got / 5 end / errcode |

**Workbook (80 B):** `reader@0` (32 B zip reader), `shared_buf@32`,
`shared_len@40`, `sheet_buf@48`, `sheet_len@56`, `sst_ptr@64`, `sst_count@72`.
**Cursor (32 B):** `xml-scanner@0` (24 B), `cur_row@24`. **Cell (56 B):**
`row@0`, `col@8`, `type@16` (0 num, 1 shared, 2 bool, 3 string, 4 empty),
`val_ptr@24`, `val_len@32`, `num@40 (double)`, `sidx@48` (shared-string index).

## Use in an LLVM-based environment

```llvm
declare i32  @universe_docparse_xlsx_open(ptr, i64, ptr)
declare i32  @universe_docparse_xlsx_cell_next(ptr, ptr, ptr)
declare void @universe_docparse_xlsx_close(ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Print `row,col,number` for every numeric cell of sheet 1.

```c
// xlsxnums.c — dump numeric cells of an .xlsx
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int32_t universe_docparse_xlsx_open(const void*, int64_t, void*);
void    universe_docparse_xlsx_sheet_init(void*, void*);
int32_t universe_docparse_xlsx_cell_next(void*, void*, void*);
void    universe_docparse_xlsx_close(void*);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.xlsx\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long z = ftell(f); rewind(f);
    unsigned char *buf = malloc(z); fread(buf, 1, z, f); fclose(f);
    unsigned char wb[80], cur[32], cell[56];
    if (universe_docparse_xlsx_open(buf, z, wb) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    universe_docparse_xlsx_sheet_init(wb, cur);
    while (universe_docparse_xlsx_cell_next(wb, cur, cell) == 0) {
        int32_t type = *(int32_t*)(cell + 16);
        if (type == 0) {                          /* numeric */
            long long row = *(int64_t*)(cell + 0), col = *(int64_t*)(cell + 8);
            double num = *(double*)(cell + 40);
            printf("%lld,%lld,%g\n", row, col, num);
        }
    }
    universe_docparse_xlsx_close(wb); free(buf);
    return 0;
}
```

```
clang -O3 xlsxnums.c build/libuniverse.a -lpthread -lm -o xlsxnums
./xlsxnums book.xlsx
```

## Notes

- All structs (workbook 80 B, cursor 32 B, cell 56 B) are caller-allocated.
  `open` mallocs the two inflated buffers; `close` frees them — always pair them.
- Reads only `sheet1.xml`; rich-text `<si>` resolves to the first `<t>` run
  (documented limitation). String cell values are RAW — decode with
  `universe_docparse_xml_decode`.
- Reentrant per workbook; single-threaded cursor.
