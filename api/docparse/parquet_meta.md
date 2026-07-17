# universe_docparse_parquet_meta

## Purpose

Apache Parquet metadata reader — footer location + Thrift-compact navigation. A
Parquet file is framed by `"PAR1"` at both ends; the tail is
`[FileMetaData (thrift-compact)][u32 footer_len (LE)]["PAR1"]`. This module is a
READ-ONLY, allocation-free navigator over a caller-owned buffer. It parses the
FileMetaData thrift stream lazily via the shared `encoding/thrift` reader: the
metadata is tiny, so random access to row group / column N re-scans from the
recorded list start rather than materializing everything. Every parsed record
lands in a caller-provided fixed struct whose byte layout is defined explicitly,
so it is identical on every target.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_docparse_parquet_footer` | `int32_t universe_docparse_parquet_footer(const void *buf, int64_t len, void **out_meta_ptr, int64_t *out_meta_len)` | Locate the FileMetaData thrift blob in a file image. | 0 / 1 / 5 / 8 |
| `universe_docparse_parquet_meta_open` | `int32_t universe_docparse_parquet_meta_open(const void *meta_buf, int64_t len, void *fmd)` | Parse into a 64-byte FileMeta handle. | 0 / 1 / 8 |
| `universe_docparse_parquet_meta_num_rows` | `int64_t universe_docparse_parquet_meta_num_rows(void *fmd)` | Total row count. | num_rows |
| `universe_docparse_parquet_meta_row_group_count` | `int64_t universe_docparse_parquet_meta_row_group_count(void *fmd)` | Row-group count. | count |
| `universe_docparse_parquet_schema_next` | `int32_t universe_docparse_parquet_schema_next(void *fmd, void *cursor, void *out_elem)` | Next schema element (48-byte struct). | 0 / 5 end / errcode |
| `universe_docparse_parquet_row_group` | `int32_t universe_docparse_parquet_row_group(void *fmd, int64_t rg_idx, void *out_rg)` | Row group `rg_idx` (48-byte struct). | 0 / 7 / 8 |
| `universe_docparse_parquet_column` | `int32_t universe_docparse_parquet_column(void *out_rg, int64_t col_idx, void *out_cc)` | Column chunk `col_idx` (56-byte struct). | 0 / 7 / 8 |
| `universe_docparse_parquet_page_header` | `int32_t universe_docparse_parquet_page_header(const void *buf, int64_t len, int64_t pos, void *out_ph, int64_t *out_next_pos)` | Parse the PageHeader at `pos` (24-byte struct). | 0 / 8 |

**i32 codes:** 0 OK, 1 NULL_PTR, 5 NOT_FOUND (also end-of-iteration),
7 INVALID_INDEX, 8 INVALID_ARG (malformed/truncated). Caller struct layouts (all
explicit offsets): **FileMeta (64 B)** `base@0`, `len@8`, `version@16`, …;
**SchemaElement out (48 B)** `type@0` (-1 absent), `type_length@4`, …;
**RowGroup out (48 B)** `base@0`, `len@8`, `col_count@16`, …; **ColumnChunk out
(56 B)** `type@0`, `codec@4`, `first_encoding@8`, …; **PageHeader out (24 B)**
`type@0`, `uncompressed_page_size@4`, `compressed_page_size@8`, ….

## Use in an LLVM-based environment

```llvm
declare i32 @universe_docparse_parquet_footer(ptr, i64, ptr, ptr)
declare i32 @universe_docparse_parquet_meta_open(ptr, i64, ptr)
declare i64 @universe_docparse_parquet_meta_num_rows(ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Print a Parquet file's total row count and row-group count.

```c
// pqmeta.c — print num_rows and row_group_count of a .parquet file
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int32_t universe_docparse_parquet_footer(const void*, int64_t, void**, int64_t*);
int32_t universe_docparse_parquet_meta_open(const void*, int64_t, void*);
int64_t universe_docparse_parquet_meta_num_rows(void*);
int64_t universe_docparse_parquet_meta_row_group_count(void*);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.parquet\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long z = ftell(f); rewind(f);
    unsigned char *buf = malloc(z); fread(buf, 1, z, f); fclose(f);
    void *mp = NULL; int64_t ml = 0;
    if (universe_docparse_parquet_footer(buf, z, &mp, &ml) != 0) { fprintf(stderr, "bad footer\n"); return 1; }
    unsigned char fmd[64];
    if (universe_docparse_parquet_meta_open(mp, ml, fmd) != 0) { fprintf(stderr, "bad meta\n"); return 1; }
    printf("rows=%lld row_groups=%lld\n",
           (long long)universe_docparse_parquet_meta_num_rows(fmd),
           (long long)universe_docparse_parquet_meta_row_group_count(fmd));
    free(buf);
    return 0;
}
```

```
clang -O3 pqmeta.c build/libuniverse.a -lpthread -lm -o pqmeta
./pqmeta data.parquet        # rows=1000 row_groups=1
```

## Notes

- Fully allocation-free: the FileMeta handle (64 B) and every out struct are
  caller-allocated; string fields are `(ptr,len)` VIEWS into the metadata buffer,
  which must outlive the handle. Uses the shared thrift reader (32-byte stack).
- Random access re-scans lazily — cheap for the tiny metadata section.
- Reentrant; read-only over the buffer.
