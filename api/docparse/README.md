# docparse — universe API

Document and columnar-format parsers: PDF, DOCX, XLSX, XML, and Apache Parquet
(metadata + column encodings). All are zero-copy / allocation-frugal readers over
a **caller-owned byte buffer** — they return slices, offsets, or fill
caller-provided structs. Shipped in `libuniverse.a` (`make lib`); C ABI,
`nounwind`. Exported names are `universe_docparse_*`.

## Modules

| Module | What it is |
| --- | --- |
| [xml](xml.md) | Zero-copy XML pull-tokenizer (drives the OOXML readers). |
| [docx](docx.md) | DOCX (WordprocessingML) visible-text extraction. |
| [xlsx](xlsx.md) | XLSX (SpreadsheetML) workbook / cell reader. |
| [pdf](pdf.md) | PDF 2.0 reader: xref, objects, FlateDecode streams, text extraction. |
| [parquet_meta](parquet_meta.md) | Parquet footer + Thrift-compact metadata navigator. |
| [parquet_encoding](parquet_encoding.md) | Parquet column-encoding decoder kernels. |

## Common conventions

- **Untrusted input.** Every read is bounds-checked against the buffer length;
  malformed/truncated framing returns an error code, never a crash. Parsers are
  built + fuzzed under ASan/UBSan (sanitizer gate).
- **IO/compute split.** The zip+inflate (or file read) IO phase fills a buffer;
  the parse phase is pure compute emitting slices / structs into caller memory.
- **Error codes (i32 entry points):** 0 OK, 1 NULL_PTR, 2 OOM, 5 NOT_FOUND
  (also "end of iteration"), 6 FULL (output capacity), 7 INVALID_INDEX,
  8 INVALID_ARG (malformed/truncated). `i64`-returning entry points return a
  produced length or a negative error.
- DOCX/XLSX/PDF build on `compress/zip` + `compress/inflate`; parquet_meta builds
  on `encoding/thrift`.

Link line: `clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.
