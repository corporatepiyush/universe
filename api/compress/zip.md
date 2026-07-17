# compress/zip — ZIP archive reader

## Purpose
A ZIP archive reader over a caller-owned buffer (APPNOTE local/central
directory layout). It is ZERO-COPY and ZERO-ALLOC: the whole `.zip` sits in the
caller buffer and every entry name is reported as a `{ptr,len}` slice INTO that
buffer, never copied. The only caller state is a tiny reader struct and a
per-entry struct the caller allocates. It locates the End-Of-Central-Directory
(scanning backward for its signature past a possible trailing comment), walks the
central directory, and inflates a chosen member's DEFLATE data in place via
`universe_compress_inflate`. This is exactly the shape the XLSX/PDF parsers need.
All multi-byte fields are read byte-wise (little-endian) so layout is identical
on every target.

## Exported API
| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_compress_zip_open(const void *buf, int64_t len, void *reader)` | Parse EOCD/central dir into `reader` (32 B, caller-allocated) | 0 OK, 1 NULL, 13 PARSE |
| `int64_t universe_compress_zip_count(void *reader)` | Number of entries | count |
| `int32_t universe_compress_zip_entry(void *reader, int64_t index, void *entry)` | Fill `entry` (48 B, caller-allocated) for `index` | 0 OK, 7 INVALID_INDEX, 13 PARSE |
| `int64_t universe_compress_zip_extract(void *reader, void *entry, void *dst, int64_t dstcap)` | Extract (stored copy or DEFLATE) into `dst` | uncompressed length, or negative error |

Caller structs (allocate as raw byte buffers, access by offset):
```
Reader (32 B): buf@0(void*) len@8(int64) cd_off@16(int64) count@24(int64)
Entry  (48 B): name@0(void*) namelen@8(int64) method@16(int64)
               compsize@24(int64) uncompsize@32(int64) localoff@40(int64)
```
`name` points into the caller's archive buffer (not NUL-terminated); use
`namelen`. `method`: 0 = stored, 8 = DEFLATE.

## Use in an LLVM-based environment
```llvm
declare i32 @universe_compress_zip_open(ptr, i64, ptr)
declare i64 @universe_compress_zip_count(ptr)
declare i32 @universe_compress_zip_entry(ptr, i64, ptr)
declare i64 @universe_compress_zip_extract(ptr, ptr, ptr, i64)
```
```
clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI
`zip_ls.c` — list entries of a `.zip` (name + sizes + method).
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int32_t universe_compress_zip_open(const void *buf, int64_t len, void *reader);
int64_t universe_compress_zip_count(void *reader);
int32_t universe_compress_zip_entry(void *reader, int64_t index, void *entry);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.zip\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(n);
    fread(buf, 1, n, f); fclose(f);

    unsigned char reader[32], entry[48];
    if (universe_compress_zip_open(buf, n, reader) != 0) {
        fprintf(stderr, "not a zip / parse error\n"); return 1;
    }
    int64_t count = universe_compress_zip_count(reader);
    for (int64_t i = 0; i < count; i++) {
        if (universe_compress_zip_entry(reader, i, entry) != 0) continue;
        char   *name    = *(char **)(entry + 0);
        int64_t namelen = *(int64_t *)(entry + 8);
        int64_t method  = *(int64_t *)(entry + 16);
        int64_t usz     = *(int64_t *)(entry + 32);
        printf("%.*s  (%lld bytes, method %lld)\n",
               (int)namelen, name, (long long)usz, (long long)method);
    }
    free(buf);
    return 0;
}
```
```
clang -O3 zip_ls.c build/libuniverse.a -lpthread -lm -o zip_ls
./zip_ls some.zip
```

## Notes
- Zero-copy: entry `name` slices point into the caller's archive buffer, which
  must outlive all entry/extract calls. The buffer is the source of truth.
- `extract` handles method 0 (stored copy) and 8 (DEFLATE, via the inflate
  module); other methods are rejected.
- Reader (32 B) and entry (48 B) structs are caller-allocated raw buffers.
- Thread-safe for concurrent reads of the same buffer (no internal state).
