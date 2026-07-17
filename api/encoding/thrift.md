# universe_encoding_thrift

## Purpose

Thrift compact-protocol reader — read-only, allocation-free cursor over a
caller-owned byte buffer. The compact protocol frames structs as unsigned LEB128
varints, zig-zag signed integers, field headers `(delta<<4)|type` (with a zig-zag
varint field id when the 4-bit delta is 0), collection headers `(size<<4)|type`
(varint size when the nibble is 15), little-endian doubles, and varint-length
binary/string. Reader state is a 32-byte struct in caller memory; binary/string
reads return a `(ptr,len)` VIEW into the input, never a copy. Every byte read is
bounds-checked; the two leaf accessors are the only places that touch the buffer,
so no public read can OOB. `skip` is depth-limited (max 64) against hostile
nesting.

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_encoding_thrift_init` | `void universe_encoding_thrift_init(void *reader, const void *buf, int64_t len)` | Init a 32-byte reader over `buf`. | — |
| `universe_encoding_thrift_read_uvarint` | `int32_t universe_encoding_thrift_read_uvarint(void *r, uint64_t *out)` | Read an unsigned LEB128 varint. | 0 / 8 |
| `universe_encoding_thrift_read_i8` | `int32_t universe_encoding_thrift_read_i8(void *r, int8_t *out)` | Read a raw byte. | 0 / 8 |
| `universe_encoding_thrift_read_i16` / `_i32` / `_i64` | `int32_t universe_encoding_thrift_read_i32(void *r, int32_t *out)` | Read a zig-zag signed integer. | 0 / 8 |
| `universe_encoding_thrift_read_double` | `int32_t universe_encoding_thrift_read_double(void *r, double *out)` | Read an LE double. | 0 / 8 |
| `universe_encoding_thrift_read_binary` | `int32_t universe_encoding_thrift_read_binary(void *r, void **out_ptr, int64_t *out_len)` | View a binary/string (ptr+len). | 0 / 8 |
| `universe_encoding_thrift_field` | `int32_t universe_encoding_thrift_field(void *r, uint8_t *out_ctype, int16_t *out_fid)` | Read a field header; STOP ⇒ `out_ctype==0`. | 0 / 8 |
| `universe_encoding_thrift_collection` | `int32_t universe_encoding_thrift_collection(void *r, uint8_t *out_etype, int32_t *out_size)` | Read a list/set/map header. | 0 / 8 |
| `universe_encoding_thrift_skip` | `int32_t universe_encoding_thrift_skip(void *r, int32_t ctype, int32_t depth)` | Skip a value of compact type `ctype`. | 0 / 8 |

**Compact type nibbles:** STOP=0, BOOL_TRUE=1, BOOL_FALSE=2, I8=3, I16=4, I32=5,
I64=6, DOUBLE=7, BINARY/STRING=8, LIST=9, SET=10, MAP=11, STRUCT=12.
`i32` codes: 0 OK, 1 NULL_PTR, 8 INVALID_ARG (malformed/truncated). Nothing
panics.

## Use in an LLVM-based environment

```llvm
declare void @universe_encoding_thrift_init(ptr, ptr, i64)
declare i32  @universe_encoding_thrift_field(ptr, ptr, ptr)
declare i32  @universe_encoding_thrift_skip(ptr, i32, i32)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Walk the top-level fields of a compact-protocol struct on stdin, printing each
field id and type nibble (skipping the value).

```c
// thriftfields.c — list top-level (fieldId, typeNibble) of a compact struct on stdin
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
void    universe_encoding_thrift_init(void*, const void*, int64_t);
int32_t universe_encoding_thrift_field(void*, uint8_t*, int16_t*);
int32_t universe_encoding_thrift_skip(void*, int32_t, int32_t);
int main(void) {
    unsigned char *b = NULL; size_t cap = 0, n = 0; int c;
    while ((c = getchar()) != EOF) { if (n==cap){cap=cap?cap*2:4096;b=realloc(b,cap);} b[n++]=c; }
    unsigned char reader[32];
    universe_encoding_thrift_init(reader, b, (int64_t)n);
    for (;;) {
        uint8_t ctype; int16_t fid;
        if (universe_encoding_thrift_field(reader, &ctype, &fid) != 0) { fprintf(stderr, "bad\n"); break; }
        if (ctype == 0) break;                    /* STOP */
        printf("field %d type %u\n", fid, ctype);
        if (universe_encoding_thrift_skip(reader, ctype, 0) != 0) { fprintf(stderr, "bad\n"); break; }
    }
    free(b);
    return 0;
}
```

```
clang -O3 thriftfields.c build/libuniverse.a -lpthread -lm -o thriftfields
./thriftfields < struct.bin
```

## Notes

- Zero-copy, zero-alloc: the 32-byte reader is caller memory; binary/string reads
  are `(ptr,len)` views into the input buffer (the source of truth). This is the
  reader that `docparse/parquet_meta` navigates Parquet metadata with.
- Untrusted-input safe: every read is bounds-checked, shift-guarded (hazard #10),
  and `skip` is depth-limited (max 64). Reentrant per reader; single-threaded.
