# universe_encoding_flatbuffers

## Purpose

FlatBuffers binary-wire reader — zero-copy, allocation-free, bounds-checked
navigation over a caller-owned byte buffer. Implements the official little-endian
wire format: `uoffset` (u32 forward reference), `soffset` (i32 table→vtable link,
sign-extended), `voffset` (u16 field offset in the vtable). The buffer root is a
`uoffset` at byte 0; a size-prefixed buffer puts a u32 byte-count first. Every
accessor returns either an `i64` byte "loc" into the buffer or a `(ptr,len)`
view — nothing is materialized. A NEGATIVE loc means absent (the typed `read_*`
readers then substitute the caller's default).

## Exported API

| Function | Signature (C ABI) | Description | Returns |
| --- | --- | --- | --- |
| `universe_encoding_flatbuffers_root` | `int64_t universe_encoding_flatbuffers_root(const void *buf, int64_t len)` | Root table loc. | loc, or < 0 |
| `universe_encoding_flatbuffers_sized_root` | `int64_t universe_encoding_flatbuffers_sized_root(const void *buf, int64_t len)` | Root of a size-prefixed buffer. | loc, or < 0 |
| `universe_encoding_flatbuffers_field` | `int64_t universe_encoding_flatbuffers_field(const void *buf, int64_t len, int64_t table, int32_t field_id)` | Loc of a field's data (vtable lookup). | loc, or < 0 (absent) |
| `universe_encoding_flatbuffers_indirect` | `int64_t universe_encoding_flatbuffers_indirect(const void *buf, int64_t len, int64_t loc)` | Follow a `uoffset` at `loc`. | target loc, or < 0 |
| `universe_encoding_flatbuffers_read_i8` … `_read_u64` | `int32_t universe_encoding_flatbuffers_read_i32(const void *buf, int64_t len, int64_t loc, int32_t dflt, int32_t *out)` | Read a scalar (or `dflt` if absent). | 0 OK / 1 / 8 |
| `universe_encoding_flatbuffers_read_f32` / `_read_f64` | `int32_t universe_encoding_flatbuffers_read_f64(const void *buf, int64_t len, int64_t loc, double dflt, double *out)` | Read a float scalar. | 0 OK / 1 / 8 |
| `universe_encoding_flatbuffers_string` | `int32_t universe_encoding_flatbuffers_string(const void *buf, int64_t len, int64_t loc, void **out_ptr, int64_t *out_len)` | View a string field. | 0 OK / 1 / 8 |
| `universe_encoding_flatbuffers_vector` | `int32_t universe_encoding_flatbuffers_vector(const void *buf, int64_t len, int64_t loc, int64_t *out_start, int64_t *out_count)` | Vector start loc + element count. | 0 OK / 1 / 8 |
| `universe_encoding_flatbuffers_vector_elem` | `int64_t universe_encoding_flatbuffers_vector_elem(int64_t vec_start, int32_t stride, int64_t i)` | Loc of element `i`. | loc, or -1 overflow |

The `read_*` scalars take a `dflt` value used when the field is absent (loc < 0).
`i32`-entry error map: 0 OK, 1 NULL_PTR, 8 INVALID_ARG (absent/malformed).
Element `stride` = scalar size, or 4 for vectors of tables/strings (each element
is itself a `uoffset` — follow with `indirect` / `string`).

## Use in an LLVM-based environment

```llvm
declare i64 @universe_encoding_flatbuffers_root(ptr, i64)
declare i64 @universe_encoding_flatbuffers_field(ptr, i64, i64, i32)
declare i32 @universe_encoding_flatbuffers_read_i32(ptr, i64, i64, i32, ptr)
```

Link: `clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog`.

## Make a CLI

Read field 0 (an int32) of the root table from a `.bin` FlatBuffer.

```c
// fbfield0.c — print root-table field 0 (i32) of a flatbuffer file
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int64_t universe_encoding_flatbuffers_root(const void*, int64_t);
int64_t universe_encoding_flatbuffers_field(const void*, int64_t, int64_t, int32_t);
int32_t universe_encoding_flatbuffers_read_i32(const void*, int64_t, int64_t, int32_t, int32_t*);
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s buf.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long z = ftell(f); rewind(f);
    unsigned char *buf = malloc(z); fread(buf, 1, z, f); fclose(f);
    int64_t root = universe_encoding_flatbuffers_root(buf, z);
    int64_t loc  = universe_encoding_flatbuffers_field(buf, z, root, /*field_id*/0);
    int32_t v = 0;
    universe_encoding_flatbuffers_read_i32(buf, z, loc, /*default*/-1, &v);
    printf("field0 = %d\n", v);       /* -1 if the field is absent */
    free(buf);
    return 0;
}
```

```
clang -O3 fbfield0.c build/libuniverse.a -lpthread -lm -o fbfield0
./fbfield0 message.bin
```

## Notes

- Zero-copy, zero-alloc: locs and views point into the caller buffer, which is
  the source of truth. Every access is bounds-checked; a corrupt offset yields an
  absent (negative) loc / INVALID_ARG rather than an OOB read.
- Little-endian wire format (native on all targets). Reentrant, stateless.
