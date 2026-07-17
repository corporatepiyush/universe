# structures_succinct — compact & succinct structures

Near-minimal-space encodings for sorted/compressible integer data.

| Module | Contents | Choose when |
|---|---|---|
| [eliasfano](eliasfano.md) | Elias-Fano sequence, delta+varint & SIMD postings, RLE | Static monotone sequences with random access + successor; compact sorted-id lists |
| [roaring](roaring.md) | Roaring bitmap (adaptive array/bitmap containers) + SIMD dense kernels | Large clustered/sparse u32 sets with fast set algebra |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`):

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

Codec functions return byte/element counts (negative on error); structures use
0 OK, 1 NULL_PTR, 2 OOM, 8 INVALID_ARG.
