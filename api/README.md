# universe — API reference

`universe` is a Go-stdlib-class SDK written entirely in hand-written LLVM IR. It
ships as a static archive `build/libuniverse.a` and a shared library
(`build/libuniverse.dylib` on macOS, `build/libuniverse.so` on Linux). Every
module exposes a flat **C ABI** (`nounwind`, default visibility), so any language
or toolchain that speaks the C ABI — C, another `.ll`, Rust `extern "C"`, Zig,
Python `ctypes` — can call it.

This folder documents **every module**: one Markdown file per
`src/<domain>/<module>.ll`, each with (1) Purpose, (2) an Exported API table with
C signatures + return/error conventions, (3) how to call it from an LLVM-based
environment, (4) a runnable CLI example, and (5) notes on layout/threading/
allocator ownership.

## Build the library

```sh
make lib      # build/libuniverse.a   (static; the usual link target)
make dylib    # build/libuniverse.dylib / .so
```

## Use a module from LLVM IR

Declare the exported symbol(s) and call them; link against the archive:

```llvm
declare i64 @universe_compress_lz4_encode(ptr, i64, ptr, i64)
```
```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

`-lpthread` is needed by the concurrent/threadpool/observ modules; `-lm` by any
module that uses libm (linear algebra, ML, dataframe float ops — macOS folds libm
into libSystem, but Linux/glibc and Alpine/musl require it explicit).

## Use a module from C (make a CLI)

Each module doc includes a small `main` that parses `argv`/stdin and calls the
public functions. The pattern:

```c
#include <stdint.h>
extern int64_t universe_compress_lz4_encode(void*, int64_t, const void*, int64_t);
int main(void) { /* read stdin, call, write stdout */ }
```
```sh
clang -O3 mycli.c build/libuniverse.a -lpthread -lm -o mycli
```

## Conventions

- **Names:** `universe_<domain>_<op>` (a few early codecs use a short
  `universe_<codec>_*` form — noted in their domain README).
- **Error codes (i32):** `0` OK, `1` NULL_PTR, `2` OUT_OF_MEMORY, `3`
  SIZE_OVERFLOW, `4` EMPTY, `5` NOT_FOUND, `6` FULL, `7` INVALID_INDEX, `8`
  INVALID_ARG, `9` DUPLICATE. Codecs commonly return a length or a **negative**
  error instead; handle-returning constructors signal failure with `NULL`. Each
  module doc states its exact convention.
- **Allocators:** prefer the SDK allocators (`api/allocator/`) — arena for
  batch/scratch, pool/slab for fixed-size churn, TLSF/buddy for general heaps.
- **Portability:** IR is triple-agnostic; the build injects the host triple. The
  suite runs on macOS and on real Linux ≥ 6.15 under both glibc and musl
  (`make docker-test-all`). SIMD kernels ship a scalar fallback and lower to
  SSE2/NEON on the 128-bit baseline.

## Domains

| Domain | Modules | What it covers |
|---|---:|---|
| [allocator](allocator/) | 3 | **slab (default, injectable)**, pool (fixed bounded), arena (intra-module scratch bursts) — single-threaded; general-purpose TLSF/buddy/hybrid removed for now (see BENCHMARKS.md) |
| [bignum](bignum/) | 1 | arbitrary-precision integers, modexp/Montgomery |
| [common](common/) | 1 | shared primitives (rotl/rotr, big-endian load/store, checked math) |
| [compress](compress/) | 5 | inflate/DEFLATE, LZ4, Snappy, Zstd, ZIP |
| [concurrent](concurrent/) | 6 | SPSC/MPSC/MPMC queues, sharded map, striped counter, reclamation |
| [crypto](crypto/) | 10 | MD5, SHA-1/256/512, HMAC, PBKDF2, bcrypt, RSA, ECDSA, Ed25519 |
| [dataframe](dataframe/) | 9 | columnar frame + SIMD arith/compare/cast, reduce, sort, groupby, join |
| [docparse](docparse/) | 6 | XML, DOCX, XLSX, PDF, Parquet metadata + encodings |
| [encoding](encoding/) | 7 | hex, base64, UTF-8/16, varint, FlatBuffers, Thrift |
| [http](http/) | 2 | HTTP/1.1 client+server, io_uring variant |
| [io](io/) | 1 | buffered readers/writers |
| [ioring](ioring/) | 1 | io_uring submission/completion (Linux) |
| [linalg](linalg/) | 3 | dense linear algebra — SIMD GEMM (`matrix`), LU/QR/Cholesky/solve/det/inv (`factor`), symmetric eigen + SVD (`eigen`) |
| [llm](llm/) | 1 | OpenAI-style chat/completion client |
| [ml](ml/) | 9 | SIMD kernels, k-means, kNN, linear/logistic, PCA, HNSW, IVF, quant, rerank |
| [net](net/) | 2 | TCP sockets, connection pool (constants build-selected per OS) |
| [observ](observ/) | 1 | async logging/metrics/tracing |
| [parse](parse/) | 2 | JSON, CSV (SIMD structural scan) |
| [search](search/) | 2 | Aho-Corasick, suffix array |
| [simd](simd/) | 1 | shared 128-bit byte-scan kernels |
| [sort](sort/) | 9 | radix, quick/intro, merge, heap, counting, + educational sorts |
| [strings](strings/) | 2 | immutable string/SSO, growable bytes (SIMD find/compare/case) |
| [structures_assoc](structures_assoc/) | 4 | swiss hashmap, minimal-perfect hash, treemap (+ sharded) |
| [structures_graph](structures_graph/) | 2 | CSR graph (+ sharded) |
| [structures_linear](structures_linear/) | 12 | array, deque, ring, stack/queue, lists, LRU, bitset, sparse set, heap |
| [structures_probabilistic](structures_probabilistic/) | 1 | cuckoo / XOR / blocked-Bloom filters |
| [structures_succinct](structures_succinct/) | 2 | Elias-Fano postings + RLE, roaring bitmap |
| [structures_trees](structures_trees/) | 9 | ART, B-tree (+ sharded), AVL/RB, fenwick, segtree, skiplist, heaps, union-find |
| [threadpool](threadpool/) | 1 | work-stealing thread pool |

Each domain folder has a `README.md` index linking its module docs.
