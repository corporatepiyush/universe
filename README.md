# Universe

A metal-speed core SDK written entirely in hand-written **LLVM IR** — an
assembly-level standard library in the spirit of a modern language's core
packages (strings, buffers, containers, hashmaps, trees, allocators, threads,
TCP/HTTP, encoding, crypto, ML kernels), where every capability is implemented
at the metal for maximum throughput.

Everything under `src/` and `tests/` is hand-authored `.ll`. There are **no C
sources or headers anywhere** — modules *and* their tests are LLVM IR. Each
module is designed from its functional spec, not ported from any existing
source.

- **98 IR modules · ~53k lines of IR · 90 test binaries.**
- Targets: **AMD64 + AArch64/ARM64** · macOS 26+ · Linux 6.15+ · FreeBSD 15+.
- Toolchain: **Homebrew LLVM ≥ 22.1**. IR modules carry no target triple; the
  build injects it, so one source compiles for every target.

## Why IR

Sitting one level below C lets the SDK control exactly what the CPU executes:
data layout, allocation strategy, atomic orderings, branch shape, and SIMD
lowering — the places a C compiler cannot reach. The same clean IR form is
lowered by the AMD64 and AArch64 backends to their respective instructions
(AVX/NEON, `cmov`/`csel`, `lock xadd`/`LDADD`), so a single implementation
targets both ISAs without per-CPU source.

## Layout

```
src/<domain>/*.ll         modules (one per algorithm / structure / variant)
tests/<domain>/test_*.ll  IR test drivers (each a standalone binary with a --bench mode)
tests/support/ut.ll       shared harness (asserts, RNG, timer, bench percentiles)
docker/                   Linux test image (runs the suite on a real kernel)
tools/                    authoring utilities
Makefile                  per-domain isolated builds (parallel-safe)
```

## Build & test

```sh
make test              # build + run the whole suite (host triple)
make test-<domain>     # one domain, e.g. make test-sort
make crosscheck        # codegen every module for 4 foreign triples (compile gate)
make lib dylib         # build/libuniverse.a + build/libuniverse.{dylib,so}
make docker-test       # RUN the suite on real Linux >= 6.15 (runtime gate)
```

Requires `brew install llvm` (>= 22.1). The toolchain path defaults to
`/opt/homebrew/opt/llvm/bin` and is overridable (`make test LLVM=... HOST_TRIPLE=...`).

Each test's `--bench` mode reports a warmed-up latency **distribution**
(min / p50 / p95 / p99), not a single number:

```
$ ./build/bin/sort/test_int_sorts --bench
BENCH radix i32 n=65536: min=4.77 p50=4.79 p95=5.13 p99=5.13 ns/op (16 reps)
BENCH qsort i32 n=65536: min=69.33 p50=71.20 p95=91.12 p99=91.12 ns/op (16 reps)
```

## What's inside

**Core types** — string (immutable view + SSO) and bytes/builder; array, deque,
ring buffer (+ concurrent), queue, stack, singly/doubly linked lists, LRU,
bitset, sparse set.

**Maps · trees · heaps** — SIMD/swiss hashmap, minimal perfect hash, ordered map
(array-backed + sharded); B-tree, adaptive radix tree (ART), skiplist, Fenwick
tree, segment tree, union-find; binary / d-ary / pairing / radix heaps.
Concurrent sharded variants of ART, B-tree, ordered map, and graph.

**Probabilistic · succinct** — bloom / cuckoo / xor filters; Elias-Fano; roaring
bitmap.

**Graph · search · sort** — adjacency-list graph (BFS / DFS / Dijkstra /
connected-components); Aho-Corasick multi-pattern search; suffix array; the sort
family (insertion, shell, merge, quick, heap, counting, LSD radix).

**Runtime** — allocators (arena, pool, slab, buddy; sequential and concurrent);
lock-free queues (SPSC, MPSC, MPMC), striped counter, sharded map; reclamation
(epoch / hazard-pointer / seqlock); thread pool; async observability
(ring-buffer logging, striped metrics, HDR-style latency histograms).

**IO · networking** — buffered reader/writer (mmap, `writev`); TCP client/server
and connection pool; HTTP/1.1 (keep-alive, chunked) with an io_uring server
variant; io_uring primitives (Linux).

**Encoding · crypto · numeric** — UTF-8/UTF-16, hex, base64 (SIMD), varint,
JSON, CSV; inflate (DEFLATE / zlib / gzip), LZ4, Zstd, ZIP; PDF / XLSX / XML
parsers. SHA-1/256/512, MD5, HMAC, PBKDF2, bcrypt, Ed25519, ECDSA (P-256), RSA —
checked against known-answer test vectors (RFC / FIPS / NIST). Bignum; SIMD scan
kernels; ML: k-means, linear models, k-NN, PCA, and BLAS-lite kernels.

Some capabilities ship in multiple implementations — different hot paths deserve
different layouts. Concurrent and Linux-only modules are noted in their headers.

## Design principles

- **SIMD-first.** Every hot data-parallel kernel ships a portable 128-bit vector
  path (lowering to SSE2 on AMD64 and NEON on AArch64 — baseline on every
  target) plus a scalar fallback that also serves as the cross-check oracle.
- **Single-allocation, pointer-lean layouts.** Header + payload in one `malloc`;
  index-linked nodes instead of pointer webs; power-of-two capacities with
  mask-wrap; overflow-checked size math.
- **Lock-free where the contract allows.** Every atomic ordering is justified;
  producer/consumer and shard state are padded to separate cache lines;
  wait-free is preferred over lock-free over locks.
- **Portable by construction.** No target triple in the IR; OS-divergent bits
  (socket constants, `sockaddr` layout, the errno accessor) are isolated into
  per-target modules the build selects by host.
- **Verified, not asserted.** Codecs and hashes are gated on external
  known-answer vectors; parsers and pointer-math code run under
  AddressSanitizer/UBSan; every module must codegen for all four cross-triples;
  and the full suite executes on a real Linux kernel via `make docker-test`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
