# Allocator benchmark — which default, which module

Measured with `tests/blackbox/bb_alloc_bench.c` (public C ABI, linked against
`build/libuniverse.a`), on **real Linux kernel 6.17** under **both glibc (Debian)
and musl (Alpine)**. Each cell is **ns/op**, median of 5 reps; churn = steady-state
(prefill a working set, then each op frees one slot + allocs a replacement, so a
cell is one free+alloc pair). `N/A` = the allocator can't serve that profile
(pool/slab are fixed-size; arena has no per-object free). All allocators floor
their OS request at 16 KiB.

## glibc (Debian, kernel 6.17)

| profile | malloc | tlsf | hybrid | buddy | pool | slab | arena |
|---|--:|--:|--:|--:|--:|--:|--:|
| node_churn_32B    | 10.5 | 17.4 | 11.3 | 12.2 | **4.7** | 7.0 | — |
| small_churn_64B   | 11.2 | 18.3 | 11.8 | 12.1 | **4.8** | 7.0 | — |
| medium_churn_512B | 10.8 | 36.4 | 19.7 | 11.9 | **4.8** | 6.4 | — |
| large_churn_8KB   | 21.6 | 30.8 | 33.8 | 11.2 | **4.9** | 5.9 | — |
| mixed_16–2048B    | 65.4 | 50.9 | 60.9 | **15.2** | — | — | — |
| bumpreset_64B     | 15.7 | 13.5 | 10.4 | 13.9 | — | — | **2.6** |
| build_alloc_64B   | 27.5 | 24.5 | 51.9 | 17.6 | — | — | **2.6** |

## musl (Alpine, kernel 6.17)

| profile | malloc | tlsf | hybrid | buddy | pool | slab | arena |
|---|--:|--:|--:|--:|--:|--:|--:|
| node_churn_32B    | 26.3 | 16.7 | 11.3 | 11.8 | **4.7** | 6.8 | — |
| small_churn_64B   | 27.3 | 17.2 | 11.8 | 11.7 | **4.7** | 6.7 | — |
| medium_churn_512B | 41.5 | 34.8 | 19.6 | 11.5 | **4.7** | 6.2 | — |
| large_churn_8KB   | 34.7 | 31.0 | 34.0 | 11.2 | **5.0** | 5.9 | — |
| mixed_16–2048B    | 51.9 | 51.6 | 63.6 | **15.2** | — | — | — |
| bumpreset_64B     | 20.8 | 24.4 | 26.6 | 15.9 | — | — | **3.3** |
| build_alloc_64B   | 23.0 | 25.5 | 57.0 | 17.7 | — | — | **11.6** |

## Winner per profile
- **fixed-size churn (node/small/medium/large): `pool` (~4.7 ns), then `slab` (~6 ns).** Both crush everything else; `slab` grows, `pool` is fixed-capacity.
- **mixed variable-size churn: `buddy` (~15 ns) — ~3× faster than tlsf/hybrid/malloc.** The surprise result.
- **bump + bulk-reset / build scratch: `arena` (~2.6 ns).** No contest.
- **large fixed churn: `buddy` (~11 ns) and pool/slab** are all excellent; malloc/tlsf/hybrid lag.

## Judgement

### Which allocator per module pattern (use this)
| Module pattern | Allocator | Why |
|---|---|---|
| Fixed-size nodes — lists, trees, hashmap/skiplist/graph nodes | **pool** (bounded) / **slab** (growable) | 4.7 / 6 ns; zero fragmentation |
| Build/parse/query scratch — alloc a burst, free all together | **arena** | 2.6 ns; O(1) reset |
| General mixed-size, bounded working set — dataframe/ml/codec buffers | **buddy** | fastest general (15 ns mixed); needs a sized region |
| General mixed-size, UNBOUNDED / must grow | **tlsf** | only growable general allocator; slower but safe |
| Small-object-heavy mixed churn | **hybrid** | slab fast-path (~11 ns) helps small, but poor on large/build |

### Which is worthy of DEFAULT — the honest answer
No single allocator is simultaneously fastest, growable, and low-fragmentation:
- **`buddy` is the fastest general-purpose allocator** (dominates mixed churn, excellent on large), BUT it manages a FIXED region and rounds to powers of two (up to ~2× internal fragmentation). Not safe as a blind universal default — a growing structure exhausts it.
- **`tlsf` is the safe universal default** (the only growable, arbitrary-size, low-fragmentation allocator), BUT the benchmark shows it is the SLOWEST on fixed churn and only middling on mixed — its good-fit search + split/coalesce cost real time.
- **`hybrid`** helps small objects but is weak on large/build/mixed — not a general default.
- **libc malloc**: glibc's ptmalloc is competitive on fixed churn but poor on mixed (65 ns); musl's is 2–3× slower across the board — so the SDK allocators' advantage is largest under musl.

**Recommendation:**
1. **Default stays `tlsf`** for correctness/safety (growable, tight) — but it is NOT the speed default; hot paths must pick the pattern-specific allocator above, never the default blindly.
2. **High-value follow-up: make `buddy` growable** (multi-region, exactly as `tlsf` got). A growable buddy would combine buddy's ~3× speed on general churn with growth safety and likely become the new default (accepting bounded internal fragmentation). Re-benchmark after.
3. The per-module table above is the operative guidance for the malloc→SDK-allocator migration: don't route everything through one allocator — match the allocator to the module's allocation lifetime/size pattern.

Reproduce: `docker context use k615 && make lib` in the Debian/Alpine container,
then `cc -O2 tests/blackbox/bb_alloc_bench.c build/libuniverse.a -lpthread -lm -o
abench && ./abench`. Harness: `tests/blackbox/bb_alloc_bench.c`.
