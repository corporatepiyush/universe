# allocator — memory allocators

Exported as `universe_alloc_*`. C ABI, `nounwind`. Link against
`build/libuniverse.a` (built by `make lib`):
`clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.

Each allocator is an opaque handle created from one `malloc` (or a caller
region). Header is ≤ 1 cache line; payload starts at +64 (single-threaded) or
+128/+384 (concurrent, so hot atomics never false-share). All size math is
overflow-checked. Error/exhaustion is a `NULL` return; introspection helpers
return `int64_t` counts.

| Module | What / when to choose | Threading |
|---|---|---|
| [arena](arena.md) | Bump allocator; many objects, shared lifetime, free-all-at-once | single-threaded |
| [pool](pool.md) | Fixed-size blocks, O(1) alloc/free, individual lifetimes | single-threaded |
| [slab](slab.md) | Fixed-size objects, growable, empty slabs returned to OS | single-threaded |

Selection guide: shared-lifetime batch / build scratch → arena; uniform fixed-size
blocks (nodes, cells) → pool (bounded) or slab (growable). These three are the
benchmark winners for their patterns (see BENCHMARKS.md). A general variable-size
allocator (TLSF/buddy/hybrid) was removed for now and will be revisited; until it
returns there is no general-purpose allocator in the SDK.
