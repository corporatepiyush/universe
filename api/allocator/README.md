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
| [buddy](buddy.md) | Power-of-two blocks, O(1) coalescing, sized-free API | single-threaded |
| [tlsf](tlsf.md) | General variable-size, O(1) worst-case alloc/free, bounded fragmentation | single-threaded |

Selection guide: shared-lifetime batch → arena; uniform blocks → pool;
growing/shrinking uniform set → slab; general variable-size with real reuse →
tlsf; power-of-two with fast merge and sizes tracked at the call site → buddy.
