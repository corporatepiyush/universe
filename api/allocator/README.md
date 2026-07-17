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
| [arena](arena.md) | Bump allocator; a burst of temp allocations, freed all at once | single-threaded |

STATUS: `arena` is currently the ONLY SDK allocator, and it is OPTIONAL — the
project uses native `malloc`/`realloc`/`free` directly for now. Reach for `arena`
inside a module only when a phase produces many fast-building temporary
allocations that all die together (build/parse/decode scratch; bump + one
`reset`/`destroy`). `pool`, `slab`, and the general-purpose `tlsf`/`buddy`/`hybrid`
were removed; the SDK-allocator-everywhere migration is deferred until a growable
general-purpose allocator is (re)introduced (see BENCHMARKS.md).
