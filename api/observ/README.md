# observ — observability primitives

Hot-path-safe observability: the signal producer never formats, allocates, or
blocks (compute/memory/IO split). All symbols are `universe_observ_*`, C ABI,
`nounwind`. Link against `build/libuniverse.a` (`make lib`).

| Module | Purpose |
|---|---|
| [observ](observ.md) | Async MPSC logging, striped metric counters + gauges, HDR-style latency histogram. |
