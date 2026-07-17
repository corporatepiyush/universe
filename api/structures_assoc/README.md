# structures_assoc — associative containers

Key/value maps: hash-based and ordered, single-threaded and sharded-concurrent.

| Module | Kind | Choose when |
|---|---|---|
| [hashmap_swiss](hashmap_swiss.md) | Open-addressing hash map, i64 key, SIMD tag probe | Mutable point map, read-heavy/mixed, `get` latency matters |
| [mph](mph.md) | Minimal perfect hash over static byte-string keys | Fixed key set, probe-free O(1) lookup, no mutation |
| [treemap](treemap.md) | Array-backed ordered map, i64 key -> i64 val | Ordered/nearest-key queries and range scans dominate |
| [treemap_sharded](treemap_sharded.md) | Concurrent range-sharded ordered map | Many threads on disjoint key ranges + ordered queries |

All symbols use the C ABI (`nounwind`). Build the static library with
`make lib` (`build/libuniverse.a`) and link:

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

Error codes (i32): 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY,
5 NOT_FOUND. `contains` returns 1/0.
