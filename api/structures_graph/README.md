# structures_graph — graphs

Flat index-linked adjacency graphs with no per-node allocation.

| Module | Kind | Choose when |
|---|---|---|
| [graph](graph.md) | Single-thread graph + BFS/DFS/Dijkstra/components | General graph work and traversal algorithms |
| [graph_sharded](graph_sharded.md) | Concurrent vertex-striped adjacency | Many threads inserting/querying edges concurrently |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`):

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

Error codes (i32): 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 7 INVALID_INDEX.
Count/id-returning functions signal errors with -1.
