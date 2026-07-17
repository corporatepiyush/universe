# structures_trees — trees, heaps & set structures

Ordered maps, priority queues, and range/connectivity structures. Node-based
maps use flat index-linked arrays (one allocation, no pointer webs).

| Module | Kind | Choose when |
|---|---|---|
| [art](art.md) | Adaptive radix tree, byte-string key -> i64 | Ordered map over variable-length byte keys, prefix scans |
| [art_sharded](art_sharded.md) | Concurrent hash-sharded ART | Many threads on byte-string keys |
| [btree](btree.md) | Array-backed B-tree, i64 -> i64 | Ordered i64 map, high fan-out, one allocation |
| [btree_sharded](btree_sharded.md) | Concurrent range-sharded B-tree | Concurrent ordered i64 map, writers spread across key space |
| [skiplist](skiplist.md) | Indexable ordered map with rank/select | Ordered map needing order-statistics |
| [fenwick](fenwick.md) | Binary indexed tree (prefix sums) | Cumulative sums with point deltas |
| [segtree](segtree.md) | Segment tree (range sum, point set) | Range aggregates with point overwrite |
| [heaps](heaps.md) | d-ary / radix-monotone / pairing heaps | Priority queues (pick by workload) |
| [unionfind](unionfind.md) | Disjoint-set union | Connectivity, MST, grouping |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`):

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

Error codes (i32): 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY,
5 NOT_FOUND, 7 INVALID_INDEX, 8 INVALID_ARG. `contains` returns 1/0; some
`unionfind` calls return values instead of codes. Sharded variants require
`-lpthread`.
