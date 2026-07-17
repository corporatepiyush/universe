# structures_linear — linear & sequence containers

Contiguous and node-pooled sequences, sets of small integers, and rings. All
avoid per-element allocation.

| Module | Kind | Choose when |
|---|---|---|
| [array](array.md) | Dynamic array (vector) | Random access, general growable buffer |
| [stack](stack.md) | Growable LIFO | Pure push/pop |
| [queue](queue.md) | Growable FIFO ring | Pure enqueue/dequeue |
| [deque](deque.md) | Double-ended ring | Push/pop at both ends |
| [ringbuf](ringbuf.md) | Fixed-capacity ring (single-thread) | Bounded buffer with backpressure |
| [ringbuf_concurrent](ringbuf_concurrent.md) | Lock-free SPSC ring | One producer + one consumer thread |
| [binheap](binheap.md) | Binary min-heap over i64 | Simple priority queue (bare keys) |
| [dlist](dlist.md) | Doubly-linked list, pooled nodes | O(1) splice/remove by node handle |
| [slist](slist.md) | Singly-linked list, pooled nodes | Forward-only list, minimal overhead |
| [lru](lru.md) | LRU cache, i64 key + fixed value | Bounded cache with eviction |
| [bitset](bitset.md) | Packed bit set + set algebra | Dense bit flags, word-parallel ops |
| [sparse_set](sparse_set.md) | Dense/sparse integer set | O(1) membership + cache-dense iteration (ECS, visited sets) |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`):

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

Error codes (i32): 0 OK, 1 NULL_PTR, 2 OOM, 3 SIZE_OVERFLOW, 4 EMPTY,
5 NOT_FOUND, 6 FULL, 7 INVALID_INDEX. `contains` returns 1/0. `ringbuf_concurrent`
requires `-lpthread` and an SPSC usage pattern.
