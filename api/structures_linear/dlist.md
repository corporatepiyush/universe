# structures_linear/dlist

## Purpose

Doubly-linked list with a chunked node pool and zero-copy data access. Nodes are
carved from 256-node chunks (one malloc per 256 inserts) via a wilderness
cursor; removed nodes go on an intrusive LIFO free list and are reused
cache-hot, so there is no per-node malloc. `universe_ds_dlist_data(node)` exposes
the element in place for copy-free iteration. Node layout is
`{ prev, next, data }`. Choose this when you need O(1) splice/remove at arbitrary
positions holding a node handle; for forward-only lists use `slist`, for
contiguous storage use `array`/`deque`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_dlist_create(int64_t elem_size)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `void *universe_ds_dlist_push_front(void *l, void *elem)` | Prepend; returns the new node | node, or NULL on OOM |
| `void *universe_ds_dlist_push_back(void *l, void *elem)` | Append; returns the new node | node, or NULL on OOM |
| `void *universe_ds_dlist_insert_after(void *l, void *node, void *elem)` | Insert after `node` | node, or NULL on OOM |
| `int32_t universe_ds_dlist_remove(void *l, void *node, void *out)` | Unlink `node`; `out` may be NULL | 0 OK, or error |
| `void *universe_ds_dlist_first(void *l)` | First node | node, or NULL if empty |
| `void *universe_ds_dlist_last(void *l)` | Last node | node, or NULL if empty |
| `void *universe_ds_dlist_next(void *node)` | Successor node | node, or NULL at end |
| `void *universe_ds_dlist_prev(void *node)` | Predecessor node | node, or NULL at start |
| `void *universe_ds_dlist_data(void *node)` | In-place pointer to the node's element | element pointer |
| `int64_t universe_ds_dlist_count(void *l)` | Element count | count |
| `void universe_ds_dlist_destroy(void *l)` | Free all chunks | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_dlist_create(i64)
declare ptr @universe_ds_dlist_push_back(ptr, ptr)
declare ptr @universe_ds_dlist_data(ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read ints, append each, then walk the list forward printing in-place data.

```c
// dlcli.c — build: clang -O3 dlcli.c build/libuniverse.a -lpthread -lm -o dlcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_dlist_create(int64_t);
extern void *universe_ds_dlist_push_back(void *, void *);
extern void *universe_ds_dlist_first(void *);
extern void *universe_ds_dlist_next(void *);
extern void *universe_ds_dlist_data(void *);
extern void universe_ds_dlist_destroy(void *);

int main(void) {
  void *l = universe_ds_dlist_create(sizeof(int));
  int x;
  while (scanf("%d", &x) == 1) universe_ds_dlist_push_back(l, &x);
  for (void *n = universe_ds_dlist_first(l); n; n = universe_ds_dlist_next(n))
    printf("%d\n", *(int *)universe_ds_dlist_data(n));
  universe_ds_dlist_destroy(l);
  return 0;
}
```

```sh
printf '10 20 30\n' | ./dlcli   # -> 10 20 30
```

## Notes

- `data(node)` returns a pointer into the node's storage — valid until the node
  is removed or the list destroyed. Do not free node pointers yourself.
- Node handles from push/insert stay stable across other insertions.
- Single-threaded.
