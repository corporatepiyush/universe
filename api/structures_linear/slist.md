# structures_linear/slist

## Purpose

Singly-linked list with the same chunked node pool as `dlist` (one malloc per
256 nodes, intrusive free list, wilderness cursor) but half the per-node
overhead: node layout `{ next, data }`. It exposes idiomatic forward-list ops
only — push_front, push_back (O(1) via a cached tail pointer), pop_front,
insert_after, remove_after — and zero-copy in-place `data(node)`. There is no
O(n) arbitrary remove; use `dlist` when you need backward links or remove by
node handle.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_slist_create(int64_t elem_size)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `void *universe_ds_slist_push_front(void *l, void *elem)` | Prepend; returns new node | node, or NULL on OOM |
| `void *universe_ds_slist_push_back(void *l, void *elem)` | Append; returns new node | node, or NULL on OOM |
| `void *universe_ds_slist_insert_after(void *l, void *node, void *elem)` | Insert after `node` | node, or NULL on OOM |
| `int32_t universe_ds_slist_pop_front(void *l, void *out)` | Remove head | 0 OK, 4 EMPTY |
| `int32_t universe_ds_slist_remove_after(void *l, void *node, void *out)` | Remove `node`'s successor | 0 OK, 5 NOT_FOUND |
| `void *universe_ds_slist_first(void *l)` | Head node | node, or NULL if empty |
| `void *universe_ds_slist_next(void *node)` | Successor node | node, or NULL at end |
| `void *universe_ds_slist_data(void *node)` | In-place pointer to element | element pointer |
| `int64_t universe_ds_slist_count(void *l)` | Element count | count |
| `void universe_ds_slist_destroy(void *l)` | Free all chunks | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_slist_create(i64)
declare ptr @universe_ds_slist_push_back(ptr, ptr)
declare ptr @universe_ds_slist_data(ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Append ints, then walk the list printing each.

```c
// slcli.c — build: clang -O3 slcli.c build/libuniverse.a -lpthread -lm -o slcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_slist_create(int64_t);
extern void *universe_ds_slist_push_back(void *, void *);
extern void *universe_ds_slist_first(void *);
extern void *universe_ds_slist_next(void *);
extern void *universe_ds_slist_data(void *);
extern void universe_ds_slist_destroy(void *);

int main(void) {
  void *l = universe_ds_slist_create(sizeof(int));
  int x;
  while (scanf("%d", &x) == 1) universe_ds_slist_push_back(l, &x);
  for (void *n = universe_ds_slist_first(l); n; n = universe_ds_slist_next(n))
    printf("%d\n", *(int *)universe_ds_slist_data(n));
  universe_ds_slist_destroy(l);
  return 0;
}
```

```sh
printf '7 8 9\n' | ./slcli   # -> 7 8 9
```

## Notes

- Forward-only; no prev links, no O(n) arbitrary remove.
- `data(node)` points into node storage (valid until removed/destroyed).
- Single-threaded.
