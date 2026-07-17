# structures_linear/array

## Purpose

Dynamic array (vector) of arbitrary fixed-size elements. Elements live in one
contiguous run; `insert`/`remove` shift with a single `llvm.memmove` (vectorized
block move) rather than an element loop, and growth doubles via `realloc` (often
extended in place). The 32-byte handle is stable — only the data pointer moves
on growth — and all size math is overflow-checked. Layout is
`{ data, count, cap, elem }`. The general-purpose growable buffer; use `stack`
for pure LIFO, `queue`/`deque` for FIFO/double-ended.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_array_create(int64_t elem_size, int64_t initial_cap)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `int32_t universe_ds_array_push(void *a, void *elem)` | Append (copies `elem_size` bytes) | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_array_get(void *a, int64_t index, void *out)` | Copy element at `index` into `out` | 0 OK, 7 INVALID_INDEX |
| `int32_t universe_ds_array_set(void *a, int64_t index, void *elem)` | Overwrite element at `index` | 0 OK, 7 INVALID_INDEX |
| `int32_t universe_ds_array_insert(void *a, int64_t index, void *elem)` | Insert before `index` (`index <= count`) | 0 OK, 2, 3, 7 |
| `int32_t universe_ds_array_remove(void *a, int64_t index, void *out)` | Remove at `index`; `out` may be NULL | 0 OK, 7 INVALID_INDEX |
| `int64_t universe_ds_array_count(void *a)` | Element count | count |
| `int64_t universe_ds_array_capacity(void *a)` | Allocated slots | count |
| `void universe_ds_array_destroy(void *a)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_array_create(i64, i64)
declare i32 @universe_ds_array_push(ptr, ptr)
declare i32 @universe_ds_array_get(ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Read ints from stdin, store them, print them back in order.

```c
// arrcli.c — build: clang -O3 arrcli.c build/libuniverse.a -lpthread -lm -o arrcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_array_create(int64_t, int64_t);
extern int32_t universe_ds_array_push(void *, void *);
extern int32_t universe_ds_array_get(void *, int64_t, void *);
extern int64_t universe_ds_array_count(void *);
extern void universe_ds_array_destroy(void *);

int main(void) {
  void *a = universe_ds_array_create(sizeof(int), 8);
  int x;
  while (scanf("%d", &x) == 1) universe_ds_array_push(a, &x);
  int64_t n = universe_ds_array_count(a);
  for (int64_t i = 0; i < n; i++) { int v; universe_ds_array_get(a, i, &v); printf("%d\n", v); }
  universe_ds_array_destroy(a);
  return 0;
}
```

```sh
printf '3 1 2\n' | ./arrcli   # -> 3 1 2
```

## Notes

- Elements are POD, copied by value at `elem_size` stride. Handle is stable
  across growth; the data pointer may move.
- Single-threaded. `remove`/`insert` are O(n) memmoves.
