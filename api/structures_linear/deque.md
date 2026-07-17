# structures_linear/deque

## Purpose

Growable double-ended queue over a power-of-two ring. Uses free-running unsigned
indices with mask wrap: `head` moves down for `push_front`, `tail` up for
`push_back`, so all four end-operations are O(1) with no modulo and no branchy
wrap. Growth doubles the ring and linearizes wrapped contents with at most two
memcpys. Occupied slots are `[head, tail)` in free-running index space. Choose
this when you need both ends; for pure FIFO use `queue`, pure LIFO use `stack`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_deque_create(int64_t elem_size, int64_t initial_cap)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `int32_t universe_ds_deque_push_back(void *d, void *elem)` | Append at back | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_deque_push_front(void *d, void *elem)` | Prepend at front | 0 OK, 1, 2, 3 |
| `int32_t universe_ds_deque_pop_front(void *d, void *out)` | Remove from front | 0 OK, 4 EMPTY |
| `int32_t universe_ds_deque_pop_back(void *d, void *out)` | Remove from back | 0 OK, 4 EMPTY |
| `int32_t universe_ds_deque_peek_front(void *d, void *out)` | Read front | 0 OK, 4 EMPTY |
| `int32_t universe_ds_deque_peek_back(void *d, void *out)` | Read back | 0 OK, 4 EMPTY |
| `int64_t universe_ds_deque_count(void *d)` | Element count | count |
| `void universe_ds_deque_destroy(void *d)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_deque_create(i64, i64)
declare i32 @universe_ds_deque_push_back(ptr, ptr)
declare i32 @universe_ds_deque_pop_front(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

`f N` pushes front, `b N` pushes back, `p` pops the front.

```c
// dqcli.c — build: clang -O3 dqcli.c build/libuniverse.a -lpthread -lm -o dqcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_deque_create(int64_t, int64_t);
extern int32_t universe_ds_deque_push_front(void *, void *);
extern int32_t universe_ds_deque_push_back(void *, void *);
extern int32_t universe_ds_deque_pop_front(void *, void *);
extern void universe_ds_deque_destroy(void *);

int main(void) {
  void *d = universe_ds_deque_create(sizeof(int), 8);
  char op[4]; int x;
  while (scanf("%3s", op) == 1) {
    if (!strcmp(op, "f")) { scanf("%d", &x); universe_ds_deque_push_front(d, &x); }
    else if (!strcmp(op, "b")) { scanf("%d", &x); universe_ds_deque_push_back(d, &x); }
    else if (!strcmp(op, "p")) { int v;
      if (universe_ds_deque_pop_front(d, &v) == 0) printf("%d\n", v); }
  }
  universe_ds_deque_destroy(d);
  return 0;
}
```

```sh
printf 'b 1 f 2 b 3 p p\n' | ./dqcli   # -> 2 then 1
```

## Notes

- Elements are POD copied by value at `elem_size` stride.
- Handle stable across growth. Single-threaded.
