# structures_linear/stack

## Purpose

Growable array stack of arbitrary fixed-size elements. Elements are contiguous
(the top of stack is the hottest, always-cached line), growth doubles via
`realloc` (often extended in place, avoiding a copy), and all size math is
overflow-checked. The 32-byte handle is stable — only the data pointer moves.
Layout `{ data, count, cap, elem }`. The idiomatic LIFO; for FIFO use `queue`,
for random access use `array`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_stack_create(int64_t elem_size, int64_t initial_cap)` | Create for `elem_size`-byte elements | handle, or NULL on OOM |
| `int32_t universe_ds_stack_push(void *st, void *elem)` | Push | 0 OK, 1 NULL, 2 OOM, 3 SIZE_OVERFLOW |
| `int32_t universe_ds_stack_pop(void *st, void *out)` | Pop the top | 0 OK, 4 EMPTY |
| `int32_t universe_ds_stack_peek(void *st, void *out)` | Read the top without removal | 0 OK, 4 EMPTY |
| `int64_t universe_ds_stack_count(void *st)` | Element count | count |
| `int64_t universe_ds_stack_capacity(void *st)` | Allocated slots | count |
| `void universe_ds_stack_destroy(void *st)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_stack_create(i64, i64)
declare i32 @universe_ds_stack_push(ptr, ptr)
declare i32 @universe_ds_stack_pop(ptr, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Reverse stdin ints (LIFO).

```c
// stcli.c — build: clang -O3 stcli.c build/libuniverse.a -lpthread -lm -o stcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_ds_stack_create(int64_t, int64_t);
extern int32_t universe_ds_stack_push(void *, void *);
extern int32_t universe_ds_stack_pop(void *, void *);
extern void universe_ds_stack_destroy(void *);

int main(void) {
  void *st = universe_ds_stack_create(sizeof(int), 8);
  int x;
  while (scanf("%d", &x) == 1) universe_ds_stack_push(st, &x);
  int v;
  while (universe_ds_stack_pop(st, &v) == 0) printf("%d\n", v);
  universe_ds_stack_destroy(st);
  return 0;
}
```

```sh
printf '1 2 3\n' | ./stcli   # -> 3 2 1
```

## Notes

- Elements are POD copied by value at `elem_size` stride; handle stable across
  growth.
- Single-threaded.
