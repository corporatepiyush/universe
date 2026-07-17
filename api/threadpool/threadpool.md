# threadpool/threadpool

## Purpose

Fixed worker thread pool with a bounded inline task ring. Instead of mallocing a
node per task and chasing a linked list, tasks are `{fn, arg}` pairs stored
inline in a power-of-two ring: submit is a store plus index increment under the
lock, and dispatch walks memory sequentially — zero per-task allocation. The
bounded ring gives natural backpressure (submit blocks on a full ring rather
than growing unbounded). Everything lives in one allocation (header, ring,
pthread_t array) with one mutex and three condvars (not_empty / not_full /
idle). `wait` blocks until the queue is empty and no tasks are in flight (true
drain); `destroy` drains pending tasks, then joins and frees.

## Exported API

Task function signature: `void fn(void *arg)`.

| C signature | Description | Returns |
|---|---|---|
| `void *universe_threadpool_create(int64_t nthreads, int64_t queue_cap)` | Start `nthreads` workers with a `queue_cap`-slot ring (rounded to pow2) | handle, or NULL on OOM |
| `int32_t universe_threadpool_submit(void *tp, void *fn, void *arg)` | Enqueue `fn(arg)`; blocks if the ring is full | 0 OK, or error code |
| `int32_t universe_threadpool_wait(void *tp)` | Block until all submitted tasks have completed | 0 OK |
| `int32_t universe_threadpool_destroy(void *tp)` | Drain, join workers, free | 0 OK |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_threadpool_create(i64, i64)
declare i32 @universe_threadpool_submit(ptr, ptr, ptr)
declare i32 @universe_threadpool_wait(ptr)
declare i32 @universe_threadpool_destroy(ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A parallel-map: square each element of an array across worker threads.

```c
// tpcli.c — build: clang -O3 tpcli.c build/libuniverse.a -lpthread -lm -o tpcli
#include <stdint.h>
#include <stdio.h>
extern void *universe_threadpool_create(int64_t, int64_t);
extern int32_t universe_threadpool_submit(void *, void *, void *);
extern int32_t universe_threadpool_wait(void *);
extern int32_t universe_threadpool_destroy(void *);

#define N 16
static long data[N];
static void square(void *arg) { long *p = arg; *p = (*p) * (*p); }

int main(void) {
  for (int i = 0; i < N; i++) data[i] = i;
  void *tp = universe_threadpool_create(4, 64);
  for (int i = 0; i < N; i++) universe_threadpool_submit(tp, (void *)square, &data[i]);
  universe_threadpool_wait(tp);
  universe_threadpool_destroy(tp);
  for (int i = 0; i < N; i++) printf("%ld ", data[i]);
  printf("\n");
  return 0;
}
```

```sh
./tpcli   # 0 1 4 9 16 25 ... 225
```

## Notes

- Thread-safe: `submit` may be called from any thread; it blocks when the ring
  is full (bounded backpressure).
- Task functions run on worker threads — synchronize any shared writes yourself;
  in the example each task owns a disjoint slot so no locking is needed.
- `wait` is a true drain (queue empty and in-flight == 0). Always `destroy` to
  join workers and free. Link with `-lpthread`.
