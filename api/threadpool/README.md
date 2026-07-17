# threadpool — worker thread pool

Fixed worker pool with a bounded inline task ring (zero per-task allocation).

| Module | Kind | Choose when |
|---|---|---|
| [threadpool](threadpool.md) | Fixed-size pool, `void(*)(void*)` tasks | Fan out independent work items across N worker threads |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`); requires `-lpthread`:

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```
