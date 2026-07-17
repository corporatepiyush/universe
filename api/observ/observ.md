# observ — async logging, striped metrics, HDR histogram

## Purpose

`universe_observ_*` are hot-path-safe observability primitives. The one hard
rule is the compute/memory/IO split: the code that *produces* a signal never
formats, heap-allocates, or blocks. Three facilities:

- **Async logging** — a logger owns a bounded lock-free MPSC ring. Many worker
  threads push a fixed 64-byte record (filled on the caller stack, no heap);
  one background drainer pops, formats integer→text, and writes. The level gate
  is a single `monotonic` atomic load + compare, so a below-threshold call is a
  true no-op. The record timestamp is a logical sequence counter, not a clock
  syscall.
- **Striped metrics** — counters partitioned into shards, each `(counter_id,
  shard)` pair on its own 128 B cache line so distinct threads incrementing
  distinct shards never share a line; `inc` is one `atomicrmw`. `sum` merges
  shards lazily (cold). Gauges are a single u64 store/load.
- **HDR-style histogram** — bucketed latency histogram giving percentiles,
  min/max/mean, and merge.

## Exported API

| C signature | Description | Return |
|---|---|---|
| `void* universe_observ_logger_create(int64_t capacity, int32_t threshold)` | Create logger with ring `capacity` records and level gate `threshold`. | handle / NULL on OOM |
| `void universe_observ_logger_destroy(void* lg)` | Free logger. | — |
| `void universe_observ_log_set_level(void* lg, int32_t threshold)` | Set the level gate. | — |
| `int universe_observ_log_enabled(void* lg, int32_t level)` | Is `level` at/above the gate? | bool |
| `int32_t universe_observ_log(void* lg, int32_t level, int32_t tid, const char* msg, int64_t argc, int64_t a0, int64_t a1, int64_t a2, int64_t a3)` | Push one record (`msg` is a static NUL string; up to 4 i64 args). | 0 OK, 1 NULL, 6 dropped/full |
| `int64_t universe_observ_log_drain(void* lg, void* writer)` | Drainer: format every pending record into buffered `writer`, collapsing consecutive identical records. | records drained |
| `int64_t universe_observ_log_pending(void* lg)` | Records currently queued. | count |
| `int64_t universe_observ_log_dropped(void* lg)` | Records dropped on full-ring pushes. | count |
| `void* universe_observ_metrics_create(int64_t ncnt, int64_t ngauge, int64_t nshards)` | Create metrics: `ncnt` counters × `nshards` shards, `ngauge` gauges. | handle / NULL |
| `void universe_observ_metrics_destroy(void* m)` | Free metrics. | — |
| `void universe_observ_counter_inc(void* m, int64_t cid, int64_t shard, int64_t delta)` | Add `delta` to counter `cid` on `shard` (one atomicrmw). | — |
| `int64_t universe_observ_counter_sum(void* m, int64_t cid)` | Sum all shards for counter `cid` (cold). | total |
| `void universe_observ_gauge_set(void* m, int64_t gid, int64_t value)` | Set gauge `gid`. | — |
| `int64_t universe_observ_gauge_get(void* m, int64_t gid)` | Read gauge `gid`. | value |
| `void* universe_observ_hist_create(int64_t min, int64_t max, int64_t sig)` | HDR histogram over `[min,max]` with `sig` significant digits. | handle / NULL |
| `void universe_observ_hist_destroy(void* h)` | Free histogram. | — |
| `void universe_observ_hist_record(void* h, int64_t value)` | Record one sample. | — |
| `int64_t universe_observ_hist_percentile(void* h, double p)` | Value at percentile `p` (0..100). | value |
| `int64_t universe_observ_hist_count(void* h)` | Total samples. | count |
| `int64_t universe_observ_hist_min(void* h)` | Minimum sample. | value |
| `int64_t universe_observ_hist_max(void* h)` | Maximum sample. | value |
| `double universe_observ_hist_mean(void* h)` | Mean sample. | value |
| `int32_t universe_observ_hist_merge(void* dst, void* src)` | Add `src` counts into `dst`. | 0 OK, else error |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the C-ABI symbols and link:

```c
#include <stdint.h>
extern void* universe_observ_hist_create(int64_t,int64_t,int64_t);
extern void  universe_observ_hist_record(void*,int64_t);
extern int64_t universe_observ_hist_percentile(void*,double);
```

```
clang -O3 yourprog.c build/libuniverse.a -lpthread -lm -o prog
```

All symbols are C ABI, `nounwind`.

## Make a CLI

A latency percentile reporter: read whitespace-separated integers from stdin,
print count/p50/p99/max.

```c
// hist_cli.c
#include <stdint.h>
#include <stdio.h>
extern void* universe_observ_hist_create(int64_t,int64_t,int64_t);
extern void  universe_observ_hist_record(void*,int64_t);
extern int64_t universe_observ_hist_percentile(void*,double);
extern int64_t universe_observ_hist_max(void*);
extern int64_t universe_observ_hist_count(void*);
extern void  universe_observ_hist_destroy(void*);

int main(void){
    void* h = universe_observ_hist_create(1, 1000000000LL, 3);
    long v;
    while (scanf("%ld", &v) == 1) universe_observ_hist_record(h, v);
    printf("n=%lld p50=%lld p99=%lld max=%lld\n",
           (long long)universe_observ_hist_count(h),
           (long long)universe_observ_hist_percentile(h, 50.0),
           (long long)universe_observ_hist_percentile(h, 99.0),
           (long long)universe_observ_hist_max(h));
    universe_observ_hist_destroy(h);
    return 0;
}
```

```
clang -O3 hist_cli.c build/libuniverse.a -lpthread -lm -o hist_cli
printf '10 20 30 40 50 500 5\n' | ./hist_cli
```

## Notes

- **Threading.** Logger producers are wait-free (XADD ticket); use one drainer
  thread calling `log_drain`. Counter `inc` is lock-free per shard — pick
  `shard = thread_id & (nshards-1)` to avoid line bouncing. `counter_sum`,
  `hist_merge`, and the drain path are cold/single-consumer.
- **Ownership.** Handles are one allocation each; destroy exactly once.
- **No clocks in the hot path.** Record timestamps are logical sequence
  numbers; wall-clock formatting, if any, belongs in the drainer/writer.
- The `writer` passed to `log_drain` is a buffered writer object (see `io`).
