# structures_probabilistic/filters

## Purpose

Three probabilistic membership filters — cuckoo, XOR, and blocked (cache-line)
Bloom — over i64 keys. All answer "is key x in the set?" with no false negatives
and a bounded false-positive rate, using far less space than an exact set, and
share one splitmix64 mixer for hash positions. **Choose by workload:**

- **Cuckoo** (`cuckoo_*`) — dynamic: supports add, contains, and **delete**.
  Buckets of four 8-bit fingerprints with partial-key cuckoo hashing (SWAR
  contains, one i32 load per bucket). Pick this when the set changes over time
  and you need deletion. On FULL the last evicted victim is dropped — stop
  inserting on FULL.
- **XOR** (`xor_*`) — static: build once from a set of distinct keys, then
  contains only. ~1.23 bytes/key, contains is 3 byte loads + xor + compare.
  Pick this for an immutable set where space and query speed matter most.
- **Blocked Bloom** (`bbloom_*`) — dynamic add + contains (no delete). All hash
  probes land in one cache-line block per key, so a query touches a single line.
  Pick this for a growing set with a tunable bits-per-key tradeoff.

## Exported API

### Cuckoo (dynamic, deletable)

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_cuckoo_create(int64_t hint)` | Create sized for ~`hint` keys | handle, or NULL on OOM |
| `int32_t universe_ds_cuckoo_add(void *f, int64_t key)` | Insert `key` | 0 OK, 6 FULL |
| `int32_t universe_ds_cuckoo_contains(void *f, int64_t key)` | Membership | 1 maybe-present, 0 absent |
| `int32_t universe_ds_cuckoo_delete(void *f, int64_t key)` | Remove one matching fingerprint | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_cuckoo_count(void *f)` | Inserted count | count |
| `int64_t universe_ds_cuckoo_capacity(void *f)` | Slot capacity | count |
| `void universe_ds_cuckoo_destroy(void *f)` | Free | — |

### XOR (static)

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_xor_build(void *keys, int64_t n)` | Build from `n` distinct i64 keys (`keys` is `int64_t*`) | handle, or NULL on failure |
| `int32_t universe_ds_xor_contains(void *f, int64_t key)` | Membership | 1 maybe-present, 0 absent |
| `void universe_ds_xor_destroy(void *f)` | Free | — |

### Blocked Bloom (dynamic, add-only)

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_bbloom_create(int64_t nkeys, int64_t bpk)` | Create sized for `nkeys` keys at `bpk` bits per key | handle, or NULL on OOM |
| `int32_t universe_ds_bbloom_add(void *f, int64_t key)` | Insert `key` | 0 OK, 1 NULL |
| `int32_t universe_ds_bbloom_contains(void *f, int64_t key)` | Membership | 1 maybe-present, 0 absent |
| `void universe_ds_bbloom_destroy(void *f)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_bbloom_create(i64, i64)
declare i32 @universe_ds_bbloom_add(ptr, i64)
declare i32 @universe_ds_bbloom_contains(ptr, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A membership filter: `add K` inserts, `q K` queries (blocked Bloom).

```c
// bloomcli.c — build: clang -O3 bloomcli.c build/libuniverse.a -lpthread -lm -o bloomcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_bbloom_create(int64_t, int64_t);
extern int32_t universe_ds_bbloom_add(void *, int64_t);
extern int32_t universe_ds_bbloom_contains(void *, int64_t);
extern void universe_ds_bbloom_destroy(void *);

int main(void) {
  void *f = universe_ds_bbloom_create(10000, 10 /* bits/key */);
  char op[4]; long long k;
  while (scanf("%3s %lld", op, &k) == 2) {
    if (!strcmp(op, "add")) universe_ds_bbloom_add(f, k);
    else if (!strcmp(op, "q"))
      printf("%lld -> %s\n", k, universe_ds_bbloom_contains(f, k) ? "maybe" : "no");
  }
  universe_ds_bbloom_destroy(f);
  return 0;
}
```

```sh
printf 'add 42\nq 42\nq 7\n' | ./bloomcli   # 42 -> maybe, 7 -> no
```

## Notes

- All filters can report false positives (`contains` == 1 for an absent key) but
  never false negatives for inserted keys.
- Cuckoo: dropping a victim on FULL or deleting a fingerprint shared by another
  key can make a previously-present key test absent — stop on FULL, delete only
  keys known present. XOR keys must be distinct and the set is fixed at build.
- Single-threaded. XOR `build` takes an `int64_t*` array of keys.
