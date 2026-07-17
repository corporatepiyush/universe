# structures_trees/art

## Purpose

Adaptive radix tree (ART) mapping byte-string keys to i64 values, single
thread. It consumes one key byte per level; each inner node adapts its kind to
the number of distinct next-bytes (Node4/16/48/256) so it never wastes 256 slots
for a few children, and uses pessimistic path compression. Node16 uses a SIMD
lookup. Leaves store the full key (in a shared arena) plus the i64 value.
**Choose ART** for an ordered map over variable-length byte keys with prefix
scans and no hashing/collisions — string keys, integer keys encoded big-endian,
or IP prefixes. For i64-only ordered maps a `btree`/`skiplist` may be simpler;
for concurrent use see `art_sharded`.

## Exported API

Callback signature: `int32_t cb(void *ctx, void *key, int64_t klen, int64_t val)`
— return nonzero to stop iteration.

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_art_create(void)` | Create an empty tree | handle, or NULL on OOM |
| `int32_t universe_ds_art_insert(void *t, void *key, int64_t klen, int64_t val)` | Insert/overwrite `key` | 0 OK, 1 NULL, 2 OOM, 8 INVALID_ARG |
| `int32_t universe_ds_art_get(void *t, void *key, int64_t klen, int64_t *out)` | Lookup value | 0 OK, 5 NOT_FOUND |
| `int32_t universe_ds_art_contains(void *t, void *key, int64_t klen)` | Membership | 1 present, 0 absent |
| `int32_t universe_ds_art_delete(void *t, void *key, int64_t klen)` | Remove `key` | 0 OK, 5 NOT_FOUND |
| `int64_t universe_ds_art_count(void *t)` | Key count | count |
| `int64_t universe_ds_art_size(void *t)` | Alias of count | count |
| `int64_t universe_ds_art_iterate(void *t, void *cb, void *ctx)` | In-order (lexicographic) visit | keys visited |
| `int64_t universe_ds_art_prefix_scan(void *t, void *pfx, int64_t plen, void *cb, void *ctx)` | Visit keys with prefix `pfx` in order | keys visited |
| `int32_t universe_ds_art_min(void *t, void **outkey, int64_t *outlen, int64_t *outval)` | Smallest key | 0 OK, 4 EMPTY |
| `int32_t universe_ds_art_max(void *t, void **outkey, int64_t *outlen, int64_t *outval)` | Largest key | 0 OK, 4 EMPTY |
| `void universe_ds_art_destroy(void *t)` | Free | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_art_create()
declare i32 @universe_ds_art_insert(ptr, ptr, i64, i64)
declare i32 @universe_ds_art_get(ptr, ptr, i64, ptr)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

A string dictionary: `put WORD N`, `get WORD`.

```c
// artcli.c — build: clang -O3 artcli.c build/libuniverse.a -lpthread -lm -o artcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_art_create(void);
extern int32_t universe_ds_art_insert(void *, void *, int64_t, int64_t);
extern int32_t universe_ds_art_get(void *, void *, int64_t, int64_t *);
extern void universe_ds_art_destroy(void *);

int main(void) {
  void *t = universe_ds_art_create();
  char op[8], w[64]; long long v;
  while (scanf("%7s %63s", op, w) == 2) {
    if (!strcmp(op, "put")) { scanf("%lld", &v);
      universe_ds_art_insert(t, w, strlen(w), v); }
    else if (!strcmp(op, "get")) { int64_t out;
      if (universe_ds_art_get(t, w, strlen(w), &out) == 0) printf("%lld\n", (long long)out);
      else printf("(none)\n"); }
  }
  universe_ds_art_destroy(t);
  return 0;
}
```

```sh
printf 'put apple 1\nput apricot 2\nget apple\nget banana\n' | ./artcli  # 1 then (none)
```

## Notes

- Keys are arbitrary byte strings of length `klen`; the tree copies key bytes
  into its own arena. `min`/`max` return a pointer into that arena (valid until
  mutation).
- Iteration is lexicographic by raw bytes; encode integer keys big-endian for
  numeric order.
- Single-threaded. Use `art_sharded` for concurrency.
