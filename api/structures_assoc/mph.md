# structures_assoc/mph

## Purpose

Minimal perfect hash over a **static** set of byte-string keys. Given N distinct
keys at build time it maps each key to a distinct slot in the dense range
`[0, N)` with zero collisions and probe-free O(1) lookup — a single string hash,
one displacement-table load, a mix, and one modulo. It is a
compress-hash-displace / displacement-array MPH: level 1 buckets keys, level 2
stores a per-bucket displacement that resolves each bucket's keys into distinct
free slots. **Choose this** for a fixed, known key set (HTTP header names,
opcode/keyword tables) where lookups dominate and no insert/delete is needed —
it does strictly fewer memory touches than any open-addressing `get`. It is
read-only after build; for a mutable map use `hashmap_swiss`.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `void *universe_ds_mph_build(void **keys, int64_t *keylens, int64_t n)` | Build an MPH over `n` distinct keys (`keys[i]` a byte pointer, `keylens[i]` its length) | handle, or NULL on failure/OOM |
| `int64_t universe_ds_mph_lookup(void *m, void *key, int64_t len)` | Slot in `[0,N)` for `key` (assumes membership; garbage for absent keys) | slot index |
| `int64_t universe_ds_mph_lookup_checked(void *m, void *key, int64_t len)` | Verified lookup; `-1` if `key` was not in the build set | slot, or -1 |
| `int64_t universe_ds_mph_slot_count(void *m)` | Number of slots N | count |
| `void universe_ds_mph_destroy(void *m)` | Free the table | — |

## Use in an LLVM-based environment

```llvm
declare ptr @universe_ds_mph_build(ptr, ptr, i64)
declare i64 @universe_ds_mph_lookup_checked(ptr, ptr, i64)
```

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Build an MPH over a fixed keyword set, then classify stdin words.

```c
// mphcli.c — build: clang -O3 mphcli.c build/libuniverse.a -lpthread -lm -o mphcli
#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern void *universe_ds_mph_build(void **, int64_t *, int64_t);
extern int64_t universe_ds_mph_lookup_checked(void *, void *, int64_t);
extern void universe_ds_mph_destroy(void *);

int main(void) {
  const char *kw[] = {"if", "else", "while", "return"};
  void *keys[4]; int64_t lens[4];
  for (int i = 0; i < 4; i++) { keys[i] = (void *)kw[i]; lens[i] = strlen(kw[i]); }
  void *m = universe_ds_mph_build(keys, lens, 4);
  char w[64];
  while (scanf("%63s", w) == 1) {
    int64_t slot = universe_ds_mph_lookup_checked(m, w, strlen(w));
    if (slot >= 0) printf("%s -> keyword #%lld\n", w, (long long)slot);
    else printf("%s -> not a keyword\n", w);
  }
  universe_ds_mph_destroy(m);
  return 0;
}
```

```sh
printf 'while foo return\n' | ./mphcli
```

## Notes

- Keys must be **distinct**. Duplicate keys make construction fail (NULL).
- The table stores its own hash structure, not the key bytes; use
  `lookup_checked` when inputs may be outside the build set — plain `lookup`
  returns an arbitrary slot for non-members.
- Single-threaded build; lookups are pure reads and safe to share across threads
  once built.
