# structures_probabilistic — approximate membership filters

Space-efficient set membership with bounded false positives and no false
negatives.

| Module | Variants | Choose when |
|---|---|---|
| [filters](filters.md) | cuckoo (deletable), XOR (static), blocked Bloom (add-only) | Approximate membership on i64 keys; pick the variant by mutability needs |

Build the static library with `make lib` (`build/libuniverse.a`) and link
(C ABI, `nounwind`):

```sh
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

`contains` returns 1 (maybe present) / 0 (definitely absent). Error codes:
0 OK, 5 NOT_FOUND, 6 FULL.
