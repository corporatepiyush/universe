# dataframe — universe API

A columnar DataFrame stack: typed `Series` columns and a `DataFrame` container,
with SIMD element-wise compute, reductions, selection, sort, group-by and joins.
Single-chunk, contiguous, eager, struct-of-arrays. Functional inspiration from
Polars (method NAMES kept); the low-level layout is our own. Shipped in
`libuniverse.a` (`make lib`); C ABI, `nounwind`. Exported names are
`universe_dataframe_*`.

## Modules

| Module | What it is |
| --- | --- |
| [frame](frame.md) | Foundation: `Series` + `DataFrame` construction, columns, reshape, lifetime. |
| [arith](arith.md) | SIMD element-wise arithmetic (`add`/`sub`/`mul`/`div`/`rem`, scalar, abs/neg, horizontal min/max). |
| [compare](compare.md) | SIMD predicates (`eq`/`lt`/…, scalar) + boolean algebra; builds filter masks. |
| [cast](cast.md) | SIMD dtype conversion + null-aware fill/clip/null-mask. |
| [reduce](reduce.md) | Column reductions (sum/mean/min/max/var/std/quantile/median/n_unique) + horizontal. |
| [select](select.md) | Row selection: filter/take/sample/drop_nulls/fill_null/unique/duplicated. |
| [sort](sort.md) | Permutation sort / sort_by / arg_sort / top_k / bottom_k (radix + stable merge). |
| [groupby](groupby.md) | Hash group-by + aggregation, partition_by. |
| [join](join.md) | Relational hash join (inner/left/outer/cross). |

## Shared contract (frame.ll)

- **DType enum (i32):** `I32=0, I64=1, F32=2, F64=3, BOOL=4` (one 0/1 byte per
  value), `STR=5` (i32 `offsets[len+1]` in `values`, bytes in `strdata`).
- **Series** = one malloc'd 56-byte header: `+0 i32 dtype`, `+8 i64 len`,
  `+16 i64 null_count`, `+24 ptr values`, `+32 ptr validity` (null ⇒ all valid,
  else bitmap `ceil(len/8)` bytes, bit=1 VALID), `+40 ptr strdata`,
  `+48 i64 strdata_len`.
- **DataFrame** = one malloc'd 40-byte header: `+0 i64 n_cols`, `+8 i64 cap_cols`,
  `+16 i64 height`, `+24 ptr names` (`{ptr,i64}` pairs), `+32 ptr columns`
  (Series ptr array).
- **Ownership:** the frame OWNS its column Series and name copies.
  `with_column`/`hstack`/`insert_column`/`replace_column` TAKE OWNERSHIP of the
  passed Series. New-frame producers (slice/head/tail/select/sort/filter/…) CLONE
  columns. Free frames with `universe_dataframe_free`, standalone series with
  `universe_dataframe_series_free`.
- **Return convention:** pointer-returning ops return `NULL` on failure (OOM /
  bad arg); `i32`-returning ops use the universe error codes (0 OK, 1 NULL_PTR,
  2 OOM, 8 INVALID_ARG, …) and write results through out-pointers.

Link line: `clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.
