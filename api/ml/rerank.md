# ml/rerank — RRF + MMR re-ranking kernels

## Purpose

Two pure re-ranking kernels for the retrieval/query side. **RRF** (Reciprocal
Rank Fusion) fuses N ranked id lists by order only: `score(id) += 1/(rrf_k +
rank)` with 0-based rank per list; an id appearing in several lists accumulates
(the point of hybrid lexical+semantic fusion). `rrf_k ≤ 0` or NaN selects the
standard default 60. Aggregation is a single open-addressing `i64→f32` map
(linear probing, power-of-two capacity ≤50% full) in ONE scratch calloc, then a
partial selection picks the top `out_cap` by score (ties broken by lowest id,
deterministic). **MMR** (Maximal Marginal Relevance) greedily re-ranks by a
relevance/diversity trade-off `λ·rel − (1−λ)·max_sim_to_chosen`, folding each
pick into a max-similarity cache; cosine similarity is a `<4 x float>` pass with
scalar tail. Both use one scratch calloc, freed on exit.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_rrf(const int64_t **lists_ids, const int64_t *lists_len, int64_t nlists, float rrf_k, int64_t *out_ids, float *out_scores, int64_t out_cap, int64_t *out_n)` | Fuse `nlists` ranked id lists; emit top-`out_cap` by score | 0 OK, 2 OOM |
| `int32_t universe_ml_mmr(const float *vecs, int64_t n, int64_t d, const float *rel, float lambda, int64_t k, int64_t *out_idx, int64_t *out_n)` | Greedy MMR re-rank of `n` candidate vectors | 0 OK, 2 OOM |

RRF: `lists_ids` is an array of `nlists` pointers to id arrays; `lists_len[i]` is
each list's length. `out_ids`/`out_scores` are caller arrays of ≥ `out_cap`;
`*out_n` = count emitted. MMR: `vecs` is `n×d` row-major candidate embeddings,
`rel[i]` the base relevance of candidate `i`, `lambda` ∈ `[0,1]` the
relevance/diversity balance; `out_idx` (≥ `k`) receives the selected candidate
indices in pick order, `*out_n` the count.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_ml_rrf(ptr, ptr, i64, float, ptr, ptr, i64, ptr)
declare i32 @universe_ml_mmr(ptr, i64, i64, ptr, float, i64, ptr, ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Fuse two fixed ranked id lists with RRF and print the fused top-k.

```c
// rrf.c    prints fused (id, score) pairs, best first
#include <stdint.h>
#include <stdio.h>
int32_t universe_ml_rrf(const int64_t **, const int64_t *, int64_t, float,
            int64_t *, float *, int64_t, int64_t *);

int main(void) {
    int64_t la[] = {10, 20, 30, 40};      // lexical ranking
    int64_t lb[] = {30, 10, 50, 20};      // semantic ranking
    const int64_t *lists[] = {la, lb};
    int64_t lens[] = {4, 4};

    int64_t out_ids[8], out_n = 0;
    float   out_scores[8];
    universe_ml_rrf(lists, lens, 2, /*rrf_k default*/0.0f,
                    out_ids, out_scores, 8, &out_n);
    for (int64_t i = 0; i < out_n; i++)
        printf("id=%lld score=%g\n", (long long)out_ids[i], out_scores[i]);
    return 0;
}
```

```
clang -O3 rrf.c build/libuniverse.a -lpthread -lm -o rrf
./rrf     # id 10 and 30 rank highest (they appear in both lists)
```

## Notes

- **RRF is order-only:** it fuses rankings, not raw scores — robust to
  incomparable score scales across retrievers. `rrf_k ≤ 0`/NaN ⇒ default 60.
- **MMR layout:** `vecs` is `n×d` row-major; cosine similarity computed on the
  fly; `lambda` near 1 favors relevance, near 0 favors diversity.
- **Determinism:** RRF ties break by lowest id.
- **Ownership:** caller owns all inputs/outputs; each kernel uses one internal
  scratch calloc, freed on exit.
- **Threading:** stateless per call.
