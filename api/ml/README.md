# ml — machine-learning kernels, models, and vector indexes

SIMD-first ML substrate over row-major f32 data: numeric kernels, classic
models, ANN vector indexes, quantization, and re-ranking. C-ABI `nounwind`; link
against `build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [kernels](kernels.md) | BLAS-1/2 SIMD primitives (dot/sum/norm/dist/axpy/gemv/argmin/max) + `_scalar` twins |
| [cluster](cluster.md) | Lloyd's k-means fit/predict |
| [neighbors](neighbors.md) | Exact brute-force kNN classify/regress (the ANN oracle) |
| [linear](linear.md) | Linear + logistic regression by batch gradient descent |
| [pca](pca.md) | PCA via covariance power-iteration + Hotelling deflation |
| [hnsw](hnsw.md) | HNSW approximate nearest-neighbor graph index |
| [ivf](ivf.md) | IVF-FLAT inverted-file vector index |
| [quant](quant.md) | Vector-quantization codec (f32/f16/int8/binary) + int8/Hamming distances |
| [rerank](rerank.md) | RRF rank fusion + MMR relevance/diversity re-rank |

Conventions across the domain: f32, row-major, unit stride; vectors passed as
raw pointers with explicit dimensions; caller owns all I/O buffers; error codes
are the SDK i32 set (0 OK, 2 OOM, 8 INVALID_ARG, 12 NOT_CONVERGED, …). Vector
metrics use **0 = cosine, 1 = L2** ([hnsw](hnsw.md), [ivf](ivf.md)). Every hot
kernel is SIMD-first (`<4 x float>`, SSE2/NEON baseline) with a `_scalar` twin
that is both the sub-4 tail and the cross-check oracle. Single-threaded
(concurrency deferred).
