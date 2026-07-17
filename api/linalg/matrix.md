# linalg/matrix — dense linear algebra (BLAS core)

Source: `src/linalg/matrix.ll` · Symbols: `universe_linalg_*` · C ABI, `nounwind`.

## Purpose

Dense, row-major, contiguous **`f64`** linear-algebra primitives — the BLAS
core of the `linalg` domain. Method names follow Julia's `LinearAlgebra`
stdlib (functional inspiration only; the layout and kernels are designed here
from first principles). The centrepiece `universe_linalg_matmul` is a
**register-blocked SIMD GEMM**: a 4×4 tile of C is accumulated in 8 independent
`<2 x double>` vector registers, the K-loop broadcasts A elements and issues
outer-product `fma` updates that reuse two B vectors across the four rows. This
lowers to packed `fmla.2d` (AArch64) / `vfmadd*pd` (x86) with a spill-free
inner loop (~8× the scalar oracle: 27 vs 3.4 GFLOP/s on an M1, 128³). Rows and
columns outside the 4-multiple tiling are finished by a scalar edge path, which
is also the SIMD-first fallback and cross-check oracle.

Every hot kernel (matmul / matvec / dot / norm) ships a `_scalar` twin that is
both the reference the vector path is validated against (< 1e-9 rel on random
input) and the portable fallback.

**Convention.** A matrix `A` with `m` rows and `n` cols is row-major, so
element `(i,j)` is at `A[i*n + j]` (leading dimension `lda = n`). The caller
owns all storage; every op writes into a caller-provided `out`/`C` buffer and
**never allocates**. Buffers are plain `double*` (8-byte aligned suffices).

## Exported API

All `i32` returns use: `0` OK, `1` NULL_PTR, `3` SIZE_OVERFLOW, `8` INVALID_ARG
(shape mismatch, or non-normalizable zero vector). Value-returning functions
(`dot`/`norm`/`tr`) return the number directly and yield `0.0` for a 0-length
input; the caller owns pointer validity for those.

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_linalg_matmul(const double* A, int64_t m, int64_t k, const double* B, int64_t k2, int64_t n, double* C)` | `C = A·B` (A is m×k, B is k×n, C is m×n) — SIMD tiled GEMM | i32 err |
| `int32_t universe_linalg_matmul_scalar(const double* A, int64_t m, int64_t k, const double* B, int64_t k2, int64_t n, double* C)` | Triple-loop GEMM oracle (reference/fallback) | i32 err |
| `int32_t universe_linalg_matvec(const double* A, int64_t m, int64_t n, const double* x, double* y)` | `y = A·x` (A is m×n, x len n, y len m) — SIMD gemv | i32 err |
| `int32_t universe_linalg_matvec_scalar(const double* A, int64_t m, int64_t n, const double* x, double* y)` | gemv oracle | i32 err |
| `int32_t universe_linalg_mul_scalar(const double* A, int64_t n, double s, double* out)` | `out = s·A` over `n` elements | i32 err |
| `int32_t universe_linalg_add(const double* A, const double* B, int64_t n, double* out)` | `out = A + B` over `n` elements | i32 err |
| `int32_t universe_linalg_sub(const double* A, const double* B, int64_t n, double* out)` | `out = A − B` over `n` elements | i32 err |
| `int32_t universe_linalg_transpose(const double* A, int64_t m, int64_t n, double* out)` | `out` (n×m) `= Aᵀ` | i32 err |
| `double universe_linalg_dot(const double* x, const double* y, int64_t n)` | `Σ x[i]·y[i]` — SIMD 4-accumulator reduction | double |
| `double universe_linalg_dot_scalar(const double* x, const double* y, int64_t n)` | dot oracle | double |
| `double universe_linalg_norm(const double* x, int64_t n, int32_t p)` | Vector p-norm: `p=2` Euclid, `p=1` sum\|·\|, `p=0` Inf (max\|·\|) | double |
| `double universe_linalg_norm_scalar(const double* x, int64_t n, int32_t p)` | norm oracle | double |
| `double universe_linalg_norm_fro(const double* A, int64_t n)` | Frobenius norm `= sqrt(dot(A,A))` over `n` elements | double |
| `double universe_linalg_norm_fro_scalar(const double* A, int64_t n)` | Frobenius oracle | double |
| `int32_t universe_linalg_normalize(const double* x, int64_t n, double* out)` | `out = x / ‖x‖₂`; returns `8` if `‖x‖₂ == 0` | i32 err |
| `int32_t universe_linalg_cross(const double* a, const double* b, double* out)` | 3-vector cross product `a × b` (all length 3) | i32 err |
| `double universe_linalg_tr(const double* A, int64_t n)` | Trace of an n×n matrix (`Σ A[i,i]`) | double |
| `int32_t universe_linalg_diag(const double* A, int64_t n, double* out)` | Extract diagonal: `out[i] = A[i,i]` (A is n×n, out len n) | i32 err |
| `int32_t universe_linalg_diagm(const double* v, int64_t n, double* out)` | Build diagonal matrix `out` (n×n) from vector `v` (len n) | i32 err |
| `int32_t universe_linalg_identity(double* out, int64_t n)` | `out = Iₙ` (n×n) | i32 err |
| `int32_t universe_linalg_kron(const double* A, int64_t am, int64_t an, const double* B, int64_t bm, int64_t bn, double* out)` | Kronecker product `A ⊗ B` → out is (am·bm)×(an·bn) | i32 err |

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a` (and `make dylib` the shared object).
Declare the symbols you call and link the archive:

```llvm
; yourprog.ll — C ABI, nounwind
declare i32 @universe_linalg_matmul(ptr, i64, i64, ptr, i64, i64, ptr)
declare double @universe_linalg_norm_fro(ptr, i64)
; ... call them on your own double buffers ...
```

Link line (host build; `-lm` for libm, `-lpthread` per the SDK convention):

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

From C, declare the prototypes from the table above (or a shared header) and
link the same archive.

## Make a CLI

A small C driver that multiplies two matrices whose dimensions and row-major
elements come from `argv`. Usage: `matmul m k n  <m*k A values> <k*n B values>`.

```c
/* matmul_cli.c — clang matmul_cli.c build/libuniverse.a -lpthread -lm -o matmul_cli */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int32_t universe_linalg_matmul(const double*, int64_t, int64_t,
                                       const double*, int64_t, int64_t, double*);

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s m k n <A...> <B...>\n", argv[0]); return 2; }
    int64_t m = atoll(argv[1]), k = atoll(argv[2]), n = atoll(argv[3]);
    int64_t na = m*k, nb = k*n;
    if (argc != 4 + na + nb) { fprintf(stderr, "need %lld A + %lld B values\n",
                                       (long long)na, (long long)nb); return 2; }
    double *A = malloc(na*sizeof(double)), *B = malloc(nb*sizeof(double));
    double *C = malloc((size_t)m*n*sizeof(double));
    for (int64_t i = 0; i < na; i++) A[i] = atof(argv[4 + i]);
    for (int64_t i = 0; i < nb; i++) B[i] = atof(argv[4 + na + i]);
    int32_t rc = universe_linalg_matmul(A, m, k, B, k, n, C);
    if (rc) { fprintf(stderr, "matmul error %d\n", rc); return 1; }
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) printf("%s%g", j ? " " : "", C[i*n + j]);
        putchar('\n');
    }
    free(A); free(B); free(C); return 0;
}
```

```
$ clang matmul_cli.c build/libuniverse.a -lpthread -lm -o matmul_cli
$ ./matmul_cli 2 3 2  1 2 3 4 5 6  7 8 9 10 11 12
58 64
139 154
```

## Notes

- **dtype/layout:** dense, row-major, contiguous `f64` only. Element `(i,j)` at
  `A[i*n + j]`. There is no leading-dimension/stride parameter — pack tightly.
- **Ownership:** the library never allocates; you own every input and output
  buffer and must size `out`/`C` correctly (matmul C is `m*n`, transpose out is
  `m*n`, kron out is `(am*bm)*(an*bn)`, identity/diagm out is `n*n`).
- **Aliasing:** `out`/`C` must not overlap the inputs (kernels assume `noalias`).
- **Thread-safety:** all functions are pure w.r.t. their argument buffers
  (`memory(argmem: …)`, `nounwind`), so concurrent calls on disjoint buffers are
  safe; no internal shared state.
- **Numerics:** the SIMD reductions use reassociation/FMA contraction, so
  results match the `_scalar` oracles to within relative tolerance (~1e-9), not
  bit-for-bit. GEMM uses fused multiply-add (more accurate than the separate-
  rounding oracle).
- **Edge dims:** any `m`/`n`/`k` works; dimensions not divisible by 4 use the
  scalar edge path. `k == 0` yields a zero `C`; `m == 0` or `n == 0` writes
  nothing.
