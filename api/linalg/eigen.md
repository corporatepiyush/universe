# linalg/eigen — symmetric eigendecomposition & real SVD (iterative)

Source: `src/linalg/eigen.ll` · Symbols: `universe_linalg_*` · C ABI, `nounwind`.

## Purpose

Iterative dense decompositions for **`f64`**, row-major matrices, following
Julia's `LinearAlgebra` names (`eigen`, `svd`). Two entry points:

- **`universe_linalg_eigen_sym`** — eigenvalues and eigenvectors of a **REAL
  SYMMETRIC** `n×n` matrix via **cyclic (two-sided) Jacobi rotations**. Jacobi
  is chosen over shifted-QR because every rotation is an exact orthogonal
  similarity, so the eigenvectors stay orthonormal to rounding and each
  eigenpair residual `‖A·v − λv‖` is tiny; it always converges for symmetric
  input (off-diagonal Frobenius mass shrinks every sweep, quadratically near the
  end). The hot rotation-apply streams two contiguous rows as a `<2 x double>`
  kernel. Eigenvalues are returned **ascending** with matching eigenvector
  columns.
- **`universe_linalg_svd`** — full real SVD `A = U·diag(S)·Vᵀ` for any `m×n`
  via **one-sided Jacobi** on the columns of a column-major working copy (so the
  column-pair dot products and rotation applies are contiguous/SIMD). Singular
  values are returned **descending**. The factorization is exact to rounding
  regardless of convergence tightness (`Wc_final = A·V` and `Vc = V` by
  construction). Works for rank-deficient and wide (`m<n`) inputs: surplus
  singular values are `0` with zero `U` columns.

**Scope — symmetric only.** `eigen_sym` assumes `A` is symmetric (it reads all
of `A` but only the symmetric part is meaningful). **General non-symmetric eigen
(complex / defective spectra) is explicitly out of scope** — it requires the
shifted-QR / Hessenberg algorithm and is left for a future module. `svd` makes
no symmetry assumption.

**Convention.** Row-major, contiguous `f64`: element `(i,j)` of an `m×n` matrix
is at `A[i*n + j]`. The caller owns all storage. These two functions **allocate
one contiguous scratch block per call via `malloc`** (never per-element/
per-iteration) and free it before returning — this is the only place in the
`linalg` domain that allocates. `A` is read-only (copied into scratch).

## Exported API

`i32` returns: `0` OK, `1` NULL_PTR, `2` OUT_OF_MEMORY, `3` SIZE_OVERFLOW,
`8` INVALID_ARG (negative dimension), `12` NOT_CONVERGED (Jacobi sweep cap of
100 hit — unreachable for well-formed input). Note: `12` denotes NOT_CONVERGED
in this numeric domain (the plan's numeric-failure convention), overloading the
shared table's `12`.

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_linalg_eigen_sym(const double* A, int64_t n, double* vals, double* vecs)` | Symmetric eigen of `A` (n×n). Writes `n` eigenvalues **ascending** to `vals`, and eigenvectors as **columns** of the n×n `vecs` (`vecs[i*n+j]` = component `i` of eigenvector `j`). | i32 err |
| `int32_t universe_linalg_svd(const double* A, int64_t m, int64_t n, double* U, double* S, double* Vt)` | Real SVD `A = U·diag(S)·Vᵀ`. `U` is m×n, `S` is length n (**descending**), `Vt` is n×n (= `Vᵀ`, right singular vectors in its rows). | i32 err |

Buffer sizes the caller must provide: `eigen_sym` — `vals[n]`, `vecs[n*n]`;
`svd` — `U[m*n]`, `S[n]`, `Vt[n*n]`.

Identities to expect: `A·vⱼ = λⱼ·vⱼ`, `VᵀV = I`, `Σλ = tr(A)`, `Πλ = det(A)`;
`U·diag(S)·Vᵀ = A`, and `S[i]² =` the eigenvalues of `AᵀA`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a` (and `make dylib` the shared object).
Declare the symbols and link the archive:

```llvm
; yourprog.ll — C ABI, nounwind
declare i32 @universe_linalg_eigen_sym(ptr, i64, ptr, ptr)
declare i32 @universe_linalg_svd(ptr, i64, i64, ptr, ptr, ptr)
; ... call them on your own double buffers ...
```

Link line (host build; `-lm` for libm, `-lpthread` per the SDK convention):

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

From C, declare the prototypes from the table above (or a shared header) and
link the same archive.

## Make a CLI

A driver that reads a symmetric `n×n` matrix from stdin (first token `n`, then
`n*n` row-major values) and prints its eigenvalues (ascending), one per line.

```c
/* eig_cli.c — clang eig_cli.c build/libuniverse.a -lpthread -lm -o eig_cli */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int32_t universe_linalg_eigen_sym(const double*, int64_t, double*, double*);

int main(void) {
    int64_t n;
    if (scanf("%lld", (long long*)&n) != 1 || n <= 0) {
        fprintf(stderr, "usage: echo 'n  <n*n row-major values>' | eig_cli\n");
        return 2;
    }
    double *A    = malloc((size_t)n*n*sizeof(double));
    double *vals = malloc((size_t)n*sizeof(double));
    double *vecs = malloc((size_t)n*n*sizeof(double));
    for (int64_t i = 0; i < n*n; i++)
        if (scanf("%lf", &A[i]) != 1) { fprintf(stderr, "need %lld values\n", (long long)n*n); return 2; }
    int32_t rc = universe_linalg_eigen_sym(A, n, vals, vecs);
    if (rc) { fprintf(stderr, "eigen_sym error %d\n", rc); return 1; }
    for (int64_t i = 0; i < n; i++) printf("%.10g\n", vals[i]);
    free(A); free(vals); free(vecs); return 0;
}
```

```
$ echo "3  2 -1 0  -1 2 -1  0 -1 2" | ./eig_cli
0.5857864376
2
3.414213562
```

## Notes

- **dtype/layout:** dense, row-major, contiguous `f64`; element `(i,j)` at
  `A[i*n+j]`. No stride/leading-dimension parameter — pack tightly.
- **Symmetry:** `eigen_sym` treats `A` as symmetric; only its symmetric part
  matters. Feeding a non-symmetric matrix is a usage error (out of scope).
- **Ownership / allocation:** unlike the rest of `linalg`, these two functions
  `malloc` one scratch block per call (sized `~2·n²` doubles for `eigen_sym`,
  `~m·n + n²` for `svd`) and free it before returning. `2` OUT_OF_MEMORY is
  returned if that allocation fails; `3` SIZE_OVERFLOW if the byte math would
  overflow `i64`.
- **Output order:** eigenvalues **ascending** (Julia convention), singular
  values **descending**; eigenvectors are `vecs` **columns**, right singular
  vectors are `Vt` **rows**.
- **Numerics:** rotations are exact orthogonal transforms → residuals and
  reconstruction are at machine precision (`~1e-15` absolute on `O(1)` data),
  well under the `1e-7` reconstruction tolerance the tests assert.
- **Thread-safety:** no shared/global state; concurrent calls on disjoint
  buffers are safe (each allocates its own scratch).
- **Convergence:** the Jacobi sweep cap is 100; `12` NOT_CONVERGED is defensive
  and not reached by well-formed symmetric / finite input.
