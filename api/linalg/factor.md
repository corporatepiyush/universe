# linalg/factor — direct dense factorizations & solvers

Source: `src/linalg/factor.ll` · Symbols: `universe_linalg_*` · C ABI, `nounwind`.

## Purpose

Direct (non-iterative) dense linear-algebra methods built on the
`linalg/matrix` BLAS core: **LU** with partial pivoting, general **solve**,
matrix **inverse**, **determinant**, **Cholesky**, Householder **QR**,
triangular solve, and **least squares**. All matrices are dense, row-major,
contiguous **`f64`**; element `(i,j)` of an `m×n` matrix is at `A[i*n + j]`
(leading dimension `lda = n`). Method names follow Julia's `LinearAlgebra`
stdlib (functional inspiration only — the layout, kernels, and IR are designed
here from first principles).

Direct factorizations are the right algorithm class for exact dense solves:
`O(n³)` elimination with a SIMD inner kernel is both fastest and most accurate
for well-conditioned systems. LU uses partial pivoting (largest-magnitude pivot
per column → numerically stable; the zero-pivot test is the singular detector).
QR uses Householder reflectors applied in their contiguous rank-1 **row** form
so the inner update vectorizes. The two hot leaves — an axpy
(`dst += coef·src`) and a dot reduction — are `<2 x double>` primary paths with
a scalar tail; they lower to packed `fmla.2d` (AArch64) / `vfmadd*pd` (x86 FMA)
in the elimination and Householder loops. lstsq avoids the normal equations
(which square the condition number), solving via QR + back-substitution.

**Allocation.** Unlike the matrix core, these methods need scratch, so each call
does **one `malloc`** (an arena carved into all the working buffers) and frees it
before returning; `tri_solve` and `cholesky` allocate nothing. The caller owns
all input and output buffers. All size math is overflow-checked.

## Exported API

`i32` returns use: `0` OK, `1` NULL_PTR, `2` OOM, `3` SIZE_OVERFLOW,
`8` INVALID_ARG (bad/mismatched dimensions), `11` SINGULAR (zero pivot or zero
triangular diagonal), `13` NOT_SPD (Cholesky on a non-positive-definite matrix).
`universe_linalg_det` returns the value directly and yields `0.0` on any
null/degenerate/singular input.

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_linalg_lu(const double* A, int64_t n, double* LU_out, int32_t* piv_out, double* sign_out)` | LU factor (partial pivot): `LU_out` holds L (strict-lower, unit diag) and U (upper); `piv_out[k]` = row swapped with `k`; `*sign_out` = permutation sign (±1) | i32 err (11 if singular) |
| `int32_t universe_linalg_solve(const double* A, int64_t n, const double* b, int64_t nrhs, double* x_out)` | Solve `A·X = B` for `nrhs` columns; `b`, `x_out` are n×nrhs row-major | i32 err (11 if singular) |
| `int32_t universe_linalg_inv(const double* A, int64_t n, double* out)` | `out = A⁻¹` (n×n) via LU-solve of I | i32 err (11 if singular) |
| `double universe_linalg_det(const double* A, int64_t n)` | Determinant = product of U's diagonal × sign | double (`0.0` if singular) |
| `int32_t universe_linalg_cholesky(const double* A, int64_t n, double* L_out)` | SPD `A = L·Lᵀ`; `L_out` is lower-triangular (upper zeroed) | i32 err (`13` if not SPD) |
| `int32_t universe_linalg_qr(const double* A, int64_t m, int64_t n, double* Q_out, double* R_out)` | Householder QR: `A = Q·R`, `Q` is m×m orthonormal, `R` is m×n upper | i32 err |
| `int32_t universe_linalg_tri_solve(const double* T, int64_t n, const double* b, int32_t upper, double* x_out)` | Triangular solve `T·x = b`; `upper != 0` = back-subst (U), else forward (L) | i32 err (11 if zero diag) |
| `int32_t universe_linalg_lstsq(const double* A, int64_t m, int64_t n, const double* b, double* x_out)` | Least squares `min‖A·x − b‖` for `m ≥ n` via QR; `x_out` len n | i32 err |

Sizing: `LU_out`, `inv out`, `L_out` are `n*n`; `piv_out` is `n` `int32_t`;
`sign_out` is one `double`. `solve` `b`/`x_out` are `n*nrhs`. QR `Q_out` is
`m*m`, `R_out` is `m*n`. `lstsq` `b` is `m`, `x_out` is `n`.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a` (and `make dylib` the shared object).
Declare the symbols you call and link the archive:

```llvm
; yourprog.ll — C ABI, nounwind
declare i32 @universe_linalg_solve(ptr, i64, ptr, i64, ptr)
declare double @universe_linalg_det(ptr, i64)
; ... call them on your own double buffers ...
```

Link line (host build; `-lm` for libm, `-lpthread` per the SDK convention):

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

From C, declare the prototypes from the table above (or a shared header) and
link the same archive.

## Make a CLI

A small C driver that solves `A·x = b`: dimension `n`, then the `n*n` row-major
`A` entries, then the `n` entries of `b`, all from `argv`.

```c
/* solve_cli.c — clang solve_cli.c build/libuniverse.a -lpthread -lm -o solve_cli */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int32_t universe_linalg_solve(const double*, int64_t, const double*,
                                      int64_t, double*);

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s n <A n*n row-major> <b n>\n", argv[0]); return 2; }
    int64_t n = atoll(argv[1]);
    int64_t need = 2 + n*n + n;
    if (n < 1 || argc != need) {
        fprintf(stderr, "need n=%lld: %lld A values + %lld b values\n",
                (long long)n, (long long)(n*n), (long long)n);
        return 2;
    }
    double *A = malloc((size_t)n*n*sizeof(double));
    double *b = malloc((size_t)n*sizeof(double));
    double *x = malloc((size_t)n*sizeof(double));
    for (int64_t i = 0; i < n*n; i++) A[i] = atof(argv[2 + i]);
    for (int64_t i = 0; i < n;   i++) b[i] = atof(argv[2 + n*n + i]);
    int32_t rc = universe_linalg_solve(A, n, b, 1, x);   /* nrhs = 1 */
    if (rc) { fprintf(stderr, "solve error %d (11 = singular)\n", rc); return 1; }
    for (int64_t i = 0; i < n; i++) printf("%s%g", i ? " " : "", x[i]);
    putchar('\n');
    free(A); free(b); free(x); return 0;
}
```

```
$ clang solve_cli.c build/libuniverse.a -lpthread -lm -o solve_cli
# 3x2 - z = 1 ;  2x - 2y + 4z = -2 ;  -x + 0.5y - z = 0   ->  (1, -2, -2)
$ ./solve_cli 3   3 2 -1  2 -2 4  -1 0.5 -1   1 -2 0
1 -2 -2
```

The same pattern drives the other entry points: read the matrix, call
`universe_linalg_det` / `_inv` / `_cholesky` / `_qr` / `_lstsq`, print the
result. (Piping from stdin instead of `argv` is a one-line change to `scanf`.)

## Notes

- **dtype/layout:** dense, row-major, contiguous `f64` only; element `(i,j)` at
  `A[i*n + j]`. No stride/leading-dimension parameter — pack tightly.
- **Allocation & ownership:** each call (except `tri_solve`/`cholesky`) does a
  single internal `malloc`/`free` for scratch and returns `2` (OOM) if it fails.
  You own every input and output buffer and must size them per the table.
- **Aliasing:** keep distinct input and output buffers. `tri_solve` tolerates
  `x_out == b`; `cholesky` tolerates `L_out == A`; other ops assume no overlap.
- **Pivoting / stability:** `lu`/`solve`/`inv`/`det` use partial pivoting.
  A zero pivot ⇒ `11` (`det` ⇒ `0.0`). `cholesky` reports `13` the moment a
  diagonal radicand is non-positive (its SPD test).
- **QR sign convention:** reflectors pick `α = −sign(x₀)·‖x‖`, so `R`'s diagonal
  may be negative; `Q·R == A` and `Qᵀ·Q == I` hold to ~1e-13. `R`'s strict lower
  triangle is explicitly zeroed.
- **lstsq:** requires `m ≥ n` (else `8`); solves via QR (not the normal
  equations) and matches a normal-equations reference to machine precision on
  well-conditioned systems.
- **Thread-safety:** no shared state; concurrent calls on disjoint buffers are
  safe (each call owns its own scratch allocation).
- **Numerics:** SIMD axpy/dot use FMA contraction; reconstructions
  (`L·U == P·A`, `Q·R == A`, `L·Lᵀ == A`) hold to < 1e-7 relative on random
  well-conditioned matrices, typically ~1e-14.
```
