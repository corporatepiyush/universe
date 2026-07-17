# ml/linear — linear + logistic regression

## Purpose

Linear and logistic regression by full-batch gradient descent over row-major f32
samples. Both share one weight-update kernel: prediction `p = f(w·x + b)` with
`f` = identity (linear) or sigmoid (logistic); residual `r = p − y`; gradients
`grad_w = (1/n) Σ r·x`, `grad_b = (1/n) Σ r`; step `w -= lr·grad_w`,
`b -= lr·grad_b`. A single internal `lr_fit_impl` carries a loop-invariant
`logistic` flag (hoisted by the optimizer), so the two models are one
implementation. The hot `w·x` dot uses an inlined 4-accumulator `<4 x float>` dot
(fmla, no cross-module call); sigmoid uses libc `expf`. Associative gradient
reductions use `fast` FP.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_linreg_fit(const float *X, const float *y, int64_t n, int64_t d, int64_t iters, float lr, float *w, float *b_out)` | Fit linear weights by GD (MSE loss) | 0 OK, 2 OOM, 8 INVALID_ARG |
| `int32_t universe_ml_linreg_predict(const float *X, int64_t n, int64_t d, const float *w, float b, float *out)` | Predict `w·x + b` for each row | 0 OK, codes |
| `int32_t universe_ml_logreg_fit(const float *X, const float *y, int64_t n, int64_t d, int64_t iters, float lr, float *w, float *b_out)` | Fit logistic weights by GD (log-loss) | 0 OK, codes |
| `int32_t universe_ml_logreg_predict_proba(const float *X, int64_t n, int64_t d, const float *w, float b, float *out)` | Predict `sigmoid(w·x + b)` per row | 0 OK, codes |

`X` is `n×d` row-major; `y` is `n` `float` (targets for linreg, labels in `{0,1}`
for logreg). `w` is out `d` weights, `b_out` out `float` bias. `out` is `n`
predictions.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32 @universe_ml_linreg_fit(ptr, ptr, i64, i64, i64, float, ptr, ptr)
declare i32 @universe_ml_linreg_predict(ptr, i64, i64, ptr, float, ptr)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Fit `y = w·x + b` on a fixed 1-D dataset, then predict stdin x values.

```c
// linreg.c   prints prediction for each stdin x
#include <stdint.h>
#include <stdio.h>
int32_t universe_ml_linreg_fit(const float *, const float *, int64_t, int64_t,
            int64_t, float, float *, float *);
int32_t universe_ml_linreg_predict(const float *, int64_t, int64_t,
            const float *, float, float *);

int main(void) {
    float X[] = {0, 1, 2, 3, 4};       // n=5, d=1
    float y[] = {1, 3, 5, 7, 9};       // y = 2x + 1
    float w[1] = {0}, b = 0;
    if (universe_ml_linreg_fit(X, y, 5, 1, 5000, 0.05f, w, &b)) return 1;
    fprintf(stderr, "w=%g b=%g\n", w[0], b);
    float q, out;
    while (scanf("%f", &q) == 1) {
        universe_ml_linreg_predict(&q, 1, 1, w, b, &out);
        printf("%g\n", out);
    }
    return 0;
}
```

```
clang -O3 linreg.c build/libuniverse.a -lpthread -lm -o linreg
printf '10\n0.5\n' | ./linreg     # ~21, ~2
```

## Notes

- **dtype/layout:** f32, `X` is `n×d` row-major; `w` has length `d`.
- **Convergence:** fixed `iters` of full-batch GD; tune `lr`/`iters` to your
  feature scaling (standardize inputs for stability).
- **Ownership:** caller owns all buffers; internal scratch freed per call.
- **Threading:** stateless per call.
