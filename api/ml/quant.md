# ml/quant — vector quantization codec + int8/binary distances

## Purpose

A vector-quantization codec plus int8/binary distance kernels — the memory axis
of an embedding store: a stored vector costs `D · bytes-per-component`, and
quantization cuts that 4×–48× with bounded recall loss. Pure compute over
caller-owned memory; never allocates. Blobs are SELF-DESCRIBING so a corpus can
mix schemes mid-migration with no out-of-band metadata:

```
[0]      scheme tag  u8  (0=f32, 1=f16, 2=int8, 3=binary)
[1..5)   dims        u32 little-endian
[5..)    payload:   f32   -> dims*4 (raw copy)
                    f16   -> dims*2 (IEEE half)
                    int8  -> 4 (f32 per-vector scale) + dims (one i8 each)
                    binary-> (dims+7)/8 (sign bit per component, LSB-first)
```

int8 uses a symmetric scale `max|x|/127`, so it divides out of a cosine ratio —
`dist_cos_i8` needs no scale argument. Binary distance is Hamming (popcount over
packed sign bits). Distance kernels ship SIMD + `_scalar` twins.

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int32_t universe_ml_quant_encode(void *dst, int64_t dcap, const float *src, int64_t dims, int32_t scheme, int64_t *outlen)` | Encode `src` (f32, `dims`) into a self-describing blob | 0 OK, 6 FULL, 8 INVALID_ARG |
| `int32_t universe_ml_quant_decode(float *dst, int64_t dcapel, const void *src, int64_t slen, int64_t *outdims)` | Decode a blob into f32 `dst` (`dcapel` element cap) | 0 OK, 6 FULL, 13 PARSE |
| `int32_t universe_ml_quant_info(const void *src, int64_t slen, int32_t *outscheme, int64_t *outdims)` | Read a blob's scheme + dims without decoding | 0 OK, 13 PARSE |
| `float universe_ml_dist_cos_i8(const int8_t *a, const int8_t *b, int64_t n)` | Cosine distance of two int8 vectors (scale-free) | cosine value |
| `int64_t universe_ml_dist_hamming(const uint8_t *a, const uint8_t *b, int64_t n)` | Hamming distance over `n` packed bytes | bit-difference count |

Twins: `universe_ml_dist_cos_i8_scalar`, `universe_ml_dist_hamming_scalar`
(fallback + oracle). `scheme` values: 0=f32, 1=f16, 2=int8, 3=binary. `outlen`
(encode) / `outdims` (decode/info) / `outscheme` (info) are caller out-params.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare i32   @universe_ml_quant_encode(ptr, i64, ptr, i64, i32, ptr)
declare i32   @universe_ml_quant_decode(ptr, i64, ptr, i64, ptr)
declare float @universe_ml_dist_cos_i8(ptr, ptr, i64)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Round-trip a stdin float vector through int8 quantization; print decoded values
and the blob size. Usage: `quant <scheme>` (0=f32,1=f16,2=int8,3=binary).

```c
// quant.c   usage: quant <scheme>  < floats.txt
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int32_t universe_ml_quant_encode(void *, int64_t, const float *, int64_t, int32_t, int64_t *);
int32_t universe_ml_quant_decode(float *, int64_t, const void *, int64_t, int64_t *);

int main(int argc, char **argv) {
    int scheme = argc > 1 ? atoi(argv[1]) : 2;   // default int8
    static float in[65536];
    int64_t dims = 0;
    while (dims < 65536 && scanf("%f", &in[dims]) == 1) dims++;
    static unsigned char blob[262144];
    int64_t blen = 0;
    if (universe_ml_quant_encode(blob, sizeof blob, in, dims, scheme, &blen)) return 1;
    static float out[65536];
    int64_t od = 0;
    if (universe_ml_quant_decode(out, 65536, blob, blen, &od)) return 1;
    printf("blob=%lld bytes, dims=%lld\n", (long long)blen, (long long)od);
    for (int64_t i = 0; i < od; i++) printf("%g ", out[i]);
    putchar('\n');
    return 0;
}
```

```
clang -O3 quant.c build/libuniverse.a -lpthread -lm -o quant
printf '0.5 -1.0 0.25 2.0\n' | ./quant 2
```

## Notes

- **dtype/layout:** encode consumes f32; blobs are self-describing (scheme + LE
  dims header, all targets little-endian).
- **Lossy:** int8/binary trade recall for footprint; `dist_cos_i8` is scale-free
  by construction, Hamming operates on packed binary blobs.
- **Ownership:** caller owns `dst`/`src` buffers; nothing is allocated
  internally.
- **Threading:** stateless, safe on disjoint buffers.
