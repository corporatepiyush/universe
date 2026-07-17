/* Black-box: kmeans smoke test through the public C ABI only.
 * Two well-separated clusters in 2-D must be split cleanly. */
#include <stdint.h>
#include <math.h>
#include <stdio.h>

/* labels are out-i32; inertia is out-float (nullable); X/C are f32 row-major. */
int32_t universe_ml_kmeans_fit(const float *X, int64_t n, int64_t d, int64_t k,
                               int64_t maxit, float tol,
                               float *C, int32_t *labels, float *inertia);
int32_t universe_ml_kmeans_predict(const float *X, int64_t n, int64_t d, int64_t k,
                                    const float *C, int32_t *labels);

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  bb_ml: %s\n", msg); fails++; } } while (0)

int main(void) {
    /* Rows interleaved A,B,A,B,A,B so first-k-rows init lands one seed per
     * cluster. Cluster A ~ (0,0), cluster B ~ (10,10). */
    float X[12] = {
        0.f, 0.f,     /* A */
        10.f, 10.f,   /* B */
        1.f, 0.f,     /* A */
        11.f, 10.f,   /* B */
        0.f, 1.f,     /* A */
        10.f, 11.f,   /* B */
    };
    int64_t n = 6, d = 2, k = 2;
    float C[4];
    int32_t labels[6];
    float inertia = -1.f;

    int32_t rc = universe_ml_kmeans_fit(X, n, d, k, 100, 1e-6f, C, labels, &inertia);
    CHECK(rc == 0, "kmeans_fit status OK");

    /* A rows (0,2,4) share a label; B rows (1,3,5) share the other. */
    CHECK(labels[0] == labels[2] && labels[2] == labels[4], "cluster A cohesive");
    CHECK(labels[1] == labels[3] && labels[3] == labels[5], "cluster B cohesive");
    CHECK(labels[0] != labels[1], "clusters separated");

    /* Inertia (within-cluster SSE) is small for tight clusters. */
    CHECK(inertia >= 0.f && inertia < 10.f, "inertia sane");

    /* Centroids land near the true means (0.33,0.33) and (10.33,10.33). */
    int la = labels[0];               /* label of cluster A */
    const float *ca = &C[la * 2];
    const float *cb = &C[(1 - la) * 2];
    CHECK(fabsf(ca[0] - 0.3333f) < 0.2f && fabsf(ca[1] - 0.3333f) < 0.2f, "centroid A near mean");
    CHECK(fabsf(cb[0] - 10.3333f) < 0.2f && fabsf(cb[1] - 10.3333f) < 0.2f, "centroid B near mean");

    /* predict must reproduce the fit labels on the same data. */
    int32_t plabels[6];
    int32_t prc = universe_ml_kmeans_predict(X, n, d, k, C, plabels);
    CHECK(prc == 0, "kmeans_predict status OK");
    for (int i = 0; i < 6; i++)
        CHECK(plabels[i] == labels[i], "predict matches fit");

    if (fails) { printf("bb_ml: %d failure(s)\n", fails); return 1; }
    return 0;
}
