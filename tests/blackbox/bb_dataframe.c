/* Black-box: DataFrame end-to-end through the public C ABI only.
 * build series -> frame -> SIMD gt_scalar mask -> filter -> group_by -> sum.
 * dtype enum: I32=0, I64=1, F32=2, F64=3, BOOL=4, STR=5. */
#include <stdint.h>
#include <math.h>
#include <stdio.h>

/* name_keys entry: frame's {ptr name, i64 len} pair representation. */
typedef struct { const char *name; int64_t len; } name_key;

void   *universe_dataframe_new(void);
int32_t universe_dataframe_with_column(void *df, const void *name, int64_t name_len, void *series);
void   *universe_dataframe_series_from(int32_t dtype, const void *values, int64_t len);
void   *universe_dataframe_column(void *df, const void *name, int64_t name_len);
void   *universe_dataframe_series_values(void *s);
int64_t universe_dataframe_series_len(void *s);
int64_t universe_dataframe_height(void *df);

void   *universe_dataframe_gt_scalar(void *a, int32_t stype, int64_t ival, double fval);
void   *universe_dataframe_filter(void *df, void *mask);

void   *universe_dataframe_group_by(void *df, const void *key_names, int64_t nk);
int32_t universe_dataframe_group_by_agg(void *gb, const void *agg_cols, int64_t nc,
                                        const void *agg_ops, void *out_df);

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  bb_dataframe: %s\n", msg); fails++; } } while (0)

int main(void) {
    /* two columns: grp (I32 key), val (F64 payload) */
    int32_t grp_vals[5] = { 0, 1, 0, 1, 0 };
    double  val_vals[5] = { 10.0, 20.0, 30.0, 40.0, 50.0 };

    void *grp = universe_dataframe_series_from(0 /*I32*/, grp_vals, 5);
    void *val = universe_dataframe_series_from(3 /*F64*/, val_vals, 5);
    CHECK(grp && val, "series_from");

    void *df = universe_dataframe_new();
    CHECK(df != NULL, "dataframe_new");
    CHECK(universe_dataframe_with_column(df, "grp", 3, grp) == 0, "with_column grp");
    CHECK(universe_dataframe_with_column(df, "val", 3, val) == 0, "with_column val");
    CHECK(universe_dataframe_height(df) == 5, "height==5");

    /* SIMD comparator builds a BOOL mask: val > 25.0 -> rows {2,3,4}. */
    void *valcol = universe_dataframe_column(df, "val", 3);
    CHECK(valcol != NULL, "column val");
    void *mask = universe_dataframe_gt_scalar(valcol, 3 /*F64*/, 0, 25.0);
    CHECK(mask != NULL, "gt_scalar mask");

    void *filtered = universe_dataframe_filter(df, mask);
    CHECK(filtered != NULL, "filter");
    CHECK(universe_dataframe_height(filtered) == 3, "filtered height==3");

    /* group_by grp, then sum(val) per group. */
    name_key keys[1] = { { "grp", 3 } };
    void *gb = universe_dataframe_group_by(filtered, keys, 1);
    CHECK(gb != NULL, "group_by");

    int64_t agg_cols[1] = { 1 };   /* source column index of "val" */
    int32_t agg_ops[1]  = { 0 };   /* 0 = sum */
    void *out = NULL;
    int32_t rc = universe_dataframe_group_by_agg(gb, agg_cols, 1, agg_ops, &out);
    CHECK(rc == 0, "group_by_agg status");
    CHECK(out != NULL, "group_by_agg out frame");

    if (out) {
        int64_t h = universe_dataframe_height(out);
        CHECK(h == 2, "grouped height==2");

        void *outgrp = universe_dataframe_column(out, "grp", 3);
        void *outsum = universe_dataframe_column(out, "val_sum", 7);
        CHECK(outgrp && outsum, "output columns present");

        if (outgrp && outsum && h == 2) {
            int32_t *g = (int32_t *)universe_dataframe_series_values(outgrp);
            double  *s = (double  *)universe_dataframe_series_values(outsum);
            /* group 0 = rows {30,50} -> 80 ; group 1 = row {40} -> 40 */
            double sum_g0 = 0, sum_g1 = 0;
            int seen0 = 0, seen1 = 0;
            for (int i = 0; i < 2; i++) {
                if (g[i] == 0) { sum_g0 = s[i]; seen0 = 1; }
                else if (g[i] == 1) { sum_g1 = s[i]; seen1 = 1; }
            }
            CHECK(seen0 && seen1, "both groups present");
            CHECK(fabs(sum_g0 - 80.0) < 1e-9, "group 0 sum == 80");
            CHECK(fabs(sum_g1 - 40.0) < 1e-9, "group 1 sum == 40");
        }
    }

    if (fails) { printf("bb_dataframe: %d failure(s)\n", fails); return 1; }
    return 0;
}
