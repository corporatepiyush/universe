/* Black-box: lz4 + snappy block codecs through the public C ABI only.
 * Round-trip identity (compress -> decompress == original) AND a
 * hand-crafted known-answer decode for each format. */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

int64_t universe_compress_lz4_bound(int64_t slen);
int64_t universe_compress_lz4_encode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_lz4_decode(void *dst, int64_t dcap, const void *src, int64_t slen);

int64_t universe_compress_snappy_bound(int64_t slen);
int64_t universe_compress_snappy_encode(void *dst, int64_t dcap, const void *src, int64_t slen);
int64_t universe_compress_snappy_decode(void *dst, int64_t dcap, const void *src, int64_t slen);

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  bb_compress: %s\n", msg); fails++; } } while (0)

static void roundtrip(const char *tag,
                      int64_t (*bound)(int64_t),
                      int64_t (*enc)(void *, int64_t, const void *, int64_t),
                      int64_t (*dec)(void *, int64_t, const void *, int64_t)) {
    /* Repetitive payload so the greedy matcher actually emits copies. */
    uint8_t in[512];
    for (int i = 0; i < 512; i++) in[i] = (uint8_t)("abcdefgh"[i % 8]);
    int64_t inlen = 512;

    int64_t cap = bound(inlen);
    CHECK(cap >= inlen, tag);
    uint8_t comp[2048];
    int64_t cn = enc(comp, cap, in, inlen);
    CHECK(cn > 0, tag);           /* produced something */
    CHECK(cn <= cap, tag);

    uint8_t out[1024];
    int64_t dn = dec(out, sizeof(out), comp, cn);
    CHECK(dn == inlen, tag);
    CHECK(dn > 0 && memcmp(out, in, (size_t)inlen) == 0, tag);
}

int main(void) {
    roundtrip("lz4 round-trip",    universe_compress_lz4_bound,
              universe_compress_lz4_encode,    universe_compress_lz4_decode);
    roundtrip("snappy round-trip", universe_compress_snappy_bound,
              universe_compress_snappy_encode, universe_compress_snappy_decode);

    /* ---- LZ4 known-answer decode: literals-only final sequence. ----
     * token = (litlen<<4)|matchlen; litlen=5, matchlen=0 -> 0x50, then "hello".
     * A literals-only block ends the stream. */
    const uint8_t lz4blk[] = { 0x50, 'h','e','l','l','o' };
    uint8_t lz4out[16];
    int64_t ln = universe_compress_lz4_decode(lz4out, sizeof(lz4out), lz4blk, (int64_t)sizeof(lz4blk));
    CHECK(ln == 5 && memcmp(lz4out, "hello", 5) == 0, "lz4 known-answer decode");

    /* ---- Snappy known-answer decode: varint len then one literal element. ----
     * len=5 (varint 0x05); literal tag = ((len-1)<<2)|00 = (4<<2) = 0x10; "hello". */
    const uint8_t snpblk[] = { 0x05, 0x10, 'h','e','l','l','o' };
    uint8_t snpout[16];
    int64_t sn = universe_compress_snappy_decode(snpout, sizeof(snpout), snpblk, (int64_t)sizeof(snpblk));
    CHECK(sn == 5 && memcmp(snpout, "hello", 5) == 0, "snappy known-answer decode");

    if (fails) { printf("bb_compress: %d failure(s)\n", fails); return 1; }
    return 0;
}
