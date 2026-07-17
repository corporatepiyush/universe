/* Black-box: hex + base64 through the public C ABI only.
 * Round-trip AND known-answer vectors (RFC 4648 for base64). */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* Public prototypes (declared here; the SDK ships no C headers). */
int64_t universe_hex_encode_len(int64_t n);
int64_t universe_hex_encode(void *dst, const void *src, int64_t n);
int64_t universe_hex_decode(void *dst, const void *src, int64_t n);

int64_t universe_base64_encode_len(int64_t n);
int64_t universe_base64_decode_len(const void *src, int64_t n);
int64_t universe_base64_encode(void *dst, const void *src, int64_t n, int32_t urlsafe);
int64_t universe_base64_decode(void *dst, const void *src, int64_t n, int32_t urlsafe);

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  bb_encoding: %s\n", msg); fails++; } } while (0)

int main(void) {
    /* ---- hex known-answer + round-trip ---- */
    const uint8_t raw[] = { 0x00, 0x0f, 0x10, 0xab, 0xff, 'z' };
    const int64_t rawn = (int64_t)sizeof(raw);
    char hexbuf[64];
    int64_t hn = universe_hex_encode(hexbuf, raw, rawn);
    CHECK(hn == 2 * rawn, "hex_encode length");
    CHECK(universe_hex_encode_len(rawn) == 2 * rawn, "hex_encode_len");
    hexbuf[hn] = 0;
    CHECK(memcmp(hexbuf, "000f10abff7a", 12) == 0, "hex known-answer");

    uint8_t back[64];
    int64_t bn = universe_hex_decode(back, hexbuf, hn);
    CHECK(bn == rawn, "hex_decode length");
    CHECK(memcmp(back, raw, (size_t)rawn) == 0, "hex round-trip");

    /* odd length must be rejected */
    CHECK(universe_hex_decode(back, "abc", 3) == -1, "hex_decode odd rejected");
    /* non-hex char must be rejected */
    CHECK(universe_hex_decode(back, "zz", 2) == -1, "hex_decode nonhex rejected");

    /* ---- base64 known-answer (RFC 4648 §10) + round-trip ---- */
    struct { const char *in; const char *out; } kat[] = {
        { "",       ""         },
        { "f",      "Zg=="     },
        { "fo",     "Zm8="     },
        { "foo",    "Zm9v"     },
        { "foob",   "Zm9vYg==" },
        { "fooba",  "Zm9vYmE=" },
        { "foobar", "Zm9vYmFy" },
    };
    for (size_t i = 0; i < sizeof(kat) / sizeof(kat[0]); i++) {
        int64_t inlen = (int64_t)strlen(kat[i].in);
        char enc[64];
        int64_t en = universe_base64_encode(enc, kat[i].in, inlen, 0);
        enc[en] = 0;
        CHECK(en == (int64_t)strlen(kat[i].out), "base64 encode length");
        CHECK(strcmp(enc, kat[i].out) == 0, "base64 encode known-answer");

        /* decode back */
        int64_t explen = universe_base64_decode_len(enc, en);
        CHECK(explen == inlen, "base64 decode_len");
        uint8_t dec[64];
        int64_t dn = universe_base64_decode(dec, enc, en, 0);
        CHECK(dn == inlen, "base64 decode length");
        CHECK(memcmp(dec, kat[i].in, (size_t)inlen) == 0, "base64 round-trip");
    }

    /* url-safe alphabet: bytes that map to '+' and '/' in standard. */
    const uint8_t urlraw[] = { 0xfb, 0xff, 0xbf };  /* -> "+/+/"-ish glyphs */
    char ustd[16], uurl[16];
    int64_t sn = universe_base64_encode(ustd, urlraw, 3, 0);
    int64_t un = universe_base64_encode(uurl, urlraw, 3, 1);
    ustd[sn] = 0; uurl[un] = 0;
    CHECK(sn == 4 && un == 4, "base64 urlsafe length");
    /* url-safe must contain no '+' or '/' */
    CHECK(strchr(uurl, '+') == NULL && strchr(uurl, '/') == NULL, "base64 urlsafe alphabet");
    /* url-safe decode round-trips */
    uint8_t udec[16];
    int64_t udn = universe_base64_decode(udec, uurl, un, 1);
    CHECK(udn == 3 && memcmp(udec, urlraw, 3) == 0, "base64 urlsafe round-trip");

    if (fails) { printf("bb_encoding: %d failure(s)\n", fails); return 1; }
    return 0;
}
