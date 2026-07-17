# llm/openai — OpenAI-protocol chat-completions client

## Purpose

A minimal OpenAI-protocol chat-completions client. It builds the request JSON by
hand (the SDK has no JSON encoder, only a tokenizer), ships it over the plaintext
HTTP/1.1 client ([`src/http`](../http/http.md)), and walks the response with the
`src/parse/json` pull tokenizer — handling both non-streaming completions and
Server-Sent-Events (SSE) streaming deltas. Three pure phases with no syscall in
any compute loop: BUILD (`build_request` assembles the body via bounded appends;
`json_escape` provides the string escaper), IO (the http client does the one POST
+ read), PARSE (`parse_response`/`extract_delta` drive the zero-copy tokenizer
and unescape the single content span into the caller buffer). **PLAINTEXT ONLY,
NO TLS** — it targets LOCAL OpenAI-compatible servers on loopback/LAN
(llama.cpp, vLLM, Ollama, LM Studio), NOT public `api.openai.com`.

**Message input** (per element, 32 B, caller array): `role_ptr`@0, `role_len`@8,
`content_ptr`@16, `content_len`@24. **Config** (caller-allocated, 72 B): filled
by `config_init` (ip, port, host, path, model, key).

## Exported API

| C signature | Description | Returns |
|---|---|---|
| `int64_t universe_llm_json_escape(void *dst, int64_t dstcap, const void *src, int64_t len)` | JSON-escape `src` into `dst` | escaped length, or `-1` if `dst` too small |
| `int64_t universe_llm_build_request(void *dst, int64_t dstcap, const void *model, int64_t mlen, const void *msgs, int64_t nmsgs, int32_t stream)` | Assemble the chat-completions request body (msgs = 32-byte entries) | body length, or `-1` on overflow |
| `int32_t universe_llm_parse_response(void *body, int64_t len, void *out, int64_t outcap, int64_t *out_len, void *out_finish, int64_t fcap, int64_t *out_flen, int64_t *out_usage)` | Extract `choices[0].message.content` (+ optional finish/usage) | 0 OK, 5 NOT_FOUND, 6 FULL, 11 error-obj, 13 PARSE, 2 OOM |
| `int32_t universe_llm_extract_delta(void *body, int64_t len, void *out, int64_t outcap, int64_t *out_len)` | Extract `choices[0].delta.content` from one SSE chunk | 0 OK, 5 no content, codes |
| `int32_t universe_llm_sse_process(void *buf, int64_t len, void *scratch, int64_t scratchcap, void *cb, void *userdata, int64_t *out_consumed)` | Split `buf` on `\n`, call `cb` with each complete `data:` delta | 0 OK; `*out_consumed` set |
| `void universe_llm_config_init(void *cfg, int32_t ip, int32_t port, const void *host, int64_t hlen, const void *path, int64_t plen, const void *model, int64_t mlen, const void *key, int64_t klen)` | Fill a caller 72-byte config struct | — |
| `int32_t universe_llm_complete(void *conn, void *cfg, void *msgs, int64_t nmsgs, void *bodybuf, int64_t bodycap, void *out, int64_t outcap, int64_t *out_len, void *out_finish, int64_t fcap, int64_t *out_flen, int64_t *out_usage)` | Networked convenience: build + POST + parse over `conn` | 0 OK, codes |

`out_usage`, if non-null, points at 3 `int64_t` slots (prompt, completion, total
tokens). `out_finish`/`out_usage` may be NULL. `cb` for `sse_process`:
`void cb(void *userdata, const void *delta, int64_t delta_len)`. `conn` is a
`universe_http_conn` / `universe_http_connect` handle.

## Use in an LLVM-based environment

`make lib` builds `build/libuniverse.a`. Declare the symbols (C-ABI, `nounwind`):

```llvm
declare void @universe_llm_config_init(ptr, i32, i32, ptr, i64, ptr, i64, ptr, i64, ptr, i64)
declare i32  @universe_llm_complete(ptr, ptr, ptr, i64, ptr, i64,
             ptr, i64, ptr, ptr, i64, ptr, ptr)
declare ptr  @universe_http_connect(i32, i32, i64)
```

```
clang -O3 yourprog.ll build/libuniverse.a -lpthread -lm -o prog
```

## Make a CLI

Chat with a local OpenAI-compatible server: send one user message from argv,
print the assistant reply. (Point it at e.g. a llama.cpp/Ollama server on
`127.0.0.1:8080`.)

```c
// chat.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void   *universe_http_connect(int32_t, int32_t, int64_t);
void    universe_http_conn_destroy(void *);
void    universe_llm_config_init(void *, int32_t, int32_t, const void *, int64_t,
            const void *, int64_t, const void *, int64_t, const void *, int64_t);
int32_t universe_llm_complete(void *, void *, void *, int64_t, void *, int64_t,
            void *, int64_t, int64_t *, void *, int64_t, int64_t *, int64_t *);

int main(int argc, char **argv) {
    const char *prompt = argc > 1 ? argv[1] : "Hello!";
    void *conn = universe_http_connect(0x7F000001, 8080, 1 << 16);   // 127.0.0.1:8080
    if (!conn) { fprintf(stderr, "connect failed\n"); return 1; }

    char cfg[72];
    universe_llm_config_init(cfg, 0x7F000001, 8080,
        "localhost", 9, "/v1/chat/completions", 20, "local-model", 11, "", 0);

    // one message: {role_ptr, role_len, content_ptr, content_len} (32 B)
    char msgs[32] = {0};
    *(const char **)(msgs + 0)  = "user"; *(int64_t *)(msgs + 8)  = 4;
    *(const char **)(msgs + 16) = prompt; *(int64_t *)(msgs + 24) = strlen(prompt);

    char body[8192], out[8192];
    int64_t out_len = 0;
    int rc = universe_llm_complete(conn, cfg, msgs, 1, body, sizeof body,
                 out, sizeof out, &out_len, NULL, 0, NULL, NULL);
    if (rc == 0) fwrite(out, 1, out_len, stdout), putchar('\n');
    else fprintf(stderr, "complete rc=%d\n", rc);
    universe_http_conn_destroy(conn);
    return rc;
}
```

```
clang -O3 chat.c build/libuniverse.a -lpthread -lm -o chat
./chat "Explain mechanical sympathy in one line."
```

## Notes

- **No TLS:** local/loopback OpenAI-compatible servers only; not for
  `api.openai.com`. An `Authorization: Bearer <key>` header is still emitted for
  servers that check a token.
- **Zero-copy parse:** the tokenizer emits offsets/lengths; only the ONE content
  span is unescaped into the caller `out` buffer. Ownership stays with the
  caller — the module never allocates the response.
- **Buffers:** `bodybuf` (request body) and `out` (decoded content) are
  caller-sized; a too-small `out` yields 6 FULL.
- **Threading:** single-threaded, one conn at a time.
