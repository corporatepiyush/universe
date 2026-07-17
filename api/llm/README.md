# llm — LLM client

OpenAI-protocol chat-completions client built on the SDK's HTTP client and JSON
tokenizer. C-ABI `nounwind`; link against `build/libuniverse.a` (`make lib`).

| Module | Summary |
|---|---|
| [openai](openai.md) | Build request JSON, POST over HTTP/1.1, parse non-streaming + SSE-streaming responses |

**Plaintext only (NO TLS)** — targets local OpenAI-compatible servers
(llama.cpp, vLLM, Ollama, LM Studio) on loopback/LAN, not public
`api.openai.com`. Three pure phases (build → IO → parse); zero-copy response
walk; all buffers caller-owned.
