# search — string search / indexing engines

Multi-pattern matching and full-text indexing over byte strings. Built
structures are immutable and safe to share read-only across threads. Symbols are
`universe_search_*`, C ABI, `nounwind`. Link against `build/libuniverse.a`.

| Module | Purpose |
|---|---|
| [ahocorasick](ahocorasick.md) | Aho-Corasick multi-pattern matcher (full-DFA goto table, one linear pass). |
| [suffix_array](suffix_array.md) | Suffix array + LCP with substring find/count (prefix doubling + Kasai). |
