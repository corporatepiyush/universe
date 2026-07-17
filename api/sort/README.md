# sort — sorting algorithms

Two families. **Generic comparator sorts** take
`(void* base, int64_t count, int64_t elem_size, int (*cmp)(const void*, const void*))`
and return `int32_t` (0 OK, 1 NULL); the comparator uses the C convention
(negative / zero / positive). **Integer-specialized sorts** take
`(int32_t* base, int64_t count)` and are the right *class* for integer keys
(~20× a comparison sort — pick these for `int32`). Symbols are `universe_sort_*`,
C ABI, `nounwind`. Link against `build/libuniverse.a`.

## Generic comparator sorts

| Module | Order / stability | Notes |
|---|---|---|
| [quick](quick.md) | O(N log N) expected, in-place, unstable | General default; median-of-3 + Lomuto, recursion-free, delegates >1 KiB elements to heap. |
| [merge](merge.md) | O(N log N), **stable** | Bottom-up ping-pong; one aux allocation (may OOM). |
| [heap](heap.md) | O(N log N) worst, in-place, unstable | Allocation-free worst-case bound. |
| [insertion](insertion.md) | O(N²) / O(N) nearly-sorted, **stable**, in-place | Best for small / nearly-sorted arrays. |
| [shell](shell.md) | sub-quadratic, in-place, unstable | Knuth gaps; allocation-free middle ground. |
| [selection](selection.md) | O(N²) compares, O(N) moves, in-place | Minimizes moves — for large/expensive elements. |
| [bubble](bubble.md) | O(N²), **stable**, in-place | Reference sort; last-swap optimization. |

## Integer-specialized sorts (int32)

| Module | Order | Notes |
|---|---|---|
| [radix](radix.md) | O(4N) LSD | Any int32 range incl. negatives; one aux buffer. |
| [counting](counting.md) | O(N + range) | Small-range int32; returns INVALID_ARG(8) past 2²⁶ buckets. |
