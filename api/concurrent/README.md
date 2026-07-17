# concurrent — lock-free / wait-free concurrency primitives

Exported as `universe_conc_*`. C ABI, `nounwind`. Link against
`build/libuniverse.a` (needs `-lpthread`):
`clang -O3 prog.ll build/libuniverse.a -lpthread -lm -o prog`.

Queue rules across the family: the shared index CLAIM is `monotonic` (a claim
counter publishes no data); all happens-before rides on a PER-SLOT
sequence/turn number (producer release-stores after writing, consumer
acquire-loads before reading). Producer/consumer state is padded 128 B apart to
avoid false sharing. Queue ops return `int32_t` (`0` OK, `6` FULL, `4` EMPTY,
`1` NULL); elements are fixed-size and copied by value. `count` is always an
approximate snapshot.

| Module | What | Threading |
|---|---|---|
| [spsc](spsc.md) | Wait-free bounded ring + unbounded segment queue | 1 producer, 1 consumer |
| [mpsc](mpsc.md) | Lock-free bounded ring + unbounded directory-paged queue; XADD claim | many producers, 1 consumer |
| [mpmc](mpmc.md) | Lock-free bounded MPMC and SPMC rings, per-slot sequence | many/1 producers, many consumers |
| [reclaim](reclaim.md) | EBR + hazard pointers + SeqLock — safe memory reclamation for lock-free node structures | multi-thread |
| [sharded_map](sharded_map.md) | Striped-lock concurrent hash map (byte-string keys → i64) | multi-thread |
| [striped_counter](striped_counter.md) | Sharded 64-bit counter, one shard per 128 B line | multi-thread |

Selection guide: 1↔1 → spsc; N→1 → mpsc; N↔M → mpmc; contended shared map →
sharded_map; contended counter → striped_counter; building your own lock-free
node structure → reclaim.
