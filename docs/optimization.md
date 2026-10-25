# Inference timeline and optimization lessons

The working rule is: identify exposed work, estimate net savings, then measure end to end. The longest dependent rank determines a distributed step. Summing overlapping microbenchmarks exaggerates the available speedup.

## A request in time order

| Stage | Work | What to measure | Common optimization |
|---|---|---|---|
| Client / gateway | Request construction, TLS, body transfer, admission | Body bytes, queue and upstream timing | Stable model IDs; avoid accidental retries and duplicate traffic |
| Template / tokenize | System, tools, conversation and images become tokens | Actual tokens and stable prefix length | Envelope-only normalization; avoid mutating tool payloads |
| Cache lookup / allocation | Find reusable blocks, allocate KV, apply leases | Cached tokens, evictions, unmatched suffix, quota | Retain reusable intermediate prefixes; stable prompt order |
| Scheduling | Split prefill/decode under per-step budget | Queue wait and each request's scheduled tokens | Competition-aware sharing instead of a fixed global reduction |
| Cold prefill | Embed and process the uncached input | Per-chunk time, memory, rank skew | Chunking and attention support; cache reuse removes whole chunks |
| Draft | Generate speculative candidates | Draft time, accepted tokens per round | Choose length using net gain, not draft speed alone |
| Target verification | Verify candidates through all model layers | Per-rank compute/communication/wait | Optimize exposed kernels and synchronization |
| Engram | Exact row selection and disk-backed reads | CPU scheduling, I/O service time, barriers | Reuse rows and overlap only when the path is exposed |
| MoE | Route, distribute, quantize, read expert weights, GEMM, combine | Bytes, occupancy, launches and dependencies | Remove provably unused work; test shape and numerical boundaries |
| Sampling / acceptance | Reject or accept candidates and produce tokens | Acceptance rate, committed tokens / round | Preserve the exact sampling contract |
| Stream | Serialize real content and deliver deltas | First **text**, first tool delta, inter-token stalls | Honest SSE; propagate cancellation to free work |

Historical diagnostic labels included a target-verification region around 52 ms and an MoE region around 22.4 ms for a particular trace. These are not universal constants or a complete additive breakdown. The export does not include raw traces, per-rank counters or private conversations, so it does not claim a newly established hardware roofline.

## Theory to decide whether an experiment is worth running

For a memory-bound kernel, the optimistic lower bound is transferred bytes divided by sustainable bandwidth. For a compute-bound region, it is operations divided by attainable throughput for that dtype and shape. Compare both with actual duration and include occupancy, launch and synchronization overhead.

An optimization's wall-time gain is bounded by the exposed fraction of its saved work. For distributed verification, use `max(rank finish time)` over the critical path, not the sum of rank-local savings. Add new branches, register pressure, lost fusion, extra launches and extra synchronization to the cost side.

For speculative decoding, a useful accounting identity is:

```
committed tokens / second ≈ mean committed tokens per round
                         / mean(draft + target verification + exposed wait)
```

A faster draft or more accepted tokens is insufficient if verification and waiting grow faster. Use observed acceptance and round costs for each workload, including JSON and tools. Do not set a universal draft length from a single prose prompt.

## Changes that earned a place in the reference

### Prefix retention

The cache initially lost useful intermediate boundaries after a turn generated output. Adding `--prefix-cache-retention-interval 1024` allowed the next turn to reuse most of the conversation and recompute roughly the unmatched suffix plus new content. In the ~25K fixture, repeated and appended requests recovered 93.6–99.5% reuse. Exact cache residency and hit rate still depend on other traffic, prefix identity and memory pressure.

### Contention-aware prefill

At a 4,096-token step budget, a large cold input could consume every slot before a later warm request was admitted. The scheduler reference gives a competing long prefill at most 2,048 tokens, while leaving a solo prefill at 4,096. Following decode requests reserve at least 64 tokens each. The targeted warm TTFT fell from 40.63 to 4.99 seconds in one controlled scenario; the contended cold request increased from 45.73 to 46.26 seconds. Report both sides of that tradeoff.

A blanket 2,048 budget slowed cold prefill by roughly 40% in the source experiment and was rejected. During a large prefill, decode can still stall between chunks; this patch does not eliminate compute contention.

### Tail-v3 MoE

The exported B12X kernel reference avoids unnecessary work associated with the final 64 padded columns for the applicable shape. Correctness includes padding, boundaries and dtype behavior; the branch/register/traffic cost belongs in the net benefit. Twelve paired natural-EOS cases showed a positive interval, while a separate four-way workload showed higher aggregate throughput. Do not extrapolate either result to every expert shape, runtime or context.

## Changes that were not promoted

| Candidate | Observation | Lesson |
|---|---|---|
| Concurrent Engram reads | ~2.6 ms saved in a microtest, only ~0.3% across a complete four-node round | Overlap, rank maxima and other waits can absorb local savings |
| Adaptive verification / draft policy | Some JSON workloads became slower | Acceptance rate and control overhead are workload-dependent |
| Parallel prefix revision | Baseline moved by ~7.5% | A claimed small gain is unresolved when the reference drifts more |
| Always reduce prefill budget | Cold input became materially slower | Condition the policy on actual contention |
| Broad new runtime swap | Native formats, SM121 support and tools must all remain correct | A new engine headline is a hypothesis, not deployable evidence |

The source workflow also encountered a patch-generation mistake where a named revision was byte-identical to its predecessor, and overlapping operators prepared or switched releases. These are reasons to bind source hashes and compare the actual runtime, not to trust a version label.

## What to investigate next

Collect round-level draft, target, accepted-token and wait timing on every rank, with a shared phase definition and clock assumptions. Separate cold weights/page cache, GPU/CPU power-state transitions, scheduler contention and synchronization skew. Compare identical prompts and cache conditions before writing a candidate. Use the isolated fifth node for CPU compilation or original-kernel diagnostics when available; a draft-model server needs an explicit algorithm and communication budget to be useful to four target ranks.

No experiment here grants permission to stop a production cluster. Keep the fastest verified recipe available; new evidence must beat it under the [promotion procedure](benchmarking.md).
