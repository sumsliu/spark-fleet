# Benchmarking and promotion

Use the same exact model and weight revision, runtime identities, prompt fixtures, reasoning policy, output conditions, cache state and concurrency on both versions. Save those identities next to receipts. Never compare a warm candidate with a cold baseline.

## Metrics

- **First text latency:** request start to first non-empty text delta. Headers, role-only deltas and SSE keepalives are not first text. Tool-call latency should be reported separately.
- **End-to-end throughput:** generated output tokens divided by total request/batch wall time, including prefill and queueing.
- **Decode throughput:** output tokens over an explicitly defined decode interval. State treatment of the first token.
- **Aggregate throughput:** sum of output tokens over a concurrent batch's wall time. It is not single-stream speed.
- **Cache reuse:** backend-reported cached tokens divided by actual prompt tokens, with the exact field names and semantics saved. A short response alone is not proof of a hit.
- **Tail behavior:** per-request TTFT, stalls, errors, cancellation recovery and cold-request cost while warm traffic competes.

## Bounded benchmark command

```sh
sparkfleet benchmark --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --phase A0 --concurrency 1 --repetitions 2 --tokens 256 --out work/A0.json
```

The CLI uses six public authored prompts and saves prompt hashes, usage, first-text timing and aggregate end-to-end throughput. It does not store generated answers. `--fixed-output` adds runtime-specific `ignore_eos`; verify support before using it. Natural EOS and fixed output answer different questions and cannot be mixed in one comparison. The built-in prompts are not a comprehensive quality benchmark or a 140K tool-chain workload.

Repeat with a candidate as B0, another candidate run as B1, then restore the exact baseline as A1. Version changes must follow your own protected maintenance workflow. This command does not switch services.

```sh
sparkfleet compare --baseline work/A0.json --candidate work/B0.json \
  --recheck work/A1.json --quality-receipt work/quality.json --out work/comparison-B0.json
sparkfleet compare --baseline work/A0.json --candidate work/B1.json \
  --recheck work/A1.json --quality-receipt work/quality.json --out work/comparison-B1.json
```

Each comparison pairs identical cases and uses the geometric mean of A0/A1 as its baseline. It reports geometric gain and a seeded 10,000-resample paired-case 95% interval. A stable-positive flag requires completed exact-model API acceptance, a positive lower interval and at most 3% aggregate baseline drift. **It never promotes a runtime automatically.** This is a case bootstrap, not independent repeated hardware evidence.

B0 and B1 should both survive the check. Inspect case-level regressions, order/thermal effects and A0/A1 drift even if the summary is positive. Save independent trials when a gain is small. A 7.5% reference drift is enough to invalidate a small claimed improvement; more repetitions of the same biased order do not fix the cause.

## Long-context and contention protocol

Use synthetic or explicitly shareable fixtures; never export users' real projects.

1. Tokenize/count the actual prompt. Test cold ~32K, ~140K and near-window cases with output headroom.
2. Generate at least 1,024 output tokens, then repeat the original request and append a genuine assistant/tool round. Record cached-token usage and first-text/tool timing every turn.
3. Run an already-decoding request; inject a large cold input; six seconds later inject a mostly cached request. Record all three timelines and cold-alone control.
4. Repeat at intended concurrency. Record cache capacity/eviction, preemption, errors and longest inter-token stall.
5. Run semantic correctness, JSON, forced and automatic tools, reasoning policy and cancellation on both versions.
6. Test the expected long output, not just acceptance of a `max_tokens` value. Validate natural termination, context accounting, sustained resource use and quality.

The built-in long qualifier supplies a bounded three-turn recall probe and usage records. A real long tool-chain fixture and competitive load remain explicit acceptance work; do not label a smoke receipt “full certification.”

## Promotion record

Save candidate and baseline image/source/weight hashes, host/boot identities privately, guard readings, unchanged variables, paired outputs, quality evaluation, rollback recipe and measured limitations. Keep the fastest **verified** version. If a candidate fails or its benefit is uncertain, restore that recipe with a fresh plan and record the reason. See [operations.md](operations.md).
