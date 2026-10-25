# Historical measurement provenance

`results.json` is a manually curated, sanitized summary of records from the original private deployment. No raw conversations, host addresses, machine identities, credentials or original repository history are included. These aggregates help formulate a hypothesis and avoid repeating rejected ideas; they are not raw independently auditable benchmark traces.

Source record families reviewed during the export:

| Record family | Material retained here |
|---|---|
| Prefix-cache root cause / retention rollout, 2026-10-21 | Repeated/appended ~25K synthetic cache and timing results |
| Contention prefill v4, 2026-10-21 | Warm/cold/decode timelines and the cold-input tradeoff |
| Tail-v3 four-node paired validation | Twelve paired natural-EOS cases and separate fixed-output concurrency result |
| Engram overlap / adaptive verification / parallel prefix diagnostics | Negative or unresolved end-to-end results; no promotion claim |
| GLM home API qualification, 2026-10-25 | Counted synthetic long input, appended turns, tool-round and remote transport timing |

All numbers are workload-specific. Weighted and arithmetic means differ; aggregate and single-stream rates differ; completion wall time and first-token latency differ. The ~62 token/s recollection comes from a separate prompt set and averaging method. It must not be merged with the ~51 token/s paired test into a trend without matching cases.

The public package's initial validation is CPU/control-plane/TLS validation. We welcome independently reproduced GPU results with complete sanitized manifests and paired receipts. Use the [benchmark template](../../.github/ISSUE_TEMPLATE/benchmark.yml) and [methodology](../../docs/benchmarking.md). Do not upload private data to make the historical record “more complete.”
