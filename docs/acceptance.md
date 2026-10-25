# Acceptance matrix

A model listing is discovery. A successful `/v1/models` call does not establish generation, tools, reasoning, cache reuse or a usable client. Qualify each exact model and intended transport before exposing it.

## Built-in bounded qualifier

```sh
export SPARKFLEET_API_KEY="$(cat /path/to/private-api-key)"
sparkfleet qualify --base-url http://127.0.0.1:8000/v1 \
  --model glm-5.3 --out work/glm-quality.json --long-tokens 140000
```

Without `--long-tokens`, the suite runs ten OpenAI checks and an Anthropic Messages/count check. `--openai-only` restricts the declared route capabilities. All reasoning probes use low effort to bound load; higher selectable efforts need their own workload acceptance.

| Check | What must happen |
|---|---|
| Catalog | Exact requested model ID appears |
| Arithmetic | Exact model returns the expected answer |
| JSON | Valid JSON with expected fields/values |
| Tool round trip | Correct function, parseable arguments, matching call ID and result consumed |
| Stream | Real text, valid JSON deltas and actual `[DONE]` |
| Reasoning | Separate reasoning and final fields with expected answer |
| Modality | GLM text-only input rejects images with 400; DS synthetic image completes |
| Concurrency | Two distinguishable requests return their own answers |
| Overflow | Impossible output/context budget returns 400 |
| Cancel/recover | Close after real text, then successfully issue another request |
| Messages/count | Anthropic message shape, exact model, actual positive count |
| Optional long recall | Count synthetic input, recall a marker over three appended turns, retain usage/timings |

Reports are written with private permissions after each test. A crashed suite remains `complete:false`; it cannot enable the gateway. Reports omit prompt/answer bodies, auth headers and key values. The long probe records actual counted tokens and **completion wall time**, not streamed TTFT. Use the contention protocol for TTFT claims.

## Additional production gates

The built-in suite is intentionally bounded. A deployment still needs:

- A real 130–140K synthetic **tool chain**, several consecutive assistant/tool-result cycles, measured cache usage and first tool/text delta. Keep tool schemas and stable prefix ordering identical across turns.
- Near-window input with output headroom; 8K and 32K actual output where required. Accepting `max_tokens:32768` is not proof of generating that length reliably.
- Cold input plus warm traffic, intended concurrency, sustained use, cancellation reclamation and memory/temperature observations.
- Quality evaluation for prose, code, math, JSON, reasoning and your real task distribution. The small arithmetic test is not a model-capability score.
- Native TTL reuse, actual expiry, quota rejection and restart clearing if your client promises cache leases.
- Each published protocol and transport, including TLS/device credentials and the client-side model mapping. Responses is supported by the generic forwarding/normalizer code but is **not certified by the default qualifier** and must not be added to an enabled model's routes without separate receipt coverage.
- Desktop/CLI behavior if you integrate a client: model switch, effort choice, child model inheritance, tools, cancellation and restart/upgrade recovery.

Do not invent a success receipt or copy a receipt from a different runtime. The optional gateway verifies exact model and declared endpoint coverage, but only the operator can bind that evidence to the correct image/weights and complete the wider matrix.

## Cache evidence

Keep the server's actual `usage` structure. Field names and cache semantics differ by runtime. Distinguish total input tokens, cached tokens, lease creation/read tokens and generated tokens. If usage omits cache counts, correlate with runtime counters in an isolated diagnostic window; do not infer a percentage from latency alone.

The public tree contains no raw production conversations or historical machine identities. [Historical aggregates](../benchmarks/reference/README.md) are context, not fresh acceptance for your cluster.
