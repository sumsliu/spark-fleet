# Fingerprint-locked reference overlays

These files are historically tested source derivatives exported with license notices, modified-file markers and payload hashes. **The portable project has not freshly GPU-qualified them.** Full compatible model runtimes are separate prerequisites; see [runtime.md](../docs/runtime.md).

| Overlay | Contents | Boundary |
|---|---|---|
| `deepseek-reference-v1` | B12X tail-v3 MoE, contention-v4 scheduler, inline-system Anthropic fix, DeepSeek Responses tokenizer fix | Four files; does not include the entire native SM121/Engram/DSpark runtime |
| `glm-native-ttl-v6` | Bounded native prefix TTL/lease implementation and vLLM integration | Sixteen files; does not include GLM weights, Marlin/MTP runtime or client UI |

Each manifest records the published file hash, reference source hash, original runtime preimage where available, license and historical qualification scope. Existing files must match the preimage exactly. New files require absence. There is no force-copy mode.

```sh
sparkfleet overlay-inspect overlays/deepseek-reference-v1
sparkfleet overlay-materialize overlays/deepseek-reference-v1 \
  --runtime-root /path/to/exported/site-packages --out work/reference-overlay
```

Materialization creates a new directory plus a file-mount list. It does not write into an installed Python tree. Check targets against the actual image's Python package layout, stage identical files on every target node, and include hashes in the private topology.

When porting to another runtime, produce a new overlay name/manifest, retain attribution and describe changes. Review the entire behavioral diff and run numerical/API/long-context/concurrency tests before collecting promotion data. Never combine the two model overlays just because both contain a `scheduler.py`.
