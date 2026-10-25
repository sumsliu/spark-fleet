# Deployment quickstart

This is a controlled deployment workflow for a fleet you administer. Plan and doctor never stop an existing workload. v0.1 does not install GPU drivers, download weights or build a compatible runtime automatically.

## 1. Prepare the controller

```sh
python3 --version  # 3.11 or newer
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
sparkfleet profiles
sparkfleet init --profile deepseek-tp4 --out cluster.json
```

Use `glm-tp8` for eight GLM nodes. Keep the generated `cluster.json` private; it is ignored by Git. Inspect [examples](../examples/) before editing.

## 2. Prepare each node

Requirements: Linux ARM64 GB10, an authenticated noninteractive SSH path with verified host keys, NVIDIA Container Toolkit, working Docker access, and a **compatible pinned runtime** as described in [runtime.md](runtime.md). Docker administration is privileged access; use only the operator account authorized for your fleet.

Set these fields in your private config:

| Field | What to supply |
|---|---|
| `nodes[].ssh` | Your existing `operator@host`; normal SSH config and known-host verification apply |
| `nodes[].data_ip` | The node's measured fabric IPv4 address, not its management address by assumption |
| `runtime.image` | A registry `image@sha256:…` or an already loaded local `sha256:…` image ID |
| `runtime.entrypoint` | Compatible engine executable argv, normally `["vllm"]`; a runtime-specific wrapper can be explicit |
| `runtime.compatible_runtime_confirmed` | Set true only after matching the runtime prerequisites and overlay fingerprints |
| `paths.weights` | Identical local path to the correct weight tree on every node; mounted read-only |
| `paths.cache` | Existing writable JIT/cache directory; not the model directory |
| `paths.state` | Existing owner-writable private state directory shared by all operators' locks on that node |
| `network.*` | Measured interface, HCA selection and GID index; per-node overrides live in `nodes[].fabric` |
| `mounts` | Optional file overlays with exact source SHA256 and absolute container target |

An operator can prepare directories on each selected node with `install -d -m 700 /srv/sparkfleet /srv/sparkfleet/cache` when that account owns the parent, or through their existing authorized administration process. Do not change ownership of unrelated trees. Install weights separately and retain their license, revision and manifest.

```sh
sparkfleet weights-manifest /your/local/model-copy --out work/weights.json
sparkfleet weights-verify /your/local/model-copy work/weights.json
```

These commands read every model file and can take time. Verify each node's actual copy. No model conversion is performed. A fast `ls` is not a provenance check.

Private optional `paths.env_file` must be a regular file owned by the SSH operator with permissions 0600. It is node-local, hashed rather than printed, and never copied into this repository. Use it for runtime-specific environment dependencies only; public gateway device keys do not belong in model containers.

## 3. Validate and observe

```sh
sparkfleet validate cluster.json
sparkfleet render cluster.json --out work/review
sparkfleet network-plan cluster.json --out work/network-plan.json
sparkfleet doctor cluster.json --out work/doctor.json
```

`render` creates commented shell previews and JSON argv; do not bypass the controller by uncommenting them. `doctor` makes read-only SSH probes of GPU temperature, available memory, image identity, overlay hashes, existing locks and compute processes. It does not create a release, take over containers or cancel users.

Check the [network procedure](network.md), [model recipe](recipes.md) and [guard contract](operations.md). A GPU already serving requests is an existing workload; arrange a drain and maintenance window through its owner. Do not delete locks to make a test pass.

## 4. Create a fresh plan and launch

```sh
sparkfleet plan cluster.json --out work/plan-new.json
sparkfleet run work/plan-new.json --receipts work/run-new --execute
```

The plan expires in one hour and binds boot IDs, local image identities, overlay and optional env-file hashes, argv and node-agent source. It is a content-integrity record, **not** a cryptographic authorization signature. Operator authorization and authenticated SSH remain prerequisites.

`run` stages a fresh bundle, rechecks all nodes, holds all four locks per node, starts only the selected ranks and monitors their state. Keep this foreground controller alive, for example inside an operator-managed `tmux` session. Closing its control channel intentionally stops this release. Do not use `nohup` as a substitute for understood service supervision. Distributed startup may take many minutes; per-node guard messages continue while the engine loads.

The API binds **loopback on the head only**. Workers use `--headless`. Model readiness and correctness are separate from Docker creation.

## 5. Accept locally, then publish routes

Run acceptance from the head node or over your existing private SSH tunnel. For an authenticated native API, read its key from a private file; an unauthenticated loopback runtime may ignore a fixture token, which does not make it safe to expose remotely.

```sh
export SPARKFLEET_API_KEY="$(cat /path/to/private-api-key)"
sparkfleet qualify --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --out work/quality.json
```

Read [acceptance.md](acceptance.md) for long-context, tool-chain, cancellation, output and quality coverage. A bounded smoke receipt is necessary but not sufficient for production. Enable the [gateway](gateway.md) only after completing the required workload matrix and saving receipts. Use [benchmarking.md](benchmarking.md) before promoting a performance change.

## 6. Recover or upgrade

Ctrl-C requests release cleanup. Check `work/run-new/STOPPED.json`: `all_exact_stops_confirmed` must be true. If false, inspect the exact container identity and node incident fence before another launch. Reuse the **recipe and image**, not an old one-use plan. Rollback always receives a new plan and release identity; see [operations.md](operations.md).
