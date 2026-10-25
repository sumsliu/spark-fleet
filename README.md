<p align="center"><img src="assets/banner.svg" alt="SparkFleet: reproducible multi-node inference on DGX Spark" width="960"></p>

<p align="center">
<a href="https://github.com/sumsliu/spark-fleet/actions/workflows/ci.yml"><img src="https://github.com/sumsliu/spark-fleet/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache 2.0"></a>
<img src="https://img.shields.io/badge/status-v0.1_alpha-orange" alt="v0.1 alpha">
<a href="README.zh-CN.md">简体中文</a>
</p>

# Make multi-node Spark inference repeatable

**SparkFleet turns hard-won DeepSeek and GLM deployment experience into reusable recipes, source-locked patches, API checks and measured optimization workflows.** Start with a topology you can review. Measure the behavior people actually notice: long-conversation latency, tool reliability and concurrency. Keep evidence for every change and a path back to the fastest verified version.

Built around Linux ARM64 **DGX Spark / GB10 / SM121**. Reference profiles cover **DeepSeek V4.1 Flash on four nodes** and **GLM 5.3 on eight nodes**, with native model-specific weight and KV formats, OpenAI-compatible APIs and Anthropic Messages integration.

**v0.1 is an alpha toolkit.** The exported overlays and tuning records come from an existing deployment; this portable controller has CPU fixture tests, not a fresh multi-node GPU certification. Compatible runtime images and separately licensed weights are prerequisites. Stock vLLM is not a drop-in substitute. Read [runtime compatibility](docs/runtime.md) before starting GPUs.

## Why this project exists

A model can pass a short chat test and still feel unusable during real development. Our recurring problems were full-prefix recomputation, cold inputs monopolizing the token budget, protocol mismatches, misleading model menus and performance gains that disappeared across nodes.

SparkFleet makes those problems visible and reproducible:

- **Recipes with context.** TP4 and TP8 profiles keep their distinct speculative decoding, attention, quantization and fabric settings.
- **Long-conversation first.** Prefix retention and contention-aware prefill references explain why warm turns can matter more than a headline decode rate.
- **Evidence before promotion.** Paired benchmarks include baseline drift, uncertainty and a separate correctness gate. A faster microkernel does not automatically win an end-to-end comparison.
- **Reviewable operations.** Read-only inventory, digest-bound plans, four shared locks, temperature/memory guards and exact-owned container cleanup.
- **Honest APIs.** An optional mTLS gateway exposes only models and routes covered by completed acceptance receipts. Disabled models stay out of discovery.
- **Useful failures.** Rejected Engram, adaptive draft and parallel-prefix experiments are documented alongside successful changes.

## Historical reference results

These are **selected measurements from the source deployment**, not results reproduced by installing this release. Different rows use different workloads and must not be combined into one speed claim.

| Workload | Before → after | What changed |
|---|---:|---|
| Repeated ~25K-token input after 1,024 generated tokens | ~10 s → ~0.5 s; cache hit 0 → 99.5% | Prefix retention every 1,024 tokens |
| Warm request arriving behind a ~105K cold input | 40.63 s → 4.99 s first token | Contention-aware prefill; cold request 45.73 → 46.26 s in that probe |
| Twelve paired natural-EOS cases | 48.89 → 51.08 weighted decode token/s | Tail-v3 MoE optimization; paired geometric gain 4.56%, 95% interval 1.92–7.21% |
| Warm 32K input, fixed 768-token output, four concurrent requests | 85.70 → 92.77 **aggregate** token/s | Same tail-v3 comparison; not single-stream throughput |

See [measurement provenance](benchmarks/reference/README.md), [structured results](benchmarks/reference/results.json) and [benchmark methodology](docs/benchmarking.md). **100 token/s single stream remains an aspiration, not an achieved project claim.** A configured 256K context and 32K output budget also require memory headroom and workload-specific validation.

## Try the planning workflow

Python 3.11+ on your controller; Linux ARM64 Spark nodes are needed only for actual model execution. The CLI core uses the Python standard library.

```sh
git clone https://github.com/sumsliu/spark-fleet.git
cd spark-fleet
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

sparkfleet init --profile deepseek-tp4 --out cluster.json
sparkfleet validate cluster.json --template
sparkfleet render cluster.json --template --out work/preview
```

This creates a placeholder topology and **review-only** commands. It does not contact or modify a cluster. For an eight-node GLM topology, use `--profile glm-tp8`.

Then follow the [deployment quickstart](docs/quickstart.md) to supply your own hosts, compatible image digest, weights, fabric mapping and local paths. `doctor` observes; `plan` binds identities; only `run ... --execute` starts the selected release. Keep the controller running for the lifetime of that release.

## Find your starting point

| You want to… | Start here |
|---|---|
| Deploy on a new fleet | [Quickstart](docs/quickstart.md) · [runtime prerequisites](docs/runtime.md) · [model recipes](docs/recipes.md) |
| Understand the optimizations | [Inference timeline and lessons](docs/optimization.md) · [reference overlays](overlays/README.md) |
| Check 200G networking properly | [Fabric measurement](docs/network.md) |
| Validate real API behavior | [Acceptance matrix](docs/acceptance.md) · [benchmarking](docs/benchmarking.md) |
| Connect a client or remote Mac | [API and client integration](docs/clients.md) · [gateway](docs/gateway.md) |
| Upgrade or recover safely | [Operations and rollback](docs/operations.md) · [troubleshooting](docs/troubleshooting.md) |
| Improve SparkFleet | [Contributing](CONTRIBUTING.md) · [roadmap](ROADMAP.md) · [architecture](docs/architecture.md) |

## Help make this portable

Have a Spark cluster? **An independent reproduction is the most valuable contribution.** Share sanitized topology, exact runtime and weight identities, warm/cold conditions, paired latency and throughput, correctness results and failures. Negative results help the next operator avoid the same dead end.

Good first contributions include a verified runtime build recipe, a new cluster reproduction, a long-context tool-call fixture, and per-rail RDMA/NCCL reports. Use the [benchmark report](https://github.com/sumsliu/spark-fleet/issues/new?template=benchmark.yml) or open a [discussion](https://github.com/sumsliu/spark-fleet/discussions). Include enough detail that someone else can repeat your result.

## License and attribution

SparkFleet is Apache-2.0. Vendored reference overlays retain upstream notices and modified-file markers; [NOTICE](NOTICE) and [the overlay manifests](overlays/README.md) describe provenance. Model weights, CUDA components and desktop applications retain their own terms and are not included. This independent project is not endorsed by NVIDIA, DeepSeek, Z.ai, Anthropic or the vLLM project.
