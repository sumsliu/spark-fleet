# SparkFleet v0.1.0 — public alpha

Reusable multi-node DGX Spark inference tooling drawn from DeepSeek TP4 and GLM TP8 deployment experience.

This first release includes model-specific profiles, read-only inventory, identity-bound plans, guarded exact-owned release supervision, licensed fingerprint-locked reference overlays, bounded API qualification, paired benchmark/drift checks and an optional mTLS gateway with acceptance-gated model discovery.

Start with [the README](../README.md), [中文说明](../README.zh-CN.md) or [the quickstart](quickstart.md). Planning commands run without a GPU cluster. Compatible SM121 runtime images and separately licensed weights are prerequisites for execution.

Historical results are labeled by workload and provenance. The portable control plane is covered by CPU fixtures and local TLS tests; it has not been newly GPU-qualified on a fresh fleet. No model binaries, client bundles, credentials, private machine identities or original private repository history are distributed. This is an alpha toolkit, not a one-command stock-vLLM install or a 100 token/s single-stream claim.

Contributions most needed: independent reproductions, complete pinned runtime build recipes, long-context tool-chain fixtures and per-rail RDMA/NCCL measurements. Please use the issue templates and include negative results.
