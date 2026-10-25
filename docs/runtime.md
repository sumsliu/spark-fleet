# Runtime compatibility and provenance

SparkFleet's Python CLI is a control plane. It is not a replacement for vLLM, CUDA, NCCL, a model loader or the SM121 kernel stack. No validated public all-in-one runtime image is shipped with v0.1. A sample image name or a local Docker ID is not a downloadable artifact.

## DeepSeek native TP4 prerequisites

The source deployment used native MXFP4 expert weights with FP8 components, DSpark speculative decoding, B12X MoE, and an exact disk-backed Engram loader on four 128 GB GB10 nodes. Engram and the total checkpoint are too large to treat all tensors as ordinary resident GPU allocations at this topology. Preserve the native format; this recipe does not convert to EXL3.

Public upstream starting points, checked during this export:

- [Tech2wild Spark TP2/TP4 recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks), MIT, reference commit [`fb1bfdfa4760759c31fb976350a79b76d14f17db`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks/tree/fb1bfdfa4760759c31fb976350a79b76d14f17db).
- [vLLM source reference](https://github.com/vllm-project/vllm/commit/e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba), Apache-2.0. Pin the commit; a development branch name can disappear or move.
- [B12X](https://github.com/local-inference-lab/b12x), Apache-2.0, historical installed version 1.3.0. The repository was archived/moved toward FlashInfer at export time. Migration needs a new comparison; do not assume a current package has the historical ABI.
- The historical starting container tag was `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c`. Resolve a real registry digest and verify its availability; this tag alone is not a complete native runtime.

The compatible image must contain all of the following:

1. A matching vLLM Python tree and `_C_stable_libtorch` extension rebuilt for SM121a/SM12x.
2. The matching FlashInfer/CUTLASS and native MXFP4/MXFP8 paths, sparse MLA/indexer and page-layout support.
3. DeepSeek V4.1 model, tokenizer, reasoning and tool parsers, DSpark and exact disk Engram patches.
4. B12X with the expected MoE interface and the tested WO_A FP8/BF16 path.
5. Compatible CUDA, PyTorch, NCCL, host driver and ARM64 ABI versions.
6. The source fingerprints in the [DeepSeek overlay manifest](../overlays/deepseek-reference-v1/manifest.json), or an explicitly reviewed port with fresh acceptance evidence.

The four exported DeepSeek overlay files add tail-v3, contention-aware prefill, inline-system and Responses tokenizer fixes. They **do not supply the other native-runtime prerequisites above**. They are reference source, not a full engine image.

Build on an isolated node with bounded CPU compilation concurrency (`MAX_JOBS=2` was useful in the source environment), then prewarm the exact attention/graph/kernel combinations. Massive first-start JIT parallelism can exhaust unified memory. A successful wheel installation does not prove that kernels run on SM121.

Review upstream scripts before using them: historical examples may assume a particular NFS mount, addresses or permission to stop an existing fleet. Those are deployment-specific and are not inherited by this project.

## GLM TP8 prerequisites

The observed GLM runtime is a separate model-specific stack: NVFP4 weights, Marlin main MoE, FP8 MLA KV (`fp8_ds_mla`), MXFP4 indexer KV and MTP3 with a Triton draft MoE path. It used explicit multi-HCA mappings and host NCCL dependencies. No redistributable complete GLM image is supplied here.

Before confirming compatibility, check that your image implements every flag in `glm-tp8`, the `glm47` parser pair and your checkpoint's architecture and tensor layout. Runtime-specific projection/shared-sum/draft flags and NCCL library mounts need to be reproduced in a compatible image or explicit local environment; the generic profile cannot infer them from a model name.

The [native TTL v6 overlay](../overlays/glm-native-ttl-v6/manifest.json) is independently fingerprint-locked. It is a cache-lease extension, not the NVFP4 runtime or model weights. Existing files require exact preimages and new extension files must not already exist.

## Using overlays without silent overwrites

```sh
sparkfleet overlay-inspect overlays/deepseek-reference-v1
sparkfleet overlay-inspect overlays/deepseek-reference-v1 \
  --runtime-root /path/to/exported/site-packages
sparkfleet overlay-materialize overlays/deepseek-reference-v1 \
  --runtime-root /path/to/exported/site-packages --out work/deepseek-overlay
```

Export the actual image's Python package tree into an isolated directory, without GPU access, for preimage verification. `materialize` creates a new directory and `mounts.json`; it never patches the installed runtime. The generated targets assume the historical Python 3.12 `/usr/local/lib/python3.12/dist-packages` layout. Adjust targets only after verifying your image's import paths and requalifying it.

Stage the identical overlay payload at the specified source paths on every node, add the entries to your private config's `mounts`, and retain the manifest. Do not mix the DeepSeek and GLM overlays in one Python tree. A preimage failure means **port and requalify**; there is no force option.

## Release identity checklist

Keep an operator-private record of base registry digest, resulting local image ID per node, source commit, build commands, Python/CUDA/PyTorch/FlashInfer/B12X/NCCL versions, overlay manifest, weight revision/SHA256, topology, fabric mapping and test receipts. Record failed builds as well. No original deployment's one-use approvals or machine identities are part of this release.
