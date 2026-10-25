# Model reference recipes

These are historically observed operating points. They require the compatible images in [runtime.md](runtime.md) and fresh acceptance on your hardware. They are not guaranteed optimal settings for every topology.

| Parameter | DeepSeek TP4 | GLM TP8 |
|---|---|---|
| Exact API model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `glm-5.3` |
| Nodes / tensor parallel | 4 / 4 | 8 / 8 |
| Architecture | GB10 SM121, Linux ARM64 | GB10 SM121, Linux ARM64 |
| Weights | Native MXFP4 / FP8; disk Engram | Model-specific NVFP4 |
| Context limit | 262,144 total tokens | 262,144 total tokens |
| Output policy budget | Up to 32,768 within remaining context | Up to 32,768 within remaining context |
| Max sequences | 4 | 8 |
| Batched tokens | 4,096 | 8,192 |
| Memory utilization | 0.80, with explicit 9 GiB KV budget | 0.72 |
| KV / indexer | Native DeepSeek runtime | `fp8_ds_mla` / `mxfp4` |
| Speculation | DSpark 5, probabilistic draft, block rejection | MTP 3, local argmax reduction |
| Adaptive verification | Off | Off |
| Main MoE | B12X, tail-v3 reference | Marlin; draft MoE Triton |
| Tool / reasoning parser | `deepseek_v41` | `glm47` |
| Prefix retention | Every 1,024 tokens | Runtime cache policy; optional native TTL v6 |
| Modality | Bounded synthetic image API check; full semantic vision requires separate evaluation | Text only in this recipe |
| Fabric reference | NCCL channels 8, cross-NIC off | Channels 2, cross-NIC on |

The renderer emits the full argv from [profiles.py](../src/sparkfleet/profiles.py). Head API binds loopback, workers are headless, weights are read-only and kernel caches are separate. The configured output budget is a client policy, not a dedicated KV allocation or a server flag overriding every request.

## Long context is a shared budget

Input, generated reasoning, final answer and protocol overhead consume the same context. A request with 256K input cannot also request an extra 32K output inside a 256K window. Larger KV capacity and prefix leases compete with concurrency. The reference GLM TTL capacity of 196,608 tokens is a lease policy, not a promise that eight full 256K conversations can coexist.

Start acceptance at bounded low reasoning effort, then explicitly test higher effort and long outputs for your workload. GLM effort selection is supported; a low-only test policy does not mean the model must be permanently locked to low.

## Choosing a batch budget

DeepSeek retained 4,096 to preserve long-context concurrency headroom. A larger prefill budget may shorten cold input latency while delaying decode or reducing available KV. The contention-v4 overlay limits a competing long prefill to 2,048 at the scheduling point while retaining the full budget when alone. It also reserves at least 64 tokens for following decode requests. See [optimization.md](optimization.md) for the measured tradeoff.

## Native prefix TTL

GLM's optional v6 overlay implements bounded native prefix leases with explicit admission/quota behavior and expiry. Best-effort prefix reuse and guaranteed lease semantics are different. A quota rejection is a real 429, not model overload. Process restart clears in-memory leases. Verify reuse, actual expiry, capacity exhaustion and post-restart behavior before enabling a TTL contract for clients.

## Fabric values must be measured

Interface names, RDMA devices and GID indices are not portable identifiers. Different nodes may enumerate the same physical port differently. Use `nodes[].fabric` overrides after checking each node. A reference value of GID 3 is not a fleet-wide truth. Do not copy GLM's multi-rail/channel settings into DeepSeek simply because both use Spark hardware.
