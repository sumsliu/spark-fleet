# Troubleshooting

| Symptom | Check first | Response |
|---|---|---|
| Catalog lists only one model | Exact enabled policy and receipt coverage | Complete acceptance for the other backend, then atomically enable it |
| 503 “pending acceptance” | Model policy; actual exact-model receipt | Do not fake an alias or list an unqualified model |
| “Invalid request” in a client | HTTP status/type, endpoint and model ID, tool/Responses envelope | Reproduce a bounded API request; do not restart the model by default |
| “Overloaded” or retries | Real upstream status, queue, concurrency and classifier status | Distinguish true load, timeout, protocol error and client permission classification |
| Long chat recomputes everything | Stable prefix, retention setting, cache pressure and backend cached-token usage | Reproduce repeat/append/tool loops; record usage rather than guessing from timing |
| Warm request waits behind cold input | Per-step token scheduling and actual overlay hash | Verify contention-v4 is present in the running runtime; test cold-alone control |
| Slow tokens during a long prefill | Chunk compute and inter-token stalls | Reduced TTFT does not remove GPU contention; measure the whole timeline |
| Patch version changed but speed did not | Actual bytes, preimages, imports and container mounts | Labels are not evidence that a patch was applied |
| Old restore loses a successful fix | Best-verified full recipe and recovery source | Preserve retention, scheduler and all overlays in the recovery recipe |
| GLM child agent runs DeepSeek | Client child model selection | Inherit the current parent, never catalog index zero |
| Claimed child task has no progress | Actual tool return and task/session ID | Treat prose-only dispatch as unconfirmed; fix client orchestration separately |
| Network fetch blocked in Auto | Client policy/sandbox and permission-classifier result | Inference server settings do not control application OS/network permissions |
| Certificate error at home | CA, DNS SAN, clock, expiry, client fingerprint and key | Fix enrollment/configuration; keep verification enabled |
| Unknown SSH host key | Existing verified host-key trust | Verify with the owner; do not auto-delete or bypass known_hosts |
| “Existing workload/lock present” | Current service owner and shared lock directory | Drain and coordinate; never kill unrelated work to make admission pass |
| Incident fence | Saved fault, exact stop, resource readings | Investigate/archive evidence and prepare a fresh scope after recovery |
| No 200G payload | Rail mapping, counters, receiver CPU, RoCE/NCCL actual transport | Separate line negotiation, TCP, RDMA and collective tests |

Before opening an issue, run the CPU tests, state the exact commit and sanitized environment, and include the smallest repeatable request. Report errors as errors. Do not publish credentials, cert/key files, private addresses, raw projects or Docker environment dumps.
