# Measuring a 200G Spark fabric

A 200 Gbit/s link has **25 GB/s raw line rate per direction**. Application payload is lower. Two ports or two directions must not be added together and described as a single 200G path. A link LED, LLDP description or negotiated rate is not a throughput test.

## Inventory first

```sh
sparkfleet network-plan cluster.json --out work/network-plan.json
```

The generated plan lists read-only `ip`, `ethtool`, FEC, `ibv_devinfo`, `rdma link` and `show_gids` commands. Inspect them on each selected node. Check physical port to netdev to HCA mapping, link speed, MTU, IPv4 RoCE GID, routing, NUMA/CPU affinity where applicable, error and drop counters. Some tools require installation or elevated read permission; doctor does not install them.

A switch login is not necessary to begin host-side tests. It can become necessary to explain port errors, flow control, oversubscription or mismatched switch configuration. Host tuning cannot prove or repair an unknown switch configuration by itself.

## Test one pair, both directions, each rail

The plan includes explicit one-off `iperf3` server and parallel client examples. Run traffic only in an authorized quiet window; saturating a shared fabric can affect production even when no models restart.

1. Measure a single rail in each direction with received-payload JSON output; retain duration, CPU utilization and counters before/after.
2. Repeat for every relevant node pair or at least every distinct path and hardware port.
3. Measure simultaneous rail use separately. Report each rail and total; verify routing really distributes traffic.
4. Parse receiver payload with `sparkfleet network-result received.json`. It reports Gbit/s, GB/s and fraction of 200G, not GPU collective performance.

Do not force MTU, ECN/PFC, FEC, link speed, CPU governors or switch settings without understanding the whole path and preserving recovery settings.

## TCP, RoCE and NCCL answer different questions

| Test | Establishes | Does not establish |
|---|---|---|
| Negotiated speed / counters | Link state and errors | Sustained useful bandwidth |
| `iperf3` | CPU TCP payload for the tested path | GPU direct RDMA or collectives |
| RDMA perftest | RDMA transport for specified HCA/GID/memory settings | All-rank model communication |
| `nccl-tests` | GPU collective latency/bandwidth for specified ranks and sizes | End-to-end model tokens per second |
| Model paired workload | User-visible effect under that workload | Universal fabric capacity |

For RDMA and NCCL, record exact tool versions, GPU-memory mode, GID, HCA, message sizes, rank count, direction and NCCL transport logs. Check that NCCL actually uses IB/RoCE rather than silently falling back to sockets. Test small-message latency as well as large transfers: decoding collectives may never fill a 200G link.

DeepSeek's reference channels/cross-NIC policy and GLM's policy differ. Test their intended topology; do not assume the setting that wins for eight ranks also wins for four. This release contains no newly reproduced claim that any specific fleet reaches 200G payload throughput.
