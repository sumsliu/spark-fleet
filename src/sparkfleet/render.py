"""Render argv, never concatenate user fields into a shell command."""

import shlex
import re
from pathlib import Path
from .config import validate
from .common import save_json, digest

BASE_ENV = {
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_DEBUG": "WARN",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_CUDA_ARCH_LIST": "12.1a",
    "MAX_JOBS": "2",
    "FLASHINFER_NVCC_THREADS": "1",
    "VLLM_USE_RUST_FRONTEND": "0",
}


def commands(c, release, *, template=False):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", release):
        raise ValueError("Invalid release identifier")
    p = validate(c, template=template)
    rows = []
    for rank, n in enumerate(c["nodes"]):
        fabric = dict(c["network"], **n.get("fabric", {}))
        name = f"sf-{c['cluster']}-{release}-r{rank}"
        env = dict(
            BASE_ENV,
            **p["env"],
            VLLM_HOST_IP=n["data_ip"],
            NCCL_IB_HCA=fabric["hca"],
            NCCL_IB_GID_INDEX=str(fabric["gid_index"]),
            NCCL_SOCKET_IFNAME=fabric["socket_interface"],
            GLOO_SOCKET_IFNAME=fabric["socket_interface"],
            VLLM_CACHE_ROOT="/cache/vllm",
            TRITON_CACHE_DIR="/cache/triton",
            TILELANG_CACHE_DIR="/cache/tilelang",
        )
        engine = (
            c["runtime"]["entrypoint"]
            + [
                "serve",
                "/models/model",
                "--served-model-name",
                p["model_id"],
                "--tensor-parallel-size",
                str(len(c["nodes"])),
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                str(len(c["nodes"])),
                "--node-rank",
                str(rank),
                "--master-addr",
                c["nodes"][0]["data_ip"],
                "--master-port",
                str(c["network"]["master_port"]),
                "--gpu-memory-utilization",
                str(p["memory_utilization"]),
                "--max-model-len",
                str(c["serving"]["context"]),
                "--max-num-seqs",
                str(c["serving"]["max_sequences"]),
                "--max-num-batched-tokens",
                str(c["serving"]["batch_tokens"]),
            ]
            + p["flags"]
        )
        engine += (
            ["--host", "127.0.0.1", "--port", str(c["serving"]["port"])]
            if rank == 0
            else ["--headless"]
        )
        argv = [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--label",
            "org.sparkfleet.managed=true",
            "--label",
            f"org.sparkfleet.cluster={c['cluster']}",
            "--label",
            f"org.sparkfleet.release={release}",
            "--network=host",
            "--ipc=host",
            "--gpus=all",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--volume",
            c["paths"]["weights"] + ":/models/model:ro",
            "--volume",
            c["paths"]["cache"] + ":/cache",
        ]
        for m in c.get("mounts", []):
            argv += ["--volume", m["source"] + ":" + m["target"] + ":ro"]
        for dev in c["network"]["rdma_devices"]:
            argv += ["--device", dev]
        if c["paths"].get("env_file"):
            argv += ["--env-file", c["paths"]["env_file"]]
        for k, v in sorted(env.items()):
            argv += ["--env", k + "=" + v]
        # Override any image entrypoint, so argv has the same meaning in every image.
        argv += ["--entrypoint", engine[0], c["runtime"]["image"]] + engine[1:]
        rows.append(
            dict(
                rank=rank,
                node=n,
                name=name,
                argv=argv,
                engine_argv=engine,
                env=env,
                expected_mounts=c.get("mounts", []),
                paths=c["paths"],
                guard=c["guard"],
                cluster=c["cluster"],
                release=release,
            )
        )
    return rows


def render(c, out, release, *, template=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = commands(c, release, template=template)
    for row in rows:
        save_json(out / (row["node"]["name"] + ".json"), row)
        # Review aid only: direct use would bypass source-bound supervision.
        (out / (row["node"]["name"] + ".preview.sh")).write_text(
            "#!/bin/sh\n# REVIEW ONLY. Run `sparkfleet run plan.json` for guarded execution.\n# "
            + shlex.join(row["argv"])
            + "\n"
        )
    save_json(
        out / "manifest.json",
        dict(
            schema=1,
            release=release,
            preview_only=template,
            files={
                p.name: digest(p.read_bytes()) for p in out.iterdir() if p.is_file()
            },
        ),
    )
    return rows
