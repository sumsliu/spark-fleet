"""Observed reference knobs, not a promise of stock-vLLM compatibility."""

import copy

PROFILES = {
    "deepseek-tp4": {
        "model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "nodes": 4,
        "context": 262144,
        "output_budget": 32768,
        "batch_tokens": 4096,
        "max_sequences": 4,
        "memory_utilization": 0.8,
        "weight_format": "native MXFP4 experts / FP8; exact disk-backed Engram",
        "requires": [
            "SM121-native DeepSeek V4.1 runtime",
            "DSpark",
            "B12X",
            "disk Engram loader",
        ],
        "flags": [
            "--block-size",
            "128",
            "--engram-config",
            '{"cpu_offload":false}',
            "--default-chat-template-kwargs",
            '{"thinking":false}',
            "--tool-call-parser",
            "deepseek_v41",
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "deepseek_v41",
            "--limit-mm-per-prompt",
            '{"image":4}',
            "--mm-processor-cache-gb",
            "1",
            "--kv-cache-memory-bytes",
            "9663676416",
            "--kernel-config",
            '{"moe_backend":"b12x"}',
            "--prefix-cache-retention-interval",
            "1024",
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}',
            "--compilation-config",
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24]}',
        ],
        "env": {
            "NCCL_MAX_NCHANNELS": "8",
            "NCCL_CROSS_NIC": "0",
            "NCCL_IB_MERGE_NICS": "0",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "DSV41_ENGRAM_DISK": "1",
        },
    },
    "glm-tp8": {
        "model_id": "glm-5.3",
        "nodes": 8,
        "context": 262144,
        "output_budget": 32768,
        "batch_tokens": 8192,
        "max_sequences": 8,
        "memory_utilization": 0.72,
        "weight_format": "NVFP4 weights / FP8 MLA KV",
        "requires": ["SM121 GLM 5.3 runtime", "Marlin NVFP4", "MTP", "fp8_ds_mla"],
        "flags": [
            "--load-format",
            "auto",
            "--language-model-only",
            "--kv-cache-dtype",
            "fp8_ds_mla",
            "--attention-config.indexer_kv_dtype",
            "mxfp4",
            "--moe-backend",
            "marlin",
            "--reasoning-parser",
            "glm47",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "glm47",
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":3,"enable_adaptive_verification":false,"moe_backend":"triton","use_local_argmax_reduction":true}',
        ],
        "env": {"NCCL_MAX_NCHANNELS": "2", "NCCL_CROSS_NIC": "1"},
    },
}


def profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    return copy.deepcopy(PROFILES[name])


def example(name):
    p = profile(name)
    return {
        "schema": 1,
        "cluster": "my-spark",
        "profile": name,
        "runtime": {
            "image": "SET_A_VALIDATED_IMAGE@sha256:" + "0" * 64,
            "entrypoint": ["vllm"],
            "compatible_runtime_confirmed": False,
        },
        "paths": {
            "state": "/srv/sparkfleet",
            "weights": "/srv/models/model",
            "cache": "/srv/sparkfleet/cache",
            "env_file": None,
        },
        "network": {
            "master_port": 29541,
            "gid_index": 3,
            "socket_interface": "SET_INTERFACE",
            "hca": "SET_RDMA_DEVICE",
            "rdma_devices": ["/dev/infiniband"],
        },
        "serving": {
            "port": 8000,
            "host": "127.0.0.1",
            "context": p["context"],
            "batch_tokens": p["batch_tokens"],
            "max_sequences": p["max_sequences"],
        },
        "guard": {
            "warning_c": 85,
            "trip_c": 90,
            "abort_c": 85,
            "memory_floor_gib": 8,
            "poll_seconds": 2,
            "max_missed_seconds": 15,
        },
        "mounts": [],
        "nodes": [
            {
                "name": f"spark-{i + 1:02}",
                "ssh": f"operator@192.0.2.{11 + i}",
                "data_ip": f"198.51.100.{11 + i}",
            }
            for i in range(p["nodes"])
        ],
    }
