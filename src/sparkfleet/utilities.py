"""Weight provenance, read-only fabric commands and release export checks."""

import hashlib
from pathlib import Path
import re
from .common import save_json


def weight_manifest(root, out):
    root = Path(root).resolve()
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink in weight tree")
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(16 * 1024 * 1024):
                h.update(chunk)
        files.append(
            dict(
                path=str(p.relative_to(root)),
                bytes=p.stat().st_size,
                sha256=h.hexdigest(),
            )
        )
    if not files:
        raise ValueError("Empty model directory")
    if Path(out).resolve().is_relative_to(root):
        raise ValueError("Write manifest outside the weight directory")
    result = dict(schema=1, files=files, total_bytes=sum(x["bytes"] for x in files))
    save_json(out, result)
    return result


def verify_weights(root, manifest):
    root = Path(root).resolve()
    names = [row["path"] for row in manifest["files"]]
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if not names or len(set(names)) != len(names) or set(names) != actual:
        raise ValueError("Model file set differs from manifest")
    for row in manifest["files"]:
        p = root / row["path"]
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError("Unsafe/missing model file")
        if p.stat().st_size != row["bytes"]:
            raise ValueError("Model size mismatch")
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(16 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != row["sha256"]:
            raise ValueError("Model SHA256 mismatch")
    return dict(
        passed=True,
        files=len(manifest["files"]),
        note="No downloading, conversion or quantization performed",
    )


def network_plan(c):
    net = c["network"]
    commands = []
    for n in c["nodes"]:
        interface = n.get("fabric", {}).get("socket_interface", net["socket_interface"])
        commands.append(
            dict(
                node=n["name"],
                ssh=n["ssh"],
                read_only=[
                    ["ip", "-j", "addr", "show", "dev", interface],
                    ["ip", "-j", "link", "show", "dev", interface],
                    ["ethtool", interface],
                    ["ethtool", "--show-fec", interface],
                    ["ibv_devinfo"],
                    ["rdma", "link", "show"],
                    ["show_gids"],
                ],
            )
        )
    return dict(
        schema=1,
        read_only_inventory=commands,
        tcp_pair_example={
            "server": ["iperf3", "--server", "--one-off", "--port", "5201"],
            "client": [
                "iperf3",
                "--client",
                c["nodes"][0]["data_ip"],
                "--port",
                "5201",
                "--parallel",
                "8",
                "--time",
                "30",
                "--json",
            ],
            "reverse": [
                "iperf3",
                "--client",
                c["nodes"][0]["data_ip"],
                "--port",
                "5201",
                "--parallel",
                "8",
                "--time",
                "30",
                "--reverse",
                "--json",
            ],
        },
        notes=[
            "Inventory commands do not alter link, MTU, switch or driver settings.",
            "Run explicit traffic tests only in an owner-authorized quiet window; test both directions.",
            "TCP throughput does not establish GPU RoCE/NCCL bandwidth. Use perftest and nccl-tests separately.",
            "200 Gbit/s is 25 GB/s raw line rate, not 200 GB/s and not a payload guarantee.",
        ],
    )


def parse_iperf(report):
    if report.get("error"):
        raise ValueError("iperf reported an error")
    end = report.get("end", {})
    row = end.get("sum_received") or end.get("sum")
    if not row or not isinstance(row.get("bits_per_second"), (int, float)):
        raise ValueError("No received-payload throughput")
    return dict(
        received_gbit_s=row["bits_per_second"] / 1e9,
        received_gbyte_s=row["bits_per_second"] / 8e9,
        fraction_of_200g_line_rate=row["bits_per_second"] / 200e9,
        scope="TCP application payload only",
    )


def public_audit(root):
    """Conservative export check. Not a replacement for review or GitHub secret scanning."""
    root = Path(root)
    findings = []
    patterns = {
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "device_hostname": re.compile(r"\b[\w.-]+\.ts\.net\b"),
        "private_home": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
        "email": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
        "private_ipv4": re.compile(
            r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b"
        ),
    }
    prohibited = {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".safetensors",
        ".gguf",
        ".dmg",
        ".exe",
        ".so",
        ".dylib",
    }
    for p in root.rglob("*"):
        if any(
            x in p.relative_to(root).parts
            for x in [
                ".git",
                ".venv",
                ".ruff_cache",
                "__pycache__",
                "work",
                "dist",
                "build",
                "sparkfleet.egg-info",
            ]
        ):
            continue
        if p.is_symlink():
            findings.append(dict(path=str(p.relative_to(root)), kind="symlink"))
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in prohibited:
            findings.append(
                dict(path=str(p.relative_to(root)), kind="prohibited_binary_or_key")
            )
        if p.stat().st_size > 2 * 1024 * 1024:
            findings.append(dict(path=str(p.relative_to(root)), kind="large_file"))
            continue
        try:
            body = p.read_text()
        except UnicodeDecodeError:
            findings.append(dict(path=str(p.relative_to(root)), kind="binary"))
            continue
        # The scanner and its tests contain detection expressions, never real tokens.
        if p.name == "utilities.py" or p.name == "test_utilities.py":
            continue
        for name, pat in patterns.items():
            if pat.search(body):
                findings.append(dict(path=str(p.relative_to(root)), kind=name))
    return dict(
        passed=not findings,
        findings=findings,
        note="Pattern scan plus manual clean-tree review required before public export",
    )
