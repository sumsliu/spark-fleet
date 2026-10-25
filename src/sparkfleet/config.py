"""Validate explicit topology and paths before generating any commands."""

import ipaddress
import math
import re
from pathlib import PurePosixPath
from .profiles import profile

IDENT = re.compile(r"[a-z][a-z0-9-]{0,39}\Z")
SSH = re.compile(r"[a-z_][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.-]*\Z")
IMAGE = re.compile(r"(?:[A-Za-z0-9_./:-]+@)?sha256:[a-f0-9]{64}\Z")


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(c in value for c in "\n\r\0:")
    ):
        raise ValueError(
            "Expected an absolute Linux path without colon/control characters"
        )
    if ".." in PurePosixPath(value).parts or value == "/":
        raise ValueError("Unsafe path")
    return value


def validate(c, *, template=False):
    if c.get("schema") != 1 or not IDENT.fullmatch(c.get("cluster", "")):
        raise ValueError("schema=1 and a lowercase cluster identifier are required")
    p = profile(c["profile"])
    nodes = c["nodes"]
    if len(nodes) != p["nodes"]:
        raise ValueError(f"{c['profile']} needs exactly {p['nodes']} nodes")
    for key in ["name", "ssh", "data_ip"]:
        if len({n[key] for n in nodes}) != len(nodes):
            raise ValueError(f"Duplicate node {key}")
    for n in nodes:
        if not IDENT.fullmatch(n["name"]) or not SSH.fullmatch(n["ssh"]):
            raise ValueError("Unsafe SSH target or node name")
        address = ipaddress.ip_address(n["data_ip"])
        if (
            address.version != 4
            or address.is_multicast
            or address.is_loopback
            or address.is_unspecified
        ):
            raise ValueError("A unicast IPv4 fabric address is required")
        documentation = [
            ipaddress.ip_network(x)
            for x in ["192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"]
        ]
        management = n["ssh"].split("@")[1]
        try:
            management_ip = ipaddress.ip_address(management)
        except ValueError:
            management_ip = None
        if not template and (
            any(address in subnet for subnet in documentation)
            or (
                management_ip
                and any(management_ip in subnet for subnet in documentation)
            )
        ):
            raise ValueError("Replace documentation-only node addresses")
        override = n.get("fabric", {})
        if set(override) - {"socket_interface", "hca", "gid_index"}:
            raise ValueError("Unknown node fabric override")
        for key in ["socket_interface", "hca"]:
            if key in override and not re.fullmatch(
                r"[A-Za-z0-9_.,:=/-]+", override[key]
            ):
                raise ValueError("Invalid per-node fabric")
        if "gid_index" in override and (
            type(override["gid_index"]) is not int
            or not 0 <= override["gid_index"] <= 255
        ):
            raise ValueError("Invalid per-node GID")
    runtime = c["runtime"]
    if not template and (
        not IMAGE.fullmatch(runtime["image"])
        or "SET_" in runtime["image"]
        or runtime["image"].endswith("0" * 64)
    ):
        raise ValueError("Pin a real compatible image by registry digest")
    if not template and runtime.get("compatible_runtime_confirmed") is not True:
        raise ValueError("Read docs/runtime.md and confirm your runtime compatibility")
    if (
        not isinstance(runtime["entrypoint"], list)
        or not runtime["entrypoint"]
        or not all(
            isinstance(v, str) and v and "\0" not in v for v in runtime["entrypoint"]
        )
    ):
        raise ValueError("runtime.entrypoint must be a non-empty argv list")
    for key in ["state", "weights", "cache"]:
        safe_path(c["paths"][key])
    if c["paths"].get("env_file"):
        safe_path(c["paths"]["env_file"])
    for mount in c.get("mounts", []):
        safe_path(mount["source"])
        safe_path(mount["target"])
        if not re.fullmatch(r"[a-f0-9]{64}", mount.get("sha256", "")):
            raise ValueError("Every overlay file needs a SHA256")
    net = c["network"]
    for key in ["socket_interface", "hca"]:
        if not re.fullmatch(r"[A-Za-z0-9_.,:=/-]+", net[key]) or (
            not template and net[key].startswith("SET_")
        ):
            raise ValueError("Set the measured socket interface and HCA")
    if type(net["gid_index"]) is not int or not 0 <= net["gid_index"] <= 255:
        raise ValueError("Invalid GID index")
    for d in net["rdma_devices"]:
        if d != "/dev/infiniband" and not d.startswith("/dev/infiniband/"):
            raise ValueError("Only explicit InfiniBand device mappings are supported")
        safe_path(d)
    g = c["guard"]
    if any(
        type(g[k]) not in (int, float) or not math.isfinite(g[k])
        for k in [
            "warning_c",
            "trip_c",
            "abort_c",
            "memory_floor_gib",
            "poll_seconds",
            "max_missed_seconds",
        ]
    ):
        raise ValueError("Guard fields must be finite numbers")
    if not (
        0 < g["abort_c"] <= 85
        and g["abort_c"] <= g["trip_c"] <= 90
        and 0 < g["warning_c"] <= g["trip_c"]
        and g["memory_floor_gib"] >= 8
    ):
        raise ValueError("Guard limits exceed the reference safety envelope")
    if not (0.5 <= g["poll_seconds"] <= 5 and 5 <= g["max_missed_seconds"] <= 30):
        raise ValueError("Invalid guard cadence")
    if net["master_port"] == c["serving"]["port"]:
        raise ValueError("Master and API ports must differ")
    for value in [net["master_port"], c["serving"]["port"]]:
        if type(value) is not int or not 1024 <= value <= 65535:
            raise ValueError("Use unprivileged TCP ports")
    if c["serving"]["host"] != "127.0.0.1":
        raise ValueError(
            "Native API binds loopback; publish through an authenticated gateway"
        )
    for key in ["context", "batch_tokens", "max_sequences"]:
        if type(c["serving"][key]) is not int or c["serving"][key] <= 0:
            raise ValueError("Serving budgets must be positive integers")
    if c["serving"]["context"] > 262144:
        raise ValueError("Larger context is outside the provided recipe")
    if c["serving"]["batch_tokens"] > 16384 or c["serving"]["max_sequences"] > 16:
        raise ValueError("Serving budget exceeds recipe envelope")
    return p
