"""Standalone Linux node agent. This file is staged verbatim and SHA-bound.

Holds four locks; closes only its exact container on fault or controller EOF.
No root installer, password handling, clock tuning or broad process selection.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import threading
import time

LOCKS = ("model-switch.lock", "phaseB.lock", "stepA.lock", "llm-switch.lock")


def call(args, timeout=15):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(
            f"Command failed: {args[0]} ({p.returncode}); inspect node locally"
        )
    return p.stdout.strip()


def hashed(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError("Expected an ordinary source file")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, data):
    p = Path(path)
    tmp = p.with_name("." + p.name + "." + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)


def snapshot(c):
    memory = {
        k: int(v.split()[0])
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    gpu = call(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(gpu) != 1:
        raise ValueError("Expected exactly one GB10 GPU per Spark")
    name, temp = gpu[0].rsplit(",", 1)
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(float(p.read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    image = json.loads(
        call(
            [
                "docker",
                "image",
                "inspect",
                c["argv"][c["argv"].index("--entrypoint") + 2],
            ]
        )
    )[0]
    files = {m["source"]: hashed(m["source"]) for m in c.get("expected_mounts", [])}
    for m in c.get("expected_mounts", []):
        if files[m["source"]] != m["sha256"]:
            raise ValueError("Overlay source digest mismatch")
    for key in ["weights", "cache", "state"]:
        if not Path(c["paths"][key]).is_dir():
            raise ValueError("Pre-create paths." + key + " on the node")
    envfile = c["paths"].get("env_file")
    env_hash = None
    if envfile:
        p = Path(envfile)
        st = p.lstat()
        if (
            not stat.S_ISREG(st.st_mode)
            or st.st_uid != os.geteuid()
            or st.st_mode & 0o077
        ):
            raise ValueError("env_file must be an owner-private regular file")
        env_hash = hashed(p)
    locks = []
    lockroot = Path(c["paths"]["state"]) / "locks"
    for name_ in LOCKS:
        path = lockroot / name_
        if not path.exists():
            locks.append({"name": name_, "available": True, "not_created": True})
            continue
        if path.is_symlink():
            raise ValueError("Lock symlink refused")
        with path.open("r") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                available = True
            except BlockingIOError:
                available = False
            locks.append({"name": name_, "available": available})
    return dict(
        at=time.time(),
        architecture=platform.machine(),
        gpu_name=name.strip(),
        gpu_c=float(temp),
        available_gib=memory["MemAvailable"] / 1048576,
        platform_c=max(temps) if temps else None,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        image_id=image["Id"],
        overlay_hashes=files,
        env_file_sha256=env_hash,
        locks=locks,
        compute_pids=call(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        ).splitlines(),
    )


def admission(observed, c, expected=None):
    if observed["architecture"] != "aarch64" or "GB10" not in observed["gpu_name"]:
        raise ValueError("This recipe targets Linux ARM64 GB10")
    if (
        observed["gpu_c"] >= c["guard"]["abort_c"]
        or observed["available_gib"] < c["guard"]["memory_floor_gib"]
    ):
        raise ValueError("Thermal or memory admission failed")
    if observed["compute_pids"] or not all(x["available"] for x in observed["locks"]):
        raise ValueError("Existing workload/lock present; it will not be stopped")
    if expected:
        for k in ["boot_id", "image_id", "overlay_hashes", "env_file_sha256"]:
            if observed[k] != expected[k]:
                raise ValueError("Plan is stale: " + k)


def owned(c, x):
    labels = x.get("Config", {}).get("Labels") or {}
    entry = c["engine_argv"]
    return (
        labels.get("org.sparkfleet.managed") == "true"
        and labels.get("org.sparkfleet.cluster") == c["cluster"]
        and labels.get("org.sparkfleet.release") == c["release"]
        and x.get("Image") == c["observed"]["image_id"]
        and x["Config"].get("Entrypoint") == [entry[0]]
        and x["Config"].get("Cmd") == entry[1:]
    )


def stop_owned(c, cid=None):
    p = subprocess.run(
        ["docker", "inspect", cid or c["name"]],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if p.returncode:
        # A daemon/permission error is not evidence that the container is absent.
        selector = ("id=" + cid) if cid else ("name=^/" + c["name"] + "$")
        existing = call(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                selector,
                "--format",
                "{{.ID}}",
            ]
        )
        if existing:
            raise RuntimeError("Container inspection failed; stop unconfirmed")
        return True
    x = json.loads(p.stdout)[0]
    if not owned(c, x) or (cid and x["Id"] != cid):
        raise ValueError("Container ownership changed; refusing stop")
    if x["State"]["Running"]:
        call(["docker", "stop", "--time", "20", x["Id"]], timeout=30)
    x = json.loads(call(["docker", "inspect", x["Id"]]))[0]
    if x["State"]["Running"]:
        raise RuntimeError("Owned container stop unconfirmed")
    return True


def run(c):
    if hashed(__file__) != c["agent_sha256"]:
        raise ValueError("Agent source changed since plan")
    state = Path(c["paths"]["state"])
    fence = state / "incident.json"
    if fence.exists():
        raise ValueError("Persistent incident fence present; investigate before reset")
    release = state / "releases" / c["release"]
    release.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(release / "used", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    lockroot = state / "locks"
    lockroot.mkdir(mode=0o700, exist_ok=True)
    locks = []
    cid = None
    done = threading.Event()
    fault = None
    stop_confirmed = True

    def receive():
        try:
            for line in sys.stdin:
                if line.strip() == "STOP":
                    break
        finally:
            done.set()

    threading.Thread(target=receive, daemon=True).start()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: done.set())
    try:
        observed = snapshot(c)
        admission(observed, c, c["observed"])
        for name in LOCKS:
            p = lockroot / name
            fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            f = os.fdopen(fd, "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                f.close()
                raise
            locks.append(f)
        # Recheck after acquiring all leases, without treating our own locks as foreign.
        check = snapshot(c)
        check["locks"] = [{"available": True}] * 4
        admission(check, c, c["observed"])
        if done.is_set():
            raise RuntimeError("Controller disconnected before launch")
        cid = call(c["argv"], timeout=45)
        x = json.loads(call(["docker", "inspect", cid]))[0]
        if not owned(c, x):
            raise ValueError("Started container identity mismatch")
        receipt = dict(
            container_id=x["Id"],
            image_id=x["Image"],
            started_at=x["State"]["StartedAt"],
            boot_id=observed["boot_id"],
            cluster=c["cluster"],
            release=c["release"],
        )
        atomic(release / "receipt.json", receipt)
        while not done.is_set():
            s = snapshot(c)
            x = json.loads(call(["docker", "inspect", cid]))[0]
            if (
                not owned(c, x)
                or x["State"]["StartedAt"] != receipt["started_at"]
                or not x["State"]["Running"]
            ):
                raise RuntimeError("Container identity/running state changed")
            if (
                s["boot_id"] != receipt["boot_id"]
                or s["gpu_c"] >= c["guard"]["abort_c"]
                or s["available_gib"] < c["guard"]["memory_floor_gib"]
            ):
                raise RuntimeError("Guard abort threshold crossed")
            # No auth headers, model prompts, environment values or container logs.
            print(
                json.dumps(
                    dict(
                        event="status",
                        rank=c["rank"],
                        gpu_c=s["gpu_c"],
                        available_gib=s["available_gib"],
                        platform_c=s["platform_c"],
                        four_locks_held=len(locks) == 4,
                        receipt=receipt,
                    )
                ),
                flush=True,
            )
            done.wait(c["guard"]["poll_seconds"])
    except BaseException as e:
        fault = type(e).__name__ + ": " + str(e)
        atomic(
            fence,
            dict(
                at=time.time(),
                fault=fault,
                cluster=c["cluster"],
                release=c["release"],
                container_id=cid,
            ),
        )
    finally:
        try:
            if cid:
                stop_owned(c, cid)
        except BaseException:
            stop_confirmed = False
            atomic(
                fence,
                dict(
                    at=time.time(),
                    fault="Stop unconfirmed; manual inspection required",
                    cluster=c["cluster"],
                    release=c["release"],
                    container_id=cid,
                ),
            )
        for f in reversed(locks):
            f.close()
        try:
            print(
                json.dumps(
                    dict(
                        event="finished",
                        rank=c["rank"],
                        fault=fault,
                        stop_confirmed=stop_confirmed,
                    )
                ),
                flush=True,
            )
        except BrokenPipeError:
            pass
    if fault or not stop_confirmed:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["probe", "run", "stop"])
    p.add_argument("--config")
    a = p.parse_args()
    c = (
        json.load(sys.stdin)
        if a.action == "probe"
        else json.loads(Path(a.config).read_text())
    )
    if a.action == "probe":
        print(json.dumps(snapshot(c)))
    elif a.action == "stop":
        print(json.dumps({"stopped": stop_owned(c)}))
    else:
        run(c)


if __name__ == "__main__":
    main()
