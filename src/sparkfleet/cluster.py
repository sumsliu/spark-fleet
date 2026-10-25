"""Read-only SSH inventory, fresh plans and foreground coordinated supervision."""

import asyncio
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import time
from . import agent
from .common import canonical, digest, save_json
from .config import validate
from .render import commands

AGENT = Path(agent.__file__)
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
]


def remote(target, command, body=None, timeout=60):
    p = subprocess.run(
        SSH + [target, command], input=body, capture_output=True, timeout=timeout
    )
    if p.returncode:
        raise RuntimeError(
            f"SSH command failed for {target}; check authentication/host keys locally"
        )
    return p.stdout


def probe(row):
    source = AGENT.read_text()
    cmd = "python3 -I -c " + shlex.quote(source) + " probe"
    return json.loads(remote(row["node"]["ssh"], cmd, canonical(row)))


def doctor(c):
    from concurrent.futures import ThreadPoolExecutor

    rows = commands(c, "inventory")
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(probe, rows))
    return dict(
        at=time.time(),
        read_only=True,
        profile=c["profile"],
        nodes=[dict(node=r["node"]["name"], observed=v) for r, v in zip(rows, values)],
    )


def plan(c):
    validate(c)
    release = time.strftime("%Y%m%d") + "-" + secrets.token_hex(5)
    rows = commands(c, release)
    for row in rows:
        row["observed"] = probe(row)
        agent.admission(row["observed"], row)
        row["agent_sha256"] = digest(AGENT.read_bytes())
    result = dict(
        schema=1,
        release=release,
        created_at=time.time(),
        config=c,
        nodes=rows,
        agent_sha256=digest(AGENT.read_bytes()),
    )
    result["plan_sha256"] = digest(canonical(result))
    return result


def verify_plan(p):
    q = dict(p)
    signature = q.pop("plan_sha256")
    if digest(canonical(q)) != signature:
        raise ValueError("Plan content changed")
    if time.time() - p["created_at"] > 3600 or p["created_at"] > time.time() + 60:
        raise ValueError("Plan expired; prepare fresh observations")
    if digest(AGENT.read_bytes()) != p["agent_sha256"]:
        raise ValueError("Agent changed since plan")
    validate(p["config"])
    expected = commands(p["config"], p["release"])
    for actual, base in zip(p["nodes"], expected):
        for key, value in base.items():
            if actual.get(key) != value:
                raise ValueError("Node argv/config differs from recipe")
        if actual["agent_sha256"] != p["agent_sha256"]:
            raise ValueError("Node agent source mismatch")
    if len(p["nodes"]) != len(expected):
        raise ValueError("Plan rank count changed")


def stage(row):
    root = Path(row["paths"]["state"]) / "releases" / row["release"] / "bundle"
    # Source/metadata only; model keys remain in a node-local owner-private env file.
    payload = {"root": str(root), "agent": AGENT.read_text(), "config": row}
    code = """import sys,json,os,pathlib
p=json.load(sys.stdin);r=pathlib.Path(p['root']);r.mkdir(parents=True,exist_ok=False,mode=0o700)
for n,data in [('agent.py',p['agent'].encode()),('node.json',json.dumps(p['config']).encode())]:
 fd=os.open(r/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data)
"""
    remote(row["node"]["ssh"], "python3 -I -c " + shlex.quote(code), canonical(payload))
    return root


async def supervise(p, receipt_dir):
    verify_plan(p)
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_json(receipt_dir / "plan.json", p)
    used = receipt_dir / "started.json"
    save_json(used, dict(at=time.time(), plan_sha256=p["plan_sha256"]), exclusive=True)
    rows = p["nodes"]
    roots = {}
    children = {}
    tasks = []
    last = {}
    seen = set()
    failed = None
    events = (receipt_dir / "events.jsonl").open("x", encoding="utf-8")
    os.chmod(events.name, 0o600)

    def record(row, event):
        events.write(
            json.dumps(dict(at=time.time(), node=row["node"]["name"], **event)) + "\n"
        )
        events.flush()

    async def read(row, child):
        while line := await child.stdout.readline():
            value = json.loads(line)
            record(row, value)
            last[row["rank"]] = time.monotonic()
            seen.add(row["rank"])
            if value.get("event") == "finished":
                raise RuntimeError("Node supervisor ended")
            if value.get("event") != "status" or not value.get("four_locks_held"):
                raise RuntimeError("Invalid guard status")
        raise RuntimeError("Node control channel closed")

    try:
        # All nodes are re-probed before anything starts. Normal workloads are refused, never cancelled.
        for row in rows:
            agent.admission(await asyncio.to_thread(probe, row), row, row["observed"])
        for row in rows:
            roots[row["rank"]] = await asyncio.to_thread(stage, row)
        for row in rows:
            root = roots[row["rank"]]
            command = (
                "python3 -I "
                + shlex.quote(str(root / "agent.py"))
                + " run --config "
                + shlex.quote(str(root / "node.json"))
            )
            child = await asyncio.create_subprocess_exec(
                *SSH,
                row["node"]["ssh"],
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            children[row["rank"]] = child
            last[row["rank"]] = time.monotonic()
            tasks.append(asyncio.create_task(read(row, child)))
        print(
            "Guarded ranks starting. Keep this controller running (tmux/systemd). API acceptance is a separate gate.",
            flush=True,
        )
        while True:
            done, _ = await asyncio.wait(
                tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            if done:
                for task in done:
                    task.result()
            if any(
                time.monotonic() - v
                > (p["config"]["guard"]["max_missed_seconds"] if rank in seen else 60)
                for rank, v in last.items()
            ):
                raise RuntimeError(
                    "Guard heartbeat expired; stopping only this release"
                )
    except asyncio.CancelledError:
        failed = "Controller cancelled"
        raise
    except BaseException as e:
        failed = type(e).__name__ + ": " + str(e)
        raise
    finally:
        for child in children.values():
            if child.stdin:
                try:
                    child.stdin.write(b"STOP\n")
                    await child.stdin.drain()
                    child.stdin.close()
                except (OSError, ConnectionError):
                    pass

        # A second exact-ownership stop path covers an agent crash after Docker creation.
        async def contain(row):
            if row["rank"] not in roots:
                return True
            root = roots[row["rank"]]
            command = (
                "python3 -I "
                + shlex.quote(str(root / "agent.py"))
                + " stop --config "
                + shlex.quote(str(root / "node.json"))
            )
            try:
                value = json.loads(
                    await asyncio.to_thread(
                        remote, row["node"]["ssh"], command, None, 45
                    )
                )
                return value.get("stopped") is True
            except Exception:
                return False

        contained = await asyncio.gather(*(contain(row) for row in rows))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for child in children.values():
            if child.returncode is None:
                try:
                    await asyncio.wait_for(child.wait(), 5)
                except asyncio.TimeoutError:
                    child.terminate()
                    await child.wait()
        save_json(
            receipt_dir / "STOPPED.json",
            dict(
                at=time.time(), reason=failed, all_exact_stops_confirmed=all(contained)
            ),
        )
        events.close()
