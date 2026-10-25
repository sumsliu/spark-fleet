"""SparkFleet CLI: no cluster mutation without the explicit run command."""

import argparse
import asyncio
import json
import os
import sys
from .common import read_json, save_json
from .config import validate
from .profiles import PROFILES, example


def main():
    parser = argparse.ArgumentParser(
        prog="sparkfleet", description="Portable multi-node DGX Spark inference tooling"
    )
    from . import __version__

    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles")
    p = sub.add_parser("init")
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("validate")
    p.add_argument("config")
    p.add_argument("--template", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--release", default="preview")
    p.add_argument("--template", action="store_true")
    p = sub.add_parser("doctor")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("plan")
    p.add_argument("--receipts", required=True)
    p.add_argument("--execute", action="store_true", required=True)
    p = sub.add_parser("qualify")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-env", default="SPARKFLEET_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--openai-only", action="store_true")
    p.add_argument("--long-tokens", type=int, default=0)
    p = sub.add_parser("benchmark")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--key-env", default="SPARKFLEET_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--phase", choices=["A0", "B0", "B1", "A1"], required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--fixed-output", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--recheck", required=True)
    p.add_argument("--quality-receipt", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("weights-manifest")
    p.add_argument("root")
    p.add_argument("--out", required=True)
    p = sub.add_parser("weights-verify")
    p.add_argument("root")
    p.add_argument("manifest")
    p = sub.add_parser("network-plan")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p = sub.add_parser("network-result")
    p.add_argument("iperf_json")
    p = sub.add_parser("audit-public")
    p.add_argument("root", nargs="?", default=".")
    p = sub.add_parser("overlay-inspect")
    p.add_argument("overlay")
    p.add_argument("--runtime-root")
    p = sub.add_parser("overlay-materialize")
    p.add_argument("overlay")
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gateway")
    p.add_argument("--config", required=True)
    a = parser.parse_args()
    try:
        result = None
        if a.command == "profiles":
            result = {
                k: {
                    "nodes": v["nodes"],
                    "model_id": v["model_id"],
                    "requires": v["requires"],
                }
                for k, v in PROFILES.items()
            }
        elif a.command == "init":
            save_json(a.out, example(a.profile), exclusive=True)
            result = {
                "created": a.out,
                "template_only": True,
                "next": "Edit addresses, runtime digest, paths and fabric; see docs/quickstart.md",
            }
        elif a.command == "validate":
            validate(read_json(a.config), template=a.template)
            result = {"valid": True, "template_only": a.template}
        elif a.command == "render":
            from .render import render

            rows = render(read_json(a.config), a.out, a.release, template=a.template)
            result = {"rendered": len(rows), "out": a.out, "executed": False}
        elif a.command in ("doctor", "plan"):
            from .cluster import doctor, plan

            result = (doctor if a.command == "doctor" else plan)(read_json(a.config))
            save_json(a.out, result, exclusive=True)
        elif a.command == "run":
            from .cluster import supervise

            asyncio.run(supervise(read_json(a.plan), a.receipts))
            return
        elif a.command in ("qualify", "benchmark"):
            from .api import Client, qualify

            key = os.environ.get(a.key_env)
            if not key:
                raise ValueError(
                    "Set " + a.key_env + " in this shell; no key is accepted on argv"
                )
            c = Client(a.base_url, key)
            if a.command == "qualify":
                result = qualify(
                    c,
                    a.model,
                    a.out,
                    anthropic=not a.openai_only,
                    long_tokens=a.long_tokens,
                )
                if not result["passed"]:
                    print(json.dumps(result, indent=2))
                    raise SystemExit(1)
            else:
                from .benchmark import benchmark

                result = benchmark(
                    c,
                    a.model,
                    a.out,
                    phase=a.phase,
                    concurrency=a.concurrency,
                    repetitions=a.repetitions,
                    tokens=a.tokens,
                    fixed=a.fixed_output,
                )
        elif a.command == "compare":
            from .benchmark import compare

            quality = read_json(a.quality_receipt)
            if not quality.get("complete"):
                raise ValueError(
                    "An actual completed API qualification receipt is required"
                )
            candidate = read_json(a.candidate)
            if quality.get("model") != candidate.get("model"):
                raise ValueError("Quality receipt model differs from benchmark")
            result = compare(
                read_json(a.baseline),
                candidate,
                read_json(a.recheck),
                quality_passed=quality.get("passed") is True,
            )
            save_json(a.out, result)
        elif a.command == "weights-manifest":
            from .utilities import weight_manifest

            result = weight_manifest(a.root, a.out)
        elif a.command == "weights-verify":
            from .utilities import verify_weights

            result = verify_weights(a.root, read_json(a.manifest))
        elif a.command == "network-plan":
            from .utilities import network_plan

            c = read_json(a.config)
            validate(c, template=True)
            result = network_plan(c)
            save_json(a.out, result)
        elif a.command == "network-result":
            from .utilities import parse_iperf

            result = parse_iperf(read_json(a.iperf_json))
        elif a.command == "audit-public":
            from .utilities import public_audit

            result = public_audit(a.root)
            if not result["passed"]:
                print(json.dumps(result, indent=2))
                raise SystemExit(1)
        elif a.command == "overlay-inspect":
            from .overlays import inspect_overlay

            result = inspect_overlay(a.overlay, a.runtime_root)
        elif a.command == "overlay-materialize":
            from .overlays import materialize

            result = materialize(a.overlay, a.runtime_root, a.out)
        elif a.command == "gateway":
            try:
                from .gateway import serve
            except ImportError:
                raise ValueError(
                    'Install optional gateway support: pip install ".[gateway]"'
                )
            serve(a.config)
            return
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        print("sparkfleet: " + str(e), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print(
            "Controller interrupted; inspect STOPPED.json for exact-stop confirmation.",
            file=sys.stderr,
        )
        raise SystemExit(130)
