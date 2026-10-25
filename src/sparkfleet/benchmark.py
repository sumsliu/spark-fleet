"""Comparable workload samples and paired bootstrap with baseline drift gate."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import random
import statistics
import time
from .api import chat
from .common import digest, save_json

PROMPTS = [
    "Explain how a prefix cache helps repeated tool-call conversations. Give a practical example.",
    "Write a Python function that merges two sorted iterators lazily, with comments and examples.",
    "Describe a reliable blue/green API gateway migration with connection continuity and rollback.",
    "Explain how to distinguish network line rate from measured application payload throughput.",
    "Create a checklist for comparing two inference runtimes using the same model and prompt set.",
    "Explain speculative decoding acceptance rate and why a fast draft model can still reduce speed.",
]


def benchmark(
    client, model, out, *, phase, concurrency=1, repetitions=2, tokens=256, fixed=False
):
    if phase not in ("A0", "B0", "B1", "A1"):
        raise ValueError("Use an ABBA phase A0/B0/B1/A1")
    if (
        concurrency not in (1, 2, 4, 8)
        or not 1 <= repetitions <= 16
        or not 32 <= tokens <= 32768
    ):
        raise ValueError("Invalid benchmark bounds")
    rows = []

    def one(text):
        p = chat(model, text, stream=True)
        p.update(max_tokens=tokens, stream_options={"include_usage": True})
        if fixed:
            p["ignore_eos"] = True
        result = client.request("/v1/chat/completions", p, stream=True)
        if len(result) != 2:
            raise ValueError("Benchmark HTTP failure")
        r, start = result
        first = None
        usage = None
        done = False
        text_chars = 0
        with r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                d = json.loads(raw)
                usage = d.get("usage") or usage
                for choice in d.get("choices", []):
                    part = choice.get("delta", {}).get("content") or ""
                    if part and first is None:
                        first = time.monotonic() - start
                    text_chars += len(part)
        if not done or first is None or not usage or not usage.get("completion_tokens"):
            raise ValueError("Incomplete stream/usage")
        return dict(
            seconds=time.monotonic() - start,
            first_text_seconds=first,
            completion_tokens=usage["completion_tokens"],
            usage=usage,
            text_characters=text_chars,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for rep in range(repetitions):
            for index, prompt in enumerate(PROMPTS):
                texts = [
                    prompt + "\nVariant " + str(i) + ": be precise."
                    for i in range(concurrency)
                ]
                started = time.monotonic()
                answers = list(pool.map(one, texts))
                elapsed = time.monotonic() - started
                count = sum(x["completion_tokens"] for x in answers)
                rows.append(
                    dict(
                        pair_key=f"{index}:{rep}:c{concurrency}",
                        aggregate_e2e_tps=count / elapsed,
                        output_tokens=count,
                        wall_seconds=elapsed,
                        requests=answers,
                        prompt_sha256=digest("\n".join(texts).encode()),
                    )
                )
                print(
                    f"{phase} prompt={index} rep={rep} aggregate_e2e_tps={count / elapsed:.2f}",
                    flush=True,
                )
                save_json(out, dict(complete=False, phase=phase, samples=rows))
    result = dict(
        schema=1,
        complete=True,
        at=time.time(),
        phase=phase,
        model=model,
        workload={
            "concurrency": concurrency,
            "max_tokens": tokens,
            "fixed_output": fixed,
            "fixture_version": 1,
        },
        samples=rows,
        metric="aggregate_e2e_tps = sum(output tokens) / batch wall time",
        caveat="Model quality is not established by speed; cache/OS state must be reported separately.",
    )
    save_json(out, result)
    return result


def compare(a, b, recheck, *, quality_passed=False, drift_limit=0.03, bootstrap=10000):
    for result in (a, b, recheck):
        if not result.get("complete") or not result.get("samples"):
            raise ValueError("Incomplete benchmark")
    if not (
        a["model"] == b["model"] == recheck["model"]
        and a["workload"] == b["workload"] == recheck["workload"]
    ):
        raise ValueError("Model/workload definitions differ")
    maps = [{r["pair_key"]: r for r in result["samples"]} for result in (a, b, recheck)]
    if any(
        len(m) != len(result["samples"]) for m, result in zip(maps, (a, b, recheck))
    ):
        raise ValueError("Duplicate paired case")
    keys = set(maps[0])
    if any(set(m) != keys for m in maps):
        raise ValueError("Paired cases differ")
    if len(keys) < 4:
        raise ValueError("At least four paired cases required")
    gains = []
    drifts = []
    for key in sorted(keys):
        rows = [m[key] for m in maps]
        if len({r["prompt_sha256"] for r in rows}) != 1:
            raise ValueError("Prompt identity mismatch")
        v = [r["aggregate_e2e_tps"] for r in rows]
        if any(not math.isfinite(x) or x <= 0 for x in v):
            raise ValueError("Invalid throughput")
        baseline = math.sqrt(v[0] * v[2])
        gains.append(math.log(v[1] / baseline))
        drifts.append(math.log(v[2] / v[0]))
    rng = random.Random(20261025)
    samples = sorted(
        math.expm1(statistics.mean(rng.choices(gains, k=len(gains))))
        for _ in range(bootstrap)
    )
    low = samples[int(0.025 * bootstrap)]
    high = samples[min(bootstrap - 1, int(0.975 * bootstrap))]
    drift = math.expm1(statistics.mean(drifts))
    gain = math.expm1(statistics.mean(gains))
    return dict(
        schema=1,
        paired_cases=len(keys),
        paired_geometric_gain=gain,
        ci95=[low, high],
        baseline_drift=drift,
        quality_passed=quality_passed,
        stable_positive_evidence=quality_passed
        and low > 0
        and abs(drift) <= drift_limit,
        automatic_promotion=False,
        drift_limit=drift_limit,
        caveat="Paired case bootstrap, not independent hardware replication; inspect phase/order/thermal/cache confounds.",
    )
