import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from sparkfleet import agent, cluster
from sparkfleet.common import canonical, digest
from sparkfleet.config import validate
from sparkfleet.profiles import example
from sparkfleet.render import commands, render
from sparkfleet.adapters import DS, GLM, normalize_messages, responses_input
from sparkfleet.overlays import inspect_overlay, materialize
from sparkfleet.benchmark import compare


def operational(profile="deepseek-tp4"):
    c = example(profile)
    c["runtime"].update(
        image="registry.invalid/runtime@sha256:" + "1" * 64,
        compatible_runtime_confirmed=True,
    )
    c["network"].update(hca="rocep1s0f0", socket_interface="enp1s0f0")
    for i, n in enumerate(c["nodes"]):
        n.update(ssh=f"operator@spark-{i + 1}.invalid", data_ip=f"100.64.0.{i + 1}")
    return c


def observed():
    return dict(
        architecture="aarch64",
        gpu_name="NVIDIA GB10",
        gpu_c=55,
        available_gib=20,
        compute_pids=[],
        locks=[{"available": True}] * 4,
        boot_id="boot",
        image_id="sha256:" + "1" * 64,
        overlay_hashes={},
        env_file_sha256=None,
    )


class ConfigTests(unittest.TestCase):
    def test_templates_cannot_execute(self):
        c = example("deepseek-tp4")
        validate(c, template=True)
        with self.assertRaises(ValueError):
            validate(c)

    def test_both_topologies(self):
        for profile, nodes in [("deepseek-tp4", 4), ("glm-tp8", 8)]:
            rows = commands(operational(profile), "test")
            self.assertEqual(len(rows), nodes)
            self.assertNotIn("--headless", rows[0]["engine_argv"])
            self.assertTrue(all("--headless" in r["engine_argv"] for r in rows[1:]))
            self.assertIn(
                "--prefix-cache-retention-interval", rows[0]["engine_argv"]
            ) if nodes == 4 else None

    def test_node_specific_fabric(self):
        c = operational()
        c["nodes"][1]["fabric"] = {"hca": "rdma1", "gid_index": 2}
        self.assertEqual(commands(c, "test")[1]["env"]["NCCL_IB_HCA"], "rdma1")

    def test_reject_unsafe_changes(self):
        mutations = [
            lambda c: c["nodes"][0].update(ssh="operator@host;touch BAD"),
            lambda c: c["paths"].update(weights="/srv/../secret"),
            lambda c: c["guard"].update(abort_c=86),
            lambda c: c["guard"].update(memory_floor_gib=7),
            lambda c: c["serving"].update(host="0.0.0.0"),
            lambda c: c["nodes"].pop(),
        ]
        for mutate in mutations:
            c = operational()
            mutate(c)
            with self.assertRaises(ValueError):
                validate(c)

    def test_render_is_only_review(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            render(example("deepseek-tp4"), out, "preview", template=True)
            self.assertIn("# REVIEW ONLY", next(out.glob("*.sh")).read_text())
            self.assertTrue(
                json.loads((out / "manifest.json").read_text())["preview_only"]
            )


class GuardTests(unittest.TestCase):
    def test_admission_boundaries_and_workload(self):
        row = commands(operational(), "test")[0]
        s = observed()
        agent.admission(s, row, s)
        for changes in [
            {"gpu_c": 85},
            {"available_gib": 7.99},
            {"compute_pids": ["123"]},
            {"locks": [{"available": False}]},
            {"architecture": "x86_64"},
        ]:
            with self.assertRaises(ValueError):
                agent.admission(dict(s, **changes), row)
        with self.assertRaises(ValueError):
            agent.admission(dict(s, boot_id="new"), row, s)

    def test_exact_owned_container(self):
        c = commands(operational(), "test")[0]
        c["observed"] = observed()
        x = {
            "Config": {
                "Labels": {
                    "org.sparkfleet.managed": "true",
                    "org.sparkfleet.cluster": c["cluster"],
                    "org.sparkfleet.release": c["release"],
                },
                "Entrypoint": c["engine_argv"][:1],
                "Cmd": c["engine_argv"][1:],
            },
            "Image": c["observed"]["image_id"],
        }
        self.assertTrue(agent.owned(c, x))
        x["Config"]["Labels"]["org.sparkfleet.release"] = "somebody-else"
        self.assertFalse(agent.owned(c, x))
        x.update(Id="other", State={"Running": True})
        with (
            patch.object(
                agent.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=json.dumps([x])),
            ),
            patch.object(agent, "call") as call,
        ):
            with self.assertRaises(ValueError):
                agent.stop_owned(c)
            call.assert_not_called()

    def test_daemon_error_never_means_stopped(self):
        c = commands(operational(), "test")[0]
        with (
            patch.object(
                agent.subprocess, "run", return_value=SimpleNamespace(returncode=1)
            ),
            patch.object(agent, "call", side_effect=RuntimeError("daemon down")),
        ):
            with self.assertRaises(RuntimeError):
                agent.stop_owned(c)
        with (
            patch.object(
                agent.subprocess, "run", return_value=SimpleNamespace(returncode=1)
            ),
            patch.object(agent, "call", return_value=""),
        ):
            self.assertTrue(agent.stop_owned(c))

    def test_plan_integrity_and_expiry(self):
        with patch.object(cluster, "probe", return_value=observed()):
            p = cluster.plan(operational())
        cluster.verify_plan(p)
        changed = copy.deepcopy(p)
        changed["nodes"][0]["argv"].append("--bad")
        with self.assertRaises(ValueError):
            cluster.verify_plan(changed)
        stale = copy.deepcopy(p)
        stale["created_at"] = time.time() - 3601
        stale.pop("plan_sha256")
        stale["plan_sha256"] = digest(canonical(stale))
        with self.assertRaises(ValueError):
            cluster.verify_plan(stale)

    def test_doctor_only_probes(self):
        with (
            patch.object(cluster, "probe", return_value=observed()) as probe,
            patch.object(cluster, "stage") as stage,
        ):
            self.assertTrue(cluster.doctor(operational())["read_only"])
            self.assertEqual(probe.call_count, 4)
            stage.assert_not_called()


class AdapterTests(unittest.TestCase):
    def test_tool_payload_opaque_and_deferred_selected(self):
        opaque = {
            "type": "tool_reference",
            "name": "private-shape",
            "role": "developer",
        }
        body = {
            "model": DS,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t", "name": "exec", "input": opaque}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [
                                {"type": "tool_reference", "tool_name": "exec"}
                            ],
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "exec", "defer_loading": True},
                {"name": "unused", "defer_loading": True},
            ],
        }
        result = normalize_messages(body)
        self.assertEqual(result["messages"][0]["content"][0]["input"], opaque)
        self.assertEqual([x["name"] for x in result["tools"]], ["exec"])
        self.assertEqual(result["chat_template_kwargs"]["reasoning_effort"], 50)
        self.assertEqual(normalize_messages(result), result)
        self.assertEqual(body["thinking"]["type"], "adaptive")

    def test_glm_effort_is_selectable(self):
        for effort, expected in [("low", "low"), ("medium", "high"), ("max", "max")]:
            result = normalize_messages(
                {
                    "model": GLM,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": effort},
                }
            )
            self.assertEqual(result["output_config"]["effort"], expected)

    def test_responses_known_envelope_only(self):
        p = {
            "model": DS,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "rule"}],
                },
                {"type": "function_call", "arguments": '{"role":"developer"}'},
            ],
        }
        q = responses_input(p)
        self.assertEqual(
            q["input"][0], {"type": "message", "role": "system", "content": "rule"}
        )
        self.assertEqual(q["input"][1], p["input"][1])


class OverlayTests(unittest.TestCase):
    def test_manifest_payloads(self):
        for p in (Path(__file__).parents[1] / "overlays").iterdir():
            if p.is_dir():
                self.assertTrue(inspect_overlay(p)["passed"])

    def test_preimage_and_no_live_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            overlay = root / "overlay"
            runtime = root / "runtime"
            overlay.mkdir()
            runtime.mkdir()
            (overlay / "x.py").write_text("new")
            (runtime / "x.py").write_text("old")
            m = {
                "name": "fixture",
                "status": "test",
                "files": {
                    "x.py": {
                        "sha256": digest(b"new"),
                        "preimage_sha256": digest(b"old"),
                        "license": "Apache-2.0",
                    }
                },
            }
            (overlay / "manifest.json").write_text(json.dumps(m))
            materialize(overlay, runtime, root / "out")
            self.assertEqual((runtime / "x.py").read_text(), "old")
            with self.assertRaises(ValueError):
                materialize(overlay, runtime, runtime / "bad")
            (runtime / "x.py").write_text("changed")
            with self.assertRaises(ValueError):
                inspect_overlay(overlay, runtime)


class BenchmarkTests(unittest.TestCase):
    def sample(self, speed):
        return dict(
            complete=True,
            model=DS,
            workload={"test": 1},
            samples=[
                dict(pair_key=str(i), prompt_sha256="same", aggregate_e2e_tps=speed)
                for i in range(6)
            ],
        )

    def test_gain_and_noise_gate(self):
        a = self.sample(50)
        b = self.sample(53)
        self.assertTrue(
            compare(a, b, a, quality_passed=True, bootstrap=1000)[
                "stable_positive_evidence"
            ]
        )
        self.assertFalse(
            compare(a, b, self.sample(53.75), quality_passed=True, bootstrap=1000)[
                "stable_positive_evidence"
            ]
        )
        self.assertFalse(
            compare(a, b, a, quality_passed=False, bootstrap=1000)[
                "stable_positive_evidence"
            ]
        )

    def test_mismatched_prompts_rejected(self):
        a = self.sample(50)
        b = self.sample(55)
        b["samples"][0]["prompt_sha256"] = "different"
        with self.assertRaises(ValueError):
            compare(a, b, a, bootstrap=1000)


class QualificationTests(unittest.TestCase):
    def test_failed_backend_cannot_produce_passing_receipt(self):
        import contextlib
        import io
        from sparkfleet.api import qualify

        class FailedBackend:
            def request(self, *args, **kwargs):
                return 503, {"error_type": "api_error"}, 0.01

        with (
            tempfile.TemporaryDirectory() as td,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            path = Path(td) / "receipt.json"
            q = qualify(FailedBackend(), GLM, path)
            self.assertTrue(q["complete"])
            self.assertFalse(q["passed"])
            self.assertEqual(len(q["tests"]), 11)
            self.assertFalse(json.loads(path.read_text())["passed"])

    def test_prelaunch_failure_is_contained_without_staging(self):
        import asyncio

        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(cluster, "probe", return_value=observed()),
        ):
            p = cluster.plan(operational())
            with (
                patch.object(
                    cluster, "probe", side_effect=RuntimeError("fixture unavailable")
                ),
                patch.object(cluster, "stage") as stage,
            ):
                out = Path(td) / "run"
                with self.assertRaises(RuntimeError):
                    asyncio.run(cluster.supervise(p, out))
                stage.assert_not_called()
                self.assertTrue(
                    json.loads((out / "STOPPED.json").read_text())[
                        "all_exact_stops_confirmed"
                    ]
                )


if __name__ == "__main__":
    unittest.main()
