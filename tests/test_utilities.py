from pathlib import Path
import tempfile
import unittest
from sparkfleet.utilities import (
    weight_manifest,
    verify_weights,
    public_audit,
    parse_iperf,
)


class UtilityTests(unittest.TestCase):
    def test_weights_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weights = root / "weights"
            weights.mkdir()
            (weights / "shard").write_bytes(b"fixture")
            m = weight_manifest(weights, root / "manifest.json")
            self.assertTrue(verify_weights(weights, m)["passed"])
            (weights / "shard").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_weights(weights, m)

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "link").symlink_to("/dev/null")
            with self.assertRaises(ValueError):
                weight_manifest(root, root.parent / "never-written.json")

    def test_public_export_scan(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "example.txt"
            p.write_text("Safe documentation")
            self.assertTrue(public_audit(td)["passed"])
            p.write_text("192.168.1.2 or ghp_" + "A" * 30)
            self.assertFalse(public_audit(td)["passed"])

    def test_network_units(self):
        r = parse_iperf({"end": {"sum_received": {"bits_per_second": 180e9}}})
        self.assertEqual(r["received_gbyte_s"], 22.5)
        self.assertEqual(r["fraction_of_200g_line_rate"], 0.9)
        with self.assertRaises(ValueError):
            parse_iperf({"error": "failed"})
