"""Tests for inference_server ITL metrics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_metrics_registry():
    metrics_path = Path(__file__).resolve().parents[1] / "infinilm" / "server" / "metrics.py"
    spec = importlib.util.spec_from_file_location("infinilm_server_metrics", metrics_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MetricsRegistry


class TestServerItlMetrics(unittest.TestCase):
    def test_itl_histogram_in_json_snapshot(self) -> None:
        MetricsRegistry = _load_metrics_registry()
        reg = MetricsRegistry("00000000-0000-4000-8000-000000000001")
        reg.record_inter_token_latency(0.05)
        reg.record_inter_token_latency(0.10)
        reg.record_inter_token_latency(0.15)

        snap = reg.json_snapshot()
        itl = snap["histograms"]["request_itl_seconds"]
        self.assertEqual(itl["count"], 3.0)
        self.assertGreater(itl["p50"], 0.0)
        self.assertGreater(itl["p99"], 0.0)

    def test_zero_itl_not_recorded(self) -> None:
        MetricsRegistry = _load_metrics_registry()
        reg = MetricsRegistry("00000000-0000-4000-8000-000000000002")
        reg.record_inter_token_latency(0.0)
        reg.record_inter_token_latency(-0.01)
        snap = reg.json_snapshot()
        self.assertEqual(snap["histograms"]["request_itl_seconds"]["count"], 0.0)


if __name__ == "__main__":
    unittest.main()
