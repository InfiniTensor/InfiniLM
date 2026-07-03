"""Tests for inference_server ITL metrics."""

from __future__ import annotations

import unittest

from infinimetadata.metrics import MetricsRegistry


class TestServerItlMetrics(unittest.TestCase):
    def test_itl_histogram_in_json_snapshot(self) -> None:
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
        reg = MetricsRegistry("00000000-0000-4000-8000-000000000002")
        reg.record_inter_token_latency(0.0)
        reg.record_inter_token_latency(-0.01)
        snap = reg.json_snapshot()
        self.assertEqual(snap["histograms"]["request_itl_seconds"]["count"], 0.0)


if __name__ == "__main__":
    unittest.main()
