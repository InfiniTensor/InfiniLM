"""Tests for /metadata and /metrics observability endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from infinilm.server.inference_server import InferenceServer
from infinimetadata.metrics import MetricsRegistry


class MetricsRegistryTest(unittest.TestCase):
    def test_record_request_finish(self):
        reg = MetricsRegistry("test-server-id")
        reg.record_request_finish(
            status="ok",
            arrival_time=100.0,
            finished_time=100.5,
            first_token_time=100.1,
            prompt_tokens=10,
            completion_tokens=5,
        )
        snap = reg.json_snapshot()
        self.assertEqual(snap["server_id"], "test-server-id")
        self.assertEqual(snap["counters"]["requests_total_ok"], 1.0)
        self.assertEqual(snap["counters"]["tokens_prompt_total"], 10.0)
        self.assertEqual(snap["histograms"]["request_ttft_seconds"]["count"], 1.0)
        text = reg.prometheus_text()
        self.assertIn("infinilm_requests_total", text)
        self.assertIn('status="ok"', text)


class InferenceServerObservabilityTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["INFINI_SERVER_ARTIFACT_ROOT"] = self._tmpdir.name
        os.environ["INFINI_SERVER_FILE_LOG"] = "0"
        self.server = InferenceServer(
            model_path="/tmp/test_model_dir",
            device="cpu",
            host="127.0.0.1",
            port=9999,
        )
        self.server.engine = MagicMock()
        self.server.engine.is_healthy = MagicMock(return_value=True)
        self.server._write_startup_artifacts()
        self.client = TestClient(self.server._create_app())

    def tearDown(self):
        self._tmpdir.cleanup()
        os.environ.pop("INFINI_SERVER_ARTIFACT_ROOT", None)
        os.environ.pop("INFINI_SERVER_FILE_LOG", None)

    def test_metadata_stable_server_id(self):
        r1 = self.client.get("/metadata")
        self.assertEqual(r1.status_code, 200)
        meta = r1.json()
        self.assertEqual(meta["server_id"], self.server.server_id)
        self.assertIn("startup_args", meta)
        self.assertIn("config", meta)
        self.assertIn("build_info", meta)
        self.assertIn("runtime_env", meta)
        self.assertEqual(meta["config"]["startup"], meta["startup_args"])
        self.assertIn("artifact_dir", meta)
        self.assertEqual(meta["model_id"], "test_model_dir")

        r2 = self.client.get("/v1/metadata")
        self.assertEqual(r2.json()["server_id"], self.server.server_id)

    def test_metrics_prometheus_only(self):
        r_prom = self.client.get("/metrics")
        self.assertEqual(r_prom.status_code, 200)
        self.assertIn("text/plain", r_prom.headers.get("content-type", ""))
        self.assertIn("infinilm_requests_total", r_prom.text)
        self.assertIn("infinilm_request_ttft_seconds_p50", r_prom.text)

    def test_metadata_json_on_disk(self):
        meta_path = os.path.join(self.server.artifact_dir, "metadata.json")
        self.assertTrue(os.path.isfile(meta_path))
        with open(meta_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["server_id"], self.server.server_id)

    def test_record_request_metrics_integration(self):
        handle = self.server.obs.begin_request(arrival_time=1000.0)
        self.server.obs.on_request_token(handle, now=1000.05)
        self.server.obs.on_request_finish(
            handle,
            status="ok",
            finished_time=1000.2,
            prompt_tokens=3,
            completion_tokens=2,
        )

        assert self.server.obs is not None
        snap = self.server.obs.json_snapshot()
        self.assertEqual(snap["counters"]["requests_total_ok"], 1.0)
        self.assertEqual(snap["histograms"]["request_ttft_seconds"]["count"], 1.0)

    def test_new_server_gets_new_id(self):
        other = InferenceServer(model_path="/tmp/other_model", device="cpu")
        self.assertNotEqual(other.server_id, self.server.server_id)


if __name__ == "__main__":
    unittest.main()
