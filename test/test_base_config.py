import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from bench.backends.infinilm import InfiniLMBenchmark
from infinilm.base_config import BaseConfig


class TestMarsBaseConfig(unittest.TestCase):
    def make_config(self, device="mars", weight_load_mode="async"):
        config = BaseConfig.__new__(BaseConfig)
        config.device = device
        config.weight_load_mode = weight_load_mode
        return config

    def test_explicit_mars_forces_sync_weight_loading(self):
        config = self.make_config()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config._force_sync_weight_loading()
        self.assertEqual(config.weight_load_mode, "sync")

    def test_auto_detects_mars_from_hpcc(self):
        config = self.make_config(device="auto")
        config._torch_device_available = lambda _device_type: False
        with mock.patch.dict(os.environ, {"HPCC_PATH": "/opt/hpcc"}, clear=True):
            with mock.patch("infinilm.base_config.shutil.which", return_value=None):
                self.assertEqual(config.detect_device(), "mars")

    def test_auto_detects_metax_from_maca(self):
        config = self.make_config(device="auto")
        config._torch_device_available = lambda _device_type: False
        with mock.patch.dict(os.environ, {"MACA_PATH": "/opt/maca"}, clear=True):
            with mock.patch("infinilm.base_config.shutil.which", return_value=None):
                self.assertEqual(config.detect_device(), "metax")

    def test_mars_uses_cuda_torch_device(self):
        config = self.make_config()
        self.assertEqual(config.get_device_str("mars"), "cuda")


class TestInfiniLMBenchmark(unittest.TestCase):
    @mock.patch("infinilm.LLM")
    def test_forwards_sync_weight_loading(self, llm):
        processor = mock.Mock()
        processor.get_tokenizer.return_value = mock.Mock()
        llm.return_value.engine.processor = processor

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps({"max_position_embeddings": 2048}),
                encoding="utf-8",
            )
            InfiniLMBenchmark(model_dir, weight_load_mode="sync")

        self.assertEqual(llm.call_args.kwargs["weight_load_mode"], "sync")


if __name__ == "__main__":
    unittest.main()
