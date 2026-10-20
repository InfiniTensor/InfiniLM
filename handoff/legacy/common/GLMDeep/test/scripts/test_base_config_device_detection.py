"""Behavior contracts for CLI device detection without a native extension fixture."""

import argparse
import ast
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "python/infinilm/base_config.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
nodes = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    and node.name in {"parse_list", "BaseConfig"}
]
namespace = {
    "argparse": argparse,
    "importlib": importlib,
    "json": json,
    "os": os,
    "shutil": shutil,
    "warnings": warnings,
    "MOE_EP_BACKEND_HELP": "test backend",
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
BaseConfig = namespace["BaseConfig"]


class BaseConfigDeviceDetectionTest(unittest.TestCase):
    def config(self, *options):
        with patch.object(sys, "argv", ["test", "--model", "fixture", *options]):
            return BaseConfig()

    def device_string(self, config):
        with contextlib.redirect_stdout(io.StringIO()):
            return config.get_device_str(config.device)

    def test_native_priority_and_single_auto_probe(self):
        order = (
            "cambricon", "ascend", "moore", "metax", "hygon",
            "iluvatar", "nvidia",
        )
        aliases = {
            "cambricon": "mlu", "ascend": "npu", "moore": "musa",
            "nvidia": "cuda",
        }
        for backend in ("cpp", "infinilm", "python"):
            for index, chosen in enumerate(order):
                with self.subTest(backend=backend, chosen=chosen):
                    calls = []

                    def get_device_count(name):
                        calls.append(name)
                        return int(name == chosen or name in order[index + 1:])

                    native = SimpleNamespace(get_device_count=get_device_count)
                    with patch.object(importlib, "import_module", return_value=native) as load, \
                         patch.object(shutil, "which", side_effect=AssertionError("CLI probe")), \
                         patch.object(os, "getenv", side_effect=AssertionError("env probe")):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            config = self.config("--backend", backend)
                        self.assertEqual(self.device_string(config), aliases.get(chosen, chosen))
                    self.assertEqual(calls, list(order[:index + 1]))
                    load.assert_called_once_with("infinilm._core")

    def test_native_none_is_cpu_and_never_imports_torch(self):
        modules = []
        names = []

        def load(name):
            modules.append(name)
            self.assertEqual(name, "infinilm._core")
            return SimpleNamespace(get_device_count=lambda device: names.append(device) or 0)

        with patch.object(importlib, "import_module", side_effect=load), \
             patch.object(shutil, "which", side_effect=AssertionError("CLI probe")), \
             patch.object(os, "getenv", side_effect=AssertionError("env probe")):
            config = self.config()
            self.assertEqual(self.device_string(config), "cpu")
        self.assertEqual(modules, ["infinilm._core"])
        self.assertEqual(len(names), 7)

    def test_native_sdk_error_propagates(self):
        def fail(_device):
            raise RuntimeError("RT enumeration failed")

        native = SimpleNamespace(get_device_count=fail)
        with patch.object(importlib, "import_module", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "RT enumeration failed"):
                self.config()

    def test_missing_native_extension_does_not_fall_back(self):
        with patch.object(importlib, "import_module", side_effect=ImportError("missing native")) as load, \
             patch.object(shutil, "which", side_effect=AssertionError("CLI probe")), \
             patch.object(os, "getenv", side_effect=AssertionError("env probe")):
            with self.assertRaisesRegex(ImportError, "missing native"):
                self.config()
        load.assert_called_once_with("infinilm._core")

    def test_explicit_device_does_not_probe(self):
        with patch.object(importlib, "import_module", side_effect=AssertionError("import")), \
             patch.object(shutil, "which", side_effect=AssertionError("CLI probe")):
            for backend in ("cpp", "torch", "transformers", "vllm"):
                with self.subTest(backend=backend):
                    config = self.config("--backend", backend, "--device", "nvidia")
                    self.assertEqual(self.device_string(config), "cuda")

    def test_unknown_explicit_device_is_rejected(self):
        with patch.object(importlib, "import_module", side_effect=AssertionError("import")):
            config = self.config("--device", "unknown-device")
            with self.assertRaisesRegex(ValueError, "unsupported device platform 'unknown-device'"):
                self.device_string(config)

    def test_auto_metax_forces_sync_and_uses_cached_choice(self):
        counts = []

        def get_device_count(name):
            counts.append(name)
            return int(name == "metax")

        native = SimpleNamespace(get_device_count=get_device_count)
        with patch.object(importlib, "import_module", return_value=native) as load:
            with self.assertWarnsRegex(UserWarning, "Auto-detected Metax"):
                config = self.config()
            self.assertEqual(config.weight_load_mode, "sync")
            self.assertEqual(self.device_string(config), "metax")
        self.assertEqual(counts, ["cambricon", "ascend", "moore", "metax"])
        load.assert_called_once_with("infinilm._core")

    def test_reference_backends_keep_torch_provider_detection(self):
        for backend in ("torch", "transformers", "vllm"):
            with self.subTest(backend=backend):
                modules = []
                fake_torch = SimpleNamespace(
                    mlu=SimpleNamespace(is_available=lambda: False),
                    npu=SimpleNamespace(is_available=lambda: True),
                )

                def load(name):
                    modules.append(name)
                    if name == "torch":
                        return fake_torch
                    self.assertNotEqual(name, "infinilm._core")
                    return SimpleNamespace()

                with patch.object(importlib, "import_module", side_effect=load):
                    config = self.config("--backend", backend)
                    self.assertEqual(self.device_string(config), "npu")
                self.assertEqual(modules, ["torch_mlu", "torch", "torch_npu", "torch"])

    def test_reference_cli_fallback_remains_available_without_torch(self):
        with patch.object(importlib, "import_module", side_effect=ImportError("no torch")), \
             patch.object(os, "getenv", return_value=None), \
             patch.object(shutil, "which", side_effect=lambda name: name == "nvidia-smi"):
            config = self.config("--backend", "transformers")
            self.assertEqual(self.device_string(config), "cuda")


if __name__ == "__main__":
    unittest.main()
