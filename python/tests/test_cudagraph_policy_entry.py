# Copyright (c) 2025, InfiniCore
"""Unit tests for INFINI_CUDAGRAPH_POLICY + entry --cudagraph-policy."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


def _load_env_module():
    """Load ``compile/env.py`` without importing ``compile/__init__.py`` (needs _infinicore)."""
    path = Path(__file__).resolve().parents[1] / "infinilm" / "compile" / "env.py"
    spec = importlib.util.spec_from_file_location("infinilm_compile_env_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_compile_env_stub(env_mod) -> None:
    if "infinilm" not in sys.modules:
        pkg = types.ModuleType("infinilm")
        pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "infinilm")]
        sys.modules["infinilm"] = pkg
    if "infinilm.compile" not in sys.modules:
        compile_pkg = types.ModuleType("infinilm.compile")
        compile_pkg.__path__ = [
            str(Path(__file__).resolve().parents[1] / "infinilm" / "compile")
        ]
        sys.modules["infinilm.compile"] = compile_pkg
    sys.modules["infinilm.compile.env"] = env_mod


def _load_entry_module():
    """Load entry.py with a stub compile.env to avoid InfiniCore at import time."""
    env_mod = _load_env_module()
    _ensure_compile_env_stub(env_mod)

    path = Path(__file__).resolve().parents[1] / "infinilm" / "server" / "entry.py"
    spec = importlib.util.spec_from_file_location("infinilm_server_entry_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCudagraphPolicyEnv(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            k: os.environ.get(k)
            for k in (
                "INFINI_CUDAGRAPH_POLICY",
                "INFINI_FA_FORCE_CAPTURE",
                "INFINI_PREFILL_NATIVE_CG",
                "INFINI_DECODE_GRAPH_ONLY",
                "INFINI_DECODE_PIECEWISE",
                "INFINI_SKIP_MONOLITHIC_DECODE_CG",
                "INFINI_DECODE_CG_BATCHES",
                "INFINI_NATIVE_CG_CAPTURE_BUCKETS",
                "INFINI_MUL_HOST_BREAK",
                "INFINI_MOE_TRITON_CAPTURE",
            )
        }
        for k in self._env_backup:
            os.environ.pop(k, None)
        self.env = _load_env_module()
        self.env._FA_FORCE_POLICY_WARNED = False

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_eager(self) -> None:
        self.assertEqual(self.env.cudagraph_policy(), "eager")

    def test_reject_track_b(self) -> None:
        os.environ["INFINI_CUDAGRAPH_POLICY"] = "track_b"
        with self.assertRaises(ValueError):
            self.env.cudagraph_policy()

    def test_apply_full_and_piecewise_expands_knobs(self) -> None:
        p = self.env.apply_cudagraph_policy_env("full_and_piecewise")
        self.assertEqual(p, "full_and_piecewise")
        self.assertEqual(os.environ["INFINI_CUDAGRAPH_POLICY"], "full_and_piecewise")
        self.assertEqual(os.environ["INFINI_PREFILL_NATIVE_CG"], "1")
        self.assertEqual(os.environ["INFINI_DECODE_GRAPH_ONLY"], "0")
        self.assertEqual(os.environ["INFINI_DECODE_PIECEWISE"], "0")
        self.assertEqual(os.environ["INFINI_DECODE_CG_BATCHES"], "1,2,4")
        self.assertIn("2048", os.environ["INFINI_NATIVE_CG_CAPTURE_BUCKETS"])
        self.assertNotIn("INFINI_FA_FORCE_CAPTURE", os.environ)
        self.assertNotIn("INFINI_MOE_TRITON_CAPTURE", os.environ)

    def test_apply_respects_prefill_native_override(self) -> None:
        os.environ["INFINI_PREFILL_NATIVE_CG"] = "0"
        self.env.apply_cudagraph_policy_env("full_and_piecewise")
        self.assertEqual(os.environ["INFINI_PREFILL_NATIVE_CG"], "0")

    def test_apply_eager_disables_cg(self) -> None:
        self.env.apply_cudagraph_policy_env("eager")
        self.assertEqual(os.environ["INFINI_PREFILL_NATIVE_CG"], "0")
        self.assertEqual(os.environ["INFINI_DECODE_GRAPH_ONLY"], "1")
        self.assertEqual(os.environ["INFINI_DECODE_PIECEWISE"], "0")


class TestEntryCudagraphPolicyCli(unittest.TestCase):
    def setUp(self) -> None:
        self._keys = (
            "INFINI_CUDAGRAPH_POLICY",
            "INFINI_PREFILL_NATIVE_CG",
            "INFINI_DECODE_GRAPH_ONLY",
            "INFINI_DECODE_PIECEWISE",
            "INFINI_DECODE_CG_BATCHES",
            "INFINI_NATIVE_CG_CAPTURE_BUCKETS",
            "INFINI_FA_FORCE_CAPTURE",
        )
        self._backup = {k: os.environ.get(k) for k in self._keys}
        for k in self._keys:
            os.environ.pop(k, None)
        self.entry = _load_entry_module()

    def tearDown(self) -> None:
        for k, v in self._backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_peel_sets_full_and_piecewise(self) -> None:
        args, remaining = self.entry._peel_entry_args(
            ["--phase", "serve", "--cudagraph-policy", "full_and_piecewise", "--model", "x"]
        )
        self.assertEqual(args.cudagraph_policy, "full_and_piecewise")
        self.assertEqual(remaining, ["--model", "x"])
        self.entry._apply_entry_cudagraph_policy(args.cudagraph_policy)
        self.assertEqual(os.environ["INFINI_CUDAGRAPH_POLICY"], "full_and_piecewise")
        self.assertEqual(os.environ["INFINI_DECODE_CG_BATCHES"], "1,2,4")

    def test_cli_wins_over_prior_env(self) -> None:
        os.environ["INFINI_CUDAGRAPH_POLICY"] = "eager"
        args, _ = self.entry._peel_entry_args(["--cudagraph-policy", "full_and_piecewise"])
        self.entry._apply_entry_cudagraph_policy(args.cudagraph_policy)
        self.assertEqual(os.environ["INFINI_CUDAGRAPH_POLICY"], "full_and_piecewise")

    def test_default_eager(self) -> None:
        args, _ = self.entry._peel_entry_args(["--phase", "serve"])
        self.assertEqual(args.cudagraph_policy, "eager")

    def test_help_lists_policies(self) -> None:
        import io
        from contextlib import redirect_stdout

        if "infinilm.base_config" not in sys.modules:
            bc = types.ModuleType("infinilm.base_config")

            class BaseConfig:
                def __new__(cls):
                    return object.__new__(cls)

                @staticmethod
                def _add_common_args(stub):
                    stub.parser.add_argument("--model")

            bc.BaseConfig = BaseConfig
            sys.modules["infinilm.base_config"] = bc

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.entry.main(["--help"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("full_and_piecewise", text)
        self.assertIn("eager", text)
        self.assertIn("cudagraph-policy", text)


if __name__ == "__main__":
    unittest.main()
