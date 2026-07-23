# Copyright (c) 2025, InfiniCore
"""Unit tests for BatchDescriptor → CudaGraphRuntimeMode dispatch table."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path


def _load_dispatcher():
    path = (
        Path(__file__).resolve().parents[1]
        / "infinilm"
        / "compile"
        / "cudagraph_dispatcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "infinilm_cudagraph_dispatcher_under_test", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCudagraphDispatcher(unittest.TestCase):
    def setUp(self) -> None:
        self._keys = (
            "INFINI_CUDAGRAPH_POLICY",
            "INFINI_DECODE_CG_BATCHES",
            "INFINI_NATIVE_CG_CAPTURE_BUCKETS",
        )
        self._backup = {k: os.environ.get(k) for k in self._keys}
        for k in self._keys:
            os.environ.pop(k, None)
        self.mod = _load_dispatcher()

    def tearDown(self) -> None:
        for k, v in self._backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _disp(self, policy: str, buckets: str = "16,64,512,1024,2048,4096"):
        os.environ["INFINI_CUDAGRAPH_POLICY"] = policy
        os.environ["INFINI_DECODE_CG_BATCHES"] = "1,2,4"
        os.environ["INFINI_NATIVE_CG_CAPTURE_BUCKETS"] = buckets
        d = self.mod.CudagraphDispatcher()
        d.initialize_from_env()
        return d

    def test_eager_always_none(self) -> None:
        d = self._disp("eager")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, _ = d.dispatch(BD(num_tokens=1, num_reqs=1, uniform_decode=True))
        self.assertEqual(mode, Mode.NONE)
        mode, _ = d.dispatch(BD(num_tokens=512, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)

    def test_uniform_decode_full(self) -> None:
        d = self._disp("full_and_piecewise")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, key = d.dispatch(BD(num_tokens=1, num_reqs=1, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        self.assertEqual(key.num_tokens, 1)
        mode, _ = d.dispatch(BD(num_tokens=4, num_reqs=4, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        # Batch size not in FULL keys → NONE
        mode, _ = d.dispatch(BD(num_tokens=3, num_reqs=3, uniform_decode=True))
        self.assertEqual(mode, Mode.NONE)

    def test_bucket_prefill_piecewise_exact(self) -> None:
        d = self._disp("full_and_piecewise")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, key = d.dispatch(BD(num_tokens=2048, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 2048)
        mode, key = d.dispatch(BD(num_tokens=512, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 512)
        mode, key = d.dispatch(BD(num_tokens=16, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 16)

    def test_pad_up_prefill_piecewise(self) -> None:
        """vLLM-style pad-up: 13 → bucket 16 under {16,64,512}."""
        d = self._disp("full_and_piecewise", buckets="16,64,512")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, key = d.dispatch(BD(num_tokens=13, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 16)
        mode, key = d.dispatch(BD(num_tokens=16, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 16)
        # Mid-ladder pad-up
        mode, key = d.dispatch(BD(num_tokens=1000, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)  # 1000 > max 512
        self.assertEqual(d.none_reason(BD(num_tokens=1000, num_reqs=1, uniform_decode=False), False), "over_max")

    def test_pad_up_mid_ladder(self) -> None:
        d = self._disp("full_and_piecewise")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, key = d.dispatch(BD(num_tokens=1000, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 1024)

    def test_past_max_none(self) -> None:
        d = self._disp("full_and_piecewise", buckets="16,64,512")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, _ = d.dispatch(BD(num_tokens=513, num_reqs=1, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)
        self.assertEqual(
            d.none_reason(BD(num_tokens=513, num_reqs=1, uniform_decode=False), False),
            "over_max",
        )

    def test_mixed_or_multi_req_none(self) -> None:
        d = self._disp("full_and_piecewise")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        # Multi-req prefill (even if total tokens hit a bucket) → NONE
        mode, _ = d.dispatch(BD(num_tokens=512, num_reqs=2, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)
        # Ragged mixed shape is not uniform_decode
        mode, _ = d.dispatch(BD(num_tokens=64, num_reqs=3, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)


if __name__ == "__main__":
    unittest.main()
