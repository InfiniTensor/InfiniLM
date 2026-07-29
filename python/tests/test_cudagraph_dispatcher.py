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
            "INFINI_DECODE_CG_PAD_UP",
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

    def _disp(
        self,
        policy: str,
        buckets: str = "16,64,512,1024,2048,4096",
        decode_batches: str = "1,2,4",
    ):
        os.environ["INFINI_CUDAGRAPH_POLICY"] = policy
        os.environ["INFINI_DECODE_CG_BATCHES"] = decode_batches
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
        # Pad-up: bs=3 → next FULL key 4
        mode, key = d.dispatch(BD(num_tokens=3, num_reqs=3, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        self.assertEqual(key.num_tokens, 4)
        self.assertEqual(key.num_reqs, 4)

    def test_decode_batch_pad_up_ladder(self) -> None:
        """Power-of-two ladder {1,2,4,8}: 3→4, 5→8; bs=9 → over_max."""
        d = self._disp("full_and_piecewise", decode_batches="1,2,4,8")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        for bs in (1, 2, 4, 8):
            mode, key = d.dispatch(
                BD(num_tokens=bs, num_reqs=bs, uniform_decode=True)
            )
            self.assertEqual(mode, Mode.FULL)
            self.assertEqual(key.num_tokens, bs)
            self.assertEqual(key.num_reqs, bs)
        mode, key = d.dispatch(BD(num_tokens=3, num_reqs=3, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        self.assertEqual(key.num_tokens, 4)
        mode, key = d.dispatch(BD(num_tokens=5, num_reqs=5, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        self.assertEqual(key.num_tokens, 8)
        mode, key = d.dispatch(BD(num_tokens=7, num_reqs=7, uniform_decode=True))
        self.assertEqual(mode, Mode.FULL)
        self.assertEqual(key.num_tokens, 8)
        mode, _ = d.dispatch(BD(num_tokens=9, num_reqs=9, uniform_decode=True))
        self.assertEqual(mode, Mode.NONE)
        self.assertEqual(
            d.none_reason(BD(num_tokens=9, num_reqs=9, uniform_decode=True), False),
            "decode_bs_over_max",
        )

    def test_decode_pad_up_kill_switch(self) -> None:
        os.environ["INFINI_DECODE_CG_PAD_UP"] = "0"
        d = self._disp("full_and_piecewise", decode_batches="1,2,4,8")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        mode, _ = d.dispatch(BD(num_tokens=3, num_reqs=3, uniform_decode=True))
        self.assertEqual(mode, Mode.NONE)
        self.assertEqual(
            d.none_reason(BD(num_tokens=3, num_reqs=3, uniform_decode=True), False),
            "decode_bs_miss",
        )

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

    def test_mixed_or_multi_req_piecewise(self) -> None:
        """vLLM mixed_mode=PIECEWISE: multi-req ragged pads num_tokens to bucket."""
        d = self._disp("full_and_piecewise")
        BD = self.mod.BatchDescriptor
        Mode = self.mod.CudaGraphRuntimeMode
        # Multi-req prefill total tokens hit a bucket → PIECEWISE
        mode, key = d.dispatch(BD(num_tokens=512, num_reqs=2, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 512)
        self.assertEqual(key.num_reqs, 2)
        # Ragged mixed shape (decode+prefill rows): pad-up 64 stays 64
        mode, key = d.dispatch(BD(num_tokens=64, num_reqs=3, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 64)
        self.assertEqual(key.num_reqs, 3)
        # Pad-up mid-ladder: e.g. 1 decode + 650 prefill → 1024
        mode, key = d.dispatch(BD(num_tokens=651, num_reqs=2, uniform_decode=False))
        self.assertEqual(mode, Mode.PIECEWISE)
        self.assertEqual(key.num_tokens, 1024)
        self.assertEqual(key.num_reqs, 2)
        # Over max still NONE
        mode, _ = d.dispatch(BD(num_tokens=5000, num_reqs=2, uniform_decode=False))
        self.assertEqual(mode, Mode.NONE)
        self.assertEqual(
            d.none_reason(BD(num_tokens=5000, num_reqs=2, uniform_decode=False), True),
            "over_max",
        )


if __name__ == "__main__":
    unittest.main()
