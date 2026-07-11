from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FakeRequest:
    request_id: str
    prefill_chunk_start: int = 0
    prefill_chunk_end: int = 0


def _scheduler_stubs():
    infinilm_pkg = types.ModuleType("infinilm")
    llm_pkg = types.ModuleType("infinilm.llm")
    request_mod = types.ModuleType("infinilm.llm.request")
    cache_manager_mod = types.ModuleType("infinilm.llm.cache_manager")

    class RequestStatus:
        WAITING = "waiting"
        RUNNING = "running"
        WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"
        FINISHED = "finished"
        CANCELED = "canceled"
        FAILED = "failed"
        TIMEOUT = "timeout"

    class BlockManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("BlockManager is not used by these tests")

    class MambaCacheManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("MambaCacheManager is not used by these tests")

    request_mod.RequestStatus = RequestStatus
    request_mod.InferenceRequest = object
    cache_manager_mod.BlockManager = BlockManager
    cache_manager_mod.MambaCacheManager = MambaCacheManager

    return {
        "infinilm": infinilm_pkg,
        "infinilm.llm": llm_pkg,
        "infinilm.llm.request": request_mod,
        "infinilm.llm.cache_manager": cache_manager_mod,
    }


def _model_runner_stubs():
    stubs = {
        "infinicore": types.ModuleType("infinicore"),
        "infinilm.distributed": types.ModuleType("infinilm.distributed"),
        "infinilm.infer_engine": types.ModuleType("infinilm.infer_engine"),
        "infinilm.cache": types.ModuleType("infinilm.cache"),
        "infinilm.cache.cache": types.ModuleType("infinilm.cache.cache"),
        "infinilm.modeling_utils": types.ModuleType("infinilm.modeling_utils"),
        "infinilm.config": types.ModuleType("infinilm.config"),
        "infinilm.config.engine_config": types.ModuleType(
            "infinilm.config.engine_config"
        ),
        "infinilm.kv_connector": types.ModuleType("infinilm.kv_connector"),
        "infinilm.processors": types.ModuleType("infinilm.processors"),
    }
    stubs["infinilm.distributed"].DistConfig = object
    stubs["infinilm.infer_engine"].InferEngine = object
    stubs["infinilm.cache.cache"].PagedKVCacheConfig = object
    stubs["infinilm.cache.cache"].StaticKVCacheConfig = object
    stubs["infinilm.modeling_utils"].load_model_state_dict_by_file = (
        lambda *args, **kwargs: None
    )
    stubs["infinilm.config.engine_config"].EngineConfig = object
    stubs["infinilm.kv_connector"].KVConnectorRole = SimpleNamespace(WORKER=1)
    stubs["infinilm.kv_connector"].KVConnectorFactory = object
    stubs["infinilm.processors"].AutoInfinilmProcessor = object
    return stubs


def _load_module(module_name: str, relative_path: str, stubs: dict[str, types.ModuleType]):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def load_scheduler_module():
    return _load_module(
        "_phase_test_scheduler",
        "python/infinilm/llm/scheduler.py",
        _scheduler_stubs(),
    )


def load_model_runner_module():
    stubs = _scheduler_stubs()
    stubs.update(_model_runner_stubs())
    return _load_module(
        "_phase_test_model_runner",
        "python/infinilm/llm/model_runner/model_runner.py",
        stubs,
    )


class PhaseAwareExecutionTest(unittest.TestCase):
    def test_scheduler_output_builds_decode_first_phase_plan(self):
        scheduler_module = load_scheduler_module()
        output = scheduler_module.SchedulerOutput(
            scheduled_requests=[
                FakeRequest("decode-0"),
                FakeRequest(
                    "prefill-0", prefill_chunk_start=32, prefill_chunk_end=96
                ),
                FakeRequest("decode-1"),
            ],
            scheduled_prefill_flags=[False, True, False],
        )

        self.assertTrue(output.is_mixed)
        self.assertTrue(output.is_prefill)
        self.assertEqual(output.num_decode_tokens, 2)
        self.assertEqual(output.num_prefill_tokens, 64)
        self.assertEqual(output.num_batched_tokens, 66)
        self.assertEqual(
            [
                (phase.kind, phase.request_indices, phase.num_tokens)
                for phase in output.execution_phases
            ],
            [
                ("decode", (0, 2), 2),
                ("prefill", (1,), 64),
            ],
        )

        decode_subset = output.make_subset((0, 2))
        self.assertFalse(decode_subset.is_prefill)
        self.assertEqual(
            [req.request_id for req in decode_subset.scheduled_requests],
            ["decode-0", "decode-1"],
        )
        self.assertEqual(decode_subset.num_decode_tokens, 2)
        self.assertEqual(decode_subset.num_prefill_tokens, 0)

        prefill_subset = output.make_subset((1,))
        self.assertTrue(prefill_subset.is_prefill)
        self.assertEqual(
            [req.request_id for req in prefill_subset.scheduled_requests],
            ["prefill-0"],
        )
        self.assertEqual(prefill_subset.num_decode_tokens, 0)
        self.assertEqual(prefill_subset.num_prefill_tokens, 64)

    def test_model_runner_splits_mixed_batch_and_restores_output_order(self):
        scheduler_module = load_scheduler_module()
        model_runner_module = load_model_runner_module()
        output = scheduler_module.SchedulerOutput(
            scheduled_requests=[
                FakeRequest("decode-0"),
                FakeRequest("prefill-0", prefill_chunk_start=0, prefill_chunk_end=128),
                FakeRequest("decode-1"),
            ],
            scheduled_prefill_flags=[False, True, False],
        )

        runner = object.__new__(model_runner_module.ModelRunner)
        runner.config = SimpleNamespace(cache_type="paged")
        calls = []

        def fake_forward_single(sub_output):
            calls.append(
                {
                    "is_prefill": sub_output.is_prefill,
                    "request_ids": [
                        req.request_id for req in sub_output.scheduled_requests
                    ],
                    "decode_tokens": sub_output.num_decode_tokens,
                    "prefill_tokens": sub_output.num_prefill_tokens,
                }
            )
            return [100 + len(calls)] * sub_output.num_requests

        runner._model_forward_single = fake_forward_single

        sampled_tokens, stats = runner._model_forward(output)

        self.assertEqual(
            calls,
            [
                {
                    "is_prefill": False,
                    "request_ids": ["decode-0", "decode-1"],
                    "decode_tokens": 2,
                    "prefill_tokens": 0,
                },
                {
                    "is_prefill": True,
                    "request_ids": ["prefill-0"],
                    "decode_tokens": 0,
                    "prefill_tokens": 128,
                },
            ],
        )
        self.assertEqual(sampled_tokens, [101, 102, 101])
        self.assertEqual(
            stats,
            {
                "split_mixed_batch": True,
                "phases": [
                    {"kind": "decode", "num_requests": 2, "num_tokens": 2},
                    {"kind": "prefill", "num_requests": 1, "num_tokens": 128},
                ],
            },
        )

    def test_model_runner_keeps_static_cache_mixed_batch_unsplit(self):
        scheduler_module = load_scheduler_module()
        model_runner_module = load_model_runner_module()
        output = scheduler_module.SchedulerOutput(
            scheduled_requests=[
                FakeRequest("decode-0"),
                FakeRequest("prefill-0", prefill_chunk_start=0, prefill_chunk_end=16),
            ],
            scheduled_prefill_flags=[False, True],
        )

        runner = object.__new__(model_runner_module.ModelRunner)
        runner.config = SimpleNamespace(cache_type="static")
        calls = []

        def fake_forward_single(sub_output):
            calls.append([req.request_id for req in sub_output.scheduled_requests])
            return [7] * sub_output.num_requests

        runner._model_forward_single = fake_forward_single

        sampled_tokens, stats = runner._model_forward(output)

        self.assertEqual(calls, [["decode-0", "prefill-0"]])
        self.assertEqual(sampled_tokens, [7, 7])
        self.assertFalse(stats["split_mixed_batch"])
        self.assertEqual(
            stats["phases"],
            [
                {"kind": "decode", "num_requests": 1, "num_tokens": 1},
                {"kind": "prefill", "num_requests": 1, "num_tokens": 16},
            ],
        )

    def test_scheduler_records_phase_execution_stats(self):
        scheduler_module = load_scheduler_module()
        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler._stats = {
            "execution_split_mixed_steps": 0,
            "execution_decode_phases": 0,
            "execution_prefill_phases": 0,
            "execution_decode_phase_tokens": 0,
            "execution_prefill_phase_tokens": 0,
        }

        scheduler._record_execution_stats(
            {
                "split_mixed_batch": True,
                "phases": [
                    {"kind": "decode", "num_tokens": 2},
                    {"kind": "prefill", "num_tokens": 128},
                ],
            }
        )

        self.assertEqual(scheduler._stats["execution_split_mixed_steps"], 1)
        self.assertEqual(scheduler._stats["execution_decode_phases"], 1)
        self.assertEqual(scheduler._stats["execution_prefill_phases"], 1)
        self.assertEqual(scheduler._stats["execution_decode_phase_tokens"], 2)
        self.assertEqual(scheduler._stats["execution_prefill_phase_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
