"""Exercise the real runner boundary without loading a native model."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_chunk_execution
from test_chunk_config import EngineConfig, load_module


def load_runner():
    replacements = {
        "infinicore": SimpleNamespace(),
        "infinilm.cache.cache": SimpleNamespace(
            PagedKVCacheConfig=object, StaticKVCacheConfig=object
        ),
        "infinilm.config.engine_config": SimpleNamespace(EngineConfig=EngineConfig),
        "infinilm.distributed": SimpleNamespace(DistConfig=object),
        "infinilm.distributed.pipeline_transport": SimpleNamespace(
            PipelineControlServer=object
        ),
        "infinilm.infer_engine": SimpleNamespace(InferEngine=object),
        "infinilm.kv_connector": SimpleNamespace(
            KVConnectorFactory=object, KVConnectorRole=object
        ),
        "infinilm.llm.model_runner.speculative_runner": SimpleNamespace(
            SpeculativeRunner=object
        ),
        "infinilm.modeling_utils": SimpleNamespace(
            load_model_state_dict_by_file=object
        ),
        "infinilm.processors": SimpleNamespace(AutoInfinilmProcessor=object),
    }
    with patch.dict(sys.modules, replacements):
        return load_module(
            "infinilm.llm.model_runner.model_runner", "llm/model_runner/model_runner.py"
        ).ModelRunner


Runner = load_runner()


class ChunkOutputTests(unittest.TestCase):
    def setup_runner(self):
        engine, req = test_chunk_execution.ChunkExecutionTests.setup_engine(self)
        runner = Runner.__new__(Runner)
        runner.config = EngineConfig("unused", prefill_chunk_size=16)
        runner.processor = SimpleNamespace(build_model_inputs=lambda *a: {})
        runner.speculative_runner = None
        runner.pipeline_control = None
        runner.kv_connector = None
        calls = []

        def forward(**kwargs):
            calls.append(kwargs.get("prefill_only", False))
            if kwargs.get("prefill_only"):
                return None
            return SimpleNamespace(
                to_numpy=lambda: SimpleNamespace(tolist=lambda: [77])
            )

        runner.model_engine = SimpleNamespace(forward=forward)
        engine.model_runner = runner
        return engine, req, calls

    def test_only_intermediate_chunks_skip_native_output(self):
        engine, req, calls = self.setup_runner()
        for _ in range(2):
            self.assertEqual(engine.step(), (True, []))
            self.assertEqual(list(req.generated_token_ids), [])
        self.assertEqual(calls, [True, True])
        engine.step()
        self.assertEqual(calls, [True, True, False])
        self.assertEqual(list(req.generated_token_ids), [77])
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.cache_manager.blocks)
        )

    def test_decode_keeps_sampling(self):
        engine, req, calls = self.setup_runner()
        req.sampling_params.max_tokens = 2
        for _ in range(4):
            engine.step()
        self.assertEqual(calls, [True, True, False, False])
        self.assertEqual(list(req.generated_token_ids), [77, 77])

    def test_legacy_output_without_chunk_metadata_keeps_sampling(self):
        engine, req, calls = self.setup_runner()
        output = SimpleNamespace(
            scheduled_requests=[req], num_requests=1, is_prefill=True
        )
        result = engine.model_runner.execute_model(output)
        self.assertEqual(calls, [False])
        self.assertEqual(result.sampled_token_ids, [77])

    def test_empty_native_output_still_finishes_cancelled_chunk(self):
        engine, req, calls = self.setup_runner()
        forward = engine.model_runner.model_engine.forward

        def abort(**kwargs):
            req.abort()
            return forward(**kwargs)

        engine.model_runner.model_engine.forward = abort
        self.assertEqual(engine.step(), (True, []))
        self.assertEqual(calls, [True])
        self.assertEqual(list(req.generated_token_ids), [])
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.cache_manager.blocks)
        )


if __name__ == "__main__":
    unittest.main()
