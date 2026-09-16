import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from chunk_test_support import MODULES
from test_chunk_config import LLM_MODULE, load_module


def load_processor():
    with patch.dict(sys.modules):
        for name, module in MODULES.items():
            sys.modules[f"infinilm.llm.{name}"] = module
        sys.modules["transformers"] = SimpleNamespace(AutoTokenizer=object)
        static = load_module("infinilm.llm.static_scheduler", "llm/static_scheduler.py")
        load_module("infinilm.processors.processor", "processors/processor.py")
        return (
            load_module(
                "infinilm.processors.basic_llm_processor",
                "processors/basic_llm_processor.py",
            ).BasicLLMProcessor,
            static,
        )


Processor, STATIC_MODULE = load_processor()


class ChunkExecutionTests(unittest.TestCase):
    def test_legacy_static_output_does_not_require_chunk_metadata(self):
        engine, req = self.setup_engine(length=1)
        engine.scheduler = STATIC_MODULE.StaticScheduler()
        self.addCleanup(engine.scheduler.waiting_queue.close)
        engine.scheduler.add_request(req)
        engine.step()
        self.assertEqual(list(req.generated_token_ids), [77])

    def test_static_processor_still_uses_its_prefix_metadata(self):
        _, req = self.setup_engine(length=17)
        output = STATIC_MODULE.StaticSchedulerOutput(
            [req], is_prefill=True, prefix_hit_len=16
        )
        processor = Processor.__new__(Processor)
        backend = SimpleNamespace(
            from_list=lambda values, **kwargs: values, int64="int64", int32="int32"
        )
        with patch.dict(sys.modules, {"infinicore": backend}):
            inputs = processor.build_model_inputs(output)
        self.assertEqual([list(row) for row in inputs["input_ids"]], [[26]])
        self.assertEqual(inputs["position_ids"], [[16]])
        self.assertEqual(inputs["past_kv_lengths"], [16])
        self.assertEqual(inputs["total_kv_lengths"], [17])
        self.assertIsNone(inputs["slot_mapping"])

    def test_unsupported_models_rejected_before_native_initialization(self):
        from test_chunk_config import EngineConfig

        for hf in ({"num_experts": 8}, {"vision_config": {}}, {"audio_config": {}}):
            with (
                self.subTest(hf=hf),
                patch.object(LLM_MODULE, "read_hf_config", return_value=hf),
            ):
                with self.assertRaisesRegex(ValueError, "dense text"):
                    LLM_MODULE.LLMEngine(EngineConfig("unused", prefill_chunk_size=16))
        with patch.object(LLM_MODULE, "model_uses_mamba_cache", return_value=True):
            with self.assertRaisesRegex(ValueError, "dense text"):
                LLM_MODULE.LLMEngine(EngineConfig("unused", prefill_chunk_size=16))

    def setup_engine(self, length=35, prefix=True):
        scheduler = MODULES["scheduler"].Scheduler(
            num_blocks=16,
            block_size=16,
            prefill_chunk_size=16,
            enable_prefix_caching=prefix,
        )
        self.addCleanup(scheduler.waiting_queue.close)
        self.addCleanup(scheduler.running_queue.close)
        req = MODULES["request"].InferenceRequest(
            "long",
            prompt_token_ids=list(range(10, 10 + length)),
            sampling_params=MODULES["sampling_params"].SamplingParams(
                max_tokens=1, ignore_eos=True
            ),
        )
        scheduler.add_request(req)
        engine = LLM_MODULE.LLMEngine.__new__(LLM_MODULE.LLMEngine)
        engine.scheduler = scheduler
        engine.tokenizer = SimpleNamespace(decode=lambda tokens: "output")
        engine.model_runner = SimpleNamespace(
            execute_model=lambda output: SimpleNamespace(
                sampled_token_ids=[77] * len(output.scheduled_requests),
                kv_connector_output=None,
            )
        )
        return engine, req

    def test_intermediate_steps_publish_only_computed_pages_and_emit_no_tokens(self):
        engine, req = self.setup_engine()
        for end, pages in ((16, 1), (32, 2)):
            worked, pending = engine.step()
            self.assertTrue(worked)
            self.assertEqual(pending, [])
            self.assertEqual(req.get_num_generated_tokens(), 0)
            self.assertEqual(req.num_computed_tokens, end)
            self.assertEqual(req.num_cache_indexed_blocks, pages)
            self.assertEqual(
                len(engine.scheduler.cache_manager.hash_to_block_ids), pages
            )
        engine.step()
        self.assertEqual(list(req.generated_token_ids), [77])
        self.assertEqual(req.status, MODULES["request"].RequestStatus.FINISHED)
        self.assertEqual(engine.scheduler.cache_manager.get_total_usable_blocks(), 16)

    def test_abort_during_intermediate_forward_releases_ownership(self):
        engine, req = self.setup_engine()

        def execute(output):
            req._aborted = True
            return SimpleNamespace(sampled_token_ids=[77], kv_connector_output=None)

        engine.model_runner.execute_model = execute
        engine.step()
        self.assertEqual(req.status, MODULES["request"].RequestStatus.CANCELED)
        self.assertFalse(engine.scheduler.chunking_queue)
        self.assertEqual(engine.scheduler.running_queue.sync_q.qsize(), 0)
        self.assertEqual(req.get_num_generated_tokens(), 0)
        self.assertEqual(req.num_cache_indexed_blocks, 1)
        self.assertEqual(engine.scheduler.cache_manager.get_total_usable_blocks(), 16)

    def test_other_request_completion_never_publishes_future_long_pages(self):
        engine, req = self.setup_engine(length=64)
        engine.step()
        short = MODULES["request"].InferenceRequest(
            "short",
            prompt_token_ids=[8],
            sampling_params=MODULES["sampling_params"].SamplingParams(
                max_tokens=1, ignore_eos=True
            ),
        )
        engine.scheduler.add_request(short)
        engine.step()
        engine.step()
        self.assertEqual(short.get_num_generated_tokens(), 1)
        self.assertEqual(req.num_computed_tokens, 32)
        self.assertNotIn(
            req.block_hashes[2], engine.scheduler.cache_manager.hash_to_block_ids
        )
        self.assertNotIn(
            req.block_hashes[3], engine.scheduler.cache_manager.hash_to_block_ids
        )

    def test_disabled_prefix_cache_still_completes_in_three_steps(self):
        engine, req = self.setup_engine(prefix=False)
        for _ in range(3):
            engine.step()
        self.assertEqual(list(req.generated_token_ids), [77])
        self.assertFalse(engine.scheduler.cache_manager.hash_to_block_ids)

    def test_abort_during_final_chunk_emits_nothing_and_releases(self):
        engine, req = self.setup_engine(length=17)
        engine.step()

        def execute(output):
            req.abort()
            return SimpleNamespace(sampled_token_ids=[77], kv_connector_output=None)

        engine.model_runner.execute_model = execute
        self.assertEqual(engine.step(), (True, []))
        self.assertEqual(req.get_num_generated_tokens(), 0)
        self.assertEqual(req.status, MODULES["request"].RequestStatus.CANCELED)
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.cache_manager.blocks)
        )

    def test_intermediate_eos_is_ignored_and_final_eos_finishes(self):
        engine, req = self.setup_engine(length=17)
        req.sampling_params.ignore_eos = False
        req.sampling_params.max_tokens = 8
        engine.eos_token_ids = [77]
        engine.step()
        self.assertFalse(req.is_finished())
        self.assertEqual(req.get_num_generated_tokens(), 0)
        engine.step()
        self.assertTrue(req.is_finished())
        self.assertEqual(list(req.generated_token_ids), [77])
        self.assertEqual(req.finish_reason, MODULES["request"].FinishReason.EOS_TOKEN)

    def test_decode_extends_hashes_and_only_publishes_computed_tokens(self):
        engine, req = self.setup_engine(length=31)
        req.sampling_params.max_tokens = 3
        engine.step()
        engine.step()
        self.assertEqual(len(req.block_hashes), 2)
        self.assertEqual(req.num_cache_indexed_blocks, 1)
        engine.step()
        self.assertEqual(req.num_cache_indexed_blocks, 2)
        engine.step()
        self.assertTrue(req.is_finished())
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.cache_manager.blocks)
        )

    def test_processor_slices_partial_prefill_positions_and_lengths(self):
        engine, req = self.setup_engine(length=35)
        engine.step()
        step = engine.scheduler.schedule()
        processor = Processor.__new__(Processor)
        backend = SimpleNamespace(
            from_list=lambda values, **kwargs: values, int64="int64", int32="int32"
        )
        with patch.dict(sys.modules, {"infinicore": backend}):
            inputs = processor.build_model_inputs(step)
        self.assertEqual(inputs["input_ids"], [list(range(26, 42))])
        self.assertEqual(inputs["position_ids"], list(range(16, 32)))
        self.assertEqual(inputs["past_kv_lengths"], [16])
        self.assertEqual(inputs["total_kv_lengths"], [32])
        self.assertEqual(inputs["input_offsets"], [0, 16])
        self.assertEqual(len(inputs["slot_mapping"]), 16)
