import unittest

from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


class RWKV5SchedulerTest(unittest.TestCase):
    def test_cacheless_state_model_does_not_reserve_attention_blocks(self):
        scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=128,
            block_size=16,
            max_num_batched_tokens=4096,
            has_mamba_cache=True,
            cacheless_state_model=True,
            num_mamba_cache_blocks=8,
            enable_prefix_caching=False,
        )
        requests = [
            InferenceRequest(
                request_id=f"rwkv5-unit-{index}",
                prompt_token_ids=[1] * 512,
                sampling_params=SamplingParams(max_tokens=128, ignore_eos=True),
            )
            for index in range(4)
        ]
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()

        self.assertIsNotNone(output)
        self.assertEqual(len(output.scheduled_requests), 4)
        self.assertTrue(all(not request.block_table for request in requests))
        self.assertTrue(all(not request.slot_mapping for request in requests))
        self.assertTrue(all(request.mamba_cache_index is not None for request in requests))

    def test_default_scheduler_still_reserves_attention_blocks(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=128,
            block_size=16,
            max_num_batched_tokens=1024,
            enable_prefix_caching=False,
        )
        request = InferenceRequest(
            request_id="transformer-unit",
            prompt_token_ids=[1] * 32,
            sampling_params=SamplingParams(max_tokens=32, ignore_eos=True),
        )
        scheduler.add_request(request)

        output = scheduler.schedule()

        self.assertIsNotNone(output)
        self.assertEqual(output.scheduled_requests, [request])
        self.assertTrue(request.block_table)
        self.assertEqual(len(request.slot_mapping), 32)

    def test_mamba_state_cache_does_not_change_attention_kv_path(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=128,
            block_size=16,
            max_num_batched_tokens=1024,
            has_mamba_cache=True,
            cacheless_state_model=False,
            num_mamba_cache_blocks=8,
            enable_prefix_caching=False,
        )
        request = InferenceRequest(
            request_id="hybrid-unit",
            prompt_token_ids=[1] * 32,
            sampling_params=SamplingParams(max_tokens=32, ignore_eos=True),
        )
        scheduler.add_request(request)

        output = scheduler.schedule()

        self.assertIsNotNone(output)
        self.assertTrue(request.block_table)
        self.assertEqual(request.mamba_cache_index, 1)


if __name__ == "__main__":
    unittest.main()
