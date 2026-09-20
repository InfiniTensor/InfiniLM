import unittest

from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


class Mamba2SchedulerTest(unittest.TestCase):
    def test_recurrent_requests_skip_attention_kv_reservation(self):
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
                request_id=f"mamba2-unit-{index}",
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
        self.assertEqual(
            {request.mamba_cache_index for request in requests}, {1, 2, 3, 4}
        )


if __name__ == "__main__":
    unittest.main()
