"""Unit tests for immediate KV block reclaim on request completion."""

from __future__ import annotations

import unittest

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


def _req(request_id: str, prompt_len: int) -> InferenceRequest:
    req = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=16),
    )
    return req


class CacheReclaimTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler(
            max_batch_size=4,
            num_blocks=64,
            block_size=256,
            enable_prefix_cache=False,
        )

    def test_cancel_reclaims_blocks_immediately(self):
        req = _req("cancel-me", 512)
        tokens = req.get_input_tokens()
        req.block_table, req.slot_mapping, req.num_cached_tokens = (
            self.scheduler.cache_manager.allocate_blocks(tokens, req.block_table)
        )
        self.scheduler.cache_manager.reset_req_blocks()
        req.num_blocks = len(req.block_table)
        used_before = len(self.scheduler.cache_manager.used_block_ids)
        free_before = self.scheduler.cache_manager.get_num_free_blocks()
        self.assertGreater(used_before, 0)

        req.status = RequestStatus.CANCELED
        self.scheduler.complete_requests([req])

        stats = self.scheduler.get_cache_stats()
        self.assertEqual(stats["num_used_blocks"], 0)
        self.assertEqual(stats["num_free_blocks"], free_before + used_before)
        self.assertEqual(
            self.scheduler.cache_manager.get_num_free_blocks(),
            self.scheduler.cache_manager.num_blocks,
        )


if __name__ == "__main__":
    unittest.main()
