"""Unit tests for aborted-request handling in v1 row scheduler update path."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from infinilm.llm.llm import LLMEngine
from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import ScheduledRow, SchedulerOutput


def _decode_req(rid: str) -> InferenceRequest:
    req = InferenceRequest(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=32),
    )
    req.is_prefill = False
    req.status = RequestStatus.RUNNING
    req.generated_token_ids = [99]
    return req


class UpdateRequestsAbortedTest(unittest.TestCase):
    def setUp(self):
        self.engine = LLMEngine.__new__(LLMEngine)
        self.engine.cache_type = "paged"
        self.engine.tokenizer = MagicMock()
        self.engine.tokenizer.decode.return_value = "x"
        self.engine.eos_token_ids = [-1]
        self.engine.scheduler = MagicMock()
        self.engine.scheduler.cache_manager = MagicMock()
        self.engine._check_request_finished = MagicMock(return_value=False)

    def test_aborted_decode_row_discards_sampled_token(self):
        """GPU forward may still produce a token after client abort; discard it."""
        req = _decode_req("abort-test")
        req.abort()
        rows = [ScheduledRow(req, 1, False, True)]
        scheduler_output = SchedulerOutput(
            scheduled_requests=[req],
            is_prefill=False,
            rows=rows,
        )

        pending = self.engine._update_requests_from_rows(
            scheduler_output, sampled_tokens=[42]
        )

        self.assertEqual(pending, [])
        self.assertEqual(req.generated_token_ids, [99])
        self.engine.scheduler.complete_requests.assert_called_once_with([req])

    def test_aborted_decode_row_mismatch_still_raises(self):
        """Real batching bugs (token count != rows needing sample) stay fatal."""
        req = _decode_req("abort-test")
        req.abort()
        rows = [ScheduledRow(req, 1, False, True)]
        scheduler_output = SchedulerOutput(
            scheduled_requests=[req],
            is_prefill=False,
            rows=rows,
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.engine._update_requests_from_rows(
                scheduler_output, sampled_tokens=[]
            )
        self.assertIn("sampled token count mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
