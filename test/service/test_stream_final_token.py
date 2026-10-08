"""Regression tests for text emitted by a stream's finishing token."""

from types import SimpleNamespace

import pytest
from infinilm.llm.llm import LLMEngine
from infinilm.llm.request import FinishReason, InferenceRequest
from infinilm.llm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    ("decoded", "sampling_params", "eos_token_ids", "expected_text", "reason"),
    [
        ("hello", SamplingParams(max_tokens=1), [], "hello", FinishReason.LENGTH),
        (
            "hello<END>",
            SamplingParams(max_tokens=2, stop=["<END>"]),
            [],
            "hello",
            FinishReason.STOP_STRING,
        ),
        (
            "<eos>",
            SamplingParams(max_tokens=2),
            [7],
            "",
            FinishReason.EOS_TOKEN,
        ),
    ],
)
def test_stream_emits_visible_text_from_finishing_token(
    decoded, sampling_params, eos_token_ids, expected_text, reason
):
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(
        commit_computed_tokens=lambda *_: None,
        complete_requests=lambda *_: None,
    )
    engine.tokenizer = SimpleNamespace(decode=lambda _: decoded)
    engine.eos_token_ids = eos_token_ids
    req = InferenceRequest(
        request_id="test",
        prompt_token_ids=[1],
        sampling_params=sampling_params,
    )
    queue = req.output_queue
    try:
        pending = engine._update_requests([req], [7])
        assert len(pending) == 1
        assert pending[0][0] is queue.async_q
        output = pending[0][1]
        assert output.finished
        assert output.finish_reason == reason
        assert output.token_text == expected_text
    finally:
        queue.close()


@pytest.mark.parametrize(
    ("final_text", "eos_token_ids", "expected_final_text", "reason"),
    [
        ("D>", [], "", FinishReason.STOP_STRING),
        ("", [8], "<EN", FinishReason.EOS_TOKEN),
    ],
)
def test_stream_holds_partial_stop_string_until_next_token(
    final_text, eos_token_ids, expected_final_text, reason
):
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(
        commit_computed_tokens=lambda *_: None,
        complete_requests=lambda *_: None,
    )
    engine.tokenizer = SimpleNamespace(
        decode=lambda ids: {7: "hello<EN", 8: final_text}[ids[0]]
    )
    engine.eos_token_ids = eos_token_ids
    req = InferenceRequest(
        request_id="test",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=3, stop=["<END>"]),
    )
    queue = req.output_queue
    try:
        first = engine._update_requests([req], [7])[0][1]
        assert first.token_text == "hello"
        assert not first.finished

        final = engine._update_requests([req], [8])[0][1]
        assert final.token_text == expected_final_text
        assert final.finish_reason == reason
        assert final.finished
    finally:
        queue.close()
