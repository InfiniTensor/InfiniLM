import pytest
from infinilm.config.engine_config import EngineConfig
from infinilm.llm.llm import validate_request_capacity


def paged_config(num_blocks: int, block_size: int) -> EngineConfig:
    return EngineConfig(
        model_path="/tmp/model",
        cache_type="paged",
        num_blocks=num_blocks,
        block_size=block_size,
    )


def test_pilotdeck_prompt_fits_configured_kv_capacity():
    validate_request_capacity(
        paged_config(num_blocks=256, block_size=256),
        prompt_tokens=8_348,
        output_tokens=512,
    )


def test_prompt_larger_than_kv_capacity_is_rejected():
    with pytest.raises(
        ValueError, match="maximum context length is 8192.*prompt contains 8348"
    ):
        validate_request_capacity(
            paged_config(num_blocks=128, block_size=64),
            prompt_tokens=8_348,
            output_tokens=512,
        )


def test_output_reservation_larger_than_remaining_capacity_is_rejected():
    with pytest.raises(ValueError, match="max_tokens must be at most 192"):
        validate_request_capacity(
            paged_config(num_blocks=128, block_size=64),
            prompt_tokens=8_000,
            output_tokens=512,
        )
