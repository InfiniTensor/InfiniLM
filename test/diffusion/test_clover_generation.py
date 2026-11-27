"""Exercise the Clover generation loop using CPU logits for two-token blocks."""

from unittest.mock import Mock, call

import pytest
import torch
from infinilm.diffusion.clover import CloverGenerator


@pytest.mark.parametrize(
    ("eos", "tokens", "iterations"),
    [([], [1, 1, 1], [1, 1]), ([1], [], [1])],
)
def test_block_handoff_replays_only_when_another_block_follows(eos, tokens, iterations):
    generator = CloverGenerator.__new__(CloverGenerator)
    generator.engine = Mock(eos_token_id=eos)
    generator.tokenizer = Mock()
    generator.tokenizer.apply_chat_template.return_value = [9]
    generator.cache_config = None
    generator.torch = torch
    generator.torch_device = "cpu"
    generator.block_size = 2
    generator.mask_id = 2
    generator.max_context = 5
    generator._logits = lambda value: value

    # Every preview is confident about token 1, so each block resolves in one
    # step: a block is replayed only to prepare the next one, and the last
    # block - or a block that ends in EOS - is not replayed.
    confident = torch.tensor([[0.0, 10.0, 0.0, -torch.inf]] * 2)
    generator._forward = Mock(
        side_effect=[
            None,
            {"logits": confident},
            None,
            {"logits": confident},
        ]
    )
    result = generator.generate("prompt", max_new_tokens=3, threshold=0.5)
    expected = [
        call([9], 0),
        call([2, 2], 1, preview=True, expert_budget=False),
    ]
    if not eos:
        expected.extend(
            [
                call([1, 1], 1),
                call([2, 2], 3, preview=True, expert_budget=False),
            ]
        )
    assert generator._forward.call_args_list == expected
    assert result.iterations == iterations
    assert result.token_ids == tokens
    assert result.prompt_token_count == 1
