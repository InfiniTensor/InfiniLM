"""Exercise progress, resolved-token preservation and high-confidence bypass."""

import importlib.util
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "clover_frontier",
    Path(__file__).parents[2] / "python/infinilm/diffusion/frontier.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_confident_tokens_are_not_delayed_by_stride():
    policy = module.FrontierPolicy()
    confidence = torch.ones(2, 32)
    masked = torch.ones(2, 32, dtype=torch.bool)
    masked[1, :7] = False
    assert torch.equal(policy.select(confidence, masked, 0), masked)


def test_stalled_rows_progress_without_touching_finished_rows():
    policy = module.FrontierPolicy()
    confidence = torch.full((3, 32), 0.1)
    masked = torch.zeros(3, 32, dtype=torch.bool)
    masked[0, 5:] = True
    masked[1, 17] = True
    selected = policy.select(confidence, masked, 3)
    assert selected.sum(-1).tolist() == [1, 1, 0]
    assert not (selected & ~masked).any()
    assert selected[0, 5] and selected[1, 17]
