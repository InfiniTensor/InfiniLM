"""Frontier-guided denoising adapted from Ringssss/dLLM-serve (Apache-2.0)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierPolicy:
    """Choose confident positions with a bounded, adaptive target stride."""

    target_iterations: int = 8
    min_unmask: int = 2
    max_unmask: int = 16
    confidence_floor: float = 0.9
    confident_threshold: float = 0.95

    def __post_init__(self):
        if not 1 <= self.min_unmask <= self.max_unmask <= 32:
            raise ValueError("Unmask bounds must satisfy 1 <= min <= max <= 32.")
        if self.target_iterations < 1 or not (
            0 <= self.confidence_floor <= self.confident_threshold <= 1
        ):
            raise ValueError("Invalid target iterations or confidence floor.")

    def select(self, confidence, masked, iteration):
        import torch

        remaining = masked.sum(-1)
        rounds = max(1, self.target_iterations - iteration)
        stride = torch.div(remaining + rounds - 1, rounds, rounding_mode="floor")
        stride = torch.minimum(
            stride.clamp(self.min_unmask, self.max_unmask), remaining
        )
        scores = confidence.masked_fill(~masked, -float("inf"))
        order = scores.argsort(dim=-1, descending=True, stable=True)
        ranks = torch.arange(masked.shape[-1], device=masked.device)
        take = ranks < stride[..., None]
        transfer = torch.zeros_like(masked).scatter(-1, order, take)
        transfer &= masked & (confidence >= self.confidence_floor)
        transfer |= masked & (confidence >= self.confident_threshold)
        stalled = masked.any(-1) & ~transfer.any(-1)
        best = scores.argmax(-1, keepdim=True)
        transfer.scatter_(-1, best, transfer.gather(-1, best) | stalled[..., None])
        return transfer
