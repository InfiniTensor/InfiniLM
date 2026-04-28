import numpy as np

import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from ..functional import RopeAlgo
from .module import InfiniCoreModule as Module


def create_sin_cos_table_numpy(max_position, head_dim, theta=10000.0):
    assert head_dim % 2 == 0, "Embedding dimension must be even."
    pos = np.arange(0, max_position)
    freqs = 1.0 / (
        theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(float) / head_dim)
    )
    angles = np.outer(pos, freqs)
    sin_table = np.sin(angles, dtype=np.float32)
    cos_table = np.cos(angles, dtype=np.float32)
    return sin_table, cos_table


def create_sin_cos_table(
    max_position,
    head_dim,
    theta=10000.0,
    device=None,
    dtype=None,
):
    sin_table_np, cos_table_np = create_sin_cos_table_numpy(
        max_position, head_dim, theta
    )

    sin_table_infini = infinicore.from_numpy(sin_table_np, dtype=dtype, device=device)
    cos_table_infini = infinicore.from_numpy(cos_table_np, dtype=dtype, device=device)

    return sin_table_infini, cos_table_infini


class RoPE(Module):
    r"""Rotary Position Embedding(RoPE)..

    Args:
        max_position_embeddings (int): The maximum sequence length that this model might ever be used with.
        rope_theta (float): The base period of the RoPE embeddings.
        head_dim (int): The attention head dimension.

    Shape:
        - Input:  hidden_states, ( bs, seq_len, num_heads, head_dim).
        - Output: hidden_states, ( bs, seq_len, num_heads, head_dim).
    """

    __constants__ = ["max_position_embeddings", "rope_theta", "head_dim"]
    max_position_embeddings: int
    rope_theta: float
    head_dim: int

    def __init__(
        self,
        max_position_embeddings: int,
        rope_theta: float,
        head_dim: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }
        super().__init__()

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.head_dim = head_dim

        self._sin_table, self._cos_table = create_sin_cos_table(
            self.max_position_embeddings,
            head_dim=self.head_dim,
            theta=self.rope_theta,
            **factory_kwargs,
        )

    def forward(self, states: Tensor, position_ids: Tensor, algo=RopeAlgo.GPT_NEOX):
        F.rope(
            states,
            position_ids,
            self._sin_table,
            self._cos_table,
            algo=algo,
            out=states,
        )
        return states
