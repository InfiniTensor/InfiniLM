from __future__ import annotations

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MAP: dict[str, int] = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    reduction_enum = _REDUCTION_MAP.get(reduction)
    if reduction_enum is None:
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    return Tensor(
        _infinicore.hinge_embedding_loss(
            input._underlying,
            target._underlying,
            float(margin),
            reduction_enum,
        )
    )
