from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_TO_INT = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    reduction_i = _REDUCTION_TO_INT.get(reduction)
    if reduction_i is None:
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    return Tensor(
        _infinicore.gaussian_nll_loss(
            input._underlying,
            target._underlying,
            var._underlying,
            bool(full),
            float(eps),
            int(reduction_i),
        )
    )
