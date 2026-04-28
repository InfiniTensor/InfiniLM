from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def addr(
    input: Tensor,
    vec1: Tensor,
    vec2: Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    out=None,
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.addr(
                input._underlying, vec1._underlying, vec2._underlying, beta, alpha
            )
        )

    _infinicore.addr_(
        out._underlying,
        input._underlying,
        vec1._underlying,
        vec2._underlying,
        beta,
        alpha,
    )
    return out
