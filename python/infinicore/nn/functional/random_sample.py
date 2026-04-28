from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def random_sample(
    logits: Tensor,
    random_val: float,
    topp: float,
    topk: int,
    temperature: float,
    *,
    out=None,
) -> Tensor:
    r"""Sample an index from logits with nucleus/top-k filtering."""

    if out is None:
        return Tensor(
            _infinicore.random_sample(
                logits._underlying,
                random_val,
                topp,
                topk,
                temperature,
            )
        )

    _infinicore.random_sample_(
        out._underlying,
        logits._underlying,
        random_val,
        topp,
        topk,
        temperature,
    )

    return out
