from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: str = "mean",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy loss with logits.

    This wraps the underlying C++/CUDA implementation exposed via `_infinicore`.

    The low-level binding treats missing ``weight`` / ``pos_weight`` via
    default-constructed tensors. Here we avoid passing ``None`` down and
    instead omit arguments when they are not provided, so pybind11 uses
    its defaults.
    """

    # Out-of-place API
    if out is None:
        # Neither weight nor pos_weight
        if weight is None and pos_weight is None:
            return Tensor(
                _infinicore.binary_cross_entropy_with_logits(
                    input._underlying,
                    target._underlying,
                    reduction=reduction,
                )
            )

        # weight provided only
        if weight is not None and pos_weight is None:
            return Tensor(
                _infinicore.binary_cross_entropy_with_logits(
                    input._underlying,
                    target._underlying,
                    weight._underlying,
                    reduction=reduction,
                )
            )

        # pos_weight provided only
        if weight is None and pos_weight is not None:
            return Tensor(
                _infinicore.binary_cross_entropy_with_logits(
                    input._underlying,
                    target._underlying,
                    pos_weight=pos_weight._underlying,
                    reduction=reduction,
                )
            )

        # both provided
        return Tensor(
            _infinicore.binary_cross_entropy_with_logits(
                input._underlying,
                target._underlying,
                weight._underlying,
                pos_weight._underlying,
                reduction,
            )
        )

    # In-place-style API with explicit out
    if weight is None and pos_weight is None:
        _infinicore.binary_cross_entropy_with_logits_(
            out._underlying,
            input._underlying,
            target._underlying,
            reduction=reduction,
        )
    elif weight is not None and pos_weight is None:
        _infinicore.binary_cross_entropy_with_logits_(
            out._underlying,
            input._underlying,
            target._underlying,
            weight._underlying,
            reduction=reduction,
        )
    elif weight is None and pos_weight is not None:
        _infinicore.binary_cross_entropy_with_logits_(
            out._underlying,
            input._underlying,
            target._underlying,
            pos_weight=pos_weight._underlying,
            reduction=reduction,
        )
    else:
        _infinicore.binary_cross_entropy_with_logits_(
            out._underlying,
            input._underlying,
            target._underlying,
            weight._underlying,
            pos_weight._underlying,
            reduction,
        )

    return out
