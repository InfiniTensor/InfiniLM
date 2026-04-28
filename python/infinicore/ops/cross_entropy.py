from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cross_entropy(
    logits,
    target,
    weight=None,
    *,
    ignore_index=None,
    reduction="none",
    out=None,
):
    """
    Token-wise cross entropy without reduction. The output tensor has the same
    shape as target and uses the logits dtype.
    """
    if weight is not None:
        raise NotImplementedError("class weights are not supported yet.")
    if ignore_index is not None:
        raise NotImplementedError("ignore_index is not supported yet.")
    if reduction not in (None, "none"):
        raise NotImplementedError("Only reduction='none' is implemented.")

    if out is None:
        return Tensor(_infinicore.cross_entropy(logits._underlying, target._underlying))

    _infinicore.cross_entropy_(
        out._underlying,
        logits._underlying,
        target._underlying,
    )
    return out
