from typing import Optional, Tuple

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def kthvalue(
    input: Tensor,
    k: int,
    dim: int = -1,
    keepdim: bool = False,
    *,
    out: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Returns a namedtuple (values, indices) where values is the k-th smallest
    element of each row of the input tensor in the given dimension.
    """

    if not input.is_contiguous():
        input = input.contiguous()

    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ValueError("out must be a tuple of two Tensors (values, indices)")

        out_values, out_indices = out

        _infinicore.kthvalue_(
            out_values._underlying,
            out_indices._underlying,
            input._underlying,
            k,
            dim,
            keepdim,
        )
        return out

    ret = _infinicore.kthvalue(input._underlying, k, dim, keepdim)

    return (Tensor(ret[0]), Tensor(ret[1]))
