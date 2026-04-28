from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def vander(
    x: Tensor,
    N: Optional[int] = None,
    increasing: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Generates a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector. The
    order of the powers is determined by the ``increasing`` boolean argument.

    Args:
        x (Tensor): 1-D input tensor.
        N (int, optional): Number of columns in the output. If None, defaults to the size of x.
        increasing (bool, optional): Order of the powers.
            If False (default), the powers are descending (x^(N-1), ..., x^0).
            If True, the powers are ascending (x^0, ..., x^(N-1)).
        out (Tensor, optional): The output tensor.
    """

    if not x.is_contiguous():
        x = x.contiguous()
    N_val = N if N is not None else 0

    if out is not None:
        _infinicore.vander_(out._underlying, x._underlying, N_val, increasing)
        return out

    return Tensor(_infinicore.vander(x._underlying, N_val, increasing))
