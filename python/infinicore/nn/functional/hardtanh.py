import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hardtanh(
    input: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
    *,
    out=None,
) -> Tensor:
    """Clamp the input tensor to the range [min_val, max_val]."""

    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
        and hasattr(infinicore.ntops.torch, "hardtanh")
    ):
        try:
            return infinicore.ntops.torch.hardtanh(
                input, min_val=min_val, max_val=max_val, inplace=inplace
            )
        except AttributeError:
            pass

    if inplace:
        _infinicore.hardtanh_(
            input._underlying, input._underlying, float(min_val), float(max_val)
        )
        return input

    if out is None:
        return Tensor(
            _infinicore.hardtanh(input._underlying, float(min_val), float(max_val))
        )

    _infinicore.hardtanh_(
        out._underlying, input._underlying, float(min_val), float(max_val)
    )
    return out
