import infinicore.tensor as tensor
from infinicore.lib import _infinicore


def add_rms_norm(a, b, weight, epsilon=1e-5, *, out=None, residual=None):
    """
    Fused Add and RMS Normalization.

    Args:
        a: First input tensor
        b: Second input tensor
        weight: Scale weights
        epsilon: Small constant for numerical stability, default is 1e-5
        out: Optional output tuple (y, residual_out) for in-place operation

    Returns:
        Tuple of (normalized_result, add_result): (RMSNorm(a + b) * weight, a + b)
        The add_result can be used as residual for subsequent layers.
    """
    if out is None:
        out = tensor.empty(a.shape, dtype=a.dtype, device=a.device)
    if residual is None:
        residual = tensor.empty(b.shape, dtype=b.dtype, device=b.device)

    _infinicore.add_rms_norm_(
        out._underlying,
        residual._underlying,
        a._underlying,
        b._underlying,
        weight._underlying,
        epsilon,
    )

    return out, residual
