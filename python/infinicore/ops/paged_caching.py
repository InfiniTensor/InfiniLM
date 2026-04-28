from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def paged_caching(
    k_cache: Tensor,
    v_cache: Tensor,
    k: Tensor,
    v: Tensor,
    slot_mapping: Tensor,
):
    Tensor(
        _infinicore.paged_caching_(
            k_cache._underlying,
            v_cache._underlying,
            k._underlying,
            v._underlying,
            slot_mapping._underlying,
        )
    )
    return (k_cache, v_cache)
