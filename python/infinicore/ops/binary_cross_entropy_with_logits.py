from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def binary_cross_entropy_with_logits(
    input, target, weight=None, pos_weight=None, reduction="mean", *, out=None
):
    """
    input: Tensor (logits)
    target: Tensor (labels)
    weight: Tensor (optional, sample-wise weight)
    pos_weight: Tensor (optional, class-wise weight)
    reduction: str ('none', 'mean', 'sum')
    """

    # 提取底层 C++ 对象，处理可选 Tensor
    weight_raw = weight._underlying if weight is not None else None
    pos_weight_raw = pos_weight._underlying if pos_weight is not None else None

    if out is None:
        # 调用非原地接口，返回新创建的 Tensor
        return Tensor(
            _infinicore.binary_cross_entropy_with_logits(
                input._underlying,
                target._underlying,
                weight_raw,
                pos_weight_raw,
                str(reduction),
            )
        )

    # 调用显式指定输出的接口 (binary_cross_entropy_with_logits_)
    _infinicore.binary_cross_entropy_with_logits_(
        out._underlying,
        input._underlying,
        target._underlying,
        weight_raw,
        pos_weight_raw,
        str(reduction),
    )

    return out
