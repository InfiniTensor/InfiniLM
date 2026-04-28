from infinicore.tensor import Tensor


def squeeze(input: Tensor, dim: int) -> Tensor:
    return Tensor(input._underlying.squeeze(dim))
