from infinicore.tensor import Tensor


def unsqueeze(input: Tensor, dim: int) -> Tensor:
    return Tensor(input._underlying.unsqueeze(dim))
