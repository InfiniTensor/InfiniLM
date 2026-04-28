from infinicore.tensor import Tensor


def narrow(input: Tensor, dim: int, start: int, length: int) -> Tensor:
    return Tensor(input._underlying.narrow(dim, start, length))
