import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from ..parameter import InfiniCoreParameter as Parameter
from .module import InfiniCoreModule as Module


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings.
        embedding_dim (int): the size of each embedding vector.

    Attributes:
        weight (Tensor): the weights of the module of shape (num_embeddings, embedding_dim).

    Shape:
        - Input: :(*), IntTensor or LongTensor of arbitrary shape containing the indices to extract.
        - Output: (*, H), where `*` is the input shape and H=embedding_dim.
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
    ]

    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {
            "dtype": infinicore.float32 if dtype is None else dtype,
            "device": infinicore.device("cpu", 0) if device is None else device,
        }
        super().__init__()
        assert (
            (padding_idx is None)
            and (max_norm is None)
            and (scale_grad_by_freq is False)
            and (sparse is False)
            and (_weight is None)
            and (_freeze is False)
        ), "Unsupported parameters."

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            infinicore.empty([num_embeddings, embedding_dim], **factory_kwargs)
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight)

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        return s.format(**self.__dict__)
