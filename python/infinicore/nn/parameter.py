from infinicore.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        if not isinstance(data, Tensor):
            raise TypeError("Parameter data must be an infinicore.Tensor")
        super().__init__(data._underlying, owner=data)
