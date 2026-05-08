# Copyright (c) 2025, InfiniCore
#
# This file contains modified code derived from PyTorch's `torch.nn.Parameter`
# implementation, which is licensed under the BSD 3-Clause License.
#
# The modifications include adaptations for the InfiniCore framework.
#
# Original PyTorch source:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py
#
# Referencing PyTorch v2.4.0
#
# The use of this file is governed by the BSD 3-Clause License.


from ..tensor import Tensor


class InfiniCoreParameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter."""

    def __init__(self, data=None):
        if not isinstance(data, Tensor):
            raise ValueError("The `data` variable must be of type `infinicore.Tensor`.")
        super().__init__(data._underlying)

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    def __deepcopy__(self, memo):
        raise ValueError("not supported!")

    def __reduce_ex__(self, proto):
        raise ValueError("not supported!")
