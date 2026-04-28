# ============================================
# Copyright (c) 2025, InfiniCore
#
# This file implements InfiniCoreModuleList, which is similar to torch.nn.ModuleList
# but based on InfiniCoreModule for inference purposes.

import operator
from collections import OrderedDict
from itertools import chain
from typing import Iterator, List, Optional, Sequence, TypeVar, Union

from .module import InfiniCoreModule as Module

# Define type variable for module compatibility (supports InfiniCoreModule)
ModuleType = TypeVar("ModuleType", bound=Union["Module"])


class InfiniCoreModuleList(Module):
    r"""Holds submodules in a list.

    InfiniCoreModuleList can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    InfiniCoreModule methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        >>> class MyModel(InfiniCoreModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linears = InfiniCoreModuleList([
        ...             torch.nn.Linear(10, 10) for i in range(10)
        ...         ])
        ...
        ...     def forward(self, x):
        ...         # ModuleList can act as an iterable, or be indexed using ints
        ...         for i, l in enumerate(self.linears):
        ...             x = self.linears[i // 2](x) + l(x)
        ...         return x
    """

    def __init__(self, modules: Optional[Sequence[ModuleType]] = None):
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[ModuleType, "InfiniCoreModuleList"]:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: ModuleType) -> None:
        idx = self._get_abs_string_index(idx)
        # Use add_module to register module
        self.add_module(idx, module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            indices_to_delete = list(range(len(self._modules)))[idx]
            for k in indices_to_delete:
                if str(k) in self._modules:
                    del self._modules[str(k)]
        else:
            idx_str = self._get_abs_string_index(idx)
            if idx_str in self._modules:
                del self._modules[idx_str]

        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        if len(self._modules) > 0:
            str_indices = [str(i) for i in range(len(self._modules))]
            self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[ModuleType]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Sequence[ModuleType]) -> "InfiniCoreModuleList":
        return self.extend(modules)

    def __add__(
        self, other: Union[Sequence[ModuleType], "InfiniCoreModuleList"]
    ) -> "InfiniCoreModuleList":
        r"""Return a new InfiniCoreModuleList by concatenating with another iterable.

        Args:
            other (iterable): iterable of modules to concatenate
        """
        if not isinstance(other, (list, tuple, InfiniCoreModuleList)):
            raise TypeError(
                f"InfiniCoreModuleList can only be concatenated with list, tuple, or InfiniCoreModuleList, "
                f"got {type(other).__name__}"
            )

        combined = InfiniCoreModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def append(self, module: ModuleType) -> "InfiniCoreModuleList":
        r"""Append a given module to the end of the list.

        Args:
            module (InfiniCoreModule): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Sequence[ModuleType]) -> "InfiniCoreModuleList":
        r"""Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, (list, tuple)):
            try:
                modules = list(modules)
            except TypeError:
                raise TypeError(
                    f"InfiniCoreModuleList.extend should be called with an "
                    f"iterable, but got {type(modules).__name__}"
                )

        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def insert(self, index: int, module: ModuleType) -> None:
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module ( InfiniCoreModule): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def pop(self, idx: int = -1) -> ModuleType:
        r"""Remove and return a module at the given index.

        Args:
            idx (int): index of the module to pop. Default: -1 (last module)

        Returns:
            Module: the module that was removed
        """
        idx_str = self._get_abs_string_index(idx)
        module = self._modules[idx_str]
        # Use __delitem__ to ensure proper cleanup
        self.__delitem__(int(idx_str))
        return module

    def __repr__(self) -> str:
        """Return a string representation of the ModuleList."""
        if len(self) == 0:
            return self.__class__.__name__ + "()"

        lines = []
        for i, module in enumerate(self):
            lines.append(f"({i}): {repr(module)}")

        main_str = self.__class__.__name__ + "(\n  "
        main_str += "\n  ".join(lines) + "\n)"
        return main_str

    def __dir__(self) -> List[str]:
        """Return a list of attribute names, excluding numeric keys."""
        keys = super().__dir__()
        # Filter out numeric keys to avoid cluttering dir() output
        keys = [key for key in keys if not key.isdigit()]
        return keys
