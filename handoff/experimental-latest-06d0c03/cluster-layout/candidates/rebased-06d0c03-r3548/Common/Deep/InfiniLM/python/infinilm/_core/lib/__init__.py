# Compatibility facade: all native types are registered once by the LM module.
import sys

from _infinilm import _core as _infinicore

sys.modules[__name__ + "._infinicore"] = _infinicore

__all__ = ["_infinicore"]
