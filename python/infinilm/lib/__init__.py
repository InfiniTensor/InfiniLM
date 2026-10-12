# Compatibility facade for the one top-level private native module.
import sys

import _infinilm

sys.modules[__name__ + "._infinilm"] = _infinilm

__all__ = ["_infinilm"]
