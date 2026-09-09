import builtins

from infinicore.lib import _infinicore

_TYPE_ALIASES = {
    "cpu": (_infinicore.Device.Type.CPU, "cpu"),
    "cuda": (_infinicore.Device.Type.NVIDIA, "cuda"),
    "nvidia": (_infinicore.Device.Type.NVIDIA, "cuda"),
    "cambricon": (_infinicore.Device.Type.CAMBRICON, "mlu"),
    "mlu": (_infinicore.Device.Type.CAMBRICON, "mlu"),
    "ascend": (_infinicore.Device.Type.ASCEND, "npu"),
    "npu": (_infinicore.Device.Type.ASCEND, "npu"),
    "metax": (_infinicore.Device.Type.METAX, "metax"),
    "moore": (_infinicore.Device.Type.MOORE, "musa"),
    "musa": (_infinicore.Device.Type.MOORE, "musa"),
    "iluvatar": (_infinicore.Device.Type.ILUVATAR, "iluvatar"),
    "hygon": (_infinicore.Device.Type.HYGON, "hygon"),
}

_NATIVE_TO_NAME = {native: name for native, name in set(_TYPE_ALIASES.values())}


class device:
    def __init__(self, type="cpu", index=None):
        if isinstance(type, device):
            self.type = type.type
            self.index = type.index
            self._underlying = type._underlying
            return

        if not isinstance(type, str):
            raise TypeError("device type must be a string or infinicore.device")
        if ":" in type:
            if index is not None:
                raise ValueError("index must not be provided twice")
            type, raw_index = type.rsplit(":", 1)
            index = builtins.int(raw_index)

        try:
            native_type, canonical_name = _TYPE_ALIASES[type.lower()]
        except KeyError as error:
            raise ValueError(f"unsupported device type: {type}") from error

        self.type = canonical_name
        self.index = 0 if index is None else builtins.int(index)
        self._underlying = _infinicore.Device(native_type, self.index)

    @classmethod
    def _from_underlying(cls, underlying):
        instance = cls.__new__(cls)
        instance.type = _NATIVE_TO_NAME[underlying.type]
        instance.index = underlying.index
        instance._underlying = underlying
        return instance

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self):
        return f"{self.type}:{self.index}"

    def __eq__(self, other):
        return (
            isinstance(other, device)
            and self._underlying.type == other._underlying.type
            and self.index == other.index
        )
