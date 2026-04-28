from infinicore.lib import _infinicore

_infinicore_2_python_dict = {}
_python_2_infinicore_dict = {}


class device:
    # Public attributes describing the device
    type: str
    index: int
    _underlying: _infinicore.Device

    def __init__(self, type=None, index=None):
        if isinstance(type, device):
            self.type = type.type
            self.index = type.index
            return

        if type is None:
            type = "cpu"

        if ":" in type:
            if index is not None:
                raise ValueError(
                    '`index` should not be provided when `type` contains `":"`.'
                )

            type, index = type.split(":")
            index = int(index)

        self.type = type
        self.index = index if index else 0

    def __getattr__(self, name):
        # Lazily construct and cache an attribute.
        # such as, self._underlying .
        if name == "_underlying":
            setattr(self, name, device._to_infinicore_device(self.type, self.index))
        else:
            raise AttributeError("{!r} object has no attribute {!r}".format(self, name))
        return getattr(self, name)

    def __repr__(self):
        return f"device(type='{self.type}'{f', index={self.index}' if self.index is not None else ''})"

    def __str__(self):
        return f"{self.type}{f':{self.index}' if self.index is not None else ''}"

    def __eq__(self, other):
        """
        Compare two device objects for equality.

        Args:
            other: The object to compare with

        Returns:
            bool: True if both objects are device instances with the same type and index
        """
        if not isinstance(other, device):
            return False
        return self.type == other.type and self.index == other.index

    @staticmethod
    def _to_infinicore_device(type: str, index: int):
        return _python_2_infinicore_dict[type][index]

    @staticmethod
    def _from_infinicore_device(infinicore_device: _infinicore.Device):
        return _infinicore_2_python_dict[infinicore_device.type][
            infinicore_device.index
        ]


_TORCH_DEVICE_MAP = {
    _infinicore.Device.Type.CPU: "cpu",
    _infinicore.Device.Type.NVIDIA: "cuda",
    _infinicore.Device.Type.CAMBRICON: "mlu",
    _infinicore.Device.Type.ASCEND: "npu",
    _infinicore.Device.Type.METAX: "cuda",
    _infinicore.Device.Type.MOORE: "musa",
    _infinicore.Device.Type.ILUVATAR: "cuda",
    _infinicore.Device.Type.KUNLUN: "cuda",
    _infinicore.Device.Type.HYGON: "cuda",
    _infinicore.Device.Type.QY: "cuda",
    _infinicore.Device.Type.ALI: "cuda",
}


def _initialize_device_relationship(all_device_types, all_device_count):
    # python_device_type_set: {'cpu', 'musa', 'npu', 'cuda', 'mlu'}
    python_device_type_set = set([type for type in _TORCH_DEVICE_MAP.values()])
    python_device_current_index = {type: 0 for type in python_device_type_set}

    infinicore_2_python_dict = {}
    python_2_infinicore_dict = {}
    for infinicore_device_type, infinicore_device_count in zip(
        all_device_types, all_device_count
    ):
        if 0 == infinicore_device_count:
            continue

        # Found one device
        for infinicore_device_index in range(infinicore_device_count):
            # Create instantiation objects for C++ devices
            infinicore_instance = _infinicore.Device(
                infinicore_device_type, infinicore_device_index
            )

            # Create instantiation objects for python devices
            python_device_type = _TORCH_DEVICE_MAP[infinicore_device_type]

            python_device_index = python_device_current_index[python_device_type]
            python_device_current_index[python_device_type] += 1

            python_instance = device(python_device_type, python_device_index)

            # Cache corresponding relationship
            if infinicore_2_python_dict.get(infinicore_device_type) is not None:
                infinicore_2_python_dict[infinicore_device_type].append(python_instance)
            else:
                infinicore_2_python_dict[infinicore_device_type] = [python_instance]

            if python_2_infinicore_dict.get(python_device_type) is not None:
                python_2_infinicore_dict[python_device_type].append(infinicore_instance)
            else:
                python_2_infinicore_dict[python_device_type] = [infinicore_instance]

    return infinicore_2_python_dict, python_2_infinicore_dict


_all_device_types = tuple(_infinicore.Device.Type.__members__.values())[:-1]
_all_device_count = tuple(
    _infinicore.get_device_count(device) for device in _all_device_types
)

_infinicore_2_python_dict, _python_2_infinicore_dict = _initialize_device_relationship(
    _all_device_types, _all_device_count
)
