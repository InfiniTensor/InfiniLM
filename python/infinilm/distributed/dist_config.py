class DistConfig:
    """
    Distributed Model Configuration.
    """

    def __init__(
        self,
        tp_size=None,
        tp_device_ids=None,
        allreduce_backend="nccl",
        moe_ep_backend="disabled",
        moe_ep_size=1,
    ):
        from infinilm.lib import _infinilm

        if tp_size is not None and tp_device_ids is not None:
            raise ValueError("Provide either tp_size OR tp_device_ids, not both")

        backend = self._parse_allreduce_backend(_infinilm, allreduce_backend)
        if tp_size is not None:
            self._underlying = _infinilm.DistConfig(tp_size, backend)
        elif tp_device_ids is not None:
            self._underlying = _infinilm.DistConfig(tp_device_ids, backend)
        else:
            self._underlying = _infinilm.DistConfig()
            self._underlying.allreduce_backend = backend
        self.moe_ep_backend = moe_ep_backend
        self.moe_ep_size = moe_ep_size

    @staticmethod
    def _parse_allreduce_backend(_infinilm, value):
        if hasattr(_infinilm, "AllReduceBackend") and isinstance(value, _infinilm.AllReduceBackend):
            return value
        normalized = str(value).lower().replace("-", "_")
        if normalized == "auto":
            return _infinilm.AllReduceBackend.AUTO
        if normalized == "nccl":
            return _infinilm.AllReduceBackend.NCCL
        if normalized == "custom":
            return _infinilm.AllReduceBackend.CUSTOM
        raise ValueError(
            f"Unsupported allreduce_backend={value!r}; expected one of auto/nccl/custom"
        )

    @property
    def tp_device_ids(self):
        return self._underlying.tp_device_ids

    @tp_device_ids.setter
    def tp_device_ids(self, value):
        self._underlying.tp_device_ids = list(value)

    @property
    def moe_ep_backend(self):
        return self._underlying.moe_ep_backend

    @moe_ep_backend.setter
    def moe_ep_backend(self, value):
        self._underlying.moe_ep_backend = str(value)

    @property
    def moe_ep_size(self):
        return self._underlying.moe_ep_size

    @moe_ep_size.setter
    def moe_ep_size(self, value):
        self._underlying.moe_ep_size = int(value)

    @property
    def allreduce_backend(self):
        return self._underlying.allreduce_backend

    @allreduce_backend.setter
    def allreduce_backend(self, value):
        from infinilm.lib import _infinilm

        self._underlying.allreduce_backend = self._parse_allreduce_backend(_infinilm, value)

    def __repr__(self):
        return repr(self._underlying)

    def __str__(self):
        return str(self._underlying)
