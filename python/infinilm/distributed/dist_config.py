class DistConfig:
    """
    Distributed Model Configuration.
    """

    def __init__(
        self,
        tp_size=None,
        tp_device_ids=None,
        moe_ep_backend="disabled",
        moe_ep_size=1,
        pp_size=1,
    ):
        from infinilm.lib import _infinilm

        if tp_size is not None and tp_device_ids is not None:
            raise ValueError("Provide either tp_size OR tp_device_ids, not both")
        if pp_size is None or pp_size < 1:
            raise ValueError("pp_size must be >= 1")

        if tp_size is not None:
            self._underlying = _infinilm.DistConfig(tp_size, pp_size)
        elif tp_device_ids is not None:
            self._underlying = _infinilm.DistConfig(tp_device_ids)
            self.pp_device_ids = range(pp_size)
        else:
            self._underlying = _infinilm.DistConfig(1, pp_size)
        self.moe_ep_backend = moe_ep_backend
        self.moe_ep_size = moe_ep_size

    @property
    def tp_device_ids(self):
        return self._underlying.tp_device_ids

    @tp_device_ids.setter
    def tp_device_ids(self, value):
        self._underlying.tp_device_ids = list(value)

    @property
    def pp_device_ids(self):
        return self._underlying.pp_device_ids

    @pp_device_ids.setter
    def pp_device_ids(self, value):
        device_ids = [int(device_id) for device_id in value]
        if not device_ids:
            raise ValueError("pp_device_ids must not be empty")
        if any(device_id < 0 for device_id in device_ids):
            raise ValueError("pp_device_ids must be non-negative")
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("pp_device_ids must be unique")
        self._underlying.pp_device_ids = device_ids

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

    def __repr__(self):
        return repr(self._underlying)

    def __str__(self):
        return str(self._underlying)
