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
        pp_stage=0,
        master_addr="127.0.0.1",
        master_port=29500,
    ):
        from infinilm.lib import _infinilm

        if tp_size is not None and tp_device_ids is not None:
            raise ValueError("Provide either tp_size OR tp_device_ids, not both")

        if tp_size is not None:
            self._underlying = _infinilm.DistConfig(tp_size)
        elif tp_device_ids is not None:
            self._underlying = _infinilm.DistConfig(tp_device_ids)
        else:
            self._underlying = _infinilm.DistConfig()
        self.moe_ep_backend = moe_ep_backend
        self.moe_ep_size = moe_ep_size
        self.pp_size = pp_size
        self.pp_stage = pp_stage
        self.master_addr = master_addr
        self.master_port = master_port

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
    def pp_size(self):
        return self._underlying.pp_size

    @pp_size.setter
    def pp_size(self, value):
        self._underlying.pp_size = int(value)

    @property
    def pp_stage(self):
        return self._underlying.pp_stage

    @pp_stage.setter
    def pp_stage(self, value):
        self._underlying.pp_stage = int(value)

    @property
    def master_addr(self):
        return self._underlying.master_addr

    @master_addr.setter
    def master_addr(self, value):
        self._underlying.master_addr = str(value)

    @property
    def master_port(self):
        return self._underlying.master_port

    @master_port.setter
    def master_port(self, value):
        self._underlying.master_port = int(value)

    def __repr__(self):
        return repr(self._underlying)

    def __str__(self):
        return str(self._underlying)
