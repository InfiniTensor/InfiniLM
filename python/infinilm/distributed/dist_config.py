class DistConfig:
    """
    Distributed Model Configuration.
    """

    def __init__(self, tp_size=None, tp_device_ids=None):
        from infinilm.lib import _infinilm

        if tp_size is not None and tp_device_ids is not None:
            raise ValueError("Provide either tp_size OR tp_device_ids, not both")

        if tp_size is not None:
            self._underlying = _infinilm.DistConfig(tp_size)
        elif tp_device_ids is not None:
            self._underlying = _infinilm.DistConfig(tp_device_ids)
        else:
            self._underlying = _infinilm.DistConfig()

    @property
    def tp_device_ids(self):
        return self._underlying.tp_device_ids

    @tp_device_ids.setter
    def tp_device_ids(self, value):
        self._underlying.tp_device_ids = list(value)

    def __repr__(self):
        return repr(self._underlying)

    def __str__(self):
        return str(self._underlying)
