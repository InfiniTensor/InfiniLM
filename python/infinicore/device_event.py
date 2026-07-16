from infinicore.device import device
from infinicore.lib import _infinicore


class DeviceEvent:
    def __init__(self, target=None, flags=None):
        args = []
        if target is not None:
            args.append(device(target)._underlying)
        if flags is not None:
            args.append(flags)
        self._underlying = _infinicore.DeviceEvent(*args)

    @property
    def device(self):
        return device._from_underlying(self._underlying.device)

    @property
    def is_recorded(self):
        return self._underlying.is_recorded

    def record(self, stream=None):
        if stream is None:
            self._underlying.record()
        else:
            self._underlying.record(stream)

    def synchronize(self):
        self._underlying.synchronize()

    def query(self):
        return self._underlying.query()

    def elapsed_time(self, other):
        return self._underlying.elapsed_time(other._underlying)

    def wait(self, stream=0):
        self._underlying.wait(stream)
