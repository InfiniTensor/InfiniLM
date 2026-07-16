from infinicore.device import device
from infinicore.lib import _infinicore


def get_device():
    return device._from_underlying(_infinicore.get_device())


def get_device_count(device_type):
    return _infinicore.get_device_count(device(device_type)._underlying.type)


def set_device(target):
    _infinicore.set_device(device(target)._underlying)


def get_stream():
    return _infinicore.get_stream()


def sync_stream():
    _infinicore.sync_stream()


def sync_device():
    _infinicore.sync_device()


def is_graph_recording():
    return _infinicore.is_graph_recording()


def start_graph_recording(target=None):
    if target is not None:
        set_device(target)
    _infinicore.start_graph_recording()


def stop_graph_recording():
    from infinicore.graph import Graph

    return Graph(_infinicore.stop_graph_recording())


def cancel_graph_recording():
    _infinicore.cancel_graph_recording()
