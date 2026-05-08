import infinilm.core as infinicore
from infinilm.core.graph import Graph
from infinilm.core.lib import _core


def get_device():
    """Get the current active device.

    Returns:
        device: The current active device object
    """
    return _core.get_device()


def get_device_count(device_type):
    """Get the number of available devices of a specific type.

    Args:
        device_type (str): The type of device to count (e.g., "cuda", "cpu", "npu")

    Returns:
        int: The number of available devices of the specified type
    """
    return _core.get_device_count(infinicore.device(device_type)._underlying.type)


def set_device(device):
    """Set the current active device.

    Args:
        device: The device to set as active
    """
    _core.set_device(device._underlying)


def sync_stream():
    """Synchronize the current stream."""
    _core.sync_stream()


def sync_device():
    """Synchronize the current device."""
    _core.sync_device()


def get_stream():
    """Get the current stream.

    Returns:
        stream: The current stream object
    """
    return _core.get_stream()


def is_graph_recording():
    """Check if the current graph is recording.

    Returns:
        bool: True if the current graph is recording, False otherwise
    """
    return _core.is_graph_recording()


def start_graph_recording(device=None):
    """Start recording the current graph."""
    if device is not None:
        set_device(device)
    _core.start_graph_recording()


def stop_graph_recording():
    """Stop recording the current graph."""
    return Graph(_core.stop_graph_recording())
