import infinicore.device
from infinicore.graph import Graph
from infinicore.lib import _infinicore


def get_device():
    """Get the current active device.

    Returns:
        device: The current active device object
    """
    return _infinicore.get_device()


def get_device_count(device_type):
    """Get the number of available devices of a specific type.

    Args:
        device_type (str): The type of device to count (e.g., "cuda", "cpu", "npu")

    Returns:
        int: The number of available devices of the specified type
    """
    return _infinicore.get_device_count(infinicore.device(device_type)._underlying.type)


def set_device(device):
    """Set the current active device.

    Args:
        device: The device to set as active
    """
    _infinicore.set_device(device._underlying)


def sync_stream():
    """Synchronize the current stream."""
    _infinicore.sync_stream()


def sync_device():
    """Synchronize the current device."""
    _infinicore.sync_device()


def get_stream():
    """Get the current stream.

    Returns:
        stream: The current stream object
    """
    return _infinicore.get_stream()


def is_graph_recording():
    """Check if the current graph is recording.

    Returns:
        bool: True if the current graph is recording, False otherwise
    """
    return _infinicore.is_graph_recording()


def start_graph_recording(device=None):
    """Start recording the current graph."""
    if device is not None:
        set_device(device)
    _infinicore.start_graph_recording()


def stop_graph_recording():
    """Stop recording the current graph."""
    return Graph(_infinicore.stop_graph_recording())
