"""One-time device conversion; submission is the native bound method."""
from .device import device
from .lib import _infinicore

StreamNotification = _infinicore.StreamNotification


def stream_notification(target, producer, consumer, event_flags,
                        producer_owner, consumer_owner, *, fixed_context=False):
    """Create a native notification; owner objects must retain both handles.

    Flags are backend-specific and must come from the actual runtime contract.
    Explicit fixed_context selects target at construction and requires the same
    backend/ordinal on subsequent submit calls. Complete before every reuse and
    explicitly close on the creating thread, or use the returned context manager.
    """
    return StreamNotification(device(target)._underlying, producer, consumer,
                              event_flags, producer_owner, consumer_owner,
                              fixed_context=fixed_context)
