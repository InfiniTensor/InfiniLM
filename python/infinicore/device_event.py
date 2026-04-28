import infinicore.device
from infinicore.lib import _infinicore


class DeviceEvent:
    """A device event for timing operations and synchronization across devices.

    Similar to torch.cuda.Event, this class provides functionality to:
    - Record events on specific device streams
    - Synchronize with events
    - Measure elapsed time between events
    - Query event completion status
    - Make streams wait for events

    Args:
        enable_timing: Whether the event should record timing data. Default: False.
        blocking: Whether to use blocking synchronization. Default: False.
        interprocess: Whether the event can be used for inter-process communication. Default: False.
        external: Whether the event is an external event. Default: False.
        device: Target device for this event. If None, uses current device.
    """

    def __init__(self, enable_timing=False, device=None):
        # Build flags based on parameters
        flags = 0
        if not enable_timing:
            flags |= 0x2  # DISABLE_TIMING
        # if blocking:
        #     flags |= 0x1  # BLOCKING_SYNC

        # Store parameters for reference
        self._enable_timing = enable_timing
        # self._blocking = blocking
        # self._interprocess = interprocess
        # self._external = external

        if device is None:
            # Use current device
            if flags == 0:
                self._underlying = _infinicore.DeviceEvent()
            else:
                self._underlying = _infinicore.DeviceEvent(flags)
        elif flags == 0:
            # Construct with device only
            self._underlying = _infinicore.DeviceEvent(device._underlying)
        else:
            # Construct with both device and flags
            self._underlying = _infinicore.DeviceEvent(device._underlying, flags)

    def record(self, stream=None):
        """Record the event.

        Args:
            stream: Stream to record the event on. If None, uses current stream.
        """
        if stream is None:
            self._underlying.record()
        else:
            self._underlying.record(stream)

    def synchronize(self):
        """Wait for the event to complete (blocking)."""
        self._underlying.synchronize()

    def query(self):
        """Check if the event has been completed.

        Returns:
            bool: True if completed, False otherwise.
        """
        return self._underlying.query()

    def elapsed_time(self, other):
        """Calculate elapsed time between this event and another event.

        Args:
            other: The other DeviceEvent to compare with

        Returns:
            float: Elapsed time in milliseconds between this event and the other event

        Raises:
            RuntimeError: If events are on different devices or not recorded,
                         or if timing is disabled on either event
        """
        if not self._enable_timing or not other._enable_timing:
            raise RuntimeError("Cannot measure elapsed time when timing is disabled")
        return self._underlying.elapsed_time(other._underlying)

    def wait(self, stream=None):
        """Make a stream wait for this event to complete.

        Args:
            stream: Stream to make wait for this event. If None, uses current stream.
        """
        self._underlying.wait(stream)

    @property
    def device(self):
        """Get the device where this event was created."""
        return infinicore.device._from_infinicore_device(self._underlying.device)

    @property
    def is_recorded(self):
        """Check if the event has been recorded."""
        return self._underlying.is_recorded

    @property
    def enable_timing(self):
        """Whether this event records timing data."""
        return self._enable_timing

    @property
    def blocking(self):
        """Whether this event uses blocking synchronization."""
        return self._blocking

    @property
    def interprocess(self):
        """Whether this event can be used for inter-process communication."""
        return self._interprocess

    def __repr__(self):
        flags_str = []
        if not self._enable_timing:
            flags_str.append("timing_disabled")
        if self._blocking:
            flags_str.append("blocking")
        if self._interprocess:
            flags_str.append("interprocess")
        if self._external:
            flags_str.append("external")
        if not flags_str:
            flags_str.append("default")

        return f"DeviceEvent(device={self.device}, flags={', '.join(flags_str)}, recorded={self.is_recorded})"
