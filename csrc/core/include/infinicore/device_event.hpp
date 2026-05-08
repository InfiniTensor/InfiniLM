#pragma once

#include "device.hpp"
#include "infinirt.h"
#include <memory>
#include <stdexcept>

namespace infinicore {

/**
 * @brief A device event for timing operations and synchronization across devices.
 *
 * Similar to torch.cuda.Event, this class provides functionality to:
 * - Record events on specific device streams
 * - Synchronize with events
 * - Measure elapsed time between events
 * - Query event completion status
 * - Make streams wait for events
 */
class DeviceEvent {
private:
    infinirtEvent_t event_; // Underlying event handle
    Device device_;         // Device where this event was created
    bool is_recorded_;      // Whether the event has been recorded

public:
    /**
     * @brief Construct a new DeviceEvent on the current device.
     */
    DeviceEvent();

    /**
     * @brief Construct a new DeviceEvent on the current device with specific flags.
     * @param flags Event creation flags (e.g., for timing, blocking sync)
     */
    explicit DeviceEvent(uint32_t flags);

    /**
     * @brief Construct a new DeviceEvent on a specific device.
     * @param device Target device for this event
     */
    explicit DeviceEvent(Device device);

    /**
     * @brief Construct a new DeviceEvent on a specific device with flags.
     * @param device Target device for this event
     * @param flags Event creation flags
     */
    DeviceEvent(Device device, uint32_t flags);

    // Disallow copying
    DeviceEvent(const DeviceEvent &) = delete;
    DeviceEvent &operator=(const DeviceEvent &) = delete;

    /**
     * @brief Move constructor.
     */
    DeviceEvent(DeviceEvent &&other) noexcept;

    /**
     * @brief Move assignment operator.
     */
    DeviceEvent &operator=(DeviceEvent &&other) noexcept;

    /**
     * @brief Destroy the DeviceEvent and release underlying resources.
     */
    ~DeviceEvent();

    /**
     * @brief Record the event on the current stream of its device.
     */
    void record();

    /**
     * @brief Record the event on a specific stream.
     * @param stream Stream to record the event on
     */
    void record(infinirtStream_t stream);

    /**
     * @brief Wait for the event to complete (blocking).
     */
    void synchronize();

    /**
     * @brief Check if the event has been completed.
     * @return true if completed, false otherwise
     */
    bool query() const;

    /**
     * @brief Calculate elapsed time between this event and another event (in milliseconds).
     * @param other The other event to compare with
     * @return Elapsed time in milliseconds
     * @throws std::runtime_error if events are on different devices or not recorded
     */
    float elapsed_time(const DeviceEvent &other) const;

    /**
     * @brief Make a stream wait for this event to complete.
     * @param stream Stream to make wait for this event (nullptr for current stream)
     */
    void wait(infinirtStream_t stream = nullptr) const;

    /**
     * @brief Get the device where this event was created.
     * @return Device associated with this event
     */
    Device device() const { return device_; }

    /**
     * @brief Get the underlying event handle.
     * @return Raw event handle
     */
    infinirtEvent_t get() const { return event_; }

    /**
     * @brief Check if the event has been recorded.
     * @return true if recorded, false otherwise
     */
    bool is_recorded() const { return is_recorded_; }
};

} // namespace infinicore
