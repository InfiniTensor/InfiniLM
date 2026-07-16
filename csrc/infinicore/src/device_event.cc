#include "infinicore.hpp"

namespace infinicore {
namespace {

void warn_cleanup_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during DeviceEvent cleanup: {}", operation, detail);
    } catch (...) {
    }
}

void warn_cleanup_failure(const char *operation, infini::rt::runtime::Error status) noexcept {
    try {
        spdlog::warn("{} failed during DeviceEvent cleanup with error code {}",
                     operation,
                     static_cast<long long>(status));
    } catch (...) {
    }
}

class ScopedDevice {
public:
    explicit ScopedDevice(const Device &target)
        : original_(context::getDevice()), changed_(original_ != target) {
        if (changed_) {
            try {
                context::setDevice(target);
            } catch (...) {
                restore();
                throw;
            }
        }
    }

    ScopedDevice(const ScopedDevice &) = delete;
    ScopedDevice &operator=(const ScopedDevice &) = delete;

    ~ScopedDevice() noexcept { restore(); }

private:
    void restore() noexcept {
        if (!changed_) {
            return;
        }
        changed_ = false;
        try {
            context::setDevice(original_);
        } catch (const std::exception &error) {
            warn_cleanup_failure("restoring the previous device", error.what());
        } catch (...) {
            warn_cleanup_failure("restoring the previous device", "unknown error");
        }
    }

    Device original_;
    bool changed_;
};

void destroy_event_noexcept(infini::rt::runtime::Event event, const Device &device) noexcept {
    if (event == nullptr) {
        return;
    }
    try {
        ScopedDevice guard{device};
        const auto status = infini::rt::runtime::EventDestroy(event);
        if (status != infini::rt::runtime::kSuccess) {
            warn_cleanup_failure("destroying the event", status);
        }
    } catch (const std::exception &error) {
        warn_cleanup_failure("destroying the event", error.what());
    } catch (...) {
        warn_cleanup_failure("destroying the event", "unknown error");
    }
}

} // namespace

DeviceEvent::DeviceEvent()
    : device_(context::getDevice()), is_recorded_(false) {
    event_ = context::createEvent();
}

DeviceEvent::DeviceEvent(uint32_t flags)
    : device_(context::getDevice()), is_recorded_(false) {
    event_ = context::createEventWithFlags(flags);
}

DeviceEvent::DeviceEvent(Device device)
    : device_(device), is_recorded_(false) {
    ScopedDevice guard{device_};
    event_ = context::createEvent();
}

DeviceEvent::DeviceEvent(Device device, uint32_t flags)
    : device_(device), is_recorded_(false) {
    ScopedDevice guard{device_};
    event_ = context::createEventWithFlags(flags);
}

DeviceEvent::DeviceEvent(DeviceEvent &&other) noexcept
    : event_(other.event_), device_(other.device_), is_recorded_(other.is_recorded_) {
    other.event_ = nullptr;
    other.is_recorded_ = false;
}

DeviceEvent &DeviceEvent::operator=(DeviceEvent &&other) noexcept {
    if (this != &other) {
        destroy_event_noexcept(event_, device_);
        event_ = other.event_;
        device_ = other.device_;
        is_recorded_ = other.is_recorded_;
        other.event_ = nullptr;
        other.is_recorded_ = false;
    }
    return *this;
}

DeviceEvent::~DeviceEvent() noexcept {
    destroy_event_noexcept(event_, device_);
}

void DeviceEvent::record() {
    ScopedDevice guard{device_};
    context::recordEvent(event_);
    is_recorded_ = true;
}

void DeviceEvent::record(infini::rt::runtime::Stream stream) {
    ScopedDevice guard{device_};
    context::recordEvent(event_, stream);
    is_recorded_ = true;
}

void DeviceEvent::synchronize() {
    ScopedDevice guard{device_};
    context::synchronizeEvent(event_);
}

bool DeviceEvent::query() const {
    ScopedDevice guard{device_};
    return context::queryEvent(event_);
}

float DeviceEvent::elapsed_time(const DeviceEvent &other) const {
    // Both events must be on the same device
    if (device_ != other.device_) {
        throw std::runtime_error("Cannot measure elapsed time between events on different devices");
    }

    // Both events must be recorded
    if (!is_recorded_ || !other.is_recorded_) {
        throw std::runtime_error("Both events must be recorded before measuring elapsed time");
    }

    ScopedDevice guard{device_};
    return context::elapsedTime(event_, other.event_);
}

void DeviceEvent::wait(infini::rt::runtime::Stream stream) const {
    ScopedDevice guard{device_};
    context::streamWaitEvent(stream, event_);
}

} // namespace infinicore
