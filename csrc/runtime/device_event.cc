#include "runtime/device_event.hpp"
#include "runtime/runtime.hpp"
#include <spdlog/spdlog.h>
#include <utility>

namespace infinilm::runtime {
namespace {

void warn_cleanup_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during DeviceEvent cleanup: {}", operation, detail);
    } catch (...) {
    }
}

void destroy_event_noexcept(infini::rt::runtime::Event event, const std::shared_ptr<Runtime> &runtime) noexcept {
    if (event == nullptr) {
        return;
    }
    try {
        runtime->destroyEvent(event);
    } catch (const std::exception &error) {
        warn_cleanup_failure("destroying the event", error.what());
    } catch (...) {
        warn_cleanup_failure("destroying the event", "unknown error");
    }
}

} // namespace

DeviceEvent::DeviceEvent(std::shared_ptr<Runtime> runtime)
    : runtime_(std::move(runtime)), is_recorded_(false) {
    if (!runtime_) throw std::invalid_argument("DeviceEvent requires a Runtime owner");
    device_ = runtime_->device();
    event_ = runtime_->createEvent();
}

DeviceEvent::DeviceEvent(std::shared_ptr<Runtime> runtime, uint32_t flags)
    : runtime_(std::move(runtime)), is_recorded_(false) {
    if (!runtime_) throw std::invalid_argument("DeviceEvent requires a Runtime owner");
    device_ = runtime_->device();
    event_ = runtime_->createEventWithFlags(flags);
}

DeviceEvent::DeviceEvent(DeviceEvent &&other) noexcept
    : event_(other.event_), runtime_(std::move(other.runtime_)), device_(other.device_), is_recorded_(other.is_recorded_) {
    other.event_ = nullptr;
    other.is_recorded_ = false;
}

DeviceEvent &DeviceEvent::operator=(DeviceEvent &&other) noexcept {
    if (this != &other) {
        destroy_event_noexcept(event_, runtime_);
        event_ = other.event_;
        runtime_ = std::move(other.runtime_);
        device_ = other.device_;
        is_recorded_ = other.is_recorded_;
        other.event_ = nullptr;
        other.is_recorded_ = false;
    }
    return *this;
}

DeviceEvent::~DeviceEvent() noexcept {
    destroy_event_noexcept(event_, runtime_);
}

void DeviceEvent::record() {
    runtime_->recordEvent(event_);
    is_recorded_ = true;
}

void DeviceEvent::record(infini::rt::runtime::Stream stream) {
    runtime_->recordEvent(event_, stream);
    is_recorded_ = true;
}

void DeviceEvent::synchronize() {
    runtime_->synchronizeEvent(event_);
}

bool DeviceEvent::query() const {
    return runtime_->queryEvent(event_);
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

    return runtime_->elapsedTime(event_, other.event_);
}

void DeviceEvent::wait(infini::rt::runtime::Stream stream) const {
    runtime_->streamWaitEvent(stream, event_);
}

} // namespace infinilm::runtime
