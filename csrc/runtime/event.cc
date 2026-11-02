#include "event.hpp"

#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include <infini/rt/device_guard.h>

#include <cstdio>
#include <stdexcept>

namespace infinilm::runtime {

std::shared_ptr<StreamEvent> StreamEvent::create(Runtime &runtime) {
    return std::shared_ptr<StreamEvent>(new StreamEvent(runtime));
}

StreamEvent::StreamEvent(Runtime &runtime) : device_(runtime.device()) {
    runtime.assertOwnerThread();
    infini::rt::DeviceGuard guard{device_.type(), device_.index(), true};
    INFINILM_CHECK_ERROR(infini::rt::runtime::EventCreate(&event_));
}

StreamEvent::~StreamEvent() noexcept {
    if (event_ == nullptr) {
        return;
    }
    infini::rt::DeviceGuard guard{device_.type(), device_.index()};
    if (!guard.active()) {
        return;
    }
    const auto status = infini::rt::runtime::EventDestroy(event_);
    if (status != infini::rt::runtime::kSuccess) {
        std::fprintf(stderr, "failed to destroy InfiniLM stream event: %lld\n",
                     static_cast<long long>(status));
    }
}

void StreamEvent::record(Runtime &runtime) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_) {
        throw std::invalid_argument("stream event record device mismatch");
    }
    infini::rt::DeviceGuard guard{device_.type(), device_.index(), true};
    INFINILM_CHECK_ERROR(
        infini::rt::runtime::EventRecord(event_, runtime.stream()));
}

void StreamEvent::wait(Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime.device() != device_) {
        throw std::invalid_argument("stream event wait device mismatch");
    }
    infini::rt::DeviceGuard guard{device_.type(), device_.index(), true};
    INFINILM_CHECK_ERROR(infini::rt::runtime::EventSynchronize(event_));
}

} // namespace infinilm::runtime
