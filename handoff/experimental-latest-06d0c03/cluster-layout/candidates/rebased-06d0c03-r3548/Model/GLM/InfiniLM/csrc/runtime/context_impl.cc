#include "context_impl.hpp"

#include "runtime/utils.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::runtime {
namespace {

constexpr std::array<Device::Type, static_cast<size_t>(Device::Type::kCount)> kDefaultDevicePriority{
    Device::Type::kNvidia,
    Device::Type::kCambricon,
    Device::Type::kAscend,
    Device::Type::kMetax,
    Device::Type::kMoore,
    Device::Type::kIluvatar,
    Device::Type::kHygon,
    Device::Type::kCpu,
};

} // namespace

thread_local std::shared_ptr<Runtime> ContextImpl::current_runtime_;
thread_local std::array<std::vector<std::weak_ptr<Runtime>>,
                        static_cast<size_t>(Device::Type::kCount)>
    ContextImpl::thread_runtimes_;

std::shared_ptr<Runtime> ContextImpl::resolveRuntime(Device device) {
    const auto type_index = static_cast<size_t>(device.type());
    const auto device_index = static_cast<size_t>(device.index());
    const auto device_count = getDeviceCount(device.type());
    if (device_index >= device_count) {
        throw std::invalid_argument("device index is unavailable for the requested backend");
    }

    auto &device_runtimes = thread_runtimes_[type_index];
    if (device_runtimes.size() < device_count) {
        device_runtimes.resize(device_count);
    }
    auto &cached = device_runtimes[device_index];
    if (auto runtime = cached.lock()) {
        return runtime;
    }

    auto runtime = Runtime::create(device);
    cached = runtime;
    return runtime;
}

template <Device::Type device_type>
void ContextImpl::initializeDeviceType() {
    if constexpr (infini::rt::DeviceEnabled<device_type>::value) {
        int device_count = 0;
        // Query only the selected backend without changing the caller's RT
        // selection. CPU loading must not initialize an unrelated GPU driver.
        INFINILM_CHECK_ERROR(infini::rt::runtime::GetDeviceCount(device_type, &device_count));
        if (device_count < 0) {
            throw std::runtime_error("runtime returned a negative device count");
        }
        device_counts_[static_cast<size_t>(device_type)] = static_cast<size_t>(device_count);
    }
}

void ContextImpl::initializeDeviceTypeLocked(Device::Type type) {
    const auto type_index = static_cast<size_t>(type);
    if (type_index >= device_counts_.size()) {
        throw std::invalid_argument("invalid device type");
    }
    if (device_types_initialized_[type_index]) {
        return;
    }
    switch (type) {
    case Device::Type::kCpu:
        initializeDeviceType<Device::Type::kCpu>();
        break;
    case Device::Type::kNvidia:
        initializeDeviceType<Device::Type::kNvidia>();
        break;
    case Device::Type::kCambricon:
        initializeDeviceType<Device::Type::kCambricon>();
        break;
    case Device::Type::kAscend:
        initializeDeviceType<Device::Type::kAscend>();
        break;
    case Device::Type::kMetax:
        initializeDeviceType<Device::Type::kMetax>();
        break;
    case Device::Type::kMoore:
        initializeDeviceType<Device::Type::kMoore>();
        break;
    case Device::Type::kIluvatar:
        initializeDeviceType<Device::Type::kIluvatar>();
        break;
    case Device::Type::kHygon:
        initializeDeviceType<Device::Type::kHygon>();
        break;
    default:
        break;
    }
    // A failed probe is not cached as an unavailable device. A caller that
    // explicitly asks for that backend still receives the original error.
    device_types_initialized_[type_index] = true;
}

Runtime *ContextImpl::getCurrentRuntime() {
    if (current_runtime_ != nullptr) {
        spdlog::debug("getCurrentRuntime() returning {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
        return current_runtime_.get();
    }

    spdlog::debug("current_runtime_ is null, performing lazy initialization");
    for (const auto device_type : kDefaultDevicePriority) {
        if (getDeviceCount(device_type) != 0) {
            current_runtime_ = resolveRuntime(Device{device_type, 0});
            break;
        }
    }

    INFINILM_ASSERT(current_runtime_ != nullptr);
    current_runtime_->activate();
    spdlog::debug("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
    return current_runtime_.get();
}

std::shared_ptr<Runtime> ContextImpl::getCurrentRuntimeShared() {
    getCurrentRuntime();
    return current_runtime_;
}

std::shared_ptr<Runtime> ContextImpl::setDevice(Device device) {
    if (current_runtime_ != nullptr && device == current_runtime_->device()) {
        current_runtime_->activate();
        return current_runtime_;
    }

    auto runtime = resolveRuntime(device);
    current_runtime_ = std::move(runtime);
    current_runtime_->activate();
    return current_runtime_;
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    const auto type_index = static_cast<size_t>(type);
    if (type_index >= device_counts_.size()) {
        throw std::invalid_argument("invalid device type");
    }
    std::lock_guard<std::mutex> lock{runtime_table_mutex_};
    initializeDeviceTypeLocked(type);
    return device_counts_[type_index];
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

namespace context {

std::shared_ptr<Runtime> currentRuntime() {
    return ContextImpl::singleton().getCurrentRuntimeShared();
}

void setDevice(Device device) {
    ContextImpl::singleton().setDevice(device);
}

std::shared_ptr<Runtime> resolveRuntime(Device device) {
    return ContextImpl::singleton().resolveRuntime(device);
}

Device getDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->device();
}

size_t getDeviceCount(Device::Type type) {
    return ContextImpl::singleton().getDeviceCount(type);
}

} // namespace context

} // namespace infinilm::runtime
