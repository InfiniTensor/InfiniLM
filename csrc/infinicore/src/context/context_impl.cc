#include "context_impl.hpp"
#include "internal.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace infinicore {
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

void warn_graph_cleanup_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during graph cleanup: {}", operation, detail);
    } catch (...) {
    }
}

} // namespace

thread_local std::shared_ptr<Runtime> ContextImpl::current_runtime_;
thread_local std::shared_ptr<Runtime> ContextImpl::graph_runtime_;

std::shared_ptr<Runtime> ContextImpl::getOrCreateRuntimeLocked(Device device, const std::thread::id &thread_id) {
    const auto type_index = static_cast<size_t>(device.type());
    const auto device_index = static_cast<size_t>(device.index());
    INFINICORE_ASSERT(type_index < runtime_table_.size());
    INFINICORE_ASSERT(device_index < runtime_table_[type_index].size());

    auto &thread_runtimes = runtime_table_[type_index][device_index];
    if (const auto found = thread_runtimes.find(thread_id); found != thread_runtimes.end()) {
        if (auto runtime = found->second.lock()) {
            return runtime;
        }
        thread_runtimes.erase(found);
    }

    auto runtime = std::shared_ptr<Runtime>(new Runtime(device));
    thread_runtimes.emplace(thread_id, runtime);
    return runtime;
}

template <Device::Type device_type>
void ContextImpl::initializeDeviceType() {
    if constexpr (infini::rt::DeviceEnabled<device_type>::value) {
        int device_count = 0;
        infini::rt::set_runtime_device_type(device_type);
        INFINICORE_CHECK_ERROR(infini::rt::runtime::GetDeviceCount(&device_count));
        INFINICORE_ASSERT(device_count >= 0);
        runtime_table_[static_cast<size_t>(device_type)].resize(static_cast<size_t>(device_count));
    }
}

Runtime *ContextImpl::getCurrentRuntime() {
    if (current_runtime_ != nullptr) {
        spdlog::debug("getCurrentRuntime() returning {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
        return current_runtime_.get();
    }

    spdlog::debug("current_runtime_ is null, performing lazy initialization");
    const auto thread_id = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock{runtime_table_mutex_};
        for (const auto device_type : kDefaultDevicePriority) {
            const auto type_index = static_cast<size_t>(device_type);
            if (!runtime_table_[type_index].empty()) {
                current_runtime_ = getOrCreateRuntimeLocked(
                    Device{device_type, 0}, thread_id);
                break;
            }
        }
    }

    INFINICORE_ASSERT(current_runtime_ != nullptr);
    current_runtime_->activate();
    spdlog::debug("Lazy init: Set current_runtime_ to {} (ptr={})", current_runtime_->device().ToString(), static_cast<void *>(current_runtime_.get()));
    return current_runtime_.get();
}

void ContextImpl::setDevice(Device device) {
    if (device == getCurrentRuntime()->device()) {
        current_runtime_->activate();
        return;
    }

    std::shared_ptr<Runtime> runtime;
    {
        std::lock_guard<std::mutex> lock{runtime_table_mutex_};
        runtime = getOrCreateRuntimeLocked(device, std::this_thread::get_id());
    }
    current_runtime_ = std::move(runtime);
    current_runtime_->activate();
}

size_t ContextImpl::getDeviceCount(Device::Type type) {
    const auto type_index = static_cast<size_t>(type);
    if (type_index >= runtime_table_.size()) {
        throw std::invalid_argument("invalid device type");
    }
    std::lock_guard<std::mutex> lock{runtime_table_mutex_};
    return runtime_table_[type_index].size();
}

bool ContextImpl::isGraphRecording() {
    getCurrentRuntime();
    return graph_runtime_ != nullptr
        && graph_runtime_ == current_runtime_
        && graph_runtime_->isGraphRecording();
}

void ContextImpl::startGraphRecording() {
    getCurrentRuntime();
    if (graph_runtime_ != nullptr) {
        throw std::runtime_error("graph recording is already active on this thread");
    }
    current_runtime_->startGraphRecording();
    graph_runtime_ = current_runtime_;
}

void ContextImpl::addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    getCurrentRuntime();
    if (graph_runtime_ == nullptr || graph_runtime_ != current_runtime_) {
        throw std::runtime_error("cannot record a graph operator on a non-capture device");
    }
    graph_runtime_->addGraphOperator(std::move(op));
}

std::shared_ptr<graph::Graph> ContextImpl::stopGraphRecording() {
    getCurrentRuntime();
    if (graph_runtime_ == nullptr) {
        return current_runtime_->stopGraphRecording();
    }

    auto owner = graph_runtime_;
    auto previous = current_runtime_;
    current_runtime_ = owner;
    bool stop_attempted = false;
    try {
        current_runtime_->activate();
        stop_attempted = true;
        auto graph = owner->stopGraphRecording();
        graph_runtime_.reset();
        current_runtime_ = previous;
        current_runtime_->activate();
        return graph;
    } catch (...) {
        const auto original_error = std::current_exception();
        if (stop_attempted) {
            graph_runtime_.reset();
        }
        current_runtime_ = previous;
        try {
            current_runtime_->activate();
        } catch (const std::exception &error) {
            warn_graph_cleanup_failure("restoring the previous runtime", error.what());
        } catch (...) {
            warn_graph_cleanup_failure("restoring the previous runtime", "unknown error");
        }
        std::rethrow_exception(original_error);
    }
}

void ContextImpl::cancelGraphRecording() noexcept {
    auto owner = std::exchange(graph_runtime_, nullptr);
    if (owner == nullptr) {
        return;
    }

    auto previous = current_runtime_;
    current_runtime_ = owner;
    try {
        owner->activate();
    } catch (const std::exception &error) {
        warn_graph_cleanup_failure("activating the graph runtime", error.what());
    } catch (...) {
        warn_graph_cleanup_failure("activating the graph runtime", "unknown error");
    }
    owner->cancelGraphRecording();

    current_runtime_ = previous;
    if (current_runtime_ != nullptr) {
        try {
            current_runtime_->activate();
        } catch (const std::exception &error) {
            warn_graph_cleanup_failure("restoring the previous runtime", error.what());
        } catch (...) {
            warn_graph_cleanup_failure("restoring the previous runtime", "unknown error");
        }
    }
}

ContextImpl &ContextImpl::singleton() {
    static ContextImpl instance;
    return instance;
}

ContextImpl::ContextImpl() {
    initializeDeviceType<Device::Type::kCpu>();
    initializeDeviceType<Device::Type::kNvidia>();
    initializeDeviceType<Device::Type::kCambricon>();
    initializeDeviceType<Device::Type::kAscend>();
    initializeDeviceType<Device::Type::kMetax>();
    initializeDeviceType<Device::Type::kMoore>();
    initializeDeviceType<Device::Type::kIluvatar>();
    initializeDeviceType<Device::Type::kHygon>();

    const bool has_runtime = std::any_of(
        runtime_table_.begin(), runtime_table_.end(),
        [](const auto &devices) { return !devices.empty(); });
    INFINICORE_ASSERT(has_runtime);
}

namespace context {

void setDevice(Device device) {
    ContextImpl::singleton().setDevice(device);
}

Device getDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->device();
}

size_t getDeviceCount(Device::Type type) {
    return ContextImpl::singleton().getDeviceCount(type);
}

infini::rt::runtime::Stream getStream() {
    return ContextImpl::singleton().getCurrentRuntime()->stream();
}

void syncStream() {
    return ContextImpl::singleton().getCurrentRuntime()->syncStream();
}

void syncDevice() {
    return ContextImpl::singleton().getCurrentRuntime()->syncDevice();
}

void trimMemory() {
    return ContextImpl::singleton().getCurrentRuntime()->trimMemory();
}

std::shared_ptr<Memory> allocateMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocateMemory(size);
}

std::shared_ptr<Memory> allocateHostMemory(size_t size) {
    setDevice(Device{Device::Type::kCpu});
    return allocateMemory(size);
}

std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->allocatePinnedHostMemory(size);
}

void memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyH2D(dst, src, size, async);
}

void memcpyD2H(void *dst, const void *src, size_t size) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2H(dst, src, size);
}

void memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2D(dst, src, size, async);
}

void memcpyH2H(void *dst, const void *src, size_t size) {
    setDevice(Device{Device::Type::kCpu});
    return ContextImpl::singleton().getCurrentRuntime()->memcpyD2D(dst, src, size);
}

void setDeviceMemory(void *ptr, int value, size_t count) {
    return ContextImpl::singleton().getCurrentRuntime()->setDeviceMemory(ptr, value, count);
}

void setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream) {
    return ContextImpl::singleton().getCurrentRuntime()->setDeviceMemoryAsync(ptr, value, count, stream);
}

// Timing API implementations
infini::rt::runtime::Event createEvent() {
    return ContextImpl::singleton().getCurrentRuntime()->createEvent();
}

infini::rt::runtime::Event createEventWithFlags(uint32_t flags) {
    return ContextImpl::singleton().getCurrentRuntime()->createEventWithFlags(flags);
}

void recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream) {
    ContextImpl::singleton().getCurrentRuntime()->recordEvent(event, stream);
}

bool queryEvent(infini::rt::runtime::Event event) {
    return ContextImpl::singleton().getCurrentRuntime()->queryEvent(event);
}

void synchronizeEvent(infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->synchronizeEvent(event);
}

void destroyEvent(infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->destroyEvent(event);
}

float elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end) {
    return ContextImpl::singleton().getCurrentRuntime()->elapsedTime(start, end);
}

void streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event) {
    ContextImpl::singleton().getCurrentRuntime()->streamWaitEvent(stream, event);
}

bool isGraphRecording() {
    return ContextImpl::singleton().isGraphRecording();
}

void startGraphRecording() {
    ContextImpl::singleton().startGraphRecording();
}

void addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    ContextImpl::singleton().addGraphOperator(std::move(op));
}

std::shared_ptr<graph::Graph> stopGraphRecording() {
    return ContextImpl::singleton().stopGraphRecording();
}

void cancelGraphRecording() noexcept {
    ContextImpl::singleton().cancelGraphRecording();
}

std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob) {
    setDevice(blob->device());
    return ContextImpl::singleton().getCurrentRuntime()->reinstantiateBlob(blob);
}

void retainGraphMemory(const std::shared_ptr<Memory> &memory) {
    ContextImpl::singleton().getCurrentRuntime()->retainGraphMemory(memory);
}

} // namespace context

} // namespace infinicore
