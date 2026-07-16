#include "runtime.hpp"

#include "../../utils.hpp"

#include "../allocators/device_pinned_allocator.hpp"
#include "../allocators/host_allocator.hpp"
#include "../allocators/pinnable_block_allocator.hpp"
#include "../allocators/stream_ordered_allocator.hpp"
#include "../context_impl.hpp"

#include <exception>

namespace infinicore {
namespace {

void warn_runtime_cleanup_failure(const char *operation, infini::rt::runtime::Error status) noexcept {
    if (status == infini::rt::runtime::kSuccess) {
        return;
    }
    try {
        spdlog::warn("{} failed during Runtime cleanup with error code {}",
                     operation,
                     static_cast<long long>(status));
    } catch (...) {
    }
}

void warn_runtime_cleanup_failure(const char *operation, const char *detail) noexcept {
    try {
        spdlog::warn("{} failed during Runtime cleanup: {}", operation, detail);
    } catch (...) {
    }
}

} // namespace

Runtime::Runtime(Device device) : device_(device), graph_manager_(std::make_unique<graph::GraphManager>()) {
    activate();
    if (device_.type() == Device::Type::kCpu) {
        device_memory_allocator_ = std::make_unique<PinnableBlockAllocator>(device);
    } else {
        device_memory_allocator_ = std::make_unique<PinnableBlockAllocator>(device);
        pinned_host_memory_allocator_ = std::make_unique<DevicePinnedHostAllocator>(device);
    }
}
Runtime::~Runtime() noexcept {
    Runtime *restore_runtime = ContextImpl::current_runtime_.get();
    infini::rt::set_runtime_device_type(device_.type());
    const auto set_device_status = infini::rt::runtime::SetDevice(device_.index());
    warn_runtime_cleanup_failure("selecting the runtime device", set_device_status);

    try {
        std::lock_guard<std::mutex> lock{stream_mutex_};
        if (stream_ != nullptr) {
            const auto synchronize_status = infini::rt::runtime::StreamSynchronize(stream_);
            warn_runtime_cleanup_failure("synchronizing the runtime stream", synchronize_status);
        }
    } catch (const std::exception &error) {
        warn_runtime_cleanup_failure("synchronizing the runtime stream", error.what());
    } catch (...) {
        warn_runtime_cleanup_failure("synchronizing the runtime stream", "unknown error");
    }

    graph_manager_.reset();
    pinned_host_memory_allocator_.reset();
    device_memory_allocator_.reset();
    try {
        std::lock_guard<std::mutex> lock{stream_mutex_};
        if (stream_ != nullptr) {
            const auto destroy_status = infini::rt::runtime::StreamDestroy(stream_);
            warn_runtime_cleanup_failure("destroying the runtime stream", destroy_status);
            stream_ = nullptr;
        }
    } catch (const std::exception &error) {
        warn_runtime_cleanup_failure("destroying the runtime stream", error.what());
    } catch (...) {
        warn_runtime_cleanup_failure("destroying the runtime stream", "unknown error");
    }

    if (restore_runtime != nullptr && restore_runtime != this) {
        infini::rt::set_runtime_device_type(restore_runtime->device().type());
        warn_runtime_cleanup_failure(
            "restoring the active runtime device",
            infini::rt::runtime::SetDevice(restore_runtime->device().index()));
    }
}

Runtime *Runtime::activate() {
    INFINICORE_ASSERT(device_.type() != infini::rt::Device::Type::kCount);
    infini::rt::set_runtime_device_type(device_.type());
    INFINICORE_CHECK_ERROR(infini::rt::runtime::SetDevice(device_.index()));
    return this;
}

Device Runtime::device() const {
    return device_;
}

infini::rt::runtime::Stream Runtime::stream() const {
    infini::rt::set_runtime_device_type(device_.type());
    INFINICORE_CHECK_ERROR(infini::rt::runtime::SetDevice(device_.index()));

    std::lock_guard<std::mutex> lock{stream_mutex_};
    if (stream_ == nullptr) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamCreate(&stream_));
    }
    return stream_;
}

void Runtime::syncStream() {
    INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(stream()));
}

void Runtime::syncDevice() {
    INFINICORE_CHECK_ERROR(infini::rt::runtime::DeviceSynchronize());
}

void Runtime::syncStreamForCleanup() noexcept {
    Runtime *restore_runtime = ContextImpl::current_runtime_.get();
    infini::rt::set_runtime_device_type(device_.type());
    const auto set_device_status = infini::rt::runtime::SetDevice(device_.index());
    warn_runtime_cleanup_failure("selecting the graph runtime device", set_device_status);

    if (set_device_status == infini::rt::runtime::kSuccess) {
        try {
            std::lock_guard<std::mutex> lock{stream_mutex_};
            if (stream_ != nullptr) {
                const auto synchronize_status = infini::rt::runtime::StreamSynchronize(stream_);
                warn_runtime_cleanup_failure("synchronizing the graph runtime stream", synchronize_status);
            }
        } catch (const std::exception &error) {
            warn_runtime_cleanup_failure("synchronizing the graph runtime stream", error.what());
        } catch (...) {
            warn_runtime_cleanup_failure("synchronizing the graph runtime stream", "unknown error");
        }
    }

    if (restore_runtime != nullptr && restore_runtime != this) {
        infini::rt::set_runtime_device_type(restore_runtime->device().type());
        warn_runtime_cleanup_failure(
            "restoring the active runtime device",
            infini::rt::runtime::SetDevice(restore_runtime->device().index()));
    }
}

void Runtime::trimMemory() {
    device_memory_allocator_->trim();
}

void Runtime::releaseDeviceMemory(std::byte *ptr) noexcept {
    Runtime *restore_runtime = ContextImpl::current_runtime_.get();
    const bool cross_runtime_release = restore_runtime != this
                                    && device_.type() != Device::Type::kCpu;

    bool release_is_safe = true;
    if (cross_runtime_release) {
        infini::rt::set_runtime_device_type(device_.type());
        const auto set_device_status = infini::rt::runtime::SetDevice(device_.index());
        warn_runtime_cleanup_failure("selecting the allocation device", set_device_status);
        if (set_device_status == infini::rt::runtime::kSuccess) {
            const auto synchronize_status = infini::rt::runtime::DeviceSynchronize();
            warn_runtime_cleanup_failure("synchronizing a cross-runtime allocation", synchronize_status);
            release_is_safe = synchronize_status == infini::rt::runtime::kSuccess;
        } else {
            release_is_safe = false;
        }
    }

    if (release_is_safe) {
        try {
            device_memory_allocator_->deallocate(ptr);
        } catch (const std::exception &error) {
            warn_runtime_cleanup_failure("releasing device memory", error.what());
        } catch (...) {
            warn_runtime_cleanup_failure("releasing device memory", "unknown error");
        }
    }

    if (cross_runtime_release && restore_runtime != nullptr) {
        infini::rt::set_runtime_device_type(restore_runtime->device().type());
        warn_runtime_cleanup_failure(
            "restoring the active runtime device",
            infini::rt::runtime::SetDevice(restore_runtime->device().index()));
    }
}

std::shared_ptr<Memory> Runtime::allocateMemory(size_t size) {
    auto runtime = shared_from_this();
    std::byte *data_ptr = runtime->device_memory_allocator_->allocate(size);
    return std::make_shared<Memory>(
        data_ptr, size, device_,
        [runtime](std::byte *p) {
            runtime->releaseDeviceMemory(p);
        });
}

std::shared_ptr<Memory> Runtime::allocatePinnedHostMemory(size_t size) {
    if (!pinned_host_memory_allocator_) {
        spdlog::warn("For CPU devices, pinned memory is not supported, falling back to regular host memory");
        return allocateMemory(size);
    }
    auto runtime = shared_from_this();
    std::byte *data_ptr = runtime->pinned_host_memory_allocator_->allocate(size);
    return std::make_shared<Memory>(
        data_ptr, size, Device{Device::Type::kCpu},
        [runtime](std::byte *p) {
            runtime->pinned_host_memory_allocator_->deallocate(p);
        },
        true);
}

std::shared_ptr<Memory> Runtime::reinstantiateBlob(std::shared_ptr<Memory> blob) {
    std::lock_guard<std::mutex> lock(reinstantiated_blob_mutex_);

    auto ptr = blob->data();
    auto it = reinstantiated_blobs_.find(ptr);
    if (it != reinstantiated_blobs_.end()) {
        if (auto memory = it->second.lock()) {
            return memory;
        }
    }

    auto runtime = shared_from_this();
    runtime->device_memory_allocator_->mark_in_use_(ptr, true);
    auto memory = std::make_shared<Memory>(
        blob->data(), blob->size(), device_,
        [runtime](std::byte *p) {
            runtime->releaseDeviceMemory(p);
        });
    reinstantiated_blobs_[ptr] = memory;
    return memory;
}

void Runtime::retainGraphMemory(const std::shared_ptr<Memory> &memory) {
    if (memory->device() != device_) {
        throw std::runtime_error(
            "graph capture cannot retain memory from a different device");
    }
    device_memory_allocator_->retain_for_capture(memory->data());
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    if (device_.type() == Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice));
        return;
    }

    const auto current_stream = stream();
    INFINICORE_CHECK_ERROR(
        infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice, current_stream));
    if (!async) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
    }
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    if (device_.type() == Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToHost));
        return;
    }

    const auto current_stream = stream();
    INFINICORE_CHECK_ERROR(
        infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyDeviceToHost, current_stream));
    INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    if (device_.type() == Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice));
        return;
    }

    const auto current_stream = stream();
    INFINICORE_CHECK_ERROR(
        infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice, current_stream));
    if (!async) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
    }
}

void Runtime::setDeviceMemory(void *ptr, int value, size_t count) {
    if (device_.type() == Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Memset(ptr, value, count));
        return;
    }

    const auto current_stream = stream();
    INFINICORE_CHECK_ERROR(infini::rt::runtime::MemsetAsync(ptr, value, count, current_stream));
    INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
}

void Runtime::setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream) {
    if (device_.type() != Device::Type::kCpu) {
        if (stream == nullptr) {
            stream = this->stream();
        }
        INFINICORE_CHECK_ERROR(infini::rt::runtime::MemsetAsync(ptr, value, count, stream));
    } else {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Memset(ptr, value, count));
    }
}

// Timing method implementations
infini::rt::runtime::Event Runtime::createEvent() {
    infini::rt::runtime::Event event = nullptr;
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventCreate(&event));
    return event;
}

infini::rt::runtime::Event Runtime::createEventWithFlags(uint32_t flags) {
    infini::rt::runtime::Event event = nullptr;
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventCreateWithFlags(&event, flags));
    return event;
}

void Runtime::recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream) {
    if (stream == nullptr) {
        stream = this->stream();
    }
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventRecord(event, stream));
}

bool Runtime::queryEvent(infini::rt::runtime::Event event) {
    // InfiniRT does not expose a portable not-ready value, so every
    // non-success query result is conservatively reported as incomplete.
    return infini::rt::runtime::EventQuery(event) == infini::rt::runtime::kSuccess;
}

void Runtime::synchronizeEvent(infini::rt::runtime::Event event) {
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventSynchronize(event));
}

void Runtime::destroyEvent(infini::rt::runtime::Event event) {
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventDestroy(event));
}

float Runtime::elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end) {
    float ms;
    INFINICORE_CHECK_ERROR(infini::rt::runtime::EventElapsedTime(&ms, start, end));
    return ms;
}

void Runtime::streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event) {
    // Use current stream if no specific stream is provided
    if (stream == nullptr) {
        stream = this->stream();
    }
    INFINICORE_CHECK_ERROR(infini::rt::runtime::StreamWaitEvent(stream, event, 0));
}

graph::GraphManager::CaptureState Runtime::graphCaptureState() const {
    return graph_manager_->capture_state();
}

bool Runtime::isGraphRecording() const {
    return graph_manager_->is_recording();
}

void Runtime::startGraphRecording() {
    graph_manager_->start_recording();
    try {
        device_memory_allocator_->begin_pin_mode();
    } catch (...) {
        graph_manager_->cancel_recording();
        throw;
    }
}

void Runtime::addGraphOperator(std::shared_ptr<graph::GraphOperator> op) {
    return graph_manager_->add_operator(op);
}

std::shared_ptr<graph::Graph> Runtime::stopGraphRecording() {
    if (!graph_manager_->is_recording()) {
        return graph_manager_->stop_recording();
    }

    std::shared_ptr<graph::Graph> graph;
    try {
        graph = graph_manager_->stop_recording();
    } catch (...) {
        const auto original_error = std::current_exception();
        try {
            device_memory_allocator_->cancel_pin_mode();
        } catch (...) {
        }
        graph_manager_->finish_recording();
        std::rethrow_exception(original_error);
    }

    std::shared_ptr<PinnableBlockAllocator::PinLease> allocation_lease;
    try {
        allocation_lease = device_memory_allocator_->commit_pin_mode();
    } catch (...) {
        const auto original_error = std::current_exception();
        try {
            device_memory_allocator_->cancel_pin_mode();
        } catch (...) {
        }
        graph_manager_->finish_recording();
        std::rethrow_exception(original_error);
    }
    graph->retain_runtime(shared_from_this(), std::move(allocation_lease));
    graph_manager_->finish_recording();
    return graph;
}

void Runtime::cancelGraphRecording() noexcept {
    try {
        graph_manager_->cancel_recording();
    } catch (const std::exception &error) {
        warn_runtime_cleanup_failure("canceling graph recording", error.what());
    } catch (...) {
        warn_runtime_cleanup_failure("canceling graph recording", "unknown error");
    }
    try {
        device_memory_allocator_->cancel_pin_mode();
    } catch (const std::exception &error) {
        warn_runtime_cleanup_failure("rolling back graph allocations", error.what());
    } catch (...) {
        warn_runtime_cleanup_failure("rolling back graph allocations", "unknown error");
    }
}

std::string Runtime::toString() const {
    return fmt::format("Runtime({})", device_.ToString());
}

} // namespace infinicore
