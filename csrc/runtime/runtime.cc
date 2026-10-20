#include "runtime.hpp"
#include "owner_drain_fail_stop.hpp"

#include "runtime/utils.hpp"

#include <infini/rt/device_guard.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>

namespace infinilm::runtime {
namespace {

uint64_t allocateOwnerGeneration() noexcept {
    static std::atomic<uint64_t> next_generation{1};
    auto candidate = next_generation.load(std::memory_order_relaxed);
    for (;;) {
        // Never wrap the counter: even a concurrent allocation cannot reuse
        // generation 1 while another thread is terminating on exhaustion.
        if (candidate == 0 || candidate == std::numeric_limits<uint64_t>::max()) {
            std::terminate();
        }
        if (next_generation.compare_exchange_weak(
                candidate, candidate + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return candidate;
        }
    }
}

uint64_t currentOwnerGeneration() noexcept {
    static thread_local const uint64_t generation = allocateOwnerGeneration();
    return generation;
}

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

std::shared_ptr<Runtime> Runtime::create(Device device) {
    if (device.index() < 0 || static_cast<size_t>(device.index()) >= context::getDeviceCount(device.type())) {
        throw std::invalid_argument("device index is unavailable for the requested backend");
    }
    infini::rt::DeviceGuard guard{device.type(), device.index(), true};
    return std::shared_ptr<Runtime>(new Runtime(device));
}

Runtime::Runtime(Device device) : device_(device), owner_generation_(currentOwnerGeneration()), graph_manager_(std::make_unique<::infinilm::execution::graph::GraphManager>()) {
    try {
        device_memory_allocator_ = std::make_unique<infini::rt::ReusableMemoryPool>(device);
        // Borrow the RT-owned execution stream. Pin leases retain its state;
        // Runtime never independently destroys this queue.
        stream_ = device_memory_allocator_->stream();
    } catch (...) {
        device_memory_allocator_.reset();
        stream_ = nullptr;
        throw;
    }
}
Runtime::~Runtime() noexcept {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index()};

    try {
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
    device_memory_allocator_.reset();
    stream_ = nullptr; // Pool state drains/frees/destroys after its last lease.
}

Runtime *Runtime::activate() {
    INFINILM_ASSERT(device_.type() != infini::rt::Device::Type::kCount);
    infini::rt::set_runtime_device_type(device_.type());
    INFINILM_CHECK_ERROR(infini::rt::runtime::SetDevice(device_.index()));
    return this;
}

void Runtime::assertOwnerThread() const {
    if (currentOwnerGeneration() != owner_generation_) {
        throw std::runtime_error("Runtime execution belongs to another worker thread");
    }
}

Device Runtime::device() const {
    return device_;
}

infini::rt::runtime::Stream Runtime::stream() const {
    return stream_;
}

void Runtime::enableOwnerDrainFailStop() {
    assertOwnerThread();
    owner_drain_fail_stop_ = true;
}
void Runtime::drainBeforeHostCopyUnwind() noexcept {
    if (!owner_drain_fail_stop_) {
        return;
    }
    try {
        assertOwnerThread();
        infini::rt::DeviceGuard guard{device_.type(), device_.index(), true};
        const auto status = infini::rt::runtime::StreamSynchronize(stream());
        detail::require_proven_owner_drain(true, status == infini::rt::runtime::kSuccess);
    } catch (...) { detail::require_proven_owner_drain(true, false); }
}
void Runtime::syncStream() {
    try {
        infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
        const auto status = infini::rt::runtime::StreamSynchronize(stream());
        detail::require_proven_owner_drain(owner_drain_fail_stop_, status == infini::rt::runtime::kSuccess);
        INFINILM_CHECK_ERROR(status);
    } catch (...) {
        detail::require_proven_owner_drain(owner_drain_fail_stop_, false);
        throw;
    }
}

void Runtime::syncDevice() {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    INFINILM_CHECK_ERROR(infini::rt::runtime::DeviceSynchronize());
}

void Runtime::syncStreamForCleanup() noexcept {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index()};
    if (device_guard.active()) {
        try {
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
}

void Runtime::trimMemory() {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    device_memory_allocator_->trim();
}

size_t Runtime::graphPinnedMemoryBytes() {
    assertOwnerThread();
    return device_memory_allocator_->pinned_bytes();
}

void Runtime::releaseDeviceMemory(std::byte *ptr) noexcept {
    bool release_is_safe = true;
    // Reuse on the owning worker remains ordered on this Runtime's stream.
    // A foreign thread must wait for already-submitted work before returning
    // the block to the pool; synchronizing the whole device is unnecessary.
    if (device_.type() != Device::Type::kCpu
        && currentOwnerGeneration() != owner_generation_) {
        infini::rt::DeviceGuard device_guard{device_.type(), device_.index()};
        if (device_guard.active()) {
            const auto status = infini::rt::runtime::StreamSynchronize(stream_);
            warn_runtime_cleanup_failure("synchronizing a foreign-thread allocation release", status);
            release_is_safe = status == infini::rt::runtime::kSuccess;
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
}

std::shared_ptr<Memory> Runtime::allocateMemory(size_t size) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    auto runtime = shared_from_this();
    std::byte *data_ptr = runtime->device_memory_allocator_->allocate(size);
    try {
        return std::make_shared<Memory>(
            data_ptr, size, device_,
            [runtime](std::byte *p) { runtime->releaseDeviceMemory(p); },
            false, size > 0, runtime); // Zero allocations have no allocator block to pin.
    } catch (...) {
        runtime->device_memory_allocator_->deallocate(data_ptr);
        throw;
    }
}

std::shared_ptr<Memory> Runtime::allocateParameterMemory(size_t size) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    auto runtime = shared_from_this();
    std::byte *data_ptr = runtime->device_memory_allocator_->allocate_exact(size);
    try {
        return std::make_shared<Memory>(
            data_ptr, size, device_,
            [runtime](std::byte *p) { runtime->releaseDeviceMemory(p); },
            false, true, runtime);
    } catch (...) {
        runtime->device_memory_allocator_->deallocate(data_ptr);
        throw;
    }
}

std::shared_ptr<Memory> Runtime::allocatePinnedHostMemory(size_t size) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    if (device_.type() == Device::Type::kCpu) {
        spdlog::warn("For CPU devices, pinned memory is not supported, falling back to regular host memory");
        return allocateMemory(size);
    }
    auto runtime = shared_from_this();
    void *data_ptr = nullptr;
    if (size != 0) {
        INFINILM_CHECK_ERROR(infini::rt::runtime::MallocHost(&data_ptr, size));
    }
    try {
        return std::make_shared<Memory>(
            static_cast<std::byte *>(data_ptr), size, Device{Device::Type::kCpu},
            [runtime](std::byte *p) {
                infini::rt::DeviceGuard guard{runtime->device_.type(), runtime->device_.index()};
                if (!guard.active() || p == nullptr) {
                    return;
                }
                try {
                    INFINILM_CHECK_ERROR(infini::rt::runtime::FreeHost(p));
                } catch (const std::exception &error) {
                    warn_runtime_cleanup_failure("releasing pinned host memory", error.what());
                } catch (...) {
                    warn_runtime_cleanup_failure("releasing pinned host memory", "unknown error");
                }
            },
            true, false, runtime);
    } catch (...) {
        if (data_ptr != nullptr) {
            INFINILM_CHECK_ERROR(infini::rt::runtime::FreeHost(data_ptr));
        }
        throw;
    }
}

std::shared_ptr<Memory> Runtime::reinstantiateBlob(std::shared_ptr<Memory> blob) {
    assertOwnerThread();
    if (!blob || blob->device() != device_) {
        throw std::invalid_argument("graph blob belongs to another runtime device");
    }
    auto runtime = shared_from_this();
    auto ptr = blob->data();
    // Every resumed view has its own logical span. The allocator's Block
    // use_count is the shared physical lease, incremented once per view and
    // decremented once by that view's Memory deleter.
    const auto block_size = device_memory_allocator_->retain_allocation(ptr);
    try {
        if (blob->size() > block_size) {
            throw std::out_of_range("graph blob exceeds its allocator block");
        }
        return std::make_shared<Memory>(
            ptr, blob->size(), device_,
            [runtime](std::byte *p) { runtime->releaseDeviceMemory(p); },
            false, true, runtime);
    } catch (...) {
        runtime->releaseDeviceMemory(ptr);
        throw;
    }
}

bool Runtime::retainGraphMemory(const std::shared_ptr<Memory> &memory) {
    assertOwnerThread();
    if (!memory || memory->device() != device_) {
        throw std::runtime_error(
            "graph capture cannot retain memory from a different device");
    }
    return device_memory_allocator_->retain_for_capture(memory->data());
}

void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async) {
    try {
        infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
        if (device_.type() == Device::Type::kCpu) {
            INFINILM_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice));
            return;
        }

        const auto current_stream = stream();
        INFINILM_CHECK_ERROR(
            infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyHostToDevice, current_stream));
        if (!async) {
            syncStream();
        }
    } catch (...) {
        drainBeforeHostCopyUnwind(); // before deep host buffer destructors
        throw;
    }
}

void Runtime::memcpyD2H(void *dst, const void *src, size_t size) {
    try {
        infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
        if (device_.type() == Device::Type::kCpu) {
            INFINILM_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToHost));
            return;
        }

        const auto current_stream = stream();
        INFINILM_CHECK_ERROR(
            infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyDeviceToHost, current_stream));
        syncStream();
    } catch (...) {
        drainBeforeHostCopyUnwind(); // before deep host buffer destructors
        throw;
    }
}

void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    if (device_.type() == Device::Type::kCpu) {
        INFINILM_CHECK_ERROR(infini::rt::runtime::Memcpy(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice));
        return;
    }

    const auto current_stream = stream();
    INFINILM_CHECK_ERROR(
        infini::rt::runtime::MemcpyAsync(dst, src, size, infini::rt::runtime::kMemcpyDeviceToDevice, current_stream));
    if (!async) {
        INFINILM_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
    }
}

void Runtime::setDeviceMemory(void *ptr, int value, size_t count) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    if (device_.type() == Device::Type::kCpu) {
        INFINILM_CHECK_ERROR(infini::rt::runtime::Memset(ptr, value, count));
        return;
    }

    const auto current_stream = stream();
    INFINILM_CHECK_ERROR(infini::rt::runtime::MemsetAsync(ptr, value, count, current_stream));
    INFINILM_CHECK_ERROR(infini::rt::runtime::StreamSynchronize(current_stream));
}

void Runtime::setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream) {
    infini::rt::DeviceGuard device_guard{device_.type(), device_.index(), true};
    if (device_.type() != Device::Type::kCpu) {
        if (stream == nullptr) {
            stream = this->stream();
        }
        INFINILM_CHECK_ERROR(infini::rt::runtime::MemsetAsync(ptr, value, count, stream));
    } else {
        INFINILM_CHECK_ERROR(infini::rt::runtime::Memset(ptr, value, count));
    }
}

::infinilm::execution::graph::GraphManager::CaptureState Runtime::graphCaptureState() const {
    return graph_manager_->capture_state();
}

bool Runtime::isGraphRecording() const {
    return graph_manager_->is_recording();
}

void Runtime::startGraphRecording(bool require_device_graph) {
    graph_manager_->start_recording(require_device_graph);
    try {
        device_memory_allocator_->begin_pin_mode();
    } catch (...) {
        graph_manager_->cancel_recording();
        throw;
    }
}

void Runtime::addGraphOperator(std::shared_ptr<::infinilm::execution::graph::RecordedCall> op) {
    return graph_manager_->add_operator(op);
}

std::shared_ptr<::infinilm::execution::graph::Graph> Runtime::stopGraphRecording() {
    if (!graph_manager_->is_recording()) {
        return graph_manager_->stop_recording();
    }

    std::shared_ptr<::infinilm::execution::graph::Graph> graph;
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

    std::shared_ptr<infini::rt::ReusableMemoryPool::PinLease> allocation_lease;
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
    try {
        graph->instantiate();
    } catch (...) {
        graph_manager_->finish_recording();
        throw;
    }
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

} // namespace infinilm::runtime
