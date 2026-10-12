#pragma once

#include <infini/rt/reusable_memory_pool.h>

#include "runtime/context.hpp"

#include "execution/graph/graph_manager.hpp"

#include <infini/rt.h>
#include <cstdint>

namespace infinilm::runtime {
class Runtime : public std::enable_shared_from_this<Runtime> {
private:
    Device device_;
    bool owner_drain_fail_stop_{false};
    void drainBeforeHostCopyUnwind() noexcept;
    const uint64_t owner_generation_;
    infini::rt::runtime::Stream stream_ = nullptr;
    std::unique_ptr<infini::rt::ReusableMemoryPool> device_memory_allocator_;
    std::unique_ptr<::infinilm::execution::graph::GraphManager> graph_manager_;

    void releaseDeviceMemory(std::byte *ptr) noexcept;
    void syncStreamForCleanup() noexcept;

    Runtime(Device device);

public:
    static std::shared_ptr<Runtime> create(Device device);
    ~Runtime() noexcept;

    Runtime *activate();
    void assertOwnerThread() const;
    // Owner-thread, monotonic opt-in before dedicated job-host allocations.
    // Existing host-copy/sync failures require proven completion or fail-stop;
    // CCL submission failure terminates the entire host process, all engines included.
    void enableOwnerDrainFailStop();
    // Read only after assertOwnerThread(); not a cross-thread/provider capability.
    bool ownerDrainFailStopEnabled() const noexcept { return owner_drain_fail_stop_; }

    Device device() const;
    infini::rt::runtime::Stream stream() const;

    void syncStream();
    void syncDevice();
    void trimMemory();

    // LM graph budgeting: unique InfiniCore allocator blocks retained by any
    // live capture. Excludes driver-internal CUDA Graph allocations.
    size_t graphPinnedMemoryBytes();

    std::shared_ptr<Memory> allocateMemory(size_t size);
    // Initial physical Parameter storage, tracked by the same owner/graph allocator.
    std::shared_ptr<Memory> allocateParameterMemory(size_t size);
    std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);
    std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob);
    bool retainGraphMemory(const std::shared_ptr<Memory> &memory);

    void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
    void memcpyD2H(void *dst, const void *src, size_t size);
    void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);

    void setDeviceMemory(void *ptr, int value, size_t count);
    void setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream);

    // Timing methods
    infini::rt::runtime::Event createEvent();
    infini::rt::runtime::Event createEventWithFlags(uint32_t flags);
    void recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream = nullptr);
    bool queryEvent(infini::rt::runtime::Event event);
    void synchronizeEvent(infini::rt::runtime::Event event);
    void destroyEvent(infini::rt::runtime::Event event);
    float elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end);
    void streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event);

    // Graph
    ::infinilm::execution::graph::GraphManager::CaptureState graphCaptureState() const;
    bool isGraphRecording() const;
    void startGraphRecording(bool require_device_graph = false);
    void addGraphOperator(std::shared_ptr<::infinilm::execution::graph::RecordedCall> op);
    std::shared_ptr<::infinilm::execution::graph::Graph> stopGraphRecording();
    void cancelGraphRecording() noexcept;

    std::string toString() const;

    friend class ::infinilm::execution::graph::Graph;
};
} // namespace infinilm::runtime
