#pragma once

#include "../allocators/pinnable_block_allocator.hpp"

#include "infinicore/context/context.hpp"

#include "../../graph/graph_manager.hpp"

#include <infini/rt.h>
#include <mutex>
#include <unordered_map>

namespace infinicore {
class ContextImpl;
class Runtime : public std::enable_shared_from_this<Runtime> {
private:
    Device device_;
    mutable std::mutex stream_mutex_;
    mutable infini::rt::runtime::Stream stream_ = nullptr;
    std::unique_ptr<PinnableBlockAllocator> device_memory_allocator_;
    std::unique_ptr<MemoryAllocator> pinned_host_memory_allocator_;
    std::unique_ptr<graph::GraphManager> graph_manager_;
    std::mutex reinstantiated_blob_mutex_;
    std::unordered_map<void *, std::weak_ptr<Memory>> reinstantiated_blobs_;

    void releaseDeviceMemory(std::byte *ptr) noexcept;
    void syncStreamForCleanup() noexcept;

protected:
    Runtime(Device device);

public:
    ~Runtime() noexcept;

    Runtime *activate();

    Device device() const;
    infini::rt::runtime::Stream stream() const;

    void syncStream();
    void syncDevice();
    void trimMemory();

    std::shared_ptr<Memory> allocateMemory(size_t size);
    std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);
    std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob);
    void retainGraphMemory(const std::shared_ptr<Memory> &memory);

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
    graph::GraphManager::CaptureState graphCaptureState() const;
    bool isGraphRecording() const;
    void startGraphRecording();
    void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
    std::shared_ptr<graph::Graph> stopGraphRecording();
    void cancelGraphRecording() noexcept;

    std::string toString() const;

    friend class ContextImpl;
    friend class graph::Graph;
};
} // namespace infinicore
