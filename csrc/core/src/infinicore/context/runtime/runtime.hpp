#pragma once

#include "../allocators/pinnable_block_allocator.hpp"

#include "infinicore/context/context.hpp"

#include "../../graph/graph_manager.hpp"

#include <infinirt.h>

namespace infinicore {
class ContextImpl;
class Runtime {
private:
    Device device_;
    infinirtStream_t stream_;
    std::unique_ptr<PinnableBlockAllocator> device_memory_allocator_;
    std::unique_ptr<MemoryAllocator> pinned_host_memory_allocator_;
    std::unique_ptr<graph::GraphManager> graph_manager_;

protected:
    Runtime(Device device);

public:
    ~Runtime();

    Runtime *activate();

    Device device() const;
    infinirtStream_t stream() const;

    void syncStream();
    void syncDevice();

    std::shared_ptr<Memory> allocateMemory(size_t size);
    std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);
    std::shared_ptr<Memory> reinstantiateBlob(std::shared_ptr<Memory> blob);

    void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
    void memcpyD2H(void *dst, const void *src, size_t size);
    void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);

    // Timing methods
    infinirtEvent_t createEvent();
    infinirtEvent_t createEventWithFlags(uint32_t flags);
    void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
    bool queryEvent(infinirtEvent_t event);
    void synchronizeEvent(infinirtEvent_t event);
    void destroyEvent(infinirtEvent_t event);
    float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);
    void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

    // Graph
    bool isGraphRecording() const;
    void startGraphRecording();
    void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
    std::shared_ptr<graph::Graph> stopGraphRecording(const graph::GraphInstantiateFence &fence = nullptr);

    std::string toString() const;

    friend class ContextImpl;
};
} // namespace infinicore
