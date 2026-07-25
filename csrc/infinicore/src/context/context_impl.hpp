#pragma once
#include "infinicore/context/context.hpp"
#include "runtime/runtime.hpp"

#include <array>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infinicore {
class ContextImpl {
private:
    using ThreadRuntimes = std::unordered_map<std::thread::id, std::weak_ptr<Runtime>>;

    mutable std::mutex runtime_table_mutex_;
    // Runtimes are isolated by device and thread so their streams, allocators,
    // and graph state cannot race with another thread.
    std::array<std::vector<ThreadRuntimes>, static_cast<size_t>(Device::Type::kCount)> runtime_table_;
    // Active runtime for current thread. Can use "static thread local" because context is a process singleton.
    static thread_local std::shared_ptr<Runtime> current_runtime_;
    static thread_local std::shared_ptr<Runtime> graph_runtime_;

    std::shared_ptr<Runtime> getOrCreateRuntimeLocked(Device device, const std::thread::id &thread_id);

    template <Device::Type device_type>
    void initializeDeviceType();

protected:
    ContextImpl();

public:
    Runtime *getCurrentRuntime();

    void setDevice(Device);

    size_t getDeviceCount(Device::Type type);

    bool isGraphRecording();
    void startGraphRecording();
    void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
    std::shared_ptr<graph::Graph> stopGraphRecording();
    void cancelGraphRecording() noexcept;

    static ContextImpl &singleton();

    friend class Runtime;
};
} // namespace infinicore
