#pragma once
#include "runtime/context.hpp"
#include "runtime/runtime.hpp"

#include <array>
#include <mutex>
#include <vector>

namespace infinilm::runtime {
class ContextImpl {
private:
    mutable std::mutex runtime_table_mutex_;
    // Device counts are process-wide; Runtime weak slots belong to the worker
    // itself and disappear on thread exit, even if its thread ID is reused.
    std::array<size_t, static_cast<size_t>(Device::Type::kCount)> device_counts_{};
    std::array<bool, static_cast<size_t>(Device::Type::kCount)> device_types_initialized_{};
    static thread_local std::array<std::vector<std::weak_ptr<Runtime>>,
                                   static_cast<size_t>(Device::Type::kCount)>
        thread_runtimes_;
    // Active runtime for current thread. Can use "static thread local" because context is a process singleton.
    static thread_local std::shared_ptr<Runtime> current_runtime_;

    template <Device::Type device_type>
    void initializeDeviceType();
    void initializeDeviceTypeLocked(Device::Type type);

protected:
    ContextImpl() = default;

public:
    Runtime *getCurrentRuntime();
    std::shared_ptr<Runtime> getCurrentRuntimeShared();

    std::shared_ptr<Runtime> setDevice(Device);
    std::shared_ptr<Runtime> resolveRuntime(Device);

    size_t getDeviceCount(Device::Type type);

    static ContextImpl &singleton();
};
} // namespace infinilm::runtime
