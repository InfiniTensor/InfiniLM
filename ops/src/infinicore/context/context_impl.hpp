#pragma once
#include "infinicore/context/context.hpp"
#include "runtime/runtime.hpp"

#include <array>
#include <vector>

namespace infinicore {
class ContextImpl {
private:
    // Table of runtimes for every device (type and index)
    std::array<std::vector<std::unique_ptr<Runtime>>, size_t(Device::Type::COUNT)> runtime_table_;
    // Active runtime for current thread. Can use "static thread local" because context is a process singleton.
    static thread_local Runtime *current_runtime_;

protected:
    ContextImpl();

public:
    Runtime *getCurrentRuntime();

    void setDevice(Device);

    size_t getDeviceCount(Device::Type type);

    static ContextImpl &singleton();

    friend class Runtime;
};
} // namespace infinicore
