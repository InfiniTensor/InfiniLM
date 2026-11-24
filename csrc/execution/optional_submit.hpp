#pragma once

#include "execution/submit.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>

namespace infinilm::execution {

template <class Op, class = void>
struct HasSdkFactory : std::false_type {};

template <class Op>
struct HasSdkFactory<Op, std::void_t<decltype(sizeof(infini::ops::detail::OperatorFactory<Op>))>>
    : std::true_type {};

template <class Op>
bool has_sdk_implementation(infini::ops::Device::Type device) {
    if constexpr (HasSdkFactory<Op>::value) {
        return !Op::active_implementation_indices(device).empty();
    } else {
        return false;
    }
}

// Guard configuration lookup as well as submission: evaluating DefaultConfig
// outside this branch can reference a provider absent from the installed SDK.
template <class Op>
infini::ops::Config optional_default_config(const char *name, infini::ops::Device::Type device) {
    if constexpr (HasSdkFactory<Op>::value) {
        return Op::DefaultConfig(device);
    } else {
        throw std::runtime_error(std::string("Installed InfiniOps SDK has no call bindings for `") + name + "`.");
    }
}

// Only optional model features use this entry. Missing SDK bindings never
// substitute a different algorithm or silently drop an operation.
template <class Op, class... Args>
void submit_optional(const char *name, infinilm::runtime::Runtime &runtime,
                     RecordingPolicy policy, const infini::ops::Config &config,
                     const Args &...args) {
    runtime.assertOwnerThread();
    if constexpr (HasSdkFactory<Op>::value) {
        submit<Op>(runtime, policy, config, args...);
    } else {
        throw std::runtime_error(std::string("Installed InfiniOps SDK has no call bindings for `") + name + "`.");
    }
}

} // namespace infinilm::execution
