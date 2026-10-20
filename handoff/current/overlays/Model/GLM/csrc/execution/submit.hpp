#pragma once

#include "execution/graph/graph.hpp"
#include "runtime/runtime.hpp"
#include "execution/owner_affinity_r2392.hpp"
#include <infini/ops.h>
#include <infini/rt/device_guard.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace infinilm::execution {

enum class RecordingPolicy { AllowHostReplay,
                             DeviceOnly,
                             HostOnly };

namespace detail {

// Ops owns instance caching and the required size. Submission supplies storage
// whose Runtime lease releases it in stream order.
class RuntimeWorkspaceHandle final : public infini::ops::Handle {
public:
    explicit RuntimeWorkspaceHandle(infinilm::runtime::Runtime &runtime) : runtime_(runtime) {
        runtime_.assertOwnerThread();
        if (runtime_.device().type() == infinilm::runtime::Device::Type::kHygon)
            affinity_r2392::apply_once(runtime_.device().index());
        set_stream(runtime_.stream());
    }

    WorkspaceLease acquire_workspace(std::size_t bytes) const override {
        if (!bytes) {
            return {};
        }
        auto memory = runtime_.allocateMemory(bytes);
        return {std::shared_ptr<void>(memory, memory->data()), memory->size()};
    }

    // Cached Ops instances retain submission fields, not a Runtime callback.
    std::unique_ptr<infini::ops::Handle> Clone() const override {
        return std::make_unique<infini::ops::Handle>(*this);
    }

private:
    infinilm::runtime::Runtime &runtime_;
};

template <class T>
void check_owner(infinilm::runtime::Runtime &, const T &);

inline void check_owner(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &tensor) {
    if (!tensor || tensor->device() != runtime.device()) {
        throw std::invalid_argument("Ops submission requires tensors on the owning Runtime device");
    }
}
template <class T>
void check_owner(infinilm::runtime::Runtime &runtime, const std::optional<T> &tensor) {
    if (tensor) {
        check_owner(runtime, *tensor);
    }
}
template <class T>
void check_owner(infinilm::runtime::Runtime &runtime, const std::vector<T> &tensors) {
    for (const auto &tensor : tensors) {
        check_owner(runtime, tensor);
    }
}
template <class T>
void check_owner(infinilm::runtime::Runtime &runtime, const T &value) {
    if constexpr (std::is_base_of_v<infinilm::runtime::Tensor, T>) {
        check_owner(runtime, static_cast<const infinilm::runtime::Tensor &>(value));
    }
}

inline auto view(const infinilm::runtime::Tensor &tensor) {
    return infini::ops::Tensor(const_cast<std::byte *>(tensor->data()),
                               tensor->shape(), tensor->dtype(), tensor->device(),
                               tensor->strides());
}
inline auto view(const infinilm::execution::graph::GraphTensor &tensor) {
    return view(static_cast<const infinilm::runtime::Tensor &>(tensor));
}
template <class T>
auto view(const T &value) {
    if constexpr (std::is_base_of_v<infinilm::runtime::Tensor, T>) {
        return view(static_cast<const infinilm::runtime::Tensor &>(value));
    } else {
        return value;
    }
}
template <class Tensor>
auto view(const std::optional<Tensor> &tensor) {
    using View = decltype(view(*tensor));
    return tensor ? std::optional<View>{view(*tensor)} : std::optional<View>{};
}
template <class Tensor>
auto view(const std::vector<Tensor> &tensors) {
    using View = decltype(view(std::declval<const Tensor &>()));
    std::vector<View> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.push_back(view(tensor));
    }
    return result;
}

inline auto snapshot(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &tensor) {
    return infinilm::execution::graph::GraphTensor{runtime, tensor};
}
template <class T>
auto snapshot(infinilm::runtime::Runtime &runtime, const T &value) {
    if constexpr (std::is_base_of_v<infinilm::runtime::Tensor, T>) {
        return snapshot(runtime, static_cast<const infinilm::runtime::Tensor &>(value));
    } else {
        return value;
    }
}
template <class T>
auto snapshot(infinilm::runtime::Runtime &runtime, const std::optional<T> &tensor) {
    using Snapshot = decltype(snapshot(runtime, *tensor));
    return tensor ? std::optional<Snapshot>{snapshot(runtime, *tensor)} : std::optional<Snapshot>{};
}
template <class T>
auto snapshot(infinilm::runtime::Runtime &runtime, const std::vector<T> &tensors) {
    using Snapshot = decltype(snapshot(runtime, std::declval<const T &>()));
    std::vector<Snapshot> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.push_back(snapshot(runtime, tensor));
    }
    return result;
}

} // namespace detail

// This boundary owns submission resources only. Callers allocate outputs and
// supply the exact public Ops arguments/configuration; Ops owns computation.
template <class OpsKey, class... Args>
void submit(infinilm::runtime::Runtime &runtime, RecordingPolicy policy,
            const infini::ops::Config &config,
            const Args &...args) {
    runtime.assertOwnerThread();
    (detail::check_owner(runtime, args), ...);
    infini::rt::DeviceGuard guard{runtime.device().type(), runtime.device().index(), true};
    const auto views = std::make_tuple(detail::view(args)...);
    if (!runtime.isGraphRecording()) {
        detail::RuntimeWorkspaceHandle handle{runtime};
        std::apply([&](const auto &...values) { OpsKey::Call(handle, config, values...); }, views);
        return;
    }

    auto instance = std::shared_ptr<infini::ops::Operator<OpsKey>>{
        std::apply([&](const auto &...values) { return OpsKey::Make(config, values...); }, views)};
    const auto bytes = instance->workspace_size_in_bytes();
    auto workspace = bytes ? runtime.allocateMemory(bytes) : std::shared_ptr<infinilm::runtime::Memory>{};
    auto recorded = std::make_tuple(detail::snapshot(runtime, args)...);
    const bool capture_safe = policy != RecordingPolicy::HostOnly && instance->capture_safe();
    runtime.addGraphOperator(std::make_shared<infinilm::execution::graph::RecordedCall>(
        [recorded = std::move(recorded), instance = std::move(instance),
         workspace = std::move(workspace), owner = runtime.shared_from_this()]() mutable {
            owner->assertOwnerThread();
            infini::rt::DeviceGuard guard{owner->device().type(), owner->device().index(), true};
            infini::ops::Handle handle;
            handle.set_stream(owner->stream());
            if (workspace) {
                handle.set_workspace(workspace->data());
                handle.set_workspace_size_in_bytes(workspace->size());
            }
            const auto values = std::apply([](const auto &...args) {
                return std::make_tuple(detail::view(args)...);
            },
                                           recorded);
            std::apply([&](const auto &...args) { (*instance)(handle, args...); }, values);
        },
        capture_safe, policy == RecordingPolicy::AllowHostReplay || policy == RecordingPolicy::HostOnly));
}

template <class OpsKey, class... Args>
void submit(infinilm::runtime::Runtime &runtime, const infini::ops::Config &config,
            const Args &...args) {
    submit<OpsKey>(runtime, RecordingPolicy::AllowHostReplay, config, args...);
}

} // namespace infinilm::execution
