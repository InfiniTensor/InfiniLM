#include "standard_dispatcher.hpp"

#include "runtime/runtime.hpp"

#include "../../../engine/distributed/communication_group.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {

StandardDispatcher::StandardDispatcher(const engine::distributed::RankInfo &rank_info)
    : device_(rank_info.device) {
    if (static_cast<int>(device_.type()) < 0
        || static_cast<int>(device_.type()) >= static_cast<int>(infinilm::runtime::Device::Type::kCount)
        || device_.index() < 0
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("MoE standard requires valid rank topology and communicator");
    }
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
}

void StandardDispatcher::initialize(infinilm::runtime::Runtime &runtime, MoeWorkspace &workspace) {
    (void)workspace;
    runtime.assertOwnerThread();
    if (runtime.device() != device_ || (runtime_owner_ && runtime_owner_.get() != &runtime)) {
        throw std::invalid_argument("MoE standard initialization Runtime differs from rank owner");
    }
    runtime_owner_ = runtime.shared_from_this();
}

void StandardDispatcher::assertInitializedRuntime(infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime_owner_.get() != &runtime || runtime.device() != device_) {
        throw std::invalid_argument("MoE standard Runtime differs from initialized resource owner");
    }
}

DispatchOutput StandardDispatcher::dispatch(infinilm::runtime::Runtime &runtime,
                                            const infinilm::runtime::Tensor &hidden_states,
                                            const TopKOutput &topk_output,
                                            MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (hidden_states && runtime.device() != hidden_states->device()) {
        throw std::invalid_argument("MoE standard input device differs from Runtime device");
    }
    if ((topk_output.topk_weights && runtime.device() != topk_output.topk_weights->device()) ||
        (topk_output.topk_ids && runtime.device() != topk_output.topk_ids->device())) {
        throw std::invalid_argument("MoE standard routing device differs from Runtime device");
    }
    (void)workspace;
    return DispatchOutput{
        DispatchOutputFormat::Standard,
        hidden_states,
        infinilm::runtime::Tensor(),
        topk_output,
    };
}

infinilm::runtime::Tensor StandardDispatcher::combine(infinilm::runtime::Runtime &runtime,
                                               const CombineInput &combine_input,
                                               MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (combine_input.hidden_states && runtime.device() != combine_input.hidden_states->device()) {
        throw std::invalid_argument("MoE standard combine device differs from Runtime device");
    }
    (void)workspace;
    if (tp_size_ > 1 && combine_input.hidden_states) {
        infinilm::engine::distributed::allreduce_(
            runtime,
            combine_input.hidden_states,
            combine_input.hidden_states,
            infinicclSum,
            communicator_);
    }
    return combine_input.hidden_states;
}

} // namespace infinilm::layers::moe
