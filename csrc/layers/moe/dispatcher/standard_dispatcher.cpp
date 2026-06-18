#include "standard_dispatcher.hpp"

#include "../../../global_state/parallel_state.hpp"

#include "infinicore/ops/distributed/allreduce.hpp"

#include <utility>

namespace infinilm::layers::moe {

StandardDispatcher::StandardDispatcher() {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
}

DispatchOutput StandardDispatcher::dispatch(const infinicore::Tensor &hidden_states,
                                            const TopKOutput &topk_output,
                                            MoeWorkspace &workspace) const {
    (void)workspace;
    return DispatchOutput{
        DispatchOutputFormat::Standard,
        hidden_states,
        infinicore::Tensor(),
        topk_output,
    };
}

infinicore::Tensor StandardDispatcher::combine(const CombineInput &combine_input,
                                               MoeWorkspace &workspace) const {
    (void)workspace;
    if (tp_size_ > 1 && communicator_ != nullptr) {
        infinicore::op::distributed::allreduce_(
            combine_input.hidden_states,
            combine_input.hidden_states,
            INFINICCL_SUM,
            communicator_);
    }
    return combine_input.hidden_states;
}

} // namespace infinilm::layers::moe
