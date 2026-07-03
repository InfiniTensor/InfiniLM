#include "allgather_reduce_scatter_dispatcher.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/distributed/reduce_scatter.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {
namespace {

bool same_device(const infinicore::Tensor &tensor, const infinicore::Device &device) {
    return tensor && tensor->device().getType() == device.getType() && tensor->device().getIndex() == device.getIndex();
}

void ensure_tensor(infinicore::Tensor &tensor,
                   const infinicore::Shape &shape,
                   infinicore::DataType dtype,
                   const infinicore::Device &device) {
    if (!same_device(tensor, device) || tensor->dtype() != dtype || tensor->shape() != shape) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE AG/RS workspace tensor was not initialized before graph capture");
        }
        tensor = infinicore::Tensor::empty(shape, dtype, device);
    }
}

} // namespace

AllGatherReduceScatterDispatcher::AllGatherReduceScatterDispatcher(EPConfig ep_config,
                                                                   size_t num_experts)
    : BaseEPDispatcher(std::move(ep_config), num_experts) {}

void AllGatherReduceScatterDispatcher::all_gather_dim0_many_(
    const std::vector<infinicore::Tensor> &outputs,
    const std::vector<infinicore::Tensor> &inputs) const {
    if (inputs.empty()) {
        return;
    }
    const auto local_dim0 = inputs.front()->shape()[0];
    for (const auto &input : inputs) {
        if (!input) {
            throw std::runtime_error("MoE AG/RS all_gather_many does not support null tensors");
        }
        if (input->ndim() == 0) {
            throw std::runtime_error("MoE AG/RS all_gather_many expects tensors with dim 0");
        }
        if (input->shape()[0] != local_dim0) {
            throw std::runtime_error("MoE AG/RS all_gather_many expects all tensors to share dim 0");
        }
    }
    infinicore::op::distributed::allgatherv_many_(
        outputs,
        inputs,
        equal_split_sizes(local_dim0),
        communicator_);
}

void AllGatherReduceScatterDispatcher::reduce_scatter_dim0_(
    infinicore::Tensor output,
    const infinicore::Tensor &input) const {
    if (!input || !output) {
        return;
    }
    if (input->ndim() == 0 || input->shape()[0] % config_.ep_size != 0) {
        throw std::runtime_error("MoE AG/RS reduce_scatter expects dim 0 to be divisible by ep_size");
    }
    const size_t local_dim0 = input->shape()[0] / config_.ep_size;
    infinicore::op::distributed::reduce_scatterv_(
        output,
        input,
        equal_split_sizes(local_dim0),
        INFINICCL_SUM,
        communicator_);
}

DispatchOutput AllGatherReduceScatterDispatcher::dispatch(
    const infinicore::Tensor &hidden_states,
    const TopKOutput &topk_output,
    MoeWorkspace &workspace) const {
    if (config_.ep_size == 1) {
        return DispatchOutput{
            DispatchOutputFormat::Standard,
            hidden_states,
            infinicore::Tensor(),
            topk_output,
            infinicore::Tensor(),
        };
    }

    const size_t local_tokens = hidden_states->shape()[0];
    const size_t global_tokens = local_tokens * config_.ep_size;
    const auto device = hidden_states->device();

    auto hidden_shape = hidden_states->shape();
    hidden_shape[0] = global_tokens;
    auto topk_weights_shape = topk_output.topk_weights->shape();
    topk_weights_shape[0] = global_tokens;
    auto topk_ids_shape = topk_output.topk_ids->shape();
    topk_ids_shape[0] = global_tokens;

    ensure_tensor(workspace.ep_gathered_hidden_states, hidden_shape, hidden_states->dtype(), device);
    ensure_tensor(workspace.ep_gathered_topk_weights, topk_weights_shape, topk_output.topk_weights->dtype(), device);
    ensure_tensor(workspace.ep_gathered_topk_ids, topk_ids_shape, topk_output.topk_ids->dtype(), device);
    workspace.ep_gathered_tokens_capacity = global_tokens;

    all_gather_dim0_many_(
        {
            workspace.ep_gathered_hidden_states,
            workspace.ep_gathered_topk_weights,
            workspace.ep_gathered_topk_ids,
        },
        {
            hidden_states,
            topk_output.topk_weights,
            topk_output.topk_ids,
        });

    TopKOutput dispatched_topk{
        workspace.ep_gathered_topk_weights,
        workspace.ep_gathered_topk_ids,
        topk_output.router_logits,
    };
    return DispatchOutput{
        DispatchOutputFormat::Standard,
        workspace.ep_gathered_hidden_states,
        infinicore::Tensor(),
        dispatched_topk,
        expert_map(hidden_states->device()),
    };
}

infinicore::Tensor AllGatherReduceScatterDispatcher::combine(
    const CombineInput &combine_input,
    MoeWorkspace &workspace) const {
    if (config_.ep_size == 1) {
        return combine_input.hidden_states;
    }
    if (!combine_input.hidden_states || combine_input.hidden_states->ndim() == 0 || combine_input.hidden_states->shape()[0] % config_.ep_size != 0) {
        throw std::runtime_error("MoE AG/RS combine expects hidden_states dim 0 to be divisible by ep_size");
    }

    auto output_shape = combine_input.hidden_states->shape();
    output_shape[0] /= config_.ep_size;
    ensure_tensor(
        workspace.ep_reduced_hidden_states,
        output_shape,
        combine_input.hidden_states->dtype(),
        combine_input.hidden_states->device());
    workspace.ep_reduced_tokens_capacity = output_shape[0];

    reduce_scatter_dim0_(workspace.ep_reduced_hidden_states, combine_input.hidden_states);
    return workspace.ep_reduced_hidden_states;
}

} // namespace infinilm::layers::moe
