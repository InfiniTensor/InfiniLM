#include "allgather_reduce_scatter_dispatcher.hpp"

#include "runtime/runtime.hpp"
#include "../../../engine/distributed/communication_group.hpp"

#include <stdexcept>
#include <utility>

namespace infinilm::layers::moe {
namespace {

bool same_device(const infinilm::runtime::Tensor &tensor, const infinilm::runtime::Device &device) {
    return tensor && tensor->device().type() == device.type() && tensor->device().index() == device.index();
}

void ensure_tensor(infinilm::runtime::Runtime &runtime,
                   infinilm::runtime::Tensor &tensor,
                   const infinilm::runtime::Shape &shape,
                   infinilm::runtime::DataType dtype,
                   const infinilm::runtime::Device &device) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) {
        throw std::invalid_argument("MoE AG/RS workspace device differs from Runtime device");
    }
    if (!same_device(tensor, device) || tensor->dtype() != dtype || tensor->shape() != shape) {
        if (runtime.isGraphRecording()) {
            throw std::runtime_error("MoE AG/RS workspace tensor was not initialized before graph capture");
        }
        tensor = infinilm::runtime::Tensor::empty(runtime, shape, dtype);
    }
}

} // namespace

AllGatherReduceScatterDispatcher::AllGatherReduceScatterDispatcher(EPConfig ep_config,
                                                                   size_t num_experts,
                                                                   const engine::distributed::RankInfo &rank_info)
    : BaseEPDispatcher(std::move(ep_config), num_experts, rank_info) {}

void AllGatherReduceScatterDispatcher::all_gather_dim0_many_(
    infinilm::runtime::Runtime &runtime,
    const std::vector<infinilm::runtime::Tensor> &outputs,
    const std::vector<infinilm::runtime::Tensor> &inputs) const {
    runtime.assertOwnerThread();
    if (inputs.empty()) {
        return;
    }
    if (outputs.size() != inputs.size() || !inputs.front() || inputs.front()->ndim() == 0) {
        throw std::invalid_argument("MoE AG/RS all_gather_many requires matching non-null tensors");
    }
    const auto local_dim0 = inputs.front()->shape()[0];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto &input = inputs[i];
        const auto &output = outputs[i];
        if (!input || !output || input->device() != runtime.device() ||
            output->device() != runtime.device()) {
            throw std::invalid_argument("MoE AG/RS all_gather_many tensor device differs from Runtime device");
        }
        if (input->ndim() == 0) {
            throw std::runtime_error("MoE AG/RS all_gather_many expects tensors with dim 0");
        }
        if (input->shape()[0] != local_dim0) {
            throw std::runtime_error("MoE AG/RS all_gather_many expects all tensors to share dim 0");
        }
    }
    infinilm::engine::distributed::allgatherv_many_(
        runtime,
        outputs,
        inputs,
        equal_split_sizes(local_dim0),
        communicator_);
}

void AllGatherReduceScatterDispatcher::reduce_scatter_dim0_(
    infinilm::runtime::Runtime &runtime,
    infinilm::runtime::Tensor output,
    const infinilm::runtime::Tensor &input) const {
    runtime.assertOwnerThread();
    if (!input || !output) {
        return;
    }
    if (input->device() != runtime.device() || output->device() != runtime.device()) {
        throw std::invalid_argument("MoE AG/RS reduce_scatter tensor device differs from Runtime device");
    }
    if (input->ndim() == 0 || input->shape()[0] % config_.ep_size != 0) {
        throw std::runtime_error("MoE AG/RS reduce_scatter expects dim 0 to be divisible by ep_size");
    }
    const size_t local_dim0 = input->shape()[0] / config_.ep_size;
    infinilm::engine::distributed::reduce_scatterv_(
        runtime,
        output,
        input,
        equal_split_sizes(local_dim0),
        infinicclSum,
        communicator_);
}

DispatchOutput AllGatherReduceScatterDispatcher::dispatch(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states,
    const TopKOutput &topk_output,
    MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (hidden_states && runtime.device() != hidden_states->device()) {
        throw std::invalid_argument("MoE AG/RS input device differs from Runtime device");
    }
    if ((topk_output.topk_weights && runtime.device() != topk_output.topk_weights->device()) ||
        (topk_output.topk_ids && runtime.device() != topk_output.topk_ids->device())) {
        throw std::invalid_argument("MoE AG/RS routing device differs from Runtime device");
    }
    if (config_.ep_size == 1) {
        return DispatchOutput{
            DispatchOutputFormat::Standard,
            hidden_states,
            infinilm::runtime::Tensor(),
            topk_output,
            infinilm::runtime::Tensor(),
        };
    }

    if (!hidden_states || !topk_output.topk_weights || !topk_output.topk_ids) {
        throw std::invalid_argument("MoE AG/RS dispatch requires hidden states and routing tensors");
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

    ensure_tensor(runtime, workspace.ep_gathered_hidden_states, hidden_shape, hidden_states->dtype(), device);
    ensure_tensor(runtime, workspace.ep_gathered_topk_weights, topk_weights_shape, topk_output.topk_weights->dtype(), device);
    ensure_tensor(runtime, workspace.ep_gathered_topk_ids, topk_ids_shape, topk_output.topk_ids->dtype(), device);
    workspace.ep_gathered_tokens_capacity = global_tokens;

    all_gather_dim0_many_(
        runtime,
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
        infinilm::runtime::Tensor(),
        dispatched_topk,
        expert_map(runtime),
    };
}

infinilm::runtime::Tensor AllGatherReduceScatterDispatcher::combine(
    infinilm::runtime::Runtime &runtime,
    const CombineInput &combine_input,
    MoeWorkspace &workspace) const {
    assertInitializedRuntime(runtime);
    if (combine_input.hidden_states && runtime.device() != combine_input.hidden_states->device()) {
        throw std::invalid_argument("MoE AG/RS combine device differs from Runtime device");
    }
    if (config_.ep_size == 1) {
        return combine_input.hidden_states;
    }
    if (!combine_input.hidden_states || combine_input.hidden_states->ndim() == 0 || combine_input.hidden_states->shape()[0] % config_.ep_size != 0) {
        throw std::runtime_error("MoE AG/RS combine expects hidden_states dim 0 to be divisible by ep_size");
    }

    auto output_shape = combine_input.hidden_states->shape();
    output_shape[0] /= config_.ep_size;
    ensure_tensor(
        runtime,
        workspace.ep_reduced_hidden_states,
        output_shape,
        combine_input.hidden_states->dtype(),
        combine_input.hidden_states->device());
    workspace.ep_reduced_tokens_capacity = output_shape[0];

    reduce_scatter_dim0_(runtime, workspace.ep_reduced_hidden_states, combine_input.hidden_states);
    return workspace.ep_reduced_hidden_states;
}

} // namespace infinilm::layers::moe
