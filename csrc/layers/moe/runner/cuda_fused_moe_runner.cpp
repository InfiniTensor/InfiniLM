#include "cuda_fused_moe_runner.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_align.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::layers::moe {

CudaFusedMoeRunner::CudaFusedMoeRunner(size_t num_local_experts,
                                       size_t hidden_size,
                                       size_t intermediate_size_per_partition,
                                       size_t align_block_size)
    : num_local_experts_(num_local_experts),
      hidden_size_(hidden_size),
      intermediate_size_per_partition_(intermediate_size_per_partition),
      align_block_size_(align_block_size) {}

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
            throw std::runtime_error("MoE runner workspace tensor was not initialized before graph capture");
        }
        tensor = infinicore::Tensor::empty(shape, dtype, device);
    }
}

std::string shape_to_string(const infinicore::Shape &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void check_packed_weight_tensor(const infinicore::Tensor &tensor,
                                const std::string &name,
                                const infinicore::Device &device,
                                const infinicore::DataType dtype,
                                const infinicore::Shape &shape) {
    if (!tensor) {
        throw std::runtime_error("MoE fused dense core requires " + name);
    }
    if (tensor->device().getType() != device.getType() || tensor->device().getIndex() != device.getIndex()) {
        throw std::runtime_error("MoE fused dense core requires packed weights on the hidden_states device");
    }
    if (tensor->dtype() != dtype) {
        throw std::runtime_error("MoE fused dense core requires packed weights to have the same dtype as hidden_states");
    }
    if (tensor->shape() != shape) {
        throw std::runtime_error(
            "MoE fused dense core packed weight shape mismatch for " + name + ": expected " + shape_to_string(shape) + ", got " + shape_to_string(tensor->shape()));
    }
}

} // namespace

CombineInput CudaFusedMoeRunner::run(const DispatchOutput &dispatch_output,
                                     const MoeWeights &weights,
                                     MoeWorkspace &workspace) const {
    auto runner_input = prepare_runner_input(
        dispatch_output,
        workspace);

    auto runner_output = run_fused_core(runner_input, weights, workspace);

    return CombineInput{
        CombineInputFormat::Standard,
        runner_output.hidden_states,
        runner_input.topk_output,
        runner_input.routing_metadata,
    };
}

CudaFusedMoeRunnerInput CudaFusedMoeRunner::prepare_runner_input(const DispatchOutput &dispatch_output,
                                                                 MoeWorkspace &workspace) const {
    const auto &topk_ids = dispatch_output.topk_output.topk_ids;
    const auto &topk_shape = topk_ids->shape();
    if (topk_shape.size() != 2) {
        throw std::runtime_error("MoE runner requires topk_ids to be a 2D tensor");
    }
    const size_t num_pairs = topk_shape[0] * topk_shape[1];
    const size_t block_size = align_block_size_;
    const size_t align_num_experts = num_local_experts_ + 1;
    const size_t max_num_tokens_padded = num_pairs < align_num_experts
                                           ? num_pairs * block_size
                                           : num_pairs + align_num_experts * (block_size - 1);
    const size_t sorted_token_ids_capacity = ((max_num_tokens_padded + 3) / 4) * 4;
    const size_t max_num_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    const auto device = topk_ids->device();

    if (!workspace.sorted_token_ids || workspace.sorted_token_ids_capacity < sorted_token_ids_capacity) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE sorted_token_ids workspace was not initialized before graph capture");
        }
        workspace.sorted_token_ids = infinicore::Tensor::empty(
            {sorted_token_ids_capacity}, infinicore::DataType::I32, device);
        workspace.sorted_token_ids_capacity = sorted_token_ids_capacity;
    }
    if (!workspace.expert_ids || workspace.expert_ids_capacity < max_num_blocks) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE expert_ids workspace was not initialized before graph capture");
        }
        workspace.expert_ids = infinicore::Tensor::empty(
            {max_num_blocks}, infinicore::DataType::I32, device);
        workspace.expert_ids_capacity = max_num_blocks;
    }
    if (!workspace.num_tokens_post_padded) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE num_tokens_post_padded workspace was not initialized before graph capture");
        }
        workspace.num_tokens_post_padded = infinicore::Tensor::empty(
            {1}, infinicore::DataType::I32, device);
    }

    if (dispatch_output.expert_map) {
        infinicore::op::moe_align_with_expert_map_(
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            dispatch_output.expert_map,
            num_local_experts_,
            block_size,
            true);
    } else {
        infinicore::op::moe_align_(
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            num_local_experts_,
            block_size,
            true);
    }
    return CudaFusedMoeRunnerInput{
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
        MoeRoutingMetadata{
            workspace.sorted_token_ids,
            workspace.expert_ids,
            workspace.num_tokens_post_padded,
        },
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_fused_core(const CudaFusedMoeRunnerInput &runner_input,
                                                            const MoeWeights &weights,
                                                            MoeWorkspace &workspace) const {
    if (!weights.has_packed_dense_weights()) {
        throw std::runtime_error("MoE fused dense runner requires load-time packed w13/w2 weights");
    }
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_, intermediate_size_per_partition_ * 2, hidden_size_});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_, hidden_size_, intermediate_size_per_partition_});
    ensure_tensor(
        workspace.fused_moe_output,
        runner_input.hidden_states->shape(),
        runner_input.hidden_states->dtype(),
        runner_input.hidden_states->device());
    workspace.fused_moe_output_tokens_capacity = runner_input.hidden_states->shape()[0];
    infinicore::op::moe_fused_dense_(
        workspace.fused_moe_output,
        runner_input.hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        runner_input.topk_output.topk_weights,
        runner_input.topk_output.topk_ids,
        runner_input.routing_metadata.sorted_token_ids,
        runner_input.routing_metadata.expert_ids,
        runner_input.routing_metadata.num_tokens_post_padded);
    return CudaFusedMoeRunnerOutput{
        workspace.fused_moe_output,
    };
}

} // namespace infinilm::layers::moe
