#include "cuda_fused_moe_runner.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/hygon_moe_marlin.hpp"
#include "infinicore/ops/moe_align.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace infinilm::layers::moe {

CudaFusedMoeRunner::CudaFusedMoeRunner(
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size_per_partition,
    size_t align_block_size)
    : num_local_experts_(num_local_experts),
      hidden_size_(hidden_size),
      intermediate_size_per_partition_(
          intermediate_size_per_partition),
      align_block_size_(align_block_size) {}

namespace {

bool same_device(
    const infinicore::Tensor &tensor,
    const infinicore::Device &device) {
    return tensor
        && tensor->device().getType() == device.getType()
        && tensor->device().getIndex() == device.getIndex();
}

void ensure_tensor(
    infinicore::Tensor &tensor,
    const infinicore::Shape &shape,
    infinicore::DataType dtype,
    const infinicore::Device &device) {
    if (!same_device(tensor, device)
        || tensor->dtype() != dtype
        || tensor->shape() != shape) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error(
                "MoE runner workspace tensor was not initialized before graph capture");
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

void check_packed_weight_tensor(
    const infinicore::Tensor &tensor,
    const std::string &name,
    const infinicore::Device &device,
    infinicore::DataType dtype,
    const infinicore::Shape &shape) {
    if (!tensor) {
        throw std::runtime_error(
            "MoE fused dense core requires " + name);
    }
    if (tensor->device().getType() != device.getType()
        || tensor->device().getIndex() != device.getIndex()) {
        throw std::runtime_error(
            "MoE fused dense core requires packed weights on the hidden_states device");
    }
    if (tensor->dtype() != dtype) {
        throw std::runtime_error(
            "MoE fused dense core packed tensor dtype mismatch for "
            + name);
    }
    if (tensor->shape() != shape) {
        throw std::runtime_error(
            "MoE fused dense core packed weight shape mismatch for "
            + name + ": expected " + shape_to_string(shape)
            + ", got " + shape_to_string(tensor->shape()));
    }
}

} // namespace

CombineInput CudaFusedMoeRunner::run(
    const DispatchOutput &dispatch_output,
    const MoeWeights &weights,
    MoeWorkspace &workspace) const {
    if (weights.is_hygon_w16a16_marlin()
        || weights.is_hygon_w8a8_marlin()) {
        const auto format = weights.is_hygon_w16a16_marlin()
                              ? infinicore::op::HygonMoeMarlinWeightFormat::W16A16
                              : infinicore::op::HygonMoeMarlinWeightFormat::W8A8;
        const infinicore::op::HygonMoeMarlinWeights marlin_weights{
            weights.packed_w13,
            weights.packed_w2,
            weights.packed_w13_scale,
            weights.packed_w2_scale,
            format,
        };
        const auto output = infinicore::op::hygon_moe_marlin_fused(
            dispatch_output.hidden_states,
            dispatch_output.topk_output.topk_weights,
            dispatch_output.topk_output.topk_ids,
            dispatch_output.expert_map,
            marlin_weights,
            workspace.hygon_marlin,
            num_local_experts_,
            hidden_size_,
            intermediate_size_per_partition_,
            align_block_size_);

        MoeRoutingMetadata routing_metadata;
        if (output.has_routing_metadata) {
            routing_metadata.sorted_token_ids = output.sorted_token_ids;
            routing_metadata.expert_ids = output.expert_ids;
            routing_metadata.num_tokens_post_padded = output.num_tokens_post_padded;
        }
        return CombineInput{
            CombineInputFormat::Standard,
            output.hidden_states,
            dispatch_output.topk_output,
            routing_metadata,
        };
    }

    auto runner_input = prepare_runner_input(
        dispatch_output,
        workspace,
        align_block_size_);
    auto runner_output = run_fused_core(runner_input, weights, workspace);
    return CombineInput{
        CombineInputFormat::Standard,
        runner_output.hidden_states,
        runner_input.topk_output,
        runner_input.routing_metadata,
    };
}

CudaFusedMoeRunnerInput CudaFusedMoeRunner::prepare_runner_input(
    const DispatchOutput &dispatch_output,
    MoeWorkspace &workspace,
    size_t block_size) const {
    const auto &topk_ids = dispatch_output.topk_output.topk_ids;
    const auto &topk_shape = topk_ids->shape();
    if (topk_shape.size() != 2) {
        throw std::runtime_error(
            "MoE runner requires topk_ids to be a 2D tensor");
    }
    const size_t num_pairs = topk_shape[0] * topk_shape[1];
    const size_t align_num_experts = num_local_experts_ + 1;
    const size_t max_num_tokens_padded = num_pairs < align_num_experts
                                           ? num_pairs * block_size
                                           : num_pairs
                                                 + align_num_experts * (block_size - 1);
    const size_t sorted_token_ids_capacity = ((max_num_tokens_padded + 3) / 4) * 4;
    const size_t max_num_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    const auto device = topk_ids->device();

    if (!workspace.sorted_token_ids
        || workspace.sorted_token_ids_capacity
               < sorted_token_ids_capacity) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error(
                "MoE sorted_token_ids workspace was not initialized before graph capture");
        }
        workspace.sorted_token_ids = infinicore::Tensor::empty(
            {sorted_token_ids_capacity},
            infinicore::DataType::I32,
            device);
        workspace.sorted_token_ids_capacity = sorted_token_ids_capacity;
    }
    if (!workspace.expert_ids
        || workspace.expert_ids_capacity < max_num_blocks) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error(
                "MoE expert_ids workspace was not initialized before graph capture");
        }
        workspace.expert_ids = infinicore::Tensor::empty(
            {max_num_blocks},
            infinicore::DataType::I32,
            device);
        workspace.expert_ids_capacity = max_num_blocks;
    }
    if (!workspace.num_tokens_post_padded) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error(
                "MoE num_tokens_post_padded workspace was not initialized before graph capture");
        }
        workspace.num_tokens_post_padded = infinicore::Tensor::empty(
            {1},
            infinicore::DataType::I32,
            device);
    }

    auto sorted_token_ids = workspace.sorted_token_ids->narrow(
        {{0, 0, sorted_token_ids_capacity}});
    auto expert_ids = workspace.expert_ids->narrow(
        {{0, 0, max_num_blocks}});
    if (dispatch_output.expert_map) {
        infinicore::op::moe_align_with_expert_map_(
            sorted_token_ids,
            expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            dispatch_output.expert_map,
            num_local_experts_,
            block_size,
            true);
    } else {
        infinicore::op::moe_align_(
            sorted_token_ids,
            expert_ids,
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
            sorted_token_ids,
            expert_ids,
            workspace.num_tokens_post_padded,
        },
    };
}

CudaFusedMoeRunnerOutput CudaFusedMoeRunner::run_fused_core(
    const CudaFusedMoeRunnerInput &runner_input,
    const MoeWeights &weights,
    MoeWorkspace &workspace) const {
    if (!weights.has_packed_dense_weights()) {
        throw std::runtime_error(
            "MoE fused dense runner requires load-time packed w13/w2 weights");
    }
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_,
         intermediate_size_per_partition_ * 2,
         hidden_size_});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        runner_input.hidden_states->device(),
        runner_input.hidden_states->dtype(),
        {num_local_experts_,
         hidden_size_,
         intermediate_size_per_partition_});
    ensure_tensor(
        workspace.fused_moe_output,
        runner_input.hidden_states->shape(),
        runner_input.hidden_states->dtype(),
        runner_input.hidden_states->device());
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
