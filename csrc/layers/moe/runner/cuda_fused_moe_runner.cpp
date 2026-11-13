#include "cuda_fused_moe_runner.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_align.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
    return tensor && tensor->device().type() == device.type() && tensor->device().index() == device.index();
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
    if (tensor->device().type() != device.type() || tensor->device().index() != device.index()) {
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
    // dl fused MoE materializes [M, topk, H] intermediates. A 7k-token
    // prefill OOMs a 32GB KS38; vLLM chunks, but its 16k chunk is still too
    // big here. Run ≤1024-token chunks on QY (graph capture stays one shot).
    constexpr size_t kQyChunkTokens = 1024;
    const auto &hidden = dispatch_output.hidden_states;
    const size_t num_tokens = hidden ? hidden->shape()[0] : 0;
    const bool chunk_qy = hidden
                       && hidden->device().type() == infinicore::Device::Type::kQy
                       && !infinicore::context::isGraphRecording()
                       && num_tokens > kQyChunkTokens
                       && hidden->shape().size() == 2;
    if (!chunk_qy) {
        auto runner_input = prepare_runner_input(dispatch_output, workspace);
        auto runner_output = run_fused_core(runner_input, weights, workspace);
        return CombineInput{
            CombineInputFormat::Standard,
            runner_output.hidden_states,
            runner_input.topk_output,
            runner_input.routing_metadata,
        };
    }

    ensure_tensor(
        workspace.fused_moe_output,
        hidden->shape(),
        hidden->dtype(),
        hidden->device());
    workspace.fused_moe_output_tokens_capacity = num_tokens;
    const auto &topk = dispatch_output.topk_output;
    for (size_t begin = 0; begin < num_tokens; begin += kQyChunkTokens) {
        const size_t rows = std::min(kQyChunkTokens, num_tokens - begin);
        DispatchOutput slice = dispatch_output;
        slice.hidden_states = hidden->narrow({{0, begin, rows}})->contiguous();
        slice.topk_output.topk_weights = topk.topk_weights->narrow({{0, begin, rows}})->contiguous();
        slice.topk_output.topk_ids = topk.topk_ids->narrow({{0, begin, rows}})->contiguous();
        // Fresh align buffers so block_size recovered from numel matches this chunk.
        MoeWorkspace chunk_workspace;
        auto runner_input = prepare_runner_input(slice, chunk_workspace);
        auto runner_output = run_fused_core(runner_input, weights, chunk_workspace);
        workspace.fused_moe_output->narrow({{0, begin, rows}})->copy_from(runner_output.hidden_states);
    }

    return CombineInput{
        CombineInputFormat::Standard,
        workspace.fused_moe_output,
        dispatch_output.topk_output,
        MoeRoutingMetadata{},
    };
}

CudaFusedMoeRunnerInput CudaFusedMoeRunner::prepare_runner_input(const DispatchOutput &dispatch_output,
                                                                 MoeWorkspace &workspace) const {
    const auto &topk_ids = dispatch_output.topk_output.topk_ids;
    const auto &topk_shape = topk_ids->shape();
    if (topk_shape.size() != 2) {
        throw std::runtime_error("MoE runner requires topk_ids to be a 2D tensor");
    }
    const size_t num_tokens = topk_shape[0];
    const size_t topk = topk_shape[1];
    const size_t num_pairs = num_tokens * topk;
    const size_t block_size = align_block_size_;
    const size_t num_experts = num_local_experts_;
    // Match vLLM moe_align_block_size Denglin (is_dl) capacity — NOT the
    // generic num_pairs + (E+1)*(block-1) bound. Oversized sorted/expert
    // buffers with BLOCK_SIZE_M=64 trip Denglin tu_program.cc:625 / SIGSEGV
    // even when align block_size matches tiles (repro: 18240/285 OK, 18336/287
    // crash for M=512,E=160,block=64).
    size_t max_num_tokens_padded = 0;
    size_t max_num_blocks = 0;
    if (block_size > 0) {
        // vLLM DL: min(numel, (numel - E) // B + E); use unsigned-safe form.
        const size_t dl_blocks = num_pairs > num_experts
                                   ? std::min(num_pairs,
                                              (num_pairs - num_experts) / block_size + num_experts)
                                   : num_pairs;
        max_num_blocks = dl_blocks;
        max_num_tokens_padded = dl_blocks * block_size;
    }
    const size_t sorted_token_ids_capacity = max_num_tokens_padded;
    const auto device = topk_ids->device();

    // Match vLLM fused_experts_impl use_moe_cu (unquantized): when
    // M*top_k/E <= 16, skip moe_align_ and pass three size-1 int32 metadata
    // tensors. That selects _invoke_fused_moe_bf16 / warpnk. Real align
    // buffers (numel>1) select take_/store_. Prefill M=512 stays on align.
    // Size-1 tensors are allocated before graph capture (ensure_tensor) and
    // kept on separate workspace fields so decode FULL stays device_segments=1.
    const bool use_moe_cu = num_experts > 0 && (static_cast<double>(num_tokens) * static_cast<double>(topk) / static_cast<double>(num_experts)) <= 16.0;

    {
        static bool logged_align = false;
        if (!logged_align) {
            logged_align = true;
            std::cerr << "CudaFusedMoeRunner moe_align block_size=" << block_size
                      << " num_local_experts=" << num_local_experts_
                      << " num_tokens=" << num_tokens << " topk=" << topk
                      << " use_moe_cu=" << (use_moe_cu ? 1 : 0)
                      << " sorted_cap=" << sorted_token_ids_capacity
                      << " expert_cap=" << max_num_blocks << std::endl;
        }
    }

    if (use_moe_cu) {
        ensure_tensor(workspace.moe_cu_sorted_token_ids,
                      {1},
                      infinicore::DataType::kInt32,
                      device);
        ensure_tensor(workspace.moe_cu_expert_ids,
                      {1},
                      infinicore::DataType::kInt32,
                      device);
        ensure_tensor(workspace.moe_cu_num_tokens_post_padded,
                      {1},
                      infinicore::DataType::kInt32,
                      device);
        return CudaFusedMoeRunnerInput{
            dispatch_output.hidden_states,
            dispatch_output.topk_output,
            MoeRoutingMetadata{
                workspace.moe_cu_sorted_token_ids,
                workspace.moe_cu_expert_ids,
                workspace.moe_cu_num_tokens_post_padded,
            },
        };
    }

    // Align block_size must equal moe_fused_dense block_size_m (TU segfault otherwise).
    if (!workspace.sorted_token_ids || workspace.sorted_token_ids_capacity < sorted_token_ids_capacity) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE sorted_token_ids workspace was not initialized before graph capture");
        }
        workspace.sorted_token_ids = infinicore::Tensor::empty(
            {sorted_token_ids_capacity}, infinicore::DataType::kInt32, device);
        workspace.sorted_token_ids_capacity = sorted_token_ids_capacity;
    }
    if (!workspace.expert_ids || workspace.expert_ids_capacity < max_num_blocks) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE expert_ids workspace was not initialized before graph capture");
        }
        workspace.expert_ids = infinicore::Tensor::empty(
            {max_num_blocks}, infinicore::DataType::kInt32, device);
        workspace.expert_ids_capacity = max_num_blocks;
    }
    if (!workspace.num_tokens_post_padded) {
        if (infinicore::context::isGraphRecording()) {
            throw std::runtime_error("MoE num_tokens_post_padded workspace was not initialized before graph capture");
        }
        workspace.num_tokens_post_padded = infinicore::Tensor::empty(
            {1}, infinicore::DataType::kInt32, device);
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
