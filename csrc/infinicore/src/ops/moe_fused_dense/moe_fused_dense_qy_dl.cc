#if defined(ENABLE_ATEN) && defined(ENABLE_QY_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/denglin_fused_moe_adaptor.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"

#include <stdexcept>
#include <vector>

namespace infinicore::op::moe_fused_dense_impl::qy_dl {

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
};

void *plan(Tensor output,
           const Tensor &hidden_states,
           const Tensor &w13,
           const Tensor &w2,
           const Tensor &topk_weights,
           const Tensor &topk_ids,
           const Tensor &sorted_token_ids,
           const Tensor &expert_ids,
           const Tensor &num_tokens_post_padded) {
    if (!adaptor::denglin_fused_moe::available()) {
        throw std::runtime_error(
            "QY moe_fused_dense requires vllm._dl_C (set INFINICORE_DENGLIN_DL_C_SO if needed)");
    }
    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13),
        graph::GraphTensor(w2),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    adaptor::set_aten_stream_to_infinicore();

    auto hidden = adaptor::to_aten_tensor(p->hidden_states);
    auto w13 = adaptor::to_aten_tensor(p->w13);
    auto w2 = adaptor::to_aten_tensor(p->w2);
    auto topk_weights = adaptor::to_aten_tensor(p->topk_weights);
    auto topk_ids = adaptor::to_aten_tensor(p->topk_ids);
    auto sorted_token_ids = adaptor::to_aten_tensor(p->sorted_token_ids);
    auto expert_ids = adaptor::to_aten_tensor(p->expert_ids);
    auto num_tokens_post_padded = adaptor::to_aten_tensor(p->num_tokens_post_padded);
    auto output = adaptor::to_aten_tensor(p->output);

    const int64_t m = hidden.size(0);
    const int64_t h = hidden.size(1);
    const int64_t topk = topk_ids.size(1);
    const int64_t inter2 = w13.size(1);
    const int64_t inter = inter2 / 2;

    // Workspace capacity rounding makes sorted_cap/expert_cap recovery inexact
    // (e.g. 17 instead of 16 for typical prefill). MiniCPM5 / CudaFusedMoeRunner
    // default moe_align_block_size is 16 — use that for _dl_C.
    constexpr int64_t block_size_m = 16;
    constexpr int64_t block_size_n = 64;
    constexpr int64_t block_size_k = 64;
    const std::vector<int64_t> block_shape; // empty == vLLM default

    // Stage 1: hidden @ w13 -> [M, topk, 2*inter]  (vLLM fused_experts_impl)
    auto cache1 = at::empty({m, topk, inter2}, hidden.options());
    auto a1 = hidden.view({-1, h});
    auto c1 = cache1.view({-1, topk, inter2});
    adaptor::denglin_fused_moe::invoke_fused_moe_opt(
        a1, w13, c1,
        std::nullopt, std::nullopt, std::nullopt,
        topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        /*mul_routed_weight=*/false,
        topk, block_size_m, block_size_n, block_size_k,
        false, false, false, false,
        block_shape, m);

    // SiLU/GLU activation (same as vLLM apply_moe_activation for gated silu)
    auto flat_gate_up = cache1.reshape({m * topk, inter2});
    auto chunks = flat_gate_up.chunk(2, /*dim=*/-1);
    auto activated = at::silu(chunks[0]) * chunks[1]; // [M*topk, inter]

    // Stage 2: activated @ w2 -> [M, topk, H], top_k=1, mul routed weights
    auto cache3 = at::empty({m, topk, h}, hidden.options());
    auto a2 = activated.contiguous();
    auto c2 = cache3.view({-1, /*top_k=*/1, h});
    adaptor::denglin_fused_moe::invoke_fused_moe_opt(
        a2, w2, c2,
        std::nullopt, std::nullopt, std::nullopt,
        topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
        /*mul_routed_weight=*/true,
        /*top_k=*/1, block_size_m, block_size_n, block_size_k,
        false, false, false, false,
        block_shape, m);

    output.copy_(cache3.sum(/*dim=*/1));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    // Override the ALLDEVICE infiniop registration for Denglin QY.
    MoeFusedDense::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    MoeFusedDense::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeFusedDense::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_fused_dense_impl::qy_dl

#endif
