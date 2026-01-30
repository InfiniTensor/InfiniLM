#include "llama_mlp.hpp"
#include "../../fusion/fusion_context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaMLP::LlamaMLP(const LlamaConfig &config,
                   const infinicore::Device &device,
                   engine::distributed::RankInfo rank_info)
    : hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size),
      use_bias_(config.mlp_bias),
      enable_fusion_(config.enable_fusion),
      rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    // Initialize projection layers
    INFINILM_GATE_UP_LINEAR_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, use_bias_,
                                 dtype, device, rank_info_);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, use_bias_,
                              dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor LlamaMLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    // Check both static config and dynamic FusionContext
    bool use_fused_swiglu = enable_fusion_ && fusion::FusionContext::get("swiglu", true);

    infinicore::Tensor intermediate;
    if (use_fused_swiglu) {
        // Fused SwiGLU: swiglu kernel computes silu(gate) * up
        intermediate = infinicore::op::swiglu(up, gate);
    } else {
        // Non-fused path: separate silu and mul
        auto activated = infinicore::op::silu(gate);
        intermediate = infinicore::op::mul(activated, up);
    }

    // 3. Project down
    auto output = down_proj_->forward(intermediate);

    return output;
}

} // namespace infinilm::models::llama
