#include "llama_mlp.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaMLP::LlamaMLP(const LlamaConfig &config, const infinicore::Device &device)
    : hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size),
      use_bias_(config.mlp_bias) {
    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size_, intermediate_size_, use_bias_,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(up_proj, hidden_size_, intermediate_size_, use_bias_,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, use_bias_,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor LlamaMLP::forward(const infinicore::Tensor &hidden_states,
                                      const HookRegistry *hook_registry,
                                      const std::string &hook_prefix,
                                      int layer_idx) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(hidden_states_mutable);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_gate_proj", gate, layer_idx);
    }

    auto up = up_proj_->forward(hidden_states_mutable);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_up_proj", up, layer_idx);
    }

    // 2. Apply SwiGLU: silu(gate) * up
    // Note: swiglu kernel expects (up, gate) and computes gate * sigmoid(gate) * up
    // So we pass (up, gate) to get the correct result: gate * sigmoid(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_intermediate", intermediate, layer_idx);
    }

    // 3. Project down
    auto output = down_proj_->forward(intermediate);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_output", output, layer_idx);
    }

    return output;
}

} // namespace infinilm::models::llama
