#include "gemma4_decoder_layer.hpp"

#include "gemma4_ops.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::models::gemma4 {

Gemma4TextDecoderLayer::Gemma4TextDecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const Gemma4TextSpec &text_spec,
    const Gemma4LayerSpec &layer_spec,
    std::shared_ptr<Gemma4RoPE> rope,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend)
    : layer_idx_(layer_spec.layer_idx),
      hidden_size_(text_spec.hidden_size),
      hidden_size_per_layer_input_(text_spec.hidden_size_per_layer_input),
      enable_moe_block_(text_spec.enable_moe_block) {
    runtime.assertOwnerThread();
    const auto dtype = model_config->get_dtype();
    const double eps = text_spec.rms_norm_eps;

    try {
        self_attn_ = this->bind_child(
            "self_attn",
            std::make_shared<Gemma4TextAttention>(
                runtime, model_config, text_spec, layer_spec, std::move(rope), rank_info, attention_backend));
        mlp_ = this->bind_child(
            "mlp",
            std::make_shared<Gemma4TextMLP>(
                runtime, model_config, hidden_size_, layer_spec.mlp_intermediate_size, rank_info));

        input_layernorm_ = this->bind_child(
            "input_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
        post_attention_layernorm_ = this->bind_child(
            "post_attention_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
        pre_feedforward_layernorm_ = this->bind_child(
            "pre_feedforward_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
        post_feedforward_layernorm_ = this->bind_child(
            "post_feedforward_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));

        layer_scalar_ = infinilm::weights::Weight(runtime, {1}, dtype);
        // Default to 1.0 in case a test checkpoint omits layer_scalar.
        infinilm::execution::submit<infini::ops::Fill>(
            runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
            layer_scalar_.tensor(), 1.0, layer_scalar_.tensor());
        this->bind_weight("layer_scalar", layer_scalar_);

        if (hidden_size_per_layer_input_ > 0) {
            per_layer_input_gate_ = this->bind_child(
                "per_layer_input_gate",
                std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                    runtime, hidden_size_, hidden_size_per_layer_input_, false /* bias */, dtype));
            per_layer_projection_ = this->bind_child(
                "per_layer_projection",
                std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                    runtime, hidden_size_per_layer_input_, hidden_size_, false /* bias */, dtype));
            post_per_layer_input_norm_ = this->bind_child(
                "post_per_layer_input_norm",
                std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
        }

        if (enable_moe_block_) {
            router_ = this->bind_child("router", std::make_shared<Gemma4TextRouter>(runtime, model_config, text_spec));
            experts_ = this->bind_child("experts", std::make_shared<Gemma4TextExperts>(runtime, model_config, text_spec, rank_info));
            post_feedforward_layernorm_1_ = this->bind_child(
                "post_feedforward_layernorm_1", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
            pre_feedforward_layernorm_2_ = this->bind_child(
                "pre_feedforward_layernorm_2", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
            post_feedforward_layernorm_2_ = this->bind_child(
                "post_feedforward_layernorm_2", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, eps, dtype));
        }
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

infinilm::runtime::Tensor Gemma4TextDecoderLayer::forward(
    const infinilm::runtime::Tensor &positions,
    const infinilm::runtime::Tensor &hidden_states_in,
    const infinilm::runtime::Tensor &per_layer_input,
    global_state::ForwardContext &forward_context,
    const Gemma4VisionSpans *vision_spans) const {
    auto &runtime = forward_context.require_runtime();

    // 1. Self-Attention block
    auto residual = hidden_states_in;
    auto h = input_layernorm_->forward(runtime, hidden_states_in);
    h = self_attn_->forward(positions, h, forward_context, vision_spans);
    h = post_attention_layernorm_->forward(runtime, h);
    h = ops::add(runtime, residual, h);

    // 2. Feed-Forward (+ optional parallel MoE) block
    residual = h;
    auto mlp_in = pre_feedforward_layernorm_->forward(runtime, h);
    auto mlp_out = mlp_->forward(runtime, mlp_in);

    if (enable_moe_block_) {
        auto h1 = post_feedforward_layernorm_1_->forward(runtime, mlp_out);

        const auto shape = residual->shape();
        auto res_flat = residual->contiguous(runtime)->view({shape[0] * shape[1], shape[2]});
        auto [top_k_weights, top_k_index] = router_->forward(runtime, res_flat);
        auto moe_in = pre_feedforward_layernorm_2_->forward(runtime, res_flat);
        auto moe_out = experts_->forward(runtime, moe_in, top_k_index, top_k_weights);
        auto h2 = post_feedforward_layernorm_2_->forward(runtime, moe_out->view(shape));

        mlp_out = ops::add(runtime, h1, h2);
    }

    h = post_feedforward_layernorm_->forward(runtime, mlp_out);
    h = ops::add(runtime, residual, h);

    // 3. Per-Layer Embedding (PLE) residual gating block (E2B / E4B)
    if (hidden_size_per_layer_input_ > 0 && per_layer_input) {
        residual = h;
        auto gate = per_layer_input_gate_->forward(runtime, h);
        gate = ops::gelu_tanh(runtime, gate);
        gate = ops::mul(runtime, gate, per_layer_input);
        auto proj = per_layer_projection_->forward(runtime, gate);
        proj = post_per_layer_input_norm_->forward(runtime, proj);
        h = ops::add(runtime, residual, proj);
    }

    // 4. Layer scalar
    ops::mul_inplace(runtime, h, layer_scalar_.tensor());
    return h;
}

} // namespace infinilm::models::gemma4
