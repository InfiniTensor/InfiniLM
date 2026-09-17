#include "gemma3_decoder_layer.hpp"

namespace infinilm::models::gemma3 {

Gemma3DecoderLayer::Gemma3DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : layer_idx_(layer_idx),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
    post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);
    pre_feedforward_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("pre_feedforward_layernorm", hidden_size, rms_norm_eps, dtype, device);
    post_feedforward_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_feedforward_layernorm", hidden_size, rms_norm_eps, dtype, device);
    self_attn_ = this->register_module<Gemma3Attention>("self_attn", model_config, layer_idx, device);
    mlp_ = this->register_module<Gemma3MLP>("mlp", model_config, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Gemma3DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                               infinicore::Tensor &hidden_states,
                                                                               infinicore::Tensor &residual) {
    // 1. Normalize the mainstream (fused: residual += incoming branch, hidden = norm(residual)).
    input_layernorm_->forward_inplace(hidden_states, residual);

    // 2. Attention on the normalized branch.
    hidden_states = self_attn_->forward(positions, hidden_states);

    // 3. Gemma order: normalize the branch FIRST, then add it to the residual stream.
    hidden_states = post_attention_layernorm_->forward(hidden_states);

    // 4. Fuse the branch addition with the pre-feedforward norm:
    //    add_rms_norm(residual, branch, w) returns (norm(residual+branch, w),
    //    residual+branch), replacing a separate add + norm pair.
    auto fused = infinicore::op::add_rms_norm(residual, hidden_states,
                                              pre_feedforward_layernorm_->weight(),
                                              static_cast<float>(rms_norm_eps_));
    residual = std::move(fused.second);
    hidden_states = std::move(fused.first);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = post_feedforward_layernorm_->forward(hidden_states);

    // 5. Contract: leave the branch un-added; the consumer (next layer's input
    //    norm or the model's final norm) performs `residual + hidden`.
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Gemma3DecoderLayer::forward(const infinicore::Tensor &positions,
                                               infinicore::Tensor &hidden_states) {
    // Naive (debug) path mirroring the HF reference exactly.
    infinicore::Tensor residual = hidden_states;

    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = pre_feedforward_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = post_feedforward_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::gemma3
