#include "ernie4_5_moe_vl_decoder_layer.hpp"

#include "infinicore/context/context.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

bool Ernie4_5_VLMoeDecoderLayer::compute_is_moe_layer(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t layer_idx) {
    // moe_layer_start_index = [1, 1], moe_layer_end_index = [29, 28],
    // moe_layer_interval = 1. Index 0 of each array is the text branch which
    // drives the (shared) layer schedule; layers in [start, end) at the given
    // interval are MoE. With 28 layers this makes layer 0 dense and 1..27 MoE.
    auto start = model_config->get_ref("moe_layer_start_index");
    auto interval = model_config->get_or<size_t>("moe_layer_interval", 1);
    size_t start_idx = start[0];
    if (layer_idx < start_idx) {
        return false;
    }
    return ((layer_idx - start_idx) % interval) == 0;
}

Ernie4_5_VLMoeDecoderLayer::Ernie4_5_VLMoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                       size_t layer_idx,
                                                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);

    is_moe_layer_ = compute_is_moe_layer(model_config, layer_idx);
    if (is_moe_layer_) {
        moe_block_ = this->register_module<Ernie4_5_VLMoeSparseMoeBlock>("mlp", model_config, device);
    } else {
        dense_mlp_ = this->register_module<infinilm::layers::mlp::MLP>("mlp", model_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
Ernie4_5_VLMoeDecoderLayer::forward(const infinicore::Tensor &positions,
                                    infinicore::Tensor &hidden_states,
                                    infinicore::Tensor &residual,
                                    const infinicore::Tensor &token_type_ids) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    // Drain the MetaX command queue. Ops are enqueued host-side and only submitted
    // on a sync; a large prefill (a video's 2448 tokens) otherwise piles up enough
    // work to block the enqueue and deadlock (GPU idle, no progress). Small prefills
    // are fast enough that the extra syncs are cheap.
    infinicore::context::syncStream();
    post_attention_layernorm_->forward_inplace(hidden_states, residual);

    if (is_moe_layer_) {
        hidden_states = moe_block_->forward(hidden_states, token_type_ids);
    } else {
        hidden_states = dense_mlp_->forward(hidden_states);
    }
    infinicore::context::syncStream();
    return std::make_tuple(hidden_states, residual);
}

} // namespace infinilm::models::ernie4_5_moe_vl
