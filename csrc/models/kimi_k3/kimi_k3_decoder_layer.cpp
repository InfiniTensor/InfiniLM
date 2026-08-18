#include "kimi_k3_decoder_layer.hpp"

#include <algorithm>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/softmax.hpp>
#include <stdexcept>
#include <vector>

namespace infinilm::models::kimi_k3 {
KimiK3DecoderLayer::KimiK3DecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      attn_res_block_size_(model_config->get<size_t>("attn_res_block_size")) {
    const auto &dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");
    const auto kda_layers = model_config->get_config_json()
                                .at("linear_attn_config")
                                .at("kda_layers")
                                .get<std::vector<size_t>>();
    is_kda_ = std::find(kda_layers.begin(), kda_layers.end(), layer_idx + 1) != kda_layers.end();
    if (is_kda_) {
        delta_attn_ = this->register_module<KimiK3DeltaAttention>(
            "self_attn", model_config, layer_idx, device);
    } else {
        mla_attn_ = this->register_module<KimiK3MLAAttention>(
            "self_attn", model_config, layer_idx, device);
    }

    const size_t first_dense_layer = model_config->get<size_t>("first_k_dense_replace");
    const size_t moe_frequency = model_config->get<size_t>("moe_layer_freq");
    use_moe_ = layer_idx >= first_dense_layer && layer_idx % moe_frequency == 0;
    if (use_moe_) {
        INFINICORE_NN_MODULE_INIT(block_sparse_moe, model_config, layer_idx, device);
    } else {
        INFINICORE_NN_MODULE_INIT(mlp, model_config,
                                  model_config->get<size_t>("intermediate_size"), device);
    }

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attention_res_norm, hidden_size_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp_res_norm, hidden_size_, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attention_res_proj,
                              hidden_size_, 1, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp_res_proj, hidden_size_, 1, false, dtype, device);
}

infinicore::Tensor KimiK3DecoderLayer::apply_attn_res(
    const infinicore::Tensor &prefix_sum,
    const infinicore::Tensor &block_residual_storage,
    size_t block_residual_count,
    const std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> &proj,
    const std::shared_ptr<infinicore::nn::RMSNorm> &norm) const {
    const auto shape = prefix_sum->shape();
    if (block_residual_count >= block_residual_storage->size(1)) {
        throw std::runtime_error("KimiK3DecoderLayer: block residual scratch slot is unavailable");
    }
    auto values = block_residual_storage->narrow(
        {{1, 0, block_residual_count + 1}});
    auto normalized = norm->forward(values);
    auto scores = proj->forward(normalized);
    infinicore::op::softmax_(scores, scores, 1);
    auto score_matrix = scores->view(
        {scores->size(0), scores->size(2), scores->size(1)});
    return infinicore::op::matmul(score_matrix, values)
        ->squeeze(1)
        ->view(shape);
}

infinicore::Tensor KimiK3DecoderLayer::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &block_residual_storage,
    size_t &block_residual_count) const {
    const auto shape = hidden_states->shape();
    auto prefix_sum = hidden_states;
    auto current = hidden_states;

    if (block_residual_count > 0) {
        current = apply_attn_res(prefix_sum, block_residual_storage,
                                 block_residual_count,
                                 self_attention_res_proj_, self_attention_res_norm_);
    }
    bool reset_prefix = layer_idx_ % attn_res_block_size_ == 0;
    if (reset_prefix) {
        if (block_residual_count + 1 >= block_residual_storage->size(1)) {
            throw std::runtime_error("KimiK3DecoderLayer: block residual storage is full");
        }
        // The current scratch slot becomes a persistent block residual. The
        // attention result will be written into the next scratch slot below.
        ++block_residual_count;
    }

    current = input_layernorm_->forward(current);
    current = is_kda_ ? delta_attn_->forward(current) : mla_attn_->forward(current);
    if (reset_prefix) {
        auto scratch = block_residual_storage
                           ->narrow({{1, block_residual_count, 1}})
                           ->view({shape[0], shape[1], hidden_size_});
        scratch->copy_from(current);
        prefix_sum = scratch;
    } else {
        infinicore::op::add_(prefix_sum, prefix_sum, current);
    }

    current = apply_attn_res(prefix_sum, block_residual_storage,
                             block_residual_count, mlp_res_proj_, mlp_res_norm_);
    current = post_attention_layernorm_->forward(current);
    current = use_moe_ ? block_sparse_moe_->forward(current) : mlp_->forward(current);
    infinicore::op::add_(prefix_sum, prefix_sum, current);
    return prefix_sum;
}

} // namespace infinilm::models::kimi_k3
