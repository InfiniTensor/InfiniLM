#include "kimi_k3_decoder_layer.hpp"

#include <algorithm>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/cat.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/softmax.hpp>
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
    const infinicore::Tensor &block_residual,
    const std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> &proj,
    const std::shared_ptr<infinicore::nn::RMSNorm> &norm) const {
    const auto shape = prefix_sum->shape();
    auto prefix_2d = prefix_sum->view({shape[0] * shape[1], hidden_size_});
    auto values = infinicore::op::cat({block_residual, prefix_2d->unsqueeze(1)}, 1);
    auto normalized = norm->forward(values);
    auto scores = proj->forward(normalized);
    infinicore::op::softmax_(scores, scores, 1);
    auto score_matrix = scores->permute({0, 2, 1})->contiguous();
    return infinicore::op::matmul(score_matrix, values)
        ->squeeze(1)
        ->view(shape);
}

std::pair<infinicore::Tensor, infinicore::Tensor>
KimiK3DecoderLayer::forward(const infinicore::Tensor &hidden_states,
                            const infinicore::Tensor &block_residual) const {
    const auto shape = hidden_states->shape();
    auto prefix_sum = hidden_states;
    auto current = hidden_states;
    auto residuals = block_residual;

    if (residuals->size(1) > 0) {
        current = apply_attn_res(prefix_sum, residuals,
                                 self_attention_res_proj_, self_attention_res_norm_);
    }
    bool reset_prefix = layer_idx_ % attn_res_block_size_ == 0;
    if (reset_prefix) {
        auto prefix_2d = prefix_sum->view({shape[0] * shape[1], hidden_size_});
        residuals = infinicore::op::cat({residuals, prefix_2d->unsqueeze(1)}, 1);
    }

    current = input_layernorm_->forward(current);
    current = is_kda_ ? delta_attn_->forward(current) : mla_attn_->forward(current);
    prefix_sum = reset_prefix ? current : infinicore::op::add(prefix_sum, current);

    current = apply_attn_res(prefix_sum, residuals, mlp_res_proj_, mlp_res_norm_);
    current = post_attention_layernorm_->forward(current);
    current = use_moe_ ? block_sparse_moe_->forward(current) : mlp_->forward(current);
    prefix_sum = infinicore::op::add(prefix_sum, current);
    return {prefix_sum, residuals};
}

} // namespace infinilm::models::kimi_k3
