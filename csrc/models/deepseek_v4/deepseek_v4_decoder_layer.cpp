#include "deepseek_v4_decoder_layer.hpp"

#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"

#include <utility>

namespace infinilm::models::deepseek_v4 {

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device)
    : DeepseekV4DecoderLayer(std::move(model_config), 0, device) {
}

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      hc_mult_(model_config->get<size_t>("hc_mult")),
      hc_sinkhorn_iters_(model_config->get<size_t>("hc_sinkhorn_iters")),
      hc_eps_(model_config->get<double>("hc_eps")) {
    const auto &dtype = model_config->get_dtype();
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const size_t mix_hc = (2 + hc_mult_) * hc_mult_;
    const size_t hc_dim = hc_mult_ * hidden_size_;

    INFINICORE_NN_MODULE_INIT(attn, model_config, layer_idx_, device);
    INFINICORE_NN_MODULE_INIT(ffn, model_config, layer_idx_, device);
    INFINICORE_NN_MODULE_INIT(attn_norm, hidden_size_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ffn_norm, hidden_size_, rms_norm_eps, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(hc_attn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_scale, ({static_cast<size_t>(3)}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_scale, ({static_cast<size_t>(3)}, infinicore::DataType::F32, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &positions,
                                const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &post_mix,
                                const infinicore::Tensor &res_mix,
                                const infinicore::Tensor & /*residual*/) const {
    const auto attn_mhc = build_mhc_params(hidden_states, hc_attn_base_, hc_attn_fn_, hc_attn_scale_,
                                           hc_mult_, hidden_size_, hc_sinkhorn_iters_, hc_eps_);

    auto attn_input = mhc_pre(hidden_states, attn_mhc);
    attn_input = attn_norm_->forward(attn_input);
    auto attn_output = attn_->forward(positions, attn_input);
    auto x = mhc_post(attn_output, hidden_states, attn_mhc);

    const auto ffn_mhc = build_mhc_params(x, hc_ffn_base_, hc_ffn_fn_, hc_ffn_scale_,
                                          hc_mult_, hidden_size_, hc_sinkhorn_iters_, hc_eps_);

    auto ffn_input = mhc_pre(x, ffn_mhc);
    ffn_input = ffn_norm_->forward(ffn_input);
    auto ffn_output = ffn_->forward(ffn_input, input_ids);
    x = mhc_post(ffn_output, x, ffn_mhc);

    return {x, x, post_mix, res_mix};
}

} // namespace infinilm::models::deepseek_v4
