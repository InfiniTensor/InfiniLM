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

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::hc_pre(const infinicore::Tensor &x,
                               const infinicore::Tensor &fn,
                               const infinicore::Tensor &scale,
                               const infinicore::Tensor &base) const {
    DeepseekV4MHCGpuCache &gpu_cache = (&fn == &hc_attn_fn_) ? hc_attn_gpu_ : hc_ffn_gpu_;
    const auto prepared = mhc_prepare(x, base, fn, scale, gpu_cache,
                                      hc_mult_, hidden_size_,
                                      hc_sinkhorn_iters_, hc_eps_);
    if (!prepared.params.post_gpu || !prepared.params.comb_gpu) {
        throw std::runtime_error("DeepseekV4MHC: hc_pre requires GPU post/comb tensors");
    }
    return {prepared.collapsed, prepared.params.post_gpu, prepared.params.comb_gpu};
}

infinicore::Tensor DeepseekV4DecoderLayer::hc_post(const infinicore::Tensor &new_x,
                                                   const infinicore::Tensor &residual,
                                                   const infinicore::Tensor &post,
                                                   const infinicore::Tensor &comb) const {
    return mhc_post_gpu(new_x, residual, post, comb);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
DeepseekV4DecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &positions,
                                const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &post_mix,
                                const infinicore::Tensor &res_mix,
                                const infinicore::Tensor & /*residual*/) const {
    auto [attn_input, attn_post, attn_comb] = hc_pre(hidden_states, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);

    attn_input = attn_norm_->forward(attn_input);
    auto attn_output = attn_->forward(positions, attn_input);
    auto x = hc_post(attn_output, hidden_states, attn_post, attn_comb);

    auto [ffn_input, ffn_post, ffn_comb] = hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);

    ffn_input = ffn_norm_->forward(ffn_input);
    auto ffn_output = ffn_->forward(ffn_input, input_ids);
    x = hc_post(ffn_output, x, ffn_post, ffn_comb);

    return {x, x, post_mix, res_mix};
}

// std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
// DeepseekV4DecoderLayer::forward(const infinicore::Tensor &hidden_states,
//                                 const infinicore::Tensor &positions,
//                                 const infinicore::Tensor &input_ids,
//                                 const infinicore::Tensor &post_mix,
//                                 const infinicore::Tensor &res_mix,
//                                 const infinicore::Tensor & /*residual*/) const {

//     auto attn_input = hidden_states;
//     attn_input = attn_norm_->forward(attn_input);
//     auto attn_output = attn_->forward(positions, attn_input);
//     auto ffn_input = attn_output;
//     ffn_input = ffn_norm_->forward(ffn_input);
//     auto ffn_output = ffn_->forward(ffn_input, input_ids);

//     return {ffn_output, ffn_output, post_mix, res_mix};
// }

} // namespace infinilm::models::deepseek_v4
