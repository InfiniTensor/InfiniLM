#include "deepseek_v4_model.hpp"

#include "deepseek_v4_utils.hpp"
#include <optional>
#include <tuple>

namespace infinilm::models::deepseek_v4 {
DeepseekV4Model::DeepseekV4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      vocab_size_(model_config->get<size_t>("vocab_size")),
      hc_mult_(model_config->get<size_t>("hc_mult")),
      hc_eps_(model_config->get<double>("hc_eps")) {
    const auto &dtype = model_config->get_dtype();
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed, vocab_size_, hidden_size_, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<DeepseekV4DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({hc_mult_}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({hc_mult_, hidden_size_ * hc_mult_}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({static_cast<size_t>(1)}, infinicore::DataType::F32, device));
}

infinicore::Tensor DeepseekV4Model::forward(const infinicore::Tensor &input_ids,
                                            const infinicore::Tensor &positions) const {
    auto hidden_states = embed_->forward(input_ids);
    hidden_states = expand_hc_stream(hidden_states, hc_mult_);

    infinicore::Tensor residual;
    infinicore::Tensor post_mix;
    infinicore::Tensor res_mix;
    for (const auto &layer : layers_) {
        auto layer_output = layer->forward(hidden_states, positions, input_ids, post_mix, res_mix, residual);
        hidden_states = std::get<0>(layer_output);
        residual = std::get<1>(layer_output);
        post_mix = std::get<2>(layer_output);
        res_mix = std::get<3>(layer_output);
    }

    hidden_states = mhc_head_pre(hidden_states, hc_head_base_, hc_head_fn_, hc_head_scale_,
                                 hc_mult_, hidden_size_, hc_eps_);
    return norm_->forward(hidden_states);
}

} // namespace infinilm::models::deepseek_v4
