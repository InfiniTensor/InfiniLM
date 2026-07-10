#include "ernie4_5_decoder_layer.hpp"

#include "infinicore/ops.hpp"

#include <algorithm>

namespace infinilm::models::ernie4_5_vl {
namespace {

size_t min_json_size(const nlohmann::json &value, size_t fallback) {
    if (value.is_array() && !value.empty()) {
        size_t result = value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::min(result, item.get<size_t>());
        }
        return result;
    }
    return value.is_number_unsigned() ? value.get<size_t>() : fallback;
}

size_t max_json_size(const nlohmann::json &value, size_t fallback) {
    if (value.is_array() && !value.empty()) {
        size_t result = value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::max(result, item.get<size_t>());
        }
        return result;
    }
    return value.is_number_unsigned() ? value.get<size_t>() : fallback;
}

bool is_moe_layer(const nlohmann::json &config, size_t layer_idx) {
    const size_t interval = config.value("moe_layer_interval", 1);
    const size_t start = config.contains("moe_layer_start_index") ? min_json_size(config.at("moe_layer_start_index"), 0) : 0;
    const size_t end = config.contains("moe_layer_end_index") ? max_json_size(config.at("moe_layer_end_index"), config.value("num_hidden_layers", 1) - 1) : config.value("num_hidden_layers", 1) - 1;
    return config.value("use_moe", false) && interval > 0 && ((layer_idx + 1) % interval == 0) && layer_idx >= start && layer_idx <= end;
}

} // namespace

Ernie45DecoderLayer::Ernie45DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);

    use_moe_ = is_moe_layer(model_config->get_config_json(), layer_idx);
    if (use_moe_) {
        mlp_ = this->register_module<Ernie45MoE>("mlp", model_config, device);
        // ERNIE also defines an unregistered mlp_text path for pure text tokens when multimodal
        // experts are enabled. This first implementation routes all tokens through mlp.
    } else {
        mlp_ = this->register_module<infinilm::layers::mlp::MLP>("mlp", model_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Ernie45DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                                infinicore::Tensor &hidden_states,
                                                                                infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);

    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = use_moe_
                      ? std::static_pointer_cast<Ernie45MoE>(mlp_)->forward(hidden_states)
                      : std::static_pointer_cast<infinilm::layers::mlp::MLP>(mlp_)->forward(hidden_states);
    return {hidden_states, residual};
}

infinicore::Tensor Ernie45DecoderLayer::forward(const infinicore::Tensor &positions,
                                                infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = use_moe_
                      ? std::static_pointer_cast<Ernie45MoE>(mlp_)->forward(hidden_states)
                      : std::static_pointer_cast<infinilm::layers::mlp::MLP>(mlp_)->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::ernie4_5_vl
