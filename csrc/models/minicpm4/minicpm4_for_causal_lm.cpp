#include "minicpm4_for_causal_lm.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm4 {

namespace {
float residual_scale(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
    if (model_config->get_or<std::string>("model_type", "") == "minicpm_eagle") {
        const float mup_denominator = model_config->get_or<float>("mup_denominator", 1.0f);
        return scale_depth / std::sqrt(mup_denominator);
    }
    const float num_hidden_layers = static_cast<float>(model_config->get<size_t>("num_hidden_layers"));
    return scale_depth / std::sqrt(num_hidden_layers);
}
} // namespace

MiniCPM4Attention::MiniCPM4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : Attention(model_config, layer_idx, device) {
    o_proj_->set_alpha(residual_scale(model_config));
}

MiniCPM4MLP::MiniCPM4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device)
    : MLP(model_config, device) {
    down_proj_->set_alpha(residual_scale(model_config));
}

MiniCPM4DecoderLayer::MiniCPM4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> MiniCPM4DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                                 infinicore::Tensor &hidden_states,
                                                                                 infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor MiniCPM4DecoderLayer::forward(const infinicore::Tensor &positions,
                                                 infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

MiniCPM4Model::MiniCPM4Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<MiniCPM4DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor MiniCPM4Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);

    infinicore::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual);
    }

    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

infinicore::Tensor MiniCPM4Model::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

MiniCPM4ForCausalLM::MiniCPM4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);

    if (model_config->get_config_json().contains("dim_model_base")) {
        const float dim_model_base = model_config->get<float>("dim_model_base");
        lm_head_->set_alpha(dim_model_base / static_cast<float>(hidden_size));
    }
}

infinilm::InfinilmModel::Output MiniCPM4ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = forward_hidden(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits, hidden_states};
}

infinicore::Tensor MiniCPM4ForCausalLM::forward_hidden(const Input &input) const {
    return model_->forward(input);
}

infinicore::Tensor MiniCPM4ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm4" != model_type && "minicpm" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm4::create_minicpm4_model_config: model_type is not minicpm4");
    }

    auto &json = model_config->get_config_json();
    if (!json.contains("head_dim")) {
        json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
    }
    if (!json.contains("rope_theta")) {
        json["rope_theta"] = 10000.0;
    }
    if (json.contains("bias")) {
        json["attention_bias"] = json["bias"];
        json["mlp_bias"] = json["bias"];
    }
    if (!json.contains("attention_bias")) {
        json["attention_bias"] = false;
    }
    if (!json.contains("mlp_bias")) {
        json["mlp_bias"] = false;
    }
    if (!json.contains("attention_output_bias")) {
        json["attention_output_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::minicpm4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm4,
    infinilm::models::minicpm4::MiniCPM4ForCausalLM,
    infinilm::models::minicpm4::create_minicpm4_model_config);
} // namespace
