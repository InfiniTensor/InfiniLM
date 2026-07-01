#include "minicpm_eagle_for_causal_lm.hpp"
#include "../../utils.hpp"
#include "../models_registry.hpp"

#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm_eagle {

MiniCPMEagleModel::MiniCPMEagleModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device)
    : dtype_(model_config->get_dtype()),
      device_(device),
      hidden_size_(model_config->get<size_t>("hidden_size")) {
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype_, device_);
    // Eagle/MTP-specific input projection weights. The draft model fuses the
    // previous draft token embedding with target-model hidden states through
    // input_norm1, input_norm2, and fc before running eagle_layers.
    INFINICORE_NN_MODULE_INIT(input_norm1, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(input_norm2, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(fc, hidden_size_ * 2, hidden_size_, false, dtype_, device_);

    eagle_layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        eagle_layers_.push_back(this->register_module<infinilm::models::minicpm4::MiniCPM4DecoderLayer>("eagle_layers." + std::to_string(i), model_config, i, device_));
    }

    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype_, device_);
}

infinicore::Tensor MiniCPMEagleModel::embed_input_ids(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

infinicore::Tensor MiniCPMEagleModel::forward_with_hidden(const infinicore::Tensor &input_ids,
                                                          const infinicore::Tensor &position_ids,
                                                          const infinicore::Tensor &target_hidden_states) const {
    auto input_embeds = input_norm1_->forward(embed_input_ids(input_ids));
    auto target_hidden = input_norm2_->forward(target_hidden_states);
    auto fused_shape = input_embeds->shape();
    fused_shape.back() = hidden_size_ * 2;
    auto fused_input = infinicore::Tensor::empty(fused_shape, input_embeds->dtype(), input_embeds->device());
    fused_input->narrow({{fused_shape.size() - 1, 0, hidden_size_}})->copy_from(input_embeds);
    fused_input->narrow({{fused_shape.size() - 1, hidden_size_, hidden_size_}})->copy_from(target_hidden);
    auto hidden_states = fc_->forward(fused_input);

    for (const auto &layer : eagle_layers_) {
        hidden_states = layer->forward(position_ids, hidden_states);
    }

    return hidden_states;
}

infinicore::Tensor MiniCPMEagleModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto zero_hidden_states = infinicore::Tensor::zeros({input_ids->shape()[0], input_ids->shape()[1], hidden_size_}, dtype_, device_);
    return forward_with_hidden(input_ids, positions, zero_hidden_states);
}

MiniCPMEagleForCausalLM::MiniCPMEagleForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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

infinilm::InfinilmModel::Output MiniCPMEagleForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    infinicore::Tensor hidden_states;
    if (input.target_hidden_states.has_value()) {
        hidden_states = model_->forward_with_hidden(input.input_ids.value(), input.position_ids.value(), input.target_hidden_states.value());
    } else {
        hidden_states = model_->forward(input);
    }
    auto logits = lm_head_->forward(hidden_states);
    return {logits, hidden_states};
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_eagle_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm_eagle" != model_type && "minicpm" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm_eagle::create_minicpm_eagle_model_config: model_type is not minicpm_eagle");
    }

    auto &json = model_config->get_config_json();
    if (!json.contains("rope_theta")) {
        json["rope_theta"] = 10000.0;
    }
    if (json.contains("bias")) {
        json["attention_bias"] = json["bias"];
        json["mlp_bias"] = json["bias"];
    }
    if (!json.contains("attention_output_bias")) {
        json["attention_output_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::minicpm_eagle

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm_eagle,
    infinilm::models::minicpm_eagle::MiniCPMEagleForCausalLM,
    infinilm::models::minicpm_eagle::create_minicpm_eagle_model_config);
} // namespace
