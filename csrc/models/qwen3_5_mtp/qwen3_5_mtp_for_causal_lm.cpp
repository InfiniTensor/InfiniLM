#include "qwen3_5_mtp_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5_mtp {

Qwen35MtpModel::Qwen35MtpModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device)
    : dtype_(model_config->get_dtype()),
      device_(device),
      hidden_size_(model_config->get<size_t>("hidden_size")) {
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    // The draft shares embed_tokens/lm_head with the target model; the loader
    // copies the target embedding into this module when loading mtp.* weights.
    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype_, device_);
    // MTP-specific input fusion: normalize the next-token embedding and the
    // target hidden state separately, concatenate, and project back to
    // hidden_size before running the draft layer.
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_embedding, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(pre_fc_norm_hidden, hidden_size_, rms_norm_eps, dtype_, device_);
    INFINICORE_NN_MODULE_INIT(fc, hidden_size_ * 2, hidden_size_, false, dtype_, device_);

    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<infinilm::models::qwen3_5::Qwen35DecoderLayer>("layers." + std::to_string(i), model_config, i, device_));
    }

    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, rms_norm_eps, dtype_, device_);
}

infinicore::Tensor Qwen35MtpModel::embed_input_ids(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

infinicore::Tensor Qwen35MtpModel::forward_with_hidden(const infinicore::Tensor &input_ids,
                                                       const infinicore::Tensor &position_ids,
                                                       const infinicore::Tensor &target_hidden_states) const {
    auto input_embeds = pre_fc_norm_embedding_->forward(embed_input_ids(input_ids));
    auto target_hidden = pre_fc_norm_hidden_->forward(target_hidden_states);
    auto fused_shape = input_embeds->shape();
    fused_shape.back() = hidden_size_ * 2;
    auto fused_input = infinicore::Tensor::empty(fused_shape, input_embeds->dtype(), input_embeds->device());
    fused_input->narrow({{fused_shape.size() - 1, 0, hidden_size_}})->copy_from(input_embeds);
    fused_input->narrow({{fused_shape.size() - 1, hidden_size_, hidden_size_}})->copy_from(target_hidden);
    auto hidden_states = fc_->forward(fused_input);

    for (const auto &layer : layers_) {
        hidden_states = layer->forward(position_ids, hidden_states);
    }

    // The checkpoint carries mtp.norm: like the target model, the draft hidden
    // state is normalized before the (tied) output head and fed back as the
    // previous hidden state of the next MTP step.
    return norm_->forward(hidden_states);
}

infinicore::Tensor Qwen35MtpModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto zero_hidden_states = infinicore::Tensor::zeros({input_ids->shape()[0], input_ids->shape()[1], hidden_size_}, dtype_, device_);
    return forward_with_hidden(input_ids, positions, zero_hidden_states);
}

Qwen35MtpForCausalLM::Qwen35MtpForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen35MtpForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    infinicore::Tensor hidden_states;
    if (input.target_hidden_states.has_value()) {
        hidden_states = model_->forward_with_hidden(input.input_ids.value(), input.position_ids.value(), input.target_hidden_states.value());
    } else {
        hidden_states = model_->forward(input);
    }
    auto logits = lm_head_->forward(hidden_states);
    return {logits, hidden_states};
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_mtp_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5_mtp" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5_mtp::create_qwen3_5_mtp_model_config: model_type is not qwen3_5_mtp");
    }

    qwen3_5::prepare_qwen3_5_model_config(model_config);
    auto &json = model_config->get_config_json();
    // The draft is a standalone single-layer model whose layer is always full
    // attention, regardless of the target model's layer type schedule.
    size_t num_hidden_layers = 1;
    if (json.contains("mtp_num_hidden_layers") && json["mtp_num_hidden_layers"].is_number_unsigned()) {
        num_hidden_layers = json["mtp_num_hidden_layers"].get<size_t>();
    }
    json["num_hidden_layers"] = num_hidden_layers;
    json["layer_types"] = std::vector<std::string>(num_hidden_layers, "full_attention");

    return model_config;
}

} // namespace infinilm::models::qwen3_5_mtp

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5_mtp,
    infinilm::models::qwen3_5_mtp::Qwen35MtpForCausalLM,
    infinilm::models::qwen3_5_mtp::create_qwen3_5_mtp_model_config);
} // namespace
