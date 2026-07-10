#include "ernie4_5_model.hpp"

#include "../../global_state/global_state.hpp"

#include <optional>
#include <string>
#include <utility>

namespace infinilm::models::ernie4_5_vl {

Ernie45Model::Ernie45Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device)
    : model_config_(model_config) {
    const auto &dtype = model_config->get_dtype();
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
        INFINICORE_NN_MODULE_INIT(resampler_model, config_json, dtype, device);
    }
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Ernie45DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor Ernie45Model::forward(const InfinilmModel::Input &input) const {
    // Text-only path. Full VL forward must detect im_patch_id tokens, run vision_model,
    // run resampler_model, and scatter image embeddings into language inputs.
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);
    infinicore::Tensor residual;
    for (auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

void Ernie45Model::reset_cache(const cache::CacheConfig *) {
    // Cache tensors are allocated by Ernie45ForConditionalGeneration, which inherits InfinilmModel.
}

} // namespace infinilm::models::ernie4_5_vl





