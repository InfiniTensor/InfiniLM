#include "qwen3_vl_text_model.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops.hpp>

#include <optional>
#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_vl {

Qwen3VLTextModel::Qwen3VLTextModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : skip_final_norm_(model_config->get_or<bool>(
        "condition_encoder_skip_final_norm", false)) {
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.pp_size != 1) {
        throw std::runtime_error("Qwen3VLTextModel: pipeline parallel is not yet "
                                 "supported with DeepStack");
    }

    const auto &dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt,
                              dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Qwen3VLDecoderLayer>(
            "layers." + std::to_string(i), model_config, i, device));
    }
    if (!skip_final_norm_) {
        INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
    }
}

infinicore::Tensor
Qwen3VLTextModel::forward(const infinilm::InfinilmModel::Input &input) const {
    return run_layers(embed_tokens_->forward(input.input_ids.value()),
                      input.position_ids.value(), {});
}

infinicore::Tensor Qwen3VLTextModel::forward_embeds(
    const infinicore::Tensor &inputs_embeds,
    const infinicore::Tensor &position_ids,
    const std::vector<Qwen3VLDeepStackEmbedding> &deepstack_embeddings) const {
    return run_layers(inputs_embeds, position_ids, deepstack_embeddings);
}

infinicore::Tensor
Qwen3VLTextModel::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

infinicore::Tensor Qwen3VLTextModel::run_layers(
    infinicore::Tensor hidden_states, const infinicore::Tensor &position_ids,
    const std::vector<Qwen3VLDeepStackEmbedding> &deepstack_embeddings) const {
    infinicore::Tensor residual;
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        layers_[layer_idx]->forward(position_ids, hidden_states, residual);

        bool materialized = false;
        for (const auto &deepstack : deepstack_embeddings) {
            if (deepstack.decoder_layer != layer_idx) {
                continue;
            }
            if (!materialized && residual) {
                hidden_states = infinicore::op::add(residual, hidden_states);
                residual = {};
                materialized = true;
            }
            const size_t token_count = deepstack.features->size(0);
            if (deepstack.token_offset + token_count > hidden_states->size(1)) {
                throw std::runtime_error(
                    "Qwen3VLTextModel: DeepStack token range is out of bounds");
            }
            auto destination = hidden_states->narrow({{1, deepstack.token_offset, token_count}});
            infinicore::op::add_(destination, destination,
                                 deepstack.features->unsqueeze(0));
        }
    }

    if (skip_final_norm_) {
        return residual ? infinicore::op::add(residual, hidden_states)
                        : hidden_states;
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

} // namespace infinilm::models::qwen3_vl
