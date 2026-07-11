#include "ernie4_5_moe_vl_text_model.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5_VLMoeModel::Ernie4_5_VLMoeModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    embed_tokens_ = this->register_module<infinicore::nn::Embedding>(
        "embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);

    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Ernie4_5_VLMoeDecoderLayer>(
            "layers." + std::to_string(i), model_config, i, device));
    }

    norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor Ernie4_5_VLMoeModel::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

infinicore::Tensor Ernie4_5_VLMoeModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);

    // Text-only path: all tokens are text modality (token_type 0). Build the zero
    // tensor explicitly on CPU then move to device -- Tensor::zeros(I64) on the MetaX
    // backend does not reliably clear device memory, which left token_type garbage and
    // routed every token through the vision experts/gate.
    std::vector<int64_t> tt_zero(input_ids->numel(), 0);
    auto token_type_ids = infinicore::Tensor::from_blob(
                              tt_zero.data(), input_ids->shape(),
                              infinicore::DataType::I64, infinicore::Device::cpu())
                              ->to(hidden_states->device());

    infinicore::Tensor residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_.at(i)->forward(positions, hidden_states, residual, token_type_ids);
        // Drain the command queue each layer. A large prefill (e.g. a video's 2448
        // tokens through the MoE) otherwise piles up enough async work to deadlock
        // the MetaX command queue; a per-layer sync keeps it bounded.
        infinicore::context::syncStream();
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

infinicore::Tensor Ernie4_5_VLMoeModel::forward_embeds(const infinicore::Tensor &inputs_embeds,
                                                       const infinicore::Tensor &position_ids,
                                                       const infinicore::Tensor &token_type_ids) const {
    auto hidden_states = inputs_embeds;
    infinicore::Tensor residual;
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_.at(i)->forward(position_ids, hidden_states, residual, token_type_ids);
        // Per-layer command-queue drain (see forward()): a video's large prefill
        // deadlocks the MetaX queue without it.
        infinicore::context::syncStream();
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

} // namespace infinilm::models::ernie4_5_moe_vl
