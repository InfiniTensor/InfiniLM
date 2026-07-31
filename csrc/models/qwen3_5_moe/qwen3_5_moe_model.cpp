#include "qwen3_5_moe_model.hpp"

namespace infinilm::models::qwen3_5_moe {

Qwen35MoeModel::Qwen35MoeModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : Qwen35ModelBase(model_config, device) {
    INFINICORE_NN_MODULE_INIT(language_model, model_config, device);
}

infinicore::Tensor Qwen35MoeModel::forward(const InfinilmModel::Input &input) const {
    if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
        auto inputs_embeds = language_model_->embed_tokens(input.input_ids.value());
        replace_image_embeddings(inputs_embeds, input);
        return language_model_->forward_embeds(inputs_embeds, input.position_ids.value());
    }
    return language_model_->forward(input);
}

} // namespace infinilm::models::qwen3_5_moe
