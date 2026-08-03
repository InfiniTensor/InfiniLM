#pragma once

#include "../../layers/common_modules.hpp"
#include "../infinilm_model.hpp"
#include "infinicore/tensor.hpp"
#include "qwen3_5_decoderLayer.hpp"
#include "qwen3_5_vision.hpp"

namespace infinilm::models::qwen3_5 {

using Qwen35LanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35DecoderLayer>;

class Qwen35ModelBase : public infinicore::nn::Module {
public:
    Qwen35ModelBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config);

protected:
    void replace_image_embeddings(infinicore::Tensor &inputs_embeds,
                                  const infinilm::InfinilmModel::Input &input) const;

    INFINICORE_NN_MODULE(Qwen35VisionModel, visual);
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

template <typename LanguageModel>
class Qwen35ModelTemplate : public Qwen35ModelBase {
public:
    Qwen35ModelTemplate(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device)
        : Qwen35ModelBase(model_config, device) {
        language_model_ = this->register_module<LanguageModel>(
            "language_model", model_config, device);
    }

    infinicore::Tensor forward(const InfinilmModel::Input &input) const {
        if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
            auto inputs_embeds = language_model_->embed_tokens(input.input_ids.value());
            replace_image_embeddings(inputs_embeds, input);
            return language_model_->forward_embeds(
                inputs_embeds, input.position_ids.value());
        }
        return language_model_->forward(input);
    }

protected:
    INFINICORE_NN_MODULE(LanguageModel, language_model);
};

using Qwen35Model = Qwen35ModelTemplate<Qwen35LanguageModel>;

} // namespace infinilm::models::qwen3_5
