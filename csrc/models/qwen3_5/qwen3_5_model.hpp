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

class Qwen35Model : public Qwen35ModelBase {
public:
    Qwen35Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                const infinicore::Device &device);

    infinicore::Tensor forward(const InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(Qwen35LanguageModel, language_model);
};

} // namespace infinilm::models::qwen3_5
