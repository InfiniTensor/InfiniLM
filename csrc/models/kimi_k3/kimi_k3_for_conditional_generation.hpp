#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "kimi_k3_text_model.hpp"
#include "kimi_k3_vision.hpp"

#include <memory>

namespace infinilm::models::kimi_k3 {

using KimiK3LanguageModel = infinilm::layers::causal_lm_templates::TextCausalLM<KimiK3TextModel>;

class KimiK3ForConditionalGeneration : public infinilm::InfinilmModel {
public:
    KimiK3ForConditionalGeneration(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    void replace_image_embeddings(infinicore::Tensor &inputs_embeds,
                                  const Input &input) const;

    INFINICORE_NN_MODULE(KimiK3VisionTower, vision_tower);
    INFINICORE_NN_MODULE(KimiK3Projector, mm_projector);
    INFINICORE_NN_MODULE(KimiK3LanguageModel, language_model);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::kimi_k3
