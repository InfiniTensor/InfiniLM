#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/causal_lm_templates/text_model.hpp"
#include "../infinilm_model.hpp"
#include "kimi_k25_decoder_layer.hpp"
#include "kimi_k25_vision.hpp"

#include <infinicore/nn/module.hpp>

#include <memory>

namespace infinilm::models::kimi_k25 {

using KimiK25TextModel = infinilm::layers::causal_lm_templates::TextModel<KimiK25DecoderLayer>;
using KimiK25LanguageModel = infinilm::layers::causal_lm_templates::TextCausalLM<KimiK25TextModel>;

class KimiK25ForConditionalGeneration : public infinilm::InfinilmModel {
public:
    KimiK25ForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    void replace_image_embeddings(infinicore::Tensor &inputs_embeds,
                                  const Input &input) const;

    INFINICORE_NN_MODULE(KimiK25VisionTower, vision_tower);
    INFINICORE_NN_MODULE(KimiK25Projector, mm_projector);
    INFINICORE_NN_MODULE(KimiK25LanguageModel, language_model);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k25_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::kimi_k25
